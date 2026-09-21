"""Offline SFT using the copied PI05 policy and LeRobot optimizer presets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from .preparation import ForceNormalizer, digest, load_contract, local_path, output_path, prepare, write_json
from .train_pi05_force import build_force_policy, check_window_contract, load_force_config
from .window_dataset import ForceVLAWindowDataset, attach_visual_language_inputs, collate_force_windows

TRAINABLE_PREFIXES = ("torque_adapter.", "state_proj.", "state_torque_fusion.", "flow_in_proj.",
                      "flow_action_out_proj.", "torque_out_proj.")


def configure_trainable(policy, mode):
    if mode not in ("adapters", "expert"):
        raise ValueError("trainable must be adapters or expert")
    policy.requires_grad_(False)
    for name, parameter in policy.model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES) or (mode == "expert" and
                name.startswith(("paligemma_with_expert.gemma_expert.model.", "time_mlp_"))):
            parameter.requires_grad_(True)
    return {name: parameter for name, parameter in policy.named_parameters() if parameter.requires_grad}


def prepare_batch(dataset, indices, normalizer, tokenizer, device):
    batch = collate_force_windows([dataset[int(index)] for index in indices])
    batch = attach_visual_language_inputs(normalizer(batch), tokenizer)
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def save_checkpoint(destination, policy, optimizer, scheduler, progress, run_info):
    destination = output_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    parameters = {name: value.detach().cpu().contiguous() for name, value in policy.named_parameters()
                  if value.requires_grad}
    save_file(parameters, str(destination / "adapter.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "progress": progress, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               destination / "training.pt")
    write_json(destination / "run.json", run_info)
    write_json(destination / "force_config.json", run_info["settings"])
    write_json(destination / "complete.json", {"step": progress["step"], "kind": "base_plus_trainable_delta"})


def load_delta(policy, checkpoint):
    checkpoint = local_path(checkpoint)
    if not (checkpoint / "complete.json").is_file():
        raise ValueError("Incomplete checkpoint")
    weights = load_file(str(checkpoint / "adapter.safetensors"))
    expected = {name for name, value in policy.named_parameters() if value.requires_grad}
    if set(weights) != expected:
        raise ValueError("Checkpoint trainable parameter set differs")
    policy.load_state_dict(weights, strict=False)


def restore_training_state(policy, optimizer, scheduler, checkpoint):
    checkpoint = local_path(checkpoint)
    load_delta(policy, checkpoint)
    state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]


def restore_for_inference(checkpoint, device="cuda"):
    checkpoint = local_path(checkpoint)
    run_info = json.loads((checkpoint / "run.json").read_text())
    if digest(local_path(run_info["base_checkpoint"]) / "model.safetensors") != run_info["base_sha256"]:
        raise ValueError("Base checkpoint changed")
    config_path = checkpoint / "force_config.json"
    if not config_path.is_file():
        raise ValueError("Missing saved force configuration")
    policy, _ = build_force_policy(load_force_config(config_path), local_path(run_info["base_checkpoint"]))
    configure_trainable(policy, run_info["settings"]["training"]["trainable"])
    load_delta(policy, checkpoint)
    policy.to(device).eval()
    return policy, ForceNormalizer(run_info["data_contract"]["statistics"])


def evaluate(policy, dataset, indices, normalizer, tokenizer, device, count):
    policy.eval()
    metrics = []
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(12345)
        selected = indices[np.linspace(0, len(indices) - 1, min(count, len(indices)), dtype=int)]
        for index in selected:
            batch = prepare_batch(dataset, [index], normalizer, tokenizer, device)
            loss, output = policy(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            metrics.append({key: output[key] for key in ("loss", "action_loss", "torque_loss")})
    policy.train()
    return {key: float(np.mean([entry[key] for entry in metrics])) for key in metrics[0]}


def run_training(args, dataset):
    settings = json.loads(local_path(args.config).read_text())
    options = settings["training"]
    total_steps = args.steps if args.steps is not None else (2 if args.smoke else options["steps"])
    if total_steps < 1 or options["batch_size"] < 1:
        raise ValueError("steps and batch_size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; invoke scripts/run.sh --gpu")
    contract = load_contract(dataset, args.contract)
    episodes = np.asarray(dataset.table["episode_index"])
    train_rows = np.flatnonzero(np.isin(episodes, contract["train_episodes"]))
    validation_rows = np.flatnonzero(np.isin(episodes, contract["validation_episodes"]))
    destination = output_path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(options["seed"])
    normalizer = ForceNormalizer(contract["statistics"])
    tokenizer = AutoTokenizer.from_pretrained(str(local_path(settings["tokenizer"])), local_files_only=True)
    print("Loading local PI05 backbone...", flush=True)
    policy, loading = build_force_policy(load_force_config(args.config), local_path(settings["base_checkpoint"]))
    parameters = configure_trainable(policy, options["trainable"])
    policy.to(args.device)
    optimizer = policy.config.get_optimizer_preset().build(parameters.values())
    scheduler = policy.config.get_scheduler_preset().build(optimizer, num_training_steps=total_steps)
    run_info = {"kind": "real_pi05_smoke" if args.smoke else "force_pi05_sft", "settings": settings,
                "base_checkpoint": settings["base_checkpoint"],
                "base_sha256": digest(local_path(settings["base_checkpoint"]) / "model.safetensors"),
                "data_contract": contract, "loading": loading, "total_steps": total_steps,
                "trainable_parameters": sum(value.numel() for value in parameters.values())}
    write_json(destination / "run.json", run_info)
    progress = {"step": 0, "epoch": 0, "cursor": 0}
    if args.resume:
        checkpoint = local_path(args.resume)
        previous = json.loads((checkpoint / "run.json").read_text())
        for key in ("settings", "base_sha256", "data_contract", "total_steps"):
            if previous[key] != run_info[key]:
                raise ValueError(f"Resume contract differs: {key}")
        progress = restore_training_state(policy, optimizer, scheduler, checkpoint)
        if progress["step"] >= total_steps:
            raise ValueError("This checkpoint already completed its configured training schedule")
    initial = {name: value.detach().clone() for name, value in parameters.items()} if args.smoke else None
    policy.train()
    print(f"Training {len(train_rows)} windows; {run_info['trainable_parameters']} trainable parameters", flush=True)
    while progress["step"] < total_steps:
        order = np.random.default_rng(options["seed"] + progress["epoch"]).permutation(train_rows)
        if progress["cursor"] >= len(order):
            progress["epoch"] += 1
            progress["cursor"] = 0
            continue
        selected = order[progress["cursor"]:progress["cursor"] + options["batch_size"]]
        batch = prepare_batch(dataset, selected, normalizer, tokenizer, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = policy(batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()),
                                                       policy.config.optimizer_grad_clip_norm,
                                                       error_if_nonfinite=True)
        if args.smoke and progress["step"] == 0:
            gradient_report = {}
            for prefix in TRAINABLE_PREFIXES:
                gradients = [value.grad for name, value in parameters.items()
                             if name.startswith("model." + prefix) and value.grad is not None]
                gradient_report[prefix] = sum(float(value.float().norm()) for value in gradients)
            if not all(value > 0 for value in gradient_report.values()):
                raise RuntimeError(f"Missing conditioning/head gradients: {gradient_report}")
            write_json(destination / "gradients.json", gradient_report)
        optimizer.step()
        scheduler.step()
        progress["step"] += 1
        progress["cursor"] += len(selected)
        metrics.update(step=progress["step"], grad_norm=float(gradient_norm), lr=scheduler.get_last_lr()[0])
        if progress["step"] % options["validation_every"] == 0 or progress["step"] == total_steps:
            metrics["validation"] = evaluate(policy, dataset, validation_rows, normalizer, tokenizer,
                                              args.device, 1 if args.smoke else options["validation_batches"])
        with (destination / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps({key: value for key, value in metrics.items() if key != "loss_per_dim"}), flush=True)
        if progress["step"] % options["save_every"] == 0 or progress["step"] == total_steps:
            checkpoint = destination / f"checkpoint-{progress['step']:06d}"
            save_checkpoint(checkpoint, policy, optimizer, scheduler, progress, run_info)
    if args.smoke:
        policy.eval()
        with torch.no_grad():
            predictions = policy.predict_action_chunk(batch, num_steps=2)
            physical = normalizer.transform(predictions, "action", inverse=True)
            if physical.shape != (len(selected), 50, 10) or not torch.isfinite(physical).all():
                raise RuntimeError("Invalid inference result")
            before = {name: value.detach().clone() for name, value in parameters.items()}
            for value in parameters.values():
                value.zero_()
            load_delta(policy, checkpoint)
            if not all(torch.equal(before[name], value) for name, value in parameters.items()):
                raise RuntimeError("Checkpoint round trip failed")
            changed = sum(not torch.equal(initial[name], value) for name, value in parameters.items())
        write_json(destination / "smoke_result.json", {"real_checkpoint_loaded": True,
                   "real_images_and_language": True, "optimizer_steps": total_steps,
                   "changed_tensors": changed, "prediction_shape": list(physical.shape),
                   "checkpoint_roundtrip": True, "validation": metrics["validation"],
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-data", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--config", default="configs/pi05_force_sft.json")
    parser.add_argument("--windows")
    parser.add_argument("--contract", default="outputs/pi05_force_prepared/data_contract.json")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    settings = json.loads(local_path(args.config).read_text())
    windows = local_path(args.windows or settings["dataset_windows"])
    if args.check_data:
        print(json.dumps(check_window_contract(windows), indent=2))
        return
    if args.output is None:
        args.output = "outputs/pi05_force_prepared" if args.prepare else "outputs/pi05_force_sft"
    dataset = ForceVLAWindowDataset(windows, include_images=True)
    if args.prepare:
        contract = prepare(dataset, args.output, settings["training"]["seed"])
        print(json.dumps({key: value for key, value in contract.items() if key != "statistics"}, indent=2))
        return
    run_training(args, dataset)


if __name__ == "__main__":
    main()
