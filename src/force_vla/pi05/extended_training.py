"""Continue a completed force SFT run with a new, explicitly budgeted schedule."""

import argparse
import json
import math
import signal
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from .evaluation import evaluate_loss, evaluate_samples
from .preparation import ForceNormalizer, digest, local_path, output_path, write_json
from .train_pi05_force import build_force_policy, load_force_config
from .training import configure_trainable, load_delta, prepare_batch, restore_training_state, save_checkpoint
from .window_dataset import ForceVLAWindowDataset
from .phases import phase_values, training_epoch_order


def continuation_scheduler(optimizer, options, steps):
    warmup = options["warmup_steps"]
    peak = options["learning_rate"]
    start = options["warmup_start_lr"]
    floor = options["minimum_lr"]
    if not 0 <= warmup < steps or not 0 < floor <= start <= peak:
        raise ValueError("Invalid continuation learning-rate schedule")
    for group in optimizer.param_groups:
        group["lr"] = peak
        group["initial_lr"] = peak

    def multiplier(step):
        if warmup and step < warmup:
            return (start + (peak - start) * step / warmup) / peak
        fraction = min(1.0, max(0.0, (step - warmup) / (steps - warmup)))
        return (floor + (peak - floor) * (1 + math.cos(math.pi * fraction)) / 2) / peak

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def advance_rows(indices, progress, count, seed, phases=None, critical_fraction=None):
    if not len(indices) or count < 1 or not 0 <= progress["cursor"] <= len(indices):
        raise ValueError("Invalid sample cursor or batch size")
    selected = []
    while len(selected) < count:
        if progress["cursor"] >= len(indices):
            progress["epoch"] += 1
            progress["cursor"] = 0
        order = training_epoch_order(indices, seed + progress["epoch"], phases, critical_fraction)
        take = min(count - len(selected), len(order) - progress["cursor"])
        selected.extend(order[progress["cursor"]:progress["cursor"] + take].tolist())
        progress["cursor"] += take
    return selected


def restore_extension(policy, optimizer, source):
    load_delta(policy, source)
    state = torch.load(source / "training.pt", map_location="cpu", weights_only=True)
    progress = dict(state["progress"])
    complete = json.loads((source / "complete.json").read_text())
    if progress["step"] != complete["step"]:
        raise ValueError("Source checkpoint progress differs from completion marker")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pi05_force_continue_30k.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--skip-initial-evaluation", action="store_true")
    args = parser.parse_args()
    options = json.loads(local_path(args.config).read_text())
    for key in ("total_steps", "micro_batch_size", "gradient_accumulation_steps", "validation_every",
                "validation_windows_per_episode", "save_every", "inference_every",
                "inference_windows_per_episode", "inference_steps", "torch_threads"):
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"Invalid {key}")
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("max-updates must be positive")
    if not options["inference_seeds"]:
        raise ValueError("At least one inference seed is required")
    source = local_path(options["source_checkpoint"])
    previous = json.loads((source / "run.json").read_text())
    settings = previous["settings"]
    settings = json.loads(json.dumps(settings))
    settings["training"]["learning_rate"] = options["learning_rate"]
    settings["training"]["steps"] = options["total_steps"]
    settings["training"]["batch_size"] = options["micro_batch_size"]
    settings["training"]["gradient_accumulation_steps"] = options["gradient_accumulation_steps"]
    for key in ("validation_every", "save_every", "validation_windows_per_episode"):
        settings["training"][key] = options[key]
    settings["training"].pop("validation_batches", None)
    dataset = ForceVLAWindowDataset(settings["dataset_windows"], include_images=True)
    phases = phase_values(dataset)
    contract = previous["data_contract"]
    if digest(dataset.parquet_path) != contract["sha256"]:
        raise ValueError("Window data changed since the source run")
    base_path = local_path(settings["base_checkpoint"])
    if digest(base_path / "model.safetensors") != previous["base_sha256"]:
        raise ValueError("Base checkpoint changed")
    train_episodes = set(contract["train_episodes"])
    validation_episodes = set(contract["validation_episodes"])
    episodes = np.asarray(dataset.table["episode_index"])
    if train_episodes & validation_episodes or train_episodes | validation_episodes != set(episodes.tolist()):
        raise ValueError("Invalid episode split")
    train_rows = np.flatnonzero(np.isin(episodes, list(train_episodes)))
    validation_rows = np.flatnonzero(np.isin(episodes, list(validation_episodes)))
    if not len(train_rows) or not len(validation_rows):
        raise ValueError("Empty split")
    origin = json.loads((source / "complete.json").read_text())["step"]
    if options["total_steps"] <= origin:
        raise ValueError("New budget must exceed source step")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use scripts/run.sh --gpu")
    torch.set_num_threads(options["torch_threads"])
    torch.manual_seed(settings["training"]["seed"])
    destination = output_path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "force_config.json", settings)
    normalizer = ForceNormalizer(contract["statistics"])
    tokenizer = AutoTokenizer.from_pretrained(str(local_path(settings["tokenizer"])), local_files_only=True)
    print("Loading base model and source adapter; hardware is inaccessible.", flush=True)
    policy, loading = build_force_policy(load_force_config(destination / "force_config.json"), base_path)
    parameters = configure_trainable(policy, settings["training"]["trainable"])
    policy.to(args.device)
    optimizer = policy.config.get_optimizer_preset().build(parameters.values())
    progress = restore_extension(policy, optimizer, source)
    scheduler = continuation_scheduler(optimizer, options, options["total_steps"] - origin)
    run_info = {"kind": "force_pi05_sft_extended", "settings": settings, "extension": options,
                "source_checkpoint": str(source), "source_adapter_sha256": digest(source / "adapter.safetensors"),
                "schedule_origin_step": origin, "base_checkpoint": settings["base_checkpoint"],
                "base_sha256": previous["base_sha256"], "data_contract": contract, "loading": loading,
                "total_steps": options["total_steps"],
                "trainable_parameters": sum(value.numel() for value in parameters.values()),
                "code_sha256": {str(path.relative_to(local_path("."))): digest(path)
                                for path in sorted(local_path("src/force_vla/pi05").glob("*.py"))}}
    if args.resume:
        checkpoint = local_path(args.resume)
        saved_run = json.loads((checkpoint / "run.json").read_text())
        for key in ("settings", "extension", "base_sha256", "data_contract", "source_adapter_sha256", "code_sha256"):
            if saved_run[key] != run_info[key]:
                raise ValueError(f"Resume contract differs: {key}")
        progress = restore_training_state(policy, optimizer, scheduler, checkpoint)
    if progress["step"] >= options["total_steps"]:
        raise ValueError("Budget already completed")
    write_json(destination / "run.json", run_info)
    write_json(destination / "extension_config.json", options)
    pending_stop = []

    def stop_requested(signum, frame):
        pending_stop.append(signum)
        print("Stop requested; saving after the current update/evaluation.", flush=True)

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)

    def checkpoint_now():
        checkpoint = destination / f"checkpoint-{progress['step']:06d}"
        if not checkpoint.exists():
            save_checkpoint(checkpoint, policy, optimizer, scheduler, progress, run_info)
        return checkpoint

    def evaluation_now(include_samples):
        result = evaluate_loss(policy, dataset, validation_rows, normalizer, tokenizer, args.device,
                               options["validation_windows_per_episode"],
                               phase_balanced=settings["training"].get("phase_balanced_evaluation", False))
        result["step"] = progress["step"]
        write_json(destination / f"validation-{progress['step']:06d}.json", result)
        checkpoint = checkpoint_now()
        best_path = destination / "best_validation.json"
        value = result["episode_macro_mean"]["action_loss"]
        if not best_path.exists() or value < json.loads(best_path.read_text())["action_loss"]:
            write_json(best_path, {"step": progress["step"], "action_loss": value,
                                  "checkpoint": str(checkpoint), "selection": "episode macro action flow loss",
                                  "deployment_ready": False})
        print(json.dumps({"step": progress["step"], "validation": result["episode_macro_mean"],
                          "windows": result["window_count"]}), flush=True)
        if include_samples:
            evaluation_options = dict(options)
            evaluation_options["phase_balanced_evaluation"] = settings["training"].get("phase_balanced_evaluation", False)
            if progress["step"] == options["total_steps"]:
                evaluation_options["inference_windows_per_episode"] = options.get(
                    "final_inference_windows_per_episode", options["inference_windows_per_episode"])
            samples = evaluate_samples(policy, dataset, validation_rows, normalizer, tokenizer, args.device, evaluation_options)
            samples["step"] = progress["step"]
            write_json(destination / f"inference-{progress['step']:06d}.json", samples)
            print(json.dumps({"step": progress["step"], "inference": {
                variant: values["episode_macro_mean"] for variant, values in samples["variants"].items()}}), flush=True)
        return result["episode_macro_mean"]

    checkpoint_now()
    if not args.skip_initial_evaluation:
        write_json(destination / "status.json", {"status": "initial_evaluation", **progress,
                   "total_steps": options["total_steps"], "updated_unix": time.time()})
        evaluation_now(True)
    updates = 0
    effective_batch = options["micro_batch_size"] * options["gradient_accumulation_steps"]
    policy.train()
    while progress["step"] < options["total_steps"] and not pending_stop:
        if args.max_updates is not None and updates >= args.max_updates:
            break
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        learning_rate_used = optimizer.param_groups[0]["lr"]
        metrics = {"loss": 0., "action_loss": 0., "torque_loss": 0.}
        dimension_losses = np.zeros(10)
        next_progress = dict(progress)
        for micro_step in range(options["gradient_accumulation_steps"]):
            selected = advance_rows(train_rows, next_progress, options["micro_batch_size"], settings["training"]["seed"],
                                    phases, settings["training"].get("critical_sample_fraction"))
            batch = prepare_batch(dataset, selected, normalizer, tokenizer, args.device)
            loss, output = policy(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; last complete checkpoint retained")
            (loss / options["gradient_accumulation_steps"]).backward()
            for key in metrics:
                metrics[key] += output[key] / options["gradient_accumulation_steps"]
            dimension_losses += np.asarray(output["loss_per_dim"]) / options["gradient_accumulation_steps"]
        gradient_norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), policy.config.optimizer_grad_clip_norm,
                                                      error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        progress = next_progress
        progress["step"] += 1
        updates += 1
        metrics.update(step=progress["step"], loss_per_dim=dimension_losses.tolist(),
                       grad_norm=float(gradient_norm), lr=learning_rate_used, next_lr=scheduler.get_last_lr()[0],
                       effective_batch=effective_batch, epoch=progress["epoch"], cursor=progress["cursor"],
                       update_seconds=time.perf_counter() - started)
        is_final = progress["step"] == options["total_steps"]
        if progress["step"] % options["save_every"] == 0 or is_final:
            checkpoint_now()
        if progress["step"] % options["validation_every"] == 0 or is_final:
            metrics["validation"] = evaluation_now(progress["step"] % options["inference_every"] == 0 or is_final)
        with (destination / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        write_json(destination / "status.json", {"status": "running", **progress,
                   "total_steps": options["total_steps"], "updated_unix": time.time(),
                   "update_seconds": metrics["update_seconds"], "effective_batch": effective_batch})
        if updates == 1 or progress["step"] % 20 == 0:
            print(json.dumps(metrics, allow_nan=False), flush=True)
    checkpoint = checkpoint_now()
    completed = progress["step"] == options["total_steps"]
    write_json(destination / "status.json", {"status": "complete" if completed else "paused", **progress,
               "total_steps": options["total_steps"], "checkpoint": str(checkpoint), "updated_unix": time.time(),
               "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
               "hardware_tested": False, "deployment_ready": False})
    print(json.dumps({"status": "complete" if completed else "paused", **progress}), flush=True)


if __name__ == "__main__":
    main()
