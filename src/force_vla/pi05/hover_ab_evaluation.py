"""Paired offline checkpoint evaluation of vertical motion and seed sensitivity."""

import argparse
import gc
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from transformers import AutoTokenizer

from .preparation import digest, local_path, output_path, write_json
from .training import prepare_batch, restore_for_inference
from .window_dataset import ForceVLAWindowDataset


def vertical_class(displacement_mm):
    values = np.asarray(displacement_mm)
    return np.where(values < -5, "down", np.where(values > 5, "up", "z_hold"))


def select_windows(dataset, validation_episodes, per_group):
    episodes = np.asarray(dataset.table["episode_index"])
    frames = np.asarray(dataset.table["anchor_frame"])
    states = np.asarray(dataset.table["state"].combine_chunks().flatten()).reshape(-1, 10)
    actions = np.asarray(dataset.table["future_actions"].combine_chunks().flatten()).reshape(-1, 50, 10)
    classes = vertical_class(1000 * (actions[:, 19, 2] - states[:, 2]))
    candidates = (states[:, 2] >= .28) & (states[:, 2] <= .34)
    selected = []
    counts = {}
    for episode in validation_episodes:
        for category in ("down", "z_hold", "up"):
            indices = np.flatnonzero(candidates & (episodes == episode) & (classes == category))
            indices = indices[np.argsort(frames[indices])]
            counts[f"{episode}/{category}"] = len(indices)
            if not len(indices):
                continue
            chosen = []
            for fraction in (np.arange(per_group) + .5) / per_group:
                ordered = sorted(indices.tolist(), key=lambda index: abs(
                    int(frames[index]) - float(np.quantile(frames[indices], fraction))))
                for index in ordered:
                    if all(abs(int(frames[index]) - int(frames[other])) >= 60 for other in chosen):
                        chosen.append(index)
                        break
            selected.extend(chosen)
    if not selected:
        raise ValueError("No matching held-out windows")
    return selected, counts


def summarize_predictions(predictions, targets, states, selections):
    summary = {}
    groups = ["all", "down", "z_hold", "up"]
    for group in groups:
        indices = [index for index, item in enumerate(selections)
                   if group == "all" or item["target_class20"] == group]
        if not indices:
            continue
        predicted = predictions[indices]
        target = targets[indices]
        current = states[indices]
        metrics = {"windows": len(indices), "horizons": {}}
        for horizon in (10, 20, 50):
            predicted_delta = 1000 * (predicted[:, :, horizon - 1, 2] - current[:, None, 2])
            target_delta = 1000 * (target[:, horizon - 1, 2] - current[:, 2])
            predicted_class = vertical_class(predicted_delta)
            target_class = vertical_class(target_delta)
            probabilities = np.stack([(predicted_class == category).mean(axis=1)
                                      for category in ("down", "z_hold", "up")], axis=-1)
            metrics["horizons"][str(horizon)] = {
                "mean_target_dz_mm": float(target_delta.mean()),
                "mean_predicted_dz_mm": float(predicted_delta.mean()),
                "z_mae_mm": float(np.abs(predicted_delta - target_delta[:, None]).mean()),
                "xyz_error_mm": float((1000 * np.linalg.norm(
                    predicted[:, :, horizon - 1, :3] - target[:, None, horizon - 1, :3], axis=-1)).mean()),
                "direction_agreement": float((predicted_class == target_class[:, None]).mean()),
                "p_down": float(probabilities[:, 0].mean()),
                "p_z_hold": float(probabilities[:, 1].mean()),
                "p_up": float(probabilities[:, 2].mean()),
                "mean_seed_z_std_mm": float(predicted_delta.std(axis=1).mean()),
                "independent_draw_disagreement": float((1 - (probabilities ** 2).sum(axis=1)).mean()),
            }
        summary[group] = metrics
    return summary


def summarize_saved(destination):
    manifest = json.loads((destination / "manifest.json").read_text())
    selections = manifest["selections"]
    fixture_indices = [index for index, item in enumerate(selections)
                       if .17 <= item["xyz"][0] <= .23 and -.48 <= item["xyz"][1] <= -.42]
    summaries, fixture_summaries, per_window = {}, {}, {}
    for label in manifest["checkpoints"]:
        with np.load(destination / f"{label}_predictions.npz") as data:
            actions, targets, states = data["actions"], data["targets"], data["states"]
            if data["rows"].tolist() != [item["row"] for item in selections]:
                raise ValueError("Prediction rows differ from manifest")
            if data["seeds"].tolist() != manifest["seeds"]:
                raise ValueError("Prediction seeds differ from manifest")
            summaries[label] = summarize_predictions(actions, targets, states, selections)
            fixture_summaries[label] = summarize_predictions(
                actions[fixture_indices], targets[fixture_indices], states[fixture_indices],
                [selections[index] for index in fixture_indices])
            per_window[label] = [dict(item, metrics=summarize_predictions(
                actions[index:index + 1], targets[index:index + 1], states[index:index + 1], [item])["all"])
                for index, item in enumerate(selections)]
    write_json(destination / "summary.json", {"models": summaries, "hardware_connected": False,
                                               "window_count": len(selections), "seeds": len(manifest["seeds"])})
    write_json(destination / "fixture_summary.json", {
        "models": fixture_summaries, "window_count": len(fixture_indices),
        "roi": {"x_m": [.17, .23], "y_m": [-.48, -.42]},
        "excluded": [item for index, item in enumerate(selections) if index not in fixture_indices],
        "note": "Geometric fixture region, checked against selected observation images; no success or contact labels inferred."})
    write_json(destination / "per_window_summary.json", per_window)
    plot_script = str(Path(__file__).with_name("hover_ab_plot.py"))
    for options in ([], ["--fixture-only"]):
        subprocess.run(["/usr/bin/python3", "-S", plot_script, str(destination), *options], check=True)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    destination = output_path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    checkpoints = {"old_aligned_30k": local_path(args.checkpoint_a), "v3_30k": local_path(args.checkpoint_b)}
    runs = {label: json.loads((path / "run.json").read_text()) for label, path in checkpoints.items()}
    dataset = ForceVLAWindowDataset(args.windows, include_images=True)
    contract = runs["v3_30k"]["data_contract"]
    if digest(dataset.parquet_path) != contract["sha256"]:
        raise ValueError("Evaluation windows do not match checkpoint B contract")
    selected, counts = select_windows(dataset, contract["validation_episodes"], args.per_group)
    selections = []
    for index in selected:
        item = dataset[index]
        state = item["observation.state"].numpy()
        target = item["action"].numpy()
        selections.append({"row": index, "episode": item["episode_index"], "frame": item["anchor_frame"],
                           "anchor_phase": item["anchor_phase"], "xyz": state[:3].tolist(),
                           "target_dz20_mm": float(1000 * (target[19, 2] - state[2])),
                           "target_class20": str(vertical_class(1000 * (target[19, 2] - state[2])))})
    provenance = {}
    for label, path in checkpoints.items():
        source_windows = local_path(runs[label]["data_contract"]["windows"])
        source_metadata = json.loads(source_windows.with_name("metadata.json").read_text())
        source_root = local_path(source_metadata["dataset_root"])
        if source_root == dataset.dataset_root:
            if set(item["episode"] for item in selections) & set(runs[label]["data_contract"]["train_episodes"]):
                raise ValueError(f"Evaluation leaks training episodes for {label}")
        provenance[label] = {"checkpoint": str(path), "adapter_sha256": digest(path / "adapter.safetensors"),
                             "training_dataset_root": str(source_root),
                             "train_episode_count": len(runs[label]["data_contract"]["train_episodes"]),
                             "base_sha256": runs[label]["base_sha256"],
                             "normalization": "checkpoint-specific training statistics"}
    manifest = {"checkpoints": provenance, "windows": str(dataset.parquet_path),
                "windows_sha256": contract["sha256"], "seeds": list(range(12345, 12345 + args.seeds)),
                "inference_steps": args.inference_steps, "selections": selections,
                "candidate_counts": counts, "hardware_connected": False,
                "selection": "B validation episodes; 0.28<=z<=0.34; balanced by target step20 dz; >=60 frame spacing within group",
                "limitations": ["Z hold is not full-pose inactivity", "Not a closed-loop success rate",
                                "Different training datasets and normalizers; not an isolated correction-data ablation",
                                "Separate dataset roots do not by themselves prove no duplicated recordings",
                                "No causal test of temporal replanning; only fixed-observation seed sensitivity"]}
    write_json(destination / "manifest.json", manifest)
    print(f"Selected {len(selected)} windows, {args.seeds} paired seeds per model", flush=True)
    targets = np.stack([dataset[index]["action"].numpy() for index in selected])
    states = np.stack([dataset[index]["observation.state"].numpy() for index in selected])
    summaries = {}
    for label, checkpoint in checkpoints.items():
        print(f"Loading {label}: {checkpoint}", flush=True)
        policy, normalizer = restore_for_inference(checkpoint, args.device)
        dataset.use_external_torque = policy.config.use_external_torque
        tokenizer = AutoTokenizer.from_pretrained(str(local_path(runs[label]["settings"]["tokenizer"])), local_files_only=True)
        actions = np.empty((len(selected), args.seeds, 50, 10), dtype=np.float32)
        torques = np.empty((len(selected), args.seeds, 50, 7), dtype=np.float32)
        with (destination / f"{label}_records.jsonl").open("x") as stream:
            for ordinal, index in enumerate(selected):
                batch = prepare_batch(dataset, [index], normalizer, tokenizer, args.device)
                images, masks = policy._preprocess_images(batch)
                for seed_index, seed in enumerate(manifest["seeds"]):
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    noise = torch.randn((1, 50, policy.config.max_action_dim + 7), generator=generator).to(args.device)
                    output = policy.model.sample_actions(
                        images, masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"],
                        batch["observation.state"], batch["joint_torque_history"], noise=noise,
                        num_steps=args.inference_steps)
                    actions[ordinal, seed_index] = normalizer.transform(output[0, :, :10], "action", inverse=True).cpu().numpy()
                    torques[ordinal, seed_index] = normalizer.transform(output[0, :, -7:], "torque", inverse=True).cpu().numpy()
                    if not np.isfinite(actions[ordinal, seed_index]).all() or not np.isfinite(torques[ordinal, seed_index]).all():
                        raise ValueError("Nonfinite sampled prediction")
                    record = {**selections[ordinal], "seed": seed, "model": label,
                              "dz_mm": {str(horizon): float(1000 * (actions[ordinal, seed_index, horizon - 1, 2] - states[ordinal, 2]))
                                        for horizon in (10, 20, 50)}}
                    stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(f"{label}: {ordinal + 1}/{len(selected)} windows complete", flush=True)
        np.savez_compressed(destination / f"{label}_predictions.npz", actions=actions, torques=torques,
                            targets=targets, states=states, rows=np.asarray(selected), seeds=np.asarray(manifest["seeds"]))
        summaries[label] = summarize_predictions(actions, targets, states, selections)
        write_json(destination / f"{label}_summary.json", summaries[label])
        del policy, normalizer, tokenizer, batch, images, masks, output, noise
        gc.collect()
        torch.cuda.empty_cache()
    summarize_saved(destination)
    print(f"Finished: {destination / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", default="outputs/pi05_force_tavla_aligned_30k_resume/checkpoint-030000")
    parser.add_argument("--checkpoint-b", default="outputs/pi05_force_v3_resume_from_014000/checkpoint-030000")
    parser.add_argument("--windows", default="outputs/force_vla_31d_windows_v3/windows.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if min(args.seeds, args.per_group, args.inference_steps) < 1:
        parser.error("counts must be positive")
    if args.summarize_only:
        summarize_saved(output_path(args.output))
    else:
        run(args)


if __name__ == "__main__":
    main()
