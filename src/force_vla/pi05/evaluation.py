"""Episode-balanced loss and sampled physical-output evaluation, without hardware."""

import time

import numpy as np
import torch

from .training import prepare_batch
from .phases import phase_name, phase_values


def select_episode_windows(dataset, indices, per_episode, phase_balanced=False):
    if per_episode < 1:
        raise ValueError("per_episode must be positive")
    episodes = np.asarray(dataset.table["episode_index"])
    frames = np.asarray(dataset.table["anchor_frame"])
    phases = phase_values(dataset)
    selected = []
    for episode in np.unique(episodes[indices]):
        candidates = indices[episodes[indices] == episode]
        candidates = candidates[np.argsort(frames[candidates], kind="stable")]
        groups = [candidates]
        if phase_balanced:
            groups = [candidates[phases[candidates] == value] for value in (0, 1)]
        for group in groups:
            positions = np.linspace(0, len(group) - 1, min(per_episode, len(group)), dtype=int)
            selected.extend(group[positions].tolist())
    if not selected:
        raise ValueError("No evaluation windows")
    return np.asarray(selected, dtype=int)


def summarize_records(records, fields):
    episodes = sorted({record["episode"] for record in records})
    per_episode = {}
    for episode in episodes:
        subset = [record for record in records if record["episode"] == episode]
        per_episode[str(episode)] = {
            field: float(np.mean([record[field] for record in subset if record[field] is not None]))
            if any(record[field] is not None for record in subset) else None for field in fields
        }
    means = {
        field: float(np.mean([value[field] for value in per_episode.values() if value[field] is not None]))
        if any(value[field] is not None for value in per_episode.values()) else None for field in fields
    }
    return {"episode_macro_mean": means, "per_episode": per_episode, "records": records}


def summarize_phases(records, fields):
    result = {}
    for name in ("marked_critical", "unmarked"):
        subset = [record for record in records if phase_name(record["anchor_phase"]) == name]
        summary = summarize_records(subset, fields)
        result[name] = {
            "record_count": len(subset),
            "window_count": len({(record["episode"], record["frame"]) for record in subset}),
            "episode_count": len(summary["per_episode"]),
            "episode_macro_mean": summary["episode_macro_mean"], "per_episode": summary["per_episode"],
        }
    return result


def evaluate_loss(policy, dataset, indices, normalizer, tokenizer, device, per_episode, phase_balanced=False):
    selected = select_episode_windows(dataset, indices, per_episode, phase_balanced)
    phases = phase_values(dataset)
    records = []
    was_training = policy.training
    policy.eval()
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for index in selected:
                torch.manual_seed(12345 + int(index))
                batch = prepare_batch(dataset, [index], normalizer, tokenizer, device)
                loss, metrics = policy(batch)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite validation loss")
                records.append({"episode": int(batch["episode_index"].item()),
                                "frame": int(batch["anchor_frame"].item()),
                                "anchor_phase": float(phases[index]),
                                **{key: metrics[key] for key in ("loss", "action_loss", "torque_loss")}})
                if len(records) % 32 == 0:
                    print(f"Validation windows: {len(records)}/{len(selected)}", flush=True)
    finally:
        policy.train(was_training)
    result = summarize_records(records, ("loss", "action_loss", "torque_loss"))
    result.update(window_count=len(selected), episode_count=len(result["per_episode"]))
    result["by_anchor_phase"] = summarize_phases(records, ("loss", "action_loss", "torque_loss"))
    result["selection"] = "episode_and_phase" if phase_balanced else "episode"
    return result


def rotation_matrices(values):
    first = values[..., :3]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first = first / np.maximum(first_norm, 1e-8)
    second = values[..., 3:] - np.sum(first * values[..., 3:], axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    second = second / np.maximum(second_norm, 1e-8)
    valid = (first_norm[..., 0] > 1e-6) & (second_norm[..., 0] > 1e-6)
    return np.stack((first, second, np.cross(first, second)), axis=-1), valid


def physical_metrics(actions, target, torque, target_torque):
    if actions.shape != (50, 10) or torque.shape != (50, 7):
        raise ValueError("Expected 50x10 action and 50x7 torque predictions")
    if not all(np.isfinite(value).all() for value in (actions, target, torque, target_torque)):
        raise ValueError("Nonfinite physical prediction or target")
    distance_mm = 1000 * np.linalg.norm(actions[:, :3] - target[:, :3], axis=-1)
    predicted_rotation, valid = rotation_matrices(actions[:, 3:9])
    target_rotation, target_valid = rotation_matrices(target[:, 3:9])
    if not target_valid.all():
        raise ValueError("Invalid reference rot6d")
    cosine = (np.sum(predicted_rotation * target_rotation, axis=(-1, -2)) - 1) / 2
    angle_deg = np.rad2deg(np.arccos(np.clip(cosine[valid], -1, 1)))
    gripper = actions[:, 9]
    differences = torque - target_torque
    return {
        "position_rmse_mm": float(np.sqrt(np.mean(distance_mm ** 2))),
        "first10_position_rmse_mm": float(np.sqrt(np.mean(distance_mm[:10] ** 2))),
        "position_p95_mm": float(np.quantile(distance_mm, 0.95)),
        "rotation_mae_deg": float(angle_deg.mean()) if len(angle_deg) else None,
        "invalid_rotation_fraction": float(1 - valid.mean()),
        "gripper_mae": float(np.abs(gripper - target[:, 9]).mean()),
        "gripper_threshold_error": float(((gripper >= 0.5) != (target[:, 9] >= 0.5)).mean()),
        "gripper_outside_unit_fraction": float(((gripper < 0) | (gripper > 1)).mean()),
        "torque_rmse_nm": float(np.sqrt(np.mean(differences ** 2))),
        "max_chunk_position_step_mm": float(1000 * np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1).max()),
    }


def evaluate_samples(policy, dataset, indices, normalizer, tokenizer, device, options):
    selected = select_episode_windows(dataset, indices, options["inference_windows_per_episode"],
                                      options.get("phase_balanced_evaluation", False))
    phases = phase_values(dataset)
    episodes = np.asarray(dataset.table["episode_index"])
    variants = options["history_variants"]
    if not set(variants) <= {"measured", "zero_nm", "other_episode"} or "measured" not in variants:
        raise ValueError("Invalid torque-history variants")
    if "other_episode" in variants and len(np.unique(episodes[selected])) < 2:
        raise ValueError("Other-episode history requires at least two validation episodes")
    records = {variant: [] for variant in variants}
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    was_training = policy.training
    policy.eval()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for ordinal, index in enumerate(selected):
                started = time.perf_counter()
                batch = prepare_batch(dataset, [index], normalizer, tokenizer, device)
                images, image_masks = policy._preprocess_images(batch)
                target = dataset[int(index)]
                history = batch["joint_torque_history"]
                alternate = {}
                if "zero_nm" in variants:
                    alternate["zero_nm"] = normalizer.transform(torch.zeros_like(history), "torque")
                if "other_episode" in variants:
                    donor = next(int(candidate) for candidate in np.roll(selected, -ordinal - 1)
                                 if episodes[candidate] != episodes[index])
                    donor_history = dataset[donor]["joint_torque_history"].unsqueeze(0).to(device)
                    alternate["other_episode"] = normalizer.transform(donor_history, "torque")
                if devices:
                    torch.cuda.synchronize(device)
                preparation_ms = 1000 * (time.perf_counter() - started)
                for seed in options["inference_seeds"]:
                    torch.manual_seed(seed + int(index))
                    noise = policy.model.sample_noise((1, 50, policy.config.max_action_dim + 7), device)
                    for variant in variants:
                        started = time.perf_counter()
                        output = policy.model.sample_actions(
                            images, image_masks, batch["observation.language.tokens"],
                            batch["observation.language.attention_mask"], batch["observation.state"],
                            history if variant == "measured" else alternate[variant],
                            noise=noise.clone(), num_steps=options["inference_steps"])
                        if devices:
                            torch.cuda.synchronize(device)
                        elapsed_ms = 1000 * (time.perf_counter() - started)
                        actions = normalizer.transform(output[0, :, :10], "action", inverse=True).cpu().numpy()
                        torque = normalizer.transform(output[0, :, -7:], "torque", inverse=True).cpu().numpy()
                        metrics = physical_metrics(actions, target["action"].numpy(), torque,
                                                   target["future_joint_torque"].numpy())
                        records[variant].append({"episode": int(episodes[index]),
                                                 "frame": int(target["anchor_frame"]), "seed": seed,
                                                 "anchor_phase": float(phases[index]),
                                                 "sampling_ms": elapsed_ms, "preparation_ms": preparation_ms,
                                                 **metrics})
                print(f"Sampled inference windows: {ordinal + 1}/{len(selected)}", flush=True)
    finally:
        policy.train(was_training)
    fields = [key for key in records["measured"][0] if key not in ("episode", "frame", "seed", "anchor_phase")]
    return {"window_count": len(selected), "inference_steps": options["inference_steps"],
            "seeds": options["inference_seeds"], "hardware_tested": False,
            "note": "Offline sampled outputs; not a task success rate or deployment approval. Latency includes first-call warmup.",
            "selection": "episode_and_phase" if options.get("phase_balanced_evaluation", False) else "episode",
            "variants": {variant: {**summarize_records(value, fields),
                                   "by_anchor_phase": summarize_phases(value, fields)}
                         for variant, value in records.items()}}
