"""Prepare immutable annotation-selected, episode-balanced offline experiments."""

import argparse
from collections import Counter
import copy
import json

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
from safetensors.torch import load_file

from .preparation import digest, local_path, output_path, write_json
from ..stage_annotations import episode_source, load_context, read_json, validate_document


STAGE_WEIGHTS = {"approach": 1., "correction": 1.5, "aligned_press": 3.,
                 "seated_hold": 1.5, "release_retreat": 1., "idle": .1}


def one_pass_candidate(document):
    segments = sorted(document["segments"], key=lambda item: item["start_frame"])
    cursor = 0
    sequence = []
    for segment in segments:
        if segment["start_frame"] != cursor:
            return False
        cursor = segment["end_frame"]
        stage = segment["stage"]
        if segment["confidence"] != "confirmed" or segment["imitation"] == "review":
            return False
        if stage == "idle":
            continue
        if stage not in ("approach", "aligned_press", "seated_hold", "release_retreat"):
            return False
        if segment["imitation"] not in ("keep", "downsample"):
            return False
        if not sequence or sequence[-1] != stage:
            sequence.append(stage)
    return cursor == document["source"]["frame_count"] and sequence in (
        ["approach", "aligned_press", "release_retreat"],
        ["approach", "aligned_press", "seated_hold", "release_retreat"])


def sampling_probabilities(rows, one_pass_episodes, one_pass_fraction=.25):
    if not 0 < one_pass_fraction < 1 or not rows:
        raise ValueError("Sampling requires nonempty rows and a fraction between zero and one")
    episodes = np.asarray([row["episode_index"] for row in rows])
    one_pass = np.isin(episodes, list(one_pass_episodes))
    if not one_pass.any() or one_pass.all():
        raise ValueError("Need both one-pass and other approved demonstrations")
    probabilities = np.zeros(len(rows), dtype=np.float64)
    for membership, fraction in ((one_pass, one_pass_fraction), (~one_pass, 1 - one_pass_fraction)):
        group_episodes = np.unique(episodes[membership])
        for episode in group_episodes:
            selected = np.flatnonzero(episodes == episode)
            weights = np.asarray([STAGE_WEIGHTS[rows[index]["anchor_stage"]]
                * rows[index]["suggested_sampling_weight"] for index in selected])
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Invalid approved window weight")
            probabilities[selected] = fraction / len(group_episodes) * weights / weights.sum()
    return probabilities / probabilities.sum()


def load_sampling_plan(dataset, contract):
    episodes = np.asarray(dataset.table["episode_index"])
    train_rows = np.flatnonzero(np.isin(episodes, contract["train_episodes"]))
    plan = contract.get("sampling_plan")
    if plan is None:
        return train_rows, None
    path = local_path(plan["path"])
    if digest(path) != plan["sha256"] or digest(dataset.parquet_path) != contract["sha256"]:
        raise ValueError("Sampling plan or source windows changed")
    table = parquet.read_table(path)
    rows = np.asarray(table["source_window_row"], dtype=np.int64)
    probabilities = np.asarray(table["probability"], dtype=np.float64)
    if not len(rows) or len(np.unique(rows)) != len(rows) or not np.isin(rows, train_rows).all():
        raise ValueError("Sampling plan contains duplicate, missing or non-training rows")
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any() or not np.isclose(probabilities.sum(), 1.):
        raise ValueError("Invalid sampling probabilities")
    return rows, probabilities


def expand_history_weight(weight, history_steps=10):
    if weight.ndim != 2 or weight.shape[1] != history_steps * 7:
        raise ValueError("Expected a seven-joint history projection")
    expanded = weight.new_zeros((weight.shape[0], history_steps, 14))
    expanded[:, :, :7] = weight.reshape(weight.shape[0], history_steps, 7)
    return expanded.flatten(1)


def initialize_stage_checkpoint(policy, checkpoint, contract, base_sha256):
    checkpoint = local_path(checkpoint)
    if not (checkpoint / "complete.json").is_file():
        raise ValueError("Incomplete initialization checkpoint")
    previous = read_json(checkpoint / "run.json")
    if previous["base_sha256"] != base_sha256:
        raise ValueError("Initialization base checkpoint differs")
    for field in ("state", "action", "torque"):
        if previous["data_contract"]["statistics"][field] != contract["statistics"][field]:
            raise ValueError(f"Initialization normalization differs: {field}")
    from .train_pi05_force import load_force_config

    old_config = load_force_config(checkpoint / "force_config.json")
    for field in ("conditioning_layout", "state_dim", "action_dim", "torque_dim", "torque_history_steps",
                  "max_action_dim", "chunk_size", "lora_rank", "lora_alpha", "lora_expert_rank", "lora_expert_alpha"):
        if getattr(old_config, field) != getattr(policy.config, field):
            raise ValueError(f"Initialization model contract differs: {field}")
    if old_config.use_external_torque:
        raise ValueError("Stage A/B initialization must use the same measured-torque-only source")
    weights = load_file(str(checkpoint / "adapter.safetensors"))
    parameters = {name: value for name, value in policy.named_parameters() if value.requires_grad}
    if set(weights) != set(parameters):
        raise ValueError("Initialization parameter set differs")
    expanded = []
    for name, parameter in parameters.items():
        if parameter.shape != weights[name].shape:
            if name != "model.torque_adapter.projection.0.weight" or not policy.config.use_external_torque:
                raise ValueError(f"Unexpected initialization tensor shape: {name}")
            weights[name] = expand_history_weight(weights[name], policy.config.torque_history_steps)
            expanded.append(name)
        if parameter.shape != weights[name].shape:
            raise ValueError(f"Initialization shape mismatch: {name}")
    policy.load_state_dict(weights, strict=False)
    return {"checkpoint": str(checkpoint), "adapter_sha256": digest(checkpoint / "adapter.safetensors"),
            "expanded_zero_external_columns": expanded, "optimizer_restored": False}


def prepare_experiment(args):
    context = load_context(args.contract)
    index = local_path(args.index)
    summary = read_json(index / "summary.json")
    candidates = []
    for audit in summary["annotations"]:
        path = local_path(audit["path"])
        if digest(path) != audit["sha256"]:
            raise ValueError("Annotations changed after index creation; rebuild the index")
        document = read_json(path)
        identity, _, _ = episode_source(context, document["source"]["episode_index"])
        validate_document(document, identity)
        if identity["split"] == "train" and one_pass_candidate(document):
            candidates.append(identity["episode_index"])
    selection = read_json(index / "eligible_train_rows.json")
    if selection["windows_sha256"] != context["contract"]["sha256"]:
        raise ValueError("Index source windows differ")
    rows = [row for row in parquet.read_table(index / "window_review.parquet").to_pylist()
            if row["all_future_actions_approved"]]
    if [row["source_window_row"] for row in rows] != selection["rows"]:
        raise ValueError("Eligible rows and masks differ")
    for row in rows:
        if len(row["action_target_mask"]) != 50 or not all(row["action_target_mask"]):
            raise ValueError("Partial action targets cannot enter this experiment")
        if row["episode_index"] not in context["contract"]["train_episodes"]:
            raise ValueError("Non-training episode in selected windows")
    probabilities = sampling_probabilities(rows, candidates, args.one_pass_fraction)
    for row, probability in zip(rows, probabilities, strict=True):
        row["probability"] = float(probability)
        row["one_pass_candidate"] = row["episode_index"] in candidates
    source = local_path(args.checkpoint)
    previous = read_json(source / "run.json")
    for field in ("state", "action", "torque"):
        if previous["data_contract"]["statistics"][field] != context["contract"]["statistics"][field]:
            raise ValueError("Source normalization must match the fixed evaluation contract")
    table = parquet.read_table(context["windows"], columns=["external_joint_torque_history"])
    selected = np.asarray(selection["rows"])
    selected = selected[np.linspace(0, len(selected) - 1, min(10000, len(selected)), dtype=int)]
    external = np.asarray(table.take(selected)["external_joint_torque_history"].combine_chunks().flatten()).reshape(-1, 7)
    if not np.isfinite(external).all():
        raise ValueError("Nonfinite external torque statistics")
    external_statistics = {"offset": external.mean(0).tolist(),
        "scale": np.maximum(external.std(0), 1e-4).tolist(), "method": "mean_std",
        "fit_window_count": len(selected), "source": "approved training history only"}
    destination = output_path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    plan = destination / "sampling_plan.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows), plan, compression="zstd")
    sampling = {"path": str(plan), "sha256": digest(plan),
        "annotation_index": str(index), "index_summary_sha256": digest(index / "summary.json"),
        "one_pass_candidates": candidates, "one_pass_fraction": args.one_pass_fraction,
        "stage_weights": STAGE_WEIGHTS, "episode_balanced_within_group": True,
        "selection": "all 50 future actions approved; replacement sampling; no RB oversampling"}
    for name, use_external in (("measured", False), ("measured_external", True)):
        contract = copy.deepcopy(context["contract"])
        contract["sampling_plan"] = sampling
        contract["selected_train_windows"] = len(rows)
        if use_external:
            contract["statistics"]["external_torque"] = external_statistics
        contract_path = destination / f"contract_{name}.json"
        write_json(contract_path, contract)
        settings = copy.deepcopy(previous["settings"])
        settings.update(dataset_windows=str(context["windows"]), use_external_torque=use_external,
                        initialize_checkpoint=str(source), data_contract=str(contract_path))
        settings["training"].pop("critical_sample_fraction", None)
        settings["training"].update(steps=3000, learning_rate=5e-6, batch_size=4,
                                   validation_every=250, save_every=500)
        write_json(destination / f"config_{name}.json", settings)
    expected = Counter()
    for row in rows:
        expected[row["anchor_stage"]] += row["probability"]
    report = {"candidate_windows": len(rows), "one_pass_candidates": candidates,
        "one_pass_basis": "confirmed interval sequence, no marked correction/misaligned contact; not a new video review",
        "expected_stage_fraction": dict(expected), "sampling": sampling,
        "validation_episodes": context["contract"]["validation_episodes"],
        "external_input": "10x14 past history: tau_J[7] then tau_ext_hat_filtered[7] per timestamp",
        "future_targets": "50x10 actions and 50x7 measured joint torque only",
        "trained": False, "hardware_started": False,
        "old_dataset_mixed": False, "normalization": "preserve source state/action/tau_J scaling; fit external on approved train history"}
    write_json(destination / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="outputs/force_vla_prepared_v3/data_contract.json")
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", default="outputs/pi05_force_v3_resume_from_014000/checkpoint-030000")
    parser.add_argument("--one-pass-fraction", type=float, default=.25)
    parser.add_argument("--output", required=True)
    prepare_experiment(parser.parse_args())


if __name__ == "__main__":
    main()
