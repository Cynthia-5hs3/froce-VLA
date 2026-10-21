"""Prepare and inspect two independent external-torque experiments without training."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .preparation import ForceNormalizer, digest, load_contract, local_path, output_path, write_json
from .stage_sampling import load_sampling_plan
from .temporal import HISTORY_OFFSETS, SAMPLE_RATE_HZ
from .window_dataset import ForceVLAWindowDataset, collate_force_windows, validate_force_window_batch


RECIPES = "configs/force_external_variants.json"


def read_json(path):
    return json.loads(local_path(path).read_text())


def fit_external_statistics(table, rows):
    if not len(rows):
        raise ValueError("No training histories available")
    selected = np.asarray(rows)[np.linspace(0, len(rows) - 1, min(10000, len(rows)), dtype=int)]
    values = np.asarray(table.take(selected)["external_joint_torque_history"].combine_chunks().flatten()).reshape(-1, 7)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite external torque history")
    return {"offset": values.mean(0).tolist(), "scale": np.maximum(values.std(0), 1e-4).tolist(),
            "method": "mean_std", "fit_window_count": len(selected), "source": "selected training history only"}


def inspect_data(dataset, contract):
    rows, probabilities = load_sampling_plan(dataset, contract)
    metadata = read_json(dataset.parquet_path.with_name("metadata.json"))
    if (tuple(metadata["history_offsets_frames"]) != HISTORY_OFFSETS
            or metadata["source_info"]["fps"] != SAMPLE_RATE_HZ
            or metadata["future_action_offsets_frames"] != list(range(50))
            or metadata["future_torque_offsets_frames"] != list(range(1, 51))):
        raise ValueError("Temporal contract differs from deployment")
    if not all(dataset.table["episode_success"].to_pylist()):
        raise ValueError("Unexpected unsuccessful episodes in SFT windows")
    if any(not task.strip() or task.strip() == "111" for task in set(dataset.table["task_text"].to_pylist())):
        raise ValueError("Unconfirmed task text")
    for column in ("state", "joint_torque_history", "external_joint_torque_history", "future_actions", "future_joint_torque"):
        values = np.asarray(dataset.table[column].combine_chunks().flatten())
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {column}")
    videos = set()
    for camera in ("base", "left_wrist"):
        for value in set(dataset.table[f"{camera}_video_path"].to_pylist()):
            path = Path(value)
            path = local_path(path if path.is_absolute() else dataset.dataset_root / path)
            if not path.is_file():
                raise FileNotFoundError(path)
            videos.add(str(path))
    return rows, {"train_episodes": len(contract["train_episodes"]),
                  "validation_episodes": len(contract["validation_episodes"]),
                  "selected_train_windows": len(rows), "stage_sampling": probabilities is not None,
                  "video_files_present": len(videos), "video_decode_checked": False,
                  "history_shape": [10, 14], "action_shape": [50, 10], "future_torque_shape": [50, 7]}


def prepare_variant(variant, recipe):
    destination = output_path(recipe["preparation_output"])
    if destination.exists():
        raise FileExistsError(f"Prepared experiment already exists; use --check: {destination}")
    reference = local_path(recipe["reference_checkpoint"])
    if not (reference / "complete.json").is_file():
        raise ValueError("Reference checkpoint is incomplete")
    previous = read_json(reference / "run.json")
    source_contract = local_path(recipe["source_contract"])
    contract = copy.deepcopy(read_json(source_contract))
    dataset = ForceVLAWindowDataset(contract["windows"], use_external_torque=True)
    load_contract(dataset, source_contract)
    for field in ("sha256", "train_episodes", "validation_episodes", "sampling_plan"):
        if contract.get(field) != previous["data_contract"].get(field):
            raise ValueError(f"Reference and experiment data differ: {field}")
    rows, report = inspect_data(dataset, contract)
    if "external_torque" not in contract["statistics"]:
        contract["statistics"]["external_torque"] = fit_external_statistics(dataset.table, rows)
    settings = copy.deepcopy(previous["settings"])
    settings.update(dataset_windows=str(dataset.parquet_path), use_external_torque=True,
                    data_contract=str(destination / "data_contract.json"))
    settings.pop("initialize_checkpoint", None)
    if recipe["initialize_checkpoint"]:
        settings["initialize_checkpoint"] = recipe["initialize_checkpoint"]
    settings["training"]["steps"] = recipe["steps"]
    batch = collate_force_windows([dataset[int(rows[0])]])
    validate_force_window_batch(ForceNormalizer(contract["statistics"])(batch), use_external_torque=True)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "data_contract.json", contract)
    write_json(destination / "config.json", settings)
    report.update(variant=variant, description=recipe["description"], reference_checkpoint=str(reference),
                  reference_run_sha256=digest(reference / "run.json"), source_contract=str(source_contract),
                  source_contract_sha256=digest(source_contract), windows_sha256=contract["sha256"],
                  initialization=settings.get("initialize_checkpoint", settings["base_checkpoint"]),
                  steps=recipe["steps"], batch_size=settings["training"]["batch_size"],
                  training_output=recipe["training_output"], trained=False, hardware_connected=False)
    write_json(destination / "preparation.json", report)
    return report


def check_variant(recipe):
    directory = local_path(recipe["preparation_output"])
    settings = read_json(directory / "config.json")
    if not settings.get("use_external_torque"):
        raise ValueError("Expected an external-torque model")
    dataset = ForceVLAWindowDataset(settings["dataset_windows"], use_external_torque=True)
    contract = load_contract(dataset, settings["data_contract"])
    rows, report = inspect_data(dataset, contract)
    batch = collate_force_windows([dataset[int(index)] for index in rows[:2]])
    validate_force_window_batch(ForceNormalizer(contract["statistics"])(batch), use_external_torque=True)
    return dict(report, trained=False, hardware_connected=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("success_30k", "stage_b"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    recipe = read_json(RECIPES)[args.variant]
    report = prepare_variant(args.variant, recipe) if args.prepare else check_variant(recipe)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
