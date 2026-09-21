"""Episode splits and train-only preprocessing for offline force PI05."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def local_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path must stay inside force-VLA: {path}")
    return path


def output_path(value):
    path = local_path(value)
    if not path.is_relative_to(ROOT / "outputs") or path == ROOT / "outputs":
        raise ValueError("Generated artifacts must be in a subdirectory of outputs")
    return path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = output_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def split_episodes(episodes, seed=42, validation_fraction=0.1):
    unique = np.unique(episodes)
    if len(unique) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Need at least two episodes and a validation fraction in (0, 1)")
    shuffled = np.random.default_rng(seed).permutation(unique)
    count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    return sorted(shuffled[count:].tolist()), sorted(shuffled[:count].tolist())


def fit_statistics(table, train_rows, max_windows=10000):
    if len(train_rows) == 0 or max_windows < 1:
        raise ValueError("No training rows for normalization")
    selected = np.asarray(train_rows)[np.linspace(0, len(train_rows) - 1,
                                                min(max_windows, len(train_rows)), dtype=int)]
    subset = table.take(selected)

    def values(name, dimension):
        array = np.asarray(subset[name].combine_chunks().flatten()).reshape(-1, dimension)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite normalization input: {name}")
        return array

    statistics = {"fit_window_count": len(selected), "sampling": "deterministic evenly spaced training windows"}
    for column, name in (("state", "state"), ("future_actions", "action")):
        array = values(column, 10)
        lower, upper = np.quantile(array, [0.01, 0.99], axis=0)
        statistics[name] = {"offset": ((lower + upper) / 2).tolist(),
                            "scale": np.maximum((upper - lower) / 2, 1e-4).tolist(),
                            "method": "q01_q99"}
    torque = np.concatenate((values("joint_torque_history", 7), values("future_joint_torque", 7)))
    statistics["torque"] = {"offset": torque.mean(axis=0).tolist(),
                            "scale": np.maximum(torque.std(axis=0), 1e-4).tolist(), "method": "mean_std"}
    return statistics


class ForceNormalizer:
    def __init__(self, statistics):
        self.statistics = statistics

    def transform(self, tensor, name, inverse=False):
        info = self.statistics[name]
        offset = tensor.new_tensor(info["offset"])
        scale = tensor.new_tensor(info["scale"])
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid normalization scale")
        return tensor * scale + offset if inverse else (tensor - offset) / scale

    def __call__(self, batch):
        result = dict(batch)
        for key, name in (("observation.state", "state"), ("action", "action"),
                          ("joint_torque_history", "torque"), ("future_joint_torque", "torque")):
            if key in batch:
                result[key] = self.transform(batch[key], name)
        return result


def prepare(dataset, destination, seed=42, validation_fraction=0.1):
    destination = output_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    episodes = np.asarray(dataset.table["episode_index"])
    if not all(dataset.table["episode_success"].to_pylist()):
        raise ValueError("SFT input includes unsuccessful episodes")
    tasks = set(dataset.table["task_text"].to_pylist())
    if any(not task.strip() or task.strip() == "111" for task in tasks):
        raise ValueError("Unconfirmed task text in windows")
    train, validation = split_episodes(episodes, seed, validation_fraction)
    train_rows = np.flatnonzero(np.isin(episodes, train))
    statistics = fit_statistics(dataset.table, train_rows)
    contract = {"windows": str(dataset.parquet_path.resolve()), "sha256": digest(dataset.parquet_path),
                "train_episodes": train, "validation_episodes": validation, "seed": seed,
                "train_windows": len(train_rows), "validation_windows": len(episodes) - len(train_rows),
                "normalization_source": "training episodes only", "statistics": statistics}
    write_json(destination / "data_contract.json", contract)
    return contract


def load_contract(dataset, path):
    contract = json.loads(local_path(path).read_text())
    if digest(dataset.parquet_path) != contract["sha256"]:
        raise ValueError("Window data changed since preparation")
    train, validation = set(contract["train_episodes"]), set(contract["validation_episodes"])
    episodes = np.asarray(dataset.table["episode_index"])
    if train & validation or train | validation != set(episodes.tolist()) or not train or not validation:
        raise ValueError("Invalid episode split")
    return contract
