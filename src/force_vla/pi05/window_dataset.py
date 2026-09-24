"""Dataset adapter for the pre-built force-VLA temporal windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .modeling_pi05_base import resize_with_pad_torch
from .preparation import local_path


class ForceVLAWindowDataset(Dataset):
    """Read one causal state/action/torque window without crossing episodes."""

    def __init__(self, parquet_path: str | Path, include_images: bool = False) -> None:
        self.parquet_path = local_path(parquet_path)
        self.table = pq.read_table(self.parquet_path)
        self.include_images = include_images
        metadata_path = self.parquet_path.with_name("metadata.json")
        metadata = {}
        if metadata_path.is_file():
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.dataset_root = local_path(metadata["dataset_root"]) if metadata.get("dataset_root") else None
        required = {
            "state",
            "joint_torque_history",
            "future_actions",
            "future_joint_torque",
            "task_text",
        }
        missing = required.difference(self.table.column_names)
        if missing:
            raise ValueError(f"windows parquet is missing fields: {sorted(missing)}")

    def __len__(self) -> int:
        return self.table.num_rows

    def _array(self, name: str, index: int, shape: tuple[int, ...]) -> Tensor:
        value = np.asarray(self.table[name][index].as_py(), dtype=np.float32)
        if value.shape != shape:
            if value.size != int(np.prod(shape)):
                raise ValueError(f"{name}[{index}] has shape {value.shape}, expected {shape}")
            value = value.reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError(f"{name}[{index}] contains non-finite values")
        return torch.from_numpy(value.copy())

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "observation.state": self._array("state", index, (10,)),
            "joint_torque_history": self._array("joint_torque_history", index, (10, 7)),
            "action": self._array("future_actions", index, (50, 10)),
            "future_joint_torque": self._array("future_joint_torque", index, (50, 7)),
            "task": str(self.table["task_text"][index].as_py()),
            "episode_index": int(self.table["episode_index"][index].as_py()),
            "anchor_frame": int(self.table["anchor_frame"][index].as_py()),
            "anchor_phase": float(self.table["anchor_phase"][index].as_py())
            if "anchor_phase" in self.table.column_names else 0.0,
        }
        if self.include_images:
            for camera in ("base", "left_wrist"):
                path = Path(self.table[f"{camera}_video_path"][index].as_py())
                item[f"{camera}_video_path"] = str(local_path(path if path.is_absolute() or self.dataset_root is None else self.dataset_root / path))
            item["base_video_frame"] = int(self.table["base_video_frame"][index].as_py())
            item["left_wrist_video_frame"] = int(self.table["left_wrist_video_frame"][index].as_py())
        return item


def collate_force_windows(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack numeric window fields and preserve video/text metadata."""
    if not items:
        raise ValueError("cannot collate an empty force-VLA batch")
    batch: dict[str, Any] = {
        "observation.state": torch.stack([item["observation.state"] for item in items]),
        "joint_torque_history": torch.stack([item["joint_torque_history"] for item in items]),
        "action": torch.stack([item["action"] for item in items]),
        "future_joint_torque": torch.stack([item["future_joint_torque"] for item in items]),
        "task": [item["task"] for item in items],
        "episode_index": torch.tensor([item["episode_index"] for item in items], dtype=torch.long),
        "anchor_frame": torch.tensor([item["anchor_frame"] for item in items], dtype=torch.long),
        "anchor_phase": torch.tensor([item.get("anchor_phase", 0.0) for item in items], dtype=torch.float32),
    }
    for key in ("base_video_path", "left_wrist_video_path"):
        if key in items[0]:
            batch[key] = [item[key] for item in items]
    for key in ("base_video_frame", "left_wrist_video_frame"):
        if key in items[0]:
            batch[key] = torch.tensor([item[key] for item in items], dtype=torch.long)
    return batch


def validate_force_window_batch(batch: dict[str, Any]) -> None:
    """Fail early if a loader drops the temporal force contract."""
    expected = {
        "observation.state": (10,),
        "joint_torque_history": (10, 7),
        "action": (50, 10),
        "future_joint_torque": (50, 7),
    }
    for name, shape in expected.items():
        value = batch[name]
        if tuple(value.shape[1:]) != shape:
            raise ValueError(f"{name} batch shape {tuple(value.shape)} does not end in {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} batch contains non-finite values")


def decode_video_frame(path: str | Path, frame_index: int, size: int = 224) -> Tensor:
    """Decode one RGB video frame as a PI0.5-ready ``[3, size, size]`` tensor."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    import av

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index == frame_index:
                image = torch.from_numpy(frame.to_ndarray(format="rgb24")).to(dtype=torch.float32) / 255.0
                return resize_with_pad_torch(image, size, size).permute(0, 3, 1, 2)[0]
    raise IndexError(f"video frame {frame_index} is unavailable in {video_path}")


def attach_visual_language_inputs(
    batch: dict[str, Any],
    tokenizer: Any,
    image_size: int = 224,
    tokenizer_max_length: int = 200,
) -> dict[str, Any]:
    """Add PI0.5 image and language tensors to a collated force batch."""
    required = (
        "base_video_path",
        "left_wrist_video_path",
        "base_video_frame",
        "left_wrist_video_frame",
        "task",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"visual-language batch is missing {missing}; use include_images=True")
    for camera in ("base", "left_wrist"):
        batch[f"observation.images.{camera}_0_rgb"] = torch.stack(
            [
                decode_video_frame(path, int(frame), image_size)
                for path, frame in zip(
                    batch[f"{camera}_video_path"],
                    batch[f"{camera}_video_frame"],
                    strict=True,
                )
            ]
        )
    states = batch["observation.state"].detach().cpu().numpy().clip(-1, 1)
    discretized = np.clip(np.digitize(states, np.linspace(-1, 1, 257)[:-1]) - 1, 0, 255)
    prompts = []
    for task, state in zip(batch["task"], discretized, strict=True):
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        if not cleaned or cleaned == "111":
            raise ValueError("Task text must be confirmed before training")
        prompts.append(f"Task: {cleaned}, State: {' '.join(map(str, state))};\nAction: ")
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        prompts,
        max_length=tokenizer_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    batch["observation.language.tokens"] = encoded["input_ids"]
    batch["observation.language.attention_mask"] = encoded["attention_mask"].to(dtype=torch.bool)
    return batch
