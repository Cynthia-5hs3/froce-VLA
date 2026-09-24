"""Local PI05 configuration, window loaders and offline SFT entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lerobot.configs.types import FeatureType, PolicyFeature
from torch.utils.data import DataLoader

from .configuration_pi05_force import ForcePI05Config
from .modeling_pi05_force import ForcePI05Policy, load_pi05_backbone
from .window_dataset import ForceVLAWindowDataset, collate_force_windows, validate_force_window_batch


def load_force_config(path: str | Path) -> ForcePI05Config:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    training = data.get("training", {})
    config = ForcePI05Config(
        max_state_dim=int(data.get("max_state_dim", 32)),
        max_action_dim=int(data.get("max_action_dim", 32)),
        chunk_size=int(data.get("future_steps", 50)),
        n_action_steps=int(data.get("future_steps", 50)),
        action_dim=int(data.get("action_dim", 10)),
        state_dim=int(data.get("state_dim", 10)),
        torque_dim=int(data.get("torque_dim", 7)),
        torque_history_steps=int(data.get("torque_history_steps", 10)),
        future_torque_loss_weight=float(data.get("future_torque_loss_weight", 0.1)),
        use_external_torque=bool(data.get("use_external_torque", False)),
        conditioning_layout=str(data.get("conditioning_layout", "fused")),
        lora_rank=int(training.get("lora_rank", 16)),
        lora_alpha=float(training.get("lora_alpha", 16.0)),
        lora_expert_rank=int(training.get("lora_expert_rank", 32)),
        lora_expert_alpha=float(training.get("lora_expert_alpha", 32.0)),
        dtype=str(training.get("dtype", "bfloat16")),
        freeze_vision_encoder=bool(training.get("freeze_vision_encoder", True)),
        train_expert_only=bool(training.get("train_expert_only", False)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        compile_model=bool(training.get("compile_model", False)),
        device="cpu",
        optimizer_lr=float(training.get("learning_rate", 2.5e-5)),
    )
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
        "observation.images.left_wrist_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(config.state_dim,)
        ),
    }
    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION, shape=(config.action_dim,)
        ),
    }
    return config


def make_force_loader(
    parquet_path: str | Path,
    batch_size: int = 1,
    shuffle: bool = True,
    include_images: bool = False,
) -> DataLoader:
    dataset = ForceVLAWindowDataset(parquet_path, include_images=include_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_force_windows)


def build_force_policy(config: ForcePI05Config, checkpoint_dir: str | Path | None = None) -> tuple[ForcePI05Policy, dict[str, int]]:
    policy = ForcePI05Policy(config)
    report: dict[str, int] = {}
    if checkpoint_dir is not None:
        report = load_pi05_backbone(policy.model, checkpoint_dir)
    return policy, report


def check_window_contract(parquet_path: str | Path, batch_size: int = 2) -> dict[str, Any]:
    loader = make_force_loader(parquet_path, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    validate_force_window_batch(batch)
    return {
        "rows": len(loader.dataset),
        "batch_size": int(batch["action"].shape[0]),
        "state_shape": list(batch["observation.state"].shape),
        "torque_history_shape": list(batch["joint_torque_history"].shape),
        "future_action_shape": list(batch["action"].shape),
        "future_torque_shape": list(batch["future_joint_torque"].shape),
    }


def main() -> None:
    from .training import main as training_main

    training_main()


if __name__ == "__main__":
    main()
