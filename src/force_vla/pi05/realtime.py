"""Force-PI05 live observation and explicitly armed single-arm inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np
import torch
import yaml

from .preparation import local_path
from .temporal import HISTORY_OFFSETS, SAMPLE_RATE_HZ
from .training import restore_for_inference


ROOT = Path(__file__).resolve().parents[3]
TASK = "Place the white lid firmly onto the jig."
ABORT_KEYS = frozenset(("q", "x", "\x1b"))
RESTART_KEYS = frozenset(("r",))


def is_abort_key(key: str) -> bool:
    return key in ABORT_KEYS


def is_restart_key(key: str) -> bool:
    return key in RESTART_KEYS


class KeyboardStop:
    def __init__(self):
        self._fd = None
        self._attributes = None
        self._restart_requested = False

    def __enter__(self):
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._attributes = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            print("Press r to restart inference; q, x, or Esc stops the loop.", flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._fd is not None and self._attributes is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._attributes)

    def requested(self) -> bool:
        if self._fd is None:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return False
        key = sys.stdin.read(1).lower()
        if is_restart_key(key):
            self._restart_requested = True
            return False
        return is_abort_key(key)

    def restart_requested(self) -> bool:
        requested = self._restart_requested
        self._restart_requested = False
        return requested

    def wait(self, duration_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline:
            if self.requested():
                return True
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return False


def load_robot_config(path: str):
    from evo_rlt.robots.franka_robotiq.config import load_franka_robotiq_config

    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not isinstance(values.get("robot"), dict):
        raise ValueError("robot config must contain a robot mapping")
    return load_franka_robotiq_config(path)


def load_task(path: str) -> str:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    task = str(values.get("deployment", {}).get("task", TASK)).strip()
    if not task or task == "111":
        raise ValueError("deployment task text is empty or unconfirmed")
    return task


def configure_threads(values: dict):
    deployment = values.get("deployment", {})
    threads = int(deployment.get("torch_num_threads", 1))
    interop = int(deployment.get("torch_num_interop_threads", 1))
    if threads < 1 or interop < 1:
        raise ValueError("torch thread counts must be positive")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    housekeeping = deployment.get("housekeeping_cpus")
    if housekeeping:
        available = set(range(os.cpu_count() or 1))
        selected = set()
        for item in str(housekeeping).split(","):
            bounds = [int(value) for value in item.split("-", 1)]
            selected.update(range(bounds[0], bounds[-1] + 1 if len(bounds) == 2 else bounds[0] + 1))
        if not selected or not selected <= available:
            raise ValueError("housekeeping CPU list is not available")
        os.sched_setaffinity(0, selected)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop)


def prepare_realtime_batch(snapshot, history, task, normalizer, tokenizer, device):
    state_raw = torch.from_numpy(np.asarray(snapshot.policy_state, dtype=np.float32)).reshape(1, 10)
    torque_raw = torch.from_numpy(np.asarray(history, dtype=np.float32)).reshape(1, 10, 7)
    state = normalizer.transform(state_raw, "state")
    torque = normalizer.transform(torque_raw, "torque")
    state_text = np.clip(state.numpy(), -1, 1)
    state_tokens = np.clip(
        np.digitize(state_text, np.linspace(-1, 1, 257)[:-1]) - 1, 0, 255
    )[0]
    prompt = f"Task: {task}, State: {' '.join(map(str, state_tokens))};\nAction: "
    encoded = tokenizer(
        [prompt], max_length=200, padding="max_length", truncation=True, return_tensors="pt"
    )
    batch = {
        "observation.state": state.to(device),
        "joint_torque_history": torque.to(device),
        "observation.language.tokens": encoded["input_ids"].to(device),
        "observation.language.attention_mask": encoded["attention_mask"].to(device).bool(),
    }
    for role in ("base", "left_wrist"):
        image = np.asarray(snapshot.images[role])
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError(f"{role} image must be 224x224 RGB uint8, got {image.shape}/{image.dtype}")
        batch[f"observation.images.{role}_0_rgb"] = (
            torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
        )
    return batch


def safe_action(action, snapshot, robot_config, max_position_jump, max_rotation_jump):
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError("model action is not a finite 10-vector")
    current = np.asarray(snapshot.policy_state, dtype=np.float64)
    position_jump = float(np.linalg.norm(value[:3] - current[:3]))
    def rotation(value):
        first = value[:3]
        first = first / max(np.linalg.norm(first), 1e-8)
        second = value[3:] - np.dot(first, value[3:]) * first
        second = second / max(np.linalg.norm(second), 1e-8)
        return np.column_stack((first, second, np.cross(first, second)))

    relative = rotation(value[3:9]) @ rotation(current[3:9]).T
    rotation_jump = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    if position_jump > max_position_jump or rotation_jump > max_rotation_jump:
        raise ValueError(
            f"action safety gate rejected jump: position={position_jump:.6f}m rotation={rotation_jump:.6f}rad"
        )
    if np.any(value[:3] < np.asarray(robot_config.workspace_min)) or np.any(value[:3] > np.asarray(robot_config.workspace_max)):
        raise ValueError("action safety gate rejected workspace violation")
    value[9] = np.clip(value[9], 0.0, 1.0)
    return value.astype(np.float32)


def infer_chunk(policy, snapshot, history, task, normalizer, tokenizer, device, num_steps):
    batch = prepare_realtime_batch(snapshot, history, task, normalizer, tokenizer, device)
    images, image_masks = policy._preprocess_images(batch)
    started = time.perf_counter()
    output = policy.model.sample_actions(
        images, image_masks, batch["observation.language.tokens"],
        batch["observation.language.attention_mask"], batch["observation.state"],
        batch["joint_torque_history"], num_steps=num_steps,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = 1000 * (time.perf_counter() - started)
    action = normalizer.transform(output[0, :, :10], "action", inverse=True).cpu().numpy()
    predicted_torque = normalizer.transform(output[0, :, -7:], "torque", inverse=True).cpu().numpy()
    return action, predicted_torque, elapsed_ms


def infer_one(policy, snapshot, history, task, normalizer, tokenizer, device, num_steps):
    actions, torques, elapsed_ms = infer_chunk(
        policy, snapshot, history, task, normalizer, tokenizer, device, num_steps)
    return actions[0], torques[0], elapsed_ms


def validate_temporal_contract(checkpoint):
    run_info = json.loads((checkpoint / "run.json").read_text())
    windows = local_path(run_info["data_contract"]["windows"])
    metadata = json.loads(windows.with_name("metadata.json").read_text())
    if (tuple(metadata["history_offsets_frames"]) != HISTORY_OFFSETS
            or metadata["source_info"]["fps"] != SAMPLE_RATE_HZ
            or metadata["future_action_offsets_frames"] != list(range(50))
            or metadata["future_torque_offsets_frames"] != list(range(1, 51))):
        raise ValueError("Checkpoint temporal contract differs from the real-time controller")


def run(args):
    config_path = Path(args.robot_config).resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configure_threads(values)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    robot_config = None
    if args.mode != "check":
        robot_config = load_robot_config(str(config_path))
    checkpoint = local_path(args.checkpoint)
    validate_temporal_contract(checkpoint)
    policy, normalizer = restore_for_inference(checkpoint, device=str(device))
    tokenizer_path = local_path(json.loads((checkpoint / "force_config.json").read_text())["tokenizer"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    task = load_task(str(config_path))
    if args.mode == "check":
        print(json.dumps({"checkpoint": str(checkpoint), "task": task, "device": str(device),
                          "hardware_connected": False, "action_dim": 10, "torque_history": [10, 7],
                          "history_offsets_seconds": [offset / SAMPLE_RATE_HZ for offset in HISTORY_OFFSETS],
                          "control_hz": SAMPLE_RATE_HZ, "inference_request_hz": args.rate_hz,
                          "action_execution": "timestamped_chunks"}, indent=2))
        return 0

    if not args.allow_hardware:
        raise PermissionError("hardware modes require --allow-hardware")
    from collection_31d.robot_31d import FrankaRobotiq31DRobot

    robot = FrankaRobotiq31DRobot(robot_config)
    log_path = Path(args.log).resolve() if args.log else None
    stream = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        print(f"Connecting observation path to {robot_config.robot_ip}; no robot motion is requested.", flush=True)
        robot.connect()
        if args.startup_open:
            opened = robot.prepare_for_control()
            print(f"Startup gripper open fraction: {opened:.3f}", flush=True)
        if args.mode == "execute":
            if not sys.stdin.isatty():
                raise PermissionError("--execute requires an interactive terminal")
            if input("Type ARM to enable one-step/limited actions: ") != "ARM":
                raise PermissionError("arming cancelled")
        from .realtime_control import run_control

        run_control(args, robot, policy, normalizer, tokenizer, task, device, stream)
    except KeyboardInterrupt:
        print("Ctrl+C received; stopping robot.", flush=True)
    finally:
        try:
            if args.mode == "execute":
                robot.stop()
        finally:
            robot.disconnect()
            if stream:
                stream.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "observe", "execute"), default="check")
    parser.add_argument("--checkpoint", default="outputs/pi05_force_sft_30k/checkpoint-027000")
    parser.add_argument("--robot-config", default="configs/deployment_force_vla.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--startup-open", action="store_true")
    parser.add_argument("--async-inference", action="store_true", help="Compatibility flag; execution is always asynchronous")
    parser.add_argument("--chunk-blend-steps", type=int, default=3)
    parser.add_argument("--teleop-assist", action="store_true")
    parser.add_argument(
        "--teleop-device",
        default="/dev/input/by-id/usb-Microsoft_Controller_3039373130303635393336353232-event-joystick",
    )
    parser.add_argument("--teleop-linear-speed", type=float, default=0.02)
    parser.add_argument("--teleop-angular-speed", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--hold-after-steps", action="store_true")
    parser.add_argument("--rate-hz", type=float, default=8.0, help="Requested inference rate; control follows the 30 Hz dataset")
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--max-position-jump", type=float, default=0.08)
    parser.add_argument("--max-rotation-jump", type=float, default=0.20)
    parser.add_argument("--log")
    parser.add_argument("--observation-retry-timeout-s", type=float, default=2.0)
    parser.add_argument("--observation-retry-interval-s", type=float, default=0.02)
    args = parser.parse_args()
    if (not args.continuous and args.max_steps < 1) or args.rate_hz <= 0 or args.inference_steps < 1:
        parser.error("max-steps, rate-hz and inference-steps must be positive unless continuous mode is enabled")
    if args.continuous and args.mode != "execute":
        parser.error("continuous mode requires --mode execute")
    if (args.teleop_assist or args.startup_open) and args.mode != "execute":
        parser.error("teleop assist and startup-open require execute mode")
    if args.chunk_blend_steps < 0 or args.chunk_blend_steps >= 50:
        parser.error("chunk-blend-steps must be between 0 and 49")
    if args.teleop_linear_speed <= 0 or args.teleop_angular_speed <= 0:
        parser.error("teleop assist speeds must be positive")
    if args.observation_retry_timeout_s <= 0 or args.observation_retry_interval_s <= 0:
        parser.error("observation retry timeout and interval must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
