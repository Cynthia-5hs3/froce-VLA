"""Force-PI05 live observation and explicitly armed single-arm inference."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import torch
import yaml

from .preparation import local_path
from .teleop_assist import XboxAssist, assist_action_from_state
from .training import restore_for_inference


ROOT = Path(__file__).resolve().parents[3]
TASK = "Place the white lid firmly onto the jig."
ABORT_KEYS = frozenset(("q", "x", "\x1b"))


def is_abort_key(key: str) -> bool:
    return key in ABORT_KEYS


class KeyboardStop:
    def __init__(self):
        self._fd = None
        self._attributes = None

    def __enter__(self):
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._attributes = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            print("Press q, x, or Esc to stop the loop.", flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._fd is not None and self._attributes is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._attributes)

    def requested(self) -> bool:
        if self._fd is None:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(readable and is_abort_key(sys.stdin.read(1)))

    def wait(self, duration_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline:
            if self.requested():
                return True
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return False


class AsyncInferenceWorker:
    def __init__(self, robot, policy, task, normalizer, tokenizer, device, inference_steps,
                 max_position_jump, max_rotation_jump, rate_hz):
        self.robot = robot
        self.policy = policy
        self.task = task
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.device = device
        self.inference_steps = inference_steps
        self.max_position_jump = max_position_jump
        self.max_rotation_jump = max_rotation_jump
        self.period_s = 1.0 / rate_hz
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.sequence = 0
        self.thread = threading.Thread(target=self._run, name="force-vla-inference", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        history = deque(maxlen=10)
        try:
            snapshot = self.robot.get_rollout_snapshot()
            for _ in range(10):
                history.append(snapshot.joint_torques.copy())
            next_deadline = time.monotonic()
            while not self.stop_event.is_set():
                if self.stop_event.wait(max(0.0, next_deadline - time.monotonic())):
                    return
                snapshot = self.robot.get_rollout_snapshot()
                history.append(snapshot.joint_torques.copy())
                action, predicted_torque, inference_ms = infer_one(
                    self.policy, snapshot, np.stack(history), self.task,
                    self.normalizer, self.tokenizer, self.device, self.inference_steps,
                )
                safe = safe_action(
                    action, snapshot, self.robot.config,
                    self.max_position_jump, self.max_rotation_jump,
                )
                with self.lock:
                    self.sequence += 1
                    self.latest = {
                        "sequence": self.sequence,
                        "snapshot": snapshot,
                        "action": safe,
                        "predicted_torque": predicted_torque,
                        "inference_ms": inference_ms,
                    }
                next_deadline = max(next_deadline + self.period_s, time.monotonic())
        except BaseException as error:
            with self.lock:
                self.error = error
            self.stop_event.set()

    def get_latest(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("asynchronous inference failed") from self.error
            return self.latest

    def close(self):
        self.request_stop()
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            raise RuntimeError("asynchronous inference worker did not stop")

    def request_stop(self):
        self.stop_event.set()


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


def infer_one(policy, snapshot, history, task, normalizer, tokenizer, device, num_steps):
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
    action = normalizer.transform(output[0, 0, :10], "action", inverse=True).cpu().numpy()
    predicted_torque = normalizer.transform(output[0, 0, -7:], "torque", inverse=True).cpu().numpy()
    return action, predicted_torque, elapsed_ms


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
    policy, normalizer = restore_for_inference(checkpoint, device=str(device))
    tokenizer_path = local_path(json.loads((checkpoint / "force_config.json").read_text())["tokenizer"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    task = load_task(str(config_path))
    if args.mode == "check":
        print(json.dumps({"checkpoint": str(checkpoint), "task": task, "device": str(device),
                          "hardware_connected": False, "action_dim": 10, "torque_history": [10, 7]}, indent=2))
        return 0

    if not args.allow_hardware:
        raise PermissionError("hardware modes require --allow-hardware")
    from collection_31d.robot_31d import FrankaRobotiq31DRobot

    robot = FrankaRobotiq31DRobot(robot_config)
    history = deque(maxlen=10)
    log_path = Path(args.log).resolve() if args.log else None
    stream = log_path.open("a", encoding="utf-8") if log_path else None
    stopped_by_user = False
    try:
        print(f"Connecting observation path to {robot_config.robot_ip}; no robot motion is requested.", flush=True)
        robot.connect()
        if args.startup_open:
            opened = robot.prepare_for_control()
            print(f"Startup gripper open fraction: {opened:.3f}", flush=True)
        snapshot = robot.get_rollout_snapshot()
        for _ in range(10):
            history.append(snapshot.joint_torques.copy())
        if args.mode == "execute":
            if not sys.stdin.isatty():
                raise PermissionError("--execute requires an interactive terminal")
            if input("Type ARM to enable one-step/limited actions: ") != "ARM":
                raise PermissionError("arming cancelled")
        if args.async_inference:
            worker = AsyncInferenceWorker(
                robot, policy, task, normalizer, tokenizer, device, args.inference_steps,
                args.max_position_jump, args.max_rotation_jump, args.rate_hz,
            )
            worker.start()
            assist = None
            if args.teleop_assist:
                assist = XboxAssist(
                    args.teleop_device,
                    linear_speed=args.teleop_linear_speed,
                    angular_speed=args.teleop_angular_speed,
                )
                try:
                    assist.start()
                except BaseException:
                    worker.close()
                    raise
                print("Xbox assist enabled: sticks/buttons override the current model target while held.", flush=True)
            try:
                with KeyboardStop() as keyboard:
                    step = 0
                    while args.continuous or step < args.max_steps:
                        if keyboard.requested():
                            print("Keyboard stop requested; stopping worker and robot.", flush=True)
                            stopped_by_user = True
                            break
                        result = worker.get_latest()
                        if result is None:
                            if keyboard.wait(0.01):
                                print("Keyboard stop requested; stopping worker and robot.", flush=True)
                                stopped_by_user = True
                                break
                            continue
                        assist_command = assist.command() if assist is not None else None
                        model_action = np.asarray(result["action"], dtype=np.float32).copy()
                        action = model_action.copy()
                        if assist_command is not None and assist_command.active:
                            state = robot.get_control_state()
                            action = assist_action_from_state(
                                state, action, assist_command, 1.0 / args.rate_hz,
                            )
                            action[:3] = np.clip(
                                action[:3], np.asarray(robot_config.workspace_min),
                                np.asarray(robot_config.workspace_max),
                            )
                        robot.send_action(action)
                        snapshot = result["snapshot"]
                        command_debug = robot.command_debug_snapshot()
                        motion_events = (
                            robot.drain_motion_events()
                            if hasattr(robot, "drain_motion_events") else []
                        )
                        record = {
                            "step": step,
                            "inference_sequence": result["sequence"],
                            "mode": args.mode,
                            "inference_ms": result["inference_ms"],
                            "measured_tau_J": snapshot.joint_torques.tolist(),
                            "measured_gripper_open_fraction": float(snapshot.gripper_open_fraction),
                            "predicted_tau_J": result["predicted_torque"].tolist(),
                            "model_action": model_action.tolist(),
                            "model_xyz": model_action[:3].tolist(),
                            "model_gripper": float(model_action[9]),
                            "action": action.tolist(),
                            "sent_xyz": action[:3].tolist(),
                            "sent_gripper": float(action[9]),
                            "camera_timestamps": snapshot.image_timestamps,
                            "motion_sent": bool(command_debug.get("motion_command_sent", True)),
                            "command_debug": command_debug,
                            "motion_events": motion_events,
                            "teleop_active": bool(assist_command is not None and assist_command.active),
                        }
                        if assist_command is not None:
                            record.update({
                                "teleop_linear_velocity": assist_command.linear_velocity.tolist(),
                                "teleop_angular_velocity": assist_command.angular_velocity.tolist(),
                                "teleop_gripper_target": assist_command.gripper_target,
                            })
                        if stream:
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            stream.flush()
                        print(json.dumps({key: record[key] for key in (
                            "step", "inference_sequence", "inference_ms", "model_xyz",
                            "model_gripper", "sent_xyz", "sent_gripper", "motion_sent",
                            "teleop_active",
                        )}), flush=True)
                        step += 1
                        if keyboard.wait(1.0 / args.rate_hz):
                            print("Keyboard stop requested; stopping worker and robot.", flush=True)
                            stopped_by_user = True
                            break
            finally:
                worker.request_stop()
                if assist is not None:
                    assist.close()
                try:
                    robot.stop()
                finally:
                    worker.close()
            return 0
        with KeyboardStop() as keyboard:
            step = 0
            while args.continuous or step < args.max_steps:
                if keyboard.requested():
                    print("Keyboard stop requested; stopping robot.", flush=True)
                    robot.stop()
                    stopped_by_user = True
                    break
                if step and not args.continuous:
                    time.sleep(max(0.0, 1.0 / args.rate_hz))
                snapshot = robot.get_rollout_snapshot()
                history.append(snapshot.joint_torques.copy())
                action, predicted_torque, inference_ms = infer_one(
                    policy, snapshot, np.stack(history), task, normalizer, tokenizer, device, args.inference_steps
                )
                if keyboard.requested():
                    print("Keyboard stop requested; stopping robot.", flush=True)
                    robot.stop()
                    stopped_by_user = True
                    break
                safe = safe_action(action, snapshot, robot_config, args.max_position_jump, args.max_rotation_jump)
                record = {"step": step, "mode": args.mode, "inference_ms": inference_ms,
                          "measured_tau_J": snapshot.joint_torques.tolist(),
                          "measured_gripper_open_fraction": float(snapshot.gripper_open_fraction),
                          "predicted_tau_J": predicted_torque.tolist(), "action": safe.tolist(),
                          "model_action": np.asarray(action, dtype=np.float32).tolist(),
                          "model_xyz": np.asarray(action, dtype=np.float32)[:3].tolist(),
                          "model_gripper": float(action[9]),
                          "sent_xyz": safe[:3].tolist(),
                          "sent_gripper": float(safe[9]),
                          "camera_timestamps": snapshot.image_timestamps,
                          "motion_sent": False}
                if args.mode == "execute":
                    robot.send_action(safe)
                    record["command_debug"] = robot.command_debug_snapshot()
                    record["motion_events"] = (
                        robot.drain_motion_events()
                        if hasattr(robot, "drain_motion_events") else []
                    )
                    record["motion_sent"] = bool(
                        record["command_debug"].get("motion_command_sent", True)
                    )
                    motion_duration_s = float(robot_config.velocity_motion_duration_ms) / 1000.0
                    if keyboard.wait(max(0.15, motion_duration_s + 0.05)):
                        print("Keyboard stop requested; stopping robot.", flush=True)
                        robot.stop()
                        stopped_by_user = True
                        if stream:
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            stream.flush()
                        break
                if stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                print(json.dumps({key: record[key] for key in (
                    "step", "inference_ms", "model_xyz", "model_gripper",
                    "sent_xyz", "sent_gripper", "motion_sent",
                )}), flush=True)
                step += 1
            if args.hold_after_steps and args.mode == "execute" and not stopped_by_user:
                print("Step budget reached; press q, x, or Esc to stop the robot.", flush=True)
                while not keyboard.wait(0.25):
                    pass
                robot.stop()
                stopped_by_user = True
    except KeyboardInterrupt:
        print("Ctrl+C received; stopping robot.", flush=True)
        robot.stop()
        stopped_by_user = True
    finally:
        try:
            if not stopped_by_user:
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
    parser.add_argument("--async-inference", action="store_true")
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
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--max-position-jump", type=float, default=0.03)
    parser.add_argument("--max-rotation-jump", type=float, default=0.20)
    parser.add_argument("--log")
    args = parser.parse_args()
    if (not args.continuous and args.max_steps < 1) or args.rate_hz <= 0 or args.inference_steps < 1:
        parser.error("max-steps, rate-hz and inference-steps must be positive unless continuous mode is enabled")
    if args.continuous and args.mode != "execute":
        parser.error("continuous mode requires --mode execute")
    if args.teleop_assist and not args.async_inference:
        parser.error("teleop assist requires --async-inference")
    if args.teleop_linear_speed <= 0 or args.teleop_angular_speed <= 0:
        parser.error("teleop assist speeds must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
