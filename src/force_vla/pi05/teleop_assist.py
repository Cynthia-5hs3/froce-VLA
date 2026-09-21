"""Direct Xbox gamepad assistance for force-VLA inference."""

from __future__ import annotations

from dataclasses import dataclass
import select
import threading
from typing import Any

import numpy as np

from evo_rlt.robots.franka_robotiq.robot import _axis_angle_matrix, matrix_to_rot6d, pose_matrix


@dataclass(frozen=True)
class AssistCommand:
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    gripper_target: float | None
    active: bool


def _deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    magnitude = min(1.0, (abs(value) - deadzone) / (1.0 - deadzone))
    return float(np.copysign(magnitude, value))


def assist_action_from_state(state: Any, model_action: np.ndarray, command: AssistCommand,
                             horizon_s: float) -> np.ndarray:
    if not command.active:
        return np.asarray(model_action, dtype=np.float32).copy()
    if horizon_s <= 0 or not np.isfinite(horizon_s):
        raise ValueError("assist horizon must be finite and positive")
    action = np.asarray(model_action, dtype=np.float64).reshape(-1).copy()
    if action.shape != (10,) or not np.isfinite(action).all():
        raise ValueError("model action must be a finite 10-vector")
    pose = pose_matrix(state.O_T_EE)
    pose[:3, 3] += command.linear_velocity * horizon_s
    angular = np.asarray(command.angular_velocity, dtype=np.float64)
    angle = float(np.linalg.norm(angular) * horizon_s)
    if angle > 1e-10:
        pose[:3, :3] = _axis_angle_matrix(angular / np.linalg.norm(angular), angle) @ pose[:3, :3]
    action[:9] = np.concatenate((pose[:3, 3], matrix_to_rot6d(pose[:3, :3])))
    if command.gripper_target is not None:
        action[9] = command.gripper_target
    return action.astype(np.float32)


class XboxAssist:
    AXIS_X = 0
    AXIS_Y = 1
    AXIS_LT = 2
    AXIS_RX = 3
    AXIS_RY = 4
    AXIS_RT = 5
    BUTTON_A = 304
    BUTTON_B = 305
    BUTTON_X = 307
    BUTTON_Y = 308

    def __init__(self, device_path: str, *, deadzone: float = 0.08,
                 linear_speed: float = 0.02, angular_speed: float = 0.15):
        if not 0 < deadzone < 1 or linear_speed <= 0 or angular_speed <= 0:
            raise ValueError("invalid Xbox assist limits")
        self.device_path = device_path
        self.deadzone = deadzone
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self._lock = threading.Lock()
        self._axes = {code: 0.0 for code in range(6)}
        self._buttons: set[int] = set()
        self._stop = threading.Event()
        self._device = None
        self._thread = None

    def start(self):
        from evdev import InputDevice

        self._device = InputDevice(self.device_path)
        self._device.grab()
        self._thread = threading.Thread(target=self._read_loop, name="force-vla-xbox-assist", daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([self._device.fd], [], [], 0.1)
                if not readable:
                    continue
                for event in self._device.read():
                    with self._lock:
                        if event.type == 3 and event.code in self._axes:
                            info = self._device.absinfo(event.code)
                            span = max(1, info.max - info.min)
                            normalized = 2.0 * (event.value - info.min) / span - 1.0
                            self._axes[event.code] = float(np.clip(normalized, -1.0, 1.0))
                        elif event.type == 1 and event.code in {
                            self.BUTTON_A, self.BUTTON_B, self.BUTTON_X, self.BUTTON_Y,
                        }:
                            if event.value:
                                self._buttons.add(event.code)
                            else:
                                self._buttons.discard(event.code)
        except (OSError, AttributeError):
            if not self._stop.is_set():
                self._stop.set()

    def command(self) -> AssistCommand:
        with self._lock:
            axes = dict(self._axes)
            buttons = set(self._buttons)
        left_x = _deadzone(axes[self.AXIS_X], self.deadzone)
        left_y = _deadzone(axes[self.AXIS_Y], self.deadzone)
        right_x = _deadzone(axes[self.AXIS_RX], self.deadzone)
        right_y = _deadzone(axes[self.AXIS_RY], self.deadzone)
        linear = np.array([
            left_y * self.linear_speed,
            -left_x * self.linear_speed,
            (float(self.BUTTON_Y in buttons) - float(self.BUTTON_A in buttons)) * self.linear_speed,
        ], dtype=np.float64)
        angular = np.array([
            right_y * self.angular_speed,
            -right_x * self.angular_speed,
            (float(self.BUTTON_X in buttons) - float(self.BUTTON_B in buttons)) * self.angular_speed,
        ], dtype=np.float64)
        lt = max(0.0, (axes[self.AXIS_LT] + 1.0) * 0.5)
        rt = max(0.0, (axes[self.AXIS_RT] + 1.0) * 0.5)
        gripper_target = 1.0 if lt - rt > self.deadzone else 0.0 if rt - lt > self.deadzone else None
        active = bool(np.linalg.norm(linear) > 0 or np.linalg.norm(angular) > 0 or gripper_target is not None)
        return AssistCommand(linear, angular, gripper_target, active)

    def close(self):
        self._stop.set()
        if self._device is not None:
            try:
                self._device.ungrab()
            except OSError:
                pass
            self._device.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._device = None
        self._thread = None
