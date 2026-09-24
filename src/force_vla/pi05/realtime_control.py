"""Independent observation, inference and control loops for the force policy."""

from dataclasses import replace
import json
import threading
import time

import numpy as np

from .temporal import ActionChunks, HistoryNotReady, SAMPLE_RATE_HZ, TorqueHistory


def is_retryable_observation_error(error):
    return isinstance(error, (TimeoutError, ConnectionError, OSError)) or (
        isinstance(error, RuntimeError) and str(error).startswith("RealSense")
    )


class ObservationSampler:
    def __init__(self, robot, retry_timeout_s=2.0, retry_interval_s=0.02):   ##观测线程
        if retry_timeout_s <= 0 or retry_interval_s <= 0:
            raise ValueError("observation retry timeouts must be positive")
        self.robot = robot
        self.retry_timeout_s = float(retry_timeout_s)
        self.retry_interval_s = float(retry_interval_s)
        self.history = TorqueHistory()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.snapshot = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name="force-vla-observations", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            deadline = time.monotonic()
            retry_started = None
            while not self.stop_event.is_set():
                try:
                    snapshot = self.robot.get_rollout_snapshot()
                except BaseException as error:
                    if not is_retryable_observation_error(error):
                        raise
                    now = time.monotonic()
                    if retry_started is None:
                        retry_started = now
                    if now - retry_started >= self.retry_timeout_s:
                        raise TimeoutError(
                            f"observation did not recover within {self.retry_timeout_s:.2f}s: {error}"
                        ) from error
                    self.stop_event.wait(self.retry_interval_s)
                    continue
                if hasattr(snapshot, "robot_state"):
                    from .teleop_assist import pose_matrix

                    snapshot = replace(snapshot, pose=pose_matrix(snapshot.robot_state.O_T_EE))
                with self.lock:
                    self.history.append(snapshot.observed_at_ns, snapshot.joint_torques)
                    self.snapshot = snapshot
                retry_started = None
                deadline = max(deadline + 1 / SAMPLE_RATE_HZ, time.monotonic())
                self.stop_event.wait(max(0., deadline - time.monotonic()))
        except BaseException as error:
            with self.lock:
                self.error = error
            self.stop_event.set()

    def latest(self, with_history=False):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Observation sampling failed") from self.error
            snapshot = self.snapshot
            if snapshot is None:
                raise HistoryNotReady("Waiting for the first observation")
            if time.monotonic_ns() - snapshot.observed_at_ns > 100_000_000:
                raise HistoryNotReady("Waiting for a fresh observation")
            if with_history:
                history, timestamps = self.history.sample(snapshot.observed_at_ns)
                return snapshot, history, timestamps
            return snapshot

    def reset_history(self):
        with self.lock:
            self.history.clear()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5.)
        if self.thread.is_alive():
            raise RuntimeError("Observation sampler did not stop")


class AsyncInferenceWorker:
    def __init__(self, observations, predict, rate_hz):
        self.observations = observations
        self.predict = predict
        self.period_s = 1 / rate_hz
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.sequence = 0
        self.generation = 0
        self.paused = False
        self.thread = threading.Thread(target=self._run, name="force-vla-inference", daemon=True)

    def start(self):
        self.thread.start()

    def set_paused(self, paused):
        with self.lock:
            if self.paused != paused:
                self.generation += 1
                self.latest = None
                self.paused = paused

    def reset(self):
        with self.lock:
            self.generation += 1
            self.latest = None

    def _run(self):
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                with self.lock:
                    generation, paused = self.generation, self.paused
                if paused:
                    self.stop_event.wait(.01)
                    continue
                try:
                    snapshot, history, timestamps = self.observations.latest(with_history=True)
                except HistoryNotReady:
                    with self.lock:
                        self.latest = None
                    self.stop_event.wait(.01)
                    continue
                actions, torques, inference_ms = self.predict(snapshot, history)
                with self.lock:
                    if generation == self.generation and not self.paused:
                        self.sequence += 1
                        self.latest = {
                            "sequence": self.sequence, "anchor_ns": snapshot.observed_at_ns,
                            "actions": actions, "torques": torques, "inference_ms": inference_ms,
                            "history_timestamps_ns": timestamps.tolist(),
                        }
                self.stop_event.wait(max(0., self.period_s - (time.monotonic() - started)))
        except BaseException as error:
            with self.lock:
                self.error = error
            self.stop_event.set()

    def get_latest(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Asynchronous inference failed") from self.error
            return self.latest

    def request_stop(self):
        self.stop_event.set()

    def close(self):
        self.request_stop()
        self.thread.join(timeout=10.)
        if self.thread.is_alive():
            raise RuntimeError("Asynchronous inference did not stop")


def hold_action(snapshot, gripper_target=None):
    action = np.asarray(snapshot.policy_state, dtype=np.float32).copy()
    action[9] = snapshot.gripper_open_fraction if gripper_target is None else gripper_target
    return action


def run_control(args, robot, policy, normalizer, tokenizer, task, device, stream):
    from .realtime import KeyboardStop, infer_chunk, safe_action
    from .teleop_assist import XboxAssist, assist_action_from_state

    observations = ObservationSampler(
        robot,
        retry_timeout_s=args.observation_retry_timeout_s,
        retry_interval_s=args.observation_retry_interval_s,
    )
    worker = AsyncInferenceWorker(
        observations,
        lambda snapshot, history: infer_chunk(
            policy, snapshot, history, task, normalizer, tokenizer, device, args.inference_steps),
        args.rate_hz,
    )
    chunks = ActionChunks(blend_steps=args.chunk_blend_steps)
    assist = None
    was_assisting = False
    last_action = None
    observations.start()
    worker.start()

    def keyboard_exit():
        worker.request_stop()
        if args.mode == "execute":
            robot.stop()

    try:
        if args.teleop_assist:
            assist = XboxAssist(args.teleop_device, linear_speed=args.teleop_linear_speed,
                               angular_speed=args.teleop_angular_speed)
            assist.start()
        print(f"Sampling/control: {SAMPLE_RATE_HZ:g} Hz; inference request: {args.rate_hz:g} Hz. "
              "Waiting for two seconds of real torque history.", flush=True)
        with KeyboardStop() as keyboard:
            step = 0
            deadline = time.monotonic()
            while args.continuous or step < args.max_steps:
                if keyboard.requested():
                    print("Keyboard stop requested.", flush=True)
                    keyboard_exit()
                    break
                if keyboard.restart_requested():
                    restarted_at_ns = time.monotonic_ns()
                    chunks.invalidate(restarted_at_ns)
                    worker.reset()
                    observations.reset_history()
                    if stream:
                        stream.write(json.dumps({
                            "event": "inference_restart",
                            "at_ns": restarted_at_ns,
                            "next_action": "wait_for_two_seconds_of_fresh_torque_history",
                        }) + "\n")
                        stream.flush()
                    print("Inference restarted; old action chunk discarded. Waiting for fresh torque history.", flush=True)
                    continue
                try:
                    snapshot = observations.latest()
                except HistoryNotReady:
                    chunks.invalidate(time.monotonic_ns())
                    if keyboard.wait(.01):
                        keyboard_exit()
                        break
                    continue
                now_ns = time.monotonic_ns()
                fallback = hold_action(snapshot, last_action[9] if last_action is not None else None)
                command = assist.command() if assist is not None else None
                assisting = bool(command is not None and command.active)
                if assisting != was_assisting:
                    chunks.invalidate(now_ns)
                    if not assisting:
                        observations.reset_history()
                    worker.set_paused(assisting)
                    was_assisting = assisting
                result = worker.get_latest()
                if not assisting:
                    chunks.install(result, now_ns, fallback)
                selected = chunks.sample(now_ns) if not assisting else None
                if assisting:
                    action = assist_action_from_state(robot.get_control_state(), fallback, command,
                                                     robot.config.velocity_tracking_tau_s)
                    action[:3] = np.clip(action[:3], robot.config.workspace_min, robot.config.workspace_max)
                    status = "teleop"
                elif selected is not None:
                    safe_action(selected["model_action"], snapshot, robot.config,
                                args.max_position_jump, args.max_rotation_jump)
                    action = selected["action"]
                    status = "policy"
                else:
                    action = fallback
                    status = "waiting_for_history_or_chunk"
                action = safe_action(action, snapshot, robot.config, args.max_position_jump, args.max_rotation_jump)
                command_debug = {}
                if args.mode == "execute" and (selected is not None or assisting):
                    robot.send_action(action)
                    last_action = action.copy()
                    command_debug = robot.command_debug_snapshot()
                model_action = selected["model_action"] if selected is not None else None
                record = {
                    "step": step, "mode": args.mode, "status": status, "control_at_ns": now_ns,
                    "inference_sequence": selected["sequence"] if selected else None,
                    "action_index": selected["action_index"] if selected else None,
                    "chunk_age_ms": (now_ns - selected["anchor_ns"]) / 1e6 if selected else None,
                    "skipped_steps_on_arrival": selected["skipped_steps_on_arrival"] if selected else None,
                    "history_timestamps_ns": selected["history_timestamps_ns"] if selected else None,
                    "inference_ms": selected["inference_ms"] if selected else None,
                    "model_xyz": model_action[:3].tolist() if model_action is not None else None,
                    "model_gripper": float(model_action[9]) if model_action is not None else None,
                    "predicted_tau_J": selected["predicted_torque"].tolist() if selected else None,
                    "measured_tau_J": snapshot.joint_torques.tolist(),
                    "measured_gripper_open_fraction": float(snapshot.gripper_open_fraction),
                    "action": action.tolist(), "sent_xyz": action[:3].tolist(),
                    "sent_gripper": float(action[9]), "camera_timestamps": snapshot.image_timestamps,
                    "motion_sent": args.mode == "execute" and bool(command_debug.get("motion_command_sent", False)),
                    "teleop_active": assisting, "command_debug": command_debug,
                    "motion_events": robot.drain_motion_events() if hasattr(robot, "drain_motion_events") else [],
                }
                if stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                if selected is not None or assisting:
                    print(json.dumps({key: record[key] for key in (
                        "step", "status", "inference_sequence", "action_index", "chunk_age_ms",
                        "model_xyz", "sent_xyz", "sent_gripper", "motion_sent")}), flush=True)
                    step += 1
                deadline = max(deadline + 1 / SAMPLE_RATE_HZ, time.monotonic())
                if keyboard.wait(max(0., deadline - time.monotonic())):
                    keyboard_exit()
                    break
            if args.hold_after_steps and step >= args.max_steps and not args.continuous:
                print("Step budget reached; q, x or Esc exits.", flush=True)
                while not keyboard.wait(.1):
                    pass
                keyboard_exit()
    except KeyboardInterrupt:
        keyboard_exit()
        raise
    finally:
        worker.request_stop()
        try:
            if assist is not None:
                assist.close()
        finally:
            try:
                worker.close()
            finally:
                observations.close()
