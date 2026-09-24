import unittest
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from force_vla.pi05.realtime import is_abort_key, is_restart_key, safe_action
from force_vla.pi05.realtime_control import AsyncInferenceWorker, ObservationSampler, hold_action, run_control


class RealtimeSafetyTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = SimpleNamespace(policy_state=np.array(
            [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5], dtype=np.float32))
        self.config = SimpleNamespace(workspace_min=(0.1, -0.5, 0.05), workspace_max=(0.8, 0.5, 0.75))

    def action(self):
        return np.array([0.401, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.4], dtype=np.float32)

    def test_clamps_only_gripper_after_pose_gate(self):
        value = safe_action(self.action(), self.snapshot, self.config, 0.03, 0.2)
        self.assertEqual(value[9], 1.0)
        np.testing.assert_allclose(value[:9], self.action()[:9])

    def test_rejects_large_position_jump(self):
        value = self.action()
        value[0] = 0.5
        with self.assertRaisesRegex(ValueError, "position"):
            safe_action(value, self.snapshot, self.config, 0.03, 0.2)

    def test_rejects_workspace_violation(self):
        value = self.action()
        value[2] = 0.04
        with self.assertRaisesRegex(ValueError, "workspace"):
            safe_action(value, self.snapshot, self.config, 0.30, 0.2)

    def test_keyboard_abort_keys(self):
        self.assertTrue(is_abort_key("q"))
        self.assertTrue(is_abort_key("x"))
        self.assertTrue(is_abort_key("\x1b"))
        self.assertFalse(is_abort_key("a"))
        self.assertTrue(is_restart_key("r"))
        self.assertFalse(is_restart_key("q"))

    def test_observe_consumes_chunks_without_sending_any_motion(self):
        self.snapshot.joint_torques = np.zeros(7)
        self.snapshot.gripper_open_fraction = 1.
        self.snapshot.image_timestamps = {"base": 1., "left_wrist": 1.}
        self.snapshot.observed_at_ns = time.monotonic_ns()
        actions = np.tile(self.action(), (50, 1))
        result = {"sequence": 1, "anchor_ns": self.snapshot.observed_at_ns - 180_000_000,
                  "actions": actions, "torques": np.zeros((50, 7)), "inference_ms": 180.,
                  "history_timestamps_ns": [1] * 10}
        args = SimpleNamespace(mode="observe", rate_hz=8., chunk_blend_steps=3, teleop_assist=False,
                               continuous=False, max_steps=1, max_position_jump=.03,
                               max_rotation_jump=.2, hold_after_steps=False, inference_steps=10,
                               observation_retry_timeout_s=2., observation_retry_interval_s=.02)
        robot = MagicMock(config=self.config)
        robot.drain_motion_events.return_value = []
        stream = MagicMock()
        with patch("force_vla.pi05.realtime_control.ObservationSampler") as sampler, \
             patch("force_vla.pi05.realtime_control.AsyncInferenceWorker") as worker, \
             patch("force_vla.pi05.realtime.KeyboardStop") as keyboard:
            sampler.return_value.latest.return_value = self.snapshot
            worker.return_value.get_latest.return_value = result
            keyboard.return_value.__enter__.return_value.requested.return_value = False
            keyboard.return_value.__enter__.return_value.restart_requested.return_value = False
            keyboard.return_value.__enter__.return_value.wait.return_value = False
            run_control(args, robot, None, None, None, "task", None, stream)
        robot.send_action.assert_not_called()
        robot.stop.assert_not_called()
        record = json.loads(stream.write.call_args[0][0])
        self.assertGreaterEqual(record["action_index"], 5)
        self.assertFalse(record["motion_sent"])

    def test_teleop_pause_discards_prediction_already_in_flight(self):
        entered, release = threading.Event(), threading.Event()
        observations = MagicMock()
        observations.latest.return_value = (SimpleNamespace(observed_at_ns=1), np.zeros((10, 7)), np.arange(10))

        def predict(snapshot, history):
            entered.set()
            if not release.wait(2.):
                raise RuntimeError("test inference was not released")
            return np.zeros((50, 10)), np.zeros((50, 7)), 1.

        worker = AsyncInferenceWorker(observations, predict, 8.)
        worker.start()
        try:
            self.assertTrue(entered.wait(2.))
            worker.set_paused(True)
            release.set()
        finally:
            release.set()
            worker.close()
        self.assertIsNone(worker.get_latest())
        self.assertEqual(worker.sequence, 0)

    def test_restart_discards_prediction_already_in_flight(self):
        entered, release = threading.Event(), threading.Event()
        observations = MagicMock()
        observations.latest.return_value = (SimpleNamespace(observed_at_ns=1), np.zeros((10, 7)), np.arange(10))

        def predict(snapshot, history):
            entered.set()
            if not release.wait(2.):
                raise RuntimeError("test inference was not released")
            return np.zeros((50, 10)), np.zeros((50, 7)), 1.

        worker = AsyncInferenceWorker(observations, predict, 8.)
        worker.start()
        try:
            self.assertTrue(entered.wait(2.))
            worker.reset()
            release.set()
        finally:
            release.set()
            worker.close()
        self.assertIsNone(worker.get_latest())
        self.assertEqual(worker.sequence, 0)

    def test_assist_uses_tracking_horizon_after_control_rate_change(self):
        self.config.velocity_tracking_tau_s = .15
        self.snapshot.joint_torques = np.zeros(7)
        self.snapshot.gripper_open_fraction = 1.
        self.snapshot.image_timestamps = {"base": 1., "left_wrist": 1.}
        args = SimpleNamespace(mode="execute", rate_hz=8., chunk_blend_steps=3, teleop_assist=True,
                               continuous=False, max_steps=1, max_position_jump=.03,
                               max_rotation_jump=.2, hold_after_steps=False, inference_steps=10,
                               teleop_device="mock", teleop_linear_speed=.02, teleop_angular_speed=.15,
                               observation_retry_timeout_s=2., observation_retry_interval_s=.02)
        robot = MagicMock(config=self.config)
        robot.drain_motion_events.return_value = []
        with patch("force_vla.pi05.realtime_control.ObservationSampler") as sampler, \
             patch("force_vla.pi05.realtime_control.AsyncInferenceWorker") as worker, \
             patch("force_vla.pi05.realtime.KeyboardStop") as keyboard, \
             patch("force_vla.pi05.teleop_assist.XboxAssist") as assist, \
             patch("force_vla.pi05.teleop_assist.assist_action_from_state", return_value=self.action()) as convert:
            sampler.return_value.latest.return_value = self.snapshot
            worker.return_value.get_latest.return_value = None
            assist.return_value.command.return_value = SimpleNamespace(active=True)
            keyboard.return_value.__enter__.return_value.requested.return_value = False
            keyboard.return_value.__enter__.return_value.restart_requested.return_value = False
            keyboard.return_value.__enter__.return_value.wait.return_value = False
            run_control(args, robot, None, None, None, "task", None, None)
            self.assertEqual(convert.call_args[0][-1], .15)
            worker.return_value.set_paused.assert_called_with(True)
        robot.send_action.assert_called_once()

    def test_hold_preserves_gripper_intent_during_partial_opening(self):
        self.snapshot.gripper_open_fraction = .5
        action = hold_action(self.snapshot, gripper_target=1.)
        self.assertEqual(action[9], 1.)
        np.testing.assert_array_equal(action[:9], self.snapshot.policy_state[:9])

    def test_warmup_sends_no_actions_and_keyboard_stop_precedes_thread_join(self):
        self.snapshot.joint_torques = np.zeros(7)
        self.snapshot.gripper_open_fraction = .5
        self.snapshot.image_timestamps = {"base": 1., "left_wrist": 1.}
        args = SimpleNamespace(mode="execute", rate_hz=8., chunk_blend_steps=3, teleop_assist=False,
                               continuous=False, max_steps=1, max_position_jump=.03,
                               max_rotation_jump=.2, hold_after_steps=False, inference_steps=10,
                               observation_retry_timeout_s=2., observation_retry_interval_s=.02)
        robot = MagicMock(config=self.config)
        robot.drain_motion_events.return_value = []
        events = []
        robot.stop.side_effect = lambda: events.append("stop")
        with patch("force_vla.pi05.realtime_control.ObservationSampler") as sampler, \
             patch("force_vla.pi05.realtime_control.AsyncInferenceWorker") as worker, \
             patch("force_vla.pi05.realtime.KeyboardStop") as keyboard:
            sampler.return_value.latest.return_value = self.snapshot
            worker.return_value.get_latest.return_value = None
            worker.return_value.close.side_effect = lambda: events.append("join")
            keyboard.return_value.__enter__.return_value.requested.return_value = False
            keyboard.return_value.__enter__.return_value.restart_requested.return_value = False
            keyboard.return_value.__enter__.return_value.wait.return_value = True
            run_control(args, robot, None, None, None, "task", None, None)
        robot.send_action.assert_not_called()
        self.assertEqual(events, ["stop", "join"])

    def test_observation_sampler_retries_transient_timeout(self):
        snapshot = SimpleNamespace(observed_at_ns=time.monotonic_ns(), joint_torques=np.zeros(7))
        calls = {"count": 0}

        class FlakyRobot:
            def get_rollout_snapshot(self):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise TimeoutError("camera frame is stale")
                snapshot.observed_at_ns = time.monotonic_ns()
                return snapshot

        sampler = ObservationSampler(FlakyRobot(), retry_timeout_s=.2, retry_interval_s=.001)
        sampler.start()
        try:
            deadline = time.monotonic() + 1.
            while calls["count"] < 2 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertGreaterEqual(calls["count"], 2)
            self.assertIs(sampler.latest(), snapshot)
        finally:
            sampler.close()

    def test_observation_sampler_fails_after_retry_timeout(self):
        class BrokenRobot:
            def get_rollout_snapshot(self):
                raise TimeoutError("no synchronized RealSense frame pair")

        sampler = ObservationSampler(BrokenRobot(), retry_timeout_s=.03, retry_interval_s=.001)
        sampler.start()
        try:
            deadline = time.monotonic() + 1.
            while sampler.error is None and time.monotonic() < deadline:
                time.sleep(.005)
            with self.assertRaisesRegex(RuntimeError, "Observation sampling failed") as context:
                sampler.latest()
            self.assertIsInstance(context.exception.__cause__, TimeoutError)
        finally:
            sampler.close()


if __name__ == "__main__":
    unittest.main()
