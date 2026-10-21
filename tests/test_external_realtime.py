from types import SimpleNamespace
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pyarrow as arrow
import torch

from force_vla.pi05.external_experiments import fit_external_statistics
from force_vla.pi05.preparation import ForceNormalizer, ROOT
from force_vla.pi05.realtime import infer_chunk, prepare_realtime_batch, run
from force_vla.pi05.realtime_control import ObservationSampler, snapshot_torque
from force_vla.pi05.temporal import HISTORY_OFFSETS, HistoryNotReady, TorqueHistory


class ExternalRealtimeTests(unittest.TestCase):
    def test_model_check_reports_effective_config_without_hardware(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            checkpoint = Path(directory)
            (checkpoint / "force_config.json").write_text(json.dumps({"use_external_torque": True, "tokenizer": "models/tokenizer"}))
            (checkpoint / "run.json").write_text(json.dumps({"data_contract": {"statistics": {
                "external_torque": {"offset": [0.] * 7, "scale": [1.] * 7}}}}))
            (checkpoint / "complete.json").write_text('{"step": 18000}')
            (checkpoint / "adapter.safetensors").write_bytes(b"test-only-weight-identity")
            args = SimpleNamespace(robot_config=str(ROOT / "configs/deployment_force_vla_smooth_test.yaml"),
                mode="check", device="cpu", checkpoint=str(checkpoint), max_position_jump=.08,
                max_rotation_jump=.35, inference_steps=10, chunk_blend_steps=3, rate_hz=8.,
                startup_open=False, continuous=False, max_steps=1)
            with patch("force_vla.pi05.realtime.configure_threads"), \
                 patch("force_vla.pi05.realtime.validate_temporal_contract"), \
                 patch("force_vla.pi05.realtime.restore_for_inference", return_value=(None, None)), \
                 patch("transformers.AutoTokenizer.from_pretrained"), \
                 patch("force_vla.pi05.realtime.load_robot_config") as hardware, \
                 patch("builtins.print") as printed:
                self.assertEqual(run(args), 0)
            hardware.assert_not_called()
            report = json.loads(printed.call_args.args[0])
            self.assertFalse(report["hardware_connected"])
            self.assertEqual(report["torque_history"], [10, 14])
            self.assertEqual(report["checkpoint_step"], 18000)
            self.assertEqual(report["robot_parameters"]["max_linear_velocity"], .05)
            self.assertEqual(len(report["adapter_sha256"]), 64)

    def test_measured_and_external_share_causal_indices_and_reset(self):
        history = TorqueHistory(torque_dim=14)
        origin = 10_000_000_000
        for frame in range(62):
            snapshot = SimpleNamespace(joint_torques=np.full(7, frame),
                                       external_joint_torques=np.full(7, 1000 + frame))
            history.append(origin + round(frame * 1e9 / 30), snapshot_torque(snapshot, True))
        values, timestamps = history.sample(origin + 2_000_000_000)
        expected = 60 + np.asarray(HISTORY_OFFSETS)
        np.testing.assert_array_equal(values[:, 0], expected)
        np.testing.assert_array_equal(values[:, 7], expected + 1000)
        self.assertTrue(np.all(timestamps <= origin + 2_000_000_000))
        self.assertEqual(timestamps[-1] - timestamps[0], 2_000_000_000)
        history.clear()
        with self.assertRaises(HistoryNotReady):
            history.sample(origin + 2_000_000_000)

    def test_live_input_equals_training_normalization_for_both_models(self):
        statistics = {name: {"offset": [offset] * size, "scale": [scale] * size}
                      for name, size, offset, scale in (("state", 10, 1, 2), ("torque", 7, 3, 4),
                                                       ("external_torque", 7, 5, 6))}
        normalizer = ForceNormalizer(statistics)
        snapshot = SimpleNamespace(policy_state=np.arange(10, dtype=np.float32),
                                   images={camera: np.zeros((224, 224, 3), dtype=np.uint8)
                                           for camera in ("base", "left_wrist")})

        def tokenizer(prompts, **kwargs):
            return {"input_ids": torch.ones(1, 3, dtype=torch.long),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long)}

        for dimension in (7, 14):
            history = np.arange(10 * dimension, dtype=np.float32).reshape(10, dimension)
            live = prepare_realtime_batch(snapshot, history, "Place lid", normalizer, tokenizer, "cpu")
            offline = normalizer({"observation.state": torch.from_numpy(snapshot.policy_state).unsqueeze(0),
                                  "joint_torque_history": torch.from_numpy(history).unsqueeze(0)})
            torch.testing.assert_close(live["joint_torque_history"], offline["joint_torque_history"])
            torch.testing.assert_close(live["observation.state"], offline["observation.state"])
            self.assertEqual(live["joint_torque_history"].shape, (1, 10, dimension))

    def test_missing_or_invalid_external_feedback_never_falls_back_to_zero(self):
        snapshot = SimpleNamespace(joint_torques=np.arange(7))
        np.testing.assert_array_equal(snapshot_torque(snapshot), np.arange(7))
        for external in (None, np.zeros(6), np.full(7, np.nan)):
            snapshot.external_joint_torques = external
            with self.assertRaisesRegex(ValueError, "synchronized"):
                snapshot_torque(snapshot, True)
        history = TorqueHistory(torque_dim=14)
        with self.assertRaises(ValueError):
            history.append(1, np.zeros(7))
        with self.assertRaises(ValueError):
            TorqueHistory(torque_dim=21)

    def test_external_checkpoint_rejects_seven_channel_history_before_inference(self):
        policy = SimpleNamespace(config=SimpleNamespace(use_external_torque=True))
        with self.assertRaisesRegex(ValueError, "14"):
            infer_chunk(policy, None, np.zeros((10, 7)), "task", None, None, "cpu", 10)

    def test_sampler_chooses_checkpoint_input_dimension(self):
        for external, expected in ((False, 7), (True, 14)):
            sampler = ObservationSampler(None, use_external_torque=external)
            self.assertEqual(sampler.history.torque_dim, expected)
            self.assertFalse(sampler.thread.is_alive())

    def test_external_statistics_exclude_validation_and_unapproved_rows(self):
        table = arrow.table({"external_joint_torque_history": [[value] * 70 for value in (2., 4., 10000.)]})
        statistics = fit_external_statistics(table, np.array([0, 1]))
        np.testing.assert_allclose(statistics["offset"], np.full(7, 3.))
        np.testing.assert_allclose(statistics["scale"], np.ones(7))
        self.assertEqual(statistics["fit_window_count"], 2)


if __name__ == "__main__":
    unittest.main()
