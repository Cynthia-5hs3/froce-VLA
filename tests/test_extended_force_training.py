import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import torch

from force_vla.pi05.evaluation import evaluate_loss, physical_metrics, select_episode_windows
from force_vla.pi05.extended_training import advance_rows, continuation_scheduler, restore_extension
from force_vla.pi05.preparation import ROOT
from force_vla.pi05.training import save_checkpoint, restore_training_state


class ExtendedTrainingTests(unittest.TestCase):
    def test_validation_includes_short_episodes_and_sorted_time(self):
        dataset = SimpleNamespace(table=pa.table({"episode_index": [4, 4, 4, 9, 12, 12],
                                                 "anchor_frame": [90, 10, 50, 0, 10, 1]}))
        selected = select_episode_windows(dataset, np.arange(6), 2)
        self.assertEqual(selected.tolist(), [1, 0, 3, 5, 4])

    def test_cursor_crosses_epoch_without_dropping_tail(self):
        indices = np.arange(7)
        progress = {"epoch": 0, "cursor": 5}
        first = np.random.default_rng(42).permutation(indices)
        second = np.random.default_rng(43).permutation(indices)
        actual = advance_rows(indices, progress, 4, 42)
        self.assertEqual(actual, first[5:].tolist() + second[:2].tolist())
        self.assertEqual(progress, {"epoch": 1, "cursor": 2})

    def test_physical_errors_and_invalid_rotation(self):
        actions = np.tile([0., 0., 0., 1., 0., 0., 0., 1., 0., 1.], (50, 1))
        torque = np.zeros((50, 7))
        predicted = actions.copy()
        predicted[:, 0] += 0.001
        metrics = physical_metrics(predicted, actions, torque + 2, torque)
        self.assertAlmostEqual(metrics["position_rmse_mm"], 1.)
        self.assertAlmostEqual(metrics["rotation_mae_deg"], 0.)
        self.assertAlmostEqual(metrics["torque_rmse_nm"], 2.)
        predicted[:, 3:9] = 0
        metrics = physical_metrics(predicted, actions, torque, torque)
        self.assertEqual(metrics["invalid_rotation_fraction"], 1.)
        self.assertIsNone(metrics["rotation_mae_deg"])

    def test_extension_preserves_moments_cursor_and_restarts_schedule(self):
        options = {"warmup_steps": 2, "learning_rate": 1e-4,
                   "warmup_start_lr": 2e-5, "minimum_lr": 1e-5}
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            policy = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .5)
            policy(torch.ones(1, 3)).sum().backward()
            optimizer.step()
            scheduler.step()
            progress = {"step": 3000, "epoch": 0, "cursor": 3000}
            original = Path(directory) / "original"
            save_checkpoint(original, policy, optimizer, scheduler, progress, {"settings": {}})
            expected_moment = optimizer.state[policy.weight]["exp_avg"].clone()
            optimizer.state.clear()
            self.assertEqual(restore_extension(policy, optimizer, original), progress)
            torch.testing.assert_close(optimizer.state[policy.weight]["exp_avg"], expected_moment)
            extended = continuation_scheduler(optimizer, options, 10)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-5)
            for update in range(3):
                optimizer.step()
                extended.step()
            resumed_checkpoint = Path(directory) / "continued"
            save_checkpoint(resumed_checkpoint, policy, optimizer, extended, progress, {"settings": {}})
            optimizer.step()
            extended.step()
            expected_lr = extended.get_last_lr()[0]
            restored_scheduler = continuation_scheduler(optimizer, options, 10)
            restore_training_state(policy, optimizer, restored_scheduler, resumed_checkpoint)
            optimizer.step()
            restored_scheduler.step()
            self.assertAlmostEqual(restored_scheduler.get_last_lr()[0], expected_lr)
            for update in range(6):
                optimizer.step()
                restored_scheduler.step()
            self.assertAlmostEqual(restored_scheduler.get_last_lr()[0], 1e-5)

    def test_validation_preserves_rng_and_training_mode(self):
        dataset = SimpleNamespace(table=pa.table({"episode_index": [1, 2], "anchor_frame": [0, 0]}))
        policy = torch.nn.Linear(1, 1).train()

        def prepare(dataset, indices, *args):
            return {"episode_index": torch.tensor([indices[0] + 1]), "anchor_frame": torch.tensor([0])}

        def forward(batch):
            value = torch.rand(())
            return value, {key: value.item() for key in ("loss", "action_loss", "torque_loss")}

        torch.manual_seed(88)
        expected = torch.rand(3)
        torch.manual_seed(88)
        with patch("force_vla.pi05.evaluation.prepare_batch", prepare), patch.object(policy, "forward", forward):
            result = evaluate_loss(policy, dataset, np.arange(2), None, None, "cpu", 1)
        torch.testing.assert_close(torch.rand(3), expected)
        self.assertTrue(policy.training)
        self.assertEqual(result["episode_count"], 2)


if __name__ == "__main__":
    unittest.main()
