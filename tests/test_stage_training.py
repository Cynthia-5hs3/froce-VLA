import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch
from safetensors.torch import save_file

from force_vla.pi05.phases import training_epoch_order
from force_vla.pi05.preparation import ForceNormalizer, ROOT, digest
from force_vla.pi05.stage_sampling import (
    expand_history_weight, initialize_stage_checkpoint, load_sampling_plan, one_pass_candidate, sampling_probabilities,
)
from force_vla.pi05.window_dataset import ForceVLAWindowDataset, collate_force_windows, validate_force_window_batch
from force_vla.pi05 import modeling_pi05_base as base
from force_vla.pi05.modeling_pi05_force import ForcePI05Policy
from force_vla.pi05.train_pi05_force import load_force_config
from force_vla.pi05.training import configure_trainable
from test_force_pi05_training import tiny_init


class StageTrainingTests(unittest.TestCase):
    def test_initialize_copies_old_adapter_and_zeroes_only_new_columns(self):
        settings = json.loads((ROOT / "configs/pi05_force_tavla_aligned.json").read_text())
        config = load_force_config(ROOT / "configs/pi05_force_tavla_aligned.json")
        config.gradient_checkpointing = False
        with patch.object(base.PI05Pytorch, "__init__", tiny_init):
            original = ForcePI05Policy(config)
        configure_trainable(original, "adapters")
        weights = {name: value.detach().clone() for name, value in original.named_parameters() if value.requires_grad}
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            checkpoint = Path(directory)
            statistics = {name: {"offset": [0.], "scale": [1.]} for name in ("state", "action", "torque")}
            contract = {"statistics": statistics}
            (checkpoint / "complete.json").write_text('{"step": 1}')
            (checkpoint / "force_config.json").write_text(json.dumps(settings))
            (checkpoint / "run.json").write_text(json.dumps({"base_sha256": "test-base", "data_contract": contract}))
            save_file(weights, str(checkpoint / "adapter.safetensors"))
            config.use_external_torque = True
            with patch.object(base.PI05Pytorch, "__init__", tiny_init):
                expanded = ForcePI05Policy(config)
            configure_trainable(expanded, "adapters")
            report = initialize_stage_checkpoint(expanded, checkpoint, contract, "test-base")
            self.assertFalse(report["optimizer_restored"])
            self.assertEqual(report["expanded_zero_external_columns"], ["model.torque_adapter.projection.0.weight"])
            measured = torch.randn(2, 10, 7)
            combined = torch.cat((measured, torch.randn_like(measured)), -1)
            torch.testing.assert_close(original.model.torque_adapter(measured), expanded.model.torque_adapter(combined))
            with self.assertRaisesRegex(ValueError, "base checkpoint"):
                initialize_stage_checkpoint(expanded, checkpoint, contract, "different-base")

    def test_one_pass_requires_confirmed_order_and_full_coverage(self):
        def segment(start, stage):
            return {"start_frame": start, "end_frame": start + 50, "stage": stage,
                    "confidence": "confirmed", "imitation": "keep"}
        document = {"source": {"frame_count": 200}, "segments": [segment(0, "approach"),
                    segment(50, "aligned_press"), segment(100, "seated_hold"), segment(150, "release_retreat")]}
        self.assertTrue(one_pass_candidate(document))
        document["segments"][-1]["imitation"] = "review"
        self.assertFalse(one_pass_candidate(document))
        document["segments"][-1]["imitation"] = "keep"
        document["segments"][1]["stage"] = "correction"
        self.assertFalse(one_pass_candidate(document))
        document["segments"][1]["stage"] = "aligned_press"
        document["segments"][0]["start_frame"] = 1
        self.assertFalse(one_pass_candidate(document))

    def test_group_episode_and_stage_weights_with_reproducible_resume_order(self):
        rows = [{"episode_index": episode, "anchor_stage": stage, "suggested_sampling_weight": weight}
                for episode, stage, weight in [(15, "approach", 1), (15, "aligned_press", 1),
                    (25, "aligned_press", 1), (25, "aligned_press", .1), (21, "correction", 1)]]
        probabilities = sampling_probabilities(rows, [15, 25], .25)
        self.assertAlmostEqual(probabilities[:4].sum(), .25)
        self.assertAlmostEqual(probabilities[:2].sum(), .125)
        self.assertAlmostEqual(probabilities[2:4].sum(), .125)
        self.assertAlmostEqual(probabilities[1] / probabilities[0], 3.)
        self.assertAlmostEqual(probabilities[2] / probabilities[3], 10.)
        indices = np.arange(5) + 100
        first = training_epoch_order(indices, 42, probabilities=probabilities)
        np.testing.assert_array_equal(first, training_epoch_order(indices, 42, probabilities=probabilities))
        self.assertTrue(np.isin(first, indices).all())
        with self.assertRaises(ValueError):
            training_epoch_order(indices, 42, critical_fraction=.25, probabilities=probabilities)

    def test_plan_rejects_validation_rows_and_changed_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            windows = Path(directory) / "windows.parquet"
            table = arrow.table({"episode_index": [15, 21, 39]})
            parquet.write_table(table, windows)
            plan = Path(directory) / "sampling.parquet"
            parquet.write_table(arrow.table({"source_window_row": [0, 2], "probability": [.5, .5]}), plan)
            contract = {"train_episodes": [15, 21], "sha256": digest(windows),
                        "sampling_plan": {"path": str(plan), "sha256": digest(plan)}}
            dataset = SimpleNamespace(table=table, parquet_path=windows)
            with self.assertRaisesRegex(ValueError, "non-training"):
                load_sampling_plan(dataset, contract)
            parquet.write_table(arrow.table({"source_window_row": [0, 1], "probability": [.5, .5]}), plan)
            with self.assertRaisesRegex(ValueError, "changed"):
                load_sampling_plan(dataset, contract)
            contract["sampling_plan"]["sha256"] = digest(plan)
            rows, probabilities = load_sampling_plan(dataset, contract)
            np.testing.assert_array_equal(rows, [0, 1])
            np.testing.assert_allclose(probabilities, [.5, .5])

    def test_external_loader_normalization_and_projection_preserve_measured_path(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            path = Path(directory) / "windows.parquet"
            parquet.write_table(arrow.Table.from_pylist([{
                "state": [0.] * 10, "joint_torque_history": [5.] * 70,
                "external_joint_torque_history": [11.] * 70, "future_joint_torque": [5.] * 350,
                "future_actions": [0.] * 500, "task_text": "Place lid", "episode_index": 15, "anchor_frame": 60}]), path)
            dataset = ForceVLAWindowDataset(path, use_external_torque=True)
            batch = collate_force_windows([dataset[0]])
            validate_force_window_batch(batch, use_external_torque=True)
            statistics = {name: {"offset": [offset] * size, "scale": [scale] * size}
                for name, size, offset, scale in [("state", 10, 0, 1), ("action", 10, 0, 1),
                    ("torque", 7, 1, 2), ("external_torque", 7, 2, 3)]}
            normalized = ForceNormalizer(statistics)(batch)
            torch.testing.assert_close(normalized["joint_torque_history"][..., :7], torch.full((1, 10, 7), 2.))
            torch.testing.assert_close(normalized["joint_torque_history"][..., 7:], torch.full((1, 10, 7), 3.))
            torch.testing.assert_close(normalized["future_joint_torque"], torch.full((1, 50, 7), 2.))
            weight = torch.randn(32, 70)
            expanded = expand_history_weight(weight)
            history = torch.randn(2, 10, 14)
            torch.testing.assert_close(history.flatten(1) @ expanded.T,
                history[..., :7].flatten(1) @ weight.T)
            self.assertEqual(int(torch.count_nonzero(expanded.reshape(32, 10, 14)[..., 7:])), 0)

    def test_external_history_receives_gradients_and_inference_still_predicts_50_actions(self):
        torch.set_num_threads(2)
        config = load_force_config("configs/pi05_force_tavla_aligned.json")
        config.use_external_torque = True
        config.gradient_checkpointing = False
        with patch.object(base.PI05Pytorch, "__init__", tiny_init):
            policy = ForcePI05Policy(config)
        configure_trainable(policy, "adapters")
        batch = {"observation.language.tokens": torch.tensor([[1, 2, 3]]),
            "observation.language.attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "observation.state": torch.randn(1, 10), "joint_torque_history": torch.randn(1, 10, 14),
            "future_joint_torque": torch.randn(1, 50, 7), "action": torch.randn(1, 50, 10)}
        with patch.object(policy, "_preprocess_images", return_value=([], [])):
            loss, _ = policy(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradient = policy.model.torque_adapter.projection[0].weight.grad.reshape(-1, 10, 14)
            self.assertGreater(float(gradient[..., 7:].abs().sum()), 0)
            self.assertGreater(float(gradient[..., :7].abs().sum()), 0)
            with torch.no_grad():
                actions = policy.predict_action_chunk(batch, num_steps=2)
            self.assertEqual(tuple(actions.shape), (1, 50, 10))
            self.assertTrue(torch.isfinite(actions).all())


if __name__ == "__main__":
    unittest.main()
