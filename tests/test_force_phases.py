import unittest
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

from force_vla.pi05.evaluation import select_episode_windows
from force_vla.pi05.extended_training import advance_rows
from force_vla.pi05.phase_evaluation import regroup_existing
from force_vla.pi05.phases import training_epoch_order


class PhaseTests(unittest.TestCase):
    def test_sampling_upsamples_only_marked_training_rows_and_is_reproducible(self):
        phases = np.zeros(120)
        phases[[3, 9, 110, 111]] = 1
        train = np.arange(100)
        sampled = training_epoch_order(train, 42, phases, .25)
        self.assertEqual(int(phases[sampled].sum()), 25)
        self.assertTrue(set(sampled) <= set(train))
        np.testing.assert_array_equal(sampled, training_epoch_order(train, 42, phases, .25))
        np.testing.assert_array_equal(training_epoch_order(train, 42), np.random.default_rng(42).permutation(train))

    def test_sampling_resume_preserves_sequence_across_epoch(self):
        indices = np.arange(8)
        phases = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        progress = {"epoch": 0, "cursor": 6}
        expected = (training_epoch_order(indices, 42, phases, .5)[6:].tolist()
                    + training_epoch_order(indices, 43, phases, .5)[:3].tolist())
        actual = advance_rows(indices, progress, 5, 42, phases, .5)
        self.assertEqual(actual, expected)
        self.assertEqual(progress, {"epoch": 1, "cursor": 3})

    def test_validation_covers_rare_marked_windows_without_training_leakage(self):
        dataset = SimpleNamespace(table=pa.table({
            "episode_index": [1] * 5 + [2] * 5 + [3] * 5,
            "anchor_frame": list(range(5)) * 3,
            "anchor_phase": [0, 0, 1, 0, 0] * 3}))
        selected = select_episode_windows(dataset, np.arange(5, 15), 2, phase_balanced=True)
        self.assertTrue({7, 12} <= set(selected))
        self.assertTrue(set(selected) <= set(range(5, 15)))
        self.assertEqual(len(selected), 6)

    def test_existing_eval_joins_by_episode_and_frame_and_counts_unique_windows(self):
        dataset = SimpleNamespace(table=pa.table({"episode_index": [1, 1, 2],
            "anchor_frame": [10, 20, 10], "anchor_phase": [0, 1, 1]}))
        evaluation = {"variants": {"measured": {"episode_macro_mean": {"error": 4.},
            "records": [{"episode": 1, "frame": 10, "error": 1.},
                        {"episode": 1, "frame": 20, "error": 3.},
                        {"episode": 1, "frame": 20, "error": 5.},
                        {"episode": 2, "frame": 10, "error": 8.}]}}}
        report = regroup_existing(dataset, evaluation)["by_variant"]["measured"]
        self.assertEqual(report["marked_critical"]["episode_macro_mean"]["error"], 6.)
        self.assertEqual(report["marked_critical"]["window_count"], 2)
        self.assertEqual(report["marked_critical"]["record_count"], 3)
        self.assertEqual(report["unmarked"]["episode_macro_mean"]["error"], 1.)


if __name__ == "__main__":
    unittest.main()
