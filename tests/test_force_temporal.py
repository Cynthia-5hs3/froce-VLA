import unittest

import numpy as np

from force_vla.pi05.temporal import ActionChunks, HISTORY_OFFSETS, HistoryNotReady, TorqueHistory


class TemporalTests(unittest.TestCase):
    def test_history_matches_training_offsets_and_excludes_future(self):
        history = TorqueHistory()
        origin = 10_000_000_000
        for frame in range(62):
            history.append(origin + round(frame * 1e9 / 30), np.full(7, frame))
        values, timestamps = history.sample(origin + 2_000_000_000)
        expected = 60 + np.asarray(HISTORY_OFFSETS)
        np.testing.assert_array_equal(values[:, 0], expected)
        self.assertTrue(np.all(timestamps <= origin + 2_000_000_000))
        self.assertEqual(timestamps[-1] - timestamps[0], 2_000_000_000)

    def test_history_requires_real_warmup_and_rejects_gaps(self):
        history = TorqueHistory()
        origin = 10_000_000_000
        for frame in range(59):
            history.append(origin + round(frame * 1e9 / 30), np.zeros(7))
        with self.assertRaises(HistoryNotReady):
            history.sample(history.samples[-1][0])
        history.clear()
        for frame in range(70):
            if not 30 <= frame <= 36:
                history.append(origin + round(frame * 1e9 / 30), np.zeros(7))
        with self.assertRaises(HistoryNotReady):
            history.sample(history.samples[-1][0])

    def chunk(self, sequence=1, anchor=10_000_000_000):
        actions = np.tile([.4, 0., .3, 1., 0., 0., 0., 1., 0., 1.], (50, 1))
        actions[:, 0] += np.arange(50) * .001
        return {"sequence": sequence, "anchor_ns": anchor, "actions": actions,
                "torques": np.repeat(np.arange(50)[:, None], 7, axis=1),
                "inference_ms": 180., "history_timestamps_ns": [anchor] * 10}

    def test_chunk_skips_compute_latency_and_advances_without_new_prediction(self):
        chunks = ActionChunks(blend_steps=0)
        chunk = self.chunk()
        now = chunk["anchor_ns"] + 180_000_000
        self.assertTrue(chunks.install(chunk, now, chunk["actions"][0]))
        first = chunks.sample(now)
        second = chunks.sample(now + 100_000_000)
        self.assertEqual(first["action_index"], 5)
        self.assertEqual(second["action_index"], 8)
        self.assertEqual(first["skipped_steps_on_arrival"], 5)
        self.assertAlmostEqual(second["action"][0], .408, places=6)
        self.assertIsNone(chunks.sample(chunk["anchor_ns"] + 1_666_666_667))

    def test_teleop_invalidation_rejects_inflight_chunk_and_accepts_fresh_one(self):
        chunks = ActionChunks(blend_steps=0)
        old = self.chunk()
        chunks.invalidate(old["anchor_ns"] + 500_000_000)
        self.assertFalse(chunks.install(old, old["anchor_ns"] + 600_000_000, old["actions"][0]))
        self.assertIsNone(chunks.sample(old["anchor_ns"] + 600_000_000))
        new = self.chunk(sequence=2, anchor=old["anchor_ns"] + 600_000_000)
        self.assertTrue(chunks.install(new, new["anchor_ns"], new["actions"][0]))

    def test_blend_preserves_gripper_and_does_not_mutate_model_output(self):
        chunks = ActionChunks(blend_steps=3)
        new = self.chunk()
        original = new["actions"].copy()
        fallback = original[0].copy()
        fallback[0], fallback[9] = .38, 0.
        chunks.install(new, new["anchor_ns"], fallback)
        result = chunks.sample(new["anchor_ns"])
        self.assertAlmostEqual(result["action"][0], .385, places=6)
        self.assertEqual(result["action"][9], 1.)
        np.testing.assert_array_equal(new["actions"], original)
        np.testing.assert_allclose(np.linalg.norm(result["action"][3:6]), 1.)


if __name__ == "__main__":
    unittest.main()
