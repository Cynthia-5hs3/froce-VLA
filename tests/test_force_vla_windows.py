import unittest

import numpy as np

from build_force_vla_windows import nearest_indices


class ForceVlaWindowTests(unittest.TestCase):
    def test_nearest_timestamp_lookup_is_causal_for_history(self):
        timestamps = np.arange(200, dtype=np.int64) * 1_000_000_000 // 30
        query = timestamps[100] + np.asarray([-60, -53, -47, -40, -33, -27, -20, -13, -7, 0]) * 1_000_000_000 // 30
        indices, distances = nearest_indices(timestamps, query)
        self.assertEqual(indices.tolist(), [40, 47, 53, 60, 67, 73, 80, 87, 93, 100])
        self.assertTrue(np.all(distances <= 1))

    def test_nearest_timestamp_lookup_clamps_outside_dataset(self):
        timestamps = np.arange(10, dtype=np.int64)
        indices, distances = nearest_indices(timestamps, np.asarray([-5, 3, 20]))
        self.assertEqual(indices.tolist(), [0, 3, 9])
        self.assertEqual(distances.tolist(), [5, 0, 11])


if __name__ == "__main__":
    unittest.main()
