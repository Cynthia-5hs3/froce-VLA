import unittest

import numpy as np

from lab import make_window


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.episode = {'time': np.arange(180) / 30.0,
                        'state': np.ones((180, 29), dtype=np.float32),
                        'action': np.ones((180, 10), dtype=np.float32),
                        'valid': np.ones(180, dtype=bool), 'segment': np.zeros(180, dtype=int)}

    def test_shapes_and_causal_history(self):
        result = make_window(self.episode, 90)
        self.assertEqual(result['history'].shape, (10, 6))
        self.assertEqual(result['future_wrench'].shape, (50, 6))
        self.assertEqual(result['future_actions'].shape, (50, 10))
        self.assertLessEqual(result['history_indices'].max(), 90)

    def test_short_history_rejected(self):
        self.assertIsNone(make_window(self.episode, 20))

    def test_episode_end_rejected(self):
        self.assertIsNone(make_window(self.episode, 160))

    def test_invalid_action_rejected(self):
        self.episode['valid'][100] = False
        self.assertIsNone(make_window(self.episode, 90))

    def test_segment_boundary_rejected(self):
        self.episode['segment'][100:] = 1
        self.assertIsNone(make_window(self.episode, 90))

    def test_time_gap_rejected(self):
        self.episode['time'][100:] += 0.3
        self.assertIsNone(make_window(self.episode, 90))

    def test_nonfinite_wrench_rejected(self):
        self.episode['state'][100, 10] = np.nan
        self.assertIsNone(make_window(self.episode, 90))


if __name__ == '__main__':
    unittest.main()
