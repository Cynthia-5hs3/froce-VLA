import unittest
from types import SimpleNamespace

import numpy as np
import torch

from force_vla.pi05.realtime import is_abort_key, safe_action


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


if __name__ == "__main__":
    unittest.main()
