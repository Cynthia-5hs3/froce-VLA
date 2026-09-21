import unittest
from types import SimpleNamespace

import numpy as np

from force_vla.pi05.teleop_assist import AssistCommand, assist_action_from_state


class TeleopAssistTests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(O_T_EE=np.eye(4, dtype=np.float64))
        self.model_action = np.array(
            [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32
        )

    def test_inactive_assist_preserves_model_action(self):
        command = AssistCommand(np.zeros(3), np.zeros(3), None, False)
        np.testing.assert_allclose(assist_action_from_state(self.state, self.model_action, command, 0.125), self.model_action)

    def test_active_assist_creates_small_position_override(self):
        command = AssistCommand(np.array([0.02, 0.0, 0.0]), np.zeros(3), None, True)
        action = assist_action_from_state(self.state, self.model_action, command, 0.125)
        np.testing.assert_allclose(action[:3], [0.0025, 0.0, 0.0])
        self.assertEqual(action[9], 1.0)


if __name__ == "__main__":
    unittest.main()
