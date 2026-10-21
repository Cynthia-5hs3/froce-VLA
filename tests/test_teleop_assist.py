import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from force_vla.pi05.teleop_assist import AssistCommand, XboxAssist, assist_action_from_state


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

    def button(self, assist, code, value=1):
        assist._update_event(SimpleNamespace(type=1, code=code, value=value))

    def test_latched_takeover_holds_until_explicit_resume(self):
        assist = XboxAssist("mock", latched=True)
        self.assertFalse(assist.command().active)
        self.button(assist, assist.BUTTON_Y)
        command = assist.command()
        self.assertTrue(command.active)
        self.assertGreater(command.linear_velocity[2], 0)
        self.button(assist, assist.BUTTON_Y, 0)
        command = assist.command()
        self.assertTrue(command.active)
        np.testing.assert_array_equal(command.linear_velocity, np.zeros(3))
        self.button(assist, assist.BUTTON_RB)
        self.assertFalse(assist.command().active)

    def test_brief_lb_press_takes_control_without_motion(self):
        assist = XboxAssist("mock", latched=True)
        self.button(assist, assist.BUTTON_LB)
        self.button(assist, assist.BUTTON_LB, 0)
        command = assist.command()
        self.assertTrue(command.active)
        np.testing.assert_array_equal(command.linear_velocity, np.zeros(3))
        self.assertIsNone(command.gripper_target)

    def test_resume_requires_neutral_controls_and_new_press(self):
        assist = XboxAssist("mock", latched=True)
        self.button(assist, assist.BUTTON_A)
        self.button(assist, assist.BUTTON_RB)
        self.assertTrue(assist.command().active)
        self.button(assist, assist.BUTTON_A, 0)
        self.assertTrue(assist.command().active)
        self.button(assist, assist.BUTTON_RB, 2)
        self.assertTrue(assist.command().active)
        self.button(assist, assist.BUTTON_RB, 0)
        self.button(assist, assist.BUTTON_RB)
        self.assertFalse(assist.command().active)

    def test_original_momentary_mode_still_releases(self):
        assist = XboxAssist("mock")
        self.button(assist, assist.BUTTON_Y)
        self.assertTrue(assist.command().active)
        self.button(assist, assist.BUTTON_Y, 0)
        self.assertFalse(assist.command().active)

    def test_released_triggers_do_not_generate_spurious_close(self):
        assist = XboxAssist("mock")
        assist._axes[assist.AXIS_LT] = -1.
        self.assertIsNone(assist.command().gripper_target)
        assist._axes[assist.AXIS_LT] = 1.
        self.assertEqual(assist.command().gripper_target, 1.)

    def test_device_failure_rejects_stale_motion(self):
        assist = XboxAssist("mock", latched=True)
        assist._device = MagicMock()
        assist._axes[assist.AXIS_X] = 1.
        with patch("force_vla.pi05.teleop_assist.select.select", side_effect=OSError("unplugged")):
            assist._read_loop()
        with self.assertRaisesRegex(RuntimeError, "Gamepad input failed"):
            assist.command()

    def test_event_overflow_rejects_unsynchronized_input(self):
        assist = XboxAssist("mock")
        with self.assertRaisesRegex(OSError, "lost synchronization"):
            assist._update_event(SimpleNamespace(type=0, code=3, value=0))


if __name__ == "__main__":
    unittest.main()
