import unittest

import torch

from force_vla.pi05.configuration_pi05_force import ForcePI05Config
from force_vla.pi05.torque_adapter import TorqueAdapter
from force_vla.pi05.window_dataset import validate_force_window_batch


class TorqueAdapterTests(unittest.TestCase):
    def test_history_is_compressed_to_one_token(self):
        adapter = TorqueAdapter(history_steps=10, torque_dim=7, token_dim=32, hidden_dim=64)
        history = torch.randn(3, 10, 7, requires_grad=True)
        token = adapter(history)
        self.assertEqual(tuple(token.shape), (3, 32))
        token.sum().backward()
        self.assertIsNotNone(history.grad)
        self.assertTrue(torch.isfinite(history.grad).all())

    def test_flat_history_is_supported(self):
        adapter = TorqueAdapter(history_steps=10, torque_dim=7, token_dim=16)
        self.assertEqual(tuple(adapter(torch.randn(2, 70)).shape), (2, 16))

    def test_bad_history_shape_is_rejected(self):
        adapter = TorqueAdapter(history_steps=10, torque_dim=7, token_dim=16)
        with self.assertRaises(ValueError):
            adapter(torch.randn(2, 9, 7))

    def test_config_rejects_external_torque_first_version(self):
        with self.assertRaises(ValueError):
            ForcePI05Config(use_external_torque=True)

    def test_window_batch_contract(self):
        batch = {
            "observation.state": torch.zeros(2, 10),
            "joint_torque_history": torch.zeros(2, 10, 7),
            "action": torch.zeros(2, 50, 10),
            "future_joint_torque": torch.zeros(2, 50, 7),
        }
        validate_force_window_batch(batch)


if __name__ == "__main__":
    unittest.main()
