"""Compress causal joint torque history into one action-expert token."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TorqueAdapter(nn.Module):
    """Map ``[batch, history, joints]`` torque samples to one model token.

    The adapter follows TA-VLA's concatenated-history route: all ten history
    samples are flattened before projection, so the decoder receives one token
    describing the complete causal window.
    """

    def __init__(
        self,
        history_steps: int = 10,
        torque_dim: int = 7,
        token_dim: int = 1024,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if history_steps < 1 or torque_dim < 1 or token_dim < 1:
            raise ValueError("history_steps, torque_dim, and token_dim must be positive")
        hidden_dim = hidden_dim or 2 * token_dim
        self.history_steps = history_steps
        self.torque_dim = torque_dim
        self.token_dim = token_dim
        self.input_dim = history_steps * torque_dim
        self.input_norm = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, torque_history: Tensor) -> Tensor:
        if torque_history.ndim == 2:
            torque_history = torque_history.reshape(torque_history.shape[0], self.history_steps, self.torque_dim)
        expected = (self.history_steps, self.torque_dim)
        if torque_history.ndim != 3 or tuple(torque_history.shape[1:]) != expected:
            raise ValueError(
                f"torque_history must have shape [B, {self.history_steps}, {self.torque_dim}], "
                f"got {tuple(torque_history.shape)}"
            )
        flattened = torque_history.reshape(torque_history.shape[0], self.input_dim)
        return self.projection(self.input_norm(flattened))
