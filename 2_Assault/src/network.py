"""Convolutional Q-network for the Assault DDQN agent."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Atari-style CNN that maps stacked frames to action Q-values."""

    def __init__(self, input_channels: int = 4, num_actions: int = 7) -> None:
        """Initializes the Q-network.

        Args:
            input_channels: Number of stacked input frames.
            num_actions: Number of discrete actions to estimate.

        Raises:
            ValueError: If channels or actions are not positive.
        """
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")

        self.input_channels = int(input_channels)
        self.num_actions = int(num_actions)
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(self._feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Computes Q-values for a batch of states.

        Args:
            states: Tensor with shape ``(batch, channels, 84, 84)``. ``uint8``
                inputs are converted to ``float32`` and normalized to ``[0, 1]``.

        Returns:
            Tensor with shape ``(batch, num_actions)``.

        Raises:
            ValueError: If the input shape is incompatible with the network.
        """
        if states.ndim != 4:
            raise ValueError(f"Expected states with 4 dimensions, got {tuple(states.shape)}.")
        if states.shape[1:] != (self.input_channels, 84, 84):
            raise ValueError(
                f"Expected states shaped (batch, {self.input_channels}, 84, 84), got {tuple(states.shape)}."
            )

        states = states.to(dtype=torch.float32)
        if states.max().detach() > 1.0:
            states = states / 255.0
        return self.head(self.features(states))

    def _feature_size(self) -> int:
        with torch.no_grad():
            sample = torch.zeros(1, self.input_channels, 84, 84)
            features = self.features(sample)
        return int(features.shape[1])


def build_q_network(config: dict) -> QNetwork:
    """Builds a Q-network from the project configuration.

    Args:
        config: Parsed YAML configuration with a ``network`` section.

    Returns:
        Configured ``QNetwork`` instance.
    """
    network_config = config["network"]
    return QNetwork(
        input_channels=int(network_config["input_channels"]),
        num_actions=int(network_config["num_actions"]),
    )
