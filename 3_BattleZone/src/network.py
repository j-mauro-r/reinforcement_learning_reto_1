"""BattleZone DQN Q-network for HU005."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class BattleZoneQNetwork(nn.Module):
    """Convolutional Q-network compatible with the HU003 BattleZone contract."""

    def __init__(
        self,
        *,
        action_dim: int,
        frame_stack: int,
        input_channels: int,
        hidden_dim: int,
        conv_channels: Iterable[int],
    ) -> None:
        """Initializes the Q-network from explicit versioned configuration."""
        super().__init__()
        channels = list(conv_channels)
        if len(channels) != 3:
            raise ValueError("conv_channels must contain exactly three values.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if frame_stack <= 0 or input_channels <= 0 or hidden_dim <= 0:
            raise ValueError("frame_stack, input_channels and hidden_dim must be positive.")

        in_channels = int(frame_stack) * int(input_channels)
        self.action_dim = int(action_dim)
        self.frame_stack = int(frame_stack)
        self.input_channels = int(input_channels)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[2] * 8 * 8, int(hidden_dim)),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(int(hidden_dim), self.action_dim)

    def preprocess_observations(self, observations: Tensor) -> Tensor:
        """Converts HU003 observations to float32 NCHW tensors."""
        if observations.ndim == 4:
            observations = observations.unsqueeze(0)
        if observations.ndim != 5:
            raise ValueError(
                "Expected observations with 4D or 5D shape: "
                "(frame_stack,H,W,C) or (batch,frame_stack,H,W,C)."
            )
        if observations.shape[1] != self.frame_stack:
            raise ValueError(
                f"Expected frame_stack={self.frame_stack}, got {observations.shape[1]}."
            )
        if observations.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected channels={self.input_channels}, got {observations.shape[-1]}."
            )
        if observations.dtype not in (torch.uint8, torch.float16, torch.float32, torch.float64):
            raise TypeError(
                "Expected observations dtype uint8/float16/float32/float64, "
                f"got {observations.dtype}."
            )

        obs = observations.to(dtype=torch.float32)
        if observations.dtype == torch.uint8:
            obs = obs / 255.0

        batch_size, frame_stack, height, width, channels = obs.shape
        obs = obs.permute(0, 1, 4, 2, 3).contiguous()
        return obs.view(batch_size, frame_stack * channels, height, width)

    def forward(self, observations: Tensor) -> Tensor:
        """Computes one Q-value per BattleZone action."""
        x = self.preprocess_observations(observations)
        x = self.features(x)
        x = self.head(x)
        return self.output_layer(x)
