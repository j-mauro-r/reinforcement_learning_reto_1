"""BattleZone DDQN Q-network for HU005."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class BattleZoneQNetwork(nn.Module):
    """Convolutional Q-network compatible with the HU003 BattleZone contract.

    The HU003 environment emits states shaped as ``(frame_stack, height, width, channels)``
    with ``uint8`` values. This module converts to float32, scales pixels to ``[0, 1]``,
    and reshapes to NCHW by combining frame and RGB channels into a single channel axis.
    """

    def __init__(
        self,
        *,
        action_dim: int = 18,
        frame_stack: int = 4,
        input_channels: int = 3,
        hidden_dim: int = 512,
        conv_channels: Iterable[int] = (32, 64, 64),
    ) -> None:
        """Initializes the BattleZone Q-network.

        Args:
            action_dim: Number of discrete actions. BattleZone requires 18.
            frame_stack: Number of stacked frames from HU003.
            input_channels: Number of channels per frame (3 for RGB).
            hidden_dim: Hidden units of the fully connected layer.
            conv_channels: Output channels for each convolution block.

        Raises:
            ValueError: If configuration values are invalid.
        """
        super().__init__()
        channels = list(conv_channels)
        if len(channels) != 3:
            raise ValueError("conv_channels must contain exactly three values.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if frame_stack <= 0 or input_channels <= 0:
            raise ValueError("frame_stack and input_channels must be positive.")

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
        """Converts HU003 observations to float32 NCHW tensors.

        Supported input layouts:
        - ``(batch, frame_stack, height, width, channels)``
        - ``(frame_stack, height, width, channels)`` (single observation)

        Args:
            observations: Observation tensor in HU003 layout.

        Returns:
            Float32 tensor in NCHW layout with values scaled to ``[0, 1]``.

        Raises:
            ValueError: If shape does not match expected dimensions.
            TypeError: If dtype is unsupported.
        """
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
        obs = obs.view(batch_size, frame_stack * channels, height, width)
        return obs

    def forward(self, observations: Tensor) -> Tensor:
        """Computes Q-values for each BattleZone action.

        Args:
            observations: HU003 observations in 4D or 5D layout.

        Returns:
            Tensor with shape ``(batch_size, action_dim)``.
        """
        x = self.preprocess_observations(observations)
        x = self.features(x)
        x = self.head(x)
        return self.output_layer(x)
