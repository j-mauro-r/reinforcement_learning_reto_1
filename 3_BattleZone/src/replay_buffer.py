"""Uniform replay buffer for BattleZone DDQN HU005."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class ReplayBuffer:
    """Stores transitions in CPU RAM with uint8 states and uniform sampling."""

    def __init__(self, *, capacity: int, state_shape: Tuple[int, ...]) -> None:
        """Initializes a fixed-capacity replay buffer.

        Args:
            capacity: Maximum number of transitions.
            state_shape: Shape of each state, e.g. ``(4, 128, 128, 3)``.

        Raises:
            ValueError: If capacity or state_shape are invalid.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if not state_shape:
            raise ValueError("state_shape must not be empty.")

        self.capacity = int(capacity)
        self.state_shape = tuple(int(dim) for dim in state_shape)
        self._position = 0
        self._size = 0

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.uint8)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

    def __len__(self) -> int:
        """Returns the number of stored transitions."""
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Adds a transition to the replay buffer.

        Args:
            state: State observation with ``state_shape`` and dtype ``uint8``.
            action: Integer action index.
            reward: Scalar reward.
            next_state: Next state observation with ``state_shape`` and dtype ``uint8``.
            done: Terminal transition flag.

        Raises:
            ValueError: If shapes are invalid.
            TypeError: If state dtypes are not uint8.
        """
        state_array = self._validate_state("state", state)
        next_state_array = self._validate_state("next_state", next_state)

        index = self._position
        self.states[index] = state_array
        self.next_states[index] = next_state_array
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = bool(done)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Samples a uniform batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with arrays for states, actions, rewards, next_states, and dones.

        Raises:
            ValueError: If ``batch_size`` is invalid or larger than current buffer size.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from replay size={self._size}."
            )

        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def _validate_state(self, name: str, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != self.state_shape:
            raise ValueError(
                f"{name} must have shape {self.state_shape}, got {array.shape}."
            )
        if array.dtype != np.uint8:
            raise TypeError(f"{name} must have dtype uint8, got {array.dtype}.")
        return array
