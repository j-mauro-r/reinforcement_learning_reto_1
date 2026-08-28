"""Uniform replay buffer for DDQN transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    """Batch of sampled replay transitions."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Returns the batch as a dictionary."""
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
        }


class ReplayBuffer:
    """Fixed-capacity circular replay buffer with uniform sampling."""

    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, int, int] = (4, 84, 84),
        seed: int | None = None,
    ) -> None:
        """Initializes the replay buffer.

        Args:
            capacity: Maximum number of transitions.
            state_shape: Shape of one visual state.
            seed: Optional seed for uniform sampling.

        Raises:
            ValueError: If capacity or state shape are invalid.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if tuple(state_shape) != (4, 84, 84):
            raise ValueError(f"HU003 expects state_shape=(4, 84, 84), got {state_shape}.")

        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape)
        self._rng = np.random.default_rng(seed)
        self._states = np.empty((self.capacity, *self.state_shape), dtype=np.uint8)
        self._next_states = np.empty((self.capacity, *self.state_shape), dtype=np.uint8)
        self._actions = np.empty((self.capacity,), dtype=np.int64)
        self._rewards = np.empty((self.capacity,), dtype=np.float32)
        self._dones = np.empty((self.capacity,), dtype=np.bool_)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        """Returns the current number of stored transitions."""
        return self._size

    @property
    def position(self) -> int:
        """Returns the next circular write position."""
        return self._position

    def state_dict(self) -> Dict[str, Any]:
        """Exports the valid replay buffer state.

        Returns:
            Serializable dictionary containing only valid occupied slots and
            the RNG state required to continue uniform sampling.
        """
        valid_slice = slice(0, self._size)
        return {
            "capacity": self.capacity,
            "state_shape": self.state_shape,
            "size": self._size,
            "position": self._position,
            "states": self._states[valid_slice].copy(),
            "next_states": self._next_states[valid_slice].copy(),
            "actions": self._actions[valid_slice].copy(),
            "rewards": self._rewards[valid_slice].copy(),
            "dones": self._dones[valid_slice].copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restores a replay buffer state exported by ``state_dict``.

        Args:
            state: Serialized replay buffer dictionary.

        Raises:
            ValueError: If the serialized state is incompatible.
        """
        required_keys = {
            "capacity",
            "state_shape",
            "size",
            "position",
            "states",
            "next_states",
            "actions",
            "rewards",
            "dones",
            "rng_state",
        }
        missing = sorted(required_keys - set(state))
        if missing:
            raise ValueError(f"Replay buffer state missing keys: {missing}")
        if int(state["capacity"]) != self.capacity:
            raise ValueError(f"Replay buffer capacity mismatch: {state['capacity']} != {self.capacity}.")
        if tuple(state["state_shape"]) != self.state_shape:
            raise ValueError(f"Replay buffer state_shape mismatch: {state['state_shape']} != {self.state_shape}.")

        size = int(state["size"])
        position = int(state["position"])
        if size < 0 or size > self.capacity:
            raise ValueError(f"Invalid replay buffer size: {size}.")
        if position < 0 or position >= self.capacity:
            raise ValueError(f"Invalid replay buffer position: {position}.")

        states = np.asarray(state["states"], dtype=np.uint8)
        next_states = np.asarray(state["next_states"], dtype=np.uint8)
        actions = np.asarray(state["actions"], dtype=np.int64)
        rewards = np.asarray(state["rewards"], dtype=np.float32)
        dones = np.asarray(state["dones"], dtype=np.bool_)
        if states.shape != (size, *self.state_shape):
            raise ValueError(f"Invalid serialized states shape: {states.shape}.")
        if next_states.shape != (size, *self.state_shape):
            raise ValueError(f"Invalid serialized next_states shape: {next_states.shape}.")
        if actions.shape != (size,) or rewards.shape != (size,) or dones.shape != (size,):
            raise ValueError("Serialized action/reward/done arrays must match replay buffer size.")

        self._states[:size] = states
        self._next_states[:size] = next_states
        self._actions[:size] = actions
        self._rewards[:size] = rewards
        self._dones[:size] = dones
        self._size = size
        self._position = position
        self._rng.bit_generator.state = state["rng_state"]

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores one transition in circular order.

        Args:
            state: Current state shaped ``(4, 84, 84)``.
            action: Discrete action index.
            reward: Scalar reward.
            next_state: Next state shaped ``(4, 84, 84)``.
            done: Whether the transition ended the episode.
        """
        self._states[self._position] = self._validate_state(state, "state")
        self._next_states[self._position] = self._validate_state(next_state, "next_state")
        self._actions[self._position] = int(action)
        self._rewards[self._position] = float(reward)
        self._dones[self._position] = bool(done)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Samples a uniform batch without replacement.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batch of replay transitions.

        Raises:
            ValueError: If ``batch_size`` is invalid or the buffer is too small.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer of size {self._size}.")

        indices = self._rng.choice(self._size, size=int(batch_size), replace=False)
        return ReplayBatch(
            states=self._states[indices].copy(),
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_states=self._next_states[indices].copy(),
            dones=self._dones[indices].copy(),
        )

    def _validate_state(self, state: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(state)
        if array.shape != self.state_shape:
            raise ValueError(f"{name} must have shape {self.state_shape}, got {array.shape}.")
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return array
