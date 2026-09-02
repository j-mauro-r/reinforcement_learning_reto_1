"""Uniform replay buffer for BattleZone DQN HU005."""

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
        """Adds a transition to the replay buffer."""
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
        """Samples a uniform batch of transitions."""
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

    def state_dict(self) -> Dict[str, np.ndarray | int | tuple[int, ...]]:
        """Returns a fully restorable replay state for full checkpoints."""
        return {
            "capacity": int(self.capacity),
            "state_shape": tuple(self.state_shape),
            "position": int(self._position),
            "size": int(self._size),
            "states": self.states.copy(),
            "next_states": self.next_states.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "dones": self.dones.copy(),
        }

    def load_state_dict(self, state: Dict[str, np.ndarray | int | tuple[int, ...]]) -> None:
        """Restores replay state and validates structural compatibility strictly."""
        required_keys = {
            "capacity",
            "state_shape",
            "position",
            "size",
            "states",
            "next_states",
            "actions",
            "rewards",
            "dones",
        }
        missing = required_keys.difference(state.keys())
        extra = set(state.keys()).difference(required_keys)
        if missing or extra:
            raise ValueError(
                f"Invalid replay state keys. missing={sorted(missing)}, extra={sorted(extra)}."
            )

        incoming_capacity = int(state["capacity"])
        incoming_shape = tuple(int(dim) for dim in state["state_shape"])
        incoming_position = int(state["position"])
        incoming_size = int(state["size"])

        if incoming_capacity != self.capacity:
            raise ValueError(
                f"Incompatible replay capacity. expected={self.capacity}, incoming={incoming_capacity}."
            )
        if incoming_shape != self.state_shape:
            raise ValueError(
                f"Incompatible replay state_shape. expected={self.state_shape}, incoming={incoming_shape}."
            )
        if not (0 <= incoming_size <= self.capacity):
            raise ValueError(
                f"Invalid replay size={incoming_size}; expected range [0, {self.capacity}]."
            )
        if not (0 <= incoming_position < self.capacity):
            raise ValueError(
                f"Invalid replay position={incoming_position}; expected range [0, {self.capacity - 1}]."
            )

        states = np.asarray(state["states"])
        next_states = np.asarray(state["next_states"])
        actions = np.asarray(state["actions"])
        rewards = np.asarray(state["rewards"])
        dones = np.asarray(state["dones"])

        expected_state_shape = (self.capacity, *self.state_shape)
        if states.shape != expected_state_shape:
            raise ValueError(
                f"Incompatible states shape. expected={expected_state_shape}, incoming={states.shape}."
            )
        if next_states.shape != expected_state_shape:
            raise ValueError(
                f"Incompatible next_states shape. expected={expected_state_shape}, incoming={next_states.shape}."
            )
        if actions.shape != (self.capacity,):
            raise ValueError(
                f"Incompatible actions shape. expected={(self.capacity,)}, incoming={actions.shape}."
            )
        if rewards.shape != (self.capacity,):
            raise ValueError(
                f"Incompatible rewards shape. expected={(self.capacity,)}, incoming={rewards.shape}."
            )
        if dones.shape != (self.capacity,):
            raise ValueError(
                f"Incompatible dones shape. expected={(self.capacity,)}, incoming={dones.shape}."
            )

        if states.dtype != self.states.dtype:
            raise TypeError(
                f"Incompatible states dtype. expected={self.states.dtype}, incoming={states.dtype}."
            )
        if next_states.dtype != self.next_states.dtype:
            raise TypeError(
                f"Incompatible next_states dtype. expected={self.next_states.dtype}, incoming={next_states.dtype}."
            )
        if actions.dtype != self.actions.dtype:
            raise TypeError(
                f"Incompatible actions dtype. expected={self.actions.dtype}, incoming={actions.dtype}."
            )
        if rewards.dtype != self.rewards.dtype:
            raise TypeError(
                f"Incompatible rewards dtype. expected={self.rewards.dtype}, incoming={rewards.dtype}."
            )
        if dones.dtype != self.dones.dtype:
            raise TypeError(
                f"Incompatible dones dtype. expected={self.dones.dtype}, incoming={dones.dtype}."
            )

        self.states[:] = states
        self.next_states[:] = next_states
        self.actions[:] = actions
        self.rewards[:] = rewards
        self.dones[:] = dones
        self._position = incoming_position
        self._size = incoming_size
