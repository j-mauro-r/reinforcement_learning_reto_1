"""Tests for the HU003 uniform replay buffer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.replay_buffer import ReplayBuffer


def _state(value: int) -> np.ndarray:
    return np.full((4, 84, 84), value, dtype=np.uint8)


def test_replay_buffer_respects_capacity_and_circular_replacement():
    buffer = ReplayBuffer(capacity=3, seed=123)
    for index in range(5):
        buffer.add(_state(index), action=index % 7, reward=float(index), next_state=_state(index + 1), done=False)

    batch = buffer.sample(3)
    sampled_values = set(int(state[0, 0, 0]) for state in batch.states)

    assert len(buffer) == 3
    assert buffer.position == 2
    assert sampled_values == {2, 3, 4}


def test_replay_buffer_sample_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=5, seed=42)
    for index in range(5):
        buffer.add(_state(index), action=index % 7, reward=index + 0.5, next_state=_state(index + 1), done=index == 4)

    batch = buffer.sample(4)

    assert batch.states.shape == (4, 4, 84, 84)
    assert batch.next_states.shape == (4, 4, 84, 84)
    assert batch.states.dtype == np.uint8
    assert batch.next_states.dtype == np.uint8
    assert batch.actions.shape == (4,)
    assert batch.actions.dtype == np.int64
    assert batch.rewards.dtype == np.float32
    assert batch.dones.dtype == np.bool_


def test_replay_buffer_rejects_sampling_more_than_available():
    buffer = ReplayBuffer(capacity=2)
    buffer.add(_state(1), action=0, reward=0.0, next_state=_state(2), done=False)

    with pytest.raises(ValueError, match="Cannot sample"):
        buffer.sample(2)
