"""Focused tests for BattleZone DQN replay buffer HU005."""

from __future__ import annotations

import numpy as np
import pytest

from src.replay_buffer import ReplayBuffer


STATE_SHAPE = (4, 128, 128, 3)


def _state(fill_value: int) -> np.ndarray:
    return np.full(STATE_SHAPE, fill_value=fill_value, dtype=np.uint8)


def test_add_and_len_with_small_capacity():
    buffer = ReplayBuffer(capacity=3, state_shape=STATE_SHAPE)
    assert len(buffer) == 0

    buffer.add(_state(1), 2, 3.0, _state(4), False)
    assert len(buffer) == 1

    buffer.add(_state(5), 1, 0.5, _state(6), True)
    buffer.add(_state(7), 0, -1.0, _state(8), False)
    buffer.add(_state(9), 3, 2.5, _state(10), True)
    assert len(buffer) == 3


def test_capacity_overwrite_replaces_oldest_transition():
    buffer = ReplayBuffer(capacity=2, state_shape=STATE_SHAPE)
    buffer.add(_state(10), 0, 1.0, _state(11), False)
    buffer.add(_state(20), 1, 2.0, _state(21), False)
    buffer.add(_state(30), 2, 3.0, _state(31), True)

    # First transition must be overwritten once capacity is exceeded.
    assert len(buffer) == 2
    assert not np.any(np.all(buffer.states == _state(10), axis=(1, 2, 3, 4)))


def test_sample_returns_consistent_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=5, state_shape=STATE_SHAPE)
    for idx in range(5):
        buffer.add(_state(idx), idx % 18, float(idx), _state(idx + 1), idx % 2 == 0)

    batch = buffer.sample(batch_size=4)
    assert batch["states"].shape == (4, *STATE_SHAPE)
    assert batch["next_states"].shape == (4, *STATE_SHAPE)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["dones"].shape == (4,)

    assert batch["states"].dtype == np.uint8
    assert batch["next_states"].dtype == np.uint8
    assert batch["actions"].dtype == np.int64
    assert batch["rewards"].dtype == np.float32
    assert batch["dones"].dtype == np.bool_


def test_sample_larger_than_buffer_raises_error():
    buffer = ReplayBuffer(capacity=2, state_shape=STATE_SHAPE)
    buffer.add(_state(1), 0, 0.0, _state(2), False)

    with pytest.raises(ValueError, match="Cannot sample"):
        buffer.sample(batch_size=2)


def test_invalid_state_shape_raises_error():
    buffer = ReplayBuffer(capacity=2, state_shape=STATE_SHAPE)
    bad_state = np.zeros((4, 128, 128), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        buffer.add(bad_state, 0, 0.0, _state(2), False)


def test_invalid_state_dtype_raises_error():
    buffer = ReplayBuffer(capacity=2, state_shape=STATE_SHAPE)
    bad_state = np.zeros(STATE_SHAPE, dtype=np.float32)

    with pytest.raises(TypeError, match="dtype"):
        buffer.add(bad_state, 0, 0.0, _state(2), False)


def test_state_dict_and_load_state_dict_restore_full_content():
    source = ReplayBuffer(capacity=4, state_shape=STATE_SHAPE)
    source.add(_state(11), 3, 1.25, _state(12), False)
    source.add(_state(21), 4, -0.75, _state(22), True)
    payload = source.state_dict()

    target = ReplayBuffer(capacity=4, state_shape=STATE_SHAPE)
    target.load_state_dict(payload)

    assert len(target) == len(source)
    assert np.array_equal(target.states, source.states)
    assert np.array_equal(target.next_states, source.next_states)
    assert np.array_equal(target.actions, source.actions)
    assert np.array_equal(target.rewards, source.rewards)
    assert np.array_equal(target.dones, source.dones)


def test_load_state_dict_rejects_incompatible_capacity():
    source = ReplayBuffer(capacity=3, state_shape=STATE_SHAPE)
    source.add(_state(1), 0, 0.0, _state(2), False)
    payload = source.state_dict()

    incompatible = ReplayBuffer(capacity=4, state_shape=STATE_SHAPE)
    with pytest.raises(ValueError, match="capacity"):
        incompatible.load_state_dict(payload)


def test_load_state_dict_rejects_incompatible_dtype():
    source = ReplayBuffer(capacity=3, state_shape=STATE_SHAPE)
    source.add(_state(1), 0, 0.0, _state(2), False)
    payload = source.state_dict()
    payload["states"] = payload["states"].astype(np.float32)

    target = ReplayBuffer(capacity=3, state_shape=STATE_SHAPE)
    with pytest.raises(TypeError, match="states dtype"):
        target.load_state_dict(payload)


def test_load_state_dict_rejects_corrupt_payload_keys():
    source = ReplayBuffer(capacity=3, state_shape=STATE_SHAPE)
    payload = source.state_dict()
    payload.pop("dones")

    target = ReplayBuffer(capacity=3, state_shape=STATE_SHAPE)
    with pytest.raises(ValueError, match="Invalid replay state keys"):
        target.load_state_dict(payload)
