"""Focused tests for BattleZone DDQN network HU005."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.network import BattleZoneQNetwork


STATE_SHAPE = (4, 128, 128, 3)


def _random_state(batch_size: int) -> torch.Tensor:
    array = np.random.default_rng(123).integers(
        low=0,
        high=256,
        size=(batch_size, *STATE_SHAPE),
        dtype=np.uint8,
    )
    return torch.from_numpy(array)


def test_forward_batch_output_shape_and_finite_values():
    network = BattleZoneQNetwork(action_dim=18)
    observations = _random_state(batch_size=3)
    q_values = network(observations)

    assert q_values.shape == (3, 18)
    assert torch.isfinite(q_values).all()


def test_forward_single_observation_is_supported():
    network = BattleZoneQNetwork(action_dim=18)
    single = _random_state(batch_size=1).squeeze(0)
    q_values = network(single)

    assert q_values.shape == (1, 18)


def test_preprocess_converts_to_nchw_float32_scaled():
    network = BattleZoneQNetwork(action_dim=18)
    observations = _random_state(batch_size=2)
    processed = network.preprocess_observations(observations)

    assert processed.shape == (2, 12, 128, 128)
    assert processed.dtype == torch.float32
    assert float(processed.min()) >= 0.0
    assert float(processed.max()) <= 1.0


def test_invalid_layout_raises_clear_error():
    network = BattleZoneQNetwork(action_dim=18)
    bad_observations = torch.zeros((2, 128, 128, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="frame_stack"):
        network.preprocess_observations(bad_observations)


def test_invalid_dtype_raises_clear_error():
    network = BattleZoneQNetwork(action_dim=18)
    bad_dtype = torch.zeros((1, *STATE_SHAPE), dtype=torch.int16)

    with pytest.raises(TypeError, match="dtype"):
        network.preprocess_observations(bad_dtype)
