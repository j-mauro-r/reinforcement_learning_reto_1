"""Tests for the HU003 Q-network."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.network import QNetwork


def test_q_network_maps_uint8_states_to_finite_q_values():
    network = QNetwork(input_channels=4, num_actions=7)
    states = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)

    q_values = network(states)

    assert q_values.shape == (2, 7)
    assert q_values.dtype == torch.float32
    assert torch.isfinite(q_values).all()


def test_q_network_runs_on_gpu_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this runtime.")
    network = QNetwork(input_channels=4, num_actions=7).cuda()
    states = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8, device="cuda")

    q_values = network(states)

    assert q_values.is_cuda
    assert q_values.shape == (2, 7)
