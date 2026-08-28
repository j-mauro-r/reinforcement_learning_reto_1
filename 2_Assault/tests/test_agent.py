"""Tests for the HU003 DDQN agent core."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.agent import DDQNAgent
from src.environment import create_assault_env
from src.replay_buffer import ReplayBatch
from src.utils import load_yaml_config


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


class FixedQNetwork(nn.Module):
    """Network stub that returns controlled Q-values for DDQN target tests."""

    def __init__(self, q_values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("q_values", q_values.to(dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self.q_values.ndim == 1:
            return self.q_values.unsqueeze(0).repeat(states.shape[0], 1)
        return self.q_values[: states.shape[0]]


def _config() -> dict:
    return load_yaml_config(CONFIG_PATH)


def _batch(batch_size: int = 4) -> ReplayBatch:
    rng = np.random.default_rng(123)
    return ReplayBatch(
        states=rng.integers(0, 256, size=(batch_size, 4, 84, 84), dtype=np.uint8),
        actions=rng.integers(0, 7, size=(batch_size,), dtype=np.int64),
        rewards=rng.normal(size=(batch_size,)).astype(np.float32),
        next_states=rng.integers(0, 256, size=(batch_size, 4, 84, 84), dtype=np.uint8),
        dones=np.array([False, False, True, False][:batch_size], dtype=np.bool_),
    )


def _parameters_clone(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def test_online_and_target_start_equal_but_independent():
    agent = DDQNAgent(_config(), device="cpu", seed=42)

    for online_parameter, target_parameter in zip(agent.online_network.parameters(), agent.target_network.parameters()):
        assert torch.equal(online_parameter, target_parameter)
        assert online_parameter.data_ptr() != target_parameter.data_ptr()
        assert target_parameter.requires_grad is False


def test_epsilon_zero_uses_greedy_action(monkeypatch):
    agent = DDQNAgent(_config(), device="cpu", seed=42)
    agent.online_network = FixedQNetwork(torch.tensor([0.0, 1.0, 2.0, 10.0, 4.0, 5.0, 6.0]))
    monkeypatch.setattr(random, "random", lambda: 1.0)

    action = agent.select_action(np.zeros((4, 84, 84), dtype=np.uint8), epsilon=0.0)

    assert action == 3


def test_epsilon_one_uses_random_valid_action(monkeypatch):
    agent = DDQNAgent(_config(), device="cpu", seed=42)
    agent.online_network = FixedQNetwork(torch.tensor([0.0, 1.0, 2.0, 10.0, 4.0, 5.0, 6.0]))
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randrange", lambda upper: upper - 1)

    action = agent.select_action(np.zeros((4, 84, 84), dtype=np.uint8), epsilon=1.0)

    assert action == 6
    assert 0 <= action < 7


def test_ddqn_target_uses_online_selection_and_target_evaluation():
    agent = DDQNAgent(_config(), device="cpu", seed=42)
    agent.online_network = FixedQNetwork(
        torch.tensor(
            [
                [0.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 8.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    agent.target_network = FixedQNetwork(
        torch.tensor(
            [
                [0.0, 10.0, 50.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 20.0, 60.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    rewards = torch.tensor([1.0, 2.0])
    next_states = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
    dones = torch.tensor([0.0, 1.0])

    targets = agent.compute_ddqn_targets(rewards, next_states, dones)

    assert torch.allclose(targets, torch.tensor([1.0 + 0.99 * 10.0, 2.0]))
    assert not torch.allclose(targets[0], torch.tensor(1.0 + 0.99 * 50.0))


def test_update_changes_online_but_keeps_target_stable_then_syncs():
    agent = DDQNAgent(_config(), device="cpu", seed=42)
    online_before = _parameters_clone(agent.online_network)
    target_before = _parameters_clone(agent.target_network)

    metrics = agent.update(_batch())

    online_after = _parameters_clone(agent.online_network)
    target_after = _parameters_clone(agent.target_network)
    assert np.isfinite(metrics["loss"])
    assert any(not torch.equal(before, after) for before, after in zip(online_before, online_after))
    assert all(torch.equal(before, after) for before, after in zip(target_before, target_after))

    agent.sync_target_network()

    for online_parameter, target_parameter in zip(agent.online_network.parameters(), agent.target_network.parameters()):
        assert torch.equal(online_parameter, target_parameter)


def test_save_and_load_restores_predictions(tmp_path):
    agent = DDQNAgent(_config(), device="cpu", seed=42)
    state = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        expected = agent.online_network(state)
    checkpoint_path = tmp_path / "agent.pt"

    agent.save(checkpoint_path)
    loaded = DDQNAgent(_config(), device="cpu", seed=7)
    loaded.load(checkpoint_path)

    with torch.no_grad():
        actual = loaded.online_network(state)
    assert torch.allclose(expected, actual, atol=1e-6)
    assert loaded.optimizer.state_dict()["param_groups"][0]["lr"] == pytest.approx(agent.learning_rate)


def test_agent_consumes_real_assault_observation_for_valid_action():
    config = _config()
    env = create_assault_env(config, mode="train", seed=42)
    try:
        observation, _ = env.reset(seed=42)
        agent = DDQNAgent(config, device="cpu", seed=42)
        action = agent.select_action(observation, epsilon=0.0)
        q_values = agent.online_network(torch.as_tensor(observation).unsqueeze(0))

        assert observation.shape == (4, 84, 84)
        assert observation.dtype == np.uint8
        assert q_values.shape == (1, 7)
        assert 0 <= action < env.action_space.n
    finally:
        env.close()
