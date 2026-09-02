"""Focused tests for BattleZone DQN agent core HU005."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
import torch

from src.agent import DQNAgent


STATE_SHAPE = (4, 128, 128, 3)
AGENT_KWARGS = {
    "action_dim": 18,
    "state_shape": STATE_SHAPE,
    "gamma": 0.99,
    "learning_rate": 1e-3,
    "replay_buffer_capacity": 32,
    "batch_size": 8,
    "network_hidden_dim": 512,
    "network_conv_channels": (32, 64, 64),
    "device": "cpu",
}


def _state(fill_value: int) -> np.ndarray:
    return np.full(STATE_SHAPE, fill_value=fill_value, dtype=np.uint8)


def _batch(batch_size: int = 8) -> Dict[str, np.ndarray]:
    return {
        "states": np.stack([_state(i) for i in range(batch_size)], axis=0),
        "actions": np.arange(batch_size, dtype=np.int64) % 18,
        "rewards": np.linspace(0.0, 1.0, num=batch_size, dtype=np.float32),
        "next_states": np.stack([_state(i + 1) for i in range(batch_size)], axis=0),
        "dones": np.array([False] * (batch_size - 1) + [True], dtype=np.bool_),
    }


@pytest.fixture()
def agent() -> DQNAgent:
    torch.manual_seed(7)
    np.random.seed(7)
    return DQNAgent(**AGENT_KWARGS)


def test_online_and_target_start_synced_but_are_distinct_objects(agent: DQNAgent):
    assert agent.online_network is not agent.target_network
    for online_param, target_param in zip(
        agent.online_network.parameters(), agent.target_network.parameters()
    ):
        assert torch.allclose(online_param, target_param)
        assert online_param.data_ptr() != target_param.data_ptr()


def test_select_action_epsilon_zero_uses_greedy(agent: DQNAgent):
    with torch.no_grad():
        for param in agent.online_network.parameters():
            param.zero_()
        agent.online_network.output_layer.bias.copy_(torch.arange(18, dtype=torch.float32))
    assert agent.select_action(_state(10), epsilon=0.0) == 17


def test_select_action_epsilon_one_returns_valid_random_actions(agent: DQNAgent):
    np.random.seed(11)
    actions = [agent.select_action(_state(10), epsilon=1.0) for _ in range(64)]
    assert all(0 <= action < 18 for action in actions)
    assert len(set(actions)) > 1


def test_select_action_rejects_invalid_epsilon(agent: DQNAgent):
    with pytest.raises(ValueError, match="epsilon"):
        agent.select_action(_state(1), epsilon=-0.1)
    with pytest.raises(ValueError, match="epsilon"):
        agent.select_action(_state(1), epsilon=1.1)


def test_dqn_target_uses_max_from_target_network(agent: DQNAgent):
    with torch.no_grad():
        for param in agent.online_network.parameters():
            param.zero_()
        for param in agent.target_network.parameters():
            param.zero_()
        # Online argmax intentionally differs from target argmax.
        agent.online_network.output_layer.bias.copy_(
            torch.tensor([0.0, 1.0, 2.0, 10.0] + [0.0] * 14, dtype=torch.float32)
        )
        agent.target_network.output_layer.bias.copy_(
            torch.tensor([0.0, 1.0, 2.0, 4.0, 0.0, 0.0, 0.0, 50.0] + [0.0] * 10, dtype=torch.float32)
        )

    batch = {
        "states": np.stack([_state(0)], axis=0),
        "actions": np.array([0], dtype=np.int64),
        "rewards": np.array([1.0], dtype=np.float32),
        "next_states": np.stack([_state(1)], axis=0),
        "dones": np.array([False], dtype=np.bool_),
    }
    targets = agent.compute_targets(batch)
    expected_dqn = 1.0 + agent.gamma * 50.0
    ddqn_reference = 1.0 + agent.gamma * 4.0
    assert torch.allclose(targets.cpu(), torch.tensor([expected_dqn], dtype=torch.float32))
    assert not torch.allclose(targets.cpu(), torch.tensor([ddqn_reference], dtype=torch.float32))


def test_terminal_mask_applies_reward_only(agent: DQNAgent):
    batch = {
        "states": np.stack([_state(0), _state(1)], axis=0),
        "actions": np.array([0, 1], dtype=np.int64),
        "rewards": np.array([2.5, -1.0], dtype=np.float32),
        "next_states": np.stack([_state(2), _state(3)], axis=0),
        "dones": np.array([True, False], dtype=np.bool_),
    }
    targets = agent.compute_targets(batch).cpu().numpy()
    assert np.isclose(targets[0], 2.5)
    with torch.no_grad():
        next_q = agent.target_network(
            torch.from_numpy(np.stack([_state(3)], axis=0))
        ).max(dim=1).values.item()
    assert np.isclose(targets[1], -1.0 + agent.gamma * next_q, atol=1e-5)


def test_optimizer_updates_only_online_and_target_stays_immutable(agent: DQNAgent):
    online_before = [param.detach().clone() for param in agent.online_network.parameters()]
    target_before = [param.detach().clone() for param in agent.target_network.parameters()]
    result = agent.update(_batch())
    assert np.isfinite(result.loss)
    assert np.isfinite(result.q_value_mean)
    assert any(
        not torch.allclose(before, after.detach())
        for before, after in zip(online_before, agent.online_network.parameters())
    )
    assert all(
        torch.allclose(before, after.detach())
        for before, after in zip(target_before, agent.target_network.parameters())
    )
    assert all(param.grad is None for param in agent.target_network.parameters())


def test_sync_target_network_realigns_parameters(agent: DQNAgent):
    agent.update(_batch())
    assert any(
        not torch.allclose(online_param, target_param)
        for online_param, target_param in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        )
    )
    agent.sync_target_network()
    assert all(
        torch.allclose(online_param, target_param)
        for online_param, target_param in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        )
    )


def test_update_result_contains_q_value_mean_without_changing_dqn_behavior(agent: DQNAgent):
    batch = _batch()
    update_result = agent.update(batch)
    assert np.isfinite(update_result.loss)
    assert np.isfinite(update_result.q_value_mean)


def test_state_dict_load_restores_gamma_and_parameters():
    source = DQNAgent(**AGENT_KWARGS)
    target = DQNAgent(**{**AGENT_KWARGS, "gamma": 0.5})
    source.update(_batch())
    target.load_state_dict(source.state_dict())
    assert target.gamma == source.gamma
    for source_param, target_param in zip(
        source.online_network.parameters(), target.online_network.parameters()
    ):
        assert torch.allclose(source_param, target_param)


def test_state_dict_load_rejects_structural_mismatch():
    source = DQNAgent(**AGENT_KWARGS)
    incompatible = DQNAgent(**{**AGENT_KWARGS, "batch_size": 4})
    with pytest.raises(ValueError, match="Incompatible agent state"):
        incompatible.load_state_dict(source.state_dict())


def test_from_config_uses_versioned_dqn_values():
    config = {
        "algorithm": "DQN",
        "environment": {"expected_action_space_n": 18},
        "validation": {"expected_final_shape": [4, 128, 128, 3]},
        "dqn": {
            "device": "cpu",
            "gamma": 0.95,
            "learning_rate": 0.0003,
            "batch_size": 4,
            "replay_buffer": {"capacity": 16},
            "network": {
                "hidden_dim": 256,
                "conv_channels": [16, 32, 32],
            },
        },
    }
    configured = DQNAgent.from_config(config)
    assert configured.gamma == 0.95
    assert configured.batch_size == 4
    assert configured.replay_buffer.capacity == 16
    assert configured.action_dim == 18


def test_replay_buffer_integration_store_sample(agent: DQNAgent):
    for idx in range(8):
        agent.store_transition(_state(idx), idx % 18, float(idx), _state(idx + 1), idx % 2 == 0)
    sampled = agent.sample_batch(batch_size=4)
    assert sampled["states"].shape == (4, *STATE_SHAPE)
    assert sampled["states"].dtype == np.uint8


def test_optimizer_tracks_only_online_parameters(agent: DQNAgent):
    online_ids = {id(param) for param in agent.online_network.parameters()}
    target_ids = {id(param) for param in agent.target_network.parameters()}
    optimizer_ids = {
        id(param)
        for group in agent.optimizer.param_groups
        for param in group["params"]
    }
    assert optimizer_ids == online_ids
    assert optimizer_ids.isdisjoint(target_ids)
