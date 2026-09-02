"""Focused tests for BattleZone DDQN agent core HU005."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
import torch

from src.agent import DDQNAgent


STATE_SHAPE = (4, 128, 128, 3)


def _state(fill_value: int) -> np.ndarray:
    return np.full(STATE_SHAPE, fill_value=fill_value, dtype=np.uint8)


def _batch(batch_size: int = 8) -> Dict[str, np.ndarray]:
    states = np.stack([_state(i) for i in range(batch_size)], axis=0)
    next_states = np.stack([_state(i + 1) for i in range(batch_size)], axis=0)
    actions = np.arange(batch_size, dtype=np.int64) % 18
    rewards = np.linspace(0.0, 1.0, num=batch_size, dtype=np.float32)
    dones = np.array([False] * (batch_size - 1) + [True], dtype=np.bool_)
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
    }


@pytest.fixture()
def agent() -> DDQNAgent:
    torch.manual_seed(7)
    np.random.seed(7)
    return DDQNAgent(
        action_dim=18,
        state_shape=STATE_SHAPE,
        gamma=0.99,
        learning_rate=1e-3,
        replay_buffer_capacity=32,
        batch_size=8,
        device="cpu",
    )


def test_online_and_target_start_synced_but_are_distinct_objects(agent: DDQNAgent):
    assert agent.online_network is not agent.target_network

    online_params = list(agent.online_network.parameters())
    target_params = list(agent.target_network.parameters())
    assert len(online_params) == len(target_params)
    for online_param, target_param in zip(online_params, target_params):
        assert torch.allclose(online_param, target_param)
        assert online_param.data_ptr() != target_param.data_ptr()


def test_select_action_epsilon_zero_uses_greedy(agent: DDQNAgent):
    with torch.no_grad():
        for param in agent.online_network.parameters():
            param.zero_()
        agent.online_network.output_layer.bias.copy_(torch.arange(18, dtype=torch.float32))

    action = agent.select_action(_state(10), epsilon=0.0)
    assert action == 17


def test_select_action_epsilon_one_returns_valid_random_actions(agent: DDQNAgent):
    np.random.seed(11)
    actions = [agent.select_action(_state(10), epsilon=1.0) for _ in range(64)]

    assert all(0 <= action < 18 for action in actions)
    assert len(set(actions)) > 1


def test_select_action_rejects_invalid_epsilon(agent: DDQNAgent):
    with pytest.raises(ValueError, match="epsilon"):
        agent.select_action(_state(1), epsilon=-0.1)

    with pytest.raises(ValueError, match="epsilon"):
        agent.select_action(_state(1), epsilon=1.1)


def test_ddqn_target_uses_online_selection_and_target_evaluation(agent: DDQNAgent):
    with torch.no_grad():
        for param in agent.online_network.parameters():
            param.zero_()
        for param in agent.target_network.parameters():
            param.zero_()

        # Online argmax -> action 3.
        agent.online_network.output_layer.bias.copy_(
            torch.tensor([0.0, 1.0, 2.0, 10.0] + [0.0] * 14, dtype=torch.float32)
        )
        # Target argmax is action 7 (value 50), but action 3 has value 4.
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
    expected_ddqn = 1.0 + agent.gamma * 4.0
    classic_dqn_wrong = 1.0 + agent.gamma * 50.0

    assert torch.allclose(targets.cpu(), torch.tensor([expected_ddqn], dtype=torch.float32))
    assert not torch.allclose(targets.cpu(), torch.tensor([classic_dqn_wrong], dtype=torch.float32))


def test_terminal_mask_applies_reward_only(agent: DDQNAgent):
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
        next_q = (
            agent.target_network(torch.from_numpy(np.stack([_state(3)], axis=0)))
            .gather(
                1,
                agent.online_network(torch.from_numpy(np.stack([_state(3)], axis=0)))
                .argmax(dim=1, keepdim=True),
            )
            .squeeze(1)
            .item()
        )
    expected_non_terminal = -1.0 + agent.gamma * next_q
    assert np.isclose(targets[1], expected_non_terminal, atol=1e-5)


def test_optimizer_updates_only_online_and_target_stays_immutable(agent: DDQNAgent):
    batch = _batch(batch_size=8)

    online_before = [param.detach().clone() for param in agent.online_network.parameters()]
    target_before = [param.detach().clone() for param in agent.target_network.parameters()]

    result = agent.update(batch)

    assert np.isfinite(result.loss)

    online_after = list(agent.online_network.parameters())
    target_after = list(agent.target_network.parameters())

    online_changed = any(
        not torch.allclose(before, after.detach())
        for before, after in zip(online_before, online_after)
    )
    assert online_changed is True

    target_unchanged = all(
        torch.allclose(before, after.detach())
        for before, after in zip(target_before, target_after)
    )
    assert target_unchanged is True

    assert all(param.grad is None for param in agent.target_network.parameters())


def test_sync_target_network_realigns_parameters(agent: DDQNAgent):
    batch = _batch(batch_size=8)
    agent.update(batch)

    mismatch_before_sync = any(
        not torch.allclose(online_param, target_param)
        for online_param, target_param in zip(
            agent.online_network.parameters(),
            agent.target_network.parameters(),
        )
    )
    assert mismatch_before_sync is True

    agent.sync_target_network()

    mismatch_after_sync = any(
        not torch.allclose(online_param, target_param)
        for online_param, target_param in zip(
            agent.online_network.parameters(),
            agent.target_network.parameters(),
        )
    )
    assert mismatch_after_sync is False


def test_state_dict_and_load_state_dict_restore_parameters():
    source_agent = DDQNAgent(device="cpu")
    target_agent = DDQNAgent(device="cpu")
    source_agent.update(_batch(batch_size=8))

    state = source_agent.state_dict()
    target_agent.load_state_dict(state)

    for source_param, target_param in zip(
        source_agent.online_network.parameters(), target_agent.online_network.parameters()
    ):
        assert torch.allclose(source_param, target_param)
    for source_param, target_param in zip(
        source_agent.target_network.parameters(), target_agent.target_network.parameters()
    ):
        assert torch.allclose(source_param, target_param)

    fixed_state = _state(12)
    source_action = source_agent.select_action(fixed_state, epsilon=0.0)
    target_action = target_agent.select_action(fixed_state, epsilon=0.0)
    assert source_action == target_action


def test_replay_buffer_integration_store_sample(agent: DDQNAgent):
    for idx in range(8):
        agent.store_transition(_state(idx), idx % 18, float(idx), _state(idx + 1), idx % 2 == 0)

    sampled = agent.sample_batch(batch_size=4)
    assert sampled["states"].shape == (4, *STATE_SHAPE)
    assert sampled["states"].dtype == np.uint8


def test_optimizer_tracks_only_online_parameters(agent: DDQNAgent):
    online_ids = {id(param) for param in agent.online_network.parameters()}
    target_ids = {id(param) for param in agent.target_network.parameters()}
    optimizer_ids = {
        id(param)
        for group in agent.optimizer.param_groups
        for param in group["params"]
    }

    assert optimizer_ids == online_ids
    assert optimizer_ids.isdisjoint(target_ids)
