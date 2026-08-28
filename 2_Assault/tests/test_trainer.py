"""Tests for HU004 timestep-based training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
import torch
from torch import nn

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.agent import DDQNAgent
from src.environment import create_assault_env
from src.replay_buffer import ReplayBuffer
from src.trainer import Trainer, compute_epsilon
from src.utils import load_yaml_config


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


class FakeActionSpace:
    """Small action space compatible with the trainer tests."""

    n = 7

    def sample(self) -> int:
        return 0


class ScriptedEnv:
    """Environment fake that emits configured episode endings."""

    action_space = FakeActionSpace()

    def __init__(self, endings: Iterable[tuple[bool, bool]]) -> None:
        self.endings = list(endings)
        self.step_index = 0
        self.reset_count = 0
        self.stored_rewards = []

    def reset(self, seed: int | None = None):
        self.reset_count += 1
        return self._observation(), {"seed": seed}

    def step(self, action: int):
        terminated, truncated = self.endings[self.step_index]
        self.step_index += 1
        reward = float(self.step_index)
        self.stored_rewards.append(reward)
        return self._observation(), reward, terminated, truncated, {"action": action}

    def close(self) -> None:
        pass

    def _observation(self) -> np.ndarray:
        return np.full((4, 84, 84), self.step_index % 256, dtype=np.uint8)


class FakeAgent:
    """Agent fake that records temporal calls while exposing parameters."""

    def __init__(self) -> None:
        self.online_network = nn.Linear(1, 1, bias=False)
        self.target_network = nn.Linear(1, 1, bias=False)
        self.epsilons = []
        self.update_calls = []
        self.sync_calls = 0

    def select_action(self, state, epsilon: float) -> int:
        self.epsilons.append(epsilon)
        return 1

    def update(self, batch) -> dict[str, float]:
        self.update_calls.append(batch)
        with torch.no_grad():
            self.online_network.weight.add_(1.0)
        return {"loss": float(len(self.update_calls))}

    def sync_target_network(self) -> None:
        self.sync_calls += 1
        self.target_network.load_state_dict(self.online_network.state_dict())


def _config(**training_overrides) -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["replay_buffer"] = {"capacity": 64, "batch_size": 4}
    config["training"] = {
        "total_timesteps": 12,
        "learning_starts": 5,
        "train_frequency": 3,
        "target_update_frequency": 4,
        "epsilon_decay_steps": 10,
    }
    config["training"].update(training_overrides)
    return config


def _run_fake_training(config: dict, endings: list[tuple[bool, bool]] | None = None):
    env = ScriptedEnv(endings or [(False, False)] * int(config["training"]["total_timesteps"]))
    agent = FakeAgent()
    buffer = ReplayBuffer(capacity=int(config["replay_buffer"]["capacity"]), seed=123)
    summary = Trainer(env, agent, buffer, config).train()
    return summary, agent, buffer, env


def test_compute_epsilon_start_middle_final_and_bounds():
    assert compute_epsilon(0, 1.0, 0.1, 10) == pytest.approx(1.0)
    assert compute_epsilon(5, 1.0, 0.1, 10) == pytest.approx(0.55)
    assert compute_epsilon(10, 1.0, 0.1, 10) == pytest.approx(0.1)
    assert compute_epsilon(20, 1.0, 0.1, 10) == pytest.approx(0.1)
    for step in range(25):
        epsilon = compute_epsilon(step, 1.0, 0.1, 10)
        assert 0.1 <= epsilon <= 1.0


def test_trainer_stops_exactly_by_timesteps_and_stores_one_transition_per_step():
    summary, _, buffer, _ = _run_fake_training(_config())

    assert summary.global_step == 12
    assert summary.transitions_stored == 12
    assert summary.final_replay_buffer_size == 12
    assert len(buffer) == 12


def test_learning_starts_blocks_premature_updates():
    summary, agent, _, _ = _run_fake_training(_config(total_timesteps=4, learning_starts=5))

    assert summary.updates_count == 0
    assert agent.update_calls == []
    assert summary.last_loss is None


def test_train_frequency_and_batch_gate_control_updates():
    summary, _, _, _ = _run_fake_training(_config())

    assert summary.update_steps == [6, 9, 12]
    assert summary.first_update_step == 6
    assert summary.updates_count == 3
    assert summary.last_loss == pytest.approx(3.0)
    assert summary.mean_loss == pytest.approx(2.0)


def test_target_sync_steps_are_exactly_configured_multiples():
    summary, agent, _, _ = _run_fake_training(_config())

    assert summary.target_sync_steps == [4, 8, 12]
    assert agent.sync_calls == 3


def test_epsilon_schedule_is_used_by_agent():
    summary, agent, _, _ = _run_fake_training(_config())

    assert summary.epsilon_initial == pytest.approx(1.0)
    assert agent.epsilons[0] == pytest.approx(1.0)
    assert agent.epsilons[5] == pytest.approx(0.505)
    assert summary.epsilon_final == pytest.approx(0.01)


def test_terminated_and_truncated_reset_but_only_terminated_is_bootstrap_done():
    config = _config(total_timesteps=5, learning_starts=99)
    endings = [(False, False), (True, False), (False, True), (False, False), (False, False)]
    summary, _, buffer, env = _run_fake_training(config, endings=endings)

    sampled = buffer.sample(5)
    assert sorted(sampled.dones.astype(int).tolist()) == [0, 0, 0, 0, 1]
    assert summary.episodes_completed == 2
    assert summary.terminated_episodes == 1
    assert summary.truncated_episodes == 1
    assert summary.episode_end_reasons == ["terminated", "truncated"]
    assert env.reset_count == 3
    assert summary.global_step == 5


def test_training_summary_reports_metrics_and_online_weight_change():
    summary, _, _, _ = _run_fake_training(_config())

    metrics = summary.as_dict()
    for key in (
        "global_step",
        "episodes_completed",
        "episode_rewards",
        "episode_lengths",
        "epsilon_final",
        "transitions_stored",
        "updates_count",
        "last_loss",
        "mean_loss",
        "target_sync_steps",
    ):
        assert key in metrics
    assert summary.online_weights_changed is True
    assert np.isfinite(summary.last_loss)


def test_short_training_with_real_assault_runs_and_updates():
    config = load_yaml_config(CONFIG_PATH)
    config["replay_buffer"] = {"capacity": 64, "batch_size": 4}
    config["training"] = {
        "total_timesteps": 8,
        "learning_starts": 4,
        "train_frequency": 2,
        "target_update_frequency": 4,
        "epsilon_decay_steps": 8,
    }
    env = create_assault_env(config, mode="train", seed=42)
    try:
        agent = DDQNAgent(config, device="cpu", seed=42)
        buffer = ReplayBuffer(capacity=64, seed=42)
        summary = Trainer(env, agent, buffer, config).train()

        assert summary.global_step == 8
        assert summary.transitions_stored == 8
        assert summary.updates_count > 0
        assert np.isfinite(summary.last_loss)
        assert summary.online_weights_changed is True
        assert summary.target_sync_steps == [4, 8]
    finally:
        env.close()
