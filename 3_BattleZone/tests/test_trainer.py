"""Focused tests for BattleZone HU006 DQN training cycle."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from src.agent import DQNAgent
from src.environment import create_battlezone_env, load_config
from src.trainer import DQNTrainer, LinearEpsilonSchedule, TrainingMode, TrainingState


STATE_SHAPE = (4, 128, 128, 3)


class FakeActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def seed(self, seed: int) -> None:
        self._seed = seed


class FakeEnv:
    def __init__(
        self,
        *,
        rewards: List[float],
        terminated_steps: set[int] | None = None,
        truncated_steps: set[int] | None = None,
    ) -> None:
        self.action_space = FakeActionSpace(18)
        self._rewards = rewards
        self._terminated_steps = terminated_steps or set()
        self._truncated_steps = truncated_steps or set()
        self._cursor = 0
        self.reset_calls = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, Dict[str, Any]]:
        self._cursor = 0
        self.reset_calls += 1
        return np.zeros(STATE_SHAPE, dtype=np.uint8), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = float(self._rewards[self._cursor % len(self._rewards)])
        self._cursor += 1
        terminated = self._cursor in self._terminated_steps
        truncated = self._cursor in self._truncated_steps
        observation = np.full(STATE_SHAPE, fill_value=self._cursor % 255, dtype=np.uint8)
        return observation, reward, terminated, truncated, {}


@dataclass
class SpyUpdateResult:
    loss: float


class SpyAgent:
    def __init__(self, *, action_dim: int = 18, batch_size: int = 4) -> None:
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.replay_buffer: list[dict[str, Any]] = []
        self.select_calls: list[float] = []
        self.store_calls: list[dict[str, Any]] = []
        self.update_calls = 0
        self.sync_calls = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        self.select_calls.append(float(epsilon))
        return 0

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        item = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self.store_calls.append(item)
        self.replay_buffer.append(item)

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        chosen = self.replay_buffer[-batch_size:]
        return {
            "states": np.stack([x["state"] for x in chosen], axis=0),
            "actions": np.array([x["action"] for x in chosen], dtype=np.int64),
            "rewards": np.array([x["reward"] for x in chosen], dtype=np.float32),
            "next_states": np.stack([x["next_state"] for x in chosen], axis=0),
            "dones": np.array([x["done"] for x in chosen], dtype=np.bool_),
        }

    def update(self, batch: Dict[str, np.ndarray]) -> SpyUpdateResult:
        self.update_calls += 1
        return SpyUpdateResult(loss=0.25)

    def sync_target_network(self) -> None:
        self.sync_calls += 1


class InvalidActionAgent(SpyAgent):
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        return self.action_dim


def test_linear_epsilon_schedule_start_intermediate_end_and_clamp():
    schedule = LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10)
    assert schedule.value(0) == pytest.approx(1.0)
    assert schedule.value(5) == pytest.approx(0.55)
    assert schedule.value(10) == pytest.approx(0.1)
    assert schedule.value(1000) == pytest.approx(0.1)
    assert schedule.value(-5) == pytest.approx(1.0)


def test_no_update_before_learning_starts():
    env = FakeEnv(rewards=[1.0])
    agent = SpyAgent(batch_size=1)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=5,
        learning_starts=10,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    summary = trainer.train()
    assert summary.updates == 0
    assert agent.update_calls == 0


def test_replay_insufficient_blocks_updates_even_after_learning_starts():
    env = FakeEnv(rewards=[1.0])
    agent = SpyAgent(batch_size=8)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=6,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    summary = trainer.train()
    assert summary.updates == 0


def test_train_frequency_gates_updates_at_expected_steps():
    env = FakeEnv(rewards=[1.0])
    agent = SpyAgent(batch_size=1)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=10,
        learning_starts=1,
        train_frequency=3,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    summary = trainer.train()
    assert summary.update_steps == [3, 6, 9]
    assert summary.updates == 3


def test_target_sync_interval_records_expected_steps():
    env = FakeEnv(rewards=[1.0])
    agent = SpyAgent(batch_size=1)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=10,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=4,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    summary = trainer.train()
    assert summary.target_sync_steps == [4, 8]
    assert summary.target_syncs == 2


def test_replay_grows_and_summary_counts_steps():
    env = FakeEnv(rewards=[1.0])
    agent = SpyAgent(batch_size=1)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=7,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    summary = trainer.train()
    assert summary.total_steps == 7
    assert summary.replay_size == 7


def test_reward_passthrough_and_episode_reward_accumulation():
    env = FakeEnv(rewards=[1.0, -0.5, 2.0], terminated_steps={3})
    agent = SpyAgent(batch_size=1)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=3,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    summary = trainer.train()
    rewards_seen = [call["reward"] for call in agent.store_calls]
    assert rewards_seen == [1.0, -0.5, 2.0]
    assert summary.episode_rewards == [2.5]


def test_terminated_sets_done_for_bootstrap_true():
    env = FakeEnv(rewards=[1.0, 1.0], terminated_steps={2})
    agent = SpyAgent(batch_size=1)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=2,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    summary = trainer.train()
    assert agent.store_calls[-1]["done"] is True
    assert summary.terminated_episodes == 1
    assert summary.truncated_episodes == 0


def test_truncated_ends_episode_but_keeps_bootstrap_done_false():
    env = FakeEnv(rewards=[1.0, 1.0], truncated_steps={2})
    agent = SpyAgent(batch_size=1)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=2,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    summary = trainer.train()
    assert agent.store_calls[-1]["done"] is False
    assert summary.terminated_episodes == 0
    assert summary.truncated_episodes == 1


def test_episode_reset_and_counters():
    env = FakeEnv(rewards=[1.0], terminated_steps={1})
    agent = SpyAgent(batch_size=1)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=3,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    summary = trainer.train()
    assert summary.completed_episodes == 3
    assert summary.episode_lengths == [1, 1, 1]
    assert env.reset_calls == 4  # first reset + one per completed episode


def test_summary_contains_epsilon_endpoints_and_last_loss():
    env = FakeEnv(rewards=[1.0])
    agent = SpyAgent(batch_size=1)
    schedule = LinearEpsilonSchedule(start=1.0, end=0.2, decay_steps=10)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=5,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=schedule,
        seed=123,
    )

    summary = trainer.train()
    assert summary.initial_epsilon == pytest.approx(1.0)
    assert summary.final_epsilon == pytest.approx(schedule.value(4))
    assert summary.last_loss == pytest.approx(0.25)


def test_invalid_action_raises_explicit_error():
    env = FakeEnv(rewards=[1.0])
    agent = InvalidActionAgent(batch_size=1)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=1,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=10),
        seed=123,
    )

    with pytest.raises(ValueError, match="out of bounds"):
        trainer.train()


def test_from_config_builds_expected_schedule_and_gates():
    config = load_config()
    trainer = DQNTrainer.from_config(config=config, env=FakeEnv(rewards=[0.0]), agent=SpyAgent(batch_size=8))
    assert trainer.total_timesteps == int(config["training"]["total_timesteps"])
    assert trainer.learning_starts == int(config["training"]["learning_starts"])
    assert trainer.train_frequency == int(config["training"]["train_frequency"])
    assert trainer.target_sync_interval == int(config["training"]["target_sync_interval"])


def test_real_integration_short_run_with_dqn_agent_and_hu003_factory():
    config = load_config()
    env = create_battlezone_env(config, mode="train", seed=2026)
    agent = DQNAgent.from_config(config)
    trainer = DQNTrainer.from_config(
        config=config,
        env=env,
        agent=agent,
        seed=2026,
        total_timesteps=64,
        learning_starts=8,
        train_frequency=4,
        target_sync_interval=16,
    )

    try:
        summary = trainer.train()
    finally:
        env.close()

    assert summary.total_steps == 64
    assert summary.replay_size >= 8
    assert summary.updates >= 1
    assert summary.last_loss is not None
    assert np.isfinite(summary.last_loss)
    assert summary.target_syncs >= 1


def test_resume_continues_global_step_without_reset_and_preserves_epsilon_continuity():
    env = FakeEnv(rewards=[1.0])
    agent = SpyAgent(batch_size=1)
    schedule = LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=100)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=64,
        learning_starts=1,
        train_frequency=1,
        target_sync_interval=100,
        epsilon_schedule=schedule,
        seed=123,
    )

    initial_state = TrainingState(global_step=32, episode_index=5, episode_step=3, episode_reward=7.0)
    summary = trainer.train(
        total_timesteps=48,
        initial_state=initial_state,
        mode=TrainingMode.RESUME_FULL,
        replay_restored=True,
    )

    assert summary.start_global_step == 32
    assert summary.total_steps == 48
    assert summary.initial_epsilon == pytest.approx(schedule.value(32))
    assert summary.run_mode == "resume_full"
    assert summary.replay_restored is True


def test_resume_lightweight_keeps_replay_empty_until_rebuilt_then_updates_after_gate():
    env = FakeEnv(rewards=[1.0])
    agent = SpyAgent(batch_size=8)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        total_timesteps=256,
        learning_starts=1,
        train_frequency=4,
        target_sync_interval=100,
        epsilon_schedule=LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=200),
        seed=123,
    )

    summary = trainer.train(
        total_timesteps=45,
        initial_state=TrainingState(global_step=32, episode_index=0),
        mode=TrainingMode.RESUME_LIGHTWEIGHT,
        replay_restored=False,
    )

    assert summary.start_global_step == 32
    assert summary.first_update_step == 40
    assert summary.updates >= 1
    assert summary.run_mode == "resume_lightweight"
