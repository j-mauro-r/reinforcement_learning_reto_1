"""Tests for HU006 TensorBoard observability."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.callbacks import TensorBoardLogger, load_tensorboard_scalars
from src.environment import create_assault_env
from src.replay_buffer import ReplayBuffer
from src.trainer import Trainer, compute_epsilon
from src.utils import load_yaml_config


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


class FakeEnv:
    def __init__(self, total_steps: int, episode_length: int = 3) -> None:
        self.total_steps = total_steps
        self.episode_length = episode_length
        self.step_index = 0
        self.episode_step = 0

    def reset(self, seed=None):
        self.episode_step = 0
        return self._obs(), {"seed": seed}

    def step(self, action: int):
        self.step_index += 1
        self.episode_step += 1
        terminated = self.episode_step >= self.episode_length
        return self._obs(), float(self.step_index), terminated, False, {"action": action}

    def close(self) -> None:
        pass

    def _obs(self) -> np.ndarray:
        return np.full((4, 84, 84), self.step_index % 256, dtype=np.uint8)


class FakeAgent:
    def __init__(self) -> None:
        self.online_network = nn.Linear(1, 1, bias=False)
        self.target_network = nn.Linear(1, 1, bias=False)
        self.epsilons: list[float] = []
        self.update_calls = 0

    def select_action(self, state, epsilon: float) -> int:
        self.epsilons.append(float(epsilon))
        return 0

    def update(self, batch) -> dict[str, float]:
        self.update_calls += 1
        with torch.no_grad():
            self.online_network.weight.add_(1.0)
        return {
            "loss": float(self.update_calls),
            "q_mean": 10.0 + float(self.update_calls),
            "learning_rate": 0.0001,
        }

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())


def _config(total_timesteps: int = 8) -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["replay_buffer"] = {"capacity": 64, "batch_size": 4}
    config["training"] = {
        "total_timesteps": total_timesteps,
        "learning_starts": 4,
        "train_frequency": 2,
        "target_update_frequency": 4,
        "epsilon_decay_steps": 8,
    }
    config["tensorboard"] = {
        "enabled": True,
        "directory": "logs/tensorboard",
        "log_frequency_steps": 4,
        "reward_window_episodes": 2,
        "flush_frequency_steps": 4,
    }
    return config


def _run_with_logger(tmp_path, run_id: str = "tb_run", initial_step: int = 0, total_timesteps: int = 8, enabled: bool = True):
    config = _config(total_timesteps=total_timesteps)
    config["tensorboard"]["enabled"] = enabled
    env = FakeEnv(total_steps=total_timesteps, episode_length=3)
    agent = FakeAgent()
    buffer = ReplayBuffer(capacity=64, seed=123)
    logger = TensorBoardLogger.from_config(config, run_id=run_id, log_root=tmp_path)
    try:
        summary = Trainer(
            env,
            agent,
            buffer,
            config,
            initial_global_step=initial_step,
            metrics_logger=logger,
        ).train()
        logger.flush()
    finally:
        logger.close()
    return summary, agent, logger


def test_logger_creates_run_id_directory_and_valid_event_file(tmp_path):
    _, _, logger = _run_with_logger(tmp_path, run_id="assault_ddqn_exp_001")

    assert logger.run_log_dir == tmp_path / "assault_ddqn_exp_001"
    assert logger.event_files()
    assert load_tensorboard_scalars(logger.run_log_dir)


def test_expected_tags_steps_and_finite_values_are_read_by_event_accumulator(tmp_path):
    summary, _, logger = _run_with_logger(tmp_path)
    scalars = load_tensorboard_scalars(logger.run_log_dir)

    expected_tags = {
        "train/epsilon",
        "train/loss",
        "train/q_mean",
        "train/learning_rate",
        "episode/reward",
        "episode/reward_mean",
        "episode/length",
    }
    assert expected_tags.issubset(scalars.keys())
    assert summary.update_steps == [4, 6, 8]
    for events in scalars.values():
        assert events
        assert all(math.isfinite(value) for _, value in events)


def test_epsilon_uses_action_value_and_post_step_global_axis(tmp_path):
    _, agent, logger = _run_with_logger(tmp_path)
    epsilon_events = load_tensorboard_scalars(logger.run_log_dir)["train/epsilon"]

    assert [step for step, _ in epsilon_events] == [4, 8]
    assert epsilon_events[0][1] == pytest.approx(compute_epsilon(3, 1.0, 0.01, 8))
    assert epsilon_events[1][1] == pytest.approx(compute_epsilon(7, 1.0, 0.01, 8))
    assert agent.epsilons[3] == pytest.approx(epsilon_events[0][1])


def test_loss_q_mean_and_learning_rate_are_logged_only_on_real_updates(tmp_path):
    _, _, logger = _run_with_logger(tmp_path)
    scalars = load_tensorboard_scalars(logger.run_log_dir)

    assert scalars["train/loss"] == [(4, pytest.approx(1.0)), (6, pytest.approx(2.0)), (8, pytest.approx(3.0))]
    assert [step for step, _ in scalars["train/q_mean"]] == [4, 6, 8]
    assert [step for step, _ in scalars["train/learning_rate"]] == [4, 6, 8]
    assert 5 not in [step for step, _ in scalars["train/loss"]]


def test_episode_reward_mean_window_and_length(tmp_path):
    _, _, logger = _run_with_logger(tmp_path)
    scalars = load_tensorboard_scalars(logger.run_log_dir)

    assert scalars["episode/reward"] == [(3, pytest.approx(6.0)), (6, pytest.approx(15.0))]
    assert scalars["episode/reward_mean"] == [(3, pytest.approx(6.0)), (6, pytest.approx(10.5))]
    assert scalars["episode/length"] == [(3, pytest.approx(3.0)), (6, pytest.approx(3.0))]


def test_resume_reuses_run_id_and_continues_global_steps(tmp_path):
    _run_with_logger(tmp_path, run_id="resume_run", total_timesteps=8)
    _, _, logger = _run_with_logger(tmp_path, run_id="resume_run", initial_step=8, total_timesteps=12)
    scalars = load_tensorboard_scalars(logger.run_log_dir)

    assert logger.run_log_dir == tmp_path / "resume_run"
    assert [step for step, _ in scalars["train/epsilon"]] == [4, 8, 12]
    assert all(step > 0 for step, _ in scalars["train/loss"])


def test_different_run_ids_do_not_mix_events(tmp_path):
    _run_with_logger(tmp_path, run_id="run_a")
    _run_with_logger(tmp_path, run_id="run_b", total_timesteps=4)

    run_a = load_tensorboard_scalars(tmp_path / "run_a")
    run_b = load_tensorboard_scalars(tmp_path / "run_b")

    assert (tmp_path / "run_a").exists()
    assert (tmp_path / "run_b").exists()
    assert len(run_a["train/loss"]) != len(run_b["train/loss"])


def test_disabled_tensorboard_creates_no_event_files_and_trainer_still_runs(tmp_path):
    summary, _, logger = _run_with_logger(tmp_path, enabled=False)

    assert summary.global_step == 8
    assert logger.event_files() == []
    assert not logger.run_log_dir.exists()


def test_trainer_still_works_with_logger_none():
    config = _config(total_timesteps=4)
    env = FakeEnv(total_steps=4, episode_length=2)
    agent = FakeAgent()
    buffer = ReplayBuffer(capacity=64, seed=123)

    summary = Trainer(env, agent, buffer, config, metrics_logger=None).train()

    assert summary.global_step == 4
    assert summary.updates_count == 1
    assert summary.last_loss == pytest.approx(1.0)


def test_short_real_assault_smoke_writes_training_scalars(tmp_path):
    config = _config(total_timesteps=8)
    env = create_assault_env(config, mode="train", seed=42)
    logger = TensorBoardLogger.from_config(config, run_id="assault_smoke", log_root=tmp_path)
    try:
        from src.agent import DDQNAgent

        agent = DDQNAgent(config, device="cpu", seed=42)
        buffer = ReplayBuffer(capacity=64, seed=42)
        summary = Trainer(env, agent, buffer, config, metrics_logger=logger).train()
        logger.flush()
    finally:
        logger.close()
        env.close()

    scalars = load_tensorboard_scalars(logger.run_log_dir)
    assert summary.global_step == 8
    assert {"train/epsilon", "train/loss", "train/q_mean", "train/learning_rate"}.issubset(scalars.keys())
