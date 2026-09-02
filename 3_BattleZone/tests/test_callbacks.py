"""HU008 tests for TensorBoard observability callbacks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.agent import DQNAgent
from src.callbacks import TensorBoardTrainingLogger
from src.environment import load_config
from src.persistence import (
    CHECKPOINT_MODE_FULL,
    CHECKPOINT_MODE_LIGHTWEIGHT,
    build_checkpoint_metadata,
    build_checkpoint_payload,
    checkpoint_config_snapshot,
    restore_training_state,
    save_checkpoint,
)
from src.trainer import DQNTrainer, TrainingMode, TrainingState


STATE_SHAPE = (4, 128, 128, 3)


class FakeActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def seed(self, seed: int) -> None:
        self._seed = seed


class FakeEnv:
    def __init__(self, *, rewards: list[float], terminated_steps: set[int] | None = None) -> None:
        self.action_space = FakeActionSpace(18)
        self._rewards = rewards
        self._terminated_steps = terminated_steps or set()
        self._cursor = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, Dict[str, Any]]:
        self._cursor = 0
        return np.zeros(STATE_SHAPE, dtype=np.uint8), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = float(self._rewards[self._cursor % len(self._rewards)])
        self._cursor += 1
        terminated = self._cursor in self._terminated_steps
        next_observation = np.full(STATE_SHAPE, fill_value=self._cursor % 255, dtype=np.uint8)
        return next_observation, reward, terminated, False, {}


def _event_files(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("events.out.tfevents.*"))


def _acc(log_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return accumulator


def _scalar_steps(log_dir: Path, tag: str) -> list[int]:
    return [int(event.step) for event in _acc(log_dir).Scalars(tag)]


def _scalar_values(log_dir: Path, tag: str) -> list[float]:
    return [float(event.value) for event in _acc(log_dir).Scalars(tag)]


def _build_logger(config: Dict[str, Any], log_dir: Path) -> TensorBoardTrainingLogger:
    tb = config["tensorboard"]
    return TensorBoardTrainingLogger(
        log_dir=log_dir,
        reward_window=int(tb["reward_window"]),
        scalar_log_interval_steps=int(tb["scalar_log_interval_steps"]),
        flush_interval_steps=int(tb["flush_interval_steps"]),
    )


def _build_trainer_with_logger(
    config: Dict[str, Any],
    *,
    log_dir: Path,
    seed: int,
    terminated_steps: set[int] | None = None,
) -> tuple[DQNTrainer, DQNAgent]:
    logger = _build_logger(config, log_dir)
    env = FakeEnv(rewards=[1.0, -0.5, 2.0], terminated_steps=terminated_steps)
    agent = DQNAgent.from_config(config)
    trainer = DQNTrainer.from_config(
        config=config,
        env=env,
        agent=agent,
        seed=seed,
        total_timesteps=64,
        learning_starts=8,
        train_frequency=4,
        target_sync_interval=16,
        logger=logger,
    )
    return trainer, agent


def _save_checkpoint(path: Path, trainer: DQNTrainer, agent: DQNAgent, config: Dict[str, Any], mode: str) -> None:
    metadata = build_checkpoint_metadata(
        checkpoint_mode=mode,
        algorithm="DQN",
        global_step=trainer.training_state.global_step,
        episode_index=trainer.training_state.episode_index,
        seed=trainer.seed,
        state_shape=agent.state_shape,
        action_dim=agent.action_dim,
        batch_size=agent.batch_size,
        schema_version=int(config["checkpointing"]["schema_version"]),
    )
    replay = agent.replay_buffer.state_dict() if mode == CHECKPOINT_MODE_FULL else None
    payload = build_checkpoint_payload(
        metadata=metadata,
        trainer_state=trainer.export_training_state(),
        agent_state=agent.state_dict(),
        replay_buffer_state=replay,
        config_snapshot=checkpoint_config_snapshot(config),
    )
    save_checkpoint(checkpoint_path=path, payload=payload, allow_overwrite=False)


def test_logger_creates_directory_and_event_file(tmp_path: Path):
    log_dir = tmp_path / "logs"
    logger = TensorBoardTrainingLogger(
        log_dir=log_dir,
        reward_window=10,
        scalar_log_interval_steps=4,
        flush_interval_steps=8,
    )
    logger.on_step(global_step=4, epsilon=0.9, replay_size=4, learning_rate=0.00025)
    logger.close()

    files = _event_files(log_dir)
    assert log_dir.exists()
    assert len(files) >= 1
    assert all(path.stat().st_size > 100 for path in files)


def test_event_accumulator_reads_required_tags(tmp_path: Path):
    log_dir = tmp_path / "tags"
    logger = TensorBoardTrainingLogger(
        log_dir=log_dir,
        reward_window=10,
        scalar_log_interval_steps=4,
        flush_interval_steps=8,
    )
    logger.on_step(global_step=4, epsilon=0.9, replay_size=4, learning_rate=0.00025)
    logger.on_update(global_step=4, loss=0.5, q_value_mean=0.2)
    logger.on_episode_end(global_step=4, episode_reward=3.0, episode_length=4)
    logger.close()

    tags = _acc(log_dir).Tags()["scalars"]
    required = {
        "train/episode_reward",
        "train/episode_reward_mean",
        "train/episode_length",
        "train/loss",
        "train/q_value_mean",
        "train/epsilon",
        "train/replay_size",
        "train/learning_rate",
    }
    assert required.issubset(set(tags))


def test_reward_moving_average_window_semantics(tmp_path: Path):
    log_dir = tmp_path / "moving"
    logger = TensorBoardTrainingLogger(
        log_dir=log_dir,
        reward_window=2,
        scalar_log_interval_steps=4,
        flush_interval_steps=8,
    )
    logger.on_episode_end(global_step=1, episode_reward=10.0, episode_length=1)
    logger.on_episode_end(global_step=2, episode_reward=20.0, episode_length=1)
    logger.on_episode_end(global_step=3, episode_reward=30.0, episode_length=1)
    logger.close()

    means = _scalar_values(log_dir, "train/episode_reward_mean")
    assert means == pytest.approx([10.0, 15.0, 25.0])


def test_scalar_interval_for_epsilon_and_replay(tmp_path: Path):
    log_dir = tmp_path / "interval"
    logger = TensorBoardTrainingLogger(
        log_dir=log_dir,
        reward_window=10,
        scalar_log_interval_steps=4,
        flush_interval_steps=16,
    )
    for step in range(1, 13):
        logger.on_step(global_step=step, epsilon=1.0 - step / 100.0, replay_size=step, learning_rate=0.00025)
    logger.close()

    epsilon_steps = _scalar_steps(log_dir, "train/epsilon")
    replay_steps = _scalar_steps(log_dir, "train/replay_size")
    lr_steps = _scalar_steps(log_dir, "train/learning_rate")

    assert epsilon_steps == [4, 8, 12]
    assert replay_steps == [4, 8, 12]
    assert lr_steps == [4, 8, 12]


def test_close_is_idempotent(tmp_path: Path):
    logger = TensorBoardTrainingLogger(
        log_dir=tmp_path / "close",
        reward_window=10,
        scalar_log_interval_steps=4,
        flush_interval_steps=8,
    )
    logger.on_step(global_step=4, epsilon=0.8, replay_size=4, learning_rate=0.00025)
    logger.close()
    logger.close()


def test_trainer_with_logger_none_still_runs():
    config = load_config()
    env = FakeEnv(rewards=[1.0])
    agent = DQNAgent.from_config(config)
    trainer = DQNTrainer.from_config(
        config=config,
        env=env,
        agent=agent,
        seed=777,
        total_timesteps=16,
        learning_starts=8,
        train_frequency=4,
        target_sync_interval=8,
        logger=None,
    )
    summary = trainer.train(total_timesteps=16)
    assert summary.total_steps == 16


def test_resume_full_keeps_tensorboard_steps_continuous(tmp_path: Path):
    config = load_config()
    log_dir = tmp_path / "resume_full"

    trainer_new, agent_new = _build_trainer_with_logger(config, log_dir=log_dir, seed=111)
    summary_new = trainer_new.train(total_timesteps=16)

    checkpoint_path = tmp_path / "full_checkpoint.pt"
    _save_checkpoint(checkpoint_path, trainer_new, agent_new, config, CHECKPOINT_MODE_FULL)

    restored_agent = DQNAgent.from_config(config)
    resume_info = restore_training_state(
        checkpoint_path=checkpoint_path,
        agent=restored_agent,
        config=config,
        expected_mode=CHECKPOINT_MODE_FULL,
    )

    trainer_resume, _ = _build_trainer_with_logger(config, log_dir=log_dir, seed=112)
    trainer_resume.agent = restored_agent
    summary_resume = trainer_resume.train(
        total_timesteps=28,
        initial_state=TrainingState(
            global_step=int(resume_info["global_step"]),
            episode_index=int(resume_info["episode_index"]),
            episode_step=int(resume_info["episode_step"]),
            episode_reward=float(resume_info["episode_reward"]),
        ),
        mode=TrainingMode.RESUME_FULL,
        replay_restored=bool(resume_info["replay_restored"]),
    )

    epsilon_steps = _scalar_steps(log_dir, "train/epsilon")
    assert summary_new.total_steps == 16
    assert summary_resume.total_steps == 28
    assert max(epsilon_steps) == 28
    assert any(step > 16 for step in epsilon_steps)


def test_resume_lightweight_keeps_tensorboard_steps_continuous(tmp_path: Path):
    config = load_config()
    log_dir = tmp_path / "resume_light"

    trainer_new, agent_new = _build_trainer_with_logger(config, log_dir=log_dir, seed=121)
    summary_new = trainer_new.train(total_timesteps=16)

    checkpoint_path = tmp_path / "light_checkpoint.pt"
    _save_checkpoint(checkpoint_path, trainer_new, agent_new, config, CHECKPOINT_MODE_LIGHTWEIGHT)

    restored_agent = DQNAgent.from_config(config)
    resume_info = restore_training_state(
        checkpoint_path=checkpoint_path,
        agent=restored_agent,
        config=config,
        expected_mode=CHECKPOINT_MODE_LIGHTWEIGHT,
    )

    trainer_resume, _ = _build_trainer_with_logger(config, log_dir=log_dir, seed=122)
    trainer_resume.agent = restored_agent
    summary_resume = trainer_resume.train(
        total_timesteps=28,
        initial_state=TrainingState(global_step=int(resume_info["global_step"]), episode_index=int(resume_info["episode_index"])),
        mode=TrainingMode.RESUME_LIGHTWEIGHT,
        replay_restored=bool(resume_info["replay_restored"]),
    )

    epsilon_steps = _scalar_steps(log_dir, "train/epsilon")
    loss_values = _scalar_values(log_dir, "train/loss")
    q_values = _scalar_values(log_dir, "train/q_value_mean")

    assert summary_new.total_steps == 16
    assert summary_resume.total_steps == 28
    assert max(epsilon_steps) == 28
    assert any(step > 16 for step in epsilon_steps)
    assert all(math.isfinite(v) for v in loss_values)
    assert all(math.isfinite(v) for v in q_values)
