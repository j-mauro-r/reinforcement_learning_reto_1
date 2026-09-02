"""HU009 end-to-end smoke tests for BattleZone."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.agent import DQNAgent
from src.callbacks import TensorBoardTrainingLogger
from src.environment import create_battlezone_env, load_config
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
    def __init__(
        self,
        *,
        rewards: list[float],
        terminated_steps: set[int] | None = None,
        truncated_steps: set[int] | None = None,
    ) -> None:
        self.action_space = FakeActionSpace(18)
        self._rewards = rewards
        self._terminated_steps = terminated_steps or set()
        self._truncated_steps = truncated_steps or set()
        self._cursor = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, Dict[str, Any]]:
        self._cursor = 0
        return np.zeros(STATE_SHAPE, dtype=np.uint8), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = float(self._rewards[self._cursor % len(self._rewards)])
        self._cursor += 1
        terminated = self._cursor in self._terminated_steps
        truncated = self._cursor in self._truncated_steps
        next_observation = np.full(STATE_SHAPE, fill_value=self._cursor % 255, dtype=np.uint8)
        return next_observation, reward, terminated, truncated, {}


@dataclass(frozen=True)
class SmokeRunArtifacts:
    summary_new_total_steps: int
    summary_resume_total_steps: int
    summary_resume_start_step: int
    summary_new_updates: int
    summary_resume_updates: int
    summary_new_target_syncs: int
    summary_new_replay_size: int
    summary_resume_replay_size: int
    summary_new_last_loss: float | None
    summary_resume_last_loss: float | None
    online_weight_changed: bool
    checkpoint_path: Path
    checkpoint_size_bytes: int
    restored_global_step: int
    restored_replay_size: int
    online_target_aligned_after_new: bool
    optimizer_restored: bool
    resume_initial_epsilon: float
    max_logged_step: int
    any_logged_step_after_n: bool
    tags: list[str]
    counts: dict[str, int]


def _acc(log_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return accumulator


def _event_files(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("events.out.tfevents.*"))


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
    replay_state = agent.replay_buffer.state_dict() if mode == CHECKPOINT_MODE_FULL else None
    payload = build_checkpoint_payload(
        metadata=metadata,
        trainer_state=trainer.export_training_state(),
        agent_state=agent.state_dict(),
        replay_buffer_state=replay_state,
        config_snapshot=checkpoint_config_snapshot(config),
    )
    save_checkpoint(
        checkpoint_path=path,
        payload=payload,
        allow_overwrite=False,
    )


def _build_logger(config: Dict[str, Any], log_dir: Path) -> TensorBoardTrainingLogger:
    tb = config["tensorboard"]
    return TensorBoardTrainingLogger(
        log_dir=log_dir,
        reward_window=int(tb["reward_window"]),
        scalar_log_interval_steps=int(tb["scalar_log_interval_steps"]),
        flush_interval_steps=int(tb["flush_interval_steps"]),
    )


def _run_controlled_full_resume_smoke(config: Dict[str, Any], tmp_path: Path) -> SmokeRunArtifacts:
    smoke_cfg = config["smoke"]
    n_steps = int(smoke_cfg["total_timesteps_new"])
    m_steps = int(smoke_cfg["total_timesteps_resume"])
    assert m_steps > n_steps

    seed = int(config["environment"]["seed"])
    log_dir = tmp_path / "tb_full"
    checkpoint_path = tmp_path / "smoke_full.pt"

    env_new = FakeEnv(rewards=[1.0, -0.25, 0.5], terminated_steps={5, 10, 15, 20, 25, 30})
    agent_new = DQNAgent.from_config(config)
    before_online = [parameter.detach().clone() for parameter in agent_new.online_network.parameters()]
    logger_new = _build_logger(config, log_dir)
    trainer_new = DQNTrainer.from_config(
        config=config,
        env=env_new,
        agent=agent_new,
        seed=seed,
        total_timesteps=n_steps,
        learning_starts=int(smoke_cfg["learning_starts"]),
        train_frequency=int(smoke_cfg["train_frequency"]),
        target_sync_interval=int(smoke_cfg["target_sync_interval"]),
        logger=logger_new,
    )
    summary_new = trainer_new.train(total_timesteps=n_steps)

    changed_any = any(
        not torch.allclose(before, after)
        for before, after in zip(before_online, agent_new.online_network.parameters())
    )
    assert changed_any

    _save_checkpoint(checkpoint_path, trainer_new, agent_new, config, CHECKPOINT_MODE_FULL)

    online_target_aligned_after_new = all(
        torch.allclose(online_parameter, target_parameter)
        for online_parameter, target_parameter in zip(
            agent_new.online_network.parameters(),
            agent_new.target_network.parameters(),
        )
    )

    restored_agent = DQNAgent.from_config(config)
    resume_info = restore_training_state(
        checkpoint_path=checkpoint_path,
        agent=restored_agent,
        config=config,
        expected_mode=CHECKPOINT_MODE_FULL,
    )
    restored_replay_size_immediate = len(restored_agent.replay_buffer)

    for src, dst in zip(agent_new.online_network.parameters(), restored_agent.online_network.parameters()):
        assert torch.allclose(src, dst)
    for src, dst in zip(agent_new.target_network.parameters(), restored_agent.target_network.parameters()):
        assert torch.allclose(src, dst)

    optimizer_restored = bool(restored_agent.optimizer.state_dict().get("state"))

    env_resume = FakeEnv(rewards=[0.5, 0.2, -0.1], terminated_steps={7, 14, 21})
    logger_resume = _build_logger(config, log_dir)
    trainer_resume = DQNTrainer.from_config(
        config=config,
        env=env_resume,
        agent=restored_agent,
        seed=seed + 1,
        total_timesteps=m_steps,
        learning_starts=int(smoke_cfg["learning_starts"]),
        train_frequency=int(smoke_cfg["train_frequency"]),
        target_sync_interval=int(smoke_cfg["target_sync_interval"]),
        logger=logger_resume,
    )
    summary_resume = trainer_resume.train(
        total_timesteps=m_steps,
        initial_state=TrainingState(
            global_step=int(resume_info["global_step"]),
            episode_index=int(resume_info["episode_index"]),
            episode_step=int(resume_info["episode_step"]),
            episode_reward=float(resume_info["episode_reward"]),
        ),
        mode=TrainingMode.RESUME_FULL,
        replay_restored=bool(resume_info["replay_restored"]),
    )

    accumulator = _acc(log_dir)
    tags = sorted(accumulator.Tags().get("scalars", []))

    required_tags = [
        "train/loss",
        "train/q_value_mean",
        "train/epsilon",
        "train/replay_size",
        "train/learning_rate",
    ]
    counts = {tag: len(accumulator.Scalars(tag)) for tag in required_tags}
    epsilon_steps = [int(event.step) for event in accumulator.Scalars("train/epsilon")]
    max_logged_step = max(epsilon_steps) if epsilon_steps else 0

    return SmokeRunArtifacts(
        summary_new_total_steps=summary_new.total_steps,
        summary_resume_total_steps=summary_resume.total_steps,
        summary_resume_start_step=summary_resume.start_global_step,
        summary_new_updates=summary_new.updates,
        summary_resume_updates=summary_resume.updates,
        summary_new_target_syncs=summary_new.target_syncs,
        summary_new_replay_size=summary_new.replay_size,
        summary_resume_replay_size=summary_resume.replay_size,
        summary_new_last_loss=summary_new.last_loss,
        summary_resume_last_loss=summary_resume.last_loss,
        online_weight_changed=changed_any,
        checkpoint_path=checkpoint_path,
        checkpoint_size_bytes=checkpoint_path.stat().st_size,
        restored_global_step=int(resume_info["global_step"]),
        restored_replay_size=restored_replay_size_immediate,
        online_target_aligned_after_new=online_target_aligned_after_new,
        optimizer_restored=optimizer_restored,
        resume_initial_epsilon=summary_resume.initial_epsilon,
        max_logged_step=max_logged_step,
        any_logged_step_after_n=any(step > n_steps for step in epsilon_steps),
        tags=tags,
        counts=counts,
    )


def _build_real_ale_result(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    smoke_cfg = config["smoke"]
    tb_cfg = config["tensorboard"]
    n_steps = int(smoke_cfg["total_timesteps_new"])
    m_steps = int(smoke_cfg["total_timesteps_resume"])
    checkpoint_mode = str(smoke_cfg["checkpoint_mode"])
    seed = int(config["environment"]["seed"])

    if checkpoint_mode != CHECKPOINT_MODE_FULL:
        raise ValueError("HU009 real smoke requires smoke.checkpoint_mode='full'.")

    run_suffix = str(int(time.time()))
    log_dir = base_dir / "logs" / f"smoke_{run_suffix}"
    checkpoint_dir = base_dir / "checkpoints" / "smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"smoke_full_{run_suffix}.pt"

    env_new = create_battlezone_env(config, mode="train", seed=seed)
    observation, _ = env_new.reset(seed=seed)
    before_online = None

    try:
        agent_new = DQNAgent.from_config(config)
        before_online = [parameter.detach().clone() for parameter in agent_new.online_network.parameters()]

        logger_new = TensorBoardTrainingLogger(
            log_dir=log_dir,
            reward_window=int(tb_cfg["reward_window"]),
            scalar_log_interval_steps=int(tb_cfg["scalar_log_interval_steps"]),
            flush_interval_steps=int(tb_cfg["flush_interval_steps"]),
        )
        trainer_new = DQNTrainer.from_config(
            config=config,
            env=env_new,
            agent=agent_new,
            seed=seed,
            total_timesteps=n_steps,
            learning_starts=int(smoke_cfg["learning_starts"]),
            train_frequency=int(smoke_cfg["train_frequency"]),
            target_sync_interval=int(smoke_cfg["target_sync_interval"]),
            logger=logger_new,
        )
        start_new = time.perf_counter()
        summary_new = trainer_new.train(total_timesteps=n_steps)
        elapsed_new = time.perf_counter() - start_new

        _save_checkpoint(checkpoint_path, trainer_new, agent_new, config, CHECKPOINT_MODE_FULL)
        restored_agent = DQNAgent.from_config(config)
        resume_info = restore_training_state(
            checkpoint_path=checkpoint_path,
            agent=restored_agent,
            config=config,
            expected_mode=CHECKPOINT_MODE_FULL,
        )
        restored_replay_size_immediate = len(restored_agent.replay_buffer)

        env_resume = create_battlezone_env(config, mode="train", seed=seed + 1)
        try:
            logger_resume = TensorBoardTrainingLogger(
                log_dir=log_dir,
                reward_window=int(tb_cfg["reward_window"]),
                scalar_log_interval_steps=int(tb_cfg["scalar_log_interval_steps"]),
                flush_interval_steps=int(tb_cfg["flush_interval_steps"]),
            )
            trainer_resume = DQNTrainer.from_config(
                config=config,
                env=env_resume,
                agent=restored_agent,
                seed=seed + 1,
                total_timesteps=m_steps,
                learning_starts=int(smoke_cfg["learning_starts"]),
                train_frequency=int(smoke_cfg["train_frequency"]),
                target_sync_interval=int(smoke_cfg["target_sync_interval"]),
                logger=logger_resume,
            )
            start_resume = time.perf_counter()
            summary_resume = trainer_resume.train(
                total_timesteps=m_steps,
                initial_state=TrainingState(
                    global_step=int(resume_info["global_step"]),
                    episode_index=int(resume_info["episode_index"]),
                    episode_step=int(resume_info["episode_step"]),
                    episode_reward=float(resume_info["episode_reward"]),
                ),
                mode=TrainingMode.RESUME_FULL,
                replay_restored=bool(resume_info["replay_restored"]),
            )
            elapsed_resume = time.perf_counter() - start_resume
        finally:
            env_resume.close()

        online_weight_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(before_online, agent_new.online_network.parameters())
        )

        optimizer_restored = bool(restored_agent.optimizer.state_dict().get("state"))
        event_files = _event_files(log_dir)
        accumulator = _acc(log_dir)
        tags = sorted(accumulator.Tags().get("scalars", []))

        tracked_tags = [
            "train/loss",
            "train/q_value_mean",
            "train/epsilon",
            "train/replay_size",
            "train/learning_rate",
            "train/episode_reward",
            "train/episode_reward_mean",
            "train/episode_length",
        ]
        counts: dict[str, int] = {}
        max_logged_step = 0
        for tag in tracked_tags:
            if tag in tags:
                values = accumulator.Scalars(tag)
                counts[tag] = len(values)
                if values:
                    max_logged_step = max(max_logged_step, max(int(event.step) for event in values))
            else:
                counts[tag] = 0

        epsilon_steps = [int(event.step) for event in accumulator.Scalars("train/epsilon")]

        return {
            "n_steps": n_steps,
            "m_steps": m_steps,
            "seed": seed,
            "runtime": {
                "python": __import__("platform").python_version(),
                "gymnasium": __import__("gymnasium").__version__,
                "ale_py": __import__("ale_py").__version__,
                "torch": torch.__version__,
                "tensorboard": __import__("tensorboard").__version__,
            },
            "device": str(agent_new.device),
            "cuda_available": bool(torch.cuda.is_available()),
            "ram": "NO MEDIDO",
            "env_contract": {
                "env_id": config["environment"]["env_id"],
                "observation_shape": tuple(int(v) for v in observation.shape),
                "observation_dtype": str(observation.dtype),
                "action_dim": int(env_new.action_space.n),
                "frameskip": int(config["environment"]["frameskip"]),
                "repeat_action_probability": float(config["environment"]["repeat_action_probability"]),
            },
            "new": {
                "total_steps": summary_new.total_steps,
                "updates": summary_new.updates,
                "target_syncs": summary_new.target_syncs,
                "replay_size": summary_new.replay_size,
                "last_loss": summary_new.last_loss,
                "initial_epsilon": summary_new.initial_epsilon,
                "final_epsilon": summary_new.final_epsilon,
                "elapsed_seconds": elapsed_new,
                "online_weight_changed": online_weight_changed,
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            },
            "restore": {
                "global_step": int(resume_info["global_step"]),
                "replay_restored": bool(resume_info["replay_restored"]),
                "replay_size": restored_replay_size_immediate,
                "optimizer_restored": optimizer_restored,
            },
            "resume": {
                "start_global_step": summary_resume.start_global_step,
                "total_steps": summary_resume.total_steps,
                "updates": summary_resume.updates,
                "replay_size": summary_resume.replay_size,
                "last_loss": summary_resume.last_loss,
                "initial_epsilon": summary_resume.initial_epsilon,
                "elapsed_seconds": elapsed_resume,
            },
            "tensorboard": {
                "log_dir": str(log_dir),
                "event_files": [str(path) for path in event_files],
                "event_file_sizes": {path.name: path.stat().st_size for path in event_files},
                "tags": tags,
                "counts": counts,
                "max_logged_step": max_logged_step,
                "epsilon_steps_tail": epsilon_steps[-10:],
                "has_step_after_n": any(step > n_steps for step in epsilon_steps),
            },
        }
    finally:
        env_new.close()


def test_config_exposes_smoke_section_with_consistent_steps():
    config = load_config()
    smoke_cfg = config["smoke"]

    assert smoke_cfg["enabled"] is True
    assert int(smoke_cfg["total_timesteps_new"]) == 32
    assert int(smoke_cfg["total_timesteps_resume"]) == 48
    assert int(smoke_cfg["total_timesteps_resume"]) > int(smoke_cfg["total_timesteps_new"])
    assert str(smoke_cfg["checkpoint_mode"]) == CHECKPOINT_MODE_FULL


def test_smoke_controlled_full_new_checkpoint_restore_resume(tmp_path: Path):
    config = load_config()
    artifacts = _run_controlled_full_resume_smoke(config, tmp_path)
    n_steps = int(config["smoke"]["total_timesteps_new"])
    m_steps = int(config["smoke"]["total_timesteps_resume"])

    assert artifacts.summary_new_total_steps == n_steps
    assert artifacts.summary_new_updates > 0
    assert artifacts.summary_new_target_syncs > 0
    assert artifacts.summary_new_replay_size > 0
    assert artifacts.summary_new_last_loss is not None
    assert math.isfinite(float(artifacts.summary_new_last_loss))
    assert artifacts.online_weight_changed is True
    assert artifacts.online_target_aligned_after_new is True

    assert artifacts.checkpoint_path.exists()
    assert artifacts.checkpoint_size_bytes > 0
    assert artifacts.restored_global_step == n_steps
    assert artifacts.restored_replay_size > 0
    assert artifacts.optimizer_restored is True

    assert artifacts.summary_resume_start_step == n_steps
    assert artifacts.summary_resume_total_steps == m_steps
    assert artifacts.summary_resume_updates > 0
    assert artifacts.summary_resume_replay_size > 0

    expected_resume_epsilon = DQNTrainer.from_config(
        config=config,
        env=FakeEnv(rewards=[1.0]),
        agent=DQNAgent.from_config(config),
        seed=int(config["environment"]["seed"]),
        total_timesteps=m_steps,
        learning_starts=int(config["smoke"]["learning_starts"]),
        train_frequency=int(config["smoke"]["train_frequency"]),
        target_sync_interval=int(config["smoke"]["target_sync_interval"]),
        logger=None,
    ).epsilon_schedule.value(n_steps)
    assert artifacts.resume_initial_epsilon == pytest.approx(expected_resume_epsilon)

    required_tags = {
        "train/loss",
        "train/q_value_mean",
        "train/epsilon",
        "train/replay_size",
        "train/learning_rate",
    }
    assert required_tags.issubset(set(artifacts.tags))
    assert artifacts.counts["train/loss"] > 0
    assert artifacts.counts["train/q_value_mean"] > 0
    assert artifacts.counts["train/epsilon"] > 0
    assert artifacts.counts["train/replay_size"] > 0
    assert artifacts.max_logged_step > n_steps
    assert artifacts.any_logged_step_after_n is True


def test_smoke_controlled_lightweight_restore_replay_gate_and_resume(tmp_path: Path):
    config = load_config()
    smoke_cfg = config["smoke"]
    n_steps = int(smoke_cfg["total_timesteps_new"])
    m_steps = int(smoke_cfg["total_timesteps_resume"])
    seed = int(config["environment"]["seed"])

    env_new = FakeEnv(rewards=[1.0, -0.25, 0.5])
    agent_new = DQNAgent.from_config(config)
    trainer_new = DQNTrainer.from_config(
        config=config,
        env=env_new,
        agent=agent_new,
        seed=seed,
        total_timesteps=n_steps,
        learning_starts=int(smoke_cfg["learning_starts"]),
        train_frequency=int(smoke_cfg["train_frequency"]),
        target_sync_interval=int(smoke_cfg["target_sync_interval"]),
        logger=None,
    )
    trainer_new.train(total_timesteps=n_steps)

    checkpoint_path = tmp_path / "smoke_light.pt"
    _save_checkpoint(checkpoint_path, trainer_new, agent_new, config, CHECKPOINT_MODE_LIGHTWEIGHT)

    restored_agent = DQNAgent.from_config(config)
    resume_info = restore_training_state(
        checkpoint_path=checkpoint_path,
        agent=restored_agent,
        config=config,
        expected_mode=CHECKPOINT_MODE_LIGHTWEIGHT,
    )

    assert int(resume_info["global_step"]) == n_steps
    assert bool(resume_info["replay_restored"]) is False
    assert len(restored_agent.replay_buffer) == 0

    env_resume = FakeEnv(rewards=[0.2, -0.1, 0.3])
    trainer_resume = DQNTrainer.from_config(
        config=config,
        env=env_resume,
        agent=restored_agent,
        seed=seed + 1,
        total_timesteps=m_steps,
        learning_starts=int(smoke_cfg["learning_starts"]),
        train_frequency=int(smoke_cfg["train_frequency"]),
        target_sync_interval=int(smoke_cfg["target_sync_interval"]),
        logger=None,
    )

    summary_resume = trainer_resume.train(
        total_timesteps=m_steps,
        initial_state=TrainingState(
            global_step=int(resume_info["global_step"]),
            episode_index=int(resume_info["episode_index"]),
        ),
        mode=TrainingMode.RESUME_LIGHTWEIGHT,
        replay_restored=False,
    )

    assert summary_resume.start_global_step == n_steps
    assert summary_resume.total_steps == m_steps
    assert summary_resume.first_update_step == 40
    assert summary_resume.updates > 0


def test_smoke_trainer_supports_logger_none_without_tensorboard_dependency():
    config = load_config()
    smoke_cfg = config["smoke"]
    env = FakeEnv(rewards=[1.0])
    agent = DQNAgent.from_config(config)
    trainer = DQNTrainer.from_config(
        config=config,
        env=env,
        agent=agent,
        seed=int(config["environment"]["seed"]),
        total_timesteps=int(smoke_cfg["total_timesteps_new"]),
        learning_starts=int(smoke_cfg["learning_starts"]),
        train_frequency=int(smoke_cfg["train_frequency"]),
        target_sync_interval=int(smoke_cfg["target_sync_interval"]),
        logger=None,
    )

    summary = trainer.train(total_timesteps=int(smoke_cfg["total_timesteps_new"]))
    assert summary.total_steps == int(smoke_cfg["total_timesteps_new"])


def test_smoke_real_ale_via_factory_when_explicitly_enabled(tmp_path: Path):
    if not bool(int(__import__("os").environ.get("BATTLEZONE_RUN_REAL_SMOKE", "0"))):
        pytest.skip("Set BATTLEZONE_RUN_REAL_SMOKE=1 to run real ALE smoke.")

    config = load_config()
    result = _build_real_ale_result(config, tmp_path)

    assert tuple(result["env_contract"]["observation_shape"]) == (4, 128, 128, 3)
    assert result["env_contract"]["observation_dtype"] == "uint8"
    assert int(result["env_contract"]["action_dim"]) == 18
    assert result["new"]["updates"] > 0
    assert result["new"]["target_syncs"] > 0
    assert result["new"]["replay_size"] > 0
    assert result["new"]["last_loss"] is not None
    assert math.isfinite(float(result["new"]["last_loss"]))
    assert result["restore"]["global_step"] == int(config["smoke"]["total_timesteps_new"])
    assert result["restore"]["replay_restored"] is True
    assert result["restore"]["replay_size"] > 0
    assert result["resume"]["start_global_step"] == int(config["smoke"]["total_timesteps_new"])
    assert result["resume"]["total_steps"] == int(config["smoke"]["total_timesteps_resume"])
    assert result["tensorboard"]["counts"]["train/loss"] > 0
    assert result["tensorboard"]["counts"]["train/q_value_mean"] > 0
    assert result["tensorboard"]["counts"]["train/epsilon"] > 0
    assert result["tensorboard"]["counts"]["train/replay_size"] > 0
    assert result["tensorboard"]["max_logged_step"] > int(config["smoke"]["total_timesteps_new"])
