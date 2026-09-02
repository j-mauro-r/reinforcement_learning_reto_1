"""HU007 tests for checkpoint persistence, resume and idempotence."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from src.agent import DQNAgent
from src.environment import load_config
from src.persistence import (
    CHECKPOINT_MODE_FULL,
    CHECKPOINT_MODE_LIGHTWEIGHT,
    build_checkpoint_metadata,
    build_checkpoint_payload,
    checkpoint_config_snapshot,
    load_checkpoint,
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
    def __init__(self, *, rewards: list[float]) -> None:
        self.action_space = FakeActionSpace(18)
        self._rewards = rewards
        self._cursor = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, Dict[str, Any]]:
        self._cursor = 0
        return np.zeros(STATE_SHAPE, dtype=np.uint8), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = float(self._rewards[self._cursor % len(self._rewards)])
        self._cursor += 1
        next_observation = np.full(STATE_SHAPE, fill_value=self._cursor % 255, dtype=np.uint8)
        return next_observation, reward, False, False, {}


def _trainer_and_agent(config: Dict[str, Any], *, seed: int = 123) -> tuple[DQNTrainer, DQNAgent]:
    agent = DQNAgent.from_config(config)
    trainer = DQNTrainer.from_config(
        config=config,
        env=FakeEnv(rewards=[1.0]),
        agent=agent,
        seed=seed,
        total_timesteps=256,
        learning_starts=8,
        train_frequency=4,
        target_sync_interval=16,
    )
    return trainer, agent


def _save_mode_checkpoint(path: Path, trainer: DQNTrainer, agent: DQNAgent, config: Dict[str, Any], mode: str) -> None:
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
        allow_overwrite=bool(config["checkpointing"]["allow_overwrite"]),
    )


def test_directory_creation_and_full_checkpoint_save(tmp_path: Path):
    config = load_config()
    trainer, agent = _trainer_and_agent(config)
    trainer.train(total_timesteps=16)

    path = tmp_path / "nested" / "checkpoint_full.pt"
    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)

    assert path.exists()
    assert path.stat().st_size > 0


def test_lightweight_checkpoint_save_and_load(tmp_path: Path):
    config = load_config()
    trainer, agent = _trainer_and_agent(config)
    trainer.train(total_timesteps=16)

    path = tmp_path / "checkpoint_light.pt"
    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_LIGHTWEIGHT)

    payload = load_checkpoint(checkpoint_path=path, map_location="cpu")
    assert payload["metadata"].checkpoint_mode == CHECKPOINT_MODE_LIGHTWEIGHT
    assert payload["replay_buffer_state"] is None


def test_load_requires_explicit_file_path(tmp_path: Path):
    with pytest.raises(ValueError, match="file path"):
        load_checkpoint(checkpoint_path=tmp_path)


def test_load_missing_file_raises_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(checkpoint_path=tmp_path / "missing.pt")


def test_overwrite_blocked_by_default(tmp_path: Path):
    config = load_config()
    trainer, agent = _trainer_and_agent(config)
    trainer.train(total_timesteps=16)
    path = tmp_path / "checkpoint.pt"

    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)
    with pytest.raises(FileExistsError):
        _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)


def test_overwrite_explicit_allowed(tmp_path: Path):
    config = load_config()
    config["checkpointing"]["allow_overwrite"] = True
    trainer, agent = _trainer_and_agent(config)
    trainer.train(total_timesteps=16)
    path = tmp_path / "checkpoint.pt"

    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)
    trainer.train(total_timesteps=20)
    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)

    payload = load_checkpoint(checkpoint_path=path)
    assert payload["metadata"].global_step == 20


def test_invalid_schema_version_raises_error(tmp_path: Path):
    config = load_config()
    trainer, agent = _trainer_and_agent(config)
    trainer.train(total_timesteps=16)

    path = tmp_path / "bad_schema.pt"
    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)
    corrupted = torch.load(path, map_location="cpu", weights_only=False)
    corrupted["schema_version"] = 999
    torch.save(corrupted, path)

    with pytest.raises(ValueError, match="schema_version"):
        load_checkpoint(checkpoint_path=path)


def test_wrong_algorithm_rejected_on_restore(tmp_path: Path):
    config = load_config()
    trainer, agent = _trainer_and_agent(config)
    trainer.train(total_timesteps=16)

    path = tmp_path / "wrong_algo.pt"
    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)
    corrupted = torch.load(path, map_location="cpu", weights_only=False)
    corrupted["metadata"]["algorithm"] = "DDQN"
    torch.save(corrupted, path)

    with pytest.raises(ValueError, match="algorithm"):
        restore_training_state(
            checkpoint_path=path,
            agent=DQNAgent.from_config(config),
            config=config,
            expected_mode=CHECKPOINT_MODE_FULL,
        )


def test_structural_mismatch_rejected_on_restore(tmp_path: Path):
    config = load_config()
    trainer, agent = _trainer_and_agent(config)
    trainer.train(total_timesteps=16)
    path = tmp_path / "mismatch.pt"
    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)

    incompatible = copy.deepcopy(config)
    incompatible["dqn"]["batch_size"] = 4
    incompatible_agent = DQNAgent.from_config(incompatible)
    with pytest.raises(ValueError, match="batch_size"):
        restore_training_state(
            checkpoint_path=path,
            agent=incompatible_agent,
            config=incompatible,
            expected_mode=CHECKPOINT_MODE_FULL,
        )


def test_full_restore_recovers_online_target_optimizer_and_replay(tmp_path: Path):
    config = load_config()
    trainer, agent = _trainer_and_agent(config)
    summary = trainer.train(total_timesteps=32)
    replay_size_before = len(agent.replay_buffer)

    path = tmp_path / "full.pt"
    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)

    restored_agent = DQNAgent.from_config(config)
    resume_info = restore_training_state(
        checkpoint_path=path,
        agent=restored_agent,
        config=config,
        expected_mode=CHECKPOINT_MODE_FULL,
    )

    assert resume_info["global_step"] == 32
    assert resume_info["replay_restored"] is True
    assert len(restored_agent.replay_buffer) == replay_size_before

    for online_src, online_dst in zip(agent.online_network.parameters(), restored_agent.online_network.parameters()):
        assert torch.allclose(online_src, online_dst)
    for target_src, target_dst in zip(agent.target_network.parameters(), restored_agent.target_network.parameters()):
        assert torch.allclose(target_src, target_dst)

    known_state = agent.replay_buffer.states[0].copy()
    assert np.array_equal(restored_agent.replay_buffer.states[0], known_state)

    restored_trainer = DQNTrainer.from_config(
        config=config,
        env=FakeEnv(rewards=[1.0]),
        agent=restored_agent,
        seed=int(resume_info["seed"]),
        total_timesteps=256,
        learning_starts=8,
        train_frequency=4,
        target_sync_interval=16,
    )
    resumed = restored_trainer.train(
        total_timesteps=48,
        initial_state=TrainingState(
            global_step=int(resume_info["global_step"]),
            episode_index=int(resume_info["episode_index"]),
            episode_step=int(resume_info["episode_step"]),
            episode_reward=float(resume_info["episode_reward"]),
        ),
        mode=TrainingMode.RESUME_FULL,
        replay_restored=bool(resume_info["replay_restored"]),
    )
    assert resumed.start_global_step == summary.total_steps
    assert resumed.total_steps == 48


def test_lightweight_restore_keeps_empty_replay_and_delays_updates_until_gate(tmp_path: Path):
    config = load_config()
    trainer, agent = _trainer_and_agent(config)
    trainer.train(total_timesteps=32)

    path = tmp_path / "light.pt"
    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_LIGHTWEIGHT)

    restored_agent = DQNAgent.from_config(config)
    resume_info = restore_training_state(
        checkpoint_path=path,
        agent=restored_agent,
        config=config,
        expected_mode=CHECKPOINT_MODE_LIGHTWEIGHT,
    )
    assert len(restored_agent.replay_buffer) == 0
    assert resume_info["global_step"] == 32

    resumed_trainer = DQNTrainer.from_config(
        config=config,
        env=FakeEnv(rewards=[1.0]),
        agent=restored_agent,
        seed=int(resume_info["seed"]),
        total_timesteps=256,
        learning_starts=8,
        train_frequency=4,
        target_sync_interval=16,
    )
    resumed = resumed_trainer.train(
        total_timesteps=45,
        initial_state=TrainingState(global_step=32, episode_index=int(resume_info["episode_index"])),
        mode=TrainingMode.RESUME_LIGHTWEIGHT,
        replay_restored=False,
    )

    expected_epsilon = resumed_trainer.epsilon_schedule.value(32)
    assert resumed.initial_epsilon == pytest.approx(expected_epsilon)
    assert resumed.first_update_step == 40


def test_checkpoint_source_unchanged_after_load(tmp_path: Path):
    config = load_config()
    trainer, agent = _trainer_and_agent(config)
    trainer.train(total_timesteps=16)
    path = tmp_path / "stable.pt"
    _save_mode_checkpoint(path, trainer, agent, config, CHECKPOINT_MODE_FULL)

    before = path.read_bytes()
    before_hash = hashlib.sha256(before).hexdigest()
    before_size = path.stat().st_size

    _ = load_checkpoint(checkpoint_path=path)

    after_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    after_size = path.stat().st_size
    assert before_hash == after_hash
    assert before_size == after_size


def test_config_snapshot_compatible_restore_passes(tmp_path: Path):
    base_config = load_config()
    trainer, agent = _trainer_and_agent(base_config)
    trainer.train(total_timesteps=16)
    checkpoint = tmp_path / "compatible.pt"
    _save_mode_checkpoint(checkpoint, trainer, agent, base_config, CHECKPOINT_MODE_FULL)

    restored = DQNAgent.from_config(base_config)
    resume_info = restore_training_state(
        checkpoint_path=checkpoint,
        agent=restored,
        config=base_config,
        expected_mode=CHECKPOINT_MODE_FULL,
    )
    assert resume_info["global_step"] == 16


def test_config_snapshot_mismatch_gamma_raises_error(tmp_path: Path):
    base_config = load_config()
    trainer, agent = _trainer_and_agent(base_config)
    trainer.train(total_timesteps=16)
    checkpoint = tmp_path / "gamma_mismatch.pt"
    _save_mode_checkpoint(checkpoint, trainer, agent, base_config, CHECKPOINT_MODE_FULL)

    active_config = copy.deepcopy(base_config)
    active_config["dqn"]["gamma"] = 0.95

    with pytest.raises(ValueError, match="dqn.gamma"):
        restore_training_state(
            checkpoint_path=checkpoint,
            agent=DQNAgent.from_config(base_config),
            config=active_config,
            expected_mode=CHECKPOINT_MODE_FULL,
        )


def test_config_snapshot_mismatch_frameskip_raises_error(tmp_path: Path):
    base_config = load_config()
    trainer, agent = _trainer_and_agent(base_config)
    trainer.train(total_timesteps=16)
    checkpoint = tmp_path / "frameskip_mismatch.pt"
    _save_mode_checkpoint(checkpoint, trainer, agent, base_config, CHECKPOINT_MODE_FULL)

    active_config = copy.deepcopy(base_config)
    active_config["environment"]["frameskip"] = 5

    with pytest.raises(ValueError, match="environment.frameskip"):
        restore_training_state(
            checkpoint_path=checkpoint,
            agent=DQNAgent.from_config(base_config),
            config=active_config,
            expected_mode=CHECKPOINT_MODE_FULL,
        )


def test_config_snapshot_mismatch_sticky_actions_raises_error(tmp_path: Path):
    base_config = load_config()
    trainer, agent = _trainer_and_agent(base_config)
    trainer.train(total_timesteps=16)
    checkpoint = tmp_path / "sticky_mismatch.pt"
    _save_mode_checkpoint(checkpoint, trainer, agent, base_config, CHECKPOINT_MODE_FULL)

    active_config = copy.deepcopy(base_config)
    active_config["environment"]["repeat_action_probability"] = 0.10

    with pytest.raises(ValueError, match="environment.repeat_action_probability"):
        restore_training_state(
            checkpoint_path=checkpoint,
            agent=DQNAgent.from_config(base_config),
            config=active_config,
            expected_mode=CHECKPOINT_MODE_FULL,
        )


def test_config_snapshot_mismatch_replay_capacity_raises_error(tmp_path: Path):
    base_config = load_config()
    trainer, agent = _trainer_and_agent(base_config)
    trainer.train(total_timesteps=16)
    checkpoint = tmp_path / "capacity_mismatch.pt"
    _save_mode_checkpoint(checkpoint, trainer, agent, base_config, CHECKPOINT_MODE_FULL)

    active_config = copy.deepcopy(base_config)
    active_config["dqn"]["replay_buffer"]["capacity"] = 256

    with pytest.raises(ValueError, match="dqn.replay_buffer.capacity"):
        restore_training_state(
            checkpoint_path=checkpoint,
            agent=DQNAgent.from_config(base_config),
            config=active_config,
            expected_mode=CHECKPOINT_MODE_FULL,
        )


@pytest.mark.parametrize(
    "field_path,new_value,expected_match",
    [
        (("training", "epsilon", "start"), 0.8, "training.epsilon.start"),
        (("training", "epsilon", "end"), 0.01, "training.epsilon.end"),
        (("training", "epsilon", "decay_steps"), 2048, "training.epsilon.decay_steps"),
    ],
)
def test_config_snapshot_mismatch_epsilon_fields_raise_error(
    tmp_path: Path,
    field_path: tuple[str, ...],
    new_value: float | int,
    expected_match: str,
):
    base_config = load_config()
    trainer, agent = _trainer_and_agent(base_config)
    trainer.train(total_timesteps=16)
    checkpoint = tmp_path / "epsilon_mismatch.pt"
    _save_mode_checkpoint(checkpoint, trainer, agent, base_config, CHECKPOINT_MODE_FULL)

    active_config = copy.deepcopy(base_config)
    node = active_config
    for key in field_path[:-1]:
        node = node[key]
    node[field_path[-1]] = new_value

    with pytest.raises(ValueError, match=expected_match):
        restore_training_state(
            checkpoint_path=checkpoint,
            agent=DQNAgent.from_config(base_config),
            config=active_config,
            expected_mode=CHECKPOINT_MODE_FULL,
        )


def test_config_snapshot_mismatch_env_id_raises_error(tmp_path: Path):
    base_config = load_config()
    trainer, agent = _trainer_and_agent(base_config)
    trainer.train(total_timesteps=16)
    checkpoint = tmp_path / "env_id_mismatch.pt"
    _save_mode_checkpoint(checkpoint, trainer, agent, base_config, CHECKPOINT_MODE_FULL)

    active_config = copy.deepcopy(base_config)
    active_config["environment"]["env_id"] = "ALE/Pong-v5"

    with pytest.raises(ValueError, match="environment.env_id"):
        restore_training_state(
            checkpoint_path=checkpoint,
            agent=DQNAgent.from_config(base_config),
            config=active_config,
            expected_mode=CHECKPOINT_MODE_FULL,
        )


def test_config_snapshot_mismatch_expected_shape_raises_error(tmp_path: Path):
    base_config = load_config()
    trainer, agent = _trainer_and_agent(base_config)
    trainer.train(total_timesteps=16)
    checkpoint = tmp_path / "shape_mismatch.pt"
    _save_mode_checkpoint(checkpoint, trainer, agent, base_config, CHECKPOINT_MODE_FULL)

    active_config = copy.deepcopy(base_config)
    active_config["validation"]["expected_final_shape"] = [4, 84, 84, 3]

    with pytest.raises(ValueError, match="validation.expected_final_shape"):
        restore_training_state(
            checkpoint_path=checkpoint,
            agent=DQNAgent.from_config(base_config),
            config=active_config,
            expected_mode=CHECKPOINT_MODE_FULL,
        )
