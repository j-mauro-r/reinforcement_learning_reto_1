"""Tests for HU005 checkpointing, resume and idempotence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src import checkpointing  # noqa: E402
from src.agent import DDQNAgent
from src.checkpointing import CheckpointManager, reconstruct_epsilon
from src.environment import create_assault_env
from src.preflight import run_preflight_checks
from src.replay_buffer import ReplayBuffer
from src.trainer import Trainer
from src.utils import load_yaml_config


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


class FakeActionSpace:
    """Small action space for resume timing tests."""

    n = 7

    def sample(self) -> int:
        return 0


class FakeEnv:
    """Deterministic environment fake used by trainer resume tests."""

    action_space = FakeActionSpace()

    def __init__(self) -> None:
        self.step_index = 0

    def reset(self, seed=None):
        return self._obs(), {"seed": seed}

    def step(self, action: int):
        self.step_index += 1
        return self._obs(), 1.0, False, False, {"action": action}

    def close(self) -> None:
        pass

    def _obs(self):
        return np.full((4, 84, 84), self.step_index % 256, dtype=np.uint8)


def _config() -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["replay_buffer"] = {"capacity": 64, "batch_size": 4}
    config["training"] = {
        "total_timesteps": 8,
        "learning_starts": 4,
        "train_frequency": 2,
        "target_update_frequency": 4,
        "epsilon_decay_steps": 8,
    }
    config["checkpointing"] = {
        "enabled": True,
        "interval_steps": 4,
        "directory": "checkpoints",
        "mode": "new",
        "run_id": "assault_ddqn_exp_test",
        "resume_checkpoint": None,
        "save_replay_buffer": True,
    }
    return config


def _state(value: int) -> np.ndarray:
    return np.full((4, 84, 84), value, dtype=np.uint8)


def _filled_buffer(capacity: int = 8, count: int = 6) -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=capacity, seed=123)
    for index in range(count):
        buffer.add(_state(index), index % 7, float(index), _state(index + 1), index % 3 == 0)
    return buffer


def _train_agent(config: dict, total_timesteps: int = 8):
    config["training"]["total_timesteps"] = total_timesteps
    env = create_assault_env(config, mode="train", seed=42)
    try:
        agent = DDQNAgent(config, device="cpu", seed=42)
        buffer = ReplayBuffer(capacity=int(config["replay_buffer"]["capacity"]), seed=42)
        summary = Trainer(env, agent, buffer, config).train()
        return agent, buffer, summary
    finally:
        env.close()


def _q_values(agent: DDQNAgent) -> torch.Tensor:
    state = torch.zeros((1, 4, 84, 84), dtype=torch.uint8, device=agent.device)
    with torch.no_grad():
        return agent.online_network(state).detach().cpu()


def test_replay_buffer_state_dict_roundtrip_preserves_valid_slots_position_and_rng():
    source = _filled_buffer(capacity=8, count=10)
    state = source.state_dict()
    restored = ReplayBuffer(capacity=8, seed=999)

    restored.load_state_dict(state)

    assert len(restored) == len(source) == 8
    assert restored.position == source.position
    assert state["states"].shape[0] == len(source)
    assert state["states"].shape[0] < source.capacity + 1
    assert np.array_equal(restored.sample(3).states, source.sample(3).states)


def test_checkpoint_path_filename_and_run_id_directory(tmp_path):
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_001")

    path = manager.checkpoint_path(40)

    assert path == tmp_path / "assault_ddqn_exp_001" / "checkpoint_step_000040.pt"


def test_checkpoint_full_roundtrip_restores_agent_optimizer_step_config_metrics_and_buffer(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=8)
    expected_q = _q_values(agent)
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test", repo_path=ASSAULT_DIR.parents[0])

    metadata = manager.save(agent, buffer, config, summary.global_step, summary, save_replay_buffer=True)
    loaded_agent = DDQNAgent(config, device="cpu", seed=999)
    loaded_buffer = ReplayBuffer(capacity=64, seed=999)
    state = manager.load(metadata.path, loaded_agent, loaded_buffer, config, mode="resume_full")

    assert metadata.path.exists()
    assert metadata.size_bytes > 0
    assert state.run_id == "assault_ddqn_exp_test"
    assert state.global_step == 8
    assert state.config["network"]["num_actions"] == 7
    assert state.training_metrics["global_step"] == 8
    assert state.replay_buffer_restored is True
    assert len(loaded_buffer) == len(buffer)
    assert loaded_buffer.position == buffer.position
    assert torch.allclose(expected_q, _q_values(loaded_agent), atol=1e-6)
    assert loaded_agent.optimizer.state_dict()["state"]


def test_checkpoint_light_roundtrip_restores_agent_but_leaves_buffer_empty(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=8)
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test")

    metadata = manager.save(agent, buffer, config, summary.global_step, summary, save_replay_buffer=False)
    loaded_agent = DDQNAgent(config, device="cpu", seed=999)
    loaded_buffer = ReplayBuffer(capacity=64, seed=999)
    state = manager.load(metadata.path, loaded_agent, loaded_buffer, config, mode="resume_light")

    assert state.replay_buffer_restored is False
    assert len(loaded_buffer) == 0
    assert state.epsilon == pytest.approx(reconstruct_epsilon(8, config))


def test_resume_full_requires_replay_state(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=8)
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test")
    metadata = manager.save(agent, buffer, config, summary.global_step, summary, save_replay_buffer=False)

    with pytest.raises(ValueError, match="resume_full capability"):
        manager.load(metadata.path, DDQNAgent(config, device="cpu"), ReplayBuffer(64), config, mode="resume_full")


def test_no_overwrite_and_explicit_overwrite(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=4)
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test")

    first = manager.save(agent, buffer, config, summary.global_step, summary)
    with pytest.raises(FileExistsError, match="Checkpoint already exists"):
        manager.save(agent, buffer, config, summary.global_step, summary)
    second = manager.save(agent, buffer, config, summary.global_step, summary, overwrite=True)

    assert first.path == second.path
    assert second.path.exists()


def test_new_run_with_existing_checkpoint_is_rejected(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=4)
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test")
    manager.save(agent, buffer, config, summary.global_step, summary)

    with pytest.raises(FileExistsError, match="Run already has checkpoints"):
        manager.ensure_new_run()


def test_resume_requires_explicit_checkpoint_path(tmp_path):
    config = _config()
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test")

    with pytest.raises(ValueError, match="explicit checkpoint_path"):
        manager.load(None, DDQNAgent(config, device="cpu"), ReplayBuffer(64), config, mode="resume_light")


def test_incompatible_config_and_schema_fail_explicitly(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=4)
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test")
    metadata = manager.save(agent, buffer, config, summary.global_step, summary)

    incompatible = _config()
    incompatible["network"]["num_actions"] = 9
    with pytest.raises(ValueError, match="network.num_actions"):
        manager.load(metadata.path, DDQNAgent(config, device="cpu"), ReplayBuffer(64), incompatible, mode="resume_light")

    payload = torch.load(metadata.path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 999
    bad_path = tmp_path / "bad_schema.pt"
    torch.save(payload, bad_path)
    with pytest.raises(ValueError, match="schema_version"):
        manager.load(bad_path, DDQNAgent(config, device="cpu"), ReplayBuffer(64), config, mode="resume_light")


def test_atomic_save_leaves_final_file_without_temporary_checkpoint(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=4)
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test")

    metadata = manager.save(agent, buffer, config, summary.global_step, summary)

    assert metadata.path.exists()
    assert not list(metadata.path.parent.glob("*.tmp"))


def test_trainer_resume_uses_global_total_timesteps_not_additional_timesteps():
    config = _config()
    config["training"].update({"total_timesteps": 12, "learning_starts": 1, "train_frequency": 2})
    agent = DDQNAgent(config, device="cpu", seed=42)
    buffer = _filled_buffer(capacity=64, count=8)

    summary = Trainer(FakeEnv(), agent, buffer, config, initial_global_step=8).train()

    assert summary.initial_global_step == 8
    assert summary.global_step == 12
    assert summary.transitions_stored == 4
    assert summary.updates_count == 2
    assert summary.update_steps == [10, 12]


def test_resume_light_keeps_global_step_and_blocks_updates_until_batch_refilled():
    config = _config()
    config["training"].update({"total_timesteps": 11, "learning_starts": 1, "train_frequency": 1})
    agent = DDQNAgent(config, device="cpu", seed=42)
    empty_buffer = ReplayBuffer(capacity=64, seed=42)

    summary = Trainer(FakeEnv(), agent, empty_buffer, config, initial_global_step=8).train()

    assert summary.global_step == 11
    assert summary.transitions_stored == 3
    assert summary.updates_count == 0
    assert len(empty_buffer) == 3
    assert summary.epsilon_initial == pytest.approx(reconstruct_epsilon(8, config))


def test_trainer_saves_periodic_checkpoints(tmp_path):
    config = _config()
    config["training"].update({"total_timesteps": 8, "learning_starts": 4, "train_frequency": 2})
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test")
    agent = DDQNAgent(config, device="cpu", seed=42)
    buffer = ReplayBuffer(capacity=64, seed=42)

    summary = Trainer(
        FakeEnv(),
        agent,
        buffer,
        config,
        checkpoint_manager=manager,
        checkpoint_interval_steps=4,
    ).train()

    assert [Path(path).name for path in summary.checkpoints_saved] == [
        "checkpoint_step_000004.pt",
        "checkpoint_step_000008.pt",
    ]
    assert all(Path(path).exists() for path in summary.checkpoints_saved)


def test_checkpoint_retention_keep_last_one_after_multiple_saves(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=4)
    manager = CheckpointManager(tmp_path, "retention_basic")

    first = manager.save(agent, buffer, config, 25, summary, keep_last=1)
    second = manager.save(agent, buffer, config, 50, summary, keep_last=1)

    assert not first.path.exists()
    assert second.path.exists()
    assert [path.name for path in manager.run_dir.glob("checkpoint_step_*.pt")] == ["checkpoint_step_000050.pt"]


def test_checkpoint_retention_does_not_prune_when_new_save_fails(tmp_path, monkeypatch):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=4)
    manager = CheckpointManager(tmp_path, "retention_failure")
    first = manager.save(agent, buffer, config, 25, summary, keep_last=1)

    def fail_save(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(checkpointing.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="disk full"):
        manager.save(agent, buffer, config, 50, summary, keep_last=1)

    assert first.path.exists()
    assert not manager.checkpoint_path(50).exists()
    assert [path.name for path in manager.run_dir.glob("checkpoint_step_*.pt")] == ["checkpoint_step_000025.pt"]


def test_checkpoint_retention_isolated_by_run_id(tmp_path):
    run_a = CheckpointManager(tmp_path, "run_A")
    run_b = CheckpointManager(tmp_path, "run_B")
    for path in [run_a.checkpoint_path(25), run_a.checkpoint_path(50), run_b.checkpoint_path(25)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")

    deleted = run_a.prune_old_checkpoints(keep_last=1)

    assert [path.name for path in deleted] == ["checkpoint_step_000025.pt"]
    assert not run_a.checkpoint_path(25).exists()
    assert run_a.checkpoint_path(50).exists()
    assert run_b.checkpoint_path(25).exists()


def test_checkpoint_retention_allows_resume_from_newest_after_next_save(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=4)
    manager = CheckpointManager(tmp_path, "resume_retention")
    checkpoint_100 = manager.save(agent, buffer, config, 100, summary, keep_last=1)

    loaded_agent = DDQNAgent(config, device="cpu", seed=999)
    loaded_buffer = ReplayBuffer(capacity=64, seed=999)
    manager.load(checkpoint_100.path, loaded_agent, loaded_buffer, config, mode="resume_full")
    checkpoint_125 = manager.save(loaded_agent, loaded_buffer, config, 125, summary, keep_last=1)

    assert not checkpoint_100.path.exists()
    assert checkpoint_125.path.exists()
    next_agent = DDQNAgent(config, device="cpu", seed=123)
    next_buffer = ReplayBuffer(capacity=64, seed=123)
    state = manager.load(checkpoint_125.path, next_agent, next_buffer, config, mode="resume_full")
    assert state.global_step == 125
    assert state.replay_buffer_restored is True


def test_checkpoint_retention_keeps_final_checkpoint_only(tmp_path):
    config = _config()
    agent, buffer, summary = _train_agent(config, total_timesteps=4)
    manager = CheckpointManager(tmp_path, "final_retention")

    checkpoint_225 = manager.save(agent, buffer, config, 225, summary, keep_last=1)
    checkpoint_250 = manager.save(agent, buffer, config, 250, summary, keep_last=1)

    assert not checkpoint_225.path.exists()
    assert checkpoint_250.path.exists()
    assert sorted(path.name for path in manager.run_dir.glob("checkpoint_step_*.pt")) == ["checkpoint_step_000250.pt"]


def test_checkpoint_retention_preserves_non_checkpoint_files(tmp_path):
    manager = CheckpointManager(tmp_path, "non_checkpoint_retention")
    old_checkpoint = manager.checkpoint_path(25)
    latest_checkpoint = manager.checkpoint_path(50)
    for path, content in [
        (old_checkpoint, b"old"),
        (latest_checkpoint, b"latest"),
        (manager.run_dir / "metadata.json", b"{}"),
        (manager.run_dir / "foo.txt", b"notes"),
        (manager.run_dir / ".checkpoint_step_000050.pt.tmp", b"tmp"),
        (manager.run_dir / "checkpoint_step_latest.pt", b"bad-name"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    manager.prune_old_checkpoints(keep_last=1)

    assert not old_checkpoint.exists()
    assert latest_checkpoint.exists()
    assert (manager.run_dir / "metadata.json").exists()
    assert (manager.run_dir / "foo.txt").exists()
    assert (manager.run_dir / ".checkpoint_step_000050.pt.tmp").exists()
    assert (manager.run_dir / "checkpoint_step_latest.pt").exists()


def test_run_ids_use_separate_paths(tmp_path):
    manager_a = CheckpointManager(tmp_path, "assault_ddqn_exp_001")
    manager_b = CheckpointManager(tmp_path, "assault_ddqn_exp_002")

    assert manager_a.checkpoint_path(4) != manager_b.checkpoint_path(4)


def test_real_assault_smoke_new_full_and_light_resume(tmp_path):
    config = _config()
    config["training"].update({"total_timesteps": 8, "learning_starts": 4, "train_frequency": 2})
    assert run_preflight_checks(config, device="cpu").passed

    agent, buffer, first_summary = _train_agent(config, total_timesteps=8)
    manager = CheckpointManager(tmp_path, "assault_ddqn_exp_test", repo_path=ASSAULT_DIR.parents[0])
    full_metadata = manager.save(agent, buffer, config, first_summary.global_step, first_summary, save_replay_buffer=True)
    light_metadata = manager.save(agent, buffer, config, first_summary.global_step + 1, first_summary, save_replay_buffer=False)

    resumed_config = _config()
    resumed_config["training"].update({"total_timesteps": 12, "learning_starts": 4, "train_frequency": 2})
    full_agent = DDQNAgent(resumed_config, device="cpu", seed=999)
    full_buffer = ReplayBuffer(capacity=64, seed=999)
    full_state = manager.load(full_metadata.path, full_agent, full_buffer, resumed_config, mode="resume_full")
    full_env = create_assault_env(resumed_config, mode="train", seed=43)
    try:
        full_summary = Trainer(
            full_env,
            full_agent,
            full_buffer,
            resumed_config,
            initial_global_step=full_state.global_step,
            initial_metrics=full_state.training_metrics,
        ).train()
    finally:
        full_env.close()

    light_agent = DDQNAgent(resumed_config, device="cpu", seed=999)
    light_buffer = ReplayBuffer(capacity=64, seed=999)
    light_state = manager.load(light_metadata.path, light_agent, light_buffer, resumed_config, mode="resume_light")

    assert full_summary.global_step == 12
    assert full_summary.updates_count > first_summary.updates_count
    assert np.isfinite(full_summary.last_loss)
    assert full_state.replay_buffer_restored is True
    assert len(full_buffer) >= len(buffer)
    assert light_state.global_step == 9
    assert light_state.epsilon == pytest.approx(reconstruct_epsilon(9, resumed_config))
    assert len(light_buffer) == 0
