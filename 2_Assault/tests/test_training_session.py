"""Tests for HU009C training-session orchestration details."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src import training_session
from src.trainer import TrainingSummary


class FakeEnv:
    """Environment stub used to avoid ALE during orchestration tests."""

    def close(self) -> None:
        pass


class FakeManager:
    """Checkpoint manager probe for run_training_session tests."""

    instances = []

    def __init__(self, directory, run_id, repo_path=".") -> None:
        self.directory = Path(directory)
        self.run_id = run_id
        self.repo_path = repo_path
        self.save_calls = []
        self.ensure_new_run_calls = 0
        FakeManager.instances.append(self)

    def ensure_new_run(self) -> None:
        self.ensure_new_run_calls += 1

    def checkpoint_path(self, step: int) -> Path:
        return self.directory / self.run_id / f"checkpoint_step_{int(step):06d}.pt"

    def save(
        self,
        agent,
        replay_buffer,
        config,
        global_step,
        training_metrics,
        save_replay_buffer=True,
        overwrite=False,
        keep_last=None,
    ):
        call = {
            "agent": agent,
            "replay_buffer": replay_buffer,
            "config": config,
            "global_step": global_step,
            "training_metrics": training_metrics,
            "save_replay_buffer": save_replay_buffer,
            "overwrite": overwrite,
            "keep_last": keep_last,
        }
        self.save_calls.append(call)
        path = self.checkpoint_path(global_step)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        return training_session.CheckpointMetadata(
            path=path,
            run_id=self.run_id,
            checkpoint_step=int(global_step),
            size_bytes=path.stat().st_size,
            save_replay_buffer=bool(save_replay_buffer),
        )


class FakeTrainer:
    """Trainer probe that can optionally emulate a final periodic checkpoint."""

    last_init_kwargs = None
    save_final_periodic = True

    def __init__(self, env, agent, replay_buffer, config, **kwargs) -> None:
        self.config = config
        self.kwargs = kwargs
        FakeTrainer.last_init_kwargs = kwargs

    def train(self):
        final_step = int(self.config["training"]["total_timesteps"])
        checkpoints_saved = []
        if self.save_final_periodic:
            manager = self.kwargs["checkpoint_manager"]
            final_path = manager.checkpoint_path(final_step)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_bytes(b"periodic-final")
            checkpoints_saved.append(str(final_path))
        return TrainingSummary(
            global_step=final_step,
            episodes_completed=0,
            episode_rewards=[],
            episode_lengths=[],
            epsilon_initial=1.0,
            epsilon_final=0.01,
            transitions_stored=final_step,
            updates_count=0,
            update_steps=[],
            first_update_step=None,
            last_loss=None,
            mean_loss=None,
            last_q_mean=None,
            mean_q_mean=None,
            last_learning_rate=None,
            target_sync_steps=[],
            online_weights_changed=False,
            duration_seconds=0.0,
            final_replay_buffer_size=0,
            checkpoints_saved=checkpoints_saved,
        )


def _config(interval_steps: int = 4, save_replay_buffer: bool = True) -> dict:
    return {
        "reproducibility": {"seed": 42},
        "replay_buffer": {"capacity": 16, "batch_size": 4},
        "training": {
            "total_timesteps": 8,
            "learning_starts": 1,
            "train_frequency": 1,
            "target_update_frequency": 4,
            "epsilon_decay_steps": 8,
        },
        "checkpointing": {"interval_steps": interval_steps, "save_replay_buffer": save_replay_buffer, "keep_last": 1},
        "tensorboard": {"enabled": False},
    }


@pytest.fixture(autouse=True)
def _patch_training_session(monkeypatch):
    FakeManager.instances = []
    FakeTrainer.last_init_kwargs = None
    FakeTrainer.save_final_periodic = True
    monkeypatch.setattr(training_session, "CheckpointManager", FakeManager)
    monkeypatch.setattr(training_session, "Trainer", FakeTrainer)
    monkeypatch.setattr(training_session, "DDQNAgent", lambda *args, **kwargs: object())
    monkeypatch.setattr(training_session, "ReplayBuffer", lambda *args, **kwargs: object())
    monkeypatch.setattr(training_session, "create_assault_env", lambda *args, **kwargs: FakeEnv())


def test_run_training_session_injects_periodic_checkpointing_and_reuses_final_checkpoint(tmp_path):
    summary = training_session.run_training_session(
        config=_config(interval_steps=4, save_replay_buffer=True),
        checkpoint_root=tmp_path / "checkpoints",
        run_id="periodic_probe",
        tracking_mode="new",
        total_timesteps=8,
        device="cpu",
    )
    manager = FakeManager.instances[0]

    assert FakeTrainer.last_init_kwargs["checkpoint_manager"] is manager
    assert FakeTrainer.last_init_kwargs["checkpoint_interval_steps"] == 4
    assert FakeTrainer.last_init_kwargs["checkpoint_save_replay_buffer"] is True
    assert FakeTrainer.last_init_kwargs["checkpoint_keep_last"] == 1
    assert summary.checkpoint.path == manager.checkpoint_path(8)
    assert summary.checkpoint.path.exists()
    assert summary.training.checkpoints_saved == [str(manager.checkpoint_path(8))]
    assert manager.save_calls == []


def test_run_training_session_saves_final_when_not_already_saved_by_periodic_checkpoint(tmp_path):
    FakeTrainer.save_final_periodic = False

    summary = training_session.run_training_session(
        config=_config(interval_steps=3, save_replay_buffer=False),
        checkpoint_root=tmp_path / "checkpoints",
        run_id="final_probe",
        tracking_mode="new",
        total_timesteps=8,
        device="cpu",
    )
    manager = FakeManager.instances[0]

    assert FakeTrainer.last_init_kwargs["checkpoint_interval_steps"] == 3
    assert FakeTrainer.last_init_kwargs["checkpoint_save_replay_buffer"] is False
    assert len(manager.save_calls) == 1
    assert manager.save_calls[0]["global_step"] == 8
    assert manager.save_calls[0]["save_replay_buffer"] is False
    assert manager.save_calls[0]["keep_last"] == 1
    assert summary.checkpoint.path == manager.checkpoint_path(8)
