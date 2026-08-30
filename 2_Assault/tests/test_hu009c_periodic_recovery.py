"""Regression tests for HU009C recovery from interrupted periodic checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.hu009c_delivery import resolve_hu009c_execution_mode  # noqa: E402
from src.session_bootstrap import prepare_training_session  # noqa: E402
from src.tracking import MLflowTracker  # noqa: E402
from src.utils import load_yaml_config  # noqa: E402


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


def _config(tmp_path: Path, mlflow_enabled: bool) -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["mlflow"] = {
        "enabled": mlflow_enabled,
        "experiment_name": "assault_ddqn_periodic_recovery_tests",
        "tracking_uri": str(tmp_path / "mlruns"),
        "local_directory": "logs/mlflow",
        "tracking_mode": "new",
        "mlflow_run_id": None,
        "tracking_session_id": None,
        "artifact_location": None,
        "log_checkpoint_binary": False,
    }
    config["replay_buffer"] = {"capacity": 16, "batch_size": 4}
    config["training"] = {
        "total_timesteps": 12,
        "learning_starts": 4,
        "train_frequency": 2,
        "target_update_frequency": 4,
        "epsilon_decay_steps": 12,
    }
    return config


def _checkpoint(tmp_path: Path, config: dict, project_run_id: str, step: int) -> Path:
    checkpoint_dir = tmp_path / "checkpoints" / project_run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_shape = (
        int(config["network"]["input_channels"]),
        int(config["preprocessing"]["resize_height"]),
        int(config["preprocessing"]["resize_width"]),
    )
    replay_state = {
        "capacity": int(config["replay_buffer"]["capacity"]),
        "state_shape": state_shape,
        "size": 2,
        "position": 2,
        "states": np.zeros((2, *state_shape), dtype=np.uint8),
        "next_states": np.zeros((2, *state_shape), dtype=np.uint8),
        "actions": np.zeros((2,), dtype=np.int64),
        "rewards": np.zeros((2,), dtype=np.float32),
        "dones": np.zeros((2,), dtype=np.bool_),
        "rng_state": np.random.default_rng(123).bit_generator.state,
    }
    payload = {
        "schema_version": 1,
        "run_id": project_run_id,
        "created_at": "2026-08-30T00:00:00+00:00",
        "checkpoint_step": step,
        "git_commit": "test-sha",
        "config": config,
        "online_network": {},
        "target_network": {},
        "optimizer": {},
        "global_step": step,
        "epsilon_state": {"global_step": step},
        "training_metrics": {"global_step": step},
        "resume_mode_capabilities": {"resume_full": True, "resume_light": True},
        "replay_buffer_state": replay_state,
    }
    path = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
    torch.save(payload, path)
    return path


def _kwargs(tmp_path: Path, config: dict, project_run_id: str, target: int = 12) -> dict:
    return {
        "base_path": tmp_path,
        "project_run_id": project_run_id,
        "target_timesteps": target,
        "requested_mode": "auto",
        "config": config,
        "checkpoint_root": tmp_path / "checkpoints",
        "tensorboard_root": tmp_path / "tensorboard",
        "tracking_uri": config["mlflow"]["tracking_uri"],
        "resume_mode": "resume_full",
        "bootstrap_ref": "feature/hu009c-delivery-artifacts",
        "bootstrap_commit": "test-sha",
        "mlflow_enabled": bool(config["mlflow"]["enabled"]),
        "mlflow_experiment_name": config["mlflow"]["experiment_name"],
    }


def test_auto_recovers_latest_orphan_periodic_checkpoint_without_manifest(tmp_path):
    config = _config(tmp_path, mlflow_enabled=False)
    project_run_id = "hu009c_periodic_recovery"
    _checkpoint(tmp_path, config, project_run_id, step=4)
    latest = _checkpoint(tmp_path, config, project_run_id, step=8)
    manifest = tmp_path / "experiments" / project_run_id / "experiment_state.json"

    mode = resolve_hu009c_execution_mode(
        run_training=None,
        execution_mode="auto",
        project_run_id=project_run_id,
        target_timesteps=12,
        final_checkpoint_path=tmp_path / "checkpoints" / project_run_id / "checkpoint_step_000012.pt",
        prepare_training_session_fn=prepare_training_session,
        prepare_training_session_kwargs=_kwargs(tmp_path, config, project_run_id),
    )

    assert mode.auto_resolution == "RESUME"
    assert mode.training_required is True
    assert mode.session_context.tracking_mode == "resume"
    assert mode.session_context.checkpoint_input == latest
    assert mode.session_context.restored_expected_step == 8
    assert mode.session_context.tracking_session_id == "session_002"
    recovered = json.loads(manifest.read_text(encoding="utf-8"))
    assert recovered["latest_checkpoint"] == str(latest)
    assert recovered["latest_global_step"] == 8
    assert recovered["recovered_from_periodic_checkpoint"] is True


def test_auto_recovery_preserves_mlflow_identity_after_interruption(tmp_path):
    config = _config(tmp_path, mlflow_enabled=True)
    project_run_id = "hu009c_periodic_mlflow_recovery"
    tracker = MLflowTracker.from_config(config)
    metadata = tracker.start_run(
        project_run_id=project_run_id,
        tracking_mode="new",
        tracking_session_id="session_001",
    )
    tracker.log_run_context(
        config=config,
        git_commit="test-sha",
        git_ref="feature/hu009c-delivery-artifacts",
        project_run_id=project_run_id,
    )
    tracker.end_run(status="FAILED")
    latest = _checkpoint(tmp_path, config, project_run_id, step=8)

    mode = resolve_hu009c_execution_mode(
        run_training=None,
        execution_mode="auto",
        project_run_id=project_run_id,
        target_timesteps=12,
        final_checkpoint_path=tmp_path / "checkpoints" / project_run_id / "checkpoint_step_000012.pt",
        prepare_training_session_fn=prepare_training_session,
        prepare_training_session_kwargs=_kwargs(tmp_path, config, project_run_id),
    )

    assert mode.auto_resolution == "RESUME"
    assert mode.session_context.checkpoint_input == latest
    assert mode.session_context.mlflow_run_id == metadata.mlflow_run_id
    assert mode.session_context.tracking_session_id == "session_002"


def test_invalid_orphan_checkpoint_fails_and_removes_recovery_manifest(tmp_path):
    config = _config(tmp_path, mlflow_enabled=False)
    project_run_id = "hu009c_invalid_periodic"
    checkpoint = _checkpoint(tmp_path, config, project_run_id, step=8)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["run_id"] = "different_run"
    torch.save(payload, checkpoint)
    manifest = tmp_path / "experiments" / project_run_id / "experiment_state.json"

    with pytest.raises(ValueError, match="Checkpoint run_id mismatch"):
        resolve_hu009c_execution_mode(
            run_training=None,
            execution_mode="auto",
            project_run_id=project_run_id,
            target_timesteps=12,
            final_checkpoint_path=tmp_path / "checkpoints" / project_run_id / "checkpoint_step_000012.pt",
            prepare_training_session_fn=prepare_training_session,
            prepare_training_session_kwargs=_kwargs(tmp_path, config, project_run_id),
        )

    assert not manifest.exists()


def test_auto_without_manifest_or_periodic_checkpoint_remains_new(tmp_path):
    config = _config(tmp_path, mlflow_enabled=False)
    project_run_id = "hu009c_clean_start"

    mode = resolve_hu009c_execution_mode(
        run_training=None,
        execution_mode="auto",
        project_run_id=project_run_id,
        target_timesteps=12,
        final_checkpoint_path=tmp_path / "checkpoints" / project_run_id / "checkpoint_step_000012.pt",
        prepare_training_session_fn=prepare_training_session,
        prepare_training_session_kwargs=_kwargs(tmp_path, config, project_run_id),
    )

    assert mode.auto_resolution == "NEW"
    assert mode.session_context.tracking_mode == "new"
    assert mode.session_context.checkpoint_input is None
