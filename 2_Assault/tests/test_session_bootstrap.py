"""Tests for HU008B automated experiment session bootstrap."""

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

from src.session_bootstrap import (  # noqa: E402
    compute_config_fingerprint,
    inspect_experiment_state,
    prepare_training_session,
    update_experiment_state_after_success,
)
from src.tracking import MLflowTracker  # noqa: E402
from src.utils import load_yaml_config  # noqa: E402


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


def _config(tmp_path: Path, enabled: bool = True) -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["mlflow"] = {
        "enabled": enabled,
        "experiment_name": "assault_ddqn_hu008b_tests",
        "tracking_uri": str(tmp_path / "mlruns"),
        "local_directory": "logs/mlflow",
        "tracking_mode": "new",
        "mlflow_run_id": None,
        "tracking_session_id": None,
        "artifact_location": None,
        "log_checkpoint_binary": False,
    }
    config["replay_buffer"] = {"capacity": 64, "batch_size": 4}
    config["training"] = {
        "total_timesteps": 8,
        "learning_starts": 4,
        "train_frequency": 2,
        "target_update_frequency": 4,
        "epsilon_decay_steps": 8,
    }
    return config


def _start_mlflow_run(tmp_path: Path, config: dict, project_run_id: str) -> str:
    tracker = MLflowTracker.from_config(config)
    metadata = tracker.start_run(project_run_id=project_run_id, tracking_mode="new", tracking_session_id="session_001")
    tracker.log_run_context(
        config=config,
        git_commit="test-sha",
        git_ref="feature/hu008b-auto-session-bootstrap",
        project_run_id=project_run_id,
        action_space="Discrete(7)",
        observation_dtype="uint8",
        runtime="local",
        device="cpu",
    )
    tracker.log_session_metadata(tracking_mode="new", initial_global_step=0, final_global_step=8)
    tracker.end_run()
    return str(metadata.mlflow_run_id)


def _checkpoint(tmp_path: Path, config: dict, project_run_id: str, step: int = 8, save_replay: bool = True) -> Path:
    checkpoint_dir = tmp_path / "checkpoints" / project_run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    replay_state = None
    if save_replay:
        replay_state = {
            "capacity": config["replay_buffer"]["capacity"],
            "state_shape": (
                config["network"]["input_channels"],
                config["preprocessing"]["resize_height"],
                config["preprocessing"]["resize_width"],
            ),
            "size": 2,
            "position": 2,
            "states": np.zeros((2, 4, 84, 84), dtype=np.uint8),
            "next_states": np.zeros((2, 4, 84, 84), dtype=np.uint8),
            "actions": np.zeros((2,), dtype=np.int64),
            "rewards": np.zeros((2,), dtype=np.float32),
            "dones": np.zeros((2,), dtype=np.bool_),
            "rng_state": np.random.default_rng(123).bit_generator.state,
        }
    payload = {
        "schema_version": 1,
        "run_id": project_run_id,
        "created_at": "2026-08-29T00:00:00+00:00",
        "checkpoint_step": step,
        "git_commit": "test-sha",
        "config": config,
        "online_network": {},
        "target_network": {},
        "optimizer": {},
        "global_step": step,
        "epsilon_state": {"global_step": step},
        "training_metrics": {"global_step": step},
        "resume_mode_capabilities": {"resume_full": save_replay, "resume_light": True},
        "replay_buffer_state": replay_state,
    }
    path = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
    torch.save(payload, path)
    return path


def _write_manifest(
    tmp_path: Path,
    config: dict,
    project_run_id: str,
    mlflow_run_id: str | None,
    checkpoint: Path,
    step: int = 8,
    latest_session: str = "session_001",
) -> Path:
    path = tmp_path / "experiments" / project_run_id / "experiment_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_run_id": project_run_id,
                "mlflow_run_id": mlflow_run_id,
                "latest_tracking_session_id": latest_session,
                "latest_checkpoint": str(checkpoint),
                "latest_global_step": step,
                "resume_mode": "resume_full",
                "bootstrap_commit": "test-sha",
                "config_fingerprint": compute_config_fingerprint(config),
                "updated_at": "2026-08-29T00:00:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _prepare(tmp_path: Path, config: dict, project_run_id: str = "hu008b_test", target: int = 12):
    return prepare_training_session(
        base_path=tmp_path,
        project_run_id=project_run_id,
        target_timesteps=target,
        requested_mode="auto",
        config=config,
        bootstrap_ref="feature/hu008b-auto-session-bootstrap",
        bootstrap_commit="test-sha",
    )


def test_auto_without_manifest_resolves_new_session_001(tmp_path):
    config = _config(tmp_path)

    context = _prepare(tmp_path, config, target=8)

    assert context.tracking_mode == "new"
    assert context.tracking_session_id == "session_001"
    assert context.mlflow_run_id is None
    assert context.checkpoint_input is None


def test_auto_with_manifest_resolves_resume_session_002_same_mlflow_run_and_checkpoint(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_resume"
    mlflow_run_id = _start_mlflow_run(tmp_path, config, project_run_id)
    checkpoint = _checkpoint(tmp_path, config, project_run_id, step=8)
    _write_manifest(tmp_path, config, project_run_id, mlflow_run_id, checkpoint, step=8)

    context = _prepare(tmp_path, config, project_run_id=project_run_id, target=12)

    assert context.tracking_mode == "resume"
    assert context.tracking_session_id == "session_002"
    assert context.mlflow_run_id == mlflow_run_id
    assert context.checkpoint_input == checkpoint
    assert context.restored_expected_step == 8


def test_resume_missing_checkpoint_fails_fast(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_missing_checkpoint"
    mlflow_run_id = _start_mlflow_run(tmp_path, config, project_run_id)
    checkpoint = tmp_path / "missing.pt"
    _write_manifest(tmp_path, config, project_run_id, mlflow_run_id, checkpoint, step=8)

    with pytest.raises(FileNotFoundError, match="Manifest checkpoint does not exist"):
        _prepare(tmp_path, config, project_run_id=project_run_id, target=12)


def test_resume_full_replay_buffer_incompatible_fails_fast(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_bad_buffer"
    mlflow_run_id = _start_mlflow_run(tmp_path, config, project_run_id)
    checkpoint = _checkpoint(tmp_path, config, project_run_id, step=8)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["replay_buffer_state"]["capacity"] = 999
    torch.save(payload, checkpoint)
    _write_manifest(tmp_path, config, project_run_id, mlflow_run_id, checkpoint, step=8)

    with pytest.raises(ValueError, match="Replay buffer capacity mismatch"):
        _prepare(tmp_path, config, project_run_id=project_run_id, target=12)


def test_resume_fingerprint_incompatible_fails_fast(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_fingerprint"
    mlflow_run_id = _start_mlflow_run(tmp_path, config, project_run_id)
    checkpoint = _checkpoint(tmp_path, config, project_run_id, step=8)
    _write_manifest(tmp_path, config, project_run_id, mlflow_run_id, checkpoint, step=8)
    changed = _config(tmp_path)
    changed["agent"]["gamma"] = 0.5

    with pytest.raises(ValueError, match="Config fingerprint mismatch"):
        _prepare(tmp_path, changed, project_run_id=project_run_id, target=12)


def test_target_not_greater_than_restored_step_fails_fast(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_target"
    mlflow_run_id = _start_mlflow_run(tmp_path, config, project_run_id)
    checkpoint = _checkpoint(tmp_path, config, project_run_id, step=8)
    _write_manifest(tmp_path, config, project_run_id, mlflow_run_id, checkpoint, step=8)

    with pytest.raises(ValueError, match="greater than the restored"):
        _prepare(tmp_path, config, project_run_id=project_run_id, target=8)


def test_next_session_duplicate_in_mlflow_fails_fast(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_duplicate"
    tracker = MLflowTracker.from_config(config)
    metadata = tracker.start_run(project_run_id=project_run_id, tracking_mode="new", tracking_session_id="session_001")
    tracker.log_run_context(config, git_commit="test-sha", git_ref="ref", project_run_id=project_run_id)
    tracker.log_session_metadata(tracking_mode="new", initial_global_step=0, final_global_step=8)
    tracker.log_dict_artifact({"occupied": True}, "sessions/session_002/session_metadata.json", session_scoped=False)
    tracker.end_run()
    checkpoint = _checkpoint(tmp_path, config, project_run_id, step=8)
    _write_manifest(tmp_path, config, project_run_id, metadata.mlflow_run_id, checkpoint, step=8)

    with pytest.raises(RuntimeError, match="already exists"):
        _prepare(tmp_path, config, project_run_id=project_run_id, target=12)


def test_mlflow_run_for_other_project_fails_fast(tmp_path):
    config = _config(tmp_path)
    good_project = "hu008b_good_project"
    other_run_id = _start_mlflow_run(tmp_path, config, "hu008b_other_project")
    checkpoint = _checkpoint(tmp_path, config, good_project, step=8)
    _write_manifest(tmp_path, config, good_project, other_run_id, checkpoint, step=8)

    with pytest.raises(ValueError, match="MLflow run project_run_id mismatch"):
        _prepare(tmp_path, config, project_run_id=good_project, target=12)


def test_corrupt_manifest_fails_fast(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_corrupt"
    path = tmp_path / "experiments" / project_run_id / "experiment_state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        _prepare(tmp_path, config, project_run_id=project_run_id, target=12)


def test_atomic_manifest_update_writes_complete_state(tmp_path):
    config = _config(tmp_path)
    context = _prepare(tmp_path, config, project_run_id="hu008b_atomic", target=8)
    checkpoint = _checkpoint(tmp_path, config, "hu008b_atomic", step=8)

    state = update_experiment_state_after_success(context, "mlflow-run-1", checkpoint, 8)

    assert context.manifest_path.exists()
    assert not context.manifest_path.with_name(".experiment_state.json.tmp").exists()
    assert state.latest_tracking_session_id == "session_001"
    assert json.loads(context.manifest_path.read_text(encoding="utf-8"))["latest_checkpoint"] == str(checkpoint)


def test_failed_manifest_update_leaves_previous_manifest_intact(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_atomic_failure"
    context = _prepare(tmp_path, config, project_run_id=project_run_id, target=8)
    checkpoint = _checkpoint(tmp_path, config, project_run_id, step=8)
    update_experiment_state_after_success(context, "mlflow-run-1", checkpoint, 8)
    before = context.manifest_path.read_text(encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        update_experiment_state_after_success(context, "mlflow-run-1", tmp_path / "missing.pt", 8)

    assert context.manifest_path.read_text(encoding="utf-8") == before


def test_mlflow_disabled_allows_new_context_without_mlflow_run(tmp_path):
    config = _config(tmp_path, enabled=False)
    context = _prepare(tmp_path, config, project_run_id="hu008b_disabled", target=8)
    checkpoint = _checkpoint(tmp_path, config, "hu008b_disabled", step=8)

    state = update_experiment_state_after_success(context, None, checkpoint, 8)

    assert context.mlflow_enabled is False
    assert state.mlflow_run_id is None


def test_inspect_experiment_state_reports_missing_checkpoint(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_inspect"
    _write_manifest(tmp_path, config, project_run_id, "run-id", tmp_path / "missing.pt", step=8)

    report = inspect_experiment_state(tmp_path, project_run_id, config=config, tracking_uri=config["mlflow"]["tracking_uri"])

    assert report.manifest_exists is True
    assert "manifest_checkpoint_missing" in report.issues


def test_checkpoint_run_id_mismatch_fails_fast(tmp_path):
    config = _config(tmp_path)
    project_run_id = "hu008b_checkpoint_owner"
    mlflow_run_id = _start_mlflow_run(tmp_path, config, project_run_id)
    checkpoint = _checkpoint(tmp_path, config, "other_checkpoint_owner", step=8)
    _write_manifest(tmp_path, config, project_run_id, mlflow_run_id, checkpoint, step=8)

    with pytest.raises(ValueError, match="Checkpoint run_id mismatch"):
        _prepare(tmp_path, config, project_run_id=project_run_id, target=12)


def test_explicit_resume_without_manifest_fails_fast(tmp_path):
    config = _config(tmp_path)

    with pytest.raises(FileNotFoundError, match="Experiment manifest not found"):
        prepare_training_session(
            base_path=tmp_path,
            project_run_id="hu008b_no_manifest",
            target_timesteps=12,
            requested_mode="resume",
            config=config,
            bootstrap_ref="feature/hu008b-auto-session-bootstrap",
            bootstrap_commit="test-sha",
        )


def test_bootstrap_requires_explicit_commit_and_ref(tmp_path):
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="bootstrap_ref"):
        prepare_training_session(
            base_path=tmp_path,
            project_run_id="hu008b_missing_ref",
            target_timesteps=8,
            requested_mode="auto",
            config=config,
            bootstrap_ref=None,
            bootstrap_commit="test-sha",
        )


def test_notebook_has_auto_bootstrap_without_historical_ids():
    source = Path(ASSAULT_DIR / "assault_ddqn.ipynb").read_text(encoding="utf-8")

    assert "prepare_training_session" in source
    assert "SESSION_BOOTSTRAP_READY=True" in source
    for forbidden in [
        "86068e5989aa480da6df72a927d8922e",
        "d8d73f89e4f34e0aa64b4c0e23239821",
        "<MLFLOW_RUN_ID_SESSION_001>",
        "ASSAULT_MLFLOW_CHECKPOINT_INPUT",
        "ASSAULT_MLFLOW_SESSION_ID",
    ]:
        assert forbidden not in source
