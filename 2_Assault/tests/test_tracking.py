"""Tests for HU008 MLflow experiment tracking."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.callbacks import TensorBoardLogger, load_tensorboard_scalars
from src.checkpointing import CheckpointMetadata
from src.evaluator import EvaluationSummary
from src.replay_buffer import ReplayBuffer
from src.tracking import MLflowTracker
from src.trainer import Trainer, TrainingSummary
from src.utils import get_git_commit, get_runtime_info, load_yaml_config

from test_tensorboard import FakeAgent, FakeEnv


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


def _config(tmp_path: Path, enabled: bool = True) -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["mlflow"] = {
        "enabled": enabled,
        "experiment_name": "assault_ddqn_tests",
        "tracking_uri": str(tmp_path / "mlruns"),
        "local_directory": "logs/mlflow",
        "tracking_mode": "new",
        "mlflow_run_id": None,
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


def _training_summary(initial_step: int = 0, final_step: int = 8) -> TrainingSummary:
    return TrainingSummary(
        global_step=final_step,
        episodes_completed=1,
        episode_rewards=[4.0],
        episode_lengths=[8],
        epsilon_initial=1.0,
        epsilon_final=0.5,
        transitions_stored=final_step - initial_step,
        updates_count=3,
        update_steps=[4, 6, 8],
        first_update_step=4,
        last_loss=0.25,
        mean_loss=0.5,
        last_q_mean=1.5,
        mean_q_mean=1.0,
        last_learning_rate=0.0001,
        target_sync_steps=[4, 8],
        online_weights_changed=True,
        duration_seconds=1.25,
        final_replay_buffer_size=8,
        initial_global_step=initial_step,
    )


def _evaluation_summary() -> EvaluationSummary:
    return EvaluationSummary(
        episodes=2,
        rewards=[1.0, 3.0],
        mean_reward=2.0,
        median_reward=2.0,
        std_reward=1.0,
        min_reward=1.0,
        max_reward=3.0,
        episode_lengths=[5, 7],
        epsilon=0.0,
        terminated_episodes=2,
        truncated_episodes=0,
    )


def _start_logged_run(tmp_path: Path, project_run_id: str = "assault_ddqn_exp_tracking"):
    config = _config(tmp_path)
    tracker = MLflowTracker.from_config(config)
    metadata = tracker.start_run(project_run_id=project_run_id, tracking_mode="new")
    tracker.log_run_context(
        config=config,
        runtime_info=get_runtime_info(),
        git_commit=get_git_commit(ASSAULT_DIR.parents[0]),
        git_ref="feature/hu008-mlflow-tracking",
        project_run_id=project_run_id,
        action_space="Discrete(7)",
        observation_dtype="uint8",
        runtime="local",
        device="cpu",
    )
    return config, tracker, metadata


def test_disabled_tracking_is_noop_without_importing_mlflow(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "mlflow":
            raise AssertionError("disabled tracker must not import mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    tracker = MLflowTracker.from_config(_config(tmp_path, enabled=False))
    metadata = tracker.start_run(project_run_id="disabled_run")
    tracker.log_training_summary(_training_summary())
    tracker.log_evaluation_summary(_evaluation_summary())
    tracker.log_checkpoint_reference(CheckpointMetadata(tmp_path / "model.pt", "disabled_run", 8, 123, True), "resume_full")
    tracker.end_run()

    assert metadata.enabled is False
    assert metadata.mlflow_run_id is None
    assert not (tmp_path / "mlruns").exists()


def test_new_run_logs_identity_params_runtime_metrics_artifacts_and_checkpoint_reference(tmp_path):
    config, tracker, metadata = _start_logged_run(tmp_path)
    checkpoint = CheckpointMetadata(tmp_path / "checkpoint_step_000008.pt", metadata.project_run_id, 8, 4096, True)

    tracker.log_config_snapshot(config)
    tracker.log_runtime_metadata(get_runtime_info(), git_commit="abc123", runtime="local")
    tracker.log_training_summary(_training_summary())
    tracker.log_evaluation_summary(_evaluation_summary())
    tracker.log_checkpoint_reference(checkpoint, resume_mode="resume_full")
    run = tracker.get_run(metadata.mlflow_run_id)
    client = tracker._mlflow.tracking.MlflowClient()
    artifacts = {
        item.path
        for directory in ("config", "metadata", "summaries", "artifacts")
        for item in client.list_artifacts(metadata.mlflow_run_id, directory)
    }
    tracker.end_run()

    assert metadata.enabled is True
    assert metadata.mlflow_run_id
    assert run.info.run_id == metadata.mlflow_run_id
    assert run.data.tags["project_run_id"] == metadata.project_run_id
    assert run.data.params["identity.algorithm"] == "DDQN"
    assert run.data.params["identity.project_run_id"] == metadata.project_run_id
    assert run.data.params["git.commit"]
    assert run.data.params["environment.id"] == "ALE/Assault-v5"
    assert run.data.params["environment.action_space"] == "Discrete(7)"
    assert run.data.params["preprocessing.frame_stack"] == "4"
    assert run.data.params["ddqn.gamma"] == "0.99"
    assert run.data.params["ddqn.batch_size"] == "4"
    assert run.data.params["versions.python"]
    assert run.data.params["versions.mlflow"]
    assert run.data.params["hardware.gpu_available"] in {"True", "False"}
    assert run.data.metrics["train/final_global_step"] == pytest.approx(8.0)
    assert run.data.metrics["train/last_loss"] == pytest.approx(0.25)
    assert run.data.metrics["train/mean_q_mean"] == pytest.approx(1.0)
    assert run.data.metrics["train/final_epsilon"] == pytest.approx(0.5)
    assert run.data.metrics["train/replay_buffer_size"] == pytest.approx(8.0)
    assert run.data.metrics["eval/mean_reward"] == pytest.approx(2.0)
    assert run.data.metrics["eval/max_reward"] == pytest.approx(3.0)
    assert run.data.metrics["eval/episodes"] == pytest.approx(2.0)
    assert run.data.metrics["checkpoint/step"] == pytest.approx(8.0)
    assert run.data.tags["checkpoint_resume_mode"] == "resume_full"
    assert "config/ddqn_config.json" in artifacts
    assert "metadata/runtime.json" in artifacts
    assert "summaries/train_summary.json" in artifacts
    assert "summaries/eval_summary.json" in artifacts
    assert "artifacts/checkpoint_reference.json" in artifacts


def test_resume_existing_run_reuses_same_mlflow_run_and_preserves_metrics(tmp_path):
    project_run_id = "assault_ddqn_exp_resume"
    _, tracker_a, metadata_a = _start_logged_run(tmp_path, project_run_id=project_run_id)
    tracker_a.log_training_summary(_training_summary(initial_step=0, final_step=8))
    tracker_a._mlflow.log_metric("custom/session_a_marker", 1.0)
    tracker_a.end_run()

    tracker_b = MLflowTracker.from_config(_config(tmp_path))
    metadata_b = tracker_b.start_run(
        project_run_id=project_run_id,
        tracking_mode="resume",
        mlflow_run_id=metadata_a.mlflow_run_id,
    )
    tracker_b.log_training_summary(_training_summary(initial_step=8, final_step=12))
    tracker_b._mlflow.log_metric("custom/session_b_marker", 2.0)
    run = tracker_b.get_run(metadata_b.mlflow_run_id)
    tracker_b.end_run()

    assert metadata_b.mlflow_run_id == metadata_a.mlflow_run_id
    assert run.data.params["identity.project_run_id"] == project_run_id
    assert run.data.metrics["custom/session_a_marker"] == pytest.approx(1.0)
    assert run.data.metrics["custom/session_b_marker"] == pytest.approx(2.0)
    assert run.data.metrics["train/initial_global_step"] == pytest.approx(8.0)
    assert run.data.metrics["train/final_global_step"] == pytest.approx(12.0)


def test_resume_requires_explicit_mlflow_run_id(tmp_path):
    tracker = MLflowTracker.from_config(_config(tmp_path))

    with pytest.raises(ValueError, match="requires explicit mlflow_run_id"):
        tracker.start_run(project_run_id="ambiguous", tracking_mode="resume")


def test_different_project_run_ids_create_isolated_mlflow_runs(tmp_path):
    _, tracker_a, metadata_a = _start_logged_run(tmp_path, project_run_id="assault_ddqn_exp_a")
    tracker_a.log_training_summary(_training_summary(final_step=8))
    tracker_a.end_run()
    _, tracker_b, metadata_b = _start_logged_run(tmp_path, project_run_id="assault_ddqn_exp_b")
    tracker_b.log_training_summary(_training_summary(final_step=12))
    tracker_b.end_run()

    run_a = tracker_a.get_run(metadata_a.mlflow_run_id)
    run_b = tracker_b.get_run(metadata_b.mlflow_run_id)

    assert metadata_a.mlflow_run_id != metadata_b.mlflow_run_id
    assert run_a.data.params["identity.project_run_id"] == "assault_ddqn_exp_a"
    assert run_b.data.params["identity.project_run_id"] == "assault_ddqn_exp_b"
    assert run_a.data.metrics["train/final_global_step"] == pytest.approx(8.0)
    assert run_b.data.metrics["train/final_global_step"] == pytest.approx(12.0)


def test_tensorboard_and_mlflow_coexist_with_same_project_run_id(tmp_path):
    config = _config(tmp_path)
    project_run_id = "assault_ddqn_exp_tb_mlflow"
    tensorboard_root = tmp_path / "tensorboard"
    logger = TensorBoardLogger.from_config(config, run_id=project_run_id, log_root=tensorboard_root)
    try:
        summary = Trainer(FakeEnv(total_steps=8), FakeAgent(), ReplayBuffer(64), config, metrics_logger=logger).train()
        logger.flush()
    finally:
        logger.close()

    tracker = MLflowTracker.from_config(config)
    metadata = tracker.start_run(project_run_id=project_run_id)
    tracker.log_training_summary(summary)
    run = tracker.get_run(metadata.mlflow_run_id)
    tracker.end_run()
    scalars = load_tensorboard_scalars(tensorboard_root / project_run_id)

    assert scalars["train/loss"]
    assert run.data.params["identity.project_run_id"] == project_run_id
    assert run.data.metrics["train/final_global_step"] == pytest.approx(summary.global_step)


def test_invalid_local_tracking_uri_fails_fast(tmp_path):
    bad_store = tmp_path / "not_a_directory"
    bad_store.write_text("occupied", encoding="utf-8")
    config = _config(tmp_path)
    config["mlflow"]["tracking_uri"] = str(bad_store)
    tracker = MLflowTracker.from_config(config)

    with pytest.raises(RuntimeError, match="not writable"):
        tracker.start_run(project_run_id="bad_store")


def test_mlflow_coupling_is_confined_to_tracking_module():
    forbidden = [
        "agent.py",
        "network.py",
        "replay_buffer.py",
        "environment.py",
        "trainer.py",
        "evaluator.py",
        "checkpointing.py",
    ]

    offenders = []
    for filename in forbidden:
        text = (ASSAULT_DIR / "src" / filename).read_text(encoding="utf-8").lower()
        if "mlflow" in text:
            offenders.append(filename)

    assert offenders == []
