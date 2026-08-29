"""Tests for HU008 MLflow experiment tracking."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.callbacks import TensorBoardLogger, load_tensorboard_scalars
from src.checkpointing import CheckpointMetadata
from src.evaluator import EvaluationSummary, evaluate_agent
from src.replay_buffer import ReplayBuffer
from src.tracking import MLflowTracker
from src.training_session import run_training_session
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
        "tracking_session_id": "session_001",
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
    config.setdefault("checkpointing", {})["save_replay_buffer"] = True
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


def _runtime_info(
    *,
    gpu_available: bool = False,
    gpu_name: str | None = None,
    ram_available_gb: float = 10.0,
    cuda_version: str | None = None,
) -> dict:
    return {
        "python_version": "3.11",
        "gymnasium_version": "1.1.1",
        "ale_py_version": "0.10.1",
        "torch_version": "2.0",
        "cuda_version": cuda_version,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_vram_total_gb": 16.0 if gpu_available else None,
        "cpu": "test-cpu",
        "cpu_count_logical": 8,
        "cpu_count_physical": 4,
        "ram_total_gb": 32.0,
        "ram_available_gb": ram_available_gb,
    }


def _start_logged_run(
    tmp_path: Path,
    project_run_id: str = "assault_ddqn_exp_tracking",
    tracking_session_id: str = "session_001",
):
    config = _config(tmp_path)
    tracker = MLflowTracker.from_config(config)
    metadata = tracker.start_run(
        project_run_id=project_run_id,
        tracking_mode="new",
        tracking_session_id=tracking_session_id,
    )
    tracker.log_run_context(
        config=config,
        runtime_info=_runtime_info(),
        git_commit=get_git_commit(ASSAULT_DIR.parents[0]),
        git_ref="feature/hu008-mlflow-tracking",
        project_run_id=project_run_id,
        action_space="Discrete(7)",
        observation_dtype="uint8",
        runtime="local",
        device="cpu",
    )
    return config, tracker, metadata


def _artifact_paths(tracker: MLflowTracker, mlflow_run_id: str, directories: tuple[str, ...]) -> set[str]:
    client = tracker._mlflow.tracking.MlflowClient()
    return {item.path for directory in directories for item in client.list_artifacts(mlflow_run_id, directory)}


def _download_json(tracker: MLflowTracker, mlflow_run_id: str, artifact_path: str, tmp_path: Path) -> dict:
    client = tracker._mlflow.tracking.MlflowClient()
    download_dir = tmp_path / "downloaded_artifacts"
    download_dir.mkdir(parents=True, exist_ok=True)
    local_path = client.download_artifacts(mlflow_run_id, artifact_path, str(download_dir))
    return json.loads(Path(local_path).read_text(encoding="utf-8"))


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
    assert metadata.tracking_session_id == "session_001"
    assert not (tmp_path / "mlruns").exists()


def test_new_run_logs_identity_params_runtime_metrics_artifacts_and_checkpoint_reference(tmp_path):
    config, tracker, metadata = _start_logged_run(tmp_path)
    checkpoint = CheckpointMetadata(tmp_path / "checkpoint_step_000008.pt", metadata.project_run_id, 8, 4096, True)

    tracker.log_config_snapshot(config, artifact_file="config/base_config.json")
    effective_config_artifact = tracker.log_session_config(config)
    tracker.log_runtime_metadata(get_runtime_info(), git_commit="abc123", runtime="local")
    tracker.log_training_summary(_training_summary())
    tracker.log_evaluation_summary(_evaluation_summary())
    tracker.log_checkpoint_reference(
        checkpoint,
        resume_mode="resume_full",
        checkpoint_output_reference=str(checkpoint.path),
    )
    tracker.log_session_metadata(
        tracking_mode="new",
        runtime_info=get_runtime_info(),
        git_commit="abc123",
        git_ref="feature/hu008-mlflow-tracking",
        runtime="local",
        device="cpu",
        initial_global_step=0,
        final_global_step=8,
        session_target_timesteps=8,
        checkpoint_input_reference=None,
        checkpoint_output_reference=str(checkpoint.path),
        effective_config_artifact=effective_config_artifact,
    )
    run = tracker.get_run(metadata.mlflow_run_id)
    artifacts = _artifact_paths(tracker, metadata.mlflow_run_id, ("config", "sessions/session_001"))
    session_metadata = _download_json(
        tracker,
        metadata.mlflow_run_id,
        "sessions/session_001/session_metadata.json",
        tmp_path,
    )
    tracker.end_run()

    assert metadata.enabled is True
    assert metadata.mlflow_run_id
    assert metadata.tracking_session_id == "session_001"
    assert run.info.run_id == metadata.mlflow_run_id
    assert run.data.tags["project_run_id"] == metadata.project_run_id
    assert run.data.tags["latest_tracking_session_id"] == "session_001"
    assert run.data.params["identity.algorithm"] == "DDQN"
    assert run.data.params["identity.project_run_id"] == metadata.project_run_id
    assert run.data.params["environment.id"] == "ALE/Assault-v5"
    assert run.data.params["environment.action_space"] == "Discrete(7)"
    assert run.data.params["preprocessing.frame_stack"] == "4"
    assert run.data.params["ddqn.gamma"] == "0.99"
    assert run.data.params["ddqn.batch_size"] == "4"
    assert "ddqn.total_timesteps" not in run.data.params
    assert "runtime.device" not in run.data.params
    assert "hardware.gpu_available" not in run.data.params
    assert "versions.python" not in run.data.params
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
    assert "config/base_config.json" in artifacts
    assert "sessions/session_001/runtime.json" in artifacts
    assert "sessions/session_001/effective_config.json" in artifacts
    assert "sessions/session_001/training_summary.json" in artifacts
    assert "sessions/session_001/evaluation_summary.json" in artifacts
    assert "sessions/session_001/checkpoint_reference.json" in artifacts
    assert "sessions/session_001/session_metadata.json" in artifacts
    assert session_metadata["tracking_session_id"] == "session_001"
    assert session_metadata["project_run_id"] == metadata.project_run_id
    assert session_metadata["mlflow_run_id"] == metadata.mlflow_run_id
    assert session_metadata["initial_global_step"] == 0
    assert session_metadata["final_global_step"] == 8
    assert session_metadata["session_target_timesteps"] == 8
    assert session_metadata["checkpoint_input_reference"] is None
    assert session_metadata["checkpoint_output_reference"] == str(checkpoint.path)
    assert session_metadata["effective_config_artifact"] == "sessions/session_001/effective_config.json"


def test_external_checkpoint_resume_restores_buffer_and_continues(tmp_path):
    config_a = _config(tmp_path)
    project_run_id = "assault_ddqn_exp_external_resume"
    session_a = run_training_session(
        config=config_a,
        checkpoint_root=tmp_path / "checkpoints",
        run_id=project_run_id,
        repo_path=ASSAULT_DIR.parents[0],
        tracking_mode="new",
        total_timesteps=8,
        device="cpu",
    )

    config_b = _config(tmp_path)
    session_b = run_training_session(
        config=config_b,
        checkpoint_root=tmp_path / "checkpoints",
        run_id=project_run_id,
        repo_path=ASSAULT_DIR.parents[0],
        tracking_mode="resume",
        checkpoint_input=session_a.checkpoint.path,
        resume_mode="resume_full",
        total_timesteps=12,
        device="cpu",
    )

    assert session_a.initial_global_step == 0
    assert session_a.final_global_step == 8
    assert session_b.checkpoint_input_loaded is True
    assert session_b.restored_global_step == session_a.checkpoint.checkpoint_step
    assert session_b.initial_global_step == session_a.checkpoint.checkpoint_step
    assert session_b.final_global_step == 12
    assert session_b.final_global_step > session_b.initial_global_step
    assert session_b.replay_buffer_restored is True
    assert session_b.checkpoint.path.exists()


def test_resume_existing_run_reuses_same_mlflow_run_loads_checkpoint_and_logs_lineage(tmp_path):
    project_run_id = "assault_ddqn_exp_resume"
    config_a, tracker_a, metadata_a = _start_logged_run(tmp_path, project_run_id=project_run_id)
    runtime_a = _runtime_info(gpu_available=False, ram_available_gb=10.0)
    effective_config_a = tracker_a.log_session_config(config_a, tracking_session_id="session_001")
    session_a = run_training_session(
        config=config_a,
        checkpoint_root=tmp_path / "checkpoints",
        run_id=project_run_id,
        repo_path=ASSAULT_DIR.parents[0],
        tracking_mode="new",
        total_timesteps=8,
        device="cpu",
    )
    evaluation_a = evaluate_agent(FakeEnv(total_steps=4, episode_length=2), session_a.agent, episodes=2, epsilon=0.0)
    tracker_a.log_training_summary(session_a.training)
    tracker_a.log_evaluation_summary(evaluation_a)
    tracker_a.log_checkpoint_reference(
        session_a.checkpoint,
        resume_mode="new",
        checkpoint_output_reference=session_a.checkpoint_output_reference,
    )
    tracker_a.log_session_metadata(
        tracking_mode="new",
        runtime_info=runtime_a,
        runtime="local",
        device="cpu",
        initial_global_step=session_a.initial_global_step,
        final_global_step=session_a.final_global_step,
        session_target_timesteps=config_a["training"]["total_timesteps"],
        checkpoint_input_loaded=session_a.checkpoint_input_loaded,
        replay_buffer_restored=session_a.replay_buffer_restored,
        checkpoint_output_reference=session_a.checkpoint_output_reference,
        effective_config_artifact=effective_config_a,
    )
    tracker_a._mlflow.log_metric("custom/session_a_marker", 1.0)
    tracker_a.end_run()

    config_b = _config(tmp_path)
    config_b["training"]["total_timesteps"] = 12
    runtime_b = _runtime_info(gpu_available=True, gpu_name="T4", ram_available_gb=20.0, cuda_version="12.1")
    tracker_b = MLflowTracker.from_config(config_b)
    metadata_b = tracker_b.start_run(
        project_run_id=project_run_id,
        tracking_mode="resume",
        mlflow_run_id=metadata_a.mlflow_run_id,
        tracking_session_id="session_002",
    )
    tracker_b.log_run_context(
        config=config_b,
        runtime_info=runtime_b,
        git_commit="session-b-sha",
        git_ref="feature/hu008-mlflow-tracking",
        project_run_id=project_run_id,
        action_space="Discrete(7)",
        observation_dtype="uint8",
        runtime="Google Colab",
        device="cuda",
    )
    effective_config_b = tracker_b.log_session_config(config_b, tracking_session_id="session_002")
    session_b = run_training_session(
        config=config_b,
        checkpoint_root=tmp_path / "checkpoints",
        run_id=project_run_id,
        repo_path=ASSAULT_DIR.parents[0],
        tracking_mode="resume",
        checkpoint_input=session_a.checkpoint.path,
        resume_mode="resume_full",
        total_timesteps=12,
        device="cpu",
    )
    evaluation_b = evaluate_agent(FakeEnv(total_steps=4, episode_length=2), session_b.agent, episodes=2, epsilon=0.0)
    tracker_b.log_training_summary(session_b.training)
    tracker_b.log_evaluation_summary(evaluation_b)
    tracker_b.log_checkpoint_reference(
        session_b.checkpoint,
        resume_mode="resume_full",
        checkpoint_input_reference=session_b.checkpoint_input_reference,
        checkpoint_output_reference=session_b.checkpoint_output_reference,
    )
    tracker_b.log_session_metadata(
        tracking_mode="resume",
        runtime_info=runtime_b,
        runtime="Google Colab",
        device="cuda",
        initial_global_step=session_b.initial_global_step,
        final_global_step=session_b.final_global_step,
        session_target_timesteps=config_b["training"]["total_timesteps"],
        checkpoint_input_reference=session_b.checkpoint_input_reference,
        checkpoint_output_reference=session_b.checkpoint_output_reference,
        checkpoint_input_loaded=session_b.checkpoint_input_loaded,
        restored_checkpoint_path=session_b.restored_checkpoint_path,
        restored_global_step=session_b.restored_global_step,
        replay_buffer_restored=session_b.replay_buffer_restored,
        resume_mode=session_b.resume_mode,
        effective_config_artifact=effective_config_b,
    )
    tracker_b._mlflow.log_metric("custom/session_b_marker", 2.0)
    run = tracker_b.get_run(metadata_b.mlflow_run_id)
    artifacts = _artifact_paths(tracker_b, metadata_b.mlflow_run_id, ("sessions/session_001", "sessions/session_002"))
    session_a = _download_json(tracker_b, metadata_b.mlflow_run_id, "sessions/session_001/session_metadata.json", tmp_path)
    session_b = _download_json(tracker_b, metadata_b.mlflow_run_id, "sessions/session_002/session_metadata.json", tmp_path)
    tracker_b.end_run()

    assert metadata_b.mlflow_run_id == metadata_a.mlflow_run_id
    assert metadata_b.project_run_id == metadata_a.project_run_id
    assert metadata_b.tracking_session_id == "session_002"
    assert run.data.params["identity.project_run_id"] == project_run_id
    assert run.data.tags["latest_tracking_session_id"] == "session_002"
    assert run.data.metrics["custom/session_a_marker"] == pytest.approx(1.0)
    assert run.data.metrics["custom/session_b_marker"] == pytest.approx(2.0)
    assert run.data.metrics["train/initial_global_step"] == pytest.approx(8.0)
    assert run.data.metrics["train/final_global_step"] == pytest.approx(12.0)
    assert run.data.metrics["eval/episodes"] == pytest.approx(2.0)
    assert run.data.metrics["eval/epsilon"] == pytest.approx(0.0)
    assert run.data.tags["latest_runtime"] == "Google Colab"
    assert run.data.tags["latest_device"] == "cuda"
    assert "ddqn.total_timesteps" not in run.data.params
    assert "runtime.name" not in run.data.params
    assert "hardware.ram_available_gb" not in run.data.params
    assert "sessions/session_001/session_metadata.json" in artifacts
    assert "sessions/session_002/session_metadata.json" in artifacts
    assert "sessions/session_001/training_summary.json" in artifacts
    assert "sessions/session_002/training_summary.json" in artifacts
    assert "sessions/session_001/evaluation_summary.json" in artifacts
    assert "sessions/session_002/evaluation_summary.json" in artifacts
    assert "sessions/session_001/effective_config.json" in artifacts
    assert "sessions/session_002/effective_config.json" in artifacts
    assert session_a["checkpoint_output_reference"] == session_b["checkpoint_input_reference"]
    assert session_b["checkpoint_input_reference"] == str(Path(session_a["checkpoint_output_reference"]))
    assert session_b["checkpoint_output_reference"] == str(Path(session_b["checkpoint_output_reference"]))
    assert session_b["checkpoint_input_loaded"] is True
    assert session_b["restored_global_step"] == session_a["final_global_step"]
    assert session_b["replay_buffer_restored"] is True
    assert session_b["initial_global_step"] == session_a["final_global_step"]
    assert session_b["final_global_step"] == 12
    assert session_b["final_global_step"] > session_b["initial_global_step"]
    assert session_a["runtime"] == "local"
    assert session_a["device"] == "cpu"
    assert session_a["gpu_available"] is False
    assert session_a["ram_available_gb"] == 10.0
    assert session_a["session_target_timesteps"] == 8
    assert session_b["runtime"] == "Google Colab"
    assert session_b["device"] == "cuda"
    assert session_b["gpu_available"] is True
    assert session_b["gpu_name"] == "T4"
    assert session_b["ram_available_gb"] == 20.0
    assert session_b["session_target_timesteps"] == 12
    assert session_a["effective_config_artifact"] == "sessions/session_001/effective_config.json"
    assert session_b["effective_config_artifact"] == "sessions/session_002/effective_config.json"


def test_duplicate_tracking_session_id_fails_fast_on_resume(tmp_path):
    project_run_id = "assault_ddqn_exp_duplicate_session"
    _, tracker_a, metadata_a = _start_logged_run(tmp_path, project_run_id=project_run_id)
    tracker_a.log_session_metadata(tracking_mode="new", initial_global_step=0, final_global_step=8)
    tracker_a.end_run()

    tracker_b = MLflowTracker.from_config(_config(tmp_path))

    with pytest.raises(RuntimeError, match="already exists"):
        tracker_b.start_run(
            project_run_id=project_run_id,
            tracking_mode="resume",
            mlflow_run_id=metadata_a.mlflow_run_id,
            tracking_session_id="session_001",
        )


def test_resume_rejects_global_ddqn_core_param_mismatch(tmp_path):
    project_run_id = "assault_ddqn_exp_core_mismatch"
    _, tracker_a, metadata_a = _start_logged_run(tmp_path, project_run_id=project_run_id)
    tracker_a.log_session_metadata(tracking_mode="new", initial_global_step=0, final_global_step=8)
    tracker_a.end_run()

    config_b = _config(tmp_path)
    config_b["agent"]["gamma"] = 0.5
    tracker_b = MLflowTracker.from_config(config_b)
    tracker_b.start_run(
        project_run_id=project_run_id,
        tracking_mode="resume",
        mlflow_run_id=metadata_a.mlflow_run_id,
        tracking_session_id="session_002",
    )
    try:
        with pytest.raises(ValueError, match="MLflow param mismatch for ddqn.gamma"):
            tracker_b.log_run_context(
                config=config_b,
                runtime_info=_runtime_info(gpu_available=True, gpu_name="T4", ram_available_gb=20.0),
                git_commit="session-b-sha",
                git_ref="feature/hu008-mlflow-tracking",
                project_run_id=project_run_id,
                action_space="Discrete(7)",
                observation_dtype="uint8",
                runtime="Google Colab",
                device="cuda",
            )
    finally:
        tracker_b.end_run(status="FAILED")


def test_resume_session_artifact_backend_failure_propagates(monkeypatch, tmp_path):
    project_run_id = "assault_ddqn_exp_backend_failure"
    _, tracker_a, metadata_a = _start_logged_run(tmp_path, project_run_id=project_run_id)
    tracker_a.log_session_metadata(tracking_mode="new", initial_global_step=0, final_global_step=8)
    tracker_a.end_run()

    tracker_b = MLflowTracker.from_config(_config(tmp_path))
    mlflow = tracker_b._require_mlflow()
    original_client_factory = mlflow.tracking.MlflowClient

    class BrokenArtifactClient:
        def __init__(self):
            self._delegate = original_client_factory()

        def get_run(self, *args, **kwargs):
            return self._delegate.get_run(*args, **kwargs)

        def list_artifacts(self, *args, **kwargs):
            raise RuntimeError("backend unavailable")

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", BrokenArtifactClient)

    with pytest.raises(RuntimeError, match="backend unavailable"):
        tracker_b.start_run(
            project_run_id=project_run_id,
            tracking_mode="resume",
            mlflow_run_id=metadata_a.mlflow_run_id,
            tracking_session_id="session_002",
        )


def test_resume_requires_explicit_mlflow_run_id(tmp_path):
    tracker = MLflowTracker.from_config(_config(tmp_path))

    with pytest.raises(ValueError, match="requires explicit mlflow_run_id"):
        tracker.start_run(project_run_id="ambiguous", tracking_mode="resume", tracking_session_id="session_001")


def test_enabled_tracking_requires_explicit_tracking_session_id(tmp_path):
    config = _config(tmp_path)
    config["mlflow"]["tracking_session_id"] = None
    tracker = MLflowTracker.from_config(config)

    with pytest.raises(ValueError, match="tracking_session_id"):
        tracker.start_run(project_run_id="missing_session", tracking_mode="new")


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
    metadata = tracker.start_run(project_run_id=project_run_id, tracking_session_id="session_001")
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
