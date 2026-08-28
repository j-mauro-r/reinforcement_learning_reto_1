"""MLflow tracking utilities for Assault DDQN experiments."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class MLflowRunMetadata:
    """Metadata that links a project run with an MLflow run."""

    enabled: bool
    project_run_id: str
    mlflow_run_id: Optional[str]
    tracking_session_id: Optional[str]
    experiment_name: Optional[str]
    experiment_id: Optional[str]
    tracking_uri: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        """Returns a serializable metadata representation."""
        return {
            "enabled": self.enabled,
            "project_run_id": self.project_run_id,
            "mlflow_run_id": self.mlflow_run_id,
            "tracking_session_id": self.tracking_session_id,
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "tracking_uri": self.tracking_uri,
        }


class MLflowTracker:
    """Small MLflow adapter used by notebooks or orchestration code.

    The class deliberately keeps MLflow imports inside the enabled execution
    path so training, evaluation and tests can run without MLflow when tracking
    is disabled.
    """

    def __init__(
        self,
        enabled: bool,
        tracking_uri: Optional[str],
        experiment_name: str,
        artifact_location: Optional[str] = None,
        log_checkpoint_binary: bool = False,
        tracking_session_id: Optional[str] = None,
    ) -> None:
        """Initializes the tracker.

        Args:
            enabled: Whether MLflow calls should be performed.
            tracking_uri: MLflow tracking URI or local filesystem path.
            experiment_name: Name of the MLflow experiment.
            artifact_location: Optional artifact root for a newly created
                experiment.
            log_checkpoint_binary: Whether checkpoint binaries may be uploaded.
                The default is false to avoid duplicating large Replay Buffers.
            tracking_session_id: Optional default session id. When MLflow is
                enabled, ``start_run`` still requires an explicit id either via
                this value or its ``tracking_session_id`` argument.
        """
        self.enabled = bool(enabled)
        self.tracking_uri = _normalize_tracking_uri(tracking_uri)
        self.experiment_name = str(experiment_name)
        self.artifact_location = artifact_location
        self.log_checkpoint_binary = bool(log_checkpoint_binary)
        self.tracking_session_id = _normalize_session_id(tracking_session_id) if tracking_session_id else None
        self._mlflow = None
        self._run = None
        self._run_metadata: Optional[MLflowRunMetadata] = None
        self._session_started_at: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> "MLflowTracker":
        """Builds a tracker from YAML configuration and environment overrides.

        Args:
            config: Parsed project configuration.
            tracking_uri: Optional explicit tracking URI override.
            experiment_name: Optional explicit experiment name override.

        Returns:
            Configured tracker. If ``mlflow.enabled`` is false, methods become
            no-ops and MLflow is not imported.
        """
        mlflow_config = dict(config.get("mlflow", {}))
        enabled = bool(mlflow_config.get("enabled", False))
        resolved_tracking_uri = (
            tracking_uri
            or os.environ.get("ASSAULT_MLFLOW_TRACKING_URI")
            or mlflow_config.get("tracking_uri")
            or mlflow_config.get("local_directory")
        )
        resolved_experiment = (
            experiment_name
            or os.environ.get("ASSAULT_MLFLOW_EXPERIMENT")
            or mlflow_config.get("experiment_name")
            or "assault_ddqn"
        )
        artifact_location = os.environ.get("ASSAULT_MLFLOW_ARTIFACT_LOCATION") or mlflow_config.get("artifact_location")
        tracking_session_id = os.environ.get("ASSAULT_MLFLOW_SESSION_ID") or mlflow_config.get("tracking_session_id")
        return cls(
            enabled=enabled,
            tracking_uri=str(resolved_tracking_uri) if resolved_tracking_uri else None,
            experiment_name=str(resolved_experiment),
            artifact_location=str(artifact_location) if artifact_location else None,
            log_checkpoint_binary=bool(mlflow_config.get("log_checkpoint_binary", False)),
            tracking_session_id=str(tracking_session_id) if tracking_session_id else None,
        )

    @property
    def run_metadata(self) -> Optional[MLflowRunMetadata]:
        """Returns metadata for the active or most recently started run."""
        return self._run_metadata

    def __enter__(self) -> "MLflowTracker":
        """Returns this tracker for context-manager usage."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Closes an active run when leaving a context manager."""
        self.end_run(status="FAILED" if exc_type else "FINISHED")

    def start_run(
        self,
        project_run_id: str,
        tracking_mode: str = "new",
        mlflow_run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
        tracking_session_id: Optional[str] = None,
    ) -> MLflowRunMetadata:
        """Starts a new MLflow run or resumes an explicit existing run.

        Args:
            project_run_id: Logical project/checkpoint/TensorBoard identifier.
            tracking_mode: Either ``"new"`` or ``"resume"``.
            mlflow_run_id: Required when ``tracking_mode="resume"``.
            run_name: Optional display name for a new MLflow run.
            tags: Optional extra run tags.
            tracking_session_id: Unique notebook/computation session id within
                the selected MLflow run. Required when tracking is enabled.

        Returns:
            Metadata linking the logical project run and MLflow run.

        Raises:
            ValueError: If the run identity or tracking mode is invalid.
            RuntimeError: If MLflow is enabled but unavailable or misconfigured.
        """
        project_run_id = str(project_run_id).strip()
        if not project_run_id:
            raise ValueError("project_run_id must be explicit and non-empty.")
        tracking_mode = str(tracking_mode).strip().lower()
        if tracking_mode not in {"new", "resume"}:
            raise ValueError("tracking_mode must be either 'new' or 'resume'.")
        if tracking_mode == "resume" and not str(mlflow_run_id or "").strip():
            raise ValueError("tracking_mode='resume' requires explicit mlflow_run_id.")
        if tracking_mode == "new" and mlflow_run_id is not None:
            raise ValueError("tracking_mode='new' must not receive mlflow_run_id.")

        resolved_session_id = _normalize_session_id(tracking_session_id or self.tracking_session_id)

        if not self.enabled:
            self._run_metadata = MLflowRunMetadata(False, project_run_id, None, resolved_session_id, None, None, self.tracking_uri)
            return self._run_metadata

        if not resolved_session_id:
            raise ValueError("tracking_session_id must be explicit and non-empty when MLflow is enabled.")

        mlflow = self._require_mlflow()
        self._validate_tracking_uri()
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        experiment_id = self._ensure_experiment()
        base_tags = {
            "algorithm": "DDQN",
            "project_run_id": project_run_id,
            "tracking_mode": tracking_mode,
            "latest_tracking_session_id": resolved_session_id,
        }
        if tags:
            base_tags.update({str(key): _to_mlflow_value(value) for key, value in tags.items()})

        if tracking_mode == "resume":
            client = mlflow.tracking.MlflowClient()
            existing = client.get_run(str(mlflow_run_id))
            _validate_existing_project_run(existing.data, project_run_id)
            self._ensure_session_is_new(client, str(mlflow_run_id), resolved_session_id)
            self._run = mlflow.start_run(run_id=str(mlflow_run_id))
            mlflow.set_tags(base_tags)
        else:
            self._run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name or project_run_id, tags=base_tags)

        active = mlflow.active_run()
        if active is None:
            raise RuntimeError("MLflow did not create an active run.")
        self._run_metadata = MLflowRunMetadata(
            enabled=True,
            project_run_id=project_run_id,
            mlflow_run_id=active.info.run_id,
            tracking_session_id=resolved_session_id,
            experiment_name=self.experiment_name,
            experiment_id=active.info.experiment_id,
            tracking_uri=mlflow.get_tracking_uri(),
        )
        self.tracking_session_id = resolved_session_id
        self._session_started_at = _utc_now()
        self._log_param_once("identity.algorithm", "DDQN")
        self._log_param_once("identity.project_run_id", project_run_id)
        self._log_param_once("identity.experiment_name", self.experiment_name)
        return self._run_metadata

    def end_run(self, status: str = "FINISHED") -> None:
        """Ends the active MLflow run when tracking is enabled."""
        if self.enabled and self._mlflow is not None and self._mlflow.active_run() is not None:
            self._mlflow.end_run(status=status)
        self._run = None

    def log_run_context(
        self,
        config: Mapping[str, Any],
        runtime_info: Optional[Mapping[str, Any]] = None,
        git_commit: Optional[str] = None,
        git_ref: Optional[str] = None,
        project_run_id: Optional[str] = None,
        action_space: Optional[str] = None,
        observation_dtype: Optional[str] = None,
        runtime: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Logs identity, environment, preprocessing, DDQN and runtime params.

        Args:
            config: Parsed configuration used by the run.
            runtime_info: Runtime metadata, preferably from ``get_runtime_info``.
            git_commit: Git SHA executed.
            git_ref: Optional requested Git ref.
            project_run_id: Optional logical run id to validate/log.
            action_space: Optional action-space string from the environment.
            observation_dtype: Optional observed dtype.
            runtime: Runtime label such as ``local`` or ``Google Colab``.
            device: Device selected for the run.
        """
        if not self.enabled:
            return
        self._require_active_run()
        runtime_info = dict(runtime_info or {})
        project_run_id = project_run_id or self._run_metadata.project_run_id if self._run_metadata else project_run_id

        params: Dict[str, Any] = {
            "identity.seed": _get_nested(config, "reproducibility", "seed"),
            "git.commit": git_commit,
            "git.ref": git_ref,
            "runtime.name": runtime,
            "runtime.device": device,
            "environment.id": _get_nested(config, "environment", "id"),
            "environment.obs_type": _get_nested(config, "environment", "obs_type"),
            "environment.action_space": action_space,
            "environment.frameskip": _get_nested(config, "environment", "frame_skip"),
            "environment.repeat_action_probability": _get_nested(config, "environment", "repeat_action_probability"),
            "environment.full_action_space": _get_nested(config, "environment", "full_action_space"),
            "preprocessing.grayscale": _get_nested(config, "preprocessing", "grayscale"),
            "preprocessing.resize_height": _get_nested(config, "preprocessing", "resize_height"),
            "preprocessing.resize_width": _get_nested(config, "preprocessing", "resize_width"),
            "preprocessing.frame_stack": _get_nested(config, "preprocessing", "frame_stack"),
            "preprocessing.dtype": observation_dtype or _get_nested(config, "preprocessing", "dtype"),
            "preprocessing.normalize_pixels_in_env": _get_nested(config, "preprocessing", "normalize_pixels_in_env"),
            "ddqn.gamma": _get_nested(config, "agent", "gamma"),
            "ddqn.learning_rate": _get_nested(config, "agent", "learning_rate"),
            "ddqn.epsilon_start": _get_nested(config, "agent", "epsilon_start"),
            "ddqn.epsilon_final": _get_nested(config, "agent", "epsilon_final"),
            "ddqn.epsilon_decay_steps": _get_nested(config, "training", "epsilon_decay_steps"),
            "ddqn.batch_size": _get_nested(config, "replay_buffer", "batch_size"),
            "ddqn.replay_buffer_capacity": _get_nested(config, "replay_buffer", "capacity"),
            "ddqn.learning_starts": _get_nested(config, "training", "learning_starts"),
            "ddqn.train_frequency": _get_nested(config, "training", "train_frequency"),
            "ddqn.target_update_frequency": _get_nested(config, "training", "target_update_frequency"),
            "ddqn.total_timesteps": _get_nested(config, "training", "total_timesteps"),
            "versions.python": runtime_info.get("python_version"),
            "versions.gymnasium": runtime_info.get("gymnasium_version"),
            "versions.ale_py": runtime_info.get("ale_py_version"),
            "versions.torch": runtime_info.get("torch_version"),
            "versions.cuda": runtime_info.get("cuda_version"),
            "versions.mlflow": _package_version("mlflow"),
            "hardware.cpu": runtime_info.get("cpu"),
            "hardware.cpu_count_logical": runtime_info.get("cpu_count_logical"),
            "hardware.cpu_count_physical": runtime_info.get("cpu_count_physical"),
            "hardware.ram_total_gb": runtime_info.get("ram_total_gb"),
            "hardware.ram_available_gb": runtime_info.get("ram_available_gb"),
            "hardware.gpu_available": runtime_info.get("gpu_available"),
            "hardware.gpu_name": runtime_info.get("gpu_name"),
            "hardware.gpu_vram_total_gb": runtime_info.get("gpu_vram_total_gb"),
        }
        if project_run_id is not None:
            params["identity.project_run_id"] = project_run_id
        self._log_params_once(params)
        self._mlflow.set_tags(
            {
                "algorithm": "DDQN",
                "project_run_id": _to_mlflow_value(project_run_id),
                "git_commit": _to_mlflow_value(git_commit),
                "runtime": _to_mlflow_value(runtime),
            }
        )

    def log_training_summary(self, summary: Any, prefix: str = "train", tracking_session_id: Optional[str] = None) -> None:
        """Logs aggregate training metrics and a JSON summary artifact."""
        if not self.enabled:
            return
        data = _as_mapping(summary)
        metric_step = _to_int(data.get("global_step"))
        metrics = {
            f"{prefix}/initial_global_step": data.get("initial_global_step"),
            f"{prefix}/final_global_step": data.get("global_step"),
            f"{prefix}/duration_seconds": data.get("duration_seconds"),
            f"{prefix}/updates_count": data.get("updates_count"),
            f"{prefix}/last_loss": data.get("last_loss"),
            f"{prefix}/mean_loss": data.get("mean_loss"),
            f"{prefix}/last_q_mean": data.get("last_q_mean"),
            f"{prefix}/mean_q_mean": data.get("mean_q_mean"),
            f"{prefix}/final_epsilon": data.get("epsilon_final"),
            f"{prefix}/replay_buffer_size": data.get("final_replay_buffer_size"),
            f"{prefix}/episodes_completed": data.get("episodes_completed"),
        }
        if data.get("episode_rewards"):
            metrics[f"{prefix}/best_episode_reward"] = max(float(value) for value in data["episode_rewards"])
        self._log_metrics(metrics, step=metric_step)
        self.log_dict_artifact(data, "training_summary.json", tracking_session_id=tracking_session_id)

    def log_evaluation_summary(self, summary: Any, prefix: str = "eval", tracking_session_id: Optional[str] = None) -> None:
        """Logs aggregate evaluation metrics and a JSON summary artifact."""
        if not self.enabled or summary is None:
            return
        data = _as_mapping(summary)
        metrics = {
            f"{prefix}/episodes": data.get("episodes"),
            f"{prefix}/mean_reward": data.get("mean_reward"),
            f"{prefix}/median_reward": data.get("median_reward"),
            f"{prefix}/std_reward": data.get("std_reward"),
            f"{prefix}/min_reward": data.get("min_reward"),
            f"{prefix}/max_reward": data.get("max_reward"),
            f"{prefix}/epsilon": data.get("epsilon"),
        }
        lengths = data.get("episode_lengths")
        if lengths:
            metrics[f"{prefix}/mean_episode_length"] = sum(float(value) for value in lengths) / len(lengths)
        self._log_metrics(metrics)
        self.log_dict_artifact(data, "evaluation_summary.json", tracking_session_id=tracking_session_id)

    def log_checkpoint_reference(
        self,
        checkpoint: Any,
        resume_mode: str,
        project_run_id: Optional[str] = None,
        checkpoint_input_reference: Optional[str] = None,
        checkpoint_output_reference: Optional[str] = None,
        tracking_session_id: Optional[str] = None,
    ) -> None:
        """Logs lightweight checkpoint metadata without duplicating binaries."""
        if not self.enabled or checkpoint is None:
            return
        data = _as_mapping(checkpoint)
        checkpoint_path = data.get("path")
        output_reference = checkpoint_output_reference or checkpoint_path
        reference = {
            "checkpoint_path": checkpoint_path,
            "project_run_id": project_run_id or data.get("run_id") or (self._run_metadata.project_run_id if self._run_metadata else None),
            "checkpoint_step": data.get("checkpoint_step"),
            "checkpoint_size_bytes": data.get("size_bytes"),
            "resume_mode": resume_mode,
            "save_replay_buffer": data.get("save_replay_buffer"),
            "checkpoint_binary_logged": False,
            "checkpoint_input_reference": checkpoint_input_reference,
            "checkpoint_output_reference": output_reference,
        }
        self._mlflow.set_tags(
            {
                "checkpoint_path": _to_mlflow_value(reference["checkpoint_output_reference"]),
                "checkpoint_step": _to_mlflow_value(reference["checkpoint_step"]),
                "checkpoint_resume_mode": _to_mlflow_value(resume_mode),
            }
        )
        self._log_metrics(
            {
                "checkpoint/step": reference["checkpoint_step"],
                "checkpoint/size_bytes": reference["checkpoint_size_bytes"],
            }
        )
        self.log_dict_artifact(reference, "checkpoint_reference.json", tracking_session_id=tracking_session_id)

    def log_config_snapshot(self, config: Mapping[str, Any], artifact_file: str = "config/ddqn_config.json") -> None:
        """Logs the effective configuration as a JSON artifact."""
        if self.enabled:
            self.log_dict_artifact(dict(config), artifact_file, session_scoped=False)

    def log_runtime_metadata(
        self,
        runtime_info: Mapping[str, Any],
        git_commit: Optional[str] = None,
        runtime: Optional[str] = None,
        artifact_file: str = "runtime.json",
        tracking_session_id: Optional[str] = None,
    ) -> None:
        """Logs runtime metadata as a JSON artifact."""
        if not self.enabled:
            return
        data = dict(runtime_info)
        data.update({"git_commit": git_commit, "runtime": runtime, "mlflow_version": _package_version("mlflow")})
        self.log_dict_artifact(data, artifact_file, tracking_session_id=tracking_session_id)

    def log_session_metadata(
        self,
        tracking_mode: str,
        runtime_info: Optional[Mapping[str, Any]] = None,
        git_commit: Optional[str] = None,
        git_ref: Optional[str] = None,
        runtime: Optional[str] = None,
        device: Optional[str] = None,
        initial_global_step: Optional[int] = None,
        final_global_step: Optional[int] = None,
        checkpoint_input_reference: Optional[str] = None,
        checkpoint_output_reference: Optional[str] = None,
        checkpoint_input_loaded: Optional[bool] = None,
        restored_checkpoint_path: Optional[str] = None,
        restored_global_step: Optional[int] = None,
        replay_buffer_restored: Optional[bool] = None,
        resume_mode: Optional[str] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        tracking_session_id: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Logs immutable per-session metadata under ``sessions/<id>/``."""
        if not self.enabled:
            return
        metadata = self._require_run_metadata()
        runtime_info = dict(runtime_info or {})
        session_id = self._resolve_session_id(tracking_session_id)
        session_metadata: Dict[str, Any] = {
            "tracking_session_id": session_id,
            "project_run_id": metadata.project_run_id,
            "mlflow_run_id": metadata.mlflow_run_id,
            "tracking_mode": str(tracking_mode).strip().lower(),
            "started_at": started_at or self._session_started_at,
            "ended_at": ended_at or _utc_now(),
            "duration_seconds": duration_seconds,
            "runtime": runtime,
            "git_commit": git_commit,
            "git_ref": git_ref,
            "device": device,
            "gpu_available": runtime_info.get("gpu_available"),
            "gpu_name": runtime_info.get("gpu_name"),
            "cpu": runtime_info.get("cpu"),
            "initial_global_step": initial_global_step,
            "final_global_step": final_global_step,
            "checkpoint_input_reference": checkpoint_input_reference,
            "checkpoint_output_reference": checkpoint_output_reference,
            "checkpoint_input_loaded": checkpoint_input_loaded,
            "restored_checkpoint_path": restored_checkpoint_path,
            "restored_global_step": restored_global_step,
            "replay_buffer_restored": replay_buffer_restored,
            "resume_mode": resume_mode,
            "versions": {
                "python": runtime_info.get("python_version"),
                "torch": runtime_info.get("torch_version"),
                "cuda": runtime_info.get("cuda_version"),
                "mlflow": _package_version("mlflow"),
            },
        }
        if extra:
            session_metadata.update(dict(extra))
        self._mlflow.set_tags(
            {
                "latest_tracking_session_id": _to_mlflow_value(session_id),
                "tracking_mode": _to_mlflow_value(session_metadata["tracking_mode"]),
            }
        )
        self.log_dict_artifact(session_metadata, "session_metadata.json", tracking_session_id=session_id)

    def log_dict_artifact(
        self,
        data: Mapping[str, Any],
        artifact_file: str,
        session_scoped: bool = True,
        tracking_session_id: Optional[str] = None,
    ) -> None:
        """Logs a small dictionary artifact to MLflow as JSON."""
        if not self.enabled:
            return
        self._require_active_run()
        serializable = _json_safe(data)
        artifact_path = str(artifact_file).replace("\\", "/").lstrip("/")
        if session_scoped and not artifact_path.startswith("sessions/"):
            artifact_path = f"{self._session_root(tracking_session_id)}/{artifact_path}"
        self._mlflow.log_dict(serializable, artifact_path)

    def list_session_artifacts(self, tracking_session_id: Optional[str] = None) -> Any:
        """Lists artifacts recorded under the active run's session namespace."""
        if not self.enabled:
            return []
        metadata = self._require_run_metadata()
        self._require_mlflow()
        return self._mlflow.tracking.MlflowClient().list_artifacts(
            metadata.mlflow_run_id,
            self._session_root(tracking_session_id),
        )

    def get_run(self, mlflow_run_id: Optional[str] = None) -> Any:
        """Fetches an MLflow run with ``MlflowClient`` for validation."""
        if not self.enabled:
            return None
        self._require_mlflow()
        if self.tracking_uri:
            self._mlflow.set_tracking_uri(self.tracking_uri)
        run_id = mlflow_run_id or (self._run_metadata.mlflow_run_id if self._run_metadata else None)
        if not run_id:
            raise ValueError("mlflow_run_id is required to query a run.")
        return self._mlflow.tracking.MlflowClient().get_run(run_id)

    def _require_mlflow(self):
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("mlflow.enabled=true but MLflow is not installed.") from exc
        self._mlflow = mlflow
        return mlflow

    def _require_active_run(self) -> None:
        self._require_mlflow()
        if self._mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")

    def _require_run_metadata(self) -> MLflowRunMetadata:
        if self._run_metadata is None:
            raise RuntimeError("No MLflow run metadata. Call start_run() first.")
        return self._run_metadata

    def _resolve_session_id(self, tracking_session_id: Optional[str] = None) -> str:
        session_id = _normalize_session_id(tracking_session_id or self.tracking_session_id)
        if not session_id:
            raise ValueError("tracking_session_id must be explicit and non-empty when MLflow is enabled.")
        return session_id

    def _session_root(self, tracking_session_id: Optional[str] = None) -> str:
        return f"sessions/{self._resolve_session_id(tracking_session_id)}"

    def _ensure_session_is_new(self, client: Any, mlflow_run_id: str, tracking_session_id: str) -> None:
        session_root = f"sessions/{tracking_session_id}"
        existing_artifacts = client.list_artifacts(mlflow_run_id, session_root)
        for artifact in existing_artifacts:
            if artifact.path == f"{session_root}/session_metadata.json":
                raise RuntimeError(
                    f"tracking_session_id='{tracking_session_id}' already exists for MLflow run {mlflow_run_id}."
                )

    def _ensure_experiment(self) -> str:
        mlflow = self._require_mlflow()
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            if self.artifact_location:
                return mlflow.create_experiment(self.experiment_name, artifact_location=self.artifact_location)
            return mlflow.create_experiment(self.experiment_name)
        return experiment.experiment_id

    def _validate_tracking_uri(self) -> None:
        if not self.tracking_uri:
            return
        parsed = urlparse(self.tracking_uri)
        if parsed.scheme in {"", "file"}:
            raw_path = parsed.path if parsed.scheme == "file" else self.tracking_uri
            if parsed.netloc and parsed.scheme == "file":
                raw_path = f"//{parsed.netloc}{parsed.path}"
            if not raw_path:
                raise RuntimeError("MLflow file tracking URI does not contain a path.")
            if parsed.scheme == "file" and len(raw_path) >= 4 and raw_path[0] == "/" and raw_path[2] == ":":
                raw_path = raw_path[1:]
            path = Path(raw_path)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(f"MLflow tracking URI is not writable: {self.tracking_uri}") from exc

    def _log_params_once(self, params: Mapping[str, Any]) -> None:
        for key, value in params.items():
            self._log_param_once(key, value)

    def _log_param_once(self, key: str, value: Any) -> None:
        if value is None:
            return
        run_id = self._run_metadata.mlflow_run_id if self._run_metadata else None
        current = {}
        if run_id:
            current = self._mlflow.tracking.MlflowClient().get_run(run_id).data.params
        serialized = _to_mlflow_value(value)
        if key in current:
            if current[key] != serialized:
                raise ValueError(f"MLflow param mismatch for {key}: {current[key]} != {serialized}")
            return
        self._mlflow.log_param(key, serialized)

    def _log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        self._require_active_run()
        for key, value in metrics.items():
            if value is None:
                continue
            scalar = _to_float(value)
            if scalar is None:
                continue
            if not math.isfinite(scalar):
                raise ValueError(f"Non-finite MLflow metric for {key}: {scalar}")
            self._mlflow.log_metric(key, scalar, step=step)


def _validate_existing_project_run(run_data: Any, project_run_id: str) -> None:
    existing = run_data.params.get("identity.project_run_id") or run_data.tags.get("project_run_id")
    if existing is not None and existing != project_run_id:
        raise ValueError(f"MLflow run project_run_id mismatch: {existing} != {project_run_id}.")


def _normalize_tracking_uri(tracking_uri: Optional[str]) -> Optional[str]:
    if tracking_uri is None:
        return None
    uri = str(tracking_uri)
    parsed = urlparse(uri)
    if parsed.scheme == "":
        path = Path(uri)
        return path.as_uri() if path.is_absolute() else uri
    if len(parsed.scheme) == 1 and len(uri) >= 3 and uri[1] == ":":
        return Path(uri).as_uri()
    return uri


def _normalize_session_id(tracking_session_id: Optional[str]) -> Optional[str]:
    if tracking_session_id is None:
        return None
    session_id = str(tracking_session_id).strip()
    if not session_id:
        return None
    if session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        raise ValueError("tracking_session_id must be a single path-safe name.")
    return session_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_nested(config: Mapping[str, Any], section: str, key: str) -> Any:
    value = config.get(section, {})
    if not isinstance(value, Mapping):
        return None
    return value.get(key)


def _as_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping or object with as_dict(), got {type(value)!r}.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _to_mlflow_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(_json_safe(value), sort_keys=True)


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _package_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
