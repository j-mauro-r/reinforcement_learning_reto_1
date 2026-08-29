"""Automated session bootstrap for resumable Assault DDQN experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

import torch


SCHEMA_VERSION = 1
VALID_REQUESTED_MODES = {"auto", "new", "resume"}
VALID_TRACKING_MODES = {"new", "resume"}
SESSION_PREFIX = "session_"
MANIFEST_FILENAME = "experiment_state.json"


@dataclass(frozen=True)
class TrainingSessionContext:
    """Resolved context required to execute one training session."""

    project_run_id: str
    tracking_mode: str
    mlflow_run_id: Optional[str]
    tracking_session_id: str
    tracking_uri: Optional[str]
    mlflow_enabled: bool
    checkpoint_root: Path
    tensorboard_root: Path
    checkpoint_input: Optional[Path]
    resume_mode: str
    target_timesteps: int
    restored_expected_step: Optional[int]
    bootstrap_ref: str
    bootstrap_commit: str
    config_fingerprint: str
    manifest_path: Path

    def as_dict(self) -> Dict[str, Any]:
        """Returns a serializable representation."""
        return {
            "project_run_id": self.project_run_id,
            "tracking_mode": self.tracking_mode,
            "mlflow_run_id": self.mlflow_run_id,
            "tracking_session_id": self.tracking_session_id,
            "tracking_uri": self.tracking_uri,
            "mlflow_enabled": self.mlflow_enabled,
            "checkpoint_root": str(self.checkpoint_root),
            "tensorboard_root": str(self.tensorboard_root),
            "checkpoint_input": str(self.checkpoint_input) if self.checkpoint_input else None,
            "resume_mode": self.resume_mode,
            "target_timesteps": self.target_timesteps,
            "restored_expected_step": self.restored_expected_step,
            "bootstrap_ref": self.bootstrap_ref,
            "bootstrap_commit": self.bootstrap_commit,
            "config_fingerprint": self.config_fingerprint,
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True)
class ExperimentState:
    """Persistent orchestration state for one logical experiment."""

    schema_version: int
    project_run_id: str
    mlflow_run_id: Optional[str]
    latest_tracking_session_id: str
    latest_checkpoint: str
    latest_global_step: int
    resume_mode: str
    bootstrap_commit: str
    config_fingerprint: str
    updated_at: str

    def as_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serializable manifest payload."""
        return {
            "schema_version": self.schema_version,
            "project_run_id": self.project_run_id,
            "mlflow_run_id": self.mlflow_run_id,
            "latest_tracking_session_id": self.latest_tracking_session_id,
            "latest_checkpoint": self.latest_checkpoint,
            "latest_global_step": self.latest_global_step,
            "resume_mode": self.resume_mode,
            "bootstrap_commit": self.bootstrap_commit,
            "config_fingerprint": self.config_fingerprint,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ExperimentInspection:
    """Non-destructive diagnostics for a persisted experiment state."""

    project_run_id: str
    manifest_path: Path
    manifest_exists: bool
    issues: List[str]
    manifest: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        """Whether no consistency issue was detected."""
        return not self.issues

    def as_dict(self) -> Dict[str, Any]:
        """Returns a notebook-friendly dictionary."""
        return {
            "project_run_id": self.project_run_id,
            "manifest_path": str(self.manifest_path),
            "manifest_exists": self.manifest_exists,
            "ok": self.ok,
            "issues": list(self.issues),
            "manifest": self.manifest,
        }


def prepare_training_session(
    base_path: str | Path,
    project_run_id: str,
    target_timesteps: int,
    requested_mode: str = "auto",
    config: Optional[Mapping[str, Any]] = None,
    checkpoint_root: str | Path | None = None,
    tensorboard_root: str | Path | None = None,
    tracking_uri: Optional[str] = None,
    resume_mode: str = "resume_full",
    bootstrap_ref: Optional[str] = None,
    bootstrap_commit: Optional[str] = None,
    mlflow_enabled: Optional[bool] = None,
    mlflow_experiment_name: Optional[str] = None,
) -> TrainingSessionContext:
    """Resolves and validates one DDQN training session before training starts.

    Args:
        base_path: Persistent experiment base directory.
        project_run_id: Logical experiment id.
        target_timesteps: Global timestep target for this session.
        requested_mode: ``auto``, ``new`` or ``resume``.
        config: Parsed DDQN configuration.
        checkpoint_root: Optional persistent checkpoint root override.
        tensorboard_root: Optional persistent TensorBoard root override.
        tracking_uri: Optional MLflow tracking URI override.
        resume_mode: Checkpoint restore mode, normally ``resume_full``.
        bootstrap_ref: Explicit Git ref requested by the notebook/bootstrap.
        bootstrap_commit: Explicit Git SHA resolved by the execution bootstrap.
        mlflow_enabled: Override for ``config["mlflow"]["enabled"]``.
        mlflow_experiment_name: Optional MLflow experiment name override.

    Returns:
        Fully resolved training session context.

    Raises:
        ValueError: If the manifest, config or target are inconsistent.
        FileNotFoundError: If a resume checkpoint is missing.
        RuntimeError: If required MLflow/session evidence is inconsistent.
    """
    project_run_id = _require_identifier("project_run_id", project_run_id)
    selected_mode = str(requested_mode or "auto").strip().lower()
    if selected_mode not in VALID_REQUESTED_MODES:
        raise ValueError("requested_mode must be one of: auto, new, resume.")
    if int(target_timesteps) <= 0:
        raise ValueError("target_timesteps must be positive.")
    selected_ref = _require_identifier("bootstrap_ref", bootstrap_ref or os.environ.get("ASSAULT_BOOTSTRAP_REF"))
    selected_commit = _require_identifier("bootstrap_commit", bootstrap_commit or os.environ.get("ASSAULT_BOOTSTRAP_COMMIT"))

    base = Path(base_path)
    resolved_config: Mapping[str, Any] = config or {}
    mlflow_config = resolved_config.get("mlflow", {}) if isinstance(resolved_config, Mapping) else {}
    checkpointing_config = resolved_config.get("checkpointing", {}) if isinstance(resolved_config, Mapping) else {}
    tensorboard_config = resolved_config.get("tensorboard", {}) if isinstance(resolved_config, Mapping) else {}
    mlflow_is_enabled = bool(mlflow_config.get("enabled", False) if mlflow_enabled is None else mlflow_enabled)
    resolved_tracking_uri = _normalize_tracking_uri(
        tracking_uri
        or os.environ.get("ASSAULT_MLFLOW_TRACKING_URI")
        or mlflow_config.get("tracking_uri")
        or base / str(mlflow_config.get("local_directory", "logs/mlflow"))
    )
    resolved_checkpoint_root = Path(
        checkpoint_root
        or os.environ.get("ASSAULT_CHECKPOINT_DIR")
        or base / str(checkpointing_config.get("directory", "checkpoints"))
    )
    resolved_tensorboard_root = Path(
        tensorboard_root
        or os.environ.get("ASSAULT_TENSORBOARD_DIR")
        or base / str(tensorboard_config.get("directory", "logs/tensorboard"))
    )
    config_fingerprint = compute_config_fingerprint(resolved_config)
    manifest_path = _manifest_path(base, project_run_id)
    manifest_exists = manifest_path.exists()

    if selected_mode == "auto":
        tracking_mode = "resume" if manifest_exists else "new"
    else:
        tracking_mode = selected_mode
    if tracking_mode == "new" and manifest_exists:
        raise FileExistsError(f"Experiment manifest already exists: {manifest_path}")
    if tracking_mode == "resume" and not manifest_exists:
        raise FileNotFoundError(f"Experiment manifest not found for resume: {manifest_path}")

    if tracking_mode == "new":
        session_id = "session_001"
        return TrainingSessionContext(
            project_run_id=project_run_id,
            tracking_mode="new",
            mlflow_run_id=None,
            tracking_session_id=session_id,
            tracking_uri=resolved_tracking_uri,
            mlflow_enabled=mlflow_is_enabled,
            checkpoint_root=resolved_checkpoint_root,
            tensorboard_root=resolved_tensorboard_root,
            checkpoint_input=None,
            resume_mode=resume_mode,
            target_timesteps=int(target_timesteps),
            restored_expected_step=None,
            bootstrap_ref=selected_ref,
            bootstrap_commit=selected_commit,
            config_fingerprint=config_fingerprint,
            manifest_path=manifest_path,
        )

    manifest = _load_manifest(manifest_path)
    _validate_manifest_identity(manifest, project_run_id)
    if manifest["config_fingerprint"] != config_fingerprint:
        raise ValueError("Config fingerprint mismatch for resume.")
    checkpoint_input = Path(manifest["latest_checkpoint"])
    if not checkpoint_input.exists():
        raise FileNotFoundError(f"Manifest checkpoint does not exist: {checkpoint_input}")
    restored_step = int(manifest["latest_global_step"])
    _validate_checkpoint_payload(
        checkpoint_input,
        project_run_id=project_run_id,
        expected_step=restored_step,
        expected_fingerprint=config_fingerprint,
        config=resolved_config,
        resume_mode=str(manifest.get("resume_mode") or resume_mode),
    )
    if int(target_timesteps) <= restored_step:
        raise ValueError("target_timesteps must be greater than the restored global_step.")
    mlflow_run_id = manifest.get("mlflow_run_id")
    if mlflow_is_enabled:
        _validate_mlflow_run(
            tracking_uri=resolved_tracking_uri,
            experiment_name=mlflow_experiment_name or mlflow_config.get("experiment_name") or "assault_ddqn",
            mlflow_run_id=mlflow_run_id,
            project_run_id=project_run_id,
        )
    session_id = _next_session_id(str(manifest["latest_tracking_session_id"]))
    if mlflow_is_enabled:
        _ensure_mlflow_session_absent(resolved_tracking_uri, mlflow_run_id, session_id)

    return TrainingSessionContext(
        project_run_id=project_run_id,
        tracking_mode="resume",
        mlflow_run_id=str(mlflow_run_id),
        tracking_session_id=session_id,
        tracking_uri=resolved_tracking_uri,
        mlflow_enabled=mlflow_is_enabled,
        checkpoint_root=resolved_checkpoint_root,
        tensorboard_root=resolved_tensorboard_root,
        checkpoint_input=checkpoint_input,
        resume_mode=str(manifest.get("resume_mode") or resume_mode),
        target_timesteps=int(target_timesteps),
        restored_expected_step=restored_step,
        bootstrap_ref=selected_ref,
        bootstrap_commit=selected_commit,
        config_fingerprint=config_fingerprint,
        manifest_path=manifest_path,
    )


def update_experiment_state_after_success(
    context: TrainingSessionContext,
    mlflow_run_id: Optional[str],
    checkpoint_output: str | Path,
    final_global_step: int,
) -> ExperimentState:
    """Atomically updates the experiment manifest after a successful session.

    The caller must invoke this only after training, checkpoint validation,
    MLflow artifact logging and MLflow ``FINISHED`` closure have completed.
    """
    checkpoint_path = Path(checkpoint_output)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint output does not exist: {checkpoint_path}")
    if int(final_global_step) != int(context.target_timesteps):
        raise ValueError("final_global_step must match the session target_timesteps.")
    if context.tracking_mode == "resume" and context.restored_expected_step is not None:
        if int(final_global_step) <= int(context.restored_expected_step):
            raise ValueError("final_global_step must be greater than restored_expected_step.")
    if context.mlflow_enabled and mlflow_run_id is None:
        raise ValueError("mlflow_run_id is required to persist a tracked session.")

    state = ExperimentState(
        schema_version=SCHEMA_VERSION,
        project_run_id=context.project_run_id,
        mlflow_run_id=str(mlflow_run_id) if mlflow_run_id else None,
        latest_tracking_session_id=context.tracking_session_id,
        latest_checkpoint=str(checkpoint_path),
        latest_global_step=int(final_global_step),
        resume_mode=context.resume_mode,
        bootstrap_commit=context.bootstrap_commit,
        config_fingerprint=context.config_fingerprint,
        updated_at=_utc_now(),
    )
    _atomic_write_json(context.manifest_path, state.as_dict())
    return state


def inspect_experiment_state(
    base_path: str | Path,
    project_run_id: str,
    config: Optional[Mapping[str, Any]] = None,
    tracking_uri: Optional[str] = None,
) -> ExperimentInspection:
    """Reports persisted-state issues without modifying files or MLflow."""
    project_run_id = _require_identifier("project_run_id", project_run_id)
    manifest_path = _manifest_path(Path(base_path), project_run_id)
    issues: List[str] = []
    manifest: Optional[Dict[str, Any]] = None
    if not manifest_path.exists():
        issues.append("manifest_missing")
        return ExperimentInspection(project_run_id, manifest_path, False, issues)
    try:
        manifest = _load_manifest(manifest_path)
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        issues.append(f"manifest_invalid:{exc}")
        return ExperimentInspection(project_run_id, manifest_path, True, issues)

    try:
        _validate_manifest_identity(manifest, project_run_id)
    except ValueError as exc:
        issues.append(f"manifest_identity_mismatch:{exc}")
    checkpoint_path = Path(str(manifest.get("latest_checkpoint", "")))
    if not checkpoint_path.exists():
        issues.append("manifest_checkpoint_missing")
    elif config is not None:
        try:
            _validate_checkpoint_payload(
                checkpoint_path,
                project_run_id=project_run_id,
                expected_step=int(manifest.get("latest_global_step", -1)),
                expected_fingerprint=str(manifest.get("config_fingerprint")),
                config=config,
                resume_mode=str(manifest.get("resume_mode") or "resume_full"),
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must collect issues.
            issues.append(f"checkpoint_inconsistent:{exc}")
    mlflow_run_id = manifest.get("mlflow_run_id")
    if mlflow_run_id and tracking_uri:
        try:
            run = _get_mlflow_run(_normalize_tracking_uri(tracking_uri), str(mlflow_run_id))
            status = getattr(run.info, "status", None)
            if status not in {None, "FINISHED"}:
                issues.append(f"mlflow_run_status:{status}")
            existing = run.data.params.get("identity.project_run_id") or run.data.tags.get("project_run_id")
            if existing and existing != project_run_id:
                issues.append("mlflow_project_run_id_mismatch")
        except Exception as exc:  # noqa: BLE001 - diagnostics must collect issues.
            issues.append(f"mlflow_run_unavailable:{exc}")
    return ExperimentInspection(project_run_id, manifest_path, True, issues, manifest)


def compute_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Computes a deterministic fingerprint of resume-invariant config fields."""
    invariants = {
        "environment": _select(config, "environment", ("id", "obs_type", "frame_skip", "repeat_action_probability", "full_action_space")),
        "preprocessing": _select(
            config,
            "preprocessing",
            ("grayscale", "resize_height", "resize_width", "frame_stack", "dtype", "normalize_pixels_in_env"),
        ),
        "network": _select(config, "network", ("input_channels", "num_actions")),
        "agent": _select(config, "agent", ("gamma", "learning_rate", "epsilon_start", "epsilon_final")),
        "epsilon_policy": _select(config, "training", ("epsilon_decay_steps",)),
        "replay_buffer": _select(config, "replay_buffer", ("capacity", "batch_size")),
        "training": _select(config, "training", ("train_frequency", "target_update_frequency")),
        "reproducibility": _select(config, "reproducibility", ("seed",)),
    }
    payload = json.dumps(_json_safe(invariants), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_path(base_path: Path, project_run_id: str) -> Path:
    return base_path / "experiments" / project_run_id / MANIFEST_FILENAME


def _load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "project_run_id",
        "mlflow_run_id",
        "latest_tracking_session_id",
        "latest_checkpoint",
        "latest_global_step",
        "resume_mode",
        "bootstrap_commit",
        "config_fingerprint",
        "updated_at",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Experiment manifest missing required keys: {missing}")
    if int(data["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment manifest schema_version: {data['schema_version']}")
    return data


def _validate_manifest_identity(manifest: Mapping[str, Any], project_run_id: str) -> None:
    if manifest.get("project_run_id") != project_run_id:
        raise ValueError(f"Manifest project_run_id mismatch: {manifest.get('project_run_id')} != {project_run_id}.")
    _parse_session_id(str(manifest.get("latest_tracking_session_id")))


def _validate_checkpoint_payload(
    checkpoint_path: Path,
    project_run_id: str,
    expected_step: int,
    expected_fingerprint: str,
    config: Mapping[str, Any],
    resume_mode: str,
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("run_id") != project_run_id:
        raise ValueError(f"Checkpoint run_id mismatch: {payload.get('run_id')} != {project_run_id}.")
    global_step = int(payload.get("global_step", -1))
    if global_step != int(expected_step):
        raise ValueError(f"Checkpoint global_step mismatch: {global_step} != {expected_step}.")
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Checkpoint missing config mapping.")
    checkpoint_fingerprint = compute_config_fingerprint(checkpoint_config)
    if checkpoint_fingerprint != expected_fingerprint or checkpoint_fingerprint != compute_config_fingerprint(config):
        raise ValueError("Checkpoint config fingerprint mismatch.")
    if resume_mode == "resume_full":
        replay_state = payload.get("replay_buffer_state")
        if replay_state is None:
            raise ValueError("resume_full requires replay_buffer_state.")
        _validate_replay_buffer_state(replay_state, config)


def _validate_replay_buffer_state(replay_state: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected_capacity = int(config["replay_buffer"]["capacity"])
    expected_shape = (
        int(config["network"]["input_channels"]),
        int(config["preprocessing"]["resize_height"]),
        int(config["preprocessing"]["resize_width"]),
    )
    if int(replay_state.get("capacity", -1)) != expected_capacity:
        raise ValueError(f"Replay buffer capacity mismatch: {replay_state.get('capacity')} != {expected_capacity}.")
    if tuple(replay_state.get("state_shape", ())) != expected_shape:
        raise ValueError(f"Replay buffer state_shape mismatch: {replay_state.get('state_shape')} != {expected_shape}.")
    size = int(replay_state.get("size", -1))
    states = replay_state.get("states")
    next_states = replay_state.get("next_states")
    if tuple(getattr(states, "shape", ())) != (size, *expected_shape):
        raise ValueError(f"Invalid replay buffer states shape: {getattr(states, 'shape', None)}.")
    if tuple(getattr(next_states, "shape", ())) != (size, *expected_shape):
        raise ValueError(f"Invalid replay buffer next_states shape: {getattr(next_states, 'shape', None)}.")


def _validate_mlflow_run(
    tracking_uri: Optional[str],
    experiment_name: str,
    mlflow_run_id: Optional[str],
    project_run_id: str,
) -> None:
    if not mlflow_run_id:
        raise ValueError("Manifest must contain mlflow_run_id when MLflow is enabled.")
    run = _get_mlflow_run(tracking_uri, str(mlflow_run_id))
    existing_project_run_id = run.data.params.get("identity.project_run_id") or run.data.tags.get("project_run_id")
    if existing_project_run_id != project_run_id:
        raise ValueError(f"MLflow run project_run_id mismatch: {existing_project_run_id} != {project_run_id}.")
    experiment_id = getattr(run.info, "experiment_id", None)
    if experiment_id is not None:
        mlflow = _require_mlflow()
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment(experiment_id)
        if experiment and experiment.name != experiment_name:
            raise ValueError(f"MLflow experiment mismatch: {experiment.name} != {experiment_name}.")


def _ensure_mlflow_session_absent(tracking_uri: Optional[str], mlflow_run_id: Optional[str], tracking_session_id: str) -> None:
    if not mlflow_run_id:
        raise ValueError("mlflow_run_id is required to validate session artifacts.")
    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    session_root = f"sessions/{tracking_session_id}"
    for artifact in client.list_artifacts(str(mlflow_run_id), session_root):
        if artifact.path == f"{session_root}/session_metadata.json":
            raise RuntimeError(f"tracking_session_id='{tracking_session_id}' already exists for MLflow run {mlflow_run_id}.")


def _get_mlflow_run(tracking_uri: Optional[str], mlflow_run_id: str) -> Any:
    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.tracking.MlflowClient().get_run(mlflow_run_id)


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow validation requires mlflow to be installed.") from exc
    return mlflow


def _next_session_id(latest_session_id: str) -> str:
    number = _parse_session_id(latest_session_id)
    return f"{SESSION_PREFIX}{number + 1:03d}"


def _parse_session_id(session_id: str) -> int:
    if not session_id.startswith(SESSION_PREFIX):
        raise ValueError(f"Invalid tracking_session_id: {session_id}")
    suffix = session_id[len(SESSION_PREFIX) :]
    if len(suffix) != 3 or not suffix.isdigit():
        raise ValueError(f"Invalid tracking_session_id: {session_id}")
    return int(suffix)


def _require_identifier(name: str, value: Optional[str]) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be explicit and non-empty.")
    if text.startswith("<") and text.endswith(">"):
        raise ValueError(f"{name} must not be a placeholder.")
    return text


def _normalize_tracking_uri(tracking_uri: Optional[str | Path]) -> Optional[str]:
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


def _select(config: Mapping[str, Any], section: str, keys: tuple[str, ...]) -> Dict[str, Any]:
    values = config.get(section, {})
    if not isinstance(values, Mapping):
        values = {}
    return {key: copy.deepcopy(values.get(key)) for key in keys}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
