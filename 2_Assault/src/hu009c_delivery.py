"""HU009C delivery orchestration helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .session_bootstrap import compute_config_fingerprint


VALID_EXECUTION_MODES = {"auto", "train", "delivery"}
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint_step_(\d+)\.pt$")


@dataclass(frozen=True)
class DeliveryExecutionMode:
    """Resolved execution mode for HU009C notebook orchestration."""

    execution_mode: str
    auto_resolution: str
    run_training: bool
    training_required: bool
    delivery_required: bool
    hu009c_post_training_ready: bool
    training_session_bootstrap_skipped: bool
    project_run_id: str
    target_timesteps: int
    final_checkpoint_path: Optional[Path] = None
    session_context: Optional[Any] = None

    def as_dict(self) -> dict[str, Any]:
        """Returns a notebook-friendly representation."""
        return {
            "ASSAULT_EXECUTION_MODE": self.execution_mode,
            "AUTO_RESOLUTION": self.auto_resolution,
            "ASSAULT_RUN_TRAINING": int(self.run_training),
            "training_required": self.training_required,
            "delivery_required": self.delivery_required,
            "HU009C_POST_TRAINING_READY": self.hu009c_post_training_ready,
            "training_session_bootstrap_skipped": self.training_session_bootstrap_skipped,
            "project_run_id": self.project_run_id,
            "target_timesteps": self.target_timesteps,
            "final_checkpoint_path": str(self.final_checkpoint_path) if self.final_checkpoint_path else None,
            "has_training_session_context": self.session_context is not None,
        }


def resolve_hu009c_execution_mode(
    run_training: bool | None,
    project_run_id: str,
    target_timesteps: int,
    prepare_training_session_fn: Callable[..., Any],
    prepare_training_session_kwargs: Mapping[str, Any],
    execution_mode: str = "auto",
    final_checkpoint_path: str | Path | None = None,
) -> DeliveryExecutionMode:
    """Resolves HU009C orchestration as auto/new-resume/delivery.

    In ``auto`` mode a completed final checkpoint goes directly to delivery.
    Otherwise the existing session bootstrap decides between NEW and RESUME.
    Before that bootstrap runs, this helper repairs the interruption case where
    a periodic checkpoint reached persistent storage but the successful-session
    manifest was not yet written.
    """
    if not str(project_run_id).strip():
        raise ValueError("project_run_id must be explicit and non-empty.")
    if int(target_timesteps) <= 0:
        raise ValueError("target_timesteps must be positive.")
    selected_mode = _normalize_execution_mode(execution_mode, run_training)
    checkpoint_path = Path(final_checkpoint_path) if final_checkpoint_path is not None else None
    final_checkpoint_exists = _checkpoint_exists(checkpoint_path)

    if selected_mode == "delivery":
        if not final_checkpoint_exists:
            raise FileNotFoundError(f"Forced delivery requires an existing final checkpoint: {checkpoint_path}")
        return DeliveryExecutionMode(
            execution_mode=selected_mode,
            auto_resolution="DELIVERY",
            run_training=False,
            training_required=False,
            delivery_required=True,
            hu009c_post_training_ready=True,
            training_session_bootstrap_skipped=True,
            project_run_id=str(project_run_id),
            target_timesteps=int(target_timesteps),
            final_checkpoint_path=checkpoint_path,
            session_context=None,
        )

    if selected_mode == "auto" and final_checkpoint_exists:
        return DeliveryExecutionMode(
            execution_mode=selected_mode,
            auto_resolution="DELIVERY",
            run_training=False,
            training_required=False,
            delivery_required=True,
            hu009c_post_training_ready=True,
            training_session_bootstrap_skipped=True,
            project_run_id=str(project_run_id),
            target_timesteps=int(target_timesteps),
            final_checkpoint_path=checkpoint_path,
            session_context=None,
        )

    recovery_rollback: Optional[str] = None
    if selected_mode in {"auto", "train"}:
        recovery_rollback = _recover_interrupted_periodic_checkpoint(
            project_run_id=str(project_run_id),
            target_timesteps=int(target_timesteps),
            prepare_training_session_kwargs=prepare_training_session_kwargs,
        )

    try:
        session_context = prepare_training_session_fn(**dict(prepare_training_session_kwargs))
    except Exception:
        if recovery_rollback is not None:
            _rollback_recovery_manifest(
                _recovery_manifest_path(prepare_training_session_kwargs, str(project_run_id)),
                recovery_rollback,
            )
        raise

    resolution = "NEW" if getattr(session_context, "tracking_mode", None) == "new" else "RESUME"
    return DeliveryExecutionMode(
        execution_mode=selected_mode,
        auto_resolution=resolution,
        run_training=True,
        training_required=True,
        delivery_required=False,
        hu009c_post_training_ready=False,
        training_session_bootstrap_skipped=False,
        project_run_id=str(project_run_id),
        target_timesteps=int(target_timesteps),
        final_checkpoint_path=checkpoint_path,
        session_context=session_context,
    )


def _recover_interrupted_periodic_checkpoint(
    project_run_id: str,
    target_timesteps: int,
    prepare_training_session_kwargs: Mapping[str, Any],
) -> Optional[str]:
    """Creates or refreshes a recovery manifest for periodic checkpoints.

    Returns:
        ``None`` when no recovery mutation was necessary, an empty string when
        a new synthetic manifest was created, or the previous manifest text
        when an existing synthetic recovery manifest was refreshed. The caller
        uses this value to roll back safely if authoritative bootstrap
        validation rejects the selected checkpoint.
    """
    kwargs = dict(prepare_training_session_kwargs)
    base_path = Path(kwargs.get("base_path") or Path.cwd())
    manifest_path = _recovery_manifest_path(kwargs, project_run_id)
    previous_manifest_text: Optional[str] = None
    previous_step = -1

    if manifest_path.exists():
        previous_manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(previous_manifest_text)
        if not bool(manifest.get("recovered_from_periodic_checkpoint")):
            return None
        previous_step = int(manifest.get("latest_global_step", -1))

    config = kwargs.get("config")
    checkpoint_root = _resolve_checkpoint_root(base_path, config, kwargs.get("checkpoint_root"))
    latest = _latest_periodic_checkpoint(checkpoint_root / project_run_id)
    if latest is None:
        return None
    checkpoint_path, checkpoint_step = latest
    if checkpoint_step <= previous_step:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("Interrupted checkpoint recovery requires the resolved training config.")
    if checkpoint_step <= 0:
        raise ValueError(f"Interrupted checkpoint step must be positive: {checkpoint_path}")
    if checkpoint_step >= int(target_timesteps):
        raise ValueError(
            "Orphan checkpoint step must be lower than target_timesteps; "
            f"found step={checkpoint_step}, target={int(target_timesteps)}."
        )

    mlflow_config = config.get("mlflow", {}) if isinstance(config, Mapping) else {}
    mlflow_enabled = bool(kwargs.get("mlflow_enabled", mlflow_config.get("enabled", False)))
    tracking_uri = kwargs.get("tracking_uri") or os.environ.get("ASSAULT_MLFLOW_TRACKING_URI") or mlflow_config.get("tracking_uri")
    experiment_name = kwargs.get("mlflow_experiment_name") or mlflow_config.get("experiment_name") or "assault_ddqn"

    if mlflow_enabled:
        mlflow_run_id, latest_session_id = _discover_mlflow_identity(
            tracking_uri=str(tracking_uri) if tracking_uri else None,
            experiment_name=str(experiment_name),
            project_run_id=project_run_id,
        )
    else:
        mlflow_run_id = None
        if previous_manifest_text:
            latest_session_id = str(json.loads(previous_manifest_text).get("latest_tracking_session_id") or "session_001")
        else:
            latest_session_id = "session_001"

    payload = {
        "schema_version": 1,
        "project_run_id": project_run_id,
        "mlflow_run_id": mlflow_run_id,
        "latest_tracking_session_id": latest_session_id,
        "latest_checkpoint": str(checkpoint_path),
        "latest_global_step": int(checkpoint_step),
        "resume_mode": str(kwargs.get("resume_mode") or "resume_full"),
        "bootstrap_commit": str(kwargs.get("bootstrap_commit") or "recovered-periodic-checkpoint"),
        "config_fingerprint": compute_config_fingerprint(config),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "recovered_from_periodic_checkpoint": True,
    }
    _atomic_write_manifest(manifest_path, payload)
    return previous_manifest_text if previous_manifest_text is not None else ""


def _rollback_recovery_manifest(path: Path, previous_manifest_text: str) -> None:
    if previous_manifest_text == "":
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(previous_manifest_text, encoding="utf-8")


def _recovery_manifest_path(prepare_training_session_kwargs: Mapping[str, Any], project_run_id: str) -> Path:
    base_path = Path(prepare_training_session_kwargs.get("base_path") or Path.cwd())
    return base_path / "experiments" / project_run_id / "experiment_state.json"


def _resolve_checkpoint_root(base_path: Path, config: Any, explicit_root: Any) -> Path:
    configured = config.get("checkpointing", {}) if isinstance(config, Mapping) else {}
    return Path(
        explicit_root
        or os.environ.get("ASSAULT_CHECKPOINT_DIR")
        or base_path / str(configured.get("directory", "checkpoints"))
    )


def _latest_periodic_checkpoint(run_dir: Path) -> Optional[tuple[Path, int]]:
    if not run_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint_step_*.pt"):
        match = _CHECKPOINT_PATTERN.match(path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    step, path = max(candidates, key=lambda item: item[0])
    return path, step


def _discover_mlflow_identity(
    tracking_uri: Optional[str],
    experiment_name: str,
    project_run_id: str,
) -> tuple[str, str]:
    """Finds the unique MLflow run created before an interrupted training."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Interrupted checkpoint recovery requires MLflow to recover run identity.") from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Periodic checkpoint exists but MLflow experiment '{experiment_name}' was not found; refusing ambiguous recovery."
        )

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000,
        order_by=["attributes.start_time DESC"],
    )
    matches = [
        run
        for run in runs
        if (run.data.params.get("identity.project_run_id") or run.data.tags.get("project_run_id")) == project_run_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Periodic checkpoint recovery requires exactly one MLflow run for "
            f"project_run_id='{project_run_id}', found {len(matches)}."
        )

    run = matches[0]
    latest_session_id = str(run.data.tags.get("latest_tracking_session_id") or "session_001")
    if not re.fullmatch(r"session_\d{3}", latest_session_id):
        raise RuntimeError(f"Invalid MLflow latest_tracking_session_id during recovery: {latest_session_id}")
    return str(run.info.run_id), latest_session_id


def _atomic_write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _normalize_execution_mode(execution_mode: str, run_training: bool | None) -> str:
    if run_training is not None:
        return "train" if bool(run_training) else "delivery"
    selected_mode = str(execution_mode or "auto").strip().lower()
    if selected_mode not in VALID_EXECUTION_MODES:
        raise ValueError("execution_mode must be one of: auto, train, delivery.")
    return selected_mode


def _checkpoint_exists(path: Optional[Path]) -> bool:
    return path is not None and path.exists() and path.is_file() and path.stat().st_size > 0
