"""HU009C delivery orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


VALID_EXECUTION_MODES = {"auto", "train", "delivery"}


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

    Args:
        run_training: Deprecated compatibility flag. When provided, true maps
            to ``execution_mode="train"`` and false maps to
            ``execution_mode="delivery"``.
        project_run_id: Existing logical run id used for delivery lineage.
        target_timesteps: Target global timestep for the selected profile.
        prepare_training_session_fn: Existing HU008B bootstrap function. It is
            called only when training is required.
        prepare_training_session_kwargs: Keyword arguments for the bootstrap.
        execution_mode: ``auto`` (default), ``train`` or ``delivery``.
        final_checkpoint_path: Expected final full checkpoint path.

    Returns:
        Resolved execution mode and optional training session context.

    Raises:
        ValueError: If identifiers are invalid.
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

    session_context = prepare_training_session_fn(**dict(prepare_training_session_kwargs))
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


def _normalize_execution_mode(execution_mode: str, run_training: bool | None) -> str:
    if run_training is not None:
        return "train" if bool(run_training) else "delivery"
    selected_mode = str(execution_mode or "auto").strip().lower()
    if selected_mode not in VALID_EXECUTION_MODES:
        raise ValueError("execution_mode must be one of: auto, train, delivery.")
    return selected_mode


def _checkpoint_exists(path: Optional[Path]) -> bool:
    return path is not None and path.exists() and path.is_file() and path.stat().st_size > 0
