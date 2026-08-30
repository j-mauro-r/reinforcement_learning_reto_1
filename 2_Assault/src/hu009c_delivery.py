"""HU009C delivery orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


@dataclass(frozen=True)
class DeliveryExecutionMode:
    """Resolved execution mode for HU009C notebook orchestration."""

    run_training: bool
    hu009c_post_training_ready: bool
    training_session_bootstrap_skipped: bool
    project_run_id: str
    target_timesteps: int
    session_context: Optional[Any] = None

    def as_dict(self) -> dict[str, Any]:
        """Returns a notebook-friendly representation."""
        return {
            "ASSAULT_RUN_TRAINING": int(self.run_training),
            "HU009C_POST_TRAINING_READY": self.hu009c_post_training_ready,
            "training_session_bootstrap_skipped": self.training_session_bootstrap_skipped,
            "project_run_id": self.project_run_id,
            "target_timesteps": self.target_timesteps,
            "has_training_session_context": self.session_context is not None,
        }


def resolve_hu009c_execution_mode(
    run_training: bool,
    project_run_id: str,
    target_timesteps: int,
    prepare_training_session_fn: Callable[..., Any],
    prepare_training_session_kwargs: Mapping[str, Any],
) -> DeliveryExecutionMode:
    """Separates HU009 training bootstrap from HU009C post-training delivery.

    Args:
        run_training: Whether the notebook should execute a training session.
        project_run_id: Existing logical run id used for delivery lineage.
        target_timesteps: Target global timestep for the selected profile.
        prepare_training_session_fn: Existing HU008B bootstrap function. It is
            called only when ``run_training`` is true.
        prepare_training_session_kwargs: Keyword arguments for the bootstrap.

    Returns:
        Resolved execution mode and optional training session context.

    Raises:
        ValueError: If identifiers are invalid.
    """
    if not str(project_run_id).strip():
        raise ValueError("project_run_id must be explicit and non-empty.")
    if int(target_timesteps) <= 0:
        raise ValueError("target_timesteps must be positive.")

    if not bool(run_training):
        return DeliveryExecutionMode(
            run_training=False,
            hu009c_post_training_ready=True,
            training_session_bootstrap_skipped=True,
            project_run_id=str(project_run_id),
            target_timesteps=int(target_timesteps),
            session_context=None,
        )

    session_context = prepare_training_session_fn(**dict(prepare_training_session_kwargs))
    return DeliveryExecutionMode(
        run_training=True,
        hu009c_post_training_ready=False,
        training_session_bootstrap_skipped=False,
        project_run_id=str(project_run_id),
        target_timesteps=int(target_timesteps),
        session_context=session_context,
    )
