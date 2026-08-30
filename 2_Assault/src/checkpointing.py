"""Checkpoint persistence for resumable Assault DDQN training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from .agent import DDQNAgent
from .replay_buffer import ReplayBuffer
from .trainer import TrainingSummary, compute_epsilon
from .utils import get_git_commit


SCHEMA_VERSION = 1
VALID_RUN_MODES = {"new", "resume_full", "resume_light"}


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata returned after saving a checkpoint."""

    path: Path
    run_id: str
    checkpoint_step: int
    size_bytes: int
    save_replay_buffer: bool

    def as_dict(self) -> Dict[str, Any]:
        """Returns a serializable representation."""
        return {
            "path": str(self.path),
            "run_id": self.run_id,
            "checkpoint_step": self.checkpoint_step,
            "size_bytes": self.size_bytes,
            "save_replay_buffer": self.save_replay_buffer,
        }


@dataclass(frozen=True)
class CheckpointState:
    """Training state restored from a checkpoint."""

    path: Path
    run_id: str
    global_step: int
    config: Dict[str, Any]
    training_metrics: Dict[str, Any]
    epsilon: float
    replay_buffer_restored: bool
    checkpoint_size_bytes: int
    raw: Dict[str, Any] = field(repr=False)

    def as_dict(self) -> Dict[str, Any]:
        """Returns a notebook-friendly dictionary."""
        return {
            "path": str(self.path),
            "run_id": self.run_id,
            "global_step": self.global_step,
            "epsilon": self.epsilon,
            "replay_buffer_restored": self.replay_buffer_restored,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "training_metrics": self.training_metrics,
        }


class CheckpointManager:
    """Saves and restores versioned DDQN training checkpoints."""

    def __init__(self, directory: str | Path, run_id: str, repo_path: str | Path = ".") -> None:
        """Initializes the checkpoint manager.

        Args:
            directory: Root directory that contains run subdirectories.
            run_id: Explicit run identifier.
            repo_path: Git repository path used to record the current commit.

        Raises:
            ValueError: If ``run_id`` is empty.
        """
        if not str(run_id).strip():
            raise ValueError("run_id must be explicit and non-empty.")
        self.directory = Path(directory)
        self.run_id = str(run_id)
        self.repo_path = Path(repo_path)

    @property
    def run_dir(self) -> Path:
        """Returns the directory for this run id."""
        return self.directory / self.run_id

    def checkpoint_path(self, step: int) -> Path:
        """Builds the deterministic checkpoint path for a global step."""
        if int(step) < 0:
            raise ValueError("checkpoint step must be non-negative.")
        return self.run_dir / f"checkpoint_step_{int(step):06d}.pt"

    def ensure_new_run(self) -> None:
        """Validates that a new run will not reuse existing checkpoints.

        Raises:
            FileExistsError: If the run directory already contains checkpoints.
        """
        if self.run_dir.exists() and any(self.run_dir.glob("checkpoint_step_*.pt")):
            raise FileExistsError(f"Run already has checkpoints: {self.run_dir}")

    def save(
        self,
        agent: DDQNAgent,
        replay_buffer: ReplayBuffer,
        config: Mapping[str, Any],
        global_step: int,
        training_metrics: Mapping[str, Any] | TrainingSummary,
        save_replay_buffer: bool = True,
        overwrite: bool = False,
        keep_last: Optional[int] = None,
    ) -> CheckpointMetadata:
        """Atomically saves a DDQN training checkpoint.

        Args:
            agent: Agent whose networks and optimizer are saved.
            replay_buffer: Replay buffer to optionally persist.
            config: Training configuration associated with the checkpoint.
            global_step: Global timestep to store in the checkpoint.
            training_metrics: Serializable metrics or ``TrainingSummary``.
            save_replay_buffer: Whether to include replay buffer state.
            overwrite: Explicit opt-in to overwrite an existing checkpoint.
            keep_last: Optional retention count for this run directory.

        Returns:
            Metadata about the saved checkpoint.

        Raises:
            FileExistsError: If the target exists and ``overwrite`` is false.
        """
        step = int(global_step)
        path = self.checkpoint_path(step)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Checkpoint already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._build_payload(
            agent=agent,
            replay_buffer=replay_buffer,
            config=config,
            global_step=step,
            training_metrics=training_metrics,
            save_replay_buffer=save_replay_buffer,
        )
        tmp_path = path.with_name(f".{path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        metadata = CheckpointMetadata(
            path=path,
            run_id=self.run_id,
            checkpoint_step=step,
            size_bytes=path.stat().st_size,
            save_replay_buffer=save_replay_buffer,
        )
        if metadata.size_bytes <= 0:
            raise RuntimeError(f"Checkpoint save produced an empty file: {path}")
        if keep_last is not None:
            self.prune_old_checkpoints(keep_last=keep_last, protected_paths=[metadata.path])
        return metadata

    def prune_old_checkpoints(self, keep_last: int = 1, protected_paths: Optional[Iterable[str | Path]] = None) -> list[Path]:
        """Deletes older checkpoints in this run directory after a successful save.

        Only files matching ``checkpoint_step_*.pt`` under ``self.run_dir`` are
        eligible. Non-checkpoint files, temporary files and other run ids are
        ignored.
        """
        keep = int(keep_last)
        if keep <= 0:
            raise ValueError("keep_last must be positive.")
        if not self.run_dir.exists():
            return []

        protected = {Path(path).resolve() for path in protected_paths or []}
        checkpoints = sorted(_checkpoint_files(self.run_dir), key=_checkpoint_sort_key)
        retained = set(checkpoints[-keep:])
        deleted: list[Path] = []
        for path in checkpoints:
            if path in retained or path.resolve() in protected:
                continue
            path.unlink()
            deleted.append(path)
        return deleted

    def load(
        self,
        checkpoint_path: str | Path,
        agent: DDQNAgent,
        replay_buffer: ReplayBuffer,
        expected_config: Mapping[str, Any],
        mode: str,
        map_location: str | torch.device | None = None,
    ) -> CheckpointState:
        """Loads a checkpoint into new agent and replay buffer objects.

        Args:
            checkpoint_path: Explicit path selected by the caller.
            agent: Agent instance to restore.
            replay_buffer: Replay buffer instance to restore or leave empty.
            expected_config: Current configuration used for compatibility
                checks.
            mode: One of ``new``, ``resume_full`` or ``resume_light``.
            map_location: Optional PyTorch map location.

        Returns:
            Restored checkpoint state and continuity metadata.

        Raises:
            ValueError: If mode, schema, run id or config are incompatible.
            FileNotFoundError: If ``checkpoint_path`` does not exist.
        """
        if mode not in VALID_RUN_MODES - {"new"}:
            raise ValueError("load mode must be 'resume_full' or 'resume_light'.")
        if checkpoint_path is None:
            raise ValueError("resume requires an explicit checkpoint_path.")
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        payload = torch.load(path, map_location=map_location or agent.device, weights_only=False)
        self._validate_payload(payload, expected_config=expected_config, mode=mode)

        if payload["run_id"] != self.run_id:
            raise ValueError(f"Checkpoint run_id mismatch: {payload['run_id']} != {self.run_id}.")

        agent.online_network.load_state_dict(payload["online_network"])
        agent.target_network.load_state_dict(payload["target_network"])
        agent.optimizer.load_state_dict(payload["optimizer"])

        replay_restored = False
        if mode == "resume_full":
            replay_state = payload.get("replay_buffer_state")
            if replay_state is None:
                raise ValueError("resume_full requires replay_buffer_state.")
            replay_buffer.load_state_dict(replay_state)
            replay_restored = True

        global_step = int(payload["global_step"])
        epsilon = reconstruct_epsilon(global_step, payload["config"])
        return CheckpointState(
            path=path,
            run_id=str(payload["run_id"]),
            global_step=global_step,
            config=dict(payload["config"]),
            training_metrics=dict(payload["training_metrics"]),
            epsilon=epsilon,
            replay_buffer_restored=replay_restored,
            checkpoint_size_bytes=path.stat().st_size,
            raw=payload,
        )

    def _build_payload(
        self,
        agent: DDQNAgent,
        replay_buffer: ReplayBuffer,
        config: Mapping[str, Any],
        global_step: int,
        training_metrics: Mapping[str, Any] | TrainingSummary,
        save_replay_buffer: bool,
    ) -> Dict[str, Any]:
        metrics = training_metrics.as_dict() if isinstance(training_metrics, TrainingSummary) else dict(training_metrics)
        epsilon_state = {
            "epsilon_start": float(config["agent"]["epsilon_start"]),
            "epsilon_final": float(config["agent"]["epsilon_final"]),
            "epsilon_decay_steps": int(config["training"]["epsilon_decay_steps"]),
            "global_step": int(global_step),
            "epsilon": reconstruct_epsilon(global_step, config),
        }
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_step": int(global_step),
            "git_commit": get_git_commit(self.repo_path),
            "config": dict(config),
            "online_network": agent.online_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "global_step": int(global_step),
            "epsilon_state": epsilon_state,
            "training_metrics": metrics,
            "resume_mode_capabilities": {
                "resume_full": bool(save_replay_buffer),
                "resume_light": True,
            },
            "replay_buffer_state": replay_buffer.state_dict() if save_replay_buffer else None,
        }
        return payload

    def _validate_payload(self, payload: Mapping[str, Any], expected_config: Mapping[str, Any], mode: str) -> None:
        required = {
            "schema_version",
            "run_id",
            "created_at",
            "checkpoint_step",
            "git_commit",
            "config",
            "online_network",
            "target_network",
            "optimizer",
            "global_step",
            "epsilon_state",
            "training_metrics",
            "resume_mode_capabilities",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")
        if int(payload["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema_version: {payload['schema_version']}")
        checkpoint_config = payload["config"]
        for section, key in (
            ("environment", "id"),
            ("preprocessing", "resize_height"),
            ("preprocessing", "resize_width"),
            ("preprocessing", "frame_stack"),
            ("network", "input_channels"),
            ("network", "num_actions"),
        ):
            expected = expected_config[section][key]
            actual = checkpoint_config[section][key]
            if actual != expected:
                raise ValueError(f"Incompatible checkpoint config {section}.{key}: {actual} != {expected}.")
        if mode == "resume_full" and not payload["resume_mode_capabilities"].get("resume_full", False):
            raise ValueError("Checkpoint was not saved with resume_full capability.")


def reconstruct_epsilon(global_step: int, config: Mapping[str, Any]) -> float:
    """Reconstructs epsilon from global step and configuration."""
    return compute_epsilon(
        int(global_step),
        float(config["agent"]["epsilon_start"]),
        float(config["agent"]["epsilon_final"]),
        int(config["training"]["epsilon_decay_steps"]),
    )


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    step = _checkpoint_step(path)
    return -1 if step is None else step, path.name


def _checkpoint_files(run_dir: Path) -> list[Path]:
    return [path for path in run_dir.glob("checkpoint_step_*.pt") if path.is_file() and _checkpoint_step(path) is not None]


def _checkpoint_step(path: Path) -> Optional[int]:
    try:
        return int(path.stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return None
