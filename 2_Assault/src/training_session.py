from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .agent import DDQNAgent
from .callbacks import TensorBoardLogger
from .checkpointing import CheckpointManager, CheckpointMetadata, CheckpointState
from .environment import create_assault_env
from .replay_buffer import ReplayBuffer
from .trainer import Trainer, TrainingSummary


@dataclass(frozen=True)
class TrainingSessionSummary:
    """Summary for one HU008 externally resumable training session."""

    tracking_mode: str
    run_id: str
    initial_global_step: int
    final_global_step: int
    checkpoint: CheckpointMetadata
    training: TrainingSummary
    checkpoint_input_reference: Optional[str] = None
    checkpoint_input_loaded: bool = False
    restored_checkpoint_path: Optional[str] = None
    restored_global_step: Optional[int] = None
    replay_buffer_restored: bool = False
    resume_mode: Optional[str] = None
    device: str = "cpu"
    agent: Optional[DDQNAgent] = field(default=None, repr=False, compare=False)
    replay_buffer: Optional[ReplayBuffer] = field(default=None, repr=False, compare=False)
    restored: Optional[CheckpointState] = field(default=None, repr=False)

    @property
    def checkpoint_output_reference(self) -> str:
        return str(self.checkpoint.path)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tracking_mode": self.tracking_mode,
            "run_id": self.run_id,
            "initial_global_step": self.initial_global_step,
            "final_global_step": self.final_global_step,
            "checkpoint_input_reference": self.checkpoint_input_reference,
            "checkpoint_input_loaded": self.checkpoint_input_loaded,
            "restored_checkpoint_path": self.restored_checkpoint_path,
            "restored_global_step": self.restored_global_step,
            "replay_buffer_restored": self.replay_buffer_restored,
            "resume_mode": self.resume_mode,
            "checkpoint_output_reference": self.checkpoint_output_reference,
            "device": self.device,
            "checkpoint": self.checkpoint.as_dict(),
            "training": self.training.as_dict(),
            "restored": self.restored.as_dict() if self.restored else None,
        }


def run_training_session(
    config: Mapping[str, Any],
    checkpoint_root: str | Path,
    run_id: str,
    repo_path: str | Path = ".",
    tracking_mode: str = "new",
    checkpoint_input: str | Path | None = None,
    resume_mode: str = "resume_full",
    tensorboard_root: str | Path | None = None,
    total_timesteps: int | None = None,
    device: str | torch.device | None = None,
    seed_offset: int = 0,
) -> TrainingSessionSummary:
    """Runs one HU008 training session and optionally restores a prior checkpoint.

    This function intentionally delegates checkpoint loading to
    ``CheckpointManager.load`` so that notebook and test resume paths use the
    same restoration mechanism as HU005/HU007.
    """
    selected_mode = str(tracking_mode).strip().lower()
    if selected_mode not in {"new", "resume"}:
        raise ValueError("tracking_mode must be 'new' or 'resume'.")
    if selected_mode == "resume" and checkpoint_input is None:
        raise ValueError("checkpoint_input is required when tracking_mode='resume'.")
    if selected_mode == "new" and checkpoint_input is not None:
        raise ValueError("checkpoint_input must be empty when tracking_mode='new'.")

    session_config = copy.deepcopy(dict(config))
    if total_timesteps is not None:
        session_config.setdefault("training", {})["total_timesteps"] = int(total_timesteps)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(session_config.get("reproducibility", {}).get("seed", 0)) + int(seed_offset)
    replay_config = session_config["replay_buffer"]
    checkpoint_config = session_config.get("checkpointing", {})

    manager = CheckpointManager(checkpoint_root, run_id, repo_path=repo_path)
    agent = DDQNAgent(session_config, device=selected_device, seed=seed)
    replay_buffer = ReplayBuffer(capacity=int(replay_config["capacity"]), seed=seed)

    restored_state: Optional[CheckpointState] = None
    initial_global_step = 0
    if selected_mode == "new":
        manager.ensure_new_run()
    else:
        restored_state = manager.load(
            checkpoint_input,
            agent,
            replay_buffer,
            expected_config=session_config,
            mode=resume_mode,
            map_location=selected_device,
        )
        initial_global_step = int(restored_state.global_step)
        if initial_global_step <= 0:
            raise RuntimeError("tracking_mode='resume' restored a checkpoint with global_step <= 0.")
        if resume_mode == "resume_full" and not restored_state.replay_buffer_restored:
            raise RuntimeError("resume_full requires a checkpoint with a restored replay buffer.")
        target_timesteps = int(session_config["training"]["total_timesteps"])
        if target_timesteps <= initial_global_step:
            raise ValueError("training.total_timesteps must be greater than the restored global_step.")

    logger = None
    env = create_assault_env(session_config, mode="train", seed=seed)
    try:
        if tensorboard_root is not None and bool(session_config.get("tensorboard", {}).get("enabled", False)):
            logger = TensorBoardLogger.from_config(session_config, run_id=run_id, log_root=tensorboard_root)
        trainer = Trainer(
            env,
            agent,
            replay_buffer,
            session_config,
            initial_global_step=initial_global_step,
            initial_metrics=restored_state.training_metrics if restored_state else None,
            checkpoint_manager=manager,
            checkpoint_interval_steps=int(checkpoint_config.get("interval_steps", 0) or 0),
            checkpoint_save_replay_buffer=bool(checkpoint_config.get("save_replay_buffer", True)),
            checkpoint_keep_last=checkpoint_config.get("keep_last"),
            metrics_logger=logger,
        )
        training_summary = trainer.train()
        if selected_mode == "resume" and restored_state and training_summary.initial_global_step != restored_state.global_step:
            raise RuntimeError("Restored global_step does not match the resumed trainer initial_global_step.")
        if selected_mode == "resume" and training_summary.global_step <= initial_global_step:
            raise RuntimeError("Resume session did not continue beyond the restored global_step.")
        final_checkpoint_path = manager.checkpoint_path(training_summary.global_step)
        if str(final_checkpoint_path) in set(training_summary.checkpoints_saved):
            checkpoint_metadata = CheckpointMetadata(
                path=final_checkpoint_path,
                run_id=run_id,
                checkpoint_step=int(training_summary.global_step),
                size_bytes=final_checkpoint_path.stat().st_size,
                save_replay_buffer=bool(checkpoint_config.get("save_replay_buffer", True)),
            )
        else:
            checkpoint_metadata = manager.save(
                agent,
                replay_buffer,
                session_config,
                training_summary.global_step,
                training_summary,
                save_replay_buffer=bool(checkpoint_config.get("save_replay_buffer", True)),
                keep_last=checkpoint_config.get("keep_last"),
            )
        if logger:
            logger.flush()
    finally:
        if logger:
            logger.close()
        env.close()

    return TrainingSessionSummary(
        tracking_mode=selected_mode,
        run_id=run_id,
        initial_global_step=int(training_summary.initial_global_step),
        final_global_step=int(training_summary.global_step),
        checkpoint=checkpoint_metadata,
        training=training_summary,
        checkpoint_input_reference=str(checkpoint_input) if checkpoint_input is not None else None,
        checkpoint_input_loaded=restored_state is not None,
        restored_checkpoint_path=str(restored_state.path) if restored_state else None,
        restored_global_step=int(restored_state.global_step) if restored_state else None,
        replay_buffer_restored=bool(restored_state.replay_buffer_restored) if restored_state else False,
        resume_mode=resume_mode if selected_mode == "resume" else None,
        device=str(selected_device),
        agent=agent,
        replay_buffer=replay_buffer,
        restored=restored_state,
    )
