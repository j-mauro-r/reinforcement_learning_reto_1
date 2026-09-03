"""HU011 long-training configuration, preflight and orchestration helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional

import numpy as np
import torch

from src.experiment import LongTrainingReadiness, validate_long_training_readiness
from src.experiment import (
    capture_git_lineage,
    capture_hardware,
    capture_runtime,
    create_run_manifest,
    fail_session,
    finish_session,
    generate_run_id,
    load_config_snapshot,
    load_run_manifest,
    start_session,
    validate_resume_compatibility,
)
from src.agent import DQNAgent
from src.callbacks import TensorBoardTrainingLogger
from src.environment import create_battlezone_env
from src.persistence import (
    CHECKPOINT_MODE_FULL,
    CHECKPOINT_MODE_LIGHTWEIGHT,
    build_checkpoint_metadata,
    build_checkpoint_payload,
    checkpoint_config_snapshot,
    checkpoint_filename,
    restore_training_state,
    save_checkpoint,
)
from src.trainer import DQNTrainer, TrainingMode, TrainingState


REFERENCE_PROFILE = "reference_v1"


class CheckpointDecision(str, Enum):
    """Checkpoint action at a global step."""

    NONE = "none"
    LIGHTWEIGHT = "lightweight"
    FULL = "full"


@dataclass(frozen=True)
class ReplayMemoryEstimate:
    """Estimated replay allocation and its share of physical RAM."""

    bytes_per_state: int
    bytes_per_transition: int
    estimated_replay_bytes: int
    estimated_replay_gib: float
    available_ram_gib: Optional[float]
    fraction_of_ram: Optional[float]


@dataclass(frozen=True)
class MemoryReadiness:
    """Memory gate for replay and temporary FULL-checkpoint copies."""

    ready: bool
    replay_gib: float
    ram_gib: Optional[float]
    fraction: Optional[float]
    full_checkpoint_ready: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class HU011Preflight:
    """Structured preflight result produced before creating a long run."""

    ready: bool
    profile: str
    target_global_step: int
    cuda_available: bool
    gpu_name: Optional[str]
    persistent_root: str
    memory: MemoryReadiness
    tracking: LongTrainingReadiness
    errors: tuple[str, ...]


def resolve_long_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolves `reference_v1` into a detached effective configuration.

    Args:
        config: Complete versioned BattleZone configuration.

    Returns:
        Deep copy with long-training overrides applied to existing contracts.

    Raises:
        ValueError: If the requested profile is absent, disabled or unsupported.
    """
    long = config.get("long_training")
    if not isinstance(long, Mapping) or long.get("enabled") is not True:
        raise ValueError("long_training must be configured and enabled.")
    if long.get("profile") != REFERENCE_PROFILE:
        raise ValueError(f"Unsupported long-training profile: {long.get('profile')!r}.")
    effective = deepcopy(dict(config))
    effective["dqn"]["batch_size"] = int(long["dqn"]["batch_size"])
    effective["dqn"]["replay_buffer"]["capacity"] = int(
        long["dqn"]["replay_buffer_capacity"]
    )
    for key in ("learning_starts", "train_frequency", "target_sync_interval"):
        effective["training"][key] = int(long["training"][key])
    effective["training"]["epsilon"] = deepcopy(dict(long["training"]["epsilon"]))
    effective["checkpointing"]["interval_steps"] = int(
        long["checkpointing"]["interval_steps"]
    )
    effective["checkpointing"]["default_mode"] = str(
        long["checkpointing"]["periodic_mode"]
    )
    effective["tensorboard"]["scalar_log_interval_steps"] = int(
        long["tensorboard"]["scalar_log_interval_steps"]
    )
    effective["tensorboard"]["flush_interval_steps"] = int(
        long["tensorboard"]["flush_interval_steps"]
    )
    effective["dqn"]["device"] = str(long["preferred_device"])
    effective["training"]["total_timesteps"] = int(long["target_global_step"])
    return effective


def estimate_replay_memory(
    *, state_shape: tuple[int, ...], capacity: int,
    available_ram_gib: Optional[float] = None,
) -> ReplayMemoryEstimate:
    """Estimates current ReplayBuffer allocation from its concrete dtypes."""
    if capacity <= 0 or not state_shape or any(int(x) <= 0 for x in state_shape):
        raise ValueError("capacity and state_shape dimensions must be positive.")
    state_bytes = math.prod(int(x) for x in state_shape) * np.dtype(np.uint8).itemsize
    transition_bytes = (
        2 * state_bytes + np.dtype(np.int64).itemsize
        + np.dtype(np.float32).itemsize + np.dtype(np.bool_).itemsize
    )
    total = transition_bytes * int(capacity)
    replay_gib = total / (1024**3)
    fraction = None if available_ram_gib is None else replay_gib / available_ram_gib
    return ReplayMemoryEstimate(
        state_bytes, transition_bytes, total, replay_gib,
        available_ram_gib, fraction,
    )


def validate_memory_readiness(
    estimate: ReplayMemoryEstimate, *, max_replay_fraction: float = 0.50,
    max_full_copy_fraction: float = 0.80,
) -> MemoryReadiness:
    """Evaluates replay allocation and the extra FULL-checkpoint copy risk."""
    errors: list[str] = []
    if estimate.available_ram_gib is None:
        errors.append("RAM unavailable")
    elif estimate.fraction_of_ram is not None and estimate.fraction_of_ram > max_replay_fraction:
        errors.append("Replay estimate exceeds safe RAM fraction")
    full_ready = bool(
        estimate.fraction_of_ram is not None
        and estimate.fraction_of_ram * 2 <= max_full_copy_fraction
    )
    return MemoryReadiness(
        not errors, estimate.estimated_replay_gib, estimate.available_ram_gib,
        estimate.fraction_of_ram, full_ready, tuple(errors),
    )


def checkpoint_decision(
    global_step: int, config: Mapping[str, Any], *, full_checkpoint_ready: bool = True,
) -> CheckpointDecision:
    """Returns the explicit reference-v1 checkpoint action for a global step."""
    if global_step <= 0:
        return CheckpointDecision.NONE
    policy = config["long_training"]["checkpointing"]
    full_interval = int(policy["full_milestone_interval_steps"])
    periodic_interval = int(policy["interval_steps"])
    if global_step % full_interval == 0 and full_checkpoint_ready:
        return CheckpointDecision.FULL
    if global_step % periodic_interval == 0:
        return CheckpointDecision.LIGHTWEIGHT
    return CheckpointDecision.NONE


def build_artifact_paths(persistent_root: str | Path, run_id: str) -> dict[str, Path]:
    """Builds isolated persistent artifact paths without discovering latest runs."""
    if not run_id:
        raise ValueError("An explicit run_id is required.")
    root = Path(persistent_root)
    return {
        "root": root / run_id,
        "results": root / "results" / run_id,
        "checkpoints": root / "checkpoints" / run_id,
        "logs": root / "logs" / run_id,
        "models": root / "models" / run_id,
        "manifest": root / "results" / run_id / "run_manifest.json",
        "final_model": root / "models" / run_id / "battlezone_dqn_final.pt",
    }


def run_hu011_preflight(
    *, base_config: Mapping[str, Any], config_path: str | Path, run_id: str,
    git: Mapping[str, Any], persistent_root: str | Path,
    ram_gib: Optional[float], cuda_available: Optional[bool] = None,
    gpu_name: Optional[str] = None,
) -> HU011Preflight:
    """Runs CUDA, memory, persistence and HU010 gates without training."""
    effective = resolve_long_training_config(base_config)
    long = base_config["long_training"]
    cuda = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    if gpu_name is None and cuda:
        gpu_name = torch.cuda.get_device_name(0)
    estimate = estimate_replay_memory(
        state_shape=tuple(effective["validation"]["expected_final_shape"]),
        capacity=int(effective["dqn"]["replay_buffer"]["capacity"]),
        available_ram_gib=ram_gib,
    )
    memory = validate_memory_readiness(estimate)
    root = Path(persistent_root)
    paths = build_artifact_paths(root, run_id)
    tracking = validate_long_training_readiness(
        config=effective, config_path=config_path, run_id=run_id, git=git,
        # create_run_manifest appends run_id to this canonical results root.
        results_dir=paths["results"].parent,
    )
    errors = list(memory.errors) + list(tracking.errors)
    if long.get("require_accelerator") is True and not cuda:
        errors.append("LONG_TRAINING_BLOCKED_NO_CUDA")
    return HU011Preflight(
        not errors, str(long["profile"]), int(long["target_global_step"]),
        cuda, gpu_name, str(root), memory, tracking, tuple(errors),
    )


def is_training_complete(final_global_step: int, target_global_step: int) -> bool:
    """Returns true only after reaching the global training target."""
    return int(final_global_step) >= int(target_global_step)


def _save_training_checkpoint(
    *, path: Path, mode: CheckpointDecision, agent: DQNAgent,
    trainer: DQNTrainer, config: Mapping[str, Any], seed: int,
) -> Path:
    checkpoint_mode = (
        CHECKPOINT_MODE_FULL if mode is CheckpointDecision.FULL
        else CHECKPOINT_MODE_LIGHTWEIGHT
    )
    state = trainer.export_training_state()
    metadata = build_checkpoint_metadata(
        checkpoint_mode=checkpoint_mode, algorithm="DQN",
        global_step=int(state["global_step"]), episode_index=int(state["episode_index"]),
        seed=seed, state_shape=agent.state_shape, action_dim=agent.action_dim,
        batch_size=agent.batch_size,
    )
    payload = build_checkpoint_payload(
        metadata=metadata, trainer_state=state, agent_state=agent.state_dict(),
        replay_buffer_state=(agent.replay_buffer.state_dict() if mode is CheckpointDecision.FULL else None),
        config_snapshot=checkpoint_config_snapshot(config),
    )
    return save_checkpoint(checkpoint_path=path, payload=payload)


def run_training_session(
    *, base_config: Mapping[str, Any], config_path: str | Path,
    persistent_root: str | Path, mode: str, repo_root: str | Path,
    run_id: Optional[str] = None, checkpoint_path: Optional[str | Path] = None,
    target_global_step_override: Optional[int] = None,
    require_accelerator_override: Optional[bool] = None,
    progress_interval_steps: int = 5000,
) -> dict[str, Any]:
    """Runs one explicit NEW/FULL/LIGHTWEIGHT HU011 training session.

    Local callers must use a small session-stop override and explicitly disable
    the accelerator gate. The logical reference target remains one million, so
    a preflight session is recorded as interrupted rather than completed.
    """
    if mode not in {item.value for item in TrainingMode}:
        raise ValueError(f"Unsupported mode={mode!r}.")
    if progress_interval_steps <= 0:
        raise ValueError("progress_interval_steps must be positive.")
    if mode != TrainingMode.NEW.value and (not run_id or not checkpoint_path):
        raise ValueError("Resume requires explicit run_id and checkpoint_path.")
    effective = resolve_long_training_config(base_config)
    logical_target = int(base_config["long_training"]["target_global_step"])
    session_target = int(
        target_global_step_override
        if target_global_step_override is not None
        else logical_target
    )
    require_accelerator = (
        bool(base_config["long_training"]["require_accelerator"])
        if require_accelerator_override is None else bool(require_accelerator_override)
    )
    if not require_accelerator:
        effective["dqn"]["device"] = "auto"
    if target_global_step_override is not None and session_target > 4096:
        raise ValueError("Local target override is limited to 4096 steps.")
    git = capture_git_lineage(repo_root)
    hardware = capture_hardware()
    resolved_run_id = run_id or generate_run_id(git_sha=str(git["commit"]))
    preflight_config = deepcopy(dict(base_config))
    preflight_config["long_training"]["require_accelerator"] = require_accelerator
    preflight = run_hu011_preflight(
        base_config=preflight_config, config_path=config_path,
        run_id=resolved_run_id, git=git, persistent_root=persistent_root,
        ram_gib=hardware["ram_gb"],
    )
    if not preflight.ready:
        raise RuntimeError("; ".join(preflight.errors))
    paths = build_artifact_paths(persistent_root, resolved_run_id)
    for key in ("checkpoints", "logs", "models"):
        paths[key].mkdir(parents=True, exist_ok=True)
    snapshot, config_sha = load_config_snapshot(config_path)
    # Manifest/checkpoint compatibility must use the effective reference profile.
    snapshot = effective
    if mode == TrainingMode.NEW.value:
        manifest, manifest_path = create_run_manifest(
            results_dir=Path(persistent_root) / "results",
            manifest_filename="run_manifest.json", run_id=resolved_run_id,
            config_path=config_path, config_snapshot=snapshot,
            config_sha256=config_sha, git=git, runtime=capture_runtime(), hardware=hardware,
        )
        initial_state = TrainingState()
        replay_restored = None
    else:
        manifest_path = paths["manifest"]
        manifest = load_run_manifest(manifest_path)
        validate_resume_compatibility(manifest, snapshot)
        initial_state = None
        replay_restored = mode == TrainingMode.RESUME_FULL.value
    agent = DQNAgent.from_config(effective)
    if mode != TrainingMode.NEW.value:
        restored = restore_training_state(
            checkpoint_path=checkpoint_path, agent=agent, config=effective,
            expected_mode=(CHECKPOINT_MODE_FULL if replay_restored else CHECKPOINT_MODE_LIGHTWEIGHT),
            map_location=agent.device,
        )
        initial_state = TrainingState(
            global_step=restored["global_step"], episode_index=restored["episode_index"],
            episode_step=restored["episode_step"], episode_reward=restored["episode_reward"],
        )
    assert initial_state is not None
    env = create_battlezone_env(effective, mode="train", seed=int(effective["environment"]["seed"]))
    manifest = start_session(
        manifest, mode=mode, start_global_step=initial_state.global_step,
        input_checkpoint=(str(checkpoint_path) if checkpoint_path else None),
        tensorboard_log_dir=str(paths["logs"]), device=str(agent.device),
        replay_restored=replay_restored, manifest_path=manifest_path,
    )
    trainer: Optional[DQNTrainer] = None
    output_checkpoint: Optional[Path] = None
    checkpoint_paths: list[Path] = []
    started_at = time.monotonic()
    try:
        tb = effective["tensorboard"]
        logger = TensorBoardTrainingLogger(
            log_dir=paths["logs"], reward_window=int(tb["reward_window"]),
            scalar_log_interval_steps=int(tb["scalar_log_interval_steps"]),
            flush_interval_steps=int(tb["flush_interval_steps"]),
        )
        trainer = DQNTrainer.from_config(
            config=effective, agent=agent, env=env, total_timesteps=session_target,
            logger=logger,
        )

        def persist_at_boundary(global_step: int, active_trainer: DQNTrainer) -> None:
            nonlocal output_checkpoint
            if global_step % progress_interval_steps == 0:
                elapsed = time.monotonic() - started_at
                percentage = 100.0 * global_step / logical_target
                epsilon = active_trainer.epsilon_schedule.value(global_step)
                print(
                    "TRAINING_PROGRESS "
                    f"global_step={global_step} target={logical_target} percentage={percentage:.2f} "
                    f"episodes={active_trainer.training_state.episode_index} epsilon={epsilon:.6f} "
                    f"replay_size={len(agent.replay_buffer)} updates={active_trainer.updates_completed} "
                    f"target_syncs={active_trainer.target_syncs_completed} "
                    f"latest_loss={active_trainer.latest_loss} elapsed_seconds={elapsed:.1f}"
                )
            decision = checkpoint_decision(
                global_step, base_config,
                full_checkpoint_ready=preflight.memory.full_checkpoint_ready,
            )
            if decision is not CheckpointDecision.NONE:
                cp_mode = decision.value
                output_checkpoint = paths["checkpoints"] / checkpoint_filename(
                    global_step=global_step, checkpoint_mode=cp_mode,
                )
                _save_training_checkpoint(
                    path=output_checkpoint, mode=decision, agent=agent,
                    trainer=active_trainer, config=effective,
                    seed=int(effective["environment"]["seed"]),
                )
                checkpoint_paths.append(output_checkpoint)
                if not output_checkpoint.exists():
                    raise RuntimeError(f"Checkpoint was not persisted: {output_checkpoint}")
                print(
                    "CHECKPOINT_SAVED "
                    f"path={output_checkpoint} mode={decision.value.upper()} global_step={global_step}"
                )

        summary = trainer.train(
            total_timesteps=session_target, initial_state=initial_state,
            mode=mode, replay_restored=bool(replay_restored),
            step_callback=persist_at_boundary,
        )
        initial_state = TrainingState(**trainer.export_training_state())
        if output_checkpoint is None or int(summary.total_steps) % int(
            base_config["long_training"]["checkpointing"]["interval_steps"]
        ) != 0:
            output_checkpoint = paths["checkpoints"] / checkpoint_filename(
                global_step=initial_state.global_step,
                checkpoint_mode=CHECKPOINT_MODE_LIGHTWEIGHT,
            )
            _save_training_checkpoint(
                path=output_checkpoint, mode=CheckpointDecision.LIGHTWEIGHT,
                agent=agent, trainer=trainer, config=effective,
                seed=int(effective["environment"]["seed"]),
            )
            checkpoint_paths.append(output_checkpoint)
        completed = is_training_complete(initial_state.global_step, logical_target)
        if completed:
            assert trainer is not None
            output_checkpoint = _save_training_checkpoint(
                path=paths["final_model"], mode=CheckpointDecision.LIGHTWEIGHT,
                agent=agent, trainer=trainer, config=effective,
                seed=int(effective["environment"]["seed"]),
            )
        manifest = finish_session(
            manifest, end_global_step=initial_state.global_step,
            episode_index=initial_state.episode_index,
            elapsed_seconds=time.monotonic() - started_at,
            output_checkpoint=str(output_checkpoint) if output_checkpoint else None,
            completed=completed, manifest_path=manifest_path,
        )
        if completed:
            manifest["artifacts"]["model_path"] = str(paths["final_model"])
            from src.experiment import write_run_manifest
            write_run_manifest(manifest_path, manifest)
        print(
            "TRAINING_ACTIVE=True "
            f"GLOBAL_STEP={summary.total_steps} UPDATES={summary.updates} "
            f"REPLAY_SIZE={summary.replay_size} EPSILON={summary.final_epsilon:.6f} "
            f"TARGET_SYNCS={summary.target_syncs} LATEST_LOSS={summary.last_loss}"
        )
        return {
            "run_id": resolved_run_id, "manifest": manifest,
            "preflight": preflight, "paths": paths, "summary": summary,
            "checkpoints": checkpoint_paths,
        }
    except KeyboardInterrupt:
        if trainer is not None:
            initial_state = TrainingState(**trainer.export_training_state())
        if trainer is not None and initial_state.global_step > manifest["sessions"][-1]["start_global_step"]:
            interrupt_checkpoint = paths["checkpoints"] / checkpoint_filename(
                global_step=initial_state.global_step, checkpoint_mode=CHECKPOINT_MODE_LIGHTWEIGHT,
            )
            if not interrupt_checkpoint.exists():
                _save_training_checkpoint(path=interrupt_checkpoint, mode=CheckpointDecision.LIGHTWEIGHT, agent=agent, trainer=trainer, config=effective, seed=int(effective["environment"]["seed"]))
            output_checkpoint = interrupt_checkpoint
            finish_session(manifest, end_global_step=initial_state.global_step, episode_index=initial_state.episode_index, elapsed_seconds=time.monotonic() - started_at, output_checkpoint=str(output_checkpoint), completed=False, manifest_path=manifest_path)
        raise
    except Exception as exc:
        fail_session(manifest, error=exc, manifest_path=manifest_path)
        raise
    finally:
        env.close()
