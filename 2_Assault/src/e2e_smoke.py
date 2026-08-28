"""End-to-end smoke orchestration for Assault DDQN."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import psutil
import torch

from .agent import DDQNAgent
from .callbacks import TensorBoardLogger, load_tensorboard_scalars
from .checkpointing import CheckpointManager, CheckpointMetadata, CheckpointState, reconstruct_epsilon
from .environment import create_assault_env, get_environment_metadata
from .evaluator import EvaluationSummary, evaluate_agent
from .preflight import PreflightReport, run_preflight_checks
from .replay_buffer import ReplayBuffer
from .trainer import Trainer, TrainingSummary
from .utils import get_runtime_info


@dataclass(frozen=True)
class MemorySnapshot:
    """Small CPU/GPU memory snapshot for smoke diagnostics."""

    label: str
    process_rss_mb: float
    ram_available_gb: float
    cuda_allocated_mb: Optional[float]
    cuda_reserved_mb: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        """Returns a serializable representation."""
        return {
            "label": self.label,
            "process_rss_mb": self.process_rss_mb,
            "ram_available_gb": self.ram_available_gb,
            "cuda_allocated_mb": self.cuda_allocated_mb,
            "cuda_reserved_mb": self.cuda_reserved_mb,
        }


@dataclass(frozen=True)
class E2ESmokeSummary:
    """Structured result for the HU007 smoke flow."""

    local_e2e_smoke_pass: bool
    e2e_smoke_pass: bool
    runtime: str
    device: str
    require_cuda: bool
    run_id: str
    observation_shape: tuple[int, ...]
    observation_dtype: str
    action_space: str
    preflight: PreflightReport
    segment_a: TrainingSummary
    checkpoint: CheckpointMetadata
    restored: CheckpointState
    segment_b: TrainingSummary
    tensorboard_event_files_before: int
    tensorboard_event_files_after: int
    tensorboard_tags: list[str]
    tensorboard_steps_before: Dict[str, list[int]]
    tensorboard_steps_after: Dict[str, list[int]]
    tensorboard_post_resume_steps: Dict[str, list[int]]
    tensorboard_previous_logs_preserved: bool
    evaluation: EvaluationSummary
    online_unchanged_during_evaluation: bool
    target_unchanged_during_evaluation: bool
    optimizer_unchanged_during_evaluation: bool
    replay_buffer_unchanged_during_evaluation: bool
    training_global_step_unchanged_during_evaluation: bool
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    runtime_info: Dict[str, Any]
    duration_seconds: float

    def as_dict(self) -> Dict[str, Any]:
        """Returns a notebook-friendly dictionary."""
        return {
            "LOCAL_E2E_SMOKE_PASS": self.local_e2e_smoke_pass,
            "E2E_SMOKE_PASS": self.e2e_smoke_pass,
            "runtime": self.runtime,
            "device": self.device,
            "require_cuda": self.require_cuda,
            "run_id": self.run_id,
            "observation_shape": self.observation_shape,
            "observation_dtype": self.observation_dtype,
            "action_space": self.action_space,
            "preflight": self.preflight.as_dict(),
            "segment_a": self.segment_a.as_dict(),
            "checkpoint": self.checkpoint.as_dict(),
            "restored": self.restored.as_dict(),
            "segment_b": self.segment_b.as_dict(),
            "tensorboard_event_files_before": self.tensorboard_event_files_before,
            "tensorboard_event_files_after": self.tensorboard_event_files_after,
            "tensorboard_tags": self.tensorboard_tags,
            "tensorboard_steps_before": self.tensorboard_steps_before,
            "tensorboard_steps_after": self.tensorboard_steps_after,
            "tensorboard_post_resume_steps": self.tensorboard_post_resume_steps,
            "tensorboard_previous_logs_preserved": self.tensorboard_previous_logs_preserved,
            "evaluation": self.evaluation.as_dict(),
            "online_unchanged_during_evaluation": self.online_unchanged_during_evaluation,
            "target_unchanged_during_evaluation": self.target_unchanged_during_evaluation,
            "optimizer_unchanged_during_evaluation": self.optimizer_unchanged_during_evaluation,
            "replay_buffer_unchanged_during_evaluation": self.replay_buffer_unchanged_during_evaluation,
            "training_global_step_unchanged_during_evaluation": self.training_global_step_unchanged_during_evaluation,
            "memory_before": self.memory_before.as_dict(),
            "memory_after": self.memory_after.as_dict(),
            "runtime_info": self.runtime_info,
            "duration_seconds": self.duration_seconds,
        }


def run_e2e_smoke(
    config: Mapping[str, Any],
    checkpoint_root: str | Path,
    tensorboard_root: str | Path,
    run_id: str,
    repo_path: str | Path = ".",
    require_cuda: Optional[bool] = None,
    device: str | torch.device | None = None,
) -> E2ESmokeSummary:
    """Runs the HU007 end-to-end smoke test.

    Args:
        config: Parsed project configuration.
        checkpoint_root: Root directory for smoke checkpoints.
        tensorboard_root: Root directory for TensorBoard events.
        run_id: Explicit smoke run identifier.
        repo_path: Git repository path recorded in checkpoint metadata.
        require_cuda: Optional override for ``e2e_smoke.require_cuda``.
        device: Optional device override used by local tests.

    Returns:
        Structured E2E smoke summary.

    Raises:
        RuntimeError: If CUDA is required but unavailable, preflight fails or
            an E2E invariant is violated.
    """
    start_time = time.perf_counter()
    smoke_config = _smoke_config(config)
    e2e_config = smoke_config["e2e_smoke"]
    selected_require_cuda = bool(e2e_config.get("require_cuda", True) if require_cuda is None else require_cuda)
    selected_device = _resolve_device(selected_require_cuda, device)
    runtime_info = get_runtime_info()
    runtime = "Google Colab" if _running_in_colab() else "local"
    memory_before = _memory_snapshot("before")

    preflight = run_preflight_checks(smoke_config, device=selected_device)
    if not preflight.ready_for_training:
        raise RuntimeError(f"Preflight failed; E2E smoke aborted: {preflight.errors}")

    env_probe = create_assault_env(smoke_config, mode="train", seed=int(smoke_config["reproducibility"]["seed"]))
    try:
        observation, _ = env_probe.reset()
        metadata = get_environment_metadata(env_probe, smoke_config, "train", int(smoke_config["reproducibility"]["seed"]))
    finally:
        env_probe.close()

    segment_a_timesteps = int(e2e_config["segment_a_timesteps"])
    final_timesteps = int(e2e_config["final_timesteps"])
    if final_timesteps <= segment_a_timesteps:
        raise ValueError("e2e_smoke.final_timesteps must be greater than segment_a_timesteps.")

    manager = CheckpointManager(checkpoint_root, run_id, repo_path=repo_path)
    manager.ensure_new_run()

    segment_a_config = copy.deepcopy(smoke_config)
    segment_a_config["training"]["total_timesteps"] = segment_a_timesteps
    agent_a = DDQNAgent(segment_a_config, device=selected_device, seed=int(segment_a_config["reproducibility"]["seed"]))
    buffer_a = ReplayBuffer(capacity=int(segment_a_config["replay_buffer"]["capacity"]), seed=int(segment_a_config["reproducibility"]["seed"]))
    logger_a = TensorBoardLogger.from_config(segment_a_config, run_id=run_id, log_root=tensorboard_root)
    env_a = create_assault_env(segment_a_config, mode="train", seed=int(segment_a_config["reproducibility"]["seed"]))
    try:
        segment_a = Trainer(env_a, agent_a, buffer_a, segment_a_config, metrics_logger=logger_a).train()
        checkpoint = manager.save(agent_a, buffer_a, segment_a_config, segment_a.global_step, segment_a, save_replay_buffer=True)
        logger_a.flush()
    finally:
        logger_a.close()
        env_a.close()

    _validate_segment_a(segment_a, checkpoint, segment_a_timesteps)
    run_log_dir = Path(tensorboard_root) / run_id
    event_files_before = logger_a.event_files()
    scalars_before = load_tensorboard_scalars(run_log_dir)

    segment_b_config = copy.deepcopy(smoke_config)
    segment_b_config["training"]["total_timesteps"] = final_timesteps
    agent_b = DDQNAgent(segment_b_config, device=selected_device, seed=int(segment_b_config["reproducibility"]["seed"]) + 1)
    buffer_b = ReplayBuffer(capacity=int(segment_b_config["replay_buffer"]["capacity"]), seed=int(segment_b_config["reproducibility"]["seed"]) + 1)
    restored = manager.load(checkpoint.path, agent_b, buffer_b, segment_b_config, mode="resume_full", map_location=selected_device)
    logger_b = TensorBoardLogger.from_config(segment_b_config, run_id=run_id, log_root=tensorboard_root)
    env_b = create_assault_env(segment_b_config, mode="train", seed=int(segment_b_config["reproducibility"]["seed"]) + 1)
    try:
        segment_b = Trainer(
            env_b,
            agent_b,
            buffer_b,
            segment_b_config,
            initial_global_step=restored.global_step,
            initial_metrics=restored.training_metrics,
            metrics_logger=logger_b,
        ).train()
        logger_b.flush()
    finally:
        logger_b.close()
        env_b.close()

    event_files_after = logger_b.event_files()
    scalars_after = load_tensorboard_scalars(run_log_dir)
    tensorboard_previous_logs_preserved = {path.name for path in event_files_before}.issubset(
        {path.name for path in event_files_after}
    )
    post_resume_steps = _post_resume_steps(scalars_after, restored.global_step)
    _validate_segment_b(segment_a, segment_b, restored, final_timesteps, post_resume_steps, tensorboard_previous_logs_preserved)

    training_buffer_size_before_eval = len(buffer_b)
    training_global_step_before_eval = segment_b.global_step
    online_before_eval = _clone_parameters(agent_b.online_network)
    target_before_eval = _clone_parameters(agent_b.target_network)
    optimizer_before_eval = copy.deepcopy(agent_b.optimizer.state_dict())

    eval_env = create_assault_env(segment_b_config, mode="eval", seed=int(segment_b_config["reproducibility"]["seed"]) + 2)
    try:
        evaluation = evaluate_agent(
            env=eval_env,
            agent=agent_b,
            episodes=int(e2e_config["evaluation_episodes"]),
            epsilon=float(e2e_config["evaluation_epsilon"]),
            max_steps_per_episode=e2e_config.get("evaluation_max_steps_per_episode"),
        )
    finally:
        eval_env.close()

    online_unchanged = _parameters_equal(online_before_eval, _clone_parameters(agent_b.online_network))
    target_unchanged = _parameters_equal(target_before_eval, _clone_parameters(agent_b.target_network))
    optimizer_unchanged = _optimizer_state_equal(optimizer_before_eval, agent_b.optimizer.state_dict())
    replay_unchanged = len(buffer_b) == training_buffer_size_before_eval
    step_unchanged = segment_b.global_step == training_global_step_before_eval
    _validate_evaluation(online_unchanged, target_unchanged, optimizer_unchanged, replay_unchanged, step_unchanged)

    memory_after = _memory_snapshot("after")
    is_colab_cuda = runtime == "Google Colab" and selected_device.type == "cuda" and torch.cuda.is_available()
    local_pass = runtime == "local" and selected_device.type == "cpu"
    return E2ESmokeSummary(
        local_e2e_smoke_pass=bool(local_pass),
        e2e_smoke_pass=bool(is_colab_cuda),
        runtime=runtime,
        device=str(selected_device),
        require_cuda=selected_require_cuda,
        run_id=run_id,
        observation_shape=tuple(observation.shape),
        observation_dtype=str(observation.dtype),
        action_space=str(metadata.action_space),
        preflight=preflight,
        segment_a=segment_a,
        checkpoint=checkpoint,
        restored=restored,
        segment_b=segment_b,
        tensorboard_event_files_before=len(event_files_before),
        tensorboard_event_files_after=len(event_files_after),
        tensorboard_tags=sorted(scalars_after.keys()),
        tensorboard_steps_before=_scalar_steps(scalars_before),
        tensorboard_steps_after=_scalar_steps(scalars_after),
        tensorboard_post_resume_steps=post_resume_steps,
        tensorboard_previous_logs_preserved=tensorboard_previous_logs_preserved,
        evaluation=evaluation,
        online_unchanged_during_evaluation=online_unchanged,
        target_unchanged_during_evaluation=target_unchanged,
        optimizer_unchanged_during_evaluation=optimizer_unchanged,
        replay_buffer_unchanged_during_evaluation=replay_unchanged,
        training_global_step_unchanged_during_evaluation=step_unchanged,
        memory_before=memory_before,
        memory_after=memory_after,
        runtime_info=runtime_info,
        duration_seconds=time.perf_counter() - start_time,
    )


def _smoke_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    smoke_config = copy.deepcopy(dict(config))
    e2e_config = smoke_config.get("e2e_smoke", {})
    if not e2e_config.get("enabled", True):
        raise RuntimeError("e2e_smoke.enabled is false.")
    smoke_config["replay_buffer"]["capacity"] = int(e2e_config.get("replay_buffer_capacity", smoke_config["replay_buffer"]["capacity"]))
    return smoke_config


def _resolve_device(require_cuda: bool, device: str | torch.device | None = None) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("e2e_smoke.require_cuda=true but CUDA is not available.")
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if require_cuda and selected.type != "cuda":
        raise RuntimeError(f"e2e_smoke.require_cuda=true but selected device is {selected}.")
    return selected


def _memory_snapshot(label: str) -> MemorySnapshot:
    process = psutil.Process()
    memory = psutil.virtual_memory()
    cuda_allocated = None
    cuda_reserved = None
    if torch.cuda.is_available():
        cuda_allocated = round(torch.cuda.memory_allocated() / (1024**2), 2)
        cuda_reserved = round(torch.cuda.memory_reserved() / (1024**2), 2)
    return MemorySnapshot(
        label=label,
        process_rss_mb=round(process.memory_info().rss / (1024**2), 2),
        ram_available_gb=round(memory.available / (1024**3), 2),
        cuda_allocated_mb=cuda_allocated,
        cuda_reserved_mb=cuda_reserved,
    )


def _validate_segment_a(segment: TrainingSummary, checkpoint: CheckpointMetadata, expected_step: int) -> None:
    if segment.global_step != expected_step or checkpoint.checkpoint_step != expected_step:
        raise RuntimeError("Segment A did not stop at the checkpoint step.")
    if segment.transitions_stored <= 0 or segment.updates_count <= 0:
        raise RuntimeError("Segment A did not produce transitions and updates.")
    if not segment.online_weights_changed:
        raise RuntimeError("Segment A did not change Online weights.")
    if segment.last_loss is None or not np.isfinite(segment.last_loss):
        raise RuntimeError("Segment A loss is missing or non-finite.")
    if segment.last_q_mean is None or not np.isfinite(segment.last_q_mean):
        raise RuntimeError("Segment A q_mean is missing or non-finite.")
    if not segment.target_sync_steps:
        raise RuntimeError("Segment A did not sync Target Network.")
    if checkpoint.size_bytes <= 0:
        raise RuntimeError("Segment A checkpoint is empty.")


def _validate_segment_b(
    segment_a: TrainingSummary,
    segment_b: TrainingSummary,
    restored: CheckpointState,
    expected_final_step: int,
    post_resume_steps: Mapping[str, list[int]],
    previous_logs_preserved: bool,
) -> None:
    if restored.global_step != segment_a.global_step or not restored.replay_buffer_restored:
        raise RuntimeError("Resume did not restore expected global step and Replay Buffer.")
    if segment_b.global_step != expected_final_step:
        raise RuntimeError("Segment B did not finish at final_timesteps.")
    if segment_b.transitions_stored <= 0 or segment_b.updates_count <= segment_a.updates_count:
        raise RuntimeError("Segment B did not produce post-resume transitions and updates.")
    if reconstruct_epsilon(restored.global_step, restored.config) != restored.epsilon:
        raise RuntimeError("Restored epsilon does not match checkpoint configuration.")
    required_tags = {"train/epsilon", "train/loss", "train/q_mean", "train/learning_rate"}
    missing_post_resume = [tag for tag in required_tags if not post_resume_steps.get(tag)]
    if missing_post_resume:
        raise RuntimeError(f"Missing post-resume TensorBoard steps: {missing_post_resume}")
    if not previous_logs_preserved:
        raise RuntimeError("TensorBoard logs from segment A were not preserved.")


def _validate_evaluation(
    online_unchanged: bool,
    target_unchanged: bool,
    optimizer_unchanged: bool,
    replay_unchanged: bool,
    step_unchanged: bool,
) -> None:
    if not all([online_unchanged, target_unchanged, optimizer_unchanged, replay_unchanged, step_unchanged]):
        raise RuntimeError("Evaluation mutated training state.")


def _scalar_steps(scalars: Mapping[str, list[tuple[int, float]]]) -> Dict[str, list[int]]:
    return {tag: [step for step, _ in events] for tag, events in scalars.items()}


def _post_resume_steps(scalars: Mapping[str, list[tuple[int, float]]], restored_step: int) -> Dict[str, list[int]]:
    return {tag: [step for step, _ in events if step > restored_step] for tag, events in scalars.items()}


def _clone_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone().cpu() for parameter in module.parameters()]


def _parameters_equal(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return all(torch.equal(left, right) for left, right in zip(before, after))


def _optimizer_state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    if left["param_groups"] != right["param_groups"]:
        return False
    if left["state"].keys() != right["state"].keys():
        return False
    for key, left_state in left["state"].items():
        right_state = right["state"][key]
        if left_state.keys() != right_state.keys():
            return False
        for state_key, left_value in left_state.items():
            right_value = right_state[state_key]
            if isinstance(left_value, torch.Tensor):
                if not torch.equal(left_value.detach().cpu(), right_value.detach().cpu()):
                    return False
            elif left_value != right_value:
                return False
    return True


def _running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True
