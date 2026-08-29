"""Training profile and full-run preflight helpers for Assault DDQN."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

import numpy as np

from .session_bootstrap import TrainingSessionContext, compute_config_fingerprint
from .trainer import compute_epsilon


VALID_TRAINING_PROFILES = {"smoke", "full"}
DEFAULT_TRAINING_PROFILE = "smoke"
FULL_TRAINING_TARGET_TIMESTEPS = 250_000
DEFAULT_FULL_REPLAY_BUFFER_CAPACITY = 50_000
DEFAULT_FULL_MEMORY_MARGIN_GIB = 4.0


@dataclass(frozen=True)
class ReplayBufferMemoryEstimate:
    """Approximate memory required by the preallocated replay buffer arrays."""

    capacity: int
    state_shape: tuple[int, int, int]
    dtype: str
    state_bytes: int
    states_bytes: int
    next_states_bytes: int
    actions_bytes: int
    rewards_bytes: int
    dones_bytes: int
    total_bytes: int

    @property
    def total_mib(self) -> float:
        """Memory estimate in MiB."""
        return self.total_bytes / (1024**2)

    @property
    def total_gib(self) -> float:
        """Memory estimate in GiB."""
        return self.total_bytes / (1024**3)

    def as_dict(self) -> Dict[str, Any]:
        """Returns a serializable estimate."""
        return {
            "capacity": self.capacity,
            "state_shape": list(self.state_shape),
            "dtype": self.dtype,
            "state_bytes": self.state_bytes,
            "states_bytes": self.states_bytes,
            "next_states_bytes": self.next_states_bytes,
            "actions_bytes": self.actions_bytes,
            "rewards_bytes": self.rewards_bytes,
            "dones_bytes": self.dones_bytes,
            "total_bytes": self.total_bytes,
            "total_mib": self.total_mib,
            "total_gib": self.total_gib,
        }


@dataclass(frozen=True)
class TrainingProfileContext:
    """Resolved training profile and effective configuration."""

    name: str
    config: Dict[str, Any]
    target_timesteps: int
    replay_buffer_memory: ReplayBufferMemoryEstimate
    memory_margin_gib: float

    def as_dict(self) -> Dict[str, Any]:
        """Returns a notebook-friendly representation."""
        return {
            "name": self.name,
            "target_timesteps": self.target_timesteps,
            "replay_buffer_memory": self.replay_buffer_memory.as_dict(),
            "memory_margin_gib": self.memory_margin_gib,
        }


@dataclass(frozen=True)
class FullTrainingGate:
    """Result of the HU009 full-training readiness gate."""

    ready: bool
    issues: List[str]
    profile: str
    device: str
    runtime: str
    replay_buffer_memory: ReplayBufferMemoryEstimate
    ram_available_gib: Optional[float]
    ram_margin_gib: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        """Returns a serializable gate result."""
        return {
            "ready": self.ready,
            "issues": list(self.issues),
            "profile": self.profile,
            "device": self.device,
            "runtime": self.runtime,
            "replay_buffer_memory": self.replay_buffer_memory.as_dict(),
            "ram_available_gib": self.ram_available_gib,
            "ram_margin_gib": self.ram_margin_gib,
        }


def resolve_training_profile(
    config: Mapping[str, Any],
    profile_name: str = DEFAULT_TRAINING_PROFILE,
    target_timesteps: Optional[int] = None,
) -> TrainingProfileContext:
    """Applies a configured training profile to the base DDQN config.

    Args:
        config: Parsed ``ddqn_config.yaml``.
        profile_name: Explicit profile name, currently ``smoke`` or ``full``.
        target_timesteps: Optional external global target override.

    Returns:
        Effective profile context with merged config and memory estimate.

    Raises:
        ValueError: If the profile does not exist or target is invalid.
    """
    selected = str(profile_name or DEFAULT_TRAINING_PROFILE).strip().lower()
    if selected not in VALID_TRAINING_PROFILES:
        raise ValueError("ASSAULT_TRAINING_PROFILE must be 'smoke' or 'full'.")
    profiles = config.get("training_profiles", {})
    if selected not in profiles:
        raise ValueError(f"training_profiles.{selected} is not configured.")

    profile_config = dict(profiles[selected])
    effective = copy.deepcopy(dict(config))
    for key in ("description", "target_timesteps", "memory_margin_gib"):
        profile_config.pop(key, None)
    _deep_merge(effective, profile_config)

    configured_target = target_timesteps if target_timesteps is not None else profiles[selected].get("target_timesteps")
    if configured_target is None:
        configured_target = effective.get("training", {}).get("total_timesteps")
    target = int(configured_target)
    if target <= 0:
        raise ValueError("target_timesteps must be positive.")
    effective.setdefault("training", {})["total_timesteps"] = target
    effective["active_training_profile"] = selected

    estimate = estimate_replay_buffer_memory(
        capacity=int(effective["replay_buffer"]["capacity"]),
        state_shape=(
            int(effective["network"]["input_channels"]),
            int(effective["preprocessing"]["resize_height"]),
            int(effective["preprocessing"]["resize_width"]),
        ),
        dtype=str(effective["preprocessing"]["dtype"]),
    )
    margin = float(profiles[selected].get("memory_margin_gib", DEFAULT_FULL_MEMORY_MARGIN_GIB if selected == "full" else 0.5))
    return TrainingProfileContext(
        name=selected,
        config=effective,
        target_timesteps=target,
        replay_buffer_memory=estimate,
        memory_margin_gib=margin,
    )


def estimate_replay_buffer_memory(
    capacity: int,
    state_shape: tuple[int, int, int] = (4, 84, 84),
    dtype: str = "uint8",
) -> ReplayBufferMemoryEstimate:
    """Estimates replay buffer array memory without allocating the buffer."""
    if int(capacity) <= 0:
        raise ValueError("Replay Buffer capacity must be positive.")
    np_dtype = np.dtype(dtype)
    state_bytes = int(math.prod(tuple(state_shape)) * np_dtype.itemsize)
    states_bytes = int(capacity) * state_bytes
    next_states_bytes = int(capacity) * state_bytes
    actions_bytes = int(capacity) * np.dtype(np.int64).itemsize
    rewards_bytes = int(capacity) * np.dtype(np.float32).itemsize
    dones_bytes = int(capacity) * np.dtype(np.bool_).itemsize
    total = states_bytes + next_states_bytes + actions_bytes + rewards_bytes + dones_bytes
    return ReplayBufferMemoryEstimate(
        capacity=int(capacity),
        state_shape=tuple(int(value) for value in state_shape),
        dtype=str(np_dtype),
        state_bytes=state_bytes,
        states_bytes=states_bytes,
        next_states_bytes=next_states_bytes,
        actions_bytes=actions_bytes,
        rewards_bytes=rewards_bytes,
        dones_bytes=dones_bytes,
        total_bytes=total,
    )


def validate_replay_buffer_memory(
    estimate: ReplayBufferMemoryEstimate,
    ram_available_gib: Optional[float],
    memory_margin_gib: float = DEFAULT_FULL_MEMORY_MARGIN_GIB,
) -> None:
    """Fails fast when available RAM cannot safely hold the replay buffer."""
    if ram_available_gib is None:
        raise RuntimeError("RAM availability is required for full training.")
    required_gib = estimate.total_gib + float(memory_margin_gib)
    if float(ram_available_gib) < required_gib:
        raise RuntimeError(
            "Insufficient RAM for Replay Buffer: "
            f"available={float(ram_available_gib):.2f} GiB, required={required_gib:.2f} GiB."
        )


def evaluate_full_training_ready(
    profile_context: TrainingProfileContext,
    session_context: TrainingSessionContext,
    preflight_report: Any,
    runtime_info: Mapping[str, Any],
    observation_shape: tuple[int, int, int],
    observation_dtype: str,
    action_space: Any,
    runtime: str,
    device: str,
) -> FullTrainingGate:
    """Evaluates the HU009 full-training gate without starting training."""
    issues: List[str] = []
    if profile_context.name != "full":
        issues.append("profile_not_full")
    if not bool(getattr(preflight_report, "ready_for_training", False)):
        issues.append("READY_FOR_TRAINING_FALSE")
    if runtime != "Google Colab":
        issues.append("runtime_not_google_colab")
    if str(device) != "cuda":
        issues.append("device_not_cuda")
    if tuple(observation_shape) != (4, 84, 84):
        issues.append(f"observation_shape_invalid:{observation_shape}")
    if str(observation_dtype) != "uint8":
        issues.append(f"observation_dtype_invalid:{observation_dtype}")
    if str(action_space) != "Discrete(7)":
        issues.append(f"action_space_invalid:{action_space}")
    if compute_config_fingerprint(profile_context.config) != session_context.config_fingerprint:
        issues.append("config_fingerprint_mismatch")
    restored = session_context.restored_expected_step if session_context.restored_expected_step is not None else 0
    if int(profile_context.target_timesteps) <= int(restored):
        issues.append("target_not_greater_than_restored_step")
    if not _path_accessible(session_context.checkpoint_root):
        issues.append("checkpoint_root_not_accessible")
    if not _path_accessible(session_context.tensorboard_root):
        issues.append("tensorboard_root_not_accessible")
    if session_context.mlflow_enabled and not _tracking_store_accessible(session_context.tracking_uri):
        issues.append("tracking_store_not_accessible")

    ram_available = _runtime_gib(runtime_info.get("ram_available_gb"))
    ram_margin = None
    try:
        validate_replay_buffer_memory(
            profile_context.replay_buffer_memory,
            ram_available,
            memory_margin_gib=profile_context.memory_margin_gib,
        )
        ram_margin = ram_available - profile_context.replay_buffer_memory.total_gib if ram_available is not None else None
    except RuntimeError as exc:
        issues.append(str(exc))

    return FullTrainingGate(
        ready=not issues,
        issues=issues,
        profile=profile_context.name,
        device=str(device),
        runtime=str(runtime),
        replay_buffer_memory=profile_context.replay_buffer_memory,
        ram_available_gib=ram_available,
        ram_margin_gib=ram_margin,
    )


def assert_training_can_start(profile_context: TrainingProfileContext, full_training_gate: FullTrainingGate) -> None:
    """Fails fast before side effects when full training is not ready.

    Smoke sessions are allowed to continue even when the full-training gate is
    false because the gate only controls HU009 full runs.

    Args:
        profile_context: Resolved training profile.
        full_training_gate: Evaluated HU009 readiness gate.

    Raises:
        RuntimeError: If the active profile is ``full`` and the gate is not ready.
    """
    if profile_context.name == "full" and not full_training_gate.ready:
        raise RuntimeError(f"FULL_TRAINING_READY=False: {full_training_gate.issues}")


def expected_epsilon_at_step(config: Mapping[str, Any], global_step: int) -> float:
    """Computes epsilon from the active profile config and global step."""
    return compute_epsilon(
        int(global_step),
        float(config["agent"]["epsilon_start"]),
        float(config["agent"]["epsilon_final"]),
        int(config["training"]["epsilon_decay_steps"]),
    )


def _deep_merge(target: Dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _runtime_gib(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _path_accessible(path: str | Path) -> bool:
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".assault_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _tracking_store_accessible(tracking_uri: Optional[str]) -> bool:
    if tracking_uri is None:
        return False
    parsed = urlparse(str(tracking_uri))
    if parsed.scheme not in {"", "file"}:
        return True
    raw_path = parsed.path if parsed.scheme == "file" else str(tracking_uri)
    if parsed.netloc and parsed.scheme == "file":
        raw_path = f"//{parsed.netloc}{parsed.path}"
    if parsed.scheme == "file" and len(raw_path) >= 4 and raw_path[0] == "/" and raw_path[2] == ":":
        raw_path = raw_path[1:]
    return _path_accessible(raw_path)
