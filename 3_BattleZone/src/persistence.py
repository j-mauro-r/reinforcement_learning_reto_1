"""Checkpoint persistence and resume helpers for BattleZone HU007."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Mapping, Optional
import os

import torch

from src.agent import DQNAgent
from src.replay_buffer import ReplayBuffer

SUPPORTED_SCHEMA_VERSION = 1
CHECKPOINT_MODE_FULL = "full"
CHECKPOINT_MODE_LIGHTWEIGHT = "lightweight"
_ALLOWED_MODES = {CHECKPOINT_MODE_FULL, CHECKPOINT_MODE_LIGHTWEIGHT}


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata stored in every checkpoint payload."""

    schema_version: int
    checkpoint_mode: str
    algorithm: str
    global_step: int
    episode_index: int
    seed: int
    state_shape: tuple[int, ...]
    action_dim: int
    batch_size: int
    created_at: str


def build_checkpoint_metadata(
    *,
    checkpoint_mode: str,
    algorithm: str,
    global_step: int,
    episode_index: int,
    seed: int,
    state_shape: tuple[int, ...],
    action_dim: int,
    batch_size: int,
    schema_version: int = SUPPORTED_SCHEMA_VERSION,
) -> CheckpointMetadata:
    """Builds validated checkpoint metadata."""
    if checkpoint_mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported checkpoint_mode={checkpoint_mode!r}.")
    if int(schema_version) != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema_version}; supported={SUPPORTED_SCHEMA_VERSION}."
        )
    if str(algorithm) != "DQN":
        raise ValueError("HU007 supports algorithm='DQN' only.")

    return CheckpointMetadata(
        schema_version=int(schema_version),
        checkpoint_mode=str(checkpoint_mode),
        algorithm=str(algorithm),
        global_step=int(global_step),
        episode_index=int(episode_index),
        seed=int(seed),
        state_shape=tuple(int(x) for x in state_shape),
        action_dim=int(action_dim),
        batch_size=int(batch_size),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def build_checkpoint_payload(
    *,
    metadata: CheckpointMetadata,
    trainer_state: Mapping[str, Any],
    agent_state: Mapping[str, Any],
    replay_buffer_state: Optional[Mapping[str, Any]],
    config_snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    """Builds a serializable checkpoint payload."""
    if metadata.checkpoint_mode == CHECKPOINT_MODE_FULL and replay_buffer_state is None:
        raise ValueError("Full checkpoints require replay_buffer_state.")
    if metadata.checkpoint_mode == CHECKPOINT_MODE_LIGHTWEIGHT and replay_buffer_state is not None:
        raise ValueError("Lightweight checkpoints must not contain replay_buffer_state.")

    return {
        "schema_version": metadata.schema_version,
        "metadata": asdict(metadata),
        "trainer_state": dict(trainer_state),
        "agent_state": dict(agent_state),
        "replay_buffer_state": None if replay_buffer_state is None else dict(replay_buffer_state),
        "config_snapshot": dict(config_snapshot),
    }


def checkpoint_config_snapshot(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Returns minimal critical config for compatibility checks."""
    return {
        "algorithm": str(config.get("algorithm")),
        "environment": {
            "env_id": str(config["environment"]["env_id"]),
            "expected_action_space_n": int(config["environment"]["expected_action_space_n"]),
            "frameskip": int(config["environment"]["frameskip"]),
            "repeat_action_probability": float(config["environment"]["repeat_action_probability"]),
        },
        "validation": {
            "expected_final_shape": list(config["validation"]["expected_final_shape"]),
        },
        "dqn": {
            "batch_size": int(config["dqn"]["batch_size"]),
            "gamma": float(config["dqn"]["gamma"]),
            "replay_buffer": {
                "capacity": int(config["dqn"]["replay_buffer"]["capacity"]),
            },
        },
        "training": {
            "epsilon": {
                "start": float(config["training"]["epsilon"]["start"]),
                "end": float(config["training"]["epsilon"]["end"]),
                "decay_steps": int(config["training"]["epsilon"]["decay_steps"]),
            },
        },
    }


def save_checkpoint(
    *,
    checkpoint_path: str | Path,
    payload: Mapping[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    """Saves checkpoint using a temp file and atomic replace."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Checkpoint already exists: {path}")

    temp_file_path: Optional[Path] = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            suffix=".tmp",
            prefix=f"{path.stem}.",
            dir=str(path.parent),
            delete=False,
        ) as temp_file:
            temp_file_path = Path(temp_file.name)
            torch.save(dict(payload), temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_file_path, path)
    except Exception:
        if temp_file_path is not None and temp_file_path.exists():
            temp_file_path.unlink()
        raise

    return path


def load_checkpoint(
    *,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Loads and validates checkpoint payload structure and schema version."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if not path.is_file():
        raise ValueError(f"checkpoint_path must be a file path, got: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint payload type; expected dict.")

    required_keys = {
        "schema_version",
        "metadata",
        "trainer_state",
        "agent_state",
        "replay_buffer_state",
        "config_snapshot",
    }
    missing = required_keys.difference(payload.keys())
    if missing:
        raise ValueError(f"Invalid checkpoint payload; missing keys: {sorted(missing)}")

    schema_version = int(payload["schema_version"])
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema_version}; supported={SUPPORTED_SCHEMA_VERSION}."
        )

    metadata_raw = payload["metadata"]
    if not isinstance(metadata_raw, dict):
        raise ValueError("Invalid metadata in checkpoint payload.")

    metadata = CheckpointMetadata(
        schema_version=int(metadata_raw["schema_version"]),
        checkpoint_mode=str(metadata_raw["checkpoint_mode"]),
        algorithm=str(metadata_raw["algorithm"]),
        global_step=int(metadata_raw["global_step"]),
        episode_index=int(metadata_raw["episode_index"]),
        seed=int(metadata_raw["seed"]),
        state_shape=tuple(int(x) for x in metadata_raw["state_shape"]),
        action_dim=int(metadata_raw["action_dim"]),
        batch_size=int(metadata_raw["batch_size"]),
        created_at=str(metadata_raw["created_at"]),
    )
    payload["metadata"] = metadata
    return payload


def validate_checkpoint_compatibility(
    *,
    payload: Mapping[str, Any],
    agent: DQNAgent,
    config: Mapping[str, Any],
    expected_mode: str,
) -> None:
    """Validates schema, mode and structural compatibility before restore."""
    metadata = payload["metadata"]
    if not isinstance(metadata, CheckpointMetadata):
        raise TypeError("payload metadata must be CheckpointMetadata.")

    if expected_mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported expected_mode={expected_mode!r}.")
    if metadata.checkpoint_mode != expected_mode:
        raise ValueError(
            f"Checkpoint mode mismatch. expected={expected_mode}, incoming={metadata.checkpoint_mode}."
        )
    if metadata.algorithm != "DQN":
        raise ValueError(f"Unsupported algorithm in checkpoint: {metadata.algorithm}")
    if str(config.get("algorithm")) != "DQN":
        raise ValueError("Active config algorithm must be DQN.")

    if int(metadata.action_dim) != int(agent.action_dim):
        raise ValueError(
            f"Incompatible action_dim. expected={agent.action_dim}, incoming={metadata.action_dim}."
        )
    if tuple(metadata.state_shape) != tuple(agent.state_shape):
        raise ValueError(
            f"Incompatible state_shape. expected={agent.state_shape}, incoming={metadata.state_shape}."
        )
    if int(metadata.batch_size) != int(agent.batch_size):
        raise ValueError(
            f"Incompatible batch_size. expected={agent.batch_size}, incoming={metadata.batch_size}."
        )

    expected_schema = int(config.get("checkpointing", {}).get("schema_version", SUPPORTED_SCHEMA_VERSION))
    if int(metadata.schema_version) != expected_schema:
        raise ValueError(
            f"schema_version mismatch. expected={expected_schema}, incoming={metadata.schema_version}."
        )

    snapshot = payload.get("config_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("config_snapshot must be a dict.")
    active_snapshot = checkpoint_config_snapshot(config)
    _validate_critical_snapshot_fields(
        checkpoint_snapshot=snapshot,
        active_snapshot=active_snapshot,
    )

    replay_state = payload.get("replay_buffer_state")
    if expected_mode == CHECKPOINT_MODE_FULL and replay_state is None:
        raise ValueError("Full checkpoint requires replay_buffer_state.")
    if expected_mode == CHECKPOINT_MODE_LIGHTWEIGHT and replay_state is not None:
        raise ValueError("Lightweight checkpoint must not include replay_buffer_state.")


def restore_training_state(
    *,
    checkpoint_path: str | Path,
    agent: DQNAgent,
    config: Mapping[str, Any],
    expected_mode: str,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Restores agent and replay state; returns trainer-compatible resume state."""
    payload = load_checkpoint(checkpoint_path=checkpoint_path, map_location=map_location)
    validate_checkpoint_compatibility(
        payload=payload,
        agent=agent,
        config=config,
        expected_mode=expected_mode,
    )

    metadata: CheckpointMetadata = payload["metadata"]
    agent.load_state_dict(payload["agent_state"])

    replay_restored = False
    if expected_mode == CHECKPOINT_MODE_FULL:
        replay_state = payload["replay_buffer_state"]
        agent.replay_buffer.load_state_dict(replay_state)
        replay_restored = True
    else:
        agent.replay_buffer = ReplayBuffer(
            capacity=agent.replay_buffer.capacity,
            state_shape=agent.state_shape,
        )

    trainer_state = payload["trainer_state"]
    return {
        "global_step": int(trainer_state["global_step"]),
        "episode_index": int(trainer_state["episode_index"]),
        "episode_step": int(trainer_state.get("episode_step", 0)),
        "episode_reward": float(trainer_state.get("episode_reward", 0.0)),
        "seed": int(metadata.seed),
        "replay_restored": bool(replay_restored),
        "checkpoint_mode": str(metadata.checkpoint_mode),
        "checkpoint_metadata": metadata,
    }


def checkpoint_filename(*, global_step: int, checkpoint_mode: str) -> str:
    """Returns deterministic checkpoint filename for a training step and mode."""
    if checkpoint_mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported checkpoint_mode={checkpoint_mode!r}.")
    return f"battlezone_dqn_step_{int(global_step):08d}_{checkpoint_mode}.pt"


def _get_nested_value(snapshot: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = snapshot
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            dotted = ".".join(path)
            raise ValueError(f"Missing checkpoint config field: {dotted}")
        value = value[key]
    return value


def _validate_critical_snapshot_fields(
    *,
    checkpoint_snapshot: Mapping[str, Any],
    active_snapshot: Mapping[str, Any],
) -> None:
    float_paths = {
        ("environment", "repeat_action_probability"),
        ("dqn", "gamma"),
        ("training", "epsilon", "start"),
        ("training", "epsilon", "end"),
    }
    shape_paths = {
        ("validation", "expected_final_shape"),
    }
    critical_paths = [
        ("algorithm",),
        ("environment", "env_id"),
        ("environment", "expected_action_space_n"),
        ("environment", "frameskip"),
        ("environment", "repeat_action_probability"),
        ("validation", "expected_final_shape"),
        ("dqn", "batch_size"),
        ("dqn", "gamma"),
        ("dqn", "replay_buffer", "capacity"),
        ("training", "epsilon", "start"),
        ("training", "epsilon", "end"),
        ("training", "epsilon", "decay_steps"),
    ]

    for path in critical_paths:
        checkpoint_value = _get_nested_value(checkpoint_snapshot, path)
        active_value = _get_nested_value(active_snapshot, path)

        equal = False
        if path in float_paths:
            equal = math.isclose(
                float(checkpoint_value),
                float(active_value),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        elif path in shape_paths:
            equal = tuple(checkpoint_value) == tuple(active_value)
        else:
            equal = checkpoint_value == active_value

        if not equal:
            dotted = ".".join(path)
            raise ValueError(
                f"Checkpoint config mismatch for {dotted}: "
                f"checkpoint={checkpoint_value!r}, active={active_value!r}."
            )
