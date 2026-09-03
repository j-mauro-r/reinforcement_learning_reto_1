"""Standalone BattleZone DQN inference artifacts for HU011B."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Optional

import numpy as np
import torch

from src.network import BattleZoneQNetwork
from src.persistence import CheckpointMetadata, checkpoint_config_snapshot, load_checkpoint


MODEL_FILENAME = "battlezone_dqn_model.pt"
MODEL_SCHEMA_VERSION = 1
MAX_MODEL_BYTES = 100 * 1024 * 1024
FORBIDDEN_KEYS = frozenset({
    "target_network", "optimizer", "optimizer_state", "replay_buffer",
    "replay_buffer_state", "trainer_state", "batches", "tensorboard",
})


@dataclass(frozen=True)
class ModelArtifactInfo:
    """Validated identity and lineage of one inference artifact."""

    path: Path
    sha256: str
    size_bytes: int
    metadata: dict[str, Any]


class InferenceDQNAgent:
    """Minimal greedy-capable agent with no optimizer, target net, or Replay."""

    def __init__(self, *, network: BattleZoneQNetwork, state_shape: tuple[int, ...],
                 action_dim: int, device: torch.device) -> None:
        self.online_network = network.to(device).eval()
        self.state_shape = tuple(state_shape)
        self.action_dim = int(action_dim)
        self.device = device

    def select_action(self, state: np.ndarray | torch.Tensor, epsilon: float = 0.0) -> int:
        """Selects epsilon-greedy action; delivery verification uses epsilon zero."""
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if float(epsilon) and np.random.random() < float(epsilon):
            return int(np.random.randint(self.action_dim))
        tensor = state if isinstance(state, torch.Tensor) else torch.as_tensor(state)
        with torch.no_grad():
            values = self.online_network(tensor.to(self.device))
        return int(values.argmax(dim=1).item())


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA256 without loading a potentially large source into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_inference_model(
    *, checkpoint_path: str | Path, output_path: str | Path,
    project_run_id: str, config: Mapping[str, Any], training_git_sha: str,
    config_fingerprint: Optional[str] = None, overwrite: bool = False,
    max_size_bytes: int = MAX_MODEL_BYTES,
    expected_source_step: Optional[int] = None,
) -> ModelArtifactInfo:
    """Exports only online-network weights and inference contracts atomically."""
    source = Path(checkpoint_path)
    destination = Path(output_path)
    if not project_run_id.strip():
        raise ValueError("project_run_id must be explicit.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Delivery model already exists: {destination}")
    checkpoint = load_checkpoint(checkpoint_path=source, map_location="cpu")
    metadata: CheckpointMetadata = checkpoint["metadata"]
    _validate_checkpoint_contract(checkpoint, config)
    if expected_source_step is not None and metadata.global_step != int(expected_source_step):
        raise ValueError(
            f"Source checkpoint global_step mismatch: {metadata.global_step} != {expected_source_step}."
        )
    network = _network_contract(config)
    fingerprint = config_fingerprint or _config_fingerprint(config)
    artifact_metadata = {
        "project_run_id": project_run_id,
        "algorithm": "DQN",
        "training_git_sha": str(training_git_sha),
        "config_fingerprint": fingerprint,
        "source_checkpoint_step": int(metadata.global_step),
        "source_checkpoint_identity": {
            "path": str(source),
            "sha256": compute_sha256(source),
            "size_bytes": source.stat().st_size,
        },
        "seed": int(metadata.seed),
        "state_shape": list(metadata.state_shape),
        "action_dim": int(metadata.action_dim),
        "environment": _environment_contract(config),
        "preprocessing": _preprocessing_contract(config),
    }
    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "network": network,
        "metadata": artifact_metadata,
        "online_network": checkpoint["agent_state"]["online_network"],
    }
    _assert_no_training_state(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _torch_save_atomic(destination, payload)
    if destination.stat().st_size >= int(max_size_bytes):
        destination.unlink(missing_ok=True)
        raise ValueError("Inference model exceeds the 100 MiB delivery guardrail.")
    digest = compute_sha256(destination)
    readable_metadata = dict(artifact_metadata, model_sha256=digest)
    _write_text_atomic(destination.with_suffix(".pt.sha256"), f"{digest}  {destination.name}\n")
    _write_json_atomic(destination.with_name(f"{destination.stem}.metadata.json"), readable_metadata)
    loaded, info = load_inference_model(
        destination, map_location="cpu", expected_sha256=digest,
        expected_project_run_id=project_run_id, expected_config=config,
    )
    del loaded
    return ModelArtifactInfo(info.path, info.sha256, info.size_bytes, readable_metadata)


def load_inference_model(
    artifact_path: str | Path, *, map_location: str | torch.device = "cpu",
    expected_sha256: Optional[str] = None,
    expected_project_run_id: Optional[str] = None,
    expected_config: Optional[Mapping[str, Any]] = None,
) -> tuple[InferenceDQNAgent, ModelArtifactInfo]:
    """Loads a compact artifact on CPU or CUDA and validates its contracts."""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Inference model not found: {path}")
    digest = compute_sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Inference model checksum mismatch.")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    _validate_artifact(payload)
    metadata = dict(payload["metadata"])
    if expected_project_run_id and metadata["project_run_id"] != expected_project_run_id:
        raise ValueError("Inference model run_id lineage mismatch.")
    if expected_config is not None:
        if metadata["environment"] != _environment_contract(expected_config):
            raise ValueError("Inference model environment contract mismatch.")
        if metadata["preprocessing"] != _preprocessing_contract(expected_config):
            raise ValueError("Inference model preprocessing contract mismatch.")
    network = payload["network"]
    device = torch.device(map_location)
    q_network = BattleZoneQNetwork(
        action_dim=int(network["action_dim"]), frame_stack=int(network["frame_stack"]),
        input_channels=int(network["input_channels"]), hidden_dim=int(network["hidden_dim"]),
        conv_channels=tuple(network["conv_channels"]),
    )
    try:
        q_network.load_state_dict(payload["online_network"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Inference weights are incompatible: {exc}") from exc
    agent = InferenceDQNAgent(
        network=q_network, state_shape=tuple(metadata["state_shape"]),
        action_dim=int(metadata["action_dim"]), device=device,
    )
    return agent, ModelArtifactInfo(path, digest, path.stat().st_size, metadata)


def load_checkpoint_inference_agent(
    checkpoint_path: str | Path, *, config: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> tuple[InferenceDQNAgent, CheckpointMetadata]:
    """Loads online weights from an explicit intermediate training checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, map_location=map_location)
    _validate_checkpoint_contract(checkpoint, config)
    metadata: CheckpointMetadata = checkpoint["metadata"]
    contract = _network_contract(config)
    network = BattleZoneQNetwork(
        action_dim=contract["action_dim"], frame_stack=contract["frame_stack"],
        input_channels=contract["input_channels"], hidden_dim=contract["hidden_dim"],
        conv_channels=contract["conv_channels"],
    )
    network.load_state_dict(checkpoint["agent_state"]["online_network"], strict=True)
    agent = InferenceDQNAgent(
        network=network, state_shape=metadata.state_shape,
        action_dim=metadata.action_dim, device=torch.device(map_location),
    )
    return agent, metadata


def resolve_delivery_model_path(
    *, battlezone_dir: str | Path, persistent_root: str | Path,
    project_run_id: str,
) -> tuple[Path, str]:
    """Resolves local delivery first, then an explicit-run persistent fallback."""
    if not project_run_id.strip():
        raise ValueError("project_run_id is required for an unambiguous Drive fallback.")
    local = Path(battlezone_dir) / MODEL_FILENAME
    fallback = Path(persistent_root) / "models" / project_run_id / MODEL_FILENAME
    if local.is_file():
        return local, "LOCAL_DELIVERY"
    if fallback.is_file():
        return fallback, "PERSISTENT_FALLBACK"
    raise FileNotFoundError(f"Delivery model not found. Searched: {local}; {fallback}")


def validate_model_artifact(
    path: str | Path, *, expected_sha256: str, expected_run_id: str,
    config: Mapping[str, Any], max_size_bytes: int = MAX_MODEL_BYTES,
) -> ModelArtifactInfo:
    """Runs checksum, size, lineage, contract, and round-trip validation."""
    model_path = Path(path)
    if model_path.stat().st_size >= max_size_bytes:
        raise ValueError("Inference model exceeds size guardrail.")
    _, info = load_inference_model(
        model_path, map_location="cpu", expected_sha256=expected_sha256,
        expected_project_run_id=expected_run_id, expected_config=config,
    )
    return info


def _validate_checkpoint_contract(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, CheckpointMetadata) or metadata.algorithm != "DQN":
        raise ValueError("Source must be a valid BattleZone DQN checkpoint.")
    expected_shape = tuple(config["validation"]["expected_final_shape"])
    expected_actions = int(config["environment"]["expected_action_space_n"])
    if metadata.state_shape != expected_shape:
        raise ValueError("Source checkpoint state_shape is incompatible.")
    if metadata.action_dim != expected_actions:
        raise ValueError("Source checkpoint action_dim is incompatible.")
    if int(metadata.batch_size) != int(config["dqn"]["batch_size"]):
        raise ValueError("Source checkpoint batch_size is incompatible.")
    if checkpoint.get("config_snapshot") != checkpoint_config_snapshot(config):
        raise ValueError("Source checkpoint configuration fingerprint contract is incompatible.")
    if config["environment"]["env_id"] != "ALE/BattleZone-v5":
        raise ValueError("Only ALE/BattleZone-v5 is supported.")


def _validate_artifact(payload: Mapping[str, Any]) -> None:
    required = {"schema_version", "created_at", "network", "metadata", "online_network"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Inference artifact missing keys: {sorted(missing)}")
    if int(payload["schema_version"]) != MODEL_SCHEMA_VERSION:
        raise ValueError("Unsupported inference artifact schema.")
    _assert_no_training_state(payload)
    metadata = payload["metadata"]
    if metadata.get("algorithm") != "DQN" or metadata["environment"].get("env_id") != "ALE/BattleZone-v5":
        raise ValueError("Artifact is not a BattleZone DQN model.")
    if tuple(metadata["state_shape"]) != (4, 128, 128, 3):
        raise ValueError("Artifact state_shape is incompatible with BattleZone.")


def _assert_no_training_state(payload: Mapping[str, Any]) -> None:
    forbidden = FORBIDDEN_KEYS.intersection(payload)
    if forbidden:
        raise ValueError(f"Training-only keys found: {sorted(forbidden)}")


def _network_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    network = config["dqn"]["network"]
    return {
        "architecture": "BattleZoneQNetwork",
        "action_dim": int(config["environment"]["expected_action_space_n"]),
        "frame_stack": int(config["preprocessing"]["frame_stack"]),
        "input_channels": int(network["input_channels"]),
        "hidden_dim": int(network["hidden_dim"]),
        "conv_channels": [int(value) for value in network["conv_channels"]],
    }


def _environment_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    env = config["environment"]
    return {
        "env_id": env["env_id"], "frameskip": int(env["frameskip"]),
        "repeat_action_probability": float(env["repeat_action_probability"]),
        "action_dim": int(env["expected_action_space_n"]),
    }


def _preprocessing_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    preprocessing = config["preprocessing"]
    return {
        "pipeline_name": preprocessing["pipeline_name"],
        "color_mode": preprocessing["color_mode"],
        "resize": [int(preprocessing["resize"]["height"]), int(preprocessing["resize"]["width"])],
        "crop_enabled": bool(preprocessing["crop"]["enabled"]),
        "frame_stack": int(preprocessing["frame_stack"]),
        "dtype": preprocessing["dtype"],
        "normalize": bool(preprocessing["normalize"]),
        "state_shape": list(config["validation"]["expected_final_shape"]),
    }


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _torch_save_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: Optional[Path] = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(dict(payload), indent=2, sort_keys=True))
