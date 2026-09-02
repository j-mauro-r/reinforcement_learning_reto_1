"""Compact inference artifacts for Assault DDQN models."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .agent import DDQNAgent
from .utils import get_git_commit


SCHEMA_VERSION = 1
MAX_INFERENCE_MODEL_BYTES = 100 * 1024 * 1024
FORBIDDEN_ARTIFACT_KEYS = {
    "target_network",
    "optimizer",
    "optimizer_state",
    "replay_buffer",
    "replay_buffer_state",
    "training_metrics",
}
REQUIRED_METADATA_KEYS = {
    "project_run_id",
    "source_checkpoint_step",
    "source_checkpoint_identity",
    "environment",
    "preprocessing",
}


@dataclass(frozen=True)
class ModelArtifactInfo:
    """Metadata returned after exporting or loading an inference artifact."""

    path: Path
    sha256: str
    size_bytes: int
    metadata: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


def export_inference_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    project_run_id: str,
    config: Mapping[str, Any] | None = None,
    source_checkpoint_step: int | None = None,
    repo_path: str | Path = ".",
    extra_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    max_size_bytes: int = MAX_INFERENCE_MODEL_BYTES,
) -> ModelArtifactInfo:
    """Exports a compact DDQN inference artifact from a training checkpoint."""
    source_path = Path(checkpoint_path)
    destination = Path(output_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source_path}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Inference model already exists: {destination}")
    if not str(project_run_id).strip():
        raise ValueError("project_run_id must be explicit and non-empty.")

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    _validate_checkpoint_for_export(checkpoint)
    checkpoint_config = checkpoint["config"]
    expected_config = dict(config) if config is not None else dict(checkpoint_config)
    _validate_contract(checkpoint_config, expected_config)

    step = int(source_checkpoint_step if source_checkpoint_step is not None else checkpoint["checkpoint_step"])
    network_config = {
        "input_channels": int(checkpoint_config["network"]["input_channels"]),
        "num_actions": int(checkpoint_config["network"]["num_actions"]),
        "architecture": "QNetwork",
    }
    metadata = _build_metadata(
        checkpoint=checkpoint,
        checkpoint_path=source_path,
        config=checkpoint_config,
        project_run_id=project_run_id,
        source_checkpoint_step=step,
        repo_path=repo_path,
        extra_metadata=extra_metadata,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "network": network_config,
        "metadata": metadata,
        "online_network": checkpoint["online_network"],
    }
    _assert_no_forbidden_keys(payload)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    size_bytes = destination.stat().st_size
    if size_bytes > int(max_size_bytes):
        destination.unlink(missing_ok=True)
        raise ValueError(f"Inference model exceeds size guardrail: {size_bytes} > {max_size_bytes} bytes.")
    sha256 = compute_sha256(destination)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    sidecar.write_text(f"{sha256}  {destination.name}\n", encoding="utf-8")
    metadata_json = destination.with_suffix(destination.suffix + ".metadata.json")
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return ModelArtifactInfo(path=destination, sha256=sha256, size_bytes=size_bytes, metadata=metadata)


def load_inference_model(
    artifact_path: str | Path,
    device: str | torch.device | None = None,
    expected_sha256: str | None = None,
    expected_project_run_id: str | None = None,
) -> tuple[DDQNAgent, ModelArtifactInfo]:
    """Loads a compact DDQN inference artifact into a fresh agent."""
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Inference model not found: {path}")
    actual_sha256 = compute_sha256(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"Inference model checksum mismatch: {actual_sha256} != {expected_sha256}.")

    payload = torch.load(path, map_location=device or "cpu", weights_only=True)
    _validate_artifact_payload(payload)
    metadata = dict(payload["metadata"])
    if expected_project_run_id and metadata["project_run_id"] != expected_project_run_id:
        raise ValueError(
            f"Inference model project_run_id mismatch: {metadata['project_run_id']} != {expected_project_run_id}."
        )

    config = _config_from_artifact(payload)
    agent = DDQNAgent(config, device=device, seed=metadata.get("seed"))
    try:
        agent.online_network.load_state_dict(payload["online_network"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Inference model weights are incompatible with the declared network: {exc}") from exc
    agent.sync_target_network()
    agent.online_network.eval()
    agent.target_network.eval()

    return agent, ModelArtifactInfo(
        path=path,
        sha256=actual_sha256,
        size_bytes=path.stat().st_size,
        metadata=metadata,
    )


def resolve_delivery_model_path(
    base: str | Path,
    assault_dir: str | Path,
    project_run_id: str,
) -> tuple[Path, str]:
    """Resolves the autonomous delivery model in the required local-first order.

    Priority:
      1. ASSAULT_DIR / "assault_ddqn_model.pt"
      2. BASE / "models" / PROJECT_RUN_ID / "assault_ddqn_model.pt"

    Returns:
        (resolved_path, source) where source is either LOCAL or DRIVE.
    """
    if not str(project_run_id).strip():
        raise ValueError("project_run_id must be explicit and non-empty.")

    local_path = Path(assault_dir) / "assault_ddqn_model.pt"
    if local_path.exists():
        return local_path, "LOCAL"

    drive_path = Path(base) / "models" / str(project_run_id) / "assault_ddqn_model.pt"
    if drive_path.exists():
        return drive_path, "DRIVE"

    searched = [
        str(local_path),
        str(drive_path),
    ]
    raise FileNotFoundError(
        "No assault DDQN delivery model found. Searched for: "
        + "; ".join(searched)
    )


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint_for_export(payload: Mapping[str, Any]) -> None:
    required = {"schema_version", "run_id", "checkpoint_step", "config", "online_network"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Checkpoint missing required keys for inference export: {missing}")
    if "optimizer" not in payload:
        raise ValueError("Source checkpoint must be a training checkpoint with optimizer metadata.")


def _validate_artifact_payload(payload: Mapping[str, Any]) -> None:
    required = {"schema_version", "created_at", "network", "metadata", "online_network"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Inference model missing required keys: {missing}")
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported inference model schema_version: {payload['schema_version']}")
    _assert_no_forbidden_keys(payload)
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("Inference model metadata must be a mapping.")
    missing_metadata = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    if missing_metadata:
        raise ValueError(f"Inference model metadata missing required keys: {missing_metadata}")
    network = payload["network"]
    if not isinstance(network, Mapping):
        raise ValueError("Inference model network metadata must be a mapping.")
    for key in ("input_channels", "num_actions", "architecture"):
        if key not in network:
            raise ValueError(f"Inference model network metadata missing {key}.")
    if int(network["input_channels"]) != 4 or int(network["num_actions"]) != 7:
        raise ValueError("Inference model network shape is incompatible with Assault DDQN.")
    if str(network["architecture"]) != "QNetwork":
        raise ValueError(f"Unsupported inference architecture: {network['architecture']}")


def _validate_contract(actual_config: Mapping[str, Any], expected_config: Mapping[str, Any]) -> None:
    for section, key in (
        ("environment", "id"),
        ("environment", "frame_skip"),
        ("environment", "repeat_action_probability"),
        ("environment", "full_action_space"),
        ("preprocessing", "grayscale"),
        ("preprocessing", "resize_height"),
        ("preprocessing", "resize_width"),
        ("preprocessing", "frame_stack"),
        ("network", "input_channels"),
        ("network", "num_actions"),
    ):
        actual = actual_config[section][key]
        expected = expected_config[section][key]
        if actual != expected:
            raise ValueError(f"Incompatible inference contract {section}.{key}: {actual} != {expected}.")


def _assert_no_forbidden_keys(payload: Mapping[str, Any]) -> None:
    forbidden = sorted(FORBIDDEN_ARTIFACT_KEYS & set(payload))
    if forbidden:
        raise ValueError(f"Inference model contains training-only keys: {forbidden}")


def _build_metadata(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    config: Mapping[str, Any],
    project_run_id: str,
    source_checkpoint_step: int,
    repo_path: str | Path,
    extra_metadata: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    checkpoint_sha256 = compute_sha256(checkpoint_path)
    metadata: Dict[str, Any] = {
        "project_run_id": str(project_run_id),
        "source_run_id": str(checkpoint["run_id"]),
        "source_checkpoint_step": int(source_checkpoint_step),
        "source_checkpoint_identity": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "source_checkpoint_git_commit": checkpoint.get("git_commit"),
        "export_git_commit": get_git_commit(repo_path),
        "config_fingerprint": (extra_metadata or {}).get("config_fingerprint"),
        "seed": config.get("reproducibility", {}).get("seed"),
        "gamma": float(config["agent"]["gamma"]),
        "learning_rate": float(config["agent"]["learning_rate"]),
        "environment": {
            "id": config["environment"]["id"],
            "frame_skip": int(config["environment"]["frame_skip"]),
            "repeat_action_probability": float(config["environment"]["repeat_action_probability"]),
            "full_action_space": bool(config["environment"]["full_action_space"]),
            "obs_type": config["environment"].get("obs_type"),
        },
        "preprocessing": {
            "grayscale": bool(config["preprocessing"]["grayscale"]),
            "resize_height": int(config["preprocessing"]["resize_height"]),
            "resize_width": int(config["preprocessing"]["resize_width"]),
            "frame_stack": int(config["preprocessing"]["frame_stack"]),
            "dtype": config["preprocessing"].get("dtype", "uint8"),
            "normalize_pixels_in_env": bool(config["preprocessing"].get("normalize_pixels_in_env", False)),
        },
    }
    training_metrics = checkpoint.get("training_metrics")
    if isinstance(training_metrics, Mapping):
        metadata["training_summary"] = dict(training_metrics)
    if extra_metadata:
        metadata.update(
            {
                key: value
                for key, value in extra_metadata.items()
                if key not in {"config_fingerprint", "training_summary"}
            }
        )
    return metadata


def _config_from_artifact(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = payload["metadata"]
    network = payload["network"]
    return {
        "environment": dict(metadata["environment"]),
        "preprocessing": dict(metadata["preprocessing"]),
        "reproducibility": {"seed": int(metadata.get("seed") or 0)},
        "network": {
            "input_channels": int(network["input_channels"]),
            "num_actions": int(network["num_actions"]),
        },
        "agent": {
            "gamma": float(metadata.get("gamma", 0.99)),
            "learning_rate": float(metadata.get("learning_rate", 0.0001)),
        },
    }
