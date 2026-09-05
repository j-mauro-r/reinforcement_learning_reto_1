"""HU011B delivery layout, manifest, and evidence gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from src.model_artifact import compute_sha256


MANDATORY_KEYS = (
    "model", "model_checksum", "model_metadata", "round_trip_load",
    "persistent_delivery_model", "local_delivery_model", "model_sha_consistency",
    "greedy_action", "standalone_episode", "training_reward_figure",
    "training_loss_figure", "training_q_epsilon_figure", "training_video",
    "training_video_metadata", "post_training_video",
    "post_training_video_metadata", "delivery_manifest", "run_lineage",
)

_FULL_CHECKPOINT_PATTERN = re.compile(
    r"^battlezone_dqn_step_(\d{8})_full\.pt$"
)


def resolve_latest_full_checkpoint(
    checkpoint_dir: str | Path, final_step: int,
) -> tuple[Path, int]:
    """Returns the latest FULL checkpoint strictly before the final step."""
    directory = Path(checkpoint_dir)
    candidates: list[tuple[int, Path]] = []
    if directory.is_dir():
        for path in directory.glob("battlezone_dqn_step_*_full.pt"):
            match = _FULL_CHECKPOINT_PATTERN.fullmatch(path.name)
            if match:
                step = int(match.group(1))
                if 0 < step < int(final_step):
                    candidates.append((step, path))
    if not candidates:
        raise RuntimeError(
            f"No FULL checkpoint found with 0 < step < {int(final_step)} in {directory}."
        )
    step, path = max(candidates, key=lambda candidate: candidate[0])
    return path, step


def build_delivery_paths(persistent_root: str | Path, project_run_id: str) -> dict[str, Path]:
    """Builds deterministic HU011B paths for one explicit run."""
    if not project_run_id.strip():
        raise ValueError("project_run_id must be explicit.")
    root = Path(persistent_root) / "delivery" / project_run_id
    return {
        "root": root, "model_dir": root / "model", "figures_dir": root / "figures",
        "videos_dir": root / "videos", "manifest": root / "delivery_manifest.json",
        "model": root / "model" / "battlezone_dqn_model.pt",
        "training_reward": root / "figures" / "training_reward.png",
        "training_loss": root / "figures" / "training_loss.png",
        "training_q_epsilon": root / "figures" / "training_q_epsilon.png",
        "exploitation_reward": root / "figures" / "exploitation_reward.png",
        "training_video": root / "videos" / "battlezone_dqn_training_process.mp4",
        "post_training_video": root / "videos" / "battlezone_dqn_post_training.mp4",
    }


def write_delivery_manifest(
    *, path: str | Path, project_run_id: str, training_git_sha: str,
    delivery_git_sha: str, source_final_checkpoint: str | Path,
    source_intermediate_checkpoint: str | Path, tensorboard_logs: str | Path,
    delivery_model_path: str | Path, persistent_model_path: str | Path,
    local_project_model_path: str | Path, figures: Mapping[str, str | Path],
    videos: Mapping[str, str | Path], environment: str = "ALE/BattleZone-v5",
    algorithm: str = "DQN",
) -> Path:
    """Writes a distinct delivery manifest atomically with source lineage."""
    delivery_model = Path(delivery_model_path)
    persistent_model = Path(persistent_model_path)
    local_model = Path(local_project_model_path)
    hashes = {compute_sha256(path) for path in (delivery_model, persistent_model, local_model)}
    if len(hashes) != 1:
        raise ValueError("Delivery, persistent, and local project model SHA256 values differ.")
    model_sha256 = hashes.pop()
    payload = {
        "schema_version": 1, "run_id": project_run_id,
        "training_git_sha": training_git_sha, "delivery_git_sha": delivery_git_sha,
        "environment": environment, "algorithm": algorithm,
        "delivery_model_path": str(delivery_model),
        "persistent_model_path": str(persistent_model),
        "local_project_model_path": str(local_model),
        "model_sha256": model_sha256,
        "source_final_checkpoint": str(source_final_checkpoint),
        "source_intermediate_checkpoint": str(source_intermediate_checkpoint),
        "tensorboard_logs": str(tensorboard_logs),
        "figures": {key: str(value) for key, value in figures.items()},
        "videos": {key: str(value) for key, value in videos.items()},
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "materialized",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def evaluate_delivery_gate(statuses: Mapping[str, bool]) -> dict[str, Any]:
    """Returns PASS only when every mandatory artifact and validation passes."""
    missing = sorted(set(MANDATORY_KEYS) - set(statuses))
    if missing:
        raise ValueError(f"HU011B delivery gate missing checks: {missing}")
    checks = {key: bool(statuses[key]) for key in MANDATORY_KEYS}
    passed = all(checks.values())
    return {
        "checks": checks,
        "exploitation_reward": "PENDING_HU013",
        "HU011B_DELIVERY_GATE": "PASS" if passed else "FAIL",
    }
