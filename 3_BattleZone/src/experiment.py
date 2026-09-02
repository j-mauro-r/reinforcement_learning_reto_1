"""Lightweight experiment traceability for BattleZone HU010."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import secrets
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Optional

import yaml


MANIFEST_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(
    r"^battlezone-dqn-\d{8}-\d{6}-[0-9a-f]{7,40}-[0-9a-f]{4,16}$"
)
RUN_STATUSES = {"created", "running", "completed", "interrupted", "failed"}
RUN_MODES = {"new", "resume_full", "resume_lightweight"}
CRITICAL_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "project",
    "algorithm",
    "status",
    "mode",
    "git",
    "environment",
    "config",
    "progress",
    "artifacts",
    "resume",
    "sessions",
}


@dataclass(frozen=True)
class LongTrainingReadiness:
    """Result of the reusable preflight gate for a long DQN run."""

    ready: bool
    checks: dict[str, bool]
    errors: list[str]
    details: dict[str, Any]


def utc_now() -> str:
    """Returns an ISO-8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(
    *, git_sha: str, now: Optional[datetime] = None, suffix: Optional[str] = None
) -> str:
    """Generates a filesystem-safe and reasonably unique BattleZone run ID.

    Args:
        git_sha: Resolved Git commit SHA.
        now: Optional timestamp, primarily for deterministic tests.
        suffix: Optional hexadecimal collision-avoidance suffix.

    Returns:
        A validated run identifier.

    Raises:
        ValueError: If the SHA or suffix cannot form a valid identifier.
    """
    sha = str(git_sha).lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        raise ValueError("git_sha must contain 7 to 40 hexadecimal characters.")
    token = (suffix or secrets.token_hex(2)).lower()
    if not re.fullmatch(r"[0-9a-f]{4,16}", token):
        raise ValueError("suffix must contain 4 to 16 hexadecimal characters.")
    instant = now or datetime.now(timezone.utc)
    run_id = f"battlezone-dqn-{instant.astimezone(timezone.utc):%Y%m%d-%H%M%S}-{sha[:7]}-{token}"
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("Generated run_id is invalid.")
    return run_id


def is_valid_run_id(run_id: str) -> bool:
    """Returns whether a run ID follows the HU010 portable format."""
    return bool(RUN_ID_PATTERN.fullmatch(str(run_id)))


def load_config_snapshot(config_path: str | Path) -> tuple[dict[str, Any], str]:
    """Loads a YAML configuration and returns its snapshot and SHA-256."""
    path = Path(config_path)
    raw = path.read_bytes()
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Configuration root must be a mapping.")
    # JSON roundtrip both proves serializability and detaches YAML-specific values.
    snapshot = json.loads(json.dumps(parsed, allow_nan=False))
    return snapshot, hashlib.sha256(raw).hexdigest()


def capture_git_lineage(repo_root: str | Path) -> dict[str, Any]:
    """Captures the current commit, branch and dirty flag without changing Git."""
    root = Path(repo_root)

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to capture Git lineage for {root}") from exc
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("Git returned an invalid commit SHA.")
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_runtime() -> dict[str, Optional[str]]:
    """Captures real runtime versions, using None when unavailable."""
    return {
        "python": platform.python_version(),
        "gymnasium": _version("gymnasium"),
        "ale_py": _version("ale-py"),
        "torch": _version("torch"),
        "tensorboard": _version("tensorboard"),
        "platform": platform.platform(),
    }


def capture_hardware(device: Optional[str] = None) -> dict[str, Any]:
    """Captures CPU, RAM and accelerator metadata without assuming CUDA."""
    cuda_available = False
    gpu_name = None
    mps_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
    except ImportError:
        pass
    selected = device or ("cuda" if cuda_available else "mps" if mps_available else "cpu")
    ram_gb = None
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass
    return {
        "device": selected,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "cpu": platform.processor() or platform.machine() or None,
        "ram_gb": ram_gb,
        "platform": platform.platform(),
    }


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validates schema version and critical manifest invariants."""
    if not isinstance(manifest, Mapping):
        raise TypeError("Manifest root must be a mapping.")
    missing = CRITICAL_MANIFEST_FIELDS.difference(manifest)
    if missing:
        raise ValueError(f"Manifest missing critical fields: {sorted(missing)}")
    version = manifest["schema_version"]
    if type(version) is not int or version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version={version!r}; supported={MANIFEST_SCHEMA_VERSION}."
        )
    if not is_valid_run_id(str(manifest["run_id"])):
        raise ValueError("Manifest contains an invalid run_id.")
    if manifest["project"] != "BattleZone" or manifest["algorithm"] != "DQN":
        raise ValueError("Manifest project/algorithm must be BattleZone/DQN.")
    if manifest["status"] not in RUN_STATUSES or manifest["mode"] not in RUN_MODES:
        raise ValueError("Manifest contains an invalid status or mode.")
    if not isinstance(manifest["sessions"], list):
        raise TypeError("Manifest sessions must be a list.")


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Validates and atomically writes a UTF-8 run manifest."""
    validate_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f"{target.name}.", suffix=".tmp",
            dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        raise
    return target


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    """Loads a manifest and fails explicitly on corrupt or unsupported JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def create_run_manifest(
    *, results_dir: str | Path, manifest_filename: str, run_id: str,
    config_path: str | Path, config_snapshot: Mapping[str, Any], config_sha256: str,
    git: Mapping[str, Any], runtime: Mapping[str, Any], hardware: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Creates a protected canonical run directory and schema-v1 manifest."""
    if not is_valid_run_id(run_id):
        raise ValueError("Invalid run_id.")
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "summaries").mkdir()
    timestamp = utc_now()
    env_cfg = config_snapshot["environment"]
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "project": "BattleZone",
        "algorithm": "DQN",
        "status": "created",
        "mode": "new",
        "created_at_utc": timestamp,
        "updated_at_utc": timestamp,
        "git": dict(git),
        "environment": {"env_id": env_cfg["env_id"], "seed": int(env_cfg["seed"])},
        "config": {"path": str(config_path), "sha256": config_sha256, "snapshot": deepcopy(dict(config_snapshot))},
        "runtime": dict(runtime),
        "hardware": dict(hardware),
        "progress": {"start_global_step": 0, "end_global_step": 0, "episode_index": 0, "elapsed_seconds": 0.0},
        "artifacts": {"tensorboard_log_dir": None, "input_checkpoint": None, "output_checkpoint": None, "model_path": None, "evaluation_path": None},
        "resume": {"parent_checkpoint": None, "replay_restored": None},
        "sessions": [],
        "notes": [],
    }
    path = run_dir / manifest_filename
    try:
        write_run_manifest(path, manifest)
    except Exception:
        # Only remove the empty structure created by this failed operation.
        (run_dir / "summaries").rmdir()
        run_dir.rmdir()
        raise
    return manifest, path


def start_session(
    manifest: Mapping[str, Any], *, mode: str, start_global_step: int,
    input_checkpoint: Optional[str], tensorboard_log_dir: str, device: str,
    replay_restored: Optional[bool], manifest_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Appends and starts one NEW or RESUME session."""
    current = deepcopy(dict(manifest))
    validate_manifest(current)
    if mode not in RUN_MODES:
        raise ValueError(f"Unsupported session mode: {mode}")
    if current["status"] == "running":
        raise ValueError("Cannot start a second session while one is running.")
    if mode == "new" and current["sessions"]:
        raise ValueError("NEW is only valid for the first session.")
    if mode != "new" and not current["sessions"]:
        raise ValueError("RESUME requires a previous session.")
    if mode != "new" and not input_checkpoint:
        raise ValueError("RESUME requires an explicit input_checkpoint.")
    if current["sessions"]:
        previous_end = int(current["sessions"][-1]["end_global_step"])
        if int(start_global_step) != previous_end:
            raise ValueError(f"Resume discontinuity: expected {previous_end}, got {start_global_step}.")
    session = {
        "session_index": len(current["sessions"]) + 1,
        "mode": mode,
        "started_at_utc": utc_now(), "ended_at_utc": None,
        "start_global_step": int(start_global_step), "end_global_step": int(start_global_step),
        "elapsed_seconds": 0.0, "input_checkpoint": input_checkpoint,
        "output_checkpoint": None, "tensorboard_log_dir": str(tensorboard_log_dir),
        "device": str(device), "replay_restored": replay_restored, "status": "running",
    }
    current["sessions"].append(session)
    current["status"] = "running"
    current["mode"] = mode
    current["updated_at_utc"] = utc_now()
    current["progress"]["start_global_step"] = int(start_global_step)
    current["artifacts"]["input_checkpoint"] = input_checkpoint
    current["artifacts"]["tensorboard_log_dir"] = str(tensorboard_log_dir)
    current["resume"] = {"parent_checkpoint": input_checkpoint, "replay_restored": replay_restored}
    if manifest_path is not None:
        write_run_manifest(manifest_path, current)
    return current


def finish_session(
    manifest: Mapping[str, Any], *, end_global_step: int, episode_index: int,
    elapsed_seconds: float, output_checkpoint: Optional[str], completed: bool,
    manifest_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Finishes the active session as completed or interrupted."""
    current = deepcopy(dict(manifest))
    validate_manifest(current)
    if current["status"] != "running" or not current["sessions"]:
        raise ValueError("No running session to finish.")
    session = current["sessions"][-1]
    end = int(end_global_step)
    if end <= int(session["start_global_step"]):
        raise ValueError("A finished session must advance global_step.")
    status = "completed" if completed else "interrupted"
    session.update({"ended_at_utc": utc_now(), "end_global_step": end, "elapsed_seconds": float(elapsed_seconds), "output_checkpoint": output_checkpoint, "status": status})
    current["status"] = status
    current["updated_at_utc"] = utc_now()
    current["progress"].update({"end_global_step": end, "episode_index": int(episode_index), "elapsed_seconds": float(elapsed_seconds)})
    current["artifacts"]["output_checkpoint"] = output_checkpoint
    if manifest_path is not None:
        write_run_manifest(manifest_path, current)
    return current


def fail_session(
    manifest: Mapping[str, Any], *, error: Exception | str,
    manifest_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Marks the active session failed while preserving its history."""
    current = deepcopy(dict(manifest))
    validate_manifest(current)
    if current["status"] != "running" or not current["sessions"]:
        raise ValueError("No running session to fail.")
    current["sessions"][-1].update({"ended_at_utc": utc_now(), "status": "failed"})
    current["status"] = "failed"
    current["updated_at_utc"] = utc_now()
    current["notes"].append(f"Session failed: {error}")
    if manifest_path is not None:
        write_run_manifest(manifest_path, current)
    return current


def validate_resume_compatibility(
    manifest: Mapping[str, Any], config_snapshot: Mapping[str, Any]
) -> None:
    """Rejects a resume when critical algorithm/environment/config differ."""
    validate_manifest(manifest)
    if config_snapshot.get("algorithm") != "DQN":
        raise ValueError("Resume requires algorithm DQN.")
    if config_snapshot.get("environment", {}).get("env_id") != "ALE/BattleZone-v5":
        raise ValueError("Resume requires ALE/BattleZone-v5.")
    previous = manifest["config"]["snapshot"]
    for key in ("algorithm", "environment", "preprocessing", "dqn"):
        if previous.get(key) != config_snapshot.get(key):
            raise ValueError(f"Incompatible resume configuration section: {key}")


def validate_long_training_readiness(
    *, config: Mapping[str, Any], config_path: str | Path, run_id: str,
    git: Mapping[str, Any], results_dir: Optional[str | Path] = None,
) -> LongTrainingReadiness:
    """Checks whether HU010 infrastructure is ready for a future long run."""
    tracking = config.get("tracking", {})
    resolved_results = Path(results_dir or tracking.get("results_dir", ""))
    checks = {
        "tracking_enabled": tracking.get("enabled") is True,
        "results_dir_writable": False,
        "run_id_valid": is_valid_run_id(run_id),
        "config_loaded": bool(config),
        "config_sha256_available": False,
        "git_sha_available": bool(re.fullmatch(r"[0-9a-fA-F]{40}", str(git.get("commit", "")))),
        "git_clean_when_required": not (tracking.get("require_clean_git_for_long_run", True) and bool(git.get("dirty"))),
        "tensorboard_configured": config.get("tensorboard", {}).get("enabled") is True and bool(config.get("tensorboard", {}).get("log_dir")),
        "checkpointing_configured": config.get("checkpointing", {}).get("enabled") is True and bool(config.get("checkpointing", {}).get("directory")),
        "manifest_writable": False,
        "algorithm_dqn": config.get("algorithm") == "DQN",
        "battlezone_env": config.get("environment", {}).get("env_id") == "ALE/BattleZone-v5",
    }
    details: dict[str, Any] = {"results_dir": str(resolved_results), "run_id": run_id}
    try:
        _, digest = load_config_snapshot(config_path)
        checks["config_sha256_available"] = bool(re.fullmatch(r"[0-9a-f]{64}", digest))
        details["config_sha256"] = digest
    except (OSError, ValueError, TypeError) as exc:
        details["config_error"] = str(exc)
    try:
        resolved_results.mkdir(parents=True, exist_ok=True)
        checks["results_dir_writable"] = os.access(resolved_results, os.W_OK)
        with NamedTemporaryFile(mode="w", dir=resolved_results, delete=True) as probe:
            probe.write("{}")
            probe.flush()
            os.fsync(probe.fileno())
        checks["manifest_writable"] = True
    except OSError as exc:
        details["manifest_write_error"] = str(exc)
    errors = [name for name, passed in checks.items() if not passed]
    return LongTrainingReadiness(not errors, checks, errors, details)
