"""Small reusable utilities for BattleZone notebooks and tests."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


def get_project_root() -> Path:
    """Returns the BattleZone project root directory."""
    return Path(__file__).resolve().parents[1]


def get_git_commit(repo_root: Optional[str | Path] = None) -> Optional[str]:
    """Returns the current Git commit SHA when available.

    Args:
        repo_root: Repository root. Defaults to the current working directory.

    Returns:
        Commit SHA string or None when Git is unavailable.
    """
    cwd = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def get_runtime_info() -> Dict[str, Any]:
    """Collects runtime and hardware information for reproducibility reports."""
    gpu_available = False
    gpu_name = None
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        torch_version = torch.__version__
    except ImportError:
        torch_version = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gymnasium": _package_version("gymnasium"),
        "ale_py": _package_version("ale-py"),
        "numpy": _package_version("numpy"),
        "pillow": _package_version("Pillow"),
        "pyyaml": _package_version("PyYAML"),
        "torch": torch_version,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
    }


def _package_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
