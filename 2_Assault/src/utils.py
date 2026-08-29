"""Small cross-cutting utilities for the Assault project."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import psutil
import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML content is empty or not a mapping.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def set_global_seed(seed: int) -> None:
    """Sets seeds for libraries used by HU002.

    Args:
        seed: Base seed used for Python and NumPy random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_commit(repo_path: str | Path = ".") -> Optional[str]:
    """Returns the current Git commit SHA when available.

    Args:
        repo_path: Repository path used as working directory for Git.

    Returns:
        Current commit SHA, or None when Git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_path),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_runtime_info() -> Dict[str, Any]:
    """Collects runtime versions and hardware information.

    Returns:
        Dictionary with Python, Gymnasium, ALE-Py, CPU, RAM and GPU metadata.
        GPU fields are populated when PyTorch can query CUDA, otherwise they
        explicitly report that no GPU is available.
    """
    memory = psutil.virtual_memory()
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "gymnasium_version": _package_version("gymnasium"),
        "ale_py_version": _package_version("ale-py"),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(memory.total / (1024**3), 2),
        "ram_available_gb": round(memory.available / (1024**3), 2),
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_total_gb": None,
        "cuda_version": None,
    }

    try:
        import torch

        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            info.update(
                {
                    "gpu_available": True,
                    "gpu_name": properties.name,
                    "gpu_vram_total_gb": round(properties.total_memory / (1024**3), 2),
                }
            )
    except ImportError:
        info["torch_version"] = None
    else:
        info["torch_version"] = _package_version("torch")

    return info
