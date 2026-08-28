"""Tests for HU004 preflight integration checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.preflight import PreflightReport, run_preflight_checks
from src.utils import load_yaml_config


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


class BrokenEnv:
    """Fake environment that violates the observation contract."""

    def reset(self, seed=None):
        return np.zeros((84, 84), dtype=np.uint8), {}

    def close(self) -> None:
        pass


def _config() -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["replay_buffer"] = {"capacity": 64, "batch_size": 4}
    return config


def test_preflight_passes_with_real_assault_cpu():
    report = run_preflight_checks(_config(), device="cpu")

    assert isinstance(report, PreflightReport)
    assert report.passed is True
    assert report.ready_for_training is True
    assert report.runtime in {"local", "Google Colab"}
    assert report.device == "cpu"
    assert all(report.checks.values())
    assert "READY_FOR_TRAINING=True" in report.format_summary()
    assert report.details["Save/load"] == "temporary_file_cleaned=True"


def test_preflight_material_failure_returns_failed_report():
    report = run_preflight_checks(_config(), device="cpu", env_factory=lambda config: BrokenEnv())

    assert report.passed is False
    assert report.ready_for_training is False
    assert report.checks["Observation"] is False
    assert report.errors
    assert "READY_FOR_TRAINING=False" in report.format_summary()


def test_preflight_result_is_structured():
    report = run_preflight_checks(_config(), device="cpu")
    data = report.as_dict()

    assert data["passed"] is True
    assert data["ready_for_training"] is True
    assert isinstance(data["checks"], dict)
    assert isinstance(data["errors"], list)
    assert isinstance(data["details"], dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this runtime.")
def test_preflight_uses_cuda_when_available():
    report = run_preflight_checks(_config(), device="cuda")

    assert report.passed is True
    assert report.device == "cuda"
