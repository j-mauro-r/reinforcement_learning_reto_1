"""Tests for HU009 training profiles and full-training gates."""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.session_bootstrap import compute_config_fingerprint, prepare_training_session  # noqa: E402
from src.training_profiles import (  # noqa: E402
    DEFAULT_FULL_REPLAY_BUFFER_CAPACITY,
    FULL_TRAINING_TARGET_TIMESTEPS,
    FullTrainingGate,
    assert_training_can_start,
    estimate_replay_buffer_memory,
    evaluate_full_training_ready,
    expected_epsilon_at_step,
    resolve_training_profile,
    validate_replay_buffer_memory,
)
from src.utils import load_yaml_config  # noqa: E402


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


@dataclass(frozen=True)
class FakePreflight:
    ready_for_training: bool


def _config() -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["mlflow"]["enabled"] = False
    return config


def _session_context(tmp_path: Path, profile_context):
    return prepare_training_session(
        base_path=tmp_path,
        project_run_id="hu009_profile_test",
        target_timesteps=profile_context.target_timesteps,
        requested_mode="auto",
        config=profile_context.config,
        checkpoint_root=tmp_path / "checkpoints",
        tensorboard_root=tmp_path / "tensorboard",
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        bootstrap_ref="feature/hu009-full-ddqn-training",
        bootstrap_commit="test-sha",
    )


def test_smoke_profile_selection_preserves_existing_short_contract():
    profile = resolve_training_profile(_config(), "smoke")

    assert profile.name == "smoke"
    assert profile.target_timesteps == 48
    assert profile.config["training"]["total_timesteps"] == 48
    assert profile.config["replay_buffer"]["capacity"] == 1024
    assert profile.config["checkpointing"]["interval_steps"] == 24


def test_full_profile_selection_uses_explicit_full_values():
    profile = resolve_training_profile(_config(), "full")

    assert profile.name == "full"
    assert profile.target_timesteps == FULL_TRAINING_TARGET_TIMESTEPS
    assert profile.config["training"]["total_timesteps"] == FULL_TRAINING_TARGET_TIMESTEPS
    assert profile.config["replay_buffer"]["capacity"] == DEFAULT_FULL_REPLAY_BUFFER_CAPACITY
    assert profile.config["checkpointing"]["save_replay_buffer"] is True
    assert profile.config["training"]["total_timesteps"] != 48


def test_target_timesteps_can_be_overridden_externally():
    profile = resolve_training_profile(_config(), "full", target_timesteps=300_000)

    assert profile.target_timesteps == 300_000
    assert profile.config["training"]["total_timesteps"] == 300_000
    assert profile.config["replay_buffer"]["capacity"] == DEFAULT_FULL_REPLAY_BUFFER_CAPACITY


def test_replay_buffer_memory_estimate_is_deterministic():
    estimate = estimate_replay_buffer_memory(capacity=50_000, state_shape=(4, 84, 84), dtype="uint8")

    assert estimate.state_bytes == 4 * 84 * 84
    assert estimate.total_bytes == 50_000 * ((2 * 4 * 84 * 84) + 8 + 4 + 1)
    assert estimate.total_gib == pytest.approx(2.62912, rel=1e-4)


def test_replay_buffer_memory_rejects_insufficient_ram():
    estimate = estimate_replay_buffer_memory(capacity=50_000)

    with pytest.raises(RuntimeError, match="Insufficient RAM"):
        validate_replay_buffer_memory(estimate, ram_available_gib=4.0, memory_margin_gib=4.0)


def test_full_training_ready_false_without_cuda(tmp_path):
    profile = resolve_training_profile(_config(), "full")
    session_context = _session_context(tmp_path, profile)

    gate = evaluate_full_training_ready(
        profile,
        session_context,
        FakePreflight(True),
        {"ram_available_gb": 12.0},
        observation_shape=(4, 84, 84),
        observation_dtype="uint8",
        action_space="Discrete(7)",
        runtime="Google Colab",
        device="cpu",
    )

    assert gate.ready is False
    assert "device_not_cuda" in gate.issues


def test_full_training_ready_true_with_valid_simulated_colab_conditions(tmp_path):
    profile = resolve_training_profile(_config(), "full")
    session_context = _session_context(tmp_path, profile)

    gate = evaluate_full_training_ready(
        profile,
        session_context,
        FakePreflight(True),
        {"ram_available_gb": 12.0},
        observation_shape=(4, 84, 84),
        observation_dtype="uint8",
        action_space="Discrete(7)",
        runtime="Google Colab",
        device="cuda",
    )

    assert gate.ready is True
    assert gate.issues == []
    assert gate.ram_margin_gib == pytest.approx(12.0 - profile.replay_buffer_memory.total_gib)


def test_resume_epsilon_is_computed_from_global_step():
    profile = resolve_training_profile(_config(), "full")

    restored = expected_epsilon_at_step(profile.config, 100_000)
    initial = expected_epsilon_at_step(profile.config, 0)

    assert restored < initial
    assert restored == pytest.approx(0.505)


def test_profile_change_changes_fingerprint_and_blocks_resume_continuity():
    smoke = resolve_training_profile(_config(), "smoke")
    full = resolve_training_profile(_config(), "full")

    assert smoke.config["replay_buffer"]["capacity"] != full.config["replay_buffer"]["capacity"]
    assert smoke.config["training"]["epsilon_decay_steps"] != full.config["training"]["epsilon_decay_steps"]
    assert compute_config_fingerprint(smoke.config) != compute_config_fingerprint(full.config)


def test_learning_frequency_change_changes_fingerprint():
    profile = resolve_training_profile(_config(), "full")
    changed = resolve_training_profile(_config(), "full")
    changed.config["training"]["train_frequency"] = 8

    assert compute_config_fingerprint(profile.config) != compute_config_fingerprint(changed.config)


def test_full_profile_checkpointing_keeps_resume_full_capability():
    profile = resolve_training_profile(_config(), "full")

    assert profile.config["checkpointing"]["save_replay_buffer"] is True
    assert profile.config["checkpointing"]["interval_steps"] == 25_000


def test_invalid_profile_fails_fast():
    with pytest.raises(ValueError, match="smoke' or 'full"):
        resolve_training_profile(_config(), "debug")


def test_full_gate_rejects_smoke_profile(tmp_path):
    profile = resolve_training_profile(_config(), "smoke")
    session_context = _session_context(tmp_path, profile)

    gate = evaluate_full_training_ready(
        profile,
        session_context,
        FakePreflight(True),
        {"ram_available_gb": 12.0},
        observation_shape=(4, 84, 84),
        observation_dtype="uint8",
        action_space="Discrete(7)",
        runtime="Google Colab",
        device="cuda",
    )

    assert gate.ready is False
    assert "profile_not_full" in gate.issues


def test_full_invalid_gate_aborts_before_mlflow_trainer_or_manifest_side_effects(tmp_path):
    profile = resolve_training_profile(_config(), "full")
    gate = _gate(profile, ready=False, issues=["device_not_cuda"])
    side_effects = []
    manifest = tmp_path / "experiments_manifest.json"

    with pytest.raises(RuntimeError, match="FULL_TRAINING_READY=False"):
        assert_training_can_start(profile, gate)
        side_effects.append("mlflow.start_run")
        side_effects.append("run_training_session")
        manifest.write_text("modified", encoding="utf-8")

    assert side_effects == []
    assert not manifest.exists()


def test_full_valid_gate_allows_mlflow_and_training_boundary():
    profile = resolve_training_profile(_config(), "full")
    gate = _gate(profile, ready=True, issues=[])
    side_effects = []

    assert_training_can_start(profile, gate)
    side_effects.append("mlflow.start_run")
    side_effects.append("run_training_session")

    assert side_effects == ["mlflow.start_run", "run_training_session"]


def test_smoke_profile_is_not_blocked_by_false_full_gate():
    profile = resolve_training_profile(_config(), "smoke")
    gate = _gate(profile, ready=False, issues=["profile_not_full", "device_not_cuda"])
    side_effects = []

    assert_training_can_start(profile, gate)
    side_effects.append("smoke_session_continue")

    assert side_effects == ["smoke_session_continue"]


def test_no_regression_hu008b_prepare_auto_new(tmp_path):
    profile = resolve_training_profile(_config(), "smoke")
    context = _session_context(tmp_path, profile)

    assert context.tracking_mode == "new"
    assert context.tracking_session_id == "session_001"
    assert context.checkpoint_input is None


def test_notebook_uses_explicit_training_profile_without_historical_full_target():
    source = (ASSAULT_DIR / "assault_ddqn.ipynb").read_text(encoding="utf-8")

    assert "ASSAULT_TRAINING_PROFILE" in source
    assert "resolve_training_profile" in source
    assert "FULL_TRAINING_READY" in source
    assert "assault_ddqn_full_001" not in source
    assert "250000" not in source


def test_notebook_full_gate_runs_before_mlflow_training_or_manifest_side_effects():
    notebook = json.loads((ASSAULT_DIR / "assault_ddqn.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    source = "\n".join(cells)
    gate_cell_index = next(
        index for index, cell in enumerate(cells) if "assert_training_can_start(profile_context, full_training_gate)" in cell
    )
    side_effect_cell_index = next(index for index, cell in enumerate(cells) if "mlflow_tracker.start_run(" in cell)
    side_effect_cell = cells[side_effect_cell_index]

    assert gate_cell_index < side_effect_cell_index
    assert "MLflowTracker.from_config" in side_effect_cell
    assert "run_training_session(" in side_effect_cell
    assert "update_experiment_state_after_success(" in side_effect_cell
    assert 'profile_context.name == "full" and not FULL_TRAINING_READY' not in source


def _gate(profile, ready: bool, issues: list[str]) -> FullTrainingGate:
    return FullTrainingGate(
        ready=ready,
        issues=issues,
        profile=profile.name,
        device="cuda" if ready else "cpu",
        runtime="Google Colab",
        replay_buffer_memory=profile.replay_buffer_memory,
        ram_available_gib=12.0,
        ram_margin_gib=12.0 - profile.replay_buffer_memory.total_gib,
    )
