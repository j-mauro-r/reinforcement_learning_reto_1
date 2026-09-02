"""Fast tests for the HU011 reference training pipeline."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.environment import load_config
from src.experiment import generate_run_id, load_config_snapshot
from src.training_run import (
    CheckpointDecision,
    build_artifact_paths,
    checkpoint_decision,
    estimate_replay_memory,
    is_training_complete,
    resolve_long_training_config,
    run_hu011_preflight,
    run_training_session,
    validate_memory_readiness,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "3_BattleZone/configs/battlezone_config.yaml"
GIT = {"commit": "a" * 40, "branch": "test", "dirty": False}


class ContinuousFakeEnv:
    def __init__(self):
        self.cursor = 0
        self.reset_calls = 0
        self.closed = 0
        self.action_space = type("ActionSpace", (), {"seed": lambda self, seed: None})()

    def reset(self, seed=None):
        self.cursor = 0
        self.reset_calls += 1
        return np.zeros((4, 128, 128, 3), dtype=np.uint8), {}

    def step(self, action):
        self.cursor += 1
        terminated = self.cursor == 6
        return np.zeros((4, 128, 128, 3), dtype=np.uint8), 1.0, terminated, False, {}

    def close(self):
        self.closed += 1


def test_reference_v1_resolution_preserves_smoke():
    base = load_config(CONFIG_PATH)
    original = deepcopy(base)
    effective = resolve_long_training_config(base)
    assert base == original
    assert base["smoke"]["total_timesteps_new"] == 32
    assert effective["long_training"]["profile"] == "reference_v1"
    assert effective["training"]["total_timesteps"] == 1_000_000
    assert effective["dqn"]["batch_size"] == 32
    assert effective["dqn"]["replay_buffer"]["capacity"] == 4096
    assert effective["training"]["epsilon"] == {
        "start": 1.0, "end": 0.05, "decay_steps": 250000,
    }


def test_memory_estimate_matches_concrete_replay_layout():
    estimate = estimate_replay_memory(
        state_shape=(4, 128, 128, 3), capacity=4096, available_ram_gib=32.0,
    )
    assert estimate.bytes_per_state == 196_608
    assert estimate.bytes_per_transition == 393_229
    assert estimate.estimated_replay_bytes == 1_610_665_984
    assert estimate.estimated_replay_gib == pytest.approx(1.500050)
    assert validate_memory_readiness(estimate).ready
    assert validate_memory_readiness(estimate).full_checkpoint_ready


def test_memory_gate_rejects_unknown_or_unsafe_ram():
    unknown = estimate_replay_memory(state_shape=(4, 128, 128, 3), capacity=4096)
    assert not validate_memory_readiness(unknown).ready
    unsafe = estimate_replay_memory(
        state_shape=(4, 128, 128, 3), capacity=4096, available_ram_gib=2.0,
    )
    result = validate_memory_readiness(unsafe)
    assert not result.ready and not result.full_checkpoint_ready


@pytest.mark.parametrize(
    ("step", "full_ready", "expected"),
    [
        (1, True, CheckpointDecision.NONE),
        (25_000, True, CheckpointDecision.LIGHTWEIGHT),
        (250_000, True, CheckpointDecision.FULL),
        (500_000, True, CheckpointDecision.FULL),
        (750_000, True, CheckpointDecision.FULL),
        (1_000_000, True, CheckpointDecision.FULL),
        (250_000, False, CheckpointDecision.LIGHTWEIGHT),
    ],
)
def test_checkpoint_policy(step, full_ready, expected):
    assert checkpoint_decision(step, load_config(CONFIG_PATH), full_checkpoint_ready=full_ready) is expected


def test_global_target_completion_semantics():
    assert not is_training_complete(999_999, 1_000_000)
    assert is_training_complete(1_000_000, 1_000_000)
    assert is_training_complete(1_000_001, 1_000_000)


def test_local_session_limit_does_not_change_reference_target():
    config = load_config(CONFIG_PATH)
    assert config["long_training"]["target_global_step"] == 1_000_000
    assert not is_training_complete(160, config["long_training"]["target_global_step"])


def test_explicit_run_paths_are_isolated(tmp_path):
    run_id = generate_run_id(git_sha=GIT["commit"])
    paths = build_artifact_paths(tmp_path, run_id)
    assert paths["manifest"] == tmp_path / "results" / run_id / "run_manifest.json"
    assert paths["final_model"].name == "battlezone_dqn_final.pt"
    with pytest.raises(ValueError, match="run_id"):
        build_artifact_paths(tmp_path, "")


def test_cuda_required_fails_and_local_override_passes(tmp_path):
    config, _ = load_config_snapshot(CONFIG_PATH)
    run_id = generate_run_id(git_sha=GIT["commit"])
    blocked = run_hu011_preflight(
        base_config=config, config_path=CONFIG_PATH, run_id=run_id, git=GIT,
        persistent_root=tmp_path / "blocked", ram_gib=32.0, cuda_available=False,
    )
    assert not blocked.ready
    assert "LONG_TRAINING_BLOCKED_NO_CUDA" in blocked.errors
    local = deepcopy(config)
    local["long_training"]["require_accelerator"] = False
    passed = run_hu011_preflight(
        base_config=local, config_path=CONFIG_PATH, run_id=run_id, git=GIT,
        persistent_root=tmp_path / "local", ram_gib=32.0, cuda_available=False,
    )
    assert passed.ready


def test_resume_requires_explicit_run_and_checkpoint(tmp_path):
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="explicit run_id"):
        run_training_session(
            base_config=config, config_path=CONFIG_PATH, persistent_root=tmp_path,
            mode="resume_lightweight", repo_root=ROOT,
            require_accelerator_override=False, target_global_step_override=4,
        )


def test_local_override_cannot_hide_a_long_local_run(tmp_path):
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="limited"):
        run_training_session(
            base_config=config, config_path=CONFIG_PATH, persistent_root=tmp_path,
            mode="new", repo_root=ROOT, require_accelerator_override=False,
            target_global_step_override=5000,
        )


def test_scope_remains_dqn_only_without_external_tracking():
    source = (ROOT / "3_BattleZone/src/training_run.py").read_text(encoding="utf-8").lower()
    assert "2_" + "assault" not in source
    assert "ml" + "flow" not in source
    config = load_config(CONFIG_PATH)
    assert config["algorithm"] == "DQN"


def test_multiple_internal_checkpoints_keep_one_session_and_resume_adds_one(tmp_path, monkeypatch):
    config = load_config(CONFIG_PATH)
    config["long_training"]["dqn"]["replay_buffer_capacity"] = 16
    config["long_training"]["dqn"]["batch_size"] = 2
    config["long_training"]["training"]["learning_starts"] = 100
    config["long_training"]["checkpointing"]["interval_steps"] = 4
    config["long_training"]["checkpointing"]["full_milestone_interval_steps"] = 100
    config["long_training"]["tensorboard"]["scalar_log_interval_steps"] = 1
    environments = []

    def make_env(*args, **kwargs):
        env = ContinuousFakeEnv()
        environments.append(env)
        return env

    monkeypatch.setattr("src.training_run.capture_git_lineage", lambda root: GIT)
    monkeypatch.setattr(
        "src.training_run.capture_hardware",
        lambda: {"device": "cpu", "ram_gb": 32.0},
    )
    monkeypatch.setattr("src.training_run.create_battlezone_env", make_env)
    first = run_training_session(
        base_config=config, config_path=CONFIG_PATH, persistent_root=tmp_path,
        mode="new", repo_root=ROOT, require_accelerator_override=False,
        target_global_step_override=12,
    )
    assert [path.name for path in first["checkpoints"]] == [
        "battlezone_dqn_step_00000004_lightweight.pt",
        "battlezone_dqn_step_00000008_lightweight.pt",
        "battlezone_dqn_step_00000012_lightweight.pt",
    ]
    assert len(first["manifest"]["sessions"]) == 1
    assert first["summary"].episode_lengths == [6, 6]
    assert first["summary"].episode_rewards == pytest.approx([6.0, 6.0])
    assert environments[0].reset_calls == 3
    assert first["manifest"]["sessions"][0]["elapsed_seconds"] > 0
    events = EventAccumulator(str(first["paths"]["logs"]))
    events.Reload()
    scalar_steps = [event.step for event in events.Scalars("train/epsilon")]
    assert 3 in scalar_steps and 5 in scalar_steps
    assert scalar_steps == sorted(scalar_steps)

    resumed = run_training_session(
        base_config=config, config_path=CONFIG_PATH, persistent_root=tmp_path,
        mode="resume_lightweight", run_id=first["run_id"],
        checkpoint_path=first["checkpoints"][-1], repo_root=ROOT,
        require_accelerator_override=False, target_global_step_override=16,
    )
    assert resumed["run_id"] == first["run_id"]
    assert len(resumed["manifest"]["sessions"]) == 2
    assert resumed["manifest"]["sessions"][1]["start_global_step"] == 12
