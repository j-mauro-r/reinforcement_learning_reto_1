"""Tests for HU010 lightweight BattleZone experiment traceability."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pytest
import yaml

from src.experiment import (
    MANIFEST_SCHEMA_VERSION,
    RUN_ID_PATTERN,
    capture_git_lineage,
    capture_hardware,
    capture_runtime,
    create_run_manifest,
    fail_session,
    finish_session,
    generate_run_id,
    load_config_snapshot,
    load_run_manifest,
    start_session,
    validate_long_training_readiness,
    validate_resume_compatibility,
    write_run_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "3_BattleZone/configs/battlezone_config.yaml"
GIT = {"commit": "a" * 40, "branch": "feature/test", "dirty": False}


def make_manifest(tmp_path: Path):
    config, digest = load_config_snapshot(CONFIG_PATH)
    run_id = generate_run_id(git_sha=GIT["commit"])
    return create_run_manifest(
        results_dir=tmp_path / "results",
        manifest_filename="run_manifest.json",
        run_id=run_id,
        config_path=CONFIG_PATH,
        config_snapshot=config,
        config_sha256=digest,
        git=GIT,
        runtime=capture_runtime(),
        hardware=capture_hardware("cpu"),
    )


def test_run_id_format_and_uniqueness():
    now = datetime(2026, 9, 2, 15, 45, tzinfo=timezone.utc)
    first = generate_run_id(git_sha="c7260cf" + "0" * 33, now=now, suffix="a1b2")
    second = generate_run_id(git_sha="c7260cf" + "0" * 33, now=now, suffix="a1b3")
    assert first == "battlezone-dqn-20260902-154500-c7260cf-a1b2"
    assert RUN_ID_PATTERN.fullmatch(first)
    assert first != second and " " not in first


def test_results_directory_manifest_roundtrip_and_collision(tmp_path):
    manifest, path = make_manifest(tmp_path)
    assert path.parent.name == manifest["run_id"]
    assert (path.parent / "summaries").is_dir()
    assert load_run_manifest(path) == manifest
    with pytest.raises(FileExistsError):
        create_run_manifest(
            results_dir=path.parents[1], manifest_filename=path.name,
            run_id=manifest["run_id"], config_path=CONFIG_PATH,
            config_snapshot=manifest["config"]["snapshot"],
            config_sha256=manifest["config"]["sha256"], git=GIT,
            runtime={}, hardware={},
        )


def test_schema_and_corrupt_json_are_rejected(tmp_path):
    manifest, path = make_manifest(tmp_path)
    manifest["schema_version"] = 2
    with pytest.raises(ValueError, match="Unsupported manifest schema"):
        write_run_manifest(path, manifest)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_run_manifest(path)


def test_config_snapshot_and_real_sha256():
    snapshot, digest = load_config_snapshot(CONFIG_PATH)
    assert snapshot["tracking"]["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert snapshot["algorithm"] == "DQN"
    assert digest == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def test_git_runtime_and_hardware_are_real_and_serializable():
    git = capture_git_lineage(REPO_ROOT)
    runtime = capture_runtime()
    hardware = capture_hardware()
    assert re.fullmatch(r"[0-9a-f]{40}", git["commit"])
    assert isinstance(git["dirty"], bool)
    assert runtime["python"] and "tensorboard" in runtime
    assert hardware["device"] in {"cpu", "cuda", "mps"}
    assert isinstance(hardware["cuda_available"], bool)
    json.dumps({"runtime": runtime, "hardware": hardware})


def test_new_and_resume_full_preserve_lineage(tmp_path):
    manifest, path = make_manifest(tmp_path)
    run_id = manifest["run_id"]
    manifest = start_session(
        manifest, mode="new", start_global_step=0, input_checkpoint=None,
        tensorboard_log_dir="logs/shared", device="cpu", replay_restored=None,
        manifest_path=path,
    )
    manifest = finish_session(
        manifest, end_global_step=32, episode_index=1, elapsed_seconds=1.25,
        output_checkpoint="checkpoints/step_32.pt", completed=False,
        manifest_path=path,
    )
    manifest = start_session(
        load_run_manifest(path), mode="resume_full", start_global_step=32,
        input_checkpoint="checkpoints/step_32.pt", tensorboard_log_dir="logs/shared",
        device="cpu", replay_restored=True, manifest_path=path,
    )
    manifest = finish_session(
        manifest, end_global_step=48, episode_index=2, elapsed_seconds=0.75,
        output_checkpoint="checkpoints/step_48.pt", completed=False,
        manifest_path=path,
    )
    loaded = load_run_manifest(path)
    assert loaded["run_id"] == run_id
    assert [s["session_index"] for s in loaded["sessions"]] == [1, 2]
    assert loaded["sessions"][0]["end_global_step"] == loaded["sessions"][1]["start_global_step"] == 32
    assert loaded["sessions"][1]["end_global_step"] == 48
    assert loaded["sessions"][1]["replay_restored"] is True
    assert loaded["sessions"][1]["input_checkpoint"] == "checkpoints/step_32.pt"
    assert loaded["sessions"][1]["output_checkpoint"] == "checkpoints/step_48.pt"
    assert loaded["sessions"][1]["tensorboard_log_dir"] == "logs/shared"


def test_resume_lightweight_metadata_and_continuity_guard(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    manifest = start_session(manifest, mode="new", start_global_step=0, input_checkpoint=None, tensorboard_log_dir="logs/run", device="cpu", replay_restored=None)
    manifest = finish_session(manifest, end_global_step=4, episode_index=0, elapsed_seconds=.1, output_checkpoint="cp4.pt", completed=False)
    with pytest.raises(ValueError, match="discontinuity"):
        start_session(manifest, mode="resume_full", start_global_step=3, input_checkpoint="cp4.pt", tensorboard_log_dir="logs/run", device="cpu", replay_restored=True)
    resumed = start_session(manifest, mode="resume_lightweight", start_global_step=4, input_checkpoint="cp4.pt", tensorboard_log_dir="logs/run", device="cpu", replay_restored=False)
    assert resumed["run_id"] == manifest["run_id"]
    assert resumed["sessions"][-1]["replay_restored"] is False


def test_failed_state_preserves_session(tmp_path):
    manifest, path = make_manifest(tmp_path)
    running = start_session(manifest, mode="new", start_global_step=0, input_checkpoint=None, tensorboard_log_dir="logs/run", device="cpu", replay_restored=None)
    failed = fail_session(running, error=RuntimeError("controlled"), manifest_path=path)
    assert failed["status"] == failed["sessions"][0]["status"] == "failed"
    assert "controlled" in failed["notes"][-1]


def test_atomic_write_failure_preserves_valid_manifest(tmp_path, monkeypatch):
    manifest, path = make_manifest(tmp_path)
    original = path.read_bytes()

    def fail_replace(source, target):
        raise OSError("controlled replace failure")

    monkeypatch.setattr("src.experiment.os.replace", fail_replace)
    manifest["notes"].append("must not replace")
    with pytest.raises(OSError, match="controlled"):
        write_run_manifest(path, manifest)
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


def test_resume_compatibility_rejects_critical_changes(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    snapshot = manifest["config"]["snapshot"]
    validate_resume_compatibility(manifest, snapshot)
    changed = yaml.safe_load(yaml.safe_dump(snapshot))
    changed["environment"]["env_id"] = "ALE/Assault-v5"
    with pytest.raises(ValueError, match="BattleZone"):
        validate_resume_compatibility(manifest, changed)


def test_readiness_pass_and_dirty_strict_failure(tmp_path):
    config, _ = load_config_snapshot(CONFIG_PATH)
    run_id = generate_run_id(git_sha=GIT["commit"])
    passed = validate_long_training_readiness(
        config=config, config_path=CONFIG_PATH, run_id=run_id, git=GIT,
        results_dir=tmp_path / "ready-results",
    )
    assert passed.ready and all(passed.checks.values()) and not passed.errors
    dirty = validate_long_training_readiness(
        config=config, config_path=CONFIG_PATH, run_id=run_id,
        git={**GIT, "dirty": True}, results_dir=tmp_path / "dirty-results",
    )
    assert not dirty.ready
    assert dirty.checks["git_clean_when_required"] is False


def test_readiness_manifest_write_failure(tmp_path):
    config, _ = load_config_snapshot(CONFIG_PATH)
    not_directory = tmp_path / "file"
    not_directory.write_text("x", encoding="utf-8")
    result = validate_long_training_readiness(
        config=config, config_path=CONFIG_PATH,
        run_id=generate_run_id(git_sha=GIT["commit"]), git=GIT,
        results_dir=not_directory,
    )
    assert not result.ready
    assert result.checks["manifest_writable"] is False


def test_scope_has_no_external_tracking_or_assault_dependency():
    source = (REPO_ROOT / "3_BattleZone/src/experiment.py").read_text(encoding="utf-8").lower()
    forbidden_service = "ml" + "flow"
    forbidden_project = "2_" + "assault"
    assert forbidden_service not in source
    assert forbidden_project not in source
