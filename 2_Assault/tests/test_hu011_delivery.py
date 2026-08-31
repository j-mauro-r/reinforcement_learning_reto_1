"""Tests for HU011 final Assault delivery evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluator import EvaluationSummary
from src.hu011_delivery import (
    MANDATORY_ASSAULT_CRITERIA,
    build_delivery_gate,
    build_evaluation_artifact,
    compare_with_random_baseline,
    criteria_from_statuses,
    evaluation_records,
    load_random_baseline,
    validate_exploitation_figure_data,
    write_json_atomic,
)
from src.model_artifact import ModelArtifactInfo

ASSAULT_DIR = Path(__file__).resolve().parents[1]
BASELINE_PATH = ASSAULT_DIR / "data" / "baseline_random_assault.json"


def _evaluation() -> EvaluationSummary:
    rewards = [588.0, 546.0, 609.0, 546.0, 483.0, 357.0, 567.0, 546.0, 672.0, 546.0]
    return EvaluationSummary(
        episodes=10,
        rewards=rewards,
        mean_reward=546.0,
        median_reward=546.0,
        std_reward=78.57480511715733,
        min_reward=357.0,
        max_reward=672.0,
        episode_lengths=[700, 710, 720, 730, 740, 750, 760, 770, 780, 790],
        epsilon=0.0,
        terminated_episodes=10,
        truncated_episodes=0,
    )


def _model_info(tmp_path: Path) -> ModelArtifactInfo:
    model_path = tmp_path / "assault_ddqn_model.pt"
    model_path.write_bytes(b"compact-model")
    return ModelArtifactInfo(
        path=model_path,
        sha256="a" * 64,
        size_bytes=model_path.stat().st_size,
        metadata={
            "project_run_id": "assault_ddqn_full_001",
            "source_checkpoint_step": 250000,
            "environment": {"id": "ALE/Assault-v5"},
        },
    )


def test_versioned_random_baseline_matches_hu001_rewards_and_normalized_stats():
    baseline = load_random_baseline(BASELINE_PATH)
    assert baseline.episodes == 10
    assert baseline.rewards == [189.0, 273.0, 273.0, 210.0, 210.0, 399.0, 315.0, 189.0, 189.0, 378.0]
    assert baseline.mean_reward == pytest.approx(262.5)
    assert baseline.median_reward == pytest.approx(241.5)
    assert baseline.std_reward == pytest.approx(75.27848298152666)
    assert baseline.min_reward == 189.0
    assert baseline.max_reward == 399.0


def test_final_evaluation_compares_above_random_baseline():
    comparison = compare_with_random_baseline(_evaluation(), load_random_baseline(BASELINE_PATH))
    assert comparison.agent_beats_random is True
    assert comparison.absolute_improvement == pytest.approx(283.5)
    assert comparison.relative_improvement_pct == pytest.approx(108.0)


def test_evaluation_records_are_aligned_and_seeded():
    rows = evaluation_records(_evaluation(), base_seed=20042)
    assert len(rows) == 10
    assert rows[0] == {"episode": 1, "seed": 20042, "reward": 588.0, "length": 700}
    assert rows[-1]["seed"] == 20051
    assert rows[-1]["reward"] == 546.0


def test_build_evaluation_artifact_links_model_and_protocol(tmp_path):
    payload = build_evaluation_artifact(
        evaluation=_evaluation(),
        model_info=_model_info(tmp_path),
        project_run_id="assault_ddqn_full_001",
        source_checkpoint_path=tmp_path / "checkpoint_step_250000.pt",
        base_seed=20042,
        git_sha="deadbeef",
    )
    assert payload["model"]["sha256"] == "a" * 64
    assert payload["model"]["source_checkpoint_step"] == 250000
    assert payload["protocol"]["episodes"] == 10
    assert payload["protocol"]["epsilon"] == 0.0
    assert payload["protocol"]["training_updates_during_evaluation"] is False
    assert payload["statistics"]["mean_reward"] == 546.0
    assert len(payload["episodes"]) == 10


def test_final_evaluation_rejects_exploration(tmp_path):
    evaluation = _evaluation()
    invalid = EvaluationSummary(**{**evaluation.__dict__, "epsilon": 0.01})
    with pytest.raises(ValueError, match="epsilon=0.0"):
        build_evaluation_artifact(invalid, _model_info(tmp_path), "assault_ddqn_full_001", tmp_path / "checkpoint.pt", 20042)


def test_atomic_evaluation_json_round_trip(tmp_path):
    path = write_json_atomic(tmp_path / "evaluation" / "final.json", {"ready": True, "episodes": 10})
    assert path.exists()
    assert '"ready": true' in path.read_text(encoding="utf-8")
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_exploitation_figure_validation_uses_exact_final_rewards():
    evaluation = _evaluation()
    assert validate_exploitation_figure_data(evaluation, evaluation.rewards) is True
    assert validate_exploitation_figure_data(evaluation, list(reversed(evaluation.rewards))) is False


def test_delivery_gate_requires_all_assault_criteria_pass():
    statuses = {criterion_id: True for criterion_id in MANDATORY_ASSAULT_CRITERIA}
    gate = build_delivery_gate(criteria_from_statuses(statuses), global_multi_algorithm="PENDING")
    assert gate.assault_method_allowed is True
    assert gate.final_delivery_gate is True
    assert gate.global_multi_algorithm == "PENDING"
    assert gate.as_dict()["HU011_FINAL_DELIVERY_GATE"] == "PASS"


def test_delivery_gate_fails_when_one_mandatory_criterion_fails():
    statuses = {criterion_id: True for criterion_id in MANDATORY_ASSAULT_CRITERIA}
    statuses["CA17"] = False
    gate = build_delivery_gate(criteria_from_statuses(statuses), global_multi_algorithm="PASS")
    assert gate.final_delivery_gate is False
    assert gate.as_dict()["HU011_FINAL_DELIVERY_GATE"] == "FAIL"


def test_delivery_gate_rejects_missing_criterion():
    statuses = {criterion_id: True for criterion_id in MANDATORY_ASSAULT_CRITERIA if criterion_id != "CA21"}
    with pytest.raises(ValueError, match="missing mandatory criteria"):
        criteria_from_statuses(statuses)
