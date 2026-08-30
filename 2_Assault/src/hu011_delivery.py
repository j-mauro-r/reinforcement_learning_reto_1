"""Final Assault delivery helpers for HU011.

This module keeps final-evaluation evidence, exploitation plotting, baseline
comparison and the academic delivery gate outside the notebook so the notebook
remains an orchestrator/report instead of duplicating domain logic.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .evaluator import EvaluationSummary
from .model_artifact import ModelArtifactInfo


MANDATORY_ASSAULT_CRITERIA = tuple(f"CA{index:02d}" for index in range(1, 27))
GLOBAL_CRITERION_ID = "CA27"
VALID_GLOBAL_MULTI_ALGORITHM_STATES = {"PASS", "PENDING"}


@dataclass(frozen=True)
class RandomBaselineSummary:
    source: str
    episodes: int
    seeds: list[int]
    rewards: list[float]
    episode_lengths: list[int]
    mean_reward: float
    median_reward: float
    std_reward: float
    min_reward: float
    max_reward: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "episodes": self.episodes,
            "seeds": list(self.seeds),
            "rewards": list(self.rewards),
            "episode_lengths": list(self.episode_lengths),
            "mean_reward": self.mean_reward,
            "median_reward": self.median_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
        }


@dataclass(frozen=True)
class BaselineComparison:
    agent_mean_reward: float
    random_mean_reward: float
    absolute_improvement: float
    relative_improvement_pct: float
    agent_beats_random: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_mean_reward": self.agent_mean_reward,
            "random_mean_reward": self.random_mean_reward,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement_pct": self.relative_improvement_pct,
            "agent_beats_random": self.agent_beats_random,
        }


@dataclass(frozen=True)
class DeliveryCriterion:
    criterion_id: str
    criterion: str
    evidence: str
    validation_method: str
    status: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "id": self.criterion_id,
            "criterion": self.criterion,
            "evidence": self.evidence,
            "validation_method": self.validation_method,
            "status": self.status,
        }


@dataclass(frozen=True)
class DeliveryGate:
    criteria: list[DeliveryCriterion]
    global_multi_algorithm: str
    assault_method_allowed: bool
    final_delivery_gate: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "criteria": [criterion.as_dict() for criterion in self.criteria],
            "GLOBAL_RETO_MULTI_ALGORITHM": self.global_multi_algorithm,
            "ASSAULT_METHOD_ALLOWED": "PASS" if self.assault_method_allowed else "FAIL",
            "HU011_FINAL_DELIVERY_GATE": "PASS" if self.final_delivery_gate else "FAIL",
        }


def load_random_baseline(path: str | Path) -> RandomBaselineSummary:
    """Loads the versioned HU001 random-policy baseline."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rewards = [float(value) for value in payload.get("rewards", [])]
    seeds = [int(value) for value in payload.get("seeds", [])]
    lengths = [int(value) for value in payload.get("episode_lengths", [])]
    episodes = int(payload.get("episodes", 0))
    if episodes < 10 or len(rewards) != episodes or len(seeds) != episodes:
        raise ValueError("Random baseline must contain at least 10 aligned episodes/rewards/seeds.")
    if lengths and len(lengths) != episodes:
        raise ValueError("Random baseline episode_lengths must align with episodes.")
    if any(not math.isfinite(value) for value in rewards):
        raise ValueError("Random baseline rewards contain NaN or Inf.")

    calculated = np.asarray(rewards, dtype=np.float64)
    statistics = payload.get("statistics", {})
    expected = {
        "mean_reward": float(np.mean(calculated)),
        "median_reward": float(np.median(calculated)),
        "std_reward": float(np.std(calculated)),
        "min_reward": float(np.min(calculated)),
        "max_reward": float(np.max(calculated)),
    }
    for key, value in expected.items():
        recorded = float(statistics.get(key, value))
        if not math.isclose(recorded, value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Random baseline statistic {key} does not match rewards: {recorded} != {value}.")

    return RandomBaselineSummary(
        source=str(payload.get("source", path)),
        episodes=episodes,
        seeds=seeds,
        rewards=rewards,
        episode_lengths=lengths,
        **expected,
    )


def compare_with_random_baseline(
    evaluation: EvaluationSummary,
    baseline: RandomBaselineSummary,
) -> BaselineComparison:
    """Compares the final compact-model evaluation with HU001 baseline."""
    if evaluation.episodes < 10:
        raise ValueError("Final evaluation must contain at least 10 episodes.")
    if not math.isclose(float(evaluation.epsilon), 0.0, abs_tol=1e-12):
        raise ValueError("Final Assault comparison requires epsilon=0.0.")
    if baseline.episodes < 10:
        raise ValueError("Random baseline must contain at least 10 episodes.")
    difference = float(evaluation.mean_reward - baseline.mean_reward)
    relative = float((difference / baseline.mean_reward) * 100.0) if baseline.mean_reward else math.inf
    return BaselineComparison(
        agent_mean_reward=float(evaluation.mean_reward),
        random_mean_reward=float(baseline.mean_reward),
        absolute_improvement=difference,
        relative_improvement_pct=relative,
        agent_beats_random=bool(difference > 0.0),
    )


def evaluation_records(
    evaluation: EvaluationSummary,
    base_seed: int,
) -> list[Dict[str, Any]]:
    """Returns one auditable row per final evaluation episode."""
    if evaluation.episodes < 10:
        raise ValueError("HU011 final evaluation requires at least 10 episodes.")
    if len(evaluation.rewards) != evaluation.episodes or len(evaluation.episode_lengths) != evaluation.episodes:
        raise ValueError("Evaluation rewards/lengths must align with episode count.")
    return [
        {
            "episode": index + 1,
            "seed": int(base_seed) + index,
            "reward": float(evaluation.rewards[index]),
            "length": int(evaluation.episode_lengths[index]),
        }
        for index in range(evaluation.episodes)
    ]


def build_evaluation_artifact(
    evaluation: EvaluationSummary,
    model_info: ModelArtifactInfo,
    project_run_id: str,
    source_checkpoint_path: str | Path,
    base_seed: int,
    git_sha: str | None = None,
) -> Dict[str, Any]:
    """Builds the machine-readable final evaluation artifact."""
    if model_info.metadata.get("project_run_id") != project_run_id:
        raise ValueError("Evaluation project_run_id does not match compact model metadata.")
    if evaluation.episodes < 10:
        raise ValueError("Final evaluation must contain at least 10 episodes.")
    if not math.isclose(float(evaluation.epsilon), 0.0, abs_tol=1e-12):
        raise ValueError("Final evaluation must use epsilon=0.0.")
    if any(not math.isfinite(float(value)) for value in evaluation.rewards):
        raise ValueError("Final evaluation contains NaN or Inf rewards.")

    return {
        "schema_version": 1,
        "project_run_id": project_run_id,
        "model": {
            "path": str(model_info.path),
            "sha256": model_info.sha256,
            "size_bytes": int(model_info.size_bytes),
            "source_checkpoint_step": int(model_info.metadata["source_checkpoint_step"]),
            "source_checkpoint_path": str(source_checkpoint_path),
        },
        "protocol": {
            "environment_id": model_info.metadata["environment"]["id"],
            "epsilon": float(evaluation.epsilon),
            "episodes": int(evaluation.episodes),
            "base_seed": int(base_seed),
            "rewards": "raw_environment_rewards",
            "training_updates_during_evaluation": False,
        },
        "episodes": evaluation_records(evaluation, base_seed=base_seed),
        "statistics": {
            "mean_reward": float(evaluation.mean_reward),
            "median_reward": float(evaluation.median_reward),
            "std_reward": float(evaluation.std_reward),
            "min_reward": float(evaluation.min_reward),
            "max_reward": float(evaluation.max_reward),
        },
        "git_sha": git_sha,
    }


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Writes a JSON evidence artifact atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def plot_exploitation_rewards(evaluation: EvaluationSummary) -> Any:
    """Plots reward per final evaluation episode plus the evaluation mean."""
    if evaluation.episodes < 10:
        raise ValueError("Exploitation figure requires at least 10 evaluation episodes.")
    import matplotlib.pyplot as plt

    episodes = np.arange(1, evaluation.episodes + 1)
    rewards = np.asarray(evaluation.rewards, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(episodes, rewards, marker="o", label="reward por episodio")
    axis.axhline(float(evaluation.mean_reward), linestyle="--", label=f"media={evaluation.mean_reward:.2f}")
    axis.set_title("Recompensa durante explotación / evaluación final")
    axis.set_xlabel("episodio de evaluación")
    axis.set_ylabel("reward raw")
    axis.set_xticks(episodes)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def validate_exploitation_figure_data(evaluation: EvaluationSummary, plotted_rewards: Sequence[float]) -> bool:
    """Checks that plotted exploitation rewards are exactly the final rewards."""
    left = np.asarray(evaluation.rewards, dtype=np.float64)
    right = np.asarray(list(plotted_rewards), dtype=np.float64)
    return bool(left.shape == right.shape and np.array_equal(left, right))


def build_delivery_gate(
    criteria: Iterable[DeliveryCriterion],
    global_multi_algorithm: str = "PENDING",
) -> DeliveryGate:
    """Validates the CA01-CA26 gate while keeping CA27 global/informative."""
    global_state = str(global_multi_algorithm).strip().upper()
    if global_state not in VALID_GLOBAL_MULTI_ALGORITHM_STATES:
        raise ValueError("global_multi_algorithm must be PASS or PENDING.")
    rows = list(criteria)
    by_id = {row.criterion_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Delivery criterion ids must be unique.")
    missing = sorted(set(MANDATORY_ASSAULT_CRITERIA) - set(by_id))
    extra = sorted(set(by_id) - set(MANDATORY_ASSAULT_CRITERIA) - {GLOBAL_CRITERION_ID})
    if missing:
        raise ValueError(f"Missing mandatory Assault criteria: {missing}")
    if extra:
        raise ValueError(f"Unknown delivery criteria: {extra}")
    mandatory_pass = all(by_id[criterion_id].status == "PASS" for criterion_id in MANDATORY_ASSAULT_CRITERIA)
    assault_method_allowed = by_id["CA01"].status == "PASS"
    return DeliveryGate(
        criteria=rows,
        global_multi_algorithm=global_state,
        assault_method_allowed=assault_method_allowed,
        final_delivery_gate=bool(mandatory_pass and assault_method_allowed),
    )


def criteria_from_statuses(
    statuses: Mapping[str, bool],
    evidence: Mapping[str, str] | None = None,
    validation_methods: Mapping[str, str] | None = None,
) -> list[DeliveryCriterion]:
    """Creates CA rows from explicit booleans without silently filling gaps."""
    evidence = dict(evidence or {})
    validation_methods = dict(validation_methods or {})
    missing = sorted(set(MANDATORY_ASSAULT_CRITERIA) - set(statuses))
    if missing:
        raise ValueError(f"Statuses missing mandatory criteria: {missing}")
    return [
        DeliveryCriterion(
            criterion_id=criterion_id,
            criterion=criterion_id,
            evidence=evidence.get(criterion_id, "evidencia calculada en notebook HU011"),
            validation_method=validation_methods.get(criterion_id, "gate programático HU011"),
            status="PASS" if bool(statuses[criterion_id]) else "FAIL",
        )
        for criterion_id in MANDATORY_ASSAULT_CRITERIA
    ]
