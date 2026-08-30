"""Tests for HU009C TensorBoard reporting helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.reporting import (
    MAX_TRAINING_FIGURES,
    TrainingFigureSpec,
    prepare_training_figures_from_scalars,
    validate_training_figure_specs,
)


def _scalars() -> dict[str, list[tuple[int, float]]]:
    return {
        "episode/reward": [(10, 1.0), (20, 3.0), (30, 5.0)],
        "train/loss": [(4, 2.0), (8, 4.0), (12, 6.0)],
        "train/q_mean": [(4, 0.5), (8, 1.0), (12, 1.5)],
        "train/epsilon": [(4, 1.0), (8, 0.5), (12, 0.1)],
    }


def test_prepare_training_figures_keeps_global_step_and_moving_average():
    figures = prepare_training_figures_from_scalars(_scalars(), reward_window=2)
    by_id = {figure.figure_id: figure for figure in figures}

    assert len(figures) == MAX_TRAINING_FIGURES
    assert by_id["reward"].series["episode/reward"].steps == [10, 20, 30]
    assert by_id["reward"].series["episode/reward_mean"].values == pytest.approx([1.0, 2.0, 4.0])
    assert by_id["loss"].series["train/loss"].steps == [4, 8, 12]
    assert by_id["q_mean_epsilon"].secondary_y_tags == ("train/epsilon",)
    assert set(by_id["q_mean_epsilon"].series) == {"train/q_mean", "train/epsilon"}


def test_existing_reward_mean_tag_is_used_without_recomputing():
    scalars = _scalars()
    scalars["episode/reward_mean"] = [(10, 9.0), (20, 8.0), (30, 7.0)]

    reward_figure = prepare_training_figures_from_scalars(scalars)[0]

    assert reward_figure.series["episode/reward_mean"].values == [9.0, 8.0, 7.0]


def test_missing_tags_nan_inf_duplicate_steps_and_negative_steps_fail():
    scalars = _scalars()
    del scalars["train/loss"]
    with pytest.raises(ValueError, match="Missing required"):
        prepare_training_figures_from_scalars(scalars)

    bad = _scalars()
    bad["train/loss"] = [(1, math.nan)]
    with pytest.raises(ValueError, match="NaN or Inf"):
        prepare_training_figures_from_scalars(bad)

    bad = _scalars()
    bad["train/loss"] = [(1, 1.0), (1, 2.0)]
    with pytest.raises(ValueError, match="duplicate"):
        prepare_training_figures_from_scalars(bad)

    bad = _scalars()
    bad["train/loss"] = [(-1, 1.0)]
    with pytest.raises(ValueError, match="negative"):
        prepare_training_figures_from_scalars(bad)


def test_figure_limit_and_structural_redundancy_are_rejected():
    figures = prepare_training_figures_from_scalars(_scalars())
    extra = TrainingFigureSpec(
        figure_id="extra",
        title="Extra",
        tags=("train/learning_rate",),
        x_label="global_step",
        y_labels=("learning_rate",),
        series={},
    )
    with pytest.raises(ValueError, match="at most"):
        validate_training_figure_specs([*figures, extra])
    copy_kwargs = dict(figures[0].__dict__)
    copy_kwargs["figure_id"] = "copy"
    with pytest.raises(ValueError, match="redundant"):
        validate_training_figure_specs([figures[0], TrainingFigureSpec(**copy_kwargs)])
