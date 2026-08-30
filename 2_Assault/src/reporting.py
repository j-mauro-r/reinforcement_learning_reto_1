"""TensorBoard reporting helpers for HU009C delivery artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from .callbacks import load_tensorboard_scalars


REQUIRED_TRAINING_TAGS = ("episode/reward", "train/loss", "train/q_mean", "train/epsilon")
MAX_TRAINING_FIGURES = 3


@dataclass(frozen=True)
class ScalarSeries:
    """One TensorBoard scalar series aligned to global_step."""

    tag: str
    steps: list[int]
    values: list[float]

    def as_dict(self) -> Dict[str, Any]:
        """Returns a serializable representation."""
        return {"tag": self.tag, "steps": self.steps, "values": self.values}


@dataclass(frozen=True)
class TrainingFigureSpec:
    """Prepared data for one HU009C training figure."""

    figure_id: str
    title: str
    tags: tuple[str, ...]
    x_label: str
    y_labels: tuple[str, ...]
    series: Dict[str, ScalarSeries]
    secondary_y_tags: tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        """Returns a serializable representation."""
        return {
            "figure_id": self.figure_id,
            "title": self.title,
            "tags": list(self.tags),
            "x_label": self.x_label,
            "y_labels": list(self.y_labels),
            "series": {key: value.as_dict() for key, value in self.series.items()},
            "secondary_y_tags": list(self.secondary_y_tags),
        }


def prepare_training_figures(
    log_dir: str | Path,
    reward_window: int = 10,
) -> list[TrainingFigureSpec]:
    """Loads TensorBoard scalars and prepares the three HU009C figures.

    Args:
        log_dir: TensorBoard run directory.
        reward_window: Episode moving average window when ``episode/reward_mean``
            is not available.

    Returns:
        Exactly three figure specifications: reward, loss and q/epsilon.

    Raises:
        FileNotFoundError: If the TensorBoard directory does not exist.
        ValueError: If required tags are missing or scalars are invalid.
    """
    path = Path(log_dir)
    if not path.exists():
        raise FileNotFoundError(f"TensorBoard log directory not found: {path}")
    scalars = load_tensorboard_scalars(path)
    return prepare_training_figures_from_scalars(scalars, reward_window=reward_window)


def prepare_training_figures_from_scalars(
    scalars: Mapping[str, Iterable[tuple[int, float]]],
    reward_window: int = 10,
) -> list[TrainingFigureSpec]:
    """Prepares HU009C figure specs from already loaded scalars."""
    if int(reward_window) <= 0:
        raise ValueError("reward_window must be positive.")
    normalized = {tag: _normalize_series(tag, values) for tag, values in scalars.items()}
    missing = sorted(tag for tag in REQUIRED_TRAINING_TAGS if tag not in normalized)
    if missing:
        raise ValueError(f"Missing required TensorBoard tags for HU009C: {missing}")

    reward_mean = normalized.get("episode/reward_mean")
    if reward_mean is None:
        reward_mean = _moving_average_series(normalized["episode/reward"], window=int(reward_window))

    figures = [
        TrainingFigureSpec(
            figure_id="reward",
            title="Recompensa por episodio y media movil",
            tags=("episode/reward", "episode/reward_mean"),
            x_label="global_step",
            y_labels=("reward",),
            series={
                "episode/reward": normalized["episode/reward"],
                "episode/reward_mean": reward_mean,
            },
        ),
        TrainingFigureSpec(
            figure_id="loss",
            title="Loss DDQN",
            tags=("train/loss", "train/loss_smooth"),
            x_label="global_step",
            y_labels=("loss",),
            series={
                "train/loss": normalized["train/loss"],
                "train/loss_smooth": _moving_average_series(normalized["train/loss"], window=min(25, len(normalized["train/loss"].values))),
            },
        ),
        TrainingFigureSpec(
            figure_id="q_mean_epsilon",
            title="Q-value medio y epsilon",
            tags=("train/q_mean", "train/epsilon"),
            x_label="global_step",
            y_labels=("q_mean", "epsilon"),
            series={
                "train/q_mean": normalized["train/q_mean"],
                "train/epsilon": normalized["train/epsilon"],
            },
            secondary_y_tags=("train/epsilon",),
        ),
    ]
    validate_training_figure_specs(figures)
    return figures


def validate_training_figure_specs(figures: Iterable[TrainingFigureSpec]) -> None:
    """Validates HU009C figure count and basic non-redundancy."""
    figure_list = list(figures)
    if len(figure_list) > MAX_TRAINING_FIGURES:
        raise ValueError(f"HU009C allows at most {MAX_TRAINING_FIGURES} training figures.")
    ids = [figure.figure_id for figure in figure_list]
    if len(ids) != len(set(ids)):
        raise ValueError("Training figure ids must be unique.")
    tag_sets = [frozenset(figure.tags) for figure in figure_list]
    if len(tag_sets) != len(set(tag_sets)):
        raise ValueError("Training figures contain redundant tag structures.")
    expected_ids = {"reward", "loss", "q_mean_epsilon"}
    if set(ids) != expected_ids:
        raise ValueError(f"HU009C training figures must be {sorted(expected_ids)}, got {sorted(ids)}.")


def plot_training_figures(figures: Iterable[TrainingFigureSpec]) -> list[Any]:
    """Creates Matplotlib figures from prepared HU009C specs.

    The import is lazy so tests and non-notebook environments can validate data
    preparation without requiring a plotting backend.
    """
    import matplotlib.pyplot as plt

    rendered = []
    for spec in figures:
        fig, axis = plt.subplots(figsize=(9, 4.5))
        secondary_axis = None
        for tag, series in spec.series.items():
            target_axis = axis
            if tag in spec.secondary_y_tags:
                if secondary_axis is None:
                    secondary_axis = axis.twinx()
                    secondary_axis.set_ylabel(spec.y_labels[-1])
                target_axis = secondary_axis
            target_axis.plot(series.steps, series.values, label=tag)
        axis.set_title(spec.title)
        axis.set_xlabel(spec.x_label)
        axis.set_ylabel(spec.y_labels[0])
        lines, labels = axis.get_legend_handles_labels()
        if secondary_axis is not None:
            extra_lines, extra_labels = secondary_axis.get_legend_handles_labels()
            lines += extra_lines
            labels += extra_labels
        axis.legend(lines, labels, loc="best")
        fig.tight_layout()
        rendered.append(fig)
    return rendered


def _normalize_series(tag: str, values: Iterable[tuple[int, float]]) -> ScalarSeries:
    rows = sorted((int(step), float(value)) for step, value in values)
    if not rows:
        raise ValueError(f"TensorBoard tag {tag} has no scalar values.")
    steps = [step for step, _ in rows]
    scalar_values = [value for _, value in rows]
    if len(steps) != len(set(steps)):
        raise ValueError(f"TensorBoard tag {tag} contains duplicate global_step values.")
    if any(step < 0 for step in steps):
        raise ValueError(f"TensorBoard tag {tag} contains negative global_step values.")
    if any(not math.isfinite(value) for value in scalar_values):
        raise ValueError(f"TensorBoard tag {tag} contains NaN or Inf values.")
    return ScalarSeries(tag=tag, steps=steps, values=scalar_values)


def _moving_average_series(series: ScalarSeries, window: int) -> ScalarSeries:
    selected_window = max(1, int(window))
    values = np.asarray(series.values, dtype=np.float64)
    averaged = []
    for index in range(len(values)):
        start = max(0, index + 1 - selected_window)
        averaged.append(float(np.mean(values[start : index + 1])))
    return ScalarSeries(tag=f"{series.tag}_moving_average_{selected_window}", steps=list(series.steps), values=averaged)
