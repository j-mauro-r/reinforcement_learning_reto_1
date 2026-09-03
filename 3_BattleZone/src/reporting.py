"""TensorBoard-backed academic figures for BattleZone HU011B."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REQUIRED_TAGS = (
    "train/episode_reward", "train/episode_reward_mean", "train/loss",
    "train/q_value_mean", "train/epsilon",
)
FIGURE_FILENAMES = (
    "training_reward.png", "training_loss.png", "training_q_epsilon.png",
)


@dataclass(frozen=True)
class ScalarSeries:
    """One finite TensorBoard series ordered by global step."""

    tag: str
    steps: tuple[int, ...]
    values: tuple[float, ...]


def load_tensorboard_scalars(
    log_dir: str | Path, required_tags: Sequence[str] = REQUIRED_TAGS,
) -> dict[str, ScalarSeries]:
    """Loads and validates real scalar events from one explicit run directory."""
    path = Path(log_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"TensorBoard log directory not found: {path}")
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = sorted(set(required_tags) - available)
    if missing:
        raise ValueError(f"Missing required TensorBoard tags: {missing}")
    return {
        tag: normalize_scalar_series(tag, ((event.step, event.value) for event in accumulator.Scalars(tag)))
        for tag in required_tags
    }


def normalize_scalar_series(tag: str, rows: Iterable[tuple[int, float]]) -> ScalarSeries:
    """Preserves global steps while rejecting duplicates and non-finite values."""
    values = sorted((int(step), float(value)) for step, value in rows)
    if not values:
        raise ValueError(f"TensorBoard tag {tag} has no values.")
    steps = tuple(step for step, _ in values)
    scalars = tuple(value for _, value in values)
    if len(steps) != len(set(steps)):
        raise ValueError(f"TensorBoard tag {tag} has duplicate global_step values.")
    if steps[0] < 0 or any(not math.isfinite(value) for value in scalars):
        raise ValueError(f"TensorBoard tag {tag} contains invalid values.")
    return ScalarSeries(tag, steps, scalars)


def rolling_mean(series: ScalarSeries, window: int = 25) -> ScalarSeries:
    """Computes a trailing mean without changing the source global steps."""
    if window <= 0:
        raise ValueError("window must be positive.")
    values = np.asarray(series.values, dtype=np.float64)
    result = tuple(float(values[max(0, index + 1 - window):index + 1].mean()) for index in range(len(values)))
    return ScalarSeries(f"{series.tag}_rolling_{window}", series.steps, result)


def build_training_figures(
    scalars: Mapping[str, ScalarSeries], output_dir: str | Path,
) -> dict[str, Any]:
    """Builds and persists the three non-redundant HU011B training figures."""
    missing = sorted(set(REQUIRED_TAGS) - set(scalars))
    if missing:
        raise ValueError(f"Missing required TensorBoard series: {missing}")
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Any] = {}

    reward_fig, reward_ax = plt.subplots(figsize=(9, 4.5))
    for tag in ("train/episode_reward", "train/episode_reward_mean"):
        series = scalars[tag]
        reward_ax.plot(series.steps, series.values, label=tag)
    _finish_axis(reward_ax, "BattleZone training reward", "reward")
    figures[FIGURE_FILENAMES[0]] = reward_fig

    loss = scalars["train/loss"]
    loss_fig, loss_ax = plt.subplots(figsize=(9, 4.5))
    loss_ax.plot(loss.steps, loss.values, alpha=0.35, label=loss.tag)
    smooth = rolling_mean(loss, min(100, len(loss.values)))
    loss_ax.plot(smooth.steps, smooth.values, label=smooth.tag)
    _finish_axis(loss_ax, "BattleZone DQN training loss", "Smooth L1 loss")
    figures[FIGURE_FILENAMES[1]] = loss_fig

    q_series, epsilon = scalars["train/q_value_mean"], scalars["train/epsilon"]
    q_fig, q_ax = plt.subplots(figsize=(9, 4.5))
    epsilon_ax = q_ax.twinx()
    q_ax.plot(q_series.steps, q_series.values, label=q_series.tag, color="tab:blue")
    epsilon_ax.plot(epsilon.steps, epsilon.values, label=epsilon.tag, color="tab:orange")
    q_ax.set_title("BattleZone mean Q-value and epsilon")
    q_ax.set_xlabel("global_step")
    q_ax.set_ylabel("mean Q-value")
    epsilon_ax.set_ylabel("epsilon")
    lines1, labels1 = q_ax.get_legend_handles_labels()
    lines2, labels2 = epsilon_ax.get_legend_handles_labels()
    q_ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    q_fig.tight_layout()
    figures[FIGURE_FILENAMES[2]] = q_fig

    for filename, figure in figures.items():
        figure.savefig(destination / filename, dpi=140, bbox_inches="tight")
        if not (destination / filename).is_file() or (destination / filename).stat().st_size == 0:
            raise RuntimeError(f"Figure was not persisted: {destination / filename}")
    return figures


def plot_exploitation_rewards(
    records: Sequence[Mapping[str, Any]], *, status: str = "DELIVERY_SANITY_ONLY",
    output_path: str | Path | None = None,
) -> Any:
    """Plots structured episode/seed/reward/steps records, ready for HU013 data."""
    if not records:
        raise ValueError("At least one exploitation record is required.")
    required = {"episode", "seed", "reward", "steps"}
    if any(required - set(record) for record in records):
        raise ValueError("Each exploitation record requires episode, seed, reward, and steps.")
    rewards = np.asarray([float(record["reward"]) for record in records], dtype=np.float64)
    if not np.isfinite(rewards).all():
        raise ValueError("Exploitation rewards contain NaN or Inf.")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4.5))
    episodes = [int(record["episode"]) for record in records]
    axis.plot(episodes, rewards, marker="o", label="raw reward")
    axis.axhline(float(rewards.mean()), linestyle="--", label=f"mean={rewards.mean():.2f}")
    _finish_axis(axis, f"Exploitation reward — {status}", "reward")
    axis.set_xlabel("episode")
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=140, bbox_inches="tight")
    return figure


def _finish_axis(axis: Any, title: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("global_step")
    axis.set_ylabel(ylabel)
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    axis.figure.tight_layout()
