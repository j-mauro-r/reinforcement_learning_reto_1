"""Tests for HU011B TensorBoard reporting."""

from pathlib import Path

import matplotlib
import pytest
from torch.utils.tensorboard import SummaryWriter

from src.reporting import (
    FIGURE_FILENAMES, REQUIRED_TAGS, build_training_figures,
    load_tensorboard_scalars, normalize_scalar_series,
    plot_exploitation_rewards, rolling_mean,
)


matplotlib.use("Agg")


def _write_events(path: Path, omit: str | None = None) -> None:
    writer = SummaryWriter(str(path))
    for step in (10, 20, 30):
        for index, tag in enumerate(REQUIRED_TAGS):
            if tag != omit:
                writer.add_scalar(tag, float(step + index), step)
    writer.close()


def test_tensorboard_parsing_preserves_steps_and_persists_three_figures(tmp_path):
    log_dir = tmp_path / "logs" / "explicit-run"
    _write_events(log_dir)
    scalars = load_tensorboard_scalars(log_dir)
    assert set(scalars) == set(REQUIRED_TAGS)
    assert scalars["train/loss"].steps == (10, 20, 30)
    smooth = rolling_mean(scalars["train/loss"], window=2)
    assert smooth.steps == (10, 20, 30)
    figures = build_training_figures(scalars, tmp_path / "figures")
    assert tuple(figures) == FIGURE_FILENAMES
    assert all((tmp_path / "figures" / name).stat().st_size > 0 for name in FIGURE_FILENAMES)


def test_missing_invalid_and_duplicate_scalars_fail_clearly(tmp_path):
    log_dir = tmp_path / "missing"
    _write_events(log_dir, omit="train/loss")
    with pytest.raises(ValueError, match="Missing required"):
        load_tensorboard_scalars(log_dir)
    with pytest.raises(ValueError, match="duplicate"):
        normalize_scalar_series("train/loss", [(1, 1.0), (1, 2.0)])
    with pytest.raises(ValueError, match="invalid"):
        normalize_scalar_series("train/loss", [(1, float("nan"))])


def test_exploitation_plot_consumes_structured_records_without_manual_values(tmp_path):
    records = [
        {"episode": 1, "reward": 2.0},
        {"episode": 2, "reward": 4.0},
    ]
    output = tmp_path / "exploitation_reward.png"
    figure = plot_exploitation_rewards(records, output_path=output)
    plotted = list(figure.axes[0].lines[0].get_ydata())
    assert plotted == [2.0, 4.0]
    assert "DELIVERY_SANITY_ONLY" in figure.axes[0].get_title()
    assert output.stat().st_size > 0
    with pytest.raises(ValueError, match="requires"):
        plot_exploitation_rewards([{"episode": 1}])
