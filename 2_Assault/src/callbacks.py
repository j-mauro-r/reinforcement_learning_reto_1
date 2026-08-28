"""Training observability callbacks for the Assault DDQN project."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Small TensorBoard wrapper used by ``Trainer``.

    The trainer owns algorithm timing; this class only writes scalar events.
    """

    def __init__(
        self,
        log_root: str | Path,
        run_id: str,
        enabled: bool = True,
        log_frequency_steps: int = 1,
        reward_window_episodes: int = 10,
        flush_frequency_steps: int = 0,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty.")
        if int(log_frequency_steps) <= 0:
            raise ValueError("log_frequency_steps must be positive.")
        if int(reward_window_episodes) <= 0:
            raise ValueError("reward_window_episodes must be positive.")
        if int(flush_frequency_steps) < 0:
            raise ValueError("flush_frequency_steps must be non-negative.")

        self.log_root = Path(log_root)
        self.run_id = str(run_id)
        self.enabled = bool(enabled)
        self.log_frequency_steps = int(log_frequency_steps)
        self.reward_window_episodes = int(reward_window_episodes)
        self.flush_frequency_steps = int(flush_frequency_steps)
        self.run_log_dir = self.log_root / self.run_id
        self._episode_rewards: list[float] = []
        self._writer = SummaryWriter(str(self.run_log_dir)) if self.enabled else None

    @classmethod
    def from_config(cls, config: Mapping[str, Any], run_id: str, log_root: str | Path | None = None) -> "TensorBoardLogger":
        tensorboard_config = dict(config.get("tensorboard", {}))
        configured_root = tensorboard_config.get("directory", "logs/tensorboard")
        return cls(
            log_root=log_root or configured_root,
            run_id=run_id,
            enabled=bool(tensorboard_config.get("enabled", True)),
            log_frequency_steps=int(tensorboard_config.get("log_frequency_steps", 1)),
            reward_window_episodes=int(tensorboard_config.get("reward_window_episodes", 10)),
            flush_frequency_steps=int(tensorboard_config.get("flush_frequency_steps", 0)),
        )

    def log_step(self, global_step: int, epsilon: float) -> None:
        if not self._should_write_step(global_step):
            return
        self._add_scalar("train/epsilon", epsilon, global_step)

    def log_update(
        self,
        global_step: int,
        loss: float,
        q_mean: float | None = None,
        learning_rate: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._add_scalar("train/loss", loss, global_step)
        if q_mean is not None:
            self._add_scalar("train/q_mean", q_mean, global_step)
        if learning_rate is not None:
            self._add_scalar("train/learning_rate", learning_rate, global_step)

    def log_episode(self, global_step: int, reward: float, length: int) -> float:
        self._episode_rewards.append(float(reward))
        window = self._episode_rewards[-self.reward_window_episodes :]
        reward_mean = float(np.mean(window))
        if self.enabled:
            self._add_scalar("episode/reward", reward, global_step)
            self._add_scalar("episode/reward_mean", reward_mean, global_step)
            self._add_scalar("episode/length", float(length), global_step)
        return reward_mean

    def set_episode_history(self, rewards: list[float]) -> None:
        self._episode_rewards = [float(reward) for reward in rewards]

    def flush_if_needed(self, global_step: int) -> None:
        if self.enabled and self.flush_frequency_steps > 0 and global_step % self.flush_frequency_steps == 0:
            self.flush()

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def event_files(self) -> list[Path]:
        if not self.run_log_dir.exists():
            return []
        return sorted(self.run_log_dir.glob("events.out.tfevents.*"))

    def _should_write_step(self, global_step: int) -> bool:
        return self.enabled and int(global_step) > 0 and int(global_step) % self.log_frequency_steps == 0

    def _add_scalar(self, tag: str, value: float, global_step: int) -> None:
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"Non-finite TensorBoard scalar for {tag} at step {global_step}: {scalar}")
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar, int(global_step))


def load_tensorboard_scalars(log_dir: str | Path) -> Dict[str, list[tuple[int, float]]]:
    """Reads scalar TensorBoard events from a run directory."""
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return {
        tag: [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }
