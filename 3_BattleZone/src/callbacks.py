"""TensorBoard observability callbacks for BattleZone HU008."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Optional

from torch.utils.tensorboard import SummaryWriter


class TensorBoardTrainingLogger:
    """Logs training scalars to TensorBoard using explicit step semantics."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        reward_window: int,
        scalar_log_interval_steps: int,
        flush_interval_steps: int,
    ) -> None:
        if reward_window <= 0:
            raise ValueError("reward_window must be positive.")
        if scalar_log_interval_steps <= 0:
            raise ValueError("scalar_log_interval_steps must be positive.")
        if flush_interval_steps <= 0:
            raise ValueError("flush_interval_steps must be positive.")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.reward_window = int(reward_window)
        self.scalar_log_interval_steps = int(scalar_log_interval_steps)
        self.flush_interval_steps = int(flush_interval_steps)
        self._episode_rewards: Deque[float] = deque(maxlen=self.reward_window)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        self._closed = False

    def on_training_start(self) -> None:
        """Marks the start of a training execution."""

    def on_step(
        self,
        *,
        global_step: int,
        epsilon: float,
        replay_size: int,
        learning_rate: float,
    ) -> None:
        """Logs periodic progress scalars on configured intervals."""
        step = int(global_step)
        if step % self.scalar_log_interval_steps != 0:
            return

        self._writer.add_scalar("train/epsilon", float(epsilon), step)
        self._writer.add_scalar("train/replay_size", float(replay_size), step)
        self._writer.add_scalar("train/learning_rate", float(learning_rate), step)
        self._maybe_flush(step)

    def on_update(self, *, global_step: int, loss: float, q_value_mean: float) -> None:
        """Logs optimizer-step metrics."""
        step = int(global_step)
        self._writer.add_scalar("train/loss", float(loss), step)
        self._writer.add_scalar("train/q_value_mean", float(q_value_mean), step)
        self._maybe_flush(step)

    def on_episode_end(self, *, global_step: int, episode_reward: float, episode_length: int) -> None:
        """Logs episode-end metrics including moving reward average."""
        step = int(global_step)
        reward = float(episode_reward)
        length = int(episode_length)

        self._episode_rewards.append(reward)
        reward_mean = sum(self._episode_rewards) / float(len(self._episode_rewards))

        self._writer.add_scalar("train/episode_reward", reward, step)
        self._writer.add_scalar("train/episode_reward_mean", float(reward_mean), step)
        self._writer.add_scalar("train/episode_length", float(length), step)
        self._maybe_flush(step)

    def on_training_end(self) -> None:
        """Flushes pending events at the end of training."""
        self.flush()

    def flush(self) -> None:
        """Flushes pending event data to disk."""
        if self._closed:
            return
        self._writer.flush()

    def close(self) -> None:
        """Closes writer safely and idempotently."""
        if self._closed:
            return
        self._writer.flush()
        self._writer.close()
        self._closed = True

    def _maybe_flush(self, global_step: int) -> None:
        if int(global_step) % self.flush_interval_steps == 0:
            self._writer.flush()
