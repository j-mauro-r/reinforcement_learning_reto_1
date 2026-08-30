"""Short timestep-based training loop for the Assault DDQN agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from .agent import DDQNAgent
from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class TrainingSummary:
    """Structured result returned by a DDQN training run."""

    global_step: int
    episodes_completed: int
    episode_rewards: List[float]
    episode_lengths: List[int]
    epsilon_initial: float
    epsilon_final: float
    transitions_stored: int
    updates_count: int
    update_steps: List[int]
    first_update_step: Optional[int]
    last_loss: Optional[float]
    mean_loss: Optional[float]
    last_q_mean: Optional[float]
    mean_q_mean: Optional[float]
    last_learning_rate: Optional[float]
    target_sync_steps: List[int]
    online_weights_changed: bool
    duration_seconds: float
    final_replay_buffer_size: int
    initial_global_step: int = 0
    checkpoints_saved: List[str] = field(default_factory=list)
    truncated_episodes: int = 0
    terminated_episodes: int = 0
    episode_end_reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Returns a notebook-friendly dictionary representation."""
        return {
            "global_step": self.global_step,
            "episodes_completed": self.episodes_completed,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "epsilon_initial": self.epsilon_initial,
            "epsilon_final": self.epsilon_final,
            "transitions_stored": self.transitions_stored,
            "updates_count": self.updates_count,
            "update_steps": self.update_steps,
            "first_update_step": self.first_update_step,
            "last_loss": self.last_loss,
            "mean_loss": self.mean_loss,
            "last_q_mean": self.last_q_mean,
            "mean_q_mean": self.mean_q_mean,
            "last_learning_rate": self.last_learning_rate,
            "target_sync_steps": self.target_sync_steps,
            "online_weights_changed": self.online_weights_changed,
            "duration_seconds": self.duration_seconds,
            "final_replay_buffer_size": self.final_replay_buffer_size,
            "initial_global_step": self.initial_global_step,
            "checkpoints_saved": self.checkpoints_saved,
            "truncated_episodes": self.truncated_episodes,
            "terminated_episodes": self.terminated_episodes,
            "episode_end_reasons": self.episode_end_reasons,
        }


def compute_epsilon(global_step: int, epsilon_start: float, epsilon_final: float, decay_steps: int) -> float:
    """Computes deterministic linear epsilon decay for a timestep.

    Args:
        global_step: Current agent decision timestep.
        epsilon_start: Initial exploration probability.
        epsilon_final: Minimum exploration probability after decay.
        decay_steps: Number of timesteps used by the linear decay.

    Returns:
        Epsilon clipped to ``[epsilon_final, epsilon_start]``.

    Raises:
        ValueError: If epsilon bounds or decay are invalid.
    """
    if global_step < 0:
        raise ValueError("global_step must be non-negative.")
    if not 0.0 <= epsilon_final <= epsilon_start <= 1.0:
        raise ValueError("Expected 0 <= epsilon_final <= epsilon_start <= 1.")
    if decay_steps <= 0:
        return float(epsilon_final)
    if global_step >= decay_steps:
        return float(epsilon_final)

    fraction = float(global_step) / float(decay_steps)
    epsilon = epsilon_start + fraction * (epsilon_final - epsilon_start)
    return float(np.clip(epsilon, epsilon_final, epsilon_start))


class Trainer:
    """Runs the HU004 DDQN interaction/update sequence by timesteps."""

    def __init__(
        self,
        env: Any,
        agent: DDQNAgent,
        replay_buffer: ReplayBuffer,
        config: Mapping[str, Any],
        initial_global_step: int = 0,
        initial_metrics: Optional[Mapping[str, Any]] = None,
        checkpoint_manager: Any = None,
        checkpoint_interval_steps: Optional[int] = None,
        checkpoint_save_replay_buffer: bool = True,
        checkpoint_keep_last: Optional[int] = None,
        metrics_logger: Any = None,
    ) -> None:
        """Initializes the trainer.

        Args:
            env: Gymnasium-compatible environment.
            agent: DDQN agent that owns action selection and updates.
            replay_buffer: Replay buffer used to store transitions.
            config: Parsed project configuration containing ``training``,
                ``agent`` and ``replay_buffer`` sections.
            initial_global_step: Restored global timestep for resume.
            initial_metrics: Optional accumulated metrics restored from a
                checkpoint.
            checkpoint_manager: Optional object exposing ``save(...)``.
            checkpoint_interval_steps: Optional periodic checkpoint interval.
            checkpoint_save_replay_buffer: Whether automatic checkpoints include
                Replay Buffer state.
            checkpoint_keep_last: Optional checkpoint retention count.
            metrics_logger: Optional observability callback exposing
                ``log_step``, ``log_update``, ``log_episode`` and flush hooks.
        """
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = config
        self.initial_global_step = int(initial_global_step)
        self.initial_metrics = dict(initial_metrics or {})
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_interval_steps = checkpoint_interval_steps
        self.checkpoint_save_replay_buffer = bool(checkpoint_save_replay_buffer)
        self.checkpoint_keep_last = checkpoint_keep_last
        self.metrics_logger = metrics_logger

    def train(self, total_timesteps: Optional[int] = None) -> TrainingSummary:
        """Executes a short DDQN training run.

        Args:
            total_timesteps: Optional override for ``training.total_timesteps``.

        Returns:
            Structured metrics for the completed run.

        Raises:
            ValueError: If training configuration is invalid.
        """
        training_config = self.config["training"]
        agent_config = self.config["agent"]
        replay_config = self.config["replay_buffer"]

        target_timesteps = int(total_timesteps if total_timesteps is not None else training_config["total_timesteps"])
        learning_starts = int(training_config["learning_starts"])
        train_frequency = int(training_config["train_frequency"])
        target_update_frequency = int(training_config["target_update_frequency"])
        epsilon_decay_steps = int(training_config["epsilon_decay_steps"])
        batch_size = int(replay_config["batch_size"])
        epsilon_start = float(agent_config["epsilon_start"])
        epsilon_final_value = float(agent_config["epsilon_final"])

        _validate_training_config(
            total_timesteps=target_timesteps,
            initial_global_step=self.initial_global_step,
            learning_starts=learning_starts,
            train_frequency=train_frequency,
            target_update_frequency=target_update_frequency,
            batch_size=batch_size,
        )

        online_before = _clone_parameters(self.agent.online_network)
        start_time = time.perf_counter()
        observation, _ = self.env.reset()
        global_step = self.initial_global_step
        episode_reward = 0.0
        episode_length = 0
        episode_rewards: List[float] = list(self.initial_metrics.get("episode_rewards", []))
        episode_lengths: List[int] = list(self.initial_metrics.get("episode_lengths", []))
        if self.metrics_logger is not None and hasattr(self.metrics_logger, "set_episode_history"):
            self.metrics_logger.set_episode_history(episode_rewards)
        episode_end_reasons: List[str] = []
        losses: List[float] = []
        q_means: List[float] = []
        learning_rates: List[float] = []
        update_steps: List[int] = []
        target_sync_steps: List[int] = []
        checkpoints_saved: List[str] = []
        initial_updates_count = int(self.initial_metrics.get("updates_count", 0) or 0)
        initial_target_sync_count = int(self.initial_metrics.get("target_sync_count", 0) or 0)
        terminated_episodes = int(self.initial_metrics.get("terminated_episodes", 0) or 0)
        truncated_episodes = int(self.initial_metrics.get("truncated_episodes", 0) or 0)
        transitions_this_run = 0

        while global_step < target_timesteps:
            epsilon = compute_epsilon(global_step, epsilon_start, epsilon_final_value, epsilon_decay_steps)
            action = self.agent.select_action(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            global_step += 1
            transitions_this_run += 1
            episode_reward += float(reward)
            episode_length += 1
            if self.metrics_logger is not None:
                self.metrics_logger.log_step(global_step=global_step, epsilon=epsilon)

            done_for_bootstrap = bool(terminated)
            self.replay_buffer.add(
                state=observation,
                action=action,
                reward=float(reward),
                next_state=next_observation,
                done=done_for_bootstrap,
            )

            if _should_update(global_step, learning_starts, train_frequency, len(self.replay_buffer), batch_size):
                metrics = self.agent.update(self.replay_buffer.sample(batch_size))
                loss = float(metrics["loss"])
                if not np.isfinite(loss):
                    raise RuntimeError(f"Non-finite DDQN loss at step {global_step}: {loss}")
                losses.append(loss)
                update_steps.append(global_step)
                q_mean = float(metrics["q_mean"]) if "q_mean" in metrics else None
                learning_rate = float(metrics["learning_rate"]) if "learning_rate" in metrics else None
                if q_mean is not None and not np.isfinite(q_mean):
                    raise RuntimeError(f"Non-finite DDQN q_mean at step {global_step}: {q_mean}")
                if learning_rate is not None and not np.isfinite(learning_rate):
                    raise RuntimeError(f"Non-finite learning rate at step {global_step}: {learning_rate}")
                if q_mean is not None:
                    q_means.append(q_mean)
                if learning_rate is not None:
                    learning_rates.append(learning_rate)
                if self.metrics_logger is not None:
                    self.metrics_logger.log_update(
                        global_step=global_step,
                        loss=loss,
                        q_mean=q_mean,
                        learning_rate=learning_rate,
                    )

            if global_step % target_update_frequency == 0:
                self.agent.sync_target_network()
                target_sync_steps.append(global_step)

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                if terminated:
                    terminated_episodes += 1
                if truncated:
                    truncated_episodes += 1
                episode_end_reasons.append(_episode_end_reason(terminated, truncated))
                if self.metrics_logger is not None:
                    self.metrics_logger.log_episode(
                        global_step=global_step,
                        reward=episode_reward,
                        length=episode_length,
                    )
                observation, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                observation = next_observation

            if self.metrics_logger is not None:
                self.metrics_logger.flush_if_needed(global_step)

            if self._should_save_checkpoint(global_step):
                metrics = _build_metrics_snapshot(
                    global_step=global_step,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    updates_count=initial_updates_count + len(losses),
                    losses=losses,
                    target_sync_count=initial_target_sync_count + len(target_sync_steps),
                    duration_seconds=time.perf_counter() - start_time + float(self.initial_metrics.get("duration_seconds", 0.0) or 0.0),
                    terminated_episodes=terminated_episodes,
                    truncated_episodes=truncated_episodes,
                )
                saved = self.checkpoint_manager.save(
                    agent=self.agent,
                    replay_buffer=self.replay_buffer,
                    config=self.config,
                    global_step=global_step,
                    training_metrics=metrics,
                    save_replay_buffer=self.checkpoint_save_replay_buffer,
                    keep_last=self.checkpoint_keep_last,
                )
                checkpoints_saved.append(str(saved.path))

        epsilon_final_observed = compute_epsilon(
            global_step,
            epsilon_start,
            epsilon_final_value,
            epsilon_decay_steps,
        )
        online_after = _clone_parameters(self.agent.online_network)
        weights_changed = any(not torch.equal(before, after) for before, after in zip(online_before, online_after))
        duration = time.perf_counter() - start_time + float(self.initial_metrics.get("duration_seconds", 0.0) or 0.0)

        return TrainingSummary(
            global_step=global_step,
            episodes_completed=len(episode_rewards),
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            epsilon_initial=compute_epsilon(self.initial_global_step, epsilon_start, epsilon_final_value, epsilon_decay_steps),
            epsilon_final=epsilon_final_observed,
            transitions_stored=transitions_this_run,
            updates_count=initial_updates_count + len(losses),
            update_steps=update_steps,
            first_update_step=update_steps[0] if update_steps else None,
            last_loss=losses[-1] if losses else None,
            mean_loss=float(np.mean(losses)) if losses else None,
            last_q_mean=q_means[-1] if q_means else None,
            mean_q_mean=float(np.mean(q_means)) if q_means else None,
            last_learning_rate=learning_rates[-1] if learning_rates else None,
            target_sync_steps=target_sync_steps,
            online_weights_changed=weights_changed,
            duration_seconds=duration,
            final_replay_buffer_size=len(self.replay_buffer),
            initial_global_step=self.initial_global_step,
            checkpoints_saved=checkpoints_saved,
            truncated_episodes=truncated_episodes,
            terminated_episodes=terminated_episodes,
            episode_end_reasons=episode_end_reasons,
        )

    def _should_save_checkpoint(self, global_step: int) -> bool:
        if self.checkpoint_manager is None:
            return False
        interval = int(self.checkpoint_interval_steps or 0)
        return interval > 0 and global_step > self.initial_global_step and global_step % interval == 0


def _should_update(
    global_step: int,
    learning_starts: int,
    train_frequency: int,
    replay_size: int,
    batch_size: int,
) -> bool:
    return global_step >= learning_starts and replay_size >= batch_size and global_step % train_frequency == 0


def _clone_parameters(module: torch.nn.Module) -> List[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _build_metrics_snapshot(
    global_step: int,
    episode_rewards: List[float],
    episode_lengths: List[int],
    updates_count: int,
    losses: List[float],
    target_sync_count: int,
    duration_seconds: float,
    terminated_episodes: int,
    truncated_episodes: int,
) -> Dict[str, Any]:
    return {
        "global_step": int(global_step),
        "episodes_completed": len(episode_rewards),
        "episode_rewards": list(episode_rewards),
        "episode_lengths": list(episode_lengths),
        "updates_count": int(updates_count),
        "last_loss": losses[-1] if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "target_sync_count": int(target_sync_count),
        "duration_seconds": float(duration_seconds),
        "terminated_episodes": int(terminated_episodes),
        "truncated_episodes": int(truncated_episodes),
    }


def _episode_end_reason(terminated: bool, truncated: bool) -> str:
    if terminated and truncated:
        return "terminated+truncated"
    if terminated:
        return "terminated"
    return "truncated"


def _validate_training_config(
    total_timesteps: int,
    initial_global_step: int,
    learning_starts: int,
    train_frequency: int,
    target_update_frequency: int,
    batch_size: int,
) -> None:
    if total_timesteps <= 0:
        raise ValueError("training.total_timesteps must be positive.")
    if initial_global_step < 0:
        raise ValueError("initial_global_step must be non-negative.")
    if initial_global_step > total_timesteps:
        raise ValueError("initial_global_step cannot exceed training.total_timesteps.")
    if learning_starts < 0:
        raise ValueError("training.learning_starts must be non-negative.")
    if train_frequency <= 0:
        raise ValueError("training.train_frequency must be positive.")
    if target_update_frequency <= 0:
        raise ValueError("training.target_update_frequency must be positive.")
    if batch_size <= 0:
        raise ValueError("replay_buffer.batch_size must be positive.")
