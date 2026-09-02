"""DQN training cycle orchestration for BattleZone HU006."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from src.agent import DQNAgent
from src.environment import create_battlezone_env, load_config


@dataclass
class TrainingState:
    """Mutable counters for one training run."""

    global_step: int = 0
    episode_index: int = 0
    episode_step: int = 0
    episode_reward: float = 0.0


@dataclass
class TrainingSummary:
    """Structured summary returned at the end of a training run."""

    total_steps: int
    completed_episodes: int
    updates: int
    target_syncs: int
    initial_epsilon: float
    final_epsilon: float
    replay_size: int
    episode_rewards: List[float]
    last_loss: Optional[float]
    episode_lengths: List[int] = field(default_factory=list)
    update_steps: List[int] = field(default_factory=list)
    target_sync_steps: List[int] = field(default_factory=list)
    terminated_episodes: int = 0
    truncated_episodes: int = 0


@dataclass(frozen=True)
class LinearEpsilonSchedule:
    """Linear epsilon schedule with deterministic clamping."""

    start: float
    end: float
    decay_steps: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.start <= 1.0):
            raise ValueError(f"epsilon start must be in [0, 1], got {self.start}.")
        if not (0.0 <= self.end <= 1.0):
            raise ValueError(f"epsilon end must be in [0, 1], got {self.end}.")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive.")

    def value(self, step: int) -> float:
        """Returns epsilon for a global step with no overshoot."""
        clamped_step = max(0, int(step))
        progress = min(clamped_step / float(self.decay_steps), 1.0)
        epsilon = self.start + progress * (self.end - self.start)
        lower, upper = (self.start, self.end) if self.start <= self.end else (self.end, self.start)
        return float(min(max(epsilon, lower), upper))


class DQNTrainer:
    """Orchestrates environment-agent interactions for HU006."""

    def __init__(
        self,
        *,
        env: Any,
        agent: DQNAgent,
        total_timesteps: int,
        learning_starts: int,
        train_frequency: int,
        target_sync_interval: int,
        epsilon_schedule: LinearEpsilonSchedule,
        seed: int,
    ) -> None:
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive.")
        if learning_starts < 0:
            raise ValueError("learning_starts must be >= 0.")
        if train_frequency <= 0:
            raise ValueError("train_frequency must be positive.")
        if target_sync_interval <= 0:
            raise ValueError("target_sync_interval must be positive.")

        self.env = env
        self.agent = agent
        self.total_timesteps = int(total_timesteps)
        self.learning_starts = int(learning_starts)
        self.train_frequency = int(train_frequency)
        self.target_sync_interval = int(target_sync_interval)
        self.epsilon_schedule = epsilon_schedule
        self.seed = int(seed)

    @classmethod
    def from_config(
        cls,
        *,
        config: Optional[Mapping[str, Any]] = None,
        agent: Optional[DQNAgent] = None,
        env: Any = None,
        seed: Optional[int] = None,
        total_timesteps: Optional[int] = None,
        learning_starts: Optional[int] = None,
        train_frequency: Optional[int] = None,
        target_sync_interval: Optional[int] = None,
    ) -> "DQNTrainer":
        """Builds trainer from versioned BattleZone configuration."""
        cfg = dict(config) if config is not None else load_config()
        if cfg.get("algorithm") != "DQN":
            raise ValueError("HU006 requires algorithm='DQN'.")

        training_cfg = cfg["training"]
        epsilon_cfg = training_cfg["epsilon"]
        resolved_seed = int(seed if seed is not None else cfg["environment"]["seed"])

        schedule = LinearEpsilonSchedule(
            start=float(epsilon_cfg["start"]),
            end=float(epsilon_cfg["end"]),
            decay_steps=int(epsilon_cfg["decay_steps"]),
        )
        resolved_agent = agent if agent is not None else DQNAgent.from_config(cfg)
        resolved_env = env if env is not None else create_battlezone_env(cfg, mode="train", seed=resolved_seed)

        return cls(
            env=resolved_env,
            agent=resolved_agent,
            total_timesteps=int(total_timesteps if total_timesteps is not None else training_cfg["total_timesteps"]),
            learning_starts=int(learning_starts if learning_starts is not None else training_cfg["learning_starts"]),
            train_frequency=int(train_frequency if train_frequency is not None else training_cfg["train_frequency"]),
            target_sync_interval=int(
                target_sync_interval
                if target_sync_interval is not None
                else training_cfg["target_sync_interval"]
            ),
            epsilon_schedule=schedule,
            seed=resolved_seed,
        )

    def train(self) -> TrainingSummary:
        """Runs one bounded DQN training cycle and returns a summary."""
        state = TrainingState()
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        update_steps: List[int] = []
        target_sync_steps: List[int] = []
        updates = 0
        target_syncs = 0
        terminated_episodes = 0
        truncated_episodes = 0
        last_loss: Optional[float] = None

        initial_epsilon = self.epsilon_schedule.value(0)
        last_epsilon = initial_epsilon

        observation, _ = self.env.reset(seed=self.seed)
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(self.seed)

        while state.global_step < self.total_timesteps:
            epsilon = self.epsilon_schedule.value(state.global_step)
            last_epsilon = epsilon
            action = int(self.agent.select_action(observation, epsilon))
            if action < 0 or action >= int(self.agent.action_dim):
                raise ValueError(
                    f"Action {action} out of bounds for action_dim={self.agent.action_dim}."
                )

            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            episode_done = bool(terminated or truncated)
            # DQN bootstrap is blocked only by MDP terminal transitions.
            done_for_bootstrap = bool(terminated)
            self.agent.store_transition(
                np.asarray(observation),
                action,
                float(reward),
                np.asarray(next_observation),
                done_for_bootstrap,
            )

            state.global_step += 1
            state.episode_step += 1
            state.episode_reward += float(reward)

            if self._should_update(state.global_step):
                batch = self.agent.sample_batch(self.agent.batch_size)
                update_result = self.agent.update(batch)
                last_loss = float(update_result.loss)
                updates += 1
                update_steps.append(state.global_step)

            if state.global_step % self.target_sync_interval == 0:
                self.agent.sync_target_network()
                target_syncs += 1
                target_sync_steps.append(state.global_step)

            if episode_done:
                episode_rewards.append(float(state.episode_reward))
                episode_lengths.append(int(state.episode_step))
                terminated_episodes += int(bool(terminated))
                truncated_episodes += int(bool(truncated))
                state.episode_index += 1
                state.episode_step = 0
                state.episode_reward = 0.0
                observation, _ = self.env.reset(seed=self.seed + state.episode_index)
            else:
                observation = next_observation

        return TrainingSummary(
            total_steps=state.global_step,
            completed_episodes=state.episode_index,
            updates=updates,
            target_syncs=target_syncs,
            initial_epsilon=initial_epsilon,
            final_epsilon=last_epsilon,
            replay_size=len(self.agent.replay_buffer),
            episode_rewards=episode_rewards,
            last_loss=last_loss,
            episode_lengths=episode_lengths,
            update_steps=update_steps,
            target_sync_steps=target_sync_steps,
            terminated_episodes=terminated_episodes,
            truncated_episodes=truncated_episodes,
        )

    def _should_update(self, global_step: int) -> bool:
        if global_step < self.learning_starts:
            return False
        if len(self.agent.replay_buffer) < int(self.agent.batch_size):
            return False
        return global_step % self.train_frequency == 0
