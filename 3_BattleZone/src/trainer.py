"""DQN training cycle orchestration for BattleZone HU006."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

import numpy as np

from src.agent import DQNAgent
from src.environment import create_battlezone_env, load_config


class TrainingLogger(Protocol):
    """Small logger contract used by trainer without hard dependency on TensorBoard."""

    def on_training_start(self) -> None: ...

    def on_step(
        self,
        *,
        global_step: int,
        epsilon: float,
        replay_size: int,
        learning_rate: float,
    ) -> None: ...

    def on_update(self, *, global_step: int, loss: float, q_value_mean: float) -> None: ...

    def on_episode_end(self, *, global_step: int, episode_reward: float, episode_length: int) -> None: ...

    def on_training_end(self) -> None: ...

    def close(self) -> None: ...


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
    start_global_step: int = 0
    run_mode: str = "new"
    replay_restored: bool = False
    first_update_step: Optional[int] = None


class TrainingMode(str, Enum):
    """Explicit training modes for idempotent NEW/RESUME behavior."""

    NEW = "new"
    RESUME_FULL = "resume_full"
    RESUME_LIGHTWEIGHT = "resume_lightweight"


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
        logger: Optional[TrainingLogger] = None,
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
        self.logger = logger
        self.training_state = TrainingState()
        self.mode = TrainingMode.NEW
        self.replay_restored = False

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
        logger: Optional[TrainingLogger] = None,
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
            logger=logger,
        )

    def train(
        self,
        *,
        total_timesteps: Optional[int] = None,
        initial_state: Optional[TrainingState] = None,
        mode: TrainingMode | str | None = None,
        replay_restored: Optional[bool] = None,
        step_callback: Optional[Callable[[int, "DQNTrainer"], None]] = None,
    ) -> TrainingSummary:
        """Runs one bounded DQN training cycle and returns a summary.

        Semantics: ``total_timesteps`` is a global final target. If the restored
        state has ``global_step=N`` and ``total_timesteps=M``, this call executes
        exactly ``M-N`` additional steps when ``M > N``.
        """
        if initial_state is not None:
            self.training_state = TrainingState(
                global_step=int(initial_state.global_step),
                episode_index=int(initial_state.episode_index),
                episode_step=int(initial_state.episode_step),
                episode_reward=float(initial_state.episode_reward),
            )
        state = self.training_state

        if mode is not None:
            self.mode = mode if isinstance(mode, TrainingMode) else TrainingMode(str(mode))
        if replay_restored is not None:
            self.replay_restored = bool(replay_restored)

        target_global_step = int(self.total_timesteps if total_timesteps is None else total_timesteps)
        if target_global_step <= 0:
            raise ValueError("total_timesteps must be positive.")

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        update_steps: List[int] = []
        target_sync_steps: List[int] = []
        updates = 0
        target_syncs = 0
        terminated_episodes = 0
        truncated_episodes = 0
        last_loss: Optional[float] = None

        initial_epsilon = self.epsilon_schedule.value(state.global_step)
        last_epsilon = initial_epsilon

        if self.mode is not TrainingMode.NEW:
            # We do not serialize ALE internals in HU007, so resumed processes
            # restart from a fresh reset while preserving global progress.
            state.episode_step = 0
            state.episode_reward = 0.0

        start_global_step = int(state.global_step)
        observation, _ = self.env.reset(seed=self.seed + state.episode_index)
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(self.seed + state.episode_index)

        if self.logger is not None:
            self.logger.on_training_start()

        try:
            while state.global_step < target_global_step:
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
                    if self.logger is not None:
                        self.logger.on_update(
                            global_step=state.global_step,
                            loss=float(update_result.loss),
                            q_value_mean=float(update_result.q_value_mean),
                        )

                if state.global_step % self.target_sync_interval == 0:
                    self.agent.sync_target_network()
                    target_syncs += 1
                    target_sync_steps.append(state.global_step)

                if self.logger is not None:
                    current_epsilon = self.epsilon_schedule.value(state.global_step)
                    self.logger.on_step(
                        global_step=state.global_step,
                        epsilon=current_epsilon,
                        replay_size=len(self.agent.replay_buffer),
                        learning_rate=float(self.agent.optimizer.param_groups[0]["lr"]),
                    )

                if episode_done:
                    episode_rewards.append(float(state.episode_reward))
                    episode_lengths.append(int(state.episode_step))
                    terminated_episodes += int(bool(terminated))
                    truncated_episodes += int(bool(truncated))
                    if self.logger is not None:
                        self.logger.on_episode_end(
                            global_step=state.global_step,
                            episode_reward=float(state.episode_reward),
                            episode_length=int(state.episode_step),
                        )
                    state.episode_index += 1
                    state.episode_step = 0
                    state.episode_reward = 0.0
                    observation, _ = self.env.reset(seed=self.seed + state.episode_index)
                else:
                    observation = next_observation

                if step_callback is not None:
                    step_callback(state.global_step, self)
        finally:
            if self.logger is not None:
                self.logger.on_training_end()
                self.logger.close()

        self.training_state = state

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
            start_global_step=start_global_step,
            run_mode=self.mode.value,
            replay_restored=self.replay_restored,
            first_update_step=(update_steps[0] if update_steps else None),
        )

    def export_training_state(self) -> Dict[str, float | int]:
        """Returns the trainer progress state suitable for checkpoint payloads."""
        return {
            "global_step": int(self.training_state.global_step),
            "episode_index": int(self.training_state.episode_index),
            "episode_step": int(self.training_state.episode_step),
            "episode_reward": float(self.training_state.episode_reward),
        }

    def _should_update(self, global_step: int) -> bool:
        if global_step < self.learning_starts:
            return False
        if len(self.agent.replay_buffer) < int(self.agent.batch_size):
            return False
        return global_step % self.train_frequency == 0
