"""Independent evaluation helpers for Assault DDQN agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class EvaluationSummary:
    """Structured result returned by a short evaluation run."""

    episodes: int
    rewards: List[float]
    mean_reward: float
    median_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    episode_lengths: List[int]
    epsilon: float
    terminated_episodes: int
    truncated_episodes: int
    max_steps_per_episode: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        """Returns a notebook-friendly dictionary representation."""
        return {
            "episodes": self.episodes,
            "rewards": self.rewards,
            "mean_reward": self.mean_reward,
            "median_reward": self.median_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "episode_lengths": self.episode_lengths,
            "epsilon": self.epsilon,
            "terminated_episodes": self.terminated_episodes,
            "truncated_episodes": self.truncated_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
        }


def evaluate_agent(
    env: Any,
    agent: Any,
    episodes: int = 2,
    epsilon: float = 0.0,
    max_steps_per_episode: Optional[int] = None,
) -> EvaluationSummary:
    """Evaluates an agent without training or mutating replay state.

    Args:
        env: Gymnasium-compatible evaluation environment.
        agent: Agent exposing ``select_action``.
        episodes: Number of evaluation episodes to run.
        epsilon: Epsilon used by the evaluation policy. Defaults to greedy
            evaluation with no additional exploration.
        max_steps_per_episode: Optional safety bound for smoke runs.

    Returns:
        Evaluation metrics based on raw rewards returned by the environment.

    Raises:
        ValueError: If episodes, epsilon or max step limits are invalid.
    """
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive.")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")
    if max_steps_per_episode is not None and int(max_steps_per_episode) <= 0:
        raise ValueError("max_steps_per_episode must be positive when provided.")

    online_was_training = getattr(getattr(agent, "online_network", None), "training", None)
    target_was_training = getattr(getattr(agent, "target_network", None), "training", None)
    rewards: List[float] = []
    lengths: List[int] = []
    terminated_count = 0
    truncated_count = 0

    try:
        for _ in range(int(episodes)):
            observation, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(observation, epsilon=float(epsilon))
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                if max_steps_per_episode is not None and episode_length >= int(max_steps_per_episode):
                    truncated = True

            if terminated:
                terminated_count += 1
            if truncated:
                truncated_count += 1
            rewards.append(episode_reward)
            lengths.append(episode_length)
    finally:
        if online_was_training is not None:
            agent.online_network.train(bool(online_was_training))
        if target_was_training is not None:
            agent.target_network.train(bool(target_was_training))

    reward_array = np.asarray(rewards, dtype=np.float64)
    return EvaluationSummary(
        episodes=int(episodes),
        rewards=[float(reward) for reward in rewards],
        mean_reward=float(np.mean(reward_array)),
        median_reward=float(np.median(reward_array)),
        std_reward=float(np.std(reward_array)),
        min_reward=float(np.min(reward_array)),
        max_reward=float(np.max(reward_array)),
        episode_lengths=[int(length) for length in lengths],
        epsilon=float(epsilon),
        terminated_episodes=terminated_count,
        truncated_episodes=truncated_count,
        max_steps_per_episode=max_steps_per_episode,
    )
