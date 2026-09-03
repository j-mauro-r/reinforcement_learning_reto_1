"""Minimal greedy evaluation for the trained BattleZone agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def evaluate_agent(
    agent: Any,
    env_factory: Callable[[], Any],
    episodes: int = 10,
) -> list[dict[str, int | float]]:
    """Run the requested number of complete greedy evaluation episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    results: list[dict[str, int | float]] = []
    for episode in range(1, episodes + 1):
        env = env_factory()
        reward_total = 0.0
        try:
            observation, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(observation, epsilon=0.0)
                observation, reward, terminated, truncated, _ = env.step(action)
                reward_total += float(reward)
        finally:
            env.close()

        results.append({
            "episode": episode,
            "reward": reward_total,
        })

    return results
