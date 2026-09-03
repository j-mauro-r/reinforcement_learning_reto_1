"""Minimal greedy evaluation for the trained BattleZone agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def evaluate_agent(
    agent: Any,
    env_factory: Callable[[], Any],
    seeds: Sequence[int],
) -> list[dict[str, int | float]]:
    """Run one complete greedy evaluation episode for each explicit seed."""
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique.")

    results: list[dict[str, int | float]] = []
    for episode, seed in enumerate(seeds, start=1):
        env = env_factory()
        reward_total = 0.0
        steps = 0
        try:
            observation, _ = env.reset(seed=int(seed))
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(observation, epsilon=0.0)
                observation, reward, terminated, truncated, _ = env.step(action)
                reward_total += float(reward)
                steps += 1
        finally:
            env.close()

        results.append({
            "episode": episode,
            "seed": int(seed),
            "reward": reward_total,
            "steps": steps,
        })

    return results
