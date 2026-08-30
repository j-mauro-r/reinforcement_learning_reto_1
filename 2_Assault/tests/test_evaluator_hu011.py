"""HU011 tests for explicit independent evaluation seeds."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluator import evaluate_agent


class SeedRecordingEnv:
    def __init__(self) -> None:
        self.reset_seeds = []
        self.steps = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action):
        self.steps += 1
        return np.zeros((4, 84, 84), dtype=np.uint8), 1.0, self.steps >= 2, False, {}


class GreedyAgent:
    online_network = None
    target_network = None

    def select_action(self, observation, epsilon=0.0):
        assert epsilon == 0.0
        return 0


def test_evaluate_agent_records_and_uses_explicit_episode_seeds():
    env = SeedRecordingEnv()
    seeds = list(range(100, 110))
    summary = evaluate_agent(env, GreedyAgent(), episodes=10, epsilon=0.0, episode_seeds=seeds)

    assert env.reset_seeds == seeds
    assert summary.episode_seeds == seeds
    assert summary.episodes == 10
    assert summary.rewards == [2.0] * 10
    assert summary.episode_lengths == [2] * 10


def test_evaluate_agent_rejects_seed_count_mismatch_and_duplicates():
    env = SeedRecordingEnv()
    with pytest.raises(ValueError, match="length must match"):
        evaluate_agent(env, GreedyAgent(), episodes=10, epsilon=0.0, episode_seeds=[1, 2])
    with pytest.raises(ValueError, match="must be unique"):
        evaluate_agent(env, GreedyAgent(), episodes=2, epsilon=0.0, episode_seeds=[1, 1])
