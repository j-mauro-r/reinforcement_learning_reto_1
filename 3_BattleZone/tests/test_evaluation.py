"""Tests for the minimal HU011C academic evaluation."""

import json
from pathlib import Path

import pytest

from src.evaluation import evaluate_agent


class FakeAgent:
    def __init__(self):
        self.epsilons = []

    def select_action(self, observation, epsilon):
        self.epsilons.append(epsilon)
        return 1


class FakeEnv:
    def __init__(self):
        self.steps = 0
        self.closed = False

    def reset(self):
        return 0, {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps == 3
        return 0, float(action), terminated, False, {}

    def close(self):
        self.closed = True


def test_evaluate_agent_runs_requested_complete_greedy_episodes():
    agent = FakeAgent()
    environments = []

    def env_factory():
        env = FakeEnv()
        environments.append(env)
        return env

    results = evaluate_agent(agent, env_factory, episodes=2)

    assert results == [
        {"episode": 1, "reward": 3.0},
        {"episode": 2, "reward": 3.0},
    ]
    assert agent.epsilons == [0.0] * 6
    assert all(env.closed for env in environments)


def test_evaluate_agent_defaults_to_ten_episodes_and_rejects_invalid_count():
    results = evaluate_agent(FakeAgent(), FakeEnv)
    assert len(results) == 10
    assert all(set(result) == {"episode", "reward"} for result in results)
    with pytest.raises(ValueError, match="positive"):
        evaluate_agent(FakeAgent(), FakeEnv, episodes=0)


def test_notebook_contains_the_minimum_hu011c_academic_evidence_flow():
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "pipeline_battlezone.ipynb").read_text(encoding="utf-8"))
    combined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for marker in (
        "HU011C — Evaluación académica de explotación",
        "EVALUATION_EPISODES = 10",
        "evaluate_agent(",
        'print(\"AVERAGE_REWARD:\", average_reward)',
        "plot_exploitation_rewards(",
        "Análisis del comportamiento aprendido — completar por el estudiante",
        "Conclusiones — completar por el estudiante",
    ):
        assert marker in combined
