"""Tests for the minimal HU011C academic evaluation."""

import json
from pathlib import Path
import re

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
        "Evaluación de la política",
        "EVALUATION_EPISODES = 10",
        "evaluate_agent(",
        'print(\"AVERAGE_REWARD:\", average_reward)',
        "plot_exploitation_rewards(",
        "Análisis del comportamiento aprendido — completar por el estudiante",
        "Conclusiones — completar por el estudiante",
    ):
        assert marker in combined


def test_every_code_cell_has_brief_markdown_without_story_identifiers():
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "pipeline_battlezone.ipynb").read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert index > 0
        previous = notebook["cells"][index - 1]
        assert previous.get("cell_type") == "markdown"
        text = "".join(previous.get("source", []))
        assert len(re.findall(r"\b\w+\b", text)) <= 100

    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert not re.search(r"\bHU\s*\d|HU\d", markdown, re.IGNORECASE)
    assert "historia de usuario" not in markdown.lower()
    assert "descripción del código" not in markdown.lower()


def test_notebook_starts_with_short_professor_execution_options():
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "pipeline_battlezone.ipynb").read_text(encoding="utf-8"))
    first = notebook["cells"][0]
    text = "".join(first.get("source", []))

    assert first.get("cell_type") == "markdown"
    assert "# BattleZone — Ejecución" in text
    assert "Opción 1 — Entrenar desde cero" in text
    assert "Run all" in text
    assert "Opción 2 — Probar el agente entrenado" in text
    assert "3_BattleZone/battlezone_dqn_model.pt" in text
    assert "3_BattleZone/requirements.txt" in text
    assert len(re.findall(r"\b\w+\b", text)) <= 80


def test_final_evaluation_uses_root_model_without_drive_or_training_result():
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "pipeline_battlezone.ipynb").read_text(encoding="utf-8"))
    evaluation_cell = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if 'EVALUATION_MODEL_PATH = PROJECT_DIR / "battlezone_dqn_model.pt"'
        in "".join(cell.get("source", []))
    )

    assert "PERSISTENT_ROOT" not in evaluation_cell
    assert "result[" not in evaluation_cell
    assert "run_training_session" not in evaluation_cell
    assert "load_inference_model" in evaluation_cell
