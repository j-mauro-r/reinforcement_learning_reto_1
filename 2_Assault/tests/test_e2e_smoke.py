"""Tests for HU007 end-to-end smoke and evaluator."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.agent import DDQNAgent
from src.e2e_smoke import run_e2e_smoke
from src.evaluator import evaluate_agent
from src.replay_buffer import ReplayBuffer
from src.utils import load_yaml_config


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


class ControlledEvalEnv:
    """Deterministic evaluation environment with raw reward schedule."""

    def __init__(self, rewards: list[list[float]]) -> None:
        self.rewards = rewards
        self.episode_index = -1
        self.step_index = 0

    def reset(self, seed=None):
        self.episode_index += 1
        self.step_index = 0
        return self._obs(), {"seed": seed}

    def step(self, action: int):
        reward = self.rewards[self.episode_index][self.step_index]
        self.step_index += 1
        terminated = self.step_index >= len(self.rewards[self.episode_index])
        return self._obs(), float(reward), terminated, False, {"action": action}

    def _obs(self) -> np.ndarray:
        return np.full((4, 84, 84), self.step_index, dtype=np.uint8)


class RecordingAgent:
    """Simple policy probe that records evaluation epsilons."""

    def __init__(self) -> None:
        self.epsilons: list[float] = []
        self.update_calls = 0

    def select_action(self, state, epsilon: float) -> int:
        self.epsilons.append(float(epsilon))
        return 0

    def update(self, batch):  # pragma: no cover - must never be called.
        self.update_calls += 1
        raise AssertionError("evaluate_agent must not call update().")


def _config() -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["replay_buffer"] = {"capacity": 64, "batch_size": 8}
    config["training"] = {
        "total_timesteps": 24,
        "learning_starts": 8,
        "train_frequency": 4,
        "target_update_frequency": 8,
        "epsilon_decay_steps": 24,
    }
    config["checkpointing"]["interval_steps"] = 16
    config["tensorboard"]["log_frequency_steps"] = 4
    config["tensorboard"]["flush_frequency_steps"] = 8
    config["e2e_smoke"] = {
        "enabled": True,
        "segment_a_timesteps": 16,
        "final_timesteps": 24,
        "evaluation_episodes": 1,
        "evaluation_epsilon": 0.0,
        "evaluation_max_steps_per_episode": 16,
        "replay_buffer_capacity": 64,
        "require_cuda": False,
    }
    return config


def _clone_parameters(agent: DDQNAgent) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in agent.online_network.parameters()]


def test_evaluator_result_structure_epsilon_raw_rewards_and_lengths():
    env = ControlledEvalEnv([[1.0, 2.0], [3.0, -1.0, 4.0]])
    agent = RecordingAgent()

    evaluation = evaluate_agent(env, agent, episodes=2)

    assert evaluation.episodes == 2
    assert evaluation.epsilon == 0.0
    assert agent.epsilons == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert evaluation.rewards == [3.0, 6.0]
    assert evaluation.episode_lengths == [2, 3]
    assert evaluation.mean_reward == pytest.approx(4.5)
    assert evaluation.median_reward == pytest.approx(4.5)
    assert evaluation.std_reward == pytest.approx(1.5)
    assert evaluation.min_reward == pytest.approx(3.0)
    assert evaluation.max_reward == pytest.approx(6.0)
    assert agent.update_calls == 0


def test_evaluator_does_not_change_online_or_target_weights():
    config = _config()
    env = ControlledEvalEnv([[0.0, 1.0]])
    agent = DDQNAgent(config, device="cpu", seed=42)
    online_before = _clone_parameters(agent)
    target_before = [parameter.detach().clone() for parameter in agent.target_network.parameters()]
    optimizer_before = agent.optimizer.state_dict()

    evaluate_agent(env, agent, episodes=1, epsilon=0.0)

    assert all(torch.equal(before, after) for before, after in zip(online_before, agent.online_network.parameters()))
    assert all(torch.equal(before, after) for before, after in zip(target_before, agent.target_network.parameters()))
    assert agent.optimizer.state_dict()["param_groups"] == optimizer_before["param_groups"]


def test_e2e_smoke_cuda_required_fails_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="require_cuda=true"):
        run_e2e_smoke(
            _config(),
            checkpoint_root=tmp_path / "checkpoints",
            tensorboard_root=tmp_path / "tensorboard",
            run_id="cuda_required",
            require_cuda=True,
        )


def test_real_assault_cpu_e2e_smoke_checkpoint_resume_tensorboard_and_evaluation(tmp_path):
    config = _config()

    summary = run_e2e_smoke(
        config,
        checkpoint_root=tmp_path / "checkpoints",
        tensorboard_root=tmp_path / "tensorboard",
        run_id="hu007_cpu_e2e",
        repo_path=ASSAULT_DIR.parents[0],
        require_cuda=False,
        device="cpu",
    )

    assert summary.local_e2e_smoke_pass is True
    assert summary.e2e_smoke_pass is False
    assert summary.runtime == "local"
    assert summary.device == "cpu"
    assert summary.observation_shape == (4, 84, 84)
    assert summary.observation_dtype == "uint8"
    assert summary.action_space == "Discrete(7)"
    assert summary.preflight.ready_for_training is True
    assert summary.segment_a.global_step == 16
    assert summary.segment_a.updates_count > 0
    assert math.isfinite(summary.segment_a.last_loss)
    assert math.isfinite(summary.segment_a.last_q_mean)
    assert summary.checkpoint.checkpoint_step == 16
    assert summary.checkpoint.size_bytes > 0
    assert summary.restored.global_step == 16
    assert summary.restored.replay_buffer_restored is True
    assert summary.segment_b.global_step == 24
    assert summary.segment_b.updates_count > summary.segment_a.updates_count
    assert summary.tensorboard_event_files_before >= 1
    assert summary.tensorboard_event_files_after >= summary.tensorboard_event_files_before
    assert summary.tensorboard_previous_logs_preserved is True
    assert {"train/epsilon", "train/loss", "train/q_mean", "train/learning_rate"}.issubset(summary.tensorboard_tags)
    assert summary.tensorboard_post_resume_steps["train/loss"] == [20, 24]
    assert summary.evaluation.episodes == 1
    assert summary.evaluation.epsilon == 0.0
    assert len(summary.evaluation.rewards) == 1
    assert len(summary.evaluation.episode_lengths) == 1
    assert summary.online_unchanged_during_evaluation is True
    assert summary.target_unchanged_during_evaluation is True
    assert summary.optimizer_unchanged_during_evaluation is True
    assert summary.replay_buffer_unchanged_during_evaluation is True
    assert summary.training_global_step_unchanged_during_evaluation is True
    assert summary.memory_after.process_rss_mb > 0


def test_e2e_summary_reports_replay_buffer_unchanged_contract(tmp_path):
    config = _config()
    buffer = ReplayBuffer(capacity=16, seed=123)
    assert len(buffer) == 0

    summary = run_e2e_smoke(
        config,
        checkpoint_root=tmp_path / "checkpoints",
        tensorboard_root=tmp_path / "tensorboard",
        run_id="hu007_summary_contract",
        require_cuda=False,
        device="cpu",
    ).as_dict()

    assert summary["LOCAL_E2E_SMOKE_PASS"] is True
    assert summary["E2E_SMOKE_PASS"] is False
    assert summary["replay_buffer_unchanged_during_evaluation"] is True
    assert summary["training_global_step_unchanged_during_evaluation"] is True
