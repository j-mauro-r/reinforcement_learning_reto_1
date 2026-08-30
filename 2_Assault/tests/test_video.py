"""Tests for HU009C video generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.agent import DDQNAgent
from src.utils import load_yaml_config
from src.video import generate_assault_demo_video


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


class SyntheticRgbEnv:
    """Small deterministic rgb_array environment for MP4 tests."""

    def __init__(self) -> None:
        self.step_index = 0
        self.closed = False

    def reset(self, seed=None):
        self.step_index = 0
        return self._obs(), {"seed": seed}

    def step(self, action: int):
        self.step_index += 1
        return self._obs(), 1.5, self.step_index >= 3, False, {"action": action}

    def render(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = self.step_index * 40
        frame[:, :, 1] = 80
        frame[:, :, 2] = 180
        return frame

    def close(self) -> None:
        self.closed = True

    def _obs(self):
        return np.full((4, 84, 84), self.step_index, dtype=np.uint8)


def _agent() -> DDQNAgent:
    config = load_yaml_config(CONFIG_PATH)
    return DDQNAgent(config, device="cpu", seed=42)


def _parameters(agent: DDQNAgent) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in agent.online_network.parameters()]


def test_generate_short_video_with_metadata_closes_env_and_does_not_mutate_agent(tmp_path):
    agent = _agent()
    before = _parameters(agent)
    env = SyntheticRgbEnv()
    output = tmp_path / "assault_ddqn_demo.mp4"

    summary = generate_assault_demo_video(
        agent=agent,
        env_factory=lambda: env,
        output_path=output,
        metadata={
            "project_run_id": "assault_ddqn_full_001",
            "source_checkpoint_step": 250000,
            "model_sha256": "a" * 64,
            "training_summary": {"final_global_step": 250000, "episodes_completed": 417},
        },
        seed=2026,
        epsilon=0.0,
        max_steps=3,
        fps=10,
        intro_frames=2,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert summary.metadata_path.exists()
    assert summary.reward == 4.5
    assert summary.steps == 3
    assert summary.epsilon == 0.0
    assert summary.seed == 2026
    assert summary.project_run_id == "assault_ddqn_full_001"
    assert summary.model_sha256 == "a" * 64
    assert env.closed is True
    assert all(torch.equal(left, right) for left, right in zip(before, agent.online_network.parameters()))
