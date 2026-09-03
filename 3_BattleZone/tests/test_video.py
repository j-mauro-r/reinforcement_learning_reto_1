"""Tests for reproducible HU011B BattleZone videos."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.video import (
    generate_post_training_demo_video, generate_training_process_demo_video,
)


RUN_ID = "battlezone-dqn-test-run"


class FakeAgent:
    def __init__(self):
        self.online_network = torch.nn.Linear(1, 2)

    def select_action(self, state, epsilon=0.0):
        return 0


class FakeEnv:
    def __init__(self, invalid_frame=False, fail_step=False):
        self.invalid_frame = invalid_frame
        self.fail_step = fail_step
        self.closed = False
        self.cursor = 0

    def reset(self, seed=None):
        self.seed = seed
        return np.zeros((4, 128, 128, 3), dtype=np.uint8), {}

    def render(self):
        return np.zeros((8, 8), dtype=np.uint8) if self.invalid_frame else np.zeros((32, 48, 3), dtype=np.uint8)

    def step(self, action):
        if self.fail_step:
            raise RuntimeError("controlled failure")
        self.cursor += 1
        return np.zeros((4, 128, 128, 3), dtype=np.uint8), 1.0, self.cursor >= 2, False, {}

    def close(self):
        self.closed = True


class FakeWriter:
    def __init__(self, path):
        self.path = path
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(np.asarray(frame))

    def close(self):
        self.closed = True
        self.path.write_bytes(b"fake-mp4" if self.frames else b"")


def _writer_factory(holder):
    def factory(path, fps):
        holder.append(FakeWriter(path))
        return holder[-1]
    return factory


def test_training_video_requires_intermediate_checkpoint_and_preserves_lineage(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    env = FakeEnv()
    writers = []
    agent = FakeAgent()
    weights = [parameter.detach().clone() for parameter in agent.online_network.parameters()]
    summary = generate_training_process_demo_video(
        agent=agent, env_factory=lambda: env,
        output_path=tmp_path / "battlezone_dqn_training_process.mp4",
        project_run_id=RUN_ID, checkpoint_path=checkpoint, checkpoint_step=250_000,
        final_step=1_000_000, checkpoint_sha256="a" * 64, epsilon=0.05,
        seed=77, git_sha="b" * 40, max_steps=2,
        writer_factory=_writer_factory(writers),
    )
    metadata = json.loads(summary.metadata_path.read_text())
    assert summary.path.suffix == ".mp4" and summary.path.stat().st_size > 0
    assert metadata["video_kind"] == "training_process"
    assert metadata["checkpoint_step"] == 250_000
    assert metadata["project_run_id"] == RUN_ID and metadata["epsilon"] == 0.05
    assert metadata["seed"] == 77 and metadata["frames"] > 0
    assert env.closed and writers[0].closed
    assert all(torch.equal(left, right) for left, right in zip(weights, agent.online_network.parameters()))
    with pytest.raises(ValueError, match="checkpoint_step"):
        generate_training_process_demo_video(
            agent=agent, env_factory=FakeEnv, output_path=tmp_path / "bad.mp4",
            project_run_id=RUN_ID, checkpoint_path=checkpoint,
            checkpoint_step=1_000_000, final_step=1_000_000,
            checkpoint_sha256="a" * 64, epsilon=0.0, seed=1, git_sha="b" * 40,
        )


def test_post_training_video_requires_delivery_model_and_epsilon_zero(tmp_path):
    model = tmp_path / "battlezone_dqn_model.pt"
    model.write_bytes(b"model")
    env = FakeEnv()
    writers = []
    summary = generate_post_training_demo_video(
        agent=FakeAgent(), env_factory=lambda: env,
        output_path=tmp_path / "battlezone_dqn_post_training.mp4",
        project_run_id=RUN_ID, model_path=model, model_sha256="c" * 64,
        seed=88, max_steps=2, writer_factory=_writer_factory(writers),
    )
    metadata = json.loads(summary.metadata_path.read_text())
    assert summary.epsilon == 0.0 and metadata["epsilon"] == 0.0
    assert metadata["model_path"] == str(model)
    assert metadata["video_kind"] == "post_training"
    assert env.closed


def test_invalid_frame_and_step_failure_close_resources(tmp_path):
    model = tmp_path / "battlezone_dqn_model.pt"
    model.write_bytes(b"model")
    for env, match in ((FakeEnv(invalid_frame=True), "RGB"), (FakeEnv(fail_step=True), "controlled")):
        writers = []
        with pytest.raises((ValueError, RuntimeError), match=match):
            generate_post_training_demo_video(
                agent=FakeAgent(), env_factory=lambda env=env: env,
                output_path=tmp_path / f"failure-{match}.mp4",
                project_run_id=RUN_ID, model_path=model, model_sha256="c" * 64,
                seed=1, writer_factory=_writer_factory(writers),
            )
        assert env.closed and writers[0].closed
