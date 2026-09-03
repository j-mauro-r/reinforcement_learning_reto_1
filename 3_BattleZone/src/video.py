"""Reproducible BattleZone delivery-video generation for HU011B."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch


@dataclass(frozen=True)
class VideoSummary:
    """Persisted video identity and episode summary."""

    path: Path
    metadata_path: Path
    video_kind: str
    project_run_id: str
    seed: int
    epsilon: float
    reward: float
    steps: int
    frames: int
    fps: int


def generate_training_process_demo_video(
    *, agent: Any, env_factory: Callable[[], Any], output_path: str | Path,
    project_run_id: str, checkpoint_path: str | Path, checkpoint_step: int,
    final_step: int, checkpoint_sha256: str, epsilon: float, seed: int,
    git_sha: str, fps: int = 30, max_steps: Optional[int] = None,
    writer_factory: Optional[Callable[[Path, int], Any]] = None,
) -> VideoSummary:
    """Generates video from an explicitly selected real intermediate checkpoint."""
    if int(checkpoint_step) <= 0 or int(checkpoint_step) >= int(final_step):
        raise ValueError("Training-process video requires checkpoint_step < final_step.")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Intermediate checkpoint not found: {checkpoint_path}")
    metadata = {
        "video_kind": "training_process",
        "project_run_id": project_run_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_sha256": checkpoint_sha256,
        "epsilon": float(epsilon),
        "seed": int(seed),
        "git_sha": git_sha,
    }
    return _generate_video(
        agent=agent, env_factory=env_factory, output_path=output_path,
        metadata=metadata, seed=seed, epsilon=epsilon, fps=fps,
        max_steps=max_steps, overlay_label="TRAINING PROCESS",
        writer_factory=writer_factory,
    )


def generate_post_training_demo_video(
    *, agent: Any, env_factory: Callable[[], Any], output_path: str | Path,
    project_run_id: str, model_path: str | Path, model_sha256: str,
    seed: int, fps: int = 30, max_steps: Optional[int] = None,
    writer_factory: Optional[Callable[[Path, int], Any]] = None,
) -> VideoSummary:
    """Generates greedy video from a newly loaded standalone delivery model."""
    if Path(model_path).name != "battlezone_dqn_model.pt" or not Path(model_path).is_file():
        raise ValueError("Post-training video must use battlezone_dqn_model.pt.")
    metadata = {
        "video_kind": "post_training",
        "project_run_id": project_run_id,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "epsilon": 0.0,
        "seed": int(seed),
    }
    return _generate_video(
        agent=agent, env_factory=env_factory, output_path=output_path,
        metadata=metadata, seed=seed, epsilon=0.0, fps=fps,
        max_steps=max_steps, overlay_label="POST-TRAINING GREEDY",
        writer_factory=writer_factory,
    )


def _generate_video(
    *, agent: Any, env_factory: Callable[[], Any], output_path: str | Path,
    metadata: Mapping[str, Any], seed: int, epsilon: float, fps: int,
    max_steps: Optional[int], overlay_label: str,
    writer_factory: Optional[Callable[[Path, int], Any]],
) -> VideoSummary:
    if not str(metadata.get("project_run_id", "")).strip():
        raise ValueError("project_run_id is required.")
    if not 0.0 <= float(epsilon) <= 1.0 or fps <= 0:
        raise ValueError("epsilon and fps are invalid.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    output = Path(output_path)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Delivery video output must be MP4.")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = (writer_factory or _open_writer)(output, int(fps))
    env = None
    before = [parameter.detach().cpu().clone() for parameter in agent.online_network.parameters()]
    reward_total = 0.0
    steps = 0
    frames = 0
    try:
        env = env_factory()
        observation, _ = env.reset(seed=int(seed))
        initial = _render_rgb(env)
        writer.append_data(_overlay(initial, f"{overlay_label} | step=0 epsilon={epsilon:.3f} reward=0.0"))
        frames += 1
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(observation, epsilon=float(epsilon))
            observation, reward, terminated, truncated, _ = env.step(action)
            reward_total += float(reward)
            steps += 1
            writer.append_data(_overlay(
                _render_rgb(env),
                f"{overlay_label} | step={steps} epsilon={epsilon:.3f} reward={reward_total:.1f}",
            ))
            frames += 1
            if max_steps is not None and steps >= max_steps:
                truncated = True
    finally:
        try:
            writer.close()
        finally:
            if env is not None:
                env.close()
    after = [parameter.detach().cpu() for parameter in agent.online_network.parameters()]
    if not all(torch.equal(left, right) for left, right in zip(before, after)):
        raise RuntimeError("Video generation mutated model weights.")
    if frames <= 0 or not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("Video output is empty or missing.")
    metadata_path = output.with_suffix(".metadata.json")
    payload = dict(metadata)
    payload.update({
        "path": str(output), "reward": reward_total, "steps": steps,
        "frames": frames, "fps": int(fps),
    })
    _write_json_atomic(metadata_path, payload)
    return VideoSummary(
        output, metadata_path, str(metadata["video_kind"]),
        str(metadata["project_run_id"]), int(seed), float(epsilon),
        reward_total, steps, frames, int(fps),
    )


def _open_writer(path: Path, fps: int) -> Any:
    import imageio.v2 as imageio
    return imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1)


def _render_rgb(env: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise ValueError("render_mode='rgb_array' is required for video.")
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
        raise ValueError(f"Expected a non-empty RGB frame, got {array.shape}.")
    return array.astype(np.uint8, copy=False)


def _overlay(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
