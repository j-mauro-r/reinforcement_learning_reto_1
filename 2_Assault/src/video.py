"""MP4 generation helpers for Assault DDQN delivery videos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class VideoSummary:
    """Metadata produced by a HU009C gameplay video render."""

    path: Path
    metadata_path: Path
    seed: int
    epsilon: float
    reward: float
    steps: int
    project_run_id: str
    model_sha256: str
    fps: int

    def as_dict(self) -> Dict[str, Any]:
        """Returns a serializable representation."""
        return {
            "path": str(self.path),
            "metadata_path": str(self.metadata_path),
            "seed": self.seed,
            "epsilon": self.epsilon,
            "reward": self.reward,
            "steps": self.steps,
            "project_run_id": self.project_run_id,
            "model_sha256": self.model_sha256,
            "fps": self.fps,
        }


def generate_assault_demo_video(
    agent: Any,
    env_factory: Callable[[], Any],
    output_path: str | Path,
    metadata: Mapping[str, Any],
    seed: int,
    epsilon: float = 0.0,
    max_steps: Optional[int] = None,
    fps: int = 30,
    intro_frames: int = 45,
    overlay_label: Optional[str] = None,
) -> VideoSummary:
    """Generates a reproducible MP4 with training evidence and gameplay.

    Args:
        agent: Loaded inference agent exposing ``select_action``.
        env_factory: Zero-argument factory returning an env with RGB rendering.
        output_path: Destination MP4 path.
        metadata: Lineage metadata including ``project_run_id`` and
            ``model_sha256``.
        seed: Environment reset seed.
        epsilon: Evaluation epsilon, normally ``0.0``.
        max_steps: Optional cap used for short demos or tests.
        fps: Output frames per second.
        intro_frames: Number of synthetic intro frames with real run metadata.
        overlay_label: Optional short label prepended to gameplay overlays.

    Returns:
        Video summary with reward, steps and metadata path.

    Raises:
        ValueError: If inputs are invalid or rendered frames are unavailable.
    """
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")
    if int(fps) <= 0:
        raise ValueError("fps must be positive.")
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("max_steps must be positive when provided.")
    project_run_id = str(metadata.get("project_run_id", "")).strip()
    model_sha256 = str(metadata.get("model_sha256", "")).strip()
    if not project_run_id:
        raise ValueError("metadata.project_run_id is required.")
    if not model_sha256:
        raise ValueError("metadata.model_sha256 is required.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(output, fps=int(fps))
    env = env_factory()
    reward_total = 0.0
    steps = 0
    online_before = _clone_parameters(getattr(agent, "online_network", None))
    try:
        observation, _ = env.reset(seed=int(seed))
        first_frame = _render_frame(env)
        frame_size = (int(first_frame.shape[1]), int(first_frame.shape[0]))
        for frame in _intro_frames(metadata, count=int(intro_frames), size=frame_size):
            writer.append_data(frame)
        overlay_prefix = f"{overlay_label} - " if overlay_label else ""
        writer.append_data(_overlay(first_frame, f"{overlay_prefix}seed={seed} epsilon={epsilon:.2f} reward=0.0 step=0"))
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.select_action(observation, epsilon=float(epsilon))
            observation, reward, terminated, truncated, _ = env.step(action)
            reward_total += float(reward)
            steps += 1
            frame = _render_frame(env)
            writer.append_data(_overlay(frame, f"{overlay_prefix}reward={reward_total:.1f} step={steps} epsilon={epsilon:.2f}"))
            if max_steps is not None and steps >= int(max_steps):
                truncated = True
    finally:
        writer.close()
        env.close()

    if online_before is not None and not _parameters_match(online_before, getattr(agent, "online_network", None)):
        raise RuntimeError("Video generation mutated the agent online network.")

    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    summary = VideoSummary(
        path=output,
        metadata_path=metadata_path,
        seed=int(seed),
        epsilon=float(epsilon),
        reward=float(reward_total),
        steps=int(steps),
        project_run_id=project_run_id,
        model_sha256=model_sha256,
        fps=int(fps),
    )
    metadata_payload = dict(metadata)
    metadata_payload.update(summary.as_dict())
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
    if not output.exists() or output.stat().st_size <= 0:
        raise ValueError(f"Video file was not produced: {output}")
    return summary


def generate_training_process_demo_video(
    agent: Any,
    env_factory: Callable[[], Any],
    output_path: str | Path,
    metadata: Mapping[str, Any],
    seed: int,
    epsilon: float,
    max_steps: Optional[int] = None,
    fps: int = 30,
    intro_frames: int = 30,
) -> VideoSummary:
    """Generates a short exploratory-policy MP4 using the shared video path."""
    enriched_metadata = dict(metadata)
    enriched_metadata.update(
        {
            "video_kind": "training_process_exploration",
            "represents_intermediate_checkpoint": bool(metadata.get("intermediate_checkpoint_path")),
            "exploration_epsilon": float(epsilon),
        }
    )
    checkpoint_step = enriched_metadata.get("source_checkpoint_step")
    represents_intermediate = bool(enriched_metadata["represents_intermediate_checkpoint"])
    label = (
        f"Entrenamiento/exploracion timestep {checkpoint_step}"
        if represents_intermediate and checkpoint_step
        else "Demostracion exploratoria de entrenamiento"
    )
    return generate_assault_demo_video(
        agent=agent,
        env_factory=env_factory,
        output_path=output_path,
        metadata=enriched_metadata,
        seed=seed,
        epsilon=epsilon,
        max_steps=max_steps,
        fps=fps,
        intro_frames=intro_frames,
        overlay_label=label,
    )


def _open_writer(path: Path, fps: int) -> Any:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("Install imageio[ffmpeg] to generate HU009C MP4 videos.") from exc
    return imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1)


def _intro_frames(metadata: Mapping[str, Any], count: int, size: tuple[int, int]) -> Iterable[np.ndarray]:
    is_exploration_demo = metadata.get("video_kind") == "training_process_exploration"
    represents_intermediate = bool(metadata.get("represents_intermediate_checkpoint"))
    if is_exploration_demo and not represents_intermediate:
        lines = [
            "Assault DDQN - demostracion exploratoria",
            f"run: {metadata.get('project_run_id', '<unknown>')}",
            "checkpoint intermedio: no disponible",
            f"model sha256: {str(metadata.get('model_sha256', '<unknown>'))[:16]}...",
            f"exploration epsilon: {metadata.get('exploration_epsilon', metadata.get('epsilon', 0.0))}",
        ]
    else:
        lines = [
            "Assault DDQN - evidencia de entrenamiento",
            f"run: {metadata.get('project_run_id', '<unknown>')}",
            f"checkpoint step: {metadata.get('source_checkpoint_step', '<unknown>')}",
            f"model sha256: {str(metadata.get('model_sha256', '<unknown>'))[:16]}...",
            f"evaluation epsilon: {metadata.get('epsilon', 0.0)}",
        ]
    training = metadata.get("training_summary")
    if isinstance(training, Mapping):
        lines.append(f"timesteps: {training.get('final_global_step', training.get('global_step', '<unknown>'))}")
        lines.append(f"episodes: {training.get('episodes_completed', '<unknown>')}")
    frame = _text_frame(lines, size=size)
    for _ in range(max(1, int(count))):
        yield frame.copy()


def _text_frame(lines: list[str], size: tuple[int, int] = (640, 360)) -> np.ndarray:
    image = Image.new("RGB", size, color=(18, 24, 31))
    draw = ImageDraw.Draw(image)
    y = 34
    for index, line in enumerate(lines):
        fill = (238, 242, 247) if index == 0 else (197, 210, 222)
        draw.text((34, y), line, fill=fill)
        y += 42 if index == 0 else 30
    return np.asarray(image, dtype=np.uint8)


def _render_frame(env: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise ValueError("Environment render() returned None; use render_mode='rgb_array'.")
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB frame, got shape {array.shape}.")
    return array


def _overlay(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _clone_parameters(network: Any) -> Optional[list[Any]]:
    if network is None or not hasattr(network, "parameters"):
        return None
    return [parameter.detach().clone() for parameter in network.parameters()]


def _parameters_match(before: list[Any], network: Any) -> bool:
    if network is None or not hasattr(network, "parameters"):
        return True
    try:
        import torch
    except ImportError:
        return True
    return all(torch.equal(left, right) for left, right in zip(before, network.parameters()))
