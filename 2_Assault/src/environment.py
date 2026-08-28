"""Environment factory and preprocessing wrappers for ALE/Assault-v5."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Literal, Optional, Tuple

import numpy as np
from PIL import Image

from .utils import set_global_seed

Mode = Literal["train", "eval"]


@dataclass(frozen=True)
class EnvironmentMetadata:
    """Describes the configured Assault environment contract."""

    env_id: str
    mode: Mode
    seed: int
    action_space: str
    action_meanings: Tuple[str, ...]
    observation_shape: Tuple[int, ...]
    observation_dtype: str
    base_frameskip: int
    wrapper_frameskip: int
    effective_frameskip: int
    repeat_action_probability: float
    full_action_space: bool


class GrayscaleResizeObservation:
    """Converts RGB Atari observations to grayscale 84x84 uint8 frames."""

    def __init__(self, env: Any, height: int, width: int) -> None:
        """Initializes the grayscale and resize wrapper.

        Args:
            env: Gymnasium environment to wrap.
            height: Target image height.
            width: Target image width.

        Raises:
            ValueError: If height or width are not positive.
        """
        if height <= 0 or width <= 0:
            raise ValueError("Resize height and width must be positive.")

        import gymnasium as gym

        self._wrapper = gym.ObservationWrapper(env)
        self._wrapper.observation = self.observation
        self.env = env
        self.height = height
        self.width = width
        self.action_space = env.action_space
        self.unwrapped = env.unwrapped
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.np_random = getattr(env, "np_random", None)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width),
            dtype=np.uint8,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Converts one RGB observation to a resized grayscale frame.

        Args:
            observation: Raw RGB observation from ALE.

        Returns:
            Grayscale resized frame with shape ``(height, width)`` and dtype uint8.
        """
        image = Image.fromarray(observation)
        image = image.convert("L").resize((self.width, self.height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return self.observation(observation), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


class FrameStackObservation:
    """Stacks the last N preprocessed frames without changing rewards or actions."""

    def __init__(self, env: Any, stack_size: int) -> None:
        """Initializes the frame stack wrapper.

        Args:
            env: Environment that emits two-dimensional grayscale observations.
            stack_size: Number of frames to stack.

        Raises:
            ValueError: If stack size is less than one.
        """
        if stack_size < 1:
            raise ValueError("Frame stack size must be at least 1.")

        import gymnasium as gym

        self.env = env
        self.stack_size = stack_size
        self.frames: Deque[np.ndarray] = deque(maxlen=stack_size)
        self.action_space = env.action_space
        self.unwrapped = env.unwrapped
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.np_random = getattr(env, "np_random", None)
        height, width = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(stack_size, height, width),
            dtype=np.uint8,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _get_observation(self) -> np.ndarray:
        return np.stack(tuple(self.frames), axis=0).astype(np.uint8, copy=False)

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(observation)
        return self._get_observation(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


def register_ale_if_needed() -> None:
    """Imports ALE-Py so Gymnasium can register Atari environments.

    Raises:
        ImportError: If ``ale_py`` is not installed.
    """
    import gymnasium as gym
    import ale_py

    try:
        gym.register_envs(ale_py)
    except AttributeError:
        pass


def validate_config(config: Dict[str, Any]) -> None:
    """Validates the HU002 environment configuration.

    Args:
        config: Parsed project configuration.

    Raises:
        ValueError: If required values are missing or unsupported.
    """
    environment = config.get("environment", {})
    preprocessing = config.get("preprocessing", {})

    if environment.get("id") != "ALE/Assault-v5":
        raise ValueError("HU002 only supports environment.id='ALE/Assault-v5'.")
    if environment.get("obs_type") != "rgb":
        raise ValueError("The base Assault environment must emit RGB observations.")
    if int(environment.get("frame_skip", 0)) != 4:
        raise ValueError("HU002 requires environment.frame_skip=4.")
    if float(environment.get("repeat_action_probability", -1.0)) != 0.25:
        raise ValueError("HU002 requires repeat_action_probability=0.25.")
    if bool(environment.get("full_action_space", True)):
        raise ValueError("HU002 requires full_action_space=false to keep Discrete(7).")
    if not bool(preprocessing.get("grayscale", False)):
        raise ValueError("HU002 requires grayscale preprocessing.")
    if int(preprocessing.get("resize_height", 0)) != 84:
        raise ValueError("HU002 requires resize_height=84.")
    if int(preprocessing.get("resize_width", 0)) != 84:
        raise ValueError("HU002 requires resize_width=84.")
    if int(preprocessing.get("frame_stack", 0)) != 4:
        raise ValueError("HU002 requires frame_stack=4.")
    if bool(preprocessing.get("normalize_pixels_in_env", True)):
        raise ValueError("HU002 keeps uint8 pixels; normalize later in the network.")


def create_base_env(config: Dict[str, Any], render_mode: Optional[str] = None) -> Any:
    """Creates the raw ALE/Assault-v5 environment.

    Args:
        config: Parsed project configuration.
        render_mode: Optional Gymnasium render mode override.

    Returns:
        Raw Gymnasium ALE environment configured with frameskip exactly once.

    Raises:
        ImportError: If Gymnasium or ALE-Py are unavailable.
        ValueError: If the environment cannot expose the expected action space.
    """
    import gymnasium as gym

    register_ale_if_needed()
    env_config = config["environment"]
    kwargs = {
        "obs_type": env_config["obs_type"],
        "frameskip": int(env_config["frame_skip"]),
        "repeat_action_probability": float(env_config["repeat_action_probability"]),
        "full_action_space": bool(env_config["full_action_space"]),
    }
    requested_render_mode = render_mode if render_mode is not None else env_config.get("render_mode")
    if requested_render_mode is not None:
        kwargs["render_mode"] = requested_render_mode

    env = gym.make(env_config["id"], **kwargs)
    _validate_action_space(env)
    return env


def create_assault_env(
    config: Dict[str, Any],
    mode: Mode = "train",
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Any:
    """Creates a reproducible preprocessed Assault environment.

    Args:
        config: Parsed project configuration.
        mode: Environment mode, either ``"train"`` or ``"eval"``.
        seed: Optional seed override. Defaults to ``reproducibility.seed``.
        render_mode: Optional Gymnasium render mode override.

    Returns:
        Gymnasium-compatible environment that emits stacked grayscale uint8
        observations with shape ``(4, 84, 84)``.

    Raises:
        ValueError: If configuration or mode are invalid.
    """
    if mode not in ("train", "eval"):
        raise ValueError("mode must be either 'train' or 'eval'.")

    validate_config(config)
    selected_seed = int(config["reproducibility"]["seed"] if seed is None else seed)
    set_global_seed(selected_seed)

    env = create_base_env(config, render_mode=render_mode)
    env.action_space.seed(selected_seed)

    preprocessing = config["preprocessing"]
    env = GrayscaleResizeObservation(
        env,
        height=int(preprocessing["resize_height"]),
        width=int(preprocessing["resize_width"]),
    )
    env = FrameStackObservation(env, stack_size=int(preprocessing["frame_stack"]))
    env.reset(seed=selected_seed)
    return env


def create_environment(
    config: Dict[str, Any],
    mode: Mode = "train",
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Any:
    """Creates the configured Assault environment.

    Args:
        config: Parsed project configuration.
        mode: Environment mode, either ``"train"`` or ``"eval"``.
        seed: Optional seed override.
        render_mode: Optional Gymnasium render mode override.

    Returns:
        Preprocessed Assault environment created by ``create_assault_env``.
    """
    return create_assault_env(config=config, mode=mode, seed=seed, render_mode=render_mode)


def get_environment_metadata(env: Any, config: Dict[str, Any], mode: Mode, seed: int) -> EnvironmentMetadata:
    """Builds metadata for the configured environment contract.

    Args:
        env: Preprocessed environment created by ``create_assault_env``.
        config: Parsed project configuration.
        mode: Environment mode used to create the environment.
        seed: Seed used to initialize the environment.

    Returns:
        Environment metadata useful for experiment traceability.
    """
    action_meanings = tuple(_get_action_meanings(env))
    frame_skip = int(config["environment"]["frame_skip"])
    return EnvironmentMetadata(
        env_id=str(config["environment"]["id"]),
        mode=mode,
        seed=int(seed),
        action_space=str(env.action_space),
        action_meanings=action_meanings,
        observation_shape=tuple(env.observation_space.shape),
        observation_dtype=str(env.observation_space.dtype),
        base_frameskip=frame_skip,
        wrapper_frameskip=1,
        effective_frameskip=frame_skip,
        repeat_action_probability=float(config["environment"]["repeat_action_probability"]),
        full_action_space=bool(config["environment"]["full_action_space"]),
    )


def validate_frameskip_once(env: Any, expected_frameskip: int = 4, steps: int = 5) -> bool:
    """Checks ALE frame counters to verify effective frameskip.

    Args:
        env: Preprocessed environment created by ``create_assault_env``.
        expected_frameskip: Expected increase in ALE frame counter per step.
        steps: Maximum number of environment steps to inspect.

    Returns:
        True when observed frame deltas match the expected frameskip.

    Raises:
        ValueError: If counters are unavailable or the observed delta differs.
    """
    _, info = env.reset()
    previous_frame = info.get("episode_frame_number", info.get("frame_number"))
    if previous_frame is None:
        raise ValueError("ALE frame counters are not available in info.")

    deltas = []
    for _ in range(steps):
        action = int(env.action_space.sample())
        _, _, terminated, truncated, info = env.step(action)
        current_frame = info.get("episode_frame_number", info.get("frame_number"))
        if current_frame is None:
            raise ValueError("ALE frame counters are not available after step().")
        deltas.append(int(current_frame) - int(previous_frame))
        previous_frame = current_frame
        if terminated or truncated:
            break

    if not deltas or any(delta != expected_frameskip for delta in deltas):
        raise ValueError(f"Expected frameskip deltas of {expected_frameskip}, got {deltas}.")
    return True


def _validate_action_space(env: Any) -> None:
    import gymnasium as gym

    if not isinstance(env.action_space, gym.spaces.Discrete) or env.action_space.n != 7:
        raise ValueError(f"Expected Discrete(7) action space, got {env.action_space}.")


def _get_action_meanings(env: Any) -> Tuple[str, ...]:
    if hasattr(env.unwrapped, "get_action_meanings"):
        return tuple(env.unwrapped.get_action_meanings())
    return tuple()
