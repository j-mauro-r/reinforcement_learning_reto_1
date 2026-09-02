"""Environment factory and preprocessing wrappers for BattleZone HU003."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from PIL import Image

try:
    import ale_py  # noqa: F401
except ImportError:  # pragma: no cover - validation reports the actionable error.
    ale_py = None


EXPECTED_ACTION_MEANINGS = [
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "battlezone_config.yaml"


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads the BattleZone environment configuration.

    Args:
        path: Optional path to a YAML configuration file. The HU003 default is
            used when omitted.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration is invalid.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validates the HU003 environment/preprocessing configuration.

    Args:
        config: Configuration dictionary loaded from YAML.

    Raises:
        ValueError: If required keys or values are invalid.
    """
    required_sections = {"environment", "preprocessing", "modes", "validation"}
    missing = required_sections.difference(config or {})
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    env = config["environment"]
    preprocessing = config["preprocessing"]

    if env.get("env_id") != "ALE/BattleZone-v5":
        raise ValueError("HU003 only supports ALE/BattleZone-v5.")
    if int(env.get("expected_action_space_n", -1)) != 18:
        raise ValueError("BattleZone action space must remain Discrete(18).")
    if int(env.get("frameskip", -1)) != 4:
        raise ValueError("BattleZone HU003 requires frameskip=4 exactly once.")
    if float(env.get("repeat_action_probability", -1)) != 0.25:
        raise ValueError("BattleZone HU003 requires repeat_action_probability=0.25.")
    if env.get("obs_type") != "rgb":
        raise ValueError("Raw BattleZone observation must be rgb for HU003.")

    if preprocessing.get("reward_transform") != "none":
        raise ValueError("HU003 must not transform rewards.")
    if bool(preprocessing.get("normalize")):
        raise ValueError("HU003 keeps observations as uint8 and does not normalize.")
    if preprocessing.get("dtype") != "uint8":
        raise ValueError("HU003 final observation dtype must be uint8.")

    color_mode = preprocessing.get("color_mode")
    if color_mode not in {"rgb", "grayscale"}:
        raise ValueError("preprocessing.color_mode must be rgb or grayscale.")

    frame_stack = int(preprocessing.get("frame_stack", 0))
    if frame_stack < 1:
        raise ValueError("preprocessing.frame_stack must be >= 1.")

    resize = preprocessing.get("resize", {})
    if bool(resize.get("enabled")):
        if int(resize.get("width", 0)) <= 0 or int(resize.get("height", 0)) <= 0:
            raise ValueError("Enabled resize requires positive width and height.")

    crop = preprocessing.get("crop", {})
    if bool(crop.get("enabled")) and int(crop.get("top", 0)) > 0:
        raise ValueError("Cropping that removes the radar region is not allowed in HU003.")


def seed_environment(env: gym.Env, seed: int) -> None:
    """Seeds controllable Gymnasium spaces used by the pipeline.

    Args:
        env: Environment or wrapper instance.
        seed: Seed applied to action and observation spaces when available.
    """
    env.action_space.seed(seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class BattleZonePreprocessObservation(gym.ObservationWrapper):
    """Applies BattleZone-specific visual preprocessing without reward changes."""

    def __init__(self, env: gym.Env, preprocessing_config: Dict[str, Any]):
        """Initializes visual preprocessing.

        Args:
            env: Raw BattleZone environment.
            preprocessing_config: Preprocessing section from configuration.
        """
        super().__init__(env)
        self.preprocessing_config = deepcopy(preprocessing_config)
        self.color_mode = preprocessing_config["color_mode"]
        self.crop_config = preprocessing_config["crop"]
        self.resize_config = preprocessing_config["resize"]

        height, width, channels = self._compute_output_shape()
        if self.color_mode == "grayscale":
            self.observation_space = spaces.Box(0, 255, shape=(height, width), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(0, 255, shape=(height, width, channels), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Transforms one raw RGB frame into the configured visual format."""
        return _preprocess_frame_array(observation, self.preprocessing_config)

    def _compute_output_shape(self) -> Tuple[int, int, int]:
        source_shape = self.env.observation_space.shape
        if source_shape != (210, 160, 3):
            raise ValueError(f"Unexpected raw observation shape: {source_shape}")

        if bool(self.crop_config.get("enabled")):
            height = int(self.crop_config["bottom"]) - int(self.crop_config["top"])
            width = int(self.crop_config["right"]) - int(self.crop_config["left"])
        else:
            height, width = int(source_shape[0]), int(source_shape[1])

        if bool(self.resize_config.get("enabled")):
            height = int(self.resize_config["height"])
            width = int(self.resize_config["width"])

        return height, width, 3


class FrameStackObservation(gym.Wrapper):
    """Stacks recent observations without repeating actions or altering rewards."""

    def __init__(self, env: gym.Env, num_stack: int):
        """Initializes a frame stack wrapper.

        Args:
            env: Environment producing preprocessed single-frame observations.
            num_stack: Number of most recent observations in each state.

        Raises:
            ValueError: If num_stack is less than one.
        """
        if num_stack < 1:
            raise ValueError("num_stack must be >= 1.")
        super().__init__(env)
        self.num_stack = int(num_stack)
        self.frames: deque[np.ndarray] = deque(maxlen=self.num_stack)
        low = np.repeat(env.observation_space.low[np.newaxis, ...], self.num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], self.num_stack, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and fills the stack with the initial frame."""
        observation, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(np.array(observation, copy=True))
        return self._get_observation(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Steps once in the underlying environment and shifts the stack."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.array(observation, copy=True))
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        return np.stack(tuple(self.frames), axis=0).astype(self.observation_space.dtype, copy=False)


def create_battlezone_env(
    config: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "train",
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Creates the single approved BattleZone environment pipeline.

    Args:
        config: Optional configuration dictionary. The versioned HU003 config is
            loaded when omitted.
        mode: Either "train" or "eval"; both share the same perceptual contract.
        seed: Optional explicit seed. If omitted, config seed plus mode offset is used.
        render_mode: Optional render override for visual inspection/video.

    Returns:
        Configured Gymnasium environment with preprocessing and frame stacking.

    Raises:
        ValueError: If the mode or resulting contract is invalid.
    """
    cfg = deepcopy(config) if config is not None else load_config()
    validate_config(cfg)
    if mode not in cfg["modes"]:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {sorted(cfg['modes'])}.")

    env_cfg = cfg["environment"]
    mode_cfg = cfg["modes"][mode]
    effective_seed = int(seed if seed is not None else int(env_cfg["seed"]) + int(mode_cfg["seed_offset"]))
    effective_render_mode = render_mode if render_mode is not None else mode_cfg.get("render_mode", env_cfg.get("render_mode"))

    raw_env = gym.make(
        env_cfg["env_id"],
        obs_type=env_cfg["obs_type"],
        frameskip=int(env_cfg["frameskip"]),
        repeat_action_probability=float(env_cfg["repeat_action_probability"]),
        mode=int(env_cfg["mode"]),
        difficulty=int(env_cfg["difficulty"]),
        render_mode=effective_render_mode,
    )
    seed_environment(raw_env, effective_seed)
    validate_raw_contract(raw_env, cfg)

    env: gym.Env = BattleZonePreprocessObservation(raw_env, cfg["preprocessing"])
    env = FrameStackObservation(env, int(cfg["preprocessing"]["frame_stack"]))
    seed_environment(env, effective_seed)

    observation, _ = env.reset(seed=effective_seed)
    seed_environment(env, effective_seed)
    validate_final_contract(env, cfg, observation)
    return env


def validate_raw_contract(env: gym.Env, config: Dict[str, Any]) -> None:
    """Validates the raw ALE BattleZone contract before preprocessing."""
    expected_raw_shape = tuple(config["validation"]["expected_raw_shape"])
    if tuple(env.observation_space.shape) != expected_raw_shape:
        raise ValueError(f"Unexpected raw shape: {env.observation_space.shape}")
    if env.observation_space.dtype != np.uint8:
        raise ValueError(f"Unexpected raw dtype: {env.observation_space.dtype}")
    validate_action_space(env, config)


def validate_final_contract(env: gym.Env, config: Dict[str, Any], observation: Optional[np.ndarray] = None) -> None:
    """Validates final observation/action contracts after wrappers."""
    expected_shape = tuple(config["validation"]["expected_final_shape"])
    if tuple(env.observation_space.shape) != expected_shape:
        raise ValueError(f"Unexpected final observation space shape: {env.observation_space.shape}")
    if env.observation_space.dtype != np.uint8:
        raise ValueError(f"Unexpected final dtype: {env.observation_space.dtype}")
    if observation is not None:
        if tuple(observation.shape) != expected_shape:
            raise ValueError(f"Unexpected final observation shape: {observation.shape}")
        if observation.dtype != np.uint8:
            raise ValueError(f"Unexpected final observation dtype: {observation.dtype}")
    validate_action_space(env, config)


def validate_action_space(env: gym.Env, config: Dict[str, Any]) -> None:
    """Validates Discrete(18) and action meanings when available."""
    expected_n = int(config["environment"]["expected_action_space_n"])
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError(f"Expected Discrete action space, got {env.action_space}.")
    if int(env.action_space.n) != expected_n:
        raise ValueError(f"Expected Discrete({expected_n}), got Discrete({env.action_space.n}).")

    unwrapped = env.unwrapped
    if hasattr(unwrapped, "get_action_meanings"):
        meanings = list(unwrapped.get_action_meanings())
        if len(meanings) != expected_n:
            raise ValueError(f"Expected {expected_n} action meanings, got {len(meanings)}.")
        if meanings != EXPECTED_ACTION_MEANINGS:
            raise ValueError(f"Unexpected action meanings: {meanings}")


def get_contract(env: gym.Env, config: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a structured environment contract for reports and tests."""
    env_cfg = config["environment"]
    preprocessing = config["preprocessing"]
    shape = tuple(int(x) for x in env.observation_space.shape)
    bytes_per_state = int(np.prod(shape) * np.dtype(env.observation_space.dtype).itemsize)
    return {
        "env_id": env_cfg["env_id"],
        "mode": int(env_cfg["mode"]),
        "difficulty": int(env_cfg["difficulty"]),
        "frameskip": int(env_cfg["frameskip"]),
        "repeat_action_probability": float(env_cfg["repeat_action_probability"]),
        "action_space": str(env.action_space),
        "action_meanings": list(env.unwrapped.get_action_meanings()) if hasattr(env.unwrapped, "get_action_meanings") else None,
        "pipeline_name": preprocessing["pipeline_name"],
        "color_mode": preprocessing["color_mode"],
        "resize": [int(preprocessing["resize"]["height"]), int(preprocessing["resize"]["width"])],
        "crop_enabled": bool(preprocessing["crop"]["enabled"]),
        "frame_stack": int(preprocessing["frame_stack"]),
        "observation_shape": list(shape),
        "observation_dtype": str(env.observation_space.dtype),
        "observation_range": [int(env.observation_space.low.min()), int(env.observation_space.high.max())],
        "bytes_per_state": bytes_per_state,
        "mb_per_state": round(bytes_per_state / (1024 * 1024), 4),
        "reward_transform": preprocessing["reward_transform"],
        "normalization": bool(preprocessing["normalize"]),
    }


def verify_single_frameskip(env: gym.Env, config: Dict[str, Any], steps: int = 3) -> Dict[str, Any]:
    """Checks wrapper chain and ALE frame counters for accidental action repeat.

    Args:
        env: Configured BattleZone environment.
        config: HU003 configuration.
        steps: Number of pipeline steps to probe.

    Returns:
        Dictionary with wrapper names, frame deltas and validation status.
    """
    seed = int(config["environment"]["seed"]) + 777
    observation, info = env.reset(seed=seed)
    seed_environment(env, seed)
    frame_numbers = [int(info["frame_number"])] if isinstance(info, dict) and "frame_number" in info else []
    for _ in range(steps):
        action = int(env.action_space.sample())
        observation, reward, terminated, truncated, info = env.step(action)
        if isinstance(info, dict) and "frame_number" in info:
            frame_numbers.append(int(info["frame_number"]))
        if terminated or truncated:
            break

    deltas = [b - a for a, b in zip(frame_numbers, frame_numbers[1:])]
    expected = int(config["environment"]["frameskip"])
    return {
        "wrapper_chain": get_wrapper_chain(env),
        "frame_numbers": frame_numbers,
        "frame_deltas": deltas,
        "expected_delta": expected,
        "has_action_repeat_wrapper": any("repeat" in name.lower() or "frameskip" in name.lower() for name in get_wrapper_chain(env)[1:]),
        "counter_check_passed": bool(deltas) and all(delta == expected for delta in deltas),
    }


def get_wrapper_chain(env: gym.Env) -> list[str]:
    """Returns wrapper class names from outermost wrapper to raw environment."""
    chain = []
    current = env
    while True:
        chain.append(current.__class__.__name__)
        if not hasattr(current, "env"):
            break
        current = current.env
    return chain


def preprocess_frame(frame: np.ndarray, preprocessing_config: Dict[str, Any]) -> np.ndarray:
    """Applies HU003 preprocessing to one frame outside an environment wrapper."""
    return _preprocess_frame_array(frame, preprocessing_config)


def build_candidate_config(base_config: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a preprocessing config from a candidate entry for visual comparison."""
    cfg = deepcopy(base_config["preprocessing"])
    cfg["pipeline_name"] = candidate["name"]
    cfg["color_mode"] = candidate["color_mode"]
    cfg["resize"]["enabled"] = True
    cfg["resize"]["width"] = int(candidate["resize"][0])
    cfg["resize"]["height"] = int(candidate["resize"][1])
    cfg["crop"]["enabled"] = bool(candidate["crop_enabled"])
    cfg["frame_stack"] = int(candidate["frame_stack"])
    return cfg


def _pil_resample(name: str) -> int:
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported interpolation: {name}")
    return mapping[name]


def _preprocess_frame_array(frame: np.ndarray, preprocessing_config: Dict[str, Any]) -> np.ndarray:
    source = np.asarray(frame, dtype=np.uint8)
    if source.shape != (210, 160, 3):
        raise ValueError(f"Expected raw RGB BattleZone frame (210, 160, 3), got {source.shape}.")

    crop_config = preprocessing_config["crop"]
    if bool(crop_config.get("enabled")):
        top = int(crop_config["top"])
        bottom = int(crop_config["bottom"])
        left = int(crop_config["left"])
        right = int(crop_config["right"])
        source = source[top:bottom, left:right]

    image = Image.fromarray(source, mode="RGB")
    if preprocessing_config["color_mode"] == "grayscale":
        image = image.convert("L")

    resize_config = preprocessing_config["resize"]
    if bool(resize_config.get("enabled")):
        width = int(resize_config["width"])
        height = int(resize_config["height"])
        interpolation = resize_config.get("interpolation", "bilinear")
        image = image.resize((width, height), resample=_pil_resample(interpolation))

    return np.asarray(image, dtype=np.uint8)
