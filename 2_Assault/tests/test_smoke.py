"""Smoke tests for HU002 Assault environment pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.environment import create_assault_env, get_environment_metadata, validate_frameskip_once
from src.utils import get_runtime_info, load_yaml_config


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


def _config():
    return load_yaml_config(CONFIG_PATH)


def test_imports_and_runtime_info():
    runtime = get_runtime_info()
    assert runtime["python_version"]
    assert "gymnasium_version" in runtime
    assert "ale_py_version" in runtime
    assert runtime["ram_total_gb"] > 0
    assert "gpu_available" in runtime


def test_create_env_action_space_and_observation_contract():
    env = create_assault_env(_config(), mode="train", seed=42)
    try:
        observation, info = env.reset(seed=42)
        assert env.action_space.n == 7
        assert env.unwrapped.get_action_meanings() == [
            "NOOP",
            "FIRE",
            "UP",
            "RIGHT",
            "LEFT",
            "RIGHTFIRE",
            "LEFTFIRE",
        ]
        assert observation.shape == (4, 84, 84)
        assert observation.dtype == np.uint8
        assert env.observation_space.shape == (4, 84, 84)
        assert env.observation_space.dtype == np.uint8
        assert "lives" in info
    finally:
        env.close()


def test_short_interaction_runs_without_shape_or_type_errors():
    env = create_assault_env(_config(), mode="train", seed=42)
    try:
        observation, _ = env.reset(seed=42)
        for _ in range(100):
            action = int(env.action_space.sample())
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation.shape == (4, 84, 84)
            assert observation.dtype == np.uint8
            assert isinstance(float(reward), float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            if terminated or truncated:
                observation, _ = env.reset()
        assert observation.shape == (4, 84, 84)
    finally:
        env.close()


def test_train_and_eval_share_environment_contract():
    config = _config()
    train_env = create_assault_env(config, mode="train", seed=42)
    eval_env = create_assault_env(config, mode="eval", seed=43)
    try:
        train_obs, _ = train_env.reset(seed=42)
        eval_obs, _ = eval_env.reset(seed=43)
        assert train_env.action_space == eval_env.action_space
        assert train_obs.shape == eval_obs.shape == (4, 84, 84)
        assert train_obs.dtype == eval_obs.dtype == np.uint8
    finally:
        train_env.close()
        eval_env.close()


def test_seeded_envs_have_same_contract_and_metadata():
    config = _config()
    env_a = create_assault_env(config, mode="train", seed=42)
    env_b = create_assault_env(config, mode="train", seed=42)
    try:
        obs_a, info_a = env_a.reset(seed=42)
        obs_b, info_b = env_b.reset(seed=42)
        metadata = get_environment_metadata(env_a, config, mode="train", seed=42)
        assert metadata.action_space == "Discrete(7)"
        assert metadata.observation_shape == (4, 84, 84)
        assert metadata.observation_dtype == "uint8"
        assert metadata.effective_frameskip == 4
        assert metadata.wrapper_frameskip == 1
        assert np.array_equal(obs_a, obs_b)
        assert info_a.get("seeds") == info_b.get("seeds")
    finally:
        env_a.close()
        env_b.close()


def test_frameskip_is_applied_once():
    env = create_assault_env(_config(), mode="train", seed=42)
    try:
        assert validate_frameskip_once(env, expected_frameskip=4, steps=5)
    finally:
        env.close()
