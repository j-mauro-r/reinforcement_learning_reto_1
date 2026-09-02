"""Focused tests for the HU003 BattleZone environment pipeline."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from src.environment import (
    EXPECTED_ACTION_MEANINGS,
    create_battlezone_env,
    get_contract,
    get_wrapper_chain,
    load_config,
    seed_environment,
    validate_config,
    verify_single_frameskip,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_config_loads_and_validates(config):
    validate_config(config)
    assert config["environment"]["env_id"] == "ALE/BattleZone-v5"
    assert config["environment"]["frameskip"] == 4
    assert config["preprocessing"]["reward_transform"] == "none"
    assert config["preprocessing"]["normalize"] is False


def test_environment_creation_reset_action_space_and_final_contract(config):
    env = create_battlezone_env(config, mode="train", seed=123)
    try:
        obs, info = env.reset(seed=123)
        seed_environment(env, 123)
        assert env.action_space.n == 18
        assert env.unwrapped.get_action_meanings() == EXPECTED_ACTION_MEANINGS
        assert obs.shape == tuple(config["validation"]["expected_final_shape"])
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        contract = get_contract(env, config)
        assert contract["observation_shape"] == config["validation"]["expected_final_shape"]
        assert contract["reward_transform"] == "none"
    finally:
        env.close()


def test_frame_stack_initialization_and_shift(config):
    env = create_battlezone_env(config, mode="train", seed=456)
    try:
        obs0, _ = env.reset(seed=456)
        seed_environment(env, 456)
        assert obs0.shape[0] == config["preprocessing"]["frame_stack"]
        assert np.array_equal(obs0[0], obs0[-1])
        action = int(env.action_space.sample())
        obs1, _, _, _, _ = env.step(action)
        assert np.array_equal(obs0[1:], obs1[:-1])
    finally:
        env.close()


def test_multiple_steps_terminated_truncated_and_close(config):
    env = create_battlezone_env(config, mode="train", seed=789)
    try:
        obs, _ = env.reset(seed=789)
        seed_environment(env, 789)
        terminated = truncated = False
        for _ in range(config["validation"]["smoke_steps"]):
            obs, reward, terminated, truncated, info = env.step(int(env.action_space.sample()))
            assert obs.shape == tuple(config["validation"]["expected_final_shape"])
            assert obs.dtype == np.uint8
            assert isinstance(float(reward), float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            if terminated or truncated:
                break
    finally:
        env.close()
    assert terminated in {True, False}
    assert truncated in {True, False}


def test_train_eval_contract_parity(config):
    train_env = create_battlezone_env(config, mode="train", seed=101)
    eval_env = create_battlezone_env(config, mode="eval", seed=202)
    try:
        train_contract = get_contract(train_env, config)
        eval_contract = get_contract(eval_env, config)
        parity_keys = [
            "env_id",
            "frameskip",
            "repeat_action_probability",
            "action_space",
            "pipeline_name",
            "color_mode",
            "resize",
            "crop_enabled",
            "frame_stack",
            "observation_shape",
            "observation_dtype",
            "reward_transform",
        ]
        for key in parity_keys:
            assert train_contract[key] == eval_contract[key]
    finally:
        train_env.close()
        eval_env.close()


def test_action_space_seed_reproducibility(config):
    env_a = create_battlezone_env(config, mode="train", seed=303)
    env_b = create_battlezone_env(config, mode="train", seed=303)
    try:
        env_a.reset(seed=303)
        env_b.reset(seed=303)
        seed_environment(env_a, 303)
        seed_environment(env_b, 303)
        actions_a = [int(env_a.action_space.sample()) for _ in range(16)]
        actions_b = [int(env_b.action_space.sample()) for _ in range(16)]
        assert actions_a == actions_b
    finally:
        env_a.close()
        env_b.close()


def test_reward_passthrough_against_raw_env(config):
    seed = 404
    actions = [0, 1, 3, 10, 17]
    env_cfg = config["environment"]
    raw_env = gym.make(
        env_cfg["env_id"],
        obs_type=env_cfg["obs_type"],
        frameskip=env_cfg["frameskip"],
        repeat_action_probability=env_cfg["repeat_action_probability"],
        mode=env_cfg["mode"],
        difficulty=env_cfg["difficulty"],
        render_mode=None,
    )
    wrapped_env = create_battlezone_env(config, mode="train", seed=seed)
    try:
        raw_env.reset(seed=seed)
        wrapped_env.reset(seed=seed)
        seed_environment(raw_env, seed)
        seed_environment(wrapped_env, seed)
        raw_rewards = []
        wrapped_rewards = []
        for action in actions:
            _, reward_raw, term_raw, trunc_raw, _ = raw_env.step(action)
            _, reward_wrapped, term_wrapped, trunc_wrapped, _ = wrapped_env.step(action)
            raw_rewards.append(float(reward_raw))
            wrapped_rewards.append(float(reward_wrapped))
            assert term_raw == term_wrapped
            assert trunc_raw == trunc_wrapped
            if term_raw or trunc_raw:
                break
        assert raw_rewards == wrapped_rewards
    finally:
        raw_env.close()
        wrapped_env.close()


def test_single_frameskip_and_wrapper_chain(config):
    env = create_battlezone_env(config, mode="train", seed=505)
    try:
        report = verify_single_frameskip(env, config, steps=3)
        assert report["counter_check_passed"] is True
        assert report["has_action_repeat_wrapper"] is False
        assert "FrameStackObservation" in report["wrapper_chain"]
        assert "BattleZonePreprocessObservation" in report["wrapper_chain"]
    finally:
        env.close()


def test_no_assault_references_in_hu003_files():
    battlezone_root = Path(__file__).resolve().parents[1]
    checked = [
        battlezone_root / "src",
        battlezone_root / "configs",
        battlezone_root / "pipeline_battlezone.ipynb",
    ]
    references = []
    for path in checked:
        if not path.exists():
            continue
        files = [path] if path.is_file() else path.rglob("*")
        for file_path in files:
            if file_path.is_file():
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                if "2_Assault" in text or "experimento_0_battlezone.ipynb" in text:
                    references.append(str(file_path))
    assert references == []
