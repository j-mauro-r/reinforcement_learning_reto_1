"""Tests for HU009C compact inference model artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src.agent import DDQNAgent
from src.checkpointing import CheckpointManager
from src.model_artifact import (
    FORBIDDEN_ARTIFACT_KEYS,
    SCHEMA_VERSION,
    compute_sha256,
    export_inference_model,
    load_inference_model,
    resolve_delivery_model_path,
)
from src.replay_buffer import ReplayBuffer
from src.utils import load_yaml_config


CONFIG_PATH = ASSAULT_DIR / "configs" / "ddqn_config.yaml"


def _config() -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["replay_buffer"] = {"capacity": 16, "batch_size": 4}
    config["training"] = {
        "total_timesteps": 4,
        "learning_starts": 1,
        "train_frequency": 1,
        "target_update_frequency": 2,
        "epsilon_decay_steps": 4,
    }
    return config


def _state(value: int) -> np.ndarray:
    return np.full((4, 84, 84), value, dtype=np.uint8)


def _checkpoint(tmp_path: Path, config: dict, run_id: str = "assault_ddqn_full_001") -> tuple[Path, DDQNAgent]:
    agent = DDQNAgent(config, device="cpu", seed=123)
    buffer = ReplayBuffer(capacity=16, seed=123)
    for index in range(6):
        buffer.add(_state(index), index % 7, float(index), _state(index + 1), False)
    batch = buffer.sample(4)
    agent.update(batch)
    manager = CheckpointManager(tmp_path / "checkpoints", run_id, repo_path=ASSAULT_DIR.parents[0])
    metadata = manager.save(
        agent,
        buffer,
        config,
        global_step=250000,
        training_metrics={"global_step": 250000, "episodes_completed": 417},
        save_replay_buffer=True,
    )
    return metadata.path, agent


def _q_values(agent: DDQNAgent, states: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        return agent.online_network(torch.as_tensor(states, device=agent.device)).detach().cpu()


def test_export_load_schema_metadata_no_training_state_and_checksum(tmp_path):
    config = _config()
    checkpoint_path, _ = _checkpoint(tmp_path, config)
    output_path = tmp_path / "models" / "assault_ddqn_model.pt"

    info = export_inference_model(
        checkpoint_path,
        output_path,
        project_run_id="assault_ddqn_full_001",
        config=config,
        repo_path=ASSAULT_DIR.parents[0],
        extra_metadata={"config_fingerprint": "abc123"},
    )
    payload = torch.load(output_path, map_location="cpu", weights_only=True)

    assert info.path == output_path
    assert info.size_bytes == output_path.stat().st_size
    assert info.sha256 == compute_sha256(output_path)
    assert output_path.with_suffix(".pt.sha256").exists()
    assert output_path.with_suffix(".pt.metadata.json").exists()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["metadata"]["project_run_id"] == "assault_ddqn_full_001"
    assert payload["metadata"]["source_checkpoint_step"] == 250000
    assert payload["metadata"]["config_fingerprint"] == "abc123"
    assert payload["network"] == {"input_channels": 4, "num_actions": 7, "architecture": "QNetwork"}
    assert "online_network" in payload
    assert FORBIDDEN_ARTIFACT_KEYS.isdisjoint(payload.keys())


def test_loaded_agent_matches_source_q_values_and_greedy_action(tmp_path):
    config = _config()
    checkpoint_path, source_agent = _checkpoint(tmp_path, config)
    output_path = tmp_path / "assault_ddqn_model.pt"
    info = export_inference_model(checkpoint_path, output_path, project_run_id="assault_ddqn_full_001", config=config)

    loaded_agent, loaded_info = load_inference_model(
        output_path,
        device="cpu",
        expected_sha256=info.sha256,
        expected_project_run_id="assault_ddqn_full_001",
    )
    states = np.stack([_state(0), _state(7)], axis=0)

    assert loaded_info.sha256 == info.sha256
    assert torch.allclose(_q_values(source_agent, states), _q_values(loaded_agent, states), atol=1e-6)
    assert source_agent.select_action(states[0], epsilon=0.0) == loaded_agent.select_action(states[0], epsilon=0.0)
    assert not hasattr(loaded_agent, "replay_buffer")


def test_export_rejects_incompatible_config_and_preserves_checkpoint(tmp_path):
    config = _config()
    checkpoint_path, _ = _checkpoint(tmp_path, config)
    before_hash = compute_sha256(checkpoint_path)
    incompatible = _config()
    incompatible["network"]["num_actions"] = 9

    with pytest.raises(ValueError, match="network.num_actions"):
        export_inference_model(
            checkpoint_path,
            tmp_path / "bad.pt",
            project_run_id="assault_ddqn_full_001",
            config=incompatible,
        )

    assert compute_sha256(checkpoint_path) == before_hash


def test_load_rejects_schema_metadata_checksum_and_shape_errors(tmp_path):
    config = _config()
    checkpoint_path, _ = _checkpoint(tmp_path, config)
    output_path = tmp_path / "model.pt"
    info = export_inference_model(checkpoint_path, output_path, project_run_id="assault_ddqn_full_001", config=config)
    payload = torch.load(output_path, map_location="cpu", weights_only=True)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_inference_model(output_path, expected_sha256="0" * 64)

    bad_schema = dict(payload)
    bad_schema["schema_version"] = 999
    bad_schema_path = tmp_path / "bad_schema.pt"
    torch.save(bad_schema, bad_schema_path)
    with pytest.raises(ValueError, match="schema_version"):
        load_inference_model(bad_schema_path)

    bad_metadata = dict(payload)
    bad_metadata["metadata"] = {"project_run_id": "assault_ddqn_full_001"}
    bad_metadata_path = tmp_path / "bad_metadata.pt"
    torch.save(bad_metadata, bad_metadata_path)
    with pytest.raises(ValueError, match="metadata missing"):
        load_inference_model(bad_metadata_path)

    bad_shape = dict(payload)
    bad_shape["network"] = dict(payload["network"], num_actions=9)
    bad_shape_path = tmp_path / "bad_shape.pt"
    torch.save(bad_shape, bad_shape_path)
    with pytest.raises(ValueError, match="network shape"):
        load_inference_model(bad_shape_path)

    forbidden = dict(payload)
    forbidden["optimizer"] = {"state": {}}
    forbidden_path = tmp_path / "forbidden.pt"
    torch.save(forbidden, forbidden_path)
    with pytest.raises(ValueError, match="training-only"):
        load_inference_model(forbidden_path, expected_sha256=compute_sha256(forbidden_path))

    assert info.sha256 == compute_sha256(output_path)


def test_resolve_delivery_model_path_prefers_local_assault_model_then_drive_fallback(tmp_path):
    assault_dir = tmp_path / "2_Assault"
    assault_dir.mkdir()
    drive_base = tmp_path / "drive_root"
    project_run_id = "assault_ddqn_delivery"

    local_path = assault_dir / "assault_ddqn_model.pt"
    local_path.write_bytes(b"local")
    drive_path = drive_base / "models" / project_run_id / "assault_ddqn_model.pt"
    drive_path.parent.mkdir(parents=True)
    drive_path.write_bytes(b"drive")

    resolved, source = resolve_delivery_model_path(base=drive_base, assault_dir=assault_dir, project_run_id=project_run_id)
    assert resolved == local_path
    assert source == "DELIVERY"

    local_path.unlink()
    resolved, source = resolve_delivery_model_path(base=drive_base, assault_dir=assault_dir, project_run_id=project_run_id)
    assert resolved == drive_path
    assert source == "DRIVE_FALLBACK"

    drive_path.unlink()
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_delivery_model_path(base=drive_base, assault_dir=assault_dir, project_run_id=project_run_id)
    message = str(exc_info.value)
    assert str(local_path) in message
    assert str(drive_path) in message

