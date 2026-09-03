"""Tests for compact HU011B BattleZone inference artifacts."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agent import DQNAgent
from src.delivery import (
    build_delivery_paths, evaluate_delivery_gate, resolve_latest_full_checkpoint,
    write_delivery_manifest,
)
from src.environment import load_config
from src.model_artifact import (
    MAX_MODEL_BYTES, compute_sha256, export_inference_model,
    load_inference_model, materialize_model_copies, resolve_delivery_model_path,
    validate_model_artifact,
)
from src.persistence import (
    build_checkpoint_metadata, build_checkpoint_payload,
    checkpoint_config_snapshot, save_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "3_BattleZone/configs/battlezone_config.yaml"
RUN_ID = "battlezone-dqn-test-run"


def test_latest_full_checkpoint_prefers_highest_intermediate_step(tmp_path):
    for step in (250_000, 500_000, 750_000, 1_000_000):
        (tmp_path / f"battlezone_dqn_step_{step:08d}_full.pt").touch()
    (tmp_path / "battlezone_dqn_step_00900000_lightweight.pt").touch()

    path, step = resolve_latest_full_checkpoint(tmp_path, final_step=1_000_000)

    assert step == 750_000
    assert path.name == "battlezone_dqn_step_00750000_full.pt"


def test_latest_full_checkpoint_falls_back_and_fails_clearly(tmp_path):
    fallback = tmp_path / "battlezone_dqn_step_00500000_full.pt"
    fallback.touch()
    (tmp_path / "battlezone_dqn_step_00750000_lightweight.pt").touch()

    assert resolve_latest_full_checkpoint(tmp_path, 1_000_000) == (fallback, 500_000)

    fallback.unlink()
    with pytest.raises(RuntimeError, match="No FULL checkpoint"):
        resolve_latest_full_checkpoint(tmp_path, 1_000_000)


@pytest.fixture()
def source_checkpoint(tmp_path):
    config = load_config(CONFIG_PATH)
    agent = DQNAgent.from_config(config)
    metadata = build_checkpoint_metadata(
        checkpoint_mode="lightweight", algorithm="DQN", global_step=100,
        episode_index=2, seed=7, state_shape=agent.state_shape,
        action_dim=agent.action_dim, batch_size=agent.batch_size,
    )
    payload = build_checkpoint_payload(
        metadata=metadata,
        trainer_state={"global_step": 100, "episode_index": 2},
        agent_state=agent.state_dict(), replay_buffer_state=None,
        config_snapshot=checkpoint_config_snapshot(config),
    )
    path = save_checkpoint(checkpoint_path=tmp_path / "source.pt", payload=payload)
    return path, config, agent


def test_export_is_compact_atomic_and_round_trips_on_cpu(source_checkpoint, tmp_path):
    source, config, original = source_checkpoint
    output = tmp_path / "battlezone_dqn_model.pt"
    info = export_inference_model(
        checkpoint_path=source, output_path=output, project_run_id=RUN_ID,
        config=config, training_git_sha="a" * 40,
    )
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert set(payload) == {"schema_version", "created_at", "network", "metadata", "online_network"}
    assert not {"replay_buffer", "replay_buffer_state", "optimizer", "target_network", "trainer_state"} & set(payload)
    assert info.sha256 == compute_sha256(output)
    assert info.size_bytes < MAX_MODEL_BYTES
    assert info.metadata["project_run_id"] == RUN_ID
    assert info.metadata["source_checkpoint_step"] == 100
    assert output.with_suffix(".pt.sha256").is_file()
    sidecar = json.loads(output.with_name("battlezone_dqn_model.metadata.json").read_text())
    assert sidecar["model_sha256"] == info.sha256
    assert not list(tmp_path.glob(".*.tmp"))

    loaded, loaded_info = load_inference_model(
        output, map_location="cpu", expected_sha256=info.sha256,
        expected_project_run_id=RUN_ID, expected_config=config,
    )
    for expected, actual in zip(original.online_network.parameters(), loaded.online_network.parameters()):
        assert torch.equal(expected.detach().cpu(), actual.detach().cpu())
    action = loaded.select_action(np.zeros((4, 128, 128, 3), dtype=np.uint8), epsilon=0.0)
    assert 0 <= action < 18
    assert loaded_info.path == output
    assert not hasattr(loaded, "optimizer") and not hasattr(loaded, "replay_buffer")

    with pytest.raises(ValueError, match="global_step mismatch"):
        export_inference_model(
            checkpoint_path=source, output_path=tmp_path / "wrong-step.pt",
            project_run_id=RUN_ID, config=config, training_git_sha="a" * 40,
            expected_source_step=1_000_000,
        )


def test_checksum_contract_and_size_failures(source_checkpoint, tmp_path):
    source, config, _ = source_checkpoint
    output = tmp_path / "battlezone_dqn_model.pt"
    info = export_inference_model(
        checkpoint_path=source, output_path=output, project_run_id=RUN_ID,
        config=config, training_git_sha="a" * 40,
    )
    with pytest.raises(ValueError, match="checksum"):
        load_inference_model(output, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="size"):
        validate_model_artifact(
            output, expected_sha256=info.sha256, expected_run_id=RUN_ID,
            config=config, max_size_bytes=1,
        )


def test_incompatible_environment_state_shape_and_lineage_fail(source_checkpoint, tmp_path):
    source, config, _ = source_checkpoint
    output = tmp_path / "battlezone_dqn_model.pt"
    info = export_inference_model(
        checkpoint_path=source, output_path=output, project_run_id=RUN_ID,
        config=config, training_git_sha="a" * 40,
    )
    incompatible = deepcopy(config)
    incompatible["environment"]["frameskip"] = 3
    with pytest.raises(ValueError, match="environment contract"):
        load_inference_model(output, expected_config=incompatible)
    with pytest.raises(ValueError, match="run_id"):
        load_inference_model(output, expected_project_run_id="wrong")
    payload = torch.load(output, weights_only=True)
    payload["metadata"]["state_shape"] = [1, 2, 3, 4]
    torch.save(payload, output)
    with pytest.raises(ValueError, match="state_shape"):
        load_inference_model(output)
    assert info.metadata["source_checkpoint_identity"]["sha256"] == compute_sha256(source)


def test_resolution_is_local_first_and_fallback_requires_explicit_run(tmp_path):
    battlezone = tmp_path / "project"
    persistent = tmp_path / "drive"
    fallback = persistent / "models" / RUN_ID / "battlezone_dqn_model.pt"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"fallback")
    assert resolve_delivery_model_path(
        battlezone_dir=battlezone, persistent_root=persistent, project_run_id=RUN_ID,
    ) == (fallback, "PERSISTENT_FALLBACK")
    local = battlezone / "battlezone_dqn_model.pt"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local")
    assert resolve_delivery_model_path(
        battlezone_dir=battlezone, persistent_root=persistent, project_run_id=RUN_ID,
    ) == (local, "LOCAL_DELIVERY")
    with pytest.raises(ValueError, match="run_id"):
        resolve_delivery_model_path(
            battlezone_dir=battlezone, persistent_root=persistent, project_run_id="",
        )
    local.unlink()
    fallback.unlink()
    with pytest.raises(FileNotFoundError, match="Searched"):
        resolve_delivery_model_path(
            battlezone_dir=battlezone, persistent_root=persistent, project_run_id=RUN_ID,
        )


def test_model_is_atomically_materialized_in_drive_and_local_project(source_checkpoint, tmp_path):
    source, config, _ = source_checkpoint
    delivery = tmp_path / "delivery" / "battlezone_dqn_model.pt"
    info = export_inference_model(
        checkpoint_path=source, output_path=delivery, project_run_id=RUN_ID,
        config=config, training_git_sha="a" * 40,
    )
    persistent = tmp_path / "drive" / "models" / RUN_ID / delivery.name
    local = tmp_path / "project" / delivery.name
    copies = materialize_model_copies(
        delivery_model_path=delivery, persistent_model_path=persistent,
        local_project_model_path=local, expected_sha256=info.sha256,
    )
    assert copies["persistent_model"] == persistent
    assert copies["local_project_model"] == local
    assert compute_sha256(delivery) == compute_sha256(persistent) == compute_sha256(local)
    delivery_metadata = json.loads(delivery.with_name("battlezone_dqn_model.metadata.json").read_text())
    assert json.loads(local.with_name("battlezone_dqn_model.metadata.json").read_text()) == delivery_metadata
    assert json.loads(persistent.with_name("battlezone_dqn_model.metadata.json").read_text()) == delivery_metadata
    assert persistent.with_suffix(".pt.sha256").read_text() == local.with_suffix(".pt.sha256").read_text()
    assert not list(tmp_path.rglob("*.tmp"))

    local.write_bytes(b"different")
    with pytest.raises(FileExistsError, match="explicit overwrite"):
        materialize_model_copies(
            delivery_model_path=delivery, persistent_model_path=persistent,
            local_project_model_path=local, expected_sha256=info.sha256,
        )
    materialize_model_copies(
        delivery_model_path=delivery, persistent_model_path=persistent,
        local_project_model_path=local, expected_sha256=info.sha256, overwrite=True,
    )
    assert compute_sha256(local) == info.sha256


def test_delivery_manifest_and_gate_require_complete_consistent_evidence(tmp_path):
    paths = build_delivery_paths(tmp_path, RUN_ID)
    paths["model"].parent.mkdir(parents=True)
    paths["model"].write_bytes(b"compact-model")
    persistent_model = tmp_path / "models" / RUN_ID / "battlezone_dqn_model.pt"
    local_model = tmp_path / "project" / "battlezone_dqn_model.pt"
    persistent_model.parent.mkdir(parents=True)
    local_model.parent.mkdir(parents=True)
    persistent_model.write_bytes(paths["model"].read_bytes())
    local_model.write_bytes(paths["model"].read_bytes())
    manifest = write_delivery_manifest(
        path=paths["manifest"], project_run_id=RUN_ID,
        training_git_sha="a" * 40, delivery_git_sha="b" * 40,
        source_final_checkpoint="final.pt", source_intermediate_checkpoint="step750k.pt",
        tensorboard_logs="logs", delivery_model_path=paths["model"],
        persistent_model_path=persistent_model, local_project_model_path=local_model,
        figures={"reward": "reward.png"}, videos={"post": "post.mp4"},
    )
    payload = json.loads(manifest.read_text())
    assert payload["run_id"] == RUN_ID
    assert payload["model_sha256"] == compute_sha256(paths["model"])
    assert payload["delivery_model_path"] == str(paths["model"])
    assert payload["persistent_model_path"] == str(persistent_model)
    assert payload["local_project_model_path"] == str(local_model)
    statuses = {key: True for key in (
        "model", "model_checksum", "model_metadata", "round_trip_load",
        "persistent_delivery_model", "local_delivery_model", "model_sha_consistency",
        "greedy_action", "standalone_episode", "training_reward_figure",
        "training_loss_figure", "training_q_epsilon_figure", "training_video",
        "training_video_metadata", "post_training_video",
        "post_training_video_metadata", "delivery_manifest", "run_lineage",
    )}
    assert evaluate_delivery_gate(statuses)["HU011B_DELIVERY_GATE"] == "PASS"
    statuses["post_training_video"] = False
    assert evaluate_delivery_gate(statuses)["HU011B_DELIVERY_GATE"] == "FAIL"
    statuses["post_training_video"] = True
    statuses["local_delivery_model"] = False
    assert evaluate_delivery_gate(statuses)["HU011B_DELIVERY_GATE"] == "FAIL"
