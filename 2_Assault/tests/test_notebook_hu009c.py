"""Notebook integrity checks for HU009C delivery report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

NOTEBOOK_PATH = ASSAULT_DIR / "assault_ddqn.ipynb"


def _notebook_text() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_is_valid_json_and_contains_required_hu009c_sections():
    text = _notebook_text()

    required_sections = [
        "Problema y objetivo",
        "Seleccion del algoritmo",
        "Entorno y preprocessing",
        "Arquitectura del agente",
        "Hiperparametros efectivos",
        "Librerias, versiones, hardware y tiempo",
        "Metricas y maximo tres graficas",
        "Evaluacion >=10 episodios",
        "Comparacion contra baseline",
        "Comportamiento observado",
        "Limitaciones",
        "Conclusion",
        "Artefactos de entrega",
    ]
    for section in required_sections:
        assert section in text
    assert "VALIDACION COLAB PENDIENTE" in text


def test_notebook_uses_auto_execution_and_keeps_training_opt_in_derived_from_resolution():
    text = _notebook_text()

    assert "from src.model_artifact import export_inference_model, load_inference_model" in text
    assert "from src.reporting import plot_training_figures, prepare_training_figures" in text
    assert "from src.video import generate_assault_demo_video" in text
    assert 'os.environ.setdefault("ASSAULT_EXECUTION_MODE", "auto")' in text
    assert 'os.environ.setdefault("ASSAULT_RUN_TRAINING", "0")' not in text
    assert "RUN_TRAINING = delivery_execution.training_required" in text
    assert "AUTO_RESOLUTION" in text
    assert "TRAINING_SKIPPED_FINAL_CHECKPOINT_EXISTS" in text
    assert "if not RUN_TRAINING:" in text


def test_notebook_displays_generated_video_inline_once_after_generation():
    text = _notebook_text()
    video_section = text.split("## 14. Evaluacion y video desde modelo compacto", maxsplit=1)[1]
    generation_index = video_section.index("video_summary = generate_assault_demo_video(")
    display_index = video_section.index("display(Video(str(VIDEO_PATH), embed=True))")

    assert text.count("generate_assault_demo_video(") == 1
    assert generation_index < display_index
    assert "from IPython.display import Video, display" in video_section
    assert "if VIDEO_PATH.exists() and VIDEO_PATH.stat().st_size > 0:" in video_section
    assert 'print("VIDEO_READY=True")' in video_section
    assert 'print("video_path=", VIDEO_PATH)' in video_section
    assert 'print("video_reward=", video_summary.reward)' in video_section
    assert 'print("video_steps=", video_summary.steps)' in video_section
    assert 'print("video_seed=", video_summary.seed)' in video_section
    assert 'print("video_epsilon=", video_summary.epsilon)' in video_section
    assert 'print("video_project_run_id=", video_summary.project_run_id)' in video_section
    assert 'print("video_model_sha256=", video_summary.model_sha256)' in video_section
    assert "VIDEO_INLINE_WARNING" in video_section


def test_notebook_plans_exactly_three_training_figures():
    text = _notebook_text()

    assert text.count("prepare_training_figures(") == 1
    assert "training_figure_count" in text
    assert "learning_rate" not in text.split("## 13. Figuras TensorBoard reales", maxsplit=1)[1].split("## 14.", maxsplit=1)[0]


def test_auto_clean_drive_requires_new_training(tmp_path):
    from src.hu009c_delivery import resolve_hu009c_execution_mode

    calls = []
    context = type("Context", (), {"tracking_mode": "new"})()

    def fake_prepare_training_session(**kwargs):
        calls.append(kwargs)
        return context

    mode = resolve_hu009c_execution_mode(
        run_training=None,
        execution_mode="auto",
        project_run_id="assault_ddqn_" + "full" + "_001",
        target_timesteps=250_000,
        final_checkpoint_path=tmp_path / "checkpoints" / "missing_final.pt",
        prepare_training_session_fn=fake_prepare_training_session,
        prepare_training_session_kwargs={"target_timesteps": 250_000, "requested_mode": "auto"},
    )

    assert calls == [{"target_timesteps": 250_000, "requested_mode": "auto"}]
    assert mode.auto_resolution == "NEW"
    assert mode.training_required is True
    assert mode.session_context is context
    assert mode.hu009c_post_training_ready is False
    assert mode.training_session_bootstrap_skipped is False


def test_auto_partial_execution_resolves_resume(tmp_path):
    from src.hu009c_delivery import resolve_hu009c_execution_mode

    calls = []
    context = type("Context", (), {"tracking_mode": "resume", "restored_expected_step": 100_000})()

    def fake_prepare_training_session(**kwargs):
        calls.append(kwargs)
        return context

    mode = resolve_hu009c_execution_mode(
        run_training=None,
        execution_mode="auto",
        project_run_id="assault_ddqn_" + "full" + "_001",
        target_timesteps=250_000,
        final_checkpoint_path=tmp_path / "checkpoints" / "missing_final.pt",
        prepare_training_session_fn=fake_prepare_training_session,
        prepare_training_session_kwargs={"target_timesteps": 250_000, "requested_mode": "auto"},
    )

    assert calls == [{"target_timesteps": 250_000, "requested_mode": "auto"}]
    assert mode.auto_resolution == "RESUME"
    assert mode.training_required is True
    assert mode.session_context is context


def test_auto_finished_experiment_uses_delivery_without_training_bootstrap(tmp_path):
    from src.hu009c_delivery import resolve_hu009c_execution_mode

    final_checkpoint = tmp_path / "checkpoints" / "assault_ddqn_full_001" / "checkpoint_step_250000.pt"
    final_checkpoint.parent.mkdir(parents=True)
    final_checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fail_if_called(**kwargs):
        calls.append(kwargs)
        raise ValueError("target_timesteps must be greater than the restored global_step.")

    mode = resolve_hu009c_execution_mode(
        run_training=None,
        execution_mode="auto",
        project_run_id="assault_ddqn_" + "full" + "_001",
        target_timesteps=250_000,
        final_checkpoint_path=final_checkpoint,
        prepare_training_session_fn=fail_if_called,
        prepare_training_session_kwargs={"target_timesteps": 250_000, "requested_mode": "auto"},
    )

    assert calls == []
    assert mode.auto_resolution == "DELIVERY"
    assert mode.session_context is None
    assert mode.training_required is False
    assert mode.hu009c_post_training_ready is True
    assert mode.training_session_bootstrap_skipped is True
    assert mode.as_dict()["HU009C_POST_TRAINING_READY"] is True
    assert mode.as_dict()["training_session_bootstrap_skipped"] is True


def test_forced_delivery_without_checkpoint_fails_fast(tmp_path):
    from src.hu009c_delivery import resolve_hu009c_execution_mode

    with pytest.raises(FileNotFoundError, match="Forced delivery"):
        resolve_hu009c_execution_mode(
            run_training=None,
            execution_mode="delivery",
            project_run_id="assault_ddqn_delivery_probe",
            target_timesteps=250_000,
            final_checkpoint_path=tmp_path / "missing.pt",
            prepare_training_session_fn=lambda **_: None,
            prepare_training_session_kwargs={},
        )


def test_forced_training_still_delegates_to_prepare_training_session(tmp_path):
    from src.hu009c_delivery import resolve_hu009c_execution_mode

    calls = []
    sentinel_context = type("Context", (), {"tracking_mode": "new"})()

    def fake_prepare_training_session(**kwargs):
        calls.append(kwargs)
        return sentinel_context

    mode = resolve_hu009c_execution_mode(
        run_training=None,
        execution_mode="train",
        project_run_id="assault_ddqn_train_probe",
        target_timesteps=64,
        final_checkpoint_path=tmp_path / "checkpoint_step_000064.pt",
        prepare_training_session_fn=fake_prepare_training_session,
        prepare_training_session_kwargs={"target_timesteps": 64, "requested_mode": "auto"},
    )

    assert calls == [{"target_timesteps": 64, "requested_mode": "auto"}]
    assert mode.auto_resolution == "NEW"
    assert mode.session_context is sentinel_context
    assert mode.training_required is True
    assert mode.hu009c_post_training_ready is False
    assert mode.training_session_bootstrap_skipped is False


def test_legacy_run_training_flag_still_maps_to_train_or_delivery(tmp_path):
    from src.hu009c_delivery import resolve_hu009c_execution_mode

    final_checkpoint = tmp_path / "checkpoint_step_000064.pt"
    final_checkpoint.write_bytes(b"checkpoint")
    calls = []

    train_mode = resolve_hu009c_execution_mode(
        True,
        "assault_ddqn_train_probe",
        64,
        lambda **kwargs: calls.append(kwargs) or type("Context", (), {"tracking_mode": "new"})(),
        {"target_timesteps": 64},
    )
    delivery_mode = resolve_hu009c_execution_mode(
        False,
        "assault_ddqn_train_probe",
        64,
        lambda **_: None,
        {},
        final_checkpoint_path=final_checkpoint,
    )

    assert train_mode.execution_mode == "train"
    assert train_mode.training_required is True
    assert calls == [{"target_timesteps": 64}]
    assert delivery_mode.execution_mode == "delivery"
    assert delivery_mode.training_required is False


def test_hu009c_execution_mode_validates_identity_target_and_mode():
    from src.hu009c_delivery import resolve_hu009c_execution_mode

    with pytest.raises(ValueError, match="project_run_id"):
        resolve_hu009c_execution_mode(False, "", 1, lambda **_: None, {})
    with pytest.raises(ValueError, match="target_timesteps"):
        resolve_hu009c_execution_mode(False, "run", 0, lambda **_: None, {})
    with pytest.raises(ValueError, match="execution_mode"):
        resolve_hu009c_execution_mode(None, "run", 1, lambda **_: None, {}, execution_mode="bad")
