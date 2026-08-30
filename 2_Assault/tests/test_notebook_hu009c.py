"""Regression checks for HU009C capabilities preserved by the HU011 notebook."""

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
    return "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
    )


def test_notebook_is_valid_json_and_preserves_hu009c_report_sections():
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


def test_notebook_keeps_auto_new_resume_delivery_orchestration():
    text = _notebook_text()
    assert "from src.model_artifact import export_inference_model, load_inference_model" in text
    assert "from src.reporting import plot_training_figures, prepare_training_figures" in text
    assert "from src.video import generate_assault_demo_video" in text
    assert 'os.environ.setdefault("ASSAULT_EXECUTION_MODE", "auto")' in text
    assert "resolve_hu009c_execution_mode(" in text
    assert "RUN_TRAINING = delivery_execution.training_required" in text
    assert "AUTO_RESOLUTION" in text
    assert "TRAINING_SKIPPED_FINAL_CHECKPOINT_EXISTS" in text
    assert "if not RUN_TRAINING:" in text


def test_notebook_preserves_compact_model_training_figures_and_inline_video():
    text = _notebook_text()
    assert text.count("export_inference_model(") == 1
    assert text.count("prepare_training_figures(") == 1
    assert text.count("generate_assault_demo_video(") == 1
    assert "training_figure_count" in text
    assert 'print("VIDEO_READY=True")' in text
    assert "display(Video(str(VIDEO_PATH), embed=True))" in text
    assert "VIDEO_INLINE_WARNING" in text


def test_notebook_plans_exactly_three_training_figures_not_exploitation_as_training():
    text = _notebook_text()
    section = text.split("## 13. Figuras TensorBoard reales", maxsplit=1)[1].split("## 14.", maxsplit=1)[0]
    assert "prepare_training_figures(" in section
    assert "learning_rate" not in section
    assert "plot_exploitation_rewards" not in section


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
        project_run_id="assault_ddqn_full_001",
        target_timesteps=250_000,
        final_checkpoint_path=tmp_path / "missing_final.pt",
        prepare_training_session_fn=fake_prepare_training_session,
        prepare_training_session_kwargs={"target_timesteps": 250_000, "requested_mode": "auto"},
    )
    assert calls
    assert mode.auto_resolution == "NEW"
    assert mode.training_required is True


def test_auto_partial_execution_resolves_resume(tmp_path):
    from src.hu009c_delivery import resolve_hu009c_execution_mode

    context = type("Context", (), {"tracking_mode": "resume", "restored_expected_step": 100_000})()
    mode = resolve_hu009c_execution_mode(
        run_training=None,
        execution_mode="auto",
        project_run_id="assault_ddqn_full_001",
        target_timesteps=250_000,
        final_checkpoint_path=tmp_path / "missing_final.pt",
        prepare_training_session_fn=lambda **_: context,
        prepare_training_session_kwargs={},
    )
    assert mode.auto_resolution == "RESUME"
    assert mode.training_required is True


def test_auto_finished_experiment_uses_delivery_without_bootstrap(tmp_path):
    from src.hu009c_delivery import resolve_hu009c_execution_mode

    final_checkpoint = tmp_path / "checkpoint_step_250000.pt"
    final_checkpoint.write_bytes(b"checkpoint")
    mode = resolve_hu009c_execution_mode(
        run_training=None,
        execution_mode="auto",
        project_run_id="assault_ddqn_full_001",
        target_timesteps=250_000,
        final_checkpoint_path=final_checkpoint,
        prepare_training_session_fn=lambda **_: pytest.fail("bootstrap must be skipped"),
        prepare_training_session_kwargs={},
    )
    assert mode.auto_resolution == "DELIVERY"
    assert mode.training_required is False
    assert mode.training_session_bootstrap_skipped is True


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
