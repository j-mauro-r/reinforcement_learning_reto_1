"""Notebook integrity checks for HU009C delivery report."""

from __future__ import annotations

import json
from pathlib import Path


ASSAULT_DIR = Path(__file__).resolve().parents[1]
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


def test_notebook_uses_src_helpers_and_training_is_opt_in():
    text = _notebook_text()

    assert "from src.model_artifact import export_inference_model, load_inference_model" in text
    assert "from src.reporting import plot_training_figures, prepare_training_figures" in text
    assert "from src.video import generate_assault_demo_video" in text
    assert 'os.environ.setdefault("ASSAULT_RUN_TRAINING", "0")' in text
    assert "if not RUN_TRAINING:" in text
    assert "ASSAULT_RUN_TRAINING=0; celda omitida" in text


def test_notebook_plans_exactly_three_training_figures():
    text = _notebook_text()

    assert text.count("prepare_training_figures(") == 1
    assert "training_figure_count" in text
    assert "learning_rate" not in text.split("## 13. Figuras TensorBoard reales", maxsplit=1)[1].split("## 14.", maxsplit=1)[0]
