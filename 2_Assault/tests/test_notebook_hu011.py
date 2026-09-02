"""Static integrity tests for the HU011 final Assault notebook."""

from __future__ import annotations

import json
from pathlib import Path

ASSAULT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ASSAULT_DIR / "assault_ddqn.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _text() -> str:
    notebook = _notebook()
    chunks = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


def test_hu011_notebook_is_valid_and_has_final_report_sections():
    notebook = _notebook()
    text = _text()
    assert notebook["nbformat"] == 4
    required = [
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
        "Matriz final de cumplimiento del enunciado",
    ]
    for section in required:
        assert section in text


def test_hu011_final_evaluation_uses_compact_model_epsilon_zero_and_explicit_seeds():
    text = _text()
    assert 'os.environ.setdefault("ASSAULT_EVALUATION_EPISODES", "10")' in text
    assert 'os.environ.setdefault("ASSAULT_EVALUATION_EPSILON", "0.0")' in text
    assert "EVALUATION_EPISODES = max(10" in text
    assert "HU011 final evaluation requires epsilon=0.0" in text
    assert "FINAL_EVAL_SEEDS" in text
    assert "episode_seeds=FINAL_EVAL_SEEDS" in text
    assert "build_evaluation_artifact(" in text
    assert "final_compact_evaluation.json" in text


def test_hu011_has_training_and_exploitation_reward_evidence():
    text = _text()
    assert "prepare_training_figures(TENSORBOARD_RUN_DIR" in text
    assert "plot_training_figures(training_figures)" in text
    assert "plot_exploitation_rewards(compact_evaluation_summary)" in text
    assert "EXPLOITATION_REWARD_FIGURE_PASS" in text
    assert "compare_with_random_baseline" in text
    assert "baseline_random_assault.json" in text


def test_hu011_model_evaluation_video_share_lineage():
    text = _text()
    assert "compact_model_info.sha256 == final_evaluation_artifact" in text
    assert "== video_summary.model_sha256" in text
    assert "video_summary.project_run_id == PROJECT_RUN_ID" in text
    assert "source_checkpoint_step" in text
    assert "ARTIFACT_LINEAGE" in text


def test_hu011_gate_covers_ca01_to_ca26_and_keeps_global_requirement_explicit():
    text = _text()
    for index in range(1, 27):
        assert f'"CA{index:02d}"' in text
    assert 'os.environ.setdefault("GLOBAL_RETO_MULTI_ALGORITHM", "PENDING")' in text
    assert "GLOBAL_RETO_MULTI_ALGORITHM" in text
    assert "HU011_FINAL_DELIVERY_GATE" in text
    assert "ENTREGABLE_ASSAULT_LISTO=True" in text


def test_hu011_notebook_keeps_domain_logic_in_src_modules():
    text = _text()
    required_imports = [
        "from src.environment import create_assault_env",
        "from src.evaluator import evaluate_agent",
        "from src.model_artifact import export_inference_model, load_inference_model",
        "from src.reporting import plot_training_figures, prepare_training_figures",
        "from src.hu011_delivery import (",
        "from src.video import generate_assault_demo_video",
    ]
    for item in required_imports:
        assert item in text
    assert "class DDQNAgent" not in text
    assert "class QNetwork" not in text
    assert "class ReplayBuffer" not in text


def test_hu012_bootstrap_default_ref_is_main():
    text = _text()
    assert 'os.environ.setdefault("ASSAULT_BOOTSTRAP_REF", "main")' in text
    assert 'os.environ.setdefault("ASSAULT_BOOTSTRAP_REF", "feature/hu011-entregable-final-assault")' not in text


def test_hu012_delivery_model_section_has_autonomous_resolution_markers():
    text = _text()
    required_markers = [
        "## 15. Modelo entregable autonomo",
        "DELIVERY_MODEL_SOURCE=",
        "DELIVERY_MODEL_PATH=",
        "DELIVERY_MODEL_LOAD_PASS=True",
        "ASSAULT_DELIVERY_MODEL_EXECUTION_PASS=True",
        "HU012_DELIVERY_MODEL_GATE=PASS",
        "DELIVERY_MODEL_SOURCE in {\"DELIVERY\", \"DRIVE_FALLBACK\"}",
        "ASSAULT_DIR / \"assault_ddqn_model.pt\"",
        "BASE / \"models\" / PROJECT_RUN_ID / \"assault_ddqn_model.pt\"",
        "resolve_delivery_model_path(",
        "load_inference_model(",
        "create_assault_env(",
        "evaluate_agent(",
        "epsilon=0.0",
        "No entrenamos ni actualizamos pesos en esta seccion",
    ]
    for marker in required_markers:
        assert marker in text


def test_hu012_delivery_model_section_avoids_training_calls():
    text = _text()
    start = text.index("## 15. Modelo entregable autonomo")
    end = text.index("## **16. Reporte tecnico academico**")
    section = text[start:end]
    forbidden = [
        "run_training_session(",
        "update_experiment_state_after_success(",
        "agent.update(",
        "optimizer.step(",
        "replay_buffer.add(",
        "prepare_training_session(",
        "resolve_hu009c_execution_mode(",
        "export_inference_model(",
        "TRAINING_COMPLETE=True",
    ]
    for token in forbidden:
        assert token not in section


def test_hu012_section_does_not_redeclare_hu011_gate():
    text = _text()
    start = text.index("## 15. Modelo entregable autonomo")
    end = text.index("## **16. Reporte tecnico academico**")
    section = text[start:end]
    assert "HU011_FINAL_DELIVERY_GATE=PASS" not in section


def test_hu011_report_presents_values_tables_analysis_and_evidence_based_conclusion():
    text = _text()
    required_report_evidence = [
        "Configuracion experimental",
        "Librerias, hardware y trazabilidad",
        "Resultado del entrenamiento",
        "Evaluacion final por episodio",
        "Estadisticas de explotacion",
        "Comparacion cuantitativa contra baseline",
        "Analisis de las curvas",
        "Conclusion basada en evidencia",
        "Mejora absoluta",
        "Mejora relativa",
        "Tiempo entrenamiento (min)",
        "Git SHA ejecutado",
    ]
    for item in required_report_evidence:
        assert item in text


def test_hu011_all_code_cells_are_syntactically_valid_python():
    notebook = _notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source_text = "".join(source) if isinstance(source, list) else str(source)
        compile(source_text, f"assault_ddqn.ipynb:cell-{index}", "exec")


def test_hu011_has_no_stale_pending_conclusion():
    text = _text()
    assert "HU009C empaqueta" not in text
    assert "HU009C queda `[IMPLEMENTADA" not in text
    assert "VALIDACION COLAB PENDIENTE" in text  # only allowed as actionable gate failure signal
