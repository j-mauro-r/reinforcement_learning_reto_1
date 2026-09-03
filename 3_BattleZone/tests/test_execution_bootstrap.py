"""Tests for the BattleZone GitHub-to-Colab execution bootstrap."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from src import execution_bootstrap as bootstrap


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
SHA = "a" * 40


@pytest.fixture(autouse=True)
def restore_process_state(monkeypatch):
    original_cwd = Path.cwd()
    original_path = list(sys.path)
    if hasattr(builtins, bootstrap._STATE_ATTR):
        monkeypatch.delattr(builtins, bootstrap._STATE_ATTR)
    yield
    monkeypatch.chdir(original_cwd)
    sys.path[:] = original_path


def test_running_in_colab_is_false_locally():
    assert bootstrap.running_in_colab() is False


def test_local_uses_existing_repo_without_clone_fetch_or_checkout(monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: False)
    real_git = bootstrap._git

    def spy(args, cwd):
        calls.append(args)
        return real_git(args, cwd)

    monkeypatch.setattr(bootstrap, "_git", spy)
    result = bootstrap.prepare_execution_environment("main")
    assert result.repo_root == REPO.resolve()
    assert result.battlezone_dir == PROJECT.resolve()
    assert not any(args[0] in {"clone", "fetch", "checkout"} for args in calls)


def test_local_commit_pin_accepts_head_and_rejects_other(monkeypatch):
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: False)
    head = bootstrap._current_sha(REPO)
    assert bootstrap.prepare_execution_environment("main", head).resolved_sha == head
    with pytest.raises(RuntimeError, match="does not match HEAD"):
        bootstrap.prepare_execution_environment("main", "0" * 40)


def test_colab_clone_fetch_resolve_and_detached_checkout(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    calls = []
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: True)
    monkeypatch.setattr(bootstrap, "_guard_stale_imports", lambda *args: None)

    def fake_git(args, cwd):
        calls.append(args)
        if args[0] == "clone":
            (root / ".git").mkdir(parents=True)
            (root / "3_BattleZone").mkdir()
            (root / "3_BattleZone/requirements.txt").write_text("pytest\n")
            return ""
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args[:2] == ["rev-parse", "HEAD"]:
            return "b" * 40
        if args[0] == "rev-parse":
            return SHA
        return ""

    monkeypatch.setattr(bootstrap, "_git", fake_git)
    result = bootstrap.prepare_execution_environment("main", colab_root=root)
    assert result.resolved_sha == SHA
    assert calls[0][0] == "clone"
    assert ["fetch", "--prune", "origin"] in calls
    assert ["checkout", "--detach", SHA] in calls


def test_colab_rejects_non_git_path(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: True)
    with pytest.raises(RuntimeError, match="not a Git repository"):
        bootstrap.prepare_execution_environment("main", colab_root=tmp_path)


def test_colab_rejects_repository_inside_drive(monkeypatch):
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: True)
    with pytest.raises(RuntimeError, match="must not be cloned inside Google Drive"):
        bootstrap.prepare_execution_environment(
            "main", colab_root="/content/drive/MyDrive/reinforcement_learning_reto_1"
        )


def test_colab_rejects_dirty_copy(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: True)
    monkeypatch.setattr(bootstrap, "_git", lambda args, cwd: " M file" if args[0] == "status" else "")
    with pytest.raises(RuntimeError, match="local changes"):
        bootstrap.prepare_execution_environment("main", colab_root=tmp_path)


def test_requested_commit_precedes_ref(monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap, "_git", lambda args, cwd: calls.append(args) or SHA)
    assert bootstrap._resolve_target_sha(REPO, "ignored", "abc123") == SHA
    assert calls == [["rev-parse", "--verify", "abc123^{commit}"]]


def test_ref_resolution_prefers_origin(monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap, "_git", lambda args, cwd: calls.append(args) or SHA)
    assert bootstrap._resolve_target_sha(REPO, "feature/x", None) == SHA
    assert calls[0][-1] == "origin/feature/x^{commit}"


def test_configure_imports_is_idempotent(monkeypatch):
    monkeypatch.chdir(REPO)
    bootstrap._configure_imports(REPO, PROJECT)
    bootstrap._configure_imports(REPO, PROJECT)
    assert Path.cwd() == PROJECT
    assert sys.path.count(str(PROJECT)) == 1
    assert sys.path[0] == str(PROJECT)


def test_verify_import_accepts_selected_copy(monkeypatch):
    result = bootstrap.BootstrapResult(False, REPO, PROJECT, "main", None, SHA, PROJECT / "requirements.txt")
    module = types.SimpleNamespace(__file__=str(PROJECT / "src/environment.py"))
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: module)
    assert bootstrap.verify_environment_import(result) == (PROJECT / "src/environment.py").resolve()


def test_verify_import_rejects_other_copy(monkeypatch, tmp_path):
    result = bootstrap.BootstrapResult(False, REPO, PROJECT, "main", None, SHA, PROJECT / "requirements.txt")
    module = types.SimpleNamespace(__file__=str(tmp_path / "src/environment.py"))
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: module)
    with pytest.raises(RuntimeError, match="unexpected path"):
        bootstrap.verify_environment_import(result)


def test_stale_commit_and_wrong_copy_are_rejected(monkeypatch, tmp_path):
    module = types.ModuleType("src.foreign")
    module.__file__ = str(tmp_path / "src/foreign.py")
    monkeypatch.setitem(sys.modules, "src.foreign", module)
    with pytest.raises(RuntimeError, match="Target commit changed"):
        bootstrap._guard_stale_imports("a" * 40, "b" * 40, PROJECT)
    with pytest.raises(RuntimeError, match="another project copy"):
        bootstrap._guard_stale_imports(SHA, SHA, PROJECT)


def test_previous_bootstrap_commit_is_guarded(monkeypatch):
    monkeypatch.setattr(builtins, bootstrap._STATE_ATTR, {"resolved_sha": "b" * 40}, raising=False)
    with pytest.raises(RuntimeError, match="another bootstrap commit"):
        bootstrap._guard_stale_imports(SHA, SHA, PROJECT)


def test_same_sha_and_same_project_copy_allow_idempotent_bootstrap(monkeypatch):
    monkeypatch.setattr(builtins, bootstrap._STATE_ATTR, {"resolved_sha": SHA}, raising=False)
    bootstrap._guard_stale_imports(SHA, SHA, PROJECT)


def test_missing_requirements_and_install_failure(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        bootstrap.install_project_requirements(tmp_path / "missing.txt")
    req = tmp_path / "requirements.txt"
    req.write_text("pytest\n")
    monkeypatch.setattr(
        bootstrap.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "pip")),
    )
    with pytest.raises(RuntimeError, match="installation failed"):
        bootstrap.install_project_requirements(req)


def test_git_failure_has_controlled_error(monkeypatch):
    monkeypatch.setattr(
        bootstrap.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "git", stderr="bad ref")),
    )
    with pytest.raises(RuntimeError, match="Git command failed"):
        bootstrap._git(["rev-parse", "HEAD"], REPO)


def test_notebook_bootstraps_before_project_imports_and_uses_drive_only_for_artifacts():
    notebook = json.loads((PROJECT / "pipeline_battlezone.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    combined = "\n".join(sources)
    bootstrap_index = next(i for i, source in enumerate(sources) if "prepare_execution_environment(" in source)
    environment_index = next(i for i, source in enumerate(sources) if "from src.environment import" in source)
    drive_index = next(i for i, source in enumerate(sources) if "drive.mount(" in source)
    assert bootstrap_index < environment_index <= drive_index
    assert "/content/reinforcement_learning_reto_1" in sources[bootstrap_index]
    assert "REQUESTED_REF" in sources[bootstrap_index]
    assert "REQUESTED_COMMIT" in sources[bootstrap_index]
    assert 'REQUESTED_REF = "feature/battlezone-colab-execution-bootstrap"' in sources[bootstrap_index]
    assert 'REQUESTED_REF = "main"' not in sources[bootstrap_index]
    assert sources[bootstrap_index].index("rev-parse") < sources[bootstrap_index].index("import src.execution_bootstrap")
    assert "bootstrap_source" in sources[bootstrap_index]
    assert "Unexpected bootstrap source" in sources[bootstrap_index]
    assert "Restart the Colab runtime and execute from the first cell" in sources[bootstrap_index]
    assert "sys.path.append(" not in combined
    assert "git\", \"clone\"" in sources[bootstrap_index]
    assert "MyDrive/reinforcement_learning_reto_1" not in sources[bootstrap_index]
    assert "PERSISTENT_ROOT" in sources[drive_index]


def test_notebook_hard_gates_cuda_and_arms_real_hu011_training_explicitly():
    notebook = json.loads((PROJECT / "pipeline_battlezone.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    combined = "\n".join(sources)
    cuda_index = next(i for i, source in enumerate(sources) if "validate_cuda_runtime(" in source)
    drive_index = next(i for i, source in enumerate(sources) if "drive.mount(" in source)
    training_index = next(i for i, source in enumerate(sources) if "RUN_LONG_TRAINING = False" in source)

    assert cuda_index <= drive_index < training_index
    assert "torch.cuda.is_available()" in sources[cuda_index]
    assert "torch.version.cuda" in sources[cuda_index]
    assert "torch.cuda.get_device_name(0)" in sources[cuda_index]
    assert "PERSISTENT_ROOT_WRITABLE" in sources[drive_index]
    assert "ENVIRONMENT_SMOKE_TEST — THIS IS NOT HU011 TRAINING" in combined
    assert "if not RUN_LONG_TRAINING:" in sources[training_index]
    assert "elif not preflight.ready:" in sources[training_index]
    assert "result = run_training_session(" in sources[training_index]
    assert "target_global_step_override" not in sources[training_index]
    assert "HU011_PREFLIGHT_READY" in sources[training_index]
    assert 'print("ARTIFACT CHECK")' in sources[training_index]
    assert 'print("manifest:"' in sources[training_index]
    assert 'print("checkpoints:"' in sources[training_index]
    assert 'print("tensorboard logs:"' in sources[training_index]


def test_colab_requirements_do_not_replace_runtime_pytorch():
    requirements = (PROJECT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    active = [line.strip().lower() for line in requirements if line.strip() and not line.lstrip().startswith("#")]
    assert not any(line == "torch" or line.startswith(("torch=", "torch<", "torch>")) for line in active)


@pytest.mark.parametrize(
    ("torch_version", "cuda_available", "cuda_version", "gpu_name"),
    [
        ("2.11.0+cpu", False, None, None),
        ("2.11.0+cpu", True, "12.8", "Fake GPU"),
    ],
)
def test_required_cuda_runtime_rejects_cpu_only_pytorch(
    torch_version, cuda_available, cuda_version, gpu_name,
):
    with pytest.raises(RuntimeError, match="CUDA_REQUIRED_FOR_HU011"):
        bootstrap.validate_cuda_runtime(
            torch_version=torch_version, cuda_available=cuda_available,
            cuda_version=cuda_version, gpu_name=gpu_name, required=True,
        )


def test_required_cuda_runtime_accepts_complete_cuda_build():
    bootstrap.validate_cuda_runtime(
        torch_version="2.11.0+cu128", cuda_available=True,
        cuda_version="12.8", gpu_name="Fake GPU", required=True,
    )


def test_notebook_contains_hu011b_delivery_and_standalone_contract():
    notebook = json.loads((PROJECT / "pipeline_battlezone.ipynb").read_text(encoding="utf-8"))
    combined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for marker in (
        "HU011B — DELIVERY ARTIFACTS", "VERIFICACIÓN AUTÓNOMA DEL MODELO ENTREGADO",
        "HU011B_DELIVERY_GATE", "battlezone_dqn_model.pt",
        "battlezone_dqn_training_process.mp4", "battlezone_dqn_post_training.mp4",
        "load_tensorboard_scalars", "delivery_manifest.json",
        "EVIDENCIA DEL PROCESO DE ENTRENAMIENTO", "COMPORTAMIENTO APRENDIDO POST-ENTRENAMIENTO",
    ):
        assert marker in combined
    assert "RUN_HU011B_DELIVERY = False" in combined
    assert "RUN_LONG_TRAINING = False" in combined
    assert "latest_checkpoint" not in combined.lower()
    assert "get_latest" not in combined.lower()
    assert "2_Assault" not in combined
    assert "mlflow" not in combined.lower()
