"""Reproducible GitHub-to-Colab execution bootstrap for BattleZone."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional


DEFAULT_REPO_URL = "https://github.com/j-mauro-r/reinforcement_learning_reto_1.git"
DEFAULT_COLAB_ROOT = Path("/content/reinforcement_learning_reto_1")
DEFAULT_BATTLEZONE_SUBDIR = "3_BattleZone"
_STATE_ATTR = "_BATTLEZONE_BOOTSTRAP_STATE"


@dataclass(frozen=True)
class BootstrapResult:
    """Describes the exact repository copy selected for execution."""

    is_colab: bool
    repo_root: Path
    battlezone_dir: Path
    requested_ref: str
    requested_commit: Optional[str]
    resolved_sha: str
    requirements_path: Path
    environment_source: Optional[Path] = None

    def as_dict(self) -> dict[str, str | bool | None]:
        """Returns notebook-friendly bootstrap metadata."""
        return {
            "is_colab": self.is_colab,
            "repo_root": str(self.repo_root),
            "battlezone_dir": str(self.battlezone_dir),
            "requested_ref": self.requested_ref,
            "requested_commit": self.requested_commit,
            "resolved_sha": self.resolved_sha,
            "requirements_path": str(self.requirements_path),
            "environment_source": str(self.environment_source) if self.environment_source else None,
        }


def running_in_colab() -> bool:
    """Returns whether the Google Colab runtime module is importable."""
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def validate_cuda_runtime(
    *, torch_version: str, cuda_available: bool,
    cuda_version: Optional[str], gpu_name: Optional[str], required: bool,
) -> None:
    """Fails fast when a required CUDA runtime is absent or CPU-only."""
    invalid = (
        not cuda_available
        or not cuda_version
        or not gpu_name
        or "+cpu" in torch_version.lower()
    )
    if required and invalid:
        raise RuntimeError(
            "CUDA_REQUIRED_FOR_HU011\n\n"
            "This runtime is CPU-only or PyTorch has no usable CUDA build.\n"
            "Select a GPU runtime in Google Colab and restart the session.\n"
            "HU011 reference_v1 training was NOT started."
        )


def prepare_execution_environment(
    requested_ref: str,
    requested_commit: Optional[str] = None,
    repo_url: str = DEFAULT_REPO_URL,
    colab_root: str | Path = DEFAULT_COLAB_ROOT,
    battlezone_subdir: str = DEFAULT_BATTLEZONE_SUBDIR,
) -> BootstrapResult:
    """Selects a local checkout or prepares a detached Colab checkout.

    Local mode is read-only with respect to Git. Colab mode clones/fetches only
    beneath ``/content``, resolves an explicit ref or preferred commit, guards
    against stale imports, and checks out the immutable SHA detached.
    """
    if not requested_ref:
        raise ValueError("requested_ref must be explicit.")
    is_colab = running_in_colab()
    repo_root = Path(colab_root) if is_colab else _local_repo_root()
    battlezone_dir = repo_root / battlezone_subdir

    if is_colab:
        if Path("/content/drive") == repo_root or Path("/content/drive") in repo_root.parents:
            raise RuntimeError("The Git repository must not be cloned inside Google Drive.")
        _ensure_colab_copy(repo_root, repo_url)
        _ensure_clean_colab_copy(repo_root)
        _git(["fetch", "--prune", "origin"], cwd=repo_root)
        target_sha = _resolve_target_sha(repo_root, requested_ref, requested_commit)
        _guard_stale_imports(_current_sha(repo_root), target_sha, battlezone_dir)
        _git(["checkout", "--detach", target_sha], cwd=repo_root)
    else:
        target_sha = _current_sha(repo_root)
        if requested_commit and requested_commit != target_sha and not target_sha.startswith(requested_commit):
            raise RuntimeError(
                "Local commit pin does not match HEAD. "
                f"HEAD={target_sha}, requested_commit={requested_commit}."
            )
        _guard_stale_imports(target_sha, target_sha, battlezone_dir)

    requirements = battlezone_dir / "requirements.txt"
    if not requirements.is_file():
        raise RuntimeError(f"Requirements file not found: {requirements}")
    _configure_imports(repo_root, battlezone_dir)
    _remember_bootstrap_sha(target_sha)
    result = BootstrapResult(
        is_colab=is_colab,
        repo_root=repo_root.resolve(),
        battlezone_dir=battlezone_dir.resolve(),
        requested_ref=requested_ref,
        requested_commit=requested_commit,
        resolved_sha=target_sha,
        requirements_path=requirements.resolve(),
    )
    print_bootstrap_summary(result)
    return result


def install_project_requirements(requirements_path: str | Path) -> None:
    """Installs the selected commit's requirements with the active Python."""
    path = Path(requirements_path)
    if not path.is_file():
        raise RuntimeError(f"Requirements file not found: {path}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(path)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("BattleZone dependency installation failed.") from exc


def verify_environment_import(result: BootstrapResult) -> Path:
    """Verifies ``src.environment`` originates from the selected BattleZone copy."""
    module = importlib.import_module("src.environment")
    module_path = Path(module.__file__).resolve()
    expected = result.battlezone_dir.resolve()
    if expected not in module_path.parents:
        raise RuntimeError(f"src.environment imported from unexpected path: {module_path}")
    object.__setattr__(result, "environment_source", module_path)
    print(f"src.environment import source: {module_path}")
    return module_path


def print_bootstrap_summary(result: BootstrapResult) -> None:
    """Prints the selected execution source and exact Git SHA."""
    print("BattleZone execution bootstrap")
    print(f"  runtime: {'Google Colab' if result.is_colab else 'local'}")
    print(f"  repository: {result.repo_root}")
    print(f"  battlezone_dir: {result.battlezone_dir}")
    print(f"  requested_ref: {result.requested_ref}")
    print(f"  requested_commit: {result.requested_commit or '<none>'}")
    print(f"  resolved_sha: {result.resolved_sha}")
    print(f"  requirements: {result.requirements_path}")


def _ensure_colab_copy(repo_root: Path, repo_url: str) -> None:
    if repo_root.exists():
        if not (repo_root / ".git").is_dir():
            raise RuntimeError(f"Colab path exists but is not a Git repository: {repo_root}")
        return
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    _git(["clone", repo_url, str(repo_root)], cwd=repo_root.parent)


def _ensure_clean_colab_copy(repo_root: Path) -> None:
    if _git(["status", "--porcelain"], cwd=repo_root).strip():
        raise RuntimeError("Colab execution copy has local changes; restart the runtime or clean /content.")


def _resolve_target_sha(repo_root: Path, requested_ref: str, requested_commit: Optional[str]) -> str:
    if requested_commit:
        return _git(["rev-parse", "--verify", f"{requested_commit}^{{commit}}"], cwd=repo_root).strip()
    try:
        return _git(["rev-parse", "--verify", f"origin/{requested_ref}^{{commit}}"], cwd=repo_root).strip()
    except RuntimeError:
        return _git(["rev-parse", "--verify", f"{requested_ref}^{{commit}}"], cwd=repo_root).strip()


def _local_repo_root() -> Path:
    return Path(_git(["rev-parse", "--show-toplevel"], cwd=Path.cwd()).strip()).resolve()


def _current_sha(repo_root: Path) -> str:
    return _git(["rev-parse", "HEAD"], cwd=repo_root).strip()


def _configure_imports(repo_root: Path, battlezone_dir: Path) -> None:
    os.chdir(battlezone_dir)
    for value in (str(repo_root), str(battlezone_dir)):
        while value in sys.path:
            sys.path.remove(value)
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(battlezone_dir))


def _guard_stale_imports(current_sha: str, target_sha: str, expected_project_dir: Path) -> None:
    loaded = sorted(name for name in sys.modules if name == "src" or name.startswith("src."))
    state = getattr(builtins, _STATE_ATTR, {})
    previous_sha = state.get("resolved_sha")
    if loaded and current_sha != target_sha:
        raise RuntimeError("Target commit changed after src modules were imported; restart the runtime/kernel.")
    if loaded and previous_sha and previous_sha != target_sha:
        raise RuntimeError("src modules belong to another bootstrap commit; restart the runtime/kernel.")
    unexpected = []
    for name in loaded:
        module_path = getattr(sys.modules[name], "__file__", None)
        if module_path and expected_project_dir.resolve() not in Path(module_path).resolve().parents:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(
            f"src modules were loaded from another project copy: {unexpected}; restart the runtime/kernel."
        )


def _remember_bootstrap_sha(resolved_sha: str) -> None:
    setattr(builtins, _STATE_ATTR, {"resolved_sha": resolved_sha})


def _git(args: list[str], cwd: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=Path(cwd), check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or str(exc)).strip()
        raise RuntimeError(f"Git command failed: git {' '.join(args)}\n{detail}") from exc
    return result.stdout
