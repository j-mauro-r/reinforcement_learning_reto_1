"""Execution bootstrap for the local to GitHub to Colab Assault workflow."""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


DEFAULT_REPO_URL = "https://github.com/j-mauro-r/reinforcement_learning_reto_1.git"
DEFAULT_COLAB_ROOT = Path("/content/reinforcement_learning_reto_1")
DEFAULT_ASSAULT_SUBDIR = "2_Assault"
_STATE_ATTR = "_ASSAULT_BOOTSTRAP_STATE"


@dataclass(frozen=True)
class BootstrapResult:
    """Describes the repository copy selected for execution."""

    is_colab: bool
    repo_root: Path
    assault_dir: Path
    requested_ref: str
    requested_commit: Optional[str]
    resolved_sha: str
    requirements_path: Path
    environment_source: Optional[Path] = None

    def as_dict(self) -> Dict[str, str | bool | None]:
        """Returns a notebook-friendly dictionary representation."""
        return {
            "is_colab": self.is_colab,
            "repo_root": str(self.repo_root),
            "assault_dir": str(self.assault_dir),
            "requested_ref": self.requested_ref,
            "requested_commit": self.requested_commit,
            "resolved_sha": self.resolved_sha,
            "requirements_path": str(self.requirements_path),
            "environment_source": str(self.environment_source) if self.environment_source else None,
        }


def running_in_colab() -> bool:
    """Detects whether Python is running inside Google Colab.

    Returns:
        True when the Google Colab runtime module is importable.
    """
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def prepare_execution_environment(
    requested_ref: str,
    requested_commit: Optional[str] = None,
    repo_url: str = DEFAULT_REPO_URL,
    colab_root: str | Path = DEFAULT_COLAB_ROOT,
    assault_subdir: str = DEFAULT_ASSAULT_SUBDIR,
) -> BootstrapResult:
    """Prepares the versioned project copy used by local or Colab execution.

    In local execution the existing Git checkout is used without fetching,
    checking out, committing, merging or pushing. In Colab the repository is
    cloned under ``/content`` when missing, otherwise fetched and moved to a
    detached commit resolved from the explicit branch/ref or commit SHA.

    Args:
        requested_ref: Explicit branch or ref used for development runs.
        requested_commit: Optional immutable commit SHA for formal runs.
        repo_url: Public GitHub repository URL.
        colab_root: Repository path used in Google Colab.
        assault_subdir: Assault project subdirectory inside the repository.

    Returns:
        Bootstrap metadata including the exact commit SHA.

    Raises:
        RuntimeError: If Git fails, the working copy is unsafe, a requested
            local commit does not match HEAD, or stale ``src.*`` imports would
            mix code from different commits.
    """
    is_colab = running_in_colab()
    repo_root = Path(colab_root) if is_colab else _local_repo_root()

    if is_colab:
        _ensure_colab_copy(repo_root=repo_root, repo_url=repo_url)
        _ensure_clean_colab_copy(repo_root)
        _git(["fetch", "--prune", "origin"], cwd=repo_root)
        target_sha = _resolve_target_sha(repo_root, requested_ref=requested_ref, requested_commit=requested_commit)
        _guard_stale_imports(current_sha=_current_sha(repo_root), target_sha=target_sha)
        _git(["checkout", "--detach", target_sha], cwd=repo_root)
    else:
        target_sha = _current_sha(repo_root)
        if requested_commit and not target_sha.startswith(requested_commit) and requested_commit != target_sha:
            raise RuntimeError(
                "Local commit pin does not match the checked-out repository. "
                f"HEAD={target_sha}, requested_commit={requested_commit}."
            )
        _guard_stale_imports(current_sha=target_sha, target_sha=target_sha)

    assault_dir = repo_root / assault_subdir
    requirements_path = assault_dir / "requirements.txt"
    if not requirements_path.exists():
        raise RuntimeError(f"Requirements file not found at {requirements_path}.")

    _configure_imports(repo_root=repo_root, assault_dir=assault_dir)
    _remember_bootstrap_sha(target_sha)

    result = BootstrapResult(
        is_colab=is_colab,
        repo_root=repo_root.resolve(),
        assault_dir=assault_dir.resolve(),
        requested_ref=requested_ref,
        requested_commit=requested_commit,
        resolved_sha=target_sha,
        requirements_path=requirements_path.resolve(),
    )
    print_bootstrap_summary(result)
    return result


def install_project_requirements(requirements_path: str | Path) -> None:
    """Installs the requirements file from the selected repository commit.

    Args:
        requirements_path: Path to ``2_Assault/requirements.txt``.

    Raises:
        RuntimeError: If pip installation fails.
    """
    path = Path(requirements_path)
    if not path.exists():
        raise RuntimeError(f"Requirements file not found: {path}")

    command = [sys.executable, "-m", "pip", "install", "-q", "-r", str(path)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Dependency installation failed with exit code {exc.returncode}.") from exc


def verify_environment_import(result: BootstrapResult) -> Path:
    """Imports ``src.environment`` and verifies it comes from the selected copy.

    Args:
        result: Bootstrap metadata returned by ``prepare_execution_environment``.

    Returns:
        Resolved path to the imported ``src.environment`` module.

    Raises:
        RuntimeError: If the module is imported from an unexpected path.
    """
    import src.environment as environment

    module_path = Path(environment.__file__).resolve()
    expected_root = result.assault_dir.resolve()
    if expected_root not in module_path.parents:
        raise RuntimeError(f"src.environment imported from unexpected path: {module_path}")
    if result.is_colab and not str(module_path).startswith("/content/reinforcement_learning_reto_1/2_Assault/src"):
        raise RuntimeError(f"Colab import did not come from /content execution copy: {module_path}")

    object.__setattr__(result, "environment_source", module_path)
    print(f"src.environment import source: {module_path}")
    return module_path


def print_bootstrap_summary(result: BootstrapResult) -> None:
    """Prints the execution copy selected by the bootstrap."""
    print("Execution bootstrap")
    print(f"  runtime: {'Google Colab' if result.is_colab else 'local'}")
    print(f"  repository: {result.repo_root}")
    print(f"  assault_dir: {result.assault_dir}")
    print(f"  requested_ref: {result.requested_ref}")
    print(f"  requested_commit: {result.requested_commit or '<none>'}")
    print(f"  resolved_sha: {result.resolved_sha}")
    print(f"  requirements: {result.requirements_path}")


def _ensure_colab_copy(repo_root: Path, repo_url: str) -> None:
    if repo_root.exists():
        if not (repo_root / ".git").exists():
            raise RuntimeError(f"Colab path exists but is not a Git repository: {repo_root}")
        return
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    _git(["clone", repo_url, str(repo_root)], cwd=repo_root.parent)


def _ensure_clean_colab_copy(repo_root: Path) -> None:
    status = _git(["status", "--porcelain"], cwd=repo_root)
    if status.strip():
        raise RuntimeError(
            "Colab execution copy has local changes. Restart the runtime or clean /content before bootstrapping."
        )


def _resolve_target_sha(repo_root: Path, requested_ref: str, requested_commit: Optional[str]) -> str:
    if requested_commit:
        return _git(["rev-parse", "--verify", f"{requested_commit}^{{commit}}"], cwd=repo_root).strip()

    remote_ref = f"origin/{requested_ref}"
    try:
        return _git(["rev-parse", "--verify", f"{remote_ref}^{{commit}}"], cwd=repo_root).strip()
    except RuntimeError:
        return _git(["rev-parse", "--verify", f"{requested_ref}^{{commit}}"], cwd=repo_root).strip()


def _local_repo_root() -> Path:
    return Path(_git(["rev-parse", "--show-toplevel"], cwd=Path.cwd()).strip()).resolve()


def _current_sha(repo_root: Path) -> str:
    return _git(["rev-parse", "HEAD"], cwd=repo_root).strip()


def _configure_imports(repo_root: Path, assault_dir: Path) -> None:
    os.chdir(assault_dir)
    for path in (str(assault_dir), str(repo_root)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _guard_stale_imports(current_sha: str, target_sha: str) -> None:
    loaded_src_modules = sorted(name for name in sys.modules if name == "src" or name.startswith("src."))
    previous_sha = getattr(builtins, _STATE_ATTR, {}).get("resolved_sha") if hasattr(builtins, _STATE_ATTR) else None
    if loaded_src_modules and current_sha != target_sha:
        raise RuntimeError(
            "The target commit differs from the currently checked-out commit after src modules were imported. "
            "Restart the runtime/kernel and run the bootstrap before importing project modules. "
            f"current_sha={current_sha}, target_sha={target_sha}, loaded={loaded_src_modules}"
        )
    if loaded_src_modules and previous_sha and previous_sha != target_sha:
        raise RuntimeError(
            "Project modules are already loaded from a different bootstrap commit. "
            "Restart the runtime/kernel before changing commits. "
            f"previous_sha={previous_sha}, target_sha={target_sha}, loaded={loaded_src_modules}"
        )


def _remember_bootstrap_sha(resolved_sha: str) -> None:
    setattr(builtins, _STATE_ATTR, {"resolved_sha": resolved_sha})


def _git(args: list[str], cwd: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        stdout = getattr(exc, "stdout", "") or ""
        detail = (stderr or stdout or str(exc)).strip()
        raise RuntimeError(f"Git command failed: git {' '.join(args)}\n{detail}") from exc
    return result.stdout
