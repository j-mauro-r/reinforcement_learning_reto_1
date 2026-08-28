"""Tests for the HU002B execution bootstrap."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

ASSAULT_DIR = Path(__file__).resolve().parents[1]
if str(ASSAULT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSAULT_DIR))

from src import execution_bootstrap
from src.execution_bootstrap import prepare_execution_environment


def _current_head() -> str:
    return execution_bootstrap._git(["rev-parse", "HEAD"], cwd=ASSAULT_DIR.parents[0]).strip()


def test_local_bootstrap_uses_existing_repository_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(execution_bootstrap, "running_in_colab", lambda: False)
    if hasattr(builtins, "_ASSAULT_BOOTSTRAP_STATE"):
        delattr(builtins, "_ASSAULT_BOOTSTRAP_STATE")

    first = prepare_execution_environment(requested_ref="feature/hu002b-pipeline-local-github-colab")
    second = prepare_execution_environment(requested_ref="feature/hu002b-pipeline-local-github-colab")

    assert first.is_colab is False
    assert first.repo_root == ASSAULT_DIR.parents[0].resolve()
    assert first.assault_dir == ASSAULT_DIR.resolve()
    assert first.resolved_sha == second.resolved_sha == _current_head()
    assert first.requirements_path.exists()


def test_local_commit_pinning_accepts_current_head(monkeypatch):
    monkeypatch.setattr(execution_bootstrap, "running_in_colab", lambda: False)
    head = _current_head()

    result = prepare_execution_environment(
        requested_ref="feature/hu002b-pipeline-local-github-colab",
        requested_commit=head,
    )

    assert result.resolved_sha == head


def test_local_commit_pinning_rejects_non_matching_commit(monkeypatch):
    monkeypatch.setattr(execution_bootstrap, "running_in_colab", lambda: False)

    with pytest.raises(RuntimeError, match="Local commit pin does not match"):
        prepare_execution_environment(
            requested_ref="feature/hu002b-pipeline-local-github-colab",
            requested_commit="0" * 40,
        )


def test_stale_import_guard_blocks_commit_change(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.environment", types.ModuleType("src.environment"))
    monkeypatch.setattr(builtins, "_ASSAULT_BOOTSTRAP_STATE", {"resolved_sha": "a" * 40}, raising=False)

    with pytest.raises(RuntimeError, match="different bootstrap commit"):
        execution_bootstrap._guard_stale_imports(current_sha="b" * 40, target_sha="b" * 40)
