"""Regression tests for repository-level CI and lint policy."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def test_static_analysis_targets_main_and_develop() -> None:
    workflow = _read(".github/workflows/static-analysis.yml")

    assert 'branches: ["main", "develop"]' in workflow
    assert 'branches: ["main", "master"]' not in workflow


def test_cpp_tests_targets_main_and_develop() -> None:
    workflow = _read(".github/workflows/cpp-tests.yml")

    assert "branches: [main, develop]" in workflow


def test_cpp_tests_uses_cxx17() -> None:
    workflow = _read(".github/workflows/cpp-tests.yml")

    assert "-std=c++17" in workflow


def test_cpp_tests_treats_warnings_as_errors() -> None:
    workflow = _read(".github/workflows/cpp-tests.yml")

    for flag in ("-Wall", "-Wextra", "-Wpedantic", "-Werror"):
        assert flag in workflow