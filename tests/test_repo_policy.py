"""Regression tests for repository-level CI and lint policy."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def test_static_analysis_targets_main_and_develop() -> None:
    workflow = _read(".github/workflows/static-analysis.yml")

    assert 'branches: ["main", "develop"]' in workflow
    assert 'branches: ["main", "master"]' not in workflow


def test_static_analysis_does_not_soft_ignore_iwyu() -> None:
    workflow = _read(".github/workflows/static-analysis.yml")

    assert "|| true" not in workflow


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


def test_cpp_tests_uses_openmp_flag() -> None:
    workflow = _read(".github/workflows/cpp-tests.yml")

    assert "-fopenmp" in workflow


def test_lint_workflow_pins_python_311() -> None:
    workflow = _read(".github/workflows/lint.yml")

    assert "actions/setup-python@v5" in workflow
    assert 'python-version: "3.11"' in workflow


def test_ci_workflow_pins_python_311() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "actions/setup-python@v5" in workflow
    assert 'python-version: "3.11"' in workflow


def test_docs_workflow_pins_python_311() -> None:
    workflow = _read(".github/workflows/docs.yml")

    assert workflow.count("actions/setup-python@v5") == 2
    assert workflow.count('python-version: "3.11"') == 2


def test_lint_workflow_tracks_toolchain_files() -> None:
    workflow = _read(".github/workflows/lint.yml")

    assert '".python-version"' in workflow
    assert '"uv.lock"' in workflow


def test_pyproject_advertises_python_311_only() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    classifiers = pyproject["project"]["classifiers"]

    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.12" not in classifiers


def test_pre_commit_clang_tidy_is_strict() -> None:
    config = _read(".pre-commit-config.yaml")

    assert '"--extra-arg=-std=c++17"' in config
    assert '"--warnings-as-errors=*"' in config
    assert '"--header-filter=^ocmesher/source/.*"' in config


def test_tests_per_file_ignores_do_not_disable_pt019() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    test_ignores = pyproject["tool"]["ruff"]["lint"]["per-file-ignores"]["tests/**/*.py"]

    assert "PT019" not in test_ignores
