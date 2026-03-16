"""Tests for benchmark output helper functions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_benchmark_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_script", module_path)
    if spec is None or spec.loader is None:
        msg = f"Failed to load benchmark module from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def test_write_stdout_appends_newline(capsys) -> None:
    benchmark._write_stdout("hello")
    out = capsys.readouterr().out
    assert out == "hello\n"


def test_write_stderr_appends_newline(capsys) -> None:
    benchmark._write_stderr("oops")
    err = capsys.readouterr().err
    assert err == "oops\n"
