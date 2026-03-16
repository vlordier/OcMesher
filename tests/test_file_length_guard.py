"""Unit tests for file-length guard helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_guard_module() -> ModuleType:
    module_path = Path(__file__).resolve().parent / "file_length_guard.py"
    spec = importlib.util.spec_from_file_location("file_length_guard", module_path)
    if spec is None or spec.loader is None:
        msg = f"Failed to load guard module from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard_module()


def test_file_length_violations_reports_missing_file(tmp_path) -> None:
    violations = guard._file_length_violations(tmp_path)
    assert any("missing file" in v for v in violations)


def test_line_count_handles_single_line_file(tmp_path) -> None:
    p = tmp_path / "one.py"
    p.write_text("x = 1", encoding="utf-8")
    assert guard._line_count(p) == 1


def test_line_count_handles_empty_file(tmp_path) -> None:
    p = tmp_path / "empty.py"
    p.write_text("", encoding="utf-8")
    assert guard._line_count(p) == 0


def test_write_violations_writes_to_stderr(capsys) -> None:
    guard._write_violations(["a", "b"])
    err = capsys.readouterr().err
    assert err == "a\nb\n"


def test_file_length_guard_tracks_torch_core() -> None:
    assert guard.MAX_LINES_BY_FILE["ocmesher/torch_core.py"] == 1100


def test_file_length_guard_uses_tightened_limits() -> None:
    assert guard.MAX_LINES_BY_FILE["benchmark.py"] == 500
    assert guard.MAX_LINES_BY_FILE["benchmarks/run_benchmark.py"] == 1235
    assert guard.MAX_LINES_BY_FILE["tests/test_benchmark_script.py"] == 300
