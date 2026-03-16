"""Enforce file-length limits for top-level scripts and benchmark modules."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

LineLimits: TypeAlias = dict[str, int]

MAX_LINES_BY_FILE: LineLimits = {
    "benchmark.py": 520,
    "benchmarks/run_benchmark.py": 1250,
    "benchmarks/bench_e2e.py": 750,
    "benchmarks/bench_python_overhead.py": 430,
    "tests/test_benchmark_script.py": 320,
}


def _line_count(path: Path) -> int:
    return path.read_text(encoding="utf-8").count("\n") + 1


def _file_length_violations(repo_root: Path, limits: LineLimits | None = None) -> list[str]:
    limits = limits or MAX_LINES_BY_FILE
    violations = []
    for rel_path, max_lines in sorted(limits.items()):
        abs_path = repo_root / rel_path
        n_lines = _line_count(abs_path)
        if n_lines > max_lines:
            violations.append(f"{rel_path}: {n_lines} lines (limit {max_lines})")
    return violations


def test_file_length_policy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    violations = _file_length_violations(repo_root)
    assert not violations, "\n".join(violations)


def test_file_length_violations_reports_offending_file(tmp_path) -> None:
    test_file = tmp_path / "example.py"
    test_file.write_text("a\n" * 10, encoding="utf-8")
    violations = _file_length_violations(tmp_path, {"example.py": 5})
    assert violations == ["example.py: 11 lines (limit 5)"]
