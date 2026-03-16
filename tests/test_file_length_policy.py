"""Enforce file-length limits for top-level scripts and benchmark modules."""

from __future__ import annotations

from pathlib import Path

MAX_LINES_BY_FILE = {
    "benchmark.py": 520,
    "benchmarks/run_benchmark.py": 1250,
    "benchmarks/bench_e2e.py": 750,
    "benchmarks/bench_python_overhead.py": 430,
    "tests/test_benchmark_script.py": 320,
}


def _line_count(path: Path) -> int:
    return path.read_text(encoding="utf-8").count("\n") + 1


def test_file_length_policy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    violations = []
    for rel_path, max_lines in MAX_LINES_BY_FILE.items():
        abs_path = repo_root / rel_path
        n_lines = _line_count(abs_path)
        if n_lines > max_lines:
            violations.append(f"{rel_path}: {n_lines} lines (limit {max_lines})")

    assert not violations, "\n".join(violations)
