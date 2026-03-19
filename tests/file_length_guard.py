"""File-length policy guard for high-churn scripts.

Run manually:
    python tests/file_length_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_LINES_BY_FILE: dict[str, int] = {
    "benchmarks/run_benchmark.py": 1400,
    "ocmesher/core.py": 1000,
    "ocmesher/torch_core.py": 1700,
    "ocmesher/mlx_core.py": 1400,
    "ocmesher/rust_backend.py": 700,
}


def _line_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if not text:
        return 0
    return len(text.splitlines())


def _file_length_violations(repo_root: Path) -> list[str]:
    violations: list[str] = []
    for rel_path, max_lines in sorted(MAX_LINES_BY_FILE.items()):
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            violations.append(f"{rel_path}: missing file (configured limit {max_lines})")
            continue
        n_lines = _line_count(abs_path)
        if n_lines > max_lines:
            violations.append(f"{rel_path}: {n_lines} lines (limit {max_lines})")
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations = _file_length_violations(repo_root)
    if not violations:
        return 0

    _write_violations(violations)
    return 1


def _write_violations(violations: list[str]) -> None:
    """Emit one violation per line to stderr."""
    for violation in violations:
        sys.stderr.write(f"{violation}\n")


if __name__ == "__main__":
    raise SystemExit(main())
