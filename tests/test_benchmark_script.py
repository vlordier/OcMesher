"""Unit tests for benchmark script helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[1] / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_script", module_path)
    if spec is None or spec.loader is None:
        msg = f"Failed to load benchmark module from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


class TestParseLastJsonLine:
    def test_parses_single_json_line(self):
        stdout = '{"elapsed_s": 1.25, "n_verts": 123, "n_faces": 456}\n'
        result = benchmark._parse_last_json_line(stdout)
        assert result == {"elapsed_s": 1.25, "n_verts": 123, "n_faces": 456}

    def test_parses_last_json_when_logs_precede_output(self):
        stdout = 'build complete\nrun 1 done\n{"elapsed_s": 0.5, "n_verts": 10, "n_faces": 20}\n'
        result = benchmark._parse_last_json_line(stdout)
        assert result["elapsed_s"] == 0.5
        assert result["n_verts"] == 10
        assert result["n_faces"] == 20

    def test_raises_when_no_json_line(self):
        with pytest.raises(RuntimeError, match="No JSON output"):
            benchmark._parse_last_json_line("hello\nworld\n")
