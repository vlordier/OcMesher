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


def test_write_stream_accepts_custom_buffer() -> None:
    chunks: list[str] = []

    class _Stream:
        def write(self, text: str) -> None:
            chunks.append(text)

    benchmark._write_stream(_Stream(), "line")
    assert chunks == ["line\n"]


def test_run_config_benchmark_aggregates_single_run(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "run_mesher_subprocess",
        lambda _ppc, _coarse, _sdf: {"elapsed_s": 0.25, "n_verts": 12, "n_faces": 34},
    )

    cfg = {"label": "small", "pixels_per_cube": 32, "coarse_count": 100_000}
    result = benchmark._run_config_benchmark(cfg, "vnoise", 1)
    assert result["config"] == "small"
    assert result["times"] == [0.25]
    assert result["n_verts"] == 12
    assert result["n_faces"] == 34


def test_collect_tier_results_uses_tier_order(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "TIERS_SPEC", [("A", "build-a", "vnoise"), ("B", "build-b", "numba")])

    def _fake_run_tier(label, _build_script, _sdf_type, _configs, _runs):
        return [{"config": label, "times": [1.0], "mean": 1.0, "median": 1.0, "stdev": 0.0, "min": 1.0, "max": 1.0, "n_verts": 1, "n_faces": 2}]

    monkeypatch.setattr(benchmark, "run_tier", _fake_run_tier)
    results = benchmark._collect_tier_results([], 1)
    assert [label for label, _ in results] == ["A", "B"]
