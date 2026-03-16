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


class TestSelectConfigs:
    def test_all_returns_all_configs(self):
        selected = benchmark._select_configs("all", benchmark.DEMO_CONFIGS)
        assert selected == benchmark.DEMO_CONFIGS

    def test_small_selects_small_label(self):
        selected = benchmark._select_configs("small", benchmark.DEMO_CONFIGS)
        assert len(selected) == 1
        assert selected[0]["label"].startswith("small")

    def test_unknown_config_returns_empty_list(self):
        selected = benchmark._select_configs("unknown", benchmark.DEMO_CONFIGS)
        assert selected == []


class TestBenchmarkArgValidation:
    def test_validate_runs_accepts_positive(self):
        benchmark._validate_runs(1)
        benchmark._validate_runs(3)

    def test_validate_runs_rejects_zero(self):
        with pytest.raises(ValueError, match="--runs must be >= 1"):
            benchmark._validate_runs(0)

    def test_validate_selected_configs_accepts_non_empty(self):
        selected = benchmark._select_configs("small", benchmark.DEMO_CONFIGS)
        benchmark._validate_selected_configs("small", selected)

    def test_validate_selected_configs_rejects_empty(self):
        with pytest.raises(ValueError, match="No benchmark configs selected"):
            benchmark._validate_selected_configs("unknown", [])


class TestShortTierLabel:
    def test_trims_suffix_in_parentheses(self):
        assert benchmark._short_tier_label("Opt-numba(+numba-SDF)") == "Opt-numba"

    def test_returns_original_without_parentheses(self):
        assert benchmark._short_tier_label("Baseline") == "Baseline"


