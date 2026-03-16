"""Unit tests for benchmark script helpers."""

from __future__ import annotations

import importlib.util
import json
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
        with pytest.raises(benchmark.BenchmarkOutputParseError, match="No JSON output"):
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


class TestTierSpec:
    def test_tiers_include_expected_sdf_types(self):
        sdf_types = [tier[2] for tier in benchmark.TIERS_SPEC]
        assert sdf_types == ["vnoise", "vnoise", "numba", "mlx"]

    def test_tiers_use_known_build_scripts(self):
        scripts = {tier[1] for tier in benchmark.TIERS_SPEC}
        assert scripts == {benchmark.BASELINE_BUILD, benchmark.OPTIMISED_BUILD}

    def test_each_tier_has_three_fields(self):
        assert all(len(tier) == 3 for tier in benchmark.TIERS_SPEC)


class TestBenchmarkExceptions:
    def test_output_parse_error_str_contains_stdout(self):
        err = benchmark.BenchmarkOutputParseError("line1\nline2")
        text = str(err)
        assert "No JSON output" in text
        assert "line1" in text

    def test_mesher_subprocess_error_str_contains_stderr(self):
        err = benchmark.MesherSubprocessError("traceback")
        text = str(err)
        assert "Mesher subprocess failed" in text
        assert "traceback" in text


class TestSdfBlocks:
    def test_registry_contains_expected_keys(self):
        assert set(benchmark.SDF_BLOCKS) == {"vnoise", "numba", "mlx"}


class TestWriteResultsJson:
    def test_writes_named_tier_results(self, tmp_path):
        out_path = tmp_path / "results.json"
        benchmark._write_results_json(
            str(out_path),
            [("Baseline", [{"config": "small", "times": [1.0], "mean": 1.0, "median": 1.0, "stdev": 0.0, "min": 1.0, "max": 1.0, "n_verts": 10, "n_faces": 20}])],
        )
        data = json.loads(out_path.read_text())
        assert list(data) == ["Baseline"]
        assert data["Baseline"][0]["n_verts"] == 10


class TestSpeedupRatio:
    def test_returns_ratio_for_positive_candidate(self):
        assert benchmark._speedup_ratio(10.0, 2.0) == 5.0

    def test_returns_zero_when_candidate_is_zero(self):
        assert benchmark._speedup_ratio(10.0, 0.0) == 0.0


