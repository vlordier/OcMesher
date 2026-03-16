"""Unit tests for benchmark script helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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


class TestJsonCandidateLines:
    def test_returns_reversed_stripped_lines(self):
        lines = benchmark._iter_json_candidate_lines(" a \n b\n")
        assert lines == ["b", "a"]


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

    def test_validate_parity_args_accepts_valid_values(self):
        benchmark._validate_parity_args(pixels_per_cube=32, coarse_count=100_000, atol=1e-12)

    def test_validate_parity_args_rejects_bad_pixels_per_cube(self):
        with pytest.raises(ValueError, match="--parity-pixels-per-cube"):
            benchmark._validate_parity_args(pixels_per_cube=0, coarse_count=100_000, atol=1e-12)

    def test_validate_parity_args_rejects_bad_coarse_count(self):
        with pytest.raises(ValueError, match="--parity-coarse-count"):
            benchmark._validate_parity_args(pixels_per_cube=32, coarse_count=0, atol=1e-12)

    def test_validate_parity_args_rejects_negative_atol(self):
        with pytest.raises(ValueError, match="--parity-atol"):
            benchmark._validate_parity_args(pixels_per_cube=32, coarse_count=100_000, atol=-1.0)


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


class TestBuildMesherScript:
    def test_includes_ppc_and_coarse_values(self):
        script = benchmark._build_mesher_script(32, 123456, "vnoise")
        assert "pixels_per_cube=32" in script
        assert "coarse_count=123456" in script

    def test_uses_requested_sdf_block(self):
        script = benchmark._build_mesher_script(16, 100000, "mlx")
        assert "import mlx.core as _mx" in script


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


class TestCreateParser:
    def test_defaults(self):
        parser = benchmark._create_parser()
        args = parser.parse_args([])
        assert args.runs == 3
        assert args.configs == "all"
        assert args.json is None
        assert args.upstream_parity is None
        assert args.upstream_parity_strict is False
        assert args.parity_pixels_per_cube == 32
        assert args.parity_coarse_count == 100_000
        assert args.parity_atol == 1e-12

    def test_upstream_parity_default_ref(self):
        parser = benchmark._create_parser()
        args = parser.parse_args(["--upstream-parity"])
        assert args.upstream_parity == "main"

    def test_upstream_parity_custom_ref(self):
        parser = benchmark._create_parser()
        args = parser.parse_args(["--upstream-parity", "develop"])
        assert args.upstream_parity == "develop"

    def test_upstream_parity_mesh_overrides(self):
        parser = benchmark._create_parser()
        args = parser.parse_args(
            [
                "--upstream-parity",
                "develop",
                "--parity-pixels-per-cube",
                "24",
                "--parity-coarse-count",
                "200000",
                "--parity-atol",
                "1e-9",
            ]
        )
        assert args.parity_pixels_per_cube == 24
        assert args.parity_coarse_count == 200000
        assert args.parity_atol == 1e-9

    def test_config_choices_include_expected_values(self):
        parser = benchmark._create_parser()
        config_arg = next(action for action in parser._actions if action.dest == "configs")
        assert tuple(config_arg.choices) == benchmark.CONFIG_CHOICES


class TestComparisonHelpers:
    def test_column_width_has_minimum_padding(self):
        assert benchmark._comparison_column_width(["A", "BB"]) == 7

    def test_column_width_expands_for_long_labels(self):
        assert benchmark._comparison_column_width(["very-long-label"]) == len("very-long-label")

    def test_header_contains_tiers_and_speedup_columns(self):
        header = benchmark._comparison_header(["Baseline", "Opt"], 8)
        assert "Config" in header
        assert "Baseline" in header
        assert "Opt" in header
        assert "x/base" in header


class TestSummarizeTierConfig:
    def test_summarizes_single_run_with_zero_stdev(self):
        result = benchmark._summarize_tier_config("small", [0.5], 10, 20, 1)
        assert result["config"] == "small"
        assert result["mean"] == 0.5
        assert result["median"] == 0.5
        assert result["stdev"] == 0.0
        assert result["n_verts"] == 10
        assert result["n_faces"] == 20


class TestTierResultsToDict:
    def test_preserves_tier_labels(self):
        results = [
            ("Baseline", [{"config": "small", "times": [1.0], "mean": 1.0, "median": 1.0, "stdev": 0.0, "min": 1.0, "max": 1.0, "n_verts": 1, "n_faces": 2}]),
            ("Opt", [{"config": "small", "times": [0.5], "mean": 0.5, "median": 0.5, "stdev": 0.0, "min": 0.5, "max": 0.5, "n_verts": 1, "n_faces": 2}]),
        ]
        mapped = benchmark._tier_results_to_dict(results)
        assert list(mapped.keys()) == ["Baseline", "Opt"]


class TestPrintTierBanner:
    def test_prints_label_and_separator(self, capsys):
        benchmark._print_tier_banner("Opt-C++")
        out = capsys.readouterr().out
        assert "Opt-C++" in out
        assert "=" * benchmark.TIER_SEPARATOR_WIDTH in out


class TestLayoutConstants:
    def test_comparison_header_uses_config_column_width(self):
        header = benchmark._comparison_header(["Base"], 7)
        assert header.startswith(f"{'Config':<{benchmark.CONFIG_COLUMN_WIDTH}}")


class TestCollectTierResults:
    def test_collects_all_tier_labels(self, monkeypatch):
        monkeypatch.setattr(benchmark, "TIERS_SPEC", [("A", "build-a", "vnoise"), ("B", "build-b", "numba")])

        def _fake_run_tier(label, _build_script, _sdf_type, _configs, _runs):
            return [{"config": label, "times": [1.0], "mean": 1.0, "median": 1.0, "stdev": 0.0, "min": 1.0, "max": 1.0, "n_verts": 1, "n_faces": 2}]

        monkeypatch.setattr(benchmark, "run_tier", _fake_run_tier)
        results = benchmark._collect_tier_results([], 1)
        assert [label for label, _ in results] == ["A", "B"]


class TestComparisonRow:
    def test_formats_medians_and_speedup_columns(self):
        row = benchmark._comparison_row("small", [1.0, 0.5], 7)
        assert row.startswith("small")
        assert "1.000s" in row
        assert "0.500s" in row
        assert "2.00x" in row

    def test_includes_one_speedup_per_non_baseline_tier(self):
        row = benchmark._comparison_row("small", [2.0, 1.0, 0.5], 7)
        assert row.count("x") >= 2


class TestComparisonCells:
    def test_speedup_cells_formats_suffix(self):
        assert benchmark._speedup_cells([2.0, 1.0, 0.5]) == "   2.00x   4.00x"

    def test_tier_medians_at_row_collects_expected_values(self):
        tables = [
            [{"median": 1.0}],
            [{"median": 0.5}],
        ]
        medians = benchmark._tier_medians_at_row(tables, 0, 2)
        assert medians == [1.0, 0.5]


class TestMainUpstreamParityMode:
    def test_main_runs_upstream_parity_and_returns(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["benchmark.py", "--upstream-parity"])
        calls: list[dict[str, object]] = []

        def _fake_run_upstream_parity(*_args, **kwargs):
            calls.append(kwargs)
            return {"delta": {"matches": True}}

        monkeypatch.setattr(benchmark, "run_upstream_parity", _fake_run_upstream_parity)

        benchmark.main()

        out = capsys.readouterr().out
        assert '"matches": true' in out
        assert calls[0]["mesh_config"] == {"pixels_per_cube": 32, "coarse_count": 100_000}
        assert calls[0]["atol"] == 1e-12

    def test_main_strict_mode_exits_nonzero_on_mismatch(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["benchmark.py", "--upstream-parity", "--upstream-parity-strict"],
        )
        monkeypatch.setattr(benchmark, "run_upstream_parity", lambda *_args, **_kwargs: {"delta": {"matches": False}})

        with pytest.raises(SystemExit, match="1"):
            benchmark.main()

        err = capsys.readouterr().err
        assert "Upstream parity mismatch" in err

    def test_main_parity_mode_writes_json_when_requested(self, monkeypatch, tmp_path):
        out_path = tmp_path / "parity.json"
        monkeypatch.setattr(
            "sys.argv",
            ["benchmark.py", "--upstream-parity", "--json", str(out_path)],
        )
        monkeypatch.setattr(
            benchmark,
            "run_upstream_parity",
            lambda *_args, **_kwargs: {"delta": {"matches": True}, "current": {}, "upstream": {}},
        )

        benchmark.main()

        written = json.loads(out_path.read_text())
        assert bool(written["delta"]["matches"])

    def test_main_rejects_strict_without_upstream_parity(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["benchmark.py", "--upstream-parity-strict"])

        with pytest.raises(SystemExit, match="2"):
            benchmark.main()

        err = capsys.readouterr().err
        assert "requires --upstream-parity" in err


