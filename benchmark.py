"""Benchmark OcMesher with and without optimisations on Apple M4.

Three tiers measured:
  1. Baseline   : install.sh  (-O3, std::map/set)             + vnoise SDF
  2. Opt-C++    : install_optimized.sh (M4-tuned, unordered)  + vnoise SDF
  3. Opt-full   : install_optimized.sh                        + numba-JIT SDF

Usage:
    python benchmark.py              # all three tiers, 3 runs each
    python benchmark.py --runs 5     # custom iterations
    python benchmark.py --configs small   # quick smoke test
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import TypeAlias, TypedDict


class DemoConfig(TypedDict):
    """Benchmark input profile for one demo configuration."""

    label: str
    pixels_per_cube: int
    coarse_count: int


class RunResult(TypedDict):
    """Subprocess run output for one benchmark invocation."""

    elapsed_s: float
    n_verts: int
    n_faces: int


class TierConfigResult(TypedDict):
    """Aggregated stats for one tier/config pair."""

    config: str
    times: list[float]
    mean: float
    median: float
    stdev: float
    min: float
    max: float
    n_verts: int
    n_faces: int


TierSpec: TypeAlias = tuple[str, str, str]


class BenchmarkError(RuntimeError):
    """Base class for benchmark script errors."""


class BenchmarkOutputParseError(BenchmarkError):
    """Raised when benchmark subprocess output does not contain result JSON."""

    def __init__(self, stdout: str) -> None:
        """Store subprocess output that failed JSON parsing."""
        self.stdout = stdout

    def __str__(self) -> str:
        """Return helpful parse error details with subprocess output."""
        return f"No JSON output.\nstdout:\n{self.stdout}"


class MesherSubprocessError(BenchmarkError):
    """Raised when mesher subprocess exits with non-zero status."""

    def __init__(self, stderr: str) -> None:
        """Store stderr captured from a failing subprocess run."""
        self.stderr = stderr

    def __str__(self) -> str:
        """Return subprocess error details with captured stderr."""
        return f"Mesher subprocess failed.\nstderr:\n{self.stderr}"

BASELINE_BUILD = "./install.sh"
OPTIMISED_BUILD = "./install_optimized.sh"

DEMO_CONFIGS: list[DemoConfig] = [
    {"label": "small (ppc=32)", "pixels_per_cube": 32, "coarse_count": 100_000},
    {"label": "medium (ppc=16)", "pixels_per_cube": 16, "coarse_count": 500_000},
    {"label": "large (ppc=8)", "pixels_per_cube": 8, "coarse_count": 500_000},
]

TIERS_SPEC: list[TierSpec] = [
    ("Baseline(-O3+vnoise)", BASELINE_BUILD, "vnoise"),
    ("Opt-C++(M4+unordered)", OPTIMISED_BUILD, "vnoise"),
    ("Opt-numba(+numba-SDF)", OPTIMISED_BUILD, "numba"),
    ("Opt-metal(+MLX-SDF)", OPTIMISED_BUILD, "mlx"),
]

PYTHON = sys.executable
TIER_SEPARATOR_WIDTH = 62
CONFIG_COLUMN_WIDTH = 20
CONFIG_CHOICES = ("small", "medium", "large", "all")

# ── SDF definitions injected verbatim into subprocess scripts ────────

_SDF_VNOISE = """
import vnoise as _vnoise
_noise = _vnoise.Noise()

def sdf(XYZ):
    scale = 2
    h = _noise.noise2(XYZ[:, 0] / scale, XYZ[:, 1] / scale, grid_mode=False, octaves=4)
    return XYZ[:, 2] - h
"""

# Multi-octave sinusoidal terrain: cheap analytic approximation of Perlin noise,
# JIT-compiled with numba for maximum parallel throughput on all M4 cores.
_SDF_NUMBA = """
import numba as _numba
import numpy as _np

@_numba.njit(parallel=True, fastmath=True)
def _terrain_jit(XYZ):
    n = len(XYZ)
    out = _np.empty(n, dtype=_np.float64)
    scale = 2.0
    for i in _numba.prange(n):
        x = XYZ[i, 0] / scale
        y = XYZ[i, 1] / scale
        h = (0.5000 * _np.sin(1.0 * x) * _np.cos(1.0 * y) +
             0.2500 * _np.sin(2.0 * x) * _np.cos(2.0 * y) +
             0.1250 * _np.sin(4.0 * x) * _np.cos(4.0 * y) +
             0.0625 * _np.sin(8.0 * x) * _np.cos(8.0 * y))
        out[i] = XYZ[i, 2] - h
    return out

# Warm up the JIT so compilation is excluded from timing
_terrain_jit(_np.zeros((256, 3), dtype=_np.float64))

def sdf(XYZ):
    return _terrain_jit(XYZ)
"""

# Metal GPU via Apple MLX — same sinusoidal terrain evaluated on M4 GPU.
# Unified memory means no PCIe transfer cost; the GPU operates directly on
# the same physical memory as the CPU.
_SDF_MLX = """
import mlx.core as _mx
import numpy as _np

# Warmup: compile MLX graph on GPU
_warm = _mx.array(_np.zeros((256, 3), dtype=_np.float32))
_mx.eval(_warm[:, 2] - 0.5 * _mx.sin(_warm[:, 0]) * _mx.cos(_warm[:, 1]))

def sdf(XYZ):
    a = _mx.array(XYZ.astype(_np.float32))
    x = a[:, 0] / 2.0
    y = a[:, 1] / 2.0
    h = (0.5000 * _mx.sin(1.0 * x) * _mx.cos(1.0 * y) +
         0.2500 * _mx.sin(2.0 * x) * _mx.cos(2.0 * y) +
         0.1250 * _mx.sin(4.0 * x) * _mx.cos(4.0 * y) +
         0.0625 * _mx.sin(8.0 * x) * _mx.cos(8.0 * y))
    result = a[:, 2] - h
    _mx.eval(result)
    return _np.array(result, dtype=_np.float64)
"""

_MESHER_TEMPLATE = """
import sys, time, json
sys.path.insert(0, '.')
import numpy as np
from ocmesher import OcMesher

{sdf_block}

cam_poses = [np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 3],
    [0, 0, 0, 1],
], dtype=np.float64)]
ks = [np.array([
    [2000, 0, 640],
    [0, 2000, 360],
    [0, 0, 1],
], dtype=np.float64)]
hs = [720]
ws = [1280]
bounds = [-5, 5, -5, 5, -2, 2]

mesher = OcMesher(
    (cam_poses, ks, hs, ws),
    bounds=bounds,
    pixels_per_cube={ppc},
    coarse_count={coarse},
)

t0 = time.perf_counter()
meshes, _ = mesher([sdf])
elapsed = time.perf_counter() - t0

result = {{
    "elapsed_s": elapsed,
    "n_verts": int(meshes[0].vertices.shape[0]),
    "n_faces": int(meshes[0].faces.shape[0]),
}}
print(json.dumps(result))
"""

SDF_BLOCKS = {
    "vnoise": _SDF_VNOISE,
    "numba": _SDF_NUMBA,
    "mlx": _SDF_MLX,
}


def _parse_last_json_line(stdout: str) -> RunResult:
    """Parse the last JSON object line emitted by benchmark subprocess output."""
    for line in _iter_json_candidate_lines(stdout):
        if line.startswith("{"):
            parsed = json.loads(line)
            return {
                "elapsed_s": float(parsed["elapsed_s"]),
                "n_verts": int(parsed["n_verts"]),
                "n_faces": int(parsed["n_faces"]),
            }
    raise BenchmarkOutputParseError(stdout)


def _iter_json_candidate_lines(stdout: str) -> list[str]:
    """Return output lines in reverse order, normalized for JSON scanning."""
    return [line.strip() for line in reversed(stdout.strip().splitlines())]


def build(script: str) -> float:
    """Run a build script and return elapsed wall-clock seconds."""
    t0 = time.perf_counter()
    subprocess.check_call(["/bin/bash", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603
    return time.perf_counter() - t0


def run_mesher_subprocess(pixels_per_cube: int, coarse_count: int, sdf_type: str) -> RunResult:
    """Run OcMesher in a subprocess for one configuration and parse JSON output."""
    script = _build_mesher_script(pixels_per_cube, coarse_count, sdf_type)
    try:
        result = subprocess.run(  # noqa: S603
            [PYTHON, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or str(exc)
        _write_stderr(f"  ERROR:\n{stderr}")
        raise MesherSubprocessError(stderr) from exc
    return _parse_last_json_line(result.stdout)


def _build_mesher_script(pixels_per_cube: int, coarse_count: int, sdf_type: str) -> str:
    """Render the subprocess script for a benchmark run configuration."""
    sdf_block = textwrap.dedent(SDF_BLOCKS[sdf_type])
    return _MESHER_TEMPLATE.format(
        sdf_block=sdf_block,
        ppc=pixels_per_cube,
        coarse=coarse_count,
    )


def run_tier(
    label: str,
    build_script: str,
    sdf_type: str,
    configs: list[DemoConfig],
    runs: int,
) -> list[TierConfigResult]:
    """Benchmark one tier across all selected configs and aggregate run stats."""
    _print_tier_banner(label)
    build_time = build(build_script)
    _write_stdout(f"  Build time: {build_time:.2f}s")
    return [_run_config_benchmark(cfg, sdf_type, runs) for cfg in configs]


def _run_config_benchmark(cfg: DemoConfig, sdf_type: str, runs: int) -> TierConfigResult:
    """Benchmark one config for a tier and return aggregated metrics."""
    times: list[float] = []
    verts = faces = 0
    _write_stdout(f"\n  Config: {cfg['label']}")
    for i in range(runs):
        r = run_mesher_subprocess(cfg["pixels_per_cube"], cfg["coarse_count"], sdf_type)
        times.append(r["elapsed_s"])
        verts = r["n_verts"]
        faces = r["n_faces"]
        _write_stdout(f"    Run {i + 1}/{runs}: {r['elapsed_s']:.3f}s  ({verts} verts, {faces} faces)")
    return _summarize_tier_config(cfg["label"], times, verts, faces, runs)


def _print_tier_banner(label: str) -> None:
    """Print heading banner for one benchmark tier."""
    _write_stdout(f"\n{'=' * TIER_SEPARATOR_WIDTH}")
    _write_stdout(f"  {label}")
    _write_stdout(f"{'=' * TIER_SEPARATOR_WIDTH}")


def _summarize_tier_config(label: str, times: list[float], n_verts: int, n_faces: int, runs: int) -> TierConfigResult:
    """Build one tier/config summary row from run timings and mesh sizes."""
    return {
        "config": label,
        "times": times,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if runs > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "n_verts": n_verts,
        "n_faces": n_faces,
    }


def _short_tier_label(name: str) -> str:
    """Return compact tier label for comparison table columns."""
    if "(" in name:
        return name.split("(", 1)[0].rstrip()
    return name


def print_comparison(tiers: list[tuple[str, list[TierConfigResult]]]) -> None:
    """Print the cross-tier comparison table for median runtime and speedup."""
    names = [t[0] for t in tiers]
    tables = [t[1] for t in tiers]
    n_cols = len(tiers)
    short = [_short_tier_label(n) for n in names]
    cw = _comparison_column_width(short)
    header = _comparison_header(short, cw)
    _print_comparison_banner(n_cols, cw)
    _write_stdout(header)
    _write_stdout("-" * len(header))

    for row in range(len(tables[0])):
        cfg_label = tables[0][row]["config"]
        medians = _tier_medians_at_row(tables, row, n_cols)
        _write_stdout(_comparison_row(cfg_label, medians, cw))
    _write_stdout()


def _comparison_row(cfg_label: str, medians: list[float], col_width: int) -> str:
    """Format one comparison row with medians and baseline speedups."""
    cells = [f"{cfg_label:<{CONFIG_COLUMN_WIDTH}}"]
    cells.extend(f"  {median:>{col_width}.3f}s" for median in medians)
    cells.append(_speedup_cells(medians))
    return "".join(cells)


def _tier_medians_at_row(tables: list[list[TierConfigResult]], row: int, n_cols: int) -> list[float]:
    """Collect median timings for one configuration row across tiers."""
    return [tables[i][row]["median"] for i in range(n_cols)]


def _speedup_cells(medians: list[float]) -> str:
    """Format speedup cell suffix for a median row."""
    baseline = medians[0]
    return "".join(f"  {_speedup_ratio(baseline, median):>5.2f}x" for median in medians[1:])


def _comparison_column_width(short_labels: list[str]) -> int:
    """Return width used for tier columns in comparison table."""
    return max([7, *[len(s) for s in short_labels]])


def _comparison_header(short_labels: list[str], col_width: int) -> str:
    """Build the comparison table header line."""
    header = f"{'Config':<{CONFIG_COLUMN_WIDTH}}"
    for label in short_labels:
        header += f"  {label:>{col_width + 1}}"
    for _ in short_labels[1:]:
        header += f"  {'x/base':>6}"
    return header


def _print_comparison_banner(num_cols: int, col_width: int) -> None:
    """Print title banner for the comparison section."""
    line_w = CONFIG_COLUMN_WIDTH + (col_width + 3) * num_cols + 7 * max(0, num_cols - 1)
    _write_stdout(f"\n{'=' * line_w}")
    _write_stdout("  COMPARISON  (median wall-clock per config)")
    _write_stdout(f"{'=' * line_w}")


def _select_configs(config_choice: str, all_configs: list[DemoConfig]) -> list[DemoConfig]:
    """Return selected benchmark configs for CLI choice."""
    if config_choice == "all":
        return all_configs
    return [c for c in all_configs if c["label"].startswith(config_choice)]


def _validate_runs(runs: int) -> None:
    """Validate that benchmark run count is a positive integer."""
    if runs < 1:
        msg = f"--runs must be >= 1, got {runs}"
        raise ValueError(msg)


def _validate_selected_configs(config_choice: str, configs: list[DemoConfig]) -> None:
    """Validate that config selection produced at least one benchmark profile."""
    if not configs:
        msg = f"No benchmark configs selected for choice '{config_choice}'"
        raise ValueError(msg)


def _write_results_json(output_path: str, results: list[tuple[str, list[TierConfigResult]]]) -> None:
    """Serialize benchmark results to a JSON file."""
    data = _tier_results_to_dict(results)
    with Path(output_path).open("w") as f:
        json.dump(data, f, indent=2)


def _tier_results_to_dict(results: list[tuple[str, list[TierConfigResult]]]) -> dict[str, list[TierConfigResult]]:
    """Convert tier result tuples into a JSON-serializable mapping."""
    return dict(results)


def _speedup_ratio(baseline: float, candidate: float) -> float:
    """Compute baseline / candidate speedup, guarding zero candidate time."""
    return baseline / candidate if candidate > 0 else 0.0


def main() -> None:
    """Parse CLI arguments, run benchmarks, and optionally write JSON output."""
    parser = _create_parser()
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    _validate_runs(args.runs)

    configs = _select_configs(args.configs, DEMO_CONFIGS)
    _validate_selected_configs(args.configs, configs)

    all_results = _collect_tier_results(configs, args.runs)

    print_comparison(all_results)

    if args.json:
        _write_results_json(args.json, all_results)
        _write_stdout(f"Results written to {args.json}")


def _create_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser for benchmark.py."""
    parser = argparse.ArgumentParser(description="Benchmark OcMesher: baseline vs opt-C++ vs opt+numba-SDF")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration (default: 3)")
    parser.add_argument("--configs", default="all", choices=CONFIG_CHOICES)
    parser.add_argument("--json", help="Write results to JSON file")
    return parser


def _collect_tier_results(configs: list[DemoConfig], runs: int) -> list[tuple[str, list[TierConfigResult]]]:
    """Run all configured tiers and return their aggregated results."""
    return [
        (label, run_tier(label, build_script, sdf_type, configs, runs))
        for label, build_script, sdf_type in TIERS_SPEC
    ]


def _write_stdout(message: str = "") -> None:
    """Write one line to stdout without relying on print()."""
    _write_stream(sys.stdout, message)


def _write_stderr(message: str) -> None:
    """Write one line to stderr without relying on print()."""
    _write_stream(sys.stderr, message)


def _write_stream(stream: object, message: str = "") -> None:
    """Write one line to a text stream."""
    stream.write(f"{message}\n")


if __name__ == "__main__":
    main()
