#!/usr/bin/env python3
"""
Benchmark OcMesher with and without optimisations on Apple M4.

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
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict


class DemoConfig(TypedDict):
    label: str
    pixels_per_cube: int
    coarse_count: int


class RunResult(TypedDict):
    elapsed_s: float
    n_verts: int
    n_faces: int


class TierConfigResult(TypedDict):
    config: str
    times: list[float]
    mean: float
    median: float
    stdev: float
    min: float
    max: float
    n_verts: int
    n_faces: int

BASELINE_BUILD = "./install.sh"
OPTIMISED_BUILD = "./install_optimized.sh"

DEMO_CONFIGS: list[DemoConfig] = [
    {"label": "small (ppc=32)", "pixels_per_cube": 32, "coarse_count": 100_000},
    {"label": "medium (ppc=16)", "pixels_per_cube": 16, "coarse_count": 500_000},
    {"label": "large (ppc=8)", "pixels_per_cube": 8, "coarse_count": 500_000},
]

PYTHON = sys.executable

# ── SDF definitions injected verbatim into subprocess scripts ────────

_SDF_VNOISE = """
import vnoise as _vnoise
_noise = _vnoise.Noise()

def sdf(XYZ):
    scale = 2
    h = _noise.noise2(XYZ[:, 0] / scale, XYZ[:, 1] / scale, grid_mode=False, octaves=4)
    return XYZ[:, 2] - h
"""

_SDF_ANALYTIC = """
import numpy as _np

def sdf(XYZ):
    x = XYZ[:, 0] / 2.0
    y = XYZ[:, 1] / 2.0
    h = (0.5000 * _np.sin(1.0 * x) * _np.cos(1.0 * y) +
         0.2500 * _np.sin(2.0 * x) * _np.cos(2.0 * y) +
         0.1250 * _np.sin(4.0 * x) * _np.cos(4.0 * y) +
         0.0625 * _np.sin(8.0 * x) * _np.cos(8.0 * y))
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
Ks = [np.array([
    [2000, 0, 640],
    [0, 2000, 360],
    [0, 0, 1],
], dtype=np.float64)]
Hs = [720]
Ws = [1280]
bounds = [-5, 5, -5, 5, -2, 2]

mesher = OcMesher(
    (cam_poses, Ks, Hs, Ws),
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


def _parse_last_json_line(stdout: str) -> RunResult:
    """Parse the last JSON object line emitted by benchmark subprocess output."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            parsed = json.loads(line)
            return {
                "elapsed_s": float(parsed["elapsed_s"]),
                "n_verts": int(parsed["n_verts"]),
                "n_faces": int(parsed["n_faces"]),
            }
    raise RuntimeError(f"No JSON output.\nstdout:\n{stdout}")


def build(script: str) -> float:
    t0 = time.perf_counter()
    subprocess.check_call(["bash", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def _is_module_importable(module: str) -> bool:
    """Return True if *module* can be imported by the benchmark Python interpreter."""
    cmd = [PYTHON, "-c", f"import {module}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0


def _build_tiers_spec(
    module_check: Callable[[str], bool] | None = None,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Build benchmark tiers based on optional dependency availability."""
    check = module_check or _is_module_importable
    notes: list[str] = []

    if check("vnoise"):
        base_label = "Baseline(-O3+vnoise)"
        opt_cpp_label = "Opt-C++(M4+unordered)"
        base_sdf = "vnoise"
    else:
        base_label = "Baseline(-O3+analytic)"
        opt_cpp_label = "Opt-C++(M4+unordered+analytic)"
        base_sdf = "analytic"
        notes.append("vnoise unavailable; using built-in analytic terrain fallback for baseline tiers.")

    tiers: list[tuple[str, str, str]] = [
        (base_label, BASELINE_BUILD, base_sdf),
        (opt_cpp_label, OPTIMISED_BUILD, base_sdf),
    ]

    if check("numba"):
        tiers.append(("Opt-numba(+numba-SDF)", OPTIMISED_BUILD, "numba"))
    else:
        notes.append("numba unavailable; skipping Opt-numba tier.")

    if check("mlx"):
        tiers.append(("Opt-metal(+MLX-SDF)", OPTIMISED_BUILD, "mlx"))
    else:
        notes.append("mlx unavailable; skipping Opt-metal tier.")

    return tiers, notes


def run_mesher_subprocess(pixels_per_cube: int, coarse_count: int, sdf_type: str) -> RunResult:
    sdf_blocks = {
        "analytic": _SDF_ANALYTIC,
        "vnoise": _SDF_VNOISE,
        "numba": _SDF_NUMBA,
        "mlx": _SDF_MLX,
    }
    sdf_block = textwrap.dedent(sdf_blocks[sdf_type])
    script = _MESHER_TEMPLATE.format(
        sdf_block=sdf_block,
        ppc=pixels_per_cube,
        coarse=coarse_count,
    )
    result = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("Mesher subprocess failed")
    return _parse_last_json_line(result.stdout)


def run_tier(
    label: str,
    build_script: str,
    sdf_type: str,
    configs: list[DemoConfig],
    runs: int,
) -> list[TierConfigResult]:
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    build_time = build(build_script)
    print(f"  Build time: {build_time:.2f}s")
    results = []
    for cfg in configs:
        times = []
        verts = faces = 0
        print(f"\n  Config: {cfg['label']}")
        for i in range(runs):
            r = run_mesher_subprocess(cfg["pixels_per_cube"], cfg["coarse_count"], sdf_type)
            times.append(r["elapsed_s"])
            verts = r["n_verts"]
            faces = r["n_faces"]
            print(f"    Run {i + 1}/{runs}: {r['elapsed_s']:.3f}s  ({verts} verts, {faces} faces)")
        results.append(
            {
                "config": cfg["label"],
                "times": times,
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if runs > 1 else 0.0,
                "min": min(times),
                "max": max(times),
                "n_verts": verts,
                "n_faces": faces,
            }
        )
    return results


def _short_tier_label(name: str) -> str:
    """Return compact tier label for comparison table columns."""
    if "(" in name:
        return name.split("(", 1)[0].rstrip()
    return name


def print_comparison(tiers: list[tuple[str, list[TierConfigResult]]]) -> None:
    """tiers: list of (label, results)"""
    names = [t[0] for t in tiers]
    tables = [t[1] for t in tiers]
    n_cols = len(tiers)
    short = [_short_tier_label(n) for n in names]
    cw = max(7, max(len(s) for s in short))

    line_w = 20 + (cw + 3) * n_cols + 7 * max(0, n_cols - 1)
    print(f"\n{'=' * line_w}")
    print("  COMPARISON  (median wall-clock per config)")
    print(f"{'=' * line_w}")

    header = f"{'Config':<20}"
    for s in short:
        header += f"  {s:>{cw + 1}}"
    for _ in short[1:]:
        header += f"  {'x/base':>6}"
    print(header)
    print("-" * len(header))

    for row in range(len(tables[0])):
        cfg_label = tables[0][row]["config"]
        medians = [tables[i][row]["median"] for i in range(n_cols)]
        line = f"{cfg_label:<20}"
        for m in medians:
            line += f"  {m:>{cw}.3f}s"
        for m in medians[1:]:
            sp = medians[0] / m if m > 0 else 0
            line += f"  {sp:>5.2f}x"
        print(line)
    print()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OcMesher: baseline vs opt-C++ vs opt+numba-SDF")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration (default: 3)")
    parser.add_argument("--configs", default="all", choices=["small", "medium", "large", "all"])
    parser.add_argument("--json", help="Write results to JSON file")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    _validate_runs(args.runs)

    configs = _select_configs(args.configs, DEMO_CONFIGS)
    _validate_selected_configs(args.configs, configs)

    tiers_spec, notes = _build_tiers_spec()
    for note in notes:
        print(f"Note: {note}")

    all_results = []
    for label, build_script, sdf_type in tiers_spec:
        results = run_tier(label, build_script, sdf_type, configs, args.runs)
        all_results.append((label, results))

    print_comparison(all_results)

    if args.json:
        data = {label: res for label, res in all_results}
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
