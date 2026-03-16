"""Narrow MLX pilot: benchmark only SDF evaluation against NumPy on macOS."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from benchmarks.result_utils import compare_flat_results, write_snapshot_json


def _write_stdout(message: str) -> None:
    """Write one line to stdout without relying on print()."""
    sys_stdout = __import__("sys").stdout
    sys_stdout.write(f"{message}\n")


def _numpy_sphere_sdf(xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz, axis=1) - 5.0


def _mlx_available() -> bool:
    try:
        importlib.import_module("mlx.core")
    except ImportError:
        return False
    return True


def _mlx_sphere_sdf(xyz: np.ndarray) -> np.ndarray:
    mx = importlib.import_module("mlx.core")

    xyz_mx = mx.array(xyz.astype(np.float32, copy=False))
    sdf = mx.sqrt(mx.sum(xyz_mx * xyz_mx, axis=1)) - 5.0
    return np.array(sdf)


def _time_fn(fn, *, repeats: int) -> dict[str, float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
        "stdev_us": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def _run_sizes(sizes: list[int], repeats: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(42)
    results: dict[str, dict[str, float]] = {}
    for size in sizes:
        xyz = rng.standard_normal((size, 3), dtype=np.float32)
        np_stats = _time_fn(lambda xyz=xyz: _numpy_sphere_sdf(xyz), repeats=repeats)
        row = {f"numpy_{k}": v for k, v in np_stats.items()}
        if _mlx_available():
            mlx_stats = _time_fn(lambda xyz=xyz: _mlx_sphere_sdf(xyz), repeats=repeats)
            row.update({f"mlx_{k}": v for k, v in mlx_stats.items()})
            row["speedup"] = np_stats["mean_us"] / mlx_stats["mean_us"] if mlx_stats["mean_us"] > 0 else 0.0
        results[str(size)] = row
    return results


def main() -> None:
    """Run the narrow MLX-vs-NumPy SDF benchmark pilot."""
    parser = argparse.ArgumentParser(description="MLX SDF-only pilot benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--snapshot-json", type=str, default=None)
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = parser.parse_args()

    if args.compare:
        for line in compare_flat_results(args.compare[0], args.compare[1]):
            _write_stdout(line)
        return

    results = {"sphere_sdf": _run_sizes(args.sizes, args.repeats)}
    for size, stats in results["sphere_sdf"].items():
        summary = "  ".join(f"{k}={v:.1f}" for k, v in stats.items())
        _write_stdout(f"N={size:>8s}  {summary}")

    if args.json:
        with Path(args.json).open("w") as f:
            json.dump(results, f, indent=2)
        _write_stdout(f"Results saved to {args.json}")
    if args.snapshot_json:
        write_snapshot_json(args.snapshot_json, results, label="bench-mlx-sdf")
        _write_stdout(f"Snapshot saved to {args.snapshot_json}")


if __name__ == "__main__":
    main()
