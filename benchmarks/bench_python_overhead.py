"""Micro-benchmark: measure Python orchestration overhead in ocmesher.core.

This benchmark does NOT require the C++ shared library.  It exercises the
pure-Python hot-path code that wraps the C++ calls — buffer allocation,
bounds masking, SDF batching — using mock SDF kernels.

Run from the repository root::

    python -m benchmarks.bench_python_overhead
    python -m benchmarks.bench_python_overhead --sizes 1000 10000 100000 1000000
    python -m benchmarks.bench_python_overhead --json results.json

To compare branches (e.g. ``develop`` vs this PR)::

    git checkout develop
    python -m benchmarks.bench_python_overhead --json develop.json

    git checkout copilot/refactor-algorithms-for-speed
    python -m benchmarks.bench_python_overhead --json optimised.json

    python -c "
    import json, sys
    a = json.load(open('develop.json'))
    b = json.load(open('optimised.json'))
    for name in a:
        for sz in a[name]:
            t_a = a[name][sz]['mean_us']
            t_b = b[name][sz]['mean_us']
            speedup = t_a / t_b if t_b > 0 else float('inf')
            print(f'{name:40s}  N={sz:>8s}  develop={t_a:10.1f}µs  opt={t_b:10.1f}µs  speedup={speedup:.2f}x')
    "
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from ocmesher.core import (
    _SDF_BATCH_SIZE,
    OcMesher,
    _np_empty,
)


# ---------------------------------------------------------------------------
# Mock SDF kernels (no C++ dependency)
# ---------------------------------------------------------------------------
def _sdf_sphere(xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz, axis=1) - 5.0


def _sdf_plane(xyz: np.ndarray) -> np.ndarray:
    return xyz[:, 2].copy()


def _sdf_gyroid(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_stub(*, enclosed: bool = True) -> OcMesher:
    """Create an OcMesher stub without loading the C++ DLL."""
    obj = object.__new__(OcMesher)
    obj.enclosed = enclosed
    obj.sdf_np_float_type = np.float32
    obj._sdf_pool = None
    obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
    obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
    obj._bounds_min_np = np.array([-10.0, -10.0, -10.0], dtype=np.float64)
    obj._bounds_max_np = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    return obj


def _time_fn(fn, *, warmup: int = 2, repeats: int = 10) -> dict:
    """Time *fn()* and return statistics in microseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns → µs
    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
        "stdev_us": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def bench_kernel_caller_single(sizes: list[int]) -> dict:
    """Benchmark kernel_caller with a single SDF kernel."""
    stub = _make_stub(enclosed=True)
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3))
        results[str(n)] = _time_fn(lambda: stub.kernel_caller([_sdf_sphere], pts))
    return results


def bench_kernel_caller_multi(sizes: list[int]) -> dict:
    """Benchmark kernel_caller with 3 SDF kernels (thread pool)."""
    stub = _make_stub(enclosed=True)
    kernels = [_sdf_sphere, _sdf_plane, _sdf_gyroid]
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3))
        results[str(n)] = _time_fn(lambda: stub.kernel_caller(kernels, pts))
    return results


def bench_kernel_caller_out_reuse(sizes: list[int]) -> dict:
    """Benchmark kernel_caller with pre-allocated out= buffer."""
    stub = _make_stub(enclosed=True)
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3))
        out = _np_empty((n, 1), dtype=np.float32)
        results[str(n)] = _time_fn(lambda: stub.kernel_caller([_sdf_sphere], pts, out=out))
    return results


def bench_bounds_mask_static(sizes: list[int]) -> dict:
    """Benchmark the static _out_of_bounds_mask method."""
    b_min = np.array([-10.0, -10.0, -10.0])
    b_max = np.array([10.0, 10.0, 10.0])
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3)) * 15
        results[str(n)] = _time_fn(lambda: OcMesher._out_of_bounds_mask(pts, b_min, b_max))
    return results


def bench_bounds_mask_into(sizes: list[int]) -> dict:
    """Benchmark the reusable-buffer _out_of_bounds_mask_into method."""
    stub = _make_stub()
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((min(n, _SDF_BATCH_SIZE), 3)) * 15
        results[str(n)] = _time_fn(lambda: stub._out_of_bounds_mask_into(pts, stub._bounds_min_np, stub._bounds_max_np))
    return results


def bench_slice_fill_vs_concatenate(sizes: list[int]) -> dict:
    """Compare pre-allocated slice-fill vs np.concatenate."""
    results = {}
    for n in sizes:
        a = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)
        b = np.random.default_rng(43).standard_normal((n, 3)).astype(np.float64)

        def _concat():
            return np.concatenate((a, b))

        def _slice_fill():
            buf = _np_empty((2 * n, 3), dtype=np.float64)
            buf[:n] = a
            buf[n:] = b
            return buf

        t_concat = _time_fn(_concat)
        t_slice = _time_fn(_slice_fill)
        results[str(n)] = {
            "concat_mean_us": t_concat["mean_us"],
            "slice_fill_mean_us": t_slice["mean_us"],
            "speedup": t_concat["mean_us"] / t_slice["mean_us"] if t_slice["mean_us"] > 0 else 0,
        }
    return results


def bench_growable_buffer_pattern(sizes: list[int]) -> dict:
    """Compare per-iteration alloc vs growable buffer reuse."""
    results = {}
    for n in sizes:
        iters = 15  # typical bisection iteration count
        pts_seq = [np.random.default_rng(42 + i).standard_normal((n, 3)).astype(np.float64) for i in range(iters)]

        def _alloc_each():
            for p in pts_seq:
                buf = _np_empty((n, 3), dtype=np.float64)
                buf[:] = p

        def _growable():
            cap = 0
            buf = None
            for p in pts_seq:
                nn = len(p)
                if nn > cap:
                    buf = _np_empty((nn, 3), dtype=np.float64)
                    cap = nn
                buf[:nn] = p

        t_alloc = _time_fn(_alloc_each)
        t_grow = _time_fn(_growable)
        results[str(n)] = {
            "alloc_each_mean_us": t_alloc["mean_us"],
            "growable_mean_us": t_grow["mean_us"],
            "speedup": t_alloc["mean_us"] / t_grow["mean_us"] if t_grow["mean_us"] > 0 else 0,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_BENCHMARKS = {
    "kernel_caller_single": bench_kernel_caller_single,
    "kernel_caller_multi_3k": bench_kernel_caller_multi,
    "kernel_caller_out_reuse": bench_kernel_caller_out_reuse,
    "bounds_mask_static": bench_bounds_mask_static,
    "bounds_mask_into": bench_bounds_mask_into,
    "slice_fill_vs_concatenate": bench_slice_fill_vs_concatenate,
    "growable_buffer_pattern": bench_growable_buffer_pattern,
}


def main():
    parser = argparse.ArgumentParser(description="Micro-benchmark for Python orchestration overhead")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 1_000, 10_000, 100_000, 1_000_000],
        help="Point counts to benchmark",
    )
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--repeats", type=int, default=10, help="Number of timing repeats")
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print("OcMesher Python Orchestration Micro-Benchmark")
    print(f"{'=' * 70}")
    print(f"Sizes: {args.sizes}")
    print(f"Repeats: {args.repeats}")
    print()

    all_results = {}
    for name, fn in _BENCHMARKS.items():
        print(f"--- {name} ---")
        results = fn(args.sizes)
        all_results[name] = results
        for sz, stats in results.items():
            if "mean_us" in stats:
                print(
                    f"  N={sz:>8s}  mean={stats['mean_us']:10.1f}µs  "
                    f"median={stats['median_us']:10.1f}µs  min={stats['min_us']:10.1f}µs"
                )
            elif "speedup" in stats:
                items = "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items())
                print(f"  N={sz:>8s}  {items}")
        print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.json}")

    print(f"{'=' * 70}")
    print("To compare branches, run this on each branch with --json and compare:")
    print("  git checkout develop && python -m benchmarks.bench_python_overhead --json develop.json")
    print("  git checkout <this-branch> && python -m benchmarks.bench_python_overhead --json opt.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
