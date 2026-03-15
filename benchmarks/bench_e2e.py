"""End-to-end benchmark: compare Python orchestration layer across branches.

This benchmark exercises the **full Python orchestration layer** that wraps the
C++ meshing calls.  It does NOT require the compiled C++ shared library — it
uses mock SDF kernels and stub objects to isolate the Python-side overhead that
all 64+ optimizations target.

The benchmark is designed to run identically on ``main``, ``develop``, and the
optimized ``copilot/refactor-algorithms-for-speed`` branch so results can be
compared directly.

Usage::

    # Run on the current branch
    python -m benchmarks.bench_e2e

    # Save results to JSON for branch comparison
    python -m benchmarks.bench_e2e --json results.json

    # Compare two branches
    python -m benchmarks.bench_e2e --compare develop.json optimized.json

    # Custom sizes and repeats
    python -m benchmarks.bench_e2e --sizes 10000 100000 1000000 --repeats 5
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import textwrap
import time

import numpy as np

# ---------------------------------------------------------------------------
# Adaptive import: works on develop/main (no __slots__, no _SDF_BATCH_SIZE
# export, no _np_empty export) and on the optimized branch.
# ---------------------------------------------------------------------------
from ocmesher.core import OcMesher

try:
    from ocmesher.core import _SDF_BATCH_SIZE
except ImportError:
    _SDF_BATCH_SIZE = 10_000_000  # same value used on all branches

_HAS_SLOTS = hasattr(OcMesher, "__slots__")


# ---------------------------------------------------------------------------
# Mock SDF kernels (pure-Python — no C++ dependency)
# ---------------------------------------------------------------------------
def _sdf_sphere(xyz: np.ndarray) -> np.ndarray:
    """Sphere: distance to origin minus radius."""
    return np.linalg.norm(xyz, axis=1).astype(np.float32) - 5.0


def _sdf_plane(xyz: np.ndarray) -> np.ndarray:
    """Plane: z-height."""
    return xyz[:, 2].astype(np.float32)


def _sdf_gyroid(xyz: np.ndarray) -> np.ndarray:
    """Gyroid: trigonometric implicit surface."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)).astype(np.float32)


# ---------------------------------------------------------------------------
# Stub factory: create an OcMesher without the C++ DLL
# ---------------------------------------------------------------------------
def _make_stub(*, enclosed: bool = True, n_cameras: int = 4) -> OcMesher:
    """Create an OcMesher stub without loading the C++ DLL.

    Sets up just enough state for ``kernel_caller``, bounds masking, and
    camera packing to work.  Adapts to the branch (slots vs dict).
    """
    obj = object.__new__(OcMesher)

    # --- Attributes needed by kernel_caller / bounds masking ---------------
    obj.enclosed = enclosed
    obj.sdf_np_float_type = np.float32

    # Branch-adaptive: __slots__ branch has _bounds_min_np/_bounds_max_np;
    # develop/main uses self.bounds (a 6-tuple).
    if _HAS_SLOTS:
        obj._bounds_min_np = np.array([-10.0, -10.0, -10.0], dtype=np.float64)
        obj._bounds_max_np = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
    else:
        obj.bounds = (-10, 10, -10, 10, -10, 10)

    return obj


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def _time_fn(fn, *, warmup: int = 3, repeats: int = 10) -> dict:
    """Time *fn()* and return statistics in microseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1_000.0)  # ns → µs
    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
        "stdev_us": statistics.stdev(times) if len(times) > 1 else 0.0,
        "repeats": repeats,
    }


# ---------------------------------------------------------------------------
# Check if kernel_caller supports out= parameter
# ---------------------------------------------------------------------------
def _supports_out_param() -> bool:
    sig = inspect.signature(OcMesher.kernel_caller)
    return "out" in sig.parameters


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def bench_kernel_caller_single_enclosed(sizes: list[int], repeats: int) -> dict:
    """Single SDF kernel with bounds checking (enclosed=True).

    This is the most common configuration and the hottest code path.
    Tests: batching, bounds masking, SDF evaluation, result buffer fill.
    """
    stub = _make_stub(enclosed=True)
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)
        results[str(n)] = _time_fn(
            lambda: stub.kernel_caller([_sdf_sphere], pts),
            repeats=repeats,
        )
    return results


def bench_kernel_caller_single_open(sizes: list[int], repeats: int) -> dict:
    """Single SDF kernel without bounds checking (enclosed=False).

    Tests the fast path where no bounds masking is needed.
    """
    stub = _make_stub(enclosed=False)
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)
        results[str(n)] = _time_fn(
            lambda: stub.kernel_caller([_sdf_sphere], pts),
            repeats=repeats,
        )
    return results


def bench_kernel_caller_multi_3k(sizes: list[int], repeats: int) -> dict:
    """Three SDF kernels with thread pool (enclosed=True).

    Tests: multi-kernel dispatch, thread pool management, result assembly.
    """
    stub = _make_stub(enclosed=True)
    kernels = [_sdf_sphere, _sdf_plane, _sdf_gyroid]
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)
        results[str(n)] = _time_fn(
            lambda: stub.kernel_caller(kernels, pts),
            repeats=repeats,
        )
    return results


def bench_kernel_caller_out_reuse(sizes: list[int], repeats: int) -> dict:
    """Single kernel with pre-allocated output buffer (out= parameter).

    Only runs on the optimized branch.  On develop/main, falls back to
    the standard single-kernel path for a fair comparison.
    """
    stub = _make_stub(enclosed=True)
    has_out = _supports_out_param()
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)
        if has_out:
            out = np.empty((n, 1), dtype=np.float32)
            results[str(n)] = _time_fn(
                lambda: stub.kernel_caller([_sdf_sphere], pts, out=out),
                repeats=repeats,
            )
        else:
            results[str(n)] = _time_fn(
                lambda: stub.kernel_caller([_sdf_sphere], pts),
                repeats=repeats,
            )
        results[str(n)]["has_out_param"] = has_out
    return results


def bench_bounds_mask(sizes: list[int], repeats: int) -> dict:
    """Out-of-bounds masking performance.

    On the optimized branch, tests the unrolled per-axis ufunc path.
    On develop/main, tests the original per-axis loop.
    """
    has_static = hasattr(OcMesher, "_out_of_bounds_mask")
    if has_static:
        b_min = np.array([-10.0, -10.0, -10.0])
        b_max = np.array([10.0, 10.0, 10.0])
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64) * 15.0
        if has_static:
            results[str(n)] = _time_fn(
                lambda: OcMesher._out_of_bounds_mask(pts, b_min, b_max),
                repeats=repeats,
            )
        else:
            # Reference implementation matching develop/main
            bounds = (-10, 10, -10, 10, -10, 10)

            def _mask():
                out_bound = np.zeros(len(pts), dtype=bool)
                for c in range(3):
                    out_bound |= pts[:, c] <= bounds[c * 2]
                    out_bound |= pts[:, c] >= bounds[c * 2 + 1]
                return out_bound

            results[str(n)] = _time_fn(_mask, repeats=repeats)
        results[str(n)]["optimized_path"] = has_static
    return results


def bench_camera_packing(sizes: list[int], repeats: int) -> dict:
    """Camera data packing (vectorized vs per-camera loop).

    Tests the __init__ camera packing with varying number of cameras.
    """
    results = {}
    for n_cameras in [1, 4, 8, 16, 32]:
        cam_poses = [
            np.eye(4, dtype=np.float64) + np.random.default_rng(42 + i).standard_normal((4, 4)) * 0.1
            for i in range(n_cameras)
        ]
        # Ensure invertible
        for p in cam_poses:
            p[3, :] = [0, 0, 0, 1]

        ks = [np.array([[2000, 0, 640], [0, 2000, 360], [0, 0, 1]], dtype=np.float64)] * n_cameras
        hs = [720] * n_cameras
        ws = [1280] * n_cameras

        np_float_type = np.float64

        def _pack_vectorized():
            """Vectorized camera packing (optimized branch style)."""
            inv_poses = np.linalg.inv(np.stack(cam_poses))[:, :3, :4].reshape(n_cameras, -1)
            ks_flat = np.array(ks).reshape(n_cameras, -1)
            h_arr = np.array(hs, dtype=np_float_type).reshape(n_cameras, 1)
            w_arr = np.array(ws, dtype=np_float_type).reshape(n_cameras, 1)
            packed = np.empty((n_cameras, 23), dtype=np_float_type)
            packed[:, :12] = inv_poses
            packed[:, 12:21] = ks_flat
            packed[:, 21:22] = h_arr
            packed[:, 22:23] = w_arr
            return packed.ravel()

        def _pack_loop():
            """Per-camera loop packing (develop/main style)."""
            cameras = np.zeros(23 * n_cameras, dtype=np_float_type)
            for i in range(n_cameras):
                cameras[23 * i : 23 * (i + 1)] = np.concatenate(
                    [
                        np.linalg.inv(cam_poses[i])[:3, :4].reshape(-1),
                        ks[i].reshape(-1),
                        [hs[i]],
                        [ws[i]],
                    ]
                ).astype(np_float_type)
            return cameras

        t_vec = _time_fn(_pack_vectorized, repeats=repeats)
        t_loop = _time_fn(_pack_loop, repeats=repeats)
        results[str(n_cameras)] = {
            "vectorized_mean_us": t_vec["mean_us"],
            "loop_mean_us": t_loop["mean_us"],
            "speedup": t_loop["mean_us"] / t_vec["mean_us"] if t_vec["mean_us"] > 0 else 0,
        }
    return results


def bench_concatenate_vs_slice_fill(sizes: list[int], repeats: int) -> dict:
    """np.concatenate vs pre-allocated slice-fill at meaningful sizes.

    This simulates the buffer assembly patterns used in bisection loops
    (L/R vertex concatenation, final vertex assembly, etc.).
    """
    results = {}
    for n in sizes:
        a = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)
        b = np.random.default_rng(43).standard_normal((n, 3)).astype(np.float64)
        c = np.random.default_rng(44).standard_normal((n, 3)).astype(np.float64)
        d = np.random.default_rng(45).standard_normal((n, 3)).astype(np.float64)

        def _concat():
            return np.concatenate((a, b, c, d))

        def _slice():
            buf = np.empty((4 * n, 3), dtype=np.float64)
            buf[:n] = a
            buf[n : 2 * n] = b
            buf[2 * n : 3 * n] = c
            buf[3 * n :] = d
            return buf

        t_concat = _time_fn(_concat, repeats=repeats)
        t_slice = _time_fn(_slice, repeats=repeats)
        results[str(n)] = {
            "concat_mean_us": t_concat["mean_us"],
            "slice_fill_mean_us": t_slice["mean_us"],
            "speedup": t_concat["mean_us"] / t_slice["mean_us"] if t_slice["mean_us"] > 0 else 0,
        }
    return results


def bench_bisection_simulation(sizes: list[int], repeats: int) -> dict:
    """Simulate 15-iteration bisection loop (the hottest inner loop).

    Models what _construct_element_mesh does: for each iteration, call
    kernel_caller, update vertices, check tolerance.  Measures the
    cumulative overhead of the full loop at realistic sizes.
    """
    stub = _make_stub(enclosed=True)
    n_iters = 15
    has_out = _supports_out_param()
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)

        def _bisection():
            """Simulate bisection: 15 iterations of kernel_caller + update."""
            if has_out:
                out_buf = np.empty((n, 1), dtype=np.float32)
                for _ in range(n_iters):
                    stub.kernel_caller([_sdf_sphere], pts, out=out_buf)  # result used via out=
                    # simulate vertex update (midpoint)
                    pts[:] += np.random.default_rng(0).standard_normal((n, 3)) * 0.01
            else:
                for _ in range(n_iters):
                    stub.kernel_caller([_sdf_sphere], pts)  # timing the call itself
                    pts[:] += np.random.default_rng(0).standard_normal((n, 3)) * 0.01

        results[str(n)] = _time_fn(_bisection, repeats=repeats)
    return results


def bench_multi_kernel_bisection(sizes: list[int], repeats: int) -> dict:
    """Simulate bisection with 3 kernels (multi-element mesh).

    Tests the full multi-kernel → thread pool → bounds mask → result
    assembly pipeline under bisection-loop pressure.
    """
    stub = _make_stub(enclosed=True)
    kernels = [_sdf_sphere, _sdf_plane, _sdf_gyroid]
    n_iters = 15
    results = {}
    for n in sizes:
        pts = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)

        def _bisection():
            for _ in range(n_iters):
                stub.kernel_caller(kernels, pts)  # timing the call itself
                pts[:] += np.random.default_rng(0).standard_normal((n, 3)) * 0.01

        results[str(n)] = _time_fn(_bisection, repeats=repeats)
    return results


# ---------------------------------------------------------------------------
# Regression check: verify numerical equivalence
# ---------------------------------------------------------------------------
def regression_check() -> list[str]:
    """Verify that kernel_caller produces correct results.

    Returns a list of failure messages (empty = all pass).
    """
    failures = []
    stub_enc = _make_stub(enclosed=True)
    stub_open = _make_stub(enclosed=False)

    rng = np.random.default_rng(12345)
    pts = rng.standard_normal((5000, 3)).astype(np.float64)

    # -- Single kernel, enclosed -----------------------------------------
    sdf = stub_enc.kernel_caller([_sdf_sphere], pts)
    if sdf.shape != (5000, 1):
        failures.append(f"single/enclosed: expected shape (5000,1), got {sdf.shape}")
    if sdf.dtype != np.float32:
        failures.append(f"single/enclosed: expected dtype float32, got {sdf.dtype}")

    # out-of-bounds points should be clamped to 1.0
    oob = np.any((pts <= -10) | (pts >= 10), axis=1)
    if oob.any():
        oob_vals = sdf[oob, 0]
        if not np.all(oob_vals == 1.0):
            failures.append(f"single/enclosed: OOB points not masked to 1.0 (got {oob_vals[:5]})")

    # in-bounds points should match sphere SDF
    ib = ~oob
    if ib.any():
        expected = (np.linalg.norm(pts[ib], axis=1) - 5.0).astype(np.float32)
        actual = sdf[ib, 0]
        if not np.allclose(actual, expected, atol=1e-6):
            failures.append(f"single/enclosed: in-bounds values differ (max err={np.max(np.abs(actual - expected))})")

    # -- Single kernel, open ---------------------------------------------
    sdf_open = stub_open.kernel_caller([_sdf_sphere], pts)
    expected_all = (np.linalg.norm(pts, axis=1) - 5.0).astype(np.float32)
    if not np.allclose(sdf_open[:, 0], expected_all, atol=1e-6):
        failures.append("single/open: values differ from expected")

    # -- Multi kernel, enclosed ------------------------------------------
    sdf_multi = stub_enc.kernel_caller([_sdf_sphere, _sdf_plane, _sdf_gyroid], pts)
    if sdf_multi.shape != (5000, 3):
        failures.append(f"multi/enclosed: expected shape (5000,3), got {sdf_multi.shape}")

    # Column 0 = sphere, column 1 = plane, column 2 = gyroid
    exp_sphere = (np.linalg.norm(pts, axis=1) - 5.0).astype(np.float32)
    exp_plane = pts[:, 2].astype(np.float32)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    exp_gyroid = (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)).astype(np.float32)

    for col, name, expected in [
        (0, "sphere", exp_sphere),
        (1, "plane", exp_plane),
        (2, "gyroid", exp_gyroid),
    ]:
        actual = sdf_multi[:, col]
        if oob.any():
            # OOB should be 1.0
            if not np.all(actual[oob] == 1.0):
                failures.append(f"multi/enclosed/{name}: OOB not masked to 1.0")
        if ib.any() and not np.allclose(actual[ib], expected[ib], atol=1e-5):
            failures.append(f"multi/enclosed/{name}: in-bounds values differ")

    # -- out= parameter (only on optimized branch) ----------------------
    if _supports_out_param():
        out_buf = np.empty((5000, 1), dtype=np.float32)
        result = stub_enc.kernel_caller([_sdf_sphere], pts, out=out_buf)
        if result is not out_buf:
            failures.append("out= parameter: returned array is not the same object")
        if not np.allclose(result[:, 0], sdf[:, 0], atol=1e-6):
            failures.append("out= parameter: results differ from non-out path")

    # -- Empty input --------------------------------------------------
    empty = np.zeros((0, 3), dtype=np.float64)
    sdf_empty = stub_enc.kernel_caller([_sdf_sphere], empty)
    if sdf_empty.shape != (0, 1):
        failures.append(f"empty input: expected shape (0,1), got {sdf_empty.shape}")

    return failures


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------
def compare_results(file_a: str, file_b: str):
    """Print side-by-side comparison of two JSON result files."""
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    header = f"{'Benchmark':<40s}  {'Size':>8s}  {'A (µs)':>10s}  {'B (µs)':>10s}  {'Speedup':>8s}"
    print("=" * len(header))
    print(f"Comparing: A={file_a}  vs  B={file_b}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in sorted(set(list(a.keys()) + list(b.keys()))):
        if name not in a or name not in b:
            print(f"  {name}: only in {'A' if name in a else 'B'}")
            continue
        for sz in sorted(a[name].keys(), key=lambda s: int(s) if s.isdigit() else 0):
            if sz not in b[name]:
                continue
            sa = a[name][sz]
            sb = b[name][sz]
            # Handle both simple (mean_us) and comparison (speedup) formats
            if "mean_us" in sa and "mean_us" in sb:
                ta = sa["mean_us"]
                tb = sb["mean_us"]
                speedup = ta / tb if tb > 0 else float("inf")
                marker = "✓" if speedup > 1.05 else ("≈" if speedup > 0.95 else "✗")
                print(f"  {name:<38s}  {sz:>8s}  {ta:>10.1f}  {tb:>10.1f}  {speedup:>7.2f}x {marker}")
            elif "speedup" in sa and "speedup" in sb:
                print(f"  {name:<38s}  {sz:>8s}  (A speedup={sa['speedup']:.2f}x  B speedup={sb['speedup']:.2f}x)")

    print("=" * len(header))
    print("Speedup > 1.0 means B is faster than A")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_BENCHMARKS: dict = {
    "kernel_single_enclosed": bench_kernel_caller_single_enclosed,
    "kernel_single_open": bench_kernel_caller_single_open,
    "kernel_multi_3k": bench_kernel_caller_multi_3k,
    "kernel_out_reuse": bench_kernel_caller_out_reuse,
    "bounds_mask": bench_bounds_mask,
    "camera_packing": bench_camera_packing,
    "concat_vs_slice_fill": bench_concatenate_vs_slice_fill,
    "bisection_15iter_single": bench_bisection_simulation,
    "bisection_15iter_multi_3k": bench_multi_kernel_bisection,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark for OcMesher Python orchestration layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example: compare branches
              git checkout develop
              python -m benchmarks.bench_e2e --json develop.json
              git checkout copilot/refactor-algorithms-for-speed
              python -m benchmarks.bench_e2e --json optimized.json
              python -m benchmarks.bench_e2e --compare develop.json optimized.json
        """),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1_000, 10_000, 100_000, 1_000_000],
        help="Point counts to benchmark (default: 1K 10K 100K 1M)",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Timing repeats per measurement")
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="Compare two JSON result files")
    parser.add_argument("--regression", action="store_true", help="Run regression checks only")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # -- Regression check first ------------------------------------------
    print("=" * 70)
    print("REGRESSION CHECK")
    print("=" * 70)
    failures = regression_check()
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print(f"\n{len(failures)} regression failure(s)")
        if args.regression:
            sys.exit(1)
    else:
        print("  All regression checks passed ✓")
    print()

    if args.regression:
        return

    # -- Feature detection -----------------------------------------------
    print("=" * 70)
    print("OcMesher End-to-End Benchmark")
    print("=" * 70)
    print("Branch features:")
    print(f"  __slots__:             {_HAS_SLOTS}")
    print(f"  out= parameter:        {_supports_out_param()}")
    print(f"  _out_of_bounds_mask:   {hasattr(OcMesher, '_out_of_bounds_mask')}")
    print(f"  _SDF_BATCH_SIZE:       {_SDF_BATCH_SIZE:,}")
    print(f"  Sizes:                 {args.sizes}")
    print(f"  Repeats:               {args.repeats}")
    print()

    # -- Run benchmarks --------------------------------------------------
    all_results: dict[str, dict] = {}
    for name, fn in _BENCHMARKS.items():
        print(f"--- {name} ---")
        results = fn(args.sizes, args.repeats)
        all_results[name] = results
        for sz, stats in results.items():
            if "mean_us" in stats:
                print(
                    f"  N={sz:>8s}  mean={stats['mean_us']:>12.1f} µs  "
                    f"median={stats['median_us']:>12.1f} µs  "
                    f"stdev={stats['stdev_us']:>10.1f} µs"
                )
            elif "speedup" in stats:
                items = "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items())
                print(f"  N={sz:>8s}  {items}")
        print()

    # -- Save results ----------------------------------------------------
    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.json}")

    # -- Summary ---------------------------------------------------------
    print("=" * 70)
    print("To compare branches:")
    print("  1. Run on develop/main:  python -m benchmarks.bench_e2e --json baseline.json")
    print("  2. Run on this branch:   python -m benchmarks.bench_e2e --json optimized.json")
    print("  3. Compare:              python -m benchmarks.bench_e2e --compare baseline.json optimized.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
