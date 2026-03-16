"""A/B benchmark: develop-branch SDF overhead vs rust-review-from-develop.

Measures the pure Python orchestration + SDF evaluation cost that differs
between the two branches — without requiring the compiled C++ ``core.so``
library to be present.

What is measured
----------------
*develop baseline*
    Pure-Python SDF evaluation as it happens on ``develop``:
    • ``evaluate_batch`` preferred over ``__call__`` (mirrors ``rust_backend.build_batched_sdf_kernels``)
    • numpy array construction per query batch
    • dict output extraction (``{"sdf": ...}``)

*rust branch optimised path*
    The same sequence as the Rust hot-path in
    ``eval_kernel_py_once`` / ``eval_sdf_full``, exercised from Python
    to measure the savings:
    • DLPack f32 capsule input (no numpy alloc)
    • ``evaluate_batch_torch`` preferred
    • ``to_dlpack()`` + raw pointer read on output (no numpy object)

Because the full Rust pipeline requires the C++ shared library, this script
benchmarks the **SDF evaluation layer only** in-process using mock kernels, so
it runs on any developer machine without building OcMesher.

Usage::

    # Quick run (default settings)
    python -m benchmarks.bench_ab_develop_vs_rust

    # Save results to JSON
    python -m benchmarks.bench_ab_develop_vs_rust --json results/ab.json

    # Custom sizes and repeats
    python -m benchmarks.bench_ab_develop_vs_rust --sizes 10000 100000 500000 --repeats 10
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_OK = False

# ---------------------------------------------------------------------------
# Mock SDF kernels (no C++ dependency)
# ---------------------------------------------------------------------------

class _SphereCPU:
    """Pure numpy sphere SDF — models develop-branch evaluate_batch kernels."""
    def evaluate_batch(self, xyz: np.ndarray) -> dict[str, np.ndarray]:
        return {"sdf": (np.linalg.norm(xyz, axis=1) - 5.0).astype(np.float32)}

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return (np.linalg.norm(xyz, axis=1) - 5.0).astype(np.float32)


class _SphereEvalBatchTorch:
    """Torch-aware sphere SDF — models kernels on rust branch with evaluate_batch_torch."""
    def evaluate_batch(self, xyz: np.ndarray) -> dict[str, np.ndarray]:
        return {"sdf": (np.linalg.norm(xyz, axis=1) - 5.0).astype(np.float32)}

    def evaluate_batch_torch(self, xyz: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        t = torch.as_tensor(xyz) if not isinstance(xyz, torch.Tensor) else xyz
        return (torch.linalg.norm(t.float(), dim=1) - 5.0)

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return (np.linalg.norm(xyz, axis=1) - 5.0).astype(np.float32)


class _GyroidEvalBatchTorch:
    """Trigonometric gyroid SDF — higher compute cost, models complex Infinigen kernels."""
    def evaluate_batch(self, xyz: np.ndarray) -> np.ndarray:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        return (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)).astype(np.float32)

    def evaluate_batch_torch(self, xyz: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        t = torch.as_tensor(xyz).float() if not isinstance(xyz, torch.Tensor) else xyz.float()
        x, y, z = t[:, 0], t[:, 1], t[:, 2]
        return torch.sin(x) * torch.cos(y) + torch.sin(y) * torch.cos(z) + torch.sin(z) * torch.cos(x)

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return self.evaluate_batch(xyz)


# ---------------------------------------------------------------------------
# Develop-baseline path (mirrors build_batched_sdf_kernels on develop)
# ---------------------------------------------------------------------------

def _extract_sdf_np(output: Any) -> np.ndarray:
    if isinstance(output, dict):
        return np.asarray(output.get("sdf", output.get("SDF")))
    return np.asarray(output)


def develop_eval_batch(kernel: Any, xyz: np.ndarray) -> np.ndarray:
    """develop branch: prefer evaluate_batch, fall back to __call__."""
    eval_batch = getattr(kernel, "evaluate_batch", None)
    if callable(eval_batch):
        return _extract_sdf_np(eval_batch(xyz))
    return _extract_sdf_np(kernel(xyz))


def develop_eval_many_kernels(
    kernels: list[Any],
    n_pts: int,
    batch_size: int,
) -> np.ndarray:
    """Replicate develop branch multi-kernel evaluation."""
    n_k = len(kernels)
    result = np.empty((n_pts, n_k), dtype=np.float32)
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-10, 10, size=(n_pts, 3)).astype(np.float64)
    for chunk_start in range(0, n_pts, batch_size):
        chunk = xyz[chunk_start : chunk_start + batch_size]
        for k_idx, kernel in enumerate(kernels):
            vals = develop_eval_batch(kernel, chunk)
            result[chunk_start : chunk_start + len(chunk), k_idx] = vals
    return result


# ---------------------------------------------------------------------------
# Rust-branch optimised path (mirrors eval_sdf_full with torch + DLPack)
# ---------------------------------------------------------------------------

def _build_dlpack_f32_tensor(xyz: np.ndarray) -> "torch.Tensor":
    """Build f32 CPU torch tensor from numpy via DLPack (no intermediate copy)."""
    # torch.from_numpy gives zero-copy view; .to(torch.float32) is the Rust
    # equivalent of the DLPack f32 capsule path when already in torch.
    return torch.from_numpy(xyz).to(torch.float32)


def _extract_sdf_via_dlpack(t: "torch.Tensor") -> np.ndarray:
    """Extract f32 values from a CPU torch tensor via DLPack (no intermediate numpy alloc).

    On the Rust side this read happens via the raw DLManagedTensor pointer
    (zero Python objects).  Here we use torch.utils.dlpack.to_dlpack +
    from_dlpack to simulate the same path from Python.
    """
    if t.device.type != "cpu":
        t = t.to("cpu")
    t = t.contiguous()
    # Round-trip through DLPack: to_dlpack → from_dlpack gives a new tensor
    # that shares the same data buffer — models the raw-pointer read in Rust.
    capsule = torch.utils.dlpack.to_dlpack(t)
    t2 = torch.utils.dlpack.from_dlpack(capsule)
    return t2.numpy().astype(np.float32, copy=False)


def rust_eval_many_kernels(
    kernels: list[Any],
    n_pts: int,
    batch_size: int,
) -> np.ndarray:
    """Rust branch: evaluate_batch_torch + DLPack input + DLPack output."""
    n_k = len(kernels)
    result = np.empty((n_pts, n_k), dtype=np.float32)
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-10, 10, size=(n_pts, 3)).astype(np.float64)

    for chunk_start in range(0, n_pts, batch_size):
        chunk_np = xyz[chunk_start : chunk_start + batch_size]
        # DLPack f32 input — no numpy array construction in the kernel
        xyz_t = _build_dlpack_f32_tensor(chunk_np)
        for k_idx, kernel in enumerate(kernels):
            eval_fn = getattr(kernel, "evaluate_batch_torch", None)
            if callable(eval_fn):
                raw = eval_fn(xyz_t)
                if isinstance(raw, dict):
                    raw = raw.get("sdf", raw.get("SDF", next(iter(raw.values()))))
                vals = _extract_sdf_via_dlpack(
                    raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
                )
            else:
                vals = develop_eval_batch(kernel, chunk_np)
            result[chunk_start : chunk_start + len(chunk_np), k_idx] = vals
    return result


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _timeit(fn, *, warmup: int = 2, repeats: int = 7) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e6)  # ms
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "samples": len(times),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_row(label: str, d: dict[str, float]) -> str:
    return (
        f"  {label:<42s}  "
        f"mean={d['mean_ms']:8.2f}ms  "
        f"median={d['median_ms']:8.2f}ms  "
        f"min={d['min_ms']:8.2f}ms"
    )


def _speedup(baseline: dict, optimised: dict) -> float:
    b = baseline["median_ms"]
    o = optimised["median_ms"]
    return b / o if o > 0 else float("inf")


def run_ab(
    sizes: list[int],
    repeats: int,
    warmup: int,
    kernels_per_call: int,
    batch_size: int,
) -> dict[str, Any]:
    kernel_sets: dict[str, list[Any]] = {
        "sphere (numpy only, no torch)": [_SphereCPU() for _ in range(kernels_per_call)],
        "sphere (evaluate_batch_torch)": [_SphereEvalBatchTorch() for _ in range(kernels_per_call)],
        "gyroid (evaluate_batch_torch)": [_GyroidEvalBatchTorch() for _ in range(kernels_per_call)],
    }

    all_results: dict[str, Any] = {
        "torch_available": _TORCH_OK,
        "kernels_per_call": kernels_per_call,
        "batch_size": batch_size,
        "repeats": repeats,
        "warmup": warmup,
        "results": {},
    }

    for sdf_label, kernels in kernel_sets.items():
        all_results["results"][sdf_label] = {}
        for n_pts in sizes:
            dev_t = _timeit(
                lambda k=kernels, n=n_pts, b=batch_size: develop_eval_many_kernels(k, n, b),
                warmup=warmup,
                repeats=repeats,
            )
            row: dict[str, Any] = {"n_pts": n_pts, "develop": dev_t}

            if _TORCH_OK:
                rust_t = _timeit(
                    lambda k=kernels, n=n_pts, b=batch_size: rust_eval_many_kernels(k, n, b),
                    warmup=warmup,
                    repeats=repeats,
                )
                row["rust"] = rust_t
                row["speedup_x"] = round(_speedup(dev_t, rust_t), 2)

            all_results["results"][sdf_label][n_pts] = row

    return all_results


def _print_report(results: dict[str, Any]) -> None:
    torch_ok = results["torch_available"]
    kpc = results["kernels_per_call"]
    bs = results["batch_size"]
    print()
    print("=" * 78)
    print(" OcMesher A/B: develop SDF overhead vs rust-review-from-develop")
    print("=" * 78)
    print(f"  kernels/call={kpc}  batch_size={bs}  torch={'yes' if torch_ok else 'NO (install torch to enable rust path)'}")
    print()

    for sdf_label, by_size in results["results"].items():
        print(f"  [{sdf_label}]")
        for n_pts, row in sorted(by_size.items(), key=lambda x: int(x[0])):
            print(f"\n    n_pts={n_pts:,}")
            print(_fmt_row("develop (numpy evaluate_batch)", row["develop"]))
            if "rust" in row:
                print(_fmt_row("rust   (DLPack f32 + eval_batch_torch)", row["rust"]))
                sp = row["speedup_x"]
                arrow = "▲" if sp >= 1 else "▼"
                print(f"    {arrow} speedup: {sp:.2f}x  (rust / develop median ratio)")
        print()
    print("=" * 78)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B benchmark: develop SDF overhead vs rust branch optimised path.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m benchmarks.bench_ab_develop_vs_rust
              python -m benchmarks.bench_ab_develop_vs_rust --sizes 5000 50000 --repeats 10
              python -m benchmarks.bench_ab_develop_vs_rust --json results/ab.json
        """),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1_000, 10_000, 100_000],
        metavar="N",
        help="Number of SDF query points to test (default: 1000 10000 100000)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Number of timed repetitions per configuration (default: 7)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of un-timed warmup runs (default: 2)",
    )
    parser.add_argument(
        "--kernels",
        type=int,
        default=3,
        metavar="K",
        help="Number of SDF kernels per meshing call (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32_768,
        metavar="B",
        help="SDF evaluation batch size (default: 32768)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write full results to JSON file",
    )
    args = parser.parse_args()

    results = run_ab(
        sizes=args.sizes,
        repeats=args.repeats,
        warmup=args.warmup,
        kernels_per_call=args.kernels,
        batch_size=args.batch_size,
    )

    _print_report(results)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
