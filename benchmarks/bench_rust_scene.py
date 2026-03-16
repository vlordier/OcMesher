"""Benchmark the Rust scene pilot entry points.

This benchmark compares the compiled native Rust scene path against the
feature-enabled ``tch`` scene path using the same primitive-spec list.

Usage::

    python -m benchmarks.bench_rust_scene
    python -m benchmarks.bench_rust_scene --runs 5 --warmup 1
    python -m benchmarks.bench_rust_scene --device mps --snapshot-json rust_scene.json
    python -m benchmarks.bench_rust_scene --compare cpu.json mps.json
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.result_utils import load_results_payload, write_snapshot_json


def _build_backend() -> Any:
    try:
        ocmesher_rust = importlib.import_module("ocmesher_rust")
    except ImportError as err:  # pragma: no cover - exercised via runtime invocation
        msg = (
            "ocmesher_rust is not installed. Build it with: "
            "uv run maturin develop --release --manifest-path ocmesher-rust/crates/ocmesher-py/Cargo.toml"
        )
        raise RuntimeError(msg) from err

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        msg = "compiled core.so not found; run `bash install.sh` first"
        raise RuntimeError(msg)

    pose = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 3],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    k = np.array(
        [
            [2000, 0, 640],
            [0, 2000, 360],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    bounds = np.array([-5.0, 5.0, -5.0, 5.0, -5.0, 5.0], dtype=np.float64)

    return ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [pose.ravel().tolist()],
            [k.ravel().tolist()],
            [720.0],
            [1280.0],
        ),
        bounds=bounds.tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )


def _scene_primitives() -> list[dict[str, object]]:
    return [
        {"type": "sphere", "radius": 1.0, "center": [0.0, 0.0, 0.0]},
        {"type": "plane", "offset": 0.0, "normal": [0.0, 0.0, 1.0]},
    ]


def _pick_device(backend: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    caps = backend.get_capabilities()
    if bool(caps.get("supports_mps")):
        return "mps"
    return "cpu"


def _time_call(fn: Any, *, warmup: int, runs: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()

    samples_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "runs": runs,
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def run_benchmark(*, runs: int, warmup: int, device: str) -> dict[str, Any]:
    backend = _build_backend()
    scene = _scene_primitives()
    tch_device = _pick_device(backend, device)

    native_result = _time_call(lambda: backend.extract_native_scene(scene), warmup=warmup, runs=runs)
    tch_result = _time_call(lambda: backend.extract_tch_scene(scene, device=tch_device), warmup=warmup, runs=runs)

    speedup = native_result["mean_ms"] / tch_result["mean_ms"] if tch_result["mean_ms"] else float("inf")
    return {
        "scene": scene,
        "native_scene": native_result,
        "tch_scene": {**tch_result, "device": tch_device},
        "speedup_native_over_tch": speedup,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "CANDIDATE"), default=None, help="compare two rust scene snapshot files")
    parser.add_argument("--runs", type=int, default=3, help="timed iterations per backend")
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations before timing")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="device for tch scene benchmarking",
    )
    parser.add_argument("--snapshot-json", type=str, default=None, help="write snapshot JSON to this path")
    args = parser.parse_args(argv)
    if args.compare is not None:
        return args
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    return args


def _format_metadata_label(metadata: dict[str, Any] | None, fallback: str) -> str:
    if metadata is None:
        return fallback
    git_commit = metadata.get("git_commit")
    if isinstance(git_commit, str) and git_commit:
        return git_commit[:7]
    return fallback


def _compare_snapshots(base_path: str, candidate_path: str) -> list[str]:
    base_results, base_meta = load_results_payload(base_path)
    candidate_results, candidate_meta = load_results_payload(candidate_path)

    base_label = _format_metadata_label(base_meta, "base")
    candidate_label = _format_metadata_label(candidate_meta, "cand")
    base_native = float(base_results["native_scene"]["mean_ms"])
    base_tch = float(base_results["tch_scene"]["mean_ms"])
    candidate_native = float(candidate_results["native_scene"]["mean_ms"])
    candidate_tch = float(candidate_results["tch_scene"]["mean_ms"])

    native_speedup = base_native / candidate_native if candidate_native else float("inf")
    tch_speedup = base_tch / candidate_tch if candidate_tch else float("inf")

    return [
        "Rust scene benchmark comparison",
        f"base: {base_path} ({base_label})",
        f"candidate: {candidate_path} ({candidate_label})",
        f"native_scene mean: {base_native:.3f} ms -> {candidate_native:.3f} ms ({native_speedup:.3f}x)",
        f"tch_scene mean: {base_tch:.3f} ms -> {candidate_tch:.3f} ms ({tch_speedup:.3f}x)",
        "speedup > 1.0 means the candidate snapshot is faster",
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.compare is not None:
        for line in _compare_snapshots(args.compare[0], args.compare[1]):
            print(line)
        return 0

    results = run_benchmark(runs=args.runs, warmup=args.warmup, device=args.device)

    print("Rust scene benchmark")
    print(f"native_scene mean: {results['native_scene']['mean_ms']:.3f} ms")
    print(
        f"tch_scene mean ({results['tch_scene']['device']}): "
        f"{results['tch_scene']['mean_ms']:.3f} ms"
    )
    print(f"native/tch speedup: {results['speedup_native_over_tch']:.3f}x")

    if args.snapshot_json:
        write_snapshot_json(args.snapshot_json, results, label="rust-scene")
        print(f"snapshot written to {args.snapshot_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())