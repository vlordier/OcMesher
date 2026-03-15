#!/usr/bin/env python3
# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""OcMesher build-profile benchmark harness.

Runs a deterministic meshing workload and records wall-clock time for every
phase exposed by the Timer context managers inside OcMesher.__call__.

Usage
-----
    # single run with the current build (outputs to stdout + JSON)
    python benchmark/benchmark.py

    # save results alongside the profile label
    python benchmark/benchmark.py --profile native --runs 3 --out results.json

    # use a more complex scene
    python benchmark/benchmark.py --scene sphere --pixels-per-cube 8

Environment variables
---------------------
    OMP_NUM_THREADS   controls the number of OpenMP threads used by the C++
                      core (default: all logical CPUs).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:  # psutil is optional - memory reporting is disabled
    _psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

# Make sure the package root is importable even when run from benchmark/
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ocmesher import OcMesher  # noqa: E402  (after sys.path insert)

# ---------------------------------------------------------------------------
# Built-in SDF scenes
# ---------------------------------------------------------------------------

def _scene_sphere(XYZ: np.ndarray) -> np.ndarray:
    """Unit sphere centred at origin."""
    return np.sqrt((XYZ ** 2).sum(axis=-1)) - 1.0


def _scene_torus(XYZ: np.ndarray) -> np.ndarray:
    """Torus with major radius 1.5 and minor radius 0.4."""
    major_r, minor_r = 1.5, 0.4
    q = np.sqrt(XYZ[:, 0] ** 2 + XYZ[:, 1] ** 2) - major_r
    return np.sqrt(q ** 2 + XYZ[:, 2] ** 2) - minor_r


def _scene_gyroid(XYZ: np.ndarray) -> np.ndarray:
    """Gyroid minimal surface (periodic, scale approx 2*pi)."""
    s = 2.0
    x, y, z = XYZ[:, 0] / s, XYZ[:, 1] / s, XYZ[:, 2] / s
    return (
        np.sin(x) * np.cos(y)
        + np.sin(y) * np.cos(z)
        + np.sin(z) * np.cos(x)
    )


def _scene_heightmap(XYZ: np.ndarray) -> np.ndarray:
    """Deterministic analytic heightmap (avoids vnoise dependency)."""
    scale = 1.5
    h = (
        0.5 * np.sin(XYZ[:, 0] / scale) * np.cos(XYZ[:, 1] / scale)
        + 0.25 * np.sin(2 * XYZ[:, 0] / scale) * np.cos(2 * XYZ[:, 1] / scale)
    )
    return XYZ[:, 2] - h


_SCENES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sphere": _scene_sphere,
    "torus": _scene_torus,
    "gyroid": _scene_gyroid,
    "heightmap": _scene_heightmap,
}


# ---------------------------------------------------------------------------
# Timing shim
# ---------------------------------------------------------------------------

class _TimingCapture:
    """Monkey-patches the Timer reference inside ocmesher.core to capture durations."""

    def __init__(self) -> None:
        self._records: dict[str, float] = {}

    def install(self) -> None:
        """Patch ocmesher.core.Timer with a capturing implementation."""
        # ocmesher.core imports Timer via `from .utils.timer import Timer`, so
        # patching ocmesher.utils.timer.Timer is too late.  We must replace the
        # name in the already-imported ocmesher.core module namespace.
        import ocmesher.core as _core_mod  # noqa: PLC0415

        records = self._records

        class _CapturingTimer:
            def __init__(self, desc: str, disable_timer: bool = False) -> None:  # noqa: FBT001,FBT002
                self._desc = desc
                self._disabled = disable_timer

            def __enter__(self) -> _CapturingTimer:  # noqa: PYI034
                if not self._disabled:
                    self._t0 = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, tb) -> None:
                if not self._disabled and exc_type is None:
                    elapsed = time.perf_counter() - self._t0
                    records[self._desc] = records.get(self._desc, 0.0) + elapsed
                    if _HAS_PSUTIL:
                        mem_gb = (
                            _psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
                        )
                        mem_str = f"  ({mem_gb:.2f} GB RSS)"
                    else:
                        mem_str = ""
                    print(f"[{self._desc}] finished in {elapsed:.3f}s{mem_str}")

        _core_mod.Timer = _CapturingTimer  # type: ignore[assignment]

    @property
    def records(self) -> dict[str, float]:
        """Return a copy of the captured timing records."""
        return dict(self._records)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _make_cameras() -> tuple:
    """Return a single camera looking down at the origin from height 3."""
    cam_pose = np.array([
        [1, 0,  0, 0],
        [0, 0,  1, 0],
        [0, -1, 0, 3],
        [0, 0,  0, 1],
    ], dtype=np.float64)
    k_mat = np.array([
        [2000, 0, 640],
        [0, 2000, 360],
        [0, 0, 1],
    ], dtype=np.float64)
    return ([cam_pose], [k_mat], [720], [1280])


def _bounds_for_scene(scene: str) -> tuple[float, ...]:
    """Return axis-aligned bounding box (xmin,xmax,ymin,ymax,zmin,zmax) for a scene."""
    if scene == "heightmap":
        return (-5.0, 5.0, -5.0, 5.0, -2.0, 2.0)
    if scene == "gyroid":
        return (
            -math.pi * 2, math.pi * 2,
            -math.pi * 2, math.pi * 2,
            -math.pi * 2, math.pi * 2,
        )
    return (-3.0, 3.0, -3.0, 3.0, -3.0, 3.0)


def run_once(
    scene: str,
    pixels_per_cube: int,
    omp_threads: int | None,
) -> tuple[dict[str, float], float]:
    """Build the mesh once and return per-phase timings + total wall time."""
    if omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    capture = _TimingCapture()
    capture.install()

    cameras = _make_cameras()
    bounds = _bounds_for_scene(scene)
    kernel = _SCENES[scene]

    mesher = OcMesher(cameras, bounds, pixels_per_cube=pixels_per_cube)

    t_start = time.perf_counter()
    _meshes, _tags = mesher([kernel])
    total = time.perf_counter() - t_start

    return capture.records, total


def run_benchmark(
    scene: str = "sphere",
    pixels_per_cube: int = 8,
    runs: int = 1,
    profile: str = "native",
    omp_threads: int | None = None,
) -> dict:
    """Run the benchmark *runs* times and aggregate min/mean/max statistics."""
    all_phases: list[dict[str, float]] = []
    all_totals: list[float] = []

    print(f"\n{'─'*60}")
    print(f"Profile  : {profile}")
    print(f"Scene    : {scene}")
    print(f"pix/cube : {pixels_per_cube}")
    print(f"Threads  : {omp_threads or os.cpu_count()} (OMP_NUM_THREADS)")
    print(f"{'─'*60}")

    for i in range(runs):
        print(f"\n-- Run {i + 1}/{runs} --")
        phases, total = run_once(scene, pixels_per_cube, omp_threads)
        all_phases.append(phases)
        all_totals.append(total)
        print(f"  total wall time: {total:.3f}s")

    # Compute per-phase and overall min/mean/max across all runs.
    phase_keys = sorted({k for p in all_phases for k in p})
    aggregated: dict[str, dict[str, float]] = {}
    for key in phase_keys:
        vals = [p.get(key, 0.0) for p in all_phases]
        aggregated[key] = {
            "min": min(vals),
            "mean": sum(vals) / len(vals),
            "max": max(vals),
        }

    total_stats = {
        "min": min(all_totals),
        "mean": sum(all_totals) / len(all_totals),
        "max": max(all_totals),
    }

    print(f"\n{'─'*60}")
    print(f"Summary  (profile={profile}, n={runs})")
    print(f"{'─'*60}")
    print(
        f"  total wall  min={total_stats['min']:.3f}s "
        f"mean={total_stats['mean']:.3f}s "
        f"max={total_stats['max']:.3f}s",
    )
    for key, stats in aggregated.items():
        print(
            f"  [{key}]  min={stats['min']:.3f}s "
            f"mean={stats['mean']:.3f}s "
            f"max={stats['max']:.3f}s",
        )

    return {
        "profile": profile,
        "scene": scene,
        "pixels_per_cube": pixels_per_cube,
        "omp_threads": omp_threads or os.cpu_count(),
        "runs": runs,
        "total_wall": total_stats,
        "phases": aggregated,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OcMesher build-profile benchmark harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--profile",
        default=os.environ.get("PROFILE", "native"),
        help="Build profile label (informational - does not rebuild the lib)",
    )
    p.add_argument(
        "--scene",
        default="sphere",
        choices=list(_SCENES),
        help="SDF scene to mesh",
    )
    p.add_argument(
        "--pixels-per-cube",
        type=int,
        default=8,
        dest="pixels_per_cube",
        help="OcMesher pixels_per_cube parameter (higher = finer mesh)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs (statistics are aggregated)",
    )
    p.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        dest="omp_threads",
        help="Override OMP_NUM_THREADS for the C++ core",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Write JSON results to this file (appends to an array if it exists)",
    )
    return p.parse_args()


def _append_json(path: str, result: dict) -> None:
    """Append *result* to the JSON array stored in *path*."""
    existing: list[dict] = []
    p = Path(path)
    if p.exists():
        with p.open() as fh:
            existing = json.load(fh)
    existing.append(result)
    with p.open("w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"\nResults appended to {path}")


def main() -> None:
    """Entry point for the benchmark harness."""
    args = _parse_args()
    result = run_benchmark(
        scene=args.scene,
        pixels_per_cube=args.pixels_per_cube,
        runs=args.runs,
        profile=args.profile,
        omp_threads=args.omp_threads,
    )
    if args.out:
        _append_json(args.out, result)
    else:
        print("\nJSON result:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

