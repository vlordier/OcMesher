# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Benchmark: original Python + C++ backend vs. PyTorch backend.

Run from the repository root::

    python -m benchmarks.run_benchmark          # quick (small grid)
    python -m benchmarks.run_benchmark --full   # full comparison
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Perlin-noise SDF used for both backends
# ---------------------------------------------------------------------------
try:
    import vnoise

    _noise = vnoise.Noise()

    def sdf_terrain(xyz: np.ndarray) -> np.ndarray:
        """Signed-distance function for a Perlin-noise height field."""
        scale = 2
        h = _noise.noise2(xyz[:, 0] / scale, xyz[:, 1] / scale, grid_mode=False, octaves=4)
        return xyz[:, 2] - h

except ImportError:

    def sdf_terrain(xyz: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        """Fallback SDF - flat plane at z=0."""
        return xyz[:, 2].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cameras():
    cam_poses = [
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 3],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
    ]
    ks = [
        np.array(
            [
                [2000, 0, 640],
                [0, 2000, 360],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
    ]
    hs = [720]
    ws = [1280]
    return (cam_poses, ks, hs, ws)


def _make_bounds():
    return (-10, 10, -10, 10, -2, 2)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _bench_original(cameras, bounds, pixels_per_cube: int, n_runs: int = 1):
    """Benchmark the original Python + C++ OcMesher."""
    try:
        from ocmesher.core import OcMesher
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not load original OcMesher: {exc}"}

    times: list[float] = []
    mesh_info: dict[str, object] = {}
    for i in range(n_runs):
        t0 = time.perf_counter()
        mesher = OcMesher(cameras, bounds, pixels_per_cube=pixels_per_cube)
        meshes, _tags = mesher([sdf_terrain])
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if i == 0:
            mesh_info = {
                "vertices": int(meshes[0].vertices.shape[0]),
                "faces": int(meshes[0].faces.shape[0]),
            }

    return {
        "backend": "python_cpp",
        "times_s": times,
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        **mesh_info,
    }


def _bench_torch(cameras, bounds, pixels_per_cube: int, n_runs: int = 1, device: str | None = None):
    """Benchmark the PyTorch TorchOcMesher."""
    try:
        import torch  # noqa: F401

        from ocmesher.torch_core import TorchOcMesher
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not load TorchOcMesher: {exc}"}

    times: list[float] = []
    mesh_info: dict[str, object] = {}
    for i in range(n_runs):
        t0 = time.perf_counter()
        mesher = TorchOcMesher(cameras, bounds, pixels_per_cube=pixels_per_cube, device=device)
        meshes, _tags = mesher([sdf_terrain])
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if i == 0:
            mesh_info = {
                "vertices": int(meshes[0].vertices.shape[0]),
                "faces": int(meshes[0].faces.shape[0]),
            }

    return {
        "backend": f"pytorch_{device or 'auto'}",
        "times_s": times,
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        **mesh_info,
    }


# ---------------------------------------------------------------------------
# Sub-operation micro-benchmarks
# ---------------------------------------------------------------------------


def _micro_sdf_eval(cameras, bounds, n_points: int = 500_000, n_runs: int = 3):
    """Compare SDF evaluation throughput (numpy vs torch tensors)."""
    results: dict[str, object] = {"n_points": n_points}

    # Random query points
    rng = np.random.default_rng(42)
    pts = rng.uniform(
        [bounds[0], bounds[2], bounds[4]],
        [bounds[1], bounds[3], bounds[5]],
        size=(n_points, 3),
    )

    # Numpy path
    times_np: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = sdf_terrain(pts.copy())
        times_np.append(time.perf_counter() - t0)
    results["numpy_mean_s"] = float(np.mean(times_np))

    # Torch path (CPU)
    try:
        import torch

        from ocmesher.torch_core import TorchOcMesher

        mesher = TorchOcMesher(cameras, bounds, device="cpu")
        pts_t = torch.from_numpy(pts).to(dtype=torch.float64, device=mesher.device)
        times_torch: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = mesher._evaluate_sdf([sdf_terrain], pts_t)
            times_torch.append(time.perf_counter() - t0)
        results["torch_cpu_mean_s"] = float(np.mean(times_torch))

        if torch.cuda.is_available():
            mesher_gpu = TorchOcMesher(cameras, bounds, device="cuda")
            pts_g = pts_t.to(mesher_gpu.device)
            times_gpu: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = mesher_gpu._evaluate_sdf([sdf_terrain], pts_g)
                torch.cuda.synchronize()
                times_gpu.append(time.perf_counter() - t0)
            results["torch_gpu_mean_s"] = float(np.mean(times_gpu))
    except Exception as exc:  # noqa: BLE001
        results["torch_error"] = str(exc)

    return results


def _micro_projection(cameras, bounds, n_cubes: int = 100_000, n_runs: int = 5):
    """Compare camera-projection throughput."""
    results: dict[str, object] = {"n_cubes": n_cubes}

    try:
        import torch

        from ocmesher.torch_core import TorchOcMesher

        mesher = TorchOcMesher(cameras, bounds, device="cpu")
        rng = np.random.default_rng(42)
        coords = torch.from_numpy(rng.integers(0, 64, size=(n_cubes, 3))).to(mesher.device)
        levels = torch.full((n_cubes,), 6, dtype=torch.int64, device=mesher.device)
        positions = mesher._cube_centers(coords, levels)

        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = mesher._projected_sizes(positions, levels)
            times.append(time.perf_counter() - t0)
        results["torch_cpu_mean_s"] = float(np.mean(times))

        if torch.cuda.is_available():
            mesher_gpu = TorchOcMesher(cameras, bounds, device="cuda")
            coords_g = coords.to(mesher_gpu.device)
            levels_g = levels.to(mesher_gpu.device)
            positions_g = mesher_gpu._cube_centers(coords_g, levels_g)
            times_gpu: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = mesher_gpu._projected_sizes(positions_g, levels_g)
                torch.cuda.synchronize()
                times_gpu.append(time.perf_counter() - t0)
            results["torch_gpu_mean_s"] = float(np.mean(times_gpu))
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run benchmarks and print results."""
    parser = argparse.ArgumentParser(description="OcMesher benchmark: Python+C++ vs PyTorch")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (slower, higher resolution)")
    parser.add_argument("--runs", type=int, default=1, help="Number of end-to-end runs")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    pixels_per_cube = 8 if args.full else 16
    cameras = _make_cameras()
    bounds = _make_bounds()

    print("=" * 70)
    print("OcMesher Benchmark: Python + C++ vs PyTorch")
    print("=" * 70)

    try:
        import torch

        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device     : {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    print(f"pixels_per_cube : {pixels_per_cube}")
    print(f"runs            : {args.runs}")
    print()

    results: dict[str, object] = {}

    # End-to-end benchmarks ------------------------------------------------
    print("-" * 70)
    print("End-to-end: Original Python + C++")
    print("-" * 70)
    r_orig = _bench_original(cameras, bounds, pixels_per_cube, n_runs=args.runs)
    if "error" in r_orig:
        print(f"  SKIPPED: {r_orig['error']}")
    else:
        print(f"  Mean time : {r_orig['mean_s']:.3f} s  (std {r_orig['std_s']:.3f})")
        print(f"  Vertices  : {r_orig.get('vertices', '?')}")
        print(f"  Faces     : {r_orig.get('faces', '?')}")
    results["original"] = r_orig
    print()

    print("-" * 70)
    print("End-to-end: PyTorch (CPU)")
    print("-" * 70)
    r_torch_cpu = _bench_torch(cameras, bounds, pixels_per_cube, n_runs=args.runs, device="cpu")
    if "error" in r_torch_cpu:
        print(f"  SKIPPED: {r_torch_cpu['error']}")
    else:
        print(f"  Mean time : {r_torch_cpu['mean_s']:.3f} s  (std {r_torch_cpu['std_s']:.3f})")
        print(f"  Vertices  : {r_torch_cpu.get('vertices', '?')}")
        print(f"  Faces     : {r_torch_cpu.get('faces', '?')}")
    results["torch_cpu"] = r_torch_cpu
    print()

    try:
        import torch

        if torch.cuda.is_available():
            print("-" * 70)
            print("End-to-end: PyTorch (CUDA)")
            print("-" * 70)
            r_torch_gpu = _bench_torch(cameras, bounds, pixels_per_cube, n_runs=args.runs, device="cuda")
            if "error" in r_torch_gpu:
                print(f"  SKIPPED: {r_torch_gpu['error']}")
            else:
                print(f"  Mean time : {r_torch_gpu['mean_s']:.3f} s  (std {r_torch_gpu['std_s']:.3f})")
                print(f"  Vertices  : {r_torch_gpu.get('vertices', '?')}")
                print(f"  Faces     : {r_torch_gpu.get('faces', '?')}")
            results["torch_gpu"] = r_torch_gpu
            print()
    except ImportError:
        pass

    # Micro-benchmarks -----------------------------------------------------
    print("-" * 70)
    print("Micro-benchmark: SDF evaluation")
    print("-" * 70)
    r_sdf = _micro_sdf_eval(cameras, bounds)
    for k, v in r_sdf.items():
        print(f"  {k}: {v}")
    results["micro_sdf"] = r_sdf
    print()

    print("-" * 70)
    print("Micro-benchmark: Camera projection")
    print("-" * 70)
    r_proj = _micro_projection(cameras, bounds)
    for k, v in r_proj.items():
        print(f"  {k}: {v}")
    results["micro_projection"] = r_proj
    print()

    # Speedup summary ------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if "error" not in r_orig and "error" not in r_torch_cpu:
        speedup = r_orig["mean_s"] / max(r_torch_cpu["mean_s"], 1e-6)
        print(f"  PyTorch CPU vs Original: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    if "torch_gpu" in results and "error" not in results["torch_gpu"] and "error" not in r_orig:
        speedup_gpu = r_orig["mean_s"] / max(results["torch_gpu"]["mean_s"], 1e-6)
        print(f"  PyTorch GPU vs Original: {speedup_gpu:.2f}x {'faster' if speedup_gpu > 1 else 'slower'}")
    print()

    if args.output:
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(json.dumps(results, indent=2, default=str))
        print(f"Results saved to {outpath}")


if __name__ == "__main__":
    main()
