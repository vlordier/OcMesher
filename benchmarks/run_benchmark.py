# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Comprehensive benchmark: original Python + C++ backend vs. PyTorch backend.

Run from the repository root::

    python -m benchmarks.run_benchmark          # quick (small grid)
    python -m benchmarks.run_benchmark --full   # full comparison
    python -m benchmarks.run_benchmark --profile # sub-operation profiling
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# SDF kernels at varying complexity levels
# ---------------------------------------------------------------------------
try:
    import vnoise

    _noise = vnoise.Noise()

    def sdf_terrain(xyz: np.ndarray) -> np.ndarray:
        """Perlin-noise height field (medium complexity)."""
        scale = 2
        h = _noise.noise2(xyz[:, 0] / scale, xyz[:, 1] / scale, grid_mode=False, octaves=4)
        return xyz[:, 2] - h

except ImportError:

    def sdf_terrain(xyz: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        """Fallback SDF - flat plane at z=0."""
        return xyz[:, 2].copy()


def sdf_sphere(xyz: np.ndarray) -> np.ndarray:
    """Simple sphere SDF (low complexity)."""
    return np.linalg.norm(xyz, axis=1) - 5.0


def sdf_gyroid(xyz: np.ndarray) -> np.ndarray:
    """Gyroid implicit surface (high complexity, trigonometric)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


_SDF_KERNELS = {
    "sphere": sdf_sphere,
    "terrain": sdf_terrain,
    "gyroid": sdf_gyroid,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cameras(n_cameras: int = 1):
    """Create *n_cameras* cameras arranged around the scene."""
    cam_poses = []
    ks = []
    hs = []
    ws = []
    for i in range(n_cameras):
        angle = 2 * np.pi * i / max(n_cameras, 1)
        c, s = np.cos(angle), np.sin(angle)
        pose = np.array(
            [
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 3],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        cam_poses.append(pose)
        ks.append(
            np.array([[2000, 0, 640], [0, 2000, 360], [0, 0, 1]], dtype=np.float64),
        )
        hs.append(720)
        ws.append(1280)
    return (cam_poses, ks, hs, ws)


def _make_bounds():
    return (-10, 10, -10, 10, -2, 2)


def _mps_available() -> bool:
    """Return True if MPS (Apple Silicon GPU) backend is available."""
    try:
        import torch

        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def _system_info():
    """Collect system information for the benchmark report."""
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }
    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
        info["mps_available"] = _mps_available()
    except ImportError:
        info["pytorch_version"] = "NOT INSTALLED"
    return info


def _stats(times: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of timings."""
    if not times:
        return {}
    result = {
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
    }
    if len(times) > 1:
        result["std_s"] = statistics.stdev(times)
        result["p95_s"] = float(np.percentile(times, 95))
    return result


# ---------------------------------------------------------------------------
# End-to-end benchmark runners
# ---------------------------------------------------------------------------
def _bench_original(
    cameras,
    bounds,
    pixels_per_cube: int,
    sdf_name: str,
    n_runs: int = 1,
    warmup: int = 0,
):
    """Benchmark the original Python + C++ OcMesher."""
    try:
        from ocmesher.core import OcMesher
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not load original OcMesher: {exc}"}

    kernel = _SDF_KERNELS[sdf_name]
    try:
        # Warmup
        for _ in range(warmup):
            mesher = OcMesher(cameras, bounds, pixels_per_cube=pixels_per_cube)
            mesher([kernel])

        times: list[float] = []
        mesh_info: dict[str, object] = {}
        for i in range(n_runs):
            t0 = time.perf_counter()
            mesher = OcMesher(cameras, bounds, pixels_per_cube=pixels_per_cube)
            meshes, _tags = mesher([kernel])
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if i == 0:
                mesh_info = {
                    "vertices": int(meshes[0].vertices.shape[0]),
                    "faces": int(meshes[0].faces.shape[0]),
                }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"C++ backend not available: {exc}"}

    return {
        "backend": "python_cpp",
        "sdf": sdf_name,
        "times_s": times,
        **_stats(times),
        **mesh_info,
    }


def _bench_torch(
    cameras,
    bounds,
    pixels_per_cube: int,
    sdf_name: str,
    n_runs: int = 1,
    warmup: int = 0,
    device: str | None = None,
    n_sdf_workers: int = 4,
):
    """Benchmark the PyTorch TorchOcMesher."""
    try:
        import torch  # noqa: F401

        from ocmesher.torch_core import TorchOcMesher
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not load TorchOcMesher: {exc}"}

    kernel = _SDF_KERNELS[sdf_name]
    # Warmup
    for _ in range(warmup):
        mesher = TorchOcMesher(
            cameras,
            bounds,
            pixels_per_cube=pixels_per_cube,
            device=device,
            n_sdf_workers=n_sdf_workers,
        )
        mesher([kernel])

    times: list[float] = []
    mesh_info: dict[str, object] = {}
    for i in range(n_runs):
        t0 = time.perf_counter()
        mesher = TorchOcMesher(
            cameras,
            bounds,
            pixels_per_cube=pixels_per_cube,
            device=device,
            n_sdf_workers=n_sdf_workers,
        )
        meshes, _tags = mesher([kernel])
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if i == 0:
            mesh_info = {
                "vertices": int(meshes[0].vertices.shape[0]),
                "faces": int(meshes[0].faces.shape[0]),
            }

    return {
        "backend": f"pytorch_{device or 'auto'}",
        "sdf": sdf_name,
        "times_s": times,
        **_stats(times),
        **mesh_info,
    }


# ---------------------------------------------------------------------------
# Sub-operation micro-benchmarks
# ---------------------------------------------------------------------------
def _micro_sdf_eval(cameras, bounds, n_points: int = 500_000, n_runs: int = 5):
    """Compare SDF evaluation throughput (numpy vs torch+threading)."""
    results: dict[str, object] = {"n_points": n_points}
    rng = np.random.default_rng(42)
    pts = rng.uniform(
        [bounds[0], bounds[2], bounds[4]],
        [bounds[1], bounds[3], bounds[5]],
        size=(n_points, 3),
    )

    for sdf_name, kernel in _SDF_KERNELS.items():
        # NumPy direct
        times_np: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = kernel(pts.copy())
            times_np.append(time.perf_counter() - t0)
        results[f"numpy_{sdf_name}_mean_s"] = statistics.mean(times_np)

    # Torch threaded eval
    try:
        import torch

        from ocmesher.torch_core import TorchOcMesher

        for workers in [1, 2, 4]:
            mesher = TorchOcMesher(cameras, bounds, device="cpu", n_sdf_workers=workers)
            pts_t = torch.from_numpy(pts).to(dtype=torch.float64, device=mesher.device)
            times_torch: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = mesher._evaluate_sdf([sdf_terrain], pts_t)
                times_torch.append(time.perf_counter() - t0)
            results[f"torch_cpu_{workers}w_mean_s"] = statistics.mean(times_torch)
    except Exception as exc:  # noqa: BLE001
        results["torch_error"] = str(exc)

    return results


def _micro_projection(cameras, bounds, n_cubes: int = 100_000, n_runs: int = 5):
    """Compare camera-projection throughput (batched vs per-camera)."""
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
        results["torch_cpu_mean_s"] = statistics.mean(times)
        results["n_cameras"] = mesher.n_cameras

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
            results["torch_gpu_mean_s"] = statistics.mean(times_gpu)

        if _mps_available():
            mesher_mps = TorchOcMesher(cameras, bounds, device="mps")
            coords_m = coords.to(mesher_mps.device)
            levels_m = levels.to(mesher_mps.device)
            positions_m = mesher_mps._cube_centers(coords_m, levels_m)
            times_mps: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = mesher_mps._projected_sizes(positions_m, levels_m)
                torch.mps.synchronize()
                times_mps.append(time.perf_counter() - t0)
            results["torch_mps_mean_s"] = statistics.mean(times_mps)
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)

    return results


def _micro_marching_cubes(cameras, bounds, n_runs: int = 3):
    """Benchmark marching cubes in isolation."""
    results: dict[str, object] = {}
    try:
        from ocmesher.torch_core import TorchOcMesher

        mesher = TorchOcMesher(cameras, bounds, device="cpu")
        # Build a small octree and get corner SDF values
        coords, levels = mesher._build_coarse_octree()
        mask, _ = mesher._find_surface_cubes([sdf_terrain], coords, levels)
        s_coords = coords[mask]
        s_levels = levels[mask]
        corners = mesher._cube_corner_positions(s_coords, s_levels)
        flat = corners.reshape(-1, 3)
        sdf_all = mesher._evaluate_sdf([sdf_terrain], flat)
        sdf_min = sdf_all.min(dim=-1).values.reshape(len(s_coords), 8)

        results["n_surface_cubes"] = len(s_coords)
        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            v, f = mesher._marching_cubes(corners, sdf_min)
            times.append(time.perf_counter() - t0)
        results["torch_cpu_mean_s"] = statistics.mean(times)
        results["output_vertices"] = int(v.shape[0])
        results["output_faces"] = int(f.shape[0])
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
    return results


def _micro_octree(cameras, bounds, n_runs: int = 3):
    """Benchmark octree construction in isolation."""
    results: dict[str, object] = {}
    try:
        from ocmesher.torch_core import TorchOcMesher

        mesher = TorchOcMesher(cameras, bounds, device="cpu")
        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            coords, _levels = mesher._build_coarse_octree()
            times.append(time.perf_counter() - t0)
        results["torch_cpu_mean_s"] = statistics.mean(times)
        results["n_cubes"] = len(coords)
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
    return results


def _micro_visibility(cameras, bounds, n_runs: int = 3):
    """Benchmark visibility filter in isolation."""
    results: dict[str, object] = {}
    try:
        import torch

        from ocmesher.torch_core import TorchOcMesher

        mesher = TorchOcMesher(cameras, bounds, device="cpu")
        rng = np.random.default_rng(42)
        n_pts = 50_000
        positions = torch.from_numpy(
            rng.uniform(
                [bounds[0], bounds[2], bounds[4]],
                [bounds[1], bounds[3], bounds[5]],
                size=(n_pts, 3),
            ),
        ).to(dtype=torch.float64, device=mesher.device)
        results["n_points"] = n_pts

        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = mesher._visibility_filter(positions)
            times.append(time.perf_counter() - t0)
        results["torch_cpu_mean_s"] = statistics.mean(times)
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
    return results


def _scaling_cameras(bounds, max_cameras: int = 8, n_runs: int = 1):
    """Measure how performance scales with number of cameras."""
    results: list[dict[str, object]] = []
    try:
        from ocmesher.torch_core import TorchOcMesher

        for nc in [1, 2, 4, max_cameras]:
            cameras = _make_cameras(nc)
            times: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                mesher = TorchOcMesher(cameras, bounds, pixels_per_cube=16, device="cpu")
                mesher([sdf_terrain])
                times.append(time.perf_counter() - t0)
            results.append(
                {
                    "n_cameras": nc,
                    **_stats(times),
                }
            )
    except Exception as exc:  # noqa: BLE001
        results.append({"error": str(exc)})
    return results


def _memory_usage(cameras, bounds):
    """Measure peak memory usage during meshing."""
    results: dict[str, object] = {}
    try:
        import tracemalloc

        from ocmesher.torch_core import TorchOcMesher

        tracemalloc.start()
        mesher = TorchOcMesher(cameras, bounds, pixels_per_cube=16, device="cpu")
        mesher([sdf_terrain])
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results["torch_current_mb"] = round(current / 1e6, 1)
        results["torch_peak_mb"] = round(peak / 1e6, 1)
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
    return results


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def _print_section(title: str):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def _print_result(result: dict):
    if "error" in result:
        print(f"  SKIPPED: {result['error']}")
        return
    for k, v in result.items():
        if k == "times_s":
            continue
        if isinstance(v, float):
            print(f"  {k:24s}: {v:.4f}")
        else:
            print(f"  {k:24s}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run benchmarks and print results."""
    parser = argparse.ArgumentParser(description="OcMesher benchmark: Python+C++ vs PyTorch")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (slower, higher resolution)")
    parser.add_argument("--profile", action="store_true", help="Run sub-operation micro-benchmarks")
    parser.add_argument("--runs", type=int, default=3, help="Number of end-to-end runs (default 3)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs (default 1)")
    parser.add_argument("--sdf", type=str, default=None, help="SDF to benchmark (sphere/terrain/gyroid or 'all')")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    pixels_per_cube = 8 if args.full else 16
    cameras = _make_cameras(1)
    bounds = _make_bounds()
    sdf_names = list(_SDF_KERNELS.keys()) if args.sdf == "all" else [args.sdf or "terrain"]

    print("=" * 70)
    print("OcMesher Comprehensive Benchmark")
    print("=" * 70)

    sys_info = _system_info()
    for k, v in sys_info.items():
        print(f"  {k:20s}: {v}")
    print(f"  pixels_per_cube   : {pixels_per_cube}")
    print(f"  runs              : {args.runs}")
    print(f"  warmup            : {args.warmup}")
    print(f"  SDFs              : {sdf_names}")

    results: dict[str, object] = {"system": sys_info}

    # End-to-end benchmarks ------------------------------------------------
    for sdf_name in sdf_names:
        _print_section(f"End-to-end: Python + C++ ({sdf_name})")
        r_orig = _bench_original(cameras, bounds, pixels_per_cube, sdf_name, n_runs=args.runs, warmup=args.warmup)
        _print_result(r_orig)
        results[f"original_{sdf_name}"] = r_orig

        _print_section(f"End-to-end: PyTorch CPU ({sdf_name})")
        r_torch = _bench_torch(
            cameras, bounds, pixels_per_cube, sdf_name, n_runs=args.runs, warmup=args.warmup, device="cpu"
        )
        _print_result(r_torch)
        results[f"torch_cpu_{sdf_name}"] = r_torch

        try:
            import torch

            if torch.cuda.is_available():
                _print_section(f"End-to-end: PyTorch CUDA ({sdf_name})")
                r_gpu = _bench_torch(
                    cameras, bounds, pixels_per_cube, sdf_name, n_runs=args.runs, warmup=args.warmup, device="cuda"
                )
                _print_result(r_gpu)
                results[f"torch_gpu_{sdf_name}"] = r_gpu
            if _mps_available():
                _print_section(f"End-to-end: PyTorch MPS ({sdf_name})")
                r_mps = _bench_torch(
                    cameras, bounds, pixels_per_cube, sdf_name, n_runs=args.runs, warmup=args.warmup, device="mps"
                )
                _print_result(r_mps)
                results[f"torch_mps_{sdf_name}"] = r_mps
        except ImportError:
            pass

    # Micro-benchmarks (only with --profile) --------------------------------
    if args.profile:
        _print_section("Micro-benchmark: SDF evaluation (threaded)")
        r_sdf = _micro_sdf_eval(cameras, bounds)
        _print_result(r_sdf)
        results["micro_sdf"] = r_sdf

        _print_section("Micro-benchmark: Camera projection (batched)")
        r_proj = _micro_projection(cameras, bounds)
        _print_result(r_proj)
        results["micro_projection"] = r_proj

        _print_section("Micro-benchmark: Marching cubes")
        r_mc = _micro_marching_cubes(cameras, bounds)
        _print_result(r_mc)
        results["micro_marching_cubes"] = r_mc

        _print_section("Micro-benchmark: Octree construction")
        r_oct = _micro_octree(cameras, bounds)
        _print_result(r_oct)
        results["micro_octree"] = r_oct

        _print_section("Micro-benchmark: Visibility filter")
        r_vis = _micro_visibility(cameras, bounds)
        _print_result(r_vis)
        results["micro_visibility"] = r_vis

        _print_section("Scaling: Multi-camera performance")
        r_scale = _scaling_cameras(bounds)
        for entry in r_scale:
            _print_result(entry)
        results["scaling_cameras"] = r_scale

        _print_section("Memory usage")
        r_mem = _memory_usage(cameras, bounds)
        _print_result(r_mem)
        results["memory"] = r_mem

    # Speedup summary ------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for sdf_name in sdf_names:
        r_orig = results.get(f"original_{sdf_name}", {})
        r_torch = results.get(f"torch_cpu_{sdf_name}", {})
        r_gpu = results.get(f"torch_gpu_{sdf_name}", {})
        r_mps = results.get(f"torch_mps_{sdf_name}", {})

        if "error" not in r_orig and "error" not in r_torch:
            sp = r_orig["mean_s"] / max(r_torch["mean_s"], 1e-6)
            tag = "faster" if sp > 1 else "slower"
            print(
                f"  [{sdf_name}] PyTorch CPU  vs C++: {sp:.2f}x {tag}  ({r_torch['mean_s']:.3f}s vs {r_orig['mean_s']:.3f}s)"
            )
        if r_gpu and "error" not in r_gpu and "error" not in r_orig:
            sp_g = r_orig["mean_s"] / max(r_gpu["mean_s"], 1e-6)
            tag_g = "faster" if sp_g > 1 else "slower"
            print(
                f"  [{sdf_name}] PyTorch CUDA vs C++: {sp_g:.2f}x {tag_g}  ({r_gpu['mean_s']:.3f}s vs {r_orig['mean_s']:.3f}s)"
            )
        if r_mps and "error" not in r_mps and "error" not in r_orig:
            sp_m = r_orig["mean_s"] / max(r_mps["mean_s"], 1e-6)
            tag_m = "faster" if sp_m > 1 else "slower"
            print(
                f"  [{sdf_name}] PyTorch MPS  vs C++: {sp_m:.2f}x {tag_m}  ({r_mps['mean_s']:.3f}s vs {r_orig['mean_s']:.3f}s)"
            )
    print()

    if args.output:
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(json.dumps(results, indent=2, default=str))
        print(f"Results saved to {outpath}")


if __name__ == "__main__":
    main()
