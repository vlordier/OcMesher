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
import logging
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import psutil

logger = logging.getLogger(__name__)

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
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }
    # CPU architecture details
    info["arch"] = platform.machine()
    try:
        import subprocess

        # All inputs are hardcoded (no user-controlled data); shell=False is the
        # default for list-form subprocess.run, so injection risk is negligible.
        result = subprocess.run(  # noqa: S603
            ["lscpu"] if platform.system() == "Linux" else ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            shell=False,
        )
        if result.returncode == 0 and platform.system() == "Linux":
            for line in result.stdout.splitlines():
                if "Model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
                if "Flags" in line:
                    flags = set(line.split(":", 1)[1].split())
                    # Best-effort ISA detection from lscpu Flags field.
                    # x86_64 flags: avx, avx2, avx512f (+ variants).
                    # aarch64 flags: neon (often listed as asimd), sve.
                    # Only a representative subset is shown here.
                    _isa_candidates = ("avx", "avx2", "avx512f", "asimd", "neon", "sve")
                    info["isa_extensions"] = sorted(f for f in _isa_candidates if f in flags)
                    break
        elif result.returncode == 0 and platform.system() == "Darwin":
            info["cpu_model"] = result.stdout.strip()
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            info["cuda_device_count"] = torch.cuda.device_count()
        info["mps_available"] = _mps_available()
        info["torch_compile"] = hasattr(torch, "compile")
        info["cpu_threads"] = torch.get_num_threads()
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
    *,
    use_compile: bool = False,
):
    """Benchmark the PyTorch TorchOcMesher."""
    try:
        import torch  # noqa: F401

        from ocmesher.torch_core import TorchOcMesher
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not load TorchOcMesher: {exc}"}

    kernel = _SDF_KERNELS[sdf_name]
    # Warmup (especially important when use_compile=True to amortise JIT cost)
    for _ in range(warmup):
        mesher = TorchOcMesher(
            cameras,
            bounds,
            pixels_per_cube=pixels_per_cube,
            device=device,
            n_sdf_workers=n_sdf_workers,
            use_compile=use_compile,
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
            use_compile=use_compile,
        )
        meshes, _tags = mesher([kernel])
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if i == 0:
            mesh_info = {
                "vertices": int(meshes[0].vertices.shape[0]),
                "faces": int(meshes[0].faces.shape[0]),
            }

    backend_tag = f"pytorch_{device or 'auto'}"
    if use_compile:
        backend_tag += "+compile"
    return {
        "backend": backend_tag,
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


def _micro_vertex_dedup(cameras, bounds, n_runs: int = 5):
    """Benchmark GPU-accelerated vertex deduplication vs numpy baseline."""
    results: dict[str, object] = {}
    try:
        import torch

        from ocmesher.torch_core import TorchOcMesher

        mesher = TorchOcMesher(cameras, bounds, device="cpu")
        coords, levels = mesher._build_coarse_octree()
        mask, _ = mesher._find_surface_cubes([sdf_terrain], coords, levels)
        s_coords = coords[mask]
        s_levels = levels[mask]
        corners = mesher._cube_corner_positions(s_coords, s_levels)
        flat = corners.reshape(-1, 3)
        sdf_all = mesher._evaluate_sdf([sdf_terrain], flat)
        sdf_min = sdf_all.min(dim=-1).values.reshape(len(s_coords), 8)

        # Generate vertex data for dedup comparison
        v, _f = mesher._marching_cubes(corners, sdf_min)
        n_verts = v.shape[0]
        results["n_input_vertices"] = int(n_verts * 3)
        results["n_dedup_vertices"] = n_verts

        # Benchmark numpy.unique baseline
        verts_t = torch.from_numpy(v).float()
        verts_dup = verts_t.repeat(3, 1)  # simulate duplicated vertices
        verts_np = verts_dup.numpy()

        times_np: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            quantized = np.round(verts_np * 1e8).astype(np.int64)
            np.unique(quantized, axis=0, return_index=True, return_inverse=True)
            times_np.append(time.perf_counter() - t0)
        results["numpy_unique_mean_s"] = statistics.mean(times_np)

        # Benchmark torch hash-based dedup
        times_torch: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            q = (verts_dup * 1e8).round().long()
            hv = q[:, 0] * 1000000007 + q[:, 1] * 1000000009 + q[:, 2] * 1000000021
            torch.unique(hv, return_inverse=True)
            times_torch.append(time.perf_counter() - t0)
        results["torch_hash_mean_s"] = statistics.mean(times_torch)

        min_reliable = 1e-6  # below this, timings are noise
        np_time = results["numpy_unique_mean_s"]
        th_time = results["torch_hash_mean_s"]
        if np_time > min_reliable and th_time > min_reliable:
            results["speedup"] = round(np_time / th_time, 1)
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
    return results


def _micro_coord_computation(_cameras, _bounds, n_runs: int = 5):
    """Benchmark ldexp vs pow for coordinate computation."""
    results: dict[str, object] = {}
    try:
        import torch

        rng = np.random.default_rng(42)
        n_cubes = 500_000
        levels = torch.from_numpy(rng.integers(1, 15, size=n_cubes))

        # Benchmark pow-based scale computation
        times_pow: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = 1.0 / (2.0 ** levels.double())
            times_pow.append(time.perf_counter() - t0)
        results["pow2_mean_s"] = statistics.mean(times_pow)

        # ldexp
        ones = torch.ones_like(levels, dtype=torch.float64)
        times_ldexp: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = 1.0 / torch.ldexp(ones, levels)
            times_ldexp.append(time.perf_counter() - t0)
        results["ldexp_mean_s"] = statistics.mean(times_ldexp)
        results["n_cubes"] = n_cubes

        min_reliable = 1e-6
        if results["pow2_mean_s"] > min_reliable and results["ldexp_mean_s"] > min_reliable:
            results["speedup"] = round(results["pow2_mean_s"] / results["ldexp_mean_s"], 1)
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


def _micro_visibility_multicam(bounds, n_runs: int = 3):
    """Benchmark visibility filter scaling with multiple cameras."""
    results: list[dict[str, object]] = []
    try:
        import torch

        from ocmesher.torch_core import TorchOcMesher

        rng = np.random.default_rng(42)
        n_pts = 50_000
        for nc in [1, 2, 4, 8]:
            cameras = _make_cameras(nc)
            mesher = TorchOcMesher(cameras, bounds, device="cpu")
            positions = torch.from_numpy(
                rng.uniform(
                    [bounds[0], bounds[2], bounds[4]],
                    [bounds[1], bounds[3], bounds[5]],
                    size=(n_pts, 3),
                ),
            ).to(dtype=torch.float64, device=mesher.device)

            times: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = mesher._visibility_filter(positions)
                times.append(time.perf_counter() - t0)
            results.append(
                {
                    "n_cameras": nc,
                    "n_points": n_pts,
                    **_stats(times),
                }
            )
    except Exception as exc:  # noqa: BLE001
        results.append({"error": str(exc)})
    return results


def _micro_triangle_extraction(cameras, bounds, n_runs: int = 5):
    """Benchmark vectorised triangle extraction in marching cubes."""
    results: dict[str, object] = {}
    try:
        from ocmesher.torch_core import TorchOcMesher

        mesher = TorchOcMesher(cameras, bounds, device="cpu")
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
        results["mean_s"] = statistics.mean(times)
        results["output_vertices"] = int(v.shape[0])
        results["output_faces"] = int(f.shape[0])
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
    return results


def _micro_init_vectorised(n_runs: int = 20):
    """Benchmark vectorised camera initialisation vs loop-based."""
    results: dict[str, object] = {}
    try:
        from ocmesher.torch_core import TorchOcMesher

        bounds = _make_bounds()
        for nc in [1, 4, 8, 16]:
            cameras = _make_cameras(nc)
            times: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                TorchOcMesher(cameras, bounds, device="cpu")
                times.append(time.perf_counter() - t0)
            results[f"init_{nc}cam_mean_s"] = statistics.mean(times)
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
    return results


def _micro_dtype_comparison(cameras, bounds, n_runs: int = 5):
    """Benchmark float32 vs float64 throughput on available accelerators.

    This micro-benchmark directly measures the impact of ``_fdtype`` on GPU
    computation kernels (projection, visibility, marching cubes).

    Results with ``dtype=float32`` are expected to be 2-4x faster than
    ``float64`` on CUDA (due to Tensor Cores and memory bandwidth) and are
    effectively *required* on MPS (float64 not supported).
    """
    results: dict[str, object] = {}
    try:
        import torch

        from ocmesher.torch_core import TorchOcMesher

        rng = np.random.default_rng(42)
        n_pts = 100_000
        pts_np = rng.uniform(
            [bounds[0], bounds[2], bounds[4]],
            [bounds[1], bounds[3], bounds[5]],
            size=(n_pts, 3),
        )

        for dev_str in ("cpu", "cuda", "mps"):
            if dev_str == "cuda" and not torch.cuda.is_available():
                continue
            if dev_str == "mps" and not _mps_available():
                continue
            try:
                # float32 path (used automatically on CUDA/MPS)
                mesher = TorchOcMesher(cameras, bounds, device=dev_str)
                actual_dtype = mesher._fdtype
                pts_t = torch.from_numpy(pts_np).to(dtype=actual_dtype, device=mesher.device)
                levels = torch.full((n_pts,), 6, dtype=torch.int64, device=mesher.device)

                times_native: list[float] = []
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    _ = mesher._projected_sizes(pts_t, levels)
                    if dev_str == "cuda":
                        torch.cuda.synchronize()
                    elif dev_str == "mps":
                        torch.mps.synchronize()
                    times_native.append(time.perf_counter() - t0)
                results[f"{dev_str}_active_dtype"] = str(actual_dtype).replace("torch.", "")
                results[f"{dev_str}_projection_mean_s"] = statistics.mean(times_native)

                # On CPU, also measure float32 explicitly for comparison
                if dev_str == "cpu":
                    pts_f32 = pts_t.float()
                    times_cpu32: list[float] = []
                    for _ in range(n_runs):
                        # Use raw ops to measure float32 overhead vs float64
                        t0 = time.perf_counter()
                        pos_h = torch.nn.functional.pad(pts_f32, (0, 1), value=1.0)
                        cam_inv = mesher.cam_inv_poses.float()
                        pos_h_t = pos_h.T.unsqueeze(0).expand(mesher.n_cameras, -1, -1)
                        torch.bmm(cam_inv, pos_h_t)
                        times_cpu32.append(time.perf_counter() - t0)
                    results["cpu_float32_projection_mean_s"] = statistics.mean(times_cpu32)
            except Exception as exc:  # noqa: BLE001
                results[f"{dev_str}_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
    return results


def _bench_cpu_threads(cameras, bounds, sdf_name: str = "terrain", n_runs: int = 3):
    """Benchmark end-to-end PyTorch CPU performance across different thread counts.

    Uses :func:`torch.set_num_threads` to vary the intra-op parallelism and
    measures how total throughput scales with thread count.  Useful for
    diagnosing over/under-subscription on machines with many cores.
    """
    results: list[dict[str, object]] = []
    try:
        import torch

        from ocmesher.torch_core import TorchOcMesher

        original_threads = torch.get_num_threads()
        cpu_count = psutil.cpu_count(logical=True) or 1
        thread_counts = sorted({1, 2, 4, min(8, cpu_count), cpu_count})
        kernel = _SDF_KERNELS[sdf_name]

        for n_threads in thread_counts:
            torch.set_num_threads(n_threads)
            times: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                mesher = TorchOcMesher(cameras, bounds, pixels_per_cube=16, device="cpu")
                mesher([kernel])
                times.append(time.perf_counter() - t0)
            results.append(
                {
                    "n_threads": n_threads,
                    **_stats(times),
                }
            )

        torch.set_num_threads(original_threads)
    except Exception as exc:  # noqa: BLE001
        results.append({"error": str(exc)})
    return results


def _bench_compile(cameras, bounds, sdf_name: str = "terrain", n_runs: int = 3, warmup: int = 2):
    """Benchmark ``torch.compile`` JIT impact on CPU and CUDA.

    Compares plain TorchOcMesher vs ``use_compile=True`` to quantify the
    overhead amortisation from JIT compilation.  The first call with
    ``use_compile=True`` includes compilation cost; subsequent calls use the
    compiled graph.
    """
    results: dict[str, object] = {}
    try:
        import torch

        if not hasattr(torch, "compile"):
            return {"error": "torch.compile not available (requires PyTorch >= 2.0)"}

        from ocmesher.torch_core import TorchOcMesher

        kernel = _SDF_KERNELS[sdf_name]
        for dev_str in ("cpu", "cuda"):
            if dev_str == "cuda" and not torch.cuda.is_available():
                continue
            for use_c in (False, True):
                label = f"{dev_str}_{'compile' if use_c else 'eager'}"
                try:
                    # warmup
                    for _ in range(warmup):
                        m = TorchOcMesher(cameras, bounds, pixels_per_cube=16, device=dev_str, use_compile=use_c)
                        m([kernel])
                    times: list[float] = []
                    for _ in range(n_runs):
                        t0 = time.perf_counter()
                        m = TorchOcMesher(cameras, bounds, pixels_per_cube=16, device=dev_str, use_compile=use_c)
                        m([kernel])
                        times.append(time.perf_counter() - t0)
                    results[f"{label}_mean_s"] = statistics.mean(times)
                except Exception as exc:  # noqa: BLE001
                    results[f"{label}_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        results["error"] = str(exc)
    return results


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _log_section(title: str):
    logger.info("")
    logger.info("%s", "-" * 70)
    logger.info("%s", title)
    logger.info("%s", "-" * 70)


def _log_result(result: dict):
    if "error" in result:
        logger.info("  SKIPPED: %s", result["error"])
        return
    for k, v in result.items():
        if k == "times_s":
            continue
        if isinstance(v, float):
            logger.info("  %-24s: %.4f", k, v)
        else:
            logger.info("  %-24s: %s", k, v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run benchmarks and log results."""
    parser = argparse.ArgumentParser(description="OcMesher benchmark: Python+C++ vs PyTorch")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (slower, higher resolution)")
    parser.add_argument("--profile", action="store_true", help="Run sub-operation micro-benchmarks")
    parser.add_argument("--runs", type=int, default=3, help="Number of end-to-end runs (default 3)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs (default 1)")
    parser.add_argument("--sdf", type=str, default=None, help="SDF to benchmark (sphere/terrain/gyroid or 'all')")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    parser.add_argument(
        "--device",
        type=str,
        default="all",
        help="Device(s) to benchmark: cpu, cuda, mps, or 'all' (default: all available)",
    )
    parser.add_argument(
        "--threads",
        action="store_true",
        help="Run CPU thread-scaling benchmark (requires --profile)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Run torch.compile impact benchmark (requires --profile)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging (includes per-step mesher output)",
    )
    args = parser.parse_args()

    # Configure logging: verbose mode shows DEBUG + timestamps; default shows INFO only.
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    pixels_per_cube = 8 if args.full else 16
    cameras = _make_cameras(1)
    bounds = _make_bounds()
    sdf_names = list(_SDF_KERNELS.keys()) if args.sdf == "all" else [args.sdf or "terrain"]

    # Resolve which devices to benchmark
    try:
        import torch as _torch

        _cuda_ok = _torch.cuda.is_available()
        _mps_ok = _mps_available()
    except ImportError:
        _cuda_ok = False
        _mps_ok = False

    if args.device == "all":
        bench_devices = ["cpu"]
        if _cuda_ok:
            bench_devices.append("cuda")
        if _mps_ok:
            bench_devices.append("mps")
    else:
        bench_devices = [d.strip() for d in args.device.split(",")]

    logger.info("%s", "=" * 70)
    logger.info("OcMesher Comprehensive Benchmark")
    logger.info("%s", "=" * 70)

    sys_info = _system_info()
    for k, v in sys_info.items():
        logger.info("  %-24s: %s", k, v)
    logger.info("  %-24s: %s", "pixels_per_cube", pixels_per_cube)
    logger.info("  %-24s: %s", "runs", args.runs)
    logger.info("  %-24s: %s", "warmup", args.warmup)
    logger.info("  %-24s: %s", "SDFs", sdf_names)
    logger.info("  %-24s: %s", "devices", bench_devices)

    results: dict[str, object] = {"system": sys_info}

    # End-to-end benchmarks ------------------------------------------------
    for sdf_name in sdf_names:
        _log_section(f"End-to-end: Python + C++ ({sdf_name})")
        r_orig = _bench_original(cameras, bounds, pixels_per_cube, sdf_name, n_runs=args.runs, warmup=args.warmup)
        _log_result(r_orig)
        results[f"original_{sdf_name}"] = r_orig

        for dev_str in bench_devices:
            label = {"cpu": "CPU", "cuda": "CUDA", "mps": "MPS"}.get(dev_str, dev_str.upper())
            _log_section(f"End-to-end: PyTorch {label} ({sdf_name})")
            r_torch = _bench_torch(
                cameras,
                bounds,
                pixels_per_cube,
                sdf_name,
                n_runs=args.runs,
                warmup=args.warmup,
                device=dev_str,
            )
            _log_result(r_torch)
            results[f"torch_{dev_str}_{sdf_name}"] = r_torch

    # Micro-benchmarks (only with --profile) --------------------------------
    if args.profile:
        _log_section("Micro-benchmark: SDF evaluation (threaded)")
        r_sdf = _micro_sdf_eval(cameras, bounds)
        _log_result(r_sdf)
        results["micro_sdf"] = r_sdf

        _log_section("Micro-benchmark: Camera projection (batched)")
        r_proj = _micro_projection(cameras, bounds)
        _log_result(r_proj)
        results["micro_projection"] = r_proj

        _log_section("Micro-benchmark: Marching cubes")
        r_mc = _micro_marching_cubes(cameras, bounds)
        _log_result(r_mc)
        results["micro_marching_cubes"] = r_mc

        _log_section("Micro-benchmark: Vertex deduplication (numpy vs torch hash)")
        r_dedup = _micro_vertex_dedup(cameras, bounds)
        _log_result(r_dedup)
        results["micro_vertex_dedup"] = r_dedup

        _log_section("Micro-benchmark: Coordinate computation (pow vs ldexp)")
        r_coord = _micro_coord_computation(cameras, bounds)
        _log_result(r_coord)
        results["micro_coord_computation"] = r_coord

        _log_section("Micro-benchmark: Octree construction")
        r_oct = _micro_octree(cameras, bounds)
        _log_result(r_oct)
        results["micro_octree"] = r_oct

        _log_section("Micro-benchmark: Visibility filter")
        r_vis = _micro_visibility(cameras, bounds)
        _log_result(r_vis)
        results["micro_visibility"] = r_vis

        _log_section("Micro-benchmark: Visibility filter multi-camera scaling")
        r_vis_mc = _micro_visibility_multicam(bounds)
        for entry in r_vis_mc:
            _log_result(entry)
        results["micro_visibility_multicam"] = r_vis_mc

        _log_section("Micro-benchmark: Triangle extraction (vectorised)")
        r_tri = _micro_triangle_extraction(cameras, bounds)
        _log_result(r_tri)
        results["micro_triangle_extraction"] = r_tri

        _log_section("Micro-benchmark: Initialisation (vectorised)")
        r_init = _micro_init_vectorised()
        _log_result(r_init)
        results["micro_init"] = r_init

        _log_section("Micro-benchmark: float32 vs float64 dtype throughput")
        r_dtype = _micro_dtype_comparison(cameras, bounds)
        _log_result(r_dtype)
        results["micro_dtype"] = r_dtype

        _log_section("Scaling: Multi-camera performance")
        r_scale = _scaling_cameras(bounds)
        for entry in r_scale:
            _log_result(entry)
        results["scaling_cameras"] = r_scale

        _log_section("Memory usage")
        r_mem = _memory_usage(cameras, bounds)
        _log_result(r_mem)
        results["memory"] = r_mem

        if args.threads:
            _log_section("Scaling: CPU thread count")
            r_thr = _bench_cpu_threads(cameras, bounds)
            for entry in r_thr:
                _log_result(entry)
            results["scaling_threads"] = r_thr

        if args.compile:
            _log_section("Benchmark: torch.compile impact")
            r_comp = _bench_compile(cameras, bounds)
            _log_result(r_comp)
            results["bench_compile"] = r_comp

    # Speedup summary ------------------------------------------------------
    logger.info("")
    logger.info("%s", "=" * 70)
    logger.info("SUMMARY")
    logger.info("%s", "=" * 70)
    for sdf_name in sdf_names:
        r_orig = results.get(f"original_{sdf_name}", {})
        r_torch_cpu = results.get(f"torch_cpu_{sdf_name}", {})
        r_gpu = results.get(f"torch_cuda_{sdf_name}", {})
        r_mps = results.get(f"torch_mps_{sdf_name}", {})

        if r_orig and "error" not in r_orig and r_torch_cpu and "error" not in r_torch_cpu:
            sp = r_orig["mean_s"] / max(r_torch_cpu["mean_s"], 1e-6)
            tag = "faster" if sp > 1 else "slower"
            logger.info(
                "  [%s] PyTorch CPU  vs C++: %.2fx %s  (%.3fs vs %.3fs)",
                sdf_name, sp, tag, r_torch_cpu["mean_s"], r_orig["mean_s"],
            )
        if r_gpu and "error" not in r_gpu and r_orig and "error" not in r_orig:
            sp_g = r_orig["mean_s"] / max(r_gpu["mean_s"], 1e-6)
            tag_g = "faster" if sp_g > 1 else "slower"
            logger.info(
                "  [%s] PyTorch CUDA vs C++: %.2fx %s  (%.3fs vs %.3fs)",
                sdf_name, sp_g, tag_g, r_gpu["mean_s"], r_orig["mean_s"],
            )
        if r_mps and "error" not in r_mps and r_orig and "error" not in r_orig:
            sp_m = r_orig["mean_s"] / max(r_mps["mean_s"], 1e-6)
            tag_m = "faster" if sp_m > 1 else "slower"
            logger.info(
                "  [%s] PyTorch MPS  vs C++: %.2fx %s  (%.3fs vs %.3fs)",
                sdf_name, sp_m, tag_m, r_mps["mean_s"], r_orig["mean_s"],
            )

        # Cross-device speedup (if multiple GPU devices available)
        if r_gpu and r_torch_cpu and "error" not in r_gpu and "error" not in r_torch_cpu:
            sp_gc = r_torch_cpu["mean_s"] / max(r_gpu["mean_s"], 1e-6)
            tag_gc = "faster" if sp_gc > 1 else "slower"
            logger.info(
                "  [%s] PyTorch CUDA vs CPU: %.2fx %s  (%.3fs vs %.3fs)",
                sdf_name, sp_gc, tag_gc, r_gpu["mean_s"], r_torch_cpu["mean_s"],
            )
        if r_mps and r_torch_cpu and "error" not in r_mps and "error" not in r_torch_cpu:
            sp_mc = r_torch_cpu["mean_s"] / max(r_mps["mean_s"], 1e-6)
            tag_mc = "faster" if sp_mc > 1 else "slower"
            logger.info(
                "  [%s] PyTorch MPS  vs CPU: %.2fx %s  (%.3fs vs %.3fs)",
                sdf_name, sp_mc, tag_mc, r_mps["mean_s"], r_torch_cpu["mean_s"],
            )
    logger.info("")

    if args.output:
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(json.dumps(results, indent=2, default=str))
        logger.info("Results saved to %s", outpath)


if __name__ == "__main__":
    main()
