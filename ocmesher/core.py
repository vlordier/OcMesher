# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Compatibility C++-style OcMesher entrypoint backed by Rust runtime.

This module restores the historical ``ocmesher.core`` import path so existing
benchmarks and integration code continue to work while the backend stack is
consolidated around ``ocmesher_rust``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._constants import CAMERA_DATA_STRIDE, SDF_BATCH_SIZE
from ._validation import coerce_kernel_sdf, out_of_bounds_mask, validate_bounds, validate_cameras, validate_kernels
from .rust_backend import make_rust_ocmesher

__all__ = [
    "CAMERA_DATA_STRIDE",
    "OcMesher",
    "_SDF_BATCH_SIZE",
    "_validate_bounds",
    "_validate_cameras",
    "_validate_kernels",
]

# Backward-compatible symbol names used by tests and external callers.
_SDF_BATCH_SIZE = SDF_BATCH_SIZE
_validate_cameras = validate_cameras
_validate_bounds = validate_bounds
_validate_kernels = validate_kernels


class OcMesher:
    """Legacy OcMesher interface compatible with older C++ backend callsites."""

    def __init__(
        self,
        cameras,
        bounds,
        pixels_per_cube: int = 8,
        inv_scale: int = 10,
        min_dist: int = 1,
        memory_limit_mb: int = 1000,
        bisection_iters: int = 15,
        bisection_tol: float = 0.0,
        enclosed: bool = True,
        simplify_occluded: bool = True,
        visible_relax_iter: int = 2,
        coarse_count: int = 500000,
    ):
        cam_poses, ks, hs, ws = validate_cameras(cameras)
        self.cameras = (cam_poses, ks, hs, ws)
        self.bounds = validate_bounds(bounds)
        self._bounds_min_np = np.array([self.bounds[0], self.bounds[2], self.bounds[4]], dtype=np.float64)
        self._bounds_max_np = np.array([self.bounds[1], self.bounds[3], self.bounds[5]], dtype=np.float64)

        self.pixels_per_cube = pixels_per_cube
        self.inv_scale = inv_scale
        self.min_dist = min_dist
        self.memory_limit_mb = memory_limit_mb
        self.bisection_iters = bisection_iters
        self.bisection_tol = bisection_tol
        self.enclosed = enclosed
        self.simplify_occluded = simplify_occluded
        self.visible_relax_iter = visible_relax_iter
        self.coarse_count = coarse_count

        self.sdf_np_float_type = np.float32
        self._backend_mesher = None

    def kernel_caller(self, kernels, xyz_all, out: np.ndarray | None = None) -> np.ndarray:
        """Evaluate one or more SDF kernels over query points in batches."""
        validate_kernels(kernels)
        xyz_all = np.asarray(xyz_all, dtype=np.float64)
        if xyz_all.ndim != 2 or xyz_all.shape[1] != 3:
            msg = f"xyz must have shape (N, 3), got {xyz_all.shape}"
            raise ValueError(msg)

        n_points = int(xyz_all.shape[0])
        n_kernels = len(kernels)
        if out is None:
            out = np.empty((n_points, n_kernels), dtype=self.sdf_np_float_type)
        elif out.shape != (n_points, n_kernels):
            msg = f"out must have shape {(n_points, n_kernels)}, got {out.shape}"
            raise ValueError(msg)

        if n_points == 0:
            return out

        for start in range(0, n_points, _SDF_BATCH_SIZE):
            end = min(start + _SDF_BATCH_SIZE, n_points)
            xyz = xyz_all[start:end]
            oob = None
            if self.enclosed:
                oob = out_of_bounds_mask(xyz, self._bounds_min_np, self._bounds_max_np)

            for kernel_idx, kernel in enumerate(kernels):
                sdf = coerce_kernel_sdf(kernel(xyz), len(xyz), f"kernels[{kernel_idx}]")
                sdf = np.asarray(sdf, dtype=self.sdf_np_float_type)
                if oob is not None:
                    sdf = sdf.copy()
                    sdf[oob] = 1
                out[start:end, kernel_idx] = sdf

        return out

    def __call__(self, kernels):
        validate_kernels(kernels)
        if self._backend_mesher is None:
            self._backend_mesher = make_rust_ocmesher(
                self.cameras,
                self.bounds,
                pixels_per_cube=self.pixels_per_cube,
                inv_scale=self.inv_scale,
                min_dist=self.min_dist,
                memory_limit_mb=self.memory_limit_mb,
                bisection_iters=self.bisection_iters,
                bisection_tol=self.bisection_tol,
                enclosed=self.enclosed,
                simplify_occluded=self.simplify_occluded,
                visible_relax_iter=self.visible_relax_iter,
                coarse_count=self.coarse_count,
                device="cpu",
            )
        return self._backend_mesher(kernels)
