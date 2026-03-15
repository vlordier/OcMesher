# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Octree-based hierarchical 3D mesher driven by signed-distance functions."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gin
import numpy as np
import trimesh
from tqdm import tqdm

from .utils.interface import (
    POINTER,
    AsBool,
    AsDouble,
    AsFloat,
    AsInt,
    c_bool,
    c_double,
    c_float,
    c_int32,
    load_cdll,
    register_func,
)
from .utils.timer import Timer

logger = logging.getLogger(__name__)

CAMERA_DATA_STRIDE = 23

# Maximum number of SDF query points evaluated in a single vectorised batch.
# Keeping this below ~10M avoids exhausting RAM on large octrees.
_SDF_BATCH_SIZE = 10_000_000

# Maximum number of SDF worker threads for multi-kernel evaluation.
# Defaults to available CPU count but can be capped by OCMESHER_SDF_WORKERS.
_MAX_SDF_WORKERS: int = int(os.environ.get("OCMESHER_SDF_WORKERS", str(os.cpu_count() or 4)))


def _validate_cameras(cameras):
    """Validate and normalise camera tuple, returning (cam_poses, Ks, Hs, Ws).

    Args:
        cameras: Tuple of (cam_poses, Ks, Hs, Ws).

    Returns:
        Tuple of numpy arrays (cam_poses, Ks, Hs, Ws).

    Raises:
        ValueError: If cameras structure, lengths, or shapes are invalid.
    """
    if not isinstance(cameras, (tuple, list)) or len(cameras) != 4:  # noqa: PLR2004
        msg = "cameras must be a tuple/list of (cam_poses, Ks, Hs, Ws)"
        raise ValueError(msg)
    cam_poses, Ks, Hs, Ws = cameras
    n = len(cam_poses)
    if len(Ks) != n or len(Hs) != n or len(Ws) != n:
        msg = f"Camera arrays must all have the same length, got poses={len(cam_poses)}, Ks={len(Ks)}, Hs={len(Hs)}, Ws={len(Ws)}"
        raise ValueError(msg)
    if n == 0:
        msg = "At least one camera is required"
        raise ValueError(msg)
    cam_poses = [np.asarray(p, dtype=np.float64) for p in cam_poses]
    Ks = [np.asarray(k, dtype=np.float64) for k in Ks]
    for i, pose in enumerate(cam_poses):
        if pose.shape != (4, 4):
            msg = f"cam_poses[{i}] must be a 4x4 matrix, got shape {pose.shape}"
            raise ValueError(msg)
    for i, k in enumerate(Ks):
        if k.shape != (3, 3):
            msg = f"Ks[{i}] must be a 3x3 matrix, got shape {k.shape}"
            raise ValueError(msg)
    return cam_poses, Ks, Hs, Ws


def _validate_bounds(bounds):
    """Validate bounds array: 6 elements ``[x_min, x_max, y_min, y_max, z_min, z_max]``.

    Args:
        bounds: Sequence of 6 numeric values.

    Returns:
        numpy array of shape ``(6,)`` with dtype ``float64``.

    Raises:
        ValueError: If bounds has wrong length or min >= max for any axis.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (6,):
        msg = f"bounds must have 6 elements [x_min, x_max, y_min, y_max, z_min, z_max], got shape {bounds.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(bounds)):
        msg = "bounds must contain only finite values (no NaN or Inf)"
        raise ValueError(msg)
    for axis, name in enumerate(["x", "y", "z"]):
        if bounds[axis * 2] >= bounds[axis * 2 + 1]:
            msg = f"bounds {name}_min ({bounds[axis * 2]}) must be less than {name}_max ({bounds[axis * 2 + 1]})"
            raise ValueError(msg)
    return bounds


def _validate_kernels(kernels):
    """Validate that *kernels* is a non-empty sequence of callables.

    Args:
        kernels: Sequence of SDF kernel functions.

    Raises:
        ValueError: If kernels is empty or not a list/tuple.
        TypeError: If any element is not callable.
    """
    if not isinstance(kernels, (list, tuple)) or len(kernels) == 0:
        msg = "kernels must be a non-empty list/tuple of callable SDF functions"
        raise ValueError(msg)
    for i, k in enumerate(kernels):
        if not callable(k):
            msg = f"kernels[{i}] must be callable, got {type(k).__name__}"
            raise TypeError(msg)


@gin.configurable
class OcMesher:
    """Octree-based mesher that extracts surfaces from SDF kernels."""

    def __init__(
        self,
        cameras,
        bounds,
        pixels_per_cube=8,
        inv_scale=10,
        min_dist=1,
        memory_limit_mb=1000,
        bisection_iters=15,
        bisection_tol=0.0,
        enclosed=True,
        simplify_occluded=True,
        visible_relax_iter=2,
        coarse_count=500000,
    ):
        """Initialise the mesher with camera intrinsics and bounds.

        ``bisection_tol``: when ``> 0``, bisection loops exit early once the
        max absolute SDF residual drops below this value (default ``0``:
        always run all ``bisection_iters`` iterations).
        """
        cam_poses, Ks, Hs, Ws = _validate_cameras(cameras)
        bounds = _validate_bounds(bounds)

        dll = load_cdll(str(Path(__file__).parent.resolve() / "lib" / "core.so"))
        self.float_type = c_double
        self.np_float_type = np.float64
        self.AF = AsDouble
        self.sdf_float_type = c_float
        self.sdf_np_float_type = np.float32
        self.sdf_AF = AsFloat
        self.bounds = bounds
        self.memory_limit_mb = memory_limit_mb

        self.n_cameras = len(cam_poses)
        self.cameras = np.zeros(CAMERA_DATA_STRIDE * self.n_cameras, dtype=self.np_float_type)
        # Vectorised camera packing: batch all cameras in one numpy pass
        # instead of per-camera Python loop + np.concatenate + .astype().
        _inv_poses = np.linalg.inv(np.asarray(cam_poses, dtype=np.float64))  # (C, 4, 4)
        _intrinsics = np.asarray(Ks, dtype=np.float64)  # (C, 3, 3)
        for i in range(self.n_cameras):
            offset = CAMERA_DATA_STRIDE * i
            self.cameras[offset : offset + 12] = _inv_poses[i, :3, :4].ravel()
            self.cameras[offset + 12 : offset + 21] = _intrinsics[i].ravel()
            self.cameras[offset + 21] = Hs[i]
            self.cameras[offset + 22] = Ws[i]

        self.inview_pixels_per_cube = self.np_float_type(pixels_per_cube)
        self.inv_scale = self.np_float_type(inv_scale)
        self.min_dist = self.np_float_type(min_dist)

        self.center = np.array(
            [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2],
            self.np_float_type,
        )
        self.size = self.np_float_type(max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 1.1)

        self.bisection_iters = bisection_iters
        self.bisection_tol = float(bisection_tol)
        self.enclosed = enclosed
        self.simplify_occluded = simplify_occluded
        self.visible_relax_iter = visible_relax_iter
        self.coarse_count = coarse_count

        # Pre-compute bound vectors for vectorised out-of-bounds masking.
        # Avoids per-axis Python loop in kernel_caller (6 temps → 2 broadcasts).
        self._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        self._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)

        # Persistent thread pool for multi-kernel SDF evaluation.
        # Avoids the overhead of constructing + tearing down a ThreadPool per
        # batch (each batch in the bisection inner loop).  Lazy-initialised:
        # only created when more than one kernel is actually used.
        self._sdf_pool: ThreadPoolExecutor | None = None

        # Reusable boolean buffers for out-of-bounds masking.
        # Pre-allocated to _SDF_BATCH_SIZE (the maximum batch slice), so
        # _out_of_bounds_mask_into() can write directly into these instead
        # of allocating fresh (N,) arrays on every batch.
        self._oob_mask: np.ndarray = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        self._oob_tmp: np.ndarray = np.empty(_SDF_BATCH_SIZE, dtype=bool)

        register_func(
            self,
            dll,
            "run_coarse",
            [
                POINTER(self.float_type),
                self.float_type,
                c_int32,
                POINTER(self.float_type),
                self.float_type,
                self.float_type,
                self.float_type,
                c_int32,
                c_int32,
                c_int32,
            ],
            c_int32,
        )
        register_func(self, dll, "fine_group", [], c_int32)
        register_func(self, dll, "fine_iteration", [POINTER(self.sdf_float_type)], c_int32)
        register_func(self, dll, "fine_iteration_output", [POINTER(self.float_type)])
        register_func(self, dll, "vis_filter", [c_bool, c_int32], c_int32)
        register_func(self, dll, "final_iteration", [], c_int32)
        register_func(self, dll, "final_iteration_occluded", [], c_int32)
        register_func(self, dll, "final_iteration2", [POINTER(self.float_type)])
        register_func(self, dll, "final_iteration3", [POINTER(self.sdf_float_type)], c_int32)
        register_func(self, dll, "final_iteration3_occluded", [POINTER(self.sdf_float_type)])
        register_func(self, dll, "final_remaining", [POINTER(c_int32)])
        register_func(self, dll, "get_verts_center", [c_int32, POINTER(self.float_type)])
        register_func(
            self,
            dll,
            "update_verts",
            [c_int32, POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type)],
        )
        register_func(self, dll, "get_lr_verts", [c_int32, POINTER(self.float_type), POINTER(self.float_type)])
        register_func(
            self,
            dll,
            "finalize_verts",
            [c_int32, POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type)],
        )
        register_func(self, dll, "construct_faces", [c_int32, POINTER(self.float_type), POINTER(c_int32)])
        register_func(self, dll, "get_extra_verts_center", [POINTER(self.float_type), POINTER(self.float_type)])
        register_func(
            self,
            dll,
            "update_extra_verts",
            [
                POINTER(self.sdf_float_type),
                POINTER(self.sdf_float_type),
                POINTER(self.sdf_float_type),
                POINTER(self.sdf_float_type),
                POINTER(self.float_type),
                POINTER(self.float_type),
            ],
        )
        register_func(
            self,
            dll,
            "get_lr_extra_verts",
            [POINTER(self.float_type), POINTER(self.float_type), POINTER(self.float_type), POINTER(self.float_type)],
        )
        register_func(
            self,
            dll,
            "finalize_extra_verts",
            [
                POINTER(self.sdf_float_type),
                POINTER(self.sdf_float_type),
                POINTER(self.float_type),
                POINTER(self.sdf_float_type),
                POINTER(self.sdf_float_type),
                POINTER(self.float_type),
            ],
        )
        register_func(self, dll, "get_faces", [POINTER(c_int32)])
        register_func(self, dll, "get_in_view_tag", [c_int32, POINTER(c_bool)])

    def __repr__(self) -> str:
        """Return a concise developer-friendly description of the mesher."""
        return (
            f"OcMesher("
            f"n_cameras={self.n_cameras}, "
            f"bounds={self.bounds.tolist()}, "
            f"bisection_iters={self.bisection_iters}, "
            f"bisection_tol={self.bisection_tol}, "
            f"enclosed={self.enclosed})"
        )

    def __enter__(self):
        """Support ``with OcMesher(...) as m:`` usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shut down the persistent thread pool on context-manager exit."""
        self._shutdown_pool()
        return False

    def _shutdown_pool(self):
        """Shut down the persistent SDF thread pool if it exists."""
        if self._sdf_pool is not None:
            self._sdf_pool.shutdown(wait=False)
            self._sdf_pool = None

    def _get_pool(self, n_kernels: int) -> ThreadPoolExecutor:
        """Return the persistent SDF thread pool, creating it lazily.

        The pool is sized to ``min(n_kernels, _MAX_SDF_WORKERS)`` and
        reused across all ``kernel_caller`` invocations so that thread
        creation overhead is incurred only once per mesher lifetime.
        """
        if self._sdf_pool is None:
            self._sdf_pool = ThreadPoolExecutor(max_workers=min(n_kernels, _MAX_SDF_WORKERS))
        return self._sdf_pool

    @staticmethod
    def _out_of_bounds_mask(xyz, b_min, b_max):
        """Build a 1-D boolean mask indicating out-of-bounds points.

        Uses per-axis ufunc calls with ``out=`` to accumulate into two
        ``(N,)`` boolean buffers, avoiding the ``(N, 3)`` temporaries that
        ``np.any((XYZ <= lo) | (XYZ >= hi), axis=1)`` would allocate.

        The three-axis loop is fully unrolled to eliminate Python iteration
        overhead in the hot path.

        Returns:
            Boolean array of shape ``(N,)`` where ``True`` means the point
            is on or outside the boundary.
        """
        n = len(xyz)
        _tmp = np.empty(n, dtype=bool)
        out_bound = np.empty(n, dtype=bool)
        # X-axis
        np.less_equal(xyz[:, 0], b_min[0], out=out_bound)
        np.greater_equal(xyz[:, 0], b_max[0], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        # Y-axis
        np.less_equal(xyz[:, 1], b_min[1], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        np.greater_equal(xyz[:, 1], b_max[1], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        # Z-axis
        np.less_equal(xyz[:, 2], b_min[2], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        np.greater_equal(xyz[:, 2], b_max[2], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        return out_bound

    def _out_of_bounds_mask_into(self, xyz, b_min, b_max):
        """Like ``_out_of_bounds_mask`` but reuses pre-allocated buffers.

        Writes into ``self._oob_mask[:n]`` and ``self._oob_tmp[:n]``,
        avoiding a per-batch allocation of two ``(N,)`` boolean arrays.
        Returns a *view* into the pre-allocated mask buffer.
        """
        n = len(xyz)
        out_bound = self._oob_mask[:n]
        _tmp = self._oob_tmp[:n]
        # X-axis
        np.less_equal(xyz[:, 0], b_min[0], out=out_bound)
        np.greater_equal(xyz[:, 0], b_max[0], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        # Y-axis
        np.less_equal(xyz[:, 1], b_min[1], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        np.greater_equal(xyz[:, 1], b_max[1], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        # Z-axis
        np.less_equal(xyz[:, 2], b_min[2], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        np.greater_equal(xyz[:, 2], b_max[2], out=_tmp)
        np.logical_or(out_bound, _tmp, out=out_bound)
        return out_bound

    def kernel_caller(self, kernels, XYZ_all):
        """Evaluate SDF *kernels* at the given *XYZ_all* positions.

        Optimisations over the naïve implementation:

        - **In-place 1-D bounds mask** — per-axis ``np.less_equal`` /
          ``np.greater_equal`` with ``out=`` accumulates a single ``(N,)``
          boolean mask, avoiding the two ``(N, 3)`` temporary arrays that
          ``np.any((XYZ <= lo) | (XYZ >= hi), axis=1)`` would allocate.
          Mask buffers are pre-allocated once and reused across batches.
        - **Persistent release-GIL thread pool** — when ``len(kernels) > 1``
          each SDF kernel is dispatched to a persistent
          ``ThreadPoolExecutor`` that is shared across all
          ``kernel_caller`` invocations (lazy-initialised on first use).
        - **Pre-allocated output buffer** filled in-place (no list
          accumulation, no ``np.stack`` / ``np.concatenate`` per batch).
        - **Single-kernel fast path** — when there is exactly one kernel,
          bypasses the pool and writes SDF values directly into the result
          column, avoiding ``_eval_kernels_batch`` dispatch overhead.
        """
        for kernel in kernels:
            if not callable(kernel):
                msg = f"Each kernel must be callable, got {type(kernel).__name__}"
                raise TypeError(msg)
        n_XYZ = len(XYZ_all)
        n_kernels = len(kernels)
        if n_XYZ == 0:
            return np.zeros((0, n_kernels), dtype=self.sdf_np_float_type)

        result = np.empty((n_XYZ, n_kernels), dtype=self.sdf_np_float_type)
        b_min = self._bounds_min_np
        b_max = self._bounds_max_np
        enclosed = self.enclosed
        single_kernel = n_kernels == 1

        for i in range(0, n_XYZ, _SDF_BATCH_SIZE):
            end = min(i + _SDF_BATCH_SIZE, n_XYZ)
            XYZ = XYZ_all[i:end]
            n = end - i

            # Reuse pre-allocated mask buffers; avoid per-batch allocation.
            out_bound = self._out_of_bounds_mask_into(XYZ, b_min, b_max) if enclosed else None

            if single_kernel:
                # Single-kernel fast path: write SDF directly, no pool overhead.
                sdf = np.asarray(kernels[0](XYZ))
                if sdf.shape != (n,):
                    msg = f"kernels[0] returned shape {sdf.shape} for {n} query points; expected ({n},)"
                    raise ValueError(msg)
                result_col = result[i:end, 0]
                result_col[:] = sdf
                if out_bound is not None:
                    result_col[out_bound] = 1
            else:
                self._eval_kernels_batch(kernels, XYZ, n, out_bound, result, i, end)

        return result

    def _eval_kernels_batch(self, kernels, xyz, batch_size, out_bound, result, start, end):
        """Evaluate all *kernels* on *xyz* and write into *result[start:end]*.

        Uses the persistent ``ThreadPoolExecutor`` to dispatch each kernel
        to a separate thread.  ctypes / C-extension SDF calls release the
        GIL, enabling true multi-core parallelism.

        SDF values are written directly into the pre-allocated *result*
        buffer; out-of-bounds clamping is applied in-place on the result
        slice, avoiding an intermediate copy.
        """
        n_kernels = len(kernels)

        def _eval_one(k_idx_kernel):
            k_idx, kernel = k_idx_kernel
            sdf = np.asarray(kernel(xyz))
            if sdf.shape != (batch_size,):
                msg = (
                    f"kernels[{k_idx}] returned shape {sdf.shape} "
                    f"for {batch_size} query points; expected ({batch_size},)"
                )
                raise ValueError(msg)
            return k_idx, sdf

        # Multi-kernel: dispatch via persistent thread pool.
        pool = self._get_pool(n_kernels)
        for k_idx, sdf in pool.map(_eval_one, enumerate(kernels)):
            result[start:end, k_idx] = sdf
        if out_bound is not None:
            result_slice = result[start:end]
            result_slice[out_bound] = 1

    def __call__(self, kernels):
        """Run the full coarse-to-fine meshing pipeline and return meshes."""
        _validate_kernels(kernels)
        n_elements = len(kernels)
        single_kernel = n_elements == 1
        # octree only considering cameras, not sdf
        with Timer("coarse step part1"):
            n_blocks = self.run_coarse(
                self.AF(self.center),
                self.size,
                self.n_cameras,
                self.AF(self.cameras),
                self.inview_pixels_per_cube,
                self.inv_scale,
                self.min_dist,
                self.coarse_count,
                self.memory_limit_mb,
                n_elements,
            )
        logger.info("coarse blocks: %d", n_blocks)
        # start considering sdf
        with Timer("coarse step part2"), tqdm(total=n_blocks) as pbar:
            while True:
                inc = self.fine_group()
                if inc == 0:
                    break
                pbar.update(inc)
                n = self.fine_iteration(POINTER(self.sdf_float_type)())
                while n > 0:
                    # np.empty is safe: fine_iteration_output fills every element.
                    positions = np.empty((n, 3), dtype=self.np_float_type)
                    self.fine_iteration_output(self.AF(positions))
                    sdf_all = self.kernel_caller(kernels, positions)
                    # Single-kernel fast path: column view avoids min() reduction.
                    sdf = sdf_all[:, 0] if single_kernel else sdf_all.min(axis=-1)
                    n = self.fine_iteration(self.sdf_AF(sdf))
        with Timer("filter visible blocks"):
            n_vis_block = self.vis_filter(self.simplify_occluded, self.visible_relax_iter)
        logger.info("visible blocks: %d", n_vis_block)

        with Timer("fine step"), tqdm(total=n_vis_block) as pbar:
            nv = np.zeros(1, dtype=np.int32)
            while True:
                n = self.final_iteration(AsInt(nv))
                if n == 0:
                    break
                positions = np.empty((n, 3), dtype=self.np_float_type)
                self.final_iteration2(self.AF(positions))
                sdf = self.kernel_caller(kernels, positions)
                inc = self.final_iteration3(self.sdf_AF(sdf))
                pbar.update(inc)
            n = self.final_iteration_occluded(AsInt(nv))
            if n != 0:
                positions = np.empty((n, 3), dtype=self.np_float_type)
                self.final_iteration2(self.AF(positions))
                sdf = self.kernel_caller(kernels, positions)
                self.final_iteration3_occluded(self.sdf_AF(sdf))
            nv = np.zeros(n_elements, dtype=np.int32)
            self.final_remaining(AsInt(nv))
            del positions, sdf

        with Timer("construct mesh"):
            meshes = []
            in_view_tags = []
            for e in range(n_elements):
                mesh, in_view_tag = self._construct_element_mesh(e, kernels[e : e + 1], nv[e])
                meshes.append(mesh)
                in_view_tags.append(in_view_tag)
                logger.info(
                    "element %d: %d vertices, %d faces",
                    e,
                    mesh.vertices.shape[0],
                    mesh.faces.shape[0],
                )
        return meshes, in_view_tags

    def _construct_element_mesh(self, e, k_e, num_verts):
        """Construct mesh for a single SDF element via bisection refinement."""
        # np.empty avoids zero-init since the C function fills these immediately.
        centers = np.empty((num_verts, 3), dtype=self.np_float_type)
        self.get_verts_center(e, self.AF(centers))
        center_sdf = self.kernel_caller(k_e, centers)
        cubes = np.empty((num_verts * 8, 3), dtype=self.np_float_type)
        self.update_verts(e, POINTER(self.sdf_float_type)(), POINTER(self.sdf_float_type)(), self.AF(cubes))
        center_sdf_ptr = self.sdf_AF(center_sdf)
        tol = self.bisection_tol
        check_tol = tol > 0
        for _ in tqdm(range(self.bisection_iters)):
            sdf = self.kernel_caller(k_e, cubes)
            self.update_verts(e, self.sdf_AF(sdf), center_sdf_ptr, self.AF(cubes))
            # Early-exit: if all SDF residuals are below tolerance the
            # surface has been located to sufficient accuracy.
            if check_tol and np.max(np.abs(sdf)) < tol:
                break
        cubes_r = np.empty((num_verts * 8, 3), dtype=self.np_float_type)
        self.get_lr_verts(e, self.AF(cubes), self.AF(cubes_r))
        # Fused left/right SDF evaluation: single kernel_caller call instead
        # of two, halving the Python→SDF round-trip overhead.
        # Pre-allocated buffer avoids np.concatenate allocation overhead.
        n_cubes = len(cubes)
        lr_combined = np.empty((n_cubes + len(cubes_r), 3), dtype=self.np_float_type)
        lr_combined[:n_cubes] = cubes
        lr_combined[n_cubes:] = cubes_r
        lr_sdf = self.kernel_caller(k_e, lr_combined)
        sdf_l = lr_sdf[:n_cubes]
        sdf_r = lr_sdf[n_cubes:]
        del cubes, cubes_r, centers, center_sdf, lr_combined, lr_sdf
        # np.empty is safe: finalize_verts writes every element before use.
        vertices = np.empty((num_verts, 3), dtype=self.np_float_type)
        self.finalize_verts(e, self.sdf_AF(sdf_l), self.sdf_AF(sdf_r), self.AF(vertices))
        del sdf_l, sdf_r

        vertices, faces = self._refine_extra_vertices(e, k_e, vertices)

        in_view_tag = np.zeros(vertices.shape[0], dtype=bool)
        self.get_in_view_tag(e, AsBool(in_view_tag))
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), in_view_tag

    def _refine_extra_vertices(self, e, k_e, vertices):
        """Compute edge/face extra vertices and assemble final faces.

        Optimisations:
        - Fused edge + face SDF evaluation per bisection iteration: a single
          ``kernel_caller`` call replaces two, halving Python→SDF round trips
          in the inner loop (30 calls → 15 for the default 15 iterations).
        - Pre-allocated combined buffer outside the bisection loop: each
          iteration fills slices instead of allocating via ``np.concatenate``,
          eliminating ~15 allocations + copies per element.
        - Early-exit convergence: when ``bisection_tol > 0``, the loop
          terminates as soon as the maximum absolute SDF residual drops
          below the tolerance, skipping unnecessary iterations.
        - Fused left/right SDF evaluation after bisection: 4 calls → 1.
        - All ``np.concatenate`` calls replaced with pre-allocated
          slice-fill to avoid allocation overhead in hot paths.
        """
        cnts = np.zeros(3, dtype=np.int32)
        self.construct_faces(e, self.AF(vertices), AsInt(cnts))
        nve, nvf, nf = cnts
        # np.empty avoids zero-init: C functions fill all elements immediately.
        edge_vertices_c = np.empty((nve, 3), dtype=self.np_float_type)
        face_vertices_c = np.empty((nvf, 3), dtype=self.np_float_type)
        self.get_extra_verts_center(self.AF(edge_vertices_c), self.AF(face_vertices_c))
        # Fused center SDF: one kernel_caller call for edge + face centres.
        # Pre-allocated buffer replaces np.concatenate.
        n_edge_c = len(edge_vertices_c)
        n_face_c = len(face_vertices_c)
        ef_centers = np.empty((n_edge_c + n_face_c, 3), dtype=self.np_float_type)
        ef_centers[:n_edge_c] = edge_vertices_c
        ef_centers[n_edge_c:] = face_vertices_c
        ef_center_sdf = self.kernel_caller(k_e, ef_centers)
        ecenter_sdf = ef_center_sdf[:n_edge_c]
        fcenter_sdf = ef_center_sdf[n_edge_c:]
        del ef_centers, ef_center_sdf
        edge_vertices_lr = np.empty((nve * 2, 3), dtype=self.np_float_type)
        face_vertices_lr = np.empty((nvf * 4, 3), dtype=self.np_float_type)
        self.update_extra_verts(
            POINTER(self.sdf_float_type)(),
            POINTER(self.sdf_float_type)(),
            POINTER(self.sdf_float_type)(),
            POINTER(self.sdf_float_type)(),
            self.AF(edge_vertices_lr),
            self.AF(face_vertices_lr),
        )
        n_edge_lr = len(edge_vertices_lr)
        # Pre-allocate combined buffer once; fill slices each iteration.
        bisection_buf = np.empty((n_edge_lr + len(face_vertices_lr), 3), dtype=self.np_float_type)
        tol = self.bisection_tol
        check_tol = tol > 0
        for _ in range(self.bisection_iters):
            bisection_buf[:n_edge_lr] = edge_vertices_lr
            bisection_buf[n_edge_lr:] = face_vertices_lr
            ef_sdf = self.kernel_caller(k_e, bisection_buf)
            e_sdf = ef_sdf[:n_edge_lr]
            f_sdf = ef_sdf[n_edge_lr:]
            self.update_extra_verts(
                self.sdf_AF(e_sdf),
                self.sdf_AF(f_sdf),
                self.sdf_AF(ecenter_sdf),
                self.sdf_AF(fcenter_sdf),
                self.AF(edge_vertices_lr),
                self.AF(face_vertices_lr),
            )
            if check_tol and np.max(np.abs(ef_sdf)) < tol:
                break
        del edge_vertices_c, face_vertices_c, ecenter_sdf, fcenter_sdf, bisection_buf
        edge_vertices_r = np.empty((nve * 2, 3), dtype=self.np_float_type)
        face_vertices_r = np.empty((nvf * 4, 3), dtype=self.np_float_type)
        self.get_lr_extra_verts(
            self.AF(edge_vertices_lr),
            self.AF(edge_vertices_r),
            self.AF(face_vertices_lr),
            self.AF(face_vertices_r),
        )
        # Fused left/right SDF: 1 call instead of 4 — all 4 vertex arrays
        # written into a pre-allocated buffer (no np.concatenate overhead).
        n_elr = len(edge_vertices_lr)
        n_err = len(edge_vertices_r)
        n_flr = len(face_vertices_lr)
        n_frr = len(face_vertices_r)
        all_lr = np.empty((n_elr + n_err + n_flr + n_frr, 3), dtype=self.np_float_type)
        all_lr[:n_elr] = edge_vertices_lr
        all_lr[n_elr : n_elr + n_err] = edge_vertices_r
        all_lr[n_elr + n_err : n_elr + n_err + n_flr] = face_vertices_lr
        all_lr[n_elr + n_err + n_flr :] = face_vertices_r
        all_lr_sdf = self.kernel_caller(k_e, all_lr)
        esdf_l = all_lr_sdf[:n_elr]
        esdf_r = all_lr_sdf[n_elr : 2 * n_elr]
        fsdf_l = all_lr_sdf[2 * n_elr : 2 * n_elr + n_flr]
        fsdf_r = all_lr_sdf[2 * n_elr + n_flr :]
        del edge_vertices_lr, edge_vertices_r, face_vertices_lr, face_vertices_r, all_lr, all_lr_sdf
        edge_vertices = np.empty((nve, 3), dtype=self.np_float_type)
        face_vertices = np.empty((nvf, 3), dtype=self.np_float_type)
        self.finalize_extra_verts(
            self.sdf_AF(esdf_l),
            self.sdf_AF(esdf_r),
            self.AF(edge_vertices),
            self.sdf_AF(fsdf_l),
            self.sdf_AF(fsdf_r),
            self.AF(face_vertices),
        )
        del esdf_l, esdf_r, fsdf_l, fsdf_r
        faces = np.empty((nf, 3), dtype=np.int32)
        self.get_faces(AsInt(faces))
        # Pre-allocated final vertex array avoids np.concatenate overhead.
        n_base = vertices.shape[0]
        n_edge = edge_vertices.shape[0]
        final_vertices = np.empty((n_base + n_edge + face_vertices.shape[0], 3), dtype=self.np_float_type)
        final_vertices[:n_base] = vertices
        final_vertices[n_base : n_base + n_edge] = edge_vertices
        final_vertices[n_base + n_edge :] = face_vertices
        return final_vertices, faces
