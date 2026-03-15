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

# Default number of SDF worker threads when os.cpu_count() is unavailable.
_DEFAULT_SDF_WORKERS = 4

# Maximum number of SDF worker threads for multi-kernel evaluation.
# Defaults to available CPU count but can be capped by OCMESHER_SDF_WORKERS.
_MAX_SDF_WORKERS: int = int(os.environ.get("OCMESHER_SDF_WORKERS", str(os.cpu_count() or _DEFAULT_SDF_WORKERS)))


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

    __slots__ = (
        "AF",
        "_bounds_max_np",
        # Pre-computed bounds vectors
        "_bounds_min_np",
        "_oob_mask",
        "_oob_tmp",
        "_sdf_null",
        # Thread pool and reusable buffers
        "_sdf_pool",
        "bisection_iters",
        "bisection_tol",
        # Configuration
        "bounds",
        "cameras",
        "center",
        "coarse_count",
        "construct_faces",
        "enclosed",
        "final_iteration",
        "final_iteration2",
        "final_iteration3",
        "final_iteration3_occluded",
        "final_iteration_occluded",
        "final_remaining",
        "finalize_extra_verts",
        "finalize_verts",
        "fine_group",
        "fine_iteration",
        "fine_iteration_output",
        # Core numeric types and converters
        "float_type",
        "get_extra_verts_center",
        "get_faces",
        "get_in_view_tag",
        "get_lr_extra_verts",
        "get_lr_verts",
        "get_verts_center",
        "inv_scale",
        "inview_pixels_per_cube",
        "memory_limit_mb",
        "min_dist",
        "n_cameras",
        "np_float_type",
        # C DLL functions (set via register_func / setattr)
        "run_coarse",
        "sdf_AF",
        "sdf_float_type",
        "sdf_np_float_type",
        "simplify_occluded",
        "size",
        "update_extra_verts",
        "update_verts",
        "vis_filter",
        "visible_relax_iter",
    )

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

        # Cached null SDF pointer — avoids constructing POINTER(c_float)()
        # on every update_verts / update_extra_verts call (~7 uses per element).
        self._sdf_null = POINTER(self.sdf_float_type)()

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

    def kernel_caller(self, kernels, XYZ_all, *, out=None):
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
          When *out* is provided, writes into the caller's buffer to
          avoid per-call allocation in tight loops (bisection iterations).
        - **Single-kernel fast path** — when there is exactly one kernel,
          bypasses the pool and writes SDF values directly into the result
          column, avoiding ``_eval_kernels_batch`` dispatch overhead.
        - **Split enclosed / non-enclosed batch loops** — the ``if enclosed``
          check is hoisted outside the batch loop, eliminating a conditional
          evaluation on every batch iteration.

        Note: kernel callability is validated once by ``__call__`` at
        pipeline entry; this method skips per-call validation to avoid
        repeated checks in the 15-iteration bisection loops.

        Args:
            kernels: Sequence of SDF kernel callables.
            XYZ_all: ``(N, 3)`` array of query positions.
            out: Optional pre-allocated ``(n_XYZ, n_kernels)`` result buffer.
                 When provided, SDF values are written into this array and
                 it is returned directly, eliminating per-call allocation
                 overhead in the bisection inner loops.
        """
        n_XYZ = len(XYZ_all)
        n_kernels = len(kernels)
        if n_XYZ == 0:
            return np.zeros((0, n_kernels), dtype=self.sdf_np_float_type)

        _sdf_dtype = self.sdf_np_float_type
        result = out if out is not None else np.empty((n_XYZ, n_kernels), dtype=_sdf_dtype)
        single_kernel = n_kernels == 1
        # Cache module-level constant and builtin to avoid LOAD_GLOBAL
        # on every iteration of the batch loop.
        _batch = _SDF_BATCH_SIZE
        _min = min
        _isinstance = isinstance
        _ndarray = np.ndarray

        if single_kernel:
            # ---- Single-kernel fast path ----
            k0 = kernels[0]
            if self.enclosed:
                _mask_into = self._out_of_bounds_mask_into
                b_min = self._bounds_min_np
                b_max = self._bounds_max_np
                for i in range(0, n_XYZ, _batch):
                    end = _min(i + _batch, n_XYZ)
                    XYZ = XYZ_all[i:end]
                    n = end - i
                    raw = k0(XYZ)
                    sdf = raw if _isinstance(raw, _ndarray) else np.asarray(raw)
                    if sdf.shape != (n,):
                        msg = f"kernels[0] returned shape {sdf.shape} for {n} query points; expected ({n},)"
                        raise ValueError(msg)
                    result_col = result[i:end, 0]
                    result_col[:] = sdf
                    result_col[_mask_into(XYZ, b_min, b_max)] = 1
            else:
                for i in range(0, n_XYZ, _batch):
                    end = _min(i + _batch, n_XYZ)
                    XYZ = XYZ_all[i:end]
                    n = end - i
                    raw = k0(XYZ)
                    sdf = raw if _isinstance(raw, _ndarray) else np.asarray(raw)
                    if sdf.shape != (n,):
                        msg = f"kernels[0] returned shape {sdf.shape} for {n} query points; expected ({n},)"
                        raise ValueError(msg)
                    result[i:end, 0] = sdf
        else:
            # ---- Multi-kernel path: dispatch via persistent thread pool ----
            pool = self._get_pool(n_kernels)
            _pool_map = pool.map
            _enumerate = enumerate

            def _make_eval_one(_xyz, _n):
                def _eval_one(k_idx_kernel):
                    k_idx, kernel = k_idx_kernel
                    raw = kernel(_xyz)
                    sdf = raw if _isinstance(raw, _ndarray) else np.asarray(raw)
                    if sdf.shape != (_n,):
                        msg = f"kernels[{k_idx}] returned shape {sdf.shape} for {_n} query points; expected ({_n},)"
                        raise ValueError(msg)
                    return k_idx, sdf

                return _eval_one

            if self.enclosed:
                _mask_into = self._out_of_bounds_mask_into
                b_min = self._bounds_min_np
                b_max = self._bounds_max_np
                for i in range(0, n_XYZ, _batch):
                    end = _min(i + _batch, n_XYZ)
                    XYZ = XYZ_all[i:end]
                    for k_idx, sdf in _pool_map(_make_eval_one(XYZ, end - i), _enumerate(kernels)):
                        result[i:end, k_idx] = sdf
                    out_bound = _mask_into(XYZ, b_min, b_max)
                    result[i:end][out_bound] = 1
            else:
                for i in range(0, n_XYZ, _batch):
                    end = _min(i + _batch, n_XYZ)
                    XYZ = XYZ_all[i:end]
                    for k_idx, sdf in _pool_map(_make_eval_one(XYZ, end - i), _enumerate(kernels)):
                        result[i:end, k_idx] = sdf

        return result

    def __call__(self, kernels):
        """Run the full coarse-to-fine meshing pipeline and return meshes."""
        _validate_kernels(kernels)
        n_elements = len(kernels)
        single_kernel = n_elements == 1

        # Cache attribute lookups for the hot loops below.
        _af = self.AF
        _sdf_af = self.sdf_AF
        _kernel_caller = self.kernel_caller
        _np_float = self.np_float_type
        _sdf_null = self._sdf_null

        # octree only considering cameras, not sdf
        with Timer("coarse step part1"):
            n_blocks = self.run_coarse(
                _af(self.center),
                self.size,
                self.n_cameras,
                _af(self.cameras),
                self.inview_pixels_per_cube,
                self.inv_scale,
                self.min_dist,
                self.coarse_count,
                self.memory_limit_mb,
                n_elements,
            )
        logger.info("coarse blocks: %d", n_blocks)
        # start considering sdf
        _fine_group = self.fine_group
        _fine_iteration = self.fine_iteration
        _fine_iteration_output = self.fine_iteration_output
        with Timer("coarse step part2"), tqdm(total=n_blocks) as pbar:
            while True:
                inc = _fine_group()
                if inc == 0:
                    break
                pbar.update(inc)
                n = _fine_iteration(_sdf_null)
                while n > 0:
                    # np.empty is safe: fine_iteration_output fills every element.
                    positions = np.empty((n, 3), dtype=_np_float)
                    _fine_iteration_output(_af(positions))
                    sdf_all = _kernel_caller(kernels, positions)
                    # Single-kernel fast path: column view avoids min() reduction.
                    sdf = sdf_all[:, 0] if single_kernel else sdf_all.min(axis=-1)
                    n = _fine_iteration(_sdf_af(sdf))
        with Timer("filter visible blocks"):
            n_vis_block = self.vis_filter(self.simplify_occluded, self.visible_relax_iter)
        logger.info("visible blocks: %d", n_vis_block)

        _final_iteration = self.final_iteration
        _final_iteration2 = self.final_iteration2
        _final_iteration3 = self.final_iteration3
        with Timer("fine step"), tqdm(total=n_vis_block) as pbar:
            nv = np.zeros(1, dtype=np.int32)
            nv_ptr = AsInt(nv)
            while True:
                n = _final_iteration(nv_ptr)
                if n == 0:
                    break
                positions = np.empty((n, 3), dtype=_np_float)
                _final_iteration2(_af(positions))
                sdf = _kernel_caller(kernels, positions)
                inc = _final_iteration3(_sdf_af(sdf))
                pbar.update(inc)
            n = self.final_iteration_occluded(nv_ptr)
            if n != 0:
                positions = np.empty((n, 3), dtype=_np_float)
                _final_iteration2(_af(positions))
                sdf = _kernel_caller(kernels, positions)
                self.final_iteration3_occluded(_sdf_af(sdf))
            nv = np.zeros(n_elements, dtype=np.int32)
            self.final_remaining(AsInt(nv))
        with Timer("construct mesh"):
            meshes = [None] * n_elements
            in_view_tags = [None] * n_elements
            for e in range(n_elements):
                k_e = (kernels[e],)  # tuple avoids list-slice copy
                mesh, in_view_tag = self._construct_element_mesh(e, k_e, nv[e])
                meshes[e] = mesh
                in_view_tags[e] = in_view_tag
                logger.info(
                    "element %d: %d vertices, %d faces",
                    e,
                    mesh.vertices.shape[0],
                    mesh.faces.shape[0],
                )
        return meshes, in_view_tags

    def _construct_element_mesh(self, e, k_e, num_verts):
        """Construct mesh for a single SDF element via bisection refinement."""
        # Bind frequently-used attributes to locals to avoid repeated
        # LOAD_ATTR lookups in the tight bisection loop (~15 iterations).
        _af = self.AF
        _sdf_af = self.sdf_AF
        _kernel_caller = self.kernel_caller
        _np_float = self.np_float_type
        _sdf_null = self._sdf_null
        _sdf_dtype = self.sdf_np_float_type

        # np.empty avoids zero-init since the C function fills these immediately.
        centers = np.empty((num_verts, 3), dtype=_np_float)
        self.get_verts_center(e, _af(centers))
        center_sdf = _kernel_caller(k_e, centers)
        cubes = np.empty((num_verts * 8, 3), dtype=_np_float)
        self.update_verts(e, _sdf_null, _sdf_null, _af(cubes))
        center_sdf_ptr = _sdf_af(center_sdf)
        tol = self.bisection_tol
        check_tol = tol > 0
        _update_verts = self.update_verts
        _bisection_iters = self.bisection_iters
        # Cache ctypes pointer for cubes — the buffer is modified in-place
        # by C but never re-allocated, so the pointer remains valid.
        cubes_ptr = _af(cubes)
        # Pre-allocate SDF result buffer for the bisection loop to avoid
        # creating a fresh (N, 1) array on every iteration.
        _n_cubes = len(cubes)
        _sdf_buf = np.empty((_n_cubes, len(k_e)), dtype=_sdf_dtype)
        # Cache ctypes pointer for _sdf_buf — kernel_caller returns _sdf_buf
        # (via out=), so the pointer is stable across all iterations.
        _sdf_buf_ptr = _sdf_af(_sdf_buf)
        # Pre-allocate fabs buffer reused by tolerance check to avoid
        # allocating a temporary array on each of the ~15 iterations.
        _fabs_buf = np.empty_like(_sdf_buf) if check_tol else None
        _np_fabs = np.fabs
        for _ in range(_bisection_iters):
            _kernel_caller(k_e, cubes, out=_sdf_buf)
            _update_verts(e, _sdf_buf_ptr, center_sdf_ptr, cubes_ptr)
            # Early-exit: if all SDF residuals are below tolerance the
            # surface has been located to sufficient accuracy.
            if check_tol and _np_fabs(_sdf_buf, out=_fabs_buf).max() < tol:
                break
        cubes_r = np.empty((num_verts * 8, 3), dtype=_np_float)
        self.get_lr_verts(e, cubes_ptr, _af(cubes_r))
        # Fused left/right SDF evaluation: single kernel_caller call instead
        # of two, halving the Python→SDF round-trip overhead.
        # Pre-allocated buffer avoids np.concatenate allocation overhead.
        lr_combined = np.empty((_n_cubes + len(cubes_r), 3), dtype=_np_float)
        lr_combined[:_n_cubes] = cubes
        lr_combined[_n_cubes:] = cubes_r
        lr_sdf = _kernel_caller(k_e, lr_combined)
        sdf_l = lr_sdf[:_n_cubes]
        sdf_r = lr_sdf[_n_cubes:]
        # np.empty is safe: finalize_verts writes every element before use.
        vertices = np.empty((num_verts, 3), dtype=_np_float)
        self.finalize_verts(e, _sdf_af(sdf_l), _sdf_af(sdf_r), _af(vertices))

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
        - Attribute lookups cached as locals for the tight bisection loop.
        - Ctypes pointers cached for arrays modified in-place by C (stable
          buffer address) to avoid per-iteration pointer construction.
        - Pre-allocated SDF result buffer reused across bisection iterations
          via the ``out=`` parameter of ``kernel_caller``.
        - Early-exit when no extra vertices exist (nve == 0 and nvf == 0),
          skipping all SDF evaluation and bisection overhead.
        """
        # Bind frequently-used attributes to locals to avoid repeated
        # LOAD_ATTR lookups in the tight bisection loop (~15 iterations).
        _af = self.AF
        _sdf_af = self.sdf_AF
        _kernel_caller = self.kernel_caller
        _np_float = self.np_float_type
        _sdf_dtype = self.sdf_np_float_type

        cnts = np.zeros(3, dtype=np.int32)
        self.construct_faces(e, _af(vertices), AsInt(cnts))
        nve, nvf, nf = cnts
        # Early-exit: when there are no extra vertices, skip all SDF
        # evaluation and bisection — just return the faces.
        if not (nve | nvf):
            faces = np.empty((nf, 3), dtype=np.int32)
            self.get_faces(AsInt(faces))
            return vertices, faces
        # np.empty avoids zero-init: C functions fill all elements immediately.
        edge_vertices_c = np.empty((nve, 3), dtype=_np_float)
        face_vertices_c = np.empty((nvf, 3), dtype=_np_float)
        self.get_extra_verts_center(_af(edge_vertices_c), _af(face_vertices_c))
        # Fused center SDF: one kernel_caller call for edge + face centres.
        # Pre-allocated buffer replaces np.concatenate.
        n_edge_c = len(edge_vertices_c)
        n_face_c = len(face_vertices_c)
        ef_centers = np.empty((n_edge_c + n_face_c, 3), dtype=_np_float)
        ef_centers[:n_edge_c] = edge_vertices_c
        ef_centers[n_edge_c:] = face_vertices_c
        ef_center_sdf = _kernel_caller(k_e, ef_centers)
        ecenter_sdf = ef_center_sdf[:n_edge_c]
        fcenter_sdf = ef_center_sdf[n_edge_c:]
        edge_vertices_lr = np.empty((nve * 2, 3), dtype=_np_float)
        face_vertices_lr = np.empty((nvf * 4, 3), dtype=_np_float)
        _update_extra_verts = self.update_extra_verts
        _sdf_null = self._sdf_null
        # Cache ctypes pointers for arrays modified in-place by C —
        # buffer addresses are stable across iterations.
        elr_ptr = _af(edge_vertices_lr)
        flr_ptr = _af(face_vertices_lr)
        _update_extra_verts(
            _sdf_null,
            _sdf_null,
            _sdf_null,
            _sdf_null,
            elr_ptr,
            flr_ptr,
        )
        n_edge_lr = len(edge_vertices_lr)
        # Pre-allocate combined buffer once; fill slices each iteration.
        bisection_buf = np.empty((n_edge_lr + len(face_vertices_lr), 3), dtype=_np_float)
        tol = self.bisection_tol
        check_tol = tol > 0
        _bisection_iters = self.bisection_iters
        # Cache ctypes pointers for center SDF arrays (unchanged across iterations).
        ecenter_ptr = _sdf_af(ecenter_sdf)
        fcenter_ptr = _sdf_af(fcenter_sdf)
        # Pre-allocate SDF result buffer for the bisection loop.
        _n_bisection = len(bisection_buf)
        _n_ke = len(k_e)
        _sdf_buf = np.empty((_n_bisection, _n_ke), dtype=_sdf_dtype)
        # Pre-create stable views into _sdf_buf for edge/face SDF slices.
        # kernel_caller writes into _sdf_buf via out=, so the underlying
        # buffer address is stable — cache the ctypes pointers once.
        _e_sdf_view = _sdf_buf[:n_edge_lr]
        _f_sdf_view = _sdf_buf[n_edge_lr:]
        _e_sdf_ptr = _sdf_af(_e_sdf_view)
        _f_sdf_ptr = _sdf_af(_f_sdf_view)
        # Pre-allocate fabs buffer reused by tolerance check to avoid
        # allocating a temporary array on each of the ~15 iterations.
        _fabs_buf = np.empty_like(_sdf_buf) if check_tol else None
        _np_fabs = np.fabs
        for _ in range(_bisection_iters):
            bisection_buf[:n_edge_lr] = edge_vertices_lr
            bisection_buf[n_edge_lr:] = face_vertices_lr
            _kernel_caller(k_e, bisection_buf, out=_sdf_buf)
            _update_extra_verts(
                _e_sdf_ptr,
                _f_sdf_ptr,
                ecenter_ptr,
                fcenter_ptr,
                elr_ptr,
                flr_ptr,
            )
            if check_tol and _np_fabs(_sdf_buf, out=_fabs_buf).max() < tol:
                break
        edge_vertices_r = np.empty((nve * 2, 3), dtype=_np_float)
        face_vertices_r = np.empty((nvf * 4, 3), dtype=_np_float)
        self.get_lr_extra_verts(
            elr_ptr,
            _af(edge_vertices_r),
            flr_ptr,
            _af(face_vertices_r),
        )
        # Fused left/right SDF: 1 call instead of 4 — all 4 vertex arrays
        # written into a pre-allocated buffer (no np.concatenate overhead).
        # Pre-compute slice offsets to avoid repeated arithmetic.
        n_elr = len(edge_vertices_lr)
        n_err = len(edge_vertices_r)
        n_flr = len(face_vertices_lr)
        n_frr = len(face_vertices_r)
        off1 = n_elr
        off2 = off1 + n_err
        off3 = off2 + n_flr
        all_lr = np.empty((off3 + n_frr, 3), dtype=_np_float)
        all_lr[:off1] = edge_vertices_lr
        all_lr[off1:off2] = edge_vertices_r
        all_lr[off2:off3] = face_vertices_lr
        all_lr[off3:] = face_vertices_r
        all_lr_sdf = _kernel_caller(k_e, all_lr)
        esdf_l = all_lr_sdf[:off1]
        esdf_r = all_lr_sdf[off1:off2]
        fsdf_l = all_lr_sdf[off2:off3]
        fsdf_r = all_lr_sdf[off3:]
        edge_vertices = np.empty((nve, 3), dtype=_np_float)
        face_vertices = np.empty((nvf, 3), dtype=_np_float)
        self.finalize_extra_verts(
            _sdf_af(esdf_l),
            _sdf_af(esdf_r),
            _af(edge_vertices),
            _sdf_af(fsdf_l),
            _sdf_af(fsdf_r),
            _af(face_vertices),
        )
        faces = np.empty((nf, 3), dtype=np.int32)
        self.get_faces(AsInt(faces))
        # Pre-allocated final vertex array avoids np.concatenate overhead.
        n_base = vertices.shape[0]
        n_edge = edge_vertices.shape[0]
        final_vertices = np.empty((n_base + n_edge + face_vertices.shape[0], 3), dtype=_np_float)
        final_vertices[:n_base] = vertices
        final_vertices[n_base : n_base + n_edge] = edge_vertices
        final_vertices[n_base + n_edge :] = face_vertices
        return final_vertices, faces
