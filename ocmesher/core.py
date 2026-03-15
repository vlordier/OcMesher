# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Octree-based hierarchical 3D mesher driven by signed-distance functions."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gin
import numpy as np
import trimesh
from tqdm import tqdm

from ._constants import (
    CAMERA_DATA_STRIDE,
)
from ._constants import (
    MAX_SDF_WORKERS as _MAX_SDF_WORKERS,
)
from ._constants import (
    SDF_BATCH_SIZE as _SDF_BATCH_SIZE,
)
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

__all__ = ["OcMesher"]

# Module-level numpy function cache — avoids LOAD_ATTR on the ``np`` module
# for ufuncs called repeatedly in the bounds-mask hot path and bisection loops.
_np_empty = np.empty
_np_fabs = np.fabs
_np_less_equal = np.less_equal
_np_greater_equal = np.greater_equal
_np_logical_or = np.logical_or
_np_asarray = np.asarray

# Re-export validation under private names for backwards compatibility.
from ._validation import _AXIS_NAMES  # noqa: E402, F401
from ._validation import out_of_bounds_mask as _out_of_bounds_mask_shared  # noqa: E402
from ._validation import preprocess_cameras as _preprocess_cameras  # noqa: E402
from ._validation import validate_bounds as _validate_bounds  # noqa: E402
from ._validation import validate_cameras as _validate_cameras  # noqa: E402
from ._validation import validate_kernels as _validate_kernels  # noqa: E402


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
        # Use shared camera preprocessing helper — DRY with TorchOcMesher.
        _n_cam = self.n_cameras
        inv_poses_3x4, _intrinsics, _, _ = _preprocess_cameras(cam_poses, Ks, Hs, Ws)
        _hs = np.asarray(Hs, dtype=self.np_float_type)
        _ws = np.asarray(Ws, dtype=self.np_float_type)
        # Build a (C, STRIDE) 2-D view and fill all cameras at once.
        cameras_2d = np.empty((_n_cam, CAMERA_DATA_STRIDE), dtype=self.np_float_type)
        cameras_2d[:, :12] = inv_poses_3x4.reshape(_n_cam, 12)
        cameras_2d[:, 12:21] = _intrinsics.reshape(_n_cam, 9)
        cameras_2d[:, 21] = _hs
        cameras_2d[:, 22] = _ws
        self.cameras = cameras_2d.ravel()

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

        Delegates to the shared ``out_of_bounds_mask`` helper in
        ``_validation``.  See that function for implementation details.
        """
        return _out_of_bounds_mask_shared(xyz, b_min, b_max)

    def _out_of_bounds_mask_into(self, xyz, b_min, b_max):
        """Like ``_out_of_bounds_mask`` but reuses pre-allocated buffers.

        Writes into ``self._oob_mask[:n]`` and ``self._oob_tmp[:n]``,
        avoiding a per-batch allocation of two ``(N,)`` boolean arrays.
        Returns a *view* into the pre-allocated mask buffer.
        Uses module-level cached ufunc refs to avoid ``LOAD_ATTR`` on ``np``.
        """
        n = len(xyz)
        _le = _np_less_equal
        _ge = _np_greater_equal
        _lor = _np_logical_or
        out_bound = self._oob_mask[:n]
        _tmp = self._oob_tmp[:n]
        # X-axis
        _le(xyz[:, 0], b_min[0], out=out_bound)
        _ge(xyz[:, 0], b_max[0], out=_tmp)
        _lor(out_bound, _tmp, out=out_bound)
        # Y-axis
        _le(xyz[:, 1], b_min[1], out=_tmp)
        _lor(out_bound, _tmp, out=out_bound)
        _ge(xyz[:, 1], b_max[1], out=_tmp)
        _lor(out_bound, _tmp, out=out_bound)
        # Z-axis
        _le(xyz[:, 2], b_min[2], out=_tmp)
        _lor(out_bound, _tmp, out=out_bound)
        _ge(xyz[:, 2], b_max[2], out=_tmp)
        _lor(out_bound, _tmp, out=out_bound)
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
            return _np_empty((0, n_kernels), dtype=self.sdf_np_float_type)

        _sdf_dtype = self.sdf_np_float_type
        result = out if out is not None else _np_empty((n_XYZ, n_kernels), dtype=_sdf_dtype)
        single_kernel = n_kernels == 1
        # Cache module-level constant and builtin to avoid LOAD_GLOBAL
        # on every iteration of the batch loop.
        _batch = _SDF_BATCH_SIZE
        _min = min
        _isinstance = isinstance
        _ndarray = np.ndarray
        _enclosed = self.enclosed
        _asarray = _np_asarray

        # Pre-cache bounds-mask helpers once (only used when enclosed).
        if _enclosed:
            _mask_into = self._out_of_bounds_mask_into
            _b_min = self._bounds_min_np
            _b_max = self._bounds_max_np

        if single_kernel:
            # ---- Single-kernel fast path ----
            k0 = kernels[0]
            if _enclosed:
                for i in range(0, n_XYZ, _batch):
                    end = _min(i + _batch, n_XYZ)
                    XYZ = XYZ_all[i:end]
                    n = end - i
                    raw = k0(XYZ)
                    sdf = raw if _isinstance(raw, _ndarray) else _asarray(raw)
                    if sdf.shape != (n,):
                        msg = f"kernels[0] returned shape {sdf.shape} for {n} query points; expected ({n},)"
                        raise ValueError(msg)
                    col = result[i:end, 0]
                    col[:] = sdf
                    oob = _mask_into(XYZ, _b_min, _b_max)
                    col[oob] = 1
            else:
                for i in range(0, n_XYZ, _batch):
                    end = _min(i + _batch, n_XYZ)
                    XYZ = XYZ_all[i:end]
                    n = end - i
                    raw = k0(XYZ)
                    sdf = raw if _isinstance(raw, _ndarray) else _asarray(raw)
                    if sdf.shape != (n,):
                        msg = f"kernels[0] returned shape {sdf.shape} for {n} query points; expected ({n},)"
                        raise ValueError(msg)
                    result[i:end, 0] = sdf
        else:
            # ---- Multi-kernel path: dispatch via persistent thread pool ----
            # Uses pool.submit directly instead of pool.map with a closure
            # factory, eliminating per-batch function object + closure dict
            # construction overhead.
            pool = self._get_pool(n_kernels)
            _submit = pool.submit

            for i in range(0, n_XYZ, _batch):
                end = _min(i + _batch, n_XYZ)
                XYZ = XYZ_all[i:end]
                n = end - i
                futures = [_submit(k, XYZ) for k in kernels]
                batch_slice = result[i:end]
                for k_idx, fut in enumerate(futures):
                    raw = fut.result()
                    sdf = raw if _isinstance(raw, _ndarray) else _asarray(raw)
                    if sdf.shape != (n,):
                        msg = f"kernels[{k_idx}] returned shape {sdf.shape} for {n} query points; expected ({n},)"
                        raise ValueError(msg)
                    batch_slice[:, k_idx] = sdf
                if _enclosed:
                    batch_slice[_mask_into(XYZ, _b_min, _b_max)] = 1

        return result

    def __call__(self, kernels) -> tuple[list[trimesh.Trimesh], list[np.ndarray]]:
        """Run the full coarse-to-fine meshing pipeline and return meshes."""
        _validate_kernels(kernels)
        n_elements = len(kernels)
        single_kernel = n_elements == 1

        # Cache attribute lookups for the hot loops below.
        _af = self.AF
        _sdf_af = self.sdf_AF
        _kernel_caller = self.kernel_caller
        _np_float = self.np_float_type
        _sdf_dtype = self.sdf_np_float_type
        _sdf_null = self._sdf_null
        _empty = _np_empty  # module-level cache avoids LOAD_ATTR on np

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
            # Growable buffers: allocated once, grown on demand, reused
            # across inner-loop iterations.  Eliminates per-iteration
            # np.empty + ctypes pointer construction overhead.
            _c_cap = 0
            _c_pos = None
            _c_pos_ptr = None
            _c_sdf = None
            _c_sdf_ptr = None  # cached pointer for _fine_iteration
            _c_min = None  # min-reduction buffer (multi-element only)
            while True:
                inc = _fine_group()
                if inc == 0:
                    break
                pbar.update(inc)
                n = _fine_iteration(_sdf_null)
                while n > 0:
                    if n > _c_cap:
                        _c_pos = _empty((n, 3), dtype=_np_float)
                        _c_pos_ptr = _af(_c_pos)
                        _c_sdf = _empty((n, n_elements), dtype=_sdf_dtype)
                        if single_kernel:
                            # (n, 1) C-ordered: column 0 is contiguous at base.
                            _c_sdf_ptr = _sdf_af(_c_sdf)
                        else:
                            _c_min = _empty(n, dtype=_sdf_dtype)
                            _c_sdf_ptr = _sdf_af(_c_min)
                        _c_cap = n
                    _fine_iteration_output(_c_pos_ptr)
                    _kernel_caller(kernels, _c_pos[:n], out=_c_sdf[:n])
                    if not single_kernel:
                        _c_sdf[:n].min(axis=-1, out=_c_min[:n])
                    n = _fine_iteration(_c_sdf_ptr)
        with Timer("filter visible blocks"):
            n_vis_block = self.vis_filter(self.simplify_occluded, self.visible_relax_iter)
        logger.info("visible blocks: %d", n_vis_block)

        _final_iteration = self.final_iteration
        _final_iteration2 = self.final_iteration2
        _final_iteration3 = self.final_iteration3
        with Timer("fine step"), tqdm(total=n_vis_block) as pbar:
            # np.empty: C function writes nv before any Python read.
            nv = _empty(1, dtype=np.int32)
            nv_ptr = AsInt(nv)
            # Growable buffers for the fine step loop — same pattern as
            # the coarse step: allocated once, grown on demand, reused.
            _f_cap = 0
            _f_pos = None
            _f_pos_ptr = None
            _f_sdf = None
            _f_sdf_ptr = None
            while True:
                n = _final_iteration(nv_ptr)
                if n == 0:
                    break
                if n > _f_cap:
                    _f_pos = _empty((n, 3), dtype=_np_float)
                    _f_pos_ptr = _af(_f_pos)
                    _f_sdf = _empty((n, n_elements), dtype=_sdf_dtype)
                    _f_sdf_ptr = _sdf_af(_f_sdf)
                    _f_cap = n
                _final_iteration2(_f_pos_ptr)
                _kernel_caller(kernels, _f_pos[:n], out=_f_sdf[:n])
                inc = _final_iteration3(_f_sdf_ptr)
                pbar.update(inc)
            n = self.final_iteration_occluded(nv_ptr)
            if n != 0:
                if n > _f_cap:
                    _f_pos = _empty((n, 3), dtype=_np_float)
                    _f_pos_ptr = _af(_f_pos)
                    _f_sdf = _empty((n, n_elements), dtype=_sdf_dtype)
                    _f_sdf_ptr = _sdf_af(_f_sdf)
                _final_iteration2(_f_pos_ptr)
                _kernel_caller(kernels, _f_pos[:n], out=_f_sdf[:n])
                self.final_iteration3_occluded(_f_sdf_ptr)
            # np.empty: final_remaining fills every element before use.
            nv = _empty(n_elements, dtype=np.int32)
            self.final_remaining(AsInt(nv))
        with Timer("construct mesh"):
            meshes = [None] * n_elements
            in_view_tags = [None] * n_elements
            _construct = self._construct_element_mesh
            for e in range(n_elements):
                k_e = (kernels[e],)  # tuple avoids list-slice copy
                mesh, in_view_tag = _construct(e, k_e, nv[e])
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
        _empty = _np_empty  # module-level cache avoids LOAD_ATTR on np

        # np.empty avoids zero-init since the C function fills these immediately.
        centers = _empty((num_verts, 3), dtype=_np_float)
        self.get_verts_center(e, _af(centers))
        center_sdf = _kernel_caller(k_e, centers)
        cubes = _empty((num_verts * 8, 3), dtype=_np_float)
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
        _n_cubes = num_verts * 8  # 8 cube corners per vertex; avoids len(cubes)
        _n_ke = len(k_e)
        _sdf_buf = _empty((_n_cubes, _n_ke), dtype=_sdf_dtype)
        # Cache ctypes pointer for _sdf_buf — kernel_caller returns _sdf_buf
        # (via out=), so the pointer is stable across all iterations.
        _sdf_buf_ptr = _sdf_af(_sdf_buf)
        # Pre-allocate fabs buffer reused by tolerance check to avoid
        # allocating a temporary array on each of the ~15 iterations.
        _fabs_buf = _empty((_n_cubes, _n_ke), dtype=_sdf_dtype) if check_tol else None
        _fabs = _np_fabs  # module-level cache
        for _ in range(_bisection_iters):
            _kernel_caller(k_e, cubes, out=_sdf_buf)
            _update_verts(e, _sdf_buf_ptr, center_sdf_ptr, cubes_ptr)
            # Early-exit: if all SDF residuals are below tolerance the
            # surface has been located to sufficient accuracy.
            if check_tol and _fabs(_sdf_buf, out=_fabs_buf).max() < tol:
                break
        cubes_r = _empty((num_verts * 8, 3), dtype=_np_float)
        self.get_lr_verts(e, cubes_ptr, _af(cubes_r))
        # Fused left/right SDF evaluation: single kernel_caller call instead
        # of two, halving the Python→SDF round-trip overhead.
        # Pre-allocated buffer avoids np.concatenate allocation overhead.
        lr_combined = _empty((_n_cubes * 2, 3), dtype=_np_float)
        lr_combined[:_n_cubes] = cubes
        lr_combined[_n_cubes:] = cubes_r
        lr_sdf = _kernel_caller(k_e, lr_combined)
        sdf_l = lr_sdf[:_n_cubes]
        sdf_r = lr_sdf[_n_cubes:]
        # np.empty is safe: finalize_verts writes every element before use.
        vertices = _empty((num_verts, 3), dtype=_np_float)
        self.finalize_verts(e, _sdf_af(sdf_l), _sdf_af(sdf_r), _af(vertices))

        vertices, faces = self._refine_extra_vertices(e, k_e, vertices)

        # np.empty: get_in_view_tag fills every element before Python reads.
        in_view_tag = _empty(vertices.shape[0], dtype=bool)
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
        - Module-level numpy function cache to avoid LOAD_ATTR on ``np``.
        - Known dimension variables reused for final assembly (avoids
          ``.shape[0]`` lookups).
        """
        # Bind frequently-used attributes to locals to avoid repeated
        # LOAD_ATTR lookups in the tight bisection loop (~15 iterations).
        _af = self.AF
        _sdf_af = self.sdf_AF
        _kernel_caller = self.kernel_caller
        _np_float = self.np_float_type
        _sdf_dtype = self.sdf_np_float_type
        _empty = _np_empty  # module-level cache avoids LOAD_ATTR on np

        # np.empty: construct_faces fills all 3 counts before Python reads.
        cnts = _empty(3, dtype=np.int32)
        self.construct_faces(e, _af(vertices), AsInt(cnts))
        nve, nvf, nf = cnts
        # Early-exit: when there are no extra vertices, skip all SDF
        # evaluation and bisection — just return the faces.
        if not (nve | nvf):
            faces = _empty((nf, 3), dtype=np.int32)
            self.get_faces(AsInt(faces))
            return vertices, faces
        # np.empty avoids zero-init: C functions fill all elements immediately.
        edge_vertices_c = _empty((nve, 3), dtype=_np_float)
        face_vertices_c = _empty((nvf, 3), dtype=_np_float)
        self.get_extra_verts_center(_af(edge_vertices_c), _af(face_vertices_c))
        # Fused center SDF: one kernel_caller call for edge + face centres.
        # Pre-allocated buffer replaces np.concatenate.
        # Use nve/nvf directly (already known) instead of len() calls.
        ef_centers = _empty((nve + nvf, 3), dtype=_np_float)
        ef_centers[:nve] = edge_vertices_c
        ef_centers[nve:] = face_vertices_c
        ef_center_sdf = _kernel_caller(k_e, ef_centers)
        ecenter_sdf = ef_center_sdf[:nve]
        fcenter_sdf = ef_center_sdf[nve:]
        edge_vertices_lr = _empty((nve * 2, 3), dtype=_np_float)
        face_vertices_lr = _empty((nvf * 4, 3), dtype=_np_float)
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
        # Compute sizes directly from known dimensions; avoids len() calls.
        n_edge_lr = nve * 2  # left + right per edge vertex
        n_face_lr = nvf * 4  # 4 quad corners per face vertex
        _n_bisection = n_edge_lr + n_face_lr
        # Pre-allocate combined buffer once; fill slices each iteration.
        bisection_buf = _empty((_n_bisection, 3), dtype=_np_float)
        tol = self.bisection_tol
        check_tol = tol > 0
        _bisection_iters = self.bisection_iters
        # Cache ctypes pointers for center SDF arrays (unchanged across iterations).
        ecenter_ptr = _sdf_af(ecenter_sdf)
        fcenter_ptr = _sdf_af(fcenter_sdf)
        # Pre-allocate SDF result buffer for the bisection loop.
        _n_ke = len(k_e)
        _sdf_buf = _empty((_n_bisection, _n_ke), dtype=_sdf_dtype)
        # Pre-create stable views into _sdf_buf for edge/face SDF slices.
        # kernel_caller writes into _sdf_buf via out=, so the underlying
        # buffer address is stable — cache the ctypes pointers once.
        _e_sdf_view = _sdf_buf[:n_edge_lr]
        _f_sdf_view = _sdf_buf[n_edge_lr:]
        _e_sdf_ptr = _sdf_af(_e_sdf_view)
        _f_sdf_ptr = _sdf_af(_f_sdf_view)
        # Pre-allocate fabs buffer reused by tolerance check to avoid
        # allocating a temporary array on each of the ~15 iterations.
        _fabs_buf = _empty((_n_bisection, _n_ke), dtype=_sdf_dtype) if check_tol else None
        _fabs = _np_fabs  # module-level cache
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
            if check_tol and _fabs(_sdf_buf, out=_fabs_buf).max() < tol:
                break
        edge_vertices_r = _empty((nve * 2, 3), dtype=_np_float)
        face_vertices_r = _empty((nvf * 4, 3), dtype=_np_float)
        self.get_lr_extra_verts(
            elr_ptr,
            _af(edge_vertices_r),
            flr_ptr,
            _af(face_vertices_r),
        )
        # Fused left/right SDF: 1 call instead of 4 — all 4 vertex arrays
        # written into a pre-allocated buffer (no np.concatenate overhead).
        # Sizes computed from known dimensions; avoids 4 len() calls.
        n_elr = n_edge_lr  # nve * 2
        n_err = n_edge_lr  # nve * 2 (same shape as edge_vertices_r)
        n_flr = n_face_lr  # nvf * 4
        n_frr = n_face_lr  # nvf * 4 (same shape as face_vertices_r)
        off1 = n_elr
        off2 = off1 + n_err
        off3 = off2 + n_flr
        all_lr = _empty((off3 + n_frr, 3), dtype=_np_float)
        all_lr[:off1] = edge_vertices_lr
        all_lr[off1:off2] = edge_vertices_r
        all_lr[off2:off3] = face_vertices_lr
        all_lr[off3:] = face_vertices_r
        all_lr_sdf = _kernel_caller(k_e, all_lr)
        esdf_l = all_lr_sdf[:off1]
        esdf_r = all_lr_sdf[off1:off2]
        fsdf_l = all_lr_sdf[off2:off3]
        fsdf_r = all_lr_sdf[off3:]
        edge_vertices = _empty((nve, 3), dtype=_np_float)
        face_vertices = _empty((nvf, 3), dtype=_np_float)
        self.finalize_extra_verts(
            _sdf_af(esdf_l),
            _sdf_af(esdf_r),
            _af(edge_vertices),
            _sdf_af(fsdf_l),
            _sdf_af(fsdf_r),
            _af(face_vertices),
        )
        faces = _empty((nf, 3), dtype=np.int32)
        self.get_faces(AsInt(faces))
        # Pre-allocated final vertex array avoids np.concatenate overhead.
        # Reuse nve/nvf (from construct_faces) instead of .shape[0] lookups.
        n_base = vertices.shape[0]
        off_edge = n_base + nve
        final_vertices = _empty((off_edge + nvf, 3), dtype=_np_float)
        final_vertices[:n_base] = vertices
        final_vertices[n_base:off_edge] = edge_vertices
        final_vertices[off_edge:] = face_vertices
        return final_vertices, faces
