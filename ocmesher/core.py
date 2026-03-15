# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""OcMesher: octree-based mesh extraction from signed distance functions."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import gin
import numpy as np
from tqdm import tqdm

from .dll import CoreDLL
from .meshing import construct_element_mesh
from .sdf import evaluate_sdfs
from .types import (
    CAMERA_DATA_STRIDE,  # noqa: F401 - re-exported for backward compat
    Bounds,
    CameraSet,
    MesherConfig,
)
from .types import SDF_BATCH_SIZE as _SDF_BATCH_SIZE
from .utils.interface import AC, POINTER, as_double, as_float, as_int, c_float
from .utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import trimesh
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone validation helpers (backward-compatible with the legacy API)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@gin.configurable
class OcMesher:
    """Octree-based mesher that extracts surfaces from signed distance functions.

    Accepts camera data and axis-aligned bounds, then uses an octree to
    adaptively evaluate SDF kernels and extract a triangle mesh via
    bisection-based surface finding.

    Args:
        cameras: Camera parameters as a ``CameraSet`` or legacy tuple
            ``(poses, intrinsics, heights, widths)``.
        bounds: Axis-aligned bounding box as a ``Bounds`` dataclass or a flat
            sequence ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
        pixels_per_cube: Minimum projected cube size in pixels for octree
            subdivision.
        inv_scale: Inverse scale factor for SDF evaluation.
        min_dist: Minimum distance from cameras for projection.
        memory_limit_mb: Memory budget in MB for fine-step batching.
        bisection_iters: Number of bisection iterations for vertex refinement.
        enclosed: If True, clamp SDF to positive outside bounds.
        simplify_occluded: If True, simplify geometry in occluded regions.
        visible_relax_iter: Number of relaxation iterations for visibility
            filtering.
        coarse_count: Maximum number of octree nodes in the coarse step.
    """

    def __init__(
        self,
        cameras: CameraSet | tuple,
        bounds: Bounds | Sequence[float],
        pixels_per_cube: int = 8,
        inv_scale: float = 10,
        min_dist: float = 1,
        memory_limit_mb: int = 1000,
        bisection_iters: int = 15,
        enclosed: bool = True,
        simplify_occluded: bool = True,
        visible_relax_iter: int = 2,
        coarse_count: int = 500_000,
    ) -> None:
        """Initialise the mesher with camera intrinsics and bounds."""
        # --- config dataclass for grouped access -------------------------
        self.config = MesherConfig(
            pixels_per_cube=pixels_per_cube,
            inv_scale=inv_scale,
            min_dist=min_dist,
            memory_limit_mb=memory_limit_mb,
            bisection_iters=bisection_iters,
            enclosed=enclosed,
            simplify_occluded=simplify_occluded,
            visible_relax_iter=visible_relax_iter,
            coarse_count=coarse_count,
        )

        # --- type system -------------------------------------------------
        self.float_type = np.float64  # kept for legacy test compat
        self.np_float_type = np.float64
        self._np_float = np.float64
        self._as_float = as_double
        self.sdf_float_type = c_float
        self._sdf_float_type = c_float
        self.sdf_np_float_type = np.float32
        self._sdf_np_float = np.float32
        self.sdf_AF = as_float
        self._as_sdf_float = as_float
        self.AF = as_double

        # --- validate & normalise cameras --------------------------------
        if isinstance(cameras, CameraSet):
            cam_set = cameras
        elif isinstance(cameras, (tuple, list)) and len(cameras) == 4:  # noqa: PLR2004
            _validate_cameras(cameras)  # fail-fast before CameraSet wraps
            cam_set = CameraSet.from_tuple(cameras)
        else:
            msg = "cameras must be a tuple/list of (cam_poses, Ks, Hs, Ws)"
            raise ValueError(msg)

        self.n_cameras = cam_set.count
        self._n_cameras = cam_set.count
        self._camera_data = cam_set.pack(dtype=self._np_float)
        self.cameras = self._camera_data

        # --- validate & normalise bounds ---------------------------------
        if isinstance(bounds, Bounds):
            self._bounds = bounds
            self.bounds = np.array(
                [
                    bounds.x_min,
                    bounds.x_max,
                    bounds.y_min,
                    bounds.y_max,
                    bounds.z_min,
                    bounds.z_max,
                ],
                dtype=np.float64,
            )
        else:
            self.bounds = _validate_bounds(bounds)
            self._bounds = Bounds.from_sequence(self.bounds)

        # Pre-compute bound vectors for vectorised out-of-bounds masking.
        self._bounds_min_np = self._bounds.mins
        self._bounds_max_np = self._bounds.maxs

        self.center = self._bounds.center.astype(self._np_float)
        self._center = self.center
        self.size = self._np_float(self._bounds.max_extent * 1.1)
        self._size = self.size
        self._inview_ppc = self._np_float(pixels_per_cube)
        self.inview_pixels_per_cube = self._inview_ppc
        self._inv_scale = self._np_float(inv_scale)
        self.inv_scale = self._inv_scale
        self._min_dist = self._np_float(min_dist)
        self.min_dist = self._min_dist

        # --- scalar config attrs (backward compat) -----------------------
        self.memory_limit_mb = memory_limit_mb
        self.bisection_iters = bisection_iters
        self.enclosed = enclosed
        self.simplify_occluded = simplify_occluded
        self.visible_relax_iter = visible_relax_iter
        self.coarse_count = coarse_count

        # --- DLL & SDF evaluator -----------------------------------------
        self._dll = CoreDLL()

        self._eval_sdfs = partial(
            evaluate_sdfs,
            sdf_dtype=self._sdf_np_float,
            bounds_min=self._bounds.mins if enclosed else None,
            bounds_max=self._bounds.maxs if enclosed else None,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise developer-friendly description of the mesher."""
        return (
            f"OcMesher("
            f"n_cameras={self.n_cameras}, "
            f"bounds={self.bounds.tolist()}, "
            f"bisection_iters={self.bisection_iters}, "
            f"enclosed={self.enclosed})"
        )

    def __enter__(self):
        """Support ``with OcMesher(...) as m:`` usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """No-op exit; provided so OcMesher can be used as a context manager."""
        return False

    # ------------------------------------------------------------------
    # Backward-compatible SDF evaluation
    # ------------------------------------------------------------------

    def kernel_caller(self, kernels, XYZ_all):
        """Evaluate SDF *kernels* at the given *XYZ_all* positions.

        Optimisations over the naïve implementation:
        - Pre-allocated output buffer filled in-place (no list accumulation,
          no ``np.stack`` / ``np.concatenate`` per batch).
        - Vectorised out-of-bounds check using pre-computed bound vectors
          with a single broadcast comparison per batch (replaces 6 temporary
          boolean arrays + 6 in-place ORs with 2 broadcasts + 2 reductions).
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

        for i in range(0, n_XYZ, _SDF_BATCH_SIZE):
            end = min(i + _SDF_BATCH_SIZE, n_XYZ)
            XYZ = XYZ_all[i:end]
            batch_size = end - i

            # Vectorised bounds check: 2 broadcasts + 2 any-reductions.
            if self.enclosed:
                out_bound = np.any(b_min >= XYZ, axis=1) | np.any(b_max <= XYZ, axis=1)

            for k_idx, kernel in enumerate(kernels):
                sdf = np.asarray(kernel(XYZ))
                if sdf.shape != (batch_size,):
                    msg = (
                        f"kernels[{k_idx}] returned shape {sdf.shape} "
                        f"for {batch_size} query points; expected ({batch_size},)"
                    )
                    raise ValueError(msg)
                if self.enclosed:
                    sdf[out_bound] = 1
                result[i:end, k_idx] = sdf

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        kernels: list[Callable],
    ) -> tuple[list[trimesh.Trimesh], list[NDArray]]:
        """Run the full meshing pipeline.

        Steps:
          1. Build coarse octree from camera projections.
          2. Refine blocks with SDF evaluation.
          3. Filter to visible blocks.
          4. Fine-step surface crossing detection.
          5. Construct triangle meshes via vertex bisection.
        """
        _validate_kernels(kernels)
        n_elements = len(kernels)
        n_blocks = self._run_coarse_octree(n_elements)
        logger.info("coarse blocks: %d", n_blocks)
        self._refine_with_sdf(kernels, n_blocks)
        n_vis_block = self._filter_visible()
        logger.info("visible blocks: %d", n_vis_block)
        nv = self._run_fine_step(kernels, n_elements, n_vis_block)
        return self._construct_meshes(kernels, n_elements, nv)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_coarse_octree(self, n_elements: int) -> int:
        """Build the initial octree considering only cameras (not SDF)."""
        with Timer("coarse step part1"):
            return self._dll.run_coarse(
                self._as_float(self._center),
                self._size,
                self._n_cameras,
                self._as_float(self._camera_data),
                self._inview_ppc,
                self._inv_scale,
                self._min_dist,
                self.config.coarse_count,
                self.config.memory_limit_mb,
                n_elements,
            )

    def _refine_with_sdf(self, kernels: list[Callable], n_blocks: int) -> None:
        """Refine octree blocks using SDF evaluation."""
        null_sdf = POINTER(self._sdf_float_type)()
        with Timer("coarse step part2"), tqdm(total=n_blocks) as pbar:
            while True:
                inc = self._dll.fine_group()
                if inc == 0:
                    break
                pbar.update(inc)
                n = self._dll.fine_iteration(null_sdf)
                while n > 0:
                    positions = AC(np.zeros((n, 3), dtype=self._np_float))
                    self._dll.fine_iteration_output(self._as_float(positions))
                    sdf = AC(
                        self._eval_sdfs(kernels, positions).min(axis=-1),
                    )
                    n = self._dll.fine_iteration(self._as_sdf_float(sdf))

    def _filter_visible(self) -> int:
        """Filter blocks to keep only visible ones."""
        with Timer("filter visible blocks"):
            return self._dll.vis_filter(
                self.config.simplify_occluded,
                self.config.visible_relax_iter,
            )

    def _run_fine_step(
        self,
        kernels: list[Callable],
        n_elements: int,
        n_vis_block: int,
    ) -> NDArray:
        """Run fine-resolution iterations to locate surface crossings."""
        with Timer("fine step"), tqdm(total=n_vis_block) as pbar:
            nv = np.zeros(1, dtype=np.int32)
            while True:
                n = self._dll.final_iteration(as_int(nv))
                if n == 0:
                    break
                positions = AC(np.empty((n, 3), dtype=self._np_float))
                self._dll.final_iteration2(self._as_float(positions))
                sdf = AC(self._eval_sdfs(kernels, positions))
                inc = self._dll.final_iteration3(self._as_sdf_float(sdf))
                pbar.update(inc)

            n = self._dll.final_iteration_occluded(as_int(nv))
            if n != 0:
                positions = AC(np.empty((n, 3), dtype=self._np_float))
                self._dll.final_iteration2(self._as_float(positions))
                sdf = AC(self._eval_sdfs(kernels, positions))
                self._dll.final_iteration3_occluded(self._as_sdf_float(sdf))

            nv = np.zeros(n_elements, dtype=np.int32)
            self._dll.final_remaining(as_int(nv))
        return nv

    def _construct_meshes(
        self,
        kernels: list[Callable],
        n_elements: int,
        nv: NDArray,
    ) -> tuple[list[trimesh.Trimesh], list[NDArray]]:
        """Construct triangle meshes for each element via bisection."""
        meshes: list[trimesh.Trimesh] = []
        in_view_tags: list[NDArray] = []
        with Timer("construct mesh"):
            for e in range(n_elements):
                mesh, tag = construct_element_mesh(
                    self._dll,
                    e,
                    kernels[e : e + 1],
                    int(nv[e]),
                    eval_fn=self._eval_sdfs,
                    as_flt=self._as_float,
                    as_sdf_flt=self._as_sdf_float,
                    sdf_ctype=self._sdf_float_type,
                    np_float=self._np_float,
                    bisection_iters=self.config.bisection_iters,
                )
                meshes.append(mesh)
                in_view_tags.append(tag)
        return meshes, in_view_tags
