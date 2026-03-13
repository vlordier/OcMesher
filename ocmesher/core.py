# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""OcMesher: octree-based mesh extraction from signed distance functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import gin
import numpy as np
import trimesh
from numpy.typing import NDArray
from tqdm import tqdm

from .dll import CoreDLL
from .meshing import construct_element_mesh
from .sdf import evaluate_sdfs
from .types import Bounds, CameraSet, MesherConfig
from .utils.interface import AC, POINTER, as_double, as_float, as_int, c_float
from .utils.timer import Timer


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

        self._np_float = np.float64
        self._as_float = as_double
        self._sdf_float_type = c_float
        self._sdf_np_float = np.float32
        self._as_sdf_float = as_float

        self._bounds = (
            bounds if isinstance(bounds, Bounds) else Bounds.from_sequence(bounds)
        )

        cam_set = (
            cameras
            if isinstance(cameras, CameraSet)
            else CameraSet.from_tuple(cameras)
        )
        self._n_cameras = cam_set.count
        self._camera_data = cam_set.pack(dtype=self._np_float)

        self._center = self._bounds.center.astype(self._np_float)
        self._size = self._np_float(self._bounds.max_extent * 1.1)
        self._inview_ppc = self._np_float(pixels_per_cube)
        self._inv_scale = self._np_float(inv_scale)
        self._min_dist = self._np_float(min_dist)

        self._dll = CoreDLL()

        self._eval_sdfs = partial(
            evaluate_sdfs,
            sdf_dtype=self._sdf_np_float,
            bounds_min=self._bounds.mins if enclosed else None,
            bounds_max=self._bounds.maxs if enclosed else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self, kernels: list[Callable],
    ) -> tuple[list[trimesh.Trimesh], list[NDArray]]:
        """Run the full meshing pipeline.

        Steps:
          1. Build coarse octree from camera projections.
          2. Refine blocks with SDF evaluation.
          3. Filter to visible blocks.
          4. Fine-step surface crossing detection.
          5. Construct triangle meshes via vertex bisection.
        """
        n_elements = len(kernels)
        n_blocks = self._run_coarse_octree(n_elements)
        self._refine_with_sdf(kernels, n_blocks)
        n_vis_block = self._filter_visible()
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
                self.config.simplify_occluded, self.config.visible_relax_iter,
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
                positions = AC(np.zeros((n, 3), dtype=self._np_float))
                self._dll.final_iteration2(self._as_float(positions))
                sdf = AC(self._eval_sdfs(kernels, positions))
                inc = self._dll.final_iteration3(self._as_sdf_float(sdf))
                pbar.update(inc)

            n = self._dll.final_iteration_occluded(as_int(nv))
            if n != 0:
                positions = AC(np.zeros((n, 3), dtype=self._np_float))
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
