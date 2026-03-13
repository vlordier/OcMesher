# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Octree-based mesh extraction from signed distance functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gin
import numpy as np
import trimesh
from numpy.typing import NDArray
from tqdm import tqdm

from .types import SDF_BATCH_SIZE, Bounds, CameraSet, MesherConfig
from .utils.interface import (
    AC,
    POINTER,
    as_bool,
    as_double,
    as_float,
    as_int,
    c_bool,
    c_double,
    c_float,
    c_int32,
    load_cdll,
    register_func,
)
from .utils.timer import Timer


@gin.configurable
class OcMesher:
    """Octree-based mesher that extracts surfaces from signed distance functions.

    Accepts camera data and axis-aligned bounds, then uses an octree to
    adaptively evaluate SDF kernels and extract a triangle mesh via
    bisection-based surface finding.
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

        self._float_type = c_double
        self._np_float = np.float64
        self._as_float = as_double
        self._sdf_float_type = c_float
        self._sdf_np_float = np.float32
        self._as_sdf_float = as_float

        self._bounds = (
            bounds if isinstance(bounds, Bounds) else Bounds.from_sequence(bounds)
        )
        self._bounds_min = self._bounds.mins
        self._bounds_max = self._bounds.maxs

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

        dll_path = str(Path(__file__).parent.resolve() / "lib" / "core.so")
        self._register_dll_functions(load_cdll(dll_path))

    def _register_dll_functions(self, dll: Any) -> None:
        """Register all C++ functions from the shared library."""
        fp = POINTER(self._float_type)
        sfp = POINTER(self._sdf_float_type)
        ip = POINTER(c_int32)

        register_func(self, dll, "run_coarse", [
            fp, self._float_type, c_int32, fp,
            self._float_type, self._float_type, self._float_type,
            c_int32, c_int32, c_int32,
        ], c_int32)
        register_func(self, dll, "fine_group", [], c_int32)
        register_func(self, dll, "fine_iteration", [sfp], c_int32)
        register_func(self, dll, "fine_iteration_output", [fp])
        register_func(self, dll, "vis_filter", [c_bool, c_int32], c_int32)
        register_func(self, dll, "final_iteration", [], c_int32)
        register_func(self, dll, "final_iteration_occluded", [], c_int32)
        register_func(self, dll, "final_iteration2", [fp])
        register_func(self, dll, "final_iteration3", [sfp], c_int32)
        register_func(self, dll, "final_iteration3_occluded", [sfp])
        register_func(self, dll, "final_remaining", [ip])
        register_func(self, dll, "get_verts_center", [c_int32, fp])
        register_func(self, dll, "update_verts", [c_int32, sfp, sfp, fp])
        register_func(self, dll, "get_lr_verts", [c_int32, fp, fp])
        register_func(self, dll, "finalize_verts", [c_int32, sfp, sfp, fp])
        register_func(self, dll, "construct_faces", [c_int32, fp, ip])
        register_func(self, dll, "get_extra_verts_center", [fp, fp])
        register_func(self, dll, "update_extra_verts", [sfp, sfp, sfp, sfp, fp, fp])
        register_func(self, dll, "get_lr_extra_verts", [fp, fp, fp, fp])
        register_func(self, dll, "finalize_extra_verts", [sfp, sfp, fp, sfp, sfp, fp])
        register_func(self, dll, "get_faces", [ip])
        register_func(self, dll, "get_in_view_tag", [c_int32, POINTER(c_bool)])

    def _evaluate_sdfs(
        self,
        kernels: list[Callable],
        positions: NDArray,
    ) -> NDArray:
        """Evaluate SDF kernels at positions with batching and bounds clamping."""
        n = len(positions)
        if n == 0:
            return np.zeros((0, len(kernels)), dtype=self._sdf_np_float)

        results: list[NDArray] = []
        for i in range(0, n, SDF_BATCH_SIZE):
            batch = positions[i : i + SDF_BATCH_SIZE]
            out_bound = None
            if self.config.enclosed:
                out_bound = np.any(batch <= self._bounds_min, axis=1) | np.any(
                    batch >= self._bounds_max, axis=1,
                )
            batch_sdfs = []
            for kernel in kernels:
                sdf = kernel(batch)
                if out_bound is not None:
                    sdf[out_bound] = 1
                batch_sdfs.append(sdf)
            results.append(np.stack(batch_sdfs, axis=-1).astype(self._sdf_np_float))
        return np.concatenate(results, axis=0)

    def __call__(
        self, kernels: list[Callable],
    ) -> tuple[list[trimesh.Trimesh], list[NDArray]]:
        """Run the full meshing pipeline: coarse octree, SDF refinement, mesh extraction."""
        n_elements = len(kernels)
        n_blocks = self._run_coarse_octree(n_elements)
        self._refine_with_sdf(kernels, n_blocks)
        n_vis_block = self._filter_visible()
        nv = self._run_fine_step(kernels, n_elements, n_vis_block)
        return self._construct_meshes(kernels, n_elements, nv)

    def _run_coarse_octree(self, n_elements: int) -> int:
        """Build the initial octree considering only cameras (not SDF)."""
        with Timer("coarse step part1"):
            n_blocks = self.run_coarse(
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
        return n_blocks

    def _refine_with_sdf(self, kernels: list[Callable], n_blocks: int) -> None:
        """Refine octree blocks using SDF evaluation."""
        null_sdf = POINTER(self._sdf_float_type)()
        with Timer("coarse step part2"), tqdm(total=n_blocks) as pbar:
            while True:
                inc = self.fine_group()
                if inc == 0:
                    break
                pbar.update(inc)
                n = self.fine_iteration(null_sdf)
                while n > 0:
                    positions = AC(np.zeros((n, 3), dtype=self._np_float))
                    self.fine_iteration_output(self._as_float(positions))
                    sdf = AC(self._evaluate_sdfs(kernels, positions).min(axis=-1))
                    n = self.fine_iteration(self._as_sdf_float(sdf))

    def _filter_visible(self) -> int:
        """Filter blocks to keep only visible ones."""
        with Timer("filter visible blocks"):
            return self.vis_filter(
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
                n = self.final_iteration(as_int(nv))
                if n == 0:
                    break
                positions = AC(np.zeros((n, 3), dtype=self._np_float))
                self.final_iteration2(self._as_float(positions))
                sdf = AC(self._evaluate_sdfs(kernels, positions))
                inc = self.final_iteration3(self._as_sdf_float(sdf))
                pbar.update(inc)
            n = self.final_iteration_occluded(as_int(nv))
            if n != 0:
                positions = AC(np.zeros((n, 3), dtype=self._np_float))
                self.final_iteration2(self._as_float(positions))
                sdf = AC(self._evaluate_sdfs(kernels, positions))
                self.final_iteration3_occluded(self._as_sdf_float(sdf))
            nv = np.zeros(n_elements, dtype=np.int32)
            self.final_remaining(as_int(nv))
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
                kernel = kernels[e : e + 1]
                mesh, tag = self._construct_single_mesh(e, kernel, int(nv[e]))
                meshes.append(mesh)
                in_view_tags.append(tag)
        return meshes, in_view_tags

    def _construct_single_mesh(
        self,
        element_idx: int,
        kernel: list[Callable],
        nv_count: int,
    ) -> tuple[trimesh.Trimesh, NDArray]:
        """Build a single element's mesh: bisect vertices, construct faces."""
        vertices = self._bisect_cube_vertices(element_idx, kernel, nv_count)

        cnts = np.zeros(3, dtype=np.int32)
        self.construct_faces(element_idx, self._as_float(vertices), as_int(cnts))
        nve, nvf, nf = int(cnts[0]), int(cnts[1]), int(cnts[2])

        edge_verts, face_verts = self._bisect_extra_vertices(kernel, nve, nvf)

        faces = AC(np.zeros((nf, 3), dtype=np.int32))
        self.get_faces(as_int(faces))
        vertices = np.concatenate((vertices, edge_verts, face_verts))

        in_view_tag = np.zeros(vertices.shape[0], dtype=bool)
        self.get_in_view_tag(element_idx, as_bool(in_view_tag))

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        print(
            f"element {element_idx} has vertices #{mesh.vertices.shape[0]}"
            f" faces #{mesh.faces.shape[0]}",
        )
        return mesh, in_view_tag

    def _bisect_cube_vertices(
        self,
        element_idx: int,
        kernel: list[Callable],
        nv_count: int,
    ) -> NDArray:
        """Refine cube vertex positions via iterative bisection."""
        centers = np.zeros((nv_count, 3), dtype=self._np_float)
        self.get_verts_center(element_idx, self._as_float(centers))
        center_sdf = self._evaluate_sdfs(kernel, centers)

        cubes = AC(np.zeros((nv_count * 8, 3), dtype=self._np_float))
        null_sdf = POINTER(self._sdf_float_type)()
        self.update_verts(element_idx, null_sdf, null_sdf, self._as_float(cubes))

        for _ in tqdm(range(self.config.bisection_iters)):
            sdf = self._evaluate_sdfs(kernel, cubes)
            self.update_verts(
                element_idx,
                self._as_sdf_float(AC(sdf)),
                self._as_sdf_float(AC(center_sdf)),
                self._as_float(cubes),
            )

        cubes_r = AC(np.zeros((nv_count * 8, 3), dtype=self._np_float))
        self.get_lr_verts(element_idx, self._as_float(cubes), self._as_float(cubes_r))
        sdf_l = self._evaluate_sdfs(kernel, cubes)
        sdf_r = self._evaluate_sdfs(kernel, cubes_r)

        vertices = np.zeros((nv_count, 3), dtype=self._np_float)
        self.finalize_verts(
            element_idx,
            self._as_sdf_float(sdf_l),
            self._as_sdf_float(sdf_r),
            self._as_float(vertices),
        )
        return vertices

    def _bisect_extra_vertices(
        self,
        kernel: list[Callable],
        nve: int,
        nvf: int,
    ) -> tuple[NDArray, NDArray]:
        """Refine edge and face vertex positions via iterative bisection."""
        edge_c = AC(np.zeros((nve, 3), dtype=self._np_float))
        face_c = AC(np.zeros((nvf, 3), dtype=self._np_float))
        self.get_extra_verts_center(self._as_float(edge_c), self._as_float(face_c))
        ecenter_sdf = self._evaluate_sdfs(kernel, edge_c)
        fcenter_sdf = self._evaluate_sdfs(kernel, face_c)

        edge_lr = AC(np.zeros((nve * 2, 3), dtype=self._np_float))
        face_lr = AC(np.zeros((nvf * 4, 3), dtype=self._np_float))
        null_sdf = POINTER(self._sdf_float_type)()
        self.update_extra_verts(
            null_sdf, null_sdf, null_sdf, null_sdf,
            self._as_float(edge_lr), self._as_float(face_lr),
        )

        for _ in range(self.config.bisection_iters):
            e_sdf = self._evaluate_sdfs(kernel, edge_lr)
            f_sdf = self._evaluate_sdfs(kernel, face_lr)
            self.update_extra_verts(
                self._as_sdf_float(e_sdf),
                self._as_sdf_float(f_sdf),
                self._as_sdf_float(ecenter_sdf),
                self._as_sdf_float(fcenter_sdf),
                self._as_float(edge_lr),
                self._as_float(face_lr),
            )

        edge_r = AC(np.zeros((nve * 2, 3), dtype=self._np_float))
        face_r = AC(np.zeros((nvf * 4, 3), dtype=self._np_float))
        self.get_lr_extra_verts(
            self._as_float(edge_lr), self._as_float(edge_r),
            self._as_float(face_lr), self._as_float(face_r),
        )
        esdf_l = self._evaluate_sdfs(kernel, edge_lr)
        esdf_r = self._evaluate_sdfs(kernel, edge_r)
        fsdf_l = self._evaluate_sdfs(kernel, face_lr)
        fsdf_r = self._evaluate_sdfs(kernel, face_r)

        edge_verts = np.zeros((nve, 3), dtype=self._np_float)
        face_verts = np.zeros((nvf, 3), dtype=self._np_float)
        self.finalize_extra_verts(
            self._as_sdf_float(esdf_l), self._as_sdf_float(esdf_r),
            self._as_float(edge_verts),
            self._as_sdf_float(fsdf_l), self._as_sdf_float(fsdf_r),
            self._as_float(face_verts),
        )
        return edge_verts, face_verts
