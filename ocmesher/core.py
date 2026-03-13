# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

from pathlib import Path

import gin
import numpy as np
import trimesh
from tqdm import tqdm

from .utils.interface import (
    AC,
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

# Camera data layout constants (must match C++ side in core.h)
CAM_INV_POSE_SIZE = 12   # 3x4 inverse pose matrix
CAM_K_SIZE = 9           # 3x3 intrinsic matrix
CAM_HW_SIZE = 2          # height, width
CAMERA_DATA_STRIDE = CAM_INV_POSE_SIZE + CAM_K_SIZE + CAM_HW_SIZE  # 23

@gin.configurable
class OcMesher:
    def __init__(self,
        cameras,
        bounds,
        pixels_per_cube=8,
        inv_scale=10,
        min_dist=1,
        memory_limit_mb=1000,
        bisection_iters=15,
        enclosed=True,
        simplify_occluded=True,
        visible_relax_iter=2,
        coarse_count=500000,
        sdf_batch_size=10_000_000,
        bounds_expansion_factor=1.1,
        pixel_downsample_factor=10,
        memory_threshold_coarse=0.6,
        memory_threshold_fine=0.8,
    ):
        dll = load_cdll(str(Path(__file__).parent.resolve()/"lib"/"core.so"))
        self.float_type = c_double
        self.np_float_type = np.float64
        self.AF = AsDouble
        self.sdf_float_type = c_float
        self.sdf_np_float_type = np.float32
        self.sdf_AF = AsFloat
        self.bounds = bounds
        self.memory_limit_mb = memory_limit_mb

        sdf_batch_size = int(sdf_batch_size)
        if sdf_batch_size < 1:
            msg = f"sdf_batch_size must be >= 1, got {sdf_batch_size}"
            raise ValueError(msg)
        self.sdf_batch_size = sdf_batch_size

        if bounds_expansion_factor <= 0:
            msg = f"bounds_expansion_factor must be > 0, got {bounds_expansion_factor}"
            raise ValueError(msg)
        self.bounds_expansion_factor = bounds_expansion_factor

        if pixel_downsample_factor <= 0:
            msg = f"pixel_downsample_factor must be > 0, got {pixel_downsample_factor}"
            raise ValueError(msg)
        self.pixel_downsample_factor = self.np_float_type(pixel_downsample_factor)

        if not (0 < memory_threshold_coarse <= 1):
            msg = f"memory_threshold_coarse must be in (0, 1], got {memory_threshold_coarse}"
            raise ValueError(msg)
        self.memory_threshold_coarse = self.np_float_type(memory_threshold_coarse)

        if not (0 < memory_threshold_fine <= 1):
            msg = f"memory_threshold_fine must be in (0, 1], got {memory_threshold_fine}"
            raise ValueError(msg)
        self.memory_threshold_fine = self.np_float_type(memory_threshold_fine)

        cam_poses, Ks, Hs, Ws = cameras
        self.n_cameras = len(cam_poses)
        self.cameras = np.zeros(CAMERA_DATA_STRIDE * self.n_cameras, dtype=self.np_float_type)
        for i in range(self.n_cameras):
            self.cameras[CAMERA_DATA_STRIDE * i: CAMERA_DATA_STRIDE * (i+1)] = np.concatenate([
                np.linalg.inv(cam_poses[i])[:3, :4].reshape(-1),
                Ks[i].reshape(-1), [Hs[i]], [Ws[i]]
            ]).astype(self.np_float_type)

        self.inview_pixels_per_cube = self.np_float_type(pixels_per_cube)
        self.inv_scale = self.np_float_type(inv_scale)
        self.min_dist = self.np_float_type(min_dist)

        self.center = np.array([(bounds[0]+bounds[1]) / 2, (bounds[2]+bounds[3]) / 2, (bounds[4]+bounds[5]) / 2], self.np_float_type)
        self.size = self.np_float_type(max(max(bounds[1] - bounds[0], bounds[3] - bounds[2]), bounds[5] - bounds[4]) * self.bounds_expansion_factor)

        self.bisection_iters = bisection_iters
        self.enclosed = enclosed
        self.simplify_occluded = simplify_occluded
        self.visible_relax_iter = visible_relax_iter
        self.coarse_count = coarse_count

        register_func(self, dll, "run_coarse", [
            POINTER(self.float_type), self.float_type,
            c_int32, POINTER(self.float_type),
            self.float_type, self.float_type, self.float_type,
            c_int32, c_int32, c_int32,
            self.float_type, self.float_type, self.float_type,
        ], c_int32)
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
        register_func(self, dll, "update_verts", [c_int32, POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type)])
        register_func(self, dll, "get_lr_verts", [c_int32, POINTER(self.float_type), POINTER(self.float_type)])
        register_func(self, dll, "finalize_verts", [c_int32, POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type)])
        register_func(self, dll, "construct_faces", [c_int32, POINTER(self.float_type), POINTER(c_int32)])
        register_func(self, dll, "get_extra_verts_center", [POINTER(self.float_type), POINTER(self.float_type)])
        register_func(self, dll, "update_extra_verts", [
            POINTER(self.sdf_float_type), POINTER(self.sdf_float_type),
            POINTER(self.sdf_float_type), POINTER(self.sdf_float_type),
            POINTER(self.float_type), POINTER(self.float_type),
        ])
        register_func(self, dll, "get_lr_extra_verts", [POINTER(self.float_type), POINTER(self.float_type), POINTER(self.float_type), POINTER(self.float_type)])
        register_func(self, dll, "finalize_extra_verts", [
            POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type),
            POINTER(self.sdf_float_type), POINTER(self.sdf_float_type), POINTER(self.float_type),
        ])
        register_func(self, dll, "get_faces", [POINTER(c_int32)])
        register_func(self, dll, "get_in_view_tag", [c_int32, POINTER(c_bool)])

    def kernel_caller(self, kernels, XYZ_all):
        n_XYZ = len(XYZ_all)
        if n_XYZ == 0: return np.zeros((0, len(kernels)), dtype=self.sdf_np_float_type)
        step = self.sdf_batch_size
        sdfs = []
        for i in range(0, n_XYZ, step):
            XYZ = XYZ_all[i: i+step]
            sdfs_i = []
            if self.enclosed:
                out_bound = np.zeros(len(XYZ), dtype=bool)
                for c in range(3):
                    out_bound |= XYZ[:, c] <= self.bounds[c*2]
                    out_bound |= XYZ[:, c] >= self.bounds[c*2+1]
            for kernel in kernels:
                sdf = kernel(XYZ)
                if self.enclosed: sdf[out_bound] = 1
                sdfs_i.append(sdf)
            sdfs.append(np.stack(sdfs_i, -1).astype(self.sdf_np_float_type))
        return np.concatenate(sdfs, 0)

    def __call__(self, kernels):
        n_elements = len(kernels)
        # octree only considering cameras, not sdf
        with Timer("coarse step part1"):
            n_blocks = self.run_coarse(
                self.AF(self.center), self.size,
                self.n_cameras, self.AF(self.cameras),
                self.inview_pixels_per_cube,
                self.inv_scale, self.min_dist,
                self.coarse_count, self.memory_limit_mb, n_elements,
                self.pixel_downsample_factor, self.memory_threshold_coarse, self.memory_threshold_fine,
            )
        # start considering sdf
        with Timer("coarse step part2"), tqdm(total=n_blocks) as pbar:
            while True:
                inc = self.fine_group()
                if inc == 0: break
                pbar.update(inc)
                n = self.fine_iteration(POINTER(self.sdf_float_type)())
                while n > 0:
                    positions = AC(np.zeros((n, 3), dtype=self.np_float_type))
                    self.fine_iteration_output(self.AF(positions))
                    sdf = AC(self.kernel_caller(kernels, positions).min(axis=-1))
                    n = self.fine_iteration(self.sdf_AF(sdf))
        with Timer("filter visible blocks"):
            n_vis_block = self.vis_filter(self.simplify_occluded, self.visible_relax_iter)

        with Timer("fine step"), tqdm(total=n_vis_block) as pbar:
            nv = np.zeros(1, dtype=np.int32)
            while True:
                n = self.final_iteration(AsInt(nv))
                if n == 0: break
                positions = AC(np.zeros((n, 3), dtype=self.np_float_type))
                self.final_iteration2(self.AF(positions))
                sdf = AC(self.kernel_caller(kernels, positions))
                inc = self.final_iteration3(self.sdf_AF(sdf))
                pbar.update(inc)
            n = self.final_iteration_occluded(AsInt(nv))
            if n != 0:
                positions = AC(np.zeros((n, 3), dtype=self.np_float_type))
                self.final_iteration2(self.AF(positions))
                sdf = AC(self.kernel_caller(kernels, positions))
                self.final_iteration3_occluded(self.sdf_AF(sdf))
            nv = np.zeros(n_elements, dtype=np.int32)
            self.final_remaining(AsInt(nv))
            del positions, sdf

        with Timer("construct mesh"):
            meshes = []
            in_view_tags = []
            for e in range(n_elements):
                mesh, in_view_tag = self._construct_element_mesh(e, kernels[e:e+1], nv[e])
                meshes.append(mesh)
                in_view_tags.append(in_view_tag)
                print(f"element {e} has vertices #{mesh.vertices.shape[0]} faces #{mesh.faces.shape[0]}")
        return meshes, in_view_tags

    def _construct_element_mesh(self, e, k_e, num_verts):
        """Construct mesh for a single SDF element via bisection refinement."""
        centers = np.zeros((num_verts, 3), dtype=self.np_float_type)
        self.get_verts_center(e, self.AF(centers))
        center_sdf = self.kernel_caller(k_e, centers)
        cubes = AC(np.zeros((num_verts * 8, 3), dtype=self.np_float_type))
        self.update_verts(e, POINTER(self.sdf_float_type)(), POINTER(self.sdf_float_type)(), self.AF(cubes))
        center_sdf_ptr = self.sdf_AF(AC(center_sdf))
        for _ in tqdm(range(self.bisection_iters)):
            sdf = self.kernel_caller(k_e, cubes)
            self.update_verts(e, self.sdf_AF(AC(sdf)), center_sdf_ptr, self.AF(cubes))
        cubes_r = AC(np.zeros((num_verts * 8, 3), dtype=self.np_float_type))
        self.get_lr_verts(e, self.AF(cubes), self.AF(cubes_r))
        sdf_l = self.kernel_caller(k_e, cubes)
        sdf_r = self.kernel_caller(k_e, cubes_r)
        del cubes, cubes_r, centers, center_sdf
        vertices = np.zeros((num_verts, 3), dtype=self.np_float_type)
        self.finalize_verts(e, self.sdf_AF(sdf_l), self.sdf_AF(sdf_r), self.AF(vertices))
        del sdf_l, sdf_r

        vertices, faces = self._refine_extra_vertices(e, k_e, vertices)

        in_view_tag = np.zeros(vertices.shape[0], dtype=bool)
        self.get_in_view_tag(e, AsBool(in_view_tag))
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), in_view_tag

    def _refine_extra_vertices(self, e, k_e, vertices):
        """Compute edge/face extra vertices and assemble final faces."""
        cnts = np.zeros(3, dtype=np.int32)
        self.construct_faces(e, self.AF(vertices), AsInt(cnts))
        nve, nvf, nf = cnts
        edge_vertices_c = AC(np.zeros((nve, 3), dtype=self.np_float_type))
        face_vertices_c = AC(np.zeros((nvf, 3), dtype=self.np_float_type))
        self.get_extra_verts_center(self.AF(edge_vertices_c), self.AF(face_vertices_c))
        ecenter_sdf = self.kernel_caller(k_e, edge_vertices_c)
        fcenter_sdf = self.kernel_caller(k_e, face_vertices_c)
        edge_vertices_lr = AC(np.zeros((nve * 2, 3), dtype=self.np_float_type))
        face_vertices_lr = AC(np.zeros((nvf * 4, 3), dtype=self.np_float_type))
        self.update_extra_verts(
            POINTER(self.sdf_float_type)(), POINTER(self.sdf_float_type)(),
            POINTER(self.sdf_float_type)(), POINTER(self.sdf_float_type)(),
            self.AF(edge_vertices_lr), self.AF(face_vertices_lr),
        )
        for _ in range(self.bisection_iters):
            e_sdf = self.kernel_caller(k_e, edge_vertices_lr)
            f_sdf = self.kernel_caller(k_e, face_vertices_lr)
            self.update_extra_verts(
                self.sdf_AF(e_sdf), self.sdf_AF(f_sdf),
                self.sdf_AF(ecenter_sdf), self.sdf_AF(fcenter_sdf),
                self.AF(edge_vertices_lr), self.AF(face_vertices_lr),
            )
        del edge_vertices_c, face_vertices_c, ecenter_sdf, fcenter_sdf
        edge_vertices_r = AC(np.zeros((nve * 2, 3), dtype=self.np_float_type))
        face_vertices_r = AC(np.zeros((nvf * 4, 3), dtype=self.np_float_type))
        self.get_lr_extra_verts(self.AF(edge_vertices_lr), self.AF(edge_vertices_r), self.AF(face_vertices_lr), self.AF(face_vertices_r))
        esdf_l = self.kernel_caller(k_e, edge_vertices_lr)
        esdf_r = self.kernel_caller(k_e, edge_vertices_r)
        fsdf_l = self.kernel_caller(k_e, face_vertices_lr)
        fsdf_r = self.kernel_caller(k_e, face_vertices_r)
        del edge_vertices_lr, edge_vertices_r, face_vertices_lr, face_vertices_r
        edge_vertices = np.zeros((nve, 3), dtype=self.np_float_type)
        face_vertices = np.zeros((nvf, 3), dtype=self.np_float_type)
        self.finalize_extra_verts(self.sdf_AF(esdf_l), self.sdf_AF(esdf_r), self.AF(edge_vertices), self.sdf_AF(fsdf_l), self.sdf_AF(fsdf_r), self.AF(face_vertices))
        del esdf_l, esdf_r, fsdf_l, fsdf_r
        faces = AC(np.zeros((nf, 3), dtype=np.int32))
        self.get_faces(AsInt(faces))
        vertices = np.concatenate((vertices, edge_vertices, face_vertices))
        return vertices, faces