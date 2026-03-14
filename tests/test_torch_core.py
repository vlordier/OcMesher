"""Tests for ``ocmesher.torch_core`` optimisations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ocmesher.torch_core import TorchOcMesher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def single_cam_mesher(sample_cameras, sample_bounds):
    """TorchOcMesher with a single camera on CPU."""
    return TorchOcMesher(sample_cameras, sample_bounds, device="cpu")


@pytest.fixture
def multi_cam_mesher(sample_camera_pose, sample_intrinsics, sample_bounds):
    """TorchOcMesher with four cameras on CPU."""
    cameras = (
        [sample_camera_pose] * 4,
        [sample_intrinsics] * 4,
        [720] * 4,
        [1280] * 4,
    )
    return TorchOcMesher(cameras, sample_bounds, device="cpu")


# ---------------------------------------------------------------------------
# Vectorised camera init
# ---------------------------------------------------------------------------
class TestVectorisedInit:
    def test_cam_inv_poses_shape(self, single_cam_mesher):
        assert single_cam_mesher.cam_inv_poses.shape == (1, 3, 4)

    def test_cam_intrinsics_shape(self, single_cam_mesher):
        assert single_cam_mesher.cam_intrinsics.shape == (1, 3, 3)

    def test_pix_ang_shape(self, single_cam_mesher):
        assert single_cam_mesher._pix_ang.shape == (1,)
        assert single_cam_mesher._pix_ang.dtype == torch.float64

    def test_multi_cam_shapes(self, multi_cam_mesher):
        assert multi_cam_mesher.cam_inv_poses.shape == (4, 3, 4)
        assert multi_cam_mesher.cam_intrinsics.shape == (4, 3, 3)
        assert multi_cam_mesher._pix_ang.shape == (4,)

    def test_pix_ang_ppc_precomputed(self, single_cam_mesher):
        expected = single_cam_mesher._pix_ang * single_cam_mesher.pixels_per_cube
        torch.testing.assert_close(single_cam_mesher._pix_ang_ppc, expected)


# ---------------------------------------------------------------------------
# Relaxation offset grid
# ---------------------------------------------------------------------------
class TestRelaxOffsets:
    def test_relax_offsets_shape(self, single_cam_mesher):
        rl = single_cam_mesher.visible_relax_iter
        expected_size = (2 * rl + 1) ** 2
        assert single_cam_mesher._relax_dx.shape == (expected_size,)
        assert single_cam_mesher._relax_dy.shape == (expected_size,)

    def test_relax_offsets_range(self, single_cam_mesher):
        rl = single_cam_mesher.visible_relax_iter
        assert single_cam_mesher._relax_dx.min().item() == -rl
        assert single_cam_mesher._relax_dx.max().item() == rl
        assert single_cam_mesher._relax_dy.min().item() == -rl
        assert single_cam_mesher._relax_dy.max().item() == rl


# ---------------------------------------------------------------------------
# Visibility filter
# ---------------------------------------------------------------------------
class TestVisibilityFilter:
    def test_returns_bool_tensor(self, single_cam_mesher):
        positions = torch.randn(10, 3, dtype=torch.float64, device=single_cam_mesher.device)
        result = single_cam_mesher._visibility_filter(positions)
        assert result.dtype == torch.bool
        assert result.shape == (10,)

    def test_empty_positions(self, single_cam_mesher):
        positions = torch.zeros(0, 3, dtype=torch.float64, device=single_cam_mesher.device)
        result = single_cam_mesher._visibility_filter(positions)
        assert result.shape == (0,)

    def test_without_simplify_occluded(self, sample_cameras, sample_bounds):
        mesher = TorchOcMesher(sample_cameras, sample_bounds, device="cpu", simplify_occluded=False)
        positions = torch.zeros(5, 3, dtype=torch.float64, device=mesher.device)
        result = mesher._visibility_filter(positions)
        assert result.dtype == torch.bool

    def test_multi_cam_visibility(self, multi_cam_mesher):
        positions = torch.randn(20, 3, dtype=torch.float64, device=multi_cam_mesher.device)
        result = multi_cam_mesher._visibility_filter(positions)
        assert result.shape == (20,)


# ---------------------------------------------------------------------------
# Marching cubes (vectorised triangle extraction)
# ---------------------------------------------------------------------------
class TestMarchingCubes:
    def test_empty_cubes(self, single_cam_mesher):
        corners = torch.zeros(0, 8, 3, dtype=torch.float64)
        sdf = torch.zeros(0, 8, dtype=torch.float32)
        verts, faces = single_cam_mesher._marching_cubes(corners, sdf)
        assert verts.shape == (0, 3)
        assert faces.shape == (0, 3)

    def test_all_positive_sdf_no_surface(self, single_cam_mesher):
        corners = torch.randn(5, 8, 3, dtype=torch.float64)
        sdf = torch.ones(5, 8, dtype=torch.float32)
        verts, faces = single_cam_mesher._marching_cubes(corners, sdf)
        assert verts.shape[0] == 0
        assert faces.shape[0] == 0

    def test_sphere_produces_mesh(self, single_cam_mesher):
        coords, levels = single_cam_mesher._build_coarse_octree()
        mask, _ = single_cam_mesher._find_surface_cubes(
            [lambda xyz: np.linalg.norm(xyz, axis=1) - 3.0],
            coords,
            levels,
        )
        s_coords = coords[mask]
        s_levels = levels[mask]
        if len(s_coords) == 0:
            pytest.skip("No surface cubes found")
        corners = single_cam_mesher._cube_corner_positions(s_coords, s_levels)
        flat = corners.reshape(-1, 3)
        sdf_all = single_cam_mesher._evaluate_sdf([lambda xyz: np.linalg.norm(xyz, axis=1) - 3.0], flat)
        sdf_min = sdf_all.min(dim=-1).values.reshape(len(s_coords), 8)
        verts, faces = single_cam_mesher._marching_cubes(corners, sdf_min)
        assert verts.shape[0] > 0
        assert faces.shape[0] > 0
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3


# ---------------------------------------------------------------------------
# SDF evaluation
# ---------------------------------------------------------------------------
class TestSDFEvaluation:
    def test_empty_positions(self, single_cam_mesher, sphere_kernel):
        positions = torch.zeros(0, 3, dtype=torch.float64, device=single_cam_mesher.device)
        result = single_cam_mesher._evaluate_sdf([sphere_kernel], positions)
        assert result.shape == (0, 1)

    def test_single_kernel(self, single_cam_mesher, sphere_kernel):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64, device=single_cam_mesher.device
        )
        result = single_cam_mesher._evaluate_sdf([sphere_kernel], positions)
        assert result.shape == (2, 1)
        assert result.dtype == torch.float32

    def test_multiple_kernels(self, single_cam_mesher, sphere_kernel, plane_kernel):
        positions = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64, device=single_cam_mesher.device)
        result = single_cam_mesher._evaluate_sdf([sphere_kernel, plane_kernel], positions)
        assert result.shape == (1, 2)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_sphere_mesh_output(self, single_cam_mesher, sphere_kernel):
        meshes, tags = single_cam_mesher([sphere_kernel])
        assert len(meshes) == 1
        assert meshes[0].vertices.shape[0] > 0
        assert meshes[0].faces.shape[0] > 0
        assert meshes[0].vertices.shape[1] == 3
        assert meshes[0].faces.shape[1] == 3
        assert len(tags) == 1

    def test_multiple_kernels_output(self, single_cam_mesher, sphere_kernel, plane_kernel):
        meshes, tags = single_cam_mesher([sphere_kernel, plane_kernel])
        assert len(meshes) == 2
        assert len(tags) == 2

    def test_multi_cam_output(self, multi_cam_mesher, sphere_kernel):
        meshes, tags = multi_cam_mesher([sphere_kernel])
        assert len(meshes) == 1
        assert meshes[0].vertices.shape[0] > 0


# ---------------------------------------------------------------------------
# Bounds precomputation
# ---------------------------------------------------------------------------
class TestBoundsPrecompute:
    def test_bounds_min_max_numpy(self, single_cam_mesher):
        assert hasattr(single_cam_mesher, "_bounds_min_np")
        assert hasattr(single_cam_mesher, "_bounds_max_np")
        assert single_cam_mesher._bounds_min_np.shape == (3,)
        assert single_cam_mesher._bounds_max_np.shape == (3,)
        np.testing.assert_array_less(single_cam_mesher._bounds_min_np, single_cam_mesher._bounds_max_np)
