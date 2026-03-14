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


# ---------------------------------------------------------------------------
# Compute dtype (_fdtype) selection
# ---------------------------------------------------------------------------
class TestFdtype:
    def test_cpu_uses_float64(self, single_cam_mesher):
        """CPU mesher must use float64 for numerical precision."""
        assert single_cam_mesher._fdtype == torch.float64

    def test_cam_tensors_match_fdtype(self, single_cam_mesher):
        """Camera tensors should use the compute dtype."""
        assert single_cam_mesher.cam_inv_poses.dtype == single_cam_mesher._fdtype
        assert single_cam_mesher.cam_intrinsics.dtype == single_cam_mesher._fdtype

    def test_bounds_tensors_match_fdtype(self, single_cam_mesher):
        """Bounds tensors should use the compute dtype."""
        assert single_cam_mesher.bounds_min.dtype == single_cam_mesher._fdtype
        assert single_cam_mesher.bounds_max.dtype == single_cam_mesher._fdtype
        assert single_cam_mesher.center.dtype == single_cam_mesher._fdtype

    def test_pix_ang_matches_fdtype(self, single_cam_mesher):
        """Pixel angular size tensors should use the compute dtype."""
        assert single_cam_mesher._pix_ang.dtype == single_cam_mesher._fdtype
        assert single_cam_mesher._pix_ang_ppc.dtype == single_cam_mesher._fdtype

    def test_cam_proj_matches_fdtype(self, single_cam_mesher):
        """Combined K@inv_pose matrix should use the compute dtype."""
        assert single_cam_mesher._cam_proj.dtype == single_cam_mesher._fdtype

    def test_cube_centers_dtype(self, single_cam_mesher):
        """_cube_centers output should match the compute dtype."""
        coords = torch.zeros((4, 3), dtype=torch.int64, device=single_cam_mesher.device)
        levels = torch.zeros(4, dtype=torch.int64, device=single_cam_mesher.device)
        centers = single_cam_mesher._cube_centers(coords, levels)
        assert centers.dtype == single_cam_mesher._fdtype

    def test_cube_corner_positions_dtype(self, single_cam_mesher):
        """_cube_corner_positions output should match the compute dtype."""
        coords = torch.zeros((4, 3), dtype=torch.int64, device=single_cam_mesher.device)
        levels = torch.zeros(4, dtype=torch.int64, device=single_cam_mesher.device)
        corners = single_cam_mesher._cube_corner_positions(coords, levels)
        assert corners.dtype == single_cam_mesher._fdtype


# ---------------------------------------------------------------------------
# use_compile parameter
# ---------------------------------------------------------------------------
class TestUseCompile:
    def test_use_compile_false_does_not_wrap(self, sample_cameras, sample_bounds):
        """use_compile=False should leave methods unwrapped."""
        mesher = TorchOcMesher(sample_cameras, sample_bounds, device="cpu", use_compile=False)
        # Methods should still be callable without error
        coords, levels = mesher._build_coarse_octree()
        assert coords.shape[1] == 3

    def test_use_compile_true_still_works(self, sample_cameras, sample_bounds, sphere_kernel):
        """use_compile=True should not break correctness on CPU (falls back gracefully)."""
        mesher = TorchOcMesher(sample_cameras, sample_bounds, device="cpu", use_compile=True)
        meshes, tags = mesher([sphere_kernel])
        assert len(meshes) == 1
        assert meshes[0].vertices.shape[0] > 0


# ---------------------------------------------------------------------------
# Refactoring: cam_heights/cam_widths are tuples (commit 1)
# ---------------------------------------------------------------------------
class TestCamDimensionTypes:
    def test_cam_heights_is_tuple(self, single_cam_mesher):
        """cam_heights must be a tuple for immutability and lower overhead."""
        assert isinstance(single_cam_mesher.cam_heights, tuple)

    def test_cam_widths_is_tuple(self, single_cam_mesher):
        """cam_widths must be a tuple for immutability and lower overhead."""
        assert isinstance(single_cam_mesher.cam_widths, tuple)

    def test_cam_heights_values_correct(self, single_cam_mesher):
        """cam_heights must preserve the correct integer values."""
        assert single_cam_mesher.cam_heights == (720,)

    def test_cam_widths_values_correct(self, single_cam_mesher):
        """cam_widths must preserve the correct integer values."""
        assert single_cam_mesher.cam_widths == (1280,)

    def test_multi_cam_heights_is_tuple(self, multi_cam_mesher):
        """Multi-camera mesher cam_heights must also be a tuple."""
        assert isinstance(multi_cam_mesher.cam_heights, tuple)
        assert len(multi_cam_mesher.cam_heights) == 4


# ---------------------------------------------------------------------------
# Refactoring: pre-allocated depth buffer (commit 3)
# ---------------------------------------------------------------------------
class TestDepthBufPreallocation:
    def test_depth_buf_exists(self, single_cam_mesher):
        """Pre-allocated depth buffer attribute must be present after init."""
        assert hasattr(single_cam_mesher, "_depth_buf")
        assert isinstance(single_cam_mesher._depth_buf, torch.Tensor)

    def test_depth_buf_dtype_matches_fdtype(self, single_cam_mesher):
        """Depth buffer dtype must match the compute dtype."""
        assert single_cam_mesher._depth_buf.dtype == single_cam_mesher._fdtype

    def test_depth_buf_size_correct(self, single_cam_mesher):
        """Depth buffer must hold at least the max reduced-resolution camera image."""
        factor = 10.0
        expected = max(
            max(1, int(h / factor)) * max(1, int(w / factor))
            for h, w in zip(single_cam_mesher.cam_heights, single_cam_mesher.cam_widths, strict=True)
        )
        assert single_cam_mesher._depth_buf.shape[0] == expected

    def test_visibility_filter_with_preallocated_buf(self, single_cam_mesher):
        """_visibility_filter must still produce correct shape with pre-allocated buf."""
        positions = torch.zeros(8, 3, dtype=torch.float64, device=single_cam_mesher.device)
        result = single_cam_mesher._visibility_filter(positions)
        assert result.shape == (8,)
        assert result.dtype == torch.bool


# ---------------------------------------------------------------------------
# Refactoring: int16 tri_table in MC cache (commit 5)
# ---------------------------------------------------------------------------
class TestMCCacheInt16:
    def test_tri_table_dtype_is_int16(self, single_cam_mesher):
        """tri_table in MC cache must use int16 to reduce memory bandwidth."""
        cache = TorchOcMesher._mc_cache[single_cam_mesher.device]
        assert cache["tri_table"].dtype == torch.int16

    def test_tri_table_values_in_valid_range(self, single_cam_mesher):
        """int16 tri_table values must be in [-1, 11] (sentinel=-1, edges 0-11)."""
        cache = TorchOcMesher._mc_cache[single_cam_mesher.device]
        tt = cache["tri_table"]
        assert int(tt.min().item()) >= -1
        assert int(tt.max().item()) <= 11

    def test_mc_produces_correct_mesh_with_int16_table(self, single_cam_mesher):
        """Marching cubes with int16 tri_table must still extract a sphere mesh."""
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
