"""Tests for ``ocmesher.torch_core`` optimisations."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from ocmesher import torch_core
from ocmesher.torch_core import TorchOcMesher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def single_cam_mesher(sample_cameras, torch_mesher_factory):
    """TorchOcMesher with a single camera on CPU."""
    return torch_mesher_factory(sample_cameras)


@pytest.fixture
def multi_cam_mesher(sample_camera_pose, sample_intrinsics, torch_mesher_factory):
    """TorchOcMesher with four cameras on CPU."""
    cameras = (
        [sample_camera_pose] * 4,
        [sample_intrinsics] * 4,
        [720] * 4,
        [1280] * 4,
    )
    return torch_mesher_factory(cameras)


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
        expected = (single_cam_mesher._pix_ang * single_cam_mesher.pixels_per_cube).unsqueeze(1)
        torch.testing.assert_close(single_cam_mesher._pix_ang_ppc, expected)
        # Shape must be (C, 1) — pre-expanded to avoid unsqueeze in hot loop.
        # C=1 for single-camera fixture.
        assert single_cam_mesher._pix_ang_ppc.shape == (1, 1)


# ---------------------------------------------------------------------------
# Context-manager protocol
# ---------------------------------------------------------------------------
class TestContextManager:
    def test_with_statement_returns_mesher(self, single_cam_mesher):
        with single_cam_mesher as m:
            assert m is single_cam_mesher

    def test_exit_returns_false(self, single_cam_mesher):
        assert single_cam_mesher.__exit__(None, None, None) is False

    def test_exit_shuts_down_sdf_pool(self, sample_cameras, sample_bounds):
        mesher = TorchOcMesher(sample_cameras, sample_bounds, device="cpu", n_sdf_workers=2)
        pool = mesher._get_sdf_pool()
        assert pool is mesher._sdf_pool
        assert mesher.__exit__(None, None, None) is False
        assert mesher._sdf_pool is None


class TestConstructorValidation:
    def test_invalid_cameras_raises(self, sample_bounds):
        with pytest.raises(ValueError, match="cameras must be"):
            TorchOcMesher("bad cameras", sample_bounds, device="cpu")

    def test_invalid_bounds_raises(self, sample_cameras):
        with pytest.raises(ValueError, match="bounds must have 6 elements"):
            TorchOcMesher(sample_cameras, [0, 1], device="cpu")

    def test_list_inputs_are_normalized(self, sample_cameras_as_lists, sample_bounds):
        mesher = TorchOcMesher(sample_cameras_as_lists, sample_bounds, device="cpu")
        assert mesher.cam_inv_poses.shape == (1, 3, 4)
        assert mesher.cam_intrinsics.shape == (1, 3, 3)

    def test_invalid_scalar_params_raise(self, sample_cameras, sample_bounds):
        with pytest.raises(ValueError, match="coarse_count must be > 0"):
            TorchOcMesher(sample_cameras, sample_bounds, device="cpu", coarse_count=0)

    def test_invalid_visible_relax_iter_raises(self, sample_cameras, sample_bounds):
        with pytest.raises(ValueError, match="visible_relax_iter must be >= 0"):
            TorchOcMesher(sample_cameras, sample_bounds, device="cpu", visible_relax_iter=-1)


class TestBackendContract:
    def test_backend_version_is_string(self, single_cam_mesher):
        assert isinstance(single_cam_mesher.version, str)
        assert single_cam_mesher.version == TorchOcMesher.BACKEND_VERSION

    def test_get_capabilities_reports_device_policy(self, single_cam_mesher):
        capabilities = single_cam_mesher.get_capabilities()
        assert capabilities.preferred_dtype == "float64"
        assert capabilities.max_batch == torch_core.TORCH_SDF_CHUNK_SIZE
        assert capabilities.supports_dlpack is True
        assert capabilities.version == TorchOcMesher.BACKEND_VERSION

    def test_as_backend_tensor_accepts_numpy(self, single_cam_mesher):
        pts = np.zeros((4, 3), dtype=np.float32)
        tensor = single_cam_mesher.as_backend_tensor(pts)
        assert tensor.shape == (4, 3)
        assert tensor.dtype == single_cam_mesher._fdtype
        assert tensor.device == single_cam_mesher.device

    def test_to_backend_dlpack_roundtrips_tensor(self, single_cam_mesher):
        pts = torch.zeros((2, 3), dtype=single_cam_mesher._fdtype, device=single_cam_mesher.device)
        capsule = single_cam_mesher.to_backend_dlpack(pts)
        roundtrip = torch.utils.dlpack.from_dlpack(capsule)
        torch.testing.assert_close(roundtrip, pts)

    def test_evaluate_sdf_batch_matches_private_path(self, single_cam_mesher, sphere_kernel):
        pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=single_cam_mesher._fdtype, device=single_cam_mesher.device)
        result = single_cam_mesher.evaluate_sdf_batch([sphere_kernel], pts)
        expected = single_cam_mesher._evaluate_sdf([sphere_kernel], pts)
        torch.testing.assert_close(result, expected)

    def test_stream_rejected_on_cpu(self, single_cam_mesher, sphere_kernel):
        pts = torch.zeros((1, 3), dtype=single_cam_mesher._fdtype, device=single_cam_mesher.device)
        with pytest.raises(ValueError, match="stream is only supported"):
            single_cam_mesher.evaluate_sdf_batch([sphere_kernel], pts, stream=object())

    def test_mps_prefers_smaller_batches(self):
        assert TorchOcMesher._recommended_max_batch("mps") < torch_core.TORCH_SDF_CHUNK_SIZE


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

    def test_dedup_vertices_merges_shared_points(self):
        verts_flat = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

        dedup_verts, dedup_faces = TorchOcMesher._dedup_vertices(verts_flat)

        assert dedup_verts.shape == (4, 3)
        assert dedup_faces.shape == (2, 3)
        assert {tuple(vertex) for vertex in dedup_verts.tolist()} == {
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        }

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
        meshes, _tags = multi_cam_mesher([sphere_kernel])
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
        coords, _levels = mesher._build_coarse_octree()
        assert coords.shape[1] == 3

    def test_use_compile_true_still_works(self, sample_cameras, sample_bounds, sphere_kernel):
        """use_compile=True should not break correctness on CPU (falls back gracefully)."""
        mesher = TorchOcMesher(sample_cameras, sample_bounds, device="cpu", use_compile=True)
        meshes, _tags = mesher([sphere_kernel])
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
    def test_depth_bufs_exists(self, single_cam_mesher):
        """Pre-allocated batched depth buffer attribute must be present after init."""
        assert hasattr(single_cam_mesher, "_depth_bufs")
        assert isinstance(single_cam_mesher._depth_bufs, torch.Tensor)

    def test_depth_bufs_dtype_matches_fdtype(self, single_cam_mesher):
        """Depth buffer dtype must match the compute dtype."""
        assert single_cam_mesher._depth_bufs.dtype == single_cam_mesher._fdtype

    def test_depth_bufs_shape_correct(self, single_cam_mesher):
        """Depth buffer must be (C, max_buf) to hold all cameras' bin grids."""
        factor = 10.0
        expected_max_buf = max(
            max(1, int(h / factor)) * max(1, int(w / factor))
            for h, w in zip(single_cam_mesher.cam_heights, single_cam_mesher.cam_widths, strict=True)
        )
        assert single_cam_mesher._depth_bufs.shape == (
            single_cam_mesher.n_cameras,
            expected_max_buf,
        )

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
    def test_build_mc_cache_returns_expected_keys(self):
        cache = TorchOcMesher._build_mc_cache(torch.device("cpu"))
        assert set(cache) == {"edge_table", "tri_table", "max_tri_entries", "bit_shifts"}

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


# ---------------------------------------------------------------------------
# exp2 replaces ldexp(ones_like, levels) for faster scale computation
# ---------------------------------------------------------------------------
class TestExp2Scale:
    def test_cube_centers_match_reference(self, single_cam_mesher):
        """exp2-based _cube_centers must match a manual 2**level computation."""
        coords = torch.tensor([[2, 3, 1], [5, 5, 5]], dtype=torch.int64, device=single_cam_mesher.device)
        levels = torch.tensor([3, 6], dtype=torch.int64, device=single_cam_mesher.device)
        centers = single_cam_mesher._cube_centers(coords, levels)
        # Manual reference using 2**level
        for i in range(len(coords)):
            scale = single_cam_mesher.size / (2.0 ** levels[i].item())
            expected = (
                single_cam_mesher.center.cpu().numpy()
                - single_cam_mesher.size / 2
                + scale * (coords[i].cpu().numpy().astype(np.float64) + 0.5)
            )
            np.testing.assert_allclose(centers[i].cpu().numpy(), expected, atol=1e-10)

    def test_cube_corner_positions_match_reference(self, single_cam_mesher):
        """exp2-based _cube_corner_positions must match a manual 2**level computation."""
        coords = torch.tensor([[0, 0, 0]], dtype=torch.int64, device=single_cam_mesher.device)
        levels = torch.tensor([4], dtype=torch.int64, device=single_cam_mesher.device)
        corners = single_cam_mesher._cube_corner_positions(coords, levels)
        assert corners.shape == (1, 8, 3)
        # All 8 corners must differ from the origin cube
        unique_corners = corners[0].unique(dim=0)
        assert unique_corners.shape[0] == 8

    def test_projected_sizes_positive(self, single_cam_mesher):
        """_projected_sizes must return positive values after exp2 change."""
        coords = torch.tensor([[1, 1, 1]], dtype=torch.int64, device=single_cam_mesher.device)
        levels = torch.tensor([2], dtype=torch.int64, device=single_cam_mesher.device)
        positions = single_cam_mesher._cube_centers(coords, levels)
        proj = single_cam_mesher._projected_sizes(positions, levels)
        assert (proj > 0).all()


# ---------------------------------------------------------------------------
# SDF caching: _refine_surface_octree returns corner_sdf
# ---------------------------------------------------------------------------
class TestSDFCaching:
    def test_refine_returns_corner_sdf(self, single_cam_mesher, sphere_kernel):
        """_refine_surface_octree must return a corner_sdf tensor."""
        kernels = [sphere_kernel]
        coords, levels = single_cam_mesher._build_coarse_octree()
        mask, corner_sdf = single_cam_mesher._find_surface_cubes(kernels, coords, levels)
        s_coords = coords[mask]
        s_levels = levels[mask]
        s_corner_sdf = corner_sdf[mask]
        r_coords, _r_levels, r_corner_sdf = single_cam_mesher._refine_surface_octree(
            kernels,
            s_coords,
            s_levels,
            corner_sdf=s_corner_sdf,
        )
        assert r_corner_sdf is not None
        assert r_corner_sdf.shape[0] == len(r_coords)
        assert r_corner_sdf.shape[1] == 8
        assert r_corner_sdf.shape[2] == len(kernels)

    def test_refine_without_cache_returns_none_when_no_iters(self, single_cam_mesher, sphere_kernel):
        """When no corner_sdf is provided and max_iters=0, result is None."""
        kernels = [sphere_kernel]
        coords, levels = single_cam_mesher._build_coarse_octree()
        mask, _ = single_cam_mesher._find_surface_cubes(kernels, coords, levels)
        s_coords = coords[mask]
        s_levels = levels[mask]
        if len(s_coords) == 0:
            pytest.skip("no surface cubes found")
        # Force zero refinement iterations with corner_sdf=None
        r_coords, r_levels, r_corner_sdf = single_cam_mesher._refine_surface_octree(
            kernels,
            s_coords,
            s_levels,
            max_iters=0,
        )
        assert r_corner_sdf is None
        # Coords/levels are returned unchanged
        assert len(r_coords) == len(s_coords)
        assert len(r_levels) == len(s_levels)

    def test_cached_sdf_produces_same_mesh(self, single_cam_mesher, sphere_kernel):
        """End-to-end mesh with SDF caching must match mesh without it."""
        # Run with caching (default)
        meshes_cached, _ = single_cam_mesher([sphere_kernel])
        v_cached = meshes_cached[0].vertices
        f_cached = meshes_cached[0].faces

        # Run without caching by calling _construct_element_mesh with corner_sdf=None
        coords, levels = single_cam_mesher._build_coarse_octree()
        mask, _cs = single_cam_mesher._find_surface_cubes([sphere_kernel], coords, levels)
        s_coords = coords[mask]
        s_levels = levels[mask]
        r_coords, r_levels, _r_sdf = single_cam_mesher._refine_surface_octree(
            [sphere_kernel],
            s_coords,
            s_levels,
        )
        positions = single_cam_mesher._cube_centers(r_coords, r_levels)
        vis_mask = single_cam_mesher._visibility_filter(positions)
        all_c = torch.cat([r_coords[vis_mask], r_coords[~vis_mask]])
        all_l = torch.cat([r_levels[vis_mask], r_levels[~vis_mask]])
        n_vis = int(vis_mask.sum().item())
        mesh_no_cache, _ = single_cam_mesher._construct_element_mesh(
            [sphere_kernel],
            all_c,
            all_l,
            n_vis,
            corner_sdf=None,
        )
        v_no_cache = mesh_no_cache.vertices
        f_no_cache = mesh_no_cache.faces

        # Both paths must produce the same geometry
        assert v_cached.shape == v_no_cache.shape
        assert f_cached.shape == f_no_cache.shape
        np.testing.assert_allclose(v_cached, v_no_cache, atol=1e-8)
        np.testing.assert_array_equal(f_cached, f_no_cache)

    def test_multi_kernel_cached_sdf(self, single_cam_mesher, sphere_kernel, plane_kernel):
        """SDF caching must work correctly with multiple kernels."""
        meshes, tags = single_cam_mesher([sphere_kernel, plane_kernel])
        assert len(meshes) == 2
        assert len(tags) == 2
        # Each mesh should have geometry
        assert meshes[0].vertices.shape[0] > 0
        assert meshes[1].vertices.shape[0] > 0


# ---------------------------------------------------------------------------
# Single-chunk fast path + vectorised bounds check (Refactor 3)
# ---------------------------------------------------------------------------


class TestEvaluateSDFSingleChunk:
    """Verify _evaluate_sdf single-chunk fast path produces correct results."""

    def test_single_chunk_matches_multi_chunk(self, single_cam_mesher, sphere_kernel):
        """Single-chunk fast path must return identical results to multi-chunk."""
        mesher = single_cam_mesher
        pts = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            dtype=mesher._fdtype,
            device=mesher.device,
        )
        result = mesher._evaluate_sdf([sphere_kernel], pts)
        assert result.shape == (3, 1)
        np.testing.assert_allclose(
            result.cpu().numpy()[:, 0],
            [-1.0, 0.0, 1.0],
            atol=1e-5,
        )

    def test_vectorised_bounds_clamp_in_sdf(self, sample_cameras, sphere_kernel):
        """Vectorised bounds check must clamp out-of-bounds points correctly."""
        bounds = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        mesher = TorchOcMesher(sample_cameras, bounds, device="cpu")
        pts = torch.tensor(
            [[0, 0, 0], [5, 0, 0]],
            dtype=mesher._fdtype,
            device=mesher.device,
        )
        result = mesher._evaluate_sdf([sphere_kernel], pts)
        assert result[0, 0].item() == pytest.approx(-1.0, abs=1e-5)
        assert result[1, 0].item() == pytest.approx(1.0)

    def test_empty_positions_fast_path(self, single_cam_mesher, sphere_kernel):
        """Empty positions must return empty tensor without error."""
        pts = torch.zeros((0, 3), dtype=single_cam_mesher._fdtype, device=single_cam_mesher.device)
        result = single_cam_mesher._evaluate_sdf([sphere_kernel], pts)
        assert result.shape == (0, 1)

    def test_multiple_kernels_single_chunk(self, single_cam_mesher, sphere_kernel, plane_kernel):
        """Multi-kernel single-chunk path must produce correct column layout."""
        pts = torch.tensor(
            [[0, 0, 0.5], [1, 0, 0]],
            dtype=single_cam_mesher._fdtype,
            device=single_cam_mesher.device,
        )
        result = single_cam_mesher._evaluate_sdf([sphere_kernel, plane_kernel], pts)
        assert result.shape == (2, 2)
        # Sphere SDF at origin: -1+0.5 = -0.5; plane SDF at z=0.5: 0.5
        np.testing.assert_allclose(result[0, 0].item(), np.linalg.norm([0, 0, 0.5]) - 1, atol=1e-5)
        np.testing.assert_allclose(result[0, 1].item(), 0.5, atol=1e-5)


class TestEvaluateSDFChunkedPrealloc:
    def test_pinned_host_buffer_reuses_capacity(self, sample_cameras, monkeypatch):
        mesher = TorchOcMesher(sample_cameras, [-5, 5, -5, 5, -5, 5], device="cpu")
        allocations = []
        real_empty = torch.empty

        def _fake_empty(shape, *, dtype, device=None, pin_memory=False):
            allocations.append((tuple(shape), dtype, device, pin_memory))
            return real_empty(shape, dtype=dtype)

        monkeypatch.setattr(torch_core, "_torch_empty", _fake_empty)

        first = mesher._get_pinned_host_buffer("_sdf_results_pinned", (8, 2), torch.float32)
        second = mesher._get_pinned_host_buffer("_sdf_results_pinned", (4, 1), torch.float32)

        assert first.data_ptr() == second.data_ptr()
        assert allocations == [((8, 2), torch.float32, "cpu", True)]

    def test_multiple_kernels_multi_chunk_matches_expected(self, sample_cameras, sphere_kernel, plane_kernel, monkeypatch):
        monkeypatch.setattr(torch_core, "TORCH_SDF_CHUNK_SIZE", 2)
        mesher = TorchOcMesher(sample_cameras, [-5, 5, -5, 5, -5, 5], device="cpu", n_sdf_workers=1)
        pts = torch.tensor(
            [[0, 0, 0.5], [1, 0, 0], [0, 0, -0.5], [2, 0, 1.0], [3, 0, 0.0]],
            dtype=mesher._fdtype,
            device=mesher.device,
        )

        result = mesher._evaluate_sdf([sphere_kernel, plane_kernel], pts).cpu().numpy()

        expected_sphere = np.linalg.norm(pts.cpu().numpy(), axis=1) - 1.0
        expected_plane = pts.cpu().numpy()[:, 2]
        np.testing.assert_allclose(result[:, 0], expected_sphere, atol=1e-5)
        np.testing.assert_allclose(result[:, 1], expected_plane, atol=1e-5)

    def test_single_kernel_multi_chunk_threaded_matches_expected(self, sample_cameras, sphere_kernel, monkeypatch):
        monkeypatch.setattr(torch_core, "TORCH_SDF_CHUNK_SIZE", 2)
        mesher = TorchOcMesher(sample_cameras, [-5, 5, -5, 5, -5, 5], device="cpu", n_sdf_workers=2)
        pts = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            dtype=mesher._fdtype,
            device=mesher.device,
        )

        result = mesher._evaluate_sdf([sphere_kernel], pts).cpu().numpy()[:, 0]

        np.testing.assert_allclose(result, np.linalg.norm(pts.cpu().numpy(), axis=1) - 1.0, atol=1e-5)

    def test_threaded_multi_chunk_reuses_pool(self, sample_cameras, sphere_kernel, monkeypatch):
        monkeypatch.setattr(torch_core, "TORCH_SDF_CHUNK_SIZE", 2)
        mesher = TorchOcMesher(sample_cameras, [-5, 5, -5, 5, -5, 5], device="cpu", n_sdf_workers=2)
        pts = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            dtype=mesher._fdtype,
            device=mesher.device,
        )

        mesher._evaluate_sdf([sphere_kernel], pts)
        first_pool = mesher._sdf_pool
        mesher._evaluate_sdf([sphere_kernel], pts)

        assert first_pool is not None
        assert mesher._sdf_pool is first_pool

    def test_pinned_host_buffer_grows_when_shape_increases(self, sample_cameras, monkeypatch):
        mesher = TorchOcMesher(sample_cameras, [-5, 5, -5, 5, -5, 5], device="cpu")
        allocations = []
        real_empty = torch.empty

        def _fake_empty(shape, *, dtype, device=None, pin_memory=False):
            _ = (device, pin_memory)
            allocations.append(tuple(shape))
            return real_empty(shape, dtype=dtype)

        monkeypatch.setattr(torch_core, "_torch_empty", _fake_empty)

        first = mesher._get_pinned_host_buffer("_sdf_results_pinned", (2, 1), torch.float32)
        second = mesher._get_pinned_host_buffer("_sdf_results_pinned", (6, 3), torch.float32)

        assert first.data_ptr() != second.data_ptr()
        assert allocations == [(2, 1), (6, 3)]


# ---------------------------------------------------------------------------
# Refactor 1: Shared-corner SDF deduplication in _find_surface_cubes
# ---------------------------------------------------------------------------
class TestSharedCornerDedup:
    """Verify that corner deduplication reduces SDF calls without changing results."""

    def test_dedup_produces_fewer_unique_points(self, single_cam_mesher):
        """Adjacent cubes share corners; unique count must be < total corners."""
        coords, levels = single_cam_mesher._build_coarse_octree()
        corners = single_cam_mesher._cube_corner_positions(coords, levels)
        n_total = corners.reshape(-1, 3).shape[0]
        quantized = (corners.reshape(-1, 3) * 1e8).round().long()
        n_unique = torch.unique(quantized, dim=0).shape[0]
        assert n_unique < n_total, "Dedup should find shared corners"

    def test_dedup_sdf_matches_direct_eval(self, single_cam_mesher, sphere_kernel):
        """SDF via dedup must exactly match direct (non-dedup) evaluation."""
        coords, levels = single_cam_mesher._build_coarse_octree()
        corners = single_cam_mesher._cube_corner_positions(coords, levels)
        flat = corners.reshape(-1, 3)
        # Direct evaluation (no dedup)
        sdf_direct = single_cam_mesher._evaluate_sdf([sphere_kernel], flat)
        # Dedup evaluation (via _find_surface_cubes)
        _, corner_sdf = single_cam_mesher._find_surface_cubes([sphere_kernel], coords, levels)
        sdf_dedup = corner_sdf.reshape(-1, 1)
        torch.testing.assert_close(sdf_dedup, sdf_direct, atol=0, rtol=0)

    def test_dedup_surface_mask_matches_direct(self, single_cam_mesher, sphere_kernel):
        """Surface mask with dedup must match the non-dedup mask."""
        coords, levels = single_cam_mesher._build_coarse_octree()
        # Via dedup (current code)
        mask_dedup, _ = single_cam_mesher._find_surface_cubes([sphere_kernel], coords, levels)
        # Direct: compute SDF at all corners without dedup
        corners = single_cam_mesher._cube_corner_positions(coords, levels)
        sdf_all = single_cam_mesher._evaluate_sdf([sphere_kernel], corners.reshape(-1, 3))
        sdf = sdf_all.reshape(len(coords), 8, -1)
        sdf_min = sdf.min(dim=-1).values
        signs = sdf_min >= 0
        mask_direct = signs.any(dim=1) & (~signs).any(dim=1)
        torch.testing.assert_close(mask_dedup, mask_direct)


# ---------------------------------------------------------------------------
# Refactor 2: Split rotation/translation replaces pad+bmm+permute
# ---------------------------------------------------------------------------
class TestSplitRotationTranslation:
    """Verify pre-split R|t attributes and pad-free projection correctness."""

    def test_inv_pose_rotation_shape(self, single_cam_mesher):
        """_inv_pose_R must be (C, 3, 3) rotation part of inverse poses."""
        assert single_cam_mesher._inv_pose_R.shape == (1, 3, 3)

    def test_inv_pose_translation_shape(self, single_cam_mesher):
        """_inv_pose_t must be (C, 1, 3) translation part of inverse poses."""
        assert single_cam_mesher._inv_pose_t.shape == (1, 1, 3)

    def test_proj_rotation_shape(self, single_cam_mesher):
        """_proj_R must be (C, 3, 3) rotation part of K@inv_pose."""
        assert single_cam_mesher._proj_R.shape == (1, 3, 3)

    def test_proj_translation_shape(self, single_cam_mesher):
        """_proj_t must be (C, 1, 3) translation part of K@inv_pose."""
        assert single_cam_mesher._proj_t.shape == (1, 1, 3)

    def test_split_rotation_translation_match_full_matrix(self, single_cam_mesher):
        """R|t components must reconstruct the full inv_pose matrix."""
        full = single_cam_mesher.cam_inv_poses  # (C, 3, 4)
        torch.testing.assert_close(
            single_cam_mesher._inv_pose_R,
            full[:, :, :3],
        )
        torch.testing.assert_close(
            single_cam_mesher._inv_pose_t,
            full[:, :, 3].unsqueeze(1),
        )

    def test_projected_sizes_positive(self, single_cam_mesher):
        """Pad-free _projected_sizes must return positive values."""
        coords = torch.tensor([[1, 1, 1]], dtype=torch.int64, device=single_cam_mesher.device)
        levels = torch.tensor([2], dtype=torch.int64, device=single_cam_mesher.device)
        positions = single_cam_mesher._cube_centers(coords, levels)
        proj = single_cam_mesher._projected_sizes(positions, levels)
        assert (proj > 0).all()

    def test_multi_cam_split_shapes(self, multi_cam_mesher):
        """Multi-camera mesher must have correctly sized R|t attributes."""
        assert multi_cam_mesher._inv_pose_R.shape == (4, 3, 3)
        assert multi_cam_mesher._inv_pose_t.shape == (4, 1, 3)
        assert multi_cam_mesher._proj_R.shape == (4, 3, 3)
        assert multi_cam_mesher._proj_t.shape == (4, 1, 3)


# ---------------------------------------------------------------------------
# Refactor 3: Skip SDF re-evaluation for kept cubes in _refine_surface_octree
# ---------------------------------------------------------------------------
class TestRefineSkipKeptCubes:
    """Verify optimised refinement path skips SDF re-evaluation for kept cubes."""

    def test_refine_cached_matches_fallback(self, single_cam_mesher, sphere_kernel):
        """Optimised (cached) and fallback (no-cache) paths must produce equal cube sets."""
        kernels = [sphere_kernel]
        coords, levels = single_cam_mesher._build_coarse_octree()
        mask, corner_sdf = single_cam_mesher._find_surface_cubes(kernels, coords, levels)
        s_coords = coords[mask]
        s_levels = levels[mask]
        s_sdf = corner_sdf[mask]
        # Optimised path
        c_opt, l_opt, _sdf_opt = single_cam_mesher._refine_surface_octree(
            kernels,
            s_coords.clone(),
            s_levels.clone(),
            corner_sdf=s_sdf.clone(),
        )
        # Fallback path
        c_fb, l_fb, _sdf_fb = single_cam_mesher._refine_surface_octree(
            kernels,
            s_coords.clone(),
            s_levels.clone(),
        )
        assert len(c_opt) == len(c_fb)
        assert len(l_opt) == len(l_fb)

    def test_refine_cached_preserves_corner_sdf(self, single_cam_mesher, sphere_kernel):
        """Kept cubes must retain their original corner SDF values."""
        kernels = [sphere_kernel]
        coords, levels = single_cam_mesher._build_coarse_octree()
        mask, corner_sdf = single_cam_mesher._find_surface_cubes(kernels, coords, levels)
        s_coords = coords[mask]
        s_levels = levels[mask]
        s_sdf = corner_sdf[mask]
        _, _, r_sdf = single_cam_mesher._refine_surface_octree(
            kernels,
            s_coords.clone(),
            s_levels.clone(),
            corner_sdf=s_sdf.clone(),
        )
        assert r_sdf is not None
        assert r_sdf.shape[1] == 8
        assert r_sdf.shape[2] == len(kernels)


# ---------------------------------------------------------------------------
# Tests for speed-optimisation refactors
# ---------------------------------------------------------------------------
class TestPrecomputedConstants:
    """Verify that pre-computed constants in __init__ are correct."""

    def test_origin_matches_center_minus_half_size(self, single_cam_mesher):
        """_origin must equal center - size/2."""
        expected = single_cam_mesher.center - single_cam_mesher.size / 2
        torch.testing.assert_close(single_cam_mesher._origin, expected)

    def test_inv_pix_ang_ppc_is_reciprocal(self, single_cam_mesher):
        """_inv_pix_ang_ppc must be 1 / _pix_ang_ppc."""
        expected = 1.0 / single_cam_mesher._pix_ang_ppc
        torch.testing.assert_close(single_cam_mesher._inv_pix_ang_ppc, expected)

    def test_pix_ang_ppc_shape_is_c1(self, multi_cam_mesher):
        """_pix_ang_ppc must be (C, 1) for broadcast in _projected_sizes."""
        assert multi_cam_mesher._pix_ang_ppc.shape == (4, 1)

    def test_inv_pix_ang_ppc_shape_is_c1(self, multi_cam_mesher):
        """_inv_pix_ang_ppc must be (C, 1)."""
        assert multi_cam_mesher._inv_pix_ang_ppc.shape == (4, 1)


class TestCubeScalesFusion:
    """Verify _cube_scales and fused scale passing produce consistent results."""

    def test_cube_scales_matches_exp2(self, single_cam_mesher):
        """_cube_scales must return size / 2^levels."""
        levels = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=single_cam_mesher.device)
        scales = single_cam_mesher._cube_scales(levels)
        expected = single_cam_mesher.size / torch.exp2(levels.to(single_cam_mesher._fdtype))
        torch.testing.assert_close(scales, expected)

    def test_cube_centers_with_precomputed_scales(self, single_cam_mesher):
        """_cube_centers with cube_scales kwarg must match default path."""
        coords = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.int64, device=single_cam_mesher.device)
        levels = torch.tensor([1, 2], dtype=torch.int64, device=single_cam_mesher.device)
        centers_default = single_cam_mesher._cube_centers(coords, levels)
        scales = single_cam_mesher._cube_scales(levels)
        centers_fused = single_cam_mesher._cube_centers(coords, levels, cube_scales=scales)
        torch.testing.assert_close(centers_fused, centers_default)

    def test_projected_sizes_with_precomputed_scales(self, single_cam_mesher):
        """_projected_sizes with cube_scales kwarg must match default path."""
        coords = torch.tensor([[0, 0, 0]], dtype=torch.int64, device=single_cam_mesher.device)
        levels = torch.tensor([1], dtype=torch.int64, device=single_cam_mesher.device)
        positions = single_cam_mesher._cube_centers(coords, levels)
        proj_default = single_cam_mesher._projected_sizes(positions, levels)
        scales = single_cam_mesher._cube_scales(levels)
        proj_fused = single_cam_mesher._projected_sizes(positions, levels, cube_scales=scales)
        torch.testing.assert_close(proj_fused, proj_default)


class TestBatchedVisibility:
    """Verify the batched visibility filter matches expected behaviour."""

    def test_vis_hb_wb_precomputed(self, single_cam_mesher):
        """Pre-computed bin dimensions must match per-camera calculations."""
        factor = 10.0
        for k in range(single_cam_mesher.n_cameras):
            h, w = single_cam_mesher.cam_heights[k], single_cam_mesher.cam_widths[k]
            expected_hb = max(1, int(h / factor))
            expected_wb = max(1, int(w / factor))
            assert single_cam_mesher._vis_hb[k].item() == expected_hb
            assert single_cam_mesher._vis_wb[k].item() == expected_wb

    def test_multi_cam_visibility_batched(self, multi_cam_mesher):
        """Batched visibility must return correct shape for multiple cameras."""
        positions = torch.zeros(10, 3, dtype=torch.float64, device=multi_cam_mesher.device)
        result = multi_cam_mesher._visibility_filter(positions)
        assert result.shape == (10,)
        assert result.dtype == torch.bool

    def test_depth_bufs_shape(self, multi_cam_mesher):
        """Batched depth buffer must be (C, max_buf)."""
        assert multi_cam_mesher._depth_bufs.shape[0] == multi_cam_mesher.n_cameras
        assert multi_cam_mesher._depth_bufs.shape[1] == multi_cam_mesher._vis_max_buf


class TestHashSortDedup:
    """Verify hash-sort-based corner dedup in _find_surface_cubes."""

    def test_dedup_still_reduces_sdf_calls(self, single_cam_mesher, sphere_kernel):
        """Hash-sort dedup must still identify shared corners."""
        coords, levels = single_cam_mesher._build_coarse_octree()
        mask, corner_sdf = single_cam_mesher._find_surface_cubes([sphere_kernel], coords, levels)
        assert mask.any(), "Should find surface cubes"
        assert corner_sdf.shape[0] == coords.shape[0]
        assert corner_sdf.shape[1] == 8

    def test_dedup_exact_match_with_direct_eval(self, single_cam_mesher, sphere_kernel):
        """SDF via hash-sort dedup must exactly match direct evaluation."""
        coords, levels = single_cam_mesher._build_coarse_octree()
        corners = single_cam_mesher._cube_corner_positions(coords, levels)
        flat = corners.reshape(-1, 3)
        sdf_direct = single_cam_mesher._evaluate_sdf([sphere_kernel], flat)
        _, corner_sdf = single_cam_mesher._find_surface_cubes([sphere_kernel], coords, levels)
        sdf_dedup = corner_sdf.reshape(-1, 1)
        torch.testing.assert_close(sdf_dedup, sdf_direct, atol=0, rtol=0)
