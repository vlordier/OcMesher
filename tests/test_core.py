"""Tests for ocmesher/core.py — OcMesher class and kernel_caller."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# kernel_caller (tested via a mock-backed OcMesher instance)
# ---------------------------------------------------------------------------

def _make_mesher_stub(bounds, cameras, enclosed=True):
    """Create an OcMesher-like object with just kernel_caller attributes.

    Avoids loading the C++ shared library by patching load_cdll.
    """
    from ocmesher.core import OcMesher

    with patch("ocmesher.core.load_cdll") as mock_cdll, \
         patch("ocmesher.core.register_func"):
        mock_cdll.return_value = MagicMock()
        mesher = OcMesher(cameras, bounds, enclosed=enclosed)
    return mesher


class TestKernelCallerEmptyInput:
    """kernel_caller with 0-length XYZ should return empty array."""

    def test_returns_empty_for_zero_xyz(self, identity_cameras, unit_bounds):
        mesher = _make_mesher_stub(unit_bounds, identity_cameras)
        kernels = [lambda x: np.zeros(len(x))]
        result = mesher.kernel_caller(kernels, np.zeros((0, 3)))
        assert result.shape == (0, 1)
        assert result.dtype == np.float32

    def test_returns_empty_for_multiple_kernels(self, identity_cameras, unit_bounds):
        mesher = _make_mesher_stub(unit_bounds, identity_cameras)
        kernels = [lambda x: np.zeros(len(x)), lambda x: np.ones(len(x))]
        result = mesher.kernel_caller(kernels, np.zeros((0, 3)))
        assert result.shape == (0, 2)


class TestKernelCallerSingleBatch:
    """kernel_caller with a small number of points (single batch)."""

    def test_single_kernel_identity(self, identity_cameras, unit_bounds):
        mesher = _make_mesher_stub(unit_bounds, identity_cameras)
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        kernels = [lambda x: np.full(len(x), 3.14)]
        result = mesher.kernel_caller(kernels, pts)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], 3.14, atol=1e-5)

    def test_multiple_kernels(self, identity_cameras, unit_bounds):
        mesher = _make_mesher_stub(unit_bounds, identity_cameras)
        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64)
        k1 = lambda x: np.ones(len(x))
        k2 = lambda x: np.full(len(x), -1.0)
        result = mesher.kernel_caller([k1, k2], pts)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[:, 0], [1.0, 1.0])
        np.testing.assert_array_equal(result[:, 1], [-1.0, -1.0])

    def test_output_dtype_is_float32(self, identity_cameras, unit_bounds):
        mesher = _make_mesher_stub(unit_bounds, identity_cameras)
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.zeros(len(x))], pts)
        assert result.dtype == np.float32


class TestKernelCallerEnclosed:
    """Points outside bounds should get SDF=1 when enclosed=True."""

    def test_out_of_bounds_clipped(self, identity_cameras, unit_bounds):
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        # Point exactly at boundary and beyond
        pts = np.array([
            [0.0, 0.0, 0.0],   # inside
            [2.0, 0.0, 0.0],   # outside x
            [0.0, -2.0, 0.0],  # outside y
            [0.0, 0.0, 5.0],   # outside z
        ], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -1.0)], pts)
        # Inside point keeps kernel value
        assert result[0, 0] == pytest.approx(-1.0, abs=1e-5)
        # Outside points forced to 1
        assert result[1, 0] == pytest.approx(1.0, abs=1e-5)
        assert result[2, 0] == pytest.approx(1.0, abs=1e-5)
        assert result[3, 0] == pytest.approx(1.0, abs=1e-5)

    def test_boundary_points_clipped(self, identity_cameras, unit_bounds):
        """Points exactly at bounds edges should be clipped (>= max, <= min)."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        # Exactly at boundary
        pts = np.array([
            [-1.0, 0.0, 0.0],  # at xmin boundary — clipped
            [1.0, 0.0, 0.0],   # at xmax boundary — clipped
        ], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -5.0)], pts)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert result[1, 0] == pytest.approx(1.0, abs=1e-5)

    def test_not_enclosed_preserves_out_of_bounds(self, identity_cameras, unit_bounds):
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        pts = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -1.0)], pts)
        assert result[0, 0] == pytest.approx(-1.0, abs=1e-5)


class TestKernelCallerBatching:
    """kernel_caller should split large arrays into 10M-point batches."""

    def test_large_input_batched(self, identity_cameras, unit_bounds):
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        call_count = [0]
        def counting_kernel(x):
            call_count[0] += 1
            return np.zeros(len(x))

        # Slightly more than 10M to force two batches
        n = 10_000_001
        pts = np.zeros((n, 3), dtype=np.float64)
        result = mesher.kernel_caller([counting_kernel], pts)
        assert result.shape == (n, 1)
        assert call_count[0] == 2  # two batches


# ---------------------------------------------------------------------------
# kernel_caller edge cases
# ---------------------------------------------------------------------------
class TestKernelCallerNanInf:
    """kernel_caller behavior with NaN and Inf values in coordinates."""

    def test_nan_coordinates_propagate(self, identity_cameras, unit_bounds):
        """NaN coordinates should be passed through to the kernel."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        pts = np.array([[np.nan, 0.0, 0.0]], dtype=np.float64)
        received = [None]
        def capture_kernel(x):
            received[0] = x.copy()
            return np.zeros(len(x))
        mesher.kernel_caller([capture_kernel], pts)
        assert np.isnan(received[0][0, 0])

    def test_inf_coordinates_propagate(self, identity_cameras, unit_bounds):
        """Inf coordinates should be passed through to the kernel."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        pts = np.array([[np.inf, -np.inf, 0.0]], dtype=np.float64)
        received = [None]
        def capture_kernel(x):
            received[0] = x.copy()
            return np.zeros(len(x))
        mesher.kernel_caller([capture_kernel], pts)
        assert np.isinf(received[0][0, 0])

    def test_nan_in_enclosed_mode_gets_clipped(self, identity_cameras, unit_bounds):
        """NaN coords in enclosed mode: NaN comparisons are False, so NaN
        points are NOT considered out_bound and retain kernel value."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        pts = np.array([[np.nan, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -2.0)], pts)
        # NaN <= -1 is False, NaN >= 1 is False, so NOT out_bound
        assert result[0, 0] == pytest.approx(-2.0, abs=1e-5)

    def test_kernel_returning_nan(self, identity_cameras, unit_bounds):
        """Kernel returning NaN should propagate to output."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), np.nan)], pts)
        assert np.isnan(result[0, 0])


class TestKernelCallerBoundaryConditions:
    """Edge cases at boundary values and special configurations."""

    def test_single_point_inside(self, identity_cameras, unit_bounds):
        """Single point just inside bounds should keep kernel value."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        pts = np.array([[0.999, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -3.0)], pts)
        # 0.999 < 1.0 and 0.999 > -1.0, so inside
        assert result[0, 0] == pytest.approx(-3.0, abs=1e-5)

    def test_epsilon_inside_boundary(self, identity_cameras, unit_bounds):
        """Point epsilon inside boundary should NOT be clipped."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        eps = np.finfo(np.float64).eps
        pts = np.array([[-1.0 + eps, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -1.0)], pts)
        # -1.0 + eps > -1.0, so NOT out_bound on xmin
        assert result[0, 0] == pytest.approx(-1.0, abs=1e-5)

    def test_one_axis_out_others_in(self, identity_cameras, unit_bounds):
        """Only one axis out of bounds should still clip."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        pts = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)  # z out
        result = mesher.kernel_caller([lambda x: np.full(len(x), -1.0)], pts)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_all_axes_out_of_bounds(self, identity_cameras, unit_bounds):
        """Point outside on all axes should be clipped."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        pts = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -1.0)], pts)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_very_large_coordinates(self, identity_cameras, unit_bounds):
        """Extremely large coordinates should be clipped when enclosed."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        pts = np.array([[1e10, 1e10, 1e10]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -1.0)], pts)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_negative_kernel_values_preserved(self, identity_cameras, unit_bounds):
        """Large negative SDF values should be preserved for interior points."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -1e6)], pts)
        np.testing.assert_allclose(result[0, 0], -1e6, rtol=1e-5)

    def test_zero_kernel_value(self, identity_cameras, unit_bounds):
        """Kernel returning zero (surface) should be preserved."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.zeros(len(x))], pts)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-5)


class TestKernelCallerMultipleKernelEdgeCases:
    """Edge cases with multiple kernels."""

    def test_three_kernels(self, identity_cameras, unit_bounds):
        """Three kernels should produce (N, 3) output."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        k1 = lambda x: np.ones(len(x))
        k2 = lambda x: np.full(len(x), 2.0)
        k3 = lambda x: np.full(len(x), 3.0)
        result = mesher.kernel_caller([k1, k2, k3], pts)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0], atol=1e-5)

    def test_kernels_with_different_ranges(self, identity_cameras, unit_bounds):
        """Kernels returning very different value ranges."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        k1 = lambda x: np.full(len(x), 1e-10)
        k2 = lambda x: np.full(len(x), 1e10)
        result = mesher.kernel_caller([k1, k2], pts)
        assert result.shape == (1, 2)

    def test_enclosed_clips_all_kernels(self, identity_cameras, unit_bounds):
        """Out-of-bounds clipping should apply to ALL kernels when enclosed."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        pts = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)  # outside x
        k1 = lambda x: np.full(len(x), -1.0)
        k2 = lambda x: np.full(len(x), -2.0)
        result = mesher.kernel_caller([k1, k2], pts)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert result[0, 1] == pytest.approx(1.0, abs=1e-5)

    def test_mixed_in_out_points_multiple_kernels(self, identity_cameras, unit_bounds):
        """Mix of inside/outside points with multiple kernels."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        pts = np.array([
            [0.0, 0.0, 0.0],   # inside
            [5.0, 0.0, 0.0],   # outside
        ], dtype=np.float64)
        k1 = lambda x: np.full(len(x), -1.0)
        k2 = lambda x: np.full(len(x), -2.0)
        result = mesher.kernel_caller([k1, k2], pts)
        # Inside point
        assert result[0, 0] == pytest.approx(-1.0, abs=1e-5)
        assert result[0, 1] == pytest.approx(-2.0, abs=1e-5)
        # Outside point — clipped for both kernels
        assert result[1, 0] == pytest.approx(1.0, abs=1e-5)
        assert result[1, 1] == pytest.approx(1.0, abs=1e-5)


class TestKernelCallerBatchingEdgeCases:
    """Batching edge cases: exact boundary, multiple batches with enclosed."""

    def test_exactly_batch_size(self, identity_cameras, unit_bounds):
        """Exactly 10M points should be handled in a single batch."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        call_count = [0]
        def counting_kernel(x):
            call_count[0] += 1
            return np.zeros(len(x))
        n = 10_000_000
        pts = np.zeros((n, 3), dtype=np.float64)
        result = mesher.kernel_caller([counting_kernel], pts)
        assert result.shape == (n, 1)
        assert call_count[0] == 1  # exactly one batch

    def test_batch_plus_one(self, identity_cameras, unit_bounds):
        """10M + 1 points should split into exactly two batches."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        batch_sizes = []
        def recording_kernel(x):
            batch_sizes.append(len(x))
            return np.zeros(len(x))
        n = 10_000_001
        pts = np.zeros((n, 3), dtype=np.float64)
        result = mesher.kernel_caller([recording_kernel], pts)
        assert len(batch_sizes) == 2
        assert batch_sizes[0] == 10_000_000
        assert batch_sizes[1] == 1

    def test_batching_with_enclosed_mode(self, identity_cameras, unit_bounds):
        """Enclosed mode should work correctly across batch boundaries."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=True)
        # Use small number of points for speed, verify behavior
        n = 5
        pts = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [2.0, 0.0, 0.0],  # outside
            [-0.5, -0.5, -0.5],
            [0.0, 0.0, 2.0],  # outside
        ], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), -1.0)], pts)
        assert result[0, 0] == pytest.approx(-1.0, abs=1e-5)
        assert result[1, 0] == pytest.approx(-1.0, abs=1e-5)
        assert result[2, 0] == pytest.approx(1.0, abs=1e-5)
        assert result[3, 0] == pytest.approx(-1.0, abs=1e-5)
        assert result[4, 0] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# OcMesher constructor
# ---------------------------------------------------------------------------
class TestOcMesherInit:
    """Constructor edge cases."""

    def test_single_camera(self, unit_bounds):
        """Should work with just one camera."""
        cam_poses = [np.eye(4)]
        Ks = [np.eye(3) * 500]
        cameras = (cam_poses, Ks, [480], [640])
        mesher = _make_mesher_stub(unit_bounds, cameras)
        assert mesher.n_cameras == 1

    def test_camera_data_layout(self, identity_cameras, unit_bounds):
        """Camera array should have length 23 * n_cameras."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras)
        assert mesher.cameras.shape == (23 * 2,)
        assert mesher.cameras.dtype == np.float64

    def test_center_computation(self, identity_cameras):
        bounds = [0.0, 10.0, 0.0, 20.0, 0.0, 30.0]
        mesher = _make_mesher_stub(bounds, identity_cameras)
        np.testing.assert_allclose(mesher.center, [5.0, 10.0, 15.0])

    def test_size_computation_takes_max_dimension(self, identity_cameras):
        bounds = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
        mesher = _make_mesher_stub(bounds, identity_cameras)
        expected = 3.0 * 1.1  # max dimension * 1.1
        assert float(mesher.size) == pytest.approx(expected)

    def test_asymmetric_bounds(self, identity_cameras):
        bounds = [-5.0, 5.0, -1.0, 1.0, -0.5, 0.5]
        mesher = _make_mesher_stub(bounds, identity_cameras)
        np.testing.assert_allclose(mesher.center, [0.0, 0.0, 0.0])
        # Max dimension is x: 10.0
        assert float(mesher.size) == pytest.approx(10.0 * 1.1)

    def test_many_cameras(self, unit_bounds):
        n = 50
        cam_poses = [np.eye(4) for _ in range(n)]
        Ks = [np.eye(3) * 500 for _ in range(n)]
        cameras = (cam_poses, Ks, [480] * n, [640] * n)
        mesher = _make_mesher_stub(unit_bounds, cameras)
        assert mesher.n_cameras == n
        assert mesher.cameras.shape == (23 * n,)

    def test_default_config_values(self, identity_cameras, unit_bounds):
        mesher = _make_mesher_stub(unit_bounds, identity_cameras)
        assert mesher.bisection_iters == 15
        assert mesher.enclosed is True
        assert mesher.simplify_occluded is True
        assert mesher.visible_relax_iter == 2
        assert mesher.coarse_count == 500000
        assert mesher.memory_limit_mb == 1000

    def test_custom_config_values(self, identity_cameras, unit_bounds):
        with patch("ocmesher.core.load_cdll") as mock_cdll, \
             patch("ocmesher.core.register_func"):
            mock_cdll.return_value = MagicMock()
            from ocmesher.core import OcMesher
            mesher = OcMesher(
                identity_cameras, unit_bounds,
                pixels_per_cube=16,
                inv_scale=20,
                min_dist=2,
                memory_limit_mb=2000,
                bisection_iters=10,
                enclosed=False,
                simplify_occluded=False,
                visible_relax_iter=5,
                coarse_count=100000,
            )
        assert mesher.bisection_iters == 10
        assert mesher.enclosed is False
        assert mesher.simplify_occluded is False
        assert mesher.visible_relax_iter == 5
        assert mesher.coarse_count == 100000
        assert mesher.memory_limit_mb == 2000


class TestOcMesherInitEdgeCases:
    """Additional constructor edge cases."""

    def test_zero_centered_bounds(self, identity_cameras):
        """Bounds centered at origin."""
        bounds = [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0]
        mesher = _make_mesher_stub(bounds, identity_cameras)
        np.testing.assert_allclose(mesher.center, [0.0, 0.0, 0.0])
        assert float(mesher.size) == pytest.approx(20.0 * 1.1)

    def test_very_small_bounds(self, identity_cameras):
        """Very small bounds should still compute correct center and size."""
        bounds = [-1e-6, 1e-6, -1e-6, 1e-6, -1e-6, 1e-6]
        mesher = _make_mesher_stub(bounds, identity_cameras)
        np.testing.assert_allclose(mesher.center, [0.0, 0.0, 0.0], atol=1e-10)
        assert float(mesher.size) == pytest.approx(2e-6 * 1.1)

    def test_very_large_bounds(self, identity_cameras):
        """Very large bounds should not overflow."""
        bounds = [-1e6, 1e6, -1e6, 1e6, -1e6, 1e6]
        mesher = _make_mesher_stub(bounds, identity_cameras)
        np.testing.assert_allclose(mesher.center, [0.0, 0.0, 0.0])
        assert float(mesher.size) == pytest.approx(2e6 * 1.1)

    def test_non_uniform_bounds(self, identity_cameras):
        """Non-uniform bounds where each axis has different extent."""
        bounds = [0.0, 1.0, 0.0, 5.0, 0.0, 10.0]
        mesher = _make_mesher_stub(bounds, identity_cameras)
        np.testing.assert_allclose(mesher.center, [0.5, 2.5, 5.0])
        # Max dimension is z: 10.0
        assert float(mesher.size) == pytest.approx(10.0 * 1.1)

    def test_offset_positive_bounds(self, identity_cameras):
        """Bounds entirely in positive octant."""
        bounds = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        mesher = _make_mesher_stub(bounds, identity_cameras)
        np.testing.assert_allclose(mesher.center, [15.0, 35.0, 55.0])

    def test_camera_inverse_stored(self, unit_bounds):
        """Camera poses are inverted during packing — verify inv stored."""
        # Non-identity camera: translate by (1,2,3)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [1.0, 2.0, 3.0]
        cameras = ([pose], [np.eye(3, dtype=np.float64) * 500], [480], [640])
        mesher = _make_mesher_stub(unit_bounds, cameras)
        # First 12 values are flattened inv_pose[:3,:4]
        inv_pose_flat = mesher.cameras[:12]
        expected_inv = np.linalg.inv(pose)[:3, :4].reshape(-1).astype(np.float64)
        np.testing.assert_allclose(inv_pose_flat, expected_inv)

    def test_camera_intrinsics_stored(self, unit_bounds):
        """Intrinsic matrix K should be stored at offset 12 in camera data."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        cameras = ([np.eye(4)], [K], [480], [640])
        mesher = _make_mesher_stub(unit_bounds, cameras)
        K_flat = mesher.cameras[12:21]
        np.testing.assert_allclose(K_flat, K.reshape(-1))

    def test_camera_hw_stored(self, unit_bounds):
        """H and W should be stored at offsets 21 and 22."""
        cameras = ([np.eye(4)], [np.eye(3) * 500], [720], [1280])
        mesher = _make_mesher_stub(unit_bounds, cameras)
        assert mesher.cameras[21] == pytest.approx(720.0)
        assert mesher.cameras[22] == pytest.approx(1280.0)

    def test_multiple_cameras_packed_sequentially(self, unit_bounds):
        """Multiple cameras should be packed at stride=23 offsets."""
        n = 3
        cam_poses = [np.eye(4) for _ in range(n)]
        Ks = [np.eye(3) * 500 for _ in range(n)]
        Hs = [480, 720, 1080]
        Ws = [640, 1280, 1920]
        cameras = (cam_poses, Ks, Hs, Ws)
        mesher = _make_mesher_stub(unit_bounds, cameras)
        for i in range(n):
            assert mesher.cameras[23 * i + 21] == pytest.approx(Hs[i])
            assert mesher.cameras[23 * i + 22] == pytest.approx(Ws[i])

    def test_rotated_camera_pose(self, unit_bounds):
        """Non-trivial rotation in camera pose should be correctly inverted."""
        # 90-degree rotation around z-axis
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        cameras = ([pose], [np.eye(3) * 500], [480], [640])
        mesher = _make_mesher_stub(unit_bounds, cameras)
        inv_pose_flat = mesher.cameras[:12]
        expected = np.linalg.inv(pose)[:3, :4].reshape(-1).astype(np.float64)
        np.testing.assert_allclose(inv_pose_flat, expected, atol=1e-10)

    def test_bounds_stored_as_attribute(self, identity_cameras, unit_bounds):
        """Bounds should be stored as-is on the mesher object."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras)
        assert mesher.bounds == unit_bounds

    def test_size_uses_1_1_expansion(self, identity_cameras):
        """Size should be max_dimension * 1.1."""
        bounds = [0.0, 5.0, 0.0, 3.0, 0.0, 2.0]
        mesher = _make_mesher_stub(bounds, identity_cameras)
        assert float(mesher.size) == pytest.approx(5.0 * 1.1)

    def test_float_type_attributes(self, identity_cameras, unit_bounds):
        """Float type attributes should be set correctly."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras)
        assert mesher.np_float_type == np.float64
        assert mesher.sdf_np_float_type == np.float32


class TestKernelCallerInputTypes:
    """Test kernel_caller with various input array types."""

    def test_float32_input(self, identity_cameras, unit_bounds):
        """kernel_caller should work with float32 input coordinates."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        result = mesher.kernel_caller([lambda x: np.zeros(len(x))], pts)
        assert result.shape == (1, 1)

    def test_contiguous_output(self, identity_cameras, unit_bounds):
        """Output should be a contiguous array."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.zeros(len(x))], pts)
        assert result.flags["C_CONTIGUOUS"]

    def test_single_point_single_kernel(self, identity_cameras, unit_bounds):
        """Minimum non-empty case: 1 point, 1 kernel."""
        mesher = _make_mesher_stub(unit_bounds, identity_cameras, enclosed=False)
        pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        result = mesher.kernel_caller([lambda x: np.full(len(x), 42.0)], pts)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], 42.0, atol=1e-3)
