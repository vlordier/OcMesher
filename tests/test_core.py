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
