"""Tests for ``ocmesher.core``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocmesher.core import (
    CAMERA_DATA_STRIDE,
    OcMesher,
    _validate_bounds,
    _validate_cameras,
)

# ---------------------------------------------------------------------------
# _validate_cameras
# ---------------------------------------------------------------------------


class TestValidateCameras:
    def test_valid_single_camera(self, sample_cameras):
        poses, _ks, _hs, _ws = _validate_cameras(sample_cameras)
        assert len(poses) == 1

    def test_valid_multiple_cameras(self, sample_camera_pose, sample_intrinsics):
        cameras = (
            [sample_camera_pose, sample_camera_pose],
            [sample_intrinsics, sample_intrinsics],
            [720, 720],
            [1280, 1280],
        )
        poses, _ks, _hs, _ws = _validate_cameras(cameras)
        assert len(poses) == 2

    def test_normalises_lists_to_arrays(self, sample_cameras_as_lists):
        poses, ks, _hs, _ws = _validate_cameras(sample_cameras_as_lists)
        assert isinstance(poses[0], np.ndarray)
        assert isinstance(ks[0], np.ndarray)
        assert poses[0].dtype == np.float64
        assert ks[0].dtype == np.float64

    def test_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="must be a tuple/list"):
            _validate_cameras("not a tuple")

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="must be a tuple/list"):
            _validate_cameras(([], []))

    def test_rejects_mismatched_lengths(self, sample_camera_pose, sample_intrinsics):
        cameras = ([sample_camera_pose], [sample_intrinsics, sample_intrinsics], [720], [1280])
        with pytest.raises(ValueError, match="same length"):
            _validate_cameras(cameras)

    def test_rejects_empty_cameras(self):
        with pytest.raises(ValueError, match="At least one camera"):
            _validate_cameras(([], [], [], []))

    def test_rejects_bad_pose_shape(self, sample_intrinsics):
        bad_pose = np.eye(3)
        cameras = ([bad_pose], [sample_intrinsics], [720], [1280])
        with pytest.raises(ValueError, match="4x4 matrix"):
            _validate_cameras(cameras)

    def test_rejects_bad_intrinsics_shape(self, sample_camera_pose):
        bad_k = np.eye(4)
        cameras = ([sample_camera_pose], [bad_k], [720], [1280])
        with pytest.raises(ValueError, match="3x3 matrix"):
            _validate_cameras(cameras)


# ---------------------------------------------------------------------------
# _validate_bounds
# ---------------------------------------------------------------------------


class TestValidateBounds:
    def test_valid_bounds(self, sample_bounds):
        result = _validate_bounds(sample_bounds)
        assert result.shape == (6,)
        assert result.dtype == np.float64

    def test_converts_list_to_array(self):
        result = _validate_bounds([-1, 1, -2, 2, -3, 3])
        assert isinstance(result, np.ndarray)
        assert result.shape == (6,)

    def test_converts_tuple_to_array(self):
        result = _validate_bounds((-10, 10, -10, 10, -2, 2))
        assert isinstance(result, np.ndarray)

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="6 elements"):
            _validate_bounds([0, 1, 2])

    def test_rejects_min_geq_max_x(self):
        with pytest.raises(ValueError, match=r"x_min.*less than.*x_max"):
            _validate_bounds([5, -5, -1, 1, -1, 1])

    def test_rejects_min_geq_max_y(self):
        with pytest.raises(ValueError, match=r"y_min.*less than.*y_max"):
            _validate_bounds([-1, 1, 5, -5, -1, 1])

    def test_rejects_min_geq_max_z(self):
        with pytest.raises(ValueError, match=r"z_min.*less than.*z_max"):
            _validate_bounds([-1, 1, -1, 1, 5, -5])

    def test_rejects_equal_min_max(self):
        with pytest.raises(ValueError, match="less than"):
            _validate_bounds([0, 0, -1, 1, -1, 1])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestCameraDataStride:
    def test_stride_value(self):
        # 12 (3x4 inv_pose) + 9 (3x3 K) + 2 (H, W) = 23
        assert CAMERA_DATA_STRIDE == 23

    def test_camera_data_packing(self, sample_camera_pose, sample_intrinsics):
        pose = sample_camera_pose
        k = sample_intrinsics
        h, w = 720, 1280

        inv_pose = np.linalg.inv(pose)[:3, :4].reshape(-1)
        packed = np.concatenate([inv_pose, k.reshape(-1), [h], [w]])
        assert packed.shape == (CAMERA_DATA_STRIDE,)


# ---------------------------------------------------------------------------
# kernel_caller
# ---------------------------------------------------------------------------


class TestKernelCaller:
    """Tests for OcMesher.kernel_caller (with mocked DLL)."""

    @staticmethod
    def _make_mesher_stub(bounds, *, enclosed=True):
        """Create a lightweight stand-in that mimics OcMesher for kernel_caller."""
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = enclosed
        obj.sdf_np_float_type = np.float32
        return obj

    def test_empty_points_returns_empty(self, sample_bounds, sphere_kernel):
        mesher = self._make_mesher_stub(sample_bounds)
        result = mesher.kernel_caller([sphere_kernel], np.zeros((0, 3)))
        assert result.shape == (0, 1)

    def test_sphere_kernel_values(self, sample_bounds, sphere_kernel):
        mesher = self._make_mesher_stub(sample_bounds)
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result[:, 0], [-1.0, 0.0, 1.0], atol=1e-6)

    def test_enclosed_clamps_out_of_bounds(self, sphere_kernel):
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=True)
        points = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        assert result[1, 0] == 1.0

    def test_not_enclosed_does_not_clamp(self, sphere_kernel):
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=False)
        points = np.array([[2, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        np.testing.assert_allclose(result[:, 0], [1.0], atol=1e-6)

    def test_multiple_kernels(self, sample_bounds, sphere_kernel, plane_kernel):
        mesher = self._make_mesher_stub(sample_bounds)
        points = np.array([[0, 0, 0.5]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel, plane_kernel], points)
        assert result.shape == (1, 2)

    def test_rejects_non_callable_kernel(self, sample_bounds):
        mesher = self._make_mesher_stub(sample_bounds)
        points = np.array([[0, 0, 0]], dtype=np.float64)
        with pytest.raises(TypeError, match="callable"):
            mesher.kernel_caller(["not a function"], points)

    def test_large_batch(self, sample_bounds, sphere_kernel):
        mesher = self._make_mesher_stub(sample_bounds)
        points = np.zeros((100, 3), dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        assert result.shape == (100, 1)


# ---------------------------------------------------------------------------
# OcMesher.__call__ validation
# ---------------------------------------------------------------------------


class TestOcMesherCallValidation:
    def test_rejects_empty_kernels(self):
        obj = object.__new__(OcMesher)
        with pytest.raises(ValueError, match="non-empty"):
            obj([])

    def test_rejects_non_list_kernels(self):
        obj = object.__new__(OcMesher)
        with pytest.raises(ValueError, match="non-empty"):
            obj("not a list")


# ---------------------------------------------------------------------------
# OcMesher.__init__ (with mocked DLL)
# ---------------------------------------------------------------------------


class TestOcMesherInit:
    @patch("ocmesher.core.load_cdll")
    def test_init_validates_cameras(self, mock_load, sample_bounds):
        mock_load.return_value = MagicMock()
        with pytest.raises(ValueError, match="must be a tuple/list"):
            OcMesher("bad cameras", sample_bounds)

    @patch("ocmesher.core.load_cdll")
    def test_init_validates_bounds(self, mock_load, sample_cameras):
        mock_load.return_value = MagicMock()
        with pytest.raises(ValueError, match="6 elements"):
            OcMesher(sample_cameras, [0, 1])

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_init_success_with_mock_dll(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        assert mesher.n_cameras == 1
        assert mesher.bisection_iters == 15
        assert mesher.enclosed is True
        np.testing.assert_allclose(mesher.center, [0, 0, 0], atol=1e-10)

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_init_camera_packing_shape(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        assert mesher.cameras.shape == (CAMERA_DATA_STRIDE,)

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_init_custom_params(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        mock_load.return_value = MagicMock()
        mesher = OcMesher(
            sample_cameras,
            sample_bounds,
            pixels_per_cube=16,
            inv_scale=20,
            min_dist=0.5,
            memory_limit_mb=2000,
            bisection_iters=10,
            enclosed=False,
            simplify_occluded=False,
            visible_relax_iter=3,
            coarse_count=100000,
        )
        assert mesher.bisection_iters == 10
        assert mesher.enclosed is False
        assert mesher.simplify_occluded is False
        assert mesher.visible_relax_iter == 3
        assert mesher.coarse_count == 100000
        assert mesher.memory_limit_mb == 2000

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_init_accepts_list_cameras(self, _mock_register, mock_load, sample_cameras_as_lists, sample_bounds):
        """Cameras given as plain Python lists should be accepted."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras_as_lists, sample_bounds)
        assert mesher.n_cameras == 1
