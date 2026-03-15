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
    _validate_kernels,
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

    def test_rejects_nan_values(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_bounds([np.nan, 1.0, -1.0, 1.0, -1.0, 1.0])

    def test_rejects_inf_values(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_bounds([-np.inf, np.inf, -1.0, 1.0, -1.0, 1.0])


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
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
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
        # [1.5, 0, 0] is outside bounds (x > 1.0); sphere SDF = 0.5, so clamping must force it to 1.0.
        points = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        assert result[1, 0] == pytest.approx(1.0)

    def test_not_enclosed_does_not_clamp(self, sphere_kernel):
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=False)
        # With enclosed=False, [1.5, 0, 0] is outside bounds but the natural SDF (0.5) must be returned.
        points = np.array([[1.5, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        np.testing.assert_allclose(result[:, 0], [0.5], atol=1e-6)

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

    def test_rejects_kernel_wrong_output_shape(self, sample_bounds):
        mesher = self._make_mesher_stub(sample_bounds)
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)

        def bad_kernel(xyz):
            return np.zeros((len(xyz), 2))  # wrong: 2D instead of 1D

        with pytest.raises(ValueError, match="returned shape"):
            mesher.kernel_caller([bad_kernel], points)

    def test_large_batch(self, sample_bounds, sphere_kernel):
        mesher = self._make_mesher_stub(sample_bounds)
        points = np.zeros((100, 3), dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        assert result.shape == (100, 1)

    def test_output_dtype_is_float32(self, sample_bounds, sphere_kernel):
        mesher = self._make_mesher_stub(sample_bounds)
        points = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        assert result.dtype == np.float32

    def test_enclosed_clamps_point_on_boundary(self, sphere_kernel):
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=True)
        # Point exactly on the boundary (x == x_max = 1.0) satisfies x >= x_max,
        # so it must be clamped.  Sphere SDF = 0.0, but clamped result must be 1.0.
        points = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        assert result[0, 0] == pytest.approx(1.0)


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

    def test_rejects_non_callable_kernel_in_call(self, sphere_kernel):
        obj = object.__new__(OcMesher)
        with pytest.raises(TypeError, match="callable"):
            obj([sphere_kernel, "not_a_function"])


# ---------------------------------------------------------------------------
# _validate_kernels
# ---------------------------------------------------------------------------


class TestValidateKernels:
    def test_accepts_single_callable(self, sphere_kernel):
        _validate_kernels([sphere_kernel])  # no exception

    def test_accepts_multiple_callables(self, sphere_kernel, plane_kernel):
        _validate_kernels([sphere_kernel, plane_kernel])  # no exception

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_kernels([])

    def test_rejects_non_sequence(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_kernels(42)

    def test_rejects_non_callable_element(self, sphere_kernel):
        with pytest.raises(TypeError, match=r"kernels\[1\] must be callable"):
            _validate_kernels([sphere_kernel, "bad"])

    def test_error_includes_index(self):
        with pytest.raises(TypeError, match=r"kernels\[0\]"):
            _validate_kernels(["not_callable"])


# ---------------------------------------------------------------------------
# OcMesher.__init__ (with mocked DLL)
# ---------------------------------------------------------------------------


class TestOcMesherInit:
    @patch("ocmesher.dll.load_cdll")
    def test_init_validates_cameras(self, mock_load, sample_bounds):
        mock_load.return_value = MagicMock()
        with pytest.raises(ValueError, match="must be a tuple/list"):
            OcMesher("bad cameras", sample_bounds)

    @patch("ocmesher.dll.load_cdll")
    def test_init_validates_bounds(self, mock_load, sample_cameras):
        mock_load.return_value = MagicMock()
        with pytest.raises(ValueError, match="6 elements"):
            OcMesher(sample_cameras, [0, 1])

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_init_success_with_mock_dll(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        assert mesher.n_cameras == 1
        assert mesher.bisection_iters == 15
        assert mesher.enclosed is True
        np.testing.assert_allclose(mesher.center, [0, 0, 0], atol=1e-10)

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_init_camera_packing_shape(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        assert mesher.cameras.shape == (CAMERA_DATA_STRIDE,)

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
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

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_init_accepts_list_cameras(self, _mock_register, mock_load, sample_cameras_as_lists, sample_bounds):
        """Cameras given as plain Python lists should be accepted."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras_as_lists, sample_bounds)
        assert mesher.n_cameras == 1

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_init_camera_packing_values(
        self, _mock_register, mock_load, sample_cameras, sample_bounds, sample_camera_pose, sample_intrinsics
    ):
        """Camera array must contain correct inv_pose, K, H, W values."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        expected_inv_pose = np.linalg.inv(sample_camera_pose)[:3, :4].reshape(-1)
        expected_k = sample_intrinsics.reshape(-1)
        cam = mesher.cameras
        np.testing.assert_allclose(cam[:12], expected_inv_pose, atol=1e-10)
        np.testing.assert_allclose(cam[12:21], expected_k, atol=1e-10)
        assert cam[21] == pytest.approx(720.0)  # H
        assert cam[22] == pytest.approx(1280.0)  # W

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_init_center_for_asymmetric_bounds(self, _mock_register, mock_load, sample_cameras):
        """Center must be the midpoint of each axis even for non-symmetric bounds."""
        mock_load.return_value = MagicMock()
        bounds = np.array([0.0, 4.0, -2.0, 6.0, 1.0, 3.0])
        mesher = OcMesher(sample_cameras, bounds)
        np.testing.assert_allclose(mesher.center, [2.0, 2.0, 2.0], atol=1e-10)

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_init_size_is_largest_dimension(self, _mock_register, mock_load, sample_cameras):
        """size must be 1.1x the largest axis range."""
        mock_load.return_value = MagicMock()
        # x-range = 10, y-range = 4, z-range = 2
        bounds = np.array([0.0, 10.0, -2.0, 2.0, -1.0, 1.0])
        mesher = OcMesher(sample_cameras, bounds)
        assert mesher.size == pytest.approx(11.0)

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_init_multi_camera_packing_shape(
        self, _mock_register, mock_load, sample_camera_pose, sample_intrinsics, sample_bounds
    ):
        """Camera array must be 2x CAMERA_DATA_STRIDE for two cameras."""
        mock_load.return_value = MagicMock()
        cameras = (
            [sample_camera_pose, sample_camera_pose],
            [sample_intrinsics, sample_intrinsics],
            [720, 480],
            [1280, 640],
        )
        mesher = OcMesher(cameras, sample_bounds)
        assert mesher.n_cameras == 2
        assert mesher.cameras.shape == (2 * CAMERA_DATA_STRIDE,)


# ---------------------------------------------------------------------------
# OcMesher.__repr__ and context manager
# ---------------------------------------------------------------------------


class TestOcMesherReprAndContextManager:
    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_repr_contains_key_fields(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        r = repr(mesher)
        assert "OcMesher(" in r
        assert "n_cameras=1" in r
        assert "bisection_iters=15" in r
        assert "enclosed=True" in r

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_context_manager_returns_self(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        mock_load.return_value = MagicMock()
        with OcMesher(sample_cameras, sample_bounds) as m:
            assert isinstance(m, OcMesher)

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_context_manager_does_not_suppress_exceptions(
        self, _mock_register, mock_load, sample_cameras, sample_bounds
    ):
        mock_load.return_value = MagicMock()
        with pytest.raises(RuntimeError), OcMesher(sample_cameras, sample_bounds):
            raise RuntimeError("boom")  # noqa: EM101


# ---------------------------------------------------------------------------
# Refactoring: np.empty for pre-allocated output buffers (commit 2)
# ---------------------------------------------------------------------------
class TestKernelCallerAllocation:
    """Verify that kernel_caller still produces correct results after the
    np.zeros -> np.empty change for write-before-read output buffers."""

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_kernel_caller_single_kernel(self, _mock_register, mock_load, sample_cameras, sample_bounds, sphere_kernel):
        """kernel_caller must return correct shape and dtype for one kernel."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result.shape == (2, 1)
        assert result.dtype == np.float32

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_kernel_caller_empty_input(self, _mock_register, mock_load, sample_cameras, sample_bounds, sphere_kernel):
        """kernel_caller with zero points must return empty float32 array."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        pts = np.zeros((0, 3), dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result.shape == (0, 1)
        assert result.dtype == np.float32

    @patch("ocmesher.dll.load_cdll")
    @patch("ocmesher.dll.register_func")
    def test_kernel_caller_large_batch_concatenates(
        self, _mock_register, mock_load, sample_cameras, sample_bounds, sphere_kernel, monkeypatch
    ):
        """kernel_caller must correctly concatenate batches > _SDF_BATCH_SIZE.

        Monkeypatches _SDF_BATCH_SIZE to a small value so the test exercises
        multi-batch logic without allocating millions of points.
        """
        monkeypatch.setattr("ocmesher.core._SDF_BATCH_SIZE", 8)
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        # 13 points forces two batches: [0:8] and [8:13]
        n = 13
        pts = np.random.default_rng(0).standard_normal((n, 3)).astype(np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result.shape == (n, 1)


# ---------------------------------------------------------------------------
# Vectorised bounds check (Refactor 1)
# ---------------------------------------------------------------------------


class TestVectorisedBoundsCheck:
    """Verify the vectorised out-of-bounds masking in kernel_caller."""

    @staticmethod
    def _make_mesher_stub(bounds, *, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = enclosed
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        return obj

    def test_precomputed_bounds_vectors_exist(self, sample_cameras, sample_bounds):
        """OcMesher.__init__ must pre-compute _bounds_min_np/_bounds_max_np."""
        with patch("ocmesher.dll.load_cdll") as mock_load:
            mock_load.return_value = MagicMock()
            mesher = OcMesher(sample_cameras, sample_bounds)
        np.testing.assert_array_equal(mesher._bounds_min_np, sample_bounds[0::2])
        np.testing.assert_array_equal(mesher._bounds_max_np, sample_bounds[1::2])

    def test_vectorised_clamp_matches_per_axis(self, sphere_kernel):
        bounds = np.array([-2.0, 2.0, -2.0, 2.0, -2.0, 2.0])
        mesher = self._make_mesher_stub(bounds, enclosed=True)
        # Mix of in-bounds and out-of-bounds points on all 3 axes.
        pts = np.array(
            [
                [0, 0, 0],        # inside
                [3, 0, 0],        # out: x > 2
                [0, -3, 0],       # out: y < -2
                [0, 0, 2.5],      # out: z > 2
                [-2.0, 0, 0],     # boundary: x == x_min
            ],
            dtype=np.float64,
        )
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result[0, 0] == pytest.approx(-1.0, abs=1e-6)  # inside → natural SDF
        assert result[1, 0] == pytest.approx(1.0)              # clamped
        assert result[2, 0] == pytest.approx(1.0)              # clamped
        assert result[3, 0] == pytest.approx(1.0)              # clamped
        assert result[4, 0] == pytest.approx(1.0)              # boundary → clamped

    def test_not_enclosed_ignores_bounds(self, sphere_kernel):
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=False)
        pts = np.array([[5, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        # Natural sphere SDF at (5,0,0) is 4.0, not clamped to 1.
        np.testing.assert_allclose(result[:, 0], [4.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Fused SDF evaluation in bisection loops (Refactor 2)
# ---------------------------------------------------------------------------


class TestFusedBisectionSDF:
    """Verify that fused SDF calls produce identical results to the unfused path."""

    @staticmethod
    def _make_mesher_stub(bounds, *, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = enclosed
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        return obj

    def test_fused_eval_matches_separate(self, sample_bounds, sphere_kernel):
        """kernel_caller on concatenated array must equal separate calls."""
        mesher = self._make_mesher_stub(sample_bounds)
        rng = np.random.default_rng(42)
        edge_pts = rng.standard_normal((20, 3))
        face_pts = rng.standard_normal((30, 3))

        # Separate calls (old path)
        e_sdf = mesher.kernel_caller([sphere_kernel], edge_pts)
        f_sdf = mesher.kernel_caller([sphere_kernel], face_pts)

        # Fused call (new path)
        combined = np.concatenate([edge_pts, face_pts])
        combined_sdf = mesher.kernel_caller([sphere_kernel], combined)
        e_sdf_fused = combined_sdf[:20]
        f_sdf_fused = combined_sdf[20:]

        np.testing.assert_array_equal(e_sdf, e_sdf_fused)
        np.testing.assert_array_equal(f_sdf, f_sdf_fused)

    def test_fused_eval_with_enclosed_clamp(self, sphere_kernel):
        """Fused eval must correctly apply enclosed clamping to all sub-arrays."""
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=True)
        # One in-bounds, one out-of-bounds per array.
        a = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float64)
        b = np.array([[0.5, 0, 0], [-5, 0, 0]], dtype=np.float64)

        combined = np.concatenate([a, b])
        sdf = mesher.kernel_caller([sphere_kernel], combined)
        # a[0] inside: natural, a[1] outside: 1.0
        assert sdf[0, 0] == pytest.approx(-1.0, abs=1e-6)
        assert sdf[1, 0] == pytest.approx(1.0)
        # b[0] inside: natural, b[1] outside: 1.0
        np.testing.assert_allclose(sdf[2, 0], np.linalg.norm([0.5, 0, 0]) - 1, atol=1e-6)
        assert sdf[3, 0] == pytest.approx(1.0)
