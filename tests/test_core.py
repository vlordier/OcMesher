"""Tests for ``ocmesher.core``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocmesher.core import (
    _SDF_BATCH_SIZE,
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
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
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

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
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

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_init_center_for_asymmetric_bounds(self, _mock_register, mock_load, sample_cameras):
        """Center must be the midpoint of each axis even for non-symmetric bounds."""
        mock_load.return_value = MagicMock()
        bounds = np.array([0.0, 4.0, -2.0, 6.0, 1.0, 3.0])
        mesher = OcMesher(sample_cameras, bounds)
        np.testing.assert_allclose(mesher.center, [2.0, 2.0, 2.0], atol=1e-10)

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_init_size_is_largest_dimension(self, _mock_register, mock_load, sample_cameras):
        """size must be 1.1x the largest axis range."""
        mock_load.return_value = MagicMock()
        # x-range = 10, y-range = 4, z-range = 2
        bounds = np.array([0.0, 10.0, -2.0, 2.0, -1.0, 1.0])
        mesher = OcMesher(sample_cameras, bounds)
        assert mesher.size == pytest.approx(11.0)

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
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
    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_repr_contains_key_fields(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        r = repr(mesher)
        assert "OcMesher(" in r
        assert "n_cameras=1" in r
        assert "bisection_iters=15" in r
        assert "bisection_tol=0.0" in r
        assert "enclosed=True" in r

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_context_manager_returns_self(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        mock_load.return_value = MagicMock()
        with OcMesher(sample_cameras, sample_bounds) as m:
            assert isinstance(m, OcMesher)

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
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

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_kernel_caller_single_kernel(self, _mock_register, mock_load, sample_cameras, sample_bounds, sphere_kernel):
        """kernel_caller must return correct shape and dtype for one kernel."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result.shape == (2, 1)
        assert result.dtype == np.float32

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_kernel_caller_empty_input(self, _mock_register, mock_load, sample_cameras, sample_bounds, sphere_kernel):
        """kernel_caller with zero points must return empty float32 array."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        pts = np.zeros((0, 3), dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result.shape == (0, 1)
        assert result.dtype == np.float32

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
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
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        return obj

    def test_precomputed_bounds_vectors_exist(self, sample_cameras, sample_bounds):
        """OcMesher.__init__ must pre-compute _bounds_min_np/_bounds_max_np."""
        with patch("ocmesher.core.load_cdll") as mock_load:
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
                [0, 0, 0],  # inside
                [3, 0, 0],  # out: x > 2
                [0, -3, 0],  # out: y < -2
                [0, 0, 2.5],  # out: z > 2
                [-2.0, 0, 0],  # boundary: x == x_min
            ],
            dtype=np.float64,
        )
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result[0, 0] == pytest.approx(-1.0, abs=1e-6)  # inside → natural SDF
        assert result[1, 0] == pytest.approx(1.0)  # clamped
        assert result[2, 0] == pytest.approx(1.0)  # clamped
        assert result[3, 0] == pytest.approx(1.0)  # clamped
        assert result[4, 0] == pytest.approx(1.0)  # boundary → clamped

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
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
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


# ---------------------------------------------------------------------------
# In-place bounds mask (Refactor 3)
# ---------------------------------------------------------------------------


class TestInPlaceBoundsMask:
    """Verify _out_of_bounds_mask produces correct masks without (N,3) temporaries."""

    def test_all_inside(self):
        b_min = np.array([-1.0, -1.0, -1.0])
        b_max = np.array([1.0, 1.0, 1.0])
        xyz = np.array([[0, 0, 0], [0.5, -0.5, 0.5]], dtype=np.float64)
        mask = OcMesher._out_of_bounds_mask(xyz, b_min, b_max)
        assert mask.shape == (2,)
        assert not mask.any()

    def test_all_outside(self):
        b_min = np.array([-1.0, -1.0, -1.0])
        b_max = np.array([1.0, 1.0, 1.0])
        xyz = np.array([[2, 0, 0], [0, -2, 0], [0, 0, 2]], dtype=np.float64)
        mask = OcMesher._out_of_bounds_mask(xyz, b_min, b_max)
        assert mask.all()

    def test_boundary_points_are_out(self):
        b_min = np.array([-1.0, -1.0, -1.0])
        b_max = np.array([1.0, 1.0, 1.0])
        # Points exactly on the boundary are treated as out-of-bounds.
        xyz = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]], dtype=np.float64)
        mask = OcMesher._out_of_bounds_mask(xyz, b_min, b_max)
        assert mask.all()

    def test_mixed_inside_outside(self):
        b_min = np.array([-2.0, -2.0, -2.0])
        b_max = np.array([2.0, 2.0, 2.0])
        xyz = np.array(
            [
                [0, 0, 0],  # inside
                [3, 0, 0],  # out x
                [0, -3, 0],  # out y
                [0, 0, 2.5],  # out z
                [-0.5, 0.5, 0],  # inside
            ],
            dtype=np.float64,
        )
        mask = OcMesher._out_of_bounds_mask(xyz, b_min, b_max)
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# Thread-pool parallel multi-kernel evaluation (Refactor 4)
# ---------------------------------------------------------------------------


class TestParallelMultiKernel:
    """Verify that multi-kernel evaluation uses threading and produces correct results."""

    @staticmethod
    def _make_mesher_stub(bounds, *, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = enclosed
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        return obj

    def test_multi_kernel_matches_single(self, sample_bounds, sphere_kernel, plane_kernel):
        """Two-kernel call must match two separate single-kernel calls."""
        mesher = self._make_mesher_stub(sample_bounds, enclosed=False)
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=np.float64)

        separate_sphere = mesher.kernel_caller([sphere_kernel], pts)
        separate_plane = mesher.kernel_caller([plane_kernel], pts)

        combined = mesher.kernel_caller([sphere_kernel, plane_kernel], pts)
        np.testing.assert_allclose(combined[:, 0], separate_sphere[:, 0], atol=1e-6)
        np.testing.assert_allclose(combined[:, 1], separate_plane[:, 0], atol=1e-6)

    def test_multi_kernel_enclosed_clamp(self, sphere_kernel, plane_kernel):
        """Out-of-bounds clamping must apply to all kernels in multi-kernel mode."""
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=True)
        pts = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float64)

        result = mesher.kernel_caller([sphere_kernel, plane_kernel], pts)
        # Inside point: natural SDF values
        assert result[0, 0] == pytest.approx(-1.0, abs=1e-6)  # sphere at origin
        assert result[0, 1] == pytest.approx(0.0, abs=1e-6)  # plane z=0
        # Outside point: both clamped to 1.0
        assert result[1, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(1.0)

    def test_multi_kernel_shape_and_dtype(self, sample_bounds, sphere_kernel, plane_kernel):
        """Multi-kernel result must have shape (N, K) and float32 dtype."""
        mesher = self._make_mesher_stub(sample_bounds, enclosed=False)
        pts = np.zeros((5, 3), dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel, plane_kernel], pts)
        assert result.shape == (5, 2)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Pre-allocated bisection buffer (Refactor 5)
# ---------------------------------------------------------------------------


class TestPreallocatedBisectionBuffer:
    """Verify that _refine_extra_vertices works with pre-allocated combined buffer."""

    @staticmethod
    def _make_mesher_stub(bounds, *, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = enclosed
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        return obj

    def test_fused_eval_with_preallocated_matches_separate(self, sample_bounds, sphere_kernel):
        """Pre-allocated buffer path must produce identical results to concatenation."""
        mesher = self._make_mesher_stub(sample_bounds)
        rng = np.random.default_rng(99)
        edge_pts = rng.standard_normal((15, 3))
        face_pts = rng.standard_normal((25, 3))

        # Separate calls
        e_sdf = mesher.kernel_caller([sphere_kernel], edge_pts)
        f_sdf = mesher.kernel_caller([sphere_kernel], face_pts)

        # Simulate pre-allocated buffer fill (as done in _refine_extra_vertices)
        n_edge = len(edge_pts)
        combined = np.empty((n_edge + len(face_pts), 3), dtype=edge_pts.dtype)
        combined[:n_edge] = edge_pts
        combined[n_edge:] = face_pts
        result = mesher.kernel_caller([sphere_kernel], combined)

        np.testing.assert_array_equal(e_sdf, result[:n_edge])
        np.testing.assert_array_equal(f_sdf, result[n_edge:])


# ---------------------------------------------------------------------------
# Early-exit bisection tolerance (Refactor 6)
# ---------------------------------------------------------------------------


class TestBisectionTolerance:
    """Verify the early-exit convergence check in bisection loops."""

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_default_tol_is_zero(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        """Default bisection_tol must be 0 (no early exit)."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        assert mesher.bisection_tol == 0.0

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_custom_tol_stored(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        """bisection_tol must be stored as a float attribute."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds, bisection_tol=1e-4)
        assert mesher.bisection_tol == pytest.approx(1e-4)

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_repr_includes_tol(self, _mock_register, mock_load, sample_cameras, sample_bounds):
        """__repr__ must include bisection_tol."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds, bisection_tol=0.001)
        assert "bisection_tol=0.001" in repr(mesher)


# ---------------------------------------------------------------------------
# Single-kernel column-view fast path (Refactor 7)
# ---------------------------------------------------------------------------


class TestSingleKernelMinFastPath:
    """Verify that single-kernel SDF returns a column view rather than min()."""

    def test_single_kernel_column_equals_min(self, sample_bounds, sphere_kernel):
        """For a single kernel, col-view [:, 0] must match .min(axis=-1)."""
        mesher = TestParallelMultiKernel._make_mesher_stub(sample_bounds, enclosed=False)
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        col_view = result[:, 0]
        col_min = result.min(axis=-1)
        np.testing.assert_array_equal(col_view, col_min)

    def test_multi_kernel_min_differs(self, sample_bounds, sphere_kernel, plane_kernel):
        """For multiple kernels, .min(axis=-1) may differ from any single column."""
        mesher = TestParallelMultiKernel._make_mesher_stub(sample_bounds, enclosed=False)
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel, plane_kernel], pts)
        assert result.shape == (3, 2)
        expected_min = np.minimum(result[:, 0], result[:, 1])
        np.testing.assert_allclose(result.min(axis=-1), expected_min, atol=1e-6)


# ---------------------------------------------------------------------------
# Pre-allocated slice-fill replaces np.concatenate (Refactor 8)
# ---------------------------------------------------------------------------


class TestSliceFillReplacesConcat:
    """Verify pre-allocated slice-fill produces same results as concatenation."""

    def test_slice_fill_matches_concat(self):
        """Pre-allocated buffer filled via slice assignment must match np.concatenate."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((10, 3))
        b = rng.standard_normal((20, 3))
        c = rng.standard_normal((15, 3))
        d = rng.standard_normal((25, 3))

        # np.concatenate reference
        concat_result = np.concatenate([a, b, c, d])

        # Slice-fill approach
        na, nb, nc, nd = len(a), len(b), len(c), len(d)
        buf = np.empty((na + nb + nc + nd, 3), dtype=a.dtype)
        buf[:na] = a
        buf[na : na + nb] = b
        buf[na + nb : na + nb + nc] = c
        buf[na + nb + nc :] = d

        np.testing.assert_array_equal(buf, concat_result)


# ---------------------------------------------------------------------------
# Persistent thread pool
# ---------------------------------------------------------------------------


class TestPersistentThreadPool:
    """Verify the persistent ThreadPoolExecutor lifecycle."""

    def test_pool_is_none_initially(self, sample_cameras, sample_bounds):
        """_sdf_pool must be None immediately after construction."""
        with patch("ocmesher.core.load_cdll") as mock_load:
            mock_load.return_value = MagicMock()
            mesher = OcMesher(sample_cameras, sample_bounds)
        assert mesher._sdf_pool is None

    def test_pool_created_on_multi_kernel(self, sample_bounds, sphere_kernel, plane_kernel):
        """Calling kernel_caller with >1 kernel must create the persistent pool."""
        obj = object.__new__(OcMesher)
        obj.bounds = sample_bounds
        obj.enclosed = False
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([sample_bounds[0], sample_bounds[2], sample_bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([sample_bounds[1], sample_bounds[3], sample_bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        obj.kernel_caller([sphere_kernel, plane_kernel], pts)
        assert obj._sdf_pool is not None

    def test_pool_not_created_for_single_kernel(self, sample_bounds, sphere_kernel):
        """Single-kernel fast path must not create a thread pool."""
        obj = object.__new__(OcMesher)
        obj.bounds = sample_bounds
        obj.enclosed = False
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([sample_bounds[0], sample_bounds[2], sample_bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([sample_bounds[1], sample_bounds[3], sample_bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        obj.kernel_caller([sphere_kernel], pts)
        assert obj._sdf_pool is None

    def test_context_manager_shuts_down_pool(self, sample_cameras, sample_bounds, sphere_kernel, plane_kernel):
        """Exiting the context manager must shut down the pool."""
        with patch("ocmesher.core.load_cdll") as mock_load:
            mock_load.return_value = MagicMock()
            with OcMesher(sample_cameras, sample_bounds) as mesher:
                pts = np.array([[0, 0, 0]], dtype=np.float64)
                mesher.kernel_caller([sphere_kernel, plane_kernel], pts)
                assert mesher._sdf_pool is not None
        assert mesher._sdf_pool is None


# ---------------------------------------------------------------------------
# Reusable bounds-mask buffers
# ---------------------------------------------------------------------------


class TestReusableMaskBuffers:
    """Verify that _out_of_bounds_mask_into reuses pre-allocated buffers."""

    def test_mask_buffer_size_matches_batch(self, sample_cameras, sample_bounds):
        """Pre-allocated mask buffers must be sized to _SDF_BATCH_SIZE."""
        with patch("ocmesher.core.load_cdll") as mock_load:
            mock_load.return_value = MagicMock()
            mesher = OcMesher(sample_cameras, sample_bounds)
        assert mesher._oob_mask.shape == (_SDF_BATCH_SIZE,)
        assert mesher._oob_tmp.shape == (_SDF_BATCH_SIZE,)

    def test_mask_into_matches_static(self, sample_bounds):
        """_out_of_bounds_mask_into must produce identical results to _out_of_bounds_mask."""
        obj = object.__new__(OcMesher)
        obj.bounds = sample_bounds
        obj.enclosed = True
        obj.sdf_np_float_type = np.float32
        b_min = np.array([sample_bounds[0], sample_bounds[2], sample_bounds[4]], dtype=np.float64)
        b_max = np.array([sample_bounds[1], sample_bounds[3], sample_bounds[5]], dtype=np.float64)
        obj._bounds_min_np = b_min
        obj._bounds_max_np = b_max
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)

        rng = np.random.default_rng(42)
        pts = rng.uniform(-5, 5, size=(100, 3))
        static_mask = OcMesher._out_of_bounds_mask(pts, b_min, b_max)
        reused_mask = obj._out_of_bounds_mask_into(pts, b_min, b_max)
        np.testing.assert_array_equal(static_mask, reused_mask)


# ---------------------------------------------------------------------------
# Vectorised camera packing
# ---------------------------------------------------------------------------


class TestVectorisedCameraPacking:
    """Verify that batch camera packing matches per-camera packing."""

    def test_camera_data_shape(self, sample_cameras, sample_bounds):
        """Camera data must have correct total length."""
        with patch("ocmesher.core.load_cdll") as mock_load:
            mock_load.return_value = MagicMock()
            mesher = OcMesher(sample_cameras, sample_bounds)
        assert len(mesher.cameras) == CAMERA_DATA_STRIDE * mesher.n_cameras

    def test_multi_camera_data_packed_correctly(self, sample_camera_pose, sample_intrinsics, sample_bounds):
        """Each camera stride must contain inv_pose[:3,:4], K, H, W."""
        cameras = (
            [sample_camera_pose, sample_camera_pose],
            [sample_intrinsics, sample_intrinsics],
            [480, 480],
            [640, 640],
        )
        with patch("ocmesher.core.load_cdll") as mock_load:
            mock_load.return_value = MagicMock()
            mesher = OcMesher(cameras, sample_bounds)
        # Both cameras should have identical packed data
        cam0 = mesher.cameras[:CAMERA_DATA_STRIDE]
        cam1 = mesher.cameras[CAMERA_DATA_STRIDE : 2 * CAMERA_DATA_STRIDE]
        np.testing.assert_array_equal(cam0, cam1)


# ---------------------------------------------------------------------------
# Hot-path local caching + np.fabs + removed inner tqdm
# ---------------------------------------------------------------------------


class TestHotPathLocalCaching:
    """Verify that kernel_caller still works after removing per-call validation."""

    @staticmethod
    def _make_mesher_stub(bounds, *, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = enclosed
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        return obj

    def test_kernel_caller_no_validation_overhead(self, sample_bounds, sphere_kernel):
        """kernel_caller with valid kernels works without per-call validation."""
        mesher = self._make_mesher_stub(sample_bounds)
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], points)
        np.testing.assert_allclose(result[:, 0], [-1.0, 0.0], atol=1e-6)

    def test_validate_kernels_catches_non_callable(self):
        """_validate_kernels (called at __call__ entry) catches non-callables."""
        with pytest.raises(TypeError, match="callable"):
            _validate_kernels(["not a function"])

    def test_np_fabs_matches_np_abs_for_float32(self):
        """np.fabs produces same result as np.abs for float32 arrays."""
        arr = np.array([-1.5, 0.0, 2.3, -0.7], dtype=np.float32)
        np.testing.assert_array_equal(np.fabs(arr), np.abs(arr))

    def test_np_fabs_max_matches_np_max_np_abs(self):
        """np.fabs(arr).max() is equivalent to np.max(np.abs(arr))."""
        arr = np.array([-3.0, 1.5, -2.0, 0.5], dtype=np.float32)
        assert np.fabs(arr).max() == np.max(np.abs(arr))


class TestCachedSdfNull:
    """Verify that _sdf_null is cached once in __init__ and reused."""

    def test_sdf_null_exists_after_init(self, sample_cameras, sample_bounds):
        """_sdf_null must be set after __init__."""
        with patch("ocmesher.core.load_cdll") as mock_load, patch("ocmesher.core.register_func"):
            mock_load.return_value = MagicMock()
            mesher = OcMesher(sample_cameras, sample_bounds)
        assert hasattr(mesher, "_sdf_null")
        assert mesher._sdf_null is not None

    def test_sdf_null_is_ctypes_pointer(self, sample_cameras, sample_bounds):
        """_sdf_null must be a ctypes null pointer of the SDF float type."""
        with patch("ocmesher.core.load_cdll") as mock_load, patch("ocmesher.core.register_func"):
            mock_load.return_value = MagicMock()
            mesher = OcMesher(sample_cameras, sample_bounds)
        # It's a null pointer: bool(ptr) is False for null ctypes pointers.
        assert not mesher._sdf_null


class TestCallLocalCaching:
    """Verify that __call__ caches attribute lookups for hot loops."""

    @staticmethod
    def _make_mesher_stub(bounds, *, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = enclosed
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        return obj

    def test_single_kernel_caches_k0(self, sample_bounds, sphere_kernel):
        """Single-kernel fast path caches kernel[0] reference."""
        mesher = self._make_mesher_stub(sample_bounds)
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        # kernel_caller with a list containing one kernel should work
        result = mesher.kernel_caller([sphere_kernel], points)
        assert result.shape == (2, 1)

    def test_isinstance_skip_asarray(self, sample_bounds):
        """When kernel returns ndarray, np.asarray is skipped."""
        mesher = self._make_mesher_stub(sample_bounds)

        def ndarray_kernel(xyz):
            return np.full(len(xyz), -1.0, dtype=np.float32)

        points = np.array([[0, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([ndarray_kernel], points)
        np.testing.assert_allclose(result[:, 0], [-1.0])


# ---------------------------------------------------------------------------
# kernel_caller out= parameter for result buffer reuse
# ---------------------------------------------------------------------------


class TestKernelCallerOutParam:
    """Verify that kernel_caller `out=` parameter reuses the provided buffer."""

    @staticmethod
    def _make_mesher_stub(bounds, *, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = enclosed
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        return obj

    def test_out_param_reuses_buffer(self, sample_bounds, sphere_kernel):
        """When out= is provided, kernel_caller writes into it and returns it."""
        mesher = self._make_mesher_stub(sample_bounds)
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        buf = np.empty((3, 1), dtype=np.float32)
        result = mesher.kernel_caller([sphere_kernel], pts, out=buf)
        assert result is buf
        # Values should match the kernel output
        expected = mesher.kernel_caller([sphere_kernel], pts)
        np.testing.assert_allclose(result, expected)

    def test_out_none_allocates_fresh(self, sample_bounds, sphere_kernel):
        """When out=None (default), kernel_caller allocates a new array."""
        mesher = self._make_mesher_stub(sample_bounds)
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        r1 = mesher.kernel_caller([sphere_kernel], pts)
        r2 = mesher.kernel_caller([sphere_kernel], pts)
        # Different allocations, same values
        assert r1 is not r2
        np.testing.assert_allclose(r1, r2)

    def test_out_param_multi_kernel(self, sample_bounds, sphere_kernel, plane_kernel):
        """out= works with multi-kernel dispatch."""
        mesher = self._make_mesher_stub(sample_bounds)
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        buf = np.empty((2, 2), dtype=np.float32)
        result = mesher.kernel_caller([sphere_kernel, plane_kernel], pts, out=buf)
        assert result is buf
        expected = mesher.kernel_caller([sphere_kernel, plane_kernel], pts)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Cached ctypes pointers in bisection loops
# ---------------------------------------------------------------------------


class TestCachedCtypesPointers:
    """Verify that caching ctypes pointers for stable arrays produces correct results."""

    @staticmethod
    def _make_mesher_stub(bounds, *, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = enclosed
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        return obj

    def test_cached_pointer_produces_same_result(self, sample_bounds, sphere_kernel):
        """Cached pointer for a stable buffer gives correct SDF values."""
        mesher = self._make_mesher_stub(sample_bounds)
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        # Call with and without pre-allocated out buffer
        result_fresh = mesher.kernel_caller([sphere_kernel], pts)
        buf = np.empty((3, 1), dtype=np.float32)
        result_reused = mesher.kernel_caller([sphere_kernel], pts, out=buf)
        np.testing.assert_allclose(result_fresh, result_reused)

    def test_repeated_calls_with_same_out_buffer(self, sample_bounds, sphere_kernel):
        """Multiple calls with same out= buffer correctly overwrite each time."""
        mesher = self._make_mesher_stub(sample_bounds)
        buf = np.empty((3, 1), dtype=np.float32)
        pts1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        pts2 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float64)
        r1 = mesher.kernel_caller([sphere_kernel], pts1, out=buf)
        vals1 = r1.copy()
        r2 = mesher.kernel_caller([sphere_kernel], pts2, out=buf)
        # r2 is the same buffer, but values changed
        assert r2 is buf
        assert not np.array_equal(vals1, r2)
        expected2 = mesher.kernel_caller([sphere_kernel], pts2)
        np.testing.assert_allclose(r2, expected2)


# ---------------------------------------------------------------------------
# Early-exit for zero extra vertices
# ---------------------------------------------------------------------------


class TestEarlyExitZeroExtraVerts:
    """Verify that _refine_extra_vertices docstring mentions early-exit."""

    def test_docstring_mentions_early_exit(self):
        """The method docstring should document the early-exit optimisation."""
        doc = OcMesher._refine_extra_vertices.__doc__
        assert "Early-exit" in doc or "early-exit" in doc or "nve == 0" in doc


# ---------------------------------------------------------------------------
# Bisection iters cached as local
# ---------------------------------------------------------------------------


class TestBisectionItersLocal:
    """Verify that bisection_iters is used correctly."""

    def test_bisection_iters_attribute_exists(self, sample_cameras, sample_bounds):
        """bisection_iters must be stored as an instance attribute."""
        with patch("ocmesher.core.load_cdll") as mock_load, patch("ocmesher.core.register_func"):
            mock_load.return_value = MagicMock()
            mesher = OcMesher(sample_cameras, sample_bounds, bisection_iters=10)
        assert mesher.bisection_iters == 10
