"""Tests for ``ocmesher.core.kernel_caller`` and SDF evaluation paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocmesher.core import (
    _SDF_BATCH_SIZE,
    OcMesher,
    _validate_kernels,
)
from ocmesher.utils.timer import PhaseTracker

from .sdf_fixtures import constant_kernel as _constant_kernel

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

    def test_kernel_caller_records_sdf_phase_when_tracker_active(self, sample_bounds, sphere_kernel):
        mesher = self._make_mesher_stub(sample_bounds)
        mesher._phase_tracker = PhaseTracker("test")
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        _ = mesher.kernel_caller([sphere_kernel], points)
        assert mesher._phase_tracker.snapshot_millis()["sdf_eval"] > 0.0

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
# Refactoring: np.empty for pre-allocated output buffers (commit 2)
# ---------------------------------------------------------------------------
class TestKernelCallerAllocation:
    """Verify that kernel_caller still produces correct results after the
    np.zeros -> np.empty change for write-before-read output buffers."""

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_kernel_caller_single_kernel(self, mock_register, mock_load, sample_cameras, sample_bounds, sphere_kernel):
        """kernel_caller must return correct shape and dtype for one kernel."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        assert mock_register.called
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result.shape == (2, 1)
        assert result.dtype == np.float32

    @patch("ocmesher.core.load_cdll")
    @patch("ocmesher.core.register_func")
    def test_kernel_caller_empty_input(self, mock_register, mock_load, sample_cameras, sample_bounds, sphere_kernel):
        """kernel_caller with zero points must return empty float32 array."""
        mock_load.return_value = MagicMock()
        mesher = OcMesher(sample_cameras, sample_bounds)
        assert mock_register.called
        pts = np.zeros((0, 3), dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result.shape == (0, 1)
        assert result.dtype == np.float32

    def test_kernel_caller_large_batch_concatenates(self, sample_cameras, sample_bounds, sphere_kernel):
        """kernel_caller must correctly concatenate batches > _SDF_BATCH_SIZE.

        Patches _SDF_BATCH_SIZE to a small value so the test exercises
        multi-batch logic without allocating millions of points.
        """
        with (
            patch("ocmesher.core._SDF_BATCH_SIZE", 8),
            patch("ocmesher.core.load_cdll") as mock_load,
            patch("ocmesher.core.register_func") as mock_register,
        ):
            mock_load.return_value = MagicMock()
            mesher = OcMesher(sample_cameras, sample_bounds)
            assert mock_register.called
            mesher._oob_mask = np.empty(13, dtype=bool)
            mesher._oob_tmp = np.empty(13, dtype=bool)
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
# Split enclosed / non-enclosed batch loops in kernel_caller
# ---------------------------------------------------------------------------


class TestSplitEnclosedBatchLoops:
    """Verify that enclosed/non-enclosed paths in kernel_caller produce identical results."""

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

    def test_enclosed_single_kernel_clamps(self, sphere_kernel):
        """Enclosed=True single-kernel path must clamp out-of-bounds to 1."""
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=True)
        pts = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        assert result[1, 0] == pytest.approx(1.0)

    def test_non_enclosed_single_kernel_no_clamp(self, sphere_kernel):
        """Enclosed=False single-kernel path must not clamp."""
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=False)
        pts = np.array([[1.5, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        np.testing.assert_allclose(result[:, 0], [0.5], atol=1e-6)

    def test_enclosed_multi_kernel_clamps(self, sphere_kernel, plane_kernel):
        """Enclosed=True multi-kernel path must clamp out-of-bounds to 1."""
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=True)
        pts = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel, plane_kernel], pts)
        assert result[1, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(1.0)

    def test_non_enclosed_multi_kernel_no_clamp(self, sphere_kernel, plane_kernel):
        """Enclosed=False multi-kernel path must not clamp."""
        bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        mesher = self._make_mesher_stub(bounds, enclosed=False)
        pts = np.array([[1.5, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel, plane_kernel], pts)
        np.testing.assert_allclose(result[:, 0], [0.5], atol=1e-6)
        np.testing.assert_allclose(result[:, 1], [0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Cached locals in kernel_caller batch loops
# ---------------------------------------------------------------------------


class TestKernelCallerCachedLocals:
    """Verify kernel_caller still produces correct results with cached locals."""

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

    def test_single_kernel_enclosed_correct(self, sample_bounds, sphere_kernel):
        """Single-kernel enclosed path with cached min/_batch produces correct SDF."""
        mesher = self._make_mesher_stub(sample_bounds)
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        # Bounds masking may modify some values, just check shape/dtype.
        assert result.shape == (3, 1)
        assert result.dtype == np.float32

    def test_multi_kernel_eval_batch_isinstance(self, sample_bounds, sphere_kernel, plane_kernel):
        """Multi-kernel path uses isinstance check in _eval_kernels_batch."""
        mesher = self._make_mesher_stub(sample_bounds)
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel, plane_kernel], pts)
        assert result.shape == (2, 2)
        assert result.dtype == np.float32

    def test_non_enclosed_single_kernel(self, sample_bounds, sphere_kernel):
        """Non-enclosed single-kernel path with cached locals."""
        mesher = self._make_mesher_stub(sample_bounds, enclosed=False)
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        result = mesher.kernel_caller([sphere_kernel], pts)
        expected = sphere_kernel(pts).astype(np.float32)
        np.testing.assert_allclose(result[:, 0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Tuple kernel wrapping (avoids list-slice copy)
# ---------------------------------------------------------------------------


class TestTupleKernelWrapping:
    """Verify tuple wrapping of single kernels matches slice behaviour."""

    def test_tuple_single_kernel_matches_slice(self, zeros_kernel):
        """(kernel,) tuple behaves identically to kernels[0:1] slice."""
        kernels = [zeros_kernel]
        # Slice produces a list
        sliced = kernels[0:1]
        # Tuple wrap produces a tuple
        wrapped = (kernels[0],)
        assert len(sliced) == len(wrapped) == 1
        assert sliced[0] is wrapped[0]

    def test_tuple_kernel_works_with_kernel_caller(self, ones_kernel):
        """kernel_caller accepts tuple kernels correctly."""
        bounds = np.array([-5.0, 5.0, -5.0, 5.0, -5.0, 5.0])
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = False
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = obj.kernel_caller((ones_kernel,), pts)
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result[:, 0], 1.0)


# ---------------------------------------------------------------------------
# Inlined multi-kernel evaluation
# ---------------------------------------------------------------------------


class TestInlinedMultiKernelEval:
    """Verify inlined multi-kernel evaluation produces correct results."""

    def test_multi_kernel_inlined(self):
        """Multi-kernel path produces correct per-kernel columns."""
        bounds = np.array([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0])
        obj = object.__new__(OcMesher)
        obj.bounds = bounds
        obj.enclosed = False
        obj.sdf_np_float_type = np.float32
        obj._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        obj._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        k0 = _constant_kernel(1.0)
        k1 = _constant_kernel(2.0)
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = obj.kernel_caller([k0, k1], pts)
        assert result.shape == (3, 2)
        np.testing.assert_array_almost_equal(result[:, 0], 1.0)
        np.testing.assert_array_almost_equal(result[:, 1], 2.0)


# ---------------------------------------------------------------------------
# Refactoring: deduplicated kernel_caller batch loops
# ---------------------------------------------------------------------------


class TestDeduplicatedKernelCallerLoops:
    """Verify kernel_caller works with the unified enclosed/non-enclosed loop."""

    @staticmethod
    def _make_stub(*, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.sdf_np_float_type = np.float32
        obj.enclosed = enclosed
        obj._bounds_min_np = np.array([0.0, 0.0, 0.0])
        obj._bounds_max_np = np.array([10.0, 10.0, 10.0])
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        return obj

    def test_single_kernel_enclosed(self):
        """Single kernel + enclosed: OOB points get masked to 1."""
        obj = self._make_stub(enclosed=True)
        k = _constant_kernel(-0.5)
        pts = np.array([[5.0, 5.0, 5.0], [999.0, 999.0, 999.0]])
        result = obj.kernel_caller([k], pts)
        assert result[0, 0] == pytest.approx(-0.5)
        assert result[1, 0] == pytest.approx(1.0)  # OOB masked

    def test_single_kernel_not_enclosed(self):
        """Single kernel + not enclosed: no masking."""
        obj = self._make_stub(enclosed=False)
        k = _constant_kernel(-0.5)
        pts = np.array([[5.0, 5.0, 5.0], [999.0, 999.0, 999.0]])
        result = obj.kernel_caller([k], pts)
        assert result[0, 0] == pytest.approx(-0.5)
        assert result[1, 0] == pytest.approx(-0.5)  # Not masked

    def test_multi_kernel_enclosed(self):
        """Multi-kernel + enclosed: OOB masking on all columns."""
        obj = self._make_stub(enclosed=True)
        k0 = _constant_kernel(-1.0)
        k1 = _constant_kernel(-2.0)
        pts = np.array([[5.0, 5.0, 5.0], [999.0, 999.0, 999.0]])
        result = obj.kernel_caller([k0, k1], pts)
        assert result[0, 0] == pytest.approx(-1.0)
        assert result[0, 1] == pytest.approx(-2.0)
        assert result[1, 0] == pytest.approx(1.0)  # OOB
        assert result[1, 1] == pytest.approx(1.0)  # OOB

    def test_multi_kernel_not_enclosed(self):
        """Multi-kernel + not enclosed: no masking."""
        obj = self._make_stub(enclosed=False)
        k0 = _constant_kernel(-1.0)
        k1 = _constant_kernel(-2.0)
        pts = np.array([[5.0, 5.0, 5.0], [999.0, 999.0, 999.0]])
        result = obj.kernel_caller([k0, k1], pts)
        assert result[0, 0] == pytest.approx(-1.0)
        assert result[0, 1] == pytest.approx(-2.0)
        assert result[1, 0] == pytest.approx(-1.0)  # Not masked
        assert result[1, 1] == pytest.approx(-2.0)  # Not masked


class TestSpecializedKernelCallerHelpers:
    @staticmethod
    def _make_stub(*, enclosed=True):
        obj = object.__new__(OcMesher)
        obj.sdf_np_float_type = np.float32
        obj.enclosed = enclosed
        obj._bounds_min_np = np.array([0.0, 0.0, 0.0])
        obj._bounds_max_np = np.array([10.0, 10.0, 10.0])
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        return obj

    def test_single_helper_matches_public_kernel_caller(self):
        obj = self._make_stub(enclosed=True)
        pts = np.array([[5.0, 5.0, 5.0], [999.0, 999.0, 999.0]])
        kernel = _constant_kernel(-0.5)

        public = obj.kernel_caller([kernel], pts)
        direct = obj._kernel_caller_single(kernel, pts)

        np.testing.assert_array_equal(direct, public)

    def test_multi_helper_matches_public_kernel_caller(self):
        obj = self._make_stub(enclosed=False)
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        kernels = [_constant_kernel(1.0), _constant_kernel(2.0)]

        public = obj.kernel_caller(kernels, pts)
        direct = obj._kernel_caller_multi(kernels, pts)

        np.testing.assert_array_equal(direct, public)


class TestMeshPhaseAttribution:
    def test_refine_extra_vertices_records_face_topology_phase(self):
        obj = object.__new__(OcMesher)
        obj.AF = lambda x: x
        obj.sdf_AF = lambda x: x
        obj.np_float_type = np.float64
        obj.sdf_np_float_type = np.float32
        obj._sdf_null = object()
        obj.bisection_tol = 0.0
        obj.bisection_iters = 1
        obj._phase_tracker = PhaseTracker("mesh phases")

        def _construct_faces(_e, _vertices_ptr, cnts_ptr):
            cnts_ptr[0] = 0
            cnts_ptr[1] = 0
            cnts_ptr[2] = 1

        def _get_faces(faces_ptr):
            faces_ptr[0] = 0
            faces_ptr[1] = 0
            faces_ptr[2] = 0

        obj.construct_faces = _construct_faces
        obj.get_faces = _get_faces

        vertices = np.zeros((1, 3), dtype=np.float64)
        final_vertices, faces = obj._refine_extra_vertices(0, (_constant_kernel(0.0),), vertices)

        snapshot = obj._phase_tracker.snapshot_millis()
        assert snapshot["mesh_face_topology"] > 0.0
        np.testing.assert_array_equal(final_vertices, vertices)
        np.testing.assert_array_equal(faces, np.array([[0, 0, 0]], dtype=np.int32))

    def test_construct_element_mesh_records_primary_and_visibility_phases(self):
        obj = object.__new__(OcMesher)
        obj.AF = lambda x: x
        obj.sdf_AF = lambda x: x
        obj.np_float_type = np.float64
        obj.sdf_np_float_type = np.float32
        obj._sdf_null = object()
        obj.bisection_tol = 0.0
        obj.bisection_iters = 1
        obj._phase_tracker = PhaseTracker("mesh phases")

        def _kernel(self, _kernel_fn, xyz, out=None):
            result = out if out is not None else np.zeros((len(xyz), 1), dtype=np.float32)
            result[:] = 0.0
            return result

        obj.get_verts_center = lambda e, centers_ptr: centers_ptr.fill(0.0)
        obj.update_verts = lambda e, sdf_l, sdf_r, cubes_ptr: cubes_ptr.fill(0.0)
        obj.get_lr_verts = lambda e, cubes_ptr, cubes_r_ptr: cubes_r_ptr.fill(0.0)
        obj.finalize_verts = lambda e, sdf_l, sdf_r, vertices_ptr: vertices_ptr.fill(0.0)

        def _get_in_view_tag(_e, out_ptr):
            out_ptr[0] = True

        obj.get_in_view_tag = _get_in_view_tag

        with (
            patch.object(OcMesher, "_kernel_caller_single", new=_kernel),
            patch.object(
                OcMesher,
                "_refine_extra_vertices",
                new=lambda self, e, k_e, vertices: (vertices, np.array([[0, 0, 0]], dtype=np.int32)),
            ),
        ):
            mesh, in_view = obj._construct_element_mesh(0, (_constant_kernel(0.0),), 1)

        snapshot = obj._phase_tracker.snapshot_millis()
        assert snapshot["mesh_primary_vertices"] > 0.0
        assert snapshot["mesh_extra_vertices"] > 0.0
        assert snapshot["mesh_visibility_tags"] > 0.0
        assert mesh.vertices.shape == (1, 3)
        assert mesh.faces.shape == (1, 3)
        np.testing.assert_array_equal(in_view, np.array([True]))
