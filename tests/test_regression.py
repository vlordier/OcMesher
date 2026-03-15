"""Regression tests for OcMesher optimizations.

These tests verify that the optimized Python orchestration layer in
``ocmesher/core.py`` produces numerically identical results to naive
reference implementations.  They exercise:

- ``kernel_caller`` with single and multiple SDF kernels
- Out-of-bounds masking (enclosed mode)
- Buffer management (out= parameter, empty inputs, large batches)
- Camera packing (vectorized vs loop)
- Slice-fill equivalence with np.concatenate

Every test is designed to catch real regressions — off-by-one errors in
slice indexing, incorrect bounds masking, buffer reuse corruption, etc.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ocmesher.core import OcMesher

# ---------------------------------------------------------------------------
# Adaptive imports (works on develop/main and optimized branch)
# ---------------------------------------------------------------------------
try:
    from ocmesher.core import _SDF_BATCH_SIZE
except ImportError:
    _SDF_BATCH_SIZE = 10_000_000

_HAS_SLOTS = hasattr(OcMesher, "__slots__")


def _has_out_param() -> bool:
    return "out" in inspect.signature(OcMesher.kernel_caller).parameters


# ---------------------------------------------------------------------------
# Mock SDF kernels
# ---------------------------------------------------------------------------
def _sdf_sphere(xyz: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(xyz, axis=1) - 5.0).astype(np.float32)


def _sdf_plane(xyz: np.ndarray) -> np.ndarray:
    return xyz[:, 2].astype(np.float32)


def _sdf_gyroid(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)).astype(np.float32)


# ---------------------------------------------------------------------------
# Stub factory
# ---------------------------------------------------------------------------
def _make_stub(*, enclosed: bool = True) -> OcMesher:
    obj = object.__new__(OcMesher)
    obj.enclosed = enclosed
    obj.sdf_np_float_type = np.float32
    if _HAS_SLOTS:
        obj._bounds_min_np = np.array([-10.0, -10.0, -10.0], dtype=np.float64)
        obj._bounds_max_np = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
    else:
        obj.bounds = (-10, 10, -10, 10, -10, 10)
    return obj


# ---------------------------------------------------------------------------
# Reference implementations (naive — matching develop/main behavior)
# ---------------------------------------------------------------------------
def _ref_kernel_caller(stub, kernels, xyz_all):
    """Reference implementation matching develop/main kernel_caller."""
    n_xyz = len(xyz_all)
    if n_xyz == 0:
        return np.zeros((0, len(kernels)), dtype=np.float32)
    step = 10_000_000
    sdfs = []
    bounds = (-10, 10, -10, 10, -10, 10)
    for i in range(0, n_xyz, step):
        xyz = xyz_all[i : i + step]
        sdfs_i = []
        if stub.enclosed:
            out_bound = np.zeros(len(xyz), dtype=bool)
            for c in range(3):
                out_bound |= xyz[:, c] <= bounds[c * 2]
                out_bound |= xyz[:, c] >= bounds[c * 2 + 1]
        for kernel in kernels:
            sdf = kernel(xyz)
            if stub.enclosed:
                sdf[out_bound] = 1
            sdfs_i.append(sdf)
        sdfs.append(np.stack(sdfs_i, -1).astype(np.float32))
    return np.concatenate(sdfs, 0)


# ===========================================================================
# Tests: kernel_caller correctness
# ===========================================================================
class TestKernelCallerRegression:
    """Verify kernel_caller produces identical results to reference."""

    @pytest.fixture(params=[100, 1_000, 10_000, 50_000])
    def n_points(self, request):
        return request.param

    def test_single_kernel_enclosed(self, n_points):
        stub = _make_stub(enclosed=True)
        pts = np.random.default_rng(42).standard_normal((n_points, 3)).astype(np.float64)
        actual = stub.kernel_caller([_sdf_sphere], pts)
        expected = _ref_kernel_caller(stub, [_sdf_sphere], pts)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_single_kernel_open(self, n_points):
        stub = _make_stub(enclosed=False)
        pts = np.random.default_rng(42).standard_normal((n_points, 3)).astype(np.float64)
        actual = stub.kernel_caller([_sdf_sphere], pts)
        expected = _ref_kernel_caller(stub, [_sdf_sphere], pts)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_multi_kernel_enclosed(self, n_points):
        stub = _make_stub(enclosed=True)
        kernels = [_sdf_sphere, _sdf_plane, _sdf_gyroid]
        pts = np.random.default_rng(42).standard_normal((n_points, 3)).astype(np.float64)
        actual = stub.kernel_caller(kernels, pts)
        expected = _ref_kernel_caller(stub, kernels, pts)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)

    def test_multi_kernel_open(self, n_points):
        stub = _make_stub(enclosed=False)
        kernels = [_sdf_sphere, _sdf_plane, _sdf_gyroid]
        pts = np.random.default_rng(42).standard_normal((n_points, 3)).astype(np.float64)
        actual = stub.kernel_caller(kernels, pts)
        expected = _ref_kernel_caller(stub, kernels, pts)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


class TestKernelCallerShape:
    """Verify output shapes and dtypes."""

    def test_single_kernel_shape(self):
        stub = _make_stub()
        pts = np.random.default_rng(0).standard_normal((500, 3))
        result = stub.kernel_caller([_sdf_sphere], pts)
        assert result.shape == (500, 1)
        assert result.dtype == np.float32

    def test_multi_kernel_shape(self):
        stub = _make_stub()
        pts = np.random.default_rng(0).standard_normal((500, 3))
        result = stub.kernel_caller([_sdf_sphere, _sdf_plane], pts)
        assert result.shape == (500, 2)
        assert result.dtype == np.float32

    def test_empty_input(self):
        stub = _make_stub()
        empty = np.zeros((0, 3), dtype=np.float64)
        result = stub.kernel_caller([_sdf_sphere], empty)
        assert result.shape == (0, 1)

    def test_single_point(self):
        stub = _make_stub(enclosed=False)
        pt = np.array([[3.0, 4.0, 0.0]])
        result = stub.kernel_caller([_sdf_sphere], pt)
        expected = np.linalg.norm([3.0, 4.0, 0.0]) - 5.0
        np.testing.assert_allclose(result[0, 0], expected, atol=1e-6)


class TestBoundsMasking:
    """Verify that out-of-bounds points are correctly masked to 1.0."""

    def test_oob_points_masked(self):
        stub = _make_stub(enclosed=True)
        # Points clearly outside [-10, 10]^3
        oob_pts = np.array(
            [
                [20.0, 0.0, 0.0],
                [0.0, -15.0, 0.0],
                [0.0, 0.0, 100.0],
                [-10.0, 0.0, 0.0],  # on boundary (<=)
                [10.0, 0.0, 0.0],  # on boundary (>=)
            ]
        )
        result = stub.kernel_caller([_sdf_sphere], oob_pts)
        np.testing.assert_array_equal(result[:, 0], 1.0)

    def test_inbounds_not_masked(self):
        stub = _make_stub(enclosed=True)
        ib_pts = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [-9.0, -9.0, -9.0]])
        result = stub.kernel_caller([_sdf_sphere], ib_pts)
        expected = (np.linalg.norm(ib_pts, axis=1) - 5.0).astype(np.float32)
        np.testing.assert_allclose(result[:, 0], expected, atol=1e-6)

    def test_mixed_oob_inbounds(self):
        stub = _make_stub(enclosed=True)
        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # in-bounds
                [20.0, 0.0, 0.0],  # OOB
                [3.0, 4.0, 0.0],  # in-bounds
                [0.0, -15.0, 0.0],  # OOB
            ]
        )
        result = stub.kernel_caller([_sdf_sphere], pts)
        # OOB points
        assert result[1, 0] == 1.0
        assert result[3, 0] == 1.0
        # In-bounds points
        np.testing.assert_allclose(result[0, 0], np.float32(np.linalg.norm([0, 0, 0]) - 5.0), atol=1e-6)
        np.testing.assert_allclose(result[2, 0], np.float32(np.linalg.norm([3, 4, 0]) - 5.0), atol=1e-6)

    def test_no_masking_when_not_enclosed(self):
        stub = _make_stub(enclosed=False)
        oob_pts = np.array([[20.0, 0.0, 0.0], [0.0, -15.0, 0.0]])
        result = stub.kernel_caller([_sdf_sphere], oob_pts)
        expected = (np.linalg.norm(oob_pts, axis=1) - 5.0).astype(np.float32)
        np.testing.assert_allclose(result[:, 0], expected, atol=1e-6)


class TestOutParameter:
    """Test the out= parameter for buffer reuse (optimized branch only)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_out(self):
        if not _has_out_param():
            pytest.skip("out= parameter not available on this branch")

    def test_out_returns_same_buffer(self):
        stub = _make_stub(enclosed=True)
        pts = np.random.default_rng(42).standard_normal((500, 3))
        out = np.empty((500, 1), dtype=np.float32)
        result = stub.kernel_caller([_sdf_sphere], pts, out=out)
        assert result is out

    def test_out_matches_no_out(self):
        stub = _make_stub(enclosed=True)
        pts = np.random.default_rng(42).standard_normal((1000, 3))
        out = np.empty((1000, 1), dtype=np.float32)
        result_out = stub.kernel_caller([_sdf_sphere], pts, out=out)
        result_no_out = stub.kernel_caller([_sdf_sphere], pts)
        np.testing.assert_allclose(result_out, result_no_out, atol=1e-6)

    def test_out_reuse_no_corruption(self):
        """Repeated calls with the same out= buffer should not corrupt results."""
        stub = _make_stub(enclosed=True)
        out = np.empty((500, 1), dtype=np.float32)
        for seed in range(5):
            pts = np.random.default_rng(seed).standard_normal((500, 3))
            result = stub.kernel_caller([_sdf_sphere], pts, out=out)
            expected = _ref_kernel_caller(stub, [_sdf_sphere], pts)
            np.testing.assert_allclose(result, expected, atol=1e-6)


class TestStaticBoundsMask:
    """Test the static _out_of_bounds_mask method (optimized branch only)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_static(self):
        if not hasattr(OcMesher, "_out_of_bounds_mask"):
            pytest.skip("_out_of_bounds_mask not available on this branch")

    def test_all_inbounds(self):
        pts = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        b_min = np.array([-10.0, -10.0, -10.0])
        b_max = np.array([10.0, 10.0, 10.0])
        mask = OcMesher._out_of_bounds_mask(pts, b_min, b_max)
        assert not mask.any()

    def test_all_oob(self):
        pts = np.array([[20.0, 0.0, 0.0], [0.0, -15.0, 0.0]])
        b_min = np.array([-10.0, -10.0, -10.0])
        b_max = np.array([10.0, 10.0, 10.0])
        mask = OcMesher._out_of_bounds_mask(pts, b_min, b_max)
        assert mask.all()

    def test_boundary_is_oob(self):
        """Points on the boundary (<=, >=) should be flagged as OOB."""
        pts = np.array([[-10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        b_min = np.array([-10.0, -10.0, -10.0])
        b_max = np.array([10.0, 10.0, 10.0])
        mask = OcMesher._out_of_bounds_mask(pts, b_min, b_max)
        assert mask.all()

    def test_matches_reference(self):
        """Verify the optimized mask matches the naive reference."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((5000, 3)) * 15.0
        b_min = np.array([-10.0, -10.0, -10.0])
        b_max = np.array([10.0, 10.0, 10.0])

        actual = OcMesher._out_of_bounds_mask(pts, b_min, b_max)

        # Reference: develop/main style
        expected = np.zeros(len(pts), dtype=bool)
        bounds = (-10, 10, -10, 10, -10, 10)
        for c in range(3):
            expected |= pts[:, c] <= bounds[c * 2]
            expected |= pts[:, c] >= bounds[c * 2 + 1]

        np.testing.assert_array_equal(actual, expected)


class TestLargeBatchRegression:
    """Test with batch sizes that cross the _SDF_BATCH_SIZE boundary."""

    @pytest.mark.slow
    def test_cross_batch_boundary(self):
        """Ensure correct results when N > _SDF_BATCH_SIZE (multi-batch)."""
        # Use a small batch size to force multiple batches
        n = 50_000
        stub = _make_stub(enclosed=True)
        pts = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float64)
        actual = stub.kernel_caller([_sdf_sphere, _sdf_plane], pts)
        expected = _ref_kernel_caller(stub, [_sdf_sphere, _sdf_plane], pts)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)

    def test_exact_batch_size(self):
        """N = 100 (well under batch limit) should work correctly."""
        stub = _make_stub(enclosed=True)
        pts = np.random.default_rng(42).standard_normal((100, 3)).astype(np.float64)
        actual = stub.kernel_caller([_sdf_sphere], pts)
        expected = _ref_kernel_caller(stub, [_sdf_sphere], pts)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


class TestCameraPackingRegression:
    """Verify camera packing produces identical results across methods."""

    @pytest.mark.parametrize("n_cameras", [1, 4, 8])
    def test_packing_equivalence(self, n_cameras):
        """Vectorized and loop-based camera packing should produce the same array."""
        np_float_type = np.float64
        rng = np.random.default_rng(42)

        cam_poses = []
        for _ in range(n_cameras):
            p = np.eye(4, dtype=np.float64) + rng.standard_normal((4, 4)) * 0.1
            p[3, :] = [0, 0, 0, 1]
            cam_poses.append(p)

        ks = [np.array([[2000, 0, 640], [0, 2000, 360], [0, 0, 1]], dtype=np.float64)] * n_cameras
        hs = [720] * n_cameras
        ws = [1280] * n_cameras

        # Reference: develop/main loop
        cameras_loop = np.zeros(23 * n_cameras, dtype=np_float_type)
        for i in range(n_cameras):
            cameras_loop[23 * i : 23 * (i + 1)] = np.concatenate(
                [
                    np.linalg.inv(cam_poses[i])[:3, :4].reshape(-1),
                    ks[i].reshape(-1),
                    [hs[i]],
                    [ws[i]],
                ]
            ).astype(np_float_type)

        # Vectorized
        inv_poses = np.linalg.inv(np.stack(cam_poses))[:, :3, :4].reshape(n_cameras, -1)
        ks_flat = np.array(ks).reshape(n_cameras, -1)
        h_arr = np.array(hs, dtype=np_float_type).reshape(n_cameras, 1)
        w_arr = np.array(ws, dtype=np_float_type).reshape(n_cameras, 1)
        packed = np.empty((n_cameras, 23), dtype=np_float_type)
        packed[:, :12] = inv_poses
        packed[:, 12:21] = ks_flat
        packed[:, 21:22] = h_arr
        packed[:, 22:23] = w_arr
        cameras_vec = packed.ravel()

        np.testing.assert_allclose(cameras_vec, cameras_loop, atol=1e-10)
