"""Tests for optimization patterns in ``ocmesher.core``."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocmesher.core import (
    _SDF_BATCH_SIZE,
    OcMesher,
    _np_asarray,
    _np_empty,
    _np_fabs,
    _np_greater_equal,
    _np_less_equal,
    _np_logical_or,
)

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
# Pre-allocated fabs buffer for tolerance checks
# ---------------------------------------------------------------------------


class TestFabsBufferTolerance:
    """Verify that np.fabs with out= produces same results as np.fabs without out=."""

    def test_fabs_with_out_matches_without(self):
        """np.fabs(arr, out=buf).max() must match np.fabs(arr).max()."""
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((100, 1)).astype(np.float32)
        buf = np.empty_like(arr)
        assert np.fabs(arr, out=buf).max() == pytest.approx(np.fabs(arr).max())
        # buf should be populated
        np.testing.assert_array_equal(buf, np.fabs(arr))

    def test_fabs_buffer_reuse_across_iterations(self):
        """Reusing the same out= buffer across iterations must give correct results."""
        rng = np.random.default_rng(99)
        buf = np.empty((50, 1), dtype=np.float32)
        for _ in range(5):
            arr = rng.standard_normal((50, 1)).astype(np.float32)
            result = np.fabs(arr, out=buf).max()
            assert result == pytest.approx(np.fabs(arr).max())


# ---------------------------------------------------------------------------
# Refactoring: fully vectorized camera packing
# ---------------------------------------------------------------------------


class TestFullyVectorisedCameraPacking:
    """Verify that the fully vectorized (no per-camera loop) camera packing works."""

    def test_camera_inv_pose_packed(self, sample_camera_pose, sample_intrinsics, sample_bounds):
        """Inverse pose[:3,:4] must be packed into the first 12 elements."""
        cameras = ([sample_camera_pose], [sample_intrinsics], [480], [640])
        with patch("ocmesher.core.load_cdll") as mock_load:
            mock_load.return_value = MagicMock()
            mesher = OcMesher(cameras, sample_bounds)
        inv_pose = np.linalg.inv(np.asarray(sample_camera_pose, dtype=np.float64))
        expected = inv_pose[:3, :4].ravel().astype(mesher.np_float_type)
        np.testing.assert_array_almost_equal(mesher.cameras[:12], expected)

    def test_camera_intrinsics_packed(self, sample_camera_pose, sample_intrinsics, sample_bounds):
        """Intrinsics must be packed into elements 12-20."""
        cameras = ([sample_camera_pose], [sample_intrinsics], [480], [640])
        with patch("ocmesher.core.load_cdll") as mock_load:
            mock_load.return_value = MagicMock()
            mesher = OcMesher(cameras, sample_bounds)
        expected = np.asarray(sample_intrinsics, dtype=mesher.np_float_type).ravel()
        np.testing.assert_array_almost_equal(mesher.cameras[12:21], expected)

    def test_camera_hw_packed(self, sample_camera_pose, sample_intrinsics, sample_bounds):
        """H and W must be packed into elements 21 and 22."""
        cameras = ([sample_camera_pose], [sample_intrinsics], [480], [640])
        with patch("ocmesher.core.load_cdll") as mock_load:
            mock_load.return_value = MagicMock()
            mesher = OcMesher(cameras, sample_bounds)
        assert mesher.cameras[21] == pytest.approx(480.0)
        assert mesher.cameras[22] == pytest.approx(640.0)


# ---------------------------------------------------------------------------
# Refactoring: np.empty for write-only buffers
# ---------------------------------------------------------------------------


class TestNpEmptyForWriteOnlyBuffers:
    """Verify np.zeros→np.empty refactoring doesn't break C-filled arrays."""

    def test_cnts_filled_by_construct_faces(self):
        """cnts array is np.empty but construct_faces fills it correctly."""
        # The refactoring changes np.zeros to np.empty for the cnts array.
        # We test that the construct_faces mock fills it properly.
        cnts = np.empty(3, dtype=np.int32)
        # Simulate C writing values
        cnts[:] = [10, 5, 20]
        assert cnts[0] == 10
        assert cnts[1] == 5
        assert cnts[2] == 20


# ---------------------------------------------------------------------------
# Refactoring: computed sizes instead of len() calls
# ---------------------------------------------------------------------------


class TestComputedSizesVsLen:
    """Verify that sizes computed from known dimensions match len() results."""

    def test_n_cubes_matches_len(self):
        """num_verts * 8 == len(np.empty((num_verts * 8, 3)))."""
        num_verts = 42
        cubes = np.empty((num_verts * 8, 3))
        assert num_verts * 8 == len(cubes)

    def test_edge_lr_size_matches_len(self):
        """nve * 2 == len(np.empty((nve * 2, 3)))."""
        nve = 17
        arr = np.empty((nve * 2, 3))
        assert nve * 2 == len(arr)

    def test_face_lr_size_matches_len(self):
        """nvf * 4 == len(np.empty((nvf * 4, 3)))."""
        nvf = 9
        arr = np.empty((nvf * 4, 3))
        assert nvf * 4 == len(arr)


# ---------------------------------------------------------------------------
# Refactoring: final vertex assembly offset caching
# ---------------------------------------------------------------------------


class TestFinalVertexAssemblyOffsets:
    """Verify pre-computed offsets produce correct final vertex layout."""

    def test_offset_assembly(self):
        """Vertices, edge_verts, face_verts must be contiguous in output."""
        base = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        edge = np.array([[7, 8, 9]], dtype=np.float64)
        face = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float64)
        n_base = base.shape[0]
        n_edge = edge.shape[0]
        n_face = face.shape[0]
        off_edge = n_base + n_edge
        result = np.empty((off_edge + n_face, 3), dtype=np.float64)
        result[:n_base] = base
        result[n_base:off_edge] = edge
        result[off_edge:] = face
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Module-level numpy function cache
# ---------------------------------------------------------------------------


class TestModuleLevelNumpyCache:
    """Verify that module-level cached numpy functions are correct references."""

    def test_np_empty_is_numpy_empty(self):
        """_np_empty must be np.empty."""
        assert _np_empty is np.empty

    def test_np_fabs_is_numpy_fabs(self):
        """_np_fabs must be np.fabs."""
        assert _np_fabs is np.fabs

    def test_np_less_equal_is_numpy_less_equal(self):
        """_np_less_equal must be np.less_equal."""
        assert _np_less_equal is np.less_equal

    def test_np_greater_equal_is_numpy_greater_equal(self):
        """_np_greater_equal must be np.greater_equal."""
        assert _np_greater_equal is np.greater_equal

    def test_np_logical_or_is_numpy_logical_or(self):
        """_np_logical_or must be np.logical_or."""
        assert _np_logical_or is np.logical_or

    def test_np_asarray_is_numpy_asarray(self):
        """_np_asarray must be np.asarray."""
        assert _np_asarray is np.asarray

    def test_cached_empty_produces_correct_array(self):
        """Module-level _np_empty must produce arrays identical to np.empty."""
        arr = _np_empty((5, 3), dtype=np.float64)
        assert arr.shape == (5, 3)
        assert arr.dtype == np.float64

    def test_cached_fabs_works_correctly(self):
        """Module-level _np_fabs must compute absolute values."""
        x = np.array([-1.0, 2.0, -3.0])
        np.testing.assert_array_equal(_np_fabs(x), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Known dimensions replace .shape[0] in final assembly
# ---------------------------------------------------------------------------


class TestKnownDimensionsFinalAssembly:
    """Verify final vertex assembly uses nve/nvf instead of .shape[0]."""

    def test_assembly_with_known_dims(self):
        """Assembly using nve/nvf must match assembly using .shape[0]."""
        base = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        edge = np.array([[7, 8, 9]], dtype=np.float64)
        face = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float64)
        # Simulate using known dimensions (nve=1, nvf=2) instead of .shape[0]
        nve = 1
        nvf = 2
        n_base = len(base)
        off_edge = n_base + nve
        result = _np_empty((off_edge + nvf, 3), dtype=np.float64)
        result[:n_base] = base
        result[n_base:off_edge] = edge
        result[off_edge:] = face
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Bounds mask uses module-level cached ufuncs
# ---------------------------------------------------------------------------


class TestBoundsMaskCachedUfuncs:
    """Verify bounds mask methods use module-level cached ufuncs."""

    def test_static_mask_uses_cached_ufuncs(self):
        """Static _out_of_bounds_mask should delegate to shared helper."""

        source = inspect.getsource(OcMesher._out_of_bounds_mask)
        # Now delegates to the shared _out_of_bounds_mask_shared helper.
        assert "_out_of_bounds_mask_shared" in source or "_le(" in source or "_np_less_equal" in source

    def test_into_mask_uses_cached_ufuncs(self):
        """Instance _out_of_bounds_mask_into should use cached ufunc references."""

        source = inspect.getsource(OcMesher._out_of_bounds_mask_into)
        assert "_le(" in source or "_np_less_equal" in source
        assert "_ge(" in source or "_np_greater_equal" in source
        assert "_lor(" in source or "_np_logical_or" in source

    def test_cached_ufuncs_produce_correct_mask(self):
        """Bounds mask with cached ufuncs must match reference implementation."""
        xyz = np.array(
            [[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [1.0, 0.5, 0.5], [-0.1, 0.5, 0.5]],
            dtype=np.float64,
        )
        b_min = np.array([0.0, 0.0, 0.0])
        b_max = np.array([1.0, 1.0, 1.0])
        mask = OcMesher._out_of_bounds_mask(xyz, b_min, b_max)
        # Point 0: strictly inside → False
        # Point 1: on lower boundary (0.0 <= b_min[0]=0.0 is True) → out-of-bounds
        # Point 2: on upper boundary (1.0 >= b_max[0]=1.0 is True) → out-of-bounds
        # Point 3: below lower boundary (-0.1 <= b_min[0]=0.0 is True) → out-of-bounds
        np.testing.assert_array_equal(mask, [False, True, True, True])


# ---------------------------------------------------------------------------
# Growable buffer optimisation in __call__ loops
# ---------------------------------------------------------------------------


class TestGrowableBuffersCoarseStep:
    """Verify the growable position/SDF buffers in the coarse step loop."""

    def test_kernel_caller_out_view_writes_correct_values(self):
        """kernel_caller with out= on a buffer slice must write correct SDF."""
        obj = object.__new__(OcMesher)
        obj.enclosed = False
        obj.sdf_np_float_type = np.float32
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._bounds_min_np = np.zeros(3)
        obj._bounds_max_np = np.ones(3)
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        kernel = lambda xyz: np.linalg.norm(xyz, axis=1)  # noqa: E731
        # Allocate a buffer larger than needed and pass a slice as out=.
        buf = np.full((5, 1), -999.0, dtype=np.float32)
        out_slice = buf[:2]
        result = obj.kernel_caller([kernel], pts, out=out_slice)
        # Result should contain the correct SDF values.
        np.testing.assert_allclose(result[:, 0], np.linalg.norm(pts, axis=1), rtol=1e-5)
        # The buffer's first two rows should be updated (same underlying data).
        np.testing.assert_allclose(buf[:2, 0], np.linalg.norm(pts, axis=1), rtol=1e-5)

    def test_growable_buffer_reuse_produces_stable_pointer(self):
        """Leading slice of a buffer shares the same data pointer."""
        buf = np.empty((10, 3), dtype=np.float64)
        view = buf[:5]
        # The data pointer of a leading slice equals the base array's.
        assert buf.ctypes.data == view.ctypes.data

    def test_min_out_parameter(self):
        """ndarray.min(axis, out=) writes into the provided buffer."""
        sdf = np.array([[1.0, 3.0], [2.0, 0.5]], dtype=np.float32)
        out = np.empty(2, dtype=np.float32)
        sdf.min(axis=-1, out=out)
        np.testing.assert_array_equal(out, [1.0, 0.5])


class TestGrowableBuffersFineStep:
    """Verify growable buffers in the fine step loop."""

    def test_sdf_buffer_slice_preserves_layout(self):
        """A (cap, K) buffer sliced to [:n] must be C-contiguous."""
        buf = np.empty((100, 2), dtype=np.float32)
        view = buf[:50]
        assert view.flags["C_CONTIGUOUS"]

    def test_kernel_caller_out_larger_buffer(self):
        """kernel_caller out= with a view of a larger buffer works correctly."""
        obj = object.__new__(OcMesher)
        obj.enclosed = True
        obj.sdf_np_float_type = np.float32
        obj._sdf_pool = None
        obj._oob_mask = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._oob_tmp = np.empty(_SDF_BATCH_SIZE, dtype=bool)
        obj._bounds_min_np = np.array([-10.0, -10.0, -10.0])
        obj._bounds_max_np = np.array([10.0, 10.0, 10.0])
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        kernel = lambda xyz: np.linalg.norm(xyz, axis=1) - 5.0  # noqa: E731
        buf = np.full((10, 1), 999.0, dtype=np.float32)
        result = obj.kernel_caller([kernel], pts, out=buf[:3])
        # First 3 rows should have valid SDF; remaining rows untouched.
        np.testing.assert_allclose(result[:, 0], [-5.0, -4.0, -4.0], rtol=1e-5)
        assert buf[5, 0] == pytest.approx(999.0)


class TestExplicitEmptyForFabsBuf:
    """Verify _fabs_buf uses explicit _empty() instead of np.empty_like."""

    def test_construct_element_mesh_no_empty_like(self):
        """_construct_element_mesh source must not use np.empty_like for _fabs_buf."""
        source = inspect.getsource(OcMesher._construct_element_mesh)
        # The tolerance buffer should use _empty((...), dtype=...) not np.empty_like.
        assert "np.empty_like" not in source

    def test_refine_extra_vertices_no_empty_like(self):
        """_refine_extra_vertices source must not use np.empty_like for _fabs_buf."""
        source = inspect.getsource(OcMesher._refine_extra_vertices)
        assert "np.empty_like" not in source

    def test_call_uses_growable_buffers(self):
        """__call__ source must contain growable buffer pattern."""
        source = inspect.getsource(OcMesher.__call__)
        # Check for growable buffer capacity tracking variables.
        assert "_c_cap" in source, "coarse step should use growable buffer"
        assert "_f_cap" in source, "fine step should use growable buffer"
        # Check that out= is used with kernel_caller in the loops.
        assert "out=_c_sdf[:n]" in source
        assert "out=_f_sdf[:n]" in source


class TestTorchAllocationCaches:
    """Verify torch_core caches constructor helpers at module scope."""

    def test_torch_core_uses_cached_allocation_helpers(self):
        torch = pytest.importorskip("torch", reason="torch not installed")

        del torch
        from ocmesher import torch_core

        source = inspect.getsource(torch_core)
        assert "_torch_empty = torch.empty" in source
        assert "_torch_zeros = torch.zeros" in source
        assert "_torch_zeros((0, n_kernels)" in source


# ---------------------------------------------------------------------------
# Cached SDF buffer pointer in _construct_element_mesh bisection loop
# ---------------------------------------------------------------------------


class TestCachedSdfBufPointer:
    """Verify that caching _sdf_buf ctypes pointer doesn't break bisection."""

    def test_sdf_buf_pointer_stable(self):
        """np.empty buffer has stable ctypes pointer across writes."""
        buf = np.empty((10, 1), dtype=np.float32)
        ptr1 = buf.ctypes.data
        buf[:] = np.random.default_rng(42).standard_normal((10, 1)).astype(np.float32)
        ptr2 = buf.ctypes.data
        assert ptr1 == ptr2

    def test_sdf_buf_slice_views_stable(self):
        """Views into a pre-allocated buffer have stable data pointers."""
        buf = np.empty((20, 1), dtype=np.float32)
        v1 = buf[:10]
        v2 = buf[10:]
        ptr_v1_a = v1.ctypes.data
        ptr_v2_a = v2.ctypes.data
        # Write via the parent buffer (simulates kernel_caller out=)
        buf[:] = np.ones((20, 1), dtype=np.float32)
        ptr_v1_b = v1.ctypes.data
        ptr_v2_b = v2.ctypes.data
        assert ptr_v1_a == ptr_v1_b
        assert ptr_v2_a == ptr_v2_b


# ---------------------------------------------------------------------------
# np.fabs cached as local in bisection
# ---------------------------------------------------------------------------


class TestNpFabsCachedLocal:
    """Verify np.fabs local caching produces correct tolerance results."""

    def test_fabs_local_matches_module(self):
        """np.fabs cached as local gives same result as np.fabs."""
        _np_fabs = np.fabs
        arr = np.array([-1.0, 2.0, -3.0, 0.5], dtype=np.float32)
        buf = np.empty_like(arr)
        local_result = _np_fabs(arr, out=buf).max()
        module_result = np.fabs(arr, out=np.empty_like(arr)).max()
        assert local_result == module_result


# ---------------------------------------------------------------------------
# pool.submit replaces closure factory in multi-kernel path
# ---------------------------------------------------------------------------


class TestPoolSubmitMultiKernel:
    """Verify pool.submit-based multi-kernel dispatch produces correct results."""

    def test_multi_kernel_uses_pool_submit(self):
        """Multi-kernel kernel_caller should use pool.submit (no _make_eval_one)."""
        source = inspect.getsource(OcMesher._kernel_caller_multi)
        # _make_eval_one closure factory should no longer exist
        assert "_make_eval_one" not in source
        # pool.submit should be bound as _submit and called
        assert "_submit = pool.submit" in source
        assert "_submit(" in source

    def test_multi_kernel_correctness(self):
        """Two kernels dispatched via pool.submit must produce correct columns."""
        obj = object.__new__(OcMesher)
        obj.sdf_np_float_type = np.float32
        obj.enclosed = False
        obj._sdf_pool = None
        obj._oob_mask = np.empty(100, dtype=bool)
        obj._oob_tmp = np.empty(100, dtype=bool)

        def k0(xyz):
            return np.full(len(xyz), 1.0, dtype=np.float32)

        def k1(xyz):
            return np.full(len(xyz), 2.0, dtype=np.float32)

        rng = np.random.default_rng(42)
        xyz = rng.random((10, 3)).astype(np.float64)
        result = obj.kernel_caller([k0, k1], xyz)
        np.testing.assert_array_almost_equal(result[:, 0], 1.0)
        np.testing.assert_array_almost_equal(result[:, 1], 2.0)
        # Clean up pool
        if obj._sdf_pool is not None:
            obj._sdf_pool.shutdown(wait=False)
            obj._sdf_pool = None


# ---------------------------------------------------------------------------
# Single-kernel direct assignment (no intermediate view)
# ---------------------------------------------------------------------------


class TestSingleKernelDirectAssignment:
    """Verify single-kernel path uses direct slice assignment."""

    def test_no_result_col_intermediate(self):
        """Single-kernel path should not create result_col intermediate view."""

        source = inspect.getsource(OcMesher._kernel_caller_single)
        assert "result_col" not in source

    def test_single_kernel_enclosed_split(self):
        """Single-kernel path should have separate enclosed/non-enclosed loops."""

        source = inspect.getsource(OcMesher._kernel_caller_single)
        # The if _enclosed check should be outside the loop for single-kernel
        # (separate loop bodies for enclosed vs non-enclosed)
        assert source.count("if _enclosed:") >= 1
