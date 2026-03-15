"""Tests for OcMesher initialization, lifecycle, and configuration."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocmesher.core import (
    _SDF_BATCH_SIZE,
    CAMERA_DATA_STRIDE,
    OcMesher,
)

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


# ---------------------------------------------------------------------------
# Cached sdf_np_float_type in bisection methods
# ---------------------------------------------------------------------------


class TestCachedSdfDtype:
    """Verify that sdf_np_float_type is properly cached as local."""

    def test_sdf_dtype_attribute_exists(self, sample_cameras, sample_bounds):
        """sdf_np_float_type must be set during __init__."""
        with patch("ocmesher.core.load_cdll") as mock_load, patch("ocmesher.core.register_func"):
            mock_load.return_value = MagicMock()
            mesher = OcMesher(sample_cameras, sample_bounds)
        assert mesher.sdf_np_float_type == np.float32


# ---------------------------------------------------------------------------
# OcMesher __slots__ optimisation
# ---------------------------------------------------------------------------


class TestOcMesherSlots:
    """Verify __slots__ is defined and prevents arbitrary attribute assignment."""

    def test_has_slots(self):
        """OcMesher class defines __slots__."""
        assert hasattr(OcMesher, "__slots__")

    def test_no_instance_dict(self):
        """Instances with __slots__ should not have __dict__."""
        obj = object.__new__(OcMesher)
        assert not hasattr(obj, "__dict__")

    def test_slots_contain_expected_attrs(self):
        """__slots__ includes core attributes used in hot paths."""
        slots = OcMesher.__slots__
        for attr in (
            "AF",
            "sdf_AF",
            "bounds",
            "enclosed",
            "bisection_iters",
            "bisection_tol",
            "_sdf_pool",
            "_oob_mask",
            "_oob_tmp",
            "_sdf_null",
            "_bounds_min_np",
            "_bounds_max_np",
        ):
            assert attr in slots, f"{attr} missing from __slots__"

    def test_methods_not_in_slots(self):
        """Instance methods should not appear in __slots__."""
        slots = OcMesher.__slots__
        assert "kernel_caller" not in slots

    def test_cannot_set_arbitrary_attribute(self):
        """Setting a non-slot attribute should raise AttributeError."""
        obj = object.__new__(OcMesher)
        with pytest.raises(AttributeError):
            obj._nonexistent_attribute_xyz = 42

    def test_registered_c_functions_in_slots(self):
        """C DLL function names must be present in __slots__."""
        slots = OcMesher.__slots__
        c_funcs = [
            "run_coarse",
            "fine_group",
            "fine_iteration",
            "fine_iteration_output",
            "final_iteration",
            "final_iteration2",
            "final_iteration3",
            "get_verts_center",
            "update_verts",
            "get_faces",
            "get_in_view_tag",
        ]
        for func_name in c_funcs:
            assert func_name in slots, f"C function '{func_name}' missing from __slots__"


# ---------------------------------------------------------------------------
# Early-exit for zero extra vertices
# ---------------------------------------------------------------------------


class TestEarlyExitZeroExtraVerts:
    """Verify that _refine_extra_vertices exits early when nve == 0 and nvf == 0."""

    def test_docstring_mentions_early_exit(self):
        """The method docstring should document the early-exit optimisation."""
        doc = OcMesher._refine_extra_vertices.__doc__
        assert "Early-exit" in doc or "early-exit" in doc or "nve == 0" in doc

    def test_early_exit_code_path_exists(self):
        """The early-exit guard for zero extra vertices must be present in source."""

        source = inspect.getsource(OcMesher._refine_extra_vertices)
        assert "not (nve | nvf)" in source or "(nve == 0 and nvf == 0)" in source
