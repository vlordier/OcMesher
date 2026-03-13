"""Tests for ocmesher/utils/interface.py — ctypes helpers."""

import ctypes
from unittest.mock import MagicMock

import numpy as np
import pytest

from ocmesher.utils.interface import (
    AC,
    AsBool,
    AsDouble,
    AsFloat,
    AsInt,
    load_cdll,
    register_func,
)


# ---------------------------------------------------------------------------
# AsInt
# ---------------------------------------------------------------------------
class TestAsInt:
    def test_basic_int32(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        ptr = AsInt(arr)
        assert ptr is not None

    def test_empty_array(self):
        arr = np.array([], dtype=np.int32)
        # ctypes pointer from empty array should still be obtainable
        ptr = AsInt(arr)
        assert ptr is not None

    def test_2d_array(self):
        arr = np.zeros((3, 4), dtype=np.int32)
        ptr = AsInt(arr)
        assert ptr is not None

    def test_non_contiguous_accepted(self):
        """ctypes data_as does not validate contiguity — users must ensure it."""
        arr = np.zeros((4, 4), dtype=np.int32)
        non_contig = arr[::2]
        assert not non_contig.flags["C_CONTIGUOUS"]
        # No error raised; callers must use AC() to ensure contiguity
        ptr = AsInt(non_contig)
        assert ptr is not None

    def test_single_element(self):
        arr = np.array([42], dtype=np.int32)
        ptr = AsInt(arr)
        assert ptr[0] == 42

    def test_large_array(self):
        """AsInt should handle large arrays without error."""
        arr = np.arange(100_000, dtype=np.int32)
        ptr = AsInt(arr)
        assert ptr[0] == 0
        assert ptr[99_999] == 99_999

    def test_negative_values(self):
        """AsInt should correctly handle negative int32 values."""
        arr = np.array([-1, -2147483648, 2147483647], dtype=np.int32)
        ptr = AsInt(arr)
        assert ptr[0] == -1
        assert ptr[1] == -2147483648
        assert ptr[2] == 2147483647

    def test_3d_array(self):
        """AsInt should work with higher-dimensional arrays."""
        arr = np.zeros((2, 3, 4), dtype=np.int32)
        ptr = AsInt(arr)
        assert ptr is not None

    def test_fortran_order_accepted(self):
        """Fortran-ordered arrays are accepted (though data layout may differ)."""
        arr = np.asfortranarray(np.zeros((3, 4), dtype=np.int32))
        assert arr.flags["F_CONTIGUOUS"]
        ptr = AsInt(arr)
        assert ptr is not None


# ---------------------------------------------------------------------------
# AsDouble
# ---------------------------------------------------------------------------
class TestAsDouble:
    def test_basic_float64(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        ptr = AsDouble(arr)
        assert ptr is not None

    def test_empty_array(self):
        arr = np.array([], dtype=np.float64)
        ptr = AsDouble(arr)
        assert ptr is not None

    def test_2d_array(self):
        arr = np.zeros((5, 3), dtype=np.float64)
        ptr = AsDouble(arr)
        assert ptr is not None

    def test_non_contiguous_accepted(self):
        """ctypes data_as does not validate contiguity — users must ensure it."""
        arr = np.zeros((4, 4), dtype=np.float64)
        non_contig = arr[::2]
        ptr = AsDouble(non_contig)
        assert ptr is not None

    def test_single_element(self):
        arr = np.array([3.14], dtype=np.float64)
        ptr = AsDouble(arr)
        assert abs(ptr[0] - 3.14) < 1e-10

    def test_nan_values(self):
        """AsDouble should handle NaN values without error."""
        arr = np.array([np.nan, 1.0, np.nan], dtype=np.float64)
        ptr = AsDouble(arr)
        assert np.isnan(ptr[0])
        assert ptr[1] == 1.0

    def test_inf_values(self):
        """AsDouble should handle infinite values."""
        arr = np.array([np.inf, -np.inf, 0.0], dtype=np.float64)
        ptr = AsDouble(arr)
        assert np.isinf(ptr[0])
        assert np.isinf(ptr[1])
        assert ptr[2] == 0.0

    def test_very_small_values(self):
        """AsDouble should preserve subnormal/very small float64 values."""
        arr = np.array([np.finfo(np.float64).tiny, np.finfo(np.float64).smallest_subnormal], dtype=np.float64)
        ptr = AsDouble(arr)
        assert ptr[0] == np.finfo(np.float64).tiny

    def test_fortran_order_accepted(self):
        """Fortran-ordered arrays are accepted."""
        arr = np.asfortranarray(np.zeros((3, 4), dtype=np.float64))
        ptr = AsDouble(arr)
        assert ptr is not None


# ---------------------------------------------------------------------------
# AsFloat
# ---------------------------------------------------------------------------
class TestAsFloat:
    def test_basic_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        ptr = AsFloat(arr)
        assert ptr is not None

    def test_empty_array(self):
        arr = np.array([], dtype=np.float32)
        ptr = AsFloat(arr)
        assert ptr is not None

    def test_non_contiguous_accepted(self):
        """ctypes data_as does not validate contiguity — users must ensure it."""
        arr = np.zeros((4, 4), dtype=np.float32)
        non_contig = arr[::2]
        ptr = AsFloat(non_contig)
        assert ptr is not None

    def test_single_element(self):
        arr = np.array([2.5], dtype=np.float32)
        ptr = AsFloat(arr)
        assert abs(ptr[0] - 2.5) < 1e-5

    def test_nan_and_inf(self):
        """AsFloat should handle NaN and Inf in float32."""
        arr = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
        ptr = AsFloat(arr)
        assert np.isnan(ptr[0])
        assert np.isinf(ptr[1])

    def test_2d_array(self):
        """AsFloat should work with multi-dimensional arrays."""
        arr = np.zeros((10, 3), dtype=np.float32)
        ptr = AsFloat(arr)
        assert ptr is not None

    def test_max_min_float32(self):
        """AsFloat should handle extreme float32 values."""
        arr = np.array([np.finfo(np.float32).max, np.finfo(np.float32).min], dtype=np.float32)
        ptr = AsFloat(arr)
        assert ptr[0] == np.finfo(np.float32).max


# ---------------------------------------------------------------------------
# AsBool
# ---------------------------------------------------------------------------
class TestAsBool:
    def test_basic_bool(self):
        arr = np.array([True, False, True], dtype=bool)
        ptr = AsBool(arr)
        assert ptr is not None

    def test_empty_array(self):
        arr = np.array([], dtype=bool)
        ptr = AsBool(arr)
        assert ptr is not None

    def test_non_contiguous_accepted(self):
        """ctypes data_as does not validate contiguity — users must ensure it."""
        arr = np.zeros((4, 4), dtype=bool)
        non_contig = arr[::2]
        ptr = AsBool(non_contig)
        assert ptr is not None

    def test_all_true(self):
        """AsBool with all-True array."""
        arr = np.ones(5, dtype=bool)
        ptr = AsBool(arr)
        assert ptr[0] is True

    def test_all_false(self):
        """AsBool with all-False array."""
        arr = np.zeros(5, dtype=bool)
        ptr = AsBool(arr)
        assert ptr[0] is False

    def test_single_element_true(self):
        arr = np.array([True], dtype=bool)
        ptr = AsBool(arr)
        assert ptr[0] is True


# ---------------------------------------------------------------------------
# AC (ascontiguousarray)
# ---------------------------------------------------------------------------
class TestAC:
    def test_contiguous_noop(self):
        arr = np.array([1, 2, 3])
        result = AC(arr)
        assert result.flags["C_CONTIGUOUS"]

    def test_makes_contiguous(self):
        arr = np.zeros((4, 4))
        non_contig = arr[::2]
        result = AC(non_contig)
        assert result.flags["C_CONTIGUOUS"]

    def test_preserves_dtype(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = AC(arr)
        assert result.dtype == np.float32

    def test_fortran_to_c_order(self):
        """AC should convert Fortran-order to C-contiguous."""
        arr = np.asfortranarray(np.zeros((3, 4), dtype=np.float64))
        assert arr.flags["F_CONTIGUOUS"]
        result = AC(arr)
        assert result.flags["C_CONTIGUOUS"]

    def test_preserves_data(self):
        """AC should preserve actual data values during conversion."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        non_contig = arr[::2]  # non-contiguous view
        result = AC(non_contig)
        np.testing.assert_array_equal(result, [[1, 2]])

    def test_empty_array(self):
        """AC on an empty array should return an empty contiguous array."""
        arr = np.array([], dtype=np.float32)
        result = AC(arr)
        assert result.flags["C_CONTIGUOUS"]
        assert len(result) == 0

    def test_already_contiguous_shares_memory(self):
        """AC on contiguous arrays should share memory (no copy)."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = AC(arr)
        assert np.shares_memory(arr, result)

    def test_non_contiguous_does_not_share_memory(self):
        """AC on non-contiguous arrays creates a copy."""
        arr = np.zeros((4, 4), dtype=np.float64)
        non_contig = arr[::2]
        result = AC(non_contig)
        assert not np.shares_memory(non_contig, result)


# ---------------------------------------------------------------------------
# register_func
# ---------------------------------------------------------------------------
class TestRegisterFunc:
    def test_registers_function(self):
        me = type("Obj", (), {})()
        dll = MagicMock()
        func_mock = MagicMock()
        dll.my_func = func_mock

        register_func(me, dll, "my_func", [ctypes.c_int], ctypes.c_int)

        assert hasattr(me, "my_func")
        assert func_mock.argtypes == [ctypes.c_int]
        assert func_mock.restype == ctypes.c_int

    def test_custom_caller_name(self):
        me = type("Obj", (), {})()
        dll = MagicMock()
        dll.cfunc = MagicMock()

        register_func(me, dll, "cfunc", caller_name="python_name")

        assert hasattr(me, "python_name")

    def test_default_argtypes_empty(self):
        me = type("Obj", (), {})()
        dll = MagicMock()
        func_mock = MagicMock()
        dll.noop = func_mock

        register_func(me, dll, "noop")

        assert func_mock.argtypes == []
        assert func_mock.restype is None

    def test_multiple_registrations_on_same_object(self):
        """Registering multiple functions on same object should all be accessible."""
        me = type("Obj", (), {})()
        dll = MagicMock()
        dll.func_a = MagicMock()
        dll.func_b = MagicMock()

        register_func(me, dll, "func_a", [ctypes.c_int], ctypes.c_int)
        register_func(me, dll, "func_b", [ctypes.c_double], ctypes.c_double)

        assert hasattr(me, "func_a")
        assert hasattr(me, "func_b")
        assert dll.func_a.argtypes == [ctypes.c_int]
        assert dll.func_b.argtypes == [ctypes.c_double]

    def test_overwrite_existing_attribute(self):
        """register_func should overwrite a pre-existing attribute."""
        me = type("Obj", (), {"my_func": "old_value"})()
        dll = MagicMock()
        dll.my_func = MagicMock()

        register_func(me, dll, "my_func")

        assert me.my_func is not "old_value"

    def test_complex_argtypes(self):
        """register_func with pointer and mixed argtypes."""
        me = type("Obj", (), {})()
        dll = MagicMock()
        dll.complex_func = MagicMock()

        argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
        ]
        register_func(me, dll, "complex_func", argtypes, ctypes.c_int32)

        assert dll.complex_func.argtypes == argtypes
        assert dll.complex_func.restype == ctypes.c_int32

    def test_caller_name_does_not_set_original_name(self):
        """When caller_name is given, the original name should NOT be set."""
        me = type("Obj", (), {})()
        dll = MagicMock()
        dll.c_name = MagicMock()

        register_func(me, dll, "c_name", caller_name="py_name")

        assert hasattr(me, "py_name")
        assert not hasattr(me, "c_name")


# ---------------------------------------------------------------------------
# load_cdll
# ---------------------------------------------------------------------------
class TestLoadCdll:
    def test_nonexistent_path_raises(self):
        with pytest.raises(OSError):
            load_cdll("/no/such/library.so")

    def test_invalid_file_raises(self):
        with pytest.raises(OSError):
            load_cdll(__file__)  # Python file, not a shared library

    def test_empty_path_raises(self):
        """Empty string path should raise an error."""
        with pytest.raises(OSError):
            load_cdll("")

    def test_directory_path_raises(self):
        """Passing a directory instead of a file should raise."""
        with pytest.raises(OSError):
            load_cdll("/tmp")
