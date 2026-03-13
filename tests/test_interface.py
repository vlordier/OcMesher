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
