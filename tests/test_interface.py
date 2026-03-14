"""Tests for ``ocmesher.utils.interface``."""

from __future__ import annotations

import sys
from ctypes import c_double, c_float, c_int32
from unittest.mock import MagicMock

import pytest

from ocmesher.utils.interface import (
    AsBool,
    AsDouble,
    AsFloat,
    AsInt,
    load_cdll,
    register_func,
)


class TestAsInt:
    def test_valid_int32_array(self, sample_numpy_int_array):
        result = AsInt(sample_numpy_int_array)
        assert result is not None
        assert result[0] == sample_numpy_int_array[0]

    def test_rejects_non_array(self):
        with pytest.raises(TypeError, match="Expected a numpy array"):
            AsInt([1, 2, 3])

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="Expected a numpy array"):
            AsInt("hello")


class TestAsDouble:
    def test_valid_float64_array(self, sample_numpy_float_array):
        result = AsDouble(sample_numpy_float_array)
        assert result is not None
        assert result[0] == sample_numpy_float_array[0]

    def test_rejects_non_array(self):
        with pytest.raises(TypeError, match="Expected a numpy array"):
            AsDouble(42.0)


class TestAsFloat:
    def test_valid_float32_array(self, sample_numpy_float32_array):
        result = AsFloat(sample_numpy_float32_array)
        assert result is not None
        assert abs(result[0] - sample_numpy_float32_array[0]) < 1e-6

    def test_rejects_non_array(self):
        with pytest.raises(TypeError, match="Expected a numpy array"):
            AsFloat(3.14)


class TestAsBool:
    def test_valid_bool_array(self, sample_numpy_bool_array):
        result = AsBool(sample_numpy_bool_array)
        assert result is not None
        assert result[0] == sample_numpy_bool_array[0]

    def test_rejects_non_array(self):
        with pytest.raises(TypeError, match="Expected a numpy array"):
            AsBool(True)  # noqa: FBT003


class TestRegisterFunc:
    def test_registers_function_on_object(self):
        obj = MagicMock()
        dll = MagicMock()
        dll.my_func = MagicMock()

        register_func(obj, dll, "my_func", argtypes=[c_int32], restype=c_double)
        assert hasattr(obj, "my_func")

    def test_registers_with_custom_caller_name(self):
        obj = type("Obj", (), {})()
        dll = MagicMock()
        dll.original_name = MagicMock()

        register_func(obj, dll, "original_name", caller_name="alias")
        assert hasattr(obj, "alias")
        assert obj.alias is dll.original_name

    def test_sets_argtypes_and_restype(self):
        obj = type("Obj", (), {})()
        dll = MagicMock()
        mock_func = MagicMock()
        dll.test_func = mock_func

        register_func(obj, dll, "test_func", argtypes=[c_int32, c_float], restype=c_double)
        assert mock_func.argtypes == [c_int32, c_float]
        assert mock_func.restype is c_double

    def test_raises_on_missing_function(self):
        obj = MagicMock()
        dll = MagicMock(spec=[])

        with pytest.raises(AttributeError, match="not found in shared library"):
            register_func(obj, dll, "nonexistent_func")

    def test_default_argtypes_is_empty_list(self):
        obj = type("Obj", (), {})()
        dll = MagicMock()
        mock_func = MagicMock()
        dll.test_func = mock_func

        register_func(obj, dll, "test_func")
        assert mock_func.argtypes == []
        assert mock_func.restype is None


class TestLoadCdll:
    def test_raises_file_not_found_for_missing_lib(self, tmp_path):
        sys.path.append(str(tmp_path))
        try:
            with pytest.raises(FileNotFoundError, match="Shared library not found"):
                load_cdll("nonexistent/lib.so")
        finally:
            sys.path.remove(str(tmp_path))

    def test_raises_on_invalid_shared_library(self, tmp_path):
        fake_lib = tmp_path / "fake.so"
        fake_lib.write_text("not a real library")

        sys.path.append(str(tmp_path))
        try:
            with pytest.raises(OSError, match="Failed to load shared library"):
                load_cdll("fake.so")
        finally:
            sys.path.remove(str(tmp_path))
