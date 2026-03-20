"""Tests for ``tests.sdf_fixtures`` — reusable SDF kernel factories."""

from __future__ import annotations

import numpy as np

from tests.sdf_fixtures import constant_kernel


class TestConstantKernel:
    def test_returns_scalar_for_single_point(self):
        kernel = constant_kernel(0.0)
        result = kernel(np.zeros((1, 3), dtype=np.float64))
        assert result.shape == (1,)
        assert result[0] == 0.0

    def test_returns_correct_length_array(self):
        kernel = constant_kernel(42.0)
        pts = np.zeros((100, 3), dtype=np.float64)
        result = kernel(pts)
        assert result.shape == (100,)
        assert all(v == 42.0 for v in result)

    def test_returns_float32_dtype(self):
        kernel = constant_kernel(1.5)
        pts = np.zeros((10, 3), dtype=np.float64)
        result = kernel(pts)
        assert result.dtype == np.float32

    def test_negative_value(self):
        kernel = constant_kernel(-99.0)
        pts = np.zeros((5, 3), dtype=np.float64)
        result = kernel(pts)
        assert all(v == -99.0 for v in result)

    def test_zero_value(self):
        kernel = constant_kernel(0.0)
        pts = np.zeros((7, 3), dtype=np.float64)
        result = kernel(pts)
        assert all(v == 0.0 for v in result)
