"""Tests for ``ocmesher._validation`` (via core re-exports)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ocmesher._validation import (
    _AXIS_NAMES,
    bounds_min_max,
    coerce_kernel_sdf,
    preprocess_cameras,
    validate_mesher_params,
)
from ocmesher.core import _validate_bounds, _validate_cameras, _validate_kernels

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

    def test_rejects_non_positive_height(self, sample_camera_pose, sample_intrinsics):
        cameras = ([sample_camera_pose], [sample_intrinsics], [0], [1280])
        with pytest.raises(ValueError, match=r"Hs\[0\] must be a positive integer"):
            _validate_cameras(cameras)

    def test_rejects_non_positive_width(self, sample_camera_pose, sample_intrinsics):
        cameras = ([sample_camera_pose], [sample_intrinsics], [720], [-1])
        with pytest.raises(ValueError, match=r"Ws\[0\] must be a positive integer"):
            _validate_cameras(cameras)

    def test_rejects_non_integral_height(self, sample_camera_pose, sample_intrinsics):
        cameras = ([sample_camera_pose], [sample_intrinsics], [720.5], [1280])
        with pytest.raises(ValueError, match=r"Hs\[0\] must be a positive integer"):
            _validate_cameras(cameras)


class TestPreprocessCameras:
    """Tests for ``preprocess_cameras`` including error paths."""

    def test_returns_correct_shapes(self, sample_camera_pose, sample_intrinsics):
        inv, intr, hs, ws = preprocess_cameras([sample_camera_pose], [sample_intrinsics], [720], [1280])
        assert inv.shape == (1, 3, 4)
        assert intr.shape == (1, 3, 3)
        assert hs == (720,)
        assert ws == (1280,)

    def test_multi_camera_shapes(self, sample_camera_pose, sample_intrinsics):
        n = 3
        inv, intr, _hs, _ws = preprocess_cameras(
            [sample_camera_pose] * n,
            [sample_intrinsics] * n,
            [720] * n,
            [1280] * n,
        )
        assert inv.shape == (n, 3, 4)
        assert intr.shape == (n, 3, 3)

    def test_inv_poses_are_contiguous(self, sample_camera_pose, sample_intrinsics):
        inv, intr, _, _ = preprocess_cameras([sample_camera_pose], [sample_intrinsics], [720], [1280])
        assert inv.flags["C_CONTIGUOUS"]
        assert intr.flags["C_CONTIGUOUS"]

    def test_singular_pose_raises_linalgerror(self, sample_intrinsics):
        singular = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(np.linalg.LinAlgError):
            preprocess_cameras([singular], [sample_intrinsics], [720], [1280])


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


class TestBoundsMinMax:
    def test_splits_bounds_into_min_and_max_vectors(self):
        b_min, b_max = bounds_min_max(_validate_bounds([-1, 1, -2, 2, -3, 3]))
        np.testing.assert_array_equal(b_min, np.array([-1.0, -2.0, -3.0]))
        np.testing.assert_array_equal(b_max, np.array([1.0, 2.0, 3.0]))

    def test_returns_float64_arrays(self):
        b_min, b_max = bounds_min_max(_validate_bounds([-1, 1, -2, 2, -3, 3]))
        assert b_min.dtype == np.float64
        assert b_max.dtype == np.float64


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


class TestCoerceKernelSdf:
    def test_accepts_numpy_array(self):
        sdf = coerce_kernel_sdf(np.array([1.0, 2.0], dtype=np.float32), 2, "kernels[0]")
        np.testing.assert_array_equal(sdf, np.array([1.0, 2.0], dtype=np.float32))

    def test_converts_list_like(self):
        sdf = coerce_kernel_sdf([1.0, 2.0], 2, "kernels[0]")
        np.testing.assert_array_equal(sdf, np.array([1.0, 2.0]))

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"kernels\[1\].*expected \(2,\)"):
            coerce_kernel_sdf(np.zeros((2, 1)), 2, "kernels[1]")

    def test_preserves_dtype(self):
        arr = np.array([1.5, 2.5], dtype=np.float64)
        result = coerce_kernel_sdf(arr, 2, "test")
        assert result.dtype == np.float64

    def test_converts_scalar_to_array(self):
        # A list of 1 element should become (1,) array
        sdf = coerce_kernel_sdf([42.0], 1, "test")
        assert sdf.shape == (1,)

    def test_rejects_0d_scalar(self):
        with pytest.raises(ValueError, match="expected"):
            coerce_kernel_sdf(np.float32(1.0), 1, "test")

    def test_rejects_2d_matching_length(self):
        # (3, 1) has len 3 but shape != (3,)
        with pytest.raises(ValueError, match="expected"):
            coerce_kernel_sdf(np.zeros((3, 1)), 3, "test")


# ---------------------------------------------------------------------------
# validate_mesher_params edge-case tests
# ---------------------------------------------------------------------------

_VALID_PARAMS: dict[str, float | int] = {
    "pixels_per_cube": 1.0,
    "inv_scale": 1.0,
    "min_dist": 0.5,
    "memory_limit_mb": 256.0,
    "bisection_iters": 10,
    "visible_relax_iter": 0,
    "coarse_count": 100,
}


class TestValidateMesherParams:
    """Edge-case coverage for validate_mesher_params."""

    @pytest.mark.parametrize(
        "param",
        [
            "pixels_per_cube",
            "inv_scale",
            "min_dist",
            "memory_limit_mb",
            "bisection_iters",
            "coarse_count",
        ],
    )
    def test_zero_rejected_for_positive_params(self, param: str):
        with pytest.raises(ValueError, match=f"{param} must be > 0"):
            validate_mesher_params(**cast("Any", {**_VALID_PARAMS, param: 0}))

    @pytest.mark.parametrize(
        "param",
        [
            "pixels_per_cube",
            "inv_scale",
            "min_dist",
            "memory_limit_mb",
            "bisection_iters",
            "coarse_count",
        ],
    )
    def test_negative_rejected_for_positive_params(self, param: str):
        with pytest.raises(ValueError, match=f"{param} must be > 0"):
            validate_mesher_params(**cast("Any", {**_VALID_PARAMS, param: -1}))

    def test_negative_visible_relax_iter_rejected(self):
        with pytest.raises(ValueError, match="visible_relax_iter must be >= 0"):
            validate_mesher_params(**{**_VALID_PARAMS, "visible_relax_iter": -1})

    def test_negative_bisection_tol_rejected(self):
        with pytest.raises(ValueError, match="bisection_tol must be >= 0"):
            validate_mesher_params(**{**_VALID_PARAMS, "bisection_tol": -0.01})

    def test_none_bisection_tol_accepted(self):
        validate_mesher_params(**_VALID_PARAMS, bisection_tol=None)

    def test_zero_bisection_tol_accepted(self):
        validate_mesher_params(**_VALID_PARAMS, bisection_tol=0.0)

    def test_valid_params_pass(self):
        validate_mesher_params(**_VALID_PARAMS)


# ---------------------------------------------------------------------------
# Refactoring: module-level _AXIS_NAMES constant
# ---------------------------------------------------------------------------


class TestAxisNamesConstant:
    """Verify _AXIS_NAMES module constant is a tuple of the 3 axis labels."""

    def test_is_tuple(self):
        assert isinstance(_AXIS_NAMES, tuple)

    def test_contents(self):
        assert _AXIS_NAMES == ("x", "y", "z")

    def test_validation_uses_correct_names(self):
        """Bounds validation error messages reference the correct axis names."""
        with pytest.raises(ValueError, match="x_min"):
            _validate_bounds([5, 1, 0, 10, 0, 10])
        with pytest.raises(ValueError, match="y_min"):
            _validate_bounds([0, 10, 5, 1, 0, 10])
        with pytest.raises(ValueError, match="z_min"):
            _validate_bounds([0, 10, 0, 10, 5, 1])


# ---------------------------------------------------------------------------
# Direct _validation module public API: validate_cameras, validate_bounds,
# validate_kernels and out_of_bounds_mask
# (core.py has its own copies; these tests exercise the _validation.py code)
# ---------------------------------------------------------------------------

from ocmesher._validation import (
    out_of_bounds_mask,
    validate_bounds,
    validate_cameras,
    validate_kernels,
)


class TestValidationModuleValidateCameras:
    """Test validate_cameras from _validation.py (distinct from core._validate_cameras)."""

    def test_valid_single_camera(self, sample_cameras):
        cam_poses, _ks, _hs, _ws = validate_cameras(sample_cameras)
        assert len(cam_poses) == 1

    def test_rejects_wrong_type(self):
        with pytest.raises(ValueError, match="must be a tuple/list"):
            validate_cameras("not_a_tuple")  # type: ignore[arg-type]

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="must be a tuple/list"):
            validate_cameras(([], []))  # type: ignore[arg-type]

    def test_rejects_mismatched_lengths(self, sample_camera_pose, sample_intrinsics):
        cameras = ([sample_camera_pose], [sample_intrinsics, sample_intrinsics], [720], [1280])
        with pytest.raises(ValueError, match="same length"):
            validate_cameras(cameras)  # type: ignore[arg-type]

    def test_rejects_empty_cameras(self):
        with pytest.raises(ValueError, match="At least one camera"):
            validate_cameras(([], [], [], []))  # type: ignore[arg-type]

    def test_rejects_bad_pose_shape(self, sample_intrinsics):
        bad_pose = np.eye(3)
        cameras = ([bad_pose], [sample_intrinsics], [720], [1280])
        with pytest.raises(ValueError, match="4x4 matrix"):
            validate_cameras(cameras)  # type: ignore[arg-type]

    def test_rejects_bad_intrinsics_shape(self, sample_camera_pose):
        bad_k = np.eye(4)
        cameras = ([sample_camera_pose], [bad_k], [720], [1280])
        with pytest.raises(ValueError, match="3x3 matrix"):
            validate_cameras(cameras)  # type: ignore[arg-type]

    def test_rejects_non_positive_height(self, sample_camera_pose, sample_intrinsics):
        cameras = ([sample_camera_pose], [sample_intrinsics], [0], [1280])
        with pytest.raises(ValueError, match=r"Hs\[0\]"):
            validate_cameras(cameras)  # type: ignore[arg-type]

    def test_rejects_non_positive_width(self, sample_camera_pose, sample_intrinsics):
        cameras = ([sample_camera_pose], [sample_intrinsics], [720], [-1])
        with pytest.raises(ValueError, match=r"Ws\[0\]"):
            validate_cameras(cameras)  # type: ignore[arg-type]

    def test_rejects_non_integral_height(self, sample_camera_pose, sample_intrinsics):
        cameras = ([sample_camera_pose], [sample_intrinsics], [720.5], [1280])
        with pytest.raises(ValueError, match=r"Hs\[0\]"):
            validate_cameras(cameras)  # type: ignore[arg-type]


class TestValidationModuleValidateBounds:
    """Test validate_bounds from _validation.py (distinct from core._validate_bounds)."""

    def test_valid_bounds(self, sample_bounds):
        result = validate_bounds(sample_bounds)
        assert result.shape == (6,)
        assert result.dtype == np.float64

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="6 elements"):
            validate_bounds([0, 1, 2])

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            validate_bounds([np.nan, 1.0, -1.0, 1.0, -1.0, 1.0])

    def test_rejects_min_geq_max_x(self):
        with pytest.raises(ValueError, match="x_min"):
            validate_bounds([5, -5, -1, 1, -1, 1])

    def test_rejects_min_geq_max_y(self):
        with pytest.raises(ValueError, match="y_min"):
            validate_bounds([-1, 1, 5, -5, -1, 1])

    def test_rejects_min_geq_max_z(self):
        with pytest.raises(ValueError, match="z_min"):
            validate_bounds([-1, 1, -1, 1, 5, -5])


class TestValidationModuleValidateKernels:
    """Test validate_kernels from _validation.py (distinct from core._validate_kernels)."""

    def test_accepts_single_callable(self, sphere_kernel):
        validate_kernels([sphere_kernel])  # no exception

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_kernels([])

    def test_rejects_non_sequence(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_kernels(42)  # type: ignore[arg-type]

    def test_rejects_non_callable_element(self, sphere_kernel):
        with pytest.raises(TypeError, match=r"kernels\[1\] must be callable"):
            validate_kernels([sphere_kernel, "bad"])  # type: ignore[list-item]


class TestOutOfBoundsMask:
    """Test out_of_bounds_mask from _validation.py."""

    def _make_bounds(self):
        b_min = np.array([-1.0, -1.0, -1.0])
        b_max = np.array([1.0, 1.0, 1.0])
        return b_min, b_max

    def test_all_inside_returns_false(self):
        b_min, b_max = self._make_bounds()
        pts = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.3]])
        mask = out_of_bounds_mask(pts, b_min, b_max)
        assert mask.shape == (2,)
        assert not np.any(mask)

    def test_all_outside_returns_true(self):
        b_min, b_max = self._make_bounds()
        pts = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        mask = out_of_bounds_mask(pts, b_min, b_max)
        assert np.all(mask)

    def test_on_boundary_returns_true(self):
        """Points exactly on the boundary are out-of-bounds."""
        b_min, b_max = self._make_bounds()
        pts = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        mask = out_of_bounds_mask(pts, b_min, b_max)
        assert np.all(mask)

    def test_mixed_points(self):
        b_min, b_max = self._make_bounds()
        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # inside
                [2.0, 0.0, 0.0],  # outside x
                [0.0, 2.0, 0.0],  # outside y
                [0.0, 0.0, -2.0],  # outside z
            ]
        )
        mask = out_of_bounds_mask(pts, b_min, b_max)
        np.testing.assert_array_equal(mask, [False, True, True, True])

    def test_single_point_inside(self):
        b_min, b_max = self._make_bounds()
        pts = np.array([[0.0, 0.0, 0.0]])
        mask = out_of_bounds_mask(pts, b_min, b_max)
        assert mask.shape == (1,)
        assert not mask[0]

    def test_returns_bool_dtype(self):
        b_min, b_max = self._make_bounds()
        pts = np.array([[0.0, 0.0, 0.0]])
        mask = out_of_bounds_mask(pts, b_min, b_max)
        assert mask.dtype == bool
