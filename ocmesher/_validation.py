"""Shared input-validation helpers for OcMesher backends.

Both :class:`~ocmesher.core.OcMesher` (C++ backend) and
:class:`~ocmesher.torch_core.TorchOcMesher` (PyTorch backend) use
identical camera / bounds / kernel validation logic.  Centralising it
here avoids duplication and ensures consistent error messages.
"""

from __future__ import annotations

import numpy as np

from ._types import BoundsLike, CamerasTuple, KernelSequence

__all__ = [
    "bounds_min_max",
    "coerce_kernel_sdf",
    "out_of_bounds_mask",
    "preprocess_cameras",
    "validate_bounds",
    "validate_cameras",
    "validate_kernels",
    "validate_mesher_params",
]

_AXIS_NAMES = ("x", "y", "z")


def validate_cameras(cameras: CamerasTuple) -> CamerasTuple:
    """Validate and normalise camera tuple, returning ``(cam_poses, Ks, Hs, Ws)``.

    Args:
        cameras: Tuple of ``(cam_poses, Ks, Hs, Ws)``.

    Returns:
        Tuple of numpy arrays ``(cam_poses, Ks, Hs, Ws)``.

    Raises:
        ValueError: If cameras structure, lengths, or shapes are invalid.
    """
    if not isinstance(cameras, (tuple, list)) or len(cameras) != 4:  # noqa: PLR2004
        msg = "cameras must be a tuple/list of (cam_poses, Ks, Hs, Ws)"
        raise ValueError(msg)
    cam_poses, Ks, Hs, Ws = cameras
    n = len(cam_poses)
    if len(Ks) != n or len(Hs) != n or len(Ws) != n:
        msg = f"Camera arrays must all have the same length, got poses={len(cam_poses)}, Ks={len(Ks)}, Hs={len(Hs)}, Ws={len(Ws)}"
        raise ValueError(msg)
    if n == 0:
        msg = "At least one camera is required"
        raise ValueError(msg)
    cam_poses = [np.asarray(p, dtype=np.float64) for p in cam_poses]
    Ks = [np.asarray(k, dtype=np.float64) for k in Ks]
    for i, pose in enumerate(cam_poses):
        if pose.shape != (4, 4):
            msg = f"cam_poses[{i}] must be a 4x4 matrix, got shape {pose.shape}"
            raise ValueError(msg)
    for i, k in enumerate(Ks):
        if k.shape != (3, 3):
            msg = f"Ks[{i}] must be a 3x3 matrix, got shape {k.shape}"
            raise ValueError(msg)
    for i, h in enumerate(Hs):
        if int(h) != h or h <= 0:
            msg = f"Hs[{i}] must be a positive integer, got {h!r}"
            raise ValueError(msg)
    for i, w in enumerate(Ws):
        if int(w) != w or w <= 0:
            msg = f"Ws[{i}] must be a positive integer, got {w!r}"
            raise ValueError(msg)
    return cam_poses, Ks, Hs, Ws


def validate_bounds(bounds: BoundsLike) -> np.ndarray:
    """Validate bounds array: 6 elements ``[x_min, x_max, y_min, y_max, z_min, z_max]``.

    Args:
        bounds: Sequence of 6 numeric values.

    Returns:
        numpy array of shape ``(6,)`` with dtype ``float64``.

    Raises:
        ValueError: If bounds has wrong length or min >= max for any axis.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (6,):
        msg = f"bounds must have 6 elements [x_min, x_max, y_min, y_max, z_min, z_max], got shape {bounds.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(bounds)):
        msg = "bounds must contain only finite values (no NaN or Inf)"
        raise ValueError(msg)
    for axis, name in enumerate(_AXIS_NAMES):
        if bounds[axis * 2] >= bounds[axis * 2 + 1]:
            msg = f"bounds {name}_min ({bounds[axis * 2]}) must be less than {name}_max ({bounds[axis * 2 + 1]})"
            raise ValueError(msg)
    return bounds


def bounds_min_max(bounds):
    """Split validated flat bounds into ``(3,)`` min/max vectors.

    Args:
        bounds: Validated bounds array of shape ``(6,)``.

    Returns:
        Tuple ``(b_min, b_max)`` of float64 numpy arrays.
    """
    b_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
    b_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
    return b_min, b_max


def validate_kernels(kernels: KernelSequence) -> None:
    """Validate that *kernels* is a non-empty sequence of callables.

    Args:
        kernels: Sequence of SDF kernel functions.

    Raises:
        ValueError: If kernels is empty or not a list/tuple.
        TypeError: If any element is not callable.
    """
    if not isinstance(kernels, (list, tuple)) or len(kernels) == 0:
        msg = "kernels must be a non-empty list/tuple of callable SDF functions"
        raise ValueError(msg)
    for i, k in enumerate(kernels):
        if not callable(k):
            msg = f"kernels[{i}] must be callable, got {type(k).__name__}"
            raise TypeError(msg)


def coerce_kernel_sdf(raw_sdf, n_points, label):
    """Convert a kernel result to ``np.ndarray`` and validate its shape.

    Args:
        raw_sdf: Kernel result object.
        n_points: Number of query points used for this kernel call.
        label: Human-readable kernel label included in error messages.

    Returns:
        ``(n_points,)`` numpy array.

    Raises:
        ValueError: If the kernel result shape is not ``(n_points,)``.
    """
    sdf = raw_sdf if isinstance(raw_sdf, np.ndarray) else np.asarray(raw_sdf)
    if sdf.shape != (n_points,):
        msg = f"{label} returned shape {sdf.shape} for {n_points} query points; expected ({n_points},)"
        raise ValueError(msg)
    return sdf


def preprocess_cameras(cam_poses, Ks, Hs, Ws):
    """Invert camera poses and normalise intrinsics as contiguous numpy arrays.

    Shared by both the C++ and PyTorch backends to avoid duplicated
    camera preprocessing logic.

    Args:
        cam_poses: List of ``(4, 4)`` float64 pose matrices (already validated).
        Ks: List of ``(3, 3)`` float64 intrinsics matrices (already validated).
        Hs: Sequence of image heights.
        Ws: Sequence of image widths.

    Returns:
        Tuple of ``(inv_poses_3x4, intrinsics, heights, widths)`` where:
        - *inv_poses_3x4*: ``(C, 3, 4)`` float64 contiguous array
        - *intrinsics*: ``(C, 3, 3)`` float64 contiguous array
        - *heights*: tuple of ints
        - *widths*: tuple of ints

    Raises:
        np.linalg.LinAlgError: If any camera pose is singular.
    """
    poses_arr = np.stack(cam_poses)  # (C, 4, 4) float64
    inv_full = np.linalg.inv(poses_arr)  # (C, 4, 4)
    inv_poses_3x4 = np.ascontiguousarray(inv_full[:, :3, :4])  # (C, 3, 4)
    intrinsics = np.ascontiguousarray(np.stack(Ks))  # (C, 3, 3) float64
    heights = tuple(int(h) for h in Hs)
    widths = tuple(int(w) for w in Ws)
    return inv_poses_3x4, intrinsics, heights, widths


def validate_mesher_params(
    *,
    pixels_per_cube,
    inv_scale,
    min_dist,
    memory_limit_mb,
    bisection_iters,
    visible_relax_iter,
    coarse_count,
    bisection_tol=None,
):
    """Validate common scalar configuration shared by both backends."""
    positive_values = {
        "pixels_per_cube": pixels_per_cube,
        "inv_scale": inv_scale,
        "min_dist": min_dist,
        "memory_limit_mb": memory_limit_mb,
        "bisection_iters": bisection_iters,
        "coarse_count": coarse_count,
    }
    for name, value in positive_values.items():
        if value <= 0:
            msg = f"{name} must be > 0, got {value!r}"
            raise ValueError(msg)
    if visible_relax_iter < 0:
        msg = f"visible_relax_iter must be >= 0, got {visible_relax_iter!r}"
        raise ValueError(msg)
    if bisection_tol is not None and bisection_tol < 0:
        msg = f"bisection_tol must be >= 0, got {bisection_tol!r}"
        raise ValueError(msg)


def out_of_bounds_mask(xyz, b_min, b_max):
    """Build a 1-D boolean mask indicating out-of-bounds points.

    Uses per-axis ufunc calls with ``out=`` to accumulate into two
    ``(N,)`` boolean buffers, avoiding the ``(N, 3)`` temporaries that
    ``np.any((XYZ <= lo) | (XYZ >= hi), axis=1)`` would allocate.

    Args:
        xyz: ``(N, 3)`` array of query positions.
        b_min: ``(3,)`` array of lower bounds.
        b_max: ``(3,)`` array of upper bounds.

    Returns:
        Boolean array of shape ``(N,)`` where ``True`` means the point
        is on or outside the boundary.
    """
    n = len(xyz)
    _le = np.less_equal
    _ge = np.greater_equal
    _lor = np.logical_or
    _tmp = np.empty(n, dtype=bool)
    mask = np.empty(n, dtype=bool)
    _le(xyz[:, 0], b_min[0], out=mask)
    _ge(xyz[:, 0], b_max[0], out=_tmp)
    _lor(mask, _tmp, out=mask)
    _le(xyz[:, 1], b_min[1], out=_tmp)
    _lor(mask, _tmp, out=mask)
    _ge(xyz[:, 1], b_max[1], out=_tmp)
    _lor(mask, _tmp, out=mask)
    _le(xyz[:, 2], b_min[2], out=_tmp)
    _lor(mask, _tmp, out=mask)
    _ge(xyz[:, 2], b_max[2], out=_tmp)
    _lor(mask, _tmp, out=mask)
    return mask
