"""Shared input-validation helpers for OcMesher backends.

Both :class:`~ocmesher.core.OcMesher` (C++ backend) and
:class:`~ocmesher.torch_core.TorchOcMesher` (PyTorch backend) use
identical camera / bounds / kernel validation logic.  Centralising it
here avoids duplication and ensures consistent error messages.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "validate_bounds",
    "validate_cameras",
    "validate_kernels",
]

_AXIS_NAMES = ("x", "y", "z")


def validate_cameras(cameras):
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
    return cam_poses, Ks, Hs, Ws


def validate_bounds(bounds):
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


def validate_kernels(kernels):
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
