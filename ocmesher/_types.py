"""Shared type aliases for OcMesher.

Provides lightweight type aliases that document the expected shapes and
semantics of common arguments without introducing heavyweight runtime
abstractions.  Import these in type-checking context (``TYPE_CHECKING``)
or at module level for documentation purposes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BoundsLike",
    "CamerasTuple",
    "KernelSequence",
    "MeshResult",
    "SDFKernel",
]

# An SDF kernel is a callable that accepts an (N, 3) float64 array of query
# positions and returns an (N,) float array of signed-distance values.
SDFKernel = Callable[[NDArray[np.float64]], NDArray[Any]]

# A validated sequence of SDF kernel callables.
KernelSequence = Sequence[SDFKernel]

# Camera tuple: (cam_poses, intrinsics, heights, widths).
# Each list has *C* elements (one per camera); poses are 4x4, Ks are 3x3.
CamerasTuple = tuple[
    Sequence[NDArray[np.float64]],  # cam_poses: list of (4, 4)
    Sequence[NDArray[np.float64]],  # Ks: list of (3, 3)
    Sequence[int],  # Hs: image heights
    Sequence[int],  # Ws: image widths
]

# Bounds: a 6-element sequence [x_min, x_max, y_min, y_max, z_min, z_max].
BoundsLike = Sequence[float] | NDArray[np.float64]

# Public return type of the coarse-to-fine meshing pipeline.
MeshResult = tuple[list[Any], list[NDArray[np.bool_]]]
