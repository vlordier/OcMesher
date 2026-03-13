# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Data structures for OcMesher."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# 12 (3×4 inv_pose) + 9 (3×3 K) + 1 (H) + 1 (W) = 23 values per camera
CAMERA_DATA_STRIDE = 23
SDF_BATCH_SIZE = 10_000_000


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned bounding box defined by min/max along each axis."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        """Validate that each min is strictly less than its corresponding max."""
        for axis in ("x", "y", "z"):
            lo = getattr(self, f"{axis}_min")
            hi = getattr(self, f"{axis}_max")
            if lo >= hi:
                msg = f"{axis}_min ({lo}) must be strictly less than {axis}_max ({hi})"
                raise ValueError(msg)

    @classmethod
    def from_sequence(cls, seq: Sequence[float]) -> Bounds:
        """Create from flat [x_min, x_max, y_min, y_max, z_min, z_max]."""
        if len(seq) != 6:
            msg = f"Expected exactly 6 values, got {len(seq)}"
            raise ValueError(msg)
        return cls(
            x_min=float(seq[0]),
            x_max=float(seq[1]),
            y_min=float(seq[2]),
            y_max=float(seq[3]),
            z_min=float(seq[4]),
            z_max=float(seq[5]),
        )

    @property
    def center(self) -> NDArray[np.float64]:
        """Center point of the bounding box."""
        return np.array([
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        ])

    @property
    def max_extent(self) -> float:
        """Largest axis-aligned dimension."""
        return max(
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )

    @property
    def mins(self) -> NDArray[np.float64]:
        """Lower bounds as [x_min, y_min, z_min]."""
        return np.array([self.x_min, self.y_min, self.z_min])

    @property
    def maxs(self) -> NDArray[np.float64]:
        """Upper bounds as [x_max, y_max, z_max]."""
        return np.array([self.x_max, self.y_max, self.z_max])


@dataclass
class CameraSet:
    """Collection of cameras with poses, intrinsics, and image dimensions."""

    poses: list[NDArray[np.float64]]
    intrinsics: list[NDArray[np.float64]]
    heights: list[int]
    widths: list[int]

    def __post_init__(self) -> None:
        n = len(self.poses)
        if not (len(self.intrinsics) == len(self.heights) == len(self.widths) == n):
            msg = "All camera arrays must have the same length"
            raise ValueError(msg)
        for i, pose in enumerate(self.poses):
            arr = np.asarray(pose)
            if arr.shape != (4, 4):
                msg = f"poses[{i}] must be (4, 4), got {arr.shape}"
                raise ValueError(msg)
        for i, k in enumerate(self.intrinsics):
            arr = np.asarray(k)
            if arr.shape != (3, 3):
                msg = f"intrinsics[{i}] must be (3, 3), got {arr.shape}"
                raise ValueError(msg)

    @classmethod
    def from_tuple(
        cls,
        cameras: tuple[list[NDArray], list[NDArray], list[int], list[int]],
    ) -> CameraSet:
        """Create from legacy (poses, Ks, Hs, Ws) tuple."""
        poses, intrinsics, heights, widths = cameras
        return cls(
            poses=list(poses),
            intrinsics=list(intrinsics),
            heights=list(heights),
            widths=list(widths),
        )

    @property
    def count(self) -> int:
        """Number of cameras."""
        return len(self.poses)

    def pack(self, dtype: type = np.float64) -> NDArray:
        """Pack camera data into flat array with stride 23 for C++ interop.

        Layout per camera (23 values):
          [0:12]  - inverse pose matrix top 3 rows (3x4), row-major
          [12:21] - intrinsic matrix K (3x3), row-major
          [21]    - image height
          [22]    - image width
        """
        n = self.count
        poses_stack = np.stack(self.poses)
        inv_poses = np.linalg.inv(poses_stack)[:, :3, :4].reshape(n, -1)
        ks_flat = np.stack(self.intrinsics).reshape(n, -1)
        hs = np.asarray(self.heights, dtype=dtype).reshape(n, 1)
        ws = np.asarray(self.widths, dtype=dtype).reshape(n, 1)
        return np.concatenate(
            [inv_poses, ks_flat, hs, ws], axis=1,
        ).astype(dtype).ravel()


@dataclass
class MesherConfig:
    """Configuration parameters for OcMesher."""

    pixels_per_cube: int = 8
    inv_scale: float = 10.0
    min_dist: float = 1.0
    memory_limit_mb: int = 1000
    bisection_iters: int = 15
    enclosed: bool = True
    simplify_occluded: bool = True
    visible_relax_iter: int = 2
    coarse_count: int = 500_000
