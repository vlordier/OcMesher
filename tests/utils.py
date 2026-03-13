"""Shared test/benchmark utilities for OcMesher."""

import numpy as np


def make_cameras():
    cam_poses = [
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    ]
    intrinsics = [
        np.array(
            [
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 180.0],
                [0.0, 0.0, 1.0],
            ]
        )
    ]
    heights = [360]
    widths = [640]
    return cam_poses, intrinsics, heights, widths


def sphere_sdf(xyz):
    return np.linalg.norm(xyz, axis=1) - 0.75
