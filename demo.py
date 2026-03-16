# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Demo script that meshes a terrain SDF with a single camera view.

Outputs the resulting mesh to ``results/demo.obj``.
"""

from pathlib import Path

import numpy as np

from ocmesher import OcMesher

try:
    import vnoise

    _noise = vnoise.Noise()

    def _terrain_height(XYZ):
        scale = 2
        return _noise.noise2(XYZ[:, 0] / scale, XYZ[:, 1] / scale, grid_mode=False, octaves=4)

except ImportError:

    def _terrain_height(XYZ):
        x = XYZ[:, 0] / 2.0
        y = XYZ[:, 1] / 2.0
        return (
            0.5000 * np.sin(1.0 * x) * np.cos(1.0 * y)
            + 0.2500 * np.sin(2.0 * x) * np.cos(2.0 * y)
            + 0.1250 * np.sin(4.0 * x) * np.cos(4.0 * y)
            + 0.0625 * np.sin(8.0 * x) * np.cos(8.0 * y)
        )


def f(XYZ):
    """Signed-distance function for the demo terrain height field."""
    h = _terrain_height(XYZ)
    return XYZ[:, 2] - h


def main():
    """Run the demo meshing pipeline."""
    cam_poses = [
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 3],
                [0, 0, 0, 1],
            ]
        )
    ]
    Ks = [
        np.array(
            [
                [2000, 0, 640],
                [0, 2000, 360],
                [0, 0, 1],
            ]
        )
    ]
    Hs = [720]
    Ws = [1280]

    bounds = (-10, 10, -10, 10, -2, 2)
    mesher = OcMesher((cam_poses, Ks, Hs, Ws), bounds, pixels_per_cube=16)
    meshes, _in_view_tags = mesher([f])
    Path("results").mkdir(parents=True, exist_ok=True)
    meshes[0].export("results/demo.obj")


if __name__ == "__main__":
    main()
