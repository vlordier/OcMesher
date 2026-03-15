# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Demo script that meshes a Perlin noise terrain with a single camera view.

Outputs the resulting mesh to ``results/demo.obj``.
"""

from pathlib import Path

import numpy as np
import vnoise

from ocmesher import Bounds, CameraSet, OcMesher

noise = vnoise.Noise()


def f(XYZ):
    """Signed-distance function for a Perlin-noise height field."""
    scale = 2
    h = noise.noise2(XYZ[:, 0] / scale, XYZ[:, 1] / scale, grid_mode=False, octaves=4)

    return XYZ[:, 2] - h


def main():
    """Run the demo meshing pipeline."""
    cameras = CameraSet(
        poses=[
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, -1, 0, 3],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
        ],
        intrinsics=[
            np.array(
                [
                    [2000, 0, 640],
                    [0, 2000, 360],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        ],
        heights=[720],
        widths=[1280],
    )

    bounds = Bounds(x_min=-10, x_max=10, y_min=-10, y_max=10, z_min=-2, z_max=2)
    mesher = OcMesher(cameras, bounds, pixels_per_cube=16)
    meshes, _in_view_tags = mesher([f])
    Path("results").mkdir(parents=True, exist_ok=True)
    meshes[0].export("results/demo.obj")


if __name__ == "__main__":
    main()
