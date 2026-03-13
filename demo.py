# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Demo script that meshes a Perlin noise terrain with a single camera view.

Outputs the resulting mesh to ``results/demo.obj``.
"""

import argparse
import os

import numpy as np
import vnoise

from ocmesher import OcMesher

# Default camera parameters
DEFAULT_FX = 2000.0
DEFAULT_FY = 2000.0
DEFAULT_CX = 640.0
DEFAULT_CY = 360.0
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720
DEFAULT_CAMERA_Z = 3.0

# Default noise parameters
DEFAULT_NOISE_SCALE = 2.0
DEFAULT_NOISE_OCTAVES = 4

# Default mesher parameters
DEFAULT_PIXELS_PER_CUBE = 16

# Default output parameters
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_OUTPUT_FILENAME = "demo.obj"


def parse_args():
    parser = argparse.ArgumentParser(description="OcMesher demo: mesh a noise terrain from a single camera view")
    # Camera intrinsics
    parser.add_argument("--fx", type=float, default=DEFAULT_FX, help="Camera focal length x")
    parser.add_argument("--fy", type=float, default=DEFAULT_FY, help="Camera focal length y")
    parser.add_argument("--cx", type=float, default=DEFAULT_CX, help="Camera principal point x")
    parser.add_argument("--cy", type=float, default=DEFAULT_CY, help="Camera principal point y")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT, help="Image height in pixels")
    parser.add_argument("--camera-z", type=float, default=DEFAULT_CAMERA_Z, help="Camera height (z position)")
    # Noise parameters
    parser.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE, help="Perlin noise spatial scale")
    parser.add_argument("--noise-octaves", type=int, default=DEFAULT_NOISE_OCTAVES, help="Perlin noise octaves")
    # Mesher parameters
    parser.add_argument("--pixels-per-cube", type=int, default=DEFAULT_PIXELS_PER_CUBE, help="Pixels per octree cube")
    # Output parameters
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--output-filename", type=str, default=DEFAULT_OUTPUT_FILENAME, help="Output mesh filename")
    return parser.parse_args()


def make_sdf_func(noise_scale, noise_octaves):
    noise = vnoise.Noise()

    def f(XYZ):
        h = noise.noise2(
            XYZ[:, 0] / noise_scale,
            XYZ[:, 1] / noise_scale,
            grid_mode=False,
            octaves=noise_octaves,
        )
        return XYZ[:, 2] - h

    return f


def main():
    args = parse_args()

    cam_poses = [np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, args.camera_z],
        [0, 0, 0, 1],
    ])]
    Ks = [np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1],
    ])]
    Hs = [args.height]
    Ws = [args.width]

    f = make_sdf_func(args.noise_scale, args.noise_octaves)

    mesher = OcMesher((cam_poses, Ks, Hs, Ws), pixels_per_cube=args.pixels_per_cube)
    meshes, in_view_tags = mesher([f])

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    meshes[0].export(output_path)
    print(f"Exported mesh to {output_path}")


if __name__ == "__main__":
    main()
