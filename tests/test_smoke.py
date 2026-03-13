from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocmesher import OcMesher


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


def make_mesher():
    return OcMesher(
        make_cameras(),
        bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
        pixels_per_cube=6,
        inv_scale=10,
        min_dist=1,
        memory_limit_mb=256,
        bisection_iters=8,
        visible_relax_iter=1,
        coarse_count=2500,
        edge_fine_factor=2,
    )


def assert_valid_mesh_output(meshes, in_view_tags):
    assert len(meshes) == 1
    assert len(in_view_tags) == 1

    mesh = meshes[0]
    in_view_tag = in_view_tags[0]

    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    assert in_view_tag.shape[0] == mesh.vertices.shape[0]
    assert in_view_tag.dtype == np.bool_


def test_ocmesher_smoke_without_structure_mesh():
    mesher = make_mesher()
    meshes, in_view_tags = mesher([sphere_sdf])
    assert_valid_mesh_output(meshes, in_view_tags)


def test_ocmesher_smoke_with_structure_mesh():
    mesher = make_mesher()
    structure_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.75)
    meshes, in_view_tags = mesher([sphere_sdf], structure_mesh=structure_mesh)
    assert_valid_mesh_output(meshes, in_view_tags)
