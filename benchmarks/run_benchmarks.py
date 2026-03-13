import argparse
import json
import platform
import sys
import time
from pathlib import Path

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


def positive_int(value: str) -> int:
    """Argparse type that ensures a positive integer (>= 1)."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
    if ivalue < 1:
        raise argparse.ArgumentTypeError("repeats must be an integer >= 1")
    return ivalue


def make_mesher(edge_fine_factor):
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
        edge_fine_factor=edge_fine_factor,
    )


def run_case(name, repeats, structure_mesh=None, edge_fine_factor=2):
    runtimes = []
    mesh_stats = None

    for _ in range(repeats):
        mesher = make_mesher(edge_fine_factor=edge_fine_factor)
        start = time.perf_counter()
        meshes, in_view_tags = mesher([sphere_sdf], structure_mesh=structure_mesh)
        elapsed = time.perf_counter() - start
        runtimes.append(elapsed)

    mesh = meshes[0]
    mesh_stats = {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "in_view_vertices": int(np.count_nonzero(in_view_tags[0])),
    }

    return {
        "name": name,
        "repeats": repeats,
        "min_seconds": min(runtimes),
        "max_seconds": max(runtimes),
        "mean_seconds": sum(runtimes) / len(runtimes),
        "mesh": mesh_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Run OcMesher smoke benchmarks.")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/latest.json"))
    args = parser.parse_args()

    cases = [run_case("sphere_smoke", repeats=args.repeats)]
    structure_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.75)
    cases.append(
        run_case(
            "sphere_structure_guided",
            repeats=args.repeats,
            structure_mesh=structure_mesh,
            edge_fine_factor=2,
        )
    )

    results = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
