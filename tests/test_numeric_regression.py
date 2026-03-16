"""Numerical regression tests for deterministic OcMesher outputs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from benchmarks.upstream_parity import run_upstream_parity
from ocmesher import OcMesher

EXPECTED_SIGNATURE = {
    "verts": 1371,
    "faces": 2738,
    "verts_sum": 1384.006159440049,
    "faces_sum": 5494984,
}


def _signature() -> dict[str, int | float]:
    cam_poses = [
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    ]
    intrinsics = [
        np.array(
            [
                [2000.0, 0.0, 640.0],
                [0.0, 2000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    ]
    heights = [720]
    widths = [1280]

    mesher = OcMesher((cam_poses, intrinsics, heights, widths), bounds=[-5, 5, -5, 5, -2, 2], pixels_per_cube=32, coarse_count=100_000)
    def sdf(xyz: np.ndarray) -> np.ndarray:
        return np.linalg.norm(xyz, axis=1) - 5.0

    meshes, _ = mesher([sdf])
    mesh = meshes[0]
    return {
        "verts": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "verts_sum": float(mesh.vertices.sum()),
        "faces_sum": int(mesh.faces.sum()),
    }


@pytest.mark.integration
def test_numeric_signature_matches_known_reference() -> None:
    try:
        subprocess.run(
            ["/bin/sh", "install.sh"],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = _signature()
    except FileNotFoundError as exc:
        pytest.skip(f"Compiled core library not available: {exc}")

    assert actual["verts"] == EXPECTED_SIGNATURE["verts"]
    assert actual["faces"] == EXPECTED_SIGNATURE["faces"]
    assert actual["faces_sum"] == EXPECTED_SIGNATURE["faces_sum"]
    assert abs(float(actual["verts_sum"]) - float(EXPECTED_SIGNATURE["verts_sum"])) <= 1e-12


@pytest.mark.integration
def test_numeric_signature_matches_main_branch() -> None:
    parity = run_upstream_parity(
        Path(__file__).resolve().parents[1],
        upstream_ref="main",
        python_exe="/Users/vincent/Work/OcMesher/.venv/bin/python",
    )

    assert bool(parity["delta"]["matches"])
