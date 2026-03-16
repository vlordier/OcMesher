"""Benchmark helpers for numerical parity checks against upstream branches."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict


class MeshSignature(TypedDict):
    """Compact numeric signature for a generated mesh."""

    verts: int
    faces: int
    verts_sum: float
    faces_sum: int


_GIT_BIN = shutil.which("git") or "git"
_SH_BIN = "/bin/sh"


def compare_signatures(lhs: MeshSignature, rhs: MeshSignature, *, atol: float = 1e-12) -> dict[str, float | int | bool]:
    """Compare two mesh signatures and return per-field differences."""
    verts_delta = lhs["verts"] - rhs["verts"]
    faces_delta = lhs["faces"] - rhs["faces"]
    verts_sum_delta = lhs["verts_sum"] - rhs["verts_sum"]
    faces_sum_delta = lhs["faces_sum"] - rhs["faces_sum"]
    matches = (
        verts_delta == 0
        and faces_delta == 0
        and abs(verts_sum_delta) <= atol
        and faces_sum_delta == 0
    )
    return {
        "matches": matches,
        "verts_delta": verts_delta,
        "faces_delta": faces_delta,
        "verts_sum_delta": verts_sum_delta,
        "faces_sum_delta": faces_sum_delta,
    }


def _signature_python_script(*, pixels_per_cube: int, coarse_count: int) -> str:
    """Return a Python one-liner body that computes a deterministic mesh signature."""
    return (
        "import json, numpy as np; "
        "from ocmesher import OcMesher; "
        "cam_poses=[np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,3],[0,0,0,1]],dtype=np.float64)]; "
        "ks=[np.array([[2000,0,640],[0,2000,360],[0,0,1]],dtype=np.float64)]; "
        "hs=[720]; ws=[1280]; "
        f"mesher=OcMesher((cam_poses,ks,hs,ws), bounds=[-5,5,-5,5,-2,2], pixels_per_cube={pixels_per_cube}, coarse_count={coarse_count}); "
        "sdf=lambda xyz: np.linalg.norm(xyz, axis=1) - 5.0; "
        "meshes,_=mesher([sdf]); mesh=meshes[0]; "
        "print(json.dumps({'verts': int(mesh.vertices.shape[0]), 'faces': int(mesh.faces.shape[0]), 'verts_sum': float(mesh.vertices.sum()), 'faces_sum': int(mesh.faces.sum())}, sort_keys=True))"
    )


def _compute_signature_in_repo(repo_path: Path, *, python_exe: str, pixels_per_cube: int, coarse_count: int) -> MeshSignature:
    """Compute signature for one repository checkout."""
    cmd = [
        python_exe,
        "-c",
        _signature_python_script(pixels_per_cube=pixels_per_cube, coarse_count=coarse_count),
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_path,
    )
    line = proc.stdout.strip().splitlines()[-1]
    parsed = json.loads(line)
    return {
        "verts": int(parsed["verts"]),
        "faces": int(parsed["faces"]),
        "verts_sum": float(parsed["verts_sum"]),
        "faces_sum": int(parsed["faces_sum"]),
    }


def run_upstream_parity(
    repo_root: Path,
    *,
    upstream_ref: str = "main",
    python_exe: str,
    pixels_per_cube: int = 32,
    coarse_count: int = 100_000,
) -> dict[str, MeshSignature | dict[str, float | int | bool]]:
    """Compare current checkout numerical signature against an upstream Git ref."""
    repo_root = repo_root.resolve()
    subprocess.run(  # noqa: S603
        [_SH_BIN, "install.sh"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    current = _compute_signature_in_repo(
        repo_root,
        python_exe=python_exe,
        pixels_per_cube=pixels_per_cube,
        coarse_count=coarse_count,
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="ocmesher-upstream-"))
    try:
        subprocess.run(  # noqa: S603
            [_GIT_BIN, "worktree", "add", "--detach", str(tmpdir), upstream_ref],
            check=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        subprocess.run(  # noqa: S603
            [_SH_BIN, "install.sh"],
            check=True,
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        upstream = _compute_signature_in_repo(
            tmpdir,
            python_exe=python_exe,
            pixels_per_cube=pixels_per_cube,
            coarse_count=coarse_count,
        )
    finally:
        subprocess.run(  # noqa: S603
            [_GIT_BIN, "worktree", "remove", str(tmpdir), "--force"],
            check=False,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "current": current,
        "upstream": upstream,
        "delta": compare_signatures(current, upstream),
    }
