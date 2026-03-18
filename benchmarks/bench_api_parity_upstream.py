#!/usr/bin/env python3
"""Extensive API + behavior parity benchmark against upstream OcMesher.

This benchmark compares the current checkout against an upstream Git ref
(default: ``main``) and writes a JSON report with:

1. API surface parity
   - module exports and availability
   - class/function signatures for selected public entry points
   - class public-method inventory
2. Behavioral parity on canonical scenes
   - mesh signatures (verts/faces/sums)
   - timing stats (best/avg/median)

The benchmark is intentionally resilient: if an API is missing it records a
structured failure instead of crashing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIT_BIN = shutil.which("git") or "git"
SH_BIN = "/bin/sh"


@dataclass
class CmdResult:
    code: int
    stdout: str
    stderr: str


def run_cmd(args: list[str], cwd: Path) -> CmdResult:
    proc = subprocess.run(  # noqa: S603
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return CmdResult(code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def parse_json_from_stdout(stdout: str) -> dict:
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def ensure_python_deps(repo_path: Path, python_exe: str, packages: list[str]) -> CmdResult:
    import_check = run_cmd(
        [python_exe, "-c", "import numpy, trimesh, gin"],
        cwd=repo_path,
    )
    if import_check.code == 0:
        return CmdResult(0, "dependencies already available", "")

    pip_attempt = run_cmd(
        [python_exe, "-m", "pip", "install", "--disable-pip-version-check", *packages],
        cwd=repo_path,
    )
    if pip_attempt.code == 0:
        return pip_attempt

    uv_bin = shutil.which("uv")
    if uv_bin:
        uv_attempt = run_cmd([uv_bin, "pip", "install", "--python", python_exe, *packages], cwd=repo_path)
        if uv_attempt.code == 0:
            return uv_attempt
        return CmdResult(
            uv_attempt.code,
            uv_attempt.stdout,
            f"pip failed: {pip_attempt.stderr.strip()}\nuv pip failed: {uv_attempt.stderr.strip()}",
        )

    return pip_attempt


def ensure_rust_extension(repo_path: Path, python_exe: str) -> CmdResult:
    import_check = run_cmd(
        [python_exe, "-c", "import ocmesher_rust"],
        cwd=repo_path,
    )
    if import_check.code == 0:
        return CmdResult(0, "ocmesher_rust already available", "")

    manifest = repo_path / "ocmesher-rust" / "crates" / "ocmesher-py" / "Cargo.toml"
    if not manifest.exists():
        return CmdResult(1, "", f"missing manifest: {manifest}")

    maturin = shutil.which("maturin")
    if maturin:
        return run_cmd([maturin, "develop", "--release", "--manifest-path", str(manifest)], cwd=repo_path)

    uv_bin = shutil.which("uv")
    if uv_bin:
        return run_cmd(
            [uv_bin, "run", "maturin", "develop", "--release", "--manifest-path", str(manifest)],
            cwd=repo_path,
        )

    return CmdResult(
        1,
        "",
        "neither maturin nor uv found in PATH; cannot build ocmesher_rust for parity benchmark",
    )


API_SNAPSHOT_SCRIPT = r"""
import inspect, json

payload = {
    "module": {},
    "symbols": {},
    "errors": [],
}

try:
    import ocmesher
except Exception as exc:
    payload["errors"].append(f"import ocmesher failed: {exc!r}")
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0)

payload["module"] = {
    "__all__": list(getattr(ocmesher, "__all__", [])),
    "public_names": sorted([n for n in dir(ocmesher) if not n.startswith("_")]),
}

symbol_names = ["OcMesher", "TorchOcMesher", "RustOcMesher", "make_rust_ocmesher"]
for name in symbol_names:
    entry = {
        "present": False,
        "kind": None,
        "signature": None,
        "init_signature": None,
        "call_signature": None,
        "public_methods": None,
        "error": None,
    }
    try:
        obj = getattr(ocmesher, name)
        entry["present"] = True
        if inspect.isclass(obj):
            entry["kind"] = "class"
            try:
                entry["init_signature"] = str(inspect.signature(obj.__init__))
            except Exception as sig_exc:
                entry["init_signature"] = f"<unavailable: {sig_exc!r}>"
            if hasattr(obj, "__call__"):
                try:
                    entry["call_signature"] = str(inspect.signature(obj.__call__))
                except Exception as sig_exc:
                    entry["call_signature"] = f"<unavailable: {sig_exc!r}>"
            methods = []
            for m in dir(obj):
                if m.startswith("_"):
                    continue
                attr = getattr(obj, m, None)
                if callable(attr):
                    methods.append(m)
            entry["public_methods"] = sorted(set(methods))
        elif callable(obj):
            entry["kind"] = "function"
            try:
                entry["signature"] = str(inspect.signature(obj))
            except Exception as sig_exc:
                entry["signature"] = f"<unavailable: {sig_exc!r}>"
        else:
            entry["kind"] = type(obj).__name__
    except Exception as exc:
        entry["error"] = repr(exc)
    payload["symbols"][name] = entry

print(json.dumps(payload, sort_keys=True))
"""


BEHAVIOR_SNAPSHOT_SCRIPT = r"""
import json, statistics, time
import numpy as np

payload = {
    "scenes": {},
    "errors": [],
}

cam_pose = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,3],[0,0,0,1]], dtype=np.float64)
K = np.array([[2000,0,640],[0,2000,360],[0,0,1]], dtype=np.float64)
cameras = ([cam_pose], [K], [720], [1280])
bounds = [-5.0, 5.0, -5.0, 5.0, -5.0, 5.0]

RUNS = 6
WARMUPS = 2

def summarize(times):
    ms = [t * 1000.0 for t in times]
    return {
        "best_ms": min(ms),
        "avg_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
    }

def mesh_signature(mesh, tag):
    return {
        "verts": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "verts_sum": float(mesh.vertices.sum()),
        "faces_sum": int(mesh.faces.sum()),
        "tag_true": int(np.count_nonzero(tag)),
    }

# Attempt upstream-compatible OcMesher callable benchmark.
try:
    from ocmesher import OcMesher

    def scene_sphere(xyz):
        return np.linalg.norm(xyz, axis=1) - 5.0

    def scene_plane(xyz):
        return xyz[:, 2]

    scenes = {
        "ocmesher_sphere": [scene_sphere],
        "ocmesher_plane": [scene_plane],
        "ocmesher_sphere_plane": [scene_sphere, scene_plane],
    }

    for name, sdfs in scenes.items():
        mesher = OcMesher(cameras, bounds, pixels_per_cube=32, coarse_count=100_000)
        meshes, tags = mesher(sdfs)
        for _ in range(WARMUPS):
            mesher(sdfs)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            mesher(sdfs)
            times.append(time.perf_counter() - t0)
        payload["scenes"][name] = {
            "ok": True,
            "timing": summarize(times),
            "signature": mesh_signature(meshes[0], tags[0]),
            "mesh_count": len(meshes),
        }
except Exception as exc:
    payload["errors"].append(f"OcMesher benchmark unavailable: {exc!r}")

# Rust wrapper benchmark if available in this checkout.
try:
    from ocmesher import make_rust_ocmesher

    def scene_sphere(xyz):
        return np.linalg.norm(xyz, axis=1) - 5.0

    def scene_plane(xyz):
        return xyz[:, 2]

    scenes = {
        "rust_wrapper_sphere": [scene_sphere],
        "rust_wrapper_plane": [scene_plane],
        "rust_wrapper_sphere_plane": [scene_sphere, scene_plane],
    }

    for name, sdfs in scenes.items():
        mesher = make_rust_ocmesher(cameras, bounds, pixels_per_cube=32, coarse_count=100_000)
        meshes, tags = mesher(sdfs)
        for _ in range(WARMUPS):
            mesher(sdfs)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            mesher(sdfs)
            times.append(time.perf_counter() - t0)
        payload["scenes"][name] = {
            "ok": True,
            "timing": summarize(times),
            "signature": mesh_signature(meshes[0], tags[0]),
            "mesh_count": len(meshes),
        }
except Exception as exc:
    payload["errors"].append(f"Rust wrapper benchmark unavailable: {exc!r}")

print(json.dumps(payload, sort_keys=True))
"""


def gather_repo_snapshot(repo_path: Path, python_exe: str) -> dict:
    api_res = run_cmd([python_exe, "-c", API_SNAPSHOT_SCRIPT], cwd=repo_path)
    behavior_res = run_cmd([python_exe, "-c", BEHAVIOR_SNAPSHOT_SCRIPT], cwd=repo_path)

    return {
        "api": parse_json_from_stdout(api_res.stdout) if api_res.code == 0 else {
            "errors": [f"api snapshot failed rc={api_res.code}: {api_res.stderr.strip()}"]
        },
        "behavior": parse_json_from_stdout(behavior_res.stdout) if behavior_res.code == 0 else {
            "errors": [f"behavior snapshot failed rc={behavior_res.code}: {behavior_res.stderr.strip()}"]
        },
    }


def compare_api(current: dict, upstream: dict) -> dict:
    csyms = current.get("symbols", {})
    usyms = upstream.get("symbols", {})

    keys = sorted(set(csyms.keys()) | set(usyms.keys()))
    symbol_cmp = {}
    exact_matches = 0

    for key in keys:
        c = csyms.get(key, {})
        u = usyms.get(key, {})
        present_match = c.get("present") == u.get("present")
        kind_match = c.get("kind") == u.get("kind")
        init_match = c.get("init_signature") == u.get("init_signature")
        call_match = c.get("call_signature") == u.get("call_signature")
        fn_match = c.get("signature") == u.get("signature")
        methods_match = c.get("public_methods") == u.get("public_methods")
        full_match = all([present_match, kind_match, init_match, call_match, fn_match, methods_match])
        if full_match:
            exact_matches += 1
        symbol_cmp[key] = {
            "full_match": full_match,
            "present_match": present_match,
            "kind_match": kind_match,
            "init_signature_match": init_match,
            "call_signature_match": call_match,
            "function_signature_match": fn_match,
            "public_methods_match": methods_match,
            "current": c,
            "upstream": u,
        }

    all_cmp = current.get("module", {}).get("__all__") == upstream.get("module", {}).get("__all__")
    return {
        "module___all___match": all_cmp,
        "symbols": symbol_cmp,
        "exact_symbol_matches": exact_matches,
        "total_symbols": len(keys),
    }


def compare_behavior(current: dict, upstream: dict, atol: float) -> dict:
    cscenes = current.get("scenes", {})
    uscenes = upstream.get("scenes", {})
    scene_keys = sorted(set(cscenes.keys()) & set(uscenes.keys()))
    scene_cmp = {}

    for scene in scene_keys:
        c = cscenes[scene]
        u = uscenes[scene]
        if not (c.get("ok") and u.get("ok")):
            scene_cmp[scene] = {
                "comparable": False,
                "reason": "scene missing or not ok in one side",
                "current": c,
                "upstream": u,
            }
            continue

        cs = c.get("signature", {})
        us = u.get("signature", {})
        sig_match = (
            cs.get("verts") == us.get("verts")
            and cs.get("faces") == us.get("faces")
            and cs.get("faces_sum") == us.get("faces_sum")
            and cs.get("tag_true") == us.get("tag_true")
            and abs(float(cs.get("verts_sum", 0.0)) - float(us.get("verts_sum", 0.0))) <= atol
        )

        ct = c.get("timing", {})
        ut = u.get("timing", {})
        speed_ratio_avg = None
        if ct.get("avg_ms") and ut.get("avg_ms"):
            speed_ratio_avg = float(ut["avg_ms"]) / float(ct["avg_ms"])

        scene_cmp[scene] = {
            "comparable": True,
            "signature_match": sig_match,
            "speed_ratio_upstream_over_current_avg": speed_ratio_avg,
            "current": c,
            "upstream": u,
        }

    return {
        "scenes": scene_cmp,
        "current_behavior_errors": current.get("errors", []),
        "upstream_behavior_errors": upstream.get("errors", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extensive API/runtime parity benchmark vs upstream OcMesher")
    parser.add_argument("--upstream-ref", default="main", help="Git ref/branch to compare against")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for verts_sum comparison")
    parser.add_argument("--json", default="", help="Optional path to write JSON report")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    required_pkgs = ["numpy", "trimesh", "gin-config"]
    bootstrap: dict[str, dict[str, Any] | None] = {
        "current": None,
        "upstream": None,
        "current_rust_extension": None,
    }

    # Build current checkout to ensure extension/core artifacts are present.
    run_cmd([SH_BIN, "install.sh"], cwd=repo_root)
    bootstrap_current = ensure_python_deps(repo_root, args.python, required_pkgs)
    bootstrap["current"] = {
        "returncode": bootstrap_current.code,
        "stderr": bootstrap_current.stderr.strip(),
    }
    rust_ext_current = ensure_rust_extension(repo_root, args.python)
    bootstrap["current_rust_extension"] = {
        "returncode": rust_ext_current.code,
        "stderr": rust_ext_current.stderr.strip(),
    }
    current_snapshot = gather_repo_snapshot(repo_root, args.python)

    # Snapshot upstream checkout in detached worktree.
    tmpdir = Path(tempfile.mkdtemp(prefix="ocmesher-api-parity-"))
    try:
        wt_add = run_cmd([GIT_BIN, "worktree", "add", "--detach", str(tmpdir), args.upstream_ref], cwd=repo_root)
        if wt_add.code != 0:
            raise RuntimeError(f"failed to add worktree: {wt_add.stderr.strip()}")

        run_cmd([SH_BIN, "install.sh"], cwd=tmpdir)
        bootstrap_upstream = ensure_python_deps(tmpdir, args.python, required_pkgs)
        bootstrap["upstream"] = {
            "returncode": bootstrap_upstream.code,
            "stderr": bootstrap_upstream.stderr.strip(),
        }
        upstream_snapshot = gather_repo_snapshot(tmpdir, args.python)
    finally:
        run_cmd([GIT_BIN, "worktree", "remove", str(tmpdir), "--force"], cwd=repo_root)
        shutil.rmtree(tmpdir, ignore_errors=True)

    report = {
        "meta": {
            "upstream_ref": args.upstream_ref,
            "python": args.python,
            "atol": args.atol,
            "dependency_bootstrap": bootstrap,
        },
        "api_comparison": compare_api(current_snapshot.get("api", {}), upstream_snapshot.get("api", {})),
        "behavior_comparison": compare_behavior(
            current_snapshot.get("behavior", {}),
            upstream_snapshot.get("behavior", {}),
            atol=args.atol,
        ),
        "current": current_snapshot,
        "upstream": upstream_snapshot,
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.json:
        Path(args.json).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
