from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_one(
    repo_root: Path,
    backend: str,
    device: str,
    sdf: str,
    runs: int,
    warmup: int,
    rust_sdf_batch_size: int | None,
) -> dict[str, object]:
    output_path = repo_root / "benchmarks" / "results" / f"isolated_{backend}_{device}_{sdf}.json"
    cmd = [
        sys.executable,
        "-c",
        (
            "import json; "
            "from benchmarks.run_benchmark import _make_bounds, _make_cameras, _bench_original, _bench_torch, _bench_rust; "
            f"cameras=_make_cameras(1); bounds=_make_bounds(); "
            f"backend={backend!r}; device={device!r}; sdf={sdf!r}; runs={runs}; warmup={warmup}; rust_sdf_batch_size={rust_sdf_batch_size!r}; "
            "fn = _bench_original if backend == 'original' else (_bench_torch if backend == 'torch' else _bench_rust); "
            "kwargs = {'n_runs': runs, 'warmup': warmup}; "
            "kwargs.update({} if backend == 'original' else {'device': device}); "
            "kwargs.update({} if backend != 'rust' else {'stream_policy': 'auto', 'sdf_batch_size': rust_sdf_batch_size}); "
            "result = fn(cameras, bounds, 16, sdf, **kwargs); "
            "print(json.dumps(result))"
        ),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "backend": backend,
            "device": device,
            "sdf": sdf,
            "error": proc.stderr.strip() or proc.stdout.strip() or f"subprocess exited with {proc.returncode}",
        }
    result = json.loads(proc.stdout.strip())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OcMesher backend benchmarks in isolated subprocesses to avoid OpenMP runtime conflicts.",
    )
    parser.add_argument("--sdf", default="terrain", help="sphere, terrain, gyroid, or all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--devices", default="cpu,mps", help="Comma-separated device list")
    parser.add_argument("--backends", default="original,torch,rust", help="Comma-separated backend list")
    parser.add_argument("--rust-sdf-batch-size", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional aggregate JSON output path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sdfs = ["sphere", "terrain", "gyroid"] if args.sdf == "all" else [args.sdf]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    results: dict[str, object] = {}
    for sdf in sdfs:
        for backend in backends:
            backend_devices = ["cpu"] if backend == "original" else devices
            for device in backend_devices:
                key = f"{backend}_{device}_{sdf}"
                results[key] = _run_one(
                    repo_root,
                    backend,
                    device,
                    sdf,
                    args.runs,
                    args.warmup,
                    args.rust_sdf_batch_size,
                )

    if args.output:
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()