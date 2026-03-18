## OcMesher: View-Dependent Octree-based Mesh Extraction in Unbounded Scenes for Procedural Synthetic Data

[![CI](https://github.com/princeton-vl/OcMesher/actions/workflows/ci.yml/badge.svg)](https://github.com/princeton-vl/OcMesher/actions/workflows/ci.yml)
[![Lint](https://github.com/princeton-vl/OcMesher/actions/workflows/lint.yml/badge.svg)](https://github.com/princeton-vl/OcMesher/actions/workflows/lint.yml)
[![Static Analysis](https://github.com/princeton-vl/OcMesher/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/princeton-vl/OcMesher/actions/workflows/static-analysis.yml)
[![Docs](https://github.com/princeton-vl/OcMesher/actions/workflows/docs.yml/badge.svg)](https://princeton-vl.github.io/OcMesher/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/princeton-vl/OcMesher/blob/main/LICENSE)

Implementation source-code for <it>OcMesher</it>, which extracts a mesh for an unbounded scene represented by signed distance functions (SDFs). Even though the scene is unbounded, the mesh is memory-efficient, and highly detailed from a given set of camera views. OcMesher is used by default in [Infinigen](https://github.com/princeton-vl/infinigen) to speed up video generation, improve rendering quality, and export terrain meshes to external simulators.

<img src=".github/OcMesher.png" width='1000'>

If you use OcMesher in your work, please cite our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2312.08364">
        View-Dependent Octree-based Mesh Extraction in Unbounded Scenes for Procedural Synthetic Data
    </a>
</h3>
<p align="center">
    <a href="https://mazeyu.github.io/">Zeyu Ma</a>, 
    <a href="http://araistrick.com/">Alexander Raistrick</a>, 
    <a href="https://www.lahavlipson.com/">Lahav Lipson</a>, 
    <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
</p>

```
@article{ocmesher2023view,
  title={View-Dependent Octree-based Mesh Extraction in Unbounded Scenes for Procedural Synthetic Data},
  author={Ma, Zeyu and Raistrick, Alexander and Lipson, Lahav and Deng, Jia},
  year={2023}
}
```

Please view the video [here](https://youtu.be/YA1c5L0Ncuw) for more qualitative results

## Getting Started

:bulb: Note: OcMesher is installed by default in Infinigen as of v1.2.0 - if you wish to use OcMesher with Infinigen please follow the Installation instructions on the infinigen repo. Use the instructions below only if you want a standalone installation & demo. 

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Python >= 3.10 (managed automatically by uv)
- A C++ compiler: `g++` on Linux, or LLVM's `clang++` on macOS

### Standalone Installation

```bash
git clone https://github.com/princeton-vl/OcMesher.git
cd OcMesher
uv sync
bash install.sh
```

If you want to use the PyTorch backend as well, install the optional torch
dependency group before running the demo or tests that import
`ocmesher.TorchOcMesher`:

```bash
uv sync --extra torch
```


### Demo

```bash
uv run python demo.py
```

This example uses one camera and the Perlin Noise from the Python library `vnoise` and outputs the resulting mesh in `results/demo.obj`.

For benchmark runs with optional acceleration backends, install benchmark extras:

```bash
uv sync --extra benchmark
```

### Running Benchmarks

To run the full benchmark suite, you can use the Makefile target:

```bash
make bench
```

This will run the end-to-end benchmarks for the available backends.

Alternatively, you can run the benchmark script directly:

```bash
uv run python benchmarks/run_benchmark.py
```

### Development Checks

For the main Python/C++ repository quality gate, use:

```bash
make quality
```

This runs Ruff, the configured type checker, and the Python test suite.

For the Rust workspace, use `cargo fmt` for formatting and `cargo clippy` for linting.

To check formatting and linting:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

### Benchmark Validation Workflow

For local performance and regression validation, use the Rust-native benchmark suite:

```bash
cargo bench
uv run python benchmarks/bench_rust_scene.py --runs 3 --snapshot-json benchmark_artifacts/generated/bench_rust_scene_snapshot.json
```

To persist benchmark results with run metadata for later branch-to-branch comparison, use the output from `cargo bench` or custom Rust-native benchmarking tools.

The benchmark helpers also support comparing snapshot files from two runs or
branches:

```bash
uv run python benchmarks/bench_python_overhead.py --compare old.json new.json
uv run python benchmarks/bench_e2e.py --compare old.json new.json
uv run python benchmarks/bench_rust_scene.py --compare benchmark_artifacts/generated/bench_rust_scene_snapshot.json benchmark_artifacts/generated/bench_rust_scene_mps_snapshot.json
```

Notes:
- Snapshot files include execution metadata such as branch, commit, Python version, platform, and command line.
- Local run outputs should go under `benchmark_artifacts/generated/` (git-ignored) to keep `benchmark_artifacts/` snapshots tidy.
- Optional runtime tiers such as `numba` and `mlx` are skipped automatically when their dependencies are not installed.
- The MLX benchmark is a narrow SDF-only pilot on macOS, not a full mesher backend.
- The Rust scene benchmark compares the compiled `extract_native_scene(...)` and `extract_tch_scene(...)` pilot paths on the same primitive list.
- The Rust scene benchmark can also compare two snapshot JSON files directly, which is useful for CPU-vs-MPS or branch-to-branch pilot comparisons.

To perform deterministic numerical parity checks against `main`, ensure both branches are built with the default build script (`bash install.sh`) and compare mesh counts/sums for the same fixed camera/SDF case.

```bash
# default parity check against main
uv run python benchmark.py --upstream-parity

# stricter automation-friendly parity check (non-zero exit on mismatch)
uv run python benchmark.py --upstream-parity --upstream-parity-strict

# custom parity mesh settings and tolerance
uv run python benchmark.py --upstream-parity main --parity-pixels-per-cube 24 --parity-coarse-count 200000 --parity-atol 1e-10

# write parity payload to JSON for tooling/CI artifacts
uv run python benchmark.py --upstream-parity --json benchmark_parity.json
```

Notes:
- `--upstream-parity-strict` requires `--upstream-parity`.
- `--parity-pixels-per-cube` and `--parity-coarse-count` must be >= 1.
- `--parity-atol` must be >= 0.

### Speedup Plot (Plotters)

To visualize scene-level speedups from the parity artifact, use the Rust Plotters
utility in the Rust workspace:

```bash
cargo run --release --manifest-path ocmesher-rust/Cargo.toml -p ocmesher-benchplot -- \
    --input benchmark_artifacts/bench_api_parity_upstream_main.json \
    --output benchmark_artifacts/bench_api_speedups.png
```

The chart below uses speedup defined as `upstream_avg_ms / current_avg_ms`, so
values above `1.0x` mean the current Rust wrapper is faster.

![API parity speedups](https://raw.githubusercontent.com/princeton-vl/OcMesher/main/benchmark_artifacts/bench_api_speedups.png)

### Torch Backend Contract

`ocmesher.TorchOcMesher` now exposes a small backend contract intended for
future accelerator backends and zero-copy integration work:

```python
from ocmesher.torch_core import TorchOcMesher

mesher = TorchOcMesher(cameras, bounds, device="cpu")
caps = mesher.get_capabilities()
points = mesher.as_backend_tensor([[0.0, 0.0, 0.0]])
distances = mesher.evaluate_sdf_batch([sdf], points)
dlpack_capsule = mesher.to_backend_dlpack(points)
```

Current contract surface:
- `version`: explicit backend contract version string.
- `get_capabilities()`: reports device/runtime support and preferred batch policy.
- `as_backend_tensor(...)`: normalises NumPy, torch, or DLPack-backed positions to backend-native tensors.
- `evaluate_sdf_batch(...)`: public batched SDF entry point with optional output reuse and CUDA stream control.
- `to_backend_dlpack(...)`: exports backend-native tensors for zero-copy exchange.

The shared protocol and capability dataclass live in [ocmesher/backend_contract.py](https://github.com/princeton-vl/OcMesher/blob/main/ocmesher/backend_contract.py).

### Backend Options (Simple User API)

Use `ocmesher.make_ocmesher(...)` to pick a backend explicitly:

```python
from ocmesher import make_ocmesher

# Original C++ backend
mesher_cpp = make_ocmesher(cameras, bounds, backend="cpp")

# PyTorch backend on CPU or MPS
mesher_torch_cpu = make_ocmesher(cameras, bounds, backend="torch", device="cpu")
mesher_torch_mps = make_ocmesher(cameras, bounds, backend="torch", device="mps")

# Rust wrapper backend with tch kernels
mesher_rust_cpu = make_ocmesher(cameras, bounds, backend="rust", device="cpu")
mesher_rust_mps = make_ocmesher(cameras, bounds, backend="rust", device="mps")

# Force Rust tch kernels on CPU or MPS (useful for Infinigen runtime wiring)
mesher_rust_tch_cpu = make_ocmesher(cameras, bounds, backend="rust", device="cpu", kernel_runtime="tch")
mesher_rust_tch_mps = make_ocmesher(cameras, bounds, backend="rust", device="mps", kernel_runtime="tch")
```

Backend selection notes:
- `backend="cpp"`: best baseline for parity with the original OcMesher behavior.
- `backend="torch"`: full PyTorch implementation (`TorchOcMesher`).
- `backend="rust"`: Rust wrapper path (`make_rust_ocmesher`) with native/tch acceleration paths.
- `kernel_runtime`: Rust wrapper kernel selection policy.
    - `"auto"` (default): prefers `tch` on `mps/cuda`, native kernels on `cpu`.
    - `"native"`: always use native Rust kernels when primitive-scene paths are available.
    - `"tch"`: force Rust `tch` kernels for supported primitive-scene paths on `cpu/mps/cuda`.

### Performance Guide: Choosing a Backend

| Backend | Platform | Relative Speed | Notes |
|---------|----------|----------------|-------|
| `torch` + `device="mps"` | Apple Silicon (M1–M4) | ⚡ ~2.2× faster than CPU | **Recommended on macOS** |
| `torch` + `device="cuda"` | NVIDIA GPU | ⚡ fastest on Linux/Windows | Requires CUDA-enabled PyTorch |
| `torch` + `device="cpu"` | Any | moderate | Good cross-platform fallback |
| `cpp` | Linux / macOS | baseline | Original impl; requires `bash install.sh` |
| `rust` + `device="mps"` | Apple Silicon | fast | Adaptive CPU/GPU routing available |
| `rust` + `device="cpu"` | Any | similar to torch CPU | Rust overhead varies with batch size |

**Quick selection guide:**

- **Apple Silicon Mac (M1/M2/M3/M4):** Use `backend="torch", device="mps"` — approximately 2.2× faster than CPU, with the coarse octree step benefiting most from GPU parallelism.
- **NVIDIA GPU (Linux/Windows):** Use `backend="torch", device="cuda"` for maximum throughput.
- **CPU-only / CI / reproducibility:** Use `backend="cpp"` for exact parity with the upstream implementation, or `backend="torch", device="cpu"` as a pure-Python fallback.
- **Rust experiments:** Use `backend="rust"` with optional `OCMESHER_TCH_MPS_ADAPTIVE=1` (Apple Silicon) or `OCMESHER_TCH_CUDA_ADAPTIVE=1` (NVIDIA) for adaptive per-batch device routing.

Typical pipeline timing on Apple M4 Pro with `device="mps"`:

| Stage | Time |
|-------|------|
| Coarse octree | ~0.26 s |
| Find surface | ~0.15 s |
| Visibility filter | ~0.03 s |
| Construct mesh | ~0.01 s |
| **Total** | **~0.45 s** |

> **macOS OpenMP note:** The C++ backend requires `libomp.dylib` from LLVM. Install it with `brew install llvm`, then rebuild with `bash install.sh`. The build script automatically embeds the correct rpath into `core.so` so it can find the library at runtime.

### Rust Backend

The repo now also carries a Rust integration workspace based on the
`rust-integration-infinigen-v1` branch. The Python entry points are
`ocmesher.RustOcMesher` and `ocmesher.make_rust_ocmesher(...)`.

Build the PyO3 extension into the active environment with:

```bash
uv sync --extra rust
uv run maturin develop --release --manifest-path ocmesher-rust/crates/ocmesher-py/Cargo.toml
```

Then use it from Python like this:

```python
from ocmesher import make_rust_ocmesher

mesher = make_rust_ocmesher(cameras, bounds)
meshes, in_view_tags = mesher([sdf])
```

Notes:
- The Rust workspace currently bridges to the existing `ocmesher/lib/core.so` shared library rather than replacing the meshing algorithm with a pure Rust core.
- `bash install.sh` still needs to be run first so `core.so` exists for the Rust extension to load.
- The Rust wrapper batches SDF evaluation through the same public backend contract used by the Torch backend.
- `ocmesher-rust/crates/ocmesher-core` now also exposes a Rust-native `SdfEvaluator` path via `run_meshing_pipeline_native(...)`, so the meshing pipeline can be driven from Rust without Python callables.
- The compiled extension also exposes `Backend.extract_native_sphere(...)`, `Backend.extract_native_plane(...)`, `Backend.extract_native_sphere_plane(...)`, and `Backend.extract_native_scene(...)` as no-Python-callback pilot paths, including a small primitive-spec scene API.
- The optional `tch-kernels` feature adds tensor-backed Rust paths, including `Backend.extract_tch_sphere(...)`, `Backend.extract_tch_plane(...)`, `Backend.extract_tch_sphere_plane(...)`, and `Backend.extract_tch_scene(...)`, and builds when `LIBTORCH` points at a valid libtorch root; on macOS the extension links with an rpath targeting the Python `torch/lib` bundle.
- For Apple Silicon, set `OCMESHER_TCH_MPS_ADAPTIVE=1` to enable adaptive per-batch MPS routing (CPU vs GPU) in the Rust `tch` kernels; this is useful when medium-sized batches run faster on CPU due transfer/dispatch overhead.
- On CUDA systems, set `OCMESHER_TCH_CUDA_ADAPTIVE=1` to enable the same adaptive CPU-vs-GPU routing policy for Rust `tch` kernels.
- Set `OCMESHER_TCH_ADAPTIVE_DEBUG=1` to print adaptive calibration/cache decisions (device, kernel, bucket, chosen route) for profiling and tuning.

Relevant files:
- [ocmesher/rust_backend.py](https://github.com/princeton-vl/OcMesher/blob/main/ocmesher/rust_backend.py)
- [ocmesher-rust/Cargo.toml](https://github.com/princeton-vl/OcMesher/blob/main/ocmesher-rust/Cargo.toml)
- [ocmesher-rust/crates/ocmesher-core/src/lib.rs](https://github.com/princeton-vl/OcMesher/blob/main/ocmesher-rust/crates/ocmesher-core/src/lib.rs)
- [ocmesher-rust/crates/ocmesher-py/src/lib.rs](https://github.com/princeton-vl/OcMesher/blob/main/ocmesher-rust/crates/ocmesher-py/src/lib.rs)
