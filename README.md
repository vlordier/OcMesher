## OcMesher: View-Dependent Octree-based Mesh Extraction in Unbounded Scenes for Procedural Synthetic Data

[![CI](https://github.com/princeton-vl/OcMesher/actions/workflows/ci.yml/badge.svg)](https://github.com/princeton-vl/OcMesher/actions/workflows/ci.yml)
[![Lint](https://github.com/princeton-vl/OcMesher/actions/workflows/lint.yml/badge.svg)](https://github.com/princeton-vl/OcMesher/actions/workflows/lint.yml)
[![Static Analysis](https://github.com/princeton-vl/OcMesher/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/princeton-vl/OcMesher/actions/workflows/static-analysis.yml)
[![Docs](https://github.com/princeton-vl/OcMesher/actions/workflows/docs.yml/badge.svg)](https://princeton-vl.github.io/OcMesher/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

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

### Development Checks

The repo uses Ruff for linting/formatting and Pytest for tests.

To mirror local and CI policy checks, install and run pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Quality policy notes:
- Naming consistency and general style are enforced by Ruff.
- Function complexity and branch depth are linted via Ruff/Pylint rule families (for example `C901`, `PLR0912`, `PLR0915`) in files that are not explicitly waived in `pyproject.toml`.
- Line length is configured at 120 columns; per-file suppressions are documented in `pyproject.toml`.
- File-size guardrails for high-churn scripts are enforced by `tests/file_length_guard.py` in pre-commit and CI.

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
uv run python tests/file_length_guard.py
```

### Benchmark Validation Workflow

For local performance and regression validation, use the same sequence as CI/maintenance work:

```bash
uv run ruff check .
uv run pytest
uv run python benchmark.py --configs small --runs 1
```

To persist benchmark results with run metadata for later branch-to-branch
comparison, write snapshot JSON artifacts instead of plain results:

```bash
uv run python benchmark.py --configs small --runs 1 --snapshot-json benchmark_artifacts/benchmark_tiers_snapshot.json
uv run python benchmarks/bench_python_overhead.py --sizes 100 1000 --snapshot-json benchmark_artifacts/bench_python_overhead_snapshot.json
uv run python benchmarks/bench_e2e.py --sizes 1000 10000 --snapshot-json benchmark_artifacts/bench_e2e_snapshot.json
uv run python benchmarks/bench_mlx_sdf.py --sizes 1000 10000 --snapshot-json benchmark_artifacts/bench_mlx_sdf_snapshot.json
```

The benchmark helpers also support comparing snapshot files from two runs or
branches:

```bash
uv run python benchmarks/bench_python_overhead.py --compare old.json new.json
uv run python benchmarks/bench_e2e.py --compare old.json new.json
```

Notes:
- Snapshot files include execution metadata such as branch, commit, Python version, platform, and command line.
- Optional runtime tiers such as `numba` and `mlx` are skipped automatically when their dependencies are not installed.
- The MLX benchmark is a narrow SDF-only pilot on macOS, not a full mesher backend.

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

The shared protocol and capability dataclass live in [ocmesher/backend_contract.py](ocmesher/backend_contract.py).

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
- The optional `tch-kernels` feature is intended for tensor-backed Rust SDF kernels, but building it requires a working libtorch / `torch-sys` setup in the build environment.

Relevant files:
- [ocmesher/rust_backend.py](ocmesher/rust_backend.py)
- [ocmesher-rust/Cargo.toml](ocmesher-rust/Cargo.toml)
- [ocmesher-rust/crates/ocmesher-core/src/lib.rs](ocmesher-rust/crates/ocmesher-core/src/lib.rs)
- [ocmesher-rust/crates/ocmesher-py/src/lib.rs](ocmesher-rust/crates/ocmesher-py/src/lib.rs)
