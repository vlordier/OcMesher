## OcMesher: View-Dependent Octree-based Mesh Extraction in Unbounded Scenes for Procedural Synthetic Data

[![CI](https://github.com/princeton-vl/OcMesher/actions/workflows/ci.yml/badge.svg)](https://github.com/princeton-vl/OcMesher/actions/workflows/ci.yml)
[![Lint](https://github.com/princeton-vl/OcMesher/actions/workflows/lint.yml/badge.svg)](https://github.com/princeton-vl/OcMesher/actions/workflows/lint.yml)
[![Static Analysis](https://github.com/princeton-vl/OcMesher/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/princeton-vl/OcMesher/actions/workflows/static-analysis.yml)
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

Quality policy notes:
- Naming consistency and general style are enforced by Ruff.
- Function complexity and branch depth are linted via Ruff/Pylint rule families (for example `C901`, `PLR0912`, `PLR0915`) in files that are not explicitly waived in `pyproject.toml`.
- Line length is configured at 120 columns; per-file suppressions are documented in `pyproject.toml`.

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

### Benchmark Validation Workflow

For local performance and regression validation, use the same sequence as CI/maintenance work:

```bash
uv run ruff check .
uv run pytest
uv run python benchmark.py --configs small --runs 1
```

To perform deterministic numerical parity checks against `main`, ensure both branches are built with the default build script (`bash install.sh`) and compare mesh counts/sums for the same fixed camera/SDF case.
