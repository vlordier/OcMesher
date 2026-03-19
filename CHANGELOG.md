# Changelog

All notable changes to OcMesher are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `CHANGELOG.md` — this file.
- `SECURITY.md` — responsible-disclosure policy.
- `CITATION.cff` — FAIR software citation metadata.
- `OcMesher.__repr__` for clear debugging output.
- `OcMesher.__enter__` / `__exit__` context-manager support.
- `kernel_caller` now validates the shape of the SDF array returned by each
  kernel, raising `ValueError` with a descriptive message on mismatch.
- `Makefile` targets: `coverage` (pytest-cov HTML report) and `typecheck` (ty).
- CI status badges in `README.md`.
- MLX backend (`MLXOcMesher`) for Apple Silicon, with octree and marching
  cubes parity against the PyTorch backend.
- Python 3.12 and 3.13 classifiers in `pyproject.toml`.

### Fixed
- `load_cdll` — replaced fragile `sys.path[-1]` lookup with an ordered search
  over all `sys.path` entries; absolute paths are used directly.
- MLX `_build_coarse_octree` / `_refine_surface_octree` — used per-cube
  levels instead of a single first-cube level for child generation.
- MLX projection math — use float64 precision to match Torch backend.
- README — corrected Python version badge (3.11+, matching
  `requires-python`), removed references to non-existent benchmark scripts.
- `file_length_guard.py` — removed stale entries for deleted scripts, added
  limits for `core.py`, `mlx_core.py`, and `rust_backend.py`.

### Changed
- Pre-commit hooks updated to the latest stable revisions.
- Moved `mkdocs` / `mkdocs-material` / `mkdocstrings` from core dependencies
  to the `docs` optional-dependency group.
- `ocmesher/__init__.py` — replaced chained if/return with a lookup-table
  pattern for lazy imports (eliminates PLR0911).
- `ocmesher/torch_core.py` — moved `_constants` import to top of file
  (fixes E402), moved `Callable` into `TYPE_CHECKING` block (fixes TC003).

---

## [1.0.0] — 2024-01-01

### Added
- Initial public release.
- Octree-based mesher driven by signed-distance functions (C++ backend).
- PyTorch backend (`TorchOcMesher`) supporting CUDA, MPS, and CPU devices.
- Input validation helpers (`_validate_cameras`, `_validate_bounds`,
  `_validate_kernels`).
- `Timer` context manager with wall-clock timing and memory reporting.
- Comprehensive Python test suite (pytest).
- Ruff lint/format configuration.
- Pre-commit hooks (ruff, clang-format, clang-tidy).
- GitHub Actions CI: build, test, lint, static analysis, ShellCheck.
- Dependabot configuration for automated dependency updates.
