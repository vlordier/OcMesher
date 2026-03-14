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

### Fixed
- `load_cdll` — replaced fragile `sys.path[-1]` lookup with an ordered search
  over all `sys.path` entries; absolute paths are used directly.

### Changed
- Pre-commit hooks updated to the latest stable revisions.

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
