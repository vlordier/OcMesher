# Contributing to OcMesher

Thank you for your interest in contributing to OcMesher!

## Getting Started

1. Fork the repository
2. Clone your fork and create a new branch for your work
3. Set up your environment:

```bash
git clone https://github.com/princeton-vl/OcMesher.git
cd OcMesher
uv sync
bash install.sh
```

> **Apple Silicon M4 note:** For Apple Silicon (M1–M4) benchmarking with the
> Rust-native backend, use `install_optimized.sh` instead:
> ```bash
> bash install_optimized.sh
> ```

4. (Optional) Install pre-commit hooks to automatically lint and format before each commit:

```bash
uv run pre-commit install
```

## Development Guidelines

- Follow the project style enforced by **ruff** (see `pyproject.toml` for rules).
- Run linting and tests before submitting (see commands below).

## Common Development Commands

### Python

```bash
# Run tests (parallel with pytest-xdist)
uv run pytest tests/ -v --ignore=tests/cpp -n auto

# Run linting and formatting
uv run ruff check .
uv run ruff format --check .

# Run all quality checks (lint + tests)
make quality
```

### Rust

```bash
# Build the Rust project (toolchain auto-detected via rust-toolchain.toml)
cargo build --release

# Run all Rust tests
cargo test

# Lint and format (clippy and fmt are auto-configured via rust-toolchain.toml)
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

You can also use the provided `Makefile` shortcuts (for convenience):
```

## Submitting Changes

1. Commit your changes with a clear, descriptive message following
   [Conventional Commits](https://www.conventionalcommits.org/) style
   (e.g. `fix: ...`, `feat: ...`, `chore: ...`, `refactor: ...`).
2. Push to your fork and open a Pull Request.
3. Describe what your changes do and why they are needed.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:

- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Your OS and Rust version

## License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.
