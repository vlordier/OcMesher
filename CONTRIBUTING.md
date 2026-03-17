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

4. (Optional) Install pre-commit hooks to automatically lint and format before each commit:

```bash
uv run pre-commit install
```

## Development Guidelines

- Follow the project style enforced by **ruff** (see `pyproject.toml` for rules).
<!-- Python and C++ guidelines removed: now Rust-native only -->
- Run linting and tests before submitting (see commands below).

## Common Development Commands

```bash
## Common Development Commands

```bash
# Build the Rust project
cargo build --release

# Run all Rust tests
cargo test

# Lint and format
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```
```

You can also use the provided `Makefile` shortcuts:

```bash
<!-- Makefile shortcuts for Python/C++ removed -->
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
