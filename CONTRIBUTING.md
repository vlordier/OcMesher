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

## Development Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code style
- Add type hints to new Python code
- Keep C++ changes compatible with both GCC and Clang
- Test your changes with `uv run python demo.py` before submitting

## Submitting Changes

1. Commit your changes with a clear, descriptive message
2. Push to your fork and open a Pull Request
3. Describe what your changes do and why they are needed

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:

- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Your OS and Python version

## License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.
