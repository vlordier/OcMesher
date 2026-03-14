# Common development targets for OcMesher.
#
# Prerequisites:
#   uv  – https://docs.astral.sh/uv/
#   g++ / clang++ – to compile the C++ shared library
#
# Usage:
#   make install   build C++ library and sync Python deps
#   make test      run Python test suite
#   make lint      ruff lint check
#   make format    ruff format check
#   make fix       auto-fix lint + apply formatting
#   make clean     remove compiled artifacts

.PHONY: install test lint format fix clean

install:
	uv sync
	bash install.sh

test:
	uv run python -m pytest tests/ -v --ignore=tests/cpp

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

fix:
	uv run ruff check . --fix
	uv run ruff format .

clean:
	rm -rf ocmesher/lib/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
