# Common development targets for OcMesher.
#
# Prerequisites:
#   uv  – https://docs.astral.sh/uv/
#   g++ / clang++ – to compile the C++ shared library
#
# Usage:
#   make install    build C++ library and sync Python deps
#   make test       run Python test suite
#   make coverage   run tests with coverage report
#   make lint       ruff lint check
#   make format     ruff format check
#   make typecheck  run ty static type checker
#   make fix        auto-fix lint + apply formatting
#   make clean      remove compiled artifacts

.PHONY: install test coverage lint format typecheck quality fix clean

install:
	uv sync
	bash install.sh

test:
	uv run python -m pytest tests/ -v --ignore=tests/cpp

coverage:
	uv run python -m pytest tests/ -v --ignore=tests/cpp --cov=ocmesher --cov-report=term-missing --cov-report=html

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run ty check ocmesher/

quality:
	uv run ruff check .
	uv run ty check ocmesher/
	uv run python -m pytest tests/ -v --ignore=tests/cpp

fix:
	uv run ruff check . --fix
	uv run ruff format .

clean:
	rm -rf ocmesher/lib/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .pyright htmlcov .coverage
	rm -f *.log *.tmp
