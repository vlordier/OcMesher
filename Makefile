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
#   make lint       run Python, Rust, and C++ lint entry points
#   make format     ruff format check
#   make typecheck  run ty static type checker
#   make lint-rust  run Rust fmt + clippy baseline
#   make lint-cpp   run C++ formatting/static-analysis checks when tools exist
#   make fix        auto-fix lint + apply formatting
#   make clean      remove compiled artifacts

.PHONY: install test coverage lint lint-python lint-rust lint-cpp format typecheck quality fix clean

install:
	uv sync
	bash install.sh

test:
	uv run python -m pytest tests/ -v --ignore=tests/cpp

coverage:
	uv run python -m pytest tests/ -v --ignore=tests/cpp --cov=ocmesher --cov-report=term-missing --cov-report=html

lint:
	$(MAKE) lint-python
	$(MAKE) lint-rust

lint-python:
	uv run ruff check .
	uv run ty check ocmesher/

lint-rust:
	cd ocmesher-rust && cargo fmt --all --check
	cd ocmesher-rust && cargo clippy --workspace --all-targets -- -D warnings -D clippy::dbg_macro -D clippy::todo -D clippy::unimplemented

lint-cpp:
	FORMAT_BIN=$$(command -v clang-format-18 || command -v clang-format); \
	TIDY_BIN=$$(command -v clang-tidy-18 || command -v clang-tidy); \
	CPPCHECK_BIN=$$(command -v cppcheck); \
	if [ -z "$$FORMAT_BIN" ] || [ -z "$$TIDY_BIN" ] || [ -z "$$CPPCHECK_BIN" ]; then \
		echo "Missing clang-format/clang-tidy/cppcheck; install C++ lint tools first."; \
		exit 1; \
	fi; \
	find ocmesher/source tests/cpp \( -name "*.cpp" -o -name "*.h" \) -print0 | xargs -0 $$FORMAT_BIN --dry-run --Werror; \
	find ocmesher/source -name "*.cpp" -print0 | xargs -0 $$TIDY_BIN --extra-arg=-std=c++17 --extra-arg=-fopenmp --warnings-as-errors='*' --; \
	$$CPPCHECK_BIN --std=c++17 --enable=all --inconclusive --suppress=missingIncludeSystem --suppress=unusedFunction --error-exitcode=1 ocmesher/source/

format:
	uv run ruff format --check .

typecheck:
	uv run ty check ocmesher/

quality:
	$(MAKE) lint-python
	$(MAKE) lint-rust
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
