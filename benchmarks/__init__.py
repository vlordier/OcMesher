"""Benchmarks comparing the C++ and PyTorch OcMesher backends.

Run from the repository root::

    python -m benchmarks.run_benchmark            # quick run (small grid)
    python -m benchmarks.run_benchmark --full     # full resolution comparison
    python -m benchmarks.run_benchmark --profile  # sub-operation profiling
    python -m benchmarks.run_benchmark --sdf all  # sphere / terrain / gyroid SDFs

Requires PyTorch for the ``torch`` backend::

    pip install torch
"""
