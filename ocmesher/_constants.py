"""Centralised numeric and string constants shared across OcMesher backends.

Having all tuneable defaults in one place makes it easy to discover,
document, and adjust them without hunting through multiple files.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = [
    "CAMERA_DATA_STRIDE",
    "CORNER_QUANT_SCALE",
    "DEFAULT_BISECTION_ITERS",
    "DEFAULT_BISECTION_TOL",
    "DEFAULT_COARSE_COUNT",
    "DEFAULT_ENCLOSED",
    "DEFAULT_INV_SCALE",
    "DEFAULT_MEMORY_LIMIT_MB",
    "DEFAULT_MIN_DIST",
    "DEFAULT_PIXELS_PER_CUBE",
    "DEFAULT_SDF_WORKERS",
    "DEFAULT_SIMPLIFY_OCCLUDED",
    "DEFAULT_VISIBLE_RELAX_ITER",
    "DENOM_EPS",
    "DEVICE_FAMILY_VALUES",
    "DEVICE_VALUES",
    "HASH_PRIME_X",
    "HASH_PRIME_Y",
    "HASH_PRIME_Z",
    "KERNEL_RUNTIME_VALUES",
    "MAX_SDF_WORKERS",
    "NUMPY_FLOAT64",
    "NUMPY_INT32",
    "SDF_BATCH_SIZE",
    "SERIAL_MULTI_KERNEL_MAX",
    "SPEC_INFERENCE_ATOL",
    "SPEC_INFERENCE_RTOL",
    "STREAM_POLICY_VALUES",
    "TORCH_SDF_CHUNK_SIZE",
]

# Number of float64 values per camera in the packed camera buffer
# consumed by the C++ backend.
CAMERA_DATA_STRIDE: int = 23

# Maximum number of SDF query points evaluated in a single vectorised batch.
# Keeping this below ~10M avoids exhausting RAM on large octrees.
SDF_BATCH_SIZE: int = 10_000_000

# Default number of SDF worker threads when ``os.cpu_count()`` is unavailable.
DEFAULT_SDF_WORKERS: int = 4

# Maximum number of SDF worker threads for multi-kernel evaluation.
# Defaults to available CPU count but can be overridden via the
# ``OCMESHER_SDF_WORKERS`` environment variable.
MAX_SDF_WORKERS: int = int(
    os.environ.get(
        "OCMESHER_SDF_WORKERS",
        str(os.cpu_count() or DEFAULT_SDF_WORKERS),
    )
)

# Epsilon for safe division in marching-cubes edge interpolation.
DENOM_EPS: float = 1e-12

# Valid device identifiers for Rust backend.
# - "cpu": use CPU via PyTorch
# - "mps": use Apple Silicon GPU via PyTorch
# - "cuda": use NVIDIA GPU via PyTorch
# - "mlx": use Apple Silicon via MLX (native Apple Silicon ML framework)
# - "cuda:<index>": use specific NVIDIA GPU by ordinal
DEVICE_VALUES: tuple[str, str, str, str] = ("cpu", "mps", "cuda", "mlx")

# Device families for grouping devices.
DEVICE_FAMILY_VALUES: tuple[str, str, str, str] = ("cpu", "cuda", "mps", "mlx")

# Scale factor for quantising corner positions to ``int64`` before
# deduplication.  1e8 gives 10-nanometre resolution — fine enough to
# distinguish corners at the deepest practical octree level.
CORNER_QUANT_SCALE: float = 1e8

# Large primes used for spatial hashing of quantised 3-D coordinates.
# ``hash = x * P0 + y * P1 + z * P2`` produces a well-distributed 1-D
# key for radix-sort-based deduplication in marching cubes and surface
# detection.  The primes are chosen to be close together (~1e9) so that
# the hash stays within int64 range for typical coordinate magnitudes.
HASH_PRIME_X: int = 1_000_000_007
HASH_PRIME_Y: int = 1_000_000_009
HASH_PRIME_Z: int = 1_000_000_021

# Chunk size for batched SDF evaluation in the PyTorch backend.
# Smaller than SDF_BATCH_SIZE (which targets the C++ backend) to
# improve cache locality during GPU→CPU transfers.
TORCH_SDF_CHUNK_SIZE: int = 2_000_000

# ---------------------------------------------------------------------------
# NumPy dtype aliases (canonical forms used throughout the codebase)
# ---------------------------------------------------------------------------
NUMPY_FLOAT64: type = np.float64  # type: ignore[attr-defined]
NUMPY_INT32: type = np.int32  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Rust-backend defaults (mirrored from OcMesher for API parity)
# ---------------------------------------------------------------------------
DEFAULT_PIXELS_PER_CUBE: int = 8
DEFAULT_INV_SCALE: int = 10
DEFAULT_MIN_DIST: int = 1
DEFAULT_MEMORY_LIMIT_MB: int = 1000
DEFAULT_BISECTION_ITERS: int = 15
DEFAULT_BISECTION_TOL: float = 0.0
DEFAULT_ENCLOSED: bool = True
DEFAULT_SIMPLIFY_OCCLUDED: bool = True
DEFAULT_VISIBLE_RELAX_ITER: int = 2
DEFAULT_COARSE_COUNT: int = 500_000

# ---------------------------------------------------------------------------
# Tolerance constants for native primitive spec inference
# ---------------------------------------------------------------------------
# Absolute tolerance for checking SDF values at test points.
SPEC_INFERENCE_ATOL: float = 1e-5

# Relative tolerance for comparing SDF values against expected primitives.
SPEC_INFERENCE_RTOL: float = 1e-4

# ---------------------------------------------------------------------------
# Serial multi-kernel threshold
# ---------------------------------------------------------------------------
# Keep very small multi-kernel batches serial to avoid pool startup overhead.
SERIAL_MULTI_KERNEL_MAX: int = 4096

# Valid values for ``kernel_runtime`` in Rust backend.
# - "auto": choose best available runtime automatically (default)
# - "native": use native Rust kernel implementations
# - "tch": use PyTorch (libtorch) via tch-rs bindings
KERNEL_RUNTIME_VALUES: tuple[str, str, str] = ("auto", "native", "tch")

# Valid values for ``stream_policy`` in Rust backend.
# - "sync": synchronous execution (default for CPU)
# - "auto": CUDA/MPS stream-based execution when available
STREAM_POLICY_VALUES: tuple[str, str] = ("sync", "auto")
