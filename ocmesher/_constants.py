"""Centralised numeric constants shared across OcMesher backends.

Having all tuneable defaults in one place makes it easy to discover,
document, and adjust them without hunting through multiple files.
"""

from __future__ import annotations

import os

__all__ = [
    "CAMERA_DATA_STRIDE",
    "CORNER_QUANT_SCALE",
    "DEFAULT_SDF_WORKERS",
    "DENOM_EPS",
    "MAX_SDF_WORKERS",
    "SDF_BATCH_SIZE",
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

# Scale factor for quantising corner positions to ``int64`` before
# deduplication.  1e8 gives 10-nanometre resolution — fine enough to
# distinguish corners at the deepest practical octree level.
CORNER_QUANT_SCALE: float = 1e8
