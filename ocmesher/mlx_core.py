# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""MLX-based octree mesher with native Apple Silicon acceleration.

Provides :class:`MLXOcMesher`, a drop-in replacement for :class:`OcMesher`
that uses Apple's MLX framework for GPU-accelerated mesh extraction on Apple Silicon.
This is an alternative to the PyTorch-based :class:`TorchOcMesher` backend.

Key advantages of MLX over PyTorch MPS:
- Native Apple Silicon optimization with unified memory architecture
- Better memory bandwidth utilization for GPU-resident data
- Optimized for Apple Silicon neural engine (ANE) when available
- Lower memory overhead compared to PyTorch

Supported devices:
- **MLX** - Apple Silicon GPUs via ``mlx`` (M1/M2/M3/M4 chips)
- **CPU** - fallback for all platforms

Note: MLX requires float32 exclusively. The mesher will automatically use
float32 for all computations when running on MLX device.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import trimesh

from ._types import BoundsLike, CamerasTuple, KernelSequence, MeshResult
from ._validation import validate_bounds, validate_cameras, validate_kernels, validate_mesher_params
from .utils.timer import Timer

logger = logging.getLogger(__name__)

__all__ = ["MLXOcMesher"]


def _mlx_available() -> bool:
    """Return True if MLX is available on this system."""
    try:
        import mlx.core as mx

        return True
    except ImportError:
        return False


from ._constants import DENOM_EPS, CORNER_QUANT_SCALE, VIS_BIN_FACTOR, BIT_SHIFTS

_DENOM_EPS = DENOM_EPS  # Alias for internal use
_CORNER_QUANT_SCALE = CORNER_QUANT_SCALE  # Alias for internal use
_VIS_BIN_FACTOR = VIS_BIN_FACTOR  # Alias for internal use
_BIT_SHIFTS = BIT_SHIFTS  # Alias for internal use

# Chunk size for SDF batch evaluation: prevents excessive memory usage per thread.
_SDF_CHUNK_SIZE = 50_000

# Edge table: for each of the 256 cube configurations, a 12-bit mask
# indicating which edges are intersected by the iso-surface.
_EDGE_TABLE: list[int] = [
    0x000, 0x109, 0x203, 0x30A, 0x406, 0x50F, 0x605, 0x70C,
    0x80C, 0x905, 0xA0F, 0xB06, 0xC0A, 0xD03, 0xE09, 0xF00,
    0x190, 0x099, 0x393, 0x29A, 0x596, 0x49F, 0x795, 0x69C,
    0x99C, 0x895, 0xB9F, 0xA96, 0xD9A, 0xC93, 0xF99, 0xE90,
    0x230, 0x339, 0x033, 0x13A, 0x636, 0x73F, 0x435, 0x53C,
    0xA3C, 0xB35, 0x83F, 0x936, 0xE3A, 0xF33, 0xC39, 0xD30,
    0x3A0, 0x2A9, 0x1A3, 0x0AA, 0x7A6, 0x6AF, 0x5A5, 0x4AC,
    0xBAC, 0xAA5, 0x9AF, 0x8A6, 0xFAA, 0xEA3, 0xDA9, 0xCA0,
    0x460, 0x569, 0x663, 0x76A, 0x066, 0x16F, 0x265, 0x36C,
    0xC6C, 0xD65, 0xE6F, 0xF66, 0x86A, 0x963, 0xA69, 0xB60,
    0x5F0, 0x4F9, 0x7F3, 0x6FA, 0x1F6, 0x0FF, 0x3F5, 0x2FC,
    0xDFC, 0xCF5, 0xFFF, 0xEF6, 0x9FA, 0x8F3, 0xBF9, 0xAF0,
    0x650, 0x759, 0x453, 0x55A, 0x256, 0x35F, 0x055, 0x15C,
    0xE5C, 0xF55, 0xC5F, 0xD56, 0xA5A, 0xB53, 0x859, 0x950,
    0x7C0, 0x6C9, 0x5C3, 0x4CA, 0x3C6, 0x2CF, 0x1C5, 0x0CC,
    0xFCC, 0xEC5, 0xDCF, 0xCC6, 0xBCA, 0xAC3, 0x9C9, 0x8C0,
    0x8C0, 0x9C9, 0xAC3, 0xBCA, 0xCC6, 0xDCF, 0xEC5, 0xFCC,
    0x0CC, 0x1C5, 0x2CF, 0x3C6, 0x4CA, 0x5C3, 0x6C9, 0x7C0,
    0x950, 0x859, 0xB53, 0xA5A, 0xD56, 0xC5F, 0xF55, 0xE5C,
    0x15C, 0x055, 0x35F, 0x256, 0x55A, 0x453, 0x759, 0x650,
    0xAF0, 0xBF9, 0x8F3, 0x9FA, 0xEF6, 0xFFF, 0xCF5, 0xDFC,
    0x2FC, 0x3F5, 0x0FF, 0x1F6, 0x6FA, 0x7F3, 0x4F9, 0x5F0,
    0xB60, 0xA69, 0x963, 0x86A, 0xF66, 0xE6F, 0xD65, 0xC6C,
    0x36C, 0x265, 0x16F, 0x066, 0x76A, 0x663, 0x569, 0x460,
    0xCA0, 0xDA9, 0xEA3, 0xFAA, 0x8A6, 0x9AF, 0xAA5, 0xBAC,
    0x4AC, 0x5A5, 0x6AF, 0x7A6, 0x0AA, 0x1A3, 0x2A9, 0x3A0,
    0xD30, 0xC39, 0xF33, 0xE3A, 0x936, 0x83F, 0xB35, 0xA3C,
    0x53C, 0x435, 0x73F, 0x636, 0x13A, 0x033, 0x339, 0x230,
    0xE90, 0xF99, 0xC93, 0xD9A, 0xA96, 0xB9F, 0x895, 0x99C,
    0x69C, 0x795, 0x49F, 0x596, 0x29A, 0x393, 0x099, 0x190,
    0xF00, 0xE09, 0xD03, 0xC0A, 0xB06, 0xA0F, 0x905, 0x80C,
    0x70C, 0x605, 0x50F, 0x406, 0x30A, 0x203, 0x109, 0x000,
]

# Triangle table: for each cube configuration, up to 5 triangles encoded as edge indices.
_TRI_TABLE: list[list[int]] = [
    [-1], [0, 8, 3, -1], [0, 1, 9, -1], [1, 8, 3, 9, 8, 1, -1],
    [1, 2, 10, -1], [0, 8, 3, 1, 2, 10, -1], [9, 2, 10, 0, 2, 9, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1], [3, 11, 2, -1], [0, 11, 2, 8, 11, 0, -1],
    [1, 9, 0, 2, 3, 11, -1], [1, 11, 2, 1, 9, 11, 9, 8, 11, -1],
    [3, 10, 1, 11, 10, 3, -1], [0, 10, 1, 0, 8, 10, 8, 11, 10, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1], [9, 8, 10, 10, 8, 11, -1],
    [4, 7, 8, -1], [4, 3, 0, 7, 3, 4, -1], [0, 1, 9, 8, 4, 7, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1], [1, 2, 10, 8, 4, 7, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1], [9, 2, 10, 9, 0, 2, 8, 4, 7, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1], [8, 4, 7, 3, 11, 2, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1], [9, 0, 1, 8, 4, 7, 2, 3, 11, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1], [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1], [4, 7, 11, 4, 11, 9, 9, 11, 10, -1],
    [9, 5, 4, -1], [9, 5, 4, 0, 8, 3, -1], [0, 5, 4, 1, 5, 0, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1], [1, 2, 10, 9, 5, 4, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1], [5, 2, 10, 5, 4, 2, 4, 0, 2, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1], [9, 5, 4, 2, 3, 11, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1], [0, 5, 4, 0, 1, 5, 2, 3, 11, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1], [10, 3, 11, 10, 1, 3, 9, 5, 4, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1], [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1], [9, 7, 8, 5, 7, 9, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1], [0, 7, 8, 0, 1, 7, 1, 5, 7, -1],
    [1, 5, 3, 3, 5, 7, -1], [9, 7, 8, 9, 5, 7, 10, 1, 2, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1], [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1], [7, 9, 5, 7, 8, 9, 3, 11, 2, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1], [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1], [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1], [11, 10, 5, 7, 11, 5, -1],
    [10, 6, 5, -1], [0, 8, 3, 5, 10, 6, -1], [9, 0, 1, 5, 10, 6, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1], [1, 6, 5, 2, 6, 1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1], [9, 6, 5, 9, 0, 6, 0, 2, 6, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1], [2, 3, 11, 10, 6, 5, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1], [0, 1, 9, 2, 3, 11, 5, 10, 6, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1], [6, 3, 11, 6, 5, 3, 5, 1, 3, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1], [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1], [5, 10, 6, 4, 7, 8, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1], [1, 9, 0, 5, 10, 6, 8, 4, 7, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, -1], [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1], [7, 8, 0, 7, 0, 6, 6, 0, 2, -1],
    [7, 3, 2, 6, 7, 2, -1], [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1], [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1], [0, 9, 1, 11, 6, 7, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1], [7, 11, 6, -1], [7, 6, 11, -1],
    [3, 0, 8, 11, 7, 6, -1], [0, 1, 9, 11, 7, 6, -1], [8, 1, 9, 8, 3, 1, 11, 7, 6, -1],
    [10, 1, 2, 6, 11, 7, -1], [1, 2, 10, 3, 0, 8, 6, 11, 7, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1], [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1],
    [7, 2, 3, 6, 2, 7, -1], [7, 0, 8, 7, 6, 0, 6, 2, 0, -1], [2, 7, 6, 2, 3, 7, 0, 1, 9, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1], [10, 7, 6, 10, 1, 7, 1, 3, 7, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1], [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1], [6, 8, 4, 11, 8, 6, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1], [8, 6, 11, 8, 4, 6, 9, 0, 1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1], [6, 8, 4, 6, 11, 8, 2, 10, 1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1], [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1], [8, 2, 3, 8, 4, 2, 4, 6, 2, -1],
    [0, 4, 2, 4, 6, 2, -1], [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1], [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1], [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1], [4, 9, 5, 7, 6, 11, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1], [5, 0, 1, 5, 4, 0, 7, 6, 11, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1], [9, 5, 4, 10, 1, 2, 7, 6, 11, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1], [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1], [7, 2, 3, 7, 6, 2, 5, 4, 9, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1], [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1], [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1], [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1], [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1], [6, 11, 3, 6, 3, 5, 5, 3, 1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1], [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1], [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1], [9, 5, 6, 9, 6, 0, 0, 6, 2, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1], [1, 5, 6, 2, 1, 6, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1], [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1],
    [0, 3, 8, 5, 6, 10, -1], [10, 5, 6, -1], [11, 5, 10, 7, 5, 11, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1], [5, 11, 7, 5, 10, 11, 1, 9, 0, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1], [11, 1, 2, 11, 7, 1, 7, 5, 1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1], [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1], [2, 5, 10, 2, 3, 5, 3, 7, 5, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1], [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1], [1, 3, 5, 3, 7, 5, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1], [9, 0, 3, 9, 3, 5, 5, 3, 7, -1],
    [9, 8, 7, 5, 9, 7, -1], [5, 8, 4, 5, 10, 8, 10, 11, 8, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1], [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1], [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1], [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1], [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1], [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1], [8, 4, 5, 8, 5, 3, 3, 5, 1, -1],
    [0, 4, 5, 1, 0, 5, -1], [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1], [9, 4, 5, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1], [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1], [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1], [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1], [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1], [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1], [1, 10, 2, 8, 7, 4, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1], [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1],
    [4, 0, 3, 7, 4, 3, -1], [4, 8, 7, -1], [9, 10, 8, 10, 11, 8, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1], [0, 1, 10, 0, 10, 8, 8, 10, 11, -1],
    [3, 1, 10, 11, 3, 10, -1], [1, 2, 11, 1, 11, 9, 9, 11, 8, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1], [0, 2, 11, 8, 0, 11, -1],
    [3, 2, 11, -1], [2, 3, 8, 2, 8, 10, 10, 8, 9, -1], [9, 10, 2, 0, 9, 2, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1], [1, 10, 2, -1], [1, 3, 8, 9, 1, 8, -1],
    [0, 9, 1, -1], [0, 3, 8, -1], [-1],
]

# Maximum number of SDF worker threads for multi-kernel evaluation.
_MLX_SDF_THREADS = 4

# Edge-to-vertex mapping: each edge connects two of the 8 cube corners.
_EDGE_VERTICES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# Cube corner offsets (x, y, z) in [0, 1].
_CORNER_OFFSETS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]


class MLXOcMesher:
    """Octree mesher using MLX for Apple Silicon GPU acceleration.

    This class provides the same interface as :class:`OcMesher` but uses
    Apple's MLX framework for GPU-accelerated computation on Apple Silicon.

    Args:
        cameras: Camera tuple ``(cam_poses, Ks, Hs, Ws)``.
        bounds: Scene bounds ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
        device: Device to use (``"mlx"`` or ``"cpu"``). Defaults to ``"mlx"`` if available.
        **kwargs: Additional parameters forwarded to parent validation.

    Raises:
        ImportError: If MLX is not installed.
        ValueError: If device is not ``"mlx"`` or ``"cpu"``.
    """

    __version__ = "0.1.0"

    # Class-level cache for MLX lookup tables (shared across instances on same device)
    _mc_cache: ClassVar[dict[str, dict]] = {}

    def __init__(
        self,
        cameras: CamerasTuple,
        bounds: BoundsLike,
        pixels_per_cube: int = 8,
        inv_scale: int = 10,
        min_dist: int = 1,
        memory_limit_mb: int = 1000,
        bisection_iters: int = 15,
        bisection_tol: float = 0.0,
        enclosed: bool = True,
        simplify_occluded: bool = True,
        visible_relax_iter: int = 2,
        coarse_count: int = 500000,
        device: str | None = None,
        n_sdf_workers: int = _MLX_SDF_THREADS,
        **kwargs: Any,
    ) -> None:
        if not _mlx_available():
            msg = "MLX is not installed. Install with: pip install mlx"
            raise ImportError(msg)

        import mlx.core as mx

        self._mx = mx

        # Validate inputs
        cam_poses, ks, hs, ws = validate_cameras(cameras)
        bounds = validate_bounds(bounds)
        validate_mesher_params(
            pixels_per_cube=pixels_per_cube,
            inv_scale=inv_scale,
            min_dist=min_dist,
            memory_limit_mb=memory_limit_mb,
            bisection_iters=bisection_iters,
            visible_relax_iter=visible_relax_iter,
            coarse_count=coarse_count,
            bisection_tol=bisection_tol,
        )

        self.cameras = (cam_poses, ks, hs, ws)
        self.bounds = bounds

        # Determine device
        if device is None:
            self.device = "mlx"
        elif device in ("mlx", "cpu"):
            self.device = device
        else:
            msg = f"MLXOcMesher only supports 'mlx' or 'cpu' devices, got {device!r}"
            raise ValueError(msg)

        # Meshing parameters
        self.pixels_per_cube = pixels_per_cube
        self.inv_scale = inv_scale
        self.min_dist = float(min_dist)
        self.memory_limit_mb = memory_limit_mb
        self.bisection_iters = bisection_iters
        self.bisection_tol = float(bisection_tol)
        self.enclosed = enclosed
        self.simplify_occluded = simplify_occluded
        self.visible_relax_iter = visible_relax_iter
        self.coarse_count = coarse_count
        self.n_sdf_workers = max(1, n_sdf_workers)

        # Convert camera data to MLX arrays
        self._setup_camera_tensors()

        # Scene bounds
        bt = np.array(bounds, dtype=np.float64)
        self.bounds_min = bt[0::2]  # (3,)
        self.bounds_max = bt[1::2]  # (3,)
        # Pre-compute float32 bounds for SDF evaluation (avoids per-call astype)
        self._bounds_min_f32 = self.bounds_min.astype(np.float32)  # (3,)
        self._bounds_max_f32 = self.bounds_max.astype(np.float32)  # (3,)
        self.center = (self.bounds_min + self.bounds_max) / 2
        extent = self.bounds_max - self.bounds_min
        self.size = float(extent.max() * 1.1)

        # Pre-compute the world-space origin (already float64)
        self._origin = (self.center - self.size / 2).astype(np.float64)  # (3,)

        # Pre-compute exp2(scales) for all possible levels (0-30) to avoid per-call exp2
        all_levels = np.arange(31, dtype=np.float64)
        self._exp2_scales = self.size / np.exp2(all_levels)  # (31,) indexed by level

        # Pre-compute octree child offsets
        self._child_offsets = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.int64)

        self._corner_offsets = np.array(_CORNER_OFFSETS, dtype=np.int64)
        self._edge_vertices = np.array(_EDGE_VERTICES, dtype=np.int64)

        # Pre-compute visibility relaxation neighbor offsets (once, not per-call)
        rl = self.visible_relax_iter
        dx_range = np.arange(-rl, rl + 1, dtype=np.int32)
        dy_range = np.arange(-rl, rl + 1, dtype=np.int32)
        grid_dx, grid_dy = np.meshgrid(dx_range, dy_range, indexing="ij")
        self._relax_dx = grid_dx.reshape(-1)
        self._relax_dy = grid_dy.reshape(-1)

        # Pre-cache marching cubes tables for CPU numpy path
        self._mc_edge_table = np.array(_EDGE_TABLE, dtype=np.int32)
        self._mc_bit_shifts = np.array(_BIT_SHIFTS, dtype=np.int32)
        max_tri_entries = 16
        self._mc_tri_table = np.full((256, max_tri_entries), -1, dtype=np.int16)
        for i, row in enumerate(_TRI_TABLE):
            self._mc_tri_table[i, : len(row)] = row

        # Pre-compute hash primes for vertex deduplication
        self._dedup_prime_x = 1000000007
        self._dedup_prime_y = 1000000009
        self._dedup_prime_z = 1000000021

        # Cache for marching cubes lookup tables (MLX GPU arrays)
        self._setup_mc_cache()

    def _setup_camera_tensors(self) -> None:
        """Convert and upload camera data to MLX device."""
        import mlx.core as mx

        # Stack camera poses: (C, 4, 4)
        cam_poses = np.stack([p.astype(np.float32) for p in self.cameras[0]], axis=0)
        self._cam_poses_np = cam_poses  # Keep numpy for projection

        # Stack intrinsics: (C, 3, 3)
        Ks = np.stack([k.astype(np.float32) for k in self.cameras[1]], axis=0)
        self._Ks_np = Ks

        # Heights and widths: (C,)
        self._Hs_np = np.array(self.cameras[2], dtype=np.int32)
        self._Ws_np = np.array(self.cameras[3], dtype=np.int32)
        self._Hs_f32 = self._Hs_np.astype(np.float32)  # Pre-cast for visibility filter
        self._Ws_f32 = self._Ws_np.astype(np.float32)

        self.n_cameras = len(self.cameras[0])

        # Compute inverse poses as numpy arrays for projection
        inv_poses_np = np.stack([
            np.linalg.inv(cam_poses[i])[:3, :4] for i in range(self.n_cameras)
        ]).astype(np.float32)
        self._inv_poses_np = inv_poses_np

        # Pre-compute per-camera pixel angular size (numpy)
        fx_all = Ks[:, 0, 0]  # (C,)
        w_all = self._Ws_f32
        self._pix_ang = np.arctan(w_all / 2 / fx_all) * 2 / w_all  # (C,)

        # Pre-compute projection angular threshold
        self._pix_ang_ppc = (self._pix_ang * self.pixels_per_cube)[:, np.newaxis]  # (C, 1)
        self._inv_pix_ang_ppc = 1.0 / self._pix_ang_ppc  # (C, 1)

        # Pre-split rotation/translation components for projection
        self._inv_pose_R = inv_poses_np[:, :, :3]  # (C, 3, 3)
        self._inv_pose_t = inv_poses_np[:, :, 3]  # (C, 3)

        # Visibility bin dimensions
        vis_hb = [max(1, int(h / _VIS_BIN_FACTOR)) for h in self.cameras[2]]
        vis_wb = [max(1, int(w / _VIS_BIN_FACTOR)) for w in self.cameras[3]]
        self._vis_hb = np.array(vis_hb, dtype=np.int32)
        self._vis_wb = np.array(vis_wb, dtype=np.int32)
        self._vis_inv_factor = float(_VIS_BIN_FACTOR)
        self._depth_inf = np.float32(np.inf)  # Pre-compute for depth buffer
        self._depth_safe_eps = np.float32(1e-10)  # Pre-compute for depth division

    def _setup_mc_cache(self) -> None:
        """Initialize marching cubes lookup tables on device."""
        import mlx.core as mx
        from mlx.core import array as mx_array

        if self.device not in self._mc_cache:
            # Edge table
            edge_table_np = np.array(_EDGE_TABLE, dtype=np.int32)
            edge_table_t = mx.array(edge_table_np)

            # Triangle table (256 x 16, padded with -1)
            max_tri_entries = 16
            tri_table_np = np.full((256, max_tri_entries), -1, dtype=np.int16)
            for i, row in enumerate(_TRI_TABLE):
                tri_table_np[i, : len(row)] = row
            tri_table_t = mx.array(tri_table_np)

            # Bit shifts
            bit_shifts = mx.array(_BIT_SHIFTS, dtype=mx.int32)

            self._mc_cache[self.device] = {
                "edge_table": edge_table_t,
                "tri_table": tri_table_t,
                "max_tri_entries": mx.array(max_tri_entries),
                "bit_shifts": bit_shifts,
            }

    def close(self) -> "MLXOcMesher":
        """Clean up resources (no-op for MLX backend). Returns self for chaining."""
        return self

    def __call__(self, kernels: KernelSequence) -> MeshResult:
        """Extract meshes from SDF kernels.

        Args:
            kernels: Sequence of SDF kernel functions.

        Returns:
            Tuple of ``(meshes, in_view_tags)``.
        """
        validate_kernels(kernels)
        with Timer("MLXOcMesher: total"):
            return self._extract_meshes(list(kernels))

    def _extract_meshes(self, kernels: list[Any]) -> MeshResult:
        """Main mesh extraction pipeline using MLX + CPU hybrid acceleration."""
        n_elements = len(kernels)

        # 1. Build coarse octree
        with Timer("MLX coarse octree"):
            coords, levels = self._build_coarse_octree()
            logger.info("coarse cubes: %d", len(coords))

        # 2. Detect surface cubes
        with Timer("MLX find surface"):
            surface_mask, corner_sdf = self._find_surface_cubes(kernels, coords, levels)
            s_coords = coords[surface_mask]
            s_levels = levels[surface_mask]
            s_corner_sdf = corner_sdf[surface_mask] if corner_sdf is not None else None
            logger.info("surface cubes: %d", len(s_coords))

        # 3. Refine surface cubes
        with Timer("MLX refine surface"):
            refined_coords, refined_levels, refined_corner_sdf = self._refine_surface_octree(
                kernels, s_coords, s_levels, corner_sdf=s_corner_sdf
            )
            s_coords, s_levels = refined_coords, refined_levels
            s_corner_sdf = refined_corner_sdf
            logger.info("refined surface cubes: %d", len(s_coords))

        # 4. Visibility filter
        with Timer("MLX visibility filter"):
            positions = self._cube_centers(s_coords, s_levels)
            vis_mask = self._visibility_filter(positions)
            vis_coords = s_coords[vis_mask]
            vis_levels = s_levels[vis_mask]
            occ_coords = s_coords[~vis_mask]
            occ_levels = s_levels[~vis_mask]
            logger.info("visible: %d, occluded: %d", len(vis_coords), len(occ_coords))

        # 5. Per-element mesh construction
        with Timer("MLX construct mesh"):
            meshes: list[trimesh.Trimesh] = []
            in_view_tags: list[np.ndarray] = []
            all_coords = np.concatenate([vis_coords, occ_coords])
            all_levels = np.concatenate([vis_levels, occ_levels])
            n_visible = len(vis_coords)

            all_corner_sdf: np.ndarray | None = None
            if s_corner_sdf is not None:
                all_corner_sdf = np.concatenate([s_corner_sdf[vis_mask], s_corner_sdf[~vis_mask]])

            for e in range(n_elements):
                mesh, ivt = self._construct_element_mesh(
                    kernels[e: e + 1],
                    all_coords,
                    all_levels,
                    n_visible,
                    corner_sdf=all_corner_sdf,
                    element_idx=e,
                )
                meshes.append(mesh)
                in_view_tags.append(ivt)
                logger.info(
                    "element %d: %d vertices, %d faces",
                    e, mesh.vertices.shape[0], mesh.faces.shape[0]
                )

        return meshes, in_view_tags

    def _cube_scales(self, levels: np.ndarray) -> np.ndarray:
        """Compute world-space cube side lengths from octree levels."""
        return self._exp2_scales[levels.astype(np.int32)]

    def _cube_corner_positions(
        self, coords: np.ndarray, levels: np.ndarray, *, cube_scales: np.ndarray | None = None
    ) -> np.ndarray:
        """Integer octree coords -> world-space center positions."""
        if cube_scales is None:
            cube_scales = self._cube_scales(levels)
        # _origin is float64, coords is int, scale is float64 → result is float64, no astype needed
        return self._origin + cube_scales[:, np.newaxis] * (coords + 0.5)

    def _cube_corner_positions(self, coords: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """Return world positions of all 8 corners for each cube."""
        corner_coords = coords[:, np.newaxis, :] + self._corner_offsets[np.newaxis, :, :]
        scale = self._cube_scales(levels)[:, np.newaxis, np.newaxis]
        # _origin is float64, scale is float64, corner_coords is int → result is float64
        return self._origin + scale * corner_coords

    def _cube_corner_positions_f64(self, coords: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """Like _cube_corner_positions but always in float64 on CPU."""
        corner_coords = coords[:, np.newaxis, :] + self._corner_offsets[np.newaxis, :, :]
        scale = self._cube_scales(levels)[:, np.newaxis, np.newaxis]
        # _origin is float64, scale is float64 → result is float64, no astype needed
        return self._origin + scale * corner_coords

    def _projected_sizes(
        self, positions: np.ndarray, levels: np.ndarray, *, cube_scales: np.ndarray | None = None
    ) -> np.ndarray:
        """Max projected pixel size across all cameras (vectorised)."""
        if cube_scales is None:
            cube_scales = self._cube_scales(levels)

        positions_f = positions.astype(np.float32)
        n_pos = positions_f.shape[0]

        # Pre-split rotation/translation from precomputed camera projection
        # cam_coords[c, n] = R[c] @ pos[n] + t[c]
        cam_coords = np.einsum("cij,nj->cni", self._inv_pose_R, positions_f) + self._inv_pose_t

        r = np.linalg.norm(cam_coords, axis=2).clip(min=self.min_dist)  # (C, N)
        proj = cube_scales[np.newaxis, :] * self._inv_pix_ang_ppc / r  # (C, N)
        return proj.max(axis=0)  # (N,)

    @staticmethod
    def _evaluate_sdf_kernel(kernel: Any, positions: np.ndarray) -> np.ndarray:
        """Evaluate a single SDF kernel at positions."""
        sdf = kernel(positions)
        if isinstance(sdf, dict):
            sdf = sdf.get("sdf", sdf.get("SDF", sdf))
        return np.asarray(sdf, dtype=np.float32).ravel()

    def _evaluate_sdf(self, kernels: list[Any], positions: np.ndarray) -> np.ndarray:
        """Evaluate SDF kernels at positions with thread-parallel chunking."""
        from concurrent.futures import ThreadPoolExecutor

        n_points = len(positions)
        n_kernels = len(kernels)

        if n_points == 0:
            return np.zeros((0, n_kernels), dtype=np.float32)

        chunk_size = max(1, min(n_points, _SDF_CHUNK_SIZE // n_kernels))
        results: list[np.ndarray] = []

        # For enclosed mode, mark out-of-bounds points
        if self.enclosed:
            b_min = self._bounds_min_f32
            b_max = self._bounds_max_f32
            out_bound = np.any(positions <= b_min, axis=1) | np.any(positions >= b_max, axis=1)

        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            chunk_pos = positions[start:end]

            with ThreadPoolExecutor(max_workers=self.n_sdf_workers) as executor:
                chunk_results = list(executor.map(
                    lambda k_p: MLXOcMesher._evaluate_sdf_kernel(k_p[0], k_p[1]),
                    [(k, chunk_pos) for k in kernels]
                ))

            sdf_chunk = np.stack(chunk_results, axis=1)  # (chunk, n_kernels)

            # For enclosed mode, set SDF to 1 for out-of-bounds points
            if self.enclosed:
                out_bound_chunk = out_bound[start:end]
                sdf_chunk[out_bound_chunk] = 1.0

            results.append(sdf_chunk)

        return np.concatenate(results, axis=0)  # (n_points, n_kernels)

    def _build_coarse_octree(self) -> tuple[np.ndarray, np.ndarray]:
        """Build coarse octree covering the scene using explicit expansion steps.

        This matches Torch's approach: iteratively expand cubes until no more
        need expansion or coarse_count budget is reached.
        """
        coords = np.array([[0, 0, 0]], dtype=np.int64)
        levels = np.array([0], dtype=np.int64)

        for _ in range(30):  # max 30 expansion steps, matching Torch
            n = len(coords)
            if n >= self.coarse_count:
                break

            cube_scales = self._cube_scales(levels)
            positions = self._cube_centers(coords, levels, cube_scales=cube_scales)
            proj = self._projected_sizes(positions, levels, cube_scales=cube_scales)
            to_expand = proj > self.inv_scale

            if not to_expand.any():
                break

            expand_idx = np.where(to_expand)[0]
            n_expand = len(expand_idx)
            n_keep = n - n_expand

            # Budget limiting: if expanding all would exceed coarse_count, prioritize highest proj
            remaining = self.coarse_count - n_keep
            if remaining < n_expand * 8:
                budget = max(1, remaining // 8)
                # Get indices of cubes with highest projected sizes
                sorted_idx = np.argsort(proj[expand_idx])[::-1]
                expand_idx = expand_idx[sorted_idx[:budget]]
                to_expand = np.zeros(n, dtype=bool)
                to_expand[expand_idx] = True
                n_expand = len(expand_idx)
                n_keep = n - n_expand

            keep_mask = ~to_expand
            # Generate 8 children for each cube to expand
            # Child coords are parent * 2 + offset (standard octree coordinate system)
            child_c = (coords[expand_idx, np.newaxis, :] * 2 + self._child_offsets[np.newaxis, :, :]).reshape(-1, 3)
            child_l = np.full(len(expand_idx) * 8, levels[expand_idx[0]] + 1, dtype=np.int64)

            coords = np.concatenate([coords[keep_mask], child_c])
            levels = np.concatenate([levels[keep_mask], child_l])

        return coords, levels

    def _find_surface_cubes(
        self, kernels: list[Any], coords: np.ndarray, levels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Detect surface cubes by evaluating SDF at corners.

        Uses straddle detection: a cube is a surface cube if some corners have
        SDF >= 0 and others have SDF < 0 (i.e., the zero isosurface crosses
        through the cube). This matches Torch's implementation.
        """
        n = len(coords)
        if n == 0:
            return np.array([], dtype=bool), None

        corner_pos = self._cube_corner_positions(coords, levels)  # (N, 8, 3)
        flat_pos = corner_pos.reshape(-1, 3)

        sdf_flat = self._evaluate_sdf(kernels, flat_pos)  # (N*8, K)
        corner_sdf = sdf_flat.reshape(n, 8, len(kernels))  # (N, 8, K)

        # Straddle detection: check if some corners are >= 0 and some are < 0
        signs = corner_sdf >= 0  # (N, 8, K)
        # A cube straddles the surface if it has both positive and negative corners
        has_pos = signs.any(axis=1)  # (N, K)
        has_neg = (~signs).any(axis=1)  # (N, K)
        surface_mask = has_pos & has_neg  # (N, K)

        # For enclosed mode, we only need to know if surface is inside
        # For non-enclosed, we also need to check that surface exits the cube
        if self.enclosed:
            # Just check straddle - any kernel with straddling is enough
            surface_mask = surface_mask.any(axis=1)  # (N,)
        else:
            sdf_min = corner_sdf.min(axis=1)  # (N, K)
            sdf_max = corner_sdf.max(axis=1)  # (N, K)
            surface_mask = surface_mask.any(axis=1) & (sdf_min.min(axis=1) < 0) & (sdf_max.max(axis=1) > 0)

        return surface_mask, corner_sdf

    def _refine_surface_octree(
        self,
        kernels: list[Any],
        coords: np.ndarray,
        levels: np.ndarray,
        corner_sdf: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Refine surface cubes to achieve target resolution."""
        n = len(coords)
        if n == 0:
            return coords, levels, corner_sdf

        refined_coords = [coords[0:1]]
        refined_levels = [levels[0:1]]
        refined_corner_sdf = [corner_sdf[0:1]] if corner_sdf is not None else None

        for i in range(n - 1):
            c = coords[i:i+1]
            l = levels[i:i+1]
            csdf = corner_sdf[i:i+1] if corner_sdf is not None else None

            proj_size = self._projected_sizes(self._cube_centers(c, l), l)[0]

            if proj_size >= self.pixels_per_cube * 2 and l[0] < 20:
                children = c + self._child_offsets
                child_levels = np.full(8, l[0] + 1, dtype=np.int64)

                child_corners = self._cube_corner_positions(children, child_levels)
                flat_child_pos = child_corners.reshape(-1, 3)
                child_sdf = self._evaluate_sdf(kernels, flat_child_pos)
                child_sdf_reshaped = child_sdf.reshape(8, 8, len(kernels))

                child_min = child_sdf_reshaped.min(axis=1)
                if self.enclosed:
                    child_surface = child_min.min(axis=1) < 0
                else:
                    child_max = child_sdf_reshaped.max(axis=1)
                    child_surface = (child_min.min(axis=1) < 0) & (child_max.max(axis=1) > 0)

                for j in range(8):
                    if child_surface[j]:
                        refined_coords.append(children[j:j+1])
                        refined_levels.append(np.array([l[0] + 1]))
                        if refined_corner_sdf is not None and csdf is not None:
                            refined_corner_sdf.append(child_sdf_reshaped[j:j+1, :, :])
            else:
                refined_coords.append(c)
                refined_levels.append(l)
                if refined_corner_sdf is not None and csdf is not None:
                    refined_corner_sdf.append(csdf)

        return (
            np.concatenate(refined_coords, axis=0),
            np.concatenate(refined_levels, axis=0),
            np.concatenate(refined_corner_sdf, axis=0) if refined_corner_sdf is not None else None,
        )

    def _visibility_filter(self, positions: np.ndarray) -> np.ndarray:
        """Depth-buffer visibility filtering using neighbor relaxation.

        Matches Torch's _visibility_filter behavior with proper neighbor relaxation.
        A point is visible if it's in any camera's view AND its depth is <= the
        depth of any point in its neighboring bins (relaxation radius).
        """
        if not self.simplify_occluded:
            return np.ones(len(positions), dtype=bool)

        positions_f = positions.astype(np.float32)
        n_pos = len(positions)

        # Project all positions to camera coordinates
        cam_coords = np.einsum("cij,nj->cni", self._inv_pose_R, positions_f) + self._inv_pose_t
        depth = cam_coords[:, :, 2]  # (C, N)
        depth_safe = depth + self._depth_safe_eps

        # Pixel coordinates (C, N)
        px = cam_coords[:, :, 0] / depth_safe
        py = cam_coords[:, :, 1] / depth_safe

        # Per-camera in-view check
        h_all = self._Hs_f32  # (C,)
        w_all = self._Ws_f32  # (C,)
        in_view_all = (depth > 0) & (px >= 0) & (px < w_all[:, np.newaxis]) & (py >= 0) & (py < h_all[:, np.newaxis])  # (C, N)

        # Early exit if no points in any view
        if not in_view_all.any():
            return np.zeros(n_pos, dtype=bool)

        # Bin coordinates for all cameras at once
        bx_all = np.clip((px * self._vis_inv_factor).astype(np.int32), 0, None)  # (C, N)
        by_all = np.clip((py * self._vis_inv_factor).astype(np.int32), 0, None)  # (C, N)

        vis_mask = np.zeros(n_pos, dtype=bool)

        # Per-camera depth buffer and neighbor relaxation
        for c in range(self.n_cameras):
            hb = int(self._vis_hb[c])
            wb = int(self._vis_wb[c])
            buf_size = hb * wb

            # Clamp bin coords to valid range for this camera
            bx_c = np.minimum(bx_all[c], wb - 1)
            by_c = np.minimum(by_all[c], hb - 1)

            in_view = in_view_all[c]  # (N,)
            valid_mask = in_view & (depth[c] > 0)

            if not valid_mask.any():
                continue

            # Linear bin indices for this camera
            idx_c = bx_c * hb + by_c  # (N,)

            # Build depth buffer: for each bin, store minimum depth
            depth_buf = np.full(buf_size, self._depth_inf, dtype=np.float32)
            valid_idx = idx_c[valid_mask]
            valid_depth = depth[c, valid_mask]
            sort_order = np.argsort(valid_idx)
            sorted_idx = valid_idx[sort_order]
            sorted_depth = valid_depth[sort_order]
            # np.unique returns first occurrence of each unique value
            unique_bins, first_pos = np.unique(sorted_idx, return_index=True)
            depth_buf[unique_bins] = sorted_depth[first_pos]

            # Relaxation: for each valid point, check if depth <= any neighbor's depth
            nb_dx = self._relax_dx  # (R,)
            nb_dy = self._relax_dy  # (R,)

            # Compute neighbor bin indices for all relaxation offsets
            # nbx, nby: (N, R)
            nbx = np.clip(bx_c[:, np.newaxis] + nb_dx[np.newaxis, :], 0, wb - 1)
            nby = np.clip(by_c[:, np.newaxis] + nb_dy[np.newaxis, :], 0, hb - 1)
            nb_idx = nbx * hb + nby  # (N, R)

            # Gather neighbor depths: (N, R)
            nb_depths = depth_buf[nb_idx]

            # Point is near front surface if its depth <= any neighbor's depth
            # depth[c]: (N,), nb_depths: (N, R) -> (N, R) comparison
            near_front = depth[c, :, np.newaxis] <= nb_depths  # (N, 1) vs (N, R) -> (N, R)
            near_front = near_front.any(axis=1)  # (N,)

            # Visible if in view AND near front surface
            camera_visible = in_view & near_front
            vis_mask |= camera_visible

        return vis_mask

    def _marching_cubes(
        self,
        corners: np.ndarray,
        sdf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run marching cubes on the given cubes (vectorised numpy on CPU)."""
        n = sdf.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

        # CPU-based marching cubes with numpy (using pre-cached tables)
        neg_mask = (sdf < 0).astype(np.int32)  # (N, 8)
        cube_idx = (neg_mask * self._mc_bit_shifts[np.newaxis, :]).sum(axis=1).astype(np.int32)  # (N,)

        edge_mask = self._mc_edge_table[cube_idx]
        active = edge_mask != 0

        if not active.any():
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

        active_idx = np.where(active)[0]
        a_sdf = sdf[active_idx]
        a_corners = corners[active_idx]
        a_cfg = cube_idx[active_idx]

        ev = self._edge_vertices
        s0 = a_sdf[:, ev[:, 0]]
        s1 = a_sdf[:, ev[:, 1]]
        denom = s0 - s1
        denom = np.where(np.abs(denom) < _DENOM_EPS, 0.5, s0 / denom)
        t = np.clip(denom, 0.0, 1.0)

        p0 = a_corners[:, ev[:, 0]]
        p1 = a_corners[:, ev[:, 1]]
        edge_positions = p0 + t[:, :, np.newaxis] * (p1 - p0)

        tri_entries = self._mc_tri_table[a_cfg]
        max_tris_per_cube = max_tri_entries // 3
        tri_edge_ids = tri_entries[:, : max_tris_per_cube * 3].reshape(-1, max_tris_per_cube, 3)
        tri_valid = tri_edge_ids[:, :, 0] >= 0

        valid_cube_idx, valid_tri_idx = np.where(tri_valid)
        if valid_cube_idx.size == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

        edge_ids = tri_edge_ids[valid_cube_idx, valid_tri_idx]
        v0 = edge_positions[valid_cube_idx, edge_ids[:, 0]]
        v1 = edge_positions[valid_cube_idx, edge_ids[:, 1]]
        v2 = edge_positions[valid_cube_idx, edge_ids[:, 2]]
        tri_verts = np.stack([v0, v1, v2], axis=1)
        verts_flat = tri_verts.reshape(-1, 3)

        quantized = (verts_flat * _CORNER_QUANT_SCALE).round().astype(np.int64)
        hash_vals = (
            quantized[:, 0] * self._dedup_prime_x +
            quantized[:, 1] * self._dedup_prime_y +
            quantized[:, 2] * self._dedup_prime_z
        )
        _, inverse = np.unique(hash_vals, return_inverse=True)
        n_unique = int(inverse.max()) + 1  # Count unique without storing them
        n_verts = len(inverse)

        rep_idx = np.zeros(n_unique, dtype=np.int64)
        rev_arange = np.arange(n_verts - 1, -1, -1)
        rep_idx[inverse[rev_arange]] = rev_arange

        dedup_verts = verts_flat[rep_idx]
        dedup_faces = inverse.reshape(-1, 3).astype(np.int32)

        return dedup_verts, dedup_faces

    def _construct_element_mesh(
        self,
        kernels: list[Any],
        coords: np.ndarray,
        levels: np.ndarray,
        n_visible: int,
        corner_sdf: np.ndarray | None = None,
        element_idx: int = 0,
    ) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Build a mesh for one SDF element using marching cubes."""
        n = len(coords)
        if n == 0:
            return trimesh.Trimesh(), np.zeros(0, dtype=bool)

        chunk_size = max(1, min(n, self.memory_limit_mb * 1000 // 64))
        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        vert_offset = 0

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            c_coords = coords[start:end]
            c_levels = levels[start:end]

            chunk_corners_f64 = self._cube_corner_positions_f64(c_coords, c_levels)

            if corner_sdf is not None:
                sdf_min = corner_sdf[start:end, :, element_idx]
            else:
                flat = chunk_corners_f64.reshape(-1, 3)
                sdf_all = self._evaluate_sdf(kernels, flat)
                sdf_min = sdf_all.min(axis=-1).reshape(end - start, 8)

            v, f = self._marching_cubes(chunk_corners_f64, sdf_min)
            if v.shape[0] > 0:
                f = f + vert_offset
                all_verts.append(v)
                all_faces.append(f)
                vert_offset += v.shape[0]

        if not all_verts:
            return trimesh.Trimesh(), np.zeros(0, dtype=bool)

        verts_np = np.concatenate(all_verts, axis=0)
        faces_np = np.concatenate(all_faces, axis=0)

        in_view = np.ones(verts_np.shape[0], dtype=bool)
        if n_visible < n and n_visible > 0:
            vis_pos = self._cube_centers(coords[:n_visible], levels[:n_visible])
            vis_min = vis_pos.min(axis=0)
            vis_max = vis_pos.max(axis=0)
            margin = self.size * 0.05
            in_view = np.all(verts_np >= vis_min - margin, axis=1) & np.all(
                verts_np <= vis_max + margin, axis=1
            )

        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        return mesh, in_view
