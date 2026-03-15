# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""PyTorch-based octree mesher with GPU acceleration.

Provides :class:`TorchOcMesher`, a drop-in replacement for :class:`OcMesher`
that replaces the C++ backend with pure PyTorch tensor operations.  All heavy
numerical work (camera projections, SDF evaluation batching, visibility
filtering) runs on the best available accelerator: CUDA, MPS (Apple Silicon),
or CPU.

Supported devices (auto-detected when ``device=None``):
- **CUDA** - NVIDIA GPUs via ``torch.cuda``
- **MPS** - Apple Silicon GPUs via ``torch.backends.mps``
- **CPU** - fallback for all platforms

Key optimisations over the reference C++ backend:
- Batched camera projection (all cameras processed simultaneously via bmm)
- Pre-allocated + cached lookup tables as persistent device tensors
- Fully vectorised marching cubes with fused configuration + edge interpolation
- Thread-pool parallel SDF evaluation chunks
- In-place tensor operations to reduce memory allocations
- Minimal CPU-GPU transfers (vertex dedup stays on GPU when possible)
- **float32 on CUDA/MPS** - reduces memory bandwidth by 2x and exploits
  Tensor Core throughput on Ampere/Ada/Hopper GPUs; MPS requires float32
- **cuDNN auto-tuning** on CUDA (``torch.backends.cudnn.benchmark = True``)
- **Optional ``torch.compile``** JIT compilation of hot-path methods
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import numpy as np
import torch
import trimesh

from .utils.timer import Timer

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    """Return True if MPS (Apple Silicon GPU) backend is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


# Epsilon for safe division in marching-cubes interpolation.
_DENOM_EPS = 1e-12

# Whether torch.compile is available (PyTorch ≥ 2.0).
_HAS_COMPILE = hasattr(torch, "compile")

# Scale factor for quantising corner positions to int64 before deduplication.
# 1e8 gives 10-nanometre resolution — fine enough to distinguish corners at
# the deepest practical octree level (2^20 subdivisions of a ~10 m scene
# yields ~10 µm cubes) while keeping quantised values within int64 range.
_CORNER_QUANT_SCALE = 1e8

# ---------------------------------------------------------------------------
# Marching-cubes lookup tables (classic Lorensen & Cline, 1987)
# ---------------------------------------------------------------------------
# Edge table: for each of the 256 cube configurations, a 12-bit mask
# indicating which edges are intersected by the iso-surface.
_EDGE_TABLE: list[int] = [
    0x000,
    0x109,
    0x203,
    0x30A,
    0x406,
    0x50F,
    0x605,
    0x70C,
    0x80C,
    0x905,
    0xA0F,
    0xB06,
    0xC0A,
    0xD03,
    0xE09,
    0xF00,
    0x190,
    0x099,
    0x393,
    0x29A,
    0x596,
    0x49F,
    0x795,
    0x69C,
    0x99C,
    0x895,
    0xB9F,
    0xA96,
    0xD9A,
    0xC93,
    0xF99,
    0xE90,
    0x230,
    0x339,
    0x033,
    0x13A,
    0x636,
    0x73F,
    0x435,
    0x53C,
    0xA3C,
    0xB35,
    0x83F,
    0x936,
    0xE3A,
    0xF33,
    0xC39,
    0xD30,
    0x3A0,
    0x2A9,
    0x1A3,
    0x0AA,
    0x7A6,
    0x6AF,
    0x5A5,
    0x4AC,
    0xBAC,
    0xAA5,
    0x9AF,
    0x8A6,
    0xFAA,
    0xEA3,
    0xDA9,
    0xCA0,
    0x460,
    0x569,
    0x663,
    0x76A,
    0x066,
    0x16F,
    0x265,
    0x36C,
    0xC6C,
    0xD65,
    0xE6F,
    0xF66,
    0x86A,
    0x963,
    0xA69,
    0xB60,
    0x5F0,
    0x4F9,
    0x7F3,
    0x6FA,
    0x1F6,
    0x0FF,
    0x3F5,
    0x2FC,
    0xDFC,
    0xCF5,
    0xFFF,
    0xEF6,
    0x9FA,
    0x8F3,
    0xBF9,
    0xAF0,
    0x650,
    0x759,
    0x453,
    0x55A,
    0x256,
    0x35F,
    0x055,
    0x15C,
    0xE5C,
    0xF55,
    0xC5F,
    0xD56,
    0xA5A,
    0xB53,
    0x859,
    0x950,
    0x7C0,
    0x6C9,
    0x5C3,
    0x4CA,
    0x3C6,
    0x2CF,
    0x1C5,
    0x0CC,
    0xFCC,
    0xEC5,
    0xDCF,
    0xCC6,
    0xBCA,
    0xAC3,
    0x9C9,
    0x8C0,
    0x8C0,
    0x9C9,
    0xAC3,
    0xBCA,
    0xCC6,
    0xDCF,
    0xEC5,
    0xFCC,
    0x0CC,
    0x1C5,
    0x2CF,
    0x3C6,
    0x4CA,
    0x5C3,
    0x6C9,
    0x7C0,
    0x950,
    0x859,
    0xB53,
    0xA5A,
    0xD56,
    0xC5F,
    0xF55,
    0xE5C,
    0x15C,
    0x055,
    0x35F,
    0x256,
    0x55A,
    0x453,
    0x759,
    0x650,
    0xAF0,
    0xBF9,
    0x8F3,
    0x9FA,
    0xEF6,
    0xFFF,
    0xCF5,
    0xDFC,
    0x2FC,
    0x3F5,
    0x0FF,
    0x1F6,
    0x6FA,
    0x7F3,
    0x4F9,
    0x5F0,
    0xB60,
    0xA69,
    0x963,
    0x86A,
    0xF66,
    0xE6F,
    0xD65,
    0xC6C,
    0x36C,
    0x265,
    0x16F,
    0x066,
    0x76A,
    0x663,
    0x569,
    0x460,
    0xCA0,
    0xDA9,
    0xEA3,
    0xFAA,
    0x8A6,
    0x9AF,
    0xAA5,
    0xBAC,
    0x4AC,
    0x5A5,
    0x6AF,
    0x7A6,
    0x0AA,
    0x1A3,
    0x2A9,
    0x3A0,
    0xD30,
    0xC39,
    0xF33,
    0xE3A,
    0x936,
    0x83F,
    0xB35,
    0xA3C,
    0x53C,
    0x435,
    0x73F,
    0x636,
    0x13A,
    0x033,
    0x339,
    0x230,
    0xE90,
    0xF99,
    0xC93,
    0xD9A,
    0xA96,
    0xB9F,
    0x895,
    0x99C,
    0x69C,
    0x795,
    0x49F,
    0x596,
    0x29A,
    0x393,
    0x099,
    0x190,
    0xF00,
    0xE09,
    0xD03,
    0xC0A,
    0xB06,
    0xA0F,
    0x905,
    0x80C,
    0x70C,
    0x605,
    0x50F,
    0x406,
    0x30A,
    0x203,
    0x109,
    0x000,
]

# Triangle table: for each cube configuration, up to 5 triangles encoded as
# edge indices.  -1 terminates the list.
_TRI_TABLE: list[list[int]] = [
    [-1],
    [0, 8, 3, -1],
    [0, 1, 9, -1],
    [1, 8, 3, 9, 8, 1, -1],
    [1, 2, 10, -1],
    [0, 8, 3, 1, 2, 10, -1],
    [9, 2, 10, 0, 2, 9, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1],
    [3, 11, 2, -1],
    [0, 11, 2, 8, 11, 0, -1],
    [1, 9, 0, 2, 3, 11, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1],
    [3, 10, 1, 11, 10, 3, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1],
    [9, 8, 10, 10, 8, 11, -1],
    [4, 7, 8, -1],
    [4, 3, 0, 7, 3, 4, -1],
    [0, 1, 9, 8, 4, 7, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1],
    [1, 2, 10, 8, 4, 7, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1],
    [8, 4, 7, 3, 11, 2, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1],
    [9, 5, 4, -1],
    [9, 5, 4, 0, 8, 3, -1],
    [0, 5, 4, 1, 5, 0, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1],
    [1, 2, 10, 9, 5, 4, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1],
    [9, 5, 4, 2, 3, 11, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1],
    [9, 7, 8, 5, 7, 9, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1],
    [1, 5, 3, 3, 5, 7, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1],
    [10, 6, 5, -1],
    [0, 8, 3, 5, 10, 6, -1],
    [9, 0, 1, 5, 10, 6, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1],
    [1, 6, 5, 2, 6, 1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1],
    [2, 3, 11, 10, 6, 5, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1],
    [5, 10, 6, 4, 7, 8, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1],
    [10, 4, 9, 6, 4, 10, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1],
    [0, 2, 4, 4, 2, 6, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1],
    [6, 4, 8, 11, 6, 8, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1],
    [7, 3, 2, 6, 7, 2, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1],
    [7, 11, 6, -1],
    [7, 6, 11, -1],
    [3, 0, 8, 11, 7, 6, -1],
    [0, 1, 9, 11, 7, 6, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1],
    [10, 1, 2, 6, 11, 7, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1],
    [7, 2, 3, 6, 2, 7, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1],
    [6, 8, 4, 11, 8, 6, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1],
    [0, 4, 2, 4, 6, 2, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1],
    [4, 9, 5, 7, 6, 11, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1],
    [0, 3, 8, 5, 6, 10, -1],
    [10, 5, 6, -1],
    [11, 5, 10, 7, 5, 11, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1],
    [9, 8, 7, 5, 9, 7, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1],
    [0, 4, 5, 1, 0, 5, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1],
    [9, 4, 5, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1],
    [4, 0, 3, 7, 4, 3, -1],
    [4, 8, 7, -1],
    [9, 10, 8, 10, 11, 8, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1],
    [3, 1, 10, 11, 3, 10, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1],
    [0, 2, 11, 8, 0, 11, -1],
    [3, 2, 11, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1],
    [9, 10, 2, 0, 9, 2, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1],
    [1, 10, 2, -1],
    [1, 3, 8, 9, 1, 8, -1],
    [0, 9, 1, -1],
    [0, 3, 8, -1],
    [-1],
]

# Edge-to-vertex mapping: each edge connects two of the 8 cube corners.
_EDGE_VERTICES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

# Cube corner offsets (x, y, z) in [0, 1].
_CORNER_OFFSETS = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
]


# Maximum number of SDF evaluation threads for parallel chunk processing.
_MAX_SDF_WORKERS = 4


class TorchOcMesher:
    """Octree-based mesher that extracts surfaces from SDF kernels using PyTorch.

    Drop-in replacement for :class:`OcMesher` - same constructor signature and
    ``__call__`` contract, but all heavy computation is performed with PyTorch
    tensors on the best available accelerator (CUDA, MPS, or CPU).

    When ``device=None`` (the default), the device is auto-detected in priority
    order: CUDA → MPS → CPU.  Pass ``device="mps"`` to explicitly select
    Apple Silicon GPU acceleration.

    Optimisations:
    - Batched camera projection via ``torch.bmm`` (all cameras in one pass)
    - Pre-allocated octree child-offsets tensor (no per-iteration allocation)
    - Cached marching-cubes lookup tables as device tensors
    - Fused cube-configuration + edge interpolation
    - Thread-pool parallel SDF evaluation over point chunks
    - In-place tensor ops to minimise memory allocations
    - **float32** arithmetic on CUDA/MPS (2-4x faster than float64 on GPU);
      float64 retained on CPU for numerical precision
    - cuDNN auto-tuning enabled automatically on CUDA
    - Optional ``torch.compile`` JIT via ``use_compile=True``
    """

    # Pre-computed marching-cubes lookup tables (shared across all instances).
    # Lazily initialised on first use per device.
    _mc_cache: ClassVar[dict[torch.device, dict[str, torch.Tensor]]] = {}

    def __init__(
        self,
        cameras,
        bounds,
        pixels_per_cube=8,
        inv_scale=10,
        min_dist=1,
        memory_limit_mb=1000,
        bisection_iters=15,
        enclosed=True,
        simplify_occluded=True,
        visible_relax_iter=2,
        coarse_count=500000,
        device=None,
        n_sdf_workers=_MAX_SDF_WORKERS,
        use_compile=False,
    ):
        """Initialise the mesher with camera intrinsics and bounds.

        Args:
            cameras: ``(cam_poses, Ks, Hs, Ws)`` camera tuple.
            bounds: flat 6-tuple ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
            pixels_per_cube: target projected cube size in pixels.
            inv_scale: octree refinement threshold.
            min_dist: minimum camera distance for projection.
            memory_limit_mb: chunk memory budget for mesh construction.
            bisection_iters: bisection iterations for surface refinement.
            enclosed: enforce SDF=1 outside scene bounds.
            simplify_occluded: depth-buffer occlusion culling.
            visible_relax_iter: neighbour relaxation radius for visibility.
            coarse_count: target coarse octree leaf count.
            device: ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` (auto).
            n_sdf_workers: thread-pool size for SDF evaluation.
            use_compile: if ``True`` and PyTorch ≥ 2.0, wrap hot-path methods
                with :func:`torch.compile` for JIT optimisation.  Incurs a
                one-time compilation cost on the first call; recommended when
                the mesher is called many times (e.g., in a training loop).
        """
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif _mps_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Select compute dtype -----------------------------------------------
        # float32 is ~4x faster than float64 on CUDA (Tensor Cores) and is the
        # only floating-point dtype supported by MPS.  CPU keeps float64 for
        # numerical precision in downstream processing.
        self._fdtype: torch.dtype = torch.float64 if self.device.type == "cpu" else torch.float32

        # Enable cuDNN auto-tuning for CUDA devices --------------------------
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        cam_poses, Ks, Hs, Ws = cameras
        self.n_cameras = len(cam_poses)

        # Pack camera data as batched tensors for bmm projection -------------
        # Vectorised: build numpy arrays first, transfer once to device.
        # Cast to _fdtype: float32 on CUDA/MPS for speed, float64 on CPU.
        inv_poses_np = np.stack([np.linalg.inv(cam_poses[i])[:3, :4] for i in range(self.n_cameras)]).astype(np.float64)
        intrinsics_np = np.stack([np.asarray(Ks[i], dtype=np.float64) for i in range(self.n_cameras)])
        # Tuples are immutable and have lower overhead than lists; camera
        # dimensions never change after initialisation.
        self.cam_heights: tuple[int, ...] = tuple(int(h) for h in Hs)
        self.cam_widths: tuple[int, ...] = tuple(int(w) for w in Ws)
        self.cam_inv_poses = torch.from_numpy(inv_poses_np).to(dtype=self._fdtype, device=self.device)  # (C, 3, 4)
        self.cam_intrinsics = torch.from_numpy(intrinsics_np).to(dtype=self._fdtype, device=self.device)  # (C, 3, 3)

        # Scene bounds -------------------------------------------------------
        self.bounds = bounds
        bt = torch.tensor(bounds, dtype=self._fdtype, device=self.device)
        self.bounds_min = bt[0::2]  # (3,)
        self.bounds_max = bt[1::2]  # (3,)
        self.center = (self.bounds_min + self.bounds_max) / 2
        extent = self.bounds_max - self.bounds_min
        self.size = float(extent.max().item() * 1.1)

        # Pre-compute the world-space origin used in coordinate conversion.
        # _cube_centers and _cube_corner_positions both need (center - size/2);
        # computing it once here avoids a subtraction on every call.
        self._origin = self.center - self.size / 2  # (3,)

        # Meshing parameters -------------------------------------------------
        self.pixels_per_cube = pixels_per_cube
        self.inv_scale = inv_scale
        self.min_dist = min_dist
        self.memory_limit_mb = memory_limit_mb
        self.bisection_iters = bisection_iters
        self.enclosed = enclosed
        self.simplify_occluded = simplify_occluded
        self.visible_relax_iter = visible_relax_iter
        self.coarse_count = coarse_count
        self.n_sdf_workers = max(1, n_sdf_workers)

        # Pre-compute per-camera pixel angular size (vectorised) --------------
        fx_all = self.cam_intrinsics[:, 0, 0]  # (C,) in _fdtype
        w_all = torch.tensor(self.cam_widths, dtype=self._fdtype, device=self.device)
        self._pix_ang = torch.atan(w_all / 2 / fx_all) * 2 / w_all  # (C,)

        # Pre-compute projection angular threshold for _projected_sizes.
        # This avoids recomputing the product every call.
        # Shape (C, 1): pre-expanded so _projected_sizes skips unsqueeze(1)
        # on every invocation (called 30+ times in _build_coarse_octree).
        self._pix_ang_ppc = (self._pix_ang * self.pixels_per_cube).unsqueeze(1)  # (C, 1)

        # Pre-compute reciprocal for _projected_sizes: replaces a division
        # with a multiplication on every call in the 30-iteration hot loop.
        self._inv_pix_ang_ppc = (1.0 / self._pix_ang_ppc)  # (C, 1)

        # Pre-compute combined K @ inv_pose for visibility filter -------------
        # This avoids two separate matmuls per camera in _visibility_filter.
        self._cam_proj = torch.bmm(
            self.cam_intrinsics,
            self.cam_inv_poses,
        )  # (C, 3, 4)

        # Pre-split rotation/translation components for pad-free projection.
        # This eliminates the homogeneous pad → bmm → permute sequence in
        # _projected_sizes and _visibility_filter, avoiding temporary
        # tensor allocations on every call.
        self._inv_pose_R = self.cam_inv_poses[:, :, :3].contiguous()  # (C, 3, 3)
        self._inv_pose_t = self.cam_inv_poses[:, :, 3].unsqueeze(1)  # (C, 1, 3)
        self._proj_R = self._cam_proj[:, :, :3].contiguous()  # (C, 3, 3)
        self._proj_t = self._cam_proj[:, :, 3].unsqueeze(1)  # (C, 1, 3)

        # Pre-compute per-camera bounds for in-view checks ------------------
        self._cam_heights_t = torch.tensor(self.cam_heights, dtype=torch.int64, device=self.device)
        self._cam_widths_t = torch.tensor(self.cam_widths, dtype=torch.int64, device=self.device)

        # Pre-compute bounds as numpy for fast out-of-bounds masking --------
        self._bounds_np = np.array(bounds, dtype=np.float64)
        # Pre-compute min/max vectors for vectorised bounds check
        self._bounds_min_np = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        self._bounds_max_np = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)

        # Pre-allocate reusable octree child offsets -------------------------
        self._child_offsets = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            dtype=torch.int64,
            device=self.device,
        )
        self._corner_offsets = torch.tensor(_CORNER_OFFSETS, dtype=torch.int64, device=self.device)
        self._edge_vertices = torch.tensor(_EDGE_VERTICES, dtype=torch.long, device=self.device)

        # Pre-compute visibility relaxation neighbor offsets ------------------
        rl = self.visible_relax_iter
        dx_range = torch.arange(-rl, rl + 1, dtype=torch.long, device=self.device)
        dy_range = torch.arange(-rl, rl + 1, dtype=torch.long, device=self.device)
        grid_dx, grid_dy = torch.meshgrid(dx_range, dy_range, indexing="ij")
        # Shape: (num_neighbors,) where num_neighbors = (2*rl+1)**2
        self._relax_dx = grid_dx.reshape(-1)
        self._relax_dy = grid_dy.reshape(-1)

        # Pre-allocate visibility depth buffer --------------------------------
        # The buffer is sized for the largest (reduced-resolution) depth image
        # across all cameras, avoiding repeated torch.full() allocation inside
        # the per-camera loop of _visibility_filter.  fill_() resets it cheaply.
        factor = 10.0
        # Pre-compute per-camera bin dimensions as tensors for batched
        # visibility filtering (avoids int()/max() inside the hot loop).
        vis_hb_list = [max(1, int(h / factor)) for h in self.cam_heights]
        vis_wb_list = [max(1, int(w / factor)) for w in self.cam_widths]
        self._vis_hb = torch.tensor(vis_hb_list, dtype=torch.int64, device=self.device)  # (C,)
        self._vis_wb = torch.tensor(vis_wb_list, dtype=torch.int64, device=self.device)  # (C,)
        self._vis_max_buf: int = max(
            hb * wb for hb, wb in zip(vis_hb_list, vis_wb_list, strict=True)
        )
        self._vis_inv_factor = 1.0 / factor
        # Batched depth buffer: (C, max_buf) — one row per camera, padded to
        # the largest camera's bin grid so scatter_reduce can be batched.
        self._depth_bufs = torch.empty(
            (self.n_cameras, self._vis_max_buf), dtype=self._fdtype, device=self.device,
        )

        # Ensure MC tables are cached for this device
        self._ensure_mc_cache()

        # Optionally JIT-compile hot-path methods for repeated use ----------
        if use_compile and _HAS_COMPILE:
            try:
                self._projected_sizes = torch.compile(  # type: ignore[method-assign]
                    self._projected_sizes,
                    dynamic=True,
                    fullgraph=False,
                )
                self._visibility_filter = torch.compile(  # type: ignore[method-assign]
                    self._visibility_filter,
                    dynamic=True,
                    fullgraph=False,
                )
                self._marching_cubes = torch.compile(  # type: ignore[method-assign]
                    self._marching_cubes,
                    dynamic=True,
                    fullgraph=False,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "torch.compile failed to initialise (backend unavailable); falling back to eager execution."
                )

    def _ensure_mc_cache(self):
        """Lazily build marching-cubes lookup tables on *self.device*."""
        if self.device in TorchOcMesher._mc_cache:
            return
        edge_table_t = torch.tensor(_EDGE_TABLE, dtype=torch.int32, device=self.device)
        max_tri_entries = max(len(row) for row in _TRI_TABLE)
        # int16 stores all entry values (range [-1, 11]) in half the memory of
        # int32, reducing GPU memory bandwidth when the table is indexed.
        tri_table_np = np.full((256, max_tri_entries), -1, dtype=np.int16)
        for i, row in enumerate(_TRI_TABLE):
            tri_table_np[i, : len(row)] = row
        tri_table_t = torch.from_numpy(tri_table_np).to(self.device)

        # Pre-compute bit-shift values for cube configuration
        bit_shifts = torch.tensor([1 << i for i in range(8)], dtype=torch.int32, device=self.device)

        TorchOcMesher._mc_cache[self.device] = {
            "edge_table": edge_table_t,
            "tri_table": tri_table_t,
            "max_tri_entries": torch.tensor(max_tri_entries),
            "bit_shifts": bit_shifts,
        }

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _cube_scales(self, levels: torch.Tensor) -> torch.Tensor:
        """Compute world-space cube side lengths from octree levels.

        This is the shared ``size / 2^level`` computation used by both
        :meth:`_cube_centers` and :meth:`_projected_sizes`.  Computing it
        once and passing the result to both methods avoids a redundant
        ``exp2`` call on every octree iteration (~33 iterations total).

        Returns:
            ``(N,)`` float tensor of cube side lengths.
        """
        return self.size / torch.exp2(levels.to(self._fdtype))

    @torch.no_grad()
    def _cube_centers(self, coords: torch.Tensor, levels: torch.Tensor, *, cube_scales: torch.Tensor | None = None) -> torch.Tensor:
        """Integer octree coords -> world-space center positions.

        Args:
            coords: ``(N, 3)`` int64 tensor of integer cube coordinates.
            levels:  ``(N,)``  int64 tensor of octree levels.
            cube_scales: optional pre-computed ``(N,)`` cube side lengths
                from :meth:`_cube_scales`.  When provided, the ``exp2``
                computation is skipped.

        Returns:
            ``(N, 3)`` float tensor (dtype matches ``self._fdtype``) world positions.
        """
        if cube_scales is None:
            cube_scales = self._cube_scales(levels)
        return self._origin.unsqueeze(0) + cube_scales.unsqueeze(1) * (coords.to(dtype=self._fdtype) + 0.5)

    @torch.no_grad()
    def _cube_corner_positions(self, coords: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """Return world positions of all 8 corners for each cube.

        Returns:
            ``(N, 8, 3)`` float tensor (dtype matches ``self._fdtype``).
        """
        corner_coords = coords.unsqueeze(1) + self._corner_offsets.unsqueeze(0)
        scale = self._cube_scales(levels).unsqueeze(1).unsqueeze(2)
        return self._origin.unsqueeze(0).unsqueeze(0) + scale * corner_coords.to(dtype=self._fdtype)

    # ------------------------------------------------------------------
    # Batched projection (all cameras at once)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _projected_sizes(self, positions: torch.Tensor, levels: torch.Tensor, *, cube_scales: torch.Tensor | None = None) -> torch.Tensor:
        """Max projected pixel size across *all* cameras (batched via bmm).

        Args:
            positions: ``(N, 3)`` world positions.
            levels:    ``(N,)``  octree levels.
            cube_scales: optional pre-computed ``(N,)`` cube side lengths
                from :meth:`_cube_scales`.  When provided, the ``exp2``
                computation is skipped (saves ~33 redundant calls per run).

        Returns:
            ``(N,)`` float tensor of projected sizes (dtype matches ``self._fdtype``).
        """
        if cube_scales is None:
            cube_scales = self._cube_scales(levels)

        # Pad-free projection: use pre-split rotation + translation to avoid
        # creating an (N, 4) padded tensor, transpose, expand, and permute.
        # einsum('cij,nj->cni', R, pos) gives (C, N, 3) directly.
        cam_coords = torch.einsum("cij,nj->cni", self._inv_pose_R, positions) + self._inv_pose_t

        # In-place clamp avoids allocating a new tensor; multiply by
        # pre-computed reciprocal replaces a division on every iteration.
        r = cam_coords.norm(dim=2).clamp_(min=self.min_dist)  # (C, N) in-place
        proj = cube_scales.unsqueeze(0) * self._inv_pix_ang_ppc / r  # (C, N)
        return proj.max(dim=0).values  # (N,)

    # ------------------------------------------------------------------
    # Threaded SDF evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_sdf(
        self,
        kernels: list,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate SDF *kernels* at *positions* with thread-parallel chunking.

        Kernel functions receive :class:`numpy.ndarray` and return the same.
        Conversion to/from tensors is handled here.

        Optimisations over the naïve implementation:
        - **Single-chunk fast path**: when all points fit in one chunk (the
          common case for moderate-sized inputs), skip list construction,
          ThreadPoolExecutor overhead, and ``np.concatenate``.
        - **Vectorised bounds check**: 2 broadcast comparisons + 2
          ``np.any`` reductions replace 6 scalar column comparisons + 6 ORs.
        - **In-place output for single kernel**: ``sdf.astype`` directly into
          the result column instead of building a list and stacking.

        Returns:
            ``(N, len(kernels))`` float32 tensor on *self.device*.
        """
        n = positions.shape[0]
        n_kernels = len(kernels)
        if n == 0:
            return torch.zeros((0, n_kernels), dtype=torch.float32, device=self.device)

        xyz_np = positions.cpu().double().numpy()
        step = 2_000_000  # chunk size tuned for cache locality
        enclosed = self.enclosed
        _use_pinned = self.device.type == "cuda"
        # Vectorised bounds: pre-computed min/max arrays for broadcast compare.
        b_min = self._bounds_min_np
        b_max = self._bounds_max_np

        if n_kernels == 1:
            # --- Fast path for the common single-kernel case ---
            kernel = kernels[0]

            def _eval_chunk(chunk_np):
                sdf = kernel(chunk_np)
                if enclosed:
                    out_bound = np.any(chunk_np <= b_min, axis=1) | np.any(chunk_np >= b_max, axis=1)
                    sdf[out_bound] = 1
                return sdf.astype(np.float32).reshape(-1, 1)
        else:

            def _eval_chunk(chunk_np):
                if enclosed:
                    out_bound = np.any(chunk_np <= b_min, axis=1) | np.any(chunk_np >= b_max, axis=1)
                cols = []
                for kernel in kernels:
                    sdf = kernel(chunk_np)
                    if enclosed:
                        sdf[out_bound] = 1
                    cols.append(sdf)
                return np.stack(cols, axis=-1).astype(np.float32)

        # Single-chunk fast path: skip list/pool/concat overhead.
        if n <= step:
            result_np = _eval_chunk(xyz_np)
        else:
            chunks = [xyz_np[i : i + step] for i in range(0, n, step)]
            if self.n_sdf_workers > 1:
                with ThreadPoolExecutor(max_workers=min(self.n_sdf_workers, len(chunks))) as pool:
                    parts = list(pool.map(_eval_chunk, chunks))
            else:
                parts = [_eval_chunk(c) for c in chunks]
            result_np = np.concatenate(parts, axis=0)

        # Use pinned memory for CUDA transfers to overlap copy with compute
        if _use_pinned:
            result_t = torch.from_numpy(result_np).pin_memory()
            return result_t.to(self.device, non_blocking=True)
        return torch.from_numpy(result_np).to(self.device)

    # ------------------------------------------------------------------
    # Octree construction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_coarse_octree(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the coarse adaptive octree driven by camera projections.

        Returns:
            ``(coords, levels)`` - leaf cubes of the coarse octree.
        """
        coords = torch.zeros((1, 3), dtype=torch.int64, device=self.device)
        levels = torch.zeros(1, dtype=torch.int64, device=self.device)

        for _ in range(30):
            n = len(coords)
            if n >= self.coarse_count:
                break
            # Compute scale once and share between centers and projection.
            cube_scales = self._cube_scales(levels)
            positions = self._cube_centers(coords, levels, cube_scales=cube_scales)
            proj = self._projected_sizes(positions, levels, cube_scales=cube_scales)
            to_expand = proj > self.inv_scale
            if not to_expand.any():
                break

            expand_idx = torch.where(to_expand)[0]
            n_expand = len(expand_idx)
            n_keep = n - n_expand

            # Budget-limit expansion
            remaining = self.coarse_count - n_keep
            if remaining < n_expand * 8:
                budget = max(1, remaining // 8)
                _, top_k = proj[expand_idx].topk(budget)
                expand_idx = expand_idx[top_k]
                to_expand = torch.zeros(n, dtype=torch.bool, device=self.device)
                to_expand[expand_idx] = True

            keep_mask = ~to_expand
            child_c = coords[expand_idx].unsqueeze(1) * 2 + self._child_offsets.unsqueeze(0)
            child_l = (levels[expand_idx] + 1).unsqueeze(1).expand(-1, 8)

            coords = torch.cat([coords[keep_mask], child_c.reshape(-1, 3)])
            levels = torch.cat([levels[keep_mask], child_l.reshape(-1)])
        return coords, levels

    # ------------------------------------------------------------------
    # Surface detection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _find_surface_cubes(
        self,
        kernels: list,
        coords: torch.Tensor,
        levels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Identify cubes that straddle the zero iso-surface.

        Adjacent octree cubes share corners, so we deduplicate corner
        positions before SDF evaluation and scatter results back.  This
        reduces SDF calls by 2-4x for typical octrees, dramatically
        cutting the dominant pipeline cost.

        Returns:
            ``(mask, corner_sdf)`` where *mask* is ``(N,)`` bool and
            *corner_sdf* is ``(N, 8, K)`` float32 tensor.
        """
        corners = self._cube_corner_positions(coords, levels)
        n = corners.shape[0]
        flat = corners.reshape(-1, 3)

        # --- shared-corner deduplication via hash-sort + exact compare ---
        # Sort corners by a 1D prime hash (fast radix sort on a single int64),
        # then detect unique rows by comparing adjacent quantised triples.
        # This is collision-free (exact row comparison) and significantly
        # faster than ``torch.unique(dim=0)`` which performs an O(N log N)
        # lexicographic sort with 3-column comparisons per step.
        quantized = (flat * _CORNER_QUANT_SCALE).round().long()
        hash_vals = (
            quantized[:, 0] * 1000000007
            + quantized[:, 1] * 1000000009
            + quantized[:, 2] * 1000000021
        )
        sort_idx = torch.argsort(hash_vals)
        sorted_q = quantized[sort_idx]

        # Consecutive rows that differ mark new unique groups.
        diff = (sorted_q[1:] != sorted_q[:-1]).any(dim=1)
        group_starts = torch.cat([
            torch.ones(1, device=flat.device, dtype=torch.bool), diff,
        ])
        group_ids = group_starts.cumsum(0) - 1  # 0-based unique-group ID

        # Map back to original (unsorted) order.
        inverse = torch.empty_like(group_ids)
        inverse[sort_idx] = group_ids
        # Pick one representative position per unique group.  The first
        # element in each sorted group (where group_starts is True) is the
        # canonical representative — deterministic and consistent with the
        # dedup strategy used in _marching_cubes.
        rep_sorted_idx = torch.where(group_starts)[0]  # indices into sort_idx
        rep_idx = sort_idx[rep_sorted_idx]  # map back to original flat indices
        unique_pts = flat[rep_idx]

        # Evaluate SDF only at unique positions, then scatter back.
        unique_sdf = self._evaluate_sdf(kernels, unique_pts)  # (U, K)
        sdf = unique_sdf[inverse]  # (N*8, K)

        sdf = sdf.reshape(n, 8, -1)
        sdf_min = sdf.min(dim=-1).values  # (N, 8)
        signs = sdf_min >= 0
        mask = signs.any(dim=1) & (~signs).any(dim=1)
        return mask, sdf

    # ------------------------------------------------------------------
    # Adaptive surface refinement
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _refine_surface_octree(
        self,
        kernels: list,
        coords: torch.Tensor,
        levels: torch.Tensor,
        corner_sdf: torch.Tensor | None = None,
        max_iters: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Iteratively subdivide surface cubes that project large on screen.

        Uses a conservative budget to avoid over-refinement.

        When *corner_sdf* is provided, kept (non-expanded) cubes reuse their
        cached SDF values, so only newly-created children are evaluated — this
        avoids ``n_keep * 8`` redundant SDF queries per iteration.

        Args:
            kernels: SDF kernel list.
            coords: ``(N, 3)`` int64 octree coordinates.
            levels: ``(N,)`` int64 octree levels.
            corner_sdf: ``(N, 8, K)`` float32 cached SDF values at cube
                corners from a prior :meth:`_find_surface_cubes` call.  When
                provided, the cached values are carried through the refinement
                loop so that the caller can reuse them for mesh construction
                without a redundant SDF evaluation.
            max_iters: maximum refinement iterations.

        Returns:
            ``(coords, levels, corner_sdf)`` where *corner_sdf* is the
            ``(N, 8, K)`` float32 tensor from the last
            :meth:`_find_surface_cubes` evaluation (or the input
            *corner_sdf* when no refinement iterations executed).
        """
        target_cubes = self.coarse_count * 4

        for _ in range(max_iters):
            n = len(coords)
            if n >= target_cubes:
                break
            # Compute scale once and share between centers and projection.
            cube_scales = self._cube_scales(levels)
            positions = self._cube_centers(coords, levels, cube_scales=cube_scales)
            proj = self._projected_sizes(positions, levels, cube_scales=cube_scales)
            to_expand = proj > self.inv_scale
            if not to_expand.any():
                break

            expand_idx = torch.where(to_expand)[0]
            n_expand = len(expand_idx)
            n_keep = n - n_expand

            budget = max(1, (target_cubes - n_keep) // 8)
            if n_expand > budget:
                _, top_k = proj[expand_idx].topk(budget)
                expand_idx = expand_idx[top_k]
                to_expand = torch.zeros(n, dtype=torch.bool, device=self.device)
                to_expand[expand_idx] = True

            keep_mask = ~to_expand
            child_c = coords[expand_idx].unsqueeze(1) * 2 + self._child_offsets.unsqueeze(0)
            child_l = (levels[expand_idx] + 1).unsqueeze(1).expand(-1, 8)
            child_coords = child_c.reshape(-1, 3)
            child_levels = child_l.reshape(-1)

            if corner_sdf is not None:
                # Optimised path: kept cubes already have valid corner SDF
                # values — evaluate only the new children.
                child_mask, child_corner_sdf = self._find_surface_cubes(
                    kernels, child_coords, child_levels,
                )
                coords = torch.cat([coords[keep_mask], child_coords[child_mask]])
                levels = torch.cat([levels[keep_mask], child_levels[child_mask]])
                corner_sdf = torch.cat([corner_sdf[keep_mask], child_corner_sdf[child_mask]])
            else:
                # Fallback: no cached SDF — evaluate everything.
                new_coords = torch.cat([coords[keep_mask], child_coords])
                new_levels = torch.cat([levels[keep_mask], child_levels])
                mask, corner_sdf_new = self._find_surface_cubes(kernels, new_coords, new_levels)
                coords = new_coords[mask]
                levels = new_levels[mask]
                corner_sdf = corner_sdf_new[mask]
        return coords, levels, corner_sdf

    # ------------------------------------------------------------------
    # Visibility filter (batched projection)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _visibility_filter(self, positions: torch.Tensor) -> torch.Tensor:
        """Classify positions as visible / occluded via depth buffering.

        Uses pre-computed K @ inv_pose projection matrix and batched operations.
        Projection, in-view checks, bin coordinate computation, and neighbour
        relaxation are fully vectorised across all cameras simultaneously.
        Only the per-camera depth-buffer scatter remains sequential.

        Returns:
            ``(N,)`` bool tensor - *True* for visible.
        """
        n = positions.shape[0]

        # Pad-free projection: use pre-split rotation + translation.
        img_all = torch.einsum("cij,nj->cni", self._proj_R, positions) + self._proj_t  # (C, N, 3)

        # Vectorised across all cameras: extract depth and pixel coords
        depth_all = img_all[:, :, 2]  # (C, N)
        depth_safe = depth_all + 1e-10
        px_all = img_all[:, :, 0] / depth_safe  # (C, N)
        py_all = img_all[:, :, 1] / depth_safe  # (C, N)

        # Per-camera bounds as tensors (C,) for vectorised in-view check
        h_all = self._cam_heights_t.unsqueeze(1)  # (C, 1)
        w_all = self._cam_widths_t.unsqueeze(1)  # (C, 1)
        in_view_all = (depth_all > 0) & (px_all >= 0) & (px_all < w_all) & (py_all >= 0) & (py_all < h_all)

        if not self.simplify_occluded:
            # Simple case: visible if in any camera view
            return in_view_all.any(dim=0)

        # --- Batched bin coordinate computation (all cameras at once) ---
        # Pre-computed per-camera bin dimensions: _vis_hb (C,), _vis_wb (C,)
        hb_col = self._vis_hb.unsqueeze(1)  # (C, 1) for broadcast
        wb_col = self._vis_wb.unsqueeze(1)  # (C, 1) for broadcast

        bx_all = (px_all * self._vis_inv_factor).long().clamp_(min=0)  # (C, N)
        by_all = (py_all * self._vis_inv_factor).long().clamp_(min=0)  # (C, N)
        bx_all = torch.minimum(bx_all, wb_col - 1)  # per-camera upper clamp
        by_all = torch.minimum(by_all, hb_col - 1)

        # Linear buffer indices: idx = bx * hb + by (per-camera hb broadcast)
        idx_all = bx_all * hb_col + by_all  # (C, N)

        valid_all = in_view_all & (depth_all > 0)  # (C, N)

        # --- Per-camera depth buffer scatter (cannot be batched) ---
        # Reset batched depth buffer to +inf; one row per camera, padded to
        # the largest camera's grid so indices don't exceed bounds.
        self._depth_bufs.fill_(float("inf"))
        for k in range(self.n_cameras):
            valid_k = valid_all[k]
            if valid_k.any():
                self._depth_bufs[k].scatter_reduce_(
                    0, idx_all[k][valid_k], depth_all[k][valid_k], reduce="amin",
                )

        # --- Batched neighbour relaxation (all cameras simultaneously) ---
        # nbx/nby: (C, N, R) via broadcast from (C, N, 1) + (1, 1, R)
        nbx = (bx_all.unsqueeze(2) + self._relax_dx).clamp_(min=0)  # (C, N, R)
        nby = (by_all.unsqueeze(2) + self._relax_dy).clamp_(min=0)
        nbx = torch.minimum(nbx, (wb_col - 1).unsqueeze(2))
        nby = torch.minimum(nby, (hb_col - 1).unsqueeze(2))
        nb_idx = nbx * hb_col.unsqueeze(2) + nby  # (C, N, R)

        # Batched gather: look up neighbour depths from all camera buffers at
        # once.  Reshape to (C, N*R) for gather, then back to (C, N, R).
        n_relax = nb_idx.shape[2]
        nb_flat = nb_idx.reshape(self.n_cameras, -1)  # (C, N*R)
        nb_depths = torch.gather(self._depth_bufs, 1, nb_flat).reshape(
            self.n_cameras, n, n_relax,
        )  # (C, N, R)

        # A point is near the front surface if its depth ≤ any neighbour's
        # buffered depth, for at least one camera.
        near_front = (depth_all.unsqueeze(2) <= nb_depths).any(dim=2)  # (C, N)
        return (in_view_all & near_front).any(dim=0)  # (N,)

    # ------------------------------------------------------------------
    # Marching cubes (fully vectorised, fused)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _marching_cubes(
        self,
        corners: torch.Tensor,
        sdf: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run marching cubes on the given cubes (fully vectorised).

        Uses cached lookup tables and fused configuration + interpolation.

        Args:
            corners: ``(N, 8, 3)`` float64 - world positions of cube corners.
            sdf:     ``(N, 8)``    float32 - SDF values at corners.

        Returns:
            ``(vertices, faces)`` as numpy arrays.
        """
        n = sdf.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

        device = sdf.device
        cache = TorchOcMesher._mc_cache[device]
        edge_table_t = cache["edge_table"]
        tri_table_t = cache["tri_table"]
        bit_shifts = cache["bit_shifts"]
        max_tri_entries = int(cache["max_tri_entries"].item())

        # --- Fused cube configuration (vectorised bit-shifts) ---
        neg_mask = (sdf < 0).int()  # (N, 8)
        cube_idx = (neg_mask * bit_shifts.unsqueeze(0)).sum(dim=1).int()  # (N,)

        edge_mask = edge_table_t[cube_idx.long()]
        active = edge_mask != 0
        if not active.any():
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

        # Keep only active cubes
        active_idx = torch.where(active)[0]
        a_sdf = sdf[active_idx]
        a_corners = corners[active_idx]
        a_cfg = cube_idx[active_idx].long()

        # --- Vectorised edge interpolation ---
        ev = self._edge_vertices
        s0 = a_sdf[:, ev[:, 0]]  # (A, 12)
        s1 = a_sdf[:, ev[:, 1]]
        denom = s0 - s1
        t = torch.where(denom.abs() < _DENOM_EPS, torch.tensor(0.5, device=device), s0 / denom)
        t.clamp_(0.0, 1.0)
        t = t.unsqueeze(-1).to(dtype=a_corners.dtype)  # (A, 12, 1) match corner dtype for lerp
        p0 = a_corners[:, ev[:, 0]]
        p1 = a_corners[:, ev[:, 1]]
        edge_positions = torch.lerp(p0, p1, t)  # fused linear interpolation

        # Lookup per-cube triangle lists
        tri_entries = tri_table_t[a_cfg]

        # --- Extract triangles (fully vectorised gather) ---
        # Reshape tri_entries into (A, max_tris, 3) for batch gather.
        max_tris_per_cube = max_tri_entries // 3
        tri_edge_ids = tri_entries[:, : max_tris_per_cube * 3].reshape(-1, max_tris_per_cube, 3)  # (A, T, 3)
        # A triangle slot is valid when its first edge index >= 0
        tri_valid = tri_edge_ids[:, :, 0] >= 0  # (A, T)
        # Flatten valid triangles
        valid_cube_idx, valid_tri_idx = torch.where(tri_valid)
        if valid_cube_idx.numel() == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
        edge_ids = tri_edge_ids[valid_cube_idx, valid_tri_idx]  # (F, 3)
        # Gather edge positions for each triangle vertex
        v0 = edge_positions[valid_cube_idx, edge_ids[:, 0].long()]  # (F, 3)
        v1 = edge_positions[valid_cube_idx, edge_ids[:, 1].long()]  # (F, 3)
        v2 = edge_positions[valid_cube_idx, edge_ids[:, 2].long()]  # (F, 3)
        tri_verts = torch.stack([v0, v1, v2], dim=1)  # (F, 3, 3)
        verts_flat = tri_verts.reshape(-1, 3)

        # --- Vertex deduplication via quantised coordinate hashing ---
        # Perform quantisation on the device (GPU or CPU) to avoid a transfer
        # for the hashing step.  Only transfer the final deduplicated arrays.
        quantized = (verts_flat * 1e8).round().long()
        # Encode (x, y, z) triples into a single int64 hash per vertex.
        # The components are bounded by the scene extent x 1e8 which fits in
        # int64 when multiplied by large primes.
        hash_vals = quantized[:, 0] * 1000000007 + quantized[:, 1] * 1000000009 + quantized[:, 2] * 1000000021
        _, inverse = torch.unique(hash_vals, return_inverse=True)
        # Build compact vertex array: pick the first representative per unique
        # hash.  Writing indices in reverse order ensures the smallest (first)
        # index wins for each unique bucket.
        n_unique = int(inverse.max().item()) + 1
        n_verts = len(inverse)
        rep_idx = torch.zeros(n_unique, dtype=torch.long, device=device)
        rev_arange = torch.arange(n_verts - 1, -1, -1, device=device)
        rep_idx.scatter_(0, inverse[rev_arange], rev_arange)
        dedup_verts = verts_flat[rep_idx].cpu().double().numpy()
        dedup_faces = inverse.reshape(-1, 3).cpu().numpy().astype(np.int32)
        return dedup_verts, dedup_faces

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def __call__(self, kernels):
        """Run the full coarse-to-fine meshing pipeline and return meshes."""
        n_elements = len(kernels)

        # 1. Build coarse octree -------------------------------------------
        with Timer("torch coarse octree"):
            coords, levels = self._build_coarse_octree()
            logger.info("coarse cubes: %d", len(coords))

        # 2. Detect surface cubes ------------------------------------------
        with Timer("torch find surface"):
            surface_mask, corner_sdf = self._find_surface_cubes(kernels, coords, levels)
            s_coords = coords[surface_mask]
            s_levels = levels[surface_mask]
            s_corner_sdf = corner_sdf[surface_mask]
            logger.info("surface cubes: %d", len(s_coords))

        # 3. Refine surface cubes ------------------------------------------
        with Timer("torch refine surface"):
            s_coords, s_levels, s_corner_sdf = self._refine_surface_octree(
                kernels,
                s_coords,
                s_levels,
                corner_sdf=s_corner_sdf,
            )
            logger.info("refined surface cubes: %d", len(s_coords))

        # 4. Visibility filter ---------------------------------------------
        with Timer("torch visibility filter"):
            positions = self._cube_centers(s_coords, s_levels)
            vis_mask = self._visibility_filter(positions)
            vis_coords = s_coords[vis_mask]
            vis_levels = s_levels[vis_mask]
            occ_coords = s_coords[~vis_mask]
            occ_levels = s_levels[~vis_mask]
            logger.info("visible: %d, occluded: %d", len(vis_coords), len(occ_coords))

        # 5. Per-element mesh construction ---------------------------------
        with Timer("torch construct mesh"):
            meshes: list[trimesh.Trimesh] = []
            in_view_tags: list[np.ndarray] = []
            all_coords = torch.cat([vis_coords, occ_coords])
            all_levels = torch.cat([vis_levels, occ_levels])
            n_visible = len(vis_coords)

            # Reorder cached SDF to match the visible-then-occluded layout.
            all_corner_sdf: torch.Tensor | None = None
            if s_corner_sdf is not None:
                all_corner_sdf = torch.cat([s_corner_sdf[vis_mask], s_corner_sdf[~vis_mask]])

            for e in range(n_elements):
                mesh, ivt = self._construct_element_mesh(
                    kernels[e : e + 1],
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
                    e,
                    mesh.vertices.shape[0],
                    mesh.faces.shape[0],
                )

        return meshes, in_view_tags

    @torch.no_grad()
    def _construct_element_mesh(
        self,
        kernels: list,
        coords: torch.Tensor,
        levels: torch.Tensor,
        n_visible: int,
        corner_sdf: torch.Tensor | None = None,
        element_idx: int = 0,
    ) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Build a mesh for one SDF element using marching cubes.

        When *corner_sdf* is provided (``(N, 8, K)`` float32 tensor cached from
        :meth:`_find_surface_cubes`), the expensive SDF re-evaluation is
        skipped entirely: the cached values for kernel *element_idx* are used
        directly, giving a significant speed-up for the mesh-construction step.
        """
        n = len(coords)
        if n == 0:
            return trimesh.Trimesh(), np.zeros(0, dtype=bool)

        # Process in chunks to control memory
        chunk_size = max(1, min(n, self.memory_limit_mb * 1000 // 64))
        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        vert_offset = 0

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            c_coords = coords[start:end]
            c_levels = levels[start:end]

            chunk_corners = self._cube_corner_positions(c_coords, c_levels)

            if corner_sdf is not None:
                # Reuse cached SDF — no redundant kernel evaluation needed.
                sdf_min = corner_sdf[start:end, :, element_idx]
            else:
                flat = chunk_corners.reshape(-1, 3)
                sdf_all = self._evaluate_sdf(kernels, flat)
                sdf_min = sdf_all.min(dim=-1).values.reshape(end - start, 8)

            v, f = self._marching_cubes(chunk_corners, sdf_min)
            if v.shape[0] > 0:
                f = f + vert_offset
                all_verts.append(v)
                all_faces.append(f)
                vert_offset += v.shape[0]

        if not all_verts:
            return trimesh.Trimesh(), np.zeros(0, dtype=bool)

        verts_np = np.concatenate(all_verts, axis=0)
        faces_np = np.concatenate(all_faces, axis=0)

        # In-view tag: compute bounding box on device, transfer only 2x3 floats
        in_view = np.ones(verts_np.shape[0], dtype=bool)
        if n_visible < n and n_visible > 0:
            vis_pos = self._cube_centers(coords[:n_visible], levels[:n_visible])
            vis_min = vis_pos.min(dim=0).values.cpu().numpy()
            vis_max = vis_pos.max(dim=0).values.cpu().numpy()
            margin = self.size * 0.05
            in_view = np.all(verts_np >= vis_min - margin, axis=1) & np.all(
                verts_np <= vis_max + margin,
                axis=1,
            )

        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        return mesh, in_view
