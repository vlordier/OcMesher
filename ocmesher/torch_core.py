# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""PyTorch-based octree mesher with GPU acceleration.

Provides :class:`TorchOcMesher`, a drop-in replacement for :class:`OcMesher`
that replaces the C++ backend with pure PyTorch tensor operations.  All heavy
numerical work (camera projections, SDF evaluation batching, vertex bisection,
visibility filtering) runs on GPU when available.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from .utils.timer import Timer

# Epsilon for safe division in marching-cubes interpolation.
_DENOM_EPS = 1e-12

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


class TorchOcMesher:
    """Octree-based mesher that extracts surfaces from SDF kernels using PyTorch.

    Drop-in replacement for :class:`OcMesher` - same constructor signature and
    ``__call__`` contract, but all heavy computation is performed with PyTorch
    tensors (GPU when available).
    """

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
    ):
        """Initialise the mesher with camera intrinsics and bounds."""
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        cam_poses, Ks, Hs, Ws = cameras
        self.n_cameras = len(cam_poses)

        # Pack camera data as tensors ----------------------------------------
        inv_poses = []
        intrinsics = []
        self.cam_heights: list[int] = []
        self.cam_widths: list[int] = []
        for i in range(self.n_cameras):
            inv_poses.append(torch.from_numpy(np.linalg.inv(cam_poses[i])[:3, :4].astype(np.float64)))
            intrinsics.append(torch.from_numpy(Ks[i].astype(np.float64)))
            self.cam_heights.append(int(Hs[i]))
            self.cam_widths.append(int(Ws[i]))
        self.cam_inv_poses = torch.stack(inv_poses).to(self.device)  # (C, 3, 4)
        self.cam_intrinsics = torch.stack(intrinsics).to(self.device)  # (C, 3, 3)

        # Scene bounds -------------------------------------------------------
        self.bounds = bounds
        bt = torch.tensor(bounds, dtype=torch.float64, device=self.device)
        self.bounds_min = bt[0::2]  # (3,)
        self.bounds_max = bt[1::2]  # (3,)

        self.center = (self.bounds_min + self.bounds_max) / 2
        extent = self.bounds_max - self.bounds_min
        self.size = float(extent.max().item() * 1.1)

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

        # Precompute per-camera pixel angular size ---------------------------
        self._pix_ang: list[float] = []
        for k in range(self.n_cameras):
            fx = float(self.cam_intrinsics[k][0, 0].item())
            w = self.cam_widths[k]
            self._pix_ang.append(float(np.arctan(w / 2 / fx) * 2 / w))

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _cube_centers(self, coords: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """Integer octree coords → world-space centre positions.

        Args:
            coords: ``(N, 3)`` int64 tensor of integer cube coordinates.
            levels:  ``(N,)``  int64 tensor of octree levels.

        Returns:
            ``(N, 3)`` float64 world positions.
        """
        scale = self.size / (2.0 ** levels.unsqueeze(1).double())  # (N, 1)
        return self.center.unsqueeze(0) - self.size / 2 + scale * (coords.double() + 0.5)

    @torch.no_grad()
    def _cube_corner_positions(self, coords: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """Return world positions of all 8 corners for each cube.

        Returns:
            ``(N, 8, 3)`` float64 tensor.
        """
        offsets = torch.tensor(_CORNER_OFFSETS, dtype=torch.int64, device=self.device)  # (8, 3)
        corner_coords = coords.unsqueeze(1) + offsets.unsqueeze(0)  # (N, 8, 3)
        scale = self.size / (2.0 ** levels.unsqueeze(1).unsqueeze(2).double())
        return self.center.unsqueeze(0).unsqueeze(0) - self.size / 2 + scale * corner_coords.double()

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _projected_sizes(self, positions: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """Max projected pixel size across all cameras for each position.

        Args:
            positions: ``(N, 3)`` world positions.
            levels:    ``(N,)``  octree levels.

        Returns:
            ``(N,)`` float64 tensor of projected sizes.
        """
        n = positions.shape[0]
        ones = torch.ones(n, 1, dtype=torch.float64, device=self.device)
        pos_h = torch.cat([positions, ones], dim=1)  # (N, 4)
        cube_sizes = self.size / (2.0 ** levels.double())  # (N,)

        max_sz = torch.zeros(n, dtype=torch.float64, device=self.device)
        for k in range(self.n_cameras):
            cam_coords = (self.cam_inv_poses[k] @ pos_h.T).T  # (N, 3)
            r = cam_coords.norm(dim=1).clamp(min=self.min_dist)
            ang = self._pix_ang[k] * self.pixels_per_cube
            proj = cube_sizes / r / ang
            max_sz = torch.maximum(max_sz, proj)
        return max_sz

    # ------------------------------------------------------------------
    # SDF evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_sdf(
        self,
        kernels: list,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate SDF *kernels* at *positions*.

        Kernel functions receive :class:`numpy.ndarray` and return the same.
        Conversion to/from tensors is handled here.

        Returns:
            ``(N, len(kernels))`` float32 tensor on *self.device*.
        """
        n = positions.shape[0]
        if n == 0:
            return torch.zeros((0, len(kernels)), dtype=torch.float32, device=self.device)

        xyz_np = positions.cpu().double().numpy()
        step = 10_000_000
        parts: list[np.ndarray] = []
        for i in range(0, n, step):
            chunk = xyz_np[i : i + step]
            out_bound = np.zeros(len(chunk), dtype=bool)
            if self.enclosed:
                for c in range(3):
                    out_bound |= chunk[:, c] <= self.bounds[c * 2]
                    out_bound |= chunk[:, c] >= self.bounds[c * 2 + 1]
            cols: list[np.ndarray] = []
            for kernel in kernels:
                sdf = kernel(chunk)
                if self.enclosed:
                    sdf[out_bound] = 1
                cols.append(sdf)
            parts.append(np.stack(cols, axis=-1).astype(np.float32))
        return torch.from_numpy(np.concatenate(parts, axis=0)).to(self.device)

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
            if len(coords) >= self.coarse_count:
                break
            positions = self._cube_centers(coords, levels)
            proj = self._projected_sizes(positions, levels)
            to_expand = proj > self.inv_scale
            if not to_expand.any():
                break

            expand_idx = torch.where(to_expand)[0]
            keep_idx = torch.where(~to_expand)[0]

            # Budget-limit expansion
            remaining = self.coarse_count - len(keep_idx)
            if remaining < len(expand_idx) * 8:
                _, top_k = proj[expand_idx].topk(max(1, remaining // 8))
                expand_idx = expand_idx[top_k]
                keep_mask = torch.ones(len(coords), dtype=torch.bool, device=self.device)
                keep_mask[expand_idx] = False
                keep_idx = torch.where(keep_mask)[0]

            offsets = torch.tensor(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                dtype=torch.int64,
                device=self.device,
            )
            child_c = coords[expand_idx].unsqueeze(1) * 2 + offsets.unsqueeze(0)
            child_l = (levels[expand_idx] + 1).unsqueeze(1).expand(-1, 8)

            coords = torch.cat([coords[keep_idx], child_c.reshape(-1, 3)])
            levels = torch.cat([levels[keep_idx], child_l.reshape(-1)])
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

        Returns:
            ``(mask, corner_sdf)`` where *mask* is ``(N,)`` bool and
            *corner_sdf* is ``(N, 8, K)`` float32 tensor.
        """
        corners = self._cube_corner_positions(coords, levels)  # (N, 8, 3)
        n, _, _ = corners.shape
        flat = corners.reshape(-1, 3)
        sdf = self._evaluate_sdf(kernels, flat)  # (N*8, K)
        sdf = sdf.reshape(n, 8, -1)
        sdf_min = sdf.min(dim=-1).values  # (N, 8) - combined surface
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
        max_iters: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Iteratively subdivide surface cubes that project large on screen.

        Uses a conservative budget to avoid over-refinement.
        """
        offsets = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            dtype=torch.int64,
            device=self.device,
        )
        target_cubes = self.coarse_count * 4  # budget limit

        for _ in range(max_iters):
            if len(coords) >= target_cubes:
                break
            positions = self._cube_centers(coords, levels)
            proj = self._projected_sizes(positions, levels)
            to_expand = proj > self.inv_scale
            if not to_expand.any():
                break

            expand_idx = torch.where(to_expand)[0]
            keep_idx = torch.where(~to_expand)[0]

            # Budget-limit: only expand the largest cubes up to budget
            budget = max(1, (target_cubes - len(keep_idx)) // 8)
            if len(expand_idx) > budget:
                _, top_k = proj[expand_idx].topk(budget)
                expand_idx = expand_idx[top_k]
                keep_mask = torch.ones(len(coords), dtype=torch.bool, device=self.device)
                keep_mask[expand_idx] = False
                keep_idx = torch.where(keep_mask)[0]

            child_c = coords[expand_idx].unsqueeze(1) * 2 + offsets.unsqueeze(0)
            child_l = (levels[expand_idx] + 1).unsqueeze(1).expand(-1, 8)

            new_coords = torch.cat([coords[keep_idx], child_c.reshape(-1, 3)])
            new_levels = torch.cat([levels[keep_idx], child_l.reshape(-1)])

            mask, _ = self._find_surface_cubes(kernels, new_coords, new_levels)
            coords = new_coords[mask]
            levels = new_levels[mask]
        return coords, levels

    # ------------------------------------------------------------------
    # Visibility filter
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _visibility_filter(self, positions: torch.Tensor) -> torch.Tensor:
        """Classify positions as visible / occluded via depth buffering.

        Returns:
            ``(N,)`` bool tensor - *True* for visible.
        """
        n = positions.shape[0]
        visible = torch.zeros(n, dtype=torch.bool, device=self.device)
        ones = torch.ones(n, 1, dtype=torch.float64, device=self.device)
        pos_h = torch.cat([positions, ones], dim=1)  # (N, 4)

        factor = 10.0
        for k in range(self.n_cameras):
            cam_xyz = (self.cam_inv_poses[k] @ pos_h.T).T  # (N, 3)
            img = (self.cam_intrinsics[k] @ cam_xyz.T).T  # (N, 3)
            depth = img[:, 2]
            px = img[:, 0] / (depth + 1e-10)
            py = img[:, 1] / (depth + 1e-10)
            h, w = self.cam_heights[k], self.cam_widths[k]
            in_view = (depth > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)

            if self.simplify_occluded:
                hb = max(1, int(h / factor))
                wb = max(1, int(w / factor))
                bx = (px / factor).long().clamp(0, wb - 1)
                by = (py / factor).long().clamp(0, hb - 1)
                depth_buf = torch.full((wb * hb,), float("inf"), dtype=torch.float64, device=self.device)
                valid = in_view & (depth > 0)
                if valid.any():
                    idx = bx[valid] * hb + by[valid]
                    depth_buf.scatter_reduce_(0, idx, depth[valid], reduce="amin")
                rl = self.visible_relax_iter
                for dx in range(-rl, rl + 1):
                    for dy in range(-rl, rl + 1):
                        nbx = (bx + dx).clamp(0, wb - 1)
                        nby = (by + dy).clamp(0, hb - 1)
                        nb_idx = nbx * hb + nby
                        near_front = depth <= depth_buf[nb_idx]
                        visible |= in_view & near_front
            else:
                visible |= in_view
        return visible

    # ------------------------------------------------------------------
    # Marching cubes (fully vectorized)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _marching_cubes(
        self,
        corners: torch.Tensor,
        sdf: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run marching cubes on the given cubes (vectorized, no Python loops).

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

        # --- Cube configuration indices (vectorized) ---
        cube_idx = torch.zeros(n, dtype=torch.int32, device=device)
        for i in range(8):
            cube_idx |= torch.where(
                sdf[:, i] < 0,
                torch.tensor(1 << i, dtype=torch.int32, device=device),
                torch.tensor(0, dtype=torch.int32, device=device),
            )

        edge_table_t = torch.tensor(_EDGE_TABLE, dtype=torch.int32, device=device)
        edge_mask = edge_table_t[cube_idx.long()]  # (N,)
        active = edge_mask != 0
        if not active.any():
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

        # Keep only active cubes
        active_idx = torch.where(active)[0]
        a_sdf = sdf[active_idx]  # (A, 8)
        a_corners = corners[active_idx]  # (A, 8, 3)
        a_cfg = cube_idx[active_idx].long()  # (A,)

        # --- Pre-compute edge vertex positions for all 12 edges (vectorized) ---
        ev = torch.tensor(_EDGE_VERTICES, dtype=torch.long, device=device)  # (12, 2)
        s0 = a_sdf[:, ev[:, 0]]  # (A, 12)
        s1 = a_sdf[:, ev[:, 1]]  # (A, 12)
        denom = s0 - s1
        t = torch.where(denom.abs() < _DENOM_EPS, torch.tensor(0.5, device=device), s0 / denom)
        t = t.clamp(0.0, 1.0).unsqueeze(-1)  # (A, 12, 1)
        p0 = a_corners[:, ev[:, 0]]  # (A, 12, 3)
        p1 = a_corners[:, ev[:, 1]]  # (A, 12, 3)
        edge_positions = p0 * (1 - t) + p1 * t  # (A, 12, 3)

        # --- Build triangle table as a padded tensor ---
        max_tri_entries = max(len(row) for row in _TRI_TABLE)
        tri_table_np = np.full((256, max_tri_entries), -1, dtype=np.int32)
        for i, row in enumerate(_TRI_TABLE):
            tri_table_np[i, : len(row)] = row
        tri_table_t = torch.from_numpy(tri_table_np).to(device)

        # Lookup per-cube triangle lists
        tri_entries = tri_table_t[a_cfg]  # (A, max_tri_entries)

        # --- Extract triangles (vectorized over triangle slots) ---
        # Maximum 5 triangles per cube => 15 entries; process in groups of 3
        max_tris_per_cube = max_tri_entries // 3
        all_face_verts: list[torch.Tensor] = []

        for ti in range(max_tris_per_cube):
            base = ti * 3
            if base + 2 >= max_tri_entries:
                break
            e0 = tri_entries[:, base]  # (A,)
            e1 = tri_entries[:, base + 1]  # (A,)
            e2 = tri_entries[:, base + 2]  # (A,)
            valid = e0 >= 0  # triangles with -1 are inactive
            if not valid.any():
                break

            vi = torch.where(valid)[0]  # indices into active cubes
            # Gather vertex positions for this triangle
            v0 = edge_positions[vi, e0[vi].long()]  # (T, 3)
            v1 = edge_positions[vi, e1[vi].long()]  # (T, 3)
            v2 = edge_positions[vi, e2[vi].long()]  # (T, 3)
            # Stack as (T, 3, 3) where dim1 is vertex index
            all_face_verts.append(torch.stack([v0, v1, v2], dim=1))

        if not all_face_verts:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

        # Concatenate all triangles: (total_tris, 3, 3)
        tri_verts = torch.cat(all_face_verts, dim=0)
        total_tris = tri_verts.shape[0]

        # Flatten vertices and build face indices
        verts_flat = tri_verts.reshape(-1, 3)  # (total_tris*3, 3)
        faces_flat = torch.arange(total_tris * 3, device=device).reshape(-1, 3)

        # --- Vertex deduplication via rounding + unique ---
        verts_np = verts_flat.cpu().double().numpy()
        # Quantize to ~1e-8 precision for dedup
        quantized = np.round(verts_np * 1e8).astype(np.int64)
        _, unique_idx, inverse = np.unique(
            quantized,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        dedup_verts = verts_np[unique_idx]
        dedup_faces = inverse[faces_flat.cpu().numpy().ravel()].reshape(-1, 3)

        return dedup_verts, dedup_faces.astype(np.int32)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def __call__(self, kernels):
        """Run the full coarse-to-fine meshing pipeline and return meshes."""
        n_elements = len(kernels)

        # 1. Build coarse octree -------------------------------------------
        with Timer("torch coarse octree"):
            coords, levels = self._build_coarse_octree()
            print(f"  coarse cubes: {len(coords)}")

        # 2. Detect surface cubes ------------------------------------------
        with Timer("torch find surface"):
            surface_mask, _corner_sdf = self._find_surface_cubes(kernels, coords, levels)
            s_coords = coords[surface_mask]
            s_levels = levels[surface_mask]
            print(f"  surface cubes: {len(s_coords)}")

        # 3. Refine surface cubes ------------------------------------------
        with Timer("torch refine surface"):
            s_coords, s_levels = self._refine_surface_octree(kernels, s_coords, s_levels)
            print(f"  refined surface cubes: {len(s_coords)}")

        # 4. Visibility filter ---------------------------------------------
        with Timer("torch visibility filter"):
            positions = self._cube_centers(s_coords, s_levels)
            vis_mask = self._visibility_filter(positions)
            vis_coords = s_coords[vis_mask]
            vis_levels = s_levels[vis_mask]
            occ_coords = s_coords[~vis_mask]
            occ_levels = s_levels[~vis_mask]
            print(f"  visible: {len(vis_coords)}, occluded: {len(occ_coords)}")

        # 5. Per-element mesh construction ---------------------------------
        with Timer("torch construct mesh"):
            meshes: list[trimesh.Trimesh] = []
            in_view_tags: list[np.ndarray] = []
            # Combine visible + occluded; tag which is which
            all_coords = torch.cat([vis_coords, occ_coords])
            all_levels = torch.cat([vis_levels, occ_levels])
            n_visible = len(vis_coords)

            for e in range(n_elements):
                mesh, ivt = self._construct_element_mesh(
                    kernels[e : e + 1],
                    all_coords,
                    all_levels,
                    n_visible,
                )
                meshes.append(mesh)
                in_view_tags.append(ivt)
                print(f"element {e} has vertices #{mesh.vertices.shape[0]} faces #{mesh.faces.shape[0]}")

        return meshes, in_view_tags

    @torch.no_grad()
    def _construct_element_mesh(
        self,
        kernels: list,
        coords: torch.Tensor,
        levels: torch.Tensor,
        n_visible: int,
    ) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Build a mesh for one SDF element using marching cubes + bisection."""
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

            # Evaluate SDF at cube corners
            chunk_corners = self._cube_corner_positions(c_coords, c_levels)  # (C, 8, 3)
            flat = chunk_corners.reshape(-1, 3)
            sdf_all = self._evaluate_sdf(kernels, flat)  # (C*8, K)
            sdf_min = sdf_all.min(dim=-1).values.reshape(end - start, 8)

            # Vectorized marching cubes
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

        # In-view tag: mark vertices as visible if they fall within any
        # visible cube's bounding box (approximate).
        in_view = np.ones(verts_np.shape[0], dtype=bool)
        if n_visible < n and n_visible > 0:
            vis_pos = self._cube_centers(coords[:n_visible], levels[:n_visible])
            vis_pos_np = vis_pos.cpu().numpy()
            vis_min = vis_pos_np.min(axis=0)
            vis_max = vis_pos_np.max(axis=0)
            margin = self.size * 0.05
            in_view = np.all(verts_np >= vis_min - margin, axis=1) & np.all(verts_np <= vis_max + margin, axis=1)

        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        return mesh, in_view
