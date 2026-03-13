# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""C++ shared library loading and function registration for OcMesher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils.interface import (
    POINTER,
    c_bool,
    c_double,
    c_float,
    c_int32,
    load_cdll,
    register_func,
)

# Pointer type aliases used in function signatures
_FP = POINTER(c_double)
_SFP = POINTER(c_float)
_IP = POINTER(c_int32)
_BP = POINTER(c_bool)

# Each entry: (name, argtypes, restype)
# Grouped by pipeline stage for readability.
_FUNCTION_SPECS: list[tuple[str, list[Any], Any]] = [
    # --- Coarse octree construction ---
    ("run_coarse", [
        _FP, c_double, c_int32, _FP,
        c_double, c_double, c_double,
        c_int32, c_int32, c_int32,
    ], c_int32),
    # --- Fine-group SDF refinement ---
    ("fine_group", [], c_int32),
    ("fine_iteration", [_SFP], c_int32),
    ("fine_iteration_output", [_FP], None),
    # --- Visibility filtering ---
    ("vis_filter", [c_bool, c_int32], c_int32),
    # --- Fine-step surface finding ---
    ("final_iteration", [], c_int32),
    ("final_iteration_occluded", [], c_int32),
    ("final_iteration2", [_FP], None),
    ("final_iteration3", [_SFP], c_int32),
    ("final_iteration3_occluded", [_SFP], None),
    ("final_remaining", [_IP], None),
    # --- Vertex bisection ---
    ("get_verts_center", [c_int32, _FP], None),
    ("update_verts", [c_int32, _SFP, _SFP, _FP], None),
    ("get_lr_verts", [c_int32, _FP, _FP], None),
    ("finalize_verts", [c_int32, _SFP, _SFP, _FP], None),
    # --- Face construction ---
    ("construct_faces", [c_int32, _FP, _IP], None),
    # --- Extra (edge/face) vertex bisection ---
    ("get_extra_verts_center", [_FP, _FP], None),
    ("update_extra_verts", [_SFP, _SFP, _SFP, _SFP, _FP, _FP], None),
    ("get_lr_extra_verts", [_FP, _FP, _FP, _FP], None),
    ("finalize_extra_verts", [_SFP, _SFP, _FP, _SFP, _SFP, _FP], None),
    # --- Output ---
    ("get_faces", [_IP], None),
    ("get_in_view_tag", [c_int32, _BP], None),
]


class CoreDLL:
    """Wrapper around the compiled C++ core shared library.

    Loads ``core.so`` and registers every exported function as an attribute,
    using the signatures defined in :data:`_FUNCTION_SPECS`.

    Example::

        dll = CoreDLL()
        n = dll.run_coarse(center_ptr, size, ...)
    """

    def __init__(self, lib_dir: str | Path | None = None) -> None:
        if lib_dir is None:
            lib_dir = Path(__file__).parent.resolve() / "lib"
        lib_path = Path(lib_dir) / "core.so"
        raw = load_cdll(str(lib_path))
        for name, argtypes, restype in _FUNCTION_SPECS:
            register_func(self, raw, name, argtypes, restype)
