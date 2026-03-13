# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""ctypes helper utilities for C++ shared library interop."""

from __future__ import annotations

from ctypes import CDLL, POINTER, RTLD_LOCAL, c_bool, c_double, c_float, c_int32
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import ascontiguousarray as AC

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _checked_ptr(x: NDArray, expected_dtype: np.dtype, ctype: type) -> Any:
    """Return a ctypes pointer after validating dtype and contiguity.

    Raises:
        TypeError: If *x* has the wrong dtype.
        ValueError: If *x* is not C-contiguous.
    """
    if x.dtype != expected_dtype:
        msg = f"Expected dtype {expected_dtype}, got {x.dtype}"
        raise TypeError(msg)
    if not x.flags["C_CONTIGUOUS"]:
        msg = "Array must be C-contiguous; call numpy.ascontiguousarray() first"
        raise ValueError(msg)
    return x.ctypes.data_as(POINTER(ctype))


def as_int(x: NDArray) -> Any:
    """Convert int32 array to ctypes pointer."""
    return _checked_ptr(x, np.dtype(np.int32), c_int32)


def as_double(x: NDArray) -> Any:
    """Convert float64 array to ctypes pointer."""
    return _checked_ptr(x, np.dtype(np.float64), c_double)


def as_float(x: NDArray) -> Any:
    """Convert float32 array to ctypes pointer."""
    return _checked_ptr(x, np.dtype(np.float32), c_float)


def as_bool(x: NDArray) -> Any:
    """Convert bool array to ctypes pointer."""
    return _checked_ptr(x, np.dtype(np.bool_), c_bool)


def register_func(
    obj: Any,
    dll: CDLL,
    name: str,
    argtypes: list | None = None,
    restype: Any = None,
    caller_name: str | None = None,
) -> None:
    """Register a C function from a DLL onto an object."""
    caller_name = caller_name or name
    setattr(obj, caller_name, getattr(dll, name))
    func = getattr(obj, caller_name)
    func.argtypes = argtypes if argtypes is not None else []
    func.restype = restype


def load_cdll(path: str | Path) -> CDLL:
    """Load a shared library as a CDLL."""
    return CDLL(str(path), mode=RTLD_LOCAL)
