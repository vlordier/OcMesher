# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Ctypes helpers for loading shared libraries and converting numpy arrays to C pointers."""

import sys
from ctypes import CDLL, POINTER, RTLD_LOCAL, c_bool, c_double, c_float, c_int32
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ascontiguousarray as AC

__all__ = [
    "AC",
    "POINTER",
    "AsBool",
    "AsDouble",
    "AsFloat",
    "AsInt",
    "as_bool",
    "as_double",
    "as_float",
    "as_int",
    "c_bool",
    "c_double",
    "c_float",
    "c_int32",
    "load_cdll",
    "register_func",
]


def _checked_ptr(x: Any, expected_dtype: np.dtype, ctype: type) -> Any:
    """Return a ctypes pointer after validating dtype and contiguity.

    Raises:
        TypeError: If *x* is not a numpy array or has the wrong dtype.
        ValueError: If *x* is not C-contiguous.
    """
    if not isinstance(x, np.ndarray):
        msg = f"Expected a numpy array, got {type(x).__name__}"
        raise TypeError(msg)
    if x.dtype != expected_dtype:
        msg = f"Expected dtype {expected_dtype}, got {x.dtype}"
        raise TypeError(msg)
    if not x.flags["C_CONTIGUOUS"]:
        msg = "Array must be C-contiguous; call numpy.ascontiguousarray() first"
        raise ValueError(msg)
    return x.ctypes.data_as(POINTER(ctype))


# snake_case helpers (preferred in new code) --------------------------------


def as_int(x: np.ndarray) -> Any:
    """Cast *x* to a ``c_int32`` pointer (validates int32 dtype and contiguity)."""
    return _checked_ptr(x, np.dtype(np.int32), c_int32)


def as_double(x: np.ndarray) -> Any:
    """Cast *x* to a ``c_double`` pointer (validates float64 dtype and contiguity)."""
    return _checked_ptr(x, np.dtype(np.float64), c_double)


def as_float(x: np.ndarray) -> Any:
    """Cast *x* to a ``c_float`` pointer (validates float32 dtype and contiguity)."""
    return _checked_ptr(x, np.dtype(np.float32), c_float)


def as_bool(x: np.ndarray) -> Any:
    """Cast *x* to a ``c_bool`` pointer (validates bool dtype and contiguity)."""
    return _checked_ptr(x, np.dtype(np.bool_), c_bool)


# PascalCase aliases for backward compatibility ------------------------------
AsInt = as_int
AsDouble = as_double
AsFloat = as_float
AsBool = as_bool


def register_func(
    me: Any, dll: CDLL, name: str, argtypes: list | None = None, restype: Any = None, caller_name: str | None = None
) -> None:
    """Register a C function from *dll* on object *me*."""
    if argtypes is None:
        argtypes = []
    if caller_name is None:
        caller_name = name
    try:
        func = getattr(dll, name)
    except AttributeError:
        msg = f"Function '{name}' not found in shared library"
        raise AttributeError(msg) from None
    func.argtypes = argtypes
    func.restype = restype
    setattr(me, caller_name, func)


def load_cdll(path: str) -> CDLL:
    """Load a shared library from *path*.

    If *path* is absolute it is used directly.  Otherwise every entry in
    ``sys.path`` is searched in order and the first match is used.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        lib_path = candidate
    else:
        for base in sys.path:
            resolved = Path(base) / path
            if resolved.exists():
                lib_path = resolved
                break
        else:
            msg = f"Shared library not found: '{path}' was not found in any sys.path entry. Run install.sh to build."
            raise FileNotFoundError(msg)
    if not lib_path.exists():
        msg = f"Shared library not found at {lib_path}. Run install.sh to build."
        raise FileNotFoundError(msg)
    try:
        return CDLL(str(lib_path), mode=RTLD_LOCAL)
    except OSError as e:
        msg = f"Failed to load shared library at {lib_path}: {e}"
        raise OSError(msg) from e
