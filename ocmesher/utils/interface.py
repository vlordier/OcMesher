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
    "c_bool",
    "c_double",
    "c_float",
    "c_int32",
    "load_cdll",
    "register_func",
]


def _checked_ptr(x: Any, ctype: type) -> Any:
    """Convert a numpy array to a ctypes pointer, validating the input type."""
    if not isinstance(x, np.ndarray):
        msg = f"Expected a numpy array, got {type(x).__name__}"
        raise TypeError(msg)
    return x.ctypes.data_as(POINTER(ctype))


# note: size of x should not exceed maximum
def AsInt(x: np.ndarray) -> "POINTER(c_int32)":
    """Cast *x* to a ``c_int32`` pointer."""
    return _checked_ptr(x, c_int32)


def AsDouble(x: np.ndarray) -> "POINTER(c_double)":
    """Cast *x* to a ``c_double`` pointer."""
    return _checked_ptr(x, c_double)


def AsFloat(x: np.ndarray) -> "POINTER(c_float)":
    """Cast *x* to a ``c_float`` pointer."""
    return _checked_ptr(x, c_float)


def AsBool(x: np.ndarray) -> "POINTER(c_bool)":
    """Cast *x* to a ``c_bool`` pointer."""
    return _checked_ptr(x, c_bool)


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
