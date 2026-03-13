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


# note: size of x should not exceed maximum
def AsInt(x: np.ndarray) -> "POINTER(c_int32)":
    """Cast *x* to a ``c_int32`` pointer."""
    return x.ctypes.data_as(POINTER(c_int32))


def AsDouble(x: np.ndarray) -> "POINTER(c_double)":
    """Cast *x* to a ``c_double`` pointer."""
    return x.ctypes.data_as(POINTER(c_double))


def AsFloat(x: np.ndarray) -> "POINTER(c_float)":
    """Cast *x* to a ``c_float`` pointer."""
    return x.ctypes.data_as(POINTER(c_float))


def AsBool(x: np.ndarray) -> "POINTER(c_bool)":
    """Cast *x* to a ``c_bool`` pointer."""
    return x.ctypes.data_as(POINTER(c_bool))


def register_func(
    me: Any, dll: CDLL, name: str, argtypes: list | None = None, restype: Any = None, caller_name: str | None = None
) -> None:
    """Register a C function from *dll* on object *me*."""
    if argtypes is None:
        argtypes = []
    if caller_name is None:
        caller_name = name
    setattr(me, caller_name, getattr(dll, name))
    func = getattr(me, caller_name)
    func.argtypes = argtypes
    func.restype = restype


def load_cdll(path: str) -> CDLL:
    """Load a shared library from *path* relative to ``sys.path``."""
    return CDLL(Path(sys.path[-1]) / path, mode=RTLD_LOCAL)
