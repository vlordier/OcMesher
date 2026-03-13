# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""C-type helpers for interfacing with the native meshing library."""

import sys
from ctypes import CDLL, POINTER, RTLD_LOCAL, c_bool, c_double, c_float, c_int32
from pathlib import Path

from numpy import ascontiguousarray as AC  # noqa: F401 - re-exported for core.py


# note: size of x should not exceed maximum
def AsInt(x):
    """Cast *x* to a ``c_int32`` pointer."""
    return x.ctypes.data_as(POINTER(c_int32))


def AsDouble(x):
    """Cast *x* to a ``c_double`` pointer."""
    return x.ctypes.data_as(POINTER(c_double))


def AsFloat(x):
    """Cast *x* to a ``c_float`` pointer."""
    return x.ctypes.data_as(POINTER(c_float))


def AsBool(x):
    """Cast *x* to a ``c_bool`` pointer."""
    return x.ctypes.data_as(POINTER(c_bool))


def register_func(me, dll, name, argtypes=None, restype=None, caller_name=None):  # noqa: PLR0913
    """Register a C function from *dll* on object *me*."""
    if caller_name is None:
        caller_name = name
    setattr(me, caller_name, getattr(dll, name))
    func = getattr(me, caller_name)
    func.argtypes = argtypes if argtypes is not None else []
    func.restype = restype


def load_cdll(path):
    """Load a shared library from *path* relative to ``sys.path``."""
    return CDLL(Path(sys.path[-1]) / path, mode=RTLD_LOCAL)
