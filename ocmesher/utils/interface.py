# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import sys
from ctypes import CDLL, POINTER, RTLD_LOCAL, c_double, c_float, c_int32, c_bool
from pathlib import Path

import numpy as np
from numpy import ascontiguousarray as AC


# note: size of x should not exceed maximum
def _as_ctype(x, ctype):
    if not isinstance(x, np.ndarray):
        msg = f"Expected a numpy array, got {type(x).__name__}"
        raise TypeError(msg)
    return x.ctypes.data_as(POINTER(ctype))


def AsInt(x):
    return _as_ctype(x, c_int32)
def AsDouble(x):
    return _as_ctype(x, c_double)
def AsFloat(x):
    return _as_ctype(x, c_float)
def AsBool(x):
    return _as_ctype(x, c_bool)

def register_func(me, dll, name, argtypes=None, restype=None, caller_name=None):
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

def load_cdll(path):
    lib_path = Path(sys.path[-1]) / path
    if not lib_path.exists():
        msg = f"Shared library not found at {lib_path}. Run install.sh to build."
        raise FileNotFoundError(msg)
    try:
        return CDLL(str(lib_path), mode=RTLD_LOCAL)
    except OSError as e:
        msg = f"Failed to load shared library at {lib_path}: {e}"
        raise OSError(msg) from e