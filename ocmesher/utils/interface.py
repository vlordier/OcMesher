# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import os
import sys
from ctypes import CDLL, POINTER, RTLD_LOCAL, c_double, c_float, c_int32, c_bool
from typing import Any, Optional

import numpy as np
from numpy import ascontiguousarray as AC


# note: size of x should not exceed maximum
def AsInt(x: np.ndarray) -> "POINTER(c_int32)":
    return x.ctypes.data_as(POINTER(c_int32))


def AsDouble(x: np.ndarray) -> "POINTER(c_double)":
    return x.ctypes.data_as(POINTER(c_double))


def AsFloat(x: np.ndarray) -> "POINTER(c_float)":
    return x.ctypes.data_as(POINTER(c_float))


def AsBool(x: np.ndarray) -> "POINTER(c_bool)":
    return x.ctypes.data_as(POINTER(c_bool))


def register_func(me: Any, dll: CDLL, name: str, argtypes: Optional[list] = None, restype: Any = None, caller_name: Optional[str] = None) -> None:
    if argtypes is None:
        argtypes = []
    if caller_name is None:
        caller_name = name
    setattr(me, caller_name, getattr(dll, name))
    func = getattr(me, caller_name)
    func.argtypes = argtypes
    func.restype = restype


def load_cdll(path: str) -> CDLL:
    return CDLL(os.path.join(sys.path[-1], path), mode=RTLD_LOCAL)