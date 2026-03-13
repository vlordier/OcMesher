# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

from __future__ import annotations

import ctypes
import sys
from ctypes import CDLL, POINTER, RTLD_LOCAL, c_bool, c_double, c_float, c_int32
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


# note: size of x should not exceed maximum
def AsInt(x: npt.NDArray[np.int32]) -> ctypes.Array[c_int32]:
    return x.ctypes.data_as(POINTER(c_int32))


def AsDouble(x: npt.NDArray[np.float64]) -> ctypes.Array[c_double]:
    return x.ctypes.data_as(POINTER(c_double))


def AsFloat(x: npt.NDArray[np.float32]) -> ctypes.Array[c_float]:
    return x.ctypes.data_as(POINTER(c_float))


def AsBool(x: npt.NDArray[np.bool_]) -> ctypes.Array[c_bool]:
    return x.ctypes.data_as(POINTER(c_bool))


def register_func(
    me: Any,
    dll: CDLL,
    name: str,
    argtypes: list[Any] | None = None,
    restype: type[ctypes._SimpleCData[Any]] | None = None,
    caller_name: str | None = None,
) -> None:
    if argtypes is None:
        argtypes = []
    if caller_name is None:
        caller_name = name
    setattr(me, caller_name, getattr(dll, name))
    func = getattr(me, caller_name)
    func.argtypes = argtypes
    func.restype = restype


def load_cdll(path: str) -> CDLL:
    return CDLL(Path(sys.path[-1]) / path, mode=RTLD_LOCAL)
