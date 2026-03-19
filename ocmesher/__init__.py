# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""OcMesher: View-dependent octree-based mesh extraction for unbounded SDF scenes."""

import logging
from importlib.metadata import PackageNotFoundError, version

# Best practice for library packages: add NullHandler so that log records are
# discarded unless the application configures a handler.  This prevents
# "No handlers could be found for logger 'ocmesher'" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["CAMERA_DATA_STRIDE", "OcMesher", "RustOcMesher", "TorchOcMesher", "MLXOcMesher", "make_ocmesher", "make_rust_ocmesher"]

try:
    __version__ = version("ocmesher")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"


def __getattr__(name: str):
    """Lazy-import backends so optional runtimes load only when needed."""
    if name == "CAMERA_DATA_STRIDE":
        from ._constants import CAMERA_DATA_STRIDE

        return CAMERA_DATA_STRIDE
    if name == "OcMesher":
        from .core import OcMesher

        return OcMesher
    if name == "TorchOcMesher":
        from .torch_core import TorchOcMesher

        return TorchOcMesher
    if name == "RustOcMesher":
        from .rust_backend import RustOcMesher

        return RustOcMesher
    if name == "MLXOcMesher":
        from .mlx_core import MLXOcMesher

        return MLXOcMesher
    if name == "make_rust_ocmesher":
        from .rust_backend import make_rust_ocmesher

        return make_rust_ocmesher
    if name == "make_ocmesher":
        from .factory import make_ocmesher

        return make_ocmesher
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
