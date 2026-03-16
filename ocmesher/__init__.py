# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""OcMesher: View-dependent octree-based mesh extraction for unbounded SDF scenes."""

import logging

from .core import OcMesher

# Best practice for library packages: add NullHandler so that log records are
# discarded unless the application configures a handler.  This prevents
# "No handlers could be found for logger 'ocmesher'" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["OcMesher", "TorchOcMesher", "RustOcMesher"]
__version__ = "1.0.0"


def __getattr__(name: str):
    """Lazy-import TorchOcMesher so torch is only required when used."""
    if name == "TorchOcMesher":
        from .torch_core import TorchOcMesher

        return TorchOcMesher
    if name == "RustOcMesher":
        from .rust_backend import RustOcMesher

        return RustOcMesher
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
