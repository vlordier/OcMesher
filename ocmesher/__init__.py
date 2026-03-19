# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""OcMesher: View-dependent octree-based mesh extraction for unbounded SDF scenes."""

import logging
from importlib.metadata import PackageNotFoundError, version

# Best practice for library packages: add NullHandler so that log records are
# discarded unless the application configures a handler.  This prevents
# "No handlers could be found for logger 'ocmesher'" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "CAMERA_DATA_STRIDE",
    "MLXOcMesher",
    "OcMesher",
    "RustOcMesher",
    "TorchOcMesher",
    "make_ocmesher",
    "make_rust_ocmesher",
]

try:
    __version__ = version("ocmesher")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"


def __getattr__(name: str):
    """Lazy-import backends so optional runtimes load only when needed."""
    lazy_imports: dict[str, tuple[str, str]] = {
        "CAMERA_DATA_STRIDE": ("._constants", "CAMERA_DATA_STRIDE"),
        "OcMesher": (".core", "OcMesher"),
        "TorchOcMesher": (".torch_core", "TorchOcMesher"),
        "RustOcMesher": (".rust_backend", "RustOcMesher"),
        "MLXOcMesher": (".mlx_core", "MLXOcMesher"),
        "make_rust_ocmesher": (".rust_backend", "make_rust_ocmesher"),
        "make_ocmesher": (".factory", "make_ocmesher"),
    }
    if name in lazy_imports:
        module_path, attr = lazy_imports[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
