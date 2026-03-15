# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""OcMesher: View-dependent octree-based mesh extraction for unbounded SDF scenes."""

import logging

from .core import OcMesher
from .dll import CoreDLL
from .types import Bounds, CameraSet, MesherConfig

# Best practice for library packages: add NullHandler so that log records are
# discarded unless the application configures a handler.  This prevents
# "No handlers could be found for logger 'ocmesher'" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["Bounds", "CameraSet", "CoreDLL", "MesherConfig", "OcMesher"]
__version__ = "1.0.0"
