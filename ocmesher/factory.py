# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""User-facing backend factory for selecting OcMesher runtime implementations."""

from __future__ import annotations

from typing import Any, Literal

from .core import OcMesher
from .rust_backend import make_rust_ocmesher
from .torch_core import TorchOcMesher

BackendName = Literal["cpp", "torch", "rust"]


def make_ocmesher(
    cameras: Any,
    bounds: Any,
    *,
    backend: BackendName = "cpp",
    device: str | None = None,
    **kwargs: Any,
):
    """Create an OcMesher backend instance from a simple backend/device selector.

    Args:
        cameras: Camera tuple ``(cam_poses, Ks, Hs, Ws)``.
        bounds: Scene bounds ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
        backend: Backend family.
            - ``"cpp"``: original C++ backend via ``OcMesher``
            - ``"torch"``: PyTorch backend via ``TorchOcMesher``
            - ``"rust"``: Rust wrapper backend via ``make_rust_ocmesher``
        device: Optional device for ``torch``/``rust`` backends (e.g. ``cpu``, ``mps``, ``cuda``).
        **kwargs: Forwarded to the selected backend constructor/factory.

    Returns:
        Backend instance implementing ``__call__(kernels)``.

    Raises:
        ValueError: If backend is not one of ``cpp``, ``torch``, or ``rust``.
    """
    if backend == "cpp":
        return OcMesher(cameras, bounds, **kwargs)

    if backend == "torch":
        return TorchOcMesher(cameras, bounds, device=device, **kwargs)

    if backend == "rust":
        if device is None:
            return make_rust_ocmesher(cameras, bounds, **kwargs)
        return make_rust_ocmesher(cameras, bounds, device=device, **kwargs)

    msg = f"Unknown backend {backend!r}. Expected one of: 'cpp', 'torch', 'rust'."
    raise ValueError(msg)
