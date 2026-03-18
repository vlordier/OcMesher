# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""User-facing backend factory for selecting OcMesher runtime implementations."""

from __future__ import annotations

from typing import Any

from .core import OcMesher

__all__ = ["make_ocmesher"]


def make_ocmesher(
    cameras: Any,
    bounds: Any,
    *,
    backend: str = "cpp",
    device: str | None = None,
    **kwargs: Any,
) -> Any:
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
            Rust backend supports ``kernel_runtime`` (``"auto"|"native"|"tch"``).

    Returns:
        Backend instance implementing ``__call__(kernels)``.

    Raises:
        ValueError: If backend is not one of ``cpp``, ``torch``, or ``rust``.
        ImportError: If the required backend module is not installed.
    """
    if backend == "cpp":
        return OcMesher(cameras, bounds, **kwargs)

    if backend == "torch":
        from .torch_core import TorchOcMesher

        return TorchOcMesher(cameras, bounds, device=device, **kwargs)

    if backend == "rust":
        from .rust_backend import make_rust_ocmesher

        if device is None:
            return make_rust_ocmesher(cameras, bounds, **kwargs)
        return make_rust_ocmesher(cameras, bounds, device=device, **kwargs)

    msg = f"Unknown backend {backend!r}. Expected one of: 'cpp', 'torch', 'rust'."
    raise ValueError(msg)
