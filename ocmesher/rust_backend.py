# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Rust backend scaffold for OcMesher.

This module defines a Python-facing contract for a Rust implementation that is
compatible with Infinigen's runtime backend loading.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence, cast

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


RESULT_ARITY = 2


def _torch_device_capabilities() -> dict[str, bool]:
    supports_cuda = False
    supports_mps = False
    if torch is not None:
        supports_cuda = bool(torch.cuda.is_available())
        supports_mps = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    return {
        "supports_cuda": supports_cuda,
        "supports_mps": supports_mps,
        "supports_cpu": True,
    }


def _extract_sdf(output: Any) -> np.ndarray[Any, Any]:
    if isinstance(output, dict):
        if "sdf" in output:
            sdf_value = cast(Any, output["sdf"])
            return np.asarray(sdf_value)
        if "SDF" in output:
            sdf_value = cast(Any, output["SDF"])
            return np.asarray(sdf_value)
        msg = "kernel output dict must contain 'sdf' or 'SDF'"
        raise KeyError(msg)
    return np.asarray(output)


def _validate_cameras(
    cameras: Sequence[Any],
) -> tuple[
    list[np.ndarray[Any, Any]],
    list[np.ndarray[Any, Any]],
    list[int],
    list[int],
]:
    if not isinstance(cameras, (tuple, list)) or len(cameras) != 4:  # noqa: PLR2004
        msg = "cameras must be a tuple/list of (cam_poses, Ks, Hs, Ws)"
        raise ValueError(msg)
    cam_poses_raw, ks_raw, hs_raw, ws_raw = cameras
    if len(cam_poses_raw) == 0:
        msg = "At least one camera is required"
        raise ValueError(msg)
    if not (len(cam_poses_raw) == len(ks_raw) == len(hs_raw) == len(ws_raw)):
        msg = "Camera arrays must all have the same length"
        raise ValueError(msg)

    cam_poses = [np.asarray(p, dtype=np.float64) for p in cam_poses_raw]
    ks = [np.asarray(k, dtype=np.float64) for k in ks_raw]
    hs = [int(h) for h in hs_raw]
    ws = [int(w) for w in ws_raw]

    for i, pose in enumerate(cam_poses):
        if pose.shape != (4, 4):
            msg = f"cam_poses[{i}] must be 4x4, got shape {pose.shape}"
            raise ValueError(msg)
    for i, k in enumerate(ks):
        if k.shape != (3, 3):
            msg = f"Ks[{i}] must be 3x3, got shape {k.shape}"
            raise ValueError(msg)

    return cam_poses, ks, hs, ws


def _validate_bounds(bounds: Any) -> np.ndarray[Any, Any]:
    bounds_np = np.asarray(bounds, dtype=np.float64)
    if bounds_np.shape != (6,):
        msg = "bounds must have 6 elements [x_min, x_max, y_min, y_max, z_min, z_max]"
        raise ValueError(msg)
    if not np.all(np.isfinite(bounds_np)):
        msg = "bounds must contain finite values"
        raise ValueError(msg)
    for axis in range(3):
        if bounds_np[axis * 2] >= bounds_np[axis * 2 + 1]:
            msg = "bounds min must be less than max on each axis"
            raise ValueError(msg)
    return bounds_np


def _validate_kernels(kernels: Sequence[Any]) -> None:
    if not isinstance(kernels, (list, tuple)) or len(kernels) == 0:
        msg = "kernels must be a non-empty list/tuple of callables"
        raise ValueError(msg)
    for i, kernel in enumerate(kernels):
        if not callable(kernel):
            msg = f"kernels[{i}] must be callable"
            raise TypeError(msg)


def build_batched_sdf_kernels(
    kernels: list[Any],
    *,
    batch_size: int | None = None,
) -> list[Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]]:
    """Build SDF callables that prefer ``evaluate_batch`` when available."""

    def _evaluate(kernel: Any, xyz: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        eval_batch = getattr(kernel, "evaluate_batch", None)
        if callable(eval_batch):
            if batch_size is not None and batch_size > 0 and len(xyz) > batch_size:
                chunks: list[np.ndarray[Any, Any]] = []
                for start in range(0, len(xyz), batch_size):
                    chunk_xyz = xyz[start : start + batch_size]
                    chunks.append(_extract_sdf(eval_batch(chunk_xyz)))
                return np.concatenate(chunks, axis=0)
            return _extract_sdf(eval_batch(xyz))
        return _extract_sdf(kernel(xyz))

    return [(lambda x, k0=k: _evaluate(k0, x)) for k in kernels]


class RustBackendProtocol(Protocol):
    """Protocol for Rust-backed mesh extractor bridge."""

    __version__: str

    def get_capabilities(self) -> dict[str, Any]:
        """Return backend capability flags and runtime hints."""
        ...

    def extract_meshes(self, sdf_kernels: list[Any]) -> tuple[Any, Any]:
        """Run extraction and return ``(meshes, in_view_tags)``."""
        ...


class RustOcMesher:
    """OcMesher-compatible Rust backend entry point.

    Constructor signature intentionally mirrors ``ocmesher.core.OcMesher`` plus
    optional runtime controls used by Infinigen.
    """

    __version__ = "0.1.0"

    def __init__(  # noqa: PLR0913, FBT001, FBT002
        self,
        cameras: Sequence[Any],
        bounds: Any,
        pixels_per_cube: int = 8,
        inv_scale: int = 10,
        min_dist: int = 1,
        memory_limit_mb: int = 1000,
        bisection_iters: int = 15,
        bisection_tol: float = 0.0,
        enclosed: bool = True,
        simplify_occluded: bool = True,
        visible_relax_iter: int = 2,
        coarse_count: int = 500000,
        *,
        device: str | None = None,
        dtype: str | None = None,
        max_batch: int | None = None,
        batch_size: int | None = None,
        sdf_batch_size: int | None = None,
        stream_policy: str = "sync",
        backend: RustBackendProtocol | None = None,
    ):
        """Create a Rust-backed mesher with OcMesher-compatible arguments."""
        cam_poses, Ks, Hs, Ws = _validate_cameras(cameras)
        self.cameras = (cam_poses, Ks, Hs, Ws)
        self.bounds = _validate_bounds(bounds)

        self.pixels_per_cube = pixels_per_cube
        self.inv_scale = inv_scale
        self.min_dist = min_dist
        self.memory_limit_mb = memory_limit_mb
        self.bisection_iters = bisection_iters
        self.bisection_tol = bisection_tol
        self.enclosed = enclosed
        self.simplify_occluded = simplify_occluded
        self.visible_relax_iter = visible_relax_iter
        self.coarse_count = coarse_count

        caps = _torch_device_capabilities()
        if device is None:
            if caps["supports_cuda"]:
                self.device = "cuda"
            elif caps["supports_mps"]:
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if dtype is None and self.device == "mps":
            self.dtype = "float32"
        else:
            self.dtype = dtype

        self.max_batch = max_batch
        self.batch_size = batch_size
        self.sdf_batch_size = sdf_batch_size
        self._effective_batch_size = (
            sdf_batch_size
            if sdf_batch_size is not None
            else (batch_size if batch_size is not None else max_batch)
        )

        if stream_policy == "auto" and self.device not in ("cuda", "mps"):
            self.stream_policy = "sync"
        else:
            self.stream_policy = stream_policy

        self._device_caps = caps
        self._backend = backend

    def capabilities(self) -> dict[str, Any]:
        """Return runtime capability flags used by Infinigen integration."""
        caps = {
            "supports_cpu": True,
            "supports_cuda": self._device_caps["supports_cuda"],
            "supports_mps": self._device_caps["supports_mps"],
            "preferred_dtype": "float32",
            "max_batch": self.max_batch,
            "max_batch_mps": self.max_batch if self.device == "mps" else None,
            "supports_async": self.device in ("cuda", "mps"),
            "default_stream_policy": self.stream_policy,
            "version": self.__version__,
        }

        if self._backend is not None and hasattr(self._backend, "get_capabilities"):
            try:
                backend_caps = self._backend.get_capabilities()
                if isinstance(backend_caps, dict):
                    caps.update(backend_caps)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass

        return caps

    def get_capabilities(self) -> dict[str, Any]:
        """Alias used by Infinigen backend capability negotiation."""
        return self.capabilities()

    def __call__(self, kernels: Sequence[Any]) -> tuple[Any, Any]:
        """Execute extraction via configured Rust backend bridge."""
        _validate_kernels(kernels)
        if self._backend is None:
            msg = (
                "Rust backend bridge is not configured. Provide backend=... when "
                "constructing RustOcMesher or install the ocmesher_rust extension."
            )
            raise RuntimeError(
                msg
            )

        sdf_kernels = build_batched_sdf_kernels(
            list(kernels),
            batch_size=self._effective_batch_size,
        )

        result = cast(tuple[Any, Any] | list[Any], self._backend.extract_meshes(sdf_kernels))
        if len(result) != RESULT_ARITY:
            msg = "Rust backend must return (meshes, in_view_tags)"
            raise TypeError(msg)
        return result[0], result[1]


# ---------------------------------------------------------------------------
# Convenience factory: build a RustOcMesher backed by ocmesher_rust.Backend
# ---------------------------------------------------------------------------

def make_rust_ocmesher(
    cameras: Sequence[Any],
    bounds: Any,
    *,
    lib_path: str | None = None,
    **kwargs: Any,
) -> "RustOcMesher":
    """Create a :class:`RustOcMesher` using the compiled ``ocmesher_rust`` extension.

    Parameters
    ----------
    cameras:
        ``(cam_poses, Ks, Hs, Ws)`` — same format as :class:`ocmesher.OcMesher`.
    bounds:
        ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
    lib_path:
        Path to ``core.so``.  When *None* the extension's :func:`find_core_so`
        helper is used to locate the library automatically.
    **kwargs:
        Forwarded to both :class:`RustOcMesher` and ``ocmesher_rust.Backend``
        (``pixels_per_cube``, ``bisection_iters``, ``device``, ``stream_policy``, …).

    Returns
    -------
    RustOcMesher
        A fully configured mesher ready to call with SDF kernels.

    Raises
    ------
    ImportError
        If ``ocmesher_rust`` is not installed.  Build it with ``maturin develop``
        inside the ``ocmesher-rust/`` workspace.
    RuntimeError
        If ``core.so`` cannot be found and *lib_path* was not supplied.

    Examples
    --------
    .. code-block:: python

        mesher = make_rust_ocmesher(cameras, bounds)
        meshes, in_view_tags = mesher(sdf_kernels)
    """
    try:
        import ocmesher_rust  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional build artifact
        msg = (
            "ocmesher_rust is not installed.  Build the Rust extension with:\n"
            "  cd ocmesher-rust && maturin develop"
        )
        raise ImportError(msg) from exc

    resolved_path: str = lib_path or ocmesher_rust.find_core_so()

    # Split kwargs between Rust extension backend and Python wrapper.
    # `batch_size` and `dtype` are wrapper-only controls; the remaining runtime
    # knobs are passed to both layers so capability negotiation stays aligned.
    _wrapper_only = {"batch_size", "dtype"}
    _shared = {"device", "max_batch", "sdf_batch_size", "stream_policy"}
    backend_kwargs = {k: v for k, v in kwargs.items() if k not in _wrapper_only}
    wrapper_kwargs = {
        k: v for k, v in kwargs.items() if k in _wrapper_only or k in _shared
    }
    if "sdf_batch_size" not in backend_kwargs and "batch_size" in wrapper_kwargs:
        backend_kwargs["sdf_batch_size"] = wrapper_kwargs["batch_size"]

    # Pack camera poses/Ks as flat lists for Rust (it handles the SE(3) inversion).
    import numpy as _np  # noqa: PLC0415 — local import avoids top-level cost
    cam_poses, Ks, Hs, Ws = cameras
    cam_poses_flat = [_np.asarray(p, dtype=_np.float64).ravel().tolist() for p in cam_poses]
    ks_flat = [_np.asarray(k, dtype=_np.float64).ravel().tolist() for k in Ks]
    hs_float = [float(h) for h in Hs]
    ws_float = [float(w) for w in Ws]
    bounds_list = _np.asarray(bounds, dtype=_np.float64).tolist()

    backend = ocmesher_rust.Backend(
        lib_path=resolved_path,
        cameras=(cam_poses_flat, ks_flat, hs_float, ws_float),
        bounds=bounds_list,
        **backend_kwargs,
    )

    return RustOcMesher(cameras, bounds, backend=backend, **wrapper_kwargs)
