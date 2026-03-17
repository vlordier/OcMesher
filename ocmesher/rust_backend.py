# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Rust backend scaffold for OcMesher.

This module defines a Python-facing contract for a Rust implementation that is
compatible with Infinigen's runtime backend loading.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


RESULT_ARITY = 2


def _torch_device_capabilities() -> dict[str, bool]:
    supports_cuda = False
    supports_mps = False
    torch = sys.modules.get("torch")
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
            return np.asarray(output["sdf"])
        if "SDF" in output:
            return np.asarray(output["SDF"])
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

    batched_kernels: list[Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]] = []
    for kernel in kernels:
        def _wrapped(xyz: np.ndarray[Any, Any], *, _kernel: Any = kernel) -> np.ndarray[Any, Any]:
            return _evaluate(_kernel, xyz)

        batched_kernels.append(_wrapped)
    return batched_kernels


def _infer_native_primitive_specs(kernels: Sequence[Any]) -> list[dict[str, Any]] | None:
    """Infer native primitive specs from simple callable kernels.

    This keeps the Python wrapper compatible with compiled backends that no
    longer accept arbitrary callable kernels through ``extract_meshes``.
    """
    specs: list[dict[str, Any]] = []
    for kernel in kernels:
        probe = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        values = np.asarray(kernel(probe), dtype=np.float64).reshape(-1)
        if values.shape[0] != probe.shape[0]:
            return None

        # Sphere centered at origin: f(x)=||x||-r (same value on unit axes).
        if np.allclose(values[1:4], values[1], atol=1e-5) and np.isclose(values[1], values[4], atol=1e-5):
            radius = max(0.0, -float(values[0]))
            if np.isclose(values[1], 1.0 - radius, atol=1e-4):
                specs.append({"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": radius})
                continue

        # Z-plane: f(x)=z-offset.
        if np.isclose(values[0], 0.0, atol=1e-5) and np.isclose(values[1], 0.0, atol=1e-5) and np.isclose(
            values[2], 0.0, atol=1e-5
        ):
            offset = -float(values[0])
            if np.isclose(values[3], 1.0 - offset, atol=1e-4) and np.isclose(values[4], -1.0 - offset, atol=1e-4):
                specs.append({"type": "plane", "normal": [0.0, 0.0, 1.0], "offset": offset})
                continue

        return None

    return specs


class RustBackendProtocol(Protocol):
    """Protocol for Rust-backed mesh extractor bridge."""

    __version__: str

    def get_capabilities(self) -> dict[str, Any]:
        """Return backend capability flags and runtime hints."""

    def extract_meshes(self, sdf_kernels: list[Any]) -> tuple[Any, Any]:
        """Run extraction and return ``(meshes, in_view_tags)``."""


class RustOcMesher:
    """OcMesher-compatible Rust backend entry point.

    Constructor signature intentionally mirrors ``ocmesher.core.OcMesher`` plus
    optional runtime controls used by Infinigen.
    """

    __version__ = "0.1.0"

    def __init__(
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
        cam_poses, ks, hs, ws = _validate_cameras(cameras)
        self.cameras = (cam_poses, ks, hs, ws)
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

        self.dtype: str | None
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
            raise RuntimeError(msg)

        kernels_list = list(kernels)
        if hasattr(self._backend, "extract_native_scene"):
            specs = _infer_native_primitive_specs(kernels_list)
            if specs is not None:
                result = self._backend.extract_native_scene(specs)
            else:
                sdf_kernels = build_batched_sdf_kernels(
                    kernels_list,
                    batch_size=self._effective_batch_size,
                )
                result = self._backend.extract_meshes(sdf_kernels)
        else:
            sdf_kernels = build_batched_sdf_kernels(
                kernels_list,
                batch_size=self._effective_batch_size,
            )
            result = self._backend.extract_meshes(sdf_kernels)
        if len(result) != RESULT_ARITY:
            msg = "Rust backend must return (meshes, in_view_tags)"
            raise TypeError(msg)
        return result[0], result[1]


def make_rust_ocmesher(
    cameras: Sequence[Any],
    bounds: Any,
    *,
    lib_path: str | None = None,
    **kwargs: Any,
) -> RustOcMesher:
    """Create a :class:`RustOcMesher` using the compiled ``ocmesher_rust`` extension."""
    try:
        import ocmesher_rust  # type: ignore[import-not-found,import-untyped]
    except ImportError as exc:  # pragma: no cover - optional build artifact
        msg = (
            "ocmesher_rust is not installed.  Build the Rust extension with:\n"
            "  uv run maturin develop --release --manifest-path "
            "ocmesher-rust/crates/ocmesher-py/Cargo.toml"
        )
        raise ImportError(msg) from exc

    resolved_path: str = lib_path or ocmesher_rust.find_core_so()

    rust_ocmesher_only = {"device", "dtype", "max_batch", "batch_size", "sdf_batch_size", "stream_policy"}
    backend_kwargs = {k: v for k, v in kwargs.items() if k not in rust_ocmesher_only}
    wrapper_kwargs = {k: v for k, v in kwargs.items() if k in rust_ocmesher_only}

    cam_poses, ks, hs, ws = cameras
    cam_poses_flat = [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses]
    ks_flat = [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks]
    hs_float = [float(h) for h in hs]
    ws_float = [float(w) for w in ws]
    bounds_list = np.asarray(bounds, dtype=np.float64).tolist()

    backend = ocmesher_rust.Backend(
        lib_path=resolved_path,
        cameras=(cam_poses_flat, ks_flat, hs_float, ws_float),
        bounds=bounds_list,
        **backend_kwargs,
    )

    return RustOcMesher(cameras, bounds, backend=backend, **wrapper_kwargs)
