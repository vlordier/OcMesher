# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Rust backend scaffold for OcMesher.

This module defines a Python-facing contract for a Rust implementation that is
compatible with Infinigen's runtime backend loading.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from ._constants import (
    DEVICE_VALUES,
    KERNEL_RUNTIME_VALUES,
    NUMPY_FLOAT64,
    SPEC_INFERENCE_ATOL,
    SPEC_INFERENCE_RTOL,
    STREAM_POLICY_VALUES,
)
from ._validation import (
    validate_bounds,
    validate_cameras,
    validate_kernels,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ._types import CamerasTuple

__all__ = [
    "RustOcMesher",
    "build_batched_sdf_kernels",
    "make_rust_ocmesher",
]

RESULT_ARITY = 2

# Probe points for native primitive spec inference.
# Five points: origin, three unit-axis tips, and the negative unit-z tip.
_SPEC_PROBE_POINTS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=NUMPY_FLOAT64,
)


def _normalize_rust_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in DEVICE_VALUES:
        return normalized
    if normalized.startswith("cuda:"):
        suffix = normalized.split(":", maxsplit=1)[1]
        if suffix.isdigit():
            return f"cuda:{int(suffix)}"
    msg = f"device must be one of: {', '.join(repr(v) for v in DEVICE_VALUES)}, or 'cuda:<index>'"
    raise ValueError(msg)


def _validate_kernel_runtime(kernel_runtime: str) -> str:
    if kernel_runtime not in KERNEL_RUNTIME_VALUES:
        msg = f"kernel_runtime must be one of: {', '.join(repr(v) for v in KERNEL_RUNTIME_VALUES)}"
        raise ValueError(msg)
    return kernel_runtime


# Device family constants for grouping.
_DEVICE_FAMILY_CPU = "cpu"
_DEVICE_FAMILY_CUDA = "cuda"
_DEVICE_FAMILY_MPS = "mps"
_DEVICE_FAMILY_MLX = "mlx"

# Preferred dtype for MPS (Apple Silicon) which requires float32.
_MPS_PREFERRED_DTYPE = "float32"

# Dict keys for SDF kernel output extraction.
_SDF_KEY_LOWER = "sdf"
_SDF_KEY_UPPER = "SDF"

# Maximum number of entries in the primitive inference cache.
_MAX_SPEC_CACHE_SIZE = 128

# Primitive type identifiers for native spec inference.
_SPEC_TYPE_SPHERE = "sphere"
_SPEC_TYPE_PLANE = "plane"

# Common coordinate vectors used in spec inference.
_ORIGIN_3D = [0.0, 0.0, 0.0]
_UNIT_Z_3D = [0.0, 0.0, 1.0]

# Error message for result arity validation.
_RESULT_ARITY_ERROR = "Rust backend must return (meshes, in_view_tags)"


def _device_family(device: str) -> str:
    normalized = device.strip().lower()
    if normalized.startswith("cuda"):
        return _DEVICE_FAMILY_CUDA
    if normalized == "mps":
        return _DEVICE_FAMILY_MPS
    if normalized == "mlx":
        return _DEVICE_FAMILY_MLX
    return _DEVICE_FAMILY_CPU


def _normalize_stream_policy(policy: str) -> str:
    if policy not in STREAM_POLICY_VALUES:
        msg = f"stream_policy must be one of: {', '.join(repr(v) for v in STREAM_POLICY_VALUES)}"
        raise ValueError(msg)
    return policy


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


def _mlx_device_capabilities() -> dict[str, bool]:
    """Return MLX device capabilities for Apple Silicon."""
    supports_mlx = False
    if sys.modules.get("mlx") is not None:
        # MLX is available - Apple Silicon with MLX supports GPU acceleration
        supports_mlx = True

    return {
        "supports_mlx": supports_mlx,
        "supports_cpu": True,
    }


def _extract_sdf(output: Any) -> np.ndarray[Any, Any]:
    if isinstance(output, dict):
        if _SDF_KEY_LOWER in output:
            return np.asarray(output[_SDF_KEY_LOWER])
        if _SDF_KEY_UPPER in output:
            return np.asarray(output[_SDF_KEY_UPPER])
        msg = "kernel output dict must contain 'sdf' or 'SDF'"
        raise KeyError(msg)
    return np.asarray(output)


def build_batched_sdf_kernels(
    kernels: list[Any],
    *,
    batch_size: int | None = None,
) -> list[Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]]:
    """Build SDF callables that prefer ``evaluate_batch`` when available."""
    effective_batch_size = int(batch_size) if batch_size is not None else 0

    def _evaluate_batched(
        eval_batch: Callable[[np.ndarray[Any, Any]], Any],
        xyz: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        n_pts = len(xyz)
        first_end = min(effective_batch_size, n_pts)
        first_chunk = _extract_sdf(eval_batch(xyz[:first_end]))

        # Fast path: fixed-shape outputs (the common case for SDF vectors).
        out_shape = (n_pts, *first_chunk.shape[1:])
        out = np.empty(out_shape, dtype=first_chunk.dtype)
        out[:first_end] = first_chunk

        cursor = first_end
        while cursor < n_pts:
            end = min(cursor + effective_batch_size, n_pts)
            chunk = _extract_sdf(eval_batch(xyz[cursor:end]))
            if chunk.shape[1:] != first_chunk.shape[1:]:
                # Keep behavior correct if a custom kernel emits varying trailing shapes.
                fallback_chunks: list[np.ndarray[Any, Any]] = [first_chunk]
                offset = first_end
                while offset < cursor:
                    next_end = min(offset + effective_batch_size, n_pts)
                    fallback_chunks.append(_extract_sdf(eval_batch(xyz[offset:next_end])))
                    offset = next_end
                fallback_chunks.append(chunk)
                offset = end
                while offset < n_pts:
                    next_end = min(offset + effective_batch_size, n_pts)
                    fallback_chunks.append(_extract_sdf(eval_batch(xyz[offset:next_end])))
                    offset = next_end
                return np.concatenate(fallback_chunks, axis=0)
            out[cursor:end] = chunk
            cursor = end

        return out

    def _evaluate(kernel: Any, xyz: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        eval_batch = getattr(kernel, "evaluate_batch", None)
        if callable(eval_batch):
            if batch_size is not None and batch_size > 0 and len(xyz) > batch_size:
                return _evaluate_batched(eval_batch, xyz)
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
        probe = _SPEC_PROBE_POINTS
        eval_batch = getattr(kernel, "evaluate_batch", None)
        if callable(eval_batch):
            values = np.asarray(_extract_sdf(eval_batch(probe)), dtype=NUMPY_FLOAT64).reshape(-1)
        else:
            values = np.asarray(kernel(probe), dtype=NUMPY_FLOAT64).reshape(-1)
        if values.shape[0] != probe.shape[0]:
            return None

        # Sphere centered at origin: f(x)=||x||-r (same value on unit axes).
        if np.allclose(values[1:4], values[1], atol=SPEC_INFERENCE_ATOL) and np.isclose(values[1], values[4], atol=SPEC_INFERENCE_ATOL):
            radius = max(0.0, -float(values[0]))
            if np.isclose(values[1], 1.0 - radius, atol=SPEC_INFERENCE_RTOL):
                specs.append({"type": _SPEC_TYPE_SPHERE, "center": _ORIGIN_3D, "radius": radius})
                continue

        # Z-plane: f(x)=z-offset.
        if np.isclose(values[0], 0.0, atol=SPEC_INFERENCE_ATOL) and np.isclose(values[1], 0.0, atol=SPEC_INFERENCE_ATOL) and np.isclose(
            values[2], 0.0, atol=SPEC_INFERENCE_ATOL
        ):
            offset = -float(values[0])
            if np.isclose(values[3], 1.0 - offset, atol=SPEC_INFERENCE_RTOL) and np.isclose(values[4], -1.0 - offset, atol=SPEC_INFERENCE_RTOL):
                specs.append({"type": _SPEC_TYPE_PLANE, "normal": _UNIT_Z_3D, "offset": offset})
                continue

        return None

    return specs


def _split_sphere_plane_specs(
    specs: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    sphere: dict[str, Any] | None = None
    plane: dict[str, Any] | None = None
    for spec in specs:
        if spec.get("type") == _SPEC_TYPE_SPHERE and sphere is None:
            sphere = spec
        elif spec.get("type") == _SPEC_TYPE_PLANE and plane is None:
            plane = spec
    return sphere, plane


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
        use_primitive_inference: bool = True,
        kernel_runtime: str = "auto",
        backend: RustBackendProtocol | None = None,
    ):
        """Create a Rust-backed mesher with OcMesher-compatible arguments."""
        cam_poses, ks, hs, ws = validate_cameras(cast("CamerasTuple", cameras))
        self.cameras = (cam_poses, ks, hs, ws)
        self.bounds = validate_bounds(bounds)

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
        mlx_caps = _mlx_device_capabilities()
        if device is None:
            if caps["supports_cuda"]:
                self.device = _DEVICE_FAMILY_CUDA
            elif caps["supports_mps"]:
                self.device = _DEVICE_FAMILY_MPS
            elif mlx_caps["supports_mlx"]:
                self.device = _DEVICE_FAMILY_MLX
            else:
                self.device = _DEVICE_FAMILY_CPU
        else:
            self.device = _normalize_rust_device(device)
        self._device_family = _device_family(self.device)

        self.dtype: str | None
        # MLX and MPS both require float32 on Apple Silicon
        if dtype is None and self._device_family in (_DEVICE_FAMILY_MPS, _DEVICE_FAMILY_MLX):
            self.dtype = _MPS_PREFERRED_DTYPE
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

        if stream_policy == "auto" and self._device_family not in (_DEVICE_FAMILY_CUDA, _DEVICE_FAMILY_MPS, _DEVICE_FAMILY_MLX):
            self.stream_policy = "sync"
        else:
            self.stream_policy = stream_policy
        self.stream_policy = _normalize_stream_policy(self.stream_policy)
        self.use_primitive_inference = use_primitive_inference
        self.kernel_runtime = _validate_kernel_runtime(kernel_runtime)

        self._device_caps = caps
        self._mlx_caps = mlx_caps
        self._backend = backend
        self._inferred_specs_cache: dict[tuple[int, ...], list[dict[str, Any]] | None] = {}

    def capabilities(self) -> dict[str, Any]:
        """Return runtime capability flags used by Infinigen integration."""
        caps = {
            "supports_cpu": True,
            "supports_cuda": self._device_caps["supports_cuda"],
            "supports_mps": self._device_caps["supports_mps"],
            "supports_mlx": self._mlx_caps["supports_mlx"],
            "preferred_dtype": _MPS_PREFERRED_DTYPE,
            "max_batch": self.max_batch,
            "max_batch_mps": self.max_batch if self._device_family in (_DEVICE_FAMILY_MPS, _DEVICE_FAMILY_MLX) else None,
            "supports_async": self._device_family in (_DEVICE_FAMILY_CUDA, _DEVICE_FAMILY_MPS, _DEVICE_FAMILY_MLX),
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
        validate_kernels(kernels)
        if self._backend is None:
            msg = (
                "Rust backend bridge is not configured. Provide backend=... when "
                "constructing RustOcMesher or install the ocmesher_rust extension."
            )
            raise RuntimeError(msg)

        kernels_list = list(kernels)
        specs: list[dict[str, Any]] | None = None
        if self.use_primitive_inference:
            cache_key = tuple(id(kernel) for kernel in kernels_list)
            specs = self._inferred_specs_cache.get(cache_key)
            if specs is None and cache_key not in self._inferred_specs_cache:
                specs = _infer_native_primitive_specs(kernels_list)
                if len(self._inferred_specs_cache) > _MAX_SPEC_CACHE_SIZE:
                    self._inferred_specs_cache.clear()
                self._inferred_specs_cache[cache_key] = specs

        if specs is not None:
            # Choose native/tch scene extraction according to runtime policy.
            sphere_spec, plane_spec = _split_sphere_plane_specs(specs)
            prefer_tch = self.kernel_runtime == "tch" or (
                self.kernel_runtime == "auto" and self._device_family in (_DEVICE_FAMILY_MPS, _DEVICE_FAMILY_CUDA)
            )
            if prefer_tch:
                if len(specs) == 1 and sphere_spec is not None and hasattr(self._backend, "extract_tch_sphere"):
                    result = self._backend.extract_tch_sphere(
                        radius=float(sphere_spec["radius"]),
                        center=list(sphere_spec["center"]),
                        device=self.device,
                    )
                elif len(specs) == 1 and plane_spec is not None and hasattr(self._backend, "extract_tch_plane"):
                    result = self._backend.extract_tch_plane(
                        offset=float(plane_spec["offset"]),
                        normal=list(plane_spec["normal"]),
                        device=self.device,
                    )
                elif (
                    len(specs) == 2
                    and sphere_spec is not None
                    and plane_spec is not None
                    and hasattr(self._backend, "extract_tch_sphere_plane")
                ):
                    result = self._backend.extract_tch_sphere_plane(
                        sphere_radius=float(sphere_spec["radius"]),
                        sphere_center=list(sphere_spec["center"]),
                        plane_offset=float(plane_spec["offset"]),
                        plane_normal=list(plane_spec["normal"]),
                        device=self.device,
                    )
                elif hasattr(self._backend, "extract_tch_scene"):
                    try:
                        result = self._backend.extract_tch_scene(specs, device=self.device)
                    except TypeError:
                        # Some backends expose extract_tch_scene without a device kwarg.
                        result = self._backend.extract_tch_scene(specs)
                elif hasattr(self._backend, "extract_native_scene"):
                    result = self._backend.extract_native_scene(specs)
                else:
                    sdf_kernels = build_batched_sdf_kernels(
                        kernels_list,
                        batch_size=self._effective_batch_size,
                    )
                    result = self._backend.extract_meshes(sdf_kernels)
            elif len(specs) == 1 and sphere_spec is not None and hasattr(self._backend, "extract_native_sphere"):
                result = self._backend.extract_native_sphere(
                    radius=float(sphere_spec["radius"]),
                    center=list(sphere_spec["center"]),
                )
            elif len(specs) == 1 and plane_spec is not None and hasattr(self._backend, "extract_native_plane"):
                result = self._backend.extract_native_plane(
                    offset=float(plane_spec["offset"]),
                    normal=list(plane_spec["normal"]),
                )
            elif (
                len(specs) == 2
                and sphere_spec is not None
                and plane_spec is not None
                and hasattr(self._backend, "extract_native_sphere_plane")
            ):
                result = self._backend.extract_native_sphere_plane(
                    sphere_radius=float(sphere_spec["radius"]),
                    sphere_center=list(sphere_spec["center"]),
                    plane_offset=float(plane_spec["offset"]),
                    plane_normal=list(plane_spec["normal"]),
                )
            elif hasattr(self._backend, "extract_native_scene"):
                result = self._backend.extract_native_scene(specs)
            elif hasattr(self._backend, "extract_tch_scene"):
                result = self._backend.extract_tch_scene(specs)
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
            raise TypeError(_RESULT_ARITY_ERROR)
        return result[0], result[1]

    def close(self):
        """Clean up resources held by the Rust backend."""
        # Signal to the Rust backend that this instance is being destroyed.
        # The Rust backend's Drop implementation handles cleanup.


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

    rust_ocmesher_only = {
        "device",
        "dtype",
        "max_batch",
        "batch_size",
        "sdf_batch_size",
        "stream_policy",
        "use_primitive_inference",
        "kernel_runtime",
    }
    backend_kwargs = {k: v for k, v in kwargs.items() if k not in rust_ocmesher_only}
    wrapper_kwargs = {k: v for k, v in kwargs.items() if k in rust_ocmesher_only}

    # Fail fast on invalid wrapper device strings before constructing the backend.
    device = wrapper_kwargs.get("device")
    if device is not None:
        if not isinstance(device, str):
            msg = "device must be a string"
            raise TypeError(msg)
        wrapper_kwargs["device"] = _normalize_rust_device(device)

    kernel_runtime = wrapper_kwargs.get("kernel_runtime")
    if kernel_runtime is not None:
        if not isinstance(kernel_runtime, str):
            msg = "kernel_runtime must be a string"
            raise TypeError(msg)
        wrapper_kwargs["kernel_runtime"] = _validate_kernel_runtime(kernel_runtime)

    # Keep backend defaults aligned with RustOcMesher/OcMesher-compatible defaults.
    backend_defaults = {
        "pixels_per_cube": 8,
        "inv_scale": 10,
        "min_dist": 1,
        "memory_limit_mb": 1000,
        "bisection_iters": 15,
        "bisection_tol": 0.0,
        "enclosed": True,
        "simplify_occluded": True,
        "visible_relax_iter": 2,
        "coarse_count": 500000,
    }
    for k, v in backend_defaults.items():
        backend_kwargs.setdefault(k, v)

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
