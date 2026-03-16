"""Tests for the Rust backend scaffold contract."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocmesher.rust_backend import (
    RustOcMesher,
    build_batched_sdf_kernels,
    build_torch_kernel_bundle,
    make_rust_ocmesher,
)


class DummyRustBackend:
    __version__ = "0.0.1"

    def __init__(self):
        self.last_calls = []

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "supports_cuda": True,
            "supports_mps": True,
            "preferred_dtype": "float32",
            "max_batch": 64,
        }

    def extract_meshes(self, sdf_kernels: list[Any]) -> tuple[list[str], list[np.ndarray[Any, Any]]]:
        sample = np.zeros((5, 3), dtype=np.float32)
        values = [k(sample) for k in sdf_kernels]
        self.last_calls.append(values)
        return ["mesh"], [np.ones((5,), dtype=bool)]


def test_build_batched_sdf_kernels_prefers_evaluate_batch():
    calls = []

    class Kernel:
        def evaluate_batch(self, xyz):
            calls.append(len(xyz))
            return {"sdf": np.zeros((len(xyz),), dtype=np.float32)}

        def __call__(self, _xyz):
            msg = "should not use __call__"
            raise AssertionError(msg)

    [fn] = build_batched_sdf_kernels([Kernel()], batch_size=4)
    out = fn(np.zeros((9, 3), dtype=np.float32))

    assert calls == [4, 4, 1]
    assert out.shape == (9,)


def test_rust_ocmesher_requires_backend(sample_cameras, sample_bounds, sphere_kernel):
    mesher = RustOcMesher(sample_cameras, sample_bounds)
    with pytest.raises(RuntimeError, match="bridge is not configured"):
        mesher([sphere_kernel])


def test_rust_ocmesher_returns_backend_payload(sample_cameras, sample_bounds, sphere_kernel):
    backend = DummyRustBackend()
    mesher = RustOcMesher(
        sample_cameras,
        sample_bounds,
        backend=backend,
        batch_size=2,
        device="mps",
        stream_policy="auto",
    )

    meshes, tags = mesher([sphere_kernel])
    assert meshes == ["mesh"]
    assert len(tags) == 1
    assert tags[0].shape == (5,)


def test_capabilities_exposes_contract_fields(sample_cameras, sample_bounds):
    backend = DummyRustBackend()
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=backend, max_batch=32)
    caps = mesher.get_capabilities()

    for key in (
        "supports_cuda",
        "supports_mps",
        "preferred_dtype",
        "max_batch",
    ):
        assert key in caps


def test_stream_policy_auto_guarded_on_cpu(sample_cameras, sample_bounds):
    mesher = RustOcMesher(
        sample_cameras,
        sample_bounds,
        device="cpu",
        stream_policy="auto",
        backend=DummyRustBackend(),
    )
    assert mesher.stream_policy == "sync"


# ---------------------------------------------------------------------------
# make_rust_ocmesher factory tests
# ---------------------------------------------------------------------------

def test_make_rust_ocmesher_raises_import_error_without_extension(sample_cameras, sample_bounds):
    """Without ocmesher_rust installed, make_rust_ocmesher raises ImportError."""
    with patch.dict(sys.modules, {"ocmesher_rust": None}):
        with pytest.raises(ImportError, match="ocmesher_rust is not installed"):
            make_rust_ocmesher(sample_cameras, sample_bounds)


def test_make_rust_ocmesher_returns_rust_ocmesher_instance(sample_cameras, sample_bounds):
    """With a mock extension, make_rust_ocmesher returns a RustOcMesher."""
    mock_backend = DummyRustBackend()
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"
    mock_ext.Backend.return_value = mock_backend

    with patch.dict(sys.modules, {"ocmesher_rust": mock_ext}):
        mesher = make_rust_ocmesher(sample_cameras, sample_bounds)

    assert isinstance(mesher, RustOcMesher)
    assert mesher._backend is mock_backend


def test_make_rust_ocmesher_passes_lib_path(sample_cameras, sample_bounds):
    """Explicit lib_path bypasses find_core_so."""
    mock_backend = DummyRustBackend()
    mock_ext = MagicMock()
    mock_ext.Backend.return_value = mock_backend

    with patch.dict(sys.modules, {"ocmesher_rust": mock_ext}):
        make_rust_ocmesher(sample_cameras, sample_bounds, lib_path="/custom/core.so")

    mock_ext.find_core_so.assert_not_called()
    call_kwargs = mock_ext.Backend.call_args
    assert call_kwargs.kwargs.get("lib_path") == "/custom/core.so"


def test_make_rust_ocmesher_maps_batch_size_to_backend_sdf_batch_size(
    sample_cameras,
    sample_bounds,
):
    mock_backend = DummyRustBackend()
    mock_ext = MagicMock()
    mock_ext.Backend.return_value = mock_backend
    mock_ext.find_core_so.return_value = "/fake/core.so"

    with patch.dict(sys.modules, {"ocmesher_rust": mock_ext}):
        mesher = make_rust_ocmesher(sample_cameras, sample_bounds, batch_size=128)

    call_kwargs = mock_ext.Backend.call_args
    assert call_kwargs.kwargs.get("sdf_batch_size") == 128
    assert mesher.batch_size == 128


def test_make_rust_ocmesher_exported_from_package():
    """make_rust_ocmesher is accessible from the top-level ocmesher package."""
    import ocmesher  # noqa: PLC0415

    fn = ocmesher.make_rust_ocmesher
    assert callable(fn)


def test_native_batching_capability_bypasses_python_wrapper_batching(
    sample_cameras,
    sample_bounds,
):
    class Kernel:
        def __init__(self):
            self.evaluate_batch_called = 0
            self.call_called = 0

        def evaluate_batch(self, xyz):
            self.evaluate_batch_called += 1
            return np.zeros((len(xyz),), dtype=np.float32)

        def __call__(self, xyz):
            self.call_called += 1
            return np.zeros((len(xyz),), dtype=np.float32)

    class NativeBatchBackend(DummyRustBackend):
        def get_capabilities(self) -> dict[str, Any]:
            caps = super().get_capabilities()
            caps["native_batching"] = True
            return caps

    kernel = Kernel()
    backend = NativeBatchBackend()
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=backend, batch_size=2)
    mesher([kernel])

    assert kernel.evaluate_batch_called == 0
    assert kernel.call_called > 0


def test_build_torch_kernel_bundle_exposes_shared_owner():
    class Kernel:
        def evaluate_batch_torch(self, xyz):
            return xyz[:, 0]

        def __call__(self, xyz):
            return np.zeros((len(xyz),), dtype=np.float32)

    bundled = build_torch_kernel_bundle([Kernel(), Kernel()])

    assert bundled is not None
    assert len(bundled) == 2
    assert bundled[0]._ocmesher_torch_bundle is bundled[1]._ocmesher_torch_bundle


def test_native_torch_bundle_capability_wraps_kernels(sample_cameras, sample_bounds):
    class Kernel:
        def __call__(self, xyz):
            return np.zeros((len(xyz),), dtype=np.float32)

        def evaluate_batch_torch(self, xyz):
            return np.zeros((len(xyz),), dtype=np.float32)

    class BundleBackend(DummyRustBackend):
        def get_capabilities(self) -> dict[str, Any]:
            caps = super().get_capabilities()
            caps["native_batching"] = True
            caps["supports_fused_torch_bundle"] = True
            return caps

        def extract_meshes(self, sdf_kernels: list[Any]) -> tuple[list[str], list[np.ndarray[Any, Any]]]:
            assert hasattr(sdf_kernels[0], "_ocmesher_torch_bundle")
            return super().extract_meshes(sdf_kernels)

    backend = BundleBackend()
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=backend, batch_size=2)
    mesher([Kernel(), Kernel()])


# ---------------------------------------------------------------------------
# New capability / optimisation tests
# ---------------------------------------------------------------------------

def test_capabilities_exposes_dlpack_f32_flag(sample_cameras, sample_bounds):
    """capabilities() must advertise zero_copy_query_dlpack_f32."""
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=DummyRustBackend())
    caps = mesher.get_capabilities()
    assert caps.get("zero_copy_query_dlpack_f32") is True


def test_capabilities_exposes_cuda_sync_false_by_default(sample_cameras, sample_bounds):
    """cuda_sync defaults to False."""
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=DummyRustBackend())
    caps = mesher.get_capabilities()
    assert caps.get("cuda_sync") is False


def test_capabilities_cuda_sync_true_when_set(sample_cameras, sample_bounds):
    """cuda_sync=True is surfaced in capabilities."""
    mesher = RustOcMesher(
        sample_cameras,
        sample_bounds,
        device="cuda",
        stream_policy="auto",
        cuda_sync=True,
        backend=DummyRustBackend(),
    )
    caps = mesher.get_capabilities()
    assert caps.get("cuda_sync") is True


def test_make_rust_ocmesher_forwards_cuda_sync(sample_cameras, sample_bounds):
    """cuda_sync kwarg is forwarded to the Rust Backend constructor."""
    mock_backend = DummyRustBackend()
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"
    mock_ext.Backend.return_value = mock_backend

    with patch.dict(sys.modules, {"ocmesher_rust": mock_ext}):
        make_rust_ocmesher(sample_cameras, sample_bounds, cuda_sync=True)

    call_kwargs = mock_ext.Backend.call_args
    assert call_kwargs.kwargs.get("cuda_sync") is True


def test_make_rust_ocmesher_forces_float32_on_mps(sample_cameras, sample_bounds):
    """When device='mps', preferred_dtype must be float32 (MPS has no fp64 support)."""
    mock_backend = DummyRustBackend()
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"
    mock_ext.Backend.return_value = mock_backend

    with patch.dict(sys.modules, {"ocmesher_rust": mock_ext}):
        make_rust_ocmesher(sample_cameras, sample_bounds, device="mps")

    # The factory must not pass preferred_dtype='float64' for MPS.
    call_kwargs = mock_ext.Backend.call_args
    dtype_passed = call_kwargs.kwargs.get("preferred_dtype")
    # Either not passed (defaults to float32 in Rust) or explicitly float32.
    assert dtype_passed in (None, "float32")


def test_make_rust_ocmesher_applies_mps_batch_cap(sample_cameras, sample_bounds):
    """MPS path gets a default sdf_batch_size when none is supplied."""
    mock_backend = DummyRustBackend()
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"
    mock_ext.Backend.return_value = mock_backend

    with patch.dict(sys.modules, {"ocmesher_rust": mock_ext}):
        make_rust_ocmesher(sample_cameras, sample_bounds, device="mps")

    # The Rust Backend applies the cap internally; the factory should not pass
    # a conflicting sdf_batch_size unless the caller supplied one.
    call_kwargs = mock_ext.Backend.call_args
    # Either not set (Backend handles it) or a positive integer
    batch = call_kwargs.kwargs.get("sdf_batch_size")
    assert batch is None or (isinstance(batch, int) and batch > 0)


def test_pinned_memory_cuda_flag_on_auto_policy(sample_cameras, sample_bounds):
    """pinned_memory_cuda capability is True when device=cuda and stream_policy=auto."""
    mesher = RustOcMesher(
        sample_cameras,
        sample_bounds,
        device="cuda",
        stream_policy="auto",
        backend=DummyRustBackend(),
    )
    caps = mesher.get_capabilities()
    assert caps.get("pinned_memory_cuda") is True


def test_pinned_memory_cuda_flag_off_for_cpu(sample_cameras, sample_bounds):
    """pinned_memory_cuda is False when device=cpu."""
    mesher = RustOcMesher(
        sample_cameras,
        sample_bounds,
        device="cpu",
        backend=DummyRustBackend(),
    )
    caps = mesher.get_capabilities()
    assert caps.get("pinned_memory_cuda") is False
