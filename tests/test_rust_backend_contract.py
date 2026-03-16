"""Tests for the Rust backend scaffold contract."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocmesher.rust_backend import RustOcMesher, build_batched_sdf_kernels, make_rust_ocmesher


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
