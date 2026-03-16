"""Tests for the Rust backend scaffold contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ocmesher.rust_backend import RustOcMesher, build_batched_sdf_kernels


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
