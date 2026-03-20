"""Tests for ``ocmesher.factory`` backend selection logic."""

from __future__ import annotations

import sys
import types

import pytest


def _sample_inputs(sample_camera_pose, sample_intrinsics):
    cameras = ([sample_camera_pose], [sample_intrinsics], [480], [640])
    bounds = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    return cameras, bounds


class TestMakeOcMesherBackendSelection:
    """Unit-test backend routing without requiring native runtimes."""

    def test_cpp_backend_returns_ocmesher(self, sample_camera_pose, sample_intrinsics, monkeypatch):
        from ocmesher import factory

        class DummyCpp:
            def __init__(self, cameras, bounds, **kwargs):
                self.cameras = cameras
                self.bounds = bounds
                self.kwargs = kwargs

        monkeypatch.setattr(factory, "OcMesher", DummyCpp)

        cameras, bounds = _sample_inputs(sample_camera_pose, sample_intrinsics)
        mesher = factory.make_ocmesher(cameras, bounds, backend="cpp")

        assert isinstance(mesher, DummyCpp)
        assert mesher.cameras == cameras
        assert mesher.bounds == bounds

    def test_default_backend_is_cpp(self, sample_camera_pose, sample_intrinsics, monkeypatch):
        from ocmesher import factory

        class DummyCpp:
            def __init__(self, _cameras, _bounds, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr(factory, "OcMesher", DummyCpp)

        cameras, bounds = _sample_inputs(sample_camera_pose, sample_intrinsics)
        mesher = factory.make_ocmesher(cameras, bounds)

        assert isinstance(mesher, DummyCpp)

    def test_cpp_backend_forwards_kwargs(self, sample_camera_pose, sample_intrinsics, monkeypatch):
        from ocmesher import factory

        class DummyCpp:
            def __init__(self, _cameras, _bounds, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr(factory, "OcMesher", DummyCpp)

        cameras, bounds = _sample_inputs(sample_camera_pose, sample_intrinsics)
        mesher = factory.make_ocmesher(
            cameras,
            bounds,
            backend="cpp",
            pixels_per_cube=16.0,
            inv_scale=5.0,
        )

        assert mesher.kwargs["pixels_per_cube"] == 16.0
        assert mesher.kwargs["inv_scale"] == 5.0

    def test_unknown_backend_raises_valueerror(self, sample_camera_pose, sample_intrinsics):
        from ocmesher import factory

        cameras, bounds = _sample_inputs(sample_camera_pose, sample_intrinsics)
        with pytest.raises(ValueError, match="Unknown backend"):
            factory.make_ocmesher(cameras, bounds, backend="invalid_backend")


class TestMakeOcMesherTorchBackend:
    """Torch backend import and argument forwarding."""

    def test_torch_backend_uses_lazy_import(self, sample_camera_pose, sample_intrinsics, monkeypatch):
        from ocmesher import factory

        captured: dict[str, object] = {}

        class DummyTorch:
            def __init__(self, _cameras, _bounds, *, device=None, **kwargs):
                captured["device"] = device
                captured["kwargs"] = kwargs

        fake_module = types.ModuleType("ocmesher.torch_core")
        fake_module.TorchOcMesher = DummyTorch
        monkeypatch.setitem(sys.modules, "ocmesher.torch_core", fake_module)

        cameras, bounds = _sample_inputs(sample_camera_pose, sample_intrinsics)
        mesher = factory.make_ocmesher(cameras, bounds, backend="torch", device="cpu", use_compile=True)

        assert isinstance(mesher, DummyTorch)
        assert captured["device"] == "cpu"
        assert captured["kwargs"] == {"use_compile": True}


class TestMakeOcMesherRustBackend:
    """Rust backend import and argument forwarding."""

    def test_rust_backend_uses_lazy_import(self, sample_camera_pose, sample_intrinsics, monkeypatch):
        from ocmesher import factory

        calls: list[dict[str, object]] = []

        def dummy_make_rust_ocmesher(_cameras, _bounds, **kwargs):
            calls.append(kwargs)
            return {"backend": "rust", "kwargs": kwargs}

        fake_module = types.ModuleType("ocmesher.rust_backend")
        fake_module.make_rust_ocmesher = dummy_make_rust_ocmesher
        monkeypatch.setitem(sys.modules, "ocmesher.rust_backend", fake_module)

        cameras, bounds = _sample_inputs(sample_camera_pose, sample_intrinsics)
        result = factory.make_ocmesher(cameras, bounds, backend="rust", device="mps", enclosed=False)

        assert result["backend"] == "rust"
        assert calls[-1] == {"device": "mps", "enclosed": False}

    def test_rust_backend_without_device(self, sample_camera_pose, sample_intrinsics, monkeypatch):
        from ocmesher import factory

        calls: list[dict[str, object]] = []

        def dummy_make_rust_ocmesher(_cameras, _bounds, **kwargs):
            calls.append(kwargs)
            return {"backend": "rust"}

        fake_module = types.ModuleType("ocmesher.rust_backend")
        fake_module.make_rust_ocmesher = dummy_make_rust_ocmesher
        monkeypatch.setitem(sys.modules, "ocmesher.rust_backend", fake_module)

        cameras, bounds = _sample_inputs(sample_camera_pose, sample_intrinsics)
        factory.make_ocmesher(cameras, bounds, backend="rust", coarse_count=123)

        assert calls[-1]["coarse_count"] == 123
        assert calls[-1]["device"] is None


class TestMakeOcMesherMlxBackend:
    """MLX backend import and argument forwarding."""

    def test_mlx_backend_uses_lazy_import(self, sample_camera_pose, sample_intrinsics, monkeypatch):
        from ocmesher import factory

        captured: dict[str, object] = {}

        class DummyMlx:
            def __init__(self, _cameras, _bounds, *, device=None, **kwargs):
                captured["device"] = device
                captured["kwargs"] = kwargs

        fake_module = types.ModuleType("ocmesher.mlx_core")
        fake_module.MLXOcMesher = DummyMlx
        monkeypatch.setitem(sys.modules, "ocmesher.mlx_core", fake_module)

        cameras, bounds = _sample_inputs(sample_camera_pose, sample_intrinsics)
        mesher = factory.make_ocmesher(cameras, bounds, backend="mlx", device="mps")

        assert isinstance(mesher, DummyMlx)
        assert captured["device"] == "mps"

    def test_mlx_backend_without_device(self, sample_camera_pose, sample_intrinsics, monkeypatch):
        from ocmesher import factory

        calls: list[dict[str, object]] = []

        def dummy_make_mlx_ocmesher(_cameras, _bounds, **kwargs):
            calls.append(kwargs)
            return {"backend": "mlx"}

        fake_module = types.ModuleType("ocmesher.mlx_core")
        fake_module.MLXOcMesher = lambda *a, **k: dummy_make_mlx_ocmesher(*a, **k)
        monkeypatch.setitem(sys.modules, "ocmesher.mlx_core", fake_module)

        cameras, bounds = _sample_inputs(sample_camera_pose, sample_intrinsics)
        result = factory.make_ocmesher(cameras, bounds, backend="mlx", enclosed=False)

        assert result["backend"] == "mlx"
        assert calls[-1]["enclosed"] is False
