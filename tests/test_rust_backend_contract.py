"""Tests for the Rust backend scaffold contract."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocmesher.rust_backend import RustOcMesher, build_batched_sdf_kernels, make_rust_ocmesher


def _mesh_signature(mesh: Any, tag: np.ndarray[Any, Any]) -> dict[str, float | int]:
    return {
        "verts": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "verts_sum": float(mesh.vertices.sum()),
        "faces_sum": int(mesh.faces.sum()),
        "tag_true": int(np.count_nonzero(tag)),
    }


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


class DummyRustBackendWithScenes(DummyRustBackend):
    def __init__(self):
        super().__init__()
        self.extract_native_scene_calls = 0
        self.extract_tch_scene_calls = 0
        self.extract_tch_sphere_calls = 0
        self.extract_native_sphere_calls = 0

    def extract_native_scene(self, specs: list[dict[str, Any]]) -> tuple[list[str], list[np.ndarray[Any, Any]]]:
        self.extract_native_scene_calls += 1
        self.last_calls.append(("native", specs))
        return ["mesh"], [np.ones((5,), dtype=bool)]

    def extract_tch_scene(
        self, specs: list[dict[str, Any]], device: str | None = None
    ) -> tuple[list[str], list[np.ndarray[Any, Any]]]:
        self.extract_tch_scene_calls += 1
        self.last_calls.append(("tch", specs, device))
        return ["mesh"], [np.ones((5,), dtype=bool)]

    def extract_tch_sphere(
        self,
        radius: float = 1.0,
        center: list[float] | None = None,
        device: str | None = None,
    ) -> tuple[list[str], list[np.ndarray[Any, Any]]]:
        self.extract_tch_sphere_calls += 1
        self.last_calls.append(("tch_sphere", radius, center, device))
        return ["mesh"], [np.ones((5,), dtype=bool)]

    def extract_native_sphere(
        self,
        radius: float = 1.0,
        center: list[float] | None = None,
    ) -> tuple[list[str], list[np.ndarray[Any, Any]]]:
        self.extract_native_sphere_calls += 1
        self.last_calls.append(("native_sphere", radius, center))
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


def test_rust_ocmesher_rejects_invalid_cameras(sample_intrinsics, sample_bounds):
    bad_pose = np.eye(3)
    bad_cameras = ([bad_pose], [sample_intrinsics], [720], [1280])

    with pytest.raises(ValueError, match="4x4"):
        RustOcMesher(bad_cameras, sample_bounds)


def test_rust_ocmesher_rejects_invalid_bounds(sample_cameras):
    bad_bounds = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0]

    with pytest.raises(ValueError, match="x_min"):
        RustOcMesher(sample_cameras, bad_bounds)


def test_rust_ocmesher_rejects_non_callable_kernel(sample_cameras, sample_bounds):
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=DummyRustBackend())

    with pytest.raises(TypeError, match=r"kernels\[0\] must be callable"):
        mesher(["not-a-kernel"])


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


def test_rust_ocmesher_prefers_tch_scene_on_mps(sample_cameras, sample_bounds, sphere_kernel):
    backend = DummyRustBackendWithScenes()
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=backend, device="mps")

    meshes, tags = mesher([sphere_kernel])

    assert meshes == ["mesh"]
    assert len(tags) == 1
    assert backend.extract_tch_sphere_calls == 1
    assert backend.extract_tch_scene_calls == 0
    assert backend.extract_native_scene_calls == 0
    assert backend.last_calls[-1][0] == "tch_sphere"
    assert backend.last_calls[-1][3] == "mps"


def test_rust_ocmesher_prefers_tch_scene_on_cuda_indexed_device(sample_cameras, sample_bounds, sphere_kernel):
    backend = DummyRustBackendWithScenes()
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=backend, device="cuda:0")

    meshes, tags = mesher([sphere_kernel])

    assert meshes == ["mesh"]
    assert len(tags) == 1
    assert backend.extract_tch_sphere_calls == 1
    assert backend.extract_native_scene_calls == 0
    assert backend.last_calls[-1][0] == "tch_sphere"
    assert backend.last_calls[-1][3] == "cuda:0"


def test_rust_ocmesher_normalizes_cuda_device_input(sample_cameras, sample_bounds, sphere_kernel):
    backend = DummyRustBackendWithScenes()
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=backend, device=" CUDA:00 ")

    meshes, tags = mesher([sphere_kernel])

    assert meshes == ["mesh"]
    assert len(tags) == 1
    assert mesher.device == "cuda:0"
    assert backend.last_calls[-1][3] == "cuda:0"


def test_rust_ocmesher_prefers_native_scene_on_cpu(sample_cameras, sample_bounds, sphere_kernel):
    backend = DummyRustBackendWithScenes()
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=backend, device="cpu")

    meshes, tags = mesher([sphere_kernel])

    assert meshes == ["mesh"]
    assert len(tags) == 1
    assert backend.extract_native_sphere_calls == 1
    assert backend.extract_native_scene_calls == 0
    assert backend.extract_tch_scene_calls == 0
    assert backend.last_calls[-1][0] == "native_sphere"


def test_rust_ocmesher_can_force_tch_scene_on_cpu(sample_cameras, sample_bounds, sphere_kernel):
    backend = DummyRustBackendWithScenes()
    mesher = RustOcMesher(
        sample_cameras,
        sample_bounds,
        backend=backend,
        device="cpu",
        kernel_runtime="tch",
    )

    meshes, tags = mesher([sphere_kernel])

    assert meshes == ["mesh"]
    assert len(tags) == 1
    assert backend.extract_tch_sphere_calls == 1
    assert backend.extract_native_sphere_calls == 0
    assert backend.extract_native_scene_calls == 0
    assert backend.last_calls[-1][0] == "tch_sphere"
    assert backend.last_calls[-1][3] == "cpu"


def test_rust_ocmesher_rejects_invalid_kernel_runtime(sample_cameras, sample_bounds):
    with pytest.raises(ValueError, match="kernel_runtime must be one of"):
        RustOcMesher(
            sample_cameras,
            sample_bounds,
            backend=DummyRustBackend(),
            kernel_runtime="bad",
        )


def test_rust_ocmesher_rejects_invalid_device_string(sample_cameras, sample_bounds):
    with pytest.raises(ValueError, match="device must be one of"):
        RustOcMesher(
            sample_cameras,
            sample_bounds,
            backend=DummyRustBackend(),
            device="gpu",
        )


def test_rust_ocmesher_caches_inferred_specs(sample_cameras, sample_bounds):
    backend = DummyRustBackendWithScenes()
    mesher = RustOcMesher(sample_cameras, sample_bounds, backend=backend, device="cpu")

    class SphereKernelWithBatch:
        def __init__(self):
            self.calls = 0

        def evaluate_batch(self, xyz):
            self.calls += 1
            return np.linalg.norm(xyz, axis=1) - 1.0

        def __call__(self, _xyz):
            msg = "inference should prefer evaluate_batch"
            raise AssertionError(msg)

    kernel = SphereKernelWithBatch()
    meshes1, tags1 = mesher([kernel])
    meshes2, tags2 = mesher([kernel])

    assert meshes1 == ["mesh"]
    assert meshes2 == ["mesh"]
    assert len(tags1) == 1
    assert len(tags2) == 1
    assert backend.extract_native_sphere_calls == 2
    assert kernel.calls == 1


def test_rust_ocmesher_can_disable_primitive_inference(sample_cameras, sample_bounds, sphere_kernel):
    backend = DummyRustBackendWithScenes()
    mesher = RustOcMesher(
        sample_cameras,
        sample_bounds,
        backend=backend,
        device="mps",
        use_primitive_inference=False,
    )

    meshes, tags = mesher([sphere_kernel])

    assert meshes == ["mesh"]
    assert len(tags) == 1
    assert backend.extract_tch_sphere_calls == 0
    assert backend.extract_native_sphere_calls == 0
    assert backend.extract_tch_scene_calls == 0
    assert backend.extract_native_scene_calls == 0
    assert len(backend.last_calls) == 1
    assert isinstance(backend.last_calls[0], list)


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


def test_stream_policy_auto_preserved_on_cuda_indexed_device(sample_cameras, sample_bounds):
    mesher = RustOcMesher(
        sample_cameras,
        sample_bounds,
        device="cuda:0",
        stream_policy="auto",
        backend=DummyRustBackend(),
    )
    assert mesher.stream_policy == "auto"


def test_rust_ocmesher_rejects_invalid_stream_policy(sample_cameras, sample_bounds):
    with pytest.raises(ValueError, match="stream_policy must be one of"):
        RustOcMesher(
            sample_cameras,
            sample_bounds,
            device="cpu",
            stream_policy="async",
            backend=DummyRustBackend(),
        )


def test_rust_ocmesher_default_device_prefers_cuda(sample_cameras, sample_bounds):
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Mps:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Backends:
        mps = _Mps()

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

    with patch.dict(sys.modules, {"torch": _Torch()}):
        mesher = RustOcMesher(sample_cameras, sample_bounds, backend=DummyRustBackend())

    assert mesher.device == "cuda"


def test_rust_ocmesher_default_device_prefers_mps_when_cuda_unavailable(sample_cameras, sample_bounds):
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Mps:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Backends:
        mps = _Mps()

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

    with patch.dict(sys.modules, {"torch": _Torch()}):
        mesher = RustOcMesher(sample_cameras, sample_bounds, backend=DummyRustBackend())

    assert mesher.device == "mps"


def test_rust_ocmesher_default_device_falls_back_to_cpu(sample_cameras, sample_bounds):
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Mps:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Backends:
        mps = _Mps()

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

    with patch.dict(sys.modules, {"torch": _Torch()}):
        mesher = RustOcMesher(sample_cameras, sample_bounds, backend=DummyRustBackend())

    assert mesher.device == "cpu"


def test_make_rust_ocmesher_raises_import_error_without_extension(sample_cameras, sample_bounds):
    """Without ocmesher_rust installed, make_rust_ocmesher raises ImportError."""
    with (
        patch.dict(sys.modules, {"ocmesher_rust": None}),
        pytest.raises(
            ImportError,
            match="ocmesher_rust is not installed",
        ),
    ):
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


def test_make_rust_ocmesher_forwards_default_backend_meshing_params(sample_cameras, sample_bounds):
    mock_backend = DummyRustBackend()
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"
    mock_ext.Backend.return_value = mock_backend

    with patch.dict(sys.modules, {"ocmesher_rust": mock_ext}):
        make_rust_ocmesher(sample_cameras, sample_bounds)

    call_kwargs = mock_ext.Backend.call_args.kwargs
    assert call_kwargs["pixels_per_cube"] == 8
    assert call_kwargs["inv_scale"] == 10
    assert call_kwargs["min_dist"] == 1
    assert call_kwargs["memory_limit_mb"] == 1000
    assert call_kwargs["bisection_iters"] == 15
    assert call_kwargs["bisection_tol"] == 0.0
    assert call_kwargs["enclosed"] is True
    assert call_kwargs["simplify_occluded"] is True
    assert call_kwargs["visible_relax_iter"] == 2
    assert call_kwargs["coarse_count"] == 500000


def test_make_rust_ocmesher_forwards_kernel_runtime_to_wrapper(sample_cameras, sample_bounds):
    mock_backend = DummyRustBackend()
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"
    mock_ext.Backend.return_value = mock_backend

    with patch.dict(sys.modules, {"ocmesher_rust": mock_ext}):
        mesher = make_rust_ocmesher(sample_cameras, sample_bounds, kernel_runtime="tch")

    assert isinstance(mesher, RustOcMesher)
    assert mesher.kernel_runtime == "tch"


def test_make_rust_ocmesher_rejects_invalid_kernel_runtime_before_backend_init(
    sample_cameras,
    sample_bounds,
):
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"

    with (
        patch.dict(sys.modules, {"ocmesher_rust": mock_ext}),
        pytest.raises(
            ValueError,
            match="kernel_runtime must be one of",
        ),
    ):
        make_rust_ocmesher(sample_cameras, sample_bounds, kernel_runtime="invalid")

    mock_ext.Backend.assert_not_called()


def test_make_rust_ocmesher_rejects_non_string_kernel_runtime(sample_cameras, sample_bounds):
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"

    with (
        patch.dict(sys.modules, {"ocmesher_rust": mock_ext}),
        pytest.raises(
            TypeError,
            match="kernel_runtime must be a string",
        ),
    ):
        make_rust_ocmesher(sample_cameras, sample_bounds, kernel_runtime=1)

    mock_ext.Backend.assert_not_called()


def test_make_rust_ocmesher_rejects_invalid_device_before_backend_init(sample_cameras, sample_bounds):
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"

    with (
        patch.dict(sys.modules, {"ocmesher_rust": mock_ext}),
        pytest.raises(
            ValueError,
            match="device must be one of",
        ),
    ):
        make_rust_ocmesher(sample_cameras, sample_bounds, device="gpu")

    mock_ext.Backend.assert_not_called()


def test_make_rust_ocmesher_normalizes_device_before_wrapper(sample_cameras, sample_bounds):
    mock_backend = DummyRustBackend()
    mock_ext = MagicMock()
    mock_ext.find_core_so.return_value = "/fake/core.so"
    mock_ext.Backend.return_value = mock_backend

    with patch.dict(sys.modules, {"ocmesher_rust": mock_ext}):
        mesher = make_rust_ocmesher(sample_cameras, sample_bounds, device=" CUDA:00 ")

    assert isinstance(mesher, RustOcMesher)
    assert mesher.device == "cuda:0"


def test_make_rust_ocmesher_exported_from_package():
    """make_rust_ocmesher is accessible from the top-level ocmesher package."""
    import ocmesher

    fn = ocmesher.make_rust_ocmesher
    assert callable(fn)


@pytest.mark.integration
def test_make_rust_ocmesher_runs_with_compiled_extension(sample_cameras, sample_bounds, sphere_kernel):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    mesher = make_rust_ocmesher(
        sample_cameras,
        sample_bounds,
        lib_path=str(core_so),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    meshes, tags = mesher([sphere_kernel])

    assert len(meshes) == 1
    assert len(tags) == 1
    assert hasattr(meshes[0], "vertices")
    assert hasattr(meshes[0], "faces")
    assert meshes[0].vertices.ndim == 2
    assert meshes[0].vertices.shape[1] == 3
    assert meshes[0].faces.ndim == 2
    assert meshes[0].faces.shape[1] == 3
    assert meshes[0].vertices.shape[0] > 0
    assert meshes[0].faces.shape[0] > 0
    assert tags[0].dtype == np.bool_
    assert tags[0].shape == (meshes[0].vertices.shape[0],)


@pytest.mark.integration
def test_compiled_extension_extract_native_sphere(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    meshes, tags = backend.extract_native_sphere(radius=1.0)

    assert len(meshes) == 1
    assert len(tags) == 1
    assert meshes[0].vertices.shape[1] == 3
    assert meshes[0].faces.shape[1] == 3
    assert meshes[0].vertices.shape[0] > 0
    assert meshes[0].faces.shape[0] > 0
    assert tags[0].shape == (meshes[0].vertices.shape[0],)
    assert tags[0].dtype == np.bool_


@pytest.mark.integration
def test_compiled_extension_extract_tch_sphere(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    if not hasattr(ocmesher_rust.Backend, "extract_tch_sphere"):
        pytest.skip("compiled Rust extension was not built with tch-kernels")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    meshes, tags = backend.extract_tch_sphere(radius=1.0)

    assert len(meshes) == 1
    assert len(tags) == 1
    assert meshes[0].vertices.shape[1] == 3
    assert meshes[0].faces.shape[1] == 3
    assert meshes[0].vertices.shape[0] > 0
    assert meshes[0].faces.shape[0] > 0
    assert tags[0].shape == (meshes[0].vertices.shape[0],)
    assert tags[0].dtype == np.bool_


@pytest.mark.integration
def test_compiled_extension_extract_native_plane(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    if not hasattr(ocmesher_rust.Backend, "extract_native_plane"):
        pytest.skip("compiled Rust extension was not built with plane support")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    meshes, tags = backend.extract_native_plane(offset=0.0)

    assert len(meshes) == 1
    assert len(tags) == 1
    assert meshes[0].vertices.shape[1] == 3
    assert meshes[0].faces.shape[1] == 3
    assert meshes[0].vertices.shape[0] > 0
    assert meshes[0].faces.shape[0] > 0
    assert tags[0].shape == (meshes[0].vertices.shape[0],)
    assert tags[0].dtype == np.bool_


@pytest.mark.integration
def test_compiled_extension_extract_tch_plane(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    if not hasattr(ocmesher_rust.Backend, "extract_tch_plane"):
        pytest.skip("compiled Rust extension was not built with tch-kernels")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    meshes, tags = backend.extract_tch_plane(offset=0.0)

    assert len(meshes) == 1
    assert len(tags) == 1
    assert meshes[0].vertices.shape[1] == 3
    assert meshes[0].faces.shape[1] == 3
    assert meshes[0].vertices.shape[0] > 0
    assert meshes[0].faces.shape[0] > 0
    assert tags[0].shape == (meshes[0].vertices.shape[0],)
    assert tags[0].dtype == np.bool_


@pytest.mark.integration
def test_compiled_extension_extract_native_sphere_plane(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    if not hasattr(ocmesher_rust.Backend, "extract_native_sphere_plane"):
        pytest.skip("compiled Rust extension was not built with composed scene support")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    meshes, tags = backend.extract_native_sphere_plane(sphere_radius=1.0, plane_offset=0.0)

    assert len(meshes) == 2
    assert len(tags) == 2
    assert all(mesh.vertices.shape[1] == 3 for mesh in meshes)
    assert all(mesh.faces.shape[1] == 3 for mesh in meshes)
    assert all(mesh.vertices.shape[0] > 0 for mesh in meshes)
    assert all(mesh.faces.shape[0] > 0 for mesh in meshes)
    assert all(tag.dtype == np.bool_ for tag in tags)
    assert all(tag.shape == (mesh.vertices.shape[0],) for mesh, tag in zip(meshes, tags, strict=True))


@pytest.mark.integration
def test_compiled_extension_extract_tch_sphere_plane(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    if not hasattr(ocmesher_rust.Backend, "extract_tch_sphere_plane"):
        pytest.skip("compiled Rust extension was not built with tch-kernels")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    meshes, tags = backend.extract_tch_sphere_plane(sphere_radius=1.0, plane_offset=0.0)

    assert len(meshes) == 2
    assert len(tags) == 2
    assert all(mesh.vertices.shape[1] == 3 for mesh in meshes)
    assert all(mesh.faces.shape[1] == 3 for mesh in meshes)
    assert all(mesh.vertices.shape[0] > 0 for mesh in meshes)
    assert all(mesh.faces.shape[0] > 0 for mesh in meshes)
    assert all(tag.dtype == np.bool_ for tag in tags)
    assert all(tag.shape == (mesh.vertices.shape[0],) for mesh, tag in zip(meshes, tags, strict=True))


@pytest.mark.integration
def test_compiled_extension_extract_native_scene(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    if not hasattr(ocmesher_rust.Backend, "extract_native_scene"):
        pytest.skip("compiled Rust extension was not built with scene-spec support")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    meshes, tags = backend.extract_native_scene(
        [
            {"type": "sphere", "radius": 1.0, "center": [0.0, 0.0, 0.0]},
            {"type": "plane", "offset": 0.0, "normal": [0.0, 0.0, 1.0]},
        ]
    )

    assert len(meshes) == 2
    assert len(tags) == 2
    assert all(mesh.vertices.shape[1] == 3 for mesh in meshes)
    assert all(mesh.faces.shape[1] == 3 for mesh in meshes)
    assert all(mesh.vertices.shape[0] > 0 for mesh in meshes)
    assert all(mesh.faces.shape[0] > 0 for mesh in meshes)
    assert all(tag.dtype == np.bool_ for tag in tags)
    assert all(tag.shape == (mesh.vertices.shape[0],) for mesh, tag in zip(meshes, tags, strict=True))


@pytest.mark.integration
def test_compiled_extension_extract_tch_scene(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    if not hasattr(ocmesher_rust.Backend, "extract_tch_scene"):
        pytest.skip("compiled Rust extension was not built with tch-kernels")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    meshes, tags = backend.extract_tch_scene(
        [
            {"type": "sphere", "radius": 1.0, "center": [0.0, 0.0, 0.0]},
            {"type": "plane", "offset": 0.0, "normal": [0.0, 0.0, 1.0]},
        ]
    )

    assert len(meshes) == 2
    assert len(tags) == 2
    assert all(mesh.vertices.shape[1] == 3 for mesh in meshes)
    assert all(mesh.faces.shape[1] == 3 for mesh in meshes)
    assert all(mesh.vertices.shape[0] > 0 for mesh in meshes)
    assert all(mesh.faces.shape[0] > 0 for mesh in meshes)
    assert all(tag.dtype == np.bool_ for tag in tags)
    assert all(tag.shape == (mesh.vertices.shape[0],) for mesh, tag in zip(meshes, tags, strict=True))


@pytest.mark.integration
def test_compiled_extension_native_vs_tch_mps_signature_regression(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")
    torch = pytest.importorskip("torch", reason="torch not installed")

    if not hasattr(ocmesher_rust.Backend, "extract_native_scene"):
        pytest.skip("compiled Rust extension was not built with scene-spec support")
    if not hasattr(ocmesher_rust.Backend, "extract_tch_scene"):
        pytest.skip("compiled Rust extension was not built with tch-kernels")
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS backend not available")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=100_000,
    )

    primitives = [{"type": "sphere", "radius": 5.0, "center": [0.0, 0.0, 0.0]}]
    native_meshes, native_tags = backend.extract_native_scene(primitives)
    tch_meshes, tch_tags = backend.extract_tch_scene(primitives, device="mps")

    native_sig = _mesh_signature(native_meshes[0], native_tags[0])
    tch_sig = _mesh_signature(tch_meshes[0], tch_tags[0])

    expected_native = {
        "verts": 9005,
        "faces": 17990,
        "verts_sum": 49640.90751688918,
        "faces_sum": 234929499,
        "tag_true": 6804,
    }
    expected_tch_mps = {
        "verts": 8833,
        "faces": 17639,
        "verts_sum": 48516.18087918505,
        "faces_sum": 228110285,
        "tag_true": 6608,
    }

    assert native_sig["verts"] == expected_native["verts"]
    assert native_sig["faces"] == expected_native["faces"]
    assert native_sig["faces_sum"] == expected_native["faces_sum"]
    assert native_sig["tag_true"] == expected_native["tag_true"]
    assert np.isclose(native_sig["verts_sum"], expected_native["verts_sum"], atol=1e-6)

    assert tch_sig["verts"] == expected_tch_mps["verts"]
    assert tch_sig["faces"] == expected_tch_mps["faces"]
    assert tch_sig["faces_sum"] == expected_tch_mps["faces_sum"]
    assert tch_sig["tag_true"] == expected_tch_mps["tag_true"]
    assert np.isclose(tch_sig["verts_sum"], expected_tch_mps["verts_sum"], atol=1e-6)


@pytest.mark.integration
def test_compiled_extension_extract_native_scene_rejects_empty(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    with pytest.raises(ValueError, match="primitives must be a non-empty list"):
        backend.extract_native_scene([])


@pytest.mark.integration
def test_compiled_extension_extract_native_scene_rejects_unknown_type(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    with pytest.raises(ValueError, match="unsupported primitive type: capsule"):
        backend.extract_native_scene([{"type": "capsule"}])


@pytest.mark.integration
def test_compiled_extension_extract_tch_scene_rejects_non_dict(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    if not hasattr(ocmesher_rust.Backend, "extract_tch_scene"):
        pytest.skip("compiled Rust extension was not built with tch-kernels")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    with pytest.raises(ValueError, match=r"primitives\[0\] must be a dict"):
        backend.extract_tch_scene(["bad"])


@pytest.mark.integration
def test_compiled_extension_extract_tch_scene_rejects_negative_radius(sample_cameras, sample_bounds):
    ocmesher_rust = pytest.importorskip("ocmesher_rust", reason="compiled Rust extension not installed")
    if not hasattr(ocmesher_rust, "Backend"):
        pytest.skip("compiled Rust extension not built")

    if not hasattr(ocmesher_rust.Backend, "extract_tch_scene"):
        pytest.skip("compiled Rust extension was not built with tch-kernels")

    core_so = Path(__file__).resolve().parents[1] / "ocmesher" / "lib" / "core.so"
    if not core_so.exists():
        pytest.skip("compiled core.so not available")

    cam_poses, ks, hs, ws = sample_cameras
    backend = ocmesher_rust.Backend(
        lib_path=str(core_so),
        cameras=(
            [np.asarray(p, dtype=np.float64).ravel().tolist() for p in cam_poses],
            [np.asarray(k, dtype=np.float64).ravel().tolist() for k in ks],
            [float(h) for h in hs],
            [float(w) for w in ws],
        ),
        bounds=np.asarray(sample_bounds, dtype=np.float64).tolist(),
        pixels_per_cube=32,
        coarse_count=20_000,
    )

    with pytest.raises(ValueError, match="radius must be positive"):
        backend.extract_tch_scene([{"type": "sphere", "radius": -1.0}])
