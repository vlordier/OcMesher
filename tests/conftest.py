"""Shared fixtures for OcMesher tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def _available_torch_devices() -> list[str]:
    """Return torch device strings available on this platform."""
    try:
        import torch
    except ImportError:
        return []
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(params=_available_torch_devices(), scope="session")
def torch_device(request):
    """Parametrised fixture yielding each available torch device string."""
    return request.param


@pytest.fixture
def torch_mesher_factory(sample_bounds):
    """Factory for creating TorchOcMesher instances in tests."""

    from ocmesher.torch_core import TorchOcMesher

    def factory(cameras, *, bounds=None, device="cpu", **kwargs):
        return TorchOcMesher(cameras, sample_bounds if bounds is None else bounds, device=device, **kwargs)

    return factory


@pytest.fixture
def sample_camera_pose():
    """A single 4x4 identity-like camera pose matrix."""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 3],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sample_intrinsics():
    """A single 3x3 camera intrinsics matrix."""
    return np.array(
        [
            [2000, 0, 640],
            [0, 2000, 360],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sample_cameras(sample_camera_pose, sample_intrinsics):
    """Camera tuple ``(poses, Ks, Hs, Ws)`` with one camera."""
    return ([sample_camera_pose], [sample_intrinsics], [720], [1280])


@pytest.fixture
def sample_cameras_as_lists():
    """Camera tuple with plain Python lists instead of numpy arrays."""
    pose = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 3], [0, 0, 0, 1]]
    k = [[2000, 0, 640], [0, 2000, 360], [0, 0, 1]]
    return ([pose], [k], [720], [1280])


@pytest.fixture
def sample_bounds():
    """Bounds array: ``[x_min, x_max, y_min, y_max, z_min, z_max]``."""
    return np.array([-5.0, 5.0, -5.0, 5.0, -5.0, 5.0])


@pytest.fixture
def sphere_kernel():
    """Simple sphere SDF kernel centred at origin with radius 1."""

    def kernel(XYZ):
        return np.linalg.norm(XYZ, axis=1) - 1.0

    return kernel


@pytest.fixture
def plane_kernel():
    """Simple plane SDF kernel at z=0."""

    def kernel(XYZ):
        return XYZ[:, 2].copy()

    return kernel


@pytest.fixture
def mock_dll():
    """A mock CDLL object with all expected C functions."""
    dll = MagicMock()
    func_names = [
        "run_coarse",
        "fine_group",
        "fine_iteration",
        "fine_iteration_output",
        "vis_filter",
        "final_iteration",
        "final_iteration_occluded",
        "final_iteration2",
        "final_iteration3",
        "final_iteration3_occluded",
        "final_remaining",
        "get_verts_center",
        "update_verts",
        "get_lr_verts",
        "finalize_verts",
        "construct_faces",
        "get_extra_verts_center",
        "update_extra_verts",
        "get_lr_extra_verts",
        "finalize_extra_verts",
        "get_faces",
        "get_in_view_tag",
    ]
    for name in func_names:
        setattr(dll, name, MagicMock())
    return dll


@pytest.fixture
def sample_numpy_int_array():
    """Sample numpy int32 array."""
    return np.array([1, 2, 3, 4], dtype=np.int32)


@pytest.fixture
def sample_numpy_float_array():
    """Sample numpy float64 array."""
    return np.array([1.0, 2.0, 3.0], dtype=np.float64)


@pytest.fixture
def sample_numpy_float32_array():
    """Sample numpy float32 array."""
    return np.array([1.0, 2.0, 3.0], dtype=np.float32)


@pytest.fixture
def sample_numpy_bool_array():
    """Sample numpy bool array."""
    return np.array([True, False, True], dtype=bool)
