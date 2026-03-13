"""Shared fixtures for OcMesher tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def identity_cameras():
    """Minimal valid camera tuple: 2 cameras at identity pose.

    Returns (cam_poses, Ks, Hs, Ws) suitable for OcMesher.__init__.
    """
    n = 2
    cam_poses = [np.eye(4, dtype=np.float64) for _ in range(n)]
    # Simple intrinsics: focal length 500, principal point (320, 240)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    Ks = [K.copy() for _ in range(n)]
    Hs = [480] * n
    Ws = [640] * n
    return (cam_poses, Ks, Hs, Ws)


@pytest.fixture
def unit_bounds():
    """Unit cube bounds: [-1, 1, -1, 1, -1, 1]."""
    return [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]


@pytest.fixture
def translated_cameras():
    """Two cameras: one at origin, one translated by (1,2,3)."""
    pose0 = np.eye(4, dtype=np.float64)
    pose1 = np.eye(4, dtype=np.float64)
    pose1[:3, 3] = [1.0, 2.0, 3.0]
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    return ([pose0, pose1], [K.copy(), K.copy()], [480, 480], [640, 640])


@pytest.fixture
def large_bounds():
    """Large bounds for stress testing: [-1000, 1000] on each axis."""
    return [-1000.0, 1000.0, -1000.0, 1000.0, -1000.0, 1000.0]


@pytest.fixture
def tiny_bounds():
    """Very small bounds for precision testing."""
    return [-1e-6, 1e-6, -1e-6, 1e-6, -1e-6, 1e-6]
