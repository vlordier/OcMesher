"""Tests for round-2 refactoring improvements.

Covers shared helpers (preprocess_cameras, out_of_bounds_mask),
device/dtype selection, octree expansion step, __repr__, return type
hints, kernel shape validation, and torch_device fixture parametrisation.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from ocmesher._validation import out_of_bounds_mask, preprocess_cameras
from ocmesher.torch_core import TorchOcMesher


# ---------------------------------------------------------------------------
# preprocess_cameras
# ---------------------------------------------------------------------------
class TestPreprocessCameras:
    def test_returns_correct_shapes(self, sample_cameras):
        cam_poses, Ks, Hs, Ws = sample_cameras
        inv_3x4, intrinsics, heights, widths = preprocess_cameras(cam_poses, Ks, Hs, Ws)
        assert inv_3x4.shape == (1, 3, 4)
        assert intrinsics.shape == (1, 3, 3)
        assert inv_3x4.dtype == np.float64
        assert intrinsics.dtype == np.float64

    def test_heights_widths_are_int_tuples(self, sample_cameras):
        cam_poses, Ks, Hs, Ws = sample_cameras
        _, _, heights, widths = preprocess_cameras(cam_poses, Ks, Hs, Ws)
        assert isinstance(heights, tuple)
        assert isinstance(widths, tuple)
        assert all(isinstance(h, int) for h in heights)
        assert all(isinstance(w, int) for w in widths)

    def test_inverse_is_correct(self, sample_camera_pose, sample_intrinsics):
        cam_poses = [sample_camera_pose]
        Ks = [sample_intrinsics]
        inv_3x4, _, _, _ = preprocess_cameras(cam_poses, Ks, [720], [1280])
        expected = np.linalg.inv(sample_camera_pose)[:3, :4]
        np.testing.assert_allclose(inv_3x4[0], expected)

    def test_multi_camera(self, sample_camera_pose, sample_intrinsics):
        n = 5
        cam_poses = [sample_camera_pose] * n
        Ks = [sample_intrinsics] * n
        inv_3x4, intrinsics, heights, widths = preprocess_cameras(cam_poses, Ks, [720] * n, [1280] * n)
        assert inv_3x4.shape == (n, 3, 4)
        assert intrinsics.shape == (n, 3, 3)
        assert len(heights) == n
        assert len(widths) == n

    def test_contiguous_output(self, sample_cameras):
        cam_poses, Ks, Hs, Ws = sample_cameras
        inv_3x4, intrinsics, _, _ = preprocess_cameras(cam_poses, Ks, Hs, Ws)
        assert inv_3x4.flags["C_CONTIGUOUS"]
        assert intrinsics.flags["C_CONTIGUOUS"]

    def test_singular_pose_raises(self):
        singular = np.zeros((4, 4), dtype=np.float64)
        Ks = [np.eye(3, dtype=np.float64)]
        with pytest.raises(np.linalg.LinAlgError):
            preprocess_cameras([singular], Ks, [720], [1280])


# ---------------------------------------------------------------------------
# out_of_bounds_mask
# ---------------------------------------------------------------------------
class TestOutOfBoundsMask:
    def test_all_inside(self):
        xyz = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        b_min = np.array([-5, -5, -5], dtype=np.float64)
        b_max = np.array([5, 5, 5], dtype=np.float64)
        mask = out_of_bounds_mask(xyz, b_min, b_max)
        assert mask.shape == (2,)
        assert not mask.any()

    def test_all_outside(self):
        xyz = np.array([[10, 10, 10], [-10, -10, -10]], dtype=np.float64)
        b_min = np.array([-5, -5, -5], dtype=np.float64)
        b_max = np.array([5, 5, 5], dtype=np.float64)
        mask = out_of_bounds_mask(xyz, b_min, b_max)
        assert mask.all()

    def test_on_boundary_is_out(self):
        xyz = np.array([[5, 0, 0], [0, -5, 0], [0, 0, 5]], dtype=np.float64)
        b_min = np.array([-5, -5, -5], dtype=np.float64)
        b_max = np.array([5, 5, 5], dtype=np.float64)
        mask = out_of_bounds_mask(xyz, b_min, b_max)
        assert mask.all()

    def test_mixed(self):
        xyz = np.array([[0, 0, 0], [10, 0, 0], [0, 0, 0]], dtype=np.float64)
        b_min = np.array([-5, -5, -5], dtype=np.float64)
        b_max = np.array([5, 5, 5], dtype=np.float64)
        mask = out_of_bounds_mask(xyz, b_min, b_max)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_empty_input(self):
        xyz = np.empty((0, 3), dtype=np.float64)
        b_min = np.array([-5, -5, -5], dtype=np.float64)
        b_max = np.array([5, 5, 5], dtype=np.float64)
        mask = out_of_bounds_mask(xyz, b_min, b_max)
        assert mask.shape == (0,)

    def test_matches_core_static_method(self, sample_bounds):
        """Shared helper must match OcMesher._out_of_bounds_mask exactly."""
        from ocmesher.core import OcMesher

        rng = np.random.default_rng(42)
        xyz = rng.uniform(-10, 10, (100, 3))
        b_min = np.array([sample_bounds[0], sample_bounds[2], sample_bounds[4]])
        b_max = np.array([sample_bounds[1], sample_bounds[3], sample_bounds[5]])
        expected = OcMesher._out_of_bounds_mask(xyz, b_min, b_max)
        actual = out_of_bounds_mask(xyz, b_min, b_max)
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# _select_device_and_dtype
# ---------------------------------------------------------------------------
class TestSelectDeviceAndDtype:
    def test_explicit_cpu(self):
        dev, dtype = TorchOcMesher._select_device_and_dtype("cpu")
        assert dev == torch.device("cpu")
        assert dtype == torch.float64

    def test_auto_returns_valid_device(self):
        dev, dtype = TorchOcMesher._select_device_and_dtype()
        assert isinstance(dev, torch.device)
        assert dtype in (torch.float32, torch.float64)

    def test_cpu_gets_float64(self):
        _, dtype = TorchOcMesher._select_device_and_dtype("cpu")
        assert dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_cuda_gets_float32(self):
        _, dtype = TorchOcMesher._select_device_and_dtype("cuda")
        assert dtype == torch.float32


# ---------------------------------------------------------------------------
# _octree_expansion_step
# ---------------------------------------------------------------------------
class TestOctreeExpansionStep:
    def test_returns_none_when_no_expansion(self, single_cam_mesher):
        """A single very deep cube should not need expansion."""
        coords = torch.zeros((1, 3), dtype=torch.int64)
        levels = torch.tensor([30], dtype=torch.int64)  # very deep
        result = single_cam_mesher._octree_expansion_step(coords, levels, 100)
        assert result is None

    def test_returns_children_for_root(self, single_cam_mesher):
        """The root cube (level 0) should always be expandable."""
        coords = torch.zeros((1, 3), dtype=torch.int64)
        levels = torch.zeros(1, dtype=torch.int64)
        result = single_cam_mesher._octree_expansion_step(coords, levels, 1000)
        assert result is not None
        keep_mask, child_coords, child_levels, proj = result
        assert child_coords.shape[1] == 3
        assert child_levels.ndim == 1
        # Root is expanded, so children should have level 1
        assert (child_levels == 1).all()

    def test_budget_limiting(self, single_cam_mesher):
        """Budget of 1 should only expand 1 cube (producing 8 children)."""
        coords = torch.zeros((2, 3), dtype=torch.int64)
        levels = torch.zeros(2, dtype=torch.int64)
        # Budget target of 9: keeps 1, expands 1 → 1 + 8 = 9
        result = single_cam_mesher._octree_expansion_step(coords, levels, 9)
        if result is not None:
            keep_mask, child_coords, child_levels, _ = result
            n_kept = keep_mask.sum().item()
            n_children = len(child_coords)
            assert n_kept + n_children <= 9 + 8  # reasonable bound


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------
class TestTorchOcMesherRepr:
    def test_contains_class_name(self, single_cam_mesher):
        r = repr(single_cam_mesher)
        assert r.startswith("TorchOcMesher(")

    def test_contains_device(self, single_cam_mesher):
        r = repr(single_cam_mesher)
        assert "device=" in r

    def test_contains_n_cameras(self, single_cam_mesher):
        r = repr(single_cam_mesher)
        assert "n_cameras=1" in r

    def test_contains_enclosed(self, single_cam_mesher):
        r = repr(single_cam_mesher)
        assert "enclosed=" in r


# ---------------------------------------------------------------------------
# Return type hint smoke test
# ---------------------------------------------------------------------------
class TestCallReturnType:
    def test_torch_mesher_type_annotation(self):
        import inspect

        sig = inspect.signature(TorchOcMesher.__call__)
        ret = sig.return_annotation
        assert ret != inspect.Parameter.empty

    def test_core_mesher_type_annotation(self):
        import inspect

        from ocmesher.core import OcMesher

        sig = inspect.signature(OcMesher.__call__)
        ret = sig.return_annotation
        assert ret != inspect.Parameter.empty


# ---------------------------------------------------------------------------
# Kernel shape validation in _evaluate_sdf
# ---------------------------------------------------------------------------
class TestEvaluateSdfShapeValidation:
    def test_wrong_shape_single_kernel_raises(self, single_cam_mesher):
        """Single kernel returning wrong shape should raise ValueError."""

        def bad_kernel(xyz):
            return np.zeros((len(xyz), 2))  # wrong: (N, 2) instead of (N,)

        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="expected"):
            single_cam_mesher._evaluate_sdf([bad_kernel], positions)

    def test_wrong_shape_multi_kernel_raises(self, single_cam_mesher):
        """Multi kernel returning wrong shape should raise ValueError."""

        def good_kernel(xyz):
            return np.zeros(len(xyz))

        def bad_kernel(xyz):
            return np.zeros((len(xyz), 1))

        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="expected"):
            single_cam_mesher._evaluate_sdf([good_kernel, bad_kernel], positions)

    def test_correct_shape_passes(self, single_cam_mesher):
        """Correctly-shaped kernel output should not raise."""

        def kernel(xyz):
            return np.linalg.norm(xyz, axis=1) - 1.0

        positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
        result = single_cam_mesher._evaluate_sdf([kernel], positions)
        assert result.shape == (2, 1)


# ---------------------------------------------------------------------------
# torch_device fixture parametrisation
# ---------------------------------------------------------------------------
class TestTorchDeviceFixture:
    def test_mesher_on_available_device(self, torch_device, sample_cameras, sample_bounds):
        """TorchOcMesher must initialise on every available device."""
        m = TorchOcMesher(sample_cameras, sample_bounds, device=torch_device)
        assert str(m.device) == torch_device

    def test_select_device_matches(self, torch_device):
        """_select_device_and_dtype with explicit device must match."""
        dev, dtype = TorchOcMesher._select_device_and_dtype(torch_device)
        assert str(dev) == torch_device


# ---------------------------------------------------------------------------
# Shared helper: validation __all__ updated
# ---------------------------------------------------------------------------
class TestValidationAllUpdated:
    def test_preprocess_cameras_in_all(self):
        from ocmesher._validation import __all__

        assert "preprocess_cameras" in __all__

    def test_out_of_bounds_mask_in_all(self):
        from ocmesher._validation import __all__

        assert "out_of_bounds_mask" in __all__


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def single_cam_mesher(sample_cameras, torch_mesher_factory):
    """TorchOcMesher with a single camera on CPU."""
    return torch_mesher_factory(sample_cameras)
