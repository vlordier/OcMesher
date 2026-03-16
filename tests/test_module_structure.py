"""Tests for new module structure: _mc_tables, _validation, _constants, _types."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# _mc_tables: table integrity
# ---------------------------------------------------------------------------


class TestMcTablesIntegrity:
    """Verify that extracted marching-cubes tables have the expected shape."""

    def test_edge_table_length(self):
        from ocmesher._mc_tables import EDGE_TABLE

        assert len(EDGE_TABLE) == 256

    def test_tri_table_length(self):
        from ocmesher._mc_tables import TRI_TABLE

        assert len(TRI_TABLE) == 256

    def test_edge_vertices_length(self):
        from ocmesher._mc_tables import EDGE_VERTICES

        assert len(EDGE_VERTICES) == 12

    def test_corner_offsets_length(self):
        from ocmesher._mc_tables import CORNER_OFFSETS

        assert len(CORNER_OFFSETS) == 8

    def test_edge_table_values_are_int(self):
        from ocmesher._mc_tables import EDGE_TABLE

        assert all(isinstance(v, int) for v in EDGE_TABLE)

    def test_edge_table_known_values(self):
        from ocmesher._mc_tables import EDGE_TABLE

        assert EDGE_TABLE[0] == 0x000
        assert EDGE_TABLE[1] == 0x109
        assert EDGE_TABLE[255] == 0x000

    def test_tri_table_entries_are_lists(self):
        from ocmesher._mc_tables import TRI_TABLE

        for entry in TRI_TABLE:
            assert isinstance(entry, list)

    def test_edge_vertices_tuples_of_two(self):
        from ocmesher._mc_tables import EDGE_VERTICES

        for ev in EDGE_VERTICES:
            assert len(ev) == 2

    def test_corner_offsets_tuples_of_three(self):
        from ocmesher._mc_tables import CORNER_OFFSETS

        for co in CORNER_OFFSETS:
            assert len(co) == 3


# ---------------------------------------------------------------------------
# _mc_tables: __all__ exports
# ---------------------------------------------------------------------------


class TestMcTablesExports:
    def test_all_defined(self):
        from ocmesher import _mc_tables

        assert hasattr(_mc_tables, "__all__")

    def test_all_contains_expected(self):
        from ocmesher._mc_tables import __all__

        for name in ("EDGE_TABLE", "TRI_TABLE", "EDGE_VERTICES", "CORNER_OFFSETS"):
            assert name in __all__


# ---------------------------------------------------------------------------
# _validation: shared usage
# ---------------------------------------------------------------------------


class TestValidationModule:
    def test_can_import_validate_cameras(self):
        from ocmesher._validation import validate_cameras

        assert callable(validate_cameras)

    def test_can_import_validate_bounds(self):
        from ocmesher._validation import validate_bounds

        assert callable(validate_bounds)

    def test_can_import_validate_kernels(self):
        from ocmesher._validation import validate_kernels

        assert callable(validate_kernels)

    def test_all_defined(self):
        from ocmesher import _validation

        assert hasattr(_validation, "__all__")

    def test_core_reexports_validation(self):
        """core.py must re-export validators so existing code still works."""
        from ocmesher.core import _validate_bounds, _validate_cameras, _validate_kernels

        assert callable(_validate_cameras)
        assert callable(_validate_bounds)
        assert callable(_validate_kernels)


# Section: _constants values and exports


class TestConstantsModule:
    def test_camera_data_stride_value(self):
        from ocmesher._constants import CAMERA_DATA_STRIDE

        assert CAMERA_DATA_STRIDE == 23

    def test_sdf_batch_size_positive(self):
        from ocmesher._constants import SDF_BATCH_SIZE

        assert SDF_BATCH_SIZE > 0

    def test_denom_eps_positive(self):
        from ocmesher._constants import DENOM_EPS

        assert 0 < DENOM_EPS < 1e-6

    def test_corner_quant_scale_positive(self):
        from ocmesher._constants import CORNER_QUANT_SCALE

        assert CORNER_QUANT_SCALE > 0

    def test_max_sdf_workers_positive(self):
        from ocmesher._constants import MAX_SDF_WORKERS

        assert MAX_SDF_WORKERS >= 1

    def test_all_defined(self):
        from ocmesher import _constants

        assert hasattr(_constants, "__all__")

    def test_core_imports_stride(self):
        """CAMERA_DATA_STRIDE re-exported from core must match _constants."""
        from ocmesher._constants import CAMERA_DATA_STRIDE as CONST_STRIDE
        from ocmesher.core import CAMERA_DATA_STRIDE as CORE_STRIDE

        assert CORE_STRIDE == CONST_STRIDE


# Section: _types importability


class TestTypesModule:
    def test_can_import_sdf_kernel(self):
        from ocmesher._types import SDFKernel

        assert SDFKernel is not None

    def test_can_import_cameras_tuple(self):
        from ocmesher._types import CamerasTuple

        assert CamerasTuple is not None

    def test_can_import_bounds_like(self):
        from ocmesher._types import BoundsLike

        assert BoundsLike is not None

    def test_can_import_kernel_sequence(self):
        from ocmesher._types import KernelSequence

        assert KernelSequence is not None

    def test_can_import_mesh_result(self):
        from ocmesher._types import MeshResult

        assert MeshResult is not None

    def test_all_defined(self):
        from ocmesher import _types

        assert hasattr(_types, "__all__")


# ---------------------------------------------------------------------------
# py.typed marker
# ---------------------------------------------------------------------------


class TestPyTypedMarker:
    def test_py_typed_exists(self):
        from pathlib import Path

        import ocmesher

        pkg_dir = Path(ocmesher.__file__).parent
        assert (pkg_dir / "py.typed").exists()


# ---------------------------------------------------------------------------
# __all__ on public modules
# ---------------------------------------------------------------------------


class TestModuleAllExports:
    def test_core_has_all(self):
        from ocmesher import core

        assert hasattr(core, "__all__")
        assert "OcMesher" in core.__all__

    def test_torch_core_has_all(self):
        pytest.importorskip("torch")
        from ocmesher import torch_core

        assert hasattr(torch_core, "__all__")
        assert "TorchOcMesher" in torch_core.__all__

    def test_package_init_has_all(self):
        import ocmesher

        assert hasattr(ocmesher, "__all__")
        assert "OcMesher" in ocmesher.__all__
        assert "TorchOcMesher" in ocmesher.__all__


# ---------------------------------------------------------------------------
# __version__ from package metadata
# ---------------------------------------------------------------------------


class TestPackageVersion:
    def test_version_is_string(self):
        import ocmesher

        assert isinstance(ocmesher.__version__, str)

    def test_version_matches_metadata(self):
        from importlib.metadata import version

        import ocmesher

        assert ocmesher.__version__ == version("ocmesher")
