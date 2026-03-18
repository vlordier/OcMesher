# API Reference

## `OcMesher`

The main C++-backed mesher. Accepts SDF kernels and camera data, runs coarse-to-fine octree meshing, and returns mesh objects.

::: ocmesher.core.OcMesher

----

## `TorchOcMesher`

PyTorch-based mesher with GPU acceleration. Drop-in replacement for `OcMesher` with CUDA, MPS, and CPU support.

::: ocmesher.torch_core.TorchOcMesher

----

## `RustOcMesher`

Rust-native backend with optimized SDF kernels. Provides native sphere and plane SDF implementations.

::: ocmesher.rust_backend.RustOcMesher

----

## Factory Functions

### `make_ocmesher`

Factory function to create the appropriate mesher backend.

::: ocmesher.factory.make_ocmesher

----

## Utility Functions

### `validate_cameras`

Validate and normalize camera data.

::: ocmesher.core._validate_cameras

### `validate_bounds`

Validate bounds array.

::: ocmesher.core._validate_bounds

### `validate_kernels`

Validate SDF kernel functions.

::: ocmesher.core._validate_kernels
