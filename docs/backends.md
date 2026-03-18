# Backend Guide

This page summarizes backend selection and Rust integration options for OcMesher.

## Backend Selection

Use `make_ocmesher(...)` for a simple backend/device entry point:

```python
from ocmesher import make_ocmesher

mesher_cpp = make_ocmesher(cameras, bounds, backend="cpp")
mesher_torch = make_ocmesher(cameras, bounds, backend="torch", device="mps")
mesher_rust = make_ocmesher(cameras, bounds, backend="rust", device="cpu")
```

Backend options:

- `backend="cpp"`: original C++ implementation.
- `backend="torch"`: PyTorch implementation (CPU/CUDA/MPS).
- `backend="rust"`: Rust wrapper backend with native and optional `tch` paths.

## Rust Wrapper Controls

`backend="rust"` supports extra runtime controls:

- `device`: `"cpu"`, `"mps"`, `"cuda"`, or `"cuda:<index>"`.
- `stream_policy`: `"sync"` or `"auto"` (`"auto"` downgrades to `"sync"` on CPU).
- `kernel_runtime`: `"auto"`, `"native"`, or `"tch"`.

Example forcing Rust `tch` kernels:

```python
from ocmesher import make_ocmesher

mesher = make_ocmesher(
    cameras,
    bounds,
    backend="rust",
    device="mps",
    kernel_runtime="tch",
)
```

## Building the Rust Extension

Base Rust extension build:

```bash
uv sync --extra rust
uv run maturin develop --release --manifest-path ocmesher-rust/crates/ocmesher-py/Cargo.toml
```

Build with `tch-kernels` enabled:

```bash
uv sync --extra rust --extra torch
uv run maturin develop --release --features tch-kernels --manifest-path ocmesher-rust/crates/ocmesher-py/Cargo.toml
```

Notes:

- The Rust integration currently loads `ocmesher/lib/core.so`.
- Run `bash install.sh` before using the Rust extension so `core.so` exists.
- Adaptive routing env vars:
  - `OCMESHER_TCH_MPS_ADAPTIVE=1`
  - `OCMESHER_TCH_CUDA_ADAPTIVE=1`
  - `OCMESHER_TCH_ADAPTIVE_DEBUG=1`

For complete setup, examples, and benchmark guidance, see the Home page and API reference.

For pipeline-focused integration steps, see [Infinigen Integration Quickstart](infinigen.md).
