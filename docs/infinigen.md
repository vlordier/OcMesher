# Infinigen Integration Quickstart

This page focuses on using OcMesher from Infinigen with explicit backend/runtime choices.

## 1. Build OcMesher in the active environment

```bash
uv sync --extra rust --extra torch
bash install.sh
uv run maturin develop --release --features tch-kernels --manifest-path ocmesher-rust/crates/ocmesher-py/Cargo.toml
```

What this gives you:

- `ocmesher/lib/core.so` built and loadable.
- `ocmesher_rust` extension installed in your active Python environment.
- Rust `tch` kernels available for CPU, MPS, and CUDA routes.

## 2. Select backend/runtime from Python

Use explicit backend selection so behavior is deterministic when wired into pipeline code.

```python
from ocmesher import make_ocmesher

# Rust + native kernels (auto picks native on CPU)
mesher_rust_native = make_ocmesher(
    cameras,
    bounds,
    backend="rust",
    device="cpu",
)

# Rust + tch kernels forced on CPU (useful for parity/profiling)
mesher_rust_tch_cpu = make_ocmesher(
    cameras,
    bounds,
    backend="rust",
    device="cpu",
    kernel_runtime="tch",
)

# Rust + tch kernels on Apple Silicon
mesher_rust_tch_mps = make_ocmesher(
    cameras,
    bounds,
    backend="rust",
    device="mps",
    kernel_runtime="tch",
)
```

Accepted Rust wrapper controls:

- `device`: `"cpu"`, `"mps"`, `"cuda"`, or `"cuda:<index>"`.
- `stream_policy`: `"sync"` or `"auto"`.
- `kernel_runtime`: `"auto"`, `"native"`, or `"tch"`.

## 3. Optional adaptive routing (Rust tch)

For per-batch CPU-vs-GPU routing calibration:

```bash
# Apple Silicon
export OCMESHER_TCH_MPS_ADAPTIVE=1

# CUDA
export OCMESHER_TCH_CUDA_ADAPTIVE=1

# Optional diagnostics
export OCMESHER_TCH_ADAPTIVE_DEBUG=1
```

## 4. Troubleshooting checklist

- `core.so not found`: rerun `bash install.sh` in the OcMesher repo.
- `ImportError: ocmesher_rust`: rerun `maturin develop` command above.
- `unsupported tch device`: use one of the accepted values exactly.
- Unexpected CPU fallback with `stream_policy="auto"`: this is expected on CPU-only device selection.

## 5. Verification smoke test

```python
from ocmesher import make_ocmesher

mesher = make_ocmesher(cameras, bounds, backend="rust", device="mps", kernel_runtime="tch")
meshes, in_view_tags = mesher([sdf])
print(len(meshes), len(in_view_tags))
```

If this runs, Infinigen-side integration can use the same `make_ocmesher(...)` call pattern.
