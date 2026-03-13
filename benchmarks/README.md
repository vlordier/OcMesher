# OcMesher Performance Benchmarks

Performance analysis suite for OcMesher's C++ core, with a focus on **arm64 Apple M-series (M4)** processors.

## Overview

These benchmarks measure the performance of OcMesher's key computational kernels, providing:

- **Micro-benchmarks**: Individual kernel-level measurements (determinant, dot product, coordinate transforms, SDF bisection, triangle–segment intersection)
- **Data structure benchmarks**: `std::map` and `std::priority_queue` performance with OcMesher's key types
- **Algorithm benchmarks**: End-to-end octree expansion, vertex enumeration, and camera projection
- **System benchmarks**: Memory bandwidth and OpenMP parallel scaling
- **ARM64 NEON SIMD benchmarks**: Scalar vs. NEON comparisons for critical hot-path kernels

Each benchmark reports **robust statistics**: min, max, mean, median, and standard deviation over configurable iterations with warmup.

## Quick Start

```bash
# Build benchmarks
bash benchmarks/build_benchmarks.sh

# Run all benchmarks
bash benchmarks/run_benchmarks.sh

# Run with custom parameters
bash benchmarks/run_benchmarks.sh --iterations 50 --warmup 5

# Save results to file
bash benchmarks/run_benchmarks.sh --output results.txt
```

## Build

The build script (`build_benchmarks.sh`) automatically detects the platform and applies optimal compiler flags.

### arm64 Apple M-series

On macOS arm64, the build script:

1. Selects the Homebrew LLVM clang++ compiler
2. Applies `-mcpu=apple-m4` (falls back to `-mcpu=apple-m2` or `-mcpu=apple-m1` if the compiler does not support M4 targeting)
3. Enables NEON SIMD (automatic on arm64)
4. Enables OpenMP for multi-core scaling tests

### Custom Compiler / Flags

```bash
# Use a specific compiler
CXX=/usr/bin/clang++ bash benchmarks/build_benchmarks.sh

# Add custom flags
BENCH_CXXFLAGS="-march=native" bash benchmarks/build_benchmarks.sh
```

## Benchmark Descriptions

### Core Algorithm Benchmarks (`bench_core`)

| Benchmark | Description | Relevance |
|-----------|-------------|-----------|
| **3x3 determinant** | 100k evaluations of the 3×3 determinant used in triangle–segment intersection | `tri_seg_intersect` in mesh construction |
| **Triangle–segment intersection** | 200k full tri-seg intersection tests | `construct_faces` inner loop |
| **Coordinate compute** | 500k vertex coordinate transforms | `compute_coords` / `compute_center` |
| **SDF bisection** | 100k vertices × 10 bisection iterations with position output | `update_verts` refinement loop |
| **Map insert / lookup** | 100k `key_cube` insertions and lookups into `std::map` | Used throughout for vertex/edge deduplication |
| **Priority queue** | 100k push + pop operations | Coarse octree expansion scheduling |
| **Octree expansion** | Build 50k-node octree via recursive expansion | `expand_octree` in coarse step |
| **Vertex enumeration** | 200 nodes at grid level 4 | `enumerate_vertices` per leaf node |
| **Camera projection** | 50k cubes × 4 cameras | `projected_size` in coarse + fine steps |
| **Memory bandwidth** | 64 MB sequential memcpy | Baseline for M-series unified memory |
| **OpenMP scaling** | 500k visibility checks at 1, 2, 4, ... threads | `vis_filter` parallel loop scaling |

### ARM64 NEON Benchmarks (`bench_arm64_neon`)

| Benchmark | Scalar | NEON | Notes |
|-----------|--------|------|-------|
| **3D dot product** | Loop over 1M pairs | `float64x2_t` SIMD | Projection inner loop |
| **3×3 determinant** | Standard formula | NEON cofactor parallelisation | Triangle intersection |
| **Coordinate transform** | Per-vertex loop | `float64x2_t` for x,y pair | `compute_center` |
| **SDF bipolar detection** | 8-element scalar loop | `float32x4_t` sign compare | `update_verts` condition |
| **Vertex position generation** | 8-corner loop per vertex | NEON add/sub pairs | `update_verts` output |

On non-arm64 platforms, NEON benchmarks are compiled but the NEON variants are skipped at runtime, reporting only scalar results.

## Interpreting Results

### Output Format

```
  <benchmark name>  samples=20  mean=  1.234 ms  median=  1.200 ms  min=  1.100 ms  max=  1.500 ms  stddev=  0.100 ms
```

- **mean**: Average across all timed iterations
- **median**: Middle value (robust against outliers)
- **stddev**: Variability; lower is better for consistency
- **min**: Best-case performance

### NEON Speedup

For NEON benchmarks, a speedup line is printed:

```
  <name>  speedup = 1.50x  (scalar=1.500 ms, neon=1.000 ms)
```

A speedup > 1.0× indicates the NEON version is faster. On Apple M4, typical speedups for double-precision NEON are 1.2–1.8× for math-heavy kernels and higher for float32 operations.

## Apple M4 Performance Notes

The Apple M4 processor features:

- **10 CPU cores**: 4 performance (P) + 6 efficiency (E) cores
- **NEON SIMD**: 128-bit vector registers, native on all arm64
- **Unified memory**: Shared between CPU and GPU, high bandwidth (~100 GB/s on M4 Pro)
- **Large caches**: 192 KB L1i + 128 KB L1d per P-core, 16 MB shared L2
- **High single-thread IPC**: Wide decode and out-of-order execution

### Optimisation Opportunities for OcMesher

1. **NEON SIMD**: The determinant, coordinate transform, and SDF bipolar detection kernels benefit from explicit vectorisation
2. **Memory access patterns**: The octree traversal and map-heavy code benefits from cache-friendly access; unified memory reduces copy overhead
3. **OpenMP scaling**: The `vis_filter` and `update_verts` loops scale well with P-core count (4 threads typical sweet spot)
4. **Branch prediction**: M4's excellent branch predictor handles the sign-change detection in SDF bisection efficiently

## Requirements

- C++17 compatible compiler (clang++ or g++)
- OpenMP support (optional but recommended)
- On macOS: Homebrew LLVM (`brew install llvm`) for OpenMP support
