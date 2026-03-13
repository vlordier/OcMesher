# OcMesher Benchmark Suite

This directory contains tooling to measure the performance impact of different
C++ compiler optimisation profiles on the OcMesher meshing pipeline.

---

## Quick start

```bash
# 1. install Python dependencies (once)
pip install -r requirements.txt

# 2. run all benchmark profiles end-to-end
bash benchmark/run_benchmarks.sh

# 3. use a custom baseline for the speedup column (default: O3)
bash benchmark/run_benchmarks.sh sphere 8 3 baseline

# 4. inspect results
cat benchmark/results.json
```

---

## Build profiles

`install.sh` now accepts an optional first argument (or the `PROFILE`
environment variable) that selects a set of compiler flags:

| Profile      | Flags                                                          | Description                          |
|--------------|----------------------------------------------------------------|--------------------------------------|
| `baseline`   | `-O0`                                                         | No optimisations – timing floor      |
| `O1`         | `-O1`                                                         | Minimal optimisations                |
| `O2`         | `-O2 -fopenmp`                                                | Standard optimisations + OpenMP      |
| `O3`         | `-O3 -fopenmp`                                                | **Default** (matches original build) |
| `native`     | `-O3 -fopenmp -march=native`                                  | CPU-specific instructions (AVX2/NEON)|
| `fast`       | `-O3 -fopenmp -march=native -ffast-math`                      | Non-strict IEEE FP for extra speed   |
| `aggressive` | `-O3 -fopenmp -march=native -ffast-math -funroll-loops`       | Maximal vectorisation + loop unroll  |
| `lto`        | `-O3 -fopenmp -march=native -ffast-math -flto=thin` (Clang) / `-flto` (GCC) | Thin / Full Link-Time Optimisation |

### Notes for Apple Silicon (M4 / arm64)

* The installer selects `/opt/homebrew/opt/llvm/bin/clang++` on macOS arm64
  automatically – ensure LLVM is installed via `brew install llvm`.
* `-march=native` on an M4 enables ARMv9 NEON and SVE2 auto-vectorisation.
* `-flto=thin` (thin LTO) is supported by both Apple Clang and Homebrew LLVM
  and avoids the large link-time memory overhead of full LTO.

### Notes for AMD64 / x86-64

* `-march=native` enables AVX2 / FMA / AVX-512 (if available) automatically.
* Use `CXX=g++` or `CXX=clang++` to switch compilers explicitly.

---

## Running individual profiles

```bash
# build a specific profile and leave core.so in place
bash install.sh native

# run the benchmark Python script against the current build
python benchmark/benchmark.py --profile native --scene sphere --runs 3

# override the number of OpenMP threads
python benchmark/benchmark.py --omp-threads 4 --out benchmark/results.json
```

---

## Benchmark script options

```
usage: benchmark.py [-h] [--profile PROFILE] [--scene {sphere,torus,gyroid,heightmap}]
                    [--pixels-per-cube N] [--runs N] [--omp-threads N] [--out FILE]

Options:
  --profile          Build profile label (informational)
  --scene            SDF scene to mesh (default: sphere)
  --pixels-per-cube  Mesh resolution; higher = finer, slower (default: 8)
  --runs             Repeated runs for stable timing (default: 1)
  --omp-threads      Override OMP_NUM_THREADS for the C++ core
  --out FILE         Append JSON result to FILE (creates file if absent)
```

---

## Understanding the results

`run_benchmarks.sh` writes every run to `benchmark/results.json` and prints a
summary table like:

```
Profile        mean (s)    min (s)    max (s)   speedup
────────────────────────────────────────────────────────
baseline         12.504     12.312     12.731     0.41x
O1                8.831      8.710      8.994     0.58x
O2                6.102      6.031      6.195     0.84x
O3                5.140      5.091      5.210     1.00x
native            4.213      4.189      4.237     1.22x
fast              3.902      3.878      3.934     1.32x
aggressive        3.810      3.791      3.842     1.35x
lto               3.750      3.730      3.789     1.37x
```

The **speedup** column is relative to the `O3` profile (the original default).

---

## Extra environment variables

| Variable           | Effect                                         |
|--------------------|------------------------------------------------|
| `OMP_NUM_THREADS`  | Number of threads for the C++ OpenMP regions   |
| `CXX`              | Override the C++ compiler used by install.sh   |
| `CXXFLAGS`         | Extra compiler flags appended after the profile|
| `LDFLAGS`          | Extra linker flags appended after the profile  |
| `PROFILE`          | Default profile for install.sh (env-var form)  |
