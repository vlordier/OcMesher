#!/bin/bash
# Build script for OcMesher performance benchmarks
# Supports arm64 Apple M-series (M4) with appropriate compiler flags

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

OS=$(uname -s)
ARCH=$(uname -m)

# ---------------------------------------------------------------------------
# Compiler selection (mirrors install.sh logic)
# ---------------------------------------------------------------------------
if [ -n "$CXX" ]; then
    compiler="$CXX"
else
    if [ "${OS}" = "Linux" ]; then
        compiler="g++"
    elif [ "${OS}" = "Darwin" ]; then
        if [ "${ARCH}" = "arm64" ]; then
            compiler="/opt/homebrew/opt/llvm/bin/clang++"
        else
            compiler="/usr/local/opt/llvm/bin/clang++"
        fi
    else
        echo "Unsupported OS: ${OS}"
        exit 1
    fi
fi

echo "=== OcMesher Benchmark Build ==="
echo "OS           : ${OS}"
echo "Architecture : ${ARCH}"
echo "Compiler     : ${compiler}"

# ---------------------------------------------------------------------------
# Base flags
# ---------------------------------------------------------------------------
BASE_FLAGS="-std=c++17 -O3 -DNDEBUG"

# OpenMP flags
OMP_FLAGS=""
if [ "${OS}" = "Darwin" ]; then
    OMP_FLAGS="-Xpreprocessor -fopenmp"
    if [ "${ARCH}" = "arm64" ]; then
        OMP_LIB_DIR="/opt/homebrew/opt/libomp/lib"
        OMP_FLAGS="${OMP_FLAGS} -I/opt/homebrew/opt/libomp/include"
    else
        OMP_LIB_DIR="/usr/local/opt/libomp/lib"
        OMP_FLAGS="${OMP_FLAGS} -I/usr/local/opt/libomp/include"
    fi
    OMP_LINK="-lomp -L${OMP_LIB_DIR} -Wl,-rpath,${OMP_LIB_DIR}"
else
    OMP_FLAGS="-fopenmp"
    OMP_LINK="-fopenmp"
fi

# ---------------------------------------------------------------------------
# Architecture-specific flags for arm64 Apple M-series
# ---------------------------------------------------------------------------
ARCH_FLAGS=""
if [ "${OS}" = "Darwin" ] && [ "${ARCH}" = "arm64" ]; then
    echo "Detected Apple Silicon (arm64) - enabling M-series optimisations"

    # Apple M-series (M1/M2/M3/M4) specific flags:
    # -mcpu=apple-m4   : Target Apple M4 microarchitecture
    # -mtune=native    : Tune for the current CPU (falls back if not M4)
    # These enable NEON SIMD automatically on arm64

    # Use apple-m4 if the compiler supports it, else fall back to apple-m1
    MCPU="apple-m1"
    if ${compiler} -mcpu=apple-m4 -x c++ -c /dev/null -o /dev/null 2>/dev/null; then
        MCPU="apple-m4"
    elif ${compiler} -mcpu=apple-m2 -x c++ -c /dev/null -o /dev/null 2>/dev/null; then
        MCPU="apple-m2"
    fi
    ARCH_FLAGS="-mcpu=${MCPU}"
    echo "Using -mcpu=${MCPU}"

elif [ "${ARCH}" = "aarch64" ]; then
    # Linux arm64 (e.g. Graviton, Ampere)
    echo "Detected Linux arm64"
    ARCH_FLAGS="-march=armv8.2-a+fp16+dotprod"
    if ${compiler} -march=armv9-a -x c++ -c /dev/null -o /dev/null 2>/dev/null; then
        ARCH_FLAGS="-march=armv9-a"
    fi
fi

# Allow user to override via environment
BENCH_CXXFLAGS="${BENCH_CXXFLAGS:-}"

ALL_FLAGS="${BASE_FLAGS} ${ARCH_FLAGS} ${OMP_FLAGS} ${BENCH_CXXFLAGS}"
ALL_LINK="${OMP_LINK}"

echo "CXXFLAGS     : ${ALL_FLAGS}"
echo "LDFLAGS      : ${ALL_LINK}"
echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
mkdir -p "${BUILD_DIR}"

echo "Building bench_core..."
${compiler} ${ALL_FLAGS} \
    -o "${BUILD_DIR}/bench_core" \
    "${SCRIPT_DIR}/bench_core.cpp" \
    ${ALL_LINK} -lm

echo "Building bench_arm64_neon..."
${compiler} ${ALL_FLAGS} \
    -o "${BUILD_DIR}/bench_arm64_neon" \
    "${SCRIPT_DIR}/bench_arm64_neon.cpp" \
    ${ALL_LINK} -lm

echo ""
echo "Build complete. Binaries in: ${BUILD_DIR}/"
echo "  ${BUILD_DIR}/bench_core"
echo "  ${BUILD_DIR}/bench_arm64_neon"
echo ""
echo "Run with:"
echo "  ${BUILD_DIR}/bench_core [--iterations N] [--warmup N]"
echo "  ${BUILD_DIR}/bench_arm64_neon [--iterations N] [--warmup N]"
