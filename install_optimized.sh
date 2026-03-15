#!/bin/bash
# Optimized build for Apple M4 (arm64) benchmarking.
# Applies aggressive compiler optimizations beyond the default -O3.

shopt -s expand_aliases
set -e

OS=$(uname -s)
ARCH=$(uname -m)

if [ "${OS}" != "Darwin" ] || [ "${ARCH}" != "arm64" ]; then
    echo "This script targets macOS arm64 (Apple Silicon). Detected: ${OS} ${ARCH}"
    exit 1
fi

if [ -n "$CXX" ]; then
    compiler="$CXX"
else
    compiler="/opt/homebrew/opt/llvm/bin/clang++"
fi

if ! command -v "$compiler" &>/dev/null; then
    echo "Compiler not found: $compiler"
    echo "Install with: brew install llvm"
    exit 1
fi

echo "Compiler: $compiler"
$compiler --version | head -1

# ── Optimization flags ──────────────────────────────────────────────
# -O3                    : full optimisation suite
# -ffast-math            : relax IEEE fp for speed (reassociation, reciprocals…)
# -fno-finite-math-only  : re-allow infinity/NaN (code uses numeric_limits::infinity)
# -mcpu=apple-m4         : tune for M4 micro-architecture (NEON, AMX)
# -flto=thin             : link-time optimisation (thin for fast link)
# -fvectorize            : auto-vectorise loops (NEON)
# -fslp-vectorize        : SLP vectorisation
# -funroll-loops         : unroll small loops
# -fomit-frame-pointer   : free up a register
# -DNDEBUG              : disable assert() overhead
# -fopenmp              : OpenMP parallelism
# -ffp-contract=fast    : fuse FP multiply-add into FMA instructions

OPT_CXXFLAGS="-O3 -ffast-math -fno-finite-math-only \
  -mcpu=apple-m4 \
  -flto=thin -fvectorize -fslp-vectorize -funroll-loops \
  -fomit-frame-pointer -ffp-contract=fast -DNDEBUG"

OPT_LDFLAGS="-flto=thin"

alias gx1="${compiler} ${OPT_CXXFLAGS} \$CXXFLAGS -c -fpic -fopenmp "
alias gx2="${compiler} ${OPT_LDFLAGS} \$LDFLAGS -O3 -shared -fopenmp "

echo ""
echo "Building with optimised flags:"
echo "  CXXFLAGS: ${OPT_CXXFLAGS}"
echo ""

mkdir -p ocmesher/lib
gx1 -o ocmesher/lib/core.o ocmesher/source/core.cpp
gx2 -o ocmesher/lib/core.so ocmesher/lib/core.o

echo ""
echo "✓ Optimised build complete: ocmesher/lib/core.so"
