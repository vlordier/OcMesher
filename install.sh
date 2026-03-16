#!/bin/bash

set -e

OS=$(uname -s)
ARCH=$(uname -m)

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
        echo "Unsupported OS"
        exit 1
    fi
fi

CXXFLAGS_ARRAY=()
if [[ -n "${CXXFLAGS:-}" ]]; then
    read -ra CXXFLAGS_ARRAY <<< "${CXXFLAGS}"
fi
LDFLAGS_ARRAY=()
if [[ -n "${LDFLAGS:-}" ]]; then
    read -ra LDFLAGS_ARRAY <<< "${LDFLAGS}"
fi

if [ "${OS}" = "Darwin" ]; then
    libomp_prefix=""
    if [ -d "/opt/homebrew/opt/libomp" ]; then
        libomp_prefix="/opt/homebrew/opt/libomp"
    elif [ -d "/usr/local/opt/libomp" ]; then
        libomp_prefix="/usr/local/opt/libomp"
    fi

    if [ -n "${libomp_prefix}" ]; then
        CXXFLAGS_ARRAY+=("-I${libomp_prefix}/include")
        LDFLAGS_ARRAY+=("-L${libomp_prefix}/lib" "-Wl,-rpath,${libomp_prefix}/lib")
    fi
fi

# ---------------------------------------------------------------------------
# Architecture-specific optimisation flags
# -march=native enables all CPU ISA extensions available on the build machine
# (AVX2/AVX-512 on x86_64, NEON/SVE on aarch64/arm64).
# -std=c++17 enables the C++17 standard for better compiler optimisations.
# The flags are guarded by a compile check so that unsupported options are
# silently dropped (e.g. cross-compile environments without -march=native).
# ---------------------------------------------------------------------------
MARCH_FLAGS=""
case "${OS}" in
    Linux)
        case "${ARCH}" in
            x86_64)
                MARCH_FLAGS="-march=native -mtune=native"
                ;;
            aarch64)
                MARCH_FLAGS="-march=native"
                ;;
        esac
        ;;
    Darwin)
        # Both Apple Silicon (arm64) and Intel Macs benefit from -march=native
        MARCH_FLAGS="-march=native"
        ;;
esac

# Verify the compiler accepts -march=native (may fail in some cross-compile
# or minimal container setups) and fall back silently if not.
if [ -n "${MARCH_FLAGS}" ]; then
    if ! "${compiler}" ${MARCH_FLAGS} -x c++ - -o /dev/null < /dev/null 2>/dev/null; then
        echo "Warning: ${compiler} does not support ${MARCH_FLAGS}, falling back to portable build."
        MARCH_FLAGS=""
    fi
fi

gx1() { "${compiler}" "${CXXFLAGS_ARRAY[@]}" -O3 -std=c++17 ${MARCH_FLAGS} -c -fpic -fopenmp "$@"; }
gx2() { "${compiler}" "${LDFLAGS_ARRAY[@]}" -O3 -shared -fopenmp "$@"; }

# Clean up intermediate object files on exit (success or failure).
cleanup() { rm -f ocmesher/lib/core.o; }
trap cleanup EXIT

mkdir -p ocmesher/lib
gx1 -o ocmesher/lib/core.o ocmesher/source/core.cpp
gx2 -o ocmesher/lib/core.so ocmesher/lib/core.o
