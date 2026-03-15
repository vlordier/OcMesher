#!/bin/bash
# Build OcMesher's C++ core with an optional optimisation profile.
#
# Usage:
#   bash install.sh [PROFILE]
#
# PROFILE selects the set of compiler flags used for compilation.
# When PROFILE is omitted the value of the PROFILE environment variable is
# used; if that is also unset the default profile "native" is used.
#
# Available profiles
# ------------------
#   baseline    -O0               (no optimisation, useful as timing baseline)
#   O1          -O1
#   O2          -O2 -fopenmp
#   O3          -O3 -fopenmp      (portable, no arch-specific flags)
#   native      -O3 -fopenmp -march=native  (default – matches prior behaviour)
#   fast        native + -ffast-math
#   aggressive  fast  + -funroll-loops
#   lto         fast  + -flto=thin (Clang) / -flto (GCC)
#
# The compiled shared library is always written to ocmesher/lib/core.so so
# that the Python package picks it up without any code changes.
#
# CXXFLAGS / LDFLAGS are honoured as additional flags on top of the profile.
#
# Examples
#   bash install.sh                   # default (native)
#   bash install.sh baseline          # no optimisation for baseline timing
#   PROFILE=fast bash install.sh      # same via env var
#   CXX=clang++ bash install.sh lto   # use a specific compiler

set -e

OS=$(uname -s)
ARCH=$(uname -m)

# ── compiler selection ────────────────────────────────────────────────────────
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

# ── extra flags from environment ──────────────────────────────────────────────
CXXFLAGS_ARRAY=()
if [[ -n "${CXXFLAGS:-}" ]]; then
    read -ra CXXFLAGS_ARRAY <<< "${CXXFLAGS}"
fi
LDFLAGS_ARRAY=()
if [[ -n "${LDFLAGS:-}" ]]; then
    read -ra LDFLAGS_ARRAY <<< "${LDFLAGS}"
fi

# ── architecture-specific march flags ────────────────────────────────────────
# -march=native enables all CPU ISA extensions available on the build machine
# (AVX2/AVX-512 on x86_64, NEON/SVE on aarch64/arm64).
# The flags are guarded by a compile check so that unsupported options are
# silently dropped (e.g. cross-compile environments without -march=native).
MARCH_FLAGS=""
case "${OS}" in
    Linux)
        case "${ARCH}" in
            x86_64)  MARCH_FLAGS="-march=native -mtune=native" ;;
            aarch64) MARCH_FLAGS="-march=native" ;;
        esac
        ;;
    Darwin)
        # Both Apple Silicon (arm64) and Intel Macs benefit from -march=native
        MARCH_FLAGS="-march=native"
        ;;
esac

if [ -n "${MARCH_FLAGS}" ]; then
    if ! "${compiler}" ${MARCH_FLAGS} -x c++ - -o /dev/null < /dev/null 2>/dev/null; then
        echo "Warning: ${compiler} does not support ${MARCH_FLAGS}, falling back to portable build."
        MARCH_FLAGS=""
    fi
fi

# ── profile selection ─────────────────────────────────────────────────────────
PROFILE="${1:-${PROFILE:-native}}"

case "${PROFILE}" in
    baseline)
        OPT_FLAGS="-O0"
        ACTIVE_MARCH=""
        ;;
    O1)
        OPT_FLAGS="-O1"
        ACTIVE_MARCH=""
        ;;
    O2)
        OPT_FLAGS="-O2 -fopenmp"
        ACTIVE_MARCH=""
        ;;
    O3)
        OPT_FLAGS="-O3 -fopenmp"
        ACTIVE_MARCH=""
        ;;
    native)
        OPT_FLAGS="-O3 -fopenmp"
        ACTIVE_MARCH="${MARCH_FLAGS}"
        ;;
    fast)
        OPT_FLAGS="-O3 -fopenmp -ffast-math"
        ACTIVE_MARCH="${MARCH_FLAGS}"
        ;;
    aggressive)
        OPT_FLAGS="-O3 -fopenmp -ffast-math -funroll-loops"
        ACTIVE_MARCH="${MARCH_FLAGS}"
        ;;
    lto)
        # Thin LTO is a Clang-specific flag; fall back to standard -flto for GCC.
        if "${compiler}" --version 2>&1 | grep -qi clang; then
            LTO_FLAG="-flto=thin"
        else
            LTO_FLAG="-flto"
        fi
        OPT_FLAGS="-O3 -fopenmp -ffast-math ${LTO_FLAG}"
        ACTIVE_MARCH="${MARCH_FLAGS}"
        ;;
    *)
        echo "Unknown profile: '${PROFILE}'"
        echo "Valid profiles: baseline O1 O2 O3 native fast aggressive lto"
        exit 1
        ;;
esac

echo "Building OcMesher with profile '${PROFILE}': -std=c++17 ${OPT_FLAGS} ${ACTIVE_MARCH}"

gx1() { "${compiler}" "${CXXFLAGS_ARRAY[@]}" -std=c++17 ${OPT_FLAGS} ${ACTIVE_MARCH} -c -fpic "$@"; }
gx2() { "${compiler}" "${LDFLAGS_ARRAY[@]}" ${OPT_FLAGS} -shared "$@"; }

mkdir -p ocmesher/lib
gx1 -o ocmesher/lib/core.o ocmesher/source/core.cpp
gx2 -o ocmesher/lib/core.so ocmesher/lib/core.o
rm -f ocmesher/lib/core.o

echo "Build complete → ocmesher/lib/core.so  (profile: ${PROFILE})"
