#!/bin/bash
# Build OcMesher's C++ core with an optional optimisation profile.
#
# Usage:
#   bash install.sh [PROFILE]
#
# PROFILE selects the set of compiler flags used for compilation.
# When PROFILE is omitted the value of the PROFILE environment variable is
# used; if that is also unset the default profile "O3" is used.
#
# Available profiles
# ------------------
#   baseline    -O0  (no optimisation, useful as timing baseline)
#   O1          -O1
#   O2          -O2 -fopenmp
#   O3          -O3 -fopenmp  (default – matches the original build)
#   native      -O3 -fopenmp -march=native
#   fast        -O3 -fopenmp -march=native -ffast-math
#   aggressive  -O3 -fopenmp -march=native -ffast-math -funroll-loops
#   lto         -O3 -fopenmp -march=native -ffast-math -flto=thin
#               (thin-LTO works on both clang and gcc ≥ 10)
#
# The compiled shared library is always written to ocmesher/lib/core.so so
# that the Python package picks it up without any code changes.
#
# Examples
#   bash install.sh                  # default (O3)
#   bash install.sh native           # architecture-tuned build
#   PROFILE=fast bash install.sh     # same via env var
#   CXX=clang++ bash install.sh lto  # use a specific compiler

shopt -s expand_aliases
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

# ── profile selection ─────────────────────────────────────────────────────────
PROFILE="${1:-${PROFILE:-O3}}"

case "${PROFILE}" in
    baseline)
        OPT_FLAGS="-O0"
        ;;
    O1)
        OPT_FLAGS="-O1"
        ;;
    O2)
        OPT_FLAGS="-O2 -fopenmp"
        ;;
    O3)
        OPT_FLAGS="-O3 -fopenmp"
        ;;
    native)
        OPT_FLAGS="-O3 -fopenmp -march=native"
        ;;
    fast)
        OPT_FLAGS="-O3 -fopenmp -march=native -ffast-math"
        ;;
    aggressive)
        OPT_FLAGS="-O3 -fopenmp -march=native -ffast-math -funroll-loops"
        ;;
    lto)
        # Thin LTO is a Clang-specific flag; fall back to standard -flto for GCC.
        if "${compiler}" --version 2>&1 | grep -qi clang; then
            LTO_FLAG="-flto=thin"
        else
            LTO_FLAG="-flto"
        fi
        OPT_FLAGS="-O3 -fopenmp -march=native -ffast-math ${LTO_FLAG}"
        ;;
    *)
        echo "Unknown profile: '${PROFILE}'"
        echo "Valid profiles: baseline O1 O2 O3 native fast aggressive lto"
        exit 1
        ;;
esac

echo "Building OcMesher with profile '${PROFILE}': ${OPT_FLAGS}"

# Allow callers to inject extra flags on top of the profile.
# CXXFLAGS / LDFLAGS are honoured as before.
alias gx1="${compiler} ${OPT_FLAGS} \$CXXFLAGS -c -fpic "
alias gx2="${compiler} ${OPT_FLAGS} \$LDFLAGS -shared "

mkdir -p ocmesher/lib
gx1 -o ocmesher/lib/core.o ocmesher/source/core.cpp
gx2 -o ocmesher/lib/core.so ocmesher/lib/core.o

echo "Build complete → ocmesher/lib/core.so  (profile: ${PROFILE})"
