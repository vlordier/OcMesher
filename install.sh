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
gx1() { "${compiler}" "${CXXFLAGS_ARRAY[@]}" -O3 -c -fpic -fopenmp "$@"; }
gx2() { "${compiler}" "${LDFLAGS_ARRAY[@]}" -O3 -shared -fopenmp "$@"; }

mkdir -p ocmesher/lib
gx1 -o ocmesher/lib/core.o ocmesher/source/core.cpp
gx2 -o ocmesher/lib/core.so ocmesher/lib/core.o
rm -f ocmesher/lib/core.o
