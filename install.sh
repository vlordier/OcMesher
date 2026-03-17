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
		MARCH_FLAGS="-march=native"
		;;
esac

if [ -n "${MARCH_FLAGS}" ]; then
	if ! "${compiler}" ${MARCH_FLAGS} -x c++ - -o /dev/null < /dev/null 2>/dev/null; then
		echo "Warning: ${compiler} does not support ${MARCH_FLAGS}, falling back to portable build."
		MARCH_FLAGS=""
	fi
fi

OPENMP_FLAGS="-fopenmp"
OPENMP_DEFINES=""
if [[ "${OCMESHER_DISABLE_OPENMP:-0}" == "1" ]]; then
	OPENMP_FLAGS=""
	OPENMP_DEFINES="-DOCMESHER_NO_OPENMP=1"
	echo "Building core.so without OpenMP (OCMESHER_DISABLE_OPENMP=1)"
fi

gx1() { "${compiler}" "${CXXFLAGS_ARRAY[@]}" -O3 -std=c++17 ${MARCH_FLAGS} ${OPENMP_DEFINES} -c -fpic ${OPENMP_FLAGS} "$@"; }
gx2() { "${compiler}" "${LDFLAGS_ARRAY[@]}" -O3 -shared ${OPENMP_FLAGS} "$@"; }

mkdir -p ocmesher/lib
gx1 -o ocmesher/lib/core.o ocmesher/source/core.cpp
gx2 -o ocmesher/lib/core.so ocmesher/lib/core.o
rm -f ocmesher/lib/core.o

echo "Building Rust core crate"
cargo build --release -p ocmesher-core --manifest-path ocmesher-rust/Cargo.toml
