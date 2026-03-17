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

gx1() {
	local out_obj="$1"
	local defines="$2"
	local omp_flags="$3"
	"${compiler}" "${CXXFLAGS_ARRAY[@]}" -O3 -std=c++17 ${MARCH_FLAGS} ${defines} -c -fpic ${omp_flags} -o "${out_obj}" ocmesher/source/core.cpp
}

gx2() {
	local out_so="$1"
	local in_obj="$2"
	local omp_flags="$3"
	"${compiler}" "${LDFLAGS_ARRAY[@]}" -O3 -shared ${omp_flags} -o "${out_so}" "${in_obj}"
}

mkdir -p ocmesher/lib

if [[ "${OCMESHER_DISABLE_OPENMP:-0}" == "1" ]]; then
	echo "Building only core_noomp.so (OCMESHER_DISABLE_OPENMP=1)"
	gx1 ocmesher/lib/core_noomp.o "-DOCMESHER_NO_OPENMP=1" ""
	gx2 ocmesher/lib/core_noomp.so ocmesher/lib/core_noomp.o ""
	cp ocmesher/lib/core_noomp.so ocmesher/lib/core.so
	rm -f ocmesher/lib/core_noomp.o
else
	echo "Building core.so (OpenMP)"
	gx1 ocmesher/lib/core.o "" "-fopenmp"
	gx2 ocmesher/lib/core.so ocmesher/lib/core.o "-fopenmp"
	rm -f ocmesher/lib/core.o

	echo "Building core_noomp.so"
	gx1 ocmesher/lib/core_noomp.o "-DOCMESHER_NO_OPENMP=1" ""
	gx2 ocmesher/lib/core_noomp.so ocmesher/lib/core_noomp.o ""
	rm -f ocmesher/lib/core_noomp.o
fi

echo "Building Rust core crate"
cargo build --release -p ocmesher-core --manifest-path ocmesher-rust/Cargo.toml
