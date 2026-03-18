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
	
	# On macOS, set rpath so the binary can find libomp.dylib at runtime
	local rpath_flags=""
	if [ "${OS}" = "Darwin" ]; then
		if [ "${ARCH}" = "arm64" ]; then
			rpath_flags="-Wl,-rpath,/opt/homebrew/opt/llvm/lib"
		else
			rpath_flags="-Wl,-rpath,/usr/local/opt/llvm/lib"
		fi
	fi
	
	"${compiler}" "${LDFLAGS_ARRAY[@]}" -O3 -shared ${omp_flags} ${rpath_flags} -o "${out_so}" "${in_obj}"
}

patch_core_omp_runtime_macos() {
	if [ "${OS}" != "Darwin" ]; then
		return
	fi
	if [ ! -f "ocmesher/lib/core.so" ]; then
		return
	fi
	local pybin=""
	if [ -x ".venv/bin/python" ]; then
		pybin=".venv/bin/python"
	elif command -v python3 >/dev/null 2>&1; then
		pybin="python3"
	else
		return
	fi
	local torch_omp
	torch_omp=$(${pybin} -c 'from pathlib import Path
import importlib.util
spec = importlib.util.find_spec("torch")
if spec is None or not spec.origin:
    raise SystemExit(0)
libomp = Path(spec.origin).resolve().parent / "lib" / "libomp.dylib"
print(libomp if libomp.exists() else "")' 2>/dev/null)
	if [ -z "${torch_omp}" ] || [ ! -f "${torch_omp}" ]; then
		return
	fi
	local linked_omp
	linked_omp=$(/usr/bin/otool -L ocmesher/lib/core.so | awk '/libomp\.dylib/ {print $1; exit}')
	if [ -z "${linked_omp}" ] || [ "${linked_omp}" = "${torch_omp}" ]; then
		return
	fi
	echo "Patching core.so OpenMP runtime: ${linked_omp} -> ${torch_omp}"
	/usr/bin/install_name_tool -change "${linked_omp}" "${torch_omp}" ocmesher/lib/core.so
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
	patch_core_omp_runtime_macos
	rm -f ocmesher/lib/core.o

	echo "Building core_noomp.so"
	gx1 ocmesher/lib/core_noomp.o "-DOCMESHER_NO_OPENMP=1" ""
	gx2 ocmesher/lib/core_noomp.so ocmesher/lib/core_noomp.o ""
	rm -f ocmesher/lib/core_noomp.o
fi

echo "Building Rust core crate"
cargo build --release -p ocmesher-core --manifest-path ocmesher-rust/Cargo.toml
