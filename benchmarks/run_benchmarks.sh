#!/bin/bash
# Run all OcMesher benchmarks and produce a summary report
# Usage: ./run_benchmarks.sh [--iterations N] [--warmup N] [--output FILE]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ITERATIONS=20
WARMUP=3
OUTPUT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            if [[ -z "${2:-}" ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --iterations requires a numeric argument" >&2; exit 1
            fi
            ITERATIONS="$2"; shift 2 ;;
        --warmup)
            if [[ -z "${2:-}" ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --warmup requires a numeric argument" >&2; exit 1
            fi
            WARMUP="$2"; shift 2 ;;
        --output)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --output requires a file path argument" >&2; exit 1
            fi
            OUTPUT="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--iterations N] [--warmup N] [--output FILE]"
            echo ""
            echo "Options:"
            echo "  --iterations N   Number of timed iterations per benchmark (default: 20)"
            echo "  --warmup N       Number of warmup iterations (default: 3)"
            echo "  --output FILE    Write output to file in addition to stdout"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Check if benchmarks are built
if [ ! -f "${BUILD_DIR}/bench_core" ] || [ ! -f "${BUILD_DIR}/bench_arm64_neon" ]; then
    echo "Benchmarks not built. Building now..."
    bash "${SCRIPT_DIR}/build_benchmarks.sh"
    echo ""
fi

ARGS=("--iterations" "${ITERATIONS}" "--warmup" "${WARMUP}")

run_bench() {
    local name="$1"
    local binary="$2"
    echo "============================================================"
    echo " Running: ${name}"
    echo "============================================================"
    echo ""
    "${binary}" "${ARGS[@]}"
    echo ""
}

# Capture output to file if requested
if [ -n "${OUTPUT}" ]; then
    exec > >(tee "${OUTPUT}") 2>&1
fi

echo "OcMesher Performance Benchmark Suite"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Host: $(uname -snrm)"
echo ""

run_bench "Core Algorithm Benchmarks" "${BUILD_DIR}/bench_core"
run_bench "ARM64 NEON SIMD Benchmarks" "${BUILD_DIR}/bench_arm64_neon"

echo "============================================================"
echo " All benchmarks complete"
echo "============================================================"

if [ -n "${OUTPUT}" ]; then
    echo ""
    echo "Results saved to: ${OUTPUT}"
fi
