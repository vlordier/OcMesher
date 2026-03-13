#!/usr/bin/env bash
# run_benchmarks.sh – build OcMesher with every optimisation profile and
# record benchmark results in benchmark/results.json.
#
# Usage:
#   cd <repo-root>
#   bash benchmark/run_benchmarks.sh [SCENE] [PIXELS_PER_CUBE] [RUNS] [BASELINE_PROFILE]
#
# Defaults:
#   SCENE             sphere
#   PIXELS_PER_CUBE   8
#   RUNS              3
#   BASELINE_PROFILE  O3   (profile used as the 1.00× reference in the summary)
#
# The script iterates over all build profiles defined in install.sh:
#   baseline  O1  O2  O3  native  fast  aggressive  lto
#
# For each profile it:
#   1. Rebuilds ocmesher/lib/core.so with that profile's flags.
#   2. Runs benchmark/benchmark.py and appends results to benchmark/results.json.
#
# Results are also printed in a human-readable summary at the end.
#
# Requirements:
#   pip install -r requirements.txt
#   A working C++ toolchain (see install.sh for compiler selection).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCENE="${1:-sphere}"
PIXELS_PER_CUBE="${2:-8}"
RUNS="${3:-3}"
BASELINE_PROFILE="${4:-O3}"
RESULTS_FILE="${SCRIPT_DIR}/results.json"

# Profiles to benchmark (edit this list to skip/add profiles)
PROFILES=(baseline O1 O2 O3 native fast aggressive lto)

cd "${REPO_ROOT}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           OcMesher optimisation benchmark                ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Scene            : ${SCENE}"
echo "║  pixels_per_cube  : ${PIXELS_PER_CUBE}"
echo "║  Runs per profile : ${RUNS}"
echo "║  Baseline profile : ${BASELINE_PROFILE}"
echo "║  Results file     : ${RESULTS_FILE}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Remove previous results so we start fresh
rm -f "${RESULTS_FILE}"

failed_profiles=()

for PROFILE in "${PROFILES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  Building profile: ${PROFILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if bash install.sh "${PROFILE}"; then
        echo "✓  Build succeeded for profile '${PROFILE}'"
    else
        echo "✗  Build FAILED for profile '${PROFILE}' – skipping benchmark"
        failed_profiles+=("${PROFILE}")
        continue
    fi

    echo ""
    echo "▶  Running benchmark (profile=${PROFILE}, scene=${SCENE}, runs=${RUNS})"

    if python benchmark/benchmark.py \
        --profile  "${PROFILE}"         \
        --scene    "${SCENE}"           \
        --pixels-per-cube "${PIXELS_PER_CUBE}" \
        --runs     "${RUNS}"            \
        --out      "${RESULTS_FILE}"; then
        echo "✓  Benchmark complete for profile '${PROFILE}'"
    else
        echo "✗  Benchmark FAILED for profile '${PROFILE}'"
        failed_profiles+=("${PROFILE}")
    fi

    echo ""
done

# ── human-readable comparison ────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  Results summary                         ║"
echo "╚══════════════════════════════════════════════════════════╝"

if [ -f "${RESULTS_FILE}" ]; then
    python - "${RESULTS_FILE}" "${BASELINE_PROFILE}" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)

if not data:
    print("No results recorded.")
    sys.exit(0)

baseline_label = sys.argv[2] if len(sys.argv) > 2 else "O3"

# Find the requested baseline; fall back to the first entry
baseline_mean = None
for r in data:
    if r.get("profile") == baseline_label:
        baseline_mean = r["total_wall"]["mean"]
        break
if baseline_mean is None:
    baseline_mean = data[0]["total_wall"]["mean"]
    baseline_label = data[0].get("profile", "first")

header = f"{'Profile':<12}  {'mean (s)':>10}  {'min (s)':>10}  {'max (s)':>10}  {'speedup':>8}"
print(header)
print("─" * len(header))
for r in data:
    prof   = r.get("profile", "?")
    tw     = r["total_wall"]
    mean   = tw["mean"]
    mn     = tw["min"]
    mx     = tw["max"]
    speedup = baseline_mean / mean if mean > 0 else float("nan")
    print(f"{prof:<12}  {mean:>10.3f}  {mn:>10.3f}  {mx:>10.3f}  {speedup:>7.2f}x")

print()
print(f"(speedup is relative to profile '{baseline_label}')")
PYEOF
else
    echo "No results file found at ${RESULTS_FILE}"
fi

if [ "${#failed_profiles[@]}" -gt 0 ]; then
    echo ""
    echo "⚠  Profiles that failed: ${failed_profiles[*]}"
fi
