"""Shared helpers for benchmark result snapshots and comparisons."""

from __future__ import annotations

import json
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_run_metadata(*, label: str) -> dict[str, Any]:
    """Collect stable metadata for one benchmark snapshot."""
    return {
        "label": label,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "git_branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git_output("rev-parse", "HEAD"),
    }


def write_snapshot_json(output_path: str, results: object, *, label: str) -> None:
    """Write benchmark results plus execution metadata to JSON."""
    payload = {
        "metadata": build_run_metadata(label=label),
        "results": results,
    }
    with Path(output_path).open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_results_payload(path: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load a benchmark payload, supporting both legacy and snapshot JSON."""
    with Path(path).open() as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "results" in payload and isinstance(payload["results"], dict):
        metadata = payload.get("metadata")
        return payload["results"], metadata if isinstance(metadata, dict) else None
    if not isinstance(payload, dict):
        msg = f"Benchmark payload at {path} must be a JSON object"
        raise TypeError(msg)
    return payload, None


def compare_flat_results(file_a: str, file_b: str) -> list[str]:
    """Return formatted lines comparing flat benchmark result JSON files."""
    a, meta_a = load_results_payload(file_a)
    b, meta_b = load_results_payload(file_b)

    label_a = _meta_label(meta_a, "A")
    label_b = _meta_label(meta_b, "B")
    header = f"{'Benchmark':<40s}  {'Size':>8s}  {label_a:>10s}  {label_b:>10s}  {'Speedup':>8s}"
    lines = ["=" * len(header), f"Comparing: {file_a}  vs  {file_b}", "=" * len(header), header, "-" * len(header)]

    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            lines.append(f"  {name}: only in {'A' if name in a else 'B'}")
            continue
        for sz in sorted(a[name].keys(), key=_size_sort_key):
            if sz not in b[name]:
                continue
            row_a = a[name][sz]
            row_b = b[name][sz]
            if "mean_us" not in row_a or "mean_us" not in row_b:
                continue
            ta = row_a["mean_us"]
            tb = row_b["mean_us"]
            speedup = ta / tb if tb > 0 else float("inf")
            lines.append(f"  {name:<38s}  {sz:>8s}  {ta:>10.1f}  {tb:>10.1f}  {speedup:>7.2f}x")

    lines.append("=" * len(header))
    lines.append("Speedup > 1.0 means the second file is faster")
    return lines


def _git_output(*args: str) -> str | None:
    """Return stripped git command output, or None outside a git checkout."""
    try:
        result = subprocess.run(  # noqa: S603
            ["/usr/bin/git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _meta_label(metadata: dict[str, Any] | None, fallback: str) -> str:
    """Choose a short label for compare output."""
    if metadata is not None and isinstance(metadata.get("label"), str):
        return str(metadata["label"])[:10]
    return fallback


def _size_sort_key(value: str) -> int:
    """Sort numeric size strings numerically, everything else lexically later."""
    return int(value) if value.isdigit() else sys.maxsize


def nested_compare(
    results_a: dict[str, Any],
    results_b: dict[str, Any],
    *,
    path: str = "",
    threshold_pct: float = 5.0,
    verbose: bool = False,
) -> list[str]:
    """Recursively compare nested benchmark results and report significant differences.

    Args:
        results_a: First benchmark results (typically "before"/baseline)
        results_b: Second benchmark results (typically "after"/current)
        path: Current path in the nested structure (used for recursion)
        threshold_pct: Percentage difference to flag as significant (default 5%)
        verbose: Include all comparisons, not just significant ones

    Returns:
        List of formatted comparison lines
    """
    lines = []
    all_keys = set(results_a.keys()) | set(results_b.keys())

    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key

        if key not in results_a:
            lines.append(f"  + {current_path}: only in B")
            continue
        if key not in results_b:
            lines.append(f"  - {current_path}: only in A")
            continue

        val_a = results_a[key]
        val_b = results_b[key]

        if isinstance(val_a, dict) and isinstance(val_b, dict):
            lines.extend(nested_compare(val_a, val_b, path=current_path, threshold_pct=threshold_pct, verbose=verbose))
        else:
            comp_result = _compare_values(val_a, val_b, current_path, threshold_pct, verbose)
            if comp_result:
                lines.append(comp_result)

    return lines


def _compare_values(
    val_a: Any,
    val_b: Any,
    path: str,
    threshold_pct: float,
    verbose: bool,
) -> str | None:
    """Compare two values, returning a formatted line if significant difference found."""
    if val_a is None or val_b is None:
        return None

    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
        if val_a == 0 and val_b == 0:
            return None

        if val_a != 0:
            pct_diff = ((val_b - val_a) / abs(val_a)) * 100
        else:
            pct_diff = float("inf") if val_b > 0 else float("-inf")

        abs_diff = val_b - val_a  # noqa: F841 # kept for debugging
        is_significant = abs(pct_diff) >= threshold_pct

        if is_significant or verbose:
            direction = "↓" if pct_diff < 0 else "↑"
            return f"  {path}: {val_a:.4f} → {val_b:.4f} ({pct_diff:+.1f}% {direction})"

    return None


def compare_nested_results(
    file_a: str,
    file_b: str,
    *,
    threshold_pct: float = 5.0,
    verbose: bool = False,
) -> list[str]:
    """Compare two nested benchmark JSON files and report significant differences.

    Args:
        file_a: Path to first JSON file (baseline/before)
        file_b: Path to second JSON file (current/after)
        threshold_pct: Report differences >= this percentage (default 5%)
        verbose: Include all comparisons, not just significant ones

    Returns:
        Formatted comparison lines
    """
    a, meta_a = load_results_payload(file_a)
    b, meta_b = load_results_payload(file_b)

    label_a = _meta_label(meta_a, "A")
    label_b = _meta_label(meta_b, "B")

    header = f"Nested Benchmark Comparison: {label_a} vs {label_b}"
    lines = [
        "=" * 60,
        header,
        f"Files: {file_a} vs {file_b}",
        f"Threshold: {threshold_pct}% difference",
        "=" * 60,
    ]

    if meta_a and meta_b:
        lines.append(f"A: {meta_a.get('label', 'N/A')} @ {meta_a.get('timestamp_utc', 'N/A')}")
        lines.append(f"B: {meta_b.get('label', 'N/A')} @ {meta_b.get('timestamp_utc', 'N/A')}")
        lines.append("")

    comp_lines = nested_compare(a, b, threshold_pct=threshold_pct, verbose=verbose)

    if comp_lines:
        lines.extend(comp_lines)
    else:
        lines.append("  (no significant differences found)")

    lines.append("=" * 60)
    return lines
