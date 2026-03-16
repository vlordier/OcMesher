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
