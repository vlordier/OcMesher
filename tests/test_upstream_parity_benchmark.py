"""Unit tests for upstream parity benchmark helpers."""

from __future__ import annotations

from pathlib import Path

import benchmarks.upstream_parity as upstream
from benchmarks.upstream_parity import compare_signatures


def test_compare_signatures_reports_exact_match() -> None:
    lhs = {"verts": 10, "faces": 20, "verts_sum": 1.5, "faces_sum": 120}
    rhs = {"verts": 10, "faces": 20, "verts_sum": 1.5, "faces_sum": 120}
    delta = compare_signatures(lhs, rhs)

    assert delta["matches"] is True
    assert delta["verts_delta"] == 0
    assert delta["faces_delta"] == 0
    assert delta["faces_sum_delta"] == 0
    assert abs(float(delta["verts_sum_delta"])) == 0.0


def test_compare_signatures_reports_deltas() -> None:
    lhs = {"verts": 12, "faces": 19, "verts_sum": 1.75, "faces_sum": 130}
    rhs = {"verts": 10, "faces": 20, "verts_sum": 1.5, "faces_sum": 120}
    delta = compare_signatures(lhs, rhs)

    assert delta["matches"] is False
    assert delta["verts_delta"] == 2
    assert delta["faces_delta"] == -1
    assert delta["faces_sum_delta"] == 10
    assert float(delta["verts_sum_delta"]) == 0.25


def test_compare_signatures_respects_tolerance() -> None:
    lhs = {"verts": 10, "faces": 20, "verts_sum": 1.5000000000001, "faces_sum": 120}
    rhs = {"verts": 10, "faces": 20, "verts_sum": 1.5, "faces_sum": 120}

    assert compare_signatures(lhs, rhs, atol=1e-10)["matches"] is True
    assert compare_signatures(lhs, rhs, atol=1e-14)["matches"] is False


def test_run_upstream_parity_orchestrates_build_and_worktree(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))

        class _Result:
            returncode = 0

        return _Result()

    signatures = [
        {"verts": 10, "faces": 20, "verts_sum": 1.5, "faces_sum": 120},
        {"verts": 10, "faces": 20, "verts_sum": 1.5, "faces_sum": 120},
    ]

    monkeypatch.setattr(upstream.subprocess, "run", _fake_run)
    monkeypatch.setattr(upstream, "_compute_signature_in_repo", lambda *_args, **_kwargs: signatures.pop(0))
    monkeypatch.setattr(
        upstream.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(Path(upstream.tempfile.gettempdir()) / "ocmesher-upstream-test"),
    )
    monkeypatch.setattr(upstream.shutil, "rmtree", lambda *_args, **_kwargs: None)

    result = upstream.run_upstream_parity(Path(), upstream_ref="main", python_exe="python")

    assert bool(result["delta"]["matches"])
    flat_calls = [" ".join(cmd) for cmd in calls]
    assert any("install.sh" in c for c in flat_calls)
    assert any("worktree add" in c for c in flat_calls)
    assert any("worktree remove" in c for c in flat_calls)
