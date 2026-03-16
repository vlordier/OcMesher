"""Unit tests for upstream parity benchmark helpers."""

from __future__ import annotations

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
