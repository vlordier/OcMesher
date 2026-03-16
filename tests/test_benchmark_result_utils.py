from __future__ import annotations

import json

from benchmarks import result_utils


def test_load_results_payload_supports_legacy_json(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"bench": {"10": {"mean_us": 1.0}}}))

    results, metadata = result_utils.load_results_payload(str(path))

    assert results["bench"]["10"]["mean_us"] == 1.0
    assert metadata is None


def test_load_results_payload_supports_snapshot_json(tmp_path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"metadata": {"label": "snap"}, "results": {"bench": {"10": {"mean_us": 2.0}}}}))

    results, metadata = result_utils.load_results_payload(str(path))

    assert results["bench"]["10"]["mean_us"] == 2.0
    assert metadata == {"label": "snap"}


def test_compare_flat_results_uses_snapshot_labels(tmp_path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"metadata": {"label": "develop"}, "results": {"bench": {"10": {"mean_us": 10.0}}}}))
    b.write_text(json.dumps({"metadata": {"label": "opt"}, "results": {"bench": {"10": {"mean_us": 5.0}}}}))

    lines = result_utils.compare_flat_results(str(a), str(b))

    assert any("develop" in line and "opt" in line for line in lines)
    assert any("2.00x" in line for line in lines)
