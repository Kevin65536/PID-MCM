"""Tests for the read-only source-anchor inventory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from source_anchor_inventory import CSV_FIELDS, build_inventory, write_inventory


EXPECTED_METHODS = {
    "BIOT",
    "CBraMod",
    "REVE",
    "NormWear",
    "EFRM",
    "STA-Net",
    "BrainFusion",
}


def test_inventory_has_one_record_per_method_and_required_fields() -> None:
    rows = build_inventory()
    assert {row["method"] for row in rows} == EXPECTED_METHODS
    assert len(rows) == len(EXPECTED_METHODS)
    for row in rows:
        assert row["official_code_available"] is True
        assert row["official_code_url"].startswith("https://")
        assert row["anchor_status"]
        assert row["blocking_items"]
        assert row["evidence_paths"]
        assert row["runnable_command_or_not_verifiable"]


def test_inventory_does_not_mark_missing_source_artifacts_as_reproduction() -> None:
    rows = {row["method"]: row for row in build_inventory()}
    assert rows["EFRM"]["anchor_status"].startswith("not_verifiable")
    assert rows["BrainFusion"]["anchor_status"].startswith("not_verifiable")
    assert rows["STA-Net"]["anchor_status"].startswith("not_verifiable")
    assert rows["NormWear"]["anchor_status"].startswith("not_verifiable")
    assert rows["CBraMod"]["anchor_status"] == "conditionally_runnable_missing_external_data"


def test_write_inventory_emits_csv_json_and_report(tmp_path: Path) -> None:
    paths = write_inventory(tmp_path)
    assert set(paths) == {"inventory", "capability", "report"}
    assert all(path.is_file() for path in paths.values())

    with paths["inventory"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert set(rows[0]) == set(CSV_FIELDS)

    capability = json.loads(paths["capability"].read_text(encoding="utf-8"))
    assert capability["schema"] == "source_anchor_inventory_v1"
    assert capability["read_only"] is True
    assert capability["external_downloads_performed"] is False
    assert len(capability["methods"]) == 7
    assert "Source-anchor inventory" in paths["report"].read_text(encoding="utf-8")
