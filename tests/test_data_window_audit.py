from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from comparative_methods.performance_analysis import data_window_audit as audit


def _row(
    *,
    subject: str,
    record: str,
    join_key: str,
    trial: str,
    condition: str = "A",
) -> dict[str, str]:
    return {
        "subject": subject,
        "record_id": record,
        "join_key": join_key,
        "trial_group": trial,
        "condition": condition,
    }


def test_task_surface_covers_all_comparison_tasks() -> None:
    assert set(audit.TASK_ORDER) == {
        "motor_imagery",
        "mental_arithmetic",
        "wg",
        "nback",
        "dsr",
        "visual",
        "refed_regression",
    }
    assert set(audit.TASK_DISPLAY.values()) == {
        "MI",
        "MA",
        "WG",
        "nback",
        "DSR",
        "Visual",
        "REFED",
    }


def test_canonical_source_id_keeps_event_and_window_identity() -> None:
    metadata = {
        "dataset_id": "simultaneous_eeg_nirs",
        "subject": "VP001",
        "record_id": "cnt_nback",
        "event_index": 7,
        "window_offset_s": 2.5,
    }
    assert audit.canonical_source_sample_id(metadata) == (
        "simultaneous_eeg_nirs|VP001|cnt_nback|event=7|offset_ms=2500"
    )
    assert audit.canonical_source_sample_id({**metadata, "event_index": 8}) != audit.canonical_source_sample_id(metadata)


def test_group_overlap_uses_subject_record_composite() -> None:
    train = [_row(subject="S01", record="session_00", join_key="S01|r0", trial="S01|r0|e0")]
    validation = [_row(subject="S02", record="session_00", join_key="S02|r0", trial="S02|r0|e0")]
    overlap = audit._group_overlap(train, validation)
    # Raw record labels can repeat between subjects, while contamination groups
    # remain disjoint.
    assert overlap["record_id"]["disjoint"] is False
    assert overlap["subject_record"]["disjoint"] is True
    assert overlap["join_key"]["disjoint"] is True
    assert overlap["trial_group"]["disjoint"] is True


def test_public_json_rejects_protected_path_before_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    path = tmp_path / "protected" / "outer0.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(audit.AuditError, match="protected path"):
        audit.load_public_json(path)


def test_public_json_rejects_protected_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    path = tmp_path / "public.json"
    path.write_text(json.dumps({"protected_indices": [1]}), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="protected index fields"):
        audit.load_public_json(path)


def test_npz_cache_audit_reports_duplicate_and_missing_ids(tmp_path: Path) -> None:
    cache = tmp_path / "features.npz"
    np.savez(
        cache,
        features=np.zeros((3, 2), dtype=np.float32),
        sample_ids=np.asarray(["a", "a", "b"]),
        targets=np.asarray([0, 0, 1]),
    )
    result = audit._audit_npz_cache(
        cache,
        expected_ids={"a", "b", "c"},
        manifest={"protected_test_opened": False},
    )
    assert result["status"] == "duplicate_sample_ids"
    assert result["sample_count"] == 3
    assert result["unique_sample_count"] == 2
    assert result["duplicate_sample_ids"] == {"a": 2}
    assert result["missing_expected_sample_ids"] == ["c"]
    assert result["unexpected_sample_ids"] == []


def test_npz_cache_without_sample_ids_is_fail_closed(tmp_path: Path) -> None:
    cache = tmp_path / "features.npz"
    np.savez(cache, features=np.zeros((2, 3), dtype=np.float32))
    result = audit._audit_npz_cache(cache, expected_ids=None, manifest={})
    assert result["status"] == "missing_sample_ids"


def test_expected_cache_id_scheme_matches_efrm_adapter_and_other_source_caches() -> None:
    expected = {("mi", 0): {"source": {"source-id"}, "adapter": {"adapter-id"}}}
    # EFRM crop IDs contain a run-time crop start; membership is checked via
    # dataset_indices, while other public caches use transparent source IDs.
    assert audit._cache_expected_ids(expected, task="mi", fold=0, method="efrm_sync") is None
    assert audit._cache_expected_ids(expected, task="mi", fold=0, method="reve") == {
        "source-id"
    }


def test_validate_indices_detects_overlap_and_duplicates() -> None:
    errors = audit._validate_indices(
        train=[0, 1, 1],
        validation=[1, 2],
        dataset_length=3,
        task="nback",
        fold=0,
    )
    assert any("duplicate train indices" in error for error in errors)
    assert any("train/validation index overlap" in error for error in errors)


def test_report_preserves_findings_and_summarizes_cache_statuses(tmp_path: Path) -> None:
    report = {
        "status": "pass_with_findings",
        "registry": {"path": "/repo/registry.json", "file_sha256": "abc"},
        "tasks": [],
        "errors": [],
        "findings": [
            "EFRM validation export repeats rows; diagonal-only positives make naive full-matrix retrieval inappropriate."
        ],
        "public_manifests_and_caches": [
            {
                "method": "efrm_sync",
                "task": "mi",
                "outer_fold": 0,
                "manifest": "/repo/public/efrm.json",
                "status": "pass",
                "cache": {"status": "pass"},
            },
            {
                "method": "efrm_sync",
                "task": "ma",
                "outer_fold": 0,
                "manifest": "/repo/public/efrm_missing.json",
                "status": "pass",
                "cache": {"status": "not_declared"},
            },
        ],
        "efrm_alignment_validation_mixing": [
            {
                "artifact": "exclude_eeg_fnirs_single_trial_stage_a",
                "row_count": 7559,
                "unique_sample_count": 4787,
                "duplicate_row_count": 2772,
                "duplicate_id_count": 777,
                "duplicate_embedding_exact": True,
                "positive_mask_diagonal_true": 7559,
                "positive_mask_off_diagonal_true": 0,
                "naive_full_matrix_retrieval_appropriate": False,
                "status": "finding_repeated_validation_rows",
            }
        ],
    }
    output = tmp_path / "REPORT.md"
    audit._write_report(output, report)
    text = output.read_text(encoding="utf-8")
    assert "Status: **pass_with_findings**" in text
    assert "## Findings" in text
    assert "finding_repeated_validation_rows" in text
    assert "naive full-matrix retrieval" in text
    assert "| efrm_sync | not_declared | 1 |" in text
    # Per-manifest rows remain in the authoritative CSV, not Markdown.
    assert text.count("/repo/public/") == 2
