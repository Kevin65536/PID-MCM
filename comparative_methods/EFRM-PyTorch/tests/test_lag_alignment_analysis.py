"""Unit tests for the public-only EFRM relative-crop lag audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efrm_pytorch.lag_alignment_analysis import (  # noqa: E402
    _deduplicate_evidence,
    _group_indices,
    _pairs_for_lag,
)


def _row(sample_id: str, start: float, *, condition: str = "A") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "dataset_id": "synthetic",
        "subject": "S01",
        "record_id": "R01",
        "join_key": "synthetic|S01|R01",
        "task_namespace": "synthetic:task",
        "condition": condition,
        "crop_start_s": start,
        "duration_s": 8.0,
    }


def test_duplicate_validation_rows_are_removed_after_consistency_check() -> None:
    metadata = (_row("a", 0.0), _row("b", 2.0), _row("a", 0.0))
    eeg = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    fnirs = eeg.copy()
    cosine = eeg @ fnirs.T
    reduced = _deduplicate_evidence(
        Path("synthetic.npz"), cosine, eeg, fnirs, metadata
    )
    reduced_cosine, _, _, reduced_metadata, indices, report = reduced
    assert reduced_cosine.shape == (2, 2)
    assert [row["sample_id"] for row in reduced_metadata] == ["a", "b"]
    assert indices == (0, 1)
    assert report["duplicate_row_count_removed"] == 1
    assert report["max_duplicate_embedding_abs_delta"] == 0.0


def test_same_record_grid_matching_reports_relative_offsets() -> None:
    metadata = tuple(_row(f"s{index}", float(index * 2)) for index in range(3))
    groups, _ = _group_indices(metadata, crop_grid_s=0.1)
    rows, columns, errors = _pairs_for_lag(
        metadata,
        groups,
        lag_s=2.0,
        crop_grid_s=0.1,
        tolerance_s=0.051,
    )
    assert list(zip(rows.tolist(), columns.tolist())) == [(0, 1), (1, 2)]
    assert np.allclose(errors, 0.0)


def test_matching_does_not_cross_records() -> None:
    metadata = (
        _row("a", 0.0),
        _row("b", 2.0),
        {**_row("c", 2.0), "join_key": "synthetic|S01|R02", "record_id": "R02"},
    )
    groups, _ = _group_indices(metadata, crop_grid_s=0.1)
    rows, columns, _ = _pairs_for_lag(
        metadata,
        groups,
        lag_s=2.0,
        crop_grid_s=0.1,
        tolerance_s=0.051,
    )
    assert list(zip(rows.tolist(), columns.tolist())) == [(0, 1)]

