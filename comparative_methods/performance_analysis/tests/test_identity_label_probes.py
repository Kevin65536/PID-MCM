from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

# Pytest's standalone path discovery can put this nested test directory ahead
# of the repository root.  Keep the test import deterministic without adding a
# repository-wide package marker to the existing comparative_methods tree.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.performance_analysis.identity_label_probes import (
    CacheRecord,
    _subject_within_splits,
    discover_caches,
    load_cache,
    run_cache,
)


def _write_npz_cache(tmp_path: Path, *, protected: bool = False, constant_dataset: bool = True) -> CacheRecord:
    tmp_path.mkdir(parents=True, exist_ok=True)
    n_subjects, rows_per_subject = 4, 6
    n = n_subjects * rows_per_subject
    subjects = np.repeat([f"subject_{i:02d}" for i in range(1, n_subjects + 1)], rows_per_subject)
    sessions = np.tile(np.repeat(["session_00", "session_01", "session_02"], 2), n_subjects)
    targets = np.tile([0, 1, 0, 1, 0, 1], n_subjects)
    dataset = (
        np.full(n, "eeg_fnirs_single_trial", dtype="<U64")
        if constant_dataset
        else np.tile(["dataset_a", "dataset_b"], n // 2)
    )
    sample_ids = np.asarray(
        [f"{d}|{s}|{session}|event={i}" for i, (d, s, session) in enumerate(zip(dataset, subjects, sessions))]
    )
    # Make subject identity deliberately easy while retaining a task signal.
    features = np.column_stack(
        [
            np.asarray([int(s[-2:]) for s in subjects], dtype=np.float32),
            targets.astype(np.float32),
            np.arange(n, dtype=np.float32) / 10.0,
        ]
    )
    npz_path = tmp_path / "cache.npz"
    sidecar_path = tmp_path / "cache.json"
    np.savez(
        npz_path,
        features=features,
        targets=targets,
        dataset_indices=np.arange(n),
        subjects=subjects,
        sample_ids=sample_ids,
    )
    sidecar_path.write_text(
        json.dumps(
            {
                "schema": "synthetic",
                "feature_cache_key": "synthetic",
                "protected_test_opened": protected,
                "outer_fold": 0,
            }
        ),
        encoding="utf-8",
    )
    return CacheRecord(
        method="biot",
        task="motor_imagery",
        feature_path=npz_path,
        metadata_path=npz_path,
        sidecar_path=sidecar_path,
        cache_id="synthetic",
        outer_fold=0,
        schema="synthetic",
    )


def test_load_cache_extracts_session_dataset_and_rejects_protected(tmp_path: Path) -> None:
    record = _write_npz_cache(tmp_path)
    data = load_cache(record)
    assert set(data.sessions) == {"session_00", "session_01", "session_02"}
    assert set(data.dataset_ids) == {"eeg_fnirs_single_trial"}
    protected = _write_npz_cache(tmp_path / "protected", protected=True)
    with pytest.raises(RuntimeError, match="protected-opened"):
        load_cache(protected)


def test_within_subject_split_has_no_subject_in_test_without_train_rows() -> None:
    subjects = np.repeat(np.asarray(["s1", "s2", "s3"]), 10)
    repeat, train, test = next(_subject_within_splits(subjects, repeats=1, test_fraction=0.2, seed=7))
    assert repeat == 0
    assert set(subjects[train]) == set(subjects[test]) == {"s1", "s2", "s3"}
    assert not set(train).intersection(test)


def test_group_safe_task_and_session_probes_and_non_estimable_subject(tmp_path: Path) -> None:
    record = _write_npz_cache(tmp_path)
    cache = load_cache(record)
    metrics, capabilities = run_cache(cache, cv_splits=2, subject_repeats=2, subject_test_fraction=0.2, seed=17)
    assert metrics
    assert all(row["standardizer_fit_scope"] == "train_only" for row in metrics)
    assert all(row["split_kind"] == "group_kfold_subject" for row in metrics if row["probe"] in {"task", "session"})
    assert any(
        row["probe"] == "subject_closed_set" and row["split_kind"] == "closed_set_row_split_subject"
        for row in metrics
    )
    status = {(row["probe"], row["status"]) for row in capabilities}
    assert ("subject_cross_subject", "not_estimable") in status
    assert ("subject_closed_set", "available") in status
    assert ("dataset", "unavailable_constant") in status


def test_discover_cache_selection_is_outer_zero(tmp_path: Path) -> None:
    root = tmp_path
    # Build a minimal BIOT cache tree with outer folds 0 and 1.
    cache_dir = root / "comparative_methods/BIOT/runs/feature_cache_v2/motor_imagery"
    cache_dir.mkdir(parents=True)
    for fold in (0, 1):
        npz = cache_dir / f"fold{fold}.npz"
        json_path = cache_dir / f"fold{fold}.json"
        np.savez(
            npz,
            features=np.ones((2, 1), dtype=np.float32),
            targets=np.asarray([0, 1]),
            dataset_indices=np.asarray([0, 1]),
            subjects=np.asarray(["subject_01", "subject_02"]),
            sample_ids=np.asarray([
                "dataset|subject_01|session_00|event=0",
                "dataset|subject_02|session_00|event=1",
            ]),
        )
        json_path.write_text(json.dumps({"outer_fold": fold, "feature_cache_key": str(fold), "protected_test_opened": False}), encoding="utf-8")
    records = discover_caches(root, ["biot"], ["motor_imagery"])
    assert [record.outer_fold for record in records] == [0]
    records = discover_caches(root, ["biot"], ["motor_imagery"], all_cache_replicates=True)
    assert [record.outer_fold for record in records] == [0, 1]
