from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from comparative_methods.performance_analysis.classical_eeg_baselines_report import (
    load_fold_predictions,
    summarize_task,
)


def _write_fold(root: Path, fold: int, indices: list[int], true: list[int], pred: list[int], subjects: list[str]) -> None:
    np.savez_compressed(
        root / f"outer{fold}_predictions.npz",
        dataset_index=np.asarray(indices),
        y_true=np.asarray(true),
        y_pred=np.asarray(pred),
        subject=np.asarray(subjects),
        join_key=np.asarray([f"record|{subject}" for subject in subjects]),
    )


def test_sample_identity_audit_detects_repeated_fold_rows(tmp_path: Path) -> None:
    summary = {
        "task": "visual",
        "protected_test_opened": False,
        "folds": [{"outer_fold": 0}, {"outer_fold": 1}],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_fold(tmp_path, 0, [1, 2], [0, 1], [0, 1], ["S1", "S2"])
    _write_fold(tmp_path, 1, [2, 3], [1, 0], [0, 0], ["S2", "S3"])
    provenance, rows = load_fold_predictions(tmp_path)
    assert provenance["raw_validation_prediction_count"] == 4
    assert provenance["unique_sample_id_count"] == 3
    assert provenance["repeated_prediction_row_count"] == 1
    assert len(rows) == 4


def test_subject_summary_uses_subject_mean_and_seeded_bootstrap(tmp_path: Path) -> None:
    summary = {
        "task": "visual",
        "protected_test_opened": False,
        "folds": [{"outer_fold": 0}],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_fold(tmp_path, 0, [1, 2, 3, 4], [0, 0, 1, 1], [0, 1, 1, 1], ["S1", "S1", "S2", "S2"])
    result = summarize_task(tmp_path, bootstrap_seed=20260816, bootstrap_replicates=100)
    assert result["subject_count"] == 2
    assert result["primary_endpoint"] == "subject_level_macro_f1_after_fold_average"
    assert len(result["subject_level_macro_f1"]["bootstrap_95ci"]) == 2
    assert result["subject_level_macro_f1"]["bootstrap_seed"] == 20260816
