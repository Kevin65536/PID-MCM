import sys
from pathlib import Path

import numpy as np
import pytest

METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch.protocol_bridge_analysis import (
    classification_subject_rows,
    load_fivefold_subject_rows,
    regression_subject_rows,
    summarize_subject_rows,
)


def test_classification_subject_rows_uses_subject_as_replication_unit():
    target = np.asarray([0, 1, 0, 1])
    probability = np.asarray([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.7, 0.3],
    ])
    subjects = np.asarray(["s1", "s1", "s2", "s2"])
    rows = classification_subject_rows(
        target,
        probability,
        subjects,
        task="motor_imagery",
        protocol="trial_random",
        source_artifact="synthetic",
    )
    assert [row["subject"] for row in rows] == ["s1", "s2"]
    assert all(row["metric"] == "macro_f1" for row in rows)
    assert all(row["sample_count"] == 2 for row in rows)
    summary = summarize_subject_rows(rows)
    assert summary["subject_count"] == 2
    assert summary["sample_count"] == 4
    assert summary["ci_lower"] <= summary["estimate_subject_mean"] <= summary["ci_upper"]


def test_regression_subject_rows_computes_ccc_over_valid_coordinates():
    target = np.asarray([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 3.0], [4.0, 5.0]],
    ])
    prediction = target.copy()
    mask = np.ones_like(target, dtype=bool)
    subjects = np.asarray(["s1", "s2"])
    rows = regression_subject_rows(
        target,
        prediction,
        mask,
        subjects,
        task="refed_regression",
        protocol="cross_subject",
        source_artifact="synthetic",
    )
    assert len(rows) == 2
    assert all(row["metric"] == "concordance_correlation" for row in rows)
    assert all(row["estimate"] == pytest.approx(1.0) for row in rows)
    assert all(row["valid_coordinate_count"] == 4 for row in rows)


def test_load_fivefold_subject_rows_requires_unique_out_of_fold_sample_ids(tmp_path):
    root = tmp_path / "fivefold"
    task_root = root / "folds" / "sample_random" / "motor_imagery"
    for fold in range(5):
        output = task_root / f"outer{fold}" / "evaluation"
        output.mkdir(parents=True)
        np.savez_compressed(
            output / "protected_predictions.npz",
            prediction=np.asarray(
                [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],
                dtype=np.float32,
            ),
            target=np.asarray([0, 1, 0, 1], dtype=np.int64),
            target_valid_mask=np.ones(4, dtype=bool),
            subject=np.asarray(["s1", "s1", "s2", "s2"]),
            sample_id=np.asarray([f"sample_{fold}_{index}" for index in range(4)]),
        )
    rows, inventory = load_fivefold_subject_rows(root, "sample_random", "motor_imagery")
    assert len(rows) == 2
    assert inventory["fold_count"] == 5
    assert inventory["sample_count"] == 20
    assert inventory["subject_count"] == 2

    duplicate = task_root / "outer4" / "evaluation" / "protected_predictions.npz"
    np.savez_compressed(
        duplicate,
        prediction=np.asarray(
            [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32,
        ),
        target=np.asarray([0, 1, 0, 1], dtype=np.int64),
        target_valid_mask=np.ones(4, dtype=bool),
        subject=np.asarray(["s1", "s1", "s2", "s2"]),
        sample_id=np.asarray(["sample_0_0", "sample_0_1", "sample_0_2", "sample_0_3"]),
    )
    with pytest.raises(RuntimeError, match="sample IDs are not unique"):
        load_fivefold_subject_rows(root, "sample_random", "motor_imagery")
