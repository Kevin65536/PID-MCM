from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = METHOD_ROOT / "adapters"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from brainfusion_gpu.stacking import (
    VIEW_ORDER,
    FoldLocalStackingClassifier,
    StackingConfig,
)


def _features(seed: int, count: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray([(index // 4) % 2 for index in range(count)], dtype=np.int64)
    output = {}
    for view_index, view in enumerate(VIEW_ORDER):
        values = rng.normal(size=(count, 6))
        values[:, view_index] += (labels * 2 - 1) * (0.5 + view_index * 0.1)
        output[view] = values
    return output, labels


def _config() -> StackingConfig:
    return StackingConfig(
        inner_folds=3,
        seed=17,
        linear_svm_c_values=(0.1, 1.0),
        rbf_svm_c_values=(1.0,),
        random_forest_estimators=24,
        meta_svm_c=1.0,
    )


def test_stacking_selection_and_meta_fit_are_outer_train_group_local(tmp_path: Path) -> None:
    train, labels = _features(17, 48)
    groups = [f"subject-{index // 4}" for index in range(48)]
    ids = [f"train-{index}" for index in range(48)]
    model = FoldLocalStackingClassifier(_config()).fit(
        train, labels, groups=groups, sample_ids=ids
    )
    audit = model.audit_state()
    assert audit["fit_sample_count"] == 48
    assert audit["fit_group_count"] == 12
    assert audit["inner_validation_covers_training_exactly_once"] is True
    assert audit["validation_or_protected_labels_consumed"] is False
    assert all(row["group_overlap"] is False for row in audit["inner_folds"])
    assert set(audit["selection"]) == set(VIEW_ORDER)
    assert all(
        row["selection_scope"] == "outer_training_inner_group_oof_only"
        for row in audit["selection"].values()
    )

    validation, _ = _features(73, 16)
    before = model.predict(validation)
    checkpoint = model.save(tmp_path / "brainfusion_stacking.joblib")
    restored = FoldLocalStackingClassifier.load(checkpoint)
    np.testing.assert_array_equal(restored.predict(validation), before)
    np.testing.assert_allclose(
        restored.decision_function(validation), model.decision_function(validation)
    )


def test_validation_labels_are_not_an_input_and_cannot_change_predictions() -> None:
    train, labels = _features(42, 48)
    validation, validation_labels = _features(99, 16)
    model = FoldLocalStackingClassifier(_config()).fit(
        train,
        labels,
        groups=[f"subject-{index // 4}" for index in range(48)],
        sample_ids=[f"train-{index}" for index in range(48)],
    )
    first = model.predict(validation)
    validation_labels[:] = 1 - validation_labels
    second = model.predict(validation)
    np.testing.assert_array_equal(first, second)


def test_stacking_refuses_duplicate_identities_and_group_leakage_shapes() -> None:
    train, labels = _features(11, 48)
    with pytest.raises(ValueError, match="identities must be unique"):
        FoldLocalStackingClassifier(_config()).fit(
            train,
            labels,
            groups=[f"subject-{index // 4}" for index in range(48)],
            sample_ids=["duplicate"] * 48,
        )
    broken = dict(train)
    broken["nvc"] = broken["nvc"][:-1]
    with pytest.raises(ValueError, match="equal N"):
        FoldLocalStackingClassifier(_config()).fit(
            broken,
            labels,
            groups=[f"subject-{index // 4}" for index in range(48)],
            sample_ids=[f"train-{index}" for index in range(48)],
        )


def test_stacking_refuses_protected_checkpoint_path(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="protected"):
        FoldLocalStackingClassifier.load(tmp_path / "protected" / "model.joblib")
