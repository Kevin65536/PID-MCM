import numpy as np
import pytest

from src.metrics.lag_conditioned_downstream import (
    classification_metrics,
    confusion_matrix,
    evaluate_logit_ablations,
    subject_equal_classification_metrics,
)


def test_fixed_label_macro_f1_and_confusion_matrix():
    targets = np.asarray([0, 0, 1, 1])
    predictions = np.asarray([0, 1, 1, 1])
    matrix = confusion_matrix(targets, predictions, class_count=2)
    metrics = classification_metrics(
        targets, predictions, class_names=("left", "right")
    )

    np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    expected_f1 = ((2 * 1.0 * 0.5 / 1.5) + (2 * (2 / 3) * 1.0 / (5 / 3))) / 2
    assert metrics["macro_f1"] == pytest.approx(expected_f1)


def test_subject_equal_metrics_do_not_weight_larger_subject_more():
    targets = np.asarray([0, 1, 0, 1, 0, 1])
    predictions = np.asarray([0, 1, 1, 1, 1, 1])
    subjects = np.asarray(["s1", "s1", "s2", "s2", "s2", "s2"])
    metrics = subject_equal_classification_metrics(
        targets, predictions, subjects, class_names=("a", "b")
    )

    assert metrics["subject_count"] == 2
    subject_rows = {row["subject"]: row for row in metrics["subject_rows"]}
    assert subject_rows["s1"]["macro_f1"] == 1.0
    assert metrics["subject_equal_macro_f1"] == pytest.approx(
        np.mean([row["macro_f1"] for row in subject_rows.values()])
    )
    assert metrics["subject_equal_macro_f1"] != metrics["pooled"]["macro_f1"]


def test_logit_ablations_are_evaluated_independently():
    targets = np.asarray([0, 1, 0, 1])
    subjects = np.asarray(["s1", "s1", "s2", "s2"])
    perfect = np.asarray([[3, 0], [0, 3], [3, 0], [0, 3]], dtype=float)
    constant = np.asarray([[3, 0]] * 4, dtype=float)
    result = evaluate_logit_ablations(
        targets,
        subjects,
        {"combined": perfect, "private_only": constant},
        class_names=("a", "b"),
    )

    assert result["combined"]["subject_equal_macro_f1"] == 1.0
    assert result["private_only"]["subject_equal_macro_f1"] < 1.0


def test_invalid_ids_and_logit_shapes_fail_closed():
    with pytest.raises(ValueError, match="invalid class ID"):
        confusion_matrix([0, 2], [0, 1], class_count=2)
    with pytest.raises(ValueError, match="shape"):
        evaluate_logit_ablations(
            [0, 1], ["s", "s"], {"x": np.zeros((2, 3))}, class_names=("a", "b")
        )
