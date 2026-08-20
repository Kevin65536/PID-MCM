"""Subject-aware downstream metrics for LC-SPVQ classifier ablations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _validate_targets_predictions(
    targets: Any,
    predictions: Any,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(class_count, (bool, np.bool_)) or int(class_count) <= 1:
        raise ValueError("class_count must be an integer greater than one")
    class_count = int(class_count)
    target = np.asarray(targets)
    predicted = np.asarray(predictions)
    if target.ndim != 1 or predicted.shape != target.shape:
        raise ValueError("targets and predictions must be matching vectors")
    if len(target) == 0:
        raise ValueError("classification metrics require at least one sample")
    for name, values in (("targets", target), ("predictions", predicted)):
        try:
            numeric = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integer class IDs") from exc
        if (
            np.any(~np.isfinite(numeric))
            or np.any(numeric != np.floor(numeric))
            or np.any(numeric < 0)
            or np.any(numeric >= class_count)
        ):
            raise ValueError(f"{name} contains an invalid class ID")
    return target.astype(np.int64), predicted.astype(np.int64)


def confusion_matrix(
    targets: Any,
    predictions: Any,
    *,
    class_count: int,
) -> np.ndarray:
    """Return a fixed-class-order integer confusion matrix."""

    target, predicted = _validate_targets_predictions(
        targets, predictions, class_count=class_count
    )
    matrix = np.zeros((int(class_count), int(class_count)), dtype=np.int64)
    np.add.at(matrix, (target, predicted), 1)
    return matrix


def classification_metrics(
    targets: Any,
    predictions: Any,
    *,
    class_names: Sequence[str],
) -> Mapping[str, Any]:
    """Compute accuracy, balanced accuracy and fixed-label macro-F1."""

    names = tuple(str(value) for value in class_names)
    if len(names) <= 1 or len(names) != len(set(names)):
        raise ValueError("class_names must contain at least two unique labels")
    target, predicted = _validate_targets_predictions(
        targets, predictions, class_count=len(names)
    )
    matrix = confusion_matrix(target, predicted, class_count=len(names))
    true_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros(len(names), dtype=np.float64),
        where=true_count > 0,
    )
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(len(names), dtype=np.float64),
        where=predicted_count > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(names), dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "sample_count": int(len(target)),
        "class_names": names,
        "confusion_matrix": matrix,
        "accuracy": float(np.mean(target == predicted)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": true_count,
    }


def subject_equal_classification_metrics(
    targets: Any,
    predictions: Any,
    subjects: Sequence[str],
    *,
    class_names: Sequence[str],
) -> Mapping[str, Any]:
    """Average complete fixed-label metrics over subjects with equal weight."""

    target = np.asarray(targets)
    predicted = np.asarray(predictions)
    subject = np.asarray(subjects, dtype=str)
    if subject.shape != target.shape or predicted.shape != target.shape:
        raise ValueError("targets, predictions, and subjects must be matching vectors")
    identities = tuple(sorted(set(subject.tolist())))
    if not identities:
        raise ValueError("subject metrics require at least one subject")
    rows = []
    for identity in identities:
        selected = subject == identity
        metrics = classification_metrics(
            target[selected], predicted[selected], class_names=class_names
        )
        rows.append(
            {
                "subject": identity,
                "sample_count": metrics["sample_count"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return {
        "subject_count": len(rows),
        "subject_rows": rows,
        "subject_equal_accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "subject_equal_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in rows])
        ),
        "subject_equal_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
        "pooled": classification_metrics(
            target, predicted, class_names=class_names
        ),
    }


def evaluate_logit_ablations(
    targets: Any,
    subjects: Sequence[str],
    logits: Mapping[str, Any],
    *,
    class_names: Sequence[str],
) -> Mapping[str, Mapping[str, Any]]:
    """Evaluate independently exported classifier-logit contributions."""

    target = np.asarray(targets)
    if not logits:
        raise ValueError("at least one logit ablation is required")
    output = {}
    for name, values in logits.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(target), len(class_names)):
            raise ValueError(
                f"logits[{name!r}] must have shape {(len(target), len(class_names))}"
            )
        if np.any(~np.isfinite(array)):
            raise ValueError(f"logits[{name!r}] must be finite")
        output[str(name)] = subject_equal_classification_metrics(
            target,
            np.argmax(array, axis=1),
            subjects,
            class_names=class_names,
        )
    return output


__all__ = [
    "classification_metrics",
    "confusion_matrix",
    "evaluate_logit_ablations",
    "subject_equal_classification_metrics",
]
