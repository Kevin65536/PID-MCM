"""Task metrics used for checkpoint selection and tuning.

The training loop deliberately computes these metrics without importing the
report generator.  This keeps checkpoint selection aligned with the scientific
endpoint instead of the composite optimization loss.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def classification_metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("classification confusion matrix must be square")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    accuracy = float(true_positive.sum() / total) if total else 0.0
    balanced_accuracy = float(recall[support > 0].mean()) if np.any(support > 0) else 0.0
    macro_f1 = float(f1[support > 0].mean()) if np.any(support > 0) else 0.0
    expected = float(np.dot(support, predicted) / (total * total)) if total else 0.0
    kappa = (accuracy - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "cohen_kappa": float(kappa),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "class_support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def classification_metrics(
    target: Sequence[int] | np.ndarray,
    predicted: Sequence[int] | np.ndarray,
    class_count: int,
) -> dict[str, Any]:
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    estimate = np.asarray(predicted, dtype=np.int64).reshape(-1)
    if truth.shape != estimate.shape:
        raise ValueError("classification target/predicted shapes differ")
    if class_count <= 1:
        raise ValueError("classification requires at least two classes")
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (truth, estimate), 1)
    return classification_metrics_from_confusion(confusion)


def selection_value(metrics: dict[str, Any], metric: str, mode: str) -> float:
    if metric not in metrics:
        raise KeyError(f"selection metric {metric!r} is absent from validation metrics")
    value = float(metrics[metric])
    if not math.isfinite(value):
        raise FloatingPointError(f"selection metric {metric!r} is non-finite: {value}")
    if mode not in {"min", "max"}:
        raise ValueError("selection mode must be 'min' or 'max'")
    return value


def improved(value: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    return value < best - min_delta if mode == "min" else value > best + min_delta
