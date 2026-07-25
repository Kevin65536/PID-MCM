"""Metrics for public-development EFRM downstream evaluation."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata


def classification_metrics(
    target: Sequence[int] | np.ndarray,
    logits: np.ndarray,
    class_names: Sequence[str],
    *,
    calibration_bins: int = 15,
) -> dict[str, Any]:
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    raw = np.asarray(logits, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != truth.size:
        raise ValueError("classification logits/target shapes differ")
    if raw.shape[1] != len(class_names) or len(class_names) < 2:
        raise ValueError("classification output dimension does not match class names")
    shifted = raw - raw.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    predicted = probability.argmax(axis=1)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    total = max(1, int(confusion.sum()))
    accuracy = float(true_positive.sum() / total)
    present = support > 0
    balanced_accuracy = float(recall[present].mean()) if present.any() else 0.0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    expected = float(np.dot(support, predicted_support) / (total * total))
    kappa = (accuracy - expected) / (1.0 - expected) if expected < 1.0 else 0.0

    confidence = probability.max(axis=1)
    correct = predicted == truth
    boundaries = np.linspace(0.0, 1.0, calibration_bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )
    true_probability = probability[np.arange(truth.size), truth]
    nll = float(-np.log(true_probability.clip(1e-12, 1.0)).mean())
    one_hot = np.eye(len(class_names), dtype=np.float64)[truth]
    brier = float(np.square(probability - one_hot).sum(axis=1).mean())
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "cohen_kappa": float(kappa),
        "negative_log_likelihood": nll,
        "expected_calibration_error": float(ece),
        "brier_score": brier,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "class_support": support.tolist(),
        "confusion_matrix": confusion.tolist(),
        "class_names": list(class_names),
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    return 2.0 * covariance / denominator if denominator > 1e-12 else float("nan")


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    target_names: Sequence[str],
) -> dict[str, Any]:
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if truth.shape != estimate.shape or truth.shape != valid.shape or truth.ndim != 3:
        raise ValueError("regression target/prediction/mask shapes differ")
    if truth.shape[1] != len(target_names):
        raise ValueError("regression coordinate count does not match target names")
    valid &= np.isfinite(truth) & np.isfinite(estimate)
    coordinate_metrics: dict[str, Any] = {}
    flattened_truth: list[np.ndarray] = []
    flattened_prediction: list[np.ndarray] = []
    for coordinate, name in enumerate(target_names):
        mask = valid[:, coordinate]
        x = truth[:, coordinate][mask]
        y = estimate[:, coordinate][mask]
        if not x.size:
            raise RuntimeError(f"regression coordinate {name} has no valid support")
        error = y - x
        residual = float(np.square(error).sum())
        total = float(np.square(x - x.mean()).sum())
        coordinate_metrics[str(name)] = {
            "ccc": _ccc(x, y),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": float(1.0 - residual / total) if total > 1e-12 else float("nan"),
            "pearson": _correlation(x, y),
            "spearman": _correlation(rankdata(x), rankdata(y)),
            "valid_count": int(x.size),
            "coverage": float(mask.mean()),
        }
        flattened_truth.append(x)
        flattened_prediction.append(y)
    x = np.concatenate(flattened_truth)
    y = np.concatenate(flattened_prediction)
    error = y - x
    finite_ccc = [
        float(row["ccc"])
        for row in coordinate_metrics.values()
        if math.isfinite(float(row["ccc"]))
    ]
    return {
        "ccc": float(np.mean(finite_ccc)) if finite_ccc else float("nan"),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "coverage": float(valid.mean()),
        "valid_count": int(valid.sum()),
        "coordinates": coordinate_metrics,
    }


def subject_metrics(
    *,
    subjects: Sequence[str],
    task_type: str,
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    names: Sequence[str],
) -> list[dict[str, Any]]:
    subject_array = np.asarray(subjects, dtype=object)
    rows: list[dict[str, Any]] = []
    for subject in sorted(set(str(value) for value in subject_array)):
        keep = subject_array == subject
        if task_type == "classification":
            metrics = classification_metrics(target[keep], prediction[keep], names)
        else:
            metrics = regression_metrics(target[keep], prediction[keep], valid_mask[keep], names)
        rows.append({"subject": subject, "sample_count": int(keep.sum()), "metrics": metrics})
    return rows
