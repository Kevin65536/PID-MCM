"""Frozen-probe and prototype utilities for the E2 semantic-token suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class FrozenProbe:
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    model: Ridge

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_scale
        return np.asarray(self.model.predict(values), dtype=np.float64)


def r2_per_coordinate(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target/prediction must share shape [samples,coordinates]")
    residual = np.sum(np.square(target - prediction), axis=0)
    centered = np.sum(np.square(target - np.mean(target, axis=0, keepdims=True)), axis=0)
    return 1.0 - residual / np.maximum(centered, 1e-12)


def fit_grouped_ridge_probe(
    features: np.ndarray,
    target: np.ndarray,
    groups: Sequence[str],
    *,
    alphas: Iterable[float],
) -> tuple[FrozenProbe, dict[str, Any]]:
    """Select one ridge alpha using train-subject group CV, then refit."""

    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.str_)
    if features.ndim != 2 or target.ndim != 2 or len(features) != len(target):
        raise ValueError("probe features/targets must be aligned matrices")
    if len(groups) != len(features):
        raise ValueError("probe groups must align with features")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("grouped probe selection requires at least two subjects")
    alpha_values = tuple(float(value) for value in alphas)
    if not alpha_values or any(value < 0.0 for value in alpha_values):
        raise ValueError("ridge alphas must be non-empty and non-negative")

    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), 1e-8)
    standardized = (features - mean) / scale
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    scores: dict[float, list[float]] = {alpha: [] for alpha in alpha_values}
    for train_index, validation_index in splitter.split(standardized, target, groups):
        for alpha in alpha_values:
            model = Ridge(alpha=alpha).fit(standardized[train_index], target[train_index])
            prediction = model.predict(standardized[validation_index])
            scores[alpha].append(float(np.mean(r2_per_coordinate(target[validation_index], prediction))))
    mean_scores = {alpha: float(np.mean(values)) for alpha, values in scores.items()}
    selected = max(alpha_values, key=lambda alpha: (mean_scores[alpha], -alpha))
    model = Ridge(alpha=selected).fit(standardized, target)
    return (
        FrozenProbe(alpha=selected, feature_mean=mean, feature_scale=scale, model=model),
        {
            "selected_alpha": selected,
            "group_cv_mean_r2": {str(alpha): mean_scores[alpha] for alpha in alpha_values},
            "group_count": int(len(unique_groups)),
        },
    )


def subject_level_r2(
    target: np.ndarray,
    prediction: np.ndarray,
    subjects: Sequence[str],
) -> dict[str, list[float]]:
    subjects = np.asarray(subjects, dtype=np.str_)
    return {
        str(subject): r2_per_coordinate(target[subjects == subject], prediction[subjects == subject]).tolist()
        for subject in np.unique(subjects)
    }


def bootstrap_subject_mean(
    subject_values: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    matrix = np.asarray(list(subject_values.values()), dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("subject bootstrap requires a non-empty coordinate matrix")
    rng = np.random.default_rng(seed)
    samples = np.empty((iterations, matrix.shape[1]), dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(matrix), size=len(matrix))
        samples[index] = np.mean(matrix[selected], axis=0)
    return {
        "mean": np.mean(matrix, axis=0).tolist(),
        "ci95_low": np.quantile(samples, 0.025, axis=0).tolist(),
        "ci95_high": np.quantile(samples, 0.975, axis=0).tolist(),
    }


def prototype_signatures(
    hard_ids: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    codebook_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    hard_ids = np.asarray(hard_ids, dtype=np.int64)
    target = np.asarray(target, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if hard_ids.shape != valid_mask.shape or target.shape[:2] != hard_ids.shape:
        raise ValueError("prototype inputs must share [samples,tokens]")
    signatures = np.full((codebook_size, target.shape[-1]), np.nan, dtype=np.float64)
    counts = np.zeros(codebook_size, dtype=np.int64)
    for code in range(codebook_size):
        selected = valid_mask & (hard_ids == code)
        counts[code] = int(np.sum(selected))
        if counts[code]:
            signatures[code] = np.mean(target[selected], axis=0)
    return signatures, counts


def match_prototype_signatures(
    left: np.ndarray,
    right: np.ndarray,
    left_counts: np.ndarray,
    right_counts: np.ndarray,
) -> dict[str, Any]:
    """Hungarian-match active prototypes in standardized signature space."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    active_left = np.flatnonzero(np.asarray(left_counts) > 0)
    active_right = np.flatnonzero(np.asarray(right_counts) > 0)
    if not len(active_left) or not len(active_right):
        return {"matched_count": 0, "mean_cosine": None, "matches": []}
    left_values = left[active_left]
    right_values = right[active_right]
    pooled = np.concatenate((left_values, right_values), axis=0)
    mean = np.nanmean(pooled, axis=0)
    scale = np.maximum(np.nanstd(pooled, axis=0), 1e-8)
    left_values = np.nan_to_num((left_values - mean) / scale)
    right_values = np.nan_to_num((right_values - mean) / scale)
    left_norm = np.maximum(np.linalg.norm(left_values, axis=1, keepdims=True), 1e-12)
    right_norm = np.maximum(np.linalg.norm(right_values, axis=1, keepdims=True), 1e-12)
    cosine = (left_values / left_norm) @ (right_values / right_norm).T
    left_index, right_index = linear_sum_assignment(1.0 - cosine)
    matches = [
        {
            "left_code": int(active_left[i]),
            "right_code": int(active_right[j]),
            "cosine": float(cosine[i, j]),
        }
        for i, j in zip(left_index, right_index)
    ]
    return {
        "matched_count": len(matches),
        "mean_cosine": float(np.mean([row["cosine"] for row in matches])),
        "matches": matches,
    }


__all__ = [
    "FrozenProbe",
    "bootstrap_subject_mean",
    "fit_grouped_ridge_probe",
    "match_prototype_signatures",
    "prototype_signatures",
    "r2_per_coordinate",
    "subject_level_r2",
]
