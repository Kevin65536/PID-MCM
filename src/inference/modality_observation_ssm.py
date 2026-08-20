"""Modality-specific linear-Gaussian observation trajectory smoothers.

This module deliberately models an observation trajectory rather than claiming
that EEG and fNIRS share one identifiable latent coordinate.  A fit is learned
from training sequences only and can be applied either to one modality or to a
concatenated privileged (joint) observation vector.  Labels are not accepted by
any public fit API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .adaptive_neurovascular_ssm import rts_smoother


@dataclass(frozen=True)
class ObservationSSMFit:
    """Frozen train-only parameters of a vector observation SSM."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    transition: np.ndarray
    process_cov: np.ndarray
    observation_cov: np.ndarray
    initial_cov: np.ndarray
    ridge: float
    max_spectral_radius: float
    training_sequence_count: int
    training_transition_count: int
    fit_scope: str
    provenance_id: str

    def __post_init__(self) -> None:
        features = len(self.feature_names)
        if features <= 0 or len(set(self.feature_names)) != features:
            raise ValueError("feature_names must be non-empty and unique")
        if self.mean.shape != (features,):
            raise ValueError("mean must have one value per feature")
        for name, value in (
            ("transition", self.transition),
            ("process_cov", self.process_cov),
            ("observation_cov", self.observation_cov),
            ("initial_cov", self.initial_cov),
        ):
            if value.shape != (features, features):
                raise ValueError(f"{name} must be square on the feature axis")
            if np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if np.any(~np.isfinite(self.mean)):
            raise ValueError("mean must be finite")
        if np.any(np.diag(self.process_cov) <= 0.0):
            raise ValueError("process covariance must have positive diagonal")
        if np.any(np.diag(self.observation_cov) <= 0.0):
            raise ValueError("observation covariance must have positive diagonal")
        if self.training_sequence_count <= 0 or self.training_transition_count <= 0:
            raise ValueError("fit must record positive training support")


@dataclass(frozen=True)
class ObservationSmootherResult:
    """Posterior observation trajectory and uncertainty in observation units."""

    reconstructed: np.ndarray
    posterior_std: np.ndarray
    observation_predictive_std: np.ndarray
    residual: np.ndarray
    innovation_log_likelihood: float


@dataclass(frozen=True)
class JointObservationSmootherResult:
    """Named projections from one privileged joint observation smoother."""

    reconstructed: Mapping[str, np.ndarray]
    posterior_std: Mapping[str, np.ndarray]
    observation_predictive_std: Mapping[str, np.ndarray]
    residual: Mapping[str, np.ndarray]
    innovation_log_likelihood: float


def _sequence_and_mask(
    sequence: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    feature_count: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] <= 0:
        raise ValueError("each observation sequence must be [time,feature] with time >= 2")
    if feature_count is not None and values.shape[1] != feature_count:
        raise ValueError("observation feature count drifted across sequences")
    if valid_mask is None:
        mask = np.isfinite(values)
    else:
        raw_mask = np.asarray(valid_mask)
        if raw_mask.shape == (values.shape[0],):
            raw_mask = np.broadcast_to(raw_mask[:, None], values.shape)
        if raw_mask.shape != values.shape:
            raise ValueError("valid_mask must be [time] or [time,feature]")
        mask = raw_mask.astype(bool) & np.isfinite(values)
    return values, mask


def _stabilize_transition(transition: np.ndarray, maximum: float) -> np.ndarray:
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if not np.isfinite(radius):
        raise ValueError("transition estimate has non-finite spectral radius")
    if radius > float(maximum):
        transition = transition * (float(maximum) / max(radius, 1e-12))
    return np.asarray(transition, dtype=np.float64)


def fit_observation_ssm(
    sequences: Sequence[np.ndarray],
    *,
    feature_names: Sequence[str],
    valid_masks: Sequence[np.ndarray | None] | None = None,
    ridge: float = 1.0,
    max_spectral_radius: float = 0.995,
    process_variance_floor: float = 1e-4,
    observation_variance_floor: float = 1e-4,
    observation_noise_fraction: float = 0.25,
    fit_scope: str = "fit_parameter_all_conditions",
    provenance_id: str = "train_only",
) -> ObservationSSMFit:
    """Fit one stable vector SSM from pooled training-condition sequences.

    The function has no label argument by design.  Transition regression uses
    only consecutive rows for which every feature is valid.  Observation noise
    is a conservative fraction of short-scale second-difference variance; the
    remainder of one-step residual variance is assigned to process noise.
    """

    names = tuple(str(value) for value in feature_names)
    if not sequences:
        raise ValueError("at least one training sequence is required")
    if len(names) <= 0 or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if not 0.0 < float(max_spectral_radius) < 1.0:
        raise ValueError("max_spectral_radius must be in (0,1)")
    if not 0.0 <= float(observation_noise_fraction) <= 1.0:
        raise ValueError("observation_noise_fraction must be in [0,1]")
    masks = (
        [None] * len(sequences)
        if valid_masks is None
        else list(valid_masks)
    )
    if len(masks) != len(sequences):
        raise ValueError("valid_masks must match the training sequence count")
    parsed = [
        _sequence_and_mask(value, mask, feature_count=len(names))
        for value, mask in zip(sequences, masks, strict=True)
    ]
    feature_values = []
    for values, mask in parsed:
        for feature in range(len(names)):
            admitted = values[mask[:, feature], feature]
            if len(admitted):
                feature_values.append((feature, admitted))
    mean = np.zeros(len(names), dtype=np.float64)
    for feature in range(len(names)):
        admitted = [value for index, value in feature_values if index == feature]
        if not admitted:
            raise ValueError(f"feature {names[feature]!r} has no training support")
        mean[feature] = float(np.mean(np.concatenate(admitted)))

    previous_rows: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []
    centered_sequences: list[np.ndarray] = []
    for values, mask in parsed:
        centered = values - mean[None, :]
        centered_sequences.append(centered)
        complete = mask[:-1].all(axis=1) & mask[1:].all(axis=1)
        if complete.any():
            previous_rows.append(centered[:-1][complete])
            current_rows.append(centered[1:][complete])
    if not previous_rows:
        raise ValueError("training data contain no complete consecutive observations")
    previous = np.concatenate(previous_rows, axis=0)
    current = np.concatenate(current_rows, axis=0)
    gram = previous.T @ previous + float(ridge) * np.eye(len(names))
    transition = np.linalg.solve(gram, previous.T @ current).T
    transition = _stabilize_transition(transition, float(max_spectral_radius))
    one_step_residual = current - previous @ transition.T
    residual_var = np.maximum(np.var(one_step_residual, axis=0), process_variance_floor)

    short_scale: list[np.ndarray] = []
    for centered, (_, mask) in zip(centered_sequences, parsed, strict=True):
        if len(centered) < 3:
            continue
        complete = mask[:-2].all(axis=1) & mask[1:-1].all(axis=1) & mask[2:].all(axis=1)
        if complete.any():
            second = centered[2:] - 2.0 * centered[1:-1] + centered[:-2]
            short_scale.append(second[complete])
    if short_scale:
        second_var = np.var(np.concatenate(short_scale, axis=0), axis=0) / 6.0
    else:
        second_var = residual_var.copy()
    observation_var = np.maximum(
        float(observation_noise_fraction) * second_var,
        float(observation_variance_floor),
    )
    process_var = np.maximum(
        residual_var - np.minimum(observation_var, 0.9 * residual_var),
        float(process_variance_floor),
    )
    process_cov = np.diag(process_var)
    observation_cov = np.diag(observation_var)
    try:
        initial_cov = solve_discrete_lyapunov(transition, process_cov)
    except Exception:
        initial_cov = np.diag(process_var / np.maximum(1.0 - np.diag(transition) ** 2, 1e-3))
    initial_cov = np.asarray((initial_cov + initial_cov.T) * 0.5, dtype=np.float64)
    minimum = np.maximum(np.diag(initial_cov), process_var)
    initial_cov += np.diag(np.maximum(minimum - np.diag(initial_cov), 0.0) + 1e-8)
    return ObservationSSMFit(
        feature_names=names,
        mean=mean,
        transition=transition,
        process_cov=process_cov,
        observation_cov=observation_cov,
        initial_cov=initial_cov,
        ridge=float(ridge),
        max_spectral_radius=float(max_spectral_radius),
        training_sequence_count=len(sequences),
        training_transition_count=len(previous),
        fit_scope=str(fit_scope),
        provenance_id=str(provenance_id),
    )


def apply_observation_ssm(
    sequence: np.ndarray,
    fit: ObservationSSMFit,
    *,
    valid_mask: np.ndarray | None = None,
) -> ObservationSmootherResult:
    """Smooth one observation trajectory with a frozen train-only fit."""

    values, mask = _sequence_and_mask(
        sequence, valid_mask, feature_count=len(fit.feature_names)
    )
    centered = values - fit.mean[None, :]
    observations = np.where(mask, centered, np.nan)
    identity = np.eye(len(fit.feature_names), dtype=np.float64)
    states, state_std, state_cov, log_likelihood = rts_smoother(
        observations,
        fit.transition,
        fit.process_cov,
        identity,
        fit.observation_cov,
        fit.initial_cov,
    )
    reconstructed = states + fit.mean[None, :]
    predictive_var = np.diagonal(state_cov, axis1=1, axis2=2) + np.diag(
        fit.observation_cov
    )[None, :]
    predictive_std = np.sqrt(np.maximum(predictive_var, 0.0))
    residual = np.where(mask, values - reconstructed, 0.0)
    return ObservationSmootherResult(
        reconstructed=reconstructed,
        posterior_std=state_std,
        observation_predictive_std=predictive_std,
        residual=residual,
        innovation_log_likelihood=float(log_likelihood),
    )


def apply_observation_ssm_batch(
    sequences: np.ndarray,
    fit: ObservationSSMFit,
    *,
    valid_masks: np.ndarray | None = None,
) -> ObservationSmootherResult:
    """Vectorize smoothing across sequences with the same missingness pattern.

    Kalman/RTS covariance recursions depend on the validity pattern but not the
    observed values.  Grouping identical masks therefore preserves the exact
    single-sequence estimator while avoiding repeated matrix inversions.
    """

    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("sequences must have shape [sample,time,feature]")
    if values.shape[2] != len(fit.feature_names):
        raise ValueError("batch observation feature count differs from fit")
    if valid_masks is None:
        masks = np.isfinite(values)
    else:
        raw = np.asarray(valid_masks)
        if raw.shape == values.shape[:2]:
            raw = np.broadcast_to(raw[:, :, None], values.shape)
        if raw.shape != values.shape:
            raise ValueError("valid_masks must be [sample,time] or [sample,time,feature]")
        masks = raw.astype(bool) & np.isfinite(values)
    centered = values - fit.mean[None, None, :]
    sample_count, steps, features = centered.shape
    reconstructed = np.zeros_like(centered)
    posterior_std = np.zeros_like(centered)
    predictive_std = np.zeros_like(centered)
    total_log_likelihood = 0.0
    groups: dict[bytes, list[int]] = {}
    for index in range(sample_count):
        groups.setdefault(np.ascontiguousarray(masks[index]).tobytes(), []).append(index)
    identity = np.eye(features, dtype=np.float64)
    for indices in groups.values():
        selected = np.asarray(indices, dtype=np.int64)
        group_values = centered[selected]
        group_mask = masks[selected[0]]
        filtered_mean = np.zeros((len(selected), steps, features), dtype=np.float64)
        filtered_cov = np.zeros((steps, features, features), dtype=np.float64)
        predicted_mean = np.zeros_like(filtered_mean)
        predicted_cov = np.zeros_like(filtered_cov)
        mean = np.zeros((len(selected), features), dtype=np.float64)
        covariance = fit.initial_cov.copy()
        for time_index in range(steps):
            if time_index:
                mean = mean @ fit.transition.T
                covariance = (
                    fit.transition @ covariance @ fit.transition.T
                    + fit.process_cov
                )
            predicted_mean[:, time_index] = mean
            predicted_cov[time_index] = covariance
            available = group_mask[time_index]
            if np.any(available):
                design = identity[available]
                noise = fit.observation_cov[np.ix_(available, available)]
                innovation = group_values[:, time_index, available] - mean @ design.T
                innovation_cov = design @ covariance @ design.T + noise
                precision = np.linalg.pinv(innovation_cov)
                gain = covariance @ design.T @ precision
                mean = mean + innovation @ gain.T
                covariance = covariance - gain @ design @ covariance
                covariance = (covariance + covariance.T) * 0.5
                sign, logdet = np.linalg.slogdet(innovation_cov)
                if sign > 0:
                    quadratic = np.einsum(
                        "bi,ij,bj->b", innovation, precision, innovation
                    )
                    total_log_likelihood += float(
                        np.sum(
                            -0.5
                            * (
                                innovation.shape[1] * np.log(2.0 * np.pi)
                                + logdet
                                + quadratic
                            )
                        )
                    )
            filtered_mean[:, time_index] = mean
            filtered_cov[time_index] = covariance
        smoothed_mean = filtered_mean.copy()
        smoothed_cov = filtered_cov.copy()
        for time_index in range(steps - 2, -1, -1):
            smoothing_gain = (
                filtered_cov[time_index]
                @ fit.transition.T
                @ np.linalg.pinv(predicted_cov[time_index + 1])
            )
            smoothed_mean[:, time_index] += (
                smoothed_mean[:, time_index + 1]
                - predicted_mean[:, time_index + 1]
            ) @ smoothing_gain.T
            smoothed_cov[time_index] += (
                smoothing_gain
                @ (smoothed_cov[time_index + 1] - predicted_cov[time_index + 1])
                @ smoothing_gain.T
            )
            smoothed_cov[time_index] = (
                smoothed_cov[time_index] + smoothed_cov[time_index].T
            ) * 0.5
        reconstructed[selected] = smoothed_mean + fit.mean[None, None, :]
        state_std = np.sqrt(
            np.maximum(np.diagonal(smoothed_cov, axis1=1, axis2=2), 0.0)
        )
        observation_std = np.sqrt(
            np.maximum(
                np.diagonal(smoothed_cov, axis1=1, axis2=2)
                + np.diag(fit.observation_cov)[None, :],
                0.0,
            )
        )
        posterior_std[selected] = state_std[None, :, :]
        predictive_std[selected] = observation_std[None, :, :]
    residual = np.where(masks, values - reconstructed, 0.0)
    return ObservationSmootherResult(
        reconstructed=reconstructed,
        posterior_std=posterior_std,
        observation_predictive_std=predictive_std,
        residual=residual,
        innovation_log_likelihood=float(total_log_likelihood),
    )


def fit_joint_observation_ssm(
    modality_sequences: Mapping[str, Sequence[np.ndarray]],
    *,
    feature_names: Mapping[str, Sequence[str]],
    valid_masks: Mapping[str, Sequence[np.ndarray | None]] | None = None,
    **kwargs: object,
) -> tuple[ObservationSSMFit, Mapping[str, slice]]:
    """Fit a privileged joint SSM and return frozen modality projections."""

    modalities = tuple(sorted(str(value) for value in modality_sequences))
    if not modalities or set(modalities) != set(map(str, feature_names)):
        raise ValueError("modality sequences and feature names must share keys")
    sequence_count = {len(modality_sequences[name]) for name in modalities}
    if len(sequence_count) != 1:
        raise ValueError("joint modalities must have matching sequence counts")
    masks_by_modality = valid_masks or {}
    slices: dict[str, slice] = {}
    names: list[str] = []
    start = 0
    for modality in modalities:
        current_names = [f"{modality}:{value}" for value in feature_names[modality]]
        stop = start + len(current_names)
        slices[modality] = slice(start, stop)
        names.extend(current_names)
        start = stop
    sequences = []
    masks = []
    for index in range(next(iter(sequence_count))):
        parts = [np.asarray(modality_sequences[name][index]) for name in modalities]
        lengths = {part.shape[0] for part in parts}
        if len(lengths) != 1:
            raise ValueError("joint modalities must share the time axis")
        sequences.append(np.concatenate(parts, axis=1))
        mask_parts = []
        for name, part in zip(modalities, parts, strict=True):
            source_masks = masks_by_modality.get(name)
            source = None if source_masks is None else source_masks[index]
            _, parsed_mask = _sequence_and_mask(
                part, source, feature_count=part.shape[1]
            )
            mask_parts.append(parsed_mask)
        masks.append(np.concatenate(mask_parts, axis=1))
    fit = fit_observation_ssm(
        sequences,
        feature_names=names,
        valid_masks=masks,
        **kwargs,
    )
    return fit, slices


def apply_joint_observation_ssm(
    modality_sequence: Mapping[str, np.ndarray],
    fit: ObservationSSMFit,
    slices: Mapping[str, slice],
    *,
    valid_masks: Mapping[str, np.ndarray | None] | None = None,
) -> JointObservationSmootherResult:
    """Apply a privileged joint fit and project outputs to each modality."""

    modalities = tuple(sorted(slices))
    if set(modalities) != set(map(str, modality_sequence)):
        raise ValueError("joint application modality keys differ from fitted slices")
    parts = [np.asarray(modality_sequence[name]) for name in modalities]
    lengths = {part.shape[0] for part in parts}
    if len(lengths) != 1:
        raise ValueError("joint modalities must share the time axis")
    mask_parts = []
    for name, part in zip(modalities, parts, strict=True):
        source = None if valid_masks is None else valid_masks.get(name)
        _, mask = _sequence_and_mask(part, source, feature_count=part.shape[1])
        mask_parts.append(mask)
    result = apply_observation_ssm(
        np.concatenate(parts, axis=1),
        fit,
        valid_mask=np.concatenate(mask_parts, axis=1),
    )
    return JointObservationSmootherResult(
        reconstructed={name: result.reconstructed[:, slices[name]] for name in modalities},
        posterior_std={name: result.posterior_std[:, slices[name]] for name in modalities},
        observation_predictive_std={
            name: result.observation_predictive_std[:, slices[name]]
            for name in modalities
        },
        residual={name: result.residual[:, slices[name]] for name in modalities},
        innovation_log_likelihood=result.innovation_log_likelihood,
    )


__all__ = [
    "JointObservationSmootherResult",
    "ObservationSSMFit",
    "ObservationSmootherResult",
    "apply_joint_observation_ssm",
    "apply_observation_ssm",
    "apply_observation_ssm_batch",
    "fit_joint_observation_ssm",
    "fit_observation_ssm",
]
