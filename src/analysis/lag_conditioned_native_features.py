"""Native, interpretable patch targets for LC-SPVQ shared branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.analysis.physiological_patch_features import (
    PatchFeatureBatch,
    extract_physiological_patch_features,
)


EEG_NATIVE_FEATURES = (
    "log_absolute_power_theta",
    "log_absolute_power_alpha",
    "log_absolute_power_beta",
    "log_absolute_power_low_gamma",
    "spectral_entropy",
    "line_length",
    "hjorth_mobility",
)
FNIRS_NATIVE_FEATURES = ("mean", "slope", "endpoint_delta", "auc")
FNIRS_COMPONENT_ROLES = ("HbO", "HbR")


@dataclass(frozen=True)
class NativeFeatureTargets:
    """Token-aligned feature values and coordinate-level support."""

    values: np.ndarray
    valid_mask: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.valid_mask)
        if values.ndim != 3 or mask.shape != values.shape:
            raise ValueError("native targets require matching [batch, token, feature] arrays")
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("native target names do not match feature dimension")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise TypeError("native target valid_mask must be boolean")
        if np.any(mask & ~np.isfinite(values)):
            raise ValueError("supported native target entries must be finite")


@dataclass(frozen=True)
class MaskedStandardizer:
    """Train-only coordinate-wise standardization statistics."""

    mean: np.ndarray
    scale: np.ndarray
    count: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean)
        scale = np.asarray(self.scale)
        count = np.asarray(self.count)
        if mean.ndim != 1 or scale.shape != mean.shape or count.shape != mean.shape:
            raise ValueError("standardizer statistics must be matching vectors")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)):
            raise ValueError("standardizer statistics must be finite")
        if np.any(scale <= 0.0) or np.any(count <= 0):
            raise ValueError("standardizer requires positive scales and support counts")

    def to_dict(self) -> dict[str, list[float] | list[int] | str]:
        return {
            "schema": "masked_native_feature_standardizer_v1",
            "mean": np.asarray(self.mean, dtype=float).tolist(),
            "scale": np.asarray(self.scale, dtype=float).tolist(),
            "count": np.asarray(self.count, dtype=int).tolist(),
            "fit_scope": "fit_parameter_subjects_only",
        }


def _validate_signal_and_mask(
    signal: np.ndarray,
    token_valid_mask: np.ndarray,
    *,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float32)
    mask = np.asarray(token_valid_mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("signal must have shape [batch, channel, sample]")
    if values.shape[-1] % int(patch_size):
        raise ValueError("signal length is not divisible by patch size")
    tokens = values.shape[-1] // int(patch_size)
    if mask.shape != (values.shape[0], tokens):
        raise ValueError(f"token mask must have shape {(values.shape[0], tokens)}")
    return values, mask


def _validate_channel_valid_mask(
    channel_valid_mask: np.ndarray | None,
    *,
    batch_size: int,
    channel_count: int,
) -> np.ndarray:
    expected_shape = (int(batch_size), int(channel_count))
    if channel_valid_mask is None:
        return np.ones(expected_shape, dtype=bool)
    mask = np.asarray(channel_valid_mask, dtype=bool)
    if mask.shape != expected_shape:
        raise ValueError(f"channel_valid_mask must have shape {expected_shape}")
    return mask


def _aggregate_feature_coordinates(
    batch: PatchFeatureBatch,
    selected_features: Sequence[str],
    channel_groups: Sequence[tuple[str, np.ndarray]],
    channel_valid_mask: np.ndarray,
) -> NativeFeatureTargets:
    lookup = {name: index for index, name in enumerate(batch.feature_names)}
    missing = [name for name in selected_features if name not in lookup]
    if missing:
        raise KeyError(f"feature extractor lacks requested coordinates: {missing}")
    outputs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    names: list[str] = []
    for group_name, channel_indices in channel_groups:
        indices = np.asarray(channel_indices, dtype=int)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"channel group {group_name!r} is empty")
        if np.any(indices < 0) or np.any(indices >= batch.values.shape[2]):
            raise IndexError(f"channel group {group_name!r} is out of range")
        for feature in selected_features:
            feature_index = lookup[feature]
            values = batch.values[:, :, indices, feature_index].astype(np.float64)
            valid = batch.feature_valid_mask[:, :, indices, feature_index]
            valid = valid & channel_valid_mask[:, None, indices]
            count = valid.sum(axis=2)
            total = np.where(valid, values, 0.0).sum(axis=2)
            mean = np.divide(
                total,
                count,
                out=np.zeros(total.shape, dtype=np.float64),
                where=count > 0,
            )
            supported = count > 0
            outputs.append(mean.astype(np.float32))
            masks.append(supported)
            names.append(feature if not group_name else f"{group_name}/{feature}")
    stacked = np.stack(outputs, axis=-1)
    valid_mask = np.stack(masks, axis=-1)
    return NativeFeatureTargets(
        values=np.where(valid_mask, stacked, 0.0).astype(np.float32),
        valid_mask=valid_mask,
        feature_names=tuple(names),
    )


def extract_eeg_native_targets(
    eeg: np.ndarray,
    token_valid_mask: np.ndarray,
    *,
    channel_valid_mask: np.ndarray | None = None,
    channel_names: Sequence[str] | None = None,
    sample_rate_hz: float = 200.0,
    patch_size: int = 400,
) -> NativeFeatureTargets:
    """Extract seven channel-mean EEG morphology/spectral coordinates.

    ``channel_valid_mask`` identifies channels that may contribute to each
    sample's channel aggregation. It is optional for backwards compatibility,
    but when supplied it must have shape ``[batch, channel]``.
    """

    values, mask = _validate_signal_and_mask(
        eeg, token_valid_mask, patch_size=int(patch_size)
    )
    channel_mask = _validate_channel_valid_mask(
        channel_valid_mask,
        batch_size=values.shape[0],
        channel_count=values.shape[1],
    )
    batch = extract_physiological_patch_features(
        values,
        modality="eeg",
        sample_rate_hz=float(sample_rate_hz),
        patch_size=int(patch_size),
        valid_mask=mask,
        channel_names=channel_names,
    )
    channels = np.arange(values.shape[1], dtype=int)
    return _aggregate_feature_coordinates(
        batch,
        EEG_NATIVE_FEATURES,
        (("", channels),),
        channel_mask,
    )


def extract_fnirs_native_targets(
    fnirs: np.ndarray,
    token_valid_mask: np.ndarray,
    *,
    component_roles: Sequence[str],
    channel_valid_mask: np.ndarray | None = None,
    channel_names: Sequence[str] | None = None,
    sample_rate_hz: float = 10.0,
    patch_size: int = 20,
) -> NativeFeatureTargets:
    """Extract component-resolved fNIRS mean/slope/delta/AUC coordinates.

    ``channel_valid_mask`` identifies channels that may contribute to each
    component aggregation. It is optional for backwards compatibility, but
    when supplied it must have shape ``[batch, channel]``.
    """

    values, mask = _validate_signal_and_mask(
        fnirs, token_valid_mask, patch_size=int(patch_size)
    )
    channel_mask = _validate_channel_valid_mask(
        channel_valid_mask,
        batch_size=values.shape[0],
        channel_count=values.shape[1],
    )
    roles = tuple(str(value) for value in component_roles)
    if len(roles) != values.shape[1]:
        raise ValueError("fNIRS component roles do not match channel count")
    batch = extract_physiological_patch_features(
        values,
        modality="fnirs",
        sample_rate_hz=float(sample_rate_hz),
        patch_size=int(patch_size),
        valid_mask=mask,
        channel_names=channel_names,
    )
    groups = []
    for role in FNIRS_COMPONENT_ROLES:
        indices = np.flatnonzero(np.asarray(roles, dtype=str) == role)
        if indices.size == 0:
            raise ValueError(f"fNIRS input lacks required component role {role}")
        groups.append((role, indices))
    return _aggregate_feature_coordinates(
        batch,
        FNIRS_NATIVE_FEATURES,
        groups,
        channel_mask,
    )


def fit_masked_standardizer(targets: NativeFeatureTargets) -> MaskedStandardizer:
    """Fit one standardizer from an explicitly supplied training partition."""

    values = np.asarray(targets.values, dtype=np.float64)
    mask = np.asarray(targets.valid_mask, dtype=bool)
    count = mask.sum(axis=(0, 1)).astype(np.int64)
    if np.any(count <= 0):
        missing = np.flatnonzero(count <= 0).tolist()
        raise ValueError(f"native target coordinates lack fit support: {missing}")
    total = np.where(mask, values, 0.0).sum(axis=(0, 1))
    mean = total / count
    square = np.where(mask, np.square(values), 0.0).sum(axis=(0, 1))
    variance = np.maximum(square / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale >= 1e-6, scale, 1.0)
    return MaskedStandardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        count=count,
    )


def apply_masked_standardizer(
    targets: NativeFeatureTargets,
    standardizer: MaskedStandardizer,
) -> NativeFeatureTargets:
    """Apply fixed train-only statistics and zero unsupported coordinates."""

    if targets.values.shape[-1] != len(standardizer.mean):
        raise ValueError("standardizer dimension does not match native targets")
    transformed = (
        np.asarray(targets.values, dtype=np.float32)
        - np.asarray(standardizer.mean, dtype=np.float32)[None, None, :]
    ) / np.asarray(standardizer.scale, dtype=np.float32)[None, None, :]
    transformed = np.where(targets.valid_mask, transformed, 0.0).astype(np.float32)
    return NativeFeatureTargets(
        values=transformed,
        valid_mask=np.asarray(targets.valid_mask, dtype=bool).copy(),
        feature_names=targets.feature_names,
    )


__all__ = [
    "EEG_NATIVE_FEATURES",
    "FNIRS_COMPONENT_ROLES",
    "FNIRS_NATIVE_FEATURES",
    "MaskedStandardizer",
    "NativeFeatureTargets",
    "apply_masked_standardizer",
    "extract_eeg_native_targets",
    "extract_fnirs_native_targets",
    "fit_masked_standardizer",
]
