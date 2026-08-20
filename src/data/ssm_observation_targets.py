"""Train-only modality observation trajectories for the SSM regression screen."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from src.inference.modality_observation_ssm import (
    ObservationSSMFit,
    apply_joint_observation_ssm,
    apply_observation_ssm,
    apply_observation_ssm_batch,
    fit_joint_observation_ssm,
    fit_observation_ssm,
)


EEG_BANDS_HZ: Mapping[str, tuple[float, float]] = {
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class ObservationChannelSelection:
    """Frozen fit-parameter target-channel indices shared by every condition."""

    eeg_indices: tuple[int, ...]
    fnirs_indices: tuple[int, ...]
    eeg_channel_names: tuple[str, ...]
    fnirs_channel_names: tuple[str, ...]
    fit_scope: str = "fit_parameter_all_conditions_only"

    def __post_init__(self) -> None:
        if not self.eeg_indices or not self.fnirs_indices:
            raise ValueError("observation channel selection cannot be empty")
        if len(self.eeg_indices) != len(self.eeg_channel_names):
            raise ValueError("EEG target indices/names differ")
        if len(self.fnirs_indices) != len(self.fnirs_channel_names):
            raise ValueError("fNIRS target indices/names differ")
        if len(set(self.eeg_indices)) != len(self.eeg_indices):
            raise ValueError("EEG target channel selection contains duplicates")
        if len(set(self.fnirs_indices)) != len(self.fnirs_indices):
            raise ValueError("fNIRS target channel selection contains duplicates")


@dataclass(frozen=True)
class ObservationFeatureBatch:
    values: np.ndarray
    valid_mask: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.valid_mask.shape != self.values.shape:
            raise ValueError("observation features must be [sample,token,feature]")
        if self.values.shape[-1] != len(self.feature_names):
            raise ValueError("feature names do not match observation coordinates")
        if np.any(~np.isfinite(self.values[self.valid_mask])):
            raise ValueError("valid observation feature coordinates must be finite")


@dataclass(frozen=True)
class ObservationStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    count: np.ndarray
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count.tolist(),
            "feature_names": list(self.feature_names),
            "fit_scope": "fit_parameter_all_conditions_only",
        }


@dataclass(frozen=True)
class SSMObservationTeacherFits:
    eeg_self: ObservationSSMFit
    fnirs_self: ObservationSSMFit
    joint: ObservationSSMFit
    joint_slices: Mapping[str, slice]
    eeg_standardizer: ObservationStandardizer
    fnirs_standardizer: ObservationStandardizer
    provenance_id: str
    labels_used: bool = False


@dataclass(frozen=True)
class SSMObservationTeacherBatch:
    eeg_observation: np.ndarray
    fnirs_observation: np.ndarray
    eeg_valid_mask: np.ndarray
    fnirs_valid_mask: np.ndarray
    self_clean_eeg: np.ndarray
    self_clean_fnirs: np.ndarray
    self_std_eeg: np.ndarray
    self_std_fnirs: np.ndarray
    joint_clean_eeg: np.ndarray
    joint_clean_fnirs: np.ndarray
    joint_std_eeg: np.ndarray
    joint_std_fnirs: np.ndarray
    provenance_id: str

    def targets(self, mode: str) -> Mapping[str, np.ndarray]:
        name = str(mode).upper()
        if name == "NATIVE":
            clean_eeg = self.eeg_observation
            clean_fnirs = self.fnirs_observation
            std_eeg = np.ones_like(clean_eeg)
            std_fnirs = np.ones_like(clean_fnirs)
        elif name in {"SSM-SELF", "SSM-SELF-XPRED"}:
            clean_eeg = self.self_clean_eeg
            clean_fnirs = self.self_clean_fnirs
            std_eeg = self.self_std_eeg
            std_fnirs = self.self_std_fnirs
        elif name == "SSM-JOINT":
            clean_eeg = self.joint_clean_eeg
            clean_fnirs = self.joint_clean_fnirs
            std_eeg = self.joint_std_eeg
            std_fnirs = self.joint_std_fnirs
        else:
            raise ValueError(f"unknown observation teacher mode: {mode}")
        return {
            "eeg_clean_target": clean_eeg,
            "fnirs_clean_target": clean_fnirs,
            "eeg_residual_target": self.eeg_observation - clean_eeg,
            "fnirs_residual_target": self.fnirs_observation - clean_fnirs,
            "eeg_predictive_std": std_eeg,
            "fnirs_predictive_std": std_fnirs,
            "eeg_target_valid_mask": self.eeg_valid_mask,
            "fnirs_target_valid_mask": self.fnirs_valid_mask,
        }


def fit_observation_channel_selection(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    *,
    eeg_channel_valid_mask: np.ndarray,
    fnirs_channel_valid_mask: np.ndarray,
    eeg_channel_names: Sequence[str],
    fnirs_channel_names: Sequence[str],
    fnirs_component_roles: Sequence[str],
    eeg_channel_count: int = 6,
) -> ObservationChannelSelection:
    """Select spatially explicit EEG and one HbO/HbR pair on fit data only."""

    eeg_values = np.asarray(eeg, dtype=np.float64)
    fnirs_values = np.asarray(fnirs, dtype=np.float64)
    eeg_mask = np.asarray(eeg_channel_valid_mask, dtype=bool)
    fnirs_mask = np.asarray(fnirs_channel_valid_mask, dtype=bool)
    if eeg_values.ndim != 3 or eeg_mask.shape != eeg_values.shape[:2]:
        raise ValueError("EEG channel selection input shape mismatch")
    if fnirs_values.ndim != 3 or fnirs_mask.shape != fnirs_values.shape[:2]:
        raise ValueError("fNIRS channel selection input shape mismatch")
    if eeg_values.shape[1] != len(eeg_channel_names):
        raise ValueError("EEG names do not match channel axis")
    if fnirs_values.shape[1] != len(fnirs_channel_names) or len(fnirs_channel_names) != len(fnirs_component_roles):
        raise ValueError("fNIRS names/roles do not match channel axis")
    count = min(int(eeg_channel_count), eeg_values.shape[1])
    if count <= 0:
        raise ValueError("eeg_channel_count must be positive")

    def canonical(name: str) -> str:
        return "".join(value for value in str(name).upper() if value.isalnum())

    # Preserve canonical lateralized sensorimotor/frontal anchors when present;
    # fill remaining slots by train-only support and variance without labels.
    priority = (
        "C3", "C4", "CZ", "CP3", "CP4", "FC3", "FC4", "F3", "F4", "FZ",
        "P3", "P4", "PZ", "O1", "O2",
    )
    canonical_names = [canonical(value) for value in eeg_channel_names]
    selected_eeg: list[int] = []
    for wanted in priority:
        match = next(
            (
                index
                for index, name in enumerate(canonical_names)
                if name == wanted or name.endswith(wanted)
            ),
            None,
        )
        if match is not None and match not in selected_eeg:
            selected_eeg.append(match)
        if len(selected_eeg) == count:
            break
    support = eeg_mask.sum(axis=0).astype(np.float64)
    variance = np.var(eeg_values, axis=(0, 2))
    ranking = sorted(
        range(eeg_values.shape[1]),
        key=lambda index: (support[index], variance[index], -index),
        reverse=True,
    )
    selected_eeg.extend(
        index for index in ranking if index not in selected_eeg
    )
    selected_eeg = selected_eeg[:count]

    roles = tuple(str(value).upper() for value in fnirs_component_roles)
    hbo = [index for index, role in enumerate(roles) if role == "HBO"]
    hbr = [index for index, role in enumerate(roles) if role == "HBR"]
    if not hbo or not hbr:
        raise ValueError("fNIRS target selection requires HbO and HbR channels")
    fnirs_support = fnirs_mask.sum(axis=0).astype(np.float64)
    fnirs_variance = np.var(fnirs_values, axis=(0, 2))
    hbo_index = max(
        hbo,
        key=lambda index: (fnirs_support[index], fnirs_variance[index], -index),
    )
    def chromophore_stem(name: str) -> str:
        return canonical(name).replace("HBO", "").replace("HBR", "")

    hbo_stem = chromophore_stem(str(fnirs_channel_names[hbo_index]))
    matched_hbr = [
        index
        for index in hbr
        if chromophore_stem(str(fnirs_channel_names[index])) == hbo_stem
    ]
    hbr_index = (
        max(
            matched_hbr,
            key=lambda index: (fnirs_support[index], fnirs_variance[index], -index),
        )
        if matched_hbr
        else min(hbr, key=lambda index: abs(index - hbo_index))
    )
    return ObservationChannelSelection(
        eeg_indices=tuple(map(int, selected_eeg)),
        fnirs_indices=(int(hbo_index), int(hbr_index)),
        eeg_channel_names=tuple(str(eeg_channel_names[index]) for index in selected_eeg),
        fnirs_channel_names=(
            str(fnirs_channel_names[hbo_index]),
            str(fnirs_channel_names[hbr_index]),
        ),
    )


def extract_eeg_spatial_band_trajectory(
    eeg: np.ndarray,
    *,
    token_valid_mask: np.ndarray,
    point_valid_mask: np.ndarray,
    channel_valid_mask: np.ndarray,
    channel_names: Sequence[str],
    sampling_rate_hz: float = 200.0,
    bands_hz: Mapping[str, tuple[float, float]] = EEG_BANDS_HZ,
    epsilon: float = 1e-8,
) -> ObservationFeatureBatch:
    """Extract per-channel alpha/beta/gamma log-power without spatial averaging."""

    values = np.asarray(eeg, dtype=np.float64)
    token_mask = np.asarray(token_valid_mask, dtype=bool)
    point_mask = np.asarray(point_valid_mask, dtype=bool)
    channel_mask = np.asarray(channel_valid_mask, dtype=bool)
    if values.ndim != 3 or point_mask.shape != (values.shape[0], values.shape[2]):
        raise ValueError("EEG values/mask shape mismatch")
    if channel_mask.shape != values.shape[:2]:
        raise ValueError("EEG channel mask shape mismatch")
    if values.shape[1] != len(channel_names):
        raise ValueError("EEG channel names do not match values")
    if token_mask.ndim != 2 or token_mask.shape[0] != values.shape[0]:
        raise ValueError("EEG token mask shape mismatch")
    tokens = token_mask.shape[1]
    if values.shape[2] % tokens:
        raise ValueError("EEG samples cannot be split into token patches")
    patch = values.shape[2] // tokens
    patches = values.reshape(values.shape[0], values.shape[1], tokens, patch)
    point_patches = point_mask.reshape(values.shape[0], tokens, patch)
    frequencies = np.fft.rfftfreq(patch, d=1.0 / float(sampling_rate_hz))
    spectrum = np.abs(np.fft.rfft(patches, axis=-1)) ** 2 / max(patch, 1)
    features = []
    masks = []
    names = []
    for band_name, (low, high) in bands_hz.items():
        selected = (frequencies >= float(low)) & (frequencies < float(high))
        if not selected.any():
            raise ValueError(f"EEG patch has no bins in {band_name}")
        band_power = np.log(np.maximum(spectrum[..., selected].mean(axis=-1), epsilon))
        # [B,C,N] -> [B,N,C]
        features.append(np.transpose(band_power, (0, 2, 1)))
        valid = (
            token_mask[:, :, None]
            & point_patches.all(axis=-1)[:, :, None]
            & channel_mask[:, None, :]
        )
        masks.append(valid)
        names.extend(f"{channel}:{band_name}" for channel in channel_names)
    # Preserve channel-major, then band-major order for interpretable prototypes.
    stacked = np.stack(features, axis=-1)  # [B,N,C,Band]
    stacked_mask = np.stack(masks, axis=-1)
    output = stacked.reshape(values.shape[0], tokens, -1)
    output_mask = stacked_mask.reshape(values.shape[0], tokens, -1)
    ordered_names = tuple(
        f"{channel}:{band_name}"
        for channel in channel_names
        for band_name in bands_hz
    )
    return ObservationFeatureBatch(
        values=output.astype(np.float32),
        valid_mask=output_mask,
        feature_names=ordered_names,
    )


def extract_fnirs_patch_trajectory(
    fnirs: np.ndarray,
    *,
    token_valid_mask: np.ndarray,
    point_valid_mask: np.ndarray,
    channel_valid_mask: np.ndarray,
    channel_names: Sequence[str],
) -> ObservationFeatureBatch:
    """Keep every HbO/HbR sample inside each token patch."""

    values = np.asarray(fnirs, dtype=np.float64)
    token_mask = np.asarray(token_valid_mask, dtype=bool)
    point_mask = np.asarray(point_valid_mask, dtype=bool)
    channel_mask = np.asarray(channel_valid_mask, dtype=bool)
    if values.ndim != 3 or values.shape[1] != len(channel_names):
        raise ValueError("fNIRS values/channel names mismatch")
    if point_mask.shape != (values.shape[0], values.shape[2]):
        raise ValueError("fNIRS point mask mismatch")
    if channel_mask.shape != values.shape[:2]:
        raise ValueError("fNIRS channel mask mismatch")
    tokens = token_mask.shape[1]
    if values.shape[2] % tokens:
        raise ValueError("fNIRS samples cannot be split into token patches")
    patch = values.shape[2] // tokens
    patches = values.reshape(values.shape[0], values.shape[1], tokens, patch)
    patches = np.transpose(patches, (0, 2, 1, 3)).reshape(values.shape[0], tokens, -1)
    point_patches = point_mask.reshape(values.shape[0], tokens, patch)
    mask = (
        token_mask[:, :, None, None]
        & channel_mask[:, None, :, None]
        & point_patches[:, :, None, :]
    ).reshape(values.shape[0], tokens, -1)
    names = tuple(
        f"{channel}:sample_{index:02d}"
        for channel in channel_names
        for index in range(patch)
    )
    return ObservationFeatureBatch(
        values=patches.astype(np.float32), valid_mask=mask, feature_names=names
    )


def fit_observation_standardizer(
    batch: ObservationFeatureBatch,
) -> ObservationStandardizer:
    values = batch.values.astype(np.float64)
    mask = batch.valid_mask
    count = mask.sum(axis=(0, 1))
    if np.any(count <= 0):
        raise ValueError("observation standardizer has unsupported features")
    mean = np.where(mask, values, 0.0).sum(axis=(0, 1)) / count
    variance = np.where(mask, (values - mean[None, None, :]) ** 2, 0.0).sum(
        axis=(0, 1)
    ) / count
    scale = np.sqrt(np.maximum(variance, 1e-8))
    return ObservationStandardizer(
        mean=mean,
        scale=scale,
        count=count,
        feature_names=batch.feature_names,
    )


def apply_observation_standardizer(
    batch: ObservationFeatureBatch,
    standardizer: ObservationStandardizer,
) -> ObservationFeatureBatch:
    if batch.feature_names != standardizer.feature_names:
        raise ValueError("observation feature names drifted from fit parameter")
    values = (batch.values - standardizer.mean[None, None, :]) / standardizer.scale[
        None, None, :
    ]
    values = np.where(batch.valid_mask, values, 0.0)
    return ObservationFeatureBatch(
        values=values.astype(np.float32),
        valid_mask=batch.valid_mask.copy(),
        feature_names=batch.feature_names,
    )


def _provenance_payload(
    eeg: ObservationFeatureBatch,
    fnirs: ObservationFeatureBatch,
    *,
    fit_scope: str,
) -> str:
    payload = {
        "schema": "ssm_modality_observation_teacher_v1",
        "eeg_features": list(eeg.feature_names),
        "fnirs_features": list(fnirs.feature_names),
        "fit_scope": str(fit_scope),
        "sequence_count": int(len(eeg.values)),
        "labels_used": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def fit_ssm_observation_teachers(
    eeg: ObservationFeatureBatch,
    fnirs: ObservationFeatureBatch,
    *,
    ridge: float = 1.0,
    max_spectral_radius: float = 0.995,
    fit_scope: str = "fit_parameter_all_subjects_all_conditions",
) -> SSMObservationTeacherFits:
    """Fit self and privileged joint teachers without accepting task labels."""

    if len(eeg.values) != len(fnirs.values) or eeg.values.shape[1] != fnirs.values.shape[1]:
        raise ValueError("EEG/fNIRS teacher sequences must be sample/time aligned")
    eeg_standardizer = fit_observation_standardizer(eeg)
    fnirs_standardizer = fit_observation_standardizer(fnirs)
    eeg_z = apply_observation_standardizer(eeg, eeg_standardizer)
    fnirs_z = apply_observation_standardizer(fnirs, fnirs_standardizer)
    provenance = _provenance_payload(eeg, fnirs, fit_scope=fit_scope)
    common = dict(
        ridge=float(ridge),
        max_spectral_radius=float(max_spectral_radius),
        fit_scope=str(fit_scope),
        provenance_id=provenance,
    )
    eeg_self = fit_observation_ssm(
        list(eeg_z.values),
        feature_names=eeg_z.feature_names,
        valid_masks=list(eeg_z.valid_mask),
        **common,
    )
    fnirs_self = fit_observation_ssm(
        list(fnirs_z.values),
        feature_names=fnirs_z.feature_names,
        valid_masks=list(fnirs_z.valid_mask),
        **common,
    )
    joint, slices = fit_joint_observation_ssm(
        {"eeg": list(eeg_z.values), "fnirs": list(fnirs_z.values)},
        feature_names={"eeg": eeg_z.feature_names, "fnirs": fnirs_z.feature_names},
        valid_masks={
            "eeg": list(eeg_z.valid_mask),
            "fnirs": list(fnirs_z.valid_mask),
        },
        **{**common, "fit_scope": f"privileged_{fit_scope}"},
    )
    return SSMObservationTeacherFits(
        eeg_self=eeg_self,
        fnirs_self=fnirs_self,
        joint=joint,
        joint_slices=slices,
        eeg_standardizer=eeg_standardizer,
        fnirs_standardizer=fnirs_standardizer,
        provenance_id=provenance,
        labels_used=False,
    )


def apply_ssm_observation_teachers(
    eeg: ObservationFeatureBatch,
    fnirs: ObservationFeatureBatch,
    fits: SSMObservationTeacherFits,
) -> SSMObservationTeacherBatch:
    """Apply frozen self and joint teachers to one experiment partition."""

    if fits.labels_used:
        raise PermissionError("SSM teacher fit records task-label use")
    eeg_z = apply_observation_standardizer(eeg, fits.eeg_standardizer)
    fnirs_z = apply_observation_standardizer(fnirs, fits.fnirs_standardizer)
    eeg_result = apply_observation_ssm_batch(
        eeg_z.values, fits.eeg_self, valid_masks=eeg_z.valid_mask
    )
    fnirs_result = apply_observation_ssm_batch(
        fnirs_z.values, fits.fnirs_self, valid_masks=fnirs_z.valid_mask
    )
    modalities = tuple(sorted(fits.joint_slices))
    joint_values = np.concatenate(
        [eeg_z.values if name == "eeg" else fnirs_z.values for name in modalities],
        axis=2,
    )
    joint_masks = np.concatenate(
        [eeg_z.valid_mask if name == "eeg" else fnirs_z.valid_mask for name in modalities],
        axis=2,
    )
    joint_result = apply_observation_ssm_batch(
        joint_values, fits.joint, valid_masks=joint_masks
    )
    eeg_slice = fits.joint_slices["eeg"]
    fnirs_slice = fits.joint_slices["fnirs"]
    return SSMObservationTeacherBatch(
        eeg_observation=eeg_z.values,
        fnirs_observation=fnirs_z.values,
        eeg_valid_mask=eeg_z.valid_mask,
        fnirs_valid_mask=fnirs_z.valid_mask,
        self_clean_eeg=np.asarray(eeg_result.reconstructed, dtype=np.float32),
        self_clean_fnirs=np.asarray(fnirs_result.reconstructed, dtype=np.float32),
        self_std_eeg=np.asarray(
            eeg_result.observation_predictive_std, dtype=np.float32
        ),
        self_std_fnirs=np.asarray(
            fnirs_result.observation_predictive_std, dtype=np.float32
        ),
        joint_clean_eeg=np.asarray(
            joint_result.reconstructed[:, :, eeg_slice], dtype=np.float32
        ),
        joint_clean_fnirs=np.asarray(
            joint_result.reconstructed[:, :, fnirs_slice], dtype=np.float32
        ),
        joint_std_eeg=np.asarray(
            joint_result.observation_predictive_std[:, :, eeg_slice],
            dtype=np.float32,
        ),
        joint_std_fnirs=np.asarray(
            joint_result.observation_predictive_std[:, :, fnirs_slice],
            dtype=np.float32,
        ),
        provenance_id=fits.provenance_id,
    )


__all__ = [
    "EEG_BANDS_HZ",
    "ObservationChannelSelection",
    "ObservationFeatureBatch",
    "ObservationStandardizer",
    "SSMObservationTeacherBatch",
    "SSMObservationTeacherFits",
    "apply_observation_standardizer",
    "apply_ssm_observation_teachers",
    "extract_eeg_spatial_band_trajectory",
    "extract_fnirs_patch_trajectory",
    "fit_observation_channel_selection",
    "fit_observation_standardizer",
    "fit_ssm_observation_teachers",
]
