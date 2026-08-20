"""Losses for uncertainty-aware SSM observation trajectory supervision."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def _broadcast_mask(mask: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    value = mask.to(dtype=torch.bool)
    while value.ndim < len(shape):
        value = value.unsqueeze(-1)
    try:
        return torch.broadcast_to(value, shape)
    except RuntimeError as exc:
        raise ValueError(f"valid_mask with shape {tuple(mask.shape)} cannot broadcast to {tuple(shape)}") from exc


def uncertainty_weighted_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    predictive_std: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    delta: float = 1.0,
    epsilon: float = 1e-6,
    weight_min: float = 0.05,
    weight_max: float = 20.0,
) -> torch.Tensor:
    """Return ``sum(w * Huber) / sum(w)`` on valid finite coordinates."""

    if prediction.shape != target.shape or prediction.shape != predictive_std.shape:
        raise ValueError("prediction, target, and predictive_std must share shape")
    if delta <= 0.0 or epsilon <= 0.0:
        raise ValueError("delta and epsilon must be positive")
    if weight_min <= 0.0 or weight_max < weight_min:
        raise ValueError("uncertainty weight bounds are invalid")
    mask = _broadcast_mask(valid_mask, prediction.shape)
    mask = (
        mask
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & torch.isfinite(predictive_std)
        & (predictive_std >= 0.0)
    )
    if not bool(mask.any()):
        raise ValueError("uncertainty-weighted loss has no valid coordinates")
    safe_prediction = torch.where(mask, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    safe_std = torch.where(mask, predictive_std, torch.ones_like(predictive_std))
    weights = torch.clamp(
        1.0 / (safe_std.square() + float(epsilon)),
        min=float(weight_min),
        max=float(weight_max),
    )
    point_loss = F.huber_loss(
        safe_prediction,
        safe_target,
        reduction="none",
        delta=float(delta),
    )
    admitted = mask.to(dtype=prediction.dtype)
    return (point_loss * weights * admitted).sum() / (
        weights * admitted
    ).sum().clamp_min(float(epsilon))


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss normalized by valid observation coordinates."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share shape")
    mask = _broadcast_mask(valid_mask, prediction.shape)
    mask = mask & torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(mask.any()):
        raise ValueError("masked Huber loss has no valid coordinates")
    safe_prediction = torch.where(mask, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    point_loss = F.huber_loss(
        safe_prediction,
        safe_target,
        reduction="none",
        delta=float(delta),
    )
    admitted = mask.to(dtype=prediction.dtype)
    return (point_loss * admitted).sum() / admitted.sum().clamp_min(1.0)


def ssm_observation_objective(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    clean_weight: float = 1.0,
    residual_weight: float = 1.0,
    cross_prediction_weight: float = 0.0,
    delta: float = 1.0,
    epsilon: float = 1e-6,
    weight_min: float = 0.05,
    weight_max: float = 20.0,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Compose EEG/fNIRS clean, residual, and optional EEG→fNIRS losses."""

    components: dict[str, torch.Tensor] = {}
    for modality in ("eeg", "fnirs"):
        components[f"{modality}_clean"] = uncertainty_weighted_huber_loss(
            outputs[f"{modality}_clean_prediction"],
            batch[f"{modality}_clean_target"],
            batch[f"{modality}_predictive_std"],
            batch[f"{modality}_target_valid_mask"],
            delta=delta,
            epsilon=epsilon,
            weight_min=weight_min,
            weight_max=weight_max,
        )
        components[f"{modality}_residual"] = masked_huber_loss(
            outputs[f"{modality}_residual_prediction"],
            batch[f"{modality}_residual_target"],
            batch[f"{modality}_target_valid_mask"],
            delta=delta,
        )
    total = float(clean_weight) * 0.5 * (
        components["eeg_clean"] + components["fnirs_clean"]
    ) + float(residual_weight) * 0.5 * (
        components["eeg_residual"] + components["fnirs_residual"]
    )
    if float(cross_prediction_weight) != 0.0:
        if "fnirs_cross_prediction" not in outputs:
            raise ValueError("cross prediction weight requires fnirs_cross_prediction")
        cross_mask = batch["fnirs_target_valid_mask"]
        if "fnirs_cross_prediction_valid_mask" in outputs:
            support = outputs["fnirs_cross_prediction_valid_mask"]
            while support.ndim < cross_mask.ndim:
                support = support.unsqueeze(-1)
            cross_mask = cross_mask.bool() & support.bool()
        components["eeg_to_fnirs"] = uncertainty_weighted_huber_loss(
            outputs["fnirs_cross_prediction"],
            batch["fnirs_clean_target"],
            batch["fnirs_predictive_std"],
            cross_mask,
            delta=delta,
            epsilon=epsilon,
            weight_min=weight_min,
            weight_max=weight_max,
        )
        total = total + float(cross_prediction_weight) * components["eeg_to_fnirs"]
    return total, components


__all__ = [
    "masked_huber_loss",
    "ssm_observation_objective",
    "uncertainty_weighted_huber_loss",
]
