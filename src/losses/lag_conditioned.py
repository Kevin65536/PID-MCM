"""Mask-strict objective primitives for LC-SPVQ training."""

from __future__ import annotations

from typing import Mapping

import torch


def masked_mean_loss(
    elementwise_loss: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Average admitted elements and return a differentiable zero if empty."""

    mask = valid_mask.to(device=elementwise_loss.device, dtype=torch.bool)
    try:
        mask = torch.broadcast_to(mask, elementwise_loss.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} cannot broadcast to "
            f"loss shape {tuple(elementwise_loss.shape)}"
        ) from exc
    nonfinite_admitted = mask & ~torch.isfinite(elementwise_loss)
    if bool(nonfinite_admitted.any()):
        raise FloatingPointError("loss contains a non-finite admitted element")
    if not bool(mask.any()):
        return elementwise_loss.nan_to_num().sum() * 0.0
    return elementwise_loss[mask].mean()


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error with invalid/non-finite target entries removed."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    finite_target = torch.isfinite(target)
    finite_prediction = torch.isfinite(prediction)
    mask = valid_mask.to(device=prediction.device, dtype=torch.bool)
    try:
        mask = torch.broadcast_to(mask, prediction.shape)
    except RuntimeError as exc:
        raise ValueError("valid_mask cannot broadcast to prediction") from exc
    if bool((mask & finite_target & ~finite_prediction).any()):
        raise FloatingPointError("prediction contains a non-finite admitted element")
    admitted = mask & finite_target
    difference = torch.where(
        admitted,
        prediction - torch.where(finite_target, target, torch.zeros_like(target)),
        torch.zeros_like(prediction),
    )
    return masked_mean_loss(difference.square(), admitted)


def raw_patch_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    point_valid_mask: torch.Tensor,
    channel_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Channel/point masked raw reconstruction loss for ``[B,C,T]`` arrays."""

    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("raw reconstruction requires matching [B,C,T] tensors")
    if point_valid_mask.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError("point_valid_mask must have shape [B,T]")
    valid = point_valid_mask.to(device=prediction.device, dtype=torch.bool).unsqueeze(1)
    if channel_valid_mask is not None:
        if channel_valid_mask.shape != prediction.shape[:2]:
            raise ValueError("channel_valid_mask must have shape [B,C]")
        valid = valid & channel_valid_mask.to(
            device=prediction.device, dtype=torch.bool
        ).unsqueeze(-1)
    return masked_mse(prediction, target, valid)


def native_feature_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Coordinate-level native-feature MSE for ``[B,token,feature]``."""

    if prediction.ndim != 3:
        raise ValueError("native feature prediction must be [B,T,F]")
    if target.shape != prediction.shape or valid_mask.shape != prediction.shape:
        raise ValueError("native feature target/mask shapes differ from prediction")
    return masked_mse(prediction, target, valid_mask)


def weighted_pretraining_loss(
    losses: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Combine an explicit named objective without silently dropping a term."""

    if set(losses) != set(weights):
        raise ValueError(
            f"loss/weight names differ: {sorted(losses)} != {sorted(weights)}"
        )
    if not losses:
        raise ValueError("pretraining objective must contain at least one term")
    weighted = {}
    total = None
    for name, loss in losses.items():
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError(f"loss {name!r} must be a finite scalar")
        weight = float(weights[name])
        if weight < 0.0:
            raise ValueError(f"weight {name!r} must be non-negative")
        value = loss * weight
        weighted[name] = value
        total = value if total is None else total + value
    assert total is not None
    return total, weighted


__all__ = [
    "masked_mean_loss",
    "masked_mse",
    "native_feature_prediction_loss",
    "raw_patch_reconstruction_loss",
    "weighted_pretraining_loss",
]
