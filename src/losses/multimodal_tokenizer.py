"""Reusable multimodal tokenizer losses for EEG/fNIRS models."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def align_pair(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    lag: int,
    target_length: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if lag < 0:
        raise ValueError('Only non-negative lag is supported')
    usable = min(tensor_a.shape[1], tensor_b.shape[1] - lag)
    if target_length is not None:
        usable = min(usable, int(target_length))
    if usable <= 0:
        return tensor_a[:, :0], tensor_b[:, :0]
    return tensor_a[:, :usable], tensor_b[:, lag:lag + usable]


def symmetric_kl_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    scale = max(float(temperature), 1e-3)
    log_probs_a = F.log_softmax(logits_a / scale, dim=-1)
    log_probs_b = F.log_softmax(logits_b / scale, dim=-1)
    probs_a = log_probs_a.exp()
    probs_b = log_probs_b.exp()
    kl_ab = F.kl_div(log_probs_a, probs_b, reduction='batchmean')
    kl_ba = F.kl_div(log_probs_b, probs_a, reduction='batchmean')
    return 0.5 * (kl_ab + kl_ba)


def coupling_kl_loss(pred_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    pred_probs = pred_probs.clamp_min(1e-8)
    pred_probs = pred_probs / pred_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target_probs = target_probs.clamp_min(1e-8)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(pred_probs.log(), target_probs, reduction='batchmean')


def batch_usage_entropy_loss(probs: torch.Tensor) -> torch.Tensor:
    if probs.numel() == 0:
        return probs.new_tensor(0.0)
    marginal = probs.reshape(-1, probs.shape[-1]).mean(dim=0)
    marginal = marginal.clamp_min(1e-8)
    marginal = marginal / marginal.sum().clamp_min(1e-8)
    entropy = -(marginal * marginal.log()).sum()
    max_entropy = math.log(float(marginal.shape[0])) if marginal.shape[0] > 1 else 1.0
    normalized_entropy = entropy / max(max_entropy, 1e-8)
    return 1.0 - normalized_entropy


def orthogonality_loss(source_z: torch.Tensor, observation_z: torch.Tensor) -> torch.Tensor:
    source_flat = F.normalize(source_z.reshape(-1, source_z.shape[-1]), dim=-1)
    observation_flat = F.normalize(observation_z.reshape(-1, observation_z.shape[-1]), dim=-1)
    cross = source_flat.t() @ observation_flat / max(source_flat.shape[0], 1)
    return torch.mean(cross.pow(2))


__all__ = [
    'align_pair',
    'batch_usage_entropy_loss',
    'coupling_kl_loss',
    'orthogonality_loss',
    'symmetric_kl_from_logits',
]