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


def straight_through_assignment_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    scale = max(float(temperature), 1e-3)
    soft_probs = F.softmax(logits / scale, dim=-1)
    hard_indices = torch.argmax(soft_probs, dim=-1)
    hard_probs = F.one_hot(hard_indices, num_classes=soft_probs.shape[-1]).to(dtype=soft_probs.dtype)
    return hard_probs + soft_probs - soft_probs.detach()


def orthogonality_loss(source_z: torch.Tensor, observation_z: torch.Tensor) -> torch.Tensor:
    source_flat = F.normalize(source_z.reshape(-1, source_z.shape[-1]), dim=-1)
    observation_flat = F.normalize(observation_z.reshape(-1, observation_z.shape[-1]), dim=-1)
    cross = source_flat.t() @ observation_flat / max(source_flat.shape[0], 1)
    return torch.mean(cross.pow(2))


def coupling_joint_probabilities(coupling_logits: torch.Tensor) -> torch.Tensor:
    if coupling_logits.ndim != 3:
        raise ValueError(
            'coupling_logits must have shape [n_lags, n_eeg_tokens, n_fnirs_tokens], '
            f'got {tuple(coupling_logits.shape)}'
        )
    n_lags, n_eeg_tokens, n_fnirs_tokens = coupling_logits.shape
    joint_logits = coupling_logits.permute(1, 0, 2).reshape(n_eeg_tokens, n_lags * n_fnirs_tokens)
    joint_probs = F.softmax(joint_logits, dim=-1)
    return joint_probs.reshape(n_eeg_tokens, n_lags, n_fnirs_tokens)


def coupling_lag_focus_loss(coupling_logits: torch.Tensor) -> torch.Tensor:
    joint_probs = coupling_joint_probabilities(coupling_logits)
    lag_probs = joint_probs.sum(dim=-1)
    if lag_probs.shape[-1] <= 1:
        return lag_probs.new_tensor(0.0)
    lag_probs = lag_probs.clamp_min(1e-8)
    entropy = -(lag_probs * lag_probs.log()).sum(dim=-1)
    max_entropy = math.log(float(lag_probs.shape[-1]))
    return entropy.mean() / max(max_entropy, 1e-8)


def _pairwise_js_divergence(anchor: torch.Tensor, neighbor: torch.Tensor) -> torch.Tensor:
    anchor = anchor.clamp_min(1e-8)
    neighbor = neighbor.clamp_min(1e-8)
    midpoint = 0.5 * (anchor + neighbor)
    anchor_kl = (anchor * (anchor.log() - midpoint.log())).sum(dim=-1)
    neighbor_kl = (neighbor * (neighbor.log() - midpoint.log())).sum(dim=-1)
    return 0.5 * (anchor_kl + neighbor_kl)


def coupling_eeg_neighbor_smoothness_loss(
    coupling_logits: torch.Tensor,
    eeg_codebook_weight: torch.Tensor,
    n_neighbors: int = 5,
) -> torch.Tensor:
    joint_probs = coupling_joint_probabilities(coupling_logits)
    n_eeg_tokens = joint_probs.shape[0]
    if n_eeg_tokens <= 1 or n_neighbors <= 0:
        return joint_probs.new_tensor(0.0)

    flat_joint = joint_probs.reshape(n_eeg_tokens, -1)
    normalized_codebook = F.normalize(eeg_codebook_weight.detach(), dim=-1)
    similarity = normalized_codebook @ normalized_codebook.t()
    neighbor_count = min(int(n_neighbors), n_eeg_tokens - 1)
    _, neighbor_indices = similarity.topk(neighbor_count + 1, dim=-1)
    neighbor_indices = neighbor_indices[:, 1:]

    anchor = flat_joint.unsqueeze(1)
    neighbor = flat_joint[neighbor_indices]
    return _pairwise_js_divergence(anchor, neighbor).mean()


__all__ = [
    'align_pair',
    'batch_usage_entropy_loss',
    'coupling_eeg_neighbor_smoothness_loss',
    'coupling_joint_probabilities',
    'coupling_lag_focus_loss',
    'orthogonality_loss',
    'straight_through_assignment_probs',
]