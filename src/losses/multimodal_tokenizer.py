"""Reusable multimodal tokenizer losses for EEG/fNIRS models."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

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


def smooth_signal(signal: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if signal.dim() != 3:
        raise ValueError('Expected signal tensor with shape [B, C, T]')
    if signal.shape[-1] <= 1:
        return signal
    kernel = max(min(int(kernel_size), signal.shape[-1]), 1)
    if kernel <= 1:
        return signal
    pad_left = (kernel - 1) // 2
    pad_right = kernel - 1 - pad_left
    padded = F.pad(signal, (pad_left, pad_right), mode='replicate')
    return F.avg_pool1d(padded, kernel_size=kernel, stride=1)


def symmetric_prob_kl(probs_a: torch.Tensor, probs_b: torch.Tensor) -> torch.Tensor:
    probs_a = probs_a.clamp_min(1e-8)
    probs_a = probs_a / probs_a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    probs_b = probs_b.clamp_min(1e-8)
    probs_b = probs_b / probs_b.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    kl_ab = F.kl_div(probs_a.log(), probs_b, reduction='batchmean')
    kl_ba = F.kl_div(probs_b.log(), probs_a, reduction='batchmean')
    return 0.5 * (kl_ab + kl_ba)


def symmetric_hard_assignment_ce(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    targets_a = logits_a.detach().argmax(dim=-1)
    targets_b = logits_b.detach().argmax(dim=-1)
    scale = max(float(temperature), 1e-3)
    ce_ab = F.cross_entropy(
        (logits_a / scale).reshape(-1, logits_a.shape[-1]),
        targets_b.reshape(-1),
    )
    ce_ba = F.cross_entropy(
        (logits_b / scale).reshape(-1, logits_b.shape[-1]),
        targets_a.reshape(-1),
    )
    return 0.5 * (ce_ab + ce_ba)


def orthogonality_loss(shared_z: torch.Tensor, private_z: torch.Tensor) -> torch.Tensor:
    shared_flat = F.normalize(shared_z.reshape(-1, shared_z.shape[-1]), dim=-1)
    private_flat = F.normalize(private_z.reshape(-1, private_z.shape[-1]), dim=-1)
    cross = shared_flat.t() @ private_flat / max(shared_flat.shape[0], 1)
    return torch.mean(cross.pow(2))


def compute_shared_alignment_losses(
    z_eeg: torch.Tensor,
    z_fnirs: torch.Tensor,
    eeg_logits: torch.Tensor,
    fnirs_logits: torch.Tensor,
    *,
    alignment_loss,
    alignment_lag_candidates: Sequence[int],
    alignment_selection: str,
    alignment_compare_mode: str,
    fixed_alignment_compare_length: int | None,
    latent_alignment_weight: float,
    assignment_alignment_weight: float,
    assignment_temperature: float,
) -> Dict[str, torch.Tensor]:
    combined_losses = []
    latent_losses = []
    assignment_losses = []
    valid_lags = []
    usable_lengths = []
    target_length = fixed_alignment_compare_length if alignment_compare_mode == 'fixed_min' else None

    for lag in alignment_lag_candidates:
        aligned_z_eeg, aligned_z_fnirs = align_pair(z_eeg, z_fnirs, lag, target_length=target_length)
        aligned_eeg_logits, aligned_fnirs_logits = align_pair(
            eeg_logits,
            fnirs_logits,
            lag,
            target_length=target_length,
        )
        if aligned_z_eeg.shape[1] == 0:
            continue

        latent_loss = alignment_loss(aligned_z_eeg, aligned_z_fnirs)
        assignment_loss = symmetric_kl_from_logits(
            aligned_eeg_logits,
            aligned_fnirs_logits,
            temperature=assignment_temperature,
        )
        combined_loss = latent_alignment_weight * latent_loss + assignment_alignment_weight * assignment_loss

        combined_losses.append(combined_loss)
        latent_losses.append(latent_loss)
        assignment_losses.append(assignment_loss)
        valid_lags.append(lag)
        usable_lengths.append(aligned_z_eeg.shape[1])

    if not combined_losses:
        zero = z_eeg.new_tensor(0.0)
        return {
            'latent_align_loss': zero,
            'assignment_align_loss': zero,
            'selected_lag': zero,
            'alignment_usable_tokens': zero,
        }

    if alignment_selection == 'mean':
        latent_align_loss = torch.stack(latent_losses).mean()
        assignment_align_loss = torch.stack(assignment_losses).mean()
        selected_lag = float(sum(valid_lags) / len(valid_lags))
        alignment_usable_tokens = float(sum(usable_lengths) / len(usable_lengths))
    else:
        best_index = int(torch.argmin(torch.stack(combined_losses)).item())
        latent_align_loss = latent_losses[best_index]
        assignment_align_loss = assignment_losses[best_index]
        selected_lag = float(valid_lags[best_index])
        alignment_usable_tokens = float(usable_lengths[best_index])

    return {
        'latent_align_loss': latent_align_loss,
        'assignment_align_loss': assignment_align_loss,
        'selected_lag': z_eeg.new_tensor(selected_lag),
        'alignment_usable_tokens': z_eeg.new_tensor(alignment_usable_tokens),
    }


def compute_factorized_shared_alignment_losses(
    z_eeg_shared: torch.Tensor,
    z_fnirs_shared: torch.Tensor,
    eeg_shared_logits: torch.Tensor,
    fnirs_shared_logits: torch.Tensor,
    *,
    alignment_loss,
    coupling_logits: torch.Tensor,
    alignment_lag_candidates: Sequence[int],
    alignment_selection: str,
    alignment_compare_mode: str,
    fixed_alignment_compare_length: int | None,
    assignment_temperature: float,
    latent_alignment_weight: float,
    coupling_weight: float,
    assignment_alignment_weight: float,
    hard_assignment_alignment_weight: float,
    coupling_bidirectional: bool,
) -> Dict[str, torch.Tensor]:
    latent_losses = []
    coupling_losses = []
    assignment_losses = []
    hard_assignment_losses = []
    combined_losses = []
    valid_lags = []
    usable_lengths = []
    target_length = fixed_alignment_compare_length if alignment_compare_mode == 'fixed_min' else None
    scale = max(float(assignment_temperature), 1e-3)
    eeg_probs = F.softmax(eeg_shared_logits / scale, dim=-1)
    fnirs_probs = F.softmax(fnirs_shared_logits / scale, dim=-1)
    shared_entropy_loss = 0.5 * (
        batch_usage_entropy_loss(eeg_probs) +
        batch_usage_entropy_loss(fnirs_probs)
    )

    for lag_index, lag in enumerate(alignment_lag_candidates):
        aligned_z_eeg, aligned_z_fnirs = align_pair(
            z_eeg_shared,
            z_fnirs_shared,
            lag,
            target_length=target_length,
        )
        aligned_eeg_probs, aligned_fnirs_probs = align_pair(
            eeg_probs,
            fnirs_probs,
            lag,
            target_length=target_length,
        )
        aligned_eeg_logits, aligned_fnirs_logits = align_pair(
            eeg_shared_logits,
            fnirs_shared_logits,
            lag,
            target_length=target_length,
        )
        if aligned_z_eeg.shape[1] == 0:
            continue

        latent_loss = alignment_loss(aligned_z_eeg, aligned_z_fnirs)
        transition = F.softmax(coupling_logits[lag_index], dim=-1)
        pred_fnirs_probs = torch.einsum('bnk,kl->bnl', aligned_eeg_probs, transition)
        coupling_loss = coupling_kl_loss(pred_fnirs_probs, aligned_fnirs_probs)
        if coupling_bidirectional:
            reverse_transition = F.softmax(coupling_logits[lag_index].transpose(0, 1), dim=-1)
            pred_eeg_probs = torch.einsum('bnk,kl->bnl', aligned_fnirs_probs, reverse_transition)
            coupling_loss = 0.5 * (
                coupling_loss + coupling_kl_loss(pred_eeg_probs, aligned_eeg_probs)
            )
        assignment_loss = symmetric_prob_kl(aligned_eeg_probs, aligned_fnirs_probs)
        hard_assignment_loss = symmetric_hard_assignment_ce(
            aligned_eeg_logits,
            aligned_fnirs_logits,
            temperature=assignment_temperature,
        )
        combined = (
            latent_alignment_weight * latent_loss +
            coupling_weight * coupling_loss +
            assignment_alignment_weight * assignment_loss +
            hard_assignment_alignment_weight * hard_assignment_loss
        )
        latent_losses.append(latent_loss)
        coupling_losses.append(coupling_loss)
        assignment_losses.append(assignment_loss)
        hard_assignment_losses.append(hard_assignment_loss)
        combined_losses.append(combined)
        valid_lags.append(lag)
        usable_lengths.append(aligned_z_eeg.shape[1])

    if not combined_losses:
        zero = z_eeg_shared.new_tensor(0.0)
        return {
            'latent_align_loss': zero,
            'coupling_loss': zero,
            'assignment_align_loss': zero,
            'hard_assignment_align_loss': zero,
            'shared_entropy_loss': zero,
            'selected_lag': zero,
            'alignment_usable_tokens': zero,
        }

    if alignment_selection == 'mean':
        latent_align_loss = torch.stack(latent_losses).mean()
        coupling_loss = torch.stack(coupling_losses).mean()
        assignment_align_loss = torch.stack(assignment_losses).mean()
        hard_assignment_align_loss = torch.stack(hard_assignment_losses).mean()
        selected_lag = float(sum(valid_lags) / len(valid_lags))
        alignment_usable_tokens = float(sum(usable_lengths) / len(usable_lengths))
    else:
        best_index = int(torch.argmin(torch.stack(combined_losses)).item())
        latent_align_loss = latent_losses[best_index]
        coupling_loss = coupling_losses[best_index]
        assignment_align_loss = assignment_losses[best_index]
        hard_assignment_align_loss = hard_assignment_losses[best_index]
        selected_lag = float(valid_lags[best_index])
        alignment_usable_tokens = float(usable_lengths[best_index])

    return {
        'latent_align_loss': latent_align_loss,
        'coupling_loss': coupling_loss,
        'assignment_align_loss': assignment_align_loss,
        'hard_assignment_align_loss': hard_assignment_align_loss,
        'shared_entropy_loss': shared_entropy_loss,
        'selected_lag': z_eeg_shared.new_tensor(selected_lag),
        'alignment_usable_tokens': z_eeg_shared.new_tensor(alignment_usable_tokens),
    }


__all__ = [
    'align_pair',
    'batch_usage_entropy_loss',
    'compute_factorized_shared_alignment_losses',
    'compute_shared_alignment_losses',
    'coupling_kl_loss',
    'orthogonality_loss',
    'smooth_signal',
    'symmetric_hard_assignment_ce',
    'symmetric_kl_from_logits',
    'symmetric_prob_kl',
]