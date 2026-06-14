"""Reusable multimodal tokenizer losses for EEG/fNIRS models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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


def _conditional_mapping_log_probs(
    coupling_slice: torch.Tensor,
    eeg_slice: torch.Tensor,
    fnirs_slice: torch.Tensor,
    *,
    residualize_fnirs_marginal: bool,
) -> torch.Tensor:
    if not residualize_fnirs_marginal:
        return F.log_softmax(coupling_slice, dim=-1)

    eeg_marginal = eeg_slice.mean(dim=(0, 1)).detach().clamp_min(1e-8)
    eeg_marginal = eeg_marginal / eeg_marginal.sum().clamp_min(1e-8)
    fnirs_marginal = fnirs_slice.mean(dim=(0, 1)).detach().clamp_min(1e-8)
    fnirs_marginal = fnirs_marginal / fnirs_marginal.sum().clamp_min(1e-8)

    # Remove EEG-independent column bias under the observed EEG occupancy. The
    # fixed fNIRS marginal then explains common token frequency, leaving the
    # trainable tensor to represent only EEG-conditioned deviations.
    column_bias = (eeg_marginal.unsqueeze(-1) * coupling_slice).sum(dim=0, keepdim=True)
    residual_logits = coupling_slice - column_bias
    return F.log_softmax(residual_logits + fnirs_marginal.log().unsqueeze(0), dim=-1)


def coupling_pair_likelihood_loss(
    coupling_logits: torch.Tensor,
    eeg_assignment_logits: torch.Tensor,
    fnirs_assignment_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    detach_tokens: bool = True,
    residualize_fnirs_marginal: bool = False,
) -> torch.Tensor:
    """Lag-balanced conditional NLL of observed EEG/fNIRS source-token pairs.

    Each lag slice parameterizes ``P(fNIRS_token | EEG_token, lag)``.  Every
    valid lag contributes one independently averaged loss term, so lag zero
    cannot dominate merely because it contains more in-window pairs.

    ``detach_tokens=True`` trains only the coupling tensor; ``False`` also
    sends a differentiable signal into the source token assignment logits.
    """
    if coupling_logits.ndim != 3:
        raise ValueError(
            'coupling_logits must have shape [n_lags, n_eeg_tokens, n_fnirs_tokens], '
            f'got {tuple(coupling_logits.shape)}'
        )
    if eeg_assignment_logits.ndim != 3 or fnirs_assignment_logits.ndim != 3:
        raise ValueError('assignment logits must have shape [batch, tokens, codebook]')
    if eeg_assignment_logits.shape[:2] != fnirs_assignment_logits.shape[:2]:
        raise ValueError(
            'EEG and fNIRS source assignment logits must share batch/token dimensions '
            f'(got {tuple(eeg_assignment_logits.shape[:2])} and {tuple(fnirs_assignment_logits.shape[:2])})'
        )

    n_lags, n_eeg_tokens, n_fnirs_tokens = coupling_logits.shape
    if eeg_assignment_logits.shape[-1] != n_eeg_tokens:
        raise ValueError(
            f'EEG assignment codebook size {eeg_assignment_logits.shape[-1]} does not match coupling tensor {n_eeg_tokens}'
        )
    if fnirs_assignment_logits.shape[-1] != n_fnirs_tokens:
        raise ValueError(
            f'fNIRS assignment codebook size {fnirs_assignment_logits.shape[-1]} does not match coupling tensor {n_fnirs_tokens}'
        )

    token_count = eeg_assignment_logits.shape[1]
    if token_count <= 0 or n_lags <= 0:
        return coupling_logits.new_tensor(0.0)

    scale = max(float(temperature), 1e-3)
    eeg_probs = straight_through_assignment_probs(eeg_assignment_logits, scale)
    fnirs_probs = straight_through_assignment_probs(fnirs_assignment_logits, scale)
    if detach_tokens:
        eeg_probs = eeg_probs.detach()
        fnirs_probs = fnirs_probs.detach()

    max_lag = min(int(n_lags), int(token_count))
    per_lag_losses = []

    for lag_index in range(max_lag):
        valid_count = token_count - lag_index
        if valid_count <= 0:
            continue
        eeg_slice = eeg_probs[:, :valid_count, :]
        fnirs_slice = fnirs_probs[:, lag_index:lag_index + valid_count, :]
        lag_mapping_log_probs = _conditional_mapping_log_probs(
            coupling_logits[lag_index],
            eeg_slice,
            fnirs_slice,
            residualize_fnirs_marginal=residualize_fnirs_marginal,
        )
        expected_log_probability = torch.einsum(
            'bti,if,btf->bt',
            eeg_slice,
            lag_mapping_log_probs,
            fnirs_slice,
        )
        per_lag_losses.append(-expected_log_probability.mean())

    if not per_lag_losses:
        return coupling_logits.new_tensor(0.0)
    return torch.stack(per_lag_losses).mean()


def coupling_lag_evidence_loss(
    coupling_logits: torch.Tensor,
    eeg_assignment_logits: torch.Tensor,
    fnirs_assignment_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    detach_tokens: bool = True,
    evidence_temperature: float = 0.25,
    residualize_fnirs_marginal: bool = False,
) -> torch.Tensor:
    """Anchor EEG-specific lag mass to conditional gain over fNIRS marginals.

    Mapping evidence is estimated independently for every lag and EEG token.
    The target lag distribution is proportional to the conditional log-
    likelihood gain over the lag-specific fNIRS marginal.  The evidence target
    is detached so this term updates lag mass rather than manufacturing gain by
    changing the mapping distribution itself.
    """
    if coupling_logits.ndim != 3:
        raise ValueError('coupling_logits must have shape [n_lags, n_eeg_tokens, n_fnirs_tokens]')
    if eeg_assignment_logits.ndim != 3 or fnirs_assignment_logits.ndim != 3:
        raise ValueError('assignment logits must have shape [batch, tokens, codebook]')
    if eeg_assignment_logits.shape[:2] != fnirs_assignment_logits.shape[:2]:
        raise ValueError('EEG and fNIRS assignment logits must share batch/token dimensions')

    n_lags, n_eeg_tokens, n_fnirs_tokens = coupling_logits.shape
    if eeg_assignment_logits.shape[-1] != n_eeg_tokens:
        raise ValueError('EEG assignment codebook size does not match coupling tensor')
    if fnirs_assignment_logits.shape[-1] != n_fnirs_tokens:
        raise ValueError('fNIRS assignment codebook size does not match coupling tensor')

    scale = max(float(temperature), 1e-3)
    eeg_probs = straight_through_assignment_probs(eeg_assignment_logits, scale)
    fnirs_probs = straight_through_assignment_probs(fnirs_assignment_logits, scale)
    if detach_tokens:
        eeg_probs = eeg_probs.detach()
        fnirs_probs = fnirs_probs.detach()

    joint_probs = coupling_joint_probabilities(coupling_logits)
    lag_probs = joint_probs.sum(dim=-1).clamp_min(1e-8)
    token_count = eeg_probs.shape[1]
    max_lag = min(int(n_lags), int(token_count))
    gain_columns = []

    for lag_index in range(max_lag):
        valid_count = token_count - lag_index
        eeg_slice = eeg_probs[:, :valid_count, :]
        fnirs_slice = fnirs_probs[:, lag_index:lag_index + valid_count, :]
        lag_mapping_log_probs = _conditional_mapping_log_probs(
            coupling_logits[lag_index],
            eeg_slice,
            fnirs_slice,
            residualize_fnirs_marginal=residualize_fnirs_marginal,
        )
        conditional_log_likelihood = torch.einsum(
            'btf,if->bti',
            fnirs_slice,
            lag_mapping_log_probs,
        )
        eeg_mass = eeg_slice.sum(dim=(0, 1)).clamp_min(1e-8)
        conditional_by_eeg = (
            eeg_slice * conditional_log_likelihood
        ).sum(dim=(0, 1)) / eeg_mass

        fnirs_marginal = fnirs_slice.mean(dim=(0, 1)).clamp_min(1e-8)
        fnirs_marginal = fnirs_marginal / fnirs_marginal.sum().clamp_min(1e-8)
        marginal_log_likelihood = torch.einsum(
            'btf,f->bt',
            fnirs_slice,
            fnirs_marginal.log(),
        )
        marginal_by_eeg = (
            eeg_slice * marginal_log_likelihood.unsqueeze(-1)
        ).sum(dim=(0, 1)) / eeg_mass
        gain_columns.append(conditional_by_eeg - marginal_by_eeg)

    if not gain_columns:
        return coupling_logits.new_tensor(0.0)

    gain = torch.stack(gain_columns, dim=-1)
    evidence_target = F.softmax(gain.detach() / max(float(evidence_temperature), 1e-3), dim=-1)
    active_lag_probs = lag_probs[:, :max_lag]
    active_lag_probs = active_lag_probs / active_lag_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(evidence_target * active_lag_probs.log()).sum(dim=-1).mean()


__all__ = [
    'batch_usage_entropy_loss',
    'coupling_eeg_neighbor_smoothness_loss',
    'coupling_joint_probabilities',
    'coupling_lag_focus_loss',
    'coupling_lag_evidence_loss',
    'coupling_pair_likelihood_loss',
    'orthogonality_loss',
    'straight_through_assignment_probs',
]
