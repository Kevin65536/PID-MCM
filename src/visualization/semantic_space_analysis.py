from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.io import write_json

from .shared_alignment_analysis import (
    _active_overlap_summary,
    _codebook_summary,
    _pair_statistics,
    _tensor_to_float,
)

SEMANTIC_GRAD_KEYS = [
    'grad_share_coupling_loss',
    'grad_share_codebook_balance_loss',
    'grad_share_shared_eeg_common_loss',
    'grad_share_shared_fnirs_common_loss',
    'grad_share_eeg_private_residual_loss',
    'grad_share_fnirs_private_residual_loss',
]

RECONSTRUCTION_GRAD_KEYS = [
    'grad_share_eeg_rec_loss',
    'grad_share_fnirs_rec_loss',
]

PENDING_METRICS = [
    'augmentation_consistency',
    'cross_modal_masked_token_prediction_gain',
    'subject_leakage_probe',
    'task_signal_probe',
    'session_device_stability_probe',
]


def _compact_codebook_summary(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        'active_codes': int(summary['active_codes']),
        'usage_rate': float(summary['usage_rate']),
        'dead_codes': int(summary['dead_codes']),
        'entropy': float(summary['entropy']),
        'perplexity': float(summary['perplexity']),
        'gini_coefficient': float(summary['gini_coefficient']),
        'top_codes': [int(x) for x in summary['top_codes'][:10]],
        'top_probs': [float(x) for x in summary['top_probs'][:10]],
        'top_20_coverage': float(summary['top_20_coverage']),
    }


def _codebook_guardrails(summary: Dict[str, object], codebook_size: int) -> Dict[str, bool]:
    return {
        'perplexity_ok': float(summary['perplexity']) >= 0.3 * max(codebook_size, 1),
        'usage_ok': float(summary['usage_rate']) >= 0.2,
        'dead_codes_ok': float(summary['dead_codes']) / max(codebook_size, 1) <= 0.3,
        'concentration_ok': float(summary['top_20_coverage']) <= 0.9,
    }


def _maybe_batch_vector(batch: Dict[str, object], keys: Iterable[str]) -> Optional[np.ndarray]:
    for key in keys:
        value = batch.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            return value.detach().cpu().reshape(-1).numpy().astype(np.int64, copy=False)
        array = np.asarray(value).reshape(-1)
        if array.size > 0:
            return array.astype(np.int64, copy=False)
    return None


def _append_feature_bank(
    feature_chunks: List[np.ndarray],
    token_chunks: List[np.ndarray],
    features: Optional[torch.Tensor],
    tokens: Optional[torch.Tensor],
    remaining: int,
) -> int:
    if remaining <= 0 or features is None or tokens is None:
        return remaining
    flat_features = features.detach().cpu().reshape(-1, features.shape[-1]).numpy().astype(np.float64, copy=False)
    flat_tokens = tokens.detach().cpu().reshape(-1).numpy().astype(np.int64, copy=False)
    if flat_features.shape[0] == 0:
        return remaining
    take = min(int(remaining), int(flat_features.shape[0]))
    feature_chunks.append(flat_features[:take])
    token_chunks.append(flat_tokens[:take])
    return remaining - take


def _safe_concat(chunks: List[np.ndarray], ndim: int) -> np.ndarray:
    if not chunks:
        shape = (0,) if ndim == 1 else (0, 0)
        return np.empty(shape, dtype=np.float64 if ndim > 1 else np.int64)
    return np.concatenate(chunks, axis=0)


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2).item())


def _compute_state_quality(features: np.ndarray, tokens: np.ndarray) -> Dict[str, object]:
    if features.size == 0 or tokens.size == 0:
        return {'available': False}

    overall_mean = features.mean(axis=0, keepdims=True)
    overall_var = float(np.mean(np.sum((features - overall_mean) ** 2, axis=1)))
    unique_tokens = np.unique(tokens)
    within = 0.0
    between = 0.0
    token_weights = {}

    for token in unique_tokens.tolist():
        mask = tokens == token
        token_features = features[mask]
        if token_features.shape[0] == 0:
            continue
        centroid = token_features.mean(axis=0, keepdims=True)
        within += float(np.sum((token_features - centroid) ** 2))
        between += float(token_features.shape[0] * np.sum((centroid - overall_mean) ** 2))
        token_weights[int(token)] = int(token_features.shape[0])

    sample_count = max(int(features.shape[0]), 1)
    within = within / sample_count
    between = between / sample_count
    itcs_norm = float(within / max(overall_var, 1e-12))
    psr = float(between / max(within, 1e-12))
    dominant_tokens = sorted(token_weights.items(), key=lambda item: item[1], reverse=True)[:10]

    return {
        'available': True,
        'sample_count': int(features.shape[0]),
        'active_tokens_in_sample': int(unique_tokens.shape[0]),
        'within_token_dispersion': float(within),
        'overall_dispersion': float(overall_var),
        'itsc_norm': itcs_norm,
        'prototype_separation_ratio': psr,
        'dominant_tokens': [{'token': int(token), 'count': int(count)} for token, count in dominant_tokens],
    }


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts / total
    mask = probs > 0
    return float(-(probs[mask] * np.log(probs[mask] + 1e-12)).sum())


def _compute_transition_metrics(token_matrix: np.ndarray, codebook_size: int, delta: int = 1) -> Dict[str, object]:
    if token_matrix.ndim != 2 or token_matrix.shape[1] <= delta:
        return {'available': False}

    current = token_matrix[:, :-delta].reshape(-1)
    future = token_matrix[:, delta:].reshape(-1)
    if current.size == 0:
        return {'available': False}

    future_counts = np.bincount(future, minlength=codebook_size).astype(np.float64)
    marginal_entropy = _entropy_from_counts(future_counts)
    pair_counts = np.bincount(current * codebook_size + future, minlength=codebook_size * codebook_size)
    pair_counts = pair_counts.reshape(codebook_size, codebook_size).astype(np.float64)
    current_counts = pair_counts.sum(axis=1)

    conditional_entropy = 0.0
    row_top1 = []
    for row_total, row in zip(current_counts.tolist(), pair_counts):
        if row_total <= 0:
            continue
        conditional_entropy += (row_total / current.size) * _entropy_from_counts(row)
        row_top1.append(float(row.max() / row_total))

    return {
        'available': True,
        'delta': int(delta),
        'marginal_entropy': float(marginal_entropy),
        'conditional_entropy': float(conditional_entropy),
        'transition_predictability_gain': float(marginal_entropy - conditional_entropy),
        'self_transition_rate': float(np.mean(current == future)),
        'mean_top1_next_concentration': float(np.mean(row_top1)) if row_top1 else 0.0,
    }


def _conditional_kl(eeg_tokens: np.ndarray, fnirs_tokens: np.ndarray, codebook_size: int, lag: int, target_length: int) -> float:
    usable = min(max(eeg_tokens.shape[1] - lag, 0), int(target_length))
    if usable <= 0:
        return 0.0
    eeg_aligned = eeg_tokens[:, :usable].reshape(-1)
    fnirs_aligned = fnirs_tokens[:, lag:lag + usable].reshape(-1)
    total = int(eeg_aligned.size)
    if total <= 0:
        return 0.0

    marginal_counts = np.bincount(fnirs_aligned, minlength=codebook_size).astype(np.float64)
    marginal = marginal_counts / max(marginal_counts.sum(), 1.0)
    pair_counts = np.bincount(eeg_aligned * codebook_size + fnirs_aligned, minlength=codebook_size * codebook_size)
    pair_counts = pair_counts.reshape(codebook_size, codebook_size).astype(np.float64)
    row_counts = pair_counts.sum(axis=1)

    kl_value = 0.0
    for row_total, row in zip(row_counts.tolist(), pair_counts):
        if row_total <= 0:
            continue
        cond = row / row_total
        mask = cond > 0
        row_kl = float(np.sum(cond[mask] * np.log((cond[mask] + 1e-12) / (marginal[mask] + 1e-12))))
        kl_value += (row_total / total) * row_kl
    return float(kl_value)


def _distribution_balance(summary_a: Dict[str, object], summary_b: Dict[str, object]) -> Dict[str, float]:
    counts_a = np.array(summary_a['counts'], dtype=np.float64)
    counts_b = np.array(summary_b['counts'], dtype=np.float64)
    probs_a = counts_a / max(counts_a.sum(), 1.0)
    probs_b = counts_b / max(counts_b.sum(), 1.0)
    total_variation = 0.5 * float(np.abs(probs_a - probs_b).sum())
    return {
        'total_variation_distance': total_variation,
        'distribution_balance': float(1.0 - total_variation),
    }


def _update_group_counts(
    accumulator: Dict[int, np.ndarray],
    group_ids: Optional[np.ndarray],
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    codebook_size: int,
) -> None:
    if group_ids is None:
        return
    for row_index, group_id in enumerate(group_ids.tolist()):
        shared_counts = np.bincount(eeg_tokens[row_index], minlength=codebook_size)
        shared_counts += np.bincount(fnirs_tokens[row_index], minlength=codebook_size)
        if int(group_id) not in accumulator:
            accumulator[int(group_id)] = np.zeros(codebook_size, dtype=np.float64)
        accumulator[int(group_id)] += shared_counts.astype(np.float64)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = float(np.sum(p[mask_p] * np.log((p[mask_p] + 1e-12) / (m[mask_p] + 1e-12))))
    kl_qm = float(np.sum(q[mask_q] * np.log((q[mask_q] + 1e-12) / (m[mask_q] + 1e-12))))
    return 0.5 * (kl_pm + kl_qm)


def _group_separation_proxy(group_counts: Dict[int, np.ndarray]) -> Dict[str, object]:
    if len(group_counts) < 2:
        return {'available': False}
    divergences = []
    for left, right in combinations(sorted(group_counts), 2):
        divergences.append(_js_divergence(group_counts[left], group_counts[right]))
    mean_js = float(np.mean(divergences)) if divergences else 0.0
    return {
        'available': True,
        'group_count': int(len(group_counts)),
        'mean_js_divergence': mean_js,
        'normalized_mean_js_divergence': float(mean_js / math.log(2.0)) if mean_js > 0 else 0.0,
    }


def _load_training_dynamics(run_dir: Optional[Path]) -> Dict[str, object]:
    if run_dir is None:
        return {'available': False, 'reason': 'run_dir_not_provided'}
    metrics_path = Path(run_dir) / 'metrics.json'
    if not metrics_path.exists():
        return {'available': False, 'reason': 'metrics_json_not_found'}

    payload = json.loads(metrics_path.read_text(encoding='utf-8'))
    epochs = payload.get('epochs', [])
    if not epochs:
        return {'available': False, 'reason': 'epoch_history_missing'}

    series = []
    for item in epochs:
        metrics = item.get('metrics', {}) or {}
        semantic_share = sum(float(metrics.get(key, 0.0)) for key in SEMANTIC_GRAD_KEYS)
        reconstruction_share = sum(float(metrics.get(key, 0.0)) for key in RECONSTRUCTION_GRAD_KEYS)
        series.append({
            'epoch': int(item.get('epoch', len(series) + 1)),
            'semantic_gradient_share': semantic_share,
            'reconstruction_gradient_share': reconstruction_share,
            'semantic_to_reconstruction_ratio': float(semantic_share / max(reconstruction_share, 1e-12)),
            'grad_conflict_rate': float(metrics.get('grad_conflict_rate', 0.0)),
            'grad_mean_pairwise_cosine': float(metrics.get('grad_mean_pairwise_cosine', 0.0)),
            'grad_min_pairwise_cosine': float(metrics.get('grad_min_pairwise_cosine', 0.0)),
        })

    final = series[-1]
    return {
        'available': True,
        'epochs_analyzed': int(len(series)),
        'final_epoch': int(final['epoch']),
        'final_semantic_gradient_share': float(final['semantic_gradient_share']),
        'peak_semantic_gradient_share': float(max(entry['semantic_gradient_share'] for entry in series)),
        'final_reconstruction_gradient_share': float(final['reconstruction_gradient_share']),
        'final_semantic_to_reconstruction_ratio': float(final['semantic_to_reconstruction_ratio']),
        'final_grad_conflict_rate': float(final['grad_conflict_rate']),
        'max_grad_conflict_rate': float(max(entry['grad_conflict_rate'] for entry in series)),
        'final_grad_mean_pairwise_cosine': float(final['grad_mean_pairwise_cosine']),
        'min_grad_mean_pairwise_cosine': float(min(entry['grad_mean_pairwise_cosine'] for entry in series)),
        'final_grad_min_pairwise_cosine': float(final['grad_min_pairwise_cosine']),
        'series': series,
    }


def _collect_split_data(
    model,
    dataloader,
    device: torch.device,
    shared_codebook_size: int,
    max_batches: Optional[int],
    max_feature_samples: int,
) -> Dict[str, object]:
    scalar_totals: Dict[str, float] = {}
    total_batches = 0

    eeg_shared_chunks: List[np.ndarray] = []
    fnirs_shared_chunks: List[np.ndarray] = []
    eeg_private_chunks: List[np.ndarray] = []
    fnirs_private_chunks: List[np.ndarray] = []

    feature_banks: Dict[str, List[np.ndarray]] = {
        'shared_combined_features': [],
        'shared_combined_tokens': [],
        'shared_eeg_features': [],
        'shared_eeg_tokens': [],
        'shared_fnirs_features': [],
        'shared_fnirs_tokens': [],
        'eeg_private_features': [],
        'eeg_private_tokens': [],
        'fnirs_private_features': [],
        'fnirs_private_tokens': [],
    }
    remaining = {
        'shared_combined': max_feature_samples,
        'shared_eeg': max_feature_samples,
        'shared_fnirs': max_feature_samples,
        'eeg_private': max_feature_samples,
        'fnirs_private': max_feature_samples,
    }

    subject_group_counts: Dict[int, np.ndarray] = {}
    label_group_counts: Dict[int, np.ndarray] = {}

    recon_totals: Dict[str, float] = {
        'eeg_full_mse': 0.0,
        'fnirs_full_mse': 0.0,
        'eeg_shared_only_mse': 0.0,
        'fnirs_shared_only_mse': 0.0,
        'eeg_private_only_mse': 0.0,
        'fnirs_private_only_mse': 0.0,
    }
    branch_totals: Dict[str, float] = {
        'eeg_shared_common_mse': 0.0,
        'eeg_private_common_mse': 0.0,
        'eeg_shared_residual_mse': 0.0,
        'eeg_private_residual_mse': 0.0,
        'fnirs_shared_common_mse': 0.0,
        'fnirs_private_common_mse': 0.0,
        'fnirs_shared_residual_mse': 0.0,
        'fnirs_private_residual_mse': 0.0,
    }

    has_private = False
    model.eval()

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= int(max_batches):
            break

        eeg = batch['eeg'].to(device)
        fnirs = batch['fnirs'].to(device)
        outputs = model(eeg, fnirs)
        total_batches += 1
        has_private = has_private or ('eeg_private_indices' in outputs)

        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim == 0:
                scalar_totals[key] = scalar_totals.get(key, 0.0) + _tensor_to_float(value)

        eeg_shared_tokens = outputs['eeg_indices'].detach().cpu().numpy().astype(np.int64, copy=False)
        fnirs_shared_tokens = outputs['fnirs_indices'].detach().cpu().numpy().astype(np.int64, copy=False)
        eeg_shared_chunks.append(eeg_shared_tokens)
        fnirs_shared_chunks.append(fnirs_shared_tokens)

        remaining['shared_combined'] = _append_feature_bank(
            feature_banks['shared_combined_features'],
            feature_banks['shared_combined_tokens'],
            outputs.get('eeg_z'),
            outputs.get('eeg_indices'),
            remaining['shared_combined'],
        )
        remaining['shared_combined'] = _append_feature_bank(
            feature_banks['shared_combined_features'],
            feature_banks['shared_combined_tokens'],
            outputs.get('fnirs_z'),
            outputs.get('fnirs_indices'),
            remaining['shared_combined'],
        )
        remaining['shared_eeg'] = _append_feature_bank(
            feature_banks['shared_eeg_features'],
            feature_banks['shared_eeg_tokens'],
            outputs.get('eeg_z'),
            outputs.get('eeg_indices'),
            remaining['shared_eeg'],
        )
        remaining['shared_fnirs'] = _append_feature_bank(
            feature_banks['shared_fnirs_features'],
            feature_banks['shared_fnirs_tokens'],
            outputs.get('fnirs_z'),
            outputs.get('fnirs_indices'),
            remaining['shared_fnirs'],
        )

        recon_totals['eeg_full_mse'] += _mse(outputs['eeg_reconstructed'], eeg)
        recon_totals['fnirs_full_mse'] += _mse(outputs['fnirs_reconstructed'], fnirs)

        subject_ids = _maybe_batch_vector(batch, ('subject', 'subject_id'))
        label_ids = _maybe_batch_vector(batch, ('label', 'labels', 'task', 'condition'))
        _update_group_counts(subject_group_counts, subject_ids, eeg_shared_tokens, fnirs_shared_tokens, shared_codebook_size)
        _update_group_counts(label_group_counts, label_ids, eeg_shared_tokens, fnirs_shared_tokens, shared_codebook_size)

        if 'eeg_private_indices' in outputs:
            eeg_private_tokens = outputs['eeg_private_indices'].detach().cpu().numpy().astype(np.int64, copy=False)
            fnirs_private_tokens = outputs['fnirs_private_indices'].detach().cpu().numpy().astype(np.int64, copy=False)
            eeg_private_chunks.append(eeg_private_tokens)
            fnirs_private_chunks.append(fnirs_private_tokens)

            remaining['eeg_private'] = _append_feature_bank(
                feature_banks['eeg_private_features'],
                feature_banks['eeg_private_tokens'],
                outputs.get('eeg_private_z'),
                outputs.get('eeg_private_indices'),
                remaining['eeg_private'],
            )
            remaining['fnirs_private'] = _append_feature_bank(
                feature_banks['fnirs_private_features'],
                feature_banks['fnirs_private_tokens'],
                outputs.get('fnirs_private_z'),
                outputs.get('fnirs_private_indices'),
                remaining['fnirs_private'],
            )

            recon_totals['eeg_shared_only_mse'] += _mse(outputs['eeg_shared_only_reconstructed'], eeg)
            recon_totals['fnirs_shared_only_mse'] += _mse(outputs['fnirs_shared_only_reconstructed'], fnirs)
            recon_totals['eeg_private_only_mse'] += _mse(outputs['eeg_private_only_reconstructed'], eeg)
            recon_totals['fnirs_private_only_mse'] += _mse(outputs['fnirs_private_only_reconstructed'], fnirs)

            if hasattr(model, '_smooth_signal') and hasattr(model, 'eeg_common_pool_kernel') and hasattr(model, 'fnirs_common_pool_kernel'):
                eeg_common = model._smooth_signal(eeg, model.eeg_common_pool_kernel)
                fnirs_common = model._smooth_signal(fnirs, model.fnirs_common_pool_kernel)
                eeg_residual = eeg - eeg_common
                fnirs_residual = fnirs - fnirs_common

                branch_totals['eeg_shared_common_mse'] += _mse(outputs['eeg_shared_only_reconstructed'], eeg_common)
                branch_totals['eeg_private_common_mse'] += _mse(outputs['eeg_private_only_reconstructed'], eeg_common)
                branch_totals['eeg_shared_residual_mse'] += _mse(outputs['eeg_shared_only_reconstructed'], eeg_residual)
                branch_totals['eeg_private_residual_mse'] += _mse(outputs['eeg_private_only_reconstructed'], eeg_residual)
                branch_totals['fnirs_shared_common_mse'] += _mse(outputs['fnirs_shared_only_reconstructed'], fnirs_common)
                branch_totals['fnirs_private_common_mse'] += _mse(outputs['fnirs_private_only_reconstructed'], fnirs_common)
                branch_totals['fnirs_shared_residual_mse'] += _mse(outputs['fnirs_shared_only_reconstructed'], fnirs_residual)
                branch_totals['fnirs_private_residual_mse'] += _mse(outputs['fnirs_private_only_reconstructed'], fnirs_residual)

    mean_scalars = {key: value / max(total_batches, 1) for key, value in scalar_totals.items()}
    mean_recon = {key: value / max(total_batches, 1) for key, value in recon_totals.items()}
    mean_branch = {key: value / max(total_batches, 1) for key, value in branch_totals.items()}

    return {
        'has_private': has_private,
        'total_batches': int(total_batches),
        'mean_scalars': mean_scalars,
        'mean_reconstruction_mse': mean_recon,
        'mean_branch_mse': mean_branch,
        'eeg_shared_tokens': _safe_concat(eeg_shared_chunks, ndim=2),
        'fnirs_shared_tokens': _safe_concat(fnirs_shared_chunks, ndim=2),
        'eeg_private_tokens': _safe_concat(eeg_private_chunks, ndim=2),
        'fnirs_private_tokens': _safe_concat(fnirs_private_chunks, ndim=2),
        'feature_banks': {
            key: _safe_concat(value, ndim=2 if 'features' in key else 1)
            for key, value in feature_banks.items()
        },
        'subject_group_counts': subject_group_counts,
        'label_group_counts': label_group_counts,
    }


def _build_branch_responsibility(mean_branch: Dict[str, float], has_private: bool) -> Dict[str, object]:
    if not has_private:
        return {'available': False, 'reason': 'no_private_branches'}

    eeg_shared_common_advantage = float(mean_branch['eeg_private_common_mse'] - mean_branch['eeg_shared_common_mse'])
    eeg_private_residual_advantage = float(mean_branch['eeg_shared_residual_mse'] - mean_branch['eeg_private_residual_mse'])
    fnirs_shared_common_advantage = float(mean_branch['fnirs_private_common_mse'] - mean_branch['fnirs_shared_common_mse'])
    fnirs_private_residual_advantage = float(mean_branch['fnirs_shared_residual_mse'] - mean_branch['fnirs_private_residual_mse'])

    return {
        'available': True,
        'eeg': {
            'shared_common_advantage': eeg_shared_common_advantage,
            'private_residual_advantage': eeg_private_residual_advantage,
            'semantic_split_ok': bool(eeg_shared_common_advantage > 0.0 and eeg_private_residual_advantage > 0.0),
        },
        'fnirs': {
            'shared_common_advantage': fnirs_shared_common_advantage,
            'private_residual_advantage': fnirs_private_residual_advantage,
            'semantic_split_ok': bool(fnirs_shared_common_advantage > 0.0 and fnirs_private_residual_advantage > 0.0),
        },
    }


def _build_layer_d(split_data: Dict[str, object]) -> Dict[str, object]:
    subject_proxy = _group_separation_proxy(split_data['subject_group_counts'])
    label_proxy = _group_separation_proxy(split_data['label_group_counts'])
    selectivity = None
    if subject_proxy.get('available') and label_proxy.get('available'):
        selectivity = float(
            label_proxy['normalized_mean_js_divergence'] /
            max(subject_proxy['normalized_mean_js_divergence'], 1e-12)
        )
    return {
        'available': bool(subject_proxy.get('available') or label_proxy.get('available')),
        'subject_distribution_separation_proxy': subject_proxy,
        'label_distribution_separation_proxy': label_proxy,
        'semantic_selectivity_proxy': selectivity,
        'pending_probe_metrics': PENDING_METRICS,
    }


def _build_reasonableness_heuristic(
    layer_a: Dict[str, object],
    layer_b: Dict[str, object],
    layer_c: Dict[str, object],
) -> Dict[str, object]:
    checks = []
    shared_eeg = layer_a['shared_eeg_codebook']
    shared_fnirs = layer_a['shared_fnirs_codebook']
    shared_state = layer_b['state_quality'].get('shared_combined', {})
    best_lag = layer_c.get('best_lag', {})
    branch = layer_a.get('branch_responsibility', {})

    checks.append({'name': 'shared_eeg_not_collapsed', 'pass': bool(shared_eeg['perplexity'] > 3.0)})
    checks.append({'name': 'shared_fnirs_not_collapsed', 'pass': bool(shared_fnirs['perplexity'] > 8.0)})
    checks.append({'name': 'shared_usage_balanced', 'pass': bool(layer_c.get('shared_usage_distribution_balance', 0.0) > 0.35)})
    checks.append({'name': 'positive_best_lag_mi_gain', 'pass': bool(best_lag.get('mi_improvement', 0.0) > 0.0)})
    checks.append({'name': 'positive_corrected_lmig', 'pass': bool(layer_c.get('corrected_lmig', 0.0) > 0.0)})
    if shared_state.get('available'):
        checks.append({'name': 'shared_state_consistent', 'pass': bool(shared_state.get('itsc_norm', 1.0) < 0.8)})
        checks.append({'name': 'shared_state_separated', 'pass': bool(shared_state.get('prototype_separation_ratio', 0.0) > 0.2)})
    if branch.get('available'):
        checks.append({'name': 'eeg_branch_semantics', 'pass': bool(branch['eeg']['semantic_split_ok'])})
        checks.append({'name': 'fnirs_branch_semantics', 'pass': bool(branch['fnirs']['semantic_split_ok'])})

    passed = sum(1 for item in checks if item['pass'])
    total = max(len(checks), 1)
    ratio = passed / total
    if ratio >= 0.75:
        verdict = 'strong'
    elif ratio >= 0.45:
        verdict = 'partial'
    else:
        verdict = 'weak'
    return {
        'heuristic_only': True,
        'passed': int(passed),
        'total': int(total),
        'pass_ratio': float(ratio),
        'verdict': verdict,
        'checks': checks,
    }


def _save_split_dashboard(path: Path, split_name: str, split_result: Dict[str, object]) -> None:
    layer_a = split_result['layer_a']
    layer_b = split_result['layer_b']
    layer_c = split_result['layer_c']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    codebook_names = ['shared_eeg', 'shared_fnirs']
    perplexity_ratios = [
        float(layer_a['shared_eeg_codebook']['perplexity_ratio']),
        float(layer_a['shared_fnirs_codebook']['perplexity_ratio']),
    ]
    gini_values = [
        float(layer_a['shared_eeg_codebook']['gini_coefficient']),
        float(layer_a['shared_fnirs_codebook']['gini_coefficient']),
    ]
    if 'eeg_private_codebook' in layer_a:
        codebook_names.extend(['eeg_private', 'fnirs_private'])
        perplexity_ratios.extend([
            float(layer_a['eeg_private_codebook']['perplexity_ratio']),
            float(layer_a['fnirs_private_codebook']['perplexity_ratio']),
        ])
        gini_values.extend([
            float(layer_a['eeg_private_codebook']['gini_coefficient']),
            float(layer_a['fnirs_private_codebook']['gini_coefficient']),
        ])

    axes[0, 0].bar(codebook_names, perplexity_ratios, color=['steelblue', 'forestgreen', 'darkorange', 'purple'][:len(codebook_names)])
    axes[0, 0].set_title(f'{split_name.upper()} Perplexity Ratio')
    axes[0, 0].set_ylabel('Perplexity / codebook_size')
    axes[0, 0].tick_params(axis='x', rotation=20)

    axes[0, 1].bar(codebook_names, gini_values, color=['steelblue', 'forestgreen', 'darkorange', 'purple'][:len(codebook_names)])
    axes[0, 1].set_title(f'{split_name.upper()} Usage Gini')
    axes[0, 1].set_ylabel('Gini coefficient')
    axes[0, 1].tick_params(axis='x', rotation=20)

    state_quality = layer_b['state_quality']
    state_names = []
    itsc_values = []
    psr_values = []
    for key in ('shared_combined', 'shared_eeg', 'shared_fnirs', 'eeg_private', 'fnirs_private'):
        entry = state_quality.get(key, {})
        if entry.get('available'):
            state_names.append(key)
            itsc_values.append(float(entry.get('itsc_norm', 0.0)))
            psr_values.append(float(entry.get('prototype_separation_ratio', 0.0)))
    axes[0, 2].bar(state_names, itsc_values, color='slateblue')
    axes[0, 2].set_title(f'{split_name.upper()} ITSC norm')
    axes[0, 2].set_ylabel('Lower is better')
    axes[0, 2].tick_params(axis='x', rotation=30)

    axes[1, 0].bar(state_names, psr_values, color='teal')
    axes[1, 0].set_title(f'{split_name.upper()} Prototype Separation')
    axes[1, 0].set_ylabel('Higher is better')
    axes[1, 0].tick_params(axis='x', rotation=30)

    transition = layer_b['transition_dynamics']
    transition_names = []
    transition_values = []
    for key in ('shared_eeg', 'shared_fnirs', 'eeg_private', 'fnirs_private'):
        entry = transition.get(key, {})
        if entry.get('available'):
            transition_names.append(key)
            transition_values.append(float(entry.get('transition_predictability_gain', 0.0)))
    axes[1, 1].bar(transition_names, transition_values, color='darkorange')
    axes[1, 1].set_title(f'{split_name.upper()} Transition Gain')
    axes[1, 1].set_ylabel('TPG')
    axes[1, 1].tick_params(axis='x', rotation=30)

    lag_metrics = layer_c.get('lag_metrics', [])
    if lag_metrics:
        lags = [int(item['lag']) for item in lag_metrics]
        mi_gain = [float(item['mi_improvement']) for item in lag_metrics]
        axes[1, 2].plot(lags, mi_gain, marker='o', color='crimson', label='Corrected MI')
        axes[1, 2].axvline(int(layer_c['best_lag']['lag']), color='black', linestyle='--', alpha=0.7, label='Best lag')
        axes[1, 2].legend()
    else:
        axes[1, 2].text(0.5, 0.5, 'No lag metrics', ha='center', va='center')
    axes[1, 2].set_title(f'{split_name.upper()} Lagged Coupling')
    axes[1, 2].set_xlabel('Lag')
    axes[1, 2].set_ylabel('MI improvement')

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_training_dynamics_dashboard(path: Path, summary: Dict[str, object]) -> None:
    series = summary.get('series', [])
    if not series:
        return

    epochs = [int(item['epoch']) for item in series]
    semantic_share = [float(item['semantic_gradient_share']) for item in series]
    reconstruction_share = [float(item['reconstruction_gradient_share']) for item in series]
    conflict_rate = [float(item['grad_conflict_rate']) for item in series]
    mean_cosine = [float(item['grad_mean_pairwise_cosine']) for item in series]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, semantic_share, color='teal', label='semantic')
    axes[0].plot(epochs, reconstruction_share, color='steelblue', label='reconstruction')
    axes[0].set_title('Gradient Share Budget')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Share')
    axes[0].legend()

    axes[1].plot(epochs, conflict_rate, color='crimson')
    axes[1].set_title('Conflict Rate')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Conflict rate')

    axes[2].plot(epochs, mean_cosine, color='slateblue')
    axes[2].axhline(0.0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Mean Pairwise Cosine')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Cosine')

    for axis in axes:
        axis.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def analyze_semantic_space(
    model,
    dataloaders: Dict[str, object],
    config: Dict[str, object],
    output_dir: Path,
    device: torch.device,
    splits: Iterable[str] = ('val', 'test'),
    run_dir: Optional[Path] = None,
    max_batches: Optional[int] = None,
    max_feature_samples: int = 20000,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    lag_set = [int(x) for x in config.get('validation', {}).get('lag_set', [0, 1, 2, 3, 4, 5])]
    shared_codebook_size = int(getattr(model, 'shared_codebook_size', getattr(model, 'codebook_size', model.get_codebook_size())))
    eeg_private_codebook_size = int(getattr(model, 'eeg_private_codebook_size', 0))
    fnirs_private_codebook_size = int(getattr(model, 'fnirs_private_codebook_size', 0))
    training_dynamics = _load_training_dynamics(run_dir)

    results: Dict[str, object] = {
        'analysis_type': 'semantic_space',
        'lag_set': lag_set,
        'shared_codebook_size': shared_codebook_size,
        'eeg_private_codebook_size': eeg_private_codebook_size,
        'fnirs_private_codebook_size': fnirs_private_codebook_size,
        'training_dynamics': training_dynamics,
        'implemented_metrics': [
            'codebook_health_guardrails',
            'reconstruction_guardrails',
            'branch_responsibility_gap',
            'lagged_mutual_information_gain',
            'conditional_kl_gain',
            'shared_usage_distribution_balance',
            'intra_token_state_consistency',
            'prototype_separation_ratio',
            'transition_predictability_gain',
            'metadata_distribution_separation_proxies',
            'gradient_semantic_support_summary',
        ],
        'pending_metrics': PENDING_METRICS,
        'splits': {},
    }

    if training_dynamics.get('available'):
        _save_training_dynamics_dashboard(output_dir / 'training_dynamics_dashboard.png', training_dynamics)

    model.eval()
    for split_name in splits:
        dataloader = dataloaders.get(split_name)
        if dataloader is None:
            continue
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_data = _collect_split_data(
            model=model,
            dataloader=dataloader,
            device=device,
            shared_codebook_size=shared_codebook_size,
            max_batches=max_batches,
            max_feature_samples=max_feature_samples,
        )
        if split_data['total_batches'] == 0:
            continue

        mean_scalars = split_data['mean_scalars']
        shared_eeg_summary = _codebook_summary(split_data['eeg_shared_tokens'].reshape(-1), shared_codebook_size)
        shared_fnirs_summary = _codebook_summary(split_data['fnirs_shared_tokens'].reshape(-1), shared_codebook_size)
        shared_overlap = _active_overlap_summary(shared_eeg_summary, shared_fnirs_summary)

        layer_a: Dict[str, object] = {
            'reconstruction_guardrails': {
                **{key: float(value) for key, value in split_data['mean_reconstruction_mse'].items()},
                'eeg_rec_loss_objective': float(mean_scalars.get('eeg_rec_loss', 0.0)),
                'fnirs_rec_loss_objective': float(mean_scalars.get('fnirs_rec_loss', 0.0)),
                'shared_eeg_common_loss_objective': float(mean_scalars.get('shared_eeg_common_loss', 0.0)),
                'shared_fnirs_common_loss_objective': float(mean_scalars.get('shared_fnirs_common_loss', 0.0)),
                'eeg_private_residual_loss_objective': float(mean_scalars.get('eeg_private_residual_loss', 0.0)),
                'fnirs_private_residual_loss_objective': float(mean_scalars.get('fnirs_private_residual_loss', 0.0)),
            },
            'shared_eeg_codebook': {
                **_compact_codebook_summary(shared_eeg_summary),
                'codebook_size': shared_codebook_size,
                'perplexity_ratio': float(shared_eeg_summary['perplexity']) / max(shared_codebook_size, 1),
                'guardrails': _codebook_guardrails(shared_eeg_summary, shared_codebook_size),
            },
            'shared_fnirs_codebook': {
                **_compact_codebook_summary(shared_fnirs_summary),
                'codebook_size': shared_codebook_size,
                'perplexity_ratio': float(shared_fnirs_summary['perplexity']) / max(shared_codebook_size, 1),
                'guardrails': _codebook_guardrails(shared_fnirs_summary, shared_codebook_size),
            },
            'shared_overlap': shared_overlap,
        }

        if split_data['has_private'] and split_data['eeg_private_tokens'].size > 0 and split_data['fnirs_private_tokens'].size > 0:
            eeg_private_summary = _codebook_summary(split_data['eeg_private_tokens'].reshape(-1), eeg_private_codebook_size)
            fnirs_private_summary = _codebook_summary(split_data['fnirs_private_tokens'].reshape(-1), fnirs_private_codebook_size)
            layer_a['eeg_private_codebook'] = {
                **_compact_codebook_summary(eeg_private_summary),
                'codebook_size': eeg_private_codebook_size,
                'perplexity_ratio': float(eeg_private_summary['perplexity']) / max(eeg_private_codebook_size, 1),
                'guardrails': _codebook_guardrails(eeg_private_summary, eeg_private_codebook_size),
            }
            layer_a['fnirs_private_codebook'] = {
                **_compact_codebook_summary(fnirs_private_summary),
                'codebook_size': fnirs_private_codebook_size,
                'perplexity_ratio': float(fnirs_private_summary['perplexity']) / max(fnirs_private_codebook_size, 1),
                'guardrails': _codebook_guardrails(fnirs_private_summary, fnirs_private_codebook_size),
            }
        layer_a['branch_responsibility'] = _build_branch_responsibility(split_data['mean_branch_mse'], split_data['has_private'])

        state_quality = {
            'shared_combined': _compute_state_quality(
                split_data['feature_banks']['shared_combined_features'],
                split_data['feature_banks']['shared_combined_tokens'],
            ),
            'shared_eeg': _compute_state_quality(
                split_data['feature_banks']['shared_eeg_features'],
                split_data['feature_banks']['shared_eeg_tokens'],
            ),
            'shared_fnirs': _compute_state_quality(
                split_data['feature_banks']['shared_fnirs_features'],
                split_data['feature_banks']['shared_fnirs_tokens'],
            ),
        }
        if split_data['has_private']:
            state_quality['eeg_private'] = _compute_state_quality(
                split_data['feature_banks']['eeg_private_features'],
                split_data['feature_banks']['eeg_private_tokens'],
            )
            state_quality['fnirs_private'] = _compute_state_quality(
                split_data['feature_banks']['fnirs_private_features'],
                split_data['feature_banks']['fnirs_private_tokens'],
            )

        transition_dynamics = {
            'shared_eeg': _compute_transition_metrics(split_data['eeg_shared_tokens'], shared_codebook_size),
            'shared_fnirs': _compute_transition_metrics(split_data['fnirs_shared_tokens'], shared_codebook_size),
        }
        if split_data['has_private'] and split_data['eeg_private_tokens'].size > 0:
            transition_dynamics['eeg_private'] = _compute_transition_metrics(split_data['eeg_private_tokens'], eeg_private_codebook_size)
            transition_dynamics['fnirs_private'] = _compute_transition_metrics(split_data['fnirs_private_tokens'], fnirs_private_codebook_size)

        fixed_compare_length = min(max(split_data['eeg_shared_tokens'].shape[1] - lag, 0) for lag in lag_set) if lag_set else split_data['eeg_shared_tokens'].shape[1]
        lag_metrics = [
            _pair_statistics(
                split_data['eeg_shared_tokens'],
                split_data['fnirs_shared_tokens'],
                shared_codebook_size,
                lag,
                target_length=fixed_compare_length,
            )
            for lag in lag_set
        ]
        lag_zero = next((item for item in lag_metrics if item['lag'] == 0), lag_metrics[0])
        best_lag = max(lag_metrics, key=lambda item: (item['mi_improvement'], -item['lag']))
        conditional_kl = _conditional_kl(
            split_data['eeg_shared_tokens'],
            split_data['fnirs_shared_tokens'],
            shared_codebook_size,
            int(best_lag['lag']),
            int(best_lag['compare_length']),
        )
        balance_summary = _distribution_balance(shared_eeg_summary, shared_fnirs_summary)

        layer_b = {
            'state_quality': state_quality,
            'transition_dynamics': transition_dynamics,
        }
        layer_c = {
            'lag_metrics': lag_metrics,
            'lag_zero': lag_zero,
            'best_lag': best_lag,
            'raw_lmig': float(best_lag['mutual_information'] - lag_zero['mutual_information']),
            'corrected_lmig': float(best_lag['mi_improvement'] - lag_zero['mi_improvement']),
            'conditional_kl': float(conditional_kl),
            'conditional_kl_gain_vs_shuffle': float(best_lag['mi_improvement']),
            'shared_usage_distribution_balance': float(balance_summary['distribution_balance']),
            'shared_usage_total_variation_distance': float(balance_summary['total_variation_distance']),
            'supplementary_overlap': shared_overlap,
        }
        layer_d = _build_layer_d(split_data)
        heuristic = _build_reasonableness_heuristic(layer_a, layer_b, layer_c)

        split_result = {
            'total_batches': int(split_data['total_batches']),
            'layer_a': layer_a,
            'layer_b': layer_b,
            'layer_c': layer_c,
            'layer_d': layer_d,
            'heuristic_reasonableness': heuristic,
        }

        write_json(split_dir / 'semantic_scorecard_summary.json', split_result)
        _save_split_dashboard(split_dir / 'semantic_scorecard_dashboard.png', split_name, split_result)
        results['splits'][split_name] = split_result

    write_json(output_dir / 'summary.json', results)
    return results


__all__ = ['analyze_semantic_space']