from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.io import write_json

from .analysis_artifacts import prepare_analysis_layout

from .shared_alignment_analysis import (
    _active_overlap_summary,
    _codebook_summary,
    _estimate_sampling_rate,
    _pair_statistics,
    _plot_single_spectral_row,
    _reduce_embeddings,
    _save_codebook_diagnostics,
    _save_cross_modal_coupling_plot,
    _save_heatmap,
    _save_lag_plot,
    _save_pairing_dashboard,
    _save_probe_style_lag_plot,
    _save_reconstruction_plot,
    _save_spectral_comparison_plot,
    _save_token_pattern_plot,
    _save_usage_plot,
    _tensor_to_float,
)


def _save_branch_usage_dashboard(
    path: Path,
    shared_overlap: Dict[str, object],
    eeg_private_summary: Dict[str, object],
    fnirs_private_summary: Dict[str, object],
    split_name: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].bar(
        ['shared_intersection', 'shared_union', 'shared_jaccard'],
        [
            float(shared_overlap['intersection_count']),
            float(shared_overlap['union_count']),
            float(shared_overlap['jaccard']),
        ],
        color=['slateblue', 'darkgray', 'teal'],
        alpha=0.85,
    )
    axes[0].set_title(f'{split_name.upper()} Shared Support')

    axes[1].bar(
        ['active', 'perplexity', 'top20cov'],
        [
            float(eeg_private_summary['active_codes']),
            float(eeg_private_summary['perplexity']),
            float(eeg_private_summary['top_20_coverage']),
        ],
        color='steelblue',
        alpha=0.85,
    )
    axes[1].set_title(f'{split_name.upper()} EEG Private Codes')

    axes[2].bar(
        ['active', 'perplexity', 'top20cov'],
        [
            float(fnirs_private_summary['active_codes']),
            float(fnirs_private_summary['perplexity']),
            float(fnirs_private_summary['top_20_coverage']),
        ],
        color='forestgreen',
        alpha=0.85,
    )
    axes[2].set_title(f'{split_name.upper()} fNIRS Private Codes')

    for axis in axes:
        axis.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _summary_with_padded_counts(summary: Dict[str, object], target_size: int) -> Dict[str, object]:
    padded = dict(summary)
    counts = np.array(summary['counts'])
    if counts.shape[0] < target_size:
        counts = np.pad(counts, (0, target_size - counts.shape[0]))
    padded['counts'] = counts
    return padded


def _save_coupling_matrix_plot(path: Path, coupling_logits: np.ndarray, lag_candidates: List[int], split_name: str):
    probs = np.exp(coupling_logits - coupling_logits.max(axis=-1, keepdims=True))
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-12, None)
    mean_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1).mean(axis=-1)
    best_index = int(np.argmin(mean_entropy)) if mean_entropy.size else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(lag_candidates, mean_entropy, marker='o', color='darkorange')
    axes[0].set_title(f'{split_name.upper()} Coupling Matrix Entropy')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Mean row entropy')
    axes[0].grid(True, alpha=0.3)

    im = axes[1].imshow(probs[best_index], aspect='auto', cmap='viridis')
    axes[1].set_title(f'{split_name.upper()} Coupling Matrix @ lag={lag_candidates[best_index]}')
    axes[1].set_xlabel('Predicted fNIRS shared code')
    axes[1].set_ylabel('EEG shared code')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_component_embedding_plot(
    path: Path,
    embeddings: np.ndarray,
    counts: np.ndarray,
    split_name: str,
    title: str,
):
    reduced, method = _reduce_embeddings(embeddings)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=np.log1p(counts),
        cmap='viridis',
        s=16 + 52 * (counts > 0).astype(np.float32),
        alpha=0.82,
    )
    ax.set_title(f'{split_name.upper()} {title} ({method.upper()})')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label='log(1 + usage)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_branch_ablation_plot(
    path: Path,
    eeg_signal: np.ndarray,
    fnirs_signal: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    split_name: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    eeg_time = np.arange(eeg_signal.shape[-1])
    fnirs_time = np.arange(fnirs_signal.shape[-1])

    axes[0, 0].plot(eeg_time, eeg_signal.mean(axis=0), color='black', linewidth=2, label='Original')
    axes[0, 0].plot(eeg_time, reconstructions['full_eeg'].mean(axis=0), color='steelblue', linewidth=1.6, label='Full')
    axes[0, 0].plot(eeg_time, reconstructions['shared_only_eeg'].mean(axis=0), color='darkorange', linewidth=1.4, label='Shared only')
    axes[0, 0].plot(eeg_time, reconstructions['private_only_eeg'].mean(axis=0), color='forestgreen', linewidth=1.4, label='Private only')
    axes[0, 0].set_title(f'{split_name.upper()} EEG Branch Ablation')
    axes[0, 0].legend()

    axes[0, 1].bar(
        ['full', 'shared_only', 'private_only'],
        [
            float(np.mean((reconstructions['full_eeg'] - eeg_signal) ** 2)),
            float(np.mean((reconstructions['shared_only_eeg'] - eeg_signal) ** 2)),
            float(np.mean((reconstructions['private_only_eeg'] - eeg_signal) ** 2)),
        ],
        color=['steelblue', 'darkorange', 'forestgreen'],
        alpha=0.85,
    )
    axes[0, 1].set_title(f'{split_name.upper()} EEG Ablation MSE')

    axes[1, 0].plot(fnirs_time, fnirs_signal.mean(axis=0), color='black', linewidth=2, label='Original')
    axes[1, 0].plot(fnirs_time, reconstructions['full_fnirs'].mean(axis=0), color='crimson', linewidth=1.6, label='Full')
    axes[1, 0].plot(fnirs_time, reconstructions['shared_only_fnirs'].mean(axis=0), color='purple', linewidth=1.4, label='Shared only')
    axes[1, 0].plot(fnirs_time, reconstructions['private_only_fnirs'].mean(axis=0), color='teal', linewidth=1.4, label='Private only')
    axes[1, 0].set_title(f'{split_name.upper()} fNIRS Branch Ablation')
    axes[1, 0].legend()

    axes[1, 1].bar(
        ['full', 'shared_only', 'private_only'],
        [
            float(np.mean((reconstructions['full_fnirs'] - fnirs_signal) ** 2)),
            float(np.mean((reconstructions['shared_only_fnirs'] - fnirs_signal) ** 2)),
            float(np.mean((reconstructions['private_only_fnirs'] - fnirs_signal) ** 2)),
        ],
        color=['crimson', 'purple', 'teal'],
        alpha=0.85,
    )
    axes[1, 1].set_title(f'{split_name.upper()} fNIRS Ablation MSE')

    for axis in axes.flatten():
        axis.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _collect_snapshot_reconstructions(model, eeg: torch.Tensor, fnirs: torch.Tensor) -> Dict[str, np.ndarray]:
    full = model.reconstruct_with_component_masks(eeg, fnirs, use_shared=True, use_private=True)
    shared_only = model.reconstruct_with_component_masks(eeg, fnirs, use_shared=True, use_private=False)
    private_only = model.reconstruct_with_component_masks(eeg, fnirs, use_shared=False, use_private=True)
    return {
        'full_eeg': full['eeg_reconstructed'][0].detach().cpu().numpy(),
        'shared_only_eeg': shared_only['eeg_reconstructed'][0].detach().cpu().numpy(),
        'private_only_eeg': private_only['eeg_reconstructed'][0].detach().cpu().numpy(),
        'full_fnirs': full['fnirs_reconstructed'][0].detach().cpu().numpy(),
        'shared_only_fnirs': shared_only['fnirs_reconstructed'][0].detach().cpu().numpy(),
        'private_only_fnirs': private_only['fnirs_reconstructed'][0].detach().cpu().numpy(),
    }


def _compute_branch_ablation_metrics(
    eeg_signal: np.ndarray,
    fnirs_signal: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
) -> Dict[str, float]:
    def mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a - b) ** 2))

    eeg_full_mse = mse(reconstructions['full_eeg'], eeg_signal)
    eeg_shared_only_mse = mse(reconstructions['shared_only_eeg'], eeg_signal)
    eeg_private_only_mse = mse(reconstructions['private_only_eeg'], eeg_signal)
    fnirs_full_mse = mse(reconstructions['full_fnirs'], fnirs_signal)
    fnirs_shared_only_mse = mse(reconstructions['shared_only_fnirs'], fnirs_signal)
    fnirs_private_only_mse = mse(reconstructions['private_only_fnirs'], fnirs_signal)

    return {
        'eeg_full_mse': eeg_full_mse,
        'eeg_shared_only_mse': eeg_shared_only_mse,
        'eeg_private_only_mse': eeg_private_only_mse,
        'eeg_shared_gap': float(eeg_shared_only_mse - eeg_full_mse),
        'eeg_private_gap': float(eeg_private_only_mse - eeg_full_mse),
        'eeg_shared_to_full_ratio': float(eeg_shared_only_mse / max(eeg_full_mse, 1e-12)),
        'eeg_private_to_full_ratio': float(eeg_private_only_mse / max(eeg_full_mse, 1e-12)),
        'fnirs_full_mse': fnirs_full_mse,
        'fnirs_shared_only_mse': fnirs_shared_only_mse,
        'fnirs_private_only_mse': fnirs_private_only_mse,
        'fnirs_shared_gap': float(fnirs_shared_only_mse - fnirs_full_mse),
        'fnirs_private_gap': float(fnirs_private_only_mse - fnirs_full_mse),
        'fnirs_shared_to_full_ratio': float(fnirs_shared_only_mse / max(fnirs_full_mse, 1e-12)),
        'fnirs_private_to_full_ratio': float(fnirs_private_only_mse / max(fnirs_full_mse, 1e-12)),
    }


def _compact_codebook_summary(summary: Dict[str, object], codebook_size: int) -> Dict[str, float | int]:
    return {
        'active_codes': int(summary['active_codes']),
        'usage_rate': float(summary['usage_rate']),
        'dead_codes': int(summary['dead_codes']),
        'dead_ratio': float(summary['dead_codes']) / max(codebook_size, 1),
        'perplexity': float(summary['perplexity']),
        'perplexity_ratio': float(summary['perplexity']) / max(codebook_size, 1),
        'gini_coefficient': float(summary['gini_coefficient']),
        'top_20_coverage': float(summary['top_20_coverage']),
    }


def _save_codebook_health_dashboard(
    path: Path,
    split_name: str,
    shared_eeg_summary: Dict[str, object],
    shared_fnirs_summary: Dict[str, object],
    eeg_private_summary: Dict[str, object],
    fnirs_private_summary: Dict[str, object],
    shared_codebook_size: int,
    eeg_private_codebook_size: int,
    fnirs_private_codebook_size: int,
):
    shared_eeg = _compact_codebook_summary(shared_eeg_summary, shared_codebook_size)
    shared_fnirs = _compact_codebook_summary(shared_fnirs_summary, shared_codebook_size)
    eeg_private = _compact_codebook_summary(eeg_private_summary, eeg_private_codebook_size)
    fnirs_private = _compact_codebook_summary(fnirs_private_summary, fnirs_private_codebook_size)
    summaries = [shared_eeg, shared_fnirs, eeg_private, fnirs_private]
    labels = ['shared_eeg', 'shared_fnirs', 'eeg_private', 'fnirs_private']
    colors = ['steelblue', 'forestgreen', 'darkorange', 'purple']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].bar(labels, [item['perplexity_ratio'] for item in summaries], color=colors, alpha=0.85)
    axes[0, 0].set_title(f'{split_name.upper()} Perplexity Ratio')
    axes[0, 0].set_ylabel('perplexity / codebook_size')

    axes[0, 1].bar(labels, [item['usage_rate'] for item in summaries], color=colors, alpha=0.85)
    axes[0, 1].set_title(f'{split_name.upper()} Usage Rate')
    axes[0, 1].set_ylabel('active_codes / codebook_size')

    axes[1, 0].bar(labels, [item['dead_ratio'] for item in summaries], color=colors, alpha=0.85)
    axes[1, 0].set_title(f'{split_name.upper()} Dead Code Ratio')
    axes[1, 0].set_ylabel('dead_codes / codebook_size')

    axes[1, 1].bar(labels, [item['top_20_coverage'] for item in summaries], color=colors, alpha=0.85)
    axes[1, 1].set_title(f'{split_name.upper()} Top-20 Coverage')
    axes[1, 1].set_ylabel('coverage')

    for axis in axes.flatten():
        axis.grid(True, axis='y', alpha=0.25)
        axis.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def analyze_factorized_alignment(
    model,
    dataloaders: Dict[str, object],
    config: Dict[str, object],
    output_dir: Path,
    device: torch.device,
    splits: Iterable[str] = ('val', 'test'),
) -> Dict[str, object]:
    layout = prepare_analysis_layout(
        suite_root=output_dir,
        analysis_name='factorized_alignment',
        splits=splits,
        metadata={
            'focus': 'architecture_guidance',
            'retained_metrics': [
                'reconstruction_guardrails',
                'codebook_health',
                'branch_responsibility',
                'lag_coupling',
                'supplementary_overlap',
            ],
        },
    )
    lag_set = [int(x) for x in config.get('validation', {}).get('lag_set', [0, 1, 2, 3, 4, 5])]
    shared_codebook_size = int(getattr(model, 'shared_codebook_size', model.get_codebook_size()))
    eeg_private_codebook_size = int(getattr(model, 'eeg_private_codebook_size', shared_codebook_size))
    fnirs_private_codebook_size = int(getattr(model, 'fnirs_private_codebook_size', shared_codebook_size))
    window_duration_s = float(config.get('data', {}).get('window', {}).get('duration_s', 1.0))
    eeg_lowpass = config.get('data', {}).get('eeg_preprocessing', {}).get('lowpass')
    fnirs_lowpass = config.get('data', {}).get('fnirs_preprocessing', {}).get('lowpass')

    results: Dict[str, object] = {
        'analysis_type': 'factorized_alignment',
        'artifact_root': str(layout['analysis_root']),
        'shared_codebook_size': shared_codebook_size,
        'eeg_private_codebook_size': eeg_private_codebook_size,
        'fnirs_private_codebook_size': fnirs_private_codebook_size,
        'lag_set': lag_set,
        'splits': {},
    }

    model.eval()
    for split_name in splits:
        dataloader = dataloaders.get(split_name)
        if dataloader is None:
            continue

        scalar_totals: Dict[str, float] = {}
        eeg_shared_batches = []
        fnirs_shared_batches = []
        eeg_private_batches = []
        fnirs_private_batches = []
        eeg_signal_batches = []
        fnirs_signal_batches = []
        shared_latent_energy = []
        private_latent_energy = []
        total_batches = 0
        snapshot = None

        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            fnirs = batch['fnirs'].to(device)
            outputs = model(eeg, fnirs)
            total_batches += 1

            for key, value in outputs.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    scalar_totals[key] = scalar_totals.get(key, 0.0) + _tensor_to_float(value)

            eeg_shared_batches.append(outputs['eeg_indices'].detach().cpu().numpy())
            fnirs_shared_batches.append(outputs['fnirs_indices'].detach().cpu().numpy())
            eeg_private_batches.append(outputs['eeg_private_indices'].detach().cpu().numpy())
            fnirs_private_batches.append(outputs['fnirs_private_indices'].detach().cpu().numpy())

            shared_latent_energy.append(float(outputs['eeg_z'].pow(2).mean().item() + outputs['fnirs_z'].pow(2).mean().item()))
            private_latent_energy.append(float(outputs['eeg_private_z'].pow(2).mean().item() + outputs['fnirs_private_z'].pow(2).mean().item()))

            if sum(batch_array.shape[0] for batch_array in eeg_signal_batches) < 128:
                eeg_signal_batches.append(eeg.detach().cpu().numpy())
                fnirs_signal_batches.append(fnirs.detach().cpu().numpy())

            if snapshot is None:
                snapshot = {
                    'eeg_signal': eeg[0].detach().cpu().numpy(),
                    'fnirs_signal': fnirs[0].detach().cpu().numpy(),
                    'full_eeg_reconstruction': outputs['eeg_reconstructed'][0].detach().cpu().numpy(),
                    'full_fnirs_reconstruction': outputs['fnirs_reconstructed'][0].detach().cpu().numpy(),
                    'ablations': _collect_snapshot_reconstructions(model, eeg[:1], fnirs[:1]),
                }

        if total_batches == 0:
            continue

        mean_metrics = {key: value / total_batches for key, value in scalar_totals.items()}
        eeg_shared_tokens = np.concatenate(eeg_shared_batches, axis=0)
        fnirs_shared_tokens = np.concatenate(fnirs_shared_batches, axis=0)
        eeg_private_tokens = np.concatenate(eeg_private_batches, axis=0)
        fnirs_private_tokens = np.concatenate(fnirs_private_batches, axis=0)
        eeg_signals = np.concatenate(eeg_signal_batches, axis=0) if eeg_signal_batches else np.empty((0, 0, 0), dtype=np.float64)
        fnirs_signals = np.concatenate(fnirs_signal_batches, axis=0) if fnirs_signal_batches else np.empty((0, 0, 0), dtype=np.float64)

        shared_eeg_summary = _codebook_summary(eeg_shared_tokens.reshape(-1), shared_codebook_size)
        shared_fnirs_summary = _codebook_summary(fnirs_shared_tokens.reshape(-1), shared_codebook_size)
        eeg_private_summary = _codebook_summary(eeg_private_tokens.reshape(-1), eeg_private_codebook_size)
        fnirs_private_summary = _codebook_summary(fnirs_private_tokens.reshape(-1), fnirs_private_codebook_size)
        shared_overlap = _active_overlap_summary(shared_eeg_summary, shared_fnirs_summary)
        fixed_compare_length = min(max(eeg_shared_tokens.shape[1] - lag, 0) for lag in lag_set) if lag_set else eeg_shared_tokens.shape[1]
        lag_metrics = [
            _pair_statistics(
                eeg_shared_tokens,
                fnirs_shared_tokens,
                shared_codebook_size,
                lag,
                target_length=fixed_compare_length,
            )
            for lag in lag_set
        ]
        lag_zero = next(item for item in lag_metrics if item['lag'] == 0)
        best_lag = max(lag_metrics, key=lambda item: (item['mi_improvement'], -item['lag']))

        split_layout = layout['splits'][split_name]
        split_metrics_dir = split_layout['metrics']
        split_figures_dir = split_layout['figures']

        _save_codebook_health_dashboard(
            split_figures_dir / 'codebook_health.png',
            split_name,
            shared_eeg_summary,
            shared_fnirs_summary,
            eeg_private_summary,
            fnirs_private_summary,
            shared_codebook_size,
            eeg_private_codebook_size,
            fnirs_private_codebook_size,
        )
        _save_lag_plot(split_figures_dir / 'lag_metrics.png', lag_metrics, split_name)
        shared_coupling = _save_cross_modal_coupling_plot(
            split_figures_dir / 'cross_modal_coupling.png',
            eeg_shared_tokens,
            fnirs_shared_tokens,
            shared_codebook_size,
            split_name,
        )

        if snapshot is not None:
            branch_ablation_metrics = _compute_branch_ablation_metrics(
                snapshot['eeg_signal'],
                snapshot['fnirs_signal'],
                snapshot['ablations'],
            )
            eeg_fs = _estimate_sampling_rate(snapshot['eeg_signal'], window_duration_s)
            fnirs_fs = _estimate_sampling_rate(snapshot['fnirs_signal'], window_duration_s)
            _save_branch_ablation_plot(
                split_figures_dir / 'branch_ablation.png',
                snapshot['eeg_signal'],
                snapshot['fnirs_signal'],
                snapshot['ablations'],
                split_name,
            )
        else:
            branch_ablation_metrics = {}

        split_result = {
            'reconstruction_guardrails': {
                'eeg_rec_loss': float(mean_metrics.get('eeg_rec_loss', 0.0)),
                'fnirs_rec_loss': float(mean_metrics.get('fnirs_rec_loss', 0.0)),
                'shared_eeg_common_loss': float(mean_metrics.get('shared_eeg_common_loss', 0.0)),
                'shared_fnirs_common_loss': float(mean_metrics.get('shared_fnirs_common_loss', 0.0)),
                'eeg_private_residual_loss': float(mean_metrics.get('eeg_private_residual_loss', 0.0)),
                'fnirs_private_residual_loss': float(mean_metrics.get('fnirs_private_residual_loss', 0.0)),
            },
            'codebook_health': {
                'shared_eeg': _compact_codebook_summary(shared_eeg_summary, shared_codebook_size),
                'shared_fnirs': _compact_codebook_summary(shared_fnirs_summary, shared_codebook_size),
                'eeg_private': _compact_codebook_summary(eeg_private_summary, eeg_private_codebook_size),
                'fnirs_private': _compact_codebook_summary(fnirs_private_summary, fnirs_private_codebook_size),
            },
            'branch_responsibility': branch_ablation_metrics,
            'lag_coupling': {
                'lag_compare_length': int(fixed_compare_length),
                'lag_zero_mutual_information': float(lag_zero['mutual_information']),
                'lag_zero_mi_improvement': float(lag_zero['mi_improvement']),
                'best_lag': int(best_lag['lag']),
                'best_lag_mutual_information': float(best_lag['mutual_information']),
                'best_lag_mi_improvement': float(best_lag['mi_improvement']),
                'best_lag_match_rate': float(best_lag['match_rate']),
                'lag_metrics': lag_metrics,
                'shared_cross_modal_coupling': shared_coupling,
            },
            'supplementary_overlap': shared_overlap,
            'optimization_signature': {
                'mean_shared_latent_energy': float(np.mean(shared_latent_energy)),
                'mean_private_latent_energy': float(np.mean(private_latent_energy)),
                'selected_alignment_lag_objective': float(mean_metrics.get('selected_alignment_lag', 0.0)),
            },
        }
        results['splits'][split_name] = split_result

        write_json(split_metrics_dir / 'summary.json', split_result)

    write_json(Path(layout['metrics_root']) / 'summary.json', results)
    return results