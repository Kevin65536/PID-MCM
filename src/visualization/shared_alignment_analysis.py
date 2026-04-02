from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.io import write_json

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _tensor_to_float(value) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


def _gini(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    if counts.size == 0 or counts.sum() == 0:
        return 0.0
    counts = np.sort(counts)
    n = counts.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2 * index - n - 1) * counts) / (n * counts.sum()))


def _entropy_and_perplexity(counts: np.ndarray) -> Tuple[float, float]:
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    probs = counts / total
    mask = probs > 0
    entropy = float(-(probs[mask] * np.log(probs[mask] + 1e-12)).sum())
    return entropy, float(np.exp(entropy))


def _codebook_summary(tokens: np.ndarray, codebook_size: int) -> Dict[str, object]:
    counts = np.bincount(tokens, minlength=codebook_size)
    entropy, perplexity = _entropy_and_perplexity(counts)
    active_codes = int((counts > 0).sum())
    active_indices = np.where(counts > 0)[0]
    top_codes = np.argsort(counts)[::-1][:20]
    total = max(int(counts.sum()), 1)
    return {
        'active_codes': active_codes,
        'usage_rate': float(active_codes / max(codebook_size, 1)),
        'dead_codes': int(codebook_size - active_codes),
        'entropy': entropy,
        'perplexity': perplexity,
        'gini_coefficient': _gini(counts),
        'top_codes': top_codes.tolist(),
        'top_probs': (counts[top_codes] / total).tolist(),
        'top_20_coverage': float(counts[top_codes].sum() / total),
        'active_indices': active_indices.tolist(),
        'counts': counts,
    }


def _active_overlap_summary(eeg_summary: Dict[str, object], fnirs_summary: Dict[str, object]) -> Dict[str, object]:
    eeg_active = set(int(x) for x in eeg_summary['active_indices'])
    fnirs_active = set(int(x) for x in fnirs_summary['active_indices'])
    intersection = sorted(eeg_active & fnirs_active)
    union = sorted(eeg_active | fnirs_active)
    return {
        'intersection_count': len(intersection),
        'union_count': len(union),
        'jaccard': float(len(intersection) / max(len(union), 1)),
        'intersection_codes': intersection[:50],
    }


def _aligned_token_arrays(
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    lag: int,
    target_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if lag < 0:
        raise ValueError('Only non-negative lag is supported')
    usable = eeg_tokens.shape[1] - lag
    if target_length is not None:
        usable = min(usable, int(target_length))
    if usable <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    eeg_aligned = eeg_tokens[:, :usable].reshape(-1)
    fnirs_aligned = fnirs_tokens[:, lag:lag + usable].reshape(-1)
    return eeg_aligned.astype(np.int64), fnirs_aligned.astype(np.int64)


def _pair_statistics(
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    codebook_size: int,
    lag: int,
    target_length: Optional[int] = None,
) -> Dict[str, object]:
    if lag < 0:
        raise ValueError('Only non-negative lag is supported')
    usable = eeg_tokens.shape[1] - lag
    if target_length is not None:
        usable = min(usable, int(target_length))
    if usable <= 0:
        return {
            'lag': lag,
            'total_pairs': 0,
            'compare_length': 0,
            'match_rate': 0.0,
            'mutual_information': 0.0,
            'normalized_mi': 0.0,
            'mi_shuffle_baseline': 0.0,
            'mi_improvement': 0.0,
            'weighted_top1_concentration': 0.0,
            'mean_top1_concentration': 0.0,
            'top_mapping_rows': [],
            'top_eeg_codes': [],
            'top_fnirs_codes': [],
            'heatmap': [],
        }

    eeg_matrix = eeg_tokens[:, :usable].astype(np.int64, copy=False)
    fnirs_matrix = fnirs_tokens[:, lag:lag + usable].astype(np.int64, copy=False)
    eeg_aligned = eeg_matrix.reshape(-1)
    fnirs_aligned = fnirs_matrix.reshape(-1)
    total_pairs = int(eeg_aligned.size)

    eeg_counts = np.bincount(eeg_aligned, minlength=codebook_size)
    fnirs_counts = np.bincount(fnirs_aligned, minlength=codebook_size)
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for eeg_code, fnirs_code in zip(eeg_aligned.tolist(), fnirs_aligned.tolist()):
        pair_counts[(eeg_code, fnirs_code)] += 1

    mi = 0.0
    for (eeg_code, fnirs_code), count in pair_counts.items():
        numerator = count * total_pairs
        denominator = eeg_counts[eeg_code] * fnirs_counts[fnirs_code]
        mi += (count / total_pairs) * math.log((numerator / max(denominator, 1)) + 1e-12)

    shuffle_rng = np.random.default_rng(12345 + int(lag))
    shuffled_fnirs = fnirs_matrix[shuffle_rng.permutation(fnirs_matrix.shape[0])].reshape(-1)
    shuffled_counts = np.bincount(shuffled_fnirs, minlength=codebook_size)
    shuffled_pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for eeg_code, fnirs_code in zip(eeg_aligned.tolist(), shuffled_fnirs.tolist()):
        shuffled_pair_counts[(eeg_code, fnirs_code)] += 1

    mi_shuffle = 0.0
    for (eeg_code, fnirs_code), count in shuffled_pair_counts.items():
        numerator = count * total_pairs
        denominator = eeg_counts[eeg_code] * shuffled_counts[fnirs_code]
        mi_shuffle += (count / total_pairs) * math.log((numerator / max(denominator, 1)) + 1e-12)

    fnirs_entropy, _ = _entropy_and_perplexity(fnirs_counts)
    top_eeg_codes = np.argsort(eeg_counts)[::-1][:12]
    top_fnirs_codes = np.argsort(fnirs_counts)[::-1][:12]
    heatmap = np.zeros((len(top_eeg_codes), len(top_fnirs_codes)), dtype=np.float64)
    for i, eeg_code in enumerate(top_eeg_codes):
        for j, fnirs_code in enumerate(top_fnirs_codes):
            heatmap[i, j] = pair_counts.get((int(eeg_code), int(fnirs_code)), 0)
    if heatmap.sum() > 0:
        heatmap = heatmap / heatmap.sum()

    top_mapping_rows = []
    for eeg_code in top_eeg_codes[:10]:
        row_total = int(eeg_counts[int(eeg_code)])
        if row_total == 0:
            continue
        best_fnirs_code = max(
            range(codebook_size),
            key=lambda fnirs_code: pair_counts.get((int(eeg_code), int(fnirs_code)), 0),
        )
        best_count = pair_counts.get((int(eeg_code), int(best_fnirs_code)), 0)
        top_mapping_rows.append({
            'eeg_code': int(eeg_code),
            'best_fnirs_code': int(best_fnirs_code),
            'pair_count': int(best_count),
            'row_total': row_total,
            'concentration': float(best_count / row_total),
        })

    weighted_best_counts = 0
    for eeg_code, row_total in enumerate(eeg_counts.tolist()):
        if row_total <= 0:
            continue
        best_count = max(
            (pair_counts.get((int(eeg_code), int(fnirs_code)), 0) for fnirs_code in range(codebook_size)),
            default=0,
        )
        weighted_best_counts += best_count

    return {
        'lag': lag,
        'total_pairs': total_pairs,
        'compare_length': int(usable),
        'match_rate': float((eeg_aligned == fnirs_aligned).mean()),
        'mutual_information': float(mi),
        'normalized_mi': float(mi / fnirs_entropy) if fnirs_entropy > 0 else 0.0,
        'mi_shuffle_baseline': float(mi_shuffle),
        'mi_improvement': float(mi - mi_shuffle),
        'weighted_top1_concentration': float(weighted_best_counts / max(total_pairs, 1)),
        'mean_top1_concentration': float(np.mean([row['concentration'] for row in top_mapping_rows])) if top_mapping_rows else 0.0,
        'top_mapping_rows': top_mapping_rows,
        'top_eeg_codes': [int(x) for x in top_eeg_codes.tolist()],
        'top_fnirs_codes': [int(x) for x in top_fnirs_codes.tolist()],
        'heatmap': heatmap.tolist(),
    }


def _save_usage_plot(path: Path, eeg_summary: Dict[str, object], fnirs_summary: Dict[str, object], split_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    eeg_counts = np.sort(np.array(eeg_summary['counts']))[::-1][:50]
    fnirs_counts = np.sort(np.array(fnirs_summary['counts']))[::-1][:50]

    axes[0].bar(np.arange(len(eeg_counts)), eeg_counts, color='steelblue', alpha=0.85)
    axes[0].set_title(f'{split_name.upper()} EEG Code Usage')
    axes[0].set_xlabel('Code rank')
    axes[0].set_ylabel('Count')

    axes[1].bar(np.arange(len(fnirs_counts)), fnirs_counts, color='forestgreen', alpha=0.85)
    axes[1].set_title(f'{split_name.upper()} fNIRS Code Usage')
    axes[1].set_xlabel('Code rank')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_codebook_diagnostics(
    path: Path,
    eeg_summary: Dict[str, object],
    fnirs_summary: Dict[str, object],
    split_name: str,
    codebook_size: int,
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    modality_specs = [
        ('EEG', eeg_summary, 'steelblue', axes[0]),
        ('fNIRS', fnirs_summary, 'forestgreen', axes[1]),
    ]

    for modality_name, summary, color, row_axes in modality_specs:
        counts = np.array(summary['counts'])
        sorted_counts = np.sort(counts)[::-1]
        nonzero_counts = sorted_counts[sorted_counts > 0]
        cumulative = np.cumsum(sorted_counts) / max(sorted_counts.sum(), 1)

        row_axes[0].bar(np.arange(len(nonzero_counts[:100])), nonzero_counts[:100], color=color, alpha=0.85)
        row_axes[0].set_title(f'{split_name.upper()} {modality_name} Rank-Frequency')
        row_axes[0].set_xlabel('Code rank')
        row_axes[0].set_ylabel('Count')
        row_axes[0].set_yscale('log')

        row_axes[1].plot(np.arange(len(cumulative[:200])), cumulative[:200], color=color, linewidth=2)
        row_axes[1].axhline(0.9, color='red', linestyle='--', alpha=0.6)
        row_axes[1].axhline(0.95, color='orange', linestyle='--', alpha=0.6)
        row_axes[1].set_title(f'{split_name.upper()} {modality_name} Coverage')
        row_axes[1].set_xlabel('Number of codes')
        row_axes[1].set_ylabel('Cumulative probability')

        side = int(np.ceil(np.sqrt(codebook_size)))
        grid = np.zeros(side * side, dtype=np.float64)
        grid[:codebook_size] = np.log1p(counts)
        grid = grid.reshape(side, side)
        im = row_axes[2].imshow(grid, aspect='auto', cmap='viridis')
        row_axes[2].set_title(
            f'{modality_name}: active={summary["active_codes"]}, '
            f'usage={summary["usage_rate"] * 100:.2f}%\n'
            f'perplexity={summary["perplexity"]:.2f}, gini={summary["gini_coefficient"]:.3f}'
        )
        row_axes[2].set_xticks([])
        row_axes[2].set_yticks([])
        fig.colorbar(im, ax=row_axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_lag_plot(path: Path, lag_metrics: List[Dict[str, object]], split_name: str):
    lags = [entry['lag'] for entry in lag_metrics]
    match_rates = [entry['match_rate'] for entry in lag_metrics]
    mi_values = [entry['mutual_information'] for entry in lag_metrics]
    mi_shuffle = [entry['mi_shuffle_baseline'] for entry in lag_metrics]
    mi_improvement = [entry['mi_improvement'] for entry in lag_metrics]
    paired_concentration = [entry['weighted_top1_concentration'] for entry in lag_metrics]
    compare_lengths = [entry['compare_length'] for entry in lag_metrics]
    best_index = int(np.argmax(mi_improvement)) if mi_improvement else 0
    best_lag = lags[best_index] if lags else 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(lags, match_rates, marker='o', color='darkorange')
    axes[0].plot(lags, paired_concentration, marker='s', color='crimson', alpha=0.75)
    axes[0].set_title(f'{split_name.upper()} Match / Pairing vs Lag')
    axes[0].set_xlabel('Lag (tokens)')
    axes[0].set_ylabel('Strength')

    axes[1].plot(lags, mi_values, marker='o', color='mediumpurple', label='Raw MI')
    axes[1].plot(lags, mi_shuffle, marker='s', color='gray', linestyle='--', label='Shuffle baseline')
    axes[1].plot(lags, mi_improvement, marker='^', color='teal', label='Corrected MI')
    axes[1].axvline(best_lag, color='red', linestyle='--', alpha=0.8, label=f'Best corrected lag={best_lag}')
    axes[1].set_title(f'{split_name.upper()} Fixed-length MI vs Lag')
    axes[1].set_xlabel('Lag (tokens)')
    axes[1].set_ylabel('Mutual information')
    axes[1].legend(fontsize=8)

    axes[2].plot(lags, compare_lengths, marker='o', color='steelblue', label='Compare length')
    axes[2].plot(lags, paired_concentration, marker='s', color='crimson', label='Weighted Top-1 Pairing')
    axes[2].set_title(f'{split_name.upper()} Fixed-length Lag Stats')
    axes[2].set_xlabel('Lag (tokens)')
    axes[2].set_ylabel('Length / strength')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_heatmap(path: Path, lag_entry: Dict[str, object], split_name: str):
    heatmap = np.array(lag_entry['heatmap'])
    if heatmap.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, aspect='auto', cmap='magma')
    ax.set_title(f'{split_name.upper()} Pair Heatmap at Lag {lag_entry["lag"]}')
    ax.set_xlabel('Top fNIRS codes')
    ax.set_ylabel('Top EEG codes')
    ax.set_xticks(np.arange(len(lag_entry['top_fnirs_codes'])))
    ax.set_xticklabels(lag_entry['top_fnirs_codes'], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(lag_entry['top_eeg_codes'])))
    ax.set_yticklabels(lag_entry['top_eeg_codes'], fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _save_pairing_dashboard(
    path: Path,
    split_name: str,
    lag_zero: Dict[str, object],
    best_lag: Dict[str, object],
    overlap_summary: Dict[str, object],
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for axis, entry, title in [
        (axes[0, 0], lag_zero, f'{split_name.upper()} Lag 0 Heatmap'),
        (axes[0, 1], best_lag, f'{split_name.upper()} Best Lag {best_lag["lag"]} Heatmap'),
    ]:
        heatmap = np.array(entry['heatmap'])
        if heatmap.size > 0:
            im = axis.imshow(heatmap, aspect='auto', cmap='magma')
            axis.set_xticks(np.arange(len(entry['top_fnirs_codes'])))
            axis.set_xticklabels(entry['top_fnirs_codes'], rotation=45, ha='right', fontsize=8)
            axis.set_yticks(np.arange(len(entry['top_eeg_codes'])))
            axis.set_yticklabels(entry['top_eeg_codes'], fontsize=8)
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
        axis.set_xlabel('Top fNIRS codes')
        axis.set_ylabel('Top EEG codes')

    mapping_rows = best_lag['top_mapping_rows'][:10]
    labels = [f"{row['eeg_code']}→{row['best_fnirs_code']}" for row in mapping_rows]
    concentrations = [row['concentration'] for row in mapping_rows]
    axes[1, 0].barh(np.arange(len(labels)), concentrations, color='slateblue', alpha=0.85)
    axes[1, 0].set_yticks(np.arange(len(labels)))
    axes[1, 0].set_yticklabels(labels, fontsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel('Top-1 concentration')
    axes[1, 0].set_title(f'{split_name.upper()} Best-Lag Pair Mapping')

    axes[1, 1].axis('off')
    summary_lines = [
        f"Active-set intersection: {overlap_summary['intersection_count']}",
        f"Active-set union: {overlap_summary['union_count']}",
        f"Active-set Jaccard: {overlap_summary['jaccard']:.4f}",
        f"Lag 0 exact match: {lag_zero['match_rate']:.4f}",
        f"Lag 0 weighted top-1: {lag_zero['weighted_top1_concentration']:.4f}",
        f"Best lag: {best_lag['lag']}",
        f"Best lag MI: {best_lag['mutual_information']:.4f}",
        f"Best lag weighted top-1: {best_lag['weighted_top1_concentration']:.4f}",
        '',
        'Interpretation:',
        'If active-set intersection is ~0, exact token identity is structurally impossible.',
        'Non-zero weighted top-1 concentration indicates stable paired-code mapping even when match rate stays 0.',
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        '\n'.join(summary_lines),
        transform=axes[1, 1].transAxes,
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _select_top_variance_channels(signal: np.ndarray, max_channels: int = 3) -> List[int]:
    if signal.ndim != 2:
        return [0]
    variances = np.var(signal, axis=1)
    sorted_indices = np.argsort(variances)[::-1]
    count = min(max_channels, signal.shape[0])
    return [int(index) for index in sorted_indices[:count]]


def _save_reconstruction_plot(
    path: Path,
    eeg_signal: np.ndarray,
    eeg_reconstruction: np.ndarray,
    fnirs_signal: np.ndarray,
    fnirs_reconstruction: np.ndarray,
    split_name: str,
):
    eeg_time = np.arange(eeg_signal.shape[1])
    fnirs_time = np.arange(fnirs_signal.shape[1])
    eeg_channels = _select_top_variance_channels(eeg_signal, max_channels=3)
    fnirs_channels = _select_top_variance_channels(fnirs_signal, max_channels=3)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    axes[0, 0].plot(eeg_time, eeg_signal.mean(axis=0), color='steelblue', linewidth=2, label='Original')
    axes[0, 0].plot(eeg_time, eeg_reconstruction.mean(axis=0), color='darkorange', linewidth=2, alpha=0.85, label='Reconstructed')
    axes[0, 0].set_title(f'{split_name.upper()} EEG Mean Reconstruction')
    axes[0, 0].set_xlabel('Time samples')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()

    for channel in eeg_channels:
        axes[0, 1].plot(eeg_time, eeg_signal[channel], linewidth=1.4, alpha=0.65, label=f'Ch {channel} original')
        axes[0, 1].plot(eeg_time, eeg_reconstruction[channel], linewidth=1.2, linestyle='--', alpha=0.85, label=f'Ch {channel} recon')
    axes[0, 1].set_title(f'{split_name.upper()} EEG Representative Channels')
    axes[0, 1].set_xlabel('Time samples')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend(fontsize=8, ncol=2)

    axes[1, 0].plot(fnirs_time, fnirs_signal.mean(axis=0), color='forestgreen', linewidth=2, label='Original')
    axes[1, 0].plot(fnirs_time, fnirs_reconstruction.mean(axis=0), color='crimson', linewidth=2, alpha=0.85, label='Reconstructed')
    axes[1, 0].set_title(f'{split_name.upper()} fNIRS Mean Reconstruction')
    axes[1, 0].set_xlabel('Time samples')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()

    for channel in fnirs_channels:
        axes[1, 1].plot(fnirs_time, fnirs_signal[channel], linewidth=1.4, alpha=0.65, label=f'Ch {channel} original')
        axes[1, 1].plot(fnirs_time, fnirs_reconstruction[channel], linewidth=1.2, linestyle='--', alpha=0.85, label=f'Ch {channel} recon')
    axes[1, 1].set_title(f'{split_name.upper()} fNIRS Representative Channels')
    axes[1, 1].set_xlabel('Time samples')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _estimate_sampling_rate(signal: np.ndarray, window_duration_s: float) -> float:
    if window_duration_s <= 0:
        return 1.0
    return float(signal.shape[-1] / window_duration_s)


def _average_psd(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if signal.ndim == 1:
        signal = signal[None, :]

    try:
        from scipy.signal import welch

        psds = []
        for channel_signal in signal:
            freqs, psd = welch(channel_signal, fs=fs, nperseg=min(256, channel_signal.shape[-1]))
            psds.append(psd)
        psd_array = np.stack(psds, axis=0)
        return freqs, psd_array.mean(axis=0), psd_array.std(axis=0)
    except ImportError:
        fft = np.fft.rfft(signal, axis=-1)
        psd_array = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(signal.shape[-1], d=1.0 / max(fs, 1e-6))
        return freqs, psd_array.mean(axis=0), psd_array.std(axis=0)


def _plot_single_spectral_row(
    axes,
    original: np.ndarray,
    reconstructed: np.ndarray,
    fs: float,
    title_prefix: str,
    max_freq: Optional[float] = None,
):
    freqs, psd_orig, std_orig = _average_psd(original, fs)
    _, psd_recon, std_recon = _average_psd(reconstructed, fs)

    if max_freq is not None:
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        psd_orig = psd_orig[freq_mask]
        std_orig = std_orig[freq_mask]
        psd_recon = psd_recon[freq_mask]
        std_recon = std_recon[freq_mask]

    axes[0].semilogy(freqs, psd_orig, color='steelblue', linewidth=2, label='Original')
    axes[0].fill_between(freqs, np.maximum(psd_orig - std_orig, 1e-12), psd_orig + std_orig, color='steelblue', alpha=0.2)
    axes[0].semilogy(freqs, psd_recon, color='darkorange', linewidth=2, linestyle='--', label='Reconstructed')
    axes[0].fill_between(freqs, np.maximum(psd_recon - std_recon, 1e-12), psd_recon + std_recon, color='darkorange', alpha=0.2)
    axes[0].set_title(f'{title_prefix} PSD')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power Spectral Density')
    axes[0].legend()

    ratio = (psd_recon + 1e-10) / (psd_orig + 1e-10)
    axes[1].plot(freqs, ratio, color='purple', linewidth=2)
    axes[1].axhline(1.0, color='black', linestyle='--', linewidth=1)
    axes[1].fill_between(freqs, 0.8, 1.2, color='mediumseagreen', alpha=0.2)
    axes[1].set_title(f'{title_prefix} PSD Ratio')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Recon / Orig')
    axes[1].set_ylim(0.0, 2.0)

    spectral_error = np.abs(psd_orig - psd_recon)
    axes[2].semilogy(freqs, spectral_error, color='crimson', linewidth=2)
    axes[2].axhline(np.mean(spectral_error), color='gray', linestyle='--', linewidth=1)
    axes[2].set_title(f'{title_prefix} Spectral Error')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('|PSD Error|')

    for axis in axes:
        axis.grid(True, alpha=0.3)


def _save_spectral_comparison_plot(
    path: Path,
    eeg_signal: np.ndarray,
    eeg_reconstruction: np.ndarray,
    fnirs_signal: np.ndarray,
    fnirs_reconstruction: np.ndarray,
    split_name: str,
    eeg_fs: float,
    fnirs_fs: float,
    eeg_max_freq: Optional[float] = None,
    fnirs_max_freq: Optional[float] = None,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _plot_single_spectral_row(axes[0], eeg_signal, eeg_reconstruction, eeg_fs, f'{split_name.upper()} EEG', eeg_max_freq)
    _plot_single_spectral_row(axes[1], fnirs_signal, fnirs_reconstruction, fnirs_fs, f'{split_name.upper()} fNIRS', fnirs_max_freq)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _reduce_embeddings(embeddings: np.ndarray) -> Tuple[np.ndarray, str]:
    if embeddings.shape[0] <= 2:
        padded = np.zeros((embeddings.shape[0], 2), dtype=np.float64)
        padded[:, :min(embeddings.shape[1], 2)] = embeddings[:, :min(embeddings.shape[1], 2)]
        return padded, 'raw'

    if HAS_SKLEARN and 90 < embeddings.shape[0] <= 1000:
        perplexity = min(30, max(5, (embeddings.shape[0] - 1) // 3))
        reduced = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)
        return reduced, 'tsne'

    if HAS_SKLEARN:
        reduced = PCA(n_components=2).fit_transform(embeddings)
        return reduced, 'pca'

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    reduced = centered @ vh[:2].T
    return reduced, 'pca'


def _save_token_embeddings_plot(
    path: Path,
    embeddings: np.ndarray,
    eeg_counts: np.ndarray,
    fnirs_counts: np.ndarray,
    split_name: str,
):
    reduced, method = _reduce_embeddings(embeddings)
    combined_counts = eeg_counts + fnirs_counts
    active_mask = combined_counts > 0
    shared_mask = (eeg_counts > 0) & (fnirs_counts > 0)
    eeg_only_mask = (eeg_counts > 0) & (fnirs_counts == 0)
    fnirs_only_mask = (eeg_counts == 0) & (fnirs_counts > 0)
    inactive_mask = combined_counts == 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    scatter = axes[0].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=np.log1p(combined_counts),
        cmap='viridis',
        s=18 + 48 * (active_mask.astype(np.float32)),
        alpha=0.8,
    )
    axes[0].set_title(f'{split_name.upper()} Shared Codebook Usage ({method.upper()})')
    axes[0].set_xlabel(f'{method.upper()} 1')
    axes[0].set_ylabel(f'{method.upper()} 2')
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label='log(1 + combined usage)')

    axes[1].scatter(reduced[inactive_mask, 0], reduced[inactive_mask, 1], color='lightgray', s=20, alpha=0.5, label='Inactive')
    axes[1].scatter(reduced[eeg_only_mask, 0], reduced[eeg_only_mask, 1], color='steelblue', s=28, alpha=0.85, label='EEG only')
    axes[1].scatter(reduced[fnirs_only_mask, 0], reduced[fnirs_only_mask, 1], color='forestgreen', s=28, alpha=0.85, label='fNIRS only')
    axes[1].scatter(reduced[shared_mask, 0], reduced[shared_mask, 1], color='purple', s=34, alpha=0.9, label='Shared active')
    axes[1].set_title(f'{split_name.upper()} Modality Support')
    axes[1].set_xlabel(f'{method.upper()} 1')
    axes[1].set_ylabel(f'{method.upper()} 2')
    axes[1].legend(loc='best', fontsize=9)

    top_codes = np.argsort(combined_counts)[::-1][:12]
    axes[2].scatter(reduced[:, 0], reduced[:, 1], color='lightgray', s=16, alpha=0.35)
    axes[2].scatter(reduced[top_codes, 0], reduced[top_codes, 1], color='darkorange', s=42, alpha=0.95)
    for code in top_codes:
        if combined_counts[code] <= 0:
            continue
        label = f'{int(code)}\nE{int(eeg_counts[code])}/F{int(fnirs_counts[code])}'
        axes[2].text(reduced[code, 0], reduced[code, 1], label, fontsize=8, ha='left', va='bottom')
    axes[2].set_title(f'{split_name.upper()} Top Active Codes')
    axes[2].set_xlabel(f'{method.upper()} 1')
    axes[2].set_ylabel(f'{method.upper()} 2')

    for axis in axes:
        axis.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _collect_code_patches(
    signals: np.ndarray,
    tokens: np.ndarray,
    code: int,
    patch_size: int,
    max_patches: int = 256,
) -> np.ndarray:
    patches = []
    for sample_index in range(min(signals.shape[0], tokens.shape[0])):
        code_positions = np.where(tokens[sample_index] == code)[0]
        for position in code_positions.tolist():
            start = position * patch_size
            end = start + patch_size
            patch = signals[sample_index, :, start:end].mean(axis=0)
            patches.append(patch)
            if len(patches) >= max_patches:
                return np.stack(patches, axis=0)
    if not patches:
        return np.empty((0, patch_size), dtype=np.float64)
    return np.stack(patches, axis=0)


def _save_token_pattern_plot(
    path: Path,
    tokens: np.ndarray,
    signals: np.ndarray,
    codebook_size: int,
    patch_size: int,
    modality_name: str,
    fs: float,
    max_freq: Optional[float] = None,
):
    counts = np.bincount(tokens.reshape(-1), minlength=codebook_size)
    top_codes = np.argsort(counts)[::-1][:8]
    fig, axes = plt.subplots(len(top_codes), 3, figsize=(15, max(2, len(top_codes)) * 2.5))
    if len(top_codes) == 1:
        axes = axes[None, :]

    time = np.arange(patch_size) / max(fs, 1e-6)
    for row_index, code in enumerate(top_codes.tolist()):
        patches = _collect_code_patches(signals, tokens, int(code), patch_size)
        if patches.shape[0] == 0:
            continue

        mean_patch = patches.mean(axis=0)
        std_patch = patches.std(axis=0)
        axes[row_index, 0].plot(time, mean_patch, color='steelblue', linewidth=2)
        axes[row_index, 0].fill_between(time, mean_patch - std_patch, mean_patch + std_patch, alpha=0.25, color='steelblue')
        axes[row_index, 0].set_title(f'Code {int(code)} (n={patches.shape[0]})')
        axes[row_index, 0].set_xlabel('Time (s)')
        axes[row_index, 0].set_ylabel('Amplitude')

        freqs = np.fft.rfftfreq(patch_size, d=1.0 / max(fs, 1e-6))
        spectra = np.abs(np.fft.rfft(patches, axis=1)) ** 2
        mean_spectrum = spectra.mean(axis=0)
        if max_freq is not None:
            freq_mask = freqs <= max_freq
            freqs = freqs[freq_mask]
            mean_spectrum = mean_spectrum[freq_mask]
        axes[row_index, 1].semilogy(freqs, mean_spectrum, color='forestgreen', linewidth=2)
        axes[row_index, 1].set_title('Average Spectrum')
        axes[row_index, 1].set_xlabel('Frequency (Hz)')
        axes[row_index, 1].set_ylabel('Power')

        for patch in patches[:20]:
            axes[row_index, 2].plot(time, patch, alpha=0.2, linewidth=0.8, color='gray')
        axes[row_index, 2].plot(time, mean_patch, color='crimson', linewidth=2)
        axes[row_index, 2].set_title('Patch Overlay')
        axes[row_index, 2].set_xlabel('Time (s)')
        axes[row_index, 2].set_ylabel('Amplitude')

        for axis in axes[row_index]:
            axis.grid(True, alpha=0.25)

    fig.suptitle(f'{modality_name} Token Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _pair_count_matrix(eeg_tokens: np.ndarray, fnirs_tokens: np.ndarray, codebook_size: int, lag: int = 0) -> np.ndarray:
    usable = min(eeg_tokens.shape[1], fnirs_tokens.shape[1] - lag)
    if usable <= 0:
        return np.zeros((codebook_size, codebook_size), dtype=np.float64)
    eeg_aligned = eeg_tokens[:, :usable].reshape(-1)
    fnirs_aligned = fnirs_tokens[:, lag:lag + usable].reshape(-1)
    pair_ids = eeg_aligned.astype(np.int64) * codebook_size + fnirs_aligned.astype(np.int64)
    counts = np.bincount(pair_ids, minlength=codebook_size * codebook_size)
    return counts.reshape(codebook_size, codebook_size).astype(np.float64)


def _mutual_information_from_counts(cooccurrence: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    total = cooccurrence.sum()
    if total <= 0:
        zeros = np.zeros(cooccurrence.shape[0], dtype=np.float64)
        return 0.0, 0.0, zeros, zeros

    joint_prob = cooccurrence / total
    p_eeg = joint_prob.sum(axis=1)
    p_fnirs = joint_prob.sum(axis=0)
    nonzero = joint_prob > 0
    h_joint = -np.sum(joint_prob[nonzero] * np.log(joint_prob[nonzero] + 1e-10))
    mask_eeg = p_eeg > 0
    h_eeg = -np.sum(p_eeg[mask_eeg] * np.log(p_eeg[mask_eeg] + 1e-10))
    mask_fnirs = p_fnirs > 0
    h_fnirs = -np.sum(p_fnirs[mask_fnirs] * np.log(p_fnirs[mask_fnirs] + 1e-10))
    mi = h_fnirs - (h_joint - h_eeg)
    normalized_mi = mi / h_fnirs if h_fnirs > 0 else 0.0
    return float(mi), float(normalized_mi), p_eeg, p_fnirs


def _save_cross_modal_coupling_plot(
    path: Path,
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    codebook_size: int,
    split_name: str,
) -> Dict[str, float]:
    cooccurrence = _pair_count_matrix(eeg_tokens, fnirs_tokens, codebook_size, lag=0)
    mutual_info, normalized_mi, p_eeg, p_fnirs = _mutual_information_from_counts(cooccurrence)

    shuffle_idx = np.random.permutation(eeg_tokens.shape[0])
    shuffled_counts = _pair_count_matrix(eeg_tokens, fnirs_tokens[shuffle_idx], codebook_size, lag=0)
    mi_shuffle, _, _, _ = _mutual_information_from_counts(shuffled_counts)

    top_eeg_codes = np.argsort(p_eeg)[::-1][:10]
    conditional_dists = {}
    kl_divergences = {}
    for eeg_code in top_eeg_codes.tolist():
        row = cooccurrence[eeg_code]
        if row.sum() <= 0:
            continue
        cond_prob = row / row.sum()
        conditional_dists[int(eeg_code)] = cond_prob
        kl = 0.0
        for fnirs_code in range(codebook_size):
            if cond_prob[fnirs_code] > 0 and p_fnirs[fnirs_code] > 0:
                kl += cond_prob[fnirs_code] * np.log(cond_prob[fnirs_code] / p_fnirs[fnirs_code])
        kl_divergences[int(eeg_code)] = float(kl)

    max_display = 100
    step = max(1, codebook_size // max_display)
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(2, 3, 1)
    im = ax1.imshow(np.log1p(cooccurrence[::step, ::step]), cmap='hot', aspect='auto')
    fig.colorbar(im, ax=ax1)
    ax1.set_title(f'{split_name.upper()} Co-occurrence Matrix')
    ax1.set_xlabel('fNIRS Token')
    ax1.set_ylabel('EEG Token')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(np.arange(min(100, p_eeg.shape[0])), np.sort(p_eeg)[::-1][:100], color='steelblue', alpha=0.75)
    ax2.set_title('Marginal Distribution P(z_EEG)')
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Probability')
    ax2.set_yscale('log')

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(np.arange(min(100, p_fnirs.shape[0])), np.sort(p_fnirs)[::-1][:100], color='forestgreen', alpha=0.75)
    ax3.set_title('Marginal Distribution P(z_fNIRS)')
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Probability')
    ax3.set_yscale('log')

    ax4 = fig.add_subplot(2, 3, 4)
    for eeg_code in list(conditional_dists.keys())[:5]:
        cond = conditional_dists[eeg_code]
        top_fnirs = np.argsort(cond)[::-1][:20]
        ax4.plot(np.arange(top_fnirs.shape[0]), cond[top_fnirs], marker='o', alpha=0.7, label=f'EEG={eeg_code}, KL={kl_divergences[eeg_code]:.2f}')
    if p_fnirs.size > 0:
        ax4.axhline(float(p_fnirs.max()), color='black', linestyle='--', linewidth=1)
    ax4.set_title('Conditional Distributions for Top EEG Codes')
    ax4.set_xlabel('fNIRS Token Rank')
    ax4.set_ylabel('P(fNIRS | EEG)')
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(2, 3, 5)
    kl_values = list(kl_divergences.values()) or [0.0]
    ax5.hist(kl_values, bins=min(20, max(5, len(kl_values))), color='mediumpurple', alpha=0.8)
    ax5.axvline(np.mean(kl_values), color='red', linestyle='--', linewidth=1)
    ax5.set_title('KL(P(fNIRS|EEG) || P(fNIRS))')
    ax5.set_xlabel('KL Divergence')
    ax5.set_ylabel('Count')

    ax6 = fig.add_subplot(2, 3, 6)
    bars = ax6.bar(['Real', 'Shuffled'], [mutual_info, mi_shuffle], color=['forestgreen', 'lightcoral'])
    ax6.set_title('EEG-fNIRS Coupling Strength')
    ax6.set_ylabel('Mutual Information (nats)')
    ax6.bar_label(bars, fmt='%.3f')

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return {
        'mutual_information': float(mutual_info),
        'normalized_mi': float(normalized_mi),
        'mi_shuffle_baseline': float(mi_shuffle),
        'mi_improvement': float(mutual_info - mi_shuffle),
    }


def _save_probe_style_lag_plot(path: Path, lag_metrics: List[Dict[str, object]], split_name: str):
    lags = [int(entry['lag']) for entry in lag_metrics]
    mi_values = [float(entry['mi_improvement']) for entry in lag_metrics]
    best_index = int(np.argmax(mi_values)) if mi_values else 0
    best_lag = lags[best_index] if lags else 0
    compare_length = int(lag_metrics[0].get('compare_length', 0)) if lag_metrics else 0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, mi_values, marker='o', color='slateblue')
    ax.axvline(best_lag, color='red', linestyle='--', label=f'Best corrected lag={best_lag}')
    ax.set_xlabel('Lag (tokens)')
    ax.set_ylabel('Corrected Mutual Information (nats)')
    ax.set_title(f'{split_name.upper()} Fixed-length EEG-fNIRS Token Coupling (L={compare_length})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def analyze_shared_alignment(
    model,
    dataloaders: Dict[str, object],
    config: Dict[str, object],
    output_dir: Path,
    device: torch.device,
    splits: Iterable[str] = ('val', 'test'),
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lag_set = [int(x) for x in config.get('validation', {}).get('lag_set', [0, 1, 2, 3, 4, 5])]
    codebook_size = int(getattr(model, 'codebook_size', model.get_codebook_size()))
    window_duration_s = float(config.get('data', {}).get('window', {}).get('duration_s', 1.0))
    eeg_lowpass = config.get('data', {}).get('eeg_preprocessing', {}).get('lowpass')
    fnirs_lowpass = config.get('data', {}).get('fnirs_preprocessing', {}).get('lowpass')

    results: Dict[str, object] = {
        'analysis_type': 'shared_alignment',
        'codebook_size': codebook_size,
        'lag_set': lag_set,
        'splits': {},
    }

    model.eval()
    for split_name in splits:
        dataloader = dataloaders.get(split_name)
        if dataloader is None:
            continue

        scalar_totals: Dict[str, float] = {}
        eeg_batches = []
        fnirs_batches = []
        eeg_signal_batches = []
        fnirs_signal_batches = []
        total_batches = 0
        reconstruction_snapshot = None

        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            fnirs = batch['fnirs'].to(device)
            outputs = model(eeg, fnirs)
            total_batches += 1

            for key, value in outputs.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    scalar_totals[key] = scalar_totals.get(key, 0.0) + _tensor_to_float(value)

            eeg_batches.append(outputs['eeg_indices'].detach().cpu().numpy())
            fnirs_batches.append(outputs['fnirs_indices'].detach().cpu().numpy())
            if sum(batch_array.shape[0] for batch_array in eeg_signal_batches) < 128:
                eeg_signal_batches.append(eeg.detach().cpu().numpy())
                fnirs_signal_batches.append(fnirs.detach().cpu().numpy())

            if reconstruction_snapshot is None:
                reconstruction_snapshot = {
                    'eeg_signal': eeg[0].detach().cpu().numpy(),
                    'eeg_reconstruction': outputs['eeg_reconstructed'][0].detach().cpu().numpy(),
                    'fnirs_signal': fnirs[0].detach().cpu().numpy(),
                    'fnirs_reconstruction': outputs['fnirs_reconstructed'][0].detach().cpu().numpy(),
                }

        if total_batches == 0:
            continue

        mean_metrics = {key: value / total_batches for key, value in scalar_totals.items()}
        eeg_tokens = np.concatenate(eeg_batches, axis=0)
        fnirs_tokens = np.concatenate(fnirs_batches, axis=0)
        eeg_signals = np.concatenate(eeg_signal_batches, axis=0) if eeg_signal_batches else np.empty((0, 0, 0), dtype=np.float64)
        fnirs_signals = np.concatenate(fnirs_signal_batches, axis=0) if fnirs_signal_batches else np.empty((0, 0, 0), dtype=np.float64)
        eeg_summary = _codebook_summary(eeg_tokens.reshape(-1), codebook_size)
        fnirs_summary = _codebook_summary(fnirs_tokens.reshape(-1), codebook_size)
        overlap_summary = _active_overlap_summary(eeg_summary, fnirs_summary)
        fixed_compare_length = min(max(eeg_tokens.shape[1] - lag, 0) for lag in lag_set) if lag_set else eeg_tokens.shape[1]
        lag_metrics = [
            _pair_statistics(eeg_tokens, fnirs_tokens, codebook_size, lag, target_length=fixed_compare_length)
            for lag in lag_set
        ]
        lag_zero = next(item for item in lag_metrics if item['lag'] == 0)
        best_lag = max(lag_metrics, key=lambda item: (item['mi_improvement'], -item['lag']))
        codebook_embeddings = model.quantizer.weight.detach().cpu().numpy()

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        _save_usage_plot(split_dir / 'codebook_usage.png', eeg_summary, fnirs_summary, split_name)
        _save_codebook_diagnostics(split_dir / 'codebook_diagnostics.png', eeg_summary, fnirs_summary, split_name, codebook_size)
        _save_lag_plot(split_dir / 'lag_metrics.png', lag_metrics, split_name)
        _save_heatmap(split_dir / 'top_pair_heatmap.png', best_lag, split_name)
        _save_pairing_dashboard(split_dir / 'pairing_diagnostics.png', split_name, lag_zero, best_lag, overlap_summary)
        coupling_summary = _save_cross_modal_coupling_plot(
            split_dir / 'exp3_cross_modal_coupling.png',
            eeg_tokens,
            fnirs_tokens,
            codebook_size,
            split_name,
        )
        _save_probe_style_lag_plot(split_dir / 'exp3_lagged_coupling.png', lag_metrics, split_name)
        if reconstruction_snapshot is not None:
            eeg_fs = _estimate_sampling_rate(reconstruction_snapshot['eeg_signal'], window_duration_s)
            fnirs_fs = _estimate_sampling_rate(reconstruction_snapshot['fnirs_signal'], window_duration_s)
            _save_reconstruction_plot(
                split_dir / 'reconstruction_examples.png',
                reconstruction_snapshot['eeg_signal'],
                reconstruction_snapshot['eeg_reconstruction'],
                reconstruction_snapshot['fnirs_signal'],
                reconstruction_snapshot['fnirs_reconstruction'],
                split_name,
            )
            _save_spectral_comparison_plot(
                split_dir / 'spectral_comparison.png',
                reconstruction_snapshot['eeg_signal'],
                reconstruction_snapshot['eeg_reconstruction'],
                reconstruction_snapshot['fnirs_signal'],
                reconstruction_snapshot['fnirs_reconstruction'],
                split_name,
                eeg_fs=eeg_fs,
                fnirs_fs=fnirs_fs,
                eeg_max_freq=float(eeg_lowpass) if eeg_lowpass is not None else None,
                fnirs_max_freq=float(fnirs_lowpass) if fnirs_lowpass is not None else None,
            )
            if eeg_signals.size > 0:
                _save_token_pattern_plot(
                    split_dir / 'exp2_token_patterns_EEG.png',
                    eeg_tokens[:eeg_signals.shape[0]],
                    eeg_signals,
                    codebook_size,
                    patch_size=int(model.eeg_patch_size),
                    modality_name='EEG',
                    fs=eeg_fs,
                    max_freq=float(eeg_lowpass) if eeg_lowpass is not None else 50.0,
                )
            if fnirs_signals.size > 0:
                _save_token_pattern_plot(
                    split_dir / 'exp2_token_patterns_fNIRS.png',
                    fnirs_tokens[:fnirs_signals.shape[0]],
                    fnirs_signals,
                    codebook_size,
                    patch_size=int(model.fnirs_patch_size),
                    modality_name='fNIRS',
                    fs=fnirs_fs,
                    max_freq=float(fnirs_lowpass) if fnirs_lowpass is not None else None,
                )
        _save_token_embeddings_plot(
            split_dir / 'token_embeddings.png',
            codebook_embeddings,
            np.array(eeg_summary['counts']),
            np.array(fnirs_summary['counts']),
            split_name,
        )

        split_result = {
            'mean_metrics': mean_metrics,
            'eeg_codebook': {k: v for k, v in eeg_summary.items() if k != 'counts'},
            'fnirs_codebook': {k: v for k, v in fnirs_summary.items() if k != 'counts'},
            'active_overlap': overlap_summary,
            'lag_metrics': lag_metrics,
            'lag_compare_length': int(fixed_compare_length),
            'best_lag': int(best_lag['lag']),
            'best_lag_mutual_information': float(best_lag['mutual_information']),
            'best_lag_mi_improvement': float(best_lag['mi_improvement']),
            'best_lag_match_rate': float(best_lag['match_rate']),
            'lag0_weighted_top1_concentration': float(lag_zero['weighted_top1_concentration']),
            'best_lag_weighted_top1_concentration': float(best_lag['weighted_top1_concentration']),
            'cross_modal_coupling': coupling_summary,
            'top_pair_mapping': best_lag['top_mapping_rows'],
        }
        results['splits'][split_name] = split_result

        write_json(split_dir / 'summary.json', split_result)

    write_json(output_dir / 'shared_alignment_analysis.json', results)
    return results