from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


SCORECARD_SCHEMA_VERSION = 'phase1_gate_scorecard_v1'


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _save_figure(figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(figure)
    return str(output_path)


def _to_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((prediction - target) ** 2).detach().item())


def _maybe_batch_vector(batch: Dict[str, object], keys: Iterable[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key not in batch:
            continue
        value = batch[key]
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.ndim == 0:
            continue
        if array.dtype.kind in {'U', 'S', 'O'}:
            _, encoded = np.unique(array.astype(str), return_inverse=True)
            return encoded.astype(np.int64, copy=False)
        if array.dtype.kind == 'f':
            return np.nan_to_num(array, nan=-1.0).astype(np.int64, copy=False)
        return array.astype(np.int64, copy=False)
    return None


def _probe_id_array(values: Optional[np.ndarray], take: int) -> np.ndarray:
    if values is None:
        return np.full((take,), -1, dtype=np.int64)
    return values[:take].astype(np.int64, copy=False)


def _source_hist_features(eeg_tokens: np.ndarray, fnirs_tokens: np.ndarray, codebook_size: int) -> np.ndarray:
    batch_size = int(eeg_tokens.shape[0])
    features = np.zeros((batch_size, codebook_size * 2), dtype=np.float64)
    for row_index in range(batch_size):
        eeg_counts = np.bincount(eeg_tokens[row_index], minlength=codebook_size).astype(np.float64)
        fnirs_counts = np.bincount(fnirs_tokens[row_index], minlength=codebook_size).astype(np.float64)
        features[row_index, :codebook_size] = eeg_counts / max(eeg_counts.sum(), 1.0)
        features[row_index, codebook_size:] = fnirs_counts / max(fnirs_counts.sum(), 1.0)
    return features


def _observation_hist_features(
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    eeg_codebook_size: int,
    fnirs_codebook_size: int,
) -> np.ndarray:
    batch_size = int(eeg_tokens.shape[0])
    features = np.zeros((batch_size, eeg_codebook_size + fnirs_codebook_size), dtype=np.float64)
    for row_index in range(batch_size):
        eeg_counts = np.bincount(eeg_tokens[row_index], minlength=eeg_codebook_size).astype(np.float64)
        fnirs_counts = np.bincount(fnirs_tokens[row_index], minlength=fnirs_codebook_size).astype(np.float64)
        features[row_index, :eeg_codebook_size] = eeg_counts / max(eeg_counts.sum(), 1.0)
        features[row_index, eeg_codebook_size:] = fnirs_counts / max(fnirs_counts.sum(), 1.0)
    return features


def _stratified_split_indices(labels: np.ndarray, seed: int, train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = labels >= 0
    candidate_indices = np.where(valid_mask)[0]
    if candidate_indices.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty

    filtered_labels = labels[candidate_indices]
    classes, counts = np.unique(filtered_labels, return_counts=True)
    valid_classes = classes[counts >= 2]
    if valid_classes.size < 2:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty

    rng = np.random.default_rng(int(seed))
    train_indices: List[np.ndarray] = []
    test_indices: List[np.ndarray] = []
    retained_classes: List[int] = []
    for cls in valid_classes.tolist():
        cls_indices = candidate_indices[filtered_labels == cls]
        permuted = rng.permutation(cls_indices)
        split = min(max(int(round(permuted.size * train_ratio)), 1), permuted.size - 1)
        train_indices.append(permuted[:split])
        test_indices.append(permuted[split:])
        retained_classes.append(int(cls))

    return (
        np.concatenate(train_indices, axis=0) if train_indices else np.empty((0,), dtype=np.int64),
        np.concatenate(test_indices, axis=0) if test_indices else np.empty((0,), dtype=np.int64),
        np.array(retained_classes, dtype=np.int64),
    )


def _nearest_centroid_probe(features: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, object]:
    if features.size == 0 or labels.size == 0:
        return {'available': False, 'reason': 'empty_features'}
    train_idx, test_idx, classes = _stratified_split_indices(labels, seed=seed)
    if train_idx.size == 0 or test_idx.size == 0 or classes.size < 2:
        return {'available': False, 'reason': 'insufficient_label_support'}

    train_features = features[train_idx]
    train_labels = labels[train_idx]
    test_features = features[test_idx]
    test_labels = labels[test_idx]
    centroids = np.stack([train_features[train_labels == cls].mean(axis=0) for cls in classes], axis=0)
    distances = ((test_features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    predictions = classes[np.argmin(distances, axis=1)]

    accuracy = float(np.mean(predictions == test_labels))
    per_class_accuracy = []
    for cls in classes.tolist():
        mask = test_labels == cls
        if mask.any():
            per_class_accuracy.append(float(np.mean(predictions[mask] == test_labels[mask])))
    balanced_accuracy = float(np.mean(per_class_accuracy)) if per_class_accuracy else accuracy
    chance = float(1.0 / max(classes.size, 1))
    normalized_lift = float(max(accuracy - chance, 0.0) / max(1.0 - chance, 1e-12))

    return {
        'available': True,
        'num_classes': int(classes.size),
        'train_samples': int(train_idx.size),
        'test_samples': int(test_idx.size),
        'chance_accuracy': chance,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'normalized_lift': normalized_lift,
    }


def _codebook_summary(indices_chunks: List[np.ndarray], codebook_size: int) -> Dict[str, object]:
    if not indices_chunks:
        return {
            'codebook_size': int(codebook_size),
            'active_codes': 0,
            'active_code_ratio': 0.0,
            'dead_code_count': int(codebook_size),
            'perplexity': 0.0,
            'top_5_coverage': 0.0,
            'gini': 0.0,
            'usage_counts': [0 for _ in range(int(codebook_size))],
            'active_code_ids': [],
            'dead_code_ids': list(range(int(codebook_size))),
            'most_used_code': None,
            'max_usage': 0,
            'median_active_usage': 0.0,
            'passes_thresholds': False,
        }

    flat_indices = np.concatenate([chunk.reshape(-1) for chunk in indices_chunks], axis=0)
    if flat_indices.size == 0:
        return {
            'codebook_size': int(codebook_size),
            'active_codes': 0,
            'active_code_ratio': 0.0,
            'dead_code_count': int(codebook_size),
            'perplexity': 0.0,
            'top_5_coverage': 0.0,
            'gini': 0.0,
            'usage_counts': [0 for _ in range(int(codebook_size))],
            'active_code_ids': [],
            'dead_code_ids': list(range(int(codebook_size))),
            'most_used_code': None,
            'max_usage': 0,
            'median_active_usage': 0.0,
            'passes_thresholds': False,
        }

    counts = np.bincount(flat_indices, minlength=codebook_size).astype(np.float64)
    total = max(float(counts.sum()), 1.0)
    probs = counts / total
    active_mask = probs > 0.0
    entropy = float(-(probs[active_mask] * np.log(probs[active_mask] + 1e-12)).sum()) if active_mask.any() else 0.0
    top_k = min(5, counts.shape[0])
    top_5_coverage = float(np.sort(counts)[-top_k:].sum() / total) if top_k > 0 else 0.0
    active_code_ratio = float(active_mask.mean()) if codebook_size > 0 else 0.0
    dead_code_count = int((~active_mask).sum())
    perplexity = float(math.exp(entropy)) if entropy > 0.0 else 0.0
    active_ids = np.where(active_mask)[0]
    dead_ids = np.where(~active_mask)[0]
    sorted_asc = np.sort(counts)
    cumsum = np.cumsum(sorted_asc)
    gini = float((counts.shape[0] + 1 - 2 * np.sum(cumsum) / (cumsum[-1] + 1e-12)) / max(counts.shape[0], 1))
    most_used_code = int(np.argmax(counts)) if counts.size > 0 else None
    max_usage = int(counts.max()) if counts.size > 0 else 0
    median_active_usage = float(np.median(counts[active_mask])) if active_mask.any() else 0.0

    passes = (
        perplexity >= 0.3 * float(codebook_size)
        and active_code_ratio >= 0.5
        and dead_code_count <= int(round(0.3 * float(codebook_size)))
        and top_5_coverage <= 0.5
    )
    return {
        'codebook_size': int(codebook_size),
        'active_codes': int(active_mask.sum()),
        'active_code_ratio': active_code_ratio,
        'dead_code_count': dead_code_count,
        'perplexity': perplexity,
        'top_5_coverage': top_5_coverage,
        'gini': gini,
        'usage_counts': counts.astype(np.int64).tolist(),
        'active_code_ids': active_ids.astype(np.int64).tolist(),
        'dead_code_ids': dead_ids.astype(np.int64).tolist(),
        'most_used_code': most_used_code,
        'max_usage': max_usage,
        'median_active_usage': median_active_usage,
        'passes_thresholds': bool(passes),
    }


def _load_metrics_payload(run_dir: Optional[Path]) -> Dict[str, object]:
    if run_dir is None:
        return {}
    metrics_path = Path(run_dir) / 'metrics.json'
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text())


def _extract_metric_series(metrics_payload: Dict[str, object], key: str) -> List[float]:
    values: List[float] = []
    for epoch_entry in metrics_payload.get('epochs', []):
        raw_value: Any = None
        if key in {'train_loss', 'val_loss'}:
            raw_value = epoch_entry.get(key)
        else:
            raw_value = epoch_entry.get('metrics', {}).get(key)
            if raw_value is None:
                raw_value = epoch_entry.get('loss_breakdown', {}).get(key)
        if isinstance(raw_value, (int, float)):
            values.append(float(raw_value))
    return values


def _extract_aligned_metric_series(metrics_payload: Dict[str, object], key: str) -> List[float]:
    values: List[float] = []
    for epoch_entry in metrics_payload.get('epochs', []):
        raw_value: Any = None
        if key in {'train_loss', 'val_loss'}:
            raw_value = epoch_entry.get(key)
        else:
            raw_value = epoch_entry.get('metrics', {}).get(key)
            if raw_value is None:
                raw_value = epoch_entry.get('loss_breakdown', {}).get(key)
        if isinstance(raw_value, (int, float)):
            values.append(float(raw_value))
        else:
            values.append(float('nan'))
    return values


def _convergence_summary(metrics_payload: Dict[str, object], key: str) -> Dict[str, object]:
    series = _extract_metric_series(metrics_payload, key)
    if len(series) < 2:
        return {'available': False}
    start_value = float(series[0])
    final_value = float(series[-1])
    best_value = float(min(series))
    return {
        'available': True,
        'start': start_value,
        'final': final_value,
        'best': best_value,
        'improved': bool(final_value <= start_value),
    }


def _compute_cross_modal_predictability(
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    transition: np.ndarray,
    lag: int,
) -> Dict[str, object]:
    if eeg_tokens.size == 0 or fnirs_tokens.size == 0:
        return {'available': False, 'reason': 'empty_tokens'}
    usable = min(eeg_tokens.shape[1], fnirs_tokens.shape[1] - int(lag))
    if usable <= 0:
        return {'available': False, 'reason': 'lag_out_of_range'}
    aligned_eeg = eeg_tokens[:, :usable]
    aligned_fnirs = fnirs_tokens[:, int(lag):int(lag) + usable]
    predictions = transition.argmax(axis=-1)[aligned_eeg]
    accuracy = float(np.mean(predictions == aligned_fnirs))
    chance = float(1.0 / max(transition.shape[1], 1))
    return {
        'available': True,
        'lag': int(lag),
        'accuracy': accuracy,
        'chance_accuracy': chance,
        'usable_tokens': int(aligned_eeg.size),
    }


def _compute_coupling_structure(model, lag: int) -> Dict[str, object]:
    logits = getattr(model, 'coupling_logits', None)
    if logits is None:
        return {'available': False, 'reason': 'missing_coupling_logits'}
    if int(lag) < 0 or int(lag) >= int(logits.shape[0]):
        return {'available': False, 'reason': 'invalid_lag'}

    transition = torch.softmax(logits[int(lag)], dim=-1).detach().float().cpu().numpy()
    row_entropy = -(transition * np.log(transition + 1e-12)).sum(axis=-1)
    max_entropy = math.log(float(transition.shape[1])) if transition.shape[1] > 1 else 1.0
    concentration_ratio = float(np.mean(transition.max(axis=-1) / np.clip(transition.mean(axis=-1), 1e-12, None)))
    return {
        'available': True,
        'lag': int(lag),
        'transition': transition,
        'row_entropy': row_entropy,
        'row_entropy_mean': float(np.mean(row_entropy)),
        'row_entropy_variance': float(np.var(row_entropy)),
        'row_entropy_ratio_to_logk': float(np.mean(row_entropy) / max(max_entropy, 1e-12)),
        'concentration_ratio': concentration_ratio,
        'max_entropy': float(max_entropy),
        'sorted_row_indices': np.argsort(row_entropy).tolist(),
    }


def _plot_coupling_heatmap(coupling: Dict[str, object], output_path: Path) -> Optional[str]:
    if not HAS_MATPLOTLIB or not coupling.get('available', False):
        return None
    transition = np.asarray(coupling['transition'])
    sorted_rows = np.asarray(coupling['sorted_row_indices'], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(transition[sorted_rows], aspect='auto', cmap='viridis')
    axis.set_title(f"Coupling Heatmap (lag={coupling['lag']})")
    axis.set_xlabel('fNIRS source token')
    axis.set_ylabel('EEG source token (sorted by row entropy)')
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save_figure(figure, output_path)


def _plot_coupling_structure_profile(coupling: Dict[str, object], output_path: Path) -> Optional[str]:
    if not HAS_MATPLOTLIB or not coupling.get('available', False):
        return None
    transition = np.asarray(coupling['transition'])
    sorted_rows = np.asarray(coupling['sorted_row_indices'], dtype=np.int64)
    row_entropy = np.asarray(coupling['row_entropy'])[sorted_rows]
    row_max = transition.max(axis=-1)[sorted_rows]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(row_entropy, color='#2E86AB', linewidth=2)
    axes[0].axhline(coupling['max_entropy'] / 2.0, color='#E74C3C', linestyle='--', linewidth=1.5)
    axes[0].set_title('Sorted Row Entropy')
    axes[0].set_xlabel('EEG source token rank')
    axes[0].set_ylabel('Entropy')
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(row_max, color='#A23B72', linewidth=2)
    axes[1].axhline(float(np.mean(row_max)), color='#F39C12', linestyle='--', linewidth=1.5)
    axes[1].set_title('Sorted Row Peak Probability')
    axes[1].set_xlabel('EEG source token rank')
    axes[1].set_ylabel('Max transition probability')
    axes[1].grid(True, alpha=0.25)

    figure.suptitle(f"Coupling Structure Profile (lag={coupling['lag']})", fontsize=13, fontweight='bold')
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save_figure(figure, output_path)


def _plot_metric_panel(axis, epochs: List[int], metrics_payload: Dict[str, object], series_specs: List[Tuple[str, str]], title: str, ylabel: str) -> None:
    plotted = False
    for key, label in series_specs:
        values = np.asarray(_extract_aligned_metric_series(metrics_payload, key), dtype=np.float64)
        if values.size == 0 or np.isnan(values).all():
            continue
        axis.plot(epochs, values, linewidth=2, label=label)
        plotted = True
    axis.set_title(title)
    axis.set_xlabel('Epoch')
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, 'No tracked series', ha='center', va='center', transform=axis.transAxes, color='#95A5A6')


def _plot_training_gate_metrics(metrics_payload: Dict[str, object], output_path: Path) -> Optional[str]:
    if not HAS_MATPLOTLIB:
        return None
    epochs_payload = metrics_payload.get('epochs', [])
    if not epochs_payload:
        return None

    epochs = [int(entry.get('epoch', index + 1)) for index, entry in enumerate(epochs_payload)]
    figure, axes = plt.subplots(3, 2, figsize=(14, 12))
    flat_axes = axes.reshape(-1)

    _plot_metric_panel(flat_axes[0], epochs, metrics_payload, [('train_loss', 'Train loss'), ('val_loss', 'Val loss')], 'Objective', 'Loss')
    _plot_metric_panel(flat_axes[1], epochs, metrics_payload, [('val_eeg_rec_loss', 'EEG reconstruction'), ('val_fnirs_rec_loss', 'fNIRS reconstruction')], 'Gate 1 Reconstruction', 'Loss')
    _plot_metric_panel(
        flat_axes[2],
        epochs,
        metrics_payload,
        [('val_source_coupling_loss', 'Source coupling'), ('val_orthogonality_loss', 'Orthogonality'), ('val_codebook_balance_loss', 'Codebook balance')],
        'Regularization',
        'Loss',
    )
    _plot_metric_panel(
        flat_axes[3],
        epochs,
        metrics_payload,
        [('val_eeg_source_perplexity', 'EEG source'), ('val_fnirs_source_perplexity', 'fNIRS source'), ('val_eeg_observation_perplexity', 'EEG observation'), ('val_fnirs_observation_perplexity', 'fNIRS observation')],
        'Codebook Perplexity',
        'Perplexity',
    )
    _plot_metric_panel(flat_axes[4], epochs, metrics_payload, [('val_utilization', 'Utilization'), ('val_source_code_overlap', 'Source overlap')], 'Usage and Overlap', 'Score')
    _plot_metric_panel(flat_axes[5], epochs, metrics_payload, [('alignment_scale', 'Alignment scale'), ('val_selected_source_lag', 'Selected lag')], 'Gate 2/3 Control Signals', 'Value')

    figure.suptitle('Training Gate Metrics', fontsize=14, fontweight='bold')
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return _save_figure(figure, output_path)


def _plot_codebook_usage(codebook_name: str, summary: Dict[str, object], output_path: Path) -> Optional[str]:
    if not HAS_MATPLOTLIB:
        return None
    usage = np.asarray(summary.get('usage_counts', []), dtype=np.float64)
    codebook_size = int(summary.get('codebook_size', usage.shape[0]))
    if usage.size == 0 or codebook_size <= 0:
        return None

    sorted_indices = np.argsort(usage)[::-1]
    sorted_usage = usage[sorted_indices]
    total_tokens = max(float(usage.sum()), 1.0)
    used_codes = int(np.count_nonzero(usage))
    dead_codes = int(codebook_size - used_codes)
    utilization = float(used_codes / max(codebook_size, 1))
    nonzero_usage = usage[usage > 0]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(codebook_size)

    axes[0, 0].bar(x, sorted_usage, color='#2E86AB', alpha=0.8, width=1.0)
    axes[0, 0].set_title('Usage Count (sorted)')
    axes[0, 0].set_xlabel('Code rank')
    axes[0, 0].set_ylabel('Token count')
    axes[0, 0].grid(True, alpha=0.25, axis='y')
    if dead_codes > 0:
        axes[0, 0].axvspan(used_codes, codebook_size, alpha=0.25, color='#E74C3C')

    cumulative = np.cumsum(sorted_usage) / total_tokens
    axes[0, 1].plot(x / max(codebook_size, 1) * 100.0, cumulative * 100.0, color='#A23B72', linewidth=2)
    axes[0, 1].plot([0, 100], [0, 100], linestyle='--', color='#95A5A6', linewidth=1)
    axes[0, 1].fill_between(x / max(codebook_size, 1) * 100.0, cumulative * 100.0, alpha=0.25, color='#A23B72')
    axes[0, 1].set_title(f"Cumulative Usage (gini={summary.get('gini', 0.0):.3f})")
    axes[0, 1].set_xlabel('% codes')
    axes[0, 1].set_ylabel('% tokens')
    axes[0, 1].grid(True, alpha=0.25)

    if nonzero_usage.size > 0:
        bins = np.logspace(0, np.log10(max(float(nonzero_usage.max()), 1.0) + 1.0), 20)
        axes[1, 0].hist(nonzero_usage, bins=bins, color='#F18F01', alpha=0.75, edgecolor='white')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_title('Active Code Usage Distribution')
        axes[1, 0].set_xlabel('Usage count (log)')
        axes[1, 0].set_ylabel('Number of codes')
        axes[1, 0].grid(True, alpha=0.25)
    else:
        axes[1, 0].text(0.5, 0.5, 'No active codes', ha='center', va='center', transform=axes[1, 0].transAxes, color='#95A5A6')
        axes[1, 0].set_title('Active Code Usage Distribution')
        axes[1, 0].axis('off')

    axes[1, 1].axis('off')
    stats_text = [
        f"Codebook: {codebook_name}",
        f"Size: {codebook_size}",
        f"Active: {used_codes} ({utilization * 100.0:.1f}%)",
        f"Dead: {dead_codes}",
        f"Perplexity: {summary.get('perplexity', 0.0):.2f}",
        f"Top-5 coverage: {summary.get('top_5_coverage', 0.0):.3f}",
        f"Most used code: {summary.get('most_used_code')}",
        f"Max usage: {summary.get('max_usage', 0)}",
        f"Median active usage: {summary.get('median_active_usage', 0.0):.1f}",
    ]
    axes[1, 1].text(0.05, 0.95, '\n'.join(stats_text), va='top', fontsize=11, family='monospace', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9}, transform=axes[1, 1].transAxes)

    figure.suptitle(f"Codebook Health - {codebook_name}", fontsize=14, fontweight='bold')
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return _save_figure(figure, output_path)


def _plot_gate_dashboard(split_name: str, gates: Dict[str, object], output_path: Path) -> Optional[str]:
    if not HAS_MATPLOTLIB or not gates:
        return None

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    flat_axes = axes.reshape(-1)

    gate1 = gates.get('gate1', {})
    codebooks = gate1.get('metrics', {}).get('codebooks', {})
    if codebooks:
        names = list(codebooks.keys())
        active_ratios = [codebooks[name]['active_code_ratio'] for name in names]
        dead_codes = [codebooks[name]['dead_code_count'] for name in names]
        positions = np.arange(len(names))
        flat_axes[0].bar(positions, active_ratios, color='#2E86AB', alpha=0.8)
        flat_axes[0].axhline(0.5, color='#E74C3C', linestyle='--', linewidth=1.5)
        flat_axes[0].set_xticks(positions)
        flat_axes[0].set_xticklabels(names, rotation=20, ha='right')
        flat_axes[0].set_ylim(0.0, 1.0)
        flat_axes[0].set_title(f"Gate 1 Health ({gate1.get('status', 'pending')})")
        flat_axes[0].set_ylabel('Active code ratio')
        for idx, dead_code in enumerate(dead_codes):
            flat_axes[0].text(idx, min(active_ratios[idx] + 0.04, 0.97), f"dead={dead_code}", ha='center', fontsize=8)
        flat_axes[0].grid(True, alpha=0.25, axis='y')
    else:
        flat_axes[0].text(0.5, 0.5, 'No Gate 1 payload', ha='center', va='center', transform=flat_axes[0].transAxes)

    gate2 = gates.get('gate2', {})
    predictability = gate2.get('metrics', {}).get('cross_modal_token_predictability', {})
    observation_gap = gate2.get('metrics', {}).get('observation_contribution_gap', {})
    if predictability.get('available', False):
        values = [float(predictability.get('accuracy', 0.0)), float(predictability.get('chance_accuracy', 0.0)), float(observation_gap.get('eeg', 0.0)), float(observation_gap.get('fnirs', 0.0))]
        labels = ['Predictability', 'Chance', 'EEG obs gap', 'fNIRS obs gap']
        colors = ['#2ECC71', '#95A5A6', '#A23B72', '#F18F01']
        flat_axes[1].bar(np.arange(len(values)), values, color=colors, alpha=0.8)
        flat_axes[1].set_xticks(np.arange(len(values)))
        flat_axes[1].set_xticklabels(labels, rotation=20, ha='right')
        flat_axes[1].set_title(f"Gate 2 Semantics ({gate2.get('status', 'pending')})")
        flat_axes[1].grid(True, alpha=0.25, axis='y')
    else:
        flat_axes[1].text(0.5, 0.5, 'Cross-modal predictability unavailable', ha='center', va='center', transform=flat_axes[1].transAxes)
        flat_axes[1].set_title(f"Gate 2 Semantics ({gate2.get('status', 'pending')})")

    gate3 = gates.get('gate3', {})
    gate3_metrics = gate3.get('metrics', {})
    if gate3_metrics:
        values = [float(gate3_metrics.get('row_entropy_ratio_to_logk', 0.0)), float(gate3_metrics.get('concentration_ratio', 0.0)), float(gate3_metrics.get('row_entropy_variance', 0.0))]
        labels = ['Entropy/logK', 'Concentration', 'Entropy variance']
        flat_axes[2].bar(np.arange(len(values)), values, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
        flat_axes[2].axhline(0.5, color='#E74C3C', linestyle='--', linewidth=1.2)
        flat_axes[2].axhline(1.5, color='#2ECC71', linestyle=':', linewidth=1.2)
        flat_axes[2].set_xticks(np.arange(len(values)))
        flat_axes[2].set_xticklabels(labels, rotation=20, ha='right')
        flat_axes[2].set_title(f"Gate 3 Structure ({gate3.get('status', 'pending')})")
        flat_axes[2].grid(True, alpha=0.25, axis='y')
    else:
        flat_axes[2].text(0.5, 0.5, 'Coupling structure unavailable', ha='center', va='center', transform=flat_axes[2].transAxes)
        flat_axes[2].set_title(f"Gate 3 Structure ({gate3.get('status', 'pending')})")

    gate4 = gates.get('gate4', {})
    gate4_metrics = gate4.get('metrics', {})
    probe_specs = [('Source subject', gate4_metrics.get('source_subject_leakage_probe', {})), ('Observation subject', gate4_metrics.get('observation_subject_leakage_probe', {})), ('Source task', gate4_metrics.get('source_task_signal_probe', {}))]
    available = [(label, payload) for label, payload in probe_specs if payload.get('available', False)]
    if available:
        labels = [label for label, _ in available]
        accuracies = [float(payload.get('accuracy', 0.0)) for _, payload in available]
        chances = [float(payload.get('chance_accuracy', 0.0)) for _, payload in available]
        positions = np.arange(len(labels))
        flat_axes[3].bar(positions - 0.15, accuracies, width=0.3, color='#2ECC71', label='Accuracy')
        flat_axes[3].bar(positions + 0.15, chances, width=0.3, color='#95A5A6', label='Chance')
        flat_axes[3].set_xticks(positions)
        flat_axes[3].set_xticklabels(labels, rotation=20, ha='right')
        flat_axes[3].set_title(f"Gate 4 Utility ({gate4.get('status', 'pending')})")
        flat_axes[3].legend(fontsize=8)
        flat_axes[3].grid(True, alpha=0.25, axis='y')
    else:
        flat_axes[3].text(0.5, 0.5, 'Utility probes unavailable', ha='center', va='center', transform=flat_axes[3].transAxes)
        flat_axes[3].set_title(f"Gate 4 Utility ({gate4.get('status', 'pending')})")

    figure.suptitle(f"Gate Dashboard - {split_name}", fontsize=14, fontweight='bold')
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return _save_figure(figure, output_path)


def _parse_frequency_range(value: object) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = float(value[0])
    high = float(value[1])
    if high <= low:
        return None
    return low, high


def _parse_positive_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0.0 else float(default)


def _parse_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _resolve_reconstruction_visualization_config(config: Dict[str, object]) -> Dict[str, object]:
    analysis_cfg = config.get('analysis', {})
    if not isinstance(analysis_cfg, dict):
        return {'enabled': False}

    reconstruction_cfg = analysis_cfg.get('reconstruction_visualization', {})
    if not isinstance(reconstruction_cfg, dict):
        return {'enabled': False}

    domains_value = reconstruction_cfg.get('domains', ['time', 'frequency', 'phase'])
    if isinstance(domains_value, str):
        domains = [domains_value]
    elif isinstance(domains_value, (list, tuple)):
        domains = [str(item) for item in domains_value]
    else:
        domains = ['time', 'frequency', 'phase']
    domains = [domain for domain in domains if domain in {'time', 'frequency', 'phase'}]
    if not domains:
        domains = ['time', 'frequency', 'phase']

    channel_indices_cfg = reconstruction_cfg.get('channel_indices', {})
    if not isinstance(channel_indices_cfg, dict):
        channel_indices_cfg = {}

    frequency_range_cfg = reconstruction_cfg.get('frequency_range_hz', {})
    if not isinstance(frequency_range_cfg, dict):
        frequency_range_cfg = {}

    time_window_cfg = reconstruction_cfg.get('time_window_s', {})
    if not isinstance(time_window_cfg, dict):
        time_window_cfg = {}

    max_time_points_cfg = reconstruction_cfg.get('max_time_points', {})
    if not isinstance(max_time_points_cfg, dict):
        max_time_points_cfg = {}

    return {
        'enabled': bool(reconstruction_cfg.get('enabled', False)),
        'subdir': str(reconstruction_cfg.get('subdir', 'reconstruction_visualizations')),
        'max_samples': max(int(reconstruction_cfg.get('max_samples', 4)), 1),
        'domains': domains,
        'channel_indices': {
            'eeg': max(int(channel_indices_cfg.get('eeg', 0)), 0),
            'fnirs': max(int(channel_indices_cfg.get('fnirs', 0)), 0),
        },
        'frequency_range_hz': {
            'eeg': _parse_frequency_range(frequency_range_cfg.get('eeg')),
            'fnirs': _parse_frequency_range(frequency_range_cfg.get('fnirs')),
        },
        'time_window_s': {
            'eeg': _parse_positive_float(time_window_cfg.get('eeg'), 2.0),
            'fnirs': _parse_positive_float(time_window_cfg.get('fnirs'), 5.0),
        },
        'max_time_points': {
            'eeg': _parse_positive_int(max_time_points_cfg.get('eeg'), 500),
            'fnirs': _parse_positive_int(max_time_points_cfg.get('fnirs'), 250),
        },
    }


def _resolve_modality_sample_rate(
    dataloader,
    config: Dict[str, object],
    modality: str,
    sample_length: int,
) -> float:
    dataset = getattr(dataloader, 'dataset', None)
    sample_rate_getter = getattr(dataset, f'get_{modality}_sample_rate', None)
    if callable(sample_rate_getter):
        try:
            return float(sample_rate_getter())
        except Exception:
            pass

    data_cfg = config.get('data', {})
    if isinstance(data_cfg, dict):
        window_cfg = data_cfg.get('window', {})
        if isinstance(window_cfg, dict):
            duration_s = window_cfg.get('duration_s')
            if duration_s is not None and float(duration_s) > 0.0:
                return float(sample_length) / float(duration_s)

        preprocessing_cfg = data_cfg.get('preprocessing', {})
        if modality == 'eeg' and isinstance(preprocessing_cfg, dict):
            resample_rate = preprocessing_cfg.get('resample_rate')
            if resample_rate is not None:
                return float(resample_rate)

    return 1.0


def _select_channel_series(batch: object, channel_index: int) -> np.ndarray:
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()
    array = np.asarray(batch)
    if array.ndim == 3:
        safe_channel_index = min(max(int(channel_index), 0), array.shape[1] - 1)
        return array[:, safe_channel_index, :]
    if array.ndim == 2:
        return array
    raise ValueError(f'Expected [B, C, T] or [B, T] arrays, got shape {array.shape}')


def _compute_signal_spectrum(signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fft = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.shape[-1], d=1.0 / max(float(sample_rate), 1e-12))
    amplitude = np.log1p(np.abs(fft))
    phase = np.unwrap(np.angle(fft)) / math.pi
    return frequencies, amplitude, phase


def _compute_welch_psd(signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    safe_sample_rate = max(float(sample_rate), 1e-12)
    try:
        from scipy.signal import welch

        frequencies, psd = welch(
            signal,
            fs=safe_sample_rate,
            nperseg=min(256, signal.shape[-1]),
        )
    except ImportError:
        fft = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(signal.shape[-1], d=1.0 / safe_sample_rate)
        psd = (np.abs(fft) ** 2) / max(signal.shape[-1], 1)
    return frequencies, 10.0 * np.log10(psd + 1e-12)


def _prepare_time_domain_view(
    signal: np.ndarray,
    sample_rate: float,
    window_s: float,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total_points = int(signal.shape[-1])
    if total_points == 0:
        return np.empty((0,), dtype=np.float64), signal

    safe_sample_rate = max(float(sample_rate), 1e-12)
    window_points = min(total_points, max(int(round(window_s * safe_sample_rate)), 1))
    view = signal[:window_points]
    indices = np.arange(window_points, dtype=np.int64)

    if view.shape[-1] > max_points:
        indices = np.linspace(0, view.shape[-1] - 1, num=max_points, dtype=np.int64)
        view = view[indices]

    return indices.astype(np.float64) / safe_sample_rate, view


def _plot_reconstruction_domain_grid(
    *,
    split_name: str,
    modality: str,
    original: object,
    branches: Dict[str, object],
    output_path: Path,
    domain: str,
    sample_rate: float,
    channel_index: int,
    frequency_range: Optional[Tuple[float, float]],
    time_window_s: float,
    max_time_points: int,
) -> Optional[str]:
    if not HAS_MATPLOTLIB:
        return None

    original_series = _select_channel_series(original, channel_index)
    if original_series.shape[0] == 0:
        return None

    branch_order = (
        ('full', 'Full'),
        ('source_only', 'Source only'),
        ('observation_only', 'Observation only'),
    )
    branch_series = {
        branch_key: _select_channel_series(branches[branch_key], channel_index)
        for branch_key, _ in branch_order
    }

    n_samples = int(original_series.shape[0])
    figure, axes = plt.subplots(
        n_samples,
        len(branch_order),
        figsize=(5.2 * len(branch_order), 2.9 * n_samples),
        squeeze=False,
    )

    domain_titles = {
        'time': 'Time-domain reconstruction',
        'frequency': 'Frequency-domain reconstruction',
        'phase': 'Phase reconstruction',
    }
    ylabels = {
        'time': 'Amplitude',
        'frequency': 'PSD (dB/Hz)',
        'phase': 'Phase / pi',
    }

    for sample_index in range(n_samples):
        original_signal = original_series[sample_index]
        for branch_column, (branch_key, branch_label) in enumerate(branch_order):
            axis = axes[sample_index, branch_column]
            reconstructed_signal = branch_series[branch_key][sample_index]

            if domain == 'time':
                x_axis, original_view = _prepare_time_domain_view(
                    original_signal,
                    sample_rate,
                    time_window_s,
                    max_time_points,
                )
                _, reconstructed_view = _prepare_time_domain_view(
                    reconstructed_signal,
                    sample_rate,
                    time_window_s,
                    max_time_points,
                )
                metric_value = float(np.mean((original_view - reconstructed_view) ** 2))
                metric_label = f'Window MSE={metric_value:.4f}'
                axis.set_xlabel('Time (s)')
            else:
                phase_frequencies, _, original_phase = _compute_signal_spectrum(original_signal, sample_rate)
                _, _, reconstructed_phase = _compute_signal_spectrum(reconstructed_signal, sample_rate)
                psd_frequencies, original_psd = _compute_welch_psd(original_signal, sample_rate)
                _, reconstructed_psd = _compute_welch_psd(reconstructed_signal, sample_rate)
                if frequency_range is not None:
                    phase_mask = (phase_frequencies >= frequency_range[0]) & (phase_frequencies <= frequency_range[1])
                    psd_mask = (psd_frequencies >= frequency_range[0]) & (psd_frequencies <= frequency_range[1])
                    phase_frequencies = phase_frequencies[phase_mask]
                    psd_frequencies = psd_frequencies[psd_mask]
                    original_phase = original_phase[phase_mask]
                    reconstructed_phase = reconstructed_phase[phase_mask]

                if domain == 'frequency':
                    x_axis = psd_frequencies
                    original_view = original_psd if frequency_range is None else original_psd[psd_mask]
                    reconstructed_view = reconstructed_psd if frequency_range is None else reconstructed_psd[psd_mask]
                    metric_value = float(np.mean(np.abs(original_view - reconstructed_view)))
                    metric_label = f'Mean |ΔPSD|={metric_value:.4f} dB'
                else:
                    x_axis = phase_frequencies
                    original_view = original_phase
                    reconstructed_view = reconstructed_phase
                    metric_value = float(np.mean(np.abs(original_view - reconstructed_view)))
                    metric_label = f'Mean |Δphase|={metric_value:.4f}'
                axis.set_xlabel('Frequency (Hz)')

            axis.plot(x_axis, original_view, color='#2E86AB', linewidth=1.4, alpha=0.9, label='Original')
            axis.plot(
                x_axis,
                reconstructed_view,
                color='#A23B72',
                linewidth=1.4,
                alpha=0.9,
                linestyle='--',
                label='Reconstructed',
            )
            axis.grid(True, alpha=0.25)
            axis.set_ylabel(ylabels[domain])
            if sample_index == 0:
                axis.set_title(branch_label)
            axis.text(
                0.02,
                0.98,
                metric_label,
                transform=axis.transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
            )
            if sample_index == 0 and branch_column == 0:
                axis.legend(loc='upper right')

    figure.suptitle(
        f'{split_name} {modality.upper()} {domain_titles[domain]} (channel {channel_index})',
        fontsize=14,
        fontweight='bold',
    )
    figure.tight_layout()
    return _save_figure(figure, output_path)


def _generate_reconstruction_visualizations(
    *,
    split_name: str,
    split_stats: Dict[str, object],
    analysis_root: Path,
    reconstruction_viz_cfg: Dict[str, object],
) -> Dict[str, object]:
    if not reconstruction_viz_cfg.get('enabled', False):
        return {}

    sample_payload = split_stats.get('reconstruction_samples')
    if not isinstance(sample_payload, dict):
        return {}

    output_root = analysis_root / str(reconstruction_viz_cfg.get('subdir', 'reconstruction_visualizations'))
    original_payload = sample_payload.get('original', {})
    branch_payload = sample_payload.get('branches', {})
    sample_rates = sample_payload.get('sample_rates', {})
    artifacts: Dict[str, object] = {}

    for modality in ('eeg', 'fnirs'):
        original = original_payload.get(modality)
        if original is None:
            continue

        modality_branches = {
            'full': branch_payload.get('full', {}).get(modality),
            'source_only': branch_payload.get('source_only', {}).get(modality),
            'observation_only': branch_payload.get('observation_only', {}).get(modality),
        }
        if any(value is None for value in modality_branches.values()):
            continue

        modality_artifacts: Dict[str, Optional[str]] = {}
        sample_rate = float(sample_rates.get(modality, 1.0))
        channel_index = int(reconstruction_viz_cfg.get('channel_indices', {}).get(modality, 0))
        frequency_range = reconstruction_viz_cfg.get('frequency_range_hz', {}).get(modality)
        time_window_s = float(reconstruction_viz_cfg.get('time_window_s', {}).get(modality, 2.0))
        max_time_points = int(reconstruction_viz_cfg.get('max_time_points', {}).get(modality, 500))

        for domain in reconstruction_viz_cfg.get('domains', []):
            output_path = output_root / f'{split_name}_{modality}_{domain}_reconstruction.png'
            modality_artifacts[f'{domain}_path'] = _plot_reconstruction_domain_grid(
                split_name=split_name,
                modality=modality,
                original=original,
                branches=modality_branches,
                output_path=output_path,
                domain=domain,
                sample_rate=sample_rate,
                channel_index=channel_index,
                frequency_range=frequency_range,
                time_window_s=time_window_s,
                max_time_points=max_time_points,
            )

        artifacts[modality] = modality_artifacts

    return artifacts


def _flatten_artifacts(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        paths: List[str] = []
        for item in value.values():
            paths.extend(_flatten_artifacts(item))
        return paths
    if isinstance(value, (list, tuple)):
        paths: List[str] = []
        for item in value:
            paths.extend(_flatten_artifacts(item))
        return paths
    return []


def _collect_split_statistics(
    model,
    dataloader,
    device: torch.device,
    *,
    config: Dict[str, object],
    reconstruction_viz_cfg: Dict[str, object],
) -> Dict[str, object]:
    scalar_totals: Dict[str, float] = {}
    index_bank: Dict[str, List[np.ndarray]] = {
        'eeg_source_indices': [],
        'fnirs_source_indices': [],
        'eeg_observation_indices': [],
        'fnirs_observation_indices': [],
    }
    recon_totals = {
        'eeg_full_mse': 0.0,
        'fnirs_full_mse': 0.0,
        'eeg_source_only_mse': 0.0,
        'fnirs_source_only_mse': 0.0,
        'eeg_observation_only_mse': 0.0,
        'fnirs_observation_only_mse': 0.0,
    }
    source_feature_chunks: List[np.ndarray] = []
    observation_feature_chunks: List[np.ndarray] = []
    subject_id_chunks: List[np.ndarray] = []
    label_id_chunks: List[np.ndarray] = []
    batch_count = 0
    capture_reconstructions = bool(reconstruction_viz_cfg.get('enabled', False))
    reconstruction_limit = int(reconstruction_viz_cfg.get('max_samples', 0)) if capture_reconstructions else 0
    reconstruction_capture_count = 0
    reconstruction_bank: Optional[Dict[str, object]] = None
    if reconstruction_limit > 0:
        reconstruction_bank = {
            'original': {'eeg': [], 'fnirs': []},
            'branches': {
                'full': {'eeg': [], 'fnirs': []},
                'source_only': {'eeg': [], 'fnirs': []},
                'observation_only': {'eeg': [], 'fnirs': []},
            },
        }

    source_codebook_size = int(getattr(model, 'source_codebook_size', model.get_codebook_size()))
    eeg_observation_codebook_size = int(getattr(model, 'eeg_observation_codebook_size', source_codebook_size))
    fnirs_observation_codebook_size = int(getattr(model, 'fnirs_observation_codebook_size', source_codebook_size))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in dataloader:
                eeg = batch['eeg'].to(device, non_blocking=True)
                fnirs = batch['fnirs'].to(device, non_blocking=True)
                outputs = model(eeg, fnirs)
                batch_count += 1

                for key, value in outputs.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        scalar_totals[key] = scalar_totals.get(key, 0.0) + _to_float(value)

                eeg_source_tokens = outputs['eeg_source_indices'].detach().cpu().numpy().astype(np.int64, copy=False)
                fnirs_source_tokens = outputs['fnirs_source_indices'].detach().cpu().numpy().astype(np.int64, copy=False)
                eeg_observation_tokens = outputs['eeg_observation_indices'].detach().cpu().numpy().astype(np.int64, copy=False)
                fnirs_observation_tokens = outputs['fnirs_observation_indices'].detach().cpu().numpy().astype(np.int64, copy=False)

                index_bank['eeg_source_indices'].append(eeg_source_tokens)
                index_bank['fnirs_source_indices'].append(fnirs_source_tokens)
                index_bank['eeg_observation_indices'].append(eeg_observation_tokens)
                index_bank['fnirs_observation_indices'].append(fnirs_observation_tokens)

                recon_totals['eeg_full_mse'] += _mse(outputs['eeg_reconstructed'], eeg)
                recon_totals['fnirs_full_mse'] += _mse(outputs['fnirs_reconstructed'], fnirs)
                recon_totals['eeg_source_only_mse'] += _mse(outputs['eeg_source_only_reconstructed'], eeg)
                recon_totals['fnirs_source_only_mse'] += _mse(outputs['fnirs_source_only_reconstructed'], fnirs)
                recon_totals['eeg_observation_only_mse'] += _mse(outputs['eeg_observation_only_reconstructed'], eeg)
                recon_totals['fnirs_observation_only_mse'] += _mse(outputs['fnirs_observation_only_reconstructed'], fnirs)

                source_feature_chunks.append(_source_hist_features(eeg_source_tokens, fnirs_source_tokens, source_codebook_size))
                observation_feature_chunks.append(
                    _observation_hist_features(
                        eeg_observation_tokens,
                        fnirs_observation_tokens,
                        eeg_observation_codebook_size,
                        fnirs_observation_codebook_size,
                    )
                )
                take = int(eeg_source_tokens.shape[0])
                subject_ids = _maybe_batch_vector(batch, ('subject', 'subject_id'))
                label_ids = _maybe_batch_vector(batch, ('label', 'labels', 'task', 'condition'))
                subject_id_chunks.append(_probe_id_array(subject_ids, take))
                label_id_chunks.append(_probe_id_array(label_ids, take))

                if reconstruction_bank is not None and reconstruction_capture_count < reconstruction_limit:
                    capture_take = min(reconstruction_limit - reconstruction_capture_count, int(eeg.shape[0]))
                    reconstruction_bank['original']['eeg'].append(eeg[:capture_take].detach().cpu())
                    reconstruction_bank['original']['fnirs'].append(fnirs[:capture_take].detach().cpu())
                    reconstruction_bank['branches']['full']['eeg'].append(outputs['eeg_reconstructed'][:capture_take].detach().cpu())
                    reconstruction_bank['branches']['full']['fnirs'].append(outputs['fnirs_reconstructed'][:capture_take].detach().cpu())
                    reconstruction_bank['branches']['source_only']['eeg'].append(
                        outputs['eeg_source_only_reconstructed'][:capture_take].detach().cpu()
                    )
                    reconstruction_bank['branches']['source_only']['fnirs'].append(
                        outputs['fnirs_source_only_reconstructed'][:capture_take].detach().cpu()
                    )
                    reconstruction_bank['branches']['observation_only']['eeg'].append(
                        outputs['eeg_observation_only_reconstructed'][:capture_take].detach().cpu()
                    )
                    reconstruction_bank['branches']['observation_only']['fnirs'].append(
                        outputs['fnirs_observation_only_reconstructed'][:capture_take].detach().cpu()
                    )
                    reconstruction_capture_count += capture_take
    finally:
        if was_training:
            model.train()

    if batch_count == 0:
        return {'available': False, 'reason': 'empty_dataloader'}

    mean_scalars = {key: value / batch_count for key, value in scalar_totals.items()}
    mean_recon = {key: value / batch_count for key, value in recon_totals.items()}
    branch_reconstruction = {
        **mean_recon,
        'eeg_source_gap': mean_recon['eeg_source_only_mse'] - mean_recon['eeg_full_mse'],
        'fnirs_source_gap': mean_recon['fnirs_source_only_mse'] - mean_recon['fnirs_full_mse'],
        'eeg_observation_gap': mean_recon['eeg_observation_only_mse'] - mean_recon['eeg_full_mse'],
        'fnirs_observation_gap': mean_recon['fnirs_observation_only_mse'] - mean_recon['fnirs_full_mse'],
    }

    result = {
        'available': True,
        'num_batches': int(batch_count),
        'mean_scalars': mean_scalars,
        'branch_reconstruction': branch_reconstruction,
        'codebook_sizes': {
            'source': source_codebook_size,
            'eeg_observation': eeg_observation_codebook_size,
            'fnirs_observation': fnirs_observation_codebook_size,
        },
        'eeg_source_tokens': np.concatenate(index_bank['eeg_source_indices'], axis=0),
        'fnirs_source_tokens': np.concatenate(index_bank['fnirs_source_indices'], axis=0),
        'eeg_observation_tokens': np.concatenate(index_bank['eeg_observation_indices'], axis=0),
        'fnirs_observation_tokens': np.concatenate(index_bank['fnirs_observation_indices'], axis=0),
        'source_features': np.concatenate(source_feature_chunks, axis=0),
        'observation_features': np.concatenate(observation_feature_chunks, axis=0),
        'subject_ids': np.concatenate(subject_id_chunks, axis=0),
        'label_ids': np.concatenate(label_id_chunks, axis=0),
        'codebooks': {
            'eeg_source': _codebook_summary(index_bank['eeg_source_indices'], source_codebook_size),
            'fnirs_source': _codebook_summary(index_bank['fnirs_source_indices'], source_codebook_size),
            'eeg_observation': _codebook_summary(index_bank['eeg_observation_indices'], eeg_observation_codebook_size),
            'fnirs_observation': _codebook_summary(index_bank['fnirs_observation_indices'], fnirs_observation_codebook_size),
        },
    }

    if reconstruction_bank is not None and reconstruction_capture_count > 0:
        reconstruction_samples = {
            'original': {
                modality: torch.cat(chunks, dim=0)
                for modality, chunks in reconstruction_bank['original'].items()
                if chunks
            },
            'branches': {
                branch_name: {
                    modality: torch.cat(chunks, dim=0)
                    for modality, chunks in branch_payload.items()
                    if chunks
                }
                for branch_name, branch_payload in reconstruction_bank['branches'].items()
            },
        }
        reconstruction_samples['sample_rates'] = {
            'eeg': _resolve_modality_sample_rate(
                dataloader,
                config,
                'eeg',
                int(reconstruction_samples['original']['eeg'].shape[-1]),
            ),
            'fnirs': _resolve_modality_sample_rate(
                dataloader,
                config,
                'fnirs',
                int(reconstruction_samples['original']['fnirs'].shape[-1]),
            ),
        }
        result['reconstruction_samples'] = reconstruction_samples

    return result


def _build_gate_1(split_stats: Dict[str, object], metrics_payload: Dict[str, object]) -> Dict[str, object]:
    codebooks = split_stats['codebooks']
    codebook_pass = all(summary['passes_thresholds'] for summary in codebooks.values())
    eeg_convergence = _convergence_summary(metrics_payload, 'val_eeg_rec_loss')
    fnirs_convergence = _convergence_summary(metrics_payload, 'val_fnirs_rec_loss')

    convergence_checks = [
        item['improved']
        for item in (eeg_convergence, fnirs_convergence)
        if item.get('available', False)
    ]
    reconstruction_converged = all(convergence_checks) if convergence_checks else None

    notes: List[str] = []
    if reconstruction_converged is None:
        notes.append('Reconstruction convergence trend is unavailable from metrics.json.')

    if codebook_pass and reconstruction_converged is True:
        status = 'pass'
    elif not codebook_pass or reconstruction_converged is False:
        status = 'fail'
    else:
        status = 'pending'

    return {
        'status': status,
        'metrics': {
            'codebooks': codebooks,
            'reconstruction': split_stats['branch_reconstruction'],
            'convergence': {
                'eeg': eeg_convergence,
                'fnirs': fnirs_convergence,
            },
        },
        'notes': notes,
        'artifacts': {
            'codebook_usage_paths': {},
        },
    }


def _build_gate_2(split_stats: Dict[str, object], best_lag: int, coupling: Dict[str, object]) -> Dict[str, object]:
    predictability = _compute_cross_modal_predictability(
        split_stats['eeg_source_tokens'],
        split_stats['fnirs_source_tokens'],
        np.asarray(coupling['transition']) if coupling.get('available', False) else np.zeros((1, 1), dtype=np.float64),
        lag=best_lag,
    ) if coupling.get('available', False) else {'available': False, 'reason': 'missing_coupling'}

    mean_scalars = split_stats['mean_scalars']
    source_target_mse = mean_scalars.get('source_target_loss')
    source_target_random_baseline = mean_scalars.get('source_target_random_baseline')
    observation_gap_ok = (
        split_stats['branch_reconstruction']['eeg_observation_gap'] > 0.0
        and split_stats['branch_reconstruction']['fnirs_observation_gap'] > 0.0
    )
    predictability_ok = predictability.get('available', False) and predictability['accuracy'] > predictability['chance_accuracy']
    source_util_gap = abs(
        split_stats['codebooks']['eeg_source']['active_code_ratio'] -
        split_stats['codebooks']['fnirs_source']['active_code_ratio']
    )
    source_independence_ok = source_util_gap < 0.3
    source_target_ready = (
        source_target_mse is not None
        and source_target_random_baseline is not None
        and float(source_target_mse) < float(source_target_random_baseline)
    )

    notes: List[str] = []
    if source_target_mse is None:
        notes.append('HRF source target metrics are not available in this run.')
    elif source_target_random_baseline is None:
        notes.append('HRF source target random baseline is missing, so Gate 2 cannot fully pass.')

    if not observation_gap_ok or not predictability_ok or not source_independence_ok:
        status = 'fail'
    elif source_target_ready:
        status = 'pass'
    else:
        status = 'pending'

    return {
        'status': status,
        'metrics': {
            'source_target_mse': source_target_mse,
            'source_target_random_baseline': source_target_random_baseline,
            'observation_contribution_gap': {
                'eeg': split_stats['branch_reconstruction']['eeg_observation_gap'],
                'fnirs': split_stats['branch_reconstruction']['fnirs_observation_gap'],
            },
            'cross_modal_token_predictability': predictability,
            'source_codebook_independence': {
                'active_code_ratio_gap': source_util_gap,
                'eeg_source_active_ratio': split_stats['codebooks']['eeg_source']['active_code_ratio'],
                'fnirs_source_active_ratio': split_stats['codebooks']['fnirs_source']['active_code_ratio'],
            },
        },
        'notes': notes,
    }


def _build_gate_3(coupling: Dict[str, object], figure_path: Optional[str], structure_profile_path: Optional[str]) -> Dict[str, object]:
    if not coupling.get('available', False):
        return {
            'status': 'pending',
            'metrics': {},
            'notes': ['Coupling logits are not available.'],
            'artifacts': {},
        }

    entropy_ok = coupling['row_entropy_mean'] < (coupling['max_entropy'] / 2.0)
    concentration_ok = coupling['concentration_ratio'] > 1.5
    variance_ok = coupling['row_entropy_variance'] > 0.0
    status = 'pass' if entropy_ok and concentration_ok and variance_ok else 'fail'

    return {
        'status': status,
        'metrics': {
            'best_lag': coupling['lag'],
            'row_entropy_mean': coupling['row_entropy_mean'],
            'row_entropy_variance': coupling['row_entropy_variance'],
            'row_entropy_ratio_to_logk': coupling['row_entropy_ratio_to_logk'],
            'concentration_ratio': coupling['concentration_ratio'],
            'heatmap_path': figure_path,
            'structure_profile_path': structure_profile_path,
        },
        'notes': [],
        'artifacts': {
            'heatmap_path': figure_path,
            'structure_profile_path': structure_profile_path,
        },
    }


def _build_gate_4(split_stats: Dict[str, object]) -> Dict[str, object]:
    source_subject_probe = _nearest_centroid_probe(split_stats['source_features'], split_stats['subject_ids'], seed=17)
    observation_subject_probe = _nearest_centroid_probe(split_stats['observation_features'], split_stats['subject_ids'], seed=19)
    source_task_probe = _nearest_centroid_probe(split_stats['source_features'], split_stats['label_ids'], seed=23)

    ssr = None
    if source_subject_probe.get('available') and source_task_probe.get('available'):
        ssr = float(
            source_task_probe['normalized_lift'] /
            max(source_subject_probe['normalized_lift'], 1e-12)
        )

    notes: List[str] = []
    if not source_subject_probe.get('available'):
        notes.append('Source subject leakage probe is unavailable for this split.')
    if not observation_subject_probe.get('available'):
        notes.append('Observation subject leakage probe is unavailable for this split.')
    if not source_task_probe.get('available'):
        notes.append('Source task probe is unavailable for this split.')

    if not source_subject_probe.get('available') or not observation_subject_probe.get('available') or not source_task_probe.get('available'):
        status = 'pending'
    else:
        subject_separation_ok = source_subject_probe['accuracy'] < observation_subject_probe['accuracy']
        task_signal_ok = source_task_probe['accuracy'] > source_task_probe['chance_accuracy']
        ssr_ok = ssr is not None and ssr > 1.0
        status = 'pass' if subject_separation_ok and task_signal_ok and ssr_ok else 'fail'

    return {
        'status': status,
        'metrics': {
            'source_subject_leakage_probe': source_subject_probe,
            'observation_subject_leakage_probe': observation_subject_probe,
            'source_task_signal_probe': source_task_probe,
            'semantic_selectivity_ratio': ssr,
        },
        'notes': notes,
    }


def _build_split_gate_summary(
    split_name: str,
    split_stats: Dict[str, object],
    metrics_payload: Dict[str, object],
    analysis_root: Path,
    reconstruction_viz_cfg: Dict[str, object],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    best_lag = int(round(float(split_stats['mean_scalars'].get('selected_source_lag', 0.0))))
    coupling = _compute_coupling_structure(model=split_stats['model_ref'], lag=best_lag)
    figure_path = _plot_coupling_heatmap(coupling, analysis_root / f"{split_name}_coupling_heatmap.png")
    structure_profile_path = _plot_coupling_structure_profile(coupling, analysis_root / f"{split_name}_coupling_structure.png")
    gate_1 = _build_gate_1(split_stats, metrics_payload)
    gate_2 = _build_gate_2(split_stats, best_lag=best_lag, coupling=coupling)
    gate_3 = _build_gate_3(coupling, figure_path=figure_path, structure_profile_path=structure_profile_path)
    gate_4 = _build_gate_4(split_stats)
    codebook_usage_paths: Dict[str, Optional[str]] = {}
    for codebook_name, codebook_payload in split_stats['codebooks'].items():
        codebook_usage_paths[codebook_name] = _plot_codebook_usage(
            codebook_name=f"{split_name}:{codebook_name}",
            summary=codebook_payload,
            output_path=analysis_root / f"{split_name}_{codebook_name}_codebook_usage.png",
        )

    gate_1['artifacts']['codebook_usage_paths'] = codebook_usage_paths
    reconstruction_visualization_paths = _generate_reconstruction_visualizations(
        split_name=split_name,
        split_stats=split_stats,
        analysis_root=analysis_root,
        reconstruction_viz_cfg=reconstruction_viz_cfg,
    )
    gates = {
        'gate1': gate_1,
        'gate2': gate_2,
        'gate3': gate_3,
        'gate4': gate_4,
    }
    dashboard_path = _plot_gate_dashboard(split_name, gates, analysis_root / f"{split_name}_gate_dashboard.png")
    return gates, {
        'gate_dashboard_path': dashboard_path,
        'coupling_heatmap_path': figure_path,
        'coupling_structure_path': structure_profile_path,
        'codebook_usage_paths': codebook_usage_paths,
        'reconstruction_visualization_paths': reconstruction_visualization_paths,
    }


def _promotion_verdict(gates: Dict[str, object]) -> str:
    statuses = {name: details['status'] for name, details in gates.items()}
    if statuses.get('gate1') == 'fail':
        return 'blocked_gate1'
    if any(status == 'fail' for gate_name, status in statuses.items() if gate_name != 'gate1'):
        return 'hold_repair'
    if all(status == 'pass' for status in statuses.values()):
        return 'promote'
    return 'hold_pending'


def _build_summary_text(primary_split: str, gates: Dict[str, object], best_lag: Optional[int]) -> str:
    gate_states = ', '.join(f"{name}={details['status']}" for name, details in gates.items())
    lag_text = 'n/a' if best_lag is None else str(best_lag)
    return f'Primary split {primary_split}: {gate_states}; best_lag={lag_text}.'


def _build_markdown_report(payload: Dict[str, object]) -> str:
    lines = [
        '# Tokenizer Gate Summary',
        '',
        f"- Primary split: {payload['primary_split']}",
        f"- Promotion verdict: {payload['promotion_verdict']}",
        f"- Best lag: {payload['final_summary'].get('best_lag')}",
        f"- Training gate metrics figure: {payload.get('artifacts', {}).get('training_gate_metrics_path')}",
        '',
    ]
    for gate_name, gate_payload in payload['gates'].items():
        lines.append(f"## {gate_name.upper()}")
        lines.append('')
        lines.append(f"- Status: {gate_payload['status']}")
        for note in gate_payload.get('notes', []):
            lines.append(f"- Note: {note}")
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def _write_manifest(analysis_root: Path, splits: Iterable[str], artifact_paths: Iterable[str]) -> None:
    manifest = {
        'suite_version': SCORECARD_SCHEMA_VERSION,
        'generated_at': datetime.now().isoformat(),
        'artifact_root': '.',
        'summary_json': 'gate_summary.json',
        'summary_markdown': 'gate_summary.md',
        'split_payloads': [f'split_{split_name}.json' for split_name in splits],
        'figure_files': sorted({Path(path).name for path in artifact_paths}),
    }
    _write_json(analysis_root / 'manifest.json', manifest)


def generate_source_observation_scorecard(
    *,
    model,
    dataloaders: Dict[str, object],
    config: Dict[str, object],
    output_dir: Path,
    device: torch.device,
    splits: Iterable[str] = ('val', 'test'),
    run_dir: Path | None = None,
) -> Dict[str, object]:
    analysis_root = Path(output_dir)
    analysis_root.mkdir(parents=True, exist_ok=True)
    reconstruction_viz_cfg = _resolve_reconstruction_visualization_config(config)

    metrics_payload = _load_metrics_payload(run_dir)
    training_gate_metrics_path = _plot_training_gate_metrics(metrics_payload, analysis_root / 'training_gate_metrics.png')
    split_payloads: Dict[str, object] = {}
    split_gates: Dict[str, object] = {}
    artifact_paths: List[str] = []
    if training_gate_metrics_path is not None:
        artifact_paths.append(training_gate_metrics_path)
    for split_name in splits:
        dataloader = dataloaders.get(split_name)
        if dataloader is None:
            continue
        split_stats = _collect_split_statistics(
            model,
            dataloader,
            device,
            config=config,
            reconstruction_viz_cfg=reconstruction_viz_cfg,
        )
        if not split_stats.get('available', False):
            split_payloads[split_name] = split_stats
            continue
        split_stats['model_ref'] = model
        split_stats['split_name'] = split_name
        gates, split_artifacts = _build_split_gate_summary(
            split_name,
            split_stats,
            metrics_payload,
            analysis_root,
            reconstruction_viz_cfg,
        )
        split_payload = {
            'available': True,
            'num_batches': split_stats['num_batches'],
            'mean_scalars': split_stats['mean_scalars'],
            'branch_reconstruction': split_stats['branch_reconstruction'],
            'codebooks': split_stats['codebooks'],
            'gates': gates,
            'artifacts': split_artifacts,
        }
        split_payloads[split_name] = split_payload
        split_gates[split_name] = gates
        artifact_paths.extend(_flatten_artifacts(split_artifacts))
        _write_json(analysis_root / f'split_{split_name}.json', split_payload)

    _write_manifest(analysis_root, split_payloads.keys(), artifact_paths)
    primary_split = 'test' if 'test' in split_payloads else next(iter(split_payloads.keys()), 'val')
    primary_gates = split_gates.get(primary_split, {})
    best_lag = None
    if primary_gates:
        best_lag = primary_gates['gate3']['metrics'].get('best_lag')
    promotion_verdict = _promotion_verdict(primary_gates) if primary_gates else 'hold_pending'
    final_summary = {
        'schema_version': SCORECARD_SCHEMA_VERSION,
        'run_name': Path(run_dir).name if run_dir is not None else None,
        'primary_split': primary_split,
        'promotion_verdict': promotion_verdict,
        'gate_verdicts': {name: details['status'] for name, details in primary_gates.items()},
        'best_checkpoint': 'checkpoints/best_model.pt',
        'best_lag': best_lag,
        'summary': _build_summary_text(primary_split, primary_gates, best_lag),
    }
    payload = {
        'schema_version': SCORECARD_SCHEMA_VERSION,
        'analysis_type': 'source_observation_gate_analysis',
        'generated_at': datetime.now().isoformat(),
        'artifact_root': str(analysis_root),
        'primary_split': primary_split,
        'promotion_verdict': promotion_verdict,
        'gates': primary_gates,
        'splits': split_payloads,
        'final_summary': final_summary,
        'artifacts': {
            'training_gate_metrics_path': training_gate_metrics_path,
        },
    }
    _write_json(analysis_root / 'gate_summary.json', payload)
    _write_text(analysis_root / 'gate_summary.md', _build_markdown_report(payload))
    return payload


__all__ = ['generate_source_observation_scorecard']