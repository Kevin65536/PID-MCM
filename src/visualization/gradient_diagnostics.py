"""Gradient diagnostics visualizations for multi-loss training."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch


DEFAULT_COLORS: Dict[str, str] = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#2ECC71',
    'danger': '#E74C3C',
}


DEFAULT_PARAMETER_GROUP_SPECS: List[Dict[str, Any]] = [
    {'name': 'eeg_patch_embed', 'label': 'EEG Patch', 'prefixes': ('eeg_patch_embed.',)},
    {'name': 'fnirs_patch_embed', 'label': 'fNIRS Patch', 'prefixes': ('fnirs_patch_embed.',)},
    {'name': 'eeg_encoder', 'label': 'EEG Encoder', 'prefixes': ('eeg_encoder.',)},
    {'name': 'fnirs_encoder', 'label': 'fNIRS Encoder', 'prefixes': ('fnirs_encoder.',)},
    {'name': 'eeg_source_proj', 'label': 'EEG Source Proj', 'prefixes': ('eeg_source_proj.',)},
    {'name': 'eeg_observation_proj', 'label': 'EEG Obs Proj', 'prefixes': ('eeg_observation_proj.',)},
    {'name': 'fnirs_source_proj', 'label': 'fNIRS Source Proj', 'prefixes': ('fnirs_source_proj.',)},
    {'name': 'fnirs_observation_proj', 'label': 'fNIRS Obs Proj', 'prefixes': ('fnirs_observation_proj.',)},
    {'name': 'eeg_source_quantizer', 'label': 'EEG Source Quant', 'prefixes': ('eeg_source_quantizer.',)},
    {'name': 'fnirs_source_quantizer', 'label': 'fNIRS Source Quant', 'prefixes': ('fnirs_source_quantizer.',)},
    {'name': 'eeg_observation_quantizer', 'label': 'EEG Obs Quant', 'prefixes': ('eeg_observation_quantizer.',)},
    {'name': 'fnirs_observation_quantizer', 'label': 'fNIRS Obs Quant', 'prefixes': ('fnirs_observation_quantizer.',)},
    {'name': 'coupling_logits', 'label': 'Coupling', 'prefixes': ('coupling_logits',)},
    {'name': 'context_coupling', 'label': 'Context Coupling', 'prefixes': (
        'context_coupling_eeg_factors',
        'context_coupling_fnirs_factors',
        'context_coupling_router.',
    )},
    {'name': 'eeg_decode_input_proj', 'label': 'EEG Decode In', 'prefixes': ('eeg_decode_input_proj.',)},
    {'name': 'fnirs_decode_input_proj', 'label': 'fNIRS Decode In', 'prefixes': ('fnirs_decode_input_proj.',)},
    {'name': 'eeg_decoder', 'label': 'EEG Decoder', 'prefixes': ('eeg_decoder.',)},
    {'name': 'fnirs_decoder', 'label': 'fNIRS Decoder', 'prefixes': ('fnirs_decoder.',)},
    {'name': 'eeg_amplitude_head', 'label': 'EEG Amp Head', 'prefixes': ('eeg_amplitude_head.',)},
    {'name': 'eeg_phase_head', 'label': 'EEG Phase Head', 'prefixes': ('eeg_phase_head.',)},
    {'name': 'fnirs_amplitude_head', 'label': 'fNIRS Amp Head', 'prefixes': ('fnirs_amplitude_head.',)},
    {'name': 'fnirs_phase_head', 'label': 'fNIRS Phase Head', 'prefixes': ('fnirs_phase_head.',)},
    {'name': 'other', 'label': 'Other', 'prefixes': ()},
]

DEFAULT_GRADIENT_QUANTILES: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)


def _normalize_parameter_group_specs(
    group_specs: Optional[Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    source_specs = group_specs or DEFAULT_PARAMETER_GROUP_SPECS
    for spec in source_specs:
        name = str(spec['name'])
        label = str(spec.get('label', name))
        prefixes = tuple(str(prefix) for prefix in spec.get('prefixes', (f'{name}.',)))
        normalized.append({
            'name': name,
            'label': label,
            'prefixes': prefixes,
        })

    if not any(spec['name'] == 'other' for spec in normalized):
        normalized.append({'name': 'other', 'label': 'Other', 'prefixes': ()})
    return normalized


def resolve_gradient_parameter_groups(
    param_names: Sequence[str],
    group_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized_specs = _normalize_parameter_group_specs(group_specs)
    other_index = next(
        (index for index, spec in enumerate(normalized_specs) if spec['name'] == 'other'),
        len(normalized_specs) - 1,
    )

    raw_group_indices: List[int] = []
    matched_counts = [0 for _ in normalized_specs]
    for param_name in param_names:
        matched_index = None
        for index, spec in enumerate(normalized_specs):
            prefixes = spec['prefixes']
            if prefixes and any(param_name.startswith(prefix) for prefix in prefixes):
                matched_index = index
                break
        if matched_index is None:
            matched_index = other_index
        raw_group_indices.append(matched_index)
        matched_counts[matched_index] += 1

    keep_indices = [index for index, count in enumerate(matched_counts) if count > 0]
    remap = {old_index: new_index for new_index, old_index in enumerate(keep_indices)}

    return {
        'group_names': [normalized_specs[index]['name'] for index in keep_indices],
        'group_labels': [normalized_specs[index]['label'] for index in keep_indices],
        'param_group_indices': [remap[index] for index in raw_group_indices],
        'param_group_counts': [matched_counts[index] for index in keep_indices],
        'group_specs': [normalized_specs[index] for index in keep_indices],
    }


def compute_component_group_attribution(
    component_entries: Sequence[Mapping[str, Any]],
    param_names: Sequence[str],
    group_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not component_entries or not param_names:
        return None

    group_layout = resolve_gradient_parameter_groups(param_names, group_specs=group_specs)
    component_group_squared = np.zeros(
        (len(component_entries), len(group_layout['group_names'])),
        dtype=np.float32,
    )
    for component_index, entry in enumerate(component_entries):
        for group_index, grad in zip(group_layout['param_group_indices'], entry['grads']):
            if grad is None:
                continue
            component_group_squared[component_index, group_index] += float(torch.sum(grad * grad).item())

    component_group_norms = np.sqrt(component_group_squared).astype(np.float32, copy=False)
    component_totals = component_group_norms.sum(axis=1, keepdims=True)
    component_group_shares = np.divide(
        component_group_norms,
        np.clip(component_totals, 1e-12, None),
        out=np.zeros_like(component_group_norms),
        where=component_totals > 0.0,
    )
    group_totals = component_group_norms.sum(axis=0, keepdims=True)
    group_component_shares = np.divide(
        component_group_norms,
        np.clip(group_totals, 1e-12, None),
        out=np.zeros_like(component_group_norms),
        where=group_totals > 0.0,
    )

    return {
        'group_names': group_layout['group_names'],
        'group_labels': group_layout['group_labels'],
        'parameter_group_counts': group_layout['param_group_counts'],
        'component_group_norms': component_group_norms.tolist(),
        'component_group_shares': component_group_shares.tolist(),
        'group_component_shares': group_component_shares.tolist(),
    }


def _format_quantile_key(quantile: float) -> str:
    return f'p{int(round(float(quantile) * 100)):02d}'


def summarize_total_gradient_groups(
    model,
    group_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    quantiles: Sequence[float] = DEFAULT_GRADIENT_QUANTILES,
) -> Optional[Dict[str, Any]]:
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not named_params:
        return None

    group_layout = resolve_gradient_parameter_groups(
        [name for name, _ in named_params],
        group_specs=group_specs,
    )
    group_total_squared = np.zeros(len(group_layout['group_names']), dtype=np.float32)
    quantile_keys = [_format_quantile_key(level) for level in quantiles]
    group_quantiles = {
        key: np.zeros(len(group_layout['group_names']), dtype=np.float32)
        for key in quantile_keys
    }

    for group_index in range(len(group_layout['group_names'])):
        group_grad_values: List[torch.Tensor] = []
        for assigned_index, (_, param) in zip(group_layout['param_group_indices'], named_params):
            if assigned_index != group_index or param.grad is None:
                continue
            grad_value = param.grad.detach()
            group_total_squared[group_index] += float(torch.sum(grad_value * grad_value).item())
            group_grad_values.append(grad_value.reshape(-1).abs())
        if not group_grad_values:
            continue

        flat_grads = torch.cat(group_grad_values, dim=0)
        quantile_levels = torch.tensor(list(quantiles), device=flat_grads.device, dtype=flat_grads.dtype)
        quantile_values = torch.quantile(flat_grads, quantile_levels)
        for key, value in zip(quantile_keys, quantile_values.tolist()):
            group_quantiles[key][group_index] = float(value)

    group_total_norms = np.sqrt(group_total_squared).astype(np.float32, copy=False)
    group_total_shares = np.divide(
        group_total_norms,
        max(float(group_total_norms.sum()), 1e-12),
        out=np.zeros_like(group_total_norms),
        where=group_total_norms >= 0.0,
    )
    return {
        'group_names': group_layout['group_names'],
        'group_labels': group_layout['group_labels'],
        'group_total_norms': group_total_norms.tolist(),
        'group_total_shares': group_total_shares.tolist(),
        'group_abs_grad_quantiles': {
            key: values.tolist()
            for key, values in group_quantiles.items()
        },
    }


def _resolve_heatmap_vmax(values: np.ndarray) -> float:
    max_value = float(np.max(values)) if values.size else 0.0
    return max(max_value, 1e-6)


def _top_group_summary(group_labels: Sequence[str], group_total_shares: np.ndarray) -> str:
    if group_total_shares.size == 0:
        return ''
    top_indices = np.argsort(group_total_shares)[::-1][: min(3, group_total_shares.shape[0])]
    summary_parts = [
        f'{group_labels[index]} {group_total_shares[index]:.0%}'
        for index in top_indices
        if group_total_shares[index] > 0.0
    ]
    return ', '.join(summary_parts)


def summarize_gradient_conflicts(cosine_matrix: Sequence[Sequence[float]]) -> Dict[str, float]:
    cosine_matrix = np.asarray(cosine_matrix, dtype=np.float32)
    if cosine_matrix.size == 0 or cosine_matrix.shape[0] <= 1:
        return {
            'mean_pairwise_cosine': 1.0,
            'min_pairwise_cosine': 1.0,
            'conflict_rate': 0.0,
        }

    pairwise = cosine_matrix[np.triu_indices(cosine_matrix.shape[0], k=1)]
    return {
        'mean_pairwise_cosine': float(np.mean(pairwise)),
        'min_pairwise_cosine': float(np.min(pairwise)),
        'conflict_rate': float(np.mean(pairwise < 0.0)),
    }


def plot_gradient_conflict_dashboard(
    component_names: List[str],
    cosine_matrix: Sequence[Sequence[float]],
    component_norms: Sequence[float],
    component_shares: Sequence[float],
    step: Optional[int] = None,
    colors: Optional[Dict[str, str]] = None,
):
    """Create a compact dashboard for gradient composition and pairwise conflicts."""
    colors = {**DEFAULT_COLORS, **(colors or {})}
    cosine_matrix = np.asarray(cosine_matrix, dtype=np.float32)
    component_norms = np.asarray(component_norms, dtype=np.float32)
    component_shares = np.asarray(component_shares, dtype=np.float32)

    fig = plt.figure(figsize=(14, 4.5))
    gs = GridSpec(1, 3, width_ratios=[1.6, 1.0, 1.0], figure=fig)

    ax_heatmap = fig.add_subplot(gs[0, 0])
    image = ax_heatmap.imshow(cosine_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    ax_heatmap.set_xticks(np.arange(len(component_names)))
    ax_heatmap.set_yticks(np.arange(len(component_names)))
    ax_heatmap.set_xticklabels(component_names, rotation=35, ha='right')
    ax_heatmap.set_yticklabels(component_names)
    ax_heatmap.set_title('Gradient Cosine Matrix')
    if len(component_names) <= 8:
        for row_idx in range(cosine_matrix.shape[0]):
            for col_idx in range(cosine_matrix.shape[1]):
                value = float(cosine_matrix[row_idx, col_idx])
                ax_heatmap.text(
                    col_idx,
                    row_idx,
                    f'{value:.2f}',
                    ha='center',
                    va='center',
                    color='white' if abs(value) > 0.45 else 'black',
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.04)

    ax_norm = fig.add_subplot(gs[0, 1])
    ax_norm.bar(component_names, component_norms, color=colors['primary'], alpha=0.85)
    ax_norm.set_title('Component Grad Norms')
    ax_norm.set_ylabel('L2 Norm')
    ax_norm.tick_params(axis='x', rotation=35)
    ax_norm.grid(True, alpha=0.25, axis='y')

    ax_share = fig.add_subplot(gs[0, 2])
    ax_share.bar(component_names, component_shares, color=colors['secondary'], alpha=0.85)
    ax_share.set_title('Component Grad Shares')
    ax_share.set_ylabel('Share')
    ax_share.set_ylim(0.0, min(max(float(component_shares.max()) * 1.15, 0.1), 1.0))
    ax_share.tick_params(axis='x', rotation=35)
    ax_share.grid(True, alpha=0.25, axis='y')

    summary = summarize_gradient_conflicts(cosine_matrix)
    if step is None:
        title = (
            'Gradient Diagnostics | '
            f"mean cosine={summary['mean_pairwise_cosine']:.3f}, "
            f"min cosine={summary['min_pairwise_cosine']:.3f}, "
            f"conflict rate={summary['conflict_rate']:.2%}"
        )
    else:
        title = (
            f'Gradient Diagnostics (step {step}) | '
            f"mean cosine={summary['mean_pairwise_cosine']:.3f}, "
            f"min cosine={summary['min_pairwise_cosine']:.3f}, "
            f"conflict rate={summary['conflict_rate']:.2%}"
        )
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_gradient_influence_dashboard(
    component_names: List[str],
    group_labels: Sequence[str],
    component_group_shares: Sequence[Sequence[float]],
    group_component_shares: Sequence[Sequence[float]],
    group_total_shares: Sequence[float],
    group_abs_grad_quantiles: Mapping[str, Sequence[float]],
    step: Optional[int] = None,
    colors: Optional[Dict[str, str]] = None,
):
    """Visualize which losses drive each architecture block and total gradient spread."""
    colors = {**DEFAULT_COLORS, **(colors or {})}
    component_group_shares_array = np.asarray(component_group_shares, dtype=np.float32)
    group_component_shares_array = np.asarray(group_component_shares, dtype=np.float32)
    group_total_shares_array = np.asarray(group_total_shares, dtype=np.float32)
    quantiles = {
        key: np.asarray(values, dtype=np.float32)
        for key, values in group_abs_grad_quantiles.items()
    }

    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0], figure=fig)

    ax_component = fig.add_subplot(gs[0, 0])
    component_image = ax_component.imshow(
        component_group_shares_array,
        aspect='auto',
        cmap='YlOrRd',
        vmin=0.0,
        vmax=_resolve_heatmap_vmax(component_group_shares_array),
    )
    ax_component.set_xticks(np.arange(len(group_labels)))
    ax_component.set_yticks(np.arange(len(component_names)))
    ax_component.set_xticklabels(group_labels, rotation=50, ha='right')
    ax_component.set_yticklabels(component_names)
    ax_component.set_title('Per-Loss Influence on Architecture Blocks')
    ax_component.set_xlabel('Architecture block')
    ax_component.set_ylabel('Loss component')
    if component_group_shares_array.size and component_group_shares_array.shape[0] * component_group_shares_array.shape[1] <= 60:
        for row_index in range(component_group_shares_array.shape[0]):
            for col_index in range(component_group_shares_array.shape[1]):
                value = float(component_group_shares_array[row_index, col_index])
                ax_component.text(
                    col_index,
                    row_index,
                    f'{value:.2f}',
                    ha='center',
                    va='center',
                    color='black' if value < 0.55 else 'white',
                    fontsize=8,
                )
    fig.colorbar(component_image, ax=ax_component, fraction=0.046, pad=0.04)

    ax_group = fig.add_subplot(gs[1, 0])
    group_image = ax_group.imshow(
        group_component_shares_array.T,
        aspect='auto',
        cmap='PuBuGn',
        vmin=0.0,
        vmax=_resolve_heatmap_vmax(group_component_shares_array),
    )
    ax_group.set_xticks(np.arange(len(component_names)))
    ax_group.set_yticks(np.arange(len(group_labels)))
    ax_group.set_xticklabels(component_names, rotation=25, ha='right')
    ax_group.set_yticklabels(group_labels)
    ax_group.set_title('Per-Block Loss Composition')
    ax_group.set_xlabel('Loss component')
    ax_group.set_ylabel('Architecture block')
    fig.colorbar(group_image, ax=ax_group, fraction=0.046, pad=0.04)

    y_positions = np.arange(len(group_labels))

    ax_total = fig.add_subplot(gs[0, 1])
    ax_total.barh(y_positions, group_total_shares_array, color=colors['primary'], alpha=0.88)
    ax_total.set_yticks(y_positions)
    ax_total.set_yticklabels(group_labels)
    ax_total.invert_yaxis()
    ax_total.set_xlim(0.0, min(max(float(group_total_shares_array.max(initial=0.0)) * 1.15, 0.1), 1.0))
    ax_total.set_xlabel('Share of total gradient norm')
    ax_total.set_title('Total Gradient Share by Block')
    ax_total.grid(True, axis='x', alpha=0.25)

    ax_distribution = fig.add_subplot(gs[1, 1])
    p05 = np.maximum(quantiles.get('p05', np.zeros_like(group_total_shares_array)), 1e-12)
    p25 = np.maximum(quantiles.get('p25', np.zeros_like(group_total_shares_array)), 1e-12)
    p50 = np.maximum(quantiles.get('p50', np.zeros_like(group_total_shares_array)), 1e-12)
    p75 = np.maximum(quantiles.get('p75', np.zeros_like(group_total_shares_array)), 1e-12)
    p95 = np.maximum(quantiles.get('p95', np.zeros_like(group_total_shares_array)), 1e-12)
    ax_distribution.hlines(y_positions, p05, p95, color=colors['primary'], alpha=0.45, linewidth=1.6)
    ax_distribution.hlines(y_positions, p25, p75, color=colors['secondary'], linewidth=4.2)
    ax_distribution.scatter(p50, y_positions, color=colors['danger'], s=22, zorder=3)
    ax_distribution.set_xscale('log')
    ax_distribution.set_yticks(y_positions)
    ax_distribution.set_yticklabels(group_labels)
    ax_distribution.invert_yaxis()
    ax_distribution.set_xlabel('|grad| quantiles (log scale)')
    ax_distribution.set_title('Total Gradient Distribution by Block')
    ax_distribution.grid(True, axis='x', alpha=0.25)

    top_summary = _top_group_summary(group_labels, group_total_shares_array)
    if step is None:
        title = 'Gradient Influence Dashboard'
    else:
        title = f'Gradient Influence Dashboard (step {step})'
    if top_summary:
        title = f'{title} | top blocks: {top_summary}'
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    return fig
