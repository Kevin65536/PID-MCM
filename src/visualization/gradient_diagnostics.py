"""Gradient diagnostics visualizations for multi-loss training."""

from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


DEFAULT_COLORS: Dict[str, str] = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
}


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