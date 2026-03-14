"""
Comprehensive Visualization Suite for UMAP Experiments

Covers:
  1. Training dynamics: multi-objective loss decomposition, LR schedule, gradient norms
  2. Classification analysis: confusion matrix, per-class metrics, ROC curves
  3. Subject-level analysis: per-subject accuracy heatmap, variance analysis
  4. Cross-modal analysis: attention heatmaps, fusion gate weights, embedding t-SNE
  5. Architecture analysis: per-layer gradient flow, FFN branch utilization
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch


# ── Styling ──────────────────────────────────────────────────────────────────

COLORS = {
    'eeg': '#2196F3',       # blue
    'fnirs': '#FF9800',     # orange
    'fusion': '#4CAF50',    # green
    'train': '#2196F3',
    'val': '#FF5722',
    'test': '#9C27B0',
    'class0': '#42A5F5',
    'class1': '#EF5350',
    'accent': '#7C4DFF',
    'grid': '#E0E0E0',
}

def _style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PRETRAIN PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_pretrain_loss_decomposition(history: dict, save_dir: Path):
    """
    Multi-panel plot showing individual loss components + stacked area chart
    showing relative contribution of each pretraining objective over time.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    train_h = history.get('train', {})
    val_h = history.get('val', {})
    n_epochs = min(len(train_h.get('loss_total', [])), len(val_h.get('loss_total', [])))
    if n_epochs == 0:
        return
    # Truncate all series to same length
    for d in (train_h, val_h):
        for k in list(d.keys()):
            d[k] = d[k][:n_epochs]
    epochs = range(n_epochs)

    # (0,0) Total loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, train_h['loss_total'], color=COLORS['train'], label='train', linewidth=1.5)
    ax.plot(epochs, val_h['loss_total'], color=COLORS['val'], label='val', linestyle='--', linewidth=1.5)
    ax.fill_between(epochs, train_h['loss_total'], val_h['loss_total'], alpha=0.1, color=COLORS['val'])
    _style_ax(ax, 'Total Loss', 'Epoch', 'Loss')
    ax.legend(fontsize=8)

    # (0,1) Individual losses overlay
    ax = fig.add_subplot(gs[0, 1])
    for key, color, label in [('loss_con', COLORS['eeg'], 'Contrastive'),
                               ('loss_mat', COLORS['fnirs'], 'Matching'),
                               ('lm_loss', COLORS['fusion'], 'Generation')]:
        ax.plot(epochs, train_h[key], color=color, label=f'{label} (train)', linewidth=1.5)
        ax.plot(epochs, val_h[key], color=color, linestyle='--', alpha=0.6, linewidth=1)
    _style_ax(ax, 'Per-Task Losses', 'Epoch', 'Loss')
    ax.legend(fontsize=7)

    # (0,2) Loss ratio (relative contribution)
    ax = fig.add_subplot(gs[0, 2])
    con = np.array(train_h['loss_con'])
    mat = np.array(train_h['loss_mat'])
    gen = np.array(train_h['lm_loss'])
    total = con + mat + gen + 1e-8
    ax.stackplot(epochs, con / total * 100, mat / total * 100, gen / total * 100,
                 labels=['Contrastive', 'Matching', 'Generation'],
                 colors=[COLORS['eeg'], COLORS['fnirs'], COLORS['fusion']], alpha=0.8)
    _style_ax(ax, 'Loss Component Ratios (%)', 'Epoch', 'Contribution %')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 100)

    # (1,0) Train-Val gap (generalization)
    ax = fig.add_subplot(gs[1, 0])
    gap = np.array(val_h['loss_total']) - np.array(train_h['loss_total'])
    ax.plot(epochs, gap, color=COLORS['accent'], linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.fill_between(epochs, 0, gap, where=gap > 0, alpha=0.15, color=COLORS['val'], label='Overfitting')
    ax.fill_between(epochs, 0, gap, where=gap <= 0, alpha=0.15, color=COLORS['fusion'], label='Underfitting')
    _style_ax(ax, 'Generalization Gap (Val - Train)', 'Epoch', 'Loss Gap')
    ax.legend(fontsize=8)

    # (1,1) Per-task val loss normalized (relative improvement from epoch 0)
    ax = fig.add_subplot(gs[1, 1])
    for key, color, label in [('loss_con', COLORS['eeg'], 'Contrastive'),
                               ('loss_mat', COLORS['fnirs'], 'Matching'),
                               ('lm_loss', COLORS['fusion'], 'Generation')]:
        vals = np.array(val_h[key])
        if vals[0] > 0:
            normalized = vals / vals[0]
            ax.plot(epochs, normalized, color=color, label=label, linewidth=1.5)
    ax.axhline(1.0, color='black', linewidth=0.5, linestyle='--')
    _style_ax(ax, 'Normalized Val Loss (relative to epoch 0)', 'Epoch', 'Relative Loss')
    ax.legend(fontsize=8)

    # (1,2) Loss convergence speed (log scale)
    ax = fig.add_subplot(gs[1, 2])
    ax.semilogy(epochs, train_h['loss_total'], color=COLORS['train'], label='train', linewidth=1.5)
    ax.semilogy(epochs, val_h['loss_total'], color=COLORS['val'], label='val', linestyle='--', linewidth=1.5)
    _style_ax(ax, 'Loss (Log Scale)', 'Epoch', 'Log Loss')
    ax.legend(fontsize=8)

    fig.suptitle('UMAP Pretraining — Loss Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_dir / 'pretrain_loss_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_pretrain_loss_correlation(history: dict, save_dir: Path):
    """Scatter plots showing correlation between loss components."""
    train_h = history.get('train', {})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    pairs = [('loss_con', 'loss_mat', 'CON vs MAT'),
             ('loss_con', 'lm_loss', 'CON vs GEN'),
             ('loss_mat', 'lm_loss', 'MAT vs GEN')]

    for ax, (k1, k2, title) in zip(axes, pairs):
        x, y = np.array(train_h[k1]), np.array(train_h[k2])
        scatter = ax.scatter(x, y, c=range(len(x)), cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Epoch', shrink=0.8)

        # Correlation coefficient
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        _style_ax(ax, title, k1, k2)

    fig.suptitle('Loss Component Correlations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'pretrain_loss_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 2. FINETUNE CLASSIFICATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray,
                          class_names: List[str], save_dir: Path, title: str = ''):
    """Publication-quality confusion matrix with percentages."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, data, fmt, subtitle in [(ax1, cm, 'd', 'Counts'), (ax2, cm_pct, '.1f', 'Percentages (%)')]:
        im = ax.imshow(data, interpolation='nearest', cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = 'white' if val > data.max() / 2 else 'black'
                ax.text(j, i, f'{val:{fmt}}', ha='center', va='center', color=color, fontsize=12)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        _style_ax(ax, subtitle, 'Predicted', 'True')

    fig.suptitle(f'Confusion Matrix — {title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_classification_dashboard(history: dict, preds: np.ndarray, labels: np.ndarray,
                                   subj_accs: dict, class_names: List[str],
                                   save_dir: Path, title: str = ''):
    """Comprehensive classification dashboard: curves, CM, per-subject, confidence."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    train_h = history.get('train', {})
    val_h = history.get('val', {})
    epochs = range(len(train_h.get('loss', [])))

    # (0,0) Loss curves
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, train_h['loss'], color=COLORS['train'], label='train', linewidth=1.5)
    if 'loss' in val_h:
        ax.plot(epochs, val_h['loss'], color=COLORS['val'], label='val', linestyle='--', linewidth=1.5)
    _style_ax(ax, 'Loss', 'Epoch', 'CE Loss')
    ax.legend(fontsize=8)

    # (0,1) Accuracy curves
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, [a * 100 for a in train_h['accuracy']], color=COLORS['train'], label='train', linewidth=1.5)
    if 'accuracy' in val_h:
        ax.plot(epochs, [a * 100 for a in val_h['accuracy']], color=COLORS['val'], label='val', linestyle='--', linewidth=1.5)
    ax.axhline(50, color='gray', linewidth=0.8, linestyle=':', label='Chance')
    _style_ax(ax, 'Accuracy', 'Epoch', 'Accuracy %')
    ax.legend(fontsize=8)

    # (0,2) Train-Val accuracy gap
    ax = fig.add_subplot(gs[0, 2])
    if 'accuracy' in val_h:
        train_acc = np.array(train_h['accuracy']) * 100
        val_acc = np.array(val_h['accuracy']) * 100
        gap = train_acc - val_acc
        ax.plot(epochs, gap, color=COLORS['accent'], linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.fill_between(epochs, 0, gap, where=gap > 0, alpha=0.15, color=COLORS['val'])
        _style_ax(ax, 'Accuracy Gap (Train - Val)', 'Epoch', 'Gap %')

    # (1,0) Confusion matrix
    if len(preds) > 0:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                color = 'white' if cm_pct[i, j] > 50 else 'black'
                ax.text(j, i, f'{cm_pct[i,j]:.1f}%\n({cm[i,j]})',
                        ha='center', va='center', color=color, fontsize=9)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        _style_ax(ax, 'Test Confusion Matrix', 'Predicted', 'True')

    # (1,1) Per-class precision / recall / F1
    if len(preds) > 0:
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f, s = precision_recall_fscore_support(labels, preds, labels=range(len(class_names)))
        ax = fig.add_subplot(gs[1, 1])
        x_pos = np.arange(len(class_names))
        w = 0.25
        ax.bar(x_pos - w, p * 100, w, label='Precision', color=COLORS['eeg'], alpha=0.8)
        ax.bar(x_pos, r * 100, w, label='Recall', color=COLORS['fnirs'], alpha=0.8)
        ax.bar(x_pos + w, f * 100, w, label='F1', color=COLORS['fusion'], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_names, fontsize=9)
        ax.axhline(50, color='gray', linewidth=0.8, linestyle=':')
        _style_ax(ax, 'Per-Class Metrics', 'Class', 'Score %')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)

    # (1,2) Per-subject accuracy bar chart
    if subj_accs:
        ax = fig.add_subplot(gs[1, 2])
        subjects = sorted(subj_accs.keys(), key=int)
        accs = [subj_accs[s] * 100 for s in subjects]
        colors = [COLORS['fusion'] if a > 50 else COLORS['val'] for a in accs]
        bars = ax.bar(range(len(subjects)), accs, color=colors, alpha=0.8, edgecolor='white')
        ax.axhline(50, color='gray', linewidth=0.8, linestyle=':', label='Chance')
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels([f'S{s}' for s in subjects], fontsize=8)
        mean_acc = np.mean(accs)
        ax.axhline(mean_acc, color=COLORS['accent'], linewidth=1, linestyle='--',
                   label=f'Mean={mean_acc:.1f}%')
        _style_ax(ax, 'Per-Subject Test Accuracy', 'Subject', 'Accuracy %')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)

    # (2, 0:2) Prediction distribution
    if len(preds) > 0:
        ax = fig.add_subplot(gs[2, 0])
        for cls_idx, (cls_name, color) in enumerate(zip(class_names, [COLORS['class0'], COLORS['class1']])):
            mask = labels == cls_idx
            pred_for_class = preds[mask]
            ax.hist(pred_for_class, bins=range(len(class_names) + 1), alpha=0.6,
                    color=color, label=f'True: {cls_name}', edgecolor='white')
        _style_ax(ax, 'Prediction Distribution', 'Predicted Class', 'Count')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.legend(fontsize=8)

    # (2,1) Accuracy over epochs with best marker
    ax = fig.add_subplot(gs[2, 1])
    if 'accuracy' in val_h:
        val_acc = np.array(val_h['accuracy']) * 100
        ax.plot(epochs, val_acc, color=COLORS['val'], linewidth=1.5, label='Val Accuracy')
        best_idx = np.argmax(val_acc)
        ax.plot(best_idx, val_acc[best_idx], 'o', color=COLORS['accent'], markersize=10,
                label=f'Best: {val_acc[best_idx]:.1f}% @ ep{best_idx}', zorder=5)
        ax.axhline(50, color='gray', linewidth=0.8, linestyle=':')
        _style_ax(ax, 'Validation Accuracy Trajectory', 'Epoch', 'Accuracy %')
        ax.legend(fontsize=8)

    # (2,2) Summary text box
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    summary_lines = [f'[{title}]', '']
    if 'accuracy' in val_h:
        summary_lines.append(f'Best Val Acc: {max(val_h["accuracy"])*100:.2f}%')
    if len(preds) > 0:
        test_acc = (preds == labels).mean() * 100
        summary_lines.append(f'Test Acc: {test_acc:.2f}%')
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, preds, average='macro') * 100
        summary_lines.append(f'Test F1 Macro: {f1:.2f}%')
    if subj_accs:
        accs_list = [v * 100 for v in subj_accs.values()]
        summary_lines.append(f'Subject Acc Mean: {np.mean(accs_list):.1f}%')
        summary_lines.append(f'Subject Acc Std:  {np.std(accs_list):.1f}%')
        summary_lines.append(f'Subject Acc Range: [{min(accs_list):.1f}, {max(accs_list):.1f}]%')

    text = '\n'.join(summary_lines)
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5', edgecolor='#E0E0E0'))

    fig.suptitle(f'Classification Dashboard — {title}', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_dir / 'classification_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 3. CROSS-MODAL ATTENTION & FUSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_attention_maps(model, sample_batch, device, mode='ft'):
    """Extract attention weights from all layers for a batch."""
    model.eval()

    # Determine model type
    is_pretrain = hasattr(model, 'UMAP')
    qformer = model.UMAP if is_pretrain else model.Qformer

    eeg = sample_batch['eeg'].to(device)
    fnirs = sample_batch['fnirs'].to(device)

    embed_dim = model.config.hidden_size

    # Compute embeddings
    eeg_embeds = model.eeg_embeds(eeg)
    eye_embeds = model.eye_embeds(fnirs)

    eeg_cls = model.eeg_cls_token.expand(eeg.size(0), -1, -1)
    eye_cls = model.eye_cls_token.expand(fnirs.size(0), -1, -1)

    eeg_embeds = torch.cat([eeg_cls, eeg_embeds], dim=1)
    eye_embeds = torch.cat([eye_cls, eye_embeds], dim=1)

    eeg_embeds = eeg_embeds + model.eeg_type_embed + model.eeg_pos_emb
    eye_embeds = eye_embeds + model.eye_type_embed + model.eye_pos_emb

    seq_len = eeg_embeds.size(1)
    batch_size = eeg.size(0)

    # Forward through Q-Former with output_attentions=True
    output = qformer(
        query_embeds=eye_embeds,
        mode=mode,
        text_embeds=eeg_embeds,
        attention_mask=torch.ones((batch_size, seq_len * 2), device=device),
        return_dict=True,
        eye_first=True,
        output_attentions=True,
        output_hidden_states=True,
    )

    return {
        'attentions': output.attentions,        # tuple of (batch, heads, seq, seq)
        'hidden_states': output.hidden_states,  # tuple of (batch, seq, dim)
        'seq_length': seq_len,
    }


def plot_attention_heatmaps(attn_data: dict, save_dir: Path, title: str = ''):
    """
    Plot cross-modal attention heatmaps for each layer and head.
    Shows how EEG attends to fNIRS and vice versa.
    """
    attentions = attn_data['attentions']
    seq_len = attn_data['seq_length']
    n_layers = len(attentions)

    if attentions is None or len(attentions) == 0:
        return

    n_heads = attentions[0].shape[1]

    fig, axes = plt.subplots(n_layers, 3, figsize=(15, 4 * n_layers))
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    for layer_idx in range(n_layers):
        attn = attentions[layer_idx][0].cpu().numpy()  # first sample, (heads, seq, seq)
        attn_avg = attn.mean(axis=0)  # average over heads

        # Full attention map
        ax = axes[layer_idx, 0]
        im = ax.imshow(attn_avg, cmap='viridis', aspect='auto')
        ax.axhline(seq_len - 0.5, color='red', linewidth=1, linestyle='--')
        ax.axvline(seq_len - 0.5, color='red', linewidth=1, linestyle='--')
        plt.colorbar(im, ax=ax, shrink=0.8)
        _style_ax(ax, f'Layer {layer_idx} — Full Attention', 'Key position', 'Query position')

        # Cross-modal: fNIRS query → EEG key (top-right quadrant)
        ax = axes[layer_idx, 1]
        cross_fnirs2eeg = attn_avg[:seq_len, seq_len:]  # fNIRS queries attending to EEG keys
        im = ax.imshow(cross_fnirs2eeg, cmap='magma', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        _style_ax(ax, f'Layer {layer_idx} — fNIRS→EEG', 'EEG key', 'fNIRS query')

        # Cross-modal: EEG query → fNIRS key (bottom-left quadrant)
        ax = axes[layer_idx, 2]
        cross_eeg2fnirs = attn_avg[seq_len:, :seq_len]  # EEG queries attending to fNIRS keys
        im = ax.imshow(cross_eeg2fnirs, cmap='magma', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        _style_ax(ax, f'Layer {layer_idx} — EEG→fNIRS', 'fNIRS key', 'EEG query')

    fig.suptitle(f'Cross-Modal Attention Heatmaps — {title}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'attention_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_summary(attn_data: dict, save_dir: Path, title: str = ''):
    """
    Summary of cross-modal vs intra-modal attention strength across layers.
    Shows whether the model learns to fuse modalities.
    """
    attentions = attn_data['attentions']
    seq_len = attn_data['seq_length']

    if attentions is None or len(attentions) == 0:
        return

    n_layers = len(attentions)
    # Compute average attention mass in each quadrant per layer
    intra_eeg, intra_fnirs, cross_fnirs2eeg, cross_eeg2fnirs = [], [], [], []

    for layer_idx in range(n_layers):
        attn = attentions[layer_idx].mean(dim=(0, 1)).cpu().numpy()  # avg over batch & heads
        # Quadrants: [fNIRS, EEG] layout (eye_first=True)
        intra_fnirs.append(attn[:seq_len, :seq_len].mean())
        cross_fnirs2eeg.append(attn[:seq_len, seq_len:].mean())
        cross_eeg2fnirs.append(attn[seq_len:, :seq_len].mean())
        intra_eeg.append(attn[seq_len:, seq_len:].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    x = np.arange(n_layers)
    w = 0.2
    ax1.bar(x - 1.5*w, intra_fnirs, w, label='Intra-fNIRS', color=COLORS['fnirs'], alpha=0.8)
    ax1.bar(x - 0.5*w, cross_fnirs2eeg, w, label='fNIRS→EEG', color='#FF6F00', alpha=0.8)
    ax1.bar(x + 0.5*w, cross_eeg2fnirs, w, label='EEG→fNIRS', color='#1565C0', alpha=0.8)
    ax1.bar(x + 1.5*w, intra_eeg, w, label='Intra-EEG', color=COLORS['eeg'], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Layer {i}' for i in x])
    _style_ax(ax1, 'Attention Mass by Quadrant', 'Layer', 'Mean Attention Weight')
    ax1.legend(fontsize=8)

    # Stacked percentage
    totals = np.array(intra_fnirs) + np.array(cross_fnirs2eeg) + np.array(cross_eeg2fnirs) + np.array(intra_eeg)
    ax2.bar(x, np.array(intra_fnirs)/totals*100, label='Intra-fNIRS', color=COLORS['fnirs'])
    bottom = np.array(intra_fnirs)/totals*100
    ax2.bar(x, np.array(cross_fnirs2eeg)/totals*100, bottom=bottom, label='fNIRS→EEG', color='#FF6F00')
    bottom += np.array(cross_fnirs2eeg)/totals*100
    ax2.bar(x, np.array(cross_eeg2fnirs)/totals*100, bottom=bottom, label='EEG→fNIRS', color='#1565C0')
    bottom += np.array(cross_eeg2fnirs)/totals*100
    ax2.bar(x, np.array(intra_eeg)/totals*100, bottom=bottom, label='Intra-EEG', color=COLORS['eeg'])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Layer {i}' for i in x])
    _style_ax(ax2, 'Attention Distribution (%)', 'Layer', 'Percentage')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylim(0, 105)

    fig.suptitle(f'Cross-Modal Attention Analysis — {title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'attention_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_head_attention(attn_data: dict, save_dir: Path, title: str = ''):
    """Show each attention head's cross-modal vs intra-modal preference."""
    attentions = attn_data['attentions']
    seq_len = attn_data['seq_length']

    if attentions is None or len(attentions) == 0:
        return

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for layer_idx in range(n_layers):
        attn = attentions[layer_idx].mean(dim=0).cpu().numpy()  # avg over batch: (heads, seq, seq)
        cross_ratios = []
        for h in range(n_heads):
            a = attn[h]
            cross = (a[:seq_len, seq_len:].mean() + a[seq_len:, :seq_len].mean()) / 2
            intra = (a[:seq_len, :seq_len].mean() + a[seq_len:, seq_len:].mean()) / 2
            cross_ratios.append(cross / (cross + intra + 1e-8) * 100)

        ax = axes[layer_idx]
        colors_h = [COLORS['fusion'] if r > 50 else COLORS['eeg'] for r in cross_ratios]
        ax.bar(range(n_heads), cross_ratios, color=colors_h, alpha=0.8, edgecolor='white')
        ax.axhline(50, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f'H{i}' for i in range(n_heads)])
        _style_ax(ax, f'Layer {layer_idx} — Cross-Modal %', 'Head', 'Cross-Modal Attn %')
        ax.set_ylim(0, 100)

    fig.suptitle(f'Per-Head Cross-Modal Attention — {title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'per_head_attention.png', dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 4. EMBEDDING / REPRESENTATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=500):
    """Extract CLS embeddings from model for visualization."""
    model.eval()
    is_pretrain = hasattr(model, 'UMAP')
    qformer = model.UMAP if is_pretrain else model.Qformer

    all_eeg_embeds = []
    all_fnirs_embeds = []
    all_labels = []
    all_subjects = []
    n = 0

    for batch in dataloader:
        eeg = batch['eeg'].to(device)
        fnirs = batch['fnirs'].to(device)

        eeg_e = model.eeg_embeds(eeg)
        eye_e = model.eye_embeds(fnirs)

        eeg_cls = model.eeg_cls_token.expand(eeg.size(0), -1, -1)
        eye_cls = model.eye_cls_token.expand(fnirs.size(0), -1, -1)

        eeg_e = torch.cat([eeg_cls, eeg_e], dim=1)
        eye_e = torch.cat([eye_cls, eye_e], dim=1)

        eeg_e = eeg_e + model.eeg_type_embed + model.eeg_pos_emb
        eye_e = eye_e + model.eye_type_embed + model.eye_pos_emb

        seq_len = eeg_e.size(1)
        batch_size = eeg.size(0)

        output = qformer(
            query_embeds=eye_e,
            mode='ft' if not is_pretrain else 'con',
            text_embeds=eeg_e,
            attention_mask=torch.ones((batch_size, seq_len * 2), device=device),
            return_dict=True,
            eye_first=True,
        )

        last_hs = output.last_hidden_state
        fnirs_cls = last_hs[:, 0, :].cpu().numpy()       # fNIRS CLS
        eeg_cls_out = last_hs[:, seq_len, :].cpu().numpy() # EEG CLS

        all_fnirs_embeds.append(fnirs_cls)
        all_eeg_embeds.append(eeg_cls_out)
        all_labels.extend(batch['label'].numpy() if isinstance(batch['label'], torch.Tensor) else batch['label'])
        all_subjects.extend([int(s) for s in batch['subject_id']])

        n += batch_size
        if n >= max_samples:
            break

    return {
        'eeg_cls': np.concatenate(all_eeg_embeds),
        'fnirs_cls': np.concatenate(all_fnirs_embeds),
        'labels': np.array(all_labels),
        'subjects': np.array(all_subjects),
    }


def plot_embedding_tsne(embed_data: dict, save_dir: Path, title: str = ''):
    """t-SNE visualization of CLS embeddings, colored by class and subject."""
    from sklearn.manifold import TSNE

    eeg_cls = embed_data['eeg_cls']
    fnirs_cls = embed_data['fnirs_cls']
    labels = embed_data['labels']
    subjects = embed_data['subjects']

    # Combine modalities
    combined = np.concatenate([eeg_cls, fnirs_cls], axis=0)
    modality_labels = np.array(['EEG'] * len(eeg_cls) + ['fNIRS'] * len(fnirs_cls))
    combined_labels = np.concatenate([labels, labels])
    combined_subjects = np.concatenate([subjects, subjects])

    if combined.shape[0] < 5:
        return

    perplexity = min(30, combined.shape[0] // 3)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(combined)

    n_eeg = len(eeg_cls)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # By modality
    ax = axes[0]
    ax.scatter(coords[:n_eeg, 0], coords[:n_eeg, 1], c=COLORS['eeg'], s=15, alpha=0.6, label='EEG')
    ax.scatter(coords[n_eeg:, 0], coords[n_eeg:, 1], c=COLORS['fnirs'], s=15, alpha=0.6, label='fNIRS')
    _style_ax(ax, 'By Modality', 't-SNE 1', 't-SNE 2')
    ax.legend(fontsize=8)

    # By class
    ax = axes[1]
    class_names = ['Left MI', 'Right MI']
    for cls_idx in range(2):
        mask = combined_labels == cls_idx
        ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.6,
                   label=class_names[cls_idx], color=[COLORS['class0'], COLORS['class1']][cls_idx])
    _style_ax(ax, 'By Class', 't-SNE 1', 't-SNE 2')
    ax.legend(fontsize=8)

    # By subject
    ax = axes[2]
    unique_subj = np.unique(combined_subjects)
    cmap = plt.cm.get_cmap('tab20', len(unique_subj))
    for i, s in enumerate(unique_subj):
        mask = combined_subjects == s
        ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.6,
                   color=cmap(i), label=f'S{s}')
    _style_ax(ax, 'By Subject', 't-SNE 1', 't-SNE 2')
    ax.legend(fontsize=6, ncol=2, loc='best')

    fig.suptitle(f'Embedding t-SNE — {title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'embedding_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_modality_alignment(embed_data: dict, save_dir: Path, title: str = ''):
    """
    Analyze whether EEG and fNIRS embeddings are aligned in the shared space.
    - Cosine similarity distribution between paired EEG-fNIRS
    - vs random (unpaired) baseline
    """
    eeg_cls = embed_data['eeg_cls']
    fnirs_cls = embed_data['fnirs_cls']
    labels = embed_data['labels']

    # Normalize
    eeg_norm = eeg_cls / (np.linalg.norm(eeg_cls, axis=1, keepdims=True) + 1e-8)
    fnirs_norm = fnirs_cls / (np.linalg.norm(fnirs_cls, axis=1, keepdims=True) + 1e-8)

    # Paired cosine similarity
    paired_sim = (eeg_norm * fnirs_norm).sum(axis=1)

    # Random pairing
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(fnirs_norm))
    random_sim = (eeg_norm * fnirs_norm[perm]).sum(axis=1)

    # Same-class vs cross-class
    same_mask = labels[:, None] == labels[None, :]
    cos_matrix = eeg_norm @ fnirs_norm.T
    same_class_sim = cos_matrix[same_mask & ~np.eye(len(labels), dtype=bool)]
    cross_class_sim = cos_matrix[~same_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Paired vs random
    ax1.hist(paired_sim, bins=30, alpha=0.7, color=COLORS['fusion'], label='Paired', density=True)
    ax1.hist(random_sim, bins=30, alpha=0.5, color='gray', label='Random', density=True)
    ax1.axvline(paired_sim.mean(), color=COLORS['fusion'], linewidth=2, linestyle='--',
                label=f'Paired mean={paired_sim.mean():.3f}')
    ax1.axvline(random_sim.mean(), color='gray', linewidth=2, linestyle='--',
                label=f'Random mean={random_sim.mean():.3f}')
    _style_ax(ax1, 'EEG-fNIRS Alignment', 'Cosine Similarity', 'Density')
    ax1.legend(fontsize=8)

    # Same class vs cross class
    if len(same_class_sim) > 0 and len(cross_class_sim) > 0:
        ax2.hist(same_class_sim, bins=30, alpha=0.7, color=COLORS['fusion'],
                 label=f'Same class (μ={same_class_sim.mean():.3f})', density=True)
        ax2.hist(cross_class_sim, bins=30, alpha=0.5, color=COLORS['val'],
                 label=f'Diff class (μ={cross_class_sim.mean():.3f})', density=True)
        _style_ax(ax2, 'Class-Conditional Alignment', 'Cosine Similarity', 'Density')
        ax2.legend(fontsize=8)

    fig.suptitle(f'Modality Alignment Analysis — {title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'modality_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 5. ARCHITECTURE ANALYSIS (MoE-like FFN Branch Utilization)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_ffn_branches(model, dataloader, device, max_batches=10):
    """
    Measure activation statistics of the modality-specific FFN branches
    (EEG FFN, fNIRS FFN, Fusion FFN) and the SeqFusion gating weights.

    UMAP has per-layer: intermediate_eeg, intermediate_eye, intermediate_fusion
    with SeqFusion gating between modality-specific and fusion branches.
    """
    model.eval()
    is_pretrain = hasattr(model, 'UMAP')
    qformer = model.UMAP if is_pretrain else model.Qformer
    encoder = qformer.encoder

    n_layers = len(encoder.layer)
    # Collect SeqFusion gate weights
    gate_stats = {i: [] for i in range(n_layers - 1)}  # last layer has no MoE structure

    # Hook into SeqFusion to capture gate weights
    hooks = []
    gate_captures = {i: [] for i in range(n_layers - 1)}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # SeqFusion computes alpha = softmax([o1, o2]) where o1=seq1@w, o2=seq2@w
            seq1, seq2 = input
            w = module.weight
            o1 = seq1 @ w  # (batch, seq, 1)
            o2 = seq2 @ w
            o = torch.cat([o1, o2], dim=-1)
            alpha = torch.softmax(o, dim=-1)  # (batch, seq, 2)
            gate_captures[layer_idx].append(alpha.cpu())
        return hook_fn

    for i in range(n_layers - 1):
        layer = encoder.layer[i]
        if hasattr(layer, 'seq_fusion'):
            h = layer.seq_fusion.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Run forward passes
    n = 0
    for batch in dataloader:
        eeg = batch['eeg'].to(device)
        fnirs = batch['fnirs'].to(device)
        if is_pretrain:
            model(eeg, fnirs)
        else:
            model(eeg=eeg, eye=fnirs)
        n += 1
        if n >= max_batches:
            break

    # Remove hooks
    for h in hooks:
        h.remove()

    return gate_captures


def plot_fusion_gate_weights(gate_captures: dict, save_dir: Path, title: str = ''):
    """
    Visualize SeqFusion gate weights across layers.
    Gate weight [:,0] = modality-specific weight, [:,1] = fusion weight.
    """
    n_layers = len(gate_captures)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for layer_idx in range(n_layers):
        captures = gate_captures[layer_idx]
        if not captures:
            continue

        gates = torch.cat(captures, dim=0)  # (total_samples, seq, 2)
        modality_w = gates[:, :, 0].numpy().flatten()
        fusion_w = gates[:, :, 1].numpy().flatten()

        ax = axes[layer_idx]
        ax.hist(modality_w, bins=50, alpha=0.7, color=COLORS['eeg'],
                label=f'Modality (μ={modality_w.mean():.3f})', density=True)
        ax.hist(fusion_w, bins=50, alpha=0.7, color=COLORS['fusion'],
                label=f'Fusion (μ={fusion_w.mean():.3f})', density=True)
        ax.axvline(0.5, color='gray', linewidth=0.8, linestyle=':')
        _style_ax(ax, f'Layer {layer_idx} Gate Distribution', 'Weight', 'Density')
        ax.legend(fontsize=8)

    fig.suptitle(f'SeqFusion Gate Weights — {title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'fusion_gate_weights.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_gradient_flow(model, sample_batch, device):
    """Compute gradient norms per layer for a single batch."""
    model.train()
    # Ensure all parameters require grad
    for p in model.parameters():
        p.requires_grad_(True)
    is_pretrain = hasattr(model, 'UMAP')

    eeg = sample_batch['eeg'].to(device)
    fnirs = sample_batch['fnirs'].to(device)

    if is_pretrain:
        loss, _, _, _ = model(eeg, fnirs)
    else:
        labels = sample_batch['label'].to(device)
        logits = model(eeg=eeg, eye=fnirs)
        loss = F.cross_entropy(logits, labels)

    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    model.zero_grad()
    return grad_norms


def plot_gradient_flow(grad_norms: dict, save_dir: Path, title: str = ''):
    """Visualize gradient norms grouped by module type."""
    groups = {
        'EEG Embed': [], 'fNIRS Embed': [],
        'EEG FFN': [], 'fNIRS FFN': [], 'Fusion FFN': [],
        'Attention': [], 'CLS Head': [], 'Other': [],
    }

    for name, norm in grad_norms.items():
        if 'eeg_embeds' in name or 'eeg_cls' in name or 'eeg_pos' in name or 'eeg_type' in name:
            groups['EEG Embed'].append(norm)
        elif 'eye_embeds' in name or 'eye_cls' in name or 'eye_pos' in name or 'eye_type' in name:
            groups['fNIRS Embed'].append(norm)
        elif 'intermediate_eeg' in name or 'output_eeg' in name:
            groups['EEG FFN'].append(norm)
        elif 'intermediate_eye' in name or 'output_eye' in name:
            groups['fNIRS FFN'].append(norm)
        elif 'intermediate_fusion' in name or 'output_fusion' in name or 'seq_fusion' in name:
            groups['Fusion FFN'].append(norm)
        elif 'attention' in name:
            groups['Attention'].append(norm)
        elif 'cls_head' in name or 'cls_fusion' in name or 'itm_head' in name:
            groups['CLS Head'].append(norm)
        else:
            groups['Other'].append(norm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    data = []
    labels_list = []
    colors_list = []
    color_map = {
        'EEG Embed': COLORS['eeg'], 'fNIRS Embed': COLORS['fnirs'],
        'EEG FFN': COLORS['eeg'], 'fNIRS FFN': COLORS['fnirs'],
        'Fusion FFN': COLORS['fusion'], 'Attention': COLORS['accent'],
        'CLS Head': COLORS['test'], 'Other': 'gray',
    }
    for g_name, g_vals in groups.items():
        if g_vals:
            data.append(g_vals)
            labels_list.append(g_name)
            colors_list.append(color_map.get(g_name, 'gray'))

    bp = ax1.boxplot(data, labels=labels_list, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)
    _style_ax(ax1, 'Gradient Norms by Module', '', 'Gradient L2 Norm')

    # Mean gradient per group (bar)
    means = [np.mean(d) for d in data]
    ax2.bar(range(len(labels_list)), means, color=colors_list, alpha=0.8, edgecolor='white')
    ax2.set_xticks(range(len(labels_list)))
    ax2.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)
    _style_ax(ax2, 'Mean Gradient Norm per Module', '', 'Mean L2 Norm')

    fig.suptitle(f'Gradient Flow Analysis — {title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPARATIVE SUMMARY PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_experiment_comparison(results: Dict[str, dict], save_dir: Path):
    """
    Side-by-side comparison of multiple experiment runs.
    Shows accuracy, F1, and subject-level variance for each configuration.
    """
    exp_names = list(results.keys())
    n = len(exp_names)

    test_accs = [results[e].get('test_accuracy', 0) * 100 for e in exp_names]
    test_f1s = [results[e].get('test_f1_macro', 0) * 100 for e in exp_names]
    val_accs = [results[e].get('best_val_accuracy', 0) * 100 for e in exp_names]

    # Subject-level stats
    subj_means, subj_stds = [], []
    for e in exp_names:
        sa = results[e].get('subject_accs', {})
        if sa:
            vals = [v * 100 for v in sa.values()]
            subj_means.append(np.mean(vals))
            subj_stds.append(np.std(vals))
        else:
            subj_means.append(0)
            subj_stds.append(0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Accuracy comparison
    ax = axes[0]
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, val_accs, w, label='Val Acc', color=COLORS['val'], alpha=0.8)
    ax.bar(x + w/2, test_accs, w, label='Test Acc', color=COLORS['test'], alpha=0.8)
    ax.axhline(50, color='gray', linewidth=0.8, linestyle=':', label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=30, ha='right', fontsize=8)
    for i, (v, t) in enumerate(zip(val_accs, test_accs)):
        ax.text(i - w/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
        ax.text(i + w/2, t + 0.5, f'{t:.1f}', ha='center', va='bottom', fontsize=7)
    _style_ax(ax, 'Accuracy Comparison', '', 'Accuracy %')
    ax.legend(fontsize=8)

    # F1 comparison
    ax = axes[1]
    colors = [COLORS['fusion'] if f > 50 else COLORS['val'] for f in test_f1s]
    ax.bar(x, test_f1s, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(50, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=30, ha='right', fontsize=8)
    for i, f in enumerate(test_f1s):
        ax.text(i, f + 0.5, f'{f:.1f}', ha='center', va='bottom', fontsize=8)
    _style_ax(ax, 'Test F1 Macro Comparison', '', 'F1 %')

    # Subject-level bar with error bars
    ax = axes[2]
    ax.bar(x, subj_means, yerr=subj_stds, color=COLORS['accent'], alpha=0.7,
           capsize=5, edgecolor='white')
    ax.axhline(50, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=30, ha='right', fontsize=8)
    _style_ax(ax, 'Subject Acc (Mean ± Std)', '', 'Accuracy %')

    fig.suptitle('Experiment Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'experiment_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_modality_robustness(results: Dict[str, dict], save_dir: Path):
    """
    Radar/bar chart comparing multimodal full vs missing-modality performance.
    Shows robustness to missing modalities.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    exp_names = list(results.keys())
    test_accs = {e: results[e].get('test_accuracy', 0) * 100 for e in exp_names}

    # Group by pretrained/not
    groups = {'No Pretrain': {}, 'Pretrained': {}}
    for e in exp_names:
        name_lower = e.lower()
        is_pt = 'pt' in name_lower and 'np' not in name_lower
        group = 'Pretrained' if is_pt else 'No Pretrain'

        if 'multi' in name_lower:
            modality = 'Multi'
        elif 'eeg' in name_lower:
            modality = 'EEG'
        elif 'fnirs' in name_lower or 'eye' in name_lower:
            modality = 'fNIRS'
        else:
            modality = e
        groups[group][modality] = test_accs[e]

    # Grouped bar
    modalities = ['Multi', 'EEG', 'fNIRS']
    x = np.arange(len(modalities))
    w = 0.35
    for i, (group_name, vals) in enumerate(groups.items()):
        acc_vals = [vals.get(m, 0) for m in modalities]
        color = COLORS['fusion'] if i == 1 else COLORS['eeg']
        ax1.bar(x + (i - 0.5) * w, acc_vals, w, label=group_name, color=color, alpha=0.8)
        for j, v in enumerate(acc_vals):
            if v > 0:
                ax1.text(j + (i - 0.5) * w, v + 0.3, f'{v:.1f}', ha='center', fontsize=8)

    ax1.axhline(50, color='gray', linewidth=0.8, linestyle=':')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modalities)
    _style_ax(ax1, 'Accuracy by Modality & Pretraining', 'Modality', 'Accuracy %')
    ax1.legend(fontsize=9)

    # Degradation from multi
    for i, (group_name, vals) in enumerate(groups.items()):
        if 'Multi' in vals and vals['Multi'] > 0:
            multi_acc = vals['Multi']
            degradations = {m: multi_acc - vals.get(m, multi_acc) for m in ['EEG', 'fNIRS']}
            x_pos = np.arange(len(degradations))
            color = COLORS['fusion'] if i == 1 else COLORS['eeg']
            offset = (i - 0.5) * w
            bars = ax2.bar(x_pos + offset, list(degradations.values()), w,
                          label=group_name, color=color, alpha=0.8)

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xticks(np.arange(2))
    ax2.set_xticklabels(['Multi→EEG', 'Multi→fNIRS'])
    _style_ax(ax2, 'Accuracy Drop (Missing Modality)', 'Missing Modality', 'Accuracy Drop %')
    ax2.legend(fontsize=9)

    fig.suptitle('Missing Modality Robustness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'modality_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 7. LEARNING RATE & TRAINING DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

def plot_lr_schedule(cfg: dict, save_dir: Path):
    """Plot the cosine LR schedule with warmup."""
    tcfg = cfg.get('training', {})
    lr = tcfg.get('lr', 1e-4)
    min_lr = tcfg.get('min_lr', 1e-6)
    warmup = tcfg.get('warmup_epochs', 10)
    total = tcfg.get('epochs', 200)

    import math
    lrs = []
    for e in range(total):
        if e < warmup:
            cur_lr = lr * e / warmup
        else:
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (e - warmup) / (total - warmup)))
        lrs.append(cur_lr)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(total), lrs, color=COLORS['accent'], linewidth=1.5)
    ax.axvline(warmup, color='gray', linewidth=0.8, linestyle='--', label=f'Warmup={warmup}')
    ax.fill_between(range(warmup), 0, lrs[:warmup], alpha=0.1, color=COLORS['accent'])
    _style_ax(ax, 'Learning Rate Schedule', 'Epoch', 'Learning Rate')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_dir / 'lr_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_pretrain_plots(run_dir: Path, model=None, dataloader=None, device=None):
    """Generate all pretrain-related plots from a completed run."""
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    history_path = run_dir / 'history.json'
    config_path = run_dir / 'config.json'

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_pretrain_loss_decomposition(history, plot_dir)
        plot_pretrain_loss_correlation(history, plot_dir)

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        plot_lr_schedule(cfg, plot_dir)

    title = run_dir.name

    # Model-dependent plots
    if model is not None and dataloader is not None and device is not None:
        # Attention analysis
        sample_batch = next(iter(dataloader))
        try:
            attn_data = extract_attention_maps(model, sample_batch, device, mode='con')
            plot_attention_heatmaps(attn_data, plot_dir, title)
            plot_attention_summary(attn_data, plot_dir, title)
            plot_per_head_attention(attn_data, plot_dir, title)
        except Exception as e:
            print(f"  Attention plots skipped: {e}")

        # Embedding analysis
        try:
            embed_data = extract_embeddings(model, dataloader, device)
            plot_embedding_tsne(embed_data, plot_dir, title)
            plot_modality_alignment(embed_data, plot_dir, title)
        except Exception as e:
            print(f"  Embedding plots skipped: {e}")

        # Fusion gate analysis
        try:
            gate_captures = analyze_ffn_branches(model, dataloader, device)
            plot_fusion_gate_weights(gate_captures, plot_dir, title)
        except Exception as e:
            print(f"  Fusion gate plots skipped: {e}")

        # Gradient flow
        try:
            grad_norms = analyze_gradient_flow(model, sample_batch, device)
            plot_gradient_flow(grad_norms, plot_dir, title)
        except Exception as e:
            print(f"  Gradient flow plots skipped: {e}")


def generate_finetune_plots(run_dir: Path, model=None, dataloader=None, device=None,
                             preds=None, labels=None, subj_accs=None):
    """Generate all finetune-related plots from a completed run."""
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    history_path = run_dir / 'history.json'
    results_path = run_dir / 'results.json'

    history = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    class_names = ['Left MI', 'Right MI']
    title = run_dir.name

    if preds is not None and labels is not None:
        plot_confusion_matrix(preds, labels, class_names, plot_dir, title)
        plot_classification_dashboard(history, preds, labels, subj_accs or {}, class_names, plot_dir, title)
    elif history:
        plot_classification_dashboard(history, np.array([]), np.array([]), {}, class_names, plot_dir, title)

    # Model-dependent plots (for multimodal models only)
    if model is not None and dataloader is not None and device is not None:
        if hasattr(model, 'eeg_embeds') and hasattr(model, 'eye_embeds'):
            sample_batch = next(iter(dataloader))

            try:
                attn_data = extract_attention_maps(model, sample_batch, device, mode='ft')
                plot_attention_heatmaps(attn_data, plot_dir, title)
                plot_attention_summary(attn_data, plot_dir, title)
                plot_per_head_attention(attn_data, plot_dir, title)
            except Exception as e:
                print(f"  Attention plots skipped: {e}")

            try:
                embed_data = extract_embeddings(model, dataloader, device)
                plot_embedding_tsne(embed_data, plot_dir, title)
                plot_modality_alignment(embed_data, plot_dir, title)
            except Exception as e:
                print(f"  Embedding plots skipped: {e}")

            try:
                gate_captures = analyze_ffn_branches(model, dataloader, device)
                plot_fusion_gate_weights(gate_captures, plot_dir, title)
            except Exception as e:
                print(f"  Fusion gate plots skipped: {e}")

            try:
                grad_norms = analyze_gradient_flow(model, sample_batch, device)
                plot_gradient_flow(grad_norms, plot_dir, title)
            except Exception as e:
                print(f"  Gradient flow plots skipped: {e}")
