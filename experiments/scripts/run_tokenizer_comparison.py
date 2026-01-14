#!/usr/bin/env python
"""
Run complete tokenizer comparison on real EEG/fNIRS data.

This script trains FSQ and VQ-VAE tokenizers on both modalities and generates
a comprehensive comparison report.

Usage:
    python run_tokenizer_comparison.py
    python run_tokenizer_comparison.py --epochs 50 --device cuda:0
"""

# Fix OMP duplicate library error (must be before numpy/torch imports)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.eeg_fnirs_dataset import EEGfNIRSDataset, create_dataloaders
from src.tokenizers.fsq import FSQTokenizer
from src.tokenizers.vqvae import VQVAETokenizer
from src.visualization.tokenizer_plots import visualize_tokenizer_run


# =============================================================================
# Experiment Configurations
# =============================================================================

def get_experiment_configs() -> Dict[str, Dict]:
    """Get all experiment configurations."""
    
    base_training = {
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,
        'gradient_clip': 1.0,
        'early_stopping_patience': 20,
    }
    
    # EEG configurations
    eeg_base = {
        'modality': 'eeg',
        'seq_length': 512,      # 2.56s @ 200Hz
        'input_channels': 1,    # Single channel (average or selected)
        'channel_idx': 0,       # Use first channel (can be changed)
        'use_all_channels': False,
        **base_training
    }
    
    # fNIRS configurations
    fnirs_base = {
        'modality': 'fnirs',
        'seq_length': 25,       # 2.5s @ 10Hz
        'input_channels': 1,    # Single channel
        'channel_idx': 0,
        'use_all_channels': False,
        **base_training
    }
    
    configs = {
        'FSQ_EEG': {
            **eeg_base,
            'model_type': 'fsq',
            'levels': [8, 8, 8, 8],  # 4096 codes
            'encoder_dims': [32, 64, 128],
            'kernel_size': 7,
            'stride': 2,
        },
        'VQVAE_EEG': {
            **eeg_base,
            'model_type': 'vqvae',
            'codebook_size': 512,
            'embedding_dim': 64,
            'commitment_cost': 0.25,
            'ema_decay': 0.99,
            'encoder_dims': [32, 64, 128],
            'kernel_size': 7,
            'stride': 2,
        },
        'FSQ_fNIRS': {
            **fnirs_base,
            'model_type': 'fsq',
            'levels': [8, 8, 8],  # 512 codes (smaller for shorter sequences)
            'encoder_dims': [32, 64],  # Shallower network
            'kernel_size': 5,
            'stride': 2,
        },
        'VQVAE_fNIRS': {
            **fnirs_base,
            'model_type': 'vqvae',
            'codebook_size': 256,
            'embedding_dim': 32,
            'commitment_cost': 0.25,
            'ema_decay': 0.99,
            'encoder_dims': [32, 64],
            'kernel_size': 5,
            'stride': 2,
        },
    }
    
    return configs


# =============================================================================
# Model Creation
# =============================================================================

def create_model(config: Dict) -> nn.Module:
    """Create tokenizer model from config."""
    model_type = config['model_type']
    
    if model_type == 'fsq':
        return FSQTokenizer(
            seq_length=config['seq_length'],
            input_channels=config['input_channels'],
            levels=config['levels'],
            encoder_dims=config['encoder_dims'],
            encoder_kernel=config['kernel_size'],
            encoder_stride=config['stride'],
        )
    elif model_type == 'vqvae':
        return VQVAETokenizer(
            seq_length=config['seq_length'],
            input_channels=config['input_channels'],
            codebook_size=config['codebook_size'],
            embedding_dim=config['embedding_dim'],
            commitment_cost=config['commitment_cost'],
            ema_decay=config['ema_decay'],
            encoder_dims=config['encoder_dims'],
            encoder_kernel=config['kernel_size'],
            encoder_stride=config['stride'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Data Loading
# =============================================================================

def create_data_loaders(
    config: Dict,
    data_root: str,
    train_subjects: List[int],
    val_subjects: List[int],
    test_subjects: List[int],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training."""
    
    modality = config['modality']
    
    # Create datasets
    train_dataset = EEGfNIRSDataset(
        data_root=data_root,
        subject_ids=train_subjects,
        task='motor_imagery',
        modality=modality,
        window_samples=config['seq_length'],
        normalize=True,
    )
    
    val_dataset = EEGfNIRSDataset(
        data_root=data_root,
        subject_ids=val_subjects,
        task='motor_imagery',
        modality=modality,
        window_samples=config['seq_length'],
        normalize=True,
    )
    
    test_dataset = EEGfNIRSDataset(
        data_root=data_root,
        subject_ids=test_subjects,
        task='motor_imagery',
        modality=modality,
        window_samples=config['seq_length'],
        normalize=True,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, test_loader


def prepare_batch(batch: Dict, config: Dict, device: torch.device) -> torch.Tensor:
    """Prepare batch data for model input."""
    data = batch['data']  # [B, n_channels, seq_length]
    
    if config['use_all_channels']:
        # Average across channels
        x = data.mean(dim=1)  # [B, seq_length]
    else:
        # Select single channel
        channel_idx = config['channel_idx']
        if channel_idx >= data.shape[1]:
            channel_idx = 0
        x = data[:, channel_idx, :]  # [B, seq_length]
    
    return x.to(device)


# =============================================================================
# Training Loop
# =============================================================================

def compute_losses(
    outputs: Dict,
    x: torch.Tensor,
    config: Dict,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Compute all loss components."""
    x_rec = outputs['x_rec']
    
    # Reconstruction loss
    l_recon = F.mse_loss(x_rec, x)
    
    # Spectral loss (multi-scale STFT)
    l_spectral = compute_spectral_loss(x, x_rec, device)
    
    # Commitment loss (for VQ-VAE)
    l_commit = outputs.get('commitment_loss', torch.tensor(0.0, device=device))
    
    # Total loss
    total = l_recon + 0.1 * l_spectral + l_commit
    
    return {
        'total': total,
        'reconstruction': l_recon,
        'spectral': l_spectral,
        'commitment': l_commit if isinstance(l_commit, torch.Tensor) else torch.tensor(l_commit, device=device),
    }


def compute_spectral_loss(x: torch.Tensor, x_rec: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute multi-scale STFT loss."""
    fft_sizes = [32, 64, 128]
    loss = torch.tensor(0.0, device=device)
    
    # Ensure minimum sequence length
    seq_len = x.shape[-1]
    valid_fft_sizes = [n for n in fft_sizes if n <= seq_len]
    
    if len(valid_fft_sizes) == 0:
        return loss
    
    for n_fft in valid_fft_sizes:
        window = torch.hann_window(n_fft, device=device)
        hop_length = max(1, n_fft // 4)
        
        try:
            stft_x = torch.stft(x, n_fft, hop_length=hop_length, window=window, return_complex=True)
            stft_rec = torch.stft(x_rec, n_fft, hop_length=hop_length, window=window, return_complex=True)
            
            mag_x = stft_x.abs()
            mag_rec = stft_rec.abs()
            loss = loss + F.mse_loss(mag_rec, mag_x)
        except Exception:
            pass
    
    return loss / max(len(valid_fft_sizes), 1)


def compute_codebook_metrics(indices: torch.Tensor, codebook_size: int) -> Dict[str, float]:
    """Compute codebook health metrics."""
    flat = indices.flatten()
    usage = torch.bincount(flat, minlength=codebook_size).float()
    usage_prob = usage / (usage.sum() + 1e-10)
    
    # Perplexity
    entropy = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()
    perplexity = torch.exp(entropy)
    
    # Utilization
    active_codes = (usage > 0).sum()
    utilization = active_codes / codebook_size
    
    # Dead codes
    dead_codes = (usage == 0).sum()
    
    return {
        'perplexity': perplexity.item(),
        'utilization': utilization.item(),
        'dead_codes': dead_codes.item(),
        'active_codes': active_codes.item(),
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def generate_experiment_visualizations(
    exp_dir: Path,
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict,
    history: Dict,
    final_metrics: Dict,
    device: torch.device,
) -> List[Path]:
    """
    Generate comprehensive visualizations for a single experiment.
    
    Returns list of generated figure paths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    figures_dir = exp_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    
    # Color palette
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'tertiary': '#F18F01',
        'success': '#2ECC71',
        'danger': '#E74C3C',
        'light': '#95A5A6',
    }
    
    # =========================================================================
    # 1. Training Curves (Complete)
    # =========================================================================
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # 1.1 Reconstruction Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_recon'], label='Train', color=colors['primary'], linewidth=2)
    ax1.plot(epochs, history['val_recon'], label='Val', color=colors['secondary'], linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('Reconstruction Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 1.2 Perplexity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['perplexity'], color=colors['tertiary'], linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Codebook Perplexity')
    ax2.grid(True, alpha=0.3)
    # Reference line
    codebook_size = model.get_codebook_size()
    ax2.axhline(y=codebook_size * 0.3, color=colors['light'], linestyle=':', alpha=0.7, label=f'30% of {codebook_size}')
    ax2.legend()
    
    # 1.3 Code Utilization
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['utilization'], color=colors['success'], linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Utilization')
    ax3.set_title('Code Utilization')
    ax3.set_ylim([0, 1.05])
    ax3.axhline(y=0.9, color=colors['light'], linestyle=':', alpha=0.7, label='90% target')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 1.4 Dead Codes
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history['dead_codes'], color=colors['danger'], linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Dead Codes')
    ax4.set_title('Dead Codes Count')
    ax4.grid(True, alpha=0.3)
    
    # 1.5 Learning Rate
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, history['lr'], color=colors['primary'], linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.set_title('Learning Rate Schedule')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # 1.6 Spectral Loss
    ax6 = fig.add_subplot(gs[1, 2])
    spectral = history['spectral']
    if any(s > 0 for s in spectral):
        ax6.plot(epochs, spectral, color=colors['secondary'], linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Spectral MSE')
        ax6.set_title('Spectral Loss')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Spectral loss\nnot applicable\n(short sequences)', 
                ha='center', va='center', transform=ax6.transAxes,
                fontsize=12, color=colors['light'])
        ax6.set_title('Spectral Loss')
    
    fig.suptitle(f'Training Progress - {config["model_type"].upper()} on {config["modality"].upper()}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = figures_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(path)
    print(f"  [Viz] Saved: training_curves.png")
    
    # =========================================================================
    # 2. Reconstruction Samples
    # =========================================================================
    model.eval()
    all_original = []
    all_recon = []
    all_indices = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = prepare_batch(batch, config, device)
            outputs = model(x)
            all_original.append(x.cpu())
            all_recon.append(outputs['x_rec'].cpu())
            all_indices.append(outputs['indices'].cpu())
            if len(all_original) * x.shape[0] >= 100:
                break
    
    original = torch.cat(all_original, dim=0).numpy()
    reconstructed = torch.cat(all_recon, dim=0).numpy()
    indices = torch.cat(all_indices, dim=0)
    
    # Get sampling rate
    fs = 200.0 if config['modality'] == 'eeg' else 10.0
    
    n_samples = min(4, len(original))
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_indices = np.linspace(0, len(original)-1, n_samples, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        orig = original[idx]
        recon = reconstructed[idx]
        t = np.arange(len(orig)) / fs
        
        # Time domain comparison
        ax_time = axes[i, 0]
        ax_time.plot(t, orig, label='Original', color=colors['primary'], alpha=0.8, linewidth=1.5)
        ax_time.plot(t, recon, label='Reconstructed', color=colors['secondary'], alpha=0.8, linewidth=1.5, linestyle='--')
        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Amplitude')
        ax_time.set_title(f'Sample {idx}: Time Domain')
        ax_time.legend(loc='upper right')
        ax_time.grid(True, alpha=0.3)
        
        mse = np.mean((orig - recon) ** 2)
        ax_time.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax_time.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residual
        ax_res = axes[i, 1]
        residual = orig - recon
        ax_res.plot(t, residual, color=colors['danger'], alpha=0.7, linewidth=1)
        ax_res.fill_between(t, residual, alpha=0.3, color=colors['danger'])
        ax_res.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax_res.set_xlabel('Time (s)')
        ax_res.set_ylabel('Residual')
        ax_res.set_title(f'Sample {idx}: Reconstruction Error')
        ax_res.grid(True, alpha=0.3)
    
    fig.suptitle('Reconstruction Quality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = figures_dir / "reconstruction_samples.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(path)
    print(f"  [Viz] Saved: reconstruction_samples.png")
    
    # =========================================================================
    # 3. Spectral Comparison (if applicable)
    # =========================================================================
    if config['seq_length'] >= 32:
        try:
            from scipy.signal import welch
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Compute average PSD
            n_use = min(50, len(original))
            psds_orig = []
            psds_recon = []
            
            for i in range(n_use):
                freqs, psd_o = welch(original[i], fs=fs, nperseg=min(64, len(original[i])))
                _, psd_r = welch(reconstructed[i], fs=fs, nperseg=min(64, len(reconstructed[i])))
                psds_orig.append(psd_o)
                psds_recon.append(psd_r)
            
            psd_orig = np.mean(psds_orig, axis=0)
            psd_recon = np.mean(psds_recon, axis=0)
            std_orig = np.std(psds_orig, axis=0)
            std_recon = np.std(psds_recon, axis=0)
            
            # PSD comparison
            ax1 = axes[0]
            ax1.semilogy(freqs, psd_orig, label='Original', color=colors['primary'], linewidth=2)
            ax1.fill_between(freqs, psd_orig - std_orig, psd_orig + std_orig, alpha=0.2, color=colors['primary'])
            ax1.semilogy(freqs, psd_recon, label='Reconstructed', color=colors['secondary'], linewidth=2, linestyle='--')
            ax1.fill_between(freqs, psd_recon - std_recon, psd_recon + std_recon, alpha=0.2, color=colors['secondary'])
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('PSD')
            ax1.set_title('Power Spectral Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # PSD Ratio
            ax2 = axes[1]
            ratio = (psd_recon + 1e-10) / (psd_orig + 1e-10)
            ax2.plot(freqs, ratio, color=colors['tertiary'], linewidth=2)
            ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
            ax2.fill_between(freqs, 0.8, 1.2, alpha=0.2, color=colors['success'])
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Ratio')
            ax2.set_title('Spectral Fidelity (Recon/Orig)')
            ax2.set_ylim([0, 2])
            ax2.grid(True, alpha=0.3)
            
            # Spectral Error
            ax3 = axes[2]
            error = np.abs(psd_orig - psd_recon)
            ax3.semilogy(freqs, error, color=colors['danger'], linewidth=2)
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('|Error|')
            ax3.set_title('Absolute Spectral Error')
            ax3.grid(True, alpha=0.3)
            
            fig.suptitle('Spectral Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            path = figures_dir / "spectral_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated.append(path)
            print(f"  [Viz] Saved: spectral_comparison.png")
            
        except Exception as e:
            print(f"  [Viz] Skipped spectral: {e}")
    
    # =========================================================================
    # 4. Codebook Usage Analysis
    # =========================================================================
    flat_indices = indices.flatten()
    usage = torch.bincount(flat_indices, minlength=codebook_size).numpy()
    
    sorted_indices = np.argsort(usage)[::-1]
    sorted_usage = usage[sorted_indices]
    
    total_tokens = usage.sum()
    used_codes = (usage > 0).sum()
    dead_codes = codebook_size - used_codes
    
    # Stats
    probs = usage / (total_tokens + 1e-10)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    perplexity = np.exp(entropy)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Usage histogram (sorted)
    ax1 = axes[0, 0]
    x = np.arange(codebook_size)
    ax1.bar(x, sorted_usage, color=colors['primary'], alpha=0.7, width=1.0)
    ax1.set_xlabel('Code Index (sorted)')
    ax1.set_ylabel('Usage Count')
    ax1.set_title('Codebook Usage (Sorted by Frequency)')
    ax1.set_xlim([0, codebook_size])
    if dead_codes > 0:
        ax1.axvspan(used_codes, codebook_size, alpha=0.3, color=colors['danger'], label=f'Dead: {dead_codes}')
        ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Lorenz curve
    ax2 = axes[0, 1]
    cumulative = np.cumsum(sorted_usage) / (total_tokens + 1e-10)
    ax2.plot(x / codebook_size * 100, cumulative * 100, color=colors['primary'], linewidth=2, label='Actual')
    ax2.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Uniform')
    ax2.fill_between(x / codebook_size * 100, cumulative * 100, alpha=0.3, color=colors['primary'])
    ax2.set_xlabel('% of Codes')
    ax2.set_ylabel('% of Tokens')
    ax2.set_title('Cumulative Usage (Lorenz Curve)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Usage distribution
    ax3 = axes[1, 0]
    nonzero = usage[usage > 0]
    if len(nonzero) > 0:
        bins = np.logspace(0, np.log10(max(nonzero) + 1), 30)
        ax3.hist(nonzero, bins=bins, color=colors['secondary'], alpha=0.7, edgecolor='white')
        ax3.set_xscale('log')
        ax3.set_xlabel('Usage Count (log)')
        ax3.set_ylabel('Number of Codes')
        ax3.set_title('Usage Distribution')
        ax3.grid(True, alpha=0.3)
    
    # Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = (
        f"Codebook Statistics\n"
        f"{'='*35}\n\n"
        f"Codebook size:    {codebook_size:>10,}\n"
        f"Active codes:     {used_codes:>10,} ({used_codes/codebook_size*100:.1f}%)\n"
        f"Dead codes:       {dead_codes:>10,} ({dead_codes/codebook_size*100:.1f}%)\n\n"
        f"Total tokens:     {total_tokens:>10,}\n"
        f"Perplexity:       {perplexity:>10.1f}\n"
        f"Max perplexity:   {codebook_size:>10}\n"
        f"Normalized:       {perplexity/codebook_size*100:>9.1f}%\n\n"
        f"Most used:        {sorted_indices[0]:>10} ({sorted_usage[0]:,}x)\n"
        f"Median usage:     {np.median(nonzero):>10.0f}\n" if len(nonzero) > 0 else ""
    )
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12, 
            family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    fig.suptitle('Codebook Health Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = figures_dir / "codebook_usage.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(path)
    print(f"  [Viz] Saved: codebook_usage.png")
    
    # =========================================================================
    # 5. Token Embedding Visualization (t-SNE/PCA)
    # =========================================================================
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        if hasattr(model, 'get_codebook_embeddings'):
            embeddings = model.get_codebook_embeddings().cpu().numpy()
            K, D = embeddings.shape
            
            # Use PCA for small codebooks, t-SNE for larger
            if K > 100:
                reducer = TSNE(n_components=2, perplexity=min(30, K//3), random_state=42)
                method = 't-SNE'
            else:
                reducer = PCA(n_components=2)
                method = 'PCA'
            
            reduced = reducer.fit_transform(embeddings)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Colored by usage
            log_usage = np.log1p(usage)
            scatter = axes[0].scatter(reduced[:, 0], reduced[:, 1], c=log_usage, 
                                     cmap='viridis', s=30, alpha=0.7)
            plt.colorbar(scatter, ax=axes[0], label='Log(1+usage)')
            axes[0].set_xlabel(f'{method} 1')
            axes[0].set_ylabel(f'{method} 2')
            axes[0].set_title(f'Embeddings Colored by Usage')
            axes[0].grid(True, alpha=0.3)
            
            # Active vs Dead
            active_mask = usage > 0
            dead_mask = usage == 0
            axes[1].scatter(reduced[active_mask, 0], reduced[active_mask, 1],
                           color=colors['success'], s=30, alpha=0.7, label=f'Active ({active_mask.sum()})')
            if dead_mask.any():
                axes[1].scatter(reduced[dead_mask, 0], reduced[dead_mask, 1],
                               color=colors['danger'], s=30, alpha=0.7, label=f'Dead ({dead_mask.sum()})')
            axes[1].legend()
            axes[1].set_xlabel(f'{method} 1')
            axes[1].set_ylabel(f'{method} 2')
            axes[1].set_title('Active vs Dead Codes')
            axes[1].grid(True, alpha=0.3)
            
            fig.suptitle(f'Codebook Embedding Space (K={K}, D={D})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            path = figures_dir / "token_embeddings.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated.append(path)
            print(f"  [Viz] Saved: token_embeddings.png")
            
    except Exception as e:
        print(f"  [Viz] Skipped embeddings: {e}")
    
    # =========================================================================
    # 6. Experiment Summary
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    EXPERIMENT SUMMARY                        ║
╠══════════════════════════════════════════════════════════════╣
║  Model:       {config['model_type'].upper():<47}║
║  Modality:    {config['modality'].upper():<47}║
║  Seq Length:  {config['seq_length']:<47}║
║  Codebook:    {codebook_size:<47}║
╠══════════════════════════════════════════════════════════════╣
║                      FINAL METRICS                           ║
╠══════════════════════════════════════════════════════════════╣
║  Test Recon MSE:     {final_metrics['test_recon_mse']:<40.6f}║
║  Test Spectral MSE:  {final_metrics['test_spectral_mse']:<40.6f}║
║  Test Perplexity:    {final_metrics['test_perplexity']:<40.1f}║
║  Test Utilization:   {final_metrics['test_utilization']*100:<39.1f}%║
║  Dead Codes:         {int(final_metrics['test_dead_codes']):<40}║
║  Training Time:      {final_metrics['training_time_s']:<39.1f}s║
║  Epochs:             {final_metrics['epochs_trained']:<40}║
╚══════════════════════════════════════════════════════════════╝
"""
    
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=11,
           family='monospace', verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))
    
    plt.tight_layout()
    
    path = figures_dir / "experiment_summary.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(path)
    print(f"  [Viz] Saved: experiment_summary.png")
    
    return generated


def train_single_experiment(
    exp_name: str,
    config: Dict,
    data_root: str,
    device: torch.device,
    train_subjects: List[int],
    val_subjects: List[int],
    test_subjects: List[int],
    run_dir: Path,
) -> Dict[str, Any]:
    """Train a single experiment and return metrics."""
    
    print(f"\n{'='*60}")
    print(f"Training: {exp_name}")
    print(f"{'='*60}")
    
    # Create data loaders
    print(f"[Data] Loading {config['modality'].upper()} data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config, data_root, train_subjects, val_subjects, test_subjects
    )
    print(f"[Data] Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    print(f"[Model] Creating {config['model_type'].upper()} tokenizer...")
    model = create_model(config).to(device)
    codebook_size = model.get_codebook_size()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Codebook size: {codebook_size}, Parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] - config['warmup_epochs']
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon': [], 'val_recon': [],
        'perplexity': [], 'utilization': [],
        'dead_codes': [], 'lr': [],
        'spectral': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        # === Training ===
        model.train()
        train_losses = []
        all_indices = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            x = prepare_batch(batch, config, device)
            outputs = model(x)
            losses = compute_losses(outputs, x, config, device)
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            
            train_losses.append({
                'total': losses['total'].item(),
                'recon': losses['reconstruction'].item(),
                'spectral': losses['spectral'].item(),
            })
            all_indices.append(outputs['indices'].detach().cpu())
        
        # Scheduler step
        if epoch > config['warmup_epochs']:
            scheduler.step()
        
        # === Validation ===
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = prepare_batch(batch, config, device)
                outputs = model(x)
                losses = compute_losses(outputs, x, config, device)
                val_losses.append({
                    'total': losses['total'].item(),
                    'recon': losses['reconstruction'].item(),
                })
        
        # Aggregate metrics
        train_loss = np.mean([l['total'] for l in train_losses])
        train_recon = np.mean([l['recon'] for l in train_losses])
        val_loss = np.mean([l['total'] for l in val_losses])
        val_recon = np.mean([l['recon'] for l in val_losses])
        
        # Codebook metrics
        all_train_indices = torch.cat(all_indices, dim=0)
        cb_metrics = compute_codebook_metrics(all_train_indices, codebook_size)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_recon)
        history['val_recon'].append(val_recon)
        history['perplexity'].append(cb_metrics['perplexity'])
        history['utilization'].append(cb_metrics['utilization'])
        history['dead_codes'].append(cb_metrics['dead_codes'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['spectral'].append(np.mean([l['spectral'] for l in train_losses]))
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch:3d}/{config['epochs']}] "
                  f"train={train_recon:.4f}, val={val_recon:.4f}, "
                  f"perp={cb_metrics['perplexity']:.1f}, util={cb_metrics['utilization']:.3f}, "
                  f"time={elapsed:.0f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"[Early Stop] No improvement for {config['early_stopping_patience']} epochs")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # === Final Evaluation ===
    print(f"\n[Evaluation] Computing final metrics...")
    model.eval()
    test_losses = []
    test_indices = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = prepare_batch(batch, config, device)
            outputs = model(x)
            losses = compute_losses(outputs, x, config, device)
            test_losses.append({
                'total': losses['total'].item(),
                'recon': losses['reconstruction'].item(),
                'spectral': losses['spectral'].item(),
            })
            test_indices.append(outputs['indices'].cpu())
    
    # Final metrics
    all_test_indices = torch.cat(test_indices, dim=0)
    test_cb_metrics = compute_codebook_metrics(all_test_indices, codebook_size)
    
    final_metrics = {
        'experiment': exp_name,
        'model_type': config['model_type'],
        'modality': config['modality'],
        'codebook_size': codebook_size,
        'n_parameters': n_params,
        'test_loss': np.mean([l['total'] for l in test_losses]),
        'test_recon_mse': np.mean([l['recon'] for l in test_losses]),
        'test_spectral_mse': np.mean([l['spectral'] for l in test_losses]),
        'test_perplexity': test_cb_metrics['perplexity'],
        'test_utilization': test_cb_metrics['utilization'],
        'test_dead_codes': test_cb_metrics['dead_codes'],
        'best_val_loss': best_val_loss,
        'training_time_s': time.time() - start_time,
        'epochs_trained': epoch,
    }
    
    # Save experiment results
    exp_dir = run_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': final_metrics,
        'history': history,
    }, exp_dir / 'checkpoint.pt')
    
    # Save metrics
    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Generate visualizations
    print(f"[Visualization] Generating figures...")
    try:
        # Build complete metrics history for visualization
        metrics_history = [
            {
                'epoch': i+1,
                'reconstruction_mse': history['train_recon'][i],
                'val_reconstruction_mse': history['val_recon'][i],
                'perplexity': history['perplexity'][i],
                'code_utilization': history['utilization'][i],
                'dead_codes': history['dead_codes'][i],
                'lr': history['lr'][i],
                'spectral_mse': history['spectral'][i],
            }
            for i in range(len(history['train_loss']))
        ]
        
        # Use custom visualization for this comparison experiment
        figures = generate_experiment_visualizations(
            exp_dir=exp_dir,
            model=model,
            test_loader=test_loader,
            config=config,
            history=history,
            final_metrics=final_metrics,
            device=device,
        )
        print(f"[Visualization] Generated {len(figures)} figures")
    except Exception as e:
        import traceback
        print(f"[Visualization] Warning: {e}")
        traceback.print_exc()
    
    print(f"\n[{exp_name}] Final Results:")
    print(f"  Test Recon MSE: {final_metrics['test_recon_mse']:.6f}")
    print(f"  Test Perplexity: {final_metrics['test_perplexity']:.1f} / {codebook_size}")
    print(f"  Utilization: {final_metrics['test_utilization']:.1%}")
    print(f"  Training Time: {final_metrics['training_time_s']:.1f}s")
    
    return final_metrics


# =============================================================================
# Report Generation
# =============================================================================

def generate_comparison_report(
    results: Dict[str, Dict],
    run_dir: Path,
) -> str:
    """Generate Markdown comparison report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Tokenizer Comparison Report

**Generated:** {timestamp}
**Device:** {results.get('device', 'unknown')}

## Summary

This report compares FSQ and VQ-VAE tokenizers trained on real EEG and fNIRS data
from the EEG+NIRS Single-Trial dataset (TU Berlin).

## Experiment Configuration

| Parameter | EEG | fNIRS |
|-----------|-----|-------|
| Sampling Rate | 200 Hz | 10 Hz |
| Window Length | 512 samples (2.56s) | 25 samples (2.5s) |
| Task | Motor Imagery | Motor Imagery |
| Train Subjects | 1-20 | 1-20 |
| Val Subjects | 21-25 | 21-25 |
| Test Subjects | 26-29 | 26-29 |

## Results Overview

### Performance Comparison

| Experiment | Modality | Model | Codebook | Test MSE ↓ | Perplexity ↑ | Utilization ↑ | Dead Codes ↓ | Time (s) |
|------------|----------|-------|----------|------------|--------------|---------------|--------------|----------|
"""
    
    # Sort by modality then model type
    exp_order = ['FSQ_EEG', 'VQVAE_EEG', 'FSQ_fNIRS', 'VQVAE_fNIRS']
    
    for exp_name in exp_order:
        if exp_name in results:
            r = results[exp_name]
            report += f"| {exp_name} | {r['modality'].upper()} | {r['model_type'].upper()} | {r['codebook_size']} | {r['test_recon_mse']:.6f} | {r['test_perplexity']:.1f} | {r['test_utilization']:.1%} | {int(r['test_dead_codes'])} | {r['training_time_s']:.0f} |\n"
    
    report += """
### Analysis by Modality

#### EEG Results

"""
    
    # EEG comparison
    if 'FSQ_EEG' in results and 'VQVAE_EEG' in results:
        fsq = results['FSQ_EEG']
        vqvae = results['VQVAE_EEG']
        
        report += f"""| Metric | FSQ | VQ-VAE | Better |
|--------|-----|--------|--------|
| Reconstruction MSE | {fsq['test_recon_mse']:.6f} | {vqvae['test_recon_mse']:.6f} | {'FSQ' if fsq['test_recon_mse'] < vqvae['test_recon_mse'] else 'VQ-VAE'} |
| Spectral MSE | {fsq['test_spectral_mse']:.6f} | {vqvae['test_spectral_mse']:.6f} | {'FSQ' if fsq['test_spectral_mse'] < vqvae['test_spectral_mse'] else 'VQ-VAE'} |
| Perplexity | {fsq['test_perplexity']:.1f} | {vqvae['test_perplexity']:.1f} | {'FSQ' if fsq['test_perplexity'] > vqvae['test_perplexity'] else 'VQ-VAE'} |
| Utilization | {fsq['test_utilization']:.1%} | {vqvae['test_utilization']:.1%} | {'FSQ' if fsq['test_utilization'] > vqvae['test_utilization'] else 'VQ-VAE'} |

"""
    
    report += """#### fNIRS Results

"""
    
    # fNIRS comparison
    if 'FSQ_fNIRS' in results and 'VQVAE_fNIRS' in results:
        fsq = results['FSQ_fNIRS']
        vqvae = results['VQVAE_fNIRS']
        
        report += f"""| Metric | FSQ | VQ-VAE | Better |
|--------|-----|--------|--------|
| Reconstruction MSE | {fsq['test_recon_mse']:.6f} | {vqvae['test_recon_mse']:.6f} | {'FSQ' if fsq['test_recon_mse'] < vqvae['test_recon_mse'] else 'VQ-VAE'} |
| Spectral MSE | {fsq['test_spectral_mse']:.6f} | {vqvae['test_spectral_mse']:.6f} | {'FSQ' if fsq['test_spectral_mse'] < vqvae['test_spectral_mse'] else 'VQ-VAE'} |
| Perplexity | {fsq['test_perplexity']:.1f} | {vqvae['test_perplexity']:.1f} | {'FSQ' if fsq['test_perplexity'] > vqvae['test_perplexity'] else 'VQ-VAE'} |
| Utilization | {fsq['test_utilization']:.1%} | {vqvae['test_utilization']:.1%} | {'FSQ' if fsq['test_utilization'] > vqvae['test_utilization'] else 'VQ-VAE'} |

"""
    
    report += """## Key Observations

### Reconstruction Quality

"""
    
    # Find best performers
    best_eeg = min(['FSQ_EEG', 'VQVAE_EEG'], key=lambda x: results.get(x, {}).get('test_recon_mse', float('inf'))) if 'FSQ_EEG' in results and 'VQVAE_EEG' in results else None
    best_fnirs = min(['FSQ_fNIRS', 'VQVAE_fNIRS'], key=lambda x: results.get(x, {}).get('test_recon_mse', float('inf'))) if 'FSQ_fNIRS' in results and 'VQVAE_fNIRS' in results else None
    
    if best_eeg and best_eeg in results:
        report += f"""- **Best EEG tokenizer:** {best_eeg} (MSE: {results[best_eeg]['test_recon_mse']:.6f})
"""
    else:
        report += "- **Best EEG tokenizer:** N/A (experiments not completed)\n"
    
    if best_fnirs and best_fnirs in results:
        report += f"""- **Best fNIRS tokenizer:** {best_fnirs} (MSE: {results[best_fnirs]['test_recon_mse']:.6f})
"""
    else:
        report += "- **Best fNIRS tokenizer:** N/A (experiments not completed)\n"

    report += """
### Codebook Utilization

"""
    
    for exp_name in exp_order:
        if exp_name in results:
            r = results[exp_name]
            status = "✓ Good" if r['test_utilization'] > 0.2 else "⚠ Low"
            report += f"- **{exp_name}:** {r['test_utilization']:.1%} ({status})\n"
    
    report += """
### Training Efficiency

"""
    
    total_time = sum(r.get('training_time_s', 0) for r in results.values() if isinstance(r, dict))
    report += f"- Total training time: {total_time:.0f}s ({total_time/60:.1f} min)\n"
    
    for exp_name in exp_order:
        if exp_name in results:
            r = results[exp_name]
            report += f"- {exp_name}: {r['epochs_trained']} epochs in {r['training_time_s']:.0f}s\n"
    
    report += """
## Recommendations

Based on these results:

"""
    
    # Generate recommendations
    if 'FSQ_EEG' in results and 'VQVAE_EEG' in results:
        if results['FSQ_EEG']['test_recon_mse'] < results['VQVAE_EEG']['test_recon_mse']:
            report += "1. **For EEG tokenization:** FSQ shows better reconstruction quality\n"
        else:
            report += "1. **For EEG tokenization:** VQ-VAE shows better reconstruction quality\n"
    
    if 'FSQ_fNIRS' in results and 'VQVAE_fNIRS' in results:
        if results['FSQ_fNIRS']['test_recon_mse'] < results['VQVAE_fNIRS']['test_recon_mse']:
            report += "2. **For fNIRS tokenization:** FSQ shows better reconstruction quality\n"
        else:
            report += "2. **For fNIRS tokenization:** VQ-VAE shows better reconstruction quality\n"
    
    report += """
## Files Generated

Each experiment generated:
- `checkpoint.pt` - Model weights and training history
- `metrics.json` - Final evaluation metrics
- `figures/` - Visualization plots

## Next Steps

1. Analyze reconstruction quality for different frequency bands
2. Evaluate cross-subject generalization
3. Test with multi-channel configurations
4. Integrate best tokenizers into PID-MCM framework
"""
    
    return report


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run tokenizer comparison experiments")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu)")
    
    # Default data root: relative to project root
    default_data_root = str(project_root / "data" / "EEG+NIRS Single-Trial")
    parser.add_argument("--data-root", type=str, default=default_data_root, 
                        help="Path to dataset root")
    parser.add_argument("--experiments", type=str, nargs='+', 
                        default=['FSQ_EEG', 'VQVAE_EEG', 'FSQ_fNIRS', 'VQVAE_fNIRS'],
                        help="Experiments to run")
    parser.add_argument("--train-subjects", type=int, nargs='+', default=list(range(1, 21)))
    parser.add_argument("--val-subjects", type=int, nargs='+', default=list(range(21, 26)))
    parser.add_argument("--test-subjects", type=int, nargs='+', default=list(range(26, 30)))
    
    args = parser.parse_args()
    
    # Setup device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print("Tokenizer Comparison Experiment")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Data: {args.data_root}")
    print(f"Experiments: {args.experiments}")
    
    # Verify CUDA
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "experiments" / "runs" / f"comparison_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Get experiment configs
    all_configs = get_experiment_configs()
    
    # Override epochs
    for config in all_configs.values():
        config['epochs'] = args.epochs
    
    # Run experiments
    results = {'device': str(device)}
    
    for exp_name in args.experiments:
        if exp_name not in all_configs:
            print(f"[Warning] Unknown experiment: {exp_name}")
            continue
        
        config = all_configs[exp_name]
        
        try:
            metrics = train_single_experiment(
                exp_name=exp_name,
                config=config,
                data_root=args.data_root,
                device=device,
                train_subjects=args.train_subjects,
                val_subjects=args.val_subjects,
                test_subjects=args.test_subjects,
                run_dir=run_dir,
            )
            results[exp_name] = metrics
            
        except Exception as e:
            print(f"[Error] Experiment {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate report
    print(f"\n{'='*60}")
    print("Generating Comparison Report")
    print(f"{'='*60}")
    
    report = generate_comparison_report(results, run_dir)
    
    # Save report
    report_path = run_dir / "COMPARISON_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    # Save all results
    results_path = run_dir / "all_results.json"
    with open(results_path, 'w') as f:
        # Filter out non-serializable items
        serializable_results = {k: v for k, v in results.items() if isinstance(v, (dict, str))}
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    print("\nFinal Results Summary:")
    print("-" * 80)
    print(f"{'Experiment':<15} {'Model':<8} {'Modality':<8} {'Test MSE':<12} {'Perplexity':<12} {'Util':<8}")
    print("-" * 80)
    
    for exp_name in args.experiments:
        if exp_name in results and isinstance(results[exp_name], dict):
            r = results[exp_name]
            print(f"{exp_name:<15} {r['model_type']:<8} {r['modality']:<8} {r['test_recon_mse']:<12.6f} {r['test_perplexity']:<12.1f} {r['test_utilization']:<8.1%}")
    
    print("-" * 80)
    print(f"\nAll outputs saved to: {run_dir}")
    
    return results


if __name__ == "__main__":
    main()
