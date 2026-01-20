#!/usr/bin/env python
"""
Train Aligned Tokenizers for EEG (30ch) and fNIRS (36ch HbO)

基于更新的数据集实现和文献推荐参数:
- EEG: 30通道 (排除EOG), 4s窗口@200Hz=800样本, 100 tokens, codebook=1024
- fNIRS: 36 HbO通道, 4s窗口@10Hz=40样本, 10 tokens, codebook=512

两个tokenizer的嵌入维度对齐(dim=64)，便于后续多模态融合。

Usage:
    python train_aligned_tokenizers_v2.py --modality eeg
    python train_aligned_tokenizers_v2.py --modality fnirs
    python train_aligned_tokenizers_v2.py --modality both
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.eeg_fnirs_dataset import EEGfNIRSDataset, create_dataloaders
from src.tokenizers.vqvae import VQVAETokenizer
from src.metrics.codebook_health import compute_codebook_health
from src.visualization.tokenizer_plots import TokenizerVisualizer


@dataclass
class TokenizerConfig:
    """Tokenizer configuration aligned with literature recommendations."""
    name: str
    modality: str
    
    # Data parameters
    seq_length: int           # samples per window
    sample_rate: float        # Hz
    time_duration_s: float    # seconds
    n_channels: int           # number of channels after filtering
    
    # Encoder architecture
    encoder_dims: tuple
    kernel_size: int
    stride: int
    
    # Quantizer
    codebook_size: int
    embedding_dim: int
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    early_stopping_patience: int = 20


def get_tokenizer_configs() -> Dict[str, TokenizerConfig]:
    """
    Get optimized tokenizer configurations based on updated dataset.
    
    Key design decisions:
    1. EEG: 30 channels (EOG excluded), VQ-VAE with 1024 codebook
    2. fNIRS: 36 HbO channels, VQ-VAE with 512 codebook
    3. Both use 4s window (MI-BCI standard) and dim=64 (for alignment)
    """
    
    return {
        # EEG: 30 channels, 4s @ 200Hz = 800 samples -> 100 tokens
        'VQVAE_EEG_30ch': TokenizerConfig(
            name='VQVAE_EEG_30ch',
            modality='eeg',
            seq_length=800,           # 4.0s @ 200Hz
            sample_rate=200.0,
            time_duration_s=4.0,
            n_channels=30,            # EOG excluded
            encoder_dims=(64, 128, 256),  # 3 layers -> 800/8 = 100 tokens
            kernel_size=7,
            stride=2,
            codebook_size=1024,
            embedding_dim=64,
        ),
        
        # fNIRS: 36 HbO channels, 4s @ 10Hz = 40 samples -> 10 tokens
        'VQVAE_fNIRS_36ch': TokenizerConfig(
            name='VQVAE_fNIRS_36ch',
            modality='fnirs',
            seq_length=40,            # 4.0s @ 10Hz
            sample_rate=10.0,
            time_duration_s=4.0,
            n_channels=36,            # HbO only
            encoder_dims=(64, 128),   # 2 layers -> 40/4 = 10 tokens
            kernel_size=5,
            stride=2,
            codebook_size=512,        # Smaller for smoother signals
            embedding_dim=64,         # Aligned with EEG
        ),
    }


def create_model(config: TokenizerConfig, device: str) -> nn.Module:
    """Create VQ-VAE tokenizer model from config."""
    model = VQVAETokenizer(
        seq_length=config.seq_length,
        input_channels=1,  # Single-channel tokenizer
        codebook_size=config.codebook_size,
        embedding_dim=config.embedding_dim,
        commitment_cost=config.commitment_cost,
        ema_decay=config.ema_decay,
        encoder_dims=list(config.encoder_dims),
        encoder_kernel=config.kernel_size,
        encoder_stride=config.stride,
    )
    return model.to(device)


def prepare_batch(batch: Dict, device: str) -> torch.Tensor:
    """
    Prepare batch for single-channel tokenizer training.
    
    Input batch['data'] shape: [B, C, T]
    Output shape: [B*C, T] (flatten channels into batch dimension)
    
    This allows training a single-channel tokenizer that can be applied
    to any channel independently.
    """
    data = batch['data'].to(device)  # [B, C, T]
    B, C, T = data.shape
    
    # Flatten: [B, C, T] -> [B*C, T]
    data = data.view(B * C, T)
    
    return data


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_quant_loss = 0.0
    all_indices = []
    n_batches = 0
    
    for batch in dataloader:
        x = prepare_batch(batch, device)
        
        optimizer.zero_grad()
        output = model(x)
        
        x_rec = output['x_rec']
        recon_loss = nn.functional.mse_loss(x_rec, x)
        quant_loss = output.get('commitment_loss', torch.tensor(0.0, device=device))
        
        loss = recon_loss + quant_loss
        loss.backward()
        
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_quant_loss += quant_loss.item() if isinstance(quant_loss, torch.Tensor) else quant_loss
        all_indices.append(output['indices'].detach().cpu())
        n_batches += 1
    
    # Compute codebook metrics
    indices = torch.cat(all_indices, dim=0)
    codebook_metrics = compute_codebook_health(indices, model.get_codebook_size())
    
    return {
        'loss': total_loss / n_batches,
        'recon_mse': total_recon_loss / n_batches,
        'quant_loss': total_quant_loss / n_batches,
        **codebook_metrics,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on dataloader."""
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    all_indices = []
    all_originals = []
    all_reconstructed = []
    n_batches = 0
    
    for batch in dataloader:
        x = prepare_batch(batch, device)
        
        output = model(x)
        x_rec = output['x_rec']
        
        recon_loss = nn.functional.mse_loss(x_rec, x)
        quant_loss = output.get('commitment_loss', torch.tensor(0.0, device=device))
        loss = recon_loss + quant_loss
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        all_indices.append(output['indices'].cpu())
        
        # Save samples for visualization (first few batches)
        if len(all_originals) < 5:
            all_originals.append(x.cpu())
            all_reconstructed.append(x_rec.cpu())
        
        n_batches += 1
    
    # Compute codebook metrics
    indices = torch.cat(all_indices, dim=0)
    codebook_metrics = compute_codebook_health(indices, model.get_codebook_size())
    
    result = {
        'loss': total_loss / n_batches,
        'recon_mse': total_recon_loss / n_batches,
        **codebook_metrics,
    }
    
    # Add samples for visualization
    if all_originals:
        result['_originals'] = torch.cat(all_originals, dim=0)
        result['_reconstructed'] = torch.cat(all_reconstructed, dim=0)
        result['_indices'] = indices
    
    return result


def train_tokenizer(
    config: TokenizerConfig,
    data_root: str,
    output_dir: Path,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Train a single tokenizer.
    
    The tokenizer is trained on all channels independently (single-channel design).
    This allows the same tokenizer to be applied to any channel.
    """
    print(f"\n{'='*70}")
    print(f"Training {config.name}")
    print(f"{'='*70}")
    print(f"Modality: {config.modality}")
    print(f"Channels: {config.n_channels}")
    print(f"Window: {config.seq_length} samples @ {config.sample_rate}Hz = {config.time_duration_s}s")
    print(f"Encoder: {config.encoder_dims}, stride={config.stride}")
    n_tokens = config.seq_length // (config.stride ** len(config.encoder_dims))
    print(f"Output: {n_tokens} tokens per channel, dim={config.embedding_dim}")
    print(f"Codebook: {config.codebook_size} codes")
    
    # Create dataloaders with channel filtering
    dataloaders = create_dataloaders(
        data_root=data_root,
        modality=config.modality,
        task='motor_imagery',
        train_subjects=list(range(1, 21)),
        val_subjects=list(range(21, 26)),
        test_subjects=list(range(26, 30)),
        window_samples=config.seq_length,
        batch_size=config.batch_size,
        num_workers=0,
        # Channel filtering options
        exclude_eog=(config.modality == 'eeg'),  # Exclude EOG for EEG
        hbo_only=(config.modality == 'fnirs'),   # HbO only for fNIRS
        hbr_only=False,
    )
    
    # Verify channel count
    train_dataset = dataloaders['train'].dataset
    actual_channels = train_dataset.get_num_channels()
    channel_names = train_dataset.get_channel_names()
    
    print(f"\nDataset info:")
    print(f"  Actual channels: {actual_channels}")
    print(f"  Expected channels: {config.n_channels}")
    print(f"  Channel names: {channel_names[:5]}..." if len(channel_names) > 5 else f"  Channel names: {channel_names}")
    print(f"  Train trials: {len(train_dataset)}")
    print(f"  Train samples (channel-flattened): {len(train_dataset) * actual_channels}")
    print(f"  Val trials: {len(dataloaders['val'].dataset)}")
    print(f"  Test trials: {len(dataloaders['test'].dataset)}")
    
    if actual_channels != config.n_channels:
        print(f"  ⚠️ Channel count mismatch! Expected {config.n_channels}, got {actual_channels}")
    
    # Create model
    model = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Verify output shape
    sample_batch = next(iter(dataloaders['train']))
    sample_x = prepare_batch(sample_batch, device)
    with torch.no_grad():
        sample_out = model(sample_x)
    print(f"\nShape verification:")
    print(f"  Input: {sample_batch['data'].shape} -> flattened: {sample_x.shape}")
    print(f"  Tokens: {sample_out['indices'].shape}")
    print(f"  Latent: {sample_out['z_q'].shape}")
    print(f"  Recon: {sample_out['x_rec'].shape}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    start_time = time.time()
    print(f"\n{'='*70}")
    print("Training Progress")
    print(f"{'='*70}")
    
    for epoch in range(config.epochs):
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, device, config.gradient_clip
        )
        val_metrics = evaluate(model, dataloaders['val'], device)
        
        # Remove visualization data from metrics for history
        val_metrics_clean = {k: v for k, v in val_metrics.items() if not k.startswith('_')}
        
        scheduler.step()
        
        history['train'].append({**train_metrics, 'lr': optimizer.param_groups[0]['lr']})
        history['val'].append(val_metrics_clean)
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
                'channel_names': channel_names,
                'n_channels': actual_channels,
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}: loss={train_metrics['loss']:.4f}, "
                  f"mse={train_metrics['recon_mse']:.4f}, "
                  f"perp={train_metrics['perplexity']:.1f}, "
                  f"util={train_metrics['code_utilization']*100:.1f}%, "
                  f"val_mse={val_metrics_clean['recon_mse']:.4f}")
        
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    # Load best model and evaluate on test
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}")
    
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, dataloaders['test'], device)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = TokenizerVisualizer(output_dir)
    
    # Training curves
    metrics_history = []
    for i, (t, v) in enumerate(zip(history['train'], history['val'])):
        metrics_history.append({
            'epoch': i + 1,
            'reconstruction_mse': t['recon_mse'],
            'val_reconstruction_mse': v['recon_mse'],
            'perplexity': t['perplexity'],
            'code_utilization': t['code_utilization'],
            'dead_codes': t['dead_codes'],
            'lr': t.get('lr', 0),
        })
    visualizer.plot_training_curves(metrics_history)
    
    # Reconstruction samples
    if '_originals' in test_metrics:
        visualizer.plot_reconstruction_samples(
            test_metrics['_originals'],
            test_metrics['_reconstructed'],
            fs=config.sample_rate,
        )
    
    # Codebook usage
    if '_indices' in test_metrics:
        visualizer.plot_codebook_usage(
            test_metrics['_indices'],
            model.get_codebook_size(),
        )
    
    # Token embeddings
    if hasattr(model, 'get_codebook_embeddings'):
        embeddings = model.get_codebook_embeddings()
        if embeddings is not None and '_indices' in test_metrics:
            usage = torch.bincount(
                test_metrics['_indices'].flatten(),
                minlength=model.get_codebook_size()
            )
            visualizer.plot_token_embeddings(embeddings, usage)
    
    visualizer.save_figure_manifest()
    
    # Clean test metrics
    test_metrics_clean = {k: v for k, v in test_metrics.items() if not k.startswith('_')}
    
    # Save results
    results = {
        'config': asdict(config),
        'actual_channels': actual_channels,
        'channel_names': channel_names,
        'n_parameters': n_params,
        'training_time_s': training_time,
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'test_metrics': test_metrics_clean,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest Results for {config.name}:")
    print(f"  Reconstruction MSE: {test_metrics_clean['recon_mse']:.6f}")
    print(f"  Perplexity: {test_metrics_clean['perplexity']:.1f}")
    print(f"  Code Utilization: {test_metrics_clean['code_utilization']*100:.1f}%")
    print(f"  Dead Codes: {test_metrics_clean['dead_codes']}/{config.codebook_size}")
    print(f"  Training Time: {training_time/60:.1f} min")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train aligned tokenizers for EEG and fNIRS')
    parser.add_argument('--modality', type=str, default='both',
                        choices=['eeg', 'fnirs', 'both'],
                        help='Which modality to train')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--data-root', type=str, 
                        default='data/EEG+NIRS Single-Trial',
                        help='Path to data directory')
    args = parser.parse_args()
    
    # Check device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = project_root / 'experiments' / 'runs' / f'aligned_tokenizers_{timestamp}'
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_base}")
    
    # Get configs
    configs = get_tokenizer_configs()
    
    # Train selected modalities
    all_results = {}
    
    if args.modality in ['eeg', 'both']:
        config = configs['VQVAE_EEG_30ch']
        config.epochs = args.epochs
        output_dir = output_base / config.name
        output_dir.mkdir(exist_ok=True)
        all_results['VQVAE_EEG_30ch'] = train_tokenizer(
            config, args.data_root, output_dir, args.device
        )
    
    if args.modality in ['fnirs', 'both']:
        config = configs['VQVAE_fNIRS_36ch']
        config.epochs = args.epochs
        output_dir = output_base / config.name
        output_dir.mkdir(exist_ok=True)
        all_results['VQVAE_fNIRS_36ch'] = train_tokenizer(
            config, args.data_root, output_dir, args.device
        )
    
    # Save combined summary
    summary = {
        'timestamp': timestamp,
        'device': args.device,
        'design_notes': {
            'eeg': '30 channels (EOG excluded), 4s window, 100 tokens, codebook=1024, dim=64',
            'fnirs': '36 HbO channels, 4s window, 10 tokens, codebook=512, dim=64',
            'alignment': 'Same window duration (4s), same embedding dim (64), same MI task offset (500ms)',
        },
    }
    
    for name, results in all_results.items():
        summary[name] = {
            'n_channels': results['actual_channels'],
            'test_mse': results['test_metrics']['recon_mse'],
            'test_perplexity': results['test_metrics']['perplexity'],
            'test_utilization': results['test_metrics']['code_utilization'],
            'test_dead_codes': results['test_metrics']['dead_codes'],
            'codebook_size': results['config']['codebook_size'],
            'embedding_dim': results['config']['embedding_dim'],
            'n_parameters': results['n_parameters'],
        }
    
    with open(output_base / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_base}")
    
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"{'Model':<20} {'Channels':<10} {'MSE':>10} {'Perplexity':>12} {'Utilization':>12} {'Dead':>8}")
    print("-"*90)
    for name, data in summary.items():
        if name in ['timestamp', 'device', 'design_notes']:
            continue
        print(f"{name:<20} {data['n_channels']:<10} {data['test_mse']:>10.6f} "
              f"{data['test_perplexity']:>12.1f} "
              f"{data['test_utilization']*100:>11.1f}% "
              f"{data['test_dead_codes']:>8}")
    print("="*90)
    
    return output_base


if __name__ == '__main__':
    main()
