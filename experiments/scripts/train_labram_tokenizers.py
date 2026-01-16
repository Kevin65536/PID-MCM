#!/usr/bin/env python
"""
Train LaBraM-standard tokenizers with aligned configurations.

按照LaBraM/NeuroLM/NeuroRVQ标准训练tokenizer:
- EEG: 200 samples @ 200Hz = 1.0s patch, codebook=8192, dim=64
- fNIRS: 10 samples @ 10Hz = 1.0s patch, 对齐EEG的codebook和dim设置

Usage:
    python train_labram_tokenizers.py --modality eeg --epochs 100
    python train_labram_tokenizers.py --modality fnirs --epochs 100
    python train_labram_tokenizers.py --modality both --epochs 100
"""

# Fix OMP duplicate library error
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
from src.tokenizers.fsq import FSQTokenizer
from src.tokenizers.vqvae import VQVAETokenizer
from src.metrics.codebook_health import compute_codebook_health
from src.visualization.tokenizer_plots import TokenizerVisualizer


@dataclass
class LaBraMTokenizerConfig:
    """
    LaBraM-standard tokenizer configuration.
    
    基于文献标准:
    - LaBraM: patch=200@200Hz=1s, codebook=8192, dim=64
    - NeuroLM: patch=200@200Hz=1s, codebook=8192, dim=128
    - NeuroRVQ: patch=200@200Hz=1s, codebook=8192, dim=128
    """
    name: str
    modality: str
    model_type: str
    
    # Patch configuration (LaBraM standard: 1 second)
    seq_length: int           # samples per patch
    sample_rate: float        # Hz
    time_duration_s: float    # seconds (should be 1.0 for LaBraM standard)
    input_channels: int = 1
    
    # Encoder/Decoder architecture
    encoder_dims: tuple = None
    kernel_size: int = 7
    stride: int = 2
    
    # Quantizer (LaBraM standard: 8192 codes, 64-128 dim)
    codebook_size: int = 8192
    embedding_dim: int = 64
    levels: tuple = None        # For FSQ
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    early_stopping_patience: int = 20


def get_labram_configs() -> Dict[str, LaBraMTokenizerConfig]:
    """
    获取LaBraM标准tokenizer配置。
    
    设计原则:
    1. EEG使用MI-BCI标准: 4秒窗口（典型MI任务长度）
    2. fNIRS与EEG对齐: 同样4秒窗口, 相同嵌入维度
    3. 码本大小根据数据量适当调整以保证利用率
    
    MI-BCI标准参考:
    - 窗口长度: 4秒 (typical for motor imagery)
    - 窗口偏移: 0.5秒 (response delay after cue)
    """
    
    return {
        # EEG: 4s MI window @ 200Hz = 800 samples
        'VQVAE_EEG_LaBraM': LaBraMTokenizerConfig(
            name='VQVAE_EEG_LaBraM',
            modality='eeg',
            model_type='vqvae',
            seq_length=800,           # 4.0s @ 200Hz (MI standard)
            sample_rate=200.0,
            time_duration_s=4.0,
            encoder_dims=(64, 128, 256),  # 3层编码器
            kernel_size=7,
            stride=2,                  # 输出: 800/8 = 100 tokens
            codebook_size=1024,
            embedding_dim=64,
            commitment_cost=0.25,
            ema_decay=0.99,
        ),
        
        # fNIRS: 4s MI window @ 10Hz = 40 samples
        'VQVAE_fNIRS_Aligned': LaBraMTokenizerConfig(
            name='VQVAE_fNIRS_Aligned',
            modality='fnirs',
            model_type='vqvae',
            seq_length=40,            # 4.0s @ 10Hz
            sample_rate=10.0,
            time_duration_s=4.0,
            encoder_dims=(64, 128),   # 2层
            kernel_size=5,
            stride=2,                  # 输出: 40/4 = 10 tokens
            codebook_size=1024,
            embedding_dim=64,
            commitment_cost=0.25,
            ema_decay=0.99,
        ),
        
        # 可选: FSQ版本的fNIRS
        'FSQ_fNIRS_Aligned': LaBraMTokenizerConfig(
            name='FSQ_fNIRS_Aligned',
            modality='fnirs',
            model_type='fsq',
            seq_length=40,            # 4.0s @ 10Hz
            sample_rate=10.0,
            time_duration_s=4.0,
            encoder_dims=(64, 128),
            kernel_size=5,
            stride=2,
            levels=(8, 8, 8, 8),      # 4096 codes, dim=4
            embedding_dim=4,
        ),
    }


def create_model(config: LaBraMTokenizerConfig, device: str) -> nn.Module:
    """Create tokenizer model from config."""
    
    if config.model_type == 'fsq':
        model = FSQTokenizer(
            seq_length=config.seq_length,
            input_channels=config.input_channels,
            levels=list(config.levels),
            encoder_dims=list(config.encoder_dims),
            encoder_kernel=config.kernel_size,
            encoder_stride=config.stride,
        )
    elif config.model_type == 'vqvae':
        model = VQVAETokenizer(
            seq_length=config.seq_length,
            input_channels=config.input_channels,
            codebook_size=config.codebook_size,
            embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost,
            ema_decay=config.ema_decay,
            encoder_dims=list(config.encoder_dims),
            encoder_kernel=config.kernel_size,
            encoder_stride=config.stride,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model.to(device)


def prepare_batch(batch: Dict, device: str, channel_mode: str = 'average') -> torch.Tensor:
    """
    Prepare batch data for training.
    
    Args:
        batch: Dict with 'data' of shape [B, C, T]
        device: Target device
        channel_mode: 'average' to average all channels, or int for specific channel
        
    Returns:
        Tensor of shape [B, T] for single-channel tokenizer
    """
    data = batch['data'].to(device)  # [B, C, T]
    
    if channel_mode == 'average':
        # Average all channels
        data = data.mean(dim=1)  # [B, T]
    elif isinstance(channel_mode, int):
        # Select specific channel
        data = data[:, channel_mode, :]  # [B, T]
    
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
        
        # Save samples for visualization
        if len(all_originals) < 10:
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
    config: LaBraMTokenizerConfig,
    data_root: str,
    output_dir: Path,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Train a single tokenizer with LaBraM standard configuration.
    
    Returns:
        Dict with training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {config.name}")
    print(f"{'='*60}")
    print(f"Modality: {config.modality}")
    print(f"Model: {config.model_type}")
    print(f"Patch: {config.seq_length} samples @ {config.sample_rate}Hz = {config.time_duration_s}s")
    print(f"Codebook: {config.codebook_size} codes, dim={config.embedding_dim}")
    
    # Create dataloaders
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
    )
    
    print(f"Train: {len(dataloaders['train'].dataset)} samples")
    print(f"Val: {len(dataloaders['val'].dataset)} samples")
    print(f"Test: {len(dataloaders['test'].dataset)} samples")
    
    # Create model
    model = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Verify output shape
    sample_batch = next(iter(dataloaders['train']))
    sample_x = prepare_batch(sample_batch, device)
    with torch.no_grad():
        sample_out = model(sample_x)
    print(f"Input shape: {sample_x.shape}")
    print(f"Token shape: {sample_out['indices'].shape}")
    print(f"Latent dim: {model.latent_dim}")
    print(f"Recon shape: {sample_out['x_rec'].shape}")
    
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
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}: train_loss={train_metrics['loss']:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"perplexity={train_metrics['perplexity']:.1f}, "
                  f"utilization={train_metrics['code_utilization']*100:.1f}%")
        
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    # Load best model and evaluate on test
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, dataloaders['test'], device)
    
    # Generate visualizations
    print("\n[Visualization] Generating figures...")
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
        'n_parameters': n_params,
        'training_time_s': training_time,
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'test_metrics': test_metrics_clean,
        'history': history,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics_clean['loss']:.4f}")
    print(f"  Recon MSE: {test_metrics_clean['recon_mse']:.4f}")
    print(f"  Perplexity: {test_metrics_clean['perplexity']:.1f}")
    print(f"  Utilization: {test_metrics_clean['code_utilization']*100:.1f}%")
    print(f"  Dead codes: {test_metrics_clean['dead_codes']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train LaBraM-standard tokenizers')
    parser.add_argument('--modality', type=str, default='both',
                        choices=['eeg', 'fnirs', 'both'],
                        help='Which modality to train')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
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
    output_base = project_root / 'experiments' / 'runs' / f'LaBraM_tokenizers_{timestamp}'
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Get configs
    configs = get_labram_configs()
    
    # Train selected modalities
    all_results = {}
    
    if args.modality in ['eeg', 'both']:
        config = configs['VQVAE_EEG_LaBraM']
        config.epochs = args.epochs
        output_dir = output_base / config.name
        output_dir.mkdir(exist_ok=True)
        all_results['VQVAE_EEG_LaBraM'] = train_tokenizer(
            config, args.data_root, output_dir, args.device
        )
    
    if args.modality in ['fnirs', 'both']:
        config = configs['VQVAE_fNIRS_Aligned']
        config.epochs = args.epochs
        output_dir = output_base / config.name
        output_dir.mkdir(exist_ok=True)
        all_results['VQVAE_fNIRS_Aligned'] = train_tokenizer(
            config, args.data_root, output_dir, args.device
        )
    
    # Save combined results
    summary = {
        'timestamp': timestamp,
        'device': args.device,
        'standard': 'LaBraM (1s patch, 8192 codebook)',
    }
    for name, results in all_results.items():
        summary[name] = {
            'test_recon_mse': results['test_metrics']['recon_mse'],
            'test_perplexity': results['test_metrics']['perplexity'],
            'test_utilization': results['test_metrics']['code_utilization'],
            'test_dead_codes': results['test_metrics']['dead_codes'],
            'n_parameters': results['n_parameters'],
            'latent_dim': results['config']['embedding_dim'],
        }
    
    with open(output_base / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_base}")
    
    # Print summary table
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Model':<25} {'MSE':>10} {'Perplexity':>12} {'Utilization':>12} {'Dead':>8}")
    print("-" * 80)
    for name, data in summary.items():
        if name in ['timestamp', 'device', 'standard']:
            continue
        print(f"{name:<25} {data['test_recon_mse']:>10.4f} "
              f"{data['test_perplexity']:>12.1f} "
              f"{data['test_utilization']*100:>11.1f}% "
              f"{data['test_dead_codes']:>8}")
    
    return output_base


if __name__ == '__main__':
    main()
