#!/usr/bin/env python
"""
Train tokenizer (FSQ / VQ-VAE) on EEG/fNIRS data.

Usage:
    python train_tokenizer.py --config phase0/P0_eeg_fsq.yaml
    python train_tokenizer.py --config phase0/P0_eeg_vqvae.yaml --epochs 50
"""

# Fix OMP duplicate library error (must be before numpy/torch imports)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import visualization module
from src.visualization.tokenizer_plots import TokenizerVisualizer, visualize_tokenizer_run
sys.path.insert(0, str(project_root))


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration with base config inheritance."""
    configs_dir = project_root / "experiments" / "configs"
    
    # Load base config
    base_path = configs_dir / "base.yaml"
    with open(base_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load experiment-specific config and merge
    exp_path = configs_dir / config_path
    if exp_path.exists():
        with open(exp_path, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f)
        config = deep_merge(config, exp_config)
    
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_config_hash(config: dict) -> str:
    """Generate a short hash of config for identification."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


# =============================================================================
# Placeholder Dataset (to be replaced with real EEG/fNIRS loaders)
# =============================================================================

class PlaceholderDataset(Dataset):
    """
    Placeholder dataset that generates synthetic data for testing.
    Replace with real EEG/fNIRS dataset implementations.
    """
    
    def __init__(
        self, 
        modality: str = "eeg",
        n_samples: int = 5000,
        seq_length: int = 512,
        n_channels: int = 1,
        seed: int = 42
    ):
        super().__init__()
        self.modality = modality
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_channels = n_channels
        
        # Generate synthetic data
        np.random.seed(seed)
        
        if modality == "eeg":
            # EEG-like: Mix of oscillations (alpha, beta) + noise
            fs = 200  # Hz
            t = np.linspace(0, seq_length / fs, seq_length)
            self.data = np.zeros((n_samples, n_channels, seq_length), dtype=np.float32)
            
            for i in range(n_samples):
                alpha = np.sin(2 * np.pi * (8 + np.random.rand() * 4) * t)  # 8-12 Hz
                beta = 0.5 * np.sin(2 * np.pi * (15 + np.random.rand() * 10) * t)  # 15-25 Hz
                noise = 0.3 * np.random.randn(seq_length)
                signal = alpha + beta + noise
                self.data[i, 0] = signal.astype(np.float32)
                
        else:  # fnirs
            # fNIRS-like: Low frequency oscillations + hemodynamic response
            fs = 10  # Hz
            t = np.linspace(0, seq_length / fs, seq_length)
            self.data = np.zeros((n_samples, n_channels, seq_length), dtype=np.float32)
            
            for i in range(n_samples):
                # Slow drift
                drift = 0.1 * np.sin(2 * np.pi * 0.05 * t)
                # Mayer waves (~0.1 Hz)
                mayer = 0.2 * np.sin(2 * np.pi * 0.1 * t)
                # Random HRF-like response
                hrf_onset = np.random.randint(0, seq_length // 2)
                hrf = np.zeros(seq_length)
                if hrf_onset < seq_length - 20:
                    hrf_kernel = np.exp(-np.linspace(0, 3, 20)) * np.linspace(0, 1, 20)
                    hrf[hrf_onset:hrf_onset+20] = hrf_kernel
                noise = 0.1 * np.random.randn(seq_length)
                signal = drift + mayer + hrf + noise
                self.data[i, 0] = signal.astype(np.float32)
        
        # Normalize
        self.data = (self.data - self.data.mean()) / (self.data.std() + 1e-8)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        if self.n_channels == 1:
            x = x.squeeze(0)  # [T] for single channel
        return {'x': x, 'idx': idx}


# =============================================================================
# Training Utilities
# =============================================================================

def create_model(config: dict) -> nn.Module:
    """Create tokenizer model based on config."""
    model_type = config['model']['type']
    
    if model_type == "fsq":
        from src.tokenizers.fsq import FSQTokenizer
        return FSQTokenizer(
            seq_length=config['model']['seq_length'],
            input_channels=config['model'].get('input_channels', 1),
            levels=config['model']['quantizer']['levels'],
            encoder_dims=config['model']['encoder']['hidden_dims'],
            encoder_kernel=config['model']['encoder']['kernel_size'],
            encoder_stride=config['model']['encoder']['stride'],
        )
    elif model_type == "vqvae":
        from src.tokenizers.vqvae import VQVAETokenizer
        return VQVAETokenizer(
            seq_length=config['model']['seq_length'],
            input_channels=config['model'].get('input_channels', 1),
            codebook_size=config['model']['quantizer']['codebook_size'],
            embedding_dim=config['model']['quantizer']['embedding_dim'],
            commitment_cost=config['model']['quantizer'].get('commitment_cost', 0.25),
            ema_decay=config['model']['quantizer'].get('ema_decay', 0.99),
            encoder_dims=config['model']['encoder']['hidden_dims'],
            encoder_kernel=config['model']['encoder']['kernel_size'],
            encoder_stride=config['model']['encoder']['stride'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_losses(outputs: dict, batch: dict, config: dict, device: torch.device) -> dict:
    """Compute all loss components."""
    x = batch['x'].to(device)
    x_rec = outputs['x_rec']
    
    loss_config = config.get('loss', {})
    
    # Reconstruction loss
    recon_type = loss_config.get('reconstruction', {}).get('type', 'mse')
    if recon_type == 'mse':
        l_recon = F.mse_loss(x_rec, x)
    elif recon_type == 'huber':
        l_recon = F.smooth_l1_loss(x_rec, x)
    else:
        l_recon = F.mse_loss(x_rec, x)
    
    # Spectral loss (optional)
    l_spectral = torch.tensor(0.0, device=device)
    spectral_config = loss_config.get('spectral', {})
    if spectral_config.get('enabled', False):
        l_spectral = compute_spectral_loss(x, x_rec, spectral_config)
    
    # Commitment loss (for VQ-VAE)
    l_commit = outputs.get('commitment_loss', torch.tensor(0.0, device=device))
    
    # Weighted sum
    w_recon = loss_config.get('reconstruction', {}).get('weight', 1.0)
    w_spectral = spectral_config.get('weight', 0.1) if spectral_config.get('enabled', False) else 0.0
    w_commit = loss_config.get('commitment', {}).get('weight', 0.25)
    
    total = w_recon * l_recon + w_spectral * l_spectral + w_commit * l_commit
    
    return {
        'total': total,
        'reconstruction_mse': l_recon.item(),
        'spectral_mse': l_spectral.item() if isinstance(l_spectral, torch.Tensor) else l_spectral,
        'commitment_loss': l_commit.item() if isinstance(l_commit, torch.Tensor) else l_commit,
    }


def compute_spectral_loss(x: torch.Tensor, x_rec: torch.Tensor, config: dict) -> torch.Tensor:
    """Compute multi-scale STFT loss."""
    fft_sizes = config.get('fft_sizes', [64, 128, 256])
    
    loss = torch.tensor(0.0, device=x.device)
    
    for n_fft in fft_sizes:
        # Ensure x and x_rec are [B, T]
        if x.dim() == 3:
            x_2d = x.squeeze(1)
            x_rec_2d = x_rec.squeeze(1)
        else:
            x_2d = x
            x_rec_2d = x_rec
        
        # Compute STFT
        window = torch.hann_window(n_fft, device=x.device)
        
        # Pad if needed
        if x_2d.shape[-1] < n_fft:
            pad_size = n_fft - x_2d.shape[-1]
            x_2d = F.pad(x_2d, (0, pad_size))
            x_rec_2d = F.pad(x_rec_2d, (0, pad_size))
        
        stft_x = torch.stft(x_2d, n_fft, hop_length=n_fft//4, window=window, return_complex=True)
        stft_rec = torch.stft(x_rec_2d, n_fft, hop_length=n_fft//4, window=window, return_complex=True)
        
        # Magnitude loss
        mag_x = stft_x.abs()
        mag_rec = stft_rec.abs()
        loss = loss + F.mse_loss(mag_rec, mag_x)
    
    return loss / len(fft_sizes)


def compute_codebook_health(indices: torch.Tensor, codebook_size: int) -> dict:
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
        'code_utilization': utilization.item(),
        'dead_codes': dead_codes.item(),
        'active_codes': active_codes.item(),
    }


# =============================================================================
# Experiment Logger
# =============================================================================

class TokenizerLogger:
    """Logger for tokenizer training experiments."""
    
    def __init__(self, config: dict, config_path: str):
        self.config = config
        self.config_path = config_path
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config['experiment']['name']
        self.run_name = f"{exp_name}_{timestamp}"
        
        self.runs_dir = project_root / "experiments" / "runs"
        self.run_dir = self.runs_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        
        # Save config
        with open(self.run_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Initialize metrics storage
        self.metrics = {
            'config_hash': get_config_hash(config),
            'started_at': datetime.now().isoformat(),
            'epochs': [],
        }
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for an epoch."""
        self.metrics['epochs'].append({
            'epoch': epoch,
            **metrics
        })
        
        # Save metrics incrementally
        with open(self.run_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_checkpoint(self, state: dict, epoch: int):
        """Save model checkpoint."""
        path = self.run_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, path)
    
    def save_final(self, final_metrics: dict):
        """Save final metrics and model."""
        self.metrics['completed_at'] = datetime.now().isoformat()
        self.metrics['final_metrics'] = final_metrics
        
        with open(self.run_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"[Logger] Results saved to: {self.run_dir}")


# =============================================================================
# Main Training Function
# =============================================================================

def train(config_path: str, epochs: Optional[int] = None):
    """Run tokenizer training."""
    
    # Load config
    config = load_config(config_path)
    
    # Override epochs if specified
    if epochs is not None:
        config['training']['epochs'] = epochs
    
    # Setup
    device = torch.device(config['experiment'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"[Config] Loaded: {config_path}")
    print(f"[Device] Using: {device}")
    print(f"[Seed] {seed}")
    
    # Create dataset (placeholder - replace with real data loading)
    print("[Data] Creating dataset...")
    dataset = PlaceholderDataset(
        modality=config['data']['modality'],
        n_samples=5000,
        seq_length=config['data']['window']['length'],
        seed=seed
    )
    
    # Split train/val/test
    split = config['data'].get('split', {'train': 0.8, 'val': 0.1, 'test': 0.1})
    n_total = len(dataset)
    n_train = int(n_total * split['train'])
    n_val = int(n_total * split['val'])
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    print(f"[Data] Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Create model
    print("[Model] Creating tokenizer...")
    model = create_model(config).to(device)
    codebook_size = model.get_codebook_size()
    print(f"[Model] Type: {config['model']['type']}, Codebook size: {codebook_size}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )
    
    # Scheduler
    scheduler_type = config['training'].get('scheduler', 'cosine')
    num_epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs
        )
    else:
        scheduler = None
    
    # Logger
    logger = TokenizerLogger(config, config_path)
    
    # Training loop
    print(f"[Training] Starting {num_epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_config = config['training'].get('early_stopping', {})
    patience = early_stop_config.get('patience', 20)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_losses = []
        all_indices = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            x = batch['x'].to(device)
            outputs = model(x)
            losses = compute_losses(outputs, batch, config, device)
            
            losses['total'].backward()
            
            # Gradient clipping
            grad_clip = config['training'].get('gradient_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            optimizer.step()
            
            train_losses.append({k: v for k, v in losses.items() if k != 'total'})
            all_indices.append(outputs['indices'].detach().cpu())
        
        # Scheduler step
        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        val_indices = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                outputs = model(x)
                losses = compute_losses(outputs, batch, config, device)
                val_losses.append({k: v for k, v in losses.items() if k != 'total'})
                val_indices.append(outputs['indices'].cpu())
        
        # Aggregate metrics
        train_metrics = {k: np.mean([l[k] for l in train_losses]) for k in train_losses[0]}
        val_metrics = {f"val_{k}": np.mean([l[k] for l in val_losses]) for k in val_losses[0]}
        
        # Codebook health
        all_train_indices = torch.cat(all_indices, dim=0)
        health_metrics = compute_codebook_health(all_train_indices, codebook_size)
        
        # Combine metrics
        epoch_metrics = {
            **train_metrics,
            **val_metrics,
            **health_metrics,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        # Log
        logger.log_epoch(epoch, epoch_metrics)
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{num_epochs}] "
                  f"recon={train_metrics['reconstruction_mse']:.4f}, "
                  f"val_recon={val_metrics['val_reconstruction_mse']:.4f}, "
                  f"perplexity={health_metrics['perplexity']:.1f}, "
                  f"util={health_metrics['code_utilization']:.3f}")
        
        # Checkpoint
        save_every = config['logging'].get('save_checkpoint_every', 10)
        if epoch % save_every == 0:
            logger.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': epoch_metrics,
            }, epoch)
        
        # Early stopping
        val_loss = val_metrics['val_reconstruction_mse']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            logger.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': epoch_metrics,
            }, epoch=0)  # epoch=0 indicates best model
        else:
            patience_counter += 1
            if early_stop_config.get('enabled', True) and patience_counter >= patience:
                print(f"[Early Stop] No improvement for {patience} epochs. Stopping.")
                break
    
    # Final evaluation
    print("[Evaluation] Computing final metrics...")
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_losses = []
    test_indices = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            outputs = model(x)
            losses = compute_losses(outputs, batch, config, device)
            test_losses.append({k: v for k, v in losses.items() if k != 'total'})
            test_indices.append(outputs['indices'].cpu())
    
    test_metrics = {f"test_{k}": np.mean([l[k] for l in test_losses]) for k in test_losses[0]}
    all_test_indices = torch.cat(test_indices, dim=0)
    test_health = compute_codebook_health(all_test_indices, codebook_size)
    test_health = {f"test_{k}": v for k, v in test_health.items()}
    
    final_metrics = {
        **test_metrics,
        **test_health,
        'best_val_reconstruction_mse': best_val_loss,
    }
    
    logger.save_final(final_metrics)
    
    print(f"\n[Done] Experiment: {logger.run_name}")
    print(f"[Results] Final test metrics:")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # ==========================================================================
    # Generate Visualizations
    # ==========================================================================
    print("\n[Visualization] Generating figures...")
    try:
        # Get metrics history from logger
        metrics_history = logger.metrics.get('epochs', [])
        
        # Generate all visualizations
        generated_figures = visualize_tokenizer_run(
            run_dir=logger.run_dir,
            model=model,
            test_loader=test_loader,
            metrics_history=metrics_history,
            config=config,
            final_metrics=final_metrics,
            device=device
        )
        
        print(f"[Visualization] Generated {len(generated_figures)} figures:")
        for fig_path in generated_figures:
            print(f"  - {fig_path.name}")
    except Exception as e:
        print(f"[Visualization] Warning: Failed to generate visualizations: {e}")
    
    return logger.run_name, final_metrics


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tokenizer on neural signals")
    parser.add_argument("--config", type=str, required=True, 
                        help="Config file path relative to experiments/configs/")
    parser.add_argument("--epochs", type=int, default=None, 
                        help="Override number of epochs")
    
    args = parser.parse_args()
    train(args.config, args.epochs)
