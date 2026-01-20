"""
Train Patch-based VQ-VAE Tokenizers (LaBraM Standard)

Following STANDARDIZATION_GUIDE.md:
- Uses ExperimentLogger for config loading and metrics tracking
- Uses TokenizerVisualizer for generating figures
- Outputs to experiments/runs/<exp_name>_<timestamp>/
- Saves config.yaml, metrics.json, checkpoints/, figures/

Usage:
    python train_patch_vqvae.py --config phase0plus/eeg_patch_vqvae_1s.yaml
    python train_patch_vqvae.py --config phase0plus/fnirs_patch_vqvae_4s.yaml
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import ExperimentLogger
from src.tokenizers.patch_vqvae import PatchVQVAETokenizer
from src.data.eeg_fnirs_dataset import EEGfNIRSDataset
from src.visualization import TokenizerVisualizer


def create_dataloader(config: dict, split: str) -> DataLoader:
    """Create dataloader for specified split."""
    data_cfg = config['data']
    
    if split == 'train':
        subjects = data_cfg['split']['train_subjects']
        shuffle = True
    elif split == 'val':
        subjects = data_cfg['split']['val_subjects']
        shuffle = False
    else:
        subjects = data_cfg['split']['test_subjects']
        shuffle = False
    
    dataset = EEGfNIRSDataset(
        data_root=data_cfg['data_root'],
        modality=data_cfg['modality'],
        subject_ids=subjects,
        task=data_cfg.get('task', 'motor_imagery'),
        window_samples=data_cfg['window']['length'],
        window_offset_ms=data_cfg['window'].get('offset_ms', 0),
        normalize=True,
        exclude_eog=data_cfg.get('exclude_eog', False),
        hbo_only=data_cfg.get('hbo_only', False),
        hbr_only=data_cfg.get('hbr_only', False),
    )
    
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=True,
        drop_last=split == 'train',
    )


def create_tokenizer(config: dict) -> PatchVQVAETokenizer:
    """Create patch-based VQ-VAE tokenizer from config."""
    model_cfg = config['model']
    patch_cfg = model_cfg.get('patch', {})
    encoder_cfg = model_cfg.get('encoder', {})
    quantizer_cfg = model_cfg.get('quantizer', {})
    
    return PatchVQVAETokenizer(
        seq_length=model_cfg['seq_length'],
        patch_size=patch_cfg.get('size', 200),
        input_channels=model_cfg.get('input_channels', 1),
        codebook_size=quantizer_cfg.get('codebook_size', 1024),
        embedding_dim=quantizer_cfg.get('embedding_dim', 64),
        hidden_dim=encoder_cfg.get('hidden_dim', 256),
        num_layers=encoder_cfg.get('num_layers', 2),
        encoder_type=encoder_cfg.get('type', 'cnn'),
        commitment_cost=quantizer_cfg.get('commitment_cost', 0.25),
        ema_decay=quantizer_cfg.get('ema_decay', 0.99),
    )


def compute_spectral_loss(original: torch.Tensor, reconstructed: torch.Tensor, 
                          config: dict) -> torch.Tensor:
    """Compute multi-scale spectral loss."""
    spectral_cfg = config.get('loss', {}).get('spectral', {})
    if not spectral_cfg.get('enabled', False):
        return torch.tensor(0.0, device=original.device)
    
    fft_sizes = spectral_cfg.get('fft_sizes', [64, 128, 256])
    hop_sizes = spectral_cfg.get('hop_sizes', [16, 32, 64])
    win_sizes = spectral_cfg.get('win_sizes', [64, 128, 256])
    
    total_loss = torch.tensor(0.0, device=original.device)
    
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        # Simple FFT-based spectral loss
        if original.shape[-1] >= fft_size:
            # Compute magnitude spectrum
            orig_fft = torch.fft.rfft(original, n=fft_size, dim=-1)
            rec_fft = torch.fft.rfft(reconstructed, n=fft_size, dim=-1)
            
            orig_mag = torch.abs(orig_fft)
            rec_mag = torch.abs(rec_fft)
            
            # Log magnitude loss (more sensitive to low-amplitude frequencies)
            eps = 1e-8
            log_orig = torch.log(orig_mag + eps)
            log_rec = torch.log(rec_mag + eps)
            
            total_loss = total_loss + F.mse_loss(log_rec, log_orig)
    
    return total_loss / len(fft_sizes)


def compute_smoothness_loss(original: torch.Tensor, reconstructed: torch.Tensor,
                            config: dict) -> torch.Tensor:
    """Compute temporal smoothness loss (L2 norm of second derivative)."""
    smoothness_cfg = config.get('loss', {}).get('smoothness', {})
    if not smoothness_cfg.get('enabled', False):
        return torch.tensor(0.0, device=original.device)
    
    # Second derivative of reconstructed signal
    diff1 = reconstructed[:, 1:] - reconstructed[:, :-1]
    diff2 = diff1[:, 1:] - diff1[:, :-1]
    
    # Same for original
    orig_diff1 = original[:, 1:] - original[:, :-1]
    orig_diff2 = orig_diff1[:, 1:] - orig_diff1[:, :-1]
    
    # Match second derivative to preserve curvature
    return F.mse_loss(diff2, orig_diff2)


def compute_derivative_loss(original: torch.Tensor, reconstructed: torch.Tensor,
                            config: dict) -> torch.Tensor:
    """Compute first derivative matching loss for trend tracking."""
    deriv_cfg = config.get('loss', {}).get('derivative', {})
    if not deriv_cfg.get('enabled', False):
        return torch.tensor(0.0, device=original.device)
    
    # First derivative
    orig_deriv = original[:, 1:] - original[:, :-1]
    rec_deriv = reconstructed[:, 1:] - reconstructed[:, :-1]
    
    return F.mse_loss(rec_deriv, orig_deriv)


def train_epoch(
    tokenizer: PatchVQVAETokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict:
    """Train for one epoch."""
    tokenizer.train()
    
    total_loss = 0.0
    total_rec_loss = 0.0
    total_vq_loss = 0.0
    total_samples = 0
    total_perplexity = 0.0
    total_utilization = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Get data
        if isinstance(batch, dict):
            x = batch['data']  # [B, C, T]
        else:
            x = batch[0]
        
        x = x.to(device)
        B, C, T = x.shape
        
        # Process each channel independently
        x_flat = x.view(B * C, T)  # [B*C, T]
        
        # Forward pass
        outputs = tokenizer(x_flat)
        
        # Reconstruction loss
        rec_loss = F.mse_loss(outputs['x_rec'], x_flat)
        
        # VQ loss (commitment)
        vq_loss = outputs['commitment_loss']
        
        # Additional losses
        spectral_loss = compute_spectral_loss(x_flat, outputs['x_rec'], config)
        smoothness_loss = compute_smoothness_loss(x_flat, outputs['x_rec'], config)
        derivative_loss = compute_derivative_loss(x_flat, outputs['x_rec'], config)
        
        # Total loss with weights
        loss_cfg = config.get('loss', {})
        rec_weight = loss_cfg.get('reconstruction', {}).get('weight', 1.0)
        spectral_weight = loss_cfg.get('spectral', {}).get('weight', 0.0)
        smoothness_weight = loss_cfg.get('smoothness', {}).get('weight', 0.0)
        derivative_weight = loss_cfg.get('derivative', {}).get('weight', 0.0)
        
        loss = (rec_weight * rec_loss + vq_loss + 
                spectral_weight * spectral_loss +
                smoothness_weight * smoothness_loss +
                derivative_weight * derivative_loss)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_clip = config['training'].get('gradient_clip', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), grad_clip)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * B
        total_rec_loss += rec_loss.item() * B
        total_vq_loss += vq_loss.item() * B
        total_samples += B
        total_perplexity += outputs['perplexity'].item()
        total_utilization += outputs['code_utilization'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / total_samples,
        'rec_loss': total_rec_loss / total_samples,
        'vq_loss': total_vq_loss / total_samples,
        'perplexity': total_perplexity / n_batches,
        'utilization': total_utilization / n_batches,
    }


@torch.no_grad()
def validate(
    tokenizer: PatchVQVAETokenizer,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate tokenizer."""
    tokenizer.eval()
    
    total_rec_loss = 0.0
    total_samples = 0
    all_indices = []
    
    for batch in dataloader:
        if isinstance(batch, dict):
            x = batch['data']
        else:
            x = batch[0]
        
        x = x.to(device)
        B, C, T = x.shape
        
        x_flat = x.view(B * C, T)
        outputs = tokenizer(x_flat)
        
        rec_loss = F.mse_loss(outputs['x_rec'], x_flat)
        total_rec_loss += rec_loss.item() * B
        total_samples += B
        
        all_indices.append(outputs['indices'].cpu())
    
    # Compute code utilization
    all_indices = torch.cat(all_indices, dim=0).flatten()
    unique_codes = torch.unique(all_indices)
    utilization = len(unique_codes) / tokenizer.codebook_size
    
    return {
        'val_rec_loss': total_rec_loss / total_samples,
        'val_utilization': utilization,
        'val_unique_codes': len(unique_codes),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Patch VQ-VAE Tokenizer")
    parser.add_argument('--config', type=str, required=True,
                        help='Config file path (relative to experiments/configs/)')
    args = parser.parse_args()
    
    # Initialize ExperimentLogger (handles config loading and run directory)
    logger = ExperimentLogger(config_path=args.config)
    config = logger.config
    
    print(f"\n{'='*60}")
    print(f"Training Patch-based VQ-VAE Tokenizer")
    print(f"{'='*60}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment'].get('description', 'N/A')}")
    print(f"Modality: {config['data']['modality']}")
    
    # Device
    device = torch.device(config['experiment'].get('device', 'cuda') 
                          if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Seed
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader = create_dataloader(config, 'train')
    val_loader = create_dataloader(config, 'val')
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create tokenizer
    print("\nCreating tokenizer...")
    tokenizer = create_tokenizer(config).to(device)
    
    # Token specification info
    patch_cfg = config['model'].get('patch', {})
    patch_size = patch_cfg.get('size', 200)
    seq_length = config['model']['seq_length']
    n_tokens = seq_length // patch_size
    sr = config['data']['preprocessing'].get('resample_rate', 200)
    token_duration = patch_size / sr
    
    print(f"\n{'='*40}")
    print("Token Specification:")
    print(f"  Patch size: {patch_size} samples ({token_duration:.2f}s)")
    print(f"  Tokens per window: {n_tokens}")
    print(f"  Codebook size: {tokenizer.codebook_size}")
    print(f"  Embedding dim: {tokenizer.embedding_dim}")
    n_channels = config['data'].get('n_channels', 1)
    print(f"  Tokens per trial: {n_tokens} × {n_channels} = {n_tokens * n_channels}")
    print(f"{'='*40}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0001),
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['learning_rate'] / 100,
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training'].get('early_stopping', {}).get('patience', 20)
    
    print(f"Starting training for {config['training']['epochs']} epochs...")
    print(f"Early stopping patience: {patience}\n")
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            tokenizer, train_loader, optimizer, device, config
        )
        
        # Validate
        val_metrics = validate(tokenizer, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch with ExperimentLogger
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['val_rec_loss'],
            loss_breakdown={
                'reconstruction': train_metrics['rec_loss'],
                'vq_commitment': train_metrics['vq_loss'],
            },
            metrics={
                'perplexity': train_metrics['perplexity'],
                'train_utilization': train_metrics['utilization'],
                'val_utilization': val_metrics['val_utilization'],
                'val_unique_codes': val_metrics['val_unique_codes'],
            }
        )
        
        # Print epoch summary
        print(f"  Val - Rec: {val_metrics['val_rec_loss']:.4f}, "
              f"Util: {val_metrics['val_utilization']*100:.1f}% "
              f"({val_metrics['val_unique_codes']}/{tokenizer.codebook_size})")
        
        # Early stopping check
        if val_metrics['val_rec_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_rec_loss']
            patience_counter = 0
            
            # Save best model
            logger.save_checkpoint(
                state_dict={
                    'model_state_dict': tokenizer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                },
                epoch=epoch,
                is_best=True
            )
            print(f"  ★ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break
        
        # Save checkpoint periodically
        save_every = config['logging'].get('save_checkpoint_every', 10)
        if epoch % save_every == 0:
            logger.save_checkpoint(
                state_dict={
                    'model_state_dict': tokenizer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                },
                epoch=epoch
            )
    
    # Final metrics
    final_metrics = {
        'best_val_rec_loss': best_val_loss,
        'final_val_utilization': val_metrics['val_utilization'],
        'final_unique_codes': val_metrics['val_unique_codes'],
        'codebook_size': tokenizer.codebook_size,
        'tokens_per_window': n_tokens,
        'patch_size_samples': patch_size,
        'patch_duration_seconds': token_duration,
    }
    
    logger.log_final(final_metrics)
    
    # =========================================================================
    # Generate Visualizations (following STANDARDIZATION_GUIDE.md)
    # =========================================================================
    print("\nGenerating visualizations...")
    
    # Initialize TokenizerVisualizer
    visualizer = TokenizerVisualizer(logger.run_dir)
    
    # 1. Training curves from metrics history
    metrics_history = logger.get_metrics_history()
    visualizer.plot_training_curves(metrics_history)
    
    # 2. Get validation samples for reconstruction visualization
    print("  Collecting samples for visualization...")
    tokenizer.eval()
    all_originals = []
    all_reconstructed = []
    all_indices = []
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                x = batch['data']
            else:
                x = batch[0]
            
            x = x.to(device)
            B, C, T = x.shape
            x_flat = x.view(B * C, T)
            
            outputs = tokenizer(x_flat)
            all_originals.append(x_flat.cpu())
            all_reconstructed.append(outputs['x_rec'].cpu())
            all_indices.append(outputs['indices'].cpu())
            
            # Limit to ~200 samples
            if sum(o.shape[0] for o in all_originals) >= 200:
                break
    
    original = torch.cat(all_originals, dim=0)
    reconstructed = torch.cat(all_reconstructed, dim=0)
    indices = torch.cat(all_indices, dim=0)
    
    # 3. Reconstruction samples
    visualizer.plot_reconstruction_samples(
        original, reconstructed, 
        n_samples=4, 
        fs=sr
    )
    
    # 4. Spectral comparison
    visualizer.plot_spectral_comparison(
        original, reconstructed, 
        fs=sr,
        n_samples=100
    )
    
    # 5. Codebook usage histogram
    visualizer.plot_codebook_usage(indices, tokenizer.codebook_size)
    
    # 6. Token embeddings (t-SNE/PCA)
    embeddings = tokenizer.get_codebook_embeddings()
    if embeddings is not None:
        flat_indices = indices.flatten()
        usage = torch.bincount(flat_indices, minlength=tokenizer.codebook_size)
        visualizer.plot_token_embeddings(embeddings, usage)
    
    # 7. Summary figure
    visualizer.generate_summary_figure(final_metrics, config)
    
    # Save figure manifest
    visualizer.save_figure_manifest()
    
    # Also generate default logger figures
    logger.generate_figures()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final utilization: {val_metrics['val_utilization']*100:.1f}%")
    print(f"Run directory: {logger.run_dir}")
    print(f"Generated figures: {len(visualizer.get_generated_figures())}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
