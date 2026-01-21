"""
Train Frequency-Domain Patch VQ-VAE Tokenizers

This script trains the new frequency-domain tokenizers that:
1. Encode in FFT domain (amplitude + phase)
2. Use multi-scale temporal encoders
3. Reconstruct via inverse FFT

Usage:
    python train_freq_tokenizer.py --config phase0plus/eeg_freq_patch_vqvae_1s.yaml
    python train_freq_tokenizer.py --config phase0plus/eeg_freq_patch_vqvae_v2.yaml
    python train_freq_tokenizer.py --config phase0plus/fnirs_freq_patch_vqvae_1s.yaml
    
后台运行 (推荐):
    nohup python train_freq_tokenizer.py --config phase0plus/eeg_freq_patch_vqvae_1s.yaml &
"""

import sys
import os
import argparse
import traceback
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import ExperimentLogger
from src.tokenizers.freq_patch_vqvae import FreqDomainPatchVQVAE, FreqDomainPatchVQVAE_V2
from src.data.eeg_fnirs_dataset import EEGfNIRSDataset
from src.visualization import TokenizerVisualizer


class TeeLogger:
    """同时输出到终端和文件的日志类"""
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a', buffering=1)  # 行缓冲
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


def setup_logging(run_dir: Path) -> TeeLogger:
    """设置日志，同时输出到终端和文件"""
    log_file = run_dir / "training.log"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee


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


def create_tokenizer(config: dict):
    """Create frequency-domain VQ-VAE tokenizer from config."""
    model_cfg = config['model']
    patch_cfg = model_cfg.get('patch', {})
    encoder_cfg = model_cfg.get('encoder', {})
    quantizer_cfg = model_cfg.get('quantizer', {})
    loss_cfg = config.get('loss', {})
    
    # Determine model type
    model_type = model_cfg.get('type', 'freq_patch_vqvae')
    
    # Common kwargs
    kwargs = dict(
        seq_length=model_cfg['seq_length'],
        patch_size=patch_cfg.get('size', 200),
        input_channels=model_cfg.get('input_channels', 1),
        codebook_size=quantizer_cfg.get('codebook_size', 2048),
        embedding_dim=quantizer_cfg.get('embedding_dim', 64),
        hidden_dim=encoder_cfg.get('hidden_dim', 256),
        num_layers=encoder_cfg.get('num_layers', 2),
        encoder_type=encoder_cfg.get('type', 'multiscale'),
        commitment_cost=quantizer_cfg.get('commitment_cost', 0.1),
        ema_decay=quantizer_cfg.get('ema_decay', 0.99),
        amplitude_loss_weight=loss_cfg.get('amplitude', {}).get('weight', 1.0),
        phase_loss_weight=loss_cfg.get('phase', {}).get('weight', 0.5),
        time_loss_weight=loss_cfg.get('time', {}).get('weight', 0.5),
        use_log_amplitude=loss_cfg.get('use_log_amplitude', True),
    )
    
    if model_type == 'freq_patch_vqvae_v2':
        return FreqDomainPatchVQVAE_V2(**kwargs)
    else:
        return FreqDomainPatchVQVAE(**kwargs)


def train_epoch(
    tokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict:
    """Train for one epoch."""
    tokenizer.train()
    
    total_loss = 0.0
    total_rec_loss = 0.0
    total_amp_loss = 0.0
    total_phase_loss = 0.0
    total_time_loss = 0.0
    total_vq_loss = 0.0
    total_samples = 0
    total_perplexity = 0.0
    total_utilization = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Get data
        if isinstance(batch, dict):
            x = batch['data']
        else:
            x = batch[0]
        
        x = x.to(device)
        B, C, T = x.shape
        
        # Process each channel independently
        x_flat = x.view(B * C, T)
        
        # Forward pass
        outputs = tokenizer(x_flat)
        
        # Get losses from model
        rec_loss = outputs['rec_loss']
        vq_loss = outputs['commitment_loss']
        
        # Total loss
        loss = rec_loss + vq_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_clip = config['training'].get('gradient_clip', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * B
        total_rec_loss += rec_loss.item() * B
        total_amp_loss += outputs['amplitude_loss'].item() * B
        total_phase_loss += outputs['phase_loss'].item() * B
        total_time_loss += outputs['time_loss'].item() * B
        total_vq_loss += vq_loss.item() * B
        total_samples += B
        total_perplexity += outputs['perplexity'].item()
        total_utilization += outputs['code_utilization'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / total_samples,
        'rec_loss': total_rec_loss / total_samples,
        'amplitude_loss': total_amp_loss / total_samples,
        'phase_loss': total_phase_loss / total_samples,
        'time_loss': total_time_loss / total_samples,
        'vq_loss': total_vq_loss / total_samples,
        'perplexity': total_perplexity / n_batches,
        'utilization': total_utilization / n_batches,
    }


@torch.no_grad()
def validate(
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate tokenizer."""
    tokenizer.eval()
    
    total_rec_loss = 0.0
    total_amp_loss = 0.0
    total_phase_loss = 0.0
    total_time_loss = 0.0
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
        
        total_rec_loss += outputs['rec_loss'].item() * B
        total_amp_loss += outputs['amplitude_loss'].item() * B
        total_phase_loss += outputs['phase_loss'].item() * B
        total_time_loss += outputs['time_loss'].item() * B
        total_samples += B
        
        all_indices.append(outputs['indices'].cpu())
    
    # Compute code utilization
    all_indices = torch.cat(all_indices, dim=0).flatten()
    unique_codes = torch.unique(all_indices)
    utilization = len(unique_codes) / tokenizer.codebook_size
    
    return {
        'val_rec_loss': total_rec_loss / total_samples,
        'val_amplitude_loss': total_amp_loss / total_samples,
        'val_phase_loss': total_phase_loss / total_samples,
        'val_time_loss': total_time_loss / total_samples,
        'val_utilization': utilization,
        'val_unique_codes': len(unique_codes),
    }


def compute_spectral_metrics(original: torch.Tensor, reconstructed: torch.Tensor, 
                             fs: int = 200) -> dict:
    """Compute spectral comparison metrics."""
    # FFT
    orig_fft = torch.fft.rfft(original, dim=-1)
    rec_fft = torch.fft.rfft(reconstructed, dim=-1)
    
    orig_mag = torch.abs(orig_fft)
    rec_mag = torch.abs(rec_fft)
    
    # Magnitude correlation
    orig_flat = orig_mag.flatten().numpy()
    rec_flat = rec_mag.flatten().numpy()
    correlation = np.corrcoef(orig_flat, rec_flat)[0, 1]
    
    # Log spectral distance
    eps = 1e-8
    log_orig = torch.log(orig_mag + eps)
    log_rec = torch.log(rec_mag + eps)
    lsd = torch.sqrt(torch.mean((log_orig - log_rec) ** 2)).item()
    
    return {
        'spectral_correlation': correlation,
        'log_spectral_distance': lsd,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Frequency-Domain Patch VQ-VAE")
    parser.add_argument('--config', type=str, required=True,
                        help='Config file path (relative to experiments/configs/)')
    args = parser.parse_args()
    
    # Initialize ExperimentLogger (this creates run_dir)
    logger = ExperimentLogger(config_path=args.config)
    config = logger.config
    
    # 设置日志，同时输出到终端和文件
    tee_logger = setup_logging(logger.run_dir)
    
    print(f"\n{'='*60}")
    print(f"Training Frequency-Domain Patch VQ-VAE Tokenizer")
    print(f"{'='*60}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run directory: {logger.run_dir}")
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
    print(f"  FFT size: {patch_size // 2 + 1}")
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
    patience = config['training'].get('early_stopping', {}).get('patience', 25)
    
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
        
        # Log epoch
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['val_rec_loss'],
            loss_breakdown={
                'amplitude': train_metrics['amplitude_loss'],
                'phase': train_metrics['phase_loss'],
                'time': train_metrics['time_loss'],
                'vq_commitment': train_metrics['vq_loss'],
            },
            metrics={
                'perplexity': train_metrics['perplexity'],
                'train_utilization': train_metrics['utilization'],
                'val_utilization': val_metrics['val_utilization'],
                'val_unique_codes': val_metrics['val_unique_codes'],
                'val_amplitude_loss': val_metrics['val_amplitude_loss'],
                'val_phase_loss': val_metrics['val_phase_loss'],
                'val_time_loss': val_metrics['val_time_loss'],
            }
        )
        
        # Print epoch summary
        print(f"Epoch {epoch:3d} | "
              f"Train: {train_metrics['loss']:.4f} (amp={train_metrics['amplitude_loss']:.4f}, "
              f"phase={train_metrics['phase_loss']:.4f}, time={train_metrics['time_loss']:.4f})")
        print(f"         | "
              f"Val: {val_metrics['val_rec_loss']:.4f}, "
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
            print(f"         | ★ New best model saved!")
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
        'fft_size': patch_size // 2 + 1,
        'model_type': config['model'].get('type', 'freq_patch_vqvae'),
    }
    
    logger.log_final(final_metrics)
    
    # =========================================================================
    # Generate Visualizations (with error handling)
    # =========================================================================
    print("\nGenerating visualizations...")
    
    try:
        visualizer = TokenizerVisualizer(logger.run_dir)
        
        # 1. Training curves
        metrics_history = logger.get_metrics_history()
        visualizer.plot_training_curves(metrics_history)
        
        # 2. Get validation samples
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
                
                if sum(o.shape[0] for o in all_originals) >= 200:
                    break
        
        original = torch.cat(all_originals, dim=0)
        reconstructed = torch.cat(all_reconstructed, dim=0)
        indices = torch.cat(all_indices, dim=0)
        
        # 3. Reconstruction samples
        try:
            visualizer.plot_reconstruction_samples(
                original, reconstructed,
                n_samples=4,
                fs=sr
            )
        except Exception as e:
            print(f"  [Warning] Failed to plot reconstruction samples: {e}")
        
        # 4. Spectral comparison
        try:
            visualizer.plot_spectral_comparison(
                original, reconstructed,
                fs=sr,
                n_samples=100
            )
        except Exception as e:
            print(f"  [Warning] Failed to plot spectral comparison: {e}")
        
        # 5. Codebook usage
        try:
            visualizer.plot_codebook_usage(indices, tokenizer.codebook_size)
        except Exception as e:
            print(f"  [Warning] Failed to plot codebook usage: {e}")
        
        # 6. Token embeddings
        try:
            embeddings = tokenizer.get_codebook_embeddings()
            if embeddings is not None:
                flat_indices = indices.flatten()
                usage = torch.bincount(flat_indices, minlength=tokenizer.codebook_size)
                visualizer.plot_token_embeddings(embeddings, usage)
        except Exception as e:
            print(f"  [Warning] Failed to plot token embeddings: {e}")
        
        # 7. Compute spectral metrics
        try:
            spectral_metrics = compute_spectral_metrics(original[:100], reconstructed[:100], fs=sr)
            print(f"\nSpectral Metrics:")
            print(f"  Correlation: {spectral_metrics['spectral_correlation']:.4f}")
            print(f"  Log Spectral Distance: {spectral_metrics['log_spectral_distance']:.4f}")
            final_metrics.update(spectral_metrics)
        except Exception as e:
            print(f"  [Warning] Failed to compute spectral metrics: {e}")
            spectral_metrics = {'spectral_correlation': 0.0, 'log_spectral_distance': 0.0}
        
        logger.log_final(final_metrics)
        
        # 8. Summary figure
        try:
            visualizer.generate_summary_figure(final_metrics, config)
            visualizer.save_figure_manifest()
        except Exception as e:
            print(f"  [Warning] Failed to generate summary figure: {e}")
            
        try:
            logger.generate_figures()
        except Exception as e:
            print(f"  [Warning] Failed to generate logger figures: {e}")
            
    except Exception as e:
        print(f"\n[Error] Visualization failed: {e}")
        traceback.print_exc()
        # 仍然保存final metrics
        logger.log_final(final_metrics)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final utilization: {val_metrics['val_utilization']*100:.1f}%")
    if 'spectral_metrics' in dir() and spectral_metrics:
        print(f"Spectral correlation: {spectral_metrics.get('spectral_correlation', 'N/A')}")
    print(f"Run directory: {logger.run_dir}")
    print(f"{'='*60}")
    
    # 关闭日志
    tee_logger.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
