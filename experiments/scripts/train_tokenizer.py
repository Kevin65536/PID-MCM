"""
Unified Tokenizer Training Script

This script provides a standardized training pipeline for all tokenizers.
Tokenizers are created via the registry system based on config file.

Usage:
    # EEG tokenizers
    python train_tokenizer.py --config phase0plus/eeg_patch_vqvae_1s_v3.yaml
    python train_tokenizer.py --config phase0plus/eeg_freq_patch_vqvae_1s.yaml
    python train_tokenizer.py --config phase0plus/eeg_neurorvq.yaml
    
    # fNIRS tokenizers
    python train_tokenizer.py --config phase0plus/fnirs_patch_vqvae_2s_v2.yaml
    python train_tokenizer.py --config phase0plus/fnirs_freq_patch_vqvae_1s.yaml
    python train_tokenizer.py --config phase0plus/fnirs_neurorvq.yaml
    
后台运行:
    nohup python train_tokenizer.py --config phase0plus/eeg_neurorvq.yaml &
"""

import sys
import os
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import ExperimentLogger
from src.tokenizers import create_tokenizer, StandardizedOutput, list_tokenizers
from src.data.eeg_fnirs_dataset import EEGfNIRSDataset
from src.visualization import TokenizerVisualizer


# ============================================================================
# Logging Utilities
# ============================================================================

class TeeLogger:
    """同时输出到终端和文件的日志类"""
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a', buffering=1)
        
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


# ============================================================================
# Data Loading
# ============================================================================

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


# ============================================================================
# Training and Validation
# ============================================================================

def get_patch_size(tokenizer, config: dict) -> int:
    """Get patch size from tokenizer or config."""
    if hasattr(tokenizer, 'patch_size'):
        return tokenizer.patch_size
    
    patch_cfg = config.get('model', {}).get('patch', {})
    return patch_cfg.get('size', 200)


def prepare_input(x: torch.Tensor, patch_size: int, tokenizer_type: str) -> torch.Tensor:
    """
    Prepare input data for the tokenizer.
    
    Different tokenizers expect different input formats:
    - PatchVQVAE / FreqPatchVQVAE: expect [B, T] (full sequence)
    - NeuroRVQ: expects [B, patch_size] (individual patches)
    
    Args:
        x: [B, C, T] input data
        patch_size: size of each patch
        tokenizer_type: type of tokenizer being used
    
    Returns:
        Prepared input tensor
    """
    B, C, T = x.shape
    
    if tokenizer_type.startswith('neurorvq'):
        # NeuroRVQ expects individual patches [B_total, patch_size]
        patches = []
        for c in range(C):
            for p in range(0, T - patch_size + 1, patch_size):
                patches.append(x[:, c, p:p+patch_size])
        
        if not patches:
            raise ValueError(f"Cannot extract patches: T={T}, patch_size={patch_size}")
        
        # Stack: [n_patches_per_sample, B, patch_size] -> [B * n_patches, patch_size]
        x_patches = torch.stack(patches, dim=1).view(-1, patch_size)
        return x_patches
    else:
        # Other tokenizers expect [B, T] per channel
        return x.view(B * C, T)


def train_epoch(
    tokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> Dict[str, float]:
    """Train for one epoch."""
    tokenizer.train()
    
    tokenizer_type = config['model'].get('type', 'patch_vqvae')
    patch_size = get_patch_size(tokenizer, config)
    grad_clip = config['training'].get('gradient', {}).get('clip_norm', 
                config['training'].get('gradient_clip', 1.0))
    
    # Accumulators
    total_loss = 0.0
    total_samples = 0
    n_batches = 0
    loss_accum = {}
    util_accum = 0.0
    
    for batch in dataloader:
        # Get data
        if isinstance(batch, dict):
            x = batch['data']
        else:
            x = batch[0]
        
        x = x.to(device)
        B = x.shape[0]
        
        # Prepare input
        x_input = prepare_input(x, patch_size, tokenizer_type)
        
        # Forward pass
        outputs = tokenizer(x_input)
        std_outputs = StandardizedOutput.standardize(outputs)
        
        # Get loss from tokenizer output
        # All tokenizers should return 'loss' for unified interface
        loss = std_outputs.get('loss')
        
        if loss is None:
            raise ValueError(
                f"Tokenizer '{tokenizer_type}' did not return a 'loss' value. "
                f"All tokenizers must return 'loss' for unified training. "
                f"Available keys: {list(outputs.keys())}"
            )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * B
        total_samples += B
        n_batches += 1
        
        # Accumulate loss breakdown
        breakdown = StandardizedOutput.get_loss_breakdown(outputs)
        for k, v in breakdown.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v * B
        
        # Utilization
        util_accum += StandardizedOutput.get_utilization(outputs)
    
    # Compute averages
    metrics = {
        'loss': total_loss / total_samples,
    }
    for k, v in loss_accum.items():
        metrics[k] = v / total_samples
    metrics['utilization'] = util_accum / n_batches if n_batches > 0 else 0.0
    
    return metrics


@torch.no_grad()
def validate(
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
) -> Dict[str, float]:
    """Validate tokenizer."""
    tokenizer.eval()
    
    tokenizer_type = config['model'].get('type', 'patch_vqvae')
    patch_size = get_patch_size(tokenizer, config)
    
    total_loss = 0.0
    total_samples = 0
    n_batches = 0
    loss_accum = {}
    util_accum = 0.0
    
    for batch in dataloader:
        if isinstance(batch, dict):
            x = batch['data']
        else:
            x = batch[0]
        
        x = x.to(device)
        B = x.shape[0]
        
        x_input = prepare_input(x, patch_size, tokenizer_type)
        outputs = tokenizer(x_input)
        std_outputs = StandardizedOutput.standardize(outputs)
        
        # Get loss from tokenizer output
        loss = std_outputs.get('loss')
        if loss is not None:
            total_loss += loss.item() * B
        
        total_samples += B
        n_batches += 1
        
        breakdown = StandardizedOutput.get_loss_breakdown(outputs)
        for k, v in breakdown.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v * B
        
        util_accum += StandardizedOutput.get_utilization(outputs)
    
    # Compute averages with 'val_' prefix
    metrics = {
        'val_loss': total_loss / total_samples if total_samples > 0 else 0.0,
    }
    for k, v in loss_accum.items():
        metrics[f'val_{k}'] = v / total_samples
    metrics['val_utilization'] = util_accum / n_batches if n_batches > 0 else 0.0
    
    return metrics


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    tokenizer,
    optimizer,
    epoch: int,
    val_loss: float,
    config: dict,
    save_path: Path,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': tokenizer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config,
        'tokenizer_type': config['model'].get('type', 'unknown'),
    }
    torch.save(checkpoint, save_path)
    
    if is_best:
        print(f"  ★ New best model saved (val_loss={val_loss:.4f})")


def load_checkpoint(
    checkpoint_path: Path,
    tokenizer,
    optimizer=None,
    device='cpu',
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


# ============================================================================
# Spectral Metrics
# ============================================================================

def compute_spectral_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """Compute spectral comparison metrics."""
    try:
        orig_fft = torch.fft.rfft(original, dim=-1)
        rec_fft = torch.fft.rfft(reconstructed, dim=-1)
        
        orig_mag = torch.abs(orig_fft)
        rec_mag = torch.abs(rec_fft)
        
        # Magnitude correlation
        orig_flat = orig_mag.flatten().cpu().numpy()
        rec_flat = rec_mag.flatten().cpu().numpy()
        
        if len(orig_flat) > 1:
            correlation = np.corrcoef(orig_flat, rec_flat)[0, 1]
        else:
            correlation = 0.0
        
        # Log spectral distance
        eps = 1e-8
        log_orig = torch.log(orig_mag + eps)
        log_rec = torch.log(rec_mag + eps)
        lsd = torch.sqrt(torch.mean((log_orig - log_rec) ** 2)).item()
        
        # Time domain correlation
        orig_flat_t = original.flatten().cpu().numpy()
        rec_flat_t = reconstructed.flatten().cpu().numpy()
        if len(orig_flat_t) > 1:
            time_corr = np.corrcoef(orig_flat_t, rec_flat_t)[0, 1]
        else:
            time_corr = 0.0
        
        return {
            'spectral_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'log_spectral_distance': lsd,
            'time_correlation': float(time_corr) if not np.isnan(time_corr) else 0.0,
        }
    except Exception as e:
        print(f"Warning: Spectral metrics computation failed: {e}")
        return {
            'spectral_correlation': 0.0,
            'log_spectral_distance': 0.0,
            'time_correlation': 0.0,
        }


# ============================================================================
# Info Display
# ============================================================================

def print_tokenizer_info(tokenizer, config: dict):
    """Print tokenizer specification info."""
    model_cfg = config.get('model', {})
    patch_cfg = model_cfg.get('patch', {})
    quantizer_cfg = model_cfg.get('quantizer', {})
    
    tokenizer_type = model_cfg.get('type', 'unknown')
    patch_size = get_patch_size(tokenizer, config)
    seq_length = model_cfg.get('seq_length', 800)
    sr = config['data']['preprocessing'].get('resample_rate', 200)
    
    print(f"\n{'='*50}")
    print(f"Tokenizer: {tokenizer_type}")
    print(f"{'='*50}")
    print(f"  Patch size: {patch_size} samples = {patch_size/sr:.2f}s @ {sr}Hz")
    print(f"  Sequence length: {seq_length} samples = {seq_length/sr:.2f}s")
    print(f"  Patches per sequence: {seq_length // patch_size}")
    
    if 'codebook_size' in quantizer_cfg:
        print(f"  Codebook size: {quantizer_cfg['codebook_size']}")
    elif 'num_codes' in quantizer_cfg:
        print(f"  Codebook size: {quantizer_cfg['num_codes']}")
    
    if 'num_quantizers' in quantizer_cfg:
        print(f"  RVQ layers: {quantizer_cfg['num_quantizers']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in tokenizer.parameters())
    trainable_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"{'='*50}")


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Tokenizer Training Script")
    parser.add_argument('--config', type=str, required=True,
                        help='Config file path (relative to experiments/configs/)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Print available tokenizers
    print(f"Available tokenizers: {list_tokenizers()}")
    
    # Initialize ExperimentLogger
    logger = ExperimentLogger(config_path=args.config)
    config = logger.config
    
    # Setup logging
    tee_logger = setup_logging(logger.run_dir)
    
    print(f"\n{'='*60}")
    print(f"Unified Tokenizer Training")
    print(f"{'='*60}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run directory: {logger.run_dir}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment'].get('description', 'N/A')}")
    print(f"Modality: {config['data']['modality']}")
    print(f"Tokenizer type: {config['model'].get('type', 'patch_vqvae')}")
    
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
    
    # Create tokenizer via registry
    print("\nCreating tokenizer...")
    tokenizer = create_tokenizer(config).to(device)
    print_tokenizer_info(tokenizer, config)
    
    # Optimizer
    train_cfg = config['training']
    opt_cfg = train_cfg.get('optimizer', {})
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=opt_cfg.get('lr', train_cfg.get('learning_rate', 1e-3)),
        weight_decay=opt_cfg.get('weight_decay', train_cfg.get('weight_decay', 0.01)),
        betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
    )
    
    # Scheduler
    sched_cfg = train_cfg.get('scheduler', {})
    # Handle both dict and string format for scheduler config
    if isinstance(sched_cfg, str):
        sched_cfg = {'type': sched_cfg}
    warmup_epochs = sched_cfg.get('warmup_epochs', train_cfg.get('warmup_epochs', 5))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['epochs'] - warmup_epochs,
        eta_min=sched_cfg.get('min_lr', 1e-6),
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(Path(args.resume), tokenizer, optimizer, device)
        start_epoch = ckpt['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # Early stopping
    es_cfg = train_cfg.get('early_stopping', {})
    patience = es_cfg.get('patience', 20)
    min_delta = es_cfg.get('min_delta', 0.0001)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Checkpoint config
    ckpt_cfg = train_cfg.get('checkpoint', {})
    save_every = ckpt_cfg.get('save_every', 10)
    
    # Checkpoint directory (correct location!)
    checkpoints_dir = logger.checkpoints_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Get learning rate for display
    actual_lr = opt_cfg.get('lr', train_cfg.get('learning_rate', 1e-3))
    
    # Training loop
    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {actual_lr}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Checkpoints saved to: {checkpoints_dir}")
    
    for epoch in range(start_epoch, train_cfg['epochs']):
        epoch_start = datetime.now()
        
        # Train
        train_metrics = train_epoch(tokenizer, train_loader, optimizer, device, config)
        
        # Validate
        val_metrics = validate(tokenizer, val_loader, device, config)
        
        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        # Format metrics for printing
        train_str = f"Loss={train_metrics['loss']:.4f}"
        val_str = f"Loss={val_metrics['val_loss']:.4f}"
        
        for key in ['amp_loss', 'phase_loss', 'time_loss', 'vq_loss', 'rec_loss']:
            if key in train_metrics:
                train_str += f" {key.replace('_loss', '').title()}={train_metrics[key]:.4f}"
            val_key = f'val_{key}'
            if val_key in val_metrics:
                val_str += f" {key.replace('_loss', '').title()}={val_metrics[val_key]:.4f}"
        
        print(f"\nEpoch {epoch+1}/{train_cfg['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train: {train_str}")
        print(f"  Val:   {val_str}")
        print(f"  Util: {train_metrics.get('utilization', 0)*100:.1f}% | LR: {current_lr:.2e}")
        
        # Log to experiment logger
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['val_loss'],
            loss_breakdown={k: v for k, v in train_metrics.items() if k != 'loss'},
            metrics={
                'lr': current_lr,
                'val_utilization': val_metrics.get('val_utilization', 0),
            }
        )
        
        # Check for improvement
        val_loss = val_metrics['val_loss']
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model (in checkpoints directory)
            best_path = checkpoints_dir / 'best_model.pt'
            save_checkpoint(tokenizer, optimizer, epoch + 1, val_loss, config, best_path, is_best=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Periodic checkpoint
        if save_every > 0 and (epoch + 1) % save_every == 0:
            ckpt_path = checkpoints_dir / f'checkpoint_epoch_{epoch+1}.pt'
            save_checkpoint(tokenizer, optimizer, epoch + 1, val_loss, config, ckpt_path)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # ========================
    # Post-training Analysis
    # ========================
    print(f"\n{'='*60}")
    print("Post-training Analysis")
    print(f"{'='*60}")
    
    # Load best model
    best_path = checkpoints_dir / 'best_model.pt'
    if best_path.exists():
        checkpoint = load_checkpoint(best_path, tokenizer, device=device)
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    tokenizer.eval()
    
    # Final validation
    final_val = validate(tokenizer, val_loader, device, config)
    print(f"\nFinal Validation Metrics:")
    for k, v in sorted(final_val.items()):
        print(f"  {k}: {v:.4f}")
    
    # Spectral metrics on sample
    print("\nComputing spectral metrics on sample...")
    try:
        sample_batch = next(iter(val_loader))
        if isinstance(sample_batch, dict):
            sample = sample_batch['data']
        else:
            sample = sample_batch[0]
        
        sample = sample.to(device)
        patch_size = get_patch_size(tokenizer, config)
        tokenizer_type = config['model'].get('type', 'patch_vqvae')
        
        # Get a single sample
        x_sample = sample[0:1]  # [1, C, T]
        x_input = prepare_input(x_sample, patch_size, tokenizer_type)
        
        outputs = tokenizer(x_input)
        std_out = StandardizedOutput.standardize(outputs)
        
        if 'reconstructed' in std_out:
            reconstructed = std_out['reconstructed']
            # Match shapes for comparison
            if reconstructed.shape == x_input.shape:
                spectral_metrics = compute_spectral_metrics(x_input, reconstructed)
                print(f"\nReconstruction Quality (sample):")
                for k, v in spectral_metrics.items():
                    print(f"  {k}: {v:.4f}")
                final_val.update({f'spectral_{k}': v for k, v in spectral_metrics.items()})
    except Exception as e:
        print(f"Warning: Spectral analysis failed: {e}")
        traceback.print_exc()
    
    # ========================
    # Generate Visualizations
    # ========================
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")
    
    try:
        # Initialize TokenizerVisualizer
        visualizer = TokenizerVisualizer(logger.run_dir)
        
        # 1. Training curves from metrics history
        metrics_history = logger.get_metrics_history()
        visualizer.plot_training_curves(metrics_history)
        
        # 2. Get validation samples for visualization
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
                x_input = prepare_input(x, patch_size, tokenizer_type)
                
                outputs = tokenizer(x_input)
                std_out = StandardizedOutput.standardize(outputs)
                
                all_originals.append(x_input.cpu())
                if 'reconstructed' in std_out:
                    all_reconstructed.append(std_out['reconstructed'].cpu())
                if 'tokens' in std_out:
                    all_indices.append(std_out['tokens'].cpu())
                
                # Limit to ~200 samples
                if sum(o.shape[0] for o in all_originals) >= 200:
                    break
        
        original = torch.cat(all_originals, dim=0)
        
        # Get sampling rate
        sr = config['data']['preprocessing'].get('resample_rate', 200)
        
        # 3. Reconstruction samples
        if all_reconstructed:
            reconstructed = torch.cat(all_reconstructed, dim=0)
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
        if all_indices:
            indices = torch.cat(all_indices, dim=0)
            codebook_size = config['model'].get('quantizer', {}).get('codebook_size', 
                           config['model'].get('quantizer', {}).get('num_codes', 2048))
            visualizer.plot_codebook_usage(indices, codebook_size)
            
            # 6. Token embeddings (if available)
            if hasattr(tokenizer, 'get_codebook_embeddings'):
                embeddings = tokenizer.get_codebook_embeddings()
                if embeddings is not None:
                    flat_indices = indices.flatten()
                    usage = torch.bincount(flat_indices.long(), minlength=codebook_size)
                    visualizer.plot_token_embeddings(embeddings, usage)
        
        # 7. Summary figure
        visualizer.generate_summary_figure(final_val, config)
        
        # Save figure manifest
        visualizer.save_figure_manifest()
        
        print(f"  Generated {len(visualizer.get_generated_figures())} figures")
        
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
        traceback.print_exc()
    
    # Log final metrics
    try:
        final_metrics = {
            'model_type': config['model'].get('type', 'unknown'),
            'best_val_loss': best_val_loss,
            **final_val,
        }
        logger.log_final(final_metrics)
    except Exception as e:
        print(f"Warning: Could not log final metrics: {e}")
    
    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"Results saved to: {logger.run_dir}")
    print(f"Best model: {checkpoints_dir / 'best_model.pt'}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    tee_logger.close()


if __name__ == '__main__':
    main()
