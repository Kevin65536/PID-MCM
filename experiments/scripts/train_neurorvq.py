"""
Train NeuroRVQ Tokenizers

This script trains the NeuroRVQ-style tokenizers that:
1. Use multi-scale temporal encoder (Inception-style)
2. Apply Residual Vector Quantization (RVQ) with multiple layers
3. Use L2-normalized EMA codebook updates
4. Reconstruct in frequency domain (amplitude + sin/cos phase)

Usage:
    python train_neurorvq.py --config phase0plus/eeg_neurorvq.yaml
    python train_neurorvq.py --config phase0plus/fnirs_neurorvq.yaml
    
后台运行 (推荐):
    nohup python train_neurorvq.py --config phase0plus/eeg_neurorvq.yaml &
"""

import sys
import os
import argparse
import traceback
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
from src.tokenizers.neurorvq import NeuroRVQTokenizer, NeuroRVQTokenizer_V2
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
    """Create NeuroRVQ tokenizer from config."""
    model_cfg = config['model']
    patch_cfg = model_cfg.get('patch', {})
    encoder_cfg = model_cfg.get('encoder', {})
    quantizer_cfg = model_cfg.get('quantizer', {})
    decoder_cfg = model_cfg.get('decoder', {})
    loss_cfg = config.get('loss', {})
    
    # Determine model type
    model_type = model_cfg.get('type', 'neurorvq')
    
    # Common kwargs
    kwargs = dict(
        patch_size=patch_cfg.get('size', 200),
        code_dim=quantizer_cfg.get('code_dim', 64),
        num_codes=quantizer_cfg.get('num_codes', 8192),
        num_quantizers=quantizer_cfg.get('num_quantizers', 8),
        hidden_channels=encoder_cfg.get('hidden_channels', 8),
        hidden_dim=decoder_cfg.get('hidden_dim', 256),
        beta=quantizer_cfg.get('beta', 1.0),
        decay=quantizer_cfg.get('decay', 0.99),
        kmeans_init=quantizer_cfg.get('kmeans_init', True),
        amplitude_weight=loss_cfg.get('amplitude', {}).get('weight', 1.0),
        phase_weight=loss_cfg.get('phase', {}).get('weight', 1.0),
        time_weight=loss_cfg.get('time', {}).get('weight', 1.0),
        vq_weight=loss_cfg.get('vq', {}).get('weight', 1.0),
    )
    
    if model_type == 'neurorvq_v2':
        return NeuroRVQTokenizer_V2(**kwargs)
    else:
        return NeuroRVQTokenizer(**kwargs)


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
    total_amp_loss = 0.0
    total_phase_loss = 0.0
    total_time_loss = 0.0
    total_vq_loss = 0.0
    total_samples = 0
    total_phase_cos_sim = 0.0
    n_batches = 0
    
    # Track RVQ layer utilization
    all_usage_ratios = []
    
    for batch in dataloader:
        # Get data
        if isinstance(batch, dict):
            x = batch['data']
        else:
            x = batch[0]
        
        x = x.to(device)
        B, C, T = x.shape
        
        # Get patch size
        patch_size = tokenizer.patch_size
        
        # Process patches from each channel
        patches = []
        for c in range(C):
            for p in range(0, T - patch_size + 1, patch_size):
                patches.append(x[:, c, p:p+patch_size])  # [B, patch_size]
        
        # Stack all patches
        if patches:
            x_patches = torch.stack(patches, dim=1).view(-1, patch_size)  # [B*C*n_patches, patch_size]
        else:
            continue
        
        # Forward pass
        outputs = tokenizer(x_patches)
        
        # Get loss
        loss = outputs['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_clip = config['training'].get('gradient', {}).get('clip_norm', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * B
        total_amp_loss += outputs['amp_loss'].item() * B
        total_phase_loss += outputs['phase_loss'].item() * B
        total_time_loss += outputs['time_loss'].item() * B
        total_vq_loss += outputs['vq_loss'].item() * B
        total_phase_cos_sim += outputs['phase_cos_sim'].item()
        total_samples += B
        n_batches += 1
        
        # Track usage ratios
        if 'usage_ratios' in outputs:
            all_usage_ratios.append(outputs['usage_ratios'])
    
    # Compute average utilization per RVQ layer
    avg_utilization = 0.0
    if all_usage_ratios:
        # Usage ratios shape: [num_quantizers]
        avg_per_layer = np.mean(all_usage_ratios, axis=0)
        avg_utilization = np.mean(avg_per_layer)
    
    return {
        'loss': total_loss / total_samples,
        'amp_loss': total_amp_loss / total_samples,
        'phase_loss': total_phase_loss / total_samples,
        'time_loss': total_time_loss / total_samples,
        'vq_loss': total_vq_loss / total_samples,
        'phase_cos_sim': total_phase_cos_sim / n_batches,
        'utilization': avg_utilization,
    }


@torch.no_grad()
def validate(
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate tokenizer."""
    tokenizer.eval()
    
    total_loss = 0.0
    total_amp_loss = 0.0
    total_phase_loss = 0.0
    total_time_loss = 0.0
    total_vq_loss = 0.0
    total_samples = 0
    total_phase_cos_sim = 0.0
    n_batches = 0
    
    # Track all tokens for utilization
    all_tokens = []
    
    for batch in dataloader:
        if isinstance(batch, dict):
            x = batch['data']
        else:
            x = batch[0]
        
        x = x.to(device)
        B, C, T = x.shape
        
        patch_size = tokenizer.patch_size
        
        # Extract patches
        patches = []
        for c in range(C):
            for p in range(0, T - patch_size + 1, patch_size):
                patches.append(x[:, c, p:p+patch_size])
        
        if not patches:
            continue
            
        x_patches = torch.stack(patches, dim=1).view(-1, patch_size)
        
        outputs = tokenizer(x_patches)
        
        total_loss += outputs['loss'].item() * B
        total_amp_loss += outputs['amp_loss'].item() * B
        total_phase_loss += outputs['phase_loss'].item() * B
        total_time_loss += outputs['time_loss'].item() * B
        total_vq_loss += outputs['vq_loss'].item() * B
        total_phase_cos_sim += outputs['phase_cos_sim'].item()
        total_samples += B
        n_batches += 1
        
        # Collect tokens
        if 'tokens' in outputs:
            all_tokens.append(outputs['tokens'].cpu())
    
    # Compute code utilization
    utilization = 0.0
    unique_codes = 0
    if all_tokens:
        # RVQ tokens: [num_quantizers, B] per batch
        stacked = torch.cat(all_tokens, dim=1)  # [num_quantizers, total_samples]
        # Count unique across all layers
        unique_codes = len(torch.unique(stacked))
        utilization = unique_codes / (tokenizer.num_codes * tokenizer.num_quantizers)
    
    return {
        'val_loss': total_loss / total_samples,
        'val_amp_loss': total_amp_loss / total_samples,
        'val_phase_loss': total_phase_loss / total_samples,
        'val_time_loss': total_time_loss / total_samples,
        'val_vq_loss': total_vq_loss / total_samples,
        'val_phase_cos_sim': total_phase_cos_sim / n_batches,
        'val_utilization': utilization,
        'val_unique_codes': unique_codes,
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
    orig_flat = orig_mag.flatten().cpu().numpy()
    rec_flat = rec_mag.flatten().cpu().numpy()
    correlation = np.corrcoef(orig_flat, rec_flat)[0, 1]
    
    # Log spectral distance
    eps = 1e-8
    log_orig = torch.log(orig_mag + eps)
    log_rec = torch.log(rec_mag + eps)
    lsd = torch.sqrt(torch.mean((log_orig - log_rec) ** 2)).item()
    
    # Time domain correlation
    orig_flat_t = original.flatten().cpu().numpy()
    rec_flat_t = reconstructed.flatten().cpu().numpy()
    time_corr = np.corrcoef(orig_flat_t, rec_flat_t)[0, 1]
    
    return {
        'spectral_correlation': correlation,
        'log_spectral_distance': lsd,
        'time_correlation': time_corr,
    }


def main():
    parser = argparse.ArgumentParser(description="Train NeuroRVQ Tokenizer")
    parser.add_argument('--config', type=str, required=True,
                        help='Config file path (relative to experiments/configs/)')
    args = parser.parse_args()
    
    # Initialize ExperimentLogger (this creates run_dir)
    logger = ExperimentLogger(config_path=args.config)
    config = logger.config
    
    # 设置日志，同时输出到终端和文件
    tee_logger = setup_logging(logger.run_dir)
    
    print(f"\n{'='*60}")
    print(f"Training NeuroRVQ Tokenizer")
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
    print("\nCreating NeuroRVQ tokenizer...")
    tokenizer = create_tokenizer(config).to(device)
    
    # Model info
    patch_cfg = config['model'].get('patch', {})
    quantizer_cfg = config['model'].get('quantizer', {})
    patch_size = patch_cfg.get('size', 200)
    seq_length = config['model']['seq_length']
    n_patches = seq_length // patch_size
    sr = config['data']['preprocessing'].get('resample_rate', 200)
    patch_duration = patch_size / sr
    
    print(f"\n{'='*40}")
    print("NeuroRVQ Token Specification:")
    print(f"  Patch size: {patch_size} samples = {patch_duration:.2f}s @ {sr}Hz")
    print(f"  Patches per window: {n_patches}")
    print(f"  Codebook size: {quantizer_cfg.get('num_codes', 8192)}")
    print(f"  Code dimension: {quantizer_cfg.get('code_dim', 64)}")
    print(f"  RVQ layers: {quantizer_cfg.get('num_quantizers', 8)}")
    print(f"  Total tokens per patch: {quantizer_cfg.get('num_quantizers', 8)} (one per RVQ layer)")
    print(f"{'='*40}")
    
    # Count parameters
    total_params = sum(p.numel() for p in tokenizer.parameters())
    trainable_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    train_cfg = config['training']
    opt_cfg = train_cfg['optimizer']
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg.get('weight_decay', 0.01),
        betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
    )
    
    # Scheduler
    sched_cfg = train_cfg.get('scheduler', {})
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['epochs'] - sched_cfg.get('warmup_epochs', 5),
        eta_min=sched_cfg.get('min_lr', 1e-6),
    )
    
    # Early stopping
    es_cfg = train_cfg.get('early_stopping', {})
    patience = es_cfg.get('patience', 20)
    min_delta = es_cfg.get('min_delta', 0.0001)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Training loop
    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {opt_cfg['lr']}")
    print(f"  Early stopping patience: {patience}")
    
    for epoch in range(train_cfg['epochs']):
        epoch_start = datetime.now()
        
        # Train
        train_metrics = train_epoch(tokenizer, train_loader, optimizer, device, config)
        
        # Validate
        val_metrics = validate(tokenizer, val_loader, device)
        
        # Step scheduler
        if epoch >= sched_cfg.get('warmup_epochs', 5):
            scheduler.step()
        
        # Current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"\nEpoch {epoch+1}/{train_cfg['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train: Loss={train_metrics['loss']:.4f} "
              f"Amp={train_metrics['amp_loss']:.4f} Phase={train_metrics['phase_loss']:.4f} "
              f"Time={train_metrics['time_loss']:.4f} VQ={train_metrics['vq_loss']:.4f}")
        print(f"  Val:   Loss={val_metrics['val_loss']:.4f} "
              f"Amp={val_metrics['val_amp_loss']:.4f} Phase={val_metrics['val_phase_loss']:.4f} "
              f"Time={val_metrics['val_time_loss']:.4f}")
        print(f"  Util: {train_metrics['utilization']*100:.1f}% | "
              f"Phase cos sim: {train_metrics['phase_cos_sim']:.4f} | LR: {current_lr:.2e}")
        
        # Log to experiment logger (using correct signature)
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['val_loss'],
            loss_breakdown={
                'amp_loss': train_metrics['amp_loss'],
                'phase_loss': train_metrics['phase_loss'],
                'time_loss': train_metrics['time_loss'],
                'vq_loss': train_metrics['vq_loss'],
            },
            metrics={
                'utilization': train_metrics['utilization'],
                'phase_cos_sim': train_metrics['phase_cos_sim'],
                'lr': current_lr,
                'val_utilization': val_metrics['val_utilization'],
            }
        )
        
        # Check for improvement
        val_loss = val_metrics['val_loss']
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model
            best_path = logger.run_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': tokenizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, best_path)
            print(f"  ★ New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Periodic checkpoint
        ckpt_cfg = train_cfg.get('checkpoint', {})
        if ckpt_cfg.get('save_every', 0) > 0 and (epoch + 1) % ckpt_cfg['save_every'] == 0:
            ckpt_path = logger.run_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': tokenizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, ckpt_path)
    
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
    best_path = logger.run_dir / 'best_model.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    tokenizer.eval()
    
    # Compute final metrics on validation set
    final_val = validate(tokenizer, val_loader, device)
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {final_val['val_loss']:.4f}")
    print(f"  Amplitude Loss: {final_val['val_amp_loss']:.4f}")
    print(f"  Phase Loss: {final_val['val_phase_loss']:.4f}")
    print(f"  Time Loss: {final_val['val_time_loss']:.4f}")
    print(f"  VQ Loss: {final_val['val_vq_loss']:.4f}")
    print(f"  Codebook Utilization: {final_val['val_utilization']*100:.1f}%")
    print(f"  Unique Codes Used: {final_val['val_unique_codes']}")
    
    # Compute spectral metrics on a sample
    print("\nComputing spectral metrics on sample...")
    try:
        sample_batch = next(iter(val_loader))
        if isinstance(sample_batch, dict):
            sample = sample_batch['data']
        else:
            sample = sample_batch[0]
        
        sample = sample.to(device)
        B, C, T = sample.shape
        patch_size = tokenizer.patch_size
        
        # Extract first patch from first channel
        x_patch = sample[0, 0, :patch_size].unsqueeze(0)  # [1, patch_size]
        
        # Tokenize and detokenize
        tokens = tokenizer.tokenize(x_patch)
        reconstructed = tokenizer.detokenize(tokens)
        
        # Compute metrics
        spectral_metrics = compute_spectral_metrics(x_patch, reconstructed)
        print(f"\nReconstruction Quality (sample):")
        print(f"  Spectral Correlation: {spectral_metrics['spectral_correlation']:.4f}")
        print(f"  Log Spectral Distance: {spectral_metrics['log_spectral_distance']:.4f}")
        print(f"  Time Domain Correlation: {spectral_metrics['time_correlation']:.4f}")
        
        # Add to final metrics
        final_val.update({f'spectral_{k}': v for k, v in spectral_metrics.items()})
    except Exception as e:
        print(f"Warning: Could not compute spectral metrics: {e}")
        traceback.print_exc()
    
    # Get per-layer codebook usage
    print("\nPer-layer Codebook Usage:")
    try:
        usage = tokenizer.get_codebook_usage()
        for i in range(tokenizer.num_quantizers):
            layer_key = f'layer_{i}_utilization'
            if layer_key in usage:
                print(f"  Layer {i}: {usage[layer_key]*100:.1f}% active codes")
    except Exception as e:
        print(f"Warning: Could not get codebook usage: {e}")
    
    # Visualization
    print("\nGenerating visualizations...")
    try:
        visualizer = TokenizerVisualizer(save_dir=str(logger.run_dir / 'visualizations'))
        
        # Get sample for visualization
        sample_batch = next(iter(val_loader))
        if isinstance(sample_batch, dict):
            sample = sample_batch['data']
            labels = sample_batch.get('label', None)
        else:
            sample = sample_batch[0]
            labels = sample_batch[1] if len(sample_batch) > 1 else None
        
        sample = sample.to(device)
        B, C, T = sample.shape
        patch_size = tokenizer.patch_size
        
        # Extract patches for visualization
        x_patch = sample[0, 0, :patch_size].unsqueeze(0)
        tokens = tokenizer.tokenize(x_patch)
        reconstructed = tokenizer.detokenize(tokens)
        
        # Reconstruction plot
        visualizer.plot_reconstruction(
            original=x_patch.cpu().numpy(),
            reconstructed=reconstructed.cpu().numpy(),
            title=f"NeuroRVQ Reconstruction ({config['data']['modality'].upper()})",
        )
        
        print(f"Visualizations saved to: {logger.run_dir / 'visualizations'}")
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
        traceback.print_exc()
    
    # Log final metrics
    try:
        final_metrics = {
            'model_type': 'neurorvq',
            'best_val_loss': best_val_loss,
            **final_val,
        }
        logger.log_final(final_metrics)
    except Exception as e:
        print(f"Warning: Could not log final metrics: {e}")
    
    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"Results saved to: {logger.run_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Close log file
    tee_logger.close()


if __name__ == '__main__':
    main()
