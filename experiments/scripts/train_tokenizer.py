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

TensorBoard:
    tensorboard --logdir experiments/runs
"""

import sys
import os
import argparse
import json
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

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
from src.metrics.reconstruction import (
    compute_band_power_loss,
    compute_mae,
    compute_multi_stft_loss,
    compute_reconstruction_mse,
    compute_smoothness_loss,
    compute_snr,
    compute_spectral_mse,
)
from src.metrics.codebook_health import compute_codebook_health
from src.visualization import TokenizerVisualizer, TensorBoardLogger


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
# GPU Selection Utilities
# ============================================================================

def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get GPU information using nvidia-smi.
    
    Returns:
        List of dicts with keys: 
        - index: GPU index
        - name: GPU name
        - memory_used: Memory used in MB
        - memory_total: Total memory in MB
        - memory_free: Free memory in MB
        - utilization: GPU utilization percentage
        - processes: Number of running processes
    """
    try:
        # Query GPU info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            return []
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': float(parts[2]),
                    'memory_total': float(parts[3]),
                    'memory_free': float(parts[4]),
                    'utilization': float(parts[5]) if parts[5] != '[N/A]' else 0.0,
                    'processes': 0,  # Will be updated below
                })
        
        # Query process count per GPU
        proc_result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        
        if proc_result.returncode == 0 and proc_result.stdout.strip():
            # Count processes per GPU by getting UUID mapping
            uuid_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            
            if uuid_result.returncode == 0:
                uuid_to_idx = {}
                for line in uuid_result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 2:
                            uuid_to_idx[parts[1]] = int(parts[0])
                
                # Count processes
                process_counts = {gpu['index']: 0 for gpu in gpus}
                for line in proc_result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 1:
                            uuid = parts[0]
                            if uuid in uuid_to_idx:
                                process_counts[uuid_to_idx[uuid]] += 1
                
                for gpu in gpus:
                    gpu['processes'] = process_counts.get(gpu['index'], 0)
        
        return gpus
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Warning: Failed to query GPU info: {e}")
        return []


def select_best_gpu(verbose: bool = True) -> Optional[int]:
    """
    Select the best available GPU based on utilization, memory, and running processes.
    
    Selection criteria (in order of priority):
    1. Fewer running processes (idle GPUs preferred)
    2. Lower GPU utilization
    3. More free memory
    
    Args:
        verbose: Whether to print GPU selection info
        
    Returns:
        Index of the best GPU, or None if no GPUs available
    """
    gpus = get_gpu_info()
    
    if not gpus:
        if verbose:
            print("No GPUs found via nvidia-smi, falling back to default")
        return None
    
    if verbose:
        print("\n" + "=" * 70)
        print("GPU Status:")
        print("-" * 70)
        print(f"{'GPU':<6} {'Name':<25} {'Util%':<8} {'Memory':<20} {'Procs':<6}")
        print("-" * 70)
        
        for gpu in gpus:
            mem_str = f"{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB"
            print(f"{gpu['index']:<6} {gpu['name'][:24]:<25} {gpu['utilization']:<8.1f} {mem_str:<20} {gpu['processes']:<6}")
        
        print("-" * 70)
    
    # Score each GPU (lower is better)
    # Scoring formula: 
    #   - 1000 * num_processes (heavily penalize GPUs with running processes)
    #   - 10 * utilization (penalize high utilization)
    #   - memory_used / memory_total * 100 (penalize high memory usage)
    def gpu_score(gpu: Dict) -> float:
        return (
            1000 * gpu['processes'] + 
            10 * gpu['utilization'] + 
            (gpu['memory_used'] / gpu['memory_total'] * 100 if gpu['memory_total'] > 0 else 100)
        )
    
    # Sort by score (lower is better)
    sorted_gpus = sorted(gpus, key=gpu_score)
    best_gpu = sorted_gpus[0]
    
    if verbose:
        print(f"Selected GPU {best_gpu['index']}: {best_gpu['name']}")
        print(f"  - Utilization: {best_gpu['utilization']:.1f}%")
        print(f"  - Memory: {best_gpu['memory_free']:.0f} MB free / {best_gpu['memory_total']:.0f} MB total")
        print(f"  - Running processes: {best_gpu['processes']}")
        print("=" * 70 + "\n")
    
    return best_gpu['index']


def setup_device(config: dict, verbose: bool = True) -> torch.device:
    """
    Setup the training device, automatically selecting the best GPU if available.
    
    Args:
        config: Experiment configuration dict
        verbose: Whether to print device selection info
        
    Returns:
        torch.device for training
    """
    device_cfg = config['experiment'].get('device', 'cuda')
    visible_devices_env = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    visible_devices = []
    if visible_devices_env:
        visible_devices = [int(device.strip()) for device in visible_devices_env.split(',') if device.strip()]
    
    # Check if specific GPU is requested in config (e.g., "cuda:1")
    if device_cfg.startswith('cuda:'):
        gpu_idx = int(device_cfg.split(':')[1])
        if torch.cuda.is_available() and gpu_idx < torch.cuda.device_count():
            device = torch.device(device_cfg)
            if verbose:
                print(f"Using specified GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")
            return device
        else:
            if verbose:
                print(f"Warning: Specified GPU {gpu_idx} not available, auto-selecting...")
    
    # Auto-select best GPU
    if torch.cuda.is_available():
        best_gpu = select_best_gpu(verbose=verbose)
        if best_gpu is not None:
            local_gpu_idx = best_gpu
            if visible_devices:
                if best_gpu in visible_devices:
                    local_gpu_idx = visible_devices.index(best_gpu)
                else:
                    local_gpu_idx = 0
                if verbose:
                    print(f"Mapping physical GPU {best_gpu} to local CUDA device {local_gpu_idx} due to CUDA_VISIBLE_DEVICES={visible_devices_env}")

            device = torch.device(f'cuda:{local_gpu_idx}')
            # Set as default device for CUDA tensors
            torch.cuda.set_device(local_gpu_idx)
            return device
        else:
            # Fall back to default CUDA device
            device = torch.device('cuda')
            if verbose:
                print(f"Using default CUDA device: {torch.cuda.get_device_name(0)}")
            return device
    else:
        if verbose:
            print("CUDA not available, using CPU")
        return torch.device('cpu')


# ============================================================================
# Data Loading
# ============================================================================

def resolve_normalization_config(data_cfg: dict) -> Tuple[bool, str]:
    """Resolve whether normalization is enabled and which mode to use."""
    norm_cfg = data_cfg.get('normalization', {})

    if isinstance(norm_cfg, dict):
        enabled = bool(norm_cfg.get('enabled', data_cfg.get('normalize', True)))
        mode = norm_cfg.get('mode', 'session' if enabled else 'none')
    else:
        enabled = bool(data_cfg.get('normalize', True))
        mode = 'session' if enabled else 'none'

    if not enabled:
        mode = 'none'

    return enabled, mode

def create_dataloader(config: dict, split: str) -> DataLoader:
    """Create dataloader for specified split."""
    data_cfg = config['data']
    normalize, normalization_mode = resolve_normalization_config(data_cfg)
    
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
        normalize=normalize,
        normalization_mode=normalization_mode,
        preprocessing=data_cfg.get('preprocessing', {}),
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


def compute_auxiliary_reconstruction_loss(
    original: torch.Tensor,
    reconstructed: Optional[torch.Tensor],
    config: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute optional reconstruction-side auxiliary losses from config."""
    if reconstructed is None:
        return torch.tensor(0.0, device=original.device), {}

    loss_cfg = config.get('loss', {})
    aux_loss = torch.tensor(0.0, device=original.device)
    aux_metrics: Dict[str, float] = {}

    spectral_cfg = loss_cfg.get('spectral', {})
    if spectral_cfg.get('enabled', False):
        spectral_weight = float(spectral_cfg.get('weight', 0.0))
        if spectral_weight > 0:
            spectral_type = spectral_cfg.get('type', 'multi_stft')
            if spectral_type == 'multi_stft':
                spectral_loss = compute_multi_stft_loss(
                    original,
                    reconstructed,
                    fft_sizes=spectral_cfg.get('fft_sizes', [64, 128, 256]),
                    hop_sizes=spectral_cfg.get('hop_sizes'),
                    win_sizes=spectral_cfg.get('win_sizes'),
                )
            elif spectral_type == 'fft_mse':
                spectral_loss = compute_spectral_mse(
                    original,
                    reconstructed,
                    n_fft=int(spectral_cfg.get('n_fft', 256)),
                )
            else:
                raise ValueError(f"Unsupported spectral loss type: {spectral_type}")

            aux_loss = aux_loss + spectral_weight * spectral_loss
            aux_metrics['spectral_loss'] = float(spectral_loss.item())

    band_cfg = loss_cfg.get('band_power', {})
    if band_cfg.get('enabled', False):
        band_weight = float(band_cfg.get('weight', 0.0))
        if band_weight > 0:
            band_losses = compute_band_power_loss(
                original,
                reconstructed,
                fs=float(config.get('data', {}).get('preprocessing', {}).get('resample_rate', 200.0)),
            )
            if band_losses:
                band_loss = torch.stack(list(band_losses.values())).mean()
                aux_loss = aux_loss + band_weight * band_loss
                aux_metrics['band_power_loss'] = float(band_loss.item())

    smooth_cfg = loss_cfg.get('smoothness', {})
    if smooth_cfg.get('enabled', False):
        smooth_weight = float(smooth_cfg.get('weight', 0.0))
        if smooth_weight > 0:
            smoothness_loss = compute_smoothness_loss(reconstructed)
            aux_loss = aux_loss + smooth_weight * smoothness_loss
            aux_metrics['smoothness_loss'] = float(smoothness_loss.item())

    if aux_metrics:
        aux_metrics['aux_loss'] = float(aux_loss.item())

    return aux_loss, aux_metrics


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

        reconstructed = std_outputs.get('reconstructed')
        aux_loss, aux_metrics = compute_auxiliary_reconstruction_loss(x_input, reconstructed, config)
        total_optimized_loss = loss + aux_loss
        
        # Backward
        optimizer.zero_grad()
        total_optimized_loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += total_optimized_loss.item() * B
        total_samples += B
        n_batches += 1
        
        # Accumulate loss breakdown
        breakdown = StandardizedOutput.get_loss_breakdown(outputs)
        if aux_metrics:
            breakdown.update(aux_metrics)
        breakdown['tokenizer_loss'] = float(loss.item())
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
        reconstructed = std_outputs.get('reconstructed')
        aux_loss, aux_metrics = compute_auxiliary_reconstruction_loss(x_input, reconstructed, config)
        total_eval_loss = loss + aux_loss if loss is not None else aux_loss
        if total_eval_loss is not None:
            total_loss += total_eval_loss.item() * B
        
        total_samples += B
        n_batches += 1
        
        breakdown = StandardizedOutput.get_loss_breakdown(outputs)
        if aux_metrics:
            breakdown.update(aux_metrics)
        if loss is not None:
            breakdown['tokenizer_loss'] = float(loss.item())
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


@torch.no_grad()
def analyze_tokenizer_split(
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
    split: str,
) -> Dict[str, Any]:
    """Run detailed reconstruction/codebook analysis on a dataset split."""
    tokenizer.eval()

    tokenizer_type = config['model'].get('type', 'patch_vqvae')
    patch_size = get_patch_size(tokenizer, config)
    sample_rate = float(config['data']['preprocessing'].get('resample_rate', 200.0))

    totals: Dict[str, float] = {}
    total_samples = 0
    token_change_num = 0.0
    token_change_den = 0.0
    all_tokens = []
    per_subject: Dict[int, Dict[str, float]] = {}

    codebook_size = None
    if hasattr(tokenizer, 'get_codebook_size'):
        codebook_size = int(tokenizer.get_codebook_size())
    else:
        quant_cfg = config['model'].get('quantizer', {})
        codebook_size = int(quant_cfg.get('n_embed') or quant_cfg.get('codebook_size') or quant_cfg.get('num_codes', 0))

    for batch in dataloader:
        if isinstance(batch, dict):
            x = batch['data']
            subject_ids = batch.get('subject_id')
        else:
            x = batch[0]
            subject_ids = None

        x = x.to(device)
        x_input = prepare_input(x, patch_size, tokenizer_type)
        outputs = tokenizer(x_input)
        std_outputs = StandardizedOutput.standardize(outputs)
        reconstructed = std_outputs.get('reconstructed')
        if reconstructed is None:
            continue

        batch_n = x_input.shape[0]
        total_samples += batch_n

        batch_metrics = {
            'mse': compute_reconstruction_mse(x_input, reconstructed),
            'mae': compute_mae(x_input, reconstructed),
            'snr_db': compute_snr(x_input, reconstructed),
            'spectral_mse': compute_spectral_mse(x_input, reconstructed, n_fft=256),
            'multi_stft': compute_multi_stft_loss(
                x_input,
                reconstructed,
                fft_sizes=[64, 128, 256],
                hop_sizes=[16, 32, 64],
                win_sizes=[64, 128, 256],
            ),
        }
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_n

        band_losses = compute_band_power_loss(x_input, reconstructed, fs=sample_rate)
        for key, value in band_losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_n

        tokens = std_outputs.get('tokens')
        if tokens is None and 'indices' in outputs:
            tokens = outputs['indices']
        if isinstance(tokens, list):
            tokens = tokens[0] if tokens else None
        if torch.is_tensor(tokens):
            tokens_cpu = tokens.detach().cpu()
            all_tokens.append(tokens_cpu.reshape(-1))
            if tokens_cpu.dim() >= 2 and tokens_cpu.shape[1] > 1:
                token_change_num += (tokens_cpu[:, 1:] != tokens_cpu[:, :-1]).float().sum().item()
                token_change_den += tokens_cpu[:, 1:].numel()

        if subject_ids is not None:
            subject_ids = subject_ids.detach().cpu()
            sample_mse = ((reconstructed - x_input) ** 2).mean(dim=-1).detach().cpu()
            if sample_mse.dim() > 1:
                sample_mse = sample_mse.mean(dim=-1)
            sample_mse = sample_mse.view(x.shape[0], -1).mean(dim=1)
            for subject_id, mse in zip(subject_ids.tolist(), sample_mse.tolist()):
                stats = per_subject.setdefault(int(subject_id), {'mse_sum': 0.0, 'count': 0})
                stats['mse_sum'] += float(mse)
                stats['count'] += 1

    averaged = {key: value / max(total_samples, 1) for key, value in totals.items()}
    health = None
    if all_tokens and codebook_size > 0:
        flat_tokens = torch.cat(all_tokens, dim=0)
        health = compute_codebook_health(flat_tokens, codebook_size, include_distribution=True, top_k=20)

    subject_summary = {}
    for subject_id, stats in sorted(per_subject.items()):
        if stats['count'] > 0:
            subject_summary[str(subject_id)] = stats['mse_sum'] / stats['count']

    result: Dict[str, Any] = {
        'split': split,
        'num_samples': int(total_samples),
        **averaged,
        'token_change_ratio': float(token_change_num / token_change_den) if token_change_den > 0 else 0.0,
        'subject_mse': subject_summary,
    }
    if health is not None:
        result['codebook'] = {
            'active_codes': int(health['active_codes']),
            'usage_rate': float(health['code_utilization']),
            'dead_codes': int(health['dead_codes']),
            'perplexity': float(health['perplexity']),
            'gini_coefficient': float(health.get('gini_coefficient', 0.0)),
            'top_20_coverage': float(health.get('top_20_coverage', 0.0)),
        }

    return result


def save_detailed_analysis(
    tokenizer,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: dict,
    run_dir: Path,
    sample_spectral_metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Save detailed analysis report into the current run directory."""
    analysis_dir = run_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'generated_at': datetime.now().isoformat(),
        'experiment_name': config.get('experiment', {}).get('name', 'unknown'),
        'model_type': config.get('model', {}).get('type', 'unknown'),
        'val': analyze_tokenizer_split(tokenizer, val_loader, device, config, split='val'),
        'test': analyze_tokenizer_split(tokenizer, test_loader, device, config, split='test'),
    }
    if sample_spectral_metrics:
        report['sample_spectral_metrics'] = sample_spectral_metrics

    report_path = analysis_dir / 'detailed_analysis.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report_path


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
        original = original.detach()
        reconstructed = reconstructed.detach()
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
    parser.add_argument('--foreground', '-f', action='store_true',
                        help='Run in foreground (default is background with nohup)')
    args = parser.parse_args()
    
    # If not foreground mode, re-launch as background process
    if not args.foreground and not os.environ.get('TOKENIZER_TRAINING_BG'):
        import subprocess
        import sys
        
        # Create log directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Extract experiment name from config
        config_name = Path(args.config).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{config_name}_{timestamp}.log'
        
        # Build command with foreground flag and env var
        cmd = [sys.executable, __file__, '--config', args.config, '--foreground']
        if args.resume:
            cmd.extend(['--resume', args.resume])
        
        # Set environment variable to mark this as background run
        env = os.environ.copy()
        env['TOKENIZER_TRAINING_BG'] = '1'
        
        # Launch background process with nohup-like behavior
        with open(log_file, 'w') as log_f:
            # Use DEVNULL for stdin to fully detach
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.DEVNULL,
                stdout=log_f, 
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        
        print(f"Training started in background (PID: {process.pid})")
        print(f"Log file: {log_file}")
        print(f"Monitor: tail -f {log_file}")
        print(f"TensorBoard: tensorboard --logdir experiments/runs")
        sys.exit(0)  # Explicitly exit parent process
    
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
    
    # Device - Auto-select best GPU
    device = setup_device(config, verbose=True)
    print(f"Training device: {device}")
    
    # Seed
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader = create_dataloader(config, 'train')
    val_loader = create_dataloader(config, 'val')
    test_loader = create_dataloader(config, 'test')
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
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
    monitor_metric = es_cfg.get('metric', 'val_loss')
    monitor_mode = es_cfg.get('mode', 'min')
    best_monitor_value = float('inf') if monitor_mode == 'min' else float('-inf')
    epochs_without_improvement = 0
    
    # Checkpoint config
    ckpt_cfg = train_cfg.get('checkpoint', {})
    save_every = ckpt_cfg.get('save_every', 10)
    
    # Visualization config
    viz_cfg = train_cfg.get('visualization', {})
    viz_interval = viz_cfg.get('interval', 10)  # Log visualizations every N epochs
    
    # Checkpoint directory (correct location!)
    checkpoints_dir = logger.checkpoints_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Get learning rate for display
    actual_lr = opt_cfg.get('lr', train_cfg.get('learning_rate', 1e-3))
    
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(run_dir=logger.run_dir)
    print(f"  TensorBoard: tensorboard --logdir {logger.run_dir / 'tensorboard'}")
    
    # Get sampling rate for visualizations
    sr = config['data']['preprocessing'].get('resample_rate', 200)
    
    # Pre-compute tokenizer settings for the training loop
    tokenizer_type = config['model'].get('type', 'patch_vqvae')
    patch_size = get_patch_size(tokenizer, config)
    
    # Training loop
    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {actual_lr}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Early stopping metric: {monitor_metric} ({monitor_mode})")
    print(f"  Checkpoints saved to: {checkpoints_dir}")
    print(f"  Visualization interval: every {viz_interval} epochs")
    
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
        
        for key in ['amp_loss', 'phase_loss', 'time_loss', 'vq_loss', 'rec_loss', 'spectral_loss', 'band_power_loss', 'smoothness_loss', 'aux_loss']:
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
        
        # ========================================
        # TensorBoard Logging
        # ========================================
        step = epoch + 1
        
        # Log scalar metrics to TensorBoard
        tb_logger.log_scalars("train", train_metrics, step)
        tb_logger.log_scalars("val", {k.replace('val_', ''): v for k, v in val_metrics.items()}, step)
        tb_logger.log_learning_rate(current_lr, step)
        
        # Log loss breakdown
        loss_breakdown = {k: v for k, v in train_metrics.items() if '_loss' in k}
        if loss_breakdown:
            tb_logger.log_loss_breakdown(loss_breakdown, step, prefix="loss_components")
        
        # Periodic visualization (reconstruction, spectral, codebook, t-SNE)
        if (epoch + 1) % viz_interval == 0 or epoch == 0:
            try:
                print(f"  [Viz] Generating epoch {epoch+1} visualizations...")
                tokenizer.eval()
                
                # Collect samples for visualization
                viz_originals = []
                viz_reconstructed = []
                viz_indices = []
                viz_latents = []
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        if isinstance(batch, dict):
                            x = batch['data']
                        else:
                            x = batch[0]
                        
                        x = x.to(device)
                        x_input = prepare_input(x, patch_size, tokenizer_type)
                        
                        outputs = tokenizer(x_input)
                        std_out = StandardizedOutput.standardize(outputs)
                        
                        viz_originals.append(x_input.cpu())
                        
                        if 'reconstructed' in std_out:
                            viz_reconstructed.append(std_out['reconstructed'].cpu())
                        if 'tokens' in std_out:
                            tokens = std_out['tokens']
                            # Handle different token formats
                            if isinstance(tokens, list):
                                # NeuroRVQ returns list of tensors (one per branch)
                                # Flatten to single list for visualization
                                if len(tokens) > 0 and torch.is_tensor(tokens[0]):
                                    # Stack: [num_branches, ...] -> flatten
                                    tokens = torch.cat([t.flatten() for t in tokens])
                            elif torch.is_tensor(tokens):
                                # Handle RVQ multi-layer tokens [num_quantizers, B, ...] -> [B]
                                if tokens.dim() > 1:
                                    if tokens.dim() == 2 and tokens.shape[0] < tokens.shape[1]:
                                        # RVQ format: use first layer
                                        tokens = tokens[0]
                                    else:
                                        tokens = tokens.flatten()
                            viz_indices.append(tokens.cpu() if torch.is_tensor(tokens) else tokens)
                        if 'quantized' in std_out:
                            quantized = std_out['quantized']
                            # Handle list format (NeuroRVQ returns list of tensors)
                            if isinstance(quantized, list):
                                if len(quantized) > 0 and torch.is_tensor(quantized[0]):
                                    # Concatenate all quantized tensors
                                    quantized = torch.cat([q.flatten(1) for q in quantized], dim=-1)
                            if torch.is_tensor(quantized):
                                viz_latents.append(quantized.cpu())
                        elif 'pre_quant' in outputs:
                            viz_latents.append(outputs['pre_quant'].cpu())
                        
                        # Limit samples
                        if sum(o.shape[0] for o in viz_originals) >= 100:
                            break
                
                original_viz = torch.cat(viz_originals, dim=0)
                
                # Log reconstruction
                if viz_reconstructed:
                    reconstructed_viz = torch.cat(viz_reconstructed, dim=0)
                    tb_logger.log_reconstruction(
                        original_viz, reconstructed_viz, step, 
                        n_samples=4, fs=sr
                    )
                    
                    # Log spectral comparison
                    tb_logger.log_spectral_comparison(
                        original_viz, reconstructed_viz, step,
                        fs=sr, n_samples=50
                    )
                
                # Log codebook usage
                if viz_indices:
                    indices_viz = torch.cat(viz_indices, dim=0)
                    # Get codebook size - prefer model method over config
                    if hasattr(tokenizer, 'get_codebook_size'):
                        codebook_size = tokenizer.get_codebook_size()
                    else:
                        quant_cfg = config['model'].get('quantizer', {})
                        codebook_size = (quant_cfg.get('n_embed') or 
                                         quant_cfg.get('codebook_size') or 
                                         quant_cfg.get('num_codes', 2048))
                    tb_logger.log_codebook_usage(indices_viz, codebook_size, step)
                
                # Log t-SNE of codebook embeddings
                if hasattr(tokenizer, 'get_codebook_embeddings') and viz_indices:
                    embeddings = tokenizer.get_codebook_embeddings()
                    if embeddings is not None:
                        flat_indices = indices_viz.flatten().long()
                        usage = torch.bincount(flat_indices, minlength=codebook_size)
                        tb_logger.log_embedding_tsne(embeddings, usage, step)
                elif hasattr(tokenizer, 'quantizer') and viz_indices:
                    # Try to get embeddings from quantizer
                    try:
                        if hasattr(tokenizer.quantizer, 'embedding'):
                            embeddings = tokenizer.quantizer.embedding.detach()
                        elif hasattr(tokenizer.quantizer, 'codebook'):
                            embeddings = tokenizer.quantizer.codebook.detach()
                        elif hasattr(tokenizer.quantizer, 'layers'):
                            # RVQ - use first layer
                            embeddings = tokenizer.quantizer.layers[0].embedding.detach()
                        else:
                            embeddings = None
                        
                        if embeddings is not None:
                            codebook_size = embeddings.shape[0]
                            flat_indices = indices_viz.flatten().long()
                            # Clamp indices to valid range
                            flat_indices = flat_indices.clamp(0, codebook_size - 1)
                            usage = torch.bincount(flat_indices, minlength=codebook_size)
                            tb_logger.log_embedding_tsne(embeddings, usage, step)
                    except Exception as e:
                        print(f"    [Viz] Could not extract embeddings: {e}")
                
                # Log latent distribution
                if viz_latents:
                    latents_viz = torch.cat(viz_latents, dim=0)
                    tb_logger.log_latent_distribution(latents_viz, step)
                
                # Log loss pie chart (every 5 visualization intervals)
                if (epoch + 1) % (viz_interval * 5) == 0 and loss_breakdown:
                    tb_logger.log_loss_pie_chart(loss_breakdown, step)
                
                tokenizer.train()
                tb_logger.flush()
                
            except Exception as e:
                print(f"    [Viz] Warning: Visualization failed: {e}")
                traceback.print_exc()
                tokenizer.train()
        
        # Check for improvement
        monitor_value = val_metrics.get(monitor_metric, val_metrics['val_loss'])
        if monitor_mode == 'max':
            improved = monitor_value > best_monitor_value + min_delta
        else:
            improved = monitor_value < best_monitor_value - min_delta

        val_loss = val_metrics['val_loss']
        if improved:
            best_monitor_value = monitor_value
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
    
    # Close TensorBoard writer
    tb_logger.close()
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best {monitor_metric}: {best_monitor_value:.4f}")
    
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
        sample_spectral_metrics = None
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
                sample_spectral_metrics = spectral_metrics
                print(f"\nReconstruction Quality (sample):")
                for k, v in spectral_metrics.items():
                    print(f"  {k}: {v:.4f}")
                final_val.update({f'spectral_{k}': v for k, v in spectral_metrics.items()})
    except Exception as e:
        print(f"Warning: Spectral analysis failed: {e}")
        traceback.print_exc()

    try:
        analysis_path = save_detailed_analysis(
            tokenizer,
            val_loader,
            test_loader,
            device,
            config,
            logger.run_dir,
            sample_spectral_metrics=sample_spectral_metrics,
        )
        print(f"Detailed analysis saved to: {analysis_path}")
    except Exception as e:
        print(f"Warning: Detailed analysis generation failed: {e}")
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
            # Get codebook size - prefer model method over config
            if hasattr(tokenizer, 'get_codebook_size'):
                codebook_size = tokenizer.get_codebook_size()
            else:
                codebook_size = config['model'].get('quantizer', {}).get('codebook_size', 
                               config['model'].get('quantizer', {}).get('num_codes', 2048))
            visualizer.plot_codebook_usage(indices, codebook_size)
            
            # 6. Token embeddings (if available)
            # --- Generate t-SNE for all validation data ---
            try:
                if hasattr(tokenizer, 'get_codebook_embeddings'):
                    embeddings = tokenizer.get_codebook_embeddings()
                    if embeddings is not None:
                        # Collect all codebook indices from validation set
                        all_indices = []
                        for batch in val_loader:
                            if isinstance(batch, dict):
                                x = batch['data']
                            else:
                                x = batch[0]
                            x = x.to(device)
                            patch_size = get_patch_size(tokenizer, config)
                            tokenizer_type = config['model'].get('type', 'patch_vqvae')
                            x_input = prepare_input(x, patch_size, tokenizer_type)
                            outputs = tokenizer(x_input)
                            std_out = StandardizedOutput.standardize(outputs)
                            if 'indices' in std_out:
                                idx = std_out['indices']
                                if isinstance(idx, (list, tuple)):
                                    idx = idx[0]
                                all_indices.append(idx.detach().cpu().flatten())
                        if all_indices:
                            all_indices = torch.cat(all_indices, dim=0)
                            codebook_size = embeddings.shape[0]
                            usage = torch.bincount(all_indices.long(), minlength=codebook_size)
                            visualizer.plot_token_embeddings(embeddings, usage, method='tsne', filename='token_embeddings_tsne.png')
            except Exception as e:
                print(f"[Summary] t-SNE generation failed: {e}")
        
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
            'best_monitor_metric': monitor_metric,
            'best_monitor_value': best_monitor_value,
            **final_val,
        }
        logger.log_final(final_metrics)
        
        # Log hyperparameters to TensorBoard
        hparams = {
            'model_type': config['model'].get('type', 'unknown'),
            'patch_size': patch_size,
            'batch_size': train_cfg['batch_size'],
            'learning_rate': actual_lr,
            'epochs': train_cfg['epochs'],
            'codebook_size': config['model'].get('quantizer', {}).get('codebook_size',
                            config['model'].get('quantizer', {}).get('num_codes', 2048)),
        }
        tb_logger.log_hparams(hparams, {
            'best_monitor_value': best_monitor_value,
            'final_val_loss': final_val.get('val_loss', 0.0),
            'final_utilization': final_val.get('val_utilization', 0.0),
        })
    except Exception as e:
        print(f"Warning: Could not log final metrics: {e}")
    
    # Generate experiment logger figures
    try:
        logger.generate_figures()
    except Exception as e:
        print(f"Warning: Could not generate logger figures: {e}")
    
    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"Results saved to: {logger.run_dir}")
    print(f"Best model: {checkpoints_dir / 'best_model.pt'}")
    print(f"TensorBoard: tensorboard --logdir {logger.run_dir / 'tensorboard'}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    tee_logger.close()


if __name__ == '__main__':
    main()
