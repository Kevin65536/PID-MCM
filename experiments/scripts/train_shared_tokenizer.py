#!/usr/bin/env python
"""
Train a shared-codebook EEG+fNIRS tokenizer with explicit alignment losses.
"""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import create_configured_multimodal_dataloaders
from src.tokenizers import create_tokenizer
from src.utils.logger import ExperimentLogger
from src.visualization import TensorBoardLogger
from src.visualization import analyze_factorized_alignment, analyze_shared_alignment


class TeeLogger:
    """Write logs to both stdout and a file."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.error_terminal = sys.stderr
        self.log_file = open(log_file, 'a', buffering=1)
        self.closed = False

    def write(self, message: str):
        if self.closed:
            self.terminal.write(message)
            return
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if not self.closed:
            self.log_file.flush()

    def close(self):
        if self.closed:
            return
        sys.stdout = self.terminal
        sys.stderr = self.error_terminal
        self.log_file.close()
        self.closed = True


def setup_logging(run_dir: Path) -> TeeLogger:
    log_file = run_dir / "training.log"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee


def resolve_normalization_config(data_cfg: dict) -> Tuple[bool, str]:
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


def setup_device(config: dict) -> torch.device:
    requested = config.get('experiment', {}).get('device', 'cuda')
    if requested.startswith('cuda') and torch.cuda.is_available():
        if requested == 'cuda':
            return torch.device('cuda')
        return torch.device(requested)
    if requested == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_multimodal_dataloaders(config: dict):
    return create_configured_multimodal_dataloaders(config)


def tensor_to_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


def get_gradient_attribution_config(config: dict) -> Dict[str, Any]:
    grad_cfg = config.get('training', {}).get('gradient_attribution', {})
    return {
        'enabled': bool(grad_cfg.get('enabled', False)),
        'max_batches': max(int(grad_cfg.get('max_batches', 1)), 1),
    }


def compute_gradient_attribution(model, outputs: Dict[str, Any]) -> Dict[str, float]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        return {}

    component_specs = {
        'eeg_rec_loss': 1.0,
        'fnirs_rec_loss': 1.0,
        'vq_loss': 1.0,
    }
    if hasattr(model, 'get_gradient_component_weights'):
        component_specs.update(getattr(model, 'get_gradient_component_weights')())
    else:
        alignment_scale = float(getattr(model, 'get_alignment_scale', lambda: 1.0)())
        component_specs.update({
            'latent_align_loss': float(getattr(model, 'latent_alignment_weight', 0.0)) * alignment_scale,
            'assignment_align_loss': float(getattr(model, 'assignment_alignment_weight', 0.0)) * alignment_scale,
        })

    component_norms: Dict[str, float] = {}
    component_values: Dict[str, float] = {}

    for name, weight in component_specs.items():
        term = outputs.get(name)
        if term is None or not torch.is_tensor(term) or term.ndim != 0 or abs(weight) <= 0.0:
            continue
        grads = torch.autograd.grad(weight * term, params, retain_graph=True, allow_unused=True)
        squared_norm = None
        for grad in grads:
            if grad is None:
                continue
            grad_value = grad.detach()
            grad_sq = torch.sum(grad_value * grad_value)
            squared_norm = grad_sq if squared_norm is None else squared_norm + grad_sq
        if squared_norm is None:
            continue
        component_norms[name] = float(torch.sqrt(squared_norm).item())
        component_values[f'weighted_term_{name}'] = float((weight * term.detach()).item())

    if not component_norms:
        return component_values

    total_component_norm = sum(component_norms.values()) + 1e-12
    attribution_metrics: Dict[str, float] = {}
    for name, norm in component_norms.items():
        attribution_metrics[f'grad_norm_{name}'] = norm
        attribution_metrics[f'grad_share_{name}'] = norm / total_component_norm
    attribution_metrics.update(component_values)
    return attribution_metrics


def strip_metric_prefix(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in metrics.items()
    }


def resolve_sampling_rate(data_cfg: dict, modality: str, default: float) -> float:
    modality_cfg = data_cfg.get(f'{modality}_preprocessing', {})
    for key in ('target_sampling_rate', 'resample_rate', 'sampling_rate', 'sample_rate'):
        value = modality_cfg.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(default)


def extract_tensorboard_hparams(config: dict) -> Dict[str, Any]:
    experiment_cfg = config.get('experiment', {})
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    quantizer_cfg = model_cfg.get('quantizer', {})
    window_cfg = data_cfg.get('window', {})

    hparams = {
        'experiment_name': experiment_cfg.get('name', 'unknown'),
        'seed': experiment_cfg.get('seed', 42),
        'device': experiment_cfg.get('device', 'auto'),
        'epochs': training_cfg.get('epochs'),
        'batch_size': training_cfg.get('batch_size'),
        'learning_rate': training_cfg.get('learning_rate'),
        'weight_decay': training_cfg.get('weight_decay', 0.0),
        'warmup_epochs': training_cfg.get('warmup_epochs', 0),
        'min_lr': training_cfg.get('min_lr', 1e-6),
        'window_duration_s': window_cfg.get('duration_s'),
        'window_offset_ms': window_cfg.get('offset_ms', 0),
        'task': data_cfg.get('task', 'motor_imagery'),
        'tokenizer_type': model_cfg.get('type', 'unknown'),
        'codebook_size': quantizer_cfg.get('n_embed', quantizer_cfg.get('codebook_size', quantizer_cfg.get('num_codes'))),
        'quantizer_beta': quantizer_cfg.get('beta'),
    }

    return {key: value for key, value in hparams.items() if value is not None}


@torch.no_grad()
def collect_visualization_artifacts(model, dataloader, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return {}

    was_training = model.training
    model.eval()
    try:
        eeg = batch['eeg'].to(device, non_blocking=True)
        fnirs = batch['fnirs'].to(device, non_blocking=True)
        outputs = model(eeg, fnirs)
    finally:
        if was_training:
            model.train()

    artifacts: Dict[str, torch.Tensor] = {
        'eeg': eeg.detach().cpu(),
        'fnirs': fnirs.detach().cpu(),
    }
    for key in (
        'eeg_reconstructed',
        'fnirs_reconstructed',
        'eeg_indices',
        'fnirs_indices',
        'eeg_z',
        'fnirs_z',
        'eeg_private_indices',
        'fnirs_private_indices',
        'eeg_private_z',
        'fnirs_private_z',
    ):
        value = outputs.get(key)
        if torch.is_tensor(value):
            artifacts[key] = value.detach().cpu()

    return artifacts


def log_tensorboard_visualizations(
    tb_logger: TensorBoardLogger,
    model,
    artifacts: Dict[str, torch.Tensor],
    step: int,
    eeg_fs: float,
    fnirs_fs: float,
):
    if not tb_logger.enabled or not artifacts:
        return

    if 'eeg' in artifacts and 'eeg_reconstructed' in artifacts:
        tb_logger.log_reconstruction(
            artifacts['eeg'],
            artifacts['eeg_reconstructed'],
            step,
            n_samples=4,
            fs=eeg_fs,
            tag='shared_eeg_reconstruction',
        )

    if 'fnirs' in artifacts and 'fnirs_reconstructed' in artifacts:
        tb_logger.log_reconstruction(
            artifacts['fnirs'],
            artifacts['fnirs_reconstructed'],
            step,
            n_samples=4,
            fs=fnirs_fs,
            tag='shared_fnirs_reconstruction',
        )

    if 'eeg_z' in artifacts:
        tb_logger.log_latent_distribution(artifacts['eeg_z'], step, tag='shared_eeg_latents')
    if 'fnirs_z' in artifacts:
        tb_logger.log_latent_distribution(artifacts['fnirs_z'], step, tag='shared_fnirs_latents')

    shared_indices = []
    for key in ('eeg_indices', 'fnirs_indices'):
        value = artifacts.get(key)
        if value is not None:
            shared_indices.append(value.reshape(-1).long())

    if not shared_indices:
        return

    combined_indices = torch.cat(shared_indices, dim=0)
    codebook_size = int(model.get_codebook_size())
    tb_logger.log_codebook_usage(combined_indices, codebook_size, step, tag='shared_codebook')

    quantizer = getattr(model, 'quantizer', None)
    if quantizer is not None and hasattr(quantizer, 'weight'):
        embeddings = quantizer.weight.detach().cpu()
        usage = torch.bincount(combined_indices.clamp(0, codebook_size - 1), minlength=codebook_size)
        tb_logger.log_embedding_tsne(embeddings, usage, step, tag='shared_codebook_embeddings')


def train_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    grad_clip: float,
    gradient_attribution_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    total_batches = 0
    grad_totals: Dict[str, float] = {}
    grad_batches = 0
    grad_cfg = gradient_attribution_cfg or {'enabled': False, 'max_batches': 1}

    for batch in dataloader:
        eeg = batch['eeg'].to(device, non_blocking=True)
        fnirs = batch['fnirs'].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(eeg, fnirs)
        loss = outputs['loss']

        if grad_cfg.get('enabled', False) and grad_batches < int(grad_cfg.get('max_batches', 1)):
            grad_metrics = compute_gradient_attribution(model, outputs)
            if grad_metrics:
                grad_batches += 1
                for key, value in grad_metrics.items():
                    grad_totals[key] = grad_totals.get(key, 0.0) + value

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_batches += 1

        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim == 0:
                totals[key] = totals.get(key, 0.0) + tensor_to_float(value)

    averaged = {key: value / max(total_batches, 1) for key, value in totals.items()}
    if grad_batches > 0:
        averaged.update({key: value / grad_batches for key, value in grad_totals.items()})
    return averaged


@torch.no_grad()
def validate_epoch(
    model,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    total_batches = 0

    for batch in dataloader:
        eeg = batch['eeg'].to(device, non_blocking=True)
        fnirs = batch['fnirs'].to(device, non_blocking=True)
        outputs = model(eeg, fnirs)
        total_batches += 1

        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim == 0:
                totals[f'val_{key}'] = totals.get(f'val_{key}', 0.0) + tensor_to_float(value)

    return {key: value / max(total_batches, 1) for key, value in totals.items()}


def load_checkpoint(path: Path, model, optimizer, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def compute_alignment_scale(epoch: int, config: dict) -> float:
    warm_cfg = config.get('training', {}).get('alignment_warmup', {})
    if not warm_cfg.get('enabled', False):
        return 1.0

    start_epoch = int(warm_cfg.get('start_epoch', 1))
    ramp_epochs = max(int(warm_cfg.get('ramp_epochs', 1)), 1)
    start_scale = float(warm_cfg.get('start_scale', 0.0))
    if epoch < start_epoch:
        return start_scale
    if ramp_epochs == 1:
        return 1.0

    progress = min(max((epoch - start_epoch) / (ramp_epochs - 1), 0.0), 1.0)
    return start_scale + (1.0 - start_scale) * progress


def maybe_apply_warm_start(model, config: dict, device: torch.device):
    warm_cfg = config.get('warm_start', {})
    if not warm_cfg:
        return

    def load_branch(checkpoint_path: str, branch_prefix: str):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        source_state = checkpoint['model_state_dict']
        target_state = model.state_dict()

        prefix_map = {
            'patch_embed.proj.': f'{branch_prefix}_patch_embed.proj.',
            'encoder.': f'{branch_prefix}_encoder.',
            'encode_task_layer.': f'{branch_prefix}_encode_proj.',
            'decode_input_proj.': f'{branch_prefix}_decode_input_proj.',
            'decoder.': f'{branch_prefix}_decoder.',
        }

        loaded_count = 0
        skipped_count = 0
        for source_prefix, target_prefix in prefix_map.items():
            for source_key, value in source_state.items():
                if not source_key.startswith(source_prefix):
                    continue
                target_key = target_prefix + source_key[len(source_prefix):]
                if target_key not in target_state or target_state[target_key].shape != value.shape:
                    skipped_count += 1
                    continue
                target_state[target_key] = value
                loaded_count += 1

        model.load_state_dict(target_state, strict=False)
        print(
            f'Warm-start {branch_prefix}: loaded {loaded_count} tensors, '
            f'skipped {skipped_count} incompatible tensors from {checkpoint_path}'
        )

    eeg_checkpoint = warm_cfg.get('eeg_checkpoint')
    fnirs_checkpoint = warm_cfg.get('fnirs_checkpoint')
    if eeg_checkpoint:
        load_branch(eeg_checkpoint, 'eeg')
    if fnirs_checkpoint:
        load_branch(fnirs_checkpoint, 'fnirs')


def build_checkpoint_payload(
    epoch: int,
    model,
    optimizer,
    config: dict,
    train_loss: float,
    val_loss: float,
    monitor_metric: str,
    monitor_value: float,
    best_epoch: int,
    best_monitor: float,
    alignment_scale: float,
    is_best: bool,
) -> Dict[str, Any]:
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'monitor_metric': monitor_metric,
        'monitor_value': float(monitor_value),
        'best_epoch': int(best_epoch),
        'best_monitor': float(best_monitor),
        'alignment_scale': float(alignment_scale),
        'is_best': bool(is_best),
    }


def finalize_training_run(
    *,
    logger: ExperimentLogger,
    tb_logger: TensorBoardLogger,
    model,
    val_loader,
    test_loader,
    config: dict,
    device: torch.device,
    best_epoch: int,
    best_monitor: float,
    skip_post_analysis: bool,
):
    best_path = logger.checkpoints_dir / 'best_model.pt'
    analysis_device = device

    if best_path.exists():
        print(f"\nLoading best checkpoint from {best_path}")
        best_checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        best_epoch = int(best_checkpoint.get('best_epoch', best_checkpoint.get('epoch', best_epoch)))
        best_monitor = float(best_checkpoint.get('best_monitor', best_monitor))
        monitor_metric = best_checkpoint.get('monitor_metric', 'val_loss')
        monitor_value = best_checkpoint.get('monitor_value')
        if monitor_value is not None:
            print(
                f"Best checkpoint metadata: epoch={best_checkpoint.get('epoch')}, "
                f"{monitor_metric}={float(monitor_value):.6f}, tracked_best_epoch={best_epoch}"
            )
    else:
        print("\nBest checkpoint not found, finalizing with in-memory model state")

    if hasattr(model, 'set_alignment_scale'):
        model.set_alignment_scale(1.0)

    test_metrics = validate_epoch(model, test_loader, device)
    final_metrics = {
        'best_epoch': best_epoch,
        'best_monitor': best_monitor,
        **test_metrics,
    }
    tb_logger.log_scalars('test', strip_metric_prefix(test_metrics, 'val_'), best_epoch)
    tb_logger.log_hparams(
        extract_tensorboard_hparams(config),
        {
            key: value
            for key, value in final_metrics.items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        },
    )
    tb_logger.flush()
    logger.log_final(final_metrics)
    logger.generate_figures()

    analysis_type = getattr(model, 'get_analysis_type', lambda: 'shared_alignment')()
    summary_path = logger.run_dir / 'analysis' / f'{analysis_type}_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(final_metrics, handle, indent=2)

    if skip_post_analysis:
        return final_metrics

    analysis_dir = logger.run_dir / 'analysis' / analysis_type
    print(f"Running default {analysis_type} analysis -> {analysis_dir}")
    analyzer = analyze_factorized_alignment if analysis_type == 'factorized_alignment' else analyze_shared_alignment
    analysis_results = analyzer(
        model=model,
        dataloaders={'val': val_loader, 'test': test_loader},
        config=config,
        output_dir=analysis_dir,
        device=analysis_device,
    )
    with open(analysis_dir / 'analysis_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(analysis_results, handle, indent=2)

    lag_set = config.get('validation', {}).get('lag_set', [])
    max_validation_lag = max(lag_set) if lag_set else None
    if max_validation_lag is not None:
        for split_name, split_result in analysis_results.get('splits', {}).items():
            best_lag = split_result.get('best_lag')
            if best_lag is not None and int(best_lag) >= int(max_validation_lag):
                print(
                    f"[Warning] {split_name} best_lag={best_lag} hit validation lag boundary "
                    f"({max_validation_lag}). Consider widening lag_set and training lag candidates."
                )

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Train shared EEG+fNIRS tokenizer")
    parser.add_argument('--config', required=True, help='Config path relative to experiments/configs')
    parser.add_argument('--resume', default=None, help='Optional checkpoint path')
    parser.add_argument('--skip-post-analysis', action='store_true', help='Skip default shared alignment analysis at the end of training')
    args = parser.parse_args()

    logger = ExperimentLogger(config_path=args.config)
    config = logger.config
    tee_logger = setup_logging(logger.run_dir)
    tb_logger = TensorBoardLogger(run_dir=logger.run_dir)

    try:
        print("=" * 70)
        print("Shared EEG+fNIRS Tokenizer Training")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Run directory: {logger.run_dir}")
        print(f"Experiment: {config['experiment']['name']}")
        print(f"Description: {config['experiment'].get('description', 'N/A')}")
        if tb_logger.enabled:
            print(f"TensorBoard: tensorboard --logdir {logger.run_dir / 'tensorboard'}")

        device = setup_device(config)
        print(f"Training device: {device}")

        seed = config['experiment'].get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("\nLoading multimodal data...")
        dataloaders = create_multimodal_dataloaders(config)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        print(f"Train trials: {len(train_loader.dataset)}")
        print(f"Val trials: {len(val_loader.dataset)}")
        print(f"Test trials: {len(test_loader.dataset)}")
        print(f"EEG channels: {train_loader.dataset.get_num_eeg_channels()}")
        print(f"fNIRS channels: {train_loader.dataset.get_num_fnirs_channels()}")
        if hasattr(train_loader.dataset, 'describe_sources'):
            print('Train source mix:')
            for source in train_loader.dataset.describe_sources():
                print(
                    f"  - {source['name']}: dataset={source['dataset']}, task={source['task']}, samples={source['length']}"
                )

        print("\nCreating tokenizer...")
        model = create_tokenizer(config).to(device)
        print(f"Model: {model.__class__.__name__}")
        print(f"Shared codebook size: {model.get_codebook_size()}")
        maybe_apply_warm_start(model, config, device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training'].get('learning_rate', 1e-4),
            weight_decay=config['training'].get('weight_decay', 0.01),
        )
        warmup_epochs = int(config['training'].get('warmup_epochs', 0))
        total_epochs = int(config['training']['epochs'])
        tb_cfg = config['training'].get('tensorboard', {})
        tb_viz_interval = max(int(tb_cfg.get('visualization_interval', 10)), 1)
        tb_flush_interval = max(int(tb_cfg.get('flush_interval', 1)), 1)
        tb_log_visualizations = bool(tb_cfg.get('log_visualizations', True))
        eeg_fs = resolve_sampling_rate(config.get('data', {}), 'eeg', default=200.0)
        fnirs_fs = resolve_sampling_rate(config.get('data', {}), 'fnirs', default=10.0)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=float(config['training'].get('min_lr', 1e-6)),
        )

        if tb_logger.enabled:
            tb_logger.log_text_summary(
                json.dumps(
                    {
                        'config': args.config,
                        'run_dir': str(logger.run_dir),
                        'device': str(device),
                    },
                    indent=2,
                ),
                step=0,
                tag='run_info',
            )

        start_epoch = 0
        if args.resume:
            checkpoint = load_checkpoint(Path(args.resume), model, optimizer, device)
            start_epoch = int(checkpoint.get('epoch', 0))
            resume_best_epoch = checkpoint.get('best_epoch')
            resume_best_monitor = checkpoint.get('best_monitor')
            print(f"Resumed from epoch {start_epoch}")
        else:
            resume_best_epoch = None
            resume_best_monitor = None

        es_cfg = config['training'].get('early_stopping', {})
        patience = int(es_cfg.get('patience', 40))
        monitor_metric = es_cfg.get('metric', 'val_loss')
        monitor_mode = es_cfg.get('mode', 'min')
        best_monitor = float('inf') if monitor_mode == 'min' else float('-inf')
        if resume_best_monitor is not None:
            best_monitor = float(resume_best_monitor)
        epochs_without_improvement = 0
        save_every = int(config['training'].get('checkpoint', {}).get('save_every', 1))
        grad_clip = float(config['training'].get('gradient', {}).get('clip_norm', 1.0))
        gradient_attribution_cfg = get_gradient_attribution_config(config)

        best_epoch = int(resume_best_epoch) if resume_best_epoch is not None else start_epoch
        interrupted = False
        stop_epoch = start_epoch
        try:
            for epoch in range(start_epoch + 1, total_epochs + 1):
                stop_epoch = epoch
                alignment_scale = compute_alignment_scale(epoch, config)
                if hasattr(model, 'set_alignment_scale'):
                    model.set_alignment_scale(alignment_scale)

                train_metrics = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    grad_clip,
                    gradient_attribution_cfg=gradient_attribution_cfg,
                )
                val_metrics = validate_epoch(model, val_loader, device)

                if epoch > warmup_epochs:
                    scheduler.step()

                lr = optimizer.param_groups[0]['lr']
                train_loss = train_metrics.get('loss', float('nan'))
                val_loss = val_metrics.get('val_loss', float('nan'))
                merged_metrics = {'lr': lr, 'alignment_scale': alignment_scale}
                merged_metrics.update({
                    key: value
                    for key, value in train_metrics.items()
                    if key.startswith('grad_') or key.startswith('weighted_term_')
                })
                merged_metrics.update({k: v for k, v in val_metrics.items() if k != 'val_loss'})
                logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    loss_breakdown={
                        k: v
                        for k, v in train_metrics.items()
                        if k != 'loss' and not k.startswith('grad_') and not k.startswith('weighted_term_')
                    },
                    metrics=merged_metrics,
                )

                step = epoch
                tb_logger.log_scalars('train', train_metrics, step)
                tb_logger.log_scalars('val', strip_metric_prefix(val_metrics, 'val_'), step)
                tb_logger.log_learning_rate(lr, step)

                train_loss_breakdown = {k: v for k, v in train_metrics.items() if k.endswith('_loss')}
                if train_loss_breakdown:
                    tb_logger.log_loss_breakdown(train_loss_breakdown, step, prefix='train_loss_components')

                val_loss_breakdown = {
                    key: value
                    for key, value in strip_metric_prefix(val_metrics, 'val_').items()
                    if key.endswith('_loss')
                }
                if val_loss_breakdown:
                    tb_logger.log_loss_breakdown(val_loss_breakdown, step, prefix='val_loss_components')

                if tb_log_visualizations and (epoch == start_epoch + 1 or step % tb_viz_interval == 0):
                    artifacts = collect_visualization_artifacts(model, val_loader, device)
                    log_tensorboard_visualizations(tb_logger, model, artifacts, step, eeg_fs=eeg_fs, fnirs_fs=fnirs_fs)

                if step % tb_flush_interval == 0:
                    tb_logger.flush()

                monitor_value = val_metrics.get(monitor_metric)
                if monitor_value is None:
                    raise ValueError(f"Monitor metric '{monitor_metric}' not found in validation metrics")

                improved = monitor_value < best_monitor if monitor_mode == 'min' else monitor_value > best_monitor
                if improved:
                    best_monitor = float(monitor_value)
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                checkpoint_payload = build_checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    monitor_metric=monitor_metric,
                    monitor_value=float(monitor_value),
                    best_epoch=best_epoch,
                    best_monitor=best_monitor,
                    alignment_scale=alignment_scale,
                    is_best=improved,
                )

                if epoch % save_every == 0 or improved:
                    logger.save_checkpoint(checkpoint_payload, epoch=epoch, is_best=improved)

                selected_lag = val_metrics.get('val_selected_alignment_lag')
                lag_candidates = getattr(model, 'alignment_lag_candidates', None)
                if lag_candidates and selected_lag is not None and int(round(float(selected_lag))) >= max(lag_candidates):
                    print(
                        f"[Warning] epoch {epoch}: selected alignment lag {selected_lag:.2f} hit the current "
                        f"training boundary {max(lag_candidates)}"
                    )

                if es_cfg.get('enabled', True) and epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                    break
        except KeyboardInterrupt:
            interrupted = True
            print(f"\nTraining interrupted at epoch {stop_epoch}. Finalizing from best available checkpoint...")

        final_metrics = finalize_training_run(
            logger=logger,
            tb_logger=tb_logger,
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            best_epoch=best_epoch,
            best_monitor=best_monitor,
            skip_post_analysis=args.skip_post_analysis,
        )

        if interrupted:
            print("\nTraining interrupted after finalization.")
        else:
            print("\nTraining complete.")
        print(f"Best epoch: {final_metrics['best_epoch']}")
        print(f"Final metrics saved to: {logger.run_dir / 'metrics.json'}")
    finally:
        tb_logger.close()
        tee_logger.close()


if __name__ == '__main__':
    main()
