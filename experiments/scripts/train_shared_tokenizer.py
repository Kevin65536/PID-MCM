#!/usr/bin/env python
"""
Train a shared-codebook EEG+fNIRS tokenizer with explicit alignment losses.
"""

import argparse
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

from src.data.eeg_fnirs_dataset import create_dataloaders
from src.tokenizers import create_tokenizer
from src.utils.logger import ExperimentLogger
from src.visualization.shared_alignment_analysis import analyze_shared_alignment


class TeeLogger:
    """Write logs to both stdout and a file."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a', buffering=1)

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


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
    data_cfg = config['data']
    normalize, normalization_mode = resolve_normalization_config(data_cfg)
    return create_dataloaders(
        data_root=data_cfg['data_root'],
        modality='both',
        task=data_cfg.get('task', 'motor_imagery'),
        train_subjects=data_cfg['split']['train_subjects'],
        val_subjects=data_cfg['split']['val_subjects'],
        test_subjects=data_cfg['split']['test_subjects'],
        window_duration_s=float(data_cfg['window']['duration_s']),
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg.get('num_workers', 0),
        window_offset_ms=float(data_cfg['window'].get('offset_ms', 0)),
        normalize=normalize,
        normalization_mode=normalization_mode,
        eeg_preprocessing=data_cfg.get('eeg_preprocessing', {}),
        fnirs_preprocessing=data_cfg.get('fnirs_preprocessing', {}),
        exclude_eog=data_cfg.get('exclude_eog', True),
        hbo_only=data_cfg.get('hbo_only', True),
        hbr_only=data_cfg.get('hbr_only', False),
    )


def tensor_to_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


def train_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    total_batches = 0

    for batch in dataloader:
        eeg = batch['eeg'].to(device)
        fnirs = batch['fnirs'].to(device)

        optimizer.zero_grad()
        outputs = model(eeg, fnirs)
        loss = outputs['loss']
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_batches += 1

        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim == 0:
                totals[key] = totals.get(key, 0.0) + tensor_to_float(value)

    return {key: value / max(total_batches, 1) for key, value in totals.items()}


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
        eeg = batch['eeg'].to(device)
        fnirs = batch['fnirs'].to(device)
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


def main():
    parser = argparse.ArgumentParser(description="Train shared EEG+fNIRS tokenizer")
    parser.add_argument('--config', required=True, help='Config path relative to experiments/configs')
    parser.add_argument('--resume', default=None, help='Optional checkpoint path')
    parser.add_argument('--skip-post-analysis', action='store_true', help='Skip default shared alignment analysis at the end of training')
    args = parser.parse_args()

    logger = ExperimentLogger(config_path=args.config)
    config = logger.config
    tee_logger = setup_logging(logger.run_dir)

    try:
        print("=" * 70)
        print("Shared EEG+fNIRS Tokenizer Training")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Run directory: {logger.run_dir}")
        print(f"Experiment: {config['experiment']['name']}")
        print(f"Description: {config['experiment'].get('description', 'N/A')}")

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
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=float(config['training'].get('min_lr', 1e-6)),
        )

        start_epoch = 0
        if args.resume:
            checkpoint = load_checkpoint(Path(args.resume), model, optimizer, device)
            start_epoch = int(checkpoint.get('epoch', 0))
            print(f"Resumed from epoch {start_epoch}")

        es_cfg = config['training'].get('early_stopping', {})
        patience = int(es_cfg.get('patience', 40))
        monitor_metric = es_cfg.get('metric', 'val_loss')
        monitor_mode = es_cfg.get('mode', 'min')
        best_monitor = float('inf') if monitor_mode == 'min' else float('-inf')
        epochs_without_improvement = 0
        save_every = int(config['training'].get('checkpoint', {}).get('save_every', 1))
        grad_clip = float(config['training'].get('gradient', {}).get('clip_norm', 1.0))

        best_epoch = start_epoch
        for epoch in range(start_epoch + 1, total_epochs + 1):
            alignment_scale = compute_alignment_scale(epoch, config)
            if hasattr(model, 'set_alignment_scale'):
                model.set_alignment_scale(alignment_scale)

            train_metrics = train_epoch(model, train_loader, optimizer, device, grad_clip)
            val_metrics = validate_epoch(model, val_loader, device)

            if epoch > warmup_epochs:
                scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            train_loss = train_metrics.get('loss', float('nan'))
            val_loss = val_metrics.get('val_loss', float('nan'))
            merged_metrics = {'lr': lr, 'alignment_scale': alignment_scale}
            merged_metrics.update({k: v for k, v in val_metrics.items() if k != 'val_loss'})
            logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                loss_breakdown={k: v for k, v in train_metrics.items() if k != 'loss'},
                metrics=merged_metrics,
            )

            monitor_value = val_metrics.get(monitor_metric)
            if monitor_value is None:
                raise ValueError(f"Monitor metric '{monitor_metric}' not found in validation metrics")

            improved = monitor_value < best_monitor if monitor_mode == 'min' else monitor_value > best_monitor
            if improved:
                best_monitor = monitor_value
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % save_every == 0 or improved:
                logger.save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                    },
                    epoch=epoch,
                    is_best=improved,
                )

            if es_cfg.get('enabled', True) and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

        best_path = logger.checkpoints_dir / "best_model.pt"
        if best_path.exists():
            print(f"\nLoading best checkpoint from {best_path}")
            best_checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])

        if hasattr(model, 'set_alignment_scale'):
            model.set_alignment_scale(1.0)

        test_metrics = validate_epoch(model, test_loader, device)
        final_metrics = {
            'best_epoch': best_epoch,
            'best_monitor': best_monitor,
            **test_metrics,
        }
        logger.log_final(final_metrics)
        logger.generate_figures()

        summary_path = logger.run_dir / "analysis" / "shared_alignment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(final_metrics, handle, indent=2)

        if not args.skip_post_analysis:
            analysis_dir = logger.run_dir / 'analysis' / 'shared_alignment'
            print(f"Running default shared alignment analysis -> {analysis_dir}")
            analysis_results = analyze_shared_alignment(
                model=model,
                dataloaders={'val': val_loader, 'test': test_loader},
                config=config,
                output_dir=analysis_dir,
                device=device,
            )
            with open(analysis_dir / 'analysis_summary.json', 'w', encoding='utf-8') as handle:
                json.dump(analysis_results, handle, indent=2)

        print("\nTraining complete.")
        print(f"Best epoch: {best_epoch}")
        print(f"Final metrics saved to: {logger.run_dir / 'metrics.json'}")
    finally:
        tee_logger.close()


if __name__ == '__main__':
    main()
