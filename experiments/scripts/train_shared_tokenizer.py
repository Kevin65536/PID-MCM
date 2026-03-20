#!/usr/bin/env python
"""
Train a shared-codebook EEG+fNIRS tokenizer with explicit alignment losses.

Enhancements over the original baseline:
- Warm-start from validated single-modality checkpoints via
  ``--init-eeg-checkpoint`` and ``--init-fnirs-checkpoint``.
- Alignment loss warmup: weights ramp from 0 to full over
  ``alignment_warmup.ramp_epochs`` epochs (configured in the YAML under
  ``training.alignment_warmup``).
- Lag-aware validation: per-epoch token-agreement is reported for lags
  [0, 1, 2, 3, 4] so delayed neurovascular coupling is visible in logs.
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
    lag_set: Optional[list] = None,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    total_batches = 0

    # Accumulators for lag-aware token agreement
    lag_totals: Dict[str, float] = {}

    for batch in dataloader:
        eeg = batch['eeg'].to(device)
        fnirs = batch['fnirs'].to(device)
        outputs = model(eeg, fnirs)
        total_batches += 1

        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim == 0:
                totals[f'val_{key}'] = totals.get(f'val_{key}', 0.0) + tensor_to_float(value)

        # Lag-aware token agreement (no gradient needed)
        eeg_idx = outputs.get('eeg_indices')
        fnirs_idx = outputs.get('fnirs_indices')
        if eeg_idx is not None and fnirs_idx is not None and hasattr(model, 'lag_token_agreement'):
            lag_metrics = model.lag_token_agreement(eeg_idx, fnirs_idx, lags=lag_set)
            for k, v in lag_metrics.items():
                lag_totals[f'val_{k}'] = lag_totals.get(f'val_{k}', 0.0) + float(v)

    result = {key: value / max(total_batches, 1) for key, value in totals.items()}

    # Average lag metrics (best_lag is averaged as float, acceptable approximation)
    for k, v in lag_totals.items():
        result[k] = v / max(total_batches, 1)

    return result


def load_checkpoint(path: Path, model, optimizer, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


# ---------------------------------------------------------------------------
# Key-remapping helpers for warm-start from single-modality checkpoints
# ---------------------------------------------------------------------------

# Map single-modality LaBraM module names to the EEG-branch names used in
# SharedLaBraMVQNSP.  The same table is used for the fNIRS branch with a
# different prefix.
_SINGLE_MODAL_TO_BRANCH_PREFIXES = [
    ("patch_embed.", "patch_embed."),
    ("encoder.", "encoder."),
    ("decode_input_proj.", "decode_input_proj."),
    ("decoder.", "decoder."),
    ("amplitude_head.", "amplitude_head."),
    ("phase_head.", "phase_head."),
]


def _remap_keys(state_dict: Dict[str, torch.Tensor], branch: str) -> Dict[str, torch.Tensor]:
    """Re-prefix a single-modality state-dict to the shared model's branch namespace.

    Args:
        state_dict: ``model_state_dict`` from a single-modality checkpoint.
        branch: ``"eeg"`` or ``"fnirs"``.

    Returns:
        New state dict with keys re-mapped to ``{branch}_{module}.{rest}``.
        Keys that do not match any known module prefix (e.g. ``quantizer.*``)
        are silently dropped because the shared quantizer is trained fresh.
    """
    remapped: Dict[str, torch.Tensor] = {}
    for src_prefix, dst_suffix in _SINGLE_MODAL_TO_BRANCH_PREFIXES:
        dst_prefix = f"{branch}_{dst_suffix}"
        for key, val in state_dict.items():
            if key.startswith(src_prefix):
                new_key = dst_prefix + key[len(src_prefix):]
                remapped[new_key] = val
    return remapped


def load_branch_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    branch: str,
    device: torch.device,
) -> None:
    """Warm-start one branch of a SharedLaBraMVQNSP from a single-modality checkpoint.

    Args:
        model: The shared tokenizer whose branch should be initialised.
        checkpoint_path: Path to the single-modality ``.pt`` checkpoint.
        branch: ``"eeg"`` or ``"fnirs"``.
        device: Target device.
    """
    if branch not in ("eeg", "fnirs"):
        raise ValueError(f"branch must be 'eeg' or 'fnirs', got '{branch}'")

    raw = torch.load(checkpoint_path, map_location=device)
    src_sd = raw.get('model_state_dict', raw)

    remapped = _remap_keys(src_sd, branch)
    if not remapped:
        print(f"  [warn] No keys remapped for branch '{branch}' from {checkpoint_path}")
        return

    current_sd = model.state_dict()
    matched, skipped_shape, skipped_missing = 0, 0, 0
    for key, val in remapped.items():
        if key not in current_sd:
            skipped_missing += 1
            continue
        if current_sd[key].shape != val.shape:
            skipped_shape += 1
            print(f"  [warn] Shape mismatch for '{key}': "
                  f"model={current_sd[key].shape}, ckpt={val.shape} — skipped")
            continue
        current_sd[key].copy_(val)
        matched += 1

    model.load_state_dict(current_sd)
    print(f"  Warm-started '{branch}' branch: "
          f"{matched} tensors loaded, "
          f"{skipped_shape} shape mismatches, "
          f"{skipped_missing} keys not in model.")


# ---------------------------------------------------------------------------
# Alignment warmup scheduler
# ---------------------------------------------------------------------------

def compute_alignment_scale(epoch: int, warmup_cfg: Dict[str, Any]) -> float:
    """Return a [0, 1] scale factor for alignment losses at the given epoch.

    The schedule is a linear ramp from ``start_scale`` to 1.0 over
    ``ramp_epochs`` epochs, beginning at ``start_epoch``.

    Args:
        epoch: Current training epoch (1-indexed).
        warmup_cfg: Dict from ``training.alignment_warmup`` in the YAML config.
            Expected keys (all optional):
            - ``enabled`` (bool, default True)
            - ``start_epoch`` (int, default 1) – epoch at which ramp begins
            - ``ramp_epochs`` (int, default 20) – number of epochs for ramp
            - ``start_scale`` (float, default 0.0) – initial scale

    Returns:
        Float in [0, 1].
    """
    if not warmup_cfg.get('enabled', True):
        return 1.0
    start_epoch = int(warmup_cfg.get('start_epoch', 1))
    ramp_epochs = int(warmup_cfg.get('ramp_epochs', 20))
    start_scale = float(warmup_cfg.get('start_scale', 0.0))

    if epoch < start_epoch:
        return start_scale
    elapsed = epoch - start_epoch
    if ramp_epochs <= 0 or elapsed >= ramp_epochs:
        return 1.0
    return start_scale + (1.0 - start_scale) * (elapsed / ramp_epochs)


def main():
    parser = argparse.ArgumentParser(description="Train shared EEG+fNIRS tokenizer")
    parser.add_argument('--config', required=True, help='Config path relative to experiments/configs')
    parser.add_argument('--resume', default=None, help='Optional checkpoint path')
    parser.add_argument(
        '--init-eeg-checkpoint',
        default=None,
        help='Path to a single-modality EEG LaBraM checkpoint for warm-starting the EEG branch',
    )
    parser.add_argument(
        '--init-fnirs-checkpoint',
        default=None,
        help='Path to a single-modality fNIRS LaBraM checkpoint for warm-starting the fNIRS branch',
    )
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

        # ------------------------------------------------------------------
        # Warm-start: load encoder/decoder weights from single-modality runs
        # ------------------------------------------------------------------
        # CLI flags take priority over YAML config entries so that workflows
        # can override without editing the config file.
        warm_cfg = config.get('warm_start', {})
        eeg_ckpt_path = args.init_eeg_checkpoint or warm_cfg.get('eeg_checkpoint')
        fnirs_ckpt_path = args.init_fnirs_checkpoint or warm_cfg.get('fnirs_checkpoint')

        if eeg_ckpt_path:
            print(f"\nWarm-starting EEG branch from: {eeg_ckpt_path}")
            load_branch_weights(model, Path(eeg_ckpt_path), 'eeg', device)
        if fnirs_ckpt_path:
            print(f"Warm-starting fNIRS branch from: {fnirs_ckpt_path}")
            load_branch_weights(model, Path(fnirs_ckpt_path), 'fnirs', device)

        # ------------------------------------------------------------------
        # Alignment warmup config
        # ------------------------------------------------------------------
        align_warmup_cfg: Dict[str, Any] = config.get('training', {}).get('alignment_warmup', {})
        lag_set: list = config.get('validation', {}).get('lag_set', [0, 1, 2, 3, 4])

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
            # Apply alignment warmup scale before the forward pass
            align_scale = compute_alignment_scale(epoch, align_warmup_cfg)
            if hasattr(model, 'set_alignment_scale'):
                model.set_alignment_scale(align_scale)

            train_metrics = train_epoch(model, train_loader, optimizer, device, grad_clip)
            val_metrics = validate_epoch(model, val_loader, device, lag_set=lag_set)

            if epoch > warmup_epochs:
                scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            train_loss = train_metrics.get('loss', float('nan'))
            val_loss = val_metrics.get('val_loss', float('nan'))
            merged_metrics = {'lr': lr, 'alignment_scale': align_scale}
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
                        'alignment_scale': align_scale,
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

        test_metrics = validate_epoch(model, test_loader, device, lag_set=lag_set)
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

        print("\nTraining complete.")
        print(f"Best epoch: {best_epoch}")
        print(f"Final metrics saved to: {logger.run_dir / 'metrics.json'}")
    finally:
        tee_logger.close()


if __name__ == '__main__':
    main()
