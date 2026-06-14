#!/usr/bin/env python
"""Calibrate the EEG/fNIRS coupling tensor against frozen source tokens."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.scripts.train_source_observation_tokenizer import (
    apply_model_schedule_state,
    build_checkpoint_payload,
    create_multimodal_dataloaders,
    filter_numeric_scalars,
    finalize_training_run,
    iter_device_batches,
    maybe_compile_model_forward,
    setup_device,
    setup_torch_performance,
    shutdown_dataloader_workers,
    write_run_manifest,
)
from src.tokenizers import create_tokenizer
from src.utils import (
    ExperimentLogger,
    load_training_checkpoint,
    require_standard_training_launcher,
    setup_logging,
    write_json,
)
from src.visualization import TensorBoardLogger


COUPLING_COMPONENT_KEYS = {
    'pair_likelihood': 'source_coupling_pair_likelihood_loss',
    'lag_focus': 'source_coupling_lag_focus_loss',
    'smoothness': 'source_coupling_smoothness_loss',
    'lag_evidence': 'source_coupling_lag_evidence_loss',
}


def configure_frozen_token_calibration(model, *, reset_coupling: bool, reset_std: float = 0.02):
    """Freeze the tokenizer and leave only the coupling tensor trainable."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    coupling_logits = getattr(model, 'coupling_logits', None)
    if not isinstance(coupling_logits, torch.nn.Parameter):
        raise ValueError('Frozen-token calibration requires model.coupling_logits')

    if reset_coupling:
        with torch.no_grad():
            torch.nn.init.trunc_normal_(coupling_logits, std=float(reset_std))
    coupling_logits.requires_grad_(True)
    model.eval()

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable != ['coupling_logits']:
        raise RuntimeError(f'Expected only coupling_logits to be trainable, got {trainable}')
    return coupling_logits


def compute_calibration_objective(
    outputs: Dict[str, torch.Tensor],
    objective_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    weighted: Dict[str, torch.Tensor] = {}
    objective: Optional[torch.Tensor] = None
    for component, output_key in COUPLING_COMPONENT_KEYS.items():
        weight = float(objective_weights.get(component, 0.0))
        if weight <= 0.0:
            continue
        value = outputs.get(output_key)
        if not torch.is_tensor(value) or value.ndim != 0:
            raise ValueError(f'Missing scalar calibration component: {output_key}')
        weighted_value = weight * value
        weighted[f'weighted_{component}_loss'] = weighted_value
        objective = weighted_value if objective is None else objective + weighted_value

    if objective is None:
        raise ValueError('Calibration objective must enable at least one coupling component')
    return objective, weighted


def run_calibration_epoch(
    model,
    dataloader,
    config: Dict[str, Any],
    device: torch.device,
    *,
    optimizer=None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    calibration_cfg = config['training']['coupling_calibration']
    objective_weights = calibration_cfg.get('objective', {'pair_likelihood': 1.0})
    prefetch = bool(config.get('training', {}).get('performance', {}).get('cuda_prefetch', False))
    totals: Dict[str, float] = {}
    count = 0

    grad_context = torch.enable_grad if optimizer is not None else torch.no_grad
    with grad_context():
        for batch_index, (eeg, fnirs, targets) in enumerate(
            iter_device_batches(dataloader, device, prefetch=prefetch),
            start=1,
        ):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(eeg, fnirs, targets=targets)
            objective, weighted = compute_calibration_objective(outputs, objective_weights)
            if optimizer is not None:
                objective.backward()
                optimizer.step()

            scalar_values = {
                'calibration_loss': objective.detach(),
                **{
                    key: outputs[key].detach()
                    for key in COUPLING_COMPONENT_KEYS.values()
                    if torch.is_tensor(outputs.get(key)) and outputs[key].ndim == 0
                },
                **{key: value.detach() for key, value in weighted.items()},
            }
            for key, value in scalar_values.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
            count += 1
            if max_batches is not None and batch_index >= max_batches:
                break

    if count == 0:
        raise RuntimeError('Calibration dataloader yielded no batches')
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description='Calibrate coupling tensor with frozen source tokens')
    parser.add_argument('--config', required=True, help='Config path relative to experiments/configs')
    parser.add_argument('--checkpoint', default=None, help='Override source tokenizer checkpoint')
    parser.add_argument('--run-name', default=None, help='Optional run directory name')
    parser.add_argument('--skip-post-analysis', action='store_true')
    args = parser.parse_args()

    require_standard_training_launcher('source-observation-coupling-calibration')
    logger = ExperimentLogger(config_path=args.config, run_name=args.run_name)
    config = logger.config
    tee_logger = setup_logging(logger.run_dir)
    dataloaders = None
    tb_logger = None

    try:
        calibration_cfg = config.get('training', {}).get('coupling_calibration', {})
        if not calibration_cfg.get('enabled', False):
            raise ValueError('training.coupling_calibration.enabled must be true')
        checkpoint_path = Path(args.checkpoint or calibration_cfg.get('source_checkpoint', ''))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'Calibration source checkpoint not found: {checkpoint_path}')

        print('=' * 70)
        print('Frozen-token EEG/fNIRS Coupling Calibration')
        print('=' * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f'Run directory: {logger.run_dir}')
        print(f'Source checkpoint: {checkpoint_path}')

        device = setup_device(config)
        setup_torch_performance(config, device)
        seed = int(config.get('experiment', {}).get('seed', 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataloaders = create_multimodal_dataloaders(config)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        print(
            f"Trials: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
            f"test={len(test_loader.dataset)}"
        )

        model = create_tokenizer(config).to(device)
        checkpoint = load_training_checkpoint(checkpoint_path, model, device=device)
        apply_model_schedule_state(model, checkpoint.get('schedule_state', {}))
        if hasattr(model, 'set_alignment_scale'):
            model.set_alignment_scale(1.0)
        coupling_logits = configure_frozen_token_calibration(
            model,
            reset_coupling=bool(calibration_cfg.get('reset_coupling', False)),
            reset_std=float(calibration_cfg.get('reset_std', 0.02)),
        )
        model = maybe_compile_model_forward(model, config, device)

        analysis_type = getattr(model, 'get_analysis_type', lambda: 'source_observation_alignment')()
        write_run_manifest(
            logger=logger,
            config=config,
            config_path=args.config,
            analysis_type=analysis_type,
        )
        write_json(
            logger.run_dir / 'calibration_manifest.json',
            {
                'schema_version': 'source_observation_coupling_calibration_v1',
                'source_checkpoint': str(checkpoint_path),
                'source_checkpoint_epoch': int(checkpoint.get('epoch', -1)),
                'reset_coupling': bool(calibration_cfg.get('reset_coupling', False)),
                'objective': calibration_cfg.get('objective', {'pair_likelihood': 1.0}),
                'trainable_parameters': ['coupling_logits'],
                'coupling_shape': list(coupling_logits.shape),
            },
        )

        learning_rate = float(calibration_cfg.get('learning_rate', 1e-3))
        min_lr = float(calibration_cfg.get('min_lr', learning_rate * 0.1))
        weight_decay = float(calibration_cfg.get('weight_decay', 0.0))
        total_epochs = int(calibration_cfg.get('epochs', 60))
        optimizer = torch.optim.Adam([coupling_logits], lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1), eta_min=min_lr)

        tb_cfg = config.get('logging', {}).get('tensorboard', {})
        if tb_cfg.get('enabled', True):
            tb_logger = TensorBoardLogger(run_dir=logger.run_dir, log_subdir=tb_cfg.get('subdir', 'tensorboard'))

        validation_interval = max(int(calibration_cfg.get('validation_interval_epochs', 1)), 1)
        save_every = max(int(calibration_cfg.get('save_every_epochs', 10)), 1)
        patience = max(int(calibration_cfg.get('patience', 15)), 1)
        max_train_batches = calibration_cfg.get('max_train_batches')
        max_val_batches = calibration_cfg.get('max_val_batches')
        max_train_batches = int(max_train_batches) if max_train_batches else None
        max_val_batches = int(max_val_batches) if max_val_batches else None
        best_monitor = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, total_epochs + 1):
            started = time.perf_counter()
            train_metrics = run_calibration_epoch(
                model,
                train_loader,
                config,
                device,
                optimizer=optimizer,
                max_batches=max_train_batches,
            )
            train_seconds = time.perf_counter() - started

            run_validation = epoch % validation_interval == 0 or epoch == total_epochs
            if run_validation:
                val_metrics = run_calibration_epoch(
                    model,
                    val_loader,
                    config,
                    device,
                    max_batches=max_val_batches,
                )
                val_loss = val_metrics['calibration_loss']
            else:
                val_metrics = {}
                val_loss = None

            scheduler.step()
            improved = val_loss is not None and val_loss < best_monitor
            if improved:
                best_monitor = float(val_loss)
                best_epoch = epoch
                epochs_without_improvement = 0
            elif run_validation:
                epochs_without_improvement += 1

            logger.log_epoch(
                epoch=epoch,
                train_loss=train_metrics['calibration_loss'],
                val_loss=val_loss,
                loss_breakdown={key: value for key, value in train_metrics.items() if key != 'calibration_loss'},
                metrics={
                    'lr': optimizer.param_groups[0]['lr'],
                    'train_epoch_seconds': train_seconds,
                    'frozen_token_calibration': 1.0,
                    **{f'val_{key}': value for key, value in val_metrics.items() if key != 'calibration_loss'},
                },
            )
            if tb_logger is not None:
                tb_logger.log_scalars('calibration/train', filter_numeric_scalars(train_metrics), epoch)
                if val_metrics:
                    tb_logger.log_scalars('calibration/val', filter_numeric_scalars(val_metrics), epoch)
                tb_logger.log_learning_rate(optimizer.param_groups[0]['lr'], epoch)
                tb_logger.flush()

            payload = build_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                train_loss=train_metrics['calibration_loss'],
                val_loss=float(val_loss) if val_loss is not None else float('nan'),
                monitor_metric='val_calibration_loss',
                monitor_value=float(val_loss) if val_loss is not None else float('nan'),
                best_epoch=best_epoch,
                best_monitor=best_monitor,
                alignment_scale=1.0,
                is_best=improved,
            )
            logger.save_checkpoint(
                payload,
                epoch=epoch,
                is_best=improved,
                keep_epoch_copy=(epoch % save_every == 0 or epoch == total_epochs),
            )
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch}; best epoch={best_epoch}')
                break

        final_metrics = finalize_training_run(
            logger=logger,
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            best_epoch=best_epoch,
            best_monitor=best_monitor,
            skip_post_analysis=args.skip_post_analysis,
        )
        print(json.dumps({'run_dir': str(logger.run_dir), 'final_metrics': final_metrics}, indent=2))
    finally:
        if tb_logger is not None:
            tb_logger.close()
        shutdown_dataloader_workers(dataloaders)
        tee_logger.close()


if __name__ == '__main__':
    main()
