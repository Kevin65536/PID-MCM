#!/usr/bin/env python
"""Train the multimodal EEG+fNIRS tokenizer mainline."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import create_configured_multimodal_dataloaders
from src.losses import compute_multi_stft_loss
from src.metrics import compute_spectral_mse
from src.tokenizers import create_tokenizer
from src.utils import (
    ExperimentLogger,
    load_checkpoint_file,
    load_training_checkpoint,
    require_standard_training_launcher,
    setup_logging,
    write_json,
)
from src.visualization import TensorBoardLogger, generate_tokenizer_analysis_suite
from src.visualization.gradient_diagnostics import (
    compute_component_group_attribution,
    summarize_total_gradient_groups,
)


def setup_device(config: dict) -> torch.device:
    requested = config.get('experiment', {}).get('device', 'cuda')
    if requested.startswith('cuda') and torch.cuda.is_available():
        if requested == 'cuda':
            return torch.device('cuda')
        return torch.device(requested)
    if requested == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_torch_performance(config: dict, device: torch.device) -> None:
    perf_cfg = config.get('training', {}).get('performance', {})
    if device.type != 'cuda':
        return

    if bool(perf_cfg.get('tf32', True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision(str(perf_cfg.get('matmul_precision', 'high')))
        except AttributeError:
            pass

    if bool(perf_cfg.get('cudnn_benchmark', True)):
        torch.backends.cudnn.benchmark = True


def shutdown_dataloader_workers(dataloaders: Any) -> None:
    """Best-effort cleanup for DataLoaders that used persistent workers."""
    if dataloaders is None:
        return
    if isinstance(dataloaders, dict):
        for dataloader in dataloaders.values():
            shutdown_dataloader_workers(dataloader)
        return
    iterator = getattr(dataloaders, '_iterator', None)
    if iterator is not None and hasattr(iterator, '_shutdown_workers'):
        try:
            iterator._shutdown_workers()
        except Exception as exc:  # pragma: no cover - cleanup should not mask analysis results.
            print(f"[Warning] Failed to shutdown DataLoader workers cleanly: {exc}")
        finally:
            try:
                dataloaders._iterator = None
            except Exception:
                pass


def maybe_compile_model_forward(model, config: dict, device: torch.device):
    perf_cfg = config.get('training', {}).get('performance', {})
    compile_cfg = perf_cfg.get('compile', perf_cfg.get('torch_compile', {}))
    if isinstance(compile_cfg, bool):
        compile_cfg = {'enabled': compile_cfg}
    if not isinstance(compile_cfg, dict) or not bool(compile_cfg.get('enabled', False)):
        return model

    if not hasattr(torch, 'compile'):
        message = 'torch.compile requested, but this PyTorch build does not expose torch.compile'
        if bool(compile_cfg.get('required', False)):
            raise RuntimeError(message)
        print(f'[Warning] {message}; continuing without compile.')
        return model

    if device.type != 'cuda' and not bool(compile_cfg.get('allow_cpu', False)):
        print('[Warning] torch.compile requested on a non-CUDA device; continuing without compile.')
        return model

    scope = str(compile_cfg.get('scope', 'forward'))
    compile_kwargs: Dict[str, Any] = {
        'mode': str(compile_cfg.get('mode', 'reduce-overhead')),
        'fullgraph': bool(compile_cfg.get('fullgraph', False)),
        'dynamic': bool(compile_cfg.get('dynamic', False)),
    }
    backend = compile_cfg.get('backend')
    if backend:
        compile_kwargs['backend'] = str(backend)

    try:
        import torch._dynamo as torch_dynamo

        torch_dynamo.config.suppress_errors = bool(compile_cfg.get('suppress_errors', True))
        torch_dynamo.config.capture_scalar_outputs = bool(compile_cfg.get('capture_scalar_outputs', False))
    except Exception as exc:
        print(f'[Warning] Unable to configure torch._dynamo options ({exc}).')

    try:
        if scope in {'encode_modalities', 'encoders'} and hasattr(model, 'encode_modalities'):
            model._gradient_attribution_uncompiled_encode_modalities = model.encode_modalities
            model.encode_modalities = torch.compile(model.encode_modalities, **compile_kwargs)
        elif scope == 'forward':
            model._gradient_attribution_uncompiled_forward = model.forward
            model.forward = torch.compile(model.forward, **compile_kwargs)
        else:
            raise ValueError("training.performance.compile.scope must be 'forward' or 'encode_modalities'")
    except Exception as exc:
        if bool(compile_cfg.get('required', False)):
            raise
        print(f'[Warning] torch.compile failed during setup ({exc}); continuing without compile.')
        return model

    print(
        f'torch.compile enabled for model.{scope} '
        f"(mode={compile_kwargs['mode']}, fullgraph={compile_kwargs['fullgraph']}, "
        f"dynamic={compile_kwargs['dynamic']})"
    )
    return model


def forward_for_gradient_attribution(model, eeg, fnirs, targets):
    compiled_encode = None
    compiled_forward = None
    uncompiled_encode = getattr(model, '_gradient_attribution_uncompiled_encode_modalities', None)
    uncompiled_forward = getattr(model, '_gradient_attribution_uncompiled_forward', None)

    if uncompiled_encode is not None:
        compiled_encode = model.encode_modalities
        model.encode_modalities = uncompiled_encode
    if uncompiled_forward is not None:
        compiled_forward = model.forward
        model.forward = uncompiled_forward
    try:
        return model(eeg, fnirs, targets=targets)
    finally:
        if compiled_forward is not None:
            model.forward = compiled_forward
        if compiled_encode is not None:
            model.encode_modalities = compiled_encode


def create_multimodal_dataloaders(config: dict):
    return create_configured_multimodal_dataloaders(config)


def move_source_observation_targets_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    targets = batch.get('targets')
    if not isinstance(targets, dict):
        return None
    moved: Dict[str, torch.Tensor] = {}
    for key, value in targets.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = torch.as_tensor(value, device=device)
    return moved


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    eeg = batch['eeg'].to(device, non_blocking=True)
    fnirs = batch['fnirs'].to(device, non_blocking=True)
    targets = move_source_observation_targets_to_device(batch, device)
    return eeg, fnirs, targets


def iter_device_batches(
    dataloader,
    device: torch.device,
    *,
    prefetch: bool,
):
    if device.type != 'cuda' or not prefetch:
        for batch in dataloader:
            yield move_batch_to_device(batch, device)
        return

    stream = torch.cuda.Stream(device=device)
    iterator = iter(dataloader)
    next_batch = None

    def preload_next() -> bool:
        nonlocal next_batch
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            next_batch = None
            return False
        with torch.cuda.stream(stream):
            next_batch = move_batch_to_device(cpu_batch, device)
        return True

    if not preload_next():
        return

    while next_batch is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        batch = next_batch
        has_next = preload_next()
        yield batch
        if not has_next:
            break


def inject_channel_names_into_config(config: Dict[str, Any], dataloaders: Dict[str, Any]) -> None:
    reference_dataset = None
    for split_name in ('train', 'val', 'test'):
        loader = dataloaders.get(split_name)
        if loader is None:
            continue
        dataset = getattr(loader, 'dataset', None)
        if dataset is None:
            continue
        if hasattr(dataset, 'get_eeg_channel_names') and hasattr(dataset, 'get_fnirs_channel_names'):
            reference_dataset = dataset
            break
    if reference_dataset is None:
        return

    data_cfg = config.setdefault('data', {})
    eeg_channel_names = list(reference_dataset.get_eeg_channel_names())
    fnirs_channel_names = list(reference_dataset.get_fnirs_channel_names())
    data_cfg['eeg_channel_names'] = eeg_channel_names
    data_cfg['fnirs_channel_names'] = fnirs_channel_names

    model_cfg = config.setdefault('model', {})
    eeg_cfg = model_cfg.setdefault('eeg', {})
    fnirs_cfg = model_cfg.setdefault('fnirs', {})
    eeg_cfg['channels'] = len(eeg_channel_names)
    fnirs_cfg['channels'] = len(fnirs_channel_names)


def tensor_to_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


def filter_numeric_scalars(
    values: Dict[str, Any],
    strip_prefix: Optional[str] = None,
) -> Dict[str, float]:
    scalars: Dict[str, float] = {}
    for key, value in values.items():
        if value is None or isinstance(value, bool):
            continue
        try:
            numeric_value = tensor_to_float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(numeric_value) or np.isinf(numeric_value):
            continue
        tag = key
        if strip_prefix and tag.startswith(strip_prefix):
            tag = tag[len(strip_prefix):]
        scalars[tag] = numeric_value
    return scalars


def _positive_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _cuda_synchronize_if_needed(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _accumulate_metric(totals: Dict[str, Any], key: str, value: Any) -> None:
    if torch.is_tensor(value):
        if value.ndim != 0:
            return
        metric_value = value.detach()
        totals[key] = totals.get(key, metric_value.new_zeros(())) + metric_value
        return
    totals[key] = float(totals.get(key, 0.0)) + float(value)


def _finalize_metric_totals(totals: Dict[str, Any], total_batches: int) -> Dict[str, float]:
    denominator = max(total_batches, 1)
    averaged: Dict[str, float] = {}
    for key, value in totals.items():
        if torch.is_tensor(value):
            averaged[key] = float((value / denominator).item())
        else:
            averaged[key] = float(value) / denominator
    return averaged


def build_tensorboard_hparams(config: Dict[str, Any]) -> Dict[str, Any]:
    experiment_cfg = config.get('experiment', {})
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    loss_cfg = config.get('loss', {})
    return {
        'experiment_name': experiment_cfg.get('name'),
        'dataset': data_cfg.get('dataset'),
        'task': data_cfg.get('task'),
        'batch_size': training_cfg.get('batch_size'),
        'learning_rate': training_cfg.get('learning_rate'),
        'weight_decay': training_cfg.get('weight_decay'),
        'epochs': training_cfg.get('epochs'),
        'source_codebook_size': model_cfg.get('source', {}).get('codebook_size'),
        'eeg_observation_codebook_size': model_cfg.get('eeg_observation', {}).get('codebook_size'),
        'fnirs_observation_codebook_size': model_cfg.get('fnirs_observation', {}).get('codebook_size'),
        'source_target_weight': loss_cfg.get('source_target', {}).get('weight', 0.0),
        'observation_target_weight': loss_cfg.get('observation_target', {}).get('weight', 0.0),
        'coupling_weight': loss_cfg.get('coupling', {}).get('weight'),
        'codebook_balance_weight': loss_cfg.get('codebook', {}).get('balance_weight'),
        'spectral_weight': loss_cfg.get('spectral', {}).get('weight', 0.0),
    }


def _resolve_gradient_component_specs(model) -> Dict[str, float]:
    component_specs = {
        'eeg_rec_loss': 1.0,
        'fnirs_rec_loss': 1.0,
        'vq_loss': 1.0,
    }
    if hasattr(model, 'get_gradient_component_weights'):
        component_specs.update(getattr(model, 'get_gradient_component_weights')())
    return component_specs


def _pairwise_metric_name(left: str, right: str) -> str:
    return f'{left}__vs__{right}'


GRADIENT_DASHBOARD_ARRAY_KEYS = (
    'cosine_matrix',
    'component_norms',
    'component_shares',
    'component_group_norms',
    'component_group_shares',
    'group_component_shares',
    'group_total_norms',
    'group_total_shares',
)


def _initialize_gradient_dashboard(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    dashboard: Dict[str, Any] = {
        'component_names': list(artifacts['component_names']),
    }
    if 'group_names' in artifacts:
        dashboard['group_names'] = list(artifacts['group_names'])
        dashboard['group_labels'] = list(artifacts.get('group_labels', artifacts['group_names']))
    if 'parameter_group_counts' in artifacts:
        dashboard['parameter_group_counts'] = list(artifacts['parameter_group_counts'])
    for key in GRADIENT_DASHBOARD_ARRAY_KEYS:
        if key in artifacts:
            dashboard[key] = np.asarray(artifacts[key], dtype=np.float32)
    if 'group_abs_grad_quantiles' in artifacts:
        dashboard['group_abs_grad_quantiles'] = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in artifacts['group_abs_grad_quantiles'].items()
        }
    return dashboard


def _accumulate_gradient_dashboard(
    aggregate: Optional[Dict[str, Any]],
    artifacts: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not artifacts:
        return aggregate, False
    if aggregate is None:
        return _initialize_gradient_dashboard(artifacts), True

    if aggregate['component_names'] != list(artifacts.get('component_names', [])):
        return aggregate, False
    if aggregate.get('group_names') != artifacts.get('group_names'):
        return aggregate, False

    for key in GRADIENT_DASHBOARD_ARRAY_KEYS:
        if key in aggregate and key in artifacts:
            aggregate[key] += np.asarray(artifacts[key], dtype=np.float32)
        elif key in aggregate or key in artifacts:
            return aggregate, False

    if 'group_abs_grad_quantiles' in aggregate and 'group_abs_grad_quantiles' in artifacts:
        if set(aggregate['group_abs_grad_quantiles']) != set(artifacts['group_abs_grad_quantiles']):
            return aggregate, False
        for key, values in artifacts['group_abs_grad_quantiles'].items():
            aggregate['group_abs_grad_quantiles'][key] += np.asarray(values, dtype=np.float32)
    elif 'group_abs_grad_quantiles' in aggregate or 'group_abs_grad_quantiles' in artifacts:
        return aggregate, False

    return aggregate, True


def _finalize_gradient_dashboard(
    aggregate: Optional[Dict[str, Any]],
    batches: int,
) -> Optional[Dict[str, Any]]:
    if aggregate is None or batches <= 0:
        return None

    finalized: Dict[str, Any] = {
        'component_names': list(aggregate['component_names']),
    }
    if 'group_names' in aggregate:
        finalized['group_names'] = list(aggregate['group_names'])
        finalized['group_labels'] = list(aggregate.get('group_labels', aggregate['group_names']))
    if 'parameter_group_counts' in aggregate:
        finalized['parameter_group_counts'] = list(aggregate['parameter_group_counts'])
    for key in GRADIENT_DASHBOARD_ARRAY_KEYS:
        if key in aggregate:
            finalized[key] = (aggregate[key] / batches).tolist()
    if 'group_abs_grad_quantiles' in aggregate:
        finalized['group_abs_grad_quantiles'] = {
            key: (values / batches).tolist()
            for key, values in aggregate['group_abs_grad_quantiles'].items()
        }
    return finalized


def compute_gradient_attribution(
    model,
    outputs: Dict[str, Any],
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not named_params:
        return {}, None
    param_names = [name for name, _ in named_params]
    params = [param for _, param in named_params]

    component_specs = _resolve_gradient_component_specs(model)
    component_entries: List[Dict[str, Any]] = []
    component_values: Dict[str, float] = {}

    for name, weight in component_specs.items():
        term = outputs.get(name)
        if term is None or not torch.is_tensor(term) or term.ndim != 0 or abs(weight) <= 0.0:
            continue
        grads = torch.autograd.grad(weight * term, params, retain_graph=True, allow_unused=True)
        grad_tensors: List[Optional[torch.Tensor]] = []
        squared_norm = None
        for grad in grads:
            if grad is None:
                grad_tensors.append(None)
                continue
            grad_value = grad.detach()
            grad_tensors.append(grad_value)
            grad_sq = torch.sum(grad_value * grad_value)
            squared_norm = grad_sq if squared_norm is None else squared_norm + grad_sq
        if squared_norm is None:
            continue
        component_entries.append({
            'name': name,
            'norm': float(torch.sqrt(squared_norm).item()),
            'grads': grad_tensors,
        })
        component_values[f'weighted_term_{name}'] = float((weight * term.detach()).item())

    if not component_entries:
        return component_values, None

    attribution_metrics: Dict[str, float] = {}
    component_names = [entry['name'] for entry in component_entries]
    component_norms = np.array([entry['norm'] for entry in component_entries], dtype=np.float32)
    component_shares = component_norms / max(float(component_norms.sum()), 1e-12)

    for entry, share in zip(component_entries, component_shares):
        name = entry['name']
        attribution_metrics[f'grad_norm_{name}'] = entry['norm']
        attribution_metrics[f'grad_share_{name}'] = float(share)

    cosine_matrix = np.eye(len(component_entries), dtype=np.float32)
    pairwise_cosines: List[float] = []
    conflict_pairs = 0
    for i, left_entry in enumerate(component_entries):
        for j in range(i + 1, len(component_entries)):
            right_entry = component_entries[j]
            dot_product = 0.0
            for left_grad, right_grad in zip(left_entry['grads'], right_entry['grads']):
                if left_grad is None or right_grad is None:
                    continue
                dot_product += float(torch.sum(left_grad * right_grad).item())
            denominator = max(left_entry['norm'] * right_entry['norm'], 1e-12)
            cosine = float(np.clip(dot_product / denominator, -1.0, 1.0))
            cosine_matrix[i, j] = cosine
            cosine_matrix[j, i] = cosine
            pairwise_cosines.append(cosine)
            if cosine < 0.0:
                conflict_pairs += 1
            pair_name = _pairwise_metric_name(left_entry['name'], right_entry['name'])
            attribution_metrics[f'grad_cosine_{pair_name}'] = cosine

    if pairwise_cosines:
        attribution_metrics['grad_mean_pairwise_cosine'] = float(np.mean(pairwise_cosines))
        attribution_metrics['grad_min_pairwise_cosine'] = float(np.min(pairwise_cosines))
        attribution_metrics['grad_conflict_rate'] = conflict_pairs / len(pairwise_cosines)

    attribution_metrics['grad_component_count'] = float(len(component_entries))
    attribution_metrics.update(component_values)

    gradient_artifacts: Dict[str, Any] = {
        'component_names': component_names,
        'component_norms': component_norms.tolist(),
        'component_shares': component_shares.tolist(),
        'cosine_matrix': cosine_matrix.tolist(),
    }
    group_specs = None
    if hasattr(model, 'get_gradient_parameter_group_specs'):
        group_specs = getattr(model, 'get_gradient_parameter_group_specs')()
    group_artifacts = compute_component_group_attribution(
        component_entries,
        param_names,
        group_specs=group_specs,
    )
    if group_artifacts is not None:
        gradient_artifacts.update(group_artifacts)
        group_names = group_artifacts['group_names']
        group_norms = np.asarray(group_artifacts['component_group_norms'], dtype=np.float32)
        group_shares = np.asarray(group_artifacts['component_group_shares'], dtype=np.float32)
        for component_index, component_name in enumerate(component_names):
            for group_index, group_name in enumerate(group_names):
                attribution_metrics[
                    f'grad_group_norm_{component_name}__{group_name}'
                ] = float(group_norms[component_index, group_index])
                attribution_metrics[
                    f'grad_group_share_{component_name}__{group_name}'
                ] = float(group_shares[component_index, group_index])

    return attribution_metrics, gradient_artifacts


def _flatten_signal_for_spectral(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.reshape(-1, x.shape[-1])
    return x


def _compute_spectral_loss(x: torch.Tensor, x_rec: torch.Tensor, spectral_cfg: Dict[str, Any]) -> torch.Tensor:
    x = _flatten_signal_for_spectral(x)
    x_rec = _flatten_signal_for_spectral(x_rec)
    spectral_type = spectral_cfg.get('type', 'multi_stft')
    if spectral_type == 'multi_stft':
        return compute_multi_stft_loss(
            x,
            x_rec,
            fft_sizes=spectral_cfg.get('fft_sizes', [64, 128, 256]),
            hop_sizes=spectral_cfg.get('hop_sizes'),
            win_sizes=spectral_cfg.get('win_sizes'),
        )
    if spectral_type == 'fft_mse':
        return compute_spectral_mse(
            x,
            x_rec,
            n_fft=int(spectral_cfg.get('n_fft', 256)),
        )
    raise ValueError(f"Unsupported spectral loss type: {spectral_type}")


def compute_multimodal_aux_losses(
    config: Dict[str, Any],
    eeg: torch.Tensor,
    fnirs: torch.Tensor,
    outputs: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    spectral_cfg = config.get('loss', {}).get('spectral', {})
    spectral_weight = float(spectral_cfg.get('weight', 0.0))
    zero = eeg.new_tensor(0.0)
    aux_loss = zero
    scalar_metrics: Dict[str, Any] = {}
    tensor_metrics: Dict[str, torch.Tensor] = {}

    if not spectral_cfg.get('enabled', False) or spectral_weight <= 0.0:
        return aux_loss, scalar_metrics, tensor_metrics

    eeg_reconstructed = outputs.get('eeg_reconstructed')
    fnirs_reconstructed = outputs.get('fnirs_reconstructed')
    if not torch.is_tensor(eeg_reconstructed) or not torch.is_tensor(fnirs_reconstructed):
        return aux_loss, scalar_metrics, tensor_metrics

    eeg_spectral_loss = _compute_spectral_loss(eeg, eeg_reconstructed, spectral_cfg)
    fnirs_spectral_loss = _compute_spectral_loss(fnirs, fnirs_reconstructed, spectral_cfg)
    spectral_loss = 0.5 * (eeg_spectral_loss + fnirs_spectral_loss)
    aux_loss = aux_loss + spectral_weight * spectral_loss

    scalar_metrics['eeg_spectral_loss'] = eeg_spectral_loss.detach()
    scalar_metrics['fnirs_spectral_loss'] = fnirs_spectral_loss.detach()
    scalar_metrics['spectral_loss'] = spectral_loss.detach()
    scalar_metrics['aux_loss'] = aux_loss.detach()
    tensor_metrics['spectral_loss'] = spectral_loss
    tensor_metrics['aux_loss'] = aux_loss
    return aux_loss, scalar_metrics, tensor_metrics


def train_epoch(
    model,
    dataloader,
    optimizer,
    config: Dict[str, Any],
    device: torch.device,
    grad_clip: float,
    gradient_attribution_cfg: Optional[Dict[str, Any]] = None,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    model.train()
    totals: Dict[str, Any] = {}
    total_batches = 0
    grad_totals: Dict[str, float] = {}
    grad_batches = 0
    grad_cfg = gradient_attribution_cfg or {'enabled': False, 'max_batches': 1}
    gradient_dashboard: Optional[Dict[str, Any]] = None
    gradient_dashboard_batches = 0
    prefetch_batches = bool(
        config.get('training', {}).get('performance', {}).get('cuda_prefetch', False)
    )

    for batch_index, (eeg, fnirs, targets) in enumerate(
        iter_device_batches(dataloader, device, prefetch=prefetch_batches),
        start=1,
    ):
        optimizer.zero_grad()
        batch_gradient_dashboard: Optional[Dict[str, Any]] = None

        if grad_cfg.get('enabled', False) and grad_batches < int(grad_cfg.get('max_batches', 1)):
            diagnostic_outputs = forward_for_gradient_attribution(model, eeg, fnirs, targets)
            _, _, diagnostic_aux_tensors = compute_multimodal_aux_losses(
                config,
                eeg,
                fnirs,
                diagnostic_outputs,
            )
            grad_outputs = dict(diagnostic_outputs)
            grad_outputs.update(diagnostic_aux_tensors)
            grad_metrics, grad_artifacts = compute_gradient_attribution(model, grad_outputs)
            if grad_metrics:
                grad_batches += 1
                for key, value in grad_metrics.items():
                    grad_totals[key] = grad_totals.get(key, 0.0) + value
            if grad_artifacts:
                batch_gradient_dashboard = grad_artifacts
            del diagnostic_outputs, diagnostic_aux_tensors, grad_outputs

        outputs = model(eeg, fnirs, targets=targets)
        aux_loss, aux_metrics, aux_tensors = compute_multimodal_aux_losses(config, eeg, fnirs, outputs)
        loss = outputs['loss'] + aux_loss

        loss.backward()

        if batch_gradient_dashboard is not None and 'group_names' in batch_gradient_dashboard:
            group_specs = None
            if hasattr(model, 'get_gradient_parameter_group_specs'):
                group_specs = getattr(model, 'get_gradient_parameter_group_specs')()
            total_gradient_groups = summarize_total_gradient_groups(model, group_specs=group_specs)
            if total_gradient_groups is not None and total_gradient_groups.get('group_names') == batch_gradient_dashboard.get('group_names'):
                batch_gradient_dashboard = {**batch_gradient_dashboard, **total_gradient_groups}

        gradient_dashboard, dashboard_added = _accumulate_gradient_dashboard(
            gradient_dashboard,
            batch_gradient_dashboard,
        )
        if dashboard_added:
            gradient_dashboard_batches += 1

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_batches += 1

        _accumulate_metric(totals, 'loss', loss.detach())
        for key, value in outputs.items():
            if key == 'loss':
                continue
            if torch.is_tensor(value) and value.ndim == 0:
                _accumulate_metric(totals, key, value)
        for key, value in aux_metrics.items():
            _accumulate_metric(totals, key, value)

        if max_batches is not None and batch_index >= max_batches:
            break

    averaged = _finalize_metric_totals(totals, total_batches)
    if grad_batches > 0:
        averaged.update({key: value / grad_batches for key, value in grad_totals.items()})
    if gradient_dashboard is not None and gradient_dashboard_batches > 0:
        gradient_dashboard = _finalize_gradient_dashboard(gradient_dashboard, gradient_dashboard_batches)
    return averaged, gradient_dashboard


@torch.no_grad()
def validate_epoch(
    model,
    dataloader,
    config: Dict[str, Any],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, Any] = {}
    total_batches = 0
    prefetch_batches = bool(
        config.get('training', {}).get('performance', {}).get('cuda_prefetch', False)
    )

    for batch_index, (eeg, fnirs, targets) in enumerate(
        iter_device_batches(dataloader, device, prefetch=prefetch_batches),
        start=1,
    ):
        outputs = model(eeg, fnirs, targets=targets)
        aux_loss, aux_metrics, _ = compute_multimodal_aux_losses(config, eeg, fnirs, outputs)
        total_batches += 1

        _accumulate_metric(totals, 'val_loss', (outputs['loss'] + aux_loss).detach())
        for key, value in outputs.items():
            if key == 'loss':
                continue
            if torch.is_tensor(value) and value.ndim == 0:
                _accumulate_metric(totals, f'val_{key}', value)
        for key, value in aux_metrics.items():
            _accumulate_metric(totals, f'val_{key}', value)

        if max_batches is not None and batch_index >= max_batches:
            break

    return _finalize_metric_totals(totals, total_batches)


def maybe_seed_best_checkpoint(
    logger: ExperimentLogger,
    resume_path: Path,
    checkpoint: Dict[str, Any],
):
    best_path = logger.checkpoints_dir / 'best_model.pt'
    if best_path.exists():
        return

    resume_epoch = int(checkpoint.get('epoch', -1))
    resume_best_epoch = checkpoint.get('best_epoch')
    resume_is_best = bool(checkpoint.get('is_best', False))
    if resume_best_epoch is None:
        return

    if resume_is_best or int(resume_best_epoch) == resume_epoch:
        shutil.copy2(resume_path, best_path)
        print(
            f"Seeded local best checkpoint from resume source: {resume_path} -> {best_path}"
        )
        return

    print(
        "[Warning] Resumed checkpoint is not itself the tracked best checkpoint and the current run "
        f"has no local best_model.pt. best_epoch={resume_best_epoch}, resumed_epoch={resume_epoch}. "
        "If no new best is found during continuation, finalization will not have an on-disk best checkpoint "
        "to reload automatically."
    )


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


def compute_schedule_value(epoch: int, schedule_cfg: Dict[str, Any], default_value: float) -> float:
    if not schedule_cfg.get('enabled', False):
        return float(default_value)

    start_epoch = int(schedule_cfg.get('start_epoch', 1))
    ramp_epochs = max(int(schedule_cfg.get('ramp_epochs', 1)), 1)
    start_scale = float(schedule_cfg.get('start_scale', default_value))
    end_scale = float(schedule_cfg.get('end_scale', default_value))
    if epoch < start_epoch:
        return start_scale
    if ramp_epochs == 1:
        return end_scale

    progress = min(max((epoch - start_epoch) / (ramp_epochs - 1), 0.0), 1.0)
    return start_scale + (end_scale - start_scale) * progress


def snapshot_model_schedule_state(model) -> Dict[str, float]:
    state: Dict[str, float] = {}
    if hasattr(model, 'get_alignment_scale'):
        state['alignment_scale'] = float(model.get_alignment_scale())
    if hasattr(model, 'get_source_target_scale'):
        state['source_target_scale'] = float(model.get_source_target_scale())
    if hasattr(model, 'get_observation_target_scale'):
        state['observation_target_scale'] = float(model.get_observation_target_scale())
    if hasattr(model, 'get_balance_scales'):
        state.update(getattr(model, 'get_balance_scales')())
    if hasattr(model, 'get_quantization_strength'):
        state['quantization_strength'] = float(model.get_quantization_strength())
    return state


def apply_model_schedule_state(model, schedule_state: Dict[str, Any]) -> None:
    if not schedule_state:
        return
    if 'alignment_scale' in schedule_state and hasattr(model, 'set_alignment_scale'):
        model.set_alignment_scale(float(schedule_state['alignment_scale']))
    if 'source_target_scale' in schedule_state and hasattr(model, 'set_source_target_scale'):
        model.set_source_target_scale(float(schedule_state['source_target_scale']))
    if 'observation_target_scale' in schedule_state and hasattr(model, 'set_observation_target_scale'):
        model.set_observation_target_scale(float(schedule_state['observation_target_scale']))
    if hasattr(model, 'set_balance_scales'):
        getattr(model, 'set_balance_scales')(
            source_scale=schedule_state.get('source_balance_scale'),
            observation_scale=schedule_state.get('observation_balance_scale'),
        )
    if 'quantization_strength' in schedule_state and hasattr(model, 'set_quantization_strength'):
        model.set_quantization_strength(float(schedule_state['quantization_strength']))


def apply_epoch_schedules(model, epoch: int, config: dict) -> Dict[str, float]:
    schedule_metrics: Dict[str, float] = {}

    alignment_scale = compute_alignment_scale(epoch, config)
    schedule_metrics['alignment_scale'] = alignment_scale
    if hasattr(model, 'set_alignment_scale'):
        model.set_alignment_scale(alignment_scale)

    source_target_cfg = config.get('loss', {}).get('source_target', {})
    source_target_warmup_cfg = config.get('training', {}).get('source_target_warmup')
    if source_target_warmup_cfg is None:
        legacy_warmup_epochs = int(
            source_target_cfg.get('warmup_epochs', source_target_cfg.get('source_target_warmup_epochs', 0))
        )
        source_target_warmup_cfg = {
            'enabled': legacy_warmup_epochs > 0,
            'start_epoch': 1,
            'ramp_epochs': max(legacy_warmup_epochs, 1),
            'start_scale': 0.0,
            'end_scale': 1.0,
        }
    source_target_scale = compute_schedule_value(epoch, source_target_warmup_cfg, 1.0)
    schedule_metrics['source_target_scale'] = source_target_scale
    if hasattr(model, 'set_source_target_scale'):
        model.set_source_target_scale(source_target_scale)

    observation_target_cfg = config.get('loss', {}).get('observation_target', {})
    observation_target_warmup_cfg = config.get('training', {}).get('observation_target_warmup')
    if observation_target_warmup_cfg is None:
        legacy_observation_warmup_epochs = int(
            observation_target_cfg.get('warmup_epochs', observation_target_cfg.get('observation_target_warmup_epochs', 0))
        )
        observation_target_warmup_cfg = {
            'enabled': legacy_observation_warmup_epochs > 0,
            'start_epoch': 1,
            'ramp_epochs': max(legacy_observation_warmup_epochs, 1),
            'start_scale': 0.0,
            'end_scale': 1.0,
        }
    observation_target_scale = compute_schedule_value(epoch, observation_target_warmup_cfg, 1.0)
    schedule_metrics['observation_target_scale'] = observation_target_scale
    if hasattr(model, 'set_observation_target_scale'):
        model.set_observation_target_scale(observation_target_scale)

    balance_warmup_cfg = config.get('training', {}).get('balance_warmup', {})
    source_default = float(getattr(model, 'default_source_balance_scale', 1.0))
    observation_default = float(getattr(model, 'default_observation_balance_scale', 1.0))
    source_balance_scale = compute_schedule_value(epoch, balance_warmup_cfg.get('source', {}), source_default)
    observation_balance_scale = compute_schedule_value(
        epoch,
        balance_warmup_cfg.get('observation', {}),
        observation_default,
    )
    schedule_metrics['source_balance_scale'] = source_balance_scale
    schedule_metrics['observation_balance_scale'] = observation_balance_scale
    if hasattr(model, 'set_balance_scales'):
        model.set_balance_scales(
            source_scale=source_balance_scale,
            observation_scale=observation_balance_scale,
        )

    quantization_warmup_cfg = config.get('training', {}).get('quantization_warmup', {})
    quantization_strength = compute_schedule_value(epoch, quantization_warmup_cfg, 1.0)
    schedule_metrics['quantization_strength'] = quantization_strength
    if hasattr(model, 'set_quantization_strength'):
        model.set_quantization_strength(quantization_strength)

    return schedule_metrics


def maybe_apply_warm_start(model, config: dict, device: torch.device):
    warm_cfg = config.get('warm_start', {})
    if not warm_cfg:
        return

    full_checkpoint = warm_cfg.get('checkpoint')
    if full_checkpoint:
        checkpoint = load_checkpoint_file(full_checkpoint, device=device)
        source_state = checkpoint['model_state_dict']
        reset_coupling = bool(warm_cfg.get('reset_coupling', False))
        if reset_coupling:
            source_state = {
                key: value
                for key, value in source_state.items()
                if key != 'coupling_logits'
            }
        incompatible = model.load_state_dict(source_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected full warm-start keys from {full_checkpoint}: {incompatible.unexpected_keys}"
            )
        allowed_missing = {'coupling_logits'} if reset_coupling else set()
        # New auxiliary probes are intentionally initialized from scratch when
        # branching from older tokenizer checkpoints.
        warm_start_compatible_prefixes = (
            'eeg_to_fnirs_source_predictor.',
            'fnirs_to_eeg_source_predictor.',
            'eeg_shared_state_proj.',
            'fnirs_shared_state_proj.',
        )
        allowed_missing.update(
            key for key in incompatible.missing_keys
            if key.startswith(warm_start_compatible_prefixes)
        )
        disallowed_missing = set(incompatible.missing_keys) - allowed_missing
        if disallowed_missing:
            raise RuntimeError(
                f"Missing full warm-start keys from {full_checkpoint}: {sorted(disallowed_missing)}"
            )
        if reset_coupling and hasattr(model, 'coupling_logits'):
            reset_std = float(warm_cfg.get('coupling_reset_std', 0.02))
            with torch.no_grad():
                model.coupling_logits.normal_(mean=0.0, std=reset_std)
        print(
            f"Full warm-start: loaded {full_checkpoint}; "
            f"reset_coupling={reset_coupling}"
        )

    def load_branch(checkpoint_path: str, branch_prefix: str):
        checkpoint = load_checkpoint_file(checkpoint_path, device=device)
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
        'schedule_state': snapshot_model_schedule_state(model),
        'is_best': bool(is_best),
    }


def resolve_git_commit(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def infer_phase_name(config_path: str) -> Optional[str]:
    for part in Path(config_path).parts:
        if part.lower().startswith('phase'):
            return part
    return None


def write_run_manifest(
    *,
    logger: ExperimentLogger,
    config: Dict[str, Any],
    config_path: str,
    analysis_type: str,
) -> Dict[str, Any]:
    manifest = {
        'schema_version': 'phase1_run_manifest_v1',
        'generated_at': datetime.now().isoformat(),
        'run_name': logger.run_name,
        'config_path': config_path,
        'config_hash': logger.config_hash,
        'dataset': config.get('data', {}).get('dataset'),
        'model_type': config.get('model', {}).get('type'),
        'phase': infer_phase_name(config_path),
        'analysis_type': analysis_type,
        'control_group': config.get('experiment', {}).get('control_group'),
        'semantics_version': 'phase1_source_observation_v1',
        'git_commit': resolve_git_commit(project_root),
    }
    write_json(logger.run_dir / 'run_manifest.json', manifest)
    return manifest


def write_final_summary(logger: ExperimentLogger, payload: Dict[str, Any]) -> None:
    write_json(logger.run_dir / 'final_summary.json', payload)


def finalize_training_run(
    *,
    logger: ExperimentLogger,
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
        best_checkpoint = load_training_checkpoint(best_path, model, device=device)
        apply_model_schedule_state(model, best_checkpoint.get('schedule_state', {}))
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

    validation_cfg = config.get('training', {}).get('validation', {})
    max_test_batches = _positive_int_or_none(
        validation_cfg.get('max_test_batches', validation_cfg.get('max_batches'))
    )
    test_metrics = validate_epoch(
        model,
        test_loader,
        config,
        device,
        max_batches=max_test_batches,
    )
    final_metrics = {
        'best_epoch': best_epoch,
        'best_monitor': best_monitor,
        **test_metrics,
    }
    logger.log_final(final_metrics)

    analysis_type = getattr(model, 'get_analysis_type', lambda: 'source_observation_alignment')()
    summary_root = logger.run_dir / 'analysis'
    summary_root.mkdir(parents=True, exist_ok=True)

    if skip_post_analysis:
        write_final_summary(
            logger,
            {
                'schema_version': 'phase1_final_summary_v1',
                'run_name': logger.run_name,
                'analysis_type': analysis_type,
                'analysis_skipped': True,
                'best_epoch': best_epoch,
                'best_monitor': best_monitor,
                'best_checkpoint': 'checkpoints/best_model.pt',
                'summary': 'Post-analysis skipped; gate scorecard was not generated.',
            },
        )
        return final_metrics

    print(f"Running tokenizer analysis suite -> {summary_root}")
    analysis_dataloaders = {'val': val_loader, 'test': test_loader}
    try:
        suite_results = generate_tokenizer_analysis_suite(
            model=model,
            dataloaders=analysis_dataloaders,
            config=config,
            run_dir=logger.run_dir,
            output_dir=summary_root,
            device=analysis_device,
            analysis_type=analysis_type,
        )
    finally:
        shutdown_dataloader_workers(analysis_dataloaders)
    scorecard_results = suite_results['scorecard']
    write_final_summary(logger, scorecard_results['final_summary'])

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Train EEG+fNIRS tokenizer")
    parser.add_argument('--config', required=True, help='Config path relative to experiments/configs')
    parser.add_argument('--resume', default=None, help='Optional checkpoint path')
    parser.add_argument('--run-name', default=None, help='Optional run directory name to reuse inside experiments/runs')
    parser.add_argument('--skip-post-analysis', action='store_true', help='Skip default tokenizer analysis suite at the end of training')
    args = parser.parse_args()

    require_standard_training_launcher('source-observation-tokenizer')

    logger = ExperimentLogger(config_path=args.config, run_name=args.run_name)
    config = logger.config
    tee_logger = setup_logging(logger.run_dir)
    tb_cfg = config.get('logging', {}).get('tensorboard', {})
    tb_logger: Optional[TensorBoardLogger] = None
    if tb_cfg.get('enabled', True):
        tb_logger = TensorBoardLogger(
            run_dir=logger.run_dir,
            log_subdir=tb_cfg.get('subdir', 'tensorboard'),
            save_figure_snapshots=bool(tb_cfg.get('save_figure_snapshots', False)),
        )

    try:
        print("=" * 70)
        print("EEG+fNIRS Tokenizer Training")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Run directory: {logger.run_dir}")
        print(f"Experiment: {config['experiment']['name']}")
        print(f"Description: {config['experiment'].get('description', 'N/A')}")
        if tb_logger is not None:
            print(f"TensorBoard: tensorboard --logdir {tb_logger.log_dir}")
            tb_logger.log_text_summary(
                json.dumps(
                    {
                        'run_name': logger.run_name,
                        'config_path': args.config,
                        'experiment': config['experiment']['name'],
                        'description': config['experiment'].get('description', ''),
                    },
                    indent=2,
                ),
                step=0,
                tag='run/metadata',
            )
            tb_logger.flush()

        device = setup_device(config)
        setup_torch_performance(config, device)
        print(f"Training device: {device}")
        if device.type == 'cuda':
            print(
                "CUDA performance: "
                f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32}, "
                f"tf32_cudnn={torch.backends.cudnn.allow_tf32}"
            )

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
        inject_channel_names_into_config(config, dataloaders)
        if hasattr(train_loader.dataset, 'describe_sources'):
            print('Train source mix:')
            for source in train_loader.dataset.describe_sources():
                print(
                    f"  - {source['name']}: dataset={source['dataset']}, task={source['task']}, samples={source['length']}"
                )

        print("\nCreating tokenizer...")
        model = create_tokenizer(config).to(device)
        print(f"Model: {model.__class__.__name__}")
        print(f"Source codebook size: {model.get_codebook_size()}")
        maybe_apply_warm_start(model, config, device)
        model = maybe_compile_model_forward(model, config, device)
        analysis_type = getattr(model, 'get_analysis_type', lambda: 'source_observation_alignment')()
        write_run_manifest(
            logger=logger,
            config=config,
            config_path=args.config,
            analysis_type=analysis_type,
        )

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
            checkpoint = load_training_checkpoint(Path(args.resume), model, optimizer, device)
            apply_model_schedule_state(model, checkpoint.get('schedule_state', {}))
            start_epoch = int(checkpoint.get('epoch', 0))
            resume_best_epoch = checkpoint.get('best_epoch')
            resume_best_monitor = checkpoint.get('best_monitor')
            maybe_seed_best_checkpoint(logger, Path(args.resume), checkpoint)
            print(f"Resumed from epoch {start_epoch}")
        else:
            resume_best_epoch = None
            resume_best_monitor = None

        es_cfg = config['training'].get('early_stopping', {})
        patience = int(es_cfg.get('patience', 40))
        monitor_metric = es_cfg.get('metric', 'val_loss')
        monitor_mode = es_cfg.get('mode', 'min')
        early_stopping_start_epoch = max(int(es_cfg.get('start_epoch', 1)), 1)
        best_monitor = float('inf') if monitor_mode == 'min' else float('-inf')
        if resume_best_monitor is not None:
            best_monitor = float(resume_best_monitor)
        epochs_without_improvement = 0
        save_every = int(config['training'].get('checkpoint', {}).get('save_every', 1))
        grad_clip = float(config['training'].get('gradient', {}).get('clip_norm', 1.0))
        gradient_attribution_cfg = {
            'enabled': bool(config['training'].get('gradient_attribution', {}).get('enabled', False)),
            'max_batches': int(config['training'].get('gradient_attribution', {}).get('max_batches', 1)),
            'interval_epochs': max(
                int(config['training'].get('gradient_attribution', {}).get('interval_epochs', 1)),
                1,
            ),
        }
        validation_cfg = config['training'].get('validation', {})
        validation_interval = max(int(validation_cfg.get('interval_epochs', 1)), 1)
        validation_start_epoch = max(int(validation_cfg.get('start_epoch', 1)), 1)
        validation_max_batches = _positive_int_or_none(validation_cfg.get('max_batches'))
        max_train_batches = _positive_int_or_none(config['training'].get('max_train_batches'))

        best_epoch = int(resume_best_epoch) if resume_best_epoch is not None else start_epoch
        interrupted = False
        stop_epoch = start_epoch
        if es_cfg.get('enabled', True) and early_stopping_start_epoch > 1:
            print(
                f"Early stopping monitor '{monitor_metric}' activates at epoch {early_stopping_start_epoch}"
            )
        try:
            for epoch in range(start_epoch + 1, total_epochs + 1):
                stop_epoch = epoch
                schedule_metrics = apply_epoch_schedules(model, epoch, config)
                alignment_scale = schedule_metrics['alignment_scale']

                _cuda_synchronize_if_needed(device)
                train_started_at = time.perf_counter()
                gradient_interval = int(gradient_attribution_cfg['interval_epochs'])
                run_gradient_attribution = bool(gradient_attribution_cfg['enabled']) and (
                    epoch == start_epoch + 1
                    or epoch == total_epochs
                    or epoch % gradient_interval == 0
                )
                epoch_gradient_attribution_cfg = {
                    **gradient_attribution_cfg,
                    'enabled': run_gradient_attribution,
                }
                train_metrics, gradient_dashboard = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    config,
                    device,
                    grad_clip,
                    gradient_attribution_cfg=epoch_gradient_attribution_cfg,
                    max_batches=max_train_batches,
                )
                _cuda_synchronize_if_needed(device)
                train_seconds = time.perf_counter() - train_started_at

                run_validation = (
                    epoch >= validation_start_epoch
                    and (
                        (epoch - validation_start_epoch) % validation_interval == 0
                        or epoch == total_epochs
                    )
                )
                if run_validation:
                    _cuda_synchronize_if_needed(device)
                    val_started_at = time.perf_counter()
                    val_metrics = validate_epoch(
                        model,
                        val_loader,
                        config,
                        device,
                        max_batches=validation_max_batches,
                    )
                    _cuda_synchronize_if_needed(device)
                    val_seconds = time.perf_counter() - val_started_at
                else:
                    val_metrics = {}
                    val_seconds = 0.0

                if epoch > warmup_epochs:
                    scheduler.step()

                lr = optimizer.param_groups[0]['lr']
                train_loss = train_metrics.get('loss', float('nan'))
                val_loss = val_metrics.get('val_loss')
                train_batches = max_train_batches or len(train_loader)
                val_batches = validation_max_batches or len(val_loader)
                merged_metrics = {
                    'lr': lr,
                    **schedule_metrics,
                    'train_epoch_seconds': train_seconds,
                    'train_batches_per_second': train_batches / max(train_seconds, 1e-12),
                    'validation_ran': 1.0 if run_validation else 0.0,
                }
                if run_validation:
                    merged_metrics.update({
                        'val_epoch_seconds': val_seconds,
                        'val_batches_per_second': val_batches / max(val_seconds, 1e-12),
                    })
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

                if tb_logger is not None:
                    tb_logger.log_scalars('train', filter_numeric_scalars(train_metrics), epoch)
                    if run_validation:
                        tb_logger.log_scalars('val', filter_numeric_scalars(val_metrics, strip_prefix='val_'), epoch)
                    tb_logger.log_learning_rate(lr, epoch)
                    tb_logger.log_scalars('schedule', schedule_metrics, epoch)
                    if gradient_dashboard is not None:
                        tb_logger.log_gradient_conflict_dashboard(
                            component_names=gradient_dashboard['component_names'],
                            cosine_matrix=gradient_dashboard['cosine_matrix'],
                            component_norms=gradient_dashboard['component_norms'],
                            component_shares=gradient_dashboard['component_shares'],
                            step=epoch,
                        )
                        if (
                            'group_names' in gradient_dashboard and
                            'component_group_shares' in gradient_dashboard and
                            'group_component_shares' in gradient_dashboard and
                            'group_total_shares' in gradient_dashboard and
                            'group_abs_grad_quantiles' in gradient_dashboard
                        ):
                            tb_logger.log_gradient_influence_dashboard(
                                component_names=gradient_dashboard['component_names'],
                                group_labels=gradient_dashboard.get('group_labels', gradient_dashboard['group_names']),
                                component_group_shares=gradient_dashboard['component_group_shares'],
                                group_component_shares=gradient_dashboard['group_component_shares'],
                                group_total_shares=gradient_dashboard['group_total_shares'],
                                group_abs_grad_quantiles=gradient_dashboard['group_abs_grad_quantiles'],
                                step=epoch,
                            )
                    tb_logger.flush()

                monitor_value = val_metrics.get(monitor_metric)
                if run_validation and monitor_value is None:
                    raise ValueError(f"Monitor metric '{monitor_metric}' not found in validation metrics")

                monitor_active = run_validation and epoch >= early_stopping_start_epoch
                improved = False
                if monitor_active:
                    improved = (
                        monitor_value < best_monitor if monitor_mode == 'min'
                        else monitor_value > best_monitor
                    )
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
                    val_loss=float(val_loss) if val_loss is not None else float('nan'),
                    monitor_metric=monitor_metric,
                    monitor_value=float(monitor_value) if monitor_value is not None else float('nan'),
                    best_epoch=best_epoch,
                    best_monitor=best_monitor,
                    alignment_scale=alignment_scale,
                    is_best=improved,
                )

                save_periodic_checkpoint = save_every > 0 and (epoch % save_every == 0)
                if save_periodic_checkpoint or improved:
                    logger.save_checkpoint(
                        checkpoint_payload,
                        epoch=epoch,
                        is_best=improved,
                        keep_epoch_copy=save_periodic_checkpoint,
                    )

                if es_cfg.get('enabled', True) and monitor_active and epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                    break
        except KeyboardInterrupt:
            interrupted = True
            print(f"\nTraining interrupted at epoch {stop_epoch}. Finalizing from best available checkpoint...")

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
        if tb_logger is not None:
            tb_logger.log_scalars(
                'test',
                filter_numeric_scalars(final_metrics),
                int(final_metrics.get('best_epoch', stop_epoch)),
            )
            if tb_cfg.get('log_hparams', True):
                tb_logger.log_hparams(
                    build_tensorboard_hparams(config),
                    filter_numeric_scalars(final_metrics),
                )
            tb_logger.flush()

        if args.skip_post_analysis:
            print(
                "[Info] Post-analysis was skipped by --skip-post-analysis. Only lightweight final summaries were "
                "written for this run."
            )

        if interrupted:
            print("\nTraining interrupted after finalization.")
        else:
            print("\nTraining complete.")
        print(f"Best epoch: {final_metrics['best_epoch']}")
        print(f"Final metrics saved to: {logger.run_dir / 'metrics.json'}")
    finally:
        if tb_logger is not None:
            tb_logger.close()
        tee_logger.close()


if __name__ == '__main__':
    main()
