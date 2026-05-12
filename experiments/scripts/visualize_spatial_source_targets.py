#!/usr/bin/env python
"""Preview spatially-informed EEG/fNIRS source targets on real samples."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import (
    compute_cross_modal_correlation_matrix,
    compute_fnirs_midpoint_diagnostics,
    create_configured_multimodal_dataloaders,
    plot_adjacency_heatmap,
    plot_channel_layout,
    plot_channel_layout_3d,
    plot_channel_layout_projected_2d,
    plot_cross_modal_correlation_heatmap,
    plot_spatial_target_preview,
)
from src.data.registry import load_experiment_config
from src.tokenizers import create_tokenizer


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
    model_cfg.setdefault('eeg', {})['channels'] = len(eeg_channel_names)
    model_cfg.setdefault('fnirs', {})['channels'] = len(fnirs_channel_names)


def resolve_preview_config(config: Dict[str, Any], *, max_samples_override: int | None) -> Dict[str, Any]:
    analysis_cfg = config.get('analysis', {})
    spatial_cfg = analysis_cfg.get('spatial_visualization', {})
    max_samples = int(spatial_cfg.get('max_samples', 4))
    if max_samples_override is not None:
        max_samples = int(max_samples_override)
    return {
        'subdir': str(spatial_cfg.get('subdir', 'spatial_target_previews')),
        'max_samples': max(max_samples, 1),
    }


def get_batch(loader, batch_index: int) -> Dict[str, Any]:
    for index, batch in enumerate(loader):
        if index == batch_index:
            return batch
    raise IndexError(f'Batch index {batch_index} is out of range for split loader')


def main() -> None:
    parser = argparse.ArgumentParser(description='Preview spatial EEG/fNIRS source targets on actual samples')
    parser.add_argument('--config', required=True, help='Config path relative to experiments/configs')
    parser.add_argument('--split', default='val', choices=('train', 'val', 'test'), help='Dataset split to preview')
    parser.add_argument('--batch-index', type=int, default=0, help='Which batch to visualize within the split')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional limit for how many samples to export from the batch')
    parser.add_argument('--output-dir', default=None, help='Optional output directory override')
    args = parser.parse_args()

    config = copy.deepcopy(load_experiment_config(args.config))
    dataloaders = create_configured_multimodal_dataloaders(config)
    inject_channel_names_into_config(config, dataloaders)

    preview_cfg = resolve_preview_config(config, max_samples_override=args.max_samples)
    experiment_name = config.get('experiment', {}).get('name', Path(args.config).stem)
    default_output_root = project_root / 'experiments' / 'results' / preview_cfg['subdir'] / f'{experiment_name}_{args.split}'
    output_root = Path(args.output_dir) if args.output_dir is not None else default_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    model = create_tokenizer(config)
    if not hasattr(model, 'has_spatial_source_prior') or not model.has_spatial_source_prior():
        raise RuntimeError(
            'Spatial prior is not enabled for this config. Set loss.source_target.spatial.enabled=true before running the preview.'
        )

    loader = dataloaders[args.split]
    batch = get_batch(loader, int(args.batch_index))
    max_samples = min(int(preview_cfg['max_samples']), int(batch['eeg'].shape[0]))
    eeg = batch['eeg'][:max_samples]
    fnirs = batch['fnirs'][:max_samples]

    with torch.no_grad():
        eeg_target = model._compute_eeg_source_target(eeg)
        fnirs_target = model._compute_fnirs_source_target(eeg, fnirs)
        correlation = compute_cross_modal_correlation_matrix(eeg, fnirs).cpu().numpy()

    spatial_info = model.spatial_adjacency_info
    if spatial_info is None:
        raise RuntimeError('Model did not expose spatial adjacency metadata after enabling the spatial prior.')

    midpoint_diagnostics = compute_fnirs_midpoint_diagnostics(spatial_info)

    artifacts = {
        'layout_midpoint_xy_path': plot_channel_layout(spatial_info, output_root / 'channel_layout_midpoint_xy.png'),
        'layout_path': plot_channel_layout_projected_2d(spatial_info, output_root / 'channel_layout.png'),
        'layout_3d_path': plot_channel_layout_3d(spatial_info, output_root / 'channel_layout_3d.png'),
        'adjacency_heatmap_path': plot_adjacency_heatmap(spatial_info, output_root / 'adjacency_heatmap.png'),
        'cross_modal_correlation_path': plot_cross_modal_correlation_heatmap(
            correlation,
            spatial_info,
            output_root / 'cross_modal_correlation.png',
        ),
    }
    sample_paths = []
    eeg_numpy = eeg.cpu().numpy()
    fnirs_numpy = fnirs.cpu().numpy()
    eeg_target_numpy = eeg_target.cpu().numpy()
    fnirs_target_numpy = fnirs_target.cpu().numpy()
    for sample_index in range(max_samples):
        sample_path = plot_spatial_target_preview(
            adjacency=spatial_info,
            eeg=eeg_numpy,
            fnirs=fnirs_numpy,
            eeg_target=eeg_target_numpy,
            fnirs_target=fnirs_target_numpy,
            eeg_sample_rate_hz=float(model.eeg_sampling_rate_hz),
            fnirs_sample_rate_hz=float(model.fnirs_sampling_rate_hz),
            output_path=output_root / f'sample_{sample_index:02d}_spatial_targets.png',
            sample_index=sample_index,
        )
        sample_paths.append(sample_path)
    artifacts['sample_target_paths'] = sample_paths

    np.save(output_root / 'adjacency_matrix.npy', spatial_info.adjacency_matrix)
    np.save(output_root / 'cross_modal_correlation.npy', correlation)

    summary = {
        'config_path': args.config,
        'split': args.split,
        'batch_index': int(args.batch_index),
        'max_samples': int(max_samples),
        'output_root': str(output_root),
        'spatial_prior': model.get_spatial_prior_info(),
        'batch_shapes': {
            'eeg': list(eeg.shape),
            'fnirs': list(fnirs.shape),
            'eeg_target': list(eeg_target.shape),
            'fnirs_target': list(fnirs_target.shape),
        },
        'target_statistics': {
            'eeg_raw_std': float(eeg.std().item()),
            'eeg_target_std': float(eeg_target.std().item()),
            'fnirs_raw_std': float(fnirs.std().item()),
            'fnirs_target_std': float(fnirs_target.std().item()),
        },
        'midpoint_diagnostics': {
            'direct_midpoint_error_mean': midpoint_diagnostics['direct_midpoint_error_mean'],
            'direct_midpoint_error_max': midpoint_diagnostics['direct_midpoint_error_max'],
            'normalized_midpoint_error_mean': midpoint_diagnostics['normalized_midpoint_error_mean'],
            'normalized_midpoint_error_max': midpoint_diagnostics['normalized_midpoint_error_max'],
        },
        'artifacts': artifacts,
    }
    with open(output_root / 'preview_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()