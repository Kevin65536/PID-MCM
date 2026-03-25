#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.eeg_fnirs_dataset import create_dataloaders
from src.tokenizers import create_tokenizer
from src.visualization.shared_alignment_analysis import analyze_shared_alignment


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


def main():
    parser = argparse.ArgumentParser(description='Analyze trained shared EEG-fNIRS tokenizer checkpoints')
    parser.add_argument('--checkpoint', required=True, help='Path to the shared tokenizer checkpoint')
    parser.add_argument('--output-dir', default=None, help='Output directory for analysis')
    parser.add_argument('--device', default='cuda', help='Preferred device')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    device = torch.device(args.device if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')

    model = create_tokenizer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataloaders = create_multimodal_dataloaders(config)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent / 'analysis' / 'shared_alignment'
    results = analyze_shared_alignment(model, dataloaders, config, output_dir, device)

    print(f'Analysis saved to: {output_dir}')
    for split_name, split_result in results.get('splits', {}).items():
        print(
            f'{split_name}: best_lag={split_result["best_lag"]}, '
            f'best_lag_mi={split_result["best_lag_mutual_information"]:.4f}, '
            f'lag0_match={split_result["lag_metrics"][0]["match_rate"]:.4f}'
        )


if __name__ == '__main__':
    main()