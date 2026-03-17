"""Generate real-data preprocessing visualizations from experiment configs.

This script has two goals:
1. Visualize raw signal segments and config-driven filtered signals.
2. Show synchronized EEG and fNIRS segments on aligned time axes.

It also records the preprocessing that is currently used by the training
pipeline in experiments/scripts/train_tokenizer.py, which still consists of:
- channel filtering via exclude_eog / hbo_only / hbr_only
- per-window z-score normalization
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    EEGfNIRSDataset,
    MultiModalEEGfNIRSDataset,
    summarize_preprocessing_config,
    visualize_filtered_dataset_sample,
    visualize_synchronized_filtered_sample,
)


DEFAULT_CONFIGS = [
    'phase0plus/eeg_patch_vqvae_1s_v3.yaml',
    'phase0plus/fnirs_patch_vqvae_2s_v2.yaml',
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(config_name: str) -> Dict[str, Any]:
    config_path = PROJECT_ROOT / 'experiments' / 'configs' / config_name
    config = load_yaml(config_path)
    base_name = config.get('_base_')
    if base_name:
        candidate_paths = [
            (config_path.parent / base_name).resolve(),
            (PROJECT_ROOT / 'experiments' / 'configs' / base_name).resolve(),
        ]
        base_path = next((path for path in candidate_paths if path.exists()), None)
        if base_path is None:
            raise FileNotFoundError(f'Could not resolve base config {base_name!r} for {config_name!r}')
        base_config = load_yaml(base_path)
        config = deep_merge(base_config, {k: v for k, v in config.items() if k != '_base_'})
    return config


def create_dataset(config: Dict[str, Any], split: str) -> EEGfNIRSDataset:
    data_cfg = config['data']
    split_cfg = data_cfg['split']
    subjects = split_cfg[f'{split}_subjects']
    return EEGfNIRSDataset(
        data_root=data_cfg['data_root'],
        subject_ids=subjects,
        task=data_cfg.get('task', 'motor_imagery'),
        modality=data_cfg['modality'],
        window_samples=data_cfg['window']['length'],
        window_offset_ms=data_cfg['window'].get('offset_ms', 0),
        normalize=True,
        exclude_eog=data_cfg.get('exclude_eog', False),
        hbo_only=data_cfg.get('hbo_only', False),
        hbr_only=data_cfg.get('hbr_only', False),
    )


def create_multimodal_dataset(
    eeg_config: Dict[str, Any],
    fnirs_config: Dict[str, Any],
    split: str,
) -> MultiModalEEGfNIRSDataset:
    eeg_data = eeg_config['data']
    fnirs_data = fnirs_config['data']

    eeg_subjects = eeg_data['split'][f'{split}_subjects']
    fnirs_subjects = fnirs_data['split'][f'{split}_subjects']
    if eeg_subjects != fnirs_subjects:
        raise ValueError('EEG and fNIRS configs must use identical subject splits for synchronized visualization')

    eeg_rate = float(eeg_data.get('preprocessing', {}).get('resample_rate', 200.0))
    fnirs_rate = float(fnirs_data.get('preprocessing', {}).get('resample_rate', 10.0))
    eeg_duration_s = eeg_data['window']['length'] / eeg_rate
    fnirs_duration_s = fnirs_data['window']['length'] / fnirs_rate

    if abs(eeg_duration_s - fnirs_duration_s) > 1e-6:
        raise ValueError(
            f'Window durations must match for synchronized visualization, got EEG={eeg_duration_s}s and fNIRS={fnirs_duration_s}s'
        )

    eeg_offset_ms = float(eeg_data['window'].get('offset_ms', 0.0))
    fnirs_offset_ms = float(fnirs_data['window'].get('offset_ms', eeg_offset_ms))
    if abs(eeg_offset_ms - fnirs_offset_ms) > 1e-6:
        raise ValueError(
            f'Window offsets must match for synchronized visualization, got EEG={eeg_offset_ms}ms and fNIRS={fnirs_offset_ms}ms'
        )

    return MultiModalEEGfNIRSDataset(
        data_root=eeg_data['data_root'],
        subject_ids=eeg_subjects,
        task=eeg_data.get('task', 'motor_imagery'),
        window_duration_s=eeg_duration_s,
        window_offset_ms=eeg_offset_ms,
        normalize=False,
        exclude_eog=eeg_data.get('exclude_eog', False),
        hbo_only=fnirs_data.get('hbo_only', False),
        hbr_only=fnirs_data.get('hbr_only', False),
    )


def select_sample_indices(dataset: EEGfNIRSDataset, num_samples: int) -> List[int]:
    if len(dataset) == 0:
        return []
    if num_samples <= 1:
        return [0]
    if len(dataset) <= num_samples:
        return list(range(len(dataset)))

    last_index = len(dataset) - 1
    return sorted({int(round(i * last_index / (num_samples - 1))) for i in range(num_samples)})


def build_summary_entry(
    config_name: str,
    config: Dict[str, Any],
    dataset: EEGfNIRSDataset,
    sample_idx: int,
    figures: Dict[str, Path],
) -> Dict[str, Any]:
    trial = dataset.trials[sample_idx]
    return {
        'config': config_name,
        'experiment_name': config['experiment']['name'],
        'sample_idx': sample_idx,
        'subject_id': trial.subject_id,
        'session_idx': trial.session_idx,
        'trial_idx': trial.trial_idx,
        'label': trial.label,
        'task_type': trial.task_type,
        'modality': dataset.modality,
        'sample_rate': dataset.get_sample_rate(),
        'window_samples': dataset.window_samples,
        'window_offset_ms': dataset.window_offset_ms,
        'training_preprocess': {
            'channel_filtering': {
                'exclude_eog': dataset.exclude_eog,
                'hbo_only': dataset.hbo_only,
                'hbr_only': dataset.hbr_only,
            },
            'per_window_zscore': dataset.normalize,
        },
        'visualization_preprocess': summarize_preprocessing_config(
            config['data'].get('preprocessing', {}),
            dataset.modality,
        ),
        'configured_preprocessing': config['data'].get('preprocessing', {}),
        'figures': {name: str(path.relative_to(PROJECT_ROOT)).replace('\\', '/') for name, path in figures.items()},
    }


def build_sync_summary_entry(
    eeg_config_name: str,
    eeg_config: Dict[str, Any],
    fnirs_config_name: str,
    fnirs_config: Dict[str, Any],
    dataset: MultiModalEEGfNIRSDataset,
    sample_idx: int,
    figures: Dict[str, Path],
) -> Dict[str, Any]:
    trial = dataset.trials[sample_idx]
    return {
        'eeg_config': eeg_config_name,
        'fnirs_config': fnirs_config_name,
        'sample_idx': sample_idx,
        'subject_id': trial.subject_id,
        'session_idx': trial.session_idx,
        'trial_idx': trial.trial_idx,
        'label': trial.label,
        'task_type': trial.task_type,
        'window_duration_s': dataset.window_duration_s,
        'window_offset_ms': dataset.window_offset_ms,
        'visualization_preprocess': {
            'eeg': summarize_preprocessing_config(eeg_config['data'].get('preprocessing', {}), 'eeg'),
            'fnirs': summarize_preprocessing_config(fnirs_config['data'].get('preprocessing', {}), 'fnirs'),
        },
        'figures': {name: str(path.relative_to(PROJECT_ROOT)).replace('\\', '/') for name, path in figures.items()},
    }


def save_markdown_summary(
    output_dir: Path,
    single_summary: List[Dict[str, Any]],
    sync_summary: List[Dict[str, Any]],
) -> None:
    lines = [
        '# Preprocess Visualization Summary',
        '',
        'This report separates visualization-time filtering from the current training-time preprocessing.',
        '',
        '- Visualization preprocess: config-driven temporal filtering used to inspect signal quality and preprocessing behavior.',
        '- Current training preprocess: channel filtering plus per-window z-score normalization.',
        '',
    ]

    lines.extend(['# Single-Modality Views', ''])

    for item in single_summary:
        lines.extend(
            [
                f"## {item['experiment_name']} | sample {item['sample_idx']}",
                '',
                f"- Modality: {item['modality']}",
                f"- Subject / session / trial: {item['subject_id']} / {item['session_idx']} / {item['trial_idx']}",
                f"- Label: {item['label']}",
                f"- Sample rate: {item['sample_rate']}",
                f"- Window samples: {item['window_samples']}",
                f"- Window offset ms: {item['window_offset_ms']}",
                f"- Visualization preprocess: {json.dumps(item['visualization_preprocess'], ensure_ascii=False)}",
                f"- Current training preprocess: {json.dumps(item['training_preprocess'], ensure_ascii=False)}",
                f"- Raw config preprocessing: {json.dumps(item['configured_preprocessing'], ensure_ascii=False)}",
                '',
                'Saved figures:',
                '',
            ]
        )
        for figure_name, figure_path in item['figures'].items():
            lines.append(f'- {figure_name}: {figure_path}')
        lines.append('')

    if sync_summary:
        lines.extend(['# Synchronized EEG-fNIRS Views', ''])

    for item in sync_summary:
        lines.extend(
            [
                f"## synchronized sample {item['sample_idx']}",
                '',
                f"- Subject / session / trial: {item['subject_id']} / {item['session_idx']} / {item['trial_idx']}",
                f"- Label: {item['label']}",
                f"- Window duration s: {item['window_duration_s']}",
                f"- Window offset ms: {item['window_offset_ms']}",
                f"- Visualization preprocess: {json.dumps(item['visualization_preprocess'], ensure_ascii=False)}",
                '',
                'Saved figures:',
                '',
            ]
        )
        for figure_name, figure_path in item['figures'].items():
            lines.append(f'- {figure_name}: {figure_path}')
        lines.append('')

    report_path = output_dir / 'summary.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize real preprocessing samples.')
    parser.add_argument(
        '--configs',
        nargs='+',
        default=DEFAULT_CONFIGS,
        help='Config files relative to experiments/configs',
    )
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='train',
        help='Subject split to sample from',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=2,
        help='Number of evenly spaced samples per config',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='',
        help='Output directory. Defaults to logs/preprocess_visualizations/<timestamp>.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'logs' / 'preprocess_visualizations' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    single_summary: List[Dict[str, Any]] = []
    sync_summary: List[Dict[str, Any]] = []
    configs_by_modality: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    for config_name in args.configs:
        config = load_experiment_config(config_name)
        configs_by_modality[config['data']['modality']] = (config_name, config)
        dataset = create_dataset(config, args.split)
        sample_indices = select_sample_indices(dataset, args.num_samples)

        config_dir = output_dir / config['experiment']['name']
        config_dir.mkdir(parents=True, exist_ok=True)

        print(f'[Viz] Config: {config_name}')
        print(f'[Viz] Dataset size: {len(dataset)} | selected indices: {sample_indices}')

        for sample_idx in sample_indices:
            sample_dir = config_dir / f'sample_{sample_idx:04d}'
            sample_dir.mkdir(parents=True, exist_ok=True)
            figures = visualize_filtered_dataset_sample(
                dataset,
                sample_idx,
                preprocessing=config['data'].get('preprocessing', {}),
                output_dir=sample_dir,
            )
            single_summary.append(build_summary_entry(config_name, config, dataset, sample_idx, figures))

    if 'eeg' in configs_by_modality and 'fnirs' in configs_by_modality:
        eeg_config_name, eeg_config = configs_by_modality['eeg']
        fnirs_config_name, fnirs_config = configs_by_modality['fnirs']
        multimodal_dataset = create_multimodal_dataset(eeg_config, fnirs_config, args.split)
        sample_indices = select_sample_indices(multimodal_dataset, args.num_samples)
        sync_dir = output_dir / 'synchronized'
        sync_dir.mkdir(parents=True, exist_ok=True)

        print(f'[Viz] Synchronized dataset size: {len(multimodal_dataset)} | selected indices: {sample_indices}')

        for sample_idx in sample_indices:
            sample_dir = sync_dir / f'sample_{sample_idx:04d}'
            sample_dir.mkdir(parents=True, exist_ok=True)
            figures = visualize_synchronized_filtered_sample(
                multimodal_dataset,
                sample_idx,
                eeg_preprocessing=eeg_config['data'].get('preprocessing', {}),
                fnirs_preprocessing=fnirs_config['data'].get('preprocessing', {}),
                output_dir=sample_dir,
            )
            sync_summary.append(
                build_sync_summary_entry(
                    eeg_config_name,
                    eeg_config,
                    fnirs_config_name,
                    fnirs_config,
                    multimodal_dataset,
                    sample_idx,
                    figures,
                )
            )

    summary_path = output_dir / 'summary.json'
    summary_payload = {
        'single_modality': single_summary,
        'synchronized': sync_summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding='utf-8')
    save_markdown_summary(output_dir, single_summary, sync_summary)

    print(f'[Viz] Saved summary: {summary_path}')
    print(f'[Viz] Output directory: {output_dir}')


if __name__ == '__main__':
    main()