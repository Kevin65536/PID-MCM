"""Visualize full-session signals with shaded task extraction windows.

Generates a four-panel figure for one subject/session:
1. EEG raw full-session trace
2. EEG filtered + session-normalized trace
3. fNIRS raw full-session trace
4. fNIRS filtered + session-normalized trace

Each task window is highlighted with a background color according to its label.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yaml

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eeg_fnirs_dataset import EEGfNIRSDataset


DEFAULT_EEG_CONFIG = 'phase0plus/eeg_patch_vqvae_1s_v3.yaml'
DEFAULT_FNIRS_CONFIG = 'phase0plus/fnirs_patch_vqvae_2s_v2.yaml'


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


def create_dataset(config: Dict[str, Any], subject_id: int) -> EEGfNIRSDataset:
    data_cfg = config['data']
    normalize, normalization_mode = resolve_normalization_config(data_cfg)
    return EEGfNIRSDataset(
        data_root=data_cfg['data_root'],
        subject_ids=[subject_id],
        task=data_cfg.get('task', 'motor_imagery'),
        modality=data_cfg['modality'],
        window_samples=data_cfg['window']['length'],
        window_offset_ms=data_cfg['window'].get('offset_ms', 0),
        normalize=normalize,
        normalization_mode=normalization_mode,
        preprocessing=data_cfg.get('preprocessing', {}),
        exclude_eog=data_cfg.get('exclude_eog', False),
        hbo_only=data_cfg.get('hbo_only', False),
        hbr_only=data_cfg.get('hbr_only', False),
    )


def pick_channel_index(channel_names: List[str], preferred: str | None) -> int:
    if preferred is None:
        return 0
    if preferred in channel_names:
        return channel_names.index(preferred)
    raise ValueError(f'Channel {preferred!r} not found in {channel_names}')


def add_trial_spans(ax: plt.Axes, regions: List[Dict[str, Any]], color_map: Dict[int, str]) -> None:
    for region in regions:
        color = color_map.get(int(region['label']), '#95A5A6')
        ax.axvspan(region['start_s'], region['end_s'], color=color, alpha=0.18)


def plot_session_overview(
    eeg_dataset: EEGfNIRSDataset,
    fnirs_dataset: EEGfNIRSDataset,
    subject_id: int,
    session_idx: int,
    trial_length_s: float,
    eeg_channel_idx: int,
    fnirs_channel_idx: int,
    output_path: Path,
) -> Dict[str, Any]:
    eeg_raw = eeg_dataset.get_session_continuous_data(subject_id, session_idx, processed=False, normalized=False)
    eeg_processed = eeg_dataset.get_session_continuous_data(subject_id, session_idx, processed=True, normalized=True)
    fnirs_raw = fnirs_dataset.get_session_continuous_data(subject_id, session_idx, processed=False, normalized=False)
    fnirs_processed = fnirs_dataset.get_session_continuous_data(subject_id, session_idx, processed=True, normalized=True)

    eeg_regions = eeg_dataset.get_session_trial_regions(subject_id, session_idx, window_duration_s=trial_length_s)
    fnirs_regions = fnirs_dataset.get_session_trial_regions(subject_id, session_idx, window_duration_s=trial_length_s)

    eeg_time = [index / eeg_dataset.get_sample_rate() for index in range(eeg_raw.shape[1])]
    fnirs_time = [index / fnirs_dataset.get_sample_rate() for index in range(fnirs_raw.shape[1])]

    eeg_names = eeg_dataset.get_channel_names()
    fnirs_names = fnirs_dataset.get_channel_names()

    color_map = {
        0: '#2E86AB',
        1: '#E67E22',
    }

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=False)

    axes[0].plot(eeg_time, eeg_raw[eeg_channel_idx], color='#1f3a5f', linewidth=0.8)
    add_trial_spans(axes[0], eeg_regions, color_map)
    axes[0].set_title(f'EEG raw session | channel {eeg_names[eeg_channel_idx]}')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(eeg_time, eeg_processed[eeg_channel_idx], color='#A23B72', linewidth=0.8)
    add_trial_spans(axes[1], eeg_regions, color_map)
    axes[1].set_title(f'EEG filtered + session normalized | channel {eeg_names[eeg_channel_idx]}')
    axes[1].set_ylabel('z-score')
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fnirs_time, fnirs_raw[fnirs_channel_idx], color='#245c43', linewidth=0.8)
    add_trial_spans(axes[2], fnirs_regions, color_map)
    axes[2].set_title(f'fNIRS raw session | channel {fnirs_names[fnirs_channel_idx]}')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(fnirs_time, fnirs_processed[fnirs_channel_idx], color='#C0392B', linewidth=0.8)
    add_trial_spans(axes[3], fnirs_regions, color_map)
    axes[3].set_title(f'fNIRS filtered + session normalized | channel {fnirs_names[fnirs_channel_idx]}')
    axes[3].set_ylabel('z-score')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.25)

    legend_handles = [
        Patch(facecolor=color_map[0], alpha=0.18, label=str(eeg_regions[0]['label_name']) if eeg_regions else 'class 0'),
        Patch(facecolor=color_map[1], alpha=0.18, label=str(eeg_regions[1]['label_name']) if len(eeg_regions) > 1 else 'class 1'),
    ]
    axes[0].legend(handles=legend_handles, loc='upper right')

    fig.suptitle(
        f'Subject {subject_id} | session {session_idx} | shaded regions = 10s extracted trials from task onset',
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'figure': str(output_path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
        'subject_id': subject_id,
        'session_idx': session_idx,
        'trial_length_s': trial_length_s,
        'eeg_channel': eeg_names[eeg_channel_idx],
        'fnirs_channel': fnirs_names[fnirs_channel_idx],
        'num_regions': len(eeg_regions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize full-session trial extraction windows.')
    parser.add_argument('--eeg-config', default=DEFAULT_EEG_CONFIG)
    parser.add_argument('--fnirs-config', default=DEFAULT_FNIRS_CONFIG)
    parser.add_argument('--subject-id', type=int, default=1)
    parser.add_argument('--session-idx', type=int, default=0)
    parser.add_argument('--trial-length-s', type=float, default=10.0)
    parser.add_argument('--eeg-channel', default='')
    parser.add_argument('--fnirs-channel', default='')
    parser.add_argument('--output-dir', default='')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'logs' / 'session_visualizations' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    eeg_config = load_experiment_config(args.eeg_config)
    fnirs_config = load_experiment_config(args.fnirs_config)

    eeg_dataset = create_dataset(eeg_config, args.subject_id)
    fnirs_dataset = create_dataset(fnirs_config, args.subject_id)

    eeg_channel_idx = pick_channel_index(eeg_dataset.get_channel_names(), args.eeg_channel or None)
    fnirs_channel_idx = pick_channel_index(fnirs_dataset.get_channel_names(), args.fnirs_channel or None)

    figure_path = output_dir / f'subject{args.subject_id}_session{args.session_idx}_trial_windows.png'
    summary = plot_session_overview(
        eeg_dataset=eeg_dataset,
        fnirs_dataset=fnirs_dataset,
        subject_id=args.subject_id,
        session_idx=args.session_idx,
        trial_length_s=args.trial_length_s,
        eeg_channel_idx=eeg_channel_idx,
        fnirs_channel_idx=fnirs_channel_idx,
        output_path=figure_path,
    )

    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'[SessionViz] Saved figure: {figure_path}')
    print(f'[SessionViz] Saved summary: {summary_path}')


if __name__ == '__main__':
    main()