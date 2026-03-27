"""Visualize raw continuous EEG/fNIRS recordings on an aligned event timeline.

This tool is intended as a human-readable correctness check for dataset loading.
It plots:
1. An event track with EEG and fNIRS onsets on the same aligned timeline.
2. Raw EEG traces for selected channels.
3. Raw fNIRS traces for selected channels.

The current implementation supports datasets whose loader is already implemented.
At the moment this means EEG+NIRS Single-Trial.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_continuous_visualization_dataset, load_experiment_config, require_dataset_loader


DEFAULT_CONFIG = 'phase0plus/shared_labram_lag_warmstart_eeg_fnirs_30s_2s_cb512.yaml'
DEFAULT_EEG_CHANNELS = ('C3', 'Cz', 'C4')
DEFAULT_FNIRS_CHANNELS = ('AF7', 'AFF5', 'AFp7', 'AF5h')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize raw continuous EEG/fNIRS recordings with aligned event markers.')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Config path relative to experiments/configs')
    parser.add_argument('--subject-id', type=int, default=1, help='Subject id to visualize')
    parser.add_argument('--session-idx', type=int, default=0, help='Session index in the raw dataset file')
    parser.add_argument('--task-duration-s', type=float, default=10.0, help='Task window duration used for shaded spans')
    parser.add_argument('--align-mode', choices=('first_event', 'recording_start'), default='first_event')
    parser.add_argument('--eeg-channels', default=','.join(DEFAULT_EEG_CHANNELS), help='Comma separated channel names or indices')
    parser.add_argument('--fnirs-channels', default=','.join(DEFAULT_FNIRS_CHANNELS), help='Comma separated channel names or indices')
    parser.add_argument('--max-plot-points', type=int, default=4000, help='Maximum samples rendered per trace after decimation')
    parser.add_argument('--focus-trial-idx', type=int, default=0, help='Trial index used for local zoom inspection')
    parser.add_argument('--zoom-pre-s', type=float, default=4.0, help='Seconds shown before the focus event in the local zoom figure')
    parser.add_argument('--zoom-post-s', type=float, default=12.0, help='Seconds shown after the focus event in the local zoom figure')
    parser.add_argument('--output-dir', default='', help='Optional output directory')
    return parser.parse_args()


def resolve_normalization_config(data_cfg: Dict[str, Any]) -> Tuple[bool, str]:
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


def resolve_sample_rate_from_config(data_cfg: Dict[str, Any], modality: str, fallback: float) -> float:
    if modality == 'eeg':
        modality_cfg = data_cfg.get('eeg_preprocessing', data_cfg.get('preprocessing', {}))
    else:
        modality_cfg = data_cfg.get('fnirs_preprocessing', data_cfg.get('preprocessing', {}))

    for key in ('target_sampling_rate', 'resample_rate', 'sampling_rate', 'sample_rate'):
        value = modality_cfg.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(fallback)


def build_dataset(config: Dict[str, Any], modality: str, subject_id: int):
    data_cfg = config['data']
    normalize, normalization_mode = resolve_normalization_config(data_cfg)
    return create_continuous_visualization_dataset(
        data_cfg,
        modality,
        subject_id,
        normalize=normalize,
        normalization_mode=normalization_mode,
    )


def get_visualization_markers(dataset: Any, subject_id: int, session_idx: int) -> Dict[str, Any]:
    if hasattr(dataset, 'get_session_segmentation_markers'):
        return dataset.get_session_segmentation_markers(subject_id, session_idx)
    return dataset.get_session_markers(subject_id, session_idx)


def infer_region_boundaries(regions: List[Dict[str, Any]], total_duration_s: float) -> List[Dict[str, Any]]:
    if not regions:
        return []
    resolved: List[Dict[str, Any]] = []
    for index, region in enumerate(regions):
        next_onset_s = float(regions[index + 1]['onset_s']) if index + 1 < len(regions) else float(total_duration_s)
        updated = dict(region)
        updated['start_s'] = float(region.get('start_s', region['onset_s']))
        updated['end_s'] = max(updated['start_s'], next_onset_s)
        resolved.append(updated)
    return resolved


def parse_channel_spec(spec: str) -> List[str]:
    return [item.strip() for item in spec.split(',') if item.strip()]


def resolve_channel_indices(channel_names: Sequence[str], spec: str, fallback_names: Sequence[str]) -> List[int]:
    requested = parse_channel_spec(spec)
    if not requested:
        requested = list(fallback_names)

    indices: List[int] = []
    for item in requested:
        if item.isdigit():
            index = int(item)
            if 0 <= index < len(channel_names):
                indices.append(index)
            continue
        if item in channel_names:
            indices.append(channel_names.index(item))

    if indices:
        return list(dict.fromkeys(indices))

    fallback_indices = [idx for idx, name in enumerate(channel_names) if name in fallback_names]
    if fallback_indices:
        return fallback_indices
    return list(range(min(4, len(channel_names))))


def decimate_for_plot(time_axis: np.ndarray, signal: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if signal.shape[1] <= max_points:
        return time_axis, signal
    step = max(1, int(math.ceil(signal.shape[1] / max_points)))
    return time_axis[::step], signal[:, ::step]


def align_time_axis(
    n_samples: int,
    sample_rate: float,
    event_times_s: np.ndarray,
    align_mode: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    raw_time = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    if align_mode == 'recording_start' or event_times_s.size == 0:
        return raw_time, event_times_s.copy(), 0.0
    anchor = float(event_times_s[0])
    return raw_time - anchor, event_times_s - anchor, anchor


def extract_event_info(markers: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    event_times_s = np.asarray(markers['time'], dtype=np.float64) / 1000.0
    class_names_raw = markers.get('className')
    class_names = [str(name) for name in class_names_raw] if class_names_raw is not None else []

    if class_names and len(class_names) == len(event_times_s):
        unique_names = list(dict.fromkeys(class_names))
        name_to_index = {name: index for index, name in enumerate(unique_names)}
        labels = np.asarray([name_to_index[name] for name in class_names], dtype=int)
        return event_times_s, labels, unique_names

    labels = np.argmax(np.asarray(markers['y']), axis=0).astype(int)
    return event_times_s, labels, class_names


def label_name_for_index(label: int, class_names: Sequence[str]) -> str:
    if 0 <= label < len(class_names):
        return str(class_names[label])
    return str(label)


def build_sync_summary(
    eeg_event_times_s: np.ndarray,
    nirs_event_times_s: np.ndarray,
    eeg_labels: np.ndarray,
    nirs_labels: np.ndarray,
    eeg_class_names: Sequence[str],
    nirs_class_names: Sequence[str],
    align_mode: str,
    eeg_anchor_s: float,
    nirs_anchor_s: float,
) -> Dict[str, Any]:
    common_events = int(min(len(eeg_event_times_s), len(nirs_event_times_s)))
    eeg_common = eeg_event_times_s[:common_events]
    nirs_common = nirs_event_times_s[:common_events]
    residual_ms = (nirs_common - eeg_common) * 1000.0
    label_match = bool(np.array_equal(eeg_labels[:common_events], nirs_labels[:common_events]))
    eeg_label_names = [label_name_for_index(int(label), eeg_class_names) for label in eeg_labels[:common_events]]
    nirs_label_names = [label_name_for_index(int(label), nirs_class_names) for label in nirs_labels[:common_events]]
    label_name_match = eeg_label_names == nirs_label_names

    return {
        'align_mode': align_mode,
        'num_eeg_events': int(len(eeg_event_times_s)),
        'num_fnirs_events': int(len(nirs_event_times_s)),
        'num_common_events_compared': common_events,
        'eeg_alignment_anchor_s': float(eeg_anchor_s),
        'fnirs_alignment_anchor_s': float(nirs_anchor_s),
        'initial_offset_ms': float((nirs_anchor_s - eeg_anchor_s) * 1000.0),
        'residual_mean_ms': float(np.mean(residual_ms)) if common_events else None,
        'residual_std_ms': float(np.std(residual_ms)) if common_events else None,
        'residual_max_abs_ms': float(np.max(np.abs(residual_ms))) if common_events else None,
        'label_sequence_match': label_match,
        'label_index_match': label_match,
        'label_name_match': bool(label_name_match),
        'label_name_note': (
            'Class names may differ across modalities even when label indices are aligned.'
            if label_match and not label_name_match
            else None
        ),
        'eeg_label_names': eeg_label_names,
        'fnirs_label_names': nirs_label_names,
    }


def plot_event_track(
    ax: plt.Axes,
    eeg_events_s: np.ndarray,
    nirs_events_s: np.ndarray,
    eeg_labels: np.ndarray,
    eeg_class_names: Sequence[str],
    task_duration_s: float,
    eeg_regions: Optional[Sequence[Dict[str, Any]]] = None,
    nirs_regions: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    if eeg_regions:
        for region in eeg_regions:
            onset_s = float(region['onset_s'])
            label_name = str(region.get('label_name', region.get('label', 'segment')))
            ax.axvline(onset_s, color='#34495E', linewidth=0.8, alpha=0.3)
            ax.axvspan(float(region['start_s']), float(region['end_s']), ymin=0.56, ymax=0.88, color='#A9CCE3', alpha=0.18)
            ax.text(
                onset_s,
                1.07,
                label_name,
                rotation=90,
                va='bottom',
                ha='center',
                fontsize=8,
                color='#2C3E50',
            )
    elif eeg_events_s.size:
        for onset_s, label in zip(eeg_events_s, eeg_labels):
            ax.axvline(float(onset_s), color='#34495E', linewidth=0.8, alpha=0.3)
            ax.axvspan(float(onset_s), float(onset_s) + task_duration_s, color='#BDC3C7', alpha=0.12)
            ax.text(
                float(onset_s),
                1.07,
                label_name_for_index(int(label), eeg_class_names),
                rotation=90,
                va='bottom',
                ha='center',
                fontsize=8,
                color='#2C3E50',
            )

    if nirs_regions:
        for region in nirs_regions:
            ax.axvspan(float(region['start_s']), float(region['end_s']), ymin=0.12, ymax=0.44, color='#F5B7B1', alpha=0.18)

    if eeg_events_s.size:
        ax.scatter(eeg_events_s, np.full_like(eeg_events_s, 0.7), color='#1F77B4', s=26, label='EEG onset')
    if nirs_events_s.size:
        ax.scatter(nirs_events_s, np.full_like(nirs_events_s, 0.3), color='#D62728', s=30, marker='^', label='fNIRS onset')

    ax.set_yticks([0.3, 0.7])
    ax.set_yticklabels(['fNIRS', 'EEG'])
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, axis='x', alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')
    ax.set_title('Aligned event timeline')


def plot_stacked_signals(
    ax: plt.Axes,
    time_axis: np.ndarray,
    signal: np.ndarray,
    channel_names: Sequence[str],
    channel_indices: Sequence[int],
    color: str,
    title: str,
    ylabel: str,
) -> None:
    selected = signal[np.asarray(channel_indices, dtype=int), :]
    scale = np.median(np.std(selected, axis=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    offset_step = float(scale * 6.0)

    ytick_positions: List[float] = []
    ytick_labels: List[str] = []
    for order, channel_idx in enumerate(channel_indices):
        offset = order * offset_step
        trace = signal[channel_idx] + offset
        ax.plot(time_axis, trace, color=color, linewidth=0.9)
        ytick_positions.append(offset)
        ytick_labels.append(str(channel_names[channel_idx]))

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def create_alignment_figure(
    *,
    eeg_time_axis: np.ndarray,
    nirs_time_axis: np.ndarray,
    eeg_signal: np.ndarray,
    nirs_signal: np.ndarray,
    eeg_channel_names: Sequence[str],
    nirs_channel_names: Sequence[str],
    eeg_channel_indices: Sequence[int],
    nirs_channel_indices: Sequence[int],
    eeg_events_s: np.ndarray,
    nirs_events_s: np.ndarray,
    eeg_labels: np.ndarray,
    eeg_class_names: Sequence[str],
    task_duration_s: float,
    eeg_regions: Optional[Sequence[Dict[str, Any]]],
    nirs_regions: Optional[Sequence[Dict[str, Any]]],
    output_path: Path,
    subject_id: int,
    session_idx: int,
    sync_summary: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(20, 12),
        sharex=True,
        gridspec_kw={'height_ratios': [1.0, 3.2, 3.2]},
    )

    plot_event_track(
        axes[0],
        eeg_events_s,
        nirs_events_s,
        eeg_labels,
        eeg_class_names,
        task_duration_s,
        eeg_regions=eeg_regions,
        nirs_regions=nirs_regions,
    )

    plot_stacked_signals(
        axes[1],
        eeg_time_axis,
        eeg_signal,
        eeg_channel_names,
        eeg_channel_indices,
        color='#1F4E79',
        title='EEG raw continuous recording',
        ylabel='Channels',
    )
    plot_stacked_signals(
        axes[2],
        nirs_time_axis,
        nirs_signal,
        nirs_channel_names,
        nirs_channel_indices,
        color='#7A2E1F',
        title='fNIRS raw continuous recording',
        ylabel='Channels',
    )

    for event_s in eeg_events_s:
        axes[1].axvline(float(event_s), color='#566573', linewidth=0.7, alpha=0.25)
    for event_s in nirs_events_s:
        axes[2].axvline(float(event_s), color='#C0392B', linewidth=0.7, alpha=0.25)

    if eeg_regions:
        for region in eeg_regions:
            axes[1].axvspan(float(region['start_s']), float(region['end_s']), color='#A9CCE3', alpha=0.18)
    else:
        for onset_s in eeg_events_s:
            axes[1].axvspan(float(onset_s), float(onset_s) + task_duration_s, color='#A9CCE3', alpha=0.18)

    if nirs_regions:
        for region in nirs_regions:
            axes[2].axvspan(float(region['start_s']), float(region['end_s']), color='#F5B7B1', alpha=0.18)
    else:
        for onset_s in nirs_events_s:
            axes[2].axvspan(float(onset_s), float(onset_s) + task_duration_s, color='#F5B7B1', alpha=0.18)

    axes[2].set_xlabel('Aligned time (s)')

    residual = sync_summary.get('residual_max_abs_ms')
    residual_text = 'n/a' if residual is None else f'{residual:.2f} ms'
    fig.suptitle(
        (
            f'Subject {subject_id} | session {session_idx} | raw continuous alignment check\n'
            f"Initial EEG->fNIRS offset: {sync_summary['initial_offset_ms']:.2f} ms | "
            f"max residual after alignment: {residual_text} | "
            f"label sequence match: {sync_summary['label_sequence_match']}"
        ),
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def crop_time_window(
    time_axis: np.ndarray,
    signal: np.ndarray,
    start_s: float,
    end_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (time_axis >= float(start_s)) & (time_axis <= float(end_s))
    if not np.any(mask):
        nearest_index = int(np.argmin(np.abs(time_axis - ((start_s + end_s) * 0.5))))
        lo = max(0, nearest_index - 10)
        hi = min(signal.shape[1], nearest_index + 11)
        return time_axis[lo:hi], signal[:, lo:hi]
    return time_axis[mask], signal[:, mask]


def create_local_zoom_figure(
    *,
    eeg_time_axis: np.ndarray,
    nirs_time_axis: np.ndarray,
    eeg_signal: np.ndarray,
    nirs_signal: np.ndarray,
    eeg_channel_names: Sequence[str],
    nirs_channel_names: Sequence[str],
    eeg_channel_indices: Sequence[int],
    nirs_channel_indices: Sequence[int],
    eeg_events_s: np.ndarray,
    nirs_events_s: np.ndarray,
    eeg_labels: np.ndarray,
    eeg_class_names: Sequence[str],
    focus_trial_idx: int,
    zoom_pre_s: float,
    zoom_post_s: float,
    task_duration_s: float,
    eeg_regions: Optional[Sequence[Dict[str, Any]]],
    nirs_regions: Optional[Sequence[Dict[str, Any]]],
    output_path: Path,
    subject_id: int,
    session_idx: int,
) -> Dict[str, Any]:
    eeg_focus_regions = list(eeg_regions or [])
    nirs_focus_regions = list(nirs_regions or [])
    if eeg_focus_regions:
        focus_trial_idx = int(np.clip(focus_trial_idx, 0, len(eeg_focus_regions) - 1))
        focus_eeg_region = eeg_focus_regions[focus_trial_idx]
        focus_onset_s = float(focus_eeg_region['onset_s'])
        focus_label = int(focus_eeg_region.get('label', 0))
        focus_label_name = str(focus_eeg_region.get('label_name', focus_label))
        task_duration_s = float(focus_eeg_region['end_s']) - float(focus_eeg_region['start_s'])
    elif eeg_events_s.size == 0:
        raise ValueError('No EEG events are available for local zoom visualization.')
    else:
        focus_trial_idx = int(np.clip(focus_trial_idx, 0, len(eeg_events_s) - 1))
        focus_onset_s = float(eeg_events_s[focus_trial_idx])
        focus_label = int(eeg_labels[focus_trial_idx])
        focus_label_name = label_name_for_index(focus_label, eeg_class_names)

    focus_nirs_region = nirs_focus_regions[min(focus_trial_idx, len(nirs_focus_regions) - 1)] if nirs_focus_regions else None
    nirs_focus_onset_s = float(nirs_events_s[min(focus_trial_idx, len(nirs_events_s) - 1)]) if nirs_events_s.size else None
    window_start_s = focus_onset_s - float(zoom_pre_s)
    window_end_s = focus_onset_s + float(zoom_post_s)

    eeg_zoom_time, eeg_zoom_signal = crop_time_window(eeg_time_axis, eeg_signal, window_start_s, window_end_s)
    nirs_zoom_time, nirs_zoom_signal = crop_time_window(nirs_time_axis, nirs_signal, window_start_s, window_end_s)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [0.9, 2.8, 2.8]},
    )

    event_mask = (eeg_events_s >= window_start_s) & (eeg_events_s <= window_end_s)
    eeg_zoom_events = eeg_events_s[event_mask]
    eeg_zoom_labels = eeg_labels[event_mask]
    if nirs_events_s.size:
        nirs_event_mask = (nirs_events_s >= window_start_s) & (nirs_events_s <= window_end_s)
        nirs_zoom_events = nirs_events_s[nirs_event_mask]
    else:
        nirs_zoom_events = np.asarray([], dtype=np.float64)

    zoom_eeg_regions = None
    if eeg_focus_regions:
        zoom_eeg_regions = [
            region for region in eeg_focus_regions
            if float(region['end_s']) >= window_start_s and float(region['start_s']) <= window_end_s
        ]

    zoom_nirs_regions = None
    if nirs_focus_regions:
        zoom_nirs_regions = [
            region for region in nirs_focus_regions
            if float(region['end_s']) >= window_start_s and float(region['start_s']) <= window_end_s
        ]

    plot_event_track(
        axes[0],
        eeg_zoom_events,
        nirs_zoom_events,
        eeg_zoom_labels,
        eeg_class_names,
        task_duration_s,
        eeg_regions=zoom_eeg_regions,
        nirs_regions=zoom_nirs_regions,
    )
    axes[0].axvline(focus_onset_s, color='#111111', linewidth=1.2, linestyle='--', alpha=0.8)

    plot_stacked_signals(
        axes[1],
        eeg_zoom_time,
        eeg_zoom_signal,
        eeg_channel_names,
        eeg_channel_indices,
        color='#1F4E79',
        title='EEG raw local zoom',
        ylabel='Channels',
    )
    plot_stacked_signals(
        axes[2],
        nirs_zoom_time,
        nirs_zoom_signal,
        nirs_channel_names,
        nirs_channel_indices,
        color='#7A2E1F',
        title='fNIRS raw local zoom',
        ylabel='Channels',
    )

    for axis in axes[1:]:
        axis.axvline(focus_onset_s, color='#111111', linewidth=1.2, linestyle='--', alpha=0.8)
    if eeg_focus_regions:
        axes[1].axvspan(float(focus_eeg_region['start_s']), float(focus_eeg_region['end_s']), color='#A9CCE3', alpha=0.22)
    else:
        axes[1].axvspan(focus_onset_s, focus_onset_s + task_duration_s, color='#A9CCE3', alpha=0.22)

    if focus_nirs_region is not None:
        axes[2].axvspan(float(focus_nirs_region['start_s']), float(focus_nirs_region['end_s']), color='#F5B7B1', alpha=0.22)
    elif nirs_focus_onset_s is not None:
        axes[2].axvspan(nirs_focus_onset_s, nirs_focus_onset_s + task_duration_s, color='#F5B7B1', alpha=0.22)

    if nirs_focus_onset_s is not None:
        axes[2].axvline(nirs_focus_onset_s, color='#C0392B', linewidth=1.0, linestyle=':', alpha=0.7)

    axes[2].set_xlabel('Aligned time (s)')
    fig.suptitle(
        (
            f'Subject {subject_id} | session {session_idx} | local alignment zoom around trial {focus_trial_idx}\n'
            f'Focus label: {focus_label_name} | window: [{window_start_s:.2f}, {window_end_s:.2f}] s'
        ),
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=170, bbox_inches='tight')
    plt.close(fig)

    return {
        'focus_trial_idx': focus_trial_idx,
        'focus_event_time_s': focus_onset_s,
        'focus_fnirs_event_time_s': nirs_focus_onset_s,
        'focus_eeg_region_start_s': None if not eeg_focus_regions else float(focus_eeg_region['start_s']),
        'focus_eeg_region_end_s': None if not eeg_focus_regions else float(focus_eeg_region['end_s']),
        'focus_fnirs_region_start_s': None if focus_nirs_region is None else float(focus_nirs_region['start_s']),
        'focus_fnirs_region_end_s': None if focus_nirs_region is None else float(focus_nirs_region['end_s']),
        'focus_label_index': focus_label,
        'focus_label_name': focus_label_name,
        'window_start_s': window_start_s,
        'window_end_s': window_end_s,
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    data_cfg = config['data']
    require_dataset_loader(data_cfg['dataset'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = (Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'logs' / 'continuous_alignment' / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eeg_dataset = build_dataset(config, 'eeg', args.subject_id)
    nirs_dataset = build_dataset(config, 'fnirs', args.subject_id)

    eeg_signal = eeg_dataset.get_session_continuous_data(args.subject_id, args.session_idx, processed=False, normalized=False)
    nirs_signal = nirs_dataset.get_session_continuous_data(args.subject_id, args.session_idx, processed=False, normalized=False)

    eeg_markers = get_visualization_markers(eeg_dataset, args.subject_id, args.session_idx)
    nirs_markers = get_visualization_markers(nirs_dataset, args.subject_id, args.session_idx)

    eeg_event_times_s, eeg_labels, eeg_class_names = extract_event_info(eeg_markers)
    nirs_event_times_s, nirs_labels, nirs_class_names = extract_event_info(nirs_markers)

    eeg_time_axis, eeg_events_aligned_s, eeg_anchor_s = align_time_axis(
        eeg_signal.shape[1],
        eeg_dataset.get_sample_rate(),
        eeg_event_times_s,
        args.align_mode,
    )
    nirs_time_axis, nirs_events_aligned_s, nirs_anchor_s = align_time_axis(
        nirs_signal.shape[1],
        nirs_dataset.get_sample_rate(),
        nirs_event_times_s,
        args.align_mode,
    )

    eeg_time_axis, eeg_signal = decimate_for_plot(eeg_time_axis, eeg_signal, args.max_plot_points)
    nirs_time_axis, nirs_signal = decimate_for_plot(nirs_time_axis, nirs_signal, args.max_plot_points)

    eeg_channel_names = eeg_dataset.get_channel_names()
    nirs_channel_names = nirs_dataset.get_channel_names()
    eeg_channel_indices = resolve_channel_indices(eeg_channel_names, args.eeg_channels, DEFAULT_EEG_CHANNELS)
    nirs_channel_indices = resolve_channel_indices(nirs_channel_names, args.fnirs_channels, DEFAULT_FNIRS_CHANNELS)
    total_duration_s = max(
        float(eeg_signal.shape[1]) / float(max(eeg_dataset.get_sample_rate(), 1e-8)),
        float(nirs_signal.shape[1]) / float(max(nirs_dataset.get_sample_rate(), 1e-8)),
    )
    eeg_regions = eeg_dataset.get_session_trial_regions(
        args.subject_id,
        args.session_idx,
        window_duration_s=float(args.task_duration_s),
        offset_ms=0.0,
    )
    nirs_regions = nirs_dataset.get_session_trial_regions(
        args.subject_id,
        args.session_idx,
        window_duration_s=float(args.task_duration_s),
        offset_ms=0.0,
    )
    if data_cfg['dataset'] == 'simultaneous_eeg_nirs':
        eeg_regions = infer_region_boundaries(
            eeg_regions,
            float(eeg_signal.shape[1]) / float(max(eeg_dataset.get_sample_rate(), 1e-8)),
        )
        nirs_regions = infer_region_boundaries(
            nirs_regions,
            float(nirs_signal.shape[1]) / float(max(nirs_dataset.get_sample_rate(), 1e-8)),
        )

    sync_summary = build_sync_summary(
        eeg_event_times_s=eeg_events_aligned_s,
        nirs_event_times_s=nirs_events_aligned_s,
        eeg_labels=eeg_labels,
        nirs_labels=nirs_labels,
        eeg_class_names=eeg_class_names,
        nirs_class_names=nirs_class_names,
        align_mode=args.align_mode,
        eeg_anchor_s=eeg_anchor_s,
        nirs_anchor_s=nirs_anchor_s,
    )

    figure_path = output_dir / f'subject{args.subject_id}_session{args.session_idx}_continuous_alignment.png'
    create_alignment_figure(
        eeg_time_axis=eeg_time_axis,
        nirs_time_axis=nirs_time_axis,
        eeg_signal=eeg_signal,
        nirs_signal=nirs_signal,
        eeg_channel_names=eeg_channel_names,
        nirs_channel_names=nirs_channel_names,
        eeg_channel_indices=eeg_channel_indices,
        nirs_channel_indices=nirs_channel_indices,
        eeg_events_s=eeg_events_aligned_s,
        nirs_events_s=nirs_events_aligned_s,
        eeg_labels=eeg_labels,
        eeg_class_names=eeg_class_names,
        task_duration_s=float(args.task_duration_s),
        eeg_regions=eeg_regions,
        nirs_regions=nirs_regions,
        output_path=figure_path,
        subject_id=args.subject_id,
        session_idx=args.session_idx,
        sync_summary=sync_summary,
    )

    local_figure_path = output_dir / f'subject{args.subject_id}_session{args.session_idx}_continuous_alignment_local.png'
    local_summary = create_local_zoom_figure(
        eeg_time_axis=eeg_time_axis,
        nirs_time_axis=nirs_time_axis,
        eeg_signal=eeg_signal,
        nirs_signal=nirs_signal,
        eeg_channel_names=eeg_channel_names,
        nirs_channel_names=nirs_channel_names,
        eeg_channel_indices=eeg_channel_indices,
        nirs_channel_indices=nirs_channel_indices,
        eeg_events_s=eeg_events_aligned_s,
        nirs_events_s=nirs_events_aligned_s,
        eeg_labels=eeg_labels,
        eeg_class_names=eeg_class_names,
        focus_trial_idx=args.focus_trial_idx,
        zoom_pre_s=float(args.zoom_pre_s),
        zoom_post_s=float(args.zoom_post_s),
        task_duration_s=float(args.task_duration_s),
        eeg_regions=eeg_regions,
        nirs_regions=nirs_regions,
        output_path=local_figure_path,
        subject_id=args.subject_id,
        session_idx=args.session_idx,
    )

    try:
        figure_relative_path = str(figure_path.resolve().relative_to(PROJECT_ROOT)).replace('\\', '/')
    except ValueError:
        figure_relative_path = str(figure_path.resolve())

    try:
        local_figure_relative_path = str(local_figure_path.resolve().relative_to(PROJECT_ROOT)).replace('\\', '/')
    except ValueError:
        local_figure_relative_path = str(local_figure_path.resolve())

    summary = {
        'config': args.config,
        'dataset': data_cfg['dataset'],
        'subject_id': args.subject_id,
        'session_idx': args.session_idx,
        'align_mode': args.align_mode,
        'task_duration_s': float(args.task_duration_s),
        'figure': figure_relative_path,
        'local_figure': local_figure_relative_path,
        'selected_eeg_channels': [eeg_channel_names[idx] for idx in eeg_channel_indices],
        'selected_fnirs_channels': [nirs_channel_names[idx] for idx in nirs_channel_indices],
        'eeg_sample_rate_hz': float(eeg_dataset.get_sample_rate()),
        'fnirs_sample_rate_hz': float(nirs_dataset.get_sample_rate()),
        'eeg_segmentation_mode': getattr(eeg_dataset, 'segmentation_mode', 'trial'),
        'fnirs_segmentation_mode': getattr(nirs_dataset, 'segmentation_mode', 'trial'),
        'num_visualized_eeg_segments': len(eeg_regions),
        'num_visualized_fnirs_segments': len(nirs_regions),
        'local_zoom': local_summary,
        'sync_summary': sync_summary,
    }
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'[ContinuousAlignment] Saved figure: {figure_path}')
    print(f'[ContinuousAlignment] Saved local figure: {local_figure_path}')
    print(f'[ContinuousAlignment] Saved summary: {summary_path}')
    print('[ContinuousAlignment] Sync summary:')
    print(json.dumps(sync_summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()