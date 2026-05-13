"""Apply a Croce-style reduced particle filter to EEG-fNIRS event averages.

This script preserves the five-state neurovascular dynamics from Croce et al. (2017)
while adapting the observation model to the information available in the
EEG+NIRS Single-Trial dataset. The dataset exposes electrode/optode geometry but not
subject-specific EEG lead fields or optical Jacobians, so the forward model is reduced
to geometry-constrained low-rank sensor maps anchored to the task-relevant motor ROI.

Outputs include:
1. Estimated hidden state trajectories.
2. Raw versus reconstructed EEG and fNIRS projections.
3. Representative channel overlays for the reconstructed clean observations.
4. A markdown summary describing the exact adaptation and quantitative fit metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    EEGfNIRSDataset,
    build_channel_adjacency,
    canonicalize_channel_label,
    load_experiment_config,
    resolve_modality_preprocessing,
    strip_fnirs_chromophore_suffix,
)


DEFAULT_CONFIG = 'source_observation/phase1/default.yaml'
DEFAULT_PRE_S = 4.0
DEFAULT_POST_S = 16.0
DEFAULT_NUM_PARTICLES = 200
DEFAULT_COMMON_FS = 10.0
DEFAULT_TASK_DURATION_S = 10.0
DEFAULT_TASK_START_DELAY_S = 2.0
DEFAULT_EEG_BAND = (8.0, 30.0)
DEFAULT_EEG_ENVELOPE_LOWPASS = 2.0
DEFAULT_EXTINCTION = {
    'low_hbo': 0.35,
    'low_hb': 1.00,
    'high_hbo': 1.00,
    'high_hb': 0.25,
}
DEFAULT_MODEL_PARAMS = {
    'epsilon': 1.0,
    'kas': 0.41,
    'kaf': 0.65,
    'tau0': 2.0,
    'alpha': 0.32,
    'e0': 0.34,
    'lambda_r': 0.10,
}
DEFAULT_PROCESS_NOISE = np.asarray([0.06, 0.05, 0.04, 0.04, 0.18], dtype=np.float64)


@dataclass(frozen=True)
class FnirsPairIndex:
    base_names: List[str]
    low_indices: np.ndarray
    high_indices: np.ndarray
    base_positions_3d: np.ndarray


@dataclass(frozen=True)
class AggregateObservation:
    label: int
    label_name: str
    anchor_channel: str
    n_trials: int
    time_s: np.ndarray
    eeg_average: np.ndarray
    fnirs_low_average: np.ndarray
    fnirs_high_average: np.ndarray
    eeg_baseline_std: np.ndarray
    fnirs_low_baseline_std: np.ndarray
    fnirs_high_baseline_std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Croce-style reduced PF reconstruction for EEG-fNIRS event averages.')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Config path relative to experiments/configs')
    parser.add_argument('--subject-ids', default='', help='Comma-separated subject ids. Empty means all subjects from split config.')
    parser.add_argument('--pre-s', type=float, default=DEFAULT_PRE_S, help='Seconds before onset included in each event window')
    parser.add_argument('--post-s', type=float, default=DEFAULT_POST_S, help='Seconds after onset included in each event window')
    parser.add_argument('--common-fs', type=float, default=DEFAULT_COMMON_FS, help='Target rate for the slow neural-drive timeline')
    parser.add_argument('--num-particles', type=int, default=DEFAULT_NUM_PARTICLES, help='Bootstrap particle count')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for particle filtering')
    parser.add_argument('--output-dir', default='', help='Optional output directory')
    return parser.parse_args()


def parse_subject_ids(spec: str, config: Mapping[str, Any]) -> List[int]:
    if spec.strip():
        return [int(item.strip()) for item in spec.split(',') if item.strip()]

    split_cfg = config['data']['split']
    subjects: List[int] = []
    for key in ('train_subjects', 'val_subjects', 'test_subjects'):
        values = split_cfg.get(key, [])
        subjects.extend(int(value) for value in values)
    return sorted(dict.fromkeys(subjects))


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = PROJECT_ROOT / 'experiments' / 'results' / f'croce_pf_reconstruction_{timestamp}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_datasets(config: Mapping[str, Any], subject_ids: Sequence[int]) -> Tuple[EEGfNIRSDataset, EEGfNIRSDataset]:
    data_cfg = config['data']
    eeg_preprocessing = resolve_modality_preprocessing(data_cfg, 'eeg')
    fnirs_preprocessing = resolve_modality_preprocessing(data_cfg, 'fnirs')
    common_kwargs = {
        'data_root': data_cfg['data_root'],
        'subject_ids': list(subject_ids),
        'task': data_cfg.get('task', 'motor_imagery'),
        'window_samples': 1,
        'window_offset_ms': 0.0,
        'normalize': False,
        'normalization_mode': 'none',
        'exclude_eog': data_cfg.get('exclude_eog', True),
        'hbo_only': False,
        'hbr_only': False,
    }
    eeg_dataset = EEGfNIRSDataset(
        modality='eeg',
        preprocessing=eeg_preprocessing,
        **common_kwargs,
    )
    fnirs_dataset = EEGfNIRSDataset(
        modality='fnirs',
        preprocessing=fnirs_preprocessing,
        **common_kwargs,
    )
    return eeg_dataset, fnirs_dataset


def pair_fnirs_channels(channel_names: Sequence[str], channel_positions_3d: np.ndarray) -> FnirsPairIndex:
    low_map: Dict[str, int] = {}
    high_map: Dict[str, int] = {}
    ordered_bases: List[str] = []

    for index, name in enumerate(channel_names):
        lowered = str(name).lower()
        base_name = strip_fnirs_chromophore_suffix(name)
        base_key = canonicalize_channel_label(base_name)
        if base_key not in low_map and base_key not in high_map:
            ordered_bases.append(base_name)
        if lowered.endswith('lowwl'):
            low_map[base_key] = index
        elif lowered.endswith('highwl'):
            high_map[base_key] = index

    base_names: List[str] = []
    low_indices: List[int] = []
    high_indices: List[int] = []
    base_positions: List[np.ndarray] = []

    for base_name in ordered_bases:
        key = canonicalize_channel_label(base_name)
        if key not in low_map or key not in high_map:
            continue
        low_index = int(low_map[key])
        high_index = int(high_map[key])
        base_names.append(base_name)
        low_indices.append(low_index)
        high_indices.append(high_index)
        base_positions.append(channel_positions_3d[low_index])

    if not base_names:
        raise RuntimeError('Could not pair any lowWL/highWL fNIRS channels')

    return FnirsPairIndex(
        base_names=base_names,
        low_indices=np.asarray(low_indices, dtype=np.int64),
        high_indices=np.asarray(high_indices, dtype=np.int64),
        base_positions_3d=np.asarray(base_positions, dtype=np.float64),
    )


def build_spatial_geometry(
    data_root: str,
    eeg_channel_names: Sequence[str],
    fnirs_channel_names: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, int], FnirsPairIndex]:
    adjacency = build_channel_adjacency(
        dataset_id='eeg_fnirs_single_trial',
        data_root=data_root,
        eeg_channel_names=eeg_channel_names,
        fnirs_channel_names=fnirs_channel_names,
        reference_subject_id=1,
    )
    eeg_positions_3d = adjacency.eeg_positions_3d.astype(np.float64, copy=False)
    eeg_index = {canonicalize_channel_label(name): idx for idx, name in enumerate(eeg_channel_names)}
    fnirs_pairs = pair_fnirs_channels(fnirs_channel_names, adjacency.fnirs_channel_positions_3d)
    return eeg_positions_3d, eeg_index, fnirs_pairs


def gaussian_forward(anchor_position: np.ndarray, positions: np.ndarray, *, softness: float = 1.75) -> np.ndarray:
    distances = np.linalg.norm(positions - anchor_position.reshape(1, -1), axis=1)
    positive = distances[distances > 1e-8]
    if positive.size == 0:
        sigma = 1.0
    else:
        sigma = float(np.median(positive) * softness)
    sigma = max(sigma, 1e-6)
    weights = np.exp(-0.5 * np.square(distances / sigma))
    norm = np.linalg.norm(weights)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.ones_like(weights) / np.sqrt(float(weights.size))
    return weights / norm


def make_bandpass_sos(sample_rate: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyquist = sample_rate * 0.5
    return butter(order, [low_hz / nyquist, high_hz / nyquist], btype='bandpass', output='sos')


def compute_eeg_envelope_session(eeg_session: np.ndarray, eeg_fs: float, common_fs: float) -> np.ndarray:
    sos = make_bandpass_sos(eeg_fs, DEFAULT_EEG_BAND[0], DEFAULT_EEG_BAND[1])
    filtered = sosfiltfilt(sos, eeg_session, axis=1)
    envelope = np.abs(hilbert(filtered, axis=1))
    envelope_sos = butter(4, DEFAULT_EEG_ENVELOPE_LOWPASS / (0.5 * eeg_fs), btype='lowpass', output='sos')
    envelope = sosfiltfilt(envelope_sos, envelope, axis=1)
    if abs(eeg_fs - common_fs) < 1e-6:
        return envelope.astype(np.float64, copy=False)
    if abs((eeg_fs / common_fs) - round(eeg_fs / common_fs)) < 1e-6:
        down = int(round(eeg_fs / common_fs))
        return resample_poly(envelope, up=1, down=down, axis=1).astype(np.float64, copy=False)

    source_time = np.arange(envelope.shape[1], dtype=np.float64) / float(eeg_fs)
    duration_s = source_time[-1]
    target_samples = int(round(duration_s * common_fs)) + 1
    target_time = np.arange(target_samples, dtype=np.float64) / float(common_fs)
    resampled = np.empty((envelope.shape[0], target_samples), dtype=np.float64)
    for channel_idx in range(envelope.shape[0]):
        resampled[channel_idx] = np.interp(target_time, source_time, envelope[channel_idx])
    return resampled


def extract_window(signal: np.ndarray, start: int, length: int) -> Optional[np.ndarray]:
    end = start + length
    if start < 0 or end > signal.shape[1]:
        return None
    return signal[:, start:end].copy()


def compute_fnirs_optical_density(window: np.ndarray, baseline_samples: int) -> np.ndarray:
    baseline = np.mean(np.clip(window[:, :baseline_samples], 1e-8, None), axis=1, keepdims=True)
    od = -np.log(np.clip(window, 1e-8, None) / np.clip(baseline, 1e-8, None))
    od = od - np.mean(od[:, :baseline_samples], axis=1, keepdims=True)
    return od


def compute_relative_eeg_change(window: np.ndarray, baseline_samples: int) -> np.ndarray:
    baseline = np.mean(np.clip(window[:, :baseline_samples], 1e-8, None), axis=1, keepdims=True)
    relative = np.log(np.clip(window, 1e-8, None) / np.clip(baseline, 1e-8, None))
    relative = relative - np.mean(relative[:, :baseline_samples], axis=1, keepdims=True)
    return relative


def infer_anchor_channel(label_name: str, label: int) -> str:
    lowered = label_name.lower()
    if 'left' in lowered:
        return 'C4'
    if 'right' in lowered:
        return 'C3'
    return 'C4' if int(label) == 0 else 'C3'


def resolve_anchor_channel(label_name: str, label: int, eeg_channel_names: Sequence[str]) -> str:
    available = {canonicalize_channel_label(name): str(name) for name in eeg_channel_names}
    ideal = infer_anchor_channel(label_name, label)
    if canonicalize_channel_label(ideal) in available:
        return available[canonicalize_channel_label(ideal)]

    if canonicalize_channel_label(ideal) == canonicalize_channel_label('C4'):
        candidates = ('FCC4h', 'CCP4h', 'P4', 'T8', 'Cz')
    else:
        candidates = ('FCC3h', 'CCP3h', 'P3', 'T7', 'Cz')

    for candidate in candidates:
        key = canonicalize_channel_label(candidate)
        if key in available:
            return available[key]
    raise KeyError(f'Could not resolve a motor anchor for label {label_name!r} from current EEG channels')


def safe_label_name(class_names: Sequence[str], label: int) -> str:
    if 0 <= int(label) < len(class_names):
        return str(class_names[int(label)])
    return f'label_{int(label)}'


def baseline_std(window: np.ndarray, baseline_samples: int, floor: float) -> np.ndarray:
    return np.maximum(window[:, :baseline_samples].std(axis=1), floor)


def accumulate_label_averages(
    eeg_dataset: EEGfNIRSDataset,
    fnirs_dataset: EEGfNIRSDataset,
    subject_ids: Sequence[int],
    pre_s: float,
    post_s: float,
    common_fs: float,
) -> Dict[int, AggregateObservation]:
    eeg_fs = float(eeg_dataset.get_sample_rate())
    fnirs_fs = float(fnirs_dataset.get_sample_rate())
    if abs(fnirs_fs - common_fs) > 1e-6:
        raise ValueError(f'This analysis expects fNIRS sampling to match common_fs, got {fnirs_fs} vs {common_fs}')

    pre_samples = int(round(pre_s * common_fs))
    post_samples = int(round(post_s * common_fs))
    total_samples = pre_samples + post_samples
    time_s = np.arange(total_samples, dtype=np.float64) / float(common_fs) - float(pre_s)

    label_sums: Dict[int, Dict[str, Any]] = {}

    for subject_id in subject_ids:
        for session_idx in eeg_dataset.session_indices:
            eeg_markers = eeg_dataset.get_session_markers(subject_id, session_idx)
            fnirs_markers = fnirs_dataset.get_session_markers(subject_id, session_idx)
            eeg_labels = np.argmax(np.asarray(eeg_markers['y']), axis=0).astype(int)
            fnirs_labels = np.argmax(np.asarray(fnirs_markers['y']), axis=0).astype(int)
            if not np.array_equal(eeg_labels, fnirs_labels):
                continue

            raw_class_names = eeg_markers.get('className')
            if raw_class_names is None:
                class_names = []
            else:
                class_names = [str(name) for name in np.asarray(raw_class_names).tolist()]

            eeg_session = eeg_dataset.get_session_continuous_data(subject_id, session_idx, processed=True, normalized=False)
            eeg_envelope = compute_eeg_envelope_session(eeg_session, eeg_fs=eeg_fs, common_fs=common_fs)
            fnirs_session = fnirs_dataset.get_session_continuous_data(subject_id, session_idx, processed=True, normalized=False)

            eeg_onsets = np.asarray(eeg_markers['time'], dtype=np.float64) / 1000.0
            fnirs_onsets = np.asarray(fnirs_markers['time'], dtype=np.float64) / 1000.0

            for trial_idx, label in enumerate(eeg_labels):
                eeg_center = int(round((eeg_onsets[trial_idx] + DEFAULT_TASK_START_DELAY_S) * common_fs))
                fnirs_center = int(round((fnirs_onsets[trial_idx] + DEFAULT_TASK_START_DELAY_S) * common_fs))
                eeg_window = extract_window(eeg_envelope, eeg_center - pre_samples, total_samples)
                fnirs_window_full = extract_window(fnirs_session, fnirs_center - pre_samples, total_samples)
                if eeg_window is None or fnirs_window_full is None:
                    continue

                eeg_window = compute_relative_eeg_change(eeg_window, baseline_samples=pre_samples)
                fnirs_window = compute_fnirs_optical_density(fnirs_window_full, baseline_samples=pre_samples)

                entry = label_sums.setdefault(
                    int(label),
                    {
                        'label_name': safe_label_name(class_names, int(label)),
                        'eeg_sum': np.zeros_like(eeg_window, dtype=np.float64),
                        'fnirs_sum': np.zeros_like(fnirs_window, dtype=np.float64),
                        'eeg_baseline_std_sum': np.zeros(eeg_window.shape[0], dtype=np.float64),
                        'fnirs_baseline_std_sum': np.zeros(fnirs_window.shape[0], dtype=np.float64),
                        'n_trials': 0,
                    },
                )
                entry['eeg_sum'] += eeg_window
                entry['fnirs_sum'] += fnirs_window
                entry['eeg_baseline_std_sum'] += baseline_std(eeg_window, pre_samples, floor=1e-6)
                entry['fnirs_baseline_std_sum'] += baseline_std(fnirs_window, pre_samples, floor=1e-6)
                entry['n_trials'] += 1

    if not label_sums:
        raise RuntimeError('No aligned trials were accumulated for Croce-style reconstruction')

    observations: Dict[int, AggregateObservation] = {}
    for label, values in label_sums.items():
        trial_count = int(values['n_trials'])
        fnirs_average = values['fnirs_sum'] / float(trial_count)
        observations[label] = AggregateObservation(
            label=int(label),
            label_name=str(values['label_name']),
            anchor_channel=infer_anchor_channel(str(values['label_name']), int(label)),
            n_trials=trial_count,
            time_s=time_s,
            eeg_average=values['eeg_sum'] / float(trial_count),
            fnirs_low_average=np.empty((0, total_samples), dtype=np.float64),
            fnirs_high_average=np.empty((0, total_samples), dtype=np.float64),
            eeg_baseline_std=values['eeg_baseline_std_sum'] / float(trial_count),
            fnirs_low_baseline_std=np.empty(0, dtype=np.float64),
            fnirs_high_baseline_std=np.empty(0, dtype=np.float64),
        )

    return observations


def split_fnirs_pairs(
    observation: AggregateObservation,
    fnirs_pairs: FnirsPairIndex,
) -> AggregateObservation:
    fnirs_low = observation.fnirs_low_average
    if fnirs_low.size != 0:
        return observation
    low_average = observation.eeg_average
    raise RuntimeError('split_fnirs_pairs must be called on raw aggregate storage before finalization')


def finalize_label_averages(
    raw_aggregates: Dict[int, AggregateObservation],
    fnirs_pairs: FnirsPairIndex,
) -> Dict[int, AggregateObservation]:
    finalized: Dict[int, AggregateObservation] = {}
    for label, aggregate in raw_aggregates.items():
        fnirs_average = aggregate.fnirs_low_average
        if fnirs_average.size == 0:
            raise RuntimeError('Expected raw fNIRS aggregate to be attached before finalization')
        low_average = fnirs_average[fnirs_pairs.low_indices]
        high_average = fnirs_average[fnirs_pairs.high_indices]
        fnirs_low_std = aggregate.fnirs_low_baseline_std[fnirs_pairs.low_indices]
        fnirs_high_std = aggregate.fnirs_low_baseline_std[fnirs_pairs.high_indices]
        finalized[label] = AggregateObservation(
            label=aggregate.label,
            label_name=aggregate.label_name,
            anchor_channel=aggregate.anchor_channel,
            n_trials=aggregate.n_trials,
            time_s=aggregate.time_s,
            eeg_average=aggregate.eeg_average,
            fnirs_low_average=low_average,
            fnirs_high_average=high_average,
            eeg_baseline_std=aggregate.eeg_baseline_std,
            fnirs_low_baseline_std=fnirs_low_std,
            fnirs_high_baseline_std=fnirs_high_std,
        )
    return finalized


def aggregate_observations(
    eeg_dataset: EEGfNIRSDataset,
    fnirs_dataset: EEGfNIRSDataset,
    subject_ids: Sequence[int],
    pre_s: float,
    post_s: float,
    common_fs: float,
) -> Dict[int, AggregateObservation]:
    eeg_fs = float(eeg_dataset.get_sample_rate())
    fnirs_fs = float(fnirs_dataset.get_sample_rate())
    if abs(fnirs_fs - common_fs) > 1e-6:
        raise ValueError(f'This analysis expects fNIRS sampling to match common_fs, got {fnirs_fs} vs {common_fs}')

    pre_samples = int(round(pre_s * common_fs))
    post_samples = int(round(post_s * common_fs))
    total_samples = pre_samples + post_samples
    time_s = np.arange(total_samples, dtype=np.float64) / float(common_fs) - float(pre_s)

    label_sums: Dict[int, Dict[str, Any]] = {}

    for subject_id in subject_ids:
        for session_idx in eeg_dataset.session_indices:
            eeg_markers = eeg_dataset.get_session_markers(subject_id, session_idx)
            fnirs_markers = fnirs_dataset.get_session_markers(subject_id, session_idx)
            eeg_labels = np.argmax(np.asarray(eeg_markers['y']), axis=0).astype(int)
            fnirs_labels = np.argmax(np.asarray(fnirs_markers['y']), axis=0).astype(int)
            if not np.array_equal(eeg_labels, fnirs_labels):
                continue

            raw_class_names = eeg_markers.get('className')
            if raw_class_names is None:
                class_names = []
            else:
                class_names = [str(name) for name in np.asarray(raw_class_names).tolist()]

            eeg_session = eeg_dataset.get_session_continuous_data(subject_id, session_idx, processed=True, normalized=False)
            eeg_envelope = compute_eeg_envelope_session(eeg_session, eeg_fs=eeg_fs, common_fs=common_fs)
            fnirs_session = fnirs_dataset.get_session_continuous_data(subject_id, session_idx, processed=True, normalized=False)

            eeg_onsets = np.asarray(eeg_markers['time'], dtype=np.float64) / 1000.0
            fnirs_onsets = np.asarray(fnirs_markers['time'], dtype=np.float64) / 1000.0

            for trial_idx, label in enumerate(eeg_labels):
                eeg_center = int(round(eeg_onsets[trial_idx] * common_fs))
                fnirs_center = int(round(fnirs_onsets[trial_idx] * common_fs))
                eeg_window = extract_window(eeg_envelope, eeg_center - pre_samples, total_samples)
                fnirs_window_full = extract_window(fnirs_session, fnirs_center - pre_samples, total_samples)
                if eeg_window is None or fnirs_window_full is None:
                    continue

                eeg_window = eeg_window - np.mean(eeg_window[:, :pre_samples], axis=1, keepdims=True)
                fnirs_window = compute_fnirs_optical_density(fnirs_window_full, baseline_samples=pre_samples)

                entry = label_sums.setdefault(
                    int(label),
                    {
                        'label_name': safe_label_name(class_names, int(label)),
                        'eeg_sum': np.zeros_like(eeg_window, dtype=np.float64),
                        'fnirs_sum': np.zeros_like(fnirs_window, dtype=np.float64),
                        'eeg_baseline_std_sum': np.zeros(eeg_window.shape[0], dtype=np.float64),
                        'fnirs_baseline_std_sum': np.zeros(fnirs_window.shape[0], dtype=np.float64),
                        'n_trials': 0,
                    },
                )
                entry['eeg_sum'] += eeg_window
                entry['fnirs_sum'] += fnirs_window
                entry['eeg_baseline_std_sum'] += baseline_std(eeg_window, pre_samples, floor=1e-6)
                entry['fnirs_baseline_std_sum'] += baseline_std(fnirs_window, pre_samples, floor=1e-6)
                entry['n_trials'] += 1

    aggregates: Dict[int, AggregateObservation] = {}
    for label, values in label_sums.items():
        trial_count = int(values['n_trials'])
        aggregates[label] = AggregateObservation(
            label=int(label),
            label_name=str(values['label_name']),
            anchor_channel=infer_anchor_channel(str(values['label_name']), int(label)),
            n_trials=trial_count,
            time_s=time_s,
            eeg_average=values['eeg_sum'] / float(trial_count),
            fnirs_low_average=values['fnirs_sum'] / float(trial_count),
            fnirs_high_average=np.empty((0, total_samples), dtype=np.float64),
            eeg_baseline_std=values['eeg_baseline_std_sum'] / float(trial_count),
            fnirs_low_baseline_std=values['fnirs_baseline_std_sum'] / float(trial_count),
            fnirs_high_baseline_std=np.empty(0, dtype=np.float64),
        )

    if not aggregates:
        raise RuntimeError('No label-wise event averages could be formed')
    return finalize_label_averages(aggregates, fnirs_pairs=pair_fnirs_channels(fnirs_dataset.get_channel_names(), build_channel_adjacency(
        dataset_id='eeg_fnirs_single_trial',
        data_root=fnirs_dataset.data_root,
        eeg_channel_names=eeg_dataset.get_channel_names(),
        fnirs_channel_names=fnirs_dataset.get_channel_names(),
        reference_subject_id=1,
    ).fnirs_channel_positions_3d.astype(np.float64, copy=False)))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_particles = weights.shape[0]
    positions = (rng.random() + np.arange(n_particles, dtype=np.float64)) / float(n_particles)
    cumulative = np.cumsum(weights)
    indices = np.zeros(n_particles, dtype=np.int64)

    i = 0
    j = 0
    while i < n_particles:
        if positions[i] <= cumulative[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


def evolve_particles(
    particles: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    process_noise: np.ndarray,
    model_params: Mapping[str, float],
) -> np.ndarray:
    state = particles.copy()
    s = state[:, 0]
    f = np.clip(state[:, 1], 1e-4, None)
    hbo = np.clip(state[:, 2], 1e-4, None)
    hb = np.clip(state[:, 3], 1e-4, None)
    r = state[:, 4]

    epsilon = float(model_params['epsilon'])
    kas = float(model_params['kas'])
    kaf = float(model_params['kaf'])
    tau0 = float(model_params['tau0'])
    alpha = float(model_params['alpha'])
    e0 = float(model_params['e0'])
    lambda_r = float(model_params['lambda_r'])

    extraction = (1.0 - np.power(1.0 - e0, 1.0 / np.clip(f, 1e-4, None))) / max(e0, 1e-6)
    hbo_term = np.power(np.clip(hbo, 1e-4, None), 1.0 / alpha)
    hb_term = hb * np.power(np.clip(hbo, 1e-4, None), (1.0 / alpha) - 1.0)

    derivatives = np.stack(
        [
            epsilon * r - kas * s - kaf * (f - 1.0),
            s,
            (f - hbo_term) / tau0,
            (f * extraction - hb_term) / tau0,
            -lambda_r * r,
        ],
        axis=1,
    )
    noise = rng.normal(loc=0.0, scale=process_noise.reshape(1, -1) * np.sqrt(dt), size=state.shape)
    state = state + dt * derivatives + noise
    state[:, 1:4] = np.clip(state[:, 1:4], 1e-4, None)
    return state


def state_to_predictions(
    particles: np.ndarray,
    eeg_forward: np.ndarray,
    fnirs_forward: np.ndarray,
    extinction: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = particles[:, 4]
    hbo_delta = particles[:, 2] - 1.0
    hb_delta = particles[:, 3] - 1.0
    eeg_pred = r[:, None] * eeg_forward.reshape(1, -1)
    low_scalar = float(extinction['low_hbo']) * hbo_delta + float(extinction['low_hb']) * hb_delta
    high_scalar = float(extinction['high_hbo']) * hbo_delta + float(extinction['high_hb']) * hb_delta
    low_pred = low_scalar[:, None] * fnirs_forward.reshape(1, -1)
    high_pred = high_scalar[:, None] * fnirs_forward.reshape(1, -1)
    return eeg_pred, low_pred, high_pred


def run_particle_filter(
    observation: AggregateObservation,
    eeg_forward: np.ndarray,
    fnirs_forward: np.ndarray,
    *,
    num_particles: int,
    seed: int,
    extinction: Mapping[str, float],
    model_params: Mapping[str, float],
    process_noise: np.ndarray,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    time_s = observation.time_s
    dt = float(np.median(np.diff(time_s)))
    eeg_obs = observation.eeg_average
    low_obs = observation.fnirs_low_average
    high_obs = observation.fnirs_high_average
    eeg_sigma = np.maximum(observation.eeg_baseline_std, 1e-4)
    low_sigma = np.maximum(observation.fnirs_low_baseline_std, 1e-4)
    high_sigma = np.maximum(observation.fnirs_high_baseline_std, 1e-4)

    particles = np.zeros((num_particles, 5), dtype=np.float64)
    particles[:, 1] = rng.normal(loc=1.0, scale=0.04, size=num_particles)
    particles[:, 2] = rng.normal(loc=1.0, scale=0.04, size=num_particles)
    particles[:, 3] = rng.normal(loc=1.0, scale=0.04, size=num_particles)
    particles[:, 0] = rng.normal(loc=0.0, scale=0.08, size=num_particles)
    particles[:, 4] = rng.normal(loc=0.0, scale=0.12, size=num_particles)
    particles[:, 1:4] = np.clip(particles[:, 1:4], 1e-4, None)
    weights = np.full(num_particles, 1.0 / float(num_particles), dtype=np.float64)

    state_estimates = np.zeros((5, time_s.shape[0]), dtype=np.float64)
    ess_trace = np.zeros(time_s.shape[0], dtype=np.float64)

    for time_idx in range(time_s.shape[0]):
        particles = evolve_particles(particles, dt=dt, rng=rng, process_noise=process_noise, model_params=model_params)
        eeg_pred, low_pred, high_pred = state_to_predictions(particles, eeg_forward, fnirs_forward, extinction)

        eeg_residual = (eeg_obs[:, time_idx][None, :] - eeg_pred) / eeg_sigma.reshape(1, -1)
        low_residual = (low_obs[:, time_idx][None, :] - low_pred) / low_sigma.reshape(1, -1)
        high_residual = (high_obs[:, time_idx][None, :] - high_pred) / high_sigma.reshape(1, -1)

        log_likelihood = -0.5 * (
            np.mean(np.square(eeg_residual), axis=1)
            + np.mean(np.square(low_residual), axis=1)
            + np.mean(np.square(high_residual), axis=1)
        )
        log_weights = np.log(np.clip(weights, 1e-300, None)) + log_likelihood
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / np.clip(weights.sum(), 1e-12, None)

        state_estimates[:, time_idx] = np.sum(particles * weights.reshape(-1, 1), axis=0)
        ess = 1.0 / np.sum(np.square(weights))
        ess_trace[time_idx] = ess

        if ess < (0.5 * num_particles):
            indices = systematic_resample(weights, rng)
            particles = particles[indices]
            weights.fill(1.0 / float(num_particles))

    hbo_delta = state_estimates[2] - 1.0
    hb_delta = state_estimates[3] - 1.0
    recon_eeg = eeg_forward.reshape(-1, 1) * state_estimates[4].reshape(1, -1)
    recon_low = fnirs_forward.reshape(-1, 1) * (
        float(extinction['low_hbo']) * hbo_delta + float(extinction['low_hb']) * hb_delta
    ).reshape(1, -1)
    recon_high = fnirs_forward.reshape(-1, 1) * (
        float(extinction['high_hbo']) * hbo_delta + float(extinction['high_hb']) * hb_delta
    ).reshape(1, -1)

    return {
        'states': state_estimates,
        'ess_trace': ess_trace,
        'recon_eeg': recon_eeg,
        'recon_low': recon_low,
        'recon_high': recon_high,
    }


def weighted_projection(signal: np.ndarray, forward: np.ndarray) -> np.ndarray:
    return np.sum(signal * forward.reshape(-1, 1), axis=0)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError('Correlation inputs must have the same shape')
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def channelwise_correlations(raw: np.ndarray, recon: np.ndarray) -> np.ndarray:
    values = np.full(raw.shape[0], np.nan, dtype=np.float64)
    for index in range(raw.shape[0]):
        values[index] = safe_corr(raw[index], recon[index])
    return values


def top_indices(weights: np.ndarray, count: int = 3) -> np.ndarray:
    count = min(int(count), weights.shape[0])
    return np.argsort(weights)[-count:][::-1]


def plot_states_and_projections(
    output_path: Path,
    observation: AggregateObservation,
    result: Mapping[str, Any],
    eeg_forward: np.ndarray,
    fnirs_forward: np.ndarray,
) -> None:
    time_s = observation.time_s
    states = np.asarray(result['states'])
    recon_eeg = np.asarray(result['recon_eeg'])
    recon_low = np.asarray(result['recon_low'])
    recon_high = np.asarray(result['recon_high'])

    raw_eeg_proj = weighted_projection(observation.eeg_average, eeg_forward)
    recon_eeg_proj = weighted_projection(recon_eeg, eeg_forward)
    raw_low_proj = weighted_projection(observation.fnirs_low_average, fnirs_forward)
    recon_low_proj = weighted_projection(recon_low, fnirs_forward)
    raw_high_proj = weighted_projection(observation.fnirs_high_average, fnirs_forward)
    recon_high_proj = weighted_projection(recon_high, fnirs_forward)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(time_s, states[4], label='r(t) neural drive', linewidth=1.8)
    axes[0].plot(time_s, states[0], label='s(t) vaso signal', linewidth=1.2)
    axes[0].plot(time_s, states[1] - 1.0, label='f(t) - 1', linewidth=1.2)
    axes[0].plot(time_s, states[2] - 1.0, label='HbO(t) - 1', linewidth=1.2)
    axes[0].plot(time_s, states[3] - 1.0, label='Hb(t) - 1', linewidth=1.2)
    axes[0].axvspan(0.0, DEFAULT_TASK_DURATION_S, color='#D6EAF8', alpha=0.35)
    axes[0].axvline(0.0, color='black', linewidth=0.8, alpha=0.7)
    axes[0].set_title(f'{observation.label_name}: estimated Croce-style hidden states')
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(time_s, raw_eeg_proj, label='raw EEG projection', color='#1F77B4', linewidth=1.8)
    axes[1].plot(time_s, recon_eeg_proj, label='reconstructed EEG projection', color='#D62728', linewidth=1.6)
    axes[1].axvspan(0.0, DEFAULT_TASK_DURATION_S, color='#D6EAF8', alpha=0.35)
    axes[1].axvline(0.0, color='black', linewidth=0.8, alpha=0.7)
    axes[1].set_ylabel('EEG envelope')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(time_s, raw_high_proj, label='raw highWL projection', color='#2CA02C', linewidth=1.8)
    axes[2].plot(time_s, recon_high_proj, label='reconstructed highWL projection', color='#FF7F0E', linewidth=1.6)
    axes[2].plot(time_s, raw_low_proj, label='raw lowWL projection', color='#9467BD', linewidth=1.8)
    axes[2].plot(time_s, recon_low_proj, label='reconstructed lowWL projection', color='#8C564B', linewidth=1.6)
    axes[2].axvspan(0.0, DEFAULT_TASK_DURATION_S, color='#D6EAF8', alpha=0.35)
    axes[2].axvline(0.0, color='black', linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel('fNIRS optical density')
    axes[2].set_xlabel('Time relative to onset (s)')
    axes[2].legend(loc='upper right', ncol=2)
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_channel_overlays(
    output_path: Path,
    observation: AggregateObservation,
    result: Mapping[str, Any],
    eeg_forward: np.ndarray,
    fnirs_forward: np.ndarray,
    eeg_channel_names: Sequence[str],
    fnirs_base_names: Sequence[str],
) -> None:
    time_s = observation.time_s
    recon_eeg = np.asarray(result['recon_eeg'])
    recon_low = np.asarray(result['recon_low'])
    recon_high = np.asarray(result['recon_high'])
    eeg_indices = top_indices(np.abs(eeg_forward), count=3)
    fnirs_indices = top_indices(np.abs(fnirs_forward), count=3)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    for index in eeg_indices:
        axes[0].plot(time_s, observation.eeg_average[index], linewidth=1.3, alpha=0.75, label=f'raw {eeg_channel_names[index]}')
        axes[0].plot(time_s, recon_eeg[index], linewidth=1.6, linestyle='--', label=f'recon {eeg_channel_names[index]}')
    axes[0].axvspan(0.0, DEFAULT_TASK_DURATION_S, color='#D6EAF8', alpha=0.35)
    axes[0].set_title(f'{observation.label_name}: top EEG channel overlays')
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].grid(True, alpha=0.2)

    for index in fnirs_indices:
        axes[1].plot(time_s, observation.fnirs_high_average[index], linewidth=1.3, alpha=0.75, label=f'raw high {fnirs_base_names[index]}')
        axes[1].plot(time_s, recon_high[index], linewidth=1.6, linestyle='--', label=f'recon high {fnirs_base_names[index]}')
    axes[1].axvspan(0.0, DEFAULT_TASK_DURATION_S, color='#D6EAF8', alpha=0.35)
    axes[1].set_title('Top highWL channel overlays')
    axes[1].legend(loc='upper right', ncol=2)
    axes[1].grid(True, alpha=0.2)

    for index in fnirs_indices:
        axes[2].plot(time_s, observation.fnirs_low_average[index], linewidth=1.3, alpha=0.75, label=f'raw low {fnirs_base_names[index]}')
        axes[2].plot(time_s, recon_low[index], linewidth=1.6, linestyle='--', label=f'recon low {fnirs_base_names[index]}')
    axes[2].axvspan(0.0, DEFAULT_TASK_DURATION_S, color='#D6EAF8', alpha=0.35)
    axes[2].set_title('Top lowWL channel overlays')
    axes[2].set_xlabel('Time relative to onset (s)')
    axes[2].legend(loc='upper right', ncol=2)
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_result(
    observation: AggregateObservation,
    result: Mapping[str, Any],
    eeg_forward: np.ndarray,
    fnirs_forward: np.ndarray,
    eeg_channel_names: Sequence[str],
    fnirs_base_names: Sequence[str],
) -> Dict[str, Any]:
    recon_eeg = np.asarray(result['recon_eeg'])
    recon_low = np.asarray(result['recon_low'])
    recon_high = np.asarray(result['recon_high'])
    state_estimates = np.asarray(result['states'])

    eeg_proj_raw = weighted_projection(observation.eeg_average, eeg_forward)
    eeg_proj_recon = weighted_projection(recon_eeg, eeg_forward)
    low_proj_raw = weighted_projection(observation.fnirs_low_average, fnirs_forward)
    low_proj_recon = weighted_projection(recon_low, fnirs_forward)
    high_proj_raw = weighted_projection(observation.fnirs_high_average, fnirs_forward)
    high_proj_recon = weighted_projection(recon_high, fnirs_forward)

    eeg_channel_corrs = channelwise_correlations(observation.eeg_average, recon_eeg)
    low_channel_corrs = channelwise_correlations(observation.fnirs_low_average, recon_low)
    high_channel_corrs = channelwise_correlations(observation.fnirs_high_average, recon_high)

    eeg_top = top_indices(np.abs(eeg_forward), count=3)
    fnirs_top = top_indices(np.abs(fnirs_forward), count=3)

    return {
        'label': int(observation.label),
        'label_name': observation.label_name,
        'anchor_channel': observation.anchor_channel,
        'n_trials': int(observation.n_trials),
        'projection_correlations': {
            'eeg': safe_corr(eeg_proj_raw, eeg_proj_recon),
            'fnirs_low': safe_corr(low_proj_raw, low_proj_recon),
            'fnirs_high': safe_corr(high_proj_raw, high_proj_recon),
        },
        'median_channel_correlation': {
            'eeg': float(np.nanmedian(eeg_channel_corrs)),
            'fnirs_low': float(np.nanmedian(low_channel_corrs)),
            'fnirs_high': float(np.nanmedian(high_channel_corrs)),
        },
        'top_eeg_channels': [str(eeg_channel_names[index]) for index in eeg_top],
        'top_fnirs_channels': [str(fnirs_base_names[index]) for index in fnirs_top],
        'state_peaks': {
            'r_peak_time_s': float(observation.time_s[int(np.argmax(np.abs(state_estimates[4])))]),
            'hbo_peak_time_s': float(observation.time_s[int(np.argmax(state_estimates[2] - 1.0))]),
            'hb_trough_time_s': float(observation.time_s[int(np.argmin(state_estimates[3] - 1.0))]),
        },
    }


def write_summary_report(
    output_path: Path,
    *,
    config_name: str,
    subject_ids: Sequence[int],
    results_summary: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        '# Croce-Style Reduced Particle Filter Summary',
        '',
        '## Design',
        '',
        f'- Config: {config_name}',
        f'- Subjects analyzed: {len(subject_ids)} ({", ".join(str(subject_id) for subject_id in subject_ids)})',
        '- Hidden-state dynamics: preserved from Croce et al. 2017 with states (s, f, HbO, Hb, r).',
        '- EEG observation adaptation: 8-30 Hz channel envelopes, downsampled to 10 Hz, baseline centered per event window.',
        '- fNIRS observation adaptation: low-pass filtered lowWL/highWL intensity converted to event-wise optical density.',
        '- Forward model adaptation: geometry-constrained Gaussian sensor maps anchored to the task-relevant motor electrode because the dataset does not contain MRI-derived lead fields or optical Jacobians.',
        '- Interpretation: the reconstructed signals are denoised slow observation features, not full 200 Hz scalp waveform synthesis.',
        '',
        '## Results',
        '',
    ]

    for item in results_summary:
        lines.extend(
            [
                f"### {item['label_name']}",
                '',
                f"- Anchor channel: {item['anchor_channel']}",
                f"- Trials: {item['n_trials']}",
                f"- Projection correlation (EEG / lowWL / highWL): "
                f"{item['projection_correlations']['eeg']:.3f} / "
                f"{item['projection_correlations']['fnirs_low']:.3f} / "
                f"{item['projection_correlations']['fnirs_high']:.3f}",
                f"- Median channel correlation (EEG / lowWL / highWL): "
                f"{item['median_channel_correlation']['eeg']:.3f} / "
                f"{item['median_channel_correlation']['fnirs_low']:.3f} / "
                f"{item['median_channel_correlation']['fnirs_high']:.3f}",
                f"- Peak times: r={item['state_peaks']['r_peak_time_s']:.2f}s, "
                f"HbO={item['state_peaks']['hbo_peak_time_s']:.2f}s, "
                f"Hb trough={item['state_peaks']['hb_trough_time_s']:.2f}s",
                '',
            ]
        )

    output_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    subject_ids = parse_subject_ids(args.subject_ids, config)
    output_dir = resolve_output_dir(args.output_dir)

    eeg_dataset, fnirs_dataset = build_datasets(config, subject_ids)
    eeg_channel_names = eeg_dataset.get_channel_names()
    fnirs_channel_names = fnirs_dataset.get_channel_names()
    eeg_positions_3d, eeg_index, fnirs_pairs = build_spatial_geometry(
        data_root=config['data']['data_root'],
        eeg_channel_names=eeg_channel_names,
        fnirs_channel_names=fnirs_channel_names,
    )

    observations = aggregate_observations(
        eeg_dataset=eeg_dataset,
        fnirs_dataset=fnirs_dataset,
        subject_ids=subject_ids,
        pre_s=float(args.pre_s),
        post_s=float(args.post_s),
        common_fs=float(args.common_fs),
    )

    results_summary: List[Dict[str, Any]] = []
    all_results: Dict[str, Any] = {
        'config': {
            'source_config': args.config,
            'subject_ids': list(subject_ids),
            'pre_s': float(args.pre_s),
            'post_s': float(args.post_s),
            'task_start_delay_s': float(DEFAULT_TASK_START_DELAY_S),
            'common_fs_hz': float(args.common_fs),
            'num_particles': int(args.num_particles),
            'eeg_band_hz': list(DEFAULT_EEG_BAND),
        },
        'labels': {},
    }

    for label, observation in sorted(observations.items(), key=lambda item: item[0]):
        resolved_anchor = resolve_anchor_channel(observation.label_name, observation.label, eeg_channel_names)
        anchor_key = canonicalize_channel_label(resolved_anchor)
        if anchor_key not in eeg_index:
            raise KeyError(f'Anchor channel {resolved_anchor!r} not found in EEG channel set')
        anchor_position = eeg_positions_3d[eeg_index[anchor_key]]
        eeg_forward = gaussian_forward(anchor_position, eeg_positions_3d)
        fnirs_forward = gaussian_forward(anchor_position, fnirs_pairs.base_positions_3d)

        result = run_particle_filter(
            observation=observation,
            eeg_forward=eeg_forward,
            fnirs_forward=fnirs_forward,
            num_particles=int(args.num_particles),
            seed=int(args.seed) + int(label),
            extinction=DEFAULT_EXTINCTION,
            model_params=DEFAULT_MODEL_PARAMS,
            process_noise=DEFAULT_PROCESS_NOISE,
        )

        plot_states_and_projections(
            output_path=output_dir / f'label_{label}_states_and_projections.png',
            observation=observation,
            result=result,
            eeg_forward=eeg_forward,
            fnirs_forward=fnirs_forward,
        )
        plot_channel_overlays(
            output_path=output_dir / f'label_{label}_channel_overlays.png',
            observation=observation,
            result=result,
            eeg_forward=eeg_forward,
            fnirs_forward=fnirs_forward,
            eeg_channel_names=eeg_channel_names,
            fnirs_base_names=fnirs_pairs.base_names,
        )

        summary = summarize_result(
            observation=AggregateObservation(
                label=observation.label,
                label_name=observation.label_name,
                anchor_channel=resolved_anchor,
                n_trials=observation.n_trials,
                time_s=observation.time_s,
                eeg_average=observation.eeg_average,
                fnirs_low_average=observation.fnirs_low_average,
                fnirs_high_average=observation.fnirs_high_average,
                eeg_baseline_std=observation.eeg_baseline_std,
                fnirs_low_baseline_std=observation.fnirs_low_baseline_std,
                fnirs_high_baseline_std=observation.fnirs_high_baseline_std,
            ),
            result=result,
            eeg_forward=eeg_forward,
            fnirs_forward=fnirs_forward,
            eeg_channel_names=eeg_channel_names,
            fnirs_base_names=fnirs_pairs.base_names,
        )
        results_summary.append(summary)
        all_results['labels'][f'label_{label}'] = summary

    (output_dir / 'summary.json').write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding='utf-8')
    write_summary_report(
        output_path=output_dir / 'design_summary.md',
        config_name=args.config,
        subject_ids=subject_ids,
        results_summary=results_summary,
    )

    print(f'Saved Croce-style reconstruction outputs to {output_dir}')


if __name__ == '__main__':
    main()