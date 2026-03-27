"""Low-level loading utilities for the Simultaneous EEG&NIRS cognitive task dataset.

This module intentionally focuses on continuous-record access first.
It provides enough structure for loader smoke tests, continuous alignment
visualization, and future task-specific window extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .eeg_fnirs_dataset import (
    TrialInfo,
    apply_temporal_filter,
    get_eeg_channel_mask,
    load_mat_struct,
    normalize_window,
)


SUPPORTED_TASKS = ('nback', 'dsr', 'wg')
SUPPORTED_MODALITIES = ('eeg', 'fnirs')
SUPPORTED_FNIRS_SIGNALS = ('oxy', 'deoxy')
DEPRECATED_TASKS = ('dsr',)


TASK_SEGMENTATION_MODES: Dict[str, Dict[str, str]] = {
    'nback': {
        'eeg': 'trial',
        'fnirs': 'session',
        'both': 'session',
    },
    'wg': {
        'eeg': 'trial',
        'fnirs': 'trial',
        'both': 'trial',
    },
    'dsr': {
        'eeg': 'session',
        'fnirs': 'session',
        'both': 'session',
    },
}


SESSION_MARKER_CODES: Dict[str, Dict[str, Dict[int, str]]] = {
    'nback': {
        'eeg': {
            112: '0-back session',
            128: '2-back session',
            144: '3-back session',
        },
        'fnirs': {
            7: '0-back session',
            8: '2-back session',
            9: '3-back session',
        },
    },
    'dsr': {
        'eeg': {
            48: 'session',
        },
        'fnirs': {
            3: 'session',
        },
    },
    'wg': {
        'eeg': {},
        'fnirs': {},
    },
}


def resolve_task_name(task: str) -> str:
    normalized = task.strip().lower().replace('-', '').replace('_', '')
    mapping = {
        'nback': 'nback',
        'dsr': 'dsr',
        'discriminationselectionresponse': 'dsr',
        'wg': 'wg',
        'wordgeneration': 'wg',
    }
    if normalized not in mapping:
        raise ValueError(f'Unsupported task: {task}')
    return mapping[normalized]


def is_deprecated_task(task: str) -> bool:
    return resolve_task_name(task) in DEPRECATED_TASKS


def require_supported_task(task: str, allow_deprecated: bool = False) -> str:
    task_name = resolve_task_name(task)
    if task_name in DEPRECATED_TASKS and not allow_deprecated:
        raise NotImplementedError(
            f"Simultaneous EEG&NIRS task '{task_name}' is deprecated in this repository and is excluded from training-ready loaders."
        )
    return task_name


def resolve_segmentation_mode(
    task: str,
    modality: Literal['eeg', 'fnirs', 'both'],
    segmentation_mode: Literal['auto', 'trial', 'session'] = 'auto',
) -> str:
    task_name = resolve_task_name(task)
    if segmentation_mode != 'auto':
        return segmentation_mode
    return TASK_SEGMENTATION_MODES[task_name][modality]


def resolve_fnirs_signal(
    fnirs_signal: Literal['oxy', 'deoxy'] = 'oxy',
    *,
    hbo_only: bool = True,
    hbr_only: bool = False,
) -> Literal['oxy', 'deoxy']:
    if hbo_only and hbr_only:
        raise ValueError('Cannot enable both hbo_only and hbr_only for Simultaneous fNIRS loading.')
    if hbr_only:
        return 'deoxy'
    return fnirs_signal


def _normalize_class_names(class_name_value: Any) -> List[str]:
    if class_name_value is None:
        return []
    if isinstance(class_name_value, np.ndarray):
        return [str(item) for item in class_name_value.tolist()]
    if isinstance(class_name_value, (list, tuple)):
        return [str(item) for item in class_name_value]
    return [str(class_name_value)]


def _normalize_marker_targets(marker_y: Any, n_events: int) -> np.ndarray:
    y = np.asarray(marker_y)
    if y.ndim == 2:
        return y.astype(np.float32, copy=False)

    if y.ndim == 1:
        if y.shape[0] == n_events:
            unique_values = list(dict.fromkeys(int(value) for value in y.tolist()))
            if not unique_values:
                return np.zeros((1, n_events), dtype=np.float32)
            matrix = np.zeros((len(unique_values), n_events), dtype=np.float32)
            value_to_index = {value: index for index, value in enumerate(unique_values)}
            for column, value in enumerate(y.tolist()):
                matrix[value_to_index[int(value)], column] = 1.0
            return matrix
        if y.shape[0] == 1:
            return np.ones((1, n_events), dtype=np.float32)

    return np.ones((1, n_events), dtype=np.float32)


def normalize_marker_struct(marker_struct: Any) -> Dict[str, Any]:
    time = np.asarray(marker_struct.time, dtype=np.float64)
    event = getattr(marker_struct, 'event', None)
    event_desc = getattr(event, 'desc', None)
    if event_desc is not None:
        event_desc = np.asarray(event_desc)

    y = _normalize_marker_targets(getattr(marker_struct, 'y', None), n_events=len(time))
    class_names = _normalize_class_names(getattr(marker_struct, 'className', None))

    return {
        'time': time,
        'y': y,
        'className': class_names,
        'event_desc': event_desc,
    }


def get_session_marker_codebook(task: str, modality: str) -> Dict[int, str]:
    task_name = resolve_task_name(task)
    if modality not in ('eeg', 'fnirs'):
        raise ValueError(f'Unsupported modality: {modality}')
    return dict(SESSION_MARKER_CODES.get(task_name, {}).get(modality, {}))


def detect_offset_blocks(residual_ms: np.ndarray, jump_threshold_ms: float = 20_000.0) -> List[Dict[str, Any]]:
    if residual_ms.size == 0:
        return []

    block_start = 0
    blocks: List[Dict[str, Any]] = []
    for index in range(1, len(residual_ms)):
        if abs(float(residual_ms[index] - residual_ms[index - 1])) > float(jump_threshold_ms):
            block_residuals = residual_ms[block_start:index]
            blocks.append(
                {
                    'start_index': int(block_start),
                    'end_index': int(index - 1),
                    'count': int(index - block_start),
                    'offset_mean_ms': float(np.mean(block_residuals)),
                    'offset_std_ms': float(np.std(block_residuals)),
                }
            )
            block_start = index

    block_residuals = residual_ms[block_start:]
    blocks.append(
        {
            'start_index': int(block_start),
            'end_index': int(len(residual_ms) - 1),
            'count': int(len(residual_ms) - block_start),
            'offset_mean_ms': float(np.mean(block_residuals)),
            'offset_std_ms': float(np.std(block_residuals)),
        }
    )
    return blocks


def _select_best_skip_alignment(longer: np.ndarray, shorter: np.ndarray) -> Tuple[int, np.ndarray]:
    best_skip_index = 0
    best_residual = shorter - np.delete(longer, 0)
    best_score = float('inf')

    for skip_index in range(len(longer)):
        candidate = shorter - np.delete(longer, skip_index)
        candidate_blocks = detect_offset_blocks(candidate)
        score = sum(block['offset_std_ms'] for block in candidate_blocks)
        if score < best_score:
            best_score = score
            best_skip_index = skip_index
            best_residual = candidate

    return best_skip_index, best_residual


class SimultaneousCognitiveLoader:
    """Load continuous EEG or fNIRS recordings for one cognitive task."""

    def __init__(
        self,
        data_root: str,
        task: str = 'nback',
        subject_ids: Optional[List[int]] = None,
        modality: Literal['eeg', 'fnirs', 'both'] = 'both',
        fnirs_signal: Literal['oxy', 'deoxy'] = 'oxy',
        allow_deprecated: bool = True,
    ):
        self.data_root = Path(data_root)
        self.task = require_supported_task(task, allow_deprecated=allow_deprecated)
        self.subject_ids = subject_ids or list(range(1, 27))
        self.modality = modality
        self.fnirs_signal = fnirs_signal
        self._validate_paths()

    def _validate_paths(self) -> None:
        if not self.data_root.exists():
            raise FileNotFoundError(f'Dataset root not found: {self.data_root}')

        first_subject = self.subject_ids[0]
        if self.modality in ('eeg', 'both'):
            eeg_dir = self._get_subject_dir(first_subject, 'eeg')
            if not eeg_dir.exists():
                raise FileNotFoundError(f'EEG subject directory not found: {eeg_dir}')
        if self.modality in ('fnirs', 'both'):
            fnirs_dir = self._get_subject_dir(first_subject, 'fnirs')
            if not fnirs_dir.exists():
                raise FileNotFoundError(f'fNIRS subject directory not found: {fnirs_dir}')

    def _get_subject_dir(self, subject_id: int, modality: str) -> Path:
        suffix = 'EEG' if modality == 'eeg' else 'NIRS'
        return self.data_root / f'VP{subject_id:03d}-{suffix}'

    def _get_file_paths(self, subject_id: int, modality: str) -> Tuple[Path, Path]:
        subject_dir = self._get_subject_dir(subject_id, modality)
        return (
            subject_dir / f'cnt_{self.task}.mat',
            subject_dir / f'mrk_{self.task}.mat',
        )

    def load_subject_data(self, subject_id: int, modality: Literal['eeg', 'fnirs']) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        cnt_path, mrk_path = self._get_file_paths(subject_id, modality)
        cnt_mat = load_mat_struct(str(cnt_path))
        mrk_mat = load_mat_struct(str(mrk_path))

        cnt_key = next(key for key in cnt_mat.keys() if not key.startswith('__'))
        mrk_key = next(key for key in mrk_mat.keys() if not key.startswith('__'))

        cnt_struct = cnt_mat[cnt_key]
        marker_info = normalize_marker_struct(mrk_mat[mrk_key])

        if modality == 'eeg':
            data = np.asarray(cnt_struct.x, dtype=np.float32)
            info = {
                'fs': float(cnt_struct.fs),
                'clab': [str(name) for name in np.asarray(cnt_struct.clab).tolist()],
                'task': self.task,
                'modality': modality,
                'title': getattr(cnt_struct, 'title', ''),
            }
            return data, marker_info, info

        signal_struct = getattr(cnt_struct, self.fnirs_signal)
        data = np.asarray(signal_struct.x, dtype=np.float32)
        info = {
            'fs': float(signal_struct.fs),
            'clab': [str(name) for name in np.asarray(signal_struct.clab).tolist()],
            'task': self.task,
            'modality': modality,
            'signal': self.fnirs_signal,
            'available_signals': list(SUPPORTED_FNIRS_SIGNALS),
            'title': getattr(signal_struct, 'title', ''),
        }
        return data, marker_info, info

    def check_marker_alignment(self, subject_id: int) -> Dict[str, Any]:
        eeg_markers = self.load_subject_data(subject_id, 'eeg')[1]
        nirs_markers = self.load_subject_data(subject_id, 'fnirs')[1]

        common_count = min(len(eeg_markers['time']), len(nirs_markers['time']))
        eeg_times = eeg_markers['time'][:common_count]
        nirs_times = nirs_markers['time'][:common_count]
        residual_ms = (nirs_times - eeg_times).astype(np.float64)

        eeg_labels = np.argmax(eeg_markers['y'], axis=0)[:common_count]
        nirs_labels = np.argmax(nirs_markers['y'], axis=0)[:common_count]

        return {
            'task': self.task,
            'num_eeg_events': int(len(eeg_markers['time'])),
            'num_fnirs_events': int(len(nirs_markers['time'])),
            'num_common_events': int(common_count),
            'initial_offset_ms': float(nirs_markers['time'][0] - eeg_markers['time'][0]) if common_count else None,
            'residual_mean_ms': float(np.mean(residual_ms)) if common_count else None,
            'residual_std_ms': float(np.std(residual_ms)) if common_count else None,
            'label_index_match': bool(np.array_equal(eeg_labels, nirs_labels)) if common_count else False,
            'eeg_class_names': eeg_markers['className'],
            'fnirs_class_names': nirs_markers['className'],
        }

    def get_session_markers(self, subject_id: int, modality: Literal['eeg', 'fnirs']) -> Dict[str, Any]:
        marker_info = self.load_subject_data(subject_id, modality)[1]
        codebook = get_session_marker_codebook(self.task, modality)
        if not codebook:
            return marker_info

        event_desc = marker_info.get('event_desc')
        if event_desc is None:
            return marker_info

        event_desc = np.asarray(event_desc)
        mask = np.isin(event_desc, list(codebook.keys()))
        session_y = marker_info['y'][:, mask]
        session_time = np.asarray(marker_info['time'], dtype=np.float64)[mask]
        session_desc = event_desc[mask]

        labels = np.argmax(session_y, axis=0) if session_y.size else np.asarray([], dtype=int)
        class_names = [codebook.get(int(session_desc[index]), str(labels[index])) for index in range(len(session_time))]

        return {
            'time': session_time,
            'y': session_y,
            'className': class_names,
            'event_desc': session_desc,
        }

    def align_session_markers(self, subject_id: int, jump_threshold_ms: float = 20_000.0) -> Dict[str, Any]:
        eeg_sessions = self.get_session_markers(subject_id, 'eeg')
        fnirs_sessions = self.get_session_markers(subject_id, 'fnirs')

        eeg_times = np.asarray(eeg_sessions['time'], dtype=np.float64)
        fnirs_times = np.asarray(fnirs_sessions['time'], dtype=np.float64)
        eeg_labels = list(eeg_sessions.get('className', []))
        fnirs_labels = list(fnirs_sessions.get('className', []))

        skipped = {'eeg_indices': [], 'fnirs_indices': []}
        if len(eeg_times) == len(fnirs_times):
            aligned_eeg_times = eeg_times
            aligned_fnirs_times = fnirs_times
            aligned_eeg_labels = eeg_labels
            aligned_fnirs_labels = fnirs_labels
        elif len(eeg_times) == len(fnirs_times) + 1:
            skip_index, residual_ms = _select_best_skip_alignment(eeg_times, fnirs_times)
            aligned_eeg_times = np.delete(eeg_times, skip_index)
            aligned_fnirs_times = fnirs_times
            aligned_eeg_labels = [label for index, label in enumerate(eeg_labels) if index != skip_index]
            aligned_fnirs_labels = fnirs_labels
            skipped['eeg_indices'] = [int(skip_index)]
        elif len(fnirs_times) == len(eeg_times) + 1:
            skip_index, residual_ms = _select_best_skip_alignment(fnirs_times, eeg_times)
            aligned_eeg_times = eeg_times
            aligned_fnirs_times = np.delete(fnirs_times, skip_index)
            aligned_eeg_labels = eeg_labels
            aligned_fnirs_labels = [label for index, label in enumerate(fnirs_labels) if index != skip_index]
            skipped['fnirs_indices'] = [int(skip_index)]
        else:
            common = min(len(eeg_times), len(fnirs_times))
            aligned_eeg_times = eeg_times[:common]
            aligned_fnirs_times = fnirs_times[:common]
            aligned_eeg_labels = eeg_labels[:common]
            aligned_fnirs_labels = fnirs_labels[:common]

        residual_ms = aligned_fnirs_times - aligned_eeg_times
        blocks = detect_offset_blocks(residual_ms, jump_threshold_ms=jump_threshold_ms)

        return {
            'task': self.task,
            'num_eeg_session_markers': int(len(eeg_times)),
            'num_fnirs_session_markers': int(len(fnirs_times)),
            'num_aligned_pairs': int(len(residual_ms)),
            'skipped_marker_indices': skipped,
            'label_sequence_match': aligned_eeg_labels == aligned_fnirs_labels,
            'eeg_labels': aligned_eeg_labels,
            'fnirs_labels': aligned_fnirs_labels,
            'residual_mean_ms': float(np.mean(residual_ms)) if residual_ms.size else None,
            'residual_std_ms': float(np.std(residual_ms)) if residual_ms.size else None,
            'residual_series_ms': residual_ms.tolist(),
            'offset_blocks': blocks,
        }


class SimultaneousContinuousDataset:
    """Continuous-record accessor that mirrors the visualization API of EEGfNIRSDataset.

    The constructor selects a single task file (nback, dsr, or wg). Methods keep the
    same session-style names as the existing dataset API so current visualization code
    can be adapted with minimal branching. The selected task is exposed as session 0.
    """

    def __init__(
        self,
        data_root: str,
        task: str,
        modality: Literal['eeg', 'fnirs'],
        subject_ids: Optional[List[int]] = None,
        normalize: bool = True,
        normalization_mode: Literal['window', 'session', 'none'] = 'session',
        preprocessing: Optional[dict] = None,
        exclude_eog: bool = True,
        fnirs_signal: Literal['oxy', 'deoxy'] = 'oxy',
        segmentation_mode: Literal['auto', 'trial', 'session'] = 'auto',
        allow_deprecated: bool = False,
    ):
        self.data_root = Path(data_root)
        self.task = require_supported_task(task, allow_deprecated=allow_deprecated)
        self.modality = modality
        self.subject_ids = subject_ids or list(range(1, 27))
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.preprocessing = dict(preprocessing or {})
        self.exclude_eog = exclude_eog
        self.fnirs_signal = fnirs_signal
        self.segmentation_mode = resolve_segmentation_mode(self.task, modality, segmentation_mode)

        self.loader = SimultaneousCognitiveLoader(
            data_root=data_root,
            task=self.task,
            subject_ids=self.subject_ids,
            modality=modality,
            fnirs_signal=fnirs_signal,
            allow_deprecated=allow_deprecated,
        )

        self._raw_cache: Dict[int, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = {}
        self._processed_cache: Dict[int, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = {}
        self._session_stats_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _validate_session_idx(self, session_idx: int) -> None:
        if session_idx != 0:
            raise IndexError(
                f'SimultaneousContinuousDataset exposes the selected task {self.task!r} as session_idx=0 only; got {session_idx}'
            )

    def _cache_subject_data(self, subject_id: int) -> None:
        if subject_id in self._processed_cache:
            return

        cnt, markers, info = self.loader.load_subject_data(subject_id, self.modality)
        channel_names = list(info['clab'])

        if self.modality == 'eeg':
            channel_mask = get_eeg_channel_mask(channel_names, exclude_eog=self.exclude_eog)
            filtered_cnt = cnt[:, channel_mask]
            filtered_channel_names = [name for index, name in enumerate(channel_names) if channel_mask[index]]
        else:
            filtered_cnt = cnt
            filtered_channel_names = channel_names

        processed_cnt = apply_temporal_filter(
            filtered_cnt,
            sample_rate=float(info['fs']),
            modality=self.modality,
            preprocessing=self.preprocessing,
        )

        filtered_info = dict(info)
        filtered_info['clab'] = filtered_channel_names
        filtered_info['original_clab'] = channel_names

        self._raw_cache[subject_id] = (filtered_cnt.astype(np.float32, copy=False), markers, filtered_info)
        self._processed_cache[subject_id] = (processed_cnt.astype(np.float32, copy=False), markers, filtered_info)
        self._session_stats_cache[subject_id] = (
            processed_cnt.mean(axis=0).astype(np.float32, copy=False),
            (processed_cnt.std(axis=0) + 1e-8).astype(np.float32, copy=False),
        )

    def _get_subject_data(self, subject_id: int, processed: bool = True) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        self._cache_subject_data(subject_id)
        if processed:
            return self._processed_cache[subject_id]
        return self._raw_cache[subject_id]

    def get_session_continuous_data(
        self,
        subject_id: int,
        session_idx: int,
        processed: bool = True,
        normalized: bool = False,
    ) -> np.ndarray:
        self._validate_session_idx(session_idx)
        cnt, _, _ = self._get_subject_data(subject_id, processed=processed)
        session = cnt.T.copy()
        if normalized and self.normalize and self.normalization_mode == 'session':
            session_mean, session_std = self._session_stats_cache[subject_id]
            session = normalize_window(session, 'session', session_mean, session_std)
        return session

    def get_session_markers(self, subject_id: int, session_idx: int) -> Dict[str, Any]:
        self._validate_session_idx(session_idx)
        _, markers, _ = self._get_subject_data(subject_id, processed=False)
        return markers

    def get_session_segmentation_markers(self, subject_id: int, session_idx: int) -> Dict[str, Any]:
        self._validate_session_idx(session_idx)
        if self.segmentation_mode == 'session':
            return self.loader.get_session_markers(subject_id, self.modality)
        return self.get_session_markers(subject_id, session_idx)

    def get_session_trial_regions(
        self,
        subject_id: int,
        session_idx: int,
        window_duration_s: float,
        offset_ms: float = 0.0,
    ) -> List[Dict[str, Any]]:
        self._validate_session_idx(session_idx)
        markers = self.get_session_segmentation_markers(subject_id, session_idx)
        labels = np.argmax(markers['y'], axis=0)
        class_names = markers.get('className', [])
        regions: List[Dict[str, Any]] = []

        for trial_idx, onset_ms in enumerate(np.asarray(markers['time'], dtype=float)):
            label = int(labels[trial_idx])
            label_name = class_names[label] if label < len(class_names) else str(label)
            regions.append(
                {
                    'trial_idx': trial_idx,
                    'label': label,
                    'label_name': str(label_name),
                    'start_s': (float(onset_ms) + offset_ms) / 1000.0,
                    'end_s': (float(onset_ms) + offset_ms) / 1000.0 + float(window_duration_s),
                    'onset_s': float(onset_ms) / 1000.0,
                    'event_desc': None if markers.get('event_desc') is None else int(markers['event_desc'][trial_idx]),
                }
            )

        return regions

    def get_channel_names(self) -> List[str]:
        if not self.subject_ids:
            return []
        _, _, info = self._get_subject_data(self.subject_ids[0], processed=True)
        return list(info['clab'])

    def get_num_channels(self) -> int:
        return len(self.get_channel_names())

    def get_sample_rate(self) -> float:
        if not self.subject_ids:
            return 0.0
        _, _, info = self._get_subject_data(self.subject_ids[0], processed=True)
        return float(info['fs'])


@dataclass
class SimultaneousTrialWindowInfo:
    subject_id: int
    session_idx: int
    trial_idx: int
    label: int
    label_name: str
    start_sample: int
    end_sample: int
    onset_time_ms: float
    event_desc: Optional[int]


class SimultaneousEEGfNIRSDataset(Dataset):
    """Training-ready single-modality window dataset for Simultaneous EEG&NIRS."""

    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[int]] = None,
        task: str = 'nback',
        modality: Literal['eeg', 'fnirs'] = 'eeg',
        window_samples: int = 512,
        window_offset_ms: float = 0.0,
        normalize: bool = True,
        normalization_mode: Literal['window', 'session', 'none'] = 'window',
        preprocessing: Optional[dict] = None,
        exclude_eog: bool = True,
        hbo_only: bool = True,
        hbr_only: bool = False,
        fnirs_signal: Literal['oxy', 'deoxy'] = 'oxy',
        segmentation_mode: Literal['auto', 'trial', 'session'] = 'auto',
    ):
        self.data_root = Path(data_root)
        self.task = require_supported_task(task, allow_deprecated=False)
        self.subject_ids = subject_ids or list(range(1, 27))
        self.modality = modality
        self.window_samples = int(window_samples)
        self.window_offset_ms = float(window_offset_ms)
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.preprocessing = dict(preprocessing or {})
        self.exclude_eog = exclude_eog
        self.fnirs_signal = resolve_fnirs_signal(fnirs_signal, hbo_only=hbo_only, hbr_only=hbr_only)
        self.segmentation_mode = resolve_segmentation_mode(self.task, modality, segmentation_mode)

        self.loader = SimultaneousCognitiveLoader(
            data_root=data_root,
            task=self.task,
            subject_ids=self.subject_ids,
            modality=modality,
            fnirs_signal=self.fnirs_signal,
            allow_deprecated=False,
        )

        self.trials: List[SimultaneousTrialWindowInfo] = []
        self._raw_cache: Dict[int, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = {}
        self._processed_cache: Dict[int, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = {}
        self._session_stats_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._build_trial_index()

    def _get_marker_info(self, subject_id: int) -> Dict[str, Any]:
        if self.segmentation_mode == 'session':
            return self.loader.get_session_markers(subject_id, self.modality)
        return self.loader.load_subject_data(subject_id, self.modality)[1]

    def _build_trial_index(self) -> None:
        for subject_id in self.subject_ids:
            try:
                _, _, info = self.loader.load_subject_data(subject_id, self.modality)
                fs = float(info['fs'])
                markers = self._get_marker_info(subject_id)
                labels = np.argmax(markers['y'], axis=0) if markers['y'].size else np.asarray([], dtype=int)
                class_names = list(markers.get('className', []))
                event_desc = markers.get('event_desc')
                offset_samples = int(round(self.window_offset_ms * fs / 1000.0))

                for trial_idx, onset_ms in enumerate(np.asarray(markers['time'], dtype=np.float64)):
                    start_sample = int(round(onset_ms * fs / 1000.0)) + offset_samples
                    end_sample = start_sample + self.window_samples
                    label = int(labels[trial_idx]) if trial_idx < len(labels) else 0
                    label_name = class_names[label] if label < len(class_names) else str(label)
                    desc = None if event_desc is None else int(np.asarray(event_desc)[trial_idx])
                    self.trials.append(
                        SimultaneousTrialWindowInfo(
                            subject_id=subject_id,
                            session_idx=0,
                            trial_idx=trial_idx,
                            label=label,
                            label_name=str(label_name),
                            start_sample=start_sample,
                            end_sample=end_sample,
                            onset_time_ms=float(onset_ms),
                            event_desc=desc,
                        )
                    )
            except Exception as error:
                print(f'Warning: Could not build Simultaneous trial index for subject {subject_id}: {error}')

    def _cache_subject_data(self, subject_id: int) -> None:
        if subject_id in self._processed_cache:
            return

        cnt, markers, info = self.loader.load_subject_data(subject_id, self.modality)
        channel_names = list(info['clab'])
        if self.modality == 'eeg':
            channel_mask = get_eeg_channel_mask(channel_names, exclude_eog=self.exclude_eog)
            filtered_cnt = cnt[:, channel_mask]
            filtered_channel_names = [name for index, name in enumerate(channel_names) if channel_mask[index]]
        else:
            filtered_cnt = cnt
            filtered_channel_names = channel_names

        filtered_cnt = filtered_cnt.astype(np.float32, copy=False)
        processed_cnt = apply_temporal_filter(
            filtered_cnt,
            sample_rate=float(info['fs']),
            modality=self.modality,
            preprocessing=self.preprocessing,
        ).astype(np.float32, copy=False)

        filtered_info = dict(info)
        filtered_info['clab'] = filtered_channel_names
        filtered_info['original_clab'] = channel_names
        self._raw_cache[subject_id] = (filtered_cnt, markers, filtered_info)
        self._processed_cache[subject_id] = (processed_cnt, markers, filtered_info)
        self._session_stats_cache[subject_id] = (
            processed_cnt.mean(axis=0).astype(np.float32, copy=False),
            (processed_cnt.std(axis=0) + 1e-8).astype(np.float32, copy=False),
        )

    def _get_subject_data(self, subject_id: int, processed: bool = True) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        self._cache_subject_data(subject_id)
        if processed:
            return self._processed_cache[subject_id]
        return self._raw_cache[subject_id]

    def _extract_window(self, cnt: np.ndarray, start: int, end: int) -> np.ndarray:
        if start < 0:
            start = 0
            end = self.window_samples
        if end > cnt.shape[0]:
            end = cnt.shape[0]
            start = max(0, end - self.window_samples)

        window = cnt[start:end, :]
        if window.shape[0] < self.window_samples:
            pad_size = self.window_samples - window.shape[0]
            window = np.pad(window, ((0, pad_size), (0, 0)), mode='constant')
        return window.T

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trial = self.trials[idx]
        cnt, _, _ = self._get_subject_data(trial.subject_id, processed=True)
        window = self._extract_window(cnt, trial.start_sample, trial.end_sample)

        normalization_mode = self.normalization_mode if self.normalize else 'none'
        if normalization_mode == 'session':
            session_mean, session_std = self._session_stats_cache[trial.subject_id]
            window = normalize_window(window, 'session', session_mean, session_std)
        elif normalization_mode == 'window':
            window = normalize_window(window, 'window')

        return {
            'data': torch.from_numpy(window).float(),
            'label': torch.tensor(trial.label, dtype=torch.long),
            'subject_id': trial.subject_id,
            'session_idx': trial.session_idx,
            'trial_idx': trial.trial_idx,
        }

    def get_channel_names(self) -> List[str]:
        if not self.trials:
            return []
        _, _, info = self._get_subject_data(self.trials[0].subject_id, processed=True)
        return list(info['clab'])

    def get_num_channels(self) -> int:
        return len(self.get_channel_names())

    def get_sample_rate(self) -> float:
        if not self.trials:
            return 0.0
        _, _, info = self._get_subject_data(self.trials[0].subject_id, processed=True)
        return float(info['fs'])


class SimultaneousMultiModalDataset(Dataset):
    """Training-ready multimodal window dataset for Simultaneous EEG&NIRS."""

    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[int]] = None,
        task: str = 'wg',
        window_duration_s: float = 10.0,
        window_offset_ms: float = 0.0,
        normalize: bool = True,
        normalization_mode: Literal['window', 'session', 'none'] = 'window',
        eeg_preprocessing: Optional[dict] = None,
        fnirs_preprocessing: Optional[dict] = None,
        exclude_eog: bool = True,
        hbo_only: bool = True,
        hbr_only: bool = False,
        fnirs_signal: Literal['oxy', 'deoxy'] = 'oxy',
        segmentation_mode: Literal['auto', 'trial', 'session'] = 'auto',
    ):
        self.data_root = Path(data_root)
        self.task = require_supported_task(task, allow_deprecated=False)
        self.subject_ids = subject_ids or list(range(1, 27))
        self.window_duration_s = float(window_duration_s)
        self.window_offset_ms = float(window_offset_ms)
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.eeg_preprocessing = dict(eeg_preprocessing or {})
        self.fnirs_preprocessing = dict(fnirs_preprocessing or {})
        self.exclude_eog = exclude_eog
        self.fnirs_signal = resolve_fnirs_signal(fnirs_signal, hbo_only=hbo_only, hbr_only=hbr_only)
        self.segmentation_mode = resolve_segmentation_mode(self.task, 'both', segmentation_mode)

        self.eeg_loader = SimultaneousCognitiveLoader(
            data_root=data_root,
            task=self.task,
            subject_ids=self.subject_ids,
            modality='eeg',
            allow_deprecated=False,
        )
        self.fnirs_loader = SimultaneousCognitiveLoader(
            data_root=data_root,
            task=self.task,
            subject_ids=self.subject_ids,
            modality='fnirs',
            fnirs_signal=self.fnirs_signal,
            allow_deprecated=False,
        )

        self.trials: List[TrialInfo] = []
        self._eeg_raw_cache: Dict[int, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = {}
        self._eeg_processed_cache: Dict[int, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = {}
        self._eeg_session_stats_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._fnirs_raw_cache: Dict[int, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = {}
        self._fnirs_processed_cache: Dict[int, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = {}
        self._fnirs_session_stats_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._build_trial_index()

    def _get_markers(self, subject_id: int, modality: Literal['eeg', 'fnirs']) -> Dict[str, Any]:
        loader = self.eeg_loader if modality == 'eeg' else self.fnirs_loader
        if self.segmentation_mode == 'session':
            return loader.get_session_markers(subject_id, modality)
        return loader.load_subject_data(subject_id, modality)[1]

    def _build_trial_index(self) -> None:
        for subject_id in self.subject_ids:
            try:
                _, _, eeg_info = self.eeg_loader.load_subject_data(subject_id, 'eeg')
                _, _, fnirs_info = self.fnirs_loader.load_subject_data(subject_id, 'fnirs')
                eeg_fs = float(eeg_info['fs'])
                fnirs_fs = float(fnirs_info['fs'])
                eeg_markers = self._get_markers(subject_id, 'eeg')
                fnirs_markers = self._get_markers(subject_id, 'fnirs')

                eeg_times = np.asarray(eeg_markers['time'], dtype=np.float64)
                fnirs_times = np.asarray(fnirs_markers['time'], dtype=np.float64)
                common_count = min(len(eeg_times), len(fnirs_times))
                eeg_labels = np.argmax(eeg_markers['y'], axis=0)[:common_count]
                fnirs_labels = np.argmax(fnirs_markers['y'], axis=0)[:common_count]
                eeg_names = list(eeg_markers.get('className', []))
                fnirs_names = list(fnirs_markers.get('className', []))

                if common_count == 0:
                    continue

                for trial_idx in range(common_count):
                    eeg_label = int(eeg_labels[trial_idx])
                    fnirs_label = int(fnirs_labels[trial_idx])
                    eeg_label_name = eeg_names[eeg_label] if eeg_label < len(eeg_names) else str(eeg_label)
                    fnirs_label_name = fnirs_names[fnirs_label] if fnirs_label < len(fnirs_names) else str(fnirs_label)
                    if str(eeg_label_name) != str(fnirs_label_name):
                        raise ValueError(
                            f'Multimodal segmentation label mismatch for subject {subject_id}, task {self.task}, index {trial_idx}: '
                            f'{eeg_label_name!r} vs {fnirs_label_name!r}'
                        )

                    eeg_window_samples = int(round(self.window_duration_s * eeg_fs))
                    fnirs_window_samples = int(round(self.window_duration_s * fnirs_fs))
                    eeg_start = int(round((eeg_times[trial_idx] + self.window_offset_ms) * eeg_fs / 1000.0))
                    fnirs_start = int(round((fnirs_times[trial_idx] + self.window_offset_ms) * fnirs_fs / 1000.0))
                    self.trials.append(
                        TrialInfo(
                            subject_id=subject_id,
                            session_idx=0,
                            trial_idx=trial_idx,
                            label=eeg_label,
                            task_type=self.task,
                            eeg_start_sample=eeg_start,
                            eeg_end_sample=eeg_start + eeg_window_samples,
                            nirs_start_sample=fnirs_start,
                            nirs_end_sample=fnirs_start + fnirs_window_samples,
                            onset_time_ms=float(eeg_times[trial_idx]),
                        )
                    )
            except Exception as error:
                print(f'Warning: Could not build Simultaneous multimodal trial index for subject {subject_id}: {error}')

    def _cache_eeg(self, subject_id: int) -> None:
        if subject_id in self._eeg_processed_cache:
            return
        cnt, markers, info = self.eeg_loader.load_subject_data(subject_id, 'eeg')
        channel_names = list(info['clab'])
        channel_mask = get_eeg_channel_mask(channel_names, exclude_eog=self.exclude_eog)
        filtered_cnt = cnt[:, channel_mask].astype(np.float32, copy=False)
        processed_cnt = apply_temporal_filter(
            filtered_cnt,
            sample_rate=float(info['fs']),
            modality='eeg',
            preprocessing=self.eeg_preprocessing,
        ).astype(np.float32, copy=False)
        filtered_info = dict(info)
        filtered_info['clab'] = [name for index, name in enumerate(channel_names) if channel_mask[index]]
        filtered_info['original_clab'] = channel_names
        self._eeg_raw_cache[subject_id] = (filtered_cnt, markers, filtered_info)
        self._eeg_processed_cache[subject_id] = (processed_cnt, markers, filtered_info)
        self._eeg_session_stats_cache[subject_id] = (
            processed_cnt.mean(axis=0).astype(np.float32, copy=False),
            (processed_cnt.std(axis=0) + 1e-8).astype(np.float32, copy=False),
        )

    def _cache_fnirs(self, subject_id: int) -> None:
        if subject_id in self._fnirs_processed_cache:
            return
        cnt, markers, info = self.fnirs_loader.load_subject_data(subject_id, 'fnirs')
        filtered_cnt = cnt.astype(np.float32, copy=False)
        processed_cnt = apply_temporal_filter(
            filtered_cnt,
            sample_rate=float(info['fs']),
            modality='fnirs',
            preprocessing=self.fnirs_preprocessing,
        ).astype(np.float32, copy=False)
        filtered_info = dict(info)
        filtered_info['original_clab'] = list(info['clab'])
        self._fnirs_raw_cache[subject_id] = (filtered_cnt, markers, filtered_info)
        self._fnirs_processed_cache[subject_id] = (processed_cnt, markers, filtered_info)
        self._fnirs_session_stats_cache[subject_id] = (
            processed_cnt.mean(axis=0).astype(np.float32, copy=False),
            (processed_cnt.std(axis=0) + 1e-8).astype(np.float32, copy=False),
        )

    def _extract_window(self, cnt: np.ndarray, start: int, end: int, target_samples: int) -> np.ndarray:
        if start < 0:
            start = 0
            end = target_samples
        if end > cnt.shape[0]:
            end = cnt.shape[0]
            start = max(0, end - target_samples)
        window = cnt[start:end, :]
        if window.shape[0] < target_samples:
            pad_size = target_samples - window.shape[0]
            window = np.pad(window, ((0, pad_size), (0, 0)), mode='constant')
        return window.T

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trial = self.trials[idx]
        self._cache_eeg(trial.subject_id)
        self._cache_fnirs(trial.subject_id)
        eeg_cnt, _, eeg_info = self._eeg_processed_cache[trial.subject_id]
        fnirs_cnt, _, fnirs_info = self._fnirs_processed_cache[trial.subject_id]
        eeg_window_samples = int(round(self.window_duration_s * float(eeg_info['fs'])))
        fnirs_window_samples = int(round(self.window_duration_s * float(fnirs_info['fs'])))
        eeg_window = self._extract_window(eeg_cnt, trial.eeg_start_sample, trial.eeg_end_sample, eeg_window_samples)
        fnirs_window = self._extract_window(fnirs_cnt, trial.nirs_start_sample, trial.nirs_end_sample, fnirs_window_samples)

        normalization_mode = self.normalization_mode if self.normalize else 'none'
        if normalization_mode == 'session':
            eeg_mean, eeg_std = self._eeg_session_stats_cache[trial.subject_id]
            fnirs_mean, fnirs_std = self._fnirs_session_stats_cache[trial.subject_id]
            eeg_window = normalize_window(eeg_window, 'session', eeg_mean, eeg_std)
            fnirs_window = normalize_window(fnirs_window, 'session', fnirs_mean, fnirs_std)
        elif normalization_mode == 'window':
            eeg_window = normalize_window(eeg_window, 'window')
            fnirs_window = normalize_window(fnirs_window, 'window')

        return {
            'eeg': torch.from_numpy(eeg_window).float(),
            'fnirs': torch.from_numpy(fnirs_window).float(),
            'label': torch.tensor(trial.label, dtype=torch.long),
            'subject_id': trial.subject_id,
            'session_idx': trial.session_idx,
            'trial_idx': trial.trial_idx,
        }

    def get_eeg_channel_names(self) -> List[str]:
        if not self.trials:
            return []
        self._cache_eeg(self.trials[0].subject_id)
        _, _, info = self._eeg_processed_cache[self.trials[0].subject_id]
        return list(info['clab'])

    def get_fnirs_channel_names(self) -> List[str]:
        if not self.trials:
            return []
        self._cache_fnirs(self.trials[0].subject_id)
        _, _, info = self._fnirs_processed_cache[self.trials[0].subject_id]
        return list(info['clab'])

    def get_num_eeg_channels(self) -> int:
        return len(self.get_eeg_channel_names())

    def get_num_fnirs_channels(self) -> int:
        return len(self.get_fnirs_channel_names())

    def get_eeg_sample_rate(self) -> float:
        if not self.trials:
            return 0.0
        self._cache_eeg(self.trials[0].subject_id)
        _, _, info = self._eeg_processed_cache[self.trials[0].subject_id]
        return float(info['fs'])

    def get_fnirs_sample_rate(self) -> float:
        if not self.trials:
            return 0.0
        self._cache_fnirs(self.trials[0].subject_id)
        _, _, info = self._fnirs_processed_cache[self.trials[0].subject_id]
        return float(info['fs'])
