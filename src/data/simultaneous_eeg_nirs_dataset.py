"""Low-level loading utilities for the Simultaneous EEG&NIRS cognitive task dataset.

This module intentionally focuses on continuous-record access first.
It provides enough structure for loader smoke tests, continuous alignment
visualization, and future task-specific window extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .eeg_fnirs_dataset import (
    apply_temporal_filter,
    get_eeg_channel_mask,
    load_mat_struct,
    normalize_window,
)


SUPPORTED_TASKS = ('nback', 'dsr', 'wg')
SUPPORTED_MODALITIES = ('eeg', 'fnirs')
SUPPORTED_FNIRS_SIGNALS = ('oxy', 'deoxy')


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


class SimultaneousCognitiveLoader:
    """Load continuous EEG or fNIRS recordings for one cognitive task."""

    def __init__(
        self,
        data_root: str,
        task: str = 'nback',
        subject_ids: Optional[List[int]] = None,
        modality: Literal['eeg', 'fnirs', 'both'] = 'both',
        fnirs_signal: Literal['oxy', 'deoxy'] = 'oxy',
    ):
        self.data_root = Path(data_root)
        self.task = resolve_task_name(task)
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
    ):
        self.data_root = Path(data_root)
        self.task = resolve_task_name(task)
        self.modality = modality
        self.subject_ids = subject_ids or list(range(1, 27))
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.preprocessing = dict(preprocessing or {})
        self.exclude_eog = exclude_eog
        self.fnirs_signal = fnirs_signal

        self.loader = SimultaneousCognitiveLoader(
            data_root=data_root,
            task=self.task,
            subject_ids=self.subject_ids,
            modality=modality,
            fnirs_signal=fnirs_signal,
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

    def get_session_trial_regions(
        self,
        subject_id: int,
        session_idx: int,
        window_duration_s: float,
        offset_ms: float = 0.0,
    ) -> List[Dict[str, Any]]:
        self._validate_session_idx(session_idx)
        markers = self.get_session_markers(subject_id, session_idx)
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
