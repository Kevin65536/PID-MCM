"""Shared dataset factories for training and visualization entry points."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from .eeg_fnirs_dataset import EEGfNIRSDataset, MultiModalEEGfNIRSDataset, create_dataloaders as create_single_trial_dataloaders
from .registry import resolve_dataset_id
from .simultaneous_eeg_nirs_dataset import (
    SimultaneousContinuousDataset,
    SimultaneousEEGfNIRSDataset,
    SimultaneousMultiModalDataset,
    resolve_fnirs_signal,
    resolve_segmentation_mode,
)


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


def _dataset_params(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = data_cfg.get('dataset_params', {})
    return dict(params) if isinstance(params, dict) else {}


def _resolve_fnirs_signal_from_config(data_cfg: Dict[str, Any]) -> str:
    params = _dataset_params(data_cfg)
    return resolve_fnirs_signal(
        params.get('fnirs_signal', 'oxy'),
        hbo_only=bool(data_cfg.get('hbo_only', True)),
        hbr_only=bool(data_cfg.get('hbr_only', False)),
    )


def create_unimodal_window_dataset(
    data_cfg: Dict[str, Any],
    subject_ids: List[int],
    modality: str,
    *,
    window_samples: int,
    normalize: bool,
    normalization_mode: str,
):
    dataset_id = resolve_dataset_id(data_cfg)
    params = _dataset_params(data_cfg)

    if dataset_id == 'eeg_fnirs_single_trial':
        preprocessing_key = 'eeg_preprocessing' if modality == 'eeg' else 'fnirs_preprocessing'
        return EEGfNIRSDataset(
            data_root=data_cfg['data_root'],
            modality=modality,
            subject_ids=subject_ids,
            task=data_cfg.get('task', 'motor_imagery'),
            window_samples=window_samples,
            window_offset_ms=data_cfg['window'].get('offset_ms', 0),
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=data_cfg.get(preprocessing_key, data_cfg.get('preprocessing', {})),
            exclude_eog=data_cfg.get('exclude_eog', False),
            hbo_only=data_cfg.get('hbo_only', False),
            hbr_only=data_cfg.get('hbr_only', False),
        )

    if dataset_id == 'simultaneous_eeg_nirs':
        preprocessing_key = 'eeg_preprocessing' if modality == 'eeg' else 'fnirs_preprocessing'
        return SimultaneousEEGfNIRSDataset(
            data_root=data_cfg['data_root'],
            modality=modality,
            subject_ids=subject_ids,
            task=data_cfg.get('task', 'nback'),
            window_samples=window_samples,
            window_offset_ms=float(data_cfg['window'].get('offset_ms', 0)),
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=data_cfg.get(preprocessing_key, data_cfg.get('preprocessing', {})),
            exclude_eog=data_cfg.get('exclude_eog', True),
            hbo_only=data_cfg.get('hbo_only', True),
            hbr_only=data_cfg.get('hbr_only', False),
            fnirs_signal=_resolve_fnirs_signal_from_config(data_cfg),
            segmentation_mode=params.get('segmentation_mode', 'auto'),
        )

    raise NotImplementedError(f'Unified unimodal dataset factory does not support dataset {dataset_id!r} yet.')


def create_multimodal_window_dataset(
    data_cfg: Dict[str, Any],
    subject_ids: List[int],
    *,
    window_duration_s: float,
    normalize: bool,
    normalization_mode: str,
):
    dataset_id = resolve_dataset_id(data_cfg)
    params = _dataset_params(data_cfg)

    if dataset_id == 'eeg_fnirs_single_trial':
        return MultiModalEEGfNIRSDataset(
            data_root=data_cfg['data_root'],
            subject_ids=subject_ids,
            task=data_cfg.get('task', 'motor_imagery'),
            window_duration_s=window_duration_s,
            window_offset_ms=float(data_cfg['window'].get('offset_ms', 0)),
            normalize=normalize,
            normalization_mode=normalization_mode,
            eeg_preprocessing=data_cfg.get('eeg_preprocessing', {}),
            fnirs_preprocessing=data_cfg.get('fnirs_preprocessing', {}),
            exclude_eog=data_cfg.get('exclude_eog', True),
            hbo_only=data_cfg.get('hbo_only', True),
            hbr_only=data_cfg.get('hbr_only', False),
        )

    if dataset_id == 'simultaneous_eeg_nirs':
        return SimultaneousMultiModalDataset(
            data_root=data_cfg['data_root'],
            subject_ids=subject_ids,
            task=data_cfg.get('task', 'wg'),
            window_duration_s=window_duration_s,
            window_offset_ms=float(data_cfg['window'].get('offset_ms', 0)),
            normalize=normalize,
            normalization_mode=normalization_mode,
            eeg_preprocessing=data_cfg.get('eeg_preprocessing', data_cfg.get('preprocessing', {})),
            fnirs_preprocessing=data_cfg.get('fnirs_preprocessing', data_cfg.get('preprocessing', {})),
            exclude_eog=data_cfg.get('exclude_eog', True),
            hbo_only=data_cfg.get('hbo_only', True),
            hbr_only=data_cfg.get('hbr_only', False),
            fnirs_signal=_resolve_fnirs_signal_from_config(data_cfg),
            segmentation_mode=params.get('segmentation_mode', 'auto'),
        )

    raise NotImplementedError(f'Unified multimodal dataset factory does not support dataset {dataset_id!r} yet.')


def create_configured_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    data_cfg = config['data']
    normalize, normalization_mode = resolve_normalization_config(data_cfg)

    if split == 'train':
        subjects = data_cfg['split']['train_subjects']
        shuffle = True
    elif split == 'val':
        subjects = data_cfg['split']['val_subjects']
        shuffle = False
    else:
        subjects = data_cfg['split']['test_subjects']
        shuffle = False

    dataset = create_unimodal_window_dataset(
        data_cfg,
        subjects,
        data_cfg['modality'],
        window_samples=int(data_cfg['window']['length']),
        normalize=normalize,
        normalization_mode=normalization_mode,
    )

    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=True,
        drop_last=split == 'train',
    )


def create_configured_multimodal_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    data_cfg = config['data']
    normalize, normalization_mode = resolve_normalization_config(data_cfg)
    dataset_id = resolve_dataset_id(data_cfg)

    if dataset_id == 'eeg_fnirs_single_trial':
        return create_single_trial_dataloaders(
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

    splits = {
        'train': data_cfg['split']['train_subjects'],
        'val': data_cfg['split']['val_subjects'],
        'test': data_cfg['split']['test_subjects'],
    }
    dataloaders: Dict[str, DataLoader] = {}
    for split_name, subjects in splits.items():
        dataset = create_multimodal_window_dataset(
            data_cfg,
            subjects,
            window_duration_s=float(data_cfg['window']['duration_s']),
            normalize=normalize,
            normalization_mode=normalization_mode,
        )
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=(split_name == 'train'),
            num_workers=data_cfg.get('num_workers', 0),
            pin_memory=True,
            drop_last=(split_name == 'train'),
        )
    return dataloaders


def create_continuous_visualization_dataset(
    data_cfg: Dict[str, Any],
    modality: str,
    subject_id: int,
    *,
    normalize: bool,
    normalization_mode: str,
):
    dataset_id = resolve_dataset_id(data_cfg)
    params = _dataset_params(data_cfg)

    if dataset_id == 'eeg_fnirs_single_trial':
        preprocessing_key = 'eeg_preprocessing' if modality == 'eeg' else 'fnirs_preprocessing'
        window_length = data_cfg.get('window', {}).get('length', 1)
        return EEGfNIRSDataset(
            data_root=data_cfg['data_root'],
            subject_ids=[subject_id],
            task=data_cfg.get('task', 'motor_imagery'),
            modality=modality,
            window_samples=int(window_length),
            window_offset_ms=float(data_cfg.get('window', {}).get('offset_ms', 0)),
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=data_cfg.get(preprocessing_key, data_cfg.get('preprocessing', {})),
            exclude_eog=data_cfg.get('exclude_eog', True),
            hbo_only=data_cfg.get('hbo_only', True),
            hbr_only=data_cfg.get('hbr_only', False),
        )

    if dataset_id == 'simultaneous_eeg_nirs':
        preprocessing_key = 'eeg_preprocessing' if modality == 'eeg' else 'fnirs_preprocessing'
        visualization_segmentation_mode = params.get('visualization_segmentation_mode', 'auto')
        if visualization_segmentation_mode == 'auto':
            visualization_segmentation_mode = resolve_segmentation_mode(
                data_cfg.get('task', 'nback'),
                'both',
                params.get('segmentation_mode', 'auto'),
            )
        return SimultaneousContinuousDataset(
            data_root=data_cfg['data_root'],
            task=data_cfg.get('task', 'nback'),
            modality=modality,
            subject_ids=[subject_id],
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=data_cfg.get(preprocessing_key, data_cfg.get('preprocessing', {})),
            exclude_eog=data_cfg.get('exclude_eog', True),
            fnirs_signal=_resolve_fnirs_signal_from_config(data_cfg),
            segmentation_mode=visualization_segmentation_mode,
        )

    raise NotImplementedError(f'Continuous visualization dataset factory does not support dataset {dataset_id!r} yet.')
