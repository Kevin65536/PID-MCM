"""
Data loading and preprocessing modules.

Available datasets:
- EEGfNIRSDataset: Single modality (EEG or fNIRS) dataset
- MultiModalEEGfNIRSDataset: Synchronized EEG+fNIRS dataset
- PIDTimeSeriesDataset: Synthetic data for debugging
"""

from .eeg_fnirs_dataset import (
    BBCIDataLoader,
    EEGfNIRSDataset,
    MultiModalEEGfNIRSDataset,
    create_dataloaders,
    TrialInfo,
    SyncInfo,
)
from .synthetic_timeseries import PIDTimeSeriesDataset

__all__ = [
    'BBCIDataLoader',
    'EEGfNIRSDataset', 
    'MultiModalEEGfNIRSDataset',
    'create_dataloaders',
    'TrialInfo',
    'SyncInfo',
    'PIDTimeSeriesDataset',
]
