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
from .signal_visualization import (
    SegmentSnapshot,
    SignalSegmentVisualizer,
    EEG_FREQUENCY_BANDS,
    FNIRS_FREQUENCY_BANDS,
    resolve_preprocessing_config,
    summarize_preprocessing_config,
    apply_preprocessing_config,
    compute_power_spectrum,
    compute_differential_entropy,
    extract_dataset_segment,
    extract_multimodal_dataset_segment,
    visualize_dataset_sample,
    visualize_filtered_dataset_sample,
    visualize_multimodal_dataset_sample,
    visualize_synchronized_filtered_sample,
)
from .synthetic_timeseries import PIDTimeSeriesDataset

__all__ = [
    'BBCIDataLoader',
    'EEGfNIRSDataset', 
    'MultiModalEEGfNIRSDataset',
    'create_dataloaders',
    'TrialInfo',
    'SyncInfo',
    'SegmentSnapshot',
    'SignalSegmentVisualizer',
    'EEG_FREQUENCY_BANDS',
    'FNIRS_FREQUENCY_BANDS',
    'resolve_preprocessing_config',
    'summarize_preprocessing_config',
    'apply_preprocessing_config',
    'compute_power_spectrum',
    'compute_differential_entropy',
    'extract_dataset_segment',
    'extract_multimodal_dataset_segment',
    'visualize_dataset_sample',
    'visualize_filtered_dataset_sample',
    'visualize_multimodal_dataset_sample',
    'visualize_synchronized_filtered_sample',
    'PIDTimeSeriesDataset',
]
