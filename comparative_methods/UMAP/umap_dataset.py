"""
UMAP Dataset Adapter for EEG + fNIRS

Wraps the project's MultiModalEEGfNIRSDataset to produce inputs
in UMAP's expected format:
    eeg:   (batch, seq_length, eeg_feature_dim)
    fnirs: (batch, seq_length, fnirs_feature_dim)

Two feature extraction modes:
  - 'channel_avg': Segment trial into seq_length windows, average over time per channel
  - 'band_power':  EEG frequency-band powers + fNIRS statistical features

Mapping to UMAP naming: "eeg" → EEG, "eye" → fNIRS
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Add project root to path (NOT src/ directly, to avoid shadowing 'tokenizers' package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset


class UMAPDataset(Dataset):
    """
    Adapter that converts (n_channels, n_samples) trial data into
    UMAP's expected (seq_length, feature_dim) format.
    """

    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[int]] = None,
        task: Literal['motor_imagery', 'mental_arithmetic'] = 'motor_imagery',
        seq_length: int = 5,
        window_duration_s: float = 10.0,
        feature_mode: Literal['channel_avg', 'band_power', 'de'] = 'channel_avg',
        normalize: bool = True,
        exclude_eog: bool = True,
        hbo_only: bool = True,
    ):
        self.seq_length = seq_length
        self.feature_mode = feature_mode
        self.normalize = normalize

        self.mm_dataset = MultiModalEEGfNIRSDataset(
            data_root=data_root,
            subject_ids=subject_ids,
            task=task,
            window_duration_s=window_duration_s,
            normalize=False,  # We normalize after feature extraction
            exclude_eog=exclude_eog,
            hbo_only=hbo_only,
        )

        # Metadata
        self.eeg_channels = self.mm_dataset.get_num_eeg_channels()
        self.fnirs_channels = self.mm_dataset.get_num_fnirs_channels()
        self.eeg_fs = self.mm_dataset.get_eeg_sample_rate()
        self.fnirs_fs = self.mm_dataset.get_fnirs_sample_rate()

        # Feature dimensions depend on extraction mode
        if feature_mode == 'channel_avg':
            self.eeg_input_dim = self.eeg_channels
            self.fnirs_input_dim = self.fnirs_channels
        elif feature_mode in ('band_power', 'de'):
            self.eeg_input_dim = self.eeg_channels * 5   # 5 frequency bands
            self.fnirs_input_dim = self.fnirs_channels * 3  # mean, slope, std
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

    def __len__(self) -> int:
        return len(self.mm_dataset)

    # ----- Feature extraction helpers -----

    def _segment_channel_avg(self, data: torch.Tensor, n_segments: int) -> torch.Tensor:
        """
        Segment (n_channels, n_samples) → (n_segments, n_channels).
        Average over time within each segment.
        """
        n_channels, n_samples = data.shape
        seg_len = n_samples // n_segments
        data = data[:, :seg_len * n_segments]
        # (n_channels, n_segments, seg_len) → mean → (n_channels, n_segments)
        data = data.reshape(n_channels, n_segments, seg_len).mean(dim=2)
        return data.permute(1, 0)  # (n_segments, n_channels)

    def _extract_band_power(self, eeg_data: torch.Tensor, n_segments: int) -> torch.Tensor:
        """
        Extract EEG band power features (log DE approximation).
        Bands: delta(0.5-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-45).
        Returns: (n_segments, n_channels * 5)
        """
        n_channels, n_samples = eeg_data.shape
        seg_len = n_samples // n_segments
        eeg_data = eeg_data[:, :seg_len * n_segments]

        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        fs = self.eeg_fs
        features = []

        for seg_idx in range(n_segments):
            segment = eeg_data[:, seg_idx * seg_len:(seg_idx + 1) * seg_len]
            seg_features = []
            fft_vals = torch.fft.rfft(segment, dim=1)
            freqs = torch.fft.rfftfreq(seg_len, d=1.0 / fs)

            for low, high in bands:
                mask = (freqs >= low) & (freqs <= high)
                band_power = (torch.abs(fft_vals[:, mask]) ** 2).mean(dim=1)
                seg_features.append(torch.log(band_power + 1e-8))

            features.append(torch.cat(seg_features, dim=0))

        return torch.stack(features, dim=0)

    def _extract_de_features(self, eeg_data: torch.Tensor, n_segments: int) -> torch.Tensor:
        """
        Extract EEG differential entropy (DE) features over canonical bands.
        DE per channel-band is computed as: 0.5 * log(2*pi*e*sigma^2).
        Returns: (n_segments, n_channels * 5)
        """
        n_channels, n_samples = eeg_data.shape
        seg_len = n_samples // n_segments
        eeg_data = eeg_data[:, :seg_len * n_segments]

        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        fs = self.eeg_fs
        eps = 1e-8
        features = []

        for seg_idx in range(n_segments):
            segment = eeg_data[:, seg_idx * seg_len:(seg_idx + 1) * seg_len]
            fft_vals = torch.fft.rfft(segment, dim=1)
            freqs = torch.fft.rfftfreq(seg_len, d=1.0 / fs)
            seg_features = []

            for low, high in bands:
                mask = (freqs >= low) & (freqs <= high)
                band_power = (torch.abs(fft_vals[:, mask]) ** 2).mean(dim=1)
                de = 0.5 * torch.log(2.0 * np.pi * np.e * (band_power + eps))
                seg_features.append(de)

            features.append(torch.cat(seg_features, dim=0))

        return torch.stack(features, dim=0)

    def _extract_fnirs_stats(self, fnirs_data: torch.Tensor, n_segments: int) -> torch.Tensor:
        """
        Extract fNIRS statistical features: mean, slope, std.
        Returns: (n_segments, n_channels * 3)
        """
        n_channels, n_samples = fnirs_data.shape
        seg_len = n_samples // n_segments
        fnirs_data = fnirs_data[:, :seg_len * n_segments]

        features = []
        for seg_idx in range(n_segments):
            segment = fnirs_data[:, seg_idx * seg_len:(seg_idx + 1) * seg_len]
            mean_feat = segment.mean(dim=1)
            std_feat = segment.std(dim=1)
            t = torch.arange(seg_len, dtype=segment.dtype, device=segment.device)
            t = t - t.mean()
            slope_feat = (segment * t.unsqueeze(0)).sum(dim=1) / ((t ** 2).sum() + 1e-8)
            features.append(torch.cat([mean_feat, slope_feat, std_feat], dim=0))

        return torch.stack(features, dim=0)

    # ----- Main accessor -----

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.mm_dataset[idx]
        eeg = sample['eeg']       # (n_eeg_ch, eeg_samples)
        fnirs = sample['fnirs']   # (n_fnirs_ch, fnirs_samples)
        label = sample['label']

        if self.feature_mode == 'channel_avg':
            eeg_features = self._segment_channel_avg(eeg, self.seq_length)
            fnirs_features = self._segment_channel_avg(fnirs, self.seq_length)
        elif self.feature_mode == 'band_power':
            eeg_features = self._extract_band_power(eeg, self.seq_length)
            fnirs_features = self._extract_fnirs_stats(fnirs, self.seq_length)
        elif self.feature_mode == 'de':
            eeg_features = self._extract_de_features(eeg, self.seq_length)
            fnirs_features = self._extract_fnirs_stats(fnirs, self.seq_length)

        if self.normalize:
            eeg_features = (eeg_features - eeg_features.mean(0, keepdim=True)) / (eeg_features.std(0, keepdim=True) + 1e-8)
            fnirs_features = (fnirs_features - fnirs_features.mean(0, keepdim=True)) / (fnirs_features.std(0, keepdim=True) + 1e-8)

        return {
            'eeg': eeg_features.float(),       # (seq_length, eeg_input_dim)
            'fnirs': fnirs_features.float(),    # (seq_length, fnirs_input_dim)
            'label': label,
            'subject_id': sample['subject_id'],
        }


def create_umap_dataloaders(
    data_root: str,
    task: str = 'motor_imagery',
    seq_length: int = 5,
    window_duration_s: float = 10.0,
    feature_mode: str = 'channel_avg',
    batch_size: int = 64,
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    test_subjects: Optional[List[int]] = None,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], dict]:
    """
    Create train/val/test DataLoaders with standard subject splits.
    Returns (dataloaders_dict, dataset_info_dict).
    """
    if train_subjects is None:
        train_subjects = list(range(1, 21))
    if val_subjects is None:
        val_subjects = list(range(21, 26))
    if test_subjects is None:
        test_subjects = list(range(26, 30))

    dataloaders = {}
    dataset_info = {}

    for split, subjects in [('train', train_subjects), ('val', val_subjects), ('test', test_subjects)]:
        ds = UMAPDataset(
            data_root=data_root,
            subject_ids=subjects,
            task=task,
            seq_length=seq_length,
            window_duration_s=window_duration_s,
            feature_mode=feature_mode,
        )
        dataloaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )
        if split == 'train':
            dataset_info = {
                'eeg_input_dim': ds.eeg_input_dim,
                'fnirs_input_dim': ds.fnirs_input_dim,
                'seq_length': seq_length,
                'n_train': len(ds),
            }

    return dataloaders, dataset_info


def create_umap_subject_dependent_dataloaders(
    data_root: str,
    task: str = 'motor_imagery',
    seq_length: int = 5,
    window_duration_s: float = 10.0,
    feature_mode: str = 'channel_avg',
    batch_size: int = 64,
    subject_ids: Optional[List[int]] = None,
    train_sessions: Optional[List[int]] = None,
    test_sessions: Optional[List[int]] = None,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], dict]:
    """
    Create train/val/test DataLoaders under subject-dependent protocol.

    Each subject contributes train/test samples according to session split.
    Validation split is sampled from the training pool with per-subject stratification.
    """
    if subject_ids is None:
        subject_ids = list(range(1, 30))
    if train_sessions is None:
        train_sessions = [0, 2]
    if test_sessions is None:
        test_sessions = [4]

    full_ds = UMAPDataset(
        data_root=data_root,
        subject_ids=subject_ids,
        task=task,
        seq_length=seq_length,
        window_duration_s=window_duration_s,
        feature_mode=feature_mode,
    )

    train_candidates = []
    test_indices = []
    for idx, trial in enumerate(full_ds.mm_dataset.trials):
        if trial.session_idx in test_sessions:
            test_indices.append(idx)
        elif trial.session_idx in train_sessions:
            train_candidates.append(idx)

    rng = np.random.default_rng(random_seed)
    train_indices = []
    val_indices = []

    by_subject = {}
    for idx in train_candidates:
        sid = int(full_ds.mm_dataset.trials[idx].subject_id)
        by_subject.setdefault(sid, []).append(idx)

    for sid, idxs in by_subject.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        n_val = min(n_val, len(idxs) - 1) if len(idxs) > 1 else 0
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])

    dataloaders = {
        'train': DataLoader(
            Subset(full_ds, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        'val': DataLoader(
            Subset(full_ds, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        'test': DataLoader(
            Subset(full_ds, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    }

    dataset_info = {
        'eeg_input_dim': full_ds.eeg_input_dim,
        'fnirs_input_dim': full_ds.fnirs_input_dim,
        'seq_length': seq_length,
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'n_test': len(test_indices),
        'split_mode': 'subject_dependent',
    }

    return dataloaders, dataset_info


def collate_missing_modality(batch, mode='multi'):
    """Custom collate for missing-modality finetuning."""
    eeg_list, fnirs_list, labels, subject_ids = [], [], [], []

    for item in batch:
        if mode in ('multi', 'eeg'):
            eeg_list.append(item['eeg'])
        if mode in ('multi', 'eye'):
            fnirs_list.append(item['fnirs'])
        labels.append(item['label'])
        subject_ids.append(item['subject_id'])

    return {
        'eeg': torch.stack(eeg_list) if eeg_list else None,
        'fnirs': torch.stack(fnirs_list) if fnirs_list else None,
        'label': torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels),
        'subject_id': subject_ids,
    }
