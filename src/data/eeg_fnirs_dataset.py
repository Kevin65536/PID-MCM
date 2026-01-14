"""
EEG+NIRS Single-Trial Dataset Loader

This module implements data loading for the TU Berlin EEG+NIRS BCI dataset.
Dataset URL: https://doc.ml.tu-berlin.de/hBCI/

Key findings from data exploration (2026-01-14):
- EEG and NIRS markers have a fixed offset (~51-55s) due to different recording start times
- Event intervals match well between modalities (mean diff ~0ms, std ~40ms)
- Labels are identical between modalities for corresponding events
- Session 0,2,4: Motor Imagery (LMI/RMI), Session 1,3,5: Mental Arithmetic (MA/BL)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class TrialInfo:
    """Information about a single trial."""
    subject_id: int
    session_idx: int
    trial_idx: int
    label: int  # 0 or 1 (left/right for MI, arithmetic/baseline for MA)
    task_type: str  # 'motor_imagery' or 'mental_arithmetic'
    eeg_start_sample: int
    eeg_end_sample: int
    nirs_start_sample: int
    nirs_end_sample: int
    onset_time_ms: float  # Event onset time in ms


@dataclass  
class SyncInfo:
    """Synchronization information between EEG and NIRS."""
    offset_ms: float  # NIRS time - EEG time at first event
    interval_diff_mean_ms: float
    interval_diff_std_ms: float
    interval_diff_max_ms: float
    labels_match: bool


def load_mat_struct(filepath: str) -> dict:
    """Load a .mat file with struct_as_record=False for easier access."""
    return sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)


class BBCIDataLoader:
    """
    Loader for BBCI Toolbox format data (EEG+NIRS Single-Trial dataset).
    
    Data structure:
    - cnt: continuous data, shape (n_samples, n_channels)
    - mrk: markers with time (ms), y (one-hot labels), className
    - mnt: montage with channel locations
    
    Sessions:
    - 0, 2, 4: Motor Imagery (left/right hand)
    - 1, 3, 5: Mental Arithmetic (arithmetic/baseline)
    """
    
    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[int]] = None,
        task: Literal['motor_imagery', 'mental_arithmetic', 'both'] = 'motor_imagery',
        modality: Literal['eeg', 'fnirs', 'both'] = 'both',
        use_artifact_data: bool = True,  # Use data with ocular artifacts (recommended)
    ):
        """
        Initialize the data loader.
        
        Args:
            data_root: Path to 'EEG+NIRS Single-Trial' directory
            subject_ids: List of subject IDs (1-29), None for all
            task: Which task to load
            modality: Which modality to load
            use_artifact_data: If True, use 'with occular artifact' folder for EEG
        """
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids or list(range(1, 30))
        self.task = task
        self.modality = modality
        self.use_artifact_data = use_artifact_data
        
        # Session indices for each task
        self.mi_sessions = [0, 2, 4]  # Motor Imagery
        self.ma_sessions = [1, 3, 5]  # Mental Arithmetic
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self):
        """Check that data directories exist."""
        eeg_dir = self.data_root / 'EEG_01-29'
        nirs_dir = self.data_root / 'NIRS_01-29'
        
        if self.modality in ['eeg', 'both'] and not eeg_dir.exists():
            raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")
        if self.modality in ['fnirs', 'both'] and not nirs_dir.exists():
            raise FileNotFoundError(f"NIRS directory not found: {nirs_dir}")
            
    def _get_subject_dir(self, subject_id: int, modality: str) -> Path:
        """Get the directory for a specific subject and modality."""
        subdir = 'EEG_01-29' if modality == 'eeg' else 'NIRS_01-29'
        subject_dir = self.data_root / subdir / f'subject {subject_id:02d}'
        
        if modality == 'eeg' and self.use_artifact_data:
            artifact_dir = subject_dir / 'with occular artifact'
            if artifact_dir.exists():
                return artifact_dir
        return subject_dir
    
    def load_subject_data(
        self, 
        subject_id: int,
        modality: str
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load all session data for a subject.
        
        Returns:
            cnt_list: List of 6 continuous data arrays, shape (n_samples, n_channels)
            mrk_list: List of 6 marker dicts with 'time', 'y', 'className'
            info: Dict with 'fs', 'clab' (channel labels)
        """
        subject_dir = self._get_subject_dir(subject_id, modality)
        
        cnt_data = load_mat_struct(str(subject_dir / 'cnt.mat'))
        mrk_data = load_mat_struct(str(subject_dir / 'mrk.mat'))
        
        cnt_list = []
        mrk_list = []
        
        for i in range(6):
            cnt_struct = cnt_data['cnt'][i]
            mrk_struct = mrk_data['mrk'][i]
            
            cnt_list.append(cnt_struct.x)
            mrk_list.append({
                'time': mrk_struct.time,
                'y': mrk_struct.y,
                'className': mrk_struct.className,
            })
        
        # Get metadata from first session
        info = {
            'fs': cnt_data['cnt'][0].fs,
            'clab': cnt_data['cnt'][0].clab,
        }
        
        return cnt_list, mrk_list, info
    
    def check_synchronization(
        self, 
        subject_id: int,
        session_idx: int = 0
    ) -> SyncInfo:
        """
        Check synchronization between EEG and NIRS markers for a session.
        
        Returns:
            SyncInfo with offset and alignment statistics
        """
        _, eeg_mrk, _ = self.load_subject_data(subject_id, 'eeg')
        _, nirs_mrk, _ = self.load_subject_data(subject_id, 'fnirs')
        
        eeg_time = eeg_mrk[session_idx]['time']
        nirs_time = nirs_mrk[session_idx]['time']
        
        # Check offset at first event
        offset_ms = float(nirs_time[0] - eeg_time[0])
        
        # Check interval alignment
        eeg_intervals = np.diff(eeg_time)
        nirs_intervals = np.diff(nirs_time)
        interval_diff = nirs_intervals - eeg_intervals
        
        # Check label matching
        eeg_labels = np.argmax(eeg_mrk[session_idx]['y'], axis=0)
        nirs_labels = np.argmax(nirs_mrk[session_idx]['y'], axis=0)
        labels_match = bool(np.all(eeg_labels == nirs_labels))
        
        return SyncInfo(
            offset_ms=offset_ms,
            interval_diff_mean_ms=float(np.mean(interval_diff)),
            interval_diff_std_ms=float(np.std(interval_diff)),
            interval_diff_max_ms=float(np.max(np.abs(interval_diff))),
            labels_match=labels_match,
        )
    
    def check_all_synchronization(self) -> Dict[int, List[SyncInfo]]:
        """Check synchronization for all subjects and sessions."""
        results = {}
        for subject_id in self.subject_ids:
            results[subject_id] = []
            for session_idx in range(6):
                try:
                    sync_info = self.check_synchronization(subject_id, session_idx)
                    results[subject_id].append(sync_info)
                except Exception as e:
                    print(f"Error checking subject {subject_id} session {session_idx}: {e}")
                    results[subject_id].append(None)
        return results
    

class EEGfNIRSDataset(Dataset):
    """
    PyTorch Dataset for EEG+NIRS Single-Trial data.
    
    Extracts fixed-length windows around event markers for training tokenizers.
    """
    
    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[int]] = None,
        task: Literal['motor_imagery', 'mental_arithmetic'] = 'motor_imagery',
        modality: Literal['eeg', 'fnirs'] = 'eeg',
        window_samples: int = 512,  # 2.56s @ 200Hz for EEG, ~51s @ 10Hz for NIRS
        window_offset_ms: float = 0,  # Offset from event onset (can be negative for pre-stimulus)
        normalize: bool = True,
        use_artifact_data: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_root: Path to 'EEG+NIRS Single-Trial' directory
            subject_ids: List of subject IDs (1-29), None for all
            task: Which task to use
            modality: 'eeg' or 'fnirs'
            window_samples: Number of samples per window
            window_offset_ms: Offset from event onset in milliseconds
            normalize: Whether to z-score normalize each window
            use_artifact_data: If True, use 'with occular artifact' folder for EEG
        """
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids or list(range(1, 30))
        self.task = task
        self.modality = modality
        self.window_samples = window_samples
        self.window_offset_ms = window_offset_ms
        self.normalize = normalize
        self.use_artifact_data = use_artifact_data
        
        # Loader for data access
        self.loader = BBCIDataLoader(
            data_root=data_root,
            subject_ids=self.subject_ids,
            task=task,
            modality=modality,
            use_artifact_data=use_artifact_data,
        )
        
        # Session indices for task
        if task == 'motor_imagery':
            self.session_indices = [0, 2, 4]
        else:
            self.session_indices = [1, 3, 5]
        
        # Build trial index
        self.trials: List[TrialInfo] = []
        self._build_trial_index()
        
        # Cache for loaded data
        self._data_cache: Dict[int, Tuple[List[np.ndarray], List[dict], dict]] = {}
        
    def _build_trial_index(self):
        """Build an index of all available trials."""
        for subject_id in self.subject_ids:
            try:
                _, mrk_list, info = self.loader.load_subject_data(subject_id, self.modality)
                fs = info['fs']
                
                for session_idx in self.session_indices:
                    mrk = mrk_list[session_idx]
                    n_trials = len(mrk['time'])
                    labels = np.argmax(mrk['y'], axis=0)
                    
                    for trial_idx in range(n_trials):
                        onset_time_ms = mrk['time'][trial_idx]
                        
                        # Calculate sample indices
                        offset_samples = int(self.window_offset_ms * fs / 1000)
                        start_sample = int(onset_time_ms * fs / 1000) + offset_samples
                        end_sample = start_sample + self.window_samples
                        
                        trial = TrialInfo(
                            subject_id=subject_id,
                            session_idx=session_idx,
                            trial_idx=trial_idx,
                            label=int(labels[trial_idx]),
                            task_type=self.task,
                            eeg_start_sample=start_sample if self.modality == 'eeg' else 0,
                            eeg_end_sample=end_sample if self.modality == 'eeg' else 0,
                            nirs_start_sample=start_sample if self.modality == 'fnirs' else 0,
                            nirs_end_sample=end_sample if self.modality == 'fnirs' else 0,
                            onset_time_ms=onset_time_ms,
                        )
                        self.trials.append(trial)
                        
            except Exception as e:
                print(f"Warning: Could not load subject {subject_id}: {e}")
                
    def _get_subject_data(self, subject_id: int) -> Tuple[List[np.ndarray], List[dict], dict]:
        """Get cached or load subject data."""
        if subject_id not in self._data_cache:
            cnt_list, mrk_list, info = self.loader.load_subject_data(subject_id, self.modality)
            self._data_cache[subject_id] = (cnt_list, mrk_list, info)
        return self._data_cache[subject_id]
    
    def __len__(self) -> int:
        return len(self.trials)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trial.
        
        Returns:
            Dict with:
                - 'data': tensor of shape (n_channels, window_samples)
                - 'label': tensor of shape () with class index
                - 'subject_id': int
                - 'session_idx': int
                - 'trial_idx': int
        """
        trial = self.trials[idx]
        cnt_list, _, _ = self._get_subject_data(trial.subject_id)
        
        # Get continuous data for this session
        cnt = cnt_list[trial.session_idx]  # shape: (n_samples, n_channels)
        
        # Extract window
        if self.modality == 'eeg':
            start = trial.eeg_start_sample
            end = trial.eeg_end_sample
        else:
            start = trial.nirs_start_sample
            end = trial.nirs_end_sample
        
        # Handle boundary cases
        if start < 0:
            start = 0
            end = self.window_samples
        if end > cnt.shape[0]:
            end = cnt.shape[0]
            start = max(0, end - self.window_samples)
            
        window = cnt[start:end, :]  # shape: (window_samples, n_channels)
        
        # Pad if necessary
        if window.shape[0] < self.window_samples:
            pad_size = self.window_samples - window.shape[0]
            window = np.pad(window, ((0, pad_size), (0, 0)), mode='constant')
        
        # Transpose to (n_channels, window_samples)
        window = window.T
        
        # Normalize
        if self.normalize:
            mean = window.mean(axis=1, keepdims=True)
            std = window.std(axis=1, keepdims=True) + 1e-8
            window = (window - mean) / std
        
        return {
            'data': torch.from_numpy(window).float(),
            'label': torch.tensor(trial.label, dtype=torch.long),
            'subject_id': trial.subject_id,
            'session_idx': trial.session_idx,
            'trial_idx': trial.trial_idx,
        }
    
    def get_channel_names(self) -> List[str]:
        """Get the channel names for the current modality."""
        if len(self.trials) == 0:
            return []
        first_subject = self.trials[0].subject_id
        _, _, info = self._get_subject_data(first_subject)
        return list(info['clab'])
    
    def get_sample_rate(self) -> float:
        """Get the sampling rate for the current modality."""
        if len(self.trials) == 0:
            return 0.0
        first_subject = self.trials[0].subject_id
        _, _, info = self._get_subject_data(first_subject)
        return float(info['fs'])


class MultiModalEEGfNIRSDataset(Dataset):
    """
    PyTorch Dataset for synchronized EEG+NIRS data.
    
    Returns aligned windows from both modalities for the same trial.
    Note: Due to different sampling rates, window durations are matched, not sample counts.
    """
    
    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[int]] = None,
        task: Literal['motor_imagery', 'mental_arithmetic'] = 'motor_imagery',
        window_duration_s: float = 2.5,  # Window duration in seconds
        window_offset_ms: float = 0,
        normalize: bool = True,
        use_artifact_data: bool = True,
    ):
        """
        Initialize the multimodal dataset.
        
        Args:
            data_root: Path to 'EEG+NIRS Single-Trial' directory
            subject_ids: List of subject IDs (1-29), None for all
            task: Which task to use
            window_duration_s: Window duration in seconds (applied to both modalities)
            window_offset_ms: Offset from event onset in milliseconds
            normalize: Whether to z-score normalize each window
            use_artifact_data: If True, use 'with occular artifact' folder for EEG
        """
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids or list(range(1, 30))
        self.task = task
        self.window_duration_s = window_duration_s
        self.window_offset_ms = window_offset_ms
        self.normalize = normalize
        
        # Loaders for both modalities
        self.eeg_loader = BBCIDataLoader(
            data_root=data_root,
            subject_ids=self.subject_ids,
            task=task,
            modality='eeg',
            use_artifact_data=use_artifact_data,
        )
        self.nirs_loader = BBCIDataLoader(
            data_root=data_root,
            subject_ids=self.subject_ids,
            task=task,
            modality='fnirs',
            use_artifact_data=use_artifact_data,
        )
        
        # Session indices for task
        if task == 'motor_imagery':
            self.session_indices = [0, 2, 4]
        else:
            self.session_indices = [1, 3, 5]
        
        # Build trial index
        self.trials: List[TrialInfo] = []
        self._build_trial_index()
        
        # Cache for loaded data
        self._eeg_cache: Dict[int, Tuple[List[np.ndarray], List[dict], dict]] = {}
        self._nirs_cache: Dict[int, Tuple[List[np.ndarray], List[dict], dict]] = {}
        
    def _build_trial_index(self):
        """Build an index of all available trials with both modalities."""
        for subject_id in self.subject_ids:
            try:
                # Load both modalities to verify synchronization
                _, eeg_mrk_list, eeg_info = self.eeg_loader.load_subject_data(subject_id, 'eeg')
                _, nirs_mrk_list, nirs_info = self.nirs_loader.load_subject_data(subject_id, 'fnirs')
                
                eeg_fs = eeg_info['fs']
                nirs_fs = nirs_info['fs']
                
                for session_idx in self.session_indices:
                    eeg_mrk = eeg_mrk_list[session_idx]
                    nirs_mrk = nirs_mrk_list[session_idx]
                    
                    n_trials = len(eeg_mrk['time'])
                    eeg_labels = np.argmax(eeg_mrk['y'], axis=0)
                    nirs_labels = np.argmax(nirs_mrk['y'], axis=0)
                    
                    # Verify labels match
                    if not np.all(eeg_labels == nirs_labels):
                        print(f"Warning: Label mismatch for subject {subject_id} session {session_idx}")
                        continue
                    
                    for trial_idx in range(n_trials):
                        eeg_onset_ms = eeg_mrk['time'][trial_idx]
                        nirs_onset_ms = nirs_mrk['time'][trial_idx]
                        
                        # Calculate EEG sample indices
                        eeg_offset_samples = int(self.window_offset_ms * eeg_fs / 1000)
                        eeg_window_samples = int(self.window_duration_s * eeg_fs)
                        eeg_start = int(eeg_onset_ms * eeg_fs / 1000) + eeg_offset_samples
                        eeg_end = eeg_start + eeg_window_samples
                        
                        # Calculate NIRS sample indices
                        nirs_offset_samples = int(self.window_offset_ms * nirs_fs / 1000)
                        nirs_window_samples = int(self.window_duration_s * nirs_fs)
                        nirs_start = int(nirs_onset_ms * nirs_fs / 1000) + nirs_offset_samples
                        nirs_end = nirs_start + nirs_window_samples
                        
                        trial = TrialInfo(
                            subject_id=subject_id,
                            session_idx=session_idx,
                            trial_idx=trial_idx,
                            label=int(eeg_labels[trial_idx]),
                            task_type=self.task,
                            eeg_start_sample=eeg_start,
                            eeg_end_sample=eeg_end,
                            nirs_start_sample=nirs_start,
                            nirs_end_sample=nirs_end,
                            onset_time_ms=eeg_onset_ms,
                        )
                        self.trials.append(trial)
                        
            except Exception as e:
                print(f"Warning: Could not load subject {subject_id}: {e}")
    
    def _get_eeg_data(self, subject_id: int) -> Tuple[List[np.ndarray], List[dict], dict]:
        """Get cached or load EEG data."""
        if subject_id not in self._eeg_cache:
            cnt_list, mrk_list, info = self.eeg_loader.load_subject_data(subject_id, 'eeg')
            self._eeg_cache[subject_id] = (cnt_list, mrk_list, info)
        return self._eeg_cache[subject_id]
    
    def _get_nirs_data(self, subject_id: int) -> Tuple[List[np.ndarray], List[dict], dict]:
        """Get cached or load NIRS data."""
        if subject_id not in self._nirs_cache:
            cnt_list, mrk_list, info = self.nirs_loader.load_subject_data(subject_id, 'fnirs')
            self._nirs_cache[subject_id] = (cnt_list, mrk_list, info)
        return self._nirs_cache[subject_id]
    
    def _extract_window(
        self, 
        cnt: np.ndarray, 
        start: int, 
        end: int, 
        target_samples: int
    ) -> np.ndarray:
        """Extract and pad a window from continuous data."""
        # Handle boundary cases
        if start < 0:
            start = 0
            end = target_samples
        if end > cnt.shape[0]:
            end = cnt.shape[0]
            start = max(0, end - target_samples)
            
        window = cnt[start:end, :]  # shape: (window_samples, n_channels)
        
        # Pad if necessary
        if window.shape[0] < target_samples:
            pad_size = target_samples - window.shape[0]
            window = np.pad(window, ((0, pad_size), (0, 0)), mode='constant')
        
        # Transpose to (n_channels, window_samples)
        window = window.T
        
        # Normalize
        if self.normalize:
            mean = window.mean(axis=1, keepdims=True)
            std = window.std(axis=1, keepdims=True) + 1e-8
            window = (window - mean) / std
            
        return window
    
    def __len__(self) -> int:
        return len(self.trials)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trial with both modalities.
        
        Returns:
            Dict with:
                - 'eeg': tensor of shape (n_eeg_channels, eeg_window_samples)
                - 'fnirs': tensor of shape (n_nirs_channels, nirs_window_samples)
                - 'label': tensor of shape () with class index
                - 'subject_id': int
                - 'session_idx': int
                - 'trial_idx': int
        """
        trial = self.trials[idx]
        
        # Get EEG data
        eeg_cnt_list, _, eeg_info = self._get_eeg_data(trial.subject_id)
        eeg_cnt = eeg_cnt_list[trial.session_idx]
        eeg_window_samples = int(self.window_duration_s * eeg_info['fs'])
        eeg_window = self._extract_window(
            eeg_cnt, trial.eeg_start_sample, trial.eeg_end_sample, eeg_window_samples
        )
        
        # Get NIRS data
        nirs_cnt_list, _, nirs_info = self._get_nirs_data(trial.subject_id)
        nirs_cnt = nirs_cnt_list[trial.session_idx]
        nirs_window_samples = int(self.window_duration_s * nirs_info['fs'])
        nirs_window = self._extract_window(
            nirs_cnt, trial.nirs_start_sample, trial.nirs_end_sample, nirs_window_samples
        )
        
        return {
            'eeg': torch.from_numpy(eeg_window).float(),
            'fnirs': torch.from_numpy(nirs_window).float(),
            'label': torch.tensor(trial.label, dtype=torch.long),
            'subject_id': trial.subject_id,
            'session_idx': trial.session_idx,
            'trial_idx': trial.trial_idx,
        }


def create_dataloaders(
    data_root: str,
    modality: Literal['eeg', 'fnirs', 'both'] = 'eeg',
    task: Literal['motor_imagery', 'mental_arithmetic'] = 'motor_imagery',
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    test_subjects: Optional[List[int]] = None,
    window_samples: int = 512,
    window_duration_s: float = 2.5,
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders with subject-based splits.
    
    Args:
        data_root: Path to 'EEG+NIRS Single-Trial' directory
        modality: 'eeg', 'fnirs', or 'both' for multimodal
        task: 'motor_imagery' or 'mental_arithmetic'
        train_subjects: Subject IDs for training (default: 1-20)
        val_subjects: Subject IDs for validation (default: 21-25)
        test_subjects: Subject IDs for testing (default: 26-29)
        window_samples: Window size for single-modality (ignored if modality='both')
        window_duration_s: Window duration for multimodal
        batch_size: Batch size
        num_workers: DataLoader workers
        **kwargs: Additional arguments for dataset
        
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    # Default subject splits
    if train_subjects is None:
        train_subjects = list(range(1, 21))  # 1-20
    if val_subjects is None:
        val_subjects = list(range(21, 26))  # 21-25
    if test_subjects is None:
        test_subjects = list(range(26, 30))  # 26-29
    
    dataloaders = {}
    
    for split, subject_ids in [
        ('train', train_subjects),
        ('val', val_subjects),
        ('test', test_subjects)
    ]:
        if modality == 'both':
            dataset = MultiModalEEGfNIRSDataset(
                data_root=data_root,
                subject_ids=subject_ids,
                task=task,
                window_duration_s=window_duration_s,
                **kwargs
            )
        else:
            dataset = EEGfNIRSDataset(
                data_root=data_root,
                subject_ids=subject_ids,
                task=task,
                modality=modality,
                window_samples=window_samples,
                **kwargs
            )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return dataloaders


if __name__ == '__main__':
    # Test the data loader
    import sys
    
    data_root = 'data/EEG+NIRS Single-Trial'
    
    print("=" * 60)
    print("Testing BBCIDataLoader")
    print("=" * 60)
    
    loader = BBCIDataLoader(data_root, subject_ids=[1, 2])
    
    # Check synchronization
    print("\nChecking synchronization for Subject 1...")
    sync_info = loader.check_synchronization(1, 0)
    print(f"  Offset: {sync_info.offset_ms:.1f} ms")
    print(f"  Interval diff: mean={sync_info.interval_diff_mean_ms:.1f}, std={sync_info.interval_diff_std_ms:.1f}, max={sync_info.interval_diff_max_ms:.1f} ms")
    print(f"  Labels match: {sync_info.labels_match}")
    
    print("\n" + "=" * 60)
    print("Testing EEGfNIRSDataset (single modality)")
    print("=" * 60)
    
    dataset = EEGfNIRSDataset(
        data_root=data_root,
        subject_ids=[1, 2],
        task='motor_imagery',
        modality='eeg',
        window_samples=512,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample rate: {dataset.get_sample_rate()} Hz")
    print(f"Channels: {len(dataset.get_channel_names())} - {dataset.get_channel_names()[:5]}...")
    
    sample = dataset[0]
    print(f"Sample data shape: {sample['data'].shape}")
    print(f"Sample label: {sample['label']}")
    
    print("\n" + "=" * 60)
    print("Testing MultiModalEEGfNIRSDataset")
    print("=" * 60)
    
    mm_dataset = MultiModalEEGfNIRSDataset(
        data_root=data_root,
        subject_ids=[1, 2],
        task='motor_imagery',
        window_duration_s=2.5,
    )
    
    print(f"Dataset size: {len(mm_dataset)}")
    
    mm_sample = mm_dataset[0]
    print(f"EEG shape: {mm_sample['eeg'].shape}")
    print(f"NIRS shape: {mm_sample['fnirs'].shape}")
    print(f"Label: {mm_sample['label']}")
    
    print("\n" + "=" * 60)
    print("Testing create_dataloaders")
    print("=" * 60)
    
    dataloaders = create_dataloaders(
        data_root=data_root,
        modality='eeg',
        task='motor_imagery',
        train_subjects=[1, 2],
        val_subjects=[3],
        test_subjects=[4],
        window_samples=512,
        batch_size=8,
    )
    
    for split, dl in dataloaders.items():
        print(f"{split}: {len(dl.dataset)} samples, {len(dl)} batches")
    
    # Test a batch
    batch = next(iter(dataloaders['train']))
    print(f"Batch data shape: {batch['data'].shape}")
    print(f"Batch labels: {batch['label']}")
    
    print("\n✓ All tests passed!")
