#!/usr/bin/env python
"""
Unified Downstream Task Training Script

This script provides a standardized training pipeline for downstream classification tasks
using pre-trained tokenizers (or raw signals as baseline).

Supports:
1. Single modality: EEG or fNIRS
2. Multi-modality: EEG + fNIRS fusion
3. Token-based or raw signal classification
4. BCI-standard 4s window for Motor Imagery

Usage:
    # Token-based classification
    python train_downstream.py --config downstream/mi_eeg_token.yaml
    python train_downstream.py --config downstream/mi_fnirs_token.yaml
    python train_downstream.py --config downstream/mi_multimodal_token.yaml
    
    # Raw signal baseline
    python train_downstream.py --config downstream/mi_eeg_raw_baseline.yaml
    python train_downstream.py --config downstream/mi_fnirs_raw_baseline.yaml
    
Background run:
    nohup python train_downstream.py --config downstream/mi_eeg_token.yaml &

TensorBoard:
    tensorboard --logdir experiments/runs
"""

import sys
import os
import argparse
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache

# Fix OMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.eeg_fnirs_dataset import (
    BBCIDataLoader,
    EEGfNIRSDataset,
    get_eeg_channel_mask,
    get_fnirs_channel_mask,
)
from src.data.registry import load_experiment_config, normalize_data_config
from src.data.factory import create_multimodal_window_dataset, create_unimodal_window_dataset
from src.data.augmentation import SignalAugmentor, DualModalityAugmentor, LabelSmoothingCrossEntropy, create_augmentor_from_config
from src.tokenizers import create_tokenizer, list_tokenizers
from src.classifiers.multi_lead import MultiLeadClassifier, DualModalityMultiLeadClassifier
from src.utils.logger import ExperimentLogger
from src.visualization import TokenizerVisualizer, TensorBoardLogger
from src.visualization.classifier_plots import visualize_classifier_run


# ============================================================================
# Logging Utilities
# ============================================================================

class TeeLogger:
    """Write output to both terminal and file."""
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a', buffering=1)
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


def setup_logging(run_dir: Path) -> TeeLogger:
    """Setup logging to terminal and file."""
    log_file = run_dir / "training.log"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee


# ============================================================================
# GPU Selection Utilities
# ============================================================================

def get_gpu_info() -> List[Dict[str, Any]]:
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            return []
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': float(parts[2]),
                    'memory_total': float(parts[3]),
                    'memory_free': float(parts[4]),
                    'utilization': float(parts[5]) if parts[5] != '[N/A]' else 0.0,
                })
        return gpus
    except Exception:
        return []


def select_best_gpu(verbose: bool = True) -> Optional[int]:
    """Select the best available GPU based on memory and utilization."""
    gpus = get_gpu_info()
    
    if not gpus:
        if verbose:
            print("No GPUs found, falling back to default")
        return None
    
    if verbose:
        print("\n" + "=" * 60)
        print("GPU Status:")
        print("-" * 60)
        for gpu in gpus:
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB")
            print(f"    Utilization: {gpu['utilization']:.1f}%")
        print("-" * 60)
    
    # Score by free memory and low utilization
    def score(gpu):
        return gpu['memory_free'] - gpu['utilization'] * 10
    
    best = max(gpus, key=score)
    
    if verbose:
        print(f"Selected GPU {best['index']}: {best['name']}")
        print("=" * 60 + "\n")
    
    return best['index']


def setup_device(config: dict, verbose: bool = True) -> torch.device:
    """Setup the training device."""
    device_cfg = config['experiment'].get('device', 'cuda')
    
    if device_cfg.startswith('cuda:'):
        gpu_idx = int(device_cfg.split(':')[1])
        if torch.cuda.is_available() and gpu_idx < torch.cuda.device_count():
            return torch.device(device_cfg)
    
    if torch.cuda.is_available():
        best_gpu = select_best_gpu(verbose=verbose)
        if best_gpu is not None:
            torch.cuda.set_device(best_gpu)
            return torch.device(f'cuda:{best_gpu}')
        return torch.device('cuda')
    
    if verbose:
        print("CUDA not available, using CPU")
    return torch.device('cpu')


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration with base config inheritance."""
    config = load_experiment_config(config_path, configs_dir=project_root / 'experiments' / 'configs')
    return normalize_downstream_config(config)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_data_root(data_root: str) -> Path:
    """Resolve dataset root relative to project root when needed."""
    candidate = Path(data_root)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def get_config_subject_ids(data_cfg: dict) -> List[int]:
    """Collect subject ids from nested split config or legacy flat config."""
    subject_ids: List[int] = []
    split_cfg = build_split_config(data_cfg)
    for key in ['train_subjects', 'val_subjects', 'test_subjects']:
        subject_ids.extend(split_cfg.get(key, []))

    if not subject_ids and 'subject_ids' in data_cfg:
        subject_ids.extend(data_cfg.get('subject_ids', []))

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(int(subject_id) for subject_id in subject_ids))


@lru_cache(maxsize=16)
def infer_num_channels_from_data(
    data_root: str,
    subject_id: int,
    modality: str,
    exclude_eog: bool,
    hbo_only: bool,
    hbr_only: bool,
) -> int:
    """Infer channel count from actual dataset metadata and channel filtering rules."""
    loader = BBCIDataLoader(str(resolve_data_root(data_root)), subject_ids=[subject_id], modality=modality)
    _, _, info = loader.load_subject_data(subject_id, modality)
    channel_names = list(info['clab'])

    if modality == 'eeg':
        channel_mask = get_eeg_channel_mask(channel_names, exclude_eog=exclude_eog)
    elif modality == 'fnirs':
        channel_mask = get_fnirs_channel_mask(channel_names, hbo_only=hbo_only, hbr_only=hbr_only)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return int(channel_mask.sum())


def infer_num_channels(modality: str, modality_cfg: dict, data_cfg: dict) -> int:
    """Infer actual lead count after channel filtering from dataset metadata."""
    if 'num_channels' in modality_cfg:
        return int(modality_cfg['num_channels'])

    data_root = data_cfg.get('data_root') or data_cfg.get('root')
    if not data_root:
        raise KeyError('data_root')

    subject_ids = get_config_subject_ids(data_cfg)
    if not subject_ids:
        raise ValueError('Cannot infer num_channels without at least one configured subject')

    return infer_num_channels_from_data(
        data_root=str(data_root),
        subject_id=int(subject_ids[0]),
        modality=modality,
        exclude_eog=bool(modality_cfg.get('exclude_eog', True)),
        hbo_only=bool(modality_cfg.get('hbo_only', True)),
        hbr_only=bool(modality_cfg.get('hbr_only', False)),
    )


def build_split_config(data_cfg: dict) -> dict:
    """Support both nested split config and flat train/val/test subject lists."""
    if 'split' in data_cfg:
        return data_cfg['split']

    split = {}
    if 'train_subjects' in data_cfg:
        split['train_subjects'] = data_cfg['train_subjects']
    if 'val_subjects' in data_cfg:
        split['val_subjects'] = data_cfg['val_subjects']
    if 'test_subjects' in data_cfg:
        split['test_subjects'] = data_cfg['test_subjects']
    return split


def build_modality_section(modality: str, source_cfg: dict) -> dict:
    """Create the modality-specific subsection expected by the training script."""
    window_cfg = source_cfg.get('window', {})
    section = {
        'window_samples': source_cfg.get('window_samples', window_cfg.get('length')),
        'preprocessing': source_cfg.get('preprocessing', {}),
    }

    if modality == 'eeg':
        section['exclude_eog'] = source_cfg.get('exclude_eog', True)
    if modality == 'fnirs':
        section['hbo_only'] = source_cfg.get('hbo_only', True)
        section['hbr_only'] = source_cfg.get('hbr_only', False)

    section['num_channels'] = infer_num_channels(modality, section, source_cfg)
    return section


def normalize_downstream_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize legacy and current downstream configs to one internal schema."""
    normalized = dict(config)
    data_cfg = normalize_data_config(normalized['data'])

    modality = normalized.get('modality', data_cfg.get('modality', 'eeg'))
    normalized['modality'] = modality

    if 'data_root' not in data_cfg and 'root' in data_cfg:
        data_cfg['data_root'] = data_cfg['root']

    data_cfg['split'] = build_split_config(data_cfg)

    if modality != 'both' and modality not in data_cfg:
        data_cfg[modality] = build_modality_section(modality, data_cfg)
    elif modality == 'both':
        for mod in ['eeg', 'fnirs']:
            if mod in data_cfg:
                nested_cfg = dict(data_cfg[mod])
                data_cfg[mod] = build_modality_section(mod, deep_merge(data_cfg, nested_cfg))

    normalized['data'] = data_cfg

    classifier_cfg = dict(normalized.get('classifier', {}))
    classifier_type = classifier_cfg.get('type', 'end_to_end')
    normalized['use_raw'] = bool(normalized.get('use_raw', classifier_type == 'raw'))

    tokenizer_cfg = normalized.get('tokenizer', {})
    if not normalized['use_raw'] and modality != 'both':
        if isinstance(tokenizer_cfg, dict) and modality not in tokenizer_cfg:
            normalized['tokenizer'] = {modality: tokenizer_cfg}

    return normalized


def resolve_normalization_config(data_cfg: dict) -> Tuple[bool, str]:
    """Resolve whether normalization is enabled and which mode to use."""
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


# ============================================================================
# Dataset Creation
# ============================================================================

class MultiLeadDataset(torch.utils.data.Dataset):
    """Dataset that returns multi-lead data (no channel averaging)."""
    
    def __init__(
        self,
        data_root: str,
        subject_ids: list,
        modality: str,
        window_samples: int,
        task: str = 'motor_imagery',
        normalize: bool = True,
        normalization_mode: str = 'session',
        preprocessing: Optional[dict] = None,
        window_offset_ms: float = 500,
        exclude_eog: bool = True,
        hbo_only: bool = True,
    ):
        self.base_dataset = EEGfNIRSDataset(
            data_root=data_root,
            subject_ids=subject_ids,
            task=task,
            modality=modality,
            window_samples=window_samples,
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=preprocessing,
            window_offset_ms=window_offset_ms,
            exclude_eog=exclude_eog,
            hbo_only=hbo_only,
        )
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]


class DualModalityDataset(torch.utils.data.Dataset):
    """Dataset that returns aligned multi-lead data for both EEG and fNIRS."""
    
    def __init__(
        self,
        data_root: str,
        subject_ids: list,
        eeg_config: dict,
        fnirs_config: dict,
        task: str = 'motor_imagery',
        normalize: bool = True,
        normalization_mode: str = 'session',
        window_offset_ms: float = 500,
    ):
        self.eeg_dataset = EEGfNIRSDataset(
            data_root=data_root,
            subject_ids=subject_ids,
            task=task,
            modality='eeg',
            window_samples=eeg_config['window_samples'],
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=eeg_config.get('preprocessing', {}),
            window_offset_ms=window_offset_ms,
            exclude_eog=eeg_config.get('exclude_eog', True),
        )
        
        self.fnirs_dataset = EEGfNIRSDataset(
            data_root=data_root,
            subject_ids=subject_ids,
            task=task,
            modality='fnirs',
            window_samples=fnirs_config['window_samples'],
            normalize=normalize,
            normalization_mode=normalization_mode,
            preprocessing=fnirs_config.get('preprocessing', {}),
            window_offset_ms=window_offset_ms,
            hbo_only=fnirs_config.get('hbo_only', True),
        )
        
        self._build_alignment()
    
    def _build_alignment(self):
        """Build index alignment between EEG and fNIRS."""
        eeg_lookup = {}
        for i, trial in enumerate(self.eeg_dataset.trials):
            key = (trial.subject_id, trial.session_idx, trial.trial_idx)
            eeg_lookup[key] = i
        
        self.aligned_indices = []
        for i, trial in enumerate(self.fnirs_dataset.trials):
            key = (trial.subject_id, trial.session_idx, trial.trial_idx)
            if key in eeg_lookup:
                self.aligned_indices.append((eeg_lookup[key], i))
    
    def __len__(self):
        return len(self.aligned_indices)
    
    def __getitem__(self, idx):
        eeg_idx, fnirs_idx = self.aligned_indices[idx]
        
        eeg_sample = self.eeg_dataset[eeg_idx]
        fnirs_sample = self.fnirs_dataset[fnirs_idx]
        
        return {
            'eeg': eeg_sample['data'],
            'fnirs': fnirs_sample['data'],
            'label': eeg_sample['label'],
            'subject_id': eeg_sample['subject_id'],
        }


def create_dataloaders(config: dict) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders based on config."""
    data_cfg = config['data']
    modality = config.get('modality', 'eeg')
    normalize, normalization_mode = resolve_normalization_config(data_cfg)
    
    splits = {
        'train': data_cfg['split']['train_subjects'],
        'val': data_cfg['split']['val_subjects'],
        'test': data_cfg['split']['test_subjects'],
    }
    
    dataloaders = {}
    
    for split_name, subjects in splits.items():
        if modality == 'both':
            dataset = create_multimodal_window_dataset(
                data_cfg,
                subjects,
                window_duration_s=float(data_cfg['eeg']['window_samples']) / float(data_cfg['eeg']['sample_rate']),
                normalize=normalize,
                normalization_mode=normalization_mode,
            )
        else:
            mod_cfg = data_cfg['eeg'] if modality == 'eeg' else data_cfg['fnirs']
            dataset = create_unimodal_window_dataset(
                data_cfg,
                subjects,
                modality,
                window_samples=mod_cfg['window_samples'],
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


# ============================================================================
# Tokenizer Loading
# ============================================================================

def load_tokenizer_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load a pre-trained tokenizer from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {checkpoint_path}")
    
    print(f"Loading tokenizer from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Create tokenizer using registry
    tokenizer = create_tokenizer(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
    else:
        tokenizer.load_state_dict(checkpoint)
    
    tokenizer.to(device)
    tokenizer.eval()
    
    print(f"  Loaded tokenizer type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"  Latent dim: {tokenizer.latent_dim}")
    
    return tokenizer


# ============================================================================
# Model Creation
# ============================================================================

def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create classifier model based on config."""
    modality = config.get('modality', 'eeg')
    use_raw = config.get('use_raw', False)
    classifier_cfg = config['classifier']
    
    if use_raw:
        # Import raw classifiers
        from src.classifiers.multi_lead import RawMultiLeadClassifier, RawDualModalityClassifier
        
        if modality == 'both':
            model = RawDualModalityClassifier(
                num_classes=classifier_cfg.get('num_classes', 2),
                eeg_num_leads=config['data']['eeg']['num_channels'],
                fnirs_num_leads=config['data']['fnirs']['num_channels'],
                eeg_input_length=config['data']['eeg']['window_samples'],
                fnirs_input_length=config['data']['fnirs']['window_samples'],
                aggregation=classifier_cfg.get('aggregation', 'attention'),
                fusion=config.get('fusion', {}).get('type', 'early'),
                hidden_dim=classifier_cfg.get('hidden_dim', 128),
                num_heads=classifier_cfg.get('num_heads', 4),
                num_layers=classifier_cfg.get('num_layers', 2),
                dropout=classifier_cfg.get('dropout', 0.1),
            )
        else:
            mod_cfg = config['data'][modality]
            model = RawMultiLeadClassifier(
                num_classes=classifier_cfg.get('num_classes', 2),
                num_leads=mod_cfg['num_channels'],
                input_length=mod_cfg['window_samples'],
                aggregation=classifier_cfg.get('aggregation', 'attention'),
                hidden_dim=classifier_cfg.get('hidden_dim', 128),
                num_heads=classifier_cfg.get('num_heads', 4),
                num_layers=classifier_cfg.get('num_layers', 2),
                dropout=classifier_cfg.get('dropout', 0.1),
            )
        return model.to(device)
    
    # Load tokenizer(s)
    tokenizer_cfg = config['tokenizer']
    
    if modality == 'both':
        # Dual modality
        eeg_checkpoint = project_root / tokenizer_cfg['eeg']['checkpoint']
        fnirs_checkpoint = project_root / tokenizer_cfg['fnirs']['checkpoint']
        
        eeg_tokenizer = load_tokenizer_from_checkpoint(eeg_checkpoint, device)
        fnirs_tokenizer = load_tokenizer_from_checkpoint(fnirs_checkpoint, device)
        
        model = DualModalityMultiLeadClassifier(
            eeg_tokenizer=eeg_tokenizer,
            fnirs_tokenizer=fnirs_tokenizer,
            num_classes=classifier_cfg.get('num_classes', 2),
            eeg_num_leads=config['data']['eeg']['num_channels'],
            fnirs_num_leads=config['data']['fnirs']['num_channels'],
            aggregation=classifier_cfg.get('aggregation', 'attention'),
            fusion=config.get('fusion', {}).get('type', 'early'),
            hidden_dim=classifier_cfg.get('hidden_dim', 128),
            num_heads=classifier_cfg.get('num_heads', 4),
            num_layers=classifier_cfg.get('num_layers', 2),
            dropout=classifier_cfg.get('dropout', 0.1),
            freeze_tokenizers=classifier_cfg.get('freeze_tokenizers', True),
        )
    else:
        # Single modality
        tok_cfg = tokenizer_cfg[modality]
        checkpoint = project_root / tok_cfg['checkpoint']
        
        tokenizer = load_tokenizer_from_checkpoint(checkpoint, device)
        
        mod_cfg = config['data'][modality]
        
        model = MultiLeadClassifier(
            tokenizer=tokenizer,
            num_classes=classifier_cfg.get('num_classes', 2),
            num_leads=mod_cfg['num_channels'],
            aggregation=classifier_cfg.get('aggregation', 'attention'),
            hidden_dim=classifier_cfg.get('hidden_dim', 128),
            num_heads=classifier_cfg.get('num_heads', 4),
            num_layers=classifier_cfg.get('num_layers', 2),
            dropout=classifier_cfg.get('dropout', 0.1),
            freeze_tokenizer=classifier_cfg.get('freeze_tokenizers', True),
        )
    
    return model.to(device)


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    modality: str,
    augmentor: Optional[SignalAugmentor] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    if augmentor is not None:
        augmentor.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        labels = batch['label'].to(device)
        
        if modality == 'both':
            eeg = batch['eeg'].to(device)
            fnirs = batch['fnirs'].to(device)
            
            # Apply augmentation
            if augmentor is not None:
                eeg, aug_labels = augmentor(eeg, labels)
                fnirs, _ = augmentor(fnirs, labels)  # Same transform for consistency
                if aug_labels is not None and aug_labels.dim() == 2:
                    labels = aug_labels  # Use soft labels from mixup
            
            outputs = model(eeg, fnirs)
        else:
            x = batch['data'].to(device)
            
            # Apply augmentation
            if augmentor is not None:
                x, aug_labels = augmentor(x, labels)
                if aug_labels is not None and aug_labels.dim() == 2:
                    labels = aug_labels  # Use soft labels from mixup
            
            outputs = model(x)
        
        logits = outputs['logits']
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        
        # For soft labels, use argmax to get hard labels for accuracy
        if labels.dim() == 2:
            hard_labels = labels.argmax(dim=-1).cpu().numpy()
        else:
            hard_labels = labels.cpu().numpy()
        all_labels.extend(hard_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    modality: str,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_subjects = []
    
    for batch in dataloader:
        labels = batch['label'].to(device)
        
        if modality == 'both':
            eeg = batch['eeg'].to(device)
            fnirs = batch['fnirs'].to(device)
            outputs = model(eeg, fnirs)
        else:
            x = batch['data'].to(device)
            outputs = model(x)
        
        logits = outputs['logits']
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        if 'subject_id' in batch:
            subj = batch['subject_id']
            if isinstance(subj, torch.Tensor):
                subj = subj.cpu().numpy()
            all_subjects.extend(subj)
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # ROC-AUC (for binary classification)
    try:
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
    except:
        roc_auc = 0.0
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    result = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix.tolist(),
    }
    
    if return_predictions:
        result['y_true'] = y_true
        result['y_probs'] = y_probs
        result['subjects'] = np.array(all_subjects) if all_subjects else None
    
    return result


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    modality: str,
) -> Dict[str, np.ndarray]:
    """Collect split-level embeddings and labels for probe analysis."""
    model.eval()

    def _reduce_to_2d(tensor: torch.Tensor) -> torch.Tensor:
        """Reduce arbitrary feature shape [B, ...] to [B, D] by mean pooling non-batch dims."""
        if tensor.dim() == 2:
            return tensor
        reduce_dims = tuple(range(1, tensor.dim() - 1))
        if len(reduce_dims) == 0:
            return tensor
        return tensor.mean(dim=reduce_dims)

    def _extract_single_modality_embedding(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'embeddings' in outputs:
            return _reduce_to_2d(outputs['embeddings'])
        if 'lead_features' in outputs:
            return _reduce_to_2d(outputs['lead_features'])
        if 'token_embeddings' in outputs:
            return _reduce_to_2d(outputs['token_embeddings'])
        raise KeyError(f"No usable embedding key found in outputs: {list(outputs.keys())}")

    def _extract_dual_modality_embedding(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'eeg_embeddings' in outputs and 'fnirs_embeddings' in outputs:
            eeg_emb = _reduce_to_2d(outputs['eeg_embeddings'])
            fnirs_emb = _reduce_to_2d(outputs['fnirs_embeddings'])
            return torch.cat([eeg_emb, fnirs_emb], dim=-1)
        if 'eeg_features' in outputs and 'fnirs_features' in outputs:
            eeg_feat = _reduce_to_2d(outputs['eeg_features'])
            fnirs_feat = _reduce_to_2d(outputs['fnirs_features'])
            return torch.cat([eeg_feat, fnirs_feat], dim=-1)
        raise KeyError(f"No usable dual-modality embedding keys found in outputs: {list(outputs.keys())}")

    embeddings = []
    labels_all = []
    subjects_all = []

    for batch in dataloader:
        labels = batch['label'].to(device)

        if modality == 'both':
            eeg = batch['eeg'].to(device)
            fnirs = batch['fnirs'].to(device)
            outputs = model(eeg, fnirs)
            emb = _extract_dual_modality_embedding(outputs)
        else:
            x = batch['data'].to(device)
            try:
                outputs = model(x, return_lead_features=True)
            except TypeError:
                outputs = model(x)
            emb = _extract_single_modality_embedding(outputs)

        embeddings.append(emb.cpu().numpy())
        labels_all.append(labels.cpu().numpy())

        if 'subject_id' in batch:
            subj = batch['subject_id']
            if isinstance(subj, torch.Tensor):
                subj = subj.cpu().numpy()
            subjects_all.append(np.array(subj))

    result = {
        'embedding': np.concatenate(embeddings, axis=0),
        'label': np.concatenate(labels_all, axis=0),
    }
    if subjects_all:
        result['subject_id'] = np.concatenate(subjects_all, axis=0)
    return result


def export_probe_data(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    run_dir: Path,
    device: torch.device,
    modality: str,
) -> Dict[str, str]:
    """Export train/val/test embeddings for factor probe scripts."""
    probe_dir = run_dir / 'probes'
    probe_dir.mkdir(exist_ok=True)

    exported = {}
    for split in ['train', 'val', 'test']:
        payload = collect_embeddings(model, dataloaders[split], device, modality)
        out_path = probe_dir / f'{split}_embeddings.npz'
        np.savez_compressed(out_path, **payload)
        exported[split] = str(out_path)

    return exported


def train(
    config: dict,
    run_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Main training function."""
    modality = config.get('modality', 'eeg')
    training_cfg = config['training']
    
    print(f"\n{'='*60}")
    print(f"Training Downstream Classifier")
    print(f"{'='*60}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Modality: {modality}")
    print(f"Device: {device}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(config)
    print(f"\nDataset sizes:")
    print(f"  Train: {len(dataloaders['train'].dataset)} samples")
    print(f"  Val: {len(dataloaders['val'].dataset)} samples")
    print(f"  Test: {len(dataloaders['test'].dataset)} samples")
    
    # Create augmentor if configured
    augmentor = create_augmentor_from_config(config)
    if augmentor is not None:
        augmentor = augmentor.to(device)
        print(f"\nData augmentation enabled:")
        if augmentor.time_shift_max > 0:
            print(f"  - Time shift: ±{augmentor.time_shift_max} samples")
        if augmentor.channel_dropout_prob > 0:
            print(f"  - Channel dropout: {augmentor.channel_dropout_prob*100:.1f}%")
        if augmentor.gaussian_noise_std > 0:
            print(f"  - Gaussian noise: std={augmentor.gaussian_noise_std}")
        if augmentor.scaling_range != (1.0, 1.0):
            print(f"  - Amplitude scaling: {augmentor.scaling_range}")
        if augmentor.mixup_alpha > 0:
            print(f"  - Mixup: alpha={augmentor.mixup_alpha}")
    
    # Create model
    model = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,} total, {n_trainable:,} trainable")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay'],
    )
    
    # Learning rate scheduler with warmup
    epochs = training_cfg['epochs']
    warmup_epochs = training_cfg.get('warmup_epochs', 5)
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    # Loss function with optional label smoothing
    label_smoothing = training_cfg.get('label_smoothing', 0.0)
    if label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        print(f"\nLabel smoothing: {label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience = training_cfg.get('early_stopping', {}).get('patience', 20)
    history = {'train': [], 'val': []}
    
    start_time = time.time()
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, criterion, device, modality, augmentor
        )
        val_metrics = evaluate(
            model, dataloaders['val'], criterion, device, modality
        )
        
        scheduler.step()
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config,
            }, run_dir / 'checkpoints' / 'best_model.pt')
            
            best_marker = " ★"
        else:
            patience_counter += 1
            best_marker = ""
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']*100:.1f}% | "
                  f"Val: acc={val_metrics['accuracy']*100:.1f}%, f1={val_metrics['f1']*100:.1f}% | "
                  f"{epoch_time:.1f}s{best_marker}")
        
        # Early stopping
        if training_cfg.get('early_stopping', {}).get('enabled', True):
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(run_dir / 'checkpoints' / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(
        model, dataloaders['test'], criterion, device, modality, return_predictions=True
    )
    
    y_true = test_metrics.pop('y_true', None)
    y_probs = test_metrics.pop('y_probs', None)
    subjects = test_metrics.pop('subjects', None)
    
    # Per-subject accuracy
    subject_accuracies = None
    if subjects is not None and len(subjects) > 0:
        subject_accuracies = {}
        for subj in np.unique(subjects):
            mask = subjects == subj
            if mask.sum() > 0:
                subj_preds = np.argmax(y_probs[mask], axis=1)
                subj_acc = np.mean(subj_preds == y_true[mask])
                subject_accuracies[int(subj)] = float(subj_acc)
    
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall: {test_metrics['recall']*100:.2f}%")
    print(f"  F1 Score: {test_metrics['f1']*100:.2f}%")
    print(f"  Cohen's Kappa: {test_metrics['kappa']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix: {test_metrics['confusion_matrix']}")
    print(f"{'='*60}")

    probe_data_paths = None
    if config.get('logging', {}).get('export_probe_data', True):
        print("\nExporting probe data (train/val/test embeddings)...")
        probe_data_paths = export_probe_data(
            model=model,
            dataloaders=dataloaders,
            run_dir=run_dir,
            device=device,
            modality=modality,
        )
        print(f"Probe data exported to: {run_dir / 'probes'}")
    
    # Save results
    results = {
        'experiment': config['experiment']['name'],
        'modality': modality,
        'n_params': n_params,
        'n_trainable': n_trainable,
        'training_time_s': training_time,
        'epochs_trained': epoch + 1,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
        'subject_accuracies': subject_accuracies,
        'history': history,
        'probe_data_paths': probe_data_paths,
    }
    
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    metrics_history = []
    for i, (train_m, val_m) in enumerate(zip(history['train'], history['val'])):
        metrics_history.append({
            'epoch': i + 1,
            'loss': train_m['loss'],
            'accuracy': train_m['accuracy'],
            'val_loss': val_m['loss'],
            'val_accuracy': val_m['accuracy'],
        })
    
    final_metrics = {
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_confusion_matrix': test_metrics['confusion_matrix'],
        'best_val_accuracy': best_val_acc,
    }
    
    visualize_classifier_run(
        run_dir=run_dir,
        metrics_history=metrics_history,
        final_metrics=final_metrics,
        config=config,
        y_true=y_true,
        y_probs=y_probs,
        subject_accuracies=subject_accuracies,
        class_names=['Left MI', 'Right MI'],
    )
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Downstream Task Training")
    parser.add_argument('--config', type=str, required=True,
                        help='Config file path (relative to experiments/configs/)')
    parser.add_argument('--foreground', '-f', action='store_true',
                        help='Run in foreground (default is background)')
    args = parser.parse_args()
    
    # Background mode handling
    if not args.foreground and not os.environ.get('DOWNSTREAM_TRAINING_BG'):
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        config_name = Path(args.config).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{config_name}_{timestamp}.log'
        
        cmd = [sys.executable, __file__, '--config', args.config, '--foreground']
        
        env = os.environ.copy()
        env['DOWNSTREAM_TRAINING_BG'] = '1'
        
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        
        print(f"Training started in background (PID: {process.pid})")
        print(f"Log file: {log_file}")
        print(f"Monitor: tail -f {log_file}")
        sys.exit(0)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config['experiment']['name']
    run_dir = project_root / 'experiments' / 'runs' / f'{exp_name}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'figures').mkdir(exist_ok=True)
    
    # Setup logging
    tee_logger = setup_logging(run_dir)
    
    # Save config
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Setup device
        device = setup_device(config)
        
        # Set random seed
        seed = config['experiment'].get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Train
        results = train(config, run_dir, device)
        
        print(f"\nResults saved to: {run_dir}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        tee_logger.close()
        sys.stdout = tee_logger.terminal
        sys.stderr = tee_logger.terminal


if __name__ == '__main__':
    main()
