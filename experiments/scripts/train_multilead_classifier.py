#!/usr/bin/env python
"""
Train multi-lead classifier with time-aligned tokenizers.

This script trains classifiers that use pre-trained tokenizers to process
multi-lead EEG/fNIRS signals for Motor Imagery classification.

Usage:
    python train_multilead_classifier.py --modality eeg --aggregation attention
    python train_multilead_classifier.py --modality fnirs --aggregation mean
    python train_multilead_classifier.py --modality both --fusion early
"""

# Fix OMP duplicate library error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.eeg_fnirs_dataset import (
    EEGfNIRSDataset, 
    MultiModalEEGfNIRSDataset,
    create_dataloaders
)
from src.tokenizers.vqvae import VQVAETokenizer
from src.tokenizers.fsq import FSQTokenizer
from src.classifiers.multi_lead import MultiLeadClassifier, DualModalityMultiLeadClassifier
from src.visualization.classifier_plots import ClassifierVisualizer, visualize_classifier_run


# =============================================================================
# Configuration
# =============================================================================

TOKENIZER_PATHS = {
    'eeg': 'experiments/runs/LaBraM_tokenizers_20260115_232415/VQVAE_EEG_LaBraM/best_model.pt',
    'fnirs': 'experiments/runs/LaBraM_tokenizers_20260115_232415/VQVAE_fNIRS_Aligned/best_model.pt',
}

# EEG config (4s MI window @ 200Hz)
EEG_CONFIG = {
    'seq_length': 800,  # 4s @ 200Hz (MI standard)
    'num_leads': 28,    # EEG channels after excluding EOG (30 - 2 = 28)
    'sample_rate': 200,
}

# fNIRS config (4s MI window @ 10Hz)
FNIRS_CONFIG = {
    'seq_length': 40,  # 4s @ 10Hz
    'num_leads': 36,   # HbO channels only (36 HbO out of 72 total)
    'sample_rate': 10,
}


def load_tokenizer(modality: str, device: str) -> nn.Module:
    """Load pre-trained tokenizer."""
    path = project_root / TOKENIZER_PATHS[modality]
    
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint['config']
    
    # Check model type from config
    model_type = config.get('model_type', 'vqvae')
    
    if model_type == 'vqvae' or 'codebook_size' in config:
        # VQ-VAE tokenizer
        tokenizer = VQVAETokenizer(
            seq_length=config['seq_length'],
            input_channels=config['input_channels'],
            codebook_size=config['codebook_size'],
            embedding_dim=config['embedding_dim'],
            encoder_dims=list(config['encoder_dims']),
            encoder_kernel=config['kernel_size'],
            encoder_stride=config['stride'],
        )
    else:
        # FSQ tokenizer
        tokenizer = FSQTokenizer(
            seq_length=config['seq_length'],
            input_channels=config['input_channels'],
            levels=list(config['levels']),
            encoder_dims=list(config['encoder_dims']),
            encoder_kernel=config['kernel_size'],
            encoder_stride=config['stride'],
        )
    
    tokenizer.load_state_dict(checkpoint['model_state_dict'])
    tokenizer.to(device)
    tokenizer.eval()
    
    print(f"Loaded {modality} tokenizer from {path}")
    print(f"  Config: seq_length={config['seq_length']}, latent_dim={tokenizer.latent_dim}")
    
    return tokenizer


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
        window_offset_ms: float = 500,  # MI response delay: 0.5s after cue
        exclude_eog: bool = True,  # Exclude EOG channels from EEG
        hbo_only: bool = True,     # Use only HbO channels for fNIRS
    ):
        self.base_dataset = EEGfNIRSDataset(
            data_root=data_root,
            subject_ids=subject_ids,
            task=task,
            modality=modality,
            window_samples=window_samples,
            normalize=normalize,
            window_offset_ms=window_offset_ms,
            exclude_eog=exclude_eog,
            hbo_only=hbo_only,
        )
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        # sample['data'] is already [C, T]
        return sample


class DualModalityMultiLeadDataset(torch.utils.data.Dataset):
    """Dataset that returns multi-lead data for both modalities."""
    
    def __init__(
        self,
        data_root: str,
        subject_ids: list,
        eeg_window_samples: int,
        fnirs_window_samples: int,
        task: str = 'motor_imagery',
        normalize: bool = True,
        window_offset_ms: float = 500,  # MI response delay
        exclude_eog: bool = True,  # Exclude EOG channels from EEG
        hbo_only: bool = True,     # Use only HbO channels for fNIRS
    ):
        self.eeg_dataset = EEGfNIRSDataset(
            data_root=data_root,
            subject_ids=subject_ids,
            task=task,
            modality='eeg',
            window_samples=eeg_window_samples,
            normalize=normalize,
            window_offset_ms=window_offset_ms,
            exclude_eog=exclude_eog,
        )
        self.fnirs_dataset = EEGfNIRSDataset(
            data_root=data_root,
            subject_ids=subject_ids,
            task=task,
            modality='fnirs',
            window_samples=fnirs_window_samples,
            normalize=normalize,
            window_offset_ms=window_offset_ms,
            hbo_only=hbo_only,
        )
        
        # Align by (subject, session, trial)
        self._build_alignment()
    
    def _build_alignment(self):
        """Build index alignment between EEG and fNIRS."""
        # Create lookup
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
            'eeg': eeg_sample['data'],  # [C_eeg, T_eeg]
            'fnirs': fnirs_sample['data'],  # [C_fnirs, T_fnirs]
            'label': eeg_sample['label'],
            'subject_id': eeg_sample['subject_id'],
        }


def create_multilead_dataloaders(
    data_root: str,
    modality: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create dataloaders for multi-lead classification."""
    
    train_subjects = list(range(1, 21))
    val_subjects = list(range(21, 26))
    test_subjects = list(range(26, 30))
    
    dataloaders = {}
    
    for split, subjects in [('train', train_subjects), ('val', val_subjects), ('test', test_subjects)]:
        if modality == 'both':
            dataset = DualModalityMultiLeadDataset(
                data_root=data_root,
                subject_ids=subjects,
                eeg_window_samples=EEG_CONFIG['seq_length'],
                fnirs_window_samples=FNIRS_CONFIG['seq_length'],
            )
        else:
            config = EEG_CONFIG if modality == 'eeg' else FNIRS_CONFIG
            dataset = MultiLeadDataset(
                data_root=data_root,
                subject_ids=subjects,
                modality=modality,
                window_samples=config['seq_length'],
            )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return dataloaders


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    modality: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        if modality == 'both':
            eeg = batch['eeg'].to(device)
            fnirs = batch['fnirs'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(eeg, fnirs)
        else:
            x = batch['data'].to(device)
            labels = batch['label'].to(device)
            
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
        all_labels.extend(labels.cpu().numpy())
    
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
    device: str,
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
        if modality == 'both':
            eeg = batch['eeg'].to(device)
            fnirs = batch['fnirs'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(eeg, fnirs)
        else:
            x = batch['data'].to(device)
            labels = batch['label'].to(device)
            
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
            all_subjects.extend(batch['subject_id'].cpu().numpy() if isinstance(batch['subject_id'], torch.Tensor) else batch['subject_id'])
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    result = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),
    }
    
    if return_predictions:
        result['y_true'] = np.array(all_labels)
        result['y_probs'] = np.array(all_probs)
        result['subjects'] = np.array(all_subjects) if all_subjects else None
    
    return result


def train_classifier(
    modality: str,
    aggregation: str,
    fusion: str,
    data_root: str,
    output_dir: Path,
    device: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
) -> Dict[str, Any]:
    """Train a multi-lead classifier."""
    
    print(f"\n{'='*60}")
    print(f"Training Multi-Lead Classifier")
    print(f"{'='*60}")
    print(f"Modality: {modality}")
    print(f"Aggregation: {aggregation}")
    if modality == 'both':
        print(f"Fusion: {fusion}")
    
    # Create dataloaders
    dataloaders = create_multilead_dataloaders(
        data_root=data_root,
        modality=modality,
        batch_size=batch_size,
    )
    
    print(f"Train: {len(dataloaders['train'].dataset)} samples")
    print(f"Val: {len(dataloaders['val'].dataset)} samples")
    print(f"Test: {len(dataloaders['test'].dataset)} samples")
    
    # Load tokenizers and create classifier
    if modality == 'both':
        eeg_tokenizer = load_tokenizer('eeg', device)
        fnirs_tokenizer = load_tokenizer('fnirs', device)
        
        model = DualModalityMultiLeadClassifier(
            eeg_tokenizer=eeg_tokenizer,
            fnirs_tokenizer=fnirs_tokenizer,
            num_classes=2,
            eeg_num_leads=EEG_CONFIG['num_leads'],
            fnirs_num_leads=FNIRS_CONFIG['num_leads'],
            aggregation=aggregation,
            fusion=fusion,
            hidden_dim=hidden_dim,
            freeze_tokenizers=True,
        ).to(device)
    else:
        tokenizer = load_tokenizer(modality, device)
        config = EEG_CONFIG if modality == 'eeg' else FNIRS_CONFIG
        
        model = MultiLeadClassifier(
            tokenizer=tokenizer,
            num_classes=2,
            num_leads=config['num_leads'],
            aggregation=aggregation,
            hidden_dim=hidden_dim,
            freeze_tokenizer=True,
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, criterion, device, modality
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
                'best_val_acc': best_val_acc,
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: train_acc={train_metrics['accuracy']*100:.1f}%, "
                  f"val_acc={val_metrics['accuracy']*100:.1f}%")
        
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    # Load best model and test
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, dataloaders['test'], criterion, device, modality, return_predictions=True)
    
    # Extract predictions for visualization
    y_true = test_metrics.pop('y_true', None)
    y_probs = test_metrics.pop('y_probs', None)
    subjects = test_metrics.pop('subjects', None)
    
    # Compute per-subject accuracy
    subject_accuracies = None
    if subjects is not None and len(subjects) > 0:
        subject_accuracies = {}
        for subj in np.unique(subjects):
            mask = subjects == subj
            if mask.sum() > 0:
                subj_preds = np.argmax(y_probs[mask], axis=1)
                subj_acc = np.mean(subj_preds == y_true[mask])
                subject_accuracies[int(subj)] = float(subj_acc)
    
    # Save results
    results = {
        'modality': modality,
        'aggregation': aggregation,
        'fusion': fusion if modality == 'both' else None,
        'n_trainable_params': n_params,
        'training_time_s': training_time,
        'epochs_trained': epoch + 1,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
        'history': history,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    print("\n[Visualization] Generating figures...")
    
    # Prepare metrics history for visualization
    metrics_history = []
    for i, (train_m, val_m) in enumerate(zip(history['train'], history['val'])):
        metrics_history.append({
            'epoch': i + 1,
            'loss': train_m['loss'],
            'accuracy': train_m['accuracy'],
            'val_loss': val_m['loss'],
            'val_accuracy': val_m['accuracy'],
        })
    
    # Create config dict for visualization
    viz_config = {
        'experiment': {'name': f'P1A_{modality}_{aggregation}'},
        'data': {'modality': modality},
        'classifier': {'type': f'multi_lead_{aggregation}'},
    }
    
    final_metrics = {
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_confusion_matrix': test_metrics['confusion_matrix'],
        'best_val_accuracy': best_val_acc,
    }
    
    visualize_classifier_run(
        run_dir=output_dir,
        metrics_history=metrics_history,
        final_metrics=final_metrics,
        config=viz_config,
        y_true=y_true,
        y_probs=y_probs,
        subject_accuracies=subject_accuracies,
        class_names=['Left MI', 'Right MI'],
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']*100:.1f}%")
    print(f"  Precision: {test_metrics['precision']*100:.1f}%")
    print(f"  Recall: {test_metrics['recall']*100:.1f}%")
    print(f"  F1: {test_metrics['f1']*100:.1f}%")
    print(f"  Confusion Matrix: {test_metrics['confusion_matrix']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train multi-lead classifier')
    parser.add_argument('--modality', type=str, default='eeg',
                        choices=['eeg', 'fnirs', 'both'])
    parser.add_argument('--aggregation', type=str, default='attention',
                        choices=['mean', 'attention', 'transformer'])
    parser.add_argument('--fusion', type=str, default='early',
                        choices=['early', 'late', 'cross_attention'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data-root', type=str,
                        default='data/EEG+NIRS Single-Trial')
    args = parser.parse_args()
    
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"P1A_{args.modality}_{args.aggregation}"
    if args.modality == 'both':
        exp_name += f"_{args.fusion}"
    output_dir = project_root / 'experiments' / 'runs' / f'{exp_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = train_classifier(
        modality=args.modality,
        aggregation=args.aggregation,
        fusion=args.fusion,
        data_root=args.data_root,
        output_dir=output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
    )
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
