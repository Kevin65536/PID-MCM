#!/usr/bin/env python
"""
Train classifier on tokenized representations or raw signals.

Supports:
1. Single modality: EEG or fNIRS tokenized/raw → classification
2. Multi-modality: EEG + fNIRS fusion → classification

Usage:
    python train_classifier.py --config phase1a/P1A_eeg_classification.yaml
    python train_classifier.py --config phase1a/P1A_fusion.yaml --epochs 50
"""

# Fix OMP duplicate library error (must be before numpy/torch imports)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.eeg_fnirs_dataset import (
    EEGfNIRSDataset, 
    MultiModalEEGfNIRSDataset, 
    create_dataloaders
)
from src.classifiers import SimpleTokenClassifier, EndToEndClassifier
from src.classifiers.end_to_end import MultiModalClassifier
from src.classifiers.simple_classifier import RawSignalClassifier
from src.tokenizers import FSQTokenizer, VQVAETokenizer


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration with base config inheritance."""
    configs_dir = project_root / "experiments" / "configs"
    
    # Load base config
    base_path = configs_dir / "base.yaml"
    with open(base_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load experiment-specific config and merge
    exp_path = configs_dir / config_path
    if exp_path.exists():
        with open(exp_path, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f)
        config = deep_merge(config, exp_config)
    
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_config_hash(config: dict) -> str:
    """Generate a short hash of config for identification."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


# =============================================================================
# Model Creation
# =============================================================================

def load_tokenizer(config: dict, device: torch.device) -> nn.Module:
    """Load pre-trained tokenizer from checkpoint."""
    tokenizer_config = config.get('tokenizer', {})
    checkpoint_path = tokenizer_config.get('checkpoint')
    
    if checkpoint_path is None:
        raise ValueError("tokenizer.checkpoint must be specified in config")
    
    # Resolve path
    checkpoint_path = project_root / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {checkpoint_path}")
    
    print(f"[Tokenizer] Loading from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine tokenizer type from checkpoint or config
    tokenizer_type = tokenizer_config.get('type', 'vqvae')
    
    # Get model parameters from config
    model_config = tokenizer_config.get('model', {})
    
    if tokenizer_type == 'vqvae':
        tokenizer = VQVAETokenizer(
            seq_length=model_config.get('seq_length', 512),
            input_channels=model_config.get('input_channels', 1),
            codebook_size=model_config.get('codebook_size', 512),
            embedding_dim=model_config.get('embedding_dim', 64),
            encoder_dims=model_config.get('encoder_dims', [64, 128, 256]),
            encoder_kernel=model_config.get('encoder_kernel', 7),
            encoder_stride=model_config.get('encoder_stride', 2),
        )
    elif tokenizer_type == 'fsq':
        tokenizer = FSQTokenizer(
            seq_length=model_config.get('seq_length', 512),
            input_channels=model_config.get('input_channels', 1),
            levels=model_config.get('levels', [8, 8, 8, 8]),
            encoder_dims=model_config.get('encoder_dims', [64, 128, 256]),
            encoder_kernel=model_config.get('encoder_kernel', 7),
            encoder_stride=model_config.get('encoder_stride', 2),
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
    else:
        tokenizer.load_state_dict(checkpoint)
    
    tokenizer.to(device)
    tokenizer.eval()
    
    print(f"[Tokenizer] Loaded {tokenizer_type} with codebook size {tokenizer.get_codebook_size()}")
    
    return tokenizer


def create_classifier(config: dict, tokenizer: Optional[nn.Module], device: torch.device) -> nn.Module:
    """Create classifier based on config."""
    classifier_config = config.get('classifier', {})
    classifier_type = classifier_config.get('type', 'end_to_end')
    
    if classifier_type == 'end_to_end':
        if tokenizer is None:
            raise ValueError("Tokenizer required for end_to_end classifier")
        
        classifier = EndToEndClassifier(
            tokenizer=tokenizer,
            num_classes=classifier_config.get('num_classes', 2),
            pool_type=classifier_config.get('pool_type', 'mean'),
            hidden_dims=classifier_config.get('hidden_dims'),
            dropout=classifier_config.get('dropout', 0.1),
            freeze_tokenizer=classifier_config.get('freeze_tokenizer', True),
            use_pre_quantized=classifier_config.get('use_pre_quantized', False),
        )
        
    elif classifier_type == 'raw':
        classifier = RawSignalClassifier(
            seq_length=config['data']['window']['length'],
            input_channels=config['data'].get('input_channels', 1),
            num_classes=classifier_config.get('num_classes', 2),
            encoder_dims=classifier_config.get('encoder_dims', [64, 128, 256]),
            latent_dim=classifier_config.get('latent_dim', 64),
            pool_type=classifier_config.get('pool_type', 'mean'),
            hidden_dims=classifier_config.get('hidden_dims'),
            dropout=classifier_config.get('dropout', 0.1),
        )
        
    elif classifier_type == 'simple_token':
        if tokenizer is None:
            raise ValueError("Tokenizer required for simple_token classifier")
        
        classifier = SimpleTokenClassifier(
            tokenizer=tokenizer,
            embedding_dim=tokenizer.latent_dim,
            num_classes=classifier_config.get('num_classes', 2),
            pool_type=classifier_config.get('pool_type', 'mean'),
            hidden_dims=classifier_config.get('hidden_dims'),
            dropout=classifier_config.get('dropout', 0.1),
            freeze_tokenizer=classifier_config.get('freeze_tokenizer', True),
            input_mode='raw',
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    return classifier.to(device)


def create_multimodal_classifier(
    config: dict, 
    eeg_tokenizer: nn.Module, 
    fnirs_tokenizer: nn.Module, 
    device: torch.device
) -> nn.Module:
    """Create multi-modal classifier."""
    classifier_config = config.get('classifier', {})
    
    classifier = MultiModalClassifier(
        eeg_tokenizer=eeg_tokenizer,
        fnirs_tokenizer=fnirs_tokenizer,
        num_classes=classifier_config.get('num_classes', 2),
        fusion_type=classifier_config.get('fusion_type', 'early'),
        pool_type=classifier_config.get('pool_type', 'mean'),
        hidden_dims=classifier_config.get('hidden_dims'),
        projection_dim=classifier_config.get('projection_dim'),
        dropout=classifier_config.get('dropout', 0.1),
        freeze_tokenizers=classifier_config.get('freeze_tokenizer', True),
    )
    
    return classifier.to(device)


# =============================================================================
# Training Utilities
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_multimodal: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        labels = batch['label'].to(device)
        
        if is_multimodal:
            eeg = batch['eeg'].to(device)
            fnirs = batch['fnirs'].to(device)
            # Handle channel dimension - if shape is [B, C, T], average over C
            if eeg.dim() == 3:
                eeg = eeg.mean(dim=1)  # [B, T]
            if fnirs.dim() == 3:
                fnirs = fnirs.mean(dim=1)  # [B, T]
            outputs = model(eeg, fnirs)
        else:
            x = batch['data'].to(device)
            # Handle channel dimension
            if x.dim() == 3:
                x = x.mean(dim=1)  # [B, T]
            outputs = model(x)
        
        logits = outputs['logits']
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_multimodal: bool = False,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)
            
            if is_multimodal:
                eeg = batch['eeg'].to(device)
                fnirs = batch['fnirs'].to(device)
                if eeg.dim() == 3:
                    eeg = eeg.mean(dim=1)
                if fnirs.dim() == 3:
                    fnirs = fnirs.mean(dim=1)
                outputs = model(eeg, fnirs)
            else:
                x = batch['data'].to(device)
                if x.dim() == 3:
                    x = x.mean(dim=1)
                outputs = model(x)
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
    }


# =============================================================================
# Experiment Logger
# =============================================================================

class ClassifierLogger:
    """Logger for classifier training experiments."""
    
    def __init__(self, config: dict, config_path: str):
        self.config = config
        self.config_path = config_path
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config['experiment']['name']
        self.run_name = f"{exp_name}_{timestamp}"
        
        self.runs_dir = project_root / "experiments" / "runs"
        self.run_dir = self.runs_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        
        # Save config
        with open(self.run_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Initialize metrics storage
        self.metrics = {
            'config_hash': get_config_hash(config),
            'started_at': datetime.now().isoformat(),
            'epochs': [],
        }
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for an epoch."""
        self.metrics['epochs'].append({
            'epoch': epoch,
            **metrics
        })
        
        # Save metrics incrementally
        with open(self.run_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_checkpoint(self, state: dict, name: str = "checkpoint"):
        """Save model checkpoint."""
        path = self.run_dir / "checkpoints" / f"{name}.pt"
        torch.save(state, path)
    
    def save_final(self, final_metrics: dict):
        """Save final metrics."""
        self.metrics['completed_at'] = datetime.now().isoformat()
        self.metrics['final_metrics'] = final_metrics
        
        with open(self.run_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"[Logger] Results saved to: {self.run_dir}")


# =============================================================================
# Main Training Function
# =============================================================================

def train(config_path: str, epochs: Optional[int] = None):
    """Run classifier training."""
    
    # Load config
    config = load_config(config_path)
    
    # Override epochs if specified
    if epochs is not None:
        config['training']['epochs'] = epochs
    
    # Setup
    device = torch.device(config['experiment'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"[Config] Loaded: {config_path}")
    print(f"[Device] Using: {device}")
    print(f"[Seed] {seed}")
    
    # Determine if multimodal
    is_multimodal = config['data'].get('modality') == 'both'
    
    # Create dataloaders
    print("[Data] Creating dataloaders...")
    data_config = config['data']
    data_root = project_root / data_config.get('root', 'data/EEG+NIRS Single-Trial')
    
    # Subject splits
    train_subjects = data_config.get('train_subjects', list(range(1, 21)))
    val_subjects = data_config.get('val_subjects', list(range(21, 26)))
    test_subjects = data_config.get('test_subjects', list(range(26, 30)))
    
    dataloaders = create_dataloaders(
        data_root=str(data_root),
        modality=data_config.get('modality', 'eeg'),
        task=data_config.get('task', 'motor_imagery'),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        window_samples=data_config['window']['length'],
        window_duration_s=data_config.get('window', {}).get('duration_s', 2.5),
        batch_size=config['training']['batch_size'],
        num_workers=data_config.get('num_workers', 0),
        normalize=data_config.get('normalize', True),
        # Channel filtering options
        exclude_eog=data_config.get('exclude_eog', True),
        hbo_only=data_config.get('hbo_only', True),
        hbr_only=data_config.get('hbr_only', False),
    )
    
    print(f"[Data] Train: {len(dataloaders['train'].dataset)}, "
          f"Val: {len(dataloaders['val'].dataset)}, "
          f"Test: {len(dataloaders['test'].dataset)}")
    
    # Create model
    print("[Model] Creating classifier...")
    
    if is_multimodal:
        # Load both tokenizers
        eeg_tokenizer_config = config.get('eeg_tokenizer', config.get('tokenizer', {}))
        fnirs_tokenizer_config = config.get('fnirs_tokenizer', config.get('tokenizer', {}))
        
        config['tokenizer'] = eeg_tokenizer_config
        eeg_tokenizer = load_tokenizer(config, device)
        
        config['tokenizer'] = fnirs_tokenizer_config
        fnirs_tokenizer = load_tokenizer(config, device)
        
        model = create_multimodal_classifier(config, eeg_tokenizer, fnirs_tokenizer, device)
    else:
        # Load single tokenizer if needed
        classifier_type = config.get('classifier', {}).get('type', 'end_to_end')
        
        if classifier_type in ['end_to_end', 'simple_token']:
            tokenizer = load_tokenizer(config, device)
        else:
            tokenizer = None
        
        model = create_classifier(config, tokenizer, device)
    
    print(f"[Model] Created {config.get('classifier', {}).get('type', 'end_to_end')} classifier")
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Trainable params: {trainable_params:,} / {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )
    
    # Scheduler
    num_epochs = config['training']['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Logger
    logger = ClassifierLogger(config, config_path)
    
    # Training loop
    print(f"[Training] Starting {num_epochs} epochs...")
    best_val_acc = 0.0
    patience_counter = 0
    patience = config['training'].get('early_stopping', {}).get('patience', 20)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, criterion, device, is_multimodal
        )
        
        # Validate
        val_metrics = evaluate(
            model, dataloaders['val'], criterion, device, is_multimodal
        )
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        
        # Step scheduler
        scheduler.step()
        
        # Combine metrics
        epoch_metrics = {
            **train_metrics,
            **val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        # Remove confusion matrix for logging (too verbose)
        if 'val_confusion_matrix' in epoch_metrics:
            del epoch_metrics['val_confusion_matrix']
        
        # Log
        logger.log_epoch(epoch, epoch_metrics)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{num_epochs}] "
                  f"loss={train_metrics['loss']:.4f}, "
                  f"acc={train_metrics['accuracy']:.3f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"val_acc={val_metrics['val_accuracy']:.3f}")
        
        # Save best model
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
            patience_counter = 0
            logger.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
            }, name="best_model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stop] No improvement for {patience} epochs.")
                break
    
    # Load best model for final evaluation
    best_checkpoint = torch.load(logger.run_dir / "checkpoints" / "best_model.pt", weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final evaluation
    print("[Evaluation] Computing final metrics on test set...")
    
    test_metrics = evaluate(
        model, dataloaders['test'], criterion, device, is_multimodal
    )
    
    final_metrics = {
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_confusion_matrix': test_metrics['confusion_matrix'],
        'best_val_accuracy': best_val_acc,
    }
    
    logger.save_final(final_metrics)
    
    print(f"\n[Done] Experiment: {logger.run_name}")
    print(f"[Results] Final test metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    cm = np.array(test_metrics['confusion_matrix'])
    print(f"    {cm}")
    
    return logger.run_name, final_metrics


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier on tokenized neural signals")
    parser.add_argument("--config", type=str, required=True,
                        help="Config file path relative to experiments/configs/")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    
    args = parser.parse_args()
    train(args.config, args.epochs)
