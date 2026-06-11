#!/usr/bin/env python
"""Test script to verify classifier pipeline works."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tokenizers import VQVAETokenizer, FSQTokenizer
from src.classifiers_legacy import EndToEndClassifier, SimpleTokenClassifier
from src.classifiers_legacy.simple_classifier import RawSignalClassifier

def test_tokenizer_loading():
    """Test loading pre-trained tokenizers."""
    print("=" * 60)
    print("Test 1: Loading pre-trained tokenizers")
    print("=" * 60)
    
    # Test VQ-VAE EEG tokenizer
    checkpoint_path = project_root / "experiments/runs/comparison_20260114_183311/VQVAE_EEG/checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        tokenizer = VQVAETokenizer(
            seq_length=512,
            input_channels=1,
            codebook_size=512,
            embedding_dim=64,
            encoder_dims=[32, 64, 128],
            encoder_kernel=7,
            encoder_stride=2,
        )
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ VQ-VAE EEG tokenizer loaded (codebook: {tokenizer.get_codebook_size()})")
        
        # Test forward pass
        x = torch.randn(4, 512)
        outputs = tokenizer(x)
        print(f"  Input: {x.shape} -> Indices: {outputs['indices'].shape}, z_q: {outputs['z_q'].shape}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
    
    # Test FSQ fNIRS tokenizer
    checkpoint_path = project_root / "experiments/runs/comparison_20260114_183311/FSQ_fNIRS/checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        tokenizer = FSQTokenizer(
            seq_length=25,
            input_channels=1,
            levels=[8, 8, 8],
            encoder_dims=[32, 64],
            encoder_kernel=5,
            encoder_stride=2,
        )
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ FSQ fNIRS tokenizer loaded (codebook: {tokenizer.get_codebook_size()})")
        
        # Test forward pass
        x = torch.randn(4, 25)
        outputs = tokenizer(x)
        print(f"  Input: {x.shape} -> Indices: {outputs['indices'].shape}, z_q: {outputs['z_q'].shape}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")


def test_classifier_creation():
    """Test creating different classifier types."""
    print("\n" + "=" * 60)
    print("Test 2: Creating classifiers")
    print("=" * 60)
    
    # Load tokenizer
    checkpoint_path = project_root / "experiments/runs/comparison_20260114_183311/VQVAE_EEG/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    tokenizer = VQVAETokenizer(
        seq_length=512,
        input_channels=1,
        codebook_size=512,
        embedding_dim=64,
        encoder_dims=[32, 64, 128],
        encoder_kernel=7,
        encoder_stride=2,
    )
    tokenizer.load_state_dict(checkpoint['model_state_dict'])
    
    # Test EndToEndClassifier
    classifier = EndToEndClassifier(
        tokenizer=tokenizer,
        num_classes=2,
        pool_type='mean',
        hidden_dims=[128],
        freeze_tokenizer=True,
    )
    
    x = torch.randn(4, 512)
    outputs = classifier(x)
    print(f"✓ EndToEndClassifier: input {x.shape} -> logits {outputs['logits'].shape}")
    
    # Test RawSignalClassifier
    raw_classifier = RawSignalClassifier(
        seq_length=512,
        input_channels=1,
        num_classes=2,
        encoder_dims=[32, 64, 128],
        latent_dim=64,
        pool_type='mean',
        hidden_dims=[128],
    )
    
    outputs = raw_classifier(x)
    print(f"✓ RawSignalClassifier: input {x.shape} -> logits {outputs['logits'].shape}")
    
    # Test SimpleTokenClassifier with embeddings
    simple_classifier = SimpleTokenClassifier(
        embedding_dim=64,
        num_classes=2,
        pool_type='attention',
        hidden_dims=[128],
        input_mode='embeddings',
    )
    
    embeddings = torch.randn(4, 8, 64)  # [B, T', D]
    outputs = simple_classifier(embeddings)
    print(f"✓ SimpleTokenClassifier: embeddings {embeddings.shape} -> logits {outputs['logits'].shape}")


def test_data_loading():
    """Test loading real data."""
    print("\n" + "=" * 60)
    print("Test 3: Loading real EEG/fNIRS data")
    print("=" * 60)
    
    from src.data.eeg_fnirs_dataset import EEGfNIRSDataset, create_dataloaders
    
    data_root = project_root / "data" / "EEG+NIRS Single-Trial"
    
    if not data_root.exists():
        print(f"✗ Data not found: {data_root}")
        return
    
    # Test EEG dataset
    try:
        dataset = EEGfNIRSDataset(
            data_root=str(data_root),
            subject_ids=[1, 2],
            task='motor_imagery',
            modality='eeg',
            window_samples=512,
        )
        sample = dataset[0]
        print(f"✓ EEG dataset: {len(dataset)} samples, shape {sample['data'].shape}, label {sample['label']}")
    except Exception as e:
        print(f"✗ EEG dataset failed: {e}")
    
    # Test fNIRS dataset
    try:
        dataset = EEGfNIRSDataset(
            data_root=str(data_root),
            subject_ids=[1, 2],
            task='motor_imagery',
            modality='fnirs',
            window_samples=25,
        )
        sample = dataset[0]
        print(f"✓ fNIRS dataset: {len(dataset)} samples, shape {sample['data'].shape}, label {sample['label']}")
    except Exception as e:
        print(f"✗ fNIRS dataset failed: {e}")


def test_end_to_end_forward():
    """Test complete forward pass with real data."""
    print("\n" + "=" * 60)
    print("Test 4: End-to-end forward pass with real data")
    print("=" * 60)
    
    from src.data.eeg_fnirs_dataset import EEGfNIRSDataset
    from torch.utils.data import DataLoader
    
    data_root = project_root / "data" / "EEG+NIRS Single-Trial"
    
    if not data_root.exists():
        print(f"✗ Data not found: {data_root}")
        return
    
    # Load tokenizer
    checkpoint_path = project_root / "experiments/runs/comparison_20260114_183311/VQVAE_EEG/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    tokenizer = VQVAETokenizer(
        seq_length=512,
        input_channels=1,
        codebook_size=512,
        embedding_dim=64,
        encoder_dims=[32, 64, 128],
        encoder_kernel=7,
        encoder_stride=2,
    )
    tokenizer.load_state_dict(checkpoint['model_state_dict'])
    
    # Create classifier
    classifier = EndToEndClassifier(
        tokenizer=tokenizer,
        num_classes=2,
        pool_type='mean',
        hidden_dims=[128],
        freeze_tokenizer=True,
    )
    
    # Load data
    dataset = EEGfNIRSDataset(
        data_root=str(data_root),
        subject_ids=[1],
        task='motor_imagery',
        modality='eeg',
        window_samples=512,
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    
    # Forward pass
    x = batch['data']  # [B, n_channels, seq_length]
    labels = batch['label']
    
    # Average channels to single channel
    if x.dim() == 3:
        x = x.mean(dim=1)  # [B, seq_length]
    
    outputs = classifier(x)
    
    print(f"✓ Forward pass successful:")
    print(f"  Input shape: {x.shape}")
    print(f"  Labels: {labels.tolist()}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Predictions: {outputs['logits'].argmax(dim=-1).tolist()}")
    
    # Test loss computation
    import torch.nn.functional as F
    loss = F.cross_entropy(outputs['logits'], labels)
    print(f"  Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_tokenizer_loading()
    test_classifier_creation()
    test_data_loading()
    test_end_to_end_forward()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
