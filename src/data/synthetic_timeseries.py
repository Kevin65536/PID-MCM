"""
Time-series synthetic dataset for PID-MCM experiments.
Generates frequency-separated PID components for validation.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional


class PIDTimeSeriesDataset(Dataset):
    """
    Synthetic time-series dataset with frequency-separated PID components.
    
    Design:
    - Redundancy (R): Low-frequency shared component (0.1-1 Hz)
    - Unique EEG (U1): High-frequency oscillations (8-30 Hz) 
    - Unique fNIRS (U2): Ultra-low frequency drift (<0.1 Hz)
    - Synergy (S): Amplitude modulation = R * U (cross-modal interaction)
    
    Args:
        n_samples: Number of samples to generate
        seq_length: Length of each time series
        fs: Sampling frequency (Hz)
        c_redundancy: Weight for redundancy component
        c_unique: Weight for unique components
        c_synergy: Weight for synergy component
        noise_level: Additive noise level
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_samples: int = 5000,
        seq_length: int = 256,
        fs: float = 200.0,
        c_redundancy: float = 1.0,
        c_unique: float = 1.0,
        c_synergy: float = 1.0,
        noise_level: float = 0.1,
        f_redundancy: Tuple[float, float] = (0.1, 1.0),
        f_unique_eeg: Tuple[float, float] = (8.0, 30.0),
        f_unique_fnirs: Tuple[float, float] = (0.01, 0.1),
        seed: int = 42
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.fs = fs
        self.c_r = c_redundancy
        self.c_u = c_unique
        self.c_s = c_synergy
        self.noise_level = noise_level
        self.f_redundancy = f_redundancy
        self.f_unique_eeg = f_unique_eeg
        self.f_unique_fnirs = f_unique_fnirs
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self._generate_data()
    
    def _generate_ar1(self, n_samples: int, length: int, alpha: float = 0.6) -> np.ndarray:
        """Generate AR(1) time series."""
        data = np.zeros((n_samples, length))
        for i in range(n_samples):
            for t in range(1, length):
                data[i, t] = alpha * data[i, t-1] + np.random.normal(0, 1)
        return data
    
    def _generate_sinusoid(self, n_samples: int, length: int, 
                           f_range: Tuple[float, float]) -> np.ndarray:
        """Generate sinusoidal signals with random frequency within range."""
        t = np.linspace(0, length / self.fs, length)
        freqs = np.random.uniform(f_range[0], f_range[1], n_samples)
        phases = np.random.uniform(0, 2 * np.pi, n_samples)
        
        data = np.zeros((n_samples, length))
        for i in range(n_samples):
            data[i] = np.sin(2 * np.pi * freqs[i] * t + phases[i])
        return data
    
    def _generate_data(self):
        """Generate all PID components and observations."""
        t = np.linspace(0, self.seq_length / self.fs, self.seq_length)
        
        # 1. Generate latent components
        
        # Redundancy: Low-frequency sinusoid (shared by both modalities)
        self.w_r = self._generate_sinusoid(self.n_samples, self.seq_length, self.f_redundancy)
        
        # Unique EEG: High-frequency oscillations
        self.w_u1 = self._generate_sinusoid(self.n_samples, self.seq_length, self.f_unique_eeg)
        
        # Unique fNIRS: Ultra-low frequency drift (AR process)
        self.w_u2 = self._generate_ar1(self.n_samples, self.seq_length, alpha=0.95)
        # Normalize
        self.w_u2 = (self.w_u2 - self.w_u2.mean(axis=1, keepdims=True)) / (self.w_u2.std(axis=1, keepdims=True) + 1e-8)
        
        # Synergy: Amplitude modulation (R * U1) - requires both modalities to separate
        # This creates sideband frequencies that are neither in R nor U1 alone
        self.w_s = self.w_r * self.w_u1
        
        # 2. Generate observations
        
        # X1 (EEG-like): R + U1 + S component
        self.x1 = (self.c_r * self.w_r + 
                   self.c_u * self.w_u1 + 
                   self.c_s * self.w_s +
                   np.random.normal(0, self.noise_level, (self.n_samples, self.seq_length)))
        
        # X2 (fNIRS-like): R + U2 + S component  
        self.x2 = (self.c_r * self.w_r + 
                   self.c_u * self.w_u2 + 
                   self.c_s * self.w_s +
                   np.random.normal(0, self.noise_level, (self.n_samples, self.seq_length)))
        
        # Normalize observations
        self.x1 = (self.x1 - self.x1.mean(axis=1, keepdims=True)) / (self.x1.std(axis=1, keepdims=True) + 1e-8)
        self.x2 = (self.x2 - self.x2.mean(axis=1, keepdims=True)) / (self.x2.std(axis=1, keepdims=True) + 1e-8)
        
        # Convert to tensors
        self.x1 = torch.tensor(self.x1, dtype=torch.float32)
        self.x2 = torch.tensor(self.x2, dtype=torch.float32)
        self.w_r = torch.tensor(self.w_r, dtype=torch.float32)
        self.w_u1 = torch.tensor(self.w_u1, dtype=torch.float32)
        self.w_u2 = torch.tensor(self.w_u2, dtype=torch.float32)
        self.w_s = torch.tensor(self.w_s, dtype=torch.float32)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x1': self.x1[idx],  # [seq_length]
            'x2': self.x2[idx],  # [seq_length]
            'w_r': self.w_r[idx],  # Ground truth redundancy
            'w_u1': self.w_u1[idx],  # Ground truth unique EEG
            'w_u2': self.w_u2[idx],  # Ground truth unique fNIRS
            'w_s': self.w_s[idx],  # Ground truth synergy
        }
    
    @staticmethod
    def compute_correlation(z: torch.Tensor, w: torch.Tensor) -> float:
        """Compute correlation between learned and ground truth latent."""
        # Flatten if needed
        z_flat = z.flatten()
        w_flat = w.flatten()
        
        z_centered = z_flat - z_flat.mean()
        w_centered = w_flat - w_flat.mean()
        
        corr = (z_centered * w_centered).sum() / (
            torch.sqrt((z_centered ** 2).sum()) * 
            torch.sqrt((w_centered ** 2).sum()) + 1e-8
        )
        return corr.item()


def create_dataset_from_config(config: Dict) -> PIDTimeSeriesDataset:
    """Create dataset from experiment config dict."""
    data_config = config.get("data", {}).get("params", {})
    return PIDTimeSeriesDataset(
        n_samples=data_config.get("n_samples", 5000),
        seq_length=data_config.get("seq_length", 256),
        fs=data_config.get("fs", 200.0),
        c_redundancy=data_config.get("c_redundancy", 1.0),
        c_unique=data_config.get("c_unique", 1.0),
        c_synergy=data_config.get("c_synergy", 1.0),
        noise_level=data_config.get("noise_level", 0.1),
        f_redundancy=tuple(data_config.get("f_redundancy", [0.1, 1.0])),
        f_unique_eeg=tuple(data_config.get("f_unique_eeg", [8.0, 30.0])),
        f_unique_fnirs=tuple(data_config.get("f_unique_fnirs", [0.01, 0.1])),
        seed=config.get("experiment", {}).get("seed", 42)
    )


if __name__ == "__main__":
    # Test
    print("Generating synthetic time-series dataset...")
    dataset = PIDTimeSeriesDataset(n_samples=100)
    sample = dataset[0]
    
    print(f"X1 shape: {sample['x1'].shape}")
    print(f"X2 shape: {sample['x2'].shape}")
    print(f"W_r shape: {sample['w_r'].shape}")
    print(f"W_s shape: {sample['w_s'].shape}")
    
    # Check frequency content
    import numpy as np
    x1_np = sample['x1'].numpy()
    fft = np.fft.fft(x1_np)
    freqs = np.fft.fftfreq(len(x1_np), 1/dataset.fs)
    
    print(f"\nDominant frequencies in X1:")
    magnitudes = np.abs(fft[:len(fft)//2])
    top_idx = np.argsort(magnitudes)[-5:]
    for i in top_idx:
        print(f"  {freqs[i]:.2f} Hz: magnitude {magnitudes[i]:.2f}")
    
    print("\nDataset generation successful!")
