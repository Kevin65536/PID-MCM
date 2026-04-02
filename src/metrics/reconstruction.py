"""Reconstruction quality metrics for tokenizer evaluation.

Loss-style reconstruction objectives live in ``src.losses.reconstruction``.
This module re-exports them for backward compatibility while keeping the
metric-oriented helpers here.
"""

import torch
import torch.nn.functional as F

from ..losses.reconstruction import (
    compute_band_power_loss,
    compute_multi_stft_loss,
    compute_smoothness_loss,
    compute_stft_loss,
)


def compute_reconstruction_mse(
    x: torch.Tensor, 
    x_rec: torch.Tensor,
    reduce: str = 'mean'
) -> torch.Tensor:
    """
    Compute Mean Squared Error between original and reconstructed signals.
    
    Args:
        x: Original signal [B, T] or [B, C, T]
        x_rec: Reconstructed signal [B, T] or [B, C, T]
        reduce: 'mean', 'sum', or 'none'
        
    Returns:
        MSE loss
    """
    if reduce == 'mean':
        return F.mse_loss(x_rec, x)
    elif reduce == 'sum':
        return F.mse_loss(x_rec, x, reduction='sum')
    else:
        return F.mse_loss(x_rec, x, reduction='none')


def compute_mae(x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error.
    
    Args:
        x: Original signal [B, T] or [B, C, T]
        x_rec: Reconstructed signal [B, T] or [B, C, T]
        
    Returns:
        MAE loss
    """
    return F.l1_loss(x_rec, x)


def compute_snr(x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        x: Original signal [B, T] or [B, C, T]
        x_rec: Reconstructed signal [B, T] or [B, C, T]
        
    Returns:
        SNR in dB (higher is better)
    """
    noise = x - x_rec
    signal_power = (x ** 2).mean()
    noise_power = (noise ** 2).mean()
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr


def compute_spectral_mse(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    n_fft: int = 256,
) -> torch.Tensor:
    """
    Compute MSE in frequency domain using FFT.
    
    Args:
        x: Original signal [B, T]
        x_rec: Reconstructed signal [B, T]
        n_fft: FFT size
        
    Returns:
        Spectral MSE
    """
    # Ensure 2D input
    if x.dim() == 3:
        x = x.squeeze(1)
        x_rec = x_rec.squeeze(1)
    
    # Pad if needed
    if x.shape[-1] < n_fft:
        pad_size = n_fft - x.shape[-1]
        x = F.pad(x, (0, pad_size))
        x_rec = F.pad(x_rec, (0, pad_size))
    
    # FFT
    fft_x = torch.fft.rfft(x, n=n_fft)
    fft_rec = torch.fft.rfft(x_rec, n=n_fft)
    
    # MSE on magnitude
    return F.mse_loss(fft_rec.abs(), fft_x.abs())


if __name__ == "__main__":
    # Test reconstruction metrics
    print("Testing reconstruction metrics...")
    
    # Simulated signals
    B, T = 32, 256
    x = torch.randn(B, T)
    noise = 0.1 * torch.randn(B, T)
    x_rec = x + noise  # Noisy reconstruction
    
    # MSE
    mse = compute_reconstruction_mse(x, x_rec)
    print(f"Reconstruction MSE: {mse.item():.4f}")
    
    # SNR
    snr = compute_snr(x, x_rec)
    print(f"SNR: {snr.item():.2f} dB")
    
    # Spectral MSE
    spectral_mse = compute_spectral_mse(x, x_rec)
    print(f"Spectral MSE: {spectral_mse.item():.4f}")
    
    # Multi-scale STFT
    ms_stft = compute_multi_stft_loss(x, x_rec, fft_sizes=[32, 64, 128])
    print(f"Multi-scale STFT loss: {ms_stft.item():.4f}")
    
    # Band power
    band_losses = compute_band_power_loss(x, x_rec)
    print(f"Band power losses:")
    for k, v in band_losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Smoothness
    smoothness = compute_smoothness_loss(x_rec)
    print(f"Smoothness loss: {smoothness.item():.4f}")
