"""
Reconstruction quality metrics for tokenizer evaluation.

These metrics measure how well the tokenizer can reconstruct
the original signal from its discrete representation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


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


def compute_stft_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    n_fft: int = 256,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute STFT-based spectral loss.
    
    Args:
        x: Original signal [B, T]
        x_rec: Reconstructed signal [B, T]
        n_fft: FFT size
        hop_length: Hop length (default: n_fft // 4)
        win_length: Window length (default: n_fft)
        
    Returns:
        Spectral loss (sum of magnitude and log-magnitude losses)
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    
    # Ensure 2D input [B, T]
    if x.dim() == 3:
        x = x.squeeze(1)
        x_rec = x_rec.squeeze(1)
    
    # Pad if needed
    if x.shape[-1] < n_fft:
        pad_size = n_fft - x.shape[-1]
        x = F.pad(x, (0, pad_size))
        x_rec = F.pad(x_rec, (0, pad_size))
    
    window = torch.hann_window(win_length, device=x.device)
    
    stft_x = torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
    stft_rec = torch.stft(x_rec, n_fft, hop_length, win_length, window, return_complex=True)
    
    # Magnitude
    mag_x = stft_x.abs()
    mag_rec = stft_rec.abs()
    
    # Magnitude loss
    mag_loss = F.l1_loss(mag_rec, mag_x)
    
    # Log-magnitude loss (with floor to avoid log(0))
    log_mag_x = torch.log(mag_x.clamp(min=1e-7))
    log_mag_rec = torch.log(mag_rec.clamp(min=1e-7))
    log_mag_loss = F.l1_loss(log_mag_rec, log_mag_x)
    
    return mag_loss + log_mag_loss


def compute_multi_stft_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    fft_sizes: List[int] = [64, 128, 256, 512],
    hop_sizes: Optional[List[int]] = None,
    win_sizes: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Compute multi-scale STFT loss.
    
    Using multiple scales helps capture both fine and coarse spectral features.
    
    Args:
        x: Original signal [B, T]
        x_rec: Reconstructed signal [B, T]
        fft_sizes: List of FFT sizes
        hop_sizes: List of hop sizes (default: fft_size // 4)
        win_sizes: List of window sizes (default: fft_size)
        
    Returns:
        Average loss across all scales
    """
    if hop_sizes is None:
        hop_sizes = [n // 4 for n in fft_sizes]
    if win_sizes is None:
        win_sizes = fft_sizes
    
    total_loss = torch.tensor(0.0, device=x.device)
    
    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
        # Adjust if signal is too short
        if x.shape[-1] < n_fft:
            continue
        total_loss = total_loss + compute_stft_loss(x, x_rec, n_fft, hop, win)
    
    return total_loss / len(fft_sizes)


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


def compute_band_power_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    fs: float = 200.0,
    bands: Dict[str, Tuple[float, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute band power preservation for EEG-specific evaluation.
    
    Args:
        x: Original signal [B, T]
        x_rec: Reconstructed signal [B, T]
        fs: Sampling frequency
        bands: Dict of band name -> (low_freq, high_freq)
        
    Returns:
        Dict with loss for each band
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
        }
    
    # Ensure 2D input
    if x.dim() == 3:
        x = x.squeeze(1)
        x_rec = x_rec.squeeze(1)
    
    n_fft = x.shape[-1]
    freqs = torch.fft.rfftfreq(n_fft, d=1/fs).to(x.device)
    
    fft_x = torch.fft.rfft(x, n=n_fft)
    fft_rec = torch.fft.rfft(x_rec, n=n_fft)
    
    losses = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        if mask.sum() == 0:
            continue
        
        power_x = (fft_x.abs()[:, mask] ** 2).mean(dim=-1)
        power_rec = (fft_rec.abs()[:, mask] ** 2).mean(dim=-1)
        
        losses[f'{band_name}_power_loss'] = F.mse_loss(power_rec, power_x)
    
    return losses


def compute_smoothness_loss(x_rec: torch.Tensor) -> torch.Tensor:
    """
    Compute smoothness loss for fNIRS reconstruction.
    
    fNIRS signals should be smooth; this penalizes high-frequency noise.
    
    Args:
        x_rec: Reconstructed signal [B, T]
        
    Returns:
        Smoothness loss (lower = smoother)
    """
    # First-order difference
    diff1 = x_rec[..., 1:] - x_rec[..., :-1]
    
    # Second-order difference
    diff2 = diff1[..., 1:] - diff1[..., :-1]
    
    return (diff2 ** 2).mean()


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
