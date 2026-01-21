"""
Frequency-Domain Patch VQ-VAE Tokenizer following LaBraM/NeuroRVQ standards.

Key Innovation (based on LaBraM):
- Instead of reconstructing in time domain, we reconstruct in frequency domain
- Input → FFT → amplitude + phase
- Encode and quantize amplitude representations
- Decode to predict amplitude and phase separately
- Loss = amplitude_loss + phase_loss (+ optional time_loss via iFFT)

This leads to much better reconstruction for EEG signals where frequency
band information is crucial.

Reference:
- LaBraM: Predicts FFT amplitude and angle (phase) separately
- NeuroRVQ: Uses multi-scale temporal convolutions + RVQ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from einops import rearrange

from .base import BaseTokenizer
from .vqvae import VectorQuantizer


class MultiScaleTemporalEncoder(nn.Module):
    """
    Inception-style multi-scale temporal encoder following NeuroRVQ.
    Uses multiple parallel conv branches with different kernel sizes
    to capture different frequency components.
    """
    
    def __init__(
        self,
        patch_size: int = 200,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_scales: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.num_scales = num_scales
        
        # Multi-scale kernel sizes (inspired by NeuroRVQ)
        # Targeting different frequency ranges at 200Hz:
        # k=21 -> ~10Hz, k=15 -> ~13Hz, k=9 -> ~22Hz, k=5 -> ~40Hz
        kernel_sizes = [21, 15, 9, 5][:num_scales]
        
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            padding = ks // 2
            branch = nn.Sequential(
                nn.Conv1d(1, hidden_dim // num_scales, kernel_size=ks, padding=padding),
                nn.GroupNorm(4, hidden_dim // num_scales),
                nn.GELU(),
                nn.AvgPool1d(kernel_size=2),
                nn.Conv1d(hidden_dim // num_scales, hidden_dim // num_scales, 
                         kernel_size=max(3, ks // 2), padding=max(1, ks // 4)),
                nn.GroupNorm(4, hidden_dim // num_scales),
                nn.GELU(),
                nn.AvgPool1d(kernel_size=4),
            )
            self.branches.append(branch)
        
        # Combine branches
        pooled_len = patch_size // 8  # After 2x and 4x pooling
        self.flatten_dim = hidden_dim * pooled_len
        self.proj = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T_patch] single patch
        Returns:
            z: [B, D] latent vector
        """
        # [B, T] -> [B, 1, T]
        x = x.unsqueeze(1)
        
        # Process through each branch
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)  # [B, C/num_scales, T']
            branch_outputs.append(out)
        
        # Concatenate along channel dimension
        combined = torch.cat(branch_outputs, dim=1)  # [B, C, T']
        
        # Flatten and project
        combined = combined.flatten(1)  # [B, C*T']
        z = self.proj(combined)  # [B, D]
        
        return z


class FreqPatchEncoder(nn.Module):
    """
    Encoder that processes FFT amplitude of each patch.
    More suitable for EEG signals where frequency information is key.
    """
    
    def __init__(
        self,
        patch_size: int = 200,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 2,
        encoder_type: str = "multiscale",  # "cnn", "mlp", "multiscale"
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        
        # FFT size (we use rfft, so output is patch_size // 2 + 1)
        self.fft_size = patch_size // 2 + 1
        
        if encoder_type == "multiscale":
            self.encoder = MultiScaleTemporalEncoder(
                patch_size=self.fft_size,  # Encode FFT magnitude
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            )
        elif encoder_type == "cnn":
            layers = []
            in_dim = 1
            current_len = self.fft_size
            
            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
                layers.extend([
                    nn.Conv1d(in_dim if i == 0 else hidden_dim, out_dim,
                             kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                ])
                current_len = (current_len + 1) // 2
            
            self.conv = nn.Sequential(*layers)
            self.flatten_dim = hidden_dim * current_len
            self.proj = nn.Linear(self.flatten_dim, output_dim)
            
        elif encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(self.fft_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
    
    def forward(self, amplitude: torch.Tensor) -> torch.Tensor:
        """
        Args:
            amplitude: [B, fft_size] FFT amplitude
        Returns:
            z: [B, D] latent vector
        """
        if self.encoder_type == "multiscale":
            z = self.encoder(amplitude)
        elif self.encoder_type == "cnn":
            x = amplitude.unsqueeze(1)  # [B, 1, F]
            x = self.conv(x)
            x = x.flatten(1)
            z = self.proj(x)
        elif self.encoder_type == "mlp":
            z = self.encoder(amplitude)
        
        return z


class FreqPatchDecoder(nn.Module):
    """
    Decoder that predicts FFT amplitude and phase separately.
    Following LaBraM's approach of separate amplitude and angle prediction.
    """
    
    def __init__(
        self,
        patch_size: int = 200,
        hidden_dim: int = 256,
        input_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.fft_size = patch_size // 2 + 1
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Separate heads for amplitude and phase (like LaBraM)
        self.amplitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.fft_size),
            nn.Softplus(),  # Amplitude should be positive
        )
        
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.fft_size),
            nn.Tanh(),  # Phase normalized to [-1, 1] (scaled from [-π, π])
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, D] latent vector
        Returns:
            amplitude: [B, fft_size] predicted FFT amplitude
            phase: [B, fft_size] predicted FFT phase (normalized)
        """
        features = self.backbone(z)
        amplitude = self.amplitude_head(features)
        phase = self.phase_head(features)
        
        return amplitude, phase


class FreqDomainPatchVQVAE(BaseTokenizer):
    """
    Frequency-Domain Patch VQ-VAE Tokenizer.
    
    Key differences from time-domain PatchVQVAE:
    1. Input is FFT-transformed before encoding
    2. Encodes FFT amplitude (log-magnitude for stability)
    3. Decodes to both amplitude and phase
    4. Reconstruction via inverse FFT
    
    This approach is more suitable for EEG/neural signals where
    frequency information is semantically meaningful.
    """
    
    def __init__(
        self,
        seq_length: int = 800,
        patch_size: int = 200,
        input_channels: int = 1,
        codebook_size: int = 2048,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        encoder_type: str = "multiscale",
        commitment_cost: float = 0.1,
        ema_decay: float = 0.99,
        amplitude_loss_weight: float = 1.0,
        phase_loss_weight: float = 0.5,
        time_loss_weight: float = 0.5,  # Optional time domain loss
        use_log_amplitude: bool = True,
        **kwargs
    ):
        super().__init__(input_dim=input_channels, latent_dim=embedding_dim)
        
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.fft_size = patch_size // 2 + 1
        
        # Loss weights
        self.amplitude_loss_weight = amplitude_loss_weight
        self.phase_loss_weight = phase_loss_weight
        self.time_loss_weight = time_loss_weight
        self.use_log_amplitude = use_log_amplitude
        
        # Calculate number of tokens
        assert seq_length % patch_size == 0, \
            f"seq_length ({seq_length}) must be divisible by patch_size ({patch_size})"
        self.n_tokens = seq_length // patch_size
        
        # Encoder
        self.encoder = FreqPatchEncoder(
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            encoder_type=encoder_type,
        )
        
        # VQ layer
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )
        
        # Decoder
        self.decoder = FreqPatchDecoder(
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            input_dim=embedding_dim,
            num_layers=num_layers,
        )
        
        print(f"FreqDomainPatchVQVAE initialized:")
        print(f"  - Input: {seq_length} samples")
        print(f"  - Patch size: {patch_size} samples ({patch_size/200:.2f}s @ 200Hz)")
        print(f"  - FFT size: {self.fft_size}")
        print(f"  - Tokens per window: {self.n_tokens}")
        print(f"  - Codebook size: {codebook_size}")
        print(f"  - Embedding dim: {embedding_dim}")
        print(f"  - Encoder type: {encoder_type}")
        print(f"  - Loss weights: amp={amplitude_loss_weight}, phase={phase_loss_weight}, time={time_loss_weight}")
    
    def _split_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Split input into patches."""
        B = x.shape[0]
        patches = x.view(B, self.n_tokens, self.patch_size)
        patches = patches.view(B * self.n_tokens, self.patch_size)
        return patches
    
    def _merge_patches(self, patches: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Merge patches back to full sequence."""
        patches = patches.view(batch_size, self.n_tokens, self.patch_size)
        x = patches.view(batch_size, -1)
        return x
    
    def _compute_fft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute FFT and return amplitude and phase.
        
        Args:
            x: [B, T] time domain signal
        Returns:
            amplitude: [B, F] FFT amplitude (optionally log-scaled)
            phase: [B, F] FFT phase (normalized to [-1, 1])
        """
        fft = torch.fft.rfft(x, dim=-1)
        amplitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Normalize amplitude (log for stability, like LaBraM)
        if self.use_log_amplitude:
            amplitude = torch.log(amplitude + 1e-8)
        
        # Normalize phase to [-1, 1]
        phase = phase / torch.pi
        
        return amplitude, phase
    
    def _compute_ifft(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct time domain signal from amplitude and phase.
        
        Args:
            amplitude: [B, F] FFT amplitude
            phase: [B, F] FFT phase (normalized to [-1, 1])
        Returns:
            x: [B, T] reconstructed time domain signal
        """
        # De-normalize
        if self.use_log_amplitude:
            amplitude = torch.exp(amplitude)
        phase = phase * torch.pi
        
        # Construct complex FFT
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        fft = torch.complex(real, imag)
        
        # Inverse FFT
        x = torch.fft.irfft(fft, n=self.patch_size, dim=-1)
        
        return x
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent representations.
        
        Args:
            x: [B, T] input signal
        Returns:
            z: [B, N, D] latent vectors
            amplitude: [B*N, F] FFT amplitudes (for loss computation)
            phase: [B*N, F] FFT phases (for loss computation)
        """
        B = x.shape[0]
        
        # Split to patches
        patches = self._split_to_patches(x)  # [B*N, T_patch]
        
        # Compute FFT
        amplitude, phase = self._compute_fft(patches)  # [B*N, F]
        
        # Encode amplitude
        z = self.encoder(amplitude)  # [B*N, D]
        
        # Reshape to [B, N, D]
        z = z.view(B, self.n_tokens, self.embedding_dim)
        
        return z, amplitude, phase
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Quantize latent vectors."""
        return self.quantizer(z)
    
    def decode(self, z_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode quantized vectors to amplitude and phase.
        
        Args:
            z_q: [B, N, D] quantized vectors
        Returns:
            amplitude: [B*N, F] predicted amplitude
            phase: [B*N, F] predicted phase
        """
        B = z_q.shape[0]
        
        # Flatten to [B*N, D]
        z_flat = z_q.view(B * self.n_tokens, self.embedding_dim)
        
        # Decode to amplitude and phase
        amplitude, phase = self.decoder(z_flat)
        
        return amplitude, phase
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with frequency domain reconstruction.
        
        Args:
            x: [B, T] input signal
        Returns:
            dict with x_rec, losses, etc.
        """
        B = x.shape[0]
        
        # Encode (also returns FFT components for loss)
        z, orig_amplitude, orig_phase = self.encode(x)
        
        # Quantize
        z_q, indices, vq_info = self.quantize(z)
        
        # Decode to amplitude and phase
        pred_amplitude, pred_phase = self.decode(z_q)
        
        # Compute losses
        # 1. Amplitude loss (MSE in log domain)
        amplitude_loss = F.mse_loss(pred_amplitude, orig_amplitude)
        
        # 2. Phase loss (MSE on normalized phase)
        phase_loss = F.mse_loss(pred_phase, orig_phase)
        
        # 3. Time domain reconstruction (optional)
        x_rec_patches = self._compute_ifft(pred_amplitude, pred_phase)
        x_rec = self._merge_patches(x_rec_patches, B)
        time_loss = F.mse_loss(x_rec, x)
        
        # Total reconstruction loss
        total_rec_loss = (
            self.amplitude_loss_weight * amplitude_loss +
            self.phase_loss_weight * phase_loss +
            self.time_loss_weight * time_loss
        )
        
        return {
            'x_rec': x_rec,
            'z': z,
            'z_q': z_q,
            'indices': indices,
            'rec_loss': total_rec_loss,
            'amplitude_loss': amplitude_loss,
            'phase_loss': phase_loss,
            'time_loss': time_loss,
            'commitment_loss': vq_info['commitment_loss'],
            'codebook_loss': vq_info['codebook_loss'],
            'perplexity': vq_info['perplexity'],
            'dead_ratio': vq_info['dead_ratio'],
            'code_utilization': vq_info['code_utilization'],
        }
    
    def get_codebook_size(self) -> int:
        return self.codebook_size
    
    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for indices."""
        return self.quantizer.embedding(indices)
    
    def get_codebook_embeddings(self) -> torch.Tensor:
        """Get all codebook embeddings."""
        return self.quantizer.embedding.weight.detach()
    
    def tokens_to_signal(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to reconstructed signal.
        
        Args:
            indices: [B, N] token indices
        Returns:
            x: [B, T] reconstructed signal
        """
        B, N = indices.shape
        
        # Get embeddings
        z_q = self.get_embedding(indices)  # [B, N, D]
        
        # Decode to amplitude and phase
        amplitude, phase = self.decode(z_q)
        
        # Reconstruct via iFFT
        x_rec_patches = self._compute_ifft(amplitude, phase)
        x_rec = self._merge_patches(x_rec_patches, B)
        
        return x_rec


class FreqDomainPatchVQVAE_V2(FreqDomainPatchVQVAE):
    """
    V2: Enhanced with sin/cos phase representation (more stable).
    Following NeuroRVQ's approach of using sin(phase) and cos(phase)
    instead of raw phase angles.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override decoder to predict sin/cos instead of raw phase
        self.decoder = FreqPatchDecoderV2(
            patch_size=self.patch_size,
            hidden_dim=kwargs.get('hidden_dim', 256),
            input_dim=self.embedding_dim,
        )
        print("  - Using sin/cos phase representation (V2)")
    
    def _compute_fft_v2(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute FFT and return amplitude, sin(phase), cos(phase).
        """
        fft = torch.fft.rfft(x, dim=-1)
        amplitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        if self.use_log_amplitude:
            amplitude = torch.log(amplitude + 1e-8)
        
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        
        return amplitude, sin_phase, cos_phase
    
    def _compute_ifft_v2(self, amplitude: torch.Tensor, 
                         sin_phase: torch.Tensor, cos_phase: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct time domain from amplitude, sin(phase), cos(phase).
        Following NeuroRVQ.
        """
        if self.use_log_amplitude:
            amplitude = torch.exp(amplitude)
        
        # real = amplitude * cos(phase), imag = amplitude * sin(phase)
        real = amplitude * cos_phase
        imag = amplitude * sin_phase
        fft = torch.complex(real, imag)
        
        x = torch.fft.irfft(fft, n=self.patch_size, dim=-1)
        return x
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """V2 forward with sin/cos phase."""
        B = x.shape[0]
        
        # Split and FFT
        patches = self._split_to_patches(x)
        orig_amplitude, orig_sin, orig_cos = self._compute_fft_v2(patches)
        
        # Encode amplitude
        z = self.encoder(orig_amplitude)
        z = z.view(B, self.n_tokens, self.embedding_dim)
        
        # Quantize
        z_q, indices, vq_info = self.quantize(z)
        
        # Decode
        z_flat = z_q.view(B * self.n_tokens, self.embedding_dim)
        pred_amplitude, pred_sin, pred_cos = self.decoder(z_flat)
        
        # Losses
        amplitude_loss = F.mse_loss(pred_amplitude, orig_amplitude)
        sin_loss = F.mse_loss(pred_sin, orig_sin)
        cos_loss = F.mse_loss(pred_cos, orig_cos)
        phase_loss = sin_loss + cos_loss
        
        # Time domain reconstruction
        x_rec_patches = self._compute_ifft_v2(pred_amplitude, pred_sin, pred_cos)
        x_rec = self._merge_patches(x_rec_patches, B)
        time_loss = F.mse_loss(x_rec, x)
        
        total_rec_loss = (
            self.amplitude_loss_weight * amplitude_loss +
            self.phase_loss_weight * phase_loss +
            self.time_loss_weight * time_loss
        )
        
        return {
            'x_rec': x_rec,
            'z': z,
            'z_q': z_q,
            'indices': indices,
            'rec_loss': total_rec_loss,
            'amplitude_loss': amplitude_loss,
            'phase_loss': phase_loss,
            'time_loss': time_loss,
            'commitment_loss': vq_info['commitment_loss'],
            'codebook_loss': vq_info['codebook_loss'],
            'perplexity': vq_info['perplexity'],
            'dead_ratio': vq_info['dead_ratio'],
            'code_utilization': vq_info['code_utilization'],
        }


class FreqPatchDecoderV2(nn.Module):
    """Decoder that predicts amplitude, sin(phase), cos(phase)."""
    
    def __init__(
        self,
        patch_size: int = 200,
        hidden_dim: int = 256,
        input_dim: int = 64,
    ):
        super().__init__()
        self.fft_size = patch_size // 2 + 1
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.amplitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.fft_size),
        )
        
        self.sin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.fft_size),
            nn.Tanh(),  # sin is in [-1, 1]
        )
        
        self.cos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.fft_size),
            nn.Tanh(),  # cos is in [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(z)
        amplitude = self.amplitude_head(features)
        sin_phase = self.sin_head(features)
        cos_phase = self.cos_head(features)
        return amplitude, sin_phase, cos_phase
