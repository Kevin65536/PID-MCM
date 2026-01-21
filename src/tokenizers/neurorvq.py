"""
NeuroRVQ-style Tokenizer Implementation.

Inspired by: "NeuroRVQ: Multi-Scale EEG Tokenization for Generative Large Brainwave Models"
Paper: https://arxiv.org/abs/2510.13068

Key Features:
1. Multi-Scale Temporal Encoder (Inception-style with different kernel sizes)
2. Residual Vector Quantization (RVQ) with multiple quantizer layers
3. L2-normalized EMA-updated codebook (NormEMAVectorQuantizer)
4. Frequency-domain reconstruction (amplitude + sin/cos phase)

This implementation adapts NeuroRVQ's architecture for our single-channel patch-based tokenization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from einops import rearrange
import math

# Note: Does not inherit from BaseTokenizer due to different interface requirements


def l2norm(t: torch.Tensor) -> torch.Tensor:
    """L2 normalize along the last dimension."""
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    """Exponential moving average update in-place."""
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    """Sample vectors for k-means initialization."""
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 10, use_cosine_sim: bool = True):
    """K-means clustering for codebook initialization."""
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    means = sample_vectors(samples, num_clusters)
    
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)
        
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, buckets.unsqueeze(-1).expand(-1, dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        
        if use_cosine_sim:
            new_means = l2norm(new_means)
        
        means = torch.where(zero_mask[..., None], means, new_means)
    
    return means, bins


class NormEMAVectorQuantizer(nn.Module):
    """
    L2-normalized EMA Vector Quantizer from NeuroRVQ/LaBraM.
    
    Key differences from standard VQ:
    1. L2 normalization of both input and codebook
    2. EMA-based codebook updates (no gradient through codebook)
    3. Optional k-means initialization
    """
    
    def __init__(
        self,
        num_codes: int = 8192,
        code_dim: int = 64,
        beta: float = 1.0,  # commitment loss weight
        decay: float = 0.99,  # EMA decay
        eps: float = 1e-5,
        kmeans_init: bool = True,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        
        # Codebook
        if kmeans_init:
            weight = torch.zeros(num_codes, code_dim)
        else:
            weight = torch.randn(num_codes, code_dim)
            weight = l2norm(weight)
        
        self.register_buffer('embedding', weight)
        self.register_buffer('cluster_size', torch.zeros(num_codes))
        self.register_buffer('embed_avg', weight.clone())
        self.register_buffer('initted', torch.tensor([not kmeans_init]))
    
    def init_embed_(self, data: torch.Tensor):
        """Initialize codebook with k-means if not already done."""
        if self.initted.item():
            return
        
        embed, cluster_size = kmeans(data, self.num_codes, 10, use_cosine_sim=True)
        self.embedding.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size.float())
        self.embed_avg.data.copy_(embed)
        self.initted.data.fill_(True)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, D] or [B, N, D] input features
        Returns:
            z_q: quantized features (same shape as z)
            loss: VQ loss
            indices: codebook indices
        """
        need_reshape = z.dim() == 3
        if need_reshape:
            B, N, D = z.shape
            z = z.reshape(B * N, D)
        
        # L2 normalize input
        z = l2norm(z)
        
        # Initialize codebook on first forward
        self.init_embed_(z)
        
        # Compute distances (squared L2 = 2 - 2*cosine for normalized vectors)
        d = z.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.pow(2).sum(dim=1) - \
            2 * torch.einsum('bd,nd->bn', z, self.embedding)
        
        # Get nearest codes
        indices = torch.argmin(d, dim=1)
        z_q = F.embedding(indices, self.embedding)
        
        # EMA updates during training
        if self.training:
            encodings = F.one_hot(indices, self.num_codes).float()
            bins = encodings.sum(0)
            
            # Update cluster sizes
            ema_inplace(self.cluster_size, bins, self.decay)
            
            # Update embeddings
            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)
            
            embed_sum = z.t() @ encodings
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(
                zero_mask[..., None], 
                self.embedding,
                embed_normalized
            )
            
            # Normalized EMA update
            self.embedding.data.mul_(self.decay).add_(embed_normalized, alpha=(1 - self.decay))
            self.embedding.data.copy_(l2norm(self.embedding.data))
        
        # Commitment loss
        loss = self.beta * F.mse_loss(z_q.detach(), z)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        if need_reshape:
            z_q = z_q.reshape(B, N, D)
            indices = indices.reshape(B, N)
        
        return z_q, loss, indices
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode without training updates."""
        need_reshape = z.dim() == 3
        if need_reshape:
            B, N, D = z.shape
            z = z.reshape(B * N, D)
        
        z = l2norm(z)
        d = z.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.pow(2).sum(dim=1) - \
            2 * torch.einsum('bd,nd->bn', z, self.embedding)
        indices = torch.argmin(d, dim=1)
        
        if need_reshape:
            indices = indices.reshape(B, N)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to quantized vectors."""
        return F.embedding(indices, self.embedding)


class ResidualVectorQuantization(nn.Module):
    """
    Residual Vector Quantization (RVQ) as used in NeuroRVQ.
    
    Uses multiple quantizer layers that sequentially quantize
    the residual from the previous layer.
    """
    
    def __init__(
        self,
        num_quantizers: int = 8,
        num_codes: int = 8192,
        code_dim: int = 64,
        beta: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
        residual_loss_weight: float = 0.4,  # Additional residual MSE loss
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.residual_loss_weight = residual_loss_weight
        
        self.layers = nn.ModuleList([
            NormEMAVectorQuantizer(
                num_codes=num_codes,
                code_dim=code_dim,
                beta=beta,
                decay=decay,
                kmeans_init=kmeans_init,
            )
            for _ in range(num_quantizers)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        """
        Args:
            x: [B, D] or [B, N, D] input features
        Returns:
            quantized_out: sum of all quantized layers
            indices: [num_quantizers, B, ...] indices for each layer
            loss: combined loss
            usage_ratios: codebook usage per layer
        """
        quantized_out = torch.zeros_like(x)
        residual = x
        
        all_losses = []
        all_indices = []
        usage_ratios = []
        
        for layer in self.layers:
            quantized, loss, indices = layer(residual)
            
            # Additional residual loss (as in NeuroRVQ)
            new_residual = residual - quantized
            loss = loss + self.residual_loss_weight * F.mse_loss(quantized, residual.detach())
            
            residual = new_residual
            quantized_out = quantized_out + quantized
            
            all_indices.append(indices)
            all_losses.append(loss)
            
            # Track codebook usage
            unique_codes = torch.unique(indices)
            usage_ratio = unique_codes.numel() / self.num_codes
            usage_ratios.append(float(usage_ratio))
        
        out_indices = torch.stack(all_indices, dim=0)  # [num_quantizers, B, ...]
        total_loss = torch.stack(all_losses).mean()
        
        return quantized_out, out_indices, total_loss, usage_ratios
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to RVQ indices."""
        residual = x
        all_indices = []
        
        for layer in self.layers:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        
        return torch.stack(all_indices, dim=0)
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode RVQ indices.
        Args:
            indices: [num_quantizers, B, ...] or list of indices
        Returns:
            quantized_out: sum of decoded vectors from all layers
        """
        quantized_out = None
        for i, layer in enumerate(self.layers):
            if isinstance(indices, (list, tuple)):
                idx = indices[i]
            else:
                idx = indices[i]
            
            quantized = layer.decode(idx)
            if quantized_out is None:
                quantized_out = quantized
            else:
                quantized_out = quantized_out + quantized
        
        return quantized_out


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoder from NeuroRVQ.
    
    Inception-style architecture with 4 parallel branches using
    different kernel sizes to capture different frequency components.
    
    At 200Hz sampling rate:
    - Branch 1 (k=21): captures ~10Hz and below
    - Branch 2 (k=15): captures ~13Hz and below  
    - Branch 3 (k=9): captures ~22Hz and below
    - Branch 4 (k=5): captures ~40Hz and below
    """
    
    def __init__(
        self,
        patch_size: int = 200,
        in_channels: int = 1,
        hidden_channels: int = 8,
        output_dim: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_branches = 4
        
        # Branch 1: >10 Hz (large kernel for low freq)
        self.conv1_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=21, padding=10)
        self.norm1_1 = nn.GroupNorm(4, hidden_channels)
        self.pool1_1 = nn.AvgPool1d(kernel_size=2)
        self.conv1_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=9, padding=4)
        self.norm1_2 = nn.GroupNorm(4, hidden_channels)
        self.pool1_2 = nn.AvgPool1d(kernel_size=4)
        
        # Branch 2: >13 Hz
        self.conv2_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=15, padding=7)
        self.norm2_1 = nn.GroupNorm(4, hidden_channels)
        self.pool2_1 = nn.AvgPool1d(kernel_size=2)
        self.conv2_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3)
        self.norm2_2 = nn.GroupNorm(4, hidden_channels)
        self.pool2_2 = nn.AvgPool1d(kernel_size=4)
        
        # Branch 3: >20 Hz
        self.conv3_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=9, padding=4)
        self.norm3_1 = nn.GroupNorm(4, hidden_channels)
        self.pool3_1 = nn.AvgPool1d(kernel_size=2)
        self.conv3_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.norm3_2 = nn.GroupNorm(4, hidden_channels)
        self.pool3_2 = nn.AvgPool1d(kernel_size=4)
        
        # Branch 4: >40 Hz (small kernel for high freq)
        self.conv4_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.norm4_1 = nn.GroupNorm(4, hidden_channels)
        self.pool4_1 = nn.AvgPool1d(kernel_size=2)
        self.conv4_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm4_2 = nn.GroupNorm(4, hidden_channels)
        self.pool4_2 = nn.AvgPool1d(kernel_size=4)
        
        self.gelu = nn.GELU()
        
        # After 2x + 4x pooling = 8x reduction
        pooled_len = patch_size // 8
        self.flatten_dim = hidden_channels * pooled_len * self.num_branches
        
        # Projection to output dim
        self.proj = nn.Sequential(
            nn.Linear(self.flatten_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: [B, T] single patch (1 channel)
        Returns:
            combined: [B, D] combined features
            branches: tuple of 4 branch outputs [B, D//4] each
        """
        # [B, T] -> [B, 1, T]
        x = x.unsqueeze(1)
        
        # Branch 1
        b1 = self.pool1_1(self.gelu(self.norm1_1(self.conv1_1(x))))
        b1 = self.pool1_2(self.gelu(self.norm1_2(self.conv1_2(b1))))  # [B, C, T/8]
        
        # Branch 2
        b2 = self.pool2_1(self.gelu(self.norm2_1(self.conv2_1(x))))
        b2 = self.pool2_2(self.gelu(self.norm2_2(self.conv2_2(b2))))
        
        # Branch 3
        b3 = self.pool3_1(self.gelu(self.norm3_1(self.conv3_1(x))))
        b3 = self.pool3_2(self.gelu(self.norm3_2(self.conv3_2(b3))))
        
        # Branch 4
        b4 = self.pool4_1(self.gelu(self.norm4_1(self.conv4_1(x))))
        b4 = self.pool4_2(self.gelu(self.norm4_2(self.conv4_2(b4))))
        
        # Flatten each branch: [B, C, T'] -> [B, C*T']
        b1_flat = b1.flatten(1)
        b2_flat = b2.flatten(1)
        b3_flat = b3.flatten(1)
        b4_flat = b4.flatten(1)
        
        # Concatenate and project
        combined = torch.cat([b1_flat, b2_flat, b3_flat, b4_flat], dim=1)
        out = self.proj(combined)
        
        return out, (b1_flat, b2_flat, b3_flat, b4_flat)


class FreqDomainDecoder(nn.Module):
    """
    Decoder that reconstructs FFT amplitude and phase (sin/cos).
    Following NeuroRVQ's frequency-domain reconstruction approach.
    """
    
    def __init__(
        self,
        code_dim: int = 64,
        hidden_dim: int = 256,
        output_size: int = 200,  # patch_size
        num_quantizers: int = 8,  # For combining RVQ outputs
    ):
        super().__init__()
        self.output_size = output_size
        self.fft_size = output_size  # Full FFT output
        
        # Input projection (from quantized codes)
        self.input_proj = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Amplitude head
        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.fft_size),
        )
        
        # Phase heads (sin and cos)
        self.sin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.fft_size),
            nn.Tanh(),  # sin is in [-1, 1]
        )
        
        self.cos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.fft_size),
            nn.Tanh(),  # cos is in [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, D] quantized latent code
        Returns:
            amplitude: [B, fft_size] predicted log amplitude
            sin_phase: [B, fft_size] predicted sin of phase
            cos_phase: [B, fft_size] predicted cos of phase
        """
        h = self.input_proj(z)
        
        amplitude = self.amp_head(h)
        sin_phase = self.sin_head(h)
        cos_phase = self.cos_head(h)
        
        return amplitude, sin_phase, cos_phase


def inverse_fft_cos_sin(amplitude: torch.Tensor, sin_phase: torch.Tensor, cos_phase: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct time-domain signal from amplitude and sin/cos phase.
    
    Args:
        amplitude: [B, fft_size] amplitude (linear scale)
        sin_phase: [B, fft_size] sin of phase angle
        cos_phase: [B, fft_size] cos of phase angle
    Returns:
        signal: [B, T] reconstructed time-domain signal
    """
    # Reconstruct complex FFT: z = amp * (cos + j*sin)
    real = amplitude * cos_phase
    imag = amplitude * sin_phase
    fft_complex = torch.complex(real, imag)
    
    # Inverse FFT
    signal = torch.fft.ifft(fft_complex).real
    
    return signal


class NeuroRVQTokenizer(nn.Module):
    """
    NeuroRVQ-style Tokenizer.
    
    Architecture:
    1. Multi-scale temporal encoder extracts features at different frequency scales
    2. RVQ quantizes the features using multiple residual quantization layers
    3. Decoder reconstructs FFT amplitude and phase separately
    4. Loss combines VQ loss, amplitude loss, phase loss, and optional time-domain loss
    
    This is designed for single-channel patches (e.g., one EEG channel segment).
    
    Note: Does not inherit from BaseTokenizer due to different interface requirements.
    """
    
    def __init__(
        self,
        patch_size: int = 200,
        code_dim: int = 64,
        num_codes: int = 8192,
        num_quantizers: int = 8,  # Number of RVQ layers
        hidden_channels: int = 8,
        hidden_dim: int = 256,
        beta: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
        # Loss weights
        amplitude_weight: float = 1.0,
        phase_weight: float = 1.0,
        time_weight: float = 1.0,
        vq_weight: float = 1.0,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.num_quantizers = num_quantizers
        
        # Loss weights
        self.amplitude_weight = amplitude_weight
        self.phase_weight = phase_weight
        self.time_weight = time_weight
        self.vq_weight = vq_weight
        
        # Encoder
        self.encoder = MultiScaleTemporalEncoder(
            patch_size=patch_size,
            in_channels=1,
            hidden_channels=hidden_channels,
            output_dim=code_dim,
        )
        
        # RVQ Quantizer
        self.quantizer = ResidualVectorQuantization(
            num_quantizers=num_quantizers,
            num_codes=num_codes,
            code_dim=code_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        
        # Decoder
        self.decoder = FreqDomainDecoder(
            code_dim=code_dim,
            hidden_dim=hidden_dim,
            output_size=patch_size,
        )
    
    def std_norm(self, x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standardize along specified dimension."""
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std, mean, std
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode patches to tokens.
        
        Args:
            x: [B, T] single patch
        Returns:
            dict with 'tokens' [num_quantizers, B], 'quantized' [B, D], etc.
        """
        # Encode
        z, _ = self.encoder(x)
        
        # Quantize
        z_q, indices, vq_loss, usage_ratios = self.quantizer(z)
        
        return {
            'tokens': indices,  # [num_quantizers, B]
            'quantized': z_q,
            'pre_quant': z,
            'vq_loss': vq_loss,
            'usage_ratios': usage_ratios,
        }
    
    def decode(self, z_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode quantized features.
        
        Args:
            z_q: [B, D] quantized features
        Returns:
            dict with amplitude, sin_phase, cos_phase, reconstructed
        """
        amplitude, sin_phase, cos_phase = self.decoder(z_q)
        
        # Reconstruct time domain
        amp_linear = torch.expm1(amplitude)  # Undo log1p
        reconstructed = inverse_fft_cos_sin(amp_linear, sin_phase, cos_phase)
        
        return {
            'amplitude': amplitude,
            'sin_phase': sin_phase,
            'cos_phase': cos_phase,
            'reconstructed': reconstructed,
        }
    
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches to tokens.
        
        Args:
            x: [B, T] patches
        Returns:
            tokens: [num_quantizers, B] RVQ indices
        """
        with torch.no_grad():
            z, _ = self.encoder(x)
            indices = self.quantizer.encode(z)
        return indices
    
    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert tokens back to patches.
        
        Args:
            tokens: [num_quantizers, B] RVQ indices
        Returns:
            reconstructed: [B, T] time domain signals
        """
        with torch.no_grad():
            z_q = self.quantizer.decode(tokens)
            decoded = self.decode(z_q)
        return decoded['reconstructed']
    
    def compute_loss(
        self,
        x: torch.Tensor,
        encode_output: Dict[str, torch.Tensor],
        decode_output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            x: [B, T] original patches
            encode_output: output from encode()
            decode_output: output from decode()
        Returns:
            dict with individual and total losses
        """
        # Compute ground truth FFT
        x_fft = torch.fft.fft(x, dim=-1)
        target_amp = torch.log1p(torch.abs(x_fft))
        target_angle = torch.angle(x_fft)
        target_sin = torch.sin(target_angle)
        target_cos = torch.cos(target_angle)
        
        # Standardize amplitudes for loss
        target_amp_norm, _, _ = self.std_norm(target_amp)
        pred_amp_norm, _, _ = self.std_norm(decode_output['amplitude'])
        
        # Amplitude loss
        amp_loss = F.mse_loss(pred_amp_norm, target_amp_norm)
        
        # Phase loss (cosine similarity + magnitude constraint)
        pred_phase_vec = torch.stack([decode_output['cos_phase'], decode_output['sin_phase']], dim=-1)
        target_phase_vec = torch.stack([target_cos, target_sin], dim=-1)
        phase_cos_sim = F.cosine_similarity(pred_phase_vec, target_phase_vec, dim=-1).mean()
        phase_mag_loss = ((decode_output['sin_phase']**2 + decode_output['cos_phase']**2 - 1)**2).mean()
        phase_loss = (1.0 - phase_cos_sim) + 0.1 * phase_mag_loss
        
        # Time domain loss
        x_norm, x_mean, x_std = self.std_norm(x)
        recon_norm, _, _ = self.std_norm(decode_output['reconstructed'])
        time_loss = F.mse_loss(recon_norm, x_norm)
        
        # VQ loss
        vq_loss = encode_output['vq_loss']
        
        # Total loss
        total_loss = (
            self.amplitude_weight * amp_loss +
            self.phase_weight * phase_loss +
            self.time_weight * time_loss +
            self.vq_weight * vq_loss
        )
        
        return {
            'loss': total_loss,
            'amp_loss': amp_loss,
            'phase_loss': phase_loss,
            'time_loss': time_loss,
            'vq_loss': vq_loss,
            'phase_cos_sim': phase_cos_sim,
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: [B, T] patches
        Returns:
            dict with all outputs and losses
        """
        # Encode and quantize
        encode_out = self.encode(x)
        
        # Decode
        decode_out = self.decode(encode_out['quantized'])
        
        # Compute losses
        losses = self.compute_loss(x, encode_out, decode_out)
        
        # Combine outputs
        return {
            **encode_out,
            **decode_out,
            **losses,
        }
    
    def get_codebook_usage(self) -> Dict[str, float]:
        """Get codebook utilization stats for each RVQ layer."""
        usage = {}
        for i, layer in enumerate(self.quantizer.layers):
            active = (layer.cluster_size > 0).sum().item()
            usage[f'layer_{i}_active'] = active
            usage[f'layer_{i}_utilization'] = active / layer.num_codes
        return usage


class NeuroRVQTokenizer_V2(NeuroRVQTokenizer):
    """
    Version 2 with separate RVQ for each encoder branch.
    
    More closely follows NeuroRVQ's multi-scale architecture where
    each frequency scale has its own RVQ codebook.
    """
    
    def __init__(
        self,
        patch_size: int = 200,
        code_dim: int = 64,
        num_codes: int = 8192,
        num_quantizers: int = 8,
        hidden_channels: int = 8,
        hidden_dim: int = 256,
        beta: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
        amplitude_weight: float = 1.0,
        phase_weight: float = 1.0,
        time_weight: float = 1.0,
        vq_weight: float = 1.0,
    ):
        # Skip parent __init__ and call grandparent
        nn.Module.__init__(self)
        
        self.patch_size = patch_size
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.num_quantizers = num_quantizers
        self.num_branches = 4
        
        # Loss weights
        self.amplitude_weight = amplitude_weight
        self.phase_weight = phase_weight
        self.time_weight = time_weight
        self.vq_weight = vq_weight
        
        # Encoder
        self.encoder = MultiScaleTemporalEncoder(
            patch_size=patch_size,
            in_channels=1,
            hidden_channels=hidden_channels,
            output_dim=code_dim,
        )
        
        # Per-branch projection (from flattened branch output to code_dim)
        branch_flat_dim = hidden_channels * (patch_size // 8)
        self.branch_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(branch_flat_dim, code_dim),
                nn.LayerNorm(code_dim),
                nn.GELU(),
            )
            for _ in range(self.num_branches)
        ])
        
        # Separate RVQ for each branch
        self.quantizers = nn.ModuleList([
            ResidualVectorQuantization(
                num_quantizers=num_quantizers,
                num_codes=num_codes,
                code_dim=code_dim,
                beta=beta,
                decay=decay,
                kmeans_init=kmeans_init,
            )
            for _ in range(self.num_branches)
        ])
        
        # Decoder (input is concatenated branch outputs)
        self.decoder = FreqDomainDecoder(
            code_dim=code_dim * self.num_branches,
            hidden_dim=hidden_dim,
            output_size=patch_size,
        )
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode with separate RVQ per branch.
        """
        # Get branch features
        _, branch_features = self.encoder(x)
        
        # Quantize each branch
        all_tokens = []
        all_quantized = []
        total_vq_loss = 0.0
        all_usage = []
        
        for i, (branch_feat, proj, quant) in enumerate(
            zip(branch_features, self.branch_projs, self.quantizers)
        ):
            # Project to code_dim
            z = proj(branch_feat)
            
            # Quantize
            z_q, indices, vq_loss, usage = quant(z)
            
            all_tokens.append(indices)
            all_quantized.append(z_q)
            total_vq_loss = total_vq_loss + vq_loss
            all_usage.append(usage)
        
        # Stack quantized outputs
        combined_z_q = torch.cat(all_quantized, dim=-1)  # [B, code_dim * 4]
        
        return {
            'tokens': all_tokens,  # List of [num_quantizers, B]
            'quantized': combined_z_q,  # [B, code_dim * 4]
            'branch_quantized': all_quantized,
            'vq_loss': total_vq_loss / self.num_branches,
            'usage_ratios': all_usage,
        }
    
    def tokenize(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert patches to tokens (one set per branch).
        """
        with torch.no_grad():
            _, branch_features = self.encoder(x)
            all_tokens = []
            
            for branch_feat, proj, quant in zip(
                branch_features, self.branch_projs, self.quantizers
            ):
                z = proj(branch_feat)
                indices = quant.encode(z)
                all_tokens.append(indices)
        
        return all_tokens
    
    def detokenize(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert tokens back to patches.
        """
        with torch.no_grad():
            all_quantized = []
            for tok, quant in zip(tokens, self.quantizers):
                z_q = quant.decode(tok)
                all_quantized.append(z_q)
            
            combined_z_q = torch.cat(all_quantized, dim=-1)
            decoded = self.decode(combined_z_q)
        
        return decoded['reconstructed']
