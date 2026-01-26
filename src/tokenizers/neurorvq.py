"""
NeuroRVQ Tokenizer Implementation.

Faithfully follows the original NeuroRVQ architecture from:
"NeuroRVQ: Multi-Scale EEG Tokenization for Generative Large Brainwave Models"
Paper: https://arxiv.org/abs/2510.13068

Key Features:
1. Multi-Scale Temporal Encoder (Inception-style with 4 branches for different frequency bands)
2. 4 separate RVQ quantizers (one per encoder branch)
3. L2-normalized EMA-updated codebook (NormEMAVectorQuantizer)
4. Frequency-domain reconstruction (amplitude + sin/cos phase)
5. Transformer-based decoder (optional, simplified version available)

This implementation adapts NeuroRVQ for single-channel patch-based tokenization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from einops import rearrange
import math


# =============================================================================
# Utility Functions
# =============================================================================

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


def kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
    use_cosine_sim: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def inverse_fft_cos_sin(
    fft_amp: torch.Tensor,
    fft_sin_pha: torch.Tensor,
    fft_cos_pha: torch.Tensor
) -> torch.Tensor:
    """
    Inverse FFT using amplitude and sin/cos phase.
    
    Args:
        fft_amp: [B, T] amplitude spectrum
        fft_sin_pha: [B, T] sin of phase
        fft_cos_pha: [B, T] cos of phase
    
    Returns:
        signal: [B, T] reconstructed time-domain signal
    """
    real = fft_amp * fft_cos_pha
    imag = fft_amp * fft_sin_pha
    fft_y = torch.complex(real, imag)
    y = torch.fft.ifft(fft_y)
    return y.real


# =============================================================================
# NormEMAVectorQuantizer - Following LaBraM/NeuroRVQ
# =============================================================================

class NormEMAVectorQuantizer(nn.Module):
    """
    L2-normalized EMA Vector Quantizer from NeuroRVQ/LaBraM.
    
    Key features:
    1. L2 normalization of both input and codebook
    2. EMA-based codebook updates (no gradient through codebook)
    3. Optional k-means initialization
    """
    
    def __init__(
        self,
        n_embed: int = 8192,
        embedding_dim: int = 64,
        beta: float = 1.0,
        decay: float = 0.99,
        eps: float = 1e-5,
        kmeans_init: bool = True,
    ):
        super().__init__()
        self.num_tokens = n_embed
        self.codebook_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        
        # Codebook initialization
        if kmeans_init:
            weight = torch.zeros(n_embed, embedding_dim)
        else:
            weight = torch.randn(n_embed, embedding_dim)
            weight = l2norm(weight)
        
        self.register_buffer('weight', weight)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', weight.clone())
        self.register_buffer('initted', torch.tensor([not kmeans_init]))
    
    def init_embed_(self, data: torch.Tensor):
        """Initialize codebook with k-means if not already done."""
        if self.initted.item():
            return
        
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size.float())
        self.embed_avg.data.copy_(embed)
        self.initted.data.fill_(True)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input features.
        
        Args:
            z: [B, C, H, W] input (following original NeuroRVQ format)
        
        Returns:
            z_q: quantized features [B, C, H, W]
            loss: commitment loss
            indices: codebook indices [B*H*W]
        """
        # Reshape: [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        z = rearrange(z, 'b c h w -> b h w c')
        shape = z.shape
        z_flat = z.reshape(-1, self.codebook_dim)
        
        # L2 normalize
        z_flat = l2norm(z_flat)
        
        # Initialize codebook on first forward
        self.init_embed_(z_flat)
        
        # Compute distances (for L2-normalized: d^2 = 2 - 2*cos)
        d = z_flat.pow(2).sum(dim=1, keepdim=True) + \
            self.weight.pow(2).sum(dim=1) - \
            2 * torch.einsum('bd,nd->bn', z_flat, self.weight)
        
        # Get nearest codes
        indices = torch.argmin(d, dim=1)
        z_q = F.embedding(indices, self.weight)
        
        # EMA updates during training
        if self.training:
            # Use scatter_add instead of one_hot for memory efficiency
            # (one_hot creates [batch, num_tokens] tensor which is huge for large codebooks)
            bins = torch.zeros(self.num_tokens, device=z.device, dtype=z_flat.dtype)
            bins.scatter_add_(0, indices, torch.ones_like(indices, dtype=z_flat.dtype))
            
            # Update cluster sizes
            ema_inplace(self.cluster_size, bins, self.decay)
            
            # Update embeddings using scatter_add
            zero_mask = (bins == 0)
            bins_clamped = bins.masked_fill(zero_mask, 1.)
            
            # Compute sum of embeddings per cluster
            embed_sum = torch.zeros_like(self.weight)
            embed_sum.scatter_add_(0, indices.unsqueeze(1).expand(-1, self.codebook_dim), z_flat)
            
            embed_normalized = embed_sum / bins_clamped.unsqueeze(1)
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(
                zero_mask[..., None],
                self.weight,
                embed_normalized
            )
            
            # Normalized EMA update
            self.weight.data.mul_(self.decay).add_(embed_normalized, alpha=(1 - self.decay))
            self.weight.data.copy_(l2norm(self.weight.data))
        
        # Commitment loss
        loss = self.beta * F.mse_loss(z_q.detach(), z_flat)
        
        # Straight-through estimator
        z_q = z_flat + (z_q - z_flat).detach()
        
        # Reshape back: [B*H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        z_q = z_q.view(shape)
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        
        return z_q, loss, indices
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode without EMA updates."""
        z = rearrange(z, 'b c h w -> b h w c')
        z_flat = z.reshape(-1, self.codebook_dim)
        z_flat = l2norm(z_flat)
        
        d = z_flat.pow(2).sum(dim=1, keepdim=True) + \
            self.weight.pow(2).sum(dim=1) - \
            2 * torch.einsum('bd,nd->bn', z_flat, self.weight)
        
        indices = torch.argmin(d, dim=1)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to quantized vectors."""
        return F.embedding(indices, self.weight)


# =============================================================================
# Residual Vector Quantization - Following Algorithm 1 in SoundStream paper
# =============================================================================

class ResidualVectorQuantization(nn.Module):
    """
    Residual Vector Quantization (RVQ) as used in NeuroRVQ.
    
    Uses multiple quantizer layers that sequentially quantize
    the residual from the previous layer.
    """
    
    def __init__(
        self,
        num_quantizers: int = 8,
        n_embed: int = 8192,
        embedding_dim: int = 64,
        beta: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.n_embed = n_embed
        self.embedding_dim = embedding_dim
        
        self.layers = nn.ModuleList([
            NormEMAVectorQuantizer(
                n_embed=n_embed,
                embedding_dim=embedding_dim,
                beta=beta,
                decay=decay,
                kmeans_init=kmeans_init,
            )
            for _ in range(num_quantizers)
        ])
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        """
        Args:
            x: [B, C, H, W] input features
        
        Returns:
            quantized_out: sum of all quantized layers
            indices: [num_quantizers, B*H*W] indices for each layer
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
            
            # Update residual
            new_residual = residual - quantized
            
            # Additional residual loss (as in NeuroRVQ)
            loss = loss + 0.4 * F.mse_loss(quantized, residual.detach())
            
            residual = new_residual
            quantized_out = quantized_out + quantized
            
            all_indices.append(indices)
            all_losses.append(loss)
            
            # Track codebook usage
            unique_codes = torch.unique(indices)
            usage_ratio = unique_codes.numel() / self.n_embed
            usage_ratios.append(float(usage_ratio))
        
        out_indices = torch.stack(all_indices, dim=0)
        total_loss = torch.stack(all_losses).mean()
        
        return quantized_out, out_indices, total_loss, usage_ratios
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to RVQ indices."""
        residual = x
        all_indices = []
        
        for layer in self.layers:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            # Reshape quantized back to spatial format
            B, C, H, W = residual.shape
            quantized = quantized.view(B, H, W, C)
            quantized = rearrange(quantized, 'b h w c -> b c h w')
            residual = residual - quantized
            all_indices.append(indices)
        
        return torch.stack(all_indices, dim=0)
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode RVQ indices.
        
        Args:
            indices: [num_quantizers, B*H*W] or list of indices
        
        Returns:
            quantized_out: sum of decoded vectors
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


# =============================================================================
# Multi-Scale Temporal Encoder - Inception-style with 4 branches
# =============================================================================

class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoder from NeuroRVQ.
    
    Inception-style architecture with 4 parallel branches using
    different kernel sizes to capture different frequency components.
    
    At 200Hz sampling rate:
    - Branch 1 (k=21): captures ~10Hz and below (delta, theta, alpha)
    - Branch 2 (k=15): captures ~13Hz and below
    - Branch 3 (k=9): captures ~22Hz and below (beta)
    - Branch 4 (k=5): captures ~40Hz and below (gamma)
    
    Each branch has 2 conv layers with pooling (total 8x downsampling).
    """
    
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 8,
        adaptive_pooling: bool = False,
        target_length: int = 25,
    ):
        super().__init__()
        self.out_chans = out_chans
        self.num_branches = 4
        self.adaptive_pooling = adaptive_pooling
        self.target_length = target_length
        
        # ========== Branch 1: >10 Hz (large kernel for low freq) ==========
        self.conv1_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 21), padding=(0, 10))
        self.norm1_1 = nn.GroupNorm(4, out_chans)
        self.pool1_1 = nn.AvgPool2d(kernel_size=(1, 2))
        
        self.conv1_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm1_2 = nn.GroupNorm(4, out_chans)
        self.pool1_2 = nn.AvgPool2d(kernel_size=(1, 4))
        
        # ========== Branch 2: >13 Hz ==========
        self.conv2_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), padding=(0, 7))
        self.norm2_1 = nn.GroupNorm(4, out_chans)
        self.pool2_1 = nn.AvgPool2d(kernel_size=(1, 2))
        
        self.conv2_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 7), padding=(0, 3))
        self.norm2_2 = nn.GroupNorm(4, out_chans)
        self.pool2_2 = nn.AvgPool2d(kernel_size=(1, 4))
        
        # ========== Branch 3: >20 Hz ==========
        self.conv3_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm3_1 = nn.GroupNorm(4, out_chans)
        self.pool3_1 = nn.AvgPool2d(kernel_size=(1, 2))
        
        self.conv3_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm3_2 = nn.GroupNorm(4, out_chans)
        self.pool3_2 = nn.AvgPool2d(kernel_size=(1, 4))
        
        # ========== Branch 4: >40 Hz (small kernel for high freq) ==========
        self.conv4_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm4_1 = nn.GroupNorm(4, out_chans)
        self.pool4_1 = nn.AvgPool2d(kernel_size=(1, 2))
        
        self.conv4_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm4_2 = nn.GroupNorm(4, out_chans)
        self.pool4_2 = nn.AvgPool2d(kernel_size=(1, 4))
        
        self.gelu = nn.GELU()
        
        # Optional adaptive pooling for variable-length inputs
        if adaptive_pooling:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, target_length))
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, A, T] where N=num_channels, A=num_patches, T=patch_size
               For single-patch input: [B, 1, 1, T]
        
        Returns:
            x1, x2, x3, x4: branch outputs, each [B, N*A, T//8 * out_chans]
        """
        # Reshape: [B, N, A, T] -> [B, N*A, T] -> [B, 1, N*A, T]
        B, N, A, T = x.shape
        x = rearrange(x, 'B N A T -> B (N A) T')
        x = x.unsqueeze(1)  # [B, 1, N*A, T]
        
        # Branch 1
        x1 = self.pool1_1(self.gelu(self.norm1_1(self.conv1_1(x))))
        x1 = self.pool1_2(self.gelu(self.norm1_2(self.conv1_2(x1))))
        
        # Branch 2
        x2 = self.pool2_1(self.gelu(self.norm2_1(self.conv2_1(x))))
        x2 = self.pool2_2(self.gelu(self.norm2_2(self.conv2_2(x2))))
        
        # Branch 3
        x3 = self.pool3_1(self.gelu(self.norm3_1(self.conv3_1(x))))
        x3 = self.pool3_2(self.gelu(self.norm3_2(self.conv3_2(x3))))
        
        # Branch 4
        x4 = self.pool4_1(self.gelu(self.norm4_1(self.conv4_1(x))))
        x4 = self.pool4_2(self.gelu(self.norm4_2(self.conv4_2(x4))))
        
        if self.adaptive_pooling:
            x1 = self.adaptive_pool(x1)
            x2 = self.adaptive_pool(x2)
            x3 = self.adaptive_pool(x3)
            x4 = self.adaptive_pool(x4)
        
        # Rearrange: [B, C, N*A, T'] -> [B, N*A, T'*C]
        x1 = rearrange(x1, 'B C NA T -> B NA (T C)')
        x2 = rearrange(x2, 'B C NA T -> B NA (T C)')
        x3 = rearrange(x3, 'B C NA T -> B NA (T C)')
        x4 = rearrange(x4, 'B C NA T -> B NA (T C)')
        
        return x1, x2, x3, x4


# =============================================================================
# Encoding/Decoding Task Layers
# =============================================================================

class EncodingTaskLayer(nn.Module):
    """Projects encoder output to codebook dimension."""
    
    def __init__(self, input_dim: int, code_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, code_dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecodingTaskLayer(nn.Module):
    """Decoding heads for amplitude and phase reconstruction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # Amplitude head (no activation constraint)
        self.amplitude_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Phase heads with Tanh for [-1, 1] range
        self.sin_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        
        self.cos_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] concatenated branch features
        
        Returns:
            amplitude: [B, N, T]
            sin_phase: [B, N, T]
            cos_phase: [B, N, T]
        """
        amplitude = self.amplitude_head(x)
        sin_phase = self.sin_head(x)
        cos_phase = self.cos_head(x)
        return amplitude, sin_phase, cos_phase


# =============================================================================
# NeuroRVQ Tokenizer - Full Implementation
# =============================================================================

class NeuroRVQTokenizer(nn.Module):
    """
    NeuroRVQ Tokenizer following the original paper design.
    
    Architecture:
    1. Multi-scale temporal encoder (4 branches, Inception-style)
    2. 4 separate encoding task layers (project to code_dim)
    3. 4 separate RVQ quantizers (one per branch)
    4. Decoding task layer (predicts amplitude + sin/cos phase)
    5. Frequency domain reconstruction with iFFT
    
    For single-channel patches (e.g., one EEG channel segment).
    """
    
    def __init__(
        self,
        patch_size: int = 200,
        n_embed: int = 8192,
        code_dim: int = 64,
        num_quantizers: int = 8,
        out_chans: int = 8,
        beta: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.n_embed = n_embed
        self.code_dim = code_dim
        self.num_quantizers = num_quantizers
        self.out_chans = out_chans
        
        # Multi-scale temporal encoder
        self.encoder = MultiScaleTemporalEncoder(
            in_chans=1,
            out_chans=out_chans,
        )
        
        # Calculate encoder output dimension
        # After 8x pooling: T' = patch_size // 8
        # Output dim per branch = T' * out_chans
        pooled_len = patch_size // 8
        encoder_out_dim = pooled_len * out_chans
        
        # 4 encoding task layers (one per branch)
        self.encode_task_layer_1 = EncodingTaskLayer(encoder_out_dim, code_dim)
        self.encode_task_layer_2 = EncodingTaskLayer(encoder_out_dim, code_dim)
        self.encode_task_layer_3 = EncodingTaskLayer(encoder_out_dim, code_dim)
        self.encode_task_layer_4 = EncodingTaskLayer(encoder_out_dim, code_dim)
        
        # 4 RVQ quantizers (one per branch)
        self.quantize_1 = ResidualVectorQuantization(
            num_quantizers=num_quantizers,
            n_embed=n_embed,
            embedding_dim=code_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        self.quantize_2 = ResidualVectorQuantization(
            num_quantizers=num_quantizers,
            n_embed=n_embed,
            embedding_dim=code_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        self.quantize_3 = ResidualVectorQuantization(
            num_quantizers=num_quantizers,
            n_embed=n_embed,
            embedding_dim=code_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        self.quantize_4 = ResidualVectorQuantization(
            num_quantizers=num_quantizers,
            n_embed=n_embed,
            embedding_dim=code_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        
        # Decoding task layer
        # Input: 4 branches concatenated = 4 * code_dim
        self.decode_task_layer = DecodingTaskLayer(
            input_dim=4 * code_dim,
            hidden_dim=code_dim * 2,
            output_dim=patch_size,
        )
        
        # MSE loss function
        self.loss_fn = F.mse_loss
    
    def std_norm(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standardize tensor."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std, mean, std
    
    def encode(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[List[float]]]:
        """
        Encode input patches.
        
        Args:
            x: [B, T] single patch per sample
        
        Returns:
            quantized: list of 4 quantized tensors [B, C, 1, 1]
            indices: list of 4 index tensors [num_quantizers, B]
            loss: combined VQ loss
            usage_ratios: list of usage ratios per quantizer
        """
        B, T = x.shape
        
        # Reshape for encoder: [B, T] -> [B, 1, 1, T]
        x_in = x.view(B, 1, 1, T)
        
        # Multi-scale encoding
        feat_1, feat_2, feat_3, feat_4 = self.encoder(x_in)
        # Each: [B, 1, D] where D = T//8 * out_chans
        
        # Squeeze channel dimension: [B, 1, D] -> [B, D]
        feat_1 = feat_1.squeeze(1)
        feat_2 = feat_2.squeeze(1)
        feat_3 = feat_3.squeeze(1)
        feat_4 = feat_4.squeeze(1)
        
        # Project to code dimension
        z_1 = self.encode_task_layer_1(feat_1)  # [B, code_dim]
        z_2 = self.encode_task_layer_2(feat_2)
        z_3 = self.encode_task_layer_3(feat_3)
        z_4 = self.encode_task_layer_4(feat_4)
        
        # Reshape for RVQ: [B, code_dim] -> [B, code_dim, 1, 1]
        z_1 = z_1.unsqueeze(-1).unsqueeze(-1)
        z_2 = z_2.unsqueeze(-1).unsqueeze(-1)
        z_3 = z_3.unsqueeze(-1).unsqueeze(-1)
        z_4 = z_4.unsqueeze(-1).unsqueeze(-1)
        
        # Quantize each branch
        q_1, idx_1, loss_1, usage_1 = self.quantize_1(z_1)
        q_2, idx_2, loss_2, usage_2 = self.quantize_2(z_2)
        q_3, idx_3, loss_3, usage_3 = self.quantize_3(z_3)
        q_4, idx_4, loss_4, usage_4 = self.quantize_4(z_4)
        
        # Combine losses
        total_loss = loss_1 + loss_2 + loss_3 + loss_4
        
        return (
            [q_1, q_2, q_3, q_4],
            [idx_1, idx_2, idx_3, idx_4],
            total_loss,
            [usage_1, usage_2, usage_3, usage_4],
        )
    
    def decode(
        self, quantized: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode quantized features to amplitude and phase.
        
        Args:
            quantized: list of 4 tensors [B, code_dim, 1, 1]
        
        Returns:
            amplitude: [B, T]
            sin_phase: [B, T]
            cos_phase: [B, T]
        """
        # Reshape: [B, C, 1, 1] -> [B, C]
        q_1 = quantized[0].squeeze(-1).squeeze(-1)
        q_2 = quantized[1].squeeze(-1).squeeze(-1)
        q_3 = quantized[2].squeeze(-1).squeeze(-1)
        q_4 = quantized[3].squeeze(-1).squeeze(-1)
        
        # Concatenate all branches
        combined = torch.cat([q_1, q_2, q_3, q_4], dim=-1)  # [B, 4*code_dim]
        
        # Add sequence dimension for decoder: [B, 4*code_dim] -> [B, 1, 4*code_dim]
        combined = combined.unsqueeze(1)
        
        # Decode
        amplitude, sin_phase, cos_phase = self.decode_task_layer(combined)
        
        # Remove sequence dimension: [B, 1, T] -> [B, T]
        amplitude = amplitude.squeeze(1)
        sin_phase = sin_phase.squeeze(1)
        cos_phase = cos_phase.squeeze(1)
        
        return amplitude, sin_phase, cos_phase
    
    def calculate_phase_loss(
        self,
        pred_sin: torch.Tensor,
        target_sin: torch.Tensor,
        pred_cos: torch.Tensor,
        target_cos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate phase loss using cosine similarity and magnitude constraint.
        """
        pred = torch.stack([pred_cos, pred_sin], dim=-1)
        target = torch.stack([target_cos, target_sin], dim=-1)
        
        # Cosine similarity for direction
        cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()
        
        # Magnitude constraint (sin^2 + cos^2 = 1)
        mag_loss = ((pred_sin ** 2 + pred_cos ** 2 - 1) ** 2).mean()
        
        phase_loss = (1.0 - cos_sim) + 0.1 * mag_loss
        return phase_loss
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: [B, T] input patches
        
        Returns:
            dict with all outputs and losses
        """
        B, T = x.shape
        
        # Compute FFT of input
        x_fft = torch.fft.fft(x, dim=-1)
        
        # Get amplitude (log scale for stability)
        target_amp = torch.abs(x_fft)
        target_amp = torch.log1p(target_amp)
        target_amp_norm, amp_mean, amp_std = self.std_norm(target_amp)
        
        # Get phase as sin/cos
        target_angle = torch.angle(x_fft)
        target_sin = torch.sin(target_angle)
        target_cos = torch.cos(target_angle)
        
        # Encode and quantize
        quantized, indices, vq_loss, usage_ratios = self.encode(x)
        
        # Decode
        pred_amp, pred_sin, pred_cos = self.decode(quantized)
        
        # Losses
        amp_loss = self.loss_fn(pred_amp, target_amp_norm)
        phase_loss = self.calculate_phase_loss(pred_sin, target_sin, pred_cos, target_cos)
        
        # Reconstruct time-domain signal
        # Unstandardize amplitude
        pred_amp_unstd = pred_amp * amp_std + amp_mean
        pred_amp_linear = torch.expm1(pred_amp_unstd)
        
        # iFFT reconstruction
        reconstructed = inverse_fft_cos_sin(pred_amp_linear, pred_sin, pred_cos)
        
        # Time domain loss (on standardized signals)
        x_norm, _, _ = self.std_norm(x)
        rec_norm, _, _ = self.std_norm(reconstructed)
        time_loss = self.loss_fn(rec_norm, x_norm)
        
        # Total loss (following original: vq + amp + phase + time)
        rec_loss = amp_loss + phase_loss + time_loss
        total_loss = vq_loss + rec_loss
        
        # Average utilization across all branches and quantizers
        all_usage = []
        for branch_usage in usage_ratios:
            all_usage.extend(branch_usage)
        avg_utilization = sum(all_usage) / len(all_usage) if all_usage else 0.0
        
        # Flatten indices to single tensor for compatibility with visualization
        # indices is list of 4 tensors, each [num_quantizers, B]
        # -> stack to [4, num_quantizers, B] -> transpose to [B, 4*num_quantizers]
        indices_stacked = torch.stack([idx.transpose(0, 1) for idx in indices], dim=1)
        # indices_stacked: [B, 4, num_quantizers]
        indices_flat = indices_stacked.reshape(B, -1)  # [B, 4*num_quantizers]
        
        return {
            # Outputs
            'reconstructed': reconstructed,
            'amplitude': pred_amp,
            'sin_phase': pred_sin,
            'cos_phase': pred_cos,
            'quantized': quantized,
            'tokens': indices_flat,         # [B, 4*num_quantizers] for visualization
            'indices_per_branch': indices,  # Original list format for detailed analysis
            # Losses
            'loss': total_loss,
            'rec_loss': rec_loss,
            'vq_loss': vq_loss,
            'amp_loss': amp_loss,
            'phase_loss': phase_loss,
            'time_loss': time_loss,
            # Stats
            'utilization': avg_utilization,
            'usage_ratios': usage_ratios,
        }
    
    def tokenize(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert patches to tokens.
        
        Args:
            x: [B, T] patches
        
        Returns:
            indices: list of 4 tensors, each [num_quantizers, B]
        """
        with torch.no_grad():
            _, indices, _, _ = self.encode(x)
        return indices
    
    def detokenize(self, indices: List[torch.Tensor], B: int) -> torch.Tensor:
        """
        Convert tokens back to patches.
        
        Args:
            indices: list of 4 index tensors [num_quantizers, B]
            B: batch size
        
        Returns:
            reconstructed: [B, T] time domain signals
        """
        with torch.no_grad():
            # Decode each quantizer
            q_1 = self.quantize_1.decode(indices[0])  # [B, code_dim]
            q_2 = self.quantize_2.decode(indices[1])
            q_3 = self.quantize_3.decode(indices[2])
            q_4 = self.quantize_4.decode(indices[3])
            
            # Reshape: [B, code_dim] -> [B, code_dim, 1, 1]
            quantized = [
                q_1.view(B, self.code_dim, 1, 1),
                q_2.view(B, self.code_dim, 1, 1),
                q_3.view(B, self.code_dim, 1, 1),
                q_4.view(B, self.code_dim, 1, 1),
            ]
            
            # Decode to amplitude and phase
            amplitude, sin_phase, cos_phase = self.decode(quantized)
            
            # Convert to linear amplitude
            amp_linear = torch.expm1(amplitude)
            
            # iFFT reconstruction
            reconstructed = inverse_fft_cos_sin(amp_linear, sin_phase, cos_phase)
        
        return reconstructed
    
    def get_codebook_usage(self) -> Dict[str, float]:
        """Get codebook utilization stats for each branch and RVQ layer."""
        usage = {}
        quantizers = [
            ('branch1', self.quantize_1),
            ('branch2', self.quantize_2),
            ('branch3', self.quantize_3),
            ('branch4', self.quantize_4),
        ]
        
        for branch_name, quant in quantizers:
            for i, layer in enumerate(quant.layers):
                active = (layer.cluster_size > 0).sum().item()
                usage[f'{branch_name}_layer{i}_active'] = active
                usage[f'{branch_name}_layer{i}_utilization'] = active / layer.num_tokens
        
        return usage
    
    def get_codebook_size(self) -> int:
        """
        Return the codebook size (n_embed).
        
        For NeuroRVQ, all 4 branches × 8 RVQ layers share the same codebook size.
        """
        return self.n_embed
    
    def get_codebook_embeddings(self) -> torch.Tensor:
        """
        Return codebook embeddings from the first branch, first RVQ layer.
        
        For visualization purposes, we use branch1's first layer as representative
        since all layers share the same codebook dimension.
        
        Returns:
            embeddings: [n_embed, code_dim] tensor
        """
        return self.quantize_1.layers[0].weight.detach()

# =============================================================================
# fNIRS-adapted NeuroRVQ (smaller kernels for lower sampling rate)
# =============================================================================

class MultiScaleTemporalEncoderFNIRS(nn.Module):
    """
    Multi-scale temporal encoder adapted for fNIRS (10Hz).
    
    Uses smaller kernel sizes appropriate for the lower sampling rate.
    """
    
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 8,
    ):
        super().__init__()
        self.out_chans = out_chans
        self.num_branches = 4
        
        # Branch 1: slow dynamics (k=7)
        self.conv1_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 7), padding=(0, 3))
        self.norm1_1 = nn.GroupNorm(4, out_chans)
        self.pool1_1 = nn.AvgPool2d(kernel_size=(1, 2))
        
        self.conv1_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm1_2 = nn.GroupNorm(4, out_chans)
        self.pool1_2 = nn.AvgPool2d(kernel_size=(1, 2))
        
        # Branch 2: medium-slow (k=5)
        self.conv2_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm2_1 = nn.GroupNorm(4, out_chans)
        self.pool2_1 = nn.AvgPool2d(kernel_size=(1, 2))
        
        self.conv2_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2_2 = nn.GroupNorm(4, out_chans)
        self.pool2_2 = nn.AvgPool2d(kernel_size=(1, 2))
        
        # Branch 3: medium-fast (k=3)
        self.conv3_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3_1 = nn.GroupNorm(4, out_chans)
        self.pool3_1 = nn.AvgPool2d(kernel_size=(1, 2))
        
        self.conv3_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3_2 = nn.GroupNorm(4, out_chans)
        self.pool3_2 = nn.AvgPool2d(kernel_size=(1, 2))
        
        # Branch 4: fast dynamics (k=2)
        self.conv4_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 2), padding=(0, 0))
        self.norm4_1 = nn.GroupNorm(4, out_chans)
        self.pool4_1 = nn.AvgPool2d(kernel_size=(1, 2))
        
        self.conv4_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 2), padding=(0, 0))
        self.norm4_2 = nn.GroupNorm(4, out_chans)
        self.pool4_2 = nn.AvgPool2d(kernel_size=(1, 2))
        
        self.gelu = nn.GELU()
        
        # Adaptive pooling to handle variable output sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))  # Fixed output length
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, A, T] where T is typically 40 for fNIRS @ 10Hz
        """
        B, N, A, T = x.shape
        x = rearrange(x, 'B N A T -> B (N A) T')
        x = x.unsqueeze(1)
        
        # Branch 1
        x1 = self.pool1_1(self.gelu(self.norm1_1(self.conv1_1(x))))
        x1 = self.pool1_2(self.gelu(self.norm1_2(self.conv1_2(x1))))
        x1 = self.adaptive_pool(x1)
        
        # Branch 2
        x2 = self.pool2_1(self.gelu(self.norm2_1(self.conv2_1(x))))
        x2 = self.pool2_2(self.gelu(self.norm2_2(self.conv2_2(x2))))
        x2 = self.adaptive_pool(x2)
        
        # Branch 3
        x3 = self.pool3_1(self.gelu(self.norm3_1(self.conv3_1(x))))
        x3 = self.pool3_2(self.gelu(self.norm3_2(self.conv3_2(x3))))
        x3 = self.adaptive_pool(x3)
        
        # Branch 4
        x4 = self.pool4_1(self.gelu(self.norm4_1(self.conv4_1(x))))
        x4 = self.pool4_2(self.gelu(self.norm4_2(self.conv4_2(x4))))
        x4 = self.adaptive_pool(x4)
        
        # Rearrange
        x1 = rearrange(x1, 'B C NA T -> B NA (T C)')
        x2 = rearrange(x2, 'B C NA T -> B NA (T C)')
        x3 = rearrange(x3, 'B C NA T -> B NA (T C)')
        x4 = rearrange(x4, 'B C NA T -> B NA (T C)')
        
        return x1, x2, x3, x4


class NeuroRVQTokenizerFNIRS(NeuroRVQTokenizer):
    """
    NeuroRVQ adapted for fNIRS signals (10Hz sampling rate).
    """
    
    def __init__(
        self,
        patch_size: int = 40,  # 4s @ 10Hz
        n_embed: int = 4096,   # Smaller codebook for simpler signals
        code_dim: int = 32,    # Smaller latent
        num_quantizers: int = 4,  # Fewer quantizers
        out_chans: int = 8,
        beta: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
    ):
        # Don't call parent __init__, we override everything
        nn.Module.__init__(self)
        
        self.patch_size = patch_size
        self.n_embed = n_embed
        self.code_dim = code_dim
        self.num_quantizers = num_quantizers
        self.out_chans = out_chans
        
        # fNIRS-adapted encoder
        self.encoder = MultiScaleTemporalEncoderFNIRS(
            in_chans=1,
            out_chans=out_chans,
        )
        
        # Encoder output dim: adaptive pooling gives 4 samples, times out_chans
        encoder_out_dim = 4 * out_chans
        
        # Encoding task layers
        self.encode_task_layer_1 = EncodingTaskLayer(encoder_out_dim, code_dim)
        self.encode_task_layer_2 = EncodingTaskLayer(encoder_out_dim, code_dim)
        self.encode_task_layer_3 = EncodingTaskLayer(encoder_out_dim, code_dim)
        self.encode_task_layer_4 = EncodingTaskLayer(encoder_out_dim, code_dim)
        
        # RVQ quantizers
        self.quantize_1 = ResidualVectorQuantization(
            num_quantizers=num_quantizers,
            n_embed=n_embed,
            embedding_dim=code_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        self.quantize_2 = ResidualVectorQuantization(
            num_quantizers=num_quantizers,
            n_embed=n_embed,
            embedding_dim=code_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        self.quantize_3 = ResidualVectorQuantization(
            num_quantizers=num_quantizers,
            n_embed=n_embed,
            embedding_dim=code_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        self.quantize_4 = ResidualVectorQuantization(
            num_quantizers=num_quantizers,
            n_embed=n_embed,
            embedding_dim=code_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        
        # Decoding task layer
        self.decode_task_layer = DecodingTaskLayer(
            input_dim=4 * code_dim,
            hidden_dim=code_dim * 2,
            output_dim=patch_size,
        )
        
        self.loss_fn = F.mse_loss
