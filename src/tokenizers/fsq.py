"""
Finite Scalar Quantization (FSQ) Tokenizer.

Reference: "Language Modeling with Finite Scalar Quantization" (Google DeepMind, 2023)

FSQ advantages:
- No codebook collapse (implicit codebook from level combinations)
- Simpler gradient flow (straight-through estimator)
- No commitment loss needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np

from .base import BaseTokenizer, Conv1dEncoder, Conv1dDecoder


class FSQuantizer(nn.Module):
    """
    Finite Scalar Quantization layer.
    
    Each dimension is quantized to a fixed number of levels.
    Effective codebook size = prod(levels).
    
    Example:
        levels = [8, 8, 8, 8] -> 8^4 = 4096 effective codes
        levels = [8, 5, 5, 5] -> 8*5*5*5 = 1000 effective codes
    """
    
    def __init__(self, levels: List[int]):
        """
        Args:
            levels: List of quantization levels per dimension.
                    E.g., [8, 8, 8, 8] means 4 dims with 8 levels each.
        """
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.codebook_size = int(np.prod(levels))
        
        # Register levels as buffer for device compatibility
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))
        
        # Precompute basis for index calculation
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0)
        self.register_buffer('_basis', basis)
    
    def _scale_and_shift(self, z: torch.Tensor) -> torch.Tensor:
        """Scale z to [-1, 1] range for each dimension based on levels."""
        # Assume z is in some reasonable range, normalize per-dim
        # z: [..., D]
        half_levels = (self._levels - 1) / 2
        return z / (half_levels + 1e-8)
    
    def _round_ste(self, z: torch.Tensor) -> torch.Tensor:
        """Round with straight-through estimator."""
        return z + (z.round() - z).detach()
    
    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound values to valid quantization range."""
        half_levels = (self._levels - 1) / 2
        return torch.clamp(z, -half_levels, half_levels)
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latent to discrete levels.
        
        Args:
            z: Continuous latent [..., D]
            
        Returns:
            z_q: Quantized latent [..., D] (same shape, quantized values)
            indices: Flattened indices [...] (single index per position)
        """
        # Scale to appropriate range
        half_levels = (self._levels - 1) / 2
        
        # Bound and round
        z_bounded = self._bound(z * half_levels)
        z_q = self._round_ste(z_bounded)
        
        # Compute flattened indices
        # Shift to [0, L-1] range
        z_shifted = z_q + half_levels
        z_shifted = z_shifted.long()
        
        # Flatten to single index
        indices = (z_shifted * self._basis.long()).sum(dim=-1)
        
        # Scale back to [-1, 1]-ish range for downstream
        z_q = z_q / (half_levels + 1e-8)
        
        return z_q, indices
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with quantization.
        
        Args:
            z: Continuous latent [..., D]
            
        Returns:
            z_q: Quantized latent
            indices: Token indices
            info: Dict with perplexity and other stats
        """
        z_q, indices = self.quantize(z)
        
        # Compute perplexity (measure of codebook usage)
        # Count unique indices in batch
        flat_indices = indices.flatten()
        usage = torch.bincount(flat_indices, minlength=self.codebook_size).float()
        usage = usage / (usage.sum() + 1e-8)
        
        # Entropy and perplexity
        entropy = -(usage * torch.log(usage + 1e-10)).sum()
        perplexity = torch.exp(entropy)
        
        # Code utilization
        code_utilization = (usage > 0).float().mean()
        
        info = {
            'perplexity': perplexity,
            'code_utilization': code_utilization,
            'commitment_loss': torch.tensor(0.0, device=z.device),  # FSQ doesn't need this
        }
        
        return z_q, indices, info


class FSQTokenizer(BaseTokenizer):
    """
    Complete FSQ-based tokenizer with encoder and decoder.
    """
    
    def __init__(
        self,
        seq_length: int = 256,
        input_channels: int = 1,
        levels: List[int] = [8, 8, 8, 8],
        encoder_dims: List[int] = [64, 128, 256],
        encoder_kernel: int = 7,
        encoder_stride: int = 2,
        **kwargs
    ):
        latent_dim = len(levels)
        super().__init__(input_dim=input_channels, latent_dim=latent_dim)
        
        self.seq_length = seq_length
        self.levels = levels
        
        # Calculate compressed sequence length
        num_layers = len(encoder_dims)
        self.compressed_length = seq_length // (encoder_stride ** num_layers)
        
        # Encoder
        self.encoder = Conv1dEncoder(
            input_dim=input_channels,
            hidden_dims=encoder_dims,
            kernel_size=encoder_kernel,
            stride=encoder_stride,
            output_dim=latent_dim
        )
        
        # FSQ quantizer
        self.quantizer = FSQuantizer(levels)
        
        # Decoder
        self.decoder = Conv1dDecoder(
            input_dim=latent_dim,
            hidden_dims=list(reversed(encoder_dims)),
            kernel_size=encoder_kernel,
            stride=encoder_stride,
            output_dim=input_channels,
            output_length=seq_length
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to continuous latent."""
        return self.encoder(x)
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Quantize latent with FSQ."""
        return self.quantizer(z)
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to signal."""
        return self.decoder(z_q)
    
    def get_codebook_size(self) -> int:
        return self.quantizer.codebook_size
    
    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """
        For FSQ, we don't have explicit embeddings.
        Convert indices back to quantized values.
        """
        # Decompose flat index to per-dimension indices
        result = []
        remainder = indices.clone()
        
        for level, basis in zip(self.levels, self.quantizer._basis):
            dim_idx = (remainder // basis) % level
            result.append(dim_idx)
            remainder = remainder % basis
        
        # Stack and convert to float
        z_q = torch.stack(result, dim=-1).float()
        
        # Scale to [-1, 1] range
        half_levels = (self.quantizer._levels - 1) / 2
        z_q = (z_q - half_levels) / (half_levels + 1e-8)
        
        return z_q
    
    def get_codebook_embeddings(self) -> Optional[torch.Tensor]:
        """
        Generate all possible codebook embeddings for FSQ.
        
        Returns:
            embeddings: [K, D] tensor where K = codebook_size, D = len(levels)
            Returns None if codebook is too large (>16384)
        """
        if self.get_codebook_size() > 16384:
            return None  # Too large to enumerate
        
        # Generate all possible indices
        indices = torch.arange(self.get_codebook_size(), device=self.quantizer._levels.device)
        return self.get_embedding(indices)


if __name__ == "__main__":
    # Test FSQ tokenizer
    print("Testing FSQ Tokenizer...")
    
    tokenizer = FSQTokenizer(
        seq_length=256,
        levels=[8, 8, 8, 8],
        encoder_dims=[32, 64, 128]
    )
    
    # Random input
    x = torch.randn(4, 256)  # [B, T]
    
    # Forward pass
    outputs = tokenizer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {outputs['x_rec'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    print(f"Quantized shape: {outputs['z_q'].shape}")
    print(f"Indices shape: {outputs['indices'].shape}")
    print(f"Codebook size: {tokenizer.get_codebook_size()}")
    print(f"Perplexity: {outputs['perplexity'].item():.2f}")
    print(f"Code utilization: {outputs['code_utilization'].item():.4f}")
    
    # Reconstruction error
    mse = F.mse_loss(outputs['x_rec'], x)
    print(f"Reconstruction MSE: {mse.item():.4f}")
    
    print("\nFSQ Tokenizer test passed!")
