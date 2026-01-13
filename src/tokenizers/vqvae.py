"""
VQ-VAE Tokenizer with EMA codebook update.

Reference: "Neural Discrete Representation Learning" (van den Oord et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base import BaseTokenizer, Conv1dEncoder, Conv1dDecoder


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with EMA codebook update.
    
    Features:
    - EMA codebook update (more stable than gradient-based)
    - Dead code detection and reinitialization
    - Commitment loss for encoder regularization
    """
    
    def __init__(
        self,
        codebook_size: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # Codebook embeddings
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        # EMA cluster size and embeddings sum
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embedding_sum', self.embedding.weight.clone())
        
        # Track code usage for dead code detection
        self.register_buffer('code_usage_count', torch.zeros(codebook_size))
    
    def _find_nearest(self, z: torch.Tensor) -> torch.Tensor:
        """
        Find nearest codebook entries for each latent vector.
        
        Args:
            z: [B, T', D]
            
        Returns:
            indices: [B, T']
        """
        # Flatten to [B*T', D]
        z_flat = z.reshape(-1, self.embedding_dim)
        
        # Compute distances
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e.T
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )
        
        # Find nearest
        indices = d.argmin(dim=1)
        
        # Reshape back
        indices = indices.view(z.shape[0], z.shape[1])
        
        return indices
    
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        """Update codebook with EMA."""
        if not self.training:
            return
        
        # Flatten
        z_flat = z.reshape(-1, self.embedding_dim)
        indices_flat = indices.reshape(-1)
        
        # One-hot encoding
        encodings = F.one_hot(indices_flat, self.codebook_size).float()
        
        # Update cluster sizes
        cluster_size = encodings.sum(dim=0)
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        
        # Update embedding sums
        embedding_sum = encodings.t() @ z_flat
        self.ema_embedding_sum.data.mul_(self.ema_decay).add_(
            embedding_sum, alpha=1 - self.ema_decay
        )
        
        # Normalize embeddings
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon) * n
        )
        
        self.embedding.weight.data.copy_(
            self.ema_embedding_sum / cluster_size.unsqueeze(1)
        )
        
        # Update usage count
        self.code_usage_count.add_(cluster_size > 0)
    
    def _reinit_dead_codes(self, z: torch.Tensor):
        """Reinitialize dead codes with samples from encoder output."""
        if not self.training:
            return
        
        # Find dead codes
        dead_codes = self.ema_cluster_size < self.threshold_ema_dead_code
        n_dead = dead_codes.sum().item()
        
        if n_dead == 0:
            return
        
        # Sample from encoder outputs
        z_flat = z.reshape(-1, self.embedding_dim)
        n_samples = min(n_dead, z_flat.shape[0])
        
        if n_samples > 0:
            indices = torch.randperm(z_flat.shape[0])[:n_samples]
            samples = z_flat[indices]
            
            # Find first n_samples dead code indices
            dead_indices = torch.where(dead_codes)[0][:n_samples]
            
            # Reinitialize
            self.embedding.weight.data[dead_indices] = samples
            self.ema_cluster_size[dead_indices] = 1.0
            self.ema_embedding_sum[dead_indices] = samples
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize latent vectors.
        
        Args:
            z: [B, T', D]
            
        Returns:
            z_q: Quantized latent [B, T', D]
            indices: Token indices [B, T']
            info: Dict with loss, perplexity, etc.
        """
        # Find nearest codebook entries
        indices = self._find_nearest(z)
        
        # Get quantized vectors
        z_q = self.embedding(indices)
        
        # Compute losses
        # Commitment loss: encourages encoder to commit to codebook
        commitment_loss = F.mse_loss(z, z_q.detach())
        
        # Codebook loss (for non-EMA training)
        codebook_loss = F.mse_loss(z.detach(), z_q)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # EMA update
        self._ema_update(z.detach(), indices)
        
        # Reinitialize dead codes occasionally
        if self.training and torch.rand(1).item() < 0.1:
            self._reinit_dead_codes(z.detach())
        
        # Compute perplexity
        flat_indices = indices.flatten()
        usage = torch.bincount(flat_indices, minlength=self.codebook_size).float()
        usage = usage / (usage.sum() + 1e-8)
        entropy = -(usage * torch.log(usage + 1e-10)).sum()
        perplexity = torch.exp(entropy)
        
        # Dead code ratio
        dead_ratio = (self.ema_cluster_size < self.threshold_ema_dead_code).float().mean()
        
        info = {
            'commitment_loss': commitment_loss * self.commitment_cost,
            'codebook_loss': codebook_loss,
            'perplexity': perplexity,
            'dead_ratio': dead_ratio,
            'code_utilization': (usage > 0).float().mean(),
        }
        
        return z_q, indices, info


class VQVAETokenizer(BaseTokenizer):
    """
    Complete VQ-VAE tokenizer with encoder and decoder.
    """
    
    def __init__(
        self,
        seq_length: int = 256,
        input_channels: int = 1,
        codebook_size: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        encoder_dims: list = [64, 128, 256],
        encoder_kernel: int = 7,
        encoder_stride: int = 2,
        **kwargs
    ):
        super().__init__(input_dim=input_channels, latent_dim=embedding_dim)
        
        self.seq_length = seq_length
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        
        # Calculate compressed sequence length
        num_layers = len(encoder_dims)
        self.compressed_length = seq_length // (encoder_stride ** num_layers)
        
        # Encoder
        self.encoder = Conv1dEncoder(
            input_dim=input_channels,
            hidden_dims=encoder_dims,
            kernel_size=encoder_kernel,
            stride=encoder_stride,
            output_dim=embedding_dim
        )
        
        # VQ layer
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )
        
        # Decoder
        self.decoder = Conv1dDecoder(
            input_dim=embedding_dim,
            hidden_dims=list(reversed(encoder_dims)),
            kernel_size=encoder_kernel,
            stride=encoder_stride,
            output_dim=input_channels,
            output_length=seq_length
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        return self.quantizer(z)
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)
    
    def get_codebook_size(self) -> int:
        return self.codebook_size
    
    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embedding(indices)
    
    def get_codebook_embeddings(self) -> torch.Tensor:
        """
        Get all codebook embeddings.
        
        Returns:
            embeddings: [K, D] tensor where K = codebook_size, D = embedding_dim
        """
        return self.quantizer.embedding.weight.detach()


if __name__ == "__main__":
    # Test VQ-VAE tokenizer
    print("Testing VQ-VAE Tokenizer...")
    
    tokenizer = VQVAETokenizer(
        seq_length=256,
        codebook_size=512,
        embedding_dim=64,
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
    print(f"Commitment loss: {outputs['commitment_loss'].item():.4f}")
    print(f"Dead ratio: {outputs['dead_ratio'].item():.4f}")
    
    # Reconstruction error
    mse = F.mse_loss(outputs['x_rec'], x)
    print(f"Reconstruction MSE: {mse.item():.4f}")
    
    print("\nVQ-VAE Tokenizer test passed!")
