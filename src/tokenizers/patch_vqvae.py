"""
Patch-based VQ-VAE Tokenizer following LaBraM/NeuroLM/NeuroRVQ standards.

Key difference from progressive VQ-VAE:
- LaBraM style: Input is split into non-overlapping patches (e.g., 200 samples = 1s)
  Each patch is encoded independently and mapped to ONE token.
- Progressive style: Input is progressively downsampled through multiple conv layers.
  Token length depends on number of downsampling layers.

This implementation follows the LaBraM standard:
- 1 token = 200 samples = 1 second @ 200Hz for EEG
- Each patch is encoded through a small encoder network
- Output: 1 code per patch (not multiple codes per patch)

Reference:
- LaBraM: patch_size=200, codebook=8192, dim=64
- NeuroLM: patch_size=200, codebook=8192, dim=128
- NeuroRVQ: patch_size=200, codebook=8192, dim=128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base import BaseTokenizer
from .vqvae import VectorQuantizer


class PatchEncoder(nn.Module):
    """
    Encoder that processes each patch independently.
    
    Takes a patch of shape [B, T_patch] and outputs [B, D].
    """
    
    def __init__(
        self,
        patch_size: int = 200,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 2,
        encoder_type: str = "cnn",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        
        if encoder_type == "cnn":
            # CNN encoder for patch
            layers = []
            in_dim = 1
            current_len = patch_size
            
            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
                layers.extend([
                    nn.Conv1d(in_dim if i == 0 else hidden_dim, out_dim, 
                             kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                ])
                current_len = current_len // 2
            
            self.conv = nn.Sequential(*layers)
            self.flatten_dim = hidden_dim * current_len
            self.proj = nn.Linear(self.flatten_dim, output_dim)
            
        elif encoder_type == "mlp":
            # Simple MLP encoder
            self.encoder = nn.Sequential(
                nn.Linear(patch_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
            
        elif encoder_type == "transformer":
            # Small transformer encoder
            self.input_proj = nn.Linear(1, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=4, 
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            # CLS token for aggregation
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T_patch] single patch
            
        Returns:
            z: [B, D] single latent vector per patch
        """
        if self.encoder_type == "cnn":
            # [B, T] -> [B, 1, T]
            x = x.unsqueeze(1)
            # [B, 1, T] -> [B, C, T']
            x = self.conv(x)
            # [B, C, T'] -> [B, C*T']
            x = x.flatten(1)
            # [B, C*T'] -> [B, D]
            z = self.proj(x)
            
        elif self.encoder_type == "mlp":
            z = self.encoder(x)
            
        elif self.encoder_type == "transformer":
            # [B, T] -> [B, T, 1] -> [B, T, H]
            x = x.unsqueeze(-1)
            x = self.input_proj(x)
            # Add CLS token
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            # Transformer
            x = self.transformer(x)
            # Take CLS output
            z = self.output_proj(x[:, 0])
            
        return z


class PatchDecoder(nn.Module):
    """
    Decoder that reconstructs each patch from its latent vector.
    
    Takes [B, D] and outputs [B, T_patch].
    """
    
    def __init__(
        self,
        patch_size: int = 200,
        hidden_dim: int = 256,
        input_dim: int = 64,
        num_layers: int = 2,
        decoder_type: str = "cnn",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.decoder_type = decoder_type
        
        if decoder_type == "cnn":
            # Calculate starting size
            self.start_len = patch_size // (2 ** num_layers)
            self.proj = nn.Linear(input_dim, hidden_dim * self.start_len)
            
            layers = []
            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else 1
                layers.extend([
                    nn.ConvTranspose1d(hidden_dim, out_dim,
                                      kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(out_dim) if i < num_layers - 1 else nn.Identity(),
                    nn.GELU() if i < num_layers - 1 else nn.Identity(),
                ])
                
            self.deconv = nn.Sequential(*layers)
            
        elif decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, patch_size),
            )
            
        elif decoder_type == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(hidden_dim, 1)
            # Learnable position queries
            self.pos_queries = nn.Parameter(torch.randn(1, patch_size, hidden_dim) * 0.02)
            
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, D] single latent vector
            
        Returns:
            x_rec: [B, T_patch] reconstructed patch
        """
        if self.decoder_type == "cnn":
            # [B, D] -> [B, H*T']
            x = self.proj(z)
            # [B, H*T'] -> [B, H, T']
            x = x.view(x.shape[0], -1, self.start_len)
            # [B, H, T'] -> [B, 1, T]
            x = self.deconv(x)
            # [B, 1, T] -> [B, T]
            x = x.squeeze(1)
            # Ensure correct length
            if x.shape[1] != self.patch_size:
                x = F.interpolate(x.unsqueeze(1), size=self.patch_size, mode='linear').squeeze(1)
            
        elif self.decoder_type == "mlp":
            x = self.decoder(z)
            
        elif self.decoder_type == "transformer":
            # [B, D] -> [B, 1, H]
            z = self.input_proj(z).unsqueeze(1)
            # Expand position queries
            queries = self.pos_queries.expand(z.shape[0], -1, -1)
            # Concat z as condition
            x = queries + z
            # Transformer
            x = self.transformer(x)
            # [B, T, H] -> [B, T, 1] -> [B, T]
            x = self.output_proj(x).squeeze(-1)
            
        return x


class PatchVQVAETokenizer(BaseTokenizer):
    """
    Patch-based VQ-VAE Tokenizer following LaBraM/NeuroLM standards.
    
    Architecture:
    1. Split input into non-overlapping patches
    2. Encode each patch to a single latent vector
    3. Quantize each latent to one code
    4. Decode each code back to patch
    
    Result: 1 token per patch (e.g., 200 samples = 1s = 1 token)
    """
    
    def __init__(
        self,
        seq_length: int = 800,
        patch_size: int = 200,
        input_channels: int = 1,
        codebook_size: int = 1024,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        encoder_type: str = "cnn",
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        **kwargs
    ):
        super().__init__(input_dim=input_channels, latent_dim=embedding_dim)
        
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        
        # Calculate number of tokens
        assert seq_length % patch_size == 0, \
            f"seq_length ({seq_length}) must be divisible by patch_size ({patch_size})"
        self.n_tokens = seq_length // patch_size
        
        # Encoder (processes one patch at a time)
        self.encoder = PatchEncoder(
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
        
        # Decoder (reconstructs one patch at a time)
        self.decoder = PatchDecoder(
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            input_dim=embedding_dim,
            num_layers=num_layers,
            decoder_type=encoder_type,  # Use same type as encoder
        )
        
        print(f"PatchVQVAE initialized:")
        print(f"  - Input: {seq_length} samples")
        print(f"  - Patch size: {patch_size} samples")
        print(f"  - Tokens per window: {self.n_tokens}")
        print(f"  - Codebook size: {codebook_size}")
        print(f"  - Embedding dim: {embedding_dim}")
    
    def _split_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split input into patches.
        
        Args:
            x: [B, T] input signal
            
        Returns:
            patches: [B*N, T_patch] where N = n_tokens
        """
        B = x.shape[0]
        # [B, T] -> [B, N, T_patch]
        patches = x.view(B, self.n_tokens, self.patch_size)
        # [B, N, T_patch] -> [B*N, T_patch]
        patches = patches.view(B * self.n_tokens, self.patch_size)
        return patches
    
    def _merge_patches(self, patches: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Merge patches back to full sequence.
        
        Args:
            patches: [B*N, T_patch]
            batch_size: original batch size
            
        Returns:
            x: [B, T] reconstructed signal
        """
        # [B*N, T_patch] -> [B, N, T_patch]
        patches = patches.view(batch_size, self.n_tokens, self.patch_size)
        # [B, N, T_patch] -> [B, T]
        x = patches.view(batch_size, -1)
        return x
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representations.
        
        Args:
            x: [B, T] input signal
            
        Returns:
            z: [B, N, D] latent vectors (N tokens per sample)
        """
        B = x.shape[0]
        
        # Split to patches
        patches = self._split_to_patches(x)  # [B*N, T_patch]
        
        # Encode each patch
        z = self.encoder(patches)  # [B*N, D]
        
        # Reshape to [B, N, D]
        z = z.view(B, self.n_tokens, self.embedding_dim)
        
        return z
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize latent vectors.
        
        Args:
            z: [B, N, D] latent vectors
            
        Returns:
            z_q: [B, N, D] quantized vectors
            indices: [B, N] token indices
            info: dict with loss, perplexity, etc.
        """
        return self.quantizer(z)
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized vectors to signal.
        
        Args:
            z_q: [B, N, D] quantized vectors
            
        Returns:
            x_rec: [B, T] reconstructed signal
        """
        B = z_q.shape[0]
        
        # Flatten to [B*N, D]
        z_flat = z_q.view(B * self.n_tokens, self.embedding_dim)
        
        # Decode each patch
        patches = self.decoder(z_flat)  # [B*N, T_patch]
        
        # Merge patches
        x_rec = self._merge_patches(patches, B)  # [B, T]
        
        return x_rec
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: [B, T] input signal
            
        Returns:
            dict with x_rec, z, z_q, indices, losses, etc.
            Always includes 'loss' for unified training interface.
        """
        # Encode
        z = self.encode(x)  # [B, N, D]
        
        # Quantize
        z_q, indices, vq_info = self.quantize(z)  # [B, N, D], [B, N]
        
        # Decode
        x_rec = self.decode(z_q)  # [B, T]
        
        # Compute reconstruction loss
        rec_loss = F.mse_loss(x_rec, x)
        
        # Total loss = reconstruction + VQ losses
        vq_loss = vq_info['commitment_loss'] + vq_info['codebook_loss']
        loss = rec_loss + vq_loss
        
        return {
            'x_rec': x_rec,
            'reconstructed': x_rec,  # Standardized alias
            'z': z,
            'z_q': z_q,
            'indices': indices,
            'tokens': indices,  # Standardized alias
            # Losses
            'loss': loss,
            'rec_loss': rec_loss,
            'vq_loss': vq_loss,
            'commitment_loss': vq_info['commitment_loss'],
            'codebook_loss': vq_info['codebook_loss'],
            # Stats
            'perplexity': vq_info['perplexity'],
            'dead_ratio': vq_info['dead_ratio'],
            'code_utilization': vq_info['code_utilization'],
            'utilization': vq_info['code_utilization'],  # Standardized alias
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
            x_rec: [B, T] reconstructed signal
        """
        z_q = self.get_embedding(indices)  # [B, N, D]
        x_rec = self.decode(z_q)
        return x_rec


if __name__ == "__main__":
    print("Testing Patch-based VQ-VAE Tokenizer (LaBraM style)...")
    print("=" * 60)
    
    # Test EEG configuration: 4s @ 200Hz, 1s tokens
    print("\n[Test 1] EEG: 4s window, 1s per token (LaBraM style)")
    eeg_tokenizer = PatchVQVAETokenizer(
        seq_length=800,      # 4s @ 200Hz
        patch_size=200,      # 1s @ 200Hz = 1 token
        codebook_size=1024,
        embedding_dim=64,
        hidden_dim=256,
        num_layers=2,
        encoder_type="cnn",
    )
    
    x_eeg = torch.randn(4, 800)  # [B, T]
    out_eeg = eeg_tokenizer(x_eeg)
    
    print(f"  Input:  {x_eeg.shape}")
    print(f"  Tokens: {out_eeg['indices'].shape} (should be [4, 4])")
    print(f"  Output: {out_eeg['x_rec'].shape}")
    print(f"  MSE:    {F.mse_loss(out_eeg['x_rec'], x_eeg).item():.4f}")
    print(f"  Perplexity: {out_eeg['perplexity'].item():.1f}")
    
    # Test fNIRS configuration: 4s @ 10Hz, 4s token
    print("\n[Test 2] fNIRS: 4s window, 4s per token (1 token per window)")
    fnirs_tokenizer = PatchVQVAETokenizer(
        seq_length=40,       # 4s @ 10Hz
        patch_size=40,       # 4s @ 10Hz = 1 token
        codebook_size=512,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        encoder_type="cnn",
    )
    
    x_fnirs = torch.randn(4, 40)  # [B, T]
    out_fnirs = fnirs_tokenizer(x_fnirs)
    
    print(f"  Input:  {x_fnirs.shape}")
    print(f"  Tokens: {out_fnirs['indices'].shape} (should be [4, 1])")
    print(f"  Output: {out_fnirs['x_rec'].shape}")
    print(f"  MSE:    {F.mse_loss(out_fnirs['x_rec'], x_fnirs).item():.4f}")
    print(f"  Perplexity: {out_fnirs['perplexity'].item():.1f}")
    
    # Test alternative: fNIRS with 1s tokens using MLP encoder (for short patches)
    print("\n[Test 3] fNIRS (alternative): 4s window, 1s per token (MLP encoder)")
    fnirs_tokenizer_1s = PatchVQVAETokenizer(
        seq_length=40,       # 4s @ 10Hz
        patch_size=10,       # 1s @ 10Hz = 1 token
        codebook_size=512,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        encoder_type="mlp",  # MLP works better for very short patches
    )
    
    out_fnirs_1s = fnirs_tokenizer_1s(x_fnirs)
    print(f"  Input:  {x_fnirs.shape}")
    print(f"  Tokens: {out_fnirs_1s['indices'].shape} (should be [4, 4])")
    print(f"  Output: {out_fnirs_1s['x_rec'].shape}")
    print(f"  MSE:    {F.mse_loss(out_fnirs_1s['x_rec'], x_fnirs).item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
