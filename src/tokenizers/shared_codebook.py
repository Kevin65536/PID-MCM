"""
Shared-Codebook Dual-Modal Tokenizer for EEG and fNIRS alignment.

Core idea
---------
Instead of training EEG and fNIRS tokenizers *independently*, this module
trains both with a **single shared codebook**.  A contrastive objective (see
``src/losses/contrastive_losses.py``) explicitly encourages the EEG and fNIRS
encoders to map the same cognitive trial to the *same* (or nearby) codebook
entries, producing a unified cross-modal token vocabulary.

Architecture
------------
::

    x_eeg   [B, C_eeg, T_eeg]   ──► MultiChannelPatchEncoder ──► z_eeg   [B, N, D]
                                                                          │
                                         Shared NormEMAVectorQuantizer ◄──┤
                                                                          │
    x_fnirs [B, C_fnirs, T_fnirs] ──► MultiChannelPatchEncoder ──► z_fnirs [B, N, D]

    z_q_eeg   [B, N, D] ──► MultiChannelPatchDecoder ──► x_rec_eeg   [B, C_eeg, T_eeg]
    z_q_fnirs [B, N, D] ──► MultiChannelPatchDecoder ──► x_rec_fnirs [B, C_fnirs, T_fnirs]

Loss (see SharedCodebookLoss in losses/contrastive_losses.py)
-----
    L = L_rec_eeg + L_rec_fnirs + L_vq + λ_c * L_InfoNCE + λ_idx * L_index_match

References
----------
- LaBraM: https://github.com/935963004/LaBraM
- SimCLR: Chen et al., 2020
- CLIP: Radford et al., 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from .labram_vqnsp import NormEMAVectorQuantizer, l2norm


# ---------------------------------------------------------------------------
# Encoder: Multi-channel patch encoder
# ---------------------------------------------------------------------------

class MultiChannelPatchEncoder(nn.Module):
    """
    Encode a multi-channel signal into a sequence of patch token embeddings.

    Processing pipeline:
    1. Split the input ``[B, C, T]`` into non-overlapping patches along T.
    2. Encode each patch of every channel independently via a small CNN.
    3. Average-pool across the channel dimension to get ``[B, N_patches, D]``.

    Averaging across channels is simple but effective for single-codebook
    tokenisation because it produces a channel-agnostic latent suitable for
    sharing with the other modality.

    Args:
        seq_length:  Number of time samples per window.
        patch_size:  Number of samples per patch.
        n_channels:  Number of input channels after filtering.
        hidden_dim:  Hidden dimension of the CNN encoder.
        output_dim:  Output embedding dimension (shared with codebook).
        num_layers:  Number of CNN downsampling layers.
    """

    def __init__(
        self,
        seq_length: int,
        patch_size: int,
        n_channels: int,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.output_dim = output_dim

        assert seq_length % patch_size == 0, (
            f"seq_length={seq_length} must be divisible by patch_size={patch_size}"
        )
        self.n_patches = seq_length // patch_size

        # CNN encoder: [B*C*N, 1, patch_size] → [B*C*N, H, T'] → flatten → [B*C*N, D]
        layers = []
        in_ch = 1
        current_len = patch_size
        for i in range(num_layers):
            out_ch = hidden_dim
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            in_ch = out_ch
            # output_len = floor((input_len + 2*padding - kernel) / stride + 1)
            #            = floor((current_len + 2*3 - 7) / 2 + 1)
            current_len = math.floor((current_len + 2 * 3 - 7) / 2 + 1)

        self.conv = nn.Sequential(*layers)
        self.flatten_dim = hidden_dim * current_len
        self.proj = nn.Linear(self.flatten_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, C, T]`` multi-channel time series.

        Returns:
            z: ``[B, N_patches, D]`` patch token embeddings.
        """
        B, C, T = x.shape
        N = self.n_patches

        # [B, C, T] → [B, C, N, patch_size]
        x = x.view(B, C, N, self.patch_size)
        # → [B*C*N, patch_size]
        x = x.reshape(B * C * N, self.patch_size)
        # → [B*C*N, 1, patch_size]
        x = x.unsqueeze(1)

        # CNN encode
        x = self.conv(x)                         # [B*C*N, H, T']
        x = x.flatten(1)                          # [B*C*N, H*T']
        z = self.proj(x)                          # [B*C*N, D]

        # → [B, C, N, D] → average over channels → [B, N, D]
        z = z.view(B, C, N, self.output_dim)
        z = z.mean(dim=1)                         # [B, N, D]

        return z


# ---------------------------------------------------------------------------
# Decoder: Multi-channel patch decoder
# ---------------------------------------------------------------------------

class MultiChannelPatchDecoder(nn.Module):
    """
    Decode a sequence of patch token embeddings back to a multi-channel signal.

    The single shared embedding ``[B, N, D]`` is first broadcast to all
    channels, then decoded per-channel via a transposed-CNN.

    Args:
        seq_length:  Target number of time samples.
        patch_size:  Number of samples per patch.
        n_channels:  Number of output channels.
        hidden_dim:  Decoder hidden dimension.
        input_dim:   Token embedding dimension (must match encoder output_dim).
        num_layers:  Number of transposed-CNN upsampling layers.
    """

    def __init__(
        self,
        seq_length: int,
        patch_size: int,
        n_channels: int,
        hidden_dim: int = 256,
        input_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.input_dim = input_dim
        self.n_patches = seq_length // patch_size

        # Starting length before upsampling
        self.start_len = math.ceil(patch_size / (2 ** num_layers))
        self.proj = nn.Linear(input_dim, hidden_dim * self.start_len)

        layers = []
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else n_channels
            layers.extend([
                nn.ConvTranspose1d(hidden_dim, out_ch,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch) if i < num_layers - 1 else nn.Identity(),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
            ])
        self.deconv = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: ``[B, N, D]`` quantised patch embeddings.

        Returns:
            x_rec: ``[B, C, T]`` reconstructed multi-channel signal.
        """
        B, N, D = z_q.shape
        C = self.n_channels

        # Flatten patches: [B*N, D]
        z = z_q.view(B * N, D)
        # Project: [B*N, H*start_len]
        x = self.proj(z)
        # Reshape: [B*N, H, start_len]
        x = x.view(B * N, -1, self.start_len)
        # Upsample: [B*N, C, patch_size'] (may be slightly larger than patch_size)
        x = self.deconv(x)                         # [B*N, C, T_patch']

        # Trim / pad to exact patch_size
        if x.shape[-1] != self.patch_size:
            x = F.interpolate(x, size=self.patch_size, mode='linear',
                              align_corners=False)

        # [B*N, C, patch_size] → [B, N, C, patch_size] → [B, C, N*patch_size]
        x = x.view(B, N, C, self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()     # [B, C, N, patch_size]
        x = x.view(B, C, self.seq_length)

        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SharedCodebookTokenizer(nn.Module):
    """
    Joint EEG-fNIRS tokenizer with a single shared codebook.

    This is the core model for contrastive cross-modal token alignment.
    Both modalities are encoded to the same embedding space, quantised through
    one shared ``NormEMAVectorQuantizer``, and decoded back independently.

    The ``forward`` method returns all quantities needed by
    ``SharedCodebookLoss`` to compute the combined training objective.

    Args:
        eeg_seq_length:    EEG samples per window (e.g. 800 for 4 s @ 200 Hz).
        eeg_patch_size:    EEG samples per patch (e.g. 200 for 1 s patches).
        eeg_channels:      Number of EEG channels after preprocessing (e.g. 30).
        eeg_hidden_dim:    EEG encoder/decoder hidden dimension.
        fnirs_seq_length:  fNIRS samples per window (e.g. 40 for 4 s @ 10 Hz).
        fnirs_patch_size:  fNIRS samples per patch (e.g. 10 for 1 s patches).
        fnirs_channels:    Number of fNIRS channels after preprocessing (e.g. 36).
        fnirs_hidden_dim:  fNIRS encoder/decoder hidden dimension.
        codebook_size:     Shared codebook size K (e.g. 1024).
        embedding_dim:     Token embedding dimension D (e.g. 64).
        num_encoder_layers: CNN layers in encoder/decoder.
        vq_beta:           Commitment loss coefficient.
        vq_decay:          EMA decay for codebook update.
        kmeans_init:       Whether to init codebook with k-means.
    """

    def __init__(
        self,
        # EEG
        eeg_seq_length: int = 800,
        eeg_patch_size: int = 200,
        eeg_channels: int = 30,
        eeg_hidden_dim: int = 256,
        # fNIRS
        fnirs_seq_length: int = 40,
        fnirs_patch_size: int = 10,
        fnirs_channels: int = 36,
        fnirs_hidden_dim: int = 128,
        # Shared codebook
        codebook_size: int = 1024,
        embedding_dim: int = 64,
        # Architecture
        num_encoder_layers: int = 2,
        # VQ hyper-params
        vq_beta: float = 1.0,
        vq_decay: float = 0.99,
        kmeans_init: bool = True,
    ):
        super().__init__()

        self.eeg_seq_length = eeg_seq_length
        self.eeg_patch_size = eeg_patch_size
        self.eeg_channels = eeg_channels
        self.fnirs_seq_length = fnirs_seq_length
        self.fnirs_patch_size = fnirs_patch_size
        self.fnirs_channels = fnirs_channels
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        # --- EEG encoder / decoder ---
        self.eeg_encoder = MultiChannelPatchEncoder(
            seq_length=eeg_seq_length,
            patch_size=eeg_patch_size,
            n_channels=eeg_channels,
            hidden_dim=eeg_hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_encoder_layers,
        )
        self.eeg_decoder = MultiChannelPatchDecoder(
            seq_length=eeg_seq_length,
            patch_size=eeg_patch_size,
            n_channels=eeg_channels,
            hidden_dim=eeg_hidden_dim,
            input_dim=embedding_dim,
            num_layers=num_encoder_layers,
        )

        # --- fNIRS encoder / decoder ---
        self.fnirs_encoder = MultiChannelPatchEncoder(
            seq_length=fnirs_seq_length,
            patch_size=fnirs_patch_size,
            n_channels=fnirs_channels,
            hidden_dim=fnirs_hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_encoder_layers,
        )
        self.fnirs_decoder = MultiChannelPatchDecoder(
            seq_length=fnirs_seq_length,
            patch_size=fnirs_patch_size,
            n_channels=fnirs_channels,
            hidden_dim=fnirs_hidden_dim,
            input_dim=embedding_dim,
            num_layers=num_encoder_layers,
        )

        # --- Shared codebook ---
        self.quantizer = NormEMAVectorQuantizer(
            n_embed=codebook_size,
            embedding_dim=embedding_dim,
            beta=vq_beta,
            decay=vq_decay,
            kmeans_init=kmeans_init,
        )

        n_eeg_tokens = eeg_seq_length // eeg_patch_size
        n_fnirs_tokens = fnirs_seq_length // fnirs_patch_size
        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"SharedCodebookTokenizer:\n"
            f"  EEG   : {eeg_channels} ch × {eeg_seq_length} samples "
            f"→ {n_eeg_tokens} tokens\n"
            f"  fNIRS : {fnirs_channels} ch × {fnirs_seq_length} samples "
            f"→ {n_fnirs_tokens} tokens\n"
            f"  Shared codebook : size={codebook_size}, dim={embedding_dim}\n"
            f"  Total parameters: {n_params:,}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_eeg(self, x_eeg: torch.Tensor) -> torch.Tensor:
        """Encode EEG to continuous patch tokens. Returns [B, N, D]."""
        return self.eeg_encoder(x_eeg)

    def encode_fnirs(self, x_fnirs: torch.Tensor) -> torch.Tensor:
        """Encode fNIRS to continuous patch tokens. Returns [B, N, D]."""
        return self.fnirs_encoder(x_fnirs)

    def quantize(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Quantize continuous tokens through the shared codebook."""
        return self.quantizer(z)

    def decode_eeg(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantised EEG tokens to signal. Returns [B, C_eeg, T_eeg]."""
        return self.eeg_decoder(z_q)

    def decode_fnirs(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantised fNIRS tokens to signal. Returns [B, C_fnirs, T_fnirs]."""
        return self.fnirs_decoder(z_q)

    def forward(
        self,
        x_eeg: torch.Tensor,
        x_fnirs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for paired (EEG, fNIRS) samples.

        Args:
            x_eeg:   ``[B, C_eeg, T_eeg]``  EEG signal.
            x_fnirs: ``[B, C_fnirs, T_fnirs]``  fNIRS signal.

        Returns:
            Dict with keys:
            - ``x_rec_eeg``     : ``[B, C_eeg, T_eeg]`` reconstructed EEG.
            - ``x_rec_fnirs``   : ``[B, C_fnirs, T_fnirs]`` reconstructed fNIRS.
            - ``z_eeg``         : ``[B, N, D]`` continuous EEG token embeddings.
            - ``z_fnirs``       : ``[B, N, D]`` continuous fNIRS token embeddings.
            - ``z_q_eeg``       : ``[B, N, D]`` quantised EEG embeddings.
            - ``z_q_fnirs``     : ``[B, N, D]`` quantised fNIRS embeddings.
            - ``indices_eeg``   : ``[B, N]`` EEG codebook indices.
            - ``indices_fnirs`` : ``[B, N]`` fNIRS codebook indices.
            - ``vq_loss``       : scalar combined commitment loss.
            - ``perplexity_eeg``: scalar EEG codebook perplexity.
            - ``perplexity_fnirs``: scalar fNIRS codebook perplexity.
            - ``utilization``   : scalar codebook utilization ratio.
            - ``codebook_weight``: ``[K, D]`` shared codebook embeddings.
        """
        # Encode
        z_eeg = self.encode_eeg(x_eeg)       # [B, N_eeg, D]
        z_fnirs = self.encode_fnirs(x_fnirs)  # [B, N_fnirs, D]

        # Quantize both through the SAME codebook
        z_q_eeg, indices_eeg, info_eeg = self.quantize(z_eeg)
        z_q_fnirs, indices_fnirs, info_fnirs = self.quantize(z_fnirs)

        # Decode
        x_rec_eeg = self.decode_eeg(z_q_eeg)
        x_rec_fnirs = self.decode_fnirs(z_q_fnirs)

        # Combined VQ loss (sum of commitment losses from both modalities)
        vq_loss = info_eeg['vq_loss'] + info_fnirs['vq_loss']

        # Index-matching rate (diagnostic only, no gradient)
        with torch.no_grad():
            # Align to the shorter token sequence if lengths differ
            n_min = min(indices_eeg.shape[1], indices_fnirs.shape[1])
            match_rate = (
                indices_eeg[:, :n_min] == indices_fnirs[:, :n_min]
            ).float().mean()

        return {
            # Reconstructions
            'x_rec_eeg': x_rec_eeg,
            'x_rec_fnirs': x_rec_fnirs,
            # Continuous embeddings (needed for contrastive loss)
            'z_eeg': z_eeg,
            'z_fnirs': z_fnirs,
            # Quantised embeddings
            'z_q_eeg': z_q_eeg,
            'z_q_fnirs': z_q_fnirs,
            # Discrete tokens
            'indices_eeg': indices_eeg,
            'indices_fnirs': indices_fnirs,
            # Losses / stats
            'vq_loss': vq_loss,
            'perplexity_eeg': info_eeg['perplexity'],
            'perplexity_fnirs': info_fnirs['perplexity'],
            'utilization': info_eeg['utilization'],
            'index_match_rate': match_rate,
            # Codebook weights (needed for IndexMatchingLoss)
            'codebook_weight': self.quantizer.weight,
        }

    def get_codebook_size(self) -> int:
        return self.codebook_size

    def get_codebook_embeddings(self) -> torch.Tensor:
        """Return the shared codebook weight matrix ``[K, D]``."""
        return self.quantizer.weight.detach()

    @torch.no_grad()
    def tokenize_eeg(self, x_eeg: torch.Tensor) -> torch.Tensor:
        """
        Encode and quantize EEG only (for inference / probing).

        Returns:
            indices: ``[B, N]`` integer token indices.
        """
        z = self.encode_eeg(x_eeg)
        _, indices, _ = self.quantize(z)
        return indices

    @torch.no_grad()
    def tokenize_fnirs(self, x_fnirs: torch.Tensor) -> torch.Tensor:
        """
        Encode and quantize fNIRS only (for inference / probing).

        Returns:
            indices: ``[B, N]`` integer token indices.
        """
        z = self.encode_fnirs(x_fnirs)
        _, indices, _ = self.quantize(z)
        return indices
