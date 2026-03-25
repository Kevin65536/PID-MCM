"""
Shared-codebook LaBraM-style tokenizer for aligned EEG and fNIRS signals.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.pid_losses import AlignmentLoss

from .base import BaseTokenizer
from .labram_vqnsp import NormEMAVectorQuantizer, TransformerDecoder, TransformerEncoder, l2norm


class MultiChannelPatchEmbedding(nn.Module):
    """Embed multi-channel temporal patches into a transformer space."""

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        embed_dim: int,
        use_frequency: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.use_frequency = use_frequency
        self.fft_size = patch_size // 2 + 1

        if use_frequency:
            input_dim = input_channels * self.fft_size * 2
        else:
            input_dim = input_channels * patch_size
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: [B, N, C, P]
        Returns:
            [B, N, D]
        """
        if self.use_frequency:
            fft = torch.fft.rfft(patches, dim=-1)
            amplitude = torch.log(torch.abs(fft) + 1e-8)
            phase = torch.angle(fft) / math.pi
            features = torch.cat([amplitude, phase], dim=-1)
        else:
            features = patches

        return self.proj(features.flatten(start_dim=2))


class SharedLaBraMVQNSP(BaseTokenizer):
    """Dual-encoder tokenizer with one shared codebook for EEG and fNIRS."""

    def __init__(
        self,
        eeg_seq_length: int = 2000,
        eeg_patch_size: int = 400,
        eeg_channels: int = 30,
        eeg_encoder_embed_dim: int = 256,
        eeg_encoder_depth: int = 8,
        eeg_encoder_num_heads: int = 8,
        eeg_decoder_embed_dim: int = 256,
        eeg_decoder_depth: int = 4,
        eeg_decoder_num_heads: int = 8,
        fnirs_seq_length: int = 100,
        fnirs_patch_size: int = 20,
        fnirs_channels: int = 36,
        fnirs_encoder_embed_dim: int = 160,
        fnirs_encoder_depth: int = 6,
        fnirs_encoder_num_heads: int = 4,
        fnirs_decoder_embed_dim: int = 160,
        fnirs_decoder_depth: int = 3,
        fnirs_decoder_num_heads: int = 4,
        codebook_size: int = 4096,
        codebook_dim: int = 64,
        beta: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
        revive_dead_codes: bool = True,
        dead_code_threshold: int = 10,
        eeg_amplitude_weight: float = 1.0,
        eeg_phase_weight: float = 1.0,
        eeg_time_weight: float = 0.75,
        fnirs_amplitude_weight: float = 1.0,
        fnirs_phase_weight: float = 0.25,
        fnirs_time_weight: float = 1.25,
        latent_alignment_weight: float = 0.1,
        assignment_alignment_weight: float = 0.05,
        assignment_temperature: float = 0.2,
        alignment_lag_candidates: List[int] | None = None,
        alignment_selection: str = 'min',
        dropout: float = 0.0,
        drop_path: float = 0.1,
        use_smooth_l1: bool = False,
        **kwargs,
    ):
        super().__init__(input_dim=2, latent_dim=codebook_dim)

        if eeg_seq_length % eeg_patch_size != 0:
            raise ValueError("eeg_seq_length must be divisible by eeg_patch_size")
        if fnirs_seq_length % fnirs_patch_size != 0:
            raise ValueError("fnirs_seq_length must be divisible by fnirs_patch_size")

        self.eeg_seq_length = eeg_seq_length
        self.eeg_patch_size = eeg_patch_size
        self.eeg_channels = eeg_channels
        self.fnirs_seq_length = fnirs_seq_length
        self.fnirs_patch_size = fnirs_patch_size
        self.fnirs_channels = fnirs_channels
        self.eeg_n_patches = eeg_seq_length // eeg_patch_size
        self.fnirs_n_patches = fnirs_seq_length // fnirs_patch_size
        if self.eeg_n_patches != self.fnirs_n_patches:
            raise ValueError(
                "EEG and fNIRS must produce the same token count per window for aligned training "
                f"(got EEG={self.eeg_n_patches}, fNIRS={self.fnirs_n_patches})"
            )
        self.n_patches = self.eeg_n_patches
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.eeg_fft_size = eeg_patch_size // 2 + 1
        self.fnirs_fft_size = fnirs_patch_size // 2 + 1
        self.eeg_amplitude_weight = eeg_amplitude_weight
        self.eeg_phase_weight = eeg_phase_weight
        self.eeg_time_weight = eeg_time_weight
        self.fnirs_amplitude_weight = fnirs_amplitude_weight
        self.fnirs_phase_weight = fnirs_phase_weight
        self.fnirs_time_weight = fnirs_time_weight
        self.latent_alignment_weight = latent_alignment_weight
        self.assignment_alignment_weight = assignment_alignment_weight
        self.assignment_temperature = assignment_temperature
        self.alignment_lag_candidates = sorted({max(int(lag), 0) for lag in (alignment_lag_candidates or [0])})
        if not self.alignment_lag_candidates:
            self.alignment_lag_candidates = [0]
        if alignment_selection not in {'min', 'mean'}:
            raise ValueError("alignment_selection must be 'min' or 'mean'")
        self.alignment_selection = alignment_selection
        self.alignment_scale = 1.0
        self.loss_fn = F.smooth_l1_loss if use_smooth_l1 else F.mse_loss
        self.alignment_loss = AlignmentLoss()

        self.eeg_patch_embed = MultiChannelPatchEmbedding(
            input_channels=eeg_channels,
            patch_size=eeg_patch_size,
            embed_dim=eeg_encoder_embed_dim,
            use_frequency=True,
        )
        self.fnirs_patch_embed = MultiChannelPatchEmbedding(
            input_channels=fnirs_channels,
            patch_size=fnirs_patch_size,
            embed_dim=fnirs_encoder_embed_dim,
            use_frequency=True,
        )

        self.eeg_encoder = TransformerEncoder(
            embed_dim=eeg_encoder_embed_dim,
            depth=eeg_encoder_depth,
            num_heads=eeg_encoder_num_heads,
            dropout=dropout,
            drop_path=drop_path,
            max_patches=self.n_patches,
        )
        self.fnirs_encoder = TransformerEncoder(
            embed_dim=fnirs_encoder_embed_dim,
            depth=fnirs_encoder_depth,
            num_heads=fnirs_encoder_num_heads,
            dropout=dropout,
            drop_path=drop_path,
            max_patches=self.n_patches,
        )

        self.eeg_encode_proj = nn.Sequential(
            nn.Linear(eeg_encoder_embed_dim, eeg_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(eeg_encoder_embed_dim, codebook_dim),
        )
        self.fnirs_encode_proj = nn.Sequential(
            nn.Linear(fnirs_encoder_embed_dim, fnirs_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(fnirs_encoder_embed_dim, codebook_dim),
        )

        self.quantizer = NormEMAVectorQuantizer(
            n_embed=codebook_size,
            embedding_dim=codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
        )

        self.eeg_decode_input_proj = nn.Linear(codebook_dim, eeg_decoder_embed_dim)
        self.fnirs_decode_input_proj = nn.Linear(codebook_dim, fnirs_decoder_embed_dim)

        self.eeg_decoder = TransformerDecoder(
            embed_dim=eeg_decoder_embed_dim,
            depth=eeg_decoder_depth,
            num_heads=eeg_decoder_num_heads,
            dropout=dropout,
            drop_path=0.0,
            max_patches=self.n_patches,
        )
        self.fnirs_decoder = TransformerDecoder(
            embed_dim=fnirs_decoder_embed_dim,
            depth=fnirs_decoder_depth,
            num_heads=fnirs_decoder_num_heads,
            dropout=dropout,
            drop_path=0.0,
            max_patches=self.n_patches,
        )

        self.eeg_amplitude_head = nn.Linear(eeg_decoder_embed_dim, eeg_channels * self.eeg_fft_size)
        self.eeg_phase_head = nn.Linear(eeg_decoder_embed_dim, eeg_channels * self.eeg_fft_size)
        self.fnirs_amplitude_head = nn.Linear(fnirs_decoder_embed_dim, fnirs_channels * self.fnirs_fft_size)
        self.fnirs_phase_head = nn.Linear(fnirs_decoder_embed_dim, fnirs_channels * self.fnirs_fft_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _split_to_patches(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        bsz, channels, seq_len = x.shape
        n_patches = seq_len // patch_size
        return x.view(bsz, channels, n_patches, patch_size).permute(0, 2, 1, 3).contiguous()

    def _compute_fft_targets(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fft = torch.fft.rfft(patches, dim=-1)
        amplitude = torch.log(torch.abs(fft) + 1e-8)
        phase = torch.angle(fft) / math.pi
        return amplitude, phase

    def _reconstruct_time(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        amp = torch.exp(amplitude)
        pha = phase * math.pi
        real = amp * torch.cos(pha)
        imag = amp * torch.sin(pha)
        fft = torch.complex(real, imag)
        patches = torch.fft.irfft(fft, n=patch_size, dim=-1)
        return patches.permute(0, 2, 1, 3).contiguous().view(patches.shape[0], patches.shape[2], -1)

    def _encode_modality(
        self,
        x: torch.Tensor,
        patch_size: int,
        patch_embed: nn.Module,
        encoder: nn.Module,
        encode_proj: nn.Module,
    ) -> torch.Tensor:
        patches = self._split_to_patches(x, patch_size)
        embeddings = patch_embed(patches)
        encoded = encoder(embeddings)
        return encode_proj(encoded)

    def _decode_modality(
        self,
        z_q: torch.Tensor,
        decode_input_proj: nn.Module,
        decoder: nn.Module,
        amplitude_head: nn.Module,
        phase_head: nn.Module,
        channels: int,
        fft_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder_input = decode_input_proj(z_q)
        decoded = decoder(decoder_input)
        amplitude = amplitude_head(decoded).view(z_q.shape[0], self.n_patches, channels, fft_size)
        phase = phase_head(decoded).view(z_q.shape[0], self.n_patches, channels, fft_size)
        return amplitude, phase

    def _assignment_logits(self, z: torch.Tensor) -> torch.Tensor:
        normalized_z = l2norm(z)
        return torch.einsum("bnd,kd->bnk", normalized_z, self.quantizer.weight)

    def _symmetric_kl(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        temperature = max(self.assignment_temperature, 1e-3)
        log_probs_a = F.log_softmax(logits_a / temperature, dim=-1)
        log_probs_b = F.log_softmax(logits_b / temperature, dim=-1)
        probs_a = log_probs_a.exp()
        probs_b = log_probs_b.exp()
        kl_ab = F.kl_div(log_probs_a, probs_b, reduction='batchmean')
        kl_ba = F.kl_div(log_probs_b, probs_a, reduction='batchmean')
        return 0.5 * (kl_ab + kl_ba)

    def _align_pair(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor, lag: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if lag < 0:
            raise ValueError('Only non-negative lag is supported')
        usable = min(tensor_a.shape[1], tensor_b.shape[1] - lag)
        if usable <= 0:
            return tensor_a[:, :0], tensor_b[:, :0]
        return tensor_a[:, :usable], tensor_b[:, lag:lag + usable]

    def _compute_alignment_losses(
        self,
        z_eeg: torch.Tensor,
        z_fnirs: torch.Tensor,
        eeg_logits: torch.Tensor,
        fnirs_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        combined_losses = []
        latent_losses = []
        assignment_losses = []
        valid_lags = []

        for lag in self.alignment_lag_candidates:
            aligned_z_eeg, aligned_z_fnirs = self._align_pair(z_eeg, z_fnirs, lag)
            aligned_eeg_logits, aligned_fnirs_logits = self._align_pair(eeg_logits, fnirs_logits, lag)
            if aligned_z_eeg.shape[1] == 0:
                continue

            latent_loss = self.alignment_loss(aligned_z_eeg, aligned_z_fnirs)
            assignment_loss = self._symmetric_kl(aligned_eeg_logits, aligned_fnirs_logits)
            combined_loss = self.latent_alignment_weight * latent_loss + self.assignment_alignment_weight * assignment_loss

            combined_losses.append(combined_loss)
            latent_losses.append(latent_loss)
            assignment_losses.append(assignment_loss)
            valid_lags.append(lag)

        if not combined_losses:
            zero = torch.tensor(0.0, device=z_eeg.device, dtype=z_eeg.dtype)
            return {
                'latent_align_loss': zero,
                'assignment_align_loss': zero,
                'selected_lag': torch.tensor(0.0, device=z_eeg.device, dtype=z_eeg.dtype),
            }

        if self.alignment_selection == 'mean':
            latent_align_loss = torch.stack(latent_losses).mean()
            assignment_align_loss = torch.stack(assignment_losses).mean()
            selected_lag = float(sum(valid_lags) / len(valid_lags))
        else:
            best_index = int(torch.argmin(torch.stack(combined_losses)).item())
            latent_align_loss = latent_losses[best_index]
            assignment_align_loss = assignment_losses[best_index]
            selected_lag = float(valid_lags[best_index])

        return {
            'latent_align_loss': latent_align_loss,
            'assignment_align_loss': assignment_align_loss,
            'selected_lag': torch.tensor(selected_lag, device=z_eeg.device, dtype=z_eeg.dtype),
        }

    def _match_rate_at_lag(self, eeg_indices: torch.Tensor, fnirs_indices: torch.Tensor, lag: int) -> torch.Tensor:
        aligned_eeg, aligned_fnirs = self._align_pair(eeg_indices, fnirs_indices, lag)
        if aligned_eeg.shape[1] == 0:
            return torch.tensor(0.0, device=eeg_indices.device, dtype=torch.float32)
        return (aligned_eeg == aligned_fnirs).float().mean()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use encode_modalities(eeg, fnirs) for the shared tokenizer")

    def encode_modalities(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_eeg = self._encode_modality(
            eeg,
            self.eeg_patch_size,
            self.eeg_patch_embed,
            self.eeg_encoder,
            self.eeg_encode_proj,
        )
        z_fnirs = self._encode_modality(
            fnirs,
            self.fnirs_patch_size,
            self.fnirs_patch_embed,
            self.fnirs_encoder,
            self.fnirs_encode_proj,
        )
        return z_eeg, z_fnirs

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        return self.quantizer(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use modality-specific decode paths for the shared tokenizer")

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if eeg.dim() != 3 or fnirs.dim() != 3:
            raise ValueError("Expected eeg and fnirs tensors with shape [B, C, T]")
        if eeg.shape[-1] != self.eeg_seq_length:
            raise ValueError(f"Expected EEG length {self.eeg_seq_length}, got {eeg.shape[-1]}")
        if fnirs.shape[-1] != self.fnirs_seq_length:
            raise ValueError(f"Expected fNIRS length {self.fnirs_seq_length}, got {fnirs.shape[-1]}")

        eeg_patches = self._split_to_patches(eeg, self.eeg_patch_size)
        fnirs_patches = self._split_to_patches(fnirs, self.fnirs_patch_size)
        eeg_target_amp, eeg_target_phase = self._compute_fft_targets(eeg_patches)
        fnirs_target_amp, fnirs_target_phase = self._compute_fft_targets(fnirs_patches)

        z_eeg, z_fnirs = self.encode_modalities(eeg, fnirs)
        z_joint = torch.cat([z_eeg, z_fnirs], dim=0)
        z_q_joint, indices_joint, quant_info = self.quantize(z_joint)
        z_q_eeg, z_q_fnirs = torch.split(z_q_joint, [z_eeg.shape[0], z_fnirs.shape[0]], dim=0)
        eeg_indices, fnirs_indices = torch.split(indices_joint, [z_eeg.shape[0], z_fnirs.shape[0]], dim=0)

        eeg_logits = self._assignment_logits(z_eeg)
        fnirs_logits = self._assignment_logits(z_fnirs)
        alignment_losses = self._compute_alignment_losses(z_eeg, z_fnirs, eeg_logits, fnirs_logits)
        latent_align_loss = alignment_losses['latent_align_loss']
        assignment_align_loss = alignment_losses['assignment_align_loss']
        selected_lag = alignment_losses['selected_lag']

        eeg_pred_amp, eeg_pred_phase = self._decode_modality(
            z_q_eeg,
            self.eeg_decode_input_proj,
            self.eeg_decoder,
            self.eeg_amplitude_head,
            self.eeg_phase_head,
            self.eeg_channels,
            self.eeg_fft_size,
        )
        fnirs_pred_amp, fnirs_pred_phase = self._decode_modality(
            z_q_fnirs,
            self.fnirs_decode_input_proj,
            self.fnirs_decoder,
            self.fnirs_amplitude_head,
            self.fnirs_phase_head,
            self.fnirs_channels,
            self.fnirs_fft_size,
        )

        eeg_amp_loss = self.loss_fn(eeg_pred_amp, eeg_target_amp)
        eeg_phase_loss = self.loss_fn(eeg_pred_phase, eeg_target_phase)
        eeg_rec = self._reconstruct_time(eeg_pred_amp, eeg_pred_phase, self.eeg_patch_size)
        eeg_time_loss = self.loss_fn(eeg_rec, eeg)
        eeg_rec_loss = (
            self.eeg_amplitude_weight * eeg_amp_loss +
            self.eeg_phase_weight * eeg_phase_loss +
            self.eeg_time_weight * eeg_time_loss
        )

        fnirs_amp_loss = self.loss_fn(fnirs_pred_amp, fnirs_target_amp)
        fnirs_phase_loss = self.loss_fn(fnirs_pred_phase, fnirs_target_phase)
        fnirs_rec = self._reconstruct_time(fnirs_pred_amp, fnirs_pred_phase, self.fnirs_patch_size)
        fnirs_time_loss = self.loss_fn(fnirs_rec, fnirs)
        fnirs_rec_loss = (
            self.fnirs_amplitude_weight * fnirs_amp_loss +
            self.fnirs_phase_weight * fnirs_phase_loss +
            self.fnirs_time_weight * fnirs_time_loss
        )

        vq_loss = quant_info['vq_loss']
        total_loss = (
            eeg_rec_loss +
            fnirs_rec_loss +
            vq_loss +
            (self.latent_alignment_weight * self.alignment_scale) * latent_align_loss +
            (self.assignment_alignment_weight * self.alignment_scale) * assignment_align_loss
        )

        token_match = (eeg_indices == fnirs_indices).float().mean()
        best_lag_token_match = self._match_rate_at_lag(eeg_indices, fnirs_indices, int(selected_lag.item()))
        eeg_unique = torch.unique(eeg_indices)
        fnirs_unique = torch.unique(fnirs_indices)
        overlap = torch.tensor(
            len(set(eeg_unique.tolist()) & set(fnirs_unique.tolist())) /
            max(len(set(eeg_unique.tolist()) | set(fnirs_unique.tolist())), 1),
            device=eeg.device,
            dtype=torch.float32,
        )

        return {
            'loss': total_loss,
            'eeg_rec_loss': eeg_rec_loss,
            'fnirs_rec_loss': fnirs_rec_loss,
            'eeg_amp_loss': eeg_amp_loss,
            'eeg_phase_loss': eeg_phase_loss,
            'eeg_time_loss': eeg_time_loss,
            'fnirs_amp_loss': fnirs_amp_loss,
            'fnirs_phase_loss': fnirs_phase_loss,
            'fnirs_time_loss': fnirs_time_loss,
            'vq_loss': vq_loss,
            'latent_align_loss': latent_align_loss,
            'assignment_align_loss': assignment_align_loss,
            'selected_alignment_lag': selected_lag,
            'alignment_scale': torch.tensor(self.alignment_scale, device=eeg.device, dtype=torch.float32),
            'token_match': token_match,
            'best_lag_token_match': best_lag_token_match,
            'code_overlap': overlap,
            'perplexity': quant_info['perplexity'],
            'utilization': quant_info['utilization'],
            'eeg_reconstructed': eeg_rec,
            'fnirs_reconstructed': fnirs_rec,
            'eeg_indices': eeg_indices,
            'fnirs_indices': fnirs_indices,
            'eeg_z': z_eeg,
            'fnirs_z': z_fnirs,
            'eeg_z_q': z_q_eeg,
            'fnirs_z_q': z_q_fnirs,
        }

    def get_codebook_size(self) -> int:
        return self.codebook_size

    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.get_codebook_entry(indices)

    def set_alignment_scale(self, scale: float):
        self.alignment_scale = max(float(scale), 0.0)

    def get_alignment_scale(self) -> float:
        return float(self.alignment_scale)
