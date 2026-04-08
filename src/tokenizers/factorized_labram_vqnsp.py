"""
Factorized LaBraM-style tokenizer with shared/private latent branches for EEG and fNIRS.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import AlignmentLoss
from src.losses.multimodal_tokenizer import (
    align_pair,
    batch_usage_entropy_loss,
    compute_factorized_shared_alignment_losses,
    coupling_kl_loss,
    orthogonality_loss,
    smooth_signal,
    symmetric_hard_assignment_ce,
    symmetric_prob_kl,
)

from .base import BaseTokenizer
from .labram_vqnsp import NormEMAVectorQuantizer, TransformerDecoder, TransformerEncoder, l2norm
from .shared_labram_vqnsp import MultiChannelPatchEmbedding


class FactorizedLaBraMVQNSP(BaseTokenizer):
    """Shared/private factorized tokenizer for EEG-fNIRS alignment and reconstruction."""

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
        shared_codebook_size: int = 128,
        shared_codebook_dim: int = 48,
        eeg_private_codebook_size: int = 256,
        eeg_private_codebook_dim: int = 64,
        fnirs_private_codebook_size: int = 128,
        fnirs_private_codebook_dim: int = 48,
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
        latent_alignment_weight: float = 0.05,
        coupling_weight: float = 0.05,
        assignment_alignment_weight: float = 0.0,
        hard_assignment_alignment_weight: float = 0.0,
        shared_entropy_weight: float = 0.0,
        private_entropy_weight: float = 0.0,
        shared_eeg_recon_weight: float = 0.0,
        shared_fnirs_recon_weight: float = 0.0,
        shared_eeg_common_weight: float = 0.0,
        shared_fnirs_common_weight: float = 0.0,
        eeg_private_residual_weight: float = 0.0,
        fnirs_private_residual_weight: float = 0.0,
        eeg_common_pool_kernel: int = 400,
        fnirs_common_pool_kernel: int = 20,
        coupling_bidirectional: bool = True,
        orthogonality_weight: float = 0.01,
        assignment_temperature: float = 0.2,
        alignment_lag_candidates: List[int] | None = None,
        alignment_selection: str = 'min',
        alignment_compare_mode: str = 'variable',
        shared_branch_dropout: float = 0.0,
        eeg_private_branch_dropout: float = 0.0,
        fnirs_private_branch_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        use_smooth_l1: bool = False,
        **kwargs,
    ):
        super().__init__(input_dim=2, latent_dim=shared_codebook_dim)

        if eeg_seq_length % eeg_patch_size != 0:
            raise ValueError('eeg_seq_length must be divisible by eeg_patch_size')
        if fnirs_seq_length % fnirs_patch_size != 0:
            raise ValueError('fnirs_seq_length must be divisible by fnirs_patch_size')

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
                'EEG and fNIRS must produce the same token count per window '
                f'(got EEG={self.eeg_n_patches}, fNIRS={self.fnirs_n_patches})'
            )
        self.n_patches = self.eeg_n_patches
        self.shared_codebook_size = shared_codebook_size
        self.codebook_size = shared_codebook_size
        self.shared_codebook_dim = shared_codebook_dim
        self.eeg_private_codebook_size = eeg_private_codebook_size
        self.eeg_private_codebook_dim = eeg_private_codebook_dim
        self.fnirs_private_codebook_size = fnirs_private_codebook_size
        self.fnirs_private_codebook_dim = fnirs_private_codebook_dim
        self.eeg_fft_size = eeg_patch_size // 2 + 1
        self.fnirs_fft_size = fnirs_patch_size // 2 + 1
        self.eeg_amplitude_weight = eeg_amplitude_weight
        self.eeg_phase_weight = eeg_phase_weight
        self.eeg_time_weight = eeg_time_weight
        self.fnirs_amplitude_weight = fnirs_amplitude_weight
        self.fnirs_phase_weight = fnirs_phase_weight
        self.fnirs_time_weight = fnirs_time_weight
        self.latent_alignment_weight = latent_alignment_weight
        self.coupling_weight = coupling_weight
        self.assignment_alignment_weight = assignment_alignment_weight
        self.hard_assignment_alignment_weight = hard_assignment_alignment_weight
        self.shared_entropy_weight = shared_entropy_weight
        self.private_entropy_weight = private_entropy_weight
        self.shared_eeg_recon_weight = shared_eeg_recon_weight
        self.shared_fnirs_recon_weight = shared_fnirs_recon_weight
        self.shared_eeg_common_weight = shared_eeg_common_weight
        self.shared_fnirs_common_weight = shared_fnirs_common_weight
        self.eeg_private_residual_weight = eeg_private_residual_weight
        self.fnirs_private_residual_weight = fnirs_private_residual_weight
        self.eeg_common_pool_kernel = max(int(eeg_common_pool_kernel), 1)
        self.fnirs_common_pool_kernel = max(int(fnirs_common_pool_kernel), 1)
        self.coupling_bidirectional = bool(coupling_bidirectional)
        self.orthogonality_weight = orthogonality_weight
        self.assignment_temperature = assignment_temperature
        self.alignment_lag_candidates = sorted({max(int(lag), 0) for lag in (alignment_lag_candidates or [0])})
        if not self.alignment_lag_candidates:
            self.alignment_lag_candidates = [0]
        if alignment_selection not in {'min', 'mean'}:
            raise ValueError("alignment_selection must be 'min' or 'mean'")
        if alignment_compare_mode not in {'variable', 'fixed_min'}:
            raise ValueError("alignment_compare_mode must be 'variable' or 'fixed_min'")
        self.alignment_selection = alignment_selection
        self.alignment_compare_mode = alignment_compare_mode
        self.fixed_alignment_compare_length = None
        if self.alignment_compare_mode == 'fixed_min':
            min_usable = min(self.n_patches - lag for lag in self.alignment_lag_candidates)
            if min_usable <= 0:
                raise ValueError(
                    'alignment_lag_candidates leave no usable tokens under fixed_min comparison '
                    f'(n_patches={self.n_patches}, lags={self.alignment_lag_candidates})'
                )
            self.fixed_alignment_compare_length = int(min_usable)
        self.shared_branch_dropout = max(float(shared_branch_dropout), 0.0)
        self.eeg_private_branch_dropout = max(float(eeg_private_branch_dropout), 0.0)
        self.fnirs_private_branch_dropout = max(float(fnirs_private_branch_dropout), 0.0)
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

        self.eeg_shared_proj = nn.Sequential(
            nn.Linear(eeg_encoder_embed_dim, eeg_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(eeg_encoder_embed_dim, shared_codebook_dim),
        )
        self.eeg_private_proj = nn.Sequential(
            nn.Linear(eeg_encoder_embed_dim, eeg_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(eeg_encoder_embed_dim, eeg_private_codebook_dim),
        )
        self.fnirs_shared_proj = nn.Sequential(
            nn.Linear(fnirs_encoder_embed_dim, fnirs_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(fnirs_encoder_embed_dim, shared_codebook_dim),
        )
        self.fnirs_private_proj = nn.Sequential(
            nn.Linear(fnirs_encoder_embed_dim, fnirs_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(fnirs_encoder_embed_dim, fnirs_private_codebook_dim),
        )

        self.shared_quantizer = NormEMAVectorQuantizer(
            n_embed=shared_codebook_size,
            embedding_dim=shared_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
        )
        self.eeg_private_quantizer = NormEMAVectorQuantizer(
            n_embed=eeg_private_codebook_size,
            embedding_dim=eeg_private_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
        )
        self.fnirs_private_quantizer = NormEMAVectorQuantizer(
            n_embed=fnirs_private_codebook_size,
            embedding_dim=fnirs_private_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
        )

        self.quantizer = self.shared_quantizer
        self.coupling_logits = nn.Parameter(
            torch.zeros(len(self.alignment_lag_candidates), shared_codebook_size, shared_codebook_size)
        )

        self.eeg_decode_input_proj = nn.Linear(shared_codebook_dim + eeg_private_codebook_dim, eeg_decoder_embed_dim)
        self.fnirs_decode_input_proj = nn.Linear(shared_codebook_dim + fnirs_private_codebook_dim, fnirs_decoder_embed_dim)

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

    def _reconstruct_time(self, amplitude: torch.Tensor, phase: torch.Tensor, patch_size: int) -> torch.Tensor:
        amp = torch.exp(amplitude)
        pha = phase * math.pi
        real = amp * torch.cos(pha)
        imag = amp * torch.sin(pha)
        fft = torch.complex(real, imag)
        patches = torch.fft.irfft(fft, n=patch_size, dim=-1)
        return patches.permute(0, 2, 1, 3).contiguous().view(patches.shape[0], patches.shape[2], -1)

    def _encode_modality(self, x: torch.Tensor, patch_size: int, patch_embed: nn.Module, encoder: nn.Module) -> torch.Tensor:
        patches = self._split_to_patches(x, patch_size)
        embeddings = patch_embed(patches)
        encoded = encoder(embeddings)
        return encoded

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

    def decode_from_components(
        self,
        eeg_shared_q: torch.Tensor,
        eeg_private_q: torch.Tensor,
        fnirs_shared_q: torch.Tensor,
        fnirs_private_q: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        eeg_decoder_latent = torch.cat([eeg_shared_q, eeg_private_q], dim=-1)
        fnirs_decoder_latent = torch.cat([fnirs_shared_q, fnirs_private_q], dim=-1)

        eeg_pred_amp, eeg_pred_phase = self._decode_modality(
            eeg_decoder_latent,
            self.eeg_decode_input_proj,
            self.eeg_decoder,
            self.eeg_amplitude_head,
            self.eeg_phase_head,
            self.eeg_channels,
            self.eeg_fft_size,
        )
        fnirs_pred_amp, fnirs_pred_phase = self._decode_modality(
            fnirs_decoder_latent,
            self.fnirs_decode_input_proj,
            self.fnirs_decoder,
            self.fnirs_amplitude_head,
            self.fnirs_phase_head,
            self.fnirs_channels,
            self.fnirs_fft_size,
        )
        return {
            'eeg_reconstructed': self._reconstruct_time(eeg_pred_amp, eeg_pred_phase, self.eeg_patch_size),
            'fnirs_reconstructed': self._reconstruct_time(fnirs_pred_amp, fnirs_pred_phase, self.fnirs_patch_size),
            'eeg_pred_amp': eeg_pred_amp,
            'eeg_pred_phase': eeg_pred_phase,
            'fnirs_pred_amp': fnirs_pred_amp,
            'fnirs_pred_phase': fnirs_pred_phase,
        }

    @torch.no_grad()
    def reconstruct_with_component_masks(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        use_shared: bool = True,
        use_private: bool = True,
    ) -> Dict[str, torch.Tensor]:
        latents = self.encode_modalities(eeg, fnirs)
        eeg_shared = latents['eeg_shared']
        eeg_private = latents['eeg_private']
        fnirs_shared = latents['fnirs_shared']
        fnirs_private = latents['fnirs_private']

        shared_joint = torch.cat([eeg_shared, fnirs_shared], dim=0)
        shared_q_joint, _, _ = self.shared_quantizer(shared_joint)
        eeg_shared_q, fnirs_shared_q = torch.split(shared_q_joint, [eeg.shape[0], fnirs.shape[0]], dim=0)
        eeg_private_q, _, _ = self.eeg_private_quantizer(eeg_private)
        fnirs_private_q, _, _ = self.fnirs_private_quantizer(fnirs_private)

        if not use_shared:
            eeg_shared_q = torch.zeros_like(eeg_shared_q)
            fnirs_shared_q = torch.zeros_like(fnirs_shared_q)
        if not use_private:
            eeg_private_q = torch.zeros_like(eeg_private_q)
            fnirs_private_q = torch.zeros_like(fnirs_private_q)

        recon = self.decode_from_components(eeg_shared_q, eeg_private_q, fnirs_shared_q, fnirs_private_q)
        recon.update({
            'eeg_shared_q': eeg_shared_q,
            'eeg_private_q': eeg_private_q,
            'fnirs_shared_q': fnirs_shared_q,
            'fnirs_private_q': fnirs_private_q,
        })
        return recon

    def get_analysis_type(self) -> str:
        return 'factorized_alignment'

    def _assignment_logits(self, z: torch.Tensor, codebook_weight: torch.Tensor) -> torch.Tensor:
        normalized_z = l2norm(z)
        return torch.einsum('bnd,kd->bnk', normalized_z, codebook_weight)

    def _align_pair(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor, lag: int, target_length: int | None = None):
        return align_pair(tensor_a, tensor_b, lag, target_length=target_length)

    def _branch_dropout(self, z: torch.Tensor, p: float) -> torch.Tensor:
        if (not self.training) or p <= 0.0:
            return z
        keep_prob = 1.0 - p
        mask = torch.bernoulli(torch.full((z.shape[0], 1, 1), keep_prob, device=z.device, dtype=z.dtype))
        return z * mask / max(keep_prob, 1e-6)

    def _orthogonality_loss(self, shared_z: torch.Tensor, private_z: torch.Tensor) -> torch.Tensor:
        return orthogonality_loss(shared_z, private_z)

    def _coupling_kl(self, pred_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        return coupling_kl_loss(pred_probs, target_probs)

    def _batch_usage_entropy_loss(self, probs: torch.Tensor) -> torch.Tensor:
        return batch_usage_entropy_loss(probs)

    def _smooth_signal(self, signal: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return smooth_signal(signal, kernel_size)

    def _symmetric_prob_kl(self, probs_a: torch.Tensor, probs_b: torch.Tensor) -> torch.Tensor:
        return symmetric_prob_kl(probs_a, probs_b)

    def _symmetric_hard_assignment_ce(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        return symmetric_hard_assignment_ce(
            logits_a,
            logits_b,
            temperature=self.assignment_temperature,
        )

    def _compute_shared_alignment_losses(
        self,
        z_eeg_shared: torch.Tensor,
        z_fnirs_shared: torch.Tensor,
        eeg_shared_logits: torch.Tensor,
        fnirs_shared_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return compute_factorized_shared_alignment_losses(
            z_eeg_shared,
            z_fnirs_shared,
            eeg_shared_logits,
            fnirs_shared_logits,
            alignment_loss=self.alignment_loss,
            coupling_logits=self.coupling_logits,
            alignment_lag_candidates=self.alignment_lag_candidates,
            alignment_selection=self.alignment_selection,
            alignment_compare_mode=self.alignment_compare_mode,
            fixed_alignment_compare_length=self.fixed_alignment_compare_length,
            assignment_temperature=self.assignment_temperature,
            latent_alignment_weight=self.latent_alignment_weight,
            coupling_weight=self.coupling_weight,
            assignment_alignment_weight=self.assignment_alignment_weight,
            hard_assignment_alignment_weight=self.hard_assignment_alignment_weight,
            coupling_bidirectional=self.coupling_bidirectional,
        )

    def _match_rate_at_lag(self, eeg_indices: torch.Tensor, fnirs_indices: torch.Tensor, lag: int) -> torch.Tensor:
        aligned_eeg, aligned_fnirs = self._align_pair(eeg_indices, fnirs_indices, lag)
        if aligned_eeg.shape[1] == 0:
            return torch.tensor(0.0, device=eeg_indices.device, dtype=torch.float32)
        return (aligned_eeg == aligned_fnirs).float().mean()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Use encode_modalities(eeg, fnirs) for the factorized tokenizer')

    def encode_modalities(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        eeg_encoded = self._encode_modality(eeg, self.eeg_patch_size, self.eeg_patch_embed, self.eeg_encoder)
        fnirs_encoded = self._encode_modality(fnirs, self.fnirs_patch_size, self.fnirs_patch_embed, self.fnirs_encoder)
        return {
            'eeg_shared': self.eeg_shared_proj(eeg_encoded),
            'eeg_private': self.eeg_private_proj(eeg_encoded),
            'fnirs_shared': self.fnirs_shared_proj(fnirs_encoded),
            'fnirs_private': self.fnirs_private_proj(fnirs_encoded),
        }

    def quantize(self, z: torch.Tensor):
        return self.shared_quantizer(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Use modality-specific decode paths for the factorized tokenizer')

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if eeg.dim() != 3 or fnirs.dim() != 3:
            raise ValueError('Expected eeg and fnirs tensors with shape [B, C, T]')
        if eeg.shape[-1] != self.eeg_seq_length:
            raise ValueError(f'Expected EEG length {self.eeg_seq_length}, got {eeg.shape[-1]}')
        if fnirs.shape[-1] != self.fnirs_seq_length:
            raise ValueError(f'Expected fNIRS length {self.fnirs_seq_length}, got {fnirs.shape[-1]}')

        eeg_patches = self._split_to_patches(eeg, self.eeg_patch_size)
        fnirs_patches = self._split_to_patches(fnirs, self.fnirs_patch_size)
        eeg_target_amp, eeg_target_phase = self._compute_fft_targets(eeg_patches)
        fnirs_target_amp, fnirs_target_phase = self._compute_fft_targets(fnirs_patches)

        latents = self.encode_modalities(eeg, fnirs)
        eeg_shared = latents['eeg_shared']
        eeg_private = latents['eeg_private']
        fnirs_shared = latents['fnirs_shared']
        fnirs_private = latents['fnirs_private']

        shared_joint = torch.cat([eeg_shared, fnirs_shared], dim=0)
        shared_q_joint, shared_idx_joint, shared_info = self.shared_quantizer(shared_joint)
        eeg_shared_q, fnirs_shared_q = torch.split(shared_q_joint, [eeg.shape[0], fnirs.shape[0]], dim=0)
        eeg_shared_indices, fnirs_shared_indices = torch.split(shared_idx_joint, [eeg.shape[0], fnirs.shape[0]], dim=0)

        eeg_private_q, eeg_private_indices, eeg_private_info = self.eeg_private_quantizer(eeg_private)
        fnirs_private_q, fnirs_private_indices, fnirs_private_info = self.fnirs_private_quantizer(fnirs_private)

        eeg_shared_q = self._branch_dropout(eeg_shared_q, self.shared_branch_dropout)
        fnirs_shared_q = self._branch_dropout(fnirs_shared_q, self.shared_branch_dropout)
        eeg_private_q = self._branch_dropout(eeg_private_q, self.eeg_private_branch_dropout)
        fnirs_private_q = self._branch_dropout(fnirs_private_q, self.fnirs_private_branch_dropout)

        eeg_shared_logits = self._assignment_logits(eeg_shared, self.shared_quantizer.weight)
        fnirs_shared_logits = self._assignment_logits(fnirs_shared, self.shared_quantizer.weight)
        eeg_private_logits = self._assignment_logits(eeg_private, self.eeg_private_quantizer.weight)
        fnirs_private_logits = self._assignment_logits(fnirs_private, self.fnirs_private_quantizer.weight)
        alignment_losses = self._compute_shared_alignment_losses(
            eeg_shared,
            fnirs_shared,
            eeg_shared_logits,
            fnirs_shared_logits,
        )
        latent_align_loss = alignment_losses['latent_align_loss']
        coupling_loss = alignment_losses['coupling_loss']
        assignment_align_loss = alignment_losses['assignment_align_loss']
        hard_assignment_align_loss = alignment_losses['hard_assignment_align_loss']
        shared_entropy_loss = alignment_losses['shared_entropy_loss']
        selected_lag = alignment_losses['selected_lag']
        alignment_usable_tokens = alignment_losses['alignment_usable_tokens']
        eeg_private_probs = F.softmax(eeg_private_logits / max(self.assignment_temperature, 1e-3), dim=-1)
        fnirs_private_probs = F.softmax(fnirs_private_logits / max(self.assignment_temperature, 1e-3), dim=-1)
        private_entropy_loss = 0.5 * (
            self._batch_usage_entropy_loss(eeg_private_probs) +
            self._batch_usage_entropy_loss(fnirs_private_probs)
        )

        reconstructions = self.decode_from_components(eeg_shared_q, eeg_private_q, fnirs_shared_q, fnirs_private_q)
        eeg_pred_amp = reconstructions['eeg_pred_amp']
        eeg_pred_phase = reconstructions['eeg_pred_phase']
        fnirs_pred_amp = reconstructions['fnirs_pred_amp']
        fnirs_pred_phase = reconstructions['fnirs_pred_phase']

        eeg_amp_loss = self.loss_fn(eeg_pred_amp, eeg_target_amp)
        eeg_phase_loss = self.loss_fn(eeg_pred_phase, eeg_target_phase)
        eeg_rec = reconstructions['eeg_reconstructed']
        eeg_time_loss = self.loss_fn(eeg_rec, eeg)
        eeg_rec_loss = (
            self.eeg_amplitude_weight * eeg_amp_loss +
            self.eeg_phase_weight * eeg_phase_loss +
            self.eeg_time_weight * eeg_time_loss
        )

        fnirs_amp_loss = self.loss_fn(fnirs_pred_amp, fnirs_target_amp)
        fnirs_phase_loss = self.loss_fn(fnirs_pred_phase, fnirs_target_phase)
        fnirs_rec = reconstructions['fnirs_reconstructed']
        fnirs_time_loss = self.loss_fn(fnirs_rec, fnirs)
        fnirs_rec_loss = (
            self.fnirs_amplitude_weight * fnirs_amp_loss +
            self.fnirs_phase_weight * fnirs_phase_loss +
            self.fnirs_time_weight * fnirs_time_loss
        )

        zero_eeg_private_q = torch.zeros_like(eeg_private_q)
        zero_fnirs_private_q = torch.zeros_like(fnirs_private_q)
        shared_only_reconstructions = self.decode_from_components(
            eeg_shared_q,
            zero_eeg_private_q,
            fnirs_shared_q,
            zero_fnirs_private_q,
        )
        shared_eeg_pred_amp = shared_only_reconstructions['eeg_pred_amp']
        shared_eeg_pred_phase = shared_only_reconstructions['eeg_pred_phase']
        shared_fnirs_pred_amp = shared_only_reconstructions['fnirs_pred_amp']
        shared_fnirs_pred_phase = shared_only_reconstructions['fnirs_pred_phase']
        shared_eeg_rec = shared_only_reconstructions['eeg_reconstructed']
        shared_fnirs_rec = shared_only_reconstructions['fnirs_reconstructed']
        shared_eeg_rec_loss = (
            self.eeg_amplitude_weight * self.loss_fn(shared_eeg_pred_amp, eeg_target_amp) +
            self.eeg_phase_weight * self.loss_fn(shared_eeg_pred_phase, eeg_target_phase) +
            self.eeg_time_weight * self.loss_fn(shared_eeg_rec, eeg)
        )
        shared_fnirs_rec_loss = (
            self.fnirs_amplitude_weight * self.loss_fn(shared_fnirs_pred_amp, fnirs_target_amp) +
            self.fnirs_phase_weight * self.loss_fn(shared_fnirs_pred_phase, fnirs_target_phase) +
            self.fnirs_time_weight * self.loss_fn(shared_fnirs_rec, fnirs)
        )

        zero_eeg_shared_q = torch.zeros_like(eeg_shared_q)
        zero_fnirs_shared_q = torch.zeros_like(fnirs_shared_q)
        component_aux_enabled = any(
            weight > 0.0
            for weight in (
                self.shared_eeg_common_weight,
                self.shared_fnirs_common_weight,
                self.eeg_private_residual_weight,
                self.fnirs_private_residual_weight,
            )
        )
        if component_aux_enabled:
            eeg_common_target = self._smooth_signal(eeg, self.eeg_common_pool_kernel)
            fnirs_common_target = self._smooth_signal(fnirs, self.fnirs_common_pool_kernel)
            eeg_residual_target = eeg - eeg_common_target
            fnirs_residual_target = fnirs - fnirs_common_target
        else:
            eeg_common_target = None
            fnirs_common_target = None
            eeg_residual_target = None
            fnirs_residual_target = None

        shared_eeg_common_loss = eeg.new_tensor(0.0)
        shared_fnirs_common_loss = fnirs.new_tensor(0.0)
        eeg_private_residual_loss = eeg.new_tensor(0.0)
        fnirs_private_residual_loss = fnirs.new_tensor(0.0)
        eeg_private_only_rec = None
        fnirs_private_only_rec = None
        if component_aux_enabled:
            private_only_reconstructions = self.decode_from_components(
                zero_eeg_shared_q,
                eeg_private_q,
                zero_fnirs_shared_q,
                fnirs_private_q,
            )
            eeg_private_only_rec = private_only_reconstructions['eeg_reconstructed']
            fnirs_private_only_rec = private_only_reconstructions['fnirs_reconstructed']
            if self.shared_eeg_common_weight > 0.0:
                shared_eeg_common_loss = self.loss_fn(shared_eeg_rec, eeg_common_target)
            if self.shared_fnirs_common_weight > 0.0:
                shared_fnirs_common_loss = self.loss_fn(shared_fnirs_rec, fnirs_common_target)
            if self.eeg_private_residual_weight > 0.0:
                eeg_private_residual_loss = self.loss_fn(eeg_private_only_rec, eeg_residual_target)
            if self.fnirs_private_residual_weight > 0.0:
                fnirs_private_residual_loss = self.loss_fn(fnirs_private_only_rec, fnirs_residual_target)

        orthogonality_loss = self._orthogonality_loss(eeg_shared, eeg_private) + self._orthogonality_loss(fnirs_shared, fnirs_private)
        vq_shared_loss = shared_info['vq_loss']
        vq_eeg_private_loss = eeg_private_info['vq_loss']
        vq_fnirs_private_loss = fnirs_private_info['vq_loss']
        vq_loss = vq_shared_loss + vq_eeg_private_loss + vq_fnirs_private_loss

        total_loss = (
            eeg_rec_loss +
            fnirs_rec_loss +
            vq_loss +
            (self.latent_alignment_weight * self.alignment_scale) * latent_align_loss +
            (self.coupling_weight * self.alignment_scale) * coupling_loss +
            (self.assignment_alignment_weight * self.alignment_scale) * assignment_align_loss +
            (self.hard_assignment_alignment_weight * self.alignment_scale) * hard_assignment_align_loss +
            self.shared_entropy_weight * shared_entropy_loss +
            self.private_entropy_weight * private_entropy_loss +
            self.shared_eeg_recon_weight * shared_eeg_rec_loss +
            self.shared_fnirs_recon_weight * shared_fnirs_rec_loss +
            self.shared_eeg_common_weight * shared_eeg_common_loss +
            self.shared_fnirs_common_weight * shared_fnirs_common_loss +
            self.eeg_private_residual_weight * eeg_private_residual_loss +
            self.fnirs_private_residual_weight * fnirs_private_residual_loss +
            self.orthogonality_weight * orthogonality_loss
        )

        token_match = (eeg_shared_indices == fnirs_shared_indices).float().mean()
        best_lag_token_match = self._match_rate_at_lag(eeg_shared_indices, fnirs_shared_indices, int(selected_lag.item()))
        eeg_unique = torch.unique(eeg_shared_indices)
        fnirs_unique = torch.unique(fnirs_shared_indices)
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
            'vq_shared_loss': vq_shared_loss,
            'vq_eeg_private_loss': vq_eeg_private_loss,
            'vq_fnirs_private_loss': vq_fnirs_private_loss,
            'latent_align_loss': latent_align_loss,
            'coupling_loss': coupling_loss,
            'assignment_align_loss': assignment_align_loss,
            'hard_assignment_align_loss': hard_assignment_align_loss,
            'shared_entropy_loss': shared_entropy_loss,
            'private_entropy_loss': private_entropy_loss,
            'shared_eeg_rec_loss': shared_eeg_rec_loss,
            'shared_fnirs_rec_loss': shared_fnirs_rec_loss,
            'shared_eeg_common_loss': shared_eeg_common_loss,
            'shared_fnirs_common_loss': shared_fnirs_common_loss,
            'eeg_private_residual_loss': eeg_private_residual_loss,
            'fnirs_private_residual_loss': fnirs_private_residual_loss,
            'orthogonality_loss': orthogonality_loss,
            'selected_alignment_lag': selected_lag,
            'alignment_usable_tokens': alignment_usable_tokens,
            'alignment_scale': torch.tensor(self.alignment_scale, device=eeg.device, dtype=torch.float32),
            'token_match': token_match,
            'best_lag_token_match': best_lag_token_match,
            'code_overlap': overlap,
            'perplexity': shared_info['perplexity'],
            'utilization': shared_info['utilization'],
            'shared_perplexity': shared_info['perplexity'],
            'shared_utilization': shared_info['utilization'],
            'eeg_private_perplexity': eeg_private_info['perplexity'],
            'eeg_private_utilization': eeg_private_info['utilization'],
            'fnirs_private_perplexity': fnirs_private_info['perplexity'],
            'fnirs_private_utilization': fnirs_private_info['utilization'],
            'eeg_reconstructed': eeg_rec,
            'fnirs_reconstructed': fnirs_rec,
            'eeg_shared_only_reconstructed': shared_eeg_rec,
            'fnirs_shared_only_reconstructed': shared_fnirs_rec,
            'eeg_private_only_reconstructed': eeg_private_only_rec if eeg_private_only_rec is not None else torch.zeros_like(eeg),
            'fnirs_private_only_reconstructed': fnirs_private_only_rec if fnirs_private_only_rec is not None else torch.zeros_like(fnirs),
            'eeg_indices': eeg_shared_indices,
            'fnirs_indices': fnirs_shared_indices,
            'eeg_private_indices': eeg_private_indices,
            'fnirs_private_indices': fnirs_private_indices,
            'eeg_z': eeg_shared,
            'fnirs_z': fnirs_shared,
            'eeg_private_z': eeg_private,
            'fnirs_private_z': fnirs_private,
            'eeg_z_q': eeg_shared_q,
            'fnirs_z_q': fnirs_shared_q,
            'eeg_private_z_q': eeg_private_q,
            'fnirs_private_z_q': fnirs_private_q,
        }

    def get_codebook_size(self) -> int:
        return self.shared_codebook_size

    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        return self.shared_quantizer.get_codebook_entry(indices)

    def set_alignment_scale(self, scale: float):
        self.alignment_scale = max(float(scale), 0.0)

    def get_alignment_scale(self) -> float:
        return float(self.alignment_scale)

    def get_gradient_component_weights(self) -> Dict[str, float]:
        alignment_scale = float(self.get_alignment_scale())
        return {
            'eeg_rec_loss': 1.0,
            'fnirs_rec_loss': 1.0,
            'vq_loss': 1.0,
            'latent_align_loss': self.latent_alignment_weight * alignment_scale,
            'coupling_loss': self.coupling_weight * alignment_scale,
            'assignment_align_loss': self.assignment_alignment_weight * alignment_scale,
            'hard_assignment_align_loss': self.hard_assignment_alignment_weight * alignment_scale,
            'shared_entropy_loss': self.shared_entropy_weight,
            'private_entropy_loss': self.private_entropy_weight,
            'shared_eeg_rec_loss': self.shared_eeg_recon_weight,
            'shared_fnirs_rec_loss': self.shared_fnirs_recon_weight,
            'shared_eeg_common_loss': self.shared_eeg_common_weight,
            'shared_fnirs_common_loss': self.shared_fnirs_common_weight,
            'eeg_private_residual_loss': self.eeg_private_residual_weight,
            'fnirs_private_residual_loss': self.fnirs_private_residual_weight,
            'orthogonality_loss': self.orthogonality_weight,
        }
