"""Source/observation LaBraM tokenizer with dual source codebooks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.multimodal_tokenizer import (
    align_pair,
    batch_usage_entropy_loss,
    coupling_kl_loss,
    orthogonality_loss,
)

from .base import BaseTokenizer
from .labram_vqnsp import NormEMAVectorQuantizer, TransformerDecoder, TransformerEncoder, l2norm
from .shared_labram_vqnsp import MultiChannelPatchEmbedding


class SourceObservationLaBraMVQNSP(BaseTokenizer):
    """Mainline multimodal tokenizer with source/observation branch semantics."""

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
        source_codebook_size: int = 128,
        eeg_source_codebook_dim: int = 48,
        fnirs_source_codebook_dim: int = 48,
        eeg_observation_codebook_size: int = 256,
        eeg_observation_codebook_dim: int = 64,
        fnirs_observation_codebook_size: int = 128,
        fnirs_observation_codebook_dim: int = 48,
        beta: float = 1.0,
        decay: float = 0.99,
        kmeans_init: bool = True,
        revive_dead_codes: bool = True,
        dead_code_threshold: int = 10,
        eeg_amplitude_weight: float = 1.0,
        eeg_phase_weight: float = 1.0,
        eeg_time_weight: float = 0.9,
        fnirs_amplitude_weight: float = 1.0,
        fnirs_phase_weight: float = 0.2,
        fnirs_time_weight: float = 1.0,
        coupling_weight: float = 0.07,
        codebook_balance_weight: float = 0.02,
        coupling_bidirectional: bool = True,
        orthogonality_weight: float = 0.01,
        assignment_temperature: float = 0.2,
        alignment_lag_candidates: List[int] | None = None,
        alignment_selection: str = 'min',
        alignment_compare_mode: str = 'variable',
        source_branch_dropout: float = 0.0,
        eeg_observation_branch_dropout: float = 0.0,
        fnirs_observation_branch_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        use_smooth_l1: bool = False,
        **kwargs: Any,
    ):
        del kwargs
        super().__init__(input_dim=2, latent_dim=max(eeg_source_codebook_dim, fnirs_source_codebook_dim))

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

        self.source_codebook_size = source_codebook_size
        self.codebook_size = source_codebook_size
        self.eeg_source_codebook_dim = eeg_source_codebook_dim
        self.fnirs_source_codebook_dim = fnirs_source_codebook_dim
        self.eeg_observation_codebook_size = eeg_observation_codebook_size
        self.eeg_observation_codebook_dim = eeg_observation_codebook_dim
        self.fnirs_observation_codebook_size = fnirs_observation_codebook_size
        self.fnirs_observation_codebook_dim = fnirs_observation_codebook_dim

        self.eeg_fft_size = eeg_patch_size // 2 + 1
        self.fnirs_fft_size = fnirs_patch_size // 2 + 1
        self.eeg_amplitude_weight = eeg_amplitude_weight
        self.eeg_phase_weight = eeg_phase_weight
        self.eeg_time_weight = eeg_time_weight
        self.fnirs_amplitude_weight = fnirs_amplitude_weight
        self.fnirs_phase_weight = fnirs_phase_weight
        self.fnirs_time_weight = fnirs_time_weight
        self.coupling_weight = coupling_weight
        self.codebook_balance_weight = codebook_balance_weight
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

        self.source_branch_dropout = max(float(source_branch_dropout), 0.0)
        self.eeg_observation_branch_dropout = max(float(eeg_observation_branch_dropout), 0.0)
        self.fnirs_observation_branch_dropout = max(float(fnirs_observation_branch_dropout), 0.0)
        self.alignment_scale = 1.0
        self.loss_fn = F.smooth_l1_loss if use_smooth_l1 else F.mse_loss

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

        self.eeg_source_proj = nn.Sequential(
            nn.Linear(eeg_encoder_embed_dim, eeg_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(eeg_encoder_embed_dim, eeg_source_codebook_dim),
        )
        self.eeg_observation_proj = nn.Sequential(
            nn.Linear(eeg_encoder_embed_dim, eeg_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(eeg_encoder_embed_dim, eeg_observation_codebook_dim),
        )
        self.fnirs_source_proj = nn.Sequential(
            nn.Linear(fnirs_encoder_embed_dim, fnirs_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(fnirs_encoder_embed_dim, fnirs_source_codebook_dim),
        )
        self.fnirs_observation_proj = nn.Sequential(
            nn.Linear(fnirs_encoder_embed_dim, fnirs_encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(fnirs_encoder_embed_dim, fnirs_observation_codebook_dim),
        )

        self.eeg_source_quantizer = NormEMAVectorQuantizer(
            n_embed=source_codebook_size,
            embedding_dim=eeg_source_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
        )
        self.fnirs_source_quantizer = NormEMAVectorQuantizer(
            n_embed=source_codebook_size,
            embedding_dim=fnirs_source_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
        )
        self.eeg_observation_quantizer = NormEMAVectorQuantizer(
            n_embed=eeg_observation_codebook_size,
            embedding_dim=eeg_observation_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
        )
        self.fnirs_observation_quantizer = NormEMAVectorQuantizer(
            n_embed=fnirs_observation_codebook_size,
            embedding_dim=fnirs_observation_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
        )

        self.quantizer = self.eeg_source_quantizer
        self.coupling_logits = nn.Parameter(
            torch.zeros(len(self.alignment_lag_candidates), source_codebook_size, source_codebook_size)
        )

        self.eeg_decode_input_proj = nn.Linear(
            eeg_source_codebook_dim + eeg_observation_codebook_dim,
            eeg_decoder_embed_dim,
        )
        self.fnirs_decode_input_proj = nn.Linear(
            fnirs_source_codebook_dim + fnirs_observation_codebook_dim,
            fnirs_decoder_embed_dim,
        )

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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SourceObservationLaBraMVQNSP':
        model_cfg = config.get('model', {})
        eeg_cfg = model_cfg.get('eeg', {})
        fnirs_cfg = model_cfg.get('fnirs', {})
        source_cfg = model_cfg.get('source', {})
        eeg_observation_cfg = model_cfg.get('eeg_observation', {})
        fnirs_observation_cfg = model_cfg.get('fnirs_observation', {})
        branch_dropout_cfg = model_cfg.get('branch_dropout', {})
        quantizer_cfg = model_cfg.get('quantizer', {})
        loss_cfg = config.get('loss', {})
        reconstruction_cfg = loss_cfg.get('reconstruction', {})
        coupling_cfg = loss_cfg.get('coupling', {})
        branch_cfg = loss_cfg.get('branch', {})
        codebook_cfg = loss_cfg.get('codebook', {})
        validation_cfg = config.get('validation', {})

        source_codebook_size = source_cfg.get('codebook_size', 128)
        return cls(
            eeg_seq_length=eeg_cfg.get('seq_length', 2000),
            eeg_patch_size=eeg_cfg.get('patch_size', 400),
            eeg_channels=eeg_cfg.get('channels', 30),
            eeg_encoder_embed_dim=eeg_cfg.get('encoder_embed_dim', 256),
            eeg_encoder_depth=eeg_cfg.get('encoder_depth', 8),
            eeg_encoder_num_heads=eeg_cfg.get('encoder_num_heads', 8),
            eeg_decoder_embed_dim=eeg_cfg.get('decoder_embed_dim', 256),
            eeg_decoder_depth=eeg_cfg.get('decoder_depth', 4),
            eeg_decoder_num_heads=eeg_cfg.get('decoder_num_heads', 8),
            fnirs_seq_length=fnirs_cfg.get('seq_length', 100),
            fnirs_patch_size=fnirs_cfg.get('patch_size', 20),
            fnirs_channels=fnirs_cfg.get('channels', 36),
            fnirs_encoder_embed_dim=fnirs_cfg.get('encoder_embed_dim', 160),
            fnirs_encoder_depth=fnirs_cfg.get('encoder_depth', 6),
            fnirs_encoder_num_heads=fnirs_cfg.get('encoder_num_heads', 4),
            fnirs_decoder_embed_dim=fnirs_cfg.get('decoder_embed_dim', 160),
            fnirs_decoder_depth=fnirs_cfg.get('decoder_depth', 3),
            fnirs_decoder_num_heads=fnirs_cfg.get('decoder_num_heads', 4),
            source_codebook_size=source_codebook_size,
            eeg_source_codebook_dim=source_cfg.get('eeg_codebook_dim', source_cfg.get('codebook_dim', 48)),
            fnirs_source_codebook_dim=source_cfg.get('fnirs_codebook_dim', source_cfg.get('codebook_dim', 48)),
            eeg_observation_codebook_size=eeg_observation_cfg.get('codebook_size', 256),
            eeg_observation_codebook_dim=eeg_observation_cfg.get('codebook_dim', 64),
            fnirs_observation_codebook_size=fnirs_observation_cfg.get('codebook_size', 128),
            fnirs_observation_codebook_dim=fnirs_observation_cfg.get('codebook_dim', 48),
            beta=quantizer_cfg.get('beta', 1.0),
            decay=quantizer_cfg.get('decay', 0.99),
            kmeans_init=quantizer_cfg.get('kmeans_init', True),
            revive_dead_codes=quantizer_cfg.get('revive_dead_codes', True),
            dead_code_threshold=quantizer_cfg.get('dead_code_threshold', 10),
            eeg_amplitude_weight=reconstruction_cfg.get('eeg_amplitude_weight', 1.0),
            eeg_phase_weight=reconstruction_cfg.get('eeg_phase_weight', 1.0),
            eeg_time_weight=reconstruction_cfg.get('eeg_time_weight', 0.9),
            fnirs_amplitude_weight=reconstruction_cfg.get('fnirs_amplitude_weight', 1.0),
            fnirs_phase_weight=reconstruction_cfg.get('fnirs_phase_weight', 0.2),
            fnirs_time_weight=reconstruction_cfg.get('fnirs_time_weight', 1.0),
            coupling_weight=coupling_cfg.get('weight', 0.07),
            codebook_balance_weight=codebook_cfg.get('balance_weight', 0.02),
            coupling_bidirectional=coupling_cfg.get('bidirectional', True),
            orthogonality_weight=branch_cfg.get('orthogonality_weight', 0.01),
            assignment_temperature=coupling_cfg.get('temperature', 0.2),
            alignment_lag_candidates=coupling_cfg.get('lag_candidates', validation_cfg.get('lag_set', [0])),
            alignment_selection=coupling_cfg.get('selection', 'min'),
            alignment_compare_mode=coupling_cfg.get('compare_mode', 'variable'),
            source_branch_dropout=branch_dropout_cfg.get('source', 0.0),
            eeg_observation_branch_dropout=branch_dropout_cfg.get('eeg_observation', 0.0),
            fnirs_observation_branch_dropout=branch_dropout_cfg.get('fnirs_observation', 0.0),
            dropout=model_cfg.get('dropout', 0.0),
            drop_path=model_cfg.get('drop_path', 0.1),
            use_smooth_l1=loss_cfg.get('use_smooth_l1', False),
        )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _split_to_patches(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        batch_size, channels, seq_len = x.shape
        n_patches = seq_len // patch_size
        return x.view(batch_size, channels, n_patches, patch_size).permute(0, 2, 1, 3).contiguous()

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
        return encoder(embeddings)

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

    def _assignment_logits(self, z: torch.Tensor, codebook_weight: torch.Tensor) -> torch.Tensor:
        normalized_z = l2norm(z)
        return torch.einsum('bnd,kd->bnk', normalized_z, codebook_weight)

    def _branch_dropout(self, z: torch.Tensor, p: float) -> torch.Tensor:
        if (not self.training) or p <= 0.0:
            return z
        keep_prob = 1.0 - p
        mask = torch.bernoulli(torch.full((z.shape[0], 1, 1), keep_prob, device=z.device, dtype=z.dtype))
        return z * mask / max(keep_prob, 1e-6)

    def _code_overlap(self, left_indices: torch.Tensor, right_indices: torch.Tensor) -> torch.Tensor:
        left_codes = set(torch.unique(left_indices).tolist())
        right_codes = set(torch.unique(right_indices).tolist())
        union_size = max(len(left_codes | right_codes), 1)
        overlap = len(left_codes & right_codes) / union_size
        return torch.tensor(overlap, device=left_indices.device, dtype=torch.float32)

    def _compute_source_coupling_loss(
        self,
        eeg_source_logits: torch.Tensor,
        fnirs_source_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        temperature = max(float(self.assignment_temperature), 1e-3)
        eeg_source_probs = F.softmax(eeg_source_logits / temperature, dim=-1)
        fnirs_source_probs = F.softmax(fnirs_source_logits / temperature, dim=-1)

        target_length = self.fixed_alignment_compare_length if self.alignment_compare_mode == 'fixed_min' else None
        coupling_losses = []
        valid_lags = []
        usable_lengths = []

        for lag_index, lag in enumerate(self.alignment_lag_candidates):
            aligned_eeg_probs, aligned_fnirs_probs = align_pair(
                eeg_source_probs,
                fnirs_source_probs,
                lag,
                target_length=target_length,
            )
            if aligned_eeg_probs.shape[1] == 0:
                continue

            transition = F.softmax(self.coupling_logits[lag_index], dim=-1)
            pred_fnirs_probs = torch.einsum('bnk,kl->bnl', aligned_eeg_probs, transition)
            coupling_loss = coupling_kl_loss(pred_fnirs_probs, aligned_fnirs_probs)
            if self.coupling_bidirectional:
                reverse_transition = F.softmax(self.coupling_logits[lag_index].transpose(0, 1), dim=-1)
                pred_eeg_probs = torch.einsum('bnk,kl->bnl', aligned_fnirs_probs, reverse_transition)
                coupling_loss = 0.5 * (coupling_loss + coupling_kl_loss(pred_eeg_probs, aligned_eeg_probs))

            coupling_losses.append(coupling_loss)
            valid_lags.append(lag)
            usable_lengths.append(aligned_eeg_probs.shape[1])

        zero = eeg_source_logits.new_tensor(0.0)
        if not coupling_losses:
            return {
                'source_coupling_loss': zero,
                'selected_source_lag': zero,
                'selected_alignment_lag': zero,
                'source_alignment_usable_tokens': zero,
                'alignment_usable_tokens': zero,
                'eeg_source_probs': eeg_source_probs,
                'fnirs_source_probs': fnirs_source_probs,
            }

        if self.alignment_selection == 'mean':
            source_coupling_loss = torch.stack(coupling_losses).mean()
            selected_source_lag = eeg_source_logits.new_tensor(float(sum(valid_lags) / len(valid_lags)))
            source_alignment_usable_tokens = eeg_source_logits.new_tensor(float(sum(usable_lengths) / len(usable_lengths)))
        else:
            best_index = int(torch.argmin(torch.stack(coupling_losses)).item())
            source_coupling_loss = coupling_losses[best_index]
            selected_source_lag = eeg_source_logits.new_tensor(float(valid_lags[best_index]))
            source_alignment_usable_tokens = eeg_source_logits.new_tensor(float(usable_lengths[best_index]))

        return {
            'source_coupling_loss': source_coupling_loss,
            'selected_source_lag': selected_source_lag,
            'selected_alignment_lag': selected_source_lag,
            'source_alignment_usable_tokens': source_alignment_usable_tokens,
            'alignment_usable_tokens': source_alignment_usable_tokens,
            'eeg_source_probs': eeg_source_probs,
            'fnirs_source_probs': fnirs_source_probs,
        }

    def decode_from_components(
        self,
        eeg_source_q: torch.Tensor,
        eeg_observation_q: torch.Tensor,
        fnirs_source_q: torch.Tensor,
        fnirs_observation_q: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        eeg_decoder_latent = torch.cat([eeg_source_q, eeg_observation_q], dim=-1)
        fnirs_decoder_latent = torch.cat([fnirs_source_q, fnirs_observation_q], dim=-1)

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
        use_source: bool = True,
        use_observation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        latents = self.encode_modalities(eeg, fnirs)
        eeg_source = latents['eeg_source']
        eeg_observation = latents['eeg_observation']
        fnirs_source = latents['fnirs_source']
        fnirs_observation = latents['fnirs_observation']

        eeg_source_q, _, _ = self.eeg_source_quantizer(eeg_source)
        fnirs_source_q, _, _ = self.fnirs_source_quantizer(fnirs_source)
        eeg_observation_q, _, _ = self.eeg_observation_quantizer(eeg_observation)
        fnirs_observation_q, _, _ = self.fnirs_observation_quantizer(fnirs_observation)

        if not use_source:
            eeg_source_q = torch.zeros_like(eeg_source_q)
            fnirs_source_q = torch.zeros_like(fnirs_source_q)
        if not use_observation:
            eeg_observation_q = torch.zeros_like(eeg_observation_q)
            fnirs_observation_q = torch.zeros_like(fnirs_observation_q)

        reconstructions = self.decode_from_components(
            eeg_source_q,
            eeg_observation_q,
            fnirs_source_q,
            fnirs_observation_q,
        )
        reconstructions.update({
            'eeg_source_q': eeg_source_q,
            'eeg_observation_q': eeg_observation_q,
            'fnirs_source_q': fnirs_source_q,
            'fnirs_observation_q': fnirs_observation_q,
        })
        return reconstructions

    def get_analysis_type(self) -> str:
        return 'source_observation_alignment'

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Use encode_modalities(eeg, fnirs) for the source/observation tokenizer')

    def encode_modalities(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        eeg_encoded = self._encode_modality(eeg, self.eeg_patch_size, self.eeg_patch_embed, self.eeg_encoder)
        fnirs_encoded = self._encode_modality(fnirs, self.fnirs_patch_size, self.fnirs_patch_embed, self.fnirs_encoder)
        return {
            'eeg_source': self.eeg_source_proj(eeg_encoded),
            'eeg_observation': self.eeg_observation_proj(eeg_encoded),
            'fnirs_source': self.fnirs_source_proj(fnirs_encoded),
            'fnirs_observation': self.fnirs_observation_proj(fnirs_encoded),
        }

    def quantize(self, z: torch.Tensor, modality: str = 'eeg_source'):
        quantizers = {
            'eeg_source': self.eeg_source_quantizer,
            'fnirs_source': self.fnirs_source_quantizer,
            'eeg_observation': self.eeg_observation_quantizer,
            'fnirs_observation': self.fnirs_observation_quantizer,
        }
        return quantizers[modality](z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Use modality-specific decode paths for the source/observation tokenizer')

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
        eeg_source = latents['eeg_source']
        eeg_observation = latents['eeg_observation']
        fnirs_source = latents['fnirs_source']
        fnirs_observation = latents['fnirs_observation']

        eeg_source_q, eeg_source_indices, eeg_source_info = self.eeg_source_quantizer(eeg_source)
        fnirs_source_q, fnirs_source_indices, fnirs_source_info = self.fnirs_source_quantizer(fnirs_source)
        eeg_observation_q, eeg_observation_indices, eeg_observation_info = self.eeg_observation_quantizer(eeg_observation)
        fnirs_observation_q, fnirs_observation_indices, fnirs_observation_info = self.fnirs_observation_quantizer(fnirs_observation)

        eeg_source_q = self._branch_dropout(eeg_source_q, self.source_branch_dropout)
        fnirs_source_q = self._branch_dropout(fnirs_source_q, self.source_branch_dropout)
        eeg_observation_q = self._branch_dropout(eeg_observation_q, self.eeg_observation_branch_dropout)
        fnirs_observation_q = self._branch_dropout(fnirs_observation_q, self.fnirs_observation_branch_dropout)

        eeg_source_logits = self._assignment_logits(eeg_source, self.eeg_source_quantizer.weight)
        fnirs_source_logits = self._assignment_logits(fnirs_source, self.fnirs_source_quantizer.weight)
        eeg_observation_logits = self._assignment_logits(eeg_observation, self.eeg_observation_quantizer.weight)
        fnirs_observation_logits = self._assignment_logits(fnirs_observation, self.fnirs_observation_quantizer.weight)

        source_coupling = self._compute_source_coupling_loss(eeg_source_logits, fnirs_source_logits)
        source_coupling_loss = source_coupling['source_coupling_loss']
        selected_source_lag = source_coupling['selected_source_lag']
        alignment_usable_tokens = source_coupling['alignment_usable_tokens']
        eeg_source_probs = source_coupling['eeg_source_probs']
        fnirs_source_probs = source_coupling['fnirs_source_probs']
        eeg_observation_probs = F.softmax(eeg_observation_logits / max(self.assignment_temperature, 1e-3), dim=-1)
        fnirs_observation_probs = F.softmax(fnirs_observation_logits / max(self.assignment_temperature, 1e-3), dim=-1)

        source_balance_loss = 0.5 * (
            batch_usage_entropy_loss(eeg_source_probs) +
            batch_usage_entropy_loss(fnirs_source_probs)
        )
        observation_balance_loss = 0.5 * (
            batch_usage_entropy_loss(eeg_observation_probs) +
            batch_usage_entropy_loss(fnirs_observation_probs)
        )
        codebook_balance_loss = 0.5 * (source_balance_loss + observation_balance_loss)

        reconstructions = self.decode_from_components(
            eeg_source_q,
            eeg_observation_q,
            fnirs_source_q,
            fnirs_observation_q,
        )
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

        zero_eeg_observation_q = torch.zeros_like(eeg_observation_q)
        zero_fnirs_observation_q = torch.zeros_like(fnirs_observation_q)
        source_only_reconstructions = self.decode_from_components(
            eeg_source_q,
            zero_eeg_observation_q,
            fnirs_source_q,
            zero_fnirs_observation_q,
        )
        zero_eeg_source_q = torch.zeros_like(eeg_source_q)
        zero_fnirs_source_q = torch.zeros_like(fnirs_source_q)
        observation_only_reconstructions = self.decode_from_components(
            zero_eeg_source_q,
            eeg_observation_q,
            zero_fnirs_source_q,
            fnirs_observation_q,
        )

        branch_orthogonality_loss = (
            orthogonality_loss(eeg_source, eeg_observation) +
            orthogonality_loss(fnirs_source, fnirs_observation)
        )

        vq_source_loss = eeg_source_info['vq_loss'] + fnirs_source_info['vq_loss']
        vq_observation_loss = eeg_observation_info['vq_loss'] + fnirs_observation_info['vq_loss']
        vq_loss = vq_source_loss + vq_observation_loss

        total_loss = (
            eeg_rec_loss +
            fnirs_rec_loss +
            vq_loss +
            (self.coupling_weight * self.alignment_scale) * source_coupling_loss +
            self.codebook_balance_weight * codebook_balance_loss +
            self.orthogonality_weight * branch_orthogonality_loss
        )

        source_overlap = self._code_overlap(eeg_source_indices, fnirs_source_indices)
        overall_perplexity = 0.5 * (eeg_source_info['perplexity'] + fnirs_source_info['perplexity'])
        overall_utilization = 0.5 * (eeg_source_info['utilization'] + fnirs_source_info['utilization'])

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
            'vq_source_loss': vq_source_loss,
            'vq_observation_loss': vq_observation_loss,
            'source_coupling_loss': source_coupling_loss,
            'codebook_balance_loss': codebook_balance_loss,
            'source_balance_loss': source_balance_loss,
            'observation_balance_loss': observation_balance_loss,
            'orthogonality_loss': branch_orthogonality_loss,
            'selected_source_lag': selected_source_lag,
            'selected_alignment_lag': selected_source_lag,
            'source_alignment_usable_tokens': alignment_usable_tokens,
            'alignment_usable_tokens': alignment_usable_tokens,
            'alignment_scale': torch.tensor(self.alignment_scale, device=eeg.device, dtype=torch.float32),
            'source_code_overlap': source_overlap,
            'perplexity': overall_perplexity,
            'utilization': overall_utilization,
            'eeg_source_perplexity': eeg_source_info['perplexity'],
            'eeg_source_utilization': eeg_source_info['utilization'],
            'fnirs_source_perplexity': fnirs_source_info['perplexity'],
            'fnirs_source_utilization': fnirs_source_info['utilization'],
            'eeg_observation_perplexity': eeg_observation_info['perplexity'],
            'eeg_observation_utilization': eeg_observation_info['utilization'],
            'fnirs_observation_perplexity': fnirs_observation_info['perplexity'],
            'fnirs_observation_utilization': fnirs_observation_info['utilization'],
            'eeg_reconstructed': eeg_rec,
            'fnirs_reconstructed': fnirs_rec,
            'eeg_source_only_reconstructed': source_only_reconstructions['eeg_reconstructed'],
            'fnirs_source_only_reconstructed': source_only_reconstructions['fnirs_reconstructed'],
            'eeg_observation_only_reconstructed': observation_only_reconstructions['eeg_reconstructed'],
            'fnirs_observation_only_reconstructed': observation_only_reconstructions['fnirs_reconstructed'],
            'eeg_source_indices': eeg_source_indices,
            'fnirs_source_indices': fnirs_source_indices,
            'eeg_observation_indices': eeg_observation_indices,
            'fnirs_observation_indices': fnirs_observation_indices,
            'eeg_source_z': eeg_source,
            'fnirs_source_z': fnirs_source,
            'eeg_observation_z': eeg_observation,
            'fnirs_observation_z': fnirs_observation,
            'eeg_source_z_q': eeg_source_q,
            'fnirs_source_z_q': fnirs_source_q,
            'eeg_observation_z_q': eeg_observation_q,
            'fnirs_observation_z_q': fnirs_observation_q,
            'eeg_indices': eeg_source_indices,
            'fnirs_indices': fnirs_source_indices,
            'eeg_z': eeg_source,
            'fnirs_z': fnirs_source,
            'eeg_z_q': eeg_source_q,
            'fnirs_z_q': fnirs_source_q,
        }

    def get_codebook_size(self) -> int:
        return self.source_codebook_size

    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        return self.eeg_source_quantizer.get_codebook_entry(indices)

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
            'source_coupling_loss': self.coupling_weight * alignment_scale,
            'codebook_balance_loss': self.codebook_balance_weight,
            'orthogonality_loss': self.orthogonality_weight,
        }


__all__ = ['SourceObservationLaBraMVQNSP']
