"""
Codebook-focused factorized LaBraM-style tokenizer for aligned EEG and fNIRS.

This mainline keeps the shared/private factorization and lagged shared coupling,
but removes legacy experimental auxiliaries from the optimization path.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .factorized_labram_vqnsp import FactorizedLaBraMVQNSP


class CodebookFocusedFactorizedLaBraMVQNSP(FactorizedLaBraMVQNSP):
    """Simplified factorized tokenizer optimized for codebook quality."""

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
        coupling_weight: float = 0.07,
        codebook_balance_weight: float = 0.02,
        shared_eeg_common_weight: float = 0.18,
        shared_fnirs_common_weight: float = 0.15,
        eeg_private_residual_weight: float = 0.08,
        fnirs_private_residual_weight: float = 0.08,
        eeg_common_pool_kernel: int = 800,
        fnirs_common_pool_kernel: int = 40,
        coupling_bidirectional: bool = True,
        orthogonality_weight: float = 0.01,
        assignment_temperature: float = 0.35,
        alignment_lag_candidates: List[int] | None = None,
        alignment_selection: str = 'min',
        alignment_compare_mode: str = 'fixed_min',
        shared_branch_dropout: float = 0.0,
        eeg_private_branch_dropout: float = 0.0,
        fnirs_private_branch_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        use_smooth_l1: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            eeg_seq_length=eeg_seq_length,
            eeg_patch_size=eeg_patch_size,
            eeg_channels=eeg_channels,
            eeg_encoder_embed_dim=eeg_encoder_embed_dim,
            eeg_encoder_depth=eeg_encoder_depth,
            eeg_encoder_num_heads=eeg_encoder_num_heads,
            eeg_decoder_embed_dim=eeg_decoder_embed_dim,
            eeg_decoder_depth=eeg_decoder_depth,
            eeg_decoder_num_heads=eeg_decoder_num_heads,
            fnirs_seq_length=fnirs_seq_length,
            fnirs_patch_size=fnirs_patch_size,
            fnirs_channels=fnirs_channels,
            fnirs_encoder_embed_dim=fnirs_encoder_embed_dim,
            fnirs_encoder_depth=fnirs_encoder_depth,
            fnirs_encoder_num_heads=fnirs_encoder_num_heads,
            fnirs_decoder_embed_dim=fnirs_decoder_embed_dim,
            fnirs_decoder_depth=fnirs_decoder_depth,
            fnirs_decoder_num_heads=fnirs_decoder_num_heads,
            shared_codebook_size=shared_codebook_size,
            shared_codebook_dim=shared_codebook_dim,
            eeg_private_codebook_size=eeg_private_codebook_size,
            eeg_private_codebook_dim=eeg_private_codebook_dim,
            fnirs_private_codebook_size=fnirs_private_codebook_size,
            fnirs_private_codebook_dim=fnirs_private_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
            eeg_amplitude_weight=eeg_amplitude_weight,
            eeg_phase_weight=eeg_phase_weight,
            eeg_time_weight=eeg_time_weight,
            fnirs_amplitude_weight=fnirs_amplitude_weight,
            fnirs_phase_weight=fnirs_phase_weight,
            fnirs_time_weight=fnirs_time_weight,
            latent_alignment_weight=0.0,
            coupling_weight=coupling_weight,
            assignment_alignment_weight=0.0,
            hard_assignment_alignment_weight=0.0,
            shared_entropy_weight=0.0,
            private_entropy_weight=0.0,
            shared_eeg_recon_weight=0.0,
            shared_fnirs_recon_weight=0.0,
            shared_eeg_common_weight=shared_eeg_common_weight,
            shared_fnirs_common_weight=shared_fnirs_common_weight,
            eeg_private_residual_weight=eeg_private_residual_weight,
            fnirs_private_residual_weight=fnirs_private_residual_weight,
            eeg_common_pool_kernel=eeg_common_pool_kernel,
            fnirs_common_pool_kernel=fnirs_common_pool_kernel,
            coupling_bidirectional=coupling_bidirectional,
            orthogonality_weight=orthogonality_weight,
            assignment_temperature=assignment_temperature,
            alignment_lag_candidates=alignment_lag_candidates,
            alignment_selection=alignment_selection,
            alignment_compare_mode=alignment_compare_mode,
            shared_branch_dropout=shared_branch_dropout,
            eeg_private_branch_dropout=eeg_private_branch_dropout,
            fnirs_private_branch_dropout=fnirs_private_branch_dropout,
            dropout=dropout,
            drop_path=drop_path,
            use_smooth_l1=use_smooth_l1,
            **kwargs,
        )
        self.codebook_balance_weight = max(float(codebook_balance_weight), 0.0)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CodebookFocusedFactorizedLaBraMVQNSP':
        model_cfg = config.get('model', {})
        eeg_cfg = model_cfg.get('eeg', {})
        fnirs_cfg = model_cfg.get('fnirs', {})
        shared_cfg = model_cfg.get('shared', {})
        eeg_private_cfg = model_cfg.get('eeg_private', {})
        fnirs_private_cfg = model_cfg.get('fnirs_private', {})
        branch_dropout_cfg = model_cfg.get('branch_dropout', {})
        quantizer_cfg = model_cfg.get('quantizer', {})
        loss_cfg = config.get('loss', {})
        align_cfg = loss_cfg.get('alignment', {})
        eeg_loss_cfg = loss_cfg.get('eeg', {})
        fnirs_loss_cfg = loss_cfg.get('fnirs', {})
        validation_cfg = config.get('validation', {})

        shared_common_weight = align_cfg.get('shared_common_weight')
        if shared_common_weight is None:
            shared_eeg_common_weight = align_cfg.get('shared_eeg_common_weight', 0.18)
            shared_fnirs_common_weight = align_cfg.get('shared_fnirs_common_weight', 0.15)
        else:
            shared_eeg_common_weight = float(shared_common_weight)
            shared_fnirs_common_weight = float(shared_common_weight)

        private_residual_weight = align_cfg.get('private_residual_weight')
        if private_residual_weight is None:
            eeg_private_residual_weight = align_cfg.get('eeg_private_residual_weight', 0.08)
            fnirs_private_residual_weight = align_cfg.get('fnirs_private_residual_weight', 0.08)
        else:
            eeg_private_residual_weight = float(private_residual_weight)
            fnirs_private_residual_weight = float(private_residual_weight)

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
            shared_codebook_size=shared_cfg.get('codebook_size', 128),
            shared_codebook_dim=shared_cfg.get('codebook_dim', 48),
            eeg_private_codebook_size=eeg_private_cfg.get('codebook_size', 256),
            eeg_private_codebook_dim=eeg_private_cfg.get('codebook_dim', 64),
            fnirs_private_codebook_size=fnirs_private_cfg.get('codebook_size', 128),
            fnirs_private_codebook_dim=fnirs_private_cfg.get('codebook_dim', 48),
            beta=quantizer_cfg.get('beta', 1.0),
            decay=quantizer_cfg.get('decay', 0.99),
            kmeans_init=quantizer_cfg.get('kmeans_init', True),
            revive_dead_codes=quantizer_cfg.get('revive_dead_codes', True),
            dead_code_threshold=quantizer_cfg.get('dead_code_threshold', 10),
            eeg_amplitude_weight=eeg_loss_cfg.get('amplitude_weight', 1.0),
            eeg_phase_weight=eeg_loss_cfg.get('phase_weight', 1.0),
            eeg_time_weight=eeg_loss_cfg.get('time_weight', 0.75),
            fnirs_amplitude_weight=fnirs_loss_cfg.get('amplitude_weight', 1.0),
            fnirs_phase_weight=fnirs_loss_cfg.get('phase_weight', 0.25),
            fnirs_time_weight=fnirs_loss_cfg.get('time_weight', 1.25),
            coupling_weight=align_cfg.get('coupling_weight', 0.07),
            codebook_balance_weight=align_cfg.get('codebook_balance_weight', 0.02),
            shared_eeg_common_weight=shared_eeg_common_weight,
            shared_fnirs_common_weight=shared_fnirs_common_weight,
            eeg_private_residual_weight=eeg_private_residual_weight,
            fnirs_private_residual_weight=fnirs_private_residual_weight,
            eeg_common_pool_kernel=align_cfg.get('eeg_common_pool_kernel', eeg_cfg.get('patch_size', 400)),
            fnirs_common_pool_kernel=align_cfg.get('fnirs_common_pool_kernel', fnirs_cfg.get('patch_size', 20)),
            coupling_bidirectional=align_cfg.get('coupling_bidirectional', True),
            orthogonality_weight=align_cfg.get('orthogonality_weight', 0.01),
            assignment_temperature=align_cfg.get('temperature', 0.35),
            alignment_lag_candidates=align_cfg.get('lag_candidates', validation_cfg.get('lag_set', [0])),
            alignment_selection=align_cfg.get('selection', 'min'),
            alignment_compare_mode=align_cfg.get('compare_mode', 'fixed_min'),
            shared_branch_dropout=branch_dropout_cfg.get('shared', 0.0),
            eeg_private_branch_dropout=branch_dropout_cfg.get('eeg_private', 0.0),
            fnirs_private_branch_dropout=branch_dropout_cfg.get('fnirs_private', 0.0),
            dropout=model_cfg.get('dropout', 0.0),
            drop_path=model_cfg.get('drop_path', 0.1),
            use_smooth_l1=loss_cfg.get('use_smooth_l1', True),
        )

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

        temperature = max(self.assignment_temperature, 1e-3)
        eeg_private_probs = F.softmax(eeg_private_logits / temperature, dim=-1)
        fnirs_private_probs = F.softmax(fnirs_private_logits / temperature, dim=-1)
        private_entropy_loss = 0.5 * (
            self._batch_usage_entropy_loss(eeg_private_probs) +
            self._batch_usage_entropy_loss(fnirs_private_probs)
        )
        codebook_balance_loss = 0.5 * (shared_entropy_loss + private_entropy_loss)

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
        private_only_reconstructions = self.decode_from_components(
            zero_eeg_shared_q,
            eeg_private_q,
            zero_fnirs_shared_q,
            fnirs_private_q,
        )
        eeg_private_only_rec = private_only_reconstructions['eeg_reconstructed']
        fnirs_private_only_rec = private_only_reconstructions['fnirs_reconstructed']

        eeg_common_target = self._smooth_signal(eeg, self.eeg_common_pool_kernel)
        fnirs_common_target = self._smooth_signal(fnirs, self.fnirs_common_pool_kernel)
        eeg_residual_target = eeg - eeg_common_target
        fnirs_residual_target = fnirs - fnirs_common_target

        shared_eeg_common_loss = self.loss_fn(shared_eeg_rec, eeg_common_target)
        shared_fnirs_common_loss = self.loss_fn(shared_fnirs_rec, fnirs_common_target)
        eeg_private_residual_loss = self.loss_fn(eeg_private_only_rec, eeg_residual_target)
        fnirs_private_residual_loss = self.loss_fn(fnirs_private_only_rec, fnirs_residual_target)

        orthogonality_loss = (
            self._orthogonality_loss(eeg_shared, eeg_private) +
            self._orthogonality_loss(fnirs_shared, fnirs_private)
        )
        vq_shared_loss = shared_info['vq_loss']
        vq_eeg_private_loss = eeg_private_info['vq_loss']
        vq_fnirs_private_loss = fnirs_private_info['vq_loss']
        vq_loss = vq_shared_loss + vq_eeg_private_loss + vq_fnirs_private_loss

        total_loss = (
            eeg_rec_loss +
            fnirs_rec_loss +
            vq_loss +
            (self.coupling_weight * self.alignment_scale) * coupling_loss +
            self.codebook_balance_weight * codebook_balance_loss +
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
            'codebook_balance_loss': codebook_balance_loss,
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
            'eeg_private_only_reconstructed': eeg_private_only_rec,
            'fnirs_private_only_reconstructed': fnirs_private_only_rec,
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

    def get_gradient_component_weights(self) -> Dict[str, float]:
        alignment_scale = float(self.get_alignment_scale())
        return {
            'eeg_rec_loss': 1.0,
            'fnirs_rec_loss': 1.0,
            'vq_loss': 1.0,
            'coupling_loss': self.coupling_weight * alignment_scale,
            'codebook_balance_loss': self.codebook_balance_weight,
            'shared_eeg_common_loss': self.shared_eeg_common_weight,
            'shared_fnirs_common_loss': self.shared_fnirs_common_weight,
            'eeg_private_residual_loss': self.eeg_private_residual_weight,
            'fnirs_private_residual_loss': self.fnirs_private_residual_weight,
            'orthogonality_loss': self.orthogonality_weight,
        }


class OverfitFactorizedLaBraMVQNSP(CodebookFocusedFactorizedLaBraMVQNSP):
    """High-capacity factorized tokenizer with aggressive reconstruction shortcuts."""

    def __init__(
        self,
        fit_hidden_multiplier: float = 1.5,
        fit_shortcut_weight: float = 0.45,
        fit_residual_weight: float = 0.35,
        fit_input_refine_weight: float = 0.2,
        fit_mix_init: float = 0.6,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.fit_shortcut_weight = max(float(fit_shortcut_weight), 0.0)
        self.fit_residual_weight = max(float(fit_residual_weight), 0.0)
        self.fit_input_refine_weight = max(float(fit_input_refine_weight), 0.0)

        eeg_latent_dim = self.shared_codebook_dim + self.eeg_private_codebook_dim
        fnirs_latent_dim = self.shared_codebook_dim + self.fnirs_private_codebook_dim

        eeg_decoder_dim = int(self.eeg_decode_input_proj.out_features)
        fnirs_decoder_dim = int(self.fnirs_decode_input_proj.out_features)
        eeg_fit_hidden = max(int(eeg_decoder_dim * fit_hidden_multiplier), eeg_decoder_dim)
        fnirs_fit_hidden = max(int(fnirs_decoder_dim * fit_hidden_multiplier), fnirs_decoder_dim)

        self.eeg_shortcut_head = nn.Sequential(
            nn.Linear(eeg_latent_dim, eeg_fit_hidden),
            nn.GELU(),
            nn.Linear(eeg_fit_hidden, self.eeg_channels * self.eeg_patch_size),
        )
        self.fnirs_shortcut_head = nn.Sequential(
            nn.Linear(fnirs_latent_dim, fnirs_fit_hidden),
            nn.GELU(),
            nn.Linear(fnirs_fit_hidden, self.fnirs_channels * self.fnirs_patch_size),
        )

        self.eeg_residual_head = nn.Sequential(
            nn.Linear(eeg_decoder_dim, eeg_fit_hidden),
            nn.GELU(),
            nn.Linear(eeg_fit_hidden, self.eeg_channels * self.eeg_patch_size),
        )
        self.fnirs_residual_head = nn.Sequential(
            nn.Linear(fnirs_decoder_dim, fnirs_fit_hidden),
            nn.GELU(),
            nn.Linear(fnirs_fit_hidden, self.fnirs_channels * self.fnirs_patch_size),
        )

        self.eeg_input_refiner = nn.Sequential(
            nn.Conv1d(self.eeg_channels, self.eeg_channels * 2, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(self.eeg_channels * 2, self.eeg_channels, kernel_size=7, padding=3),
        )
        self.fnirs_input_refiner = nn.Sequential(
            nn.Conv1d(self.fnirs_channels, self.fnirs_channels * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(self.fnirs_channels * 2, self.fnirs_channels, kernel_size=5, padding=2),
        )

        init_logit = torch.logit(torch.tensor(min(max(float(fit_mix_init), 1e-4), 1.0 - 1e-4)))
        self.eeg_shortcut_logit = nn.Parameter(init_logit.clone())
        self.eeg_residual_logit = nn.Parameter(init_logit.clone())
        self.eeg_input_refine_logit = nn.Parameter(init_logit.clone())
        self.fnirs_shortcut_logit = nn.Parameter(init_logit.clone())
        self.fnirs_residual_logit = nn.Parameter(init_logit.clone())
        self.fnirs_input_refine_logit = nn.Parameter(init_logit.clone())

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OverfitFactorizedLaBraMVQNSP':
        model_cfg = config.get('model', {})
        fit_cfg = model_cfg.get('fit', {})
        loss_cfg = config.get('loss', {})
        fit_loss_cfg = loss_cfg.get('fit', {})

        eeg_cfg = model_cfg.get('eeg', {})
        fnirs_cfg = model_cfg.get('fnirs', {})
        shared_cfg = model_cfg.get('shared', {})
        eeg_private_cfg = model_cfg.get('eeg_private', {})
        fnirs_private_cfg = model_cfg.get('fnirs_private', {})
        branch_dropout_cfg = model_cfg.get('branch_dropout', {})
        quantizer_cfg = model_cfg.get('quantizer', {})
        align_cfg = loss_cfg.get('alignment', {})
        eeg_loss_cfg = loss_cfg.get('eeg', {})
        fnirs_loss_cfg = loss_cfg.get('fnirs', {})
        validation_cfg = config.get('validation', {})

        shared_common_weight = align_cfg.get('shared_common_weight')
        if shared_common_weight is None:
            shared_eeg_common_weight = align_cfg.get('shared_eeg_common_weight', 0.18)
            shared_fnirs_common_weight = align_cfg.get('shared_fnirs_common_weight', 0.15)
        else:
            shared_eeg_common_weight = float(shared_common_weight)
            shared_fnirs_common_weight = float(shared_common_weight)

        private_residual_weight = align_cfg.get('private_residual_weight')
        if private_residual_weight is None:
            eeg_private_residual_weight = align_cfg.get('eeg_private_residual_weight', 0.08)
            fnirs_private_residual_weight = align_cfg.get('fnirs_private_residual_weight', 0.08)
        else:
            eeg_private_residual_weight = float(private_residual_weight)
            fnirs_private_residual_weight = float(private_residual_weight)

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
            shared_codebook_size=shared_cfg.get('codebook_size', 128),
            shared_codebook_dim=shared_cfg.get('codebook_dim', 48),
            eeg_private_codebook_size=eeg_private_cfg.get('codebook_size', 256),
            eeg_private_codebook_dim=eeg_private_cfg.get('codebook_dim', 64),
            fnirs_private_codebook_size=fnirs_private_cfg.get('codebook_size', 128),
            fnirs_private_codebook_dim=fnirs_private_cfg.get('codebook_dim', 48),
            beta=quantizer_cfg.get('beta', 1.0),
            decay=quantizer_cfg.get('decay', 0.99),
            kmeans_init=quantizer_cfg.get('kmeans_init', True),
            revive_dead_codes=quantizer_cfg.get('revive_dead_codes', True),
            dead_code_threshold=quantizer_cfg.get('dead_code_threshold', 10),
            eeg_amplitude_weight=eeg_loss_cfg.get('amplitude_weight', 1.0),
            eeg_phase_weight=eeg_loss_cfg.get('phase_weight', 1.0),
            eeg_time_weight=eeg_loss_cfg.get('time_weight', 0.75),
            fnirs_amplitude_weight=fnirs_loss_cfg.get('amplitude_weight', 1.0),
            fnirs_phase_weight=fnirs_loss_cfg.get('phase_weight', 0.25),
            fnirs_time_weight=fnirs_loss_cfg.get('time_weight', 1.25),
            coupling_weight=align_cfg.get('coupling_weight', 0.07),
            codebook_balance_weight=align_cfg.get('codebook_balance_weight', 0.02),
            shared_eeg_common_weight=shared_eeg_common_weight,
            shared_fnirs_common_weight=shared_fnirs_common_weight,
            eeg_private_residual_weight=eeg_private_residual_weight,
            fnirs_private_residual_weight=fnirs_private_residual_weight,
            eeg_common_pool_kernel=align_cfg.get('eeg_common_pool_kernel', eeg_cfg.get('patch_size', 400)),
            fnirs_common_pool_kernel=align_cfg.get('fnirs_common_pool_kernel', fnirs_cfg.get('patch_size', 20)),
            coupling_bidirectional=align_cfg.get('coupling_bidirectional', True),
            orthogonality_weight=align_cfg.get('orthogonality_weight', 0.01),
            assignment_temperature=align_cfg.get('temperature', 0.35),
            alignment_lag_candidates=align_cfg.get('lag_candidates', validation_cfg.get('lag_set', [0])),
            alignment_selection=align_cfg.get('selection', 'min'),
            alignment_compare_mode=align_cfg.get('compare_mode', 'fixed_min'),
            shared_branch_dropout=branch_dropout_cfg.get('shared', 0.0),
            eeg_private_branch_dropout=branch_dropout_cfg.get('eeg_private', 0.0),
            fnirs_private_branch_dropout=branch_dropout_cfg.get('fnirs_private', 0.0),
            dropout=model_cfg.get('dropout', 0.0),
            drop_path=model_cfg.get('drop_path', 0.1),
            use_smooth_l1=loss_cfg.get('use_smooth_l1', True),
            fit_hidden_multiplier=fit_cfg.get('hidden_multiplier', 1.5),
            fit_mix_init=fit_cfg.get('mix_init', 0.6),
            fit_shortcut_weight=fit_loss_cfg.get('shortcut_weight', 0.45),
            fit_residual_weight=fit_loss_cfg.get('residual_weight', 0.35),
            fit_input_refine_weight=fit_loss_cfg.get('input_refine_weight', 0.2),
        )

    def _patches_to_signal(self, patches: torch.Tensor) -> torch.Tensor:
        return patches.permute(0, 2, 1, 3).contiguous().view(patches.shape[0], patches.shape[2], -1)

    def _decode_time_shortcut(self, latent: torch.Tensor, head: nn.Module, channels: int, patch_size: int) -> torch.Tensor:
        patches = head(latent).view(latent.shape[0], self.n_patches, channels, patch_size)
        return self._patches_to_signal(patches)

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = super().forward(eeg, fnirs)

        eeg_shared = outputs['eeg_z']
        fnirs_shared = outputs['fnirs_z']
        eeg_private = outputs['eeg_private_z']
        fnirs_private = outputs['fnirs_private_z']
        eeg_shared_q = outputs['eeg_z_q']
        fnirs_shared_q = outputs['fnirs_z_q']
        eeg_private_q = outputs['eeg_private_z_q']
        fnirs_private_q = outputs['fnirs_private_z_q']

        eeg_quant_latent = torch.cat([eeg_shared_q, eeg_private_q], dim=-1)
        fnirs_quant_latent = torch.cat([fnirs_shared_q, fnirs_private_q], dim=-1)
        eeg_unquant_latent = torch.cat([eeg_shared, eeg_private], dim=-1)
        fnirs_unquant_latent = torch.cat([fnirs_shared, fnirs_private], dim=-1)

        eeg_decoded = self.eeg_decoder(self.eeg_decode_input_proj(eeg_quant_latent))
        fnirs_decoded = self.fnirs_decoder(self.fnirs_decode_input_proj(fnirs_quant_latent))

        eeg_residual_rec = self._decode_time_shortcut(
            eeg_decoded,
            self.eeg_residual_head,
            self.eeg_channels,
            self.eeg_patch_size,
        )
        fnirs_residual_rec = self._decode_time_shortcut(
            fnirs_decoded,
            self.fnirs_residual_head,
            self.fnirs_channels,
            self.fnirs_patch_size,
        )
        eeg_shortcut_rec = self._decode_time_shortcut(
            eeg_unquant_latent,
            self.eeg_shortcut_head,
            self.eeg_channels,
            self.eeg_patch_size,
        )
        fnirs_shortcut_rec = self._decode_time_shortcut(
            fnirs_unquant_latent,
            self.fnirs_shortcut_head,
            self.fnirs_channels,
            self.fnirs_patch_size,
        )
        eeg_input_delta = self.eeg_input_refiner(eeg)
        fnirs_input_delta = self.fnirs_input_refiner(fnirs)

        eeg_rec = outputs['eeg_reconstructed']
        fnirs_rec = outputs['fnirs_reconstructed']

        eeg_fused_rec = (
            eeg_rec +
            torch.sigmoid(self.eeg_residual_logit) * eeg_residual_rec +
            torch.sigmoid(self.eeg_shortcut_logit) * eeg_shortcut_rec +
            torch.sigmoid(self.eeg_input_refine_logit) * eeg_input_delta
        )
        fnirs_fused_rec = (
            fnirs_rec +
            torch.sigmoid(self.fnirs_residual_logit) * fnirs_residual_rec +
            torch.sigmoid(self.fnirs_shortcut_logit) * fnirs_shortcut_rec +
            torch.sigmoid(self.fnirs_input_refine_logit) * fnirs_input_delta
        )

        eeg_time_loss = self.loss_fn(eeg_fused_rec, eeg)
        fnirs_time_loss = self.loss_fn(fnirs_fused_rec, fnirs)
        eeg_shortcut_loss = self.loss_fn(eeg_shortcut_rec, eeg)
        fnirs_shortcut_loss = self.loss_fn(fnirs_shortcut_rec, fnirs)
        eeg_residual_fit_loss = self.loss_fn(eeg_residual_rec + eeg_rec, eeg)
        fnirs_residual_fit_loss = self.loss_fn(fnirs_residual_rec + fnirs_rec, fnirs)
        eeg_input_refine_loss = self.loss_fn(eeg_input_delta + eeg_rec, eeg)
        fnirs_input_refine_loss = self.loss_fn(fnirs_input_delta + fnirs_rec, fnirs)

        base_eeg_time_loss = outputs['eeg_time_loss']
        base_fnirs_time_loss = outputs['fnirs_time_loss']
        base_eeg_rec_loss = outputs['eeg_rec_loss']
        base_fnirs_rec_loss = outputs['fnirs_rec_loss']

        outputs['eeg_time_loss'] = eeg_time_loss
        outputs['fnirs_time_loss'] = fnirs_time_loss
        outputs['eeg_rec_loss'] = base_eeg_rec_loss - self.eeg_time_weight * base_eeg_time_loss + self.eeg_time_weight * eeg_time_loss
        outputs['fnirs_rec_loss'] = base_fnirs_rec_loss - self.fnirs_time_weight * base_fnirs_time_loss + self.fnirs_time_weight * fnirs_time_loss
        outputs['eeg_reconstructed'] = eeg_fused_rec
        outputs['fnirs_reconstructed'] = fnirs_fused_rec

        fit_loss = (
            self.fit_shortcut_weight * (eeg_shortcut_loss + fnirs_shortcut_loss) +
            self.fit_residual_weight * (eeg_residual_fit_loss + fnirs_residual_fit_loss) +
            self.fit_input_refine_weight * (eeg_input_refine_loss + fnirs_input_refine_loss)
        )
        outputs['fit_loss'] = fit_loss
        outputs['eeg_shortcut_loss'] = eeg_shortcut_loss
        outputs['fnirs_shortcut_loss'] = fnirs_shortcut_loss
        outputs['eeg_residual_fit_loss'] = eeg_residual_fit_loss
        outputs['fnirs_residual_fit_loss'] = fnirs_residual_fit_loss
        outputs['eeg_input_refine_loss'] = eeg_input_refine_loss
        outputs['fnirs_input_refine_loss'] = fnirs_input_refine_loss
        outputs['base_eeg_reconstructed'] = eeg_rec
        outputs['base_fnirs_reconstructed'] = fnirs_rec

        outputs['loss'] = outputs['loss'] + fit_loss
        return outputs

    def get_gradient_component_weights(self) -> Dict[str, float]:
        weights = super().get_gradient_component_weights()
        weights.update({
            'fit_loss': 1.0,
        })
        return weights