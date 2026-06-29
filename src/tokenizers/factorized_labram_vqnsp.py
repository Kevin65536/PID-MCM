"""Source/observation LaBraM tokenizer with dual source codebooks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.channel_adjacency import (
    SpatialAdjacencyInfo,
    build_channel_adjacency,
    compute_per_channel_rms_envelope,
    compute_spatial_fnirs_driver,
)
from src.losses.multimodal_tokenizer import (
    batch_usage_entropy_loss,
    coupling_eeg_neighbor_smoothness_loss,
    coupling_effective_neighbor_smoothness_loss,
    coupling_interaction_lag_sparsity_loss,
    coupling_joint_entropy_loss,
    coupling_lag_evidence_loss,
    coupling_lag_focus_loss,
    coupling_pair_likelihood_loss,
    context_residual_coupling_loss,
    local_residual_coupling_loss,
    orthogonality_loss,
    straight_through_assignment_probs,
)

from .base import BaseTokenizer
from .cross_modal_fusion import (
    LagAwareCrossModalFusion,
    attention_lag_statistics,
    lag_aware_temporal_nce,
    masked_alignment_losses,
)
from .labram_vqnsp import (
    MultiChannelPatchEmbedding,
    NormEMAVectorQuantizer,
    TransformerDecoder,
    TransformerEncoder,
    kmeans,
    l2norm,
)


class CausalLowRankCrossAdapter(nn.Module):
    """Low-rank causal EEG-to-fNIRS adapter for pre-VQ source exchange."""

    def __init__(
        self,
        *,
        eeg_dim: int,
        fnirs_dim: int,
        rank: int = 16,
        adapter_dim: int = 64,
        max_lag_tokens: int = 5,
        residual_init: float = 0.1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.max_lag_tokens = max(int(max_lag_tokens), 0)
        self.eeg_context_proj = nn.Linear(eeg_dim, rank)
        self.fnirs_query_proj = nn.Linear(fnirs_dim, rank)
        self.context_to_adapter = nn.Linear(rank, adapter_dim)
        self.adapter_to_fnirs = nn.Linear(adapter_dim, fnirs_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))

    def _lagged_eeg_context(self, eeg_low_rank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, rank = eeg_low_rank.shape
        lags = min(self.max_lag_tokens, max(tokens - 1, 0)) + 1
        contexts = []
        valid = []
        for lag in range(lags):
            if lag == 0:
                shifted = eeg_low_rank
                mask = torch.ones(tokens, dtype=torch.bool, device=eeg_low_rank.device)
            else:
                padding = eeg_low_rank.new_zeros(batch, lag, rank)
                shifted = torch.cat([padding, eeg_low_rank[:, :-lag, :]], dim=1)
                mask = torch.arange(tokens, device=eeg_low_rank.device) >= lag
            contexts.append(shifted)
            valid.append(mask)
        return torch.stack(contexts, dim=2), torch.stack(valid, dim=0)

    def forward(
        self,
        eeg_encoded: torch.Tensor,
        fnirs_encoded: torch.Tensor,
        *,
        detach_context: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if detach_context:
            eeg_encoded = eeg_encoded.detach()
        eeg_low_rank = self.eeg_context_proj(eeg_encoded)
        fnirs_query = self.fnirs_query_proj(fnirs_encoded)
        lagged_eeg, valid_mask = self._lagged_eeg_context(eeg_low_rank)
        scores = (lagged_eeg * fnirs_query.unsqueeze(2)).sum(dim=-1)
        scores = scores / math.sqrt(max(float(fnirs_query.shape[-1]), 1.0))
        scores = scores.masked_fill(~valid_mask.t().unsqueeze(0), -1.0e4)
        lag_weights = torch.softmax(scores, dim=-1)
        context = (lag_weights.unsqueeze(-1) * lagged_eeg).sum(dim=2)
        adapter_hidden = F.gelu(self.context_to_adapter(context))
        adapter_update = self.adapter_to_fnirs(self.dropout(adapter_hidden))
        return self.residual_gate * adapter_update, lag_weights


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
        fnirs_spatial_anchors: Optional[int] = None,
        fnirs_optical_components: Optional[int] = None,
        fnirs_component_labels: Optional[Sequence[str]] = None,
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
        eeg_time_weight: float = 0.9,
        fnirs_amplitude_weight: float = 1.0,
        fnirs_time_weight: float = 1.0,
        coupling_weight: float = 0.0,
        coupling_lag_focus_weight: float = 1.0,
        coupling_joint_entropy_weight: float = 0.0,
        coupling_smoothness_weight: float = 0.2,
        coupling_smoothness_neighbors: int = 5,
        coupling_pair_likelihood_weight: float = 0.0,
        coupling_pair_detach_tokens: bool = True,
        coupling_pair_gradient_target: Optional[str] = None,
        coupling_pair_temperature: Optional[float] = None,
        coupling_residualize_fnirs_marginal: bool = False,
        coupling_lag_evidence_weight: float = 0.0,
        coupling_lag_evidence_temperature: float = 0.25,
        coupling_max_lag_tokens: Optional[int] = None,
        coupling_fixed_eeg_marginal: Optional[Sequence[Sequence[float]]] = None,
        coupling_fixed_fnirs_marginal: Optional[Sequence[Sequence[float]]] = None,
        coupling_effective_smoothness_weight: float = 0.0,
        coupling_interaction_lag_sparsity_weight: float = 0.0,
        coupling_local_residual_enabled: bool = False,
        coupling_local_residual_pair_weight: float = 1.0,
        coupling_local_residual_alpha: float = 0.5,
        coupling_context_residual_enabled: bool = False,
        coupling_context_states: int = 4,
        coupling_context_rank: int = 16,
        coupling_context_router_type: str = 'learned',
        coupling_context_pair_weight: float = 1.0,
        coupling_context_entropy_weight: float = 0.01,
        coupling_context_balance_weight: float = 0.01,
        coupling_context_residual_l1_weight: float = 0.001,
        coupling_context_gradient_target: Optional[str] = None,
        interaction_aux_weight: float = 0.0,
        interaction_aux_direction: str = 'eeg_to_fnirs',
        interaction_aux_stop_gradient: bool = True,
        shared_state_bottleneck_weight: float = 0.0,
        shared_state_bottleneck_dim: int = 32,
        shared_state_bottleneck_stop_gradient: bool = True,
        cross_modal_exchange_enabled: bool = False,
        cross_modal_exchange_mode: str = 'low_rank_causal_adapter',
        cross_modal_exchange_direction: str = 'eeg_to_fnirs',
        cross_modal_exchange_target_branch: str = 'fnirs_source',
        cross_modal_exchange_rank: int = 16,
        cross_modal_exchange_adapter_dim: int = 64,
        cross_modal_exchange_max_lag_tokens: int = 5,
        cross_modal_exchange_detach_context: bool = True,
        cross_modal_exchange_residual_init: float = 0.1,
        cross_modal_exchange_dropout: float = 0.05,
        cross_modal_fusion_enabled: bool = False,
        cross_modal_fusion_mode: str = 'causal_cross_attention',
        cross_modal_fusion_embed_dim: int = 128,
        cross_modal_fusion_depth: int = 2,
        cross_modal_fusion_num_heads: int = 4,
        cross_modal_fusion_max_lag_tokens: int = 5,
        cross_modal_fusion_relative_lag_bias: bool = True,
        cross_modal_fusion_dropout: float = 0.1,
        source_codebook_mode: str = 'independent',
        cross_modal_temporal_nce_weight: float = 0.0,
        cross_modal_masked_latent_weight: float = 0.0,
        cross_modal_soft_code_weight: float = 0.0,
        cross_modal_alignment_temperature: float = 0.1,
        cross_modal_positive_lag_weights: Optional[Sequence[float]] = None,
        cross_modal_token_mask_ratio: float = 0.5,
        cross_modal_modality_dropout_probability: float = 0.25,
        source_target_weight: float = 0.0,
        eeg_source_aux_weight: float = 0.5,
        source_target_correlation_weight: float = 0.0,
        eeg_source_aux_correlation_weight: float = 0.0,
        observation_target_weight: float = 0.0,
        codebook_balance_weight: float = 0.02,
        source_balance_scale: float = 1.0,
        observation_balance_scale: float = 1.0,
        orthogonality_weight: float = 0.01,
        assignment_temperature: float = 1.0,
        source_balance_temperature: float | None = None,
        observation_balance_temperature: float | None = None,
        source_simvq_enabled: bool = False,
        source_simvq_loss_weight: float = 1.0,
        observation_simvq_enabled: bool = False,
        observation_simvq_loss_weight: float = 1.0,
        source_branch_dropout: float = 0.0,
        eeg_observation_branch_dropout: float = 0.0,
        fnirs_observation_branch_dropout: float = 0.0,
        window_duration_s: float = 10.0,
        hrf_kernel_duration_s: float = 24.0,
        hrf_peak_delay_s: float = 6.0,
        hrf_peak_dispersion_s: float = 1.0,
        hrf_undershoot_delay_s: float = 16.0,
        hrf_undershoot_dispersion_s: float = 1.5,
        hrf_peak_scale: float = 1.0,
        hrf_undershoot_scale: float = 0.25,
        spatial_source_prior_enabled: bool = False,
        eeg_source_target_mode: str = 'signed_rms_carrier',
        eeg_source_target_smoothing_ms: float = 200.0,
        eeg_rms_smoothing_ms: float = 200.0,
        shared_state_alpha: float = 0.90,
        dataset_id: Optional[str] = None,
        data_root: Optional[str] = None,
        eeg_channel_names: Optional[Sequence[str]] = None,
        fnirs_channel_names: Optional[Sequence[str]] = None,
        spatial_prior_reference_subject_id: int = 1,
        spatial_prior_use_artifact_data: bool = True,
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
        self.fnirs_spatial_anchors = int(fnirs_spatial_anchors) if fnirs_spatial_anchors is not None else None
        self.fnirs_optical_components = (
            int(fnirs_optical_components) if fnirs_optical_components is not None else None
        )
        self.fnirs_component_labels = list(fnirs_component_labels) if fnirs_component_labels is not None else []
        if self.fnirs_spatial_anchors is not None and self.fnirs_optical_components is not None:
            expected_fnirs_channels = self.fnirs_spatial_anchors * self.fnirs_optical_components
            if expected_fnirs_channels != self.fnirs_channels:
                raise ValueError(
                    'fnirs.channels must equal fnirs.spatial_anchors * fnirs.optical_components '
                    f'(got channels={self.fnirs_channels}, spatial_anchors={self.fnirs_spatial_anchors}, '
                    f'optical_components={self.fnirs_optical_components})'
                )
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
        self.register_buffer('eeg_fft_loss_window', torch.hann_window(eeg_patch_size), persistent=False)
        self.register_buffer('fnirs_fft_loss_window', torch.hann_window(fnirs_patch_size), persistent=False)
        self.eeg_amplitude_weight = eeg_amplitude_weight
        self.eeg_time_weight = eeg_time_weight
        self.fnirs_amplitude_weight = fnirs_amplitude_weight
        self.fnirs_time_weight = fnirs_time_weight
        self.coupling_weight = coupling_weight
        self.coupling_lag_focus_weight = max(float(coupling_lag_focus_weight), 0.0)
        self.coupling_joint_entropy_weight = max(float(coupling_joint_entropy_weight), 0.0)
        self.coupling_smoothness_weight = max(float(coupling_smoothness_weight), 0.0)
        self.coupling_smoothness_neighbors = max(int(coupling_smoothness_neighbors), 0)
        self.coupling_pair_likelihood_weight = max(float(coupling_pair_likelihood_weight), 0.0)
        self.coupling_pair_detach_tokens = bool(coupling_pair_detach_tokens)
        if coupling_pair_gradient_target is None:
            coupling_pair_gradient_target = "none" if self.coupling_pair_detach_tokens else "both"
        self.coupling_pair_gradient_target = str(coupling_pair_gradient_target).lower()
        if self.coupling_pair_gradient_target not in {"none", "eeg", "fnirs", "both"}:
            raise ValueError(
                "coupling_pair_gradient_target must be one of none/eeg/fnirs/both, "
                f"got {self.coupling_pair_gradient_target!r}"
            )
        self.coupling_pair_temperature = (
            None if coupling_pair_temperature is None else max(float(coupling_pair_temperature), 1e-3)
        )
        self.coupling_residualize_fnirs_marginal = bool(coupling_residualize_fnirs_marginal)
        self.coupling_lag_evidence_weight = max(float(coupling_lag_evidence_weight), 0.0)
        self.coupling_lag_evidence_temperature = max(float(coupling_lag_evidence_temperature), 1e-3)
        self.coupling_effective_smoothness_weight = max(float(coupling_effective_smoothness_weight), 0.0)
        self.coupling_interaction_lag_sparsity_weight = max(float(coupling_interaction_lag_sparsity_weight), 0.0)
        self.coupling_local_residual_enabled = bool(coupling_local_residual_enabled)
        self.coupling_local_residual_pair_weight = max(float(coupling_local_residual_pair_weight), 0.0)
        self.coupling_local_residual_alpha = max(float(coupling_local_residual_alpha), 1e-6)
        self.coupling_context_residual_enabled = bool(coupling_context_residual_enabled)
        self.coupling_context_states = max(int(coupling_context_states), 1)
        self.coupling_context_rank = max(int(coupling_context_rank), 1)
        self.coupling_context_router_type = str(coupling_context_router_type).lower()
        if self.coupling_context_router_type not in {'uniform', 'learned', 'oracle'}:
            raise ValueError(
                "coupling_context_router_type must be uniform/learned/oracle, "
                f"got {self.coupling_context_router_type!r}"
            )
        if self.coupling_context_router_type == 'uniform':
            self.coupling_context_states = 1
        if self.coupling_context_router_type == 'oracle':
            raise ValueError(
                "coupling_context_router_type='oracle' is reserved for offline diagnostics "
                "and is not available inside the tokenizer forward pass"
            )
        self.coupling_context_pair_weight = max(float(coupling_context_pair_weight), 0.0)
        self.coupling_context_entropy_weight = max(float(coupling_context_entropy_weight), 0.0)
        self.coupling_context_balance_weight = max(float(coupling_context_balance_weight), 0.0)
        self.coupling_context_residual_l1_weight = max(float(coupling_context_residual_l1_weight), 0.0)
        if coupling_context_gradient_target is None:
            coupling_context_gradient_target = self.coupling_pair_gradient_target
        self.coupling_context_gradient_target = str(coupling_context_gradient_target).lower()
        if self.coupling_context_gradient_target not in {"none", "eeg", "fnirs", "both"}:
            raise ValueError(
                "coupling_context_gradient_target must be one of none/eeg/fnirs/both, "
                f"got {self.coupling_context_gradient_target!r}"
            )
        self.interaction_aux_weight = max(float(interaction_aux_weight), 0.0)
        self.interaction_aux_direction = str(interaction_aux_direction).lower()
        if self.interaction_aux_direction not in {'eeg_to_fnirs', 'fnirs_to_eeg', 'bidirectional'}:
            raise ValueError(
                "interaction_aux_direction must be eeg_to_fnirs/fnirs_to_eeg/bidirectional, "
                f"got {self.interaction_aux_direction!r}"
            )
        self.interaction_aux_stop_gradient = bool(interaction_aux_stop_gradient)
        self.shared_state_bottleneck_weight = max(float(shared_state_bottleneck_weight), 0.0)
        self.shared_state_bottleneck_dim = max(int(shared_state_bottleneck_dim), 1)
        self.shared_state_bottleneck_stop_gradient = bool(shared_state_bottleneck_stop_gradient)
        self.cross_modal_exchange_enabled = bool(cross_modal_exchange_enabled)
        self.cross_modal_exchange_mode = str(cross_modal_exchange_mode).lower()
        self.cross_modal_exchange_direction = str(cross_modal_exchange_direction).lower()
        self.cross_modal_exchange_target_branch = str(cross_modal_exchange_target_branch).lower()
        self.cross_modal_exchange_detach_context = bool(cross_modal_exchange_detach_context)
        self.cross_modal_exchange_max_lag_tokens = max(int(cross_modal_exchange_max_lag_tokens), 0)
        if self.cross_modal_exchange_mode != 'low_rank_causal_adapter':
            raise ValueError(
                "cross_modal_exchange mode must be 'low_rank_causal_adapter', "
                f"got {self.cross_modal_exchange_mode!r}"
            )
        if self.cross_modal_exchange_direction != 'eeg_to_fnirs':
            raise ValueError(
                "cross_modal_exchange direction must be 'eeg_to_fnirs' in v1, "
                f"got {self.cross_modal_exchange_direction!r}"
            )
        if self.cross_modal_exchange_target_branch != 'fnirs_source':
            raise ValueError(
                "cross_modal_exchange target_branch must be 'fnirs_source' in v1, "
                f"got {self.cross_modal_exchange_target_branch!r}"
            )
        self.cross_modal_fusion_enabled = bool(cross_modal_fusion_enabled)
        self.cross_modal_fusion_mode = str(cross_modal_fusion_mode).lower()
        if self.cross_modal_fusion_mode not in {'causal_cross_attention', 'bidirectional_cross_attention'}:
            raise ValueError(
                "cross_modal_fusion mode must be causal_cross_attention/bidirectional_cross_attention, "
                f"got {self.cross_modal_fusion_mode!r}"
            )
        if self.cross_modal_fusion_enabled and self.cross_modal_exchange_enabled:
            raise ValueError('cross_modal_fusion and legacy cross_modal_exchange cannot both be enabled')
        self.cross_modal_fusion_max_lag_tokens = max(int(cross_modal_fusion_max_lag_tokens), 0)
        self.source_codebook_mode = str(source_codebook_mode).lower()
        if self.source_codebook_mode not in {'independent', 'shared_joint'}:
            raise ValueError("source_codebook mode must be independent/shared_joint")
        if self.source_codebook_mode == 'shared_joint' and eeg_source_codebook_dim != fnirs_source_codebook_dim:
            raise ValueError('shared_joint source codebook requires equal EEG/fNIRS source dimensions')
        self.cross_modal_temporal_nce_weight = max(float(cross_modal_temporal_nce_weight), 0.0)
        self.cross_modal_masked_latent_weight = max(float(cross_modal_masked_latent_weight), 0.0)
        self.cross_modal_soft_code_weight = max(float(cross_modal_soft_code_weight), 0.0)
        self.cross_modal_alignment_temperature = max(float(cross_modal_alignment_temperature), 1e-6)
        default_lag_weights = [0.0, 0.1, 0.4, 0.4, 0.1, 0.0]
        self.cross_modal_positive_lag_weights = list(
            default_lag_weights if cross_modal_positive_lag_weights is None else cross_modal_positive_lag_weights
        )
        if not self.cross_modal_positive_lag_weights or max(self.cross_modal_positive_lag_weights) <= 0:
            raise ValueError('cross_modal positive_lag_weights must contain a positive value')
        self.cross_modal_token_mask_ratio = min(max(float(cross_modal_token_mask_ratio), 0.0), 1.0)
        self.cross_modal_modality_dropout_probability = min(
            max(float(cross_modal_modality_dropout_probability), 0.0), 1.0
        )
        fixed_eeg = torch.as_tensor(coupling_fixed_eeg_marginal or [], dtype=torch.float32)
        fixed_fnirs = torch.as_tensor(coupling_fixed_fnirs_marginal or [], dtype=torch.float32)
        self.register_buffer('coupling_fixed_eeg_marginal', fixed_eeg, persistent=False)
        self.register_buffer('coupling_fixed_fnirs_marginal', fixed_fnirs, persistent=False)
        self.source_target_weight = max(float(source_target_weight), 0.0)
        self.eeg_source_aux_weight = max(float(eeg_source_aux_weight), 0.0)
        self.source_target_correlation_weight = max(float(source_target_correlation_weight), 0.0)
        self.eeg_source_aux_correlation_weight = max(float(eeg_source_aux_correlation_weight), 0.0)
        self.observation_target_weight = max(float(observation_target_weight), 0.0)
        self.codebook_balance_weight = codebook_balance_weight
        self.source_balance_scale = max(float(source_balance_scale), 0.0)
        self.observation_balance_scale = max(float(observation_balance_scale), 0.0)
        self.default_source_balance_scale = float(self.source_balance_scale)
        self.default_observation_balance_scale = float(self.observation_balance_scale)
        self.orthogonality_weight = orthogonality_weight
        self.assignment_temperature = assignment_temperature
        self.balance_assignment_temperature = assignment_temperature
        self.source_balance_temperature = float(
            assignment_temperature if source_balance_temperature is None else source_balance_temperature
        )
        self.observation_balance_temperature = float(
            assignment_temperature if observation_balance_temperature is None else observation_balance_temperature
        )
        if coupling_max_lag_tokens is None:
            n_coupling_lags = max(int(self.n_patches), 1)
        else:
            max_lag = max(int(coupling_max_lag_tokens), 0)
            n_coupling_lags = min(max_lag + 1, max(int(self.n_patches), 1))
        self.coupling_lags = list(range(n_coupling_lags))
        self.n_coupling_lags = len(self.coupling_lags)
        if fixed_eeg.numel() and tuple(fixed_eeg.shape) != (self.n_coupling_lags, source_codebook_size):
            raise ValueError(
                "fixed_eeg_marginal must have shape "
                f"[{self.n_coupling_lags}, {source_codebook_size}], got {tuple(fixed_eeg.shape)}"
            )
        if fixed_fnirs.numel() and tuple(fixed_fnirs.shape) != (self.n_coupling_lags, source_codebook_size):
            raise ValueError(
                "fixed_fnirs_marginal must have shape "
                f"[{self.n_coupling_lags}, {source_codebook_size}], got {tuple(fixed_fnirs.shape)}"
            )

        self.source_branch_dropout = max(float(source_branch_dropout), 0.0)
        self.eeg_observation_branch_dropout = max(float(eeg_observation_branch_dropout), 0.0)
        self.fnirs_observation_branch_dropout = max(float(fnirs_observation_branch_dropout), 0.0)
        self.source_branch_enabled = self._is_branch_enabled(self.source_branch_dropout)
        self.eeg_observation_branch_enabled = self._is_branch_enabled(self.eeg_observation_branch_dropout)
        self.fnirs_observation_branch_enabled = self._is_branch_enabled(self.fnirs_observation_branch_dropout)
        self.alignment_scale = 1.0
        self.cross_modal_gradient_scale = 1.0
        self.source_target_scale = 1.0
        self.observation_target_scale = 1.0
        self.use_smooth_l1 = bool(use_smooth_l1)
        self.loss_fn = F.smooth_l1_loss if use_smooth_l1 else F.mse_loss
        self.window_duration_s = max(float(window_duration_s), 1e-3)
        self.eeg_sampling_rate_hz = float(self.eeg_seq_length) / self.window_duration_s
        self.fnirs_sampling_rate_hz = float(self.fnirs_seq_length) / self.window_duration_s
        self.eeg_to_fnirs_downsample_factor = (
            self.eeg_seq_length // self.fnirs_seq_length
            if self.eeg_seq_length % self.fnirs_seq_length == 0 else None
        )
        self.hrf_kernel_duration_s = max(float(hrf_kernel_duration_s), 1.0 / self.fnirs_sampling_rate_hz)
        self.hrf_peak_delay_s = nn.Parameter(torch.tensor(float(hrf_peak_delay_s), dtype=torch.float32))
        self.hrf_peak_dispersion_s = nn.Parameter(torch.tensor(float(hrf_peak_dispersion_s), dtype=torch.float32))
        self.hrf_undershoot_delay_s = nn.Parameter(torch.tensor(float(hrf_undershoot_delay_s), dtype=torch.float32))
        self.hrf_undershoot_dispersion_s = nn.Parameter(torch.tensor(float(hrf_undershoot_dispersion_s), dtype=torch.float32))
        self.hrf_peak_scale = nn.Parameter(torch.tensor(float(hrf_peak_scale), dtype=torch.float32))
        self.hrf_undershoot_scale = nn.Parameter(torch.tensor(float(hrf_undershoot_scale), dtype=torch.float32))
        self.eeg_channel_names = list(eeg_channel_names) if eeg_channel_names is not None else [f'eeg_{index}' for index in range(eeg_channels)]
        self.fnirs_channel_names = list(fnirs_channel_names) if fnirs_channel_names is not None else [f'fnirs_{index}' for index in range(fnirs_channels)]
        self.spatial_source_prior_requested = bool(spatial_source_prior_enabled)
        self.spatial_source_prior_enabled = bool(spatial_source_prior_enabled)
        self.eeg_source_target_mode = str(eeg_source_target_mode)
        self.eeg_rms_smoothing_ms = max(float(eeg_rms_smoothing_ms), 0.0)
        self.eeg_rms_smoothing_samples = max(
            int(round(self.eeg_sampling_rate_hz * self.eeg_rms_smoothing_ms / 1000.0)),
            1,
        )
        self.shared_state_alpha = max(0.0, min(float(shared_state_alpha), 1.0))
        self.eeg_source_target_smoothing_ms = max(float(eeg_source_target_smoothing_ms), 0.0)
        self.eeg_source_target_smoothing_samples = max(
            int(round(self.eeg_sampling_rate_hz * self.eeg_source_target_smoothing_ms / 1000.0)),
            1,
        )
        if self.eeg_source_target_mode not in {'rms_envelope', 'signed_rms_carrier'}:
            raise ValueError(
                "eeg_source_target_mode must be 'rms_envelope' or 'signed_rms_carrier'"
            )
        self.spatial_adjacency_info: Optional[SpatialAdjacencyInfo] = None
        self.spatial_prior_warnings: List[str] = []
        self.spatial_prior_reference_subject_id = int(spatial_prior_reference_subject_id)
        adjacency_buffer = torch.empty((0, 0), dtype=torch.float32)
        if self.spatial_source_prior_enabled and dataset_id and data_root and eeg_channel_names and fnirs_channel_names:
            try:
                self.spatial_adjacency_info = build_channel_adjacency(
                    dataset_id=str(dataset_id),
                    data_root=str(data_root),
                    eeg_channel_names=self.eeg_channel_names,
                    fnirs_channel_names=self.fnirs_channel_names,
                    reference_subject_id=self.spatial_prior_reference_subject_id,
                    use_artifact_data=bool(spatial_prior_use_artifact_data),
                )
                adjacency_buffer = torch.as_tensor(self.spatial_adjacency_info.adjacency_matrix, dtype=torch.float32)
                self.spatial_prior_warnings = list(self.spatial_adjacency_info.warnings)
            except Exception as exc:
                self.spatial_source_prior_enabled = False
                self.spatial_prior_warnings = [f'Failed to initialize spatial prior: {exc}']
        elif self.spatial_source_prior_enabled:
            self.spatial_source_prior_enabled = False
            self.spatial_prior_warnings = ['Spatial prior requested but dataset_id/data_root/channel names were incomplete.']
        self.register_buffer('fnirs_spatial_adjacency', adjacency_buffer, persistent=False)

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
            learnable_codebook_transform=source_simvq_enabled,
            codebook_transform_loss_weight=source_simvq_loss_weight,
        )
        self.fnirs_source_quantizer = NormEMAVectorQuantizer(
            n_embed=source_codebook_size,
            embedding_dim=fnirs_source_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
            learnable_codebook_transform=source_simvq_enabled,
            codebook_transform_loss_weight=source_simvq_loss_weight,
        )
        self.shared_source_quantizer = (
            NormEMAVectorQuantizer(
                n_embed=source_codebook_size,
                embedding_dim=eeg_source_codebook_dim,
                beta=beta,
                decay=decay,
                kmeans_init=kmeans_init,
                revive_dead_codes=revive_dead_codes,
                dead_code_threshold=dead_code_threshold,
                learnable_codebook_transform=source_simvq_enabled,
                codebook_transform_loss_weight=source_simvq_loss_weight,
            )
            if self.source_codebook_mode == 'shared_joint' else None
        )
        self.eeg_observation_quantizer = NormEMAVectorQuantizer(
            n_embed=eeg_observation_codebook_size,
            embedding_dim=eeg_observation_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
            learnable_codebook_transform=observation_simvq_enabled,
            codebook_transform_loss_weight=observation_simvq_loss_weight,
        )
        self.fnirs_observation_quantizer = NormEMAVectorQuantizer(
            n_embed=fnirs_observation_codebook_size,
            embedding_dim=fnirs_observation_codebook_dim,
            beta=beta,
            decay=decay,
            kmeans_init=kmeans_init,
            revive_dead_codes=revive_dead_codes,
            dead_code_threshold=dead_code_threshold,
            learnable_codebook_transform=observation_simvq_enabled,
            codebook_transform_loss_weight=observation_simvq_loss_weight,
        )

        self.quantizer = self.eeg_source_quantizer
        self.quantization_strength = 1.0
        self.coupling_logits = nn.Parameter(
            torch.zeros(self.n_coupling_lags, source_codebook_size, source_codebook_size)
        )
        nn.init.trunc_normal_(self.coupling_logits, std=0.02)
        self.context_coupling_eeg_factors = nn.Parameter(
            torch.zeros(
                self.coupling_context_states,
                self.n_coupling_lags,
                source_codebook_size,
                self.coupling_context_rank,
            )
        )
        self.context_coupling_fnirs_factors = nn.Parameter(
            torch.zeros(
                self.coupling_context_states,
                self.n_coupling_lags,
                source_codebook_size,
                self.coupling_context_rank,
            )
        )
        nn.init.trunc_normal_(self.context_coupling_eeg_factors, std=0.02)
        nn.init.trunc_normal_(self.context_coupling_fnirs_factors, std=0.02)
        if self.coupling_context_router_type == 'learned' and self.coupling_context_states > 1:
            self.context_coupling_router = nn.Sequential(
                nn.LayerNorm(2 * source_codebook_size),
                nn.Linear(2 * source_codebook_size, self.coupling_context_states),
            )
        else:
            self.context_coupling_router = None
        self.eeg_to_fnirs_source_predictor = nn.Sequential(
            nn.LayerNorm(eeg_source_codebook_dim),
            nn.Linear(eeg_source_codebook_dim, fnirs_source_codebook_dim),
        )
        self.fnirs_to_eeg_source_predictor = nn.Sequential(
            nn.LayerNorm(fnirs_source_codebook_dim),
            nn.Linear(fnirs_source_codebook_dim, eeg_source_codebook_dim),
        )
        self.eeg_shared_state_proj = nn.Sequential(
            nn.LayerNorm(eeg_source_codebook_dim),
            nn.Linear(eeg_source_codebook_dim, self.shared_state_bottleneck_dim),
        )
        self.fnirs_shared_state_proj = nn.Sequential(
            nn.LayerNorm(fnirs_source_codebook_dim),
            nn.Linear(fnirs_source_codebook_dim, self.shared_state_bottleneck_dim),
        )
        if self.cross_modal_exchange_enabled:
            self.cross_modal_exchange = CausalLowRankCrossAdapter(
                eeg_dim=eeg_encoder_embed_dim,
                fnirs_dim=fnirs_encoder_embed_dim,
                rank=max(int(cross_modal_exchange_rank), 1),
                adapter_dim=max(int(cross_modal_exchange_adapter_dim), 1),
                max_lag_tokens=self.cross_modal_exchange_max_lag_tokens,
                residual_init=float(cross_modal_exchange_residual_init),
                dropout=max(float(cross_modal_exchange_dropout), 0.0),
            )
        else:
            self.cross_modal_exchange = None
        self.cross_modal_fusion = (
            LagAwareCrossModalFusion(
                eeg_dim=eeg_source_codebook_dim,
                fnirs_dim=fnirs_source_codebook_dim,
                embed_dim=max(int(cross_modal_fusion_embed_dim), 1),
                depth=max(int(cross_modal_fusion_depth), 1),
                num_heads=max(int(cross_modal_fusion_num_heads), 1),
                max_lag_tokens=self.cross_modal_fusion_max_lag_tokens,
                relative_lag_bias=bool(cross_modal_fusion_relative_lag_bias),
                dropout=max(float(cross_modal_fusion_dropout), 0.0),
                mode=self.cross_modal_fusion_mode,
            )
            if self.cross_modal_fusion_enabled else None
        )
        alignment_dim = max(min(eeg_source_codebook_dim, fnirs_source_codebook_dim), 8)
        self.cross_modal_eeg_projection = nn.Sequential(
            nn.LayerNorm(eeg_source_codebook_dim),
            nn.Linear(eeg_source_codebook_dim, alignment_dim),
        )
        self.cross_modal_fnirs_projection = nn.Sequential(
            nn.LayerNorm(fnirs_source_codebook_dim),
            nn.Linear(fnirs_source_codebook_dim, alignment_dim),
        )
        self.eeg_source_mask_token = nn.Parameter(torch.zeros(1, 1, eeg_source_codebook_dim))
        self.fnirs_source_mask_token = nn.Parameter(torch.zeros(1, 1, fnirs_source_codebook_dim))

        self.eeg_source_decode_input_proj = nn.Linear(
            eeg_source_codebook_dim,
            eeg_decoder_embed_dim,
        )
        self.eeg_observation_decode_input_proj = nn.Linear(
            eeg_observation_codebook_dim,
            eeg_decoder_embed_dim,
        )
        self.fnirs_source_decode_input_proj = nn.Linear(
            fnirs_source_codebook_dim,
            fnirs_decoder_embed_dim,
        )
        self.fnirs_observation_decode_input_proj = nn.Linear(
            fnirs_observation_codebook_dim,
            fnirs_decoder_embed_dim,
        )

        self.eeg_source_decoder = TransformerDecoder(
            embed_dim=eeg_decoder_embed_dim,
            depth=eeg_decoder_depth,
            num_heads=eeg_decoder_num_heads,
            dropout=dropout,
            drop_path=0.0,
            max_patches=self.n_patches,
        )
        self.eeg_observation_decoder = TransformerDecoder(
            embed_dim=eeg_decoder_embed_dim,
            depth=eeg_decoder_depth,
            num_heads=eeg_decoder_num_heads,
            dropout=dropout,
            drop_path=0.0,
            max_patches=self.n_patches,
        )
        self.fnirs_source_decoder = TransformerDecoder(
            embed_dim=fnirs_decoder_embed_dim,
            depth=fnirs_decoder_depth,
            num_heads=fnirs_decoder_num_heads,
            dropout=dropout,
            drop_path=0.0,
            max_patches=self.n_patches,
        )
        self.fnirs_observation_decoder = TransformerDecoder(
            embed_dim=fnirs_decoder_embed_dim,
            depth=fnirs_decoder_depth,
            num_heads=fnirs_decoder_num_heads,
            dropout=dropout,
            drop_path=0.0,
            max_patches=self.n_patches,
        )

        self.eeg_source_amplitude_head = nn.Linear(eeg_decoder_embed_dim, eeg_channels * self.eeg_fft_size)
        self.eeg_source_phase_head = nn.Linear(eeg_decoder_embed_dim, eeg_channels * self.eeg_fft_size)
        self.eeg_observation_amplitude_head = nn.Linear(eeg_decoder_embed_dim, eeg_channels * self.eeg_fft_size)
        self.eeg_observation_phase_head = nn.Linear(eeg_decoder_embed_dim, eeg_channels * self.eeg_fft_size)
        self.fnirs_source_amplitude_head = nn.Linear(fnirs_decoder_embed_dim, fnirs_channels * self.fnirs_fft_size)
        self.fnirs_source_phase_head = nn.Linear(fnirs_decoder_embed_dim, fnirs_channels * self.fnirs_fft_size)
        self.fnirs_observation_amplitude_head = nn.Linear(fnirs_decoder_embed_dim, fnirs_channels * self.fnirs_fft_size)
        self.fnirs_observation_phase_head = nn.Linear(fnirs_decoder_embed_dim, fnirs_channels * self.fnirs_fft_size)

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
        source_target_cfg = loss_cfg.get('source_target', {})
        observation_target_cfg = loss_cfg.get('observation_target', {})
        branch_cfg = loss_cfg.get('branch', {})
        codebook_cfg = loss_cfg.get('codebook', {})
        exchange_cfg = model_cfg.get('cross_modal_exchange', {})
        fusion_cfg = model_cfg.get('cross_modal_fusion', {})
        source_codebook_cfg = model_cfg.get('source_codebook', {})
        cross_alignment_cfg = loss_cfg.get('cross_modal_alignment', {})
        data_cfg = config.get('data', {})
        window_cfg = data_cfg.get('window', {})
        hrf_cfg = source_target_cfg.get('hrf', {})
        spatial_cfg = source_target_cfg.get('spatial', {})

        source_codebook_size = source_cfg.get('codebook_size', 128)
        fnirs_spatial_anchors = fnirs_cfg.get('spatial_anchors')
        fnirs_optical_components = fnirs_cfg.get('optical_components')
        if fnirs_cfg.get('channels') is not None:
            fnirs_channels = int(fnirs_cfg.get('channels'))
        elif fnirs_spatial_anchors is not None and fnirs_optical_components is not None:
            fnirs_channels = int(fnirs_spatial_anchors) * int(fnirs_optical_components)
        else:
            fnirs_channels = 36
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
            fnirs_channels=fnirs_channels,
            fnirs_spatial_anchors=fnirs_spatial_anchors,
            fnirs_optical_components=fnirs_optical_components,
            fnirs_component_labels=fnirs_cfg.get('component_labels', fnirs_cfg.get('fnirs_component_labels')),
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
            eeg_time_weight=reconstruction_cfg.get('eeg_time_weight', 0.9),
            fnirs_amplitude_weight=reconstruction_cfg.get('fnirs_amplitude_weight', 1.0),
            fnirs_time_weight=reconstruction_cfg.get('fnirs_time_weight', 1.0),
            coupling_weight=coupling_cfg.get('weight', 0.0),
            coupling_lag_focus_weight=coupling_cfg.get('lag_focus_weight', 1.0),
            coupling_joint_entropy_weight=coupling_cfg.get('joint_entropy_weight', 0.0),
            coupling_smoothness_weight=coupling_cfg.get('smoothness_weight', 0.2),
            coupling_smoothness_neighbors=coupling_cfg.get('smoothness_neighbors', 5),
            coupling_pair_likelihood_weight=coupling_cfg.get('pair_likelihood_weight', 0.0),
            coupling_pair_detach_tokens=coupling_cfg.get('pair_detach_tokens', True),
            coupling_pair_gradient_target=coupling_cfg.get('pair_gradient_target'),
            coupling_pair_temperature=coupling_cfg.get('pair_temperature'),
            coupling_residualize_fnirs_marginal=coupling_cfg.get('residualize_fnirs_marginal', False),
            coupling_lag_evidence_weight=coupling_cfg.get('lag_evidence_weight', 0.0),
            coupling_lag_evidence_temperature=coupling_cfg.get('lag_evidence_temperature', 0.25),
            coupling_max_lag_tokens=coupling_cfg.get('max_lag_tokens'),
            coupling_fixed_eeg_marginal=coupling_cfg.get('fixed_eeg_marginal'),
            coupling_fixed_fnirs_marginal=coupling_cfg.get('fixed_fnirs_marginal'),
            coupling_effective_smoothness_weight=coupling_cfg.get('effective_smoothness_weight', 0.0),
            coupling_interaction_lag_sparsity_weight=coupling_cfg.get('interaction_lag_sparsity_weight', 0.0),
            coupling_local_residual_enabled=coupling_cfg.get('local_residual_enabled', False),
            coupling_local_residual_pair_weight=coupling_cfg.get('local_residual_pair_weight', 1.0),
            coupling_local_residual_alpha=coupling_cfg.get('local_residual_alpha', 0.5),
            coupling_context_residual_enabled=coupling_cfg.get('context_residual_enabled', False),
            coupling_context_states=coupling_cfg.get('context_states', 4),
            coupling_context_rank=coupling_cfg.get('context_rank', 16),
            coupling_context_router_type=coupling_cfg.get('context_router_type', 'learned'),
            coupling_context_pair_weight=coupling_cfg.get('context_pair_weight', 1.0),
            coupling_context_entropy_weight=coupling_cfg.get('context_entropy_weight', 0.01),
            coupling_context_balance_weight=coupling_cfg.get('context_balance_weight', 0.01),
            coupling_context_residual_l1_weight=coupling_cfg.get('context_residual_l1_weight', 0.001),
            coupling_context_gradient_target=coupling_cfg.get('context_gradient_target'),
            interaction_aux_weight=loss_cfg.get('interaction_aux', {}).get('weight', 0.0),
            interaction_aux_direction=loss_cfg.get('interaction_aux', {}).get('direction', 'eeg_to_fnirs'),
            interaction_aux_stop_gradient=loss_cfg.get('interaction_aux', {}).get('stop_gradient', True),
            shared_state_bottleneck_weight=loss_cfg.get('shared_state_bottleneck', {}).get('weight', 0.0),
            shared_state_bottleneck_dim=loss_cfg.get('shared_state_bottleneck', {}).get('dim', 32),
            shared_state_bottleneck_stop_gradient=loss_cfg.get('shared_state_bottleneck', {}).get('stop_gradient', True),
            cross_modal_exchange_enabled=exchange_cfg.get('enabled', False),
            cross_modal_exchange_mode=exchange_cfg.get('mode', 'low_rank_causal_adapter'),
            cross_modal_exchange_direction=exchange_cfg.get('direction', 'eeg_to_fnirs'),
            cross_modal_exchange_target_branch=exchange_cfg.get('target_branch', 'fnirs_source'),
            cross_modal_exchange_rank=exchange_cfg.get('rank', 16),
            cross_modal_exchange_adapter_dim=exchange_cfg.get('adapter_dim', 64),
            cross_modal_exchange_max_lag_tokens=exchange_cfg.get('max_lag_tokens', 5),
            cross_modal_exchange_detach_context=exchange_cfg.get('detach_context', True),
            cross_modal_exchange_residual_init=exchange_cfg.get('residual_init', 0.1),
            cross_modal_exchange_dropout=exchange_cfg.get('dropout', 0.05),
            cross_modal_fusion_enabled=fusion_cfg.get('enabled', False),
            cross_modal_fusion_mode=fusion_cfg.get('mode', 'causal_cross_attention'),
            cross_modal_fusion_embed_dim=fusion_cfg.get('embed_dim', 128),
            cross_modal_fusion_depth=fusion_cfg.get('depth', 2),
            cross_modal_fusion_num_heads=fusion_cfg.get('num_heads', 4),
            cross_modal_fusion_max_lag_tokens=fusion_cfg.get('max_lag_tokens', 5),
            cross_modal_fusion_relative_lag_bias=fusion_cfg.get('relative_lag_bias', True),
            cross_modal_fusion_dropout=fusion_cfg.get('dropout', 0.1),
            source_codebook_mode=source_codebook_cfg.get('mode', 'independent'),
            cross_modal_temporal_nce_weight=cross_alignment_cfg.get('temporal_nce_weight', 0.0),
            cross_modal_masked_latent_weight=cross_alignment_cfg.get('masked_latent_weight', 0.0),
            cross_modal_soft_code_weight=cross_alignment_cfg.get('soft_code_distillation_weight', 0.0),
            cross_modal_alignment_temperature=cross_alignment_cfg.get('temperature', 0.1),
            cross_modal_positive_lag_weights=cross_alignment_cfg.get('positive_lag_weights'),
            cross_modal_token_mask_ratio=cross_alignment_cfg.get('token_mask_ratio', 0.5),
            cross_modal_modality_dropout_probability=cross_alignment_cfg.get(
                'modality_dropout_probability', 0.25
            ),
            source_target_weight=source_target_cfg.get('weight', source_target_cfg.get('source_target_weight', 0.0)),
            eeg_source_aux_weight=source_target_cfg.get('eeg_aux_weight', source_target_cfg.get('eeg_source_aux_weight', 0.5)),
            source_target_correlation_weight=source_target_cfg.get(
                'correlation_weight',
                source_target_cfg.get('source_correlation_weight', 0.0),
            ),
            eeg_source_aux_correlation_weight=source_target_cfg.get(
                'eeg_aux_correlation_weight',
                source_target_cfg.get('eeg_source_aux_correlation_weight', 0.0),
            ),
            observation_target_weight=observation_target_cfg.get('weight', 0.0),
            codebook_balance_weight=codebook_cfg.get('balance_weight', 0.02),
            source_balance_scale=codebook_cfg.get('source_balance_scale', 1.0),
            observation_balance_scale=codebook_cfg.get('observation_balance_scale', 1.0),
            orthogonality_weight=branch_cfg.get('orthogonality_weight', 0.01),
            assignment_temperature=codebook_cfg.get('assignment_temperature', 1.0),
            source_balance_temperature=codebook_cfg.get('source_assignment_temperature'),
            observation_balance_temperature=codebook_cfg.get('observation_assignment_temperature'),
            source_simvq_enabled=quantizer_cfg.get('source_simvq_enabled', False),
            source_simvq_loss_weight=quantizer_cfg.get('source_simvq_loss_weight', 1.0),
            observation_simvq_enabled=quantizer_cfg.get('observation_simvq_enabled', False),
            observation_simvq_loss_weight=quantizer_cfg.get('observation_simvq_loss_weight', 1.0),
            source_branch_dropout=branch_dropout_cfg.get('source', 0.0),
            eeg_observation_branch_dropout=branch_dropout_cfg.get('eeg_observation', 0.0),
            fnirs_observation_branch_dropout=branch_dropout_cfg.get('fnirs_observation', 0.0),
            window_duration_s=window_cfg.get('duration_s', 10.0),
            hrf_kernel_duration_s=hrf_cfg.get('kernel_duration_s', 24.0),
            hrf_peak_delay_s=hrf_cfg.get('peak_delay_s', 6.0),
            hrf_peak_dispersion_s=hrf_cfg.get('peak_dispersion_s', 1.0),
            hrf_undershoot_delay_s=hrf_cfg.get('undershoot_delay_s', 16.0),
            hrf_undershoot_dispersion_s=hrf_cfg.get('undershoot_dispersion_s', 1.5),
            hrf_peak_scale=hrf_cfg.get('peak_scale', 1.0),
            hrf_undershoot_scale=hrf_cfg.get('undershoot_scale', 0.25),
            spatial_source_prior_enabled=spatial_cfg.get('enabled', False),
            eeg_source_target_mode=source_target_cfg.get('eeg_target_mode', 'signed_rms_carrier'),
            shared_state_alpha=source_target_cfg.get('shared_state_alpha', 0.90),
            eeg_source_target_smoothing_ms=source_target_cfg.get(
                'eeg_target_smoothing_ms',
                spatial_cfg.get('eeg_rms_smoothing_ms', 200.0),
            ),
            eeg_rms_smoothing_ms=spatial_cfg.get('eeg_rms_smoothing_ms', 200.0),
            dataset_id=data_cfg.get('dataset'),
            data_root=data_cfg.get('data_root'),
            eeg_channel_names=data_cfg.get('eeg_channel_names'),
            fnirs_channel_names=data_cfg.get('fnirs_channel_names'),
            spatial_prior_reference_subject_id=spatial_cfg.get('reference_subject_id', 1),
            spatial_prior_use_artifact_data=data_cfg.get('use_artifact_data', True),
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

    def _compute_fft_amplitude_targets(
        self,
        patches: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        shaped_window = window.to(device=patches.device, dtype=patches.dtype).view(1, 1, 1, -1)
        fft = torch.fft.rfft(patches * shaped_window, dim=-1)
        return torch.log1p(torch.abs(fft))

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

    def _gamma_pdf(self, time_axis: torch.Tensor, delay_s: torch.Tensor, dispersion_s: torch.Tensor) -> torch.Tensor:
        delay = delay_s.clamp_min(1e-3)
        dispersion = dispersion_s.clamp_min(1e-3)
        concentration = (delay / dispersion).clamp_min(1e-3) + 1.0
        safe_time = time_axis.clamp_min(1e-6)
        log_pdf = (
            (concentration - 1.0) * torch.log(safe_time) -
            (safe_time / dispersion) -
            torch.lgamma(concentration) -
            concentration * torch.log(dispersion)
        )
        return torch.exp(log_pdf)

    def _build_hrf_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        kernel_steps = max(int(round(self.hrf_kernel_duration_s * self.fnirs_sampling_rate_hz)), 1)
        time_axis = torch.arange(kernel_steps, device=device, dtype=dtype) / self.fnirs_sampling_rate_hz
        peak = self._gamma_pdf(time_axis, self.hrf_peak_delay_s.to(device=device, dtype=dtype), self.hrf_peak_dispersion_s.to(device=device, dtype=dtype))
        undershoot = self._gamma_pdf(
            time_axis,
            self.hrf_undershoot_delay_s.to(device=device, dtype=dtype),
            self.hrf_undershoot_dispersion_s.to(device=device, dtype=dtype),
        )
        peak_scale = self.hrf_peak_scale.to(device=device, dtype=dtype).clamp_min(1e-3)
        undershoot_scale = self.hrf_undershoot_scale.to(device=device, dtype=dtype).clamp_min(0.0)
        kernel = peak_scale * peak - undershoot_scale * undershoot
        kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
        return kernel.view(1, 1, -1)

    def _compute_eeg_power_envelope(self, eeg: torch.Tensor) -> torch.Tensor:
        return eeg.pow(2).mean(dim=1, keepdim=True)

    def _compute_shared_neural_state(self, eeg: torch.Tensor) -> torch.Tensor:
        """Estimate shared latent neural state via AR(1)-smoothed EEG power.

        Croce et al. 2017 physical model: a latent neural driver s(t) is
        observed instantaneously by EEG (as broadband power) and with HRF delay
        by fNIRS.  We approximate the posterior mean with a causal exponential
        smoother:

            s_k = α · s_{k-1}  +  (1 − α) · x_k

        where x_k is the channel-averaged (or spatially-weighted) EEG power
        downsampled to the fNIRS rate, and α = shared_state_alpha controls
        temporal smoothness:

        α → 1   : only sub-0.1 Hz hemodynamic fluctuations (SMC limit)
        α ≈ 0.90: ~1 s half-life — alpha/beta-band power envelope preserved
        α → 0   : raw EEG power, no smoothing

        When spatial_source_prior is enabled, x_k is computed per fNIRS
        channel via the adjacency matrix, yielding [B, F, T_fnirs].
        Otherwise x_k is the global mean power → [B, 1, T_fnirs].

        Returns:
            shared_state: [B, 1_or_F, T_fnirs] slow neural driver at fNIRS rate
        """
        alpha = self.shared_state_alpha
        if self.spatial_source_prior_enabled and self.fnirs_spatial_adjacency.numel() > 0:
            # Per-fNIRS-channel spatially-weighted EEG power, then AR-smooth each
            eeg_power = eeg.pow(2)  # [B, E, T_eeg]
            weighted = eeg_power @ self.fnirs_spatial_adjacency.T.to(
                device=eeg.device, dtype=eeg.dtype
            )  # note: adjacency is [F, E], so adjacency.T is [E, F]
            # Actually adjacency is [F, E], we need einsum:
            # weighted[b, f, t] = sum_e adj[f, e] * eeg_power[b, e, t]
            weighted = torch.einsum(
                'fe,bet->bft',
                self.fnirs_spatial_adjacency.to(device=eeg.device, dtype=eeg.dtype),
                eeg_power,
            )
            driver = self._downsample_neural_driver(weighted, self.fnirs_seq_length)  # [B, F, T_fnirs]
            if alpha <= 0.0:
                return driver
            smoothed = driver.clone()
            for t in range(1, smoothed.shape[-1]):
                smoothed[:, :, t] = alpha * smoothed[:, :, t - 1] + (1.0 - alpha) * driver[:, :, t]
            return smoothed

        # Global mode: channel-averaged power
        eeg_power = self._compute_eeg_power_envelope(eeg)          # [B, 1, T_eeg]
        driver = self._downsample_neural_driver(eeg_power, self.fnirs_seq_length)  # [B, 1, T_fnirs]
        if alpha <= 0.0:
            return driver
        smoothed = driver.clone()
        for t in range(1, smoothed.shape[-1]):
            smoothed[:, :, t] = alpha * smoothed[:, :, t - 1] + (1.0 - alpha) * driver[:, :, t]
        return smoothed

    def _apply_hrf_convolution(self, driver: torch.Tensor) -> torch.Tensor:
        hrf_kernel = self._build_hrf_kernel(device=driver.device, dtype=driver.dtype)
        batch_size, channels, time_steps = driver.shape
        flat_driver = driver.reshape(batch_size * channels, 1, time_steps)
        conv = F.conv1d(flat_driver, hrf_kernel, padding=hrf_kernel.shape[-1] - 1)
        conv = conv[..., :self.fnirs_seq_length]
        return conv.reshape(batch_size, channels, self.fnirs_seq_length)

    def _downsample_neural_driver(self, driver: torch.Tensor, target_length: int) -> torch.Tensor:
        if driver.shape[-1] == target_length:
            return driver
        if self.eeg_to_fnirs_downsample_factor is not None and target_length == self.fnirs_seq_length:
            factor = max(int(self.eeg_to_fnirs_downsample_factor), 1)
            return F.avg_pool1d(driver, kernel_size=factor, stride=factor)
        return F.interpolate(driver, size=target_length, mode='linear', align_corners=False)

    def _match_fnirs_target_scale(self, base_target: torch.Tensor, fnirs: torch.Tensor) -> torch.Tensor:
        centered = base_target - base_target.mean(dim=-1, keepdim=True)
        normalized = centered / centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        channel_mean = fnirs.mean(dim=-1, keepdim=True)
        channel_std = fnirs.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return normalized * channel_std + channel_mean

    def _compute_fnirs_source_target(self, shared_state: torch.Tensor, fnirs: torch.Tensor) -> torch.Tensor:
        """fNIRS source target from shared neural state via HRF convolution.

        Croce et al. physical model: fNIRS observes the HRF-convolved neural
        state.  The shared_state is already temporally smoothed and at fNIRS
        rate; we apply the double-gamma HRF and rescale to match per-channel
        fNIRS statistics.  The result is time-synchronous with the original
        fNIRS (HRF convolution absorbs the neurovascular delay).

        When the spatial prior is enabled, shared_state is [B, F, T_fnirs] and
        each fNIRS channel receives its own spatially-weighted driver.
        Otherwise shared_state is [B, 1, T_fnirs] broadcast to all channels.

        Args:
            shared_state: [B, 1_or_F, T_fnirs] from _compute_shared_neural_state
            fnirs: [B, F, T_fnirs] original fNIRS for scale matching
        """
        conv = self._apply_hrf_convolution(shared_state)
        return self._match_fnirs_target_scale(conv, fnirs)

    def _smooth_eeg_waveform(self, eeg: torch.Tensor) -> torch.Tensor:
        kernel_size = max(int(self.eeg_source_target_smoothing_samples), 1)
        if kernel_size <= 1:
            return eeg
        if kernel_size % 2 == 0:
            kernel_size += 1
        batch_size, channels, time_steps = eeg.shape
        working = eeg.reshape(batch_size * channels, 1, time_steps)
        padding = kernel_size // 2
        padded = F.pad(working, (padding, padding), mode='reflect')
        smoothed = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
        return smoothed.reshape(batch_size, channels, time_steps)

    def _compute_eeg_source_target(self, eeg: torch.Tensor) -> torch.Tensor:
        """EEG source target: temporally smoothed, signed, same physical units as EEG.

        Two modes (via eeg_source_target_mode):
        - 'signed_rms_carrier' (DEFAULT): RMS amplitude × sign(smoothed voltage)
          → signed, μV units, additive decomposition physically meaningful
        - 'rms_envelope': per-channel RMS envelope
          → non-negative, μV units, preserves spatial structure

        Temporal smoothing uses shared_state_alpha (AR(1) exponential) to
        remove fast noise while preserving task-relevant power dynamics.
        """
        alpha = self.shared_state_alpha

        if self.eeg_source_target_mode == 'signed_rms_carrier':
            if self.spatial_source_prior_enabled:
                amplitude = compute_per_channel_rms_envelope(
                    eeg,
                    smoothing_samples=self.eeg_rms_smoothing_samples,
                )
                # Apply temporal smoothing per channel
                if alpha > 0.0:
                    smoothed_amp = amplitude.clone()
                    for t in range(1, smoothed_amp.shape[-1]):
                        smoothed_amp[:, :, t] = (
                            alpha * smoothed_amp[:, :, t - 1]
                            + (1.0 - alpha) * amplitude[:, :, t]
                        )
                    amplitude = smoothed_amp
            else:
                # Global amplitude from channel-averaged power, smoothed, sqrt → μV
                power = self._compute_eeg_power_envelope(eeg).expand(-1, eeg.shape[1], -1)
                if alpha > 0.0:
                    smoothed_power = power.clone()
                    for t in range(1, smoothed_power.shape[-1]):
                        smoothed_power[:, :, t] = (
                            alpha * smoothed_power[:, :, t - 1]
                            + (1.0 - alpha) * power[:, :, t]
                        )
                    power = smoothed_power
                amplitude = torch.sqrt(power.clamp_min(1e-8))

            signed_carrier = self._smooth_eeg_waveform(eeg)
            carrier_std = signed_carrier.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return amplitude * torch.tanh(signed_carrier / carrier_std)

        if self.spatial_source_prior_enabled:
            amplitude = compute_per_channel_rms_envelope(
                eeg,
                smoothing_samples=self.eeg_rms_smoothing_samples,
            )
            if alpha > 0.0:
                smoothed_amp = amplitude.clone()
                for t in range(1, smoothed_amp.shape[-1]):
                    smoothed_amp[:, :, t] = (
                        alpha * smoothed_amp[:, :, t - 1]
                        + (1.0 - alpha) * amplitude[:, :, t]
                    )
                amplitude = smoothed_amp
            return amplitude

        # Global RMS envelope: channel-averaged power → sqrt for voltage units → broadcast
        power = self._compute_eeg_power_envelope(eeg).expand(-1, eeg.shape[1], -1)
        if alpha > 0.0:
            smoothed_power = power.clone()
            for t in range(1, smoothed_power.shape[-1]):
                smoothed_power[:, :, t] = (
                    alpha * smoothed_power[:, :, t - 1]
                    + (1.0 - alpha) * power[:, :, t]
                )
            power = smoothed_power
        return torch.sqrt(power.clamp_min(1e-8))

    def _compute_observation_target(self, original: torch.Tensor, source_target: torch.Tensor) -> torch.Tensor:
        return original - source_target

    def _compute_source_target_random_baseline(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if target.shape[0] > 1:
            randomized_target = target.roll(shifts=1, dims=0)
        else:
            shift = max(target.shape[-1] // 3, 1)
            randomized_target = target.roll(shifts=shift, dims=-1)
        return self.loss_fn(prediction.detach(), randomized_target)

    def _signal_correlation(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_flat = prediction.reshape(prediction.shape[0], -1).float()
        target_flat = target.reshape(target.shape[0], -1).float()
        prediction_centered = prediction_flat - prediction_flat.mean(dim=-1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=-1, keepdim=True)
        numerator = (prediction_centered * target_centered).sum(dim=-1)
        denominator = (
            prediction_centered.square().sum(dim=-1).sqrt()
            * target_centered.square().sum(dim=-1).sqrt()
        ).clamp_min(1e-6)
        return (numerator / denominator).clamp(-1.0, 1.0).mean()

    def _signal_correlation_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._signal_correlation(prediction, target)

    def _randomized_signal_correlation_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if target.shape[0] > 1:
            randomized_target = target.roll(shifts=1, dims=0)
        else:
            shift = max(target.shape[-1] // 3, 1)
            randomized_target = target.roll(shifts=shift, dims=-1)
        return self._signal_correlation_loss(prediction.detach(), randomized_target)

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

    def _decode_branch_reconstruction(
        self,
        z_q: torch.Tensor,
        decode_input_proj: nn.Module,
        decoder: nn.Module,
        amplitude_head: nn.Module,
        phase_head: nn.Module,
        channels: int,
        fft_size: int,
        patch_size: int,
    ) -> torch.Tensor:
        amplitude, phase = self._decode_modality(
            z_q,
            decode_input_proj,
            decoder,
            amplitude_head,
            phase_head,
            channels,
            fft_size,
        )
        return self._reconstruct_time(amplitude, phase, patch_size)

    def _zeros_like_reconstruction(
        self,
        latent: torch.Tensor,
        channels: int,
        seq_length: int,
    ) -> torch.Tensor:
        return latent.new_zeros((latent.shape[0], channels, seq_length))

    def _assignment_logits(self, z: torch.Tensor, codebook_weight: torch.Tensor) -> torch.Tensor:
        normalized_z = l2norm(z)
        return torch.einsum('bnd,kd->bnk', normalized_z, codebook_weight)

    def _context_coupling_probs(
        self,
        eeg_source_logits: torch.Tensor,
        fnirs_source_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.coupling_context_states <= 1 or self.context_coupling_router is None:
            return eeg_source_logits.new_ones((eeg_source_logits.shape[0], 1))
        temperature = max(
            float(
                self.source_balance_temperature
                if self.coupling_pair_temperature is None else self.coupling_pair_temperature
            ),
            1e-3,
        )
        eeg_probs = F.softmax(eeg_source_logits.detach() / temperature, dim=-1).mean(dim=1)
        fnirs_probs = F.softmax(fnirs_source_logits.detach() / temperature, dim=-1).mean(dim=1)
        router_input = torch.cat([eeg_probs, fnirs_probs], dim=-1)
        return F.softmax(self.context_coupling_router(router_input), dim=-1)

    @staticmethod
    def _is_branch_enabled(dropout: float) -> bool:
        return float(dropout) < (1.0 - 1e-6)

    def _branch_dropout(self, z: torch.Tensor, p: float) -> torch.Tensor:
        if not self._is_branch_enabled(p):
            return torch.zeros_like(z)
        if (not self.training) or p <= 0.0:
            return z
        keep_prob = 1.0 - p
        mask = torch.bernoulli(torch.full((z.shape[0], 1, 1), keep_prob, device=z.device, dtype=z.dtype))
        return z * mask / max(keep_prob, 1e-6)

    def _code_overlap(self, left_indices: torch.Tensor, right_indices: torch.Tensor) -> torch.Tensor:
        left_codes = torch.unique(left_indices.reshape(-1))
        right_codes = torch.unique(right_indices.reshape(-1))
        if left_codes.numel() == 0 and right_codes.numel() == 0:
            return left_indices.new_tensor(0.0, dtype=torch.float32)
        overlap = torch.isin(left_codes, right_codes).sum()
        union = left_codes.numel() + right_codes.numel() - overlap
        return overlap.to(dtype=torch.float32) / union.clamp_min(1).to(dtype=torch.float32)

    def _compute_source_alignment_state(
        self,
        eeg_source_logits: torch.Tensor,
        fnirs_source_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        lag_focus_loss = coupling_lag_focus_loss(self.coupling_logits)
        joint_entropy_loss = coupling_joint_entropy_loss(self.coupling_logits)
        smoothness_loss = coupling_eeg_neighbor_smoothness_loss(
            self.coupling_logits,
            self.eeg_source_quantizer.get_codebook_weight(),
            n_neighbors=self.coupling_smoothness_neighbors,
        )
        pair_likelihood_loss = coupling_pair_likelihood_loss(
            self.coupling_logits,
            eeg_source_logits,
            fnirs_source_logits,
            temperature=(
                self.source_balance_temperature
                if self.coupling_pair_temperature is None else self.coupling_pair_temperature
            ),
            gradient_target=self.coupling_pair_gradient_target,
            residualize_fnirs_marginal=self.coupling_residualize_fnirs_marginal,
            fixed_eeg_marginal=(
                self.coupling_fixed_eeg_marginal
                if self.coupling_fixed_eeg_marginal.numel() else None
            ),
            fixed_fnirs_marginal=(
                self.coupling_fixed_fnirs_marginal
                if self.coupling_fixed_fnirs_marginal.numel() else None
            ),
        )
        lag_evidence_loss = coupling_lag_evidence_loss(
            self.coupling_logits,
            eeg_source_logits,
            fnirs_source_logits,
            temperature=(
                self.source_balance_temperature
                if self.coupling_pair_temperature is None else self.coupling_pair_temperature
            ),
            detach_tokens=self.coupling_pair_detach_tokens,
            evidence_temperature=self.coupling_lag_evidence_temperature,
            residualize_fnirs_marginal=self.coupling_residualize_fnirs_marginal,
        )
        if self.coupling_fixed_eeg_marginal.numel() and self.coupling_fixed_fnirs_marginal.numel():
            effective_smoothness_loss = coupling_effective_neighbor_smoothness_loss(
                self.coupling_logits,
                self.eeg_source_quantizer.get_codebook_weight(),
                self.coupling_fixed_eeg_marginal,
                self.coupling_fixed_fnirs_marginal,
                n_neighbors=self.coupling_smoothness_neighbors,
            )
            interaction_lag_sparsity_loss = coupling_interaction_lag_sparsity_loss(
                self.coupling_logits,
                self.coupling_fixed_eeg_marginal,
            )
        else:
            effective_smoothness_loss = self.coupling_logits.new_zeros(())
            interaction_lag_sparsity_loss = self.coupling_logits.new_zeros(())
        if self.coupling_local_residual_enabled:
            local_residual_loss, local_components = local_residual_coupling_loss(
                eeg_source_logits,
                fnirs_source_logits,
                n_lags=self.n_coupling_lags,
                temperature=(
                    self.source_balance_temperature
                    if self.coupling_pair_temperature is None else self.coupling_pair_temperature
                ),
                gradient_target=self.coupling_pair_gradient_target,
                alpha=self.coupling_local_residual_alpha,
                pair_weight=self.coupling_local_residual_pair_weight,
                effective_smoothness_weight=self.coupling_effective_smoothness_weight,
                interaction_lag_sparsity_weight=self.coupling_interaction_lag_sparsity_weight,
                eeg_codebook_weight=self.eeg_source_quantizer.get_codebook_weight(),
                n_neighbors=self.coupling_smoothness_neighbors,
            )
        else:
            local_residual_loss = self.coupling_logits.new_zeros(())
            local_components = {
                'pair_likelihood': self.coupling_logits.new_zeros(()),
                'effective_smoothness': self.coupling_logits.new_zeros(()),
                'interaction_lag_sparsity': self.coupling_logits.new_zeros(()),
            }
        if self.coupling_context_residual_enabled:
            context_probs = self._context_coupling_probs(eeg_source_logits, fnirs_source_logits)
            context_residual_loss, context_components = context_residual_coupling_loss(
                self.context_coupling_eeg_factors,
                self.context_coupling_fnirs_factors,
                eeg_source_logits,
                fnirs_source_logits,
                context_probs,
                temperature=(
                    self.source_balance_temperature
                    if self.coupling_pair_temperature is None else self.coupling_pair_temperature
                ),
                gradient_target=self.coupling_context_gradient_target,
                pair_weight=self.coupling_context_pair_weight,
                entropy_weight=self.coupling_context_entropy_weight,
                balance_weight=self.coupling_context_balance_weight,
                residual_l1_weight=self.coupling_context_residual_l1_weight,
            )
            context_entropy = context_components['entropy']
            context_max_prob = context_probs.max(dim=-1).values.mean()
        else:
            context_probs = None
            context_residual_loss = self.coupling_logits.new_zeros(())
            context_components = {
                'pair_likelihood': self.coupling_logits.new_zeros(()),
                'entropy_loss': self.coupling_logits.new_zeros(()),
                'entropy': self.coupling_logits.new_zeros(()),
                'balance': self.coupling_logits.new_zeros(()),
                'residual_l1': self.coupling_logits.new_zeros(()),
            }
            context_entropy = self.coupling_logits.new_zeros(())
            context_max_prob = self.coupling_logits.new_zeros(())
        source_coupling_loss = (
            self.coupling_lag_focus_weight * lag_focus_loss +
            self.coupling_joint_entropy_weight * joint_entropy_loss +
            self.coupling_smoothness_weight * smoothness_loss +
            self.coupling_pair_likelihood_weight * pair_likelihood_loss +
            self.coupling_lag_evidence_weight * lag_evidence_loss +
            self.coupling_effective_smoothness_weight * effective_smoothness_loss +
            self.coupling_interaction_lag_sparsity_weight * interaction_lag_sparsity_loss +
            local_residual_loss +
            context_residual_loss
        )

        coupling_lag_count = eeg_source_logits.new_tensor(float(self.n_coupling_lags))

        return {
            'source_coupling_loss': source_coupling_loss,
            'source_coupling_lag_focus_loss': lag_focus_loss,
            'source_coupling_joint_entropy_loss': joint_entropy_loss,
            'source_coupling_smoothness_loss': smoothness_loss,
            'source_coupling_pair_likelihood_loss': pair_likelihood_loss,
            'source_coupling_lag_evidence_loss': lag_evidence_loss,
            'source_coupling_effective_smoothness_loss': effective_smoothness_loss,
            'source_coupling_interaction_lag_sparsity_loss': interaction_lag_sparsity_loss,
            'source_coupling_local_residual_loss': local_residual_loss,
            'source_coupling_local_pair_likelihood_loss': local_components['pair_likelihood'],
            'source_coupling_local_effective_smoothness_loss': local_components['effective_smoothness'],
            'source_coupling_local_interaction_lag_sparsity_loss': local_components['interaction_lag_sparsity'],
            'source_coupling_context_residual_loss': context_residual_loss,
            'source_coupling_context_pair_likelihood_loss': context_components['pair_likelihood'],
            'source_coupling_context_entropy_loss': context_components.get(
                'entropy_loss',
                context_components['entropy'],
            ),
            'source_coupling_context_balance_loss': context_components['balance'],
            'source_coupling_context_residual_l1_loss': context_components['residual_l1'],
            'source_coupling_context_entropy': context_entropy,
            'source_coupling_context_max_prob': context_max_prob,
            'source_coupling_context_probs': (
                context_probs if context_probs is not None
                else self.coupling_logits.new_ones((eeg_source_logits.shape[0], 1))
            ),
            'coupling_lag_count': coupling_lag_count,
        }

    def decode_from_components(
        self,
        eeg_source_q: torch.Tensor,
        eeg_observation_q: torch.Tensor,
        fnirs_source_q: torch.Tensor,
        fnirs_observation_q: torch.Tensor,
        decode_source: bool = True,
        decode_observation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        decode_source_branch = decode_source and self.source_branch_enabled
        decode_eeg_observation_branch = decode_observation and self.eeg_observation_branch_enabled
        decode_fnirs_observation_branch = decode_observation and self.fnirs_observation_branch_enabled

        if decode_source_branch:
            eeg_source_reconstructed = self._decode_branch_reconstruction(
                eeg_source_q,
                self.eeg_source_decode_input_proj,
                self.eeg_source_decoder,
                self.eeg_source_amplitude_head,
                self.eeg_source_phase_head,
                self.eeg_channels,
                self.eeg_fft_size,
                self.eeg_patch_size,
            )
            fnirs_source_reconstructed = self._decode_branch_reconstruction(
                fnirs_source_q,
                self.fnirs_source_decode_input_proj,
                self.fnirs_source_decoder,
                self.fnirs_source_amplitude_head,
                self.fnirs_source_phase_head,
                self.fnirs_channels,
                self.fnirs_fft_size,
                self.fnirs_patch_size,
            )
        else:
            eeg_source_reconstructed = self._zeros_like_reconstruction(
                eeg_source_q,
                self.eeg_channels,
                self.eeg_seq_length,
            )
            fnirs_source_reconstructed = self._zeros_like_reconstruction(
                fnirs_source_q,
                self.fnirs_channels,
                self.fnirs_seq_length,
            )

        if decode_eeg_observation_branch:
            eeg_observation_reconstructed = self._decode_branch_reconstruction(
                eeg_observation_q,
                self.eeg_observation_decode_input_proj,
                self.eeg_observation_decoder,
                self.eeg_observation_amplitude_head,
                self.eeg_observation_phase_head,
                self.eeg_channels,
                self.eeg_fft_size,
                self.eeg_patch_size,
            )
        else:
            eeg_observation_reconstructed = self._zeros_like_reconstruction(
                eeg_observation_q,
                self.eeg_channels,
                self.eeg_seq_length,
            )

        if decode_fnirs_observation_branch:
            fnirs_observation_reconstructed = self._decode_branch_reconstruction(
                fnirs_observation_q,
                self.fnirs_observation_decode_input_proj,
                self.fnirs_observation_decoder,
                self.fnirs_observation_amplitude_head,
                self.fnirs_observation_phase_head,
                self.fnirs_channels,
                self.fnirs_fft_size,
                self.fnirs_patch_size,
            )
        else:
            fnirs_observation_reconstructed = self._zeros_like_reconstruction(
                fnirs_observation_q,
                self.fnirs_channels,
                self.fnirs_seq_length,
            )

        return {
            'eeg_source_reconstructed': eeg_source_reconstructed,
            'fnirs_source_reconstructed': fnirs_source_reconstructed,
            'eeg_observation_reconstructed': eeg_observation_reconstructed,
            'fnirs_observation_reconstructed': fnirs_observation_reconstructed,
            'eeg_reconstructed': eeg_source_reconstructed + eeg_observation_reconstructed,
            'fnirs_reconstructed': fnirs_source_reconstructed + fnirs_observation_reconstructed,
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
            decode_source=use_source,
            decode_observation=use_observation,
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        coupling_logits = state_dict.get('coupling_logits') if isinstance(state_dict, dict) else None
        if torch.is_tensor(coupling_logits) and tuple(coupling_logits.shape) != tuple(self.coupling_logits.shape):
            if coupling_logits.ndim != 3:
                raise RuntimeError(
                    f'Invalid coupling_logits shape in checkpoint: expected 3D tensor, got {tuple(coupling_logits.shape)}'
                )
            self.n_coupling_lags = int(coupling_logits.shape[0])
            self.coupling_lags = list(range(self.n_coupling_lags))
            self.coupling_logits = nn.Parameter(
                torch.zeros(
                    tuple(coupling_logits.shape),
                    dtype=self.coupling_logits.dtype,
                    device=self.coupling_logits.device,
                )
            )

        if strict and isinstance(state_dict, dict):
            compatibility_prefixes = (
                'eeg_to_fnirs_source_predictor.',
                'fnirs_to_eeg_source_predictor.',
                'eeg_shared_state_proj.',
                'fnirs_shared_state_proj.',
                'cross_modal_exchange.',
                'cross_modal_fusion.',
                'cross_modal_eeg_projection.',
                'cross_modal_fnirs_projection.',
                'shared_source_quantizer.',
                'eeg_source_mask_token',
                'fnirs_source_mask_token',
            )
            own_keys = set(self.state_dict().keys())
            state_keys = set(state_dict.keys())
            has_compatibility_gap = any(
                key not in state_keys and key.startswith(compatibility_prefixes)
                for key in own_keys
            )
            if has_compatibility_gap:
                result = super().load_state_dict(state_dict, strict=False, assign=assign)
                unexpected = list(getattr(result, 'unexpected_keys', []))
                missing = [
                    key for key in getattr(result, 'missing_keys', [])
                    if not key.startswith(compatibility_prefixes)
                ]
                if unexpected or missing:
                    details = []
                    if missing:
                        details.append(f'Missing key(s): {missing}')
                    if unexpected:
                        details.append(f'Unexpected key(s): {unexpected}')
                    raise RuntimeError('Error(s) in loading state_dict for SourceObservationLaBraMVQNSP:\n\t' + '\n\t'.join(details))
                return result

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def initialize_shared_source_codebook(self, seed: int = 0) -> None:
        """Initialize the joint source vocabulary from both pretrained source codebooks."""
        if self.shared_source_quantizer is None or self.shared_source_quantizer.initted.item():
            return
        samples = torch.cat([
            l2norm(self.eeg_source_quantizer.get_codebook_weight().detach()),
            l2norm(self.fnirs_source_quantizer.get_codebook_weight().detach()),
        ], dim=0)
        devices = [samples.device] if samples.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            means, counts = kmeans(
                samples,
                self.shared_source_quantizer.num_tokens,
                num_iters=20,
                use_cosine_sim=True,
            )
        self.shared_source_quantizer.weight.copy_(means)
        self.shared_source_quantizer.cluster_size.copy_(counts.to(means.dtype).clamp_min(1.0))
        self.shared_source_quantizer.initted.fill_(1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Use encode_modalities(eeg, fnirs) for the source/observation tokenizer')

    def encode_modalities(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        eeg_encoded = self._encode_modality(eeg, self.eeg_patch_size, self.eeg_patch_embed, self.eeg_encoder)
        fnirs_encoded = self._encode_modality(fnirs, self.fnirs_patch_size, self.fnirs_patch_embed, self.fnirs_encoder)
        fnirs_source_input = fnirs_encoded
        result: Dict[str, torch.Tensor] = {}
        if self.cross_modal_exchange is not None:
            exchange_update, exchange_lag_weights = self.cross_modal_exchange(
                eeg_encoded,
                fnirs_encoded,
                detach_context=self.cross_modal_exchange_detach_context,
            )
            fnirs_source_input = fnirs_encoded + exchange_update
            result['cross_modal_exchange_update'] = exchange_update
            result['cross_modal_exchange_lag_weights'] = exchange_lag_weights
        eeg_source_pre = self.eeg_source_proj(eeg_encoded)
        fnirs_source_pre = self.fnirs_source_proj(fnirs_source_input)
        eeg_source = eeg_source_pre
        fnirs_source = fnirs_source_pre
        if self.cross_modal_fusion is not None:
            fused = self.cross_modal_fusion(eeg_source_pre, fnirs_source_pre)
            eeg_source = fused['eeg_source']
            fnirs_source = fused['fnirs_source']
            result['cross_modal_fusion_fnirs_attention'] = fused['fnirs_attention']
            result['cross_modal_fusion_eeg_attention'] = fused['eeg_attention']
        return {
            **result,
            'eeg_source_pre': eeg_source_pre,
            'fnirs_source_pre': fnirs_source_pre,
            'eeg_source': eeg_source,
            'eeg_observation': self.eeg_observation_proj(eeg_encoded),
            'fnirs_source': fnirs_source,
            'fnirs_observation': self.fnirs_observation_proj(fnirs_encoded),
        }

    def quantize(self, z: torch.Tensor, modality: str = 'eeg_source'):
        if modality in {'eeg_source', 'fnirs_source'} and self.shared_source_quantizer is not None:
            return self.shared_source_quantizer(z)
        quantizers = {
            'eeg_source': self.eeg_source_quantizer,
            'fnirs_source': self.fnirs_source_quantizer,
            'eeg_observation': self.eeg_observation_quantizer,
            'fnirs_observation': self.fnirs_observation_quantizer,
        }
        return quantizers[modality](z)

    def _quantize_source_pair(
        self,
        eeg_source: torch.Tensor,
        fnirs_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.shared_source_quantizer is None:
            eeg_q, eeg_indices, eeg_info = self.eeg_source_quantizer(eeg_source)
            fnirs_q, fnirs_indices, fnirs_info = self.fnirs_source_quantizer(fnirs_source)
            return eeg_q, eeg_indices, eeg_info, fnirs_q, fnirs_indices, fnirs_info
        tokens = eeg_source.shape[1]
        joint_q, joint_indices, joint_info = self.shared_source_quantizer(
            torch.cat([eeg_source, fnirs_source], dim=1)
        )
        eeg_info = {key: value * 0.5 if key == 'vq_loss' else value for key, value in joint_info.items()}
        fnirs_info = {key: value * 0.5 if key == 'vq_loss' else value for key, value in joint_info.items()}
        def branch_perplexity(indices: torch.Tensor) -> torch.Tensor:
            counts = F.one_hot(indices.reshape(-1), self.source_codebook_size).float().mean(dim=0)
            return torch.exp(-(counts * (counts + 1e-10).log()).sum())
        eeg_info['perplexity'] = branch_perplexity(joint_indices[:, :tokens])
        fnirs_info['perplexity'] = branch_perplexity(joint_indices[:, tokens:])
        return (
            joint_q[:, :tokens],
            joint_indices[:, :tokens],
            eeg_info,
            joint_q[:, tokens:],
            joint_indices[:, tokens:],
            fnirs_info,
        )

    def _source_codebook_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_source_quantizer is not None:
            weight = self.shared_source_quantizer.get_codebook_weight()
            return weight, weight
        return (
            self.eeg_source_quantizer.get_codebook_weight(),
            self.fnirs_source_quantizer.get_codebook_weight(),
        )

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Use modality-specific decode paths for the source/observation tokenizer')

    def _resolve_explicit_targets(
        self,
        targets: Optional[Dict[str, torch.Tensor]],
        *,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if targets is None:
            return None
        required = {
            'eeg_source': eeg,
            'eeg_observation': eeg,
            'fnirs_source': fnirs,
            'fnirs_observation': fnirs,
        }
        resolved: Dict[str, torch.Tensor] = {}
        for name, reference in required.items():
            if name not in targets:
                raise KeyError(f"Explicit source/observation targets must include {name!r}")
            value = targets[name]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.to(device=reference.device, dtype=reference.dtype, non_blocking=True)
            if tuple(value.shape) != tuple(reference.shape):
                raise ValueError(
                    f"Target {name!r} must have shape {tuple(reference.shape)}, got {tuple(value.shape)}"
                )
            resolved[name] = value
        return resolved

    def _alignment_masks(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens = reference.shape[:2]
        if self.training:
            fnirs_mask = torch.rand(batch, tokens, device=reference.device) < self.cross_modal_token_mask_ratio
            eeg_mask = torch.rand(batch, tokens, device=reference.device) < self.cross_modal_token_mask_ratio
            dropout_draw = torch.rand(batch, device=reference.device)
        else:
            count = max(int(round(tokens * self.cross_modal_token_mask_ratio)), 1)
            fnirs_mask = torch.zeros(batch, tokens, dtype=torch.bool, device=reference.device)
            eeg_mask = torch.zeros_like(fnirs_mask)
            fnirs_mask[:, :count] = True
            eeg_mask[:, tokens - count:] = True
            dropout_draw = torch.arange(batch, device=reference.device, dtype=torch.float32) / max(batch, 1)
        probability = self.cross_modal_modality_dropout_probability
        if self.cross_modal_fusion_mode == 'causal_cross_attention':
            fnirs_mask[dropout_draw < probability] = True
            eeg_mask.zero_()
        else:
            fnirs_mask[dropout_draw < probability * 0.5] = True
            eeg_mask[(dropout_draw >= probability * 0.5) & (dropout_draw < probability)] = True
        return eeg_mask, fnirs_mask

    def _compute_cross_modal_alignment_losses(
        self,
        eeg_source_pre: torch.Tensor,
        fnirs_source_pre: torch.Tensor,
        eeg_source_weight: torch.Tensor,
        fnirs_source_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = eeg_source_pre.new_zeros(())
        temporal = zero
        if self.cross_modal_temporal_nce_weight > 0.0:
            temporal = lag_aware_temporal_nce(
                eeg_source_pre,
                fnirs_source_pre,
                self.cross_modal_eeg_projection,
                self.cross_modal_fnirs_projection,
                self.cross_modal_positive_lag_weights,
                self.cross_modal_alignment_temperature,
                bidirectional=self.cross_modal_fusion_mode == 'bidirectional_cross_attention',
            )
        latent = zero
        soft_code = zero
        masked_fraction = zero
        shuffled_latent = zero
        pairing_gain = zero
        if self.cross_modal_fusion is not None and (
            self.cross_modal_masked_latent_weight > 0.0 or self.cross_modal_soft_code_weight > 0.0
        ):
            eeg_mask, fnirs_mask = self._alignment_masks(eeg_source_pre)
            masked_eeg = torch.where(eeg_mask.unsqueeze(-1), self.eeg_source_mask_token, eeg_source_pre)
            masked_fnirs = torch.where(fnirs_mask.unsqueeze(-1), self.fnirs_source_mask_token, fnirs_source_pre)
            predicted = self.cross_modal_fusion(masked_eeg, masked_fnirs)
            fnirs_latent, fnirs_code = masked_alignment_losses(
                predicted['fnirs_source'],
                fnirs_source_pre,
                fnirs_mask,
                self._assignment_logits(predicted['fnirs_source'], fnirs_source_weight),
                self._assignment_logits(fnirs_source_pre, fnirs_source_weight),
                self.cross_modal_alignment_temperature,
            )
            latent = fnirs_latent
            soft_code = fnirs_code
            masked_fraction = fnirs_mask.float().mean()
            if not self.training and eeg_source_pre.shape[1] > 1:
                shuffled_eeg = torch.roll(masked_eeg, shifts=1, dims=1)
                shuffled_prediction = self.cross_modal_fusion(shuffled_eeg, masked_fnirs)['fnirs_source']
                shuffled_values = 1.0 - F.cosine_similarity(
                    shuffled_prediction[fnirs_mask],
                    fnirs_source_pre.detach()[fnirs_mask],
                    dim=-1,
                )
                shuffled_latent = shuffled_values.mean() if shuffled_values.numel() else zero
                pairing_gain = shuffled_latent - fnirs_latent
            if self.cross_modal_fusion_mode == 'bidirectional_cross_attention':
                eeg_latent, eeg_code = masked_alignment_losses(
                    predicted['eeg_source'],
                    eeg_source_pre,
                    eeg_mask,
                    self._assignment_logits(predicted['eeg_source'], eeg_source_weight),
                    self._assignment_logits(eeg_source_pre, eeg_source_weight),
                    self.cross_modal_alignment_temperature,
                )
                latent = 0.5 * (latent + eeg_latent)
                soft_code = 0.5 * (soft_code + eeg_code)
                masked_fraction = 0.5 * (masked_fraction + eeg_mask.float().mean())
        return {
            'cross_modal_temporal_nce_loss': temporal,
            'cross_modal_masked_latent_loss': latent,
            'cross_modal_soft_code_distillation_loss': soft_code,
            'cross_modal_masked_fraction': masked_fraction,
            'cross_modal_masked_shuffled_latent_loss': shuffled_latent,
            'cross_modal_masked_pairing_gain': pairing_gain,
        }

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if eeg.dim() != 3 or fnirs.dim() != 3:
            raise ValueError('Expected eeg and fnirs tensors with shape [B, C, T]')
        if eeg.shape[1] != self.eeg_channels:
            raise ValueError(f'Expected EEG channels {self.eeg_channels}, got {eeg.shape[1]}')
        if fnirs.shape[1] != self.fnirs_channels:
            raise ValueError(f'Expected fNIRS channels {self.fnirs_channels}, got {fnirs.shape[1]}')
        if eeg.shape[-1] != self.eeg_seq_length:
            raise ValueError(f'Expected EEG length {self.eeg_seq_length}, got {eeg.shape[-1]}')
        if fnirs.shape[-1] != self.fnirs_seq_length:
            raise ValueError(f'Expected fNIRS length {self.fnirs_seq_length}, got {fnirs.shape[-1]}')
        explicit_targets = self._resolve_explicit_targets(targets, eeg=eeg, fnirs=fnirs)

        eeg_patches = self._split_to_patches(eeg, self.eeg_patch_size)
        fnirs_patches = self._split_to_patches(fnirs, self.fnirs_patch_size)
        eeg_target_amp = self._compute_fft_amplitude_targets(
            eeg_patches,
            self.eeg_fft_loss_window,
        )
        fnirs_target_amp = self._compute_fft_amplitude_targets(
            fnirs_patches,
            self.fnirs_fft_loss_window,
        )

        latents = self.encode_modalities(eeg, fnirs)
        eeg_source = latents['eeg_source']
        eeg_observation = latents['eeg_observation']
        fnirs_source = latents['fnirs_source']
        fnirs_observation = latents['fnirs_observation']
        eeg_source_pre = latents['eeg_source_pre']
        fnirs_source_pre = latents['fnirs_source_pre']

        (
            eeg_source_q,
            eeg_source_indices,
            eeg_source_info,
            fnirs_source_q,
            fnirs_source_indices,
            fnirs_source_info,
        ) = self._quantize_source_pair(eeg_source, fnirs_source)
        eeg_observation_q, eeg_observation_indices, eeg_observation_info = self.eeg_observation_quantizer(eeg_observation)
        fnirs_observation_q, fnirs_observation_indices, fnirs_observation_info = self.fnirs_observation_quantizer(fnirs_observation)

        eeg_source_q = self._branch_dropout(eeg_source_q, self.source_branch_dropout)
        fnirs_source_q = self._branch_dropout(fnirs_source_q, self.source_branch_dropout)
        eeg_observation_q = self._branch_dropout(eeg_observation_q, self.eeg_observation_branch_dropout)
        fnirs_observation_q = self._branch_dropout(fnirs_observation_q, self.fnirs_observation_branch_dropout)

        eeg_source_weight, fnirs_source_weight = self._source_codebook_weights()
        eeg_source_logits = self._assignment_logits(eeg_source, eeg_source_weight)
        fnirs_source_logits = self._assignment_logits(fnirs_source, fnirs_source_weight)
        eeg_observation_logits = self._assignment_logits(
            eeg_observation,
            self.eeg_observation_quantizer.get_codebook_weight(),
        )
        fnirs_observation_logits = self._assignment_logits(
            fnirs_observation,
            self.fnirs_observation_quantizer.get_codebook_weight(),
        )
        cross_modal_alignment = self._compute_cross_modal_alignment_losses(
            eeg_source_pre,
            fnirs_source_pre,
            eeg_source_weight,
            fnirs_source_weight,
        )
        cross_modal_temporal_nce_loss = cross_modal_alignment['cross_modal_temporal_nce_loss']
        cross_modal_masked_latent_loss = cross_modal_alignment['cross_modal_masked_latent_loss']
        cross_modal_soft_code_distillation_loss = cross_modal_alignment[
            'cross_modal_soft_code_distillation_loss'
        ]
        cross_modal_masked_fraction = cross_modal_alignment['cross_modal_masked_fraction']
        cross_modal_masked_shuffled_latent_loss = cross_modal_alignment[
            'cross_modal_masked_shuffled_latent_loss'
        ]
        cross_modal_masked_pairing_gain = cross_modal_alignment['cross_modal_masked_pairing_gain']

        source_alignment = self._compute_source_alignment_state(eeg_source_logits, fnirs_source_logits)
        source_coupling_loss = source_alignment['source_coupling_loss']
        source_coupling_lag_focus_loss = source_alignment['source_coupling_lag_focus_loss']
        source_coupling_joint_entropy_loss = source_alignment['source_coupling_joint_entropy_loss']
        source_coupling_smoothness_loss = source_alignment['source_coupling_smoothness_loss']
        source_coupling_pair_likelihood_loss = source_alignment['source_coupling_pair_likelihood_loss']
        source_coupling_lag_evidence_loss = source_alignment['source_coupling_lag_evidence_loss']
        source_coupling_effective_smoothness_loss = source_alignment['source_coupling_effective_smoothness_loss']
        source_coupling_interaction_lag_sparsity_loss = source_alignment['source_coupling_interaction_lag_sparsity_loss']
        source_coupling_local_residual_loss = source_alignment['source_coupling_local_residual_loss']
        source_coupling_local_pair_likelihood_loss = source_alignment['source_coupling_local_pair_likelihood_loss']
        source_coupling_local_effective_smoothness_loss = source_alignment[
            'source_coupling_local_effective_smoothness_loss'
        ]
        source_coupling_local_interaction_lag_sparsity_loss = source_alignment[
            'source_coupling_local_interaction_lag_sparsity_loss'
        ]
        source_coupling_context_residual_loss = source_alignment['source_coupling_context_residual_loss']
        source_coupling_context_pair_likelihood_loss = source_alignment[
            'source_coupling_context_pair_likelihood_loss'
        ]
        source_coupling_context_entropy_loss = source_alignment['source_coupling_context_entropy_loss']
        source_coupling_context_balance_loss = source_alignment['source_coupling_context_balance_loss']
        source_coupling_context_residual_l1_loss = source_alignment['source_coupling_context_residual_l1_loss']
        source_coupling_context_entropy = source_alignment['source_coupling_context_entropy']
        source_coupling_context_max_prob = source_alignment['source_coupling_context_max_prob']
        source_coupling_context_probs = source_alignment['source_coupling_context_probs']
        coupling_lag_count = source_alignment['coupling_lag_count']

        source_balance_temperature = max(float(self.source_balance_temperature), 1e-3)
        observation_balance_temperature = max(float(self.observation_balance_temperature), 1e-3)
        eeg_source_balance_probs = straight_through_assignment_probs(eeg_source_logits, source_balance_temperature)
        fnirs_source_balance_probs = straight_through_assignment_probs(fnirs_source_logits, source_balance_temperature)
        eeg_observation_balance_probs = straight_through_assignment_probs(
            eeg_observation_logits,
            observation_balance_temperature,
        )
        fnirs_observation_balance_probs = straight_through_assignment_probs(
            fnirs_observation_logits,
            observation_balance_temperature,
        )

        zero_scalar = eeg_source_q.new_zeros(())
        cross_modal_exchange_update_norm = zero_scalar
        cross_modal_exchange_lag_entropy = zero_scalar
        cross_modal_exchange_residual_gate = zero_scalar
        exchange_update = latents.get('cross_modal_exchange_update')
        exchange_lag_weights = latents.get('cross_modal_exchange_lag_weights')
        if torch.is_tensor(exchange_update):
            cross_modal_exchange_update_norm = exchange_update.pow(2).mean().sqrt()
        if torch.is_tensor(exchange_lag_weights):
            lag_entropy = -(exchange_lag_weights.clamp_min(1.0e-12).log() * exchange_lag_weights).sum(dim=-1)
            lag_count = max(int(exchange_lag_weights.shape[-1]), 1)
            cross_modal_exchange_lag_entropy = lag_entropy.mean() / math.log(max(lag_count, 2))
        if self.cross_modal_exchange is not None:
            cross_modal_exchange_residual_gate = self.cross_modal_exchange.residual_gate.detach()
        cross_modal_fusion_lag_entropy = zero_scalar
        cross_modal_fusion_physiologic_lag_mass = zero_scalar
        cross_modal_fusion_reverse_lag_mass = zero_scalar
        fnirs_attention = latents.get('cross_modal_fusion_fnirs_attention')
        if torch.is_tensor(fnirs_attention):
            cross_modal_fusion_lag_entropy, cross_modal_fusion_physiologic_lag_mass = attention_lag_statistics(
                fnirs_attention,
                self.cross_modal_fusion_max_lag_tokens,
                'eeg_to_fnirs',
            )
        eeg_attention = latents.get('cross_modal_fusion_eeg_attention')
        if torch.is_tensor(eeg_attention) and eeg_attention.numel() > 0:
            _, cross_modal_fusion_reverse_lag_mass = attention_lag_statistics(
                eeg_attention,
                self.cross_modal_fusion_max_lag_tokens,
                'fnirs_to_eeg',
            )
        source_balance_terms = []
        if self.source_branch_enabled:
            source_balance_terms.extend([
                batch_usage_entropy_loss(eeg_source_balance_probs),
                batch_usage_entropy_loss(fnirs_source_balance_probs),
            ])
        source_balance_loss = sum(source_balance_terms) / len(source_balance_terms) if source_balance_terms else zero_scalar
        observation_balance_terms = []
        if self.eeg_observation_branch_enabled:
            observation_balance_terms.append(batch_usage_entropy_loss(eeg_observation_balance_probs))
        if self.fnirs_observation_branch_enabled:
            observation_balance_terms.append(batch_usage_entropy_loss(fnirs_observation_balance_probs))
        observation_balance_loss = (
            sum(observation_balance_terms) / len(observation_balance_terms)
            if observation_balance_terms else zero_scalar
        )
        codebook_balance_loss = 0.5 * (
            self.source_balance_scale * source_balance_loss +
            self.observation_balance_scale * observation_balance_loss
        )

        reconstructions = self.decode_from_components(
            eeg_source_q,
            eeg_observation_q,
            fnirs_source_q,
            fnirs_observation_q,
        )

        eeg_rec = reconstructions['eeg_reconstructed']
        eeg_rec_patches = self._split_to_patches(eeg_rec, self.eeg_patch_size)
        eeg_rec_amp = self._compute_fft_amplitude_targets(eeg_rec_patches, self.eeg_fft_loss_window)
        eeg_amp_loss = self.loss_fn(eeg_rec_amp, eeg_target_amp)
        eeg_time_loss = self.loss_fn(eeg_rec, eeg)
        eeg_rec_loss = (
            self.eeg_amplitude_weight * eeg_amp_loss +
            self.eeg_time_weight * eeg_time_loss
        )

        fnirs_rec = reconstructions['fnirs_reconstructed']
        fnirs_rec_patches = self._split_to_patches(fnirs_rec, self.fnirs_patch_size)
        fnirs_rec_amp = self._compute_fft_amplitude_targets(fnirs_rec_patches, self.fnirs_fft_loss_window)
        fnirs_amp_loss = self.loss_fn(fnirs_rec_amp, fnirs_target_amp)
        fnirs_time_loss = self.loss_fn(fnirs_rec, fnirs)
        fnirs_rec_loss = (
            self.fnirs_amplitude_weight * fnirs_amp_loss +
            self.fnirs_time_weight * fnirs_time_loss
        )

        if explicit_targets is None:
            shared_state = self._compute_shared_neural_state(eeg)
            fnirs_source_target = self._compute_fnirs_source_target(shared_state, fnirs)
            eeg_source_aux_target = self._compute_eeg_source_target(eeg)
            eeg_observation_target = self._compute_observation_target(eeg, eeg_source_aux_target)
            fnirs_observation_target = self._compute_observation_target(fnirs, fnirs_source_target)
        else:
            fnirs_source_target = explicit_targets['fnirs_source']
            eeg_source_aux_target = explicit_targets['eeg_source']
            eeg_observation_target = explicit_targets['eeg_observation']
            fnirs_observation_target = explicit_targets['fnirs_observation']
        explicit_targets_used = eeg.new_tensor(1.0 if explicit_targets is not None else 0.0)

        eeg_source_reconstructed = reconstructions['eeg_source_reconstructed']
        fnirs_source_reconstructed = reconstructions['fnirs_source_reconstructed']
        eeg_observation_reconstructed = reconstructions['eeg_observation_reconstructed']
        fnirs_observation_reconstructed = reconstructions['fnirs_observation_reconstructed']

        source_target_loss = (
            self.loss_fn(fnirs_source_reconstructed, fnirs_source_target)
            if self.source_branch_enabled else zero_scalar
        )
        source_target_random_baseline = self._compute_source_target_random_baseline(
            fnirs_source_reconstructed,
            fnirs_source_target,
        )
        source_target_corr_loss = (
            self._signal_correlation_loss(fnirs_source_reconstructed, fnirs_source_target)
            if self.source_branch_enabled else zero_scalar
        )
        source_target_corr_random_baseline = (
            self._randomized_signal_correlation_loss(fnirs_source_reconstructed, fnirs_source_target)
            if self.source_branch_enabled else zero_scalar
        )
        eeg_source_aux_loss = (
            self.loss_fn(eeg_source_reconstructed, eeg_source_aux_target)
            if self.source_branch_enabled else zero_scalar
        )
        eeg_source_aux_corr_loss = (
            self._signal_correlation_loss(eeg_source_reconstructed, eeg_source_aux_target)
            if self.source_branch_enabled else zero_scalar
        )
        eeg_source_aux_corr_random_baseline = (
            self._randomized_signal_correlation_loss(eeg_source_reconstructed, eeg_source_aux_target)
            if self.source_branch_enabled else zero_scalar
        )
        eeg_observation_loss = (
            self.loss_fn(eeg_observation_reconstructed, eeg_observation_target)
            if self.eeg_observation_branch_enabled else zero_scalar
        )
        fnirs_observation_loss = (
            self.loss_fn(fnirs_observation_reconstructed, fnirs_observation_target)
            if self.fnirs_observation_branch_enabled else zero_scalar
        )
        observation_loss = eeg_observation_loss + fnirs_observation_loss

        branch_orthogonality_loss = zero_scalar
        if self.eeg_observation_branch_enabled:
            branch_orthogonality_loss = branch_orthogonality_loss + orthogonality_loss(eeg_source, eeg_observation)
        if self.fnirs_observation_branch_enabled:
            branch_orthogonality_loss = branch_orthogonality_loss + orthogonality_loss(fnirs_source, fnirs_observation)

        interaction_aux_loss = zero_scalar
        if self.interaction_aux_weight > 0.0:
            if self.interaction_aux_direction in {'eeg_to_fnirs', 'bidirectional'}:
                target = fnirs_source.detach() if self.interaction_aux_stop_gradient else fnirs_source
                prediction = self.eeg_to_fnirs_source_predictor(eeg_source)
                interaction_aux_loss = interaction_aux_loss + self.loss_fn(
                    F.normalize(prediction, dim=-1),
                    F.normalize(target, dim=-1),
                )
            if self.interaction_aux_direction in {'fnirs_to_eeg', 'bidirectional'}:
                target = eeg_source.detach() if self.interaction_aux_stop_gradient else eeg_source
                prediction = self.fnirs_to_eeg_source_predictor(fnirs_source)
                interaction_aux_loss = interaction_aux_loss + self.loss_fn(
                    F.normalize(prediction, dim=-1),
                    F.normalize(target, dim=-1),
                )
            if self.interaction_aux_direction == 'bidirectional':
                interaction_aux_loss = 0.5 * interaction_aux_loss

        shared_state_bottleneck_loss = zero_scalar
        if self.shared_state_bottleneck_weight > 0.0:
            eeg_shared = F.normalize(self.eeg_shared_state_proj(eeg_source), dim=-1)
            fnirs_shared = F.normalize(self.fnirs_shared_state_proj(fnirs_source), dim=-1)
            target_shared = fnirs_shared.detach() if self.shared_state_bottleneck_stop_gradient else fnirs_shared
            shared_state_bottleneck_loss = self.loss_fn(eeg_shared, target_shared)

        vq_source_loss = (
            eeg_source_info['vq_loss'] + fnirs_source_info['vq_loss']
            if self.source_branch_enabled else zero_scalar
        )
        vq_observation_loss = zero_scalar
        if self.eeg_observation_branch_enabled:
            vq_observation_loss = vq_observation_loss + eeg_observation_info['vq_loss']
        if self.fnirs_observation_branch_enabled:
            vq_observation_loss = vq_observation_loss + fnirs_observation_info['vq_loss']
        vq_loss = vq_source_loss + vq_observation_loss

        source_coupling_weighted_loss = (
            self.coupling_weight * self.alignment_scale
        ) * source_coupling_loss
        cross_modal_alignment_unscaled_loss = (
            self.cross_modal_temporal_nce_weight * cross_modal_temporal_nce_loss +
            self.cross_modal_masked_latent_weight * cross_modal_masked_latent_loss +
            self.cross_modal_soft_code_weight * cross_modal_soft_code_distillation_loss
        )
        cross_modal_alignment_weighted_loss = (
            self.alignment_scale * self.cross_modal_gradient_scale
        ) * cross_modal_alignment_unscaled_loss

        total_loss = (
            eeg_rec_loss +
            fnirs_rec_loss +
            vq_loss +
            (self.source_target_weight * self.source_target_scale) * source_target_loss +
            (self.source_target_weight * self.eeg_source_aux_weight * self.source_target_scale) * eeg_source_aux_loss +
            (self.source_target_correlation_weight * self.source_target_scale) * source_target_corr_loss +
            (self.eeg_source_aux_correlation_weight * self.source_target_scale) * eeg_source_aux_corr_loss +
            (self.observation_target_weight * self.observation_target_scale) * observation_loss +
            source_coupling_weighted_loss +
            cross_modal_alignment_weighted_loss +
            self.interaction_aux_weight * interaction_aux_loss +
            self.shared_state_bottleneck_weight * shared_state_bottleneck_loss +
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
            'eeg_time_loss': eeg_time_loss,
            'fnirs_amp_loss': fnirs_amp_loss,
            'fnirs_time_loss': fnirs_time_loss,
            'vq_loss': vq_loss,
            'vq_source_loss': vq_source_loss,
            'vq_observation_loss': vq_observation_loss,
            'source_target_loss': source_target_loss,
            'source_target_random_baseline': source_target_random_baseline,
            'source_target_corr_loss': source_target_corr_loss,
            'source_target_corr_random_baseline': source_target_corr_random_baseline,
            'eeg_source_aux_loss': eeg_source_aux_loss,
            'eeg_source_aux_corr_loss': eeg_source_aux_corr_loss,
            'eeg_source_aux_corr_random_baseline': eeg_source_aux_corr_random_baseline,
            'eeg_observation_loss': eeg_observation_loss,
            'fnirs_observation_loss': fnirs_observation_loss,
            'observation_loss': observation_loss,
            'source_coupling_loss': source_coupling_loss,
            'source_coupling_weighted_loss': source_coupling_weighted_loss,
            'source_coupling_lag_focus_loss': source_coupling_lag_focus_loss,
            'source_coupling_joint_entropy_loss': source_coupling_joint_entropy_loss,
            'source_coupling_smoothness_loss': source_coupling_smoothness_loss,
            'source_coupling_pair_likelihood_loss': source_coupling_pair_likelihood_loss,
            'source_coupling_lag_evidence_loss': source_coupling_lag_evidence_loss,
            'source_coupling_effective_smoothness_loss': source_coupling_effective_smoothness_loss,
            'source_coupling_interaction_lag_sparsity_loss': source_coupling_interaction_lag_sparsity_loss,
            'source_coupling_local_residual_loss': source_coupling_local_residual_loss,
            'source_coupling_local_pair_likelihood_loss': source_coupling_local_pair_likelihood_loss,
            'source_coupling_local_effective_smoothness_loss': source_coupling_local_effective_smoothness_loss,
            'source_coupling_local_interaction_lag_sparsity_loss': source_coupling_local_interaction_lag_sparsity_loss,
            'source_coupling_context_residual_loss': source_coupling_context_residual_loss,
            'source_coupling_context_pair_likelihood_loss': source_coupling_context_pair_likelihood_loss,
            'source_coupling_context_entropy_loss': source_coupling_context_entropy_loss,
            'source_coupling_context_balance_loss': source_coupling_context_balance_loss,
            'source_coupling_context_residual_l1_loss': source_coupling_context_residual_l1_loss,
            'source_coupling_context_entropy': source_coupling_context_entropy,
            'source_coupling_context_max_prob': source_coupling_context_max_prob,
            'source_coupling_context_probs': source_coupling_context_probs,
            'coupling_lag_count': coupling_lag_count,
            'interaction_aux_loss': interaction_aux_loss,
            'shared_state_bottleneck_loss': shared_state_bottleneck_loss,
            'cross_modal_exchange_update_norm': cross_modal_exchange_update_norm,
            'cross_modal_exchange_lag_entropy': cross_modal_exchange_lag_entropy,
            'cross_modal_exchange_residual_gate': cross_modal_exchange_residual_gate,
            'cross_modal_temporal_nce_loss': cross_modal_temporal_nce_loss,
            'cross_modal_masked_latent_loss': cross_modal_masked_latent_loss,
            'cross_modal_soft_code_distillation_loss': cross_modal_soft_code_distillation_loss,
            'cross_modal_alignment_unscaled_loss': cross_modal_alignment_unscaled_loss,
            'cross_modal_alignment_weighted_loss': cross_modal_alignment_weighted_loss,
            'cross_modal_gradient_scale': torch.tensor(
                self.cross_modal_gradient_scale, device=eeg.device, dtype=torch.float32
            ),
            'cross_modal_masked_fraction': cross_modal_masked_fraction,
            'cross_modal_masked_shuffled_latent_loss': cross_modal_masked_shuffled_latent_loss,
            'cross_modal_masked_pairing_gain': cross_modal_masked_pairing_gain,
            'cross_modal_fusion_lag_entropy': cross_modal_fusion_lag_entropy,
            'cross_modal_fusion_physiologic_lag_mass': cross_modal_fusion_physiologic_lag_mass,
            'cross_modal_fusion_reverse_lag_mass': cross_modal_fusion_reverse_lag_mass,
            'codebook_balance_loss': codebook_balance_loss,
            'source_balance_loss': source_balance_loss,
            'observation_balance_loss': observation_balance_loss,
            'orthogonality_loss': branch_orthogonality_loss,
            'alignment_scale': torch.tensor(self.alignment_scale, device=eeg.device, dtype=torch.float32),
            'source_target_scale': torch.tensor(self.source_target_scale, device=eeg.device, dtype=torch.float32),
            'observation_target_scale': torch.tensor(self.observation_target_scale, device=eeg.device, dtype=torch.float32),
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
            'eeg_source_reconstructed': eeg_source_reconstructed,
            'fnirs_source_reconstructed': fnirs_source_reconstructed,
            'eeg_observation_reconstructed': eeg_observation_reconstructed,
            'fnirs_observation_reconstructed': fnirs_observation_reconstructed,
            'eeg_source_only_reconstructed': eeg_source_reconstructed,
            'fnirs_source_only_reconstructed': fnirs_source_reconstructed,
            'eeg_observation_only_reconstructed': eeg_observation_reconstructed,
            'fnirs_observation_only_reconstructed': fnirs_observation_reconstructed,
            'eeg_source_indices': eeg_source_indices,
            'fnirs_source_indices': fnirs_source_indices,
            'eeg_observation_indices': eeg_observation_indices,
            'fnirs_observation_indices': fnirs_observation_indices,
            'eeg_source_z': eeg_source,
            'fnirs_source_z': fnirs_source,
            'eeg_source_pre_fusion_z': eeg_source_pre,
            'fnirs_source_pre_fusion_z': fnirs_source_pre,
            'eeg_observation_z': eeg_observation,
            'fnirs_observation_z': fnirs_observation,
            'eeg_source_z_q': eeg_source_q,
            'fnirs_source_z_q': fnirs_source_q,
            'eeg_observation_z_q': eeg_observation_q,
            'fnirs_observation_z_q': fnirs_observation_q,
            'fnirs_source_target': fnirs_source_target,
            'eeg_source_target': eeg_source_aux_target,
            'eeg_source_aux_target': eeg_source_aux_target,
            'eeg_observation_target': eeg_observation_target,
            'fnirs_observation_target': fnirs_observation_target,
            'explicit_targets_used': explicit_targets_used,
            'croce_targets_used': explicit_targets_used,
            'spatial_source_prior_enabled': torch.tensor(
                1.0 if self.spatial_source_prior_enabled else 0.0,
                device=eeg.device,
                dtype=torch.float32,
            ),
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
        quantizer = self.shared_source_quantizer or self.eeg_source_quantizer
        return quantizer.get_codebook_entry(indices)

    def has_spatial_source_prior(self) -> bool:
        return bool(self.spatial_source_prior_enabled and self.fnirs_spatial_adjacency.numel() > 0)

    def get_spatial_prior_info(self) -> Optional[Dict[str, Any]]:
        if self.spatial_adjacency_info is None:
            return None
        payload = self.spatial_adjacency_info.to_serializable()
        payload['enabled'] = self.has_spatial_source_prior()
        payload['eeg_rms_smoothing_ms'] = float(self.eeg_rms_smoothing_ms)
        payload['warning_count'] = len(self.spatial_prior_warnings)
        return payload

    def set_alignment_scale(self, scale: float):
        self.alignment_scale = max(float(scale), 0.0)

    def get_alignment_scale(self) -> float:
        return float(self.alignment_scale)

    def set_source_target_scale(self, scale: float):
        self.source_target_scale = max(float(scale), 0.0)

    def get_source_target_scale(self) -> float:
        return float(self.source_target_scale)

    def set_observation_target_scale(self, scale: float):
        self.observation_target_scale = max(float(scale), 0.0)

    def get_observation_target_scale(self) -> float:
        return float(self.observation_target_scale)

    def set_balance_scales(
        self,
        source_scale: float | None = None,
        observation_scale: float | None = None,
    ):
        if source_scale is not None:
            self.source_balance_scale = max(float(source_scale), 0.0)
        if observation_scale is not None:
            self.observation_balance_scale = max(float(observation_scale), 0.0)

    def get_balance_scales(self) -> Dict[str, float]:
        return {
            'source_balance_scale': float(self.source_balance_scale),
            'observation_balance_scale': float(self.observation_balance_scale),
        }

    def set_quantization_strength(self, strength: float):
        self.quantization_strength = min(max(float(strength), 0.0), 1.0)
        for quantizer in (
            self.eeg_source_quantizer,
            self.fnirs_source_quantizer,
            self.shared_source_quantizer,
            self.eeg_observation_quantizer,
            self.fnirs_observation_quantizer,
        ):
            if quantizer is not None and hasattr(quantizer, 'set_quantization_strength'):
                quantizer.set_quantization_strength(self.quantization_strength)

    def get_quantization_strength(self) -> float:
        return float(self.quantization_strength)

    def set_source_codebook_updates_enabled(self, enabled: bool) -> None:
        for quantizer in (self.eeg_source_quantizer, self.fnirs_source_quantizer, self.shared_source_quantizer):
            if quantizer is not None and hasattr(quantizer, 'set_ema_updates_enabled'):
                quantizer.set_ema_updates_enabled(enabled)

    def set_cross_modal_gradient_scale(self, scale: float) -> None:
        self.cross_modal_gradient_scale = min(max(float(scale), 0.1), 20.0)

    def get_cross_modal_gradient_scale(self) -> float:
        return float(self.cross_modal_gradient_scale)

    def get_gradient_component_weights(self) -> Dict[str, float]:
        alignment_scale = float(self.get_alignment_scale())
        source_target_scale = float(self.get_source_target_scale())
        return {
            'eeg_rec_loss': 1.0,
            'fnirs_rec_loss': 1.0,
            'vq_loss': 1.0,
            'source_target_loss': self.source_target_weight * source_target_scale,
            'eeg_source_aux_loss': self.source_target_weight * self.eeg_source_aux_weight * source_target_scale,
            'source_target_corr_loss': self.source_target_correlation_weight * source_target_scale,
            'eeg_source_aux_corr_loss': self.eeg_source_aux_correlation_weight * source_target_scale,
            'observation_loss': self.observation_target_weight * float(self.get_observation_target_scale()),
            'source_coupling_lag_focus_loss': (
                self.coupling_weight * self.coupling_lag_focus_weight * alignment_scale
            ),
            'source_coupling_joint_entropy_loss': (
                self.coupling_weight * self.coupling_joint_entropy_weight * alignment_scale
            ),
            'source_coupling_smoothness_loss': (
                self.coupling_weight * self.coupling_smoothness_weight * alignment_scale
            ),
            'source_coupling_pair_likelihood_loss': (
                self.coupling_weight * self.coupling_pair_likelihood_weight * alignment_scale
            ),
            'source_coupling_lag_evidence_loss': (
                self.coupling_weight * self.coupling_lag_evidence_weight * alignment_scale
            ),
            'source_coupling_effective_smoothness_loss': (
                self.coupling_weight * self.coupling_effective_smoothness_weight * alignment_scale
            ),
            'source_coupling_interaction_lag_sparsity_loss': (
                self.coupling_weight * self.coupling_interaction_lag_sparsity_weight * alignment_scale
            ),
            'source_coupling_local_residual_loss': (
                self.coupling_weight * alignment_scale
            ),
            'source_coupling_context_residual_loss': (
                self.coupling_weight * alignment_scale
            ),
            'interaction_aux_loss': self.interaction_aux_weight,
            'shared_state_bottleneck_loss': self.shared_state_bottleneck_weight,
            'cross_modal_temporal_nce_loss': (
                self.cross_modal_temporal_nce_weight * alignment_scale * self.cross_modal_gradient_scale
            ),
            'cross_modal_masked_latent_loss': (
                self.cross_modal_masked_latent_weight * alignment_scale * self.cross_modal_gradient_scale
            ),
            'cross_modal_soft_code_distillation_loss': (
                self.cross_modal_soft_code_weight * alignment_scale * self.cross_modal_gradient_scale
            ),
            'codebook_balance_loss': self.codebook_balance_weight,
            'orthogonality_loss': self.orthogonality_weight,
        }

    def get_coupling_analysis_logits(self) -> torch.Tensor:
        if not self.coupling_residualize_fnirs_marginal:
            return self.coupling_logits
        return self.coupling_logits - self.coupling_logits.mean(dim=1, keepdim=True)

    def get_gradient_parameter_group_specs(self) -> List[Dict[str, Any]]:
        return [
            {'name': 'eeg_patch_embed', 'label': 'EEG Patch', 'prefixes': ('eeg_patch_embed.',)},
            {'name': 'fnirs_patch_embed', 'label': 'fNIRS Patch', 'prefixes': ('fnirs_patch_embed.',)},
            {'name': 'eeg_encoder', 'label': 'EEG Encoder', 'prefixes': ('eeg_encoder.',)},
            {'name': 'fnirs_encoder', 'label': 'fNIRS Encoder', 'prefixes': ('fnirs_encoder.',)},
            {'name': 'eeg_source_proj', 'label': 'EEG Source Proj', 'prefixes': ('eeg_source_proj.',)},
            {'name': 'eeg_observation_proj', 'label': 'EEG Obs Proj', 'prefixes': ('eeg_observation_proj.',)},
            {'name': 'fnirs_source_proj', 'label': 'fNIRS Source Proj', 'prefixes': ('fnirs_source_proj.',)},
            {'name': 'fnirs_observation_proj', 'label': 'fNIRS Obs Proj', 'prefixes': ('fnirs_observation_proj.',)},
            {'name': 'source_interaction_aux', 'label': 'Source Interaction Aux', 'prefixes': (
                'eeg_to_fnirs_source_predictor.',
                'fnirs_to_eeg_source_predictor.',
                'eeg_shared_state_proj.',
                'fnirs_shared_state_proj.',
            )},
            {'name': 'cross_modal_exchange', 'label': 'Cross Exchange', 'prefixes': ('cross_modal_exchange.',)},
            {'name': 'cross_modal_fusion', 'label': 'Cross Fusion', 'prefixes': ('cross_modal_fusion.',)},
            {'name': 'cross_modal_alignment', 'label': 'Cross Alignment', 'prefixes': (
                'cross_modal_eeg_projection.',
                'cross_modal_fnirs_projection.',
                'eeg_source_mask_token',
                'fnirs_source_mask_token',
            )},
            {'name': 'eeg_source_quantizer', 'label': 'EEG Source Quant', 'prefixes': ('eeg_source_quantizer.',)},
            {'name': 'fnirs_source_quantizer', 'label': 'fNIRS Source Quant', 'prefixes': ('fnirs_source_quantizer.',)},
            {'name': 'shared_source_quantizer', 'label': 'Shared Source Quant', 'prefixes': ('shared_source_quantizer.',)},
            {'name': 'eeg_observation_quantizer', 'label': 'EEG Obs Quant', 'prefixes': ('eeg_observation_quantizer.',)},
            {'name': 'fnirs_observation_quantizer', 'label': 'fNIRS Obs Quant', 'prefixes': ('fnirs_observation_quantizer.',)},
            {'name': 'coupling_logits', 'label': 'Coupling', 'prefixes': ('coupling_logits',)},
            {'name': 'context_coupling', 'label': 'Context Coupling', 'prefixes': (
                'context_coupling_eeg_factors',
                'context_coupling_fnirs_factors',
                'context_coupling_router.',
            )},
            {'name': 'hrf_parameters', 'label': 'HRF Params', 'prefixes': ('hrf_',)},
            {'name': 'eeg_source_decode_input_proj', 'label': 'EEG Source In', 'prefixes': ('eeg_source_decode_input_proj.',)},
            {'name': 'eeg_observation_decode_input_proj', 'label': 'EEG Obs In', 'prefixes': ('eeg_observation_decode_input_proj.',)},
            {'name': 'fnirs_source_decode_input_proj', 'label': 'fNIRS Source In', 'prefixes': ('fnirs_source_decode_input_proj.',)},
            {'name': 'fnirs_observation_decode_input_proj', 'label': 'fNIRS Obs In', 'prefixes': ('fnirs_observation_decode_input_proj.',)},
            {'name': 'eeg_source_decoder', 'label': 'EEG Source Decoder', 'prefixes': ('eeg_source_decoder.',)},
            {'name': 'eeg_observation_decoder', 'label': 'EEG Obs Decoder', 'prefixes': ('eeg_observation_decoder.',)},
            {'name': 'fnirs_source_decoder', 'label': 'fNIRS Source Decoder', 'prefixes': ('fnirs_source_decoder.',)},
            {'name': 'fnirs_observation_decoder', 'label': 'fNIRS Obs Decoder', 'prefixes': ('fnirs_observation_decoder.',)},
            {'name': 'eeg_source_amplitude_head', 'label': 'EEG Source Amp', 'prefixes': ('eeg_source_amplitude_head.',)},
            {'name': 'eeg_source_phase_head', 'label': 'EEG Source Phase', 'prefixes': ('eeg_source_phase_head.',)},
            {'name': 'eeg_observation_amplitude_head', 'label': 'EEG Obs Amp', 'prefixes': ('eeg_observation_amplitude_head.',)},
            {'name': 'eeg_observation_phase_head', 'label': 'EEG Obs Phase', 'prefixes': ('eeg_observation_phase_head.',)},
            {'name': 'fnirs_source_amplitude_head', 'label': 'fNIRS Source Amp', 'prefixes': ('fnirs_source_amplitude_head.',)},
            {'name': 'fnirs_source_phase_head', 'label': 'fNIRS Source Phase', 'prefixes': ('fnirs_source_phase_head.',)},
            {'name': 'fnirs_observation_amplitude_head', 'label': 'fNIRS Obs Amp', 'prefixes': ('fnirs_observation_amplitude_head.',)},
            {'name': 'fnirs_observation_phase_head', 'label': 'fNIRS Obs Phase', 'prefixes': ('fnirs_observation_phase_head.',)},
        ]


__all__ = ['CausalLowRankCrossAdapter', 'LagAwareCrossModalFusion', 'SourceObservationLaBraMVQNSP']
