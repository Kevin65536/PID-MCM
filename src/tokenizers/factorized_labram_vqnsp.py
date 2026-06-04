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
    align_pair,
    batch_usage_entropy_loss,
    coupling_eeg_neighbor_smoothness_loss,
    coupling_joint_probabilities,
    coupling_lag_focus_loss,
    orthogonality_loss,
    straight_through_assignment_probs,
)

from .base import BaseTokenizer
from .labram_vqnsp import (
    MultiChannelPatchEmbedding,
    NormEMAVectorQuantizer,
    TransformerDecoder,
    TransformerEncoder,
    l2norm,
)


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
        coupling_smoothness_weight: float = 0.2,
        coupling_smoothness_neighbors: int = 5,
        source_target_weight: float = 0.0,
        eeg_source_aux_weight: float = 0.5,
        observation_target_weight: float = 0.0,
        codebook_balance_weight: float = 0.02,
        source_balance_scale: float = 1.0,
        observation_balance_scale: float = 1.0,
        coupling_bidirectional: bool = True,
        orthogonality_weight: float = 0.01,
        assignment_temperature: float = 1.0,
        source_balance_temperature: float | None = None,
        observation_balance_temperature: float | None = None,
        coupling_temperature: float = 0.2,
        source_simvq_enabled: bool = False,
        source_simvq_loss_weight: float = 1.0,
        observation_simvq_enabled: bool = False,
        observation_simvq_loss_weight: float = 1.0,
        alignment_lag_candidates: List[int] | None = None,
        alignment_selection: str = 'min',
        alignment_compare_mode: str = 'variable',
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
        self.coupling_smoothness_weight = max(float(coupling_smoothness_weight), 0.0)
        self.coupling_smoothness_neighbors = max(int(coupling_smoothness_neighbors), 0)
        self.source_target_weight = max(float(source_target_weight), 0.0)
        self.eeg_source_aux_weight = max(float(eeg_source_aux_weight), 0.0)
        self.observation_target_weight = max(float(observation_target_weight), 0.0)
        self.codebook_balance_weight = codebook_balance_weight
        self.source_balance_scale = max(float(source_balance_scale), 0.0)
        self.observation_balance_scale = max(float(observation_balance_scale), 0.0)
        self.default_source_balance_scale = float(self.source_balance_scale)
        self.default_observation_balance_scale = float(self.observation_balance_scale)
        self.coupling_bidirectional = bool(coupling_bidirectional)
        self.orthogonality_weight = orthogonality_weight
        self.assignment_temperature = assignment_temperature
        self.balance_assignment_temperature = assignment_temperature
        self.source_balance_temperature = float(
            assignment_temperature if source_balance_temperature is None else source_balance_temperature
        )
        self.observation_balance_temperature = float(
            assignment_temperature if observation_balance_temperature is None else observation_balance_temperature
        )
        self.coupling_temperature = coupling_temperature
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
        self.source_branch_enabled = self._is_branch_enabled(self.source_branch_dropout)
        self.eeg_observation_branch_enabled = self._is_branch_enabled(self.eeg_observation_branch_dropout)
        self.fnirs_observation_branch_enabled = self._is_branch_enabled(self.fnirs_observation_branch_dropout)
        self.alignment_scale = 1.0
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
            torch.zeros(len(self.alignment_lag_candidates), source_codebook_size, source_codebook_size)
        )

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
        data_cfg = config.get('data', {})
        window_cfg = data_cfg.get('window', {})
        validation_cfg = config.get('validation', {})
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
            coupling_smoothness_weight=coupling_cfg.get('smoothness_weight', 0.2),
            coupling_smoothness_neighbors=coupling_cfg.get('smoothness_neighbors', 5),
            source_target_weight=source_target_cfg.get('weight', source_target_cfg.get('source_target_weight', 0.0)),
            eeg_source_aux_weight=source_target_cfg.get('eeg_aux_weight', source_target_cfg.get('eeg_source_aux_weight', 0.5)),
            observation_target_weight=observation_target_cfg.get('weight', 0.0),
            codebook_balance_weight=codebook_cfg.get('balance_weight', 0.02),
            source_balance_scale=codebook_cfg.get('source_balance_scale', 1.0),
            observation_balance_scale=codebook_cfg.get('observation_balance_scale', 1.0),
            coupling_bidirectional=coupling_cfg.get('bidirectional', True),
            orthogonality_weight=branch_cfg.get('orthogonality_weight', 0.01),
            assignment_temperature=codebook_cfg.get('assignment_temperature', 1.0),
            source_balance_temperature=codebook_cfg.get('source_assignment_temperature'),
            observation_balance_temperature=codebook_cfg.get('observation_assignment_temperature'),
            coupling_temperature=coupling_cfg.get('temperature', 0.2),
            source_simvq_enabled=quantizer_cfg.get('source_simvq_enabled', False),
            source_simvq_loss_weight=quantizer_cfg.get('source_simvq_loss_weight', 1.0),
            observation_simvq_enabled=quantizer_cfg.get('observation_simvq_enabled', False),
            observation_simvq_loss_weight=quantizer_cfg.get('observation_simvq_loss_weight', 1.0),
            alignment_lag_candidates=coupling_cfg.get('lag_candidates', validation_cfg.get('lag_set', [0])),
            alignment_selection=coupling_cfg.get('selection', 'min'),
            alignment_compare_mode=coupling_cfg.get('compare_mode', 'variable'),
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
        left_codes = set(torch.unique(left_indices).tolist())
        right_codes = set(torch.unique(right_indices).tolist())
        union_size = max(len(left_codes | right_codes), 1)
        overlap = len(left_codes & right_codes) / union_size
        return torch.tensor(overlap, device=left_indices.device, dtype=torch.float32)

    def _compute_source_alignment_state(
        self,
        eeg_source_logits: torch.Tensor,
        fnirs_source_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        temperature = max(float(self.coupling_temperature), 1e-3)
        eeg_source_probs = F.softmax(eeg_source_logits / temperature, dim=-1)
        fnirs_source_probs = F.softmax(fnirs_source_logits / temperature, dim=-1)

        lag_focus_loss = coupling_lag_focus_loss(self.coupling_logits)
        smoothness_loss = coupling_eeg_neighbor_smoothness_loss(
            self.coupling_logits,
            self.eeg_source_quantizer.get_codebook_weight(),
            n_neighbors=self.coupling_smoothness_neighbors,
        )
        source_coupling_loss = (
            self.coupling_lag_focus_weight * lag_focus_loss +
            self.coupling_smoothness_weight * smoothness_loss
        )

        coupling_joint_probs = coupling_joint_probabilities(self.coupling_logits).detach()
        lag_mass = coupling_joint_probs.sum(dim=-1)
        lag_profile = lag_mass.mean(dim=0) if lag_mass.numel() > 0 else eeg_source_logits.new_zeros((1,))
        selected_lag_index = int(torch.argmax(lag_profile).item()) if lag_profile.numel() > 0 else 0
        selected_lag = int(self.alignment_lag_candidates[selected_lag_index]) if self.alignment_lag_candidates else 0
        target_length = self.fixed_alignment_compare_length if self.alignment_compare_mode == 'fixed_min' else None
        aligned_eeg_probs, _ = align_pair(
            eeg_source_probs,
            fnirs_source_probs,
            selected_lag,
            target_length=target_length,
        )
        selected_source_lag = eeg_source_logits.new_tensor(float(selected_lag))
        source_alignment_usable_tokens = eeg_source_logits.new_tensor(float(aligned_eeg_probs.shape[1]))

        return {
            'source_coupling_loss': source_coupling_loss,
            'source_coupling_lag_focus_loss': lag_focus_loss,
            'source_coupling_smoothness_loss': smoothness_loss,
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

        eeg_source_q, eeg_source_indices, eeg_source_info = self.eeg_source_quantizer(eeg_source)
        fnirs_source_q, fnirs_source_indices, fnirs_source_info = self.fnirs_source_quantizer(fnirs_source)
        eeg_observation_q, eeg_observation_indices, eeg_observation_info = self.eeg_observation_quantizer(eeg_observation)
        fnirs_observation_q, fnirs_observation_indices, fnirs_observation_info = self.fnirs_observation_quantizer(fnirs_observation)

        eeg_source_q = self._branch_dropout(eeg_source_q, self.source_branch_dropout)
        fnirs_source_q = self._branch_dropout(fnirs_source_q, self.source_branch_dropout)
        eeg_observation_q = self._branch_dropout(eeg_observation_q, self.eeg_observation_branch_dropout)
        fnirs_observation_q = self._branch_dropout(fnirs_observation_q, self.fnirs_observation_branch_dropout)

        eeg_source_logits = self._assignment_logits(eeg_source, self.eeg_source_quantizer.get_codebook_weight())
        fnirs_source_logits = self._assignment_logits(fnirs_source, self.fnirs_source_quantizer.get_codebook_weight())
        eeg_observation_logits = self._assignment_logits(
            eeg_observation,
            self.eeg_observation_quantizer.get_codebook_weight(),
        )
        fnirs_observation_logits = self._assignment_logits(
            fnirs_observation,
            self.fnirs_observation_quantizer.get_codebook_weight(),
        )

        source_alignment = self._compute_source_alignment_state(eeg_source_logits, fnirs_source_logits)
        source_coupling_loss = source_alignment['source_coupling_loss']
        source_coupling_lag_focus_loss = source_alignment['source_coupling_lag_focus_loss']
        source_coupling_smoothness_loss = source_alignment['source_coupling_smoothness_loss']
        selected_source_lag = source_alignment['selected_source_lag']
        alignment_usable_tokens = source_alignment['alignment_usable_tokens']
        eeg_source_probs = source_alignment['eeg_source_probs']
        fnirs_source_probs = source_alignment['fnirs_source_probs']

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
        eeg_source_aux_loss = (
            self.loss_fn(eeg_source_reconstructed, eeg_source_aux_target)
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

        total_loss = (
            eeg_rec_loss +
            fnirs_rec_loss +
            vq_loss +
            (self.source_target_weight * self.source_target_scale) * source_target_loss +
            (self.source_target_weight * self.eeg_source_aux_weight * self.source_target_scale) * eeg_source_aux_loss +
            (self.observation_target_weight * self.observation_target_scale) * observation_loss +
            self.coupling_weight * source_coupling_loss +
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
            'eeg_source_aux_loss': eeg_source_aux_loss,
            'eeg_observation_loss': eeg_observation_loss,
            'fnirs_observation_loss': fnirs_observation_loss,
            'observation_loss': observation_loss,
            'source_coupling_loss': source_coupling_loss,
            'source_coupling_lag_focus_loss': source_coupling_lag_focus_loss,
            'source_coupling_smoothness_loss': source_coupling_smoothness_loss,
            'codebook_balance_loss': codebook_balance_loss,
            'source_balance_loss': source_balance_loss,
            'observation_balance_loss': observation_balance_loss,
            'orthogonality_loss': branch_orthogonality_loss,
            'selected_source_lag': selected_source_lag,
            'selected_alignment_lag': selected_source_lag,
            'source_alignment_usable_tokens': alignment_usable_tokens,
            'alignment_usable_tokens': alignment_usable_tokens,
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
        return self.eeg_source_quantizer.get_codebook_entry(indices)

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
            self.eeg_observation_quantizer,
            self.fnirs_observation_quantizer,
        ):
            if hasattr(quantizer, 'set_quantization_strength'):
                quantizer.set_quantization_strength(self.quantization_strength)

    def get_quantization_strength(self) -> float:
        return float(self.quantization_strength)

    def get_gradient_component_weights(self) -> Dict[str, float]:
        alignment_scale = float(self.get_alignment_scale())
        source_target_scale = float(self.get_source_target_scale())
        return {
            'eeg_rec_loss': 1.0,
            'fnirs_rec_loss': 1.0,
            'vq_loss': 1.0,
            'source_target_loss': self.source_target_weight * source_target_scale,
            'eeg_source_aux_loss': self.source_target_weight * self.eeg_source_aux_weight * source_target_scale,
            'observation_loss': self.observation_target_weight * float(self.get_observation_target_scale()),
            'source_coupling_loss': self.coupling_weight,
            'codebook_balance_loss': self.codebook_balance_weight,
            'orthogonality_loss': self.orthogonality_weight,
        }

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
            {'name': 'eeg_source_quantizer', 'label': 'EEG Source Quant', 'prefixes': ('eeg_source_quantizer.',)},
            {'name': 'fnirs_source_quantizer', 'label': 'fNIRS Source Quant', 'prefixes': ('fnirs_source_quantizer.',)},
            {'name': 'eeg_observation_quantizer', 'label': 'EEG Obs Quant', 'prefixes': ('eeg_observation_quantizer.',)},
            {'name': 'fnirs_observation_quantizer', 'label': 'fNIRS Obs Quant', 'prefixes': ('fnirs_observation_quantizer.',)},
            {'name': 'coupling_logits', 'label': 'Coupling', 'prefixes': ('coupling_logits',)},
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


__all__ = ['SourceObservationLaBraMVQNSP']
