"""Continuous shared/private tokenizer for modality-specific SSM observations.

The shared EEG and fNIRS branches reconstruct separate clean observation
trajectories.  Private branches reconstruct only the corresponding observation
residual.  No token ID alignment or symmetric cross-modal matching is present.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import torch
import torch.nn as nn

from .lag_conditioned_shared_private_vq import (
    FullWindowPatchEncoder,
    LocalCausalPatchEncoder,
    _masked_mean,
)


class TokenObservationDecoder(nn.Module):
    """Decode each latent token into one flattened observation patch."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if min(int(input_dim), int(hidden_dim), int(output_dim)) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def forward(
        self, tokens: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("decoder expects [B,N,D] tokens and [B,N] mask")
        output = self.network(tokens)
        return output.masked_fill(~valid_mask.bool().unsqueeze(-1), 0.0)


class CausalFIRTransferHead(nn.Module):
    """Low-capacity asymmetric EEG→future-fNIRS token trajectory head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        lags: Sequence[int] = (1, 2, 3, 4, 5),
    ) -> None:
        super().__init__()
        values = tuple(int(value) for value in lags)
        if not values or any(value < 0 for value in values):
            raise ValueError("FIR lags must be a non-empty non-negative sequence")
        if len(values) != len(set(values)):
            raise ValueError("FIR lags must be unique")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.lags = tuple(sorted(values))
        self.weight = nn.Parameter(
            torch.empty(len(self.lags), self.input_dim, self.output_dim)
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        eeg_shared: torch.Tensor,
        eeg_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eeg_shared.ndim != 3 or eeg_shared.shape[-1] != self.input_dim:
            raise ValueError("FIR input has the wrong shape")
        if eeg_valid_mask.shape != eeg_shared.shape[:2]:
            raise ValueError("FIR mask must match EEG token axes")
        batch, tokens, _ = eeg_shared.shape
        output = torch.zeros(
            batch,
            tokens,
            self.output_dim,
            device=eeg_shared.device,
            dtype=eeg_shared.dtype,
        )
        support = torch.zeros(
            batch, tokens, device=eeg_shared.device, dtype=torch.bool
        )
        for lag_index, lag in enumerate(self.lags):
            if lag >= tokens:
                continue
            source = eeg_shared[:, : tokens - lag]
            source_mask = eeg_valid_mask[:, : tokens - lag].bool()
            contribution = torch.einsum(
                "btd,df->btf", source, self.weight[lag_index]
            )
            output[:, lag:] = output[:, lag:] + contribution.masked_fill(
                ~source_mask.unsqueeze(-1), 0.0
            )
            support[:, lag:] |= source_mask
        output = (output + self.bias).masked_fill(~support.unsqueeze(-1), 0.0)
        return output, support


class ContinuousLagInteractionHead(nn.Module):
    """Bias-free, lag-balanced interaction over two continuous shared streams."""

    def __init__(
        self,
        shared_dim: int,
        class_count: int,
        *,
        rank: int = 8,
        allowed_lags: Sequence[int] = (0, 1, 2, 3, 4, 5),
    ) -> None:
        super().__init__()
        lags = tuple(sorted(int(value) for value in allowed_lags))
        if not lags or any(value < 0 for value in lags):
            raise ValueError("interaction lags must be non-empty and non-negative")
        if class_count <= 1 or rank <= 0:
            raise ValueError("interaction class_count/rank is invalid")
        self.allowed_lags = lags
        self.class_count = int(class_count)
        self.rank = int(rank)
        self.eeg_projection = nn.Linear(int(shared_dim), self.rank, bias=False)
        self.fnirs_projection = nn.Linear(int(shared_dim), self.rank, bias=False)
        self.class_factor = nn.Parameter(
            torch.empty(len(lags), self.class_count, self.rank)
        )
        nn.init.xavier_uniform_(self.class_factor)

    def forward(
        self,
        eeg_shared: torch.Tensor,
        fnirs_shared: torch.Tensor,
        eeg_valid_mask: torch.Tensor,
        fnirs_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eeg_shared.shape[:2] != eeg_valid_mask.shape:
            raise ValueError("EEG shared mask mismatch")
        if fnirs_shared.shape[:2] != fnirs_valid_mask.shape:
            raise ValueError("fNIRS shared mask mismatch")
        if eeg_shared.shape[0] != fnirs_shared.shape[0]:
            raise ValueError("interaction batch sizes differ")
        query = self.eeg_projection(eeg_shared)
        key = self.fnirs_projection(fnirs_shared)
        per_lag = []
        support = []
        for lag_index, lag in enumerate(self.allowed_lags):
            length = min(eeg_shared.shape[1], fnirs_shared.shape[1] - lag)
            if length <= 0:
                continue
            pair_mask = (
                eeg_valid_mask[:, :length].bool()
                & fnirs_valid_mask[:, lag : lag + length].bool()
            )
            interaction = query[:, :length] * key[:, lag : lag + length]
            pooled = (
                interaction.masked_fill(~pair_mask.unsqueeze(-1), 0.0).sum(dim=1)
                / pair_mask.to(interaction.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            logits = torch.einsum(
                "br,cr->bc", pooled, self.class_factor[lag_index]
            )
            per_lag.append(logits)
            support.append(pair_mask.any(dim=1))
        if not per_lag:
            raise ValueError("no configured interaction lag has token support")
        stacked = torch.stack(per_lag, dim=1)
        admitted = torch.stack(support, dim=1)
        logits = stacked.masked_fill(~admitted.unsqueeze(-1), 0.0).sum(dim=1)
        logits = logits / admitted.to(stacked.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        # Class centering removes every sample-wise common offset.  There are no
        # class or lag biases in this head.
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return logits, admitted


class ContinuousDecomposedTaskHead(nn.Module):
    """Export private, shared marginal, and incremental interaction logits."""

    def __init__(
        self,
        *,
        shared_dim: int,
        eeg_private_dim: int,
        fnirs_private_dim: int,
        class_count: int,
        hidden_dim: int = 64,
        interaction_rank: int = 8,
        allowed_lags: Sequence[int] = (0, 1, 2, 3, 4, 5),
        interaction_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.private_head = nn.Sequential(
            nn.LayerNorm(int(eeg_private_dim) + int(fnirs_private_dim)),
            nn.Linear(int(eeg_private_dim) + int(fnirs_private_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(class_count)),
        )
        self.shared_marginal_head = nn.Sequential(
            nn.LayerNorm(2 * int(shared_dim)),
            nn.Linear(2 * int(shared_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(class_count)),
        )
        self.interaction_head = ContinuousLagInteractionHead(
            shared_dim,
            class_count,
            rank=interaction_rank,
            allowed_lags=allowed_lags,
        )
        self.interaction_weight = float(interaction_weight)

    def forward(
        self,
        *,
        eeg_shared: torch.Tensor,
        fnirs_shared: torch.Tensor,
        eeg_private: torch.Tensor,
        fnirs_private: torch.Tensor,
        eeg_mask: torch.Tensor,
        fnirs_mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        private_pooled = torch.cat(
            (
                _masked_mean(eeg_private, eeg_mask),
                _masked_mean(fnirs_private, fnirs_mask),
            ),
            dim=-1,
        )
        shared_pooled = torch.cat(
            (
                _masked_mean(eeg_shared, eeg_mask),
                _masked_mean(fnirs_shared, fnirs_mask),
            ),
            dim=-1,
        )
        private = self.private_head(private_pooled)
        marginal = self.shared_marginal_head(shared_pooled)
        interaction, lag_support = self.interaction_head(
            eeg_shared, fnirs_shared, eeg_mask, fnirs_mask
        )
        private_marginal = private + marginal
        full = private_marginal + self.interaction_weight * interaction
        return {
            "private_only_logits": private,
            "shared_marginal_only_logits": marginal,
            "interaction_only_logits": interaction,
            "private_plus_shared_marginal_logits": private_marginal,
            "private_shared_interaction_logits": full,
            "interaction_lag_support": lag_support,
        }


class SSMObservationSharedPrivateModel(nn.Module):
    """Four-encoder continuous model with separate clean/residual duties."""

    architecture_name = "ssm_observation_shared_private_continuous_v1"
    vector_quantization = False

    def __init__(
        self,
        *,
        eeg_channels: int,
        fnirs_channels: int,
        eeg_patch_samples: int,
        fnirs_patch_samples: int,
        num_tokens: int,
        eeg_target_dim: int,
        fnirs_target_dim: int,
        shared_dim: int = 64,
        eeg_private_dim: int = 64,
        fnirs_private_dim: int = 32,
        eeg_shared_history_tokens: int = 2,
        fnirs_shared_history_tokens: int = 3,
        encoder_depth: int = 2,
        encoder_num_heads: int = 4,
        encoder_feedforward_dim: int = 256,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.1,
        class_count: int | None = None,
        interaction_rank: int = 8,
        allowed_lags: Sequence[int] = (0, 1, 2, 3, 4, 5),
        interaction_weight: float = 1.0,
        cross_prediction_lags: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.eeg_target_dim = int(eeg_target_dim)
        self.fnirs_target_dim = int(fnirs_target_dim)
        shared_kwargs = dict(
            num_tokens=int(num_tokens),
            latent_dim=int(shared_dim),
            depth=int(encoder_depth),
            num_heads=int(encoder_num_heads),
            feedforward_dim=int(encoder_feedforward_dim),
            dropout=float(dropout),
        )
        self.eeg_shared_encoder = LocalCausalPatchEncoder(
            input_channels=int(eeg_channels),
            patch_samples=int(eeg_patch_samples),
            history_patches=int(eeg_shared_history_tokens),
            **shared_kwargs,
        )
        self.fnirs_shared_encoder = LocalCausalPatchEncoder(
            input_channels=int(fnirs_channels),
            patch_samples=int(fnirs_patch_samples),
            history_patches=int(fnirs_shared_history_tokens),
            **shared_kwargs,
        )
        private_kwargs = dict(
            num_tokens=int(num_tokens),
            depth=int(encoder_depth),
            num_heads=int(encoder_num_heads),
            feedforward_dim=int(encoder_feedforward_dim),
            dropout=float(dropout),
        )
        self.eeg_private_encoder = FullWindowPatchEncoder(
            input_channels=int(eeg_channels),
            patch_samples=int(eeg_patch_samples),
            latent_dim=int(eeg_private_dim),
            **private_kwargs,
        )
        self.fnirs_private_encoder = FullWindowPatchEncoder(
            input_channels=int(fnirs_channels),
            patch_samples=int(fnirs_patch_samples),
            latent_dim=int(fnirs_private_dim),
            **private_kwargs,
        )
        self.eeg_clean_decoder = TokenObservationDecoder(
            shared_dim, decoder_hidden_dim, self.eeg_target_dim
        )
        self.fnirs_clean_decoder = TokenObservationDecoder(
            shared_dim, decoder_hidden_dim, self.fnirs_target_dim
        )
        # Private residual decoders never consume shared tokens.
        self.eeg_residual_decoder = TokenObservationDecoder(
            eeg_private_dim, decoder_hidden_dim, self.eeg_target_dim
        )
        self.fnirs_residual_decoder = TokenObservationDecoder(
            fnirs_private_dim, decoder_hidden_dim, self.fnirs_target_dim
        )
        self.cross_prediction_head = (
            None
            if cross_prediction_lags is None
            else CausalFIRTransferHead(
                shared_dim,
                self.fnirs_target_dim,
                lags=cross_prediction_lags,
            )
        )
        self.task_head = (
            None
            if class_count is None
            else ContinuousDecomposedTaskHead(
                shared_dim=shared_dim,
                eeg_private_dim=eeg_private_dim,
                fnirs_private_dim=fnirs_private_dim,
                class_count=int(class_count),
                interaction_rank=int(interaction_rank),
                allowed_lags=allowed_lags,
                interaction_weight=float(interaction_weight),
            )
        )

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        eeg_token_valid_mask: torch.Tensor | None = None,
        fnirs_token_valid_mask: torch.Tensor | None = None,
    ) -> Mapping[str, Any]:
        eeg_mask = self.eeg_shared_encoder.resolve_valid_mask(
            eeg, eeg_token_valid_mask
        )
        fnirs_mask = self.fnirs_shared_encoder.resolve_valid_mask(
            fnirs, fnirs_token_valid_mask
        )
        eeg_shared = self.eeg_shared_encoder(eeg, eeg_mask)
        fnirs_shared = self.fnirs_shared_encoder(fnirs, fnirs_mask)
        eeg_private = self.eeg_private_encoder(eeg, eeg_mask)
        fnirs_private = self.fnirs_private_encoder(fnirs, fnirs_mask)
        eeg_clean = self.eeg_clean_decoder(eeg_shared, eeg_mask)
        fnirs_clean = self.fnirs_clean_decoder(fnirs_shared, fnirs_mask)
        eeg_residual = self.eeg_residual_decoder(eeg_private, eeg_mask)
        fnirs_residual = self.fnirs_residual_decoder(fnirs_private, fnirs_mask)
        output: dict[str, Any] = {
            "eeg_shared": eeg_shared,
            "fnirs_shared": fnirs_shared,
            "eeg_private": eeg_private,
            "fnirs_private": fnirs_private,
            "eeg_clean_prediction": eeg_clean,
            "fnirs_clean_prediction": fnirs_clean,
            "eeg_residual_prediction": eeg_residual,
            "fnirs_residual_prediction": fnirs_residual,
            "eeg_observation_reconstruction": eeg_clean + eeg_residual,
            "fnirs_observation_reconstruction": fnirs_clean + fnirs_residual,
            "eeg_token_valid_mask": eeg_mask,
            "fnirs_token_valid_mask": fnirs_mask,
        }
        if self.cross_prediction_head is not None:
            prediction, support = self.cross_prediction_head(eeg_shared, eeg_mask)
            output["fnirs_cross_prediction"] = prediction
            output["fnirs_cross_prediction_valid_mask"] = support
        if self.task_head is not None:
            output.update(
                self.task_head(
                    eeg_shared=eeg_shared,
                    fnirs_shared=fnirs_shared,
                    eeg_private=eeg_private,
                    fnirs_private=fnirs_private,
                    eeg_mask=eeg_mask,
                    fnirs_mask=fnirs_mask,
                )
            )
        return output


__all__ = [
    "CausalFIRTransferHead",
    "ContinuousDecomposedTaskHead",
    "ContinuousLagInteractionHead",
    "SSMObservationSharedPrivateModel",
    "TokenObservationDecoder",
]
