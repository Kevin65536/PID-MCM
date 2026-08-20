"""Local/causal shared-private VQ core with lag-conditioned coupling.

This module is intentionally self contained.  It does not alter the frozen
``continuous_shared_private`` implementation and exposes only reusable model
components: patch encoders, independent EMA quantizers, continuous projection
heads, native-feature decoder hooks, a low-rank coupling head, and a strict
mask-aware lag matching objective.

The formal LC-SPVQ defaults are two-second non-overlapping patches, a 64
-dimensional shared space, and independent 16-entry EEG/fNIRS codebooks.  The
small dimensions and token counts remain configurable for synthetic tests and
ablation work, while the shared encoders are always local and causal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ema_vector_quantizer import EMAVectorQuantizer, QuantizerOutput


# ---------------------------------------------------------------------------
# Small shape/masking helpers
# ---------------------------------------------------------------------------


def _check_signal(
    signal: torch.Tensor,
    *,
    input_channels: int,
    expected_samples: int,
) -> None:
    if signal.ndim != 3 or signal.shape[1] != input_channels:
        raise ValueError(
            f"expected [B,{input_channels},T] signal, got {tuple(signal.shape)}"
        )
    if signal.shape[-1] != expected_samples:
        raise ValueError(
            f"expected exactly {expected_samples} samples, got {signal.shape[-1]}"
        )


def _resolve_token_mask(
    signal: torch.Tensor,
    token_valid_mask: torch.Tensor | None,
    *,
    num_tokens: int,
) -> torch.Tensor:
    if token_valid_mask is None:
        return torch.ones(
            signal.shape[0], num_tokens, device=signal.device, dtype=torch.bool
        )
    if tuple(token_valid_mask.shape) != (signal.shape[0], num_tokens):
        raise ValueError(
            f"token_valid_mask must be [B,{num_tokens}], got "
            f"{tuple(token_valid_mask.shape)}"
        )
    return token_valid_mask.to(device=signal.device, dtype=torch.bool)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the token axis, returning zero for an all-invalid row."""

    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("values must be [B,N,D] and mask must be [B,N]")
    admitted = mask.to(device=values.device, dtype=torch.bool).unsqueeze(-1)
    masked_values = torch.where(admitted, values, torch.zeros_like(values))
    if not bool(torch.isfinite(masked_values).all()):
        raise FloatingPointError("masked mean contains a non-finite admitted value")
    weights = admitted.to(dtype=values.dtype)
    return masked_values.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_pair_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4 or mask.shape != values.shape[:3]:
        raise ValueError("values must be [B,N,M,C] and mask must be [B,N,M]")
    admitted = mask.to(device=values.device, dtype=torch.bool).unsqueeze(-1)
    masked_values = torch.where(admitted, values, torch.zeros_like(values))
    if not bool(torch.isfinite(masked_values).all()):
        raise FloatingPointError("masked pair mean contains a non-finite admitted value")
    weights = admitted.to(dtype=values.dtype)
    return masked_values.sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1.0)


# ---------------------------------------------------------------------------
# Patch encoders
# ---------------------------------------------------------------------------


class LocalCausalPatchEncoder(nn.Module):
    """Patch encoder with a finite *past-only* token receptive field.

    ``history_patches`` counts the current patch.  Thus the formal EEG value
    ``2`` exposes only ``[t-1, t]`` and never a future patch; a fNIRS value of
    ``3`` exposes ``[t-2, t-1, t]``.  The attention mask is registered as a
    buffer so that the temporal scope is directly auditable in a checkpoint.
    """

    token_temporal_scope = "causal_local_history"

    def __init__(
        self,
        *,
        input_channels: int,
        patch_samples: int,
        num_tokens: int,
        latent_dim: int,
        history_patches: int = 2,
        depth: int = 1,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or patch_samples <= 0 or num_tokens <= 0:
            raise ValueError("input_channels, patch_samples, and num_tokens must be positive")
        if latent_dim <= 0 or num_heads <= 0 or latent_dim % num_heads:
            raise ValueError("num_heads must divide a positive latent_dim")
        if history_patches <= 0 or history_patches > num_tokens:
            raise ValueError("history_patches must be in [1, num_tokens]")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_channels = int(input_channels)
        self.patch_samples = int(patch_samples)
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.history_patches = int(history_patches)
        self.expected_samples = self.patch_samples * self.num_tokens
        self.patch_duration_s = 2.0

        self.patch_projection = nn.Conv1d(
            self.input_channels,
            self.latent_dim,
            kernel_size=self.patch_samples,
            stride=self.patch_samples,
        )
        self.patch_norm = nn.LayerNorm(self.latent_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_tokens, self.latent_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim or self.latent_dim * 4),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=int(depth),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

        # ``True`` means forbidden for MultiheadAttention.  Each query keeps
        # itself and only the requested number of preceding token positions.
        positions = torch.arange(self.num_tokens)
        lag = positions[:, None] - positions[None, :]
        allowed = (lag >= 0) & (lag < self.history_patches)
        self.register_buffer("attention_mask", ~allowed, persistent=True)

    def resolve_valid_mask(
        self,
        signal: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_signal(
            signal,
            input_channels=self.input_channels,
            expected_samples=self.expected_samples,
        )
        return _resolve_token_mask(
            signal, token_valid_mask, num_tokens=self.num_tokens
        )

    # Alias used by callers of the full-window encoder in this repository.
    resolve_mask = resolve_valid_mask

    def forward(
        self,
        signal: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_signal(
            signal,
            input_channels=self.input_channels,
            expected_samples=self.expected_samples,
        )
        valid = _resolve_token_mask(
            signal, token_valid_mask, num_tokens=self.num_tokens
        )
        tokens = self.patch_projection(signal).transpose(1, 2)
        tokens = self.patch_norm(F.gelu(tokens))
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)
        # This also prevents NaNs in an explicitly invalid patch from entering
        # the attention block as a key/value.
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)

        encoded = torch.zeros_like(tokens)
        nonempty = valid.any(dim=1)
        if bool(nonempty.any()):
            active = self.context(
                tokens[nonempty],
                mask=self.attention_mask.to(device=tokens.device),
                src_key_padding_mask=~valid[nonempty],
            )
            active = self.output_norm(active).masked_fill(
                ~valid[nonempty].unsqueeze(-1), 0.0
            )
            encoded = encoded.index_copy(
                0, nonempty.nonzero(as_tuple=False).squeeze(-1), active
            )
        return encoded


class FullWindowPatchEncoder(nn.Module):
    """Bidirectional full-window patch encoder for private information."""

    token_temporal_scope = "bidirectional_full_window"

    def __init__(
        self,
        *,
        input_channels: int,
        patch_samples: int,
        num_tokens: int,
        latent_dim: int,
        depth: int = 1,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or patch_samples <= 0 or num_tokens <= 0:
            raise ValueError("input_channels, patch_samples, and num_tokens must be positive")
        if latent_dim <= 0 or num_heads <= 0 or latent_dim % num_heads:
            raise ValueError("num_heads must divide a positive latent_dim")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_channels = int(input_channels)
        self.patch_samples = int(patch_samples)
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.expected_samples = self.patch_samples * self.num_tokens
        self.patch_duration_s = 2.0
        self.patch_projection = nn.Conv1d(
            self.input_channels,
            self.latent_dim,
            kernel_size=self.patch_samples,
            stride=self.patch_samples,
        )
        self.patch_norm = nn.LayerNorm(self.latent_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_tokens, self.latent_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim or self.latent_dim * 4),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=int(depth),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def resolve_valid_mask(
        self,
        signal: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_signal(
            signal,
            input_channels=self.input_channels,
            expected_samples=self.expected_samples,
        )
        return _resolve_token_mask(
            signal, token_valid_mask, num_tokens=self.num_tokens
        )

    resolve_mask = resolve_valid_mask

    def forward(
        self,
        signal: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_signal(
            signal,
            input_channels=self.input_channels,
            expected_samples=self.expected_samples,
        )
        valid = _resolve_token_mask(
            signal, token_valid_mask, num_tokens=self.num_tokens
        )
        tokens = self.patch_projection(signal).transpose(1, 2)
        tokens = self.patch_norm(F.gelu(tokens))
        tokens = (tokens + self.position_embedding.to(dtype=tokens.dtype)).masked_fill(
            ~valid.unsqueeze(-1), 0.0
        )
        encoded = torch.zeros_like(tokens)
        nonempty = valid.any(dim=1)
        if bool(nonempty.any()):
            active = self.context(
                tokens[nonempty], src_key_padding_mask=~valid[nonempty]
            )
            active = self.output_norm(active).masked_fill(
                ~valid[nonempty].unsqueeze(-1), 0.0
            )
            encoded = encoded.index_copy(
                0, nonempty.nonzero(as_tuple=False).squeeze(-1), active
            )
        return encoded


# Clear aliases make the intended branch semantics discoverable to downstream
# code without requiring knowledge of the internal class names.
CausalLocalPatchEncoder = LocalCausalPatchEncoder
PrivateFullWindowEncoder = FullWindowPatchEncoder


# ---------------------------------------------------------------------------
# Continuous projections, native decoder hooks, and coupling/classification
# ---------------------------------------------------------------------------


class ContinuousProjectionHead(nn.Module):
    """Project a shared continuous token into a matching/coupling space."""

    def __init__(
        self,
        input_dim: int = 64,
        projection_dim: int = 64,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or projection_dim <= 0:
            raise ValueError("input_dim and projection_dim must be positive")
        hidden_dim = int(hidden_dim or max(input_dim, projection_dim))
        self.input_dim = int(input_dim)
        self.projection_dim = int(projection_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.projection_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected [B,N,{self.input_dim}] latent, got {tuple(latent.shape)}"
            )
        return self.network(latent)


class NativeFeatureDecoder(nn.Module):
    """Historical raw decoder: ``[shared, private] -> [B,C,T]``.

    The model exposes instances under ``raw_decoders`` and retains
    ``native_feature_decoders`` only as a compatibility alias.
    """

    def __init__(
        self,
        *,
        shared_dim: int,
        private_dim: int,
        output_channels: int,
        patch_samples: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(shared_dim, private_dim, output_channels, patch_samples) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.shared_dim = int(shared_dim)
        self.private_dim = int(private_dim)
        self.output_channels = int(output_channels)
        self.patch_samples = int(patch_samples)
        hidden_dim = int(hidden_dim or max(128, shared_dim + private_dim))
        self.network = nn.Sequential(
            nn.LayerNorm(self.shared_dim + self.private_dim),
            nn.Linear(self.shared_dim + self.private_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_channels * self.patch_samples),
        )

    def forward(self, shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
        if shared.ndim != 3 or private.ndim != 3:
            raise ValueError("shared and private latents must be [B,N,D]")
        if shared.shape[:2] != private.shape[:2]:
            raise ValueError("shared/private token axes differ")
        if shared.shape[-1] != self.shared_dim or private.shape[-1] != self.private_dim:
            raise ValueError("shared/private latent dimensions differ")
        patches = self.network(torch.cat((shared, private), dim=-1))
        batch, tokens = patches.shape[:2]
        return (
            patches.view(batch, tokens, self.output_channels, self.patch_samples)
            .permute(0, 2, 1, 3)
            .reshape(batch, self.output_channels, tokens * self.patch_samples)
        )


class NativeTargetFeatureDecoder(nn.Module):
    """Decode only a pre-VQ shared token into native target features.

    Unlike :class:`NativeFeatureDecoder` (the historical raw decoder), this
    hook intentionally receives no private branch and never detaches its
    input.  Its output is token-major ``[B, N, F]`` so a native-feature loss
    can provide a direct gradient path into the shared encoder.
    """

    def __init__(
        self,
        *,
        shared_dim: int,
        native_feature_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if shared_dim <= 0 or native_feature_dim <= 0:
            raise ValueError("shared_dim and native_feature_dim must be positive")
        self.shared_dim = int(shared_dim)
        self.native_feature_dim = int(native_feature_dim)
        hidden_dim = int(hidden_dim or max(64, self.shared_dim))
        self.network = nn.Sequential(
            nn.LayerNorm(self.shared_dim),
            nn.Linear(self.shared_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.native_feature_dim),
        )

    def forward(self, pre_vq_shared: torch.Tensor) -> torch.Tensor:
        if pre_vq_shared.ndim != 3 or pre_vq_shared.shape[-1] != self.shared_dim:
            raise ValueError(
                f"expected [B,N,{self.shared_dim}] pre-VQ shared tokens, "
                f"got {tuple(pre_vq_shared.shape)}"
            )
        return self.network(pre_vq_shared)


# The original public name is retained, but its semantics are explicitly raw:
# [detach(shared), private] -> native waveform/feature layout.
RawFeatureDecoder = NativeFeatureDecoder


class LowRankLagCouplingHead(nn.Module):
    """Continuous lag-conditioned bilinear coupling with a low-rank factorization.

    The pairwise output has shape ``[B, N_eeg, N_fnirs, num_classes]``.  The
    interaction is factorized into lag-specific rank ``rank`` query/key
    products; the class interaction is the compact ``[lag,num_classes,rank]``
    factor rather than a full token-pair table.
    Inputs are posterior distributions (K=16 in the formal model), and the
    returned pair mask admits only explicitly configured non-negative lags.
    """

    def __init__(
        self,
        *,
        input_dim: int = 16,
        posterior_dim: int | None = None,
        rank: int = 8,
        allowed_lags: Sequence[int] = (0, 1, 2, 3, 4, 5),
        max_lag: int | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if posterior_dim is not None:
            input_dim = int(posterior_dim)
        if input_dim <= 0 or rank <= 0 or num_classes <= 0:
            raise ValueError("input_dim, rank, and num_classes must be positive")
        raw_lags = tuple(int(lag) for lag in allowed_lags)
        # ``max_lag`` is retained for the previous API.  When explicitly
        # supplied, it narrows the historical default to [0, max_lag]; an
        # explicit allowed_lags always wins.
        if max_lag is not None:
            if max_lag < 0:
                raise ValueError("max_lag must be non-negative")
            if raw_lags == (0, 1, 2, 3, 4, 5):
                raw_lags = tuple(range(int(max_lag) + 1))
        if not raw_lags or any(lag < 0 for lag in raw_lags):
            raise ValueError("allowed_lags must contain at least one non-negative lag")
        if len(set(raw_lags)) != len(raw_lags):
            raise ValueError("allowed_lags must not contain duplicates")
        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.allowed_lags = tuple(sorted(raw_lags))
        self.max_lag = max(self.allowed_lags)
        self.num_classes = int(num_classes)
        lag_count = len(self.allowed_lags)
        self.query_factor = nn.Linear(
            self.input_dim, lag_count * self.rank, bias=False
        )
        self.key_factor = nn.Linear(
            self.input_dim, lag_count * self.rank, bias=False
        )
        self.rank_to_class = nn.Parameter(
            torch.empty(lag_count, self.num_classes, self.rank)
        )
        self.lag_bias = nn.Parameter(
            torch.zeros(lag_count, self.num_classes)
        )
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.register_buffer(
            "allowed_lag_values", torch.as_tensor(self.allowed_lags, dtype=torch.long),
            persistent=True,
        )
        nn.init.xavier_uniform_(self.rank_to_class)

    def forward(
        self,
        eeg_projection: torch.Tensor,
        fnirs_projection: torch.Tensor,
        *,
        eeg_valid_mask: torch.Tensor | None = None,
        fnirs_valid_mask: torch.Tensor | None = None,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if eeg_projection.ndim != 3 or fnirs_projection.ndim != 3:
            raise ValueError("coupling projections must be [B,N,D]")
        if eeg_projection.shape[0] != fnirs_projection.shape[0]:
            raise ValueError("EEG and fNIRS batch sizes differ")
        if eeg_projection.shape[-1] != self.input_dim:
            raise ValueError(f"EEG projection dimension must be {self.input_dim}")
        if fnirs_projection.shape[-1] != self.input_dim:
            raise ValueError(f"fNIRS projection dimension must be {self.input_dim}")
        batch, eeg_tokens = eeg_projection.shape[:2]
        fnirs_tokens = fnirs_projection.shape[1]
        lag_count = len(self.allowed_lags)
        q = self.query_factor(eeg_projection).view(
            batch, eeg_tokens, lag_count, self.rank
        )
        k = self.key_factor(fnirs_projection).view(
            batch, fnirs_tokens, lag_count, self.rank
        )

        eeg_pos = torch.arange(eeg_tokens, device=eeg_projection.device)
        fnirs_pos = torch.arange(fnirs_tokens, device=eeg_projection.device)
        relative_lag = fnirs_pos[None, :] - eeg_pos[:, None]
        allowed_values = self.allowed_lag_values.to(device=eeg_projection.device)
        lag_matches = relative_lag.unsqueeze(-1) == allowed_values
        lag_index = lag_matches.to(torch.long).argmax(dim=-1)
        allowed_pair = lag_matches.any(dim=-1)

        # Select distinct a_{r,tau} and b_{r,tau} factors for each pair's lag.
        all_interactions = q.unsqueeze(2) * k.unsqueeze(1)
        gather_index = lag_index.view(1, eeg_tokens, fnirs_tokens, 1, 1).expand(
            batch, eeg_tokens, fnirs_tokens, 1, self.rank
        )
        interaction = torch.gather(
            all_interactions, dim=3, index=gather_index
        ).squeeze(3)
        class_factor = self.rank_to_class[lag_index]
        logits = torch.einsum("bijr,ijcr->bijc", interaction, class_factor)
        lag_term = self.lag_bias[lag_index]
        logits = logits + lag_term.unsqueeze(0) + self.class_bias

        if eeg_valid_mask is None:
            eeg_valid_mask = torch.ones(
                batch, eeg_tokens, device=logits.device, dtype=torch.bool
            )
        else:
            if tuple(eeg_valid_mask.shape) != (batch, eeg_tokens):
                raise ValueError("eeg_valid_mask must match EEG token axes")
            eeg_valid_mask = eeg_valid_mask.to(device=logits.device, dtype=torch.bool)
        if fnirs_valid_mask is None:
            fnirs_valid_mask = torch.ones(
                batch, fnirs_tokens, device=logits.device, dtype=torch.bool
            )
        else:
            if tuple(fnirs_valid_mask.shape) != (batch, fnirs_tokens):
                raise ValueError("fnirs_valid_mask must match fNIRS token axes")
            fnirs_valid_mask = fnirs_valid_mask.to(
                device=logits.device, dtype=torch.bool
            )
        pair_mask = (
            eeg_valid_mask.unsqueeze(-1)
            & fnirs_valid_mask.unsqueeze(-2)
            & allowed_pair.unsqueeze(0)
        )
        # Keep invalid entries finite and auditable.  The mask, not a sentinel
        # logit, carries validity into the loss/pooling code.  In particular,
        # negative lags and lags outside ``allowed_lags`` can never be pooled.
        logits = logits.masked_fill(~pair_mask.unsqueeze(-1), 0.0)
        if return_mask:
            return logits, pair_mask
        return logits


class SharedMarginalClassifier(nn.Module):
    """Classify from masked EEG/fNIRS posterior marginal distributions."""

    def __init__(
        self,
        *,
        codebook_size: int = 16,
        num_classes: int = 2,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if codebook_size <= 0 or num_classes <= 0:
            raise ValueError("codebook_size and num_classes must be positive")
        self.codebook_size = int(codebook_size)
        self.num_classes = int(num_classes)
        hidden_dim = int(hidden_dim or max(32, 2 * self.codebook_size))
        self.network = nn.Sequential(
            nn.LayerNorm(2 * self.codebook_size),
            nn.Linear(2 * self.codebook_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(
        self,
        eeg_posterior: torch.Tensor,
        fnirs_posterior: torch.Tensor,
        *,
        eeg_valid_mask: torch.Tensor | None = None,
        fnirs_valid_mask: torch.Tensor | None = None,
        return_pooled: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if eeg_posterior.ndim != 3 or fnirs_posterior.ndim != 3:
            raise ValueError("posteriors must be [B,N,K]")
        if eeg_posterior.shape[0] != fnirs_posterior.shape[0]:
            raise ValueError("posterior batch sizes differ")
        if eeg_posterior.shape[-1] != self.codebook_size:
            raise ValueError("EEG posterior codebook dimension mismatch")
        if fnirs_posterior.shape[-1] != self.codebook_size:
            raise ValueError("fNIRS posterior codebook dimension mismatch")
        b = eeg_posterior.shape[0]
        eeg_mask = (
            torch.ones(
                b, eeg_posterior.shape[1], device=eeg_posterior.device, dtype=torch.bool
            )
            if eeg_valid_mask is None
            else eeg_valid_mask.to(device=eeg_posterior.device, dtype=torch.bool)
        )
        fnirs_mask = (
            torch.ones(
                b, fnirs_posterior.shape[1], device=fnirs_posterior.device, dtype=torch.bool
            )
            if fnirs_valid_mask is None
            else fnirs_valid_mask.to(device=fnirs_posterior.device, dtype=torch.bool)
        )
        if eeg_mask.shape != eeg_posterior.shape[:2] or fnirs_mask.shape != fnirs_posterior.shape[:2]:
            raise ValueError("posterior masks must match token axes")
        pooled = torch.cat(
            (_masked_mean(eeg_posterior, eeg_mask), _masked_mean(fnirs_posterior, fnirs_mask)),
            dim=-1,
        )
        logits = self.network(pooled)
        if return_pooled:
            return logits, pooled
        return logits


class PrivatePooledClassifier(nn.Module):
    """Classify from masked pooled private EEG/fNIRS tokens."""

    def __init__(
        self,
        *,
        eeg_private_dim: int,
        fnirs_private_dim: int,
        num_classes: int = 2,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(eeg_private_dim, fnirs_private_dim, num_classes) <= 0:
            raise ValueError("classifier dimensions must be positive")
        self.eeg_private_dim = int(eeg_private_dim)
        self.fnirs_private_dim = int(fnirs_private_dim)
        self.num_classes = int(num_classes)
        hidden_dim = int(hidden_dim or max(64, self.eeg_private_dim + self.fnirs_private_dim))
        self.network = nn.Sequential(
            nn.LayerNorm(self.eeg_private_dim + self.fnirs_private_dim),
            nn.Linear(self.eeg_private_dim + self.fnirs_private_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(
        self,
        eeg_private: torch.Tensor,
        fnirs_private: torch.Tensor,
        *,
        eeg_valid_mask: torch.Tensor | None = None,
        fnirs_valid_mask: torch.Tensor | None = None,
        return_pooled: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if eeg_private.ndim != 3 or fnirs_private.ndim != 3:
            raise ValueError("private latents must be [B,N,D]")
        if eeg_private.shape[0] != fnirs_private.shape[0]:
            raise ValueError("private branch batch sizes differ")
        if eeg_private.shape[-1] != self.eeg_private_dim:
            raise ValueError("EEG private dimension mismatch")
        if fnirs_private.shape[-1] != self.fnirs_private_dim:
            raise ValueError("fNIRS private dimension mismatch")
        b = eeg_private.shape[0]
        eeg_mask = (
            torch.ones(
                b, eeg_private.shape[1], device=eeg_private.device, dtype=torch.bool
            )
            if eeg_valid_mask is None
            else eeg_valid_mask.to(device=eeg_private.device, dtype=torch.bool)
        )
        fnirs_mask = (
            torch.ones(
                b, fnirs_private.shape[1], device=fnirs_private.device, dtype=torch.bool
            )
            if fnirs_valid_mask is None
            else fnirs_valid_mask.to(device=fnirs_private.device, dtype=torch.bool)
        )
        if eeg_mask.shape != eeg_private.shape[:2] or fnirs_mask.shape != fnirs_private.shape[:2]:
            raise ValueError("private masks must match private token axes")
        pooled = torch.cat(
            (_masked_mean(eeg_private, eeg_mask), _masked_mean(fnirs_private, fnirs_mask)),
            dim=-1,
        )
        logits = self.network(pooled)
        if return_pooled:
            return logits, pooled
        return logits


# ---------------------------------------------------------------------------
# Lag-aware continuous matching loss
# ---------------------------------------------------------------------------


def _lag_pairs(
    positive_lag_weights: Mapping[int, float] | Sequence[float] | torch.Tensor,
    positive_lags: Sequence[int] | None,
) -> list[tuple[int, float]]:
    if isinstance(positive_lag_weights, Mapping):
        pairs = [(int(lag), float(weight)) for lag, weight in positive_lag_weights.items()]
    else:
        values = torch.as_tensor(positive_lag_weights).flatten().tolist()
        lags = list(range(len(values))) if positive_lags is None else [int(v) for v in positive_lags]
        if len(values) != len(lags):
            raise ValueError("positive_lags and positive_lag_weights must have equal length")
        pairs = list(zip(lags, (float(v) for v in values)))
    pairs = [(lag, weight) for lag, weight in pairs if weight > 0.0]
    if not pairs:
        raise ValueError("at least one positive lag must have positive weight")
    return pairs


def _metadata_flat(
    value: Any,
    *,
    batch: int,
    tokens: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} cannot be None here")
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.to(device=device)
    if tuple(tensor.shape) == (batch,):
        return tensor.repeat_interleave(tokens)
    if tuple(tensor.shape) == (batch, tokens):
        return tensor.reshape(-1)
    raise ValueError(
        f"{name} must be [B] or [B,N], got {tuple(tensor.shape)} for [B={batch},N={tokens}]"
    )


def _split_metadata(value: Any) -> tuple[Any, Any]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, Mapping) and "query" in value and "target" in value:
        return value["query"], value["target"]
    return value, value


def _pair_mask_to_flat(
    mask: torch.Tensor,
    *,
    query_batch: int,
    query_tokens: int,
    target_batch: int,
    target_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    mask = mask.to(device=device, dtype=torch.bool)
    expected = (query_batch * query_tokens, target_batch * target_tokens)
    if tuple(mask.shape) == expected:
        return mask
    if tuple(mask.shape) == (query_batch, target_batch):
        return mask.repeat_interleave(query_tokens, dim=0).repeat_interleave(
            target_tokens, dim=1
        )
    if tuple(mask.shape) == (query_batch, query_tokens, target_tokens):
        result = torch.zeros(expected, device=device, dtype=torch.bool)
        for index in range(query_batch):
            q0 = index * query_tokens
            q1 = q0 + query_tokens
            t0 = index * target_tokens
            t1 = t0 + target_tokens
            result[q0:q1, t0:t1] = mask[index]
        return result
    if tuple(mask.shape) == (
        query_batch,
        query_tokens,
        target_batch,
        target_tokens,
    ):
        return mask.reshape(expected)
    raise ValueError(
        "negative_mask must be [Bq,Bt], [Bq,Nq,Nt], [Bq,Nq,Bt,Nt], "
        f"or [Bq*Nq,Bt*Nt], got {tuple(mask.shape)}"
    )


def _direction_matching_loss(
    query: torch.Tensor,
    target: torch.Tensor,
    *,
    query_valid_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
    pairs: list[tuple[int, float]],
    temperature: float,
    query_trial_ids: Any,
    target_trial_ids: Any,
    query_subject_ids: Any,
    target_subject_ids: Any,
    query_relative_time: Any,
    target_relative_time: Any,
    negative_mask: torch.Tensor | None,
    additional_negative_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if query.ndim != 3 or target.ndim != 3:
        raise ValueError("matching features must be [B,N,D]")
    if query.shape[-1] != target.shape[-1]:
        raise ValueError("matching feature dimensions differ")
    qb, qn, _ = query.shape
    tb, tn, _ = target.shape
    if query_valid_mask.shape != (qb, qn):
        raise ValueError("query_valid_mask must match query token axes")
    if target_valid_mask.shape != (tb, tn):
        raise ValueError("target_valid_mask must match target token axes")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    qmask = query_valid_mask.to(device=query.device, dtype=torch.bool)
    tmask = target_valid_mask.to(device=target.device, dtype=torch.bool)
    # Strict masking happens before projection/normalization, so even NaNs in
    # an invalid input patch cannot contaminate a valid row.
    query = torch.where(qmask.unsqueeze(-1), query, torch.zeros_like(query))
    target = torch.where(tmask.unsqueeze(-1), target, torch.zeros_like(target))
    q_flat = F.normalize(query.reshape(-1, query.shape[-1]), dim=-1)
    t_flat = F.normalize(target.reshape(-1, target.shape[-1]), dim=-1)
    logits = q_flat @ t_flat.transpose(0, 1) / float(temperature)

    q_trial = (
        torch.arange(qb, device=query.device).repeat_interleave(qn)
        if query_trial_ids is None
        else _metadata_flat(
            query_trial_ids,
            batch=qb,
            tokens=qn,
            device=query.device,
            name="query_trial_ids",
        )
    )
    t_trial = (
        torch.arange(tb, device=query.device).repeat_interleave(tn)
        if target_trial_ids is None
        else _metadata_flat(
            target_trial_ids,
            batch=tb,
            tokens=tn,
            device=query.device,
            name="target_trial_ids",
        )
    )
    q_subject = (
        None
        if query_subject_ids is None
        else _metadata_flat(
            query_subject_ids,
            batch=qb,
            tokens=qn,
            device=query.device,
            name="query_subject_ids",
        )
    )
    t_subject = (
        None
        if target_subject_ids is None
        else _metadata_flat(
            target_subject_ids,
            batch=tb,
            tokens=tn,
            device=query.device,
            name="target_subject_ids",
        )
    )
    q_time = (
        None
        if query_relative_time is None
        else _metadata_flat(
            query_relative_time,
            batch=qb,
            tokens=qn,
            device=query.device,
            name="query_relative_time",
        )
    )
    t_time = (
        None
        if target_relative_time is None
        else _metadata_flat(
            target_relative_time,
            batch=tb,
            tokens=tn,
            device=query.device,
            name="target_relative_time",
        )
    )

    q_position = torch.arange(qn, device=query.device).repeat(qb)
    t_position = torch.arange(tn, device=query.device).repeat(tb)
    same_trial = q_trial[:, None] == t_trial[None, :]
    positive_weights = torch.zeros_like(logits)
    for lag, weight in pairs:
        relation = (t_position[None, :] - q_position[:, None]) == int(lag)
        positive_weights = positive_weights + (
            relation & same_trial
        ).to(dtype=logits.dtype) * float(weight)
    positive_weights = positive_weights * qmask.reshape(-1, 1).to(logits.dtype)
    positive_weights = positive_weights * tmask.reshape(1, -1).to(logits.dtype)
    positive_present = positive_weights.sum(dim=-1) > 0.0

    if negative_mask is not None:
        candidate = _pair_mask_to_flat(
            negative_mask,
            query_batch=qb,
            query_tokens=qn,
            target_batch=tb,
            target_tokens=tn,
            device=query.device,
        )
    elif q_subject is not None and t_subject is not None and q_time is not None and t_time is not None:
        # Explicit same-subject/same-relative-time/other-trial interface.
        candidate = (
            (q_subject[:, None] == t_subject[None, :])
            & (q_trial[:, None] != t_trial[None, :])
            & (q_time[:, None] == t_time[None, :])
        )
    else:
        # In-batch negatives are the safe default when no metadata interface is
        # supplied.  Positives are always unioned below.
        candidate = torch.ones_like(logits, dtype=torch.bool)
    if additional_negative_mask is not None:
        if tuple(additional_negative_mask.shape) != tuple(candidate.shape):
            raise ValueError("additional_negative_mask must match flattened candidate axes")
        candidate = candidate | additional_negative_mask.to(
            device=query.device, dtype=torch.bool
        )
    candidate = candidate & tmask.reshape(1, -1)
    candidate = candidate | positive_weights.gt(0.0)
    candidate = candidate & qmask.reshape(-1, 1)
    valid_rows = positive_present & candidate.any(dim=-1)

    if not bool(valid_rows.any()):
        zero = query.sum() * 0.0
        return zero, {
            "valid_query_mask": valid_rows,
            "positive_count": positive_weights.gt(0.0).sum().to(logits.dtype),
            "candidate_count": candidate.sum().to(logits.dtype),
            "logits": logits,
        }
    masked_logits = logits.masked_fill(~candidate, -torch.inf)
    log_prob = F.log_softmax(masked_logits, dim=-1)
    normalized_targets = positive_weights / positive_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # ``0 * -inf`` would be NaN on non-candidate positions, so explicitly
    # replace their contribution by zero.
    contribution = torch.where(
        normalized_targets > 0.0, normalized_targets * log_prob, torch.zeros_like(log_prob)
    )
    row_loss = -contribution.sum(dim=-1)
    loss = row_loss[valid_rows].mean()
    return loss, {
        "valid_query_mask": valid_rows,
        "positive_count": positive_weights.gt(0.0).sum().to(logits.dtype),
        "candidate_count": candidate.sum().to(logits.dtype),
        "logits": logits,
    }


def _project_for_matching(
    features: torch.Tensor,
    projection: nn.Module | None,
    *,
    detach: bool,
) -> torch.Tensor:
    if projection is not None:
        features = projection(features)
    return features.detach() if detach else features


def lag_aware_continuous_matching_loss(
    query: torch.Tensor,
    target: torch.Tensor,
    query_projection: nn.Module | Sequence[float] | Mapping[int, float] | None = None,
    target_projection: nn.Module | float | None = None,
    positive_lag_weights: Mapping[int, float] | Sequence[float] | torch.Tensor = (1.0,),
    temperature: float = 0.07,
    *,
    positive_lags: Sequence[int] | None = None,
    query_valid_mask: torch.Tensor | None = None,
    target_valid_mask: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    eeg_valid_mask: torch.Tensor | None = None,
    fnirs_valid_mask: torch.Tensor | None = None,
    trial_ids: Any = None,
    subject_ids: Any = None,
    relative_time: Any = None,
    query_trial_ids: Any = None,
    target_trial_ids: Any = None,
    query_subject_ids: Any = None,
    target_subject_ids: Any = None,
    query_relative_time: Any = None,
    target_relative_time: Any = None,
    negative_mask: torch.Tensor | None = None,
    same_subject_other_trial_mask: torch.Tensor | None = None,
    deranged_target: torch.Tensor | None = None,
    negative_target: torch.Tensor | None = None,
    deranged_target_valid_mask: torch.Tensor | None = None,
    negative_target_valid_mask: torch.Tensor | None = None,
    deranged_target_negative_mask: torch.Tensor | None = None,
    deranged_query: torch.Tensor | None = None,
    deranged_query_valid_mask: torch.Tensor | None = None,
    deranged_query_negative_mask: torch.Tensor | None = None,
    target_encoder: nn.Module | None = None,
    bidirectional: bool = True,
    target_stop_gradient: bool = True,
    return_details: bool = False,
) -> torch.Tensor | Dict[str, Any]:
    """Compute mask-aware bidirectional lag-mixture InfoNCE.

    Positive target positions satisfy ``target_position = query_position +
    lag`` and are weighted by ``positive_lag_weights``.  If metadata are
    supplied, the explicit negative interface is same subject + same relative
    time + different trial; ``negative_mask`` can instead provide any exact
    candidate relation. An explicit ``deranged_target`` is appended as a
    guaranteed negative bank; ``deranged_target_negative_mask`` and its reverse
    counterpart can restrict that bank to registered same-time donor pairs.

    The target branch is detached by default (or can be produced by a frozen
    ``target_encoder``), making the function compatible with a momentum target
    updated outside the loss.  Invalid query/target tokens are removed before
    projection, normalization, and softmax.
    """

    # Backward-compatible positional surface: callers may pass
    # ``(query, target, weights, temperature)`` without projection heads.
    if query_projection is not None and not isinstance(query_projection, nn.Module) and not callable(query_projection):
        default_weights = (
            isinstance(positive_lag_weights, (tuple, list))
            and tuple(float(value) for value in positive_lag_weights) == (1.0,)
        )
        if default_weights:
            positive_lag_weights = query_projection  # type: ignore[assignment]
        if target_projection is not None and not isinstance(target_projection, nn.Module) and not callable(target_projection):
            temperature = float(target_projection)
        query_projection = None
        target_projection = None

    if query.ndim != 3 or target.ndim != 3:
        raise ValueError("query and target must be [B,N,D]")
    q_batch, q_tokens, _ = query.shape
    t_batch, t_tokens, _ = target.shape
    if valid_mask is not None:
        if query_valid_mask is None:
            query_valid_mask = valid_mask
        if target_valid_mask is None:
            target_valid_mask = valid_mask
    if negative_target is not None:
        if deranged_target is not None:
            raise ValueError("provide only one of deranged_target and negative_target")
        deranged_target = negative_target
    if negative_target_valid_mask is not None:
        if deranged_target_valid_mask is not None:
            raise ValueError(
                "provide only one of deranged_target_valid_mask and negative_target_valid_mask"
            )
        deranged_target_valid_mask = negative_target_valid_mask
    if query_valid_mask is None:
        query_valid_mask = query_mask
    if target_valid_mask is None:
        target_valid_mask = target_mask
    if query_valid_mask is None:
        query_valid_mask = eeg_valid_mask
    if target_valid_mask is None:
        target_valid_mask = fnirs_valid_mask
    if query_valid_mask is None:
        query_valid_mask = torch.ones(
            q_batch, q_tokens, device=query.device, dtype=torch.bool
        )
    if target_valid_mask is None:
        target_valid_mask = torch.ones(
            t_batch, t_tokens, device=target.device, dtype=torch.bool
        )
    query_valid_mask = query_valid_mask.to(device=query.device, dtype=torch.bool)
    target_valid_mask = target_valid_mask.to(device=target.device, dtype=torch.bool)
    if query_valid_mask.shape != (q_batch, q_tokens):
        raise ValueError("query_valid_mask must match query [B,N]")
    if target_valid_mask.shape != (t_batch, t_tokens):
        raise ValueError("target_valid_mask must match target [B,N]")

    if trial_ids is not None:
        query_trial_ids, target_trial_ids = _split_metadata(trial_ids)
    if subject_ids is not None:
        query_subject_ids, target_subject_ids = _split_metadata(subject_ids)
    if relative_time is not None:
        query_relative_time, target_relative_time = _split_metadata(relative_time)
    if same_subject_other_trial_mask is not None and negative_mask is None:
        negative_mask = same_subject_other_trial_mask

    pairs = _lag_pairs(positive_lag_weights, positive_lags)

    if target_encoder is not None:
        if target_stop_gradient:
            with torch.no_grad():
                target = target_encoder(target)
        else:
            target = target_encoder(target)
        if target.ndim != 3:
            raise ValueError("target_encoder must return [B,N,D]")
        t_batch, t_tokens, _ = target.shape

    # Append explicit deranged examples as negatives while retaining original
    # target trial IDs as the only source of positives.
    forward_target = target
    forward_target_mask = target_valid_mask
    forward_target_trial_ids = target_trial_ids
    forward_target_subject_ids = target_subject_ids
    forward_target_relative_time = target_relative_time
    forward_negative_mask = negative_mask
    forward_additional_negative_mask = None
    if deranged_target is not None:
        if deranged_target.ndim != 3 or tuple(deranged_target.shape[1:]) != (t_tokens, target.shape[-1]):
            raise ValueError("deranged_target must match target [B,N,D] axes")
        forward_target = torch.cat((target, deranged_target), dim=0)
        if deranged_target_valid_mask is None:
            deranged_mask = torch.ones(
                deranged_target.shape[0], t_tokens,
                device=deranged_target.device,
                dtype=torch.bool,
            )
        else:
            deranged_mask = deranged_target_valid_mask.to(
                device=deranged_target.device, dtype=torch.bool
            )
            if tuple(deranged_mask.shape) != (deranged_target.shape[0], t_tokens):
                raise ValueError(
                    "deranged_target_valid_mask must be [B_deranged,N_target]"
                )
        forward_target_mask = torch.cat((target_valid_mask, deranged_mask), dim=0)
        base = (
            _pair_mask_to_flat(
                negative_mask,
                query_batch=q_batch,
                query_tokens=q_tokens,
                target_batch=t_batch,
                target_tokens=t_tokens,
                device=query.device,
            )
            if negative_mask is not None
            else torch.zeros(
                q_batch * q_tokens,
                t_batch * t_tokens,
                device=query.device,
                dtype=torch.bool,
            )
        )
        extra = (
            _pair_mask_to_flat(
                deranged_target_negative_mask,
                query_batch=q_batch,
                query_tokens=q_tokens,
                target_batch=deranged_target.shape[0],
                target_tokens=t_tokens,
                device=query.device,
            )
            if deranged_target_negative_mask is not None
            else torch.ones(
                q_batch * q_tokens,
                deranged_target.shape[0] * t_tokens,
                device=query.device,
                dtype=torch.bool,
            )
        )
        forward_negative_mask = torch.cat((base, extra), dim=1)
        forward_additional_negative_mask = None
        # Metadata are optional; when present, append unique trial IDs so the
        # explicit derangement remains a negative even under metadata masking.
        if forward_target_trial_ids is not None:
            base = _metadata_flat(
                forward_target_trial_ids,
                batch=t_batch,
                tokens=t_tokens,
                device=query.device,
                name="target_trial_ids",
            )
            extra = torch.arange(
                t_batch, t_batch + deranged_target.shape[0], device=query.device
            ).repeat_interleave(t_tokens)
            forward_target_trial_ids = torch.cat((base, extra)).reshape(
                t_batch + deranged_target.shape[0], t_tokens
            )
        if forward_target_subject_ids is not None:
            base = _metadata_flat(
                forward_target_subject_ids,
                batch=t_batch,
                tokens=t_tokens,
                device=query.device,
                name="target_subject_ids",
            )
            extra = torch.full(
                (deranged_target.shape[0] * t_tokens,), -1, device=query.device, dtype=base.dtype
            )
            forward_target_subject_ids = torch.cat((base, extra)).reshape(
                t_batch + deranged_target.shape[0], t_tokens
            )
        if forward_target_relative_time is not None:
            base = _metadata_flat(
                forward_target_relative_time,
                batch=t_batch,
                tokens=t_tokens,
                device=query.device,
                name="target_relative_time",
            )
            extra = torch.arange(
                deranged_target.shape[0] * t_tokens, device=query.device, dtype=base.dtype
            )
            forward_target_relative_time = torch.cat((base, extra)).reshape(
                t_batch + deranged_target.shape[0], t_tokens
            )

    q_projected = _project_for_matching(query, query_projection if isinstance(query_projection, nn.Module) else None, detach=False)
    t_projected = _project_for_matching(
        forward_target,
        target_projection if isinstance(target_projection, nn.Module) else None,
        detach=target_stop_gradient,
    )
    forward_loss, forward_details = _direction_matching_loss(
        q_projected,
        t_projected,
        query_valid_mask=query_valid_mask,
        target_valid_mask=forward_target_mask,
        pairs=pairs,
        temperature=float(temperature),
        query_trial_ids=query_trial_ids,
        target_trial_ids=forward_target_trial_ids,
        query_subject_ids=query_subject_ids,
        target_subject_ids=forward_target_subject_ids,
        query_relative_time=query_relative_time,
        target_relative_time=forward_target_relative_time,
        negative_mask=forward_negative_mask,
        additional_negative_mask=forward_additional_negative_mask,
    )

    losses = [forward_loss]
    details: Dict[str, Any] = {"forward": forward_details}
    if bidirectional:
        # Reverse direction uses the negated lag convention and transposes an
        # explicitly supplied pair mask.  A separately supplied deranged_query
        # keeps the negative bank modality-consistent when needed.
        reverse_pairs = [(-lag, weight) for lag, weight in pairs]
        reverse_negative = None
        if negative_mask is not None:
            reverse_negative = _pair_mask_to_flat(
                negative_mask,
                query_batch=q_batch,
                query_tokens=q_tokens,
                target_batch=t_batch,
                target_tokens=t_tokens,
                device=query.device,
            ).transpose(0, 1)
            reverse_negative = reverse_negative.reshape(t_batch, t_tokens, q_batch, q_tokens)
        reverse_target = query
        reverse_target_mask = query_valid_mask
        reverse_target_trial_ids = query_trial_ids
        reverse_target_subject_ids = query_subject_ids
        reverse_target_relative_time = query_relative_time
        reverse_deranged = deranged_query
        reverse_deranged_mask = None
        if reverse_deranged is not None:
            if reverse_deranged.ndim != 3 or tuple(reverse_deranged.shape[1:]) != (
                q_tokens,
                query.shape[-1],
            ):
                raise ValueError("deranged_query must match query [B,N,D] axes")
            reverse_target = torch.cat((query, reverse_deranged), dim=0)
            if deranged_query_valid_mask is None:
                reverse_deranged_mask = torch.ones(
                    reverse_deranged.shape[0], reverse_deranged.shape[1],
                    device=reverse_deranged.device, dtype=torch.bool
                )
            else:
                reverse_deranged_mask = deranged_query_valid_mask.to(
                    device=reverse_deranged.device, dtype=torch.bool
                )
                if tuple(reverse_deranged_mask.shape) != tuple(reverse_deranged.shape[:2]):
                    raise ValueError(
                        "deranged_query_valid_mask must be [B_deranged,N_query]"
                    )
            reverse_target_mask = torch.cat(
                (query_valid_mask, reverse_deranged_mask), dim=0
            )
            reverse_base = (
                _pair_mask_to_flat(
                    reverse_negative,
                    query_batch=t_batch,
                    query_tokens=t_tokens,
                    target_batch=q_batch,
                    target_tokens=q_tokens,
                    device=query.device,
                )
                if reverse_negative is not None
                else torch.zeros(
                    t_batch * t_tokens,
                    q_batch * q_tokens,
                    device=query.device,
                    dtype=torch.bool,
                )
            )
            reverse_extra = (
                _pair_mask_to_flat(
                    deranged_query_negative_mask,
                    query_batch=t_batch,
                    query_tokens=t_tokens,
                    target_batch=reverse_deranged.shape[0],
                    target_tokens=q_tokens,
                    device=query.device,
                )
                if deranged_query_negative_mask is not None
                else torch.ones(
                    t_batch * t_tokens,
                    reverse_deranged.shape[0] * q_tokens,
                    device=query.device,
                    dtype=torch.bool,
                )
            )
            reverse_negative = torch.cat((reverse_base, reverse_extra), dim=1)
            reverse_target_trial_ids = None
            reverse_target_subject_ids = None
            reverse_target_relative_time = None
        reverse_q = _project_for_matching(
            target,
            target_projection if isinstance(target_projection, nn.Module) else None,
            detach=False,
        )
        reverse_t = _project_for_matching(
            reverse_target,
            query_projection if isinstance(query_projection, nn.Module) else None,
            detach=target_stop_gradient,
        )
        reverse_loss, reverse_details = _direction_matching_loss(
            reverse_q,
            reverse_t,
            query_valid_mask=target_valid_mask,
            target_valid_mask=reverse_target_mask,
            pairs=reverse_pairs,
            temperature=float(temperature),
            query_trial_ids=target_trial_ids,
            target_trial_ids=reverse_target_trial_ids,
            query_subject_ids=target_subject_ids,
            target_subject_ids=reverse_target_subject_ids,
            query_relative_time=target_relative_time,
            target_relative_time=reverse_target_relative_time,
            negative_mask=reverse_negative,
        )
        losses.append(reverse_loss)
        details["reverse"] = reverse_details

    loss = torch.stack(losses).mean()
    if return_details:
        details["loss"] = loss
        return details
    return loss


class LagAwareContinuousMatchingLoss(nn.Module):
    """Mask-aware lag matching with optional learnable lag mixture.

    With ``learnable_lag_mixture=True``, each configured lag is evaluated as a
    separate (optionally bidirectional) loss and the scalar losses are combined
    with ``softmax(lag_mixture_logits)``.  This keeps the symmetric term intact
    while allowing the model to learn which physiologic lag is useful.
    """

    def __init__(
        self,
        *,
        positive_lag_weights: Mapping[int, float] | Sequence[float] = (1.0,),
        positive_lags: Sequence[int] | None = None,
        temperature: float = 0.07,
        bidirectional: bool = True,
        target_stop_gradient: bool = True,
        query_projection: nn.Module | None = None,
        target_projection: nn.Module | None = None,
        target_encoder: nn.Module | None = None,
        learnable_lag_mixture: bool = False,
        learnable_lags: bool | None = None,
        lag_mixture_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if learnable_lags is not None:
            learnable_lag_mixture = bool(learnable_lags)
        if lag_mixture_temperature <= 0.0:
            raise ValueError("lag_mixture_temperature must be positive")
        self.positive_lag_weights = positive_lag_weights
        self.positive_lags = positive_lags
        self.temperature = float(temperature)
        self.bidirectional = bool(bidirectional)
        self.target_stop_gradient = bool(target_stop_gradient)
        self.query_projection = query_projection
        self.target_projection = target_projection
        self.target_encoder = target_encoder
        self.learnable_lag_mixture = bool(learnable_lag_mixture)
        self.lag_mixture_temperature = float(lag_mixture_temperature)
        configured_pairs = _lag_pairs(positive_lag_weights, positive_lags)
        self.lag_values = tuple(lag for lag, _ in configured_pairs)
        initial_weights = torch.tensor(
            [weight for _, weight in configured_pairs], dtype=torch.float32
        )
        if self.learnable_lag_mixture:
            self.lag_mixture_logits = nn.Parameter(initial_weights.clamp_min(1e-8).log())
        else:
            self.register_buffer("lag_mixture_logits", initial_weights.clamp_min(1e-8).log())

    @property
    def lag_logits(self) -> torch.Tensor:
        """Compatibility alias for the learnable lag logits."""

        return self.lag_mixture_logits

    @property
    def lag_mixture_weights(self) -> torch.Tensor:
        return F.softmax(
            self.lag_mixture_logits / self.lag_mixture_temperature, dim=0
        )

    @torch.no_grad()
    def update_momentum_target(self, online_encoder: nn.Module, momentum: float = 0.99) -> None:
        """EMA-update an optional target encoder outside the loss graph."""

        if self.target_encoder is None:
            raise RuntimeError("no target_encoder was supplied")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        target_params = dict(self.target_encoder.named_parameters())
        online_params = dict(online_encoder.named_parameters())
        if target_params.keys() != online_params.keys():
            raise ValueError("online and target encoders must have matching parameter names")
        for name, parameter in target_params.items():
            parameter.mul_(float(momentum)).add_(online_params[name], alpha=1.0 - float(momentum))
        target_buffers = dict(self.target_encoder.named_buffers())
        online_buffers = dict(online_encoder.named_buffers())
        for name, buffer in target_buffers.items():
            if name in online_buffers and buffer.shape == online_buffers[name].shape:
                buffer.copy_(online_buffers[name])

    def forward(self, query: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor | Dict[str, Any]:
        if not self.learnable_lag_mixture:
            return lag_aware_continuous_matching_loss(
                query,
                target,
                query_projection=self.query_projection,
                target_projection=self.target_projection,
                positive_lag_weights=self.positive_lag_weights,
                temperature=self.temperature,
                positive_lags=self.positive_lags,
                target_encoder=self.target_encoder,
                bidirectional=self.bidirectional,
                target_stop_gradient=self.target_stop_gradient,
                **kwargs,
            )

        return_details = bool(kwargs.pop("return_details", False))
        per_lag: list[torch.Tensor] = []
        per_lag_details: list[Dict[str, Any]] = []
        for lag in self.lag_values:
            result = lag_aware_continuous_matching_loss(
                query,
                target,
                query_projection=self.query_projection,
                target_projection=self.target_projection,
                positive_lag_weights={lag: 1.0},
                temperature=self.temperature,
                target_encoder=self.target_encoder,
                bidirectional=self.bidirectional,
                target_stop_gradient=self.target_stop_gradient,
                return_details=return_details,
                **kwargs,
            )
            if return_details:
                assert isinstance(result, dict)
                per_lag.append(result["loss"])
                per_lag_details.append(result)
            else:
                assert torch.is_tensor(result)
                per_lag.append(result)
        weights = self.lag_mixture_weights
        loss = torch.stack(per_lag).mul(weights).sum()
        if return_details:
            return {
                "loss": loss,
                "lag_values": self.lag_values,
                "lag_weights": weights,
                "per_lag": per_lag_details,
            }
        return loss


# ---------------------------------------------------------------------------
# LC-SPVQ model
# ---------------------------------------------------------------------------


class LCSPVQModel(nn.Module):
    """Four-encoder local-causal shared/private EEG-fNIRS LC-SPVQ core."""

    architecture_name = "lag_conditioned_shared_private_vq_v1"
    shared_dim_default = 64
    codebook_size_default = 16
    token_temporal_scope = "shared_local_causal_private_full_window"

    def __init__(
        self,
        *,
        eeg_channels: int = 6,
        fnirs_channels: int = 2,
        eeg_patch_samples: int = 400,
        fnirs_patch_samples: int = 20,
        num_tokens: int = 10,
        shared_dim: int = 64,
        eeg_private_dim: int = 64,
        fnirs_private_dim: int = 32,
        codebook_size: int = 16,
        eeg_shared_history_patches: int = 2,
        fnirs_shared_history_patches: int = 3,
        encoder_depth: int = 1,
        encoder_num_heads: int = 4,
        encoder_feedforward_dim: int | None = None,
        private_encoder_depth: int | None = None,
        private_encoder_num_heads: int | None = None,
        private_encoder_feedforward_dim: int | None = None,
        dropout: float = 0.0,
        projection_dim: int = 64,
        coupling_rank: int = 8,
        allowed_lags: Sequence[int] = (0, 1, 2, 3, 4, 5),
        coupling_allowed_lags: Sequence[int] | None = None,
        coupling_max_lag: int | None = None,
        num_classes: int = 2,
        private_classifier_hidden_dim: int | None = None,
        shared_marginal_classifier_hidden_dim: int | None = None,
        native_decoder_hidden_dim: int | None = None,
        raw_decoders: Mapping[str, nn.Module] | None = None,
        native_feature_decoders: Mapping[str, nn.Module] | None = None,
        native_target_decoders: Mapping[str, nn.Module] | None = None,
        eeg_native_feature_dim: int | None = None,
        fnirs_native_feature_dim: int | None = None,
        quantizer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if shared_dim <= 0 or codebook_size <= 0:
            raise ValueError("shared_dim and codebook_size must be positive")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if int(eeg_shared_history_patches) not in (1, 2):
            raise ValueError(
                "EEG shared history must be 1 or 2 patches (current plus at most one past)"
            )
        if int(fnirs_shared_history_patches) not in (2, 3):
            raise ValueError("fNIRS shared history must be configurable to 2 or 3 patches")
        self.eeg_channels = int(eeg_channels)
        self.fnirs_channels = int(fnirs_channels)
        self.num_tokens = int(num_tokens)
        self.shared_dim = int(shared_dim)
        self.eeg_private_dim = int(eeg_private_dim)
        self.fnirs_private_dim = int(fnirs_private_dim)
        self.codebook_size = int(codebook_size)
        self.eeg_patch_samples = int(eeg_patch_samples)
        self.fnirs_patch_samples = int(fnirs_patch_samples)
        self.eeg_shared_history_patches = int(eeg_shared_history_patches)
        self.fnirs_shared_history_patches = int(fnirs_shared_history_patches)

        shared_kwargs = {
            "num_tokens": self.num_tokens,
            "latent_dim": self.shared_dim,
            "depth": int(encoder_depth),
            "num_heads": int(encoder_num_heads),
            "feedforward_dim": encoder_feedforward_dim,
            "dropout": float(dropout),
        }
        # Four modules are constructed independently; no encoder or parameter
        # object is shared across modality/branch boundaries.
        self.eeg_shared_encoder = LocalCausalPatchEncoder(
            input_channels=self.eeg_channels,
            patch_samples=self.eeg_patch_samples,
            history_patches=self.eeg_shared_history_patches,
            **shared_kwargs,
        )
        self.fnirs_shared_encoder = LocalCausalPatchEncoder(
            input_channels=self.fnirs_channels,
            patch_samples=self.fnirs_patch_samples,
            history_patches=self.fnirs_shared_history_patches,
            **shared_kwargs,
        )
        private_depth = int(private_encoder_depth or encoder_depth)
        private_heads = int(private_encoder_num_heads or encoder_num_heads)
        private_ff = private_encoder_feedforward_dim or encoder_feedforward_dim
        private_kwargs = {
            "num_tokens": self.num_tokens,
            "depth": private_depth,
            "num_heads": private_heads,
            "feedforward_dim": private_ff,
            "dropout": float(dropout),
        }
        self.eeg_private_encoder = FullWindowPatchEncoder(
            input_channels=self.eeg_channels,
            patch_samples=self.eeg_patch_samples,
            latent_dim=self.eeg_private_dim,
            **private_kwargs,
        )
        self.fnirs_private_encoder = FullWindowPatchEncoder(
            input_channels=self.fnirs_channels,
            patch_samples=self.fnirs_patch_samples,
            latent_dim=self.fnirs_private_dim,
            **private_kwargs,
        )

        vq_kwargs = dict(quantizer_kwargs or {})
        supplied_codebook_size = int(vq_kwargs.pop("codebook_size", self.codebook_size))
        supplied_embedding_dim = int(vq_kwargs.pop("embedding_dim", self.shared_dim))
        if supplied_codebook_size != self.codebook_size:
            raise ValueError("quantizer codebook_size conflicts with model codebook_size")
        if supplied_embedding_dim != self.shared_dim:
            raise ValueError("quantizer embedding_dim must equal shared_dim")
        self.eeg_quantizer = EMAVectorQuantizer(
            codebook_size=self.codebook_size,
            embedding_dim=self.shared_dim,
            **vq_kwargs,
        )
        self.fnirs_quantizer = EMAVectorQuantizer(
            codebook_size=self.codebook_size,
            embedding_dim=self.shared_dim,
            **vq_kwargs,
        )

        self.eeg_shared_projection_head = ContinuousProjectionHead(
            self.shared_dim, int(projection_dim)
        )
        self.fnirs_shared_projection_head = ContinuousProjectionHead(
            self.shared_dim, int(projection_dim)
        )
        # Common shorter names are properties of the same independently
        # parameterized heads, not shared modules.
        self.eeg_projection_head = self.eeg_shared_projection_head
        self.fnirs_projection_head = self.fnirs_shared_projection_head

        selected_allowed_lags = (
            tuple(coupling_allowed_lags)
            if coupling_allowed_lags is not None
            else tuple(allowed_lags)
        )
        self.allowed_lags = tuple(int(lag) for lag in selected_allowed_lags)
        self.coupling_head = LowRankLagCouplingHead(
            input_dim=self.codebook_size,
            rank=int(coupling_rank),
            allowed_lags=self.allowed_lags,
            max_lag=coupling_max_lag,
            num_classes=int(num_classes),
        )
        self.allowed_lags = self.coupling_head.allowed_lags
        self.private_classifier = PrivatePooledClassifier(
            eeg_private_dim=self.eeg_private_dim,
            fnirs_private_dim=self.fnirs_private_dim,
            num_classes=int(num_classes),
            hidden_dim=private_classifier_hidden_dim,
        )
        self.shared_marginal_classifier = SharedMarginalClassifier(
            codebook_size=self.codebook_size,
            num_classes=int(num_classes),
            hidden_dim=shared_marginal_classifier_hidden_dim,
        )
        self.num_classes = int(num_classes)

        if raw_decoders is not None and native_feature_decoders is not None:
            raise ValueError(
                "raw_decoders and native_feature_decoders are aliases; provide one"
            )
        supplied_decoders = dict(
            raw_decoders if raw_decoders is not None else native_feature_decoders or {}
        )
        decoders = nn.ModuleDict()
        for modality in ("eeg", "fnirs"):
            if modality in supplied_decoders:
                decoder = supplied_decoders[modality]
                if not isinstance(decoder, nn.Module):
                    raise TypeError("native feature decoders must be nn.Module instances")
                decoders[modality] = decoder
            elif modality == "eeg":
                decoders[modality] = NativeFeatureDecoder(
                    shared_dim=self.shared_dim,
                    private_dim=self.eeg_private_dim,
                    output_channels=self.eeg_channels,
                    patch_samples=self.eeg_patch_samples,
                    hidden_dim=native_decoder_hidden_dim,
                )
            else:
                decoders[modality] = NativeFeatureDecoder(
                    shared_dim=self.shared_dim,
                    private_dim=self.fnirs_private_dim,
                    output_channels=self.fnirs_channels,
                    patch_samples=self.fnirs_patch_samples,
                    hidden_dim=native_decoder_hidden_dim,
                )
        # ``native_feature_decoders`` was the historical raw surface.  Keep
        # it as an exact compatibility alias while making the semantics
        # explicit for new callers.
        self.raw_decoders = decoders
        self.native_feature_decoders = self.raw_decoders

        target_decoders = nn.ModuleDict()
        supplied_target_decoders = dict(native_target_decoders or {})
        eeg_target_dim = int(
            self.eeg_channels if eeg_native_feature_dim is None else eeg_native_feature_dim
        )
        fnirs_target_dim = int(
            self.fnirs_channels
            if fnirs_native_feature_dim is None
            else fnirs_native_feature_dim
        )
        if eeg_target_dim <= 0 or fnirs_target_dim <= 0:
            raise ValueError("native feature dimensions must be positive")
        for modality in ("eeg", "fnirs"):
            if modality in supplied_target_decoders:
                decoder = supplied_target_decoders[modality]
                if not isinstance(decoder, nn.Module):
                    raise TypeError("native target decoders must be nn.Module instances")
                target_decoders[modality] = decoder
            else:
                target_decoders[modality] = NativeTargetFeatureDecoder(
                    shared_dim=self.shared_dim,
                    native_feature_dim=(
                        eeg_target_dim if modality == "eeg" else fnirs_target_dim
                    ),
                    hidden_dim=native_decoder_hidden_dim,
                )
        self.native_target_decoders = target_decoders
        self.eeg_native_feature_dim = eeg_target_dim
        self.fnirs_native_feature_dim = fnirs_target_dim

    @property
    def eeg_shared_vq(self) -> EMAVectorQuantizer:
        return self.eeg_quantizer

    @property
    def fnirs_shared_vq(self) -> EMAVectorQuantizer:
        return self.fnirs_quantizer

    def set_quantization_strength(self, strength: float) -> None:
        self.eeg_quantizer.set_quantization_strength(strength)
        self.fnirs_quantizer.set_quantization_strength(strength)

    def get_quantization_strength(self) -> float:
        eeg = self.eeg_quantizer.get_quantization_strength()
        fnirs = self.fnirs_quantizer.get_quantization_strength()
        if eeg != fnirs:
            raise RuntimeError("EEG and fNIRS quantization strengths diverged")
        return eeg

    def set_posterior_temperature(self, temperature: float) -> None:
        """Set the common posterior temperature on both independent VQs."""

        if float(temperature) <= 0.0:
            raise ValueError("posterior temperature must be positive")
        self.eeg_quantizer.temperature = float(temperature)
        self.fnirs_quantizer.temperature = float(temperature)

    def get_posterior_temperature(self) -> float:
        eeg = float(self.eeg_quantizer.temperature)
        fnirs = float(self.fnirs_quantizer.temperature)
        if eeg != fnirs:
            raise RuntimeError("EEG and fNIRS posterior temperatures diverged")
        return eeg

    @property
    def posterior_temperature(self) -> float:
        return self.get_posterior_temperature()

    # A short alias is convenient for schedulers while the explicit name keeps
    # the distinction from EMA quantization strength auditable.
    set_vq_temperature = set_posterior_temperature
    set_temperature = set_posterior_temperature

    def register_raw_decoder(self, modality: str, decoder: nn.Module) -> None:
        if modality not in {"eeg", "fnirs"}:
            raise ValueError("modality must be 'eeg' or 'fnirs'")
        if not isinstance(decoder, nn.Module):
            raise TypeError("decoder must be an nn.Module")
        self.raw_decoders[modality] = decoder

    def register_native_feature_decoder(self, modality: str, decoder: nn.Module) -> None:
        """Compatibility alias for the historical raw decoder hook."""

        self.register_raw_decoder(modality, decoder)

    def register_native_target_decoder(self, modality: str, decoder: nn.Module) -> None:
        if modality not in {"eeg", "fnirs"}:
            raise ValueError("modality must be 'eeg' or 'fnirs'")
        if not isinstance(decoder, nn.Module):
            raise TypeError("decoder must be an nn.Module")
        self.native_target_decoders[modality] = decoder

    @staticmethod
    def _decoder_mask(
        values: torch.Tensor,
        token_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if token_valid_mask is None:
            return torch.ones(
                values.shape[0], values.shape[1], device=values.device, dtype=torch.bool
            )
        token_valid_mask = token_valid_mask.to(device=values.device, dtype=torch.bool)
        if token_valid_mask.shape != values.shape[:2]:
            raise ValueError("token_valid_mask must match decoder token axes")
        return token_valid_mask

    def decode_raw(
        self,
        modality: str,
        shared: torch.Tensor,
        private: torch.Tensor,
        *,
        token_valid_mask: torch.Tensor | None = None,
        isolate_shared_gradient: bool = True,
    ) -> torch.Tensor:
        """Decode raw/native waveform features from detached shared + private."""

        if modality not in {"eeg", "fnirs"}:
            raise ValueError(f"unsupported modality {modality!r}")
        token_valid_mask = self._decoder_mask(shared, token_valid_mask)
        admitted_shared = shared.detach() if isolate_shared_gradient else shared
        decoder = self.raw_decoders[modality]
        try:
            decoded = decoder(admitted_shared, private)
        except TypeError:
            decoded = decoder(admitted_shared)
        if decoded.ndim >= 3 and decoded.shape[:2] == shared.shape[:2]:
            return decoded.masked_fill(~token_valid_mask.unsqueeze(-1), 0.0)
        if decoded.ndim == 3:
            if decoded.shape[-1] % shared.shape[1] != 0:
                raise ValueError("raw decoder output length must be divisible by token count")
            patch_points = decoded.shape[-1] // shared.shape[1]
            return decoded.masked_fill(
                ~token_valid_mask.repeat_interleave(patch_points, dim=1).unsqueeze(1),
                0.0,
            )
        raise ValueError("raw decoder must return a rank-3 tensor")

    def decode_native_target_features(
        self,
        modality: str,
        pre_vq_shared: torch.Tensor,
        *,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode token-major native targets from *pre-VQ shared only*."""

        if modality not in {"eeg", "fnirs"}:
            raise ValueError(f"unsupported modality {modality!r}")
        token_valid_mask = self._decoder_mask(pre_vq_shared, token_valid_mask)
        decoded = self.native_target_decoders[modality](pre_vq_shared)
        if decoded.ndim != 3 or decoded.shape[:2] != pre_vq_shared.shape[:2]:
            raise ValueError("native target decoder must return [B,N,F]")
        return decoded.masked_fill(~token_valid_mask.unsqueeze(-1), 0.0)

    def decode_native_features(
        self,
        modality: str,
        shared: torch.Tensor,
        private: torch.Tensor | None = None,
        *,
        token_valid_mask: torch.Tensor | None = None,
        isolate_shared_gradient: bool = True,
    ) -> torch.Tensor:
        """Flexible compatibility surface.

        Supplying ``private`` selects the historical raw decoder.  Omitting it
        selects the new pre-VQ native-target decoder.
        """

        if private is None:
            return self.decode_native_target_features(
                modality, shared, token_valid_mask=token_valid_mask
            )
        return self.decode_raw(
            modality,
            shared,
            private,
            token_valid_mask=token_valid_mask,
            isolate_shared_gradient=isolate_shared_gradient,
        )

    # Alias for downstream integrations; the optional private argument keeps
    # old raw-hook calls working while enabling native-target calls.
    decode_native = decode_native_features

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        mask = self.eeg_shared_encoder.resolve_valid_mask(eeg, token_valid_mask)
        pre = self.eeg_shared_encoder(eeg, mask)
        q = self.eeg_quantizer(pre, valid_mask=mask)
        private = self.eeg_private_encoder(eeg, mask)
        return self._branch_surface(pre, private, q, mask)

    def encode_fnirs(
        self,
        fnirs: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        mask = self.fnirs_shared_encoder.resolve_valid_mask(fnirs, token_valid_mask)
        pre = self.fnirs_shared_encoder(fnirs, mask)
        q = self.fnirs_quantizer(pre, valid_mask=mask)
        private = self.fnirs_private_encoder(fnirs, mask)
        return self._branch_surface(pre, private, q, mask)

    @staticmethod
    def _branch_surface(
        pre: torch.Tensor,
        private: torch.Tensor,
        quantizer: QuantizerOutput,
        valid_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        return {
            "pre_vq_latent": pre,
            "pre_vq": pre,
            "latent": pre,
            "private": private,
            "posterior": quantizer.posterior,
            "hard_ids": quantizer.hard_ids,
            "expected_embedding": quantizer.expected_embedding,
            "expected": quantizer.expected_embedding,
            "annealed_embedding": quantizer.annealed_quantized,
            "annealed": quantizer.annealed_quantized,
            "quantized_embedding": quantizer.quantized,
            "quantized": quantizer.quantized,
            "logits": quantizer.logits,
            "codebook": quantizer.codebook,
            "commitment_loss": quantizer.commitment_loss,
            "health": quantizer.health,
            "quantizer": quantizer,
            "valid_mask": valid_mask,
        }

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        eeg_token_valid_mask: torch.Tensor | None = None,
        fnirs_token_valid_mask: torch.Tensor | None = None,
        *,
        token_valid_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> Dict[str, Any]:
        if token_valid_masks is not None:
            eeg_token_valid_mask = token_valid_masks.get("eeg", eeg_token_valid_mask)
            fnirs_token_valid_mask = token_valid_masks.get("fnirs", fnirs_token_valid_mask)
        eeg_mask = self.eeg_shared_encoder.resolve_valid_mask(eeg, eeg_token_valid_mask)
        fnirs_mask = self.fnirs_shared_encoder.resolve_valid_mask(fnirs, fnirs_token_valid_mask)

        eeg_pre = self.eeg_shared_encoder(eeg, eeg_mask)
        fnirs_pre = self.fnirs_shared_encoder(fnirs, fnirs_mask)
        eeg_vq = self.eeg_quantizer(eeg_pre, valid_mask=eeg_mask)
        fnirs_vq = self.fnirs_quantizer(fnirs_pre, valid_mask=fnirs_mask)
        eeg_private = self.eeg_private_encoder(eeg, eeg_mask)
        fnirs_private = self.fnirs_private_encoder(fnirs, fnirs_mask)
        eeg_shared = eeg_vq.expected_embedding
        fnirs_shared = fnirs_vq.expected_embedding

        # The warm-start matching objective is intentionally continuous and
        # therefore projects the pre-VQ latents. The coupling head itself
        # deliberately consumes the K=16 posterior, preserving code-occupancy
        # semantics instead of feeding expected D64 embeddings into the
        # pairwise interaction.
        eeg_projection = self.eeg_shared_projection_head(eeg_pre)
        fnirs_projection = self.fnirs_shared_projection_head(fnirs_pre)
        coupling_pair_logits, coupling_pair_mask = self.coupling_head(
            eeg_vq.posterior,
            fnirs_vq.posterior,
            eeg_valid_mask=eeg_mask,
            fnirs_valid_mask=fnirs_mask,
            return_mask=True,
        )
        coupling_only_logits = _masked_pair_mean(
            coupling_pair_logits, coupling_pair_mask
        )
        shared_marginal_only_logits, shared_marginal_pooled = self.shared_marginal_classifier(
            eeg_vq.posterior,
            fnirs_vq.posterior,
            eeg_valid_mask=eeg_mask,
            fnirs_valid_mask=fnirs_mask,
            return_pooled=True,
        )
        private_only_logits, private_pooled = self.private_classifier(
            eeg_private,
            fnirs_private,
            eeg_valid_mask=eeg_mask,
            fnirs_valid_mask=fnirs_mask,
            return_pooled=True,
        )
        # The registered primary head is coupling + private. Shared marginals
        # are an exported diagnostic ablation, not an additional task shortcut.
        combined_logits = coupling_only_logits + private_only_logits

        # Historical native-feature hooks are raw decoders and isolate shared
        # gradients.  New native target hooks consume only pre-VQ shared and
        # intentionally retain its gradient path.
        eeg_raw = self.decode_raw(
            "eeg", eeg_vq.annealed_quantized, eeg_private, token_valid_mask=eeg_mask
        )
        fnirs_raw = self.decode_raw(
            "fnirs", fnirs_vq.annealed_quantized, fnirs_private, token_valid_mask=fnirs_mask
        )
        eeg_native_target = self.decode_native_target_features(
            "eeg", eeg_pre, token_valid_mask=eeg_mask
        )
        fnirs_native_target = self.decode_native_target_features(
            "fnirs", fnirs_pre, token_valid_mask=fnirs_mask
        )
        eeg_surface = self._branch_surface(eeg_pre, eeg_private, eeg_vq, eeg_mask)
        fnirs_surface = self._branch_surface(fnirs_pre, fnirs_private, fnirs_vq, fnirs_mask)
        output: Dict[str, Any] = {
            "eeg": eeg_surface,
            "fnirs": fnirs_surface,
            "eeg_shared_pre_vq": eeg_pre,
            "fnirs_shared_pre_vq": fnirs_pre,
            "eeg_pre_vq": eeg_pre,
            "fnirs_pre_vq": fnirs_pre,
            "eeg_shared_latent": eeg_pre,
            "fnirs_shared_latent": fnirs_pre,
            "eeg_shared": eeg_shared,
            "fnirs_shared": fnirs_shared,
            "eeg_shared_expected": eeg_vq.expected_embedding,
            "fnirs_shared_expected": fnirs_vq.expected_embedding,
            "eeg_expected_embedding": eeg_vq.expected_embedding,
            "fnirs_expected_embedding": fnirs_vq.expected_embedding,
            "eeg_shared_annealed": eeg_vq.annealed_quantized,
            "fnirs_shared_annealed": fnirs_vq.annealed_quantized,
            "eeg_annealed_embedding": eeg_vq.annealed_quantized,
            "fnirs_annealed_embedding": fnirs_vq.annealed_quantized,
            "eeg_shared_quantized": eeg_vq.quantized,
            "fnirs_shared_quantized": fnirs_vq.quantized,
            "eeg_quantized_embedding": eeg_vq.quantized,
            "fnirs_quantized_embedding": fnirs_vq.quantized,
            "eeg_shared_posterior": eeg_vq.posterior,
            "fnirs_shared_posterior": fnirs_vq.posterior,
            "eeg_shared_hard_ids": eeg_vq.hard_ids,
            "fnirs_shared_hard_ids": fnirs_vq.hard_ids,
            "eeg_quantizer": eeg_vq,
            "fnirs_quantizer": fnirs_vq,
            "eeg_private": eeg_private,
            "fnirs_private": fnirs_private,
            "eeg_shared_projection": eeg_projection,
            "fnirs_shared_projection": fnirs_projection,
            "eeg_projection": eeg_projection,
            "fnirs_projection": fnirs_projection,
            "coupling_pair_logits": coupling_pair_logits,
            "coupling_logits": coupling_pair_logits,
            "coupling_pair_valid_mask": coupling_pair_mask,
            "coupling_only_logits": coupling_only_logits,
            "coupling_sample_logits": coupling_only_logits,
            "coupling_pooled_logits": coupling_only_logits,
            "shared_marginal_pooled": shared_marginal_pooled,
            "shared_marginal_logits": shared_marginal_only_logits,
            "shared_marginal_only_logits": shared_marginal_only_logits,
            "private_pooled": private_pooled,
            "private_logits": private_only_logits,
            "private_pooled_logits": private_only_logits,
            "private_only_logits": private_only_logits,
            "combined_logits": combined_logits,
            "eeg_raw": eeg_raw,
            "fnirs_raw": fnirs_raw,
            # Compatibility aliases for the old raw decoder output names.
            "eeg_native_features": eeg_raw,
            "fnirs_native_features": fnirs_raw,
            "eeg_native": eeg_raw,
            "fnirs_native": fnirs_raw,
            "eeg_native_target_prediction": eeg_native_target,
            "fnirs_native_target_prediction": fnirs_native_target,
            "eeg_token_valid_mask": eeg_mask,
            "fnirs_token_valid_mask": fnirs_mask,
            "commitment_loss": eeg_vq.commitment_loss + fnirs_vq.commitment_loss,
        }
        return output


# Names likely to be used by experiment code; all refer to the same reusable
# implementation and do not create additional parameterized classes.
LagConditionedSharedPrivateVQ = LCSPVQModel
LagConditionedSharedPrivateVQModel = LCSPVQModel
LCSPVQ = LCSPVQModel


__all__ = [
    "CausalLocalPatchEncoder",
    "ContinuousProjectionHead",
    "FullWindowPatchEncoder",
    "LagAwareContinuousMatchingLoss",
    "LagConditionedSharedPrivateVQ",
    "LagConditionedSharedPrivateVQModel",
    "LCSPVQ",
    "LCSPVQModel",
    "LocalCausalPatchEncoder",
    "LowRankLagCouplingHead",
    "NativeFeatureDecoder",
    "NativeTargetFeatureDecoder",
    "PrivateFullWindowEncoder",
    "PrivatePooledClassifier",
    "RawFeatureDecoder",
    "SharedMarginalClassifier",
    "lag_aware_continuous_matching_loss",
]
