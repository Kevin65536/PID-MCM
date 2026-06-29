"""Lag-aware source-branch fusion and self-supervised alignment losses."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _directional_attention_mask(
    tokens: int,
    max_lag_tokens: int,
    direction: str,
    lag_bias: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    query = torch.arange(tokens, device=device).view(-1, 1)
    key = torch.arange(tokens, device=device).view(1, -1)
    if direction == "eeg_to_fnirs":
        lag = query - key
    elif direction == "fnirs_to_eeg":
        lag = key - query
    else:
        raise ValueError(f"Unsupported fusion direction: {direction!r}")
    valid = (lag >= 0) & (lag <= max_lag_tokens)
    mask = torch.full((tokens, tokens), float("-inf"), device=device, dtype=dtype)
    if lag_bias is None:
        mask.masked_fill_(valid, 0.0)
    else:
        selected = lag_bias.to(device=device, dtype=dtype)[lag.clamp(0, max_lag_tokens)]
        mask = torch.where(valid, selected, mask)
    return mask


class LagAwareCrossAttentionBlock(nn.Module):
    """Pre-norm cross attention with a directional lag window."""

    def __init__(self, dim: int, num_heads: int, max_lag_tokens: int, dropout: float, relative_lag_bias: bool):
        super().__init__()
        self.max_lag_tokens = max(int(max_lag_tokens), 0)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.relative_lag_bias = (
            nn.Parameter(torch.zeros(self.max_lag_tokens + 1)) if relative_lag_bias else None
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor, direction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = _directional_attention_mask(
            query.shape[1],
            self.max_lag_tokens,
            direction,
            self.relative_lag_bias,
            device=query.device,
            dtype=query.dtype,
        )
        update, weights = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=False,
        )
        fused = query + self.dropout(update)
        fused = fused + self.ffn(self.ffn_norm(fused))
        return fused, weights


class LagAwareCrossModalFusion(nn.Module):
    """Full source-branch cross attention with causal or bidirectional exchange."""

    def __init__(
        self,
        eeg_dim: int,
        fnirs_dim: int,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        max_lag_tokens: int = 5,
        relative_lag_bias: bool = True,
        dropout: float = 0.1,
        mode: str = "causal_cross_attention",
    ) -> None:
        super().__init__()
        if mode not in {"causal_cross_attention", "bidirectional_cross_attention"}:
            raise ValueError(f"Unsupported cross-modal fusion mode: {mode!r}")
        if embed_dim % num_heads:
            raise ValueError("cross-modal fusion embed_dim must be divisible by num_heads")
        self.mode = mode
        self.max_lag_tokens = max(int(max_lag_tokens), 0)
        self.eeg_input = nn.Linear(eeg_dim, embed_dim) if eeg_dim != embed_dim else nn.Identity()
        self.fnirs_input = nn.Linear(fnirs_dim, embed_dim) if fnirs_dim != embed_dim else nn.Identity()
        self.eeg_output = nn.Linear(embed_dim, eeg_dim) if eeg_dim != embed_dim else nn.Identity()
        self.fnirs_output = nn.Linear(embed_dim, fnirs_dim) if fnirs_dim != embed_dim else nn.Identity()
        self.fnirs_blocks = nn.ModuleList([
            LagAwareCrossAttentionBlock(embed_dim, num_heads, self.max_lag_tokens, dropout, relative_lag_bias)
            for _ in range(max(int(depth), 1))
        ])
        self.eeg_blocks = nn.ModuleList([
            LagAwareCrossAttentionBlock(embed_dim, num_heads, self.max_lag_tokens, dropout, relative_lag_bias)
            for _ in range(max(int(depth), 1))
        ]) if mode == "bidirectional_cross_attention" else nn.ModuleList()

    def forward(self, eeg_source: torch.Tensor, fnirs_source: torch.Tensor) -> Dict[str, torch.Tensor]:
        eeg_state = self.eeg_input(eeg_source)
        fnirs_state = self.fnirs_input(fnirs_source)
        fnirs_weights = []
        eeg_weights = []
        for index, fnirs_block in enumerate(self.fnirs_blocks):
            previous_eeg = eeg_state
            previous_fnirs = fnirs_state
            fnirs_state, weights = fnirs_block(previous_fnirs, previous_eeg, "eeg_to_fnirs")
            fnirs_weights.append(weights)
            if self.mode == "bidirectional_cross_attention":
                eeg_state, weights = self.eeg_blocks[index](previous_eeg, previous_fnirs, "fnirs_to_eeg")
                eeg_weights.append(weights)
        return {
            "eeg_source": eeg_source if self.mode == "causal_cross_attention" else self.eeg_output(eeg_state),
            "fnirs_source": self.fnirs_output(fnirs_state),
            "fnirs_attention": torch.stack(fnirs_weights, dim=1),
            "eeg_attention": (
                torch.stack(eeg_weights, dim=1)
                if eeg_weights else eeg_source.new_zeros(eeg_source.shape[0], 0, 0, eeg_source.shape[1], eeg_source.shape[1])
            ),
        }


def lag_aware_temporal_nce(
    eeg_source: torch.Tensor,
    fnirs_source: torch.Tensor,
    eeg_projection: nn.Module,
    fnirs_projection: nn.Module,
    positive_lag_weights: Sequence[float],
    temperature: float,
    *,
    bidirectional: bool,
) -> torch.Tensor:
    """Within-window multi-positive InfoNCE; no cross-window identity shortcut."""
    eeg_query = F.normalize(eeg_projection(eeg_source), dim=-1)
    fnirs_key = F.normalize(fnirs_projection(fnirs_source), dim=-1).detach()
    logits = torch.matmul(eeg_query, fnirs_key.transpose(1, 2)) / max(float(temperature), 1e-6)
    tokens = logits.shape[-1]
    positions = torch.arange(tokens, device=logits.device)
    lag = positions.view(1, -1) - positions.view(-1, 1)
    lag_weights = logits.new_tensor(list(positive_lag_weights))
    valid = (lag >= 0) & (lag < lag_weights.numel())
    targets = torch.zeros_like(logits[0])
    targets[valid] = lag_weights[lag[valid]]
    valid_rows = targets.sum(dim=-1) > 0
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    forward_loss = -(targets[valid_rows] * F.log_softmax(logits[:, valid_rows, :], dim=-1)).sum(dim=-1).mean()
    if not bidirectional:
        return forward_loss
    reverse_targets = targets.transpose(0, 1)
    valid_rows = reverse_targets.sum(dim=-1) > 0
    reverse_targets = reverse_targets / reverse_targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    fnirs_query = F.normalize(fnirs_projection(fnirs_source), dim=-1)
    eeg_key = F.normalize(eeg_projection(eeg_source), dim=-1).detach()
    reverse_logits = torch.matmul(fnirs_query, eeg_key.transpose(1, 2)) / max(float(temperature), 1e-6)
    reverse_loss = -(
        reverse_targets[valid_rows] * F.log_softmax(reverse_logits[:, valid_rows, :], dim=-1)
    ).sum(dim=-1).mean()
    return 0.5 * (forward_loss + reverse_loss)


def masked_alignment_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    predicted_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine latent prediction and soft-code distillation on masked positions."""
    if not bool(mask.any()):
        zero = predicted.new_zeros(())
        return zero, zero
    latent = 1.0 - F.cosine_similarity(predicted[mask], target.detach()[mask], dim=-1)
    teacher = F.softmax(target_logits.detach()[mask] / max(float(temperature), 1e-6), dim=-1)
    student = F.log_softmax(predicted_logits[mask] / max(float(temperature), 1e-6), dim=-1)
    distillation = F.kl_div(student, teacher, reduction="batchmean")
    return latent.mean(), distillation


def attention_lag_statistics(attention: torch.Tensor, max_lag_tokens: int, direction: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized entropy and mass on physiologic lags 2/3."""
    if attention.numel() == 0:
        zero = attention.new_zeros(())
        return zero, zero
    mean_attention = attention.mean(dim=(1, 2))
    tokens = mean_attention.shape[-1]
    query = torch.arange(tokens, device=attention.device).view(-1, 1)
    key = torch.arange(tokens, device=attention.device).view(1, -1)
    lag = query - key if direction == "eeg_to_fnirs" else key - query
    valid = (lag >= 0) & (lag <= max_lag_tokens)
    entropy = -(mean_attention.clamp_min(1e-12).log() * mean_attention).masked_fill(~valid, 0.0).sum(dim=-1)
    valid_count = valid.sum(dim=-1).clamp_min(2).to(entropy.dtype)
    entropy = (entropy / valid_count.log()).mean()
    physiologic = ((lag == 2) | (lag == 3)) & valid
    physiologic_mass = mean_attention.masked_fill(~physiologic, 0.0).sum(dim=-1).mean()
    return entropy, physiologic_mass
