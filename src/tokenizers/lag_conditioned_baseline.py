"""Continuous B0 baseline for the LC-SPVQ development generation."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from src.tokenizers.continuous_shared_private import (
    RawPatchDecoder,
    TemporalPatchEncoder,
)


def masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool ``[B,T,D]`` tokens without admitting invalid positions."""

    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("masked token pooling requires [B,T,D] and [B,T]")
    admitted = mask.to(device=values.device, dtype=torch.bool).unsqueeze(-1)
    masked_values = torch.where(admitted, values, torch.zeros_like(values))
    if not bool(torch.isfinite(masked_values).all()):
        raise FloatingPointError("masked token pooling contains a non-finite admitted value")
    weights = admitted.to(dtype=values.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return masked_values.sum(dim=1) / denominator


class NativeFeatureDecoder(nn.Module):
    """Map one modality's continuous shared tokens to native coordinates."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("native decoder dimensions must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("native decoder expects [B,T,D]")
        return self.network(tokens)


class ContinuousAblationHead(nn.Module):
    """Export shared, private, and additive B0 task-logit contributions."""

    def __init__(
        self,
        *,
        shared_dim: int,
        eeg_private_dim: int,
        fnirs_private_dim: int,
        class_count: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if class_count <= 1:
            raise ValueError("class_count must exceed one")
        self.shared_head = nn.Sequential(
            nn.LayerNorm(2 * int(shared_dim)),
            nn.Linear(2 * int(shared_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(class_count)),
        )
        self.private_head = nn.Sequential(
            nn.LayerNorm(int(eeg_private_dim) + int(fnirs_private_dim)),
            nn.Linear(
                int(eeg_private_dim) + int(fnirs_private_dim), int(hidden_dim)
            ),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(class_count)),
        )

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
        shared = torch.cat(
            (
                masked_token_mean(eeg_shared, eeg_mask),
                masked_token_mean(fnirs_shared, fnirs_mask),
            ),
            dim=-1,
        )
        private = torch.cat(
            (
                masked_token_mean(eeg_private, eeg_mask),
                masked_token_mean(fnirs_private, fnirs_mask),
            ),
            dim=-1,
        )
        shared_logits = self.shared_head(shared)
        private_logits = self.private_head(private)
        return {
            "shared_marginal_only": shared_logits,
            "private_only": private_logits,
            "combined": shared_logits + private_logits,
        }


class B0ContinuousSharedPrivate(nn.Module):
    """Full-window continuous B0 with native shared and private raw objectives."""

    architecture_name = "lc_spvq_b0_continuous_v1"
    shared_temporal_scope = "bidirectional_full_window"
    private_temporal_scope = "bidirectional_full_window"
    vector_quantization = False

    def __init__(
        self,
        *,
        eeg_channels: int,
        fnirs_channels: int,
        eeg_native_dim: int,
        fnirs_native_dim: int,
        class_count: int,
        eeg_patch_samples: int = 400,
        fnirs_patch_samples: int = 20,
        num_tokens: int = 10,
        shared_dim: int = 64,
        eeg_private_dim: int = 64,
        fnirs_private_dim: int = 32,
        encoder_depth: int = 2,
        encoder_num_heads: int = 4,
        encoder_feedforward_dim: int = 256,
        native_decoder_hidden_dim: int = 128,
        raw_decoder_hidden_dim: int = 256,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.shared_dim = int(shared_dim)
        self.private_dims = {
            "eeg": int(eeg_private_dim),
            "fnirs": int(fnirs_private_dim),
        }
        common: dict[str, Any] = {
            "num_tokens": self.num_tokens,
            "depth": int(encoder_depth),
            "num_heads": int(encoder_num_heads),
            "feedforward_dim": int(encoder_feedforward_dim),
            "dropout": float(dropout),
        }
        self.eeg_shared_encoder = TemporalPatchEncoder(
            input_channels=int(eeg_channels),
            patch_samples=int(eeg_patch_samples),
            latent_dim=self.shared_dim,
            **common,
        )
        self.fnirs_shared_encoder = TemporalPatchEncoder(
            input_channels=int(fnirs_channels),
            patch_samples=int(fnirs_patch_samples),
            latent_dim=self.shared_dim,
            **common,
        )
        self.eeg_private_encoder = TemporalPatchEncoder(
            input_channels=int(eeg_channels),
            patch_samples=int(eeg_patch_samples),
            latent_dim=self.private_dims["eeg"],
            **common,
        )
        self.fnirs_private_encoder = TemporalPatchEncoder(
            input_channels=int(fnirs_channels),
            patch_samples=int(fnirs_patch_samples),
            latent_dim=self.private_dims["fnirs"],
            **common,
        )
        self.eeg_native_decoder = NativeFeatureDecoder(
            self.shared_dim, int(native_decoder_hidden_dim), int(eeg_native_dim)
        )
        self.fnirs_native_decoder = NativeFeatureDecoder(
            self.shared_dim, int(native_decoder_hidden_dim), int(fnirs_native_dim)
        )
        self.eeg_raw_decoder = RawPatchDecoder(
            shared_dim=self.shared_dim,
            private_dim=self.private_dims["eeg"],
            hidden_dim=int(raw_decoder_hidden_dim),
            output_channels=int(eeg_channels),
            patch_samples=int(eeg_patch_samples),
        )
        self.fnirs_raw_decoder = RawPatchDecoder(
            shared_dim=self.shared_dim,
            private_dim=self.private_dims["fnirs"],
            hidden_dim=int(raw_decoder_hidden_dim),
            output_channels=int(fnirs_channels),
            patch_samples=int(fnirs_patch_samples),
        )
        self.classifier = ContinuousAblationHead(
            shared_dim=self.shared_dim,
            eeg_private_dim=self.private_dims["eeg"],
            fnirs_private_dim=self.private_dims["fnirs"],
            class_count=int(class_count),
            hidden_dim=int(classifier_hidden_dim),
        )

    @staticmethod
    def _mask_raw(
        raw: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        patch_samples: int,
    ) -> torch.Tensor:
        point_mask = token_mask.repeat_interleave(int(patch_samples), dim=1)
        return raw.masked_fill(~point_mask.unsqueeze(1), 0.0)

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        eeg_token_valid_mask: torch.Tensor | None = None,
        fnirs_token_valid_mask: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]]:
        eeg_mask = self.eeg_shared_encoder.resolve_mask(eeg, eeg_token_valid_mask)
        fnirs_mask = self.fnirs_shared_encoder.resolve_mask(
            fnirs, fnirs_token_valid_mask
        )
        eeg_shared = self.eeg_shared_encoder(eeg, eeg_mask)
        fnirs_shared = self.fnirs_shared_encoder(fnirs, fnirs_mask)
        eeg_private = self.eeg_private_encoder(eeg, eeg_mask)
        fnirs_private = self.fnirs_private_encoder(fnirs, fnirs_mask)
        eeg_raw = self.eeg_raw_decoder(eeg_shared.detach(), eeg_private)
        fnirs_raw = self.fnirs_raw_decoder(fnirs_shared.detach(), fnirs_private)
        eeg_native = self.eeg_native_decoder(eeg_shared).masked_fill(
            ~eeg_mask.unsqueeze(-1), 0.0
        )
        fnirs_native = self.fnirs_native_decoder(fnirs_shared).masked_fill(
            ~fnirs_mask.unsqueeze(-1), 0.0
        )
        logits = self.classifier(
            eeg_shared=eeg_shared,
            fnirs_shared=fnirs_shared,
            eeg_private=eeg_private,
            fnirs_private=fnirs_private,
            eeg_mask=eeg_mask,
            fnirs_mask=fnirs_mask,
        )
        return {
            "eeg_shared": eeg_shared,
            "fnirs_shared": fnirs_shared,
            "eeg_private": eeg_private,
            "fnirs_private": fnirs_private,
            "eeg_native": eeg_native,
            "fnirs_native": fnirs_native,
            "eeg_raw": self._mask_raw(
                eeg_raw,
                eeg_mask,
                patch_samples=self.eeg_shared_encoder.patch_samples,
            ),
            "fnirs_raw": self._mask_raw(
                fnirs_raw,
                fnirs_mask,
                patch_samples=self.fnirs_shared_encoder.patch_samples,
            ),
            "eeg_token_valid_mask": eeg_mask,
            "fnirs_token_valid_mask": fnirs_mask,
            "logits": logits,
        }


__all__ = [
    "B0ContinuousSharedPrivate",
    "ContinuousAblationHead",
    "NativeFeatureDecoder",
    "masked_token_mean",
]
