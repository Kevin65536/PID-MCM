"""Continuous shared/private EEG-fNIRS model without vector quantization."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TemporalPatchEncoder(nn.Module):
    """Encode a complete window as non-overlapping contextual patch tokens."""

    token_temporal_scope = "bidirectional_full_window"

    def __init__(
        self,
        *,
        input_channels: int,
        patch_samples: int,
        num_tokens: int,
        latent_dim: int,
        depth: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or patch_samples <= 0 or num_tokens <= 0:
            raise ValueError("input, patch, and token dimensions must be positive")
        if latent_dim <= 0 or num_heads <= 0 or latent_dim % num_heads:
            raise ValueError("num_heads must divide a positive latent_dim")
        self.input_channels = int(input_channels)
        self.patch_samples = int(patch_samples)
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.expected_samples = self.patch_samples * self.num_tokens

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
            dim_feedforward=int(feedforward_dim),
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

    def resolve_mask(
        self, signal: torch.Tensor, token_valid_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if token_valid_mask is None:
            return torch.ones(
                signal.shape[0], self.num_tokens, dtype=torch.bool, device=signal.device
            )
        if tuple(token_valid_mask.shape) != (signal.shape[0], self.num_tokens):
            raise ValueError(
                f"token_valid_mask must be [B,{self.num_tokens}], got "
                f"{tuple(token_valid_mask.shape)}"
            )
        return token_valid_mask.to(device=signal.device, dtype=torch.bool)

    def forward(
        self, signal: torch.Tensor, token_valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        expected = (self.input_channels, self.expected_samples)
        if signal.ndim != 3 or tuple(signal.shape[1:]) != expected:
            raise ValueError(
                f"expected [B,{expected[0]},{expected[1]}], got {tuple(signal.shape)}"
            )
        valid = self.resolve_mask(signal, token_valid_mask)
        tokens = self.patch_projection(signal).transpose(1, 2)
        tokens = self.patch_norm(torch.nn.functional.gelu(tokens))
        tokens = (tokens + self.position_embedding.to(tokens.dtype)).masked_fill(
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


class SharedTrajectoryDecoder(nn.Module):
    """One modality-agnostic decoder for either shared latent sequence."""

    def __init__(self, latent_dim: int, hidden_dim: int, target_points: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(target_points)),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"expected [B,N,{self.latent_dim}] shared latent")
        return self.network(latent)


class RawPatchDecoder(nn.Module):
    """Decode one target-modality raw patch from shared and private tokens."""

    def __init__(
        self,
        *,
        shared_dim: int,
        private_dim: int,
        hidden_dim: int,
        output_channels: int,
        patch_samples: int,
    ) -> None:
        super().__init__()
        self.shared_dim = int(shared_dim)
        self.private_dim = int(private_dim)
        self.output_channels = int(output_channels)
        self.patch_samples = int(patch_samples)
        self.network = nn.Sequential(
            nn.LayerNorm(self.shared_dim + self.private_dim),
            nn.Linear(self.shared_dim + self.private_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(
                int(hidden_dim), self.output_channels * self.patch_samples
            ),
        )

    def forward(
        self, shared: torch.Tensor, private: torch.Tensor
    ) -> torch.Tensor:
        if shared.ndim != 3 or private.ndim != 3:
            raise ValueError("shared and private latents must be [B,N,D]")
        if shared.shape[:2] != private.shape[:2]:
            raise ValueError("shared/private token axes differ")
        if shared.shape[-1] != self.shared_dim or private.shape[-1] != self.private_dim:
            raise ValueError("shared/private latent dimension mismatch")
        patches = self.network(torch.cat((shared, private), dim=-1))
        batch, tokens = patches.shape[:2]
        return (
            patches.view(
                batch,
                tokens,
                self.output_channels,
                self.patch_samples,
            )
            .permute(0, 2, 1, 3)
            .reshape(batch, self.output_channels, tokens * self.patch_samples)
        )


class ContinuousSharedPrivateModel(nn.Module):
    """Independent continuous shared/private branches for paired physiology."""

    architecture_name = "continuous_shared_private_v1"
    token_temporal_scope = "bidirectional_full_window"

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
        target_points: int = 20,
        encoder_depth: int = 2,
        encoder_num_heads: int = 4,
        encoder_feedforward_dim: int = 256,
        trajectory_decoder_hidden_dim: int = 128,
        raw_decoder_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.shared_dim = int(shared_dim)
        self.private_dims = {
            "eeg": int(eeg_private_dim),
            "fnirs": int(fnirs_private_dim),
        }
        common = {
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
        self.trajectory_decoder = SharedTrajectoryDecoder(
            self.shared_dim,
            int(trajectory_decoder_hidden_dim),
            int(target_points),
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

    def decode_raw(
        self,
        modality: str,
        shared: torch.Tensor,
        private: torch.Tensor,
        *,
        isolate_shared_gradient: bool = True,
    ) -> torch.Tensor:
        if modality not in {"eeg", "fnirs"}:
            raise ValueError(f"unsupported modality {modality!r}")
        decoder = self.eeg_raw_decoder if modality == "eeg" else self.fnirs_raw_decoder
        admitted_shared = shared.detach() if isolate_shared_gradient else shared
        return decoder(admitted_shared, private)

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        eeg_token_valid_mask: torch.Tensor | None = None,
        fnirs_token_valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        eeg_mask = self.eeg_shared_encoder.resolve_mask(eeg, eeg_token_valid_mask)
        fnirs_mask = self.fnirs_shared_encoder.resolve_mask(fnirs, fnirs_token_valid_mask)
        eeg_shared = self.eeg_shared_encoder(eeg, eeg_mask)
        fnirs_shared = self.fnirs_shared_encoder(fnirs, fnirs_mask)
        eeg_private = self.eeg_private_encoder(eeg, eeg_mask)
        fnirs_private = self.fnirs_private_encoder(fnirs, fnirs_mask)
        output = {
            "eeg_shared": eeg_shared,
            "fnirs_shared": fnirs_shared,
            "eeg_private": eeg_private,
            "fnirs_private": fnirs_private,
            "eeg_driver": self.trajectory_decoder(eeg_shared).masked_fill(
                ~eeg_mask.unsqueeze(-1), 0.0
            ),
            "fnirs_driver": self.trajectory_decoder(fnirs_shared).masked_fill(
                ~fnirs_mask.unsqueeze(-1), 0.0
            ),
            "eeg_raw": self.decode_raw("eeg", eeg_shared, eeg_private),
            "fnirs_raw": self.decode_raw("fnirs", fnirs_shared, fnirs_private),
            "eeg_token_valid_mask": eeg_mask,
            "fnirs_token_valid_mask": fnirs_mask,
        }
        output["eeg_raw"] = output["eeg_raw"].masked_fill(
            ~eeg_mask.repeat_interleave(
                self.eeg_shared_encoder.patch_samples, dim=1
            ).unsqueeze(1),
            0.0,
        )
        output["fnirs_raw"] = output["fnirs_raw"].masked_fill(
            ~fnirs_mask.repeat_interleave(
                self.fnirs_shared_encoder.patch_samples, dim=1
            ).unsqueeze(1),
            0.0,
        )
        return output


__all__ = [
    "ContinuousSharedPrivateModel",
    "RawPatchDecoder",
    "SharedTrajectoryDecoder",
    "TemporalPatchEncoder",
]
