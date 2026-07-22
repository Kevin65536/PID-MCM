"""Variable-channel EFRM with source-faithful MAE and symmetric alignment losses."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block


def _sincos_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("sinusoidal embedding dimensions must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10_000**omega)
    angles = positions.reshape(-1, 1) * omega.reshape(1, -1)
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)


def sincos_2d(embed_dim: int, height: int, width: int, *, cls_token: bool = True) -> torch.Tensor:
    if embed_dim % 4:
        raise ValueError("2D sinusoidal embedding dimension must be divisible by four")
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    embedding = np.concatenate(
        (_sincos_1d(embed_dim // 2, rows.reshape(-1)), _sincos_1d(embed_dim // 2, cols.reshape(-1))),
        axis=1,
    )
    if cls_token:
        embedding = np.concatenate((np.zeros((1, embed_dim), dtype=np.float64), embedding), axis=0)
    return torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0)


class DynamicPatchEmbed(nn.Module):
    """Patch only along time while accepting the measured channel count of each record."""

    def __init__(self, in_channels: int, patch_samples: int, embed_dim: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_samples = int(patch_samples)
        self.proj = nn.Conv2d(
            self.in_channels,
            int(embed_dim),
            kernel_size=(1, self.patch_samples),
            stride=(1, self.patch_samples),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if values.ndim != 4 or values.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [batch,{self.in_channels},channels,time], got {tuple(values.shape)}"
            )
        if values.shape[-1] % self.patch_samples:
            raise ValueError("time length must be divisible by the physical patch size")
        embedded = self.proj(values)
        height, width = int(embedded.shape[-2]), int(embedded.shape[-1])
        return embedded.flatten(2).transpose(1, 2), height, width


class VariableChannelMAE(nn.Module):
    """EFRM MAE whose channel axis and sinusoidal position grid are dynamic."""

    def __init__(
        self,
        *,
        in_channels: int,
        patch_samples: int,
        mask_ratio: float = 0.5,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.patch_samples = int(patch_samples)
        self.in_channels = int(in_channels)
        self.mask_ratio = float(mask_ratio)
        self.embed_dim = int(embed_dim)
        self.decoder_embed_dim = int(decoder_embed_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.patch_embed = DynamicPatchEmbed(in_channels, patch_samples, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_samples * in_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _run_blocks(
        self,
        values: torch.Tensor,
        blocks: nn.ModuleList,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in blocks:
            if self.activation_checkpointing and self.training:
                values = checkpoint(
                    lambda current, layer=block: layer(current, attn_mask=attention_mask),
                    values,
                    use_reentrant=False,
                )
            else:
                values = block(values, attn_mask=attention_mask)
        return values

    @staticmethod
    def _attention_mask(valid: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        cls_valid = torch.ones((valid.shape[0], 1), dtype=torch.bool, device=valid.device)
        token_valid = torch.cat((cls_valid, valid.bool()), dim=1)
        mask = torch.zeros(
            (valid.shape[0], 1, 1, token_valid.shape[1]),
            dtype=dtype,
            device=valid.device,
        )
        return mask.masked_fill(~token_valid[:, None, None, :], torch.finfo(dtype).min)

    def patchify(self, values: torch.Tensor) -> torch.Tensor:
        batch, components, channels, time = values.shape
        if time % self.patch_samples:
            raise ValueError("time length is not divisible by patch_samples")
        width = time // self.patch_samples
        patches = values.reshape(batch, components, channels, width, self.patch_samples)
        return patches.permute(0, 2, 3, 4, 1).reshape(
            batch, channels * width, self.patch_samples * components
        )

    @staticmethod
    def _flatten_valid(valid: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        if valid.shape != (valid.shape[0], height, width):
            raise ValueError(f"patch-valid mask must be [batch,{height},{width}], got {tuple(valid.shape)}")
        return valid.reshape(valid.shape[0], height * width).bool()

    def random_masking(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, dim = values.shape
        valid_counts = valid.sum(dim=1)
        if int(valid_counts.min()) < 2:
            raise ValueError("each sample requires at least two valid patch tokens")
        keep_count = max(1, int(torch.floor(valid_counts.min().float() * (1.0 - self.mask_ratio)).item()))
        noise = torch.rand(batch, length, device=values.device)
        noise = noise.masked_fill(~valid, 2.0)
        shuffled = torch.argsort(noise, dim=1)
        restore = torch.argsort(shuffled, dim=1)
        keep = shuffled[:, :keep_count]
        kept = torch.gather(values, 1, keep.unsqueeze(-1).expand(-1, -1, dim))
        reconstruction_mask = valid.float()
        reconstruction_mask.scatter_(1, keep, 0.0)
        return kept, reconstruction_mask, restore

    def forward_encoder(
        self,
        values: torch.Tensor,
        patch_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        embedded, height, width = self.patch_embed(values)
        valid = self._flatten_valid(patch_valid, height=height, width=width)
        positions = sincos_2d(self.embed_dim, height, width).to(device=embedded.device, dtype=embedded.dtype)
        embedded = embedded + positions[:, 1:]
        embedded, reconstruction_mask, restore = self.random_masking(embedded, valid)
        cls = (self.cls_token + positions[:, :1]).expand(values.shape[0], -1, -1)
        latent = self._run_blocks(torch.cat((cls, embedded), dim=1), self.blocks)
        return self.norm(latent), reconstruction_mask, restore, height, width, valid

    def forward_decoder(
        self,
        latent: torch.Tensor,
        restore: torch.Tensor,
        *,
        height: int,
        width: int,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        decoded = self.decoder_embed(latent)
        missing = restore.shape[1] + 1 - decoded.shape[1]
        mask_tokens = self.mask_token.expand(decoded.shape[0], missing, -1)
        patches = torch.cat((decoded[:, 1:], mask_tokens), dim=1)
        patches = torch.gather(
            patches,
            1,
            restore.unsqueeze(-1).expand(-1, -1, decoded.shape[-1]),
        )
        decoded = torch.cat((decoded[:, :1], patches), dim=1)
        positions = sincos_2d(self.decoder_embed_dim, height, width).to(
            device=decoded.device, dtype=decoded.dtype
        )
        decoded = self._run_blocks(
            decoded + positions,
            self.decoder_blocks,
            self._attention_mask(valid, dtype=decoded.dtype),
        )
        prediction = self.decoder_pred(self.decoder_norm(decoded))[:, 1:]
        return prediction

    def reconstruction_loss(
        self,
        values: torch.Tensor,
        prediction: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self.patchify(values)
        per_patch = (prediction - target).square().mean(dim=-1)
        denominator = reconstruction_mask.sum().clamp_min(1.0)
        return (per_patch * reconstruction_mask).sum() / denominator

    def reconstruct(self, values: torch.Tensor, patch_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, mask, restore, height, width, valid = self.forward_encoder(values, patch_valid)
        prediction = self.forward_decoder(
            latent, restore, height=height, width=width, valid=valid
        )
        return {
            "loss": self.reconstruction_loss(values, prediction, mask),
            "prediction": prediction,
            "reconstruction_mask": mask,
        }

    def forward_embed(self, values: torch.Tensor, patch_valid: torch.Tensor) -> torch.Tensor:
        embedded, height, width = self.patch_embed(values)
        valid = self._flatten_valid(patch_valid, height=height, width=width)
        positions = sincos_2d(self.embed_dim, height, width).to(device=embedded.device, dtype=embedded.dtype)
        if int(valid.sum(dim=1).min()) == 0:
            raise ValueError("embedding requires at least one valid physical patch")
        embedded = (embedded + positions[:, 1:]) * valid.unsqueeze(-1)
        cls = (self.cls_token + positions[:, :1]).expand(values.shape[0], -1, -1)
        encoded = self.norm(
            self._run_blocks(
                torch.cat((cls, embedded), dim=1),
                self.blocks,
                self._attention_mask(valid, dtype=embedded.dtype),
            )
        )[:, 1:]
        return (encoded * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1)


class EFRMSyncModel(nn.Module):
    """Two EFRM MAEs trained from the same synchronized paired batch."""

    def __init__(
        self,
        *,
        eeg_patch_samples: int = 50,
        fnirs_patch_samples: int = 20,
        mask_ratio: float = 0.5,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        clip_logit_multiplier: float = 0.1,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        shared: dict[str, Any] = {
            "mask_ratio": mask_ratio,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "decoder_embed_dim": decoder_embed_dim,
            "decoder_depth": decoder_depth,
            "decoder_num_heads": decoder_num_heads,
            "mlp_ratio": mlp_ratio,
            "activation_checkpointing": activation_checkpointing,
        }
        self.eeg_model = VariableChannelMAE(in_channels=1, patch_samples=eeg_patch_samples, **shared)
        self.fnirs_model = VariableChannelMAE(in_channels=2, patch_samples=fnirs_patch_samples, **shared)
        self.embed_dim = int(embed_dim)
        self.clip_logit_multiplier = float(clip_logit_multiplier)

    def encode(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        eeg_patch_valid: torch.Tensor,
        fnirs_patch_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.eeg_model.forward_embed(eeg, eeg_patch_valid),
            self.fnirs_model.forward_embed(fnirs, fnirs_patch_valid),
        )

    def alignment(self, eeg_embedding: torch.Tensor, fnirs_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        eeg_normalized = F.normalize(eeg_embedding, p=2, dim=-1)
        fnirs_normalized = F.normalize(fnirs_embedding, p=2, dim=-1)
        cosine = eeg_normalized @ fnirs_normalized.T
        logits_eeg = self.clip_logit_multiplier * cosine
        logits_fnirs = self.clip_logit_multiplier * cosine.T
        labels = torch.arange(cosine.shape[0], device=cosine.device)
        loss = (F.cross_entropy(logits_eeg, labels) + F.cross_entropy(logits_fnirs, labels)) / 2.0
        return {
            "loss": loss,
            "cosine_similarity": cosine,
            "eeg_to_fnirs_logits": logits_eeg,
            "fnirs_to_eeg_logits": logits_fnirs,
        }

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        eeg_patch_valid: torch.Tensor,
        fnirs_patch_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eeg_reconstruction = self.eeg_model.reconstruct(eeg, eeg_patch_valid)
        fnirs_reconstruction = self.fnirs_model.reconstruct(fnirs, fnirs_patch_valid)
        eeg_embedding, fnirs_embedding = self.encode(
            eeg, fnirs, eeg_patch_valid, fnirs_patch_valid
        )
        alignment = self.alignment(eeg_embedding, fnirs_embedding)
        total = eeg_reconstruction["loss"] + fnirs_reconstruction["loss"] + alignment["loss"]
        return {
            "loss": total,
            "eeg_reconstruction_loss": eeg_reconstruction["loss"],
            "fnirs_reconstruction_loss": fnirs_reconstruction["loss"],
            "clip_alignment_loss": alignment["loss"],
            "eeg_embedding": eeg_embedding,
            "fnirs_embedding": fnirs_embedding,
            "cosine_similarity": alignment["cosine_similarity"],
            "eeg_to_fnirs_logits": alignment["eeg_to_fnirs_logits"],
            "fnirs_to_eeg_logits": alignment["fnirs_to_eeg_logits"],
            "eeg_reconstruction_mask": eeg_reconstruction["reconstruction_mask"],
            "fnirs_reconstruction_mask": fnirs_reconstruction["reconstruction_mask"],
        }


class EFRMDownstreamModel(nn.Module):
    """Linear-probe or fine-tuning head for classification and sequence regression."""

    def __init__(
        self,
        backbone: EFRMSyncModel,
        *,
        output_dim: int,
        modality: str = "paired",
        target_length: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if modality not in {"eeg", "fnirs", "paired"}:
            raise ValueError("modality must be eeg, fnirs, or paired")
        self.backbone = backbone
        self.modality = modality
        self.output_dim = int(output_dim)
        self.target_length = int(target_length)
        self.norm = nn.LayerNorm(backbone.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.embed_dim, self.output_dim * self.target_length)

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        eeg_patch_valid: torch.Tensor,
        fnirs_patch_valid: torch.Tensor,
    ) -> torch.Tensor:
        eeg_embedding, fnirs_embedding = self.backbone.encode(
            eeg, fnirs, eeg_patch_valid, fnirs_patch_valid
        )
        if self.modality == "eeg":
            features = eeg_embedding
        elif self.modality == "fnirs":
            features = fnirs_embedding
        else:
            features = eeg_embedding + fnirs_embedding
        output = self.head(self.dropout(self.norm(features)))
        if self.target_length > 1:
            output = output.reshape(output.shape[0], self.output_dim, self.target_length)
        return output
