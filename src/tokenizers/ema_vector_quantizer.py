"""Correct count-and-sum EMA vector quantization for physiology-semantic tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    """Complete quantizer output used by training, export, and diagnostics."""

    logits: torch.Tensor
    posterior: torch.Tensor
    hard_ids: torch.Tensor
    quantized: torch.Tensor
    expected_embedding: torch.Tensor
    codebook: torch.Tensor
    commitment_loss: torch.Tensor
    health: Dict[str, torch.Tensor]


class EMAVectorQuantizer(nn.Module):
    """Euclidean VQ with independently maintained count and vector-sum EMA."""

    def __init__(
        self,
        codebook_size: int = 128,
        embedding_dim: int = 64,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
        temperature: float = 1.0,
        assignment: str = "euclidean",
        revive_dead_codes: bool = False,
        dead_code_threshold: float = 0.1,
        revival_warmup_steps: int = 100,
        revival_interval: int = 100,
    ) -> None:
        super().__init__()
        if codebook_size <= 0 or embedding_dim <= 0:
            raise ValueError("codebook_size and embedding_dim must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if assignment not in {"euclidean", "cosine"}:
            raise ValueError("assignment must be 'euclidean' or 'cosine'")

        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.decay = float(decay)
        self.eps = float(eps)
        self.commitment_cost = float(commitment_cost)
        self.temperature = float(temperature)
        self.assignment = assignment
        self.revive_dead_codes = bool(revive_dead_codes)
        self.dead_code_threshold = float(dead_code_threshold)
        self.revival_warmup_steps = int(revival_warmup_steps)
        self.revival_interval = max(int(revival_interval), 1)

        scale = embedding_dim ** -0.5
        codebook = torch.empty(codebook_size, embedding_dim).uniform_(-scale, scale)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("ema_sum", codebook.clone())
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("revival_count", torch.zeros((), dtype=torch.long))

    def get_codebook_weight(self) -> torch.Tensor:
        return self.codebook

    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.codebook)

    def _assignment_logits(self, flat: torch.Tensor) -> torch.Tensor:
        if self.assignment == "cosine":
            flat_norm = F.normalize(flat, dim=-1)
            codebook_norm = F.normalize(self.codebook, dim=-1)
            return flat_norm @ codebook_norm.t() / self.temperature
        distances = (
            flat.square().sum(dim=-1, keepdim=True)
            - 2.0 * flat @ self.codebook.t()
            + self.codebook.square().sum(dim=-1).unsqueeze(0)
        )
        return -distances / self.temperature

    @staticmethod
    def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            value = value.clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, hard_ids: torch.Tensor) -> torch.Tensor:
        assignments = F.one_hot(hard_ids, self.codebook_size).to(flat.dtype)
        batch_count = self._distributed_sum(assignments.sum(dim=0))
        batch_sum = self._distributed_sum(assignments.t() @ flat)
        assigned = batch_count > 0

        self.ema_count[assigned] = (
            self.ema_count[assigned] * self.decay + batch_count[assigned] * (1.0 - self.decay)
        )
        self.ema_sum[assigned] = (
            self.ema_sum[assigned] * self.decay + batch_sum[assigned] * (1.0 - self.decay)
        )
        self.codebook[assigned] = self.ema_sum[assigned] / self.ema_count[assigned].unsqueeze(-1).clamp_min(self.eps)
        self.update_count.add_(1)

        revived = torch.zeros((), device=flat.device, dtype=torch.long)
        if (
            self.revive_dead_codes
            and int(self.update_count.item()) >= self.revival_warmup_steps
            and int(self.update_count.item()) % self.revival_interval == 0
            and flat.shape[0] > 0
        ):
            dead = torch.where(self.ema_count < self.dead_code_threshold)[0]
            if dead.numel() > 0:
                sample_idx = torch.arange(dead.numel(), device=flat.device) % flat.shape[0]
                replacements = flat[sample_idx]
                self.codebook[dead] = replacements
                self.ema_count[dead] = max(self.dead_code_threshold, self.eps)
                self.ema_sum[dead] = replacements * self.ema_count[dead].unsqueeze(-1)
                revived = torch.tensor(dead.numel(), device=flat.device, dtype=torch.long)
                self.revival_count.add_(revived.to(self.revival_count.device))
        return revived

    @torch.no_grad()
    def _health(self, hard_ids: torch.Tensor, drift: torch.Tensor, revived: torch.Tensor) -> Dict[str, torch.Tensor]:
        counts = torch.bincount(hard_ids, minlength=self.codebook_size).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        effective_codes = entropy.exp()

        normalized = F.normalize(self.codebook, dim=-1)
        cosine = normalized @ normalized.t()
        cosine.fill_diagonal_(-1.0)
        nearest_cosine = cosine.max(dim=1).values.mean()
        effective_rank = torch.linalg.matrix_rank(self.codebook.float()).to(self.codebook.dtype)
        return {
            "assignment_entropy": entropy,
            "effective_codes": effective_codes,
            "batch_active_codes": (counts > 0).sum().to(self.codebook.dtype),
            "ema_active_fraction": (self.ema_count >= self.dead_code_threshold).float().mean(),
            "effective_rank": effective_rank,
            "nearest_neighbor_cosine": nearest_cosine,
            "prototype_drift": drift,
            "revived_codes": revived.to(self.codebook.dtype),
            "total_revivals": self.revival_count.to(self.codebook.dtype),
        }

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        if latent.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected latent dimension {self.embedding_dim}, got {latent.shape[-1]}"
            )
        original_shape = latent.shape[:-1]
        flat = latent.reshape(-1, self.embedding_dim)
        logits_flat = self._assignment_logits(flat)
        posterior_flat = logits_flat.softmax(dim=-1)
        hard_flat = posterior_flat.argmax(dim=-1)
        lookup_flat = F.embedding(hard_flat, self.codebook)
        expected_flat = posterior_flat @ self.codebook
        commitment_loss = self.commitment_cost * F.mse_loss(flat, lookup_flat.detach())

        previous = self.codebook.detach().clone()
        revived = torch.zeros((), device=flat.device, dtype=torch.long)
        if self.training:
            revived = self._ema_update(flat.detach(), hard_flat.detach())
        drift = (self.codebook - previous).norm(dim=-1).mean()

        quantized_flat = flat + (lookup_flat - flat).detach()
        health = self._health(hard_flat.detach(), drift.detach(), revived.detach())
        return QuantizerOutput(
            logits=logits_flat.reshape(*original_shape, self.codebook_size),
            posterior=posterior_flat.reshape(*original_shape, self.codebook_size),
            hard_ids=hard_flat.reshape(*original_shape),
            quantized=quantized_flat.reshape(*original_shape, self.embedding_dim),
            expected_embedding=expected_flat.reshape(*original_shape, self.embedding_dim),
            codebook=self.codebook,
            commitment_loss=commitment_loss,
            health=health,
        )


__all__ = ["EMAVectorQuantizer", "QuantizerOutput"]
