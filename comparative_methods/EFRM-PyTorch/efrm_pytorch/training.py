"""Exact-gradient cached optimization utilities for synchronized EFRM pretraining."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch

from .model import EFRMSyncModel


TENSOR_KEYS = ("eeg", "fnirs", "eeg_patch_valid", "fnirs_patch_valid")


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    output = dict(batch)
    for key in TENSOR_KEYS:
        output[key] = batch[key].to(device, non_blocking=True)
    return output


def _chunk(batch: Mapping[str, Any], start: int, stop: int) -> dict[str, torch.Tensor]:
    return {key: batch[key][start:stop] for key in TENSOR_KEYS}


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def cached_pretrain_backward(
    model: EFRMSyncModel,
    batch: Mapping[str, Any],
    *,
    chunk_size: int,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    eeg_reconstruction_weight: float = 1.0,
    fnirs_reconstruction_weight: float = 1.0,
    clip_alignment_weight: float = 1.0,
) -> dict[str, float]:
    """Backpropagate a full-batch CLIP loss with bounded activation memory.

    The first pass caches detached embeddings for the complete contrastive
    matrix. Gradients of that matrix with respect to every embedding are then
    computed exactly. The second pass recomputes each chunk and injects those
    cached embedding gradients while also backpropagating both MAE losses.
    This is exact for the deterministic EFRM encoders and is not ordinary
    gradient accumulation.
    """

    device = batch["eeg"].device
    pair_count = int(batch["eeg"].shape[0])
    if pair_count < 2:
        raise ValueError("EFRM contrastive optimization requires at least two pairs")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    cached_eeg: list[torch.Tensor] = []
    cached_fnirs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, pair_count, chunk_size):
            part = _chunk(batch, start, min(start + chunk_size, pair_count))
            with _autocast(device, amp_dtype):
                eeg_embedding, fnirs_embedding = model.encode(**part)
            cached_eeg.append(eeg_embedding.float())
            cached_fnirs.append(fnirs_embedding.float())

    eeg_leaf = torch.cat(cached_eeg).detach().requires_grad_(True)
    fnirs_leaf = torch.cat(cached_fnirs).detach().requires_grad_(True)
    alignment = model.alignment(eeg_leaf, fnirs_leaf)
    weighted_clip = float(clip_alignment_weight) * alignment["loss"]
    eeg_gradient, fnirs_gradient = torch.autograd.grad(weighted_clip, (eeg_leaf, fnirs_leaf))

    eeg_reconstruction_total = 0.0
    fnirs_reconstruction_total = 0.0
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        part = _chunk(batch, start, stop)
        fraction = (stop - start) / pair_count
        with _autocast(device, amp_dtype):
            eeg_reconstruction = model.eeg_model.reconstruct(
                part["eeg"], part["eeg_patch_valid"]
            )["loss"]
            fnirs_reconstruction = model.fnirs_model.reconstruct(
                part["fnirs"], part["fnirs_patch_valid"]
            )["loss"]
            eeg_embedding, fnirs_embedding = model.encode(**part)
            gradient_surrogate = (
                (eeg_embedding.float() * eeg_gradient[start:stop]).sum()
                + (fnirs_embedding.float() * fnirs_gradient[start:stop]).sum()
            )
            reconstruction = fraction * (
                float(eeg_reconstruction_weight) * eeg_reconstruction
                + float(fnirs_reconstruction_weight) * fnirs_reconstruction
            )
            backward_proxy = reconstruction + gradient_surrogate
        backward_proxy.backward()
        eeg_reconstruction_total += fraction * float(eeg_reconstruction.detach())
        fnirs_reconstruction_total += fraction * float(fnirs_reconstruction.detach())

    clip_value = float(alignment["loss"].detach())
    total = (
        float(eeg_reconstruction_weight) * eeg_reconstruction_total
        + float(fnirs_reconstruction_weight) * fnirs_reconstruction_total
        + float(clip_alignment_weight) * clip_value
    )
    return {
        "loss": total,
        "eeg_reconstruction_loss": eeg_reconstruction_total,
        "fnirs_reconstruction_loss": fnirs_reconstruction_total,
        "clip_alignment_loss": clip_value,
        "pair_count": float(pair_count),
    }


@torch.no_grad()
def evaluate_pretrain_batch(
    model: EFRMSyncModel,
    batch: Mapping[str, Any],
    *,
    chunk_size: int,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    eeg_reconstruction_weight: float = 1.0,
    fnirs_reconstruction_weight: float = 1.0,
    clip_alignment_weight: float = 1.0,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    device = batch["eeg"].device
    pair_count = int(batch["eeg"].shape[0])
    eeg_embeddings: list[torch.Tensor] = []
    fnirs_embeddings: list[torch.Tensor] = []
    eeg_reconstruction_total = 0.0
    fnirs_reconstruction_total = 0.0
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        part = _chunk(batch, start, stop)
        fraction = (stop - start) / pair_count
        with _autocast(device, amp_dtype):
            eeg_reconstruction = model.eeg_model.reconstruct(
                part["eeg"], part["eeg_patch_valid"]
            )["loss"]
            fnirs_reconstruction = model.fnirs_model.reconstruct(
                part["fnirs"], part["fnirs_patch_valid"]
            )["loss"]
            eeg_embedding, fnirs_embedding = model.encode(**part)
        eeg_reconstruction_total += fraction * float(eeg_reconstruction)
        fnirs_reconstruction_total += fraction * float(fnirs_reconstruction)
        eeg_embeddings.append(eeg_embedding.float())
        fnirs_embeddings.append(fnirs_embedding.float())
    eeg_all = torch.cat(eeg_embeddings)
    fnirs_all = torch.cat(fnirs_embeddings)
    alignment = model.alignment(eeg_all, fnirs_all)
    clip_value = float(alignment["loss"])
    total = (
        float(eeg_reconstruction_weight) * eeg_reconstruction_total
        + float(fnirs_reconstruction_weight) * fnirs_reconstruction_total
        + float(clip_alignment_weight) * clip_value
    )
    return {
        "loss": total,
        "eeg_reconstruction_loss": eeg_reconstruction_total,
        "fnirs_reconstruction_loss": fnirs_reconstruction_total,
        "clip_alignment_loss": clip_value,
        "pair_count": float(pair_count),
    }, {
        "eeg_embedding": eeg_all,
        "fnirs_embedding": fnirs_all,
        "cosine_similarity": alignment["cosine_similarity"],
    }
