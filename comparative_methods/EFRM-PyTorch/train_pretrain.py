#!/usr/bin/env python3
"""Leakage-bounded, resumable synchronized EFRM development pretraining."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efrm_pytorch.data import (
    EFRMPairedWindowAdapter,
    EFRMSyncPretrainDataset,
    InventoryDiverseBatchSampler,
    RecordGroupedBatchSampler,
    collate_efrm_pairs,
)
from efrm_pytorch.model import EFRMSyncModel
from efrm_pytorch.protocol import PretrainingBoundary, role_counts
from efrm_pytorch.training import cached_pretrain_backward, evaluate_pretrain_batch, move_batch
from efrm_pytorch.visualization import export_alignment_evidence, render_alignment_report
from preflight import DEVELOPMENT_MANIFESTS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _subset(dataset: EFRMSyncPretrainDataset, selected: Iterable[int]) -> EFRMSyncPretrainDataset:
    view = copy.copy(dataset)
    view.indices = [dataset.indices[int(index)] for index in selected]
    return view


def _loader(
    dataset: EFRMSyncPretrainDataset,
    *,
    batch_size: int,
    seed: int,
    workers: int,
    inventory_diverse: bool,
    inventory_cache_path: Path | None,
) -> tuple[DataLoader, Any]:
    sampler_class = InventoryDiverseBatchSampler if inventory_diverse else RecordGroupedBatchSampler
    sampler = sampler_class(
        dataset,
        batch_size=batch_size,
        seed=seed,
        drop_last=not inventory_diverse,
        **({"inventory_cache_path": inventory_cache_path} if inventory_diverse else {}),
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": collate_efrm_pairs,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs), sampler


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("epoch produced no batches")
    weights = np.asarray([row["pair_count"] for row in rows], dtype=np.float64)
    result = {"batch_count": float(len(rows)), "pair_count": float(weights.sum())}
    for key in ("loss", "eeg_reconstruction_loss", "fnirs_reconstruction_loss", "clip_alignment_loss"):
        result[key] = float(np.average([row[key] for row in rows], weights=weights))
    return result


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _checkpoint(
    path: Path,
    *,
    model: EFRMSyncModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_loss: float,
    patience: int,
    boundary_sha256: str,
) -> None:
    torch.save({
        "schema": "efrm_sync_pretrain_checkpoint_v1",
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_validation_loss": best_loss,
        "patience": patience,
        "boundary_sha256": boundary_sha256,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(METHOD_ROOT / "configs/pretrain_sync.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--architecture-smoke", action="store_true")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    seed = int(training["seed"])
    _seed_everything(seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full EFRM pretraining requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_efrm_sync_dev")
    run_dir = METHOD_ROOT / "runs/pretraining" / run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    for child in ("checkpoints", "metrics", "figures", "figure_data"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "resolved_config.yaml")

    tasks = list(DEVELOPMENT_MANIFESTS)
    paths = [DEVELOPMENT_MANIFESTS[task] for task in tasks]
    boundary = PretrainingBoundary.from_manifests(
        paths, tasks=tasks, mode="development_public_only", cache_root=config["data"]["cache_root"]
    )
    boundary_manifest = boundary.manifest()
    (run_dir / "boundary_manifest.json").write_text(
        json.dumps(boundary_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    full_dataset = EFRMSyncPretrainDataset(
        cache_root=config["data"]["cache_root"],
        dataset_ids=tuple(config["data"]["dataset_ids"]),
        seed=seed,
        adapter=EFRMPairedWindowAdapter(
            duration_s=float(config["data"]["window_duration_s"]),
            eeg_rate_hz=float(config["data"]["eeg_sample_rate_hz"]),
            fnirs_rate_hz=float(config["data"]["fnirs_sample_rate_hz"]),
            eeg_patch_samples=int(config["model"]["eeg_patch_samples"]),
            fnirs_patch_samples=int(config["model"]["fnirs_patch_samples"]),
            require_full_analysis_support=bool(config["data"]["require_full_analysis_support"]),
        ),
    )
    train_dataset = _subset(full_dataset, boundary.indices_for(full_dataset, "train"))
    validation_dataset = _subset(full_dataset, boundary.indices_for(full_dataset, "validation"))
    batch_size = int(training["effective_batch_size"])
    inventory_cache = (
        METHOD_ROOT / "runs/cache" /
        f"inventory_{boundary_manifest['boundary_sha256']}.json"
    )
    train_loader, train_sampler = _loader(
        train_dataset, batch_size=batch_size, seed=seed, workers=args.num_workers,
        inventory_diverse=not args.architecture_smoke,
        inventory_cache_path=inventory_cache,
    )
    validation_loader, validation_sampler = _loader(
        validation_dataset, batch_size=batch_size, seed=seed + 1, workers=args.num_workers,
        inventory_diverse=not args.architecture_smoke,
        inventory_cache_path=inventory_cache,
    )

    model_config = config["model"]
    model = EFRMSyncModel(
        eeg_patch_samples=int(model_config["eeg_patch_samples"]),
        fnirs_patch_samples=int(model_config["fnirs_patch_samples"]),
        mask_ratio=float(model_config["mask_ratio"]),
        embed_dim=int(model_config["embedding_dim"]),
        depth=int(model_config["encoder_depth"]),
        num_heads=int(model_config["encoder_heads"]),
        decoder_embed_dim=int(model_config["decoder_embedding_dim"]),
        decoder_depth=int(model_config["decoder_depth"]),
        decoder_num_heads=int(model_config["decoder_heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        clip_logit_multiplier=float(model_config["clip_logit_multiplier"]),
        activation_checkpointing=(
            bool(model_config["activation_checkpointing"])
            and not args.no_activation_checkpointing
        ),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["adam_betas"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(args.epochs or training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_config = config["loss"]
    loss_kwargs = {
        "eeg_reconstruction_weight": float(loss_config["eeg_reconstruction_weight"]),
        "fnirs_reconstruction_weight": float(loss_config["fnirs_reconstruction_weight"]),
        "clip_alignment_weight": float(loss_config["clip_alignment_weight"]),
    }

    start_epoch = 0
    best_loss = math.inf
    patience = 0
    latest = run_dir / "checkpoints/latest.pt"
    if args.resume:
        payload = torch.load(latest, map_location=device, weights_only=False)
        if payload["boundary_sha256"] != boundary_manifest["boundary_sha256"]:
            raise RuntimeError("resume boundary hash does not match")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_loss = float(payload["best_validation_loss"])
        patience = int(payload["patience"])

    vendor = METHOD_ROOT.parent / "EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model"
    manifest = {
        "schema": "efrm_sync_pretraining_run_v1",
        "status": "running",
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "method_revision": _git_revision(REPO_ROOT),
        "upstream_revision": _git_revision(vendor),
        "config_sha256": _sha256(config_path),
        "boundary_sha256": boundary_manifest["boundary_sha256"],
        "protected_test_opened": False,
        "device": str(device),
        "parameter_count": parameter_count,
        "contrastive_batch_size": batch_size,
        "recompute_chunk_size": args.chunk_size,
        "gradient_cache": "two_pass_exact_v1",
        "activation_checkpointing": model.eeg_model.activation_checkpointing,
        "architecture_smoke_sampler_override": (
            "record_grouped_for_fast_memory_measurement" if args.architecture_smoke else None
        ),
        "train": role_counts(train_dataset, range(len(train_dataset))),
        "validation": role_counts(validation_dataset, range(len(validation_dataset))),
        "train_sampler": train_sampler.manifest(),
        "validation_sampler": validation_sampler.manifest(),
        "inventory_cache": None if args.architecture_smoke else str(inventory_cache.resolve()),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "running", "epoch": start_epoch}), encoding="utf-8")

    max_train = (
        (args.max_train_batches or 1) if args.architecture_smoke else args.max_train_batches
    )
    max_validation = (
        (args.max_validation_batches or 1)
        if args.architecture_smoke else args.max_validation_batches
    )
    amp_dtype = torch.bfloat16
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        model.train()
        train_rows: list[dict[str, float]] = []
        for batch_index, raw_batch in enumerate(train_loader):
            if max_train is not None and batch_index >= max_train:
                break
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            metrics = cached_pretrain_backward(
                model, batch, chunk_size=args.chunk_size, amp_dtype=amp_dtype, **loss_kwargs
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite EFRM gradient norm")
            optimizer.step()
            metrics["gradient_norm"] = float(gradient_norm)
            metrics["epoch"] = float(epoch)
            metrics["batch"] = float(batch_index)
            train_rows.append(metrics)
            _append_jsonl(run_dir / "metrics/train_steps.jsonl", metrics)

        validation_sampler.set_epoch(epoch)
        model.eval()
        validation_rows: list[dict[str, float]] = []
        final_evidence: tuple[dict[str, Any], dict[str, torch.Tensor]] | None = None
        for batch_index, raw_batch in enumerate(validation_loader):
            if max_validation is not None and batch_index >= max_validation:
                break
            batch = move_batch(raw_batch, device)
            metrics, evidence = evaluate_pretrain_batch(
                model, batch, chunk_size=args.chunk_size, amp_dtype=amp_dtype, **loss_kwargs
            )
            validation_rows.append(metrics)
            final_evidence = (raw_batch, evidence)
        train_epoch = _mean_metrics(train_rows)
        validation_epoch = _mean_metrics(validation_rows)
        scheduler.step()
        epoch_row = {
            "epoch": epoch,
            "seconds": time.time() - epoch_start,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_epoch,
            "validation": validation_epoch,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        _append_jsonl(run_dir / "metrics/epochs.jsonl", epoch_row)
        improved = validation_epoch["loss"] < best_loss
        if improved:
            best_loss = validation_epoch["loss"]
            patience = 0
        else:
            patience += 1
        if not args.architecture_smoke:
            _checkpoint(
                latest, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                best_loss=best_loss, patience=patience,
                boundary_sha256=boundary_manifest["boundary_sha256"],
            )
            if improved:
                shutil.copy2(latest, run_dir / "checkpoints/best.pt")

        if final_evidence is not None:
            raw_batch, evidence = final_evidence
            metadata = [
                {key: raw_batch[key][index] for key in (
                    "sample_id", "dataset_id", "subject", "record_id", "join_key",
                    "task_namespace", "condition", "crop_start_s", "duration_s",
                )}
                for index in range(len(raw_batch["sample_id"]))
            ]
            evidence_path = export_alignment_evidence(
                run_dir / "figure_data",
                eeg_embeddings=evidence["eeg_embedding"],
                fnirs_embeddings=evidence["fnirs_embedding"],
                metadata=metadata,
                logit_multiplier=float(model_config["clip_logit_multiplier"]),
            )
            render_alignment_report(evidence_path, run_dir)

        status = {
            "status": "smoke_passed" if args.architecture_smoke else "running",
            "epoch": epoch,
            "best_validation_loss": best_loss,
            "patience": patience,
            "cuda_peak_allocated_gib": epoch_row["cuda_peak_allocated_gib"],
            "cuda_peak_reserved_gib": epoch_row["cuda_peak_reserved_gib"],
            "protected_test_opened": False,
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(epoch_row), flush=True)
        if args.architecture_smoke:
            break
        if epoch + 1 >= int(training["min_epochs"]) and patience >= int(training["early_stopping_patience"]):
            break

    manifest["status"] = "smoke_passed" if args.architecture_smoke else "completed"
    manifest["completed_at"] = datetime.now().isoformat()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
