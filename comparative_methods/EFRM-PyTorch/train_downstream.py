#!/usr/bin/env python3
"""Train one leakage-bounded public-development EFRM transfer task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efrm_pytorch.metrics import (
    classification_metrics,
    regression_metrics,
    subject_metrics,
)
from efrm_pytorch.model import EFRMDownstreamModel, EFRMSyncModel
from efrm_pytorch.tasks import (
    EFRMUnifiedTaskDataset,
    collate_efrm_task,
    get_task_spec,
)


SCHEMA = "efrm_downstream_public_v1"
TASKS = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
)
_STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()


class RecordGroupedBatchSampler(Sampler[list[int]]):
    """Batch only one measured channel inventory/record at a time."""

    def __init__(
        self,
        dataset: EFRMUnifiedTaskDataset,
        indices: Sequence[int],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[str, list[int]] = {}
        for index in indices:
            key = str(dataset.lightweight_metadata(int(index))["join_key"])
            groups.setdefault(key, []).append(int(index))
        if not groups:
            raise RuntimeError("record-grouped sampler received no indices")
        self.groups = groups

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        keys = sorted(self.groups)
        if self.shuffle:
            rng.shuffle(keys)
        for key in keys:
            indices = list(self.groups[key])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]
        self.epoch += 1

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.groups.values()
        )


def make_loader(
    dataset: EFRMUnifiedTaskDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> tuple[DataLoader, RecordGroupedBatchSampler]:
    sampler = RecordGroupedBatchSampler(
        dataset,
        indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": int(workers),
        "pin_memory": True,
        "collate_fn": collate_efrm_task,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, **kwargs), sampler


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _build_backbone(config: Mapping[str, Any]) -> EFRMSyncModel:
    model = dict(config.get("pretraining_model", {}))
    return EFRMSyncModel(
        eeg_patch_samples=int(model.get("eeg_patch_samples", 50)),
        fnirs_patch_samples=int(model.get("fnirs_patch_samples", 20)),
        mask_ratio=float(model.get("mask_ratio", 0.5)),
        embed_dim=int(model.get("embedding_dim", 768)),
        depth=int(model.get("encoder_depth", 12)),
        num_heads=int(model.get("encoder_heads", 12)),
        decoder_embed_dim=int(model.get("decoder_embedding_dim", 512)),
        decoder_depth=int(model.get("decoder_depth", 8)),
        decoder_num_heads=int(model.get("decoder_heads", 16)),
        mlp_ratio=float(model.get("mlp_ratio", 4.0)),
        clip_logit_multiplier=float(model.get("clip_logit_multiplier", 0.1)),
        activation_checkpointing=bool(model.get("activation_checkpointing", False)),
    )


def _subject_sets(
    dataset: EFRMUnifiedTaskDataset,
    indices: Sequence[int],
) -> set[str]:
    return {
        str(dataset.lightweight_metadata(int(index))["subject"])
        for index in indices
    }


def _validate_pretraining_boundary(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    dataset: EFRMUnifiedTaskDataset,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> dict[str, Any]:
    run_dir = checkpoint_path.parent.parent
    boundary_path = run_dir / "boundary_manifest.json"
    manifest_path = run_dir / "manifest.json"
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if boundary.get("schema") != "efrm_pretraining_boundary_v1":
        raise ValueError("downstream development requires a subject-bounded pretraining run")
    if boundary.get("protected_test_opened") or run_manifest.get("protected_test_opened"):
        raise PermissionError("pretraining artifacts report an opened protected test")
    if checkpoint.get("boundary_sha256") != boundary.get("boundary_sha256"):
        raise RuntimeError("pretraining checkpoint/boundary hash mismatch")
    dataset_id = dataset.spec.dataset_id
    pretrain_train = {
        str(value)
        for value in boundary["train_subjects_by_dataset"].get(dataset_id, ())
    }
    pretrain_validation = {
        str(value)
        for value in boundary["validation_subjects_by_dataset"].get(dataset_id, ())
    }
    downstream_train = _subject_sets(dataset, train_indices)
    downstream_validation = _subject_sets(dataset, validation_indices)
    if downstream_validation - pretrain_validation:
        raise RuntimeError(
            "downstream validation subjects were not held out by pretraining: "
            f"{sorted(downstream_validation - pretrain_validation)}"
        )
    if downstream_validation & pretrain_train:
        raise RuntimeError("downstream validation subjects leaked into pretraining train")
    return {
        "boundary_path": str(boundary_path),
        "boundary_sha256": str(boundary["boundary_sha256"]),
        "pretraining_train_subject_count": len(pretrain_train),
        "pretraining_validation_subject_count": len(pretrain_validation),
        "downstream_train_subject_count": len(downstream_train),
        "downstream_validation_subject_count": len(downstream_validation),
    }


def _classification_weights(
    dataset: EFRMUnifiedTaskDataset,
    indices: Sequence[int],
    policy: str,
) -> torch.Tensor | None:
    if policy == "none":
        return None
    counts = np.zeros(dataset.spec.output_dim, dtype=np.float64)
    for index in indices:
        row = dataset.lightweight_metadata(int(index))
        counts[dataset.class_to_index[str(row["condition"])]] += 1.0
    if (counts <= 0).any():
        raise RuntimeError(f"training split omits a class: {counts.tolist()}")
    if policy == "inverse_frequency":
        weights = 1.0 / counts
    elif policy == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError("class weighting must be none, inverse_sqrt, or inverse_frequency")
    return torch.as_tensor(weights / weights.mean(), dtype=torch.float32)


def _loss(
    prediction: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    task_type: str,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    regression_loss: str,
) -> torch.Tensor:
    if task_type == "classification":
        return F.cross_entropy(
            prediction,
            batch["target"],
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    valid = batch["target_valid_mask"].to(dtype=prediction.dtype)
    if regression_loss == "smooth_l1":
        elementwise = F.smooth_l1_loss(
            prediction, batch["target"], reduction="none"
        )
    elif regression_loss == "mse":
        elementwise = F.mse_loss(prediction, batch["target"], reduction="none")
    else:
        raise ValueError("regression loss must be smooth_l1 or mse")
    return (elementwise * valid).sum() / valid.sum().clamp_min(1.0)


def _evaluate(
    *,
    model: EFRMDownstreamModel,
    loader: DataLoader,
    dataset: EFRMUnifiedTaskDataset,
    device: torch.device,
    autocast_context: Any,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    regression_loss: str,
    max_batches: int = 0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    losses: list[float] = []
    sample_counts: list[int] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    subjects: list[str] = []
    sample_ids: list[str] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, raw in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = move_batch(raw, device)
            with autocast_context():
                output = model(
                    batch["eeg"],
                    batch["fnirs"],
                    batch["eeg_patch_valid"],
                    batch["fnirs_patch_valid"],
                )
                loss = _loss(
                    output,
                    batch,
                    task_type=dataset.spec.task_type,
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                    regression_loss=regression_loss,
                )
            count = int(output.shape[0])
            losses.append(float(loss.detach()) * count)
            sample_counts.append(count)
            predictions.append(output.detach().float().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
            masks.append(batch["target_valid_mask"].detach().cpu().numpy())
            subjects.extend(str(value) for value in batch["subject"])
            sample_ids.extend(str(value) for value in batch["sample_id"])
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks)
    metrics: dict[str, Any] = {
        "loss": float(sum(losses) / max(1, sum(sample_counts))),
        "sample_count": int(sum(sample_counts)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if dataset.spec.task_type == "classification":
        detailed = classification_metrics(
            target,
            prediction,
            dataset.spec.class_names,
        )
        metrics.update(detailed)
        native_prediction = prediction
        native_target = target
        names = dataset.spec.class_names
    else:
        valid_float = mask.astype(np.float64)
        error = prediction - target
        metrics["masked_mae_scaled"] = float(
            (np.abs(error) * valid_float).sum() / max(1.0, valid_float.sum())
        )
        metrics["masked_rmse_scaled"] = float(
            np.sqrt((np.square(error) * valid_float).sum() / max(1.0, valid_float.sum()))
        )
        if dataset.target_center is None or dataset.target_scale is None:
            raise RuntimeError("regression evaluation requires a train-only target scaler")
        center = dataset.target_center[None]
        scale = dataset.target_scale[None]
        native_prediction = prediction * scale + center
        native_target = target * scale + center
        detailed = regression_metrics(
            native_target,
            native_prediction,
            mask,
            dataset.spec.target_names,
        )
        metrics.update({f"native_{key}": value for key, value in detailed.items()})
        names = dataset.spec.target_names
    per_subject = subject_metrics(
        subjects=subjects,
        task_type=dataset.spec.task_type,
        target=native_target,
        prediction=native_prediction,
        valid_mask=mask,
        names=names,
    )
    metrics["subject_metrics"] = per_subject
    evidence = {
        "prediction": native_prediction,
        "target": native_target,
        "target_valid_mask": mask,
        "subject": np.asarray(subjects, dtype=str),
        "sample_id": np.asarray(sample_ids, dtype=str),
    }
    return metrics, evidence


def _selection(metrics: Mapping[str, Any], task_type: str) -> tuple[str, str, float]:
    metric = "macro_f1" if task_type == "classification" else "masked_rmse_scaled"
    mode = "max" if task_type == "classification" else "min"
    value = float(metrics[metric])
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite selection metric {metric}: {value}")
    return metric, mode, value


def _improved(value: float, best: float, mode: str) -> bool:
    return value > best if mode == "max" else value < best


def _transfer_state(model: EFRMDownstreamModel) -> dict[str, torch.Tensor]:
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _checkpoint_payload(
    *,
    model: EFRMDownstreamModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    optimizer_step: int,
    best_metric: float,
    selection_metric: str,
    selection_mode: str,
    target_scaler: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "task": args.task,
        "transfer_mode": args.transfer_mode,
        "modality": args.modality,
        "initialization": args.initialization,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "transfer_state": _transfer_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "best_validation_metric": float(best_metric),
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "target_scaler": target_scaler,
    }


def _handle_stop(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def run(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(
        status_path,
        {"schema": SCHEMA, "status": "initializing", "pid": os.getpid(), "started_at": utc_now()},
    )
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError(f"expected config schema {SCHEMA}")
    task_cfg = dict(config.get("task_overrides", {}).get(args.task, {}))
    train_cfg = {**config["training"], **task_cfg}
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(train_cfg.get("torch_cpu_threads", 2)))
    if not torch.cuda.is_available():
        raise RuntimeError("EFRM downstream training requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    spec = get_task_spec(args.task)
    dataset = EFRMUnifiedTaskDataset(
        spec,
        cache_root=str(config["data"]["cache_root"]),
    )
    split_path = Path(args.split_manifest).resolve()
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    train_indices, validation_indices = dataset.validate_shared_public_split(split_path)
    target_scaler = (
        dataset.fit_target_scaler(train_indices)
        if spec.task_type == "regression"
        else None
    )

    backbone = _build_backbone(config)
    pretraining_boundary: dict[str, Any] | None = None
    checkpoint_sha256: str | None = None
    if args.initialization == "pretrained":
        if not args.pretrained_checkpoint:
            raise ValueError("pretrained initialization requires --pretrained-checkpoint")
        pretrain_path = Path(args.pretrained_checkpoint).resolve()
        checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "efrm_sync_pretrain_checkpoint_v1":
            raise ValueError("unsupported EFRM pretraining checkpoint schema")
        backbone.load_state_dict(checkpoint["model"], strict=True)
        pretraining_boundary = _validate_pretraining_boundary(
            pretrain_path,
            checkpoint,
            dataset,
            train_indices,
            validation_indices,
        )
        checkpoint_sha256 = sha256_file(pretrain_path)
        del checkpoint
    elif args.initialization != "scratch":
        raise ValueError("initialization must be pretrained or scratch")

    model = EFRMDownstreamModel(
        backbone,
        output_dim=spec.output_dim,
        modality=args.modality,
        target_length=spec.target_length,
        dropout=float(train_cfg.get("dropout", 0.5)),
    )
    model.configure_transfer(args.transfer_mode)
    model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    if not trainable_parameters:
        raise RuntimeError("transfer configuration produced no trainable parameters")

    batch_size = int(train_cfg.get(
        "linear_probe_batch_size"
        if args.transfer_mode == "linear_probe"
        else "full_finetune_batch_size",
        train_cfg.get("batch_size", 16),
    ))
    workers = int(train_cfg.get("num_workers", 0))
    train_loader, train_sampler = make_loader(
        dataset,
        train_indices,
        batch_size=batch_size,
        workers=workers,
        shuffle=True,
        seed=seed,
    )
    validation_loader, _ = make_loader(
        dataset,
        validation_indices,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
    )
    class_weights = (
        _classification_weights(
            dataset,
            train_indices,
            str(train_cfg.get("class_weighting", "none")),
        ).to(device)
        if spec.task_type == "classification"
        and str(train_cfg.get("class_weighting", "none")) != "none"
        else None
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_cfg.get(
            "linear_probe_lr" if args.transfer_mode == "linear_probe" else "full_finetune_lr",
            1e-3 if args.transfer_mode == "linear_probe" else 1e-4,
        )),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
        fused=bool(train_cfg.get("fused_optimizer", True)),
    )
    epochs = int(
        args.epochs
        if args.epochs is not None
        else train_cfg.get(
            "linear_probe_epochs"
            if args.transfer_mode == "linear_probe"
            else "full_finetune_epochs",
            train_cfg.get("epochs", 20),
        )
    )
    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=float(train_cfg.get("minimum_lr", 1e-6)),
    )
    amp_enabled = bool(train_cfg.get("amp", True))
    amp_dtype = (
        torch.bfloat16
        if str(train_cfg.get("amp_dtype", "bfloat16")) == "bfloat16"
        else torch.float16
    )

    def autocast_context() -> Any:
        return (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )

    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    start_epoch = 1
    optimizer_step = 0
    best_metric = -math.inf if spec.task_type == "classification" else math.inf
    if args.resume:
        resumed = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        for field, expected in (
            ("schema", SCHEMA),
            ("task", args.task),
            ("transfer_mode", args.transfer_mode),
            ("modality", args.modality),
            ("initialization", args.initialization),
        ):
            if resumed.get(field) != expected:
                raise ValueError(f"resume checkpoint {field} mismatch")
        model.load_state_dict(resumed["transfer_state"], strict=False)
        optimizer.load_state_dict(resumed["optimizer_state"])
        scheduler.load_state_dict(resumed["scheduler_state"])
        grad_scaler.load_state_dict(resumed["grad_scaler_state"])
        start_epoch = int(resumed["epoch"]) + 1
        optimizer_step = int(resumed["optimizer_step"])
        best_metric = float(resumed["best_validation_metric"])

    output_dir.joinpath("resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "adapter_manifest.json", dataset.adapter.manifest())
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "scope": "public_development_pilot",
        "protected_test_opened": False,
        "task": asdict(spec),
        "transfer_mode": args.transfer_mode,
        "modality": args.modality,
        "initialization": args.initialization,
        "pid": os.getpid(),
        "device": str(device),
        "started_at": utc_now(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "pretraining_boundary": pretraining_boundary,
        "train_sample_count": len(train_indices),
        "validation_sample_count": len(validation_indices),
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": batch_size,
        "train_batch_count": len(train_loader),
        "validation_batch_limit": int(args.max_validation_batches or 0),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(status_path, manifest)

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    regression_loss = str(train_cfg.get("regression_loss", "smooth_l1"))
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))
    max_steps = int(args.max_steps or 0)
    step_limit_reached = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if args.transfer_mode == "linear_probe":
            model.backbone.eval()
        train_sampler.epoch = epoch
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_started = time.perf_counter()
        for raw in train_loader:
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                output = model(
                    batch["eeg"],
                    batch["fnirs"],
                    batch["eeg_patch_valid"],
                    batch["fnirs_patch_valid"],
                )
                loss = _loss(
                    output,
                    batch,
                    task_type=spec.task_type,
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                    regression_loss=regression_loss,
                )
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            optimizer_step += 1
            count = int(output.shape[0])
            epoch_loss += float(loss.detach()) * count
            epoch_samples += count
            append_jsonl(
                output_dir / "metrics" / "train_steps.jsonl",
                {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "loss": float(loss.detach()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(grad_norm),
                    "batch_size": count,
                },
            )
            if max_steps and optimizer_step >= max_steps:
                step_limit_reached = True
                break
            if _STOP_REQUESTED:
                break

        validation_metrics, _ = _evaluate(
            model=model,
            loader=validation_loader,
            dataset=dataset,
            device=device,
            autocast_context=autocast_context,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            regression_loss=regression_loss,
            max_batches=int(args.max_validation_batches or 0),
        )
        selection_metric, selection_mode, current_metric = _selection(
            validation_metrics, spec.task_type
        )
        is_best = _improved(current_metric, best_metric, selection_mode)
        if is_best:
            best_metric = current_metric
        epoch_row = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "train_loss": epoch_loss / max(1, epoch_samples),
            "train_sample_count": epoch_samples,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "is_best": is_best,
            **validation_metrics,
        }
        append_jsonl(output_dir / "metrics" / "validation_epochs.jsonl", epoch_row)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=grad_scaler,
            epoch=epoch,
            optimizer_step=optimizer_step,
            best_metric=best_metric,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
            target_scaler=target_scaler,
            args=args,
        )
        _atomic_checkpoint(output_dir / "checkpoint_latest.pt", payload)
        if is_best:
            _atomic_checkpoint(output_dir / "checkpoint_best.pt", payload)
        last_epoch = epoch
        write_json(
            status_path,
            {
                **manifest,
                "status": "running",
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "best_validation_metric": best_metric,
                "selection_metric": selection_metric,
                "updated_at": utc_now(),
            },
        )
        if _STOP_REQUESTED or step_limit_reached:
            break

    best = torch.load(output_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["transfer_state"], strict=False)
    final_metrics, evidence = _evaluate(
        model=model,
        loader=validation_loader,
        dataset=dataset,
        device=device,
        autocast_context=autocast_context,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        regression_loss=regression_loss,
        max_batches=int(args.max_validation_batches or 0),
    )
    np.savez_compressed(output_dir / "validation_predictions.npz", **evidence)
    write_json(output_dir / "validation_metrics.json", final_metrics)
    final_status = (
        "interrupted_checkpointed"
        if _STOP_REQUESTED
        else "step_limit_reached"
        if step_limit_reached
        else "completed"
    )
    manifest.update(
        {
            "status": final_status,
            "completed_at": utc_now(),
            "last_epoch": last_epoch,
            "optimizer_steps": optimizer_step,
            "selection_metric": best["selection_metric"],
            "selection_mode": best["selection_mode"],
            "best_validation_metric": best_metric,
            "best_epoch": int(best["epoch"]),
            "validation_metrics": final_metrics,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(status_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument(
        "--transfer-mode",
        required=True,
        choices=("linear_probe", "full_finetune"),
    )
    parser.add_argument("--modality", default="paired", choices=("eeg", "fnirs", "paired"))
    parser.add_argument(
        "--initialization", default="pretrained", choices=("pretrained", "scratch")
    )
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    try:
        run(parsed)
    except Exception as error:
        output = Path(parsed.output_dir).resolve()
        write_json(
            output / "status.json",
            {
                "schema": SCHEMA,
                "status": "failed",
                "pid": os.getpid(),
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
