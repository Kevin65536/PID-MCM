#!/usr/bin/env python3
"""Train one PyTorch STA-Net task with grouped loading and validation."""

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
import yaml
from torch.utils.data import DataLoader, Sampler

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch import (
    STANet,
    STANetConfig,
    STANetObjective,
    STANetUnifiedTaskDataset,
    collate_sta_net,
    get_sta_net_task_spec,
    task_contract_sha256,
)
from sta_net_pytorch.metrics import classification_metrics_from_confusion, improved, selection_value
from sta_net_pytorch.splits import development_subject_split, validate_public_manifest

SCHEMA = "sta_net_pytorch_training_v2"
_STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item() if value.numel() == 1 else value.detach().float().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")
        handle.flush()


class RecordGroupedBatchSampler(Sampler[list[int]]):
    """Keep windows from one record together so the two-record loader cache remains effective."""

    def __init__(
        self,
        dataset: STANetUnifiedTaskDataset,
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
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


class PackedRecordBatchSampler(RecordGroupedBatchSampler):
    """Keep record samples adjacent while filling batches across record boundaries."""

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        keys = sorted(self.groups)
        if self.shuffle:
            rng.shuffle(keys)
        ordered: list[int] = []
        for key in keys:
            indices = list(self.groups[key])
            if self.shuffle:
                rng.shuffle(indices)
            ordered.extend(indices)
        for start in range(0, len(ordered), self.batch_size):
            yield ordered[start : start + self.batch_size]
        self.epoch += 1

    def __len__(self) -> int:
        return math.ceil(sum(len(indices) for indices in self.groups.values()) / self.batch_size)


def grouped_subject_split(dataset: STANetUnifiedTaskDataset, seed: int) -> tuple[list[int], list[int], dict[str, Any]]:
    return development_subject_split(dataset, seed)


def make_loader(
    dataset: STANetUnifiedTaskDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    prefetch_factor: int = 2,
    pack_record_batches: bool = False,
) -> tuple[DataLoader, RecordGroupedBatchSampler]:
    sampler_class = PackedRecordBatchSampler if pack_record_batches else RecordGroupedBatchSampler
    sampler = sampler_class(
        dataset, indices, batch_size=batch_size, shuffle=shuffle, seed=seed
    )
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": collate_sta_net,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": max(1, int(prefetch_factor))})
    return DataLoader(dataset, **kwargs), sampler


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def checkpoint_payload(
    *,
    spec: Any,
    model_config: STANetConfig,
    model: STANet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    grad_scaler: torch.amp.GradScaler,
    epoch: int,
    optimizer_step: int,
    target_scaler: Mapping[str, Any] | None,
    best_validation_loss: float,
    selection_metric: str,
    selection_mode: str,
    best_validation_metric: float,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "task": asdict(spec),
        "model_config": asdict(model_config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "grad_scaler_state": grad_scaler.state_dict(),
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "target_scaler": target_scaler,
        "best_validation_loss": best_validation_loss,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "best_validation_metric": best_validation_metric,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }


def evaluate(
    *,
    model: STANet,
    objective: STANetObjective,
    loader: DataLoader,
    device: torch.device,
    task_type: str,
    class_count: int,
    autocast_context: Any,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    correct = 0
    confusion = np.zeros((class_count, class_count), dtype=np.int64) if task_type == "classification" else None
    valid_count = 0.0
    absolute_error = 0.0
    squared_error = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with autocast_context():
                outputs = model(batch["eeg"], batch["fnirs"])
                losses = objective(
                    outputs,
                    batch["target"],
                    batch["target_valid_mask"] if task_type == "regression" else None,
                )
            current_batch = int(batch["eeg"].shape[0])
            loss_sum += float(losses["total"].detach()) * current_batch
            sample_count += current_batch
            prediction = outputs["prediction"]
            if task_type == "classification":
                predicted = prediction.argmax(dim=-1)
                correct += int((predicted == batch["target"]).sum().detach())
                truth_np = batch["target"].detach().cpu().numpy().astype(np.int64)
                predicted_np = predicted.detach().cpu().numpy().astype(np.int64)
                np.add.at(confusion, (truth_np, predicted_np), 1)
            else:
                weights = batch["target_valid_mask"].to(dtype=prediction.dtype)
                error = prediction - batch["target"]
                valid_count += float(weights.sum().detach())
                absolute_error += float((error.abs() * weights).sum().detach())
                squared_error += float((error.square() * weights).sum().detach())
    metrics = {
        "loss": loss_sum / max(1, sample_count),
        "sample_count": float(sample_count),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if task_type == "classification":
        metrics.update(classification_metrics_from_confusion(confusion))
    else:
        metrics["masked_mae_scaled"] = absolute_error / max(1.0, valid_count)
        metrics["masked_rmse_scaled"] = math.sqrt(squared_error / max(1.0, valid_count))
        metrics["valid_coordinate_count"] = valid_count
    return metrics


def _handle_stop(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def classification_weights(
    dataset: STANetUnifiedTaskDataset,
    indices: Sequence[int],
    policy: str,
) -> torch.Tensor | None:
    if policy == "none":
        return None
    counts = np.zeros(dataset.spec.output_dim, dtype=np.float64)
    class_to_index = {name: index for index, name in enumerate(dataset.spec.class_names)}
    for index in indices:
        condition = str(dataset.lightweight_metadata(int(index))["condition"])
        counts[class_to_index[condition]] += 1.0
    if (counts <= 0).any():
        raise RuntimeError(f"training split omits a class: counts={counts.tolist()}")
    if policy == "inverse_frequency":
        weights = 1.0 / counts
    elif policy == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError("class_weighting must be none, inverse_sqrt, or inverse_frequency")
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    warmup_ratio: float,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if name == "constant" and warmup_ratio <= 0.0:
        return None
    if name not in {"constant", "cosine"}:
        raise ValueError("scheduler must be constant or cosine")
    warmup_steps = int(round(total_steps * warmup_ratio))
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        if name == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"schema": SCHEMA, "status": "initializing", "pid": os.getpid(), "started_at": utc_now()})
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config_path = Path(args.config).resolve()
    if config.get("schema") != SCHEMA:
        raise ValueError(f"Expected config schema {SCHEMA}")
    spec = get_sta_net_task_spec(args.task)
    task_cfg = dict(config.get("task_overrides", {}).get(args.task, {}))
    train_cfg = {**config.get("training", {}), **task_cfg}
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(train_cfg.get("torch_cpu_threads", 2)))
    if not torch.cuda.is_available():
        raise RuntimeError("STA-Net training requires CUDA for this protocol")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = STANetUnifiedTaskDataset(
        spec, cache_root=str(config["data"]["cache_root"]),
        adapted_cache_size=int(train_cfg.get("adapted_sample_cache_size", 0)),
    )
    if args.split_manifest:
        split_path = Path(args.split_manifest).resolve()
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        train_indices, validation_indices = validate_public_manifest(dataset, split_manifest)
    else:
        split_path = None
        train_indices, validation_indices, split_manifest = grouped_subject_split(dataset, seed)
    scaler = dataset.fit_regression_target_scaler(train_indices) if spec.task_type == "regression" else None
    split_manifest["regression_target_scaler"] = scaler
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "adapter_manifest.json", dataset.adapter.manifest())

    batch_size = int(train_cfg.get("batch_size", 32))
    workers = int(train_cfg.get("num_workers", 2))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    pack_record_batches = bool(train_cfg.get("pack_record_batches", False))
    train_loader, train_sampler = make_loader(
        dataset, train_indices, batch_size=batch_size, workers=workers, shuffle=True, seed=seed,
        prefetch_factor=prefetch_factor,
        pack_record_batches=pack_record_batches,
    )
    validation_loader, _ = make_loader(
        dataset, validation_indices, batch_size=batch_size, workers=workers, shuffle=False, seed=seed,
        prefetch_factor=prefetch_factor,
        pack_record_batches=pack_record_batches,
    )

    model_cfg = config.get("model", {})
    resolved_model = STANetConfig(
        task_type=spec.task_type,
        output_dim=spec.output_dim,
        sequence_length=spec.target_length,
        dropout=float(model_cfg.get("dropout", 0.5)),
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        attention_heads=int(model_cfg.get("attention_heads", 10)),
        attention_key_dim=int(model_cfg.get("attention_key_dim", 256)),
        max_lags=spec.fnirs_lag_count,
    )
    model = STANet(resolved_model).to(device)
    loss_cfg = config.get("loss", {})
    class_weight_policy = str(loss_cfg.get("class_weighting", "none"))
    class_weight_tensor = (
        classification_weights(dataset, train_indices, class_weight_policy)
        if spec.task_type == "classification" else None
    )
    if class_weight_tensor is not None:
        class_weight_tensor = class_weight_tensor.to(device)
    objective = STANetObjective(
        spec.task_type,
        main_weight=float(loss_cfg.get("main_weight", 1.0)),
        eeg_aux_weight=float(loss_cfg.get("eeg_aux_weight", 1.0)),
        alignment_weight=float(loss_cfg.get("alignment_weight", 1.0)),
        regression_loss=str(loss_cfg.get("regression_loss", "smooth_l1")),
        class_weights=class_weight_tensor,
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        fused=bool(train_cfg.get("fused_optimizer", True)),
    )
    amp_enabled = bool(train_cfg.get("amp", True))
    amp_name = str(train_cfg.get("amp_dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    def autocast_context() -> Any:
        return torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 40))
    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", 0))
    validation_every = int(train_cfg.get("validation_every_epochs", 1))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    log_every = int(train_cfg.get("log_every_steps", 10))
    selection_metric = str(train_cfg.get(
        "selection_metric", "macro_f1" if spec.task_type == "classification" else "masked_rmse_scaled"
    ))
    selection_mode = str(train_cfg.get(
        "selection_mode", "max" if spec.task_type == "classification" else "min"
    ))
    selection_min_delta = float(train_cfg.get("selection_min_delta", 0.0))
    scheduler_name = str(train_cfg.get("scheduler", "constant"))
    scheduler_total_epochs = int(train_cfg.get("scheduler_total_epochs", max(epochs, 100)))
    scheduler = make_scheduler(
        optimizer, name=scheduler_name, warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        total_steps=max(1, scheduler_total_epochs * len(train_loader)),
    )
    global_step = 0
    start_epoch = 1
    best_validation_loss = math.inf
    best_validation_metric = math.inf if selection_mode == "min" else -math.inf
    if args.resume:
        resumed = torch.load(Path(args.resume), map_location=device, weights_only=False)
        if resumed.get("schema") != SCHEMA or resumed.get("task", {}).get("key") != spec.key:
            raise ValueError("Resume checkpoint schema/task does not match this run")
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        if scheduler is not None and resumed.get("scheduler_state") is not None:
            scheduler.load_state_dict(resumed["scheduler_state"])
        if resumed.get("grad_scaler_state"):
            grad_scaler.load_state_dict(resumed["grad_scaler_state"])
        start_epoch = int(resumed["epoch"]) + 1
        global_step = int(resumed["optimizer_step"])
        best_validation_loss = float(resumed.get("best_validation_loss", math.inf))
        best_validation_metric = float(resumed.get("best_validation_metric", best_validation_metric))
        rng = resumed.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            # map_location=device also moves serialized RNG ByteTensors to CUDA,
            # while both RNG restoration APIs require CPU ByteTensors.
            torch.set_rng_state(rng["torch"].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])

    manifest = {
        "schema": SCHEMA,
        "task": asdict(spec),
        "task_contract_sha256": task_contract_sha256(spec),
        "pid": os.getpid(),
        "device": str(device),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "tensorflow_used": False,
        "model_config": asdict(resolved_model),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": workers,
        "prefetch_factor": prefetch_factor,
        "adapted_sample_cache_size": int(train_cfg.get("adapted_sample_cache_size", 0)),
        "pack_record_batches": pack_record_batches,
        "sampling": "record_grouped_batches_v1",
        "amp": amp_enabled,
        "amp_dtype": amp_name if amp_enabled else None,
        "tf32": True,
        "protected_test_opened": False,
        "validation_enabled": validation_every > 0,
        "implementation_sha256": {
            "trainer": sha256(Path(__file__).resolve()),
            "model": sha256(METHOD_ROOT / "sta_net_pytorch" / "model.py"),
            "adapter": sha256(METHOD_ROOT / "sta_net_pytorch" / "data.py"),
            "config": sha256(config_path),
        },
        "resume_checkpoint": args.resume,
        "split_manifest_source": None if split_path is None else str(split_path),
        "split_sha256": split_manifest.get("split_sha256"),
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "class_weighting": class_weight_policy,
        "label_smoothing": float(loss_cfg.get("label_smoothing", 0.0)),
        "scheduler": scheduler_name,
        "started_at": utc_now(),
    }
    (output_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({**config, "resolved_task": args.task, "resolved_training": train_cfg}, sort_keys=False),
        encoding="utf-8",
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(status_path, {**manifest, "status": "data_ready", "optimizer_step": global_step})
    print(json.dumps({"status": "data_ready", "task": args.task, "train_samples": len(train_indices)}), flush=True)

    last_epoch = start_epoch - 1
    step_limit_reached = False
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        train_sampler.epoch = epoch - 1
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_started = time.perf_counter()
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            step_started = time.perf_counter()
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                outputs = model(batch["eeg"], batch["fnirs"])
                losses = objective(
                    outputs,
                    batch["target"],
                    batch["target_valid_mask"] if spec.task_type == "regression" else None,
                )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite training loss at step {global_step + 1}")
            grad_scaler.scale(losses["total"]).backward()
            grad_scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            current_batch = int(batch["eeg"].shape[0])
            loss_value = float(losses["total"].detach())
            epoch_loss += loss_value * current_batch
            epoch_samples += current_batch
            event = {
                "time": utc_now(), "task": args.task, "epoch": epoch, "batch": batch_index,
                "optimizer_step": global_step, "loss": loss_value, "main_loss": losses["main"],
                "eeg_aux_loss": losses["eeg_aux"], "alignment_loss": losses["alignment"],
                "gradient_norm_before_clip": gradient_norm, "batch_size": current_batch,
                "step_seconds": time.perf_counter() - step_started,
            }
            if global_step == 1 or global_step % log_every == 0:
                append_jsonl(output_dir / "metrics" / "train_steps.jsonl", event)
                write_json(status_path, {**manifest, "status": "training", **event})
                print(json.dumps(jsonable(event), sort_keys=True), flush=True)
            if _STOP_REQUESTED or (max_steps > 0 and global_step >= max_steps):
                step_limit_reached = max_steps > 0 and global_step >= max_steps
                break

        elapsed = time.perf_counter() - epoch_started
        epoch_event = {
            "time": utc_now(), "epoch": epoch, "optimizer_step": global_step,
            "mean_train_loss": epoch_loss / max(1, epoch_samples), "sample_count": epoch_samples,
            "elapsed_seconds": elapsed, "samples_per_second": epoch_samples / max(elapsed, 1e-9),
            "partial_epoch": bool(_STOP_REQUESTED or step_limit_reached),
        }
        append_jsonl(output_dir / "metrics" / "train_epochs.jsonl", epoch_event)
        validation_metrics = None
        if validation_every > 0 and not _STOP_REQUESTED and not step_limit_reached and epoch % validation_every == 0:
            validation_metrics = evaluate(
                model=model, objective=objective, loader=validation_loader, device=device,
                task_type=spec.task_type, autocast_context=autocast_context,
                class_count=spec.output_dim,
            )
            validation_metrics.update({"time": utc_now(), "epoch": epoch, "optimizer_step": global_step})
            append_jsonl(output_dir / "metrics" / "validation_epochs.jsonl", validation_metrics)
        validation_improved = False
        if validation_metrics is not None:
            best_validation_loss = min(best_validation_loss, float(validation_metrics["loss"]))
            current_selection = selection_value(validation_metrics, selection_metric, selection_mode)
            validation_improved = improved(
                current_selection, best_validation_metric, selection_mode, selection_min_delta
            )
            if validation_improved:
                best_validation_metric = current_selection
        payload = checkpoint_payload(
            spec=spec, model_config=resolved_model, model=model, optimizer=optimizer,
            scheduler=scheduler, grad_scaler=grad_scaler,
            epoch=epoch, optimizer_step=global_step, target_scaler=scaler,
            best_validation_loss=best_validation_loss,
            selection_metric=selection_metric, selection_mode=selection_mode,
            best_validation_metric=best_validation_metric,
        )
        atomic_checkpoint(output_dir / "checkpoint_latest.pt", payload)
        if validation_improved:
            atomic_checkpoint(output_dir / "checkpoint_best.pt", payload)
        if _STOP_REQUESTED or step_limit_reached:
            break

    if _STOP_REQUESTED:
        final_status = "interrupted_checkpointed"
    elif step_limit_reached:
        final_status = "step_limit_reached"
    else:
        final_status = "completed"
    manifest.update({
        "status": final_status, "completed_at": utc_now(), "optimizer_steps": global_step,
        "last_epoch": last_epoch, "best_validation_loss": None if math.isinf(best_validation_loss) else best_validation_loss,
        "selection_metric": selection_metric, "selection_mode": selection_mode,
        "best_validation_metric": None if math.isinf(best_validation_metric) else best_validation_metric,
    })
    write_json(output_dir / "manifest.json", manifest)
    write_json(status_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=(
        "motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual", "refed_regression"
    ))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--split-manifest", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    try:
        run(parsed)
    except Exception as error:
        output = Path(parsed.output_dir).resolve()
        write_json(output / "status.json", {
            "schema": SCHEMA, "status": "failed", "pid": os.getpid(), "failed_at": utc_now(),
            "error_type": type(error).__name__, "error": str(error),
        })
        raise
