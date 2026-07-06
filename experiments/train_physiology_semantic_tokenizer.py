#!/usr/bin/env python3
"""Gate-aware full trainer for the physiology-semantic tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.factory import create_configured_multimodal_dataloaders
from src.losses.physiology_semantic import PhysiologySemanticLoss
from src.teachers.physical_state_teacher import PhysicalStateTeacher
from src.tokenizers.registry import create_tokenizer
import src.tokenizers  # noqa: F401  # active registry side effects


RUN_SCHEMA = "physiology_semantic_training_v2"
E0_SCHEMA = "physiology_semantic_e0_v1"
EEG_COORDINATES = ("r_mean", "r_slope", "r_logvar", "s_mean", "s_slope", "s_logvar")
FNIRS_COORDINATES = (
    "delta_f_mean", "delta_hbo_mean", "delta_hb_mean",
    "delta_f_slope", "delta_hbo_slope", "delta_hb_slope",
    "delta_f_logvar", "delta_hbo_logvar", "delta_hb_logvar",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(payload)), sort_keys=True) + "\n")


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_run_dir(config: Mapping[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    group = experiment.get("run_group", "physiology_semantic_tokenizer/training")
    name = experiment.get("name", "tokenizer_training")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "experiments" / "runs" / str(group) / f"{stamp}_{name}"


def _resolve_device(training: Mapping[str, Any]) -> torch.device:
    requested = str(training.get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _dataset_subjects(dataset: Any) -> set[int]:
    if hasattr(dataset, "entries"):
        return {int(entry.subject_id) for entry in dataset.entries}
    if hasattr(dataset, "sources"):
        subjects: set[int] = set()
        for source in dataset.sources:
            subjects.update(_dataset_subjects(source["dataset"]))
        return subjects
    raise TypeError(f"Cannot audit subjects for dataset type {type(dataset).__name__}")


def _validate_loader_subjects(dataloaders: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    declared = config.get("data", {}).get("split", {})
    for split, key in (("train", "train_subjects"), ("val", "val_subjects"), ("test", "test_subjects")):
        expected = {int(value) for value in declared.get(key, [])}
        observed = _dataset_subjects(dataloaders[split].dataset)
        if observed != expected:
            raise RuntimeError(
                f"{split} cache coverage mismatch: expected subjects {sorted(expected)}, observed {sorted(observed)}"
            )


def _load_e0_gate(config: Mapping[str, Any], *, require_pass: bool) -> tuple[dict[str, Any] | None, str | None]:
    gate_value = config.get("validation", {}).get("e0_gate_path")
    if not gate_value:
        if require_pass:
            raise RuntimeError("Training requires validation.e0_gate_path; boolean e0_passed is not accepted")
        return None, None
    path = Path(gate_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"E0 gate file not found: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("schema") != E0_SCHEMA or gate.get("gate") != "G0":
        raise ValueError(f"Unsupported E0 gate schema in {path}")
    expected_split = hashlib.sha256(
        json.dumps(config.get("data", {}).get("split", {}), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if gate.get("split_sha256") != expected_split:
        raise ValueError("E0 gate subject split does not match the training configuration")
    if gate.get("data_contract") != config.get("data", {}).get("contract"):
        raise ValueError("E0 gate data contract does not match the training configuration")
    expected_roots = [source.get("root") for source in config.get("data", {}).get("cache_sources", [])]
    if gate.get("cache_source_roots") != expected_roots:
        raise ValueError("E0 gate cache sources do not match the training configuration")
    if require_pass and not bool(gate.get("e0_passed", False)):
        raise RuntimeError(f"E0 gate did not pass: {gate.get('status', 'unknown')}")
    return gate, _sha256(path)


def _coordinate_mask(names: tuple[str, ...], admitted: Iterable[str] | None) -> torch.Tensor:
    if admitted is None:
        return torch.ones(len(names), dtype=torch.bool)
    admitted = set(admitted)
    return torch.tensor([name in admitted for name in names], dtype=torch.bool)


def _teacher_supervision_requested(config: Mapping[str, Any]) -> bool:
    loss = config.get("loss", {})
    return any(
        float(loss.get(name, {}).get("weight", 0.0)) > 0.0
        for name in ("state", "prototype", "masked_state")
    )


def _loss_from_config(config: Mapping[str, Any], gate: Mapping[str, Any] | None) -> PhysiologySemanticLoss:
    loss = config.get("loss", {})
    admitted = None if gate is None else gate.get("admissible_coordinates", {})
    eeg_admitted = None if admitted is None else admitted.get("eeg", [])
    fnirs_admitted = None if admitted is None else admitted.get("fnirs", [])
    return PhysiologySemanticLoss(
        state_weight=loss.get("state", {}).get("weight", 1.0),
        prototype_weight=loss.get("prototype", {}).get("weight", 1.0),
        masked_state_weight=loss.get("masked_state", {}).get("weight", 1.0),
        reconstruction_weight=loss.get("reconstruction", {}).get("weight", 1.0),
        vq_weight=loss.get("vq", {}).get("weight", 1.0),
        private_weight=loss.get("private", {}).get("weight", 0.0),
        eeg_coordinate_mask=_coordinate_mask(EEG_COORDINATES, eeg_admitted),
        fnirs_coordinate_mask=_coordinate_mask(FNIRS_COORDINATES, fnirs_admitted),
    )


def _scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    teacher_adapter: PhysicalStateTeacher,
    criterion: PhysiologySemanticLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    grad_clip: float,
    global_step: int,
    max_steps: int | None,
) -> tuple[dict[str, float], int, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    sample_count = 0
    last_health: dict[str, Any] = {}
    for batch in loader:
        if max_steps is not None and global_step >= max_steps:
            break
        batch = _move_to_device(batch, device)
        batch_size = int(batch["eeg"].shape[0])
        teacher = teacher_adapter(batch["teacher"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), _amp_context(device, amp_enabled):
            outputs = model(batch["eeg"], batch["fnirs"])
            losses = criterion(outputs, teacher)
        if training:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch_size
        sample_count += batch_size
        last_health = {
            "eeg": outputs["eeg"].quantizer.health,
            "fnirs": outputs["fnirs"].quantizer.health,
        }
    if sample_count == 0:
        raise RuntimeError("Epoch consumed zero samples")
    return {key: value / sample_count for key, value in sums.items()}, global_step, last_health


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    best_validation: float,
    epochs_without_improvement: int,
    e0_gate_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": RUN_SCHEMA,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": dict(config),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation": best_validation,
            "epochs_without_improvement": epochs_without_improvement,
            "e0_gate_sha256": e0_gate_hash,
        },
        path,
    )


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.e0_gate:
        config.setdefault("validation", {})["e0_gate_path"] = args.e0_gate
    if args.smoke_optimizer_steps is not None:
        config.setdefault("training", {})["smoke_optimizer_steps"] = args.smoke_optimizer_steps
    training = config.get("training", {})
    seed = int(training.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    optimizer_requested = bool(args.train or (args.smoke and int(training.get("smoke_optimizer_steps", 0)) > 0))
    teacher_supervision = _teacher_supervision_requested(config)
    gate, gate_hash = _load_e0_gate(
        config, require_pass=bool(optimizer_requested and teacher_supervision)
    )
    device = _resolve_device(training)
    run_dir = Path(args.output_dir).resolve() if args.output_dir else _default_run_dir(config)
    for relative in ("checkpoints", "metrics", "diagnostics"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    protocol = {
        "schema": RUN_SCHEMA,
        "selection_metric": "validation total loss",
        "stopping_rule": "validation early stopping with configured patience",
        "protected_test_policy": "test split is never evaluated by the trainer",
        "e0_gate_sha256": gate_hash,
        "objective": "teacher_supervised" if teacher_supervision else "teacher_free",
    }
    (run_dir / "decision_protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "metric_registry.json", {
        "primary": "validation_total_loss", "training": list(config.get("loss", {})),
        "diagnostic": ["quantizer_health", "learning_rate"],
    })
    _write_json(run_dir / "evidence_calibration.json", {
        "source": "E0 gate", "e0_gate_sha256": gate_hash, "protected_test_opened": False,
    })

    dataloaders = create_configured_multimodal_dataloaders(config)
    _validate_loader_subjects(dataloaders, config)
    model = create_tokenizer(config).to(device)
    teacher_adapter = PhysicalStateTeacher().to(device)
    criterion = _loss_from_config(config, gate).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        betas=tuple(training.get("betas", [0.9, 0.98])),
    )

    epochs = int(training.get("epochs", 1))
    steps_per_epoch = max(len(dataloaders["train"]), 1)
    total_steps = max(epochs * steps_per_epoch, 1)
    smoke_steps = int(training.get("smoke_optimizer_steps", 0))
    max_steps = smoke_steps if args.smoke and smoke_steps > 0 else None
    if max_steps is not None:
        total_steps = max_steps
        epochs = max(1, math.ceil(max_steps / steps_per_epoch))
    scheduler = _scheduler(optimizer, int(training.get("warmup_steps", 0)), total_steps)
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    best_validation = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("schema") != RUN_SCHEMA:
            raise ValueError("Resume checkpoint schema mismatch")
        if checkpoint.get("e0_gate_sha256", "") != (gate_hash or ""):
            raise ValueError("Resume checkpoint E0 gate differs from current gate")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint["best_validation"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if max_steps is not None and global_step < max_steps and start_epoch >= epochs:
            epochs = start_epoch + 1

    start_time = time.time()
    status = "dry_run_passed"
    last_health: dict[str, Any] = {}
    if args.dry_run or (args.smoke and not optimizer_requested):
        train_metrics, _, last_health = _run_epoch(
            model=model,
            loader=[next(iter(dataloaders["train"]))],
            teacher_adapter=teacher_adapter,
            criterion=criterion,
            device=device,
            optimizer=None,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip=0.0,
            global_step=0,
            max_steps=None,
        )
        _append_jsonl(run_dir / "metrics" / "train.jsonl", {"epoch": 0, **train_metrics})
        status = "dry_run_passed" if args.dry_run else "smoke_passed_optimizer_not_requested"
    else:
        patience = int(training.get("early_stopping_patience", 10))
        min_delta = float(training.get("early_stopping_min_delta", 0.0))
        grad_clip = float(training.get("grad_clip_norm", 1.0))
        for epoch in range(start_epoch, epochs):
            train_metrics, global_step, last_health = _run_epoch(
                model=model,
                loader=dataloaders["train"],
                teacher_adapter=teacher_adapter,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip=grad_clip,
                global_step=global_step,
                max_steps=max_steps,
            )
            validation_metrics, _, validation_health = _run_epoch(
                model=model,
                loader=dataloaders["val"],
                teacher_adapter=teacher_adapter,
                criterion=criterion,
                device=device,
                optimizer=None,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip=0.0,
                global_step=global_step,
                max_steps=None,
            )
            last_health = validation_health
            learning_rate = optimizer.param_groups[0]["lr"]
            _append_jsonl(run_dir / "metrics" / "train.jsonl", {
                "epoch": epoch, "global_step": global_step, "learning_rate": learning_rate, **train_metrics,
            })
            _append_jsonl(run_dir / "metrics" / "validation.jsonl", {
                "epoch": epoch, "global_step": global_step, **validation_metrics,
            })

            improved = validation_metrics["total"] < best_validation - min_delta
            if improved:
                best_validation = validation_metrics["total"]
                epochs_without_improvement = 0
                _save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    config=config, epoch=epoch, global_step=global_step,
                    best_validation=best_validation,
                    epochs_without_improvement=epochs_without_improvement,
                    e0_gate_hash=gate_hash or "",
                )
            else:
                epochs_without_improvement += 1
            _save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                config=config, epoch=epoch, global_step=global_step,
                best_validation=best_validation,
                epochs_without_improvement=epochs_without_improvement,
                e0_gate_hash=gate_hash or "",
            )
            if max_steps is not None and global_step >= max_steps:
                break
            if epochs_without_improvement >= patience:
                status = "early_stopped"
                break
        if status != "early_stopped":
            status = "smoke_passed" if args.smoke else "training_complete"

    _write_json(run_dir / "diagnostics" / "quantizer_health.json", last_health)
    checkpoint_hashes = {
        path.name: _sha256(path) for path in (run_dir / "checkpoints").glob("*.pt")
    }
    split_hash = hashlib.sha256(
        json.dumps(config.get("data", {}).get("split", {}), sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": RUN_SCHEMA,
        "status": status,
        "mode": "train" if args.train else ("smoke" if args.smoke else "dry_run"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(_git_value("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "seed": seed,
        "device": str(device),
        "e0_gate_sha256": gate_hash,
        "objective": "teacher_supervised" if teacher_supervision else "teacher_free",
        "global_step": global_step,
        "best_validation": None if math.isinf(best_validation) else best_validation,
        "split_sha256": split_hash,
        "checkpoint_sha256": checkpoint_hashes,
        "protected_test_opened": False,
        "start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "manifest.json", manifest)
    summary_lines = [
        "# Physiology-semantic tokenizer run summary",
        "",
        f"- Status: `{status}`",
        f"- Objective: `{'teacher_supervised' if teacher_supervision else 'teacher_free'}`",
        f"- Device: `{device}`",
        f"- Global optimizer steps: `{global_step}`",
        f"- Best validation total loss: `{manifest['best_validation']}`",
        f"- Protected test opened: `False`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "status": status, "global_step": global_step}, sort_keys=True))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--e0-gate", help="Override validation.e0_gate_path with a concrete gate_decision.json")
    parser.add_argument("--smoke-optimizer-steps", type=int, help="Override smoke step budget")
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
