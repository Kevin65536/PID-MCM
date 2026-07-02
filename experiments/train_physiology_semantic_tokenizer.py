#!/usr/bin/env python3
"""Dry-run and software-smoke entrypoint for physiology-semantic tokenization."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

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


RUN_SCHEMA = "physiology_semantic_run_v1"


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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _default_run_dir(config: Dict[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    group = experiment.get("run_group", "physiology_semantic_tokenizer/software_smoke")
    name = experiment.get("name", "migration_smoke")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "experiments" / "runs" / str(group) / f"{stamp}_{name}"


def _loss_from_config(config: Dict[str, Any]) -> PhysiologySemanticLoss:
    loss = config.get("loss", {})
    return PhysiologySemanticLoss(
        state_weight=loss.get("state", {}).get("weight", 1.0),
        prototype_weight=loss.get("prototype", {}).get("weight", 1.0),
        masked_state_weight=loss.get("masked_state", {}).get("weight", 1.0),
        reconstruction_weight=loss.get("reconstruction", {}).get("weight", 1.0),
        vq_weight=loss.get("vq", {}).get("weight", 1.0),
        private_weight=loss.get("private", {}).get("weight", 0.0),
    )


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("training", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_dir = Path(args.output_dir).resolve() if args.output_dir else _default_run_dir(config)
    for relative in ("checkpoints", "metrics", "diagnostics"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    dataloaders = create_configured_multimodal_dataloaders(config)
    model = create_tokenizer(config)
    teacher_adapter = PhysicalStateTeacher()
    criterion = _loss_from_config(config)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    optimizer_steps_requested = int(config.get("training", {}).get("optimizer_steps", 0))
    e0_passed = bool(config.get("validation", {}).get("e0_passed", False))
    optimizer_steps = optimizer_steps_requested if e0_passed and args.smoke else 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("training", {}).get("lr", 1e-4)))

    batch = next(iter(dataloaders["train"]))
    teacher = teacher_adapter(batch["teacher"])
    if optimizer_steps > 0:
        model.train()
    else:
        model.eval()

    last_outputs = None
    last_losses = None
    steps = max(optimizer_steps, 1)
    for _ in range(steps):
        if optimizer_steps > 0:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["eeg"], batch["fnirs"])
            losses = criterion(outputs, teacher)
            losses["total"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(batch["eeg"], batch["fnirs"])
                losses = criterion(outputs, teacher)
        last_outputs, last_losses = outputs, losses

    checkpoint_path = run_dir / "checkpoints" / "software_smoke.pt"
    torch.save(
        {
            "schema": RUN_SCHEMA,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer_steps > 0 else None,
            "config": config,
            "e0_passed": e0_passed,
            "optimizer_steps": optimizer_steps,
        },
        checkpoint_path,
    )
    metrics = {key: float(value.detach().cpu()) for key, value in last_losses.items()}
    (run_dir / "metrics" / "train.jsonl").write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    _write_json(
        run_dir / "diagnostics" / "quantizer_health.json",
        {
            "eeg": last_outputs["eeg"].quantizer.health,
            "fnirs": last_outputs["fnirs"].quantizer.health,
        },
    )
    _write_json(
        run_dir / "diagnostics" / "teacher_diagnostics.json",
        {
            "valid_patches": teacher.valid_mask.sum(),
            "total_patches": teacher.valid_mask.numel(),
            "eeg_target_shape": list(teacher.eeg_target.shape),
            "fnirs_target_shape": list(teacher.fnirs_target.shape),
        },
    )
    _write_json(
        run_dir / "split_manifest.json",
        config.get("data", {}).get("split", {}),
    )
    _write_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    )
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    status = "dry_run_passed" if args.dry_run else "smoke_passed"
    if args.smoke and optimizer_steps_requested > 0 and not e0_passed:
        status = "smoke_passed_optimizer_blocked_by_e0"
    _write_json(
        run_dir / "manifest.json",
        {
            "schema": RUN_SCHEMA,
            "status": status,
            "git_commit": _git_value("rev-parse", "HEAD"),
            "dirty_worktree": bool(_git_value("status", "--porcelain")),
            "checkpoint": str(checkpoint_path.relative_to(run_dir)),
            "checkpoint_sha256": checkpoint_hash,
            "e0_passed": e0_passed,
            "optimizer_steps": optimizer_steps,
            "mode": "dry_run" if args.dry_run else "smoke",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps({"run_dir": str(run_dir), "status": status}, sort_keys=True))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
