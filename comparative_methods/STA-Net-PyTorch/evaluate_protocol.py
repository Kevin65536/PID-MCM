#!/usr/bin/env python3
"""Evaluate a frozen STA-Net checkpoint on an explicitly unlocked protected fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch import STANet, STANetConfig, STANetUnifiedTaskDataset, get_sta_net_task_spec
from sta_net_pytorch.splits import PROTECTED_SCHEMA
from train import make_loader, move_batch
from visualize_results import classification_metrics, regression_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--protected-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--unlock-protected-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_protected_test:
        raise RuntimeError("protected evaluation requires explicit --unlock-protected-test")
    frozen_path = Path(args.freeze_manifest).resolve()
    protected_path = Path(args.protected_manifest).resolve()
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    protected = json.loads(protected_path.read_text(encoding="utf-8"))
    if frozen.get("schema") != "sta_net_frozen_tuning_winner_v1":
        raise ValueError("invalid freeze manifest")
    if protected.get("schema") != PROTECTED_SCHEMA:
        raise ValueError("invalid protected split manifest")
    task = str(frozen["task"])
    if protected.get("task") != task:
        raise ValueError("frozen task and protected split task differ")
    if frozen.get("protocol") is not None and frozen["protocol"] != protected.get("protocol"):
        raise ValueError("frozen protocol and protected split protocol differ")
    if frozen.get("fold_id") is not None and frozen["fold_id"] != protected.get("fold_id"):
        raise ValueError("frozen fold and protected split fold differ")
    checkpoint_path = Path(frozen["checkpoint"])
    config_path = Path(frozen["config"])
    if sha256(checkpoint_path) != frozen["checkpoint_sha256"] or sha256(config_path) != frozen["config_sha256"]:
        raise RuntimeError("frozen config/checkpoint hash drift")
    split_path_raw = frozen.get("split_manifest")
    split_hash = frozen.get("split_manifest_sha256")
    if split_path_raw is not None and split_hash is not None:
        split_path = Path(split_path_raw)
        if sha256(split_path) != split_hash:
            raise RuntimeError("frozen public split manifest hash drift")
    test_indices = sorted(int(value) for value in protected["test_indices"])
    if sha256_json(test_indices) != protected["indices_sha256"]:
        raise RuntimeError("protected test index hash drift")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    spec = get_sta_net_task_spec(task)
    dataset = STANetUnifiedTaskDataset(spec, cache_root=str(config["data"]["cache_root"]))
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler = checkpoint.get("target_scaler")
    if scaler is not None:
        dataset.adapter.set_target_scaler(scaler["center"], scaler["scale"])
    batch_size = int(config.get("task_overrides", {}).get(task, {}).get(
        "batch_size", config["training"].get("batch_size", 32)
    ))
    loader, _ = make_loader(
        dataset, test_indices, batch_size=batch_size, workers=args.workers,
        shuffle=False, seed=int(config["training"].get("seed", 42)),
        prefetch_factor=int(config["training"].get("prefetch_factor", 2)),
    )
    model = STANet(STANetConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    dtype = torch.bfloat16 if config["training"].get("amp_dtype") == "bfloat16" else torch.float16
    predictions, targets, masks, subjects, sample_ids = [], [], [], [], []
    with torch.inference_mode():
        for raw in loader:
            batch = move_batch(raw, device)
            context = torch.autocast(device_type="cuda", dtype=dtype) if amp else nullcontext()
            with context:
                output = model(batch["eeg"], batch["fnirs"])["prediction"]
            predictions.append(output.detach().float().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
            masks.append(batch["target_valid_mask"].detach().cpu().numpy())
            subjects.extend(batch["subject"])
            sample_ids.extend(batch["sample_id"])
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks)
    prediction_to_save = prediction
    target_to_save = target
    coordinate_space = "class_probability"
    if spec.task_type == "classification":
        metrics = classification_metrics(target, prediction, spec.class_names)
    else:
        if scaler is None:
            raise RuntimeError("regression protected evaluation requires the train-only scaler")
        center = np.asarray(scaler["center"], dtype=np.float32)[None, :, None]
        scale = np.asarray(scaler["scale"], dtype=np.float32)[None, :, None]
        target_to_save = target * scale + center
        prediction_to_save = prediction * scale + center
        coordinate_space = "native_target_units"
        metrics = regression_metrics(
            target_to_save, prediction_to_save, mask, spec.target_names
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "protected_predictions.npz",
        prediction=prediction_to_save,
        target=target_to_save,
        target_valid_mask=mask, subject=np.asarray(subjects), sample_id=np.asarray(sample_ids),
    )
    summary = {
        "schema": "sta_net_protocol_evaluation_v1", "task": task,
        "protocol": protected["protocol"], "fold_id": protected.get("fold_id"),
        "outer_fold": protected.get("outer_fold"), "metrics": metrics,
        "sample_count": len(test_indices), "subject_count": len(set(subjects)),
        "prediction_coordinate_space": coordinate_space,
        "freeze_manifest": str(frozen_path), "protected_manifest": str(protected_path),
        "protected_test_opened": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
