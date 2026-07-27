#!/usr/bin/env python3
"""Open and evaluate one frozen protected EFRM target fold exactly once."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efrm_pytorch.model import EFRMDownstreamModel
from efrm_pytorch.tasks import EFRMUnifiedTaskDataset, get_task_spec
from train_downstream import (
    _build_backbone,
    _classification_weights,
    _evaluate,
    make_loader,
    sha256_file,
    write_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(args: argparse.Namespace) -> dict[str, Any]:
    job_dir = Path(args.job_dir).resolve()
    manifest_path = job_dir / "manifest.json"
    metrics_path = job_dir / "test_metrics.json"
    if metrics_path.exists() or (job_dir / "protected_access_manifest.json").exists():
        raise FileExistsError(f"protected fold was already opened: {job_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError("public train/validation job must complete before protected access")
    if manifest.get("protected_test_opened") is not False:
        raise PermissionError("job manifest already reports protected test access")
    split = json.loads((job_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if split.get("schema") != "efrm_target_public_fold_v1":
        raise ValueError("formal evaluation requires an EFRM target public fold")

    protected_path = Path(args.protected_manifest).resolve()
    protected = json.loads(protected_path.read_text(encoding="utf-8"))
    if protected.get("schema") != "efrm_target_protected_fold_v1":
        raise ValueError("unsupported protected fold schema")
    for field in ("protocol_id", "protocol", "task", "outer_fold"):
        public_value = (
            split.get(field)
            if field != "outer_fold"
            else int(split.get(field))
        )
        protected_value = (
            protected.get(field)
            if field != "outer_fold"
            else int(protected.get(field))
        )
        if public_value != protected_value:
            raise RuntimeError(f"public/protected fold {field} mismatch")
    if protected["public_split_sha256"] != split["split_sha256"]:
        raise RuntimeError("protected fold does not bind to the public split")
    if sha256_file(job_dir / "split_manifest.json") != manifest["split_sha256"]:
        raise RuntimeError("job's copied public manifest drifted after training")

    config = yaml.safe_load(
        (job_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    task = str(protected["task"])
    spec = get_task_spec(task)
    dataset = EFRMUnifiedTaskDataset(
        spec, cache_root=str(config["data"]["cache_root"])
    )
    if dataset.metadata_fingerprint() != protected["metadata_sha256"]:
        raise RuntimeError("protected fold metadata fingerprint drifted")
    test_indices = [int(value) for value in protected["test_indices"]]
    public_indices = {
        int(value) for value in split["train_indices"] + split["validation_indices"]
    }
    if not test_indices or public_indices & set(test_indices):
        raise RuntimeError("protected test is empty or overlaps public indices")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal protected evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)

    best_path = job_dir / "checkpoint_best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best.get("task") != task:
        raise RuntimeError("best downstream checkpoint task mismatch")
    if spec.task_type == "regression":
        scaler = best.get("target_scaler")
        if not scaler:
            raise RuntimeError("REFED best checkpoint omits train-only target scaler")
        dataset.set_target_scaler(scaler["center"], scaler["scale"])

    pretrain_path = Path(str(manifest["pretrained_checkpoint"])).resolve()
    if sha256_file(pretrain_path) != manifest["pretrained_checkpoint_sha256"]:
        raise RuntimeError("frozen source checkpoint hash drifted")
    pretrained = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    backbone = _build_backbone(config)
    backbone.load_state_dict(pretrained["model"], strict=True)
    model = EFRMDownstreamModel(
        backbone,
        output_dim=spec.output_dim,
        modality=str(manifest["modality"]),
        target_length=spec.target_length,
        dropout=float(
            {
                **config["training"],
                **config.get("task_overrides", {}).get(task, {}),
            }.get("dropout", 0.5)
        ),
    )
    model.configure_transfer(str(manifest["transfer_mode"]))
    model.load_state_dict(best["transfer_state"], strict=False)
    model.to(device)

    train_cfg = {
        **config["training"],
        **config.get("task_overrides", {}).get(task, {}),
    }
    batch_size = int(
        train_cfg.get(
            "linear_probe_batch_size",
            train_cfg.get("batch_size", 16),
        )
    )
    loader, _ = make_loader(
        dataset,
        test_indices,
        batch_size=batch_size,
        workers=int(train_cfg.get("num_workers", 0)),
        shuffle=False,
        seed=int(train_cfg.get("seed", 42)),
    )
    class_weights = None
    policy = str(train_cfg.get("class_weighting", "none"))
    if spec.task_type == "classification" and policy != "none":
        class_weights = _classification_weights(
            dataset, [int(value) for value in split["train_indices"]], policy
        ).to(device)
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

    opened_at = utc_now()
    access = {
        "schema": "efrm_protected_access_v1",
        "protocol_id": protected["protocol_id"],
        "task": task,
        "protocol": protected["protocol"],
        "outer_fold": int(protected["outer_fold"]),
        "opened_at": opened_at,
        "protected_test_opened": True,
        "protected_manifest": str(protected_path),
        "protected_manifest_sha256": sha256_file(protected_path),
        "public_split_sha256": split["split_sha256"],
        "best_checkpoint_sha256": sha256_file(best_path),
        "source_checkpoint_sha256": manifest["pretrained_checkpoint_sha256"],
    }
    write_json(job_dir / "protected_access_manifest.json", access)
    started = time.perf_counter()
    metrics, evidence = _evaluate(
        model=model,
        loader=loader,
        dataset=dataset,
        device=device,
        autocast_context=autocast_context,
        class_weights=class_weights,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        regression_loss=str(train_cfg.get("regression_loss", "smooth_l1")),
    )
    metrics.update(
        {
            "schema": "efrm_formal_fold_metrics_v1",
            "protocol_id": protected["protocol_id"],
            "reporting_name": protected["reporting_name"],
            "protocol": protected["protocol"],
            "task": task,
            "outer_fold": int(protected["outer_fold"]),
            "protected_test_opened": True,
            "protected_opened_at": opened_at,
            "evaluation_seconds": time.perf_counter() - started,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "test_indices_sha256": protected["test_indices_sha256"],
            "best_checkpoint_sha256": access["best_checkpoint_sha256"],
            "source_checkpoint_sha256": access["source_checkpoint_sha256"],
        }
    )
    np.savez_compressed(job_dir / "test_predictions.npz", **evidence)
    write_json(metrics_path, metrics)
    manifest.update(
        {
            "status": "protected_evaluation_completed",
            "protected_test_opened": True,
            "protected_opened_at": opened_at,
            "test_metrics": metrics,
            "test_predictions": str((job_dir / "test_predictions.npz").resolve()),
        }
    )
    write_json(manifest_path, manifest)
    write_json(job_dir / "status.json", manifest)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--protected-manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
