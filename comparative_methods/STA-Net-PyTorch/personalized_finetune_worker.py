#!/usr/bin/env python3
"""Fine-tune pretrained STA-Net checkpoints and evaluate protected target-subject folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

METHOD_ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] command: {json.dumps(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=METHOD_ROOT.parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {command}")


def completed_training(run_dir: Path) -> bool:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("status") == "completed" and (run_dir / "checkpoint_best.pt").is_file()


def freeze_fold(job: Mapping[str, Any], training_dir: Path, output: Path) -> None:
    public_split = Path(str(job["public_manifest"])).resolve()
    protected_split = Path(str(job["protected_manifest"])).resolve()
    config = Path(str(job["config"])).resolve()
    pretrained = Path(str(job["pretrained_checkpoint"])).resolve()
    checkpoint = training_dir / "checkpoint_best.pt"
    train_manifest = json.loads((training_dir / "manifest.json").read_text(encoding="utf-8"))
    public = json.loads(public_split.read_text(encoding="utf-8"))
    protected = json.loads(protected_split.read_text(encoding="utf-8"))
    if train_manifest.get("status") != "completed":
        raise RuntimeError("cannot freeze an incomplete fine-tuning run")
    if train_manifest.get("split_sha256") != public.get("split_sha256"):
        raise RuntimeError("fine-tuning run does not match the public calibration split")
    if public.get("fold_id") != protected.get("fold_id") or public.get("task") != protected.get("task"):
        raise RuntimeError("public/protected fold pairing mismatch")
    if sha256(pretrained) != job["pretrained_checkpoint_sha256"]:
        raise RuntimeError("pretrained checkpoint hash drift")
    write_json(output, {
        "schema": "sta_net_frozen_tuning_winner_v1",
        "study_id": "sta_net_personalized_finetune_v1",
        "study_name": "cross_subject_pretrained__target_subject_calibration_finetune",
        "task": str(job["task"]),
        # evaluate_protocol requires the split protocol to match exactly.
        "protocol": str(protected["protocol"]),
        "fold_id": str(job["fold_id"]),
        "trial_number": job["source_trial"],
        "objective_value": None,
        "parameters": {
            "epochs": int(job["epochs"]),
            "learning_rate_scale": float(job["learning_rate_scale"]),
            "selection_metric": str(job["selection_metric"]),
            "selection_mode": str(job["selection_mode"]),
        },
        "config": str(config),
        "config_sha256": sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "pretrained_checkpoint": str(pretrained),
        "pretrained_checkpoint_sha256": sha256(pretrained),
        "split_manifest": str(public_split),
        "split_manifest_sha256": sha256(public_split),
        "protected_manifest_sha256": sha256(protected_split),
        "trainer_sha256": sha256(METHOD_ROOT / "train.py"),
        "model_sha256": sha256(METHOD_ROOT / "sta_net_pytorch" / "model.py"),
        "adapter_sha256": sha256(METHOD_ROOT / "sta_net_pytorch" / "data.py"),
        "training_manifest_sha256": sha256(training_dir / "manifest.json"),
        "frozen_at": utc_now(),
        "protected_test_opened": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--unlock-protected-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_protected_test:
        raise RuntimeError("personalized protected evaluation requires explicit unlock")

    root = Path(args.run_root).resolve()
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    status_path = root / "status" / f"{args.lane_id}.json"
    completed_jobs: list[str] = []
    write_json(status_path, {
        "schema": "sta_net_personalized_finetune_lane_v1",
        "status": "running",
        "lane_id": args.lane_id,
        "pid": os.getpid(),
        "job_count": len(jobs),
        "completed_count": 0,
        "started_at": utc_now(),
    })
    try:
        for position, job in enumerate(jobs):
            key = f"{job['task']}/{job['fold_id']}"
            fold_dir = root / "folds" / str(job["task"]) / str(job["fold_id"])
            training_dir = fold_dir / "training"
            evaluation_dir = fold_dir / "evaluation"
            freeze_path = fold_dir / "freeze_manifest.json"
            if (evaluation_dir / "summary.json").exists():
                completed_jobs.append(key)
                continue
            write_json(status_path, {
                "schema": "sta_net_personalized_finetune_lane_v1",
                "status": "running",
                "lane_id": args.lane_id,
                "pid": os.getpid(),
                "job_count": len(jobs),
                "completed_count": len(completed_jobs),
                "position": position,
                "active_job": key,
                "updated_at": utc_now(),
            })
            if not completed_training(training_dir):
                latest = training_dir / "checkpoint_latest.pt"
                command = [
                    sys.executable,
                    "-u",
                    str(METHOD_ROOT / "train.py"),
                    "--config",
                    str(job["config"]),
                    "--task",
                    str(job["task"]),
                    "--device",
                    "cuda:0",
                    "--output-dir",
                    str(training_dir),
                    "--epochs",
                    str(job["epochs"]),
                    "--split-manifest",
                    str(job["public_manifest"]),
                ]
                if latest.exists():
                    command.extend(["--resume", str(latest)])
                else:
                    command.extend(["--init-checkpoint", str(job["pretrained_checkpoint"])])
                run_command(command, fold_dir / "train.log")
            freeze_fold(job, training_dir, freeze_path)
            run_command([
                sys.executable,
                "-u",
                str(METHOD_ROOT / "evaluate_protocol.py"),
                "--freeze-manifest",
                str(freeze_path),
                "--protected-manifest",
                str(job["protected_manifest"]),
                "--output-dir",
                str(evaluation_dir),
                "--device",
                "cuda:0",
                "--workers",
                str(args.workers),
                "--unlock-protected-test",
            ], fold_dir / "evaluation.log")
            completed_jobs.append(key)
        write_json(status_path, {
            "schema": "sta_net_personalized_finetune_lane_v1",
            "status": "completed",
            "lane_id": args.lane_id,
            "pid": os.getpid(),
            "job_count": len(jobs),
            "completed_count": len(completed_jobs),
            "completed_at": utc_now(),
        })
    except Exception as error:
        write_json(status_path, {
            "schema": "sta_net_personalized_finetune_lane_v1",
            "status": "failed",
            "lane_id": args.lane_id,
            "pid": os.getpid(),
            "job_count": len(jobs),
            "completed_count": len(completed_jobs),
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at": utc_now(),
        })
        raise


if __name__ == "__main__":
    main()
