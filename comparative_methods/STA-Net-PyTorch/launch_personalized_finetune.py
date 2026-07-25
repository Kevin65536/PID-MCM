#!/usr/bin/env python3
"""Launch leakage-safe target-subject fine-tuning from selected cross-subject checkpoints."""

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

import yaml

METHOD_ROOT = Path(__file__).resolve().parent
TASKS = (
    "motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual",
    "refed_regression",
)
OLD_SELECTION = (
    METHOD_ROOT / "runs" / "tuning" / "20260722_sta_net_hpo_v2_checkpoint_objective_100ep"
    / "final_validation_selection_v2" / "selection_manifest.json"
)
FINAL_SELECTION = (
    METHOD_ROOT / "runs" / "tuning" / "20260724_sta_net_mi_wg_final_targeted_v1_100ep"
    / "final_validation_selection_v2" / "selection_manifest.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_assets(tasks: list[str]) -> dict[str, dict[str, Any]]:
    old = json.loads(OLD_SELECTION.read_text(encoding="utf-8"))["tasks"]
    final = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))["tasks"]
    merged = {**old, **final}
    result: dict[str, dict[str, Any]] = {}
    for task in tasks:
        selected = merged[task]["selected"]
        config = Path(selected["config"]).resolve()
        checkpoint = Path(selected["checkpoint"]).resolve()
        run_dir = Path(selected["run_dir"]).resolve()
        if sha256(config) != selected["config_sha256"]:
            raise RuntimeError(f"selected config hash drift for {task}")
        if sha256(checkpoint) != selected["checkpoint_sha256"]:
            raise RuntimeError(f"selected checkpoint hash drift for {task}")
        pretrain_split_path = run_dir / "split_manifest.json"
        pretrain_split = json.loads(pretrain_split_path.read_text(encoding="utf-8"))
        targets = [str(value) for value in pretrain_split["reserved_test_subjects"]]
        exposed = set(map(str, pretrain_split["train_subjects"])) | set(
            map(str, pretrain_split["validation_subjects"])
        )
        if set(targets) & exposed:
            raise RuntimeError(f"reserved target subject leaked into pretraining for {task}")
        result[task] = {
            "source_trial": int(selected["trial_number"]),
            "source_config": str(config),
            "source_config_sha256": sha256(config),
            "pretrained_checkpoint": str(checkpoint),
            "pretrained_checkpoint_sha256": sha256(checkpoint),
            "pretraining_split": str(pretrain_split_path),
            "pretraining_split_sha256": sha256(pretrain_split_path),
            "pretraining_train_subjects": list(map(str, pretrain_split["train_subjects"])),
            "pretraining_validation_subjects": list(map(str, pretrain_split["validation_subjects"])),
            "target_subjects": targets,
            "selection_manifest": str(FINAL_SELECTION if task in final else OLD_SELECTION),
        }
    return result


def fine_tune_config(
    source_path: Path,
    *,
    task: str,
    epochs: int,
    learning_rate_scale: float,
) -> dict[str, Any]:
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    training = config.setdefault("training", {})
    task_override = config.setdefault("task_overrides", {}).setdefault(task, {})
    base_lr = float(task_override.get("lr", training["lr"]))
    fine_lr = base_lr * learning_rate_scale
    training.update({
        "epochs": epochs,
        "lr": fine_lr,
        "scheduler": "cosine",
        "scheduler_total_epochs": epochs,
        "warmup_ratio": 0.0,
        "selection_metric": "masked_rmse_scaled" if task == "refed_regression" else "loss",
        "selection_mode": "min",
        "selection_min_delta": 0.0,
    })
    for key in (
        "epochs", "lr", "scheduler", "scheduler_total_epochs", "warmup_ratio",
        "selection_metric", "selection_mode", "selection_min_delta",
    ):
        if key in task_override:
            task_override[key] = training[key]
    config["personalization"] = {
        "schema": "sta_net_personalized_finetune_v1",
        "initialization": "selected_cross_subject_checkpoint_model_weights_only",
        "optimizer": "fresh",
        "scheduler": "fresh",
        "full_model_finetune": True,
        "epochs": epochs,
        "base_learning_rate": base_lr,
        "learning_rate_scale": learning_rate_scale,
        "effective_learning_rate": fine_lr,
        "initial_epoch_zero_validation_candidate": True,
        "classification_selection_metric": "validation_total_loss",
    }
    return config


def build_jobs(
    tasks: list[str],
    assets: Mapping[str, Mapping[str, Any]],
    config_paths: Mapping[str, Path],
    *,
    epochs: int,
    learning_rate_scale: float,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    registry = METHOD_ROOT / "split_registry"
    for task in tasks:
        targets = set(map(str, assets[task]["target_subjects"]))
        found_targets: set[str] = set()
        public_dir = registry / task / "single_subject" / "public"
        protected_dir = registry / task / "single_subject" / "protected"
        for public_path in sorted(public_dir.glob("*.json")):
            protected_path = protected_dir / public_path.name
            public = json.loads(public_path.read_text(encoding="utf-8"))
            protected = json.loads(protected_path.read_text(encoding="utf-8"))
            subject = str(protected["subject"])
            if subject not in targets:
                continue
            found_targets.add(subject)
            if public["fold_id"] != protected["fold_id"] or public["task"] != protected["task"]:
                raise RuntimeError(f"split identity mismatch for {public_path}")
            if set(public["train_indices"]) & set(protected["test_indices"]):
                raise RuntimeError(f"calibration train/test overlap for {public_path}")
            if set(public["validation_indices"]) & set(protected["test_indices"]):
                raise RuntimeError(f"calibration validation/test overlap for {public_path}")
            if set(map(str, public["train_subjects"])) != {subject}:
                raise RuntimeError(f"calibration train subject mismatch for {public_path}")
            if set(map(str, public["validation_subjects"])) != {subject}:
                raise RuntimeError(f"calibration validation subject mismatch for {public_path}")
            jobs.append({
                "task": task,
                "fold_id": public["fold_id"],
                "subject": subject,
                "config": str(config_paths[task].resolve()),
                "config_sha256": sha256(config_paths[task]),
                "source_trial": assets[task]["source_trial"],
                "pretrained_checkpoint": assets[task]["pretrained_checkpoint"],
                "pretrained_checkpoint_sha256": assets[task]["pretrained_checkpoint_sha256"],
                "pretraining_split": assets[task]["pretraining_split"],
                "pretraining_split_sha256": assets[task]["pretraining_split_sha256"],
                "public_manifest": str(public_path.resolve()),
                "public_manifest_sha256": sha256(public_path),
                "protected_manifest": str(protected_path.resolve()),
                "protected_manifest_sha256": sha256(protected_path),
                "train_sample_count": int(public["train_sample_count"]),
                "validation_sample_count": int(public["validation_sample_count"]),
                "test_sample_count": int(public["protected_test"]["sample_count"]),
                "epochs": epochs,
                "learning_rate_scale": learning_rate_scale,
                "selection_metric": (
                    "masked_rmse_scaled" if task == "refed_regression" else "loss"
                ),
                "selection_mode": "min",
            })
        missing = targets - found_targets
        if missing:
            raise RuntimeError(f"target subjects lack eligible calibration folds for {task}: {sorted(missing)}")
    return jobs


def distribute(jobs: list[dict[str, Any]], lane_count: int) -> list[list[dict[str, Any]]]:
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(lane_count)]
    loads = [0.0] * lane_count
    for job in sorted(
        jobs,
        key=lambda row: (
            float(row["train_sample_count"]) + float(row["validation_sample_count"])
        ) * float(row["epochs"]),
        reverse=True,
    ):
        lane = min(range(lane_count), key=loads.__getitem__)
        lanes[lane].append(job)
        loads[lane] += (
            float(job["train_sample_count"]) + float(job["validation_sample_count"]) + 32.0
        ) * float(job["epochs"])
    return lanes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d_sta_net_personalized_finetune_v1"),
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--lanes-per-gpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unlock-protected-test", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.unlock_protected_test:
        raise RuntimeError("launch requires explicit --unlock-protected-test")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0.0 < args.learning_rate_scale <= 1.0:
        raise ValueError("learning-rate-scale must be in (0, 1]")

    tasks = list(args.tasks)
    assets = selected_assets(tasks)
    if args.dry_run:
        placeholder_configs = {
            task: Path(assets[task]["source_config"]) for task in tasks
        }
        jobs = build_jobs(
            tasks, assets, placeholder_configs,
            epochs=args.epochs, learning_rate_scale=args.learning_rate_scale,
        )
        lanes = distribute(jobs, len(args.gpus) * args.lanes_per_gpu)
        print(json.dumps({
            "status": "dry_run",
            "tasks": tasks,
            "target_subjects": {
                task: assets[task]["target_subjects"] for task in tasks
            },
            "job_count": len(jobs),
            "lane_count": len(lanes),
            "jobs_per_lane": [len(lane) for lane in lanes],
            "epochs": args.epochs,
            "learning_rate_scale": args.learning_rate_scale,
            "protected_test_opened": False,
        }, indent=2))
        return

    root = METHOD_ROOT / "runs" / "personalized_finetune" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    config_paths: dict[str, Path] = {}
    for task in tasks:
        config_path = root / "configs" / f"{task}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = fine_tune_config(
            Path(assets[task]["source_config"]),
            task=task,
            epochs=args.epochs,
            learning_rate_scale=args.learning_rate_scale,
        )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        config_paths[task] = config_path

    jobs = build_jobs(
        tasks, assets, config_paths,
        epochs=args.epochs, learning_rate_scale=args.learning_rate_scale,
    )
    lane_count = len(args.gpus) * args.lanes_per_gpu
    lanes = distribute(jobs, lane_count)
    protocol = {
        "schema": "sta_net_personalized_finetune_protocol_freeze_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "protocol": "cross_subject_pretrained_target_subject_finetune",
        "report_title": "STA-Net target-subject calibration fine-tuning evaluation",
        "report_description": (
            "Selected cross-subject checkpoints are fine-tuned on public calibration trials "
            "from pretraining-reserved target subjects and evaluated on disjoint protected groups."
        ),
        "claim_boundary": (
            "Post-hoc personalized evaluation: these protected groups were previously opened by "
            "the subject-only analysis; results are not a newly untouched confirmatory test."
        ),
        "tasks": tasks,
        "epochs": args.epochs,
        "learning_rate_scale": args.learning_rate_scale,
        "optimizer_policy": "fresh AdamW; pretrained optimizer and scheduler states are not loaded",
        "model_policy": "load all pretrained model weights and fine-tune all parameters",
        "checkpoint_selection": (
            "minimum public target-subject validation loss across initialization epoch 0 and "
            "fine-tuning epochs 1..N; regression uses masked scaled RMSE"
        ),
        "seed": 42,
        "classification_primary_endpoint": "mean per-target-subject macro F1",
        "regression_primary_endpoint": "mean per-target-subject concordance correlation",
        "bootstrap_draws": 10_000,
        "assets": assets,
        "configs": {
            task: {
                "path": str(config_paths[task]),
                "sha256": sha256(config_paths[task]),
            }
            for task in tasks
        },
        "jobs": jobs,
        "jobs_sha256": sha256_json(jobs),
        "job_count": len(jobs),
        "trainer_sha256": sha256(METHOD_ROOT / "train.py"),
        "worker_sha256": sha256(METHOD_ROOT / "personalized_finetune_worker.py"),
        "evaluator_sha256": sha256(METHOD_ROOT / "evaluate_protocol.py"),
        "aggregator_sha256": sha256(METHOD_ROOT / "aggregate_within_subject.py"),
        "protected_test_opened": True,
        "protected_open_authorization": (
            "explicit user request for performance with partial target-subject trial visibility"
        ),
    }
    write_json(root / "protocol_freeze_manifest.json", protocol)

    launches = []
    for lane_index, jobs_for_lane in enumerate(lanes):
        lane_id = f"lane_{lane_index:02d}"
        jobs_path = root / "jobs" / f"{lane_id}.json"
        write_json(jobs_path, jobs_for_lane)
        gpu = args.gpus[lane_index % len(args.gpus)]
        log_path = root / "logs" / f"{lane_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            str(METHOD_ROOT / "personalized_finetune_worker.py"),
            "--run-root",
            str(root),
            "--jobs",
            str(jobs_path),
            "--lane-id",
            lane_id,
            "--workers",
            str(args.workers),
            "--unlock-protected-test",
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=METHOD_ROOT.parents[1],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launches.append({
            "lane_id": lane_id,
            "physical_gpu": gpu,
            "pid": process.pid,
            "job_count": len(jobs_for_lane),
            "jobs": str(jobs_path),
            "log": str(log_path),
        })

    supervisor_log = root / "logs" / "supervisor.log"
    supervisor_command = [
        sys.executable,
        "-u",
        str(METHOD_ROOT / "within_subject_supervisor.py"),
        "--run-root",
        str(root),
        "--lane-count",
        str(lane_count),
    ]
    with supervisor_log.open("w", encoding="utf-8") as log:
        supervisor = subprocess.Popen(
            supervisor_command,
            cwd=METHOD_ROOT.parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    manifest = {
        "schema": "sta_net_personalized_finetune_launch_v1",
        "status": "launched",
        "created_at": utc_now(),
        "run_root": str(root),
        "jobs": len(jobs),
        "lanes": launches,
        "supervisor_pid": supervisor.pid,
        "supervisor_log": str(supervisor_log),
    }
    write_json(root / "launch_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
