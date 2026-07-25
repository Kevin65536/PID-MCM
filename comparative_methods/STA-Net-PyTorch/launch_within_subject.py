#!/usr/bin/env python3
"""Freeze and launch the complete STA-Net non-cross-subject protocol."""

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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_configs() -> dict[str, dict[str, Any]]:
    old = json.loads(OLD_SELECTION.read_text(encoding="utf-8"))["tasks"]
    final = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))["tasks"]
    merged = {**old, **final}
    result = {}
    for task in TASKS:
        selected = merged[task]["selected"]
        config = Path(selected["config"]).resolve()
        if sha256(config) != selected["config_sha256"]:
            raise RuntimeError(f"selected config hash drift for {task}")
        result[task] = {
            "config": str(config),
            "config_sha256": selected["config_sha256"],
            "source_trial": int(selected["trial_number"]),
            "selection_manifest": str(FINAL_SELECTION if task in final else OLD_SELECTION),
        }
    return result


def build_jobs(tasks: list[str], configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    registry = METHOD_ROOT / "split_registry"
    for task in tasks:
        public_dir = registry / task / "single_subject" / "public"
        protected_dir = registry / task / "single_subject" / "protected"
        for public_path in sorted(public_dir.glob("*.json")):
            protected_path = protected_dir / public_path.name
            if not protected_path.exists():
                raise RuntimeError(f"missing protected pair for {public_path}")
            public = json.loads(public_path.read_text(encoding="utf-8"))
            protected = json.loads(protected_path.read_text(encoding="utf-8"))
            if public["fold_id"] != protected["fold_id"] or public["task"] != protected["task"]:
                raise RuntimeError(f"split identity mismatch for {public_path}")
            if set(public["train_indices"]) & set(protected["test_indices"]):
                raise RuntimeError(f"train/test overlap for {public_path}")
            if set(public["validation_indices"]) & set(protected["test_indices"]):
                raise RuntimeError(f"validation/test overlap for {public_path}")
            jobs.append({
                "task": task,
                "fold_id": public["fold_id"],
                "subject": protected["subject"],
                "config": configs[task]["config"],
                "config_sha256": configs[task]["config_sha256"],
                "source_trial": configs[task]["source_trial"],
                "public_manifest": str(public_path.resolve()),
                "public_manifest_sha256": sha256(public_path),
                "protected_manifest": str(protected_path.resolve()),
                "protected_manifest_sha256": sha256(protected_path),
                "train_sample_count": public["train_sample_count"],
                "validation_sample_count": public["validation_sample_count"],
                "test_sample_count": public["protected_test"]["sample_count"],
            })
    return jobs


def distribute(jobs: list[dict[str, Any]], lane_count: int) -> list[list[dict[str, Any]]]:
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(lane_count)]
    loads = [0.0] * lane_count
    for job in sorted(
        jobs,
        key=lambda row: (float(row["train_sample_count"]) + float(row["validation_sample_count"])),
        reverse=True,
    ):
        lane = min(range(lane_count), key=loads.__getitem__)
        lanes[lane].append(job)
        loads[lane] += float(job["train_sample_count"]) + float(job["validation_sample_count"]) + 32.0
    return lanes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_sta_net_within_subject_v1"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--lanes-per-gpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unlock-protected-test", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.unlock_protected_test:
        raise RuntimeError("launch requires explicit --unlock-protected-test")
    if args.epochs != 100:
        raise ValueError("formal within-subject protocol is frozen at 100 epochs")
    configs = selected_configs()
    jobs = build_jobs(list(args.tasks), configs)
    lane_count = len(args.gpus) * args.lanes_per_gpu
    lanes = distribute(jobs, lane_count)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "tasks": args.tasks,
            "job_count": len(jobs),
            "lane_count": lane_count,
            "jobs_per_lane": [len(lane) for lane in lanes],
            "protected_test_opened": False,
        }, indent=2))
        return

    root = METHOD_ROOT / "runs" / "within_subject" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "sta_net_within_subject_protocol_freeze_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "protocol": "single_subject_nested_cv",
        "description": "same-subject training/validation/test with dependency-group isolation",
        "tasks": list(args.tasks),
        "epochs": args.epochs,
        "seed": 42,
        "selection_rule": "fixed final cross-subject-selected hyperparameters; per-fold best public validation checkpoint",
        "classification_primary_endpoint": "mean per-subject macro F1 after concatenating protected OOF groups",
        "regression_primary_endpoint": "mean per-subject concordance correlation after concatenating protected OOF groups",
        "source_aligned_primary_endpoint": "mean per-subject Accuracy for MI/MA/WG",
        "bootstrap_draws": 10_000,
        "configs": {task: configs[task] for task in args.tasks},
        "jobs": jobs,
        "jobs_sha256": sha256_json(jobs),
        "job_count": len(jobs),
        "trainer_sha256": sha256(METHOD_ROOT / "train.py"),
        "evaluator_sha256": sha256(METHOD_ROOT / "evaluate_protocol.py"),
        "aggregator_sha256": sha256(METHOD_ROOT / "aggregate_within_subject.py"),
        "protected_test_opened": True,
        "protected_open_authorization": "explicit user request to run non-cross-subject performance testing",
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
            str(METHOD_ROOT / "within_subject_worker.py"),
            "--run-root",
            str(root),
            "--jobs",
            str(jobs_path),
            "--lane-id",
            lane_id,
            "--epochs",
            str(args.epochs),
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
        "schema": "sta_net_within_subject_launch_v1",
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
