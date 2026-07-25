#!/usr/bin/env python3
"""Launch concurrent detached STA-Net tuning workers on two GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import optuna
import yaml

from tune import DEFAULT_PRUNE_AFTER_EPOCH, OBJECTIVE_POLICY, RUNG_EPOCHS, SCHEMA

METHOD_ROOT = Path(__file__).resolve().parent
TASKS = (
    "motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual",
    "refed_regression",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lane_plan(n_trials: int) -> list[dict[str, object]]:
    """Shard the two longest tasks while preserving per-task trial totals."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    first = int(math.ceil(n_trials / 2))
    second = n_trials - first
    plan: list[dict[str, object]] = [
        {"gpu": 0, "lane": 0, "tasks": ("refed_regression",), "quota": first},
        {"gpu": 0, "lane": 2, "tasks": ("motor_imagery", "wg", "dsr"), "quota": n_trials},
        {"gpu": 1, "lane": 0, "tasks": ("visual",), "quota": first},
        {"gpu": 1, "lane": 2, "tasks": ("mental_arithmetic", "nback"), "quota": n_trials},
    ]
    if second:
        plan.extend([
            {"gpu": 0, "lane": 1, "tasks": ("refed_regression",), "quota": second},
            {"gpu": 1, "lane": 1, "tasks": ("visual",), "quota": second},
        ])
    return sorted(plan, key=lambda row: (int(row["gpu"]), int(row["lane"])))


def targeted_lane_plan(
    tasks: Sequence[str],
    n_trials: int,
    *,
    lanes_per_task: int = 3,
) -> list[dict[str, object]]:
    """Give up to two targeted tasks one GPU each and shard their trial quotas."""
    if not tasks or len(tasks) > 2:
        raise ValueError("targeted tuning requires one or two tasks")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if lanes_per_task <= 0:
        raise ValueError("lanes_per_task must be positive")
    plan: list[dict[str, object]] = []
    for gpu, task in enumerate(tasks):
        base_quota, remainder = divmod(n_trials, lanes_per_task)
        for lane in range(lanes_per_task):
            quota = base_quota + (1 if lane < remainder else 0)
            if quota:
                plan.append({
                    "gpu": gpu,
                    "lane": lane,
                    "tasks": (task,),
                    "quota": quota,
                    "worker_id": f"gpu{gpu}_{task}_lane{lane}",
                })
    return plan


def tuning_anchors(base: dict[str, Any], task: str) -> list[dict[str, Any]]:
    anchors = base.get("tuning_search", {}).get("anchors", {}).get(task, [])
    if not isinstance(anchors, list) or not all(isinstance(row, dict) for row in anchors):
        raise ValueError(f"tuning anchors for {task} must be a list of mappings")
    return [dict(row) for row in anchors]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default=datetime.now().strftime("%Y%m%d_%H%M%S_sta_net_hpo"))
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--startup-trials", type=int, default=6)
    parser.add_argument("--base-config", default=str(METHOD_ROOT / "configs" / "tuning_base.yaml"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--prune-after-epoch", type=int, default=None)
    args = parser.parse_args()
    run_root = METHOD_ROOT / "runs" / "tuning" / args.study_id
    run_root.mkdir(parents=True, exist_ok=False)
    storage = f"sqlite:///{(run_root / 'optuna.sqlite3').resolve()}"
    base_config = Path(args.base_config).resolve()
    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    tasks = tuple(dict.fromkeys(args.tasks))
    if set(tasks) == set(TASKS) and len(tasks) == len(TASKS):
        plan = lane_plan(args.n_trials)
    else:
        plan = targeted_lane_plan(tasks, args.n_trials)
    configured_prune_epoch = int(
        base.get("tuning_search", {}).get("prune_after_epoch", DEFAULT_PRUNE_AFTER_EPOCH)
    )
    prune_after_epoch = (
        configured_prune_epoch
        if args.prune_after_epoch is None
        else int(args.prune_after_epoch)
    )
    if prune_after_epoch not in RUNG_EPOCHS[:-1]:
        raise ValueError(
            f"prune_after_epoch must be one of {RUNG_EPOCHS[:-1]}, got {prune_after_epoch}"
        )
    enqueued_anchors: dict[str, int] = {}
    for task in tasks:
        study = optuna.create_study(
            study_name=f"{args.study_id}__{task}__development_cross_subject",
            storage=storage, direction="maximize", load_if_exists=True,
        )
        anchors = tuning_anchors(base, task)
        if len(anchors) > args.n_trials:
            raise ValueError(
                f"{task} defines {len(anchors)} anchors for only {args.n_trials} trials"
            )
        for anchor in anchors:
            study.enqueue_trial(anchor, skip_if_exists=True)
        enqueued_anchors[task] = len(anchors)
    launches = []
    for row in plan:
        gpu = int(row["gpu"])
        lane = int(row["lane"])
        row_tasks = tuple(str(task) for task in row["tasks"])
        quota = int(row["quota"])
        worker_id = str(row.get("worker_id", f"gpu{gpu}_lane{lane}"))
        command = [
            sys.executable, "-u", str(METHOD_ROOT / "tune.py"),
            "--study-id", args.study_id, "--run-root", str(run_root),
            "--base-config", str(base_config),
            "--physical-gpu", str(gpu), "--n-trials", str(quota),
            "--startup-trials", str(args.startup_trials),
            "--prune-after-epoch", str(prune_after_epoch),
            "--sampler-seed-offset", str(1000 * gpu + 100 * lane),
            "--worker-id", worker_id,
            "--tasks", *row_tasks,
        ]
        log_path = run_root / f"gpu{gpu}_lane{lane}_tuning.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=METHOD_ROOT.parents[1], stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        launches.append({
            "worker_id": worker_id, "physical_gpu": gpu, "lane": lane,
            "pid": process.pid, "tasks": list(row_tasks),
            "trial_quota_per_task": quota, "log": str(log_path),
        })
    manifest = {
        "schema": "sta_net_optuna_launch_v3", "status": "workers_launched",
        "tuning_schema": SCHEMA,
        "study_id": args.study_id, "run_root": str(run_root),
        "objective_policy": OBJECTIVE_POLICY,
        "rung_epochs": list(RUNG_EPOCHS), "n_trials_per_task": args.n_trials,
        "prune_after_epoch": prune_after_epoch,
        "tpe_startup_trials": args.startup_trials,
        "gpu_concurrency": {
            str(gpu): sum(1 for row in plan if int(row["gpu"]) == gpu)
            for gpu in sorted({int(row["gpu"]) for row in plan})
        },
        "trial_allocation": {task: args.n_trials for task in tasks},
        "enqueued_anchor_count": enqueued_anchors,
        "search_profile": str(base.get("tuning_search", {}).get("profile", "standard")),
        "base_config": str(base_config),
        "implementation_sha256": {
            "launcher": sha256(Path(__file__).resolve()),
            "tuner": sha256(METHOD_ROOT / "tune.py"),
            "trainer": sha256(METHOD_ROOT / "train.py"),
            "finalizer": sha256(METHOD_ROOT / "finalize_tuning.py"),
            "base_config": sha256(base_config),
        },
        "created_at": datetime.now(timezone.utc).isoformat(), "workers": launches,
    }
    (run_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    supervisor_command = [
        sys.executable, "-u", str(METHOD_ROOT / "finalize_tuning.py"),
        "--run-root", str(run_root),
    ]
    supervisor_log = run_root / "supervisor.log"
    with supervisor_log.open("w", encoding="utf-8") as log:
        supervisor = subprocess.Popen(
            supervisor_command,
            cwd=METHOD_ROOT.parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    manifest["supervisor"] = {
        "pid": supervisor.pid,
        "log": str(supervisor_log),
        "command": supervisor_command,
    }
    (run_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
