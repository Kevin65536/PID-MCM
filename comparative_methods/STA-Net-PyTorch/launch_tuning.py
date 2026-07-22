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

import optuna

from tune import OBJECTIVE_POLICY, RUNG_EPOCHS, SCHEMA

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default=datetime.now().strftime("%Y%m%d_%H%M%S_sta_net_hpo"))
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--startup-trials", type=int, default=6)
    parser.add_argument("--base-config", default=str(METHOD_ROOT / "configs" / "tuning_base.yaml"))
    args = parser.parse_args()
    run_root = METHOD_ROOT / "runs" / "tuning" / args.study_id
    run_root.mkdir(parents=True, exist_ok=False)
    storage = f"sqlite:///{(run_root / 'optuna.sqlite3').resolve()}"
    base_config = Path(args.base_config).resolve()
    plan = lane_plan(args.n_trials)
    for task in TASKS:
        optuna.create_study(
            study_name=f"{args.study_id}__{task}__development_cross_subject",
            storage=storage, direction="maximize", load_if_exists=True,
        )
    launches = []
    for row in plan:
        gpu = int(row["gpu"])
        lane = int(row["lane"])
        tasks = tuple(str(task) for task in row["tasks"])
        quota = int(row["quota"])
        worker_id = f"gpu{gpu}_lane{lane}"
        command = [
            sys.executable, "-u", str(METHOD_ROOT / "tune.py"),
            "--study-id", args.study_id, "--run-root", str(run_root),
            "--base-config", str(base_config),
            "--physical-gpu", str(gpu), "--n-trials", str(quota),
            "--startup-trials", str(args.startup_trials),
            "--sampler-seed-offset", str(1000 * gpu + 100 * lane),
            "--worker-id", worker_id,
            "--tasks", *tasks,
        ]
        log_path = run_root / f"gpu{gpu}_lane{lane}_tuning.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=METHOD_ROOT.parents[1], stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        launches.append({
            "worker_id": worker_id, "physical_gpu": gpu, "lane": lane,
            "pid": process.pid, "tasks": list(tasks),
            "trial_quota_per_task": quota, "log": str(log_path),
        })
    manifest = {
        "schema": "sta_net_optuna_launch_v2", "status": "workers_launched",
        "tuning_schema": SCHEMA,
        "study_id": args.study_id, "run_root": str(run_root),
        "objective_policy": OBJECTIVE_POLICY,
        "rung_epochs": list(RUNG_EPOCHS), "n_trials_per_task": args.n_trials,
        "tpe_startup_trials": args.startup_trials,
        "gpu_concurrency": {"0": 3, "1": 3},
        "trial_allocation": {task: args.n_trials for task in TASKS},
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
