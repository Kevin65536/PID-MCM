#!/usr/bin/env python3
"""Launch distant-tail helper lanes for an active within-subject run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parent


def digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--tail-start", type=int, default=90)
    parser.add_argument("--tail-end", type=int, default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--lanes-per-gpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--unlock-protected-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_protected_test:
        raise RuntimeError("auxiliary lanes require explicit protected-test unlock")
    root = Path(args.run_root).resolve()
    original_paths = sorted((root / "jobs").glob("lane_[0-9][0-9].json"))
    if not original_paths:
        raise RuntimeError("no original lane job manifests found")
    tail_jobs = []
    for path in original_paths:
        jobs = json.loads(path.read_text(encoding="utf-8"))
        tail_jobs.extend(jobs[args.tail_start:args.tail_end])
    keys = [f"{job['task']}/{job['fold_id']}" for job in tail_jobs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("auxiliary tail selection contains duplicate jobs")
    started = [
        key for key, job in zip(keys, tail_jobs, strict=True)
        if (root / "folds" / job["task"] / job["fold_id"]).exists()
    ]
    if started:
        raise RuntimeError(f"tail safety margin is insufficient; already started={started[:5]}")
    lane_count = len(args.gpus) * args.lanes_per_gpu
    lanes = [[] for _ in range(lane_count)]
    for index, job in enumerate(tail_jobs):
        lanes[index % lane_count].append(job)
    launches = []
    for index, lane_jobs in enumerate(lanes):
        lane_id = f"aux_{args.tail_start}_{index:02d}"
        jobs_path = root / "jobs" / f"{lane_id}.json"
        write_json(jobs_path, lane_jobs)
        gpu = args.gpus[index % len(args.gpus)]
        log_path = root / "logs" / f"{lane_id}.log"
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
            "job_count": len(lane_jobs),
            "jobs": str(jobs_path),
            "log": str(log_path),
        })
    amendment = {
        "schema": "sta_net_within_subject_execution_amendment_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "add distant-tail helper lanes because REFED input loading left GPU capacity idle",
        "scientific_protocol_changed": False,
        "unchanged_factors": [
            "model configuration", "optimizer configuration", "epochs", "seed",
            "public/protected splits", "checkpoint selection", "primary endpoints",
        ],
        "tail_start_per_original_lane": args.tail_start,
        "tail_end_per_original_lane": args.tail_end,
        "job_count": len(tail_jobs),
        "jobs_sha256": digest(tail_jobs),
        "original_lane_count": len(original_paths),
        "auxiliary_lanes": launches,
        "race_safety": "all selected fold directories were absent before launch; original lanes skip completed summaries",
    }
    range_name = f"{args.tail_start}_{args.tail_end if args.tail_end is not None else 'tail'}"
    write_json(root / f"execution_amendment_auxiliary_lanes_{range_name}.json", amendment)
    print(json.dumps(amendment, indent=2))


if __name__ == "__main__":
    main()
