#!/usr/bin/env python3
"""Launch two detached per-GPU queues covering all STA-Net tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parent
GPU_QUEUES = {
    0: ("dsr", "refed_regression", "motor_imagery"),
    1: ("visual", "wg", "mental_arithmetic", "nback"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(METHOD_ROOT / "configs" / "train_all_tasks.yaml"))
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S_all_tasks"))
    args = parser.parse_args()
    run_root = METHOD_ROOT / "runs" / "training" / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    launches = []
    for physical_gpu, tasks in GPU_QUEUES.items():
        log_path = run_root / f"gpu{physical_gpu}_queue.log"
        command = [
            sys.executable, "-u", str(METHOD_ROOT / "queue_worker.py"),
            "--config", str(Path(args.config).resolve()), "--run-root", str(run_root),
            "--physical-gpu", str(physical_gpu), "--tasks", *tasks,
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(physical_gpu), "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        })
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=METHOD_ROOT.parents[1], env=environment,
            stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log_handle.close()
        launches.append({
            "physical_gpu": physical_gpu, "queue_pid": process.pid, "tasks": list(tasks),
            "log": str(log_path), "command": command,
        })
    manifest = {
        "schema": "sta_net_pytorch_queued_launch_v2", "status": "queues_launched",
        "created_at": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
        "config": str(Path(args.config).resolve()), "tensorflow_used": False, "queues": launches,
        "policy": "one active training task per physical GPU; remaining tasks run sequentially",
    }
    manifest_path = run_root / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_root": str(run_root), "manifest": str(manifest_path), "queues": launches}, indent=2))


if __name__ == "__main__":
    main()
