#!/usr/bin/env python3
"""Run a sequence of STA-Net tasks on one physical GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parent


def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    status_path = run_root / f"gpu{args.physical_gpu}_queue_status.json"
    completed: list[str] = []
    for position, task in enumerate(args.tasks):
        task_dir = run_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        log_path = task_dir / "process.log"
        command = [
            sys.executable, "-u", str(METHOD_ROOT / "train.py"),
            "--config", str(Path(args.config).resolve()), "--task", task,
            "--device", "cuda:0", "--output-dir", str(task_dir),
        ]
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(command, cwd=METHOD_ROOT.parents[1], stdout=log_handle, stderr=subprocess.STDOUT)
            write_status(status_path, {
                "schema": "sta_net_gpu_queue_v1", "status": "running", "physical_gpu": args.physical_gpu,
                "queue_pid": os.getpid(), "active_task": task, "active_pid": process.pid,
                "position": position, "tasks": args.tasks, "completed_tasks": completed,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            return_code = process.wait()
        if return_code != 0:
            write_status(status_path, {
                "schema": "sta_net_gpu_queue_v1", "status": "failed", "physical_gpu": args.physical_gpu,
                "queue_pid": os.getpid(), "failed_task": task, "return_code": return_code,
                "tasks": args.tasks, "completed_tasks": completed,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            raise SystemExit(return_code)
        completed.append(task)
    write_status(status_path, {
        "schema": "sta_net_gpu_queue_v1", "status": "completed", "physical_gpu": args.physical_gpu,
        "queue_pid": os.getpid(), "tasks": args.tasks, "completed_tasks": completed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
