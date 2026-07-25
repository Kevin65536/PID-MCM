#!/usr/bin/env python3
"""Launch and supervise detached public-development EFRM transfer queues."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
TRAIN_ENTRYPOINT = METHOD_ROOT / "train_downstream.py"
TASKS = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
)
SCHEMA = "efrm_downstream_launcher_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_path(split_root: Path, task: str) -> Path:
    path = split_root / "splits" / task / "development_cross_subject.json"
    if not path.exists():
        raise FileNotFoundError(f"missing public split for {task}: {path}")
    manifest = read_json(path)
    if manifest.get("schema") != "sta_net_split_registry_v2":
        raise ValueError(f"expected v2 public split for {task}")
    if manifest.get("task") != task or manifest.get("protected_test_opened", False):
        raise PermissionError(f"invalid or opened public split for {task}")
    return path.resolve()


def _job_output(run_root: Path, job: Mapping[str, Any]) -> Path:
    return (
        run_root
        / str(job["task"])
        / str(job["transfer_mode"])
        / str(job["modality"])
        / str(job["initialization"])
    )


def worker(queue_path: Path) -> None:
    queue = read_json(queue_path)
    run_root = Path(queue["run_root"])
    worker_id = str(queue["worker_id"])
    physical_gpu = int(queue["physical_gpu"])
    status_path = run_root / "workers" / f"{worker_id}.json"
    completed: list[str] = []
    write_json(
        status_path,
        {
            "schema": SCHEMA,
            "status": "running",
            "worker_id": worker_id,
            "physical_gpu": physical_gpu,
            "pid": os.getpid(),
            "job_count": len(queue["jobs"]),
            "started_at": utc_now(),
        },
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    try:
        for job in queue["jobs"]:
            job_name = str(job["job_name"])
            output_dir = _job_output(run_root, job)
            output_dir.mkdir(parents=True, exist_ok=True)
            existing_status = output_dir / "status.json"
            if existing_status.exists() and read_json(existing_status).get("status") == "completed":
                completed.append(job_name)
                continue
            command = [
                sys.executable,
                "-u",
                str(TRAIN_ENTRYPOINT),
                "--config",
                str(queue["config"]),
                "--task",
                str(job["task"]),
                "--transfer-mode",
                str(job["transfer_mode"]),
                "--modality",
                str(job["modality"]),
                "--initialization",
                str(job["initialization"]),
                "--split-manifest",
                str(job["split_manifest"]),
                "--device",
                "cuda:0",
                "--output-dir",
                str(output_dir),
            ]
            if job["initialization"] == "pretrained":
                command.extend(
                    ["--pretrained-checkpoint", str(queue["pretrained_checkpoint"])]
                )
            latest = output_dir / "checkpoint_latest.pt"
            if latest.exists():
                command.extend(["--resume", str(latest)])
            write_json(
                status_path,
                {
                    "schema": SCHEMA,
                    "status": "running",
                    "worker_id": worker_id,
                    "physical_gpu": physical_gpu,
                    "pid": os.getpid(),
                    "active_job": job_name,
                    "completed_jobs": completed,
                    "job_count": len(queue["jobs"]),
                    "command": command,
                    "updated_at": utc_now(),
                },
            )
            with (output_dir / "process.log").open("a", encoding="utf-8") as log:
                completed_process = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed_process.returncode != 0:
                raise RuntimeError(
                    f"{job_name} failed with exit code {completed_process.returncode}"
                )
            completed.append(job_name)
        write_json(
            status_path,
            {
                "schema": SCHEMA,
                "status": "completed",
                "worker_id": worker_id,
                "physical_gpu": physical_gpu,
                "pid": os.getpid(),
                "completed_jobs": completed,
                "job_count": len(queue["jobs"]),
                "completed_at": utc_now(),
            },
        )
    except Exception as error:
        write_json(
            status_path,
            {
                "schema": SCHEMA,
                "status": "failed",
                "worker_id": worker_id,
                "physical_gpu": physical_gpu,
                "pid": os.getpid(),
                "completed_jobs": completed,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": utc_now(),
            },
        )
        raise


def start(args: argparse.Namespace) -> None:
    run_root = (METHOD_ROOT / "runs" / "downstream" / args.run_id).resolve()
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run already exists; use --resume: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    config = Path(args.config).resolve()
    checkpoint = Path(args.pretrained_checkpoint).resolve()
    split_root = Path(args.split_root).resolve()
    if not config.exists() or not checkpoint.exists():
        raise FileNotFoundError("config or pretrained checkpoint is missing")
    jobs: list[dict[str, Any]] = []
    for transfer_mode in args.transfer_modes:
        for task in args.tasks:
            for modality in args.modalities:
                for initialization in args.initializations:
                    jobs.append(
                        {
                            "job_name": (
                                f"{task}__{transfer_mode}__{modality}__{initialization}"
                            ),
                            "task": task,
                            "transfer_mode": transfer_mode,
                            "modality": modality,
                            "initialization": initialization,
                            "split_manifest": str(_split_path(split_root, task)),
                        }
                    )
    queues: list[dict[str, Any]] = []
    for index, physical_gpu in enumerate(args.gpus):
        worker_jobs = jobs[index :: len(args.gpus)]
        if not worker_jobs:
            continue
        queues.append(
            {
                "schema": SCHEMA,
                "run_id": args.run_id,
                "run_root": str(run_root),
                "worker_id": f"gpu{physical_gpu}",
                "physical_gpu": physical_gpu,
                "config": str(config),
                "pretrained_checkpoint": str(checkpoint),
                "jobs": worker_jobs,
            }
        )
    launch_rows = []
    for queue in queues:
        queue_path = run_root / "queues" / f"{queue['worker_id']}.json"
        write_json(queue_path, queue)
        log_path = run_root / "logs" / f"{queue['worker_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "worker", str(queue_path)],
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        log.close()
        launch_rows.append(
            {
                "worker_id": queue["worker_id"],
                "physical_gpu": queue["physical_gpu"],
                "pid": process.pid,
                "queue": str(queue_path),
                "log": str(log_path),
                "job_count": len(queue["jobs"]),
            }
        )
    manifest = {
        "schema": SCHEMA,
        "status": "launched",
        "scope": "public_development_pilot",
        "protected_test_opened": False,
        "run_id": args.run_id,
        "run_root": str(run_root),
        "config": str(config),
        "pretrained_checkpoint": str(checkpoint),
        "split_root": str(split_root),
        "tasks": args.tasks,
        "transfer_modes": args.transfer_modes,
        "modalities": args.modalities,
        "initializations": args.initializations,
        "job_count": len(jobs),
        "workers": launch_rows,
        "launched_at": utc_now(),
    }
    write_json(run_root / "launcher_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def status(run_id: str) -> None:
    run_root = (METHOD_ROOT / "runs" / "downstream" / run_id).resolve()
    manifest = read_json(run_root / "launcher_manifest.json")
    workers = []
    for path in sorted((run_root / "workers").glob("*.json")):
        workers.append(read_json(path))
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_root": str(run_root),
                "launcher": manifest,
                "workers": workers,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--run-id", required=True)
    start_parser.add_argument(
        "--config",
        default=str(METHOD_ROOT / "configs" / "downstream_public_pilot.yaml"),
    )
    start_parser.add_argument("--pretrained-checkpoint", required=True)
    start_parser.add_argument("--split-root", required=True)
    start_parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    start_parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    start_parser.add_argument(
        "--transfer-modes",
        nargs="+",
        choices=("linear_probe", "full_finetune"),
        default=["linear_probe", "full_finetune"],
    )
    start_parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("eeg", "fnirs", "paired"),
        default=["paired"],
    )
    start_parser.add_argument(
        "--initializations",
        nargs="+",
        choices=("pretrained", "scratch"),
        default=["pretrained"],
    )
    start_parser.add_argument("--resume", action="store_true")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("queue")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("run_id")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.command == "start":
        start(parsed)
    elif parsed.command == "worker":
        worker(Path(parsed.queue).resolve())
    else:
        status(parsed.run_id)
