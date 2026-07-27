#!/usr/bin/env python3
"""Detached, resumable orchestrator for the frozen 70-job EFRM protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
PROTOCOL_ID = "efrm_resource_bounded_dual_protocol_v1"
DEFAULT_ROOT = METHOD_ROOT / f"runs/formal/{PROTOCOL_ID}"

for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_formal_protocol import build_folds
from train_downstream import sha256_file, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_status(root: Path, **values: Any) -> dict[str, Any]:
    path = root / "status.json"
    status = read_json(path)
    status.update(values)
    write_json(path, status)
    return status


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] command: {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def freeze_source_checkpoint(root: Path, checkpoint: Path) -> dict[str, Any]:
    run_dir = checkpoint.parent.parent
    manifest = read_json(run_dir / "manifest.json")
    boundary = read_json(run_dir / "boundary_manifest.json")
    cohort_path = root / "protocol/cohort_manifest.json"
    cohort = read_json(cohort_path)
    if manifest.get("status") != "completed":
        raise RuntimeError("source-only pretraining manifest is not completed")
    if (
        manifest.get("protected_test_opened") is not False
        or boundary.get("protected_test_opened") is not False
        or boundary.get("target_opened_during_pretraining") is not False
    ):
        raise PermissionError("source-only pretraining reports target/protected access")
    if boundary.get("mode") != "source_target_source_only_v1":
        raise RuntimeError("source checkpoint does not use the frozen source-only boundary")
    if boundary.get("cohort_manifest_sha256") != sha256_file(cohort_path):
        raise RuntimeError("source checkpoint cohort hash does not match frozen cohort")
    for dataset_id, row in cohort["datasets"].items():
        expected = {
            "train_subjects_by_dataset": row["source_train_subjects"],
            "validation_subjects_by_dataset": row["source_validation_subjects"],
            "target_subjects_by_dataset": row["target_subjects"],
        }
        for field, subjects in expected.items():
            if sorted(boundary[field][dataset_id]) != sorted(subjects):
                raise RuntimeError(f"source boundary {field}/{dataset_id} drifted")
    checkpoint_hash = sha256_file(checkpoint)
    frozen = {
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_run_manifest": str((run_dir / "manifest.json").resolve()),
        "source_run_manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "source_boundary_manifest": str((run_dir / "boundary_manifest.json").resolve()),
        "source_boundary_manifest_sha256": sha256_file(
            run_dir / "boundary_manifest.json"
        ),
        "source_resolved_config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
        "source_best_validation_loss": float(
            __import__("torch").load(
                checkpoint, map_location="cpu", weights_only=False
            )["best_validation_loss"]
        ),
    }
    write_json(root / "protocol/source_checkpoint_freeze.json", frozen)
    return update_status(
        root,
        status="source_pretraining_completed",
        source_pretraining_completed_at=utc_now(),
        **frozen,
    )


def run_source_pretraining(root: Path, gpu: str) -> Path:
    run_id = f"{PROTOCOL_ID}__source_seed42_chunk8"
    run_dir = METHOD_ROOT / "runs/pretraining" / run_id
    config = root / "protocol/configs/pretrain_source_only.yaml"
    command = [
        str(PYTHON),
        str(METHOD_ROOT / "train_pretrain.py"),
        "--config",
        str(config),
        "--run-id",
        run_id,
        "--device",
        f"cuda:{gpu}",
        "--chunk-size",
        "8",
        "--num-workers",
        "0",
    ]
    if run_dir.exists():
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path).get("status") == "completed":
            return run_dir / "checkpoints/best.pt"
        if not (run_dir / "checkpoints/latest.pt").is_file():
            raise RuntimeError(
                f"source run exists without a resumable checkpoint: {run_dir}"
            )
        command.append("--resume")
    update_status(
        root,
        status="source_pretraining_running",
        source_pretraining_started_at=utc_now(),
        source_run_id=run_id,
        worker_pid=os.getpid(),
    )
    run_logged(command, root / "logs/source_pretraining.log")
    checkpoint = run_dir / "checkpoints/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("source-only pretraining did not produce best.pt")
    return checkpoint


def public_job(root: Path, matrix: dict[str, Any], job: dict[str, Any], gpu: str) -> None:
    job_dir = root / "jobs" / str(job["job_id"])
    manifest_path = job_dir / "manifest.json"
    if manifest_path.exists():
        status = read_json(manifest_path).get("status")
        if status in {"completed", "protected_evaluation_completed"}:
            return
    command = [
        str(PYTHON),
        str(METHOD_ROOT / "train_downstream.py"),
        "--config",
        str(matrix["downstream_config"]),
        "--task",
        str(job["task"]),
        "--transfer-mode",
        "linear_probe",
        "--modality",
        "paired",
        "--initialization",
        "pretrained",
        "--pretrained-checkpoint",
        str(matrix["source_checkpoint"]),
        "--split-manifest",
        str(job["public_manifest"]),
        "--device",
        f"cuda:{gpu}",
        "--output-dir",
        str(job_dir),
    ]
    checkpoint_latest = job_dir / "checkpoint_latest.pt"
    if checkpoint_latest.is_file():
        command.extend(["--resume", str(checkpoint_latest)])
    run_logged(command, root / "logs/jobs" / f"{job['job_id']}.public.log")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise RuntimeError(f"public job did not complete: {job['job_id']}")


def protected_job(root: Path, job: dict[str, Any], gpu: str) -> None:
    job_dir = root / "jobs" / str(job["job_id"])
    manifest_path = job_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") == "protected_evaluation_completed":
        return
    command = [
        str(PYTHON),
        str(METHOD_ROOT / "evaluate_formal_fold.py"),
        "--job-dir",
        str(job_dir),
        "--protected-manifest",
        str(job["protected_manifest"]),
        "--device",
        f"cuda:{gpu}",
    ]
    run_logged(command, root / "logs/jobs" / f"{job['job_id']}.protected.log")
    if read_json(manifest_path).get("status") != "protected_evaluation_completed":
        raise RuntimeError(f"protected job did not complete: {job['job_id']}")


def run_parallel_jobs(
    root: Path,
    matrix: dict[str, Any],
    gpus: list[str],
    *,
    protected: bool,
) -> None:
    jobs = list(matrix["jobs"])
    failures: list[str] = []

    def run_queue(queue_index: int) -> None:
        gpu = gpus[queue_index]
        for position in range(queue_index, len(jobs), len(gpus)):
            job = jobs[position]
            if protected:
                protected_job(root, job, gpu)
            else:
                public_job(root, matrix, job, gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(run_queue, queue_index): queue_index
            for queue_index in range(len(gpus))
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                failures.append(traceback.format_exc())
    if failures:
        raise RuntimeError("\n".join(failures))


def worker(root: Path, gpus: list[str]) -> None:
    try:
        status = read_json(root / "status.json")
        if status["status"] in {
            "source_cohort_materialized",
            "source_pretraining_running",
        }:
            checkpoint = run_source_pretraining(root, gpus[0])
            status = freeze_source_checkpoint(root, checkpoint)
        if status["status"] == "source_pretraining_completed":
            checkpoint = Path(status["source_checkpoint"])
            status = build_folds(
                root,
                str(read_json(root / "protocol/cohort_manifest.json")["cache_root"]),
                checkpoint,
            )
        matrix = read_json(root / "protocol/job_matrix.json")
        if status["status"] in {
            "folds_jobs_metrics_frozen",
            "public_training_running",
        }:
            update_status(
                root,
                status="public_training_running",
                public_training_started_at=utc_now(),
            )
            run_parallel_jobs(root, matrix, gpus, protected=False)
            status = update_status(
                root,
                status="public_training_completed",
                public_training_completed_at=utc_now(),
                completed_public_jobs=70,
                protected_test_opened=False,
            )
        if status["status"] in {
            "public_training_completed",
            "protected_evaluation_running",
        }:
            update_status(
                root,
                status="protected_evaluation_running",
                protected_evaluation_started_at=utc_now(),
                protected_test_opened=True,
            )
            run_parallel_jobs(root, matrix, gpus, protected=True)
            update_status(
                root,
                status="protected_evaluation_completed",
                protected_evaluation_completed_at=utc_now(),
                completed_protected_jobs=70,
                protected_test_opened=True,
            )
        run_logged(
            [
                str(PYTHON),
                str(METHOD_ROOT / "aggregate_formal_results.py"),
                "--root",
                str(root),
            ],
            root / "logs/aggregate.log",
        )
    except Exception as error:
        update_status(
            root,
            status="failed",
            failed_at=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise


def start(root: Path, gpus: list[str]) -> dict[str, Any]:
    launcher_log = root / "logs/orchestrator.log"
    launcher_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "worker",
        "--root",
        str(root),
        "--gpus",
        *gpus,
    ]
    with launcher_log.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return update_status(
        root,
        orchestrator_pid=process.pid,
        orchestrator_started_at=utc_now(),
        assigned_gpus=gpus,
        orchestrator_log=str(launcher_log.resolve()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "worker"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", default=str(DEFAULT_ROOT))
        child.add_argument("--gpus", nargs="+", default=["0", "1"])
    status = subparsers.add_parser("status")
    status.add_argument("--root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if args.command == "status":
        print(json.dumps(read_json(root / "status.json"), indent=2))
    elif args.command == "start":
        print(json.dumps(start(root, [str(value) for value in args.gpus]), indent=2))
    else:
        worker(root, [str(value) for value in args.gpus])


if __name__ == "__main__":
    main()
