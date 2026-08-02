#!/usr/bin/env python3
"""Execute an explicitly authorized BrainFusion public matrix serially."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence

import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import METHOD_ID
from audit_public_run_v2 import audit_run
from run_public_development_v2 import (
    load_runner_config,
    portable_path,
    resolve_repo_path,
    sha256_file,
    write_json,
)


LAUNCH_SCHEMA = "brainfusion_public_matrix_launch_v2"
MATRIX_SCHEMA = "brainfusion_public_job_matrix_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_controller_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("BrainFusion public matrix controller is already running") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_mapping(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected BrainFusion controller input: {resolved}")
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"BrainFusion controller input must be a mapping: {path}")
    return value


def load_launch(path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    launch_path = resolve_repo_path(path)
    launch = load_mapping(launch_path)
    if launch.get("schema") != LAUNCH_SCHEMA:
        raise ValueError(f"expected {LAUNCH_SCHEMA}: {launch_path}")
    if launch.get("method_id") != METHOD_ID or launch.get("mode") != "public_development_only":
        raise PermissionError("launch must remain BrainFusion public development only")
    authorization = launch.get("authorization", {})
    if authorization.get("public_matrix_launch_authorized") is not True:
        raise PermissionError("BrainFusion public matrix launch is not explicitly authorized")
    if authorization.get("protected_evaluation_authorized") is not False:
        raise PermissionError("BrainFusion controller cannot authorize protected evaluation")
    if authorization.get("normwear_work_authorized") is not False:
        raise PermissionError("BrainFusion launch cannot authorize concurrent NormWear work")
    if launch.get("protected_test_default") != "locked":
        raise PermissionError("protected test must remain locked")
    if int(launch["execution"]["max_concurrent_jobs"]) != 1:
        raise ValueError("BrainFusion controller requires max_concurrent_jobs=1")
    if int(launch["execution"]["automatic_retry_count"]) != 0:
        raise ValueError("BrainFusion controller does not admit automatic retries")
    if launch["execution"].get("device") != "cuda:1":
        raise ValueError("BrainFusion public matrix is frozen to GPU1")

    pilot_path = resolve_repo_path(launch["pilot_evidence"]["path"])
    if sha256_file(pilot_path) != str(launch["pilot_evidence"]["sha256"]):
        raise RuntimeError("BrainFusion full-fold pilot fingerprint drifted")
    pilot = load_mapping(pilot_path)
    if pilot.get("status") != launch["pilot_evidence"]["required_status"]:
        raise RuntimeError("BrainFusion full-fold pilot did not pass")
    if pilot.get("mode") != "public_development":
        raise RuntimeError("BrainFusion launch is not bound to a full-fold pilot")
    if pilot.get("cached_validation_matches_raw_adapter") is not True:
        raise RuntimeError("BrainFusion pilot did not verify the public tensor cache")
    if pilot.get("protected_test_opened") is not False:
        raise PermissionError("BrainFusion pilot reports protected access")

    controller_path = resolve_repo_path(launch["controller"]["path"])
    if controller_path != Path(__file__).resolve():
        raise RuntimeError("launch manifest names a different BrainFusion controller")
    if sha256_file(controller_path) != str(launch["controller"]["sha256"]):
        raise RuntimeError("BrainFusion controller source fingerprint drifted")
    matrix_path = resolve_repo_path(launch["matrix"]["path"])
    if sha256_file(matrix_path) != str(launch["matrix"]["sha256"]):
        raise RuntimeError("BrainFusion candidate matrix file fingerprint drifted")
    matrix = load_mapping(matrix_path)
    if matrix.get("schema") != MATRIX_SCHEMA or matrix.get("method_id") != METHOD_ID:
        raise ValueError("launch references an unexpected BrainFusion job matrix")
    if matrix.get("matrix_identity_sha256") != launch["matrix"]["identity_sha256"]:
        raise RuntimeError("launch matrix identity differs from the retained candidate")
    if int(launch["matrix"]["expected_job_count"]) != int(matrix.get("job_count", -1)):
        raise RuntimeError("launch expected job count differs from retained matrix")
    if int(matrix.get("job_count", -1)) != 75:
        raise ValueError("BrainFusion launch requires exactly 75 public jobs")
    if int(matrix.get("max_concurrent_jobs", -1)) != 1:
        raise ValueError("retained BrainFusion matrix is not serial")
    if int(matrix.get("automatic_retry_count", -1)) != 0:
        raise ValueError("retained BrainFusion matrix admits retries")
    if matrix.get("public_matrix_launch_authorized") is not False:
        raise PermissionError("candidate matrix improperly self-authorizes launch")
    if matrix.get("protected_evaluation_authorized") is not False:
        raise PermissionError("retained matrix crossed the protected boundary")
    if matrix.get("protected_test_opened") is not False:
        raise PermissionError("retained matrix reports protected access")
    if matrix.get("runner_sha256") != launch["runner"]["sha256"]:
        raise RuntimeError("launch runner identity differs from retained matrix")
    if matrix.get("runner_config_sha256") != launch["runner"]["config_sha256"]:
        raise RuntimeError("launch config identity differs from retained matrix")
    runner_path = resolve_repo_path(launch["runner"]["path"])
    config_path = resolve_repo_path(launch["runner"]["config"])
    if sha256_file(runner_path) != str(launch["runner"]["sha256"]):
        raise RuntimeError("BrainFusion runner source fingerprint drifted")
    if sha256_file(config_path) != str(launch["runner"]["config_sha256"]):
        raise RuntimeError("BrainFusion runner config fingerprint drifted")
    return launch, launch_path, matrix, matrix_path


def validate_jobs(
    matrix: Mapping[str, Any], *, run_root: Path
) -> list[Mapping[str, Any]]:
    jobs = list(matrix.get("jobs", ()))
    if len(jobs) != 75 or len(jobs) != int(matrix["job_count"]):
        raise ValueError("BrainFusion matrix job list length drifted")
    if [int(job["order"]) for job in jobs] != list(range(75)):
        raise ValueError("BrainFusion matrix job order is not contiguous")
    job_ids = [str(job["job_id"]) for job in jobs]
    if len(set(job_ids)) != 75:
        raise ValueError("BrainFusion matrix contains duplicate job identities")
    output_dirs: list[Path] = []
    expected_cells = []
    for job in jobs:
        command = [str(value) for value in job["command"]]
        if "protected" in " ".join(command).lower():
            raise PermissionError(f"public command crossed protected boundary: {job['job_id']}")
        if "--smoke" in command:
            raise ValueError(f"matrix job is not a full-fold run: {job['job_id']}")
        if command.count("--device") != 1 or command[command.index("--device") + 1] != "cuda:1":
            raise ValueError(f"matrix job is not frozen to GPU1: {job['job_id']}")
        output = resolve_repo_path(job["output_dir"])
        try:
            output.relative_to(run_root)
        except ValueError as exc:
            raise PermissionError(f"job output is outside matrix root: {output}") from exc
        if command[-2:] != ["--output-dir", portable_path(output)]:
            raise ValueError(f"job output argument differs from declaration: {job['job_id']}")
        output_dirs.append(output)
        expected_cells.append((str(job["task"]), int(job["outer_fold"]), int(job["seed"])))
    if len(set(output_dirs)) != 75:
        raise ValueError("BrainFusion matrix contains duplicate output directories")
    frozen_cells = [
        (task, fold, seed)
        for task in ("motor_imagery", "mental_arithmetic", "wg", "nback", "visual")
        for fold in range(5)
        for seed in (17, 42, 73)
    ]
    if expected_cells != frozen_cells:
        raise ValueError("BrainFusion matrix task/fold/seed order drifted")
    return jobs


def retained_failure(output_dir: Path) -> dict[str, Any] | None:
    status_path = output_dir / "status.json"
    if not status_path.is_file():
        return None
    status = load_mapping(status_path)
    if status.get("status") != "failed":
        return None
    return {
        "output_dir": portable_path(output_dir),
        "error_type": status.get("error_type"),
        "error": status.get("error"),
        "failed_at": status.get("failed_at"),
    }


def completed_report(output_dir: Path) -> bool:
    report_path = output_dir / "run_report.json"
    if not report_path.is_file():
        return False
    report = load_mapping(report_path)
    return (
        report.get("status") == "pass"
        and report.get("mode") == "public_development"
        and report.get("table_admissible") is False
        and report.get("protected_test_opened") is False
    )


def controller_status(
    *,
    launch_path: Path,
    matrix: Mapping[str, Any],
    state: str,
    next_order: int,
    completed: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "brainfusion_public_matrix_controller_status_v2",
        "status": state,
        "launch_path": portable_path(launch_path),
        "matrix_identity_sha256": matrix["matrix_identity_sha256"],
        "expected_job_count": 75,
        "completed_job_count": len(completed),
        "failed_job_count": len(failures),
        "next_job_order": next_order,
        "completed_jobs": list(completed),
        "failures": list(failures),
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
        "device": "cuda:1",
        "protected_test_opened": False,
        "updated_at": utc_now(),
    }


def _execute_unlocked(launch_path: str | Path, *, dry_run: bool) -> dict[str, Any]:
    launch, resolved_launch, matrix, _matrix_path = load_launch(launch_path)
    matrix_root = resolve_repo_path(launch["execution"]["matrix_run_root"])
    jobs = validate_jobs(matrix, run_root=matrix_root)
    config_path = resolve_repo_path(matrix["runner_config_path"])
    load_runner_config(config_path)
    if dry_run:
        return {
            "schema": "brainfusion_public_matrix_controller_dry_run_v2",
            "status": "pass",
            "job_count": len(jobs),
            "max_concurrent_jobs": 1,
            "automatic_retry_count": 0,
            "device": "cuda:1",
            "public_matrix_launch_authorized": True,
            "protected_evaluation_authorized": False,
            "normwear_work_authorized": False,
            "protected_test_opened": False,
        }

    status_path = matrix_root / "controller_status.json"
    completed_audit_path = matrix_root / "completed_public_audit.json"
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, job in enumerate(jobs):
        output_dir = resolve_repo_path(job["output_dir"])
        failure = retained_failure(output_dir)
        if failure is not None:
            failures.append({"job_id": job["job_id"], **failure})
            stopped = controller_status(
                launch_path=resolved_launch,
                matrix=matrix,
                state="stopped_on_retained_failure",
                next_order=position,
                completed=completed,
                failures=failures,
            )
            write_json(status_path, stopped)
            return stopped

        if completed_report(output_dir):
            try:
                audited = audit_run(
                    output_dir / "run_report.json", config_path=config_path, device="cuda:1"
                )
                write_json(output_dir / "artifact_audit.json", audited)
            except Exception as exc:
                failure = {
                    "job_id": job["job_id"],
                    "output_dir": portable_path(output_dir),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at": utc_now(),
                }
                failures.append(failure)
                stopped = controller_status(
                    launch_path=resolved_launch,
                    matrix=matrix,
                    state="stopped_on_audit_failure",
                    next_order=position,
                    completed=completed,
                    failures=failures,
                )
                write_json(status_path, stopped)
                return stopped
            completed.append({"job_id": job["job_id"], **audited})
            print(f"[{position + 1}/75] retained pass {job['job_id']}", flush=True)
            continue

        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"job has nonterminal retained output: {output_dir}")
        running = controller_status(
            launch_path=resolved_launch,
            matrix=matrix,
            state="running",
            next_order=position,
            completed=completed,
            failures=failures,
        )
        running["current_job_id"] = str(job["job_id"])
        write_json(status_path, running)
        print(f"[{position + 1}/75] start {job['job_id']}", flush=True)
        result = subprocess.run([str(value) for value in job["command"]], cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            failure_status = {
                "schema": "brainfusion_public_job_failure_v2",
                "status": "failed",
                "job_id": job["job_id"],
                "error_type": "RunnerExitCode",
                "error": f"runner exited with code {result.returncode}",
                "failed_at": utc_now(),
                "automatic_retry_count": 0,
                "protected_test_opened": False,
            }
            write_json(output_dir / "status.json", failure_status)
            failures.append(
                {
                    "job_id": job["job_id"],
                    "output_dir": portable_path(output_dir),
                    **failure_status,
                }
            )
            stopped = controller_status(
                launch_path=resolved_launch,
                matrix=matrix,
                state="stopped_on_failure",
                next_order=position,
                completed=completed,
                failures=failures,
            )
            write_json(status_path, stopped)
            return stopped
        try:
            audited = audit_run(
                output_dir / "run_report.json", config_path=config_path, device="cuda:1"
            )
            write_json(output_dir / "artifact_audit.json", audited)
        except Exception as exc:
            failure_status = {
                "schema": "brainfusion_public_job_failure_v2",
                "status": "failed",
                "job_id": job["job_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": utc_now(),
                "automatic_retry_count": 0,
                "protected_test_opened": False,
            }
            write_json(output_dir / "status.json", failure_status)
            failures.append(
                {
                    "job_id": job["job_id"],
                    "output_dir": portable_path(output_dir),
                    **failure_status,
                }
            )
            stopped = controller_status(
                launch_path=resolved_launch,
                matrix=matrix,
                state="stopped_on_audit_failure",
                next_order=position,
                completed=completed,
                failures=failures,
            )
            write_json(status_path, stopped)
            return stopped
        completed.append({"job_id": job["job_id"], **audited})
        write_json(
            status_path,
            controller_status(
                launch_path=resolved_launch,
                matrix=matrix,
                state="running",
                next_order=position + 1,
                completed=completed,
                failures=failures,
            ),
        )
        print(f"[{position + 1}/75] pass {job['job_id']}", flush=True)

    final = controller_status(
        launch_path=resolved_launch,
        matrix=matrix,
        state="completed",
        next_order=75,
        completed=completed,
        failures=failures,
    )
    write_json(status_path, final)
    write_json(
        completed_audit_path,
        {
            "schema": "brainfusion_public_matrix_completed_audit_v2",
            "status": "pass",
            "matrix_identity_sha256": matrix["matrix_identity_sha256"],
            "job_count": 75,
            "run_reports": completed,
            "protected_test_opened": False,
            "completed_at": utc_now(),
        },
    )
    return final


def execute(launch_path: str | Path, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return _execute_unlocked(launch_path, dry_run=True)
    launch, _launch_path, _matrix, _matrix_path = load_launch(launch_path)
    matrix_root = resolve_repo_path(launch["execution"]["matrix_run_root"])
    with exclusive_controller_lock(matrix_root / ".controller.lock"):
        return _execute_unlocked(launch_path, dry_run=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = execute(args.launch, dry_run=bool(args.dry_run))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] in {"pass", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
