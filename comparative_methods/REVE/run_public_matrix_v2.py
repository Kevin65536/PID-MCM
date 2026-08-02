#!/usr/bin/env python3
"""Execute an explicitly authorized REVE public matrix one job at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.REVE.audit_public_run_v2 import audit_run, utc_now
from comparative_methods.REVE.run_public_development_v2 import (
    load_runner_config,
    portable_path,
    resolve_repo_path,
    sha256_file,
    write_json,
)


LAUNCH_SCHEMA = "reve_public_matrix_launch_v2"
MATRIX_SCHEMA = "reve_public_job_matrix_v2"


def load_mapping(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected controller input: {resolved}")
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"controller input must be a mapping: {path}")
    return value


def load_launch(path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    launch_path = resolve_repo_path(path)
    launch = load_mapping(launch_path)
    if launch.get("schema") != LAUNCH_SCHEMA:
        raise ValueError(f"expected {LAUNCH_SCHEMA}: {launch_path}")
    if launch.get("method_id") != "reve" or launch.get("mode") != "public_development_only":
        raise PermissionError("launch must remain REVE public development only")
    authorization = launch.get("authorization", {})
    if authorization.get("public_matrix_launch_authorized") is not True:
        raise PermissionError("REVE public matrix launch is not explicitly authorized")
    if authorization.get("protected_evaluation_authorized") is not False:
        raise PermissionError("REVE public controller cannot authorize protected evaluation")
    if authorization.get("brainfusion_work_authorized") is not False:
        raise PermissionError("REVE public launch cannot authorize concurrent BrainFusion work")
    if launch.get("protected_test_default") != "locked":
        raise PermissionError("protected test must remain locked")
    if int(launch["execution"]["max_concurrent_jobs"]) != 1:
        raise ValueError("REVE matrix controller requires max_concurrent_jobs=1")
    if int(launch["execution"]["automatic_retry_count"]) != 0:
        raise ValueError("REVE matrix controller does not admit automatic retries")
    pilot_path = resolve_repo_path(launch["pilot_evidence"]["path"])
    if sha256_file(pilot_path) != str(launch["pilot_evidence"]["sha256"]):
        raise RuntimeError("REVE public pilot evidence fingerprint drifted")
    pilot = load_mapping(pilot_path)
    if pilot.get("status") != launch["pilot_evidence"]["required_status"]:
        raise RuntimeError("REVE public pilot does not have the required reviewed status")
    if pilot.get("protected_test_opened") is not False:
        raise PermissionError("REVE public pilot reports protected access")
    worker_path = resolve_repo_path(launch["controller"]["path"])
    if worker_path != Path(__file__).resolve():
        raise RuntimeError("launch manifest names a different REVE controller")
    if sha256_file(worker_path) != str(launch["controller"]["sha256"]):
        raise RuntimeError("REVE controller source fingerprint drifted")
    matrix_path = resolve_repo_path(launch["matrix"]["path"])
    if sha256_file(matrix_path) != str(launch["matrix"]["sha256"]):
        raise RuntimeError("REVE candidate matrix file fingerprint drifted")
    matrix = load_mapping(matrix_path)
    if matrix.get("schema") != MATRIX_SCHEMA or matrix.get("method_id") != "reve":
        raise ValueError("launch references an unexpected job matrix")
    if matrix.get("matrix_identity_sha256") != launch["matrix"]["identity_sha256"]:
        raise RuntimeError("launch matrix identity differs from the retained candidate")
    if int(launch["matrix"]["expected_job_count"]) != int(matrix.get("job_count", -1)):
        raise RuntimeError("launch expected job count differs from the retained matrix")
    if int(matrix.get("job_count", -1)) != 90:
        raise ValueError("REVE launch requires exactly 90 public jobs")
    if int(matrix.get("max_concurrent_jobs", -1)) != 1:
        raise ValueError("retained REVE matrix is not serial")
    if int(matrix.get("automatic_retry_count", -1)) != 0:
        raise ValueError("retained REVE matrix admits retries")
    if matrix.get("protected_evaluation_authorized") is not False:
        raise PermissionError("retained matrix crossed the protected boundary")
    if matrix.get("protected_test_opened") is not False:
        raise PermissionError("retained matrix reports protected access")
    if matrix.get("runner_sha256") != launch["runner"]["sha256"]:
        raise RuntimeError("launch runner identity differs from the retained matrix")
    if matrix.get("runner_config_sha256") != launch["runner"]["config_sha256"]:
        raise RuntimeError("launch config identity differs from the retained matrix")
    runner_path = resolve_repo_path(launch["runner"]["path"])
    config_path = resolve_repo_path(launch["runner"]["config"])
    if sha256_file(runner_path) != str(launch["runner"]["sha256"]):
        raise RuntimeError("REVE runner source fingerprint drifted")
    if sha256_file(config_path) != str(launch["runner"]["config_sha256"]):
        raise RuntimeError("REVE runner config fingerprint drifted")
    return launch, launch_path, matrix, matrix_path


def validate_jobs(
    matrix: Mapping[str, Any], *, run_root: Path
) -> list[Mapping[str, Any]]:
    jobs = list(matrix.get("jobs", ()))
    if len(jobs) != int(matrix["job_count"]):
        raise ValueError("REVE matrix job list length drifted")
    if [int(job["order"]) for job in jobs] != list(range(len(jobs))):
        raise ValueError("REVE matrix job order is not contiguous")
    job_ids = [str(job["job_id"]) for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("REVE matrix contains duplicate job identities")
    output_dirs: list[Path] = []
    for job in jobs:
        command = [str(value) for value in job["command"]]
        if "protected" in " ".join(command).lower():
            raise PermissionError(f"public command crossed protected boundary: {job['job_id']}")
        output = resolve_repo_path(job["output_dir"])
        try:
            output.relative_to(run_root)
        except ValueError as exc:
            raise PermissionError(f"job output is outside matrix root: {output}") from exc
        if command[-2:] != ["--output-dir", portable_path(output)]:
            raise ValueError(f"job output argument differs from its declaration: {job['job_id']}")
        output_dirs.append(output)
    if len(set(output_dirs)) != len(output_dirs):
        raise ValueError("REVE matrix contains duplicate output directories")
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


def completed_manifest(output_dir: Path) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = load_mapping(manifest_path)
    return manifest.get("status") == "completed" and not manifest.get(
        "protected_test_opened", False
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
        "schema": "reve_public_matrix_controller_status_v2",
        "status": state,
        "launch_path": portable_path(launch_path),
        "matrix_identity_sha256": matrix["matrix_identity_sha256"],
        "expected_job_count": int(matrix["job_count"]),
        "completed_job_count": len(completed),
        "failed_job_count": len(failures),
        "next_job_order": next_order,
        "completed_jobs": list(completed),
        "failures": list(failures),
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
        "protected_test_opened": False,
        "updated_at": utc_now(),
    }


def execute(launch_path: str | Path, *, dry_run: bool) -> dict[str, Any]:
    launch, resolved_launch, matrix, _matrix_path = load_launch(launch_path)
    matrix_root = resolve_repo_path(launch["execution"]["matrix_run_root"])
    jobs = validate_jobs(matrix, run_root=matrix_root)
    config_path = resolve_repo_path(matrix["runner_config_path"])
    config, resolved_config, alignment, alignment_path = load_runner_config(config_path)
    if dry_run:
        return {
            "schema": "reve_public_matrix_controller_dry_run_v2",
            "status": "pass",
            "job_count": len(jobs),
            "max_concurrent_jobs": 1,
            "automatic_retry_count": 0,
            "public_matrix_launch_authorized": True,
            "protected_evaluation_authorized": False,
            "protected_test_opened": False,
        }

    status_path = matrix_root / "controller_status.json"
    audit_path = matrix_root / "completed_public_audit.json"
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, job in enumerate(jobs):
        output_dir = resolve_repo_path(job["output_dir"])
        failure = retained_failure(output_dir)
        if failure is not None:
            failures.append({"job_id": job["job_id"], **failure})
            status = controller_status(
                launch_path=resolved_launch,
                matrix=matrix,
                state="stopped_on_retained_failure",
                next_order=position,
                completed=completed,
                failures=failures,
            )
            write_json(status_path, status)
            return status
        if completed_manifest(output_dir):
            report = audit_run(
                output_dir,
                config=config,
                config_path=resolved_config,
                alignment=alignment,
                alignment_path=alignment_path,
            )
            completed.append({"job_id": job["job_id"], **report})
            print(f"[{position + 1}/{len(jobs)}] retained pass {job['job_id']}", flush=True)
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
        print(f"[{position + 1}/{len(jobs)}] start {job['job_id']}", flush=True)
        result = subprocess.run(
            [str(value) for value in job["command"]],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            failure = retained_failure(output_dir) or {
                "output_dir": portable_path(output_dir),
                "error_type": "RunnerExitCode",
                "error": f"runner exited with code {result.returncode}",
            }
            failures.append({"job_id": job["job_id"], **failure})
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
        report = audit_run(
            output_dir,
            config=config,
            config_path=resolved_config,
            alignment=alignment,
            alignment_path=alignment_path,
        )
        completed.append({"job_id": job["job_id"], **report})
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
        print(f"[{position + 1}/{len(jobs)}] pass {job['job_id']}", flush=True)

    final = controller_status(
        launch_path=resolved_launch,
        matrix=matrix,
        state="completed",
        next_order=len(jobs),
        completed=completed,
        failures=failures,
    )
    write_json(status_path, final)
    write_json(
        audit_path,
        {
            "schema": "reve_public_matrix_completed_audit_v2",
            "status": "pass",
            "matrix_identity_sha256": matrix["matrix_identity_sha256"],
            "job_count": len(jobs),
            "run_reports": completed,
            "protected_test_opened": False,
            "completed_at": utc_now(),
        },
    )
    return final


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
