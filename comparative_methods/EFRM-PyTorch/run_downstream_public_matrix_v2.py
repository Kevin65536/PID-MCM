#!/usr/bin/env python3
"""Execute an explicitly authorized EFRM LODO-v2 public matrix serially."""

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
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from audit_downstream_public_run_v2 import audit as audit_run  # noqa: E402
from build_downstream_public_matrix_v2 import MATRIX_SCHEMA  # noqa: E402
from run_downstream_public_v2 import (  # noqa: E402
    METHOD_ID,
    PROTOCOL_ID,
    portable_path,
    resolve_repo_path,
    sha256_file,
    utc_now,
    write_json,
)


LAUNCH_SCHEMA = "efrm_lodo_downstream_public_launch_v2"
STATUS_SCHEMA = "efrm_lodo_downstream_public_controller_status_v2"
COMPLETED_SCHEMA = "efrm_lodo_downstream_public_completed_audit_v2"


def load_mapping(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected EFRM controller input: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.suffix.lower() == ".json":
        value = json.loads(resolved.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"EFRM controller input must be a mapping: {resolved}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_launch(
    launch_path: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    resolved_launch = resolve_repo_path(launch_path)
    launch = load_mapping(resolved_launch)
    require(launch.get("schema") == LAUNCH_SCHEMA, "unexpected EFRM launch schema")
    require(launch.get("protocol_id") == PROTOCOL_ID, "EFRM launch protocol drifted")
    require(launch.get("method_id") == METHOD_ID, "EFRM launch method drifted")
    require(launch.get("mode") == "public_development_only", "launch is not public-only")
    require(launch.get("protected_test_default") == "locked", "protected test is not locked")
    authorization = launch.get("authorization", {})
    require(
        authorization.get("public_matrix_launch_authorized") is True,
        "EFRM public matrix lacks explicit authorization",
    )
    require(
        authorization.get("protected_evaluation_authorized") is False,
        "EFRM public launch cannot authorize protected evaluation",
    )
    require(int(launch["execution"]["max_concurrent_jobs"]) == 1, "matrix must be serial")
    require(int(launch["execution"]["automatic_retry_count"]) == 0, "retries must be zero")

    for pilot in launch.get("pilot_evidence", ()):
        path = resolve_repo_path(pilot["path"])
        require(sha256_file(path) == str(pilot["sha256"]), "pilot audit hash drifted")
        report = load_mapping(path)
        require(report.get("status") == "pass", "pilot audit did not pass")
        require(
            report.get("mode") == "public_selection_and_refit",
            "launch pilot is only a smoke run",
        )
        require(report.get("protected_test_opened") is False, "pilot opened protected data")

    for section in ("runner", "auditor", "controller"):
        path = resolve_repo_path(launch[section]["path"])
        require(sha256_file(path) == str(launch[section]["sha256"]), f"{section} hash drifted")
    config_path = resolve_repo_path(launch["runner"]["config"])
    require(
        sha256_file(config_path) == str(launch["runner"]["config_sha256"]),
        "runner config hash drifted",
    )

    matrix_path = resolve_repo_path(launch["matrix"]["path"])
    require(sha256_file(matrix_path) == str(launch["matrix"]["sha256"]), "matrix file hash drifted")
    matrix = load_mapping(matrix_path)
    require(matrix.get("schema") == MATRIX_SCHEMA, "unexpected EFRM matrix schema")
    require(matrix.get("method_id") == METHOD_ID, "matrix method drifted")
    require(matrix.get("protocol_id") == PROTOCOL_ID, "matrix protocol drifted")
    require(int(matrix.get("job_count", -1)) == 105, "EFRM matrix must contain 105 jobs")
    require(int(matrix.get("max_concurrent_jobs", -1)) == 1, "retained matrix is not serial")
    require(int(matrix.get("automatic_retry_count", -1)) == 0, "retained matrix admits retries")
    require(
        matrix.get("matrix_identity_sha256") == launch["matrix"]["identity_sha256"],
        "matrix identity drifted",
    )
    require(matrix.get("protected_evaluation_authorized") is False, "matrix unlocks protected data")
    require(matrix.get("target_dataset_exposure") is False, "matrix admits target exposure")
    require(matrix.get("protected_test_opened") is False, "matrix reports protected access")
    require(matrix.get("runner_sha256") == launch["runner"]["sha256"], "runner identity drifted")
    require(matrix.get("config_sha256") == launch["runner"]["config_sha256"], "config identity drifted")
    return launch, resolved_launch, matrix, matrix_path


def validate_jobs(matrix: Mapping[str, Any], *, run_root: Path) -> list[Mapping[str, Any]]:
    jobs = list(matrix.get("jobs", ()))
    require(len(jobs) == 105, "EFRM matrix job list length drifted")
    require(
        [int(job["order"]) for job in jobs] == list(range(105)),
        "EFRM matrix order is not contiguous",
    )
    job_ids = [str(job["job_id"]) for job in jobs]
    require(len(set(job_ids)) == 105, "EFRM matrix has duplicate job IDs")
    outputs: list[Path] = []
    for job in jobs:
        command = [str(value) for value in job["command"]]
        require("protected" not in " ".join(command).lower(), "public command crosses protected boundary")
        output = resolve_repo_path(job["output_dir"])
        try:
            output.relative_to(run_root)
        except ValueError as exc:
            raise PermissionError(f"EFRM job output is outside the matrix root: {output}") from exc
        require(
            command[-2:] == ["--output-dir", portable_path(output)],
            f"job output argument drifted: {job['job_id']}",
        )
        outputs.append(output)
    require(len(set(outputs)) == 105, "EFRM matrix has duplicate output directories")
    return jobs


def retained_failure(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "status.json"
    if not path.is_file():
        return None
    status = load_mapping(path)
    if status.get("status") != "failed":
        return None
    return {
        "output_dir": portable_path(output_dir),
        "error_type": status.get("error_type"),
        "error": status.get("error"),
        "failed_at": status.get("failed_at"),
    }


def completed_manifest(output_dir: Path) -> bool:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return False
    manifest = load_mapping(path)
    return (
        manifest.get("status") == "completed"
        and manifest.get("mode") == "public_selection_and_refit"
        and manifest.get("protected_test_opened") is False
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
        "schema": STATUS_SCHEMA,
        "status": state,
        "launch_path": portable_path(launch_path),
        "matrix_identity_sha256": matrix["matrix_identity_sha256"],
        "expected_job_count": 105,
        "completed_job_count": len(completed),
        "failed_job_count": len(failures),
        "next_job_order": int(next_order),
        "completed_jobs": list(completed),
        "failures": list(failures),
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
        "target_dataset_exposure": False,
        "protected_test_opened": False,
        "updated_at": utc_now(),
    }


def execute(launch_path: str | Path, *, dry_run: bool) -> dict[str, Any]:
    launch, resolved_launch, matrix, _matrix_path = load_launch(launch_path)
    run_root = resolve_repo_path(launch["execution"]["matrix_run_root"])
    jobs = validate_jobs(matrix, run_root=run_root)
    config_path = resolve_repo_path(launch["runner"]["config"])
    if dry_run:
        return {
            "schema": "efrm_lodo_downstream_public_controller_dry_run_v2",
            "status": "pass",
            "job_count": len(jobs),
            "max_concurrent_jobs": 1,
            "automatic_retry_count": 0,
            "public_matrix_launch_authorized": True,
            "protected_evaluation_authorized": False,
            "target_dataset_exposure": False,
            "protected_test_opened": False,
        }

    status_path = run_root / "controller_status.json"
    completed_path = run_root / "completed_public_audit.json"
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
        if not completed_manifest(output_dir):
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
            print(f"[{position + 1}/105] start {job['job_id']}", flush=True)
            result = subprocess.run(
                [str(value) for value in job["command"]], cwd=REPO_ROOT, check=False
            )
            if result.returncode != 0:
                failure = retained_failure(output_dir) or {
                    "output_dir": portable_path(output_dir),
                    "error_type": "RunnerExitCode",
                    "error": f"runner exited with code {result.returncode}",
                }
                failures.append({"job_id": job["job_id"], **failure})
                status = controller_status(
                    launch_path=resolved_launch,
                    matrix=matrix,
                    state="stopped_on_failure",
                    next_order=position,
                    completed=completed,
                    failures=failures,
                )
                write_json(status_path, status)
                return status
        report = audit_run(output_dir, config_path)
        completed.append({"job_id": str(job["job_id"]), **report})
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
        print(f"[{position + 1}/105] pass {job['job_id']}", flush=True)

    status = controller_status(
        launch_path=resolved_launch,
        matrix=matrix,
        state="completed",
        next_order=105,
        completed=completed,
        failures=failures,
    )
    write_json(status_path, status)
    write_json(
        completed_path,
        {
            "schema": COMPLETED_SCHEMA,
            "status": "pass",
            "matrix_identity_sha256": matrix["matrix_identity_sha256"],
            "job_count": 105,
            "run_reports": completed,
            "table_admissible": False,
            "protected_evaluation_authorized": False,
            "target_dataset_exposure": False,
            "protected_test_opened": False,
            "completed_at": utc_now(),
        },
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = execute(args.launch, dry_run=bool(args.dry_run))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] in {"pass", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
