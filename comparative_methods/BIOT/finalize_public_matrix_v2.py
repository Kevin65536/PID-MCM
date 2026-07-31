#!/usr/bin/env python3
"""Finalize retained BIOT A8 evidence after the serial public matrix completes."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.BIOT.alignment_data import SUPPORTED_TASKS, stable_hash
from comparative_methods.BIOT.run_public_development_v2 import (
    portable_path,
    resolve_repo_path,
    sha256_file,
    write_json,
)
from comparative_methods.audit_adapter_alignment import audit as audit_alignment


MATRIX_ROOT = METHOD_ROOT / "runs/public_development_v2/matrix_v2"
ALIGNMENT_EVIDENCE_ROOT = METHOD_ROOT / "evidence/alignment_v2"
COMPLETION_PATH = METHOD_ROOT / "evidence/public_development_v2/matrix_completion_summary.json"
ALIGNMENT_CONTRACT = REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected finalization input: {resolved}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"finalization input must be an object: {path}")
    if value.get("protected_test_opened", False):
        raise PermissionError(f"artifact reports protected access: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_matrix_completion() -> tuple[dict[str, Any], dict[str, Any]]:
    status_path = MATRIX_ROOT / "controller_status.json"
    audit_path = MATRIX_ROOT / "completed_public_audit.json"
    status = load_json(status_path)
    completed = load_json(audit_path)
    require(status.get("status") == "completed", "BIOT matrix controller is not complete")
    require(int(status.get("expected_job_count", -1)) == 90, "unexpected job count")
    require(int(status.get("completed_job_count", -1)) == 90, "BIOT matrix is incomplete")
    require(int(status.get("failed_job_count", -1)) == 0, "BIOT matrix retained failures")
    require(status.get("failures") == [], "BIOT matrix failure list is non-empty")
    require(status.get("next_job_order") == 90, "BIOT controller did not reach queue end")
    require(status.get("max_concurrent_jobs") == 1, "BIOT jobs were not serial")
    require(status.get("automatic_retry_count") == 0, "BIOT jobs admitted retries")
    require(completed.get("schema") == "biot_public_matrix_completed_audit_v2", "bad audit schema")
    require(completed.get("status") == "pass", "BIOT completed audit did not pass")
    require(int(completed.get("job_count", -1)) == 90, "completed audit job count drifted")
    require(
        completed.get("matrix_identity_sha256") == status.get("matrix_identity_sha256"),
        "controller and completed audit matrix identities differ",
    )
    reports = list(completed.get("run_reports", ()))
    require(len(reports) == 90, "completed audit does not retain 90 run reports")
    job_ids = [str(report["job_id"]) for report in reports]
    require(len(set(job_ids)) == 90, "completed audit contains duplicate jobs")
    for report in reports:
        require(report.get("status") == "pass", f"run audit failed: {report['job_id']}")
        require(report.get("table_admissible") is False, "public run claims table admission")
        require(report.get("protected_test_opened") is False, "run reports protected access")
    return status, completed


def completion_summary(status: Mapping[str, Any], completed: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    fingerprints: list[dict[str, str]] = []
    for report in completed["run_reports"]:
        grouped[str(report["task"])].append(report)
        fingerprints.append(
            {
                "job_id": str(report["job_id"]),
                "manifest_sha256": str(report["manifest_sha256"]),
                "checkpoint_sha256": str(report["public_refit_checkpoint_sha256"]),
                "feature_cache_sha256": str(report["feature_cache_sha256"]),
            }
        )
    require(set(grouped) == set(SUPPORTED_TASKS), "completed matrix task set drifted")
    tasks: list[dict[str, Any]] = []
    for task in SUPPORTED_TASKS:
        reports = grouped[task]
        require(len(reports) == 15, f"task {task} does not have 15 fold-seed jobs")
        cells = {(int(row["outer_fold"]), int(row["seed"])) for row in reports}
        require(
            cells == {(fold, seed) for fold in range(5) for seed in (17, 42, 73)},
            f"task {task} fold-seed membership drifted",
        )
        values = np.asarray(
            [float(row["validation_macro_f1"]) for row in reports], dtype=np.float64
        )
        fold_means = []
        for fold in range(5):
            fold_values = [
                float(row["validation_macro_f1"])
                for row in reports
                if int(row["outer_fold"]) == fold
            ]
            fold_means.append(float(np.mean(fold_values)))
        tasks.append(
            {
                "task": task,
                "job_count": len(reports),
                "public_validation_macro_f1_cell_mean": float(values.mean()),
                "public_validation_macro_f1_cell_sd": float(values.std(ddof=1)),
                "seed_mean_by_outer_fold": fold_means,
                "claim_boundary": "public_development_only_not_table_admissible",
            }
        )
    return {
        "schema": "biot_public_matrix_completion_summary_v2",
        "status": "pass",
        "method_id": "biot",
        "matrix_identity_sha256": status["matrix_identity_sha256"],
        "job_count": 90,
        "completed_job_count": 90,
        "failed_job_count": 0,
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
        "tasks": tasks,
        "job_artifact_fingerprint_sha256": stable_hash(fingerprints),
        "local_controller_status_sha256": sha256_file(MATRIX_ROOT / "controller_status.json"),
        "local_completed_audit_sha256": sha256_file(
            MATRIX_ROOT / "completed_public_audit.json"
        ),
        "finalizer_path": portable_path(Path(__file__)),
        "finalizer_sha256": sha256_file(Path(__file__)),
        "table_admissible": False,
        "protected_evaluation_authorized": False,
        "protected_test_opened": False,
        "completed_at": utc_now(),
    }


def promote_alignment_cells(summary: Mapping[str, Any]) -> list[Path]:
    cell_paths: list[Path] = []
    for task in SUPPORTED_TASKS:
        path = ALIGNMENT_EVIDENCE_ROOT / f"{task}.json"
        cell = load_json(path)
        require(cell.get("evidence_scope") == "public_complete", "cell lacks public evidence")
        require(
            all(cell["gate_status"][f"A{index}"] == "pass" for index in range(8)),
            f"cell {task} lacks A0-A7 pass",
        )
        cell["gate_status"]["A8"] = "pass"
        cell["cell_status"] = "pass"
        cell["protocol_freeze"] = {
            "public_matrix_completion_path": portable_path(COMPLETION_PATH),
            "public_matrix_completion_sha256": sha256_file(COMPLETION_PATH),
            "matrix_identity_sha256": summary["matrix_identity_sha256"],
            "protected_evaluation_authorized": False,
            "protected_test_opened": False,
        }
        write_json(path, cell)
        cell_paths.append(path)

    refed_path = ALIGNMENT_EVIDENCE_ROOT / "refed_regression.json"
    refed = load_json(refed_path)
    require(refed.get("cell_status") == "unsupported", "REFED disposition drifted")
    refed["gate_status"]["A8"] = "not_applicable"
    refed["protocol_freeze"] = {
        "public_matrix_completion_path": portable_path(COMPLETION_PATH),
        "public_matrix_completion_sha256": sha256_file(COMPLETION_PATH),
        "protected_evaluation_authorized": False,
        "protected_test_opened": False,
    }
    write_json(refed_path, refed)
    cell_paths.append(refed_path)
    return cell_paths


def update_alignment_summary(cell_paths: list[Path], completion: Mapping[str, Any]) -> None:
    previous = load_json(ALIGNMENT_EVIDENCE_ROOT / "summary.json")
    schema_report = audit_alignment(ALIGNMENT_CONTRACT, cell_paths)
    for report in schema_report["cell_reports"]:
        report["source"] = portable_path(Path(str(report["source"])))
    tasks = []
    for task in SUPPORTED_TASKS:
        cell = load_json(ALIGNMENT_EVIDENCE_ROOT / f"{task}.json")
        previous_task = next(row for row in previous["tasks"] if row["task"] == task)
        tasks.append(
            {
                **previous_task,
                "status": "A0-A8_pass_public_matrix_complete_protected_locked",
                "cell_status": cell["cell_status"],
            }
        )
    refed = next(row for row in previous["tasks"] if row["task"] == "refed_regression")
    tasks.append(refed)
    output = {
        **previous,
        "status": "public_development_complete_A0_A8_pass_protected_locked",
        "created_at": utc_now(),
        "tasks": tasks,
        "schema_audit": schema_report,
        "public_matrix_completion_path": portable_path(COMPLETION_PATH),
        "public_matrix_completion_sha256": sha256_file(COMPLETION_PATH),
        "matrix_identity_sha256": completion["matrix_identity_sha256"],
        "protected_evaluation_authorized": False,
        "protected_test_opened": False,
    }
    # Keep the A0-A7 summary immutable because the reviewed public runner
    # fingerprints it. A8 completion receives a new retained artifact.
    write_json(ALIGNMENT_EVIDENCE_ROOT / "summary_final.json", output)


def main() -> int:
    status, completed = validate_matrix_completion()
    summary = completion_summary(status, completed)
    write_json(COMPLETION_PATH, summary)
    cell_paths = promote_alignment_cells(summary)
    update_alignment_summary(cell_paths, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
