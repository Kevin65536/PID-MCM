#!/usr/bin/env python3
"""Finalize retained BrainFusion A8 evidence after its serial matrix completes."""

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
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import METHOD_ID, SUPPORTED_TASKS, UNSUPPORTED_TASKS, stable_hash
from run_public_development_v2 import portable_path, sha256_file, write_json
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
        raise PermissionError(f"refusing protected BrainFusion finalization input: {resolved}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"BrainFusion finalization input must be an object: {path}")
    if value.get("protected_test_opened", False):
        raise PermissionError(f"BrainFusion artifact reports protected access: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_matrix_completion() -> tuple[dict[str, Any], dict[str, Any]]:
    status = load_json(MATRIX_ROOT / "controller_status.json")
    completed = load_json(MATRIX_ROOT / "completed_public_audit.json")
    require(status.get("status") == "completed", "BrainFusion controller is not complete")
    require(int(status.get("expected_job_count", -1)) == 75, "unexpected job count")
    require(int(status.get("completed_job_count", -1)) == 75, "matrix is incomplete")
    require(int(status.get("failed_job_count", -1)) == 0, "matrix retained failures")
    require(status.get("failures") == [], "matrix failure list is non-empty")
    require(int(status.get("next_job_order", -1)) == 75, "controller did not reach queue end")
    require(status.get("max_concurrent_jobs") == 1, "BrainFusion jobs were not serial")
    require(status.get("automatic_retry_count") == 0, "BrainFusion jobs admitted retries")
    require(status.get("device") == "cuda:1", "BrainFusion matrix device drifted")
    require(
        completed.get("schema") == "brainfusion_public_matrix_completed_audit_v2",
        "bad completed-audit schema",
    )
    require(completed.get("status") == "pass", "completed audit did not pass")
    require(int(completed.get("job_count", -1)) == 75, "completed job count drifted")
    require(
        completed.get("matrix_identity_sha256") == status.get("matrix_identity_sha256"),
        "controller and completed audit matrix identities differ",
    )
    reports = list(completed.get("run_reports", ()))
    require(len(reports) == 75, "completed audit does not retain 75 reports")
    job_ids = [str(report["job_id"]) for report in reports]
    require(len(set(job_ids)) == 75, "completed audit contains duplicate jobs")
    for report in reports:
        require(report.get("status") == "pass", f"run audit failed: {report['job_id']}")
        require(report.get("mode") == "public_development", "matrix contains a smoke run")
        require(report.get("membership_recomputed") is True, "membership was not audited")
        require(report.get("targets_recomputed") is True, "targets were not audited")
        require(report.get("metric_recomputed") is True, "metric was not audited")
        require(report.get("tensor_cache_audited") is True, "tensor cache was not audited")
        require(
            report.get("cached_validation_matches_raw_adapter") is True,
            "cached validation differs from the raw adapter",
        )
        require(
            report.get("checkpoint_predictions_recomputed") is True,
            "checkpoint predictions were not independently recomputed",
        )
        require(report.get("table_admissible") is False, "public run claims table admission")
        report_path = REPO_ROOT / str(report["run_report_path"])
        require(report_path.is_file(), f"run report is missing: {report_path}")
        require(
            sha256_file(report_path) == report["run_report_sha256"],
            f"run report fingerprint drifted: {report['job_id']}",
        )
    return status, completed


def completion_summary(
    status: Mapping[str, Any], completed: Mapping[str, Any]
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    fingerprints: list[dict[str, str]] = []
    for report in completed["run_reports"]:
        grouped[str(report["task"])].append(report)
        fingerprints.append(
            {
                "job_id": str(report["job_id"]),
                "run_report_sha256": str(report["run_report_sha256"]),
            }
        )
    require(set(grouped) == set(SUPPORTED_TASKS), "completed task set drifted")
    tasks: list[dict[str, Any]] = []
    for task in SUPPORTED_TASKS:
        reports = grouped[task]
        require(len(reports) == 15, f"task {task} does not have 15 fold-seed jobs")
        cells = {(int(row["outer_fold"]), int(row["seed"])) for row in reports}
        require(
            cells == {(fold, seed) for fold in range(5) for seed in (17, 42, 73)},
            f"task {task} fold-seed membership drifted",
        )
        tracks = {
            str(load_json(REPO_ROOT / str(row["run_report_path"]))["track"])
            for row in reports
        }
        require(len(tracks) == 1, f"task {task} reporting track drifted")
        values = np.asarray(
            [float(row["validation_macro_f1"]) for row in reports], dtype=np.float64
        )
        fold_means = [
            float(
                np.mean(
                    [
                        float(row["validation_macro_f1"])
                        for row in reports
                        if int(row["outer_fold"]) == fold
                    ]
                )
            )
            for fold in range(5)
        ]
        tasks.append(
            {
                "task": task,
                "track": tracks.pop(),
                "job_count": 15,
                "public_validation_macro_f1_cell_mean": float(values.mean()),
                "public_validation_macro_f1_cell_sd": float(values.std(ddof=1)),
                "seed_mean_by_outer_fold": fold_means,
                "claim_boundary": "public_development_only_not_table_admissible",
            }
        )
    return {
        "schema": "brainfusion_public_matrix_completion_summary_v2",
        "status": "pass",
        "method_id": METHOD_ID,
        "matrix_identity_sha256": status["matrix_identity_sha256"],
        "job_count": 75,
        "completed_job_count": 75,
        "failed_job_count": 0,
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
        "device": "cuda:1",
        "tasks": tasks,
        "job_artifact_fingerprint_sha256": stable_hash(fingerprints),
        "local_controller_status_sha256": sha256_file(
            MATRIX_ROOT / "controller_status.json"
        ),
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
    paths: list[Path] = []
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
        paths.append(path)

    for task in UNSUPPORTED_TASKS:
        path = ALIGNMENT_EVIDENCE_ROOT / f"{task}.json"
        cell = load_json(path)
        require(cell.get("cell_status") == "unsupported", f"{task} disposition drifted")
        cell["gate_status"]["A8"] = "not_applicable"
        cell["protocol_freeze"] = {
            "public_matrix_completion_path": portable_path(COMPLETION_PATH),
            "public_matrix_completion_sha256": sha256_file(COMPLETION_PATH),
            "matrix_identity_sha256": summary["matrix_identity_sha256"],
            "protected_evaluation_authorized": False,
            "protected_test_opened": False,
        }
        write_json(path, cell)
        paths.append(path)
    return paths


def update_alignment_summary(
    cell_paths: list[Path], completion: Mapping[str, Any]
) -> None:
    previous = load_json(ALIGNMENT_EVIDENCE_ROOT / "summary.json")
    schema_report = audit_alignment(ALIGNMENT_CONTRACT, cell_paths)
    for report in schema_report["cell_reports"]:
        report["source"] = portable_path(Path(str(report["source"])))
    tasks = []
    for row in previous["tasks"]:
        task = str(row["task"])
        if task in SUPPORTED_TASKS:
            tasks.append(
                {
                    **row,
                    "status": "A0-A8_pass_public_matrix_complete_protected_locked",
                    "cell_status": "pass",
                }
            )
        else:
            tasks.append(row)
    output = {
        **previous,
        "status": "public_development_complete_A0_A8_pass_protected_locked",
        "created_at": utc_now(),
        "tasks": tasks,
        "schema_audit": schema_report,
        "public_matrix_completion_path": portable_path(COMPLETION_PATH),
        "public_matrix_completion_sha256": sha256_file(COMPLETION_PATH),
        "matrix_identity_sha256": completion["matrix_identity_sha256"],
        "completed_public_job_count": 75,
        "protected_evaluation_authorized": False,
        "protected_test_opened": False,
    }
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
