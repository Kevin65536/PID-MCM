#!/usr/bin/env python3
"""Finalize retained NormWear A8 evidence after its serial matrix completes."""

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

from alignment_data import METHOD_ID, SUPPORTED_TASKS, UNSUPPORTED_TASKS, stable_hash  # noqa: E402
from run_public_development_v2 import portable_path, sha256_file, write_json  # noqa: E402
from comparative_methods.audit_adapter_alignment import audit as audit_alignment  # noqa: E402


MATRIX_ROOT = METHOD_ROOT / "runs/public_development_v2/matrix_v2"
ALIGNMENT_EVIDENCE_ROOT = METHOD_ROOT / "evidence/alignment_v2"
COMPLETION_PATH = METHOD_ROOT / "evidence/public_development_v2/matrix_completion_summary.json"
ALIGNMENT_CONTRACT = REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected NormWear finalization input: {resolved}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"NormWear finalization input must be an object: {path}")
    if value.get("protected_test_opened", False):
        raise PermissionError(f"NormWear artifact reports protected access: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_matrix_completion() -> tuple[dict[str, Any], dict[str, Any]]:
    status = load_json(MATRIX_ROOT / "controller_status.json")
    completed = load_json(MATRIX_ROOT / "completed_public_audit.json")
    require(status.get("status") == "completed", "NormWear controller is not complete")
    require(int(status.get("expected_job_count", -1)) == 90, "unexpected job count")
    require(int(status.get("completed_job_count", -1)) == 90, "matrix is incomplete")
    require(int(status.get("failed_job_count", -1)) == 0, "matrix retained failures")
    require(status.get("failures") == [], "matrix failure list is non-empty")
    require(int(status.get("next_job_order", -1)) == 90, "controller did not reach queue end")
    require(status.get("max_concurrent_jobs") == 1, "NormWear jobs were not serial")
    require(status.get("automatic_retry_count") == 0, "NormWear jobs admitted retries")
    require(
        completed.get("schema") == "normwear_public_matrix_completed_audit_v2",
        "bad completed-audit schema",
    )
    require(completed.get("status") == "pass", "completed audit did not pass")
    require(int(completed.get("job_count", -1)) == 90, "completed job count drifted")
    require(
        completed.get("matrix_identity_sha256") == status.get("matrix_identity_sha256"),
        "controller and completed audit matrix identities differ",
    )
    reports = list(completed.get("run_reports", ()))
    require(len(reports) == 90, "completed audit does not retain 90 reports")
    job_ids = [str(report["job_id"]) for report in reports]
    require(len(set(job_ids)) == 90, "completed audit contains duplicate jobs")
    for report in reports:
        require(report.get("status") == "pass", f"run audit failed: {report['job_id']}")
        require(
            report.get("mode") == "public_selection_and_refit",
            "matrix contains a smoke run",
        )
        require(report.get("table_admissible") is False, "public run claims table admission")
        require(report.get("protected_test_opened") is False, "run reports protected access")
        run_dir = REPO_ROOT / str(report["run_dir"])
        require(run_dir.is_dir(), f"run directory is missing: {run_dir}")
        require(
            sha256_file(run_dir / "manifest.json") == report["manifest_sha256"],
            f"run manifest fingerprint drifted: {report['job_id']}",
        )
        require(
            sha256_file(run_dir / "checkpoint_public_refit.pt")
            == report["public_refit_checkpoint_sha256"],
            f"run checkpoint fingerprint drifted: {report['job_id']}",
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
                "manifest_sha256": str(report["manifest_sha256"]),
                "checkpoint_sha256": str(report["public_refit_checkpoint_sha256"]),
                "selected_public_feature_sha256": str(
                    report["selected_public_feature_sha256"]
                ),
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
        fold_feature_hashes: dict[str, str] = {}
        for fold in range(5):
            hashes = {
                str(row["selected_public_feature_sha256"])
                for row in reports
                if int(row["outer_fold"]) == fold
            }
            require(len(hashes) == 1, f"task {task} fold {fold} feature identity drifted")
            fold_feature_hashes[str(fold)] = hashes.pop()
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
                "job_count": 15,
                "public_validation_macro_f1_cell_mean": float(values.mean()),
                "public_validation_macro_f1_cell_sd": float(values.std(ddof=1)),
                "public_validation_macro_f1_min": float(values.min()),
                "public_validation_macro_f1_max": float(values.max()),
                "seed_mean_by_outer_fold": fold_means,
                "selected_public_feature_sha256_by_outer_fold": fold_feature_hashes,
                "claim_boundary": "public_development_only_not_table_admissible",
            }
        )
    return {
        "schema": "normwear_public_matrix_completion_summary_v2",
        "status": "pass",
        "method_id": METHOD_ID,
        "matrix_identity_sha256": status["matrix_identity_sha256"],
        "job_count": 90,
        "completed_job_count": 90,
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
    tasks: list[dict[str, Any]] = []
    for task in SUPPORTED_TASKS:
        cell_path = ALIGNMENT_EVIDENCE_ROOT / f"{task}.json"
        cell = load_json(cell_path)
        tasks.append(
            {
                "task": task,
                "path": portable_path(cell_path),
                "sample_count": int(cell["public_adapter_audit"]["unique_sample_count"]),
                "feature_sha256": str(cell["public_adapter_audit"]["feature_sha256"]),
                "status": "A0-A8_pass_public_matrix_complete_protected_locked",
                "cell_status": "pass",
            }
        )
    for task in UNSUPPORTED_TASKS:
        cell_path = ALIGNMENT_EVIDENCE_ROOT / f"{task}.json"
        tasks.append(
            {
                "task": task,
                "path": portable_path(cell_path),
                "status": "unsupported_preregistered",
                "cell_status": "unsupported",
            }
        )
    output = {
        **previous,
        "status": "public_development_complete_A0_A8_pass_protected_locked",
        "created_at": utc_now(),
        "tasks": tasks,
        "schema_audit": schema_report,
        "public_matrix_completion_path": portable_path(COMPLETION_PATH),
        "public_matrix_completion_sha256": sha256_file(COMPLETION_PATH),
        "matrix_identity_sha256": completion["matrix_identity_sha256"],
        "completed_public_job_count": 90,
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
