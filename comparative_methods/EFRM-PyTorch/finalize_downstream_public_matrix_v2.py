#!/usr/bin/env python3
"""Finalize the audited EFRM LODO-v2 public matrix without table admission."""

from __future__ import annotations

from collections import defaultdict
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

from run_downstream_public_matrix_v2 import COMPLETED_SCHEMA, STATUS_SCHEMA  # noqa: E402
from run_downstream_public_v2 import (  # noqa: E402
    DEFAULT_CONFIG,
    METHOD_ID,
    PROTOCOL_ID,
    load_config,
    portable_path,
    resolve_repo_path,
    sha256_file,
    stable_hash,
    utc_now,
    write_json,
)


SUMMARY_SCHEMA = "efrm_lodo_downstream_public_matrix_completion_summary_v2"
TASKS = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
)
SEEDS = (17, 42, 73)


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected EFRM finalization input: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"EFRM finalization input must be an object: {resolved}")
    if value.get("protected_test_opened", False):
        raise PermissionError(f"EFRM finalization input reports protected access: {resolved}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finalize(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, resolved_config = load_config(config_path)
    run_root = resolve_repo_path(config["resources"]["run_root"])
    status_path = run_root / "controller_status.json"
    completed_path = run_root / "completed_public_audit.json"
    status = load_json(status_path)
    completed = load_json(completed_path)
    require(status.get("schema") == STATUS_SCHEMA, "unexpected EFRM controller status schema")
    require(status.get("status") == "completed", "EFRM public controller is not complete")
    require(int(status.get("completed_job_count", -1)) == 105, "EFRM matrix is incomplete")
    require(int(status.get("failed_job_count", -1)) == 0, "EFRM matrix retained failures")
    require(status.get("failures") == [], "EFRM matrix failure list is non-empty")
    require(int(status.get("next_job_order", -1)) == 105, "EFRM controller did not reach queue end")
    require(status.get("max_concurrent_jobs") == 1, "EFRM matrix was not serial")
    require(status.get("automatic_retry_count") == 0, "EFRM matrix admitted retries")
    require(status.get("target_dataset_exposure") is False, "EFRM matrix reports target exposure")
    require(completed.get("schema") == COMPLETED_SCHEMA, "unexpected completed-audit schema")
    require(completed.get("status") == "pass", "EFRM completed audit did not pass")
    require(int(completed.get("job_count", -1)) == 105, "completed audit job count drifted")
    require(
        completed.get("matrix_identity_sha256") == status.get("matrix_identity_sha256"),
        "controller/completed matrix identities differ",
    )

    reports = list(completed.get("run_reports", ()))
    require(len(reports) == 105, "EFRM completed audit lacks 105 run reports")
    require(
        len({str(report["job_id"]) for report in reports}) == 105,
        "EFRM completed audit contains duplicate jobs",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    artifact_rows: list[dict[str, str]] = []
    for report in reports:
        require(report.get("status") == "pass", f"run audit failed: {report['job_id']}")
        require(
            report.get("mode") == "public_selection_and_refit",
            "EFRM matrix contains a smoke run",
        )
        require(report.get("table_admissible") is False, "public run claims table admission")
        require(report.get("target_dataset_exposure") is False, "public run reports target exposure")
        require(report.get("protected_test_opened") is False, "public run reports protected access")
        task = str(report["task"])
        grouped[task].append(report)
        run_dir = resolve_repo_path(report["run_dir"])
        artifact_rows.append(
            {
                "job_id": str(report["job_id"]),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "checkpoint_sha256": str(report["checkpoint_sha256"]),
                "feature_cache_sha256": str(report["feature_cache_sha256"]),
            }
        )
    require(set(grouped) == set(TASKS), "EFRM completed task set drifted")

    task_rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_reports = grouped[task]
        require(len(task_reports) == 15, f"EFRM task {task} does not have 15 jobs")
        expected_cells = {(fold, seed) for fold in range(5) for seed in SEEDS}
        actual_cells = {
            (int(report["outer_fold"]), int(report["seed"]))
            for report in task_reports
        }
        require(actual_cells == expected_cells, f"EFRM task {task} cell membership drifted")
        metric_names = {str(report["primary_metric"]) for report in task_reports}
        expected_metric = "ccc" if task == "refed_regression" else "macro_f1"
        require(metric_names == {expected_metric}, f"EFRM task {task} metric drifted")
        values = np.asarray(
            [float(report["public_validation_primary"]) for report in task_reports],
            dtype=np.float64,
        )
        fold_means = [
            float(
                np.mean(
                    [
                        float(report["public_validation_primary"])
                        for report in task_reports
                        if int(report["outer_fold"]) == fold
                    ]
                )
            )
            for fold in range(5)
        ]
        task_rows.append(
            {
                "task": task,
                "primary_metric": expected_metric,
                "job_count": 15,
                "public_validation_primary_cell_mean": float(values.mean()),
                "public_validation_primary_cell_sd": float(values.std(ddof=1)),
                "seed_mean_by_outer_fold": fold_means,
                "public_validation_outer_fold_mean": float(np.mean(fold_means)),
                "public_validation_outer_fold_sample_sd": float(
                    np.std(fold_means, ddof=1)
                ),
                "claim_boundary": "public_development_only_not_table_admissible",
            }
        )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "pass",
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "matrix_identity_sha256": status["matrix_identity_sha256"],
        "job_count": 105,
        "completed_job_count": 105,
        "failed_job_count": 0,
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
        "tasks": task_rows,
        "job_artifact_fingerprint_sha256": stable_hash(artifact_rows),
        "controller_status_path": portable_path(status_path),
        "controller_status_sha256": sha256_file(status_path),
        "completed_audit_path": portable_path(completed_path),
        "completed_audit_sha256": sha256_file(completed_path),
        "runner_config_path": portable_path(resolved_config),
        "runner_config_sha256": sha256_file(resolved_config),
        "table_admissible": False,
        "protected_evaluation_authorized": False,
        "target_dataset_exposure": False,
        "protected_test_opened": False,
        "completed_at": utc_now(),
    }
    output = METHOD_ROOT / "evidence/public_development_v2/matrix_completion_summary.json"
    write_json(output, summary)
    return summary


def main() -> int:
    summary = finalize()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
