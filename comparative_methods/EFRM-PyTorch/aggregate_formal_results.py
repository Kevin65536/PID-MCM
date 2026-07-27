#!/usr/bin/env python3
"""Validate and aggregate the complete 70-cell frozen EFRM result matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efrm_pytorch.tasks import EFRMUnifiedTaskDataset, get_task_spec
from train_downstream import sha256_file, write_json


T95_DF4 = 2.7764451051977987


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scalar_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    excluded = {
        "outer_fold",
        "sample_count",
        "elapsed_seconds",
        "evaluation_seconds",
        "cuda_peak_allocated_gib",
        "cuda_peak_reserved_gib",
    }
    common = set.intersection(
        *[
            {
                key
                for key, value in row.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and key not in excluded
                and math.isfinite(float(value))
            }
            for row in rows
        ]
    )
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(common):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        mean = float(values.mean())
        sample_sd = float(values.std(ddof=1))
        half_width = float(T95_DF4 * sample_sd / math.sqrt(5))
        result[key] = {
            "fold_values": values.tolist(),
            "mean": mean,
            "sample_sd_ddof_1": sample_sd,
            "t95_df4": [mean - half_width, mean + half_width],
        }
    return result


def aggregate(root: Path) -> dict[str, Any]:
    protocol_root = root / "protocol"
    matrix_path = protocol_root / "job_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    cohort_path = Path(matrix["cohort_manifest"])
    if sha256_file(cohort_path) != matrix["cohort_manifest_sha256"]:
        raise RuntimeError("cohort manifest hash drifted before aggregation")
    if len(matrix["jobs"]) != 70:
        raise RuntimeError("formal aggregation requires all 70 declared jobs")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    completion: list[dict[str, Any]] = []
    oof_root = root / "aggregate/oof_predictions"
    for job in matrix["jobs"]:
        job_dir = root / "jobs" / str(job["job_id"])
        manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "protected_evaluation_completed"
            or manifest.get("protected_test_opened") is not True
        ):
            raise RuntimeError(f"incomplete formal job: {job['job_id']}")
        metrics_path = job_dir / "test_metrics.json"
        predictions_path = job_dir / "test_predictions.npz"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"missing protected evidence for {job['job_id']}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if int(metrics["outer_fold"]) != int(job["outer_fold"]):
            raise RuntimeError(f"fold identity mismatch for {job['job_id']}")
        grouped.setdefault((str(job["task"]), str(job["protocol"])), []).append(
            {"job": job, "metrics": metrics, "predictions": predictions_path}
        )
        completion.append(
            {
                "job_id": job["job_id"],
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_sha256": sha256_file(predictions_path),
                "status": "protected_evaluation_completed",
            }
        )

    aggregates: dict[str, Any] = {}
    for (task, protocol), cells in sorted(grouped.items()):
        cells.sort(key=lambda row: int(row["job"]["outer_fold"]))
        if [int(row["job"]["outer_fold"]) for row in cells] != list(range(5)):
            raise RuntimeError(f"{task}/{protocol} does not contain folds 0..4 exactly once")
        spec = get_task_spec(task)
        dataset = EFRMUnifiedTaskDataset(
            spec, cache_root=str(cohort["cache_root"])
        )
        target_subjects = set(cohort["tasks"][task]["eligible_target_subjects"])
        expected_indices = {
            index
            for index in range(len(dataset))
            if str(dataset.lightweight_metadata(index)["subject"]) in target_subjects
        }
        protected_test_indices: list[int] = []
        prediction_blocks: dict[str, list[np.ndarray]] = {}
        for cell in cells:
            protected = json.loads(
                Path(cell["job"]["protected_manifest"]).read_text(encoding="utf-8")
            )
            protected_test_indices.extend(int(value) for value in protected["test_indices"])
            with np.load(cell["predictions"], allow_pickle=False) as evidence:
                for key in evidence.files:
                    prediction_blocks.setdefault(key, []).append(evidence[key])
        if (
            len(protected_test_indices) != len(expected_indices)
            or set(protected_test_indices) != expected_indices
        ):
            raise RuntimeError(f"{task}/{protocol} protected tests are not an exact partition")
        oof_path = oof_root / task / f"{protocol}.npz"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            oof_path,
            **{
                key: np.concatenate(blocks, axis=0)
                for key, blocks in prediction_blocks.items()
            },
        )
        rows = [cell["metrics"] for cell in cells]
        metrics = scalar_metrics(rows)
        primary_key = "macro_f1" if spec.task_type == "classification" else "native_ccc"
        companion_key = "accuracy" if spec.task_type == "classification" else "native_rmse"
        if primary_key not in metrics or companion_key not in metrics:
            raise RuntimeError(f"{task}/{protocol} omits required primary metrics")
        aggregates.setdefault(task, {})[protocol] = {
            "reporting_name": cells[0]["job"]["reporting_name"],
            "task_type": spec.task_type,
            "fold_count": 5,
            "uncertainty_label": "sample SD across five target outer folds (ddof=1)",
            "primary_metric": primary_key,
            "companion_metric": companion_key,
            "metrics": metrics,
            "test_partition_exact": True,
            "target_sample_count": len(expected_indices),
            "oof_predictions": str(oof_path.resolve()),
            "oof_predictions_sha256": sha256_file(oof_path),
        }

    if len(grouped) != 14:
        raise RuntimeError(f"expected 14 task/protocol aggregates, got {len(grouped)}")
    summary = {
        "schema": "efrm_resource_bounded_dual_protocol_summary_v1",
        "protocol_id": matrix["protocol_id"],
        "completed_at": utc_now(),
        "job_count": 70,
        "all_jobs_complete": True,
        "all_test_partitions_exact": True,
        "protected_test_opened": True,
        "pretraining_seed_count": 1,
        "fold_uncertainty": "sample SD across five target outer folds (ddof=1)",
        "comparison_boundary": (
            "source-to-target EFRM transfer only; do not directly rank against the "
            "existing full-dataset STA-Net aggregate"
        ),
        "cohort_manifest_sha256": matrix["cohort_manifest_sha256"],
        "job_matrix_sha256": sha256_file(matrix_path),
        "completion_evidence": completion,
        "results": aggregates,
    }
    aggregate_root = root / "aggregate"
    write_json(aggregate_root / "summary.json", summary)
    lines = [
        "# EFRM frozen resource-bounded five-fold results",
        "",
        "Values are fold mean ± sample SD across five target outer folds (ddof=1).",
        "",
        "| Task | Protocol | Primary | Companion |",
        "| --- | --- | ---: | ---: |",
    ]
    for task, protocols in aggregates.items():
        for protocol, row in protocols.items():
            primary = row["metrics"][row["primary_metric"]]
            companion = row["metrics"][row["companion_metric"]]
            lines.append(
                f"| {task} | {protocol} | "
                f"{row['primary_metric']} {primary['mean']:.6f} ± "
                f"{primary['sample_sd_ddof_1']:.6f} | "
                f"{row['companion_metric']} {companion['mean']:.6f} ± "
                f"{companion['sample_sd_ddof_1']:.6f} |"
            )
    lines.extend(
        [
            "",
            "These are source-to-target transfer estimates conditioned on one frozen "
            "source-only EFRM checkpoint. They are not directly ranked against the "
            "current full-dataset STA-Net aggregate.",
            "",
        ]
    )
    (aggregate_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    status.update(
        {
            "status": "completed",
            "completed_at": summary["completed_at"],
            "completed_public_jobs": 70,
            "completed_protected_jobs": 70,
            "protected_test_opened": True,
            "summary": str((aggregate_root / "summary.json").resolve()),
            "summary_sha256": sha256_file(aggregate_root / "summary.json"),
        }
    )
    write_json(root / "status.json", status)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(
            METHOD_ROOT / "runs/formal/efrm_resource_bounded_dual_protocol_v1"
        ),
    )
    args = parser.parse_args()
    summary = aggregate(Path(args.root).resolve())
    print(json.dumps({"status": "completed", "job_count": summary["job_count"]}, indent=2))


if __name__ == "__main__":
    main()
