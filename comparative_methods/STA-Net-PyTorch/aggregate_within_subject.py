#!/usr/bin/env python3
"""Aggregate protected STA-Net within-subject folds at the subject level."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch import get_sta_net_task_spec
from visualize_results import classification_metrics, regression_metrics


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def bootstrap_mean_ci(values: Sequence[float], seed: int = 42, draws: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[metric] = {
            "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "subject_bootstrap_95_ci": bootstrap_mean_ci(values),
            "subject_count": len(values),
        }
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    protocol = json.loads((root / "protocol_freeze_manifest.json").read_text(encoding="utf-8"))
    protocol_name = str(protocol.get("protocol", "single_subject_nested_cv"))
    report_title = str(protocol.get("report_title", "STA-Net non-cross-subject protected evaluation"))
    report_description = str(protocol.get(
        "report_description",
        "Training and test samples may come from the same subject, while all session, record, "
        "video, or semantic-trial dependency groups remain disjoint across partitions.",
    ))
    jobs = protocol["jobs"]
    missing = []
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    fold_rows: list[dict[str, Any]] = []
    for job in jobs:
        fold_dir = root / "folds" / job["task"] / job["fold_id"]
        summary_path = fold_dir / "evaluation" / "summary.json"
        prediction_path = fold_dir / "evaluation" / "protected_predictions.npz"
        if not summary_path.exists() or not prediction_path.exists():
            missing.append(f"{job['task']}/{job['fold_id']}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        protected = json.loads(Path(job["protected_manifest"]).read_text(encoding="utf-8"))
        if summary.get("fold_id") != job["fold_id"] or summary.get("task") != job["task"]:
            raise RuntimeError(f"evaluation identity mismatch for {job['task']}/{job['fold_id']}")
        grouped[job["task"]][str(protected["subject"])].append(prediction_path)
        metrics = summary["metrics"]
        row = {
            "task": job["task"],
            "subject": protected["subject"],
            "fold_id": job["fold_id"],
            "sample_count": summary["sample_count"],
        }
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa",
                    "mae_native", "rmse_native", "r2_native", "pearson_r",
                    "concordance_correlation"):
            if key in metrics:
                row[key] = finite_float(metrics[key])
        fold_rows.append(row)
    if missing:
        raise RuntimeError(f"{len(missing)} within-subject folds are incomplete; first={missing[:5]}")

    output = root / "aggregate"
    all_subject_rows: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    for task, subject_files in sorted(grouped.items()):
        spec = get_sta_net_task_spec(task)
        subject_rows: list[dict[str, Any]] = []
        pooled_prediction, pooled_target, pooled_mask = [], [], []
        for subject, paths in sorted(subject_files.items()):
            predictions, targets, masks, sample_ids = [], [], [], []
            for path in paths:
                data = np.load(path)
                predictions.append(data["prediction"])
                targets.append(data["target"])
                masks.append(data["target_valid_mask"])
                sample_ids.extend(data["sample_id"].astype(str).tolist())
            if len(sample_ids) != len(set(sample_ids)):
                raise RuntimeError(f"duplicate protected samples for {task}/{subject}")
            prediction = np.concatenate(predictions)
            target = np.concatenate(targets)
            mask = np.concatenate(masks)
            pooled_prediction.append(prediction)
            pooled_target.append(target)
            pooled_mask.append(mask)
            if spec.task_type == "classification":
                metrics = classification_metrics(target, prediction, spec.class_names)
                row = {
                    "task": task,
                    "subject": subject,
                    "fold_count": len(paths),
                    "sample_count": len(target),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "cohen_kappa": finite_float(metrics["cohen_kappa"]),
                }
            else:
                metrics = regression_metrics(target, prediction, mask, spec.target_names)
                keep = mask.astype(bool)
                rho = spearmanr(target[keep], prediction[keep]).statistic
                row = {
                    "task": task,
                    "subject": subject,
                    "fold_count": len(paths),
                    "sample_count": len(target),
                    "mae_native": metrics["mae_native"],
                    "rmse_native": metrics["rmse_native"],
                    "r2_native": finite_float(metrics["r2_native"]),
                    "pearson_r": finite_float(metrics["pearson_r"]),
                    "spearman_rho": finite_float(rho),
                    "concordance_correlation": finite_float(metrics["concordance_correlation"]),
                }
            subject_rows.append(row)
            all_subject_rows.append(row)
        prediction = np.concatenate(pooled_prediction)
        target = np.concatenate(pooled_target)
        mask = np.concatenate(pooled_mask)
        if spec.task_type == "classification":
            metric_names = ("macro_f1", "accuracy", "balanced_accuracy", "cohen_kappa")
            pooled = classification_metrics(target, prediction, spec.class_names)
            primary = "macro_f1"
        else:
            metric_names = ("concordance_correlation", "mae_native", "rmse_native",
                            "r2_native", "pearson_r", "spearman_rho")
            pooled = regression_metrics(target, prediction, mask, spec.target_names)
            pooled["spearman_rho"] = finite_float(
                spearmanr(target[mask.astype(bool)], prediction[mask.astype(bool)]).statistic
            )
            primary = "concordance_correlation"
        task_summary = {
            "task": task,
            "task_type": spec.task_type,
            "protocol": protocol_name,
            "primary_endpoint": f"mean_per_subject_{primary}",
            "fold_count": sum(len(paths) for paths in subject_files.values()),
            "subject_count": len(subject_rows),
            "sample_count": int(sum(int(row["sample_count"]) for row in subject_rows)),
            "subject_level": summarize(subject_rows, metric_names),
            "pooled_sample_diagnostic": pooled,
            "protected_test_opened": True,
        }
        task_summaries[task] = task_summary
        write_json(output / task / "summary.json", task_summary)
        write_csv(output / task / "subject_metrics.csv", subject_rows)

    source_tasks = ("motor_imagery", "mental_arithmetic", "wg")
    source_rows = [row for row in all_subject_rows if row["task"] in source_tasks]
    source_summary = {
        task: {
            "accuracy": task_summaries[task]["subject_level"]["accuracy"],
            "cohen_kappa": task_summaries[task]["subject_level"]["cohen_kappa"],
        }
        for task in source_tasks if task in task_summaries
    }
    result = {
        "schema": "sta_net_within_subject_aggregate_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol_name,
        "aggregation_unit": "subject_after_concatenating_disjoint_out_of_fold_dependency_groups",
        "task_summaries": task_summaries,
        "source_aligned_mi_ma_wg": source_summary,
        "fold_count": len(fold_rows),
        "protected_test_opened": True,
    }
    write_json(output / "summary.json", result)
    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "subject_metrics.csv", all_subject_rows)
    write_csv(output / "source_aligned_subject_metrics.csv", source_rows)

    lines = [
        f"# {report_title}",
        "",
        report_description,
        "",
        "| Task | Subjects | Folds | Primary endpoint | Mean | 95% subject bootstrap CI |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for task, summary in task_summaries.items():
        metric = "macro_f1" if summary["task_type"] == "classification" else "concordance_correlation"
        cell = summary["subject_level"][metric]
        ci = cell["subject_bootstrap_95_ci"]
        lines.append(
            f"| {task} | {summary['subject_count']} | {summary['fold_count']} | "
            f"mean subject {metric} | {cell['mean']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
        )
    lines.extend([
        "",
        "Pooled-sample metrics are diagnostics only and are not substituted for subject-level endpoints.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "output": str(output),
        "fold_count": len(fold_rows),
        "tasks": list(task_summaries),
    }, indent=2))


if __name__ == "__main__":
    main()
