#!/usr/bin/env python3
"""Aggregate strict cross-subject and sample-random STA-Net five-fold results."""

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
from scipy.stats import t

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch import get_sta_net_task_spec
from visualize_results import classification_metrics, regression_metrics

TASK_ORDER = (
    "motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual",
    "refed_regression",
)
TASK_LABELS = {
    "motor_imagery": "Motor imagery (MI)",
    "mental_arithmetic": "Mental arithmetic (MA)",
    "wg": "Word generation (WG)",
    "nback": "n-back",
    "dsr": "DSR",
    "visual": "Visual",
    "refed_regression": "REFED regression",
}
PROTOCOL_LABELS = {
    "strict_cross_subject": "Strict cross-subject",
    "sample_random": "Sample-level random split",
}
SOURCE_PAPER = {
    "motor_imagery": {"mean": 0.6965, "sample_sd": 0.0952},
    "mental_arithmetic": {"mean": 0.8514, "sample_sd": 0.0717},
    "wg": {"mean": 0.7903, "sample_sd": 0.0841},
}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def summarize(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 5:
        raise RuntimeError(f"formal five-fold summary expected 5 values, got {array.size}")
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1))
    half_width = float(t.ppf(0.975, df=4) * sample_sd / math.sqrt(5))
    return {
        "mean": mean,
        "sample_sd": sample_sd,
        "fold_t_95_ci": [mean - half_width, mean + half_width],
        "fold_count": 5,
        "values": array.tolist(),
    }


def metric_names(task: str) -> tuple[str, ...]:
    if get_sta_net_task_spec(task).task_type == "classification":
        return (
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            "cohen_kappa", "macro_roc_auc_ovr",
        )
    return (
        "concordance_correlation", "mae_native", "rmse_native",
        "r2_native", "pearson_r",
    )


def formatted(summary: Mapping[str, Any], *, percent: bool) -> str:
    scale = 100.0 if percent else 1.0
    decimals = 2 if percent else 3
    return (
        f"{float(summary['mean']) * scale:.{decimals}f} ± "
        f"{float(summary['sample_sd']) * scale:.{decimals}f}"
    )


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    protocol = json.loads((root / "protocol_freeze_manifest.json").read_text(encoding="utf-8"))
    jobs = protocol["jobs"]
    if len(jobs) != 2 * len(protocol["tasks"]) * 5:
        raise RuntimeError("protocol freeze does not contain two complete five-fold task grids")

    grouped_jobs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    fold_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    for job in jobs:
        grouped_jobs[(str(job["protocol_key"]), str(job["task"]))].append(job)

    summaries: dict[str, dict[str, Any]] = defaultdict(dict)
    for (protocol_key, task), task_jobs in sorted(grouped_jobs.items()):
        if len(task_jobs) != 5:
            raise RuntimeError(f"{protocol_key}/{task} has {len(task_jobs)} folds instead of 5")
        seen_test_indices: list[int] = []
        predictions, targets, masks = [], [], []
        fold_metric_values: dict[str, list[float]] = defaultdict(list)
        for job in sorted(task_jobs, key=lambda row: int(row["outer_fold"])):
            fold_dir = (
                root / "folds" / protocol_key / task / str(job["fold_id"]) / "evaluation"
            )
            summary_path = fold_dir / "summary.json"
            prediction_path = fold_dir / "protected_predictions.npz"
            if not summary_path.is_file() or not prediction_path.is_file():
                raise RuntimeError(f"incomplete fold {protocol_key}/{task}/{job['fold_id']}")
            evaluation = json.loads(summary_path.read_text(encoding="utf-8"))
            training_path = fold_dir.parent / "training" / "manifest.json"
            training = json.loads(training_path.read_text(encoding="utf-8"))
            if (
                training.get("status") != "completed"
                or training.get("convergence_reached") is not True
                or training.get("stop_reason") != "early_stopping_converged"
            ):
                raise RuntimeError(f"fold lacks validated convergence evidence: {training_path}")
            convergence_rows.append({
                "protocol": protocol_key,
                "task": task,
                "outer_fold": int(job["outer_fold"]),
                "last_epoch": int(training["last_epoch"]),
                "best_validation_epoch": int(training["best_validation_epoch"]),
                "epochs_after_best": (
                    int(training["last_epoch"]) - int(training["best_validation_epoch"])
                ),
                "epochs_without_improvement": int(training["epochs_without_improvement"]),
                "stop_reason": str(training["stop_reason"]),
                "selection_metric": str(training["selection_metric"]),
                "best_validation_metric": float(training["best_validation_metric"]),
            })
            protected = json.loads(Path(job["protected_manifest"]).read_text(encoding="utf-8"))
            if evaluation.get("task") != task:
                raise RuntimeError(f"evaluation task mismatch in {summary_path}")
            if int(evaluation.get("outer_fold")) != int(job["outer_fold"]):
                raise RuntimeError(f"evaluation outer-fold mismatch in {summary_path}")
            test_indices = [int(value) for value in protected["test_indices"]]
            seen_test_indices.extend(test_indices)
            data = np.load(prediction_path)
            predictions.append(data["prediction"])
            targets.append(data["target"])
            masks.append(data["target_valid_mask"])
            metrics = evaluation["metrics"]
            row: dict[str, Any] = {
                "protocol": protocol_key,
                "task": task,
                "outer_fold": int(job["outer_fold"]),
                "sample_count": int(evaluation["sample_count"]),
                "subject_count": int(evaluation["subject_count"]),
            }
            for name in metric_names(task):
                value = finite_float(metrics.get(name))
                row[name] = value
                if value is not None:
                    fold_metric_values[name].append(value)
            fold_rows.append(row)

        dataset_count = int(task_jobs[0]["dataset_sample_count"])
        if sorted(seen_test_indices) != list(range(dataset_count)):
            raise RuntimeError(
                f"{protocol_key}/{task} protected folds do not partition all {dataset_count} samples"
            )
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        mask = np.concatenate(masks)
        spec = get_sta_net_task_spec(task)
        if spec.task_type == "classification":
            pooled = classification_metrics(target, prediction, spec.class_names)
            primary = "macro_f1"
        else:
            pooled = regression_metrics(target, prediction, mask, spec.target_names)
            primary = "concordance_correlation"
        metric_summary = {
            name: summarize(values)
            for name, values in fold_metric_values.items()
            if len(values) == 5
        }
        if primary not in metric_summary:
            raise RuntimeError(f"missing primary metric {primary} for {protocol_key}/{task}")
        summaries[protocol_key][task] = {
            "task": task,
            "task_type": spec.task_type,
            "protocol": protocol_key,
            "primary_endpoint": primary,
            "fold_level": metric_summary,
            "pooled_out_of_fold_diagnostic": pooled,
            "sample_count": dataset_count,
            "subject_count": len({
                subject
                for job in task_jobs
                for subject in json.loads(
                    Path(job["public_manifest"]).read_text(encoding="utf-8")
                )["train_subjects"]
            }),
            "outer_fold_count": 5,
        }

    comparison_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        if task not in protocol["tasks"]:
            continue
        strict = summaries["strict_cross_subject"][task]
        random = summaries["sample_random"][task]
        primary = strict["primary_endpoint"]
        strict_primary = strict["fold_level"][primary]
        random_primary = random["fold_level"][primary]
        comparison_rows.append({
            "task": task,
            "task_label": TASK_LABELS[task],
            "metric": primary,
            "strict_mean": strict_primary["mean"],
            "strict_sample_sd": strict_primary["sample_sd"],
            "sample_random_mean": random_primary["mean"],
            "sample_random_sample_sd": random_primary["sample_sd"],
            "sample_random_minus_strict": (
                random_primary["mean"] - strict_primary["mean"]
            ),
            "fold_count": 5,
        })
        if strict["task_type"] == "classification":
            strict_accuracy = strict["fold_level"]["accuracy"]
            random_accuracy = random["fold_level"]["accuracy"]
            accuracy_rows.append({
                "task": task,
                "task_label": TASK_LABELS[task],
                "strict_mean": strict_accuracy["mean"],
                "strict_sample_sd": strict_accuracy["sample_sd"],
                "sample_random_mean": random_accuracy["mean"],
                "sample_random_sample_sd": random_accuracy["sample_sd"],
                "sample_random_minus_strict": (
                    random_accuracy["mean"] - strict_accuracy["mean"]
                ),
                "fold_count": 5,
            })
            if task in SOURCE_PAPER:
                source = SOURCE_PAPER[task]
                source_rows.append({
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "strict_mean": strict_accuracy["mean"],
                    "strict_sample_sd": strict_accuracy["sample_sd"],
                    "sample_random_mean": random_accuracy["mean"],
                    "sample_random_sample_sd": random_accuracy["sample_sd"],
                    "original_paper_mean": source["mean"],
                    "original_paper_sample_sd": source["sample_sd"],
                    "strict_minus_original": strict_accuracy["mean"] - source["mean"],
                    "sample_random_minus_original": random_accuracy["mean"] - source["mean"],
                })

    output = root / "aggregate"
    result = {
        "schema": "sta_net_strict_vs_sample_random_5fold_aggregate_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregation_unit": "outer_fold",
        "uncertainty": "sample standard deviation across five outer folds (ddof=1)",
        "task_summaries": dict(summaries),
        "primary_comparison": comparison_rows,
        "classification_accuracy_comparison": accuracy_rows,
        "source_paper_accuracy_comparison": source_rows,
        "source_paper": protocol["source_paper"],
        "artifact_mask_policy": protocol.get("artifact_mask_policy"),
        "adapter_validity_source": protocol.get("adapter_validity_source"),
        "convergence_audit": {
            "all_folds_converged": True,
            "fold_count": len(convergence_rows),
            "stop_rule": (
                "at least 40 epochs and at least 30 validation epochs without an "
                "improvement in the task checkpoint-selection metric"
            ),
        },
        "protected_test_opened": True,
    }
    write_json(output / "summary.json", result)
    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "primary_comparison.csv", comparison_rows)
    write_csv(output / "classification_accuracy_comparison.csv", accuracy_rows)
    write_csv(output / "source_paper_accuracy_comparison.csv", source_rows)
    write_csv(output / "training_convergence.csv", convergence_rows)

    lines = [
        "# STA-Net five-fold benchmark: strict cross-subject vs sample-level random split",
        "",
        "Values are mean ± sample SD across the five outer folds. Classification values "
        "are percentages; REFED CCC is unitless. Hyperparameters were frozen before opening "
        "any outer test fold.",
        "",
        (
            f"All {len(convergence_rows)} fold trainings ended by the frozen validation-"
            "convergence rule rather than the maximum-epoch cap. Artifact masks were not "
            "consumed; only real record support from `valid_mask` was used."
        ),
        "",
        "## Primary endpoints",
        "",
        "| Task | Metric | Strict cross-subject | Sample-level random split | Δ random − strict |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison_rows:
        percent = row["metric"] == "macro_f1"
        delta_scale = 100.0 if percent else 1.0
        lines.append(
            f"| {row['task_label']} | {'Macro-F1 (%)' if percent else 'CCC'} | "
            f"{formatted({'mean': row['strict_mean'], 'sample_sd': row['strict_sample_sd']}, percent=percent)} | "
            f"{formatted({'mean': row['sample_random_mean'], 'sample_sd': row['sample_random_sample_sd']}, percent=percent)} | "
            f"{row['sample_random_minus_strict'] * delta_scale:+.{2 if percent else 3}f} |"
        )
    lines.extend([
        "",
        "## Classification accuracy",
        "",
        "| Task | Strict cross-subject Accuracy (%) | Sample-level random Accuracy (%) | Δ (pp) |",
        "|---|---:|---:|---:|",
    ])
    for row in accuracy_rows:
        lines.append(
            f"| {row['task_label']} | "
            f"{formatted({'mean': row['strict_mean'], 'sample_sd': row['strict_sample_sd']}, percent=True)} | "
            f"{formatted({'mean': row['sample_random_mean'], 'sample_sd': row['sample_random_sample_sd']}, percent=True)} | "
            f"{row['sample_random_minus_strict'] * 100:+.2f} |"
        )
    lines.extend([
        "",
        "## Accuracy comparison with the original STA-Net paper",
        "",
        "| Source task | Strict cross-subject (%) | Sample-level random (%) | Original paper (%) | "
        "Δ strict − paper (pp) | Δ random − paper (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in source_rows:
        lines.append(
            f"| {row['task_label']} | "
            f"{formatted({'mean': row['strict_mean'], 'sample_sd': row['strict_sample_sd']}, percent=True)} | "
            f"{formatted({'mean': row['sample_random_mean'], 'sample_sd': row['sample_random_sample_sd']}, percent=True)} | "
            f"{row['original_paper_mean'] * 100:.2f} ± {row['original_paper_sample_sd'] * 100:.2f} | "
            f"{row['strict_minus_original'] * 100:+.2f} | "
            f"{row['sample_random_minus_original'] * 100:+.2f} |"
        )
    lines.extend([
        "",
        "The original-paper column is contextual rather than a same-protocol statistical "
        "comparison: the paper reports subject-specific MI/MA/WG evaluation, whereas the two "
        "new columns use this project's unified preprocessing and outer five-fold definitions.",
        "",
        "The sample-level split deliberately does not isolate subjects, recordings, trials, or "
        "overlapping-window dependency groups. It measures an information-visible upper-bound "
        "setting requested for this benchmark and must be labeled exactly as such.",
        "",
        "n-back, DSR, Visual, and REFED are project adaptations not evaluated by the original "
        "STA-Net paper; no original-paper number is imputed for them.",
    ])
    (output / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{STA-Net performance under strict cross-subject and sample-level random "
        r"five-fold evaluation. Values are mean $\pm$ sample SD across outer folds.}",
        r"\label{tab:sta_net_fivefold}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Task & Metric & Strict cross-subject & Sample-random & $\Delta$ \\",
        r"\midrule",
    ]
    for row in comparison_rows:
        percent = row["metric"] == "macro_f1"
        scale = 100.0 if percent else 1.0
        decimals = 2 if percent else 3
        metric_label = "Macro-F1 (\\%)" if percent else "CCC"
        latex.append(
            f"{latex_escape(row['task_label'])} & {metric_label} & "
            f"{row['strict_mean'] * scale:.{decimals}f} $\\pm$ "
            f"{row['strict_sample_sd'] * scale:.{decimals}f} & "
            f"{row['sample_random_mean'] * scale:.{decimals}f} $\\pm$ "
            f"{row['sample_random_sample_sd'] * scale:.{decimals}f} & "
            f"{row['sample_random_minus_strict'] * scale:+.{decimals}f} \\\\"
        )
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    (output / "paper_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "output": str(output),
        "job_count": len(jobs),
        "task_count": len(protocol["tasks"]),
    }, indent=2))


if __name__ == "__main__":
    main()
