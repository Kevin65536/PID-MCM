#!/usr/bin/env python3
"""Audit and summarize the existing STA-Net protocol bridge.

This analysis is deliberately read-only with respect to the completed protected
runs.  It reconstructs subject-level metrics from the five-fold out-of-fold
predictions so that trial-random and strict cross-subject estimates use the
same aggregation unit as the dependency-group-safe within-subject aggregate.
The resulting values are a descriptive protocol-sensitivity bridge, not a
causal or additive decomposition.

The default inputs are the completed 2026-07-25 five-fold run and the completed
2026-07-24 within-subject run.  A target-subject fine-tuning run is reported as
context only and is never inserted into the three-protocol bridge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)

METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_FIVEFOLD_ROOT = (
    METHOD_ROOT
    / "runs"
    / "fivefold"
    / "20260725_sta_net_strict_vs_sample_random_5fold_v2_cached_100ep"
)
DEFAULT_WITHIN_ROOT = (
    METHOD_ROOT
    / "runs"
    / "within_subject"
    / "20260724_sta_net_within_subject_all_tasks_v1_100ep"
)
DEFAULT_PERSONALIZED_ROOT = (
    METHOD_ROOT
    / "runs"
    / "personalized_finetune"
    / "20260725_sta_net_personalized_finetune_final_v1_20ep"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "comparative_methods"
    / "runs"
    / "performance_analysis"
    / "20260816_p0"
    / "stanet_bridge"
)

TASK_ORDER = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
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
CLASSIFICATION_TASKS = tuple(task for task in TASK_ORDER if task != "refed_regression")
PROTOCOL_ORDER = ("trial_random", "group_safe_within_subject", "cross_subject")
PROTOCOL_LABELS = {
    "trial_random": "Trial-random",
    "group_safe_within_subject": "Group-safe within-subject",
    "cross_subject": "Strict cross-subject",
}
PROTOCOL_COLORS = {
    "trial_random": "#0072B2",  # Okabe-Ito blue
    "group_safe_within_subject": "#D55E00",  # vermillion
    "cross_subject": "#009E73",  # bluish green
}
PROTOCOL_MARKERS = {
    "trial_random": "o",
    "group_safe_within_subject": "s",
    "cross_subject": "^",
}
# Chance is task-specific; a single 0.5 line would be wrong for n-back and Visual.
CHANCE_LEVELS = {
    "motor_imagery": 0.5,
    "mental_arithmetic": 0.5,
    "wg": 0.5,
    "nback": 1.0 / 3.0,
    "dsr": 0.5,
    "visual": 0.25,
}
PRIMARY_METRIC = {
    task: ("concordance_correlation" if task == "refed_regression" else "macro_f1")
    for task in TASK_ORDER
}


def jsonable(value: Any) -> Any:
    """Convert numpy/path scalars before writing a JSON artifact."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(jsonable(rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = 20260816, draws: int = 10_000
) -> tuple[float | None, float | None]:
    """Return percentile bootstrap CI for a subject-level mean.

    Subjects, rather than windows, are the resampling units.  The function is
    intentionally deterministic and returns missing values for an empty input.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None, None
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def concordance_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.size == 0 or prediction.size != target.size:
        return float("nan")
    covariance = float(np.mean((target - target.mean()) * (prediction - prediction.mean())))
    denominator = float(
        target.var() + prediction.var() + (target.mean() - prediction.mean()) ** 2
    )
    return 2.0 * covariance / denominator if denominator > 0 else float("nan")


def classification_subject_rows(
    target: np.ndarray,
    probability: np.ndarray,
    subjects: np.ndarray,
    *,
    task: str,
    protocol: str,
    source_artifact: str,
) -> list[dict[str, Any]]:
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64)
    subjects = np.asarray(subjects, dtype=str).reshape(-1)
    if probability.ndim != 2 or probability.shape[0] != target.size:
        raise ValueError(f"classification prediction shape mismatch for {task}: {probability.shape}")
    if subjects.size != target.size:
        raise ValueError(f"subject shape mismatch for {task}: {subjects.shape} vs {target.shape}")
    predicted = probability.argmax(axis=1)
    labels = np.arange(probability.shape[1])
    rows: list[dict[str, Any]] = []
    for subject in sorted(np.unique(subjects).tolist()):
        keep = subjects == subject
        if len(np.unique(target[keep])) < 2 and len(np.unique(predicted[keep])) < 2:
            kappa = None
        else:
            kappa = finite(cohen_kappa_score(target[keep], predicted[keep], labels=labels))
        rows.append({
            "task": task,
            "task_label": TASK_LABELS[task],
            "protocol": protocol,
            "protocol_label": PROTOCOL_LABELS[protocol],
            "subject": str(subject),
            "metric": "macro_f1",
            "estimate": float(f1_score(
                target[keep], predicted[keep], labels=labels, average="macro", zero_division=0
            )),
            "accuracy": float(np.mean(predicted[keep] == target[keep])),
            "balanced_accuracy": float(
                balanced_accuracy_score(target[keep], predicted[keep])
            ),
            "cohen_kappa": kappa,
            "sample_count": int(keep.sum()),
            "valid_coordinate_count": int(keep.sum()),
            "source_artifact": source_artifact,
        })
    return rows


def regression_subject_rows(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    subjects: np.ndarray,
    *,
    task: str,
    protocol: str,
    source_artifact: str,
) -> list[dict[str, Any]]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    subjects = np.asarray(subjects, dtype=str).reshape(-1)
    if target.shape != prediction.shape or target.shape != valid_mask.shape:
        raise ValueError(
            f"regression shape mismatch for {task}: target={target.shape}, "
            f"prediction={prediction.shape}, mask={valid_mask.shape}"
        )
    if target.ndim < 2 or target.shape[0] != subjects.size:
        raise ValueError(f"regression subject shape mismatch for {task}: {target.shape}")
    rows: list[dict[str, Any]] = []
    for subject in sorted(np.unique(subjects).tolist()):
        keep = subjects == subject
        coordinate_keep = valid_mask[keep]
        truth = target[keep][coordinate_keep]
        estimate = prediction[keep][coordinate_keep]
        error = estimate - truth
        pearson = (
            float(pearsonr(truth, estimate).statistic)
            if truth.size > 1 and np.std(truth) > 0 and np.std(estimate) > 0
            else None
        )
        ccc = concordance_correlation(truth, estimate)
        rows.append({
            "task": task,
            "task_label": TASK_LABELS[task],
            "protocol": protocol,
            "protocol_label": PROTOCOL_LABELS[protocol],
            "subject": str(subject),
            "metric": "concordance_correlation",
            "estimate": finite(ccc),
            "mae_native": finite(np.mean(np.abs(error))) if truth.size else None,
            "rmse_native": finite(np.sqrt(np.mean(error**2))) if truth.size else None,
            "pearson_r": pearson,
            "sample_count": int(keep.sum()),
            "valid_coordinate_count": int(truth.size),
            "source_artifact": source_artifact,
        })
    return rows


def _prediction_paths(root: Path, protocol_key: str, task: str) -> list[Path]:
    paths = sorted(
        (root / "folds" / protocol_key / task).glob("outer*/evaluation/protected_predictions.npz")
    )
    return paths


def load_fivefold_subject_rows(
    root: Path,
    protocol_key: str,
    task: str,
    *,
    expected_folds: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one five-fold protocol and reconstruct subject-level OOF metrics."""

    paths = _prediction_paths(root, protocol_key, task)
    if len(paths) != expected_folds:
        raise RuntimeError(
            f"{protocol_key}/{task}: expected {expected_folds} protected prediction files, "
            f"found {len(paths)}"
        )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    for path in paths:
        data = np.load(path)
        required = {"prediction", "target", "target_valid_mask", "subject", "sample_id"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"{path} missing NPZ keys: {sorted(missing)}")
        predictions.append(np.asarray(data["prediction"]))
        targets.append(np.asarray(data["target"]))
        masks.append(np.asarray(data["target_valid_mask"]))
        subjects.append(np.asarray(data["subject"], dtype=str))
        sample_ids.append(np.asarray(data["sample_id"], dtype=str))
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    mask = np.concatenate(masks, axis=0)
    subject = np.concatenate(subjects, axis=0)
    sample_id = np.concatenate(sample_ids, axis=0)
    if sample_id.size != len(set(sample_id.tolist())):
        raise RuntimeError(f"{protocol_key}/{task}: protected sample IDs are not unique")
    source = str((root / "folds" / protocol_key / task).resolve())
    if task == "refed_regression":
        rows = regression_subject_rows(
            target, prediction, mask, subject,
            task=task, protocol=("trial_random" if protocol_key == "sample_random" else "cross_subject"),
            source_artifact=source,
        )
    else:
        rows = classification_subject_rows(
            target, prediction, subject,
            task=task, protocol=("trial_random" if protocol_key == "sample_random" else "cross_subject"),
            source_artifact=source,
        )
    inventory = {
        "protocol_key": protocol_key,
        "protocol": "trial_random" if protocol_key == "sample_random" else "cross_subject",
        "task": task,
        "fold_count": len(paths),
        "sample_count": int(sample_id.size),
        "subject_count": int(len(set(subject.tolist()))),
        "subjects": sorted(set(subject.tolist())),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(sample_id.tolist())).encode("utf-8")
        ).hexdigest(),
        "source_files": [str(path.resolve()) for path in paths],
    }
    return rows, inventory


def load_within_subject_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the already aggregated group-safe within-subject endpoint."""

    aggregate = root / "aggregate"
    summary_path = aggregate / "summary.json"
    subject_path = aggregate / "subject_metrics.csv"
    if not summary_path.is_file() or not subject_path.is_file():
        raise RuntimeError(f"within-subject aggregate is incomplete under {aggregate}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with subject_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            task = raw.get("task", "")
            if task not in PRIMARY_METRIC:
                continue
            metric = PRIMARY_METRIC[task]
            value = finite(raw.get(metric))
            if value is None:
                continue
            rows.append({
                "task": task,
                "task_label": TASK_LABELS[task],
                "protocol": "group_safe_within_subject",
                "protocol_label": PROTOCOL_LABELS["group_safe_within_subject"],
                "subject": str(raw.get("subject", "")),
                "metric": metric,
                "estimate": value,
                "accuracy": finite(raw.get("accuracy")),
                "balanced_accuracy": finite(raw.get("balanced_accuracy")),
                "cohen_kappa": finite(raw.get("cohen_kappa")),
                "mae_native": finite(raw.get("mae_native")),
                "rmse_native": finite(raw.get("rmse_native")),
                "pearson_r": finite(raw.get("pearson_r")),
                "sample_count": int(raw.get("sample_count", 0) or 0),
                "valid_coordinate_count": None,
                "source_artifact": str(subject_path.resolve()),
            })
    inventory = {
        "protocol": "group_safe_within_subject",
        "protocol_key": "single_subject_nested_cv",
        "task_count": len({row["task"] for row in rows}),
        "source_summary": str(summary_path.resolve()),
        "source_subject_metrics": str(subject_path.resolve()),
        "aggregation_unit": summary.get(
            "aggregation_unit",
            "subject_after_concatenating_disjoint_out_of_fold_dependency_groups",
        ),
        "protected_test_opened": bool(summary.get("protected_test_opened", True)),
        "subject_count_by_task": {
            task: len({row["subject"] for row in rows if row["task"] == task})
            for task in TASK_ORDER
        },
        "sample_count_by_task": {
            task: int(sum(row["sample_count"] for row in rows if row["task"] == task))
            for task in TASK_ORDER
        },
    }
    return rows, inventory


def load_personalized_context(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load target-subject fine-tuning values solely for a context table."""

    path = root / "aggregate" / "subject_metrics.csv"
    summary_path = root / "aggregate" / "summary.json"
    if not path.is_file():
        return [], {
            "available": False,
            "protocol": "target_subject_finetune",
            "reason": f"missing {path}",
        }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            task = raw.get("task", "")
            if task not in PRIMARY_METRIC:
                continue
            metric = PRIMARY_METRIC[task]
            value = finite(raw.get(metric))
            if value is None:
                continue
            rows.append({
                "task": task,
                "task_label": TASK_LABELS[task],
                "protocol": "target_subject_finetune",
                "metric": metric,
                "subject": str(raw.get("subject", "")),
                "estimate": value,
                "sample_count": int(raw.get("sample_count", 0) or 0),
                "source_artifact": str(path.resolve()),
                "bridge_status": "context_only",
                "reason": (
                    "Target-subject calibration/fine-tuning changes the estimand; "
                    "it is not one of the three bridge protocols."
                ),
            })
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return rows, {
        "available": True,
        "protocol": "target_subject_finetune",
        "source_summary": str(summary_path.resolve()) if summary_path.is_file() else None,
        "source_subject_metrics": str(path.resolve()),
        "task_summaries": {
            task: summary.get("task_summaries", {}).get(task, {})
            for task in TASK_ORDER
        },
    }


def summarize_subject_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one task/protocol's subject rows with deterministic CIs."""

    if not rows:
        return {
            "estimate_subject_mean": None,
            "subject_sample_sd": None,
            "ci_lower": None,
            "ci_upper": None,
            "subject_count": 0,
            "sample_count": 0,
        }
    values = np.asarray(
        [float(row["estimate"]) for row in rows if finite(row.get("estimate")) is not None],
        dtype=np.float64,
    )
    ci_lower, ci_upper = bootstrap_mean_ci(values)
    sample_count = sum(int(row.get("sample_count", 0) or 0) for row in rows)
    return {
        "estimate_subject_mean": float(values.mean()) if values.size else None,
        "subject_sample_sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "subject_count": int(values.size),
        "sample_count": int(sample_count),
    }


def _subject_sets(by_protocol: Mapping[str, Sequence[Mapping[str, Any]]], task: str) -> dict[str, set[str]]:
    return {
        protocol: {str(row["subject"]) for row in rows if row["task"] == task}
        for protocol, rows in by_protocol.items()
    }


def build_bridge(
    fivefold_root: Path,
    within_root: Path,
    personalized_root: Path,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    inventories: dict[str, Any] = {}
    protocol_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for protocol_key, protocol in (("sample_random", "trial_random"), ("strict_cross_subject", "cross_subject")):
        for task in TASK_ORDER:
            rows, inventory = load_fivefold_subject_rows(fivefold_root, protocol_key, task)
            all_rows.extend(rows)
            protocol_rows[protocol].extend(rows)
            inventories[f"{protocol}/{task}"] = inventory
    within_rows, within_inventory = load_within_subject_rows(within_root)
    all_rows.extend(within_rows)
    protocol_rows["group_safe_within_subject"].extend(within_rows)
    inventories["group_safe_within_subject"] = within_inventory
    personalized_rows, personalized_inventory = load_personalized_context(personalized_root)

    summary_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        task_rows = {
            protocol: [row for row in protocol_rows.get(protocol, []) if row["task"] == task]
            for protocol in PROTOCOL_ORDER
        }
        sets = _subject_sets(task_rows, task)
        random_cross_match = sets["trial_random"] == sets["cross_subject"] and bool(sets["trial_random"])
        within_match = sets["group_safe_within_subject"] == sets["cross_subject"] and bool(sets["group_safe_within_subject"])
        sample_counts = {
            protocol: sum(int(row.get("sample_count", 0) or 0) for row in rows)
            for protocol, rows in task_rows.items()
        }
        random_inventory = inventories[f"trial_random/{task}"]
        cross_inventory = inventories[f"cross_subject/{task}"]
        sample_id_universe_match = (
            random_inventory.get("sample_ids_sha256") == cross_inventory.get("sample_ids_sha256")
        )
        random_cross_sample_match = (
            sample_counts["trial_random"] == sample_counts["cross_subject"]
            and sample_id_universe_match
        )
        within_sample_match = sample_counts["group_safe_within_subject"] == sample_counts["cross_subject"]
        pair_eligible = random_cross_match and random_cross_sample_match
        trio_eligible = pair_eligible and within_match and within_sample_match
        reasons: list[str] = []
        if not random_cross_match:
            reasons.append("trial-random and cross-subject subject sets differ")
        if not random_cross_sample_match:
            reasons.append(
                "trial-random and cross-subject sample universe differs"
                if not sample_id_universe_match
                else "trial-random and cross-subject sample counts differ"
            )
        if not within_match:
            reasons.append(
                "group-safe within-subject subject set differs from five-fold protocols "
                f"({len(sets['group_safe_within_subject'])} vs {len(sets['cross_subject'])})"
            )
        if not within_sample_match:
            reasons.append(
                "group-safe within-subject sample count differs from five-fold protocols "
                f"({sample_counts['group_safe_within_subject']} vs {sample_counts['cross_subject']})"
            )
        if not reasons:
            reasons.append(
                "same task, primary metric, subject-level aggregation, subject universe, "
                "and sample count; protocol training/dependency budgets still differ"
            )
        eligibility_rows.append({
            "task": task,
            "task_label": TASK_LABELS[task],
            "metric": PRIMARY_METRIC[task],
            "chance_baseline": CHANCE_LEVELS.get(task),
            "trial_random_subject_count": len(sets["trial_random"]),
            "group_safe_within_subject_count": len(sets["group_safe_within_subject"]),
            "cross_subject_count": len(sets["cross_subject"]),
            "trial_random_sample_count": sample_counts["trial_random"],
            "group_safe_within_subject_sample_count": sample_counts["group_safe_within_subject"],
            "cross_subject_sample_count": sample_counts["cross_subject"],
            "trial_random_cross_subject_sample_universe_match": sample_id_universe_match,
            "within_subject_sample_universe_check": "count_only; aggregate has no sample IDs",
            "trial_random_cross_subject_pair_eligible": pair_eligible,
            "three_protocol_bridge_eligible": trio_eligible,
            "reason": " ".join(reasons),
        })
        for protocol in PROTOCOL_ORDER:
            rows = task_rows[protocol]
            summary = summarize_subject_rows(rows)
            row = {
                "task": task,
                "task_label": TASK_LABELS[task],
                "metric": PRIMARY_METRIC[task],
                "chance_baseline": CHANCE_LEVELS.get(task),
                "protocol": protocol,
                "protocol_label": PROTOCOL_LABELS[protocol],
                "aggregation_unit": "subject_mean_over_OOF_predictions",
                "direct_pair_bridge_eligible": pair_eligible if protocol in {"trial_random", "cross_subject"} else False,
                "three_protocol_bridge_eligible": trio_eligible,
                "comparability_note": (
                    "directly composable within the three-protocol descriptive bridge"
                    if trio_eligible
                    else "reported but excluded from three-protocol bridge for this task"
                ),
                "source_artifact": (
                    str(within_root.resolve() / "aggregate" / "subject_metrics.csv")
                    if protocol == "group_safe_within_subject"
                    else inventories[f"{protocol}/{task}"]["source_files"][0]
                ),
            }
            row.update(summary)
            summary_rows.append(row)
        if not trio_eligible:
            missing_rows.append({
                "task": task,
                "missing_protocol": "group_safe_within_subject" if not within_match or not within_sample_match else "none",
                "status": "not_directly_composable",
                "reason": " ".join(reasons),
                "source_artifact": str(within_root.resolve() / "aggregate" / "subject_metrics.csv"),
            })
    for row in personalized_rows:
        context_rows.append(row)
    return {
        "subject_rows": all_rows,
        "summary_rows": summary_rows,
        "eligibility_rows": eligibility_rows,
        "context_rows": context_rows,
        "missing_rows": missing_rows,
        "inventories": inventories,
        "personalized_inventory": personalized_inventory,
    }


def _plot_bridge(summary_rows: Sequence[Mapping[str, Any]], output_stem: Path) -> list[str]:
    """Create a static point-and-CI figure without implying a causal path."""

    frame = list(summary_rows)
    fig, axes = plt.subplots(
        2, 1, figsize=(11.0, 7.2), gridspec_kw={"height_ratios": [2.5, 1.2]}, constrained_layout=True
    )
    x_tasks = list(CLASSIFICATION_TASKS)
    x = np.arange(len(x_tasks), dtype=float)
    offsets = {"trial_random": -0.22, "group_safe_within_subject": 0.0, "cross_subject": 0.22}
    for protocol in PROTOCOL_ORDER:
        rows = [row for row in frame if row["protocol"] == protocol and row["task"] in x_tasks]
        values = {row["task"]: row for row in rows}
        xx, yy, lo, hi = [], [], [], []
        for index, task in enumerate(x_tasks):
            row = values.get(task)
            if not row or row.get("estimate_subject_mean") is None:
                continue
            if protocol == "group_safe_within_subject" and not row.get("three_protocol_bridge_eligible", False):
                # Visual has a different subject/sample universe in the existing
                # within-subject run.  Do not draw it as a bridge node.
                continue
            xx.append(index + offsets[protocol])
            yy.append(float(row["estimate_subject_mean"]))
            lower = row.get("ci_lower")
            upper = row.get("ci_upper")
            lo.append(float(yy[-1] - lower) if lower is not None else 0.0)
            hi.append(float(upper - yy[-1]) if upper is not None else 0.0)
        axes[0].errorbar(
            xx, yy, yerr=[lo, hi], fmt=PROTOCOL_MARKERS[protocol], ms=6, capsize=3,
            color=PROTOCOL_COLORS[protocol],
            label=PROTOCOL_LABELS[protocol],
        )
    for index, task in enumerate(x_tasks):
        axes[0].hlines(
            CHANCE_LEVELS[task], index - 0.36, index + 0.36,
            color="#777777", lw=1.0, ls="--",
            label="Task-specific chance" if index == 0 else None,
        )
    visual_index = x_tasks.index("visual")
    axes[0].scatter(
        [visual_index], [0.31], marker="x", s=56, linewidths=1.8,
        color=PROTOCOL_COLORS["group_safe_within_subject"],
        label="Within-subject unavailable" if visual_index == 0 else None,
        zorder=4,
    )
    axes[0].text(
        visual_index,
        0.32,
        "within NA\n(11/16 subj.)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=PROTOCOL_COLORS["group_safe_within_subject"],
    )
    axes[0].set(
        title="STA-Net protocol bridge — classification",
        ylabel="Subject-level macro-F1",
        xticks=x,
        xticklabels=[TASK_LABELS[task] for task in x_tasks],
        ylim=(-0.02, 0.90),
    )
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].legend(frameon=False, ncol=4, loc="upper right")
    axes[0].grid(axis="y", alpha=0.2)
    reg_rows = [row for row in frame if row["task"] == "refed_regression"]
    xreg = np.arange(1, dtype=float)
    for protocol in PROTOCOL_ORDER:
        row = next((item for item in reg_rows if item["protocol"] == protocol), None)
        if not row or row.get("estimate_subject_mean") is None:
            continue
        y = float(row["estimate_subject_mean"])
        lower = row.get("ci_lower")
        upper = row.get("ci_upper")
        axes[1].errorbar(
            xreg[0] + offsets[protocol], y,
            yerr=[[y - float(lower)] if lower is not None else [0.0],
                  [float(upper) - y] if upper is not None else [0.0]],
            fmt=PROTOCOL_MARKERS[protocol], ms=6, capsize=3, color=PROTOCOL_COLORS[protocol],
        )
    axes[1].axhline(0.0, color="#777777", lw=0.8, ls="--", label="CCC = 0")
    axes[1].set(
        title="STA-Net protocol bridge — REFED regression",
        ylabel="Subject-level CCC",
        xticks=xreg,
        xticklabels=[TASK_LABELS["refed_regression"]],
        ylim=(-0.20, 0.30),
    )
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Subject-level means with 95% bootstrap CIs (subjects are the resampling unit)\n"
        "Descriptive protocol sensitivity; not a causal/additive decomposition",
        fontsize=12,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _input_files(
    fivefold_root: Path, within_root: Path, personalized_root: Path
) -> list[Path]:
    paths: list[Path] = []
    for root in (fivefold_root, within_root, personalized_root):
        if not root.exists():
            continue
        for path in sorted(root.rglob("protected_predictions.npz")):
            paths.append(path)
        for name in (
            "protocol_freeze_manifest.json", "aggregate/summary.json",
            "aggregate/subject_metrics.csv", "aggregate/protocol_audit.json",
        ):
            candidate = root / name
            if candidate.is_file():
                paths.append(candidate)
    return paths


def write_report(
    output: Path,
    result: Mapping[str, Any],
    *,
    fivefold_root: Path,
    within_root: Path,
    personalized_root: Path,
    figure_paths: Sequence[str],
) -> None:
    eligibility = result["eligibility_rows"]
    summaries = result["summary_rows"]
    summary_lookup = {(row["task"], row["protocol"]): row for row in summaries}
    lines = [
        "# STA-Net protocol bridge analysis (P0)",
        "",
        "**Analysis date:** 2026-08-16  ",
        "**Role:** read-only post-hoc analysis of already completed protected runs",
        "",
        "## Scope and estimand",
        "",
        "This report audits whether existing STA-Net results can be placed on a common",
        "trial-random → group-safe within-subject → strict cross-subject bridge. The",
        "five-fold prediction artifacts are re-aggregated at the subject level; this",
        "puts their endpoint on the same subject-level unit as the within-subject",
        "aggregate. The result is descriptive protocol sensitivity, not a causal or",
        "additive decomposition: training data volume, dependency isolation, and",
        "optimization opportunities differ across protocols.",
        "",
        "Primary endpoints are macro-F1 for classification tasks and concordance",
        "correlation coefficient (CCC) for REFED regression. Each sample is used once",
        "in the five-fold out-of-fold prediction reconstruction. Bootstrap intervals",
        "resample subjects (10,000 draws; seed 20260816), never individual windows.",
        "",
        "## Existing inputs",
        "",
        f"- Trial-random / strict cross-subject: `{fivefold_root}`",
        f"- Group-safe within-subject: `{within_root}`",
        f"- Target-subject fine-tuning (context only): `{personalized_root}`",
        "",
        "No model was retrained and no protected prediction was changed by this audit.",
        "Source-paper accuracy rows, source-aligned MI/MA/WG rows, and personalized",
        "target-subject fine-tuning are not silently mixed into the bridge.",
        "",
        "## Bridge eligibility",
        "",
        "| Task | B0 | Trial-random | Group-safe within-subject | Strict cross-subject | Three-protocol bridge |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in eligibility:
        task = row["task"]
        chance = CHANCE_LEVELS.get(task)
        chance_text = f"{chance:.4f}" if chance is not None else "NA"
        cells = []
        for protocol in PROTOCOL_ORDER:
            summary = summary_lookup.get((task, protocol), {})
            value = summary.get("estimate_subject_mean")
            if protocol == "group_safe_within_subject" and not row["three_protocol_bridge_eligible"]:
                cells.append("NA")
                continue
            cells.append("NA" if value is None else f"{float(value):.4f}")
        lines.append(
            f"| {row['task_label']} ({row['metric']}) | {chance_text} | {cells[0]} | {cells[1]} | "
            f"{cells[2]} | {'YES' if row['three_protocol_bridge_eligible'] else 'NO'} |"
        )
    lines.extend([
        "",
        "The values above are shown as subject-level means. A three-protocol value is",
        "eligible only when all three protocols have the same task/metric, subject",
        "universe, and sample count. Visual is intentionally excluded from the full",
        "three-protocol bridge because the existing within-subject run contains 11 of",
        "the 16 subjects and 5,476 of the 7,720 five-fold samples. Trial-random and",
        "strict cross-subject sample universes are checked by sorted sample-ID SHA-256;",
        "the within-subject aggregate exposes counts but no sample IDs, so its check",
        "is explicitly count-level. The Visual within value remains in the machine",
        "summary with an ineligible flag, but is displayed as NA in the bridge tables.",
        "",
        "## Numeric summaries",
        "",
        "The machine-readable `bridge_protocol_summary.csv` is authoritative. Its CIs",
        "are subject bootstrap percentile intervals. `direct_pair_bridge_eligible`",
        "refers only to the trial-random/cross-subject pair;",
        "`three_protocol_bridge_eligible` is the stricter flag.",
        "",
        "| Task | B0 | Protocol | Endpoint | Mean | 95% subject CI | n subjects | n samples |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ])
    for task in TASK_ORDER:
        chance = CHANCE_LEVELS.get(task)
        chance_text = f"{chance:.4f}" if chance is not None else "NA"
        for protocol in PROTOCOL_ORDER:
            row = summary_lookup[(task, protocol)]
            mean = row["estimate_subject_mean"]
            if protocol == "group_safe_within_subject" and not row["three_protocol_bridge_eligible"]:
                lines.append(
                    f"| {TASK_LABELS[task]} | {chance_text} | {PROTOCOL_LABELS[protocol]} | {PRIMARY_METRIC[task]} | NA | "
                    "non-comparable (see eligibility CSV) | "
                    f"{int(row['subject_count'])} | {int(row['sample_count'])} |"
                )
                continue
            if mean is None:
                lines.append(
                    f"| {TASK_LABELS[task]} | {chance_text} | {PROTOCOL_LABELS[protocol]} | {PRIMARY_METRIC[task]} | NA | NA | 0 | 0 |"
                )
                continue
            lines.append(
                f"| {TASK_LABELS[task]} | {chance_text} | {PROTOCOL_LABELS[protocol]} | {PRIMARY_METRIC[task]} | "
                f"{float(mean):.4f} | [{float(row['ci_lower']):.4f}, {float(row['ci_upper']):.4f}] | "
                f"{int(row['subject_count'])} | {int(row['sample_count'])} |"
            )
    lines.extend([
        "",
        "## Non-comparable or context-only evidence",
        "",
        "- The original paper's subject-specific accuracy references are contextual",
        "  only; they use different estimands and metric names and are not bridge nodes.",
        "- Target-subject fine-tuning is emitted in `context_only_metrics.csv`. It",
        "  includes target calibration and therefore cannot be interpreted as either",
        "  strict cross-subject or group-safe within-subject performance.",
        "- The existing within-subject aggregate is valid as its own endpoint, but its",
        "  MI/MA/WG source-aligned accuracy companion is not used here because the",
        "  metric and split differ from the shared macro-F1/CCC endpoint.",
        "",
        "## Figures and provenance",
        "",
        "`bridge_protocols.png` and `bridge_protocols.pdf` show the three protocol",
        "categories as separate point estimates with CIs. No line is used to imply a",
        "continuous causal path. Marker shape redundantly encodes protocol. Chance",
        "baselines are task-specific (B0=1/2 for binary, 1/3 for n-back, 1/4 for Visual);",
        "they are drawn as short local dashed segments rather than one global line. Visual",
        "within-subject is left as an explicit gap and marked ×/NA.",
        "",
        "Input paths and SHA-256 fingerprints are recorded in `analysis_manifest.json`.",
        f"Figure outputs: {', '.join(f'`{Path(path).name}`' for path in figure_paths)}. "
        "Alt text is in `bridge_protocols_alt_text.md`.",
        "",
        "## Interpretation guardrail",
        "",
        "A rise from strict cross-subject to within-subject would support protocol",
        "sensitivity, but would not isolate subject identity leakage, sample budget,",
        "window/session grouping, or method adaptation as the sole cause. Those claims",
        "require the dedicated module and intervention experiments in the parent",
        "analysis plan.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_alt_text(path: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a compact text alternative for the two-panel static figure."""

    lookup = {(row["task"], row["protocol"]): row for row in summary_rows}
    lines = [
        "# Alt text — STA-Net protocol bridge",
        "",
        "Two-panel point-and-error-bar plot. The y values are subject-level means and",
        "the error bars are 95% percentile bootstrap intervals over subjects. Marker",
        "shapes distinguish trial-random (circle), group-safe within-subject (square),",
        "and strict cross-subject (triangle). No connecting lines imply a causal path.",
        "Short dashed horizontal segments show task-specific chance: 0.5 for binary",
        "tasks, 1/3 for n-back, and 1/4 for Visual.",
        "",
        "Classification panel (macro-F1), values in protocol order trial-random /",
        "group-safe within-subject / strict cross-subject:",
    ]
    for task in CLASSIFICATION_TASKS:
        values = []
        for protocol in PROTOCOL_ORDER:
            row = lookup[(task, protocol)]
            if protocol == "group_safe_within_subject" and not row["three_protocol_bridge_eligible"]:
                values.append("NA (existing within run has 11/16 subjects)")
            else:
                values.append(f"{float(row['estimate_subject_mean']):.4f}")
        lines.append(f"- {TASK_LABELS[task]} (B0={CHANCE_LEVELS[task]:.4f}): " + " / ".join(values))
    refed = [
        f"{float(lookup[('refed_regression', protocol)]['estimate_subject_mean']):.4f}"
        for protocol in PROTOCOL_ORDER
    ]
    lines.extend([
        "",
        "Regression panel (subject-level CCC), in the same protocol order:",
        f"- REFED regression (CCC=0 baseline): " + " / ".join(refed),
        "",
        "The Visual within-subject marker is explicitly absent and replaced by an x/NA",
        "annotation; its value is retained only in the machine-readable context table.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    *,
    fivefold_root: Path,
    within_root: Path,
    personalized_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    result = build_bridge(fivefold_root, within_root, personalized_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "bridge_subject_metrics.csv", result["subject_rows"])
    write_csv(output_root / "bridge_protocol_summary.csv", result["summary_rows"])
    write_csv(output_root / "bridge_eligibility.csv", result["eligibility_rows"])
    write_csv(output_root / "context_only_metrics.csv", result["context_rows"])
    write_csv(output_root / "missing_and_noncomparable.csv", result["missing_rows"])
    figure_paths = _plot_bridge(result["summary_rows"], output_root / "bridge_protocols")
    write_alt_text(output_root / "bridge_protocols_alt_text.md", result["summary_rows"])
    inputs = _input_files(fivefold_root, within_root, personalized_root)
    input_manifest = [
        {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in inputs
    ]
    output_names = (
        "bridge_subject_metrics.csv",
        "bridge_protocol_summary.csv",
        "bridge_eligibility.csv",
        "context_only_metrics.csv",
        "missing_and_noncomparable.csv",
        "bridge_protocols.png",
        "bridge_protocols.pdf",
        "bridge_protocols_alt_text.md",
        "bridge_protocol_summary.json",
        "analysis_manifest.json",
        "REPORT.md",
    )
    manifest = {
        "schema": "sta_net_protocol_bridge_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_date": "2026-08-16",
        "analysis_role": "read_only_post_hoc_protocol_bridge_audit",
        "protected_predictions_retrained": False,
        "protected_predictions_modified": False,
        "primary_metrics": PRIMARY_METRIC,
        "aggregation_unit": "subject_level_mean",
        "bootstrap": {"unit": "subject", "draws": 10000, "seed": 20260816, "interval": "percentile_95"},
        "inputs": {
            "fivefold_root": str(fivefold_root.resolve()),
            "within_subject_root": str(within_root.resolve()),
            "personalized_root": str(personalized_root.resolve()),
        },
        "input_files": input_manifest,
        "git_revision": _git_revision(),
        "figure_paths": figure_paths,
        "output_files": [str((output_root / name).resolve()) for name in output_names],
        "inventories": result["inventories"],
        "personalized_inventory": result["personalized_inventory"],
    }
    write_json(output_root / "bridge_protocol_summary.json", {
        "schema": "sta_net_protocol_bridge_summary_v1",
        "analysis_manifest": str((output_root / "analysis_manifest.json").resolve()),
        "eligibility": result["eligibility_rows"],
        "summary": result["summary_rows"],
        "noncomparable": result["missing_rows"],
    })
    write_report(
        output_root, result,
        fivefold_root=fivefold_root, within_root=within_root,
        personalized_root=personalized_root, figure_paths=figure_paths,
    )
    # Write the manifest last so its output inventory includes every deliverable.
    write_json(output_root / "analysis_manifest.json", manifest)
    return {
        "status": "completed",
        "output_root": str(output_root.resolve()),
        "summary_rows": len(result["summary_rows"]),
        "subject_rows": len(result["subject_rows"]),
        "trio_eligible_tasks": [
            row["task"] for row in result["eligibility_rows"] if row["three_protocol_bridge_eligible"]
        ],
        "pair_only_tasks": [
            row["task"] for row in result["eligibility_rows"]
            if row["trial_random_cross_subject_pair_eligible"] and not row["three_protocol_bridge_eligible"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fivefold-root", type=Path, default=DEFAULT_FIVEFOLD_ROOT)
    parser.add_argument("--within-subject-root", type=Path, default=DEFAULT_WITHIN_ROOT)
    parser.add_argument("--personalized-root", type=Path, default=DEFAULT_PERSONALIZED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = run_analysis(
        fivefold_root=args.fivefold_root.resolve(),
        within_root=args.within_subject_root.resolve(),
        personalized_root=args.personalized_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
