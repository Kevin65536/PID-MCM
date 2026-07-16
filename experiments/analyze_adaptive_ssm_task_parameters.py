#!/usr/bin/env python3
"""Audit whether adaptive SSM fits differ across dataset-native task labels.

The statistical unit is one subject-task fit.  All task conditions use the
same number of events.  The primary ``fixed_pooled`` path keeps the fNIRS
anchor, neighbouring EEG channels, normalization, and EEG PCA projection fixed
within subject and dataset.  The ``task_specific`` path reproduces the deployed
selection behaviour and is treated as a sensitivity analysis.

This is an exploratory parameter audit.  It does not establish that fitted
parameters are identifiable physiological constants or that task labels cause
the differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import rankdata, wilcoxon
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_adaptive_shared_neural_ssm import (
    _apply_eeg_adapter,
    _chromophore_targets,
    _fit_eeg_adapter,
    _fit_model,
    _local_eeg_indices,
    _paired_hbr_indices,
)
from experiments.evaluate_shared_neural_driver_unified import Trial, _select_active_hbo
from src.data.unified_physiology import UnifiedPhysiologyWindowDataset
from src.inference.adaptive_neurovascular_ssm import fit_to_mapping


SCHEMA = "adaptive_ssm_task_parameter_audit_v1"
PRIMARY_PARAMETERS = ("epsilon", "kas", "kaf", "tau0", "alpha", "e0", "phi", "q_driver")
NUISANCE_PARAMETERS = (
    "q_scale", "fnirs_noise_scale", "hbo_gain", "hbr_gain", "eeg_noise",
    "hbo_noise_base", "hbr_noise_base",
)
PARAMETER_FAMILIES = {
    **{parameter: "dynamics_driver" for parameter in PRIMARY_PARAMETERS},
    **{parameter: "observation_nuisance" for parameter in NUISANCE_PARAMETERS},
}
PARAMETER_BOUNDS = {
    "epsilon": (0.0, 1.0),
    "kas": (0.25, 1.50),
    "kaf": (0.05, 0.90),
    "tau0": (0.60, 5.00),
    "alpha": (0.18, 0.55),
    "e0": (0.20, 0.65),
    "phi": (0.45, 0.995),
    "q_driver": (1e-4, 4.0),
    "q_scale": (0.5, 2.0),
    "fnirs_noise_scale": (0.25, 4.0),
    "hbo_noise_base": (0.20, 4.0),
    "hbr_noise_base": (0.20, 4.0),
}


def adjust_pvalues(p_values: Sequence[float], method: str) -> np.ndarray:
    """Return monotone BH-FDR or Holm adjusted p-values without extra deps."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("p_values must be a finite, non-empty vector")
    order = np.argsort(values)
    ranked = values[order]
    count = len(values)
    if method == "fdr_bh":
        scaled = ranked * count / np.arange(1, count + 1, dtype=np.float64)
        adjusted_ranked = np.minimum.accumulate(scaled[::-1])[::-1]
    elif method == "holm":
        scaled = ranked * np.arange(count, 0, -1, dtype=np.float64)
        adjusted_ranked = np.maximum.accumulate(scaled)
    else:
        raise ValueError(f"unsupported adjustment method: {method}")
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_payload() -> dict[str, str]:
    def call(*args: str) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, check=False, capture_output=True, text=True).stdout.strip()

    return {"commit": call("git", "rev-parse", "HEAD"), "status_short": call("git", "status", "--short")}


def _resolve_subjects(config: Mapping[str, Any]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    required = int(config["data"]["trials_per_condition"])
    resolved: dict[str, list[str]] = {}
    coverage: list[dict[str, Any]] = []
    for dataset in config["data"]["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        unified = UnifiedPhysiologyWindowDataset(
            cache_root=config["data"]["cache_root"],
            dataset_ids=(dataset_id,),
            window_duration_s=float(config["data"]["window_duration_s"]),
            window_offset_s=float(config["data"]["window_offset_s"]),
            eeg_signal_branch=str(dataset["eeg_signal_branch"]),
        )
        counts: Counter[tuple[str, str, str]] = Counter(
            (
                str(ref.record.canonical_subject_id),
                str(ref.record.base_record_id),
                str(ref.event.get("label")),
            )
            for ref in unified.windows
        )
        candidates = sorted({key[0] for key in counts})
        complete: list[str] = []
        for subject in candidates:
            condition_counts = [
                counts[(subject, str(condition["record_id"]), str(condition["event_label"]))]
                for condition in dataset["conditions"]
            ]
            admitted = bool(condition_counts) and min(condition_counts) >= required
            coverage.append({
                "dataset_id": dataset_id,
                "subject": subject,
                "complete": admitted,
                "minimum_condition_events": min(condition_counts) if condition_counts else 0,
                "admitted_condition_event_counts": "|".join(str(value) for value in condition_counts),
            })
            if admitted:
                complete.append(subject)
        if not complete:
            raise RuntimeError(f"{dataset_id}: no subjects have complete task coverage")
        resolved[dataset_id] = complete
    return resolved, coverage


def _trial_from_sample(
    sample: Mapping[str, Any],
    *,
    condition_id: str,
    dataset_id: str,
    subject: str,
    baseline_n: int,
) -> Trial:
    fnirs = np.asarray(sample["fnirs"], dtype=np.float64).T
    fnirs -= np.mean(fnirs[:baseline_n], axis=0, keepdims=True)
    artifact_mask = np.asarray(sample["artifact_mask"]["eeg"], dtype=bool)
    return Trial(
        condition_id=condition_id,
        dataset_id=dataset_id,
        subject=subject,
        record_id=str(sample["record_id"]),
        event_index=int(sample["event"].get("event_index", 0)),
        eeg=np.asarray(sample["eeg"], dtype=np.float64).T,
        fnirs=fnirs,
        fnirs_channel_names=tuple(str(value) for value in sample["channel_names"]["fnirs"]),
        fnirs_roles=tuple(str(value) for value in sample["component_roles"]["fnirs"]),
        eeg_artifact_fraction=float(np.mean(artifact_mask)),
        eeg_channel_names=tuple(str(value) for value in sample["channel_names"]["eeg"]),
        eeg_positions=np.asarray([
            [row.get(axis, np.nan) for axis in ("x", "y", "z")]
            for row in sample["channel_geometry"]["eeg"]
        ], dtype=np.float64),
        fnirs_positions=np.asarray([
            [row.get(axis, np.nan) for axis in ("x", "y", "z")]
            for row in sample["channel_geometry"]["fnirs"]
        ], dtype=np.float64),
    )


def _load_subject_trials(
    dataset_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    subject: str,
) -> dict[str, list[Trial]]:
    dataset_id = str(dataset_config["dataset_id"])
    dataset = UnifiedPhysiologyWindowDataset(
        cache_root=data_config["cache_root"],
        dataset_ids=(dataset_id,),
        window_duration_s=float(data_config["window_duration_s"]),
        window_offset_s=float(data_config["window_offset_s"]),
        eeg_signal_branch=str(dataset_config["eeg_signal_branch"]),
    )
    baseline_n = int(round(float(data_config["baseline_duration_s"]) * 10.0))
    count = int(data_config["trials_per_condition"])
    output: dict[str, list[Trial]] = {}
    for condition in dataset_config["conditions"]:
        task_id = str(condition["task_id"])
        indices = [
            index for index, ref in enumerate(dataset.windows)
            if ref.record.canonical_subject_id == subject
            and ref.record.base_record_id == str(condition["record_id"])
            and str(ref.event.get("label")) == str(condition["event_label"])
        ]
        indices.sort(key=lambda index: int(dataset.windows[index].event.get("event_index", index)))
        if len(indices) < count:
            raise RuntimeError(f"{dataset_id}:{subject}:{task_id} has {len(indices)} events, expected {count}")
        output[task_id] = [
            _trial_from_sample(
                dataset[index],
                condition_id=task_id,
                dataset_id=dataset_id,
                subject=subject,
                baseline_n=baseline_n,
            )
            for index in indices[:count]
        ]
    return output


def _fit_task(
    trials: Sequence[Trial],
    *,
    hbo_indices: np.ndarray,
    eeg_indices: np.ndarray,
    ssm_config: Mapping[str, Any],
    baseline_samples: int,
    fixed_adapter: Any | None = None,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    hbr_indices = _paired_hbr_indices(trials[0], hbo_indices)
    hbo, hbr = _chromophore_targets(trials, hbo_indices, hbr_indices)
    if fixed_adapter is None:
        adapter, drivers = _fit_eeg_adapter(trials, eeg_indices)
    else:
        adapter = fixed_adapter
        drivers = [_apply_eeg_adapter(trial, adapter) for trial in trials]
    fit = _fit_model(drivers, hbo, hbr, ssm_config, baseline_samples)
    fnirs_names = tuple(trials[0].fnirs_channel_names[int(index)] for index in np.r_[hbo_indices, hbr_indices])
    return dict(fit_to_mapping(fit)), fnirs_names, tuple(adapter.channel_names)


def _fit_subject_job(
    dataset_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    analysis_config: Mapping[str, Any],
    subject: str,
) -> list[dict[str, Any]]:
    with threadpool_limits(limits=1):
        trials_by_task = _load_subject_trials(dataset_config, data_config, subject)
        condition_by_id = {str(value["task_id"]): value for value in dataset_config["conditions"]}
        pooled = [trial for condition in dataset_config["conditions"] for trial in trials_by_task[str(condition["task_id"])]]
        baseline_samples = int(round(
            float(data_config["baseline_duration_s"]) * float(analysis_config["ssm"]["fs_hz"])
        ))
        rows: list[dict[str, Any]] = []
        fixed_hbo, _, _ = _select_active_hbo(
            pooled,
            baseline_duration_s=float(data_config["baseline_duration_s"]),
            task_duration_s=float(data_config["task_duration_s"]),
            count=int(analysis_config["fnirs_active_hbo_channels"]),
        )
        fixed_eeg = _local_eeg_indices(pooled[0], fixed_hbo, int(analysis_config["local_eeg_channels"]))
        fixed_adapter, _ = _fit_eeg_adapter(pooled, fixed_eeg)
        for condition in dataset_config["conditions"]:
            task_id = str(condition["task_id"])
            trials = trials_by_task[task_id]
            for anchor_mode in analysis_config["anchor_modes"]:
                if anchor_mode == "fixed_pooled":
                    hbo_indices = fixed_hbo
                    eeg_indices = fixed_eeg
                    adapter = fixed_adapter
                elif anchor_mode == "task_specific":
                    hbo_indices, _, _ = _select_active_hbo(
                        trials,
                        baseline_duration_s=float(data_config["baseline_duration_s"]),
                        task_duration_s=float(data_config["task_duration_s"]),
                        count=int(analysis_config["fnirs_active_hbo_channels"]),
                    )
                    eeg_indices = _local_eeg_indices(
                        trials[0], hbo_indices, int(analysis_config["local_eeg_channels"]),
                    )
                    adapter = None
                else:
                    raise ValueError(f"unsupported anchor mode: {anchor_mode}")
                fitted, fnirs_names, eeg_names = _fit_task(
                    trials,
                    hbo_indices=hbo_indices,
                    eeg_indices=eeg_indices,
                    ssm_config=analysis_config["ssm"],
                    baseline_samples=baseline_samples,
                    fixed_adapter=adapter,
                )
                rows.append({
                    "dataset_id": str(dataset_config["dataset_id"]),
                    "subject": subject,
                    "task_id": task_id,
                    "task_family": str(condition["task_family"]),
                    "record_id": str(condition["record_id"]),
                    "event_label": str(condition["event_label"]),
                    "anchor_mode": str(anchor_mode),
                    "n_trials": len(trials),
                    "selected_fnirs_channels": "|".join(fnirs_names),
                    "selected_eeg_channels": "|".join(eeg_names),
                    **fitted,
                })
        return rows


def friedman_permutation_test(
    matrix: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    """Friedman statistic, Monte-Carlo permutation p, and Kendall's W."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 3:
        raise ValueError("matrix must contain at least two subjects and three repeated conditions")
    n_subjects, n_tasks = values.shape
    ranks = rankdata(values, axis=1, method="average")
    tie_sum = 0.0
    for row in values:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += float(np.sum(counts**3 - counts))
    correction = 1.0 - tie_sum / float(n_subjects * (n_tasks**3 - n_tasks))
    if correction <= 1e-12:
        return 0.0, 1.0, 0.0

    def statistic(rank_values: np.ndarray) -> np.ndarray:
        sums = np.sum(rank_values, axis=-2)
        raw = 12.0 * np.sum(sums**2, axis=-1) / float(n_subjects * n_tasks * (n_tasks + 1))
        return (raw - 3.0 * n_subjects * (n_tasks + 1)) / correction

    observed = float(statistic(ranks))
    if observed <= 1e-12:
        return max(observed, 0.0), 1.0, 0.0
    rng = np.random.default_rng(int(seed))
    exceedances = 0
    generated = 0
    batch_size = min(2000, int(iterations))
    while generated < int(iterations):
        batch = min(batch_size, int(iterations) - generated)
        order = np.argsort(rng.random((batch, n_subjects, n_tasks)), axis=2)
        permuted = np.take_along_axis(ranks[None, :, :], order, axis=2)
        exceedances += int(np.sum(statistic(permuted) >= observed - 1e-12))
        generated += batch
    p_value = (exceedances + 1.0) / (int(iterations) + 1.0)
    kendall_w = float(np.clip(observed / (n_subjects * (n_tasks - 1)), 0.0, 1.0))
    return observed, float(p_value), kendall_w


def _subject_task_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    anchor_mode: str,
    task_ids: Sequence[str],
    parameter: str,
) -> tuple[list[str], np.ndarray]:
    selected = [
        row for row in rows
        if row["dataset_id"] == dataset_id and row["anchor_mode"] == anchor_mode
    ]
    lookup = {(str(row["subject"]), str(row["task_id"])): float(row[parameter]) for row in selected}
    subjects = sorted({str(row["subject"]) for row in selected})
    matrix = np.asarray([[lookup[(subject, task)] for task in task_ids] for subject in subjects], dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"non-finite or incomplete matrix for {dataset_id}/{anchor_mode}/{parameter}")
    return subjects, matrix


def _descriptive_rows(
    fitted_rows: Sequence[Mapping[str, Any]],
    dataset_task_order: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    output = []
    for dataset_id, tasks in dataset_task_order.items():
        for anchor_mode in ("fixed_pooled", "task_specific"):
            for parameter in PARAMETER_FAMILIES:
                _, matrix = _subject_task_matrix(
                    fitted_rows,
                    dataset_id=dataset_id,
                    anchor_mode=anchor_mode,
                    task_ids=tasks,
                    parameter=parameter,
                )
                bounds = PARAMETER_BOUNDS.get(parameter)
                for task_index, task_id in enumerate(tasks):
                    values = matrix[:, task_index]
                    boundary = np.zeros(len(values), dtype=bool)
                    if bounds is not None:
                        tolerance = max((bounds[1] - bounds[0]) * 1e-4, 1e-8)
                        boundary = (np.abs(values - bounds[0]) <= tolerance) | (np.abs(values - bounds[1]) <= tolerance)
                    output.append({
                        "dataset_id": dataset_id,
                        "anchor_mode": anchor_mode,
                        "parameter_family": PARAMETER_FAMILIES[parameter],
                        "parameter": parameter,
                        "task_id": task_id,
                        "subjects": len(values),
                        "mean": float(np.mean(values)),
                        "sd": float(np.std(values, ddof=1)),
                        "median": float(np.median(values)),
                        "q1": float(np.quantile(values, 0.25)),
                        "q3": float(np.quantile(values, 0.75)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                        "boundary_fraction": float(np.mean(boundary)),
                    })
    return output


def _omnibus_rows(
    fitted_rows: Sequence[Mapping[str, Any]],
    dataset_task_order: Mapping[str, Sequence[str]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    test_index = 0
    for dataset_id, tasks in dataset_task_order.items():
        for anchor_mode in ("fixed_pooled", "task_specific"):
            for parameter in PARAMETER_FAMILIES:
                subjects, matrix = _subject_task_matrix(
                    fitted_rows,
                    dataset_id=dataset_id,
                    anchor_mode=anchor_mode,
                    task_ids=tasks,
                    parameter=parameter,
                )
                statistic, p_value, kendall_w = friedman_permutation_test(
                    matrix,
                    iterations=iterations,
                    seed=seed + test_index * 7919,
                )
                output.append({
                    "dataset_id": dataset_id,
                    "anchor_mode": anchor_mode,
                    "parameter_family": PARAMETER_FAMILIES[parameter],
                    "parameter": parameter,
                    "subjects": len(subjects),
                    "tasks": len(tasks),
                    "task_order": "|".join(tasks),
                    "friedman_statistic": statistic,
                    "permutation_iterations": iterations,
                    "p_value": p_value,
                    "kendall_w": kendall_w,
                })
                test_index += 1
    for anchor_mode in ("fixed_pooled", "task_specific"):
        for family in ("dynamics_driver", "observation_nuisance"):
            indices = [
                index for index, row in enumerate(output)
                if row["anchor_mode"] == anchor_mode and row["parameter_family"] == family
            ]
            adjusted = adjust_pvalues([float(output[index]["p_value"]) for index in indices], "fdr_bh")
            for index, q_value in zip(indices, adjusted):
                output[index]["fdr_scope"] = f"{anchor_mode}:{family}:across_datasets_and_parameters"
                output[index]["q_value_bh"] = float(q_value)
                output[index]["significant_fdr_0_05"] = bool(q_value < 0.05)
    return output


def _rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    nonzero = values[np.abs(values) > 1e-12]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    denominator = float(np.sum(ranks))
    return float((np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0])) / denominator)


def _pairwise_rows(
    fitted_rows: Sequence[Mapping[str, Any]],
    dataset_task_order: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    output = []
    for dataset_id, tasks in dataset_task_order.items():
        for anchor_mode in ("fixed_pooled", "task_specific"):
            for parameter in PARAMETER_FAMILIES:
                subjects, matrix = _subject_task_matrix(
                    fitted_rows,
                    dataset_id=dataset_id,
                    anchor_mode=anchor_mode,
                    task_ids=tasks,
                    parameter=parameter,
                )
                family_rows = []
                for left in range(len(tasks)):
                    for right in range(left + 1, len(tasks)):
                        differences = matrix[:, left] - matrix[:, right]
                        if np.all(np.abs(differences) <= 1e-12):
                            p_value = 1.0
                        else:
                            p_value = float(wilcoxon(
                                differences,
                                zero_method="wilcox",
                                correction=False,
                                alternative="two-sided",
                                method="auto",
                            ).pvalue)
                        family_rows.append({
                            "dataset_id": dataset_id,
                            "anchor_mode": anchor_mode,
                            "parameter_family": PARAMETER_FAMILIES[parameter],
                            "parameter": parameter,
                            "task_a": tasks[left],
                            "task_b": tasks[right],
                            "subjects": len(subjects),
                            "median_paired_difference_a_minus_b": float(np.median(differences)),
                            "rank_biserial_correlation": _rank_biserial(differences),
                            "p_value": p_value,
                        })
                adjusted = adjust_pvalues([row["p_value"] for row in family_rows], "holm")
                for row, p_holm in zip(family_rows, adjusted):
                    row["p_value_holm_within_parameter"] = float(p_holm)
                    row["significant_holm_0_05"] = bool(p_holm < 0.05)
                    output.append(row)
    return output


def _anchor_consistency_rows(fitted_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fitted_rows:
        if row["anchor_mode"] == "task_specific":
            grouped[(str(row["dataset_id"]), str(row["subject"]))].append(row)
    return [{
        "dataset_id": dataset_id,
        "subject": subject,
        "tasks": len(rows),
        "unique_fnirs_anchor_pairs": len({str(row["selected_fnirs_channels"]) for row in rows}),
        "same_anchor_all_tasks": len({str(row["selected_fnirs_channels"]) for row in rows}) == 1,
    } for (dataset_id, subject), rows in sorted(grouped.items())]


def _plot_effects(omnibus: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    parameters = list(PRIMARY_PARAMETERS + NUISANCE_PARAMETERS)
    datasets = ("eeg_fnirs_single_trial", "simultaneous_eeg_nirs")
    modes = ("fixed_pooled", "task_specific")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0), sharex=True, sharey=True)
    lookup = {(row["dataset_id"], row["anchor_mode"], row["parameter"]): row for row in omnibus}
    y = np.arange(len(parameters))
    for row_index, dataset_id in enumerate(datasets):
        for column_index, anchor_mode in enumerate(modes):
            axis = axes[row_index, column_index]
            values = [lookup[(dataset_id, anchor_mode, parameter)] for parameter in parameters]
            effects = np.asarray([float(value["kendall_w"]) for value in values])
            significant = np.asarray([bool(value["significant_fdr_0_05"]) for value in values])
            colors = np.where(significant, "#d55e00", "#0072b2")
            axis.scatter(effects, y, c=colors, s=np.where(significant, 75, 42), zorder=3)
            for yi, effect, is_significant in zip(y, effects, significant):
                axis.plot([0.0, effect], [yi, yi], color="#cccccc", linewidth=1.0, zorder=1)
                if is_significant:
                    axis.text(min(effect + 0.018, 0.98), yi, "FDR<.05", va="center", fontsize=7.5, color="#b54200")
            axis.axhline(len(PRIMARY_PARAMETERS) - 0.5, color="#666666", linestyle="--", linewidth=0.9)
            axis.set_xlim(-0.01, 1.0)
            axis.grid(axis="x", color="#e5e5e5", linewidth=0.7)
            axis.set_title(f"{dataset_id}\n{anchor_mode}", fontsize=10)
            axis.set_yticks(y, parameters)
            axis.invert_yaxis()
            if row_index == 1:
                axis.set_xlabel("Kendall's W task effect")
    fig.suptitle("Adaptive SSM parameter differences across task labels\norange = BH-FDR q < .05 within parameter family", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _summary_markdown(
    fitted: Sequence[Mapping[str, Any]],
    omnibus: Sequence[Mapping[str, Any]],
    anchor_consistency: Sequence[Mapping[str, Any]],
    resolved_subjects: Mapping[str, Sequence[str]],
    dataset_task_order: Mapping[str, Sequence[str]],
) -> str:
    primary = [row for row in omnibus if row["anchor_mode"] == "fixed_pooled"]
    significant = sorted(
        [row for row in primary if _as_bool(row["significant_fdr_0_05"])],
        key=lambda row: float(row["q_value_bh"]),
    )
    top = sorted(primary, key=lambda row: float(row["p_value"]))[:10]
    optimizer_success = float(np.mean([_as_bool(row["optimizer_success"]) for row in fitted]))
    anchor_same = defaultdict(list)
    for row in anchor_consistency:
        anchor_same[str(row["dataset_id"])].append(_as_bool(row["same_anchor_all_tasks"]))
    lines = [
        "# Adaptive SSM task-parameter audit",
        "",
        "## Scope and statistical unit",
        "",
        "Each row entering inference is one full-data fit for one subject and one dataset-native task condition. "
        "The primary fixed-pooled analysis holds the fNIRS anchor, six local EEG channels, normalization, and EEG PCA loading fixed across tasks within each subject. The task-specific path is a sensitivity analysis.",
        "",
        "| dataset | complete subjects | task conditions |",
        "|---|---:|---|",
    ]
    for dataset_id, subjects in resolved_subjects.items():
        lines.append(f"| {dataset_id} | {len(subjects)} | {', '.join(dataset_task_order[dataset_id])} |")
    lines.extend([
        "",
        "## Primary result",
        "",
        f"Optimizer success: {optimizer_success:.1%} across {len(fitted)} subject-task-anchor fits.",
        "",
    ])
    if significant:
        lines.extend([
            f"The fixed-representation primary analysis found {len(significant)} parameter-level task effects after within-family BH-FDR control:",
            "",
            "| dataset | family | parameter | W | permutation p | BH q |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in significant:
            lines.append(
                f"| {row['dataset_id']} | {row['parameter_family']} | {row['parameter']} | "
                f"{float(row['kendall_w']):.3f} | {float(row['p_value']):.5f} | {float(row['q_value_bh']):.5f} |"
            )
    else:
        lines.append("No fixed-representation parameter survived BH-FDR at q < .05.")
    lines.extend([
        "",
        "## Strongest fixed-representation omnibus effects",
        "",
        "| dataset | family | parameter | W | permutation p | BH q |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in top:
        lines.append(
            f"| {row['dataset_id']} | {row['parameter_family']} | {row['parameter']} | "
            f"{float(row['kendall_w']):.3f} | {float(row['p_value']):.5f} | {float(row['q_value_bh']):.5f} |"
        )
    lines.extend([
        "",
        "## Spatial-selection sensitivity",
        "",
    ])
    for dataset_id, values in anchor_same.items():
        lines.append(
            f"- {dataset_id}: the task-specific selector retained exactly the same fNIRS anchor for all tasks in "
            f"{np.mean(values):.1%} of subjects."
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- The test is exploratory and label-aware; it is not a preregistered confirmatory analysis.",
        "- No parameter is compared numerically across datasets. Dataset, paradigm, measurement family, and subject cohort are otherwise confounded.",
        "- Frequent boundary solutions and the epsilon/gain scale gauge limit physiological identifiability. A task effect in a fitted parameter is evidence about this model's task-conditioned optimum, not proof that the biological constant changes with task.",
        "- N-back and DSR labels are session-block onsets, whereas WG and Single-Trial labels are trial events. Their omnibus comparison includes event-structure differences.",
        "- Pairwise Wilcoxon tests and Holm adjustments are stored for localization; they are secondary to the repeated-measures omnibus tests.",
        "",
    ])
    return "\n".join(lines)


def run(config_path: Path, run_dir: Path, resource_path: Path | None = None) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resolved_subjects, coverage = _resolve_subjects(config)
    _write_csv(run_dir / "subject_coverage.csv", coverage)

    dataset_by_id = {str(value["dataset_id"]): value for value in config["data"]["datasets"]}
    jobs = [
        (dataset_by_id[dataset_id], config["data"], config["analysis"], subject)
        for dataset_id, subjects in resolved_subjects.items()
        for subject in subjects
    ]
    fitted: list[dict[str, Any]] = []
    workers = min(int(config["analysis"]["workers"]), len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fit_subject_job, *job): (job[0]["dataset_id"], job[3]) for job in jobs}
        for future in as_completed(futures):
            dataset_id, subject = futures[future]
            try:
                fitted.extend(future.result())
            except Exception as error:
                raise RuntimeError(f"fit failed for {dataset_id}:{subject}") from error
    fitted.sort(key=lambda row: (row["dataset_id"], row["subject"], row["task_id"], row["anchor_mode"]))
    _write_csv(run_dir / "subject_task_parameters.csv", fitted)

    dataset_task_order = {
        str(dataset["dataset_id"]): [str(condition["task_id"]) for condition in dataset["conditions"]]
        for dataset in config["data"]["datasets"]
    }
    descriptive = _descriptive_rows(fitted, dataset_task_order)
    omnibus = _omnibus_rows(
        fitted,
        dataset_task_order,
        iterations=int(config["analysis"]["permutation_iterations"]),
        seed=int(config["analysis"]["seed"]),
    )
    pairwise = _pairwise_rows(fitted, dataset_task_order)
    anchor_consistency = _anchor_consistency_rows(fitted)
    _write_csv(run_dir / "descriptive_statistics.csv", descriptive)
    _write_csv(run_dir / "omnibus_tests.csv", omnibus)
    _write_csv(run_dir / "pairwise_tests.csv", pairwise)
    _write_csv(run_dir / "anchor_consistency.csv", anchor_consistency)
    figure_path = figures_dir / "task_parameter_effects.svg"
    _plot_effects(omnibus, figure_path)
    (run_dir / "summary.md").write_text(
        _summary_markdown(fitted, omnibus, anchor_consistency, resolved_subjects, dataset_task_order),
        encoding="utf-8",
    )

    resources = None
    if resource_path is not None and resource_path.exists():
        resources = json.loads(resource_path.read_text(encoding="utf-8"))
    output_paths = [
        run_dir / "config.yaml",
        run_dir / "subject_coverage.csv",
        run_dir / "subject_task_parameters.csv",
        run_dir / "descriptive_statistics.csv",
        run_dir / "omnibus_tests.csv",
        run_dir / "pairwise_tests.csv",
        run_dir / "anchor_consistency.csv",
        run_dir / "summary.md",
        figure_path,
    ]
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "git": _git_payload(),
        "config_source": str(config_path),
        "resolved_subjects": resolved_subjects,
        "resources": resources,
        "outputs": [
            {"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)} for path in output_paths
        ],
        "inference": {
            "statistical_unit": "one full-data subject-task fit",
            "primary_anchor_mode": "fixed_pooled",
            "omnibus": "Friedman rank statistic with within-subject task-label permutations",
            "effect_size": "Kendall's W",
            "multiple_comparisons": "BH-FDR within anchor mode and parameter family across datasets and parameters",
            "pairwise": "paired Wilcoxon signed-rank with Holm adjustment within dataset-anchor-parameter",
        },
    }
    _write_json(run_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/adaptive_ssm_task_parameter_audit.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resource-json", type=Path)
    args = parser.parse_args()
    run(args.config, args.run_dir, args.resource_json)


if __name__ == "__main__":
    main()
