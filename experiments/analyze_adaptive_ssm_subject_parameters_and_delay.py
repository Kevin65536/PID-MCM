#!/usr/bin/env python3
"""Derive subject effects, descriptive statistics, and impulse delays for E0-D8.

The input is the completed E0-D8 subject-task parameter table.  No model is
refitted.  The primary analysis uses only the ``fixed_pooled`` representation,
which holds the spatial anchor and EEG projection fixed across tasks within a
subject.

Subject effects reverse the existing task-effect Friedman design: tasks are
blocks and subject identities are the repeated conditions.  The Monte-Carlo
null independently permutes subject labels within every task.  A significant
result therefore means that some subjects are stably high or low across tasks;
it does not establish that a fitted optimum is an identified physiological
constant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_adaptive_ssm_task_parameters import (  # noqa: E402
    NUISANCE_PARAMETERS,
    PARAMETER_FAMILIES,
    PRIMARY_PARAMETERS,
    adjust_pvalues,
    friedman_permutation_test,
)
from src.inference.adaptive_neurovascular_ssm import (  # noqa: E402
    HemodynamicParameters,
    simulate_hemodynamics,
)


SCHEMA = "adaptive_ssm_subject_parameter_delay_audit_v1"
DEFAULT_SOURCE_RUN = REPO_ROOT / (
    "experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/"
    "20260716_adaptive_ssm_task_parameter_audit_v1"
)

# Croce et al. (2017), DOI 10.1088/1741-2552/aa7321, equation (4a) and table 1:
# ds/dt = epsilon*r - k_as*s - k_af*(f-1).  Note that the current adaptive
# implementation swaps the paper's 0.41/0.65 assignments and changes E0 from
# 0.34 to 0.40 in its regularization center.
CROCE_2017_PRESET = {
    "epsilon": 1.0,
    "kas": 0.41,
    "kaf": 0.65,
    "tau0": 2.0,
    "alpha": 0.32,
    "e0": 0.34,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_payload() -> dict[str, str]:
    def call(*args: str) -> str:
        return subprocess.run(
            args, cwd=REPO_ROOT, check=False, capture_output=True, text=True,
        ).stdout.strip()

    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "status_short": call("git", "status", "--short"),
    }


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _fixed_rows(source_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    output = [dict(row) for row in source_rows if row["anchor_mode"] == "fixed_pooled"]
    if not output:
        raise RuntimeError("source table has no fixed_pooled rows")
    if not all(_as_bool(row["optimizer_success"]) for row in output):
        raise RuntimeError("source table contains unsuccessful fixed_pooled fits")
    return output


def _dataset_task_orders(rows: Sequence[Mapping[str, str]]) -> dict[str, list[str]]:
    preferred = {
        "eeg_fnirs_single_trial": ["MA", "BL_MA", "LMI", "RMI"],
        "simultaneous_eeg_nirs": ["WG", "BL_WG", "0BACK", "2BACK", "3BACK", "DSR"],
    }
    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        observed[row["dataset_id"]].add(row["task_id"])
    output: dict[str, list[str]] = {}
    for dataset_id, tasks in observed.items():
        ordered = [task for task in preferred.get(dataset_id, []) if task in tasks]
        ordered.extend(sorted(tasks - set(ordered)))
        output[dataset_id] = ordered
    return output


def _subject_task_matrix(
    rows: Sequence[Mapping[str, str]],
    *,
    dataset_id: str,
    task_order: Sequence[str],
    parameter: str,
) -> tuple[list[str], np.ndarray]:
    selected = [row for row in rows if row["dataset_id"] == dataset_id]
    subjects = sorted({row["subject"] for row in selected})
    lookup = {
        (row["subject"], row["task_id"]): float(row[parameter])
        for row in selected
    }
    matrix = np.asarray(
        [[lookup[(subject, task)] for task in task_order] for subject in subjects],
        dtype=np.float64,
    )
    if matrix.shape != (len(subjects), len(task_order)) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"incomplete matrix for {dataset_id}/{parameter}")
    return subjects, matrix


def _aggregate_statistics(
    rows: Sequence[Mapping[str, str]],
    task_orders: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    adaptive_center = HemodynamicParameters()
    output: list[dict[str, Any]] = []
    for dataset_id, tasks in task_orders.items():
        for parameter in PRIMARY_PARAMETERS + NUISANCE_PARAMETERS:
            subjects, matrix = _subject_task_matrix(
                rows,
                dataset_id=dataset_id,
                task_order=tasks,
                parameter=parameter,
            )
            values = matrix.reshape(-1)
            subject_means = np.mean(matrix, axis=1)
            min_subject_index = int(np.argmin(subject_means))
            max_subject_index = int(np.argmax(subject_means))
            paper_value = CROCE_2017_PRESET.get(parameter)
            center_value = getattr(adaptive_center, parameter, None)
            output.append({
                "dataset_id": dataset_id,
                "parameter_family": PARAMETER_FAMILIES[parameter],
                "parameter": parameter,
                "subjects": len(subjects),
                "tasks": len(tasks),
                "fits": len(values),
                "mean": float(np.mean(values)),
                "sample_variance": float(np.var(values, ddof=1)),
                "sample_sd": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "subject_mean_minimum": float(subject_means[min_subject_index]),
                "subject_mean_minimum_id": subjects[min_subject_index],
                "subject_mean_maximum": float(subject_means[max_subject_index]),
                "subject_mean_maximum_id": subjects[max_subject_index],
                "croce_2017_preset": "" if paper_value is None else float(paper_value),
                "mean_minus_croce_2017": "" if paper_value is None else float(np.mean(values) - paper_value),
                "adaptive_regularizer_center": "" if center_value is None else float(center_value),
                "mean_minus_adaptive_center": "" if center_value is None else float(np.mean(values) - center_value),
            })
    return output


def _task_statistics(
    rows: Sequence[Mapping[str, str]],
    task_orders: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset_id, tasks in task_orders.items():
        for parameter in PRIMARY_PARAMETERS + NUISANCE_PARAMETERS:
            _, matrix = _subject_task_matrix(
                rows,
                dataset_id=dataset_id,
                task_order=tasks,
                parameter=parameter,
            )
            for task_index, task_id in enumerate(tasks):
                values = matrix[:, task_index]
                output.append({
                    "dataset_id": dataset_id,
                    "task_id": task_id,
                    "parameter_family": PARAMETER_FAMILIES[parameter],
                    "parameter": parameter,
                    "subjects": len(values),
                    "mean": float(np.mean(values)),
                    "sample_variance": float(np.var(values, ddof=1)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                })
    return output


def _subject_effect_tests(
    rows: Sequence[Mapping[str, str]],
    task_orders: Mapping[str, Sequence[str]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    test_index = 0
    for dataset_id, tasks in task_orders.items():
        for parameter in PRIMARY_PARAMETERS + NUISANCE_PARAMETERS:
            subjects, matrix = _subject_task_matrix(
                rows,
                dataset_id=dataset_id,
                task_order=tasks,
                parameter=parameter,
            )
            statistic, p_value, kendall_w = friedman_permutation_test(
                matrix.T,
                iterations=iterations,
                seed=seed + test_index * 7919,
            )
            output.append({
                "dataset_id": dataset_id,
                "parameter_family": PARAMETER_FAMILIES[parameter],
                "parameter": parameter,
                "subjects": len(subjects),
                "tasks_as_blocks": len(tasks),
                "friedman_statistic": statistic,
                "permutation_iterations": iterations,
                "p_value": p_value,
                "kendall_w_subject_effect": kendall_w,
            })
            test_index += 1

    for family in ("dynamics_driver", "observation_nuisance"):
        indices = [
            index for index, row in enumerate(output)
            if row["parameter_family"] == family
        ]
        adjusted = adjust_pvalues(
            [float(output[index]["p_value"]) for index in indices], "fdr_bh",
        )
        for index, q_value in zip(indices, adjusted):
            output[index]["fdr_scope"] = f"{family}:across_datasets_and_parameters"
            output[index]["q_value_bh"] = float(q_value)
            output[index]["significant_fdr_0_05"] = bool(q_value < 0.05)
    return output


def _impulse_metrics(params: HemodynamicParameters, *, fs_hz: float, duration_s: float) -> dict[str, float]:
    sample_count = int(round(float(duration_s) * float(fs_hz)))
    driver = np.zeros(sample_count, dtype=np.float64)
    driver[0] = float(fs_hz)  # unit-area impulse; amplitude does not affect latency
    states = simulate_hemodynamics(driver, params, fs_hz=float(fs_hz))
    time_s = (np.arange(sample_count, dtype=np.float64) + 1.0) / float(fs_hz)

    def metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
        magnitude = np.abs(np.asarray(values, dtype=np.float64))
        peak_index = int(np.argmax(magnitude))
        peak = float(magnitude[peak_index])
        onset = np.flatnonzero(magnitude >= 0.10 * peak)
        return {
            f"{prefix}_onset_10pct_s": float(time_s[int(onset[0])]) if len(onset) else float("nan"),
            f"{prefix}_absolute_peak_s": float(time_s[peak_index]),
            f"{prefix}_signed_peak": float(values[peak_index]),
            f"{prefix}_absolute_center_of_mass_s": float(np.sum(time_s * magnitude) / np.sum(magnitude)),
        }

    output: dict[str, float] = {}
    output.update(metrics(states[:, 2], "hbo_state"))
    output.update(metrics(states[:, 3], "hbr_state"))
    vector_magnitude = np.linalg.norm(states[:, 2:4], axis=1)
    output.update(metrics(vector_magnitude, "joint_fnirs_state"))
    return output


def _delay_rows(
    aggregate: Sequence[Mapping[str, Any]],
    *,
    fs_hz: float,
    duration_s: float,
) -> list[dict[str, Any]]:
    by_dataset = defaultdict(dict)
    for row in aggregate:
        by_dataset[str(row["dataset_id"])][str(row["parameter"])] = float(row["mean"])

    parameter_sets: list[tuple[str, str, HemodynamicParameters]] = []
    for dataset_id, values in by_dataset.items():
        parameter_sets.append((
            "fitted_mean",
            dataset_id,
            HemodynamicParameters(**{
                key: values[key] for key in ("epsilon", "kas", "kaf", "tau0", "alpha", "e0")
            }),
        ))
    parameter_sets.extend([
        ("croce_2017_preset", "reference", HemodynamicParameters(**CROCE_2017_PRESET)),
        ("adaptive_regularizer_center", "reference", HemodynamicParameters()),
    ])

    output = []
    for source, dataset_id, params in parameter_sets:
        output.append({
            "parameter_source": source,
            "dataset_id": dataset_id,
            "fs_hz": float(fs_hz),
            "impulse_duration_s": float(duration_s),
            "epsilon": params.epsilon,
            "kas": params.kas,
            "kaf": params.kaf,
            "tau0": params.tau0,
            "alpha": params.alpha,
            "e0": params.e0,
            **_impulse_metrics(params, fs_hz=fs_hz, duration_s=duration_s),
        })
    return output


def _summary_markdown(
    aggregate: Sequence[Mapping[str, Any]],
    subject_tests: Sequence[Mapping[str, Any]],
    task_tests: Sequence[Mapping[str, str]],
    delays: Sequence[Mapping[str, Any]],
) -> str:
    subject_significant = [row for row in subject_tests if row["significant_fdr_0_05"]]
    task_primary = [row for row in task_tests if row["anchor_mode"] == "fixed_pooled"]
    task_significant = [row for row in task_primary if _as_bool(row["significant_fdr_0_05"])]
    physiological = [
        row for row in aggregate
        if row["parameter"] in CROCE_2017_PRESET
    ]
    lines = [
        "# Adaptive SSM subject, task, and delay audit",
        "",
        "## Decision",
        "",
        f"- Subject effect: {len(subject_significant)}/{len(subject_tests)} parameters pass within-family BH-FDR at q < .05.",
        f"- Task effect: {len(task_significant)}/{len(task_primary)} fixed-representation parameters pass the existing BH-FDR analysis.",
        "- A subject effect means stable fitted-rank differences across tasks. It does not make the fitted values identified biological constants.",
        "",
        "## Physiological parameter descriptives",
        "",
        "Values pool all fixed-representation subject-task fits within a dataset. Variance is the sample variance across fits.",
        "",
        "| dataset | parameter | mean | variance | min | max | Croce 2017 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in physiological:
        lines.append(
            f"| {row['dataset_id']} | {row['parameter']} | {float(row['mean']):.6g} | "
            f"{float(row['sample_variance']):.6g} | {float(row['minimum']):.6g} | "
            f"{float(row['maximum']):.6g} | {float(row['croce_2017_preset']):.6g} |"
        )
    lines.extend([
        "",
        "## Mean-parameter impulse delays",
        "",
        "A unit-area EEG-driver impulse is propagated through the linearized hemodynamic transition. Peak latency is based on absolute state magnitude; this is a model-implied response delay, not an independently measured EEG-fNIRS lag.",
        "",
        "| source | dataset | HbO onset 10% | HbO peak | HbR peak | joint peak |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in delays:
        lines.append(
            f"| {row['parameter_source']} | {row['dataset_id']} | "
            f"{float(row['hbo_state_onset_10pct_s']):.2f} s | "
            f"{float(row['hbo_state_absolute_peak_s']):.2f} s | "
            f"{float(row['hbr_state_absolute_peak_s']):.2f} s | "
            f"{float(row['joint_fnirs_state_absolute_peak_s']):.2f} s |"
        )
    lines.extend([
        "",
        "## Identifiability and reference boundary",
        "",
        "- `epsilon` is gauge-calibrated after fitting and is not comparable to the paper's unit input gain as an identified physiological coefficient.",
        "- `q_scale` and `fnirs_noise_scale` are selected from discrete grids; `hbo_gain` and `hbr_gain` absorb measurement scale and polarity. They are included in the CSV, but should not be compared with Croce physiological presets.",
        "- The two datasets are reported separately because dataset, protocol, task set, measurement family, and cohort are confounded.",
        "- Frequent bounded solutions remain the main reason not to turn statistical subject effects into physiological subject phenotypes.",
        "- Croce 2017 table 1 and the maintained legacy solver assign 0.41/0.65 to `kas`/`kaf`; the current adaptive model uses 0.65/0.41 and also moves `E0` from 0.34 to 0.40. This implementation drift should be resolved before any paper-faithful reproduction claim.",
        "",
    ])
    return "\n".join(lines)


def run(
    source_run: Path,
    run_dir: Path,
    *,
    iterations: int,
    seed: int,
    fs_hz: float,
    impulse_duration_s: float,
) -> None:
    source_parameters = source_run / "subject_task_parameters.csv"
    source_omnibus = source_run / "omnibus_tests.csv"
    if not source_parameters.exists() or not source_omnibus.exists():
        raise FileNotFoundError(f"incomplete source run: {source_run}")
    run_dir.mkdir(parents=True, exist_ok=False)

    rows = _fixed_rows(_read_csv(source_parameters))
    task_orders = _dataset_task_orders(rows)
    aggregate = _aggregate_statistics(rows, task_orders)
    task_statistics = _task_statistics(rows, task_orders)
    subject_tests = _subject_effect_tests(
        rows, task_orders, iterations=iterations, seed=seed,
    )
    task_tests = _read_csv(source_omnibus)
    delays = _delay_rows(aggregate, fs_hz=fs_hz, duration_s=impulse_duration_s)

    outputs = {
        "aggregate_parameter_statistics.csv": aggregate,
        "task_parameter_statistics.csv": task_statistics,
        "subject_effect_tests.csv": subject_tests,
        "mean_parameter_impulse_delays.csv": delays,
    }
    for name, output_rows in outputs.items():
        _write_csv(run_dir / name, output_rows)
    summary_path = run_dir / "summary.md"
    summary_path.write_text(
        _summary_markdown(aggregate, subject_tests, task_tests, delays),
        encoding="utf-8",
    )

    output_paths = [run_dir / name for name in outputs] + [summary_path]
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "git": _git_payload(),
        "source_run": str(source_run),
        "source_files": {
            str(source_parameters.relative_to(REPO_ROOT)): _sha256(source_parameters),
            str(source_omnibus.relative_to(REPO_ROOT)): _sha256(source_omnibus),
        },
        "analysis": {
            "primary_anchor_mode": "fixed_pooled",
            "subject_effect_null": "independent subject-label permutation within each task block",
            "subject_effect_statistic": "Friedman rank statistic; Kendall's W effect size",
            "subject_effect_multiple_comparisons": "BH-FDR within parameter family across datasets and parameters",
            "iterations": int(iterations),
            "seed": int(seed),
            "delay_definition": "unit-area driver impulse propagated through linearized states; absolute response timing",
            "fs_hz": float(fs_hz),
            "impulse_duration_s": float(impulse_duration_s),
            "croce_2017_doi": "10.1088/1741-2552/aa7321",
            "croce_2017_reference": "equation (4a) and table 1 in the repository PDF",
            "croce_2017_preset": CROCE_2017_PRESET,
        },
        "outputs": [
            {"path": path.name, "sha256": _sha256(path)} for path in output_paths
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--fs-hz", type=float, default=10.0)
    parser.add_argument("--impulse-duration-s", type=float, default=60.0)
    args = parser.parse_args()
    run(
        args.source_run,
        args.run_dir,
        iterations=args.iterations,
        seed=args.seed,
        fs_hz=args.fs_hz,
        impulse_duration_s=args.impulse_duration_s,
    )


if __name__ == "__main__":
    main()
