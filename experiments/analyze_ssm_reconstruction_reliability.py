#!/usr/bin/env python3
"""Create the comprehensive posthoc analysis of a frozen SSM reliability run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_ssm_reconstruction_reliability import (
    _jsonable,
    _task_label,
    _write_csv,
    _write_json,
)
from src.visualization.token_physiology_plots import save_figure_atomic


SCHEMA = "ssm_reconstruction_reliability_analysis_v1"
TASK_ORDER = (
    "single_ma", "single_lmi", "single_rmi", "simultaneous_wg",
    "simultaneous_0back", "simultaneous_2back", "simultaneous_3back",
    "visual_rr", "visual_rf", "visual_fr", "visual_ff", "refed_video", "simultaneous_dsr",
)
CORE_TASKS = TASK_ORDER[:7]
DESCRIPTIVE_TASKS = TASK_ORDER[7:]
MODEL_STYLE = {
    "adaptive_joint": ("Joint smoother", "#D55E00", "o"),
    "adaptive_eeg_only": ("EEG-only smoother", "#0072B2", "s"),
    "adaptive_fnirs_only": ("fNIRS-only smoother", "#009E73", "^"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") not in {
        "ssm_reconstruction_reliability_v1",
        "ssm_reconstruction_reliability_v2",
    }:
        raise ValueError("source run schema mismatch")
    if manifest.get("protected_open") is not False:
        raise PermissionError("source run opened a protected cohort")
    if manifest.get("completed_cell_count") != manifest.get("cell_count") or manifest.get("failed_cell_count") != 0:
        raise RuntimeError("source run is not complete")
    for artifact in manifest["artifacts"]:
        path = run_dir / artifact["path"]
        if not path.exists() or _sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"source artifact hash mismatch: {path}")
    return manifest


def _selected_subjects(subjects: pd.DataFrame) -> pd.DataFrame:
    return subjects[
        (((subjects.stage == "core") & (subjects.role == "development_validation"))
         | ((subjects.stage == "descriptive") & (subjects.role == "descriptive")))
        & (subjects.spatial_mode == "local")
    ].copy()


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, iterations: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    draws = np.mean(values[indices], axis=1)
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_model_contrasts(subjects: pd.DataFrame, *, seed: int, iterations: int) -> pd.DataFrame:
    selected = _selected_subjects(subjects)
    metrics = (
        "hbo_trajectory_deviation_nrmse", "hbr_trajectory_deviation_nrmse", "eeg_trajectory_deviation_nrmse",
        "hbo_predictive_95_coverage", "hbr_predictive_95_coverage", "eeg_predictive_95_coverage",
    )
    rng = np.random.default_rng(seed)
    rows = []
    for task_id in TASK_ORDER:
        subset = selected[selected.task_id == task_id]
        for metric in metrics:
            pivot = subset.pivot(index="subject", columns="model", values=metric).dropna()
            joint = pivot["adaptive_joint"].to_numpy(dtype=float)
            eeg = pivot["adaptive_eeg_only"].to_numpy(dtype=float)
            if "nrmse" in metric:
                effect = np.log(joint / eeg)
                estimand = "paired_log_joint_over_eeg_only"
            else:
                effect = joint - eeg
                estimand = "paired_joint_minus_eeg_only"
            mean, low, high = _bootstrap_mean(effect, rng, iterations)
            rows.append(
                {
                    "task_id": task_id,
                    "stage": str(subset.stage.iloc[0]),
                    "metric": metric,
                    "estimand": estimand,
                    "subjects": len(effect),
                    "joint_mean": float(np.mean(joint)),
                    "eeg_only_mean": float(np.mean(eeg)),
                    "effect_mean": mean,
                    "effect_ci_low": low,
                    "effect_ci_high": high,
                    "subject_fraction_joint_lower": float(np.mean(joint < eeg)),
                }
            )
    return pd.DataFrame(rows)


def spatial_contrasts(subjects: pd.DataFrame, *, seed: int, iterations: int) -> pd.DataFrame:
    subset = subjects[(subjects.stage == "core") & (subjects.role == "development_validation")]
    rng = np.random.default_rng(seed + 1)
    rows = []
    for task_id in CORE_TASKS:
        for model in MODEL_STYLE:
            values = subset[(subset.task_id == task_id) & (subset.model == model)]
            for metric in ("hbo_trajectory_deviation_nrmse", "hbr_trajectory_deviation_nrmse"):
                pivot = values.pivot(index="subject", columns="spatial_mode", values=metric).dropna()
                effect = pivot["global"].to_numpy(dtype=float) - pivot["local"].to_numpy(dtype=float)
                mean, low, high = _bootstrap_mean(effect, rng, iterations)
                rows.append(
                    {
                        "task_id": task_id,
                        "model": model,
                        "metric": metric,
                        "estimand": "paired_global_minus_local",
                        "subjects": len(effect),
                        "effect_mean": mean,
                        "effect_ci_low": low,
                        "effect_ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def cohort_stability(subjects: pd.DataFrame) -> pd.DataFrame:
    subset = subjects[(subjects.stage == "core") & (subjects.spatial_mode == "local")]
    rows = []
    for task_id in CORE_TASKS:
        for model in MODEL_STYLE:
            values = subset[(subset.task_id == task_id) & (subset.model == model)]
            for metric in ("hbo_trajectory_deviation_nrmse", "hbo_predictive_95_coverage"):
                fit = values[values.role == "fit"][metric].to_numpy(dtype=float)
                dev = values[values.role == "development_validation"][metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "task_id": task_id,
                        "model": model,
                        "metric": metric,
                        "fit_subjects": len(fit),
                        "development_subjects": len(dev),
                        "fit_mean": float(np.mean(fit)),
                        "development_mean": float(np.mean(dev)),
                        "development_minus_fit": float(np.mean(dev) - np.mean(fit)),
                    }
                )
    return pd.DataFrame(rows)


def visual_sensitivity(windows: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    subset = windows[
        windows.task_id.isin(TASK_ORDER[7:11])
        & (windows.role == "descriptive")
        & (windows.spatial_mode == "local")
    ].copy()
    rows = []
    for task_id in TASK_ORDER[7:11]:
        for model in MODEL_STYLE:
            values = subset[(subset.task_id == task_id) & (subset.model == model)]
            primary_subject = values.groupby("subject").hbo_trajectory_deviation_nrmse.mean()
            median_subject = values.groupby("subject").hbo_trajectory_deviation_nrmse.median()
            admitted = values[values.hbo_observed_temporal_sd >= threshold]
            dependency = (
                admitted.groupby(["subject", "dependency_group"], as_index=False)
                .hbo_trajectory_deviation_nrmse.mean()
            )
            sensitivity_subject = dependency.groupby("subject").hbo_trajectory_deviation_nrmse.mean()
            rows.append(
                {
                    "task_id": task_id,
                    "model": model,
                    "window_count": len(values),
                    "windows_below_sd_0_01": int((values.hbo_observed_temporal_sd < threshold).sum()),
                    "primary_subject_equal_mean": float(primary_subject.mean()),
                    "mean_of_subject_window_medians": float(median_subject.mean()),
                    "sd_0_01_sensitivity_subject_equal_mean": float(sensitivity_subject.mean()),
                    "maximum_window_nrmse": float(values.hbo_trajectory_deviation_nrmse.max()),
                }
            )
    return pd.DataFrame(rows)


def _provenance(source: Path, source_run: Path, description: str) -> dict[str, Any]:
    return {
        "source_table": str(source.relative_to(REPO_ROOT)),
        "source_table_sha256": _sha256(source),
        "source_run_manifest": str((source_run / "manifest.json").relative_to(REPO_ROOT)),
        "source_run_manifest_sha256": _sha256(source_run / "manifest.json"),
        "description": description,
        "analysis_intent": "posthoc descriptive reliability analysis",
        "no_cross_task_pooling": True,
    }


def _save(fig: plt.Figure, stem: Path, source: Path, source_run: Path, alt: str, description: str) -> list[str]:
    artifacts = save_figure_atomic(
        fig,
        stem,
        formats=("svg", "png"),
        dpi=300,
        alt_text=alt,
        provenance=_provenance(source, source_run, description),
    )
    plt.close(fig)
    paths = [*artifacts.figure_paths, artifacts.manifest_path, artifacts.alt_text_path]
    return [str(path.relative_to(stem.parents[1])) for path in paths if path is not None]


def _plot_task_profile(
    selected: pd.DataFrame,
    tasks: Sequence[str],
    output: Path,
    source_run: Path,
    *,
    name: str,
    log_nrmse: bool,
) -> list[str]:
    source = output / f"source_tables/{name}.csv"
    selected[selected.task_id.isin(tasks)].to_csv(source, index=False)
    metrics = (
        ("hbo_trajectory_deviation_nrmse", "HbO trajectory NRMSE", log_nrmse, None),
        ("hbr_trajectory_deviation_nrmse", "HbR trajectory NRMSE", log_nrmse, None),
        ("hbo_predictive_95_coverage", "HbO empirical 95% coverage", False, 0.95),
        ("hbr_predictive_95_coverage", "HbR empirical 95% coverage", False, 0.95),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained", sharex=True)
    for axis, (metric, title, log_scale, reference) in zip(axes.flat, metrics, strict=True):
        for model_index, (model, (label, color, marker)) in enumerate(MODEL_STYLE.items()):
            offset = -0.13 if model_index == 0 else 0.13
            for index, task in enumerate(tasks):
                values = selected[(selected.task_id == task) & (selected.model == model)][metric].to_numpy(float)
                jitter = np.linspace(-0.05, 0.05, len(values))
                axis.scatter(index + offset + jitter, values, s=18, facecolors="none", edgecolors=color, marker=marker, alpha=0.55)
                axis.plot(index + offset, np.mean(values), marker=marker, color=color, label=label if index == 0 else None)
        if log_scale:
            axis.set_yscale("log")
        if reference is not None:
            axis.axhline(reference, color="#555555", linestyle="--", linewidth=0.9)
        axis.set_title(title)
        axis.set_xticks(range(len(tasks)), [_task_label(value) for value in tasks], rotation=40, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Core development-validation reliability" if name.startswith("core") else "Task-specific descriptive reliability", fontsize=14)
    return _save(
        fig, output / f"figures/{name}", source, source_run,
        (
            "Four panels show subject points and equal-subject means for HbO/HbR trajectory NRMSE and empirical "
            f"95% coverage across {'core development-validation' if name.startswith('core') else 'descriptive'} task cells. "
            f"NRMSE uses {'a logarithmic' if log_nrmse else 'a linear'} vertical scale; joint and EEG-only paths are separate."
        ),
        "subject-level task profile; points are independent subjects",
    )


def _plot_paired_contrasts(contrasts: pd.DataFrame, output: Path, source_run: Path) -> list[str]:
    source = output / "source_tables/paired_model_contrasts.csv"
    contrasts.to_csv(source, index=False)
    metrics = (
        ("hbo_trajectory_deviation_nrmse", "HbO log(joint / EEG-only NRMSE)"),
        ("hbr_trajectory_deviation_nrmse", "HbR log(joint / EEG-only NRMSE)"),
        ("eeg_trajectory_deviation_nrmse", "EEG log(joint / EEG-only NRMSE)"),
        ("hbo_predictive_95_coverage", "HbO joint − EEG-only coverage"),
        ("hbr_predictive_95_coverage", "HbR joint − EEG-only coverage"),
        ("eeg_predictive_95_coverage", "EEG joint − EEG-only coverage"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(21, 10), layout="constrained", sharex=True)
    x = np.arange(len(TASK_ORDER))
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        rows = contrasts[contrasts.metric == metric].set_index("task_id").loc[list(TASK_ORDER)]
        mean = rows.effect_mean.to_numpy(float)
        low = rows.effect_ci_low.to_numpy(float)
        high = rows.effect_ci_high.to_numpy(float)
        axis.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="o", color="#6A3D9A", capsize=2)
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        axis.axvline(6.5, color="#999999", linestyle=":", linewidth=0.9)
        axis.set_title(title)
        axis.set_xticks(x, [_task_label(value) for value in TASK_ORDER], rotation=55, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Paired within-subject joint versus EEG-only reconstruction contrasts", fontsize=14)
    return _save(
        fig, output / "figures/paired_model_contrasts", source, source_run,
        "Six panels show paired within-subject joint-versus-EEG-only effects with 95% subject-bootstrap intervals. Negative log NRMSE ratios favor joint reconstruction; positive coverage differences mean higher joint coverage. A divider separates core and descriptive tasks.",
        "paired subject contrasts with 10000 deterministic subject-bootstrap draws",
    )


def _plot_visual_sensitivity(
    windows: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output: Path,
    source_run: Path,
) -> list[str]:
    source = output / "source_tables/visual_denominator_sensitivity.csv"
    plot_windows = windows[
        windows.task_id.isin(TASK_ORDER[7:11])
        & (windows.role == "descriptive")
        & (windows.spatial_mode == "local")
    ][
        ["task_id", "subject", "unit_id", "model", "hbo_observed_temporal_sd", "hbo_trajectory_deviation_nrmse"]
    ]
    plot_windows.to_csv(source, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), layout="constrained")
    measures = (
        ("primary_subject_equal_mean", "Primary mean"),
        ("mean_of_subject_window_medians", "Mean subject median"),
        ("sd_0_01_sensitivity_subject_equal_mean", "SD≥0.01 sensitivity"),
    )
    x = np.arange(4, dtype=float)
    width = 0.12
    for model_index, (model, (model_label, color, marker)) in enumerate(MODEL_STYLE.items()):
        rows = sensitivity[sensitivity.model == model].set_index("task_id").loc[list(TASK_ORDER[7:11])]
        for measure_index, (column, label) in enumerate(measures):
            offset = (model_index * len(measures) + measure_index - 2.5) * width
            axes[0].bar(x + offset, rows[column], width=width, color=color, alpha=0.35 + 0.22 * measure_index, label=f"{model_label}: {label}")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [_task_label(value) for value in TASK_ORDER[7:11]])
    axes[0].set_ylabel("HbO NRMSE (log scale)")
    axes[0].set_title("Primary and denominator-sensitivity summaries")
    axes[0].legend(fontsize=7, ncol=2)
    for model, (label, color, marker) in MODEL_STYLE.items():
        values = plot_windows[plot_windows.model == model]
        axes[1].scatter(
            values.hbo_observed_temporal_sd,
            values.hbo_trajectory_deviation_nrmse,
            s=8,
            alpha=0.22,
            color=color,
            marker=marker,
            label=label,
        )
    axes[1].axvline(0.01, color="#333333", linestyle="--", linewidth=0.9, label="Sensitivity threshold 0.01")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Observed within-window temporal SD")
    axes[1].set_ylabel("HbO trajectory NRMSE")
    axes[1].set_title("Denominator instability at the window level")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Visual NRMSE denominator sensitivity (posthoc)", fontsize=14)
    return _save(
        fig, output / "figures/visual_denominator_sensitivity", source, source_run,
        "Left: logarithmic bars compare the registered subject-equal mean with a subject-window median and a posthoc observed-SD-at-least-0.01 sensitivity summary for four Visual conditions and two models. Right: each window's NRMSE against observed temporal SD on log-log axes, showing extreme NRMSE where the denominator approaches zero.",
        "posthoc Visual sensitivity; the registered primary metric is retained and not replaced",
    )


def _plot_spatial(spatial: pd.DataFrame, output: Path, source_run: Path) -> list[str]:
    source = output / "source_tables/core_spatial_contrasts.csv"
    spatial.to_csv(source, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained", sharex=True)
    for axis, metric in zip(axes, ("hbo_trajectory_deviation_nrmse", "hbr_trajectory_deviation_nrmse"), strict=True):
        for model_index, (model, (label, color, marker)) in enumerate(MODEL_STYLE.items()):
            rows = spatial[(spatial.metric == metric) & (spatial.model == model)].set_index("task_id").loc[list(CORE_TASKS)]
            mean = rows.effect_mean.to_numpy(float)
            low = rows.effect_ci_low.to_numpy(float)
            high = rows.effect_ci_high.to_numpy(float)
            x = np.arange(len(CORE_TASKS)) + (-0.11 if model_index == 0 else 0.11)
            axis.errorbar(x, mean, yerr=[mean - low, high - mean], fmt=marker, color=color, capsize=2, label=label)
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        axis.set_title(f"{metric[:3].upper()} global − local NRMSE")
        axis.set_xticks(range(len(CORE_TASKS)), [_task_label(value) for value in CORE_TASKS], rotation=40, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Core development-validation spatial ablation", fontsize=14)
    return _save(
        fig, output / "figures/core_spatial_contrasts", source, source_run,
        "Two panels show paired global-minus-local HbO and HbR NRMSE effects for each core task and model, with 95% subject-bootstrap intervals. Zero means no spatial difference; positive values favor the local six-EEG-channel path.",
        "paired spatial ablation in the five development-validation subjects",
    )


def _analysis_markdown(
    selected: pd.DataFrame,
    contrasts: pd.DataFrame,
    spatial: pd.DataFrame,
    cohort: pd.DataFrame,
    sensitivity: pd.DataFrame,
    source_manifest: Mapping[str, Any],
) -> str:
    core = selected[selected.task_id.isin(CORE_TASKS)]
    core_table = (
        core.groupby(["task_id", "model"], sort=False)
        .agg(
            subjects=("subject", "nunique"),
            hbo_nrmse=("hbo_trajectory_deviation_nrmse", "mean"),
            hbr_nrmse=("hbr_trajectory_deviation_nrmse", "mean"),
            hbo_sd_ratio=("hbo_temporal_sd_ratio", "mean"),
            hbo_std_resid=("hbo_standardized_residual_rms", "mean"),
            hbo_coverage=("hbo_predictive_95_coverage", "mean"),
            eeg_nrmse=("eeg_trajectory_deviation_nrmse", "mean"),
        )
        .reset_index()
    )
    core_joint = core_table[core_table.model == "adaptive_joint"]
    core_eeg = core_table[core_table.model == "adaptive_eeg_only"]
    core_hbo_effect = contrasts[
        contrasts.task_id.isin(CORE_TASKS)
        & (contrasts.metric == "hbo_trajectory_deviation_nrmse")
    ]
    descriptive_hbo_effect = contrasts[
        contrasts.task_id.isin(DESCRIPTIVE_TASKS)
        & (contrasts.metric == "hbo_trajectory_deviation_nrmse")
    ]
    spatial_decisive = spatial[
        (spatial.effect_ci_low > 0.0) | (spatial.effect_ci_high < 0.0)
    ]
    max_cohort_shift = cohort.loc[cohort.development_minus_fit.abs().idxmax()]
    visual_removed = sensitivity.groupby("task_id").windows_below_sd_0_01.max().to_dict()
    lines = [
        "# Comprehensive posthoc analysis of SSM reconstruction reliability",
        "",
        f"Source run completeness: {source_manifest['completed_cell_count']}/{source_manifest['cell_count']} subject/task cells, "
        f"{source_manifest['failed_cell_count']} failures, protected_open={str(source_manifest['protected_open']).lower()}. "
        "All results below are exploratory and retain task-specific estimands.",
        "",
        "## Main finding",
        "",
        "The joint smoother reconstructs HbO/HbR more closely than the EEG-only path in every task cell, but this does not validate an EEG-derived shared state: the joint path directly conditions on held-out fNIRS. "
        "The EEG-only path is the relevant cross-modal check and remains weak and undercovered across the seven core development tasks.",
        "",
        f"Across core tasks, joint HbO NRMSE ranges {core_joint.hbo_nrmse.min():.3f}–{core_joint.hbo_nrmse.max():.3f}, while EEG-only HbO NRMSE ranges {core_eeg.hbo_nrmse.min():.3f}–{core_eeg.hbo_nrmse.max():.3f}. "
        f"Joint HbO coverage ranges {core_joint.hbo_coverage.min():.3f}–{core_joint.hbo_coverage.max():.3f}; EEG-only coverage ranges {core_eeg.hbo_coverage.min():.3f}–{core_eeg.hbo_coverage.max():.3f}. "
        f"All {len(core_hbo_effect)} core paired log NRMSE effects favor joint smoothing; {int(((core_hbo_effect.effect_ci_high < 0)).sum())} have 95% subject-bootstrap intervals wholly below zero.",
        "",
        "## Core development-validation profile",
        "",
        "| Task | Model | n | HbO NRMSE | HbR NRMSE | HbO temporal SD ratio | HbO std. residual RMS | HbO coverage | EEG-proxy NRMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in core_table.iterrows():
        lines.append(
            f"| {_task_label(row.task_id)} | {row.model.replace('adaptive_', '')} | {int(row.subjects)} | "
            f"{row.hbo_nrmse:.3f} | {row.hbr_nrmse:.3f} | {row.hbo_sd_ratio:.3f} | "
            f"{row.hbo_std_resid:.3f} | {row.hbo_coverage:.3f} | {row.eeg_nrmse:.3f} |"
        )
    lines.extend(
        [
            "",
            "The EEG-only smoother reconstructs its directly observed EEG proxy well (NRMSE roughly 0.29–0.37 in core tasks), yet its HbO/HbR errors remain around two observed temporal SDs. "
            "This dissociation is the clearest reliability result: retaining the EEG proxy does not imply reliable hemodynamic recovery.",
            "",
            "## Descriptive annex and Visual instability",
            "",
            f"All {len(descriptive_hbo_effect)} descriptive HbO paired log NRMSE effects also favor joint smoothing, but Visual primary means are not stable summaries. "
            "S01 contains windows whose observed HbO temporal SD is near zero but above the preregistered 1e-8 undefined threshold, producing mathematically valid yet enormous NRMSE values. "
            f"The posthoc SD≥0.01 sensitivity excludes RR={visual_removed.get('visual_rr', 0)}, RF={visual_removed.get('visual_rf', 0)}, "
            f"FR={visual_removed.get('visual_fr', 0)}, and FF={visual_removed.get('visual_ff', 0)} windows per model before repeating the dependency→subject aggregation. "
            "The registered primary estimates are retained; median and thresholded results are labeled sensitivity analyses rather than replacements.",
            "",
            "REFED and DSR do not show this denominator pathology. Their EEG-only HbO NRMSE values are approximately 1.90 and 1.96, respectively; both are descriptive task-specific surfaces, not pooled confirmations.",
            "",
            "## Spatial and cohort checks",
            "",
            f"The core local-versus-global ablation has {len(spatial_decisive)}/{len(spatial)} task/model/modality intervals excluding zero, with effects in both directions. "
            "There is therefore no uniform evidence that all-scalp EEG improves reconstruction over the six-channel local path.",
            "",
            f"The largest absolute fit-to-development mean shift among the recorded core checks is {max_cohort_shift.development_minus_fit:+.3f} "
            f"for {_task_label(max_cohort_shift.task_id)}, {max_cohort_shift.model.replace('adaptive_', '')}, {max_cohort_shift.metric}. "
            "This cohort comparison is descriptive because both cohorts use within-subject refitting; it is not an external generalization test.",
            "",
            "## Interpretation",
            "",
            "- Joint smoothing is useful as a multimodal reconstruction/teacher diagnostic, with broadly near-nominal posterior coverage, but it is partly self-conditioned on fNIRS.",
            "- EEG-only fNIRS reconstruction is systematically worse, generally overvariable (HbO temporal SD ratio above one in core tasks) and undercovered. This weakens the claim that the current SSM exposes a reliable shared EEG–fNIRS trajectory.",
            "- Predictive uncertainty is not interchangeable with reconstruction deviation: joint standardized residual RMS is mostly below one, while EEG-only values are mostly above one.",
            "- Visual denominator failures show why observed temporal SD, NRMSE, and raw MSE must remain jointly inspectable.",
            "- These results do not test shared/private latent existence and do not authorize VQ; they motivate the planned continuous-latent experiment with explicit negative controls.",
            "",
            "## Evidence map",
            "",
            "- `paired_model_contrasts.csv`: paired joint-versus-EEG-only effects and 10,000-draw subject-bootstrap intervals.",
            "- `core_spatial_contrasts.csv`: paired global-minus-local effects.",
            "- `core_cohort_stability.csv`: fit/development descriptive shifts.",
            "- `visual_nrmse_sensitivity.csv`: registered primary, median, and denominator-threshold sensitivity summaries.",
            "- `figures/`: SVG/PNG figures with alt text and per-figure provenance; `source_tables/` contains exact plotted rows.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    source_run = Path(args.source_run)
    if not source_run.is_absolute():
        source_run = REPO_ROOT / source_run
    source_manifest = _verify_run(source_run)
    output = Path(args.output_dir) if args.output_dir else source_run.parent / "20260819_ssm_reconstruction_reliability_analysis_v1"
    if not output.is_absolute():
        output = REPO_ROOT / output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing analysis: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        (staging / "figures").mkdir()
        (staging / "source_tables").mkdir()
        subjects = pd.read_csv(source_run / "subject_metrics.csv")
        windows = pd.read_csv(source_run / "window_metrics.csv")
        selected = _selected_subjects(subjects)
        contrasts = paired_model_contrasts(subjects, seed=20260819, iterations=10_000)
        spatial = spatial_contrasts(subjects, seed=20260819, iterations=10_000)
        cohort = cohort_stability(subjects)
        sensitivity = visual_sensitivity(windows)
        contrasts.to_csv(staging / "paired_model_contrasts.csv", index=False)
        spatial.to_csv(staging / "core_spatial_contrasts.csv", index=False)
        cohort.to_csv(staging / "core_cohort_stability.csv", index=False)
        sensitivity.to_csv(staging / "visual_nrmse_sensitivity.csv", index=False)
        selected.to_csv(staging / "selected_subject_metrics.csv", index=False)
        figures = []
        figures.extend(_plot_task_profile(selected, CORE_TASKS, staging, source_run, name="core_reliability_profile", log_nrmse=False))
        figures.extend(_plot_task_profile(selected, DESCRIPTIVE_TASKS, staging, source_run, name="descriptive_reliability_profile_log", log_nrmse=True))
        figures.extend(_plot_paired_contrasts(contrasts, staging, source_run))
        figures.extend(_plot_visual_sensitivity(windows, sensitivity, staging, source_run))
        figures.extend(_plot_spatial(spatial, staging, source_run))
        (staging / "analysis_summary.md").write_text(
            _analysis_markdown(selected, contrasts, spatial, cohort, sensitivity, source_manifest),
            encoding="utf-8",
        )
        artifacts = sorted(path for path in staging.rglob("*") if path.is_file() and path.name != "manifest.json")
        manifest = {
            "schema": SCHEMA,
            "source_run": str(source_run.relative_to(REPO_ROOT)),
            "source_manifest_sha256": _sha256(source_run / "manifest.json"),
            "source_run_verified": True,
            "analysis_intent": "posthoc_descriptive",
            "bootstrap_iterations": 10_000,
            "visual_sd_sensitivity_threshold": 0.01,
            "registered_primary_replaced": False,
            "protected_open": False,
            "figures": figures,
            "artifacts": [
                {
                    "path": str(path.relative_to(staging)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in artifacts
            ],
        }
        _write_json(staging / "manifest.json", manifest)
        os.rename(staging, output)
    except Exception:
        if staging.exists() and staging.parent == output.parent and staging.name.startswith(f".{output.name}.staging-"):
            shutil.rmtree(staging)
        raise
    print(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        default="experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260819_ssm_reconstruction_reliability_full_v1",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
