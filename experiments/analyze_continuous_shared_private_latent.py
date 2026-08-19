#!/usr/bin/env python3
"""Analyze a frozen continuous shared/private latent validation suite."""

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

from experiments.run_continuous_shared_private_latent import (
    ENDPOINTS,
    TASK_ORDER,
    _jsonable,
    _sha256,
    _write_csv,
    _write_json,
)
from src.visualization.token_physiology_plots import save_figure_atomic


SCHEMA = "continuous_shared_private_analysis_v1"
TASK_LABELS = {
    "mental_arithmetic": "Mental arithmetic",
    "motor_imagery": "Motor imagery",
    "word_generation": "Word generation",
    "n_back": "N-back",
}
ENDPOINT_LABELS = {
    "eeg_target_delta_r2": "EEG → SSM target ΔR²",
    "fnirs_target_delta_r2": "fNIRS → SSM target ΔR²",
    "fnirs_to_eeg_swap_delta_r2": "fNIRS shared → EEG swap ΔR²",
    "eeg_to_fnirs_swap_delta_r2": "EEG shared → fNIRS swap ΔR²",
}
COLORS = {
    "mental_arithmetic": "#0072B2",
    "motor_imagery": "#D55E00",
    "word_generation": "#009E73",
    "n_back": "#CC79A7",
}


def _verify_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "continuous_shared_private_suite_v1":
        raise ValueError("continuous suite schema mismatch")
    if manifest.get("mode") != "full":
        raise ValueError("analysis requires a full suite")
    if manifest.get("protected_open") is not False:
        raise PermissionError("source suite opened a protected cohort")
    if manifest.get("vector_quantization") is not False:
        raise ValueError("source suite introduced vector quantization")
    if manifest.get("completed_cell_count") != 12 or manifest.get("failed_cell_count") != 0:
        raise RuntimeError("source suite is incomplete")
    for artifact in manifest["artifacts"]:
        path = run_dir / artifact["path"]
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"source artifact hash mismatch: {path}")
    return manifest


def seed_average_subjects(endpoints: pd.DataFrame, expected_seeds: int = 3) -> pd.DataFrame:
    counts = endpoints.groupby(
        ["task_id", "dataset_id", "subject", "endpoint"]
    ).seed.nunique()
    if not bool((counts == expected_seeds).all()):
        raise ValueError("seed averaging received an incomplete subject/cell")
    return (
        endpoints.groupby(
            ["task_id", "dataset_id", "subject", "endpoint"], as_index=False
        )
        .agg(value=("value", "mean"), seed_sd=("value", "std"), seeds=("seed", "nunique"))
    )


def simultaneous_max_stat_intervals(
    seed_averaged: pd.DataFrame,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    cell_keys = [(task, endpoint) for task in TASK_ORDER for endpoint in ENDPOINTS]
    cell_labels = [f"{task}|{endpoint}" for task, endpoint in cell_keys]
    lookup: dict[tuple[str, str], pd.Series] = {}
    dataset_by_task: dict[str, str] = {}
    for task, endpoint in cell_keys:
        selected = seed_averaged[
            (seed_averaged.task_id == task) & (seed_averaged.endpoint == endpoint)
        ]
        if len(selected) == 0:
            raise ValueError(f"missing primary cell {task}/{endpoint}")
        dataset_by_task[task] = str(selected.dataset_id.iloc[0])
        lookup[(task, endpoint)] = selected.set_index("subject").value.sort_index()
    subjects_by_dataset: dict[str, list[str]] = {}
    for dataset in sorted(set(dataset_by_task.values())):
        tasks = [task for task in TASK_ORDER if dataset_by_task[task] == dataset]
        subject_sets = [set(lookup[(task, ENDPOINTS[0])].index) for task in tasks]
        if any(values != subject_sets[0] for values in subject_sets[1:]):
            raise ValueError(f"task-paired subject set differs inside {dataset}")
        subjects_by_dataset[dataset] = sorted(subject_sets[0])
    observed = np.asarray([float(lookup[key].mean()) for key in cell_keys])
    rng = np.random.default_rng(seed)
    draws = np.empty((int(iterations), len(cell_keys)), dtype=np.float64)
    for draw_index in range(int(iterations)):
        sampled = {
            dataset: rng.choice(subjects, size=len(subjects), replace=True).tolist()
            for dataset, subjects in subjects_by_dataset.items()
        }
        for cell_index, (task, endpoint) in enumerate(cell_keys):
            dataset = dataset_by_task[task]
            draws[draw_index, cell_index] = float(
                lookup[(task, endpoint)].loc[sampled[dataset]].mean()
            )
    standard_error = draws.std(axis=0, ddof=1)
    if np.any(standard_error <= 0) or np.any(~np.isfinite(standard_error)):
        raise ValueError("max-stat bootstrap has a degenerate cell standard error")
    maximum = np.max(np.abs((draws - observed) / standard_error), axis=1)
    critical = float(np.quantile(maximum, confidence_level))
    alpha = (1.0 - confidence_level) / 2.0
    rows = []
    for index, (task, endpoint) in enumerate(cell_keys):
        values = lookup[(task, endpoint)]
        rows.append(
            {
                "task_id": task,
                "dataset_id": dataset_by_task[task],
                "endpoint": endpoint,
                "subjects": len(values),
                "seed_averaged_subject_mean": observed[index],
                "ordinary_bootstrap_ci_low": float(np.quantile(draws[:, index], alpha)),
                "ordinary_bootstrap_ci_high": float(np.quantile(draws[:, index], 1.0 - alpha)),
                "max_stat_standard_error": standard_error[index],
                "max_stat_critical_value": critical,
                "simultaneous_ci_low": observed[index] - critical * standard_error[index],
                "simultaneous_ci_high": observed[index] + critical * standard_error[index],
                "positive_subject_count": int((values > 0).sum()),
                "strict_cell_pass": bool(observed[index] - critical * standard_error[index] > 0),
            }
        )
    return pd.DataFrame(rows), draws, cell_labels


def _provenance(source: Path, source_run: Path, description: str) -> dict[str, Any]:
    return {
        "source_table": str(source.relative_to(REPO_ROOT)),
        "source_table_sha256": _sha256(source),
        "source_run_manifest": str((source_run / "manifest.json").relative_to(REPO_ROOT)),
        "source_run_manifest_sha256": _sha256(source_run / "manifest.json"),
        "description": description,
        "analysis_intent": "exploratory continuous shared/private validation",
        "replication_unit": "subject; three algorithmic seeds averaged before bootstrap",
    }


def _save(
    fig: plt.Figure,
    stem: Path,
    source: Path,
    source_run: Path,
    alt: str,
    description: str,
) -> list[str]:
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


def _plot_primary(
    subjects: pd.DataFrame,
    intervals: pd.DataFrame,
    output: Path,
    source_run: Path,
) -> list[str]:
    source = output / "source_tables/primary_16_cell_intervals.csv"
    intervals.to_csv(source, index=False)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained", sharex=True)
    positions = np.arange(len(TASK_ORDER), dtype=float)
    for axis, endpoint in zip(axes.flat, ENDPOINTS, strict=True):
        for task_index, task in enumerate(TASK_ORDER):
            values = subjects[
                (subjects.task_id == task) & (subjects.endpoint == endpoint)
            ].value.to_numpy(float)
            jitter = np.linspace(-0.055, 0.055, len(values))
            axis.scatter(
                task_index + jitter,
                values,
                s=26,
                marker="o",
                facecolors="none",
                edgecolors=COLORS[task],
                alpha=0.75,
            )
            row = intervals[
                (intervals.task_id == task) & (intervals.endpoint == endpoint)
            ].iloc[0]
            mean = float(row.seed_averaged_subject_mean)
            axis.errorbar(
                task_index,
                mean,
                yerr=[
                    [mean - float(row.simultaneous_ci_low)],
                    [float(row.simultaneous_ci_high) - mean],
                ],
                fmt="D",
                color=COLORS[task],
                capsize=3,
                linewidth=1.4,
            )
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        axis.set_title(ENDPOINT_LABELS[endpoint])
        axis.set_xticks(positions, [TASK_LABELS[value] for value in TASK_ORDER], rotation=30, ha="right")
        axis.set_ylabel("ΔR²")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Registered 16-cell continuous sharedness family", fontsize=14)
    return _save(
        fig,
        output / "figures/primary_16_cell_sharedness",
        source,
        source_run,
        "Four panels show five seed-averaged development subjects per task and endpoint. Diamonds are subject-equal means and whiskers are simultaneous 95% studentized max-stat intervals. Zero is the registered success boundary. EEG target effects are modestly positive, fNIRS target effects are negative, and both cross-modal swap directions cluster near zero.",
        "seed-averaged subject observations and simultaneous max-stat intervals",
    )


def _plot_raw_ablation(raw: pd.DataFrame, output: Path, source_run: Path) -> list[str]:
    averaged = (
        raw.groupby(["task_id", "dataset_id", "subject", "modality", "mode"], as_index=False)
        .r2_vs_train_phase.mean()
    )
    source = output / "source_tables/raw_ablation_seed_averaged.csv"
    averaged.to_csv(source, index=False)
    modes = ("self", "matched", "deranged", "shared_only", "private_only")
    labels = ("Self", "Matched swap", "Deranged", "Shared only", "Private only")
    markers = ("o", "s", "^", "v", "D")
    colors = ("#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), layout="constrained", sharex=True)
    for axis, modality in zip(axes, ("eeg", "fnirs"), strict=True):
        for mode_index, (mode, label, marker, color) in enumerate(zip(modes, labels, markers, colors, strict=True)):
            for task_index, task in enumerate(TASK_ORDER):
                values = averaged[
                    (averaged.task_id == task)
                    & (averaged.modality == modality)
                    & (averaged["mode"] == mode)
                ].r2_vs_train_phase.to_numpy(float)
                x = task_index + (mode_index - 2) * 0.11
                axis.scatter(
                    x + np.linspace(-0.025, 0.025, len(values)), values,
                    s=14, marker=marker, facecolors="none", edgecolors=color, alpha=0.42,
                )
                axis.plot(x, values.mean(), marker=marker, color=color, linestyle="none", label=label if task_index == 0 else None)
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        axis.set_title(f"{modality.upper()} normalized raw reconstruction")
        axis.set_ylabel("R² vs train-only condition/time mean")
        axis.set_xticks(range(4), [TASK_LABELS[value] for value in TASK_ORDER], rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Raw reconstruction ablations after seed averaging", fontsize=14)
    return _save(
        fig,
        output / "figures/raw_reconstruction_ablation",
        source,
        source_run,
        "Two panels show seed-averaged subject R-squared for EEG and fNIRS raw reconstruction under self, matched cross-modal swap, deranged swap, shared-only, and private-only modes. Self reconstruction is best; matched and deranged swaps nearly overlap, while private-only retains substantial signal and shared-only is below the train-only baseline.",
        "subject-level raw reconstruction ablations; algorithmic seeds averaged",
    )


def _plot_probes(probes: pd.DataFrame, output: Path, source_run: Path) -> list[str]:
    averaged = (
        probes.groupby(["task_id", "dataset_id", "subject", "modality", "latent_class"], as_index=False)
        .target_delta_r2.mean()
    )
    source = output / "source_tables/ridge_probe_seed_averaged.csv"
    averaged.to_csv(source, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained", sharex=True, sharey=True)
    styles = (("shared", "Shared latent", "#0072B2", "o"), ("private", "Private latent", "#D55E00", "s"))
    for axis, modality in zip(axes, ("eeg", "fnirs"), strict=True):
        for class_index, (latent_class, label, color, marker) in enumerate(styles):
            for task_index, task in enumerate(TASK_ORDER):
                values = averaged[
                    (averaged.task_id == task)
                    & (averaged.modality == modality)
                    & (averaged.latent_class == latent_class)
                ].target_delta_r2.to_numpy(float)
                x = task_index + (-0.1 if class_index == 0 else 0.1)
                axis.scatter(
                    x + np.linspace(-0.035, 0.035, len(values)), values,
                    s=20, marker=marker, facecolors="none", edgecolors=color, alpha=0.55,
                )
                axis.plot(x, values.mean(), marker=marker, color=color, linestyle="none", label=label if task_index == 0 else None)
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        axis.set_title(f"{modality.upper()} encoder")
        axis.set_ylabel("32-component ridge probe target ΔR²")
        axis.set_xticks(range(4), [TASK_LABELS[value] for value in TASK_ORDER], rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Capacity-matched SSM-target leakage probes", fontsize=14)
    return _save(
        fig,
        output / "figures/shared_private_ridge_probes",
        source,
        source_run,
        "Two panels compare 32-component ridge decoding of the SSM target from shared and private latents for EEG and fNIRS. Subject points are shown after averaging three seeds. EEG shared probes are sometimes positive but private leakage is also present; fNIRS probes are generally below zero.",
        "capacity-matched train-fit ridge probes evaluated on development subjects",
    )


def _plot_latent_diagnostics(diagnostics: pd.DataFrame, output: Path, source_run: Path) -> list[str]:
    source = output / "source_tables/latent_diagnostics.csv"
    diagnostics.to_csv(source, index=False)
    cross = diagnostics[diagnostics.diagnostic == "shared_cross_modal"]
    representation = diagnostics[diagnostics.diagnostic == "representation"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), layout="constrained")
    for index, label in enumerate(("matched", "deranged")):
        means = [
            cross[(cross.task_id == task) & (cross.representation == label)].linear_cka.mean()
            for task in TASK_ORDER
        ]
        axes[0].plot(
            np.arange(4) + (-0.08 if index == 0 else 0.08), means,
            marker="o" if index == 0 else "s", linestyle="none",
            color="#0072B2" if index == 0 else "#D55E00", label=label.title(),
        )
    axes[0].set_title("Cross-modal shared linear CKA")
    axes[0].set_ylabel("Linear CKA")
    rep_order = ("eeg_shared", "fnirs_shared", "eeg_private", "fnirs_private")
    for rep_index, rep in enumerate(rep_order):
        means = [
            representation[(representation.task_id == task) & (representation.representation == rep)].effective_rank.mean()
            for task in TASK_ORDER
        ]
        axes[1].plot(
            np.arange(4) + (rep_index - 1.5) * 0.07,
            means,
            marker=("o", "s", "^", "D")[rep_index],
            linestyle="none",
            color=("#0072B2", "#D55E00", "#009E73", "#CC79A7")[rep_index],
            label=rep.replace("_", " "),
        )
    axes[1].set_title("Latent effective rank")
    axes[1].set_ylabel("Effective rank")
    for axis in axes:
        axis.set_xticks(range(4), [TASK_LABELS[value] for value in TASK_ORDER], rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Descriptive latent geometry diagnostics", fontsize=14)
    return _save(
        fig,
        output / "figures/latent_geometry_diagnostics",
        source,
        source_run,
        "Left: matched and trial-deranged cross-modal shared-latent linear CKA are both low and nearly indistinguishable. Right: effective rank for EEG/fNIRS shared and private representations across tasks and seeds. These are descriptive geometry diagnostics, not biological replicates or primary endpoints.",
        "seed-level descriptive latent CKA and effective rank",
    )


def _plot_losses(losses: pd.DataFrame, output: Path, source_run: Path) -> list[str]:
    source = output / "source_tables/loss_curves.csv"
    losses.to_csv(source, index=False)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained", sharex=False)
    for axis, task in zip(axes.flat, TASK_ORDER, strict=True):
        for seed, rows in losses[losses.task_id == task].groupby("seed"):
            axis.plot(rows.epoch, rows.validation_shared_equal, label=str(seed), linewidth=1.2)
        axis.set_title(TASK_LABELS[task])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation shared-target MSE")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Checkpoint-selection trajectories", fontsize=14)
    return _save(
        fig,
        output / "figures/training_validation_curves",
        source,
        source_run,
        "Four panels show development equal-modality SSM-target mean squared error by epoch for each task and algorithmic seed. These curves determine checkpoint selection; raw reconstruction loss is not used for selection.",
        "per-epoch validation selection metric for all 12 task-seed cells",
    )


def _pattern_source(source_run: Path, output: Path) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        path = source_run / "cells" / task / "seed_20260819" / "validation_predictions.npz"
        with np.load(path) as values:
            sample_id = str(values["sample_id"][0])
            for modality, rate, downsample in (("eeg", 200.0, 20), ("fnirs", 10.0, 1)):
                for mode, key in (
                    ("observed", f"{modality}_observed"),
                    ("matched", f"{modality}_matched"),
                    ("deranged", f"{modality}_deranged"),
                ):
                    signal = np.asarray(values[key][0], dtype=float)
                    if downsample > 1:
                        signal = signal.reshape(signal.shape[0], -1, downsample).mean(axis=2)
                    time_s = np.arange(signal.shape[1]) / (rate / downsample) - 5.0
                    for channel in range(signal.shape[0]):
                        for time, value in zip(time_s, signal[channel], strict=True):
                            rows.append(
                                {
                                    "task_id": task,
                                    "sample_id": sample_id,
                                    "modality": modality,
                                    "mode": mode,
                                    "channel_slot": channel + 1,
                                    "time_s": float(time),
                                    "normalized_value": float(value),
                                }
                            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "source_tables/representative_matched_patterns.csv", index=False)
    return frame


def _plot_patterns(frame: pd.DataFrame, output: Path, source_run: Path) -> list[str]:
    source = output / "source_tables/representative_matched_patterns.csv"
    artifacts: list[str] = []
    for task in TASK_ORDER:
        task_frame = frame[frame.task_id == task]
        fig, axes = plt.subplots(2, 3, figsize=(16, 6.8), layout="constrained")
        for row_index, modality in enumerate(("eeg", "fnirs")):
            values = task_frame[task_frame.modality == modality]
            limit = float(np.quantile(np.abs(values.normalized_value), 0.99))
            limit = max(limit, 1e-6)
            image = None
            for column_index, mode in enumerate(("observed", "matched", "deranged")):
                selected = values[values["mode"] == mode]
                matrix = selected.pivot(index="channel_slot", columns="time_s", values="normalized_value").to_numpy()
                image = axes[row_index, column_index].imshow(
                    matrix,
                    aspect="auto",
                    interpolation="nearest",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    extent=(-5.0, 15.0, matrix.shape[0] + 0.5, 0.5),
                )
                axes[row_index, column_index].set_title(f"{modality.upper()} {mode}")
                axes[row_index, column_index].set_xlabel("Time from event/block onset (s)")
                axes[row_index, column_index].set_ylabel("Channel slot")
            fig.colorbar(image, ax=axes[row_index, :], label="Train-normalized signal")
        fig.suptitle(f"Representative paired pattern: {TASK_LABELS[task]}", fontsize=14)
        artifacts.extend(
            _save(
                fig,
                output / f"figures/representative_pattern_{task}",
                source,
                source_run,
                f"Two rows of heatmaps for one deterministic {TASK_LABELS[task]} development sample. EEG and fNIRS observed normalized patterns are compared with matched cross-modal shared-latent reconstruction and a same-subject same-condition trial-deranged reconstruction. Matched and deranged patterns are visually similar; each modality row uses one symmetric color range.",
                f"representative seed-20260819 normalized pattern for {task}; EEG downsampled to 10 Hz by non-overlapping means for display only",
            )
        )
    return artifacts


def _summary(
    manifest: Mapping[str, Any],
    intervals: pd.DataFrame,
    raw: pd.DataFrame,
    probes: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    strict_pass = bool(intervals.strict_cell_pass.all())
    target = intervals[intervals.endpoint.isin(ENDPOINTS[:2])]
    swap = intervals[intervals.endpoint.isin(ENDPOINTS[2:])]
    raw_average = raw.groupby(["task_id", "modality", "mode"]).r2_vs_train_phase.mean()
    cka = diagnostics[diagnostics.diagnostic == "shared_cross_modal"].groupby(
        ["task_id", "representation"]
    ).linear_cka.mean()
    lines = [
        "# Continuous shared/private latent validation",
        "",
        f"Source suite: {manifest['completed_cell_count']}/{manifest['cell_count']} task-seed cells completed, "
        f"{manifest['source_sample_count']} canonical EEG-fNIRS windows, protected_open=false, vector_quantization=false.",
        "",
        "## Registered decision",
        "",
        f"Strict 16-cell rule: **{'supported' if strict_pass else 'not supported'}**. "
        f"{int(intervals.strict_cell_pass.sum())}/16 simultaneous 95% lower bounds are above zero.",
        "",
        "| Task | Endpoint | Mean ΔR² | Simultaneous 95% interval | Positive subjects | Pass |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in intervals.itertuples(index=False):
        lines.append(
            f"| {TASK_LABELS[row.task_id]} | {ENDPOINT_LABELS[row.endpoint]} | "
            f"{row.seed_averaged_subject_mean:.4f} | [{row.simultaneous_ci_low:.4f}, {row.simultaneous_ci_high:.4f}] | "
            f"{row.positive_subject_count}/{row.subjects} | {'yes' if row.strict_cell_pass else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"EEG-to-target means range {target[target.endpoint == 'eeg_target_delta_r2'].seed_averaged_subject_mean.min():.4f}–"
            f"{target[target.endpoint == 'eeg_target_delta_r2'].seed_averaged_subject_mean.max():.4f}; "
            f"fNIRS-to-target means range {target[target.endpoint == 'fnirs_target_delta_r2'].seed_averaged_subject_mean.min():.4f}–"
            f"{target[target.endpoint == 'fnirs_target_delta_r2'].seed_averaged_subject_mean.max():.4f}. "
            f"Cross-modal swap means range {swap.seed_averaged_subject_mean.min():.4f}–{swap.seed_averaged_subject_mean.max():.4f}.",
            "",
            "The architecture reconstructs each modality best with its own shared+private pair, but matched cross-modal shared substitution does not improve over a same-subject same-condition derangement. Private-only reconstruction retains substantial raw signal, while shared-only reconstruction is below the train-only condition/time baseline. This is evidence of modality-specific reconstruction capacity, not matched EEG-fNIRS shared patterns.",
            "",
            f"Matched versus deranged shared-latent CKA ranges {min(cka.min(), cka.max()):.4f}–{max(cka.min(), cka.max()):.4f} across task/matching cells and is descriptive only. Capacity-matched private probes are reported without an equivalence claim.",
            "",
            "## Claim boundary",
            "",
            "The result is development-only and relative to an SSM proxy whose EEG-only fNIRS reliability was weak in the preceding audit. It does not prove absence of physiological coupling in the measurements. It shows that this unconstrained no-VQ shared/private architecture did not construct an interchangeable matched latent under the registered controls, and it does not authorize VQ.",
        ]
    )
    return "\n".join(lines) + "\n"


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "manifest.json" and path.parent == root:
            continue
        output.append(
            {"path": str(path.relative_to(root)), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    return output


def run(args: argparse.Namespace) -> Path:
    source_run = args.source_run.resolve()
    source_manifest = _verify_run(source_run)
    target = args.output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        (staging / "figures").mkdir()
        (staging / "source_tables").mkdir()
        endpoints = pd.read_csv(source_run / "subject_endpoints.csv")
        raw = pd.read_csv(source_run / "raw_ablation_metrics.csv")
        probes = pd.read_csv(source_run / "ridge_probe_metrics.csv")
        diagnostics = pd.read_csv(source_run / "latent_diagnostics.csv")
        losses = pd.read_csv(source_run / "loss_curves.csv")
        seed_averaged = seed_average_subjects(endpoints)
        intervals, draws, labels = simultaneous_max_stat_intervals(
            seed_averaged,
            iterations=int(args.bootstrap_iterations),
            confidence_level=0.95,
            seed=int(args.seed),
        )
        seed_averaged.to_csv(staging / "seed_averaged_subject_endpoints.csv", index=False)
        intervals.to_csv(staging / "simultaneous_intervals.csv", index=False)
        np.savez_compressed(
            staging / "bootstrap_draws.npz",
            schema=np.asarray("continuous_shared_private_max_stat_bootstrap_v1"),
            cell_label=np.asarray(labels),
            draws=draws,
            iterations=np.asarray(args.bootstrap_iterations),
            seed=np.asarray(args.seed),
        )
        figure_artifacts = []
        figure_artifacts.extend(_plot_primary(seed_averaged, intervals, staging, source_run))
        figure_artifacts.extend(_plot_raw_ablation(raw, staging, source_run))
        figure_artifacts.extend(_plot_probes(probes, staging, source_run))
        figure_artifacts.extend(_plot_latent_diagnostics(diagnostics, staging, source_run))
        figure_artifacts.extend(_plot_losses(losses, staging, source_run))
        pattern_frame = _pattern_source(source_run, staging)
        figure_artifacts.extend(_plot_patterns(pattern_frame, staging, source_run))
        summary = _summary(source_manifest, intervals, raw, probes, diagnostics)
        (staging / "analysis_summary.md").write_text(summary, encoding="utf-8")
        manifest = {
            "schema": SCHEMA,
            "status": "completed",
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "source_run": str(source_run.relative_to(REPO_ROOT)),
            "source_manifest_sha256": _sha256(source_run / "manifest.json"),
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "bootstrap_seed": int(args.seed),
            "seed_averaging_before_subject_bootstrap": True,
            "simultaneous_interval": "studentized_max_stat_95pct",
            "primary_cell_count": 16,
            "strict_rule_passed": bool(intervals.strict_cell_pass.all()),
            "strict_cell_pass_count": int(intervals.strict_cell_pass.sum()),
            "protected_open": False,
            "vector_quantization": False,
            "figure_artifacts": figure_artifacts,
            "inputs": [
                {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "sha256": _sha256(Path(__file__).resolve())},
                {"path": str((source_run / "manifest.json").relative_to(REPO_ROOT)), "sha256": _sha256(source_run / "manifest.json")},
            ],
            "artifacts": _artifact_inventory(staging),
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        return target
    except Exception:
        print(f"failed staging retained at {staging}", file=sys.stderr)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=REPO_ROOT / "experiments/runs/physiology_semantic_tokenizer/continuous_shared_private/20260819_continuous_shared_private_full_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/runs/physiology_semantic_tokenizer/continuous_shared_private/20260819_continuous_shared_private_analysis_v1",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
