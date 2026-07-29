#!/usr/bin/env python3
"""Create a subject-level diagnostic package for one completed R2-D run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


MODALITIES = ("eeg", "fnirs")
COLORS = {"eeg": "#0072B2", "fnirs": "#D55E00"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _delta_r2(
    observed: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
) -> float:
    model_sse = float(np.square(observed - prediction)[mask].sum())
    baseline_sse = float(np.square(observed - baseline)[mask].sum())
    if baseline_sse <= 0.0:
        raise ValueError("Phase-baseline SSE must be positive")
    return 1.0 - model_sse / baseline_sse


def _patch_timing(
    config: dict[str, Any], patch_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive patch intervals from the frozen window, never a hand-written axis."""

    window = config["data"]["window"]
    offset = float(window["offset_s"])
    duration = float(window["duration_s"])
    configured_count = int(config["model"]["num_tokens"])
    if patch_count <= 0 or configured_count != patch_count or duration <= 0.0:
        raise ValueError("Patch timing and frozen model contract are inconsistent")
    width = duration / patch_count
    starts = offset + np.arange(patch_count, dtype=np.float64) * width
    ends = starts + width
    centers = 0.5 * (starts + ends)
    return starts, centers, ends


def _subject_bootstrap_summary(
    values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Reproduce the source run's subject-equal percentile bootstrap."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Subject bootstrap expects one value per subject")
    if iterations <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Invalid subject bootstrap contract")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, values.size, size=(int(iterations), values.size)
    )
    draws = values[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "subject_count": int(values.size),
        "subject_equal_mean_delta_r2": float(values.mean()),
        "confidence_level": float(confidence_level),
        "cluster_bootstrap_iterations": int(iterations),
        "cluster_bootstrap_ci": [float(lower), float(upper)],
        "positive_subject_count": int((values > 0.0).sum()),
    }


def _assert_primary_consistency(
    source: dict[str, Any],
    recomputed: dict[str, dict[str, Any]],
) -> None:
    """Fail closed if predictions and the claimed primary summary diverge."""

    scalar_keys = (
        "subject_count",
        "subject_equal_mean_delta_r2",
        "confidence_level",
        "cluster_bootstrap_iterations",
        "positive_subject_count",
    )
    for modality in (*MODALITIES, "equal_modalities"):
        claimed = source[modality]
        observed = recomputed[modality]
        for key in scalar_keys:
            if not np.isclose(
                float(claimed[key]),
                float(observed[key]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Primary summary mismatch for {modality}.{key}"
                )
        if not np.allclose(
            claimed["cluster_bootstrap_ci"],
            observed["cluster_bootstrap_ci"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Primary summary mismatch for {modality}.cluster_bootstrap_ci"
            )


def _simultaneous_bootstrap_bands(
    values_by_modality: dict[str, np.ndarray],
    bootstrap_indices: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, list[float]]]:
    """Centered max-deviation bands across the full modality × patch family."""

    family = np.concatenate(
        [np.asarray(values_by_modality[key], dtype=np.float64) for key in MODALITIES],
        axis=1,
    )
    means = family.mean(axis=0)
    draws = family[bootstrap_indices].mean(axis=1)
    radius = float(
        np.quantile(
            np.max(np.abs(draws - means[None, :]), axis=1),
            confidence_level,
        )
    )
    output: dict[str, dict[str, list[float]]] = {}
    start = 0
    for modality in MODALITIES:
        width = values_by_modality[modality].shape[1]
        modality_mean = means[start : start + width]
        output[modality] = {
            "simultaneous_ci95_lower": (modality_mean - radius).tolist(),
            "simultaneous_ci95_upper": (modality_mean + radius).tolist(),
        }
        start += width
    return output


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.bootstrap_iterations < 1_000:
        raise ValueError("Use at least 1,000 subject bootstrap iterations")

    summary_path = args.run_dir / "summary.json"
    manifest_path = args.run_dir / "manifest.json"
    config_path = args.run_dir / "resolved_config.yaml"
    prediction_path = args.run_dir / "predictions" / "validation_predictions.npz"
    curve_path = args.run_dir / "metrics" / "loss_curves.csv"
    required = (
        summary_path,
        manifest_path,
        config_path,
        prediction_path,
        curve_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Completed R2-D artifacts are missing: {missing}")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "completed":
        raise RuntimeError("R2-D analysis requires a completed source run")
    if source_summary.get("mode") != "formal_one_seed":
        raise ValueError("Smoke/dry-run artifacts cannot enter scientific analysis")
    if source_manifest.get("protected_open") or source_manifest.get(
        "protected_loader_constructed"
    ):
        raise PermissionError("Protected data entered the source R2-D run")

    with np.load(prediction_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    target = np.asarray(arrays["target"], dtype=np.float64)
    subjects = np.asarray(arrays["subject"]).astype(str)
    unique_subjects = sorted(set(subjects.tolist()))
    if len(unique_subjects) != 5 or target.shape != (50, 10, 20):
        raise ValueError(
            f"Expected 5 subjects and [50,10,20] targets, got "
            f"{len(unique_subjects)} and {target.shape}"
        )
    expected_subjects = sorted(
        map(str, source_manifest["validation_subject_keys"])
    )
    protected_subjects = set(
        map(str, source_manifest["closed_protected_subject_keys"])
    )
    if unique_subjects != expected_subjects:
        raise ValueError("Prediction subjects differ from the frozen validation split")
    if set(unique_subjects).intersection(protected_subjects):
        raise PermissionError("Protected subjects entered validation predictions")
    for modality in MODALITIES:
        for key in ("prediction", "phase", "mask"):
            array = np.asarray(arrays[f"{modality}_{key}"])
            if array.shape != target.shape:
                raise ValueError(
                    f"{modality}_{key} shape differs from the target"
                )
        mask = np.asarray(arrays[f"{modality}_mask"], dtype=bool)
        for key in ("target", f"{modality}_prediction", f"{modality}_phase"):
            if np.any(~np.isfinite(np.asarray(arrays[key])[mask])):
                raise ValueError(f"{key} contains non-finite supported values")

    patch_starts, patch_centers, patch_ends = _patch_timing(
        source_config, target.shape[1]
    )

    rng = np.random.default_rng(args.seed)
    subject_rows: list[dict[str, Any]] = []
    patch_values = {
        modality: np.empty((len(unique_subjects), 10), dtype=np.float64)
        for modality in MODALITIES
    }
    subject_values = {
        modality: np.empty(len(unique_subjects), dtype=np.float64)
        for modality in MODALITIES
    }
    for subject_index, subject in enumerate(unique_subjects):
        selected = subjects == subject
        for modality in MODALITIES:
            prediction = np.asarray(
                arrays[f"{modality}_prediction"], dtype=np.float64
            )[selected]
            phase = np.asarray(
                arrays[f"{modality}_phase"], dtype=np.float64
            )[selected]
            mask = np.asarray(arrays[f"{modality}_mask"], dtype=bool)[selected]
            observed = target[selected]
            overall = _delta_r2(observed, prediction, phase, mask)
            subject_values[modality][subject_index] = overall
            correlation = _safe_corr(observed[mask], prediction[mask])
            phase_correlation = _safe_corr(observed[mask], phase[mask])
            subject_rows.append(
                {
                    "subject": subject,
                    "modality": modality,
                    "delta_r2": overall,
                    "prediction_target_correlation": correlation,
                    "phase_target_correlation": phase_correlation,
                    "supported_points": int(mask.sum()),
                }
            )
            for patch in range(10):
                patch_values[modality][subject_index, patch] = _delta_r2(
                    observed[:, patch],
                    prediction[:, patch],
                    phase[:, patch],
                    mask[:, patch],
                )

    source_statistics = source_config["statistics"]
    primary_recomputed = {
        modality: _subject_bootstrap_summary(
            subject_values[modality],
            iterations=int(source_statistics["bootstrap_iterations"]),
            confidence_level=float(source_statistics["confidence_level"]),
            seed=int(source_statistics["bootstrap_seed"]) + index,
        )
        for index, modality in enumerate(MODALITIES)
    }
    equal_values = 0.5 * (
        subject_values["eeg"] + subject_values["fnirs"]
    )
    primary_recomputed["equal_modalities"] = _subject_bootstrap_summary(
        equal_values,
        iterations=int(source_statistics["bootstrap_iterations"]),
        confidence_level=float(source_statistics["confidence_level"]),
        seed=int(source_statistics["bootstrap_seed"]) + 2,
    )
    _assert_primary_consistency(
        source_summary["subject_level_delta_r2"], primary_recomputed
    )

    bootstrap_indices = rng.integers(
        0,
        len(unique_subjects),
        size=(args.bootstrap_iterations, len(unique_subjects)),
    )
    patch_summary: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        draws = patch_values[modality][bootstrap_indices].mean(axis=1)
        lower, upper = np.percentile(draws, [2.5, 97.5], axis=0)
        patch_summary[modality] = {
            "subject_equal_mean": patch_values[modality].mean(axis=0).tolist(),
            "ci95_lower": lower.tolist(),
            "ci95_upper": upper.tolist(),
            "positive_subject_count": (
                patch_values[modality] > 0.0
            ).sum(axis=0).astype(int).tolist(),
        }

    curves = _load_csv(curve_path)
    epochs = np.asarray([int(row["epoch"]) for row in curves], dtype=int)
    best_epoch = int(source_summary["best_epoch"])
    best_row = next(row for row in curves if int(row["epoch"]) == best_epoch)
    first_row = curves[0]
    final_row = curves[-1]
    learning_diagnostic = {}
    for modality in MODALITIES:
        learning_diagnostic[modality] = {
            "train_mse_epoch0": float(first_row[f"train_{modality}_mse"]),
            "train_mse_best_epoch": float(
                best_row[f"train_{modality}_mse"]
            ),
            "train_relative_reduction_to_best": float(
                1.0
                - float(best_row[f"train_{modality}_mse"])
                / float(first_row[f"train_{modality}_mse"])
            ),
            "validation_mse_epoch0": float(
                first_row[f"validation_{modality}_mse"]
            ),
            "validation_mse_best_epoch": float(
                best_row[f"validation_{modality}_mse"]
            ),
            "validation_relative_reduction_to_best": float(
                1.0
                - float(best_row[f"validation_{modality}_mse"])
                / float(first_row[f"validation_{modality}_mse"])
            ),
            "validation_mse_final_epoch": float(
                final_row[f"validation_{modality}_mse"]
            ),
        }
    simultaneous_bands = _simultaneous_bootstrap_bands(
        patch_values, bootstrap_indices
    )
    for modality in MODALITIES:
        patch_summary[modality].update(simultaneous_bands[modality])

    primary = source_summary["subject_level_delta_r2"]
    failure_reasons = []
    for modality in MODALITIES:
        if float(primary[modality]["subject_equal_mean_delta_r2"]) <= 0.0:
            failure_reasons.append(f"{modality}_subject_equal_mean_not_positive")
        if int(primary[modality]["positive_subject_count"]) < 4:
            failure_reasons.append(f"{modality}_fewer_than_4_of_5_positive")
    diagnostic = {
        "schema": "r2d_continuous_observability_diagnostic_v1",
        "source_run": str(args.run_dir),
        "source_hashes": {
            "summary_sha256": _sha256(summary_path),
            "manifest_sha256": _sha256(manifest_path),
            "resolved_config_sha256": _sha256(config_path),
            "predictions_sha256": _sha256(prediction_path),
            "loss_curves_sha256": _sha256(curve_path),
        },
        "seed": int(source_summary["seed"]),
        "teacher_scope": source_summary["teacher_scope"],
        "promotion_eligible": False,
        "protected_open": False,
        "registered_r2p_gate_evaluated": False,
        "primary_artifact_consistency_verified": True,
        "bilateral_development_feasibility_passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "primary_subject_level": primary,
        "primary_inference_scope": {
            "role": "exploratory_r2d_one_seed_development_endpoint",
            "validation_reuse": (
                "subjects_19_23_used_for_early_stopping_and_endpoint_estimation"
            ),
            "interval_uncertainty_included": "subject_resampling_only",
            "interval_uncertainty_excluded": [
                "checkpoint_selection",
                "training_seed",
                "teacher_estimation",
            ],
        },
        "learning_diagnostic": learning_diagnostic,
        "patch_starts_relative_to_onset_s": patch_starts.tolist(),
        "patch_centers_relative_to_onset_s": patch_centers.tolist(),
        "patch_ends_relative_to_onset_s": patch_ends.tolist(),
        "patch_level": patch_summary,
        "patch_level_inference_scope": {
            "role": "post_hoc_exploratory_diagnostic",
            "family_size": int(len(MODALITIES) * target.shape[1]),
            "pointwise_intervals": (
                "uncorrected_percentile_subject_bootstrap_95"
            ),
            "simultaneous_intervals": (
                "centered_max_absolute_deviation_subject_bootstrap_95_"
                "across_all_20_modality_patch_cells"
            ),
            "confirmatory_patch_claim_allowed": False,
        },
        "interpretation": (
            "The exploratory R1-D coordinate is weakly observable from EEG but "
            "not from fNIRS in heldout development subjects. The bilateral R2 "
            "gate fails. R1-D uses subject/fold-specific teacher parameters and "
            "cannot adjudicate the population-frozen R2-P hypothesis."
        ),
    }

    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    tables.mkdir(parents=True)
    figures.mkdir()
    with (tables / "subject_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0]))
        writer.writeheader()
        writer.writerows(subject_rows)
    patch_rows = []
    centers = diagnostic["patch_centers_relative_to_onset_s"]
    starts = diagnostic["patch_starts_relative_to_onset_s"]
    ends = diagnostic["patch_ends_relative_to_onset_s"]
    for modality in MODALITIES:
        for index, center in enumerate(centers):
            patch_rows.append(
                {
                    "modality": modality,
                    "patch": index,
                    "start_s": starts[index],
                    "center_s": center,
                    "end_s": ends[index],
                    "subject_equal_mean_delta_r2": patch_summary[modality][
                        "subject_equal_mean"
                    ][index],
                    "ci95_lower": patch_summary[modality]["ci95_lower"][index],
                    "ci95_upper": patch_summary[modality]["ci95_upper"][index],
                    "simultaneous_ci95_lower": patch_summary[modality][
                        "simultaneous_ci95_lower"
                    ][index],
                    "simultaneous_ci95_upper": patch_summary[modality][
                        "simultaneous_ci95_upper"
                    ][index],
                    "positive_subject_count": patch_summary[modality][
                        "positive_subject_count"
                    ][index],
                }
            )
    with (tables / "patch_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0]))
        writer.writeheader()
        writer.writerows(patch_rows)
    np.savez_compressed(
        args.output_dir / "subject_cluster_bootstrap.npz",
        schema=np.asarray("r2d_patch_subject_bootstrap_v1"),
        bootstrap_subject_indices=bootstrap_indices,
        eeg_patch_delta_r2=patch_values["eeg"],
        fnirs_patch_delta_r2=patch_values["fnirs"],
    )
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)

    ax = axes[0]
    for modality in MODALITIES:
        ax.plot(
            epochs,
            [float(row[f"train_{modality}_mse"]) for row in curves],
            color=COLORS[modality],
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label=f"{modality.upper()} train",
        )
        ax.plot(
            epochs,
            [float(row[f"validation_{modality}_mse"]) for row in curves],
            color=COLORS[modality],
            linewidth=1.5,
            label=f"{modality.upper()} validation",
        )
    ax.axvline(best_epoch, color="0.3", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked point MSE")
    ax.set_title("Continuous-student learning")
    ax.legend(frameon=False, ncol=2, fontsize=6)

    ax = axes[1]
    subject_x = np.arange(len(unique_subjects))
    offsets = {"eeg": -0.08, "fnirs": 0.08}
    for modality in MODALITIES:
        values = np.asarray(
            [
                row["delta_r2"]
                for row in subject_rows
                if row["modality"] == modality
            ],
            dtype=np.float64,
        )
        ax.scatter(
            subject_x + offsets[modality],
            values,
            color=COLORS[modality],
            s=22,
            label=modality.upper(),
            zorder=3,
        )
    for index in subject_x:
        pair = [
            row["delta_r2"]
            for row in subject_rows
            if row["subject"] == unique_subjects[index]
        ]
        ax.plot(
            [index - 0.08, index + 0.08],
            pair,
            color="0.7",
            linewidth=0.7,
            zorder=1,
        )
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xticks(subject_x, [subject[-2:] for subject in unique_subjects])
    ax.set_xlabel("Held-out development subject")
    ax.set_ylabel(r"$\Delta R^2$ vs phase baseline")
    ax.set_title("Subject-level observability")
    ax.legend(frameon=False)

    ax = axes[2]
    centers_array = np.asarray(centers)
    for modality in MODALITIES:
        mean = np.asarray(patch_summary[modality]["subject_equal_mean"])
        lower = np.asarray(patch_summary[modality]["ci95_lower"])
        upper = np.asarray(patch_summary[modality]["ci95_upper"])
        ax.fill_between(
            centers_array,
            lower,
            upper,
            color=COLORS[modality],
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            centers_array,
            mean,
            color=COLORS[modality],
            marker="o",
            markersize=2.8,
            linewidth=1.2,
            label=modality.upper(),
        )
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Patch center relative to onset (s)")
    ax.set_ylabel(r"Subject-equal patch $\Delta R^2$")
    ax.set_title("Time-resolved diagnostic\n(pointwise 95% CI, uncorrected)")
    ax.legend(frameon=False)

    for label, ax in zip(("A", "B", "C"), axes):
        ax.text(
            -0.16,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
            va="bottom",
        )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            figures / f"r2d_continuous_observability.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
