#!/usr/bin/env python3
"""Audit the geometry of the exploratory R1-D shared-driver target.

This analysis is intentionally upstream of tokenizer training.  It compares the
gauge-corrected adaptive-joint trajectory with its paired EEG-only trajectory
and asks whether the correction is phase-locked, cross-subject stable, and most
closely associated with the teacher's flow, HbO, or HbR state update.

The source teacher is subject-specific leave-one-trial-out.  Consequently every
artifact produced here is diagnostic and explicitly ineligible for promotion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("adaptive_joint", "adaptive_eeg_only")
STATE_COLUMNS = (
    "target_shared_driver",
    "target_flow_delta",
    "target_hbo_state",
    "target_hbr_state",
)
TRAIN_SUBJECTS = tuple(f"subject_{index:02d}" for index in range(1, 19))
VALIDATION_SUBJECTS = tuple(f"subject_{index:02d}" for index in range(19, 24))
PROTECTED_SUBJECTS = frozenset(f"subject_{index:02d}" for index in range(24, 30))
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="E0 run directory containing base_model/trajectories.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty_worktree": bool(run("git", "status", "--porcelain")),
    }


def _load_trajectories(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    rows: dict[tuple[str, int, float], dict[str, np.ndarray]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(("subject", "heldout_trial", "time_s", "model", *STATE_COLUMNS))
        missing.difference_update(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Trajectory CSV is missing columns: {sorted(missing)}")
        for row in reader:
            subject = str(row["subject"])
            if subject in PROTECTED_SUBJECTS:
                raise RuntimeError("Protected subjects must not enter an R1-D diagnostic")
            model = str(row["model"])
            if model not in MODELS:
                continue
            key = (subject, int(row["heldout_trial"]), float(row["time_s"]))
            if model in rows[key]:
                raise ValueError(f"Duplicate trajectory row for {key}, model={model}")
            rows[key][model] = np.asarray(
                [float(row[column]) for column in STATE_COLUMNS],
                dtype=np.float64,
            )

    subjects = sorted({key[0] for key in rows})
    expected_subjects = list(TRAIN_SUBJECTS + VALIDATION_SUBJECTS)
    if subjects != expected_subjects:
        raise ValueError(
            "R1-D source subjects differ from the frozen development split: "
            f"{subjects}"
        )
    trials = sorted({key[1] for key in rows})
    times = np.asarray(sorted({key[2] for key in rows}), dtype=np.float64)
    if trials != list(range(10)) or times.shape != (200,):
        raise ValueError(
            f"Expected 10 trials and 200 time points, got {trials} and {times.shape}"
        )
    if not np.allclose(np.diff(times), 0.1, atol=1e-9, rtol=0.0):
        raise ValueError("Teacher time coordinate is not the registered 0.1 s grid")

    values = np.empty(
        (len(subjects), len(trials), len(times), len(MODELS), len(STATE_COLUMNS)),
        dtype=np.float64,
    )
    for subject_index, subject in enumerate(subjects):
        for trial_index, trial in enumerate(trials):
            for time_index, time_s in enumerate(times.tolist()):
                key = (subject, trial, time_s)
                if set(rows.get(key, {})) != set(MODELS):
                    raise ValueError(f"Unpaired teacher trajectories for {key}")
                for model_index, model in enumerate(MODELS):
                    values[subject_index, trial_index, time_index, model_index] = (
                        rows[key][model]
                    )
    if not np.isfinite(values).all():
        raise ValueError("Teacher trajectories contain non-finite values")
    return values, subjects, times


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _delta_r2(observed: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(observed).sum())
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.square(observed - prediction).sum()) / denominator


def _percentile_interval(samples: np.ndarray) -> list[float]:
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return [float(lower), float(upper)]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    if args.bootstrap_iterations < 1_000:
        raise ValueError("Use at least 1,000 subject bootstrap iterations")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing overwrite: {args.output_dir}"
        )
    source_csv = args.source_run / "base_model" / "trajectories.csv"
    source_manifest = args.source_run / "base_model" / "manifest.json"
    if not source_csv.is_file() or not source_manifest.is_file():
        raise FileNotFoundError("Source run lacks base_model trajectory/manifest artifacts")

    values, subjects, times = _load_trajectories(source_csv)
    train_count = len(TRAIN_SUBJECTS)
    joint = values[:, :, :, 0, :]
    eeg_only = values[:, :, :, 1, :]

    # A single training-subject scale is used for rJ and rE.  The subtraction
    # therefore remains in a common gauge and mirrors the R1-D sidecar contract.
    driver_train = joint[:train_count, :, :, 0]
    driver_mean = float(driver_train.mean())
    driver_scale = float(driver_train.std())
    if not np.isfinite(driver_scale) or driver_scale <= 0.0:
        raise ValueError("Invalid training-only shared-driver scale")
    joint_driver = (joint[:, :, :, 0] - driver_mean) / driver_scale
    eeg_driver = (eeg_only[:, :, :, 0] - driver_mean) / driver_scale
    correction = joint_driver - eeg_driver
    state_corrections = joint[:, :, :, 1:] - eeg_only[:, :, :, 1:]

    subject_profile = correction.mean(axis=1)
    population_profile = subject_profile.mean(axis=0)
    phase_template = correction[:train_count].mean(axis=(0, 1))
    scalar_template = float(correction[:train_count].mean())
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0,
        len(subjects),
        size=(args.bootstrap_iterations, len(subjects)),
    )
    bootstrap_profiles = subject_profile[bootstrap_indices].mean(axis=1)
    profile_lower, profile_upper = np.percentile(
        bootstrap_profiles,
        [2.5, 97.5],
        axis=0,
    )

    subject_rows: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects):
        odd = correction[index, 0::2].mean(axis=0)
        even = correction[index, 1::2].mean(axis=0)
        within_template = correction[index].mean(axis=0)
        observed = correction[index]
        joint_centered = joint_driver[index] - joint_driver[index].mean()
        row: dict[str, Any] = {
            "subject": subject,
            "split": "train" if index < train_count else "validation",
            "correction_mean_train_sd": float(observed.mean()),
            "correction_rms_train_sd": float(np.sqrt(np.square(observed).mean())),
            "correction_to_centered_joint_rms": float(
                np.sqrt(np.square(observed).mean())
                / np.sqrt(np.square(joint_centered).mean())
            ),
            "split_half_phase_correlation": _safe_corr(odd, even),
            "within_subject_phase_delta_r2_vs_zero": _delta_r2(
                observed,
                within_template[None, :],
            ),
            "train_phase_delta_r2_vs_zero": _delta_r2(
                observed,
                phase_template[None, :],
            ),
            "train_scalar_delta_r2_vs_zero": _delta_r2(
                observed,
                np.full_like(observed, scalar_template),
            ),
        }
        for state_index, state in enumerate(("flow", "hbo", "hbr")):
            row[f"correction_correlation_{state}"] = _safe_corr(
                observed.ravel(),
                state_corrections[index, :, :, state_index].ravel(),
            )
        subject_rows.append(row)

    time_rows = []
    for time_index, time_s in enumerate(times):
        time_rows.append(
            {
                "time_s": float(time_s),
                "correction_mean_train_sd": float(population_profile[time_index]),
                "correction_ci95_lower": float(profile_lower[time_index]),
                "correction_ci95_upper": float(profile_upper[time_index]),
                "flow_correction_mean_native": float(
                    state_corrections[:, :, time_index, 0].mean()
                ),
                "hbo_correction_mean_native": float(
                    state_corrections[:, :, time_index, 1].mean()
                ),
                "hbr_correction_mean_native": float(
                    state_corrections[:, :, time_index, 2].mean()
                ),
            }
        )

    metrics = {
        key: np.asarray([float(row[key]) for row in subject_rows], dtype=np.float64)
        for key in (
            "correction_rms_train_sd",
            "correction_to_centered_joint_rms",
            "split_half_phase_correlation",
            "within_subject_phase_delta_r2_vs_zero",
            "correction_correlation_flow",
            "correction_correlation_hbo",
            "correction_correlation_hbr",
        )
    }
    validation_phase = np.asarray(
        [
            float(row["train_phase_delta_r2_vs_zero"])
            for row in subject_rows
            if row["split"] == "validation"
        ],
        dtype=np.float64,
    )
    metric_bootstrap: dict[str, np.ndarray] = {}
    for name, value in metrics.items():
        metric_bootstrap[name] = value[bootstrap_indices].mean(axis=1)
    validation_indices = rng.integers(
        0,
        len(VALIDATION_SUBJECTS),
        size=(args.bootstrap_iterations, len(VALIDATION_SUBJECTS)),
    )
    metric_bootstrap["validation_phase_delta_r2"] = validation_phase[
        validation_indices
    ].mean(axis=1)

    peak_index = int(np.argmax(np.abs(population_profile)))
    summary: dict[str, Any] = {
        "schema": "r1d_teacher_geometry_audit_v1",
        "scope": "development_crossfit_diagnostic",
        "promotion_eligible": False,
        "protected_test_included": False,
        "source": {
            "run": str(args.source_run),
            "trajectories_sha256": _sha256(source_csv),
            "manifest_sha256": _sha256(source_manifest),
        },
        "git": _git_state(),
        "sample_counts": {
            "subjects": len(subjects),
            "train_subjects": len(TRAIN_SUBJECTS),
            "validation_subjects": len(VALIDATION_SUBJECTS),
            "trials_per_subject": 10,
            "time_points_per_trial": len(times),
        },
        "normalization": {
            "fit_subjects": list(TRAIN_SUBJECTS),
            "shared_driver_mean": driver_mean,
            "shared_driver_scale": driver_scale,
            "same_transform_applied_to_joint_and_eeg_only": True,
        },
        "findings": {
            "population_correction_peak_time_s": float(times[peak_index]),
            "population_correction_peak_signed_amplitude_train_sd": float(
                population_profile[peak_index]
            ),
            "subject_mean_correction_rms_train_sd": float(
                metrics["correction_rms_train_sd"].mean()
            ),
            "subject_mean_correction_rms_ci95": _percentile_interval(
                metric_bootstrap["correction_rms_train_sd"]
            ),
            "subject_mean_correction_to_centered_joint_rms": float(
                metrics["correction_to_centered_joint_rms"].mean()
            ),
            "median_split_half_phase_correlation": float(
                np.nanmedian(metrics["split_half_phase_correlation"])
            ),
            "positive_split_half_subjects": int(
                np.sum(metrics["split_half_phase_correlation"] > 0.0)
            ),
            "subject_mean_within_subject_phase_delta_r2": float(
                metrics["within_subject_phase_delta_r2_vs_zero"].mean()
            ),
            "validation_train_phase_delta_r2_values": validation_phase.tolist(),
            "validation_train_phase_delta_r2_mean": float(validation_phase.mean()),
            "validation_train_phase_delta_r2_ci95": _percentile_interval(
                metric_bootstrap["validation_phase_delta_r2"]
            ),
            "validation_train_phase_positive_subjects": int(
                np.sum(validation_phase > 0.0)
            ),
            "median_subject_correlations": {
                state: float(
                    np.nanmedian(metrics[f"correction_correlation_{state}"])
                )
                for state in ("flow", "hbo", "hbr")
            },
        },
        "interpretation_boundary": (
            "The source teacher is subject-specific LOTO and previously failed "
            "physical-reconstruction/calibration gates. These results describe "
            "teacher target geometry only; they are not evidence of biological "
            "coupling or tokenizer co-occurrence."
        ),
    }

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True)
    tables_dir.mkdir()
    _write_csv(
        tables_dir / "subject_metrics.csv",
        list(subject_rows[0]),
        subject_rows,
    )
    _write_csv(
        tables_dir / "time_profile.csv",
        list(time_rows[0]),
        time_rows,
    )
    np.savez_compressed(
        args.output_dir / "subject_bootstrap_distributions.npz",
        schema=np.asarray("r1d_teacher_geometry_bootstrap_v1"),
        **metric_bootstrap,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.3),
        constrained_layout=True,
    )
    ax = axes[0, 0]
    ax.fill_between(
        times,
        profile_lower,
        profile_upper,
        color=OKABE_ITO["sky"],
        alpha=0.28,
        linewidth=0,
        label="95% subject bootstrap CI",
    )
    ax.plot(
        times,
        population_profile,
        color=OKABE_ITO["blue"],
        linewidth=1.5,
        label="Subject-equal mean",
    )
    ax.axvline(0.0, color="0.3", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="0.65", linewidth=0.6)
    ax.set_xlabel("Time relative to task onset (s)")
    ax.set_ylabel(r"$r^J-r^E$ (training SD)")
    ax.set_title("Joint teacher correction")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    vmax = float(np.percentile(np.abs(subject_profile), 99))
    image = ax.imshow(
        subject_profile,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(times[0], times[-1], len(subjects) + 0.5, 0.5),
        interpolation="nearest",
    )
    ax.axhline(train_count + 0.5, color="black", linewidth=1.0)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Time relative to task onset (s)")
    ax.set_ylabel("Subject")
    ax.set_yticks(np.arange(1, len(subjects) + 1, 2))
    ax.set_title("Trial-averaged correction by subject")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Training SD")

    ax = axes[1, 0]
    correlation_names = ("flow", "hbo", "hbr")
    correlation_colors = (
        OKABE_ITO["green"],
        OKABE_ITO["orange"],
        OKABE_ITO["purple"],
    )
    for position, (name, color) in enumerate(
        zip(correlation_names, correlation_colors),
        start=1,
    ):
        values_for_state = metrics[f"correction_correlation_{name}"]
        jitter = rng.normal(0.0, 0.035, size=len(values_for_state))
        ax.scatter(
            np.full(len(values_for_state), position) + jitter,
            values_for_state,
            s=13,
            alpha=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.25,
        )
        ax.plot(
            [position - 0.18, position + 0.18],
            [np.nanmedian(values_for_state)] * 2,
            color="black",
            linewidth=1.5,
        )
    ax.axhline(0.0, color="0.55", linewidth=0.7)
    ax.set_xticks([1, 2, 3], ["Flow", "HbO", "HbR"])
    ax.set_ylabel("Within-subject trajectory correlation")
    ax.set_title("Correction coordinate association (n=23)")

    ax = axes[1, 1]
    validation_x = np.arange(1, len(VALIDATION_SUBJECTS) + 1)
    ax.scatter(
        validation_x,
        validation_phase,
        color=OKABE_ITO["vermillion"],
        s=28,
        zorder=3,
    )
    ax.axhline(0.0, color="0.45", linewidth=0.8)
    ax.set_xticks(validation_x, [value[-2:] for value in VALIDATION_SUBJECTS])
    ax.set_xlabel("Held-out development subject")
    ax.set_ylabel(r"Phase-template $\Delta R^2$ vs zero")
    ax.set_title("Train-subject phase template transfer")

    for label, ax in zip(("A", "B", "C", "D"), axes.ravel()):
        ax.text(
            -0.13,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            figures_dir / f"r1d_teacher_geometry.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    readme = f"""# R1-D teacher geometry audit

This is a development-only diagnostic of the paired adaptive-joint and
EEG-only teacher trajectories. It is not promotion-eligible and does not
constitute evidence of biological coupling.

Key machine-readable results are in `summary.json`; subject-level and
time-resolved values are in `tables/`; subject-cluster bootstrap draws are
preserved in `subject_bootstrap_distributions.npz`.

Source trajectory SHA-256: `{summary["source"]["trajectories_sha256"]}`
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary["findings"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
