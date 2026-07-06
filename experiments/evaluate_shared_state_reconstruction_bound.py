#!/usr/bin/env python3
"""Estimate capacity-conditional EEG/fNIRS shared-state reconstruction bounds.

The validation-oracle PCA result is an algebraic in-sample lower bound only for
linear rank-k reconstruction in the declared representation. CCA and
train-fitted PCA results are achievable held-out-subject errors, not universal
information-theoretic lower bounds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "shared_state_reconstruction_bound_v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    path.write_text(json.dumps(convert(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_cache(root: Path, subject: int) -> Path:
    paths = sorted((root / f"subject_{subject}").glob("*.npz"))
    if len(paths) != 1:
        raise ValueError(f"expected one NPZ for subject {subject}, found {len(paths)}")
    return paths[0]


def _slice_bounds(config: Mapping[str, Any], fs: float, start_key: str, duration_key: str | None = None) -> tuple[int, int]:
    cache_start = float(config["cache_window_start_s"])
    start = int(round((float(config[start_key]) - cache_start) * fs))
    if duration_key is None:
        end = int(round((float(config["baseline_end_s"]) - cache_start) * fs))
    else:
        end = start + int(round(float(config[duration_key]) * fs))
    return start, end


def _patchify(signal: np.ndarray, patch_samples: int) -> np.ndarray:
    usable = signal.shape[0] // patch_samples * patch_samples
    return signal[:usable].reshape(-1, patch_samples, signal.shape[1]).transpose(0, 2, 1)


def _descriptor(patches: np.ndarray, spectral_bins: int) -> np.ndarray:
    n = patches.shape[-1]
    time = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    mean = patches.mean(axis=-1)
    std = patches.std(axis=-1)
    slope = np.sum(patches * time, axis=-1) / max(float(np.sum(time * time)), 1e-12)
    delta = patches[..., -1] - patches[..., 0]
    power = np.abs(np.fft.rfft(patches, axis=-1)) ** 2
    available = power[..., 1:]
    groups = np.array_split(np.arange(available.shape[-1]), min(spectral_bins, available.shape[-1]))
    band_power = np.stack([np.log1p(available[..., group].mean(axis=-1)) for group in groups], axis=-1)
    features = np.concatenate((mean[..., None], std[..., None], slope[..., None], delta[..., None], band_power), axis=-1)
    return features.reshape(features.shape[0], -1)


def _load_subject(path: Path, subject: int, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    eeg_fs = float(config["eeg_fs_hz"])
    fnirs_fs = float(config["fnirs_fs_hz"])
    eeg_analysis = _slice_bounds(config, eeg_fs, "analysis_start_s", "analysis_duration_s")
    fnirs_analysis = _slice_bounds(config, fnirs_fs, "analysis_start_s", "analysis_duration_s")
    eeg_baseline = _slice_bounds(config, eeg_fs, "baseline_start_s")
    fnirs_baseline = _slice_bounds(config, fnirs_fs, "baseline_start_s")
    eeg_patch = int(round(float(config["patch_duration_s"]) * eeg_fs))
    fnirs_patch = int(round(float(config["patch_duration_s"]) * fnirs_fs))
    rows: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as archive:
        prefixes = sorted(key[: -len("/obs_eeg")] for key in archive.files if key.endswith("/obs_eeg"))
        for prefix in prefixes:
            anchor, event = prefix.split("/")
            eeg = np.asarray(archive[f"{prefix}/obs_eeg"], dtype=np.float64)
            fnirs = np.column_stack([
                np.asarray(archive[f"{prefix}/obs_fnirs_optical_channel_{channel}"], dtype=np.float64).reshape(-1)
                for channel in config["fnirs_channels"]
            ])
            eeg = eeg - eeg[slice(*eeg_baseline)].mean(axis=0, keepdims=True)
            fnirs = fnirs - fnirs[slice(*fnirs_baseline)].mean(axis=0, keepdims=True)
            eeg_patches = _patchify(eeg[slice(*eeg_analysis)], eeg_patch)
            fnirs_patches = _patchify(fnirs[slice(*fnirs_analysis)], fnirs_patch)
            if len(eeg_patches) != len(fnirs_patches):
                raise ValueError(f"patch count mismatch for {subject}:{prefix}")
            eeg_desc = _descriptor(eeg_patches, spectral_bins=8)
            fnirs_desc = _descriptor(fnirs_patches, spectral_bins=5)
            for patch in range(len(eeg_patches)):
                rows.append({
                    "subject": subject,
                    "anchor": anchor,
                    "event": event,
                    "patch": patch,
                    "eeg_waveform": eeg_patches[patch].reshape(-1),
                    "fnirs_waveform": fnirs_patches[patch].reshape(-1),
                    "eeg_descriptor": eeg_desc[patch],
                    "fnirs_descriptor": fnirs_desc[patch],
                    "eeg_shape": eeg_patches[patch].shape,
                    "fnirs_shape": fnirs_patches[patch].shape,
                })
    return rows


def _stack(rows: Sequence[Mapping[str, Any]], modality: str, representation: str) -> np.ndarray:
    return np.stack([np.asarray(row[f"{modality}_{representation}"], dtype=np.float64) for row in rows])


def _fit_scalers(train_eeg: np.ndarray, train_fnirs: np.ndarray) -> tuple[StandardScaler, StandardScaler]:
    return StandardScaler().fit(train_eeg), StandardScaler().fit(train_fnirs)


def _weighted(eeg: np.ndarray, fnirs: np.ndarray) -> np.ndarray:
    return np.concatenate((eeg / np.sqrt(eeg.shape[1]), fnirs / np.sqrt(fnirs.shape[1])), axis=1)


def _split_weighted(joint: np.ndarray, eeg_dim: int, fnirs_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return joint[:, :eeg_dim] * np.sqrt(eeg_dim), joint[:, eeg_dim:] * np.sqrt(fnirs_dim)


def _pca_reconstruct(model: PCA, data: np.ndarray, dimension: int) -> np.ndarray:
    centered = data - model.mean_
    components = model.components_[:dimension]
    return centered @ components.T @ components + model.mean_


def _cca_latents(
    train_eeg: np.ndarray,
    train_fnirs: np.ndarray,
    val_eeg: np.ndarray,
    val_fnirs: np.ndarray,
    precompress: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eeg_count = min(precompress, train_eeg.shape[0] - 1, train_eeg.shape[1])
    fnirs_count = min(precompress, train_fnirs.shape[0] - 1, train_fnirs.shape[1])
    eeg_pca = PCA(n_components=eeg_count, svd_solver="randomized", random_state=1).fit(train_eeg)
    fnirs_pca = PCA(n_components=fnirs_count, svd_solver="randomized", random_state=2).fit(train_fnirs)
    train_eeg_white = eeg_pca.transform(train_eeg) / np.sqrt(np.maximum(eeg_pca.explained_variance_, 1e-12))
    train_fnirs_white = fnirs_pca.transform(train_fnirs) / np.sqrt(np.maximum(fnirs_pca.explained_variance_, 1e-12))
    val_eeg_white = eeg_pca.transform(val_eeg) / np.sqrt(np.maximum(eeg_pca.explained_variance_, 1e-12))
    val_fnirs_white = fnirs_pca.transform(val_fnirs) / np.sqrt(np.maximum(fnirs_pca.explained_variance_, 1e-12))
    cross = train_eeg_white.T @ train_fnirs_white / max(len(train_eeg_white) - 1, 1)
    left, correlations, right_t = np.linalg.svd(cross, full_matrices=False)
    right = right_t.T
    return (
        train_eeg_white @ left,
        train_fnirs_white @ right,
        val_eeg_white @ left,
        val_fnirs_white @ right,
        correlations,
    )


def _decode(train_z: np.ndarray, train_y: np.ndarray, val_z: np.ndarray, alpha: float) -> np.ndarray:
    return Ridge(alpha=alpha).fit(train_z, train_y).predict(val_z)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metric_row(
    truth_z: np.ndarray,
    prediction_z: np.ndarray,
    scaler: StandardScaler,
    *,
    modality: str,
    representation: str,
    shape: Sequence[int],
) -> dict[str, float]:
    error_z = prediction_z - truth_z
    mse = float(np.mean(error_z**2))
    r2 = 1.0 - float(np.sum(error_z**2)) / max(float(np.sum(truth_z**2)), 1e-12)
    truth = scaler.inverse_transform(truth_z)
    prediction = scaler.inverse_transform(prediction_z)
    relative_rmse = float(np.sqrt(np.mean((prediction - truth) ** 2) / max(np.mean(truth**2), 1e-12)))
    amplitude_ratio = float(np.std(prediction) / max(np.std(truth), 1e-12))
    output: dict[str, float] = {
        "standardized_mse": mse,
        "standardized_r2": r2,
        "relative_rmse": relative_rmse,
        "amplitude_ratio": amplitude_ratio,
        "abs_log_variance_ratio": float(abs(np.log(max(np.var(prediction), 1e-12) / max(np.var(truth), 1e-12)))),
    }
    if representation == "waveform":
        truth_wave = truth.reshape(len(truth), int(shape[0]), int(shape[1]))
        pred_wave = prediction.reshape(len(prediction), int(shape[0]), int(shape[1]))
        time = np.linspace(-1.0, 1.0, int(shape[1]))
        denom = max(float(np.sum(time * time)), 1e-12)
        truth_slope = np.sum(truth_wave * time, axis=-1) / denom
        pred_slope = np.sum(pred_wave * time, axis=-1) / denom
        output["patch_mean_correlation"] = _safe_corr(truth_wave.mean(axis=-1), pred_wave.mean(axis=-1))
        output["patch_slope_correlation"] = _safe_corr(truth_slope, pred_slope)
        output["patch_std_ratio"] = float(pred_wave.std(axis=-1).mean() / max(truth_wave.std(axis=-1).mean(), 1e-12))
    return output


def _bootstrap_ci(values: Mapping[int, Sequence[float]], iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    subjects = np.asarray(sorted(values))
    if len(subjects) == 0:
        return float("nan"), float("nan")
    subject_means = np.asarray([np.mean(values[int(subject)]) for subject in subjects])
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = rng.choice(subject_means, size=len(subject_means), replace=True).mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _anchor_analysis(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    representation: str,
    dimensions: Sequence[int],
    precompress: int,
    alpha: float,
) -> list[dict[str, Any]]:
    train_eeg_raw = _stack(train_rows, "eeg", representation)
    train_fnirs_raw = _stack(train_rows, "fnirs", representation)
    val_eeg_raw = _stack(val_rows, "eeg", representation)
    val_fnirs_raw = _stack(val_rows, "fnirs", representation)
    eeg_scaler, fnirs_scaler = _fit_scalers(train_eeg_raw, train_fnirs_raw)
    train_eeg = eeg_scaler.transform(train_eeg_raw)
    train_fnirs = fnirs_scaler.transform(train_fnirs_raw)
    val_eeg = eeg_scaler.transform(val_eeg_raw)
    val_fnirs = fnirs_scaler.transform(val_fnirs_raw)
    train_joint = _weighted(train_eeg, train_fnirs)
    val_joint = _weighted(val_eeg, val_fnirs)
    max_dim = min(max(dimensions), train_joint.shape[0] - 1, train_joint.shape[1])
    train_pca = PCA(n_components=max_dim, svd_solver="randomized", random_state=3).fit(train_joint)
    val_oracle_count = min(max(dimensions), val_joint.shape[0] - 1, val_joint.shape[1])
    val_oracle = PCA(n_components=val_oracle_count, svd_solver="randomized", random_state=4).fit(val_joint)
    separate_eeg_count = min(max(dimensions), train_eeg.shape[0] - 1, train_eeg.shape[1])
    separate_fnirs_count = min(max(dimensions), train_fnirs.shape[0] - 1, train_fnirs.shape[1])
    separate_eeg_pca = PCA(
        n_components=separate_eeg_count, svd_solver="randomized", random_state=5
    ).fit(train_eeg)
    separate_fnirs_pca = PCA(
        n_components=separate_fnirs_count, svd_solver="randomized", random_state=6
    ).fit(train_fnirs)
    train_ce, train_cf, val_ce, val_cf, canonical_correlations = _cca_latents(
        train_eeg, train_fnirs, val_eeg, val_fnirs, precompress
    )
    rows: list[dict[str, Any]] = []
    shapes = {
        "eeg": train_rows[0]["eeg_shape"],
        "fnirs": train_rows[0]["fnirs_shape"],
    }
    truth = {"eeg": val_eeg, "fnirs": val_fnirs}
    scalers = {"eeg": eeg_scaler, "fnirs": fnirs_scaler}
    subjects = np.asarray([int(row["subject"]) for row in val_rows])

    def add(model: str, dimension: int, pred_eeg: np.ndarray, pred_fnirs: np.ndarray, extra: Mapping[str, Any] | None = None) -> None:
        for modality, prediction in (("eeg", pred_eeg), ("fnirs", pred_fnirs)):
            metrics = _metric_row(
                truth[modality], prediction, scalers[modality], modality=modality,
                representation=representation, shape=shapes[modality],
            )
            row: dict[str, Any] = {
                "anchor": train_rows[0]["anchor"],
                "representation": representation,
                "model": model,
                "latent_dimension": dimension,
                "modality": modality,
                "rows": len(val_rows),
                **metrics,
            }
            if extra:
                row.update(extra)
            for subject in np.unique(subjects):
                mask = subjects == subject
                subject_metrics = _metric_row(
                    truth[modality][mask], prediction[mask], scalers[modality], modality=modality,
                    representation=representation, shape=shapes[modality],
                )
                row[f"subject_{subject}_standardized_mse"] = subject_metrics["standardized_mse"]
            rows.append(row)

    for dimension in dimensions:
        k = min(int(dimension), max_dim)
        oracle_k = min(int(dimension), val_oracle_count)
        oracle_eeg, oracle_fnirs = _split_weighted(
            _pca_reconstruct(val_oracle, val_joint, oracle_k), val_eeg.shape[1], val_fnirs.shape[1]
        )
        add("validation_oracle_joint_pca", int(dimension), oracle_eeg, oracle_fnirs)

        fitted_eeg, fitted_fnirs = _split_weighted(
            _pca_reconstruct(train_pca, val_joint, k), val_eeg.shape[1], val_fnirs.shape[1]
        )
        component_energy_eeg = np.sum(train_pca.components_[:k, : val_eeg.shape[1]] ** 2, axis=1)
        component_energy_fnirs = np.sum(train_pca.components_[:k, val_eeg.shape[1] :] ** 2, axis=1)
        balance = np.minimum(component_energy_eeg, component_energy_fnirs) / np.maximum(
            np.maximum(component_energy_eeg, component_energy_fnirs), 1e-12
        )
        add(
            "train_fitted_joint_pca", int(dimension), fitted_eeg, fitted_fnirs,
            {"median_cross_modal_loading_balance": float(np.median(balance))},
        )

        separate_eeg = _pca_reconstruct(separate_eeg_pca, val_eeg, min(int(dimension), separate_eeg_count))
        separate_fnirs = _pca_reconstruct(separate_fnirs_pca, val_fnirs, min(int(dimension), separate_fnirs_count))
        add("separate_modality_pca_k_each", int(dimension), separate_eeg, separate_fnirs)

        cca_k = min(int(dimension), train_ce.shape[1])
        train_joint_z = 0.5 * (train_ce[:, :cca_k] + train_cf[:, :cca_k])
        val_joint_z = 0.5 * (val_ce[:, :cca_k] + val_cf[:, :cca_k])
        mean_corr = float(np.mean(canonical_correlations[:cca_k]))
        validation_correlations = np.asarray([
            _safe_corr(val_ce[:, index], val_cf[:, index]) for index in range(cca_k)
        ])
        finite_validation_correlations = validation_correlations[np.isfinite(validation_correlations)]
        validation_corr = float(np.mean(finite_validation_correlations)) if finite_validation_correlations.size else float("nan")
        validation_positive_fraction = float(np.mean(finite_validation_correlations > 0)) if finite_validation_correlations.size else float("nan")
        cca_diagnostics = {
            "mean_train_canonical_correlation": mean_corr,
            "mean_validation_canonical_correlation": validation_corr,
            "positive_validation_canonical_fraction": validation_positive_fraction,
        }
        add(
            "cca_joint_shared", int(dimension),
            _decode(train_joint_z, train_eeg, val_joint_z, alpha),
            _decode(train_joint_z, train_fnirs, val_joint_z, alpha),
            cca_diagnostics,
        )
        add(
            "cca_eeg_inferred", int(dimension),
            _decode(train_ce[:, :cca_k], train_eeg, val_ce[:, :cca_k], alpha),
            _decode(train_ce[:, :cca_k], train_fnirs, val_ce[:, :cca_k], alpha),
            cca_diagnostics,
        )
        add(
            "cca_fnirs_inferred", int(dimension),
            _decode(train_cf[:, :cca_k], train_eeg, val_cf[:, :cca_k], alpha),
            _decode(train_cf[:, :cca_k], train_fnirs, val_cf[:, :cca_k], alpha),
            cca_diagnostics,
        )
    return rows


def _aggregate(anchor_rows: Sequence[Mapping[str, Any]], bootstrap_iterations: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in anchor_rows:
        key = (row["representation"], row["model"], row["latent_dimension"], row["modality"])
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    metric_names = (
        "standardized_mse", "standardized_r2", "relative_rmse", "amplitude_ratio",
        "abs_log_variance_ratio", "patch_mean_correlation", "patch_slope_correlation", "patch_std_ratio",
        "median_cross_modal_loading_balance", "mean_train_canonical_correlation",
        "mean_validation_canonical_correlation", "positive_validation_canonical_fraction",
    )
    for key, rows in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "representation": key[0], "model": key[1], "latent_dimension": key[2], "modality": key[3],
            "anchors": len(rows), "validation_rows": int(sum(int(row["rows"]) for row in rows)),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows if metric in row and np.isfinite(float(row[metric]))]
            if values:
                aggregate[metric] = float(np.mean(values))
        subject_values: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            for field, value in row.items():
                if field.startswith("subject_") and field.endswith("_standardized_mse"):
                    subject = int(field.split("_")[1])
                    subject_values[subject].append(float(value))
        lower, upper = _bootstrap_ci(subject_values, bootstrap_iterations, rng)
        aggregate["standardized_mse_subject_bootstrap_ci_low"] = lower
        aggregate["standardized_mse_subject_bootstrap_ci_high"] = upper
        output.append(aggregate)
    return output


def _plot(summary_rows: Sequence[Mapping[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    colors = {
        "validation_oracle_joint_pca": "#111827",
        "train_fitted_joint_pca": "#2563eb",
        "separate_modality_pca_k_each": "#16a34a",
        "cca_joint_shared": "#dc2626",
        "cca_eeg_inferred": "#f59e0b",
        "cca_fnirs_inferred": "#7c3aed",
    }
    for representation in sorted({str(row["representation"]) for row in summary_rows}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for axis, modality in zip(axes, ("eeg", "fnirs")):
            selected = [row for row in summary_rows if row["representation"] == representation and row["modality"] == modality]
            for model in colors:
                model_rows = sorted((row for row in selected if row["model"] == model), key=lambda row: int(row["latent_dimension"]))
                if not model_rows:
                    continue
                axis.plot(
                    [row["latent_dimension"] for row in model_rows],
                    [max(float(row["standardized_mse"]), 1e-6) for row in model_rows],
                    marker="o", linewidth=1.8, label=model.replace("_", " "), color=colors[model],
                )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            dimensions = sorted({int(row["latent_dimension"]) for row in selected})
            axis.set_xticks(dimensions, labels=[str(value) for value in dimensions])
            axis.set_xlabel("latent dimension")
            axis.set_ylabel("held-out standardized MSE (display floor 1e-6)")
            axis.set_title(modality.upper())
            axis.grid(alpha=0.25)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=8)
        fig.suptitle(f"Capacity-conditional reconstruction: {representation}")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.82, wspace=0.22)
        for suffix, dpi in (("svg", None), ("png", 300)):
            path = run_dir / "figures" / f"capacity_curve_{representation}.{suffix}"
            fig.savefig(path, dpi=dpi)
            artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
        plt.close(fig)
    return artifacts


def _summary_markdown(summary_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> str:
    dimensions = set(int(value) for value in config["analysis"]["latent_dimensions"])
    focus = 5 if 5 in dimensions else min(dimensions)
    lines = [
        "# Shared-state reconstruction-bound diagnostic", "",
        "This run did not access protected-test subjects and does not change the E0 gate.", "",
        "The validation-oracle PCA is a lower bound only within rank-k linear reconstruction on the observed validation matrix. "
        "CCA and train-fitted results are held-out achievable errors, not universal lower bounds.", "",
        f"## Five-dimensional reference (k={focus})", "",
        "| Representation | Model | Modality | standardized MSE | R2 | amplitude ratio |", "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if int(row["latent_dimension"]) != focus:
            continue
        if row["model"] not in {"validation_oracle_joint_pca", "train_fitted_joint_pca", "separate_modality_pca_k_each", "cca_joint_shared", "cca_eeg_inferred", "cca_fnirs_inferred"}:
            continue
        lines.append(
            f"| {row['representation']} | {row['model']} | {row['modality']} | "
            f"{row['standardized_mse']:.6f} | {row['standardized_r2']:.6f} | {row['amplitude_ratio']:.6f} |"
        )
    lines.extend(["", "See `summary.json`, `metrics.csv`, and the capacity figures for the complete curves.", ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    analysis = config["analysis"]
    train_subjects = [int(value) for value in data["train_subjects"]]
    val_subjects = [int(value) for value in data["validation_subjects"]]
    protected = {int(value) for value in data["protected_test_subjects"]}
    if set(train_subjects) & set(val_subjects) or (set(train_subjects) | set(val_subjects)) & protected:
        raise ValueError("train, validation, and protected subject sets must be disjoint")
    root = (REPO_ROOT / data["cache_root"]).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() if args.output_dir else (
        REPO_ROOT / "experiments" / "runs" / "physiology_semantic_tokenizer" / "e0_teacher_validity" /
        f"{stamp}_{config['experiment']['name']}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "figure_data").mkdir()
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "decision_protocol.yaml").write_text(yaml.safe_dump({
        "schema": SCHEMA,
        "metric_role": "diagnostic_non_gate",
        "protected_test_status": "closed",
        "lower_bound_scope": "validation-matrix rank-k linear reconstruction only",
        "generalization_scope": "subject-held-out validation subjects 19-23",
        "primary_contrast": "shared-only CCA versus separate-modality PCA at matched k",
        "interpretation_rule": "shared-only error cannot be used as an absolute biological noise floor",
    }, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "metric_registry.json", {
        "schema": SCHEMA,
        "primary": ["standardized_mse", "standardized_r2"],
        "secondary": ["relative_rmse", "amplitude_ratio", "patch_mean_correlation", "patch_slope_correlation"],
        "diagnostic": ["cross_modal_loading_balance", "canonical_correlation", "oracle_generalization_gap"],
    })

    all_rows: list[dict[str, Any]] = []
    input_files = []
    for split, subjects in (("train", train_subjects), ("validation", val_subjects)):
        for subject in subjects:
            path = _subject_cache(root, subject)
            input_files.append({"split": split, "subject": subject, "path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)})
            all_rows.extend(_load_subject(path, subject, data))
    train_rows = [row for row in all_rows if int(row["subject"]) in set(train_subjects)]
    val_rows = [row for row in all_rows if int(row["subject"]) in set(val_subjects)]
    anchors = sorted({str(row["anchor"]) for row in all_rows})
    anchor_metrics: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_train = [row for row in train_rows if row["anchor"] == anchor]
        anchor_val = [row for row in val_rows if row["anchor"] == anchor]
        for representation in analysis["representations"]:
            anchor_metrics.extend(_anchor_analysis(
                anchor_train, anchor_val, str(representation),
                [int(value) for value in analysis["latent_dimensions"]],
                int(analysis["cca_precompression_dimension"]), float(analysis["ridge_alpha"]),
            ))
    summary_rows = _aggregate(
        anchor_metrics, int(analysis["subject_bootstrap_iterations"]), int(analysis["seed"])
    )
    _write_csv(run_dir / "anchor_metrics.csv", anchor_metrics)
    _write_csv(run_dir / "metrics.csv", summary_rows)
    figure_artifacts = _plot(summary_rows, run_dir)
    focus_rows = [row for row in summary_rows if int(row["latent_dimension"]) == 5]
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "formal_complete_diagnostic",
        "protected_test_used": False,
        "lower_bound_scope": "rank-k linear validation-oracle; conditional on representation, baseline correction, and anchor-specific decoder",
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "protected_test_subjects_unopened": sorted(protected),
        "anchors": anchors,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "five_dimensional_results": focus_rows,
        "figures": figure_artifacts,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "cache_manifest_hashes.json", {"files": input_files})
    _write_json(run_dir / "evidence_calibration.json", {
        "schema": SCHEMA,
        "latent_dimensions": analysis["latent_dimensions"],
        "subject_bootstrap_iterations": analysis["subject_bootstrap_iterations"],
        "seed": analysis["seed"],
        "thresholds": "none; capacity curves and subject-bootstrap intervals are diagnostic",
        "protected_test_used": False,
    })
    _write_json(run_dir / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "git_status_porcelain": subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines(),
    })
    _write_json(run_dir / "manifest.json", {
        "schema": SCHEMA,
        "status": "formal_complete",
        "metric_role": "diagnostic",
        "protected_test_used": False,
        "config": "config.yaml",
        "summary": "summary.json",
        "metrics": "metrics.csv",
    })
    (run_dir / "summary.md").write_text(_summary_markdown(summary_rows, config), encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/shared_state_reconstruction_bound.yaml"))
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
