#!/usr/bin/env python3
"""Evaluate an adaptive Croce-linearized fixed-interval EEG-fNIRS smoother.

This experiment keeps a five-state physiological transition, restores the
local two-fNIRS/six-EEG protocol, and compares it with an all-EEG spatial
ablation.  Every held-out trial is excluded from channel selection, EEG-adapter
fitting, hemodynamic-parameter fitting, and modality-balance selection.

The joint smoother consumes held-out EEG, HbO, and HbR and is therefore a
two-modality compromise diagnostic.  The EEG-only smoother is retained as the
strict cross-modal test; joint reconstruction alone is not evidence that EEG
can predict fNIRS.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_shared_neural_driver_unified import (
    Trial,
    _downsample_eeg_power,
    _load_trials,
    _safe_corr,
    _select_active_hbo,
    _waveform_metrics,
    _write_csv,
    _write_json,
)
from src.inference.adaptive_neurovascular_ssm import (
    AdaptiveSSMFit,
    apply_adaptive_ssm,
    fit_adaptive_ssm,
    fit_to_mapping,
    measurement_aligned_state_gauge,
)
from src.metrics.trajectory_reliability import trajectory_reliability_metrics


SCHEMA = "adaptive_shared_neural_ssm_v2"
STATE_NAMES = ("vasodilation_s", "flow_delta", "hbo_state", "hbr_state", "shared_driver")


@dataclass(frozen=True)
class EEGAdapter:
    indices: np.ndarray
    channel_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    pca_mean: np.ndarray
    loading: np.ndarray
    pc_scale: float


@dataclass
class AdaptivePrediction:
    condition_id: str
    dataset_id: str
    subject: str
    heldout_trial: int
    model: str
    spatial_mode: str
    truth_hbo: np.ndarray
    estimate_hbo: np.ndarray
    truth_hbr: np.ndarray
    estimate_hbr: np.ndarray
    eeg_observation: np.ndarray
    eeg_reconstruction: np.ndarray
    observation_predictive_std: np.ndarray
    eeg_valid_mask: np.ndarray
    fnirs_valid_mask: np.ndarray
    states: np.ndarray
    state_std: np.ndarray
    target_states: np.ndarray
    target_state_std: np.ndarray
    gauge_scales: np.ndarray
    gauge_offsets: np.ndarray
    gauge_reconstruction_max_abs_delta: float
    selected_fnirs_channels: tuple[str, ...]
    selected_eeg_channels: tuple[str, ...]


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


def _paired_hbr_indices(trial: Trial, hbo_indices: np.ndarray) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(trial.fnirs_channel_names)}
    output = []
    for hbo_index in np.asarray(hbo_indices, dtype=int):
        hbo_name = trial.fnirs_channel_names[int(hbo_index)]
        candidates = (
            hbo_name.removesuffix("_HbO") + "_HbR",
            hbo_name.removesuffix("HbO") + "HbR",
        )
        match = next((name_to_index[name] for name in candidates if name in name_to_index), None)
        if match is None:
            position = None if trial.fnirs_positions is None else trial.fnirs_positions[int(hbo_index)]
            hbr_indices = np.flatnonzero(np.asarray(trial.fnirs_roles, dtype=object) == "HbR")
            if position is None or not np.all(np.isfinite(position)):
                raise ValueError(f"cannot pair HbR channel for {hbo_name}")
            distances = np.linalg.norm(trial.fnirs_positions[hbr_indices] - position[None, :], axis=1)
            match = int(hbr_indices[int(np.argmin(distances))])
        output.append(int(match))
    return np.asarray(output, dtype=int)


def _local_eeg_indices(trial: Trial, hbo_indices: np.ndarray, count: int) -> np.ndarray:
    if trial.eeg_positions is None or trial.fnirs_positions is None:
        raise ValueError("local spatial mode requires unified channel geometry")
    anchors = np.asarray(trial.fnirs_positions[np.asarray(hbo_indices, dtype=int)], dtype=np.float64)
    eeg_positions = np.asarray(trial.eeg_positions, dtype=np.float64)
    finite_anchors = anchors[np.all(np.isfinite(anchors), axis=1)]
    if not len(finite_anchors):
        raise ValueError("selected fNIRS anchors have no finite positions")
    # Distance to the nearest selected fNIRS anchor preserves both local fields
    # when the two active channels are not colocated.
    distances = np.min(
        np.linalg.norm(eeg_positions[:, None, :] - finite_anchors[None, :, :], axis=2),
        axis=1,
    )
    distances[~np.all(np.isfinite(eeg_positions), axis=1)] = np.inf
    selected = np.argsort(distances)[: int(count)]
    if np.any(~np.isfinite(distances[selected])):
        raise ValueError("not enough EEG channels with finite geometry")
    return np.asarray(selected, dtype=int)


def _eeg_log_power(eeg: np.ndarray) -> np.ndarray:
    power = _downsample_eeg_power(np.asarray(eeg, dtype=np.float64))
    channel_floor = np.maximum(np.median(power, axis=0, keepdims=True) * 1e-6, 1e-12)
    return np.log(np.maximum(power, channel_floor))


def _fit_eeg_adapter(trials: Sequence[Trial], indices: np.ndarray) -> tuple[EEGAdapter, list[np.ndarray]]:
    features = [_eeg_log_power(trial.eeg)[:, indices] for trial in trials]
    stacked = np.concatenate(features, axis=0)
    feature_mean = np.mean(stacked, axis=0)
    feature_std = np.maximum(np.std(stacked, axis=0), 1e-8)
    normalized = [(value - feature_mean) / feature_std for value in features]
    normalized_stacked = np.concatenate(normalized, axis=0)
    pca_mean = np.mean(normalized_stacked, axis=0)
    _, _, vt = np.linalg.svd(normalized_stacked - pca_mean, full_matrices=False)
    loading = np.asarray(vt[0], dtype=np.float64)
    if float(np.sum(loading)) < 0.0:
        loading *= -1.0
    raw = [(value - pca_mean) @ loading for value in normalized]
    pc_scale = max(float(np.std(np.concatenate(raw))), 1e-8)
    drivers = [np.asarray(value / pc_scale, dtype=np.float64) for value in raw]
    names = tuple(trials[0].eeg_channel_names[int(index)] for index in indices)
    return EEGAdapter(
        indices=np.asarray(indices, dtype=int),
        channel_names=names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        pca_mean=pca_mean,
        loading=loading,
        pc_scale=pc_scale,
    ), drivers


def _apply_eeg_adapter(trial: Trial, adapter: EEGAdapter) -> np.ndarray:
    features = _eeg_log_power(trial.eeg)[:, adapter.indices]
    normalized = (features - adapter.feature_mean) / adapter.feature_std
    return np.asarray(((normalized - adapter.pca_mean) @ adapter.loading) / adapter.pc_scale, dtype=np.float64)


def _downsample_valid_mask(mask: np.ndarray | None, target_length: int) -> np.ndarray:
    """Require every source sample contributing to a 10 Hz point to be valid."""

    if mask is None:
        return np.ones(int(target_length), dtype=bool)
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if target_length <= 0 or len(values) % int(target_length):
        raise ValueError("EEG validity mask does not align to the 10 Hz driver")
    factor = len(values) // int(target_length)
    return values.reshape(int(target_length), factor).all(axis=1)


def _trial_valid_masks(trial: Trial) -> tuple[np.ndarray, np.ndarray]:
    """Validate the full-support contract of the current 20 s SSM runner."""

    target_length = len(trial.fnirs)
    eeg_mask = _downsample_valid_mask(trial.eeg_valid_mask, target_length)
    fnirs_mask = (
        np.ones(target_length, dtype=bool)
        if trial.fnirs_valid_mask is None
        else np.asarray(trial.fnirs_valid_mask, dtype=bool).copy()
    )
    if fnirs_mask.shape != (target_length,):
        raise ValueError("fNIRS validity mask does not match the SSM target")
    if not np.all(eeg_mask) or not np.all(fnirs_mask):
        raise RuntimeError(
            "adaptive SSM core evaluation requires fully supported EEG/fNIRS windows"
        )
    if not np.all(np.isfinite(trial.eeg)) or not np.all(np.isfinite(trial.fnirs)):
        raise RuntimeError("adaptive SSM core evaluation requires finite observations")
    return eeg_mask, fnirs_mask


def _chromophore_targets(
    trials: Sequence[Trial],
    hbo_indices: np.ndarray,
    hbr_indices: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    hbo = [np.mean(trial.fnirs[:, hbo_indices], axis=1, dtype=np.float64) for trial in trials]
    hbr = [np.mean(trial.fnirs[:, hbr_indices], axis=1, dtype=np.float64) for trial in trials]
    return hbo, hbr


def _fit_model(
    eeg_drivers: Sequence[np.ndarray],
    hbo_targets: Sequence[np.ndarray],
    hbr_targets: Sequence[np.ndarray],
    config: Mapping[str, Any],
    baseline_samples: int,
) -> AdaptiveSSMFit:
    return fit_adaptive_ssm(
        eeg_drivers,
        hbo_targets,
        hbr_targets,
        fs_hz=float(config["fs_hz"]),
        prior_strength=float(config["prior_strength"]),
        max_iterations=int(config["max_iterations"]),
        q_scale_candidates=tuple(float(value) for value in config["q_scale_candidates"]),
        fnirs_noise_scale_candidates=tuple(float(value) for value in config["fnirs_noise_scale_candidates"]),
        balance_penalty=float(config["balance_penalty"]),
        baseline_samples=int(baseline_samples),
        max_flow_perturbation=float(config["max_flow_perturbation"]),
    )


def _run_subject(
    condition_id: str,
    subject_index: int,
    subject: str,
    trial_values: Sequence[Trial],
    data_config: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> tuple[list[AdaptivePrediction], list[dict[str, Any]]]:
    predictions: list[AdaptivePrediction] = []
    fit_rows: list[dict[str, Any]] = []
    trials = list(trial_values)
    trial_masks = [_trial_valid_masks(trial) for trial in trials]
    with threadpool_limits(limits=1):
        for heldout in range(len(trials)):
            train_indices = [index for index in range(len(trials)) if index != heldout]
            train_trials = [trials[index] for index in train_indices]
            test_trial = trials[heldout]
            hbo_indices, hbo_names, _ = _select_active_hbo(
                train_trials,
                baseline_duration_s=float(data_config["baseline_duration_s"]),
                task_duration_s=float(data_config["task_duration_s"]),
                count=int(analysis["fnirs_active_hbo_channels"]),
            )
            hbr_indices = _paired_hbr_indices(train_trials[0], hbo_indices)
            hbr_names = tuple(train_trials[0].fnirs_channel_names[int(index)] for index in hbr_indices)
            train_hbo, train_hbr = _chromophore_targets(train_trials, hbo_indices, hbr_indices)
            test_hbo, test_hbr = _chromophore_targets([test_trial], hbo_indices, hbr_indices)
            for spatial_mode in analysis["spatial_modes"]:
                if spatial_mode == "local":
                    eeg_indices = _local_eeg_indices(
                        train_trials[0], hbo_indices, int(analysis["local_eeg_channels"]),
                    )
                elif spatial_mode == "global":
                    # Some unified datasets retain EOG/ECG channels in the
                    # record.  The all-EEG ablation must still mean scalp EEG.
                    eeg_indices = np.asarray([
                        index for index, name in enumerate(train_trials[0].eeg_channel_names)
                        if not any(token in name.upper() for token in ("EOG", "ECG", "EMG"))
                    ], dtype=int)
                else:
                    raise ValueError(f"unsupported spatial mode: {spatial_mode}")
                adapter, train_drivers = _fit_eeg_adapter(train_trials, eeg_indices)
                test_driver = _apply_eeg_adapter(test_trial, adapter)
                fit = _fit_model(
                    train_drivers,
                    train_hbo,
                    train_hbr,
                    analysis["ssm"],
                    int(round(float(data_config["baseline_duration_s"]) * float(analysis["ssm"]["fs_hz"]))),
                )
                outputs = {
                    "adaptive_joint": apply_adaptive_ssm(
                        test_driver,
                        fit,
                        hbo_observation=test_hbo[0],
                        hbr_observation=test_hbr[0],
                    ),
                    "adaptive_eeg_only": apply_adaptive_ssm(test_driver, fit),
                }
                eeg_valid_mask, fnirs_valid_mask = trial_masks[heldout]
                if len(eeg_valid_mask) != len(test_driver):
                    raise ValueError("EEG validity mask does not match the SSM driver")
                for model, result in outputs.items():
                    gauge = measurement_aligned_state_gauge(result, fit)
                    predictions.append(AdaptivePrediction(
                        condition_id=condition_id,
                        dataset_id=test_trial.dataset_id,
                        subject=subject,
                        heldout_trial=heldout,
                        model=model,
                        spatial_mode=str(spatial_mode),
                        truth_hbo=test_hbo[0],
                        estimate_hbo=result.hbo_reconstructed,
                        truth_hbr=test_hbr[0],
                        estimate_hbr=result.hbr_reconstructed,
                        eeg_observation=test_driver,
                        eeg_reconstruction=result.eeg_reconstructed,
                        observation_predictive_std=result.observation_predictive_std,
                        eeg_valid_mask=eeg_valid_mask,
                        fnirs_valid_mask=fnirs_valid_mask,
                        states=result.states,
                        state_std=result.state_std,
                        target_states=gauge.states,
                        target_state_std=gauge.state_std,
                        gauge_scales=gauge.scales,
                        gauge_offsets=gauge.offsets,
                        gauge_reconstruction_max_abs_delta=gauge.reconstruction_max_abs_delta,
                        selected_fnirs_channels=tuple(hbo_names + hbr_names),
                        selected_eeg_channels=adapter.channel_names,
                    ))
                fit_rows.append({
                    "condition_id": condition_id,
                    "subject": subject,
                    "heldout_trial": heldout,
                    "spatial_mode": spatial_mode,
                    "selected_fnirs_channels": "|".join(hbo_names + hbr_names),
                    "selected_eeg_channels": "|".join(adapter.channel_names),
                    **fit_to_mapping(fit),
                })
    return predictions, fit_rows


def _turning_points(signal: np.ndarray) -> int:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    prominence = max(float(np.std(values)) * 0.20, 1e-8)
    peaks, _ = find_peaks(values, prominence=prominence)
    troughs, _ = find_peaks(-values, prominence=prominence)
    return int(len(peaks) + len(troughs))


def _low_frequency_fraction(signal: np.ndarray, fs_hz: float = 10.0, threshold_hz: float = 0.075) -> float:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    values = values - np.mean(values)
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / fs_hz)
    power = np.abs(np.fft.rfft(values)) ** 2
    admitted = (frequencies >= 0.01) & (frequencies <= 0.20)
    slow = admitted & (frequencies <= threshold_hz)
    return float(np.sum(power[slow]) / max(float(np.sum(power[admitted])), 1e-12))


def _prediction_metrics(prediction: AdaptivePrediction, baseline_n: int) -> dict[str, Any]:
    hbo = _waveform_metrics(prediction.truth_hbo, prediction.estimate_hbo, baseline_n)
    hbr = _waveform_metrics(prediction.truth_hbr, prediction.estimate_hbr, baseline_n)
    eeg = _waveform_metrics(prediction.eeg_observation, prediction.eeg_reconstruction, baseline_n)
    hbo_reliability = trajectory_reliability_metrics(
        prediction.truth_hbo,
        prediction.estimate_hbo,
        predictive_std=prediction.observation_predictive_std[:, 1],
        valid_mask=prediction.fnirs_valid_mask,
    )
    hbr_reliability = trajectory_reliability_metrics(
        prediction.truth_hbr,
        prediction.estimate_hbr,
        predictive_std=prediction.observation_predictive_std[:, 2],
        valid_mask=prediction.fnirs_valid_mask,
    )
    eeg_reliability = trajectory_reliability_metrics(
        prediction.eeg_observation,
        prediction.eeg_reconstruction,
        predictive_std=prediction.observation_predictive_std[:, 0],
        valid_mask=prediction.eeg_valid_mask,
    )
    driver = prediction.states[:, 4]
    return {
        "condition_id": prediction.condition_id,
        "dataset_id": prediction.dataset_id,
        "subject": prediction.subject,
        "validation": "leave_one_trial",
        "heldout_trial": prediction.heldout_trial,
        "model": prediction.model,
        "spatial_mode": prediction.spatial_mode,
        "selected_fnirs_channels": "|".join(prediction.selected_fnirs_channels),
        "selected_eeg_channels": "|".join(prediction.selected_eeg_channels),
        **hbo,
        **hbo_reliability,
        **{f"hbr_{key}": value for key, value in hbr.items()},
        **{f"hbr_{key}": value for key, value in hbr_reliability.items()},
        **{f"eeg_{key}": value for key, value in eeg.items()},
        **{f"eeg_{key}": value for key, value in eeg_reliability.items()},
        "hbo_turning_points_truth": _turning_points(prediction.truth_hbo),
        "hbo_turning_points_estimate": _turning_points(prediction.estimate_hbo),
        "driver_turning_points": _turning_points(driver),
        "driver_monotonic_fraction": float(max(np.mean(np.diff(driver) >= 0.0), np.mean(np.diff(driver) <= 0.0))),
        "hbo_low_frequency_fraction_truth": _low_frequency_fraction(prediction.truth_hbo),
        "hbo_low_frequency_fraction_estimate": _low_frequency_fraction(prediction.estimate_hbo),
        "relative_flow_min": float(np.min(1.0 + prediction.states[:, 1])),
        "relative_flow_nonpositive_fraction": float(np.mean(1.0 + prediction.states[:, 1] <= 0.0)),
    }


def _aggregate_metrics(
    fold_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def finite_mean(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(np.mean(finite)) if len(finite) else float("nan")

    identifiers = {
        "condition_id", "dataset_id", "subject", "validation", "heldout_trial", "model", "spatial_mode",
        "selected_fnirs_channels", "selected_eeg_channels",
    }
    metric_names = [
        key for key, value in fold_rows[0].items()
        if key not in identifiers and isinstance(value, (float, int, np.floating, np.integer))
    ]
    per_subject: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        per_subject[(str(row["condition_id"]), str(row["subject"]), str(row["model"]), str(row["spatial_mode"]))].append(row)
    subject_rows = []
    for key, values in sorted(per_subject.items()):
        row: dict[str, Any] = {
            "condition_id": key[0], "subject": key[1], "model": key[2], "spatial_mode": key[3], "folds": len(values),
        }
        for metric in metric_names:
            row[metric] = finite_mean([float(value[metric]) for value in values])
        subject_rows.append(row)

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        groups[(str(row["condition_id"]), str(row["model"]), str(row["spatial_mode"]))].append(row)
    rng = np.random.default_rng(seed)
    summary_rows = []
    reliability_ci_suffixes = {
        "trajectory_deviation_nrmse",
        "temporal_sd_ratio",
        "posterior_predictive_sd_mean",
        "standardized_residual_rms",
        "predictive_95_coverage",
    }
    ci_metrics = {"r2", "pcc", "variance_ratio", "eeg_r2", "eeg_pcc"}
    ci_metrics.update(
        metric
        for metric in metric_names
        if any(
            metric == suffix or metric.endswith(f"_{suffix}")
            for suffix in reliability_ci_suffixes
        )
    )
    for key, values in sorted(groups.items()):
        row = {"condition_id": key[0], "model": key[1], "spatial_mode": key[2], "subjects": len(values)}
        for metric in metric_names:
            observed = np.asarray([float(value[metric]) for value in values], dtype=np.float64)
            finite_observed = observed[np.isfinite(observed)]
            row[metric] = finite_mean(observed)
            if metric in ci_metrics:
                if not len(finite_observed):
                    row[f"{metric}_ci_low"] = float("nan")
                    row[f"{metric}_ci_high"] = float("nan")
                    continue
                draws = np.empty(int(bootstrap_iterations), dtype=np.float64)
                for iteration in range(len(draws)):
                    draws[iteration] = float(
                        np.mean(
                            rng.choice(
                                finite_observed,
                                size=len(finite_observed),
                                replace=True,
                            )
                        )
                    )
                row[f"{metric}_ci_low"] = float(np.nanquantile(draws, 0.025))
                row[f"{metric}_ci_high"] = float(np.nanquantile(draws, 0.975))
        summary_rows.append(row)
    return subject_rows, summary_rows


def _save_trajectories(path: Path, predictions: Sequence[AdaptivePrediction]) -> None:
    rows = []
    for prediction in predictions:
        for index in range(len(prediction.truth_hbo)):
            row = {
                "condition_id": prediction.condition_id,
                "subject": prediction.subject,
                "heldout_trial": prediction.heldout_trial,
                "model": prediction.model,
                "spatial_mode": prediction.spatial_mode,
                "time_s": index / 10.0 - 5.0,
                "eeg_observation": prediction.eeg_observation[index],
                "eeg_reconstruction": prediction.eeg_reconstruction[index],
                "eeg_valid": prediction.eeg_valid_mask[index],
                "eeg_predictive_std": prediction.observation_predictive_std[index, 0],
                "hbo_truth": prediction.truth_hbo[index],
                "hbo_estimate": prediction.estimate_hbo[index],
                "hbo_predictive_std": prediction.observation_predictive_std[index, 1],
                "hbr_truth": prediction.truth_hbr[index],
                "hbr_estimate": prediction.estimate_hbr[index],
                "hbr_predictive_std": prediction.observation_predictive_std[index, 2],
                "fnirs_valid": prediction.fnirs_valid_mask[index],
            }
            row.update({name: prediction.states[index, state_index] for state_index, name in enumerate(STATE_NAMES)})
            row.update({
                f"{name}_std": prediction.state_std[index, state_index]
                for state_index, name in enumerate(STATE_NAMES)
            })
            row.update({
                f"target_{name}": prediction.target_states[index, state_index]
                for state_index, name in enumerate(STATE_NAMES)
            })
            row.update({
                f"target_{name}_std": prediction.target_state_std[index, state_index]
                for state_index, name in enumerate(STATE_NAMES)
            })
            row.update({
                f"gauge_{name}_scale": prediction.gauge_scales[state_index]
                for state_index, name in enumerate(STATE_NAMES)
            })
            row.update({
                f"gauge_{name}_offset": prediction.gauge_offsets[state_index]
                for state_index, name in enumerate(STATE_NAMES)
            })
            row["gauge_reconstruction_max_abs_delta"] = prediction.gauge_reconstruction_max_abs_delta
            rows.append(row)
    _write_csv(path, rows)


def _plot_summary(summary: Sequence[Mapping[str, Any]], run_dir: Path) -> str:
    conditions = list(dict.fromkeys(str(row["condition_id"]) for row in summary))
    paths = list(dict.fromkeys((str(row["model"]), str(row["spatial_mode"])) for row in summary))
    labels = [f"{model.replace('adaptive_', '').replace('_', '-')} {spatial}" for model, spatial in paths]
    lookup = {(row["condition_id"], row["model"], row["spatial_mode"]): row for row in summary}
    metrics = [("r2", "HbO R²"), ("pcc", "HbO correlation"), ("variance_ratio", "HbO variance ratio"), ("eeg_r2", "EEG-proxy R²")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(len(conditions), dtype=float)
    width = 0.19
    for axis, (metric, title) in zip(axes.flat, metrics):
        for path_index, ((model, spatial_mode), label) in enumerate(zip(paths, labels)):
            values = [float(lookup[(condition, model, spatial_mode)][metric]) for condition in conditions]
            offset = path_index - (len(paths) - 1) / 2.0
            axis.bar(x + offset * width, values, width=width, label=label)
        axis.axhline(0.0, color="black", linewidth=0.8)
        if metric == "variance_ratio":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xticks(x, [value.replace("_", "\n") for value in conditions])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Adaptive physiology-constrained shared-state smoother")
    fig.tight_layout()
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    output = figures / "adaptive_model_metric_summary.svg"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return str(output.relative_to(run_dir))


def _plot_representative(
    predictions: Sequence[AdaptivePrediction],
    fold_rows: Sequence[Mapping[str, Any]],
    run_dir: Path,
    count: int,
) -> tuple[str, list[dict[str, Any]]]:
    candidates = [
        row for row in fold_rows
        if row["model"] == "adaptive_joint" and row["spatial_mode"] == "local"
        and float(row["pcc"]) > 0.0 and float(row["variance_ratio"]) > 0.0
    ]
    if not candidates:
        candidates = [
            row for row in fold_rows
            if row["model"] == "adaptive_joint" and row["spatial_mode"] == "local"
        ]
    candidates.sort(key=lambda row: abs(np.log(float(row["variance_ratio"]))))
    selected_rows = candidates[: int(count)]
    lookup = {
        (prediction.condition_id, prediction.subject, prediction.heldout_trial, prediction.model, prediction.spatial_mode): prediction
        for prediction in predictions
    }
    fig, axes = plt.subplots(6, len(selected_rows), figsize=(7 * len(selected_rows), 17), sharex="col")
    if len(selected_rows) == 1:
        axes = axes[:, None]
    selected_manifest = []
    for column, row in enumerate(selected_rows):
        key = (row["condition_id"], row["subject"], int(row["heldout_trial"]), "adaptive_joint", "local")
        prediction = lookup[key]
        time = np.arange(len(prediction.truth_hbo)) / 10.0 - 5.0
        axes[0, column].plot(time, prediction.states[:, 4], color="#0072b2", label="smoothed shared driver")
        axes[0, column].plot(time, prediction.eeg_observation, color="#777777", alpha=0.65, label="EEG proxy")
        axes[1, column].plot(time, prediction.eeg_observation, color="#222222", label="EEG proxy observed")
        axes[1, column].plot(time, prediction.eeg_reconstruction, color="#0072b2", label="state reconstruction")
        axes[1, column].fill_between(
            time,
            prediction.eeg_reconstruction - 1.96 * prediction.observation_predictive_std[:, 0],
            prediction.eeg_reconstruction + 1.96 * prediction.observation_predictive_std[:, 0],
            color="#0072b2",
            alpha=0.16,
            label="95% posterior predictive band",
        )
        axes[2, column].plot(time, prediction.states[:, 0], color="#cc79a7", label="vasodilation s")
        axes[3, column].plot(time, 1.0 + prediction.states[:, 1], color="#009e73", label="relative flow 1+delta_f")
        axes[4, column].plot(time, prediction.truth_hbo, color="#222222", label="HbO observed")
        axes[4, column].plot(time, prediction.estimate_hbo, color="#d55e00", label="HbO reconstructed")
        axes[4, column].fill_between(
            time,
            prediction.estimate_hbo - 1.96 * prediction.observation_predictive_std[:, 1],
            prediction.estimate_hbo + 1.96 * prediction.observation_predictive_std[:, 1],
            color="#d55e00",
            alpha=0.16,
            label="95% posterior predictive band",
        )
        axes[5, column].plot(time, prediction.truth_hbr, color="#222222", label="HbR observed")
        axes[5, column].plot(time, prediction.estimate_hbr, color="#56b4e9", label="HbR reconstructed")
        axes[5, column].fill_between(
            time,
            prediction.estimate_hbr - 1.96 * prediction.observation_predictive_std[:, 2],
            prediction.estimate_hbr + 1.96 * prediction.observation_predictive_std[:, 2],
            color="#56b4e9",
            alpha=0.16,
            label="95% posterior predictive band",
        )
        for axis in axes[:, column]:
            axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, loc="best")
        axes[0, column].set_title(
            f"{prediction.condition_id} / {prediction.subject} / fold {prediction.heldout_trial}\n"
            f"HbO PCC={float(row['pcc']):.3f}, variance ratio={float(row['variance_ratio']):.3f}"
        )
        axes[-1, column].set_xlabel("event-relative time (s)")
        selected_manifest.append({
            "condition_id": prediction.condition_id,
            "subject": prediction.subject,
            "heldout_trial": prediction.heldout_trial,
            "hbo_pcc": float(row["pcc"]),
            "hbo_variance_ratio": float(row["variance_ratio"]),
            "selected_fnirs_channels": list(prediction.selected_fnirs_channels),
            "selected_eeg_channels": list(prediction.selected_eeg_channels),
        })
    row_labels = ["shared driver / EEG proxy", "EEG", "vasodilation", "blood flow", "HbO", "HbR"]
    for row_index, label in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(label)
    fig.suptitle("Representative local joint fixed-interval state trajectories", y=0.995)
    fig.tight_layout()
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    output = figures / "adaptive_representative_full_trajectories.svg"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return str(output.relative_to(run_dir)), selected_manifest


def _legacy_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    directory = REPO_ROOT / str(config.get("legacy_reference_dir", ""))
    path = directory / "summary_metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_markdown(
    summary: Sequence[Mapping[str, Any]],
    legacy: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Adaptive physiology-constrained shared neural state test",
        "",
        "This exploratory test replaces the causal scalar-HRF filter with a Croce-linearized five-state fixed-interval smoother. "
        "The local path uses one selected fNIRS spatial anchor as paired HbO/HbR observations (two fNIRS channels) and its six nearest EEG channels; the global path is a same-fold all-scalp-EEG ablation.",
        "",
        "`adaptive_joint` is the requested EEG/fNIRS compromise state because held-out observations from both modalities enter smoothing. "
        "`adaptive_eeg_only` is the strict cross-modal control and must be used for any EEG-to-fNIRS prediction claim.",
        "",
        "## Leave-one-trial results",
        "",
        "| Condition | Model | Spatial | Subjects | HbO R2 | HbO PCC | HbO SD ratio | HbO variance ratio | EEG R2 | Driver monotonic fraction | HbO turns truth / estimate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition_id']} | {row['model']} | {row['spatial_mode']} | {row['subjects']} | "
            f"{float(row['r2']):.4f} | {float(row['pcc']):.4f} | {float(row['amplitude_ratio']):.4f} | "
            f"{float(row['variance_ratio']):.4f} | {float(row['eeg_r2']):.4f} | {float(row['driver_monotonic_fraction']):.4f} | "
            f"{float(row['hbo_turning_points_truth']):.2f} / {float(row['hbo_turning_points_estimate']):.2f} |"
        )
    lines.extend([
        "",
        "## Reconstruction reliability",
        "",
        "Trajectory NRMSE is reconstruction RMSE divided by the observed temporal SD. "
        "Predictive SD and coverage are separate posterior diagnostics. Joint-smoother coverage is a posterior fit diagnostic because the held-out observation entered smoothing.",
        "",
        "| Condition | Model | Spatial | HbO NRMSE | HbR NRMSE | EEG-proxy NRMSE | HbO reconstructed SD | HbO predictive SD | HbO 95% coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary:
        lines.append(
            f"| {row['condition_id']} | {row['model']} | {row['spatial_mode']} | "
            f"{float(row['trajectory_deviation_nrmse']):.4f} | "
            f"{float(row['hbr_trajectory_deviation_nrmse']):.4f} | "
            f"{float(row['eeg_trajectory_deviation_nrmse']):.4f} | "
            f"{float(row['reconstructed_temporal_sd']):.4f} | "
            f"{float(row['posterior_predictive_sd_mean']):.4f} | "
            f"{float(row['predictive_95_coverage']):.4f} |"
        )
    focus_legacy = [
        row for row in legacy
        if row.get("validation") == "leave_one_trial"
        and row.get("condition_id") in {"single_trial_clean_v3", "simultaneous_unified"}
        and (
            row.get("model") in {"croce_joint", "croce_eeg_only"}
            or (row.get("model") == "lin_trtd" and row.get("hrf_mode") == "optimized")
        )
    ]
    if focus_legacy:
        lines.extend([
            "",
            "## Corrected legacy reference",
            "",
            "The reference rerun includes sequential particle-weight carry-over, stationary initial covariance, correct modality-specific saved drivers, and corrected Lin CP tensor products.",
            "",
            "| Condition | Model | HRF | HbO R2 | HbO PCC | HbO variance ratio |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ])
        for row in focus_legacy:
            lines.append(
                f"| {row['condition_id']} | {row['model']} | {row['hrf_mode']} | {float(row['r2']):.4f} | "
                f"{float(row['pcc']):.4f} | {float(row['variance_ratio']):.4f} |"
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- Joint smoothing is allowed to trade EEG fit for fNIRS fit; it is not a deployable EEG-only predictor.",
        "- HbO/HbR measurement gains absorb canonical-unit scaling, while the transition shape remains bounded by Croce/Balloon parameters.",
        "- Physiological parameters and modality balance are fitted only on the non-held-out training trials inside each fold.",
        "- A better joint waveform supports use as a soft multimodal teacher only if the retained EEG reconstruction is non-trivial; it does not establish a unique physical neural source.",
        "- The EEG-only path determines how much delayed hemodynamic structure is independently recoverable from EEG.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    config_path = REPO_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.smoke:
        for condition in config["data"]["conditions"]:
            condition["subjects"] = list(condition["subjects"][:1])
            condition["max_trials_per_subject"] = 3
        config["analysis"]["bootstrap_iterations"] = 100
        config["analysis"]["ssm"]["max_iterations"] = 8
        config["analysis"]["ssm"]["q_scale_candidates"] = [1.0]
        config["analysis"]["ssm"]["fnirs_noise_scale_candidates"] = [0.5, 1.0]
    run_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity"
        / datetime.now().strftime("%Y%m%d_%H%M%S_adaptive_shared_neural_ssm_v2")
    )
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    grouped, contracts = _load_trials(config)
    tasks = []
    subject_index = 0
    for condition_id, subjects in grouped.items():
        for subject, trials in subjects.items():
            tasks.append((condition_id, subject_index, subject, trials, config["data"], config["analysis"]))
            subject_index += 1
    predictions: list[AdaptivePrediction] = []
    fit_rows: list[dict[str, Any]] = []
    workers = int(config["analysis"].get("workers", 1))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_subject, *task) for task in tasks]
            for future in as_completed(futures):
                subject_predictions, subject_fits = future.result()
                predictions.extend(subject_predictions)
                fit_rows.extend(subject_fits)
    else:
        for task in tasks:
            subject_predictions, subject_fits = _run_subject(*task)
            predictions.extend(subject_predictions)
            fit_rows.extend(subject_fits)
    predictions.sort(key=lambda value: (value.condition_id, value.subject, value.heldout_trial, value.model, value.spatial_mode))
    fit_rows.sort(key=lambda value: (value["condition_id"], value["subject"], value["heldout_trial"], value["spatial_mode"]))
    baseline_n = int(round(float(config["data"]["baseline_duration_s"]) * 10.0))
    fold_rows = [_prediction_metrics(prediction, baseline_n) for prediction in predictions]
    subject_rows, summary_rows = _aggregate_metrics(
        fold_rows,
        bootstrap_iterations=int(config["analysis"]["bootstrap_iterations"]),
        seed=int(config["analysis"]["seed"]),
    )
    _write_csv(run_dir / "fold_metrics.csv", fold_rows)
    _write_csv(run_dir / "subject_metrics.csv", subject_rows)
    _write_csv(run_dir / "summary_metrics.csv", summary_rows)
    _write_csv(run_dir / "fit_parameters.csv", fit_rows)
    _save_trajectories(run_dir / "trajectories.csv", predictions)
    legacy = _legacy_rows(config)
    (run_dir / "summary.md").write_text(_summary_markdown(summary_rows, legacy), encoding="utf-8")
    figures = [_plot_summary(summary_rows, run_dir)]
    representative_figure, representative_samples = _plot_representative(
        predictions, fold_rows, run_dir, int(config["analysis"]["representative_samples"]),
    )
    figures.append(representative_figure)
    sources = [
        run_dir / "config.yaml",
        Path(__file__),
        REPO_ROOT / "experiments/evaluate_shared_neural_driver_unified.py",
        REPO_ROOT / "src/inference/adaptive_neurovascular_ssm.py",
        REPO_ROOT / "src/metrics/trajectory_reliability.py",
        REPO_ROOT / "src/inference/neurovascular_smc.py",
        REPO_ROOT / str(config["data"]["cache_root"]) / "cache_manifest.json",
        REPO_ROOT / str(config["data"]["cache_root"]) / "event_index/event_manifest.json",
        REPO_ROOT / str(config["data"]["cache_root"]) / "channel_geometry/geometry_manifest.json",
    ]
    manifest = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "formal_exploratory",
        "protected_open": False,
        "git": _git_payload(),
        "platform": {"python": platform.python_version(), "platform": platform.platform()},
        "input_hashes": [{"path": str(path), "sha256": _sha256(path)} for path in sources],
        "loader_contracts": contracts,
        "prediction_count": len(predictions),
        "fit_count": len(fit_rows),
        "representative_samples": representative_samples,
        "reliability_contract": {
            "primary_deviation_metric": "trajectory_deviation_nrmse",
            "observed_std_floor": 1e-8,
            "predictive_std_floor": 1e-8,
            "predictive_interval": "normal_95_percent_z_1.959963984540054",
            "aggregation": "fold_then_subject_equal_then_condition",
            "bootstrap_unit": "subject",
        },
        "artifacts": [
            "config.yaml", "fold_metrics.csv", "subject_metrics.csv", "summary_metrics.csv",
            "fit_parameters.csv", "trajectories.csv", "summary.md", *figures,
        ],
        "claim_boundary": [
            "joint output is a multimodal compromise, not EEG-only prediction",
            "EEG-only output is the cross-modal reconstruction control",
            "bounded linearized physiology is not proof of unique latent-state identity",
            "posterior predictive bands are fit diagnostics for joint smoothing",
            "EEG observation means the log-power PCA proxy, not the raw EEG waveform",
            "exploratory validation; not a protected E0 gate decision",
        ],
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/physiology_semantic_tokenizer/adaptive_shared_neural_ssm.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
