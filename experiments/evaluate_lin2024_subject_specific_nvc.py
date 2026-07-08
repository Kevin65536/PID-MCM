#!/usr/bin/env python3
"""Lin 2024 inspired subject-specific EEG-fNIRS NVC diagnostic.

This is not a direct reproduction of Lin et al. 2024: the current cache stores
paired optical fNIRS channels rather than HbO/HbR concentration, and Python does
not provide the Tensorlab CP decomposition used in the paper. The diagnostic
keeps Lin's two testable ideas: task-related EEG component extraction and
subject-specific double-gamma HRF fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import gamma
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lin2024_subject_specific_nvc_v1"


@dataclass(frozen=True)
class EventRecord:
    subject: int
    anchor: str
    event: str
    eeg: np.ndarray
    eeg_features: np.ndarray
    fnirs: np.ndarray
    stimulus: np.ndarray


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


def _band_envelopes(eeg: np.ndarray, eeg_fs: float, target_fs: float, bands: Mapping[str, Sequence[float]]) -> np.ndarray:
    step = int(round(eeg_fs / target_fs))
    if step <= 0 or eeg.shape[0] % step != 0:
        raise ValueError("EEG sampling rate must be an integer multiple of target fNIRS sampling rate")
    features = []
    for low, high in bands.values():
        sos = butter(4, [float(low), float(high)], btype="bandpass", fs=eeg_fs, output="sos")
        filtered = sosfiltfilt(sos, eeg, axis=0)
        envelope = np.log1p(np.abs(hilbert(filtered, axis=0)) ** 2)
        rebinned = envelope.reshape(envelope.shape[0] // step, step, envelope.shape[1]).mean(axis=1)
        features.append(rebinned)
    return np.concatenate(features, axis=1)


def _load_subject(path: Path, subject: int, config: Mapping[str, Any]) -> list[EventRecord]:
    eeg_fs = float(config["eeg_fs_hz"])
    fnirs_fs = float(config["fnirs_fs_hz"])
    eeg_analysis = _slice_bounds(config, eeg_fs, "analysis_start_s", "analysis_duration_s")
    fnirs_analysis = _slice_bounds(config, fnirs_fs, "analysis_start_s", "analysis_duration_s")
    eeg_baseline = _slice_bounds(config, eeg_fs, "baseline_start_s")
    fnirs_baseline = _slice_bounds(config, fnirs_fs, "baseline_start_s")
    bands = config["eeg_bands_hz"]
    rows: list[EventRecord] = []
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
            eeg_window = eeg[slice(*eeg_analysis)]
            fnirs_window = fnirs[slice(*fnirs_analysis)]
            eeg_features = _band_envelopes(eeg_window, eeg_fs, fnirs_fs, bands)
            if len(eeg_features) != len(fnirs_window):
                raise ValueError(f"length mismatch for {subject}:{prefix}")
            stimulus = np.zeros(len(fnirs_window), dtype=np.float64)
            stimulus[: int(round(float(config["task_duration_s"]) * fnirs_fs))] = 1.0
            rows.append(EventRecord(subject, anchor, event, eeg_window, eeg_features, fnirs_window, stimulus))
    return rows


def _stack_events(events: Sequence[EventRecord], field: str) -> np.ndarray:
    return np.concatenate([getattr(event, field) for event in events], axis=0)


def _event_matrix(events: Sequence[EventRecord], field: str) -> np.ndarray:
    return np.stack([getattr(event, field) for event in events], axis=0)


def _fit_component(method: str, train_events: Sequence[EventRecord]) -> tuple[Any, StandardScaler | None]:
    if method == "stimulus":
        return None, None
    features = _stack_events(train_events, "eeg_features")
    scaler = StandardScaler().fit(features)
    z_features = scaler.transform(features)
    if method == "band_average":
        return {"kind": "band_average"}, scaler
    if method == "task_pls_eeg":
        target = _stack_events(train_events, "stimulus").reshape(-1, 1)
        model = PLSRegression(n_components=1, scale=False).fit(z_features, target)
        return {"kind": "pls", "model": model}, scaler
    if method == "fnirs_pls_eeg":
        target = _stack_events(train_events, "fnirs").mean(axis=1, keepdims=True)
        model = PLSRegression(n_components=1, scale=False).fit(z_features, target)
        return {"kind": "pls", "model": model}, scaler
    if method == "pca_eeg":
        model = PCA(n_components=1, random_state=7).fit(z_features)
        return {"kind": "pca", "model": model}, scaler
    raise ValueError(f"unknown component method {method}")


def _component_series(method: str, model: Any, scaler: StandardScaler | None, events: Sequence[EventRecord]) -> list[np.ndarray]:
    if method == "stimulus":
        return [event.stimulus.copy() for event in events]
    assert scaler is not None
    output = []
    for event in events:
        features = scaler.transform(event.eeg_features)
        if model["kind"] == "band_average":
            series = features.mean(axis=1)
        elif model["kind"] == "pls":
            series = model["model"].transform(features).reshape(-1)
        elif model["kind"] == "pca":
            series = model["model"].transform(features).reshape(-1)
        else:
            raise ValueError(model["kind"])
        if np.std(series) > 1e-12:
            series = (series - np.mean(series)) / np.std(series)
        output.append(series)
    return output


def _double_gamma(params: Sequence[float], fs: float, duration_s: float) -> np.ndarray:
    a1, a2, b1, b2, c = [float(value) for value in params]
    t = np.arange(0, duration_s, 1.0 / fs, dtype=np.float64)
    peak = (b1**a1) * np.power(t, a1 - 1.0) * np.exp(-b1 * t) / max(float(gamma(a1)), 1e-12)
    undershoot = (b2**a2) * np.power(t, a2 - 1.0) * np.exp(-b2 * t) / max(float(c) * float(gamma(a2)), 1e-12)
    hrf = peak - undershoot
    hrf[~np.isfinite(hrf)] = 0.0
    scale = np.max(np.abs(hrf))
    return hrf / scale if scale > 1e-12 else hrf


def _canonical_params() -> np.ndarray:
    return np.asarray([6.0, 16.0, 1.0, 1.0, 6.0], dtype=np.float64)


def _shape_penalty(params: Sequence[float]) -> float:
    a1, a2, b1, b2, _ = [float(value) for value in params]
    checks = [
        (a1 / max(b1, 1e-12), 3.0, 7.0),
        (a2 / max(b2, 1e-12), 9.0, 18.0),
        (2.35 * np.sqrt(max(a1 - 1.0, 0.0)) / max(b1, 1e-12), 3.0, 6.0),
        (2.35 * np.sqrt(max(a2 - 1.0, 0.0)) / max(b2, 1e-12), 7.0, 11.0),
    ]
    penalty = 0.0
    for value, low, high in checks:
        if value < low:
            penalty += (low - value) ** 2
        elif value > high:
            penalty += (value - high) ** 2
    return penalty


def _convolve_component(series: np.ndarray, params: Sequence[float], fs: float) -> np.ndarray:
    hrf = _double_gamma(params, fs=fs, duration_s=30.0)
    return np.convolve(series, hrf, mode="full")[: len(series)] / fs


def _fit_linear_scale(x_events: Sequence[np.ndarray], y_events: Sequence[np.ndarray], alpha: float) -> np.ndarray:
    x = np.concatenate([value.reshape(-1, 1) for value in x_events], axis=0)
    y = np.concatenate(y_events, axis=0)
    design = np.column_stack([x, np.ones(len(x))])
    regularizer = np.diag([alpha, 0.0])
    return np.linalg.solve(design.T @ design + regularizer, design.T @ y)


def _predict_linear_scale(x_events: Sequence[np.ndarray], coefficients: np.ndarray) -> list[np.ndarray]:
    return [np.column_stack([x.reshape(-1, 1), np.ones(len(x))]) @ coefficients for x in x_events]


def _fit_predict_nvc(
    train_events: Sequence[EventRecord],
    test_events: Sequence[EventRecord],
    method: str,
    hrf_mode: str,
    fs: float,
    alpha: float,
    max_iterations: int,
) -> dict[str, Any]:
    component_model, scaler = _fit_component(method, train_events)
    train_components = _component_series(method, component_model, scaler, train_events)
    test_components = _component_series(method, component_model, scaler, test_events)
    params, coeff, train_mse = _fit_hrf(
        train_components, [event.fnirs for event in train_events], fs, hrf_mode, alpha, max_iterations
    )
    drivers = [_convolve_component(series, params, fs) for series in test_components]
    pred = _predict_linear_scale(drivers, coeff)
    return {
        "params": params,
        "coefficients": coeff,
        "train_mse": train_mse,
        "components": test_components,
        "drivers": drivers,
        "predictions": pred,
    }


def _fit_hrf(
    train_components: Sequence[np.ndarray],
    train_y: Sequence[np.ndarray],
    fs: float,
    mode: str,
    alpha: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if mode == "canonical":
        params = _canonical_params()
        x_events = [_convolve_component(series, params, fs) for series in train_components]
        coeff = _fit_linear_scale(x_events, train_y, alpha)
        pred = _predict_linear_scale(x_events, coeff)
        mse = float(np.mean((np.concatenate(pred) - np.concatenate(train_y)) ** 2))
        return params, coeff, mse
    if mode != "optimized":
        raise ValueError(mode)

    bounds = [(2.0, 10.0), (6.0, 25.0), (0.5, 2.0), (0.05, 1.5), (0.2, 15.0)]

    def objective(params: np.ndarray) -> float:
        x_events = [_convolve_component(series, params, fs) for series in train_components]
        coeff = _fit_linear_scale(x_events, train_y, alpha)
        pred = _predict_linear_scale(x_events, coeff)
        mse = float(np.mean((np.concatenate(pred) - np.concatenate(train_y)) ** 2))
        return mse + 0.05 * _shape_penalty(params)

    result = minimize(
        objective,
        _canonical_params(),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iterations), "ftol": 1e-9},
    )
    params = np.asarray(result.x if result.success else _canonical_params(), dtype=np.float64)
    x_events = [_convolve_component(series, params, fs) for series in train_components]
    coeff = _fit_linear_scale(x_events, train_y, alpha)
    pred = _predict_linear_scale(x_events, coeff)
    mse = float(np.mean((np.concatenate(pred) - np.concatenate(train_y)) ** 2))
    return params, coeff, mse


def _metrics(y_true_events: Sequence[np.ndarray], y_pred_events: Sequence[np.ndarray], train_y_events: Sequence[np.ndarray]) -> dict[str, float]:
    y_true = np.concatenate(y_true_events, axis=0)
    y_pred = np.concatenate(y_pred_events, axis=0)
    train_y = np.concatenate(train_y_events, axis=0)
    mean = train_y.mean(axis=0, keepdims=True)
    std = np.maximum(train_y.std(axis=0, keepdims=True), 1e-12)
    truth_z = (y_true - mean) / std
    pred_z = (y_pred - mean) / std
    error = pred_z - truth_z
    r2 = 1.0 - float(np.sum(error**2)) / max(float(np.sum(truth_z**2)), 1e-12)
    flat_true = truth_z.reshape(-1)
    flat_pred = pred_z.reshape(-1)
    if np.std(flat_true) > 1e-12 and np.std(flat_pred) > 1e-12:
        pcc = float(np.corrcoef(flat_true, flat_pred)[0, 1])
    else:
        pcc = float("nan")
    time = np.linspace(-1.0, 1.0, y_true.shape[0] // len(y_true_events), dtype=np.float64)
    event_truth = np.stack(y_true_events)
    event_pred = np.stack(y_pred_events)
    slope_truth = np.sum(event_truth * time[None, :, None], axis=1) / max(float(np.sum(time * time)), 1e-12)
    slope_pred = np.sum(event_pred * time[None, :, None], axis=1) / max(float(np.sum(time * time)), 1e-12)
    return {
        "standardized_mse": float(np.mean(error**2)),
        "standardized_r2": r2,
        "pcc": pcc,
        "relative_rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2) / max(np.mean(y_true**2), 1e-12))),
        "amplitude_ratio": float(np.std(y_pred) / max(np.std(y_true), 1e-12)),
        "event_mean_pcc": _safe_corr(event_truth.mean(axis=1), event_pred.mean(axis=1)),
        "event_slope_pcc": _safe_corr(slope_truth, slope_pred),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    flat_a = np.asarray(a).reshape(-1)
    flat_b = np.asarray(b).reshape(-1)
    if len(flat_a) < 2 or np.std(flat_a) <= 1e-12 or np.std(flat_b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(flat_a, flat_b)[0, 1])


def _mean_trajectory_baseline(train_events: Sequence[EventRecord], test_events: Sequence[EventRecord]) -> list[np.ndarray]:
    mean = _event_matrix(train_events, "fnirs").mean(axis=0)
    return [mean.copy() for _ in test_events]


def _self_persistence_baseline(train_events: Sequence[EventRecord], test_events: Sequence[EventRecord]) -> list[np.ndarray]:
    first = _event_matrix(train_events, "fnirs")[:, 0, :].mean(axis=0)
    preds = []
    for event in test_events:
        pred = np.empty_like(event.fnirs)
        pred[0] = first
        pred[1:] = event.fnirs[:-1]
        preds.append(pred)
    return preds


def _evaluate_fold(
    split: str,
    anchor: str,
    train_events: Sequence[EventRecord],
    test_events: Sequence[EventRecord],
    methods: Sequence[str],
    hrf_modes: Sequence[str],
    fs: float,
    alpha: float,
    max_iterations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_subjects = sorted({event.subject for event in test_events})
    y_train = [event.fnirs for event in train_events]
    y_test = [event.fnirs for event in test_events]
    for baseline, pred in (
        ("fnirs_task_mean", _mean_trajectory_baseline(train_events, test_events)),
        ("fnirs_self_persistence", _self_persistence_baseline(train_events, test_events)),
    ):
        row = {
            "split": split,
            "anchor": anchor,
            "component_method": baseline,
            "hrf_mode": "none",
            "train_events": len(train_events),
            "test_events": len(test_events),
            "test_subjects": ",".join(str(value) for value in test_subjects),
            **_metrics(y_test, pred, y_train),
        }
        rows.append(row)
    for method in methods:
        for hrf_mode in hrf_modes:
            fit = _fit_predict_nvc(train_events, test_events, method, hrf_mode, fs, alpha, max_iterations)
            params = fit["params"]
            pred = fit["predictions"]
            train_mse = fit["train_mse"]
            a1, a2, b1, b2, c = [float(value) for value in params]
            row = {
                "split": split,
                "anchor": anchor,
                "component_method": method,
                "hrf_mode": hrf_mode,
                "train_events": len(train_events),
                "test_events": len(test_events),
                "test_subjects": ",".join(str(value) for value in test_subjects),
                "train_fit_mse": train_mse,
                "hrf_a1": a1,
                "hrf_a2": a2,
                "hrf_b1": b1,
                "hrf_b2": b2,
                "hrf_c": c,
                "hrf_ttp": a1 / max(b1, 1e-12),
                "hrf_ttu": a2 / max(b2, 1e-12),
                **_metrics(y_test, pred, y_train),
            }
            rows.append(row)
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]], bootstrap_iterations: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["component_method"], row["hrf_mode"])].append(row)
    metric_names = (
        "standardized_mse",
        "standardized_r2",
        "pcc",
        "relative_rmse",
        "amplitude_ratio",
        "event_mean_pcc",
        "event_slope_pcc",
        "train_fit_mse",
        "hrf_ttp",
        "hrf_ttu",
        "hrf_c",
    )
    rng = np.random.default_rng(seed)
    output = []
    for key, group_rows in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "split": key[0],
            "component_method": key[1],
            "hrf_mode": key[2],
            "folds": len(group_rows),
            "test_events": int(sum(int(row["test_events"]) for row in group_rows)),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows if metric in row and np.isfinite(float(row[metric]))]
            if values:
                aggregate[metric] = float(np.mean(values))
        mse_values = np.asarray([float(row["standardized_mse"]) for row in group_rows])
        if len(mse_values):
            draws = np.empty(int(bootstrap_iterations), dtype=np.float64)
            for index in range(len(draws)):
                draws[index] = rng.choice(mse_values, size=len(mse_values), replace=True).mean()
            aggregate["standardized_mse_bootstrap_ci_low"] = float(np.quantile(draws, 0.025))
            aggregate["standardized_mse_bootstrap_ci_high"] = float(np.quantile(draws, 0.975))
        output.append(aggregate)
    return output


def _plot(summary_rows: Sequence[Mapping[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for split in sorted({str(row["split"]) for row in summary_rows}):
        selected = [row for row in summary_rows if row["split"] == split]
        labels = [f"{row['component_method']}\n{row['hrf_mode']}" for row in selected]
        mse = [float(row["standardized_mse"]) for row in selected]
        pcc = [float(row.get("pcc", np.nan)) for row in selected]
        x = np.arange(len(selected))
        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(selected) * 0.65), 8), sharex=True)
        axes[0].bar(x, mse, color="#2563eb")
        axes[0].set_ylabel("standardized MSE")
        axes[0].set_title(split)
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(x, pcc, color="#16a34a")
        axes[1].set_ylabel("PCC")
        axes[1].set_xticks(x, labels=labels, rotation=45, ha="right", fontsize=8)
        axes[1].grid(axis="y", alpha=0.25)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.28, hspace=0.15)
        for suffix, dpi in (("svg", None), ("png", 300)):
            path = run_dir / "figures" / f"lin2024_nvc_{split}.{suffix}"
            fig.savefig(path, dpi=dpi)
            artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
        plt.close(fig)
    return artifacts


def _plot_hrf_parameters(anchor_rows: Sequence[Mapping[str, Any]], run_dir: Path, fs: float) -> list[dict[str, Any]]:
    rows = [
        row for row in anchor_rows
        if row.get("hrf_mode") == "optimized" and "hrf_a1" in row and row.get("component_method") not in {"fnirs_task_mean", "fnirs_self_persistence"}
    ]
    if not rows:
        return []
    artifacts: list[dict[str, Any]] = []
    labels = sorted({f"{row['split']}\n{row['component_method']}" for row in rows})
    metrics = [("hrf_ttp", "time to peak"), ("hrf_ttu", "time to undershoot"), ("hrf_c", "peak/undershoot ratio")]
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(labels) * 0.9), 4.8), sharex=True)
    rng = np.random.default_rng(123)
    for axis, (field, title) in zip(axes, metrics):
        for index, label in enumerate(labels):
            values = [
                float(row[field]) for row in rows
                if f"{row['split']}\n{row['component_method']}" == label and np.isfinite(float(row[field]))
            ]
            if not values:
                continue
            jitter = rng.uniform(-0.18, 0.18, len(values))
            axis.scatter(np.full(len(values), index) + jitter, values, s=12, alpha=0.55)
            axis.plot([index - 0.22, index + 0.22], [np.median(values), np.median(values)], color="#111827", linewidth=2)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", fontsize=7)
    fig.suptitle("Optimized HRF parameter distribution")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.34, top=0.84, wspace=0.28)
    for suffix, dpi in (("svg", None), ("png", 300)):
        path = run_dir / "figures" / f"lin2024_hrf_parameter_distribution.{suffix}"
        fig.savefig(path, dpi=dpi)
        artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
    plt.close(fig)

    curve_groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (str(row["split"]), str(row["component_method"]))
        curve_groups[key].append(np.asarray([float(row["hrf_a1"]), float(row["hrf_a2"]), float(row["hrf_b1"]), float(row["hrf_b2"]), float(row["hrf_c"])]))
    fig, axis = plt.subplots(figsize=(10, 5))
    t = np.arange(0, 30.0, 1.0 / fs)
    for key, params_list in sorted(curve_groups.items()):
        params = np.median(np.stack(params_list), axis=0)
        axis.plot(t, _double_gamma(params, fs, 30.0), linewidth=1.8, label=f"{key[0]} / {key[1]}")
    axis.axhline(0, color="#111827", linewidth=0.8)
    axis.set_xlabel("seconds")
    axis.set_ylabel("normalized HRF")
    axis.set_title("Median optimized HRF curves")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for suffix, dpi in (("svg", None), ("png", 300)):
        path = run_dir / "figures" / f"lin2024_hrf_curves.{suffix}"
        fig.savefig(path, dpi=dpi)
        artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
    plt.close(fig)
    return artifacts


def _plot_trajectory_examples(
    rows: Sequence[EventRecord],
    config: Mapping[str, Any],
    run_dir: Path,
    fs: float,
    eeg_fs: float,
    alpha: float,
    max_iterations: int,
) -> list[dict[str, Any]]:
    visualization = config.get("visualization", {})
    subjects = [int(value) for value in visualization.get("example_subjects", [])]
    anchors = [str(value) for value in visualization.get("example_anchors", [])]
    methods = [str(value) for value in visualization.get("example_methods", ["stimulus"])]
    hrf_mode = str(visualization.get("example_hrf_mode", "optimized"))
    max_examples = int(visualization.get("max_examples", 8))
    if not subjects:
        subjects = sorted({row.subject for row in rows})[:1]
    if not anchors:
        anchors = sorted({row.anchor for row in rows})[:2]
    artifacts: list[dict[str, Any]] = []
    data_rows: list[dict[str, Any]] = []
    example_count = 0
    for subject in subjects:
        for anchor in anchors:
            subject_events = sorted(
                [row for row in rows if row.subject == subject and row.anchor == anchor],
                key=lambda row: row.event,
            )
            if len(subject_events) < 2:
                continue
            for method in methods:
                fit = _fit_predict_nvc(subject_events, subject_events, method, hrf_mode, fs, alpha, max_iterations)
                params = fit["params"]
                for event, component, driver, prediction in zip(
                    subject_events, fit["components"], fit["drivers"], fit["predictions"]
                ):
                    if example_count >= max_examples:
                        _write_csv(run_dir / "figure_data" / "trajectory_examples.csv", data_rows)
                        return artifacts
                    eeg_mean = event.eeg.mean(axis=1)
                    eeg_t = np.arange(len(eeg_mean)) / eeg_fs
                    fnirs_t = np.arange(len(event.fnirs)) / fs
                    component_z = (component - component.mean()) / max(component.std(), 1e-12)
                    driver_z = (driver - driver.mean()) / max(driver.std(), 1e-12)
                    eeg_z = (eeg_mean - eeg_mean.mean()) / max(eeg_mean.std(), 1e-12)
                    residual = event.fnirs - prediction
                    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=False)
                    axes[0].plot(eeg_t, eeg_z, color="#111827", linewidth=0.8, label="mean EEG, z")
                    axes[0].set_ylabel("EEG")
                    axes[0].legend(loc="upper right", fontsize=8)
                    axes[0].grid(alpha=0.2)
                    axes[1].plot(fnirs_t, component_z, color="#7c3aed", linewidth=1.2, label="EEG component")
                    axes[1].plot(fnirs_t, driver_z, color="#f59e0b", linewidth=1.2, label="HRF-convolved driver")
                    axes[1].set_ylabel("driver")
                    axes[1].legend(loc="upper right", fontsize=8)
                    axes[1].grid(alpha=0.2)
                    for channel, color in enumerate(("#2563eb", "#dc2626")):
                        axes[2].plot(fnirs_t, event.fnirs[:, channel], color=color, linewidth=1.2, label=f"true fNIRS ch{channel}")
                        axes[2].plot(fnirs_t, prediction[:, channel], color=color, linewidth=1.2, linestyle="--", label=f"pred ch{channel}")
                    axes[2].set_ylabel("fNIRS")
                    axes[2].legend(loc="upper right", fontsize=7, ncol=2)
                    axes[2].grid(alpha=0.2)
                    axes[3].plot(fnirs_t, residual[:, 0], color="#2563eb", linewidth=1.0, label="residual ch0")
                    axes[3].plot(fnirs_t, residual[:, 1], color="#dc2626", linewidth=1.0, label="residual ch1")
                    axes[3].axhline(0, color="#111827", linewidth=0.8)
                    axes[3].set_xlabel("seconds from task onset")
                    axes[3].set_ylabel("residual")
                    axes[3].legend(loc="upper right", fontsize=8)
                    axes[3].grid(alpha=0.2)
                    a1, a2, b1, b2, c = [float(value) for value in params]
                    fig.suptitle(
                        f"subject {subject} {anchor} {event.event} / {method} {hrf_mode} "
                        f"TTP={a1 / max(b1, 1e-12):.2f}s TTU={a2 / max(b2, 1e-12):.2f}s c={c:.2f}"
                    )
                    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.92, hspace=0.28)
                    stem = f"trajectory_subject{subject}_{anchor}_{event.event}_{method}_{hrf_mode}"
                    for suffix, dpi in (("svg", None), ("png", 300)):
                        path = run_dir / "figures" / f"{stem}.{suffix}"
                        fig.savefig(path, dpi=dpi)
                        artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
                    plt.close(fig)
                    for index, t_value in enumerate(fnirs_t):
                        data_rows.append({
                            "subject": subject,
                            "anchor": anchor,
                            "event": event.event,
                            "method": method,
                            "hrf_mode": hrf_mode,
                            "time_s": float(t_value),
                            "eeg_component_z": float(component_z[index]),
                            "shared_driver_z": float(driver_z[index]),
                            "fnirs_ch0": float(event.fnirs[index, 0]),
                            "fnirs_ch1": float(event.fnirs[index, 1]),
                            "pred_fnirs_ch0": float(prediction[index, 0]),
                            "pred_fnirs_ch1": float(prediction[index, 1]),
                            "residual_ch0": float(residual[index, 0]),
                            "residual_ch1": float(residual[index, 1]),
                            "hrf_a1": a1,
                            "hrf_a2": a2,
                            "hrf_b1": b1,
                            "hrf_b2": b2,
                            "hrf_c": c,
                        })
                    example_count += 1
    _write_csv(run_dir / "figure_data" / "trajectory_examples.csv", data_rows)
    return artifacts


def _summary_markdown(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Lin 2024 inspired subject-specific NVC diagnostic",
        "",
        "This run is diagnostic only. It did not use protected-test subjects and does not change the E0 gate.",
        "",
        "| Split | Component | HRF | MSE | R2 | PCC | amplitude ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['component_method']} | {row['hrf_mode']} | "
            f"{row.get('standardized_mse', float('nan')):.6f} | "
            f"{row.get('standardized_r2', float('nan')):.6f} | "
            f"{row.get('pcc', float('nan')):.6f} | "
            f"{row.get('amplitude_ratio', float('nan')):.6f} |"
        )
    lines.extend(["", "See `summary.json`, `metrics.csv`, and `anchor_metrics.csv` for full details.", ""])
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
        "paper_method": "Lin 2024 task-related EEG component plus subject-specific double-gamma HRF",
        "implementation_boundary": "Python approximation; no Tensorlab CPD; fNIRS target is paired optical channel data, not HbO concentration",
        "primary_contrast": "validation-subject leave-one-event subject-specific HRF, with subject-held-out group HRF as a stress-control only",
        "interpretation_rule": "fNIRS self-persistence is a private/history reference, not shared-state evidence",
    }, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "metric_registry.json", {
        "schema": SCHEMA,
        "primary": ["standardized_mse", "standardized_r2", "pcc"],
        "secondary": ["relative_rmse", "amplitude_ratio", "event_mean_pcc", "event_slope_pcc"],
        "diagnostic": ["hrf_ttp", "hrf_ttu", "hrf_c", "train_fit_mse"],
    })

    rows: list[EventRecord] = []
    input_files = []
    loader_config = {**data, "eeg_bands_hz": analysis["eeg_bands_hz"]}
    for split, subjects in (("train", train_subjects), ("validation", val_subjects)):
        for subject in subjects:
            path = _subject_cache(root, subject)
            input_files.append({"split": split, "subject": subject, "path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)})
            rows.extend(_load_subject(path, subject, loader_config))
    anchors = sorted({row.anchor for row in rows})
    train_rows = [row for row in rows if row.subject in set(train_subjects)]
    val_rows = [row for row in rows if row.subject in set(val_subjects)]
    methods = [str(value) for value in analysis["component_methods"]]
    hrf_modes = [str(value) for value in analysis["hrf_modes"]]
    anchor_rows: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_train = [row for row in train_rows if row.anchor == anchor]
        anchor_val = [row for row in val_rows if row.anchor == anchor]
        anchor_rows.extend(_evaluate_fold(
            "subject_held_out_group",
            anchor,
            anchor_train,
            anchor_val,
            methods,
            hrf_modes,
            float(data["fnirs_fs_hz"]),
            float(analysis["ridge_alpha"]),
            int(analysis["hrf_max_iterations"]),
        ))
        for subject in val_subjects:
            subject_events = [row for row in anchor_val if row.subject == subject]
            for heldout in sorted({row.event for row in subject_events}):
                fold_train = [row for row in subject_events if row.event != heldout]
                fold_test = [row for row in subject_events if row.event == heldout]
                anchor_rows.extend(_evaluate_fold(
                    "subject_specific_leave_one_event",
                    anchor,
                    fold_train,
                    fold_test,
                    methods,
                    hrf_modes,
                    float(data["fnirs_fs_hz"]),
                    float(analysis["ridge_alpha"]),
                    int(analysis["hrf_max_iterations"]),
                ))
            anchor_rows.extend(_evaluate_fold(
                "subject_specific_fit_all",
                anchor,
                subject_events,
                subject_events,
                methods,
                hrf_modes,
                float(data["fnirs_fs_hz"]),
                float(analysis["ridge_alpha"]),
                int(analysis["hrf_max_iterations"]),
            ))
    summary_rows = _aggregate(
        anchor_rows,
        int(analysis["subject_bootstrap_iterations"]),
        int(analysis["seed"]),
    )
    _write_csv(run_dir / "anchor_metrics.csv", anchor_rows)
    _write_csv(run_dir / "metrics.csv", summary_rows)
    figure_artifacts = _plot(summary_rows, run_dir)
    figure_artifacts.extend(_plot_hrf_parameters(anchor_rows, run_dir, float(data["fnirs_fs_hz"])))
    figure_artifacts.extend(_plot_trajectory_examples(
        val_rows,
        config,
        run_dir,
        float(data["fnirs_fs_hz"]),
        float(data["eeg_fs_hz"]),
        float(analysis["ridge_alpha"]),
        int(analysis["hrf_max_iterations"]),
    ))
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "formal_complete_diagnostic",
        "protected_test_used": False,
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "protected_test_subjects_unopened": sorted(protected),
        "anchors": anchors,
        "train_events": len(train_rows),
        "validation_events": len(val_rows),
        "summary_rows": summary_rows,
        "figures": figure_artifacts,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "cache_manifest_hashes.json", {"files": input_files})
    _write_json(run_dir / "evidence_calibration.json", {
        "schema": SCHEMA,
        "thresholds": "none; diagnostic comparison against private/history baselines",
        "subject_bootstrap_iterations": int(analysis["subject_bootstrap_iterations"]),
        "seed": int(analysis["seed"]),
        "protected_test_used": False,
    })
    _write_json(run_dir / "environment.json", {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_status_porcelain": subprocess.run(
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
        "anchor_metrics": "anchor_metrics.csv",
    })
    (run_dir / "summary.md").write_text(_summary_markdown(summary_rows), encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/lin2024_subject_specific_nvc.yaml"),
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
