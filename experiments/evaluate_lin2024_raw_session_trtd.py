#!/usr/bin/env python3
"""Run a Lin-2024-style TRTD + subject-specific HRF diagnostic on one raw session.

The script reads continuous MATLAB `cnt/mrk` files from the EEG+NIRS
Single-Trial dataset and avoids the pre-cut Croce cache. It follows the Lin
pipeline as closely as the local dataset permits:

- EEG: 1-40 Hz filtering, event epochs -5..15 s, 0.5 Hz time-frequency power.
- TRTD: shared nonnegative spatial/frequency CP factors with trial-specific
  temporal factors, followed by a TRCA component filter.
- fNIRS: paired 760/850 optical intensity to approximate HbO by MBLL, 0.01-0.2
  Hz filtering, GLM active-channel selection, top-3 HbO averaging.
- NVC: canonical and optimized double-gamma HRF with leave-one-trial validation.

This is a diagnostic upper-bound probe, not an E0 gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.linalg import pinv
from scipy.optimize import minimize
from scipy.signal import butter, spectrogram, sosfiltfilt
from scipy.special import gamma


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lin2024_raw_session_trtd_v1"


@dataclass(frozen=True)
class RawSession:
    subject: int
    session_index: int
    eeg: np.ndarray
    eeg_fs: float
    eeg_labels: list[str]
    eeg_marker_time_ms: np.ndarray
    eeg_marker_desc: np.ndarray
    fnirs_intensity: np.ndarray
    fnirs_fs: float
    fnirs_labels: list[str]
    fnirs_marker_time_ms: np.ndarray
    fnirs_marker_desc: np.ndarray
    fnirs_spatial_labels: list[str]


@dataclass(frozen=True)
class TrialSet:
    eeg_epochs: np.ndarray
    eeg_tensors: np.ndarray
    hbo_epochs: np.ndarray
    hbo_average: np.ndarray
    stimulus_epochs: np.ndarray
    selected_channels: list[int]
    selected_channel_labels: list[str]
    active_channel_t: np.ndarray
    trial_indices: list[int]
    trial_onsets_s: list[float]


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


def _cell(payload: Any, index: int) -> Any:
    return np.asarray(payload).ravel()[index]


def _labels(raw: Any) -> list[str]:
    return [str(value) for value in np.asarray(raw).ravel().tolist()]


def _load_raw_session(data_root: Path, subject: int, session_index: int) -> RawSession:
    subject_dir = f"subject {subject:02d}"
    eeg_cnt_path = data_root / "EEG_01-29" / subject_dir / "with occular artifact" / "cnt.mat"
    eeg_mrk_path = data_root / "EEG_01-29" / subject_dir / "with occular artifact" / "mrk.mat"
    fnirs_cnt_path = data_root / "NIRS_01-29" / subject_dir / "cnt.mat"
    fnirs_mrk_path = data_root / "NIRS_01-29" / subject_dir / "mrk.mat"
    fnirs_mnt_path = data_root / "NIRS_01-29" / subject_dir / "mnt.mat"

    eeg_cnt = _cell(loadmat(eeg_cnt_path, squeeze_me=True, struct_as_record=False)["cnt"], session_index)
    eeg_mrk = _cell(loadmat(eeg_mrk_path, squeeze_me=True, struct_as_record=False)["mrk"], session_index)
    fnirs_cnt = _cell(loadmat(fnirs_cnt_path, squeeze_me=True, struct_as_record=False)["cnt"], session_index)
    fnirs_mrk = _cell(loadmat(fnirs_mrk_path, squeeze_me=True, struct_as_record=False)["mrk"], session_index)
    fnirs_mnt = loadmat(fnirs_mnt_path, squeeze_me=True, struct_as_record=False)["mnt"]

    return RawSession(
        subject=subject,
        session_index=session_index,
        eeg=np.asarray(eeg_cnt.x, dtype=np.float64),
        eeg_fs=float(eeg_cnt.fs),
        eeg_labels=_labels(eeg_cnt.clab),
        eeg_marker_time_ms=np.asarray(eeg_mrk.time, dtype=np.float64),
        eeg_marker_desc=np.asarray(eeg_mrk.event.desc, dtype=int),
        fnirs_intensity=np.asarray(fnirs_cnt.x, dtype=np.float64),
        fnirs_fs=float(fnirs_cnt.fs),
        fnirs_labels=_labels(fnirs_cnt.clab),
        fnirs_marker_time_ms=np.asarray(fnirs_mrk.time, dtype=np.float64),
        fnirs_marker_desc=np.asarray(fnirs_mrk.event.desc, dtype=int),
        fnirs_spatial_labels=_labels(fnirs_mnt.clab),
    )


def _eeg_channel_mask(labels: Sequence[str]) -> np.ndarray:
    excluded = {"VEOG", "HEOG"}
    return np.asarray([label not in excluded and "EOG" not in label for label in labels], dtype=bool)


def _sos_bandpass(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    low = max(float(low), 1e-6)
    high = min(float(high), nyquist * 0.99)
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def _canonical_params() -> np.ndarray:
    return np.asarray([6.0, 16.0, 1.0, 1.0, 6.0], dtype=np.float64)


def _double_gamma(params: Sequence[float], fs: float, duration_s: float = 32.0) -> np.ndarray:
    a1, a2, b1, b2, c = [float(value) for value in params]
    t = np.arange(0.0, duration_s, 1.0 / fs, dtype=np.float64)
    peak = (b1**a1) * np.power(t, a1 - 1.0) * np.exp(-b1 * t) / max(float(gamma(a1)), 1e-12)
    under = (b2**a2) * np.power(t, a2 - 1.0) * np.exp(-b2 * t) / max(float(c) * float(gamma(a2)), 1e-12)
    hrf = peak - under
    hrf[~np.isfinite(hrf)] = 0.0
    scale = np.max(np.abs(hrf))
    return hrf / scale if scale > 1e-12 else hrf


def _convolve_same(x: np.ndarray, kernel: np.ndarray, fs: float) -> np.ndarray:
    return np.convolve(x, kernel, mode="full")[: len(x)] / fs


def _stimulus_from_markers(time_ms: np.ndarray, desc: np.ndarray, fs: float, n: int, target_desc: int, duration_s: float) -> np.ndarray:
    stimulus = np.zeros(n, dtype=np.float64)
    duration = int(round(duration_s * fs))
    for onset_ms, code in zip(time_ms, desc):
        if int(code) != int(target_desc):
            continue
        start = int(round(float(onset_ms) * fs / 1000.0))
        end = min(n, start + duration)
        if 0 <= start < n:
            stimulus[start:end] = 1.0
    return stimulus


def _optical_to_hbo(intensity: np.ndarray, labels: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    spatial = [label.replace("lowWL", "") for label in labels if label.endswith("lowWL")]
    hbo = np.empty((intensity.shape[0], len(spatial)), dtype=np.float64)
    # Approximate extinction coefficients for 760/850 nm. Units are arbitrary
    # here; correlation and shape diagnostics are invariant to a global scale.
    extinction = np.asarray([[0.586, 1.548], [1.058, 0.691]], dtype=np.float64)
    transform = pinv(extinction)
    for index, name in enumerate(spatial):
        low_idx = labels.index(f"{name}lowWL")
        high_idx = labels.index(f"{name}highWL")
        pair = np.column_stack((intensity[:, low_idx], intensity[:, high_idx]))
        pair = np.maximum(pair, np.nanpercentile(pair[pair > 0], 1) if np.any(pair > 0) else 1e-6)
        baseline = np.maximum(np.nanmedian(pair, axis=0, keepdims=True), 1e-6)
        od = -np.log(pair / baseline)
        concentrations = od @ transform.T
        hbo[:, index] = concentrations[:, 0]
    return hbo, spatial


def _epoch(signal: np.ndarray, fs: float, onset_ms: float, start_s: float, end_s: float) -> np.ndarray | None:
    start = int(round((float(onset_ms) / 1000.0 + start_s) * fs))
    end = int(round((float(onset_ms) / 1000.0 + end_s) * fs))
    if start < 0 or end > signal.shape[0] or end <= start:
        return None
    return signal[start:end].copy()


def _eeg_tensor(epoch: np.ndarray, fs: float, target_len: int) -> np.ndarray:
    channels = []
    for channel in range(epoch.shape[1]):
        freq, time, power = spectrogram(
            epoch[:, channel],
            fs=fs,
            window="hann",
            nperseg=int(round(2.0 * fs)),
            noverlap=int(round(1.9 * fs)),
            detrend=False,
            scaling="density",
            mode="psd",
        )
        freq_mask = (freq >= 1.0) & (freq <= 40.0)
        selected = np.log1p(power[freq_mask].T)
        target_t = np.linspace(time[0], time[-1], target_len)
        interp = np.empty((target_len, selected.shape[1]), dtype=np.float64)
        for f_idx in range(selected.shape[1]):
            interp[:, f_idx] = np.interp(target_t, time, selected[:, f_idx])
        channels.append(interp)
    tensor = np.stack(channels, axis=2)
    baseline = tensor[: target_len // 4].mean(axis=0, keepdims=True)
    tensor = tensor - baseline
    tensor = tensor - np.min(tensor)
    return tensor


def _prepare_trials(raw: RawSession, target_desc: int, epoch_start_s: float, epoch_end_s: float, task_duration_s: float) -> TrialSet:
    eeg_mask = _eeg_channel_mask(raw.eeg_labels)
    eeg = _sos_bandpass(raw.eeg[:, eeg_mask], raw.eeg_fs, 1.0, 40.0)
    hbo, spatial_labels = _optical_to_hbo(raw.fnirs_intensity, raw.fnirs_labels)
    hbo = _sos_bandpass(hbo, raw.fnirs_fs, 0.01, 0.2, order=3)

    continuous_stimulus = _stimulus_from_markers(
        raw.fnirs_marker_time_ms, raw.fnirs_marker_desc, raw.fnirs_fs, len(hbo), target_desc, task_duration_s
    )
    design_signal = _convolve_same(continuous_stimulus, _double_gamma(_canonical_params(), raw.fnirs_fs), raw.fnirs_fs)
    design = np.column_stack((np.ones(len(design_signal)), design_signal))
    beta = np.linalg.lstsq(design, hbo, rcond=None)[0]
    prediction = design @ beta
    residual = hbo - prediction
    dof = max(len(hbo) - design.shape[1], 1)
    sigma2 = np.sum(residual**2, axis=0) / dof
    cov_design = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(sigma2 * cov_design[1, 1], 1e-12))
    t_values = beta[1] / se
    selected = np.argsort(t_values)[-3:][::-1]

    eeg_epochs = []
    hbo_epochs = []
    stimulus_epochs = []
    tensors = []
    trial_indices = []
    trial_onsets = []
    for trial_idx, (eeg_time, eeg_desc, fnirs_time, fnirs_desc) in enumerate(zip(
        raw.eeg_marker_time_ms, raw.eeg_marker_desc, raw.fnirs_marker_time_ms, raw.fnirs_marker_desc
    )):
        if int(eeg_desc) != 16 or int(fnirs_desc) != target_desc:
            continue
        eeg_epoch = _epoch(eeg, raw.eeg_fs, eeg_time, epoch_start_s, epoch_end_s)
        hbo_epoch = _epoch(hbo, raw.fnirs_fs, fnirs_time, epoch_start_s, epoch_end_s)
        if eeg_epoch is None or hbo_epoch is None:
            continue
        baseline_n = int(round(abs(epoch_start_s) * raw.fnirs_fs))
        hbo_epoch = hbo_epoch - hbo_epoch[:baseline_n].mean(axis=0, keepdims=True)
        stimulus = np.zeros(len(hbo_epoch), dtype=np.float64)
        onset_index = int(round(abs(epoch_start_s) * raw.fnirs_fs))
        stimulus[onset_index:onset_index + int(round(task_duration_s * raw.fnirs_fs))] = 1.0
        eeg_epochs.append(eeg_epoch)
        hbo_epochs.append(hbo_epoch)
        stimulus_epochs.append(stimulus)
        tensors.append(_eeg_tensor(eeg_epoch, raw.eeg_fs, len(hbo_epoch)))
        trial_indices.append(trial_idx)
        trial_onsets.append(float(fnirs_time) / 1000.0)
    return TrialSet(
        eeg_epochs=np.stack(eeg_epochs),
        eeg_tensors=np.stack(tensors),
        hbo_epochs=np.stack(hbo_epochs),
        hbo_average=np.stack(hbo_epochs)[:, :, selected].mean(axis=2),
        stimulus_epochs=np.stack(stimulus_epochs),
        selected_channels=[int(value) for value in selected],
        selected_channel_labels=[spatial_labels[int(value)] for value in selected],
        active_channel_t=t_values,
        trial_indices=trial_indices,
        trial_onsets_s=trial_onsets,
    )


def _khatri_rao(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cols = [np.kron(left[:, r], right[:, r]) for r in range(left.shape[1])]
    return np.stack(cols, axis=1)


def _fit_shared_cp(tensors: np.ndarray, rank: int, iterations: int, seed: int) -> dict[str, Any]:
    # tensors: K x T x F x C. CP notation uses temporal Ck, frequency B, spatial A.
    rng = np.random.default_rng(seed)
    _, t_count, f_count, c_count = tensors.shape
    spatial = rng.random((c_count, rank)) + 0.1
    frequency = rng.random((f_count, rank)) + 0.1
    temporal = [rng.normal(size=(t_count, rank)) for _ in range(len(tensors))]
    eps = 1e-8
    for _ in range(iterations):
        kr = _khatri_rao(spatial, frequency)
        gram = (spatial.T @ spatial) * (frequency.T @ frequency) + eps * np.eye(rank)
        temporal = [
            tensor.reshape(t_count, f_count * c_count) @ kr @ np.linalg.pinv(gram)
            for tensor in tensors
        ]

        gram_acc = np.zeros((rank, rank), dtype=np.float64)
        numer = np.zeros((c_count, rank), dtype=np.float64)
        for tensor, time_factor in zip(tensors, temporal):
            kr = _khatri_rao(frequency, time_factor)
            gram_acc += (time_factor.T @ time_factor) * (frequency.T @ frequency)
            numer += tensor.transpose(2, 0, 1).reshape(c_count, t_count * f_count) @ kr
        spatial = np.maximum(numer @ np.linalg.pinv(gram_acc + eps * np.eye(rank)), eps)

        gram_acc = np.zeros((rank, rank), dtype=np.float64)
        numer = np.zeros((f_count, rank), dtype=np.float64)
        for tensor, time_factor in zip(tensors, temporal):
            kr = _khatri_rao(spatial, time_factor)
            gram_acc += (time_factor.T @ time_factor) * (spatial.T @ spatial)
            numer += tensor.transpose(1, 0, 2).reshape(f_count, t_count * c_count) @ kr
        frequency = np.maximum(numer @ np.linalg.pinv(gram_acc + eps * np.eye(rank)), eps)

        for r in range(rank):
            scale = max(float(np.linalg.norm(spatial[:, r])), eps)
            spatial[:, r] /= scale
            for k in range(len(temporal)):
                temporal[k][:, r] *= scale
            scale = max(float(np.linalg.norm(frequency[:, r])), eps)
            frequency[:, r] /= scale
            for k in range(len(temporal)):
                temporal[k][:, r] *= scale
    return {"spatial": spatial, "frequency": frequency, "temporal": temporal}


def _project_temporal(tensor: np.ndarray, spatial: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    t_count, f_count, c_count = tensor.shape
    kr = _khatri_rao(spatial, frequency)
    gram = (spatial.T @ spatial) * (frequency.T @ frequency)
    return tensor.reshape(t_count, f_count * c_count) @ kr @ np.linalg.pinv(gram + 1e-8 * np.eye(gram.shape[0]))


def _trca_filter(temporal: Sequence[np.ndarray]) -> np.ndarray:
    rank = temporal[0].shape[1]
    q = np.zeros((rank, rank), dtype=np.float64)
    s = np.zeros((rank, rank), dtype=np.float64)
    centered = [x - x.mean(axis=0, keepdims=True) for x in temporal]
    for i, xi in enumerate(centered):
        q += xi.T @ xi
        for j, xj in enumerate(centered):
            if i == j:
                continue
            s += xi.T @ xj
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(q + 1e-8 * np.eye(rank)) @ s)
    index = int(np.argmax(np.real(eigvals)))
    weights = np.real(eigvecs[:, index])
    norm = np.linalg.norm(weights)
    return weights / norm if norm > 1e-12 else weights


def _standardize_component(component: np.ndarray) -> np.ndarray:
    return (component - component.mean()) / max(float(component.std()), 1e-12)


def _fit_linear(x_trials: Sequence[np.ndarray], y_trials: Sequence[np.ndarray]) -> np.ndarray:
    x = np.concatenate([trial.reshape(-1, 1) for trial in x_trials], axis=0)
    y = np.concatenate([trial.reshape(-1, 1) for trial in y_trials], axis=0)
    design = np.column_stack((x, np.ones(len(x))))
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _predict_linear(x_trials: Sequence[np.ndarray], coeff: np.ndarray) -> list[np.ndarray]:
    return [(np.column_stack((trial.reshape(-1, 1), np.ones(len(trial)))) @ coeff).reshape(-1) for trial in x_trials]


def _shape_penalty(params: Sequence[float]) -> float:
    a1, a2, b1, b2, _ = [float(value) for value in params]
    checks = [
        (a1 / max(b1, 1e-12), 3.0, 7.0),
        (a2 / max(b2, 1e-12), 9.0, 18.0),
        (2.35 * np.sqrt(max(a1 - 1.0, 0.0)) / max(b1, 1e-12), 3.0, 6.0),
        (2.35 * np.sqrt(max(a2 - 1.0, 0.0)) / max(b2, 1e-12), 7.0, 11.0),
    ]
    total = 0.0
    for value, low, high in checks:
        if value < low:
            total += (low - value) ** 2
        elif value > high:
            total += (value - high) ** 2
    return total


def _fit_hrf(component_trials: Sequence[np.ndarray], target_trials: Sequence[np.ndarray], fs: float, mode: str) -> tuple[np.ndarray, np.ndarray, float]:
    def prediction_for(params: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, float]:
        hrf = _double_gamma(params, fs)
        convolved = [_convolve_same(_standardize_component(trial), hrf, fs) for trial in component_trials]
        coeff = _fit_linear(convolved, target_trials)
        pred = _predict_linear(convolved, coeff)
        mse = float(np.mean((np.concatenate(pred) - np.concatenate(target_trials)) ** 2))
        return pred, coeff, mse
    if mode == "canonical":
        _, coeff, mse = prediction_for(_canonical_params())
        return _canonical_params(), coeff, mse
    bounds = [(2.0, 10.0), (6.0, 25.0), (0.5, 2.0), (0.05, 1.5), (0.2, 15.0)]
    def objective(params: np.ndarray) -> float:
        _, _, mse = prediction_for(params)
        return mse + 0.05 * _shape_penalty(params)
    result = minimize(objective, _canonical_params(), method="L-BFGS-B", bounds=bounds, options={"maxiter": 120})
    params = np.asarray(result.x if result.success else _canonical_params(), dtype=np.float64)
    _, coeff, mse = prediction_for(params)
    return params, coeff, mse


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, train_reference: np.ndarray) -> dict[str, float]:
    mean = train_reference.mean()
    std = max(float(train_reference.std()), 1e-12)
    truth = (y_true - mean) / std
    pred = (y_pred - mean) / std
    error = pred - truth
    r2 = 1.0 - float(np.sum(error**2)) / max(float(np.sum((truth - truth.mean()) ** 2)), 1e-12)
    pcc = float(np.corrcoef(truth.reshape(-1), pred.reshape(-1))[0, 1]) if np.std(pred) > 1e-12 else float("nan")
    return {
        "mse": float(np.mean(error**2)),
        "r2": r2,
        "pcc": pcc,
        "amplitude_ratio": float(np.std(y_pred) / max(float(np.std(y_true)), 1e-12)),
        "mean_bias": float(np.mean(y_pred - y_true)),
    }


def _evaluate_loso(trials: TrialSet, rank: int, cp_iterations: int, seed: int, fs: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictions = []
    for heldout in range(len(trials.eeg_tensors)):
        train_idx = [idx for idx in range(len(trials.eeg_tensors)) if idx != heldout]
        cp = _fit_shared_cp(trials.eeg_tensors[train_idx], rank=rank, iterations=cp_iterations, seed=seed + heldout)
        weights = _trca_filter(cp["temporal"])
        train_components = [_standardize_component(component @ weights) for component in cp["temporal"]]
        test_temporal = _project_temporal(trials.eeg_tensors[heldout], cp["spatial"], cp["frequency"])
        test_component = _standardize_component(test_temporal @ weights)
        train_targets = [trials.hbo_average[idx] for idx in train_idx]
        for mode in ("canonical", "optimized"):
            params, coeff, train_mse = _fit_hrf(train_components, train_targets, fs, mode)
            hrf = _double_gamma(params, fs)
            train_drivers = [_convolve_same(component, hrf, fs) for component in train_components]
            test_driver = _convolve_same(test_component, hrf, fs)
            pred = _predict_linear([test_driver], coeff)[0]
            metrics = _metrics(trials.hbo_average[heldout], pred, np.concatenate(train_targets))
            a1, a2, b1, b2, c = [float(value) for value in params]
            rows.append({
                "model": "TRTD",
                "validation": "leave_one_trial",
                "heldout_trial": int(heldout),
                "trial_index": int(trials.trial_indices[heldout]),
                "hrf_mode": mode,
                "train_mse_raw": train_mse,
                "hrf_a1": a1,
                "hrf_a2": a2,
                "hrf_b1": b1,
                "hrf_b2": b2,
                "hrf_c": c,
                "hrf_ttp": a1 / max(b1, 1e-12),
                "hrf_ttu": a2 / max(b2, 1e-12),
                **metrics,
            })
            if mode == "optimized":
                predictions.append({
                    "heldout": heldout,
                    "component": test_component,
                    "driver": test_driver,
                    "prediction": pred,
                    "truth": trials.hbo_average[heldout],
                    "params": params,
                    "spatial": cp["spatial"],
                    "frequency": cp["frequency"],
                    "weights": weights,
                })
    # Upper-bound in-sample fit: train CP and HRF on all task trials, then inspect fit.
    cp_all = _fit_shared_cp(trials.eeg_tensors, rank=rank, iterations=cp_iterations, seed=seed + 999)
    weights_all = _trca_filter(cp_all["temporal"])
    components_all = [_standardize_component(component @ weights_all) for component in cp_all["temporal"]]
    for mode in ("canonical", "optimized"):
        params, coeff, train_mse = _fit_hrf(components_all, list(trials.hbo_average), fs, mode)
        hrf = _double_gamma(params, fs)
        pred = np.stack(_predict_linear([_convolve_same(component, hrf, fs) for component in components_all], coeff))
        metrics = _metrics(trials.hbo_average, pred, trials.hbo_average.reshape(-1))
        a1, a2, b1, b2, c = [float(value) for value in params]
        rows.append({
            "model": "TRTD",
            "validation": "in_sample_upper_bound",
            "heldout_trial": -1,
            "trial_index": -1,
            "hrf_mode": mode,
            "train_mse_raw": train_mse,
            "hrf_a1": a1,
            "hrf_a2": a2,
            "hrf_b1": b1,
            "hrf_b2": b2,
            "hrf_c": c,
            "hrf_ttp": a1 / max(b1, 1e-12),
            "hrf_ttu": a2 / max(b2, 1e-12),
            **metrics,
        })
    return rows, {"predictions": predictions, "cp_all": cp_all, "weights_all": weights_all}


def _evaluate_baselines(trials: TrialSet) -> list[dict[str, Any]]:
    rows = []
    for heldout in range(len(trials.hbo_average)):
        train_idx = [idx for idx in range(len(trials.hbo_average)) if idx != heldout]
        train = trials.hbo_average[train_idx]
        truth = trials.hbo_average[heldout]
        mean_pred = train.mean(axis=0)
        persistence = np.empty_like(truth)
        persistence[0] = train[:, 0].mean()
        persistence[1:] = truth[:-1]
        for name, pred in (("trial_mean", mean_pred), ("self_persistence", persistence)):
            rows.append({
                "model": name,
                "validation": "leave_one_trial",
                "heldout_trial": int(heldout),
                "trial_index": int(trials.trial_indices[heldout]),
                "hrf_mode": "none",
                **_metrics(truth, pred, train.reshape(-1)),
            })
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["validation"]), str(row["hrf_mode"]))
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        aggregate: dict[str, Any] = {"model": key[0], "validation": key[1], "hrf_mode": key[2], "folds": len(values)}
        for metric in ("mse", "r2", "pcc", "amplitude_ratio", "mean_bias", "hrf_ttp", "hrf_ttu", "hrf_c"):
            finite = [float(row[metric]) for row in values if metric in row and np.isfinite(float(row[metric]))]
            if finite:
                aggregate[metric] = float(np.mean(finite))
                aggregate[f"{metric}_median"] = float(np.median(finite))
        output.append(aggregate)
    return output


def _plot_outputs(trials: TrialSet, fit_artifacts: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], run_dir: Path, fs: float) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    predictions = fit_artifacts["predictions"]
    if predictions:
        pred = predictions[0]
        t = np.arange(len(pred["truth"])) / fs - 5.0
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(t, pred["component"], color="#7c3aed", label="TRTD temporal component")
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.2)
        axes[1].plot(t, pred["driver"], color="#f59e0b", label="HRF-convolved driver")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.2)
        axes[2].plot(t, pred["truth"], color="#2563eb", label="true active-channel HbO")
        axes[2].plot(t, pred["prediction"], color="#dc2626", linestyle="--", label="predicted HbO")
        axes[2].legend(loc="upper right")
        axes[2].grid(alpha=0.2)
        axes[3].plot(t, pred["truth"] - pred["prediction"], color="#111827", label="residual")
        axes[3].axhline(0, color="#6b7280", linewidth=0.8)
        axes[3].legend(loc="upper right")
        axes[3].grid(alpha=0.2)
        axes[3].set_xlabel("seconds from task onset")
        params = pred["params"]
        fig.suptitle(
            f"LOO trial {pred['heldout']} | TTP={params[0] / max(params[2], 1e-12):.2f}s "
            f"TTU={params[1] / max(params[3], 1e-12):.2f}s c={params[4]:.2f}"
        )
        fig.tight_layout()
        for suffix, dpi in (("svg", None), ("png", 300)):
            path = run_dir / "figures" / f"loo_trajectory_example.{suffix}"
            fig.savefig(path, dpi=dpi)
            artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
        plt.close(fig)

        ordered = sorted(predictions, key=lambda item: int(item["heldout"]))
        truth = np.stack([item["truth"] for item in ordered])
        prediction = np.stack([item["prediction"] for item in ordered])
        residual = truth - prediction
        t = np.arange(truth.shape[1]) / fs - 5.0
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
        vmax = float(np.max(np.abs(truth)))
        for axis, matrix, title in (
            (axes[0, 0], truth, "true HbO"),
            (axes[0, 1], prediction, "predicted HbO"),
            (axes[0, 2], residual, "residual"),
        ):
            limit = vmax if title != "residual" else float(np.max(np.abs(residual)))
            image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, extent=[t[0], t[-1], len(matrix), 0])
            axis.set_title(title)
            axis.set_ylabel("held-out trial")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
        for axis, matrix, title, color in (
            (axes[1, 0], truth, "true mean +/- sd", "#2563eb"),
            (axes[1, 1], prediction, "predicted mean +/- sd", "#dc2626"),
            (axes[1, 2], residual, "residual mean +/- sd", "#111827"),
        ):
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0)
            axis.plot(t, mean, color=color)
            axis.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.18)
            axis.axhline(0, color="#6b7280", linewidth=0.8)
            axis.set_title(title)
            axis.set_xlabel("seconds from task onset")
            axis.grid(alpha=0.2)
        fig.suptitle("Leave-one-trial TRTD + optimized HRF: true vs predicted active-channel HbO")
        fig.tight_layout()
        for suffix, dpi in (("svg", None), ("png", 300)):
            path = run_dir / "figures" / f"loo_trial_heatmap.{suffix}"
            fig.savefig(path, dpi=dpi)
            artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
        plt.close(fig)

        trajectory_rows = []
        for item in ordered:
            for index, time_s in enumerate(t):
                trajectory_rows.append({
                    "heldout_trial": int(item["heldout"]),
                    "time_s": float(time_s),
                    "truth_hbo": float(item["truth"][index]),
                    "predicted_hbo": float(item["prediction"][index]),
                    "residual_hbo": float(item["truth"][index] - item["prediction"][index]),
                    "trtd_component": float(item["component"][index]),
                    "hrf_driver": float(item["driver"][index]),
                })
        _write_csv(run_dir / "figure_data" / "loo_trial_trajectories.csv", trajectory_rows)

    summary = _aggregate(rows)
    labels = [f"{row['model']}\n{row['validation']}\n{row['hrf_mode']}" for row in summary]
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.85), 7), sharex=True)
    x = np.arange(len(summary))
    axes[0].bar(x, [float(row.get("r2", np.nan)) for row in summary], color="#2563eb")
    axes[0].set_ylabel("R2")
    axes[0].axhline(0, color="#111827", linewidth=0.8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [float(row.get("amplitude_ratio", np.nan)) for row in summary], color="#16a34a")
    axes[1].set_ylabel("amplitude ratio")
    axes[1].set_xticks(x, labels=labels, rotation=45, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Raw-session Lin-style NVC performance")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.32, top=0.88, hspace=0.12)
    for suffix, dpi in (("svg", None), ("png", 300)):
        path = run_dir / "figures" / f"performance_summary.{suffix}"
        fig.savefig(path, dpi=dpi)
        artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
    plt.close(fig)

    spatial = fit_artifacts["cp_all"]["spatial"]
    frequency = fit_artifacts["cp_all"]["frequency"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].imshow(spatial, aspect="auto", cmap="viridis")
    axes[0].set_title("Shared spatial CP factors")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("EEG channel index")
    axes[1].imshow(frequency, aspect="auto", cmap="viridis", origin="lower")
    axes[1].set_title("Shared frequency CP factors")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("frequency bin")
    fig.tight_layout()
    for suffix, dpi in (("svg", None), ("png", 300)):
        path = run_dir / "figures" / f"trtd_shared_factors.{suffix}"
        fig.savefig(path, dpi=dpi)
        artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
    plt.close(fig)
    return artifacts


def _summary_markdown(summary: Sequence[Mapping[str, Any]], selected_labels: Sequence[str]) -> str:
    lines = [
        "# Lin 2024 raw-session TRTD diagnostic",
        "",
        "This run reads one continuous raw session and does not use Croce cache windows.",
        "",
        f"Selected active HbO channels: {', '.join(selected_labels)}",
        "",
        "| Model | Validation | HRF | R2 | PCC | MSE | amplitude ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['validation']} | {row['hrf_mode']} | "
            f"{row.get('r2', float('nan')):.6f} | {row.get('pcc', float('nan')):.6f} | "
            f"{row.get('mse', float('nan')):.6f} | {row.get('amplitude_ratio', float('nan')):.6f} |"
        )
    lines.extend(["", "See `summary.json`, `metrics.csv`, `fold_metrics.csv`, and `figures/` for complete artifacts.", ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    data_root = (REPO_ROOT / args.data_root).resolve()
    raw = _load_raw_session(data_root, args.subject, args.session_index)
    run_dir = Path(args.output_dir).resolve() if args.output_dir else (
        REPO_ROOT / "experiments" / "runs" / "physiology_semantic_tokenizer" / "e0_teacher_validity" /
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lin2024_raw_session_trtd_s{args.subject:02d}_sess{args.session_index + 1}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "figure_data").mkdir()

    trials = _prepare_trials(raw, args.target_desc, args.epoch_start_s, args.epoch_end_s, args.task_duration_s)
    fold_rows, fit_artifacts = _evaluate_loso(trials, args.rank, args.cp_iterations, args.seed, raw.fnirs_fs)
    baseline_rows = _evaluate_baselines(trials)
    all_rows = fold_rows + baseline_rows
    summary_rows = _aggregate(all_rows)
    figures = _plot_outputs(trials, fit_artifacts, all_rows, run_dir, raw.fnirs_fs)

    _write_csv(run_dir / "fold_metrics.csv", all_rows)
    _write_csv(run_dir / "metrics.csv", summary_rows)
    _write_csv(run_dir / "active_channel_t_values.csv", [
        {"channel_index": idx, "channel_label": label, "t_value": float(trials.active_channel_t[idx])}
        for idx, label in enumerate(raw.fnirs_spatial_labels)
    ])
    _write_json(run_dir / "summary.json", {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_diagnostic",
        "subject": args.subject,
        "session_index_zero_based": args.session_index,
        "task": "mental_arithmetic" if args.target_desc == 1 else str(args.target_desc),
        "input_files": {
            "eeg_cnt": str((data_root / "EEG_01-29" / f"subject {args.subject:02d}" / "with occular artifact" / "cnt.mat").relative_to(REPO_ROOT)),
            "eeg_mrk": str((data_root / "EEG_01-29" / f"subject {args.subject:02d}" / "with occular artifact" / "mrk.mat").relative_to(REPO_ROOT)),
            "fnirs_cnt": str((data_root / "NIRS_01-29" / f"subject {args.subject:02d}" / "cnt.mat").relative_to(REPO_ROOT)),
            "fnirs_mrk": str((data_root / "NIRS_01-29" / f"subject {args.subject:02d}" / "mrk.mat").relative_to(REPO_ROOT)),
        },
        "paper_alignment": {
            "followed": [
                "continuous raw cnt/mrk inputs",
                "20s epochs from -5 to 15s",
                "EEG 1-40 Hz filtering",
                "0.5Hz time-frequency representation",
                "shared spatial/frequency tensor factors with trial-specific temporal factors",
                "TRCA temporal component filter",
                "fNIRS 0.01-0.2Hz filtering",
                "GLM active-channel selection",
                "subject-specific double-gamma HRF",
                "leave-one-trial validation",
            ],
            "dataset_limited": [
                "uses EEG+NIRS BCI mental arithmetic rather than Lin finger tapping",
                "uses approximate MBLL from optical intensity rather than HOMER HbO pipeline",
                "no short-distance fNIRS regressors are available",
                "no BSS-CCA muscle-artifact removal was applied",
            ],
        },
        "selected_channels": trials.selected_channel_labels,
        "trial_indices": trials.trial_indices,
        "trial_onsets_s": trials.trial_onsets_s,
        "summary_rows": summary_rows,
        "figures": figures,
        "interpretation": {
            "status": "diagnostic_only",
            "upper_bound_scope": "single subject/session, same-session leave-one-trial and in-sample fit",
        },
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
    })
    _write_json(run_dir / "environment.json", {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines(),
    })
    (run_dir / "summary.md").write_text(_summary_markdown(summary_rows, trials.selected_channel_labels), encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EEG+NIRS Single-Trial")
    parser.add_argument("--subject", type=int, default=19)
    parser.add_argument("--session-index", type=int, default=1, help="0-based; 1 is mental-arithmetic session 2")
    parser.add_argument("--target-desc", type=int, default=1, help="NIRS MA marker code")
    parser.add_argument("--epoch-start-s", type=float, default=-5.0)
    parser.add_argument("--epoch-end-s", type=float, default=15.0)
    parser.add_argument("--task-duration-s", type=float, default=10.0)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--cp-iterations", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
