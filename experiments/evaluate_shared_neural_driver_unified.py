#!/usr/bin/env python3
"""Retest Croce-2017 and Lin-2024-inspired shared neural drivers.

The diagnostic deliberately reads observations through
``UnifiedPhysiologyWindowDataset``.  It never reads the derived Croce target
cache.  Single-Trial raw and admitted v3 EEG branches are evaluated on the
same fNIRS trials so that the effect of artifact correction is paired.

This is an implementation-level, exploratory model-family comparison, not a
direct reproduction of either paper and not an E0 gate by itself.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.signal import butter, sosfiltfilt, spectrogram
from scipy.special import gamma
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.unified_physiology import UnifiedPhysiologyWindowDataset
from src.inference.neurovascular_smc import NeurovascularSMCFilter, double_gamma_hrf


SCHEMA = "shared_neural_driver_unified_retest_v1"


@dataclass(frozen=True)
class Trial:
    condition_id: str
    dataset_id: str
    subject: str
    record_id: str
    event_index: int
    eeg: np.ndarray  # [T_eeg, C_eeg]
    fnirs: np.ndarray  # [T_fnirs, C_fnirs], baseline corrected
    fnirs_channel_names: tuple[str, ...]
    fnirs_roles: tuple[str, ...]
    eeg_artifact_fraction: float
    eeg_channel_names: tuple[str, ...] = ()
    eeg_positions: np.ndarray | None = None
    fnirs_positions: np.ndarray | None = None
    eeg_valid_mask: np.ndarray | None = None
    fnirs_valid_mask: np.ndarray | None = None


@dataclass
class Prediction:
    condition_id: str
    dataset_id: str
    subject: str
    validation: str
    heldout_trial: int
    model: str
    hrf_mode: str
    truth: np.ndarray
    estimate: np.ndarray
    selected_channels: tuple[str, ...]
    driver: np.ndarray | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        writer.writerows([{key: _jsonable(value) for key, value in row.items()} for row in rows])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_payload() -> dict[str, Any]:
    def call(*args: str) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, check=False, capture_output=True, text=True).stdout.strip()

    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "status_short": call("git", "status", "--short"),
    }


def _load_trials(config: Mapping[str, Any]) -> tuple[dict[str, dict[str, list[Trial]]], list[dict[str, Any]]]:
    data_cfg = config["data"]
    cache_root = Path(str(data_cfg["cache_root"]))
    if not cache_root.is_absolute():
        cache_root = REPO_ROOT / cache_root
    grouped: dict[str, dict[str, list[Trial]]] = {}
    contracts: list[dict[str, Any]] = []
    baseline_n = int(round(float(data_cfg["baseline_duration_s"]) * 10.0))
    for condition in data_cfg["conditions"]:
        dataset = UnifiedPhysiologyWindowDataset(
            cache_root=cache_root,
            dataset_ids=(condition["dataset_id"],),
            window_duration_s=float(data_cfg["window_duration_s"]),
            window_offset_s=float(data_cfg["window_offset_s"]),
            eeg_signal_branch=str(condition["eeg_signal_branch"]),
        )
        allowed_subjects = {str(value) for value in condition["subjects"]}
        selected = []
        for index, ref in enumerate(dataset.windows):
            if ref.record.canonical_subject_id not in allowed_subjects:
                continue
            if ref.record.base_record_id != condition["record_id"]:
                continue
            if str(ref.event.get("label")) != str(condition["target_label"]):
                continue
            selected.append(index)
        selected.sort(key=lambda index: (
            dataset.windows[index].record.canonical_subject_id,
            dataset.windows[index].record.base_record_id,
            int(dataset.windows[index].event.get("event_index", -1)),
            float(dataset.windows[index].window_offset_s),
        ))

        per_subject: dict[str, list[Trial]] = defaultdict(list)
        for index in selected:
            sample = dataset[index]
            subject = str(sample["subject"])
            if len(per_subject[subject]) >= int(condition["max_trials_per_subject"]):
                continue
            fnirs = np.asarray(sample["fnirs"], dtype=np.float64).T
            fnirs = fnirs - fnirs[:baseline_n].mean(axis=0, keepdims=True)
            artifact_mask = np.asarray(sample["artifact_mask"]["eeg"], dtype=bool)
            per_subject[subject].append(Trial(
                condition_id=str(condition["condition_id"]),
                dataset_id=str(condition["dataset_id"]),
                subject=subject,
                record_id=str(sample["record_id"]),
                event_index=int(sample["event"].get("event_index", len(per_subject[subject]))),
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
                eeg_valid_mask=np.asarray(
                    sample["valid_mask"]["eeg"], dtype=bool
                ).copy(),
                fnirs_valid_mask=np.asarray(
                    sample["valid_mask"]["fnirs"], dtype=bool
                ).copy(),
            ))
        missing = sorted(allowed_subjects - set(per_subject))
        if missing:
            raise RuntimeError(f"{condition['condition_id']}: missing selected trials for {missing}")
        for subject, trials in per_subject.items():
            if len(trials) < 3:
                raise RuntimeError(f"{condition['condition_id']}:{subject} has only {len(trials)} trials")
        grouped[str(condition["condition_id"])] = dict(per_subject)
        # Snapshot after selected records have been loaded so branch/fallback
        # counters describe the data actually consumed by this run.
        contracts.append({
            "condition_id": condition["condition_id"],
            **dataset.contract_summary(),
        })
    return grouped, contracts


def _canonical_lin_params() -> np.ndarray:
    return np.asarray([6.0, 16.0, 1.0, 1.0, 6.0], dtype=np.float64)


def _lin_hrf(params: Sequence[float], fs: float, duration_s: float = 32.0) -> np.ndarray:
    a1, a2, b1, b2, c = [float(value) for value in params]
    t = np.arange(0.0, duration_s, 1.0 / fs, dtype=np.float64)
    peak = (b1**a1) * np.power(t, a1 - 1.0) * np.exp(-b1 * t) / max(float(gamma(a1)), 1e-12)
    under = (b2**a2) * np.power(t, a2 - 1.0) * np.exp(-b2 * t) / max(c * float(gamma(a2)), 1e-12)
    hrf = peak - under
    hrf[~np.isfinite(hrf)] = 0.0
    scale = float(np.max(np.abs(hrf)))
    return hrf / scale if scale > 1e-12 else hrf


def _convolve(signal: np.ndarray, kernel: np.ndarray, fs: float) -> np.ndarray:
    return np.convolve(np.asarray(signal), np.asarray(kernel), mode="full")[: len(signal)] / fs


def _select_active_hbo(
    train_trials: Sequence[Trial],
    *,
    baseline_duration_s: float,
    task_duration_s: float,
    count: int,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    roles = np.asarray(train_trials[0].fnirs_roles, dtype=object)
    hbo_indices = np.flatnonzero(roles == "HbO")
    if len(hbo_indices) < count:
        raise ValueError(f"only {len(hbo_indices)} HbO channels available")
    mean_trajectory = np.stack([trial.fnirs[:, hbo_indices] for trial in train_trials]).mean(axis=0)
    fs = 10.0
    stimulus = np.zeros(mean_trajectory.shape[0], dtype=np.float64)
    start = int(round(baseline_duration_s * fs))
    stop = min(len(stimulus), start + int(round(task_duration_s * fs)))
    stimulus[start:stop] = 1.0
    design_signal = _convolve(stimulus, _lin_hrf(_canonical_lin_params(), fs), fs)
    design = np.column_stack((np.ones(len(design_signal)), design_signal))
    beta = np.linalg.lstsq(design, mean_trajectory, rcond=None)[0]
    residual = mean_trajectory - design @ beta
    dof = max(len(mean_trajectory) - design.shape[1], 1)
    sigma2 = np.sum(residual**2, axis=0) / dof
    cov = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(sigma2 * cov[1, 1], 1e-12))
    scores = beta[1] / se
    local = np.argsort(scores)[-count:][::-1]
    selected = hbo_indices[local]
    names = tuple(train_trials[0].fnirs_channel_names[int(index)] for index in selected)
    return selected, names, scores


def _targets(trials: Sequence[Trial], indices: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(trial.fnirs[:, indices].mean(axis=1), dtype=np.float64) for trial in trials]


def _downsample_eeg_power(eeg: np.ndarray, source_hz: float = 200.0, target_hz: float = 10.0) -> np.ndarray:
    factor = int(round(source_hz / target_hz))
    usable = (len(eeg) // factor) * factor
    power = np.asarray(eeg[:usable], dtype=np.float64) ** 2
    return power.reshape(usable // factor, factor, power.shape[1]).mean(axis=1)


def _lowpass(signal: np.ndarray, fs: float = 10.0, cutoff: float = 0.2) -> np.ndarray:
    sos = butter(4, min(cutoff / (0.5 * fs), 0.9), btype="lowpass", output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))


def _fit_croce_model(
    eeg_events: Sequence[np.ndarray],
    target_events: Sequence[np.ndarray],
    *,
    particles: int,
    resample_threshold: float,
    hrf_duration_s: float,
    seed: int,
) -> tuple[NeurovascularSMCFilter, dict[str, Any]]:
    eeg_power = [_downsample_eeg_power(event) for event in eeg_events]
    stacked_power = np.concatenate(eeg_power, axis=0)
    eeg_mean = stacked_power.mean(axis=0)
    eeg_std = np.maximum(stacked_power.std(axis=0), 1e-8)
    normalized = [(event - eeg_mean) / eeg_std for event in eeg_power]
    stacked = np.concatenate(normalized, axis=0)
    pca_mean = stacked.mean(axis=0)
    _, _, vt = np.linalg.svd(stacked - pca_mean, full_matrices=False)
    loading = vt[0].copy()
    pivot = int(np.argmax(np.abs(loading)))
    if loading[pivot] < 0:
        loading *= -1.0
    raw_pc = [(event - pca_mean) @ loading for event in normalized]
    pc_scale = max(float(np.std(np.concatenate(raw_pc))), 1e-8)
    raw_pc = [event / pc_scale for event in raw_pc]
    slow_pc_unscaled = [_lowpass(event) for event in raw_pc]
    slow_scale = max(float(np.std(np.concatenate(slow_pc_unscaled))), 1e-8)
    slow_pc = [event / slow_scale for event in slow_pc_unscaled]

    lag = np.concatenate([event[:-1] for event in slow_pc])
    current = np.concatenate([event[1:] for event in slow_pc])
    alpha = float(np.dot(current, lag) / max(float(np.dot(lag, lag)), 1e-8))
    alpha = float(np.clip(alpha, 0.90, 0.999))
    q = max(float(np.var(current - alpha * lag)), 1e-6)

    target_concat = np.concatenate(target_events)
    fnirs_mean = float(target_concat.mean())
    fnirs_std = max(float(target_concat.std()), 1e-8)
    target_norm = [(event - fnirs_mean) / fnirs_std for event in target_events]
    hrf = double_gamma_hrf(10.0, duration_s=hrf_duration_s).astype(np.float64)
    hrf_drivers = [np.convolve(event, hrf, mode="full")[: len(event)] for event in slow_pc]
    driver_concat = np.concatenate(hrf_drivers)
    target_norm_concat = np.concatenate(target_norm)
    denominator = max(float(np.dot(driver_concat, driver_concat)), 1e-8)
    h_fnirs = float(np.dot(target_norm_concat, driver_concat) / denominator)
    residual = target_norm_concat - h_fnirs * driver_concat
    r_fnirs = max(float(np.var(residual)), 1e-4)
    raw_concat = np.concatenate(raw_pc)
    slow_concat = np.concatenate(slow_pc)
    r_eeg = max(float(np.var(raw_concat - slow_concat)), 0.05)

    model = NeurovascularSMCFilter(
        hrf_kernel=hrf,
        state_transition_matrix=np.asarray([[alpha]], dtype=np.float64),
        process_noise_cov=np.asarray([[q]], dtype=np.float64),
        eeg_forward=np.ones((1, 1), dtype=np.float64),
        fnirs_forward=np.asarray([[h_fnirs]], dtype=np.float64),
        eeg_noise_cov=np.asarray([[r_eeg]], dtype=np.float64),
        fnirs_noise_cov=np.asarray([[r_fnirs]], dtype=np.float64),
        n_particles=int(particles),
        resample_threshold=float(resample_threshold),
        seed=int(seed),
    )
    state = {
        "eeg_mean": eeg_mean,
        "eeg_std": eeg_std,
        "pca_mean": pca_mean,
        "loading": loading,
        "pc_scale": pc_scale,
        "fnirs_mean": fnirs_mean,
        "fnirs_std": fnirs_std,
        "alpha": alpha,
        "q": q,
        "h_fnirs": h_fnirs,
        "r_eeg": r_eeg,
        "r_fnirs": r_fnirs,
    }
    return model, state


def _apply_croce_eeg(eeg: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    power = _downsample_eeg_power(eeg)
    normalized = (power - state["eeg_mean"]) / state["eeg_std"]
    pc = (normalized - state["pca_mean"]) @ state["loading"]
    return np.asarray(pc / state["pc_scale"], dtype=np.float64)


def _state_to_fnirs(model: NeurovascularSMCFilter, state: np.ndarray, fit: Mapping[str, Any]) -> np.ndarray:
    convolved = np.convolve(np.asarray(state).reshape(-1), model.hrf_kernel, mode="full")[: len(state)]
    normalized = convolved * float(model.H_fnirs[0, 0])
    return normalized * float(fit["fnirs_std"]) + float(fit["fnirs_mean"])


def _croce_predictions(
    train_trials: Sequence[Trial],
    test_trials: Sequence[Trial],
    selected: np.ndarray,
    selected_names: tuple[str, ...],
    *,
    condition_id: str,
    validation: str,
    heldout_trial: int,
    config: Mapping[str, Any],
    seed: int,
) -> list[Prediction]:
    train_targets = _targets(train_trials, selected)
    test_targets = _targets(test_trials, selected)
    model, fit = _fit_croce_model(
        [trial.eeg for trial in train_trials],
        train_targets,
        particles=int(config["n_particles"]),
        resample_threshold=float(config["resample_threshold"]),
        hrf_duration_s=float(config["hrf_duration_s"]),
        seed=seed,
    )
    predictions: list[Prediction] = []
    for offset, (trial, truth) in enumerate(zip(test_trials, test_targets)):
        np.random.seed(seed + offset)
        eeg_observation = _apply_croce_eeg(trial.eeg, fit)[:, None]
        fnirs_observation = ((truth - fit["fnirs_mean"]) / fit["fnirs_std"])[:, None]
        result = model.filter(eeg_observation, fnirs_observation, return_particles=False)
        outputs = {
            "croce_joint": (
                result.fnirs_reconstructed[:, 0] * fit["fnirs_std"] + fit["fnirs_mean"],
                result.state_mean[:, 0],
            ),
            "croce_eeg_only": (
                _state_to_fnirs(model, result.eeg_only_state_mean, fit),
                result.eeg_only_state_mean[:, 0],
            ),
            "croce_fnirs_only": (
                _state_to_fnirs(model, result.fnirs_only_state_mean, fit),
                result.fnirs_only_state_mean[:, 0],
            ),
        }
        for name, (estimate, driver) in outputs.items():
            predictions.append(Prediction(
                condition_id=condition_id,
                dataset_id=trial.dataset_id,
                subject=trial.subject,
                validation=validation,
                heldout_trial=heldout_trial if len(test_trials) == 1 else offset,
                model=name,
                hrf_mode="canonical",
                truth=truth,
                estimate=np.asarray(estimate),
                selected_channels=selected_names,
                driver=np.asarray(driver),
            ))
    return predictions


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
        mask = (freq >= 1.0) & (freq <= 40.0)
        selected = np.log1p(power[mask].T)
        target_t = np.linspace(time[0], time[-1], target_len)
        interp = np.empty((target_len, selected.shape[1]), dtype=np.float64)
        for index in range(selected.shape[1]):
            interp[:, index] = np.interp(target_t, time, selected[:, index])
        channels.append(interp)
    tensor = np.stack(channels, axis=2)
    tensor -= tensor[: target_len // 4].mean(axis=0, keepdims=True)
    tensor -= np.min(tensor)
    return tensor


def _khatri_rao(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([np.kron(left[:, rank], right[:, rank]) for rank in range(left.shape[1])], axis=1)


def _fit_shared_cp(tensors: np.ndarray, rank: int, iterations: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    _, t_count, f_count, c_count = tensors.shape
    spatial = rng.random((c_count, rank)) + 0.1
    frequency = rng.random((f_count, rank)) + 0.1
    temporal = [rng.normal(size=(t_count, rank)) for _ in range(len(tensors))]
    eps = 1e-8
    for _ in range(iterations):
        # C-order [T,F,C] flattening uses compound index f*C+c.
        kr = _khatri_rao(frequency, spatial)
        gram = (spatial.T @ spatial) * (frequency.T @ frequency) + eps * np.eye(rank)
        temporal = [tensor.reshape(t_count, f_count * c_count) @ kr @ np.linalg.pinv(gram) for tensor in tensors]
        gram_acc = np.zeros((rank, rank), dtype=np.float64)
        numerator = np.zeros((c_count, rank), dtype=np.float64)
        for tensor, time_factor in zip(tensors, temporal):
            # [C,T,F] flattening uses compound index t*F+f.
            kr = _khatri_rao(time_factor, frequency)
            gram_acc += (time_factor.T @ time_factor) * (frequency.T @ frequency)
            numerator += tensor.transpose(2, 0, 1).reshape(c_count, t_count * f_count) @ kr
        spatial = np.maximum(numerator @ np.linalg.pinv(gram_acc + eps * np.eye(rank)), eps)
        gram_acc.fill(0.0)
        numerator = np.zeros((f_count, rank), dtype=np.float64)
        for tensor, time_factor in zip(tensors, temporal):
            # [F,T,C] flattening uses compound index t*C+c.
            kr = _khatri_rao(time_factor, spatial)
            gram_acc += (time_factor.T @ time_factor) * (spatial.T @ spatial)
            numerator += tensor.transpose(1, 0, 2).reshape(f_count, t_count * c_count) @ kr
        frequency = np.maximum(numerator @ np.linalg.pinv(gram_acc + eps * np.eye(rank)), eps)
        for component in range(rank):
            scale = max(float(np.linalg.norm(spatial[:, component])), eps)
            spatial[:, component] /= scale
            for time_factor in temporal:
                time_factor[:, component] *= scale
            scale = max(float(np.linalg.norm(frequency[:, component])), eps)
            frequency[:, component] /= scale
            for time_factor in temporal:
                time_factor[:, component] *= scale
    return {"spatial": spatial, "frequency": frequency, "temporal": temporal}


def _project_temporal(tensor: np.ndarray, spatial: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    t_count, f_count, c_count = tensor.shape
    kr = _khatri_rao(frequency, spatial)
    gram = (spatial.T @ spatial) * (frequency.T @ frequency)
    return tensor.reshape(t_count, f_count * c_count) @ kr @ np.linalg.pinv(gram + 1e-8 * np.eye(gram.shape[0]))


def _trca_filter(temporal: Sequence[np.ndarray]) -> np.ndarray:
    rank = temporal[0].shape[1]
    q = np.zeros((rank, rank), dtype=np.float64)
    s = np.zeros((rank, rank), dtype=np.float64)
    centered = [value - value.mean(axis=0, keepdims=True) for value in temporal]
    for left, x in enumerate(centered):
        q += x.T @ x
        for right, y in enumerate(centered):
            if left != right:
                s += x.T @ y
    values, vectors = np.linalg.eig(np.linalg.pinv(q + 1e-8 * np.eye(rank)) @ s)
    weights = np.real(vectors[:, int(np.argmax(np.real(values)))])
    norm = float(np.linalg.norm(weights))
    return weights / norm if norm > 1e-12 else weights


def _standardize(signal: np.ndarray) -> np.ndarray:
    return (np.asarray(signal) - float(np.mean(signal))) / max(float(np.std(signal)), 1e-12)


def _lin_shape_penalty(params: Sequence[float]) -> float:
    a1, a2, b1, b2, _ = [float(value) for value in params]
    checks = (
        (a1 / max(b1, 1e-12), 3.0, 7.0),
        (a2 / max(b2, 1e-12), 9.0, 18.0),
        (2.35 * np.sqrt(max(a1 - 1.0, 0.0)) / max(b1, 1e-12), 3.0, 6.0),
        (2.35 * np.sqrt(max(a2 - 1.0, 0.0)) / max(b2, 1e-12), 7.0, 11.0),
    )
    total = 0.0
    for value, low, high in checks:
        total += (low - value) ** 2 if value < low else (value - high) ** 2 if value > high else 0.0
    return total


def _fit_linear(drivers: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> np.ndarray:
    x = np.concatenate([value.reshape(-1, 1) for value in drivers])
    y = np.concatenate([value.reshape(-1, 1) for value in targets])
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _linear_predict(driver: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return (np.column_stack((driver.reshape(-1, 1), np.ones(len(driver)))) @ coefficients).reshape(-1)


def _fit_lin_hrf(
    components: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    mode: str,
    fs: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    def fit(params: np.ndarray) -> tuple[np.ndarray, float]:
        hrf = _lin_hrf(params, fs)
        drivers = [_convolve(_standardize(value), hrf, fs) for value in components]
        coefficients = _fit_linear(drivers, targets)
        predictions = [_linear_predict(driver, coefficients) for driver in drivers]
        mse = float(np.mean((np.concatenate(predictions) - np.concatenate(targets)) ** 2))
        return coefficients, mse

    if mode == "canonical":
        coefficients, _ = fit(_canonical_lin_params())
        return _canonical_lin_params(), coefficients
    if mode != "optimized":
        raise ValueError(mode)
    bounds = [(2.0, 10.0), (6.0, 25.0), (0.5, 2.0), (0.05, 1.5), (0.2, 15.0)]

    def objective(params: np.ndarray) -> float:
        _, mse = fit(params)
        return mse + 0.05 * _lin_shape_penalty(params)

    result = minimize(
        objective,
        _canonical_lin_params(),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-9},
    )
    params = np.asarray(result.x if result.success else _canonical_lin_params(), dtype=np.float64)
    coefficients, _ = fit(params)
    return params, coefficients


def _lin_predictions(
    train_trials: Sequence[Trial],
    test_trials: Sequence[Trial],
    train_tensors: np.ndarray,
    test_tensors: np.ndarray,
    selected: np.ndarray,
    selected_names: tuple[str, ...],
    *,
    condition_id: str,
    validation: str,
    heldout_trial: int,
    config: Mapping[str, Any],
    seed: int,
) -> list[Prediction]:
    cp = _fit_shared_cp(
        train_tensors,
        rank=int(config["cp_rank"]),
        iterations=int(config["cp_iterations"]),
        seed=seed,
    )
    weights = _trca_filter(cp["temporal"])
    train_components = [_standardize(value @ weights) for value in cp["temporal"]]
    if validation == "in_sample_upper_bound" and len(test_tensors) == len(train_tensors):
        # A true fit-quality upper bound must evaluate the temporal factors that
        # were optimized on these same trials.  Re-projecting them through the
        # final ALS factors is a separate reconstruction check and can be worse
        # than the fitted intercept-only reference.
        test_components = [value.copy() for value in train_components]
    else:
        test_components = [
            _standardize(_project_temporal(tensor, cp["spatial"], cp["frequency"]) @ weights)
            for tensor in test_tensors
        ]
    train_targets = _targets(train_trials, selected)
    test_targets = _targets(test_trials, selected)
    predictions: list[Prediction] = []
    for mode in config["hrf_modes"]:
        params, coefficients = _fit_lin_hrf(train_components, train_targets, str(mode))
        hrf = _lin_hrf(params, 10.0)
        for offset, (trial, component, truth) in enumerate(zip(test_trials, test_components, test_targets)):
            driver = _convolve(component, hrf, 10.0)
            estimate = _linear_predict(driver, coefficients)
            predictions.append(Prediction(
                condition_id=condition_id,
                dataset_id=trial.dataset_id,
                subject=trial.subject,
                validation=validation,
                heldout_trial=heldout_trial if len(test_trials) == 1 else offset,
                model="lin_trtd",
                hrf_mode=str(mode),
                truth=truth,
                estimate=estimate,
                selected_channels=selected_names,
                driver=driver,
            ))
    return predictions


def _baseline_predictions(
    train_trials: Sequence[Trial],
    test_trial: Trial,
    selected: np.ndarray,
    selected_names: tuple[str, ...],
    *,
    condition_id: str,
    heldout_trial: int,
) -> list[Prediction]:
    train_targets = np.stack(_targets(train_trials, selected))
    truth = _targets([test_trial], selected)[0]
    trial_mean = train_targets.mean(axis=0)
    persistence = np.empty_like(truth)
    persistence[0] = float(train_targets[:, 0].mean())
    persistence[1:] = truth[:-1]
    return [
        Prediction(condition_id, test_trial.dataset_id, test_trial.subject, "leave_one_trial", heldout_trial,
                   "trial_mean", "none", truth, trial_mean, selected_names),
        Prediction(condition_id, test_trial.dataset_id, test_trial.subject, "leave_one_trial", heldout_trial,
                   "self_persistence", "none", truth, persistence, selected_names),
    ]


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _waveform_metrics(truth: np.ndarray, estimate: np.ndarray, baseline_n: int) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    error = estimate - truth
    denominator = max(float(np.sum((truth - truth.mean()) ** 2)), 1e-12)
    truth_std = max(float(np.std(truth)), 1e-12)
    amplitude_ratio = float(np.std(estimate) / truth_std)
    truth_range = max(float(np.ptp(truth)), 1e-12)
    time = np.linspace(-1.0, 1.0, len(truth) - baseline_n)
    slope_denominator = max(float(np.sum(time**2)), 1e-12)
    truth_post = truth[baseline_n:]
    estimate_post = estimate[baseline_n:]
    truth_slope = float(np.sum((truth_post - truth_post.mean()) * time) / slope_denominator)
    estimate_slope = float(np.sum((estimate_post - estimate_post.mean()) * time) / slope_denominator)
    truth_diff = np.diff(truth_post)
    estimate_diff = np.diff(estimate_post)
    direction = float(np.mean(np.sign(truth_diff) == np.sign(estimate_diff))) if len(truth_diff) else float("nan")
    affine_design = np.column_stack((estimate, np.ones(len(estimate))))
    affine = affine_design @ np.linalg.lstsq(affine_design, truth, rcond=None)[0]
    return {
        "mse": float(np.mean(error**2)),
        "r2": 1.0 - float(np.sum(error**2)) / denominator,
        "pcc": _safe_corr(truth, estimate),
        "amplitude_ratio": amplitude_ratio,
        "variance_ratio": amplitude_ratio**2,
        "peak_to_peak_ratio": float(np.ptp(estimate) / truth_range),
        "mean_bias": float(np.mean(error)),
        "baseline_bias": float(np.mean(error[:baseline_n])),
        "poststimulus_bias": float(np.mean(error[baseline_n:])),
        "trend_direction_agreement": direction,
        "poststimulus_slope_truth": truth_slope,
        "poststimulus_slope_estimate": estimate_slope,
        "poststimulus_slope_sign_agreement": float(np.sign(truth_slope) == np.sign(estimate_slope)),
        "affine_oracle_r2": 1.0 - float(np.sum((affine - truth) ** 2)) / denominator,
    }


def _run_subject_models(
    condition_id: str,
    subject_index: int,
    subject: str,
    trials_value: Sequence[Trial],
    data_cfg: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> list[Prediction]:
    predictions: list[Prediction] = []
    with threadpool_limits(limits=1):
        trials = list(trials_value)
        tensors = np.stack([_eeg_tensor(trial.eeg, 200.0, trial.fnirs.shape[0]) for trial in trials])
        for heldout in range(len(trials)):
            train_indices = [index for index in range(len(trials)) if index != heldout]
            train_trials = [trials[index] for index in train_indices]
            test_trials = [trials[heldout]]
            selected, names, _ = _select_active_hbo(
                train_trials,
                baseline_duration_s=float(data_cfg["baseline_duration_s"]),
                task_duration_s=float(data_cfg["task_duration_s"]),
                count=int(analysis["fnirs_active_hbo_channels"]),
            )
            fold_seed = int(analysis["seed"]) + subject_index * 1000 + heldout
            predictions.extend(_baseline_predictions(
                train_trials, test_trials[0], selected, names,
                condition_id=condition_id, heldout_trial=heldout,
            ))
            predictions.extend(_croce_predictions(
                train_trials, test_trials, selected, names,
                condition_id=condition_id, validation="leave_one_trial", heldout_trial=heldout,
                config=analysis["croce2017"], seed=fold_seed,
            ))
            predictions.extend(_lin_predictions(
                train_trials, test_trials, tensors[train_indices], tensors[[heldout]], selected, names,
                condition_id=condition_id, validation="leave_one_trial", heldout_trial=heldout,
                config=analysis["lin2024"], seed=fold_seed,
            ))

        selected, names, _ = _select_active_hbo(
            trials,
            baseline_duration_s=float(data_cfg["baseline_duration_s"]),
            task_duration_s=float(data_cfg["task_duration_s"]),
            count=int(analysis["fnirs_active_hbo_channels"]),
        )
        upper_seed = int(analysis["seed"]) + subject_index * 1000 + 999
        predictions.extend(_croce_predictions(
            trials, trials, selected, names,
            condition_id=condition_id, validation="in_sample_upper_bound", heldout_trial=-1,
            config=analysis["croce2017"], seed=upper_seed,
        ))
        predictions.extend(_lin_predictions(
            trials, trials, tensors, tensors, selected, names,
            condition_id=condition_id, validation="in_sample_upper_bound", heldout_trial=-1,
            config=analysis["lin2024"], seed=upper_seed,
        ))
    return predictions


def _run_models(grouped: Mapping[str, Mapping[str, Sequence[Trial]]], config: Mapping[str, Any]) -> list[Prediction]:
    data_cfg = config["data"]
    analysis = config["analysis"]
    tasks = []
    for condition_id, subjects in grouped.items():
        for subject_index, (subject, trials) in enumerate(sorted(subjects.items())):
            tasks.append((condition_id, subject_index, subject, list(trials), data_cfg, analysis))
    workers = max(1, min(int(analysis.get("workers", 1)), len(tasks)))
    if workers == 1:
        return [_prediction for task in tasks for _prediction in _run_subject_models(*task)]
    predictions: list[Prediction] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_subject_models, *task): (task[0], task[2]) for task in tasks}
        for future in as_completed(futures):
            condition_id, subject = futures[future]
            values = future.result()
            print(f"completed {condition_id}:{subject} predictions={len(values)}", flush=True)
            predictions.extend(values)
    return predictions


def _metric_tables(
    predictions: Sequence[Prediction],
    *,
    baseline_n: int,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows = []
    for prediction in predictions:
        fold_rows.append({
            "condition_id": prediction.condition_id,
            "dataset_id": prediction.dataset_id,
            "subject": prediction.subject,
            "validation": prediction.validation,
            "heldout_trial": prediction.heldout_trial,
            "model": prediction.model,
            "hrf_mode": prediction.hrf_mode,
            "selected_channels": "|".join(prediction.selected_channels),
            **_waveform_metrics(prediction.truth, prediction.estimate, baseline_n),
        })

    groups: dict[tuple[str, ...], list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        key = (
            prediction.condition_id,
            prediction.dataset_id,
            prediction.subject,
            prediction.validation,
            prediction.model,
            prediction.hrf_mode,
        )
        groups[key].append(prediction)
    subject_rows = []
    for key, values in sorted(groups.items()):
        truth = np.concatenate([value.truth for value in values])
        estimate = np.concatenate([value.estimate for value in values])
        subject_rows.append({
            "condition_id": key[0],
            "dataset_id": key[1],
            "subject": key[2],
            "validation": key[3],
            "model": key[4],
            "hrf_mode": key[5],
            "trials": len(values),
            **_waveform_metrics(truth, estimate, baseline_n),
        })

    metrics = (
        "mse", "r2", "pcc", "amplitude_ratio", "variance_ratio", "peak_to_peak_ratio",
        "mean_bias", "baseline_bias", "poststimulus_bias", "trend_direction_agreement",
        "poststimulus_slope_sign_agreement", "affine_oracle_r2",
    )
    summary_groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        summary_groups[(row["condition_id"], row["dataset_id"], row["validation"], row["model"], row["hrf_mode"])].append(row)
    rng = np.random.default_rng(seed)
    summary_rows = []
    for key, values in sorted(summary_groups.items()):
        row: dict[str, Any] = {
            "condition_id": key[0],
            "dataset_id": key[1],
            "validation": key[2],
            "model": key[3],
            "hrf_mode": key[4],
            "subjects": len(values),
            "trials": int(sum(int(value["trials"]) for value in values)),
        }
        for metric in metrics:
            observed = np.asarray([float(value[metric]) for value in values if np.isfinite(float(value[metric]))])
            if not len(observed):
                continue
            row[metric] = float(observed.mean())
            row[f"{metric}_median"] = float(np.median(observed))
            draws = np.empty(bootstrap_iterations, dtype=np.float64)
            for iteration in range(bootstrap_iterations):
                draws[iteration] = rng.choice(observed, size=len(observed), replace=True).mean()
            row[f"{metric}_ci_low"] = float(np.quantile(draws, 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(draws, 0.975))
        summary_rows.append(row)
    return fold_rows, subject_rows, summary_rows


def _artifact_contrast(subject_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["condition_id"]), str(row["subject"]), str(row["validation"]), str(row["model"]), str(row["hrf_mode"])): row
        for row in subject_rows
    }
    metrics = ("r2", "pcc", "amplitude_ratio", "variance_ratio", "baseline_bias", "affine_oracle_r2")
    output = []
    for key, clean in sorted(lookup.items()):
        condition, subject, validation, model, hrf_mode = key
        if condition != "single_trial_clean_v3":
            continue
        raw = lookup.get(("single_trial_raw", subject, validation, model, hrf_mode))
        if raw is None:
            continue
        row: dict[str, Any] = {
            "subject": subject,
            "validation": validation,
            "model": model,
            "hrf_mode": hrf_mode,
        }
        for metric in metrics:
            row[f"clean_{metric}"] = float(clean[metric])
            row[f"raw_{metric}"] = float(raw[metric])
            row[f"delta_{metric}"] = float(clean[metric]) - float(raw[metric])
        output.append(row)
    return output


def _artifact_contrast_summary(
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in contrast_rows:
        groups[(str(row["validation"]), str(row["model"]), str(row["hrf_mode"]))].append(row)
    rng = np.random.default_rng(seed)
    output = []
    metrics = ("delta_r2", "delta_pcc", "delta_amplitude_ratio", "delta_variance_ratio")
    for key, values in sorted(groups.items()):
        summary: dict[str, Any] = {
            "validation": key[0],
            "model": key[1],
            "hrf_mode": key[2],
            "subjects": len(values),
        }
        for metric in metrics:
            observed = np.asarray([float(value[metric]) for value in values], dtype=np.float64)
            draws = np.empty(bootstrap_iterations, dtype=np.float64)
            for iteration in range(bootstrap_iterations):
                draws[iteration] = rng.choice(observed, size=len(observed), replace=True).mean()
            summary[metric] = float(observed.mean())
            summary[f"{metric}_ci_low"] = float(np.quantile(draws, 0.025))
            summary[f"{metric}_ci_high"] = float(np.quantile(draws, 0.975))
            summary[f"{metric}_positive_subjects"] = int(np.count_nonzero(observed > 0.0))
        output.append(summary)
    return output


def _save_trajectories(path: Path, predictions: Sequence[Prediction]) -> None:
    rows = []
    for prediction in predictions:
        for index, (truth, estimate) in enumerate(zip(prediction.truth, prediction.estimate)):
            rows.append({
                "condition_id": prediction.condition_id,
                "subject": prediction.subject,
                "validation": prediction.validation,
                "heldout_trial": prediction.heldout_trial,
                "model": prediction.model,
                "hrf_mode": prediction.hrf_mode,
                "time_s": index / 10.0 - 5.0,
                "truth": float(truth),
                "estimate": float(estimate),
                "driver": float(prediction.driver[index]) if prediction.driver is not None else "",
            })
    _write_csv(path, rows)


def _plot_summary(summary: Sequence[Mapping[str, Any]], run_dir: Path) -> list[str]:
    selected = [
        row for row in summary
        if row["validation"] == "leave_one_trial"
        and (
            row["model"] in {"croce_joint", "croce_eeg_only", "trial_mean", "self_persistence"}
            or (row["model"] == "lin_trtd" and row["hrf_mode"] == "optimized")
        )
    ]
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    condition_order = ["single_trial_raw", "single_trial_clean_v3", "simultaneous_unified"]
    model_order = ["croce_joint", "croce_eeg_only", "lin_trtd", "trial_mean", "self_persistence"]
    lookup = {(row["condition_id"], row["model"]): row for row in selected}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = [
        ("r2", "Cross-validated R²"),
        ("pcc", "Waveform correlation"),
        ("amplitude_ratio", "Amplitude (SD) ratio"),
        ("variance_ratio", "Variance ratio"),
    ]
    x = np.arange(len(condition_order), dtype=float)
    width = 0.15
    for axis, (metric, title) in zip(axes.flat, metrics):
        for model_index, model in enumerate(model_order):
            values = [float(lookup.get((condition, model), {}).get(metric, np.nan)) for condition in condition_order]
            axis.bar(x + (model_index - 2) * width, values, width=width, label=model)
        axis.axhline(0.0, color="black", linewidth=0.8)
        if metric in {"amplitude_ratio", "variance_ratio"}:
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xticks(x, [value.replace("_", "\n") for value in condition_order])
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Unified-loader shared neural driver retest")
    fig.tight_layout()
    output = figures / "model_metric_summary.svg"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return [str(output.relative_to(run_dir))]


def _plot_representative(predictions: Sequence[Prediction], run_dir: Path, subject: str) -> list[str]:
    selected = [
        prediction for prediction in predictions
        if prediction.subject == subject
        and prediction.validation == "leave_one_trial"
        and prediction.heldout_trial == 0
        and (
            prediction.model == "croce_joint"
            or (prediction.model == "lin_trtd" and prediction.hrf_mode == "optimized")
        )
        and prediction.condition_id in {"single_trial_raw", "single_trial_clean_v3"}
    ]
    if not selected:
        return []
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    time = np.arange(len(selected[0].truth)) / 10.0 - 5.0
    positions = {
        ("single_trial_raw", "croce_joint"): axes[0, 0],
        ("single_trial_clean_v3", "croce_joint"): axes[0, 1],
        ("single_trial_raw", "lin_trtd"): axes[1, 0],
        ("single_trial_clean_v3", "lin_trtd"): axes[1, 1],
    }
    for prediction in selected:
        axis = positions[(prediction.condition_id, prediction.model)]
        axis.plot(time, prediction.truth, color="#222222", linewidth=2, label="observed HbO")
        axis.plot(time, prediction.estimate, color="#d55e00", linewidth=1.8, label="recovered HbO")
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_title(f"{prediction.condition_id} / {prediction.model}")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("canonical robust SD")
    for axis in axes[-1]:
        axis.set_xlabel("event-relative time (s)")
    fig.suptitle(f"Representative held-out trial: {subject}")
    fig.tight_layout()
    output = figures / "representative_raw_vs_clean_overlay.svg"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return [str(output.relative_to(run_dir))]


def _summary_markdown(
    summary: Sequence[Mapping[str, Any]],
    contrast: Sequence[Mapping[str, Any]],
    contrast_summary: Sequence[Mapping[str, Any]],
) -> str:
    focus = [
        row for row in summary
        if row["validation"] == "leave_one_trial"
        and (
            row["model"] in {"croce_joint", "croce_eeg_only", "trial_mean", "self_persistence"}
            or (row["model"] == "lin_trtd" and row["hrf_mode"] == "optimized")
        )
    ]
    lines = [
        "# Unified-loader Croce 2017 / Lin 2024 shared-driver retest",
        "",
        "This is an exploratory validation diagnostic, not a direct paper reproduction and not an E0 gate decision.",
        "Single-Trial raw and admitted v3-clean EEG use identical fNIRS trials; active HbO channels are selected inside each training fold.",
        "",
        "## Subject-aggregated leave-one-trial results",
        "",
        "| Condition | Model | HRF | Subjects | R2 | PCC | SD ratio | Variance ratio | Baseline bias | Affine-oracle R2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in focus:
        lines.append(
            f"| {row['condition_id']} | {row['model']} | {row['hrf_mode']} | {row['subjects']} | "
            f"{row.get('r2', float('nan')):.4f} | {row.get('pcc', float('nan')):.4f} | "
            f"{row.get('amplitude_ratio', float('nan')):.4f} | {row.get('variance_ratio', float('nan')):.4f} | "
            f"{row.get('baseline_bias', float('nan')):.4f} | {row.get('affine_oracle_r2', float('nan')):.4f} |"
        )
    paired = [
        row for row in contrast
        if row["validation"] == "leave_one_trial"
        and (row["model"] == "croce_joint" or (row["model"] == "lin_trtd" and row["hrf_mode"] == "optimized"))
    ]
    lines.extend([
        "",
        "## Paired Single-Trial clean-minus-raw contrast",
        "",
        "| Subject | Model | delta R2 | delta PCC | delta SD ratio | delta variance ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in paired:
        lines.append(
            f"| {row['subject']} | {row['model']} | {row['delta_r2']:.4f} | {row['delta_pcc']:.4f} | "
            f"{row['delta_amplitude_ratio']:.4f} | {row['delta_variance_ratio']:.4f} |"
        )
    paired_summary = [
        row for row in contrast_summary
        if row["validation"] == "leave_one_trial"
        and (row["model"] == "croce_joint" or (row["model"] == "lin_trtd" and row["hrf_mode"] == "optimized"))
    ]
    lines.extend([
        "",
        "| Model | Mean delta R2 [95% subject bootstrap] | Mean delta PCC [95% subject bootstrap] | Mean delta SD ratio |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in paired_summary:
        lines.append(
            f"| {row['model']} | {row['delta_r2']:.4f} "
            f"[{row['delta_r2_ci_low']:.4f}, {row['delta_r2_ci_high']:.4f}] | "
            f"{row['delta_pcc']:.4f} [{row['delta_pcc_ci_low']:.4f}, {row['delta_pcc_ci_high']:.4f}] | "
            f"{row['delta_amplitude_ratio']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- `croce_joint` uses both held-out EEG and held-out fNIRS in filtering; `croce_eeg_only` is the stricter cross-modal reconstruction control.",
        "- `lin_trtd` is a Lin-inspired CP/TRCA plus subject-specific double-gamma HRF implementation, not the original Tensorlab pipeline.",
        "- `affine_oracle_r2` is a held-out shape diagnostic fitted after prediction and is not deployable predictive performance.",
        "- `self_persistence` consumes the target modality's previous sample and is a private-history reference, not evidence for a shared driver.",
        "- Subject bootstrap intervals and complete fold/subject tables are in `summary_metrics.csv`, `subject_metrics.csv`, and `fold_metrics.csv`.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.smoke:
        config = json.loads(json.dumps(config))
        config["data"]["conditions"] = [config["data"]["conditions"][0]]
        config["data"]["conditions"][0]["subjects"] = [config["data"]["conditions"][0]["subjects"][0]]
        config["data"]["conditions"][0]["max_trials_per_subject"] = min(
            4, int(config["data"]["conditions"][0]["max_trials_per_subject"])
        )
        config["analysis"]["bootstrap_iterations"] = 100
        config["analysis"]["workers"] = 1
        config["analysis"]["lin2024"]["cp_iterations"] = 3
        config["analysis"]["croce2017"]["n_particles"] = 40
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity"
        / f"{stamp}_{config['experiment']['name']}{'_smoke' if args.smoke else ''}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    grouped, contracts = _load_trials(config)
    predictions = _run_models(grouped, config)
    baseline_n = int(round(float(config["data"]["baseline_duration_s"]) * 10.0))
    fold_rows, subject_rows, summary_rows = _metric_tables(
        predictions,
        baseline_n=baseline_n,
        bootstrap_iterations=int(config["analysis"]["bootstrap_iterations"]),
        seed=int(config["analysis"]["seed"]),
    )
    contrast = _artifact_contrast(subject_rows)
    contrast_summary = _artifact_contrast_summary(
        contrast,
        bootstrap_iterations=int(config["analysis"]["bootstrap_iterations"]),
        seed=int(config["analysis"]["seed"]),
    )
    _write_csv(run_dir / "fold_metrics.csv", fold_rows)
    _write_csv(run_dir / "subject_metrics.csv", subject_rows)
    _write_csv(run_dir / "summary_metrics.csv", summary_rows)
    _write_csv(run_dir / "artifact_branch_contrast.csv", contrast)
    _write_csv(run_dir / "artifact_branch_contrast_summary.csv", contrast_summary)
    _save_trajectories(run_dir / "trajectories.csv", predictions)
    figures = _plot_summary(summary_rows, run_dir)
    figures += _plot_representative(predictions, run_dir, str(config["analysis"]["representative_subject"]))
    (run_dir / "summary.md").write_text(
        _summary_markdown(summary_rows, contrast, contrast_summary), encoding="utf-8"
    )

    sources = [
        run_dir / "config.yaml",
        Path(__file__),
        REPO_ROOT / "src/data/unified_physiology.py",
        REPO_ROOT / "src/inference/neurovascular_smc.py",
        REPO_ROOT / str(config["data"]["cache_root"]) / "cache_manifest.json",
        REPO_ROOT / str(config["data"]["cache_root"]) / "event_index/event_manifest.json",
        REPO_ROOT / str(config["data"]["cache_root"]) / "channel_geometry/geometry_manifest.json",
        REPO_ROOT / str(config["data"]["cache_root"]) / "eeg_artifact_clean_v4/cache_manifest.json",
    ]
    trial_inventory = []
    for condition_id, subjects in grouped.items():
        for subject, trials in sorted(subjects.items()):
            trial_inventory.append({
                "condition_id": condition_id,
                "subject": subject,
                "trials": len(trials),
                "mean_eeg_artifact_fraction": float(np.mean([trial.eeg_artifact_fraction for trial in trials])),
                "event_indices": [trial.event_index for trial in trials],
            })
    manifest = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "formal_exploratory",
        "git": _git_payload(),
        "platform": {"python": platform.python_version(), "platform": platform.platform()},
        "input_hashes": [{"path": str(path), "sha256": _sha256(path)} for path in sources],
        "loader_contracts": contracts,
        "trial_inventory": trial_inventory,
        "prediction_count": len(predictions),
        "artifacts": [
            "config.yaml", "fold_metrics.csv", "subject_metrics.csv", "summary_metrics.csv",
            "artifact_branch_contrast.csv", "artifact_branch_contrast_summary.csv",
            "trajectories.csv", "summary.md", *figures,
        ],
        "claim_boundary": [
            "exploratory model-family comparison",
            "not a direct reproduction of Croce 2017 or Lin 2024",
            "not an E0 pass decision without the full protected protocol",
        ],
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/physiology_semantic_tokenizer/shared_neural_driver_unified_retest.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
