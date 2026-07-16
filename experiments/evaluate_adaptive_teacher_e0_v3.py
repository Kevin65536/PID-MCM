#!/usr/bin/env python3
"""Re-run the validation-only E0 contract for the adaptive shared-state teacher.

Protected subject identifiers are recorded in the preregistered split, but are
deliberately absent from ``data.conditions`` and are never loaded.  Passing
this script's validation layers can only make a separate protected-test run
eligible; it cannot itself pass the protected E0 gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.e0_v2_measurement_audit import run_measurement_audit
from experiments.evaluate_adaptive_shared_neural_ssm import run as run_base_model
from experiments.evaluate_physical_teacher_e0_v2 import (
    EEG_LOCAL_NAMES,
    FNIRS_LOCAL_NAMES,
    Evidence,
    _context_audit,
    _physical_audit,
    _vocabulary_audit,
    _write_csv,
    _write_json,
)
from src.inference.adaptive_neurovascular_ssm import (
    HemodynamicParameters,
    apply_adaptive_ssm,
    fit_adaptive_ssm,
    simulate_hemodynamics,
)


SCHEMA = "physiology_semantic_adaptive_teacher_e0_v3"
STATE_NAMES = ("vasodilation_s", "flow_delta", "hbo_state", "hbr_state", "shared_driver")
SYNTHETIC_NAMES = ("s", "delta_f", "delta_HbO", "delta_HbR", "r")
GAUGE_MODE = "train_fold_observation_aligned_chromophore_v1"


def _subject_number(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    if match is None:
        raise ValueError(f"subject identifier has no numeric suffix: {value}")
    return int(match.group(1))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patches(values: np.ndarray, patch_size: int = 20) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    usable = array.shape[0] // patch_size * patch_size
    if usable == 0:
        raise ValueError("trajectory is shorter than one patch")
    return array[:usable].reshape(usable // patch_size, patch_size, array.shape[1])


def _features(values: np.ndarray, spectral_bins: int = 8) -> np.ndarray:
    patch = _patches(values)
    time = np.linspace(-1.0, 1.0, patch.shape[1], dtype=np.float64)
    denominator = max(float(np.dot(time, time)), 1e-12)
    mean = np.mean(patch, axis=1)
    standard = np.std(patch, axis=1)
    slope = np.einsum("ptc,t->pc", patch, time) / denominator
    delta = patch[:, -1] - patch[:, 0]
    spectrum = np.log(np.maximum(np.abs(np.fft.rfft(patch, axis=1)), 1e-8))
    bins = min(int(spectral_bins), spectrum.shape[1] - 1)
    spectral = spectrum[:, 1 : bins + 1].reshape(len(patch), -1)
    return np.concatenate((mean, standard, slope, delta, spectral), axis=1)


def _state_targets(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patch = _patches(states)
    time = np.linspace(-1.0, 1.0, patch.shape[1], dtype=np.float64)
    denominator = max(float(np.dot(time, time)), 1e-12)
    means = np.mean(patch, axis=1)
    slopes = np.einsum("ptc,t->pc", patch, time) / denominator
    eeg = np.column_stack((means[:, 4], slopes[:, 4], means[:, 0], slopes[:, 0]))
    fnirs = np.column_stack((means[:, 1:4], slopes[:, 1:4]))
    return eeg, fnirs, means, slopes


def _state_target_variance(state_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    patch = _patches(state_std)
    time = np.linspace(-1.0, 1.0, patch.shape[1], dtype=np.float64)
    weights = time / max(float(np.dot(time, time)), 1e-12)
    mean_variance = np.mean(np.square(patch), axis=1)
    # A conservative bound avoids claiming independent posterior errors across
    # adjacent RTS time points when the smoother covariance is not exported.
    slope_variance = np.square(np.einsum("t,ptc->pc", np.abs(weights), patch))
    eeg = np.column_stack((
        mean_variance[:, 4], slope_variance[:, 4],
        mean_variance[:, 0], slope_variance[:, 0],
    ))
    fnirs = np.column_stack((mean_variance[:, 1:4], slope_variance[:, 1:4]))
    return eeg, fnirs


def _physical_patch_rows(
    *,
    subject: int,
    heldout_trial: int,
    eeg_observed: np.ndarray,
    eeg_clean: np.ndarray,
    fnirs_observed: np.ndarray,
    fnirs_clean: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    for modality, observed, clean in (
        ("eeg", eeg_observed, eeg_clean),
        ("fnirs", fnirs_observed, fnirs_clean),
    ):
        observed_patch = _patches(observed)
        clean_patch = _patches(clean)
        history = np.zeros_like(observed_patch)
        history[1:] = observed_patch[:-1]
        for patch_index in range(len(observed_patch)):
            rows.append({
                "subject": subject,
                "heldout_trial": heldout_trial,
                "patch": patch_index,
                "modality": modality,
                "mse_clean": float(np.mean(np.square(observed_patch[patch_index] - clean_patch[patch_index]))),
                "mse_zero": float(np.mean(np.square(observed_patch[patch_index]))),
                "mse_history": float(np.mean(np.square(observed_patch[patch_index] - history[patch_index]))),
                "clean_correction": "none; adaptive fold-specific observation model",
            })
    fnirs_observed_patch = _patches(fnirs_observed)
    fnirs_clean_patch = _patches(fnirs_clean)
    fnirs_history = np.zeros_like(fnirs_observed_patch)
    fnirs_history[1:] = fnirs_observed_patch[:-1]
    for channel, name in enumerate(("hbo", "hbr")):
        for patch_index in range(len(fnirs_observed_patch)):
            components.append({
                "subject": subject,
                "heldout_trial": heldout_trial,
                "patch": patch_index,
                "component": name,
                "mse_clean": float(np.mean(np.square(
                    fnirs_observed_patch[patch_index, :, channel]
                    - fnirs_clean_patch[patch_index, :, channel]
                ))),
                "mse_zero": float(np.mean(np.square(fnirs_observed_patch[patch_index, :, channel]))),
                "mse_history": float(np.mean(np.square(
                    fnirs_observed_patch[patch_index, :, channel]
                    - fnirs_history[patch_index, :, channel]
                ))),
            })
    return rows, components


def _collect_evidence(
    rows: Sequence[Mapping[str, str]],
    subjects: Sequence[str],
    *,
    overlay: bool,
    use_target_gauge: bool = True,
) -> tuple[Evidence, list[dict[str, Any]], list[dict[str, Any]]]:
    admitted = {str(value) for value in subjects}
    selected = [
        row for row in rows
        if row["subject"] in admitted
        and row["model"] == "adaptive_joint"
        and row["spatial_mode"] == "local"
    ]
    grouped: dict[tuple[str, int], list[Mapping[str, str]]] = {}
    for row in selected:
        grouped.setdefault((row["subject"], int(row["heldout_trial"])), []).append(row)
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in (
        "eeg_features", "fnirs_features", "eeg_target", "fnirs_target",
        "eeg_uncertainty", "fnirs_uncertainty", "subjects", "patch_index",
        "fnirs_history", "eeg_history", "fnirs_level", "fnirs_innovation",
    )}
    physical_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    gauge_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    overlay_payload: dict[str, Any] | None = None
    for (subject_name, heldout_trial), values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: float(row["time_s"]))
        subject = _subject_number(subject_name)
        eeg_observed = np.asarray([float(row["eeg_observation"]) for row in values])
        eeg_clean = np.asarray([float(row["eeg_reconstruction"]) for row in values])
        hbo_observed = np.asarray([float(row["hbo_truth"]) for row in values])
        hbo_clean = np.asarray([float(row["hbo_estimate"]) for row in values])
        hbr_observed = np.asarray([float(row["hbr_truth"]) for row in values])
        hbr_clean = np.asarray([float(row["hbr_estimate"]) for row in values])
        fnirs_observed = np.column_stack((hbo_observed, hbr_observed))
        fnirs_clean = np.column_stack((hbo_clean, hbr_clean))
        raw_states = np.column_stack([
            np.asarray([float(row[name]) for row in values]) for name in STATE_NAMES
        ])
        raw_state_std = np.column_stack([
            np.asarray([float(row[f"{name}_std"]) for row in values]) for name in STATE_NAMES
        ])
        target_columns = [f"target_{name}" for name in STATE_NAMES]
        target_std_columns = [f"target_{name}_std" for name in STATE_NAMES]
        has_target_gauge = all(name in values[0] for name in target_columns + target_std_columns)
        if use_target_gauge and not has_target_gauge:
            raise ValueError("base-model trajectories do not contain the registered target gauge")
        if use_target_gauge:
            states = np.column_stack([
                np.asarray([float(row[name]) for row in values]) for name in target_columns
            ])
            state_std = np.column_stack([
                np.asarray([float(row[name]) for row in values]) for name in target_std_columns
            ])
        else:
            states = raw_states
            state_std = raw_state_std
        eeg_target, fnirs_target, state_means, _ = _state_targets(states)
        eeg_uncertainty, fnirs_uncertainty = _state_target_variance(state_std)
        patch_count = len(eeg_target)
        arrays["eeg_features"].append(_features(eeg_observed))
        arrays["fnirs_features"].append(_features(fnirs_observed))
        arrays["eeg_target"].append(eeg_target)
        arrays["fnirs_target"].append(fnirs_target)
        arrays["eeg_uncertainty"].append(eeg_uncertainty)
        arrays["fnirs_uncertainty"].append(fnirs_uncertainty)
        arrays["subjects"].append(np.full(patch_count, subject, dtype=int))
        arrays["patch_index"].append(np.arange(patch_count, dtype=int))
        fnirs_state = state_means[:, 1:4]
        eeg_state = state_means[:, [4, 0]]
        for target_index in range(5, patch_count):
            arrays["fnirs_history"].append(fnirs_state[target_index - 5 : target_index].reshape(1, -1))
            arrays["eeg_history"].append(eeg_state[target_index - 5 : target_index].reshape(1, -1))
            arrays["fnirs_level"].append(fnirs_state[target_index : target_index + 1])
            arrays["fnirs_innovation"].append(
                (fnirs_state[target_index] - fnirs_state[target_index - 1]).reshape(1, -1)
            )
        local_rows, components = _physical_patch_rows(
            subject=subject,
            heldout_trial=heldout_trial,
            eeg_observed=eeg_observed,
            eeg_clean=eeg_clean,
            fnirs_observed=fnirs_observed,
            fnirs_clean=fnirs_clean,
        )
        physical_rows.extend(local_rows)
        component_rows.extend(components)
        if has_target_gauge:
            gauge_scales = np.asarray([
                float(values[0][f"gauge_{name}_scale"]) for name in STATE_NAMES
            ])
            gauge_offsets = np.asarray([
                float(values[0][f"gauge_{name}_offset"]) for name in STATE_NAMES
            ])
            gauge_rows.append({
                "subject": subject,
                "heldout_trial": heldout_trial,
                "mode": GAUGE_MODE,
                "hbo_scale": float(gauge_scales[2]),
                "hbr_scale": float(gauge_scales[3]),
                "hbo_offset": float(gauge_offsets[2]),
                "hbr_offset": float(gauge_offsets[3]),
                "hbo_scale_sign": int(np.sign(gauge_scales[2])),
                "hbr_scale_sign": int(np.sign(gauge_scales[3])),
                "reconstruction_max_abs_delta": float(max(
                    np.max(np.abs(states[:, 2] - hbo_clean)) if use_target_gauge else 0.0,
                    np.max(np.abs(states[:, 3] - hbr_clean)) if use_target_gauge else 0.0,
                    float(values[0].get("gauge_reconstruction_max_abs_delta", 0.0)),
                )),
                "finite_non_singular": bool(
                    np.all(np.isfinite(gauge_scales[[2, 3]]))
                    and np.all(np.abs(gauge_scales[[2, 3]]) > 1e-12)
                ),
            })
        mask_rows.append({
            "subject": subject,
            "heldout_trial": heldout_trial,
            "local_valid_patches": patch_count,
            "context_valid_patches": max(0, patch_count - 5),
            "total_patches": patch_count,
            "local_coverage": 1.0,
            "context_coverage": max(0, patch_count - 5) / max(patch_count, 1),
        })
        if overlay and overlay_payload is None:
            time_s = np.asarray([float(row["time_s"]) for row in values])
            overlay_payload = {
                "subject": subject,
                "heldout_trial": heldout_trial,
                "eeg_time_s": time_s.tolist(),
                "eeg_observed_envelope": np.abs(eeg_observed).tolist(),
                "eeg_clean_envelope": np.abs(eeg_clean).tolist(),
                "fnirs_time_s": time_s.tolist(),
                "fnirs_observed": fnirs_observed.tolist(),
                "fnirs_clean": fnirs_clean.tolist(),
                "state_mean": raw_states.tolist(),
                "target_state_mean": states.tolist(),
                "target_gauge_mode": GAUGE_MODE if use_target_gauge else "raw_latent_state",
            }
    if not grouped:
        raise ValueError(f"no adaptive joint/local rows for subjects: {sorted(admitted)}")
    packed = {
        key: np.concatenate(value, axis=0) if value else np.empty((0, 0), dtype=np.float64)
        for key, value in arrays.items()
    }
    return Evidence(
        eeg_features=packed["eeg_features"],
        fnirs_features=packed["fnirs_features"],
        eeg_target=packed["eeg_target"],
        fnirs_target=packed["fnirs_target"],
        eeg_uncertainty=packed["eeg_uncertainty"],
        fnirs_uncertainty=packed["fnirs_uncertainty"],
        subjects=packed["subjects"],
        patch_index=packed["patch_index"],
        fnirs_history=packed["fnirs_history"],
        eeg_history=packed["eeg_history"],
        fnirs_level=packed["fnirs_level"],
        fnirs_innovation=packed["fnirs_innovation"],
        physical_rows=physical_rows,
        mask_rows=mask_rows,
        overlay=overlay_payload,
    ), component_rows, gauge_rows


def _select_alpha_by_train_subject_cv(
    train_x: np.ndarray,
    train_y: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
    *,
    folds: int = 5,
) -> tuple[float, float]:
    """Select ridge regularization without reading validation-subject labels."""

    unique = np.unique(np.asarray(groups, dtype=int))
    fold_count = min(max(2, int(folds)), len(unique))
    partitions = [value for value in np.array_split(unique, fold_count) if len(value)]
    best: tuple[float, float] | None = None
    for alpha in alphas:
        error = 0.0
        baseline = 0.0
        for heldout in partitions:
            inner_val = np.isin(groups, heldout)
            inner_train = ~inner_val
            scaler = StandardScaler().fit(train_x[inner_train])
            model = Ridge(alpha=float(alpha)).fit(
                scaler.transform(train_x[inner_train]), train_y[inner_train]
            )
            prediction = model.predict(scaler.transform(train_x[inner_val]))
            error += float(np.sum(np.square(train_y[inner_val] - prediction)))
            baseline += float(np.sum(np.square(
                train_y[inner_val] - np.mean(train_y[inner_train])
            )))
        score = 1.0 - error / max(baseline, 1e-12)
        if best is None or score > best[0] or (score == best[0] and float(alpha) > best[1]):
            best = (score, float(alpha))
    if best is None:
        raise ValueError("at least one ridge alpha is required")
    return best[1], best[0]


def _coordinate_audit_train_cv(
    train: Evidence,
    val: Evidence,
    *,
    alphas: Sequence[float],
    permutations: int,
    quantile: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Patch-local audit with alpha frozen by training-subject group CV."""

    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    for modality, names in (("eeg", EEG_LOCAL_NAMES), ("fnirs", FNIRS_LOCAL_NAMES)):
        train_x = getattr(train, f"{modality}_features")
        val_x = getattr(val, f"{modality}_features")
        train_y = getattr(train, f"{modality}_target")
        val_y = getattr(val, f"{modality}_target")
        uncertainty = getattr(val, f"{modality}_uncertainty")
        for index, name in enumerate(names):
            alpha, inner_cv_r2 = _select_alpha_by_train_subject_cv(
                train_x, train_y[:, index], train.subjects, alphas,
            )
            scaler = StandardScaler().fit(train_x)
            train_scaled = scaler.transform(train_x)
            val_scaled = scaler.transform(val_x)
            model = Ridge(alpha=alpha).fit(train_scaled, train_y[:, index])
            prediction = model.predict(val_scaled)
            baseline = float(np.sum(np.square(val_y[:, index] - np.mean(train_y[:, index]))))
            r2 = 1.0 - float(np.sum(np.square(val_y[:, index] - prediction))) / max(baseline, 1e-12)
            null = []
            for _ in range(int(permutations)):
                permuted = rng.permutation(train_y[:, index])
                null_model = Ridge(alpha=alpha).fit(train_scaled, permuted)
                null_prediction = null_model.predict(val_scaled)
                null.append(
                    1.0 - float(np.sum(np.square(val_y[:, index] - null_prediction)))
                    / max(baseline, 1e-12)
                )
            threshold = float(np.quantile(null, quantile))
            error = prediction - val_y[:, index]
            sigma = np.sqrt(np.maximum(uncertainty[:, index], 1e-12))
            statistically_observable = bool(r2 > max(0.0, threshold))
            rows.append({
                "modality": modality,
                "coordinate": name,
                "alpha": alpha,
                "alpha_selection": "five_fold_train_subject_group_cv",
                "inner_cv_r2": inner_cv_r2,
                "validation_r2": r2,
                "permutation_q": threshold,
                "statistically_observable": statistically_observable,
                "admitted_local_target": statistically_observable,
                "posterior_interval_90_coverage": float(np.mean(np.abs(error) <= 1.645 * sigma)),
                "standardized_rmse": float(np.sqrt(np.mean(np.square(error / sigma)))),
            })
            keep = min(400, len(prediction))
            traces[f"{modality}:{name}"] = {
                "target": val_y[:keep, index].tolist(),
                "prediction": prediction[:keep].tolist(),
                "subject": val.subjects[:keep].tolist(),
            }
    return rows, traces


def _apply_local_target_contract(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    required = {
        modality: {str(value) for value in contract["required_local_coordinates"][modality]}
        for modality in ("eeg", "fnirs")
    }
    optional = {
        modality: {str(value) for value in contract.get("optional_local_coordinates", {}).get(modality, [])}
        for modality in ("eeg", "fnirs")
    }
    output = []
    for original in rows:
        row = dict(original)
        modality = str(row["modality"])
        coordinate = str(row["coordinate"])
        if coordinate in required[modality]:
            role = "local_required"
        elif coordinate in optional[modality]:
            role = "local_optional"
        else:
            role = "context_only"
        eligible = role != "context_only"
        row["target_role"] = role
        row["eligible_for_local_admission"] = eligible
        row["admitted_local_target"] = bool(row["statistically_observable"] and eligible)
        row["coordinate_space"] = (
            "observation_aligned_canonical_measurement"
            if modality == "fnirs" and coordinate.startswith(("delta_hbo", "delta_hb"))
            else "adaptive_latent_state"
        )
        output.append(row)
    return output


def _required_local_pass(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> bool:
    return all(
        any(
            row["modality"] == modality
            and row["coordinate"] == coordinate
            and bool(row["admitted_local_target"])
            for row in rows
        )
        for modality in ("eeg", "fnirs")
        for coordinate in contract["required_local_coordinates"][modality]
    )


def _gauge_gain_rows(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    old = {(row["modality"], row["coordinate"]): row for row in before}
    output = []
    for row in after:
        baseline = old[(row["modality"], row["coordinate"])]
        output.append({
            "modality": row["modality"],
            "coordinate": row["coordinate"],
            "target_role": row["target_role"],
            "r2_pre_gauge": float(baseline["validation_r2"]),
            "r2_post_gauge": float(row["validation_r2"]),
            "r2_gain": float(row["validation_r2"]) - float(baseline["validation_r2"]),
            "coverage_pre_gauge": float(baseline["posterior_interval_90_coverage"]),
            "coverage_post_gauge": float(row["posterior_interval_90_coverage"]),
            "standardized_rmse_pre_gauge": float(baseline["standardized_rmse"]),
            "standardized_rmse_post_gauge": float(row["standardized_rmse"]),
            "admitted_pre_gauge": bool(baseline["admitted_local_target"]),
            "admitted_post_gauge": bool(row["admitted_local_target"]),
        })
    return output


def _simulate_trial(
    seed: int,
    synthetic: Mapping[str, Any],
    *,
    fs_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = int(round(20.0 * fs_hz))
    burn = int(round(float(synthetic["burn_in_s"]) * fs_hz))
    phi = float(synthetic["phi"])
    q_driver = float(synthetic["q_driver"])
    driver = np.zeros(samples + burn, dtype=np.float64)
    for index in range(1, len(driver)):
        driver[index] = phi * driver[index - 1] + rng.normal(scale=np.sqrt(q_driver))
    states = simulate_hemodynamics(driver, HemodynamicParameters(), fs_hz)[burn:]
    truth_driver = driver[burn:]
    eeg = truth_driver + rng.normal(scale=float(synthetic["eeg_noise_sd"]), size=samples)
    hbo = float(synthetic["hbo_gain"]) * states[:, 2] + rng.normal(
        scale=float(synthetic["hbo_noise_sd"]), size=samples,
    )
    hbr = float(synthetic["hbr_gain"]) * states[:, 3] + rng.normal(
        scale=float(synthetic["hbr_noise_sd"]), size=samples,
    )
    truth = np.column_stack((states, truth_driver))
    return eeg, hbo, hbr, truth


def _synthetic_seed(payload: tuple[int, Mapping[str, Any], Mapping[str, Any]]) -> dict[str, Any]:
    seed, synthetic, ssm = payload
    fs_hz = float(ssm["fs_hz"])
    trial_count = int(synthetic["trials_per_seed"])
    heldout = int(synthetic["heldout_trial"])
    trials = [_simulate_trial(seed * 1000 + index, synthetic, fs_hz=fs_hz) for index in range(trial_count)]
    train_indices = [index for index in range(trial_count) if index != heldout]
    with threadpool_limits(limits=1):
        fit = fit_adaptive_ssm(
            [trials[index][0] for index in train_indices],
            [trials[index][1] for index in train_indices],
            [trials[index][2] for index in train_indices],
            fs_hz=fs_hz,
            prior_strength=float(ssm["prior_strength"]),
            max_iterations=int(ssm["max_iterations"]),
            q_scale_candidates=tuple(float(value) for value in ssm["q_scale_candidates"]),
            fnirs_noise_scale_candidates=tuple(float(value) for value in ssm["fnirs_noise_scale_candidates"]),
            balance_penalty=float(ssm["balance_penalty"]),
            baseline_samples=int(round(5.0 * fs_hz)),
            max_flow_perturbation=float(ssm["max_flow_perturbation"]),
        )
        train_estimates = []
        train_truth = []
        for index in train_indices:
            eeg, hbo, hbr, truth = trials[index]
            result = apply_adaptive_ssm(eeg, fit, hbo_observation=hbo, hbr_observation=hbr)
            train_estimates.append(result.states)
            train_truth.append(truth)
        estimate = np.concatenate(train_estimates, axis=0)
        truth = np.concatenate(train_truth, axis=0)
        scale = np.sum(estimate * truth, axis=0) / np.maximum(np.sum(np.square(estimate), axis=0), 1e-12)
        eeg, hbo, hbr, heldout_truth = trials[heldout]
        result = apply_adaptive_ssm(eeg, fit, hbo_observation=hbo, hbr_observation=hbr)
    aligned = result.states * scale
    aligned_std = result.state_std * np.abs(scale)
    return {
        "seed": seed,
        "error": aligned - heldout_truth,
        "variance": np.square(aligned_std),
        "alignment_scale": scale,
    }


def _synthetic_posterior_calibration(
    config: Mapping[str, Any],
    *,
    workers: int,
) -> dict[str, Any]:
    synthetic = config["e0_v3"]["synthetic_calibration"]
    calibration_seeds = [int(value) for value in synthetic["calibration_seeds"]]
    validation_seeds = [int(value) for value in synthetic["validation_seeds"]]
    payloads = [(seed, synthetic, config["analysis"]["ssm"]) for seed in calibration_seeds + validation_seeds]
    results: dict[int, dict[str, Any]] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(payloads))) as executor:
            futures = {executor.submit(_synthetic_seed, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                results[int(result["seed"])] = result
    else:
        for payload in payloads:
            result = _synthetic_seed(payload)
            results[int(result["seed"])] = result
    calibration_error = np.concatenate([results[seed]["error"] for seed in calibration_seeds], axis=0)
    calibration_variance = np.concatenate([results[seed]["variance"] for seed in calibration_seeds], axis=0)
    variance_scale = np.mean(np.square(calibration_error), axis=0) / np.maximum(
        np.mean(calibration_variance, axis=0), 1e-12,
    )
    variance_scale = np.clip(variance_scale, 1e-8, 1e8)
    points = int(synthetic["points_per_validation_seed"])
    validation_error = []
    validation_variance = []
    for seed in validation_seeds:
        rng = np.random.default_rng(seed + 99173)
        count = len(results[seed]["error"])
        indices = np.sort(rng.choice(count, size=min(points, count), replace=False))
        validation_error.append(results[seed]["error"][indices])
        validation_variance.append(results[seed]["variance"][indices])
    error = np.concatenate(validation_error, axis=0)
    variance = np.concatenate(validation_variance, axis=0)
    samples = len(error)
    tolerance = 1.96 * np.sqrt(0.9 * 0.1 / max(samples, 1))
    rows = []
    for index, name in enumerate(SYNTHETIC_NAMES):
        standard = np.sqrt(np.maximum(variance[:, index], 1e-16))
        scaled_standard = standard * np.sqrt(variance_scale[index])
        unscaled_coverage = float(np.mean(np.abs(error[:, index]) <= 1.645 * standard))
        scaled_coverage = float(np.mean(np.abs(error[:, index]) <= 1.645 * scaled_standard))
        coverage_error = abs(scaled_coverage - 0.9)
        rows.append({
            "coordinate": name,
            "variance_scale_from_calibration_seeds": float(variance_scale[index]),
            "unscaled_90_coverage": unscaled_coverage,
            "scaled_90_coverage": scaled_coverage,
            "scaled_standardized_rmse": float(np.sqrt(np.mean(np.square(
                error[:, index] / np.maximum(scaled_standard, 1e-8)
            )))),
            "coverage_abs_error": coverage_error,
            "coverage_tolerance_95": float(tolerance),
            "calibrated_coverage_pass": bool(coverage_error <= tolerance),
        })
    unscaled_error = float(np.mean([abs(row["unscaled_90_coverage"] - 0.9) for row in rows]))
    scaled_error = float(np.mean([abs(row["scaled_90_coverage"] - 0.9) for row in rows]))
    return {
        "schema": f"{SCHEMA}_synthetic_posterior_calibration",
        "calibration_seeds": calibration_seeds,
        "validation_seeds": validation_seeds,
        "alignment": "zero-intercept state gauge learned from each seed's nine training trials",
        "rows": rows,
        "mean_abs_coverage_error_unscaled": unscaled_error,
        "mean_abs_coverage_error_scaled": scaled_error,
        "calibration_improves_heldout_coverage": bool(scaled_error < unscaled_error),
        "evaluation_samples_per_coordinate": samples,
        "posterior_calibration_pass": bool(all(row["calibrated_coverage_pass"] for row in rows)),
    }


def _failed_vocabulary_payload(modality: str, codebook_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return ({
        "modality": modality,
        "codebook_size": codebook_size,
        "validation_global_r2": float("nan"),
        "random_reference_q": float("nan"),
        "above_random_reference": False,
        "active_codes": 0,
        "perplexity": 0.0,
        "coordinate_r2": [],
        "target_coordinates": [],
        "failure_reason": "no locally observable coordinate admitted",
    }, {
        "pca": [[0.0, 0.0]],
        "codes": [0],
        "occupancy": [0] * codebook_size,
        "coordinate_r2": [],
        "target_coordinates": [],
    })


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    validation = summary["validation"]
    physical = {row["modality"]: row for row in summary["physical_observation"]}
    posterior = {row["coordinate"]: row for row in summary["posterior_calibration"]["rows"]}
    gauge = summary["target_gauge"]
    fnirs_gains = [
        row for row in gauge["coordinate_gains"]
        if row["modality"] == "fnirs" and row["target_role"] == "local_required"
    ]
    decision = "支持进入受保护测试" if validation["machine_validation_pass"] else "不支持进入受保护测试"
    lines = [
        "# Adaptive shared-state E0-v3 validation",
        "",
        "_Validation-only status report · 2026-07-16 · protected subjects remain closed_",
        "",
        "---",
        "",
        "## 📋 结论",
        "",
        f"**{decision}**。24–29 号受保护测试对象未打开，当前结果不能形成 G0 通过结论。",
        "",
        "本轮使用可变 Croce/Balloon 参数的五状态固定区间平滑器，在每个对象内部做留一 trial 拟合。"
        "被评估 trial 不参与 fNIRS 锚点选择、六邻近 EEG 通道选择、EEG 投影或生理参数拟合。",
        "",
        "## 📊 分层判据",
        "",
        "| E0 validation layer | Pass |",
        "| --- | --- |",
    ]
    labels = (
        ("measurement_contract_pass", "Measurement contract"),
        ("target_gauge_contract_pass", "Target gauge invariance"),
        ("local_target_observability_pass", "Local target observability"),
        ("finite_vocabulary_transmissibility_pass", "K=128 vocabulary transmissibility"),
        ("continuous_coupling_upper_bound_pass", "Continuous coupling upper bound"),
        ("physical_observation_pass", "Physical observation reconstruction"),
        ("posterior_uncertainty_calibration_pass", "Synthetic posterior calibration"),
        ("machine_validation_pass", "Machine validation conjunction"),
    )
    for key, label in labels:
        lines.append(f"| {label} | {validation[key]} |")
    lines.append(f"| Visual review | {validation['visual_review']} |")
    lines.extend([
        "",
        "## 🔍 主要失败证据",
        "",
        "| Evidence | Result | Gate implication |",
        "| --- | ---: | --- |",
        f"| Gauge reconstruction max abs delta | {float(gauge['max_reconstruction_abs_delta']):.3e} | "
        f"{'Pass' if validation['target_gauge_contract_pass'] else 'Fail'} |",
        f"| Required local targets before / after gauge | "
        f"{gauge['required_local_pass_pre_gauge']} / {gauge['required_local_pass_post_gauge']} | "
        f"{'Pass' if validation['local_target_observability_pass'] else 'Fail'} |",
        f"| EEG physical gain | {float(physical['eeg']['mean_gain']):.4f} | "
        f"{'Pass' if float(physical['eeg']['mean_gain']) > 0 else 'Fail'} |",
        f"| fNIRS physical gain | {float(physical['fnirs']['mean_gain']):.4f} | "
        f"{'Pass' if float(physical['fnirs']['mean_gain']) > 0 else 'Fail'} |",
        f"| fNIRS positive-subject fraction | {float(physical['fnirs']['positive_subject_fraction']):.2f} | Diagnostic |",
        f"| Synthetic HbR 90% coverage | {float(posterior['delta_HbR']['scaled_90_coverage']):.3f} | "
        f"{'Pass' if posterior['delta_HbR']['calibrated_coverage_pass'] else 'Fail'} |",
        f"| Synthetic coverage tolerance | ±{float(posterior['delta_HbR']['coverage_tolerance_95']):.4f} | Frozen criterion |",
        "",
        "Gauge 校正把 HbO/HbR posterior means 映射到训练折 observation adapter 定义的 canonical "
        "measurement space；它不改变物理重建，也不把 `delta_f` 升格为局部目标。",
        "",
        "| Required fNIRS coordinate | R² pre | R² post | Gain |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in fnirs_gains:
        lines.append(
            f"| {row['coordinate']} | {float(row['r2_pre_gauge']):.4f} | "
            f"{float(row['r2_post_gauge']):.4f} | {float(row['r2_gain']):+.4f} |"
        )
    lines.extend([
        "",
        "## 🔐 解释边界",
        "",
        "- `adaptive_joint` 是同时使用 held-out EEG、HbO 与 HbR 的妥协状态，不是 EEG-only 的跨模态预测。",
        "- E0 各层为合取关系；任何一层失败都不能由其他层的较好结果抵消。",
        "- 本验证若通过，也只允许另行开启冻结的 24–29 号受保护测试；不会直接形成 G0 通过结论。",
        "- 参数可变提高模型容量，但不消除状态尺度规约、可识别性和 posterior calibration 的要求。",
        "- Gauge 校正后的 HbO/HbR local targets 是 observation-aligned auxiliary targets，不是未经限定的物理隐状态。",
        "- 本轮是在查看旧 validation 结果后进行的方法修订，属于 recalibration evidence；即使 validation 通过也必须另开冻结 protected test。",
        "",
        "## 🔗 核心产物",
        "",
        "- `summary.json`：机器判据与完整指标",
        "- `physical_observation_checks.csv`：EEG/fNIRS 物理观测门",
        "- `posterior_calibration.csv`：冻结合成种子上的后验覆盖率",
        "- `local_target_observability.csv`：局部状态可观测性与置换阈值",
        "- `gauge_target_gain.csv`：同一 base-model posterior 的校正前后增益",
        "- `gauge_alignment.csv`：逐 fold gauge 参数与重建不变性",
        "- `visual_review.yaml`：图像复核与受保护测试决定",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    e0 = config["e0_v3"]
    seed = int(config["analysis"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    run_dir = Path(args.output_dir).resolve() if args.output_dir else (
        REPO_ROOT / "experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity"
        / f"{datetime.now():%Y%m%d_%H%M%S}_adaptive_teacher_e0_v3_validation"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "figure_data").mkdir()
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    protocol = {
        "schema": SCHEMA,
        "created_before_model_evaluation": True,
        "created_before_protected_test_access": True,
        "protected_test_default": "closed",
        "protected_test_subjects": e0["split"]["protected_test_subjects"],
        "measurement_gate": "finite auditable transforms plus exact crop-position invariance",
        "target_gauge_gate": "finite non-singular train-fold gauge with reconstruction invariance <= 1e-8",
        "target_gauge_mode": GAUGE_MODE,
        "local_target_gate": (
            "all preregistered required coordinates exceed zero and the label-permutation q95; "
            "ridge alpha selected only by training-subject group CV"
        ),
        "required_local_coordinates": e0["target_gauge"]["required_local_coordinates"],
        "vocabulary_gate": "K=128 target reconstruction exceeds random-centroid q95 for both modalities",
        "coupling_gate": "conditional log-determinant gain exceeds zero and shuffled-EEG q95",
        "physical_observation_gate": "both EEG and fNIRS clean reconstructions improve the selected zero/history baseline",
        "uncertainty_gate": "all five synthetic coordinates have held-out 90% coverage inside the binomial 95% band",
        "visual_gate": "registered figures reviewed before a separate protected test may open",
        "validation_conjunction": "all machine layers must pass; no compensatory averaging",
        "null_quantile": float(e0["null_quantile"]),
    }
    (run_dir / "decision_protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "metric_registry.json", {
        "schema": SCHEMA,
        "primary": [
            "local_target_validation_r2", "physical_observation_gain",
            "continuous_conditional_information_nats", "synthetic_90_interval_coverage",
        ],
        "gate": [
            "measurement_contract", "target_gauge_contract", "finite_vocabulary_transmissibility",
            "posterior_uncertainty_calibration", "visual_review",
        ],
        "diagnostic": [
            "student_error_over_teacher_sd", "component_physical_gain", "mask_coverage",
            "adaptive_parameter_boundary_fraction", "eeg_only_cross_modal_reconstruction",
            "pre_to_post_gauge_target_r2", "gauge_reconstruction_max_abs_delta",
        ],
    })

    base_dir = run_base_model(Namespace(
        config=str(config_path), output_dir=str(run_dir / "base_model"), smoke=False,
    ))
    measurement = run_measurement_audit(config, run_dir / "figure_data")
    _write_csv(run_dir / "unit_scale_audit.csv", measurement["summary_rows"])
    posterior = _synthetic_posterior_calibration(config, workers=int(config["analysis"].get("workers", 1)))
    _write_csv(run_dir / "posterior_calibration.csv", posterior["rows"])
    _write_json(run_dir / "figure_data" / "posterior_calibration.json", posterior)

    trajectory_rows = _read_csv(base_dir / "trajectories.csv")
    train, _, train_gauge_rows = _collect_evidence(
        trajectory_rows, e0["split"]["train_subjects"], overlay=False, use_target_gauge=True,
    )
    val, component_rows, val_gauge_rows = _collect_evidence(
        trajectory_rows, e0["split"]["validation_subjects"], overlay=True, use_target_gauge=True,
    )
    train_raw, _, _ = _collect_evidence(
        trajectory_rows, e0["split"]["train_subjects"], overlay=False, use_target_gauge=False,
    )
    val_raw, _, _ = _collect_evidence(
        trajectory_rows, e0["split"]["validation_subjects"], overlay=False, use_target_gauge=False,
    )
    alphas = [float(value) for value in e0["ridge_alphas"]]
    coordinate_rows_raw, observability_traces_raw = _coordinate_audit_train_cv(
        train_raw, val_raw, alphas=alphas, permutations=int(e0["permutation_iterations"]),
        quantile=float(e0["null_quantile"]), rng=np.random.default_rng(seed + 101),
    )
    coordinate_rows_raw = _apply_local_target_contract(coordinate_rows_raw, e0["target_gauge"])
    coordinate_rows, observability_traces = _coordinate_audit_train_cv(
        train, val, alphas=alphas, permutations=int(e0["permutation_iterations"]),
        quantile=float(e0["null_quantile"]), rng=np.random.default_rng(seed + 101),
    )
    coordinate_rows = _apply_local_target_contract(coordinate_rows, e0["target_gauge"])
    gauge_gain_rows = _gauge_gain_rows(coordinate_rows_raw, coordinate_rows)
    admitted = {
        modality: [
            index for index, name in enumerate(EEG_LOCAL_NAMES if modality == "eeg" else FNIRS_LOCAL_NAMES)
            if any(
                row["modality"] == modality
                and row["coordinate"] == name
                and row["admitted_local_target"]
                for row in coordinate_rows
            )
        ]
        for modality in ("eeg", "fnirs")
    }
    vocabulary_rows: list[dict[str, Any]] = []
    vocabulary_plot: dict[str, Any] = {}
    if all(admitted[modality] for modality in ("eeg", "fnirs")):
        vocabulary_rows, vocabulary_plot = _vocabulary_audit(
            train, val,
            codebook_size=int(e0["codebook_size"]),
            random_references=int(e0["random_reference_iterations"]),
            quantile=float(e0["null_quantile"]),
            rng=np.random.default_rng(seed + 211),
            admitted=admitted,
        )
    else:
        # Vocabulary admission is a two-modality conjunction.  If either local
        # target family is empty, there is no joint target contract to admit.
        for modality in ("eeg", "fnirs"):
            row, plot = _failed_vocabulary_payload(modality, int(e0["codebook_size"]))
            if admitted[modality]:
                row["failure_reason"] = "other modality has no admitted local coordinate"
            vocabulary_rows.append(row)
            vocabulary_plot[modality] = plot
    context_rows, context_traces = _context_audit(
        train, val, alphas=alphas, shuffles=int(e0["context_shuffle_iterations"]),
        quantile=float(e0["null_quantile"]), rng=np.random.default_rng(seed + 307),
    )
    context_rows_raw, context_traces_raw = _context_audit(
        train_raw, val_raw, alphas=alphas, shuffles=int(e0["context_shuffle_iterations"]),
        quantile=float(e0["null_quantile"]), rng=np.random.default_rng(seed + 307),
    )
    physical_rows = _physical_audit(val.physical_rows)
    component_summary = []
    for component in ("hbo", "hbr"):
        selected = [row for row in component_rows if row["component"] == component]
        clean = np.asarray([row["mse_clean"] for row in selected])
        zero = np.asarray([row["mse_zero"] for row in selected])
        history = np.asarray([row["mse_history"] for row in selected])
        baseline_name, baseline = ("zero", zero) if np.mean(zero) <= np.mean(history) else ("history", history)
        gains = baseline - clean
        subjects = np.asarray([row["subject"] for row in selected])
        per_subject = [float(np.mean(gains[subjects == subject])) for subject in np.unique(subjects)]
        component_summary.append({
            "component": component,
            "baseline": baseline_name,
            "clean_mse": float(np.mean(clean)),
            "baseline_mse": float(np.mean(baseline)),
            "mean_gain": float(np.mean(per_subject)),
            "positive_subject_fraction": float(np.mean(np.asarray(per_subject) > 0)),
            "subjects": len(per_subject),
        })

    _write_csv(run_dir / "local_target_observability.csv", coordinate_rows)
    _write_csv(run_dir / "local_target_observability_pre_gauge.csv", coordinate_rows_raw)
    _write_csv(run_dir / "gauge_target_gain.csv", gauge_gain_rows)
    _write_csv(run_dir / "gauge_alignment.csv", train_gauge_rows + val_gauge_rows)
    _write_csv(run_dir / "vocabulary_transmissibility.csv", vocabulary_rows)
    _write_csv(run_dir / "continuous_coupling_upper_bound.csv", context_rows)
    _write_csv(run_dir / "continuous_coupling_upper_bound_pre_gauge.csv", context_rows_raw)
    _write_csv(run_dir / "physical_observation_checks.csv", physical_rows)
    _write_csv(run_dir / "physical_observation_components.csv", component_summary)
    _write_csv(run_dir / "teacher_mask_coverage.csv", val.mask_rows)
    _write_csv(run_dir / "mask_coverage.csv", val.mask_rows)
    _write_json(run_dir / "figure_data" / "target_observability.json", {
        "rows": coordinate_rows, "traces": observability_traces,
    })
    _write_json(run_dir / "figure_data" / "gauge_alignment.json", {
        "mode": GAUGE_MODE,
        "gain_rows": gauge_gain_rows,
        "fold_rows": train_gauge_rows + val_gauge_rows,
        "pre_gauge_rows": coordinate_rows_raw,
        "post_gauge_rows": coordinate_rows,
        "pre_gauge_traces": observability_traces_raw,
    })
    _write_json(run_dir / "figure_data" / "vocabulary_transmissibility.json", {
        "rows": vocabulary_rows, "plot": vocabulary_plot,
    })
    _write_json(run_dir / "figure_data" / "continuous_coupling_upper_bound.json", {
        "rows": context_rows, "traces": context_traces,
        "pre_gauge_rows": context_rows_raw,
        "pre_gauge_traces": context_traces_raw,
    })
    _write_json(run_dir / "figure_data" / "physical_teacher_overlay.json", val.overlay or {})
    _write_json(run_dir / "figure_data" / "physical_observation_checks.json", {"rows": physical_rows})
    _write_json(run_dir / "target_contract.json", {
        "schema": SCHEMA,
        "teacher": "adaptive_joint local fixed-interval smoother",
        "target_gauge": {
            "mode": GAUGE_MODE,
            "training_only_parameters": [
                "hbo_gain", "hbr_gain", "hbo_std", "hbr_std",
            ],
            "event_baseline": "same deterministic pre-task baseline transform as canonical input",
            "chromophore_coordinates": "posterior observation means in canonical measurement space",
            "flow_coordinate_role": "context_only latent diagnostic",
            "reconstruction_invariant": True,
        },
        "local_state_projection": {
            "eeg": [EEG_LOCAL_NAMES[index] for index in admitted["eeg"]],
            "fnirs": [FNIRS_LOCAL_NAMES[index] for index in admitted["fnirs"]],
        },
        "context_transition_target": ["fnirs_level", "fnirs_innovation"],
        "state_order": list(STATE_NAMES),
        "required_local_coordinates": e0["target_gauge"]["required_local_coordinates"],
        "optional_local_coordinates": e0["target_gauge"].get("optional_local_coordinates", {}),
        "posterior_covariance_role": "uncertainty weighting only after held-out synthetic variance calibration",
        "physical_observation_mean_role": "soft multimodal teacher decoder target",
        "cross_modal_claim_control": "base_model adaptive_eeg_only path",
        "protected_test_status": "closed",
    })
    _write_json(run_dir / "evidence_calibration.json", {
        "schema": SCHEMA,
        "label_permutation_iterations": int(e0["permutation_iterations"]),
        "ridge_alpha_selection": "five_fold_train_subject_group_cv",
        "random_centroid_iterations": int(e0["random_reference_iterations"]),
        "context_shuffle_iterations": int(e0["context_shuffle_iterations"]),
        "null_quantile": float(e0["null_quantile"]),
        "synthetic_posterior_calibration": posterior,
        "protected_test_used": False,
    })

    measurement_pass = bool(
        measurement["crop_position_invariance_max_abs_delta"] <= 1e-12
        and all(row["finite_fraction"] == 1.0 for row in measurement["summary_rows"])
    )
    all_gauge_rows = train_gauge_rows + val_gauge_rows
    gauge_pass = bool(
        all(row["finite_non_singular"] for row in all_gauge_rows)
        and max(float(row["reconstruction_max_abs_delta"]) for row in all_gauge_rows) <= 1e-8
    )
    local_pass = _required_local_pass(coordinate_rows, e0["target_gauge"])
    local_pass_pre_gauge = _required_local_pass(coordinate_rows_raw, e0["target_gauge"])
    vocabulary_pass = all(bool(row["above_random_reference"]) for row in vocabulary_rows)
    coupling_pass = any(
        row.get("coordinate") == "joint_logdet" and row.get("above_shuffled_eeg")
        for row in context_rows
    )
    physical_pass = all(float(row["mean_gain"]) > 0.0 for row in physical_rows)
    uncertainty_pass = bool(posterior["posterior_calibration_pass"])
    machine_pass = bool(
        measurement_pass and gauge_pass and local_pass and vocabulary_pass
        and coupling_pass and physical_pass and uncertainty_pass
    )
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "validation": {
            "measurement_contract_pass": measurement_pass,
            "target_gauge_contract_pass": gauge_pass,
            "local_target_observability_pass": local_pass,
            "finite_vocabulary_transmissibility_pass": vocabulary_pass,
            "continuous_coupling_upper_bound_pass": coupling_pass,
            "physical_observation_pass": physical_pass,
            "posterior_uncertainty_calibration_pass": uncertainty_pass,
            "machine_validation_pass": machine_pass,
            "visual_review": "pending",
        },
        "protected_test": {
            "opened": False,
            "subjects": e0["split"]["protected_test_subjects"],
            "eligible_after_visual_review": machine_pass,
            "reason": "requires machine validation and completed visual review in a separate run",
        },
        "measurement": {
            "canonical_scale_max_ratio": measurement["validation_canonical_scale_max_ratio"],
            "crop_position_max_abs_delta": measurement["crop_position_invariance_max_abs_delta"],
        },
        "target_gauge": {
            "mode": GAUGE_MODE,
            "contract_pass": gauge_pass,
            "max_reconstruction_abs_delta": max(
                float(row["reconstruction_max_abs_delta"]) for row in all_gauge_rows
            ),
            "finite_non_singular_fold_fraction": float(np.mean([
                bool(row["finite_non_singular"]) for row in all_gauge_rows
            ])),
            "required_local_pass_pre_gauge": local_pass_pre_gauge,
            "required_local_pass_post_gauge": local_pass,
            "coordinate_gains": gauge_gain_rows,
        },
        "local_targets": coordinate_rows,
        "local_targets_pre_gauge": coordinate_rows_raw,
        "vocabulary": vocabulary_rows,
        "continuous_coupling": context_rows,
        "continuous_coupling_pre_gauge": context_rows_raw,
        "physical_observation": physical_rows,
        "physical_observation_components": component_summary,
        "posterior_calibration": posterior,
        "sample_counts": {
            "train_local_patches": int(len(train.subjects)),
            "validation_local_patches": int(len(val.subjects)),
            "train_context_rows": int(len(train.fnirs_level)),
            "validation_context_rows": int(len(val.fnirs_level)),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    source_paths = [
        config_path,
        Path(__file__),
        REPO_ROOT / "experiments/evaluate_adaptive_shared_neural_ssm.py",
        REPO_ROOT / "experiments/evaluate_physical_teacher_e0_v2.py",
        REPO_ROOT / "experiments/e0_v2_measurement_audit.py",
        REPO_ROOT / "experiments/scripts/visualize_e0_v2_audit.py",
        REPO_ROOT / "src/inference/adaptive_neurovascular_ssm.py",
        base_dir / "manifest.json",
    ]
    _write_json(run_dir / "manifest.json", {
        "schema": SCHEMA,
        "command": " ".join(sys.argv),
        "completion_status": "validation_complete",
        "start_policy": "validation-only; protected subjects 24-29 closed",
        "git_commit": summary["git_commit"],
        "dirty_worktree": bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()),
        "seed": seed,
        "base_model_run": str(base_dir.relative_to(run_dir)),
        "input_hashes": [{"path": str(path), "sha256": _sha256(path)} for path in source_paths],
        "decision_protocol": "decision_protocol.yaml",
        "metric_registry": "metric_registry.json",
        "evidence_calibration": "evidence_calibration.json",
        "protected_test_used": False,
    })
    visualizer = REPO_ROOT / "experiments/scripts/visualize_e0_v2_audit.py"
    subprocess.run([sys.executable, str(visualizer), "--run-dir", str(run_dir)], cwd=REPO_ROOT, check=True)
    print(run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/physiology_semantic_tokenizer/adaptive_teacher_e0_v3.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
