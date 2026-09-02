#!/usr/bin/env python3
"""Measured EEG/fNIRS reconstruction and cross-modal null diagnostics.

This is a deliberately small, development-only measured entry point.  It
opens one public Single-Trial condition (MA, session_01), fits on subjects
01--18, and applies frozen transforms/parameters to subjects 19--23.  The
protected subjects 24--29 are rejected before the shared loader is called.

The four models are observation baselines (T0/T1), the existing adaptive
Croce-like smoother (T2b), and the fixed-parameter robust Balloon smoother
(T3a).  A real-data run has no clean physiological target; all reconstruction
metrics therefore compare against the held-out measured target and all nulls
are labelled cross-modal specificity diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_adaptive_shared_neural_ssm import (
    EEGAdapter,
    _apply_eeg_adapter,
    _fit_eeg_adapter,
    _paired_hbr_indices,
)
from experiments.evaluate_shared_neural_driver_unified import (
    Trial,
    _load_trials,
    _safe_corr,
    _select_active_hbo,
    _write_csv,
)
from experiments.evaluate_t3a_balloon_robust_p0 import _fit_lds_1d, _kalman_smooth_1d
from src.inference.adaptive_neurovascular_ssm import (
    AdaptiveSSMFit,
    apply_adaptive_ssm,
    fit_adaptive_ssm,
    fit_to_mapping,
)
from src.inference.t3a_balloon_robust_ssm import (
    BalloonConfig,
    BalloonFixedParameters,
    BalloonFreeParameters,
    BalloonObservationSpec,
    BalloonParameters,
    smooth_balloon,
)
from src.metrics.trajectory_reliability import trajectory_reliability_metrics


SCHEMA = "t3_measured_reconstruction_null_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/t3_measured_reconstruction_null_v1.yaml"
MODEL_IDS = ("T0", "T1", "T2b", "T3a")
MODE_IDS = ("joint", "center_masked_eeg", "center_masked_fnirs", "eeg_only", "fnirs_only")
NULL_IDS = ("independent", "pairing", "time_shift")
OBS_NAMES = ("EEG", "HbO", "HbR")
MODEL_LABELS = {
    "T0": "T0 持续性基线",
    "T1": "T1 独立线性模型",
    "T2b": "T2b 自适应 Croce 类模型",
    "T3a": "T3a 鲁棒气球模型",
}
CJK_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "AR PL UMing CN",
        "DejaVu Sans",
    ],
    "axes.unicode_minus": True,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


@dataclass(frozen=True)
class PreparedTrial:
    trial: Trial
    eeg_driver: np.ndarray
    hbo: np.ndarray
    hbr: np.ndarray


@dataclass(frozen=True)
class ModelBundle:
    t0: Mapping[str, Any]
    t1: tuple[tuple[float, float, float], ...]
    t2b: AdaptiveSSMFit | None
    t3a: tuple[BalloonParameters, BalloonObservationSpec, BalloonConfig]


@dataclass(frozen=True)
class NullCase:
    receiver: PreparedTrial
    values: np.ndarray
    donor_hbo: np.ndarray
    donor_hbr: np.ndarray
    donor_dataset_id: str
    donor_subject: str
    donor_record_id: str
    donor_event_index: int


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _subject_range(prefix: str, first: int, last: int) -> list[str]:
    return [f"{prefix}{index:02d}" for index in range(first, last + 1)]


def _physical_check_status(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "pass" if bool(value) else "fail"
    return "diagnostic"


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the measured boundary before any loader/data access."""

    if str(config.get("schema", "")) != SCHEMA:
        raise ValueError("measured SSM config schema mismatch")
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment section is required")
    if experiment.get("scope") != "measured_development_exploratory":
        raise ValueError("unexpected measured experiment scope")
    if experiment.get("measured_data_enabled") is not True:
        raise ValueError("measured data must be explicitly enabled")
    if experiment.get("protected_data_enabled") is not False:
        raise ValueError("protected data must remain disabled")
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("data section is required")
    conditions = data.get("conditions")
    if not isinstance(conditions, Sequence) or len(conditions) != 1:
        raise ValueError("the measured entry point requires exactly one condition")
    condition = conditions[0]
    required = {
        "condition_id": "single_trial_ma_session_01",
        "dataset_id": "eeg_fnirs_single_trial",
        "record_id": "session_01",
        "target_label": "MA",
        "eeg_signal_branch": "raw_with_ocular_artifact",
    }
    for key, expected in required.items():
        if condition.get(key) != expected:
            raise ValueError(f"condition.{key} must be {expected!r}")
    fit_subjects = [str(item) for item in condition.get("fit_subjects", ())]
    validation_subjects = [str(item) for item in condition.get("validation_subjects", ())]
    protected_subjects = [str(item) for item in condition.get("protected_subjects", ())]
    expected_fit = _subject_range("subject_", 1, 18)
    expected_validation = _subject_range("subject_", 19, 23)
    expected_protected = _subject_range("subject_", 24, 29)
    if fit_subjects != expected_fit or validation_subjects != expected_validation:
        raise ValueError("fit/validation subject registries do not match the public split")
    if protected_subjects != expected_protected:
        raise ValueError("protected subject registry does not match the closed split")
    subjects = [str(item) for item in condition.get("subjects", ())]
    if subjects != expected_fit + expected_validation:
        raise ValueError("loader subjects must contain fit followed by validation only")
    if set(subjects) & set(protected_subjects):
        raise ValueError("protected subjects are present in the loader subject list")
    if int(condition.get("max_trials_per_subject", 0)) < 3:
        raise ValueError("at least three trials per subject are required")
    if str(data.get("cache_root", "")).startswith("/tmp"):
        raise ValueError("the measured cache must not use a system temporary path")
    for key, expected in {
        "window_duration_s": 20.0,
        "window_offset_s": -5.0,
        "baseline_duration_s": 5.0,
        "task_duration_s": 10.0,
    }.items():
        if not math.isclose(float(data.get(key, float("nan"))), expected):
            raise ValueError(f"data.{key} must be {expected}")

    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("analysis section is required")
    if tuple(analysis.get("models", ())) != MODEL_IDS:
        raise ValueError(f"models must be exactly {MODEL_IDS}")
    if tuple(analysis.get("modes", ())) != MODE_IDS:
        raise ValueError(f"modes must be exactly {MODE_IDS}")
    if tuple(analysis.get("nulls", ())) != NULL_IDS:
        raise ValueError(f"nulls must be exactly {NULL_IDS}")
    if not math.isclose(float(analysis.get("sampling_hz", 0.0)), 10.0):
        raise ValueError("the measured entry point requires 10 Hz fNIRS/driver coordinates")
    if not math.isclose(float(analysis.get("time_shift_s", 0.0)), 10.0):
        raise ValueError("the measured null shift must be +10 s")
    if bool(analysis.get("time_shift_circular", True)):
        raise ValueError("time-shift null must be non-circular")
    center = analysis.get("center_mask")
    if not isinstance(center, Mapping) or float(center.get("duration_s", 0.0)) <= 0.0:
        raise ValueError("center_mask requires a positive duration")
    calibration = analysis.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("fit-cohort calibration is required")
    if calibration.get("eeg_loading") != "fit_sd_to_driver_target_sd":
        raise ValueError("EEG loading must use the fit-cohort SD gauge")
    if calibration.get("fnirs_common_scale") != "pooled_std_and_hbr_q001_physical_margin":
        raise ValueError("fNIRS must use the fit-cohort physical-domain gauge")
    if calibration.get("observation_scale") != "robust_first_difference_student_t":
        raise ValueError("observation scale must use the fit-cohort robust Student-t gauge")
    driver_target_sd = float(calibration.get("driver_target_sd", 0.0))
    hbr_lower_margin = float(calibration.get("hbr_lower_margin", 0.0))
    if not np.isfinite(driver_target_sd) or driver_target_sd <= 0.0:
        raise ValueError("calibration.driver_target_sd must be positive")
    if not np.isfinite(hbr_lower_margin) or not -0.35 < hbr_lower_margin < 0.0:
        raise ValueError("calibration.hbr_lower_margin must lie inside the fixed Q0=-0.35 boundary")
    floors = calibration.get("observation_scale_floor")
    if not isinstance(floors, Mapping) or any(float(floors.get(name, 0.0)) <= 0.0 for name in OBS_NAMES):
        raise ValueError("positive observation floors for EEG/HbO/HbR are required")

    ssm = config.get("ssm")
    if not isinstance(ssm, Mapping) or not isinstance(ssm.get("t2b"), Mapping) or not isinstance(ssm.get("t3a"), Mapping):
        raise ValueError("t2b and t3a settings are required")
    fixed = ssm["t3a"].get("fixed")
    if not isinstance(fixed, Mapping):
        raise ValueError("t3a.fixed is required")
    if not math.isclose(float(fixed.get("kappa_per_s", 0.0)), 0.64) or not math.isclose(float(fixed.get("tau_s", 0.0)), 2.0):
        raise ValueError("T3a kappa=.64 and tau=2 must remain fixed")
    parameter_fit = ssm["t3a"].get("parameter_fit")
    if not isinstance(parameter_fit, Mapping):
        raise ValueError("t3a.parameter_fit is required")
    expected_parameters = {"beta", "kappa", "tau", "gamma", "alpha", "E0"}
    parameter_specs = parameter_fit.get("parameters")
    if not isinstance(parameter_specs, Mapping) or set(parameter_specs) != expected_parameters:
        raise ValueError("t3a.parameter_fit.parameters must define beta/kappa/tau/gamma/alpha/E0")
    for name, spec in parameter_specs.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"parameter-fit specification for {name} must be a mapping")
        bounds = tuple(float(value) for value in spec.get("bounds", ()))
        if len(bounds) != 2 or not bounds[0] < float(spec.get("prior_mean", np.nan)) < bounds[1]:
            raise ValueError(f"parameter-fit bounds/prior for {name} are invalid")
        if float(spec.get("prior_sd", 0.0)) <= 0.0:
            raise ValueError(f"parameter-fit prior SD for {name} must be positive")
    stages = parameter_fit.get("stages")
    if (
        not isinstance(stages, Sequence)
        or not stages
        or not isinstance(stages[0], Mapping)
        or tuple(stages[0].get("free", ())) != ()
    ):
        raise ValueError("parameter-fit stages must begin with a fixed baseline")
    stage_ids: list[str] = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ValueError("each parameter-fit stage must be a mapping")
        stage_id = str(stage.get("id", ""))
        free = tuple(str(value) for value in stage.get("free", ()))
        if not stage_id or stage_id in stage_ids or not set(free).issubset(expected_parameters):
            raise ValueError("parameter-fit stage ids/free parameters are invalid")
        stage_ids.append(stage_id)
    heldout_positions = tuple(int(value) for value in parameter_fit.get("heldout_trial_positions", ()))
    if len(heldout_positions) != 2 or len(set(heldout_positions)) != 2 or min(heldout_positions) < 0:
        raise ValueError("parameter fit requires two distinct non-negative heldout trial positions")
    if int(parameter_fit.get("workers", 0)) <= 0 or int(parameter_fit.get("optimizer_max_iterations", 0)) <= 0:
        raise ValueError("parameter-fit workers and optimizer iterations must be positive")
    output = config.get("output")
    expected_root = "experiments/runs/physiology_semantic_tokenizer/t3_measured_reconstruction_null"
    if not isinstance(output, Mapping) or str(output.get("root", "")) != expected_root:
        raise ValueError(f"output.root must be the registered workspace path {expected_root}")


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError("configuration must be a mapping")
    config = dict(value)
    validate_config(config)
    return config


def _require_full_support(trial: Trial) -> None:
    eeg_mask = np.asarray(trial.eeg_valid_mask, dtype=bool) if trial.eeg_valid_mask is not None else np.ones(len(trial.eeg), dtype=bool)
    fnirs_mask = np.asarray(trial.fnirs_valid_mask, dtype=bool) if trial.fnirs_valid_mask is not None else np.ones(len(trial.fnirs), dtype=bool)
    if eeg_mask.shape != (len(trial.eeg),) or fnirs_mask.shape != (len(trial.fnirs),):
        raise RuntimeError(f"{trial.subject} event {trial.event_index}: validity mask shape mismatch")
    if not np.all(eeg_mask) or not np.all(fnirs_mask) or not np.all(np.isfinite(trial.eeg)) or not np.all(np.isfinite(trial.fnirs)):
        raise RuntimeError(f"{trial.subject} event {trial.event_index}: unsupported missing/non-finite window")


def _targets(trials: Sequence[Trial], indices: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    hbo = [np.mean(trial.fnirs[:, indices], axis=1, dtype=np.float64) for trial in trials]
    hbr_indices = _paired_hbr_indices(trials[0], indices)
    hbr = [np.mean(trial.fnirs[:, hbr_indices], axis=1, dtype=np.float64) for trial in trials]
    return hbo, hbr


def _robust_scale(values: Sequence[np.ndarray]) -> float:
    stacked = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
    stacked = stacked[np.isfinite(stacked)]
    if len(stacked) < 2:
        return 1.0
    median = float(np.median(stacked))
    mad = float(np.median(np.abs(stacked - median))) * 1.4826
    if mad > 1e-8 and np.isfinite(mad):
        return mad
    std = float(np.std(stacked))
    return std if std > 1e-8 and np.isfinite(std) else 1.0


def _first_difference_scale(values: Sequence[np.ndarray]) -> float:
    differences = [np.diff(np.asarray(value, dtype=np.float64).reshape(-1)) for value in values]
    return _robust_scale(differences) / math.sqrt(2.0)


def _fit_t0(values: Sequence[np.ndarray]) -> dict[str, Any]:
    differences = np.concatenate([np.diff(np.asarray(value, dtype=np.float64), axis=0) for value in values], axis=0)
    variance = np.nanvar(differences, axis=0) * 0.5
    return {"variance": np.maximum(np.nan_to_num(variance, nan=0.05), 1e-8)}


def _apply_t0(values: np.ndarray, fit: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = np.zeros_like(values)
    for column in range(values.shape[1]):
        previous = 0.0
        for index, value in enumerate(values[:, column]):
            mean[index, column] = previous
            if np.isfinite(value):
                previous = float(value)
    variance = np.broadcast_to(np.asarray(fit["variance"])[None, :], values.shape).copy()
    return mean, np.sqrt(np.maximum(variance, 0.0)), np.full_like(mean, np.nan)


def _fit_t1(values: Sequence[np.ndarray]) -> tuple[tuple[float, float, float], ...]:
    # NaN separators prevent an artificial AR transition between event windows.
    separated = [
        np.concatenate(
            [part for index, value in enumerate(values) for part in (
                np.asarray(value)[:, column],
                np.asarray([np.nan]) if index + 1 < len(values) else np.asarray([], dtype=np.float64),
            )]
        )
        for column in range(3)
    ]
    return tuple(_fit_lds_1d(value) for value in separated)


def _apply_t1(values: np.ndarray, fit: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, variances = [], []
    for column in range(values.shape[1]):
        mean, variance = _kalman_smooth_1d(values[:, column], *fit[column])
        means.append(mean)
        variances.append(variance + float(fit[column][2]))
    estimate = np.column_stack(means)
    std = np.sqrt(np.maximum(np.column_stack(variances), 0.0))
    return estimate, std, np.full_like(estimate, np.nan)


def _center_mask(length: int, fs_hz: float, start_s: float, duration_s: float) -> np.ndarray:
    mask = np.zeros(int(length), dtype=bool)
    start = max(0, int(round(float(start_s) * fs_hz)))
    stop = min(int(length), start + max(1, int(round(float(duration_s) * fs_hz))))
    mask[start:stop] = True
    return mask


def _identity_fields(
    item: PreparedTrial,
    condition: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    start_s = float(data_config["window_offset_s"])
    end_s = start_s + float(data_config["window_duration_s"])
    trial = item.trial
    sample_id = (
        f"{trial.dataset_id}|{trial.subject}|{trial.record_id}"
        f"|task={condition['target_label']}|event={int(trial.event_index)}"
        f"|start_ms={int(round(start_s * 1000.0))}"
        f"|eeg_branch={condition['eeg_signal_branch']}"
    )
    return {
        "sample_id": sample_id,
        "dataset_id": trial.dataset_id,
        "condition_id": trial.condition_id,
        "subject": trial.subject,
        "record_id": trial.record_id,
        "task_label": str(condition["target_label"]),
        "event_index": int(trial.event_index),
        "window_start_s": start_s,
        "window_end_s": end_s,
        "eeg_signal_branch": str(condition["eeg_signal_branch"]),
    }


def _mode_input(series: PreparedTrial, mode: str, center: np.ndarray | None = None) -> np.ndarray:
    values = np.column_stack((series.eeg_driver, series.hbo, series.hbr)).astype(np.float64, copy=True)
    if mode == "center_masked_eeg":
        assert center is not None
        values[center, 0] = np.nan
    elif mode == "center_masked_fnirs":
        assert center is not None
        values[center, 1:] = np.nan
    elif mode == "eeg_only":
        values[:, 1:] = np.nan
    elif mode == "fnirs_only":
        values[:, 0] = np.nan
    elif mode != "joint":
        raise ValueError(f"unsupported mode {mode}")
    return values


def _fit_models(
    fit_series: Sequence[PreparedTrial],
    config: Mapping[str, Any],
    *,
    fit_comparison_models: bool = True,
) -> tuple[ModelBundle, dict[str, Any]]:
    values = [np.column_stack((item.eeg_driver, item.hbo, item.hbr)) for item in fit_series]
    calibration = {
        "eeg_signal_scale": _robust_scale([item.eeg_driver for item in fit_series]),
        "hbo_signal_scale": _robust_scale([item.hbo for item in fit_series]),
        "hbr_signal_scale": _robust_scale([item.hbr for item in fit_series]),
        "hbt_signal_scale": _robust_scale([item.hbo + item.hbr for item in fit_series]),
    }
    calibration_cfg = config["analysis"]["calibration"]
    floors_cfg = calibration_cfg["observation_scale_floor"]
    raw_observation_scales = {
        "EEG": _first_difference_scale([item.eeg_driver for item in fit_series]),
        "HbO": _first_difference_scale([item.hbo for item in fit_series]),
        "HbR": _first_difference_scale([item.hbr for item in fit_series]),
    }
    student_nu = float(config["ssm"]["t3a"]["fixed"]["student_t_df"])
    student_scale_factor = math.sqrt((student_nu - 2.0) / student_nu)
    observation_scales = {
        name: max(
            float(raw_observation_scales[name]) * student_scale_factor,
            float(floors_cfg[name]),
        )
        for name in OBS_NAMES
    }
    fit_driver_sd = float(np.std(np.concatenate([item.eeg_driver for item in fit_series])))
    driver_target_sd = float(calibration_cfg["driver_target_sd"])
    eeg_loading = fit_driver_sd / driver_target_sd
    hbr_lower = float(np.quantile(np.concatenate([item.hbr for item in fit_series]), 0.001))
    hbr_margin = float(calibration_cfg["hbr_lower_margin"])
    pooled_fnirs = np.concatenate([np.concatenate((item.hbo, item.hbr)) for item in fit_series])
    pooled_fnirs_std = float(np.std(pooled_fnirs))
    fnirs_common_scale = max(
        pooled_fnirs_std,
        float(calibration["hbt_signal_scale"]),
        float(calibration["hbr_signal_scale"]),
        abs(hbr_lower) / abs(hbr_margin),
    )
    calibration.update({
        "fit_driver_sd": fit_driver_sd,
        "driver_target_sd": driver_target_sd,
        "eeg_loading": eeg_loading,
        "hbr_q001": hbr_lower,
        "hbr_lower_margin": hbr_margin,
        "fnirs_pooled_std": pooled_fnirs_std,
        "fnirs_common_scale": fnirs_common_scale,
        "observation_scale_raw_EEG": float(raw_observation_scales["EEG"]),
        "observation_scale_raw_HbO": float(raw_observation_scales["HbO"]),
        "observation_scale_raw_HbR": float(raw_observation_scales["HbR"]),
        "observation_scale_EEG": float(observation_scales["EEG"]),
        "observation_scale_HbO": float(observation_scales["HbO"]),
        "observation_scale_HbR": float(observation_scales["HbR"]),
        "student_t_sd_to_scale_factor": student_scale_factor,
        "observation_scale_floors": {name: float(floors_cfg[name]) for name in OBS_NAMES},
    })
    t0 = _fit_t0(values) if fit_comparison_models else {}
    t1 = _fit_t1(values) if fit_comparison_models else ()
    t2b: AdaptiveSSMFit | None = None
    if fit_comparison_models:
        t2_cfg = config["ssm"]["t2b"]
        t2b = fit_adaptive_ssm(
            [item.eeg_driver for item in fit_series],
            [item.hbo for item in fit_series],
            [item.hbr for item in fit_series],
            fs_hz=float(config["analysis"]["sampling_hz"]),
            prior_strength=float(t2_cfg["prior_strength"]),
            max_iterations=int(t2_cfg["max_iterations"]),
            q_scale_candidates=tuple(float(value) for value in t2_cfg["q_scale_candidates"]),
            fnirs_noise_scale_candidates=tuple(float(value) for value in t2_cfg["fnirs_noise_scale_candidates"]),
            balance_penalty=float(t2_cfg["balance_penalty"]),
            max_flow_perturbation=float(t2_cfg["max_flow_perturbation"]),
            baseline_samples=int(round(float(config["data"]["baseline_duration_s"]) * float(config["analysis"]["sampling_hz"]))),
        )
    t3_cfg = config["ssm"]["t3a"]
    fixed_cfg = t3_cfg["fixed"]
    # P0/Q0 are observation-coordinate gauges, not absolute concentrations.
    # Preserve the fixed 1:0.35 ratio while a common fit-only scale keeps HbR
    # inside the positive-compartment domain.
    p0 = float(fnirs_common_scale)
    q0 = 0.35 * float(fnirs_common_scale)
    calibration.update({"t3a_P0_gauge": p0, "t3a_Q0_gauge": q0})
    fixed = BalloonFixedParameters(
        alpha=float(fixed_cfg["alpha"]),
        E0=float(fixed_cfg["e0"]),
        gamma=float(fixed_cfg["gamma"]),
        P0=p0,
        Q0=q0,
        driver_decay_per_s=float(fixed_cfg["driver_decay_per_s"]),
        process_std=tuple(float(value) for value in fixed_cfg["process_std"]),
        observation_scale=(
            float(observation_scales["EEG"]),
            float(observation_scales["HbO"]),
            float(observation_scales["HbR"]),
        ),
        student_nu=float(fixed_cfg["student_t_df"]),
        eeg_loading=float(eeg_loading),
        eeg_offset=float(fixed_cfg["eeg_offset"]),
        neurovascular_gain=float(fixed_cfg["neurovascular_gain"]),
    )
    parameters = BalloonParameters(
        fixed=fixed,
        free=BalloonFreeParameters(kappa=float(fixed_cfg["kappa_per_s"]), tau=float(fixed_cfg["tau_s"])),
    )
    balloon_config = BalloonConfig(
        dt=1.0 / float(config["analysis"]["sampling_hz"]),
        rk4_substeps=int(t3_cfg["rk4_substeps"]),
        irls_iterations=int(t3_cfg["irls_iterations"]),
        irls_weight_floor=float(t3_cfg["irls_weight_floor"]),
        initial_state_std=tuple(float(value) for value in t3_cfg["initial_state_std"]),
    )
    spec = BalloonObservationSpec(
        eeg_loading=fixed.eeg_loading,
        eeg_offset=fixed.eeg_offset,
        observation_scale=fixed.observation_scale,
        student_nu=fixed.student_nu,
    )
    parameters.validate()
    spec.validate()
    return ModelBundle(t0=t0, t1=t1, t2b=t2b, t3a=(parameters, spec, balloon_config)), calibration


def _run_model(model: str, bundle: ModelBundle, values: np.ndarray) -> dict[str, Any]:
    if model == "T0":
        mean, std, _ = _apply_t0(values, bundle.t0)
        return {"estimate": mean, "predictive_std": std, "driver": None, "driver_std": None, "states": None, "state_std": None, "state_names": ()}
    if model == "T1":
        mean, std, _ = _apply_t1(values, bundle.t1)
        return {"estimate": mean, "predictive_std": std, "driver": None, "driver_std": None, "states": None, "state_std": None, "state_names": ()}
    if model == "T2b":
        if bundle.t2b is None:
            raise RuntimeError("T2b comparison model was not fitted")
        mode = "joint" if np.any(np.isfinite(values[:, 1])) and np.any(np.isfinite(values[:, 2])) else "eeg_only"
        if not np.any(np.isfinite(values[:, 0])):
            mode = "fnirs_only"
        result = apply_adaptive_ssm(
            values[:, 0] if np.any(np.isfinite(values[:, 0])) else None,
            bundle.t2b,
            hbo_observation=values[:, 1] if mode in {"joint", "fnirs_only"} else None,
            hbr_observation=values[:, 2] if mode in {"joint", "fnirs_only"} else None,
            observation_mode=mode,
        )
        return {
            "estimate": np.column_stack((result.eeg_reconstructed, result.hbo_reconstructed, result.hbr_reconstructed)),
            "predictive_std": np.asarray(result.observation_predictive_std, dtype=np.float64),
            "driver": np.asarray(result.states[:, 4], dtype=np.float64),
            "driver_std": np.asarray(result.state_std[:, 4], dtype=np.float64),
            "states": np.asarray(result.states, dtype=np.float64),
            "state_std": np.asarray(result.state_std, dtype=np.float64),
            "state_names": ("vasodilation_s", "flow_delta", "hbo_state", "hbr_state", "shared_driver"),
            "physical_checks": None,
        }
    if model == "T3a":
        parameters, spec, balloon_config = bundle.t3a
        mask = np.isfinite(values)
        result = smooth_balloon(values, parameters=parameters, observation_spec=spec, config=balloon_config, observation_mask=mask)
        return {
            "estimate": np.asarray(result.observation_mean, dtype=np.float64),
            "predictive_std": np.sqrt(np.maximum(np.asarray(result.total_variance, dtype=np.float64), 0.0)),
            "driver": np.asarray(result.state_mean[:, 0], dtype=np.float64),
            "driver_std": np.sqrt(np.maximum(np.asarray(result.state_variance[:, 0], dtype=np.float64), 0.0)),
            "states": np.asarray(result.state_mean, dtype=np.float64),
            "state_std": np.sqrt(np.maximum(np.asarray(result.state_variance, dtype=np.float64), 0.0)),
            "state_names": tuple(result.state_names),
            "physical_checks": dict(result.physical_checks),
        }
    raise ValueError(f"unknown model {model}")


def _masked_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
    predictive_std: np.ndarray | None = None,
) -> dict[str, float]:
    truth_array = np.asarray(truth, dtype=np.float64)
    estimate_array = np.asarray(estimate, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(truth_array) & np.isfinite(estimate_array)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "nrmse": float("nan"), "low_observed_variance": True, "observed_temporal_sd": float("nan"), "reconstructed_temporal_sd": float("nan"), "temporal_sd_ratio": float("nan"), "pcc": float("nan"), "r2": float("nan"), "mean_bias": float("nan"), "variance_ratio": float("nan"), "standardized_residual_rms": float("nan"), "coverage_95_gaussian_approx": float("nan"), "predictive_valid_point_count": 0, "mean_predictive_std": float("nan"), "median_predictive_std": float("nan"), "interval_width_95_gaussian_approx": float("nan")}
    reliability = trajectory_reliability_metrics(
        truth_array,
        estimate_array,
        predictive_std=predictive_std,
        valid_mask=valid,
    )
    reference = truth_array[valid]
    prediction = estimate_array[valid]
    error = prediction - reference
    low_observed_variance = bool(reliability["low_observed_variance"])
    evaluation_variance = float(np.var(reference))
    rmse = float(np.sqrt(np.mean(error**2)))
    mean_std = float(reliability["posterior_predictive_sd_mean"])
    return {
        "n": n,
        "rmse": rmse,
        "nrmse": float(reliability["trajectory_deviation_nrmse"]),
        "low_observed_variance": low_observed_variance,
        "observed_temporal_sd": float(reliability["observed_temporal_sd"]),
        "reconstructed_temporal_sd": float(reliability["reconstructed_temporal_sd"]),
        "temporal_sd_ratio": float(reliability["temporal_sd_ratio"]),
        "pcc": _safe_corr(reference, prediction) if n >= 2 else float("nan"),
        "r2": float("nan") if low_observed_variance else 1.0 - float(np.mean(error**2)) / evaluation_variance,
        "mean_bias": float(np.mean(error)),
        "variance_ratio": float("nan") if low_observed_variance else float(np.var(prediction) / evaluation_variance),
        "standardized_residual_rms": float(reliability["standardized_residual_rms"]),
        "coverage_95_gaussian_approx": float(reliability["predictive_95_coverage"]),
        "predictive_valid_point_count": int(reliability["predictive_valid_point_count"]),
        "mean_predictive_std": mean_std,
        "median_predictive_std": float(reliability["posterior_predictive_sd_median"]),
        "interval_width_95_gaussian_approx": 2.0 * 1.959963984540054 * mean_std,
    }


def _null_inputs(
    validation: Sequence[PreparedTrial],
    null_type: str,
    shift_steps: int,
) -> list[NullCase]:
    if null_type not in NULL_IDS:
        raise ValueError(null_type)
    by_subject: dict[str, list[PreparedTrial]] = defaultdict(list)
    for item in validation:
        by_subject[item.trial.subject].append(item)
    output: list[NullCase] = []
    subjects = sorted(by_subject)
    for item in validation:
        candidates = by_subject[item.trial.subject]
        trial_index = next(index for index, candidate in enumerate(candidates) if candidate is item)
        if null_type == "pairing":
            if len(candidates) < 2:
                raise RuntimeError(f"pairing null requires at least two validation trials for {item.trial.subject}")
            donor = candidates[(trial_index + 1) % len(candidates)]
            donor_hbo, donor_hbr = donor.hbo.copy(), donor.hbr.copy()
            fnirs = np.column_stack((donor_hbo, donor_hbr))
        elif null_type == "independent":
            donor_subject = subjects[(subjects.index(item.trial.subject) + 1) % len(subjects)]
            donor_candidates = by_subject[donor_subject]
            donor = donor_candidates[trial_index % len(donor_candidates)]
            donor_hbo, donor_hbr = donor.hbo.copy(), donor.hbr.copy()
            fnirs = np.column_stack((donor_hbo, donor_hbr))
        else:
            donor = item
            donor_hbo = _shift_non_circular(item.hbo, shift_steps)
            donor_hbr = _shift_non_circular(item.hbr, shift_steps)
            fnirs = np.column_stack((donor_hbo, donor_hbr))
        values = np.column_stack((item.eeg_driver, fnirs))
        output.append(NullCase(
            receiver=item,
            values=values,
            donor_hbo=donor_hbo,
            donor_hbr=donor_hbr,
            donor_dataset_id=donor.trial.dataset_id,
            donor_subject=donor.trial.subject,
            donor_record_id=donor.trial.record_id,
            donor_event_index=int(donor.trial.event_index),
        ))
    return output


def _shift_non_circular(values: np.ndarray, shift_steps: int) -> np.ndarray:
    result = np.full_like(np.asarray(values, dtype=np.float64), np.nan)
    if shift_steps <= 0:
        return np.asarray(values, dtype=np.float64).copy()
    if shift_steps < len(result):
        result[shift_steps:] = np.asarray(values, dtype=np.float64)[:-shift_steps]
    return result


def _matched_null_metrics(
    paired_target: np.ndarray,
    donor_target: np.ndarray,
    estimate: np.ndarray,
    predictive_std: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score paired and Null targets on their shared finite time support."""

    support = np.isfinite(paired_target) & np.isfinite(donor_target)
    return (
        _masked_metrics(paired_target, estimate, support, predictive_std),
        _masked_metrics(donor_target, estimate, support, predictive_std),
    )


def _fit_rows(bundle: ModelBundle, calibration: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in bundle.t0.items():
        for index, item in enumerate(np.asarray(value).reshape(-1)):
            rows.append({"model": "T0", "parameter": f"{name}_{OBS_NAMES[index]}", "value": float(item), "scope": "fit_subjects"})
    for index, values in enumerate(bundle.t1):
        rows.extend([
            {"model": "T1", "parameter": f"{OBS_NAMES[index]}_ar", "value": float(values[0]), "scope": "fit_subjects"},
            {"model": "T1", "parameter": f"{OBS_NAMES[index]}_process_variance", "value": float(values[1]), "scope": "fit_subjects"},
            {"model": "T1", "parameter": f"{OBS_NAMES[index]}_observation_variance", "value": float(values[2]), "scope": "fit_subjects"},
        ])
    if bundle.t2b is None:
        raise RuntimeError("fit rows require the comparison-model bundle")
    for key, value in fit_to_mapping(bundle.t2b).items():
        rows.append({"model": "T2b", "parameter": str(key), "value": float(value) if isinstance(value, (float, int, np.floating, np.integer)) else str(value), "scope": "fit_subjects"})
    params, _, _ = bundle.t3a
    for key, value in {
        "kappa_per_s": params.free.kappa,
        "tau_s": params.free.tau,
        "alpha": params.fixed.alpha,
        "E0": params.fixed.E0,
        "gamma": params.fixed.gamma,
        "neurovascular_gain": params.fixed.neurovascular_gain,
        "P0_observation_scale": params.fixed.P0,
        "Q0_observation_scale": params.fixed.Q0,
        "driver_decay_per_s": params.fixed.driver_decay_per_s,
        "student_t_df": params.fixed.student_nu,
    }.items():
        rows.append({"model": "T3a", "parameter": key, "value": float(value), "scope": "fixed_or_fit_cohort_gauge"})
    for key, value in calibration.items():
        if isinstance(value, Mapping):
            for child, child_value in value.items():
                rows.append({"model": "calibration", "parameter": f"{key}_{child}", "value": float(child_value), "scope": "fit_subjects"})
        else:
            rows.append({"model": "calibration", "parameter": key, "value": float(value), "scope": "fit_subjects"})
    return rows


def _subject_equal_aggregate(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Average trials within subject, then report subject medians and IQR."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in ("subject", *group_fields))].append(row)
    subject_rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        output: dict[str, Any] = {"subject": key[0], **dict(zip(group_fields, key[1:])), "trial_count": len(values)}
        for metric in metric_fields:
            observed = np.asarray([float(value.get(metric, np.nan)) for value in values], dtype=np.float64)
            finite = observed[np.isfinite(observed)]
            output[metric] = float(np.mean(finite)) if len(finite) else float("nan")
        subject_rows.append(output)

    summary_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        summary_groups[tuple(row[field] for field in group_fields)].append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, values in sorted(summary_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        output = {**dict(zip(group_fields, key)), "subjects": len(values), "replication_unit": "subject"}
        for metric in metric_fields:
            observed = np.asarray([float(value.get(metric, np.nan)) for value in values], dtype=np.float64)
            finite = observed[np.isfinite(observed)]
            output[f"{metric}_median"] = float(np.median(finite)) if len(finite) else float("nan")
            output[f"{metric}_q25"] = float(np.quantile(finite, 0.25)) if len(finite) else float("nan")
            output[f"{metric}_q75"] = float(np.quantile(finite, 0.75)) if len(finite) else float("nan")
        summary_rows.append(output)
    return subject_rows, summary_rows


def _plot_reconstruction(
    example: PreparedTrial,
    predictions: Mapping[str, Mapping[str, Mapping[str, np.ndarray | None]]],
    center: np.ndarray,
    run_dir: Path,
    fs_hz: float,
    dpi: int,
) -> None:
    time = np.arange(len(example.hbo), dtype=np.float64) / fs_hz - 5.0
    observed = np.column_stack((example.eeg_driver, example.hbo, example.hbr))
    names = ("EEG 10 Hz 功率代理", "HbO 标准化坐标", "HbR 标准化坐标")
    modes = ("center_masked_eeg", "center_masked_fnirs", "center_masked_fnirs")
    colors = {"T0": "#999999", "T1": "#56B4E9", "T2b": "#009E73", "T3a": "#D55E00"}
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey="row")
    for row, (name, column, mode) in enumerate(zip(names, range(3), modes)):
        masked_input = _mode_input(example, mode, center)[:, column]
        for col, model in enumerate(MODEL_IDS):
            axis = axes[row, col]
            axis.plot(time, observed[:, column], color="#222222", linewidth=1.0)
            axis.plot(time, masked_input, color="#777777", linestyle=":", linewidth=0.9)
            result = predictions[model][mode]
            estimate = np.asarray(result["estimate"], dtype=np.float64)[:, column]
            predictive_std = np.asarray(result["predictive_std"], dtype=np.float64)[:, column]
            axis.fill_between(
                time,
                estimate - 1.96 * predictive_std,
                estimate + 1.96 * predictive_std,
                color=colors[model],
                alpha=0.10,
                linewidth=0.0,
            )
            axis.plot(time, estimate, color=colors[model], linewidth=1.1)
            if np.any(center):
                axis.axvspan(time[np.flatnonzero(center)[0]], time[np.flatnonzero(center)[-1]], color="#E69F00", alpha=0.14)
            if row == 0:
                axis.set_title(f"{MODEL_LABELS[model]}：遮挡重建")
            if col == 0:
                axis.set_ylabel(name)
            if row == len(names) - 1:
                axis.set_xlabel("事件相对时间（秒）")
            axis.grid(alpha=0.2)
    legend_handles = [
        Line2D([0], [0], color="#222222", linewidth=1.2, label="实测观测（无干净真值）"),
        Line2D([0], [0], color="#777777", linestyle=":", linewidth=1.1, label="本行目标模态遮挡后输入"),
        *[Line2D([0], [0], color=colors[model], linewidth=1.4, label=f"{MODEL_LABELS[model]} 重建") for model in MODEL_IDS],
        Patch(facecolor="#777777", alpha=0.16, label="方差匹配高斯 95% 近似区间（非覆盖保证）"),
        Patch(facecolor="#E69F00", alpha=0.20, label="中心 4 秒遮挡段"),
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False)
    fig.suptitle("真实验证窗口：按行遮挡的 EEG/fNIRS 中心段重建", y=0.992)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "典型窗口_EEG_fNIRS中心遮挡重建.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_null(null_summary: Sequence[Mapping[str, Any]], run_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, layout="constrained")
    x = np.arange(len(NULL_IDS), dtype=float)
    width = 0.18
    colors = {"T0": "#777777", "T1": "#56B4E9", "T2b": "#009E73", "T3a": "#D55E00"}
    null_labels = ["跨受试者", "同受试者错配", "+10 秒非循环平移"]
    lookup = {(row["model"], row["null_type"], row["target"]): row for row in null_summary}
    for target_index, target in enumerate(("HbO", "HbR")):
        delta_axis = axes[0, target_index]
        leak_axis = axes[1, target_index]
        for offset, model in enumerate(MODEL_IDS):
            positions = x + (offset - 1.5) * width
            delta = [float(lookup.get((model, null_type, target), {}).get("delta_nrmse_null_minus_paired_median", np.nan)) for null_type in NULL_IDS]
            delta_axis.bar(positions, delta, width=width, color=colors[model], label=MODEL_LABELS[model])
            q25 = np.asarray([float(lookup.get((model, null_type, target), {}).get("delta_nrmse_null_minus_paired_q25", np.nan)) for null_type in NULL_IDS])
            q75 = np.asarray([float(lookup.get((model, null_type, target), {}).get("delta_nrmse_null_minus_paired_q75", np.nan)) for null_type in NULL_IDS])
            values = np.asarray(delta, dtype=np.float64)
            finite = np.isfinite(values) & np.isfinite(q25) & np.isfinite(q75)
            delta_axis.errorbar(positions[finite], values[finite], yerr=np.vstack((values[finite] - q25[finite], q75[finite] - values[finite])), fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
            for position, value in zip(positions, values):
                if np.isfinite(value) and abs(value) < 1e-12:
                    delta_axis.text(position, 0.0, "0", ha="center", va="bottom", fontsize=6)
        for offset, model in enumerate(("T2b", "T3a")):
            positions = x + (offset - 0.5) * 0.28
            leak = [float(lookup.get((model, null_type, target), {}).get("abs_driver_donor_pcc_median", np.nan)) for null_type in NULL_IDS]
            leak_axis.bar(positions, leak, width=0.28, color=colors[model], label=MODEL_LABELS[model])
            q25 = np.asarray([float(lookup.get((model, null_type, target), {}).get("abs_driver_donor_pcc_q25", np.nan)) for null_type in NULL_IDS])
            q75 = np.asarray([float(lookup.get((model, null_type, target), {}).get("abs_driver_donor_pcc_q75", np.nan)) for null_type in NULL_IDS])
            values = np.asarray(leak, dtype=np.float64)
            finite = np.isfinite(values) & np.isfinite(q25) & np.isfinite(q75)
            leak_axis.errorbar(positions[finite], values[finite], yerr=np.vstack((values[finite] - q25[finite], q75[finite] - values[finite])), fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
        delta_axis.axhline(0.0, color="black", linewidth=0.9)
        leak_axis.axhline(0.35, color="black", linestyle="--", linewidth=0.9, label="合成参考 0.35（非实测门）")
        delta_axis.set_title(f"{target}：受试者内 NRMSE 差值的跨受试者中位数")
        leak_axis.set_title(f"{target}：Null 联合状态对供体信号的相关")
        delta_axis.set_ylabel("Null NRMSE − 真实配对 NRMSE")
        leak_axis.set_ylabel("|r 与供体信号相关|")
        leak_axis.set_xticks(x, null_labels)
        delta_axis.grid(axis="y", alpha=0.2)
        leak_axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("真实验证集 Null：配对特异性与共享状态泄漏（n=5，中位数·四分位距）")
    fig.supxlabel("差值 > 0 表示错配供体更难预测；差值 < 0 不能单独解释为生理泄漏。", fontsize=8)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "Null配对特异性与共享状态泄漏.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_primary_metrics(subject_rows: Sequence[Mapping[str, Any]], run_dir: Path, dpi: int) -> None:
    primary = [row for row in subject_rows if row["evaluation_kind"] == "primary_center_masked"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False, layout="constrained")
    colors = {"T0": "#777777", "T1": "#56B4E9", "T2b": "#009E73", "T3a": "#D55E00"}
    for axis, target in zip(axes, OBS_NAMES):
        for index, model in enumerate(MODEL_IDS):
            values = np.asarray([float(row["nrmse"]) for row in primary if row["target"] == target and row["model"] == model], dtype=np.float64)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.asarray([0.0])
            axis.scatter(index + offsets, values, s=22, facecolor="white", edgecolor=colors[model], linewidth=1.0, zorder=2)
            median = float(np.median(values))
            q25, q75 = np.quantile(values, (0.25, 0.75))
            axis.errorbar(index, median, yerr=[[median - q25], [q75 - median]], fmt="o", color=colors[model], capsize=4, linewidth=2.0, zorder=3)
        axis.set_xticks(range(len(MODEL_IDS)), [MODEL_LABELS[model].replace(" ", "\n", 1) for model in MODEL_IDS], rotation=12)
        axis.set_title(f"{target} 中心遮挡重建")
        axis.set_ylabel("NRMSE")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("四模型中心遮挡重建：同段标准差归一化、各目标独立纵轴（n=5，无干净真值）")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "四模型中心遮挡重建指标.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_shared_driver(
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    run_dir: Path,
    fs_hz: float,
    dpi: int,
) -> None:
    time = np.arange(len(np.asarray(predictions["T2b"]["joint"]["driver"]))) / fs_hz - 5.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    colors = {"T2b": "#009E73", "T3a": "#D55E00"}
    for axis, model in zip(axes, ("T2b", "T3a")):
        driver = np.asarray(predictions[model]["joint"]["driver"], dtype=np.float64)
        std = np.asarray(predictions[model]["joint"]["driver_std"], dtype=np.float64)
        axis.fill_between(time, driver - 1.96 * std, driver + 1.96 * std, color=colors[model], alpha=0.18, label="方差匹配高斯 95% 近似区间（非覆盖保证）")
        axis.plot(time, driver, color=colors[model], linewidth=1.3, label="共享神经驱动 r")
        axis.set_title(MODEL_LABELS[model])
        axis.set_ylabel("模型内部状态坐标")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[-1].set_xlabel("事件相对时间（秒）")
    fig.suptitle("joint 同点描述性后验：使用完整 EEG+HbO+HbR，非遮挡验证/教师证据（模型量纲不可直接比）")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "共享神经驱动与状态不确定性.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _prepare_measured_series(
    config: Mapping[str, Any],
) -> tuple[
    list[Trial],
    list[PreparedTrial],
    list[PreparedTrial],
    Mapping[str, Any],
    EEGAdapter,
    tuple[str, ...],
    np.ndarray,
]:
    """Load the registered public split and apply the one fit-cohort gauge."""

    # Critical boundary: this is the first call that can read the physiology cache.
    grouped, contracts = _load_trials(config)
    condition = config["data"]["conditions"][0]
    condition_id = str(condition["condition_id"])
    loaded = grouped.get(condition_id, {})
    fit_subjects = list(map(str, condition["fit_subjects"]))
    validation_subjects = list(map(str, condition["validation_subjects"]))
    if set(loaded) != set(fit_subjects + validation_subjects):
        raise RuntimeError("loader returned an unexpected subject set")
    fit_trials = [trial for subject in fit_subjects for trial in loaded[subject]]
    validation_trials = [trial for subject in validation_subjects for trial in loaded[subject]]
    for trial in fit_trials + validation_trials:
        _require_full_support(trial)
    reference_trial = fit_trials[0]
    for trial in fit_trials[1:] + validation_trials:
        if (
            trial.eeg_channel_names != reference_trial.eeg_channel_names
            or trial.fnirs_channel_names != reference_trial.fnirs_channel_names
            or trial.fnirs_roles != reference_trial.fnirs_roles
        ):
            raise RuntimeError("cross-trial channel identity/order mismatch")
    hbo_indices, hbo_names, _ = _select_active_hbo(
        fit_trials,
        baseline_duration_s=float(config["data"]["baseline_duration_s"]),
        task_duration_s=float(config["data"]["task_duration_s"]),
        count=int(config["analysis"]["active_hbo_channels"]),
    )
    hbr_indices = _paired_hbr_indices(fit_trials[0], hbo_indices)
    eeg_indices = np.asarray([
        index for index, name in enumerate(fit_trials[0].eeg_channel_names)
        if not any(token in name.upper() for token in ("EOG", "ECG", "EMG"))
    ], dtype=int)
    if not len(eeg_indices):
        raise RuntimeError("no scalp EEG channels remain after EOG/ECG/EMG exclusion")
    adapter, fit_drivers = _fit_eeg_adapter(fit_trials, eeg_indices)
    validation_drivers = [_apply_eeg_adapter(trial, adapter) for trial in validation_trials]
    fit_hbo = [np.mean(trial.fnirs[:, hbo_indices], axis=1, dtype=np.float64) for trial in fit_trials]
    fit_hbr = [np.mean(trial.fnirs[:, hbr_indices], axis=1, dtype=np.float64) for trial in fit_trials]
    fit_series = [PreparedTrial(trial, driver, hbo, hbr) for trial, driver, hbo, hbr in zip(fit_trials, fit_drivers, fit_hbo, fit_hbr)]
    val_hbo = [np.mean(trial.fnirs[:, hbo_indices], axis=1, dtype=np.float64) for trial in validation_trials]
    val_hbr = [np.mean(trial.fnirs[:, hbr_indices], axis=1, dtype=np.float64) for trial in validation_trials]
    validation = [PreparedTrial(trial, driver, hbo, hbr) for trial, driver, hbo, hbr in zip(validation_trials, validation_drivers, val_hbo, val_hbr)]
    selected_keys = [
        (item.trial.dataset_id, item.trial.subject, item.trial.record_id, int(item.trial.event_index))
        for item in (*fit_series, *validation)
    ]
    if len(selected_keys) != len(set(selected_keys)):
        raise RuntimeError("selected trial identities are not unique")
    return fit_trials, fit_series, validation, contracts, adapter, tuple(hbo_names), hbr_indices


PARAMETER_NAMES = ("beta", "kappa", "tau", "gamma", "alpha", "E0")


def _parameter_values(parameters: BalloonParameters) -> dict[str, float]:
    return {
        "beta": float(parameters.fixed.neurovascular_gain),
        "kappa": float(parameters.free.kappa),
        "tau": float(parameters.free.tau),
        "gamma": float(parameters.fixed.gamma),
        "alpha": float(parameters.fixed.alpha),
        "E0": float(parameters.fixed.E0),
    }


def _replace_parameter_values(
    parameters: BalloonParameters,
    values: Mapping[str, float],
) -> BalloonParameters:
    fixed = replace(
        parameters.fixed,
        neurovascular_gain=float(values.get("beta", parameters.fixed.neurovascular_gain)),
        gamma=float(values.get("gamma", parameters.fixed.gamma)),
        alpha=float(values.get("alpha", parameters.fixed.alpha)),
        E0=float(values.get("E0", parameters.fixed.E0)),
    )
    free = replace(
        parameters.free,
        kappa=float(values.get("kappa", parameters.free.kappa)),
        tau=float(values.get("tau", parameters.free.tau)),
    )
    result = BalloonParameters(fixed=fixed, free=free)
    result.validate()
    return result


def _to_optimizer_coordinate(name: str, value: float) -> float:
    if name == "E0":
        return math.log(float(value) / (1.0 - float(value)))
    return math.log(float(value))


def _from_optimizer_coordinate(name: str, value: float) -> float:
    if name == "E0":
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-float(value)))
        exp_value = math.exp(float(value))
        return exp_value / (1.0 + exp_value)
    return math.exp(float(value))


def _finite_hessian_bounded(
    optimum: np.ndarray,
    objective: Any,
    bounds: Sequence[tuple[float, float]],
    relative_step: float,
) -> np.ndarray:
    dimension = len(optimum)
    hessian = np.full((dimension, dimension), np.nan, dtype=np.float64)
    if dimension == 0:
        return np.empty((0, 0), dtype=np.float64)
    steps = np.asarray([
        min(
            float(relative_step) * max(1.0, abs(float(optimum[index]))),
            0.25 * (float(optimum[index]) - float(bounds[index][0])),
            0.25 * (float(bounds[index][1]) - float(optimum[index])),
        )
        for index in range(dimension)
    ])
    if np.any(steps <= 1e-7):
        return hessian
    base = float(objective(optimum))
    for index in range(dimension):
        delta = np.zeros(dimension, dtype=np.float64)
        delta[index] = steps[index]
        plus = float(objective(optimum + delta))
        minus = float(objective(optimum - delta))
        hessian[index, index] = (plus - 2.0 * base + minus) / steps[index] ** 2
        for other in range(index):
            delta_other = np.zeros(dimension, dtype=np.float64)
            delta_other[other] = steps[other]
            value = (
                float(objective(optimum + delta + delta_other))
                - float(objective(optimum + delta - delta_other))
                - float(objective(optimum - delta + delta_other))
                + float(objective(optimum - delta - delta_other))
            ) / (4.0 * steps[index] * steps[other])
            hessian[index, other] = value
            hessian[other, index] = value
    return (hessian + hessian.T) * 0.5


def _gaussian_negative_log_score(
    truth: np.ndarray,
    estimate: np.ndarray,
    predictive_std: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(truth)
        & np.isfinite(estimate)
        & np.isfinite(predictive_std)
        & (np.asarray(predictive_std) > 0.0)
    )
    if not np.any(valid):
        return float("nan"), 0
    residual = np.asarray(truth, dtype=np.float64)[valid] - np.asarray(estimate, dtype=np.float64)[valid]
    std = np.asarray(predictive_std, dtype=np.float64)[valid]
    score = 0.5 * (math.log(2.0 * math.pi) + 2.0 * np.log(std) + np.square(residual / std))
    return float(np.mean(score)), int(np.count_nonzero(valid))


def _stage_prior_penalty(
    vector: np.ndarray,
    active: Sequence[str],
    specs: Mapping[str, Mapping[str, Any]],
) -> float:
    return float(sum(
        0.5 * (
            (_from_optimizer_coordinate(name, float(value)) - float(specs[name]["prior_mean"]))
            / float(specs[name]["prior_sd"])
        ) ** 2
        for name, value in zip(active, vector)
    ))


def _fit_subject_stage(task: Mapping[str, Any]) -> dict[str, Any]:
    """Fit one subject/stage; every likelihood term starts a fresh smoother."""

    subject = str(task["subject"])
    split = str(task["split"])
    fit_scope = str(task["fit_scope"])
    stage = dict(task["stage"])
    stage_id = str(stage["id"])
    active = tuple(str(value) for value in stage.get("free", ()))
    specs = task["parameter_specs"]
    base_parameters: BalloonParameters = task["base_parameters"]
    observation_spec: BalloonObservationSpec = task["observation_spec"]
    balloon_config: BalloonConfig = task["balloon_config"]
    fit_config = task["fit_config"]
    train_trials = tuple(task["train_trials"])
    heldout_trials = tuple(task.get("heldout_trials", ()))
    center = np.asarray(task["center_mask"], dtype=bool)
    base_values = _parameter_values(base_parameters)
    initial_values = {**base_values, **{str(key): float(value) for key, value in task.get("initial_values", {}).items()}}
    transformed_bounds = tuple(
        (
            _to_optimizer_coordinate(name, float(specs[name]["bounds"][0])),
            _to_optimizer_coordinate(name, float(specs[name]["bounds"][1])),
        )
        for name in active
    )
    start = np.asarray([
        np.clip(_to_optimizer_coordinate(name, initial_values[name]), *transformed_bounds[index])
        for index, name in enumerate(active)
    ], dtype=np.float64)
    likelihood_cache: dict[tuple[float, ...], float] = {}

    def vector_values(vector: Sequence[float]) -> dict[str, float]:
        return {
            name: _from_optimizer_coordinate(name, float(value))
            for name, value in zip(active, vector)
        }

    def likelihood_nll(vector: Sequence[float]) -> float:
        key = tuple(round(float(value), 12) for value in vector)
        cached = likelihood_cache.get(key)
        if cached is not None:
            return cached
        try:
            parameters = _replace_parameter_values(base_parameters, vector_values(vector))
            value = -sum(
                float(smooth_balloon(
                    np.asarray(observations, dtype=np.float64),
                    parameters=parameters,
                    observation_spec=observation_spec,
                    config=balloon_config,
                ).predictive_log_likelihood)
                for _, observations in train_trials
            )
            if not np.isfinite(value):
                value = 1.0e12
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            value = 1.0e12
        likelihood_cache[key] = float(value)
        return float(value)

    def objective(vector: Sequence[float]) -> float:
        array = np.asarray(vector, dtype=np.float64)
        return likelihood_nll(array) + _stage_prior_penalty(array, active, specs)

    start_records: list[dict[str, Any]] = []
    if active:
        candidates = [start]
        if int(fit_config["optimizer_starts"]) > 1:
            alternate = np.asarray([
                np.clip(
                    value + (0.12 if index % 2 == 0 else -0.12) * (upper - lower),
                    lower,
                    upper,
                )
                for index, (value, (lower, upper)) in enumerate(zip(start, transformed_bounds))
            ])
            candidates.append(alternate)
        best: Any | None = None
        for candidate in candidates:
            result = minimize(
                objective,
                candidate,
                method="L-BFGS-B",
                bounds=transformed_bounds,
                options={
                    "maxiter": int(fit_config["optimizer_max_iterations"]),
                    "ftol": float(fit_config["optimizer_ftol"]),
                    "maxls": 12,
                },
            )
            raw_estimate = vector_values(result.x)
            start_records.append({
                "start": vector_values(candidate),
                "estimate": raw_estimate,
                "objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
            })
            if best is None or float(result.fun) < float(best.fun):
                best = result
        if best is None or not np.all(np.isfinite(best.x)):
            raise RuntimeError(f"{subject} {stage_id}: no finite optimizer result")
        optimum = np.asarray(best.x, dtype=np.float64)
        optimizer_success = bool(best.success) and float(best.fun) < 1.0e11
        optimizer_message = str(best.message)
    else:
        optimum = np.empty(0, dtype=np.float64)
        optimizer_success = True
        optimizer_message = "fixed baseline"
        start_records.append({"start": {}, "estimate": {}, "objective": objective(optimum), "success": True, "message": optimizer_message})

    estimates = {**base_values, **vector_values(optimum)}
    fitted_parameters = _replace_parameter_values(base_parameters, estimates)
    objective_value = float(objective(optimum))
    likelihood_value = float(likelihood_nll(optimum))
    n_observations = int(sum(np.count_nonzero(np.isfinite(values)) for _, values in train_trials))
    raw_boundary: dict[str, bool] = {}
    start_spread: dict[str, float] = {}
    for name in active:
        lower, upper = map(float, specs[name]["bounds"])
        span = upper - lower
        raw_boundary[name] = min(estimates[name] - lower, upper - estimates[name]) <= float(fit_config["boundary_fraction"]) * span
        start_estimates = [float(record["estimate"][name]) for record in start_records]
        start_spread[name] = (max(start_estimates) - min(start_estimates)) / span

    posterior_hessian = np.full((len(active), len(active)), np.nan, dtype=np.float64)
    data_hessian = np.full((len(active), len(active)), np.nan, dtype=np.float64)
    covariance = np.full((len(active), len(active)), np.nan, dtype=np.float64)
    posterior_condition = float("nan")
    data_condition = float("nan")
    max_abs_correlation = float("nan")
    positive_posterior = not active
    positive_data = not active
    if active and not any(raw_boundary.values()):
        posterior_hessian = _finite_hessian_bounded(
            optimum, objective, transformed_bounds, float(fit_config["hessian_step"])
        )
        prior_hessian = _finite_hessian_bounded(
            optimum,
            lambda vector: _stage_prior_penalty(np.asarray(vector), active, specs),
            transformed_bounds,
            float(fit_config["hessian_step"]),
        )
        data_hessian = posterior_hessian - prior_hessian
        try:
            posterior_eigen = np.linalg.eigvalsh(posterior_hessian)
            data_eigen = np.linalg.eigvalsh(data_hessian)
            positive_posterior = bool(np.all(np.isfinite(posterior_eigen)) and np.min(posterior_eigen) > 1e-7 * max(1.0, float(np.max(np.abs(posterior_eigen)))))
            positive_data = bool(np.all(np.isfinite(data_eigen)) and np.min(data_eigen) > 1e-7 * max(1.0, float(np.max(np.abs(data_eigen)))))
            posterior_condition = float(np.max(posterior_eigen) / np.min(posterior_eigen)) if positive_posterior else float("inf")
            data_condition = float(np.max(data_eigen) / np.min(data_eigen)) if positive_data else float("inf")
            covariance = np.linalg.pinv(posterior_hessian) if positive_posterior and posterior_condition < 1.0e10 else np.full_like(posterior_hessian, np.nan)
            if np.all(np.isfinite(covariance)) and len(active) > 1:
                scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
                correlation = covariance / np.outer(scale, scale)
                max_abs_correlation = float(np.nanmax(np.abs(correlation - np.eye(len(active)))))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            covariance = np.full((len(active), len(active)), np.nan, dtype=np.float64)
            positive_posterior = False
            positive_data = False

    parameter_rows: list[dict[str, Any]] = []
    active_statuses: list[str] = []
    for name in PARAMETER_NAMES:
        spec = specs[name]
        estimate = float(estimates[name])
        is_free = name in active
        posterior_sd = float("nan")
        lower_95 = float("nan")
        upper_95 = float("nan")
        contraction = float("nan")
        boundary_status = "FIXED"
        identifiability_status = "FIXED"
        spread = float("nan")
        if is_free:
            index = active.index(name)
            spread = float(start_spread[name])
            boundary_status = "BOUNDARY" if raw_boundary[name] else "INTERIOR"
            if covariance.shape == (len(active), len(active)) and np.all(np.isfinite(covariance)):
                transformed_sd = math.sqrt(max(float(covariance[index, index]), 0.0))
                derivative = estimate * (1.0 - estimate) if name == "E0" else estimate
                posterior_sd = derivative * transformed_sd
                lower_95 = _from_optimizer_coordinate(name, float(optimum[index]) - 1.959963984540054 * transformed_sd)
                upper_95 = _from_optimizer_coordinate(name, float(optimum[index]) + 1.959963984540054 * transformed_sd)
                hard_lower, hard_upper = map(float, spec["bounds"])
                lower_95 = max(hard_lower, lower_95)
                upper_95 = min(hard_upper, upper_95)
                contraction = 1.0 - posterior_sd / float(spec["prior_sd"])
            if raw_boundary[name]:
                identifiability_status = "BOUNDARY"
            elif not optimizer_success or not positive_posterior or not positive_data or data_condition >= 1.0e8:
                identifiability_status = "UNIDENTIFIABLE"
            elif np.isfinite(max_abs_correlation) and max_abs_correlation >= 0.95:
                identifiability_status = "COMPENSATORY"
            elif not np.isfinite(contraction) or contraction <= 0.01:
                identifiability_status = "PRIOR_DOMINATED"
            else:
                identifiability_status = "IDENTIFIABLE"
            active_statuses.append(identifiability_status)
        parameter_rows.append({
            "subject": subject,
            "split": split,
            "fit_scope": fit_scope,
            "stage": stage_id,
            "stage_label_zh": str(stage["label_zh"]),
            "parameter": name,
            "parameter_label_zh": str(spec["label_zh"]),
            "unit_zh": str(spec["unit_zh"]),
            "is_free": is_free,
            "estimate": estimate,
            "posterior_sd_laplace": posterior_sd,
            "lower_95_laplace": lower_95,
            "upper_95_laplace": upper_95,
            "prior_mean": float(spec["prior_mean"]),
            "prior_sd": float(spec["prior_sd"]),
            "hard_lower": float(spec["bounds"][0]),
            "hard_upper": float(spec["bounds"][1]),
            "posterior_contraction": contraction,
            "boundary_status": boundary_status,
            "identifiability_status": identifiability_status,
            "multistart_spread_fraction": spread,
            "optimizer_success": optimizer_success,
        })

    if not active:
        stage_status = "FIXED"
    elif "BOUNDARY" in active_statuses:
        stage_status = "BOUNDARY"
    elif "UNIDENTIFIABLE" in active_statuses:
        stage_status = "UNIDENTIFIABLE"
    elif "COMPENSATORY" in active_statuses:
        stage_status = "COMPENSATORY"
    elif "PRIOR_DOMINATED" in active_statuses:
        stage_status = "PRIOR_DOMINATED"
    else:
        stage_status = "IDENTIFIABLE"

    trial_rows: list[dict[str, Any]] = []
    physical_failures = 0
    for event_index, observations in train_trials:
        result = smooth_balloon(
            np.asarray(observations, dtype=np.float64),
            parameters=fitted_parameters,
            observation_spec=observation_spec,
            config=balloon_config,
        )
        failures = sum(
            isinstance(value, (bool, np.bool_)) and not bool(value)
            for value in result.physical_checks.values()
        )
        physical_failures += int(failures)
        trial_rows.append({
            "subject": subject,
            "split": split,
            "fit_scope": fit_scope,
            "stage": stage_id,
            "event_index": int(event_index),
            "trial_role": "fit",
            "predictive_log_likelihood": float(result.predictive_log_likelihood),
            "finite_observations": int(np.count_nonzero(np.isfinite(observations))),
            "physical_boolean_failures": int(failures),
        })

    metric_rows: list[dict[str, Any]] = []
    for event_index, observations in heldout_trials:
        truth = np.asarray(observations, dtype=np.float64)
        for mode, columns in (("center_masked_eeg", (0,)), ("center_masked_fnirs", (1, 2))):
            values = truth.copy()
            values[np.ix_(center, np.asarray(columns, dtype=int))] = np.nan
            result = smooth_balloon(
                values,
                parameters=fitted_parameters,
                observation_spec=observation_spec,
                config=balloon_config,
                observation_mask=np.isfinite(values),
            )
            predictive_std = np.sqrt(np.maximum(result.total_variance, 0.0))
            for column in columns:
                metrics = _masked_metrics(
                    truth[:, column], result.observation_mean[:, column], center, predictive_std[:, column]
                )
                negative_log_score, score_n = _gaussian_negative_log_score(
                    truth[:, column], result.observation_mean[:, column], predictive_std[:, column], center
                )
                metric_rows.append({
                    "subject": subject,
                    "split": split,
                    "fit_scope": fit_scope,
                    "stage": stage_id,
                    "stage_label_zh": str(stage["label_zh"]),
                    "event_index": int(event_index),
                    "mode": mode,
                    "target": OBS_NAMES[column],
                    "gaussian_negative_log_score": negative_log_score,
                    "score_n": score_n,
                    **metrics,
                })

    geometry_row = {
        "subject": subject,
        "split": split,
        "fit_scope": fit_scope,
        "stage": stage_id,
        "stage_status": stage_status,
        "free_parameter_count": len(active),
        "posterior_hessian_condition": posterior_condition,
        "likelihood_hessian_condition": data_condition,
        "max_abs_posterior_parameter_correlation": max_abs_correlation,
        "objective": objective_value,
        "likelihood_nll": likelihood_value,
        "objective_per_observation": objective_value / max(n_observations, 1),
        "likelihood_nll_per_observation": likelihood_value / max(n_observations, 1),
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "physical_boolean_failures": physical_failures,
        "fit_trial_count": len(train_trials),
        "heldout_trial_count": len(heldout_trials),
    }
    return {
        "subject": subject,
        "stage": stage_id,
        "values": estimates,
        "parameter_rows": parameter_rows,
        "metric_rows": metric_rows,
        "trial_rows": trial_rows,
        "geometry_row": geometry_row,
        "start_records": start_records,
    }


def _trial_observation(item: PreparedTrial) -> np.ndarray:
    return np.column_stack((item.eeg_driver, item.hbo, item.hbr)).astype(np.float64)


def _split_stage_trials(
    series: Sequence[PreparedTrial],
    subjects: Sequence[str],
    heldout_positions: Sequence[int],
) -> tuple[dict[str, tuple[tuple[int, np.ndarray], ...]], dict[str, tuple[tuple[int, np.ndarray], ...]]]:
    grouped: dict[str, list[PreparedTrial]] = defaultdict(list)
    for item in series:
        grouped[item.trial.subject].append(item)
    train: dict[str, tuple[tuple[int, np.ndarray], ...]] = {}
    heldout: dict[str, tuple[tuple[int, np.ndarray], ...]] = {}
    heldout_set = set(map(int, heldout_positions))
    for subject in subjects:
        ordered = sorted(grouped[subject], key=lambda item: int(item.trial.event_index))
        if heldout_set and max(heldout_set) >= len(ordered):
            raise RuntimeError(f"{subject}: heldout trial position exceeds available trials")
        train[subject] = tuple(
            (int(item.trial.event_index), _trial_observation(item))
            for index, item in enumerate(ordered)
            if index not in heldout_set
        )
        heldout[subject] = tuple(
            (int(item.trial.event_index), _trial_observation(item))
            for index, item in enumerate(ordered)
            if index in heldout_set
        )
    return train, heldout


def _run_stage_tasks(tasks: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks))) as executor:
        futures = {executor.submit(_fit_subject_stage, task): (task["subject"], task["stage"]["id"]) for task in tasks}
        for future in as_completed(futures):
            subject, stage_id = futures[future]
            result = future.result()
            results.append(result)
            print(f"完成 {stage_id}: {subject}", flush=True)
    return sorted(results, key=lambda result: str(result["subject"]))


def _stage_summaries(
    metric_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    parameter_rows: Sequence[Mapping[str, Any]],
    stages: Sequence[Mapping[str, Any]],
    fit_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        if row["target"] in {"HbO", "HbR"}:
            grouped[(str(row["subject"]), str(row["stage"]))].append(row)
    subject_rows: list[dict[str, Any]] = []
    for (subject, stage_id), rows in sorted(grouped.items()):
        subject_rows.append({
            "subject": subject,
            "stage": stage_id,
            "heldout_trial_count": len({int(row["event_index"]) for row in rows}),
            "fNIRS_gaussian_negative_log_score": float(np.nanmean([float(row["gaussian_negative_log_score"]) for row in rows])),
            "fNIRS_nrmse": float(np.nanmean([float(row["nrmse"]) for row in rows])),
            "fNIRS_pcc": float(np.nanmean([float(row["pcc"]) for row in rows])),
        })
    subject_lookup = {(row["subject"], row["stage"]): row for row in subject_rows}
    baseline = {row["subject"]: float(row["fNIRS_gaussian_negative_log_score"]) for row in subject_rows if row["stage"] == "M0_fixed"}
    geometry_lookup = {(str(row["subject"]), str(row["stage"])): row for row in geometry_rows}
    stage_summary: list[dict[str, Any]] = []
    for stage in stages:
        stage_id = str(stage["id"])
        rows = [row for row in subject_rows if row["stage"] == stage_id]
        scores = np.asarray([float(row["fNIRS_gaussian_negative_log_score"]) for row in rows], dtype=np.float64)
        nrmses = np.asarray([float(row["fNIRS_nrmse"]) for row in rows], dtype=np.float64)
        deltas = np.asarray([
            float(row["fNIRS_gaussian_negative_log_score"]) - baseline[str(row["subject"])]
            for row in rows
        ], dtype=np.float64)
        statuses = [str(geometry_lookup[(str(row["subject"]), stage_id)]["stage_status"]) for row in rows]
        free_rows = [row for row in parameter_rows if row["stage"] == stage_id and bool(row["is_free"])]
        boundary_fraction = float(np.mean([row["boundary_status"] == "BOUNDARY" for row in free_rows])) if free_rows else 0.0
        unidentifiable_fraction = float(np.mean([
            status in {"BOUNDARY", "UNIDENTIFIABLE", "COMPENSATORY", "PRIOR_DOMINATED"}
            for status in statuses
        ])) if statuses else 1.0
        physical_failures = int(sum(int(geometry_lookup[(str(row["subject"]), stage_id)]["physical_boolean_failures"]) for row in rows))
        median_delta = float(np.nanmedian(deltas))
        eligible = bool(
            stage_id == "M0_fixed"
            or (
                bool(stage.get("recommendation_eligible", True))
                and median_delta < 0.0
                and boundary_fraction <= float(fit_config["max_stage_boundary_fraction"])
                and unidentifiable_fraction <= float(fit_config["max_stage_unidentifiable_fraction"])
                and physical_failures == 0
            )
        )
        stage_summary.append({
            "stage": stage_id,
            "stage_label_zh": str(stage["label_zh"]),
            "free_parameters": "+".join(map(str, stage.get("free", ()))) or "none",
            "free_parameter_count": len(tuple(stage.get("free", ()))),
            "subject_count": len(rows),
            "fNIRS_gaussian_negative_log_score_median": float(np.nanmedian(scores)),
            "fNIRS_gaussian_negative_log_score_q25": float(np.nanquantile(scores, 0.25)),
            "fNIRS_gaussian_negative_log_score_q75": float(np.nanquantile(scores, 0.75)),
            "delta_score_vs_M0_median": median_delta,
            "fNIRS_nrmse_median": float(np.nanmedian(nrmses)),
            "boundary_fraction_free_parameter_rows": boundary_fraction,
            "non_identifiable_subject_fraction": unidentifiable_fraction,
            "physical_boolean_failures": physical_failures,
            "recommendation_eligible_by_contract": bool(stage.get("recommendation_eligible", True)),
            "passes_exploratory_selection_rule": eligible,
        })
    finite = [row for row in stage_summary if np.isfinite(float(row["fNIRS_gaussian_negative_log_score_median"]))]
    best_predictive = min(finite, key=lambda row: (float(row["fNIRS_gaussian_negative_log_score_median"]), int(row["free_parameter_count"])))["stage"]
    eligible = [row for row in finite if bool(row["passes_exploratory_selection_rule"])]
    recommended = min(eligible, key=lambda row: (float(row["fNIRS_gaussian_negative_log_score_median"]), int(row["free_parameter_count"])))["stage"]
    return subject_rows, stage_summary, str(recommended), str(best_predictive)


def _population_parameters(
    base: BalloonParameters,
    stage: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> BalloonParameters:
    values = _parameter_values(base)
    for name in stage.get("free", ()):
        observed = [float(result["values"][name]) for result in results]
        values[str(name)] = float(np.median(observed))
    return _replace_parameter_values(base, values)


def _evaluate_frozen_nulls(
    models: Mapping[str, BalloonParameters],
    validation: Sequence[PreparedTrial],
    observation_spec: BalloonObservationSpec,
    balloon_config: BalloonConfig,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shift_steps = int(round(float(config["analysis"]["time_shift_s"]) * float(config["analysis"]["sampling_hz"])))
    for model_id, parameters in models.items():
        eeg_only: dict[tuple[str, int], Any] = {}
        for item in validation:
            values = _trial_observation(item)
            values[:, 1:] = np.nan
            eeg_only[(item.trial.subject, int(item.trial.event_index))] = smooth_balloon(
                values,
                parameters=parameters,
                observation_spec=observation_spec,
                config=balloon_config,
                observation_mask=np.isfinite(values),
            )
        for null_type in NULL_IDS:
            for case in _null_inputs(validation, null_type, shift_steps):
                item = case.receiver
                identity = _identity_fields(item, config["data"]["conditions"][0], config["data"])
                base_result = eeg_only[(item.trial.subject, int(item.trial.event_index))]
                base_std = np.sqrt(np.maximum(base_result.total_variance, 0.0))
                joint = smooth_balloon(
                    case.values,
                    parameters=parameters,
                    observation_spec=observation_spec,
                    config=balloon_config,
                    observation_mask=np.isfinite(case.values),
                )
                joint_std = np.sqrt(np.maximum(joint.total_variance, 0.0))
                for column, (target_name, paired_target, donor_target) in enumerate((
                    ("HbO", item.hbo, case.donor_hbo),
                    ("HbR", item.hbr, case.donor_hbr),
                ), start=1):
                    paired, donor = _matched_null_metrics(
                        paired_target,
                        donor_target,
                        base_result.observation_mean[:, column],
                        base_std[:, column],
                    )
                    joint_donor = _masked_metrics(donor_target, joint.observation_mean[:, column], np.isfinite(donor_target), joint_std[:, column])
                    driver_valid = np.isfinite(donor_target)
                    driver_pcc = _safe_corr(joint.state_mean[driver_valid, 0], np.asarray(donor_target)[driver_valid])
                    rows.append({
                        **identity,
                        "model": model_id,
                        "null_type": null_type,
                        "target": target_name,
                        "donor_subject": case.donor_subject,
                        "donor_event_index": int(case.donor_event_index),
                        "matched_support_n": int(donor["n"]),
                        "paired_nrmse": paired["nrmse"],
                        "null_nrmse": donor["nrmse"],
                        "delta_nrmse_null_minus_paired": float(donor["nrmse"] - paired["nrmse"]),
                        "joint_null_nrmse": joint_donor["nrmse"],
                        "driver_donor_pcc": driver_pcc,
                        "abs_driver_donor_pcc": abs(driver_pcc) if np.isfinite(driver_pcc) else float("nan"),
                    })
    return rows


def _plot_stage_release(
    subject_rows: Sequence[Mapping[str, Any]],
    stages: Sequence[Mapping[str, Any]],
    run_dir: Path,
    dpi: int,
) -> None:
    stage_ids = [str(stage["id"]) for stage in stages]
    labels = [str(stage["label_zh"]) for stage in stages]
    lookup = {(str(row["subject"]), str(row["stage"])): row for row in subject_rows}
    subjects = sorted({str(row["subject"]) for row in subject_rows})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), layout="constrained")
    for axis, metric, ylabel in (
        (axes[0], "fNIRS_gaussian_negative_log_score", "方差匹配高斯负对数评分/点（越低越好）"),
        (axes[1], "fNIRS_nrmse", "HbO/HbR 平均 NRMSE（描述性）"),
    ):
        matrix = np.asarray([
            [float(lookup[(subject, stage_id)][metric]) for stage_id in stage_ids]
            for subject in subjects
        ])
        for row in matrix:
            axis.plot(range(len(stage_ids)), row, color="#999999", alpha=0.30, linewidth=0.7)
        medians = np.nanmedian(matrix, axis=0)
        q25 = np.nanquantile(matrix, 0.25, axis=0)
        q75 = np.nanquantile(matrix, 0.75, axis=0)
        axis.fill_between(range(len(stage_ids)), q25, q75, color="#0072B2", alpha=0.18, label="18 名被试四分位区间")
        axis.plot(range(len(stage_ids)), medians, color="#0072B2", marker="o", linewidth=2.0, label="被试等权中位数")
        axis.set_xticks(range(len(stage_ids)), labels, rotation=28, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_yscale("symlog", linthresh=2.0)
    axes[0].set_ylabel("方差匹配高斯负对数评分/点（对称对数轴，越低越好）")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("HbO/HbR 平均 NRMSE（对数轴，描述性）")
    fig.suptitle("01–18 号被试：逐级释放生理参数后的内部试次留出表现")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "逐级参数释放与留出表现.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_parameter_distribution(
    rows: Sequence[Mapping[str, Any]],
    stage: Mapping[str, Any],
    run_dir: Path,
    dpi: int,
) -> None:
    names = tuple(map(str, stage.get("free", ())))
    if not names:
        return
    columns = 2
    rows_count = int(math.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(15, 3.8 * rows_count), squeeze=False, layout="constrained")
    subjects = sorted({str(row["subject"]) for row in rows})
    positions = {subject: index for index, subject in enumerate(subjects)}
    colors = {"fit": "#0072B2", "validation": "#D55E00"}
    markers = {"fit": "o", "validation": "s"}
    for axis, name in zip(axes.flat, names):
        selected = [row for row in rows if row["parameter"] == name and bool(row["is_free"])]
        if not selected:
            axis.set_visible(False)
            continue
        spec = selected[0]
        prior_mean = float(spec["prior_mean"])
        prior_sd = float(spec["prior_sd"])
        hard_lower = float(spec["hard_lower"])
        hard_upper = float(spec["hard_upper"])
        axis.axhspan(max(hard_lower, prior_mean - prior_sd), min(hard_upper, prior_mean + prior_sd), color="#CC79A7", alpha=0.12, label="先验中心 ±1 SD")
        axis.axhline(prior_mean, color="#CC79A7", linestyle="--", linewidth=1.0, label="先验中心")
        axis.axhline(hard_lower, color="#555555", linestyle=":", linewidth=0.8)
        axis.axhline(hard_upper, color="#555555", linestyle=":", linewidth=0.8, label="硬边界")
        for row in selected:
            split = str(row["split"])
            x = positions[str(row["subject"])]
            estimate = float(row["estimate"])
            lower = float(row["lower_95_laplace"])
            upper = float(row["upper_95_laplace"])
            if np.isfinite(lower) and np.isfinite(upper):
                axis.errorbar(x, estimate, yerr=[[estimate - lower], [upper - estimate]], fmt=markers[split], color=colors[split], markersize=4, linewidth=0.8, capsize=2)
            else:
                axis.scatter(x, estimate, marker=markers[split], s=25, facecolor="white", edgecolor=colors[split], linewidth=1.0)
            if row["boundary_status"] == "BOUNDARY":
                axis.scatter(x, estimate, marker="x", s=45, color="#000000", linewidth=1.2)
        axis.axvline(17.5, color="#666666", linewidth=0.9)
        axis.text(8.5, hard_upper, "拟合队列 01–18", ha="center", va="top", fontsize=8)
        axis.text(20.0, hard_upper, "验证队列\n描述性重拟合", ha="center", va="top", fontsize=8)
        axis.set_xticks(range(len(subjects)), [subject.replace("subject_", "") for subject in subjects], fontsize=7)
        axis.set_ylim(hard_lower - 0.03 * (hard_upper - hard_lower), hard_upper + 0.03 * (hard_upper - hard_lower))
        axis.set_title(f"{spec['parameter_label_zh']}（{spec['unit_zh']}）")
        axis.set_xlabel("被试编号")
        axis.set_ylabel("参数估计与局部 Laplace 95% 区间")
        axis.grid(axis="y", alpha=0.20)
    for axis in axes.flat[len(names):]:
        axis.set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color=colors["fit"], linestyle="none", label="01–18 全试次描述性重拟合"),
        Line2D([0], [0], marker="s", color=colors["validation"], linestyle="none", label="19–23 选型后描述性重拟合"),
        Line2D([0], [0], marker="x", color="#000000", linestyle="none", label="命中硬边界"),
        Patch(facecolor="#CC79A7", alpha=0.18, label="先验中心 ±1 SD"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=4, frameon=False)
    fig.suptitle(f"{stage['label_zh']}：逐被试标准化坐标有效参数（非绝对生理速率；验证被试不参与模型选择）", y=1.015)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "逐被试生理参数分布.png", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_identifiability(
    geometry_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    stages: Sequence[Mapping[str, Any]],
    distribution_stage: Mapping[str, Any],
    run_dir: Path,
    dpi: int,
) -> None:
    fit_geometry = [row for row in geometry_rows if row["split"] == "fit"]
    subjects = sorted({str(row["subject"]) for row in fit_geometry})
    stage_ids = [str(stage["id"]) for stage in stages]
    status_value = {"FIXED": 0, "IDENTIFIABLE": 1, "PRIOR_DOMINATED": 2, "COMPENSATORY": 3, "UNIDENTIFIABLE": 4, "BOUNDARY": 5}
    lookup = {(str(row["subject"]), str(row["stage"])): status_value[str(row["stage_status"])] for row in fit_geometry}
    matrix = np.asarray([[lookup[(subject, stage)] for stage in stage_ids] for subject in subjects])
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#BDBDBD", "#009E73", "#56B4E9", "#CC79A7", "#E69F00", "#D55E00"])
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": (1.25, 1.0)}, layout="constrained")
    axes[0].imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    axes[0].set_xticks(range(len(stage_ids)), [str(stage["label_zh"]) for stage in stages], rotation=30, ha="right")
    axes[0].set_yticks(range(len(subjects)), [subject.replace("subject_", "") for subject in subjects])
    axes[0].set_xlabel("参数释放阶段")
    axes[0].set_ylabel("拟合被试编号")
    axes[0].set_title("被试×阶段可辨识性分类")
    legend = [Patch(facecolor=cmap(index), label=label) for index, label in enumerate(("固定", "可辨识", "先验主导", "参数补偿", "不可辨识", "边界"))]
    axes[0].legend(handles=legend, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3, frameon=False)

    names = tuple(map(str, distribution_stage.get("free", ())))
    all_subjects = sorted({str(row["subject"]) for row in final_rows})
    contraction_lookup = {
        (str(row["subject"]), str(row["parameter"])): 1.0 - float(row["posterior_contraction"])
        for row in final_rows
        if bool(row["is_free"]) and np.isfinite(float(row["posterior_contraction"]))
    }
    ratios = np.full((len(all_subjects), len(names)), np.nan)
    for row_index, subject in enumerate(all_subjects):
        for column_index, name in enumerate(names):
            ratios[row_index, column_index] = contraction_lookup.get((subject, name), np.nan)
    masked = np.ma.masked_invalid(ratios)
    ratio_cmap = mpl.colormaps["viridis"].copy()
    ratio_cmap.set_bad("#EEEEEE")
    image = axes[1].imshow(masked, aspect="auto", cmap=ratio_cmap, vmin=0.0, vmax=2.0)
    axes[1].axhline(17.5, color="white", linewidth=2.0)
    axes[1].set_xticks(range(len(names)), [next(row["parameter_label_zh"] for row in final_rows if row["parameter"] == name) for name in names], rotation=25, ha="right")
    axes[1].set_yticks(range(len(all_subjects)), [subject.replace("subject_", "") for subject in all_subjects])
    axes[1].set_xlabel("最终描述性阶段参数")
    axes[1].set_ylabel("被试编号（白线以下为验证队列）")
    axes[1].set_title("后验 SD / 先验 SD（局部 Laplace）")
    if np.any(np.isfinite(ratios)):
        colorbar = fig.colorbar(image, ax=axes[1], shrink=0.8)
        colorbar.set_label("<1 表示局部区间较先验收缩；灰色表示无法估计")
    else:
        axes[1].text(
            0.5, 0.5,
            "所有被试的六参数拟合均触及至少一个硬边界\n局部 Laplace 区间不可用",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#8B0000",
        )
    fig.suptitle("参数可辨识性、边界与先验主导诊断")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "参数可辨识性与边界诊断.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_recommended_kappa(
    parameter_rows: Sequence[Mapping[str, Any]],
    subject_stage_rows: Sequence[Mapping[str, Any]],
    run_dir: Path,
    dpi: int,
) -> None:
    rows = [
        row for row in parameter_rows
        if row["stage"] == "M1_kappa" and row["parameter"] == "kappa"
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: str(row["subject"]))
    subjects = [str(row["subject"]) for row in rows]
    baseline = {
        str(row["subject"]): row
        for row in subject_stage_rows
        if row["stage"] == "M0_fixed"
    }
    fitted = {
        str(row["subject"]): row
        for row in subject_stage_rows
        if row["stage"] == "M1_kappa"
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), layout="constrained")
    spec = rows[0]
    prior_mean = float(spec["prior_mean"])
    prior_sd = float(spec["prior_sd"])
    axes[0].axhspan(prior_mean - prior_sd, prior_mean + prior_sd, color="#CC79A7", alpha=0.14, label="先验中心 ±1 SD")
    axes[0].axhline(prior_mean, color="#CC79A7", linestyle="--", linewidth=1.1, label="先验中心 0.64/s")
    axes[0].axhline(float(spec["hard_lower"]), color="#555555", linestyle=":", linewidth=0.9)
    axes[0].axhline(float(spec["hard_upper"]), color="#555555", linestyle=":", linewidth=0.9, label="硬边界")
    for index, row in enumerate(rows):
        estimate = float(row["estimate"])
        lower = float(row["lower_95_laplace"])
        upper = float(row["upper_95_laplace"])
        if np.isfinite(lower) and np.isfinite(upper):
            axes[0].errorbar(index, estimate, yerr=[[estimate - lower], [upper - estimate]], fmt="o", color="#0072B2", capsize=2, markersize=4)
        else:
            axes[0].scatter(index, estimate, facecolor="white", edgecolor="#D55E00", s=35)
        if row["boundary_status"] == "BOUNDARY":
            axes[0].scatter(index, estimate, marker="x", color="black", s=55, linewidth=1.2)
    axes[0].set_xticks(range(len(subjects)), [subject.replace("subject_", "") for subject in subjects])
    axes[0].set_ylim(float(spec["hard_lower"]) - 0.04, float(spec["hard_upper"]) + 0.04)
    axes[0].set_xlabel("拟合被试编号")
    axes[0].set_ylabel("κ（每秒）与局部 Laplace 95% 区间")
    axes[0].set_title("8 个训练试次共享的逐被试 κ")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    delta_nll = [float(fitted[s]["fNIRS_gaussian_negative_log_score"]) - float(baseline[s]["fNIRS_gaussian_negative_log_score"]) for s in subjects]
    delta_nrmse = [float(fitted[s]["fNIRS_nrmse"]) - float(baseline[s]["fNIRS_nrmse"]) for s in subjects]
    x = np.arange(len(subjects), dtype=float)
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].scatter(x - 0.10, delta_nll, marker="o", facecolor="white", edgecolor="#0072B2", label="负对数评分差（M1−M0）")
    axes[1].scatter(x + 0.10, delta_nrmse, marker="s", facecolor="white", edgecolor="#D55E00", label="NRMSE 差（M1−M0）")
    axes[1].set_yscale("symlog", linthresh=0.25)
    axes[1].set_xticks(range(len(subjects)), [subject.replace("subject_", "") for subject in subjects])
    axes[1].set_xlabel("拟合被试编号")
    axes[1].set_ylabel("相对 M0 的变化（对称对数轴；<0 改善）")
    axes[1].set_title("同一被试 2 个内部留出试次的 fNIRS 变化")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("推荐阶段 M1：仅释放标准化坐标下的有效 κ（01–18；目标为含噪实测轨迹）")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "推荐M1κ逐被试拟合与留出变化.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_validation_and_null(
    validation_subject_rows: Sequence[Mapping[str, Any]],
    null_subject_rows: Sequence[Mapping[str, Any]],
    model_labels: Mapping[str, str],
    run_dir: Path,
    dpi: int,
) -> None:
    models = list(model_labels)
    colors = ["#777777", "#0072B2", "#D55E00"][:len(models)]
    targets = ("HbO", "HbR")
    nulls = (
        ("independent", "跨被试独立供体"),
        ("pairing", "同被试错误试次"),
        ("time_shift", "非循环平移 +10 秒†"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex="col", layout="constrained")

    def draw(axis: Any, rows: Sequence[Mapping[str, Any]], field: str, **filters: str) -> None:
        for index, model in enumerate(models):
            values = np.asarray([
                float(row[field])
                for row in rows
                if all(str(row[key]) == value for key, value in filters.items())
                and str(row.get("stage", row.get("model"))) == model
            ])
            offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.asarray([0.0])
            axis.scatter(index + offsets, values, facecolor="white", edgecolor=colors[index], s=28)
            if len(values):
                median = float(np.median(values))
                q25, q75 = np.quantile(values, (0.25, 0.75))
                axis.errorbar(index, median, yerr=[[median - q25], [q75 - median]], fmt="o", color=colors[index], capsize=4, linewidth=2)
        axis.grid(axis="y", alpha=0.22)
        axis.set_xticks(range(len(models)), [model_labels[model] for model in models], rotation=18, ha="right")

    for row_index, target in enumerate(targets):
        draw(axes[row_index, 0], validation_subject_rows, "nrmse", target=target)
        axes[row_index, 0].set_title(f"{target}：中心遮挡")
        axes[row_index, 0].set_ylabel("中心遮挡 NRMSE\n（含噪实测目标）")
        for column_index, (null_id, null_label) in enumerate(nulls, start=1):
            axis = axes[row_index, column_index]
            draw(axis, null_subject_rows, "delta_nrmse_null_minus_paired", target=target, null_type=null_id)
            axis.axhline(0.0, color="black", linewidth=0.9)
            axis.set_title(f"{target}：{null_label}")
            if column_index == 1:
                axis.set_ylabel("空对照 − 真实配对 NRMSE\n（>0 表示正确配对更优）")
    fig.suptitle("冻结参数验证与配对空对照（点为被试；实心点及误差线为中位数与四分位距）")
    fig.text(0.995, 0.002, "† 平移项在共同有限支持上比较（每试次 100 点；其他空对照 200 点），不同空对照的幅度勿直接横比。", ha="right", va="bottom", fontsize=8, color="#8B0000")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "冻结参数验证与Null特异性.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_population_waveform(
    example: PreparedTrial,
    models: Mapping[str, BalloonParameters],
    model_labels: Mapping[str, str],
    observation_spec: BalloonObservationSpec,
    balloon_config: BalloonConfig,
    center: np.ndarray,
    fs_hz: float,
    run_dir: Path,
    dpi: int,
) -> None:
    time = np.arange(len(example.hbo), dtype=np.float64) / fs_hz - 5.0
    truth = _trial_observation(example)
    fig, axes = plt.subplots(3, len(models), figsize=(6.2 * len(models), 9.5), sharex=True, sharey="row", squeeze=False, layout="constrained")
    names = ("EEG 10 Hz 功率代理", "HbO 标准化坐标", "HbR 标准化坐标")
    colors = ["#777777", "#0072B2", "#D55E00"][:len(models)]
    for column_index, (model_id, parameters) in enumerate(models.items()):
        for row_index in range(3):
            values = truth.copy()
            masked_columns = (0,) if row_index == 0 else (1, 2)
            values[np.ix_(center, np.asarray(masked_columns))] = np.nan
            result = smooth_balloon(values, parameters=parameters, observation_spec=observation_spec, config=balloon_config, observation_mask=np.isfinite(values))
            estimate = result.observation_mean[:, row_index]
            std = np.sqrt(np.maximum(result.total_variance[:, row_index], 0.0))
            axis = axes[row_index, column_index]
            axis.plot(time, truth[:, row_index], color="#222222", linewidth=1.0, label="原始实测（非干净真值）")
            axis.plot(time, estimate, color=colors[column_index], linewidth=1.3, label="中心遮挡重建")
            axis.fill_between(time, estimate - 1.96 * std, estimate + 1.96 * std, color=colors[column_index], alpha=0.16, label="近似 95% 预测区间（非覆盖保证）")
            axis.axvspan(
                time[np.flatnonzero(center)[0]], time[np.flatnonzero(center)[-1]],
                color="#E69F00", alpha=0.12,
                label="中心 4 秒：本行目标被遮挡",
            )
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(model_labels[model_id])
            if column_index == 0:
                axis.set_ylabel(names[row_index])
            if row_index == 2:
                axis.set_xlabel("事件相对时间（秒）")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=7)
    subject_label = example.trial.subject.replace("subject_", "")
    fig.suptitle(f"预先固定代表窗口：被试 {subject_label}，事件 {int(example.trial.event_index)}")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "固定代表窗口_群体参数重建.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _run_parameter_fit_validated(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run the staged subject-parameter experiment on the registered split."""

    condition = config["data"]["conditions"][0]
    fit_subjects = list(map(str, condition["fit_subjects"]))
    validation_subjects = list(map(str, condition["validation_subjects"]))
    fit_trials, fit_series, validation, contracts, adapter, hbo_names, hbr_indices = _prepare_measured_series(config)
    bundle, calibration = _fit_models(fit_series, config, fit_comparison_models=False)
    base_parameters, observation_spec, balloon_config = bundle.t3a
    fit_config = config["ssm"]["t3a"]["parameter_fit"]
    parameter_specs = fit_config["parameters"]
    stages = [dict(stage) for stage in fit_config["stages"]]
    stage_lookup = {str(stage["id"]): stage for stage in stages}
    center_cfg = config["analysis"]["center_mask"]
    center = _center_mask(
        len(fit_series[0].hbo),
        float(config["analysis"]["sampling_hz"]),
        float(center_cfg["relative_start_s"]),
        float(center_cfg["duration_s"]),
    )
    stage_train, stage_heldout = _split_stage_trials(
        fit_series, fit_subjects, tuple(map(int, fit_config["heldout_trial_positions"]))
    )
    base_values = _parameter_values(base_parameters)
    warm_values = {subject: dict(base_values) for subject in fit_subjects}
    stage_results: dict[str, list[dict[str, Any]]] = {}
    parameter_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    optimizer_start_rows: list[dict[str, Any]] = []
    print("开始 01–18 号被试的逐级参数释放拟合", flush=True)
    for stage in stages:
        tasks = [{
            "subject": subject,
            "split": "fit",
            "fit_scope": "eight_trials_fit_two_trials_internal_holdout",
            "stage": stage,
            "parameter_specs": parameter_specs,
            "base_parameters": base_parameters,
            "observation_spec": observation_spec,
            "balloon_config": balloon_config,
            "fit_config": fit_config,
            "train_trials": stage_train[subject],
            "heldout_trials": stage_heldout[subject],
            "center_mask": center,
            "initial_values": warm_values[subject],
        } for subject in fit_subjects]
        results = _run_stage_tasks(tasks, int(fit_config["workers"]))
        stage_id = str(stage["id"])
        stage_results[stage_id] = results
        for result in results:
            for name in stage.get("free", ()):
                warm_values[str(result["subject"])][str(name)] = float(result["values"][str(name)])
            parameter_rows.extend(result["parameter_rows"])
            metric_rows.extend(result["metric_rows"])
            trial_rows.extend(result["trial_rows"])
            geometry_rows.append(result["geometry_row"])
            for start_index, record in enumerate(result["start_records"]):
                optimizer_start_rows.append({
                    "subject": result["subject"],
                    "split": "fit",
                    "fit_scope": "eight_trials_fit_two_trials_internal_holdout",
                    "stage": stage_id,
                    "start_index": start_index,
                    "start_values": json.dumps(_jsonable(record["start"]), ensure_ascii=False, sort_keys=True),
                    "estimate_values": json.dumps(_jsonable(record["estimate"]), ensure_ascii=False, sort_keys=True),
                    "objective": float(record["objective"]),
                    "success": bool(record["success"]),
                    "message": str(record["message"]),
                })
        _write_csv(run_dir / "subject_stage_parameters.csv", parameter_rows)
        _write_csv(run_dir / "subject_stage_metrics.csv", metric_rows)
        _write_csv(run_dir / "subject_stage_geometry.csv", geometry_rows)

    stage_subject_rows, stage_summary_rows, recommended_stage_id, best_predictive_stage_id = _stage_summaries(
        metric_rows, geometry_rows, parameter_rows, stages, fit_config
    )
    nonbaseline = [
        row for row in stage_summary_rows
        if row["stage"] != "M0_fixed"
        and np.isfinite(float(row["fNIRS_gaussian_negative_log_score_median"]))
    ]
    if not nonbaseline:
        raise RuntimeError("no non-baseline parameter stage produced finite heldout scores")
    distribution_stage_id = min(
        nonbaseline,
        key=lambda row: (float(row["fNIRS_gaussian_negative_log_score_median"]), int(row["free_parameter_count"])),
    )["stage"]
    distribution_stage = stage_lookup[str(distribution_stage_id)]

    population_models: dict[str, BalloonParameters] = {"M0_fixed": base_parameters}
    for stage_id in (recommended_stage_id, best_predictive_stage_id):
        if stage_id != "M0_fixed" and stage_id not in population_models:
            population_models[stage_id] = _population_parameters(
                base_parameters, stage_lookup[stage_id], stage_results[stage_id]
            )
    model_labels = {model_id: str(stage_lookup[model_id]["label_zh"]) for model_id in population_models}

    validation_full, _ = _split_stage_trials(validation, validation_subjects, ())
    frozen_results: list[dict[str, Any]] = []
    print("开始 19–23 号被试的冻结群体参数纯应用评价", flush=True)
    for model_id, parameters in population_models.items():
        tasks = [{
            "subject": subject,
            "split": "validation",
            "fit_scope": "validation_pure_apply_frozen_fit_population",
            "stage": {"id": model_id, "label_zh": model_labels[model_id], "free": []},
            "parameter_specs": parameter_specs,
            "base_parameters": parameters,
            "observation_spec": observation_spec,
            "balloon_config": balloon_config,
            "fit_config": fit_config,
            "train_trials": (),
            "heldout_trials": validation_full[subject],
            "center_mask": center,
            "initial_values": _parameter_values(parameters),
        } for subject in validation_subjects]
        frozen_results.extend(_run_stage_tasks(tasks, min(5, int(fit_config["workers"]))))
    validation_metric_rows = [row for result in frozen_results for row in result["metric_rows"]]
    validation_subject_rows, validation_summary_rows = _subject_equal_aggregate(
        validation_metric_rows,
        ("stage", "mode", "target"),
        ("gaussian_negative_log_score", "nrmse", "pcc", "coverage_95_gaussian_approx", "standardized_residual_rms"),
    )

    print("开始冻结群体参数 Null 评价", flush=True)
    null_rows = _evaluate_frozen_nulls(
        population_models, validation, observation_spec, balloon_config, config
    )
    null_subject_rows, null_summary_rows = _subject_equal_aggregate(
        null_rows,
        ("model", "null_type", "target"),
        ("matched_support_n", "paired_nrmse", "null_nrmse", "delta_nrmse_null_minus_paired", "joint_null_nrmse", "driver_donor_pcc", "abs_driver_donor_pcc"),
    )

    print(f"开始 {distribution_stage_id} 的 01–23 全 trial 描述性重拟合", flush=True)
    all_series = [*fit_series, *validation]
    all_subjects = [*fit_subjects, *validation_subjects]
    all_full, _ = _split_stage_trials(all_series, all_subjects, ())
    population_initial = _parameter_values(_population_parameters(
        base_parameters, distribution_stage, stage_results[str(distribution_stage_id)]
    ))
    selected_result_lookup = {str(result["subject"]): result for result in stage_results[str(distribution_stage_id)]}
    full_tasks = []
    for subject in all_subjects:
        is_fit = subject in fit_subjects
        full_tasks.append({
            "subject": subject,
            "split": "fit" if is_fit else "validation",
            "fit_scope": "post_selection_all_trials_descriptive" if is_fit else "validation_post_selection_descriptive",
            "stage": distribution_stage,
            "parameter_specs": parameter_specs,
            "base_parameters": base_parameters,
            "observation_spec": observation_spec,
            "balloon_config": balloon_config,
            "fit_config": fit_config,
            "train_trials": all_full[subject],
            "heldout_trials": (),
            "center_mask": center,
            "initial_values": selected_result_lookup[subject]["values"] if is_fit else population_initial,
        })
    final_results = _run_stage_tasks(full_tasks, int(fit_config["workers"]))
    final_parameter_rows = [row for result in final_results for row in result["parameter_rows"]]
    final_geometry_rows = [result["geometry_row"] for result in final_results]
    final_trial_rows = [row for result in final_results for row in result["trial_rows"]]

    distribution_rows: list[dict[str, Any]] = []
    for split in ("fit", "validation"):
        for name in distribution_stage.get("free", ()):
            rows = [
                row for row in final_parameter_rows
                if row["split"] == split and row["parameter"] == name and bool(row["is_free"])
            ]
            estimates = np.asarray([float(row["estimate"]) for row in rows], dtype=np.float64)
            distribution_rows.append({
                "split": split,
                "fit_scope": rows[0]["fit_scope"],
                "stage": distribution_stage_id,
                "parameter": name,
                "parameter_label_zh": rows[0]["parameter_label_zh"],
                "unit_zh": rows[0]["unit_zh"],
                "subject_count": len(rows),
                "estimate_median": float(np.median(estimates)),
                "estimate_q25": float(np.quantile(estimates, 0.25)),
                "estimate_q75": float(np.quantile(estimates, 0.75)),
                "estimate_min": float(np.min(estimates)),
                "estimate_max": float(np.max(estimates)),
                "boundary_fraction": float(np.mean([row["boundary_status"] == "BOUNDARY" for row in rows])),
                "identifiable_fraction": float(np.mean([row["identifiability_status"] == "IDENTIFIABLE" for row in rows])),
                "prior_dominated_fraction": float(np.mean([row["identifiability_status"] == "PRIOR_DOMINATED" for row in rows])),
                "unidentifiable_or_compensatory_fraction": float(np.mean([row["identifiability_status"] in {"UNIDENTIFIABLE", "COMPENSATORY"} for row in rows])),
            })

    parameter_registry = [{
        "parameter": name,
        "parameter_label_zh": str(parameter_specs[name]["label_zh"]),
        "unit_zh": str(parameter_specs[name]["unit_zh"]),
        "prior_mean": float(parameter_specs[name]["prior_mean"]),
        "prior_sd": float(parameter_specs[name]["prior_sd"]),
        "hard_lower": float(parameter_specs[name]["bounds"][0]),
        "hard_upper": float(parameter_specs[name]["bounds"][1]),
        "gauge_policy": "P0/Q0, EEG loading, driver scale and noise fixed from subjects 01-18",
    } for name in PARAMETER_NAMES]
    trial_inventory: list[dict[str, Any]] = []
    heldout_positions = set(map(int, fit_config["heldout_trial_positions"]))
    by_subject: dict[str, list[PreparedTrial]] = defaultdict(list)
    for item in all_series:
        by_subject[item.trial.subject].append(item)
    for subject in all_subjects:
        ordered = sorted(by_subject[subject], key=lambda item: int(item.trial.event_index))
        for position, item in enumerate(ordered):
            trial_inventory.append({
                **_identity_fields(item, condition, config["data"]),
                "split": "fit" if subject in fit_subjects else "validation",
                "stage_selection_role": "internal_holdout" if subject in fit_subjects and position in heldout_positions else ("stage_fit" if subject in fit_subjects else "validation_pure_apply"),
                "used_in_post_selection_descriptive_refit": True,
            })

    _write_csv(run_dir / "parameter_registry.csv", parameter_registry)
    _write_csv(run_dir / "stage_subject_summary.csv", stage_subject_rows)
    _write_csv(run_dir / "stage_summary.csv", stage_summary_rows)
    _write_csv(run_dir / "subject_stage_parameters.csv", parameter_rows)
    _write_csv(run_dir / "subject_stage_metrics.csv", metric_rows)
    _write_csv(run_dir / "subject_stage_geometry.csv", geometry_rows)
    _write_csv(run_dir / "optimizer_starts.csv", optimizer_start_rows)
    _write_csv(run_dir / "stage_trial_diagnostics.csv", trial_rows)
    _write_csv(run_dir / "validation_frozen_metrics.csv", validation_metric_rows)
    _write_csv(run_dir / "validation_frozen_subject_metrics.csv", validation_subject_rows)
    _write_csv(run_dir / "validation_frozen_summary.csv", validation_summary_rows)
    _write_csv(run_dir / "validation_null_metrics.csv", null_rows)
    _write_csv(run_dir / "validation_null_subject_metrics.csv", null_subject_rows)
    _write_csv(run_dir / "validation_null_summary.csv", null_summary_rows)
    _write_csv(run_dir / "subject_final_parameters.csv", final_parameter_rows)
    _write_csv(run_dir / "subject_final_geometry.csv", final_geometry_rows)
    _write_csv(run_dir / "subject_final_trial_diagnostics.csv", final_trial_rows)
    _write_csv(run_dir / "subject_parameter_distribution.csv", distribution_rows)
    _write_csv(run_dir / "trial_inventory.csv", trial_inventory)
    _write_json(run_dir / "calibration.json", {
        "scope": "fit_subjects_01_18_only",
        "selected_hbo_channels": list(hbo_names),
        "selected_hbr_channels": [fit_trials[0].fnirs_channel_names[int(index)] for index in hbr_indices],
        "selected_eeg_channels": list(adapter.channel_names),
        **calibration,
    })
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(_jsonable(config), sort_keys=False), encoding="utf-8")

    representative = min(
        (item for item in validation if item.trial.subject == "subject_19"),
        key=lambda item: int(item.trial.event_index),
    )
    with mpl.rc_context(CJK_STYLE):
        _plot_stage_release(stage_subject_rows, stages, run_dir, int(config["output"]["figure_dpi"]))
        _plot_parameter_distribution(final_parameter_rows, distribution_stage, run_dir, int(config["output"]["figure_dpi"]))
        _plot_identifiability(geometry_rows, final_parameter_rows, stages, distribution_stage, run_dir, int(config["output"]["figure_dpi"]))
        _plot_recommended_kappa(parameter_rows, stage_subject_rows, run_dir, int(config["output"]["figure_dpi"]))
        _plot_validation_and_null(validation_subject_rows, null_subject_rows, model_labels, run_dir, int(config["output"]["figure_dpi"]))
        _plot_population_waveform(
            representative, population_models, model_labels, observation_spec, balloon_config,
            center, float(config["analysis"]["sampling_hz"]), run_dir, int(config["output"]["figure_dpi"]),
        )

    stage_lines = [
        f"- {row['stage_label_zh']}：留出 fNIRS 负对数评分中位数 {float(row['fNIRS_gaussian_negative_log_score_median']):.4f}，相对 M0 差 {float(row['delta_score_vs_M0_median']):+.4f}，边界率 {float(row['boundary_fraction_free_parameter_rows']):.1%}，非可辨识被试率 {float(row['non_identifiable_subject_fraction']):.1%}。"
        for row in stage_summary_rows
    ]
    distribution_lines = [
        f"- {row['parameter_label_zh']}（{row['split']}）：中位数 {float(row['estimate_median']):.4g}，IQR [{float(row['estimate_q25']):.4g}, {float(row['estimate_q75']):.4g}]，范围 [{float(row['estimate_min']):.4g}, {float(row['estimate_max']):.4g}]，边界率 {float(row['boundary_fraction']):.1%}，可辨识率 {float(row['identifiable_fraction']):.1%}。"
        for row in distribution_rows
    ]
    (run_dir / "summary.md").write_text(
        "# T3a 实测被试生理参数拟合与分布\n\n"
        "01–18 号被试用于逐级释放参数和内部 trial 留出比较；每个 trial 独立从静息状态重置，先验每名被试只计一次。19–23 号被试的主性能与 Null 使用冻结的 01–18 群体参数；其逐被试参数仅在选型之后重拟合并单列为描述性结果。24–29 未开放。P0/Q0、EEG loading、共享驱动尺度及噪声保持固定，因此这里的参数仍是标准化观测坐标下的有效模型参数。\n\n"
        f"## 选型结论\n\n生理可辨识约束下推荐 `{recommended_stage_id}`；单看内部留出预测，最佳为 `{best_predictive_stage_id}`；用于展示 23 人可比参数分布的选型后描述性阶段为 `{distribution_stage_id}`。E0 阶段按合同仅作诊断。\n\n"
        "## 逐级结果\n\n" + "\n".join(stage_lines) + "\n\n"
        "## 逐被试分布摘要\n\n" + "\n".join(distribution_lines) + "\n\n"
        "局部 Laplace 区间、优化成功和边界内估计都不自动等于参数可辨识；最终解释以 `subject_stage_geometry.csv` 与图中的分类为准。\n",
        encoding="utf-8",
    )
    summary = {
        "schema": SCHEMA,
        "analysis_kind": "staged_subject_parameter_fit",
        "scope": "measured_development_exploratory",
        "condition_id": str(condition["condition_id"]),
        "fit_subjects": fit_subjects,
        "validation_subjects": validation_subjects,
        "closed_protected_subjects": list(condition["protected_subjects"]),
        "fit_trial_count": len(fit_series),
        "validation_trial_count": len(validation),
        "models": list(population_models),
        "modes": ["center_masked_eeg", "center_masked_fnirs"],
        "nulls": list(NULL_IDS),
        "stage_order": [str(stage["id"]) for stage in stages],
        "recommended_stage": recommended_stage_id,
        "best_internal_predictive_stage": best_predictive_stage_id,
        "distribution_stage": str(distribution_stage_id),
        "fit_policy": "one subject parameter vector shared across trials; every trial independently resets at rest; prior applied once per subject",
        "internal_holdout_policy": f"sorted trial positions {list(map(int, fit_config['heldout_trial_positions']))} held out within subjects 01-18 under the fixed fit-cohort gauge",
        "validation_policy": "subjects 19-23 use frozen fit-population parameters for primary metrics/null; later per-subject fits are descriptive only",
        "validation_fit_used_for_primary": False,
        "parameter_interval": "local transformed-coordinate Laplace approximation; invalid at boundary/non-positive curvature",
        "parameter_claim_limit": "effective standardized-coordinate model parameters; no absolute OEF/CMRO2 or molecular-rate claim",
        "target_semantics": "held-out measured signal; no clean physiological truth",
        "protected_data_opened": False,
        "qualification_eligible": False,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": mpl.__version__,
            "pyyaml": yaml.__version__,
        },
        "representative_trial": {"subject": representative.trial.subject, "event_index": int(representative.trial.event_index), "selection": "pre_fixed_first_subject_19_event"},
        "loader_contracts": contracts,
        "files": [
            "manifest.json", "summary.json", "summary.md", "resolved_config.yaml", "calibration.json",
            "parameter_registry.csv", "stage_subject_summary.csv", "stage_summary.csv",
            "subject_stage_parameters.csv", "subject_stage_metrics.csv", "subject_stage_geometry.csv",
            "optimizer_starts.csv", "stage_trial_diagnostics.csv", "validation_frozen_metrics.csv",
            "validation_frozen_subject_metrics.csv", "validation_frozen_summary.csv", "validation_null_metrics.csv",
            "validation_null_subject_metrics.csv", "validation_null_summary.csv", "subject_final_parameters.csv",
            "subject_final_geometry.csv", "subject_final_trial_diagnostics.csv", "subject_parameter_distribution.csv",
            "trial_inventory.csv", "figures/逐级参数释放与留出表现.png", "figures/逐被试生理参数分布.png",
            "figures/参数可辨识性与边界诊断.png", "figures/冻结参数验证与Null特异性.png",
            "figures/固定代表窗口_群体参数重建.png", "figures/推荐M1κ逐被试拟合与留出变化.png",
        ],
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _run_validated(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Execute the measured run after the boundary and output are prepared."""

    condition = config["data"]["conditions"][0]
    condition_id = str(condition["condition_id"])
    fit_subjects = list(map(str, condition["fit_subjects"]))
    validation_subjects = list(map(str, condition["validation_subjects"]))
    fit_trials, fit_series, validation, contracts, adapter, hbo_names, hbr_indices = _prepare_measured_series(config)
    bundle, calibration = _fit_models(fit_series, config)

    center_cfg = config["analysis"]["center_mask"]
    center = _center_mask(len(validation[0].hbo), float(config["analysis"]["sampling_hz"]), float(center_cfg["relative_start_s"]), float(center_cfg["duration_s"]))
    trajectory_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    all_predictions: dict[str, dict[tuple[str, int, str], dict[str, Any]]] = {}
    for item in validation:
        identity = _identity_fields(item, condition, config["data"])
        truth = np.column_stack((item.eeg_driver, item.hbo, item.hbr))
        for mode in MODE_IDS:
            values = _mode_input(item, mode, center)
            for model in MODEL_IDS:
                result = _run_model(model, bundle, values)
                all_predictions.setdefault(model, {})[(item.trial.subject, int(item.trial.event_index), mode)] = result
                estimate = np.asarray(result["estimate"], dtype=np.float64)
                pred_std = np.asarray(result["predictive_std"], dtype=np.float64)
                for column, name in enumerate(OBS_NAMES):
                    for region, region_mask in (("all", np.ones(len(center), dtype=bool)), ("center", center)):
                        primary = region == "center" and (
                            (mode == "center_masked_eeg" and name == "EEG")
                            or (mode == "center_masked_fnirs" and name in {"HbO", "HbR"})
                        )
                        if primary:
                            evaluation_kind = "primary_center_masked"
                        elif mode == "joint":
                            evaluation_kind = "same_point_posterior_descriptive"
                        elif (mode == "eeg_only" and name in {"HbO", "HbR"}) or (mode == "fnirs_only" and name == "EEG"):
                            evaluation_kind = "cross_modal_descriptive"
                        else:
                            evaluation_kind = "unheld_coordinate_descriptive"
                        metrics = _masked_metrics(
                            truth[:, column], estimate[:, column], region_mask, pred_std[:, column]
                        )
                        metric_rows.append({
                            **identity,
                            "model": model,
                            "mode": mode,
                            "region": region,
                            "target": name,
                            "evaluation_kind": evaluation_kind,
                            **metrics,
                        })
                for index, time_s in enumerate(np.arange(len(item.hbo), dtype=np.float64) / float(config["analysis"]["sampling_hz"]) - 5.0):
                    row = {**identity, "model": model, "mode": mode, "null_type": "none", "time_s": float(time_s), "center_mask": bool(center[index])}
                    for column, name in enumerate(("eeg", "hbo", "hbr")):
                        row[f"{name}_observed"] = float(truth[index, column])
                        row[f"{name}_input"] = float(values[index, column]) if np.isfinite(values[index, column]) else None
                        row[f"{name}_reconstructed"] = float(estimate[index, column])
                        row[f"{name}_predictive_std"] = float(pred_std[index, column]) if np.isfinite(pred_std[index, column]) else None
                    row["shared_driver"] = float(result["driver"][index]) if result["driver"] is not None else None
                    row["shared_driver_std"] = float(result["driver_std"][index]) if result["driver_std"] is not None else None
                    if result["states"] is not None:
                        for state_index, state_name in enumerate(result["state_names"]):
                            row[f"state_{state_name}"] = float(result["states"][index, state_index])
                            row[f"state_{state_name}_std"] = float(result["state_std"][index, state_index])
                    trajectory_rows.append(row)
                if isinstance(result.get("physical_checks"), Mapping):
                    for check, value in result["physical_checks"].items():
                        physical_rows.append({
                            **identity,
                            "model": model,
                            "mode": mode,
                            "check": check,
                            "value": value,
                            "status": _physical_check_status(value),
                        })

    null_rows: list[dict[str, Any]] = []
    shift_steps = int(round(float(config["analysis"]["time_shift_s"]) * float(config["analysis"]["sampling_hz"])))
    for null_type in NULL_IDS:
        for null_case in _null_inputs(validation, null_type, shift_steps):
            item = null_case.receiver
            identity = _identity_fields(item, condition, config["data"])
            for model in MODEL_IDS:
                key = (item.trial.subject, int(item.trial.event_index), "eeg_only")
                eeg_only_result = all_predictions[model][key]
                eeg_only_estimate = np.asarray(eeg_only_result["estimate"], dtype=np.float64)
                eeg_only_std = np.asarray(eeg_only_result["predictive_std"], dtype=np.float64)
                joint_result = _run_model(model, bundle, null_case.values) if model in {"T2b", "T3a"} else None
                for column, (target_name, paired_target, donor_target) in enumerate((
                    ("HbO", item.hbo, null_case.donor_hbo),
                    ("HbR", item.hbr, null_case.donor_hbr),
                ), start=1):
                    paired, donor = _matched_null_metrics(
                        paired_target,
                        donor_target,
                        eeg_only_estimate[:, column],
                        eeg_only_std[:, column],
                    )
                    driver_donor_pcc = float("nan")
                    driver_eeg_pcc = float("nan")
                    driver_shift_nrms = float("nan")
                    joint_donor_nrmse = float("nan")
                    if joint_result is not None:
                        driver = np.asarray(joint_result["driver"], dtype=np.float64)
                        donor_valid = np.isfinite(donor_target)
                        driver_donor_pcc = _safe_corr(driver[donor_valid], donor_target[donor_valid])
                        driver_eeg_pcc = _safe_corr(driver, item.eeg_driver)
                        base_driver = np.asarray(eeg_only_result["driver"], dtype=np.float64)
                        driver_shift_nrms = float(
                            np.sqrt(np.mean((driver - base_driver) ** 2))
                            / max(float(np.std(base_driver)), 1e-8)
                        )
                        joint_estimate = np.asarray(joint_result["estimate"], dtype=np.float64)
                        joint_std = np.asarray(joint_result["predictive_std"], dtype=np.float64)
                        joint_donor_nrmse = _masked_metrics(
                            donor_target,
                            joint_estimate[:, column],
                            donor_valid,
                            joint_std[:, column],
                        )["nrmse"]
                    delta = (
                        float(donor["nrmse"] - paired["nrmse"])
                        if np.isfinite(donor["nrmse"]) and np.isfinite(paired["nrmse"])
                        else float("nan")
                    )
                    null_rows.append({
                        **identity,
                        "donor_dataset_id": null_case.donor_dataset_id,
                        "donor_subject": null_case.donor_subject,
                        "donor_record_id": null_case.donor_record_id,
                        "donor_event_index": null_case.donor_event_index,
                        "donor_is_receiver_trial": bool(
                            null_case.donor_dataset_id == item.trial.dataset_id
                            and null_case.donor_subject == item.trial.subject
                            and null_case.donor_record_id == item.trial.record_id
                            and null_case.donor_event_index == int(item.trial.event_index)
                        ),
                        "model": model,
                        "null_type": null_type,
                        "target": target_name,
                        "support_n": donor["n"],
                        "paired_low_observed_variance": paired["low_observed_variance"],
                        "null_low_observed_variance": donor["low_observed_variance"],
                        "paired_nrmse": paired["nrmse"],
                        "null_nrmse": donor["nrmse"],
                        "delta_nrmse_null_minus_paired": delta,
                        "joint_null_nrmse": joint_donor_nrmse,
                        "driver_donor_pcc": driver_donor_pcc,
                        "abs_driver_donor_pcc": abs(driver_donor_pcc) if np.isfinite(driver_donor_pcc) else float("nan"),
                        "driver_base_eeg_pcc": driver_eeg_pcc,
                        "driver_shift_nrms_from_eeg_only": driver_shift_nrms,
                        "synthetic_abs_correlation_reference": 0.35,
                        "synthetic_reference_is_measured_gate": False,
                    })

    metric_fields = (
        "n", "rmse", "nrmse", "low_observed_variance", "observed_temporal_sd",
        "reconstructed_temporal_sd", "temporal_sd_ratio", "pcc", "r2", "mean_bias", "variance_ratio",
        "standardized_residual_rms", "coverage_95_gaussian_approx",
        "predictive_valid_point_count", "mean_predictive_std", "median_predictive_std",
        "interval_width_95_gaussian_approx",
    )
    subject_rows, summary_rows = _subject_equal_aggregate(
        metric_rows,
        ("condition_id", "model", "mode", "region", "target", "evaluation_kind"),
        metric_fields,
    )
    null_metric_fields = (
        "support_n", "paired_low_observed_variance", "null_low_observed_variance",
        "paired_nrmse", "null_nrmse", "delta_nrmse_null_minus_paired",
        "joint_null_nrmse", "driver_donor_pcc", "abs_driver_donor_pcc",
        "driver_base_eeg_pcc", "driver_shift_nrms_from_eeg_only",
    )
    null_subject_rows, null_summary_rows = _subject_equal_aggregate(
        null_rows,
        ("condition_id", "model", "null_type", "target"),
        null_metric_fields,
    )

    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "subject_metrics.csv", subject_rows)
    _write_csv(run_dir / "summary_metrics.csv", summary_rows)
    _write_csv(run_dir / "null_metrics.csv", null_rows)
    _write_csv(run_dir / "null_subject_metrics.csv", null_subject_rows)
    _write_csv(run_dir / "null_summary.csv", null_summary_rows)
    _write_csv(run_dir / "trajectories.csv", trajectory_rows)
    _write_csv(run_dir / "physical_checks.csv", physical_rows)
    _write_csv(run_dir / "fit_parameters.csv", _fit_rows(bundle, calibration))
    _write_json(run_dir / "calibration.json", {"scope": "fit_subjects_01_18", "selected_hbo_channels": list(hbo_names), "selected_hbr_channels": [fit_trials[0].fnirs_channel_names[int(index)] for index in hbr_indices], "selected_eeg_channels": list(adapter.channel_names), **calibration})
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(_jsonable(config), sort_keys=False), encoding="utf-8")

    t3_primary: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in metric_rows:
        if row["model"] == "T3a" and row["evaluation_kind"] == "primary_center_masked":
            t3_primary[(str(row["subject"]), int(row["event_index"]))].append(float(row["nrmse"]))
    ranked = sorted(
        ((float(np.nanmean(values)), key) for key, values in t3_primary.items()),
        key=lambda item: (item[0], item[1]),
    )
    if ranked:
        score_median = float(np.median([score for score, _ in ranked]))
        representative_key = min(ranked, key=lambda item: (abs(item[0] - score_median), item[1]))[1]
    else:
        representative_key = (validation[0].trial.subject, int(validation[0].trial.event_index))
    trial_lookup = {(item.trial.subject, int(item.trial.event_index)): item for item in validation}
    representative = trial_lookup[representative_key]
    representative_predictions = {
        model: {
            mode: all_predictions[model][(representative_key[0], representative_key[1], mode)]
            for mode in MODE_IDS
        }
        for model in MODEL_IDS
    }
    with mpl.rc_context(CJK_STYLE):
        _plot_reconstruction(
            representative,
            representative_predictions,
            center,
            run_dir,
            float(config["analysis"]["sampling_hz"]),
            int(config["output"]["figure_dpi"]),
        )
        _plot_primary_metrics(subject_rows, run_dir, int(config["output"]["figure_dpi"]))
        _plot_null(null_summary_rows, run_dir, int(config["output"]["figure_dpi"]))
        _plot_shared_driver(
            representative_predictions,
            run_dir,
            float(config["analysis"]["sampling_hz"]),
            int(config["output"]["figure_dpi"]),
        )
    trial_inventory = [
        {**_identity_fields(item, condition, config["data"]), "split": split}
        for split, trials in (("fit", fit_series), ("validation", validation))
        for item in trials
    ]
    summary = {
        "schema": SCHEMA,
        "scope": "measured_development_exploratory",
        "condition_id": condition_id,
        "fit_subjects": fit_subjects,
        "validation_subjects": validation_subjects,
        "closed_protected_subjects": list(condition["protected_subjects"]),
        "fit_trial_count": len(fit_trials),
        "validation_trial_count": len(validation),
        "selected_trial_count": len(trial_inventory),
        "trial_inventory": trial_inventory,
        "models": list(MODEL_IDS),
        "modes": list(MODE_IDS),
        "nulls": list(NULL_IDS),
        "null_fit_policy": "fit_on_subjects_01_18_then_apply_to_validation_null_inputs",
        "target_semantics": "held_out_measured_signal; no clean physiological truth",
        "eeg_coordinate": "10_Hz_log_power_fit_fold_PCA_proxy; not native 200_Hz voltage waveform",
        "fnirs_coordinate": "baseline_corrected canonical robust-standardized HbO/HbR; not absolute concentration",
        "channel_selection_policy": "fit-fold MA event-response HbO selection; supervised exploratory diagnostic, not label-blind teacher fitting",
        "channel_order_checked_across_selected_trials": True,
        "primary_reconstruction": "continuous_center_block_withheld_from_target_modality_updates",
        "nrmse_definition": "RMSE divided by population SD of the same finite evaluation region; undefined at observed SD <= 1e-8",
        "aggregation_policy": "compute each trial/region metric independently; arithmetic mean over 10 trials within subject; median and IQR over 5 equally weighted subjects",
        "low_observed_variance_metric_rows": int(sum(bool(row["low_observed_variance"]) for row in metric_rows)),
        "physical_check_scope": "T3a validation windows and non-null modes only; boolean domain checks are pass/fail and numeric ranges are diagnostic",
        "physical_boolean_failures": int(sum(row["status"] == "fail" for row in physical_rows)),
        "physical_diagnostic_rows": int(sum(row["status"] == "diagnostic" for row in physical_rows)),
        "same_point_joint_status": "descriptive_posterior_fit_only",
        "uncertainty_interval": "variance_matched_Gaussian_approximation_for_cross_model_display",
        "t3a_parameter_policy": "kappa_tau_and_other_physiology_fixed; fit_subject_observation_gauges_only",
        "physiology_claim_limit": "fNIRS is a canonical standardized coordinate; no absolute concentration or OEF claim",
        "qualification_eligible": False,
        "protected_data_opened": False,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": mpl.__version__,
            "pyyaml": yaml.__version__,
        },
        "representative_trial": {"subject": representative_key[0], "event_index": representative_key[1], "selection": "closest_to_median_T3a_primary_NRMSE"},
        "loader_contracts": contracts,
        "loader_window_count_semantics": "contract window_count_by_dataset is indexed availability; selected_trial_count is actual fit+validation consumption",
        "files": [
            "manifest.json",
            "summary.json",
            "metrics.csv", "subject_metrics.csv", "summary_metrics.csv",
            "null_metrics.csv", "null_subject_metrics.csv", "null_summary.csv",
            "trajectories.csv", "physical_checks.csv", "fit_parameters.csv",
            "calibration.json", "resolved_config.yaml",
            "figures/典型窗口_EEG_fNIRS中心遮挡重建.png",
            "figures/四模型中心遮挡重建指标.png",
            "figures/Null配对特异性与共享状态泄漏.png",
            "figures/共享神经驱动与状态不确定性.png",
            "summary.md",
        ],
    }
    best_lines = []
    for target in OBS_NAMES:
        candidates = [
            row for row in summary_rows
            if row["evaluation_kind"] == "primary_center_masked" and row["target"] == target
        ]
        if candidates:
            best = min(candidates, key=lambda row: float(row["nrmse_median"]))
            best_lines.append(f"- {target}：{MODEL_LABELS[str(best['model'])]}，受试者等权中位 NRMSE={float(best['nrmse_median']):.3f}")
    (run_dir / "summary.md").write_text(
        "# 真实 EEG/fNIRS 重建与 Null 开发性诊断\n\n"
        "本结果使用 01–18 号受试者拟合，19–23 号受试者纯应用；24–29 未开放。中心遮挡是主要重建证据，同点 joint 仅为描述。每个试次独立计算局部指标，再在受试者内平均 10 个试次，最后对 5 位受试者报中位数和四分位距。EEG 是 10 Hz 功率/PCA 代理而非 200 Hz 原始电位波形；HbO/HbR 是标准化坐标而非绝对浓度。实测观测不是干净真值，本运行不具备教师合格资格。\n\n"
        "## 中心遮挡重建\n\n" + "\n".join(best_lines) +
        "\n\n## Null 解释\n\n正的 `Null NRMSE − 真实配对 NRMSE` 表示 EEG-only 预测更贴近真实配对；共享状态对供体信号的相关和相对 EEG-only 状态位移用于识别 fNIRS 泄漏。0.35 仅保留为合成实验参考，不是实测通过门。\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "summary.json", summary)
    return summary


def run(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    subject_parameter_fit: bool = False,
) -> dict[str, Any]:
    """Validate the contract, then leave an incomplete manifest on failure."""

    validate_config(config)
    output_root = (REPO_ROOT / str(config["output"]["root"])).resolve()
    resolved_run_dir = Path(run_dir).resolve()
    try:
        resolved_run_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"run directory must be below configured workspace root {output_root}") from exc
    if resolved_run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {resolved_run_dir}")
    resolved_run_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        resolved_run_dir / "manifest.json",
        {
            "schema": SCHEMA,
            "analysis_kind": "staged_subject_parameter_fit" if subject_parameter_fit else "reconstruction_and_null",
            "scope": "measured_development_exploratory",
            "completion_status": "incomplete",
            "stage": "before_data_load",
            "started_at": started_at,
            "fit_subjects": list(config["data"]["conditions"][0]["fit_subjects"]),
            "validation_subjects": list(config["data"]["conditions"][0]["validation_subjects"]),
            "closed_protected_subjects": list(config["data"]["conditions"][0]["protected_subjects"]),
            "protected_data_enabled": False,
            "protected_data_opened": False,
            "qualification_eligible": False,
        },
    )
    try:
        summary = (
            _run_parameter_fit_validated(config, resolved_run_dir)
            if subject_parameter_fit
            else _run_validated(config, resolved_run_dir)
        )
    except Exception as exc:
        _write_json(
            resolved_run_dir / "manifest.json",
            {
                "schema": SCHEMA,
                "analysis_kind": "staged_subject_parameter_fit" if subject_parameter_fit else "reconstruction_and_null",
                "scope": "measured_development_exploratory",
                "completion_status": "incomplete",
                "stage": "failed",
                "started_at": started_at,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "condition_id": config["data"]["conditions"][0]["condition_id"],
                "fit_subjects": list(config["data"]["conditions"][0]["fit_subjects"]),
                "validation_subjects": list(config["data"]["conditions"][0]["validation_subjects"]),
                "closed_protected_subjects": list(config["data"]["conditions"][0]["protected_subjects"]),
                "protected_data_enabled": False,
                "protected_data_opened": False,
                "qualification_eligible": False,
            },
        )
        raise
    _write_json(
        resolved_run_dir / "manifest.json",
        {
            "schema": SCHEMA,
            "analysis_kind": summary.get("analysis_kind", "reconstruction_and_null"),
            "scope": "measured_development_exploratory",
            "completion_status": "complete",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "condition_id": summary["condition_id"],
            "fit_subjects": list(config["data"]["conditions"][0]["fit_subjects"]),
            "validation_subjects": list(config["data"]["conditions"][0]["validation_subjects"]),
            "closed_protected_subjects": list(config["data"]["conditions"][0]["protected_subjects"]),
            "fit_trial_count": summary["fit_trial_count"],
            "validation_trial_count": summary["validation_trial_count"],
            "models": summary["models"],
            "modes": summary["modes"],
            "nulls": summary["nulls"],
            "runtime": summary["runtime"],
            "protected_data_enabled": False,
            "protected_data_opened": False,
            "qualification_eligible": False,
            "summary": "summary.json",
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--subject-parameter-fit",
        action="store_true",
        help="run staged subject-level T3a physiology fitting and validation",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_dir is None:
        root = REPO_ROOT / str(config["output"]["root"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = root / timestamp
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    summary = run(config, run_dir, subject_parameter_fit=bool(args.subject_parameter_fit))
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
