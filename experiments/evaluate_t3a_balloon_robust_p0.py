#!/usr/bin/env python3
"""Synthetic-only qualification harness for the T3a Balloon teacher.

The generator is intentionally independent of the candidate implementations:
it integrates a small Tak-style Balloon system with :func:`solve_ivp` and
never imports a teacher solver.  Candidates receive observations and masks;
the evaluator alone retains clean trajectories, parameters, and artifacts.

This module is a P0 diagnostic entry point.  It does not open measured data or
select a teacher.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.integrate import solve_ivp
from scipy.stats import kstest, norm, spearmanr, t as student_t


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/t3a_balloon_robust_p0.yaml"
SCHEMA = "t3a_balloon_robust_p0_v1"
STATE_NAMES = ("r", "s", "f", "v", "p", "q")
STATE_UNITS = {
    "r": "s^-2",
    "s": "s^-1",
    "f": "1",
    "v": "1",
    "p": "1",
    "q": "1",
}
OBS_NAMES = ("EEG", "HbO", "HbR")
NOMINAL_LEVELS = (0.50, 0.80, 0.95)


@dataclass(frozen=True)
class SyntheticCase:
    """One hidden-truth synthetic observation window."""

    replicate_id: int
    scenario_id: str
    stress_case: str
    severity: float
    null_type: str | None
    time_s: np.ndarray
    clean: np.ndarray  # [T, 3], evaluator-only clean observations
    observations: np.ndarray  # [T, 3], candidate input; NaN denotes dropout
    artifact: np.ndarray  # [T, 3], additive known nuisance
    artifact_mask: np.ndarray  # [T, 3]
    observation_mask: np.ndarray  # [T, 3]
    truth_states: np.ndarray  # [T, 6], r/s/f/v/p/q
    true_parameters: Mapping[str, float]
    geometry: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    artifact_scope: str = "none"


@dataclass
class CandidatePrediction:
    """Common candidate output; unavailable components remain NaN, not guesses."""

    model_id: str
    observation_mean: np.ndarray
    total_variance: np.ndarray
    aleatoric_variance: np.ndarray
    epistemic_variance: np.ndarray
    state_mean: np.ndarray | None = None
    state_variance: np.ndarray | None = None
    parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    distribution: str = "Gaussian"
    student_nu: float | None = None


def _copy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(config))


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = DEFAULT_CONFIG_PATH if path is None else Path(path)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("P0 config must be a mapping")
    config = _copy_config(loaded)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject measured/protected input before any data-like path is touched."""

    if str(config.get("schema", "")) != SCHEMA:
        raise ValueError("P0 config schema mismatch")
    experiment = config.get("experiment", {})
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment must be a mapping")
    if experiment.get("scope") != "synthetic_only":
        raise ValueError("T3a P0 only accepts experiment.scope=synthetic_only")
    if experiment.get("measured_data_enabled") is not False:
        raise ValueError("measured data must be disabled for P0")
    if experiment.get("protected_data_enabled") is not False:
        raise ValueError("protected data must be disabled for P0")
    if "data" in config or "cache_root" in config:
        raise ValueError("P0 config cannot declare measured data or cache paths")
    simulation = config.get("simulation", {})
    if not isinstance(simulation, Mapping):
        raise ValueError("simulation must be a mapping")
    fs_hz = float(simulation.get("sampling_hz", 0.0))
    duration = float(simulation.get("duration_s", 0.0))
    replicates = int(simulation.get("replicates", 0))
    prior_draws = int(simulation.get("prior_predictive_draws", 0))
    severities = simulation.get("severity_levels", [])
    if not np.isfinite(fs_hz) or fs_hz <= 0.0 or not np.isfinite(duration) or duration <= 4.0:
        raise ValueError("simulation sampling_hz and duration_s must be positive")
    if replicates < 2 or prior_draws < 1 or not isinstance(severities, Sequence) or not severities:
        raise ValueError("simulation requires at least two replicates plus prior_predictive_draws and severity_levels")
    if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in severities):
        raise ValueError("severity_levels must be finite and non-negative")
    if simulation.get("state_process_noise") is not False:
        raise ValueError("independent generator state process noise must be explicitly false")
    if simulation.get("truth_parameter_design") != "bounded_prior_draw":
        raise ValueError("simulation.truth_parameter_design must be bounded_prior_draw")
    physiology = config.get("physiology", {})
    if not isinstance(physiology, Mapping):
        raise ValueError("physiology must be a mapping")
    fixed = physiology.get("fixed", {})
    truth = physiology.get("truth", {})
    free = physiology.get("free", {})
    if not isinstance(fixed, Mapping) or not isinstance(truth, Mapping) or not isinstance(free, Mapping):
        raise ValueError("physiology fixed/truth/free sections are required")
    if set(free) != {"kappa_per_s", "tau_s"}:
        raise ValueError("P0 free parameter set must be exactly kappa_per_s and tau_s")
    tau_v = float(fixed.get("tau_v_s", float("nan")))
    if not np.isfinite(tau_v) or abs(tau_v) > 1e-12:
        raise ValueError("P0 uses the declared minimal Balloon model with tau_v_s=0")
    alpha = float(fixed.get("alpha", float("nan")))
    e0 = float(fixed.get("e0", float("nan")))
    if not np.isfinite(alpha) or alpha <= 0.0 or not np.isfinite(e0) or not 0.0 < e0 < 1.0:
        raise ValueError("fixed alpha and e0 violate the Balloon domain")
    fixed_positive = np.asarray(
        [fixed.get("gamma"), fixed.get("p0"), fixed.get("q0")], dtype=np.float64
    )
    if not np.all(np.isfinite(fixed_positive)) or np.any(fixed_positive <= 0.0):
        raise ValueError("fixed gamma/p0/q0 must be positive")
    if float(fixed.get("q0", 0.0)) > float(fixed.get("p0", 0.0)):
        raise ValueError("q0 must not exceed p0")
    for name in ("kappa_per_s", "tau_s"):
        spec = free[name]
        if not isinstance(spec, Mapping) or len(spec.get("bounds", [])) != 2:
            raise ValueError(f"free prior for {name} must contain two bounds")
        lower, upper = [float(value) for value in spec["bounds"]]
        value = float(truth.get(name, float("nan")))
        if not np.isfinite(lower + upper + value) or lower <= 0.0 or upper <= lower or not lower <= value <= upper:
            raise ValueError(f"truth/bounds invalid for {name}")
        prior_sd = float(spec.get("prior_sd", float("nan")))
        if not np.isfinite(prior_sd) or prior_sd <= 0.0:
            raise ValueError(f"prior_sd must be positive for {name}")
        prior_mean = float(spec.get("prior_mean", float("nan")))
        if not np.isfinite(prior_mean) or not lower <= prior_mean <= upper:
            raise ValueError(f"prior_mean must lie inside bounds for {name}")
    observation = config.get("observation", {})
    student_df = float(observation.get("student_t_df", float("nan"))) if isinstance(observation, Mapping) else float("nan")
    if not isinstance(observation, Mapping) or not np.isfinite(student_df) or student_df <= 2.0:
        raise ValueError("observation.student_t_df must exceed two")
    if tuple(observation.get("coordinates", ())) != OBS_NAMES:
        raise ValueError("observation.coordinates must be (EEG, HbO, HbR)")
    eeg_loading = float(observation.get("eeg_loading", float("nan")))
    if not np.isfinite(eeg_loading) or eeg_loading <= 0.0 or not np.isfinite(float(observation.get("eeg_offset", float("nan")))):
        raise ValueError("EEG observation loading must be positive and offset finite")
    scales = observation.get("scale", {})
    scale_values = (
        np.asarray([scales.get(name, float("nan")) for name in OBS_NAMES], dtype=np.float64)
        if isinstance(scales, Mapping)
        else np.asarray([], dtype=np.float64)
    )
    if scale_values.shape != (3,) or not np.all(np.isfinite(scale_values)) or np.any(scale_values <= 0.0):
        raise ValueError("observation scales for EEG/HbO/HbR must be positive")
    stress = config.get("stress_tests", {})
    if not isinstance(stress, Mapping) or not isinstance(stress.get("artifact_families"), Sequence) or not stress.get("artifact_families"):
        raise ValueError("stress_tests.artifact_families is required")
    if not isinstance(stress.get("nulls"), Sequence) or not stress.get("nulls"):
        raise ValueError("stress_tests.nulls is required")
    unknown_artifacts = set(str(value) for value in stress["artifact_families"]) - {"spike", "drift", "step", "high_frequency_burst", "dropout"}
    unknown_nulls = set(str(value) for value in stress["nulls"]) - {"independent", "time_shift", "pairing"}
    if unknown_artifacts or unknown_nulls:
        raise ValueError("stress_tests contains an unsupported artifact or null")
    scopes = stress.get("artifact_scopes", {})
    allowed_scopes = {"eeg_only", "fnirs_only", "systemic_fnirs", "composite_missing", "systemic"}
    if not isinstance(scopes, Mapping):
        raise ValueError("stress_tests.artifact_scopes is required")
    if any(str(family) not in scopes for family in stress["artifact_families"]):
        raise ValueError("artifact_scopes must define every configured family")
    if any(str(scope) not in allowed_scopes for scope in scopes.values()):
        raise ValueError("artifact_scopes contains an unsupported scope")
    dropout_fraction = float(stress.get("dropout_fraction", float("nan")))
    if not np.isfinite(dropout_fraction) or not 0.0 < dropout_fraction < 1.0:
        raise ValueError("dropout_fraction must lie strictly between zero and one")
    time_shift_s = float(stress.get("time_shift_s", float("nan")))
    if not np.isfinite(time_shift_s) or time_shift_s <= 0.0:
        raise ValueError("time_shift_s must be finite and positive")
    process_sd = config.get("observation", {}).get("process_sd", {})
    if not isinstance(process_sd, Mapping) or any(
        not np.isfinite(float(process_sd.get(name, float("nan")))) or float(process_sd.get(name, 0.0)) < 0.0
        for name in ("r", "s", "log_f", "log_v", "log_p", "log_q")
    ):
        raise ValueError("observation.process_sd must declare non-negative finite entries for all states")
    driver = config.get("driver", {})
    if (
        not isinstance(driver, Mapping)
        or not np.isfinite(float(driver.get("decay_per_s", float("nan"))))
        or float(driver.get("decay_per_s", 0.0)) <= 0.0
        or not np.isfinite(float(driver.get("diffusion_sd_per_sqrt_s", float("nan"))))
        or float(driver.get("diffusion_sd_per_sqrt_s", 0.0)) < 0.0
        or not np.isfinite(float(driver.get("pulse_amplitude", float("nan"))))
    ):
        raise ValueError("driver decay/diffusion/pulse values are invalid")
    starts = config.get("inference", {}).get("starts", ())
    if not isinstance(starts, Sequence) or not starts:
        raise ValueError("inference.starts must contain at least one explicit kappa/tau start")
    for start in starts:
        if not isinstance(start, Sequence) or len(start) != 2 or not np.all(np.isfinite(np.asarray(start, dtype=np.float64))):
            raise ValueError("every inference start must be a finite [kappa, tau] pair")
        if not (float(free["kappa_per_s"]["bounds"][0]) <= float(start[0]) <= float(free["kappa_per_s"]["bounds"][1])):
            raise ValueError("inference kappa start lies outside bounds")
        if not (float(free["tau_s"]["bounds"][0]) <= float(start[1]) <= float(free["tau_s"]["bounds"][1])):
            raise ValueError("inference tau start lies outside bounds")
    gates = config.get("gates", {})
    required_gates = {
        "max_solver_fail_fraction": (0.0, 1.0),
        "max_boundary_contact_fraction": (0.0, 1.0),
        "min_response_amplitude": (0.0, float("inf")),
        "max_response_amplitude": (0.0, float("inf")),
        "min_delay_s": (0.0, float("inf")),
        "max_delay_s": (0.0, float("inf")),
        "min_sbc_replicates": (2.0, float("inf")),
        "min_sbc_ks_pvalue": (0.0, 1.0),
        "min_parameter_coverage_95": (0.0, 1.0),
        "max_multistart_spread_fraction": (0.0, 1.0),
        "max_driver_nrmse": (0.0, float("inf")),
        "max_off_artifact_distortion": (0.0, float("inf")),
        "min_artifact_attenuation": (-float("inf"), 1.0),
        "max_artifact_residual_relative_rmse": (0.0, float("inf")),
        "min_uncertainty_increase_artifact": (0.0, float("inf")),
        "min_uncertainty_increase_missing": (0.0, float("inf")),
        "min_predictive_coverage_95": (0.0, 1.0),
        "max_predictive_coverage_95": (0.0, 1.0),
        "min_uncertainty_risk_spearman": (-1.0, 1.0),
        "max_normalized_crps": (0.0, float("inf")),
        "max_standardized_nll": (-float("inf"), float("inf")),
        "max_reconstruction_temporal_acf_error": (0.0, float("inf")),
        "max_reconstruction_spectral_shape_error": (0.0, float("inf")),
        "max_null_driver_correlation": (0.0, 1.0),
    }
    if not isinstance(gates, Mapping):
        raise ValueError("gates must be a mapping")
    for name, (lower, upper) in required_gates.items():
        value = float(gates.get(name, float("nan")))
        if not np.isfinite(value) or value < lower or value > upper:
            raise ValueError(f"gate {name} is missing or outside its declared range")
    if int(gates["min_sbc_replicates"]) != float(gates["min_sbc_replicates"]):
        raise ValueError("gate min_sbc_replicates must be an integer")
    if float(gates["min_response_amplitude"]) > float(gates["max_response_amplitude"]):
        raise ValueError("response-amplitude gates are not ordered")
    if float(gates["min_delay_s"]) > float(gates["max_delay_s"]):
        raise ValueError("delay gates are not ordered")
    if float(gates["min_predictive_coverage_95"]) > float(gates["max_predictive_coverage_95"]):
        raise ValueError("predictive-coverage gates are not ordered")
    ledger = physiology.get("source_ledger")
    if not isinstance(ledger, Mapping) or "model_family" not in ledger:
        raise ValueError("physiology.source_ledger.model_family is required")
    required_ledger_fields = {"primary_source", "url", "scope", "species", "population", "challenge"}
    if not required_ledger_fields.issubset(set(ledger["model_family"])):
        raise ValueError("source ledger model_family lacks required provenance fields")
    for name in ("alpha", "e0", "gamma", "kappa", "tau", "p0_q0"):
        entry = ledger.get(name)
        if not isinstance(entry, Mapping) or not {"unit", "role", "compartment", "primary_source_ref", "context"}.issubset(set(entry)):
            raise ValueError(f"source ledger entry incomplete: {name}")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        field_names: list[str] = []
        for row in rows:
            for key in row:
                if key not in field_names:
                    field_names.append(key)
    else:
        field_names = list(fields)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in field_names})
    os.replace(temporary, path)


def _parameter_values(config: Mapping[str, Any], sampled: Mapping[str, float] | None = None) -> dict[str, float]:
    fixed = config["physiology"]["fixed"]
    truth = config["physiology"]["truth"]
    values = {
        "kappa_per_s": float(truth["kappa_per_s"]),
        "tau_s": float(truth["tau_s"]),
        "alpha": float(fixed["alpha"]),
        "e0": float(fixed["e0"]),
        "gamma": float(fixed["gamma"]),
        "p0": float(fixed["p0"]),
        "q0": float(fixed["q0"]),
        "tau_v_s": float(fixed.get("tau_v_s", 0.0)),
    }
    if sampled:
        values.update({str(key): float(value) for key, value in sampled.items()})
    return values


def _sample_free_parameters(
    rng: np.random.Generator, config: Mapping[str, Any]
) -> dict[str, float]:
    """Draw the two free parameters from their declared bounded priors."""

    sampled: dict[str, float] = {}
    for name, spec in config["physiology"]["free"].items():
        lower, upper = (float(value) for value in spec["bounds"])
        for _ in range(10_000):
            value = float(rng.normal(float(spec["prior_mean"]), float(spec["prior_sd"])))
            if lower <= value <= upper:
                sampled[name] = value
                break
        else:
            raise RuntimeError(f"bounded prior rejection failed for {name}")
    return sampled


def _driver(time_s: np.ndarray, rng: np.random.Generator, config: Mapping[str, Any]) -> np.ndarray:
    spec = config["driver"]
    dt = float(time_s[1] - time_s[0])
    fs_hz = 1.0 / max(dt, 1e-12)
    phi = math.exp(-float(spec["decay_per_s"]) / fs_hz)
    values = np.zeros(len(time_s), dtype=np.float64)
    diffusion_sd = float(spec["diffusion_sd_per_sqrt_s"])
    for index in range(1, len(values)):
        values[index] = phi * values[index - 1] + rng.normal(scale=diffusion_sd * math.sqrt(dt))
    center = 0.52 * float(time_s[-1])
    width = max(0.8, 0.06 * float(time_s[-1]))
    values += float(spec["pulse_amplitude"]) * np.exp(-0.5 * ((time_s - center) / width) ** 2)
    return values


def _balloon_trajectory(
    time_s: np.ndarray,
    driver: np.ndarray,
    parameters: Mapping[str, float],
) -> tuple[np.ndarray, bool]:
    """Independent solve_ivp implementation of the minimal Tak equations.

    The p equation deliberately follows Eq. 3 in Tak et al.:
    ``tau*dp/dt = f - f_out*p/v``.  The viscoelastic ``tau_v`` extension is
    fixed to zero, so ``f_out = v**(1/alpha)``.
    """

    tau = float(parameters["tau_s"])
    alpha = float(parameters["alpha"])
    e0 = float(parameters["e0"])
    kappa = float(parameters["kappa_per_s"])
    gamma = float(parameters["gamma"])
    if not np.isfinite(tau + alpha + e0 + kappa + gamma) or tau <= 0.0 or alpha <= 0.0 or kappa <= 0.0 or gamma <= 0.0 or not 0.0 < e0 < 1.0:
        raise ValueError("invalid Balloon parameters")

    def rhs(current_time: float, state: np.ndarray) -> np.ndarray:
        s, f, v, p, q = state
        if not np.isfinite(s + f + v + p + q) or f <= 0.0 or v <= 0.0 or p <= 0.0 or q <= 0.0:
            raise FloatingPointError("truth integration crossed a positive Balloon boundary")
        f_out = float(v ** (1.0 / alpha))
        log_one_minus = float(np.log1p(-e0))
        extraction = float(-np.expm1(log_one_minus / f))
        drive = float(np.interp(current_time, time_s, driver))
        return np.asarray(
            [
                drive - kappa * s - gamma * (f - 1.0),
                s,
                (f - f_out) / tau,
                (f - f_out * p / v) / tau,
                (f * extraction / e0 - f_out * q / v) / tau,
            ],
            dtype=np.float64,
        )

    try:
        result = solve_ivp(
            rhs,
            (float(time_s[0]), float(time_s[-1])),
            np.asarray([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            t_eval=time_s,
            rtol=1e-8,
            atol=1e-10,
            max_step=float(time_s[1] - time_s[0]) / 4.0,
        )
    except (FloatingPointError, ValueError, OverflowError):
        return np.full((len(time_s), 6), np.nan), False
    if not result.success or result.y.shape != (5, len(time_s)):
        return np.full((len(time_s), 6), np.nan), False
    states = np.column_stack((driver, result.y.T))
    valid = np.all(np.isfinite(states))
    valid &= np.all(states[:, 2:6] > 0.0)
    return states, bool(valid)


def _artifact(
    clean: np.ndarray,
    time_s: np.ndarray,
    family: str,
    severity: float,
    rng: np.random.Generator,
    dropout_fraction: float,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return additive artifact, artifact mask, and missing-observation mask."""

    length = len(time_s)
    artifact = np.zeros_like(clean, dtype=np.float64)
    mask = np.zeros_like(clean, dtype=bool)
    missing = np.zeros_like(clean, dtype=bool)
    start = max(1, int(round(0.30 * length)))
    stop = min(length, max(start + 3, int(round(0.42 * length))))
    support = np.arange(start, stop)
    scale = np.maximum(np.std(clean, axis=0), 1e-6)
    if family == "dropout":
        count = max(1, int(round(float(dropout_fraction) * length)))
        begin = int(rng.integers(0, max(length - count, 1)))
        channels = np.arange(clean.shape[1]) if scope == "composite_missing" else np.asarray([0, 1, 2])
        missing[begin : begin + count, channels] = True
        return artifact, mask, missing
    if scope == "eeg_only":
        channels = np.asarray([0])
    elif scope in {"fnirs_only", "systemic_fnirs"}:
        channels = np.asarray([1, 2])
    elif scope in {"systemic", "composite_missing"}:
        channels = np.arange(clean.shape[1])
    else:
        raise ValueError(f"unknown artifact scope: {scope}")
    mask[np.ix_(support, channels)] = True
    phase = np.linspace(0.0, 1.0, len(support), endpoint=False)
    if family == "spike":
        picks = support[:: max(1, len(support) // 3)]
        artifact[np.ix_(picks, channels)] = float(severity) * 5.0 * scale[None, channels]
        if len(channels) > 1:
            artifact[np.ix_(picks[::2], channels[1:])] *= -1.0
    elif family == "drift":
        shape = (phase - 0.5) + 0.25 * np.sin(2.0 * np.pi * phase)
        artifact[np.ix_(support, channels)] = float(severity) * 3.0 * shape[:, None] * scale[None, channels]
    elif family == "step":
        artifact[np.ix_(support, channels)] = float(severity) * 2.0 * scale[None, channels]
        if 2 in channels:
            artifact[support, 2] *= -1.0
    elif family == "high_frequency_burst":
        shape = np.sin(2.0 * np.pi * 0.40 * np.arange(len(support)))
        artifact[np.ix_(support, channels)] = float(severity) * 3.0 * shape[:, None] * scale[None, channels]
    else:
        raise ValueError(f"unknown artifact family: {family}")
    return artifact, mask, missing


def generate_case(
    replicate_id: int,
    seed: int,
    config: Mapping[str, Any],
    *,
    sampled_parameters: Mapping[str, float] | None = None,
) -> SyntheticCase:
    simulation = config["simulation"]
    fs_hz = float(simulation["sampling_hz"])
    count = max(int(round(float(simulation["duration_s"]) * fs_hz)), 8)
    time_s = np.arange(count, dtype=np.float64) / fs_hz
    rng = np.random.default_rng(int(seed))
    parameters = _parameter_values(config, sampled_parameters)
    driver = _driver(time_s, rng, config)
    truth_states, solver_ok = _balloon_trajectory(time_s, driver, parameters)
    if not solver_ok:
        raise RuntimeError("independent Balloon truth integration failed")
    p0 = parameters["p0"]
    q0 = parameters["q0"]
    delta_hbt = p0 * (truth_states[:, 4] - 1.0)
    delta_hbr = q0 * (truth_states[:, 5] - 1.0)
    clean = np.column_stack((driver, delta_hbt - delta_hbr, delta_hbr))
    noise_df = max(float(config["observation"].get("student_t_df", 5.0)), 2.1)
    noise_scale = np.asarray(
        [float(config["observation"]["scale"].get(name, 0.05)) for name in OBS_NAMES],
        dtype=np.float64,
    )
    hetero = 1.0 + 0.45 * np.abs((clean - np.mean(clean, axis=0)) / np.maximum(np.std(clean, axis=0), 1e-8))
    noise = student_t.rvs(noise_df, size=clean.shape, random_state=rng) * noise_scale[None, :] * hetero
    artifact = np.zeros_like(clean)
    artifact_mask = np.zeros_like(clean, dtype=bool)
    observations = clean + noise
    observation_mask = np.isfinite(observations)
    # This synthetic contract has three modality coordinates, not sensor
    # locations.  Keep geometry explicitly empty so downstream renderers do
    # not infer a fictitious whole-brain spatial map.
    geometry = np.empty((0, 3), dtype=np.float64)
    return SyntheticCase(
        replicate_id=int(replicate_id),
        scenario_id="clean",
        stress_case="clean",
        severity=0.0,
        null_type=None,
        time_s=time_s,
        clean=clean,
        observations=observations,
        artifact=artifact,
        artifact_mask=artifact_mask,
        observation_mask=observation_mask,
        truth_states=truth_states,
        true_parameters=parameters,
        geometry=geometry,
        artifact_scope="none",
    )


def make_null_case(a: SyntheticCase, b: SyntheticCase, null_type: str, config: Mapping[str, Any]) -> SyntheticCase:
    """Preserve marginals while breaking EEG--fNIRS shared timing."""

    if null_type not in {"independent", "time_shift", "pairing"}:
        raise ValueError(f"unknown null type: {null_type}")
    observations = np.asarray(a.observations, dtype=np.float64).copy()
    clean = np.asarray(a.clean, dtype=np.float64).copy()
    artifact = np.asarray(a.artifact, dtype=np.float64).copy()
    artifact_mask = np.asarray(a.artifact_mask, dtype=bool).copy()
    observation_mask = np.asarray(a.observation_mask, dtype=bool).copy()
    if null_type in {"independent", "pairing"}:
        observations[:, 1:] = b.observations[:, 1:]
        clean[:, 1:] = b.clean[:, 1:]
        artifact[:, 1:] = b.artifact[:, 1:]
        artifact_mask[:, 1:] = b.artifact_mask[:, 1:]
        observation_mask[:, 1:] = b.observation_mask[:, 1:]
    else:
        shift = int(round(float(config["stress_tests"].get("time_shift_s", 20.0)) * (1.0 / (a.time_s[1] - a.time_s[0]))))
        shift = max(1, min(abs(shift), len(observations) - 1))
        observations[:, 1:] = np.roll(observations[:, 1:], shift, axis=0)
        clean[:, 1:] = np.roll(clean[:, 1:], shift, axis=0)
        artifact[:, 1:] = np.roll(artifact[:, 1:], shift, axis=0)
        artifact_mask[:, 1:] = np.roll(artifact_mask[:, 1:], shift, axis=0)
        observation_mask[:, 1:] = np.roll(observation_mask[:, 1:], shift, axis=0)
    observations[~observation_mask] = np.nan
    return SyntheticCase(
        replicate_id=a.replicate_id,
        scenario_id=f"null_{null_type}",
        stress_case="null",
        severity=1.0,
        null_type=null_type,
        time_s=a.time_s.copy(),
        clean=clean,
        observations=observations,
        artifact=artifact,
        artifact_mask=artifact_mask,
        observation_mask=observation_mask,
        truth_states=a.truth_states.copy(),
        true_parameters=a.true_parameters,
        geometry=a.geometry.copy(),
        artifact_scope="none",
    )


def _derive_stress_case(
    base: SyntheticCase,
    config: Mapping[str, Any],
    family: str,
    severity: float,
    seed: int,
) -> SyntheticCase:
    """Apply nuisance corruption to a base case without changing its truth/noise."""

    scope = str(config["stress_tests"].get("artifact_scopes", {}).get(family, "systemic"))
    rng = np.random.default_rng(int(seed))
    artifact, artifact_mask, missing = _artifact(
        base.clean,
        base.time_s,
        family,
        severity,
        rng,
        float(config["stress_tests"].get("dropout_fraction", 0.15)),
        scope,
    )
    observations = base.observations + artifact
    observation_mask = np.isfinite(observations) & ~missing
    observations = observations.copy()
    observations[~observation_mask] = np.nan
    return SyntheticCase(
        replicate_id=base.replicate_id,
        scenario_id=f"{family}_s{severity:g}",
        stress_case=family,
        severity=float(severity),
        null_type=None,
        time_s=base.time_s.copy(),
        clean=base.clean.copy(),
        observations=observations,
        artifact=artifact,
        artifact_mask=artifact_mask,
        observation_mask=observation_mask,
        truth_states=base.truth_states.copy(),
        true_parameters=base.true_parameters,
        geometry=np.empty((0, 3), dtype=np.float64),
        artifact_scope=scope,
    )


def _derive_missing_modality_case(base: SyntheticCase, modality: str) -> SyntheticCase:
    if modality not in {"missing_eeg", "missing_fnirs"}:
        raise ValueError(f"unknown missing modality: {modality}")
    observations = base.observations.copy()
    observation_mask = base.observation_mask.copy()
    channels = np.asarray([0]) if modality == "missing_eeg" else np.asarray([1, 2])
    observation_mask[:, channels] = False
    observations[:, channels] = np.nan
    return SyntheticCase(
        replicate_id=base.replicate_id,
        scenario_id=modality,
        stress_case=modality,
        severity=1.0,
        null_type=None,
        time_s=base.time_s.copy(),
        clean=base.clean.copy(),
        observations=observations,
        artifact=np.zeros_like(base.clean),
        artifact_mask=np.zeros_like(base.artifact_mask),
        observation_mask=observation_mask,
        truth_states=base.truth_states.copy(),
        true_parameters=base.true_parameters,
        geometry=np.empty((0, 3), dtype=np.float64),
        artifact_scope=modality,
    )


def _finite_mean(values: np.ndarray, default: float = float("nan")) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else default


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    left, right = left[valid], right[valid]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rmse(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray | None = None) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    valid = np.isfinite(reference) & np.isfinite(estimate)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return float(np.sqrt(np.mean((reference[valid] - estimate[valid]) ** 2))) if np.any(valid) else float("nan")


def _nrmse(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    scale = max(float(np.std(np.asarray(reference)[valid])), 1e-8)
    return _rmse(reference, estimate, valid) / scale


def _empty_prediction(model_id: str, length: int) -> CandidatePrediction:
    empty_obs = np.full((length, 3), np.nan, dtype=np.float64)
    return CandidatePrediction(
        model_id=model_id,
        observation_mean=empty_obs,
        total_variance=empty_obs.copy(),
        aleatoric_variance=empty_obs.copy(),
        epistemic_variance=empty_obs.copy(),
        metadata={"status": "not_available"},
    )


class CandidateAdapter:
    model_id = "candidate"

    def fit(self, cases: Sequence[SyntheticCase], config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def predict(self, fit: Any, case: SyntheticCase, config: Mapping[str, Any]) -> CandidatePrediction:
        raise NotImplementedError


class T0PersistenceAdapter(CandidateAdapter):
    model_id = "T0-native"

    def fit(self, cases: Sequence[SyntheticCase], config: Mapping[str, Any]) -> np.ndarray:
        differences = []
        for case in cases:
            values = case.observations
            differences.append(np.diff(values, axis=0))
        stacked = np.concatenate(differences, axis=0)
        variance = np.nanvar(stacked, axis=0) * 0.5
        return np.maximum(np.nan_to_num(variance, nan=0.05), 1e-6)

    def predict(self, fit: np.ndarray, case: SyntheticCase, config: Mapping[str, Any]) -> CandidatePrediction:
        values = case.observations
        mean = np.zeros_like(values, dtype=np.float64)
        for channel in range(values.shape[1]):
            previous = 0.0
            for index, value in enumerate(values[:, channel]):
                mean[index, channel] = previous
                if np.isfinite(value):
                    previous = float(value)
        variance = np.broadcast_to(np.asarray(fit)[None, :], mean.shape).copy()
        unavailable = np.full_like(variance, np.nan)
        return CandidatePrediction(self.model_id, mean, variance, unavailable, unavailable, metadata={"state_status": "not_applicable"})


def _fit_lds_1d(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values)
    pairs = valid[1:] & valid[:-1]
    if np.count_nonzero(pairs) < 3:
        return 0.9, 0.05, 0.05
    previous, current = values[:-1][pairs], values[1:][pairs]
    coefficient = float(np.dot(previous, current) / max(float(np.dot(previous, previous)), 1e-8))
    coefficient = float(np.clip(coefficient, -0.995, 0.995))
    residual = current - coefficient * previous
    process = max(float(np.var(residual)), 1e-6)
    observation = max(float(np.nanvar(np.diff(values[valid])) * 0.5), 1e-6)
    return coefficient, process, observation


def _kalman_smooth_1d(values: np.ndarray, coefficient: float, process: float, observation: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    length = len(values)
    filtered_mean = np.zeros(length, dtype=np.float64)
    filtered_var = np.zeros(length, dtype=np.float64)
    predicted_mean = np.zeros(length, dtype=np.float64)
    predicted_var = np.zeros(length, dtype=np.float64)
    mean = 0.0
    variance = max(process / max(1.0 - coefficient**2, 1e-4), 1e-5)
    for index, value in enumerate(values):
        if index:
            mean = coefficient * mean
            variance = coefficient**2 * variance + process
        predicted_mean[index], predicted_var[index] = mean, variance
        if np.isfinite(value):
            gain = variance / max(variance + observation, 1e-8)
            mean += gain * (float(value) - mean)
            variance = max((1.0 - gain) * variance, 1e-8)
        filtered_mean[index], filtered_var[index] = mean, variance
    smooth_mean, smooth_var = filtered_mean.copy(), filtered_var.copy()
    for index in range(length - 2, -1, -1):
        denominator = max(predicted_var[index + 1], 1e-8)
        gain = filtered_var[index] * coefficient / denominator
        smooth_mean[index] += gain * (smooth_mean[index + 1] - predicted_mean[index + 1])
        smooth_var[index] = max(filtered_var[index] + gain**2 * (smooth_var[index + 1] - predicted_var[index + 1]), 1e-8)
    return smooth_mean, smooth_var


class T1IndependentLDSAdapter(CandidateAdapter):
    model_id = "T1-self"

    def fit(self, cases: Sequence[SyntheticCase], config: Mapping[str, Any]) -> list[tuple[float, float, float]]:
        return [_fit_lds_1d(np.concatenate([case.observations[:, channel] for case in cases])) for channel in range(3)]

    def predict(self, fit: list[tuple[float, float, float]], case: SyntheticCase, config: Mapping[str, Any]) -> CandidatePrediction:
        means, variances = [], []
        for channel, values in enumerate(case.observations.T):
            mean, variance = _kalman_smooth_1d(values, *fit[channel])
            means.append(mean)
            variances.append(variance)
        mean = np.column_stack(means)
        total = np.column_stack(variances)
        unavailable = np.full_like(total, np.nan)
        return CandidatePrediction(self.model_id, mean, total, unavailable, unavailable, metadata={"state_status": "not_applicable", "shared_driver_status": "not_applicable"})


class T2bLegacyAdapter(CandidateAdapter):
    model_id = "T2b-adaptive-legacy"

    def fit(self, cases: Sequence[SyntheticCase], config: Mapping[str, Any]) -> Any:
        from src.inference.adaptive_neurovascular_ssm import fit_adaptive_ssm

        ssm = {
            "fs_hz": float(config["simulation"]["sampling_hz"]),
            "prior_strength": 0.03,
            "max_iterations": int(config.get("inference", {}).get("max_iterations", 60)),
            "q_scale_candidates": (1.0,),
            "fnirs_noise_scale_candidates": (1.0,),
            "balance_penalty": 0.25,
            "max_flow_perturbation": 0.25,
        }
        return fit_adaptive_ssm(
            [case.observations[:, 0] for case in cases],
            [case.observations[:, 1] for case in cases],
            [case.observations[:, 2] for case in cases],
            fs_hz=ssm["fs_hz"],
            prior_strength=ssm["prior_strength"],
            max_iterations=ssm["max_iterations"],
            q_scale_candidates=ssm["q_scale_candidates"],
            fnirs_noise_scale_candidates=ssm["fnirs_noise_scale_candidates"],
            balance_penalty=ssm["balance_penalty"],
            max_flow_perturbation=ssm["max_flow_perturbation"],
        )

    def predict(self, fit: Any, case: SyntheticCase, config: Mapping[str, Any]) -> CandidatePrediction:
        from src.inference.adaptive_neurovascular_ssm import apply_adaptive_ssm

        result = apply_adaptive_ssm(
            case.observations[:, 0],
            fit,
            hbo_observation=case.observations[:, 1],
            hbr_observation=case.observations[:, 2],
        )
        mean = np.column_stack((result.eeg_reconstructed, result.hbo_reconstructed, result.hbr_reconstructed))
        total = np.square(np.asarray(result.observation_predictive_std, dtype=np.float64))
        unavailable = np.full_like(total, np.nan)
        states = np.full((len(mean), 6), np.nan, dtype=np.float64)
        state_variance = np.full_like(states, np.nan)
        states[:, 0] = result.states[:, 4]  # operational r only; legacy dynamics are not p/q
        state_variance[:, 0] = np.square(result.state_std[:, 4])
        return CandidatePrediction(
            self.model_id,
            mean,
            total,
            unavailable,
            unavailable,
            states,
            state_variance,
            metadata={
                "state_status": "legacy_r_only",
                "physical_status": "not_eligible",
                "uncertainty_decomposition": "not_available",
            },
        )


def _mapping_value(value: Any, names: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


class T3aRobustAdapter(CandidateAdapter):
    model_id = "T3a-balloon-robust"

    def __init__(self) -> None:
        self.module: Any | None = None
        self.import_error: str | None = None
        try:
            from src.inference import t3a_balloon_robust_ssm as module

            self.module = module
        except ImportError as exc:
            self.module = None
            self.import_error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.module is not None and callable(getattr(self.module, "fit_balloon", None))

    def fit(self, cases: Sequence[SyntheticCase], config: Mapping[str, Any]) -> Any:
        if not self.available:
            return None
        if len(cases) != 1:
            raise ValueError("T3a fit accepts exactly one synthetic case")
        module = self.module
        assert module is not None
        fixed_cls = module.BalloonFixedParameters
        spec_cls = module.BalloonObservationSpec
        config_cls = module.BalloonConfig
        simulation = config["simulation"]
        physiology = config["physiology"]
        fixed_cfg = physiology["fixed"]
        observation_cfg = config["observation"]
        process_cfg = observation_cfg.get("process_sd", {})
        process_std = tuple(
            float(process_cfg.get(key, 0.02))
            for key in ("r", "s", "log_f", "log_v", "log_p", "log_q")
        )
        scale = tuple(float(observation_cfg["scale"][name]) for name in OBS_NAMES)
        fixed = fixed_cls(
            alpha=float(fixed_cfg["alpha"]),
            E0=float(fixed_cfg["e0"]),
            gamma=float(fixed_cfg["gamma"]),
            P0=float(fixed_cfg["p0"]),
            Q0=float(fixed_cfg["q0"]),
            driver_decay_per_s=float(config["driver"]["decay_per_s"]),
            process_std=process_std,
            observation_scale=scale,
            student_nu=float(observation_cfg["student_t_df"]),
            eeg_loading=float(observation_cfg.get("eeg_loading", 1.0)),
            eeg_offset=float(observation_cfg.get("eeg_offset", 0.0)),
        )
        inference = config.get("inference", {})
        free_cfg = physiology["free"]
        configured_starts = inference.get("starts", ())
        if configured_starts:
            starts = tuple(
                (float(item[0]), float(item[1]))
                for item in configured_starts
            )
        else:
            # The prior centre is a deterministic fallback only when a
            # configuration omits an explicit multi-start panel.
            starts = ((
                float(free_cfg["kappa_per_s"]["prior_mean"]),
                float(free_cfg["tau_s"]["prior_mean"]),
            ),)
        starts = starts[: max(1, int(inference.get("multistarts", len(starts))))]
        balloon_config = config_cls(
            dt=1.0 / float(simulation["sampling_hz"]),
            irls_iterations=int(inference.get("irls_iterations", 3)),
            optimizer_max_iterations=int(inference.get("max_iterations", 60)),
            optimizer_starts=starts,
            kappa_bounds=tuple(float(item) for item in free_cfg["kappa_per_s"]["bounds"]),
            tau_bounds=tuple(float(item) for item in free_cfg["tau_s"]["bounds"]),
            kappa_prior_mean=float(free_cfg["kappa_per_s"]["prior_mean"]),
            kappa_prior_sd=float(free_cfg["kappa_per_s"]["prior_sd"]),
            tau_prior_mean=float(free_cfg["tau_s"]["prior_mean"]),
            tau_prior_sd=float(free_cfg["tau_s"]["prior_sd"]),
            hessian_step=float(inference.get("finite_difference_step", 1e-3)),
        )
        spec = spec_cls(
            eeg_loading=float(observation_cfg.get("eeg_loading", 1.0)),
            eeg_offset=float(observation_cfg.get("eeg_offset", 0.0)),
            observation_scale=scale,
            student_nu=float(observation_cfg["student_t_df"]),
        )
        case = cases[0]
        return module.fit_balloon(
            case.observations,
            fixed=fixed,
            observation_spec=spec,
            config=balloon_config,
            observation_mask=case.observation_mask,
            starts=starts,
        )

    def predict(self, fit: Any, case: SyntheticCase, config: Mapping[str, Any]) -> CandidatePrediction:
        if not self.available or fit is None:
            return _empty_prediction(self.model_id, len(case.time_s))
        module = self.module
        assert module is not None
        raw = module.smooth_balloon(
            case.observations,
            parameters=fit.parameters,
            observation_spec=module.BalloonObservationSpec(
                eeg_loading=float(config["observation"].get("eeg_loading", 1.0)),
                eeg_offset=float(config["observation"].get("eeg_offset", 0.0)),
                observation_scale=tuple(float(config["observation"]["scale"][name]) for name in OBS_NAMES),
                student_nu=float(config["observation"]["student_t_df"]),
            ),
            config=module.BalloonConfig(
                dt=1.0 / float(config["simulation"]["sampling_hz"]),
                irls_iterations=int(config.get("inference", {}).get("irls_iterations", 3)),
                optimizer_max_iterations=int(config.get("inference", {}).get("max_iterations", 60)),
            ),
            observation_mask=case.observation_mask,
        )
        prediction = self._coerce(raw, len(case.time_s))
        prediction.parameters = fit.parameter_summary if hasattr(fit, "parameter_summary") else {}
        physical_checks = _mapping_value(raw, ("physical_checks",))
        if isinstance(physical_checks, Mapping):
            prediction.metadata = {**prediction.metadata, "physical_checks": physical_checks}
        prediction.distribution = "Student-t"
        prediction.student_nu = float(config["observation"]["student_t_df"])
        return prediction

    def _coerce(self, raw: Any, length: int) -> CandidatePrediction:
        observation_mean = _mapping_value(raw, ("observation_mean", "trajectory_mean", "reconstruction", "reconstructed", "mean"))
        if observation_mean is None:
            channels = [_mapping_value(raw, (f"{name.lower()}_reconstructed", f"{name.lower()}_mean")) for name in OBS_NAMES]
            if all(value is not None for value in channels):
                observation_mean = np.column_stack(channels)
        if observation_mean is None:
            raise ValueError("T3a output lacks observation_mean/trajectory_mean")
        observation_mean = np.asarray(observation_mean, dtype=np.float64)
        if observation_mean.ndim == 1:
            observation_mean = observation_mean[:, None]
        if observation_mean.shape == (3, length):
            observation_mean = observation_mean.T
        if observation_mean.shape != (length, 3):
            raise ValueError(f"T3a observation_mean must have shape {(length, 3)}, got {observation_mean.shape}")

        def variance(names: Sequence[str]) -> np.ndarray | None:
            value = None
            matched_name = ""
            # Resolve one alias at a time.  Looking at the whole alias tuple
            # would incorrectly square a ``*_variance`` field merely because
            # a later ``*_std`` alias exists.
            for name in names:
                candidate = _mapping_value(raw, (name,))
                if candidate is not None:
                    value = candidate
                    matched_name = name
                    break
            if value is None:
                return None
            array = np.asarray(value, dtype=np.float64)
            if array.shape == (3, length):
                array = array.T
            if array.shape != (length, 3):
                return None
            return np.square(array) if matched_name.endswith("_std") else np.maximum(array, 0.0)

        aleatoric = variance(("aleatoric_variance", "observation_noise_variance", "aleatoric_std", "observation_noise_std"))
        epistemic = variance(("epistemic_variance", "state_observation_variance", "epistemic_std", "state_observation_std"))
        total = variance(("total_variance", "observation_variance", "predictive_variance", "predictive_std"))
        if total is None and aleatoric is not None and epistemic is not None:
            total = aleatoric + epistemic
        if total is None:
            total = np.full_like(observation_mean, np.nan)
        if aleatoric is None:
            aleatoric = np.full_like(observation_mean, np.nan)
        if epistemic is None:
            epistemic = np.full_like(observation_mean, np.nan)

        raw_states = _mapping_value(raw, ("state_mean", "states", "physiological_state_mean"))
        state_mean = np.full((length, 6), np.nan, dtype=np.float64)
        if isinstance(raw_states, Mapping):
            for index, name in enumerate(STATE_NAMES):
                value = raw_states.get(name)
                if value is not None:
                    state_mean[:, index] = np.asarray(value, dtype=np.float64).reshape(-1)[:length]
        elif raw_states is not None:
            array = np.asarray(raw_states, dtype=np.float64)
            if array.shape == (length, 6):
                state_mean = array
            elif array.shape == (6, length):
                state_mean = array.T
        raw_state_var = _mapping_value(raw, ("state_variance", "state_var", "physiological_state_variance"))
        state_variance = None
        if raw_state_var is not None:
            array = np.asarray(raw_state_var, dtype=np.float64)
            if array.shape == (6, length):
                array = array.T
            if array.shape == (length, 6):
                state_variance = np.maximum(array, 0.0)
        parameter_summary = _mapping_value(raw, ("parameter_summary", "parameters", "fit_parameters")) or {}
        return CandidatePrediction(
            self.model_id,
            observation_mean,
            total,
            aleatoric,
            epistemic,
            state_mean,
            state_variance,
            parameter_summary if isinstance(parameter_summary, Mapping) else {},
            metadata={"physical_status": "candidate", "raw_type": type(raw).__name__},
        )


def candidate_panel() -> list[CandidateAdapter]:
    return [T0PersistenceAdapter(), T1IndependentLDSAdapter(), T2bLegacyAdapter(), T3aRobustAdapter()]


def _crps_gaussian(error: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form Gaussian CRPS used for the common calibration table."""

    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    z = np.asarray(error, dtype=np.float64) / sigma
    density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    # scipy.stats.norm.cdf is intentionally avoided in the hot loop.
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * density - 1.0 / np.sqrt(np.pi))


def _crps_student_t_mc(
    error: np.ndarray,
    scale: np.ndarray,
    nu: float,
    *,
    seed: int = 0,
    draws: int = 256,
) -> np.ndarray:
    """Deterministic Monte-Carlo CRPS for a Student-t predictive law.

    The evaluator deliberately labels this approximation in the output; it
    is not a closed-form Student-t CRPS implementation.  Two independent,
    fixed-seed draws estimate ``E|X-y| - 1/2 E|X-X'|`` pointwise.
    """

    error = np.asarray(error, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    first = rng.standard_t(float(nu), size=(int(draws), len(error))) * scale[None, :]
    second = rng.standard_t(float(nu), size=(int(draws), len(error))) * scale[None, :]
    return np.mean(np.abs(first - error[None, :]), axis=0) - 0.5 * np.mean(
        np.abs(first - second), axis=0
    )


def _calibration(
    reference: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    *,
    distribution: str = "Gaussian",
    student_nu: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    variance = np.asarray(variance, dtype=np.float64).reshape(-1)
    valid = np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(reference) & np.isfinite(mean) & np.isfinite(variance)
    if np.count_nonzero(valid) < 3:
        return {"status": "not_computed", "support_n": int(np.count_nonzero(valid))}
    total_variance = np.maximum(variance[valid], 1e-10)
    is_student = str(distribution).lower() in {"student-t", "student_t", "studentt"}
    nu = float(student_nu) if student_nu is not None else float("nan")
    if is_student and (not np.isfinite(nu) or nu <= 2.0):
        return {"status": "not_computed", "support_n": int(np.count_nonzero(valid)), "reason": "invalid_student_nu"}
    # T3a reports a total *variance* with Student-t degrees of freedom.  Its
    # scale is therefore sqrt(var*(nu-2)/nu), not sqrt(var).
    sigma = np.sqrt(total_variance * ((nu - 2.0) / nu) if is_student else total_variance)
    error = reference[valid] - mean[valid]
    z = error / sigma
    if is_student:
        levels = {f"{level:.2f}": float(student_t.ppf((1.0 + level) / 2.0, df=nu)) for level in NOMINAL_LEVELS}
        pit = student_t.cdf(z, df=nu)
        crps = float(np.mean(_crps_student_t_mc(error, sigma, nu, seed=seed)))
        nll = float(np.mean(student_t.logpdf(z, df=nu) - np.log(sigma)) * -1.0)
        method = "deterministic_mc"
        distribution_name = "Student-t"
    else:
        levels = {
            "0.50": 0.6744897501960817,
            "0.80": 1.2815515655446004,
            "0.95": 1.959963984540054,
        }
        pit = norm.cdf(z)
        crps = float(np.mean(_crps_gaussian(error, sigma)))
        nll = float(np.mean(np.log(sigma) + 0.5 * z**2 + 0.5 * np.log(2.0 * np.pi)))
        method = "closed_form"
        distribution_name = "Gaussian"
    risk = np.abs(error)
    if np.std(sigma) > 1e-12 and np.std(risk) > 1e-12:
        risk_corr = float(spearmanr(sigma, risk).statistic)
    else:
        risk_corr = float("nan")
    return {
        "status": "computed",
        "support_n": int(len(z)),
        "coverage": {key: float(np.mean(np.abs(z) <= threshold)) for key, threshold in levels.items()},
        "mean_pit": float(np.mean(pit)),
        "crps": crps,
        "nll": nll,
        "interval_width_95": float(2.0 * levels["0.95"] * np.mean(sigma)),
        "uncertainty_risk_spearman": risk_corr,
        "distribution": distribution_name,
        "student_nu": nu if is_student else None,
        "method": method,
    }


def _reconstruction_shape_errors(
    truth: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Small reconstruction temporal/spectral discrepancy summaries.

    These are descriptive checks, not claims of a full posterior predictive
    simulation.  They compare lag-one autocorrelation and normalized power
    spectra on the jointly observed samples.
    """

    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    valid = np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(truth) & np.isfinite(estimate)
    if np.count_nonzero(valid) < 8:
        return float("nan"), float("nan")
    truth = truth[valid]
    estimate = estimate[valid]
    if np.std(truth) <= 1e-12 or np.std(estimate) <= 1e-12:
        return float("nan"), float("nan")
    temporal = abs(_safe_corr(truth[:-1], truth[1:]) - _safe_corr(estimate[:-1], estimate[1:]))
    truth_power = np.abs(np.fft.rfft(truth - np.mean(truth))) ** 2
    estimate_power = np.abs(np.fft.rfft(estimate - np.mean(estimate))) ** 2
    truth_power /= max(float(np.sum(truth_power)), 1e-12)
    estimate_power /= max(float(np.sum(estimate_power)), 1e-12)
    spectral = float(np.mean(np.abs(truth_power - estimate_power)))
    return float(temporal), spectral


def _observation_metrics(case: SyntheticCase, prediction: CandidatePrediction) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    for channel, name in enumerate(OBS_NAMES):
        truth = case.clean[:, channel]
        observed = case.observations[:, channel]
        estimate = prediction.observation_mean[:, channel]
        valid = case.observation_mask[:, channel]
        artifact_mask = case.artifact_mask[:, channel]
        total = prediction.total_variance[:, channel]
        aleatoric = prediction.aleatoric_variance[:, channel]
        epistemic = prediction.epistemic_variance[:, channel]
        clean_nrmse = _nrmse(truth, estimate, valid)
        corrupted_nrmse = _nrmse(truth, observed, valid)
        artifact_nrmse = _nrmse(truth, estimate, artifact_mask & valid)
        attenuation = float(1.0 - artifact_nrmse / max(_nrmse(truth, observed, artifact_mask & valid), 1e-8)) if np.any(artifact_mask & valid) else float("nan")
        observation_residual = observed - estimate
        artifact_support = artifact_mask & valid
        artifact_relative_rmse = (
            _rmse(case.artifact[:, channel], observation_residual, artifact_support)
            / max(float(np.sqrt(np.mean(case.artifact[artifact_support, channel] ** 2))), 1e-8)
            if np.any(artifact_support)
            else float("nan")
        )
        off_mask = valid & ~artifact_mask
        rows = {
            "clean_nrmse": clean_nrmse,
            "corrupted_nrmse": corrupted_nrmse,
            "truth_pcc": _safe_corr(truth[valid], estimate[valid]),
            "artifact_nrmse": artifact_nrmse,
            "artifact_attenuation": attenuation,
            "artifact_residual_relative_rmse": artifact_relative_rmse,
            "off_artifact_distortion": _nrmse(truth, estimate, off_mask),
            "support_n": int(np.count_nonzero(valid)),
            "mean_total_variance": _finite_mean(total),
            "mean_aleatoric_variance": _finite_mean(aleatoric),
            "mean_epistemic_variance": _finite_mean(epistemic),
        }
        temporal_error, spectral_error = _reconstruction_shape_errors(truth, estimate, valid)
        rows.update({
            "reconstruction_temporal_acf_error": temporal_error,
            "reconstruction_spectral_shape_error": spectral_error,
        })
        for metric, value in rows.items():
            metric_rows.append({
                "scenario_id": case.scenario_id,
                "stress_case": case.stress_case,
                "severity": case.severity,
                "mask_fraction": float(1.0 - np.mean(valid)),
                "null_type": case.null_type or "",
                "artifact_scope": case.artifact_scope,
                "replicate_id": case.replicate_id,
                "model_id": prediction.model_id,
                "target": name,
                "metric": metric,
                "value": value,
                "lower": None,
                "upper": None,
                "units": "normalized",
                "status": "computed" if np.isfinite(value) else "not_computed",
            })
        calibration = _calibration(
            truth,
            estimate,
            total,
            valid,
            distribution=prediction.distribution,
            student_nu=prediction.student_nu,
            seed=case.replicate_id * 100 + channel,
        )
        if calibration["status"] == "computed":
            for nominal in NOMINAL_LEVELS:
                key = f"{nominal:.2f}"
                calibration_rows.append({
                    "target": name,
                    "nominal_level": nominal,
                    "empirical_coverage": calibration["coverage"][key],
                    "pit": calibration["mean_pit"],
                    "crps": calibration["crps"],
                    "nll": calibration["nll"],
                    "uncertainty_risk_spearman": calibration["uncertainty_risk_spearman"],
                    "interval_width_95": calibration["interval_width_95"],
                    "distribution": calibration.get("distribution", prediction.distribution),
                    "student_nu": calibration.get("student_nu"),
                    "method": calibration.get("method", "not_computed"),
                    "group": case.scenario_id,
                    "model_id": prediction.model_id,
                    "replicate_id": case.replicate_id,
                    "status": "computed",
                    "support_n": calibration["support_n"],
                })
        else:
            calibration_rows.append({
                "target": name,
                "nominal_level": None,
                "empirical_coverage": None,
                "pit": None,
                "crps": None,
                "nll": None,
                "uncertainty_risk_spearman": None,
                "interval_width_95": None,
                "distribution": prediction.distribution,
                "student_nu": prediction.student_nu,
                "method": None,
                "group": case.scenario_id,
                "model_id": prediction.model_id,
                "replicate_id": case.replicate_id,
                "status": "not_computed",
                "support_n": calibration["support_n"],
            })
        for index, time_s in enumerate(case.time_s):
            uncertainty_rows.append({
                "time_s": float(time_s),
                "scenario_id": case.scenario_id,
                "replicate_id": case.replicate_id,
                "model_id": prediction.model_id,
                "component": name,
                "aleatoric_variance": aleatoric[index],
                "epistemic_variance": epistemic[index],
                "total_variance": total[index],
                "status": "computed" if np.isfinite(total[index]) else "not_computed",
            })
    return metric_rows, calibration_rows, uncertainty_rows


def _state_rows(case: SyntheticCase, prediction: CandidatePrediction) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    means = prediction.state_mean
    variances = prediction.state_variance
    for index, name in enumerate(STATE_NAMES):
        truth = case.truth_states[:, index]
        estimate = means[:, index] if means is not None and means.shape == (len(truth), 6) else np.full(len(truth), np.nan)
        variance = variances[:, index] if variances is not None and variances.shape == (len(truth), 6) else np.full(len(truth), np.nan)
        status = "computed" if np.count_nonzero(np.isfinite(estimate)) >= 3 else "not_applicable"
        for point, time_s in enumerate(case.time_s):
            state_rows.append({
                "time_s": float(time_s),
                "scenario_id": case.scenario_id,
                "replicate_id": case.replicate_id,
                "model_id": prediction.model_id,
                "state_name": name,
                "truth": truth[point],
                "posterior_mean": estimate[point],
                "state_variance": variance[point],
                "unit": STATE_UNITS[name],
                "state_valid": status == "computed",
            })
        if status == "computed":
            valid = np.isfinite(estimate)
            metric_rows.extend([
                {
                    "scenario_id": case.scenario_id,
                    "stress_case": case.stress_case,
                    "severity": case.severity,
                    "mask_fraction": float(1.0 - np.mean(case.observation_mask)),
                    "null_type": case.null_type or "",
                    "artifact_scope": case.artifact_scope,
                    "replicate_id": case.replicate_id,
                    "model_id": prediction.model_id,
                    "target": name,
                    "metric": metric,
                    "value": value,
                    "lower": None,
                    "upper": None,
                    "units": STATE_UNITS[name],
                    "status": "computed" if np.isfinite(value) else "not_computed",
                }
                for metric, value in (
                    ("truth_pcc", _safe_corr(truth[valid], estimate[valid])),
                    ("truth_nrmse", _nrmse(truth, estimate, valid)),
                )
            ])
    return state_rows, metric_rows


def _trajectory_rows(case: SyntheticCase, prediction: CandidatePrediction) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(case.time_s):
        for channel, name in enumerate(OBS_NAMES):
            rows.append({
                "time_s": float(time_s),
                "scenario_id": case.scenario_id,
                "replicate_id": case.replicate_id,
                "model_id": prediction.model_id,
                "coordinate": name,
                "artifact_scope": case.artifact_scope,
                "clean_truth": case.clean[index, channel],
                "corrupted_observation": case.observations[index, channel],
                "posterior_mean": prediction.observation_mean[index, channel],
                "artifact": case.artifact[index, channel],
                "artifact_mask": bool(case.artifact_mask[index, channel]),
                "valid": bool(case.observation_mask[index, channel]),
                "aleatoric_variance": prediction.aleatoric_variance[index, channel],
                "epistemic_variance": prediction.epistemic_variance[index, channel],
                "total_variance": prediction.total_variance[index, channel],
            })
    return rows


def _extract_estimate(value: Any) -> tuple[float, float, float, float, str]:
    if isinstance(value, Mapping):
        estimate = value.get("estimate", value.get("mean", value.get("value")))
        sd = value.get("sd", value.get("std", value.get("posterior_sd")))
        lower = value.get("lower", value.get("q025"))
        upper = value.get("upper", value.get("q975"))
        status = str(value.get("status", "computed"))
    else:
        estimate, sd, lower, upper, status = value, None, None, None, "computed"
    return (
        float(estimate) if estimate is not None and np.isfinite(float(estimate)) else float("nan"),
        float(sd) if sd is not None and np.isfinite(float(sd)) else float("nan"),
        float(lower) if lower is not None and np.isfinite(float(lower)) else float("nan"),
        float(upper) if upper is not None and np.isfinite(float(upper)) else float("nan"),
        status,
    )


def _parameter_rows(prediction: CandidatePrediction, true_parameters: Mapping[str, float], fit: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aliases = {"kappa_per_s": ("kappa_per_s", "kappa"), "tau_s": ("tau_s", "tau")}
    covariance = np.asarray(_mapping_value(fit, ("parameter_covariance",)), dtype=np.float64) if fit is not None and _mapping_value(fit, ("parameter_covariance",)) is not None else np.full((2, 2), np.nan)
    fit_boundary = str(_mapping_value(fit, ("boundary_status",)) or "") if fit is not None else ""
    fit_identifiability = str(_mapping_value(fit, ("identifiability_status",)) or "not_computed") if fit is not None else "not_computed"
    likelihood_hessian = np.asarray(_mapping_value(fit, ("likelihood_hessian",)), dtype=np.float64) if fit is not None and _mapping_value(fit, ("likelihood_hessian",)) is not None else np.full((2, 2), np.nan)
    data_hessian_status = "computed" if likelihood_hessian.shape == (2, 2) and np.all(np.isfinite(likelihood_hessian)) else "not_computed"
    starts = _mapping_value(fit, ("starts",)) if fit is not None else None
    if isinstance(starts, Sequence) and not isinstance(starts, (str, bytes)):
        for start_id, record in enumerate(starts):
            estimate_values = _mapping_value(record, ("estimate",))
            objective = _mapping_value(record, ("objective", "loss", "nll"))
            success = _mapping_value(record, ("success",))
            if estimate_values is None:
                continue
            estimate_values = np.asarray(estimate_values, dtype=np.float64).reshape(-1)
            for index, name in enumerate(("kappa_per_s", "tau_s")):
                if index >= len(estimate_values):
                    continue
                estimate = float(estimate_values[index])
                rows.append({
                    "parameter_name": name,
                    "true_value": true_parameters.get(name),
                    "estimate": estimate,
                    "sd": None,
                    "lower": None,
                    "upper": None,
                    "relative_error": abs(estimate - true_parameters[name]) / max(abs(true_parameters[name]), 1e-8),
                    "identifiability_status": "start_success" if bool(success) else "start_failed",
                    "start_id": start_id,
                    "objective": objective,
                    "likelihood_hessian_status": data_hessian_status,
                    "data_identifiability_status": fit_identifiability,
                    "approximation": "EKF_Laplace",
                    "optimizer_success": bool(success),
                    "covered_95": None,
                    "sbc_cdf": None,
                })
    for name in ("kappa_per_s", "tau_s"):
        value = None
        if isinstance(prediction.parameters, Mapping):
            for alias in aliases[name]:
                if alias in prediction.parameters:
                    value = prediction.parameters[alias]
                    break
        estimate, summary_sd, summary_lower, summary_upper, status = _extract_estimate(value) if value is not None else (float("nan"), float("nan"), float("nan"), float("nan"), "not_computed")
        if fit is not None and status == "computed":
            status = fit_identifiability
        index = 0 if name == "kappa_per_s" else 1
        if covariance.shape == (2, 2) and np.all(np.isfinite(covariance)) and fit_boundary.upper() == "INTERIOR":
            variance = float(covariance[index, index])
            sd = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) and variance >= 0.0 else float("nan")
            lower = float(estimate - 1.96 * sd) if np.isfinite(estimate) and np.isfinite(sd) else float("nan")
            upper = float(estimate + 1.96 * sd) if np.isfinite(estimate) and np.isfinite(sd) else float("nan")
        else:
            sd, lower, upper = float("nan"), float("nan"), float("nan")
        true_value = float(true_parameters[name])
        covered_95 = bool(
            np.isfinite(lower) and np.isfinite(upper) and lower <= true_value <= upper
        ) if np.isfinite(lower) and np.isfinite(upper) else None
        sbc_cdf = float(norm.cdf((true_value - estimate) / sd)) if np.isfinite(estimate) and np.isfinite(sd) and sd > 0.0 else None
        rows.append({
            "parameter_name": name,
            "true_value": true_value,
            "estimate": estimate,
            "sd": sd,
            "lower": lower,
            "upper": upper,
            "relative_error": abs(estimate - true_parameters[name]) / max(abs(true_parameters[name]), 1e-8) if np.isfinite(estimate) else None,
            "identifiability_status": status,
            "start_id": "best",
            "objective": _mapping_value(fit, ("objective", "loss", "nll")) if fit is not None else None,
            "likelihood_hessian_status": data_hessian_status,
            "data_identifiability_status": fit_identifiability,
            "approximation": "EKF_Laplace",
            "optimizer_success": bool(_mapping_value(fit, ("optimizer_success",))) if fit is not None else False,
            "covered_95": covered_95,
            "sbc_cdf": sbc_cdf,
        })
    return rows


def _profile_rows(
    fit: Any,
    config: Mapping[str, Any],
    true_parameters: Mapping[str, float],
    case: SyntheticCase | None = None,
) -> list[dict[str, Any]]:
    """Serialize a model-supplied profile, or an explicit uncomputed grid."""

    rows: list[dict[str, Any]] = []
    supplied = _mapping_value(fit, ("profile_likelihood", "profile",)) if fit is not None else None
    if isinstance(supplied, Mapping):
        for parameter_name, points in supplied.items():
            if isinstance(points, Mapping):
                points = [{"grid_value": key, **(value if isinstance(value, Mapping) else {"delta_objective": value})} for key, value in points.items()]
            for point in points if isinstance(points, Sequence) and not isinstance(points, (str, bytes)) else []:
                rows.append({
                    "parameter_name": parameter_name,
                    "grid_value": _mapping_value(point, ("grid_value", "value")),
                    "objective": _mapping_value(point, ("objective", "nll")),
                    "delta_objective": _mapping_value(point, ("delta_objective", "delta_nll")),
                    "status": "computed",
                    "geometry_kind": "model_profile",
                })
    if rows:
        return rows
    # T3a currently exposes no dedicated profile API.  Evaluate a genuine
    # objective slice with ``smooth_balloon`` at each grid point, holding the
    # other fitted parameter fixed.  This is intentionally performed only for
    # a representative clean case; unsupported candidates remain explicit
    # ``not_computed`` rows.
    if case is not None and fit is not None and hasattr(fit, "parameters"):
        try:
            from src.inference import t3a_balloon_robust_ssm as module

            inference = config.get("inference", {})
            observation_cfg = config["observation"]
            spec = module.BalloonObservationSpec(
                eeg_loading=float(observation_cfg.get("eeg_loading", 1.0)),
                eeg_offset=float(observation_cfg.get("eeg_offset", 0.0)),
                observation_scale=tuple(float(observation_cfg["scale"][name]) for name in OBS_NAMES),
                student_nu=float(observation_cfg["student_t_df"]),
            )
            numerical = module.BalloonConfig(
                dt=1.0 / float(config["simulation"]["sampling_hz"]),
                irls_iterations=int(inference.get("irls_iterations", 3)),
                optimizer_max_iterations=int(inference.get("max_iterations", 60)),
                kappa_bounds=tuple(float(value) for value in config["physiology"]["free"]["kappa_per_s"]["bounds"]),
                tau_bounds=tuple(float(value) for value in config["physiology"]["free"]["tau_s"]["bounds"]),
                kappa_prior_mean=float(config["physiology"]["free"]["kappa_per_s"]["prior_mean"]),
                kappa_prior_sd=float(config["physiology"]["free"]["kappa_per_s"]["prior_sd"]),
                tau_prior_mean=float(config["physiology"]["free"]["tau_s"]["prior_mean"]),
                tau_prior_sd=float(config["physiology"]["free"]["tau_s"]["prior_sd"]),
                hessian_step=float(inference.get("finite_difference_step", 1e-3)),
            )
            optimum = fit.parameters.free
            grids: dict[str, np.ndarray] = {}
            for parameter_name in ("kappa_per_s", "tau_s"):
                lo, hi = [float(value) for value in config["physiology"]["free"][parameter_name]["bounds"]]
                grids[parameter_name] = np.linspace(lo, hi, max(int(inference.get("profile_points", 0)), 1))
            objective_values: dict[str, list[float]] = {name: [] for name in grids}
            for parameter_name, grid in grids.items():
                for value in grid:
                    free = module.BalloonFreeParameters(
                        kappa=float(value if parameter_name == "kappa_per_s" else optimum.kappa),
                        tau=float(value if parameter_name == "tau_s" else optimum.tau),
                    )
                    params = module.BalloonParameters(fixed=fit.parameters.fixed, free=free)
                    result = module.smooth_balloon(
                        case.observations,
                        parameters=params,
                        observation_spec=spec,
                        config=numerical,
                        observation_mask=case.observation_mask,
                    )
                    prior_penalty = 0.5 * (
                        ((free.kappa - numerical.kappa_prior_mean) / numerical.kappa_prior_sd) ** 2
                        + ((free.tau - numerical.tau_prior_mean) / numerical.tau_prior_sd) ** 2
                    )
                    objective_values[parameter_name].append(-float(result.predictive_log_likelihood) + prior_penalty)
            for parameter_name, grid in grids.items():
                values = np.asarray(objective_values[parameter_name], dtype=np.float64)
                finite = np.isfinite(values)
                if not np.any(finite):
                    raise FloatingPointError(f"profile objective is non-finite for {parameter_name}")
                baseline = float(np.nanmin(values))
                for value, objective_value in zip(grid, values):
                    rows.append({
                        "parameter_name": parameter_name,
                        "grid_value": float(value),
                        "objective": float(objective_value),
                        "delta_objective": float(objective_value - baseline),
                        "status": "computed",
                        "geometry_kind": "objective_slice",
                    })
            return rows
        except Exception:
            # Preserve an honest not-computed grid below; the caller records
            # the profile status without manufacturing an objective.
            rows = []
    for name in ("kappa_per_s", "tau_s"):
        spec = config["physiology"]["free"][name]
        lower, upper = [float(item) for item in spec["bounds"]]
        points = int(config.get("inference", {}).get("profile_points", 0))
        grid = np.linspace(lower, upper, max(points, 1))
        for value in grid:
            rows.append({
                "parameter_name": name,
                "grid_value": float(value),
                "objective": None,
                "delta_objective": None,
                "status": "not_computed",
                "geometry_kind": "not_computed",
            })
    return rows


def _prior_predictive(config: Mapping[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    simulation = config["simulation"]
    fs_hz = float(simulation["sampling_hz"])
    duration = float(simulation["duration_s"])
    times = np.arange(max(8, int(round(duration * fs_hz))), dtype=np.float64) / fs_hz
    for draw in range(int(count)):
        parameters = _parameter_values(config, _sample_free_parameters(rng, config))
        driver = _driver(times, rng, config)
        states, ok = _balloon_trajectory(times, driver, parameters)
        for index, name in enumerate(STATE_NAMES):
            values = states[:, index]
            finite = np.isfinite(values)
            rows.append({
                "draw_id": draw,
                "parameter_name": name,
                "min": float(np.nanmin(values)) if np.any(finite) else None,
                "max": float(np.nanmax(values)) if np.any(finite) else None,
                "finite": bool(np.all(finite)),
                "solver_ok": bool(ok),
                "boundary_contact": bool(index >= 2 and np.any(values <= 0.0)),
                "status": "computed" if ok else "invalid",
            })
        if ok:
            # Prior response ranges are hemodynamic, not the exogenous EEG
            # driver: use the explicit HbO observation map and report its
            # peak delay relative to the synthetic window.
            response = (
                float(parameters["p0"]) * (states[:, 4] - 1.0)
                - float(parameters["q0"]) * (states[:, 5] - 1.0)
            )
            centered_driver = states[:, 0] - np.mean(states[:, 0])
            centered_response = response - np.mean(response)
            max_lag = min(int(round(20.0 * fs_hz)), len(times) // 2)
            lag_correlations = np.asarray(
                [
                    _safe_corr(
                        centered_driver[: len(times) - lag],
                        centered_response[lag:],
                    )
                    for lag in range(max_lag + 1)
                ],
                dtype=np.float64,
            )
            peak_lag = int(np.nanargmax(lag_correlations)) if np.any(np.isfinite(lag_correlations)) else 0
            rows.append({
                "draw_id": draw,
                "parameter_name": "response_amplitude",
                "min": float(np.min(response)),
                "max": float(np.max(response)),
                "delay_s": float(peak_lag / fs_hz),
                "delay_correlation": float(lag_correlations[peak_lag]),
                "response_target": "HbO",
                "finite": bool(np.all(np.isfinite(response))),
                "solver_ok": bool(ok),
                "boundary_contact": False,
                "status": "computed",
            })
    return rows


def _null_metric(case: SyntheticCase, prediction: CandidatePrediction) -> float:
    if prediction.state_mean is None or prediction.state_mean.shape != (len(case.time_s), 6):
        return float("nan")
    driver = prediction.state_mean[:, 0]
    # For nulls, fNIRS is deliberately detached from the EEG truth.  This is
    # a cross-modal consistency diagnostic, not a state-recovery score.
    return _safe_corr(driver, case.clean[:, 1])


def _status_from(values: Sequence[str], *, allow_inconclusive: bool = True) -> str:
    statuses = list(values)
    if any(value == "FAIL" for value in statuses):
        return "FAIL"
    if any(value == "INVALID" for value in statuses):
        return "INVALID"
    if any(value in {"INCONCLUSIVE", "not_computed", "not_available"} for value in statuses):
        return "INCONCLUSIVE" if allow_inconclusive else "FAIL"
    return "PASS"


def _truth_compartment_rhs(
    state: Sequence[float],
    parameters: Mapping[str, float],
    driver: float = 0.0,
) -> np.ndarray:
    """Physical-coordinate RHS used by generator contract sentinels."""

    _, s, f, v, p, q = [float(item) for item in state]
    tau = float(parameters["tau_s"])
    alpha = float(parameters["alpha"])
    e0 = float(parameters["e0"])
    kappa = float(parameters["kappa_per_s"])
    gamma = float(parameters["gamma"])
    f_out = v ** (1.0 / alpha)
    extraction = float(-np.expm1(np.log1p(-e0) / f))
    return np.asarray(
        [
            0.0,
            driver - kappa * s - gamma * (f - 1.0),
            s,
            (f - f_out) / tau,
            (f - f_out * p / v) / tau,
            (f * extraction / e0 - f_out * q / v) / tau,
        ],
        dtype=np.float64,
    )


def _contract_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    """Executable P0 contract checks independent of candidate implementations."""

    fixed = config["physiology"]["fixed"]
    truth = config["physiology"]["truth"]
    free = config["physiology"]["free"]
    parameters = _parameter_values(config)
    rest = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    rest_rhs = _truth_compartment_rhs(rest, parameters)
    sentinel = rest.copy()
    sentinel[2] = 1.1
    sentinel_rhs = _truth_compartment_rhs(sentinel, parameters)
    f_out = float(sentinel[3] ** (1.0 / float(fixed["alpha"])))
    # The dynamic stability check uses a finite-difference Jacobian of the
    # (s,f,v,p,q) compartment equations at rest with r/driver held at zero.
    dynamic_indices = np.asarray([1, 2, 3, 4, 5])
    jacobian = np.zeros((5, 5), dtype=np.float64)
    step = 1.0e-6
    for col, state_index in enumerate(dynamic_indices):
        plus, minus = rest.copy(), rest.copy()
        plus[state_index] += step
        minus[state_index] -= step
        jacobian[:, col] = (
            _truth_compartment_rhs(plus, parameters)[dynamic_indices]
            - _truth_compartment_rhs(minus, parameters)[dynamic_indices]
        ) / (2.0 * step)
    try:
        eigenvalues = np.linalg.eigvals(jacobian)
        stable = bool(np.all(np.isfinite(eigenvalues)) and np.max(np.real(eigenvalues)) <= 1.0e-8)
    except np.linalg.LinAlgError:
        stable = False
    # Explicit observation-map sign/baseline sentinel in the evaluator's
    # declared HbT/HbR/HbO coordinates.
    baseline_hbt = float(fixed["p0"] * rest[4])
    baseline_hbr = float(fixed["q0"] * rest[5])
    baseline_hbo = baseline_hbt - baseline_hbr
    return {
        "state_names_order": tuple(STATE_NAMES) == ("r", "s", "f", "v", "p", "q"),
        "free_set_exact_kappa_tau": set(free) == {"kappa_per_s", "tau_s"},
        "tau_v_fixed_zero": abs(float(fixed.get("tau_v_s", 0.0))) <= 1.0e-12,
        "fixed_positive_sign_bounds": bool(
            float(fixed["alpha"]) > 0.0
            and 0.0 < float(fixed["e0"]) < 1.0
            and float(fixed["gamma"]) > 0.0
            and float(fixed["p0"]) > 0.0
            and float(fixed["q0"]) > 0.0
            and float(fixed["q0"]) <= float(fixed["p0"])
            and float(config["observation"]["eeg_loading"]) > 0.0
        ),
        "rest_rhs_zero": bool(np.all(np.isfinite(rest_rhs)) and np.allclose(rest_rhs, 0.0, atol=1.0e-12)),
        "nonrest_p_balance_sentinel": bool(np.isfinite(sentinel_rhs[4]) and abs(float(sentinel_rhs[4])) > 1.0e-8),
        "f_out_finite_positive": bool(np.isfinite(f_out) and f_out > 0.0),
        "rest_jacobian_no_positive_real": stable,
        "explicit_hbt_hbr_hbo_map": bool(
            np.isfinite(baseline_hbt + baseline_hbr + baseline_hbo)
            and baseline_hbt >= baseline_hbr >= 0.0
            and baseline_hbo >= 0.0
        ),
        "source_ledger_complete": isinstance(config["physiology"].get("source_ledger"), Mapping),
        "truth_bounds_inside_free": bool(
            float(free["kappa_per_s"]["bounds"][0]) <= float(truth["kappa_per_s"]) <= float(free["kappa_per_s"]["bounds"][1])
            and float(free["tau_s"]["bounds"][0]) <= float(truth["tau_s"]) <= float(free["tau_s"]["bounds"][1])
        ),
    }


def _gate_results(
    config: Mapping[str, Any],
    prior_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
    parameter_rows: Sequence[Mapping[str, Any]],
    *,
    available_models: Mapping[str, bool],
    profile_rows: Sequence[Mapping[str, Any]] = (),
    contract_checks: Mapping[str, bool] | None = None,
    physical_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    gates = config.get("gates", {})
    smoke_mode = config["experiment"].get("qualification_mode") == "smoke"
    expected_prior_draws = int(config["simulation"]["prior_predictive_draws"])
    expected_draw_ids = set(range(expected_prior_draws))
    observed_draw_ids = {row.get("draw_id") for row in prior_rows}
    prior_state_rows = [row for row in prior_rows if row.get("parameter_name") in STATE_NAMES]
    prior_state_keys = {
        (row.get("draw_id"), row.get("parameter_name")) for row in prior_state_rows
    }
    expected_state_keys = {
        (draw_id, state_name)
        for draw_id in expected_draw_ids
        for state_name in STATE_NAMES
    }
    prior_row_fields = {"draw_id", "parameter_name", "finite", "solver_ok", "boundary_contact", "status"}
    prior_draw_support = observed_draw_ids == expected_draw_ids
    prior_state_support = (
        len(prior_state_rows) == expected_prior_draws * len(STATE_NAMES)
        and prior_state_keys == expected_state_keys
        and all(prior_row_fields.issubset(row) for row in prior_state_rows)
    )
    prior_state_computed = bool(prior_state_rows) and all(
        str(row.get("status")) == "computed" for row in prior_state_rows
    )
    generator_physical_check_names = {
        "finite",
        "positive_fvpq",
        "rest_initial",
        "extraction_in_0_1",
        "absolute_hb_nonnegative",
        "hbr_not_above_hbt",
        "compartment_rhs_finite",
        "compartment_rhs_residual",
    }
    candidate_physical_check_names = {
        "finite",
        "positive_fvpq",
        "oxygen_extraction_in_unit_interval",
        "absolute_hb_nonnegative",
        "hbr_not_above_hbt",
        "rest_equilibrium",
        "minimum_f",
        "minimum_v",
        "minimum_p",
        "minimum_q",
        "minimum_extraction",
        "maximum_extraction",
    }
    generator_rows = [row for row in physical_rows if row.get("model_id") == "generator"]
    generator_names = {str(row.get("check")) for row in generator_rows}
    expected_generator_keys = {
        (replicate_id, "clean", check)
        for replicate_id in range(int(config["simulation"]["replicates"]))
        for check in generator_physical_check_names
    }
    generator_keys = {
        (row.get("replicate_id"), row.get("scenario_id"), row.get("check"))
        for row in generator_rows
    }
    contract_complete = bool(contract_checks) and all(bool(value) for value in contract_checks.values())
    p0_checks = {
        "prior_draw_support": prior_draw_support,
        "prior_state_support": prior_state_support,
        "prior_state_status_computed": prior_state_computed,
        "finite_prior_truth": all(bool(row.get("finite")) for row in prior_rows),
        "solver_success": all(bool(row.get("solver_ok")) for row in prior_rows),
        "no_boundary_contact": not any(bool(row.get("boundary_contact")) for row in prior_rows),
        "minimal_tau_v": float(config["physiology"]["fixed"].get("tau_v_s", 0.0)) == 0.0,
        "generator_physical_check_set": generator_names == generator_physical_check_names
        and len(generator_rows) == len(expected_generator_keys)
        and generator_keys == expected_generator_keys,
        "generator_physical_checks": bool(generator_rows)
        and all(str(row.get("status", "fail")).lower() == "pass" for row in generator_rows),
        "semantic_contract_complete": contract_complete,
    }
    if contract_checks:
        p0_checks.update({str(name): bool(value) for name, value in contract_checks.items()})
    p0_substantive_checks = {
        name: value
        for name, value in p0_checks.items()
        if name not in {"prior_draw_support", "prior_state_support"}
    }
    if not prior_rows or not prior_draw_support or not prior_state_support:
        p0_status = "INCONCLUSIVE"
    elif all(bool(value) for value in p0_substantive_checks.values()):
        p0_status = "PASS"
    else:
        p0_status = "FAIL"
    draw_ids = observed_draw_ids
    solver_by_draw = {
        draw_id: all(bool(row.get("solver_ok")) for row in prior_rows if row.get("draw_id") == draw_id)
        for draw_id in draw_ids
    }
    boundary_by_draw = {
        draw_id: any(bool(row.get("boundary_contact")) for row in prior_rows if row.get("draw_id") == draw_id)
        for draw_id in draw_ids
    }
    solver_fail_fraction = float(np.mean([not value for value in solver_by_draw.values()])) if solver_by_draw else 1.0
    boundary_fraction = float(np.mean(list(boundary_by_draw.values()))) if boundary_by_draw else 1.0
    response_rows = [
        row for row in prior_rows
        if row.get("parameter_name") == "response_amplitude"
        and np.isfinite(float(row.get("max", np.nan)))
        and np.isfinite(float(row.get("min", np.nan)))
    ]
    prior_amplitudes = [float(row["max"]) - float(row["min"]) for row in response_rows]
    prior_delays = [
        float(row["delay_s"])
        for row in response_rows
        if np.isfinite(float(row.get("delay_s", np.nan)))
    ]
    response_range = {
        "min": min(prior_amplitudes) if prior_amplitudes else None,
        "max": max(prior_amplitudes) if prior_amplitudes else None,
    }
    delay_range = {
        "min": min(prior_delays) if prior_delays else None,
        "max": max(prior_delays) if prior_delays else None,
    }
    threshold_names = ("min_response_amplitude", "max_response_amplitude", "min_delay_s", "max_delay_s")
    thresholds_frozen = all(name in gates for name in threshold_names)
    response_draw_ids = {row.get("draw_id") for row in response_rows}
    response_fields = prior_row_fields | {"min", "max", "delay_s", "response_target"}
    response_support = (
        len(response_rows) == expected_prior_draws
        and response_draw_ids == expected_draw_ids
        and all(response_fields.issubset(row) for row in response_rows)
        and all(str(row.get("status")) == "computed" for row in response_rows)
        and all(str(row.get("response_target")) == "HbO" for row in response_rows)
    )
    p1_checks = {
        "solver_fail_fraction": solver_fail_fraction,
        "boundary_contact_fraction": boundary_fraction,
        "solver_threshold": solver_fail_fraction <= float(gates.get("max_solver_fail_fraction", 0.0)),
        "boundary_threshold": boundary_fraction <= float(gates.get("max_boundary_contact_fraction", 0.0)),
        "prior_response_range": response_range,
        "prior_delay_range_s": delay_range,
        "prior_draw_support": prior_draw_support,
        "prior_state_support": prior_state_support and prior_state_computed,
        "response_support": response_support and len(prior_amplitudes) == expected_prior_draws,
        "delay_support": response_support and len(prior_delays) == expected_prior_draws,
        "response_thresholds_frozen": thresholds_frozen,
    }
    if thresholds_frozen:
        p1_checks.update({
            "response_amplitude_bounds": bool(prior_amplitudes)
            and min(prior_amplitudes) >= float(gates["min_response_amplitude"])
            and max(prior_amplitudes) <= float(gates["max_response_amplitude"]),
            "response_delay_bounds": bool(prior_delays)
            and min(prior_delays) >= float(gates["min_delay_s"])
            and max(prior_delays) <= float(gates["max_delay_s"]),
        })
    if not p1_checks["solver_threshold"] or not p1_checks["boundary_threshold"]:
        p1_status = "FAIL"
    elif not p1_checks["prior_draw_support"] or not p1_checks["prior_state_support"] or not p1_checks["response_support"] or not p1_checks["delay_support"] or not thresholds_frozen:
        p1_status = "INCONCLUSIVE"
    elif not p1_checks["response_amplitude_bounds"] or not p1_checks["response_delay_bounds"]:
        p1_status = "FAIL"
    else:
        p1_status = "PASS"

    t3a_parameter_rows = [row for row in parameter_rows if row.get("model_id") == "T3a-balloon-robust"]
    p2_values = [str(row.get("identifiability_status")) for row in t3a_parameter_rows]
    t3a_profile_rows = [
        row for row in profile_rows
        if row.get("model_id") == "T3a-balloon-robust" and row.get("status") == "computed"
    ]
    parameter_names = ("kappa_per_s", "tau_s")
    replicate_ids = set(range(int(config["simulation"]["replicates"])))
    expected_profile_keys = {
        (0, parameter_name, round(float(grid_value), 12))
        for parameter_name in parameter_names
        for grid_value in np.linspace(
            *[float(value) for value in config["physiology"]["free"][parameter_name]["bounds"]],
            int(config["inference"]["profile_points"]),
        )
    }
    profile_keys = {
        (row.get("replicate_id"), row.get("parameter_name"), round(float(row.get("grid_value", np.nan)), 12))
        for row in t3a_profile_rows
        if np.isfinite(float(row.get("grid_value", np.nan)))
    }
    profile_key_support = len(t3a_profile_rows) == len(expected_profile_keys) and profile_keys == expected_profile_keys
    profile_computed = profile_key_support and all(
        np.isfinite(float(row.get("objective", np.nan)))
        and row.get("scenario_id") == "clean"
        and row.get("geometry_kind") == "objective_slice"
        for row in t3a_profile_rows
    )
    best_parameter_rows = [row for row in t3a_parameter_rows if str(row.get("start_id")) == "best"]
    expected_best_keys = {(replicate_id, parameter_name) for replicate_id in replicate_ids for parameter_name in parameter_names}
    best_keys = {(row.get("replicate_id"), row.get("parameter_name")) for row in best_parameter_rows}
    best_key_support = len(best_parameter_rows) == len(expected_best_keys) and best_keys == expected_best_keys
    best_identifiable = bool(best_parameter_rows) and all(
        str(row.get("identifiability_status")) == "IDENTIFIABLE" for row in best_parameter_rows
    )
    best_hessian = bool(best_parameter_rows) and all(
        str(row.get("likelihood_hessian_status")) == "computed" for row in best_parameter_rows
    )
    best_optimizer_success = bool(best_parameter_rows) and all(
        bool(row.get("optimizer_success")) for row in best_parameter_rows
    )
    coverage_by_parameter: dict[str, float | None] = {}
    rank_pvalue_by_parameter: dict[str, float | None] = {}
    min_sbc_replicates = int(gates["min_sbc_replicates"])
    for parameter_name in parameter_names:
        rows = [row for row in best_parameter_rows if row.get("parameter_name") == parameter_name]
        covered = [row.get("covered_95") for row in rows if row.get("covered_95") is not None]
        ranks = [
            float(row["sbc_cdf"])
            for row in rows
            if row.get("sbc_cdf") is not None and np.isfinite(float(row["sbc_cdf"]))
        ]
        coverage_by_parameter[parameter_name] = float(np.mean(covered)) if len(covered) >= min_sbc_replicates else None
        rank_pvalue_by_parameter[parameter_name] = (
            float(kstest(ranks, "uniform").pvalue) if len(ranks) >= min_sbc_replicates else None
        )
    start_rows = [row for row in t3a_parameter_rows if str(row.get("start_id")) != "best"]
    expected_start_keys = {
        (replicate_id, parameter_name, str(start_id))
        for replicate_id in replicate_ids
        for parameter_name in parameter_names
        for start_id in range(int(config["inference"]["multistarts"]))
    }
    start_keys = {
        (row.get("replicate_id"), row.get("parameter_name"), str(row.get("start_id")))
        for row in start_rows
    }
    start_key_support = len(start_rows) == len(expected_start_keys) and start_keys == expected_start_keys
    start_success = start_key_support and all(bool(row.get("optimizer_success")) for row in start_rows)
    spreads: list[float] = []
    for replicate_id in replicate_ids:
        for parameter_name in parameter_names:
            estimates = [
                float(row["estimate"])
                for row in start_rows
                if row.get("replicate_id") == replicate_id
                and row.get("parameter_name") == parameter_name
                and bool(row.get("optimizer_success"))
                and np.isfinite(float(row.get("estimate", np.nan)))
            ]
            if len(estimates) >= 2:
                lower, upper = config["physiology"]["free"][parameter_name]["bounds"]
                spreads.append((max(estimates) - min(estimates)) / (float(upper) - float(lower)))
    coverage_support = best_key_support and all(
        value is not None for value in coverage_by_parameter.values()
    )
    rank_support = all(value is not None for value in rank_pvalue_by_parameter.values())
    coverage_pass = coverage_support and all(
        float(value) >= float(gates["min_parameter_coverage_95"])
        for value in coverage_by_parameter.values()
        if value is not None
    )
    rank_pass = rank_support and all(
        float(value) >= float(gates["min_sbc_ks_pvalue"])
        for value in rank_pvalue_by_parameter.values()
        if value is not None
    )
    multistart_pass = bool(spreads) and max(spreads) <= float(gates["max_multistart_spread_fraction"])
    if any(value.upper() in {"PRIOR_DOMINATED", "UNIDENTIFIABLE", "START_FAILED"} for value in p2_values):
        p2_status = "INCONCLUSIVE"
    elif not (
        profile_computed
        and best_identifiable
        and best_hessian
        and best_optimizer_success
        and start_success
        and coverage_support
        and rank_support
        and spreads
    ):
        p2_status = "INCONCLUSIVE"
    elif not coverage_pass or not rank_pass or not multistart_pass:
        p2_status = "FAIL"
    else:
        p2_status = "PASS"

    t3a_metrics = [
        row for row in metric_rows
        if row.get("model_id") == "T3a-balloon-robust" and not row.get("null_type")
    ]
    driver_rows = [
        row for row in t3a_metrics
        if row.get("scenario_id") == "clean"
        and row.get("target") == "r"
        and row.get("metric") == "truth_nrmse"
        and row.get("status") == "computed"
        and np.isfinite(float(row.get("value", np.nan)))
    ]
    distortion_rows = [
        row for row in t3a_metrics
        if row.get("stress_case") not in {"clean", "null"}
        and row.get("metric") == "off_artifact_distortion"
        and row.get("status") == "computed"
        and np.isfinite(float(row.get("value", np.nan)))
    ]
    additive_artifacts = set(config["stress_tests"]["artifact_families"]) - {"dropout"}
    attenuation_rows = [
        row for row in t3a_metrics
        if row.get("stress_case") in additive_artifacts
        and row.get("metric") == "artifact_attenuation"
        and row.get("status") == "computed"
        and np.isfinite(float(row.get("value", np.nan)))
    ]
    residual_rows = [
        row for row in t3a_metrics
        if row.get("stress_case") in additive_artifacts
        and row.get("metric") == "artifact_residual_relative_rmse"
        and row.get("status") == "computed"
        and np.isfinite(float(row.get("value", np.nan)))
    ]
    variance_rows = [
        row for row in t3a_metrics
        if row.get("metric") == "mean_epistemic_variance"
        and row.get("status") == "computed"
        and np.isfinite(float(row.get("value", np.nan)))
    ]
    clean_variance = {
        (row.get("replicate_id"), row.get("target")): float(row["value"])
        for row in variance_rows
        if row.get("stress_case") == "clean"
    }

    def affected_coordinate(row: Mapping[str, Any]) -> bool:
        target = row.get("target")
        scope = row.get("artifact_scope")
        if scope == "eeg_only" or scope == "missing_eeg":
            return target == "EEG"
        if scope in {"fnirs_only", "systemic_fnirs", "missing_fnirs"}:
            return target in {"HbO", "HbR"}
        return scope in {"systemic", "composite_missing"}

    def affected_targets(scope: str) -> set[str]:
        if scope in {"eeg_only", "missing_eeg"}:
            return {"EEG"}
        if scope in {"fnirs_only", "systemic_fnirs", "missing_fnirs"}:
            return {"HbO", "HbR"}
        return set(OBS_NAMES)

    artifact_uncertainty_deltas: list[float] = []
    missing_uncertainty_deltas: list[float] = []
    artifact_uncertainty_keys: set[tuple[Any, Any, Any]] = set()
    missing_uncertainty_keys: set[tuple[Any, Any, Any]] = set()
    for row in variance_rows:
        if not affected_coordinate(row):
            continue
        baseline = clean_variance.get((row.get("replicate_id"), row.get("target")))
        if baseline is None:
            continue
        delta = float(row["value"]) - baseline
        if row.get("stress_case") in config["stress_tests"]["artifact_families"]:
            artifact_uncertainty_deltas.append(delta)
            artifact_uncertainty_keys.add((row.get("replicate_id"), row.get("scenario_id"), row.get("target")))
        elif row.get("stress_case") in {"missing_eeg", "missing_fnirs"}:
            missing_uncertainty_deltas.append(delta)
            missing_uncertainty_keys.add((row.get("replicate_id"), row.get("scenario_id"), row.get("target")))
    t3a_null_rows = [
        row for row in null_rows
        if row.get("model_id") == "T3a-balloon-robust" and row.get("status") == "computed"
    ]
    candidate_physical_rows = [
        row for row in physical_rows if row.get("model_id") == "T3a-balloon-robust"
    ]
    replicate_count = int(config["simulation"]["replicates"])
    positive_severities = [float(value) for value in config["simulation"]["severity_levels"] if float(value) > 0.0]
    stress_design = [
        (
            str(family),
            f"{family}_s{severity:g}",
            affected_targets(str(config["stress_tests"]["artifact_scopes"][family])),
        )
        for family in config["stress_tests"]["artifact_families"]
        for severity in positive_severities
    ]
    expected_driver_keys = {(replicate_id, "r") for replicate_id in range(replicate_count)}
    expected_distortion_keys = {
        (replicate_id, scenario_id, target)
        for replicate_id in range(replicate_count)
        for _family, scenario_id, _targets in stress_design
        for target in OBS_NAMES
    } | {
        (replicate_id, scenario_id, target)
        for replicate_id in range(replicate_count)
        for scenario_id, targets in (("missing_eeg", {"HbO", "HbR"}), ("missing_fnirs", {"EEG"}))
        for target in targets
    }
    expected_additive_keys = {
        (replicate_id, scenario_id, target)
        for replicate_id in range(replicate_count)
        for family, scenario_id, targets in stress_design
        if family in additive_artifacts
        for target in targets
    }
    expected_artifact_uncertainty_keys = {
        (replicate_id, scenario_id, target)
        for replicate_id in range(replicate_count)
        for _family, scenario_id, targets in stress_design
        for target in targets
    }
    expected_missing_uncertainty_keys = {
        (replicate_id, scenario_id, target)
        for replicate_id in range(replicate_count)
        for scenario_id, targets in (("missing_eeg", {"EEG"}), ("missing_fnirs", {"HbO", "HbR"}))
        for target in targets
    }
    expected_case_count = replicate_count * (
        1
        + len(config["stress_tests"]["artifact_families"]) * len(positive_severities)
        + 2
        + len(config["stress_tests"]["nulls"])
    )
    candidate_physical_cases = {
        (row.get("replicate_id"), row.get("scenario_id")) for row in candidate_physical_rows
    }
    driver_keys = {(row.get("replicate_id"), row.get("target")) for row in driver_rows}
    distortion_keys = {(row.get("replicate_id"), row.get("scenario_id"), row.get("target")) for row in distortion_rows}
    attenuation_keys = {(row.get("replicate_id"), row.get("scenario_id"), row.get("target")) for row in attenuation_rows}
    residual_keys = {(row.get("replicate_id"), row.get("scenario_id"), row.get("target")) for row in residual_rows}
    expected_null_keys = {
        (replicate_id, str(null_type))
        for replicate_id in range(replicate_count)
        for null_type in config["stress_tests"]["nulls"]
    }
    null_keys = {(row.get("replicate_id"), row.get("null_type")) for row in t3a_null_rows}
    expected_case_keys = {
        (replicate_id, scenario_id)
        for replicate_id in range(replicate_count)
        for scenario_id in (
            ["clean", "missing_eeg", "missing_fnirs"]
            + [scenario_id for _family, scenario_id, _targets in stress_design]
            + [f"null_{null_type}" for null_type in config["stress_tests"]["nulls"]]
        )
    }
    physical_keys = {
        (row.get("replicate_id"), row.get("scenario_id"), row.get("check"))
        for row in candidate_physical_rows
    }
    expected_physical_keys = {
        (replicate_id, scenario_id, check)
        for replicate_id, scenario_id in expected_case_keys
        for check in candidate_physical_check_names
    }
    p3_checks = {
        "model_available": bool(available_models.get("T3a-balloon-robust", False)),
        "driver_support": len(driver_rows) == len(expected_driver_keys) and driver_keys == expected_driver_keys,
        "driver_nrmse": bool(driver_rows) and max(float(row["value"]) for row in driver_rows) <= float(gates["max_driver_nrmse"]),
        "off_artifact_support": len(distortion_rows) == len(expected_distortion_keys) and distortion_keys == expected_distortion_keys,
        "off_artifact_distortion": bool(distortion_rows) and max(float(row["value"]) for row in distortion_rows) <= float(gates["max_off_artifact_distortion"]),
        "artifact_separation_support": len(attenuation_rows) == len(expected_additive_keys) and attenuation_keys == expected_additive_keys and len(residual_rows) == len(expected_additive_keys) and residual_keys == expected_additive_keys,
        "artifact_attenuation": bool(attenuation_rows) and min(float(row["value"]) for row in attenuation_rows) >= float(gates["min_artifact_attenuation"]),
        "artifact_residual": bool(residual_rows) and max(float(row["value"]) for row in residual_rows) <= float(gates["max_artifact_residual_relative_rmse"]),
        "artifact_uncertainty_support": len(artifact_uncertainty_deltas) == len(expected_artifact_uncertainty_keys) and artifact_uncertainty_keys == expected_artifact_uncertainty_keys,
        "artifact_uncertainty_increase": bool(artifact_uncertainty_deltas) and min(artifact_uncertainty_deltas) >= float(gates["min_uncertainty_increase_artifact"]),
        "missing_uncertainty_support": len(missing_uncertainty_deltas) == len(expected_missing_uncertainty_keys) and missing_uncertainty_keys == expected_missing_uncertainty_keys,
        "missing_uncertainty_increase": bool(missing_uncertainty_deltas) and min(missing_uncertainty_deltas) >= float(gates["min_uncertainty_increase_missing"]),
        "null_support": len(t3a_null_rows) == len(expected_null_keys) and null_keys == expected_null_keys,
        "nulls": len(t3a_null_rows) == len(expected_null_keys) and null_keys == expected_null_keys and all(bool(row.get("pass")) for row in t3a_null_rows),
        "candidate_physical_support": len(candidate_physical_cases) == expected_case_count and candidate_physical_cases == expected_case_keys and len(candidate_physical_rows) == len(expected_physical_keys) and physical_keys == expected_physical_keys,
        "candidate_physical_checks": bool(candidate_physical_rows) and all(str(row.get("status")).lower() == "pass" for row in candidate_physical_rows),
    }
    support_names = {
        "model_available", "driver_support", "off_artifact_support",
        "artifact_separation_support", "artifact_uncertainty_support",
        "missing_uncertainty_support", "null_support", "candidate_physical_support",
    }
    if not p3_checks["model_available"] or not all(p3_checks[name] for name in support_names - {"model_available"}):
        p3_status = "INCONCLUSIVE"
    elif not all(bool(value) for name, value in p3_checks.items() if name not in support_names):
        p3_status = "FAIL"
    else:
        p3_status = "PASS"

    t3a_calibration = [
        row for row in calibration_rows
        if row.get("model_id") == "T3a-balloon-robust"
        and row.get("status") == "computed"
        and row.get("nominal_level") == 0.95
        and row.get("group") == "clean"
    ]
    replicate_ids = set(range(int(config["simulation"]["replicates"])))
    expected_clean_keys = {(replicate_id, target) for replicate_id in replicate_ids for target in OBS_NAMES}
    calibration_keys = {(row.get("replicate_id"), row.get("target")) for row in t3a_calibration}
    calibration_key_support = len(t3a_calibration) == len(expected_clean_keys) and calibration_keys == expected_clean_keys
    predictive_distribution_contract = bool(t3a_calibration) and all(
        str(row.get("distribution")) == "Student-t"
        and np.isclose(float(row.get("student_nu", np.nan)), float(config["observation"]["student_t_df"]))
        and str(row.get("method")) == "deterministic_mc"
        for row in t3a_calibration
    )
    coverages = [float(row["empirical_coverage"]) for row in t3a_calibration if np.isfinite(float(row.get("empirical_coverage", np.nan)))]
    risk_rows = [
        row for row in calibration_rows
        if row.get("model_id") == "T3a-balloon-robust"
        and row.get("status") == "computed"
        and row.get("group") == "clean"
        and row.get("nominal_level") == 0.95
    ]
    normalized_crps = [
        float(row["crps"]) / float(config["observation"]["scale"][str(row["target"])])
        for row in risk_rows if np.isfinite(float(row.get("crps", np.nan)))
    ]
    standardized_nll = [
        float(row["nll"]) - math.log(float(config["observation"]["scale"][str(row["target"])]))
        for row in risk_rows if np.isfinite(float(row.get("nll", np.nan)))
    ]
    spearmans = [float(row["uncertainty_risk_spearman"]) for row in risk_rows if np.isfinite(float(row.get("uncertainty_risk_spearman", np.nan)))]
    clean_metrics = [row for row in t3a_metrics if row.get("scenario_id") == "clean"]
    reconstruction_temporal = [row for row in clean_metrics if row.get("metric") == "reconstruction_temporal_acf_error" and row.get("status") == "computed" and np.isfinite(float(row.get("value", np.nan)))]
    reconstruction_spectral = [row for row in clean_metrics if row.get("metric") == "reconstruction_spectral_shape_error" and row.get("status") == "computed" and np.isfinite(float(row.get("value", np.nan)))]
    expected_clean_targets = int(config["simulation"]["replicates"]) * len(OBS_NAMES)
    temporal_keys = {(row.get("replicate_id"), row.get("target")) for row in reconstruction_temporal}
    spectral_keys = {(row.get("replicate_id"), row.get("target")) for row in reconstruction_spectral}
    g4_checks = {
        "calibration_key_support": calibration_key_support,
        "predictive_distribution_contract": predictive_distribution_contract,
        "coverage_support": len(coverages) == expected_clean_targets and calibration_key_support,
        "coverage_floor": bool(coverages) and min(coverages) >= float(gates.get("min_predictive_coverage_95", 0.75)),
        "coverage_ceiling": bool(coverages) and max(coverages) <= float(gates.get("max_predictive_coverage_95", 1.0)),
        "normalized_crps_support": len(normalized_crps) == expected_clean_targets,
        "standardized_nll_support": len(standardized_nll) == expected_clean_targets,
        "normalized_crps": bool(normalized_crps) and max(normalized_crps) <= float(gates["max_normalized_crps"]),
        "standardized_nll": bool(standardized_nll) and max(standardized_nll) <= float(gates["max_standardized_nll"]),
        "uncertainty_risk_spearman_support": len(spearmans) == expected_clean_targets,
        "uncertainty_risk_spearman": bool(spearmans) and float(np.median(spearmans)) >= float(gates["min_uncertainty_risk_spearman"]),
        "reconstruction_temporal_support": len(reconstruction_temporal) == expected_clean_targets and temporal_keys == expected_clean_keys,
        "reconstruction_spectral_support": len(reconstruction_spectral) == expected_clean_targets and spectral_keys == expected_clean_keys,
        "reconstruction_temporal_threshold": bool(reconstruction_temporal) and max(float(row["value"]) for row in reconstruction_temporal) <= float(gates["max_reconstruction_temporal_acf_error"]),
        "reconstruction_spectral_threshold": bool(reconstruction_spectral) and max(float(row["value"]) for row in reconstruction_spectral) <= float(gates["max_reconstruction_spectral_shape_error"]),
    }
    g4_support = all(
        bool(g4_checks[name]) for name in (
            "calibration_key_support", "coverage_support", "normalized_crps_support", "standardized_nll_support",
            "uncertainty_risk_spearman_support", "reconstruction_temporal_support", "reconstruction_spectral_support",
            "predictive_distribution_contract",
        )
    )
    if not available_models.get("T3a-balloon-robust", False) or not g4_support:
        g4_status = "INCONCLUSIVE"
    elif not all(bool(value) for name, value in g4_checks.items() if not name.endswith("_support")):
        g4_status = "FAIL"
    else:
        g4_status = "PASS"
    if smoke_mode:
        p1_status = "INCONCLUSIVE" if p1_status == "PASS" else p1_status
        p2_status = "INCONCLUSIVE" if p2_status == "PASS" else p2_status
        p3_status = "INCONCLUSIVE" if p3_status == "PASS" else p3_status
        g4_status = "INCONCLUSIVE" if g4_status == "PASS" else g4_status
    return {
        "qualification_mode": "smoke" if smoke_mode else "formal",
        "student_t_dof": float(config["observation"]["student_t_df"]),
        "T-P0": {"status": p0_status, "checks": p0_checks},
        "T-P1": {"status": p1_status, "checks": p1_checks},
        "T-P2": {
            "status": p2_status,
            "checks": {
                "rows": len(t3a_parameter_rows),
                "profile_rows": len([row for row in profile_rows if row.get("model_id") == "T3a-balloon-robust"]),
                "profile_key_support": profile_key_support,
                "profile_computed": profile_computed,
                "profile_geometry": "fixed_other_parameter_objective_slice",
                "best_parameter_key_support": best_key_support,
                "best_rows_identifiable": best_identifiable,
                "best_rows_hessian": best_hessian,
                "best_optimizer_success": best_optimizer_success,
                "multistart_success": start_success,
                "multistart_key_support": start_key_support,
                "max_multistart_spread_fraction": max(spreads) if spreads else None,
                "multistart_spread_pass": multistart_pass,
                "coverage_95": coverage_by_parameter,
                "coverage_pass": coverage_pass,
                "sbc_rank_ks_pvalue": rank_pvalue_by_parameter,
                "sbc_rank_pass": rank_pass,
                "sbc_method": "EKF_Laplace_posterior_CDF_approximation",
                "truth_parameter_design": config["simulation"]["truth_parameter_design"],
            },
        },
        "T-P3": {"status": p3_status, "checks": p3_checks},
        "synthetic-T-G4": {"status": g4_status, "checks": g4_checks},
        "models": {name: {"available": bool(available), "role": "panel"} for name, available in available_models.items()},
    }


TRAJECTORY_FIELDS = (
    "time_s", "scenario_id", "replicate_id", "model_id", "coordinate", "artifact_scope", "clean_truth",
    "corrupted_observation", "posterior_mean", "artifact", "artifact_mask", "valid",
    "aleatoric_variance", "epistemic_variance", "total_variance",
)
STATE_FIELDS = (
    "time_s", "scenario_id", "replicate_id", "model_id", "state_name", "truth",
    "posterior_mean", "state_variance", "unit", "state_valid",
)
UNCERTAINTY_FIELDS = (
    "time_s", "scenario_id", "replicate_id", "model_id", "component",
    "aleatoric_variance", "epistemic_variance", "total_variance", "status",
)
METRIC_FIELDS = (
    "scenario_id", "stress_case", "severity", "mask_fraction", "null_type", "artifact_scope", "replicate_id",
    "model_id", "target", "metric", "value", "lower", "upper", "units", "status",
)
PARAMETER_FIELDS = (
    "model_id", "replicate_id", "scenario_id", "parameter_name", "true_value", "estimate",
    "sd", "lower", "upper", "relative_error", "identifiability_status", "start_id", "objective",
    "likelihood_hessian_status", "data_identifiability_status", "approximation",
    "optimizer_success", "covered_95", "sbc_cdf",
)
CALIBRATION_FIELDS = (
    "target", "nominal_level", "empirical_coverage", "pit", "crps", "nll",
    "uncertainty_risk_spearman", "interval_width_95", "distribution", "student_nu", "method",
    "group", "model_id", "replicate_id", "status", "support_n",
)
NULL_FIELDS = (
    "null_type", "scenario_id", "replicate_id", "model_id", "metric", "value", "threshold", "pass", "status",
)
PHYSICAL_FIELDS = (
    "model_id", "scenario_id", "replicate_id", "check", "value", "status", "detail",
)
PROFILE_FIELDS = (
    "model_id", "replicate_id", "scenario_id", "parameter_name", "grid_value", "objective", "delta_objective", "status", "geometry_kind",
)


def _truth_physical_rows(case: SyntheticCase) -> list[dict[str, Any]]:
    values = case.truth_states
    checks: dict[str, Any] = {
        "finite": bool(np.all(np.isfinite(values))),
        "positive_fvpq": bool(np.all(values[:, 2:] > 0.0)),
        "rest_initial": bool(np.allclose(values[0], np.asarray([case.truth_states[0, 0], 0.0, 1.0, 1.0, 1.0, 1.0]), atol=1e-7)),
    }
    if checks["positive_fvpq"]:
        e0 = float(case.true_parameters["e0"])
        extraction = -np.expm1(np.log1p(-e0) / values[:, 2])
        checks["extraction_in_0_1"] = bool(np.all((extraction > 0.0) & (extraction < 1.0)))
        p0, q0 = case.true_parameters["p0"], case.true_parameters["q0"]
        hbt, hbr = p0 * values[:, 4], q0 * values[:, 5]
        checks["absolute_hb_nonnegative"] = bool(np.all((hbt >= 0.0) & (hbr >= 0.0) & (hbt - hbr >= 0.0)))
        checks["hbr_not_above_hbt"] = bool(np.all(hbr <= hbt + 1e-12))
    else:
        checks["extraction_in_0_1"] = False
        checks["absolute_hb_nonnegative"] = False
        checks["hbr_not_above_hbt"] = False
    # Residual against the declared continuous RHS and a finite-difference
    # step-size sentinel make solver/integration failures visible in the
    # evidence rather than silently treating finite arrays as truth.
    rhs = (
        np.vstack([_truth_compartment_rhs(row, case.true_parameters, driver=float(row[0])) for row in values])
        if checks["finite"]
        else np.full_like(values, np.nan)
    )
    numerical = np.gradient(values, case.time_s, axis=0) if checks["finite"] else np.full_like(values, np.nan)
    residual = np.abs(numerical[1:-1, 1:] - rhs[1:-1, 1:])
    max_residual = float(np.nanmax(residual)) if np.any(np.isfinite(residual)) else float("nan")
    checks["compartment_rhs_finite"] = bool(np.isfinite(max_residual))
    checks["compartment_rhs_residual"] = bool(np.isfinite(max_residual) and max_residual <= 0.05)
    return [
        {
            "model_id": "generator",
            "scenario_id": case.scenario_id,
            "replicate_id": case.replicate_id,
            "check": name,
            "value": value,
            "status": "pass" if bool(value) else "fail",
            "detail": "independent solve_ivp truth; generator state process noise=0",
        }
        for name, value in checks.items()
    ]


def _apply_smoke_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    result = _copy_config(config)
    result["experiment"]["qualification_mode"] = "smoke"
    simulation = result["simulation"]
    simulation["duration_s"] = min(float(simulation["duration_s"]), 24.0)
    simulation["replicates"] = min(int(simulation["replicates"]), 2)
    simulation["prior_predictive_draws"] = min(int(simulation["prior_predictive_draws"]), 4)
    simulation["severity_levels"] = [0.0, max(float(max(simulation["severity_levels"])), 0.75)]
    inference = result["inference"]
    inference["multistarts"] = min(int(inference.get("multistarts", 2)), 2)
    inference["max_iterations"] = min(int(inference.get("max_iterations", 8)), 8)
    inference["profile_points"] = min(int(inference.get("profile_points", 3)), 3)
    return result


def _make_cases(config: Mapping[str, Any]) -> tuple[list[SyntheticCase], list[SyntheticCase]]:
    seed = int(config["experiment"].get("seed", 0))
    replicates = int(config["simulation"]["replicates"])
    truth_rng = np.random.default_rng(seed + 400_000)
    bases = [
        generate_case(
            index,
            seed + index * 1009,
            config,
            sampled_parameters=_sample_free_parameters(truth_rng, config),
        )
        for index in range(replicates)
    ]
    stress: list[SyntheticCase] = list(bases)
    levels = [float(value) for value in config["simulation"]["severity_levels"] if float(value) > 0.0]
    for index, base in enumerate(bases):
        for family_index, family in enumerate(config["stress_tests"]["artifact_families"]):
            for severity in levels:
                stress.append(_derive_stress_case(
                    base,
                    config,
                    str(family),
                    severity,
                    seed + index * 1009 + 50000 + family_index * 97,
                ))
        # Missing-modality replay keeps the same latent truth and observation
        # noise as the clean base while removing one measurement stream.
        stress.append(_derive_missing_modality_case(base, "missing_eeg"))
        stress.append(_derive_missing_modality_case(base, "missing_fnirs"))
    nulls: list[SyntheticCase] = []
    # Pairing is a deterministic cohort permutation; independent creates a
    # fresh driver so the two null operators are not aliases.
    for index, base in enumerate(bases):
        if len(bases) < 2:
            continue
        paired = bases[(index + 1) % len(bases)]
        independent = generate_case(
            base.replicate_id,
            seed + 900000 + index * 101,
            config,
            sampled_parameters=_sample_free_parameters(truth_rng, config),
        )
        if "pairing" in config["stress_tests"]["nulls"]:
            nulls.append(make_null_case(base, paired, "pairing", config))
        if "independent" in config["stress_tests"]["nulls"]:
            nulls.append(make_null_case(base, independent, "independent", config))
        if "time_shift" in config["stress_tests"]["nulls"]:
            nulls.append(make_null_case(base, base, "time_shift", config))
    return bases, stress + nulls


def _t3a_available(panel: Sequence[CandidateAdapter]) -> dict[str, bool]:
    result = {}
    for candidate in panel:
        result[candidate.model_id] = bool(getattr(candidate, "available", True))
    return result


def run_suite(config: Mapping[str, Any], output_dir: Path) -> Path:
    """Run P0 in a fresh directory and leave an explicit incomplete record on error."""

    validate_config(config)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    _atomic_json(output_dir / "summary.json", {
        "schema": SCHEMA,
        "status": "incomplete",
        "completion_status": "incomplete",
        "started_at": started,
        "scope": "synthetic_only",
    })
    try:
        bases, cases = _make_cases(config)
        prior_rows = _prior_predictive(
            config,
            int(config["simulation"]["prior_predictive_draws"]),
            int(config["experiment"].get("seed", 0)) + 700000,
        )
        panel = candidate_panel()
        available_models = _t3a_available(panel)
        t3a_candidate = next((item for item in panel if item.model_id == "T3a-balloon-robust"), None)
        if t3a_candidate is None or not available_models.get("T3a-balloon-robust", False):
            detail = getattr(t3a_candidate, "import_error", None) or "fit_balloon API unavailable"
            raise RuntimeError(f"T3a candidate is required for P0 but unavailable: {detail}")
        fit_by_model: dict[str, dict[int, Any]] = {candidate.model_id: {} for candidate in panel}
        fit_errors: dict[str, dict[int, str]] = {candidate.model_id: {} for candidate in panel}
        prediction_errors: dict[str, dict[str, str]] = {candidate.model_id: {} for candidate in panel}
        for candidate in panel:
            for base in bases:
                try:
                    fit_by_model[candidate.model_id][base.replicate_id] = candidate.fit([base], config)
                except Exception as exc:  # one unavailable arm must not hide other P0 evidence
                    fit_by_model[candidate.model_id][base.replicate_id] = None
                    fit_errors[candidate.model_id][base.replicate_id] = f"{type(exc).__name__}: {exc}"

        trajectory_rows: list[dict[str, Any]] = []
        state_rows: list[dict[str, Any]] = []
        uncertainty_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        calibration_rows: list[dict[str, Any]] = []
        null_rows: list[dict[str, Any]] = []
        parameter_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        physical_rows: list[dict[str, Any]] = []
        for case in cases:
            if case.null_type is None and case.stress_case == "clean":
                physical_rows.extend(_truth_physical_rows(case))
            for candidate in panel:
                fit = fit_by_model[candidate.model_id].get(case.replicate_id)
                if fit is None:
                    prediction = _empty_prediction(candidate.model_id, len(case.time_s))
                    prediction.metadata = {"status": "not_available", "error": fit_errors[candidate.model_id].get(case.replicate_id, "fit unavailable")}
                else:
                    try:
                        prediction = candidate.predict(fit, case, config)
                    except Exception as exc:
                        prediction = _empty_prediction(candidate.model_id, len(case.time_s))
                        prediction.metadata = {"status": "prediction_failed", "error": f"{type(exc).__name__}: {exc}"}
                        prediction_errors[candidate.model_id][
                            f"{case.replicate_id}:{case.scenario_id}"
                        ] = prediction.metadata["error"]
                trajectory_rows.extend(_trajectory_rows(case, prediction))
                if case.null_type is None:
                    states, state_metrics = _state_rows(case, prediction)
                    state_rows.extend(states)
                    metric_rows.extend(state_metrics)
                obs_metrics, calibration, uncertainty = _observation_metrics(case, prediction)
                metric_rows.extend(obs_metrics)
                calibration_rows.extend(calibration)
                uncertainty_rows.extend(uncertainty)
                if case.null_type:
                    value = _null_metric(case, prediction)
                    threshold = float(config["gates"].get("max_null_driver_correlation", 0.35))
                    passed = bool(np.isfinite(value) and abs(value) <= threshold)
                    null_rows.append({
                        "null_type": case.null_type,
                        "scenario_id": case.scenario_id,
                        "replicate_id": case.replicate_id,
                        "model_id": candidate.model_id,
                        "metric": "cross_modal_driver_correlation",
                        "value": value,
                        "threshold": threshold,
                        "pass": passed if np.isfinite(value) else None,
                        "status": "computed" if np.isfinite(value) else "not_applicable",
                    })
                if (
                    case.null_type is None
                    and case.stress_case == "clean"
                    and candidate.model_id == "T3a-balloon-robust"
                ):
                    parameter = _parameter_rows(prediction, case.true_parameters, fit)
                    for row in parameter:
                        row.update({"model_id": candidate.model_id, "replicate_id": case.replicate_id, "scenario_id": case.scenario_id})
                    parameter_rows.extend(parameter)
                    if case.replicate_id == bases[0].replicate_id:
                        profile = _profile_rows(fit, config, case.true_parameters, case)
                        for row in profile:
                            row.update({"model_id": candidate.model_id, "replicate_id": case.replicate_id, "scenario_id": case.scenario_id})
                        profile_rows.extend(profile)
                physical = prediction.metadata.get("physical_checks") if isinstance(prediction.metadata, Mapping) else None
                if isinstance(physical, Mapping):
                    for name, value in physical.items():
                        physical_rows.append({
                            "model_id": candidate.model_id,
                            "scenario_id": case.scenario_id,
                            "replicate_id": case.replicate_id,
                            "check": name,
                            "value": value,
                            "status": "pass" if bool(value) else "fail",
                            "detail": "candidate-reported physical check",
                        })

        runtime_models = {
            model_id: bool(
                available_models.get(model_id, False)
                and not fit_errors[model_id]
                and not prediction_errors[model_id]
            )
            for model_id in available_models
        }
        gates = _gate_results(
            config,
            prior_rows,
            metric_rows,
            calibration_rows,
            null_rows,
            parameter_rows,
            available_models=runtime_models,
            profile_rows=profile_rows,
            contract_checks=_contract_checks(config),
            physical_rows=physical_rows,
        )
        _atomic_csv(output_dir / "trajectories.csv", trajectory_rows, TRAJECTORY_FIELDS)
        _atomic_csv(output_dir / "states.csv", state_rows, STATE_FIELDS)
        _atomic_csv(output_dir / "uncertainty.csv", uncertainty_rows, UNCERTAINTY_FIELDS)
        _atomic_csv(output_dir / "metrics.csv", metric_rows, METRIC_FIELDS)
        _atomic_csv(output_dir / "parameter_recovery.csv", parameter_rows, PARAMETER_FIELDS)
        _atomic_csv(output_dir / "calibration.csv", calibration_rows, CALIBRATION_FIELDS)
        _atomic_csv(output_dir / "null_metrics.csv", null_rows, NULL_FIELDS)
        _atomic_csv(output_dir / "physical_checks.csv", physical_rows, PHYSICAL_FIELDS)
        _atomic_csv(output_dir / "prior_predictive.csv", prior_rows)
        _atomic_csv(output_dir / "profile_likelihood.csv", profile_rows, PROFILE_FIELDS)
        _atomic_json(output_dir / "gates.json", gates)
        _atomic_write(output_dir / "resolved_config.yaml", yaml.safe_dump(_jsonable(config), sort_keys=False))
        completed = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema": SCHEMA,
            "suite": str(config["experiment"].get("name", SCHEMA)),
            "scope": "synthetic_only",
            "protected_data_enabled": False,
            "measured_data_enabled": False,
            "completion_status": "complete",
            "started_at": started,
            "completed_at": completed,
            "panel": [candidate.model_id for candidate in panel],
            "generator": {
                "implementation": "independent scipy.solve_ivp",
                "equations": "minimal Tak tau_v=0; tau*dp/dt=f-f_out*p/v",
                "state_process_noise": 0.0,
                "driver_process_noise": "continuous OU diffusion discretized as diffusion*sqrt(dt)",
                "driver_deterministic_pulse": "configured exogenous Gaussian pulse; recorded as challenge",
                "observation_noise": "heteroscedastic Student-t; injected independently",
                "geometry": "not generated; spatial null not applicable",
            },
            "counts": {
                "base_cases": len(bases),
                "cases": len(cases),
                "trajectory_rows": len(trajectory_rows),
                "state_rows": len(state_rows),
                "metric_rows": len(metric_rows),
                "null_rows": len(null_rows),
            },
            "model_import_availability": available_models,
            "model_availability": runtime_models,
            "fit_errors": fit_errors,
            "prediction_errors": prediction_errors,
            "artifacts": [
                "resolved_config.yaml", "trajectories.csv", "states.csv", "uncertainty.csv", "metrics.csv",
                "parameter_recovery.csv", "calibration.csv", "null_metrics.csv", "physical_checks.csv",
                "prior_predictive.csv", "profile_likelihood.csv", "gates.json", "summary.json",
            ],
        }
        _atomic_json(output_dir / "manifest.json", manifest)
        summary = {
            "schema": SCHEMA,
            "status": "complete",
            "completion_status": "complete",
            "scope": "synthetic_only",
            "gates": gates,
            "counts": manifest["counts"],
            "models": runtime_models,
        }
        _atomic_json(output_dir / "summary.json", summary)
        return output_dir
    except Exception as exc:
        _atomic_json(output_dir / "summary.json", {
            "schema": SCHEMA,
            "status": "incomplete",
            "completion_status": "incomplete",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        })
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="run a small synthetic-only smoke suite")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.smoke:
        config = _apply_smoke_overrides(config)
    output = args.output_dir
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_t3a_balloon_robust_p0")
        output = REPO_ROOT / "experiments/runs/physiology_semantic_tokenizer/t3a_balloon_robust_p0" / stamp
    return run_suite(config, output)


if __name__ == "__main__":
    print(main())
