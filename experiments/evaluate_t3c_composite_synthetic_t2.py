#!/usr/bin/env python3
"""Synthetic-only T-P2 screen for the C1 gain and time composites.

The fitter receives only noisy training observations.  Known truth, drivers,
and held-out observations stay in the evaluation process and are joined by
replicate id after fitting.  No measured-data module is imported here.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import scipy
import yaml
from scipy.integrate import cumulative_trapezoid, solve_ivp, trapezoid
from scipy.optimize import minimize
from scipy.stats import beta as beta_distribution
from scipy.stats import kstest, t as student_t, truncnorm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.t3a_balloon_robust_ssm import (
    BalloonConfig,
    BalloonFixedParameters,
    BalloonFreeParameters,
    BalloonObservationSpec,
    BalloonParameters,
    simulate_balloon,
    smooth_balloon,
)


SCHEMA = "t3c_composite_synthetic_t2_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2_v1.yaml"
OUTPUT_ROOT = "experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2"
CANDIDATES = ("C1_G", "C1_T")
COORDINATE_FOR = {"C1_G": "log_gain_relative", "C1_T": "log_time_relative"}
OBSERVATION_NAMES = ("EEG", "HbO", "HbR")
EXPECTED_SOURCES = {
    "model": ("src/inference/t3a_balloon_robust_ssm.py", "4221b4a53e9b2041d6db5e0274e4d5b509ab8108291e004c01ab67d9054326e4"),
    "synthetic_reference": ("experiments/configs/physiology_semantic_tokenizer/t3a_balloon_robust_p0.yaml", "f8343378c00cb8e0237aba6db82a4bebdf383816ca1f70daccb480c23ce16e31"),
    "composite_contract": ("experiments/configs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission_v1.yaml", "6ce049571bd268f5e68e7f3f1b1e43a66f875b06e3b056b28e36c7ec77bec0b1"),
}


@dataclass(frozen=True)
class FitDataset:
    """Only the inputs allowed to cross the truth-to-fitter boundary."""

    candidate: str
    replicate_id: int
    train_observations: tuple[np.ndarray, ...]
    optimizer_seed: int
    fit_contract: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_jsonable) + "\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})
    os.replace(temporary, path)


def _git_payload() -> dict[str, str]:
    def call(*args: str) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, check=False, capture_output=True, text=True).stdout.strip()

    return {"commit": call("git", "rev-parse", "HEAD"), "status_short": call("git", "status", "--short")}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("T-P2 config must be a mapping")
    return dict(value)


def raw_from_relative(
    log_gain_relative: float,
    log_time_relative: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Invert the fixed-zeta/fixed-Tv composite map without clipping."""

    g = float(log_gain_relative)
    t = float(log_time_relative)
    if not np.isfinite(g + t):
        raise ValueError("composite coordinates must be finite")
    reference = config["composite"]["reference"]
    raw = {
        "beta": float(reference["beta"]) * math.exp(g - 2.0 * t),
        "kappa": float(reference["kappa"]) * math.exp(-t),
        "gamma": float(reference["gamma"]) * math.exp(-2.0 * t),
        "tau": float(reference["tau"]),
        "alpha": float(reference["alpha"]),
        "E0": float(reference["E0"]),
    }
    for name, (lower, upper) in config["composite"]["raw_bounds"].items():
        value = raw[name]
        if not float(lower) <= value <= float(upper):
            raise ValueError(f"induced raw parameter is outside its registered bound: {name}={value}")
    return raw


def relative_from_raw(raw: Mapping[str, float], config: Mapping[str, Any]) -> dict[str, float]:
    reference = config["composite"]["reference"]
    gain = float(raw["beta"]) / float(raw["gamma"])
    time_scale = 1.0 / math.sqrt(float(raw["gamma"]))
    gain_reference = float(reference["beta"]) / float(reference["gamma"])
    time_reference = 1.0 / math.sqrt(float(reference["gamma"]))
    return {
        "log_gain_relative": math.log(gain / gain_reference),
        "log_time_relative": math.log(time_scale / time_reference),
    }


def _validate_sources(config: Mapping[str, Any]) -> None:
    for source in config["sources"].values():
        path = REPO_ROOT / str(source["path"])
        if not path.is_file() or _sha256(path) != str(source["sha256"]):
            raise ValueError(f"registered source hash mismatch: {source['path']}")


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ValueError("T-P2 config schema mismatch")
    experiment = config.get("experiment", {})
    expected = {
        "name": SCHEMA,
        "scope": "synthetic_known_truth_only",
        "measured_data_enabled": False,
        "validation_data_enabled": False,
        "protected_data_enabled": False,
        "qualification_eligible": False,
        "decision_eligibility": False,
        "seed": 20260903,
    }
    if any(experiment.get(key) != value for key, value in expected.items()):
        raise ValueError("T-P2 experiment boundary mismatch")
    if set(config.get("sources", {})) != set(EXPECTED_SOURCES) or any(
        config["sources"][name].get("path") != path or config["sources"][name].get("sha256") != sha256
        for name, (path, sha256) in EXPECTED_SOURCES.items()
    ):
        raise ValueError("registered source registry drifted")
    _validate_sources(config)
    composite = config.get("composite", {})
    if tuple(composite.get("coordinates", ())) != ("log_gain_relative", "log_time_relative"):
        raise ValueError("composite coordinates have drifted")
    if tuple(composite.get("candidates", {})) != CANDIDATES:
        raise ValueError("C1 candidate order has drifted")
    if tuple(composite.get("fixed_gauge", ())) != ("zeta", "tv", "alpha", "E0", "P0", "Q0", "optical_observation", "eeg_loading"):
        raise ValueError("fixed composite gauge drifted")
    expected_reference = {"beta": 1.0, "kappa": 0.64, "gamma": 0.32, "tau": 2.0, "alpha": 0.32, "E0": 0.32}
    expected_raw_bounds = {"beta": [0.25, 4.0], "kappa": [0.20, 1.50], "gamma": [0.10, 1.00], "tau": [0.50, 5.00], "alpha": [0.10, 0.80], "E0": [0.10, 0.80]}
    if composite.get("reference") != expected_reference or composite.get("raw_bounds") != expected_raw_bounds:
        raise ValueError("composite reference or raw bounds drifted")
    if composite.get("c2_policy") != "separate_run_only_if_C1_G_and_C1_T_pass":
        raise ValueError("C2 fail-closed policy is required")
    for candidate in CANDIDATES:
        spec = composite["candidates"][candidate]
        if tuple(spec.get("active", ())) != (COORDINATE_FOR[candidate],):
            raise ValueError(f"{candidate} must have exactly its registered scalar coordinate")
        expected_fixed = {"log_time_relative": 0.0} if candidate == "C1_G" else {"log_gain_relative": 0.0}
        if spec.get("fixed") != expected_fixed:
            raise ValueError(f"{candidate} fixed companion coordinate drifted")
        registered = tuple(map(float, spec.get("registered_bounds", ())))
        expanded = tuple(map(float, spec.get("expanded_bounds", ())))
        support = tuple(map(float, spec.get("truth_support", ())))
        if len(registered) != 2 or len(expanded) != 2 or len(support) != 2:
            raise ValueError("coordinate bounds are malformed")
        if not (expanded[0] < registered[0] < support[0] < support[1] < registered[1] < expanded[1]):
            raise ValueError(f"{candidate} support/registered/expanded bounds are not nested")
        for value in (*registered, *expanded, *support):
            g, t = (value, 0.0) if candidate == "C1_G" else (0.0, value)
            raw_from_relative(g, t, config)
    expected_candidate_numbers = {
        "C1_G": {"registered_bounds": [-0.60, 0.60], "expanded_bounds": [-0.80, 0.80], "truth_prior_mean": 0.0, "truth_prior_sd": 0.20, "truth_support": [-0.40, 0.40]},
        "C1_T": {"registered_bounds": [-0.50, 0.50], "expanded_bounds": [-0.55, 0.55], "truth_prior_mean": 0.0, "truth_prior_sd": 0.15, "truth_support": [-0.40, 0.40]},
    }
    if any(any(composite["candidates"][candidate].get(key) != value for key, value in values.items()) for candidate, values in expected_candidate_numbers.items()):
        raise ValueError("candidate prior or bound contract drifted")
    simulation = config.get("simulation", {})
    frozen_simulation = {
        "independent_replicates_per_direction": 60,
        "sampling_hz": 4.0,
        "duration_s_per_trial": 40.0,
        "train_trials_per_replicate": 3,
        "heldout_trials_per_replicate": 1,
        "heldout_fnirs_mask_from_s": 12.0,
        "state_process_noise": False,
    }
    if any(simulation.get(key) != value for key, value in frozen_simulation.items()):
        raise ValueError("formal simulation design drifted")
    if int(simulation["independent_replicates_per_direction"]) < int(config["gates"]["minimum_independent_sbc_replicates"]):
        raise ValueError("formal SBC does not contain enough independent truth replicates")
    mask_from = float(simulation.get("heldout_fnirs_mask_from_s", float("nan")))
    if not np.isfinite(mask_from) or not 0.0 < mask_from < float(simulation.get("duration_s_per_trial", 0.0)):
        raise ValueError("held-out fNIRS mask onset must lie inside every trial")
    if tuple(simulation.get("heldout_score_coordinates", ())) != ("HbO", "HbR"):
        raise ValueError("held-out scoring must target the two masked fNIRS coordinates")
    if simulation.get("independent_trial_reset") is not True or simulation.get("truth_integrator") != "independent_solve_ivp":
        raise ValueError("independent reset and truth integration contracts are required")
    if simulation.get("noise", {}).get("family") != "student_t_homoscedastic":
        raise ValueError("noise family drifted")
    expected_driver = {"decay_per_s": 0.45, "diffusion_sd_per_sqrt_s": 0.08, "pulse_amplitude": 0.12}
    expected_noise = {"family": "student_t_homoscedastic", "df": 5.0, "scale": {"EEG": 0.080, "HbO": 0.025, "HbR": 0.015}}
    expected_observation = {
        "P0": 1.0,
        "Q0": 0.35,
        "eeg_loading": 1.0,
        "eeg_offset": 0.0,
        "process_sd": {"r": 0.080, "s": 0.010, "log_f": 0.006, "log_v": 0.004, "log_p": 0.004, "log_q": 0.004},
    }
    if simulation.get("driver") != expected_driver or simulation.get("noise") != expected_noise or simulation.get("observation") != expected_observation:
        raise ValueError("synthetic driver/noise/observation gauge drifted")
    inference = config.get("inference", {})
    exact = {"multistarts": 16, "profile_points": 21, "posterior_grid_points": 81, "workers": 16}
    if any(int(inference.get(key, -1)) != value for key, value in exact.items()):
        raise ValueError("formal optimizer/grid counts drifted")
    if inference.get("primary_sbc_approximation") != "EKF_Laplace_truncated_normal" or inference.get("fallback_sbc_approximation") != "exact_1d_grid_quadrature_under_EKF_likelihood":
        raise ValueError("SBC approximation/fallback contract drifted")
    if inference.get("objective") != "likelihood_only_for_point_fit_and_profile" or inference.get("sbc_posterior") != "truncated_normal_prior_with_EKF_likelihood":
        raise ValueError("fit/SBC objective contract drifted")
    if inference.get("fallback_selection_rule") != "use_grid_only_if_Laplace_fails_prefrozen_calibration_gates":
        raise ValueError("SBC fallback selection rule drifted")
    if not np.isclose(float(inference.get("boundary_fraction_of_span", float("nan"))), 0.01):
        raise ValueError("boundary diagnostic fraction drifted")
    frozen_inference = {
        "optimizer_max_iterations": 60,
        "optimizer_ftol": 1.0e-7,
        "profile_delta_nll": 1.920729410347062,
        "profile_reference_tolerance_nll": 0.01,
        "posterior_hessian_step": 0.005,
        "sensitivity_step": 0.01,
        "sensitivity_relative_singular_threshold": 0.05,
        "prediction_equivalence_whitened_rmse": 0.10,
        "material_parameter_difference_fraction_of_span": 0.01,
        "driver_stability_max_nrmse": 0.10,
        "driver_stability_min_correlation": 0.95,
        "bootstrap_repetitions": 10000,
    }
    if any(not np.isclose(float(inference.get(key, float("nan"))), value) for key, value in frozen_inference.items()):
        raise ValueError("formal inference setting drifted")
    expected_gates = {
        "minimum_independent_sbc_replicates": 60,
        "maximum_solver_failure_fraction": 0.0,
        "maximum_estimate_boundary_contact_fraction": 0.0,
        "maximum_profile_boundary_contact_fraction": 0.05,
        "minimum_profile_truth_coverage": 0.90,
        "maximum_profile_truth_coverage": 0.99,
        "maximum_profile_reference_difference_nll": 0.01,
        "minimum_multistart_success_fraction": 1.0,
        "maximum_multistart_nll_spread_per_observation": 0.10,
        "maximum_multistart_parameter_spread_fraction": 0.10,
        "maximum_expanded_outside_registered_near_optimal_count": 0,
        "maximum_material_prediction_equivalent_count": 0,
        "sbc_ks_alpha_critical_multiplier": 1.36,
        "minimum_sbc_rank_mean": 0.45,
        "maximum_sbc_rank_mean": 0.55,
        "minimum_sbc_coverage_95": 0.90,
        "maximum_sbc_coverage_95": 0.99,
        "required_nominal_coverage_in_binomial_interval": 0.95,
        "minimum_posterior_sd_grid_steps": 2.0,
        "maximum_posterior_grid_cdf_difference": 0.02,
        "maximum_absolute_bias_fraction_of_span": 0.05,
        "maximum_rmse_fraction_of_span": 0.20,
        "maximum_bias_ci_extent_fraction_of_span": 0.10,
        "maximum_heldout_excess_oracle_nll_per_observation": 0.10,
        "heldout_noninferiority_margin_nll_per_observation": 0.05,
    }
    if any(not np.isclose(float(config.get("gates", {}).get(key, float("nan"))), value) for key, value in expected_gates.items()):
        raise ValueError("formal decision threshold drifted")
    if config.get("output", {}).get("root") != OUTPUT_ROOT:
        raise ValueError("T-P2 output root drifted")


def _effective_config(config: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result["experiment"]["run_mode"] = "smoke" if smoke else "formal"
    if smoke:
        result["simulation"]["independent_replicates_per_direction"] = 2
        result["simulation"]["duration_s_per_trial"] = 12.0
        result["simulation"]["heldout_fnirs_mask_from_s"] = 4.0
        result["simulation"]["train_trials_per_replicate"] = 1
        result["inference"]["multistarts"] = 3
        result["inference"]["workers"] = 1
        result["inference"]["profile_points"] = 5
        result["inference"]["posterior_grid_points"] = 9
    return result


def _fit_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Strip realized-truth generation seeds and held-out data from worker input."""

    return {
        "composite": copy.deepcopy(config["composite"]),
        "simulation": {
            "sampling_hz": float(config["simulation"]["sampling_hz"]),
            "noise": copy.deepcopy(config["simulation"]["noise"]),
            "observation": copy.deepcopy(config["simulation"]["observation"]),
            "driver": {"decay_per_s": float(config["simulation"]["driver"]["decay_per_s"])},
        },
        "inference": copy.deepcopy(config["inference"]),
    }


def _parameters(raw: Mapping[str, float], config: Mapping[str, Any]) -> tuple[BalloonParameters, BalloonObservationSpec, BalloonConfig]:
    simulation = config["simulation"]
    observation = simulation["observation"]
    noise = simulation["noise"]
    process = observation["process_sd"]
    fixed = BalloonFixedParameters(
        alpha=float(raw["alpha"]),
        E0=float(raw["E0"]),
        gamma=float(raw["gamma"]),
        P0=float(observation["P0"]),
        Q0=float(observation["Q0"]),
        driver_decay_per_s=float(simulation["driver"]["decay_per_s"]),
        process_std=tuple(float(process[name]) for name in ("r", "s", "log_f", "log_v", "log_p", "log_q")),
        observation_scale=tuple(float(noise["scale"][name]) for name in OBSERVATION_NAMES),
        student_nu=float(noise["df"]),
        eeg_loading=float(observation["eeg_loading"]),
        eeg_offset=float(observation["eeg_offset"]),
        neurovascular_gain=float(raw["beta"]),
    )
    parameters = BalloonParameters(fixed=fixed, free=BalloonFreeParameters(kappa=float(raw["kappa"]), tau=float(raw["tau"])))
    spec = BalloonObservationSpec().resolved(fixed)
    model_config = BalloonConfig(
        dt=1.0 / float(simulation["sampling_hz"]),
        rk4_substeps=2,
        irls_iterations=3,
        optimizer_max_iterations=int(config["inference"]["optimizer_max_iterations"]),
    )
    parameters.validate()
    model_config.validate()
    return parameters, spec, model_config


def _driver(time_s: np.ndarray, rng: np.random.Generator, config: Mapping[str, Any]) -> np.ndarray:
    spec = config["simulation"]["driver"]
    dt = float(time_s[1] - time_s[0])
    phi = math.exp(-float(spec["decay_per_s"]) * dt)
    values = np.zeros(len(time_s), dtype=np.float64)
    for index in range(1, len(values)):
        values[index] = phi * values[index - 1] + rng.normal(scale=float(spec["diffusion_sd_per_sqrt_s"]) * math.sqrt(dt))
    center = 0.52 * float(time_s[-1])
    width = max(0.8, 0.06 * float(time_s[-1]))
    values += float(spec["pulse_amplitude"]) * np.exp(-0.5 * ((time_s - center) / width) ** 2)
    return values


def _truth_trial(raw: Mapping[str, float], seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    simulation = config["simulation"]
    fs_hz = float(simulation["sampling_hz"])
    count = max(8, int(round(float(simulation["duration_s_per_trial"]) * fs_hz)))
    time_s = np.arange(count, dtype=np.float64) / fs_hz
    rng = np.random.default_rng(int(seed))
    driver = _driver(time_s, rng, config)
    beta = float(raw["beta"])
    kappa = float(raw["kappa"])
    gamma = float(raw["gamma"])
    tau = float(raw["tau"])
    alpha = float(raw["alpha"])
    e0 = float(raw["E0"])

    def rhs(current_time: float, state: np.ndarray) -> np.ndarray:
        s, flow, volume, total_hb, deoxy_hb = state
        if not np.all(np.isfinite(state)) or min(flow, volume, total_hb, deoxy_hb) <= 0.0:
            raise FloatingPointError("truth integration crossed a positive state boundary")
        flow_out = float(volume ** (1.0 / alpha))
        extraction = float(-np.expm1(np.log1p(-e0) / flow))
        neural = float(np.interp(current_time, time_s, driver))
        return np.asarray([
            beta * neural - kappa * s - gamma * (flow - 1.0),
            s,
            (flow - flow_out) / tau,
            (flow - flow_out * total_hb / volume) / tau,
            (flow * extraction / e0 - flow_out * deoxy_hb / volume) / tau,
        ])

    try:
        solved = solve_ivp(
            rhs,
            (float(time_s[0]), float(time_s[-1])),
            np.asarray([0.0, 1.0, 1.0, 1.0, 1.0]),
            t_eval=time_s,
            rtol=1.0e-8,
            atol=1.0e-10,
            max_step=(1.0 / fs_hz) / 4.0,
        )
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise RuntimeError("independent composite truth integration failed") from exc
    if not solved.success or solved.y.shape != (5, count):
        raise RuntimeError("independent composite truth integration failed")
    states = np.column_stack((driver, solved.y.T))
    p0 = float(simulation["observation"]["P0"])
    q0 = float(simulation["observation"]["Q0"])
    delta_hbt = p0 * (states[:, 4] - 1.0)
    delta_hbr = q0 * (states[:, 5] - 1.0)
    clean = np.column_stack((driver, delta_hbt - delta_hbr, delta_hbr))
    scales = np.asarray([float(simulation["noise"]["scale"][name]) for name in OBSERVATION_NAMES])
    observations = clean + rng.standard_t(float(simulation["noise"]["df"]), size=clean.shape) * scales[None, :]
    physical_valid = bool(np.all(np.isfinite(states)) and np.all(states[:, 2:] > 0.0))
    physical_valid &= bool(np.all(p0 * states[:, 4] + 1.0e-12 >= q0 * states[:, 5]))
    if not physical_valid or not np.all(np.isfinite(observations)):
        raise RuntimeError("synthetic truth failed physical or finite checks")
    return {"observations": observations, "driver": driver, "clean": clean, "states": states, "physical_valid": physical_valid}


def _coordinate_raw(candidate: str, value: float, config: Mapping[str, Any]) -> dict[str, float]:
    if candidate == "C1_G":
        return raw_from_relative(value, 0.0, config)
    if candidate == "C1_T":
        return raw_from_relative(0.0, value, config)
    raise ValueError(f"unknown candidate: {candidate}")


def _nll_function(dataset: FitDataset) -> tuple[Any, int, Any, Any]:
    raw_reference = _coordinate_raw(dataset.candidate, 0.0, dataset.fit_contract)
    _, observation_spec, model_config = _parameters(raw_reference, dataset.fit_contract)
    cache: dict[float, float] = {}

    def objective(vector: np.ndarray | Sequence[float] | float) -> float:
        value = float(np.asarray(vector, dtype=np.float64).reshape(-1)[0])
        key = round(value, 12)
        if key in cache:
            return cache[key]
        try:
            raw = _coordinate_raw(dataset.candidate, value, dataset.fit_contract)
            parameters, _, _ = _parameters(raw, dataset.fit_contract)
            nll = -sum(
                float(smooth_balloon(observations, parameters=parameters, observation_spec=observation_spec, config=model_config).predictive_log_likelihood)
                for observations in dataset.train_observations
            )
            cache[key] = float(nll) if np.isfinite(nll) else 1.0e12
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            cache[key] = 1.0e12
        return cache[key]

    n_observations = int(sum(np.count_nonzero(np.isfinite(value)) for value in dataset.train_observations))
    return objective, n_observations, observation_spec, model_config


def _starts(lower: float, upper: float, count: int, seed: int) -> tuple[float, ...]:
    if count < 1:
        raise ValueError("at least one start is required")
    if count == 1:
        return (float(np.clip(0.0, lower, upper)),)
    rng = np.random.default_rng(seed)
    lhs = (rng.permutation(count - 1) + rng.random(count - 1)) / float(count - 1)
    return (float(np.clip(0.0, lower, upper)), *tuple(float(lower + value * (upper - lower)) for value in lhs))


def _optimize_starts(
    objective: Any,
    bounds: tuple[float, float],
    count: int,
    seed: int,
    config: Mapping[str, Any],
    variant: str,
) -> list[dict[str, Any]]:
    result = []
    for start_id, start in enumerate(_starts(*bounds, count, seed)):
        optimized = minimize(
            lambda vector: objective(vector),
            np.asarray([start]),
            method="L-BFGS-B",
            bounds=[bounds],
            options={
                "maxiter": int(config["inference"]["optimizer_max_iterations"]),
                "ftol": float(config["inference"]["optimizer_ftol"]),
                "maxls": 20,
            },
        )
        estimate = float(optimized.x[0])
        nll = float(objective(estimate))
        result.append({
            "variant": variant,
            "start_id": start_id,
            "start": start,
            "estimate": estimate,
            "likelihood_nll": nll,
            "success": bool(optimized.success and nll < 1.0e11),
            "message": str(optimized.message),
            "nfev": int(optimized.nfev),
            "nit": int(optimized.nit),
        })
    return result


def _posterior_grid(
    objective: Any,
    support: tuple[float, float],
    points: int,
    prior_mean: float,
    prior_sd: float,
) -> dict[str, Any]:
    grid = np.linspace(*support, int(points), dtype=np.float64)
    nll = np.asarray([objective(value) for value in grid])
    if not np.all(np.isfinite(nll)) or np.any(nll >= 1.0e11):
        raise RuntimeError("posterior grid contains a failed likelihood evaluation")
    log_density = -nll - 0.5 * np.square((grid - prior_mean) / prior_sd)
    log_density -= float(np.max(log_density))
    density = np.exp(log_density)
    normalizer = float(trapezoid(density, grid))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise RuntimeError("posterior grid could not be normalized")
    density /= normalizer
    cdf = cumulative_trapezoid(density, grid, initial=0.0)
    cdf /= float(cdf[-1])
    mean = float(trapezoid(grid * density, grid))
    variance = float(trapezoid(np.square(grid - mean) * density, grid))
    coarse_grid = grid[::2]
    coarse_log_density = -nll[::2] - 0.5 * np.square((coarse_grid - prior_mean) / prior_sd)
    coarse_log_density -= float(np.max(coarse_log_density))
    coarse_density = np.exp(coarse_log_density)
    coarse_density /= float(trapezoid(coarse_density, coarse_grid))
    coarse_cdf = cumulative_trapezoid(coarse_density, coarse_grid, initial=0.0)
    coarse_cdf /= float(coarse_cdf[-1])
    grid_cdf_difference = float(np.max(np.abs(cdf[::2] - coarse_cdf)))
    return {
        "grid": grid,
        "nll": nll,
        "density": density,
        "cdf": cdf,
        "mean": mean,
        "sd": math.sqrt(max(variance, 0.0)),
        "lower95": float(np.interp(0.025, cdf, grid)),
        "upper95": float(np.interp(0.975, cdf, grid)),
        "coarse_grid_max_cdf_difference": grid_cdf_difference,
    }


def _laplace_posterior(
    objective: Any,
    support: tuple[float, float],
    prior_mean: float,
    prior_sd: float,
    start: float,
    step: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    def posterior_nll(vector: np.ndarray | Sequence[float] | float) -> float:
        value = float(np.asarray(vector).reshape(-1)[0])
        return float(objective(value) + 0.5 * ((value - prior_mean) / prior_sd) ** 2)

    fitted = minimize(
        posterior_nll,
        np.asarray([float(np.clip(start, *support))]),
        method="L-BFGS-B",
        bounds=[support],
        options={"maxiter": int(config["inference"]["optimizer_max_iterations"]), "ftol": float(config["inference"]["optimizer_ftol"])},
    )
    mode = float(fitted.x[0])
    h = min(float(step), 0.25 * (mode - support[0]), 0.25 * (support[1] - mode))
    if not fitted.success or h <= 1.0e-8:
        return {"mode": mode, "sd": float("nan"), "lower95": float("nan"), "upper95": float("nan"), "success": False}
    curvature = (posterior_nll(mode + h) - 2.0 * posterior_nll(mode) + posterior_nll(mode - h)) / h ** 2
    if not np.isfinite(curvature) or curvature <= 0.0:
        return {"mode": mode, "sd": float("nan"), "lower95": float("nan"), "upper95": float("nan"), "success": False}
    sd = math.sqrt(1.0 / curvature)
    a, b = (support[0] - mode) / sd, (support[1] - mode) / sd
    distribution = truncnorm(a, b, loc=mode, scale=sd)
    return {
        "mode": mode,
        "sd": sd,
        "lower95": float(distribution.ppf(0.025)),
        "upper95": float(distribution.ppf(0.975)),
        "success": True,
    }


def _smooth_outputs(dataset: FitDataset, value: float, observation_spec: BalloonObservationSpec, model_config: BalloonConfig) -> tuple[np.ndarray, np.ndarray]:
    parameters, _, _ = _parameters(_coordinate_raw(dataset.candidate, value, dataset.fit_contract), dataset.fit_contract)
    predictions, drivers = [], []
    for observations in dataset.train_observations:
        smoothed = smooth_balloon(observations, parameters=parameters, observation_spec=observation_spec, config=model_config)
        predictions.append(smoothed.trajectory_mean)
        drivers.append(smoothed.state_mean[:, 0])
    return np.concatenate(predictions), np.concatenate(drivers)


def _fit_one(dataset: FitDataset) -> dict[str, Any]:
    config = dataset.fit_contract
    spec = config["composite"]["candidates"][dataset.candidate]
    registered = tuple(map(float, spec["registered_bounds"]))
    expanded = tuple(map(float, spec["expanded_bounds"]))
    count = int(config["inference"]["multistarts"])
    seed = int(dataset.optimizer_seed)
    objective, n_observations, observation_spec, model_config = _nll_function(dataset)
    registered_rows = _optimize_starts(objective, registered, count, seed, config, "registered")
    expanded_rows = _optimize_starts(objective, expanded, count, seed + 1_000_000, config, "expanded")
    finite_registered = [row for row in registered_rows if row["likelihood_nll"] < 1.0e11]
    if not finite_registered:
        raise RuntimeError("all registered multistarts failed")
    best = min(finite_registered, key=lambda row: row["likelihood_nll"])
    best_nll = float(best["likelihood_nll"])
    profile_grid = np.linspace(*registered, int(config["inference"]["profile_points"]))
    profile_nll = np.asarray([objective(value) for value in profile_grid])
    profile_delta = profile_nll - best_nll
    support_mask = np.isfinite(profile_nll) & (profile_nll < 1.0e11) & (profile_delta <= float(config["inference"]["profile_delta_nll"]))
    support_indices = np.flatnonzero(support_mask)
    profile_summary = {
        "all_finite": bool(np.all(np.isfinite(profile_nll)) and np.all(profile_nll < 1.0e11)),
        "support_contiguous": bool(len(support_indices) and np.array_equal(support_indices, np.arange(support_indices[0], support_indices[-1] + 1))),
        "touches_boundary": bool(len(support_indices) and (support_indices[0] == 0 or support_indices[-1] == len(profile_grid) - 1)),
        "reference_difference_nll": float(np.min(profile_nll) - best_nll),
        "support_lower": float(profile_grid[support_indices[0]]) if len(support_indices) else float("nan"),
        "support_upper": float(profile_grid[support_indices[-1]]) if len(support_indices) else float("nan"),
    }
    posterior = _posterior_grid(
        objective,
        tuple(map(float, spec["truth_support"])),
        int(config["inference"]["posterior_grid_points"]),
        float(spec["truth_prior_mean"]),
        float(spec["truth_prior_sd"]),
    )
    laplace = _laplace_posterior(
        objective,
        tuple(map(float, spec["truth_support"])),
        float(spec["truth_prior_mean"]),
        float(spec["truth_prior_sd"]),
        float(best["estimate"]),
        float(config["inference"]["posterior_hessian_step"]),
        config,
    )
    profile_alternatives = [
        {"estimate": float(value), "likelihood_nll": float(nll), "success": bool(nll < 1.0e11)}
        for value, nll in zip(profile_grid, profile_nll)
    ]
    alternatives = [*registered_rows, *profile_alternatives, min(expanded_rows, key=lambda row: row["likelihood_nll"])]
    near = [
        row for row in alternatives
        if row["likelihood_nll"] <= best_nll + float(config["inference"]["profile_delta_nll"])
        and abs(float(row["estimate"]) - float(best["estimate"])) >= float(config["inference"]["material_parameter_difference_fraction_of_span"]) * (registered[1] - registered[0])
    ]
    confounding = {
        "alternative_estimate": float("nan"),
        "parameter_distance_fraction": 0.0,
        "observation_whitened_rmse": float("nan"),
        "driver_nrmse": float("nan"),
        "driver_correlation": float("nan"),
        "prediction_equivalent": False,
        "driver_stable": True,
        "material_prediction_equivalent_count": 0,
        "near_alternatives_tested": 0,
    }
    if near:
        reference_prediction, reference_driver = _smooth_outputs(dataset, float(best["estimate"]), observation_spec, model_config)
        scales = np.asarray(observation_spec.observation_scale) * math.sqrt(float(observation_spec.student_nu) / (float(observation_spec.student_nu) - 2.0))
        unique_near = {
            round(float(row["estimate"]), 10): row
            for row in sorted(near, key=lambda row: abs(float(row["estimate"]) - float(best["estimate"])), reverse=True)
        }
        for tested, alternative in enumerate(unique_near.values(), 1):
            alternative_prediction, alternative_driver = _smooth_outputs(dataset, float(alternative["estimate"]), observation_spec, model_config)
            whitened_rmse = float(np.sqrt(np.mean(np.square((alternative_prediction - reference_prediction) / scales[None, :]))))
            driver_nrmse = float(np.sqrt(np.mean(np.square(alternative_driver - reference_driver))) / max(np.std(reference_driver), 1.0e-12))
            driver_correlation = float(np.corrcoef(reference_driver, alternative_driver)[0, 1]) if np.std(reference_driver) > 0 and np.std(alternative_driver) > 0 else float("nan")
            prediction_equivalent = whitened_rmse <= float(config["inference"]["prediction_equivalence_whitened_rmse"])
            driver_stable = bool(driver_nrmse <= float(config["inference"]["driver_stability_max_nrmse"]) and driver_correlation >= float(config["inference"]["driver_stability_min_correlation"]))
            confounding = {
                "alternative_estimate": float(alternative["estimate"]),
                "parameter_distance_fraction": abs(float(alternative["estimate"]) - float(best["estimate"])) / (registered[1] - registered[0]),
                "observation_whitened_rmse": whitened_rmse,
                "driver_nrmse": driver_nrmse,
                "driver_correlation": driver_correlation,
                "prediction_equivalent": prediction_equivalent,
                "driver_stable": driver_stable,
                "material_prediction_equivalent_count": int(prediction_equivalent),
                "near_alternatives_tested": tested,
            }
            if prediction_equivalent:
                break
    return {
        "candidate": dataset.candidate,
        "replicate_id": dataset.replicate_id,
        "n_observations": n_observations,
        "best_estimate": float(best["estimate"]),
        "best_nll": best_nll,
        "registered_multistarts": registered_rows,
        "expanded_multistarts": expanded_rows,
        "profile_grid": profile_grid,
        "profile_nll": profile_nll,
        "profile_delta": profile_delta,
        "profile_summary": profile_summary,
        "posterior_grid": posterior,
        "laplace": laplace,
        "confounding": confounding,
    }


def _score(observations: Sequence[np.ndarray], candidate: str, value: float, config: Mapping[str, Any]) -> tuple[float, int]:
    raw = _coordinate_raw(candidate, value, config)
    parameters, observation_spec, model_config = _parameters(raw, config)
    first_target = int(round(float(config["simulation"]["heldout_fnirs_mask_from_s"]) * float(config["simulation"]["sampling_hz"])))
    scores: list[np.ndarray] = []
    for item in observations:
        mask = np.isfinite(item)
        mask[first_target:, 1:] = False
        smoothed = smooth_balloon(item, parameters=parameters, observation_spec=observation_spec, config=model_config, observation_mask=mask)
        target = np.zeros_like(mask)
        target[first_target:, 1:] = np.isfinite(item[first_target:, 1:])
        variance = np.asarray(smoothed.total_variance, dtype=np.float64)
        valid = target & np.isfinite(smoothed.trajectory_mean) & np.isfinite(variance) & (variance > 0.0)
        residual = np.asarray(item)[valid] - np.asarray(smoothed.trajectory_mean)[valid]
        nu = float(observation_spec.student_nu)
        predictive_scale = np.sqrt(variance[valid] * (nu - 2.0) / nu)
        scores.append(-student_t.logpdf(residual, df=nu, loc=0.0, scale=predictive_scale))
    finite = np.concatenate(scores) if scores else np.empty(0)
    if not len(finite) or not np.all(np.isfinite(finite)):
        return float("nan"), 0
    return float(np.mean(finite)), int(len(finite))


def _sensitivity(candidate: str, raw: Mapping[str, float], drivers: Sequence[np.ndarray], config: Mapping[str, Any]) -> dict[str, Any]:
    relative = relative_from_raw(raw, config)
    step = float(config["inference"]["sensitivity_step"])
    _, observation_spec, model_config = _parameters(raw, config)
    scales = np.asarray(observation_spec.observation_scale) * math.sqrt(float(observation_spec.student_nu) / (float(observation_spec.student_nu) - 2.0))
    columns = []
    for coordinate in ("log_gain_relative", "log_time_relative"):
        plus_relative = dict(relative)
        minus_relative = dict(relative)
        plus_relative[coordinate] += step
        minus_relative[coordinate] -= step
        plus_parameters, _, _ = _parameters(raw_from_relative(plus_relative["log_gain_relative"], plus_relative["log_time_relative"], config), config)
        minus_parameters, _, _ = _parameters(raw_from_relative(minus_relative["log_gain_relative"], minus_relative["log_time_relative"], config), config)
        pieces = []
        for driver in drivers:
            plus = simulate_balloon(driver, plus_parameters, observation_spec=observation_spec, config=model_config, add_noise=False).clean_observations
            minus = simulate_balloon(driver, minus_parameters, observation_spec=observation_spec, config=model_config, add_noise=False).clean_observations
            pieces.append(((plus - minus) / (2.0 * step) / scales[None, :]).reshape(-1))
        columns.append(np.concatenate(pieces))
    jacobian = np.column_stack(columns)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    leading = float(singular[0]) if len(singular) else 0.0
    relative_singular = singular / leading if leading > 0.0 else np.zeros_like(singular)
    threshold = float(config["inference"]["sensitivity_relative_singular_threshold"])
    return {
        "active_singular_value": float(np.linalg.norm(columns[CANDIDATES.index(candidate)])),
        "active_effective_rank": int(np.isfinite(np.linalg.norm(columns[CANDIDATES.index(candidate)])) and np.linalg.norm(columns[CANDIDATES.index(candidate)]) > 0.0),
        "singular_value_1": float(singular[0]),
        "singular_value_2": float(singular[1]),
        "relative_singular_value_2": float(relative_singular[1]),
        "effective_rank": int(np.count_nonzero(relative_singular >= threshold)),
        "relative_threshold": threshold,
        "rows": int(jacobian.shape[0]),
        "columns": int(jacobian.shape[1]),
        "evaluated_at_candidate_truth": candidate,
    }


def _truncated_draws(spec: Mapping[str, Any], count: int, seed: int) -> np.ndarray:
    mean = float(spec["truth_prior_mean"])
    sd = float(spec["truth_prior_sd"])
    lower, upper = map(float, spec["truth_support"])
    return np.asarray(truncnorm.rvs((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd, size=count, random_state=np.random.default_rng(seed)))


def _clopper_pearson(successes: int, count: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta_distribution.ppf(0.025, successes, count - successes + 1))
    upper = 1.0 if successes == count else float(beta_distribution.ppf(0.975, successes + 1, count - successes))
    return lower, upper


def _bootstrap_mean_ci(values: Sequence[float], repetitions: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(int(repetitions), len(array)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _calibration(rows: Sequence[Mapping[str, Any]], method: str, config: Mapping[str, Any]) -> dict[str, Any]:
    selected = [row for row in rows if row["posterior_method"] == method]
    ranks = np.asarray([float(row["rank_u"]) for row in selected])
    covered = int(sum(bool(row["covered95"]) for row in selected))
    count = len(selected)
    coverage = covered / max(count, 1)
    cp_lower, cp_upper = _clopper_pearson(covered, count)
    ks = kstest(ranks, "uniform")
    critical = float(config["gates"]["sbc_ks_alpha_critical_multiplier"]) / math.sqrt(max(count, 1))
    minimum_coverage = float(config["gates"]["minimum_sbc_coverage_95"])
    maximum_coverage = float(config["gates"]["maximum_sbc_coverage_95"])
    nominal = float(config["gates"]["required_nominal_coverage_in_binomial_interval"])
    resolution_pass = bool(all(bool(row.get("resolution_pass", True)) for row in selected))
    passed = bool(
        count >= int(config["gates"]["minimum_independent_sbc_replicates"])
        and np.all(np.isfinite(ranks))
        and float(ks.statistic) <= critical
        and float(config["gates"]["minimum_sbc_rank_mean"]) <= float(np.mean(ranks)) <= float(config["gates"]["maximum_sbc_rank_mean"])
        and minimum_coverage <= coverage <= maximum_coverage
        and cp_lower <= nominal <= cp_upper
        and resolution_pass
    )
    return {
        "posterior_method": method,
        "independent_replicates": count,
        "ks_D": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "ks_critical": critical,
        "rank_mean": float(np.mean(ranks)),
        "coverage95": coverage,
        "coverage_count": covered,
        "coverage_cp_lower": cp_lower,
        "coverage_cp_upper": cp_upper,
        "resolution_pass": resolution_pass,
        "passed": passed,
    }


def _gate(candidate: str, gate_id: str, observed: Any, comparator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {"candidate": candidate, "gate_id": gate_id, "observed": observed, "comparator": comparator, "threshold": threshold, "passed": bool(passed)}


def c2_gate_state(candidate_decisions: Mapping[str, str], *, smoke: bool = False) -> str:
    """Keep the two-dimensional candidate fail-closed behind both C1 gates."""

    if smoke:
        return "NOT_RUN_SMOKE"
    if set(candidate_decisions) != set(CANDIDATES):
        return "NOT_RUN_C1_GATE_NOT_MET"
    return (
        "ELIGIBLE_FOR_SEPARATE_REGISTERED_RUN"
        if all(candidate_decisions[name] == "PASS" for name in CANDIDATES)
        else "NOT_RUN_C1_GATE_NOT_MET"
    )


def _summarize_candidate(
    candidate: str,
    recovery_rows: Sequence[Mapping[str, Any]],
    posterior_rows: Sequence[Mapping[str, Any]],
    fit_results: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    svd_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    gates = config["gates"]
    candidate_results = [row for row in fit_results if row["candidate"] == candidate]
    candidate_recovery = [row for row in recovery_rows if row["candidate"] == candidate]
    candidate_heldout = [row for row in heldout_rows if row["candidate"] == candidate]
    errors = np.asarray([float(row["error"]) for row in candidate_recovery])
    spec = config["composite"]["candidates"][candidate]
    registered = tuple(map(float, spec["registered_bounds"]))
    span = registered[1] - registered[0]
    bias_ci = _bootstrap_mean_ci(errors, int(config["inference"]["bootstrap_repetitions"]), int(config["experiment"]["seed"]) + 300_000 + CANDIDATES.index(candidate))
    laplace_calibration = _calibration([row for row in posterior_rows if row["candidate"] == candidate], "EKF_Laplace_truncated_normal", config)
    grid_calibration = _calibration([row for row in posterior_rows if row["candidate"] == candidate], "exact_1d_grid_quadrature_under_EKF_likelihood", config)
    calibration = [dict(candidate=candidate, **laplace_calibration), dict(candidate=candidate, **grid_calibration)]
    selected_method = "EKF_Laplace_truncated_normal" if laplace_calibration["passed"] else "exact_1d_grid_quadrature_under_EKF_likelihood"
    selected_calibration = laplace_calibration if laplace_calibration["passed"] else grid_calibration
    all_multistarts = [row for result in candidate_results for row in (*result["registered_multistarts"], *result["expanded_multistarts"])]
    success_fraction = float(np.mean([bool(row["success"]) for row in all_multistarts]))
    multistart_nll_spreads, multistart_parameter_spreads = [], []
    for result in candidate_results:
        rows = result["registered_multistarts"]
        multistart_nll_spreads.append((max(float(row["likelihood_nll"]) for row in rows) - min(float(row["likelihood_nll"]) for row in rows)) / int(result["n_observations"]))
        multistart_parameter_spreads.append((max(float(row["estimate"]) for row in rows) - min(float(row["estimate"]) for row in rows)) / span)
    profile_boundary_fraction = float(np.mean([bool(row["profile_summary"]["touches_boundary"]) for row in candidate_results]))
    profile_truth_coverage = float(np.mean([bool(row["profile_truth_covered"]) for row in candidate_recovery]))
    profile_complete = all(bool(row["profile_summary"]["all_finite"] and row["profile_summary"]["support_contiguous"]) for row in candidate_results)
    profile_reference_max = max(abs(float(row["profile_summary"]["reference_difference_nll"])) for row in candidate_results)
    confounding_count = sum(int(row["confounding"]["material_prediction_equivalent_count"]) for row in candidate_results)
    expanded_outside_count = sum(
        int(
            not registered[0] <= float(expanded_best["estimate"]) <= registered[1]
            and float(expanded_best["likelihood_nll"]) <= float(result["best_nll"]) + float(config["inference"]["profile_delta_nll"])
        )
        for result in candidate_results
        for expanded_best in [min(result["expanded_multistarts"], key=lambda row: float(row["likelihood_nll"]))]
    )
    candidate_svd = [row for row in svd_rows if row["candidate"] == candidate]
    svd_pass = all(
        int(row["active_effective_rank"]) == 1
        and np.isfinite(float(row["active_singular_value"]))
        and float(row["active_singular_value"]) > 0.0
        for row in candidate_svd
    )
    joint_svd_pass = all(
        int(row["effective_rank"]) == 2
        and float(row["relative_singular_value_2"]) >= float(config["inference"]["sensitivity_relative_singular_threshold"])
        for row in candidate_svd
    )
    solver_failure_fraction = float(np.mean([
        not (
            all(bool(row["success"]) for row in (*result["registered_multistarts"], *result["expanded_multistarts"]))
            and bool(result["profile_summary"]["all_finite"])
        )
        for result in candidate_results
    ]))
    boundary_fraction = float(np.mean([
        min(
            float(result["best_estimate"]) - registered[0],
            registered[1] - float(result["best_estimate"]),
        ) <= float(config["inference"]["boundary_fraction_of_span"]) * span
        for result in candidate_results
    ]))
    delta_m0 = [float(row["candidate_minus_M0_nll_per_observation"]) for row in candidate_heldout]
    excess_oracle = [float(row["candidate_excess_oracle_nll_per_observation"]) for row in candidate_heldout]
    delta_ci = _bootstrap_mean_ci(delta_m0, int(config["inference"]["bootstrap_repetitions"]), int(config["experiment"]["seed"]) + 400_000 + CANDIDATES.index(candidate))
    excess_ci = _bootstrap_mean_ci(excess_oracle, int(config["inference"]["bootstrap_repetitions"]), int(config["experiment"]["seed"]) + 500_000 + CANDIDATES.index(candidate))
    gate_rows = [
        _gate(candidate, "independent_sbc_replicates", len(candidate_results), ">=", gates["minimum_independent_sbc_replicates"], len(candidate_results) >= int(gates["minimum_independent_sbc_replicates"])),
        _gate(candidate, "solver_failure_fraction", solver_failure_fraction, "<=", gates["maximum_solver_failure_fraction"], solver_failure_fraction <= float(gates["maximum_solver_failure_fraction"])),
        _gate(candidate, "estimate_boundary_contact_fraction", boundary_fraction, "<=", gates["maximum_estimate_boundary_contact_fraction"], boundary_fraction <= float(gates["maximum_estimate_boundary_contact_fraction"])),
        _gate(candidate, "multistart_success_fraction", success_fraction, ">=", gates["minimum_multistart_success_fraction"], success_fraction >= float(gates["minimum_multistart_success_fraction"])),
        _gate(candidate, "multistart_nll_spread_per_observation", max(multistart_nll_spreads), "<=", gates["maximum_multistart_nll_spread_per_observation"], max(multistart_nll_spreads) <= float(gates["maximum_multistart_nll_spread_per_observation"])),
        _gate(candidate, "multistart_parameter_spread_fraction", max(multistart_parameter_spreads), "<=", gates["maximum_multistart_parameter_spread_fraction"], max(multistart_parameter_spreads) <= float(gates["maximum_multistart_parameter_spread_fraction"])),
        _gate(candidate, "expanded_outside_registered_near_optimal_count", expanded_outside_count, "<=", gates["maximum_expanded_outside_registered_near_optimal_count"], expanded_outside_count <= int(gates["maximum_expanded_outside_registered_near_optimal_count"])),
        _gate(candidate, "profile_complete_contiguous", profile_complete, "==", True, profile_complete),
        _gate(candidate, "profile_reference_difference_nll", profile_reference_max, "<=", gates["maximum_profile_reference_difference_nll"], profile_reference_max <= float(gates["maximum_profile_reference_difference_nll"])),
        _gate(candidate, "profile_boundary_contact_fraction", profile_boundary_fraction, "<=", gates["maximum_profile_boundary_contact_fraction"], profile_boundary_fraction <= float(gates["maximum_profile_boundary_contact_fraction"])),
        _gate(candidate, "profile_truth_coverage", profile_truth_coverage, "within", [gates["minimum_profile_truth_coverage"], gates["maximum_profile_truth_coverage"]], float(gates["minimum_profile_truth_coverage"]) <= profile_truth_coverage <= float(gates["maximum_profile_truth_coverage"])),
        _gate(candidate, "sbc_selected_posterior_calibration", selected_calibration["passed"], "==", True, bool(selected_calibration["passed"])),
        _gate(candidate, "absolute_bias_fraction", abs(float(np.mean(errors))) / span, "<=", gates["maximum_absolute_bias_fraction_of_span"], abs(float(np.mean(errors))) / span <= float(gates["maximum_absolute_bias_fraction_of_span"])),
        _gate(candidate, "rmse_fraction", float(np.sqrt(np.mean(np.square(errors)))) / span, "<=", gates["maximum_rmse_fraction_of_span"], float(np.sqrt(np.mean(np.square(errors)))) / span <= float(gates["maximum_rmse_fraction_of_span"])),
        _gate(candidate, "bias_ci_extent_fraction", max(abs(bias_ci[0]), abs(bias_ci[1])) / span, "<=", gates["maximum_bias_ci_extent_fraction_of_span"], max(abs(bias_ci[0]), abs(bias_ci[1])) / span <= float(gates["maximum_bias_ci_extent_fraction_of_span"])),
        _gate(candidate, "material_prediction_equivalent_count", confounding_count, "<=", gates["maximum_material_prediction_equivalent_count"], confounding_count <= int(gates["maximum_material_prediction_equivalent_count"])),
        _gate(candidate, "active_composite_sensitivity_rank", svd_pass, "==", True, svd_pass),
        _gate(candidate, "heldout_excess_oracle_ci_upper", excess_ci[1], "<=", gates["maximum_heldout_excess_oracle_nll_per_observation"], excess_ci[1] <= float(gates["maximum_heldout_excess_oracle_nll_per_observation"])),
        _gate(candidate, "heldout_noninferiority_ci_upper", delta_ci[1], "<=", gates["heldout_noninferiority_margin_nll_per_observation"], delta_ci[1] <= float(gates["heldout_noninferiority_margin_nll_per_observation"])),
    ]
    passed = all(bool(row["passed"]) for row in gate_rows)
    summary = {
        "decision": "PASS" if passed else "FAIL",
        "selected_sbc_posterior": selected_method,
        "laplace_calibrated": bool(laplace_calibration["passed"]),
        "grid_fallback_calibrated": bool(grid_calibration["passed"]),
        "independent_replicates": len(candidate_results),
        "estimate_bias": float(np.mean(errors)),
        "estimate_bias_95_bootstrap_ci": list(bias_ci),
        "estimate_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "profile_truth_coverage": profile_truth_coverage,
        "profile_boundary_contact_fraction": profile_boundary_fraction,
        "multistart_success_fraction": success_fraction,
        "solver_failure_fraction": solver_failure_fraction,
        "estimate_boundary_contact_fraction": boundary_fraction,
        "joint_gain_time_sensitivity_rank_diagnostic_pass": joint_svd_pass,
        "material_prediction_equivalent_count": confounding_count,
        "expanded_outside_registered_near_optimal_count": expanded_outside_count,
        "heldout_candidate_minus_M0_mean_nll_per_observation": float(np.mean(delta_m0)),
        "heldout_candidate_minus_M0_95_bootstrap_ci": list(delta_ci),
        "heldout_candidate_excess_oracle_mean_nll_per_observation": float(np.mean(excess_oracle)),
        "heldout_candidate_excess_oracle_95_bootstrap_ci": list(excess_ci),
        "failed_gates": [row["gate_id"] for row in gate_rows if not row["passed"]],
    }
    return summary, gate_rows, calibration


def _artifact_entry(path: Path, row_unit: str | None = None, expected_rows: int | None = None, formula: str | None = None) -> dict[str, Any]:
    rows = None
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            rows = max(sum(1 for _ in handle) - 1, 0)
    return {"required": True, "present": path.is_file(), "row_unit": row_unit, "row_formula": formula, "expected_rows": expected_rows, "rows_data": rows, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _expected_rows(config: Mapping[str, Any], *, smoke: bool) -> tuple[dict[str, int], dict[str, str]]:
    arms = len(CANDIDATES)
    replicates = int(config["simulation"]["independent_replicates_per_direction"])
    trials = int(config["simulation"]["train_trials_per_replicate"]) + int(config["simulation"]["heldout_trials_per_replicate"])
    starts = int(config["inference"]["multistarts"])
    profiles = int(config["inference"]["profile_points"])
    posterior_points = int(config["inference"]["posterior_grid_points"])
    counts = {
        "truth_parameters.csv": arms * replicates,
        "synthetic_inventory.csv": arms * replicates * trials,
        "multistart_results.csv": arms * replicates * 2 * starts,
        "profile_likelihood.csv": arms * replicates * profiles,
        "posterior_grid.csv": arms * replicates * posterior_points,
        "posterior_diagnostics.csv": arms * replicates * 2,
        "parameter_recovery.csv": arms * replicates,
        "state_confounding.csv": arms * replicates,
        "heldout_scores.csv": arms * replicates,
        "sensitivity_svd.csv": arms * replicates,
        "calibration.csv": 0 if smoke else arms * 2,
        "gates.csv": 0 if smoke else arms * 19,
    }
    formulas = {
        "truth_parameters.csv": "arms*replicates",
        "synthetic_inventory.csv": "arms*replicates*(train_trials+heldout_trials)",
        "multistart_results.csv": "arms*replicates*(registered+expanded)*starts",
        "profile_likelihood.csv": "arms*replicates*profile_points",
        "posterior_grid.csv": "arms*replicates*posterior_grid_points",
        "posterior_diagnostics.csv": "arms*replicates*posterior_methods",
        "parameter_recovery.csv": "arms*replicates",
        "state_confounding.csv": "arms*replicates",
        "heldout_scores.csv": "arms*replicates",
        "sensitivity_svd.csv": "arms*replicates",
        "calibration.csv": "formal*arms*posterior_methods",
        "gates.csv": "formal*arms*registered_gate_count",
    }
    return counts, formulas


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Composite synthetic T-P2 result",
        "",
        f"Run mode: `{summary['run_mode']}`; decision: `{summary['decision']}`.",
        "",
    ]
    for candidate in CANDIDATES:
        item = summary["candidates"][candidate]
        lines.extend([f"- `{candidate}`: `{item['decision']}`; failed gates: {', '.join(item.get('failed_gates', ())) or 'none'}."])
    lines.extend([
        "",
        f"`C2_GT`: `{summary['C2_GT_state']}`.",
        "Measured, validation, and protected metadata/arrays were not opened by this run.",
        "This is synthetic identifiability evidence only; it is not a trait, qualification, teacher, or tokenizer claim.",
        "",
    ])
    return "\n".join(lines)


def run(config: Mapping[str, Any], run_dir: Path, *, config_path: Path = DEFAULT_CONFIG_PATH, smoke: bool = False) -> dict[str, Any]:
    validate_config(config)
    effective = _effective_config(config, smoke)
    resolved = Path(run_dir)
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"run directory must be new or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    _atomic_write(resolved / "resolved_config.yaml", yaml.safe_dump(effective, sort_keys=False, allow_unicode=True))
    boundary = {
        "scope": "synthetic_known_truth_only",
        "measured_data_enabled": False,
        "measured_metadata_opened": False,
        "measured_arrays_opened": False,
        "validation_data_enabled": False,
        "protected_data_enabled": False,
        "validation_subject_array_access_count": 0,
        "protected_subject_array_access_count": 0,
        "truth_passed_to_fitter": False,
        "qualification_eligible": False,
        "decision_eligibility": False,
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "incomplete",
        "run_state": "initial",
        "completion_status": "incomplete",
        "stage": "before_simulation",
        "started_at": started_at,
        "run_mode": effective["experiment"]["run_mode"],
        "config_path": str(Path(config_path).resolve().relative_to(REPO_ROOT)),
        "config_sha256": _sha256(Path(config_path).resolve()),
        "runner_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "resolved_config_sha256": _sha256(resolved / "resolved_config.yaml"),
        "boundary": boundary,
    }
    _atomic_json(resolved / "manifest.json", manifest)
    try:
        replicate_count = int(effective["simulation"]["independent_replicates_per_direction"])
        truth_rows: list[dict[str, Any]] = []
        inventory_rows: list[dict[str, Any]] = []
        truth_registry: dict[tuple[str, int], dict[str, Any]] = {}
        fit_tasks: list[FitDataset] = []
        for candidate_index, candidate in enumerate(CANDIDATES):
            candidate_spec = effective["composite"]["candidates"][candidate]
            truths = _truncated_draws(candidate_spec, replicate_count, int(effective["experiment"]["seed"]) + 100_000 * (candidate_index + 1))
            for replicate_id, truth in enumerate(truths):
                raw = _coordinate_raw(candidate, float(truth), effective)
                recovered = relative_from_raw(raw, effective)
                if abs(recovered[COORDINATE_FOR[candidate]] - float(truth)) > 1.0e-12:
                    raise RuntimeError("composite truth round-trip failed")
                train_trials, heldout_trials, truth_drivers = [], [], []
                total_trials = int(effective["simulation"]["train_trials_per_replicate"]) + int(effective["simulation"]["heldout_trials_per_replicate"])
                for trial_index in range(total_trials):
                    trial_seed = int(effective["experiment"]["seed"]) + 10_000_000 * (candidate_index + 1) + 10_000 * replicate_id + trial_index
                    generated = _truth_trial(raw, trial_seed, effective)
                    role = "train" if trial_index < int(effective["simulation"]["train_trials_per_replicate"]) else "heldout"
                    (train_trials if role == "train" else heldout_trials).append(generated["observations"])
                    if role == "train":
                        truth_drivers.append(generated["driver"])
                    inventory_rows.append({
                        "candidate": candidate,
                        "replicate_id": replicate_id,
                        "trial_id": trial_index,
                        "role": role,
                        "seed": trial_seed,
                        "samples": len(generated["driver"]),
                        "observations_sha256": _array_sha256(generated["observations"]),
                        "driver_sha256": _array_sha256(generated["driver"]),
                        "physical_valid": bool(generated["physical_valid"]),
                    })
                truth_registry[(candidate, replicate_id)] = {
                    "coordinate": float(truth),
                    "raw": raw,
                    "drivers": tuple(truth_drivers),
                    "heldout_observations": tuple(heldout_trials),
                }
                truth_rows.append({
                    "candidate": candidate,
                    "replicate_id": replicate_id,
                    "coordinate": COORDINATE_FOR[candidate],
                    "truth": float(truth),
                    **{f"truth_{name}": value for name, value in raw.items()},
                    "truth_passed_to_fitter": False,
                })
                optimizer_seed = 71_000_000 + 100_000 * candidate_index + replicate_id
                fit_tasks.append(FitDataset(candidate, replicate_id, tuple(train_trials), optimizer_seed, _fit_contract(effective)))
        if any("truth" in field for task in fit_tasks for field in task.__dataclass_fields__):
            raise RuntimeError("truth field crossed into the fitter task")
        manifest.update({"stage": "fitting", "status": "running", "run_state": "running"})
        _atomic_json(resolved / "manifest.json", manifest)
        fit_results: list[dict[str, Any]] = []
        workers = min(int(effective["inference"]["workers"]), len(fit_tasks))
        if workers == 1:
            for index, task in enumerate(fit_tasks, 1):
                fit_results.append(_fit_one(task))
                print(json.dumps({"stage": "fitting", "completed": index, "total": len(fit_tasks)}), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_fit_one, task): task for task in fit_tasks}
                for index, future in enumerate(as_completed(futures), 1):
                    fit_results.append(future.result())
                    if index == 1 or index % 5 == 0 or index == len(fit_tasks):
                        print(json.dumps({"stage": "fitting", "completed": index, "total": len(fit_tasks)}), flush=True)
        fit_results.sort(key=lambda row: (CANDIDATES.index(row["candidate"]), int(row["replicate_id"])))
        _atomic_csv(resolved / "truth_parameters.csv", truth_rows)
        _atomic_csv(resolved / "synthetic_inventory.csv", inventory_rows)
        manifest.update({"stage": "postfit_evaluation"})
        _atomic_json(resolved / "manifest.json", manifest)
        multistart_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        posterior_grid_rows: list[dict[str, Any]] = []
        posterior_rows: list[dict[str, Any]] = []
        recovery_rows: list[dict[str, Any]] = []
        confounding_rows: list[dict[str, Any]] = []
        heldout_rows: list[dict[str, Any]] = []
        svd_rows: list[dict[str, Any]] = []
        for result in fit_results:
            candidate = result["candidate"]
            replicate_id = int(result["replicate_id"])
            truth = truth_registry[(candidate, replicate_id)]
            for row in (*result["registered_multistarts"], *result["expanded_multistarts"]):
                multistart_rows.append({"candidate": candidate, "replicate_id": replicate_id, **row})
            for grid_index, (value, nll, delta) in enumerate(zip(result["profile_grid"], result["profile_nll"], result["profile_delta"])):
                profile_rows.append({"candidate": candidate, "replicate_id": replicate_id, "grid_index": grid_index, "fixed_value": float(value), "likelihood_nll": float(nll), "delta_nll": float(delta), "in_95_support": bool(delta <= float(effective["inference"]["profile_delta_nll"]))})
            posterior = result["posterior_grid"]
            for grid_index, (value, nll, density, cdf) in enumerate(zip(posterior["grid"], posterior["nll"], posterior["density"], posterior["cdf"])):
                posterior_grid_rows.append({"candidate": candidate, "replicate_id": replicate_id, "grid_index": grid_index, "coordinate_value": float(value), "likelihood_nll": float(nll), "posterior_density": float(density), "posterior_cdf": float(cdf)})
            truth_value = float(truth["coordinate"])
            laplace = result["laplace"]
            if bool(laplace["success"]):
                support = tuple(map(float, effective["composite"]["candidates"][candidate]["truth_support"]))
                a, b = (support[0] - float(laplace["mode"])) / float(laplace["sd"]), (support[1] - float(laplace["mode"])) / float(laplace["sd"])
                laplace_rank = float(truncnorm.cdf(truth_value, a, b, loc=float(laplace["mode"]), scale=float(laplace["sd"])))
            else:
                laplace_rank = float("nan")
            posterior_spacing = float(posterior["grid"][1] - posterior["grid"][0])
            methods = (
                ("EKF_Laplace_truncated_normal", float(laplace["mode"]), float(laplace["sd"]), float(laplace["lower95"]), float(laplace["upper95"]), laplace_rank, bool(laplace["success"])),
                ("exact_1d_grid_quadrature_under_EKF_likelihood", float(posterior["mean"]), float(posterior["sd"]), float(posterior["lower95"]), float(posterior["upper95"]), float(np.interp(truth_value, posterior["grid"], posterior["cdf"])), float(posterior["sd"]) / posterior_spacing >= float(effective["gates"]["minimum_posterior_sd_grid_steps"]) and float(posterior["coarse_grid_max_cdf_difference"]) <= float(effective["gates"]["maximum_posterior_grid_cdf_difference"])),
            )
            for method, estimate, sd, lower, upper, rank, resolution_pass in methods:
                posterior_rows.append({
                    "candidate": candidate,
                    "replicate_id": replicate_id,
                    "posterior_method": method,
                    "truth": truth_value,
                    "posterior_estimate": estimate,
                    "posterior_sd": sd,
                    "lower95": lower,
                    "upper95": upper,
                    "rank_u": rank,
                    "covered95": bool(lower <= truth_value <= upper),
                    "resolution_pass": resolution_pass,
                })
            profile_truth_delta = float(np.interp(truth_value, result["profile_grid"], result["profile_nll"]) - result["best_nll"])
            recovery_rows.append({
                "candidate": candidate,
                "replicate_id": replicate_id,
                "coordinate": COORDINATE_FOR[candidate],
                "truth": truth_value,
                "estimate": float(result["best_estimate"]),
                "error": float(result["best_estimate"]) - truth_value,
                "profile_truth_delta_nll": profile_truth_delta,
                "profile_truth_covered": bool(profile_truth_delta <= float(effective["inference"]["profile_delta_nll"])),
            })
            result["profile_truth_covered"] = recovery_rows[-1]["profile_truth_covered"]
            confounding_rows.append({"candidate": candidate, "replicate_id": replicate_id, **result["confounding"]})
            heldout = truth["heldout_observations"]
            candidate_score, n_heldout = _score(heldout, candidate, float(result["best_estimate"]), effective)
            m0_score, _ = _score(heldout, candidate, 0.0, effective)
            oracle_score, _ = _score(heldout, candidate, truth_value, effective)
            if n_heldout < 1 or not np.all(np.isfinite([candidate_score, m0_score, oracle_score])):
                raise RuntimeError("held-out masked predictive score is unavailable")
            heldout_rows.append({
                "candidate": candidate,
                "replicate_id": replicate_id,
                "finite_observations": n_heldout,
                "candidate_nll_per_observation": candidate_score,
                "M0_nll_per_observation": m0_score,
                "oracle_nll_per_observation": oracle_score,
                "candidate_minus_M0_nll_per_observation": candidate_score - m0_score,
                "candidate_excess_oracle_nll_per_observation": candidate_score - oracle_score,
            })
            sensitivity = _sensitivity(candidate, truth["raw"], truth["drivers"], effective)
            svd_rows.append({"candidate": candidate, "replicate_id": replicate_id, "definition": "joint_gain_time_conditional_forward_fixed_truth_driver_whitened_by_declared_student_t_marginal_sd", **sensitivity})
        gate_rows: list[dict[str, Any]] = []
        calibration_rows: list[dict[str, Any]] = []
        candidate_summaries: dict[str, Any] = {}
        if smoke:
            for candidate in CANDIDATES:
                candidate_summaries[candidate] = {"decision": "NOT_EVALUATED_SMOKE", "failed_gates": []}
            decision = "SMOKE_COMPLETE_NOT_EVIDENCE"
            c2_state = c2_gate_state({candidate: candidate_summaries[candidate]["decision"] for candidate in CANDIDATES}, smoke=True)
        else:
            for candidate in CANDIDATES:
                item, candidate_gates, candidate_calibration = _summarize_candidate(candidate, recovery_rows, posterior_rows, fit_results, heldout_rows, svd_rows, effective)
                candidate_summaries[candidate] = item
                gate_rows.extend(candidate_gates)
                calibration_rows.extend(candidate_calibration)
            both_pass = all(candidate_summaries[candidate]["decision"] == "PASS" for candidate in CANDIDATES)
            decision = "PASS_C1_BOTH" if both_pass else "BLOCKED_C1_COMPOSITE_IDENTIFIABILITY"
            c2_state = c2_gate_state({candidate: candidate_summaries[candidate]["decision"] for candidate in CANDIDATES})
        tables = {
            "truth_parameters.csv": (truth_rows, "candidate_replicate_truth"),
            "synthetic_inventory.csv": (inventory_rows, "candidate_replicate_trial"),
            "multistart_results.csv": (multistart_rows, "candidate_replicate_variant_start"),
            "profile_likelihood.csv": (profile_rows, "candidate_replicate_profile_grid_point"),
            "posterior_grid.csv": (posterior_grid_rows, "candidate_replicate_posterior_grid_point"),
            "posterior_diagnostics.csv": (posterior_rows, "candidate_replicate_posterior_method"),
            "parameter_recovery.csv": (recovery_rows, "candidate_replicate"),
            "state_confounding.csv": (confounding_rows, "candidate_replicate"),
            "heldout_scores.csv": (heldout_rows, "candidate_replicate"),
            "sensitivity_svd.csv": (svd_rows, "candidate_replicate_joint_gain_time_singular_values"),
            "calibration.csv": (calibration_rows, "candidate_posterior_method"),
            "gates.csv": (gate_rows, "candidate_gate"),
        }
        expected_rows, row_formulas = _expected_rows(effective, smoke=smoke)
        actual_rows = {name: len(rows) for name, (rows, _) in tables.items()}
        if actual_rows != expected_rows:
            raise RuntimeError(f"artifact row-count contract mismatch: expected={expected_rows}, actual={actual_rows}")
        for name, (rows, _) in tables.items():
            _atomic_csv(resolved / name, rows)
        completed_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "schema": SCHEMA,
            "analysis_kind": "composite_synthetic_T_P2_C1_screen",
            "run_mode": effective["experiment"]["run_mode"],
            "status": "complete",
            "completion_status": "complete",
            "decision": decision,
            "candidates": candidate_summaries,
            "C2_GT_state": c2_state,
            "C2_GT_policy": effective["composite"]["c2_policy"],
            "measured_hierarchical_arm_state": "blocked_prerequisite_not_started",
            "independent_replicate_unit": "one independently drawn composite truth with independently reset training and held-out trials",
            "independent_replicates_per_direction": replicate_count,
            "truth_to_fitter_boundary": "FitDataset contains candidate, replicate_id, noisy training observations, an optimizer-only seed, and a fit contract stripped of realized truth/driver/generation seeds",
            "heldout_prediction_contract": "freeze fitted composite; observe EEG throughout and fNIRS before 12 s; mask HbO/HbR from 12 s onward and score only those masked targets",
            "source_hashes": {str(source["path"]): str(source["sha256"]) for source in effective["sources"].values()},
            "artifact_row_counts": actual_rows,
            "expected_artifact_row_counts": expected_rows,
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_seconds": time.perf_counter() - start_clock,
            "boundary": boundary,
            "claim_boundary": "synthetic composite identifiability only; no measured trait, qualification, teacher, or tokenizer claim",
        }
        _atomic_json(resolved / "summary.json", summary)
        _atomic_write(resolved / "summary.md", _markdown(summary))
        artifacts = {"resolved_config.yaml": _artifact_entry(resolved / "resolved_config.yaml")}
        artifacts.update({name: _artifact_entry(resolved / name, unit, expected_rows[name], row_formulas[name]) for name, (_, unit) in tables.items()})
        artifacts.update({"summary.json": _artifact_entry(resolved / "summary.json"), "summary.md": _artifact_entry(resolved / "summary.md")})
        manifest = {
            **manifest,
            "status": "smoke_complete" if smoke else "complete",
            "run_state": "complete",
            "completion_status": "complete",
            "stage": "complete",
            "updated_at": completed_at,
            "completed_at": completed_at,
            "elapsed_seconds": summary["elapsed_seconds"],
            "decision": decision,
            "C2_GT_state": c2_state,
            "source_hashes": summary["source_hashes"],
            "artifacts": artifacts,
            "artifact_row_counts": summary["artifact_row_counts"],
            "row_count_contract": {
                name: {"row_unit": tables[name][1], "formula": row_formulas[name], "expected_rows": expected_rows[name], "actual_rows": actual_rows[name]}
                for name in tables
            },
            "summary_pointer": "summary.json",
            "git": _git_payload(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        }
        _atomic_json(resolved / "manifest.json", manifest)
        print(json.dumps({"stage": "complete", "decision": decision, "run_dir": str(resolved)}), flush=True)
        return summary
    except Exception as exc:
        failed_at = datetime.now(timezone.utc).isoformat()
        failed_stage = str(manifest.get("stage", "unknown"))
        partial_artifacts = sorted(path.name for path in resolved.iterdir() if path.is_file() and path.name != "manifest.json")
        _atomic_json(resolved / "manifest.json", {
            **manifest,
            "status": "incomplete_failed",
            "run_state": "failure",
            "completion_status": "incomplete",
            "stage": "failed",
            "failed_stage": failed_stage,
            "partial_artifacts": partial_artifacts,
            "failed_at": failed_at,
            "elapsed_seconds": time.perf_counter() - start_clock,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
        })
        raise


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_composite_t2_v1")
        run_dir = REPO_ROOT / str(config["output"]["root"]) / stamp
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    run(config, run_dir, config_path=config_path, smoke=bool(args.smoke))
    return run_dir


if __name__ == "__main__":
    main()
