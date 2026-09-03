#!/usr/bin/env python3
"""Fit-only T3 identifiability diagnostic for one synthetic and three measured cases.

The measured branch calibrates observation gauges on subjects 01--18, selects
low/median/high fixed-model residual representatives, and analyzes only their
eight fit trials.  Validation subjects 19--23 and protected subjects 24--29
are removed from loader selection before any of their arrays are loaded; the
canonical dataset index metadata is still constructed by the shared loader.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Every optimizer already uses process-level parallelism.  Do not multiply it
# by an implicit BLAS thread pool in each worker.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import scipy
import yaml
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_t3_measured_reconstruction_null import (
    _fit_models,
    _from_optimizer_coordinate,
    _identity_fields,
    _parameter_values,
    _prepare_measured_series,
    _replace_parameter_values,
    _split_stage_trials,
    _to_optimizer_coordinate,
    load_config as load_measured_config,
)
from experiments.evaluate_t3a_balloon_robust_p0 import (
    _atomic_csv,
    _atomic_json,
    _atomic_write,
    generate_case,
    load_config as load_synthetic_config,
)
from src.inference.t3a_balloon_robust_ssm import (
    BalloonConfig,
    BalloonFixedParameters,
    BalloonFreeParameters,
    BalloonObservationSpec,
    BalloonParameters,
    simulate_balloon,
    smooth_balloon,
)


SCHEMA = "t3_identifiability_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/t3_identifiability_v1.yaml"
MEASURED_CONFIG_PATH = "experiments/configs/physiology_semantic_tokenizer/t3_measured_reconstruction_null_v1.yaml"
SYNTHETIC_CONFIG_PATH = "experiments/configs/physiology_semantic_tokenizer/t3a_balloon_robust_p0.yaml"
MEASURED_CONFIG_SHA256 = "09317a7fd6eb50b44c829d1ad3f2e5a4319a2fe29e16544448d44095801a939e"
SYNTHETIC_CONFIG_SHA256 = "f8343378c00cb8e0237aba6db82a4bebdf383816ca1f70daccb480c23ce16e31"
OUTPUT_ROOT = "experiments/runs/physiology_semantic_tokenizer/t3_identifiability"
PARENT_MANIFEST_PATH = "experiments/runs/physiology_semantic_tokenizer/t3_measured_reconstruction_null/20260828_subject_parameter_fit_v2/manifest.json"
PARENT_RUN_ID = "t3_measured_reconstruction_null/20260828_subject_parameter_fit_v2"
PARAMETER_NAMES = ("beta", "kappa", "tau", "gamma", "alpha", "E0")
ACTIVE_PARAMETERS = ("beta", "kappa", "tau")


@dataclass(frozen=True)
class FitProblem:
    case_id: str
    source_kind: str
    representative_role: str
    trials: tuple[tuple[int, np.ndarray], ...]
    base_parameters: BalloonParameters
    observation_spec: BalloonObservationSpec
    balloon_config: BalloonConfig
    active: tuple[str, ...]
    parameter_specs: dict[str, dict[str, Any]]
    truth_parameters: dict[str, float] | None = None
    truth_drivers: tuple[np.ndarray, ...] | None = None


def _subject_range(first: int, last: int) -> list[str]:
    return [f"subject_{index:02d}" for index in range(first, last + 1)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_payload() -> dict[str, str]:
    def call(*args: str) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, check=False, capture_output=True, text=True).stdout.strip()

    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "status_short": call("git", "status", "--short"),
    }


def _cache_provenance(measured_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(measured_config["data"]["cache_root"]))
    if not root.is_absolute():
        root = REPO_ROOT / root
    relative_paths = (
        "cache_manifest.json",
        "event_index/event_manifest.json",
        "event_index/events.jsonl",
        "channel_geometry/geometry_manifest.json",
        "eeg_artifact_clean_v3/cache_manifest.json",
        "eeg_artifact_clean_v4/cache_manifest.json",
        "simultaneous_eeg_eog_clean_v1/cache_manifest.json",
    )
    rows = []
    for relative in relative_paths:
        path = root / relative
        if path.exists():
            rows.append({
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            })
    if not rows or rows[0]["path"] != str((root / "cache_manifest.json").relative_to(REPO_ROOT)):
        raise RuntimeError("measured cache manifest is unavailable")
    return rows


def _validate_measured_source(config: Mapping[str, Any]) -> None:
    if config["data"].get("cache_root") != "data/cache/physiology_semantic_clean_v1":
        raise ValueError("identifiability suite requires the canonical physiology cache")
    conditions = config["data"].get("conditions", [])
    if len(conditions) != 1:
        raise ValueError("identifiability suite requires exactly one measured condition")
    condition = conditions[0]
    expected = {
        "condition_id": "single_trial_ma_session_01",
        "dataset_id": "eeg_fnirs_single_trial",
        "record_id": "session_01",
        "target_label": "MA",
        "eeg_signal_branch": "raw_with_ocular_artifact",
    }
    if any(condition.get(key) != value for key, value in expected.items()):
        raise ValueError("measured source identity has drifted")
    if (
        list(map(str, condition.get("fit_subjects", ()))) != _subject_range(1, 18)
        or list(map(str, condition.get("validation_subjects", ()))) != _subject_range(19, 23)
        or list(map(str, condition.get("protected_subjects", ()))) != _subject_range(24, 29)
    ):
        raise ValueError("measured source subject registries have drifted")


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject split, objective, and output drift before any data loader call."""

    if config.get("schema") != SCHEMA:
        raise ValueError("identifiability config schema mismatch")
    experiment = config.get("experiment", {})
    expected_experiment = {
        "name": "t3_identifiability_v1",
        "scope": "fit_only_measured_development_diagnostic",
        "measured_data_enabled": True,
        "protected_data_enabled": False,
        "qualification_eligible": False,
        "decision_eligibility": False,
    }
    if not isinstance(experiment, Mapping) or any(experiment.get(key) != value for key, value in expected_experiment.items()):
        raise ValueError("identifiability experiment boundary mismatch")
    if isinstance(experiment.get("seed"), bool) or not isinstance(experiment.get("seed"), (int, np.integer)) or int(experiment["seed"]) != 20260902:
        raise ValueError("experiment seed must remain the registered integer 20260902")
    sources = config.get("sources", {})
    if sources.get("measured_config") != MEASURED_CONFIG_PATH or sources.get("synthetic_config") != SYNTHETIC_CONFIG_PATH:
        raise ValueError("identifiability sources must be the registered T3 configs")
    source_contracts = (
        ("measured_config_sha256", MEASURED_CONFIG_PATH, MEASURED_CONFIG_SHA256),
        ("synthetic_config_sha256", SYNTHETIC_CONFIG_PATH, SYNTHETIC_CONFIG_SHA256),
    )
    for key, relative_path, expected_sha256 in source_contracts:
        if sources.get(key) != expected_sha256 or _sha256(REPO_ROOT / relative_path) != expected_sha256:
            raise ValueError(f"registered source hash mismatch: {relative_path}")
    if (
        sources.get("parent_run_role") != "context_only_not_consumed"
        or sources.get("parent_run_id") != PARENT_RUN_ID
        or sources.get("parent_manifest") != PARENT_MANIFEST_PATH
    ):
        raise ValueError("context-only parent run identity is required")
    estimand = config.get("estimand", {})
    if (
        estimand.get("estimand_id") != "fit_only_practical_identifiability_beta_kappa_tau_v1"
        or estimand.get("hypothesis") != "beta_kappa_tau_are_practically_identifiable_on_fit_only_T3a_observations"
        or estimand.get("primary_endpoint") != "every_beta_kappa_tau_profile_grid_is_complete_and_support_is_finite_contiguous_and_inside_registered_bounds"
        or estimand.get("operator") != "likelihood_only_profile_with_companion_parameters_and_latent_states_reoptimized"
        or estimand.get("null") != "no_cross_modal_null_operator_diagnostic_only"
    ):
        raise ValueError("identifiability estimand contract mismatch")
    selection = config.get("selection", {})
    if selection.get("metric") != "M0_fixed_pooled_predictive_nll_per_finite_observation":
        raise ValueError("representatives must use the frozen fit-only M0 score")
    if tuple(selection.get("heldout_trial_positions", ())) != (4, 9):
        raise ValueError("selection must exclude the two registered internal holdouts")
    if tuple(selection.get("roles", ())) != ("low", "median", "high"):
        raise ValueError("selection roles must be low/median/high")
    synthetic = config.get("synthetic", {})
    if (
        isinstance(synthetic.get("replicate_id"), bool)
        or not isinstance(synthetic.get("replicate_id"), (int, np.integer))
        or int(synthetic.get("replicate_id", -1)) != 0
        or isinstance(synthetic.get("seed"), bool)
        or not isinstance(synthetic.get("seed"), (int, np.integer))
        or int(synthetic["seed"]) != 20260902
        or synthetic.get("truth_parameters") != "prior_center"
        or synthetic.get("observations") != "noisy_clean_scenario"
    ):
        raise ValueError("synthetic case must be the registered noisy prior-centre case")
    analysis = config.get("analysis", {})
    if analysis.get("stage") != "M2_beta_kappa_tau" or tuple(analysis.get("active_parameters", ())) != ACTIVE_PARAMETERS:
        raise ValueError("the primary diagnostic must be M2 beta/kappa/tau")
    if tuple(analysis.get("sensitivity_parameters", ())) != PARAMETER_NAMES:
        raise ValueError("sensitivity SVD must cover the six registered raw parameters")
    if analysis.get("objective") != "likelihood_only":
        raise ValueError("identifiability optimization must be likelihood-only")
    integer_keys = (
        "transformed_multistarts",
        "optimizer_max_iterations",
        "workers",
        "profile_points",
        "profile_multistarts",
    )
    if any(isinstance(analysis.get(key), bool) or not isinstance(analysis.get(key), (int, np.integer)) for key in integer_keys):
        raise ValueError("optimizer counts and worker count must be integers")
    starts = int(analysis.get("transformed_multistarts", 0))
    if not 16 <= starts <= 32:
        raise ValueError("transformed multistarts must lie in [16, 32]")
    if int(analysis["optimizer_max_iterations"]) < 1 or not 1 <= int(analysis["workers"]) <= 18:
        raise ValueError("optimizer iterations/workers are outside the registered range")
    if int(analysis["profile_points"]) < 7 or int(analysis["profile_points"]) % 2 != 1:
        raise ValueError("profile grid must contain an odd number of at least seven points")
    if int(analysis["profile_multistarts"]) != 2:
        raise ValueError("profile points require exactly two deterministic starts")
    positive = (
        "optimizer_max_iterations",
        "optimizer_ftol",
        "workers",
        "profile_likelihood_ratio_delta_nll",
        "profile_reference_consistency_tolerance_nll",
        "expanded_bound_fraction_in_transformed_space",
        "boundary_fraction_in_transformed_space",
        "sensitivity_step_in_transformed_space",
        "sensitivity_relative_singular_value_threshold",
        "prediction_equivalence_max_whitened_rmse",
        "material_parameter_difference_min_fraction_of_transformed_span",
        "driver_stability_max_nrmse",
        "driver_stability_min_correlation",
    )
    if any(not np.isfinite(float(analysis.get(key, np.nan))) or float(analysis[key]) <= 0.0 for key in positive):
        raise ValueError("identifiability numeric settings must be finite and positive")
    if not 0.0 < float(analysis["boundary_fraction_in_transformed_space"]) < 0.5:
        raise ValueError("boundary fraction must lie in (0, .5)")
    if not 0.0 < float(analysis["expanded_bound_fraction_in_transformed_space"]) <= 1.0:
        raise ValueError("bound expansion must lie in (0, 1]")
    if not 0.0 < float(analysis["sensitivity_relative_singular_value_threshold"]) <= 1.0:
        raise ValueError("relative SVD threshold must lie in (0, 1]")
    if not 0.0 < float(analysis["sensitivity_step_in_transformed_space"]) < 0.5:
        raise ValueError("sensitivity step must lie in (0, .5)")
    if float(analysis["material_parameter_difference_min_fraction_of_transformed_span"]) != 0.01:
        raise ValueError("material parameter-set separation must remain one percent of transformed span")
    if float(analysis["driver_stability_min_correlation"]) > 1.0:
        raise ValueError("driver correlation threshold cannot exceed one")
    if config.get("output", {}).get("root") != OUTPUT_ROOT:
        raise ValueError(f"output.root must be {OUTPUT_ROOT}")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("identifiability configuration must be a mapping")
    config = copy.deepcopy(dict(loaded))
    validate_config(config)
    return config


def _fit_only_measured_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a loader view that physically cannot select subjects 19--29."""

    result = copy.deepcopy(dict(config))
    condition = result["data"]["conditions"][0]
    fit_subjects = list(map(str, condition["fit_subjects"]))
    if fit_subjects != _subject_range(1, 18):
        raise ValueError("measured source fit registry is not subjects 01--18")
    if list(map(str, condition["validation_subjects"])) != _subject_range(19, 23):
        raise ValueError("measured source validation registry has drifted")
    if list(map(str, condition["protected_subjects"])) != _subject_range(24, 29):
        raise ValueError("measured source protected registry has drifted")
    condition["subjects"] = fit_subjects
    condition["validation_subjects"] = []
    condition["protected_subjects"] = []
    if set(condition["subjects"]) & set(_subject_range(19, 29)):
        raise ValueError("fit-only loader view contains validation/protected subjects")
    return result


def select_representative_fit_subjects(
    scores: Mapping[str, float], expected_subjects: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Select low, closest-to-median, and high scores with lexical tie breaks."""

    finite = {str(subject): float(value) for subject, value in scores.items() if np.isfinite(float(value))}
    if expected_subjects is not None and set(map(str, expected_subjects)) != set(finite):
        raise ValueError("representative scores do not cover the exact fit-subject registry")
    if len(finite) < 3:
        raise ValueError("at least three finite fit-subject scores are required")
    median_value = float(np.median(list(finite.values())))
    selected = {
        "low": min(finite, key=lambda subject: (finite[subject], subject)),
        "median": min(finite, key=lambda subject: (abs(finite[subject] - median_value), subject)),
        "high": min(finite, key=lambda subject: (-finite[subject], subject)),
    }
    if len(set(selected.values())) != 3:
        raise ValueError("low/median/high selection did not produce three distinct subjects")
    return [
        {
            "role": role,
            "subject": selected[role],
            "score": finite[selected[role]],
            "sample_median_score": median_value,
        }
        for role in ("low", "median", "high")
    ]


def _transformed_bounds(
    names: Sequence[str], specs: Mapping[str, Mapping[str, Any]]
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            _to_optimizer_coordinate(name, float(specs[name]["bounds"][0])),
            _to_optimizer_coordinate(name, float(specs[name]["bounds"][1])),
        )
        for name in names
    )


def _validated_bounds(
    bounds: Sequence[tuple[float, float]], dimension: int | None = None
) -> tuple[tuple[float, float], ...]:
    resolved = tuple((float(lower), float(upper)) for lower, upper in bounds)
    if dimension is not None and len(resolved) != int(dimension):
        raise ValueError("optimizer bounds have the wrong dimension")
    if any(
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or not np.isfinite(upper - lower)
        or lower >= upper
        for lower, upper in resolved
    ):
        raise ValueError("optimizer bounds must be finite and ordered")
    return resolved


def expanded_transformed_bounds(
    bounds: Sequence[tuple[float, float]], fraction: float
) -> tuple[tuple[float, float], ...]:
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("bound expansion fraction must be positive")
    resolved = _validated_bounds(bounds)
    return tuple(
        (float(lower) - fraction * (float(upper) - float(lower)), float(upper) + fraction * (float(upper) - float(lower)))
        for lower, upper in resolved
    )


def transformed_multistarts(
    names: Sequence[str],
    specs: Mapping[str, Mapping[str, Any]],
    count: int,
    seed: int,
    *,
    bounds: Sequence[tuple[float, float]] | None = None,
    warm_values: Mapping[str, float] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Generate one warm/prior start plus a deterministic transformed-space LHS."""

    if count < 1:
        raise ValueError("at least one transformed start is required")
    resolved_bounds = _validated_bounds(bounds or _transformed_bounds(names, specs), len(names))
    centre = warm_values or {name: float(specs[name]["prior_mean"]) for name in names}
    first = tuple(
        float(np.clip(_to_optimizer_coordinate(name, float(centre[name])), *resolved_bounds[index]))
        for index, name in enumerate(names)
    )
    if count == 1:
        return (first,)
    rng = np.random.default_rng(int(seed))
    lhs = np.empty((count - 1, len(names)), dtype=np.float64)
    for dimension, (lower, upper) in enumerate(resolved_bounds):
        unit = (rng.permutation(count - 1) + rng.random(count - 1)) / float(count - 1)
        lhs[:, dimension] = float(lower) + unit * (float(upper) - float(lower))
    return (first, *tuple(tuple(float(value) for value in row) for row in lhs))


def minimize_profile_point(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    fixed_index: int,
    fixed_value: float,
    starts: Sequence[Sequence[float]],
    *,
    max_iterations: int,
    ftol: float,
) -> dict[str, Any]:
    """Fix one coordinate exactly and reoptimize every companion coordinate."""

    dimension = len(bounds)
    resolved_bounds = _validated_bounds(bounds, dimension)
    if not 0 <= int(fixed_index) < dimension:
        raise ValueError("profile fixed index is out of range")
    if not resolved_bounds[int(fixed_index)][0] <= float(fixed_value) <= resolved_bounds[int(fixed_index)][1]:
        raise ValueError("profile fixed value lies outside its registered bound")
    movable = tuple(index for index in range(dimension) if index != int(fixed_index))
    reduced_bounds = tuple(resolved_bounds[index] for index in movable)
    best: dict[str, Any] | None = None
    best_successful: dict[str, Any] | None = None
    for start_id, full_start in enumerate(starts):
        start = np.asarray(full_start, dtype=np.float64)
        if start.shape != (dimension,):
            raise ValueError("every profile start must span the full parameter vector")
        if not np.all(np.isfinite(start)) or any(not lower <= start[index] <= upper for index, (lower, upper) in enumerate(resolved_bounds)):
            raise ValueError("profile start lies outside finite optimizer bounds")

        def reduced_objective(reduced: np.ndarray) -> float:
            full = start.copy()
            full[int(fixed_index)] = float(fixed_value)
            full[list(movable)] = np.asarray(reduced, dtype=np.float64)
            return float(objective(full))

        result = minimize(
            reduced_objective,
            start[list(movable)],
            method="L-BFGS-B",
            bounds=reduced_bounds,
            options={"maxiter": int(max_iterations), "ftol": float(ftol), "maxls": 20},
        )
        full = start.copy()
        full[int(fixed_index)] = float(fixed_value)
        full[list(movable)] = np.asarray(result.x, dtype=np.float64)
        record = {
            "start_id": int(start_id),
            "x": tuple(float(value) for value in full),
            "objective": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
        }
        if best is None or record["objective"] < best["objective"]:
            best = record
        if record["success"] and (best_successful is None or record["objective"] < best_successful["objective"]):
            best_successful = record
    if best is None:
        raise RuntimeError("profile point has no optimizer result")
    return best_successful or best


def svd_diagnostics(
    jacobian: np.ndarray,
    parameter_names: Sequence[str],
    relative_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(parameter_names) or not np.all(np.isfinite(matrix)):
        raise ValueError("sensitivity Jacobian shape/values are invalid")
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    for index in range(len(right)):
        pivot = int(np.argmax(np.abs(right[index])))
        if right[index, pivot] < 0.0:
            right[index] *= -1.0
    leading = float(singular_values[0]) if len(singular_values) else 0.0
    relative = singular_values / leading if leading > 0.0 else np.zeros_like(singular_values)
    rank = int(np.count_nonzero(relative >= float(relative_threshold)))
    rows = []
    for index, value in enumerate(singular_values):
        row = {
            "component": int(index + 1),
            "singular_value": float(value),
            "relative_singular_value": float(relative[index]),
            "above_relative_threshold": bool(relative[index] >= float(relative_threshold)),
        }
        row.update({f"loading_{name}": float(right[index, column]) for column, name in enumerate(parameter_names)})
        rows.append(row)
    nonzero = singular_values[singular_values > max(leading * 1e-12, 1e-15)]
    condition = float(leading / nonzero[-1]) if len(nonzero) else float("inf")
    return rows, {
        "effective_rank": rank,
        "relative_threshold": float(relative_threshold),
        "condition_on_nonzero_singular_values": condition,
        "row_count": int(matrix.shape[0]),
        "column_count": int(matrix.shape[1]),
    }


def _values_from_vector(
    problem: FitProblem,
    vector: Sequence[float],
    *,
    fixed_values: Mapping[str, float] | None = None,
) -> dict[str, float]:
    values = _parameter_values(problem.base_parameters)
    if fixed_values:
        values.update({str(name): float(value) for name, value in fixed_values.items()})
    values.update({
        name: _from_optimizer_coordinate(name, float(value))
        for name, value in zip(problem.active, vector)
    })
    return values


def _prior_penalty(problem: FitProblem, values: Mapping[str, float]) -> float:
    return float(sum(
        0.5 * ((float(values[name]) - float(problem.parameter_specs[name]["prior_mean"])) / float(problem.parameter_specs[name]["prior_sd"])) ** 2
        for name in problem.active
    ))


def _likelihood_nll(problem: FitProblem, values: Mapping[str, float]) -> float:
    try:
        parameters = _replace_parameter_values(problem.base_parameters, values)
        value = -sum(
            float(smooth_balloon(
                np.asarray(observations, dtype=np.float64),
                parameters=parameters,
                observation_spec=problem.observation_spec,
                config=problem.balloon_config,
            ).predictive_log_likelihood)
            for _, observations in problem.trials
        )
        return float(value) if np.isfinite(value) else 1.0e12
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
        return 1.0e12


def _objective(problem: FitProblem) -> Callable[[np.ndarray], float]:
    cache: dict[tuple[float, ...], float] = {}

    def evaluate(vector: np.ndarray) -> float:
        key = tuple(round(float(value), 12) for value in vector)
        if key not in cache:
            cache[key] = _likelihood_nll(problem, _values_from_vector(problem, vector))
        return cache[key]

    return evaluate


def _optimizer_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    problem: FitProblem = task["problem"]
    start = np.asarray(task["start"], dtype=np.float64)
    objective = _objective(problem)
    try:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=task["bounds"],
            options={
                "maxiter": int(task["max_iterations"]),
                "ftol": float(task["ftol"]),
                "maxls": 20,
            },
        )
        vector = np.asarray(result.x, dtype=np.float64)
        likelihood = float(objective(vector))
        values = _values_from_vector(problem, vector)
        return {
            "start_id": int(task["start_id"]),
            "start": tuple(float(value) for value in start),
            "estimate": tuple(float(value) for value in vector),
            "values": values,
            "likelihood_nll": likelihood,
            "prior_penalty": _prior_penalty(problem, values),
            "posterior_objective": likelihood + _prior_penalty(problem, values),
            "success": bool(result.success) and likelihood < 1.0e11,
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
        }
    except Exception as exc:
        values = _values_from_vector(problem, start)
        return {
            "start_id": int(task["start_id"]),
            "start": tuple(float(value) for value in start),
            "estimate": tuple(float(value) for value in start),
            "values": values,
            "likelihood_nll": 1.0e12,
            "prior_penalty": _prior_penalty(problem, values),
            "posterior_objective": 1.0e12,
            "success": False,
            "message": f"{type(exc).__name__}: {exc}",
            "nfev": 0,
            "nit": 0,
        }


def _profile_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    problem: FitProblem = task["problem"]
    objective = _objective(problem)
    try:
        result = minimize_profile_point(
            objective,
            task["bounds"],
            int(task["fixed_index"]),
            float(task["fixed_value"]),
            task["starts"],
            max_iterations=int(task["max_iterations"]),
            ftol=float(task["ftol"]),
        )
        values = _values_from_vector(problem, result["x"])
        result.update({
            "values": values,
            "likelihood_nll": float(result.pop("objective")),
            "prior_penalty": _prior_penalty(problem, values),
        })
        return {**dict(task["identity"]), **result}
    except Exception as exc:
        fallback = list(float(value) for value in task["starts"][0])
        fallback[int(task["fixed_index"])] = float(task["fixed_value"])
        return {
            **dict(task["identity"]),
            "start_id": -1,
            "x": tuple(fallback),
            "values": _values_from_vector(problem, fallback),
            "likelihood_nll": 1.0e12,
            "prior_penalty": float("nan"),
            "success": False,
            "message": f"{type(exc).__name__}: {exc}",
            "nfev": 0,
            "nit": 0,
        }


def _score_worker(problem: FitProblem) -> tuple[str, float, int]:
    values = _parameter_values(problem.base_parameters)
    n_observations = int(sum(np.count_nonzero(np.isfinite(observations)) for _, observations in problem.trials))
    likelihood = _likelihood_nll(problem, values)
    score = likelihood / max(n_observations, 1) if likelihood < 1.0e11 else float("nan")
    return problem.case_id, score, n_observations


def _parallel_map(function: Callable[[Any], Any], tasks: Sequence[Any], workers: int) -> list[Any]:
    if not tasks:
        return []
    if int(workers) == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks))) as pool:
        return list(pool.map(function, tasks))


def _boundary_names(
    values: Mapping[str, float],
    names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    fraction: float,
) -> list[str]:
    result = []
    for name, (lower, upper) in zip(names, bounds):
        transformed = _to_optimizer_coordinate(name, float(values[name]))
        if min(transformed - lower, upper - transformed) <= float(fraction) * (upper - lower):
            result.append(name)
    return result


def _bound_expansion_status(
    name: str,
    registered_reference_boundary: Sequence[str],
    expanded_at_or_beyond_registered_boundary: Sequence[str],
    expanded_outer_boundary: Sequence[str],
) -> str:
    if name in registered_reference_boundary:
        if name in expanded_outer_boundary:
            return "persistent_at_expanded_boundary"
        if name in expanded_at_or_beyond_registered_boundary:
            return "active_constraint_relieved_estimate_at_or_beyond_registered_limit"
        return "relieved_after_expansion_into_registered_interior"
    if name in expanded_outer_boundary:
        return "new_expanded_boundary_contact"
    if name in expanded_at_or_beyond_registered_boundary:
        return "moved_to_or_beyond_registered_boundary"
    return "interior_both_ranges"


def _parameter_distance_fraction(
    reference_values: Mapping[str, float],
    candidate_values: Mapping[str, float],
    names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
) -> float:
    resolved = _validated_bounds(bounds, len(names))
    return max(
        abs(
            _to_optimizer_coordinate(name, float(candidate_values[name]))
            - _to_optimizer_coordinate(name, float(reference_values[name]))
        )
        / (upper - lower)
        for name, (lower, upper) in zip(names, resolved)
    )


def _run_multistart(
    problem: FitProblem,
    config: Mapping[str, Any],
    *,
    variant: str,
    bounds: Sequence[tuple[float, float]],
    warm_values: Mapping[str, float] | None,
    seed_offset: int,
) -> list[dict[str, Any]]:
    analysis = config["analysis"]
    starts = transformed_multistarts(
        problem.active,
        problem.parameter_specs,
        int(analysis["transformed_multistarts"]),
        int(config["experiment"]["seed"]) + int(seed_offset),
        bounds=bounds,
        warm_values=warm_values,
    )
    tasks = [
        {
            "problem": problem,
            "start": start,
            "start_id": index,
            "bounds": tuple(bounds),
            "max_iterations": int(analysis["optimizer_max_iterations"]),
            "ftol": float(analysis["optimizer_ftol"]),
        }
        for index, start in enumerate(starts)
    ]
    records = _parallel_map(_optimizer_worker, tasks, int(analysis["workers"]))
    boundary_fraction = float(analysis["boundary_fraction_in_transformed_space"])
    for record in records:
        record.update({
            "case_id": problem.case_id,
            "source_kind": problem.source_kind,
            "representative_role": problem.representative_role,
            "fit_variant": variant,
            "boundary_parameters": ";".join(_boundary_names(record["values"], problem.active, bounds, boundary_fraction)),
        })
        for name in problem.active:
            record[f"start_{name}"] = _from_optimizer_coordinate(name, record["start"][problem.active.index(name)])
            record[f"estimate_{name}"] = float(record["values"][name])
        record.pop("start")
        record.pop("estimate")
    return sorted(records, key=lambda row: int(row["start_id"]))


def _profile_grid(lower: float, upper: float, optimum: float, points: int) -> np.ndarray:
    grid = np.linspace(float(lower), float(upper), int(points), dtype=np.float64)
    if lower < optimum < upper and len(grid) > 2:
        replace_index = min(range(1, len(grid) - 1), key=lambda index: abs(float(grid[index]) - float(optimum)))
        grid[replace_index] = float(optimum)
        grid.sort()
    return grid


def _run_profiles(
    problem: FitProblem,
    reference_values: Mapping[str, float],
    config: Mapping[str, Any],
    bounds: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    analysis = config["analysis"]
    reference = tuple(_to_optimizer_coordinate(name, float(reference_values[name])) for name in problem.active)
    prior = tuple(_to_optimizer_coordinate(name, float(problem.parameter_specs[name]["prior_mean"])) for name in problem.active)
    starts = (reference, prior)[: int(analysis["profile_multistarts"])]
    tasks: list[dict[str, Any]] = []
    for fixed_index, parameter in enumerate(problem.active):
        lower, upper = bounds[fixed_index]
        grid = _profile_grid(lower, upper, reference[fixed_index], int(analysis["profile_points"]))
        for grid_index, fixed_value in enumerate(grid):
            tasks.append({
                "problem": problem,
                "bounds": tuple(bounds),
                "fixed_index": fixed_index,
                "fixed_value": float(fixed_value),
                "starts": starts,
                "max_iterations": int(analysis["optimizer_max_iterations"]),
                "ftol": float(analysis["optimizer_ftol"]),
                "identity": {
                    "case_id": problem.case_id,
                    "source_kind": problem.source_kind,
                    "representative_role": problem.representative_role,
                    "parameter": parameter,
                    "grid_index": int(grid_index),
                    "fixed_value": _from_optimizer_coordinate(parameter, float(fixed_value)),
                    "fixed_transformed_value": float(fixed_value),
                },
            })
    rows = _parallel_map(_profile_worker, tasks, int(analysis["workers"]))
    for row in rows:
        for name in problem.active:
            row[f"estimate_{name}"] = float(row["values"][name])
        row.pop("x", None)
    return rows


def _smooth_outputs(
    problem: FitProblem, values: Mapping[str, float]
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    parameters = _replace_parameter_values(problem.base_parameters, values)
    observations = []
    drivers = []
    likelihood = 0.0
    for _, trial in problem.trials:
        result = smooth_balloon(
            np.asarray(trial, dtype=np.float64),
            parameters=parameters,
            observation_spec=problem.observation_spec,
            config=problem.balloon_config,
        )
        observations.append(np.asarray(result.observation_mean, dtype=np.float64))
        drivers.append(np.asarray(result.state_mean[:, 0], dtype=np.float64))
        likelihood -= float(result.predictive_log_likelihood)
    return observations, drivers, float(likelihood)


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 3 or np.std(a[valid]) <= 1e-12 or np.std(b[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _trajectory_difference(
    reference_observations: Sequence[np.ndarray],
    reference_drivers: Sequence[np.ndarray],
    alternative_observations: Sequence[np.ndarray],
    alternative_drivers: Sequence[np.ndarray],
    spec: BalloonObservationSpec,
) -> dict[str, float]:
    ref_obs = np.concatenate(reference_observations, axis=0)
    alt_obs = np.concatenate(alternative_observations, axis=0)
    scales = np.asarray(spec.observation_scale, dtype=np.float64) * math.sqrt(float(spec.student_nu) / (float(spec.student_nu) - 2.0))
    whitened_rmse = float(np.sqrt(np.mean(np.square((alt_obs - ref_obs) / scales[None, :]))))
    ref_driver = np.concatenate(reference_drivers)
    alt_driver = np.concatenate(alternative_drivers)
    driver_scale = max(float(np.std(ref_driver)), 1.0e-8)
    driver_nrmse = float(np.sqrt(np.mean(np.square(alt_driver - ref_driver))) / driver_scale)
    return {
        "observation_whitened_rmse": whitened_rmse,
        "driver_nrmse": driver_nrmse,
        "driver_correlation": _safe_correlation(ref_driver, alt_driver),
    }


def _sensitivity_svd(
    problem: FitProblem,
    reference_values: Mapping[str, float],
    reference_drivers: Sequence[np.ndarray],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = tuple(config["analysis"]["sensitivity_parameters"])
    step = float(config["analysis"]["sensitivity_step_in_transformed_space"])
    scales = np.asarray(problem.observation_spec.observation_scale, dtype=np.float64) * math.sqrt(
        float(problem.observation_spec.student_nu) / (float(problem.observation_spec.student_nu) - 2.0)
    )
    columns = []
    for name in names:
        centre = _to_optimizer_coordinate(name, float(reference_values[name]))
        plus_values = dict(reference_values)
        minus_values = dict(reference_values)
        plus_values[name] = _from_optimizer_coordinate(name, centre + step)
        minus_values[name] = _from_optimizer_coordinate(name, centre - step)
        plus_parameters = _replace_parameter_values(problem.base_parameters, plus_values)
        minus_parameters = _replace_parameter_values(problem.base_parameters, minus_values)
        pieces = []
        for driver in reference_drivers:
            plus = simulate_balloon(
                driver,
                parameters=plus_parameters,
                observation_spec=problem.observation_spec,
                config=problem.balloon_config,
                add_noise=False,
            ).clean_observations
            minus = simulate_balloon(
                driver,
                parameters=minus_parameters,
                observation_spec=problem.observation_spec,
                config=problem.balloon_config,
                add_noise=False,
            ).clean_observations
            pieces.append(((plus - minus) / (2.0 * step) / scales[None, :]).reshape(-1))
        columns.append(np.concatenate(pieces))
    jacobian = np.column_stack(columns)
    rows, summary = svd_diagnostics(
        jacobian,
        names,
        float(config["analysis"]["sensitivity_relative_singular_value_threshold"]),
    )
    active_indices = [names.index(name) for name in problem.active]
    active_rows, active_summary = svd_diagnostics(
        jacobian[:, active_indices],
        problem.active,
        float(config["analysis"]["sensitivity_relative_singular_value_threshold"]),
    )
    summary.update({
        "active_parameter_names": list(problem.active),
        "active_parameter_effective_rank": active_summary["effective_rank"],
        "active_parameter_condition_on_nonzero_singular_values": active_summary["condition_on_nonzero_singular_values"],
        "active_parameter_singular_values": [row["singular_value"] for row in active_rows],
    })
    for row in rows:
        row.update({
            "case_id": problem.case_id,
            "source_kind": problem.source_kind,
            "representative_role": problem.representative_role,
            "sensitivity_definition": "conditional_forward_at_fixed_driver_whitened_by_declared_student_t_marginal_sd",
        })
    return rows, summary


def _near_optimal_candidates(
    candidates: Sequence[Mapping[str, Any]],
    active: Sequence[str],
    reference_nll: float,
    delta: float,
) -> list[Mapping[str, Any]]:
    eligible = [row for row in candidates if float(row["likelihood_nll"]) <= float(reference_nll) + float(delta)]
    unique: dict[tuple[float, ...], Mapping[str, Any]] = {}
    for row in sorted(eligible, key=lambda value: float(value["likelihood_nll"])):
        key = tuple(round(float(row["values"][name]), 10) for name in active)
        unique.setdefault(key, row)
    return list(unique.values())


def _profile_support(
    rows: Sequence[Mapping[str, Any]], active: Sequence[str], delta: float
) -> dict[str, dict[str, Any]]:
    result = {}
    for name in active:
        ordered = sorted((row for row in rows if row["parameter"] == name), key=lambda row: int(row["grid_index"]))
        supported = [row for row in ordered if bool(row["success"]) and float(row["delta_nll"]) <= float(delta)]
        supported_indices = [int(row["grid_index"]) for row in supported]
        result[name] = {
            "support_lower": min((float(row["fixed_value"]) for row in supported), default=float("nan")),
            "support_upper": max((float(row["fixed_value"]) for row in supported), default=float("nan")),
            "touches_lower_grid": bool(supported and supported[0] is ordered[0]),
            "touches_upper_grid": bool(supported and supported[-1] is ordered[-1]),
            "support_is_contiguous": bool(
                supported_indices
                and supported_indices == list(range(supported_indices[0], supported_indices[-1] + 1))
            ),
            "grid_points": len(ordered),
            "converged_finite_grid_points": int(sum(bool(row["success"]) and float(row["likelihood_nll"]) < 1.0e11 for row in ordered)),
        }
    return result


def _profile_reference_check(
    rows: Sequence[Mapping[str, Any]],
    active: Sequence[str],
    reference_nll: float,
    tolerance: float,
) -> tuple[dict[str, float], bool]:
    differences = {
        name: min(
            (
                float(row["likelihood_nll"]) - float(reference_nll)
                for row in rows
                if row["parameter"] == name
                and bool(row["success"])
                and float(row["likelihood_nll"]) < 1.0e11
            ),
            default=float("nan"),
        )
        for name in active
    }
    consistent = all(np.isfinite(value) and abs(value) <= float(tolerance) for value in differences.values())
    return differences, bool(consistent)


def primary_endpoint_values(cases: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    values: dict[str, bool] = {}
    for case in cases:
        profile_support = case["profile_support"]
        supports = profile_support.values()
        values[str(case["case_id"])] = bool(
            set(profile_support) == set(ACTIVE_PARAMETERS)
            and len(profile_support) == len(ACTIVE_PARAMETERS)
            and case["diagnostic_flags"]["profile_reference_consistent"]
            and all(
                int(value["converged_finite_grid_points"]) == int(value["grid_points"])
                and np.isfinite(float(value["support_lower"]))
                and np.isfinite(float(value["support_upper"]))
                and bool(value.get("support_is_contiguous", False))
                and not value["touches_lower_grid"]
                and not value["touches_upper_grid"]
                for value in supports
            )
        )
    return values


def _interpret_case(
    profile_support: Mapping[str, Mapping[str, Any]],
    original_boundary: Sequence[str],
    expanded_at_or_beyond_registered_boundary: Sequence[str],
    expanded_outer_boundary: Sequence[str],
    expanded_likelihood_near_or_better: bool,
    expanded_difference: Mapping[str, float],
    profile_reference_consistent: bool,
    state_rows: Sequence[Mapping[str, Any]],
    active_svd_rank: int,
    config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    analysis = config["analysis"]
    prediction_limit = float(analysis["prediction_equivalence_max_whitened_rmse"])
    parameter_difference_limit = float(analysis["material_parameter_difference_min_fraction_of_transformed_span"])
    driver_limit = float(analysis["driver_stability_max_nrmse"])
    correlation_limit = float(analysis["driver_stability_min_correlation"])
    profile_expected = set(profile_support) == set(ACTIVE_PARAMETERS) and len(profile_support) == len(ACTIVE_PARAMETERS)
    profile_touches = any(bool(value["touches_lower_grid"] or value["touches_upper_grid"]) for value in profile_support.values())
    profile_complete = profile_expected and all(int(value["converged_finite_grid_points"]) == int(value["grid_points"]) for value in profile_support.values())
    profile_supported = profile_expected and all(
        np.isfinite(float(value["support_lower"])) and np.isfinite(float(value["support_upper"]))
        for value in profile_support.values()
    )
    profile_contiguous = profile_expected and all(bool(value.get("support_is_contiguous", False)) for value in profile_support.values())
    alternatives = [row for row in state_rows if row["candidate_source"] != "reference"]
    equivalent = [row for row in alternatives if float(row["observation_whitened_rmse"]) <= prediction_limit]
    distinct_equivalent = [
        row for row in equivalent
        if float(row["parameter_distance_fraction_of_transformed_span"]) >= parameter_difference_limit
    ]
    latent_unstable = any(
        not np.isfinite(float(row["driver_nrmse"]))
        or float(row["driver_nrmse"]) > driver_limit
        or not np.isfinite(float(row["driver_correlation"]))
        or float(row["driver_correlation"]) < correlation_limit
        for row in distinct_equivalent
    )
    equivalent_stable = bool(distinct_equivalent) and not latent_unstable
    expanded_equivalent = float(expanded_difference["observation_whitened_rmse"]) <= prediction_limit
    flags = {
        "profile_touches_registered_bound": profile_touches,
        "profile_complete": profile_complete,
        "profile_has_likelihood_support": profile_supported,
        "profile_support_contiguous": profile_contiguous,
        "profile_reference_consistent": bool(profile_reference_consistent),
        "original_boundary_parameters": list(original_boundary),
        "expanded_at_or_beyond_registered_boundary_parameters": list(expanded_at_or_beyond_registered_boundary),
        "expanded_outer_boundary_parameters": list(expanded_outer_boundary),
        "expanded_likelihood_near_or_better": bool(expanded_likelihood_near_or_better),
        "expanded_prediction_equivalent": expanded_equivalent,
        "near_optimal_prediction_equivalent_count": len(equivalent),
        "materially_distinct_prediction_equivalent_count": len(distinct_equivalent),
        "material_parameter_difference_threshold_fraction_of_transformed_span": parameter_difference_limit,
        "near_optimal_equivalent_driver_unstable": latent_unstable,
        "near_optimal_equivalent_driver_stable": equivalent_stable,
        "sensitivity_active_effective_rank": int(active_svd_rank),
    }
    if latent_unstable:
        interpretation = "latent_nonunique_no_tokenizer_promotion"
    elif expanded_outer_boundary and expanded_likelihood_near_or_better and expanded_equivalent:
        interpretation = "persistent_boundary_ridge_or_model_misfit"
    elif equivalent_stable:
        interpretation = "parameters_nonidentifiable_but_state_stable"
    elif profile_complete and profile_supported and profile_contiguous and profile_reference_consistent and not profile_touches and not original_boundary and int(active_svd_rank) >= len(profile_support):
        interpretation = "locally_supported_within_registered_range_exploratory_only"
    else:
        interpretation = "inconclusive"
    return interpretation, flags


def _run_case(problem: FitProblem, config: Mapping[str, Any], case_index: int) -> dict[str, Any]:
    started = time.monotonic()
    analysis = config["analysis"]
    bounds = _transformed_bounds(problem.active, problem.parameter_specs)
    multistart = _run_multistart(
        problem,
        config,
        variant="registered_bounds",
        bounds=bounds,
        warm_values=None,
        seed_offset=case_index * 1000,
    )
    finite_multistart = [row for row in multistart if row["success"] and float(row["likelihood_nll"]) < 1.0e11]
    if not finite_multistart:
        raise RuntimeError(f"{problem.case_id}: all registered-bound optimizer starts failed")
    initial_best = min(finite_multistart, key=lambda row: float(row["likelihood_nll"]))
    profiles = _run_profiles(problem, initial_best["values"], config, bounds)
    finite_profiles = [row for row in profiles if row["success"] and float(row["likelihood_nll"]) < 1.0e11]
    reference = initial_best
    reference_nll = float(reference["likelihood_nll"])
    for row in profiles:
        row["delta_nll"] = float(row["likelihood_nll"]) - reference_nll
        row["within_profile_likelihood_support"] = bool(
            row["success"] and float(row["delta_nll"]) <= float(analysis["profile_likelihood_ratio_delta_nll"])
        )
        row["latent_state_reoptimized"] = bool(row["success"])
    profile_support = _profile_support(
        profiles, problem.active, float(analysis["profile_likelihood_ratio_delta_nll"])
    )
    profile_minimum_minus_reference, profile_reference_consistent = _profile_reference_check(
        profiles,
        problem.active,
        reference_nll,
        float(analysis["profile_reference_consistency_tolerance_nll"]),
    )
    expanded_bounds = expanded_transformed_bounds(
        bounds, float(analysis["expanded_bound_fraction_in_transformed_space"])
    )
    expanded = _run_multistart(
        problem,
        config,
        variant="expanded_bounds",
        bounds=expanded_bounds,
        warm_values=reference["values"],
        seed_offset=case_index * 1000 + 500,
    )
    finite_expanded = [row for row in expanded if row["success"] and float(row["likelihood_nll"]) < 1.0e11]
    if not finite_expanded:
        raise RuntimeError(f"{problem.case_id}: all expanded-bound optimizer starts failed")
    expanded_best = min(finite_expanded, key=lambda row: float(row["likelihood_nll"]))
    expanded_likelihood_near_or_better = bool(
        float(expanded_best["likelihood_nll"])
        <= reference_nll + float(analysis["profile_likelihood_ratio_delta_nll"])
    )
    reference_observations, reference_drivers, reference_check_nll = _smooth_outputs(problem, reference["values"])
    expanded_observations, expanded_drivers, expanded_check_nll = _smooth_outputs(problem, expanded_best["values"])
    expanded_difference = _trajectory_difference(
        reference_observations,
        reference_drivers,
        expanded_observations,
        expanded_drivers,
        problem.observation_spec,
    )
    original_boundary = _boundary_names(
        reference["values"], problem.active, bounds, float(analysis["boundary_fraction_in_transformed_space"])
    )
    expanded_at_or_beyond_registered_boundary = _boundary_names(
        expanded_best["values"], problem.active, bounds, float(analysis["boundary_fraction_in_transformed_space"])
    )
    expanded_outer_boundary = _boundary_names(
        expanded_best["values"], problem.active, expanded_bounds, float(analysis["boundary_fraction_in_transformed_space"])
    )
    expanded_rows = []
    for index, name in enumerate(problem.active):
        expansion_status = _bound_expansion_status(
            name,
            original_boundary,
            expanded_at_or_beyond_registered_boundary,
            expanded_outer_boundary,
        )
        expanded_rows.append({
            "case_id": problem.case_id,
            "source_kind": problem.source_kind,
            "representative_role": problem.representative_role,
            "parameter": name,
            "registered_lower": _from_optimizer_coordinate(name, bounds[index][0]),
            "registered_upper": _from_optimizer_coordinate(name, bounds[index][1]),
            "expanded_lower": _from_optimizer_coordinate(name, expanded_bounds[index][0]),
            "expanded_upper": _from_optimizer_coordinate(name, expanded_bounds[index][1]),
            "registered_estimate": float(reference["values"][name]),
            "expanded_estimate": float(expanded_best["values"][name]),
            "registered_boundary": name in original_boundary,
            "expanded_at_or_beyond_registered_boundary": name in expanded_at_or_beyond_registered_boundary,
            "expanded_outer_boundary": name in expanded_outer_boundary,
            "bound_expansion_status": expansion_status,
            "registered_likelihood_nll": reference_nll,
            "expanded_likelihood_nll": float(expanded_best["likelihood_nll"]),
            "expanded_minus_registered_nll": float(expanded_best["likelihood_nll"]) - reference_nll,
            **expanded_difference,
        })
    sensitivity_drivers = list(problem.truth_drivers) if problem.truth_drivers is not None else reference_drivers
    svd_rows, svd_summary = _sensitivity_svd(problem, reference["values"], sensitivity_drivers, config)
    candidate_rows: list[Mapping[str, Any]] = []
    for row in finite_multistart:
        candidate_rows.append({**row, "candidate_source": f"multistart_{row['start_id']}"})
    for row in finite_profiles:
        candidate_rows.append({**row, "candidate_source": f"profile_{row['parameter']}_{row['grid_index']}"})
    candidate_rows.append({**expanded_best, "candidate_source": "expanded_best"})
    near_optimal_candidates = _near_optimal_candidates(
        candidate_rows,
        problem.active,
        reference_nll,
        float(analysis["profile_likelihood_ratio_delta_nll"]),
    )
    state_rows = [{
        "case_id": problem.case_id,
        "source_kind": problem.source_kind,
        "representative_role": problem.representative_role,
        "candidate_source": "reference",
        "likelihood_nll": reference_check_nll,
        "delta_nll": 0.0,
        "observation_whitened_rmse": 0.0,
        "driver_nrmse": 0.0,
        "driver_correlation": 1.0,
        "parameter_distance_fraction_of_transformed_span": 0.0,
        **{f"estimate_{name}": float(reference["values"][name]) for name in problem.active},
    }]
    reference_key = tuple(round(float(reference["values"][name]), 10) for name in problem.active)
    for candidate in near_optimal_candidates:
        key = tuple(round(float(candidate["values"][name]), 10) for name in problem.active)
        if key == reference_key:
            continue
        observations, drivers, checked_nll = _smooth_outputs(problem, candidate["values"])
        state_rows.append({
            "case_id": problem.case_id,
            "source_kind": problem.source_kind,
            "representative_role": problem.representative_role,
            "candidate_source": str(candidate["candidate_source"]),
            "likelihood_nll": checked_nll,
            "delta_nll": checked_nll - reference_check_nll,
            "parameter_distance_fraction_of_transformed_span": _parameter_distance_fraction(
                reference["values"], candidate["values"], problem.active, bounds
            ),
            **_trajectory_difference(reference_observations, reference_drivers, observations, drivers, problem.observation_spec),
            **{f"estimate_{name}": float(candidate["values"][name]) for name in problem.active},
        })
    interpretation, flags = _interpret_case(
        profile_support,
        original_boundary,
        expanded_at_or_beyond_registered_boundary,
        expanded_outer_boundary,
        expanded_likelihood_near_or_better,
        expanded_difference,
        profile_reference_consistent,
        state_rows,
        int(svd_summary["active_parameter_effective_rank"]),
        config,
    )
    recovery_rows = []
    if problem.truth_parameters is not None:
        for name in PARAMETER_NAMES:
            truth = float(problem.truth_parameters[name])
            estimate = float(reference["values"][name])
            recovery_rows.append({
                "case_id": problem.case_id,
                "parameter": name,
                "is_fitted": name in problem.active,
                "truth": truth,
                "estimate": estimate,
                "error": estimate - truth,
                "relative_error": (estimate - truth) / max(abs(truth), 1.0e-12),
            })
    summary = {
        "case_id": problem.case_id,
        "source_kind": problem.source_kind,
        "representative_role": problem.representative_role,
        "trial_count": len(problem.trials),
        "finite_observations": int(sum(np.count_nonzero(np.isfinite(values)) for _, values in problem.trials)),
        "registered_multistarts": len(multistart),
        "registered_multistart_successes": len(finite_multistart),
        "expanded_multistarts": len(expanded),
        "expanded_multistart_successes": len(finite_expanded),
        "profile_grid_points_per_parameter": int(analysis["profile_points"]),
        "profile_converged_points": len(finite_profiles),
        "multistart_parameter_ranges": {
            name: {
                "minimum": min(float(row["values"][name]) for row in finite_multistart),
                "maximum": max(float(row["values"][name]) for row in finite_multistart),
            }
            for name in problem.active
        },
        "reference_source": str(reference.get("candidate_source", "profile" if "parameter" in reference else f"multistart_{reference.get('start_id', -1)}")),
        "reference_likelihood_nll": reference_nll,
        "reference_parameters": {name: float(reference["values"][name]) for name in PARAMETER_NAMES},
        "profile_support": profile_support,
        "profile_minimum_minus_unconstrained_reference_nll_by_parameter": profile_minimum_minus_reference,
        "expanded_parameters": {name: float(expanded_best["values"][name]) for name in PARAMETER_NAMES},
        "expanded_likelihood_nll": float(expanded_best["likelihood_nll"]),
        "expanded_trajectory_difference": expanded_difference,
        "sensitivity_svd": svd_summary,
        "state_stability_candidate_count": len(state_rows),
        "interpretation": interpretation,
        "diagnostic_flags": flags,
        "elapsed_seconds": time.monotonic() - started,
    }
    return {
        "summary": summary,
        "multistart_rows": [*multistart, *expanded],
        "profile_rows": profiles,
        "expanded_rows": expanded_rows,
        "svd_rows": svd_rows,
        "state_rows": state_rows,
        "recovery_rows": recovery_rows,
    }


def _synthetic_problem(config: Mapping[str, Any], measured_parameter_specs: Mapping[str, Mapping[str, Any]]) -> FitProblem:
    source_path = REPO_ROOT / str(config["sources"]["synthetic_config"])
    source = load_synthetic_config(source_path)
    synthetic = config["synthetic"]
    if synthetic.get("truth_parameters") != "prior_center":
        raise ValueError("the synthetic identifiability case must use prior-centre truth")
    case = generate_case(int(synthetic["replicate_id"]), int(synthetic["seed"]), source)
    fixed_cfg = source["physiology"]["fixed"]
    observation_cfg = source["observation"]
    process_cfg = observation_cfg["process_sd"]
    fixed = BalloonFixedParameters(
        alpha=float(fixed_cfg["alpha"]),
        E0=float(fixed_cfg["e0"]),
        gamma=float(fixed_cfg["gamma"]),
        P0=float(fixed_cfg["p0"]),
        Q0=float(fixed_cfg["q0"]),
        driver_decay_per_s=float(source["driver"]["decay_per_s"]),
        process_std=tuple(float(process_cfg[name]) for name in ("r", "s", "log_f", "log_v", "log_p", "log_q")),
        observation_scale=tuple(float(observation_cfg["scale"][name]) for name in ("EEG", "HbO", "HbR")),
        student_nu=float(observation_cfg["student_t_df"]),
        eeg_loading=float(observation_cfg["eeg_loading"]),
        eeg_offset=float(observation_cfg["eeg_offset"]),
        neurovascular_gain=1.0,
    )
    parameters = BalloonParameters(
        fixed=fixed,
        free=BalloonFreeParameters(
            kappa=float(source["physiology"]["truth"]["kappa_per_s"]),
            tau=float(source["physiology"]["truth"]["tau_s"]),
        ),
    )
    balloon_config = BalloonConfig(
        dt=1.0 / float(source["simulation"]["sampling_hz"]),
        rk4_substeps=2,
        irls_iterations=int(source["inference"]["irls_iterations"]),
    )
    spec = BalloonObservationSpec(
        eeg_loading=fixed.eeg_loading,
        eeg_offset=fixed.eeg_offset,
        observation_scale=fixed.observation_scale,
        student_nu=fixed.student_nu,
    )
    truth = {
        "beta": 1.0,
        "kappa": float(case.true_parameters["kappa_per_s"]),
        "tau": float(case.true_parameters["tau_s"]),
        "gamma": float(case.true_parameters["gamma"]),
        "alpha": float(case.true_parameters["alpha"]),
        "E0": float(case.true_parameters["e0"]),
    }
    return FitProblem(
        case_id="synthetic_replicate_0",
        source_kind="synthetic_known_truth",
        representative_role="synthetic",
        trials=((int(case.replicate_id), np.asarray(case.observations, dtype=np.float64)),),
        base_parameters=parameters,
        observation_spec=spec,
        balloon_config=balloon_config,
        active=ACTIVE_PARAMETERS,
        parameter_specs=copy.deepcopy({name: dict(value) for name, value in measured_parameter_specs.items()}),
        truth_parameters=truth,
        truth_drivers=(np.asarray(case.truth_states[:, 0], dtype=np.float64),),
    )


def _measured_problems(
    config: Mapping[str, Any], measured: Mapping[str, Any]
) -> tuple[list[FitProblem], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _validate_measured_source(measured)
    loader_config = _fit_only_measured_config(measured)
    heldout_positions = tuple(config["selection"]["heldout_trial_positions"])
    raw_trials, fit_series, validation, contracts, adapter, hbo_names, hbr_indices = _prepare_measured_series(
        loader_config,
        gauge_excluded_trial_positions=heldout_positions,
    )
    if validation:
        raise RuntimeError("fit-only loader unexpectedly returned validation series")
    fit_subjects = list(map(str, measured["data"]["conditions"][0]["fit_subjects"]))
    if {trial.subject for trial in raw_trials} != set(fit_subjects):
        raise RuntimeError("fit-only preprocessing returned an unexpected subject set")
    fit_config = measured["ssm"]["t3a"]["parameter_fit"]
    excluded = set(map(int, heldout_positions))
    gauge_series = []
    for subject in fit_subjects:
        ordered = sorted(
            (item for item in fit_series if item.trial.subject == subject),
            key=lambda item: int(item.trial.event_index),
        )
        if len(ordered) != 10:
            raise RuntimeError(f"{subject}: expected exactly ten fit-subject trials")
        gauge_series.extend(item for index, item in enumerate(ordered) if index not in excluded)
    bundle, calibration = _fit_models(gauge_series, measured, fit_comparison_models=False)
    base_parameters, observation_spec, balloon_config = bundle.t3a
    train, _ = _split_stage_trials(fit_series, fit_subjects, heldout_positions)
    specs = copy.deepcopy({name: dict(value) for name, value in fit_config["parameters"].items()})
    all_problems = [
        FitProblem(
            case_id=subject,
            source_kind="measured_fit_only",
            representative_role="unselected",
            trials=train[subject],
            base_parameters=base_parameters,
            observation_spec=observation_spec,
            balloon_config=balloon_config,
            active=ACTIVE_PARAMETERS,
            parameter_specs=specs,
        )
        for subject in fit_subjects
    ]
    score_records = _parallel_map(_score_worker, all_problems, int(config["analysis"]["workers"]))
    scores = {subject: score for subject, score, _ in score_records}
    selected = select_representative_fit_subjects(scores, fit_subjects)
    selected_map = {row["subject"]: row["role"] for row in selected}
    score_details = {subject: n for subject, _, n in score_records}
    selection_rows = []
    ordered = sorted(scores, key=lambda subject: (scores[subject], subject))
    for rank, subject in enumerate(ordered, start=1):
        selection_rows.append({
            "subject": subject,
            "rank": rank,
            "score": scores[subject],
            "finite_observations": score_details[subject],
            "selected": subject in selected_map,
            "role": selected_map.get(subject, "not_selected"),
            "metric": config["selection"]["metric"],
            "trial_scope": "eight_fit_trials_excluding_positions_4_and_9",
            "event_indices": ";".join(str(event_index) for event_index, _ in train[subject]),
        })
    problems = [
        FitProblem(
            case_id=problem.case_id,
            source_kind=problem.source_kind,
            representative_role=selected_map[problem.case_id],
            trials=problem.trials,
            base_parameters=problem.base_parameters,
            observation_spec=problem.observation_spec,
            balloon_config=problem.balloon_config,
            active=problem.active,
            parameter_specs=problem.parameter_specs,
        )
        for problem in all_problems
        if problem.case_id in selected_map
    ]
    problems.sort(key=lambda problem: ("low", "median", "high").index(problem.representative_role))
    trial_inventory = []
    condition = measured["data"]["conditions"][0]
    for subject in fit_subjects:
        ordered_series = sorted(
            (item for item in fit_series if item.trial.subject == subject),
            key=lambda item: int(item.trial.event_index),
        )
        for position, item in enumerate(ordered_series):
            gauge_included = position not in excluded
            analysis_included = gauge_included and subject in selected_map
            observations = np.column_stack((item.eeg_driver, item.hbo, item.hbr))
            trial_inventory.append({
                **_identity_fields(item, condition, measured["data"]),
                "trial_position": position,
                "trial_position_base": 0,
                "loader_included": True,
                "gauge_included": gauge_included,
                "selection_score_included": gauge_included,
                "identifiability_analysis_included": analysis_included,
                "trial_role": "fit_train" if gauge_included else "internal_holdout_not_used",
                "representative_role": selected_map.get(subject, "not_selected"),
                "finite_derived_observations": int(np.count_nonzero(np.isfinite(observations))),
                "derived_observation_count": int(observations.size),
            })
    calibration = dict(calibration)
    calibration.update({
        "scope": "subjects_01_18_fit_only",
        "selected_hbo_channels": list(hbo_names),
        "selected_hbr_channels": [raw_trials[0].fnirs_channel_names[int(index)] for index in hbr_indices],
        "eeg_adapter": {
            "indices": adapter.indices,
            "channel_names": adapter.channel_names,
            "feature_mean": adapter.feature_mean,
            "feature_std": adapter.feature_std,
            "pca_mean": adapter.pca_mean,
            "loading": adapter.loading,
            "pc_scale": adapter.pc_scale,
        },
    })
    counts = {
        "gauge_fit_subjects": len(fit_subjects),
        "loaded_fit_trials": len(raw_trials),
        "gauge_fit_trials": len(gauge_series),
        "selection_fit_trials": sum(len(value) for value in train.values()),
        "analyzed_subjects": len(problems),
        "analyzed_fit_trials": sum(len(problem.trials) for problem in problems),
        "validation_trials_loaded": 0,
        "protected_trials_loaded": 0,
    }
    return problems, selection_rows, trial_inventory, calibration, contracts, counts


def _markdown_report(summary: Mapping[str, Any]) -> str:
    selected = [row for row in summary["representative_selection"] if row["selected"]]
    lines = [
        "# T3 第二步：可辨识性实验报告",
        "",
        f"- 状态：**{summary['completion_status']}**",
        f"- 科学判定：`{summary['scientific_verdict']}`。",
        f"- 范围：`{summary['scope']}`；仅拟合集，资格与决策均不适用。",
        f"- 父运行：`{summary['parent_run_id']}`，仅作上下文，未消费其数组。",
        f"- estimand：`{summary['estimand_id']}`。",
        f"- 假设：`{summary['hypothesis']}`。",
        f"- 主终点：`{summary['primary_endpoint']['id']}`；全部案例支持 = **{summary['primary_endpoint']['supported_in_all_cases']}**。",
        f"- operator：`{summary['operator']}`；null：`{summary['null']}`（本 suite 不运行 cross-modal null）。",
        "- 数据边界：01–18 每人 8 个训练 trial 用于 gauge 与 M0 选择；仅 3 名代表的这些训练 trial 进入 M2；19–23 与 24–29 的数组未加载、窗口样本未物化（共享 loader 仍构建规范数据集索引元数据和 window refs）。",
        "- 主分析：β/κ/τ 的 likelihood-only 16 个变换空间起点、真 profile likelihood、扩大边界重拟合；六参数条件前向白化 SVD。",
        "",
        "## 代表被试",
        "",
        "| 角色 | 被试 | M0 pooled NLL/观测 |",
        "| --- | --- | ---: |",
    ]
    lines.extend(f"| {row['role']} | {row['subject']} | {float(row['score']):.6g} |" for row in selected)
    lines.extend(["", "## 每案例结果", "", "| 案例 | NLL/观测 | β | κ | τ | profile 触界 | 扩边越原界 | 扩边外界 | SVD rank（六参数/βκτ） | 解释 |", "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |"])
    for case in summary["cases"]:
        parameters = case["reference_parameters"]
        per_observation = float(case["reference_likelihood_nll"]) / max(int(case["finite_observations"]), 1)
        profile_touch = ",".join(name for name, value in case["profile_support"].items() if value["touches_lower_grid"] or value["touches_upper_grid"]) or "无"
        expanded_registered = ",".join(case["diagnostic_flags"]["expanded_at_or_beyond_registered_boundary_parameters"]) or "无"
        expanded_outer = ",".join(case["diagnostic_flags"]["expanded_outer_boundary_parameters"]) or "无"
        lines.append(
            f"| {case['case_id']} | {per_observation:.6g} | {parameters['beta']:.5g} | {parameters['kappa']:.5g} | {parameters['tau']:.5g} | {profile_touch} | {expanded_registered} | {expanded_outer} | {case['sensitivity_svd']['effective_rank']}/{case['sensitivity_svd']['active_parameter_effective_rank']} | `{case['interpretation']}` |"
        )
    recovery = summary.get("synthetic_parameter_recovery", [])
    lines.extend(["", "## 合成真值恢复", "", "| 参数 | 真值 | 估计 | 相对误差 |", "| --- | ---: | ---: | ---: |"])
    for row in recovery:
        if row["is_fitted"]:
            lines.append(f"| {row['parameter']} | {float(row['truth']):.6g} | {float(row['estimate']):.6g} | {float(row['relative_error']):.2%} |")
    lines.extend([
        "",
        "## 解释边界",
        "",
        "profile 的每个网格点固定目标参数，同时重新优化另外两个参数；每次目标函数评估都会重新运行 smoother，因此 latent state 也被条件重估。SVD 是在固定 driver 下对 log/logit 参数的条件前向敏感度，不等价于完整的联合可辨识性证明。",
        "",
        "合成案例固定在 prior-centre 真值并加入 P0 clean scenario 的异方差 Student-t 观测噪声；独立真值由 solve_ivp 生成，而条件敏感度使用候选 RK4 前向器，因此二者不是同一个数值积分器。输入 manifest/hash 的唯一 owner 是 manifest.json；逐 trial 纳入状态见 trial_inventory.csv。",
        "",
        "本实验是 exploratory fit-only diagnostic。任何局部支持、状态稳定或合成恢复结果都不能覆盖现有 P0 失败，不能授权验证集、保护集、teacher 资格或 tokenizer promotion。",
        "",
    ])
    return "\n".join(lines)


def _write_result_tables(output_dir: Path, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    def public(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{key: value for key, value in row.items() if key != "values"} for row in values]

    _atomic_csv(output_dir / "representative_selection.csv", rows["selection"])
    _atomic_csv(output_dir / "trial_inventory.csv", rows["inventory"])
    _atomic_csv(output_dir / "multistart_results.csv", public(rows["multistart"]))
    _atomic_csv(output_dir / "profile_likelihood.csv", public(rows["profile"]))
    _atomic_csv(output_dir / "expanded_bounds.csv", rows["expanded"])
    _atomic_csv(output_dir / "sensitivity_svd.csv", rows["svd"])
    _atomic_csv(output_dir / "state_stability.csv", rows["state"])
    _atomic_csv(output_dir / "parameter_recovery.csv", rows["recovery"])


def run(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run synthetic first, then open only the fit-subject measured view."""

    validate_config(config)
    output_root = (REPO_ROOT / str(config["output"]["root"])).resolve()
    resolved = Path(run_dir).resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"run directory must be below {output_root}") from exc
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    _atomic_write(resolved / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    started = datetime.now(timezone.utc).isoformat()
    boundary = {
        "schema": SCHEMA,
        "scope": "fit_only_measured_development_diagnostic",
        "status": "incomplete",
        "completion_status": "incomplete",
        "stage": "before_synthetic",
        "started_at": started,
        "gauge_fit_subjects": _subject_range(1, 18),
        "analyzed_measured_subjects": "pending_deterministic_M0_selection",
        "validation_subject_arrays_not_dereferenced": _subject_range(19, 23),
        "protected_subjects_closed": _subject_range(24, 29),
        "protected_subject_arrays_not_dereferenced": _subject_range(24, 29),
        "shared_loader_index_metadata_scope": "canonical dataset index; may enumerate records outside the fit-only selection",
        "validation_data_opened": False,
        "protected_data_opened": False,
        "measured_data_enabled": True,
        "protected_data_enabled": False,
        "qualification_eligible": False,
        "decision_eligibility": False,
        "parent_run_id": config["sources"]["parent_run_id"],
        "parent_run_role": config["sources"]["parent_run_role"],
        "hypothesis": config["estimand"]["hypothesis"],
        "estimand_id": config["estimand"]["estimand_id"],
        "primary_endpoint": config["estimand"]["primary_endpoint"],
        "operator": config["estimand"]["operator"],
        "null": config["estimand"]["null"],
        "split_identity": "subjects_01_18_positions_0_1_2_3_5_6_7_8_fit_only",
    }
    progress = dict(boundary)
    _atomic_json(resolved / "manifest.json", progress)
    _atomic_json(resolved / "summary.json", boundary)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("selection", "inventory", "multistart", "profile", "expanded", "svd", "state", "recovery")}
    try:
        measured_source = load_measured_config(REPO_ROOT / str(config["sources"]["measured_config"]))
        _validate_measured_source(measured_source)
        parameter_specs = measured_source["ssm"]["t3a"]["parameter_fit"]["parameters"]
        synthetic_result = _run_case(_synthetic_problem(config, parameter_specs), config, 0)
        for target, source in (("multistart", "multistart_rows"), ("profile", "profile_rows"), ("expanded", "expanded_rows"), ("svd", "svd_rows"), ("state", "state_rows"), ("recovery", "recovery_rows")):
            rows[target].extend(synthetic_result[source])
        _write_result_tables(resolved, rows)
        progress = {**progress, "stage": "synthetic_complete_before_measured_load", "synthetic_case": synthetic_result["summary"]}
        _atomic_json(resolved / "manifest.json", progress)

        cache_provenance = _cache_provenance(measured_source)
        progress = {
            **progress,
            "stage": "measured_fit_only_load_started",
            "input_hashes": {row["path"]: row["sha256"] for row in cache_provenance},
        }
        _atomic_json(resolved / "manifest.json", progress)
        validate_config(config)
        measured_problems, selection_rows, trial_inventory, calibration, contracts, measured_counts = _measured_problems(config, measured_source)
        rows["selection"].extend(selection_rows)
        rows["inventory"].extend(trial_inventory)
        data_identity = {
            "cache_root": measured_source["data"]["cache_root"],
            "dataset_id": measured_source["data"]["conditions"][0]["dataset_id"],
            "condition_id": measured_source["data"]["conditions"][0]["condition_id"],
            "record_id": measured_source["data"]["conditions"][0]["record_id"],
            "target_label": measured_source["data"]["conditions"][0]["target_label"],
            "eeg_signal_branch": measured_source["data"]["conditions"][0]["eeg_signal_branch"],
            "window_start_s": measured_source["data"]["window_offset_s"],
            "window_duration_s": measured_source["data"]["window_duration_s"],
            "sampling_hz": measured_source["analysis"]["sampling_hz"],
            "heldout_trial_positions_zero_based_not_used": list(config["selection"]["heldout_trial_positions"]),
            "mask_policy": "full_support_required_before_derived_observations",
            "trial_inventory": "trial_inventory.csv",
        }
        _write_result_tables(resolved, rows)
        progress = {
            **progress,
            "stage": "measured_fit_only_loaded_and_selected",
            "measured_counts": measured_counts,
            "analyzed_measured_subjects": [problem.case_id for problem in measured_problems],
            "data_identity": data_identity,
            "input_hashes": {row["path"]: row["sha256"] for row in cache_provenance},
            "loader_contracts": contracts,
            "trial_inventory": "trial_inventory.csv",
        }
        _atomic_json(resolved / "manifest.json", progress)
        measured_results = []
        for index, problem in enumerate(measured_problems, start=1):
            result = _run_case(problem, config, index)
            measured_results.append(result)
            for target, source in (("multistart", "multistart_rows"), ("profile", "profile_rows"), ("expanded", "expanded_rows"), ("svd", "svd_rows"), ("state", "state_rows"), ("recovery", "recovery_rows")):
                rows[target].extend(result[source])
            _write_result_tables(resolved, rows)
            progress = {
                **progress,
                "stage": f"measured_{problem.representative_role}_complete",
                "analyzed_measured_subjects": [item["summary"]["case_id"] for item in measured_results],
                "synthetic_case": synthetic_result["summary"],
                "completed_measured_cases": [item["summary"]["case_id"] for item in measured_results],
            }
            _atomic_json(resolved / "manifest.json", progress)

        completed = datetime.now(timezone.utc).isoformat()
        validate_config(config)
        cases = [synthetic_result["summary"], *(item["summary"] for item in measured_results)]
        endpoint_by_case = primary_endpoint_values(cases)
        endpoint_supported = bool(all(endpoint_by_case.values()))
        summary = {
            "schema": SCHEMA,
            "status": "exploratory_diagnostic_complete",
            "completion_status": "complete",
            "scope": "fit_only_measured_development_diagnostic",
            "started_at": started,
            "completed_at": completed,
            "qualification_eligible": False,
            "decision_eligibility": False,
            "parent_run_id": config["sources"]["parent_run_id"],
            "estimand_id": config["estimand"]["estimand_id"],
            "operator": config["estimand"]["operator"],
            "null": config["estimand"]["null"],
            "hypothesis": config["estimand"]["hypothesis"],
            "primary_endpoint": {
                "id": config["estimand"]["primary_endpoint"],
                "supported_in_all_cases": endpoint_supported,
                "value_by_case": endpoint_by_case,
            },
            "scientific_verdict": (
                "primary_practical_identifiability_hypothesis_supported_exploratory_only"
                if endpoint_supported
                else "primary_practical_identifiability_hypothesis_not_supported"
            ),
            "claim_boundary": "does not qualify a physical teacher or authorize validation, protected evaluation, or tokenizer promotion",
            "data_boundary": measured_counts,
            "data_identity": data_identity,
            "representative_selection": selection_rows,
            "cases": cases,
            "synthetic_parameter_recovery": rows["recovery"],
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pyyaml": yaml.__version__,
            },
        }
        _atomic_json(resolved / "calibration.json", calibration)
        _atomic_json(resolved / "data_contracts.json", {
            "data_identity": summary["data_identity"],
            "input_hash_owner": "manifest.json",
            "loader_contracts": contracts,
            "trial_inventory": "trial_inventory.csv",
        })
        _atomic_json(resolved / "summary.json", summary)
        _atomic_write(resolved / "summary.md", _markdown_report(summary))
        launch_config = Path(config_path).resolve() if config_path is not None else resolved / "resolved_config.yaml"
        parent_manifest = REPO_ROOT / str(config["sources"]["parent_manifest"])
        source_paths = {
            "resolved_config": resolved / "resolved_config.yaml",
            "measured_config": REPO_ROOT / str(config["sources"]["measured_config"]),
            "synthetic_config": REPO_ROOT / str(config["sources"]["synthetic_config"]),
            "runner": Path(__file__).resolve(),
            "measured_runner": REPO_ROOT / "experiments/evaluate_t3_measured_reconstruction_null.py",
            "synthetic_runner": REPO_ROOT / "experiments/evaluate_t3a_balloon_robust_p0.py",
            "shared_loader_runner": REPO_ROOT / "experiments/evaluate_shared_neural_driver_unified.py",
            "unified_loader": REPO_ROOT / "src/data/unified_physiology.py",
            "balloon_model": REPO_ROOT / "src/inference/t3a_balloon_robust_ssm.py",
        }
        if parent_manifest.is_file():
            parent_payload = json.loads(parent_manifest.read_text(encoding="utf-8"))
            parent_provenance = {
                "path": str(config["sources"]["parent_manifest"]),
                "available": True,
                "sha256": _sha256(parent_manifest),
                "completion_status": parent_payload.get("completion_status"),
                "arrays_consumed": False,
            }
        else:
            parent_provenance = {
                "path": str(config["sources"]["parent_manifest"]),
                "available": False,
                "sha256": None,
                "completion_status": None,
                "arrays_consumed": False,
            }

        def display_path(path: Path) -> str:
            try:
                return str(path.relative_to(REPO_ROOT))
            except ValueError:
                return str(path)

        manifest = {
            **boundary,
            "status": "exploratory_diagnostic_complete",
            "completion_status": "complete",
            "stage": "complete",
            "completed_at": completed,
            "measured_counts": measured_counts,
            "cache_root": str(measured_source["data"]["cache_root"]),
            "data_identity": summary["data_identity"],
            "input_hashes": {row["path"]: row["sha256"] for row in cache_provenance},
            "loader_contracts": contracts,
            "trial_inventory": "trial_inventory.csv",
            "claim_boundary": "exploratory fit-only practical identifiability only; does not qualify a physical teacher or authorize validation, protected evaluation, or tokenizer promotion",
            "parent_manifest": parent_provenance,
            "source_schemas": {
                "identifiability": SCHEMA,
                "measured": measured_source["schema"],
                "synthetic": load_synthetic_config(REPO_ROOT / str(config["sources"]["synthetic_config"]))["schema"],
            },
            "primary_endpoint_result": summary["primary_endpoint"],
            "analyzed_measured_subjects": [item["summary"]["case_id"] for item in measured_results],
            "representative_selection": [
                {key: row[key] for key in ("role", "subject", "score")}
                for row in selection_rows if row["selected"]
            ],
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "source_paths": {name: display_path(path) for name, path in source_paths.items()},
            "launch_config_path": display_path(launch_config),
            "git": _git_payload(),
            "runtime": summary["runtime"],
            "artifacts": [
                "resolved_config.yaml",
                "manifest.json",
                "summary.json",
                "summary.md",
                "calibration.json",
                "data_contracts.json",
                "representative_selection.csv",
                "trial_inventory.csv",
                "multistart_results.csv",
                "profile_likelihood.csv",
                "expanded_bounds.csv",
                "sensitivity_svd.csv",
                "state_stability.csv",
                "parameter_recovery.csv",
            ],
        }
        _atomic_json(resolved / "manifest.json", manifest)
        return summary
    except Exception as exc:
        _write_result_tables(resolved, rows)
        failure = {
            **progress,
            "status": "incomplete_failed",
            "completion_status": "incomplete",
            "failed_after_stage": progress["stage"],
            "stage": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
            "partial_artifacts": [
                "resolved_config.yaml",
                "manifest.json",
                "summary.json",
                "summary.md",
                "representative_selection.csv",
                "trial_inventory.csv",
                "multistart_results.csv",
                "profile_likelihood.csv",
                "expanded_bounds.csv",
                "sensitivity_svd.csv",
                "state_stability.csv",
                "parameter_recovery.csv",
            ],
        }
        _atomic_json(resolved / "manifest.json", failure)
        _atomic_json(resolved / "summary.json", failure)
        _atomic_write(
            resolved / "summary.md",
            "\n".join([
                "# T3 第二步：未完成实验报告",
                "",
                f"- 状态：**incomplete**（阶段 `{failure['stage']}`）。",
                f"- 失败类型：`{failure['error_type']}`。",
                f"- 错误：{failure['error']}",
                "- 边界：仅拟合集探索；validation/protected 数组未解引用；不具资格或决策效力。",
                "- `manifest.json` 保留失败时的最后阶段、输入 hash、边界与 traceback；CSV 保留已完成的部分结果。",
                "",
            ]),
        )
        raise


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_step2_v1")
        run_dir = REPO_ROOT / str(config["output"]["root"]) / stamp
    elif not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run(config, run_dir, config_path=args.config)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
