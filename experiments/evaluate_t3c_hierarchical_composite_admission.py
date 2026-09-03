#!/usr/bin/env python3
"""Step 4 admission gate for low-dimensional hierarchical composites.

This gate is deliberately array-free.  It checks frozen Step 2/3 evidence and
the analytic gain/time-scale coordinate before any measured hierarchy may run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_t3a_balloon_robust_p0 import _atomic_csv, _atomic_json, _atomic_write


SCHEMA = "t3c_hierarchical_composite_admission_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission_v1.yaml"
OUTPUT_ROOT = "experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission"
COORDINATES = ("log_gain", "log_time", "logit_zeta", "log_tv")


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


def _positive(raw: Mapping[str, float], keys: Sequence[str]) -> None:
    if any(not np.isfinite(float(raw.get(key, np.nan))) or float(raw[key]) <= 0.0 for key in keys):
        raise ValueError(f"strictly positive finite values required: {tuple(keys)}")


def composite_from_raw(raw: Mapping[str, float]) -> dict[str, float]:
    """Map primitive Balloon parameters to the plan's absolute composites."""

    _positive(raw, ("beta", "kappa", "gamma", "tau", "alpha"))
    gamma = float(raw["gamma"])
    time = 1.0 / math.sqrt(gamma)
    zeta = float(raw["kappa"]) * time / 2.0
    if not 0.0 < zeta < 1.0:
        raise ValueError("logit_zeta requires 0 < zeta < 1")
    return {
        "log_gain": math.log(float(raw["beta"]) / gamma),
        "log_time": math.log(time),
        "logit_zeta": math.log(zeta / (1.0 - zeta)),
        "log_tv": math.log(float(raw["tau"]) * float(raw["alpha"])),
    }


def raw_from_composite(
    phi: Mapping[str, float],
    *,
    alpha_gauge: float,
    raw_bounds: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, float]:
    """Invert the four composites after explicitly fixing the alpha gauge."""

    if not np.isfinite(alpha_gauge) or alpha_gauge <= 0.0:
        raise ValueError("a positive finite alpha_gauge is required")
    if set(phi) != set(COORDINATES) or any(not np.isfinite(float(phi[key])) for key in COORDINATES):
        raise ValueError(f"phi must contain exactly {COORDINATES} with finite values")
    gain = math.exp(float(phi["log_gain"]))
    time_scale = math.exp(float(phi["log_time"]))
    logit_zeta = float(phi["logit_zeta"])
    zeta = 1.0 / (1.0 + math.exp(-logit_zeta)) if logit_zeta >= 0.0 else math.exp(logit_zeta) / (1.0 + math.exp(logit_zeta))
    tv = math.exp(float(phi["log_tv"]))
    raw = {
        "beta": gain / (time_scale * time_scale),
        "kappa": 2.0 * zeta / time_scale,
        "gamma": 1.0 / (time_scale * time_scale),
        "tau": tv / float(alpha_gauge),
        "alpha": float(alpha_gauge),
    }
    _positive(raw, tuple(raw))
    if raw_bounds is not None:
        for key, bounds in raw_bounds.items():
            if key not in raw or len(bounds) != 2:
                continue
            lower, upper = map(float, bounds)
            if not lower <= raw[key] <= upper:
                raise ValueError(f"induced primitive {key}={raw[key]:.9g} is outside [{lower}, {upper}]")
    return raw


def normal_normal_partial_pool(
    local: np.ndarray,
    variance: np.ndarray,
    *,
    population_mean: float,
    population_variance: float,
) -> dict[str, np.ndarray]:
    """Closed-form scalar Normal--Normal shrinkage software primitive."""

    estimates = np.asarray(local, dtype=np.float64)
    measurement_variance = np.asarray(variance, dtype=np.float64)
    if estimates.shape != measurement_variance.shape or estimates.ndim != 1 or not len(estimates):
        raise ValueError("local and variance must be non-empty equal-length vectors")
    if not np.all(np.isfinite(estimates)) or not np.all(np.isfinite(measurement_variance)) or np.any(measurement_variance <= 0.0):
        raise ValueError("local estimates must be finite and variances strictly positive")
    if not np.isfinite(population_mean) or not np.isfinite(population_variance) or population_variance < 0.0:
        raise ValueError("population mean/variance are invalid")
    if population_variance == 0.0:
        weight = np.zeros_like(estimates)
        posterior_variance = np.zeros_like(estimates)
    else:
        weight = population_variance / (population_variance + measurement_variance)
        posterior_variance = population_variance * measurement_variance / (population_variance + measurement_variance)
    posterior_mean = float(population_mean) + weight * (estimates - float(population_mean))
    return {
        "posterior_mean": posterior_mean,
        "posterior_variance": posterior_variance,
        "shrinkage_weight": weight,
    }


def _source_items(config: Mapping[str, Any]) -> dict[str, tuple[Path, str]]:
    result: dict[str, tuple[Path, str]] = {}
    for name, item in config["sources"].items():
        path = REPO_ROOT / str(item["path"])
        result[str(name)] = (path, str(item["sha256"]))
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ValueError("Step 4 admission schema mismatch")
    experiment = config.get("experiment", {})
    expected = {
        "name": SCHEMA,
        "scope": "preflight_only_no_measured_arrays",
        "measured_data_enabled": False,
        "validation_data_enabled": False,
        "protected_data_enabled": False,
        "qualification_eligible": False,
        "decision_eligibility": False,
        "seed": 20260903,
    }
    if not isinstance(experiment, Mapping) or any(experiment.get(key) != value for key, value in expected.items()):
        raise ValueError("Step 4 admission boundary mismatch")
    if set(config.get("sources", {})) != {
        "step2_manifest", "step2_summary", "step3_manifest", "step3_summary", "step3_fold_calibration"
    }:
        raise ValueError("Step 4 frozen evidence registry is incomplete")
    for name, (path, expected_digest) in _source_items(config).items():
        if not path.is_file() or _sha256(path) != expected_digest:
            raise ValueError(f"frozen source evidence mismatch: {name}")
    composite = config.get("composite", {})
    if tuple(composite.get("coordinates", ())) != COORDINATES:
        raise ValueError("composite coordinate contract mismatch")
    if tuple(composite.get("candidate_ladder", ())) != ("C0_fixed", "C1_G", "C1_T", "C2_GT"):
        raise ValueError("candidate ladder mismatch")
    if tuple(composite.get("random_effect_coordinates", ())) != ("gain", "time"):
        raise ValueError("only gain and time may be proposed as random effects")
    if tuple(composite.get("deferred_coordinates", ())) != ("zeta", "tv", "alpha", "E0"):
        raise ValueError("deferred coordinate contract mismatch")
    reference = composite.get("reference", {})
    _positive(reference, ("beta", "kappa", "gamma", "tau", "alpha"))
    if not 0.0 < float(reference.get("E0", np.nan)) < 1.0:
        raise ValueError("reference E0 must lie in (0, 1)")
    roundtrip = raw_from_composite(
        composite_from_raw(reference), alpha_gauge=float(reference["alpha"]), raw_bounds=composite.get("raw_bounds")
    )
    if any(not math.isclose(roundtrip[key], float(reference[key]), rel_tol=1e-12, abs_tol=1e-12) for key in roundtrip):
        raise ValueError("composite reference does not round-trip")
    hierarchy = config.get("hierarchy", {})
    if (
        hierarchy.get("family") != "independent_normal_normal_partial_pooling"
        or hierarchy.get("full_covariance_enabled") is not False
        or int(hierarchy.get("maximum_random_effect_dimensions", 0)) != 2
    ):
        raise ValueError("hierarchy must remain one/two-dimensional diagonal partial pooling")
    required = tuple(config.get("admission", {}).get("required_checks", ()))
    if len(required) != 8 or len(set(required)) != len(required):
        raise ValueError("admission checks must be the eight unique frozen requirements")
    if config.get("admission", {}).get("on_unmet") != "stop_before_measured_metadata_or_array_access":
        raise ValueError("admission failure must stop before measured access")
    if config.get("output", {}).get("root") != OUTPUT_ROOT:
        raise ValueError(f"output.root must remain {OUTPUT_ROOT}")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError("configuration must be a mapping")
    config = dict(value)
    validate_config(config)
    return config


def _gauge_fingerprint(calibration: Mapping[str, Any]) -> str:
    payload = {
        "hbo": calibration.get("selected_hbo_channels"),
        "hbr": calibration.get("selected_hbr_channels"),
        "eeg_indices": calibration.get("eeg_adapter", {}).get("indices"),
        "eeg_loading": calibration.get("eeg_adapter", {}).get("loading"),
        "eeg_pc_scale": calibration.get("eeg_adapter", {}).get("pc_scale"),
        "observation": calibration.get("observation_calibration"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evaluate_admission(
    config: Mapping[str, Any],
    step2_summary: Mapping[str, Any],
    step3_summary: Mapping[str, Any],
    fold_calibration: Mapping[str, Any],
    *,
    step2_manifest: Mapping[str, Any] | None = None,
    step3_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen prerequisites without accepting post-hoc substitutes."""

    validate_config(config)
    folds = list(fold_calibration.get("folds", ()))
    endpoints = {
        (tuple(map(str, item.get("selected_hbo_channels", ()))), tuple(map(str, item.get("selected_hbr_channels", ()))))
        for item in folds
    }
    fingerprints = {_gauge_fingerprint(item) for item in folds}
    step2_supported = step2_summary.get("primary_endpoint", {}).get("supported_in_all_cases") is True
    parameter_screen_failed = step3_summary.get("parameter_stability", {}).get("screen_passed") is False
    cross_session_gauge_invariant = step3_summary.get("driver_stability", {}).get("cross_session_descriptive", {}).get("gauge_invariant") is True
    source_preflight = step3_summary.get("synthetic_preflight", {})
    source_preflight_is_formal = all(
        source_preflight.get(key) is True
        for key in ("simulation_based_calibration_passed", "profile_passed", "multistart_recovery_passed")
    )
    step2_manifest = step2_manifest or {}
    step3_manifest = step3_manifest or {}
    step3_boundary = step3_manifest.get("boundary", {})
    expected_validation = [f"subject_{index:02d}" for index in range(19, 24)]
    expected_protected = [f"subject_{index:02d}" for index in range(24, 30)]
    closed = bool(
        step2_manifest.get("validation_data_opened") is False
        and step2_manifest.get("protected_data_opened") is False
        and step2_manifest.get("validation_subject_arrays_not_dereferenced") == expected_validation
        and step2_manifest.get("protected_subject_arrays_not_dereferenced") == expected_protected
        and step3_boundary.get("validation_data_opened") is False
        and step3_boundary.get("protected_data_opened") is False
        and int(step3_boundary.get("validation_subject_array_access_count", -1)) == 0
        and int(step3_boundary.get("protected_subject_array_access_count", -1)) == 0
        and step3_boundary.get("validation_subjects_closed") == expected_validation
        and step3_boundary.get("protected_subjects_closed") == expected_protected
    )
    checks = [
        ("prior_runs_complete", step2_manifest.get("completion_status") == "complete" and step3_manifest.get("completion_status") == "complete",
         f"Step 2/3 completion_status={step2_manifest.get('completion_status')}/{step3_manifest.get('completion_status')}"),
        ("t_p2_identifiability_supported", step2_supported,
         f"Step 2 primary_endpoint.supported_in_all_cases={step2_supported}"),
        ("cross_subject_stability_failure_in_common_gauge", parameter_screen_failed and cross_session_gauge_invariant,
         f"parameter_screen_failed={parameter_screen_failed}; cross_session_gauge_invariant={cross_session_gauge_invariant}"),
        ("common_observation_and_driver_gauge_frozen", False,
         f"no independent Step 4 gauge registry; fold_count={len(folds)}; distinct_gauge_fingerprints={len(fingerprints)}"),
        ("fixed_local_fnirs_endpoint_frozen", False,
         f"no prospective Step 4 endpoint registry; fold_count={len(folds)}; distinct_post_hoc_endpoints={len(endpoints)}"),
        ("composite_synthetic_sbc_profile_multistart_passed", source_preflight_is_formal,
         f"Step 3 synthetic purpose={source_preflight.get('purpose', 'missing')}"),
        ("practical_margin_frozen_before_measured_scoring", False,
         "no independent composite technical-repeat margin is registered"),
        ("validation_and_protected_arrays_closed", closed,
         "Step 2/3 manifests report validation/protected arrays closed; admission runner reads no arrays"),
    ]
    rows = [
        {"requirement": name, "required": True, "met": bool(met), "evidence": evidence}
        for name, met, evidence in checks
    ]
    configured = tuple(config["admission"]["required_checks"])
    if tuple(row["requirement"] for row in rows) != configured:
        raise RuntimeError("implemented admission checks differ from the frozen config")
    all_met = all(row["met"] for row in rows)
    return {
        "decision": "ADMITTED_EXPLORATORY_MEASURED_DIAGNOSTIC" if all_met else "BLOCKED_PREREQUISITE",
        "required_met": all_met,
        "checks": rows,
        "blockers": [row["requirement"] for row in rows if not row["met"]],
        "measured_metadata_access_count": 0,
        "measured_array_access_count": 0,
        "validation_subject_array_access_count": 0,
        "protected_subject_array_access_count": 0,
        "observed_fold_endpoints": [
            {
                "fold_id": item.get("fold_id"),
                "hbo": item.get("selected_hbo_channels"),
                "hbr": item.get("selected_hbr_channels"),
                "gauge_fingerprint": _gauge_fingerprint(item),
            }
            for item in folds
        ],
    }


def _software_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    reference = {key: float(value) for key, value in config["composite"]["reference"].items()}
    phi = composite_from_raw(reference)
    recovered = raw_from_composite(
        phi,
        alpha_gauge=reference["alpha"],
        raw_bounds=config["composite"]["raw_bounds"],
    )
    pooling = normal_normal_partial_pool(
        np.asarray([-0.4, 0.0, 0.4]),
        np.asarray([0.01, 0.04, 0.16]),
        population_mean=0.0,
        population_variance=0.09,
    )
    roundtrip_error = max(abs(recovered[key] - reference[key]) for key in recovered)
    shrinkage_between = bool(np.all(np.abs(pooling["posterior_mean"]) <= np.asarray([0.4, 0.0, 0.4]) + 1e-15))
    passed = bool(roundtrip_error <= 1e-12 and shrinkage_between and np.all(np.diff(pooling["shrinkage_weight"]) < 0.0))
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "purpose": "analytic coordinate and Normal-Normal software smoke only; not SBC, profile likelihood, multistart recovery, or a practical margin",
        "reference_raw": reference,
        "reference_composite": phi,
        "roundtrip_raw": recovered,
        "maximum_roundtrip_absolute_error": roundtrip_error,
        "partial_pooling_smoke": {key: value.tolist() for key, value in pooling.items()},
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(value)


def _artifact_entry(path: Path, row_unit: str | None = None) -> dict[str, Any]:
    rows = None
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            rows = max(sum(1 for _ in handle) - 1, 0)
    return {
        "required": True,
        "present": path.is_file(),
        "row_unit": row_unit,
        "rows_data": rows,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    admission = summary["admission"]
    failed = [row for row in admission["checks"] if not row["met"]]
    lines = [
        "# T3c 第四步准入结果",
        "",
        f"判定：`{admission['decision']}`。准入检查已完成，但 measured hierarchical partial pooling 未启动。",
        "",
        "未满足条件：",
        "",
        *[f"- `{row['requirement']}`：{row['evidence']}" for row in failed],
        "",
        "综合参数解析映射与 Normal–Normal 收缩软件自检通过；它不是 SBC、profile/multistart、预测优势或 practical margin 证据。",
        "subjects 01–29 的 measured metadata/array 均未由本运行读取；validation/protected 数组访问为 0。",
        "",
    ]
    return "\n".join(lines)


def run(config: Mapping[str, Any], run_dir: Path, *, config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    validate_config(config)
    resolved = Path(run_dir)
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"run directory must be new or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    resolved_config = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
    _atomic_write(resolved / "resolved_config.yaml", resolved_config)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "incomplete",
        "run_state": "initial",
        "completion_status": "incomplete",
        "stage": "before_evidence_read",
        "started_at": started_at,
        "config_path": str(Path(config_path).resolve().relative_to(REPO_ROOT)),
        "config_sha256": _sha256(Path(config_path).resolve()),
        "runner_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "resolved_config_sha256": _sha256(resolved / "resolved_config.yaml"),
        "boundary": {
            "scope": "preflight_only_no_measured_arrays",
            "measured_data_enabled": False,
            "measured_metadata_opened": False,
            "measured_arrays_opened": False,
            "validation_data_enabled": False,
            "protected_data_enabled": False,
            "validation_subject_array_access_count": 0,
            "protected_subject_array_access_count": 0,
            "qualification_eligible": False,
            "decision_eligibility": False,
        },
    }
    _atomic_json(resolved / "manifest.json", manifest)
    try:
        source_items = _source_items(config)
        evidence = {name: _load_json(path) for name, (path, _) in source_items.items()}
        source_hashes = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path, _ in source_items.values()}
        preflight = _software_preflight(config)
        if not preflight["passed"]:
            raise RuntimeError("composite software preflight failed")
        _atomic_json(resolved / "software_preflight.json", preflight)
        admission = evaluate_admission(
            config,
            evidence["step2_summary"],
            evidence["step3_summary"],
            evidence["step3_fold_calibration"],
            step2_manifest=evidence["step2_manifest"],
            step3_manifest=evidence["step3_manifest"],
        )
        _atomic_csv(resolved / "admission_checks.csv", admission["checks"])
        _atomic_json(resolved / "composite_contract.json", {
            "schema": SCHEMA,
            "coordinates": config["composite"]["coordinates"],
            "definitions": config["composite"]["definitions"],
            "candidate_ladder": config["composite"]["candidate_ladder"],
            "random_effect_coordinates": config["composite"]["random_effect_coordinates"],
            "fixed_coordinate_gauge": config["composite"]["fixed_coordinate_gauge"],
            "reference": config["composite"]["reference"],
            "raw_bounds": config["composite"]["raw_bounds"],
            "hierarchy": config["hierarchy"],
            "claim_boundary": "method contract only; no parameter identifiability, trait, teacher, or tokenizer claim",
        })
        completed_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "schema": SCHEMA,
            "analysis_kind": "step4_hierarchical_composite_admission_only",
            "status": "admission_check_complete",
            "completion_status": "complete",
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_seconds": time.perf_counter() - start_clock,
            "software_preflight": preflight,
            "admission": admission,
            "measured_hierarchical_arm_state": "not_started",
            "scientific_verdict": "hierarchical_measured_arm_blocked_by_unmet_prerequisites",
            "claim_boundary": "array-free admission evidence only; no measured fit, trait, qualification, teacher, or tokenizer promotion claim",
        }
        _atomic_json(resolved / "summary.json", summary)
        _atomic_write(resolved / "summary.md", _markdown(summary))
        artifacts = {
            "resolved_config.yaml": _artifact_entry(resolved / "resolved_config.yaml"),
            "software_preflight.json": _artifact_entry(resolved / "software_preflight.json"),
            "admission_checks.csv": _artifact_entry(resolved / "admission_checks.csv", "admission_requirement"),
            "composite_contract.json": _artifact_entry(resolved / "composite_contract.json"),
            "summary.json": _artifact_entry(resolved / "summary.json"),
            "summary.md": _artifact_entry(resolved / "summary.md"),
        }
        manifest = {
            **manifest,
            "status": "admission_check_complete",
            "run_state": "complete",
            "completion_status": "complete",
            "stage": "complete",
            "updated_at": completed_at,
            "completed_at": completed_at,
            "elapsed_seconds": summary["elapsed_seconds"],
            "admission_decision": admission["decision"],
            "required_met": admission["required_met"],
            "measured_hierarchical_arm_state": "not_started",
            "scientific_verdict": summary["scientific_verdict"],
            "source_hashes": source_hashes,
            "artifacts": artifacts,
            "summary_pointer": "summary.json",
            "git": _git_payload(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        }
        _atomic_json(resolved / "manifest.json", manifest)
        print(json.dumps({"stage": "complete", "admission_decision": admission["decision"], "run_dir": str(resolved)}), flush=True)
        return summary
    except Exception as exc:
        failed_at = datetime.now(timezone.utc).isoformat()
        _atomic_json(resolved / "manifest.json", {
            **manifest,
            "status": "incomplete_failed",
            "run_state": "failure",
            "completion_status": "incomplete",
            "stage": "failed",
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
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_step4_admission_v1")
        run_dir = REPO_ROOT / str(config["output"]["root"]) / stamp
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    run(config, run_dir, config_path=config_path)
    return run_dir


if __name__ == "__main__":
    main()
