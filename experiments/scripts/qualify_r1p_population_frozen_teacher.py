#!/usr/bin/env python3
"""Qualify an R1-P population-frozen teacher without validation tuning.

The evaluator deliberately has two phases.  It first loads only subjects
01-18, freezes all templates, scales, ridge hyperparameters, null thresholds,
and decision thresholds, and writes ``threshold_manifest.json``.  Only then
does it dereference subjects 19-23 for pure application.  Subjects 24-29 are
rejected before measured-array access.

Physical reconstruction is recomputed from the serialized parameter bundle
because the trajectory sidecar contains rJ/rE but not observed/reconstructed
HbO/HbR.  No parameter, gauge, feature scale, or threshold is refit on
validation data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from threadpoolctl import threadpool_limits


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_adaptive_shared_neural_ssm import (  # noqa: E402
    _apply_eeg_adapter,
    _chromophore_targets,
)
from experiments.scripts.build_r1p_population_frozen_teacher import (  # noqa: E402
    PARAMETER_MANIFEST_SCHEMA,
    PROTECTED_SUBJECT_IDS,
    PopulationFrozenBundle,
    PopulationTrial,
    _indices_by_name,
    load_population_bundle,
    load_registered_trials,
    validate_population_config,
)
from src.inference.adaptive_neurovascular_ssm import apply_adaptive_ssm  # noqa: E402


SCHEMA = "r1p_population_frozen_teacher_qualification_v1"
THRESHOLD_SCHEMA = "r1p_teacher_frozen_thresholds_v1"
DEFAULT_REGISTRY = (
    REPO_ROOT
    / "experiments/configs/physiology_semantic_tokenizer/"
    "r1p_teacher_qualification_registry.json"
)
DEFAULT_PERTURBATION_REGISTRY = (
    REPO_ROOT
    / "experiments/configs/physiology_semantic_tokenizer/"
    "r1p_teacher_perturbation_registry.json"
)
DEFAULT_PREVALIDATION_SEAL = (
    REPO_ROOT
    / "docs/physiology_semantic_tokenizer/architecture/"
    "r1p_prevalidation_seal.json"
)
MECHANICAL_AMENDMENT_SEAL_STATUS = (
    "sealed_mechanical_serialization_amendment_after_uninspected_validation_compute"
)
MECHANICAL_AMENDMENT_KIND = "numpy_json_serialization_only"
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#777777",
}


@dataclass(frozen=True)
class PanelRecord:
    sample_key: str
    subject: str
    subject_key: str
    split: str
    session: str
    condition: str
    event_index: int
    rj: np.ndarray
    rj_masked: np.ndarray
    re: np.ndarray
    hbo_observed: np.ndarray
    hbr_observed: np.ndarray
    hbo_joint: np.ndarray
    hbr_joint: np.ndarray
    hbo_eeg_only: np.ndarray
    hbr_eeg_only: np.ndarray
    eeg_features: np.ndarray
    fnirs_features: np.ndarray


@dataclass(frozen=True)
class RidgeModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        return standardized @ self.coefficients + self.intercept


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Convert NumPy containers/scalars without changing their values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "r1p_teacher_qualification_registry_v1":
        raise ValueError("R1-P qualification registry schema mismatch")
    if payload.get("status") != "frozen_before_formal_validation_results":
        raise ValueError("Qualification registry is not declared pre-validation frozen")
    if int(payload.get("revision", -1)) != 4:
        raise ValueError("Qualification registry revision mismatch")
    contract = payload["input_contract"]
    if bool(contract.get("protected_open", True)):
        raise ValueError("Qualification registry must keep protected_open=false")
    if set(contract["protected_subject_ids"]) != set(PROTECTED_SUBJECT_IDS):
        raise ValueError("Qualification protected-subject registry drifted")
    required = {item["gate_id"] for item in payload["primary_gates"]}
    if required != set(payload["promotion_rule"]["required_gate_ids"]):
        raise ValueError("Promotion rule and primary-gate registry disagree")
    policy = payload["threshold_policy"]
    if (
        int(policy["null_replicates"]) < 2000
        or policy.get("null_quantile_method") != "higher"
    ):
        raise ValueError("Qualification null precision contract drifted")
    return payload


def load_perturbation_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "r1p_teacher_perturbation_registry_v1":
        raise ValueError("R1-P perturbation registry schema mismatch")
    if payload.get("status") != "frozen_before_formal_validation_results":
        raise ValueError("Perturbation registry is not declared pre-validation frozen")
    if int(payload.get("revision", -1)) != 3:
        raise ValueError("Perturbation registry revision mismatch")
    if bool(payload["common_contract"].get("protected_open", True)):
        raise ValueError("Perturbation registry must keep protected_open=false")
    perturbations = payload["perturbations"]
    if len(perturbations) != 3:
        raise ValueError("G4 requires exactly three registered perturbations")
    identifiers = {item["perturbation_id"] for item in perturbations}
    outputs = {item["output_name"] for item in perturbations}
    if len(identifiers) != 3 or len(outputs) != 3:
        raise ValueError("Perturbation identifiers and output names must be unique")
    allowed = {f"subject_{index:02d}" for index in range(1, 19)}
    for item in perturbations:
        retained = set(item["retained_fit_subjects"])
        excluded = set(item["excluded_fit_subjects"])
        if retained | excluded != allowed or retained & excluded:
            raise ValueError("Perturbation train-subject partition drifted")
        if len(retained) != 15 or int(item["expected_fit_windows"]) != 900:
            raise ValueError("Each registered perturbation must fit 15 subjects/900 rows")
    return payload


def validate_prevalidation_seal_state(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "r1p_prevalidation_seal_v1":
        raise ValueError("R1-P prevalidation seal schema mismatch")
    if payload.get("status") != MECHANICAL_AMENDMENT_SEAL_STATUS:
        raise ValueError("R1-P mechanical-amendment seal status mismatch")
    disclosure = payload.get("validation_metric_disclosure")
    if disclosure != {
        "computed_in_memory": True,
        "serialized": False,
        "inspected_by_operator": False,
        "failure_point": "final_panel_summary_json_serialization",
    }:
        raise RuntimeError("Validation-metric disclosure is incomplete or changed")
    aborted = payload.get("aborted_formal_run")
    if not isinstance(aborted, Mapping) or any(
        aborted.get(key) is not expected
        for key, expected in {
            "formal_output_absent": True,
            "validation_metrics_computed_in_memory": True,
            "validation_metrics_serialized": False,
            "validation_metrics_inspected_by_operator": False,
            "temporary_output_cleaned": True,
            "partial_artifacts_survived_cleanup": False,
        }.items()
    ):
        raise RuntimeError("Aborted formal-run disclosure is incomplete or changed")
    amendment = payload.get("mechanical_amendment")
    if not isinstance(amendment, Mapping):
        raise RuntimeError("Mechanical amendment declaration is missing")
    if (
        amendment.get("kind") != MECHANICAL_AMENDMENT_KIND
        or amendment.get("registry_changed") is not False
        or amendment.get("threshold_changed") is not False
        or amendment.get("gate_changed") is not False
        or amendment.get("mathematical_path_changed") is not False
    ):
        raise RuntimeError("Seal exceeds the allowed mechanical amendment scope")


def load_prevalidation_seal(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path != DEFAULT_PREVALIDATION_SEAL.resolve():
        raise RuntimeError(
            "Qualification requires the tracked default prevalidation seal path"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_prevalidation_seal_state(payload)
    for item in payload["sealed_files"]:
        source = REPO_ROOT / str(item["path"])
        if not source.is_file() or _sha256(source) != item["sha256"]:
            raise RuntimeError(
                f"Prevalidation-sealed source changed: {item['path']}"
            )
    return payload


def verify_qualification_sealed_inputs(
    seal: Mapping[str, Any],
    *,
    config_path: Path,
    bundle_root: Path,
    registry_path: Path,
    perturbation_registry_path: Path,
) -> dict[str, Any]:
    sealed = {str(item["role"]): item for item in seal["sealed_files"]}
    checks = {
        "teacher_config": _sha256(Path(config_path).resolve())
        == sealed["teacher_config"]["sha256"],
        "qualification_registry": _sha256(Path(registry_path).resolve())
        == sealed["qualification_registry"]["sha256"],
        "perturbation_registry": _sha256(
            Path(perturbation_registry_path).resolve()
        )
        == sealed["perturbation_registry"]["sha256"],
    }
    parameter_manifest_path = (
        Path(bundle_root).resolve() / "parameter_bundle/manifest.json"
    )
    checks["base_parameter_manifest"] = (
        _sha256(parameter_manifest_path)
        == sealed["base_parameter_manifest"]["sha256"]
    )
    parameter = json.loads(parameter_manifest_path.read_text(encoding="utf-8"))
    identity = seal["base_bundle_identity"]
    checks["base_parameter_bundle_sha256"] = (
        parameter.get("bundle_sha256") == identity["parameter_bundle_sha256"]
    )
    checks["base_parameter_arrays_sha256"] = (
        parameter.get("arrays_sha256") == identity["parameter_arrays_sha256"]
    )
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(
            "Qualification CLI/base inputs differ from prevalidation seal: "
            f"{failed}"
        )
    return checks


def reverify_qualification_seal_before_validation(
    *,
    prevalidation_seal_path: Path,
    expected_prevalidation_seal_sha256: str,
    expected_input_checks: Mapping[str, Any],
    config_path: Path,
    bundle_root: Path,
    registry_path: Path,
    perturbation_registry_path: Path,
) -> dict[str, Any]:
    """Fail closed on any seal/input change since calibration began."""

    prevalidation_seal_path = Path(prevalidation_seal_path).resolve()
    if _sha256(prevalidation_seal_path) != expected_prevalidation_seal_sha256:
        raise RuntimeError("Prevalidation seal changed before validation load")
    reloaded_seal = load_prevalidation_seal(prevalidation_seal_path)
    repeated_checks = verify_qualification_sealed_inputs(
        reloaded_seal,
        config_path=config_path,
        bundle_root=bundle_root,
        registry_path=registry_path,
        perturbation_registry_path=perturbation_registry_path,
    )
    if repeated_checks != dict(expected_input_checks):
        raise RuntimeError("Sealed input checks changed before validation load")
    return repeated_checks


def _feature_indices(trial: PopulationTrial, bundle: PopulationFrozenBundle) -> tuple[np.ndarray, np.ndarray]:
    eeg_indices = _indices_by_name(
        trial.eeg_channel_names,
        bundle.adapter.channel_names,
        modality="EEG",
    )
    fnirs_indices = _indices_by_name(
        trial.fnirs_channel_names,
        bundle.selected_fnirs_channels,
        modality="fNIRS",
    )
    if trial.eeg_bad_channel_mask[eeg_indices].any():
        raise RuntimeError("Frozen EEG view is rejected in panel input")
    if trial.fnirs_bad_channel_mask[fnirs_indices].any():
        raise RuntimeError("Frozen fNIRS view is rejected in panel input")
    return eeg_indices, fnirs_indices


def _eeg_features(signal: np.ndarray) -> np.ndarray:
    """Ten patch log-power and log-difference-power values per channel."""

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != 4000:
        raise ValueError("R1-P EEG feature input must be [4000, channels]")
    patches = signal.reshape(10, 400, signal.shape[1])
    power = np.log(np.mean(np.square(patches), axis=1) + 1e-12)
    difference = np.diff(patches, axis=1)
    diff_power = np.log(np.mean(np.square(difference), axis=1) + 1e-12)
    return np.concatenate((power.reshape(-1), diff_power.reshape(-1)))


def _fnirs_features(signal: np.ndarray) -> np.ndarray:
    """Ten patch mean, standard deviation, and slope values per channel."""

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != 200:
        raise ValueError("R1-P fNIRS feature input must be [200, channels]")
    patches = signal.reshape(10, 20, signal.shape[1])
    means = np.mean(patches, axis=1)
    stds = np.std(patches, axis=1)
    centered_time = np.arange(20, dtype=np.float64) - 9.5
    denominator = float(np.sum(np.square(centered_time)))
    slopes = np.einsum("t,ptc->pc", centered_time, patches) / denominator
    return np.concatenate((means.reshape(-1), stds.reshape(-1), slopes.reshape(-1)))


def evaluate_trials(
    trials: Sequence[PopulationTrial],
    bundle: PopulationFrozenBundle,
) -> list[PanelRecord]:
    """Pure-apply a frozen bundle and retain panel diagnostics."""

    records: list[PanelRecord] = []
    mean = float(bundle.normalization["mean"])
    scale = float(bundle.normalization["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Frozen shared-driver gauge is singular")
    with threadpool_limits(limits=1):
        for trial in trials:
            if trial.subject in PROTECTED_SUBJECT_IDS:
                raise RuntimeError("Qualification cannot dereference protected arrays")
            eeg_indices, fnirs_indices = _feature_indices(trial, bundle)
            hbo_count = len(bundle.selected_hbo_indices)
            hbo_indices = fnirs_indices[:hbo_count]
            hbr_indices = fnirs_indices[hbo_count:]
            driver = _apply_eeg_adapter(trial, bundle.adapter)
            hbo, hbr = _chromophore_targets([trial], hbo_indices, hbr_indices)
            joint = apply_adaptive_ssm(
                driver,
                bundle.fit,
                hbo_observation=hbo[0],
                hbr_observation=hbr[0],
            )
            eeg_only = apply_adaptive_ssm(driver, bundle.fit)
            masked_joint_states = np.empty(200, dtype=np.float64)
            masked_hbo_reconstruction = np.empty(200, dtype=np.float64)
            masked_hbr_reconstruction = np.empty(200, dtype=np.float64)
            covered = np.zeros(200, dtype=bool)
            for parity in (0, 1):
                patch_mask = np.repeat(
                    np.arange(10, dtype=int) % 2 == parity,
                    20,
                )
                hbo_masked = np.asarray(hbo[0], dtype=np.float64).copy()
                hbr_masked = np.asarray(hbr[0], dtype=np.float64).copy()
                hbo_masked[patch_mask] = np.nan
                hbr_masked[patch_mask] = np.nan
                masked = apply_adaptive_ssm(
                    driver,
                    bundle.fit,
                    hbo_observation=hbo_masked,
                    hbr_observation=hbr_masked,
                )
                masked_joint_states[patch_mask] = masked.states[patch_mask, 4]
                masked_hbo_reconstruction[patch_mask] = (
                    masked.hbo_reconstructed[patch_mask]
                )
                masked_hbr_reconstruction[patch_mask] = (
                    masked.hbr_reconstructed[patch_mask]
                )
                covered[patch_mask] = True
            if not covered.all():
                raise RuntimeError("Patch-parity scoring did not cover all target points")
            arrays = (
                joint.states,
                masked_joint_states,
                eeg_only.states,
                hbo[0],
                hbr[0],
                masked_hbo_reconstruction,
                masked_hbr_reconstruction,
                eeg_only.hbo_reconstructed,
                eeg_only.hbr_reconstructed,
            )
            if any(np.asarray(value).shape[0] != 200 for value in arrays):
                raise RuntimeError("Qualification requires 200-point trajectories")
            if not all(np.isfinite(value).all() for value in arrays):
                raise RuntimeError("Qualification input contains non-finite values")
            records.append(
                PanelRecord(
                    sample_key=trial.sample_key,
                    subject=trial.subject,
                    subject_key=trial.subject_key,
                    split=(
                        "train"
                        if trial.development_role == "train_fit"
                        else "validation"
                    ),
                    session=trial.record_id,
                    condition=trial.condition,
                    event_index=int(trial.event_index),
                    rj=(np.asarray(joint.states[:, 4]) - mean) / scale,
                    rj_masked=(masked_joint_states - mean) / scale,
                    re=(np.asarray(eeg_only.states[:, 4]) - mean) / scale,
                    hbo_observed=np.asarray(hbo[0]),
                    hbr_observed=np.asarray(hbr[0]),
                    hbo_joint=masked_hbo_reconstruction,
                    hbr_joint=masked_hbr_reconstruction,
                    hbo_eeg_only=np.asarray(eeg_only.hbo_reconstructed),
                    hbr_eeg_only=np.asarray(eeg_only.hbr_reconstructed),
                    eeg_features=_eeg_features(trial.eeg[:, eeg_indices]),
                    fnirs_features=_fnirs_features(trial.fnirs[:, fnirs_indices]),
                )
            )
    return records


def _stack(records: Sequence[PanelRecord], field: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(record, field)) for record in records])


def _condition_templates(
    records: Sequence[PanelRecord],
    field: str,
    *,
    excluded_subject: str | None = None,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    cells = sorted({_template_cell(record) for record in records})
    for cell in cells:
        values = [
            np.asarray(getattr(record, field))
            for record in records
            if _template_cell(record) == cell and record.subject != excluded_subject
        ]
        if not values:
            raise RuntimeError(f"No training template rows for cell {cell!r}")
        output[cell] = np.mean(np.stack(values), axis=0)
    return output


def _template_cell(record: PanelRecord) -> str:
    return f"{record.session}|{record.condition}"


def _gain(observed: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> float:
    denominator = float(np.sum(np.square(observed - baseline)))
    numerator = float(np.sum(np.square(observed - prediction)))
    return 1.0 - numerator / max(denominator, 1e-12)


def physical_subject_metrics(
    records: Sequence[PanelRecord],
    templates: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in sorted({record.subject for record in records}):
        selected = [record for record in records if record.subject == subject]
        metrics: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "hbo_observed",
                "hbr_observed",
                "hbo_joint",
                "hbr_joint",
                "hbo_eeg_only",
                "hbr_eeg_only",
                "hbo_baseline",
                "hbr_baseline",
            )
        }
        for record in selected:
            for name in (
                "hbo_observed",
                "hbr_observed",
                "hbo_joint",
                "hbr_joint",
                "hbo_eeg_only",
                "hbr_eeg_only",
            ):
                metrics[name].append(np.asarray(getattr(record, name)))
            cell = _template_cell(record)
            metrics["hbo_baseline"].append(templates["hbo_observed"][cell])
            metrics["hbr_baseline"].append(templates["hbr_observed"][cell])
        arrays = {name: np.concatenate(value) for name, value in metrics.items()}
        rj = np.concatenate([record.rj_masked for record in selected])
        correction = np.concatenate(
            [record.rj_masked - record.re for record in selected]
        )
        rows.append(
            {
                "subject": subject,
                "split": selected[0].split,
                "hbo_physical_gain": _gain(
                    arrays["hbo_observed"], arrays["hbo_joint"], arrays["hbo_baseline"]
                ),
                "hbr_physical_gain": _gain(
                    arrays["hbr_observed"], arrays["hbr_joint"], arrays["hbr_baseline"]
                ),
                "hbo_jointness_gain": _gain(
                    arrays["hbo_observed"], arrays["hbo_joint"], arrays["hbo_eeg_only"]
                ),
                "hbr_jointness_gain": _gain(
                    arrays["hbr_observed"], arrays["hbr_joint"], arrays["hbr_eeg_only"]
                ),
                "correction_rms_ratio": float(
                    np.sqrt(np.mean(np.square(correction)))
                    / max(np.sqrt(np.mean(np.square(rj))), 1e-12)
                ),
                "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
            }
        )
    return rows


def physical_train_subject_metrics_loto(
    records: Sequence[PanelRecord],
) -> list[dict[str, Any]]:
    """Compute train calibration metrics with subject-excluded templates."""

    rows = []
    for subject in sorted({record.subject for record in records}):
        selected = [record for record in records if record.subject == subject]
        templates = {
            "hbo_observed": _condition_templates(
                records, "hbo_observed", excluded_subject=subject
            ),
            "hbr_observed": _condition_templates(
                records, "hbr_observed", excluded_subject=subject
            ),
        }
        rows.extend(physical_subject_metrics(selected, templates))
    return rows


def _patch_parity_mask(parity: int) -> np.ndarray:
    return np.repeat(np.arange(10, dtype=int) % 2 == int(parity), 20)


def _heldout_sse(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    scale: float,
    parity: int,
    first_difference: bool,
) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    total = 0.0
    for patch in range(int(parity), 10, 2):
        start = patch * 20
        stop = start + 20
        residual = (observed[start:stop] - predicted[start:stop]) / scale
        if first_difference:
            residual = np.diff(residual)
        total += float(np.sum(np.square(residual)))
    return total


def train_physical_scales(records: Sequence[PanelRecord]) -> dict[str, float]:
    return {
        "hbo": max(
            float(np.std(np.concatenate([record.hbo_observed for record in records]))),
            1e-12,
        ),
        "hbr": max(
            float(np.std(np.concatenate([record.hbr_observed for record in records]))),
            1e-12,
        ),
    }


def heldout_parity_subject_metrics(
    records: Sequence[PanelRecord],
    scales: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for subject in sorted({record.subject for record in records}):
        selected = [record for record in records if record.subject == subject]
        joint_level = eeg_level = joint_difference = eeg_difference = 0.0
        for record in selected:
            for parity in (0, 1):
                for modality in ("hbo", "hbr"):
                    observed = getattr(record, f"{modality}_observed")
                    joint = getattr(record, f"{modality}_joint")
                    eeg_only = getattr(record, f"{modality}_eeg_only")
                    joint_level += _heldout_sse(
                        observed,
                        joint,
                        scale=float(scales[modality]),
                        parity=parity,
                        first_difference=False,
                    )
                    eeg_level += _heldout_sse(
                        observed,
                        eeg_only,
                        scale=float(scales[modality]),
                        parity=parity,
                        first_difference=False,
                    )
                    joint_difference += _heldout_sse(
                        observed,
                        joint,
                        scale=float(scales[modality]),
                        parity=parity,
                        first_difference=True,
                    )
                    eeg_difference += _heldout_sse(
                        observed,
                        eeg_only,
                        scale=float(scales[modality]),
                        parity=parity,
                        first_difference=True,
                    )
        rows.append(
            {
                "subject": subject,
                "split": selected[0].split,
                "heldout_level_gain": 1.0
                - joint_level / max(eeg_level, 1e-12),
                "heldout_first_difference_gain": 1.0
                - joint_difference / max(eeg_difference, 1e-12),
            }
        )
    return rows


def precompute_shifted_solver_null(
    trials: Sequence[PopulationTrial],
    records: Sequence[PanelRecord],
    bundle: PopulationFrozenBundle,
    scales: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Cache nine shifted-input solver outcomes for every train row/parity."""

    if len(trials) != len(records):
        raise ValueError("Shift-null trial/record count mismatch")
    shape = (len(records), 2, 9)
    joint_level = np.empty(shape, dtype=np.float64)
    joint_difference = np.empty(shape, dtype=np.float64)
    eeg_level = np.zeros((len(records), 2), dtype=np.float64)
    eeg_difference = np.zeros((len(records), 2), dtype=np.float64)
    with threadpool_limits(limits=1):
        for row_index, (trial, record) in enumerate(zip(trials, records)):
            eeg_indices, fnirs_indices = _feature_indices(trial, bundle)
            del eeg_indices
            hbo_count = len(bundle.selected_hbo_indices)
            hbo_indices = fnirs_indices[:hbo_count]
            hbr_indices = fnirs_indices[hbo_count:]
            driver = _apply_eeg_adapter(trial, bundle.adapter)
            hbo, hbr = _chromophore_targets([trial], hbo_indices, hbr_indices)
            observations = {"hbo": hbo[0], "hbr": hbr[0]}
            for parity in (0, 1):
                for modality in ("hbo", "hbr"):
                    eeg_level[row_index, parity] += _heldout_sse(
                        observations[modality],
                        getattr(record, f"{modality}_eeg_only"),
                        scale=float(scales[modality]),
                        parity=parity,
                        first_difference=False,
                    )
                    eeg_difference[row_index, parity] += _heldout_sse(
                        observations[modality],
                        getattr(record, f"{modality}_eeg_only"),
                        scale=float(scales[modality]),
                        parity=parity,
                        first_difference=True,
                    )
                mask = _patch_parity_mask(parity)
                for shift_index, patch_shift in enumerate(range(1, 10)):
                    shifted_hbo = np.roll(hbo[0], patch_shift * 20).copy()
                    shifted_hbr = np.roll(hbr[0], patch_shift * 20).copy()
                    shifted_hbo[mask] = np.nan
                    shifted_hbr[mask] = np.nan
                    result = apply_adaptive_ssm(
                        driver,
                        bundle.fit,
                        hbo_observation=shifted_hbo,
                        hbr_observation=shifted_hbr,
                    )
                    level_sse = difference_sse = 0.0
                    for modality in ("hbo", "hbr"):
                        prediction = getattr(result, f"{modality}_reconstructed")
                        level_sse += _heldout_sse(
                            observations[modality],
                            prediction,
                            scale=float(scales[modality]),
                            parity=parity,
                            first_difference=False,
                        )
                        difference_sse += _heldout_sse(
                            observations[modality],
                            prediction,
                            scale=float(scales[modality]),
                            parity=parity,
                            first_difference=True,
                        )
                    joint_level[row_index, parity, shift_index] = level_sse
                    joint_difference[row_index, parity, shift_index] = difference_sse
    return {
        "joint_level_sse": joint_level,
        "joint_first_difference_sse": joint_difference,
        "eeg_level_sse": eeg_level,
        "eeg_first_difference_sse": eeg_difference,
        "shift_patches": np.arange(1, 10, dtype=np.int64),
    }


def shifted_solver_null_draws(
    records: Sequence[PanelRecord],
    cache: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Aggregate cached solver outcomes under group-common random shifts."""

    groups = sorted(
        {
            f"{record.subject}|{record.session}|{record.condition}"
            for record in records
        }
    )
    group_lookup = {value: index for index, value in enumerate(groups)}
    record_groups = np.asarray(
        [
            group_lookup[f"{record.subject}|{record.session}|{record.condition}"]
            for record in records
        ],
        dtype=np.int64,
    )
    subjects = sorted({record.subject for record in records})
    subject_rows = {
        subject: np.asarray(
            [index for index, record in enumerate(records) if record.subject == subject],
            dtype=np.int64,
        )
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    shift_draws = rng.integers(
        0,
        9,
        size=(int(replicates), len(groups)),
        dtype=np.int16,
    )
    level_draws = np.empty(int(replicates), dtype=np.float64)
    difference_draws = np.empty(int(replicates), dtype=np.float64)
    joint_level = np.asarray(cache["joint_level_sse"])
    joint_difference = np.asarray(cache["joint_first_difference_sse"])
    eeg_level = np.asarray(cache["eeg_level_sse"])
    eeg_difference = np.asarray(cache["eeg_first_difference_sse"])
    for draw_index, choices in enumerate(shift_draws):
        level_subject = []
        difference_subject = []
        for subject in subjects:
            rows = subject_rows[subject]
            selected_shift = choices[record_groups[rows]]
            selected_joint_level = 0.0
            selected_joint_difference = 0.0
            for local_index, row in enumerate(rows):
                for parity in (0, 1):
                    shift = int(selected_shift[local_index])
                    selected_joint_level += joint_level[row, parity, shift]
                    selected_joint_difference += joint_difference[row, parity, shift]
            level_subject.append(
                1.0
                - selected_joint_level / max(float(np.sum(eeg_level[rows])), 1e-12)
            )
            difference_subject.append(
                1.0
                - selected_joint_difference
                / max(float(np.sum(eeg_difference[rows])), 1e-12)
            )
        level_draws[draw_index] = float(np.mean(level_subject))
        difference_draws[draw_index] = float(np.mean(difference_subject))
    return {
        "shift_draws": shift_draws,
        "group_names": np.asarray(groups, dtype=np.str_),
        "level_gain_draws": level_draws,
        "first_difference_gain_draws": difference_draws,
    }


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> RidgeModel:
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
    standardized = (features - feature_mean) / feature_scale
    intercept = np.mean(target, axis=0)
    centered = target - intercept
    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        standardized.T @ centered,
    )
    return RidgeModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        intercept=intercept,
        alpha=float(alpha),
    )


def _subject_delta_r2(
    records: Sequence[PanelRecord],
    predictions: np.ndarray,
    templates: Mapping[str, np.ndarray],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for subject in sorted({record.subject for record in records}):
        indices = [
            index for index, record in enumerate(records) if record.subject == subject
        ]
        observed = np.stack([records[index].rj for index in indices])
        predicted = predictions[indices]
        baseline = np.stack(
            [templates[_template_cell(records[index])] for index in indices]
        )
        baseline_sse = float(np.sum(np.square(observed - baseline)))
        model_sse = float(np.sum(np.square(observed - predicted)))
        output[subject] = (baseline_sse - model_sse) / max(baseline_sse, 1e-12)
    return output


def _subject_delta_r2_loto(
    records: Sequence[PanelRecord],
    predictions: np.ndarray,
) -> dict[str, float]:
    """Score train CV predictions against subject-excluded phase templates."""

    output = {}
    for subject in sorted({record.subject for record in records}):
        indices = [
            index for index, record in enumerate(records) if record.subject == subject
        ]
        templates = _condition_templates(records, "rj", excluded_subject=subject)
        observed = np.stack([records[index].rj for index in indices])
        predicted = predictions[indices]
        baseline = np.stack(
            [templates[_template_cell(records[index])] for index in indices]
        )
        baseline_sse = float(np.sum(np.square(observed - baseline)))
        model_sse = float(np.sum(np.square(observed - predicted)))
        output[subject] = (baseline_sse - model_sse) / max(baseline_sse, 1e-12)
    return output


def fit_observability(
    records: Sequence[PanelRecord],
    modality: str,
    *,
    null_replicates: int,
    null_quantile: float,
    seed: int,
) -> tuple[RidgeModel, dict[str, Any], dict[str, float]]:
    feature_field = f"{modality}_features"
    features = _stack(records, feature_field)
    target = _stack(records, "rj")
    subjects = sorted({record.subject for record in records})
    templates = _condition_templates(records, "rj")
    best: tuple[float, float, np.ndarray, dict[str, float]] | None = None
    for alpha in ALPHAS:
        prediction = np.empty_like(target)
        for subject in subjects:
            train_indices = [
                index for index, record in enumerate(records) if record.subject != subject
            ]
            held_indices = [
                index for index, record in enumerate(records) if record.subject == subject
            ]
            model = _fit_ridge(features[train_indices], target[train_indices], alpha)
            prediction[held_indices] = model.predict(features[held_indices])
        subject_scores = _subject_delta_r2_loto(records, prediction)
        score = float(np.mean(list(subject_scores.values())))
        if best is None or score > best[0]:
            best = (score, float(alpha), prediction.copy(), subject_scores)
    assert best is not None
    cv_score, alpha, cv_prediction, subject_scores = best
    null = _optimized_loso_target_permutation_null(
        features,
        target,
        records,
        alphas=ALPHAS,
        replicates=int(null_replicates),
        seed=seed,
    )
    null_threshold = float(
        np.quantile(null, null_quantile, method="higher")
    )
    model = _fit_ridge(features, target, alpha)
    metadata = {
        "alpha": alpha,
        "alpha_candidates": list(ALPHAS),
        "alpha_selection": "leave_one_train_subject_out",
        "train_cv_subject_equal_delta_r2": cv_score,
        "train_subject_block_target_permutation_q": null_threshold,
        "null_crossfit_refit": True,
        "null_crossfit_fold_count": len(subjects),
        "null_repeats_full_alpha_selection": True,
        "null_held_subject_targets_excluded_from_each_fit": True,
        "subject_block_alignment": (
            "session_condition_sorted_raw_event_to_within_cell_rank_v1"
        ),
        "raw_event_ids_may_differ_across_subjects": True,
        "null_quantile": null_quantile,
        "null_replicates": int(null_replicates),
        "null_quantile_method": "higher",
        "feature_dimension": int(features.shape[1]),
        "target_dimension": int(target.shape[1]),
    }
    metadata["_null_draws"] = np.asarray(null, dtype=np.float64)
    return model, metadata, subject_scores


def _balanced_subject_indices(
    records: Sequence[PanelRecord],
) -> dict[str, np.ndarray]:
    """Return subject blocks in session/condition/within-cell-rank order.

    Raw event identifiers are only required to be unique inside each
    subject/session/condition cell.  They may differ across subjects.  Sorting
    those identifiers defines a zero-based within-cell rank, which is the
    exchangeable row identity used by the G5 subject-block permutation.
    """

    subjects = sorted({record.subject for record in records})
    if not subjects:
        raise RuntimeError("Subject-block permutation requires at least one subject")

    output: dict[str, np.ndarray] = {}
    reference_signature: list[tuple[str, str, int, tuple[int, ...]]] | None = None
    for subject in subjects:
        cells: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for index, record in enumerate(records):
            if record.subject != subject:
                continue
            cell = (record.session, record.condition)
            cells.setdefault(cell, []).append((int(record.event_index), index))

        canonical_indices: list[int] = []
        signature: list[tuple[str, str, int, tuple[int, ...]]] = []
        for (session, condition), event_rows in sorted(cells.items()):
            event_ids = [event_id for event_id, _ in event_rows]
            if len(event_ids) != len(set(event_ids)):
                raise RuntimeError(
                    "Subject-block permutation requires unique raw event IDs "
                    "within every subject/session/condition cell"
                )
            for within_cell_rank, (_, index) in enumerate(sorted(event_rows)):
                canonical_indices.append(index)
                signature.append(
                    (
                        session,
                        condition,
                        within_cell_rank,
                        tuple(np.asarray(records[index].rj).shape),
                    )
                )

        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise RuntimeError(
                "Subject-block permutation requires aligned "
                "session/condition/within-cell-rank/time order"
            )
        output[subject] = np.asarray(canonical_indices, dtype=int)
    return output


def _optimized_loso_target_permutation_null(
    features: np.ndarray,
    target: np.ndarray,
    records: Sequence[PanelRecord],
    *,
    alphas: Sequence[float],
    replicates: int,
    seed: int,
    batch_size: int = 100,
) -> np.ndarray:
    """Exact LOSO/refit null via cached ridge linear operators.

    For a held subject and alpha, ridge prediction is linear in the training
    targets.  Subject-block permutation preserves the training-target mean,
    so the feature scaler, intercept, and ridge influence matrix are constant
    across draws.  Caching the destination-block × donor-block contributions
    is therefore mathematically identical to refitting every permuted ridge.
    """

    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    subjects = sorted({record.subject for record in records})
    indices = _balanced_subject_indices(records)
    rng = np.random.default_rng(seed)
    score_cube = np.empty(
        (int(replicates), len(subjects), len(alphas)),
        dtype=np.float64,
    )
    for held_position, held_subject in enumerate(subjects):
        training_subjects = [
            subject for subject in subjects if subject != held_subject
        ]
        training_indices = np.concatenate(
            [indices[subject] for subject in training_subjects]
        )
        held_indices = indices[held_subject]
        rows_per_subject = len(held_indices)
        permutations = np.stack(
            [
                rng.permutation(len(training_subjects))
                for _ in range(int(replicates))
            ]
        )
        x_train = features[training_indices]
        feature_mean = np.mean(x_train, axis=0)
        feature_scale = np.std(x_train, axis=0)
        feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
        z_train = (x_train - feature_mean) / feature_scale
        z_held = (features[held_indices] - feature_mean) / feature_scale
        target_mean = np.mean(target[training_indices], axis=0)
        donor_targets = np.stack(
            [target[indices[subject]] - target_mean for subject in training_subjects]
        )
        templates = _condition_templates(
            records, "rj", excluded_subject=held_subject
        )
        held_target = target[held_indices]
        held_baseline = np.stack(
            [templates[_template_cell(records[index])] for index in held_indices]
        )
        baseline_sse = max(
            float(np.sum(np.square(held_target - held_baseline))),
            1e-12,
        )
        for alpha_position, alpha in enumerate(alphas):
            gram = z_train.T @ z_train
            influence = z_held @ np.linalg.solve(
                gram + float(alpha) * np.eye(gram.shape[0]),
                z_train.T,
            )
            contributions = np.empty(
                (
                    len(training_subjects),
                    len(training_subjects),
                    rows_per_subject,
                    target.shape[1],
                ),
                dtype=np.float64,
            )
            for destination in range(len(training_subjects)):
                start = destination * rows_per_subject
                stop = start + rows_per_subject
                contributions[destination] = np.einsum(
                    "ij,kjt->kit",
                    influence[:, start:stop],
                    donor_targets,
                    optimize=True,
                )
            for start in range(0, int(replicates), int(batch_size)):
                stop = min(start + int(batch_size), int(replicates))
                choices = permutations[start:stop]
                prediction = np.broadcast_to(
                    target_mean,
                    (stop - start, rows_per_subject, target.shape[1]),
                ).copy()
                for destination in range(len(training_subjects)):
                    prediction += contributions[
                        destination, choices[:, destination]
                    ]
                model_sse = np.sum(
                    np.square(prediction - held_target[None, :, :]),
                    axis=(1, 2),
                )
                score_cube[start:stop, held_position, alpha_position] = (
                    1.0 - model_sse / baseline_sse
                )
    subject_equal = np.mean(score_cube, axis=1)
    return np.max(subject_equal, axis=1)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else 0.0


def _lin_concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    variance_x = float(np.var(x))
    variance_y = float(np.var(y))
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    denominator = variance_x + variance_y + (mean_x - mean_y) ** 2
    return 2.0 * covariance / denominator if denominator > 1e-12 else 0.0


def _bootstrap_mean(
    values: Sequence[float],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or not len(array):
        raise ValueError("Bootstrap requires finite non-empty subject values")
    rng = np.random.default_rng(seed)
    draws = np.mean(
        array[rng.integers(0, len(array), size=(int(replicates), len(array)))],
        axis=1,
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
    }


def freeze_calibration(
    train_records: Sequence[PanelRecord],
    registry: Mapping[str, Any],
    shifted_null_cache: Mapping[str, np.ndarray],
    *,
    strict_contract: bool = True,
) -> tuple[dict[str, Any], dict[str, RidgeModel], dict[str, Any]]:
    expected_subjects = {f"subject_{index:02d}" for index in range(1, 19)}
    observed_subjects = {record.subject for record in train_records}
    if any(record.split != "train" for record in train_records):
        raise ValueError("Calibration accepts train records only")
    if observed_subjects & set(PROTECTED_SUBJECT_IDS):
        raise RuntimeError("Calibration cannot accept protected records")
    if strict_contract and (
        len(train_records) != 1080 or observed_subjects != expected_subjects
    ):
        raise ValueError(
            "Formal calibration requires exactly 1080 records from subjects 01-18"
        )
    policy = registry["threshold_policy"]
    templates = {
        "hbo_observed": _condition_templates(train_records, "hbo_observed"),
        "hbr_observed": _condition_templates(train_records, "hbr_observed"),
        "rj": _condition_templates(train_records, "rj"),
    }
    train_physical = physical_train_subject_metrics_loto(train_records)
    retained = float(policy["retained_training_effect_fraction"])
    thresholds = {}
    for metric in (
        "hbo_physical_gain",
        "hbr_physical_gain",
        "hbo_jointness_gain",
        "hbr_jointness_gain",
    ):
        median = float(np.median([float(row[metric]) for row in train_physical]))
        thresholds[metric] = max(0.0, retained * median)
    correction_values = np.asarray(
        [float(row["correction_rms_ratio"]) for row in train_physical]
    )
    thresholds["correction_rms_ratio_lower"] = max(
        0.01, float(np.quantile(correction_values, 0.10)) / 2.0
    )
    thresholds["correction_rms_ratio_upper"] = min(
        2.0, 2.0 * float(np.quantile(correction_values, 0.90))
    )
    models: dict[str, RidgeModel] = {}
    observability: dict[str, Any] = {}
    observability_subject: dict[str, dict[str, float]] = {}
    observability_null_draws: dict[str, np.ndarray] = {}
    for index, modality in enumerate(("eeg", "fnirs")):
        model, metadata, subject_scores = fit_observability(
            train_records,
            modality,
            null_replicates=int(policy["null_replicates"]),
            null_quantile=float(policy["null_quantile"]),
            seed=int(registry["uncertainty"]["seed"]) + index,
        )
        observability_null_draws[modality] = np.asarray(
            metadata.pop("_null_draws"), dtype=np.float64
        )
        models[modality] = model
        observability[modality] = metadata
        observability_subject[modality] = subject_scores
        thresholds[f"{modality}_observability_delta_r2"] = max(
            0.0, float(metadata["train_subject_block_target_permutation_q"])
        )
    physical_scales = train_physical_scales(train_records)
    heldout_train = heldout_parity_subject_metrics(
        train_records, physical_scales
    )
    null_draws = shifted_solver_null_draws(
        train_records,
        shifted_null_cache,
        replicates=int(policy["null_replicates"]),
        seed=int(registry["uncertainty"]["seed"]) + 10,
    )
    thresholds["heldout_level_gain"] = float(
        np.quantile(
            null_draws["level_gain_draws"],
            float(policy["null_quantile"]),
            method="higher",
        )
    )
    thresholds["heldout_first_difference_gain"] = float(
        np.quantile(
            null_draws["first_difference_gain_draws"],
            float(policy["null_quantile"]),
            method="higher",
        )
    )
    calibration = {
        "schema": THRESHOLD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_subject_ids": sorted({record.subject for record in train_records}),
        "fit_sample_count": len(train_records),
        "validation_subjects_used": False,
        "protected_subjects_used": False,
        "templates": {
            family: {key: value.tolist() for key, value in mapping.items()}
            for family, mapping in templates.items()
        },
        "thresholds": thresholds,
        "observability": observability,
        "physical_scales": physical_scales,
        "physical_scoring_mask": {
            "patch_duration_s": 2.0,
            "patch_count": 10,
            "pass_even_masked_patches": [0, 2, 4, 6, 8],
            "pass_odd_masked_patches": [1, 3, 5, 7, 9],
            "score_masked_points_only": True,
            "hbo_hbr_masked_together": True,
        },
        "heldout_shift_null": {
            "replicates": int(policy["null_replicates"]),
            "quantile": float(policy["null_quantile"]),
            "quantile_method": "higher",
            "shift_patches": list(range(1, 10)),
            "grouping": "subject_x_session_x_condition",
            "level_q": thresholds["heldout_level_gain"],
            "first_difference_q": thresholds[
                "heldout_first_difference_gain"
            ],
        },
    }
    diagnostics = {
        "train_physical": train_physical,
        "train_heldout_parity": heldout_train,
        "train_observability_subject": observability_subject,
        "observability_null_draws": observability_null_draws,
        "shifted_solver_cache": {
            key: np.asarray(value) for key, value in shifted_null_cache.items()
        },
        "heldout_shift_null_draws": null_draws,
    }
    return calibration, models, diagnostics


def _templates_from_calibration(
    calibration: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        family: {
            key: np.asarray(value, dtype=np.float64) for key, value in mapping.items()
        }
        for family, mapping in calibration["templates"].items()
    }


def _observability_validation_rows(
    records: Sequence[PanelRecord],
    models: Mapping[str, RidgeModel],
    templates: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows = []
    by_modality = {}
    for modality in ("eeg", "fnirs"):
        features = _stack(records, f"{modality}_features")
        prediction = models[modality].predict(features)
        scores = _subject_delta_r2(records, prediction, templates)
        by_modality[modality] = scores
        for subject, value in sorted(scores.items()):
            rows.append(
                {
                    "subject": subject,
                    "split": "validation",
                    "modality": modality,
                    "delta_r2": value,
                    "alpha": models[modality].alpha,
                }
            )
    return rows, by_modality


def save_frozen_calibration_arrays(
    path: Path,
    models: Mapping[str, RidgeModel],
    diagnostics: Mapping[str, Any],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for modality, model in models.items():
        arrays[f"{modality}_feature_mean"] = model.feature_mean
        arrays[f"{modality}_feature_scale"] = model.feature_scale
        arrays[f"{modality}_coefficients"] = model.coefficients
        arrays[f"{modality}_intercept"] = model.intercept
        arrays[f"{modality}_alpha"] = np.asarray(model.alpha)
        subject_scores = diagnostics["train_observability_subject"][modality]
        arrays[f"{modality}_train_subject_ids"] = np.asarray(
            sorted(subject_scores), dtype=np.str_
        )
        arrays[f"{modality}_train_subject_delta_r2"] = np.asarray(
            [subject_scores[key] for key in sorted(subject_scores)],
            dtype=np.float64,
        )
        arrays[f"{modality}_target_permutation_null_draws"] = np.asarray(
            diagnostics["observability_null_draws"][modality],
            dtype=np.float64,
        )
    for key, value in diagnostics["shifted_solver_cache"].items():
        arrays[f"g6_cache_{key}"] = np.asarray(value)
    for key, value in diagnostics["heldout_shift_null_draws"].items():
        arrays[f"g6_null_{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_frozen_models(path: Path) -> dict[str, RidgeModel]:
    with np.load(path, allow_pickle=False) as arrays:
        return {
            modality: RidgeModel(
                feature_mean=np.asarray(
                    arrays[f"{modality}_feature_mean"], dtype=np.float64
                ),
                feature_scale=np.asarray(
                    arrays[f"{modality}_feature_scale"], dtype=np.float64
                ),
                coefficients=np.asarray(
                    arrays[f"{modality}_coefficients"], dtype=np.float64
                ),
                intercept=np.asarray(
                    arrays[f"{modality}_intercept"], dtype=np.float64
                ),
                alpha=float(np.asarray(arrays[f"{modality}_alpha"]).item()),
            )
            for modality in ("eeg", "fnirs")
        }


def load_and_verify_frozen_calibration(
    threshold_path: Path,
) -> tuple[dict[str, Any], dict[str, RidgeModel]]:
    calibration = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
    if calibration.get("schema") != THRESHOLD_SCHEMA:
        raise ValueError("Frozen threshold manifest schema mismatch")
    root = Path(threshold_path).parent
    arrays_path = root / str(calibration["calibration_arrays_file"])
    if _sha256(arrays_path) != calibration["calibration_arrays_sha256"]:
        raise RuntimeError("Frozen calibration array hash mismatch")
    diagnostics_path = root / str(calibration["train_diagnostics_file"])
    if _sha256(diagnostics_path) != calibration["train_diagnostics_sha256"]:
        raise RuntimeError("Frozen train diagnostics hash mismatch")
    models = load_frozen_models(arrays_path)
    quantile = float(calibration["heldout_shift_null"]["quantile"])
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for modality in ("eeg", "fnirs"):
            draws = np.asarray(
                arrays[f"{modality}_target_permutation_null_draws"],
                dtype=np.float64,
            )
            observed = float(
                np.quantile(draws, quantile, method="higher")
            )
            expected = float(
                calibration["thresholds"][
                    f"{modality}_observability_delta_r2"
                ]
            )
            if max(0.0, observed) != expected:
                raise RuntimeError(
                    f"Frozen {modality} observability null threshold mismatch"
                )
            subject_ids = np.asarray(
                arrays[f"{modality}_train_subject_ids"]
            ).astype(str)
            subject_scores = np.asarray(
                arrays[f"{modality}_train_subject_delta_r2"],
                dtype=np.float64,
            )
            if len(subject_ids) != 18 or len(subject_scores) != 18:
                raise RuntimeError(
                    f"Frozen {modality} train subject diagnostics are incomplete"
                )
        for field, threshold_key in (
            ("g6_null_level_gain_draws", "heldout_level_gain"),
            (
                "g6_null_first_difference_gain_draws",
                "heldout_first_difference_gain",
            ),
        ):
            observed = float(
                np.quantile(
                    np.asarray(arrays[field], dtype=np.float64),
                    quantile,
                    method="higher",
                )
            )
            if observed != float(calibration["thresholds"][threshold_key]):
                raise RuntimeError(
                    f"Frozen G6 threshold mismatch for {threshold_key}"
                )
    return calibration, models


def provenance_gate(
    bundle_root: Path,
    config_path: Path,
    train_records: Sequence[PanelRecord],
    validation_records: Sequence[PanelRecord],
) -> dict[str, Any]:
    top_manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    leakage = json.loads((bundle_root / "leakage_audit.json").read_text(encoding="utf-8"))
    parameter = json.loads(
        (bundle_root / "parameter_bundle/manifest.json").read_text(encoding="utf-8")
    )
    target_manifest = json.loads(
        (bundle_root / "trajectory_targets/manifest.json").read_text(encoding="utf-8")
    )
    arrays_path = bundle_root / "trajectory_targets" / target_manifest["arrays_file"]
    if _sha256(arrays_path) != target_manifest["arrays_sha256"]:
        raise RuntimeError("Trajectory target hash mismatch during qualification")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        sample_keys = np.asarray(arrays["sample_key"]).astype(str)
        subject = np.asarray(arrays["subject_id"]).astype(str)
        split = np.asarray(arrays["development_split"]).astype(str)
        protected_present = bool(set(subject) & set(PROTECTED_SUBJECT_IDS))
        if protected_present:
            raise RuntimeError(
                "Protected sidecar rows rejected before target dereference"
            )
        rj = np.asarray(
            arrays["target_shared_driver"], dtype=np.float64
        ).reshape((-1, 200))
        re = np.asarray(
            arrays["target_eeg_only_driver"], dtype=np.float64
        ).reshape((-1, 200))
        bundle_hashes = set(np.asarray(arrays["parameter_bundle_sha256"]).astype(str))
        gauge_hashes = set(np.asarray(arrays["teacher_gauge_hash"]).astype(str))
    train_values = rj[split == "train"].reshape(-1)
    max_rj_delta, max_re_delta = sidecar_recompute_deltas(
        [*train_records, *validation_records],
        sample_keys,
        rj,
        re,
        sidecar_subjects=subject,
        sidecar_splits=split,
    )
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cache_root = REPO_ROOT / str(config["data"]["cache_root"])
    source_paths = {
        "config": Path(config_path),
        "builder": REPO_ROOT
        / "experiments/scripts/build_r1p_population_frozen_teacher.py",
        "adaptive_evaluator": REPO_ROOT
        / "experiments/evaluate_adaptive_shared_neural_ssm.py",
        "adaptive_solver": REPO_ROOT
        / "src/inference/adaptive_neurovascular_ssm.py",
        "cache_manifest": cache_root / "cache_manifest.json",
        "event_manifest": cache_root / "event_index/event_manifest.json",
        "geometry_manifest": cache_root
        / "channel_geometry/geometry_manifest.json",
        "eeg_artifact_manifest": cache_root
        / "eeg_artifact_clean_v4/cache_manifest.json",
    }
    source_hashes = parameter.get("source", {}).get("input_hashes", {})
    source_checks = {
        name: path.is_file() and source_hashes.get(name) == _sha256(path)
        for name, path in source_paths.items()
    }
    checks = {
        "validation_fit_calls_zero": int(leakage.get("validation_fit_calls", -1)) == 0,
        "validation_normalization_calls_zero": int(
            leakage.get("validation_normalization_calls", -1)
        )
        == 0,
        "protected_array_dereference_count_zero": int(
            leakage.get("protected_array_dereference_count", -1)
        )
        == 0,
        "protected_open_false": not bool(leakage.get("protected_open", True)),
        "protected_rows_absent": not protected_present,
        "single_parameter_bundle_hash": bundle_hashes == {parameter["bundle_sha256"]},
        "single_shared_gauge_hash": len(gauge_hashes) == 1,
        "same_normalization_declared": target_manifest["paired_control"].get(
            "shared_driver_gauge_sha256"
        )
        == parameter["normalization"]["sha256"]
        and gauge_hashes == {parameter["normalization"]["sha256"]},
        "train_rj_mean_zero": abs(float(np.mean(train_values))) <= 1e-8,
        "train_rj_std_one": abs(float(np.std(train_values)) - 1.0) <= 1e-8,
        "record_counts_match": len(train_records) == 1080
        and len(validation_records) == 300,
        "recomputed_rj_matches_frozen_sidecar": max_rj_delta <= 1e-10,
        "recomputed_re_matches_frozen_sidecar": max_re_delta <= 1e-10,
        "source_hashes_match_current_code_and_inputs": all(source_checks.values()),
        "top_parameter_bundle_hash_matches": top_manifest["parameter_bundle"][
            "bundle_sha256"
        ]
        == parameter["bundle_sha256"],
        "top_parameter_manifest_hash_matches": top_manifest["parameter_bundle"][
            "manifest_sha256"
        ]
        == _sha256(bundle_root / "parameter_bundle/manifest.json"),
        "top_target_manifest_hash_matches": top_manifest["trajectory_targets"][
            "manifest_sha256"
        ]
        == _sha256(bundle_root / "trajectory_targets/manifest.json"),
        "top_target_arrays_hash_matches": top_manifest["trajectory_targets"][
            "arrays_sha256"
        ]
        == target_manifest["arrays_sha256"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "train_rj_mean": float(np.mean(train_values)),
        "train_rj_std": float(np.std(train_values)),
        "parameter_bundle_sha256": parameter["bundle_sha256"],
        "gauge_hashes": sorted(gauge_hashes),
        "max_recomputed_rj_abs_delta": max_rj_delta,
        "max_recomputed_re_abs_delta": max_re_delta,
        "source_hash_checks": source_checks,
    }


def sidecar_recompute_deltas(
    records: Sequence[PanelRecord],
    sample_keys: Sequence[str],
    frozen_rj: np.ndarray,
    frozen_re: np.ndarray,
    *,
    sidecar_subjects: Sequence[str] | None = None,
    sidecar_splits: Sequence[str] | None = None,
) -> tuple[float, float]:
    record_keys = [record.sample_key for record in records]
    normalized_sidecar_keys = [str(key) for key in sample_keys]
    if len(record_keys) != len(set(record_keys)):
        raise RuntimeError("Recomputed panel sample keys are not unique")
    if len(normalized_sidecar_keys) != len(set(normalized_sidecar_keys)):
        raise RuntimeError("Frozen sidecar sample keys are not unique")
    if (
        len(record_keys) != len(normalized_sidecar_keys)
        or set(record_keys) != set(normalized_sidecar_keys)
    ):
        raise RuntimeError(
            "Recomputed panel and frozen sidecar cohorts are not exactly equal"
        )
    if (
        len(frozen_rj) != len(normalized_sidecar_keys)
        or len(frozen_re) != len(normalized_sidecar_keys)
    ):
        raise RuntimeError("Frozen sidecar key/target row counts disagree")
    if sidecar_subjects is not None and len(sidecar_subjects) != len(
        normalized_sidecar_keys
    ):
        raise RuntimeError("Frozen sidecar key/subject row counts disagree")
    if sidecar_splits is not None and len(sidecar_splits) != len(
        normalized_sidecar_keys
    ):
        raise RuntimeError("Frozen sidecar key/split row counts disagree")
    lookup = {
        str(key): (
            np.asarray(frozen_rj[index], dtype=np.float64).reshape(-1),
            np.asarray(frozen_re[index], dtype=np.float64).reshape(-1),
        )
        for index, key in enumerate(sample_keys)
    }
    subject_lookup = (
        {
            normalized_sidecar_keys[index]: str(value)
            for index, value in enumerate(sidecar_subjects)
        }
        if sidecar_subjects is not None
        else None
    )
    split_lookup = (
        {
            normalized_sidecar_keys[index]: str(value)
            for index, value in enumerate(sidecar_splits)
        }
        if sidecar_splits is not None
        else None
    )
    max_rj_delta = 0.0
    max_re_delta = 0.0
    for record in records:
        if record.sample_key not in lookup:
            raise RuntimeError("Recomputed panel row is absent from frozen sidecar")
        expected_rj, expected_re = lookup[record.sample_key]
        if subject_lookup is not None and (
            subject_lookup[record.sample_key] != record.subject
        ):
            raise RuntimeError("Frozen sidecar subject/sample mapping drifted")
        if split_lookup is not None and (
            split_lookup[record.sample_key] != record.split
        ):
            raise RuntimeError("Frozen sidecar split/sample mapping drifted")
        max_rj_delta = max(
            max_rj_delta,
            float(np.max(np.abs(record.rj - expected_rj))),
        )
        max_re_delta = max(
            max_re_delta,
            float(np.max(np.abs(record.re - expected_re))),
        )
    return max_rj_delta, max_re_delta


def perturbation_stability_gate(
    primary_root: Path,
    perturbation_roots: Sequence[Path],
    registry: Mapping[str, Any],
    *,
    registry_sha256: str,
    prevalidation_seal: Mapping[str, Any],
    prevalidation_seal_sha256: str,
) -> dict[str, Any]:
    registered = {
        str(item["output_name"]): item for item in registry["perturbations"]
    }
    resolved_roots = [Path(root).resolve() for root in perturbation_roots]
    if len(resolved_roots) < 3:
        return {
            "status": "not_evaluated",
            "pass": False,
            "reason": "requires_at_least_three_train_only_perturbation_bundles",
            "provided_bundle_count": len(perturbation_roots),
            "registered_output_names": sorted(registered),
        }
    if len(resolved_roots) != 3 or len(set(resolved_roots)) != 3:
        raise ValueError("G4 requires exactly three unique resolved bundle roots")
    observed_names = {root.name for root in resolved_roots}
    if observed_names != set(registered):
        raise ValueError(
            "Provided perturbation roots do not match the frozen output-name registry"
        )
    roots = [Path(primary_root).resolve(), *resolved_roots]
    payloads = []
    sealed = {
        str(item["role"]): item for item in prevalidation_seal["sealed_files"]
    }
    sealed_config_path = REPO_ROOT / sealed["teacher_config"]["path"]
    sealed_config = yaml.safe_load(
        sealed_config_path.read_text(encoding="utf-8")
    )
    configured_cache_root = Path(str(sealed_config["data"]["cache_root"]))
    cache_root = (
        configured_cache_root
        if configured_cache_root.is_absolute()
        else REPO_ROOT / configured_cache_root
    )
    cache_source_paths = {
        "clean_cache_manifest": cache_root / "cache_manifest.json",
        "event_manifest": cache_root / "event_index/event_manifest.json",
        "geometry_manifest": cache_root
        / "channel_geometry/geometry_manifest.json",
        "eeg_artifact_manifest": cache_root
        / "eeg_artifact_clean_v4/cache_manifest.json",
    }
    for root_index, root in enumerate(roots):
        top_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        parameter = json.loads(
            (root / "parameter_bundle/manifest.json").read_text(encoding="utf-8")
        )
        if parameter.get("schema") != PARAMETER_MANIFEST_SCHEMA:
            raise ValueError("Perturbation parameter schema mismatch")
        if bool(parameter.get("validation_subjects_used_for_any_fit", True)):
            raise RuntimeError("Perturbation bundle admits validation fitting")
        if bool(parameter.get("protected_subjects_used_for_any_fit", True)):
            raise RuntimeError("Perturbation bundle admits protected fitting")
        leakage = json.loads((root / "leakage_audit.json").read_text(encoding="utf-8"))
        if (
            int(leakage.get("validation_fit_calls", -1)) != 0
            or int(leakage.get("validation_normalization_calls", -1)) != 0
            or int(leakage.get("protected_array_dereference_count", -1)) != 0
            or bool(leakage.get("protected_open", True))
        ):
            raise RuntimeError(f"Perturbation {root.name} leakage audit is unsafe")
        parameter_manifest_path = root / "parameter_bundle/manifest.json"
        parameter_arrays_path = (
            root / "parameter_bundle" / str(parameter["arrays_file"])
        )
        if _sha256(parameter_arrays_path) != parameter["arrays_sha256"]:
            raise RuntimeError(f"Perturbation {root.name} parameter arrays hash mismatch")
        if (
            top_manifest["parameter_bundle"]["bundle_sha256"]
            != parameter["bundle_sha256"]
            or top_manifest["parameter_bundle"]["manifest_sha256"]
            != _sha256(parameter_manifest_path)
        ):
            raise RuntimeError(f"Perturbation {root.name} top/parameter hash mismatch")
        if root_index > 0:
            definition = registered[root.name]
            expected_subject_keys = {
                f"eeg_fnirs_single_trial|{subject}"
                for subject in definition["retained_fit_subjects"]
            }
            if set(parameter["fit_subject_keys"]) != expected_subject_keys:
                raise RuntimeError(
                    f"Perturbation {root.name} fit cohort differs from registry"
                )
            if (
                int(top_manifest.get("train_sample_count", -1))
                != int(definition["expected_fit_windows"])
                or int(top_manifest.get("validation_sample_count", -1)) != 300
                or int(top_manifest.get("sample_count", -1))
                != int(definition["expected_fit_windows"]) + 300
            ):
                raise RuntimeError(
                    f"Perturbation {root.name} sample counts differ from registry"
                )
            normalization = parameter.get("normalization", {})
            if (
                int(normalization.get("fit_sample_count", -1))
                != int(definition["expected_fit_windows"])
                or set(normalization.get("fit_subject_keys", []))
                != expected_subject_keys
            ):
                raise RuntimeError(
                    f"Perturbation {root.name} gauge-fit cohort differs from registry"
                )
            source = parameter.get("source", {})
            expected_source = {
                "perturbation_id": definition["perturbation_id"],
                "perturbation_registry_sha256": registry_sha256,
                "perturbation_definition_sha256": _json_sha256(definition),
                "anchor_fit_sessions": definition["anchor_rows"]["sessions"],
                "anchor_fit_condition": definition["anchor_rows"]["condition"],
                "prevalidation_seal_sha256": prevalidation_seal_sha256,
                "prevalidation_seal_path": str(
                    DEFAULT_PREVALIDATION_SEAL.resolve()
                ),
            }
            for key, expected in expected_source.items():
                if source.get(key) != expected:
                    raise RuntimeError(
                        f"Perturbation {root.name} provenance field {key} differs "
                        "from the frozen registry"
                    )
            expected_input_hashes = {
                "config": sealed["teacher_config"]["sha256"],
                "perturbation_registry": sealed["perturbation_registry"][
                    "sha256"
                ],
                "builder": sealed["perturbation_builder"]["sha256"],
                "base_builder": sealed["teacher_builder"]["sha256"],
                "adaptive_evaluator": sealed["adaptive_evaluator"]["sha256"],
                "adaptive_solver": sealed["adaptive_solver"]["sha256"],
                "prevalidation_seal": prevalidation_seal_sha256,
                **{
                    key: _sha256(path)
                    for key, path in cache_source_paths.items()
                },
            }
            input_hashes = source.get("input_hashes", {})
            for key, expected in expected_input_hashes.items():
                if input_hashes.get(key) != expected:
                    raise RuntimeError(
                        f"Perturbation {root.name} input hash {key} differs "
                        "from the prevalidation seal"
                    )
        manifest = json.loads(
            (root / "trajectory_targets/manifest.json").read_text(encoding="utf-8")
        )
        raw_manifest = json.loads(
            (root / "raw_view_registry/manifest.json").read_text(encoding="utf-8")
        )
        raw_path = root / "raw_view_registry" / raw_manifest["arrays_file"]
        if _sha256(raw_path) != raw_manifest["arrays_sha256"]:
            raise RuntimeError(f"Perturbation {root.name} raw arrays hash mismatch")
        target_path = root / "trajectory_targets" / manifest["arrays_file"]
        if _sha256(target_path) != manifest["arrays_sha256"]:
            raise RuntimeError(f"Perturbation {root.name} trajectory hash mismatch")
        with np.load(target_path, allow_pickle=False) as arrays:
            split = np.asarray(arrays["development_split"]).astype(str)
            all_subject_array = np.asarray(arrays["subject_id"]).astype(str)
            all_subjects = set(all_subject_array)
            if all_subjects & set(PROTECTED_SUBJECT_IDS):
                raise RuntimeError(
                    f"Perturbation {root.name} contains protected rows"
                )
            all_keys = np.asarray(arrays["sample_key"]).astype(str)
            keys = all_keys[split == "validation"]
            target = np.asarray(arrays["target_shared_driver"], dtype=np.float64)[
                split == "validation"
            ].reshape((-1, 200))
            subjects = np.asarray(arrays["subject_id"]).astype(str)[split == "validation"]
            row_bundle_hashes = set(
                np.asarray(arrays["parameter_bundle_sha256"]).astype(str)
            )
            row_gauge_hashes = set(
                np.asarray(arrays["teacher_gauge_hash"]).astype(str)
            )
        if set(split) != {"train", "validation"}:
            raise RuntimeError(
                f"Perturbation {root.name} contains unregistered split labels"
            )
        if len(all_keys) != len(set(all_keys)):
            raise RuntimeError(f"Perturbation {root.name} sample keys are not unique")
        validation_subject_array = all_subject_array[split == "validation"]
        validation_counts = Counter(validation_subject_array.tolist())
        expected_validation_subjects = {
            f"subject_{index:02d}" for index in range(19, 24)
        }
        if (
            len(keys) != 300
            or set(validation_counts) != expected_validation_subjects
            or set(validation_counts.values()) != {60}
        ):
            raise RuntimeError(
                f"Perturbation {root.name} validation cohort is not exact 5x60"
            )
        train_subject_array = all_subject_array[split == "train"]
        train_counts = Counter(train_subject_array.tolist())
        expected_train_subjects = (
            {f"subject_{index:02d}" for index in range(1, 19)}
            if root_index == 0
            else set(registered[root.name]["retained_fit_subjects"])
        )
        expected_train_rows = 1080 if root_index == 0 else 900
        if (
            len(train_subject_array) != expected_train_rows
            or set(train_counts) != expected_train_subjects
            or set(train_counts.values()) != {60}
        ):
            raise RuntimeError(
                f"Perturbation {root.name} train cohort is not exact"
            )
        with np.load(raw_path, allow_pickle=False) as raw_arrays:
            raw_keys = np.asarray(raw_arrays["sample_key"]).astype(str)
            raw_subjects = np.asarray(raw_arrays["subject_id"]).astype(str)
        if not np.array_equal(raw_keys, all_keys) or not np.array_equal(
            raw_subjects, all_subject_array
        ):
            raise RuntimeError(
                f"Perturbation {root.name} raw/trajectory row order drifted"
            )
        if row_bundle_hashes != {parameter["bundle_sha256"]}:
            raise RuntimeError(f"Perturbation {root.name} row bundle hashes drifted")
        expected_gauge = str(parameter["normalization"]["sha256"])
        if row_gauge_hashes != {expected_gauge}:
            raise RuntimeError(f"Perturbation {root.name} row gauge hashes drifted")
        if (
            top_manifest["trajectory_targets"]["manifest_sha256"]
            != _sha256(root / "trajectory_targets/manifest.json")
            or top_manifest["trajectory_targets"]["arrays_sha256"]
            != manifest["arrays_sha256"]
        ):
            raise RuntimeError(f"Perturbation {root.name} top/target hash mismatch")
        if (
            top_manifest["raw_view_registry"]["manifest_sha256"]
            != _sha256(root / "raw_view_registry/manifest.json")
            or top_manifest["raw_view_registry"]["arrays_sha256"]
            != raw_manifest["arrays_sha256"]
        ):
            raise RuntimeError(f"Perturbation {root.name} top/raw hash mismatch")
        if (
            manifest.get("sample_count") != raw_manifest.get("sample_count")
            or manifest.get("sample_order_sha256")
            != raw_manifest.get("sample_order_sha256")
            or raw_manifest.get("parameter_bundle_sha256")
            != parameter.get("bundle_sha256")
        ):
            raise RuntimeError(
                f"Perturbation {root.name} trajectory/raw cohort closure failed"
            )
        if root_index > 0:
            definition = registered[root.name]
            expected_definition_sha = _json_sha256(definition)
            definition_manifest = json.loads(
                (root / "perturbation_definition.json").read_text(
                    encoding="utf-8"
                )
            )
            for artifact_name, artifact in (
                ("top", top_manifest),
                ("parameter", parameter),
                ("trajectory", manifest),
                ("raw", raw_manifest),
            ):
                if (
                    artifact.get("perturbation_registry_sha256")
                    != registry_sha256
                    or artifact.get("perturbation_definition_sha256")
                    != expected_definition_sha
                    or artifact.get("prevalidation_seal_sha256")
                    != prevalidation_seal_sha256
                ):
                    raise RuntimeError(
                        f"Perturbation {root.name} {artifact_name} provenance drifted"
                    )
            if (
                definition_manifest.get("definition") != definition
                or definition_manifest.get("definition_sha256")
                != expected_definition_sha
                or definition_manifest.get("registry_sha256")
                != registry_sha256
                or definition_manifest.get("prevalidation_seal_sha256")
                != prevalidation_seal_sha256
            ):
                raise RuntimeError(
                    f"Perturbation {root.name} definition provenance drifted"
                )
            for artifact_name, artifact in (
                ("top", top_manifest),
                ("trajectory", manifest),
                ("raw", raw_manifest),
                ("leakage", leakage),
            ):
                if artifact.get("source") != parameter.get("source"):
                    raise RuntimeError(
                        f"Perturbation {root.name} {artifact_name} source "
                        "does not close to parameter source"
                    )
        payloads.append((root.name, keys, target, subjects))
    _, primary_keys, primary, subjects = payloads[0]
    metric_contract = registry["qualification_metric"]
    ccc_threshold = float(
        metric_contract["minimum_median_subject_ccc_per_perturbation"]
    )
    nrmse_threshold = float(
        metric_contract["maximum_median_subject_identity_nrmse_per_perturbation"]
    )
    required_ccc_subjects = int(
        metric_contract[
            "minimum_subjects_with_ccc_at_least_0_7_per_perturbation"
        ]
    )
    required_nrmse_subjects = int(
        metric_contract[
            "minimum_subjects_with_identity_nrmse_at_most_0_75_per_perturbation"
        ]
    )
    details = []
    for root_name, keys, target, other_subjects in payloads[1:]:
        if not np.array_equal(keys, primary_keys) or not np.array_equal(
            subjects, other_subjects
        ):
            raise RuntimeError("Perturbation targets are not sample aligned")
        ccc_values = []
        nrmse_values = []
        for subject in sorted(set(subjects)):
            mask = subjects == subject
            base = primary[mask].reshape(-1)
            perturbed = target[mask].reshape(-1)
            ccc_values.append(_lin_concordance_correlation(base, perturbed))
            nrmse_values.append(
                float(
                    np.sqrt(np.mean(np.square(base - perturbed)))
                    / max(float(np.std(base)), 1e-12)
                )
            )
        median_ccc = float(np.median(ccc_values))
        median_nrmse = float(np.median(nrmse_values))
        ccc_count = sum(value >= ccc_threshold for value in ccc_values)
        nrmse_count = sum(value <= nrmse_threshold for value in nrmse_values)
        passed = (
            median_ccc >= ccc_threshold
            and ccc_count >= required_ccc_subjects
            and median_nrmse <= nrmse_threshold
            and nrmse_count >= required_nrmse_subjects
        )
        details.append(
            {
                "output_name": root_name,
                "median_subject_ccc": median_ccc,
                "subjects_ccc_at_or_above_threshold": ccc_count,
                "median_subject_identity_nrmse": median_nrmse,
                "subjects_nrmse_at_or_below_threshold": nrmse_count,
                "pass": passed,
            }
        )
    return {
        "status": "evaluated",
        "pass": all(item["pass"] for item in details),
        "per_perturbation": details,
        "threshold_ccc": ccc_threshold,
        "threshold_identity_nrmse": nrmse_threshold,
        "required_subject_count_ccc": required_ccc_subjects,
        "required_subject_count_nrmse": required_nrmse_subjects,
        "perturbation_bundle_count": len(resolved_roots),
        "resolved_roots": [str(root) for root in resolved_roots],
    }


def _gate_rows(
    provenance: Mapping[str, Any],
    physical_rows: Sequence[Mapping[str, Any]],
    observability: Mapping[str, Mapping[str, float]],
    correction_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    perturbation: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    uncertainty = registry["uncertainty"]
    bootstrap_kwargs = {
        "replicates": int(uncertainty["replicates"]),
        "confidence": float(uncertainty["confidence_level"]),
        "seed": int(uncertainty["seed"]),
    }
    positive_floor = int(registry["threshold_policy"]["positive_subject_floor"])
    gates = []

    def effect_gate(gate_id: str, metrics: Sequence[str]) -> tuple[bool, dict[str, Any]]:
        details = {}
        passed = True
        for offset, metric in enumerate(metrics):
            values = [float(row[metric]) for row in physical_rows]
            ci = _bootstrap_mean(
                values,
                **{**bootstrap_kwargs, "seed": bootstrap_kwargs["seed"] + offset},
            )
            threshold = float(thresholds[metric])
            positive = sum(value > threshold for value in values)
            metric_pass = ci["lower"] > threshold and positive >= positive_floor
            details[metric] = {
                **ci,
                "threshold": threshold,
                "positive_subject_count": positive,
                "pass": metric_pass,
            }
            passed = passed and metric_pass
        return passed, details

    g2, g2_details = effect_gate(
        "G2_PHYSICAL_RECONSTRUCTION",
        ("hbo_physical_gain", "hbr_physical_gain"),
    )
    joint_pass, joint_details = effect_gate(
        "G3_JOINTNESS_AND_NONDEGENERATE_CORRECTION",
        ("hbo_jointness_gain", "hbr_jointness_gain"),
    )
    ratios = np.asarray(
        [float(row["correction_rms_ratio"]) for row in physical_rows]
    )
    lower = float(thresholds["correction_rms_ratio_lower"])
    upper = float(thresholds["correction_rms_ratio_upper"])
    nonzero = sum(np.isfinite(ratios) & (ratios > 1e-12))
    correction_pass = (
        lower < float(np.median(ratios)) < upper
        and nonzero >= positive_floor
    )
    joint_details["correction_rms_ratio"] = {
        "median": float(np.median(ratios)),
        "lower": lower,
        "upper": upper,
        "finite_nonzero_subject_count": int(nonzero),
        "pass": correction_pass,
    }
    g3 = joint_pass and correction_pass
    g5_details = {}
    g5 = True
    for offset, modality in enumerate(("eeg", "fnirs")):
        values = list(observability[modality].values())
        ci = _bootstrap_mean(
            values,
            **{**bootstrap_kwargs, "seed": bootstrap_kwargs["seed"] + 20 + offset},
        )
        threshold = float(thresholds[f"{modality}_observability_delta_r2"])
        positive = sum(value > threshold for value in values)
        passed = ci["lower"] > threshold and positive >= positive_floor
        g5_details[modality] = {
            **ci,
            "threshold": threshold,
            "positive_subject_count": positive,
            "pass": passed,
        }
        g5 = g5 and passed
    correction_details = {}
    g6 = True
    for metric, threshold_key in (
        ("heldout_level_gain", "heldout_level_gain"),
        ("heldout_first_difference_gain", "heldout_first_difference_gain"),
    ):
        values = [float(row[metric]) for row in correction_rows]
        mean = float(np.mean(values))
        threshold = float(thresholds[threshold_key])
        positive = sum(value > threshold for value in values)
        passed = mean > threshold and positive >= positive_floor
        correction_details[metric] = {
            "subject_equal_mean": mean,
            "threshold": threshold,
            "positive_null_margin_subject_count": positive,
            "pass": passed,
        }
        g6 = g6 and passed
    gate_payloads = [
        ("G1_PROVENANCE_AND_SHARED_GAUGE", bool(provenance["pass"]), provenance),
        ("G2_PHYSICAL_RECONSTRUCTION", g2, g2_details),
        (
            "G3_JOINTNESS_AND_NONDEGENERATE_CORRECTION",
            g3,
            joint_details,
        ),
        (
            "G4_TRAIN_ONLY_PERTURBATION_STABILITY",
            bool(perturbation["pass"]),
            perturbation,
        ),
        ("G5_RAW_ONLY_TARGET_OBSERVABILITY", g5, g5_details),
        ("G6_CORRECTION_NULL_EXCLUSION", g6, correction_details),
    ]
    details = {}
    for gate_id, passed, payload in gate_payloads:
        gates.append(
            {
                "gate_id": gate_id,
                "status": "pass" if passed else "fail",
                "pass": passed,
            }
        )
        details[gate_id] = payload
    return gates, details


def _plot_panel(
    output_root: Path,
    physical_rows: Sequence[Mapping[str, Any]],
    observability: Mapping[str, Mapping[str, float]],
    correction_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    gate_rows: Sequence[Mapping[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    subjects = [str(row["subject"]).replace("subject_", "") for row in physical_rows]
    x = np.arange(len(subjects))
    ax = axes[0, 0]
    for metric, color, marker, label in (
        ("hbo_physical_gain", OKABE_ITO["blue"], "o", "HbO"),
        ("hbr_physical_gain", OKABE_ITO["orange"], "s", "HbR"),
    ):
        values = [float(row[metric]) for row in physical_rows]
        ax.scatter(x, values, color=color, marker=marker, label=label, zorder=3)
        ax.axhline(float(thresholds[metric]), color=color, linestyle="--", linewidth=1)
    ax.axhline(0, color=OKABE_ITO["gray"], linewidth=0.7)
    ax.set_xticks(x, subjects)
    ax.set_ylabel("ΔNMSE vs train template")
    ax.set_title("Physical reconstruction")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for offset, (modality, color, marker) in enumerate(
        (("eeg", OKABE_ITO["green"], "o"), ("fnirs", OKABE_ITO["purple"], "s"))
    ):
        values = [observability[modality][f"subject_{int(value):02d}"] for value in subjects]
        ax.scatter(x + (offset - 0.5) * 0.12, values, color=color, marker=marker, label=modality.upper())
        ax.axhline(
            float(thresholds[f"{modality}_observability_delta_r2"]),
            color=color,
            linestyle="--",
            linewidth=1,
        )
    ax.axhline(0, color=OKABE_ITO["gray"], linewidth=0.7)
    ax.set_xticks(x, subjects)
    ax.set_ylabel("ΔR² vs train template")
    ax.set_title("Raw-only target observability")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    level = [float(row["heldout_level_gain"]) for row in correction_rows]
    difference = [
        float(row["heldout_first_difference_gain"])
        for row in correction_rows
    ]
    ax.scatter(x - 0.06, level, color=OKABE_ITO["sky"], marker="o", label="Level")
    ax.scatter(
        x + 0.06,
        difference,
        color=OKABE_ITO["vermillion"],
        marker="^",
        label="First difference",
    )
    ax.axhline(
        float(thresholds["heldout_level_gain"]),
        color=OKABE_ITO["sky"],
        linestyle="--",
        linewidth=1,
    )
    ax.axhline(
        float(thresholds["heldout_first_difference_gain"]),
        color=OKABE_ITO["vermillion"],
        linestyle=":",
        linewidth=1,
    )
    ax.set_xticks(x, subjects)
    ax.set_ylabel("Heldout SSE gain vs EEG-only")
    ax.set_title("Shift/smooth null exclusion")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    names = [str(row["gate_id"]).split("_", 1)[0] for row in gate_rows]
    passed = [bool(row["pass"]) for row in gate_rows]
    colors = [OKABE_ITO["green"] if value else OKABE_ITO["vermillion"] for value in passed]
    ax.bar(np.arange(len(names)), np.ones(len(names)), color=colors, width=0.7)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_yticks([])
    ax.set_ylim(0, 1.15)
    ax.set_title("Qualification gates")
    for index, value in enumerate(passed):
        ax.text(index, 0.5, "PASS" if value else "FAIL", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    for label, ax in zip("ABCD", axes.reshape(-1)):
        ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.suptitle(
        "R1-P population-frozen teacher qualification",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figures = output_root / "figures"
    figures.mkdir()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            figures / f"r1p_teacher_qualification.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def qualify(
    config_path: Path,
    bundle_root: Path,
    output_root: Path,
    registry_path: Path,
    perturbation_registry_path: Path,
    prevalidation_seal_path: Path,
    perturbation_roots: Sequence[Path] = (),
) -> Path:
    config_path = Path(config_path).resolve()
    bundle_root = Path(bundle_root).resolve()
    output_root = Path(output_root).resolve()
    registry_path = Path(registry_path).resolve()
    perturbation_registry_path = Path(perturbation_registry_path).resolve()
    prevalidation_seal_path = Path(prevalidation_seal_path).resolve()
    if prevalidation_seal_path != DEFAULT_PREVALIDATION_SEAL.resolve():
        raise RuntimeError(
            "Qualification requires the tracked default prevalidation seal path"
        )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    registry = load_registry(registry_path)
    perturbation_registry = load_perturbation_registry(
        perturbation_registry_path
    )
    prevalidation_seal = load_prevalidation_seal(prevalidation_seal_path)
    seal_input_checks = verify_qualification_sealed_inputs(
        prevalidation_seal,
        config_path=config_path,
        bundle_root=bundle_root,
        registry_path=registry_path,
        perturbation_registry_path=perturbation_registry_path,
    )
    sealed_lookup = {
        str(item["role"]): item for item in prevalidation_seal["sealed_files"]
    }
    if sealed_lookup["qualification_registry"]["sha256"] != _sha256(registry_path):
        raise RuntimeError("CLI qualification registry differs from prevalidation seal")
    if sealed_lookup["perturbation_registry"]["sha256"] != _sha256(
        perturbation_registry_path
    ):
        raise RuntimeError("CLI perturbation registry differs from prevalidation seal")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = validate_population_config(config)
    bundle = load_population_bundle(bundle_root / "parameter_bundle")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.tmp-",
        dir=output_root.parent,
    ) as temporary:
        temporary_root = Path(temporary)

        # Phase 1: train-only calibration.  No validation arrays have been
        # dereferenced when the threshold manifest is serialized.
        train_trials, train_audit = load_registered_trials(
            config,
            allowed_subject_keys=split["train_subject_keys"],
            development_role="train_fit",
        )
        train_records = evaluate_trials(train_trials, bundle)
        if (
            int(train_audit.get("sample_count", -1)) != 1080
            or int(train_audit.get("protected_array_dereference_count", -1)) != 0
        ):
            raise RuntimeError("Instrumented train registry audit violates calibration contract")
        physical_scales = train_physical_scales(train_records)
        shifted_null_cache = precompute_shifted_solver_null(
            train_trials,
            train_records,
            bundle,
            physical_scales,
        )
        calibration, models, train_diagnostics = freeze_calibration(
            train_records,
            registry,
            shifted_null_cache,
            strict_contract=True,
        )
        calibration["registry_path"] = str(registry_path)
        calibration["registry_sha256"] = _sha256(registry_path)
        calibration["perturbation_registry_path"] = str(
            perturbation_registry_path
        )
        calibration["perturbation_registry_sha256"] = _sha256(
            perturbation_registry_path
        )
        calibration["config_sha256"] = _sha256(config_path)
        calibration["prevalidation_seal_path"] = str(prevalidation_seal_path)
        calibration["prevalidation_seal_sha256"] = _sha256(
            prevalidation_seal_path
        )
        calibration["prevalidation_seal_input_checks"] = seal_input_checks
        calibration["parameter_bundle_sha256"] = bundle.bundle_sha256
        calibration["train_registry_audit"] = train_audit
        calibration_arrays = temporary_root / "frozen_calibration_arrays.npz"
        save_frozen_calibration_arrays(
            calibration_arrays,
            models,
            train_diagnostics,
        )
        calibration["calibration_arrays_file"] = calibration_arrays.name
        calibration["calibration_arrays_sha256"] = _sha256(calibration_arrays)
        train_rows = []
        heldout_lookup = {
            row["subject"]: row
            for row in train_diagnostics["train_heldout_parity"]
        }
        for row in train_diagnostics["train_physical"]:
            train_rows.append(
                {
                    **row,
                    **{
                        key: value
                        for key, value in heldout_lookup[row["subject"]].items()
                        if key not in {"subject", "split"}
                    },
                    "eeg_observability_delta_r2": train_diagnostics[
                        "train_observability_subject"
                    ]["eeg"][row["subject"]],
                    "fnirs_observability_delta_r2": train_diagnostics[
                        "train_observability_subject"
                    ]["fnirs"][row["subject"]],
                }
            )
        train_diagnostics_path = temporary_root / "train_diagnostics.tsv"
        _write_tsv(train_diagnostics_path, train_rows)
        calibration["train_diagnostics_file"] = train_diagnostics_path.name
        calibration["train_diagnostics_sha256"] = _sha256(
            train_diagnostics_path
        )
        threshold_path = temporary_root / "threshold_manifest.json"
        _write_json(threshold_path, calibration)
        calibration, models = load_and_verify_frozen_calibration(
            threshold_path
        )

        # Phase 2: validation is pure application of already-frozen objects.
        reverify_qualification_seal_before_validation(
            prevalidation_seal_path=prevalidation_seal_path,
            expected_prevalidation_seal_sha256=calibration[
                "prevalidation_seal_sha256"
            ],
            expected_input_checks=seal_input_checks,
            config_path=config_path,
            bundle_root=bundle_root,
            registry_path=registry_path,
            perturbation_registry_path=perturbation_registry_path,
        )
        validation_trials, validation_audit = load_registered_trials(
            config,
            allowed_subject_keys=split["validation_subject_keys"],
            development_role="validation_pure_apply",
        )
        validation_records = evaluate_trials(validation_trials, bundle)
        templates = _templates_from_calibration(calibration)
        physical_rows = physical_subject_metrics(validation_records, templates)
        observability_rows, observability = _observability_validation_rows(
            validation_records,
            models,
            templates["rj"],
        )
        correction_rows = heldout_parity_subject_metrics(
            validation_records,
            calibration["physical_scales"],
        )
        provenance = provenance_gate(
            bundle_root,
            config_path,
            train_records,
            validation_records,
        )
        perturbation = perturbation_stability_gate(
            bundle_root,
            perturbation_roots,
            perturbation_registry,
            registry_sha256=_sha256(perturbation_registry_path),
            prevalidation_seal=prevalidation_seal,
            prevalidation_seal_sha256=_sha256(prevalidation_seal_path),
        )
        gates, gate_details = _gate_rows(
            provenance,
            physical_rows,
            observability,
            correction_rows,
            calibration["thresholds"],
            perturbation,
            registry,
        )
        promotion = all(bool(row["pass"]) for row in gates)
        summary = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "registry_path": str(registry_path),
            "registry_sha256": _sha256(registry_path),
            "perturbation_registry_path": str(perturbation_registry_path),
            "perturbation_registry_sha256": _sha256(
                perturbation_registry_path
            ),
            "prevalidation_seal_path": str(prevalidation_seal_path),
            "prevalidation_seal_sha256": _sha256(prevalidation_seal_path),
            "threshold_manifest_sha256": _sha256(
                temporary_root / "threshold_manifest.json"
            ),
            "config_path": str(config_path),
            "bundle_root": str(bundle_root),
            "parameter_bundle_sha256": bundle.bundle_sha256,
            "train_sample_count": len(train_records),
            "validation_sample_count": len(validation_records),
            "validation_loaded_after_threshold_freeze": True,
            "validation_fit_calls": 0,
            "validation_normalization_calls": 0,
            "protected_array_dereference_count": 0,
            "protected_open": False,
            "gates": gates,
            "gate_details": gate_details,
            "promotion_eligible": promotion,
            "promotion_blocker": None
            if promotion
            else "one_or_more_r1p_teacher_qualification_gates_failed",
            "next_action": "enter_r2_p" if promotion else "do_not_enter_r2_p",
            "validation_registry_audit": validation_audit,
        }
        subject_rows = []
        correction_lookup = {row["subject"]: row for row in correction_rows}
        for row in physical_rows:
            subject_rows.append({**row, **{
                key: value
                for key, value in correction_lookup[row["subject"]].items()
                if key not in {"subject", "split"}
            }})
        _write_tsv(temporary_root / "subject_metrics.tsv", subject_rows)
        _write_tsv(temporary_root / "observability.tsv", observability_rows)
        _write_tsv(temporary_root / "gate_results.tsv", gates)
        _write_json(temporary_root / "panel_summary.json", summary)
        _plot_panel(
            temporary_root,
            physical_rows,
            observability,
            correction_rows,
            calibration["thresholds"],
            gates,
        )
        temporary_root.replace(output_root)
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "experiments/configs/physiology_semantic_tokenizer/"
            "r1p_population_frozen_teacher.yaml"
        ),
    )
    parser.add_argument(
        "--bundle-root",
        default="data/cache/shared_driver_r1_v1/r1_p_development_v1",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument(
        "--perturbation-registry",
        default=str(DEFAULT_PERTURBATION_REGISTRY),
    )
    parser.add_argument(
        "--prevalidation-seal",
        default=str(DEFAULT_PREVALIDATION_SEAL),
    )
    parser.add_argument(
        "--perturbation-bundle-root",
        action="append",
        default=[],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = qualify(
        Path(args.config),
        Path(args.bundle_root),
        Path(args.output_root),
        Path(args.registry),
        Path(args.perturbation_registry),
        Path(args.prevalidation_seal),
        [Path(value) for value in args.perturbation_bundle_root],
    )
    summary = json.loads((output / "panel_summary.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_root": str(output),
                "promotion_eligible": summary["promotion_eligible"],
                "next_action": summary["next_action"],
                "protected_open": summary["protected_open"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
