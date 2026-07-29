#!/usr/bin/env python3
"""Build registered train-only R1-P perturbation teacher bundles.

Each perturbation refits its anchor, EEG projection, adaptive SSM, and scalar
driver gauge using only the 15/900 rows declared in the runtime perturbation
registry.  Validation subjects 19-23 are pure apply and are first dereferenced
after parameter and gauge serialization.  Protected subjects 24-29 are
rejected before measured-array access.

This builder produces G4 inputs only.  It does not evaluate qualification
metrics and it never modifies the base R1-P bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from threadpoolctl import threadpool_limits


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.build_r1p_population_frozen_teacher import (  # noqa: E402
    ARCHITECTURE_GENERATION,
    PARAMETER_ARRAY_SCHEMA,
    PARAMETER_MANIFEST_SCHEMA,
    PATCH_COUNT,
    POINTS_PER_PATCH,
    POINT_COUNT,
    PROTECTED_SUBJECT_IDS,
    SCHEMA,
    TARGET_FAMILY,
    TEACHER_SCOPE,
    VALIDATION_SUBJECT_IDS,
    PairedDriverResult,
    PopulationFrozenBundle,
    PopulationTrial,
    _anchor_id,
    _assert_common_channel_contract,
    _chromophore_targets,
    _condition_lookup,
    _fit_eeg_adapter,
    _fit_model,
    _git_state,
    _indices_by_name,
    _json_sha256,
    _local_eeg_indices,
    _normalized_pair,
    _paired_hbr_indices,
    _parameter_arrays,
    _select_active_hbo,
    _sha256,
    apply_paired_driver,
    load_population_bundle,
    validate_population_config,
)
from src.data.shared_driver_targets import (  # noqa: E402
    RAW_VIEW_ARRAY_SCHEMA,
    RAW_VIEW_REGISTRY_SCHEMA,
    SHARED_DRIVER_ARRAY_SCHEMA,
    SHARED_DRIVER_SIDECAR_SCHEMA,
    string_array,
)
from src.data.unified_physiology import UnifiedPhysiologyWindowDataset  # noqa: E402
from src.inference.adaptive_neurovascular_ssm import fit_to_mapping  # noqa: E402


PERTURBATION_REGISTRY_SCHEMA = "r1p_teacher_perturbation_registry_v1"
PERTURBATION_BUNDLE_SCHEMA = "shared_driver_r1p_train_only_perturbation_bundle_v1"
TARGET_VERSION = "r1_p_population_frozen_perturbation_full_trajectory_v1"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/physiology_semantic_tokenizer/"
    "r1p_population_frozen_teacher.yaml"
)
DEFAULT_REGISTRY = (
    REPO_ROOT
    / "experiments/configs/physiology_semantic_tokenizer/"
    "r1p_teacher_perturbation_registry.json"
)
DEFAULT_PREVALIDATION_SEAL = (
    REPO_ROOT
    / "docs/physiology_semantic_tokenizer/architecture/"
    "r1p_prevalidation_seal.json"
)
TRAIN_UNIVERSE = frozenset(f"subject_{index:02d}" for index in range(1, 19))
MECHANICAL_AMENDMENT_SEAL_STATUS = (
    "sealed_mechanical_serialization_amendment_after_uninspected_validation_compute"
)
MECHANICAL_AMENDMENT_KIND = "numpy_json_serialization_only"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _subject_keys(subject_ids: Sequence[str]) -> set[str]:
    return {f"eeg_fnirs_single_trial|{value}" for value in subject_ids}


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


def load_prevalidation_seal(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    if path != DEFAULT_PREVALIDATION_SEAL.resolve():
        raise RuntimeError(
            "Perturbation builder requires the tracked default prevalidation seal"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_prevalidation_seal_state(payload)
    for item in payload["sealed_files"]:
        source = REPO_ROOT / str(item["path"])
        if not source.is_file() or _sha256(source) != item["sha256"]:
            raise RuntimeError(f"Prevalidation-sealed source changed: {item['path']}")
    return payload, _sha256(path)


def verify_builder_sealed_inputs(
    seal: Mapping[str, Any],
    *,
    config_path: Path,
    registry_path: Path,
    builder_path: Path | None = None,
) -> dict[str, bool]:
    sealed = {str(item["role"]): item for item in seal["sealed_files"]}
    checks = {
        "teacher_config": _sha256(Path(config_path).resolve())
        == sealed["teacher_config"]["sha256"],
        "perturbation_registry": _sha256(Path(registry_path).resolve())
        == sealed["perturbation_registry"]["sha256"],
        "perturbation_builder": _sha256(
            Path(builder_path or __file__).resolve()
        )
        == sealed["perturbation_builder"]["sha256"],
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(
            "Perturbation builder CLI/source differs from prevalidation seal: "
            f"{failed}"
        )
    return checks


def reverify_builder_seal_before_validation(
    *,
    prevalidation_seal_path: Path,
    expected_prevalidation_seal_sha256: str,
    expected_input_checks: Mapping[str, bool],
    config_path: Path,
    registry_path: Path,
) -> dict[str, bool]:
    """Fail closed on any seal/input change since train-only fitting began."""

    prevalidation_seal_path = Path(prevalidation_seal_path).resolve()
    if _sha256(prevalidation_seal_path) != expected_prevalidation_seal_sha256:
        raise RuntimeError("Prevalidation seal changed before validation load")
    seal, seal_sha = load_prevalidation_seal(prevalidation_seal_path)
    if seal_sha != expected_prevalidation_seal_sha256:
        raise RuntimeError("Reloaded prevalidation seal hash changed")
    repeated_checks = verify_builder_sealed_inputs(
        seal,
        config_path=config_path,
        registry_path=registry_path,
    )
    if repeated_checks != dict(expected_input_checks):
        raise RuntimeError("Sealed builder input checks changed before validation load")
    return repeated_checks


def load_perturbation_registry(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PERTURBATION_REGISTRY_SCHEMA:
        raise ValueError("R1-P perturbation registry schema mismatch")
    if payload.get("status") != "frozen_before_formal_validation_results":
        raise ValueError("R1-P perturbation registry is not frozen")
    if int(payload.get("revision", -1)) != 3:
        raise ValueError("R1-P perturbation registry revision mismatch")
    common = payload["common_contract"]
    if bool(common.get("protected_open", True)):
        raise PermissionError("Perturbation registry opens the protected split")
    if common.get("allowed_fit_subject_universe") != "subject_01..subject_18":
        raise ValueError("Perturbation fit universe drifted")
    if common.get("pure_apply_subjects") != "subject_19..subject_23":
        raise ValueError("Perturbation validation universe drifted")
    if common.get("protected_subjects") != "subject_24..subject_29":
        raise ValueError("Perturbation protected universe drifted")
    if set(common.get("fit_conditions", [])) != {"BL", "MA"}:
        raise ValueError("Perturbation condition contract drifted")
    if set(common.get("fit_sessions", [])) != {
        "session_01",
        "session_03",
        "session_05",
    }:
        raise ValueError("Perturbation session contract drifted")
    if int(common.get("windows_per_retained_subject", -1)) != 60:
        raise ValueError("Perturbation per-subject window contract drifted")

    perturbations = list(payload.get("perturbations", []))
    if len(perturbations) != 3:
        raise ValueError("Exactly three registered perturbations are required")
    if len({item["perturbation_id"] for item in perturbations}) != 3:
        raise ValueError("Perturbation identifiers must be unique")
    if len({item["output_name"] for item in perturbations}) != 3:
        raise ValueError("Perturbation output names must be unique")
    anchor_sessions = []
    for definition in perturbations:
        retained = set(map(str, definition["retained_fit_subjects"]))
        excluded = set(map(str, definition["excluded_fit_subjects"]))
        if retained | excluded != set(TRAIN_UNIVERSE) or retained & excluded:
            raise ValueError("Perturbation train partition is not exact")
        if len(retained) != 15 or len(excluded) != 3:
            raise ValueError("Each perturbation must retain 15 and exclude 3 subjects")
        if int(definition["expected_fit_windows"]) != 900:
            raise ValueError("Each perturbation must register 900 fit windows")
        anchor = definition["anchor_rows"]
        sessions = tuple(map(str, anchor["sessions"]))
        if (
            anchor.get("condition") != "MA"
            or len(sessions) != 1
            or sessions[0] not in common["fit_sessions"]
            or int(anchor.get("expected_windows", -1)) != 150
        ):
            raise ValueError("Perturbation anchor contract must be 150 MA rows")
        anchor_sessions.extend(sessions)
    if len(set(anchor_sessions)) != 3:
        raise ValueError("Registered perturbations must use three distinct anchor sessions")
    return payload, _sha256(path)


def select_definition(
    registry: Mapping[str, Any], perturbation_id: str
) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in registry["perturbations"]
        if str(item["perturbation_id"]) == str(perturbation_id)
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown perturbation_id: {perturbation_id}")
    return matches[0]


def assert_requested_subjects(subject_ids: Sequence[str], *, role: str) -> set[str]:
    subjects = {str(value) for value in subject_ids}
    protected = sorted(subjects & set(PROTECTED_SUBJECT_IDS))
    if protected:
        raise PermissionError(
            "Protected subjects rejected before measured-array access: "
            f"{protected}"
        )
    expected = (
        set(VALIDATION_SUBJECT_IDS)
        if role == "validation_pure_apply"
        else set(TRAIN_UNIVERSE)
        if role == "train_fit_universe"
        else None
    )
    if expected is None or not subjects <= expected:
        raise ValueError(f"Invalid subject request for role {role!r}")
    return subjects


def load_perturbation_trials(
    config: Mapping[str, Any],
    *,
    subject_ids: Sequence[str],
    development_role: str,
) -> tuple[list[PopulationTrial], dict[str, Any]]:
    """Load only explicitly authorized measured rows for one perturbation phase."""

    base_contract = validate_population_config(config)
    requested = {str(value) for value in subject_ids}
    if development_role == "train_fit":
        if len(requested) != 15 or not requested <= set(TRAIN_UNIVERSE):
            raise ValueError("Perturbation fitting requires exactly 15 train subjects")
    elif development_role == "validation_pure_apply":
        if requested != set(VALIDATION_SUBJECT_IDS):
            raise ValueError("Perturbation pure apply requires subjects 19-23")
    else:
        raise ValueError(f"Unsupported development role: {development_role!r}")
    assert_requested_subjects(
        requested,
        role=(
            "validation_pure_apply"
            if development_role == "validation_pure_apply"
            else "train_fit_universe"
        ),
    )

    data = config["data"]
    dataset = UnifiedPhysiologyWindowDataset(
        cache_root=data["cache_root"],
        dataset_ids=(base_contract["dataset_id"],),
        window_duration_s=float(data["window_duration_s"]),
        window_offset_s=float(data["window_offset_s"]),
        eeg_signal_branch=str(data["eeg_signal_branch"]),
    )
    allowed_keys = _subject_keys(sorted(requested))
    lookup = _condition_lookup(config)
    selected: list[tuple[int, str, str, str]] = []
    for index, ref in enumerate(dataset.windows):
        subject = str(ref.record.canonical_subject_id)
        subject_key = f"{ref.record.dataset_id}|{subject}"
        if subject_key not in allowed_keys:
            continue
        record_id = str(ref.record.base_record_id)
        condition = str(ref.event.get("label"))
        condition_id = lookup.get((record_id, condition))
        if condition_id is not None:
            selected.append((index, condition_id, condition, subject_key))

    baseline_n = int(round(float(data["baseline_duration_s"]) * 10.0))
    trials: list[PopulationTrial] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    for index, condition_id, condition, subject_key in selected:
        sample = dataset[index]
        subject = str(sample["subject"])
        if subject in PROTECTED_SUBJECT_IDS:
            raise RuntimeError("Protected measured array was dereferenced")
        eeg_valid = np.asarray(sample["valid_mask"]["eeg"], dtype=bool)
        fnirs_valid = np.asarray(sample["valid_mask"]["fnirs"], dtype=bool)
        eeg = np.asarray(sample["eeg"], dtype=np.float64).T
        fnirs = np.asarray(sample["fnirs"], dtype=np.float64).T
        if not eeg_valid.all() or not fnirs_valid.all():
            raise RuntimeError("Registered perturbation window has boundary padding")
        if not np.isfinite(eeg).all() or not np.isfinite(fnirs).all():
            raise RuntimeError("Registered perturbation window is non-finite")
        fnirs = fnirs - fnirs[:baseline_n].mean(axis=0, keepdims=True)
        trials.append(
            PopulationTrial(
                condition_id=condition_id,
                condition=condition,
                dataset_id=str(sample["dataset_id"]),
                subject=subject,
                subject_key=subject_key,
                development_role=development_role,
                record_id=str(sample["record_id"]),
                event_index=int(sample["event"].get("event_index")),
                eeg=eeg,
                fnirs=fnirs,
                eeg_channel_names=tuple(map(str, sample["channel_names"]["eeg"])),
                fnirs_channel_names=tuple(map(str, sample["channel_names"]["fnirs"])),
                fnirs_roles=tuple(map(str, sample["component_roles"]["fnirs"])),
                eeg_positions=np.asarray(
                    [
                        [row.get(axis, np.nan) for axis in ("x", "y", "z")]
                        for row in sample["channel_geometry"]["eeg"]
                    ],
                    dtype=np.float64,
                ),
                fnirs_positions=np.asarray(
                    [
                        [row.get(axis, np.nan) for axis in ("x", "y", "z")]
                        for row in sample["channel_geometry"]["fnirs"]
                    ],
                    dtype=np.float64,
                ),
                eeg_bad_channel_mask=np.asarray(
                    sample["bad_channel_mask"]["eeg"], dtype=bool
                ),
                fnirs_bad_channel_mask=np.asarray(
                    sample["bad_channel_mask"]["fnirs"], dtype=bool
                ),
            )
        )
        counts[(subject, str(sample["record_id"]), condition)] += 1

    expected_registry = data["expected_registry"]
    expected_pairs = {
        (str(session), str(condition))
        for session in expected_registry["sessions"]
        for condition in expected_registry["conditions"]
    }
    expected_per_cell = int(
        expected_registry["windows_per_subject_session_condition"]
    )
    expected_cells = {
        (subject, session, condition)
        for subject in requested
        for session, condition in expected_pairs
    }
    expected_count = len(requested) * len(expected_pairs) * expected_per_cell
    if len(trials) != expected_count:
        raise RuntimeError(
            f"{development_role} row count mismatch: {len(trials)} != {expected_count}"
        )
    if set(counts) != expected_cells or any(
        counts[cell] != expected_per_cell for cell in expected_cells
    ):
        raise RuntimeError("Perturbation session/condition coverage is incomplete")
    trials.sort(
        key=lambda trial: (
            trial.subject,
            trial.record_id,
            trial.condition,
            trial.event_index,
        )
    )
    if len({trial.sample_key for trial in trials}) != len(trials):
        raise RuntimeError("Perturbation sample identities are not unique")
    return trials, {
        "development_role": development_role,
        "sample_count": len(trials),
        "subject_keys": sorted(allowed_keys),
        "session_condition_cells": len(counts),
        "all_boundary_valid": True,
        "all_measurements_finite": True,
        "protected_array_dereference_count": 0,
        "sample_order_sha256": hashlib.sha256(
            "\n".join(trial.sample_key for trial in trials).encode("utf-8")
        ).hexdigest(),
    }


def validate_fit_trials(
    trials: Sequence[PopulationTrial], definition: Mapping[str, Any]
) -> dict[str, Any]:
    retained = set(map(str, definition["retained_fit_subjects"]))
    expected_keys = _subject_keys(sorted(retained))
    if len(trials) != int(definition["expected_fit_windows"]):
        raise ValueError("Perturbation fit row count differs from registry")
    if any(trial.development_role != "train_fit" for trial in trials):
        raise ValueError("Perturbation fit accepts train_fit rows only")
    if {trial.subject_key for trial in trials} != expected_keys:
        raise ValueError("Perturbation fit subject cohort differs from registry")
    if any(trial.subject in PROTECTED_SUBJECT_IDS for trial in trials):
        raise PermissionError("Protected row entered perturbation fitting")
    anchor = definition["anchor_rows"]
    anchor_trials = [
        trial
        for trial in trials
        if trial.condition == str(anchor["condition"])
        and trial.record_id in set(map(str, anchor["sessions"]))
    ]
    if len(anchor_trials) != int(anchor["expected_windows"]):
        raise ValueError("Perturbation anchor rows differ from registry")
    return {
        "fit_subject_keys": sorted(expected_keys),
        "fit_sample_count": len(trials),
        "anchor_fit_sample_count": len(anchor_trials),
        "anchor_fit_sessions": list(map(str, anchor["sessions"])),
        "anchor_fit_condition": str(anchor["condition"]),
    }


def fit_perturbation_bundle(
    trials: Sequence[PopulationTrial],
    config: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> PopulationFrozenBundle:
    """Fit anchor, EEG projection, and SSM from the exact retained cohort."""

    validate_fit_trials(trials, definition)
    _assert_common_channel_contract(trials)
    anchor = definition["anchor_rows"]
    anchor_trials = [
        trial
        for trial in trials
        if trial.condition == str(anchor["condition"])
        and trial.record_id in set(map(str, anchor["sessions"]))
    ]
    analysis = config["analysis"]
    data = config["data"]
    with threadpool_limits(limits=1):
        hbo_indices, hbo_names, _ = _select_active_hbo(
            anchor_trials,
            baseline_duration_s=float(data["baseline_duration_s"]),
            task_duration_s=float(data["task_duration_s"]),
            count=int(analysis["fnirs_active_hbo_channels"]),
        )
        hbr_indices = _paired_hbr_indices(anchor_trials[0], hbo_indices)
        hbr_names = tuple(
            anchor_trials[0].fnirs_channel_names[int(index)]
            for index in hbr_indices
        )
        eeg_indices = _local_eeg_indices(
            anchor_trials[0],
            hbo_indices,
            int(analysis["local_eeg_channels"]),
        )
        if any(
            trial.eeg_bad_channel_mask[eeg_indices].any()
            or trial.fnirs_bad_channel_mask[hbo_indices].any()
            or trial.fnirs_bad_channel_mask[hbr_indices].any()
            for trial in trials
        ):
            raise RuntimeError("Perturbation-selected view contains a rejected channel")
        adapter, drivers = _fit_eeg_adapter(trials, eeg_indices)
        hbo, hbr = _chromophore_targets(
            trials, hbo_indices, hbr_indices
        )
        fit = _fit_model(
            drivers,
            hbo,
            hbr,
            analysis["ssm"],
            int(
                round(
                    float(data["baseline_duration_s"])
                    * float(analysis["ssm"]["fs_hz"])
                )
            ),
        )
    return PopulationFrozenBundle(
        adapter=adapter,
        fit=fit,
        selected_hbo_indices=np.asarray(hbo_indices, dtype=int),
        selected_hbr_indices=np.asarray(hbr_indices, dtype=int),
        selected_fnirs_channels=tuple(hbo_names + hbr_names),
        anchor_id=_anchor_id(hbo_names[0]),
        normalization={},
        fit_subject_keys=tuple(
            sorted({trial.subject_key for trial in trials})
        ),
        fit_sample_order_sha256=hashlib.sha256(
            "\n".join(trial.sample_key for trial in trials).encode("utf-8")
        ).hexdigest(),
    )


def fit_perturbation_normalization(
    trials: Sequence[PopulationTrial],
    results: Sequence[PairedDriverResult],
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    validate_fit_trials(trials, definition)
    if len(trials) != len(results):
        raise ValueError("Perturbation normalization row/result count differs")
    values = np.concatenate(
        [np.asarray(result.joint, dtype=np.float64) for result in results]
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Perturbation train driver is non-finite")
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-6:
        raise RuntimeError("Perturbation scalar gauge is degenerate")
    payload = {
        "policy": "scalar_joint_train_subject_points_v1",
        "coordinate": "adaptive_state_index_4_shared_driver",
        "fit_subject_keys": sorted({trial.subject_key for trial in trials}),
        "fit_sample_count": len(trials),
        "fit_point_count": int(values.size),
        "mean": mean,
        "scale": scale,
        "applied_identically_to": [
            "target_shared_driver",
            "target_eeg_only_driver",
        ],
        "validation_subjects_used": False,
        "protected_subjects_used": False,
    }
    payload["sha256"] = _json_sha256(payload)
    return payload


def definition_source(
    *,
    config_path: Path,
    registry_path: Path,
    registry_sha256: str,
    definition: Mapping[str, Any],
    prevalidation_seal_path: Path,
    prevalidation_seal_sha256: str,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    registry_path = Path(registry_path).resolve()
    prevalidation_seal_path = Path(prevalidation_seal_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_cache_root = Path(str(config["data"]["cache_root"]))
    cache_root = (
        configured_cache_root
        if configured_cache_root.is_absolute()
        else REPO_ROOT / configured_cache_root
    )
    source_paths = {
        "config": config_path,
        "perturbation_registry": registry_path,
        "prevalidation_seal": prevalidation_seal_path,
        "builder": Path(__file__).resolve(),
        "base_builder": (
            REPO_ROOT
            / "experiments/scripts/build_r1p_population_frozen_teacher.py"
        ),
        "adaptive_evaluator": (
            REPO_ROOT / "experiments/evaluate_adaptive_shared_neural_ssm.py"
        ),
        "adaptive_solver": (
            REPO_ROOT / "src/inference/adaptive_neurovascular_ssm.py"
        ),
        "clean_cache_manifest": cache_root / "cache_manifest.json",
        "event_manifest": cache_root / "event_index/event_manifest.json",
        "geometry_manifest": (
            cache_root / "channel_geometry/geometry_manifest.json"
        ),
        "eeg_artifact_manifest": (
            cache_root / "eeg_artifact_clean_v4/cache_manifest.json"
        ),
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Perturbation provenance input missing: {missing}")
    return {
        "perturbation_id": str(definition["perturbation_id"]),
        "perturbation_registry_path": str(registry_path),
        "perturbation_registry_sha256": str(registry_sha256),
        "perturbation_definition_sha256": _json_sha256(definition),
        "prevalidation_seal_path": str(prevalidation_seal_path),
        "prevalidation_seal_sha256": str(prevalidation_seal_sha256),
        "anchor_fit_sessions": list(
            map(str, definition["anchor_rows"]["sessions"])
        ),
        "anchor_fit_condition": str(definition["anchor_rows"]["condition"]),
        "input_hashes": {
            name: _sha256(path) for name, path in sorted(source_paths.items())
        },
        "raw_source_provenance": {
            "cache_root": str(cache_root),
            "eeg_signal_branch": str(config["data"]["eeg_signal_branch"]),
            "window_duration_s": float(config["data"]["window_duration_s"]),
            "window_offset_s": float(config["data"]["window_offset_s"]),
            "baseline_duration_s": float(config["data"]["baseline_duration_s"]),
        },
        **_git_state(),
    }


def save_perturbation_parameter_bundle(
    root: Path,
    bundle: PopulationFrozenBundle,
    *,
    source: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> str:
    expected_keys = tuple(
        sorted(_subject_keys(definition["retained_fit_subjects"]))
    )
    if tuple(bundle.fit_subject_keys) != expected_keys:
        raise ValueError("Perturbation parameter fit cohort differs from registry")
    normalization = dict(bundle.normalization)
    if (
        int(normalization.get("fit_sample_count", -1))
        != int(definition["expected_fit_windows"])
        or tuple(sorted(normalization.get("fit_subject_keys", [])))
        != expected_keys
        or bool(normalization.get("validation_subjects_used", True))
        or bool(normalization.get("protected_subjects_used", True))
    ):
        raise ValueError("Perturbation parameter gauge provenance is unsafe")
    root.mkdir()
    arrays_path = root / "arrays.npz"
    np.savez_compressed(arrays_path, **_parameter_arrays(bundle))
    fit_mapping = {
        key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
        for key, value in fit_to_mapping(bundle.fit).items()
    }
    identity = {
        "arrays_sha256": _sha256(arrays_path),
        "fit_scalars": fit_mapping,
        "normalization": normalization,
        "fit_subject_keys": list(bundle.fit_subject_keys),
        "fit_sample_order_sha256": bundle.fit_sample_order_sha256,
        "selected_fnirs_channels": list(bundle.selected_fnirs_channels),
        "anchor_id": bundle.anchor_id,
    }
    bundle_sha = _json_sha256(identity)
    _write_json(
        root / "manifest.json",
        {
            "schema": PARAMETER_MANIFEST_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "teacher_scope": TEACHER_SCOPE,
            "perturbation_id": definition["perturbation_id"],
            "output_name": definition["output_name"],
            "perturbation_registry_sha256": source[
                "perturbation_registry_sha256"
            ],
            "perturbation_definition_sha256": source[
                "perturbation_definition_sha256"
            ],
            "prevalidation_seal_path": source["prevalidation_seal_path"],
            "prevalidation_seal_sha256": source[
                "prevalidation_seal_sha256"
            ],
            "parameter_scope": (
                f"{definition['perturbation_id']}:fit_15_train_apply_5_validation"
            ),
            "arrays_file": arrays_path.name,
            "arrays_sha256": identity["arrays_sha256"],
            "fit_scalars": fit_mapping,
            "normalization": normalization,
            "fit_subject_keys": list(bundle.fit_subject_keys),
            "fit_sample_order_sha256": bundle.fit_sample_order_sha256,
            "selected_fnirs_channels": list(bundle.selected_fnirs_channels),
            "selected_eeg_channels": list(bundle.adapter.channel_names),
            "anchor_id": bundle.anchor_id,
            "bundle_sha256": bundle_sha,
            "validation_subjects_used_for_any_fit": False,
            "protected_subjects_used_for_any_fit": False,
            "joint_and_eeg_only_share_exact_bundle": True,
            "eeg_only_difference": "fNIRS observation update omitted at apply only",
            "source": dict(source),
        },
    )
    return bundle_sha


def write_perturbation_artifacts(
    root: Path,
    trials: Sequence[PopulationTrial],
    results: Sequence[PairedDriverResult],
    bundle: PopulationFrozenBundle,
    *,
    source: Mapping[str, Any],
    definition: Mapping[str, Any],
    train_audit: Mapping[str, Any],
    validation_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if len(trials) != len(results):
        raise ValueError("Perturbation artifact trial/result count mismatch")
    train_trials = [
        trial for trial in trials if trial.development_role == "train_fit"
    ]
    validation_trials = [
        trial
        for trial in trials
        if trial.development_role == "validation_pure_apply"
    ]
    if len(train_trials) != int(definition["expected_fit_windows"]):
        raise ValueError("Perturbation artifact train row count is not 900")
    if {trial.subject for trial in train_trials} != set(
        map(str, definition["retained_fit_subjects"])
    ):
        raise ValueError("Perturbation artifact train cohort differs from registry")
    expected_validation_count = len(VALIDATION_SUBJECT_IDS) * 60
    if len(validation_trials) != expected_validation_count:
        raise ValueError("Perturbation artifact validation row count is not 300")
    if {trial.subject for trial in validation_trials} != set(
        VALIDATION_SUBJECT_IDS
    ):
        raise ValueError(
            "Perturbation artifact validation cohort differs from pure-apply split"
        )
    if len(train_trials) + len(validation_trials) != len(trials):
        raise ValueError("Perturbation artifact contains an unregistered row role")
    sample_keys = [trial.sample_key for trial in trials]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Perturbation artifact sample keys are not unique")
    if any(trial.subject in PROTECTED_SUBJECT_IDS for trial in trials):
        raise PermissionError("Protected row entered perturbation artifact writer")
    normalized = [
        _normalized_pair(result, bundle.normalization) for result in results
    ]
    joint = np.stack([item[0] for item in normalized])
    joint_std = np.stack([item[1] for item in normalized])
    eeg_only = np.stack([item[2] for item in normalized])
    eeg_only_std = np.stack([item[3] for item in normalized])
    point_mask = np.ones_like(joint, dtype=bool)
    split_names = [
        "train" if trial.development_role == "train_fit" else "validation"
        for trial in trials
    ]
    order_sha = hashlib.sha256("\n".join(sample_keys).encode("utf-8")).hexdigest()
    gauge_hash = str(bundle.normalization["sha256"])
    parameter_fold = (
        f"population_frozen:{definition['perturbation_id']}:"
        f"fit_subjects_15:{bundle.bundle_sha256}"
    )
    source_hashes = [
        _json_sha256(
            {
                "bundle_sha256": bundle.bundle_sha256,
                "sample_key": trial.sample_key,
                "teacher_input_sha256": result.teacher_input_sha256,
            }
        )
        for trial, result in zip(trials, results)
    ]
    time_s = -5.0 + np.arange(POINT_COUNT, dtype=np.float32).reshape(
        PATCH_COUNT, POINTS_PER_PATCH
    ) / 10.0
    split_payload = {
        "fit_subject_keys": list(bundle.fit_subject_keys),
        "validation_subject_keys": sorted(
            {
                trial.subject_key
                for trial in trials
                if trial.development_role == "validation_pure_apply"
            }
        ),
        "protected_open": False,
        "validation_pure_apply": True,
    }

    teacher_root = root / "trajectory_targets"
    teacher_root.mkdir()
    teacher_arrays = teacher_root / "arrays.npz"
    np.savez_compressed(
        teacher_arrays,
        schema=np.asarray(SHARED_DRIVER_ARRAY_SCHEMA),
        sample_key=string_array(sample_keys),
        dataset_id=string_array([trial.dataset_id for trial in trials]),
        subject_id=string_array([trial.subject for trial in trials]),
        subject_key=string_array([trial.subject_key for trial in trials]),
        session_id=string_array([trial.record_id for trial in trials]),
        condition=string_array([trial.condition for trial in trials]),
        condition_id=string_array([trial.condition_id for trial in trials]),
        event_index=np.asarray([trial.event_index for trial in trials], dtype=np.int64),
        development_split=string_array(split_names),
        parameter_role=string_array(
            [trial.development_role for trial in trials]
        ),
        target_time_s=np.broadcast_to(
            time_s, (len(trials), PATCH_COUNT, POINTS_PER_PATCH)
        ).copy(),
        target_shared_driver=joint,
        target_shared_driver_std=joint_std,
        target_valid_mask=point_mask.any(axis=-1),
        target_point_valid_mask=point_mask,
        target_eeg_only_driver=eeg_only,
        target_eeg_only_driver_std=eeg_only_std,
        eeg_only_valid_mask=point_mask.any(axis=-1),
        eeg_only_point_valid_mask=point_mask,
        teacher_scope=string_array([TEACHER_SCOPE] * len(trials)),
        teacher_parameter_fold=string_array([parameter_fold] * len(trials)),
        teacher_gauge_hash=string_array([gauge_hash] * len(trials)),
        teacher_source_hash=string_array(source_hashes),
        parameter_bundle_sha256=string_array(
            [bundle.bundle_sha256] * len(trials)
        ),
    )
    teacher_manifest = {
        "schema": SHARED_DRIVER_SIDECAR_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_generation": ARCHITECTURE_GENERATION,
        "target_family": TARGET_FAMILY,
        "target_version": TARGET_VERSION,
        "teacher_scope": TEACHER_SCOPE,
        "perturbation_id": definition["perturbation_id"],
        "output_name": definition["output_name"],
        "perturbation_registry_sha256": source[
            "perturbation_registry_sha256"
        ],
        "perturbation_definition_sha256": source[
            "perturbation_definition_sha256"
        ],
        "prevalidation_seal_path": source["prevalidation_seal_path"],
        "prevalidation_seal_sha256": source["prevalidation_seal_sha256"],
        "arrays_file": teacher_arrays.name,
        "arrays_sha256": _sha256(teacher_arrays),
        "sample_count": len(trials),
        "sample_order_sha256": order_sha,
        "trajectory_shape": [PATCH_COUNT, POINTS_PER_PATCH],
        "parameter_bundle_sha256": bundle.bundle_sha256,
        "normalization": dict(bundle.normalization),
        "paired_control": {
            "joint_and_eeg_only_exact_parameter_bundle": True,
            "only_difference": "fNIRS observation update omitted for rE",
            "parameter_bundle_sha256": bundle.bundle_sha256,
            "shared_driver_gauge_sha256": gauge_hash,
        },
        "source": dict(source),
        "split": split_payload,
        "promotion_eligible": False,
        "promotion_blocker": "perturbation_control_only_not_promotable",
        "protected_open": False,
        "protected_test_included": False,
        **_git_state(),
    }
    _write_json(teacher_root / "manifest.json", teacher_manifest)

    raw_root = root / "raw_view_registry"
    raw_root.mkdir()
    raw_arrays = raw_root / "arrays.npz"
    np.savez_compressed(
        raw_arrays,
        schema=np.asarray(RAW_VIEW_ARRAY_SCHEMA),
        sample_key=string_array(sample_keys),
        dataset_id=string_array([trial.dataset_id for trial in trials]),
        subject_id=string_array([trial.subject for trial in trials]),
        subject_key=string_array([trial.subject_key for trial in trials]),
        session_id=string_array([trial.record_id for trial in trials]),
        condition=string_array([trial.condition for trial in trials]),
        event_index=np.asarray([trial.event_index for trial in trials], dtype=np.int64),
        selected_eeg_channels=np.asarray(
            [bundle.adapter.channel_names] * len(trials), dtype=np.str_
        ),
        selected_fnirs_channels=np.asarray(
            [bundle.selected_fnirs_channels] * len(trials), dtype=np.str_
        ),
        anchor_id=string_array([bundle.anchor_id] * len(trials)),
        selection_fold=string_array([parameter_fold] * len(trials)),
        selection_source_hash=string_array(
            [bundle.bundle_sha256] * len(trials)
        ),
    )
    raw_manifest = {
        "schema": RAW_VIEW_REGISTRY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_generation": ARCHITECTURE_GENERATION,
        "raw_view_policy": "train_only_perturbation_anchor_and_eeg_projection_v1",
        "selection_scope": TEACHER_SCOPE,
        "perturbation_id": definition["perturbation_id"],
        "output_name": definition["output_name"],
        "perturbation_registry_sha256": source[
            "perturbation_registry_sha256"
        ],
        "perturbation_definition_sha256": source[
            "perturbation_definition_sha256"
        ],
        "prevalidation_seal_path": source["prevalidation_seal_path"],
        "prevalidation_seal_sha256": source["prevalidation_seal_sha256"],
        "arrays_file": raw_arrays.name,
        "arrays_sha256": _sha256(raw_arrays),
        "sample_count": len(trials),
        "sample_order_sha256": order_sha,
        "selected_eeg_channels": list(bundle.adapter.channel_names),
        "selected_fnirs_channels": list(bundle.selected_fnirs_channels),
        "anchor_id": bundle.anchor_id,
        "parameter_bundle_sha256": bundle.bundle_sha256,
        "source": dict(source),
        "split": split_payload,
        "promotion_eligible": False,
        "protected_open": False,
        "protected_test_included": False,
        **_git_state(),
    }
    _write_json(raw_root / "manifest.json", raw_manifest)

    leakage = {
        "schema": "shared_driver_r1p_perturbation_leakage_audit_v1",
        "perturbation_id": definition["perturbation_id"],
        "output_name": definition["output_name"],
        "prevalidation_seal_path": source["prevalidation_seal_path"],
        "prevalidation_seal_sha256": source["prevalidation_seal_sha256"],
        "fit_subject_keys": list(bundle.fit_subject_keys),
        "fit_sample_count": int(definition["expected_fit_windows"]),
        "anchor_fit_sessions": list(definition["anchor_rows"]["sessions"]),
        "anchor_fit_condition": definition["anchor_rows"]["condition"],
        "validation_loaded_after_parameter_and_normalization_freeze": True,
        "validation_fit_calls": 0,
        "validation_normalization_calls": 0,
        "protected_array_dereference_count": 0,
        "protected_open": False,
        "joint_and_eeg_only_exact_parameter_bundle": True,
        "raw_view_and_target_separate": True,
        "train_registry_audit": dict(train_audit),
        "validation_registry_audit": dict(validation_audit),
        "source": dict(source),
    }
    _write_json(root / "leakage_audit.json", leakage)
    _write_json(
        root / "perturbation_definition.json",
        {
            "definition": dict(definition),
            "definition_sha256": _json_sha256(definition),
            "registry_sha256": source["perturbation_registry_sha256"],
            "prevalidation_seal_sha256": source[
                "prevalidation_seal_sha256"
            ],
        },
    )
    top = {
        "schema": PERTURBATION_BUNDLE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_generation": ARCHITECTURE_GENERATION,
        "teacher_scope": TEACHER_SCOPE,
        "target_family": TARGET_FAMILY,
        "target_version": TARGET_VERSION,
        "perturbation_id": definition["perturbation_id"],
        "output_name": definition["output_name"],
        "perturbation_registry_sha256": source[
            "perturbation_registry_sha256"
        ],
        "perturbation_definition_sha256": source[
            "perturbation_definition_sha256"
        ],
        "prevalidation_seal_path": source["prevalidation_seal_path"],
        "prevalidation_seal_sha256": source["prevalidation_seal_sha256"],
        "sample_count": len(trials),
        "train_sample_count": sum(value == "train" for value in split_names),
        "validation_sample_count": sum(
            value == "validation" for value in split_names
        ),
        "parameter_bundle": {
            "path": "parameter_bundle",
            "bundle_sha256": bundle.bundle_sha256,
            "manifest_sha256": _sha256(root / "parameter_bundle/manifest.json"),
        },
        "trajectory_targets": {
            "path": "trajectory_targets",
            "manifest_sha256": _sha256(teacher_root / "manifest.json"),
            "arrays_sha256": teacher_manifest["arrays_sha256"],
        },
        "raw_view_registry": {
            "path": "raw_view_registry",
            "manifest_sha256": _sha256(raw_root / "manifest.json"),
            "arrays_sha256": raw_manifest["arrays_sha256"],
        },
        "normalization": dict(bundle.normalization),
        "leakage_audit": "leakage_audit.json",
        "source": dict(source),
        "promotion_eligible": False,
        "promotion_blocker": "perturbation_control_only_not_promotable",
        "protected_open": False,
        "protected_test_included": False,
        **_git_state(),
    }
    _write_json(root / "manifest.json", top)
    return top


def build_one_perturbation(
    *,
    config_path: Path,
    registry_path: Path,
    perturbation_id: str,
    output_parent: Path,
    prevalidation_seal_path: Path = DEFAULT_PREVALIDATION_SEAL,
    expected_registry_sha256: str | None = None,
    expected_prevalidation_seal_sha256: str | None = None,
) -> Path:
    """Build one registered bundle atomically; never reads the base bundle."""

    config_path = Path(config_path).resolve()
    registry_path = Path(registry_path).resolve()
    output_parent = Path(output_parent).resolve()
    prevalidation_seal_path = Path(prevalidation_seal_path).resolve()
    seal, seal_sha = load_prevalidation_seal(prevalidation_seal_path)
    seal_input_checks = verify_builder_sealed_inputs(
        seal,
        config_path=config_path,
        registry_path=registry_path,
    )
    if (
        expected_prevalidation_seal_sha256 is not None
        and seal_sha != str(expected_prevalidation_seal_sha256)
    ):
        raise RuntimeError(
            "Prevalidation seal changed after this build invocation was sealed"
        )
    registry, registry_sha = load_perturbation_registry(registry_path)
    if (
        expected_registry_sha256 is not None
        and registry_sha != str(expected_registry_sha256)
    ):
        raise RuntimeError(
            "Perturbation registry changed after this build invocation was sealed"
        )
    definition = select_definition(registry, perturbation_id)
    output_root = output_parent / str(definition["output_name"])
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite perturbation: {output_root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_population_config(config)
    source = definition_source(
        config_path=config_path,
        registry_path=registry_path,
        registry_sha256=registry_sha,
        definition=definition,
        prevalidation_seal_path=prevalidation_seal_path,
        prevalidation_seal_sha256=seal_sha,
    )

    train_trials, train_audit = load_perturbation_trials(
        config,
        subject_ids=definition["retained_fit_subjects"],
        development_role="train_fit",
    )
    core = fit_perturbation_bundle(train_trials, config, definition)
    with threadpool_limits(limits=1):
        train_results = [apply_paired_driver(trial, core) for trial in train_trials]
    normalization = fit_perturbation_normalization(
        train_trials, train_results, definition
    )
    frozen = replace(core, normalization=normalization)

    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.tmp-", dir=output_parent
    ) as temporary:
        temporary_root = Path(temporary)
        save_perturbation_parameter_bundle(
            temporary_root / "parameter_bundle",
            frozen,
            source=source,
            definition=definition,
        )
        reloaded = load_population_bundle(temporary_root / "parameter_bundle")
        sentinel = apply_paired_driver(train_trials[0], reloaded)
        if not (
            np.array_equal(sentinel.joint, train_results[0].joint)
            and np.array_equal(sentinel.eeg_only, train_results[0].eeg_only)
        ):
            raise RuntimeError("Serialized perturbation changed sentinel output")

        reverify_builder_seal_before_validation(
            prevalidation_seal_path=prevalidation_seal_path,
            expected_prevalidation_seal_sha256=seal_sha,
            expected_input_checks=seal_input_checks,
            config_path=config_path,
            registry_path=registry_path,
        )
        validation_trials, validation_audit = load_perturbation_trials(
            config,
            subject_ids=sorted(VALIDATION_SUBJECT_IDS),
            development_role="validation_pure_apply",
        )
        with threadpool_limits(limits=1):
            validation_results = [
                apply_paired_driver(trial, reloaded)
                for trial in validation_trials
            ]
        write_perturbation_artifacts(
            temporary_root,
            train_trials + validation_trials,
            train_results + validation_results,
            reloaded,
            source=source,
            definition=definition,
            train_audit=train_audit,
            validation_audit=validation_audit,
        )
        temporary_root.replace(output_root)
    return output_root


def registry_audit(
    config_path: Path,
    registry_path: Path,
    prevalidation_seal_path: Path = DEFAULT_PREVALIDATION_SEAL,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    registry_path = Path(registry_path).resolve()
    prevalidation_seal_path = Path(prevalidation_seal_path).resolve()
    seal, seal_sha = load_prevalidation_seal(prevalidation_seal_path)
    seal_checks = verify_builder_sealed_inputs(
        seal,
        config_path=config_path,
        registry_path=registry_path,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_population_config(config)
    registry, registry_sha = load_perturbation_registry(registry_path)
    return {
        "schema": "r1p_train_only_perturbation_builder_audit_v1",
        "registry_path": str(registry_path),
        "registry_sha256": registry_sha,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "prevalidation_seal_path": str(prevalidation_seal_path),
        "prevalidation_seal_sha256": seal_sha,
        "prevalidation_seal_input_checks": seal_checks,
        "perturbations": [
            {
                "perturbation_id": item["perturbation_id"],
                "output_name": item["output_name"],
                "fit_subject_count": len(item["retained_fit_subjects"]),
                "fit_sample_count": item["expected_fit_windows"],
                "anchor_fit_sessions": item["anchor_rows"]["sessions"],
                "anchor_fit_condition": item["anchor_rows"]["condition"],
                "definition_sha256": _json_sha256(item),
            }
            for item in registry["perturbations"]
        ],
        "validation_subjects": sorted(VALIDATION_SUBJECT_IDS),
        "protected_subjects": sorted(PROTECTED_SUBJECT_IDS),
        "protected_open": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument(
        "--prevalidation-seal",
        default=str(DEFAULT_PREVALIDATION_SEAL),
    )
    parser.add_argument("--output-parent")
    parser.add_argument("--perturbation-id", action="append", default=[])
    parser.add_argument("--registry-audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.registry_audit_only:
        print(
            json.dumps(
                registry_audit(
                    Path(args.config),
                    Path(args.registry),
                    Path(args.prevalidation_seal),
                ),
                sort_keys=True,
            )
        )
        return
    if not args.output_parent or not args.perturbation_id:
        raise SystemExit(
            "--output-parent and at least one --perturbation-id are required"
        )
    invocation_seal, invocation_seal_sha = load_prevalidation_seal(
        Path(args.prevalidation_seal)
    )
    verify_builder_sealed_inputs(
        invocation_seal,
        config_path=Path(args.config),
        registry_path=Path(args.registry),
    )
    registry, invocation_registry_sha = load_perturbation_registry(
        Path(args.registry)
    )
    requested = list(args.perturbation_id)
    if requested == ["all"]:
        requested = [item["perturbation_id"] for item in registry["perturbations"]]
    if len(requested) != len(set(requested)):
        raise SystemExit("Duplicate perturbation requests are forbidden")
    definitions = [select_definition(registry, value) for value in requested]
    output_parent = Path(args.output_parent).resolve()
    existing = [
        str(output_parent / definition["output_name"])
        for definition in definitions
        if (output_parent / definition["output_name"]).exists()
    ]
    if existing:
        raise FileExistsError(f"Refusing to overwrite perturbations: {existing}")
    outputs = [
        build_one_perturbation(
            config_path=Path(args.config),
            registry_path=Path(args.registry),
            perturbation_id=value,
            output_parent=output_parent,
            prevalidation_seal_path=Path(args.prevalidation_seal),
            expected_registry_sha256=invocation_registry_sha,
            expected_prevalidation_seal_sha256=invocation_seal_sha,
        )
        for value in requested
    ]
    print(json.dumps({"outputs": [str(value) for value in outputs]}, sort_keys=True))


if __name__ == "__main__":
    main()
