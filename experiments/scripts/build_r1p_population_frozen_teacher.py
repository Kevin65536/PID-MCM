#!/usr/bin/env python3
"""Fit and apply an R1-P development population-frozen teacher bundle.

The only fitting population is subjects 01-18.  Subjects 19-23 are loaded
only after the parameter bundle and shared-driver normalization have been
frozen and serialized.  Subjects 24-29 are forbidden at every entry point.

The registered 1,380-window cohort consists of 10 BL and 10 MA windows from
each of sessions 01, 03, and 05 for 23 development subjects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from threadpoolctl import threadpool_limits


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_adaptive_shared_neural_ssm import (  # noqa: E402
    EEGAdapter,
    _apply_eeg_adapter,
    _chromophore_targets,
    _fit_eeg_adapter,
    _fit_model,
    _local_eeg_indices,
    _paired_hbr_indices,
    _select_active_hbo,
)
from src.data.physiology_semantic_targets import target_sample_key  # noqa: E402
from src.data.shared_driver_targets import (  # noqa: E402
    RAW_VIEW_ARRAY_SCHEMA,
    RAW_VIEW_REGISTRY_SCHEMA,
    SHARED_DRIVER_ARRAY_SCHEMA,
    SHARED_DRIVER_SIDECAR_SCHEMA,
    string_array,
)
from src.data.unified_physiology import UnifiedPhysiologyWindowDataset  # noqa: E402
from src.inference.adaptive_neurovascular_ssm import (  # noqa: E402
    AdaptiveSSMFit,
    HemodynamicParameters,
    apply_adaptive_ssm,
    fit_to_mapping,
)


SCHEMA = "shared_driver_r1p_population_frozen_bundle_v1"
PARAMETER_ARRAY_SCHEMA = "shared_driver_r1p_parameter_arrays_v1"
PARAMETER_MANIFEST_SCHEMA = "shared_driver_r1p_parameter_bundle_v1"
ARCHITECTURE_GENERATION = "shared_driver_semantic_vq_v1"
TARGET_FAMILY = "adaptive_joint_full_trajectory"
TARGET_VERSION = "r1_p_population_frozen_full_trajectory_v1"
TEACHER_SCOPE = "population_frozen"
PATCH_COUNT = 10
POINTS_PER_PATCH = 20
POINT_COUNT = PATCH_COUNT * POINTS_PER_PATCH
TRAIN_SUBJECT_IDS = frozenset(f"subject_{index:02d}" for index in range(1, 19))
VALIDATION_SUBJECT_IDS = frozenset(
    f"subject_{index:02d}" for index in range(19, 24)
)
PROTECTED_SUBJECT_IDS = frozenset(
    f"subject_{index:02d}" for index in range(24, 30)
)


@dataclass(frozen=True)
class PopulationTrial:
    condition_id: str
    condition: str
    dataset_id: str
    subject: str
    subject_key: str
    development_role: str
    record_id: str
    event_index: int
    eeg: np.ndarray
    fnirs: np.ndarray
    eeg_channel_names: tuple[str, ...]
    fnirs_channel_names: tuple[str, ...]
    fnirs_roles: tuple[str, ...]
    eeg_positions: np.ndarray
    fnirs_positions: np.ndarray
    eeg_bad_channel_mask: np.ndarray
    fnirs_bad_channel_mask: np.ndarray

    @property
    def sample_key(self) -> str:
        return target_sample_key(
            self.dataset_id,
            self.subject,
            self.record_id,
            self.event_index,
        )


@dataclass(frozen=True)
class PopulationFrozenBundle:
    adapter: EEGAdapter
    fit: AdaptiveSSMFit
    selected_hbo_indices: np.ndarray
    selected_hbr_indices: np.ndarray
    selected_fnirs_channels: tuple[str, str]
    anchor_id: str
    normalization: Mapping[str, Any]
    fit_subject_keys: tuple[str, ...]
    fit_sample_order_sha256: str
    bundle_sha256: str = ""


@dataclass(frozen=True)
class PairedDriverResult:
    joint: np.ndarray
    joint_std: np.ndarray
    eeg_only: np.ndarray
    eeg_only_std: np.ndarray
    hbo_observed: np.ndarray
    hbr_observed: np.ndarray
    hbo_reconstructed: np.ndarray
    hbr_reconstructed: np.ndarray
    eeg_reconstructed: np.ndarray
    teacher_input_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _subject_parts(subject_key: str) -> tuple[str, str]:
    parts = str(subject_key).split("|", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid subject key: {subject_key!r}")
    return parts[0], parts[1]


def _subject_keys(dataset_id: str, subject_ids: Sequence[str]) -> set[str]:
    return {f"{dataset_id}|{subject}" for subject in subject_ids}


def validate_population_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail before measured-data construction if the population split drifts."""

    data = config["data"]
    experiment = config["experiment"]
    dataset_id = str(data["dataset_id"])
    if not np.isclose(float(data["window_duration_s"]), 20.0):
        raise ValueError("R1-P trajectory contract requires 20-second windows")
    if not np.isclose(float(data["window_offset_s"]), -5.0):
        raise ValueError("R1-P trajectory contract requires window_offset_s=-5")
    if not np.isclose(float(data["baseline_duration_s"]), 5.0):
        raise ValueError("R1-P teacher contract requires a five-second baseline")
    if not np.isclose(float(config["analysis"]["ssm"]["fs_hz"]), 10.0):
        raise ValueError("R1-P trajectory target must remain on the 10 Hz grid")
    if str(data["eeg_signal_branch"]) != "single_trial_eeg_artifact_clean_v4":
        raise ValueError("R1-P-dev requires the frozen clean-v4 EEG branch")
    split = data["split"]
    train = {str(value) for value in split["train_subject_keys"]}
    validation = {str(value) for value in split["val_subject_keys"]}
    protected = {str(value) for value in split["test_subject_keys"]}
    expected_train = _subject_keys(dataset_id, sorted(TRAIN_SUBJECT_IDS))
    expected_validation = _subject_keys(dataset_id, sorted(VALIDATION_SUBJECT_IDS))
    expected_protected = _subject_keys(dataset_id, sorted(PROTECTED_SUBJECT_IDS))
    if train != expected_train:
        raise ValueError("R1-P-dev fit subjects must be exactly subject_01 through subject_18")
    if validation != expected_validation:
        raise ValueError(
            "R1-P-dev pure-apply subjects must be exactly subject_19 through subject_23"
        )
    if protected != expected_protected:
        raise ValueError(
            "Protected registry must remain exactly subject_24 through subject_29"
        )
    if (train & validation) or (train & protected) or (validation & protected):
        raise ValueError("R1-P split sets must be disjoint")
    if str(experiment.get("teacher_scope")) != TEACHER_SCOPE:
        raise ValueError("R1-P config must declare teacher_scope=population_frozen")
    if bool(experiment.get("protected_open", True)):
        raise ValueError("R1-P development config must keep protected_open=false")
    if bool(experiment.get("promotion_eligible", True)):
        raise ValueError(
            "Teacher construction alone is not promotion eligible before the R1-P panel"
        )

    conditions = list(data["conditions"])
    condition_ids = [str(item["condition_id"]) for item in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("R1-P condition_id values must be unique")
    development_ids = TRAIN_SUBJECT_IDS | VALIDATION_SUBJECT_IDS
    observed_pairs: set[tuple[str, str]] = set()
    for condition in conditions:
        condition_dataset = str(condition["dataset_id"])
        if condition_dataset != dataset_id:
            raise ValueError("All registered R1-P conditions must use one dataset")
        if str(condition["eeg_signal_branch"]) != str(data["eeg_signal_branch"]):
            raise ValueError("Condition EEG branch differs from the frozen data branch")
        allowed = {str(value) for value in condition["subjects"]}
        if allowed != development_ids:
            raise ValueError(
                f"{condition['condition_id']} must list exactly subjects 01-23"
            )
        if allowed & PROTECTED_SUBJECT_IDS:
            raise RuntimeError("Protected subjects are forbidden in data.conditions")
        pair = (str(condition["record_id"]), str(condition["target_label"]))
        if pair in observed_pairs:
            raise ValueError(f"Duplicate registered session/condition pair: {pair}")
        observed_pairs.add(pair)

    expected = data["expected_registry"]
    expected_pairs = {
        (str(session), str(condition))
        for session in expected["sessions"]
        for condition in expected["conditions"]
    }
    if observed_pairs != expected_pairs:
        raise ValueError(
            "Registered R1-P session/condition grid differs from expected_registry"
        )
    expected_per_cell = int(expected["windows_per_subject_session_condition"])
    if any(
        int(condition["max_trials_per_subject"]) != expected_per_cell
        for condition in conditions
    ):
        raise ValueError("Every registered R1-P session/condition cell must have 10 windows")
    expected_train_count = len(train) * len(expected_pairs) * expected_per_cell
    expected_validation_count = (
        len(validation) * len(expected_pairs) * expected_per_cell
    )
    if expected_train_count != int(expected["train_windows"]):
        raise ValueError("expected_registry.train_windows is inconsistent")
    if expected_validation_count != int(expected["validation_windows"]):
        raise ValueError("expected_registry.validation_windows is inconsistent")
    if expected_train_count + expected_validation_count != int(
        expected["development_windows"]
    ):
        raise ValueError("expected_registry.development_windows is inconsistent")

    anchor_conditions = {
        str(value) for value in config["analysis"]["anchor_fit_condition_ids"]
    }
    parameter_conditions = {
        str(value) for value in config["analysis"]["parameter_fit_condition_ids"]
    }
    if not anchor_conditions or not anchor_conditions <= set(condition_ids):
        raise ValueError("anchor_fit_condition_ids must be a non-empty registered subset")
    condition_by_id = {
        str(item["condition_id"]): str(item["target_label"]) for item in conditions
    }
    if any(condition_by_id[value] != "MA" for value in anchor_conditions):
        raise ValueError("Spatial anchor calibration is registered on MA training rows only")
    if parameter_conditions != set(condition_ids):
        raise ValueError("Population SSM/projection fitting must use all registered conditions")
    return {
        "dataset_id": dataset_id,
        "train_subject_keys": train,
        "validation_subject_keys": validation,
        "protected_subject_keys": protected,
        "condition_pairs": expected_pairs,
        "condition_ids": set(condition_ids),
        "anchor_condition_ids": anchor_conditions,
        "parameter_condition_ids": parameter_conditions,
        "expected_train_count": expected_train_count,
        "expected_validation_count": expected_validation_count,
    }


def _condition_lookup(config: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    return {
        (str(item["record_id"]), str(item["target_label"])): str(
            item["condition_id"]
        )
        for item in config["data"]["conditions"]
    }


def load_registered_trials(
    config: Mapping[str, Any],
    *,
    allowed_subject_keys: set[str],
    development_role: str,
) -> tuple[list[PopulationTrial], dict[str, Any]]:
    """Load only an explicitly authorized development partition."""

    registry = validate_population_config(config)
    if allowed_subject_keys & registry["protected_subject_keys"]:
        raise RuntimeError("Protected samples cannot be loaded by the R1-P-dev builder")
    permitted = (
        registry["train_subject_keys"]
        if development_role == "train_fit"
        else registry["validation_subject_keys"]
        if development_role == "validation_pure_apply"
        else set()
    )
    if allowed_subject_keys != permitted:
        raise ValueError(
            f"{development_role!r} subject set does not equal its frozen registry"
        )

    data = config["data"]
    dataset = UnifiedPhysiologyWindowDataset(
        cache_root=data["cache_root"],
        dataset_ids=(registry["dataset_id"],),
        window_duration_s=float(data["window_duration_s"]),
        window_offset_s=float(data["window_offset_s"]),
        eeg_signal_branch=str(data["eeg_signal_branch"]),
    )
    condition_lookup = _condition_lookup(config)
    selected: list[tuple[int, str, str, str]] = []
    for index, ref in enumerate(dataset.windows):
        subject = str(ref.record.canonical_subject_id)
        subject_key = f"{ref.record.dataset_id}|{subject}"
        if subject_key not in allowed_subject_keys:
            continue
        record_id = str(ref.record.base_record_id)
        condition = str(ref.event.get("label"))
        condition_id = condition_lookup.get((record_id, condition))
        if condition_id is None:
            continue
        selected.append((index, condition_id, condition, subject_key))

    baseline_n = int(
        round(float(data["baseline_duration_s"]) * 10.0)
    )
    trials: list[PopulationTrial] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    for index, condition_id, condition, subject_key in selected:
        sample = dataset[index]
        subject = str(sample["subject"])
        if subject in PROTECTED_SUBJECT_IDS:
            raise RuntimeError("Protected measured array dereference was attempted")
        eeg_valid = np.asarray(sample["valid_mask"]["eeg"], dtype=bool)
        fnirs_valid = np.asarray(sample["valid_mask"]["fnirs"], dtype=bool)
        eeg = np.asarray(sample["eeg"], dtype=np.float64).T
        fnirs = np.asarray(sample["fnirs"], dtype=np.float64).T
        if not eeg_valid.all() or not fnirs_valid.all():
            raise RuntimeError(
                f"Registered R1-P window has boundary padding: {subject_key}, "
                f"{sample['record_id']}, event={sample['event'].get('event_index')}"
            )
        if not np.isfinite(eeg).all() or not np.isfinite(fnirs).all():
            raise RuntimeError("Registered R1-P window contains non-finite measurements")
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
                eeg_channel_names=tuple(
                    str(value) for value in sample["channel_names"]["eeg"]
                ),
                fnirs_channel_names=tuple(
                    str(value) for value in sample["channel_names"]["fnirs"]
                ),
                fnirs_roles=tuple(
                    str(value) for value in sample["component_roles"]["fnirs"]
                ),
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

    expected = data["expected_registry"]
    expected_count = (
        registry["expected_train_count"]
        if development_role == "train_fit"
        else registry["expected_validation_count"]
    )
    if len(trials) != expected_count:
        raise RuntimeError(
            f"{development_role} registry count mismatch: "
            f"observed={len(trials)}, expected={expected_count}"
        )
    expected_per_cell = int(expected["windows_per_subject_session_condition"])
    expected_subjects = {
        _subject_parts(value)[1] for value in allowed_subject_keys
    }
    expected_cells = {
        (subject, str(session), str(condition))
        for subject in expected_subjects
        for session in expected["sessions"]
        for condition in expected["conditions"]
    }
    if set(counts) != expected_cells or any(
        counts[key] != expected_per_cell for key in expected_cells
    ):
        raise RuntimeError(f"{development_role} session/condition coverage is incomplete")
    trials.sort(
        key=lambda value: (
            value.subject,
            value.record_id,
            value.condition,
            value.event_index,
        )
    )
    if len({trial.sample_key for trial in trials}) != len(trials):
        raise RuntimeError("Registered R1-P sample identities are not unique")
    audit = {
        "development_role": development_role,
        "sample_count": len(trials),
        "subject_keys": sorted(allowed_subject_keys),
        "session_condition_cells": len(counts),
        "all_boundary_valid": True,
        "all_measurements_finite": True,
        "protected_array_dereference_count": 0,
        "sample_order_sha256": hashlib.sha256(
            "\n".join(trial.sample_key for trial in trials).encode("utf-8")
        ).hexdigest(),
        "loader_contract": dataset.contract_summary(),
    }
    return trials, audit


def _assert_common_channel_contract(trials: Sequence[PopulationTrial]) -> None:
    if not trials:
        raise ValueError("Population bundle fitting requires training trials")
    eeg_names = trials[0].eeg_channel_names
    fnirs_names = trials[0].fnirs_channel_names
    roles = trials[0].fnirs_roles
    for trial in trials:
        if (
            trial.eeg_channel_names != eeg_names
            or trial.fnirs_channel_names != fnirs_names
            or trial.fnirs_roles != roles
        ):
            raise RuntimeError(
                "Population-frozen projection requires a common named channel contract"
            )


def _anchor_id(channel_name: str) -> str:
    value = str(channel_name)
    for suffix in ("_HbO", "_HbR"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def fit_population_bundle(
    train_trials: Sequence[PopulationTrial],
    config: Mapping[str, Any],
) -> PopulationFrozenBundle:
    """Fit every adaptive parameter exclusively from train_fit trials."""

    trials = list(train_trials)
    if not trials or any(trial.development_role != "train_fit" for trial in trials):
        raise ValueError("Population fitting accepts train_fit trials only")
    registry = validate_population_config(config)
    observed_subjects = {trial.subject_key for trial in trials}
    if observed_subjects != registry["train_subject_keys"]:
        raise ValueError("Population fitting requires the complete subjects 01-18 registry")
    if len(trials) != registry["expected_train_count"]:
        raise ValueError("Population fitting requires all 1080 registered train windows")
    if any(trial.subject in PROTECTED_SUBJECT_IDS for trial in trials):
        raise RuntimeError("Protected samples cannot enter population fitting")
    _assert_common_channel_contract(trials)

    anchor_trials = [
        trial
        for trial in trials
        if trial.condition_id in registry["anchor_condition_ids"]
    ]
    parameter_trials = [
        trial
        for trial in trials
        if trial.condition_id in registry["parameter_condition_ids"]
    ]
    if not anchor_trials or len(parameter_trials) != len(trials):
        raise RuntimeError("Registered anchor/parameter training cohorts are incomplete")
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
            raise RuntimeError("Population-frozen view selects a rejected measured channel")
        adapter, train_drivers = _fit_eeg_adapter(parameter_trials, eeg_indices)
        train_hbo, train_hbr = _chromophore_targets(
            parameter_trials,
            hbo_indices,
            hbr_indices,
        )
        fit = _fit_model(
            train_drivers,
            train_hbo,
            train_hbr,
            analysis["ssm"],
            int(
                round(
                    float(data["baseline_duration_s"])
                    * float(analysis["ssm"]["fs_hz"])
                )
            ),
        )
    fit_subject_keys = tuple(sorted(observed_subjects))
    order_hash = hashlib.sha256(
        "\n".join(trial.sample_key for trial in parameter_trials).encode("utf-8")
    ).hexdigest()
    return PopulationFrozenBundle(
        adapter=adapter,
        fit=fit,
        selected_hbo_indices=np.asarray(hbo_indices, dtype=int),
        selected_hbr_indices=np.asarray(hbr_indices, dtype=int),
        selected_fnirs_channels=tuple(hbo_names + hbr_names),
        anchor_id=_anchor_id(hbo_names[0]),
        normalization={},
        fit_subject_keys=fit_subject_keys,
        fit_sample_order_sha256=order_hash,
    )


def _indices_by_name(
    available: Sequence[str],
    selected: Sequence[str],
    *,
    modality: str,
) -> np.ndarray:
    lookup = {str(name): index for index, name in enumerate(available)}
    missing = [str(name) for name in selected if str(name) not in lookup]
    if missing:
        raise RuntimeError(
            f"Population-frozen {modality} channels are absent at apply: {missing}"
        )
    return np.asarray([lookup[str(name)] for name in selected], dtype=int)


def apply_paired_driver(
    trial: PopulationTrial,
    bundle: PopulationFrozenBundle,
) -> PairedDriverResult:
    """Apply one frozen bundle twice, removing only the fNIRS update for rE."""

    if trial.subject in PROTECTED_SUBJECT_IDS:
        raise RuntimeError("R1-P-dev cannot apply to protected arrays")
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
    if not np.array_equal(eeg_indices, bundle.adapter.indices):
        raise RuntimeError("EEG channel ordering drifted from the frozen adapter")
    hbo_count = len(bundle.selected_hbo_indices)
    hbo_indices = fnirs_indices[:hbo_count]
    hbr_indices = fnirs_indices[hbo_count:]
    if (
        trial.eeg_bad_channel_mask[eeg_indices].any()
        or trial.fnirs_bad_channel_mask[fnirs_indices].any()
    ):
        raise RuntimeError("Frozen population view is rejected in an apply sample")

    driver = _apply_eeg_adapter(trial, bundle.adapter)
    hbo, hbr = _chromophore_targets([trial], hbo_indices, hbr_indices)
    joint = apply_adaptive_ssm(
        driver,
        bundle.fit,
        hbo_observation=hbo[0],
        hbr_observation=hbr[0],
    )
    eeg_only = apply_adaptive_ssm(driver, bundle.fit)
    if joint.states.shape != (POINT_COUNT, 5) or eeg_only.states.shape != (
        POINT_COUNT,
        5,
    ):
        raise RuntimeError("Population teacher output does not match [200,5]")
    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(trial.eeg[:, eeg_indices], dtype=np.float32).view(
            np.uint8
        )
    )
    digest.update(
        np.ascontiguousarray(trial.fnirs[:, fnirs_indices], dtype=np.float32).view(
            np.uint8
        )
    )
    return PairedDriverResult(
        joint=np.asarray(joint.states[:, 4], dtype=np.float64),
        joint_std=np.asarray(joint.state_std[:, 4], dtype=np.float64),
        eeg_only=np.asarray(eeg_only.states[:, 4], dtype=np.float64),
        eeg_only_std=np.asarray(eeg_only.state_std[:, 4], dtype=np.float64),
        hbo_observed=np.asarray(hbo[0], dtype=np.float64),
        hbr_observed=np.asarray(hbr[0], dtype=np.float64),
        hbo_reconstructed=np.asarray(joint.hbo_reconstructed, dtype=np.float64),
        hbr_reconstructed=np.asarray(joint.hbr_reconstructed, dtype=np.float64),
        eeg_reconstructed=np.asarray(joint.eeg_reconstructed, dtype=np.float64),
        teacher_input_sha256=digest.hexdigest(),
    )


def fit_shared_driver_normalization(
    train_trials: Sequence[PopulationTrial],
    train_results: Sequence[PairedDriverResult],
) -> dict[str, Any]:
    if len(train_trials) != len(train_results) or not train_trials:
        raise ValueError("Normalization requires paired non-empty training results")
    if any(trial.development_role != "train_fit" for trial in train_trials):
        raise ValueError("Shared-driver normalization accepts train_fit rows only")
    expected_subjects = _subject_keys(
        "eeg_fnirs_single_trial",
        sorted(TRAIN_SUBJECT_IDS),
    )
    if (
        len(train_trials) != 1080
        or {trial.subject_key for trial in train_trials} != expected_subjects
    ):
        raise ValueError(
            "Shared-driver normalization requires all 1080 subjects 01-18 rows"
        )
    values = np.concatenate(
        [np.asarray(result.joint, dtype=np.float64) for result in train_results]
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Training joint driver contains non-finite values")
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale < 1e-6:
        raise RuntimeError("Shared-driver population normalization is degenerate")
    payload = {
        "policy": "scalar_joint_train_subject_points_v1",
        "coordinate": "adaptive_state_index_4_shared_driver",
        "fit_subject_keys": sorted(
            {trial.subject_key for trial in train_trials}
        ),
        "fit_sample_count": len(train_trials),
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


def _parameter_arrays(bundle: PopulationFrozenBundle) -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(PARAMETER_ARRAY_SCHEMA),
        "eeg_indices": np.asarray(bundle.adapter.indices, dtype=np.int64),
        "eeg_channel_names": string_array(bundle.adapter.channel_names),
        "eeg_feature_mean": np.asarray(bundle.adapter.feature_mean, dtype=np.float64),
        "eeg_feature_std": np.asarray(bundle.adapter.feature_std, dtype=np.float64),
        "eeg_pca_mean": np.asarray(bundle.adapter.pca_mean, dtype=np.float64),
        "eeg_loading": np.asarray(bundle.adapter.loading, dtype=np.float64),
        "eeg_pc_scale": np.asarray(bundle.adapter.pc_scale, dtype=np.float64),
        "selected_hbo_indices": np.asarray(
            bundle.selected_hbo_indices, dtype=np.int64
        ),
        "selected_hbr_indices": np.asarray(
            bundle.selected_hbr_indices, dtype=np.int64
        ),
        "selected_fnirs_channels": string_array(bundle.selected_fnirs_channels),
        "transition": np.asarray(bundle.fit.transition, dtype=np.float64),
        "process_cov": np.asarray(bundle.fit.process_cov, dtype=np.float64),
        "observation": np.asarray(bundle.fit.observation, dtype=np.float64),
        "observation_cov": np.asarray(
            bundle.fit.observation_cov, dtype=np.float64
        ),
        "initial_cov": np.asarray(bundle.fit.initial_cov, dtype=np.float64),
    }


def save_population_bundle(
    root: Path,
    bundle: PopulationFrozenBundle,
    *,
    source: Mapping[str, Any],
) -> str:
    expected_fit_subjects = tuple(
        sorted(
            _subject_keys(
                "eeg_fnirs_single_trial",
                sorted(TRAIN_SUBJECT_IDS),
            )
        )
    )
    if tuple(bundle.fit_subject_keys) != expected_fit_subjects:
        raise ValueError("Population parameter bundle has a non-registered fit cohort")
    if not bundle.normalization or bool(
        bundle.normalization.get("validation_subjects_used", True)
    ) or bool(bundle.normalization.get("protected_subjects_used", True)):
        raise ValueError("Population parameter bundle has an unsafe normalization")
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
        "normalization": dict(bundle.normalization),
        "fit_subject_keys": list(bundle.fit_subject_keys),
        "fit_sample_order_sha256": bundle.fit_sample_order_sha256,
        "selected_fnirs_channels": list(bundle.selected_fnirs_channels),
        "anchor_id": bundle.anchor_id,
    }
    bundle_sha = _json_sha256(identity)
    manifest = {
        "schema": PARAMETER_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_scope": TEACHER_SCOPE,
        "parameter_scope": "fit_subjects_01_18_apply_subjects_19_23",
        "arrays_file": arrays_path.name,
        "arrays_sha256": identity["arrays_sha256"],
        "fit_scalars": fit_mapping,
        "normalization": dict(bundle.normalization),
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
    }
    _write_json(root / "manifest.json", manifest)
    return bundle_sha


def load_population_bundle(root: Path) -> PopulationFrozenBundle:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PARAMETER_MANIFEST_SCHEMA:
        raise ValueError("Population parameter-bundle schema mismatch")
    if bool(manifest.get("validation_subjects_used_for_any_fit", True)):
        raise RuntimeError("Population bundle provenance admits validation fitting")
    if bool(manifest.get("protected_subjects_used_for_any_fit", True)):
        raise RuntimeError("Population bundle provenance admits protected fitting")
    arrays_path = root / str(manifest["arrays_file"])
    if _sha256(arrays_path) != manifest["arrays_sha256"]:
        raise RuntimeError("Population parameter array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    if str(np.asarray(arrays["schema"]).item()) != PARAMETER_ARRAY_SCHEMA:
        raise ValueError("Population parameter-array schema mismatch")
    scalars = manifest["fit_scalars"]
    params = HemodynamicParameters(
        epsilon=float(scalars["epsilon"]),
        kas=float(scalars["kas"]),
        kaf=float(scalars["kaf"]),
        tau0=float(scalars["tau0"]),
        alpha=float(scalars["alpha"]),
        e0=float(scalars["e0"]),
    )
    fit = AdaptiveSSMFit(
        params=params,
        transition=arrays["transition"],
        process_cov=arrays["process_cov"],
        observation=arrays["observation"],
        observation_cov=arrays["observation_cov"],
        initial_cov=arrays["initial_cov"],
        hbo_mean=float(scalars["hbo_mean"]),
        hbo_std=float(scalars["hbo_std"]),
        hbr_mean=float(scalars["hbr_mean"]),
        hbr_std=float(scalars["hbr_std"]),
        baseline_samples=int(round(float(scalars["baseline_samples"]))),
        phi=float(scalars["phi"]),
        q_driver=float(scalars["q_driver"]),
        q_scale=float(scalars["q_scale"]),
        fnirs_noise_scale=float(scalars["fnirs_noise_scale"]),
        hbo_gain=float(scalars["hbo_gain"]),
        hbr_gain=float(scalars["hbr_gain"]),
        eeg_noise=float(scalars["eeg_noise"]),
        hbo_noise_base=float(scalars["hbo_noise_base"]),
        hbr_noise_base=float(scalars["hbr_noise_base"]),
        training_score=float(scalars["training_score"]),
        optimizer_success=bool(scalars["optimizer_success"]),
        optimizer_objective=float(scalars["optimizer_objective"]),
    )
    adapter = EEGAdapter(
        indices=np.asarray(arrays["eeg_indices"], dtype=int),
        channel_names=tuple(
            str(value) for value in arrays["eeg_channel_names"].tolist()
        ),
        feature_mean=arrays["eeg_feature_mean"],
        feature_std=arrays["eeg_feature_std"],
        pca_mean=arrays["eeg_pca_mean"],
        loading=arrays["eeg_loading"],
        pc_scale=float(np.asarray(arrays["eeg_pc_scale"]).item()),
    )
    identity = {
        "arrays_sha256": manifest["arrays_sha256"],
        "fit_scalars": scalars,
        "normalization": manifest["normalization"],
        "fit_subject_keys": manifest["fit_subject_keys"],
        "fit_sample_order_sha256": manifest["fit_sample_order_sha256"],
        "selected_fnirs_channels": manifest["selected_fnirs_channels"],
        "anchor_id": manifest["anchor_id"],
    }
    observed_sha = _json_sha256(identity)
    if observed_sha != manifest["bundle_sha256"]:
        raise RuntimeError("Population parameter-bundle identity hash mismatch")
    hbo_count = len(np.asarray(arrays["selected_hbo_indices"]).reshape(-1))
    fnirs_channels = tuple(str(value) for value in manifest["selected_fnirs_channels"])
    return PopulationFrozenBundle(
        adapter=adapter,
        fit=fit,
        selected_hbo_indices=np.asarray(
            arrays["selected_hbo_indices"], dtype=int
        ),
        selected_hbr_indices=np.asarray(
            arrays["selected_hbr_indices"], dtype=int
        ),
        selected_fnirs_channels=(
            tuple(fnirs_channels[:hbo_count]) + tuple(fnirs_channels[hbo_count:])
        ),
        anchor_id=str(manifest["anchor_id"]),
        normalization=dict(manifest["normalization"]),
        fit_subject_keys=tuple(str(value) for value in manifest["fit_subject_keys"]),
        fit_sample_order_sha256=str(manifest["fit_sample_order_sha256"]),
        bundle_sha256=observed_sha,
    )


def _normalized_pair(
    result: PairedDriverResult,
    normalization: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = float(normalization["mean"])
    scale = float(normalization["scale"])
    joint = (np.asarray(result.joint, dtype=np.float64) - mean) / scale
    eeg = (np.asarray(result.eeg_only, dtype=np.float64) - mean) / scale
    joint_std = np.asarray(result.joint_std, dtype=np.float64) / abs(scale)
    eeg_std = np.asarray(result.eeg_only_std, dtype=np.float64) / abs(scale)
    arrays = (joint, joint_std, eeg, eeg_std)
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("Normalized population target contains non-finite values")
    return tuple(
        value.reshape(PATCH_COUNT, POINTS_PER_PATCH).astype(np.float32)
        for value in arrays
    )


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"source_commit": commit, "dirty_worktree": dirty}


def _write_artifacts(
    root: Path,
    trials: Sequence[PopulationTrial],
    results: Sequence[PairedDriverResult],
    bundle: PopulationFrozenBundle,
    *,
    config_path: Path,
    train_audit: Mapping[str, Any],
    validation_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if len(trials) != len(results):
        raise ValueError("Trial/result count mismatch")
    sample_keys = [trial.sample_key for trial in trials]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Population target sample keys must be unique")
    normalized = [
        _normalized_pair(result, bundle.normalization) for result in results
    ]
    joint = np.stack([value[0] for value in normalized])
    joint_std = np.stack([value[1] for value in normalized])
    eeg = np.stack([value[2] for value in normalized])
    eeg_std = np.stack([value[3] for value in normalized])
    point_mask = np.ones_like(joint, dtype=bool)
    time_s = (
        -5.0
        + np.arange(POINT_COUNT, dtype=np.float32).reshape(
            PATCH_COUNT, POINTS_PER_PATCH
        )
        / 10.0
    )
    parameter_fold = (
        "population_frozen:r1_p_dev:fit_subjects_01_18:"
        f"{bundle.bundle_sha256}"
    )
    gauge_hash = str(bundle.normalization["sha256"])
    teacher_source_hashes = [
        _json_sha256(
            {
                "bundle_sha256": bundle.bundle_sha256,
                "sample_key": trial.sample_key,
                "teacher_input_sha256": result.teacher_input_sha256,
            }
        )
        for trial, result in zip(trials, results)
    ]
    split_names = [
        "train" if trial.development_role == "train_fit" else "validation"
        for trial in trials
    ]

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
        target_eeg_only_driver=eeg,
        target_eeg_only_driver_std=eeg_std,
        eeg_only_valid_mask=point_mask.any(axis=-1),
        eeg_only_point_valid_mask=point_mask,
        teacher_scope=string_array([TEACHER_SCOPE] * len(trials)),
        teacher_parameter_fold=string_array([parameter_fold] * len(trials)),
        teacher_gauge_hash=string_array([gauge_hash] * len(trials)),
        teacher_source_hash=string_array(teacher_source_hashes),
        parameter_bundle_sha256=string_array(
            [bundle.bundle_sha256] * len(trials)
        ),
    )

    split_payload = {
        "fit_subject_keys": list(bundle.fit_subject_keys),
        "validation_subject_keys": sorted(
            {trial.subject_key for trial in trials if trial.development_role == "validation_pure_apply"}
        ),
        "protected_subject_keys": sorted(
            _subject_keys(
                trials[0].dataset_id,
                sorted(PROTECTED_SUBJECT_IDS),
            )
        ),
        "validation_pure_apply": True,
        "protected_open": False,
    }
    source = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": _sha256(Path(__file__).resolve()),
        "parameter_bundle_sha256": bundle.bundle_sha256,
    }
    order_sha = hashlib.sha256("\n".join(sample_keys).encode("utf-8")).hexdigest()
    teacher_manifest = {
        "schema": SHARED_DRIVER_SIDECAR_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_generation": ARCHITECTURE_GENERATION,
        "target_family": TARGET_FAMILY,
        "target_version": TARGET_VERSION,
        "teacher_scope": TEACHER_SCOPE,
        "teacher_parameter_scope": "fit_01_18_apply_19_23",
        "trajectory_shape": [PATCH_COUNT, POINTS_PER_PATCH],
        "sample_rate_hz": 10.0,
        "arrays_file": teacher_arrays.name,
        "arrays_sha256": _sha256(teacher_arrays),
        "sample_count": len(trials),
        "sample_order_sha256": order_sha,
        "normalization": dict(bundle.normalization),
        "paired_control": {
            "joint_and_eeg_only_exact_parameter_bundle": True,
            "only_difference": "fNIRS observation update omitted for rE",
            "parameter_bundle_sha256": bundle.bundle_sha256,
            "shared_driver_gauge_sha256": gauge_hash,
        },
        "source": source,
        "split": split_payload,
        "promotion_eligible": False,
        "promotion_blocker": "population_frozen_teacher_panel_not_run",
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
        "raw_view_policy": "population_frozen_anchor_and_eeg_projection_v1",
        "selection_scope": TEACHER_SCOPE,
        "arrays_file": raw_arrays.name,
        "arrays_sha256": _sha256(raw_arrays),
        "sample_count": len(trials),
        "sample_order_sha256": order_sha,
        "selected_eeg_channels": list(bundle.adapter.channel_names),
        "selected_fnirs_channels": list(bundle.selected_fnirs_channels),
        "anchor_id": bundle.anchor_id,
        "parameter_bundle_sha256": bundle.bundle_sha256,
        "source": source,
        "split": split_payload,
        "promotion_eligible": False,
        "protected_open": False,
        "protected_test_included": False,
        **_git_state(),
    }
    _write_json(raw_root / "manifest.json", raw_manifest)

    coverage_path = root / "data_coverage_by_subject_session_condition_patch.csv"
    coverage_rows: dict[tuple[str, str, str, str, int], int] = defaultdict(int)
    for trial in trials:
        split = (
            "train"
            if trial.development_role == "train_fit"
            else "validation"
        )
        for patch in range(PATCH_COUNT):
            coverage_rows[
                (trial.subject_key, trial.record_id, trial.condition, split, patch)
            ] += 1
    fields = [
        "subject_key",
        "session_id",
        "condition",
        "development_split",
        "patch_index",
        "sample_count",
        "supported_points",
        "possible_points",
        "point_coverage_fraction",
    ]
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(coverage_rows):
            count = coverage_rows[key]
            writer.writerow(
                {
                    "subject_key": key[0],
                    "session_id": key[1],
                    "condition": key[2],
                    "development_split": key[3],
                    "patch_index": key[4],
                    "sample_count": count,
                    "supported_points": count * POINTS_PER_PATCH,
                    "possible_points": count * POINTS_PER_PATCH,
                    "point_coverage_fraction": 1.0,
                }
            )

    apply_rows = []
    for split in ("train", "validation"):
        selected = [
            trial
            for trial in trials
            if (
                split == "train"
                and trial.development_role == "train_fit"
            )
            or (
                split == "validation"
                and trial.development_role == "validation_pure_apply"
            )
        ]
        for subject_key in sorted({trial.subject_key for trial in selected}):
            rows = [trial for trial in selected if trial.subject_key == subject_key]
            apply_rows.append(
                {
                    "subject_key": subject_key,
                    "development_split": split,
                    "sample_count": len(rows),
                    "parameter_bundle_sha256": bundle.bundle_sha256,
                    "participated_in_global_parameter_fit": split == "train",
                    "global_parameter_bundle_fit_count": 1,
                    "subject_specific_fit_calls": 0,
                    "validation_refit_calls": 0,
                    "protected_array_dereference_count": 0,
                }
            )
    apply_path = root / "population_frozen_apply_audit.csv"
    with apply_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(apply_rows[0]))
        writer.writeheader()
        writer.writerows(apply_rows)

    leakage = {
        "schema": "shared_driver_r1p_leakage_audit_v1",
        "fit_subject_keys": list(bundle.fit_subject_keys),
        "validation_loaded_after_parameter_and_normalization_freeze": True,
        "validation_fit_calls": 0,
        "validation_normalization_calls": 0,
        "protected_array_dereference_count": 0,
        "protected_open": False,
        "joint_and_eeg_only_exact_parameter_bundle": True,
        "raw_view_and_target_separate": True,
        "train_registry_audit": dict(train_audit),
        "validation_registry_audit": dict(validation_audit),
    }
    _write_json(root / "leakage_audit.json", leakage)
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_generation": ARCHITECTURE_GENERATION,
        "teacher_scope": TEACHER_SCOPE,
        "target_family": TARGET_FAMILY,
        "target_version": TARGET_VERSION,
        "sample_count": len(trials),
        "train_sample_count": sum(value == "train" for value in split_names),
        "validation_sample_count": sum(
            value == "validation" for value in split_names
        ),
        "parameter_bundle": {
            "path": "parameter_bundle",
            "bundle_sha256": bundle.bundle_sha256,
            "manifest_sha256": _sha256(root / "parameter_bundle" / "manifest.json"),
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
        "coverage_file": coverage_path.name,
        "coverage_sha256": _sha256(coverage_path),
        "apply_audit": apply_path.name,
        "apply_audit_sha256": _sha256(apply_path),
        "leakage_audit": "leakage_audit.json",
        "promotion_eligible": False,
        "promotion_blocker": "population_frozen_teacher_panel_not_run",
        "protected_open": False,
        "protected_test_included": False,
        "source": source,
        **_git_state(),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _source_payload(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_root = REPO_ROOT / str(config["data"]["cache_root"])
    paths = {
        "config": config_path,
        "builder": Path(__file__).resolve(),
        "adaptive_evaluator": (
            REPO_ROOT / "experiments/evaluate_adaptive_shared_neural_ssm.py"
        ),
        "adaptive_solver": (
            REPO_ROOT / "src/inference/adaptive_neurovascular_ssm.py"
        ),
        "cache_manifest": cache_root / "cache_manifest.json",
        "event_manifest": cache_root / "event_index" / "event_manifest.json",
        "geometry_manifest": (
            cache_root / "channel_geometry" / "geometry_manifest.json"
        ),
        "eeg_artifact_manifest": (
            cache_root / "eeg_artifact_clean_v4" / "cache_manifest.json"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R1-P provenance input is missing: {missing}")
    return {
        "input_hashes": {
            name: _sha256(path) for name, path in sorted(paths.items())
        },
        **_git_state(),
    }


def build_population_frozen_teacher(
    config_path: Path,
    output_root: Path,
) -> Path:
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registry = validate_population_config(config)

    # Validation arrays are deliberately not loaded until all parameters and
    # the shared-driver normalization have been frozen below.
    train_trials, train_audit = load_registered_trials(
        config,
        allowed_subject_keys=registry["train_subject_keys"],
        development_role="train_fit",
    )
    core_bundle = fit_population_bundle(train_trials, config)
    with threadpool_limits(limits=1):
        train_results = [
            apply_paired_driver(trial, core_bundle) for trial in train_trials
        ]
    normalization = fit_shared_driver_normalization(
        train_trials,
        train_results,
    )
    frozen_bundle = PopulationFrozenBundle(
        adapter=core_bundle.adapter,
        fit=core_bundle.fit,
        selected_hbo_indices=core_bundle.selected_hbo_indices,
        selected_hbr_indices=core_bundle.selected_hbr_indices,
        selected_fnirs_channels=core_bundle.selected_fnirs_channels,
        anchor_id=core_bundle.anchor_id,
        normalization=normalization,
        fit_subject_keys=core_bundle.fit_subject_keys,
        fit_sample_order_sha256=core_bundle.fit_sample_order_sha256,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.tmp-",
        dir=output_root.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        save_population_bundle(
            temporary_root / "parameter_bundle",
            frozen_bundle,
            source=_source_payload(config_path),
        )
        reloaded = load_population_bundle(temporary_root / "parameter_bundle")
        sentinel = apply_paired_driver(train_trials[0], reloaded)
        if not (
            np.array_equal(sentinel.joint, train_results[0].joint)
            and np.array_equal(sentinel.eeg_only, train_results[0].eeg_only)
        ):
            raise RuntimeError("Serialized population bundle changed sentinel output")

        # Pure-apply rows are first dereferenced only after bundle freeze/reload.
        validation_trials, validation_audit = load_registered_trials(
            config,
            allowed_subject_keys=registry["validation_subject_keys"],
            development_role="validation_pure_apply",
        )
        with threadpool_limits(limits=1):
            validation_results = [
                apply_paired_driver(trial, reloaded)
                for trial in validation_trials
            ]
        all_trials = train_trials + validation_trials
        all_results = train_results + validation_results
        _write_artifacts(
            temporary_root,
            all_trials,
            all_results,
            reloaded,
            config_path=config_path,
            train_audit=train_audit,
            validation_audit=validation_audit,
        )
        temporary_root.replace(output_root)
    return output_root


def registry_audit(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registry = validate_population_config(config)
    train, train_audit = load_registered_trials(
        config,
        allowed_subject_keys=registry["train_subject_keys"],
        development_role="train_fit",
    )
    validation, validation_audit = load_registered_trials(
        config,
        allowed_subject_keys=registry["validation_subject_keys"],
        development_role="validation_pure_apply",
    )
    _assert_common_channel_contract(train + validation)
    anchor_condition_ids = registry["anchor_condition_ids"]
    anchor_trials = [
        trial for trial in train if trial.condition_id in anchor_condition_ids
    ]
    with threadpool_limits(limits=1):
        hbo_indices, hbo_names, _ = _select_active_hbo(
            anchor_trials,
            baseline_duration_s=float(config["data"]["baseline_duration_s"]),
            task_duration_s=float(config["data"]["task_duration_s"]),
            count=int(config["analysis"]["fnirs_active_hbo_channels"]),
        )
        hbr_indices = _paired_hbr_indices(anchor_trials[0], hbo_indices)
        hbr_names = tuple(
            anchor_trials[0].fnirs_channel_names[int(index)]
            for index in hbr_indices
        )
        eeg_indices = _local_eeg_indices(
            anchor_trials[0],
            hbo_indices,
            int(config["analysis"]["local_eeg_channels"]),
        )
    if any(
        trial.eeg_bad_channel_mask[eeg_indices].any()
        or trial.fnirs_bad_channel_mask[hbo_indices].any()
        or trial.fnirs_bad_channel_mask[hbr_indices].any()
        for trial in train + validation
    ):
        raise RuntimeError(
            "Train-selected population raw view is rejected in development rows"
        )
    eeg_names = tuple(
        anchor_trials[0].eeg_channel_names[int(index)] for index in eeg_indices
    )
    return {
        "schema": "shared_driver_r1p_registry_audit_v1",
        "config": str(config_path),
        "train": train_audit,
        "validation": validation_audit,
        "development_sample_count": len(train) + len(validation),
        "common_named_channel_contract": True,
        "eeg_channel_count": len(train[0].eeg_channel_names),
        "fnirs_channel_count": len(train[0].fnirs_channel_names),
        "train_only_view_selection_audit": {
            "anchor_fit_sample_count": len(anchor_trials),
            "selected_eeg_channels": list(eeg_names),
            "selected_fnirs_channels": list(hbo_names + hbr_names),
            "anchor_id": _anchor_id(hbo_names[0]),
            "admitted_in_all_development_rows": True,
            "validation_rows_used_for_selection": False,
        },
        "protected_array_dereference_count": 0,
        "protected_open": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "experiments/configs/physiology_semantic_tokenizer/"
            "r1p_population_frozen_teacher.yaml"
        ),
    )
    parser.add_argument("--output-root")
    parser.add_argument("--registry-audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if args.registry_audit_only:
        print(json.dumps(registry_audit(config_path), sort_keys=True))
        return
    if not args.output_root:
        raise SystemExit("--output-root is required unless --registry-audit-only is used")
    output = build_population_frozen_teacher(
        config_path,
        Path(args.output_root),
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_root": str(output),
                "sample_count": manifest["sample_count"],
                "teacher_scope": manifest["teacher_scope"],
                "promotion_eligible": manifest["promotion_eligible"],
                "protected_open": manifest["protected_open"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
