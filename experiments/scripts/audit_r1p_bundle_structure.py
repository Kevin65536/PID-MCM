#!/usr/bin/env python3
"""Audit an R1-P bundle without evaluating validation physiology effects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.build_r1p_population_frozen_teacher import (
    _json_sha256,
    _parameter_arrays,
    fit_to_mapping,
    load_population_bundle,
    validate_population_config,
)
from src.data.shared_driver_dataset import SharedDriverWindowDataset
from src.data.shared_driver_targets import (
    PhysiologyRawViewRegistry,
    SharedDriverTrajectorySidecar,
)


SCHEMA = "r1p_bundle_structural_audit_v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-dataset-reader",
        action="store_true",
        help="Skip measured-data dataset construction; intended only for unit tests.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def _sample_order_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def _string_set(values: np.ndarray) -> set[str]:
    return {str(value) for value in np.asarray(values).tolist()}


def _check_manifest_hashes(
    root: Path,
    top: Mapping[str, Any],
) -> dict[str, str]:
    paths = {
        "coverage": root / str(top["coverage_file"]),
        "apply_audit": root / str(top["apply_audit"]),
        "leakage_audit": root / str(top["leakage_audit"]),
        "parameter_manifest": root / "parameter_bundle/manifest.json",
        "parameter_arrays": root / "parameter_bundle/arrays.npz",
        "trajectory_manifest": root / "trajectory_targets/manifest.json",
        "trajectory_arrays": root / "trajectory_targets/arrays.npz",
        "raw_view_manifest": root / "raw_view_registry/manifest.json",
        "raw_view_arrays": root / "raw_view_registry/arrays.npz",
    }
    for path in paths.values():
        _require(path.is_file(), f"Missing R1-P artifact: {path}")
    observed = {key: _sha256(path) for key, path in paths.items()}
    _require(observed["coverage"] == top["coverage_sha256"], "Coverage hash drift")
    _require(
        observed["apply_audit"] == top["apply_audit_sha256"],
        "Apply-audit hash drift",
    )
    _require(
        observed["parameter_manifest"]
        == top["parameter_bundle"]["manifest_sha256"],
        "Parameter manifest hash drift",
    )
    _require(
        observed["trajectory_manifest"]
        == top["trajectory_targets"]["manifest_sha256"],
        "Trajectory manifest hash drift",
    )
    _require(
        observed["raw_view_manifest"]
        == top["raw_view_registry"]["manifest_sha256"],
        "Raw-view manifest hash drift",
    )
    for label, child in (
        ("parameter", _json(paths["parameter_manifest"])),
        ("trajectory", _json(paths["trajectory_manifest"])),
        ("raw_view", _json(paths["raw_view_manifest"])),
    ):
        _require(
            observed[f"{label}_arrays"] == child["arrays_sha256"],
            f"{label} array hash drift",
        )
    _require(
        observed["trajectory_arrays"]
        == top["trajectory_targets"]["arrays_sha256"],
        "Top-level trajectory array hash drift",
    )
    _require(
        observed["raw_view_arrays"] == top["raw_view_registry"]["arrays_sha256"],
        "Top-level raw-view array hash drift",
    )
    return observed


def _check_parameter_roundtrip(
    root: Path,
    top: Mapping[str, Any],
) -> dict[str, Any]:
    parameter_root = root / "parameter_bundle"
    manifest = _json(parameter_root / "manifest.json")
    arrays = _npz(parameter_root / "arrays.npz")
    loaded = load_population_bundle(parameter_root)
    reconstructed = _parameter_arrays(loaded)
    _require(set(arrays) == set(reconstructed), "Parameter round-trip key drift")
    for key in arrays:
        _require(
            np.array_equal(arrays[key], reconstructed[key]),
            f"Parameter round-trip changed {key}",
        )
    fit_mapping = {
        key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
        for key, value in fit_to_mapping(loaded.fit).items()
    }
    _require(
        fit_mapping == manifest["fit_scalars"],
        "Parameter scalar round-trip drift",
    )
    finite_arrays = [
        value
        for key, value in arrays.items()
        if key != "schema" and np.issubdtype(value.dtype, np.number)
    ]
    _require(
        all(np.isfinite(value).all() for value in finite_arrays),
        "Parameter bundle contains non-finite numeric values",
    )
    identity = {
        "arrays_sha256": manifest["arrays_sha256"],
        "fit_scalars": manifest["fit_scalars"],
        "normalization": manifest["normalization"],
        "fit_subject_keys": manifest["fit_subject_keys"],
        "fit_sample_order_sha256": manifest["fit_sample_order_sha256"],
        "selected_fnirs_channels": manifest["selected_fnirs_channels"],
        "anchor_id": manifest["anchor_id"],
    }
    identity_hash = _json_sha256(identity)
    expected = str(top["parameter_bundle"]["bundle_sha256"])
    _require(identity_hash == manifest["bundle_sha256"] == expected, "Bundle identity drift")
    _require(loaded.bundle_sha256 == expected, "Loaded bundle identity drift")
    return {
        "bundle_sha256": expected,
        "array_field_count": len(arrays),
        "roundtrip_array_equal": True,
        "roundtrip_fit_scalars_equal": True,
        "numeric_arrays_finite": True,
    }


def _check_target_and_raw_join(
    root: Path,
    config: Mapping[str, Any],
    top: Mapping[str, Any],
) -> dict[str, Any]:
    teacher_manifest = _json(root / "trajectory_targets/manifest.json")
    raw_manifest = _json(root / "raw_view_registry/manifest.json")
    parameter_manifest = _json(root / "parameter_bundle/manifest.json")
    leakage_manifest = _json(root / "leakage_audit.json")
    teacher = _npz(root / "trajectory_targets/arrays.npz")
    raw = _npz(root / "raw_view_registry/arrays.npz")
    expected = config["data"]["expected_registry"]
    expected_count = int(expected["development_windows"])
    teacher_keys = [str(value) for value in teacher["sample_key"].tolist()]
    raw_keys = [str(value) for value in raw["sample_key"].tolist()]
    _require(len(teacher_keys) == expected_count, "Trajectory count drift")
    _require(teacher_keys == raw_keys, "Raw-view/target sample order differs")
    _require(len(set(teacher_keys)) == expected_count, "Duplicate sample keys")
    order_hash = _sample_order_hash(teacher_keys)
    _require(
        order_hash
        == teacher_manifest["sample_order_sha256"]
        == raw_manifest["sample_order_sha256"],
        "Raw-view/target sample-order hash drift",
    )
    for field in (
        "dataset_id",
        "subject_id",
        "subject_key",
        "session_id",
        "condition",
        "event_index",
    ):
        _require(
            np.array_equal(teacher[field], raw[field]),
            f"Raw-view/target metadata differs for {field}",
        )

    train_subjects = set(map(str, config["data"]["split"]["train_subject_keys"]))
    validation_subjects = set(
        map(str, config["data"]["split"]["val_subject_keys"])
    )
    protected_subjects = set(
        map(str, config["data"]["split"]["test_subject_keys"])
    )
    subject_keys = np.asarray(teacher["subject_key"]).astype(str)
    splits = np.asarray(teacher["development_split"]).astype(str)
    roles = np.asarray(teacher["parameter_role"]).astype(str)
    train_rows = splits == "train"
    validation_rows = splits == "validation"
    _require(int(train_rows.sum()) == int(expected["train_windows"]), "Train count drift")
    _require(
        int(validation_rows.sum()) == int(expected["validation_windows"]),
        "Validation count drift",
    )
    _require(
        _string_set(subject_keys[train_rows]) == train_subjects,
        "Train subject cohort drift",
    )
    _require(
        _string_set(subject_keys[validation_rows]) == validation_subjects,
        "Validation subject cohort drift",
    )
    _require(
        not _string_set(subject_keys).intersection(protected_subjects),
        "Protected subject appears in R1-P arrays",
    )
    train_order_hash = _sample_order_hash(
        np.asarray(teacher["sample_key"]).astype(str)[train_rows].tolist()
    )
    validation_order_hash = _sample_order_hash(
        np.asarray(teacher["sample_key"]).astype(str)[validation_rows].tolist()
    )
    _require(
        train_order_hash
        == parameter_manifest["fit_sample_order_sha256"]
        == leakage_manifest["train_registry_audit"]["sample_order_sha256"],
        "Train-fit sample order drift",
    )
    _require(
        validation_order_hash
        == leakage_manifest["validation_registry_audit"]["sample_order_sha256"],
        "Validation pure-apply sample order drift",
    )
    _require(
        set(roles[train_rows].tolist()) == {"train_fit"},
        "Train parameter-role drift",
    )
    _require(
        set(roles[validation_rows].tolist()) == {"validation_pure_apply"},
        "Validation parameter-role drift",
    )

    expected_sessions = set(map(str, expected["sessions"]))
    expected_conditions = set(map(str, expected["conditions"]))
    grouped: dict[tuple[str, str, str], int] = {}
    for subject, session, condition in zip(
        subject_keys.tolist(),
        np.asarray(teacher["session_id"]).astype(str).tolist(),
        np.asarray(teacher["condition"]).astype(str).tolist(),
    ):
        key = (subject, session, condition)
        grouped[key] = grouped.get(key, 0) + 1
    expected_cells = len(subject_keys.tolist()) // int(
        expected["windows_per_subject_session_condition"]
    )
    _require(len(grouped) == expected_cells == 138, "Session/condition cell drift")
    _require(
        all(
            count == int(expected["windows_per_subject_session_condition"])
            for count in grouped.values()
        ),
        "Window count per subject/session/condition drift",
    )
    _require(
        {key[1] for key in grouped} == expected_sessions,
        "Registered session set drift",
    )
    _require(
        {key[2] for key in grouped} == expected_conditions,
        "Registered condition set drift",
    )

    target_shape = (expected_count, 10, 20)
    numeric_fields = (
        "target_time_s",
        "target_shared_driver",
        "target_shared_driver_std",
        "target_eeg_only_driver",
        "target_eeg_only_driver_std",
    )
    _require(
        all(np.asarray(teacher[field]).shape == target_shape for field in numeric_fields),
        "Trajectory field shape drift",
    )
    _require(
        all(np.isfinite(np.asarray(teacher[field])).all() for field in numeric_fields),
        "Trajectory structural field contains non-finite values",
    )
    _require(
        np.all(np.asarray(teacher["target_shared_driver_std"]) >= 0.0)
        and np.all(np.asarray(teacher["target_eeg_only_driver_std"]) >= 0.0),
        "Trajectory uncertainty contains negative values",
    )
    joint_mask = np.asarray(teacher["target_point_valid_mask"], dtype=bool)
    eeg_mask = np.asarray(teacher["eeg_only_point_valid_mask"], dtype=bool)
    _require(joint_mask.shape == eeg_mask.shape == target_shape, "Point-mask shape drift")
    _require(np.array_equal(joint_mask, eeg_mask), "Joint/rE support differs")
    _require(bool(joint_mask.all()), "R1-P has incomplete target support")
    _require(
        np.array_equal(
            np.asarray(teacher["target_valid_mask"], dtype=bool),
            joint_mask.any(axis=-1),
        )
        and np.array_equal(
            np.asarray(teacher["eeg_only_valid_mask"], dtype=bool),
            eeg_mask.any(axis=-1),
        ),
        "Patch and point masks disagree",
    )
    expected_time = (
        -5.0
        + np.arange(200, dtype=np.float32).reshape(10, 20) / 10.0
    )
    _require(
        np.array_equal(
            np.asarray(teacher["target_time_s"]),
            np.broadcast_to(expected_time, target_shape),
        ),
        "Trajectory time grid drift",
    )

    bundle_sha = str(top["parameter_bundle"]["bundle_sha256"])
    gauge_sha = str(top["normalization"]["sha256"])
    expected_fold = f"population_frozen:r1_p_dev:fit_subjects_01_18:{bundle_sha}"
    folds = _string_set(teacher["teacher_parameter_fold"])
    gauges = _string_set(teacher["teacher_gauge_hash"])
    bundle_rows = _string_set(teacher["parameter_bundle_sha256"])
    _require(folds == {expected_fold}, "Teacher parameter fold is not common/frozen")
    _require(gauges == {gauge_sha}, "Teacher gauge is not common/frozen")
    _require(bundle_rows == {bundle_sha}, "Teacher bundle identity differs by row")
    _require(
        _string_set(teacher["teacher_scope"]) == {"population_frozen"},
        "Teacher scope drift",
    )
    source_hashes = [str(value) for value in teacher["teacher_source_hash"].tolist()]
    _require(
        len(set(source_hashes)) == expected_count
        and all(HEX64.fullmatch(value) for value in source_hashes),
        "Teacher source hashes are malformed or non-unique",
    )
    _require(
        _string_set(raw["selection_fold"]) == {expected_fold}
        and _string_set(raw["selection_source_hash"]) == {bundle_sha},
        "Raw-view selection is not tied to the common frozen bundle",
    )
    _require(
        np.array_equal(raw["selection_fold"], teacher["teacher_parameter_fold"]),
        "Raw-view and target parameter folds differ",
    )
    _require(
        set(map(tuple, raw["selected_eeg_channels"].tolist()))
        == {tuple(raw_manifest["selected_eeg_channels"])},
        "EEG selection differs across rows",
    )
    _require(
        set(map(tuple, raw["selected_fnirs_channels"].tolist()))
        == {tuple(raw_manifest["selected_fnirs_channels"])},
        "fNIRS selection differs across rows",
    )
    normalization = dict(top["normalization"])
    _require(
        normalization
        == parameter_manifest["normalization"]
        == teacher_manifest["normalization"],
        "Normalization provenance differs across manifests",
    )
    _require(
        normalization["fit_sample_count"] == 1080
        and normalization["fit_point_count"] == 216000
        and set(normalization["fit_subject_keys"]) == train_subjects
        and normalization["validation_subjects_used"] is False
        and normalization["protected_subjects_used"] is False,
        "Train-only normalization contract drift",
    )
    _require(
        teacher_manifest["paired_control"][
            "joint_and_eeg_only_exact_parameter_bundle"
        ]
        is True
        and teacher_manifest["paired_control"]["parameter_bundle_sha256"]
        == bundle_sha
        and teacher_manifest["paired_control"]["shared_driver_gauge_sha256"]
        == gauge_sha
        and parameter_manifest["joint_and_eeg_only_share_exact_bundle"] is True,
        "Joint/rE paired provenance drift",
    )
    for manifest in (top, teacher_manifest, raw_manifest):
        _require(
            manifest["protected_open"] is False
            and manifest["protected_test_included"] is False,
            "Protected boundary differs across manifests",
        )
    return {
        "sample_count": expected_count,
        "train_sample_count": int(train_rows.sum()),
        "validation_sample_count": int(validation_rows.sum()),
        "subject_count": len(_string_set(subject_keys)),
        "session_condition_cells": len(grouped),
        "windows_per_subject_session_condition": sorted(set(grouped.values())),
        "sample_order_sha256": order_hash,
        "train_sample_order_sha256": train_order_hash,
        "validation_sample_order_sha256": validation_order_hash,
        "raw_view_target_exact_order_join": True,
        "target_shape": list(target_shape),
        "all_structural_numeric_fields_finite": True,
        "joint_and_eeg_only_masks_equal_and_complete": True,
        "common_parameter_fold": expected_fold,
        "common_gauge_sha256": gauge_sha,
        "teacher_source_hash_unique_count": len(set(source_hashes)),
    }


def _check_coverage_and_leakage(
    root: Path,
    config: Mapping[str, Any],
    join: Mapping[str, Any],
) -> dict[str, Any]:
    expected = config["data"]["expected_registry"]
    with (root / "data_coverage_by_subject_session_condition_patch.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        coverage = list(csv.DictReader(handle))
    expected_rows = int(join["session_condition_cells"]) * 10
    _require(len(coverage) == expected_rows == 1380, "Coverage row count drift")
    coverage_keys = {
        (
            row["subject_key"],
            row["session_id"],
            row["condition"],
            row["development_split"],
            int(row["patch_index"]),
        )
        for row in coverage
    }
    _require(len(coverage_keys) == len(coverage), "Duplicate coverage rows")
    _require(
        all(
            int(row["sample_count"])
            == int(expected["windows_per_subject_session_condition"])
            and int(row["supported_points"]) == 200
            and int(row["possible_points"]) == 200
            and float(row["point_coverage_fraction"]) == 1.0
            for row in coverage
        ),
        "Coverage support drift",
    )
    train_subjects = set(map(str, config["data"]["split"]["train_subject_keys"]))
    validation_subjects = set(
        map(str, config["data"]["split"]["val_subject_keys"])
    )
    for row in coverage:
        expected_split = (
            "train" if row["subject_key"] in train_subjects else "validation"
        )
        _require(
            row["subject_key"] in train_subjects | validation_subjects
            and row["development_split"] == expected_split,
            "Coverage split/subject drift",
        )

    with (root / "population_frozen_apply_audit.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        apply_rows = list(csv.DictReader(handle))
    _require(len(apply_rows) == 23, "Apply-audit subject count drift")
    _require(
        all(
            int(row["sample_count"]) == 60
            and int(row["global_parameter_bundle_fit_count"]) == 1
            and int(row["subject_specific_fit_calls"]) == 0
            and int(row["validation_refit_calls"]) == 0
            and int(row["protected_array_dereference_count"]) == 0
            for row in apply_rows
        ),
        "Apply-audit fit/refit count drift",
    )
    expected_bundle_sha = _json(root / "manifest.json")["parameter_bundle"][
        "bundle_sha256"
    ]
    for row in apply_rows:
        expected_participation = row["subject_key"] in train_subjects
        _require(
            (row["participated_in_global_parameter_fit"] == "True")
            == expected_participation,
            "Apply-audit participation drift",
        )
        _require(
            row["parameter_bundle_sha256"] == expected_bundle_sha,
            "Apply-audit bundle identity drift",
        )

    leakage = _json(root / "leakage_audit.json")
    _require(leakage["validation_fit_calls"] == 0, "Validation fit call recorded")
    _require(
        leakage["validation_normalization_calls"] == 0,
        "Validation normalization call recorded",
    )
    _require(
        leakage["validation_loaded_after_parameter_and_normalization_freeze"] is True,
        "Validation load/freeze ordering drift",
    )
    _require(
        leakage["protected_array_dereference_count"] == 0,
        "Protected array dereference recorded",
    )
    _require(
        leakage["joint_and_eeg_only_exact_parameter_bundle"] is True,
        "Paired parameter-bundle claim drift",
    )
    _require(
        leakage["raw_view_and_target_separate"] is True,
        "Raw-view/target separation claim drift",
    )
    return {
        "coverage_row_count": len(coverage),
        "coverage_fraction": 1.0,
        "apply_audit_subject_count": len(apply_rows),
        "validation_fit_calls": 0,
        "validation_normalization_calls": 0,
        "protected_array_dereference_count": 0,
        "validation_loaded_after_freeze": True,
    }


def _check_readers(
    root: Path,
    config: Mapping[str, Any],
    *,
    verify_dataset_reader: bool,
) -> dict[str, Any]:
    teacher = SharedDriverTrajectorySidecar(
        root / "trajectory_targets",
        expected_scope="population_frozen",
        expected_family="adaptive_joint_full_trajectory",
    )
    raw = PhysiologyRawViewRegistry(root / "raw_view_registry")
    _require(len(teacher) == len(raw) == 1380, "Sidecar reader count drift")
    _require(teacher.sample_keys == raw.sample_keys, "Sidecar reader join drift")
    result: dict[str, Any] = {
        "trajectory_reader_count": len(teacher),
        "raw_view_reader_count": len(raw),
        "reader_sample_order_equal": True,
        "dataset_reader_verified": bool(verify_dataset_reader),
    }
    if not verify_dataset_reader:
        return result
    data = config["data"]
    common = {
        "cache_root": str(REPO_ROOT / str(data["cache_root"])),
        "raw_view_registry_root": str(root / "raw_view_registry"),
        "trajectory_sidecar_root": str(root / "trajectory_targets"),
        "expected_teacher_scope": "population_frozen",
        "expected_target_family": "adaptive_joint_full_trajectory",
        "require_trajectory_target": True,
        "restrict_to_registered_views": True,
        "dataset_ids": (str(data["dataset_id"]),),
        "task_namespaces": ("eeg_fnirs_single_trial:mental_arithmetic",),
        "window_duration_s": float(data["window_duration_s"]),
        "window_offset_s": float(data["window_offset_s"]),
        "reject_unknown_labels": True,
        "eeg_signal_branch": str(data["eeg_signal_branch"]),
    }
    train = SharedDriverWindowDataset(
        subject_keys=tuple(data["split"]["train_subject_keys"]), **common
    )
    validation = SharedDriverWindowDataset(
        subject_keys=tuple(data["split"]["val_subject_keys"]), **common
    )
    _require(len(train) == 1080, "Dataset reader train count drift")
    _require(len(validation) == 300, "Dataset reader validation count drift")
    train_keys = [
        train._sample_key_by_base_index[entry.base_index]
        for entry in train.entries
    ]
    validation_keys = [
        validation._sample_key_by_base_index[entry.base_index]
        for entry in validation.entries
    ]
    teacher_arrays = teacher.arrays
    expected_train_keys = set(
        np.asarray(teacher_arrays["sample_key"]).astype(str)[
            np.asarray(teacher_arrays["development_split"]).astype(str) == "train"
        ].tolist()
    )
    expected_validation_keys = set(
        np.asarray(teacher_arrays["sample_key"]).astype(str)[
            np.asarray(teacher_arrays["development_split"]).astype(str)
            == "validation"
        ].tolist()
    )
    _require(
        len(set(train_keys)) == len(train_keys)
        and set(train_keys) == expected_train_keys,
        "Dataset reader train join drift",
    )
    _require(
        len(set(validation_keys)) == len(validation_keys)
        and set(validation_keys) == expected_validation_keys,
        "Dataset reader validation join drift",
    )
    result.update(
        {
            "dataset_train_count": len(train),
            "dataset_validation_count": len(validation),
        }
    )
    return result


def audit_bundle(
    bundle_root: Path,
    config_path: Path,
    *,
    verify_dataset_reader: bool,
) -> dict[str, Any]:
    root = bundle_root.resolve()
    config_path = config_path.resolve()
    before = _tree_hashes(root)
    top = _json(root / "manifest.json")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_population_config(config)
    _require(top["schema"] == "shared_driver_r1p_population_frozen_bundle_v1", "Top schema drift")
    _require(top["sample_count"] == 1380, "Top sample count drift")
    _require(top["train_sample_count"] == 1080, "Top train count drift")
    _require(top["validation_sample_count"] == 300, "Top validation count drift")
    _require(top["promotion_eligible"] is False, "Unqualified bundle marked promotable")
    _require(
        top["promotion_blocker"] == "population_frozen_teacher_panel_not_run",
        "Qualification blocker drift",
    )
    _require(top["protected_open"] is False, "Protected data marked open")
    _require(top["protected_test_included"] is False, "Protected data marked included")
    _require(_sha256(config_path) == top["source"]["config_sha256"], "Config hash drift")
    builder_path = Path(str(top["source"]["builder"]))
    _require(
        builder_path.is_file()
        and _sha256(builder_path) == top["source"]["builder_sha256"],
        "Builder hash drift",
    )

    hashes = _check_manifest_hashes(root, top)
    parameter = _check_parameter_roundtrip(root, top)
    join = _check_target_and_raw_join(root, config, top)
    coverage = _check_coverage_and_leakage(root, config, join)
    readers = _check_readers(
        root, config, verify_dataset_reader=verify_dataset_reader
    )
    after = _tree_hashes(root)
    _require(before == after, "Audit modified the source bundle")
    return {
        "schema": SCHEMA,
        "status": "passed",
        "audit_scope": "structure_leakage_serialization_only",
        "validation_physiology_effects_evaluated": False,
        "qualification_registry_state_at_audit": "frozen_before_audit",
        "qualification_registry_thresholds_informed": False,
        "bundle_root": str(root),
        "bundle_tree_unchanged": True,
        "bundle_file_count": len(before),
        "source_bundle_tree_sha256": _json_sha256(before),
        "source_file_hashes": hashes,
        "counts_and_join": join,
        "parameter_bundle": parameter,
        "coverage_and_leakage": coverage,
        "readers": readers,
        "qualification_boundary": {
            "promotion_eligible": False,
            "promotion_blocker": "population_frozen_teacher_panel_not_run",
            "r1p_teacher_panel_evaluated": False,
        },
        "checks": {
            "all_manifest_and_array_hashes_match": True,
            "registered_1080_300_split_matches": True,
            "subject_session_condition_coverage_complete": True,
            "numeric_arrays_finite_without_effect_summarization": True,
            "target_masks_complete": True,
            "joint_and_eeg_only_share_parameter_fold_and_gauge": True,
            "raw_view_target_join_exact": True,
            "protected_array_dereference_zero": True,
            "parameter_bundle_roundtrip_exact": True,
            "source_bundle_unchanged": True,
        },
    }


def _markdown(audit: Mapping[str, Any]) -> str:
    join = audit["counts_and_join"]
    parameter = audit["parameter_bundle"]
    leakage = audit["coverage_and_leakage"]
    readers = audit["readers"]
    return f"""# R1-P bundle structural and leakage audit

Status: **PASSED**

The qualification registry was already frozen before this independent audit.
The audit was intentionally limited to structure, provenance, leakage
boundaries and serialization. It did not compute or inspect validation
physiology effects and did not influence the frozen qualification thresholds.

## Verified facts

- Exact split: {join["train_sample_count"]} train and
  {join["validation_sample_count"]} validation rows; {join["sample_count"]}
  total rows across {join["subject_count"]} development subjects.
- Coverage: {join["session_condition_cells"]} subject × session × condition
  cells, each with {join["windows_per_subject_session_condition"][0]} windows,
  ten 2 s patches and complete point support.
- Join: raw-view and trajectory registries have identical unique sample keys,
  row order and metadata.
- Frozen provenance: all rows use one parameter fold, one bundle SHA and one
  shared-driver gauge; joint and EEG-only targets have identical masks.
- Leakage boundary: validation fit calls = {leakage["validation_fit_calls"]},
  validation normalization calls = {leakage["validation_normalization_calls"]},
  protected dereferences = {leakage["protected_array_dereference_count"]}.
- Readers: trajectory = {readers["trajectory_reader_count"]}, raw-view =
  {readers["raw_view_reader_count"]}, dataset train =
  {readers.get("dataset_train_count", "skipped")}, dataset validation =
  {readers.get("dataset_validation_count", "skipped")}.
- Serialization: all {parameter["array_field_count"]} parameter-array fields
  and all fit scalars round-trip exactly; bundle SHA is
  `{parameter["bundle_sha256"]}`.
- The source bundle tree was unchanged by the audit.

## Qualification boundary

The bundle remains non-promotable. The population-frozen teacher panel was not
evaluated here, and no R1-P threshold or validation physiological result is
reported.
"""


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    audit = audit_bundle(
        args.bundle_root,
        args.config,
        verify_dataset_reader=not args.skip_dataset_reader,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "AUDIT.md").write_text(
        _markdown(audit),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
