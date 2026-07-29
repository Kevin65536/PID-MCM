#!/usr/bin/env python3
"""Build an exploratory R1-D full-trajectory teacher/raw-view bundle.

This adapter intentionally does not refit the E0 state-space model.  It turns
the admitted development-crossfit E0 trajectories into the R-series P1/P2
storage contract while preserving their restricted scientific scope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.physiology_semantic_targets import target_sample_key  # noqa: E402
from src.data.shared_driver_targets import (  # noqa: E402
    RAW_VIEW_ARRAY_SCHEMA,
    RAW_VIEW_REGISTRY_SCHEMA,
    SHARED_DRIVER_ARRAY_SCHEMA,
    SHARED_DRIVER_SIDECAR_SCHEMA,
    string_array,
)
from src.data.unified_physiology import UnifiedPhysiologyWindowDataset  # noqa: E402


BUNDLE_SCHEMA = "shared_driver_r1d_bundle_v1"
TARGET_FAMILY = "adaptive_joint_full_trajectory"
TARGET_VERSION = "r1_d_loto_full_trajectory_v1"
TEACHER_SCOPE = "development_crossfit"
ARCHITECTURE_GENERATION = "shared_driver_semantic_vq_v1"
JOINT_MODEL = "adaptive_joint"
EEG_ONLY_MODEL = "adaptive_eeg_only"
MODELS = (JOINT_MODEL, EEG_ONLY_MODEL)
SPATIAL_MODE = "local"
PATCH_COUNT = 10
POINTS_PER_PATCH = 20
POINT_COUNT = PATCH_COUNT * POINTS_PER_PATCH
PROTECTED_SUBJECT_IDS = frozenset(f"subject_{index:02d}" for index in range(24, 30))


@dataclass(frozen=True, order=True)
class FoldKey:
    condition_id: str
    subject: str
    heldout_trial: int


@dataclass(frozen=True)
class MeasuredEvent:
    dataset_id: str
    record_id: str
    event_index: int
    condition: str
    dataset: UnifiedPhysiologyWindowDataset
    dataset_index: int


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _subject_parts(subject_key: str) -> tuple[str, str]:
    parts = str(subject_key).split("|", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid subject key: {subject_key!r}")
    return parts[0], parts[1]


def _split_registry(
    split_config: Mapping[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    try:
        split = split_config["data"]["split"]
        train = {str(value) for value in split["train_subject_keys"]}
        validation = {str(value) for value in split["val_subject_keys"]}
        protected = {str(value) for value in split["test_subject_keys"]}
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "Split config requires data.split train_subject_keys, "
            "val_subject_keys, and test_subject_keys"
        ) from exc
    if not train or not validation:
        raise ValueError("R1-D requires non-empty development train and validation splits")
    if (train & validation) or (train & protected) or (validation & protected):
        raise ValueError("Train, validation, and protected subject sets must be disjoint")
    protected_ids = {_subject_parts(key)[1] for key in protected}
    if protected_ids != PROTECTED_SUBJECT_IDS:
        raise ValueError(
            "Protected split must remain exactly subject_24 through subject_29; "
            f"observed={sorted(protected_ids)}"
        )
    return train, validation, protected


def _event_lookup(
    source_config: Mapping[str, Any],
    *,
    development_subject_keys: set[str],
    protected_subject_keys: set[str],
) -> dict[FoldKey, MeasuredEvent]:
    """Map E0 fold ordinals to unified measured-window identities.

    The unified index currently has no subject predicate at construction time,
    so it may enumerate metadata for all records.  This function checks source
    subject authorization before construction and dereferences measured arrays
    only for explicitly admitted development subjects.
    """

    data_cfg = source_config["data"]
    lookup: dict[FoldKey, MeasuredEvent] = {}
    for condition in data_cfg["conditions"]:
        condition_id = str(condition["condition_id"])
        dataset_id = str(condition["dataset_id"])
        allowed_subjects = {str(value) for value in condition["subjects"]}
        allowed_keys = {f"{dataset_id}|{subject}" for subject in allowed_subjects}
        forbidden = allowed_keys & protected_subject_keys
        if forbidden or (allowed_subjects & PROTECTED_SUBJECT_IDS):
            raise RuntimeError(
                f"Source condition {condition_id!r} requests protected subjects: "
                f"{sorted(forbidden or (allowed_subjects & PROTECTED_SUBJECT_IDS))}"
            )
        unexpected = allowed_keys - development_subject_keys
        if unexpected:
            raise RuntimeError(
                f"Source condition {condition_id!r} contains non-development subjects: "
                f"{sorted(unexpected)}"
            )

        dataset = UnifiedPhysiologyWindowDataset(
            cache_root=data_cfg["cache_root"],
            dataset_ids=(dataset_id,),
            window_duration_s=float(data_cfg["window_duration_s"]),
            window_offset_s=float(data_cfg["window_offset_s"]),
            eeg_signal_branch=str(condition["eeg_signal_branch"]),
        )
        selected: dict[str, list[tuple[Any, int]]] = defaultdict(list)
        for dataset_index, ref in enumerate(dataset.windows):
            subject = str(ref.record.canonical_subject_id)
            if subject not in allowed_subjects:
                continue
            if str(ref.record.base_record_id) != str(condition["record_id"]):
                continue
            if str(ref.event.get("label")) != str(condition["target_label"]):
                continue
            if len(selected[subject]) < int(condition["max_trials_per_subject"]):
                selected[subject].append((ref, dataset_index))
        for subject, refs in selected.items():
            for heldout, (ref, dataset_index) in enumerate(refs):
                key = FoldKey(condition_id, subject, heldout)
                if key in lookup:
                    raise ValueError(f"Duplicate measured fold mapping: {key}")
                lookup[key] = MeasuredEvent(
                    dataset_id=str(ref.record.dataset_id),
                    record_id=str(ref.record.base_record_id),
                    event_index=int(ref.event.get("event_index", heldout)),
                    condition=str(condition["target_label"]),
                    dataset=dataset,
                    dataset_index=int(dataset_index),
                )
    return lookup


def _trajectory_groups(
    trajectories_path: Path,
) -> dict[tuple[FoldKey, str], list[dict[str, str]]]:
    groups: dict[tuple[FoldKey, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(trajectories_path):
        if str(row.get("spatial_mode")) != SPATIAL_MODE:
            continue
        model = str(row.get("model"))
        if model not in MODELS:
            continue
        key = FoldKey(
            str(row["condition_id"]),
            str(row["subject"]),
            int(row["heldout_trial"]),
        )
        groups[(key, model)].append(row)
    if not groups:
        raise RuntimeError("No local adaptive joint/EEG-only trajectories found")
    return groups


def _fit_lookup(fits_path: Path) -> dict[FoldKey, dict[str, str]]:
    lookup: dict[FoldKey, dict[str, str]] = {}
    for row in _read_csv(fits_path):
        if str(row.get("spatial_mode")) != SPATIAL_MODE:
            continue
        key = FoldKey(
            str(row["condition_id"]),
            str(row["subject"]),
            int(row["heldout_trial"]),
        )
        if key in lookup:
            raise ValueError(f"Duplicate local fit row: {key}")
        lookup[key] = row
    if not lookup:
        raise RuntimeError("No local fit-parameter rows found")
    return lookup


def _float_column(rows: Sequence[Mapping[str, str]], name: str) -> np.ndarray:
    try:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Trajectory field {name!r} is missing or non-numeric") from exc


def _constant_fields(
    rows: Sequence[Mapping[str, str]],
    names: Sequence[str],
) -> dict[str, float]:
    payload: dict[str, float] = {}
    for name in names:
        values = _float_column(rows, name)
        if not np.isfinite(values).all():
            raise ValueError(f"Gauge field {name!r} contains non-finite values")
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"Gauge field {name!r} varies within one fold/model")
        payload[name] = float(values[0])
    return payload


def reshape_full_trajectory(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_start_s: float,
    sample_rate_hz: float,
) -> dict[str, Any]:
    """Validate and reshape one 20-second E0 trajectory onto [10,20]."""

    ordered = sorted(rows, key=lambda row: float(row["time_s"]))
    if len(ordered) != POINT_COUNT:
        raise ValueError(
            f"Expected {POINT_COUNT} trajectory points, got {len(ordered)}"
        )
    time_s = _float_column(ordered, "time_s")
    expected = expected_start_s + np.arange(POINT_COUNT, dtype=np.float64) / float(
        sample_rate_hz
    )
    if not np.allclose(time_s, expected, rtol=0.0, atol=1e-8):
        maximum = float(np.max(np.abs(time_s - expected)))
        raise ValueError(f"Trajectory time grid is misaligned (max delta={maximum:g} s)")
    if len(np.unique(time_s)) != POINT_COUNT:
        raise ValueError("Trajectory time grid contains duplicate points")

    values = _float_column(ordered, "target_shared_driver")
    uncertainty = _float_column(ordered, "target_shared_driver_std")
    point_mask = np.isfinite(values) & np.isfinite(uncertainty) & (uncertainty >= 0.0)
    safe_values = np.where(point_mask, values, 0.0)
    gauge_names = sorted(
        name for name in ordered[0] if str(name).startswith("gauge_")
    )
    if not gauge_names:
        raise ValueError("Trajectory rows contain no gauge provenance")
    return {
        "target": safe_values.reshape(PATCH_COUNT, POINTS_PER_PATCH),
        "point_mask": point_mask.reshape(PATCH_COUNT, POINTS_PER_PATCH),
        "time_s": time_s.reshape(PATCH_COUNT, POINTS_PER_PATCH),
        "gauge": _constant_fields(ordered, gauge_names),
        "source_payload": {
            "time_s": time_s.tolist(),
            "target_shared_driver": [
                None if not np.isfinite(value) else float(value) for value in values
            ],
            "target_shared_driver_std": [
                None if not np.isfinite(value) else float(value)
                for value in uncertainty
            ],
        },
    }


def _anchor_from_measured(
    sample: Mapping[str, Any],
    selected_fnirs: Sequence[str],
    fnirs_indices: Sequence[int],
) -> str:
    geometry = sample.get("channel_geometry", {}).get("fnirs", [])
    base_names: list[str] = []
    for name, index in zip(selected_fnirs, fnirs_indices):
        row = geometry[int(index)] if int(index) < len(geometry) else {}
        base = str(row.get("base_channel_name", "")).strip()
        if not base:
            base = str(name)
            for suffix in ("_HbO", "_HbR"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
        base_names.append(base)
    if len(set(base_names)) != 1 or not base_names[0]:
        raise ValueError(
            f"Selected fNIRS channels do not form one paired anchor: {selected_fnirs}"
        )
    return base_names[0]


def _validate_raw_view(
    measured: MeasuredEvent,
    selected_eeg: Sequence[str],
    selected_fnirs: Sequence[str],
) -> str:
    if len(selected_eeg) != 6 or len(set(selected_eeg)) != 6:
        raise ValueError("Each R1-D fold must select six unique EEG channels")
    if len(selected_fnirs) != 2 or len(set(selected_fnirs)) != 2:
        raise ValueError("Each R1-D fold must select two unique fNIRS channels")
    sample = measured.dataset[measured.dataset_index]
    eeg_names = [str(value) for value in sample["channel_names"]["eeg"]]
    fnirs_names = [str(value) for value in sample["channel_names"]["fnirs"]]
    missing_eeg = sorted(set(selected_eeg) - set(eeg_names))
    missing_fnirs = sorted(set(selected_fnirs) - set(fnirs_names))
    if missing_eeg or missing_fnirs:
        raise ValueError(
            "Frozen source view is absent from measured data: "
            f"EEG={missing_eeg}, fNIRS={missing_fnirs}"
        )
    eeg_indices = [eeg_names.index(name) for name in selected_eeg]
    fnirs_indices = [fnirs_names.index(name) for name in selected_fnirs]
    bad_eeg = np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)[eeg_indices]
    bad_fnirs = np.asarray(sample["bad_channel_mask"]["fnirs"], dtype=bool)[
        fnirs_indices
    ]
    if bad_eeg.any() or bad_fnirs.any():
        raise ValueError("Frozen source view selects a currently rejected measured channel")
    roles = [
        str(sample["component_roles"]["fnirs"][index]) for index in fnirs_indices
    ]
    if roles != ["HbO", "HbR"]:
        raise ValueError(
            f"Frozen fNIRS selection must be ordered [HbO,HbR], got {roles}"
        )
    return _anchor_from_measured(sample, selected_fnirs, fnirs_indices)


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


def _source_layout(source_run: Path) -> tuple[Path, dict[str, Path]]:
    base = source_run / "base_model" if (source_run / "base_model").is_dir() else source_run
    paths = {
        "source_config": base / "config.yaml",
        "trajectories": base / "trajectories.csv",
        "fit_parameters": base / "fit_parameters.csv",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    optional = {
        "source_run_manifest": source_run / "manifest.json",
        "source_model_manifest": base / "manifest.json",
        "source_target_contract": source_run / "target_contract.json",
    }
    paths.update({name: path for name, path in optional.items() if path.is_file()})
    return base, paths


def _normalization(
    records: Sequence[dict[str, Any]],
    train_subject_keys: set[str],
) -> dict[str, Any]:
    selected = [
        np.asarray(record["joint_target"], dtype=np.float64)[
            np.asarray(record["joint_point_mask"], dtype=bool)
        ]
        for record in records
        if record["subject_key"] in train_subject_keys
    ]
    if not selected:
        raise RuntimeError("No supported training-subject points for normalization")
    values = np.concatenate(selected)
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale < 1e-6:
        raise RuntimeError(
            f"Degenerate train-only shared-driver normalization: mean={mean}, scale={scale}"
        )
    payload = {
        "policy": "scalar_joint_train_subject_points_v1",
        "fit_subject_keys": sorted(train_subject_keys),
        "fit_sample_count": sum(
            record["subject_key"] in train_subject_keys for record in records
        ),
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


def _assemble_records(
    *,
    source_config: Mapping[str, Any],
    trajectory_groups: Mapping[tuple[FoldKey, str], Sequence[Mapping[str, str]]],
    fit_lookup: Mapping[FoldKey, Mapping[str, str]],
    event_lookup: Mapping[FoldKey, MeasuredEvent],
    train_subject_keys: set[str],
    validation_subject_keys: set[str],
    protected_subject_keys: set[str],
    source_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folds_by_model = {
        model: {key for key, row_model in trajectory_groups if row_model == model}
        for model in MODELS
    }
    if folds_by_model[JOINT_MODEL] != folds_by_model[EEG_ONLY_MODEL]:
        raise RuntimeError("Joint and EEG-only trajectory fold identities differ")
    fold_keys = folds_by_model[JOINT_MODEL]
    if fold_keys != set(fit_lookup):
        raise RuntimeError("Trajectory and fit-parameter fold identities differ")
    if fold_keys != set(event_lookup):
        raise RuntimeError(
            "Trajectory and unified measured-event fold identities differ: "
            f"missing={sorted(fold_keys - set(event_lookup))}, "
            f"unexpected={sorted(set(event_lookup) - fold_keys)}"
        )

    expected_start_s = float(source_config["data"]["window_offset_s"])
    sample_rate_hz = float(source_config["analysis"]["ssm"]["fs_hz"])
    window_duration_s = float(source_config["data"]["window_duration_s"])
    if not np.isclose(window_duration_s * sample_rate_hz, POINT_COUNT):
        raise ValueError(
            "Source time contract does not equal 20 seconds × 10 Hz = 200 points"
        )

    development_subject_keys = train_subject_keys | validation_subject_keys
    records: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for fold in sorted(fold_keys):
        measured = event_lookup[fold]
        subject_key = f"{measured.dataset_id}|{fold.subject}"
        if subject_key in protected_subject_keys or fold.subject in PROTECTED_SUBJECT_IDS:
            raise RuntimeError(f"Protected source fold is forbidden: {fold}")
        if subject_key not in development_subject_keys:
            raise RuntimeError(f"Non-development source fold is forbidden: {fold}")

        joint = reshape_full_trajectory(
            trajectory_groups[(fold, JOINT_MODEL)],
            expected_start_s=expected_start_s,
            sample_rate_hz=sample_rate_hz,
        )
        eeg_only = reshape_full_trajectory(
            trajectory_groups[(fold, EEG_ONLY_MODEL)],
            expected_start_s=expected_start_s,
            sample_rate_hz=sample_rate_hz,
        )
        if not np.array_equal(joint["time_s"], eeg_only["time_s"]):
            raise ValueError(f"Joint/EEG-only time grids differ for {fold}")
        for coordinate in ("gauge_shared_driver_scale", "gauge_shared_driver_offset"):
            if not np.isclose(
                float(joint["gauge"][coordinate]),
                float(eeg_only["gauge"][coordinate]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Joint/EEG-only shared-driver gauge differs for {fold}: {coordinate}"
                )

        fit = dict(fit_lookup[fold])
        selected_eeg = tuple(
            value for value in str(fit["selected_eeg_channels"]).split("|") if value
        )
        selected_fnirs = tuple(
            value for value in str(fit["selected_fnirs_channels"]).split("|") if value
        )
        anchor_id = _validate_raw_view(measured, selected_eeg, selected_fnirs)
        sample_key = target_sample_key(
            measured.dataset_id,
            fold.subject,
            measured.record_id,
            measured.event_index,
        )
        if sample_key in seen_samples:
            raise ValueError(f"Duplicate measured sample identity: {sample_key}")
        seen_samples.add(sample_key)

        gauge_payload = {
            JOINT_MODEL: joint["gauge"],
            EEG_ONLY_MODEL: eeg_only["gauge"],
            "shared_driver_coordinate_assertion": (
                "joint_and_eeg_only_scale_offset_equal"
            ),
        }
        fit_source_hash = _json_sha256(fit)
        teacher_source_hash = _json_sha256(
            {
                "fold": {
                    "condition_id": fold.condition_id,
                    "subject": fold.subject,
                    "heldout_trial": fold.heldout_trial,
                },
                "input_hashes": dict(source_hashes),
                JOINT_MODEL: joint["source_payload"],
                EEG_ONLY_MODEL: eeg_only["source_payload"],
                "fit_row_sha256": fit_source_hash,
            }
        )
        parameter_fold = (
            f"development_crossfit:{fold.condition_id}:{fold.subject}:"
            f"loto={fold.heldout_trial}"
        )
        records.append(
            {
                "sample_key": sample_key,
                "dataset_id": measured.dataset_id,
                "subject": fold.subject,
                "subject_key": subject_key,
                "session_id": measured.record_id,
                "condition": measured.condition,
                "condition_id": fold.condition_id,
                "event_index": measured.event_index,
                "heldout_trial": fold.heldout_trial,
                "split": (
                    "train"
                    if subject_key in train_subject_keys
                    else "validation"
                ),
                "joint_target": joint["target"],
                "joint_point_mask": joint["point_mask"],
                "eeg_only_target": eeg_only["target"],
                "eeg_only_point_mask": eeg_only["point_mask"],
                "time_s": joint["time_s"],
                "teacher_parameter_fold": parameter_fold,
                "teacher_gauge_hash": _json_sha256(gauge_payload),
                "teacher_source_hash": teacher_source_hash,
                "selected_eeg_channels": selected_eeg,
                "selected_fnirs_channels": selected_fnirs,
                "anchor_id": anchor_id,
                "selection_fold": parameter_fold,
                "selection_source_hash": fit_source_hash,
            }
        )

    observed_subjects = {record["subject_key"] for record in records}
    if observed_subjects != development_subject_keys:
        raise RuntimeError(
            "R1-D target subject coverage is not the complete registered "
            "development cohort: "
            f"missing={sorted(development_subject_keys - observed_subjects)}, "
            f"unexpected={sorted(observed_subjects - development_subject_keys)}"
        )
    normalization = _normalization(records, train_subject_keys)
    mean = float(normalization["mean"])
    scale = float(normalization["scale"])
    for record in records:
        for target_key, mask_key in (
            ("joint_target", "joint_point_mask"),
            ("eeg_only_target", "eeg_only_point_mask"),
        ):
            target = (np.asarray(record[target_key], dtype=np.float64) - mean) / scale
            mask = np.asarray(record[mask_key], dtype=bool)
            record[target_key] = np.where(mask, target, 0.0).astype(np.float32)
    return records, normalization


def _write_teacher_artifact(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    normalization: Mapping[str, Any],
    source: Mapping[str, Any],
    split: Mapping[str, Any],
    git_state: Mapping[str, Any],
) -> dict[str, Any]:
    root.mkdir()
    arrays_path = root / "arrays.npz"
    sample_keys = [str(record["sample_key"]) for record in records]
    joint = np.stack([record["joint_target"] for record in records]).astype(np.float32)
    joint_mask = np.stack([record["joint_point_mask"] for record in records]).astype(
        bool
    )
    eeg_only = np.stack([record["eeg_only_target"] for record in records]).astype(
        np.float32
    )
    eeg_only_mask = np.stack(
        [record["eeg_only_point_mask"] for record in records]
    ).astype(bool)
    target_time_s = np.stack([record["time_s"] for record in records]).astype(
        np.float32
    )
    np.savez_compressed(
        arrays_path,
        schema=np.asarray(SHARED_DRIVER_ARRAY_SCHEMA),
        sample_key=string_array(sample_keys),
        dataset_id=string_array([record["dataset_id"] for record in records]),
        subject_id=string_array([record["subject"] for record in records]),
        subject_key=string_array([record["subject_key"] for record in records]),
        session_id=string_array([record["session_id"] for record in records]),
        condition=string_array([record["condition"] for record in records]),
        condition_id=string_array([record["condition_id"] for record in records]),
        event_index=np.asarray([record["event_index"] for record in records], dtype=np.int64),
        heldout_trial=np.asarray(
            [record["heldout_trial"] for record in records], dtype=np.int64
        ),
        development_split=string_array([record["split"] for record in records]),
        target_time_s=target_time_s,
        target_shared_driver=joint,
        target_valid_mask=joint_mask.any(axis=-1),
        target_point_valid_mask=joint_mask,
        target_eeg_only_driver=eeg_only,
        eeg_only_valid_mask=eeg_only_mask.any(axis=-1),
        eeg_only_point_valid_mask=eeg_only_mask,
        teacher_scope=string_array([TEACHER_SCOPE] * len(records)),
        teacher_parameter_fold=string_array(
            [record["teacher_parameter_fold"] for record in records]
        ),
        teacher_gauge_hash=string_array(
            [record["teacher_gauge_hash"] for record in records]
        ),
        teacher_source_hash=string_array(
            [record["teacher_source_hash"] for record in records]
        ),
    )
    manifest = {
        "schema": SHARED_DRIVER_SIDECAR_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_generation": ARCHITECTURE_GENERATION,
        "target_family": TARGET_FAMILY,
        "target_version": TARGET_VERSION,
        "teacher_scope": TEACHER_SCOPE,
        "teacher_parameter_scope": "subject_specific_leave_one_trial",
        "target_identity": "privileged_joint_shared_driver_proxy_not_ground_truth",
        "trajectory_shape": [PATCH_COUNT, POINTS_PER_PATCH],
        "sample_rate_hz": 10.0,
        "patch_duration_s": 2.0,
        "arrays_file": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "sample_count": len(records),
        "sample_order_sha256": hashlib.sha256(
            "\n".join(sample_keys).encode("utf-8")
        ).hexdigest(),
        "normalization": dict(normalization),
        "paired_control": {
            "joint_model": JOINT_MODEL,
            "eeg_only_model": EEG_ONLY_MODEL,
            "time_grid_equal_per_fold": True,
            "shared_driver_scale_offset_equal_per_fold": True,
            "model_specific_full_gauge_bundles_preserved_in_teacher_gauge_hash": True,
            "r1p_same_parameter_bundle_claimed": False,
        },
        "source": dict(source),
        "split": dict(split),
        "promotion_eligible": False,
        "allowed_use": "R2-D/R3-D_exploration_only",
        "protected_open": False,
        "protected_test_included": False,
        **dict(git_state),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _write_raw_view_artifact(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    split: Mapping[str, Any],
    git_state: Mapping[str, Any],
) -> dict[str, Any]:
    root.mkdir()
    arrays_path = root / "arrays.npz"
    sample_keys = [str(record["sample_key"]) for record in records]
    np.savez_compressed(
        arrays_path,
        schema=np.asarray(RAW_VIEW_ARRAY_SCHEMA),
        sample_key=string_array(sample_keys),
        dataset_id=string_array([record["dataset_id"] for record in records]),
        subject_id=string_array([record["subject"] for record in records]),
        subject_key=string_array([record["subject_key"] for record in records]),
        session_id=string_array([record["session_id"] for record in records]),
        condition=string_array([record["condition"] for record in records]),
        event_index=np.asarray([record["event_index"] for record in records], dtype=np.int64),
        selected_eeg_channels=np.asarray(
            [record["selected_eeg_channels"] for record in records], dtype=np.str_
        ),
        selected_fnirs_channels=np.asarray(
            [record["selected_fnirs_channels"] for record in records], dtype=np.str_
        ),
        anchor_id=string_array([record["anchor_id"] for record in records]),
        selection_fold=string_array(
            [record["selection_fold"] for record in records]
        ),
        selection_source_hash=string_array(
            [record["selection_source_hash"] for record in records]
        ),
    )
    manifest = {
        "schema": RAW_VIEW_REGISTRY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_generation": ARCHITECTURE_GENERATION,
        "raw_view_policy": "source_e0_development_crossfit_geometry_selection_v1",
        "selection_scope": TEACHER_SCOPE,
        "selection_authority": (
            "independent_registry_joined_before_privileged_target"
        ),
        "arrays_file": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "sample_count": len(records),
        "sample_order_sha256": hashlib.sha256(
            "\n".join(sample_keys).encode("utf-8")
        ).hexdigest(),
        "eeg_channel_count": 6,
        "fnirs_channel_count": 2,
        "source": dict(source),
        "split": dict(split),
        "promotion_eligible": False,
        "allowed_use": "R2-D/R3-D_exploration_only",
        "protected_open": False,
        "protected_test_included": False,
        **dict(git_state),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _write_coverage(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate: dict[tuple[str, str, str, str, int], dict[str, int]] = defaultdict(
        lambda: {
            "sample_count": 0,
            "joint_supported_points": 0,
            "eeg_only_supported_points": 0,
            "joint_supported_patches": 0,
            "eeg_only_supported_patches": 0,
        }
    )
    for record in records:
        joint_mask = np.asarray(record["joint_point_mask"], dtype=bool)
        eeg_mask = np.asarray(record["eeg_only_point_mask"], dtype=bool)
        for patch_index in range(PATCH_COUNT):
            key = (
                str(record["subject_key"]),
                str(record["session_id"]),
                str(record["condition"]),
                str(record["split"]),
                patch_index,
            )
            row = aggregate[key]
            row["sample_count"] += 1
            row["joint_supported_points"] += int(joint_mask[patch_index].sum())
            row["eeg_only_supported_points"] += int(eeg_mask[patch_index].sum())
            row["joint_supported_patches"] += int(joint_mask[patch_index].any())
            row["eeg_only_supported_patches"] += int(eeg_mask[patch_index].any())

    fields = [
        "subject_key",
        "session_id",
        "condition",
        "development_split",
        "patch_index",
        "patch_start_s",
        "sample_count",
        "possible_points",
        "joint_supported_points",
        "joint_point_coverage_fraction",
        "eeg_only_supported_points",
        "eeg_only_point_coverage_fraction",
        "joint_supported_patches",
        "eeg_only_supported_patches",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(aggregate):
            subject_key, session_id, condition, split, patch_index = key
            row = aggregate[key]
            possible = row["sample_count"] * POINTS_PER_PATCH
            writer.writerow(
                {
                    "subject_key": subject_key,
                    "session_id": session_id,
                    "condition": condition,
                    "development_split": split,
                    "patch_index": patch_index,
                    "patch_start_s": -5.0 + 2.0 * patch_index,
                    "sample_count": row["sample_count"],
                    "possible_points": possible,
                    "joint_supported_points": row["joint_supported_points"],
                    "joint_point_coverage_fraction": (
                        row["joint_supported_points"] / possible
                    ),
                    "eeg_only_supported_points": row[
                        "eeg_only_supported_points"
                    ],
                    "eeg_only_point_coverage_fraction": (
                        row["eeg_only_supported_points"] / possible
                    ),
                    "joint_supported_patches": row["joint_supported_patches"],
                    "eeg_only_supported_patches": row[
                        "eeg_only_supported_patches"
                    ],
                }
            )
    joint_points = sum(
        int(np.asarray(record["joint_point_mask"], dtype=bool).sum())
        for record in records
    )
    eeg_points = sum(
        int(np.asarray(record["eeg_only_point_mask"], dtype=bool).sum())
        for record in records
    )
    possible_points = len(records) * POINT_COUNT
    return {
        "sample_count": len(records),
        "coverage_row_count": len(aggregate),
        "possible_point_count": possible_points,
        "joint_supported_point_count": joint_points,
        "joint_point_coverage_fraction": joint_points / possible_points,
        "eeg_only_supported_point_count": eeg_points,
        "eeg_only_point_coverage_fraction": eeg_points / possible_points,
        "coverage_sha256": _sha256(path),
    }


def build_r1d_bundle(
    source_run: Path,
    split_config_path: Path,
    output_root: Path,
    *,
    event_lookup_override: Mapping[FoldKey, MeasuredEvent] | None = None,
) -> Path:
    """Build a fail-closed, non-promotable R1-D artifact bundle."""

    source_run = Path(source_run).resolve()
    split_config_path = Path(split_config_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    if not split_config_path.is_file():
        raise FileNotFoundError(split_config_path)
    _, source_paths = _source_layout(source_run)
    source_config = yaml.safe_load(
        source_paths["source_config"].read_text(encoding="utf-8")
    )
    split_config = yaml.safe_load(split_config_path.read_text(encoding="utf-8"))
    train_subjects, validation_subjects, protected_subjects = _split_registry(
        split_config
    )
    development_subjects = train_subjects | validation_subjects

    source_hashes = {
        name: _sha256(path) for name, path in sorted(source_paths.items())
    }
    source_hashes["split_config"] = _sha256(split_config_path)
    trajectory_groups = _trajectory_groups(source_paths["trajectories"])
    fits = _fit_lookup(source_paths["fit_parameters"])
    events = (
        dict(event_lookup_override)
        if event_lookup_override is not None
        else _event_lookup(
            source_config,
            development_subject_keys=development_subjects,
            protected_subject_keys=protected_subjects,
        )
    )
    records, normalization = _assemble_records(
        source_config=source_config,
        trajectory_groups=trajectory_groups,
        fit_lookup=fits,
        event_lookup=events,
        train_subject_keys=train_subjects,
        validation_subject_keys=validation_subjects,
        protected_subject_keys=protected_subjects,
        source_hashes=source_hashes,
    )
    git_state = _git_state()
    split_payload = {
        "train_subject_keys": sorted(train_subjects),
        "validation_subject_keys": sorted(validation_subjects),
        "protected_subject_keys": sorted(protected_subjects),
        "development_subject_count": len(development_subjects),
        "protected_open": False,
        "sha256": _json_sha256(split_config["data"]["split"]),
    }
    source_payload = {
        "source_run": str(source_run),
        "input_hashes": source_hashes,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": _sha256(Path(__file__).resolve()),
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.tmp-",
        dir=output_root.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        teacher_manifest = _write_teacher_artifact(
            temporary_root / "trajectory_targets",
            records,
            normalization=normalization,
            source=source_payload,
            split=split_payload,
            git_state=git_state,
        )
        raw_view_manifest = _write_raw_view_artifact(
            temporary_root / "raw_view_registry",
            records,
            source=source_payload,
            split=split_payload,
            git_state=git_state,
        )
        coverage_path = (
            temporary_root
            / "data_coverage_by_subject_session_condition_patch.csv"
        )
        coverage = _write_coverage(coverage_path, records)
        leakage_audit = {
            "schema": "shared_driver_r1d_leakage_audit_v1",
            "teacher_scope": TEACHER_SCOPE,
            "promotion_eligible": False,
            "development_crossfit_within_subject_other_trials_allowed": True,
            "normalization_fit_subject_keys": sorted(train_subjects),
            "normalization_validation_subjects_used": False,
            "normalization_protected_subjects_used": False,
            "raw_view_and_target_stored_as_separate_artifacts": True,
            "raw_view_selected_before_target_join": True,
            "protected_subject_keys": sorted(protected_subjects),
            "protected_sample_count": 0,
            "protected_open": False,
            "unified_loader_protected_array_dereference": False,
            "source_model_limitations": {
                "r1p_population_frozen": False,
                "same_parameter_bundle_joint_and_eeg_only": False,
                "shared_driver_coordinate_gauge_equal": True,
            },
        }
        _write_json(temporary_root / "leakage_audit.json", leakage_audit)
        root_manifest = {
            "schema": BUNDLE_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "architecture_generation": ARCHITECTURE_GENERATION,
            "target_family": TARGET_FAMILY,
            "target_version": TARGET_VERSION,
            "teacher_scope": TEACHER_SCOPE,
            "promotion_eligible": False,
            "allowed_use": "R2-D/R3-D_exploration_only",
            "sample_count": len(records),
            "trajectory_targets": {
                "path": "trajectory_targets",
                "manifest_sha256": _sha256(
                    temporary_root / "trajectory_targets" / "manifest.json"
                ),
                "arrays_sha256": teacher_manifest["arrays_sha256"],
            },
            "raw_view_registry": {
                "path": "raw_view_registry",
                "manifest_sha256": _sha256(
                    temporary_root / "raw_view_registry" / "manifest.json"
                ),
                "arrays_sha256": raw_view_manifest["arrays_sha256"],
            },
            "normalization": normalization,
            "coverage": coverage,
            "coverage_file": coverage_path.name,
            "leakage_audit": "leakage_audit.json",
            "source": source_payload,
            "split": split_payload,
            "protected_open": False,
            "protected_test_included": False,
            **git_state,
        }
        _write_json(temporary_root / "manifest.json", root_manifest)
        temporary_root.replace(output_root)
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        required=True,
        help="E0 run root containing base_model/trajectories.csv.",
    )
    parser.add_argument(
        "--split-config",
        required=True,
        help="Config containing the frozen data.split subject registry.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="New versioned bundle root; existing paths are never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_r1d_bundle(
        Path(args.source_run),
        Path(args.split_config),
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
