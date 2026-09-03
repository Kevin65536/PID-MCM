#!/usr/bin/env python3
"""Step 3: fit-only three-session LOSO diagnostic with a nominal recovery window.

Only subjects 01--18 and the MA records session_01/session_03/session_05 are
materialized.  Subjects 19--23 remain closed validation data and 24--29 remain
protected.  The one-dimensional kappa coordinate is an effective adaptation
coefficient, not a recovered biological rate or a qualification result.
"""

from __future__ import annotations

import argparse
import csv
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# The outer subject pool owns parallelism.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import scipy
import yaml
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_adaptive_shared_neural_ssm import (
    _apply_eeg_adapter,
    _fit_eeg_adapter,
    _paired_hbr_indices,
)
from experiments.evaluate_shared_neural_driver_unified import (
    Trial,
    _load_trials,
    _safe_corr,
    _select_active_hbo,
)
from experiments.evaluate_t3_measured_reconstruction_null import (
    PreparedTrial,
    _fit_models,
    _gaussian_negative_log_score,
    _masked_metrics,
    _replace_parameter_values,
    _require_full_support,
    load_config as load_measured_config,
)
from experiments.evaluate_t3a_balloon_robust_p0 import _atomic_csv, _atomic_json, _atomic_write
from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
from src.inference.t3a_balloon_robust_ssm import (
    BalloonConfig,
    BalloonFixedParameters,
    BalloonFreeParameters,
    BalloonObservationSpec,
    BalloonParameters,
    simulate_balloon,
    smooth_balloon,
)


SCHEMA = "t3_multisession_loso_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/t3_multisession_loso_v1.yaml"
MEASURED_CONFIG_PATH = "experiments/configs/physiology_semantic_tokenizer/t3_measured_reconstruction_null_v1.yaml"
MEASURED_CONFIG_SHA256 = "09317a7fd6eb50b44c829d1ad3f2e5a4319a2fe29e16544448d44095801a939e"
OUTPUT_ROOT = "experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso"
SESSION_IDS = ("session_01", "session_03", "session_05")
MODEL_IDS = ("M0_fixed", "M1_kappa_shared", "M1_kappa_session_nuisance")
MASK_IDS = ("recovery", "response")
TARGETS = ("HbO", "HbR")


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

    return {"commit": call("git", "rev-parse", "HEAD"), "status_short": call("git", "status", "--short")}


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject scope, split, and estimand drift before any cache is opened."""

    if config.get("schema") != SCHEMA:
        raise ValueError("Step 3 config schema mismatch")
    experiment = config.get("experiment", {})
    expected_experiment = {
        "name": SCHEMA,
        "scope": "fit_only_multisession_measured_exploratory",
        "measured_data_enabled": True,
        "validation_data_enabled": False,
        "protected_data_enabled": False,
        "qualification_eligible": False,
        "decision_eligibility": False,
    }
    if not isinstance(experiment, Mapping) or any(experiment.get(key) != value for key, value in expected_experiment.items()):
        raise ValueError("Step 3 experiment boundary mismatch")
    if not _is_int(experiment.get("seed")) or int(experiment["seed"]) != 20260902:
        raise ValueError("Step 3 seed must remain 20260902")

    sources = config.get("sources", {})
    if sources.get("measured_config") != MEASURED_CONFIG_PATH:
        raise ValueError("Step 3 must inherit the registered measured T3 coordinate")
    if sources.get("measured_config_sha256") != MEASURED_CONFIG_SHA256:
        raise ValueError("registered measured-config digest mismatch")
    if _sha256(REPO_ROOT / MEASURED_CONFIG_PATH) != MEASURED_CONFIG_SHA256:
        raise ValueError("measured source config has drifted")
    expected_metadata_paths = {
        "cache_manifest.json",
        "event_index/event_manifest.json",
        "event_index/events.jsonl",
        "event_index/alignment_reports.jsonl",
        "eeg_artifact_clean_v4/cache_manifest.json",
        "channel_geometry/geometry_manifest.json",
        "channel_geometry/channels.jsonl",
    }
    metadata_hashes = sources.get("metadata_sha256")
    if not isinstance(metadata_hashes, Mapping) or set(metadata_hashes) != expected_metadata_paths:
        raise ValueError("metadata digest registry is incomplete")

    data = config.get("data", {})
    if data.get("cache_root") != "data/cache/physiology_semantic_clean_v1":
        raise ValueError("Step 3 requires the canonical physiology cache")
    expected_data = {
        "dataset_id": "eeg_fnirs_single_trial",
        "target_label": "MA",
        "eeg_signal_branch": "raw_with_ocular_artifact",
        "trials_per_subject_session": 10,
        "expected_selected_record_count": 54,
        "expected_all_event_count": 1080,
        "expected_unique_target_trial_count": 540,
    }
    if any(data.get(key) != value for key, value in expected_data.items()):
        raise ValueError("Step 3 data identity/count contract mismatch")
    if list(map(str, data.get("subjects", ()))) != _subject_range(1, 18):
        raise ValueError("Step 3 loader is restricted to subjects 01--18")
    if list(map(str, data.get("validation_subjects_closed", ()))) != _subject_range(19, 23):
        raise ValueError("validation registry must remain closed")
    if list(map(str, data.get("protected_subjects_closed", ()))) != _subject_range(24, 29):
        raise ValueError("protected registry must remain closed")
    sessions = data.get("sessions", ())
    if not isinstance(sessions, Sequence) or tuple(str(item.get("record_id")) for item in sessions) != SESSION_IDS:
        raise ValueError(f"sessions must be exactly {SESSION_IDS}")
    if tuple(int(item.get("source_session_idx_zero_based", -1)) for item in sessions) != (1, 3, 5):
        raise ValueError("cache-session/source-session mapping has drifted")
    for key, expected in {
        "window_offset_s": -5.0,
        "window_duration_s": 30.0,
        "baseline_duration_s": 5.0,
        "task_duration_s": 10.0,
        "nominal_recovery_duration_s": 15.0,
        "next_event_guard_s": 0.5,
    }.items():
        if not math.isclose(float(data.get(key, float("nan"))), expected):
            raise ValueError(f"data.{key} must remain {expected}")

    analysis = config.get("analysis", {})
    if tuple(analysis.get("models", ())) != MODEL_IDS:
        raise ValueError(f"models must be exactly {MODEL_IDS}")
    if analysis.get("primary_model") != MODEL_IDS[2] or analysis.get("primary_reference") != MODEL_IDS[0]:
        raise ValueError("primary contrast must remain session-nuisance kappa versus M0")
    if analysis.get("primary_metric") != "variance_matched_gaussian_negative_log_score":
        raise ValueError("primary score contract mismatch")
    if not math.isclose(float(analysis.get("sampling_hz", 0.0)), 10.0) or int(analysis.get("active_hbo_channels", 0)) != 1:
        raise ValueError("Step 3 requires one HbO/HbR pair on the 10 Hz coordinate")
    mask_contract = {
        "heldout_input_mask": (0.0, 25.0),
        "primary_recovery_mask": (10.0, 25.0),
        "secondary_response_mask": (0.0, 25.0),
    }
    for name, (start, stop) in mask_contract.items():
        item = analysis.get(name, {})
        if not math.isclose(float(item.get("relative_start_s", float("nan"))), start) or not math.isclose(float(item.get("relative_stop_s", float("nan"))), stop):
            raise ValueError(f"{name} has drifted")
    coordinate = analysis.get("kappa_coordinate", {})
    if coordinate.get("interpretation") != "effective_one_dimensional_adaptation_coefficient_not_biological_trait":
        raise ValueError("kappa interpretation boundary mismatch")
    if tuple(float(value) for value in coordinate.get("bounds", ())) != (0.2, 1.5):
        raise ValueError("kappa bounds must remain [0.2, 1.5]")
    if not math.isclose(float(coordinate.get("prior_mean", 0.0)), 0.64) or not math.isclose(float(coordinate.get("prior_sd", 0.0)), 0.20):
        raise ValueError("kappa prior has drifted")
    if coordinate.get("heldout_rule") != "geometric_center_of_two_training_session_estimates":
        raise ValueError("heldout kappa rule mismatch")
    optimizer = analysis.get("optimizer", {})
    if not all(_is_int(optimizer.get(key)) for key in ("transformed_grid_points", "max_iterations", "workers")):
        raise ValueError("optimizer counts must be integers")
    if int(optimizer["transformed_grid_points"]) != 9 or not 1 <= int(optimizer["workers"]) <= 18:
        raise ValueError("optimizer grid/workers outside the registered contract")
    if int(optimizer["max_iterations"]) < 1 or float(optimizer.get("xatol", 0.0)) <= 0.0:
        raise ValueError("optimizer iteration/tolerance contract invalid")
    bootstrap = analysis.get("bootstrap", {})
    if bootstrap.get("unit") != "subject" or int(bootstrap.get("replicates", 0)) != 10000 or int(bootstrap.get("seed", -1)) != 20260902:
        raise ValueError("subject-block bootstrap contract mismatch")
    if config.get("output", {}).get("root") != OUTPUT_ROOT:
        raise ValueError(f"output.root must remain {OUTPUT_ROOT}")


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError("configuration must be a mapping")
    config = dict(value)
    validate_config(config)
    return config


def _metadata_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = REPO_ROOT / str(config["data"]["cache_root"])
    return {str(relative): root / str(relative) for relative in config["sources"]["metadata_sha256"]}


def _validate_metadata(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Validate the complete three-session window envelope without loading arrays."""

    paths = _metadata_paths(config)
    input_hashes: dict[str, str] = {}
    for relative, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"required metadata is unavailable: {path}")
        digest = _sha256(path)
        if digest != str(config["sources"]["metadata_sha256"][relative]):
            raise RuntimeError(f"metadata digest mismatch: {relative}")
        input_hashes[str(path.relative_to(REPO_ROOT))] = digest

    data = config["data"]
    subjects = tuple(map(str, data["subjects"]))
    subject_set = set(subjects)
    session_map = {str(item["record_id"]): int(item["source_session_idx_zero_based"]) for item in data["sessions"]}
    cache_root = REPO_ROOT / str(data["cache_root"])
    index = CleanPhysiologyCacheIndex(cache_root)
    records = [
        record for record in index.records
        if record.dataset_id == data["dataset_id"]
        and record.canonical_subject_id in subject_set
        and record.base_record_id in session_map
    ]
    expected_keys = {
        f"{data['dataset_id']}|{subject}|{session}"
        for subject in subjects for session in SESSION_IDS
    }
    if len(records) != int(data["expected_selected_record_count"]) or {record.join_key for record in records} != expected_keys:
        raise RuntimeError("metadata record registry is not the exact 18 x 3 selection")

    eeg_manifest = json.loads(paths["eeg_artifact_clean_v4/cache_manifest.json"].read_text(encoding="utf-8"))
    selected_eeg_rows = [row for row in eeg_manifest.get("records", ()) if str(row.get("join_key")) in expected_keys]
    eeg_rows = {str(row["join_key"]): row for row in selected_eeg_rows}
    if len(selected_eeg_rows) != 54 or set(eeg_rows) != expected_keys:
        raise RuntimeError("EEG length sidecar does not cover the exact selected records")

    offset_s = float(data["window_offset_s"])
    duration_s = float(data["window_duration_s"])
    guard_s = float(data["next_event_guard_s"])
    target = str(data["target_label"])
    rows: list[dict[str, Any]] = []
    all_event_count = 0
    source_hashes: dict[str, str] = {}
    for record in sorted(records, key=lambda item: (item.canonical_subject_id, item.base_record_id)):
        if not math.isclose(float(record.sample_rate_hz), 10.0):
            raise RuntimeError(f"unexpected fNIRS sample rate: {record.join_key}")
        contract = record.manifest.get("homer2_aligned_contract", {})
        shape = contract.get("summary", {}).get("shape", ())
        if (
            record.manifest.get("schema") != "clean_eeg_fnirs_cache_v1"
            or record.signal_branch != "homer2_wavelength_pair"
            or contract.get("array_key") != "homer2_aligned_fnirs"
            or len(shape) != 2
            or int(shape[1]) != 72
            or not math.isclose(float(contract.get("summary", {}).get("finite_fraction", float("nan"))), 1.0)
            or not record.npz_path.is_file()
        ):
            raise RuntimeError(f"missing fNIRS length contract: {record.join_key}")
        fnirs_samples = int(shape[0])
        eeg_samples = int(eeg_rows[record.join_key].get("sample_count", 0))
        if fnirs_samples <= 0 or eeg_samples <= 0:
            raise RuntimeError(f"invalid record length metadata: {record.join_key}")
        for source in record.manifest.get("source_files", ()):
            if source.get("path") and source.get("sha256"):
                source_hashes[str(source["path"])] = str(source["sha256"])

        events = sorted(index.events_by_join_key.get(record.join_key, ()), key=lambda row: int(row.get("event_index", -1)))
        reports = index.reports_by_join_key.get(record.join_key, ())
        all_event_count += len(events)
        if len(events) != 20 or sum(str(event.get("label")) == target for event in events) != int(data["trials_per_subject_session"]):
            raise RuntimeError(f"event count/label contract failed: {record.join_key}")
        event_ids = [int(event.get("event_index", -1)) for event in events]
        if event_ids != list(range(20)):
            raise RuntimeError(f"event index ordering failed: {record.join_key}")
        if len(reports) != 1:
            raise RuntimeError(f"alignment report cardinality failed: {record.join_key}")
        report = reports[0]
        if (
            report.get("alignment_case") != "stable_fixed_offset"
            or report.get("label_sequence_match") is not True
            or any(int(report.get(key, -1)) != 20 for key in ("num_eeg_events", "num_fnirs_events", "num_aligned_events"))
        ):
            raise RuntimeError(f"alignment contract failed: {record.join_key}")

        fnirs_times = [float(event["fnirs_time_ms"]) / 1000.0 for event in events]
        eeg_times = [float(event["eeg_time_ms"]) / 1000.0 for event in events]
        if any(right <= left for left, right in zip(fnirs_times, fnirs_times[1:])) or any(right <= left for left, right in zip(eeg_times, eeg_times[1:])):
            raise RuntimeError(f"non-increasing event clock: {record.join_key}")
        for position, event in enumerate(events):
            metadata = event.get("metadata", {})
            if (
                event.get("schema") != "physiology_event_alignment_v1"
                or event.get("event_type") != "trial"
                or int(metadata.get("session_idx", -1)) != session_map[record.base_record_id]
                or metadata.get("task") != "mental_arithmetic"
            ):
                raise RuntimeError(f"session/task mapping failed: {record.join_key}")
            if str(event.get("label")) != target:
                continue
            fnirs_start = int(round((fnirs_times[position] + offset_s) * 10.0))
            fnirs_stop = fnirs_start + int(round(duration_s * 10.0))
            eeg_start = int(round((eeg_times[position] + offset_s) * 200.0))
            eeg_stop = eeg_start + int(round(duration_s * 200.0))
            next_fnirs = int(round(fnirs_times[position + 1] * 10.0)) if position + 1 < len(events) else None
            next_eeg = int(round(eeg_times[position + 1] * 200.0)) if position + 1 < len(events) else None
            if fnirs_start < 0 or eeg_start < 0 or fnirs_stop > fnirs_samples or eeg_stop > eeg_samples:
                raise RuntimeError(f"record support failed: {record.join_key} event {event['event_index']}")
            if next_fnirs is not None and fnirs_stop + int(round(guard_s * 10.0)) > next_fnirs:
                raise RuntimeError(f"fNIRS next-event guard failed: {record.join_key} event {event['event_index']}")
            if next_eeg is not None and eeg_stop + int(round(guard_s * 200.0)) > next_eeg:
                raise RuntimeError(f"EEG next-event guard failed: {record.join_key} event {event['event_index']}")
            rows.append({
                "sample_id": f"{record.join_key}|event={int(event['event_index'])}|start_ms={int(round(offset_s * 1000))}",
                "dataset_id": record.dataset_id,
                "subject": record.canonical_subject_id,
                "record_id": record.base_record_id,
                "source_session_idx_zero_based": session_map[record.base_record_id],
                "event_index": int(event["event_index"]),
                "label": target,
                "window_start_s": offset_s,
                "window_end_s": offset_s + duration_s,
                "eeg_start_margin_samples": eeg_start,
                "fnirs_start_margin_samples": fnirs_start,
                "eeg_next_event_margin_samples": None if next_eeg is None else next_eeg - eeg_stop,
                "fnirs_next_event_margin_samples": None if next_fnirs is None else next_fnirs - fnirs_stop,
                "eeg_record_end_margin_samples": eeg_samples - eeg_stop,
                "fnirs_record_end_margin_samples": fnirs_samples - fnirs_stop,
                "alignment_case": report["alignment_case"],
                "label_sequence_match": report["label_sequence_match"],
            })

    if all_event_count != int(data["expected_all_event_count"]) or len(rows) != int(data["expected_unique_target_trial_count"]):
        raise RuntimeError("metadata aggregate count contract failed")
    finite_next_eeg = [int(row["eeg_next_event_margin_samples"]) for row in rows if row["eeg_next_event_margin_samples"] is not None]
    finite_next_fnirs = [int(row["fnirs_next_event_margin_samples"]) for row in rows if row["fnirs_next_event_margin_samples"] is not None]
    if not finite_next_eeg or not finite_next_fnirs:
        raise RuntimeError("next-event margin evidence is empty")
    summary = {
        "metadata_only": True,
        "selected_record_count": len(records),
        "selected_all_event_count": all_event_count,
        "selected_target_trial_count": len(rows),
        "subjects": len(subjects),
        "sessions": list(SESSION_IDS),
        "trials_per_subject_session": int(data["trials_per_subject_session"]),
        "alignment_reports_passed": len(records),
        "window_samples": {"eeg_200hz": 6000, "fnirs_10hz": 300},
        "minimum_next_event_margin_samples": {
            "eeg_200hz": min(finite_next_eeg),
            "fnirs_10hz": min(finite_next_fnirs),
        },
        "minimum_next_event_margin_s": {
            "eeg": min(finite_next_eeg) / 200.0,
            "fnirs": min(finite_next_fnirs) / 10.0,
        },
        "minimum_record_end_margin_s": {
            "eeg": min(int(row["eeg_record_end_margin_samples"]) for row in rows) / 200.0,
            "fnirs": min(int(row["fnirs_record_end_margin_samples"]) for row in rows) / 10.0,
        },
        "last_event_without_next_count": sum(row["fnirs_next_event_margin_samples"] is None for row in rows),
        "event_duration_available": False,
        "recovery_description": "15-second nominal post-task recovery envelope; exact rest end is not indexed",
        "eeg_length_sidecar_role": "metadata-only record support; measured branch remains raw_with_ocular_artifact",
    }
    input_hashes.update({f"manifest_declared_source::{path}": digest for path, digest in sorted(source_hashes.items())})
    return summary, rows, input_hashes


def _loader_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["data"]
    subjects = list(map(str, data["subjects"]))
    return {
        "data": {
            "cache_root": str(data["cache_root"]),
            "window_duration_s": float(data["window_duration_s"]),
            "window_offset_s": float(data["window_offset_s"]),
            "baseline_duration_s": float(data["baseline_duration_s"]),
            "task_duration_s": float(data["task_duration_s"]),
            "conditions": [
                {
                    "condition_id": f"single_trial_ma_{session}",
                    "dataset_id": str(data["dataset_id"]),
                    "subjects": subjects,
                    "record_id": session,
                    "target_label": str(data["target_label"]),
                    "eeg_signal_branch": str(data["eeg_signal_branch"]),
                    "max_trials_per_subject": int(data["trials_per_subject_session"]),
                }
                for session in SESSION_IDS
            ],
        }
    }


def _load_selected_trials(
    config: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, list[Trial]]], list[dict[str, Any]]]:
    """First array-access boundary: materialize only the exact 01--18 view."""

    validate_config(config)
    loader_config = _loader_config(config)
    forbidden = set(config["data"]["validation_subjects_closed"]) | set(config["data"]["protected_subjects_closed"])
    selected_subjects = {
        str(subject)
        for condition in loader_config["data"]["conditions"]
        for subject in condition["subjects"]
    }
    if selected_subjects != set(config["data"]["subjects"]) or selected_subjects & forbidden:
        raise RuntimeError("array loader scope includes a closed subject")
    grouped, contracts = _load_trials(loader_config)
    metadata_keys = {
        (str(row["subject"]), str(row["record_id"]), int(row["event_index"]))
        for row in metadata_rows
    }
    loaded_keys: set[tuple[str, str, int]] = set()
    expected_subjects = set(map(str, config["data"]["subjects"]))
    expected_trials = int(config["data"]["trials_per_subject_session"])
    for session in SESSION_IDS:
        condition_id = f"single_trial_ma_{session}"
        per_subject = grouped.get(condition_id, {})
        if set(per_subject) != expected_subjects:
            raise RuntimeError(f"unexpected loaded subject set: {condition_id}")
        for subject, trials in per_subject.items():
            if len(trials) != expected_trials:
                raise RuntimeError(f"{subject}/{session}: expected exactly {expected_trials} trials")
            for trial in trials:
                _require_full_support(trial)
                if trial.subject != subject or trial.record_id != session:
                    raise RuntimeError("loaded trial identity mismatch")
                if trial.eeg.shape[0] != 6000 or trial.fnirs.shape[0] != 300:
                    raise RuntimeError(f"{subject}/{session}: expected 6000 EEG and 300 fNIRS samples")
                loaded_keys.add((trial.subject, trial.record_id, int(trial.event_index)))
    if loaded_keys != metadata_keys or len(loaded_keys) != int(config["data"]["expected_unique_target_trial_count"]):
        raise RuntimeError("materialized trial identities differ from the metadata-approved view")
    return grouped, contracts


def _folds() -> tuple[dict[str, Any], ...]:
    folds = []
    for heldout in SESSION_IDS:
        train = tuple(session for session in SESSION_IDS if session != heldout)
        if set(train) & {heldout} or set(train) | {heldout} != set(SESSION_IDS):
            raise RuntimeError("invalid LOSO split")
        folds.append({"fold_id": f"holdout_{heldout}", "heldout_session": heldout, "train_sessions": train})
    return tuple(folds)


def _source_model_config(source: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    model_config = copy.deepcopy(dict(source))
    for key in ("window_offset_s", "window_duration_s", "baseline_duration_s", "task_duration_s"):
        model_config["data"][key] = float(config["data"][key])
    model_config["analysis"]["sampling_hz"] = float(config["analysis"]["sampling_hz"])
    model_config["analysis"]["active_hbo_channels"] = int(config["analysis"]["active_hbo_channels"])
    return model_config


def _prepare_fold(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    grouped: Mapping[str, Mapping[str, Sequence[Trial]]],
    fold: Mapping[str, Any],
) -> tuple[dict[str, list[PreparedTrial]], tuple[BalloonParameters, BalloonObservationSpec, BalloonConfig], dict[str, Any]]:
    train_sessions = tuple(map(str, fold["train_sessions"]))
    heldout_session = str(fold["heldout_session"])
    train_trials = [
        trial
        for session in train_sessions
        for subject in config["data"]["subjects"]
        for trial in grouped[f"single_trial_ma_{session}"][str(subject)]
    ]
    heldout_trials = [
        trial
        for subject in config["data"]["subjects"]
        for trial in grouped[f"single_trial_ma_{heldout_session}"][str(subject)]
    ]
    if len(train_trials) != 360 or len(heldout_trials) != 180:
        raise RuntimeError(f"{fold['fold_id']}: expected 360 training and 180 heldout trials")
    if {trial.record_id for trial in train_trials} & {trial.record_id for trial in heldout_trials}:
        raise RuntimeError(f"{fold['fold_id']}: training/heldout session overlap")

    reference = train_trials[0]
    for trial in train_trials[1:] + heldout_trials:
        if (
            trial.eeg_channel_names != reference.eeg_channel_names
            or trial.fnirs_channel_names != reference.fnirs_channel_names
            or trial.fnirs_roles != reference.fnirs_roles
        ):
            raise RuntimeError(f"{fold['fold_id']}: cross-session channel identity/order mismatch")
    hbo_indices, hbo_names, _ = _select_active_hbo(
        train_trials,
        baseline_duration_s=float(config["data"]["baseline_duration_s"]),
        task_duration_s=float(config["data"]["task_duration_s"]),
        count=int(config["analysis"]["active_hbo_channels"]),
    )
    hbr_indices = _paired_hbr_indices(reference, hbo_indices)
    eeg_indices = np.asarray([
        index for index, name in enumerate(reference.eeg_channel_names)
        if not any(token in name.upper() for token in ("EOG", "ECG", "EMG"))
    ], dtype=int)
    if not len(eeg_indices):
        raise RuntimeError("no scalp EEG channels remain")
    adapter, _ = _fit_eeg_adapter(train_trials, eeg_indices)

    prepared: dict[str, list[PreparedTrial]] = {session: [] for session in SESSION_IDS}
    for session in SESSION_IDS:
        condition_id = f"single_trial_ma_{session}"
        for subject in config["data"]["subjects"]:
            for trial in grouped[condition_id][str(subject)]:
                prepared[session].append(PreparedTrial(
                    trial=trial,
                    eeg_driver=_apply_eeg_adapter(trial, adapter),
                    hbo=np.mean(trial.fnirs[:, hbo_indices], axis=1, dtype=np.float64),
                    hbr=np.mean(trial.fnirs[:, hbr_indices], axis=1, dtype=np.float64),
                ))
    fit_series = [item for session in train_sessions for item in prepared[session]]
    model_config = _source_model_config(source, config)
    bundle, calibration = _fit_models(fit_series, model_config, fit_comparison_models=False)
    detail = {
        "fold_id": str(fold["fold_id"]),
        "train_sessions": list(train_sessions),
        "heldout_session": heldout_session,
        "fit_trial_count": len(train_trials),
        "heldout_trial_count": len(heldout_trials),
        "selected_hbo_channels": list(hbo_names),
        "selected_hbr_channels": [reference.fnirs_channel_names[int(index)] for index in hbr_indices],
        "selected_eeg_channels": list(adapter.channel_names),
        "eeg_adapter": {
            "indices": adapter.indices,
            "feature_mean": adapter.feature_mean,
            "feature_std": adapter.feature_std,
            "pca_mean": adapter.pca_mean,
            "loading": adapter.loading,
            "pc_scale": adapter.pc_scale,
            "heldout_trial_numerical_floor_note": "per-trial deterministic log-power floor uses heldout EEG input; it is not a fitted gauge",
        },
        "observation_calibration": calibration,
        "heldout_fit_calls": 0,
        "heldout_channel_selection_calls": 0,
        "heldout_calibration_calls": 0,
        "canonical_preprocessing_note": "record-level canonical preprocessing predates this fold and is not learned by the LOSO runner",
    }
    return prepared, bundle.t3a, detail


def _observations(item: PreparedTrial) -> np.ndarray:
    return np.column_stack((item.eeg_driver, item.hbo, item.hbr)).astype(np.float64)


def _physical_failure_count(checks: Mapping[str, Any]) -> int:
    return sum(isinstance(value, (bool, np.bool_)) and not bool(value) for value in checks.values())


def _fit_scalar(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    grid_points: int,
    max_iterations: int,
    xatol: float,
) -> dict[str, Any]:
    grid = np.linspace(lower, upper, int(grid_points))
    values = np.asarray([float(objective(float(value))) for value in grid], dtype=np.float64)
    finite = np.isfinite(values) & (values < 1.0e11)
    if not np.any(finite):
        raise RuntimeError("kappa grid produced no finite objective")
    best_index = int(np.nanargmin(np.where(finite, values, np.inf)))
    bracket_lower = float(grid[max(0, best_index - 1)])
    bracket_upper = float(grid[min(len(grid) - 1, best_index + 1)])
    candidates = [(float(grid[best_index]), float(values[best_index]), "grid")]
    optimizer_success = True
    optimizer_message = "grid endpoint"
    if bracket_upper > bracket_lower:
        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=(bracket_lower, bracket_upper),
            options={"maxiter": int(max_iterations), "xatol": float(xatol)},
        )
        optimizer_success = bool(result.success) and np.isfinite(result.fun) and float(result.fun) < 1.0e11
        optimizer_message = str(result.message)
        if np.isfinite(result.fun):
            candidates.append((float(result.x), float(result.fun), "bounded_refinement"))
    estimate, value, source = min(candidates, key=lambda item: item[1])
    return {
        "estimate_log_kappa": estimate,
        "objective": value,
        "selection_source": source,
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "grid_best_log_kappa": float(grid[best_index]),
        "grid_best_objective": float(values[best_index]),
        "grid_finite_count": int(np.count_nonzero(finite)),
    }


def _fit_subject_models(task: Mapping[str, Any]) -> dict[str, Any]:
    """Fit pooled and per-training-session scalar kappa coordinates."""

    subject = str(task["subject"])
    fold_id = str(task["fold_id"])
    train_sessions = tuple(map(str, task["train_sessions"]))
    if len(train_sessions) != 2:
        raise RuntimeError("session-nuisance decomposition requires exactly two training sessions")
    trials = {session: tuple(task["train_by_session"][session]) for session in train_sessions}
    base: BalloonParameters = task["base_parameters"]
    spec: BalloonObservationSpec = task["observation_spec"]
    balloon_config: BalloonConfig = task["balloon_config"]
    fit_cfg = task["fit_config"]
    lower_kappa, upper_kappa = map(float, fit_cfg["bounds"])
    lower, upper = math.log(lower_kappa), math.log(upper_kappa)
    prior_mean, prior_sd = float(fit_cfg["prior_mean"]), float(fit_cfg["prior_sd"])
    cache: dict[tuple[str, float], tuple[float, int]] = {}

    def likelihood(session: str, log_kappa: float) -> tuple[float, int]:
        key = (session, round(float(log_kappa), 12))
        if key in cache:
            return cache[key]
        try:
            parameters = _replace_parameter_values(base, {"kappa": math.exp(float(log_kappa))})
            nll = 0.0
            failures = 0
            for observations in trials[session]:
                result = smooth_balloon(
                    np.asarray(observations, dtype=np.float64),
                    parameters=parameters,
                    observation_spec=spec,
                    config=balloon_config,
                )
                nll -= float(result.predictive_log_likelihood)
                failures += _physical_failure_count(result.physical_checks)
            if not np.isfinite(nll):
                nll = 1.0e12
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            nll, failures = 1.0e12, -1
        cache[key] = (float(nll), int(failures))
        return cache[key]

    def penalty(log_kappa: float) -> float:
        return 0.5 * ((math.exp(float(log_kappa)) - prior_mean) / prior_sd) ** 2

    settings = (
        lower,
        upper,
        int(fit_cfg["grid_points"]),
        int(fit_cfg["max_iterations"]),
        float(fit_cfg["xatol"]),
    )
    with threadpool_limits(limits=1):
        shared = _fit_scalar(
            lambda value: sum(likelihood(session, value)[0] for session in train_sessions) + penalty(value),
            *settings,
        )
        by_session = {
            session: _fit_scalar(lambda value, session=session: likelihood(session, value)[0] + penalty(value), *settings)
            for session in train_sessions
        }
    session_logs = {session: float(result["estimate_log_kappa"]) for session, result in by_session.items()}
    center_log = float(np.mean(list(session_logs.values())))
    center_kappa = math.exp(center_log)
    log_span = upper - lower
    boundary_fraction = float(fit_cfg["boundary_fraction"])

    optimizer_rows = []
    for fit_role, result, sessions in [
        ("shared_subject", shared, train_sessions),
        *((f"session_effective_{session}", by_session[session], (session,)) for session in train_sessions),
    ]:
        log_estimate = float(result["estimate_log_kappa"])
        likelihood_nll = sum(likelihood(session, log_estimate)[0] for session in sessions)
        physical_failures = sum(likelihood(session, log_estimate)[1] for session in sessions)
        optimizer_rows.append({
            "fold_id": fold_id,
            "heldout_session": str(task["heldout_session"]),
            "subject": subject,
            "fit_role": fit_role,
            "fit_sessions": "+".join(sessions),
            "estimate_kappa": math.exp(log_estimate),
            "estimate_log_kappa": log_estimate,
            "objective": float(result["objective"]),
            "likelihood_nll": likelihood_nll,
            "prior_penalty": penalty(log_estimate),
            "objective_at_registered_prior": sum(likelihood(session, math.log(prior_mean))[0] for session in sessions),
            "optimizer_success": bool(result["optimizer_success"]),
            "optimizer_message": str(result["optimizer_message"]),
            "selection_source": str(result["selection_source"]),
            "grid_finite_count": int(result["grid_finite_count"]),
            "grid_points": int(fit_cfg["grid_points"]),
            "physical_boolean_failures": physical_failures,
            "boundary": min(log_estimate - lower, upper - log_estimate) <= boundary_fraction * log_span,
        })
    session_a, session_b = train_sessions
    parameter_row = {
        "fold_id": fold_id,
        "heldout_session": str(task["heldout_session"]),
        "subject": subject,
        "train_session_a": session_a,
        "train_session_b": session_b,
        "fit_trial_count": sum(len(values) for values in trials.values()),
        "heldout_trial_count": len(task["heldout_trials"]),
        "shared_kappa": math.exp(float(shared["estimate_log_kappa"])),
        "session_a_kappa": math.exp(session_logs[session_a]),
        "session_b_kappa": math.exp(session_logs[session_b]),
        "session_center_kappa": center_kappa,
        "session_center_log_kappa": center_log,
        "session_a_log_deviation": session_logs[session_a] - center_log,
        "session_b_log_deviation": session_logs[session_b] - center_log,
        "shared_boundary": optimizer_rows[0]["boundary"],
        "session_a_boundary": optimizer_rows[1]["boundary"],
        "session_b_boundary": optimizer_rows[2]["boundary"],
        "all_optimizer_success": all(bool(row["optimizer_success"]) for row in optimizer_rows),
        "training_physical_boolean_failures": sum(int(row["physical_boolean_failures"]) for row in optimizer_rows),
        "heldout_kappa_rule": "geometric_center_of_two_training_session_estimates",
        "heldout_fit_calls": 0,
    }
    return {
        "parameter_row": parameter_row,
        "optimizer_rows": optimizer_rows,
        "shared_parameters": _replace_parameter_values(base, {"kappa": parameter_row["shared_kappa"]}),
        "session_center_parameters": _replace_parameter_values(base, {"kappa": center_kappa}),
    }


def _time_masks(config: Mapping[str, Any], length: int) -> dict[str, np.ndarray]:
    fs = float(config["analysis"]["sampling_hz"])
    offset = float(config["data"]["window_offset_s"])
    time_axis = offset + np.arange(int(length), dtype=np.float64) / fs

    def interval(spec: Mapping[str, Any]) -> np.ndarray:
        start = float(spec["relative_start_s"])
        stop = float(spec["relative_stop_s"])
        return (time_axis >= start - 1e-12) & (time_axis < stop - 1e-12)

    masks = {
        "heldout_input": interval(config["analysis"]["heldout_input_mask"]),
        "recovery": interval(config["analysis"]["primary_recovery_mask"]),
        "response": interval(config["analysis"]["secondary_response_mask"]),
    }
    if np.count_nonzero(masks["heldout_input"]) != 250 or np.count_nonzero(masks["recovery"]) != 150 or np.count_nonzero(masks["response"]) != 250:
        raise RuntimeError("heldout mask sample counts have drifted")
    if np.any(masks["recovery"] & ~masks["heldout_input"]):
        raise RuntimeError("recovery score is not fully target-masked")
    return masks


def _symmetric_nrmse(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)[mask]
    right = np.asarray(right, dtype=np.float64)[mask]
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    scale = math.sqrt(max(0.5 * (float(np.var(left[valid])) + float(np.var(right[valid]))), 1e-16))
    return float(np.sqrt(np.mean(np.square(left[valid] - right[valid]))) / scale)


def _fit_and_score_subject(task: Mapping[str, Any]) -> dict[str, Any]:
    fit = _fit_subject_models(task)
    base: BalloonParameters = task["base_parameters"]
    models = {
        "M0_fixed": base,
        "M1_kappa_shared": fit["shared_parameters"],
        "M1_kappa_session_nuisance": fit["session_center_parameters"],
    }
    masks = _time_masks(task["config"], len(task["heldout_trials"][0].hbo))
    spec: BalloonObservationSpec = task["observation_spec"]
    balloon_config: BalloonConfig = task["balloon_config"]
    metric_rows: list[dict[str, Any]] = []
    driver_comparisons: list[dict[str, Any]] = []
    drivers: dict[str, list[np.ndarray]] = {model: [] for model in MODEL_IDS}
    for item in task["heldout_trials"]:
        truth = _observations(item)
        masked = truth.copy()
        masked[np.ix_(masks["heldout_input"], np.asarray([1, 2]))] = np.nan
        results = {}
        for model, parameters in models.items():
            result = smooth_balloon(
                masked,
                parameters=parameters,
                observation_spec=spec,
                config=balloon_config,
                observation_mask=np.isfinite(masked),
            )
            results[model] = result
            driver = np.asarray(result.state_mean[:, 0], dtype=np.float64)
            drivers[model].append(driver)
            predictive_std = np.sqrt(np.maximum(np.asarray(result.total_variance, dtype=np.float64), 0.0))
            failures = _physical_failure_count(result.physical_checks)
            for mask_id in MASK_IDS:
                for column, target in ((1, "HbO"), (2, "HbR")):
                    metrics = _masked_metrics(
                        truth[:, column],
                        np.asarray(result.observation_mean)[:, column],
                        masks[mask_id],
                        predictive_std[:, column],
                    )
                    score, score_n = _gaussian_negative_log_score(
                        truth[:, column],
                        np.asarray(result.observation_mean)[:, column],
                        predictive_std[:, column],
                        masks[mask_id],
                    )
                    metric_rows.append({
                        "fold_id": str(task["fold_id"]),
                        "heldout_session": str(task["heldout_session"]),
                        "train_sessions": "+".join(task["train_sessions"]),
                        "subject": str(task["subject"]),
                        "event_index": int(item.trial.event_index),
                        "model": model,
                        "mask": mask_id,
                        "target": target,
                        "gaussian_negative_log_score": score,
                        "score_n": score_n,
                        "target_observed_by_smoother": False,
                        "heldout_parameter_fit_calls": 0,
                        "physical_boolean_failures": failures,
                        **metrics,
                    })
        for model in ("M1_kappa_shared", "M1_kappa_session_nuisance"):
            left = np.asarray(results["M0_fixed"].state_mean[:, 0], dtype=np.float64)
            right = np.asarray(results[model].state_mean[:, 0], dtype=np.float64)
            driver_comparisons.append({
                "kind": "within_trial_model_contrast",
                "fold_id": str(task["fold_id"]),
                "heldout_session": str(task["heldout_session"]),
                "subject": str(task["subject"]),
                "event_index": int(item.trial.event_index),
                "left": "M0_fixed",
                "right": model,
                "mask": "response",
                "nrmse": _symmetric_nrmse(left, right, masks["response"]),
                "correlation": _safe_corr(left[masks["response"]], right[masks["response"]]),
            })
    mean_drivers = {model: np.mean(np.stack(values), axis=0) for model, values in drivers.items()}
    return {
        "parameter_row": fit["parameter_row"],
        "optimizer_rows": fit["optimizer_rows"],
        "metric_rows": metric_rows,
        "driver_comparison_rows": driver_comparisons,
        "mean_drivers": mean_drivers,
    }


def _synthetic_preflight(config: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise the scalar decomposition before any measured-array access."""

    fixed_cfg = source["ssm"]["t3a"]["fixed"]
    fixed = BalloonFixedParameters(
        alpha=float(fixed_cfg["alpha"]),
        E0=float(fixed_cfg["e0"]),
        gamma=float(fixed_cfg["gamma"]),
        P0=1.0,
        Q0=0.35,
        driver_decay_per_s=float(fixed_cfg["driver_decay_per_s"]),
        process_std=tuple(float(value) for value in fixed_cfg["process_std"]),
        observation_scale=(0.05, 0.015, 0.010),
        student_nu=float(fixed_cfg["student_t_df"]),
        eeg_loading=1.0,
        eeg_offset=0.0,
        neurovascular_gain=float(fixed_cfg["neurovascular_gain"]),
    )
    base = BalloonParameters(
        fixed=fixed,
        free=BalloonFreeParameters(kappa=0.64, tau=float(fixed_cfg["tau_s"])),
    )
    spec = BalloonObservationSpec(
        eeg_loading=fixed.eeg_loading,
        eeg_offset=fixed.eeg_offset,
        observation_scale=fixed.observation_scale,
        student_nu=fixed.student_nu,
    )
    model_cfg = source["ssm"]["t3a"]
    balloon_config = BalloonConfig(
        dt=0.1,
        rk4_substeps=int(model_cfg["rk4_substeps"]),
        irls_iterations=int(model_cfg["irls_iterations"]),
        irls_weight_floor=float(model_cfg["irls_weight_floor"]),
        initial_state_std=tuple(float(value) for value in model_cfg["initial_state_std"]),
    )
    preflight = config["analysis"]["synthetic_preflight"]
    center = float(preflight["true_center_kappa"])
    deviation = float(preflight["training_log_deviation"])
    truth_kappa = {
        "session_01": center * math.exp(deviation),
        "session_03": center * math.exp(-deviation),
        "session_05": center,
    }
    time_axis = float(config["data"]["window_offset_s"]) + np.arange(300) / 10.0
    train_by_session: dict[str, list[np.ndarray]] = {session: [] for session in SESSION_IDS[:2]}
    heldout: list[np.ndarray] = []
    seed = int(config["experiment"]["seed"])
    for session_index, session in enumerate(SESSION_IDS):
        parameters = _replace_parameter_values(base, {"kappa": truth_kappa[session]})
        for trial_index in range(int(preflight["trials_per_session"])):
            task_mask = (time_axis >= 0.0) & (time_axis < 10.0)
            driver = np.zeros_like(time_axis)
            driver[task_mask] = (0.11 + 0.01 * trial_index) * (
                1.0 + 0.12 * np.sin(2.0 * math.pi * (time_axis[task_mask] + session_index) / 5.0)
            )
            simulation = simulate_balloon(
                driver,
                parameters=parameters,
                observation_spec=spec,
                config=balloon_config,
                rng=np.random.default_rng(seed + 100 * session_index + trial_index),
                add_noise=True,
            )
            observations = np.asarray(simulation.observations, dtype=np.float64)
            if session in train_by_session:
                train_by_session[session].append(observations)
            else:
                heldout.append(observations)
    fit_cfg = config["analysis"]["optimizer"]
    coordinate = config["analysis"]["kappa_coordinate"]
    fit = _fit_subject_models({
        "subject": "synthetic_subject_01",
        "fold_id": "synthetic_holdout_session_05",
        "heldout_session": "session_05",
        "train_sessions": SESSION_IDS[:2],
        "train_by_session": train_by_session,
        "heldout_trials": heldout,
        "base_parameters": base,
        "observation_spec": spec,
        "balloon_config": balloon_config,
        "fit_config": {
            "bounds": coordinate["bounds"],
            "prior_mean": coordinate["prior_mean"],
            "prior_sd": coordinate["prior_sd"],
            "grid_points": fit_cfg["transformed_grid_points"],
            "max_iterations": fit_cfg["max_iterations"],
            "xatol": fit_cfg["xatol"],
            "boundary_fraction": fit_cfg["boundary_fraction_of_log_span"],
        },
    })
    masks = _time_masks(config, 300)
    scores: dict[str, float] = {}
    physical_failures = 0
    for model, parameters in {
        "M0_fixed": base,
        "M1_kappa_session_nuisance": fit["session_center_parameters"],
    }.items():
        values = []
        for truth in heldout:
            masked = truth.copy()
            masked[np.ix_(masks["heldout_input"], np.asarray([1, 2]))] = np.nan
            result = smooth_balloon(masked, parameters=parameters, observation_spec=spec, config=balloon_config, observation_mask=np.isfinite(masked))
            physical_failures += _physical_failure_count(result.physical_checks)
            predictive_std = np.sqrt(np.maximum(result.total_variance, 0.0))
            for column in (1, 2):
                score, n = _gaussian_negative_log_score(
                    truth[:, column], result.observation_mean[:, column], predictive_std[:, column], masks["recovery"]
                )
                if n != 150:
                    raise RuntimeError("synthetic recovery support mismatch")
                values.append(score)
        scores[model] = float(np.mean(values))
    objective_checks = [
        float(row["objective"]) <= float(row["objective_at_registered_prior"]) + 1e-8
        for row in fit["optimizer_rows"]
    ]
    finite = bool(
        all(np.isfinite(value) for value in scores.values())
        and all(np.isfinite(float(row["estimate_kappa"])) for row in fit["optimizer_rows"])
    )
    passed = bool(finite and all(objective_checks) and physical_failures == 0)
    return {
        "status": "pass" if passed else "fail",
        "purpose": "deterministic software preflight only; not simulation-based calibration or a practical margin",
        "independent_trial_resets": True,
        "truth_not_passed_to_fitter": True,
        "truth_kappa": truth_kappa,
        "estimated_shared_kappa": fit["parameter_row"]["shared_kappa"],
        "estimated_session_center_kappa": fit["parameter_row"]["session_center_kappa"],
        "estimated_training_session_kappa": {
            fit["parameter_row"]["train_session_a"]: fit["parameter_row"]["session_a_kappa"],
            fit["parameter_row"]["train_session_b"]: fit["parameter_row"]["session_b_kappa"],
        },
        "recovery_gaussian_nll": scores,
        "objective_not_worse_than_prior_by_fit": objective_checks,
        "optimizer_rows": fit["optimizer_rows"],
        "physical_boolean_failures": physical_failures,
        "passed": passed,
    }


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int, replicates: int, confidence: float) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
    rng = np.random.default_rng(int(seed))
    sampled = array[rng.integers(0, len(array), size=(int(replicates), len(array)))].mean(axis=1)
    alpha = 1.0 - float(confidence)
    return {
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(sampled, alpha / 2.0)),
        "upper": float(np.quantile(sampled, 1.0 - alpha / 2.0)),
    }


def _subject_fold_metrics(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["fold_id"]), str(row["heldout_session"]), str(row["subject"]), str(row["model"]), str(row["mask"]))].append(row)
    output = []
    for (fold_id, heldout, subject, model, mask_id), rows in sorted(grouped.items()):
        expected_rows = 20
        expected_score_n = 150 if mask_id == "recovery" else 250
        scores = np.asarray([float(row["gaussian_negative_log_score"]) for row in rows], dtype=np.float64)
        valid = bool(
            len(rows) == expected_rows
            and len({int(row["event_index"]) for row in rows}) == 10
            and {str(row["target"]) for row in rows} == set(TARGETS)
            and all(int(row["score_n"]) == expected_score_n for row in rows)
            and np.all(np.isfinite(scores))
        )
        output.append({
            "fold_id": fold_id,
            "heldout_session": heldout,
            "subject": subject,
            "model": model,
            "mask": mask_id,
            "trial_count": len({int(row["event_index"]) for row in rows}),
            "target_trial_row_count": len(rows),
            "score_point_count": sum(int(row["score_n"]) for row in rows),
            "mean_gaussian_negative_log_score": float(np.mean(scores)) if valid else float("nan"),
            "mean_nrmse": float(np.mean([float(row["nrmse"]) for row in rows])) if valid else float("nan"),
            "physical_boolean_failures": max(int(row["physical_boolean_failures"]) for row in rows),
            "support_valid": valid,
        })
    return output


def _contrast_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    mask_id: str,
    candidate: str,
    reference: str,
    fold_id: str | None = None,
) -> dict[str, float]:
    lookup = {
        (str(row["fold_id"]), str(row["subject"]), str(row["model"])): float(row["mean_gaussian_negative_log_score"])
        for row in rows
        if row["mask"] == mask_id and (fold_id is None or row["fold_id"] == fold_id)
    }
    folds = (fold_id,) if fold_id is not None else tuple(item["fold_id"] for item in _folds())
    output: dict[str, float] = {}
    for subject in _subject_range(1, 18):
        values = [lookup[(str(current), subject, candidate)] - lookup[(str(current), subject, reference)] for current in folds]
        output[subject] = float(np.mean(values))
    return output


def _fold_summary(
    subject_fold_rows: Sequence[Mapping[str, Any]],
    parameter_rows: Sequence[Mapping[str, Any]],
    fold: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = config["analysis"]["bootstrap"]
    fold_id = str(fold["fold_id"])
    primary = _contrast_values(
        subject_fold_rows,
        mask_id="recovery",
        candidate="M1_kappa_session_nuisance",
        reference="M0_fixed",
        fold_id=fold_id,
    )
    shared = _contrast_values(
        subject_fold_rows,
        mask_id="recovery",
        candidate="M1_kappa_shared",
        reference="M0_fixed",
        fold_id=fold_id,
    )
    nuisance_vs_shared = _contrast_values(
        subject_fold_rows,
        mask_id="recovery",
        candidate="M1_kappa_session_nuisance",
        reference="M1_kappa_shared",
        fold_id=fold_id,
    )
    current_parameters = [row for row in parameter_rows if row["fold_id"] == fold_id]
    seed = int(bootstrap["seed"]) + SESSION_IDS.index(str(fold["heldout_session"]))
    return {
        "fold_id": fold_id,
        "train_sessions": list(fold["train_sessions"]),
        "heldout_session": str(fold["heldout_session"]),
        "fit_subject_count": 18,
        "fit_trial_count": 360,
        "heldout_subject_count": 18,
        "heldout_trial_count": 180,
        "primary_delta_nll_candidate_minus_M0": {
            **_bootstrap_mean_ci(primary.values(), seed=seed, replicates=int(bootstrap["replicates"]), confidence=float(bootstrap["confidence"])),
            "median": float(np.median(list(primary.values()))),
        },
        "secondary_delta_nll_shared_minus_M0": {
            **_bootstrap_mean_ci(shared.values(), seed=seed + 10, replicates=int(bootstrap["replicates"]), confidence=float(bootstrap["confidence"])),
            "median": float(np.median(list(shared.values()))),
        },
        "secondary_delta_nll_nuisance_minus_shared": {
            **_bootstrap_mean_ci(nuisance_vs_shared.values(), seed=seed + 20, replicates=int(bootstrap["replicates"]), confidence=float(bootstrap["confidence"])),
            "median": float(np.median(list(nuisance_vs_shared.values()))),
        },
        "optimizer_failure_count": sum(not bool(row["all_optimizer_success"]) for row in current_parameters),
        "training_physical_boolean_failures_per_fit_role_total": sum(int(row["training_physical_boolean_failures"]) for row in current_parameters),
        "any_session_boundary_subject_count": sum(bool(row["session_a_boundary"] or row["session_b_boundary"]) for row in current_parameters),
    }


def _driver_session_rows(
    mean_driver_records: Mapping[tuple[str, str, str], np.ndarray],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    mask = _time_masks(config, 300)["response"]
    rows = []
    for subject in config["data"]["subjects"]:
        for left_session, right_session in combinations(SESSION_IDS, 2):
            left = mean_driver_records[(str(subject), left_session, "M1_kappa_session_nuisance")]
            right = mean_driver_records[(str(subject), right_session, "M1_kappa_session_nuisance")]
            rows.append({
                "kind": "cross_session_mean_driver",
                "fold_id": "across_all_folds",
                "heldout_session": "pair",
                "subject": str(subject),
                "event_index": "mean_of_10_trials",
                "left": left_session,
                "right": right_session,
                "mask": "response",
                "nrmse": _symmetric_nrmse(left, right, mask),
                "correlation": _safe_corr(left[mask], right[mask]),
            })
    return rows


def _trial_inventory(metadata_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for fold in _folds():
        heldout = str(fold["heldout_session"])
        for item in metadata_rows:
            role = "heldout" if item["record_id"] == heldout else "fit"
            rows.append({
                "fold_id": fold["fold_id"],
                "train_sessions": "+".join(fold["train_sessions"]),
                "heldout_session": heldout,
                "role": role,
                "sample_id": item["sample_id"],
                "subject": item["subject"],
                "record_id": item["record_id"],
                "event_index": item["event_index"],
                "used_for_gauge_channel_or_calibration": role == "fit",
                "used_for_parameter_fit": role == "fit",
                "used_for_heldout_score": role == "heldout",
                "heldout_fit_calls": 0 if role == "heldout" else "not_heldout",
            })
    return rows


def _fold_summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    primary = summary["primary_delta_nll_candidate_minus_M0"]
    shared = summary["secondary_delta_nll_shared_minus_M0"]
    nuisance = summary["secondary_delta_nll_nuisance_minus_shared"]
    return {
        "fold_id": summary["fold_id"],
        "train_sessions": "+".join(summary["train_sessions"]),
        "heldout_session": summary["heldout_session"],
        "fit_subject_count": summary["fit_subject_count"],
        "fit_trial_count": summary["fit_trial_count"],
        "heldout_subject_count": summary["heldout_subject_count"],
        "heldout_trial_count": summary["heldout_trial_count"],
        "primary_delta_nll_mean": primary["mean"],
        "primary_delta_nll_median": primary["median"],
        "primary_delta_nll_ci_lower": primary["lower"],
        "primary_delta_nll_ci_upper": primary["upper"],
        "shared_delta_nll_mean": shared["mean"],
        "shared_delta_nll_median": shared["median"],
        "shared_delta_nll_ci_lower": shared["lower"],
        "shared_delta_nll_ci_upper": shared["upper"],
        "nuisance_minus_shared_delta_nll_mean": nuisance["mean"],
        "nuisance_minus_shared_delta_nll_median": nuisance["median"],
        "nuisance_minus_shared_delta_nll_ci_lower": nuisance["lower"],
        "nuisance_minus_shared_delta_nll_ci_upper": nuisance["upper"],
        "optimizer_failure_count": summary["optimizer_failure_count"],
        "training_physical_boolean_failures_per_fit_role_total": summary["training_physical_boolean_failures_per_fit_role_total"],
        "any_session_boundary_subject_count": summary["any_session_boundary_subject_count"],
    }


def _write_tables(run_dir: Path, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    paths = {
        "metadata": "metadata_boundary.csv",
        "inventory": "trial_inventory.csv",
        "parameters": "parameter_estimates.csv",
        "optimizers": "optimizer_diagnostics.csv",
        "metrics": "heldout_metrics.csv",
        "subject_fold": "subject_fold_metrics.csv",
        "drivers": "driver_stability.csv",
        "trajectories": "driver_mean_trajectories.csv",
        "fold_summary": "fold_summary.csv",
        "subject_summary": "subject_summary.csv",
    }
    sort_keys = {
        "metadata": ("subject", "record_id", "event_index"),
        "inventory": ("fold_id", "subject", "record_id", "event_index"),
        "parameters": ("fold_id", "subject"),
        "optimizers": ("fold_id", "subject", "fit_role"),
        "metrics": ("fold_id", "subject", "event_index", "model", "mask", "target"),
        "subject_fold": ("fold_id", "subject", "model", "mask"),
        "drivers": ("kind", "subject", "fold_id", "event_index", "right"),
        "trajectories": ("fold_id", "subject", "model", "time_index"),
        "fold_summary": ("fold_id",),
        "subject_summary": ("subject",),
    }
    for key, filename in paths.items():
        values = list(rows.get(key, ()))
        fields = sort_keys[key]
        values.sort(key=lambda row, fields=fields: tuple(str(row.get(field, "")) for field in fields))
        _atomic_csv(run_dir / filename, values)


def _final_summary(
    config: Mapping[str, Any],
    metadata_summary: Mapping[str, Any],
    preflight: Mapping[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    fold_summaries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subject_fold = _subject_fold_metrics(rows["metrics"])
    rows["subject_fold"] = subject_fold
    primary_by_subject = _contrast_values(
        subject_fold,
        mask_id="recovery",
        candidate="M1_kappa_session_nuisance",
        reference="M0_fixed",
    )
    shared_by_subject = _contrast_values(
        subject_fold,
        mask_id="recovery",
        candidate="M1_kappa_shared",
        reference="M0_fixed",
    )
    nuisance_by_subject = _contrast_values(
        subject_fold,
        mask_id="recovery",
        candidate="M1_kappa_session_nuisance",
        reference="M1_kappa_shared",
    )
    bootstrap = config["analysis"]["bootstrap"]
    primary_ci = _bootstrap_mean_ci(
        primary_by_subject.values(),
        seed=int(bootstrap["seed"]),
        replicates=int(bootstrap["replicates"]),
        confidence=float(bootstrap["confidence"]),
    )
    shared_ci = _bootstrap_mean_ci(
        shared_by_subject.values(),
        seed=int(bootstrap["seed"]) + 100,
        replicates=int(bootstrap["replicates"]),
        confidence=float(bootstrap["confidence"]),
    )
    nuisance_ci = _bootstrap_mean_ci(
        nuisance_by_subject.values(),
        seed=int(bootstrap["seed"]) + 200,
        replicates=int(bootstrap["replicates"]),
        confidence=float(bootstrap["confidence"]),
    )

    parameter_by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows["parameters"]:
        parameter_by_subject[str(row["subject"])].append(row)
    driver_cross = [row for row in rows["drivers"] if row["kind"] == "cross_session_mean_driver"]
    driver_within = [
        row for row in rows["drivers"]
        if row["kind"] == "within_trial_model_contrast" and row["right"] == "M1_kappa_session_nuisance"
    ]
    cross_by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    within_by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in driver_cross:
        cross_by_subject[str(row["subject"])].append(row)
    for row in driver_within:
        within_by_subject[str(row["subject"])].append(row)
    screens = config["analysis"]["exploratory_screens"]
    log_span = math.log(1.5) - math.log(0.2)
    subject_rows: list[dict[str, Any]] = []
    for subject in config["data"]["subjects"]:
        parameter_values = parameter_by_subject[str(subject)]
        if len(parameter_values) != 3:
            raise RuntimeError(f"{subject}: expected three LOSO parameter centers")
        logs = np.asarray([float(row["session_center_log_kappa"]) for row in parameter_values])
        normalized_range = float(np.ptp(logs) / log_span)
        cross_values = cross_by_subject[str(subject)]
        within_values = within_by_subject[str(subject)]
        if len(cross_values) != 3 or len(within_values) != 30:
            raise RuntimeError(f"{subject}: expected three cross-session driver pairs")
        median_driver_nrmse = float(np.median([float(row["nrmse"]) for row in within_values]))
        median_driver_corr = float(np.median([float(row["correlation"]) for row in within_values]))
        subject_rows.append({
            "subject": str(subject),
            "primary_delta_nll_candidate_minus_M0": primary_by_subject[str(subject)],
            "secondary_delta_nll_shared_minus_M0": shared_by_subject[str(subject)],
            "secondary_delta_nll_nuisance_minus_shared": nuisance_by_subject[str(subject)],
            "center_kappa_geometric_mean_across_folds": float(np.exp(np.mean(logs))),
            "center_log_kappa_range_fraction": normalized_range,
            "parameter_stable_operational_screen": normalized_range <= float(screens["center_log_range_fraction_max"]),
            "any_effective_kappa_boundary": any(bool(row["session_a_boundary"] or row["session_b_boundary"] or row["shared_boundary"]) for row in parameter_values),
            "all_optimizer_success": all(bool(row["all_optimizer_success"]) for row in parameter_values),
            "median_within_fold_candidate_vs_M0_driver_nrmse": median_driver_nrmse,
            "median_within_fold_candidate_vs_M0_driver_correlation": median_driver_corr,
            "driver_stable_operational_screen": (
                median_driver_nrmse <= float(screens["driver_nrmse_max"])
                and median_driver_corr >= float(screens["driver_correlation_min"])
            ),
            "median_cross_session_fold_gauge_dependent_driver_nrmse": float(np.median([float(row["nrmse"]) for row in cross_values])),
            "median_cross_session_fold_gauge_dependent_driver_correlation": float(np.median([float(row["correlation"]) for row in cross_values])),
        })

    rows["subject_summary"] = subject_rows

    expected_counts = {
        "metadata": 540,
        "inventory": 1620,
        "parameters": 54,
        "optimizers": 162,
        "metrics": 6480,
        "subject_fold": 324,
        "drivers": 1134,
        "trajectories": 48600,
        "fold_summary": 3,
        "subject_summary": 18,
    }
    actual_counts = {key: len(rows[key]) for key in expected_counts}
    if actual_counts != expected_counts:
        raise RuntimeError(f"artifact row-count contract failed: expected={expected_counts}, actual={actual_counts}")
    required_score_rows = [
        row for row in rows["metrics"]
        if row["mask"] == "recovery" and row["model"] in {"M0_fixed", "M1_kappa_session_nuisance"}
    ]
    support_valid = bool(
        len(required_score_rows) == 2160
        and all(int(row["score_n"]) == 150 for row in required_score_rows)
        and all(np.isfinite(float(row["gaussian_negative_log_score"])) for row in required_score_rows)
        and all(int(row["physical_boolean_failures"]) == 0 for row in required_score_rows)
        and all(bool(row["all_optimizer_success"]) for row in rows["parameters"])
        and all(int(row["training_physical_boolean_failures"]) == 0 for row in rows["parameters"])
    )
    fold_medians = [float(summary["primary_delta_nll_candidate_minus_M0"]["median"]) for summary in fold_summaries]
    if not support_valid:
        predictive_status = "invalid"
    elif primary_ci["upper"] < 0.0 and all(value < 0.0 for value in fold_medians):
        predictive_status = "exploratory_directional_support"
    elif primary_ci["lower"] > 0.0:
        predictive_status = "exploratory_directional_failure"
    else:
        predictive_status = "inconclusive"

    stable_fraction = float(np.mean([bool(row["parameter_stable_operational_screen"]) for row in subject_rows]))
    boundary_fraction = float(np.mean([bool(row["any_effective_kappa_boundary"]) for row in subject_rows]))
    parameter_screen = bool(
        stable_fraction >= float(screens["minimum_stable_subject_fraction"])
        and boundary_fraction <= float(screens["maximum_boundary_fit_fraction"])
        and all(bool(row["all_optimizer_success"]) for row in subject_rows)
    )
    driver_stable_fraction = float(np.mean([bool(row["driver_stable_operational_screen"]) for row in subject_rows]))
    driver_screen = driver_stable_fraction >= float(screens["minimum_stable_subject_fraction"])
    if predictive_status == "exploratory_directional_support" and parameter_screen and driver_screen:
        verdict = "exploratory_support_for_stable_effective_adaptation_coordinate_not_a_biological_trait"
    elif predictive_status == "exploratory_directional_support" and not parameter_screen and driver_screen:
        verdict = "predictive_support_but_parameter_unstable_state_only_candidate"
    elif predictive_status == "exploratory_directional_support" and not driver_screen:
        verdict = "predictive_support_but_cross_session_latent_state_unstable"
    elif predictive_status == "invalid":
        verdict = "invalid_step3_run"
    else:
        verdict = "no_conclusive_cross_session_support_for_subject_trait"

    summary = {
        "schema": SCHEMA,
        "analysis_kind": "three_session_leave_one_session_out",
        "manifest_status_owner": "manifest.json",
        "scientific_verdict": verdict,
        "scope": config["experiment"]["scope"],
        "qualification_eligible": False,
        "decision_eligibility": False,
        "claim_boundary": "exploratory fit-only target-masked LOSO; no physical parameter, trait, teacher qualification, validation/protected access, or tokenizer promotion claim",
        "metadata_boundary": metadata_summary,
        "synthetic_preflight": preflight,
        "data_counts": {
            "subjects": 18,
            "sessions": 3,
            "unique_trials": 540,
            "folds": 3,
            "training_trial_uses": 1080,
            "heldout_trial_scores": 540,
            "validation_subject_arrays_loaded": 0,
            "protected_subject_arrays_loaded": 0,
        },
        "window": {
            "relative_start_s": -5.0,
            "relative_stop_s": 25.0,
            "task_s": [0.0, 10.0],
            "nominal_recovery_s": [10.0, 25.0],
            "event_duration_indexed": False,
        },
        "estimand": {
            "primary": "heldout-session target-masked recovery HbO/HbR variance-matched Gaussian NLL",
            "target_input_mask_s": [0.0, 25.0],
            "score_mask_s": [10.0, 25.0],
            "candidate": "M1_kappa_session_nuisance",
            "reference": "M0_fixed",
            "delta_direction": "candidate_minus_M0; negative favors candidate",
            "operator": "fixed-interval smoother with heldout HbO/HbR target absent and heldout EEG plus baseline fNIRS present",
            "causal_forecast": False,
            "proper_score_approximation": "Gaussian with total predictive variance; not the exact Student-t-plus-state convolution",
            "practical_margin": None,
        },
        "primary_result": {
            "status": predictive_status,
            "delta_nll_candidate_minus_M0_subject_equal": {
                **primary_ci,
                "median": float(np.median(list(primary_by_subject.values()))),
            },
            "fold_medians": fold_medians,
            "all_three_fold_medians_negative": all(value < 0.0 for value in fold_medians),
            "support_valid": support_valid,
            "bootstrap": dict(bootstrap),
        },
        "secondary_predictive_results": {
            "shared_minus_M0": {**shared_ci, "median": float(np.median(list(shared_by_subject.values())))},
            "nuisance_minus_shared": {**nuisance_ci, "median": float(np.median(list(nuisance_by_subject.values())))},
        },
        "parameter_stability": {
            "screen_passed": parameter_screen,
            "stable_subject_fraction": stable_fraction,
            "any_boundary_subject_fraction": boundary_fraction,
            "thresholds": {
                "center_log_range_fraction_max": screens["center_log_range_fraction_max"],
                "minimum_stable_subject_fraction": screens["minimum_stable_subject_fraction"],
                "maximum_boundary_fit_fraction": screens["maximum_boundary_fit_fraction"],
            },
        },
        "driver_stability": {
            "screen_passed": driver_screen,
            "stable_subject_fraction": driver_stable_fraction,
            "screen_operator": "within the same heldout trial and fold gauge: session-nuisance candidate versus M0",
            "thresholds": {
                "nrmse_max": screens["driver_nrmse_max"],
                "correlation_min": screens["driver_correlation_min"],
            },
            "cross_session_descriptive": {
                "median_nrmse": float(np.median([float(row["nrmse"]) for row in driver_cross])),
                "median_correlation": float(np.median([float(row["correlation"]) for row in driver_cross])),
                "gauge_invariant": False,
                "interpretation": "descriptive only because each heldout session was transformed under a different train-fold PCA/observation gauge",
            },
        },
        "session_nuisance": {
            "coordinate": "zero-sum deviation in log effective kappa across the two training sessions",
            "heldout_value": 0.0,
            "source_interpretation": "generic session residual only",
            "ECG_or_respiration_available_to_runner": False,
            "causal_attribution": False,
        },
        "folds": list(fold_summaries),
        "artifact_row_counts": actual_counts,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyyaml": yaml.__version__,
        },
    }
    return summary, subject_rows


def _markdown_report(summary: Mapping[str, Any], subject_rows: Sequence[Mapping[str, Any]]) -> str:
    primary = summary["primary_result"]
    contrast = primary["delta_nll_candidate_minus_M0_subject_equal"]
    lines = [
        "# T3 实验第三步：三 session LOSO 与名义恢复期详细报告",
        "",
        f"_运行状态由 `manifest.json` 唯一负责；科学判定：`{summary['scientific_verdict']}`。_",
        "",
        "## 结论",
        "",
        (
            "本轮是 fit-only exploratory session-LOSO，不是生理参数恢复或 teacher 资格实验。"
            f"主比较（session-nuisance κ 中心 − 固定 M0）的被试等权恢复段 ΔNLL 均值为 {contrast['mean']:.6g}，"
            f"95% subject-block bootstrap CI [{contrast['lower']:.6g}, {contrast['upper']:.6g}]，"
            f"方向性判定为 `{primary['status']}`。负值有利于候选模型。"
        ),
        "",
        "即使方向性 CI 完全低于零，本合同也没有 synthetic-frozen practical margin，故最多称 exploratory predictive support；"
        "不得称 κ 为真实血管衰减率、稳定健康 trait，亦不得据此开放 validation/protected、认定 physical teacher 或推动 tokenizer promotion。",
        "",
        "## 1. 数据边界与恢复窗口",
        "",
        "- 数据仅为 `subject_01–18`、MA、cache `session_01/03/05`；它们对应源 metadata 的 zero-based `session_idx=1/3/5`。",
        "- 共有 18 名被试、3 个 session、540 个唯一 trial；每折 360 个训练 trial、180 个 held-out trial，三折累计 1080 次训练使用和 540 次 held-out 评分。",
        "- 窗口为事件相对 `[-5,+25] s`（300 个 10 Hz 点）。任务名义区间 `[0,+10) s`，主评分恢复区间 `[+10,+25) s`（150 点）。",
        "- 事件索引没有 trial/rest duration，因此这里是 15 秒“名义恢复包络”，不是已标注的精确休息终点。",
        f"- metadata-only 检查的下一事件最小余量：EEG {summary['metadata_boundary']['minimum_next_event_margin_s']['eeg']:.3f} s，fNIRS {summary['metadata_boundary']['minimum_next_event_margin_s']['fnirs']:.3f} s；54/54 alignment reports 通过。",
        "- 19–23 validation 和 24–29 protected 的数组读取均为 0；共享索引会枚举全局 metadata，但只对批准的 01–18 window 调用数组访问。",
        "",
        "## 2. 冻结模型与防泄漏操作",
        "",
        "- `M0_fixed`：κ 固定为 0.64。",
        "- `M1_kappa_shared`：每名被试用两个训练 session 的 20 个 trial 共同拟合一个 κ。",
        "- `M1_kappa_session_nuisance`：分别在两个训练 session 拟合有效 κ；log-κ 的均值是 subject center，两个偏差严格零和；held-out session 固定使用两者几何中心，held-out nuisance 固定为 0。",
        "- β、τ、γ、α、E0 全部固定；第二步失败的联合 β/κ/τ 不在本轮恢复。κ 仅是一维 effective adaptation coordinate。",
        "- 每折的 HbO/HbR 通道、EEG PCA gauge、P0/Q0 和观测噪声尺度仅由两个训练 session 决定。held-out HbO/HbR 从任务开始至窗口末端全部遮挡；仅 baseline fNIRS 和完整 EEG 进入 smoother。",
        "- 算子为 fixed-interval target-masked reconstruction，会使用 held-out session 的未来 EEG；它不是 causal forecast。主分数是以 `total_variance` 匹配的 Gaussian NLL，也不是精确 Student-t 与 latent uncertainty 卷积。",
        "",
        "## 3. 合成软件前置检查",
        "",
        f"前置检查状态为 `{summary['synthetic_preflight']['status']}`。它使用独立 trial reset，真值不传入拟合器；"
        "只检查 scalar decomposition、遮挡评分、有限性和优化目标不劣于注册 prior 起点，不是 simulation-based calibration，也不产生 measured practical margin。",
        "",
        "## 4. 三个 LOSO fold",
        "",
        "| Held-out | 训练 sessions | 主 ΔNLL mean | 95% CI | 中位数 | optimizer fail | session-boundary subjects |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for fold in summary["folds"]:
        value = fold["primary_delta_nll_candidate_minus_M0"]
        lines.append(
            f"| {fold['heldout_session']} | {' + '.join(fold['train_sessions'])} | {value['mean']:.6g} | "
            f"[{value['lower']:.6g}, {value['upper']:.6g}] | {value['median']:.6g} | "
            f"{fold['optimizer_failure_count']} | {fold['any_session_boundary_subject_count']}/18 |"
        )
    secondary = summary["secondary_predictive_results"]
    lines.extend([
        "",
        "三折均对同一 18 名被试执行，因此 bootstrap 的独立单位是 subject，不是 54 个 subject×fold。"
        f"次要比较 shared−M0 的均值为 {secondary['shared_minus_M0']['mean']:.6g} "
        f"（CI [{secondary['shared_minus_M0']['lower']:.6g}, {secondary['shared_minus_M0']['upper']:.6g}]）；"
        f"nuisance−shared 的均值为 {secondary['nuisance_minus_shared']['mean']:.6g} "
        f"（CI [{secondary['nuisance_minus_shared']['lower']:.6g}, {secondary['nuisance_minus_shared']['upper']:.6g}]）。",
        "",
        "## 5. 参数与 r(t) 稳定性",
        "",
        f"预注册参数稳定性 operational screen：`{summary['parameter_stability']['screen_passed']}`；"
        f"稳定被试比例 {summary['parameter_stability']['stable_subject_fraction']:.1%}，任一有效 κ 命中边界的被试比例 {summary['parameter_stability']['any_boundary_subject_fraction']:.1%}。",
        f"同一 fold/同一 trial 内候选模型相对 M0 的 r(t) operational screen：`{summary['driver_stability']['screen_passed']}`；"
        f"稳定被试比例 {summary['driver_stability']['stable_subject_fraction']:.1%}。阈值 NRMSE≤0.10、correlation≥0.95 只是沿用第二步的描述性 screen。",
        f"另行计算的跨 session 平均 r(t) 中位 NRMSE={summary['driver_stability']['cross_session_descriptive']['median_nrmse']:.4f}、"
        f"correlation={summary['driver_stability']['cross_session_descriptive']['median_correlation']:.4f}；由于每个 session 属于不同训练 fold gauge，该比较不具 gauge invariance，只作描述。",
        "",
        "| 被试 | 主 ΔNLL | κ center 几何均值 | log-range/span | 参数 screen | 同 fold Δr NRMSE 中位 | 同 fold Δr corr 中位 | driver screen |",
        "| --- | ---: | ---: | ---: | :---: | ---: | ---: | :---: |",
    ])
    for row in subject_rows:
        lines.append(
            f"| {row['subject']} | {float(row['primary_delta_nll_candidate_minus_M0']):.6g} | "
            f"{float(row['center_kappa_geometric_mean_across_folds']):.6g} | "
            f"{float(row['center_log_kappa_range_fraction']):.4f} | "
            f"{'pass' if row['parameter_stable_operational_screen'] else 'fail'} | "
            f"{float(row['median_within_fold_candidate_vs_M0_driver_nrmse']):.4f} | "
            f"{float(row['median_within_fold_candidate_vs_M0_driver_correlation']):.4f} | "
            f"{'pass' if row['driver_stable_operational_screen'] else 'fail'} |"
        )
    lines.extend([
        "",
        "## 6. Session nuisance 与辅助生理边界",
        "",
        "本轮的 nuisance 只是两个训练 session 在 log effective-κ 坐标中的零和偏差。当前 canonical API 不暴露 ECG 或呼吸；"
        "本地 EEG 源/缓存仅确认 30 个 scalp 通道及用于清理的 VEOG/HEOG。因此任何 session drift 的来源均保持不确定，"
        "不能归因于紧张、疲劳、心率或呼吸。ECG/呼吸与低秩全阵列 nuisance 属于计划第五步，不在本轮伪造。",
        "",
        "## 7. 判定边界",
        "",
        f"最终科学判定为 `{summary['scientific_verdict']}`。参数 screen、driver screen 和 NLL 方向证据必须联合阅读；"
        "无显著差异不是等价，稳定性可能来自先验/尺度，session 顺序也与疲劳或漂移混杂。三 session 只能提供初步重复测量证据。",
        "",
        "所有数字可由 `heldout_metrics.csv` → `subject_fold_metrics.csv` → `subject_summary.csv` 逐层重算；"
        "split、输入 hash、fold progress、文件行数和 SHA-256 的唯一 owner 是 `manifest.json`。",
        "",
    ])
    return "\n".join(lines)


def _artifact_audit(run_dir: Path, row_counts: Mapping[str, int]) -> dict[str, Any]:
    row_artifacts = {
        "metadata_boundary.csv": (row_counts["metadata"], "approved_unique_trial"),
        "trial_inventory.csv": (row_counts["inventory"], "trial_fold_use"),
        "parameter_estimates.csv": (row_counts["parameters"], "subject_fold"),
        "optimizer_diagnostics.csv": (row_counts["optimizers"], "fit_role_subject_fold"),
        "heldout_metrics.csv": (row_counts["metrics"], "trial_model_mask_target"),
        "subject_fold_metrics.csv": (row_counts["subject_fold"], "subject_fold_model_mask"),
        "driver_stability.csv": (row_counts["drivers"], "driver_comparison"),
        "driver_mean_trajectories.csv": (row_counts["trajectories"], "subject_session_model_timepoint"),
        "fold_summary.csv": (row_counts["fold_summary"], "fold"),
        "subject_summary.csv": (row_counts["subject_summary"], "subject"),
    }
    other = [
        "resolved_config.yaml",
        "synthetic_preflight.json",
        "fold_calibration.json",
        "summary.json",
        "summary.md",
        *(f"fold_{session}_report.json" for session in SESSION_IDS),
    ]
    artifacts: dict[str, Any] = {}
    for name, (expected, unit) in row_artifacts.items():
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"required artifact missing: {name}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            actual = sum(1 for _ in csv.DictReader(handle))
        if actual != int(expected):
            raise RuntimeError(f"artifact row count mismatch: {name}: {actual} != {expected}")
        artifacts[name] = {
            "required": True,
            "present": True,
            "rows_data": actual,
            "row_unit": unit,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    for name in other:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"required artifact missing/empty: {name}")
        artifacts[name] = {
            "required": True,
            "present": True,
            "rows_data": None,
            "row_unit": None,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return artifacts


def _source_provenance(config_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    paths = {
        "launch_config": config_path.resolve(),
        "runner": Path(__file__).resolve(),
        "measured_config": REPO_ROOT / MEASURED_CONFIG_PATH,
        "measured_runner": REPO_ROOT / "experiments/evaluate_t3_measured_reconstruction_null.py",
        "shared_loader_runner": REPO_ROOT / "experiments/evaluate_shared_neural_driver_unified.py",
        "adaptive_adapter_runner": REPO_ROOT / "experiments/evaluate_adaptive_shared_neural_ssm.py",
        "unified_loader": REPO_ROOT / "src/data/unified_physiology.py",
        "cache_index": REPO_ROOT / "src/data/clean_physiology_cache.py",
        "balloon_model": REPO_ROOT / "src/inference/t3a_balloon_robust_ssm.py",
    }

    def display(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    return ({name: display(path) for name, path in paths.items()}, {name: _sha256(path) for name, path in paths.items()})


def run(config: Mapping[str, Any], run_dir: Path, *, config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Run preflight, metadata gate, measured folds, and atomic publication."""

    validate_config(config)
    output_root = (REPO_ROOT / str(config["output"]["root"])).resolve()
    resolved = Path(run_dir).resolve()
    try:
        relative_run = resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"run directory must be below {output_root}") from exc
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {resolved}")
    source_paths, source_hashes = _source_provenance(Path(config_path))
    resolved.mkdir(parents=True, exist_ok=False)
    _atomic_write(resolved / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    started_at = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    folds = _folds()
    base_manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": str(relative_run),
        "analysis_kind": "three_session_leave_one_session_out",
        "scope": config["experiment"]["scope"],
        "status": "incomplete",
        "run_state": "initial",
        "completion_status": "incomplete",
        "stage": "before_synthetic_preflight",
        "started_at": started_at,
        "updated_at": started_at,
        "source_paths": source_paths,
        "source_sha256": source_hashes,
        "input_hash_scope": "metadata manifests plus manifest-declared selected public source hashes; selected array content is not fully hashed",
        "boundary": {
            "subjects": list(config["data"]["subjects"]),
            "sessions": list(SESSION_IDS),
            "target": "MA",
            "validation_subjects_closed": list(config["data"]["validation_subjects_closed"]),
            "protected_subjects_closed": list(config["data"]["protected_subjects_closed"]),
            "measured_data_enabled": True,
            "validation_data_enabled": False,
            "protected_data_enabled": False,
            "validation_data_opened": False,
            "protected_data_opened": False,
            "validation_subject_array_access_count": 0,
            "protected_subject_array_access_count": 0,
            "global_metadata_index_may_enumerate_closed_records": True,
            "closed_record_arrays_not_dereferenced": True,
            "qualification_eligible": False,
            "decision_eligibility": False,
        },
        "split_proof": {
            "unit": "whole_session_within_subject",
            "folds": [dict(item) for item in folds],
            "expected_unique_trial_count": 540,
            "expected_fold_use_count": 1620,
            "expected_training_trial_uses": 1080,
            "expected_heldout_trial_scores": 540,
            "fit_heldout_session_overlap": 0,
            "heldout_fit_gauge_channel_calibration_or_warm_state_calls": 0,
        },
        "fold_progress": [
            {**dict(fold), "run_state": "pending", "completed_subjects": []}
            for fold in folds
        ],
    }
    progress = dict(base_manifest)
    _atomic_json(resolved / "manifest.json", progress)
    rows: dict[str, list[dict[str, Any]]] = {
        key: []
        for key in ("metadata", "inventory", "parameters", "optimizers", "metrics", "subject_fold", "drivers", "trajectories", "fold_summary", "subject_summary")
    }
    fold_calibrations: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    mean_driver_records: dict[tuple[str, str, str], np.ndarray] = {}
    measured_array_load_attempted = False
    measured_arrays_may_have_opened = False
    measured_array_load_completed = False
    try:
        source = load_measured_config(REPO_ROOT / str(config["sources"]["measured_config"]))
        preflight_started = time.perf_counter()
        preflight = _synthetic_preflight(config, source)
        preflight["elapsed_seconds"] = time.perf_counter() - preflight_started
        _atomic_json(resolved / "synthetic_preflight.json", preflight)
        if not preflight["passed"]:
            raise RuntimeError("synthetic software preflight failed")
        progress = {
            **progress,
            "run_state": "partial",
            "stage": "synthetic_preflight_complete_before_metadata",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "synthetic_preflight": {"status": preflight["status"], "elapsed_seconds": preflight["elapsed_seconds"]},
        }
        _atomic_json(resolved / "manifest.json", progress)
        print(json.dumps({"stage": progress["stage"], "preflight": progress["synthetic_preflight"]}), flush=True)

        metadata_summary, metadata_rows, input_hashes = _validate_metadata(config)
        rows["metadata"] = list(metadata_rows)
        rows["inventory"] = _trial_inventory(metadata_rows)
        _write_tables(resolved, rows)
        progress = {
            **progress,
            "stage": "metadata_boundary_complete_before_array_load",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "input_hashes": input_hashes,
            "metadata_boundary": metadata_summary,
        }
        _atomic_json(resolved / "manifest.json", progress)
        print(json.dumps({"stage": progress["stage"], "metadata": metadata_summary}), flush=True)

        validate_config(config)
        measured_array_load_attempted = True
        measured_arrays_may_have_opened = True
        grouped, loader_contracts = _load_selected_trials(config, metadata_rows)
        measured_array_load_completed = True
        progress = {
            **progress,
            "stage": "approved_fit_arrays_loaded",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "approved_unique_trial_windows_materialized": 540,
            "loader_contracts": loader_contracts,
        }
        _atomic_json(resolved / "manifest.json", progress)
        print(json.dumps({"stage": progress["stage"], "approved_unique_trials": 540}), flush=True)

        fit_config = {
            "bounds": config["analysis"]["kappa_coordinate"]["bounds"],
            "prior_mean": config["analysis"]["kappa_coordinate"]["prior_mean"],
            "prior_sd": config["analysis"]["kappa_coordinate"]["prior_sd"],
            "grid_points": config["analysis"]["optimizer"]["transformed_grid_points"],
            "max_iterations": config["analysis"]["optimizer"]["max_iterations"],
            "xatol": config["analysis"]["optimizer"]["xatol"],
            "boundary_fraction": config["analysis"]["optimizer"]["boundary_fraction_of_log_span"],
        }
        workers = min(int(config["analysis"]["optimizer"]["workers"]), len(config["data"]["subjects"]))
        for fold_index, fold in enumerate(folds):
            fold_started = time.perf_counter()
            prepared, (base_parameters, observation_spec, balloon_config), calibration = _prepare_fold(config, source, grouped, fold)
            fold_calibrations.append(calibration)
            by_session_subject: dict[str, dict[str, list[PreparedTrial]]] = {
                session: defaultdict(list) for session in SESSION_IDS
            }
            for session, values in prepared.items():
                for item in values:
                    by_session_subject[session][item.trial.subject].append(item)
            tasks = []
            for subject in config["data"]["subjects"]:
                train_by_session = {
                    session: tuple(_observations(item) for item in by_session_subject[session][str(subject)])
                    for session in fold["train_sessions"]
                }
                heldout_trials = tuple(by_session_subject[str(fold["heldout_session"])][str(subject)])
                if any(len(values) != 10 for values in train_by_session.values()) or len(heldout_trials) != 10:
                    raise RuntimeError(f"{fold['fold_id']}/{subject}: trial cardinality mismatch")
                tasks.append({
                    "subject": str(subject),
                    "fold_id": str(fold["fold_id"]),
                    "heldout_session": str(fold["heldout_session"]),
                    "train_sessions": tuple(fold["train_sessions"]),
                    "train_by_session": train_by_session,
                    "heldout_trials": heldout_trials,
                    "base_parameters": base_parameters,
                    "observation_spec": observation_spec,
                    "balloon_config": balloon_config,
                    "fit_config": fit_config,
                    "config": config,
                })
            progress_folds = [dict(item) for item in progress["fold_progress"]]
            progress_folds[fold_index] = {**progress_folds[fold_index], "run_state": "running", "started_at": datetime.now(timezone.utc).isoformat()}
            progress = {
                **progress,
                "stage": f"{fold['fold_id']}_running",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "fold_progress": progress_folds,
            }
            _atomic_json(resolved / "manifest.json", progress)
            completed_subjects: list[str] = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_subject = {executor.submit(_fit_and_score_subject, task): task["subject"] for task in tasks}
                for future in as_completed(future_to_subject):
                    subject = str(future_to_subject[future])
                    result = future.result()
                    rows["parameters"].append(result["parameter_row"])
                    rows["optimizers"].extend(result["optimizer_rows"])
                    rows["metrics"].extend(result["metric_rows"])
                    rows["drivers"].extend(result["driver_comparison_rows"])
                    heldout_session = str(fold["heldout_session"])
                    for model, driver in result["mean_drivers"].items():
                        mean_driver_records[(subject, heldout_session, model)] = np.asarray(driver, dtype=np.float64)
                        rows["trajectories"].extend({
                            "fold_id": str(fold["fold_id"]),
                            "heldout_session": heldout_session,
                            "subject": subject,
                            "model": model,
                            "time_index": index,
                            "relative_time_s": float(config["data"]["window_offset_s"]) + index / float(config["analysis"]["sampling_hz"]),
                            "mean_driver_over_10_trials": float(value),
                        } for index, value in enumerate(driver))
                    completed_subjects.append(subject)
                    completed_subjects.sort()
                    progress_folds = [dict(item) for item in progress["fold_progress"]]
                    progress_folds[fold_index] = {**progress_folds[fold_index], "completed_subjects": completed_subjects}
                    progress = {
                        **progress,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "fold_progress": progress_folds,
                    }
                    _atomic_json(resolved / "manifest.json", progress)
                    print(json.dumps({"fold": fold["fold_id"], "subject_complete": subject, "completed": len(completed_subjects), "total": 18}), flush=True)

            rows["subject_fold"] = _subject_fold_metrics(rows["metrics"])
            current_fold_summary = _fold_summary(rows["subject_fold"], rows["parameters"], fold, config)
            current_fold_summary["elapsed_seconds"] = time.perf_counter() - fold_started
            fold_summaries.append(current_fold_summary)
            rows["fold_summary"].append(_fold_summary_row(current_fold_summary))
            _write_tables(resolved, rows)
            _atomic_json(resolved / f"fold_{fold['heldout_session']}_report.json", {
                "schema": SCHEMA,
                "manifest_status_owner": "manifest.json",
                "fold": current_fold_summary,
                "calibration": calibration,
                "claim_boundary": "exploratory subject-within-session LOSO only; no trait or qualification claim",
            })
            progress_folds = [dict(item) for item in progress["fold_progress"]]
            progress_folds[fold_index] = {
                **progress_folds[fold_index],
                "run_state": "complete",
                "completed_subjects": list(config["data"]["subjects"]),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "report": f"fold_{fold['heldout_session']}_report.json",
                "primary_delta_nll": current_fold_summary["primary_delta_nll_candidate_minus_M0"],
            }
            progress = {
                **progress,
                "stage": f"{fold['fold_id']}_complete",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "fold_progress": progress_folds,
            }
            _atomic_json(resolved / "manifest.json", progress)
            print(json.dumps({"fold_complete": fold["fold_id"], "summary": current_fold_summary}), flush=True)

        rows["drivers"].extend(_driver_session_rows(mean_driver_records, config))
        summary, subject_rows = _final_summary(config, metadata_summary, preflight, rows, fold_summaries)
        completed_at = datetime.now(timezone.utc).isoformat()
        summary.update({
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_seconds": time.perf_counter() - start_clock,
        })
        rows["subject_summary"] = subject_rows
        _write_tables(resolved, rows)
        _atomic_json(resolved / "fold_calibration.json", {
            "schema": SCHEMA,
            "fit_fold_only": True,
            "heldout_fit_calls": 0,
            "folds": fold_calibrations,
        })
        _atomic_json(resolved / "summary.json", summary)
        _atomic_write(resolved / "summary.md", _markdown_report(summary, subject_rows))
        artifacts = _artifact_audit(resolved, summary["artifact_row_counts"])
        manifest = {
            **progress,
            "status": "exploratory_complete",
            "run_state": "complete",
            "completion_status": "complete",
            "stage": "complete",
            "updated_at": completed_at,
            "completed_at": completed_at,
            "elapsed_seconds": summary["elapsed_seconds"],
            "input_hashes": input_hashes,
            "input_hash_scope": "metadata manifests plus manifest-declared selected public source hashes; selected array content is not fully hashed",
            "approved_unique_trial_windows_materialized": 540,
            "measured_fit_array_load_attempted": True,
            "measured_fit_arrays_may_have_opened": True,
            "measured_fit_array_load_completed": True,
            "validation_subject_array_access_count": 0,
            "protected_subject_array_access_count": 0,
            "primary_result": summary["primary_result"],
            "scientific_verdict": summary["scientific_verdict"],
            "claim_boundary": summary["claim_boundary"],
            "artifact_row_counts": summary["artifact_row_counts"],
            "artifacts": artifacts,
            "summary_pointer": "summary.json",
            "git": _git_payload(),
            "runtime": summary["runtime"],
        }
        # Complete publication remains inside the failure-capture boundary.
        _atomic_json(resolved / "manifest.json", manifest)
        print(json.dumps({"stage": "complete", "scientific_verdict": summary["scientific_verdict"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
        return summary
    except Exception as exc:
        try:
            _write_tables(resolved, rows)
        except Exception:
            pass
        failed_at = datetime.now(timezone.utc).isoformat()
        failure = {
            **progress,
            "status": "incomplete_failed",
            "run_state": "failure",
            "completion_status": "incomplete",
            "failed_after_stage": progress.get("stage"),
            "stage": "failed",
            "updated_at": failed_at,
            "failed_at": failed_at,
            "elapsed_seconds": time.perf_counter() - start_clock,
            "measured_fit_array_load_attempted": measured_array_load_attempted,
            "measured_fit_arrays_may_have_opened": measured_arrays_may_have_opened,
            "measured_fit_array_load_completed": measured_array_load_completed,
            "validation_data_opened": False,
            "protected_data_opened": False,
            "validation_subject_array_access_count": 0,
            "protected_subject_array_access_count": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=16),
            "partial_row_counts": {key: len(value) for key, value in rows.items()},
            "partial_artifacts": sorted(path.name for path in resolved.iterdir() if path.is_file()),
        }
        _atomic_json(resolved / "manifest.json", failure)
        _atomic_json(resolved / "summary.json", {
            "schema": SCHEMA,
            "manifest_status_owner": "manifest.json",
            "scientific_result": "none_incomplete_run",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        _atomic_write(resolved / "summary.md", "\n".join([
            "# T3 实验第三步：未完成报告",
            "",
            f"运行在 `{failure['failed_after_stage']}` 后失败；没有完整科学结果。",
            f"错误：`{failure['error_type']}: {failure['error']}`。",
            "状态、边界、partial rows 与 traceback 见 `manifest.json`。validation/protected 始终关闭。",
            "",
        ]))
        raise


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_step3_v1")
        run_dir = REPO_ROOT / str(config["output"]["root"]) / stamp
    else:
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    run(config, run_dir, config_path=config_path)
    return run_dir


if __name__ == "__main__":
    main()
