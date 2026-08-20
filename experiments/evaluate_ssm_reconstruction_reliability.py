#!/usr/bin/env python3
"""Run the complete development-only SSM reconstruction reliability audit.

The runner owns the S1--S3 matrix declared in
``docs/analysis/SSM_RECONSTRUCTION_RELIABILITY_PLAN.md``.  It cross-fits every
subject/task cell, keeps native dependency groups intact, writes one atomic
artifact bundle, and never dereferences the protected/unused cohorts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_adaptive_shared_neural_ssm import (
    STATE_NAMES,
    _apply_eeg_adapter,
    _chromophore_targets,
    _downsample_valid_mask,
    _fit_eeg_adapter,
    _fit_model,
    _local_eeg_indices,
    _paired_hbr_indices,
)
from experiments.evaluate_shared_neural_driver_unified import (
    Trial,
    _select_active_hbo,
)
from src.data.unified_physiology import (
    REFEDContinuousSequenceDataset,
    UnifiedPhysiologyWindowDataset,
)
from src.inference.adaptive_neurovascular_ssm import (
    apply_adaptive_ssm,
    fit_to_mapping,
)
from src.metrics.trajectory_reliability import trajectory_reliability_metrics
from src.visualization.token_physiology_plots import save_figure_atomic


SCHEMA = "ssm_reconstruction_reliability_v2"
TRAJECTORY_SCHEMA = "ssm_reconstruction_trajectory_v2"
MODEL_OBSERVATION_CONTRACT = {
    "adaptive_joint": {
        "observed_modalities": ("EEG", "HbO", "HbR"),
        "target_modalities": ("EEG", "HbO", "HbR"),
        "role": "posterior_self_fit_diagnostic",
    },
    "adaptive_eeg_only": {
        "observed_modalities": ("EEG",),
        "target_modalities": ("HbO", "HbR"),
        "role": "out_of_modality_fnirs_reconstruction",
    },
    "adaptive_fnirs_only": {
        "observed_modalities": ("HbO", "HbR"),
        "target_modalities": ("EEG",),
        "role": "out_of_modality_eeg_proxy_reconstruction",
    },
}
PRIMARY_BOOTSTRAP_METRICS = (
    "hbo_trajectory_deviation_nrmse",
    "hbr_trajectory_deviation_nrmse",
    "eeg_trajectory_deviation_nrmse",
    "hbo_temporal_sd_ratio",
    "hbr_temporal_sd_ratio",
    "eeg_temporal_sd_ratio",
    "hbo_standardized_residual_rms",
    "hbr_standardized_residual_rms",
    "eeg_standardized_residual_rms",
    "hbo_predictive_95_coverage",
    "hbr_predictive_95_coverage",
    "eeg_predictive_95_coverage",
)


@dataclass(frozen=True)
class Unit:
    trial: Trial
    unit_id: str
    dependency_group: str
    stratum: str


@dataclass(frozen=True)
class Job:
    task: dict[str, Any]
    subject: str
    role: str
    config: dict[str, Any]
    smoke_max_units: int | None = None


@dataclass
class Prediction:
    task_id: str
    family: str
    stage: str
    dataset_id: str
    role: str
    subject: str
    stratum: str
    dependency_group: str
    unit_id: str
    fold_index: int
    model: str
    spatial_mode: str
    observed_modalities: tuple[str, ...]
    target_modalities: tuple[str, ...]
    observation_role: str
    fit_source: str
    fit_parameter_hash: str
    time_s: np.ndarray
    truth_hbo: np.ndarray
    estimate_hbo: np.ndarray
    truth_hbr: np.ndarray
    estimate_hbr: np.ndarray
    eeg_observation: np.ndarray
    eeg_reconstruction: np.ndarray
    predictive_std: np.ndarray
    eeg_valid_mask: np.ndarray
    fnirs_valid_mask: np.ndarray
    states: np.ndarray
    state_std: np.ndarray
    selected_fnirs_channels: tuple[str, ...]
    selected_eeg_channels: tuple[str, ...]


@dataclass
class JobResult:
    predictions: list[Prediction]
    fit_rows: list[dict[str, Any]]
    inventory: dict[str, Any]
    subject_curve_rows: list[dict[str, Any]]
    failure: str | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _jsonable(row.get(key, "")) for key in fields} for row in values])


def _git_payload() -> dict[str, str]:
    def call(*args: str) -> str:
        return subprocess.run(
            args, cwd=REPO_ROOT, check=False, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "status_short": call("git", "status", "--short"),
    }


def _subject_number(subject: str) -> int:
    match = re.search(r"(\d+)$", str(subject))
    if match is None:
        raise ValueError(f"subject has no numeric suffix: {subject}")
    return int(match.group(1))


def _role_for_subject(dataset_id: str, subject: str, data: Mapping[str, Any]) -> str:
    group = "single_trial" if dataset_id == "eeg_fnirs_single_trial" else "simultaneous"
    if subject in set(data["core_fit_subjects"][group]):
        return "fit"
    if subject in set(data["core_development_subjects"][group]):
        return "development_validation"
    raise PermissionError(f"core subject outside declared development scope: {dataset_id}/{subject}")


def _core_subjects(task: Mapping[str, Any], data: Mapping[str, Any]) -> list[tuple[str, str]]:
    group = "single_trial" if task["dataset_id"] == "eeg_fnirs_single_trial" else "simultaneous"
    return [
        *((str(value), "fit") for value in data["core_fit_subjects"][group]),
        *((str(value), "development_validation") for value in data["core_development_subjects"][group]),
    ]


def _visual_pair_id(record_id: str, epoch_id: Any) -> str:
    stem = re.sub(r"_Probe[12]$", "", str(record_id))
    return f"{stem}|epoch={int(epoch_id)}"


def _visual_probe(record_id: str) -> str:
    match = re.search(r"_(Probe[12])$", str(record_id))
    if match is None:
        raise ValueError(f"visual record does not identify its probe: {record_id}")
    return match.group(1)


def assign_group_folds(group_ids: Sequence[str], fold_count: int, seed: int) -> dict[str, int]:
    """Assign complete dependency groups to deterministic balanced folds."""

    unique = sorted(set(str(value) for value in group_ids))
    if len(unique) < 2:
        raise ValueError("cross-fitting requires at least two dependency groups")
    count = min(int(fold_count), len(unique))
    if count < 2:
        raise ValueError("fold_count must permit at least two folds")
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest(),
    )
    return {group: index % count for index, group in enumerate(ranked)}


def _sample_to_trial(
    sample: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    unit_index: int,
) -> Trial:
    baseline_n = int(round(float(task["baseline_duration_s"]) * 10.0))
    fnirs = np.asarray(sample["fnirs"], dtype=np.float64).T
    if baseline_n > 0:
        if baseline_n >= len(fnirs):
            raise ValueError("baseline consumes the complete fNIRS window")
        fnirs = fnirs - np.mean(fnirs[:baseline_n], axis=0, keepdims=True)
    artifact_mask = np.asarray(sample["artifact_mask"]["eeg"], dtype=bool)
    return Trial(
        condition_id=str(task["task_id"]),
        dataset_id=str(task["dataset_id"]),
        subject=str(sample["subject"]),
        record_id=str(sample["record_id"]),
        event_index=int(sample["event"].get("event_index", unit_index)),
        eeg=np.asarray(sample["eeg"], dtype=np.float64).T,
        fnirs=fnirs,
        fnirs_channel_names=tuple(str(value) for value in sample["channel_names"]["fnirs"]),
        fnirs_roles=tuple(str(value) for value in sample["component_roles"]["fnirs"]),
        eeg_artifact_fraction=float(np.mean(artifact_mask)),
        eeg_channel_names=tuple(str(value) for value in sample["channel_names"]["eeg"]),
        eeg_positions=np.asarray(
            [[row.get(axis, np.nan) for axis in ("x", "y", "z")] for row in sample["channel_geometry"]["eeg"]],
            dtype=np.float64,
        ),
        fnirs_positions=np.asarray(
            [[row.get(axis, np.nan) for axis in ("x", "y", "z")] for row in sample["channel_geometry"]["fnirs"]],
            dtype=np.float64,
        ),
        eeg_valid_mask=np.asarray(sample["valid_mask"]["eeg"], dtype=bool).copy(),
        fnirs_valid_mask=np.asarray(sample["valid_mask"]["fnirs"], dtype=bool).copy(),
    )


def _trial_modality_masks(trial: Trial) -> tuple[np.ndarray, np.ndarray]:
    """Return independent 10 Hz EEG and fNIRS observation support masks."""

    target_length = len(trial.fnirs)
    eeg_finite = np.all(np.isfinite(trial.eeg), axis=1)
    if trial.eeg_valid_mask is not None:
        eeg_finite &= np.asarray(trial.eeg_valid_mask, dtype=bool).reshape(-1)
    eeg_mask = _downsample_valid_mask(eeg_finite, target_length)
    fnirs_mask = (
        np.ones(target_length, dtype=bool)
        if trial.fnirs_valid_mask is None
        else np.asarray(trial.fnirs_valid_mask, dtype=bool).copy()
    )
    if fnirs_mask.shape != (target_length,):
        raise ValueError("fNIRS validity mask does not match the SSM target")
    fnirs_mask &= np.all(np.isfinite(trial.fnirs), axis=1)
    return eeg_mask, fnirs_mask


def _unit_has_any_support(unit: Unit) -> bool:
    try:
        eeg_mask, fnirs_mask = _trial_modality_masks(unit.trial)
    except ValueError:
        return False
    return bool(np.any(eeg_mask) or np.any(fnirs_mask))


def _unit_is_full(unit: Unit) -> bool:
    """Training fits retain the historical fully observed window contract."""

    try:
        eeg_mask, fnirs_mask = _trial_modality_masks(unit.trial)
    except ValueError:
        return False
    return bool(np.all(eeg_mask) and np.all(fnirs_mask))


def _load_core_units(job: Job) -> tuple[list[Unit], dict[str, Any]]:
    task, data = job.task, job.config["data"]
    dataset = UnifiedPhysiologyWindowDataset(
        cache_root=data["cache_root"],
        dataset_ids=(task["dataset_id"],),
        window_duration_s=float(task["window_duration_s"]),
        window_offset_s=float(task["window_offset_s"]),
        eeg_signal_branch=str(data["eeg_signal_branch"]),
        require_eeg_artifact_cache=task["dataset_id"] == "eeg_fnirs_single_trial",
    )
    selected = [
        index
        for index, ref in enumerate(dataset.windows)
        if ref.record.canonical_subject_id == job.subject
        and ref.record.base_record_id == task["record_id"]
        and str(ref.event.get("label")) == str(task["label"])
    ]
    if job.smoke_max_units is not None:
        selected = selected[: int(job.smoke_max_units)]
    units = []
    for local_index, index in enumerate(selected):
        sample = dataset[index]
        trial = _sample_to_trial(sample, task, unit_index=local_index)
        unit_id = f"{job.subject}|{trial.record_id}|event={trial.event_index}"
        unit = Unit(trial=trial, unit_id=unit_id, dependency_group=unit_id, stratum=trial.record_id)
        if not _unit_has_any_support(unit):
            raise RuntimeError(f"core window lacks complete support: {unit_id}")
        units.append(unit)
    if len(units) < 2:
        raise RuntimeError(f"{task['task_id']}/{job.subject}: fewer than two admitted windows")
    return units, {
        "source_window_count": len(selected),
        "admitted_window_count": len(units),
        "excluded_incomplete_count": 0,
        "dependency_group_count": len(units),
        "loader_contract": dataset.contract_summary(),
    }


def _load_visual_units(job: Job) -> tuple[list[Unit], dict[str, Any]]:
    task, data = job.task, job.config["data"]
    dataset = UnifiedPhysiologyWindowDataset(
        cache_root=data["cache_root"],
        dataset_ids=("visual_cognitive_motivation",),
        window_duration_s=float(task["window_duration_s"]),
        window_offset_s=float(task["window_offset_s"]),
        eeg_signal_branch=str(data["eeg_signal_branch"]),
    )
    selected = [
        index
        for index, ref in enumerate(dataset.windows)
        if ref.record.canonical_subject_id == job.subject
        and str(ref.event.get("label")) == str(task["label"])
        and str(ref.event.get("label")) != "unknown"
    ]
    grouped_refs: dict[str, list[int]] = defaultdict(list)
    for index in selected:
        ref = dataset.windows[index]
        epoch_id = ref.event.get("metadata", {}).get("epoch_id")
        if epoch_id is None:
            continue
        grouped_refs[_visual_pair_id(ref.record.base_record_id, epoch_id)].append(index)
    exact_pairs = {key: value for key, value in grouped_refs.items() if len(value) == 2}
    if job.smoke_max_units is not None:
        kept_groups = sorted(exact_pairs)[: max(2, int(job.smoke_max_units) // 2)]
        exact_pairs = {key: exact_pairs[key] for key in kept_groups}
    loaded: dict[str, list[Unit]] = defaultdict(list)
    incomplete_groups: set[str] = set()
    for group_id, indices in sorted(exact_pairs.items()):
        for index in indices:
            sample = dataset[index]
            trial = _sample_to_trial(sample, task, unit_index=index)
            unit_id = f"{group_id}|{_visual_probe(trial.record_id)}"
            unit = Unit(
                trial=trial,
                unit_id=unit_id,
                dependency_group=group_id,
                stratum=_visual_probe(trial.record_id),
            )
            if not _unit_has_any_support(unit):
                incomplete_groups.add(group_id)
            loaded[group_id].append(unit)
    units = [
        unit
        for group_id, values in sorted(loaded.items())
        if group_id not in incomplete_groups
        for unit in values
    ]
    if len(set(unit.dependency_group for unit in units)) < 2:
        raise RuntimeError(f"{task['task_id']}/{job.subject}: fewer than two complete probe pairs")
    return units, {
        "source_window_count": len(selected),
        "candidate_exact_pair_count": len(exact_pairs),
        "admitted_window_count": len(units),
        "excluded_nonpair_window_count": len(selected) - 2 * len(grouped_refs),
        "excluded_incomplete_dependency_group_count": len(incomplete_groups),
        "dependency_group_count": len(set(unit.dependency_group for unit in units)),
        "loader_contract": dataset.contract_summary(),
    }


def _load_refed_units(job: Job) -> tuple[list[Unit], dict[str, Any]]:
    task, data = job.task, job.config["data"]
    dataset = REFEDContinuousSequenceDataset(
        cache_root=data["cache_root"],
        window_duration_s=float(task["window_duration_s"]),
        window_stride_s=float(task["window_duration_s"]),
        include_partial_windows=False,
        eeg_signal_branch=str(data["eeg_signal_branch"]),
    )
    selected = [
        index for index, ref in enumerate(dataset.windows)
        if ref.record.canonical_subject_id == job.subject
    ]
    if job.smoke_max_units is not None:
        videos = sorted({dataset.windows[index].record.base_record_id for index in selected})[:2]
        per_video = max(1, int(job.smoke_max_units) // len(videos))
        # Rebuild per video so smoke preserves the dependency contract instead
        # of truncating several consecutive windows from the first video.
        selected = [
            index
            for video in videos
            for index in [
                value
                for value in range(len(dataset.windows))
                if dataset.windows[value].record.canonical_subject_id == job.subject
                and dataset.windows[value].record.base_record_id == video
            ][:per_video]
        ]
    units = []
    for index in selected:
        ref = dataset.windows[index]
        sample = dataset[index]
        trial = _sample_to_trial(sample, task, unit_index=index)
        start_ms = int(round(float(ref.window_offset_s) * 1000.0))
        group_id = f"{job.subject}|{trial.record_id}"
        unit = Unit(
            trial=trial,
            unit_id=f"{group_id}|start_ms={start_ms}",
            dependency_group=group_id,
            stratum="all_videos",
        )
        if not _unit_has_any_support(unit):
            raise RuntimeError(f"REFED full-only loader emitted incomplete unit: {unit.unit_id}")
        units.append(unit)
    if len(set(unit.dependency_group for unit in units)) < 2:
        raise RuntimeError(f"{task['task_id']}/{job.subject}: fewer than two videos")
    return units, {
        "source_window_count": len(selected),
        "admitted_window_count": len(units),
        "excluded_partial_windows_by_loader": int(
            dataset.contract_summary().get("partial_window_count", 0)
        ),
        "dependency_group_count": len(set(unit.dependency_group for unit in units)),
        "loader_contract": dataset.contract_summary(),
    }


def _dsr_segment_refs(dataset: UnifiedPhysiologyWindowDataset, subject: str) -> list[tuple[Any, str, int]]:
    by_block: dict[int, list[Any]] = defaultdict(list)
    for ref in dataset.windows:
        if ref.record.canonical_subject_id != subject or ref.record.base_record_id != "cnt_dsr":
            continue
        block_index = int(ref.event.get("metadata", {}).get("block_index", -1))
        if block_index >= 0:
            by_block[block_index].append(ref)
    output = []
    for block_index, refs in sorted(by_block.items()):
        refs.sort(key=lambda ref: float(ref.event["eeg_time_ms"]))
        anchor = refs[0]
        duration_s = (
            float(refs[-1].event["eeg_time_ms"])
            - float(anchor.event["eeg_time_ms"])
        ) / 1000.0 + 2.0
        starts = np.arange(0.0, duration_s - dataset.window_duration_s + 1e-9, dataset.window_duration_s)
        group_id = f"{subject}|block={block_index}"
        for start_s in starts.tolist():
            output.append((replace(anchor, window_offset_s=float(start_s)), group_id, int(round(start_s * 1000.0))))
    return output


def _load_dsr_units(job: Job) -> tuple[list[Unit], dict[str, Any]]:
    task, data = job.task, job.config["data"]
    dataset = UnifiedPhysiologyWindowDataset(
        cache_root=data["cache_root"],
        dataset_ids=("simultaneous_eeg_nirs",),
        window_duration_s=float(task["window_duration_s"]),
        window_offset_s=0.0,
        eeg_signal_branch=str(data["eeg_signal_branch"]),
        include_event_types={"stimulus"},
    )
    generated = _dsr_segment_refs(dataset, job.subject)
    if job.smoke_max_units is not None:
        groups = sorted({value[1] for value in generated})[:2]
        generated = [value for value in generated if value[1] in set(groups)][: int(job.smoke_max_units)]
    dataset.windows = [value[0] for value in generated]
    units = []
    incomplete = 0
    for index, (_ref, group_id, start_ms) in enumerate(generated):
        sample = dataset[index]
        trial = _sample_to_trial(sample, task, unit_index=index)
        unit = Unit(
            trial=trial,
            unit_id=f"{group_id}|start_ms={start_ms}",
            dependency_group=group_id,
            stratum="all_blocks",
        )
        if _unit_has_any_support(unit):
            units.append(unit)
        else:
            incomplete += 1
    if len(set(unit.dependency_group for unit in units)) < 2:
        raise RuntimeError(f"{task['task_id']}/{job.subject}: fewer than two complete DSR blocks")
    return units, {
        "source_stimulus_count": sum(
            1 for ref in dataset._build_windows()
            if ref.record.canonical_subject_id == job.subject and ref.record.base_record_id == "cnt_dsr"
        ),
        "generated_segment_count": len(generated),
        "admitted_window_count": len(units),
        "excluded_incomplete_count": incomplete,
        "dependency_group_count": len(set(unit.dependency_group for unit in units)),
        "loader_contract": dataset.contract_summary(),
    }


def _load_units(job: Job) -> tuple[list[Unit], dict[str, Any]]:
    if job.task["stage"] == "core":
        return _load_core_units(job)
    if job.task["dataset_id"] == "visual_cognitive_motivation":
        return _load_visual_units(job)
    if job.task["dataset_id"] == "refed":
        return _load_refed_units(job)
    if job.task["task_id"] == "simultaneous_dsr":
        return _load_dsr_units(job)
    raise ValueError(f"unsupported task loader: {job.task['task_id']}")


def _global_eeg_indices(trial: Trial) -> np.ndarray:
    selected = [
        index
        for index, name in enumerate(trial.eeg_channel_names)
        if not any(token in name.upper() for token in ("EOG", "ECG", "EMG"))
    ]
    if not selected:
        raise ValueError("global spatial mode has no admitted scalp EEG channels")
    return np.asarray(selected, dtype=int)


def _spatial_modes(job: Job) -> tuple[str, ...]:
    key = "core_spatial_modes" if job.task["stage"] == "core" else "descriptive_spatial_modes"
    return tuple(str(value) for value in job.config["analysis"][key])


def _run_stratum(
    job: Job,
    units: Sequence[Unit],
    fold_map: Mapping[str, int],
) -> tuple[list[Prediction], list[dict[str, Any]]]:
    task = job.task
    analysis = job.config["analysis"]
    predictions: list[Prediction] = []
    fit_rows: list[dict[str, Any]] = []
    fold_indices = sorted(set(fold_map.values()))
    baseline_samples = int(round(float(task["baseline_duration_s"]) * float(analysis["ssm"]["fs_hz"])))
    time_s = (
        np.arange(int(round(float(task["window_duration_s"]) * float(analysis["ssm"]["fs_hz"]))))
        / float(analysis["ssm"]["fs_hz"])
        + float(task["window_offset_s"])
    )
    for fold_index in fold_indices:
        candidate_train_units = [
            unit for unit in units if fold_map[unit.dependency_group] != fold_index
        ]
        train_units = [unit for unit in candidate_train_units if _unit_is_full(unit)]
        test_units = [unit for unit in units if fold_map[unit.dependency_group] == fold_index]
        if not train_units or not test_units:
            raise RuntimeError(f"empty train/test fold {fold_index} for {task['task_id']}/{job.subject}")
        train_trials = [unit.trial for unit in train_units]
        test_trials = [unit.trial for unit in test_units]
        channel_signatures = {
            (trial.eeg_channel_names, trial.fnirs_channel_names) for trial in train_trials + test_trials
        }
        if len(channel_signatures) != 1:
            raise RuntimeError(
                f"channel signature drift inside {task['task_id']}/{job.subject}/{units[0].stratum}"
            )
        hbo_indices, hbo_names, _ = _select_active_hbo(
            train_trials,
            baseline_duration_s=float(task["baseline_duration_s"]),
            task_duration_s=float(task["window_duration_s"]) - float(task["baseline_duration_s"]),
            count=int(analysis["fnirs_active_hbo_channels"]),
        )
        hbr_indices = _paired_hbr_indices(train_trials[0], hbo_indices)
        hbr_names = tuple(train_trials[0].fnirs_channel_names[int(index)] for index in hbr_indices)
        train_hbo, train_hbr = _chromophore_targets(train_trials, hbo_indices, hbr_indices)
        test_hbo, test_hbr = _chromophore_targets(test_trials, hbo_indices, hbr_indices)
        for spatial_mode in _spatial_modes(job):
            if spatial_mode == "local":
                eeg_indices = _local_eeg_indices(
                    train_trials[0], hbo_indices, int(analysis["local_eeg_channels"])
                )
            elif spatial_mode == "global":
                eeg_indices = _global_eeg_indices(train_trials[0])
            else:
                raise ValueError(f"unsupported spatial mode: {spatial_mode}")
            adapter, train_drivers = _fit_eeg_adapter(train_trials, eeg_indices)
            fit = _fit_model(
                train_drivers,
                train_hbo,
                train_hbr,
                analysis["ssm"],
                baseline_samples,
            )
            fit_mapping = fit_to_mapping(fit)
            fit_parameter_hash = hashlib.sha256(
                json.dumps(
                    _jsonable(fit_mapping),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            fit_rows.append(
                {
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "stage": task["stage"],
                    "dataset_id": task["dataset_id"],
                    "role": job.role,
                    "subject": job.subject,
                    "stratum": units[0].stratum,
                    "fold_index": fold_index,
                    "train_unit_count": len(train_units),
                    "train_partial_support_excluded_count": len(candidate_train_units) - len(train_units),
                    "test_unit_count": len(test_units),
                    "train_dependency_group_count": len({unit.dependency_group for unit in train_units}),
                    "test_dependency_group_count": len({unit.dependency_group for unit in test_units}),
                    "spatial_mode": spatial_mode,
                    "selected_fnirs_channels": "|".join(hbo_names + hbr_names),
                    "selected_eeg_channels": "|".join(adapter.channel_names),
                    "fit_source": "within_subject_crossfit_training_dependency_groups",
                    "fit_parameter_hash": fit_parameter_hash,
                    **fit_mapping,
                }
            )
            for test_unit, truth_hbo, truth_hbr in zip(test_units, test_hbo, test_hbr, strict=True):
                driver = _apply_eeg_adapter(test_unit.trial, adapter)
                eeg_mask, fnirs_mask = _trial_modality_masks(test_unit.trial)
                if len(driver) != len(time_s):
                    raise ValueError(
                        f"SSM time length drift: {task['task_id']}/{job.subject}/{test_unit.unit_id}"
                    )
                masked_driver = np.where(eeg_mask, driver, np.nan)
                masked_hbo = np.where(fnirs_mask, truth_hbo, np.nan)
                masked_hbr = np.where(fnirs_mask, truth_hbr, np.nan)
                outputs = {}
                if np.any(eeg_mask) and np.any(fnirs_mask):
                    outputs["adaptive_joint"] = apply_adaptive_ssm(
                        masked_driver,
                        fit,
                        hbo_observation=masked_hbo,
                        hbr_observation=masked_hbr,
                        observation_mode="joint",
                    )
                if np.any(eeg_mask):
                    outputs["adaptive_eeg_only"] = apply_adaptive_ssm(
                        masked_driver, fit, observation_mode="eeg_only"
                    )
                if np.any(fnirs_mask):
                    outputs["adaptive_fnirs_only"] = apply_adaptive_ssm(
                        None,
                        fit,
                        hbo_observation=masked_hbo,
                        hbr_observation=masked_hbr,
                        observation_mode="fnirs_only",
                    )
                for model, output in outputs.items():
                    observation_contract = MODEL_OBSERVATION_CONTRACT[model]
                    predictions.append(
                        Prediction(
                            task_id=str(task["task_id"]),
                            family=str(task["family"]),
                            stage=str(task["stage"]),
                            dataset_id=str(task["dataset_id"]),
                            role=job.role,
                            subject=job.subject,
                            stratum=test_unit.stratum,
                            dependency_group=test_unit.dependency_group,
                            unit_id=test_unit.unit_id,
                            fold_index=fold_index,
                            model=model,
                            spatial_mode=spatial_mode,
                            observed_modalities=observation_contract["observed_modalities"],
                            target_modalities=observation_contract["target_modalities"],
                            observation_role=str(observation_contract["role"]),
                            fit_source="within_subject_crossfit_training_dependency_groups",
                            fit_parameter_hash=fit_parameter_hash,
                            time_s=time_s.copy(),
                            truth_hbo=np.asarray(truth_hbo, dtype=np.float64),
                            estimate_hbo=output.hbo_reconstructed,
                            truth_hbr=np.asarray(truth_hbr, dtype=np.float64),
                            estimate_hbr=output.hbr_reconstructed,
                            eeg_observation=driver,
                            eeg_reconstruction=output.eeg_reconstructed,
                            predictive_std=output.observation_predictive_std,
                            eeg_valid_mask=eeg_mask,
                            fnirs_valid_mask=fnirs_mask,
                            states=output.states,
                            state_std=output.state_std,
                            selected_fnirs_channels=tuple(hbo_names + hbr_names),
                            selected_eeg_channels=adapter.channel_names,
                        )
                    )
    return predictions, fit_rows


def _subject_curves(predictions: Sequence[Prediction]) -> list[dict[str, Any]]:
    by_path: dict[tuple[str, ...], list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_path[
            (
                prediction.task_id,
                prediction.family,
                prediction.stage,
                prediction.dataset_id,
                prediction.role,
                prediction.subject,
                prediction.model,
                prediction.spatial_mode,
            )
        ].append(prediction)
    output: list[dict[str, Any]] = []
    for key, values in sorted(by_path.items()):
        by_dependency: dict[str, list[Prediction]] = defaultdict(list)
        for value in values:
            by_dependency[value.dependency_group].append(value)
        for modality in ("hbo", "hbr", "eeg"):
            group_observed = []
            group_reconstructed = []
            group_predictive = []
            for group_values in by_dependency.values():
                if modality == "hbo":
                    observed = [value.truth_hbo for value in group_values]
                    reconstructed = [value.estimate_hbo for value in group_values]
                    predictive = [value.predictive_std[:, 1] for value in group_values]
                elif modality == "hbr":
                    observed = [value.truth_hbr for value in group_values]
                    reconstructed = [value.estimate_hbr for value in group_values]
                    predictive = [value.predictive_std[:, 2] for value in group_values]
                else:
                    observed = [value.eeg_observation for value in group_values]
                    reconstructed = [value.eeg_reconstruction for value in group_values]
                    predictive = [value.predictive_std[:, 0] for value in group_values]
                group_observed.append(np.mean(np.stack(observed), axis=0))
                group_reconstructed.append(np.mean(np.stack(reconstructed), axis=0))
                group_predictive.append(np.mean(np.stack(predictive), axis=0))
            observed_curve = np.mean(np.stack(group_observed), axis=0)
            reconstructed_curve = np.mean(np.stack(group_reconstructed), axis=0)
            predictive_curve = np.mean(np.stack(group_predictive), axis=0)
            for index, time_value in enumerate(values[0].time_s):
                output.append(
                    {
                        "task_id": key[0],
                        "family": key[1],
                        "stage": key[2],
                        "dataset_id": key[3],
                        "role": key[4],
                        "subject": key[5],
                        "model": key[6],
                        "spatial_mode": key[7],
                        "modality": modality,
                        "time_s": float(time_value),
                        "dependency_groups": len(by_dependency),
                        "observed": float(observed_curve[index]),
                        "reconstructed": float(reconstructed_curve[index]),
                        "posterior_predictive_sd": float(predictive_curve[index]),
                    }
                )
    return output


def _execute_job(job: Job) -> JobResult:
    with threadpool_limits(limits=1):
        units, inventory = _load_units(job)
        fold_map = assign_group_folds(
            [unit.dependency_group for unit in units],
            int(job.config["analysis"]["folds"]),
            int(job.config["analysis"]["seed"]),
        )
        predictions: list[Prediction] = []
        fit_rows: list[dict[str, Any]] = []
        by_stratum: dict[str, list[Unit]] = defaultdict(list)
        for unit in units:
            by_stratum[unit.stratum].append(unit)
        for stratum, stratum_units in sorted(by_stratum.items()):
            stratum_groups = {unit.dependency_group for unit in stratum_units}
            if len(stratum_groups) < 2:
                raise RuntimeError(
                    f"{job.task['task_id']}/{job.subject}/{stratum}: fewer than two dependency groups"
                )
            stratum_predictions, stratum_fits = _run_stratum(job, stratum_units, fold_map)
            predictions.extend(stratum_predictions)
            fit_rows.extend(stratum_fits)
        predictions.sort(
            key=lambda value: (
                value.stratum,
                value.dependency_group,
                value.unit_id,
                value.fold_index,
                value.model,
                value.spatial_mode,
            )
        )
        inventory.update(
            {
                "task_id": job.task["task_id"],
                "family": job.task["family"],
                "stage": job.task["stage"],
                "dataset_id": job.task["dataset_id"],
                "role": job.role,
                "subject": job.subject,
                "stratum_count": len(by_stratum),
                "fold_count": len(set(fold_map.values())),
                "prediction_count": len(predictions),
            }
        )
        return JobResult(
            predictions=predictions,
            fit_rows=fit_rows,
            inventory=inventory,
            subject_curve_rows=_subject_curves(predictions),
        )


def _execute_job_safe(job: Job) -> JobResult:
    try:
        return _execute_job(job)
    except Exception:
        return JobResult(
            predictions=[],
            fit_rows=[],
            inventory={
                "task_id": job.task["task_id"],
                "family": job.task["family"],
                "stage": job.task["stage"],
                "dataset_id": job.task["dataset_id"],
                "role": job.role,
                "subject": job.subject,
                "status": "failed",
            },
            subject_curve_rows=[],
            failure=traceback.format_exc(),
        )


def _masked_basic_metrics(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(observed, dtype=np.float64)
    estimate = np.asarray(reconstructed, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(truth) & np.isfinite(estimate)
    truth = truth[mask]
    estimate = estimate[mask]
    residual = truth - estimate
    mse = float(np.mean(residual**2))
    bias = float(np.mean(estimate - truth))
    centered = truth - np.mean(truth)
    denominator = float(np.sum(centered**2))
    r2 = float("nan") if denominator <= 1e-12 else 1.0 - float(np.sum(residual**2)) / denominator
    if len(truth) < 2 or np.std(truth) <= 1e-12 or np.std(estimate) <= 1e-12:
        pcc = float("nan")
    else:
        pcc = float(np.corrcoef(truth, estimate)[0, 1])
    return {"mse": mse, "bias": bias, "pcc": pcc, "r2": r2}


def prediction_metrics(prediction: Prediction) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": prediction.task_id,
        "family": prediction.family,
        "stage": prediction.stage,
        "dataset_id": prediction.dataset_id,
        "role": prediction.role,
        "subject": prediction.subject,
        "stratum": prediction.stratum,
        "dependency_group": prediction.dependency_group,
        "unit_id": prediction.unit_id,
        "fold_index": prediction.fold_index,
        "model": prediction.model,
        "spatial_mode": prediction.spatial_mode,
        "observed_modalities": "|".join(prediction.observed_modalities),
        "target_modalities": "|".join(prediction.target_modalities),
        "observation_role": prediction.observation_role,
        "fit_source": prediction.fit_source,
        "fit_parameter_hash": prediction.fit_parameter_hash,
        "selected_fnirs_channels": "|".join(prediction.selected_fnirs_channels),
        "selected_eeg_channels": "|".join(prediction.selected_eeg_channels),
    }
    modalities = {
        "hbo": (
            prediction.truth_hbo,
            prediction.estimate_hbo,
            prediction.predictive_std[:, 1],
            prediction.fnirs_valid_mask,
        ),
        "hbr": (
            prediction.truth_hbr,
            prediction.estimate_hbr,
            prediction.predictive_std[:, 2],
            prediction.fnirs_valid_mask,
        ),
        "eeg": (
            prediction.eeg_observation,
            prediction.eeg_reconstruction,
            prediction.predictive_std[:, 0],
            prediction.eeg_valid_mask,
        ),
    }
    for modality, (observed, reconstructed, predictive, mask) in modalities.items():
        values = {
            **_masked_basic_metrics(observed, reconstructed, mask),
            **trajectory_reliability_metrics(
                observed,
                reconstructed,
                predictive_std=predictive,
                valid_mask=mask,
            ),
        }
        row.update({f"{modality}_{key}": value for key, value in values.items()})
    for state_index, state_name in enumerate(STATE_NAMES):
        row[f"state_{state_name}_posterior_sd_mean"] = float(
            np.mean(prediction.state_std[:, state_index])
        )
        row[f"state_{state_name}_posterior_sd_median"] = float(
            np.median(prediction.state_std[:, state_index])
        )
    row["relative_flow_min"] = float(np.min(1.0 + prediction.states[:, 1]))
    row["relative_flow_nonpositive_fraction"] = float(
        np.mean(1.0 + prediction.states[:, 1] <= 0.0)
    )
    return row


def _finite_mean(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _numeric_metrics(rows: Sequence[Mapping[str, Any]], identifiers: set[str]) -> list[str]:
    metrics = []
    for key, value in rows[0].items():
        if key not in identifiers and isinstance(value, (int, float, np.integer, np.floating, bool)):
            metrics.append(key)
    return metrics


def _aggregate_level(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    metric_names: Sequence[str],
    *,
    count_fields: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in group_fields)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(group_fields, key, strict=True))
        for output_name, source_field in (count_fields or {}).items():
            item[output_name] = len({str(value[source_field]) for value in values})
        for metric in metric_names:
            item[metric] = _finite_mean(value[metric] for value in values)
        output.append(item)
    return output


def aggregate_metrics(
    window_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
    list[str],
]:
    """Aggregate window -> dependency group -> subject -> task cell."""

    identifiers = {
        "task_id", "family", "stage", "dataset_id", "role", "subject", "stratum",
        "dependency_group", "unit_id", "fold_index", "model", "spatial_mode",
        "selected_fnirs_channels", "selected_eeg_channels",
    }
    metrics = _numeric_metrics(window_rows, identifiers)
    dependency_fields = (
        "task_id", "family", "stage", "dataset_id", "role", "subject", "model",
        "spatial_mode", "dependency_group",
    )
    dependency_rows = _aggregate_level(
        window_rows,
        dependency_fields,
        metrics,
        count_fields={"windows": "unit_id"},
    )
    subject_fields = (
        "task_id", "family", "stage", "dataset_id", "role", "subject", "model", "spatial_mode",
    )
    subject_rows = _aggregate_level(
        dependency_rows,
        subject_fields,
        metrics,
        count_fields={"dependency_groups": "dependency_group"},
    )
    task_fields = ("task_id", "family", "stage", "dataset_id", "role", "model", "spatial_mode")
    task_groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        task_groups[tuple(str(row[field]) for field in task_fields)].append(row)
    rng = np.random.default_rng(int(seed))
    summary_rows: list[dict[str, Any]] = []
    bootstrap_arrays = []
    bootstrap_keys = []
    for key, values in sorted(task_groups.items()):
        item: dict[str, Any] = dict(zip(task_fields, key, strict=True))
        item["subjects"] = len(values)
        item["dependency_groups"] = int(sum(int(value["dependency_groups"]) for value in values))
        for metric in metrics:
            observed = np.asarray([float(value[metric]) for value in values], dtype=np.float64)
            finite = observed[np.isfinite(observed)]
            item[metric] = float(np.mean(finite)) if len(finite) else float("nan")
            if metric not in PRIMARY_BOOTSTRAP_METRICS:
                continue
            if len(finite):
                sample_indices = rng.integers(0, len(finite), size=(int(bootstrap_iterations), len(finite)))
                draws = np.mean(finite[sample_indices], axis=1)
                item[f"{metric}_ci_low"] = float(np.quantile(draws, 0.025))
                item[f"{metric}_ci_high"] = float(np.quantile(draws, 0.975))
            else:
                draws = np.full(int(bootstrap_iterations), np.nan, dtype=np.float64)
                item[f"{metric}_ci_low"] = float("nan")
                item[f"{metric}_ci_high"] = float("nan")
            bootstrap_arrays.append(draws.astype(np.float32))
            bootstrap_keys.append("|".join((*key, metric)))
        summary_rows.append(item)
    draws_matrix = (
        np.stack(bootstrap_arrays)
        if bootstrap_arrays
        else np.empty((0, int(bootstrap_iterations)), dtype=np.float32)
    )
    return dependency_rows, subject_rows, summary_rows, draws_matrix, bootstrap_keys


def aggregate_timecourses(
    subject_curve_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    group_fields = (
        "task_id", "family", "stage", "dataset_id", "role", "model", "spatial_mode", "modality",
    )
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in subject_curve_rows:
        groups[tuple(str(row[field]) for field in group_fields)].append(row)
    output = []
    rng = np.random.default_rng(int(seed) + 1009)
    for key, rows in sorted(groups.items()):
        subjects = sorted({str(row["subject"]) for row in rows})
        times = sorted({float(row["time_s"]) for row in rows})
        subject_index = {value: index for index, value in enumerate(subjects)}
        time_index = {value: index for index, value in enumerate(times)}
        matrices = {
            field: np.full((len(subjects), len(times)), np.nan, dtype=np.float64)
            for field in ("observed", "reconstructed", "posterior_predictive_sd")
        }
        for row in rows:
            i = subject_index[str(row["subject"])]
            j = time_index[float(row["time_s"])]
            for field, matrix in matrices.items():
                matrix[i, j] = float(row[field])
        if any(np.any(~np.isfinite(matrix)) for matrix in matrices.values()):
            raise RuntimeError(f"time-course support drift for {'/'.join(key)}")
        weights = rng.multinomial(
            len(subjects),
            np.full(len(subjects), 1.0 / len(subjects)),
            size=int(bootstrap_iterations),
        ).astype(np.float64) / len(subjects)
        summaries = {}
        for field, matrix in matrices.items():
            draws = weights @ matrix
            summaries[field] = (
                np.mean(matrix, axis=0),
                np.quantile(draws, 0.025, axis=0),
                np.quantile(draws, 0.975, axis=0),
            )
        for time_position, time_value in enumerate(times):
            item: dict[str, Any] = dict(zip(group_fields, key, strict=True))
            item.update({"time_s": time_value, "subjects": len(subjects)})
            for field, (mean, low, high) in summaries.items():
                item[f"{field}_mean"] = float(mean[time_position])
                item[f"{field}_ci_low"] = float(low[time_position])
                item[f"{field}_ci_high"] = float(high[time_position])
            output.append(item)
    return output


TRAJECTORY_FIELDS = (
    "schema", "task_id", "family", "stage", "dataset_id", "role", "subject", "stratum",
    "dependency_group", "unit_id", "fold_index", "model", "spatial_mode",
    "observed_modalities", "target_modalities", "observation_role", "fit_source",
    "fit_parameter_hash", "time_s",
    "eeg_observed", "eeg_reconstructed", "eeg_predictive_std", "eeg_valid",
    "hbo_observed", "hbo_reconstructed", "hbo_predictive_std", "hbr_observed",
    "hbr_reconstructed", "hbr_predictive_std", "fnirs_valid",
    *STATE_NAMES,
    *(f"{name}_posterior_sd" for name in STATE_NAMES),
)


def _write_predictions(writer: csv.DictWriter, predictions: Sequence[Prediction]) -> int:
    count = 0
    for prediction in predictions:
        for index, time_value in enumerate(prediction.time_s):
            row: dict[str, Any] = {
                "schema": TRAJECTORY_SCHEMA,
                "task_id": prediction.task_id,
                "family": prediction.family,
                "stage": prediction.stage,
                "dataset_id": prediction.dataset_id,
                "role": prediction.role,
                "subject": prediction.subject,
                "stratum": prediction.stratum,
                "dependency_group": prediction.dependency_group,
                "unit_id": prediction.unit_id,
                "fold_index": prediction.fold_index,
                "model": prediction.model,
                "spatial_mode": prediction.spatial_mode,
                "observed_modalities": "|".join(prediction.observed_modalities),
                "target_modalities": "|".join(prediction.target_modalities),
                "observation_role": prediction.observation_role,
                "fit_source": prediction.fit_source,
                "fit_parameter_hash": prediction.fit_parameter_hash,
                "time_s": float(time_value),
                "eeg_observed": float(prediction.eeg_observation[index]),
                "eeg_reconstructed": float(prediction.eeg_reconstruction[index]),
                "eeg_predictive_std": float(prediction.predictive_std[index, 0]),
                "eeg_valid": bool(prediction.eeg_valid_mask[index]),
                "hbo_observed": float(prediction.truth_hbo[index]),
                "hbo_reconstructed": float(prediction.estimate_hbo[index]),
                "hbo_predictive_std": float(prediction.predictive_std[index, 1]),
                "hbr_observed": float(prediction.truth_hbr[index]),
                "hbr_reconstructed": float(prediction.estimate_hbr[index]),
                "hbr_predictive_std": float(prediction.predictive_std[index, 2]),
                "fnirs_valid": bool(prediction.fnirs_valid_mask[index]),
            }
            row.update(
                {name: float(prediction.states[index, state_index]) for state_index, name in enumerate(STATE_NAMES)}
            )
            row.update(
                {
                    f"{name}_posterior_sd": float(prediction.state_std[index, state_index])
                    for state_index, name in enumerate(STATE_NAMES)
                }
            )
            writer.writerow(row)
            count += 1
    return count


def _preferred_role(stage: str) -> str:
    return "development_validation" if stage == "core" else "descriptive"


def _task_label(task_id: str) -> str:
    replacements = {
        "single_ma": "MA",
        "single_lmi": "LMI",
        "single_rmi": "RMI",
        "simultaneous_wg": "WG",
        "simultaneous_0back": "0-back",
        "simultaneous_2back": "2-back",
        "simultaneous_3back": "3-back",
        "visual_rr": "Visual RR",
        "visual_rf": "Visual RF",
        "visual_fr": "Visual FR",
        "visual_ff": "Visual FF",
        "refed_video": "REFED",
        "simultaneous_dsr": "DSR block",
    }
    return replacements.get(task_id, task_id)


def _selected_summary_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    task_order: Sequence[str],
) -> list[Mapping[str, Any]]:
    admitted = set(task_order)
    return [
        row
        for row in summary_rows
        if row["task_id"] in admitted
        and row["role"] == _preferred_role(str(row["stage"]))
        and row["spatial_mode"] == "local"
    ]


def _figure_provenance(source_path: Path, source_role: str) -> dict[str, Any]:
    return {
        "source_table": str(source_path.relative_to(REPO_ROOT)),
        "source_table_sha256": _sha256(source_path),
        "source_role": source_role,
        "estimator": "equal-subject mean",
        "interval": "95% percentile interval from 10000 deterministic subject bootstrap draws",
        "analysis_intent": "exploratory descriptive reliability audit",
        "no_cross_task_pooling": True,
    }


def _save_figure(
    figure: plt.Figure,
    stem: Path,
    *,
    alt_text: str,
    source_path: Path,
    source_role: str,
) -> list[str]:
    artifacts = save_figure_atomic(
        figure,
        stem,
        formats=("svg", "png"),
        dpi=300,
        alt_text=alt_text,
        provenance=_figure_provenance(source_path, source_role),
    )
    plt.close(figure)
    paths = [*artifacts.figure_paths]
    if artifacts.manifest_path is not None:
        paths.append(artifacts.manifest_path)
    if artifacts.alt_text_path is not None:
        paths.append(artifacts.alt_text_path)
    return [str(path.relative_to(stem.parents[1])) for path in paths]


def plot_reliability_overview(
    subject_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    task_order: Sequence[str],
    run_dir: Path,
) -> list[str]:
    selected_summary = _selected_summary_rows(summary_rows, task_order)
    selected_subjects = [
        row
        for row in subject_rows
        if row["task_id"] in set(task_order)
        and row["role"] == _preferred_role(str(row["stage"]))
        and row["spatial_mode"] == "local"
    ]
    source_rows = [dict(row) for row in selected_subjects]
    source_path = run_dir / "figure_sources/task_reliability_overview.csv"
    _write_csv(source_path, source_rows)
    lookup = {
        (str(row["task_id"]), str(row["model"])): row for row in selected_summary
    }
    subject_lookup: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_subjects:
        subject_lookup[(str(row["task_id"]), str(row["model"]))].append(row)
    metrics = [
        ("hbo_trajectory_deviation_nrmse", "HbO trajectory NRMSE", None),
        ("hbr_trajectory_deviation_nrmse", "HbR trajectory NRMSE", None),
        ("eeg_trajectory_deviation_nrmse", "EEG-proxy trajectory NRMSE", None),
        ("hbo_predictive_95_coverage", "HbO empirical 95% coverage", 0.95),
        ("hbr_predictive_95_coverage", "HbR empirical 95% coverage", 0.95),
        ("eeg_predictive_95_coverage", "EEG-proxy empirical 95% coverage", 0.95),
    ]
    models = (
        ("adaptive_joint", "Joint smoother", "#D55E00", "o"),
        ("adaptive_eeg_only", "EEG-only smoother", "#0072B2", "s"),
        ("adaptive_fnirs_only", "fNIRS-only smoother", "#009E73", "^"),
    )
    with matplotlib.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(22, 10), layout="constrained", sharex=True)
        positions = np.arange(len(task_order), dtype=np.float64)
        for axis, (metric, label, reference) in zip(axes.flat, metrics, strict=True):
            for model_index, (model, model_label, color, marker) in enumerate(models):
                offset = (-0.18, 0.0, 0.18)[model_index]
                for task_index, task_id in enumerate(task_order):
                    row = lookup.get((task_id, model))
                    if row is None:
                        continue
                    values = [
                        float(value[metric])
                        for value in subject_lookup[(task_id, model)]
                        if np.isfinite(float(value[metric]))
                    ]
                    jitter = np.linspace(-0.045, 0.045, len(values)) if values else np.asarray([])
                    axis.scatter(
                        task_index + offset + jitter,
                        values,
                        s=14,
                        marker=marker,
                        facecolors="none",
                        edgecolors=color,
                        alpha=0.45,
                        linewidths=0.7,
                    )
                    mean = float(row[metric])
                    low = float(row[f"{metric}_ci_low"])
                    high = float(row[f"{metric}_ci_high"])
                    axis.errorbar(
                        task_index + offset,
                        mean,
                        yerr=[[mean - low], [high - mean]],
                        fmt=marker,
                        color=color,
                        markersize=5,
                        capsize=2,
                        linewidth=1.2,
                        label=model_label if task_index == 0 else None,
                    )
            if reference is not None:
                axis.axhline(reference, color="#555555", linestyle="--", linewidth=0.9, label="Nominal 0.95")
            axis.axvline(6.5, color="#999999", linestyle=":", linewidth=0.9)
            axis.set_title(label)
            axis.set_xticks(positions, [_task_label(value) for value in task_order], rotation=55, ha="right")
            axis.grid(axis="y", alpha=0.2)
        axes[0, 0].legend(fontsize=8, ncol=2)
        axes[1, 0].legend(fontsize=8, ncol=3)
        fig.suptitle(
            "SSM reconstruction reliability by task (development-validation core; descriptive annex)",
            fontsize=14,
        )
    return _save_figure(
        fig,
        run_dir / "figures/task_reliability_overview",
        alt_text=(
            "Six-panel task profile. The top row shows subject-level and equal-subject mean HbO, HbR, "
            "and EEG-proxy trajectory NRMSE with 95% subject-bootstrap intervals. The bottom row shows "
            "empirical posterior predictive coverage with a nominal 0.95 reference. Orange circles are "
            "joint smoothing, blue squares are EEG-only smoothing, and green triangles are fNIRS-only smoothing. A dotted divider separates the seven "
            "core task cells from the descriptive Visual, REFED, and DSR cells; no cross-task average is shown."
        ),
        source_path=source_path,
        source_role="subject-level reconstruction metrics used with task_summary bootstrap intervals",
    )


def plot_spread_calibration(
    subject_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    task_order: Sequence[str],
    run_dir: Path,
) -> list[str]:
    selected_summary = _selected_summary_rows(summary_rows, task_order)
    source_path = run_dir / "figure_sources/task_spread_calibration.csv"
    _write_csv(source_path, [dict(row) for row in selected_summary])
    lookup = {(str(row["task_id"]), str(row["model"])): row for row in selected_summary}
    metrics = [
        ("hbo_temporal_sd_ratio", "HbO reconstructed/observed temporal SD", 1.0),
        ("hbr_temporal_sd_ratio", "HbR reconstructed/observed temporal SD", 1.0),
        ("eeg_temporal_sd_ratio", "EEG-proxy reconstructed/observed temporal SD", 1.0),
        ("hbo_standardized_residual_rms", "HbO standardized residual RMS", 1.0),
        ("hbr_standardized_residual_rms", "HbR standardized residual RMS", 1.0),
        ("eeg_standardized_residual_rms", "EEG-proxy standardized residual RMS", 1.0),
    ]
    models = (
        ("adaptive_joint", "Joint smoother", "#D55E00", "o"),
        ("adaptive_eeg_only", "EEG-only smoother", "#0072B2", "s"),
        ("adaptive_fnirs_only", "fNIRS-only smoother", "#009E73", "^"),
    )
    with matplotlib.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(22, 10), layout="constrained", sharex=True)
        x = np.arange(len(task_order), dtype=np.float64)
        for axis, (metric, label, reference) in zip(axes.flat, metrics, strict=True):
            for model_index, (model, model_label, color, marker) in enumerate(models):
                offset = (-0.18, 0.0, 0.18)[model_index]
                values = [
                    float(lookup[(task_id, model)][metric])
                    if (task_id, model) in lookup
                    else float("nan")
                    for task_id in task_order
                ]
                axis.plot(
                    x + offset,
                    values,
                    color=color,
                    marker=marker,
                    linestyle="none",
                    label=model_label,
                )
            axis.axhline(reference, color="#555555", linestyle="--", linewidth=0.9)
            axis.axvline(6.5, color="#999999", linestyle=":", linewidth=0.9)
            axis.set_title(label)
            axis.set_xticks(x, [_task_label(value) for value in task_order], rotation=55, ha="right")
            axis.grid(axis="y", alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Temporal spread and posterior calibration diagnostics by task", fontsize=14)
    return _save_figure(
        fig,
        run_dir / "figures/task_spread_calibration",
        alt_text=(
            "Six-panel task profile of temporal spread ratios and standardized residual RMS. Values of one "
            "match observed temporal spread or predictive standardization. Orange circles show the joint "
            "smoother, blue squares the EEG-only smoother, and green triangles the fNIRS-only smoother. Core and descriptive task cells are separated "
            "and are not pooled."
        ),
        source_path=source_path,
        source_role="equal-subject task summary",
    )


def plot_timecourse_families(
    timecourse_rows: Sequence[Mapping[str, Any]],
    task_order: Sequence[str],
    run_dir: Path,
) -> list[str]:
    families = {
        "single_trial_core": [value for value in task_order if value.startswith("single_")],
        "simultaneous_core": [
            value for value in task_order
            if value.startswith("simultaneous_") and value != "simultaneous_dsr"
        ],
        "visual": [value for value in task_order if value.startswith("visual_")],
        "refed": ["refed_video"],
        "dsr": ["simultaneous_dsr"],
    }
    artifacts: list[str] = []
    colors = {
        "adaptive_joint": "#D55E00",
        "adaptive_eeg_only": "#0072B2",
        "adaptive_fnirs_only": "#009E73",
    }
    labels = {
        "adaptive_joint": "Joint reconstruction",
        "adaptive_eeg_only": "EEG-only reconstruction",
        "adaptive_fnirs_only": "fNIRS-only reconstruction",
    }
    for family_name, tasks in families.items():
        subset = [
            row
            for row in timecourse_rows
            if row["task_id"] in set(tasks)
            and row["role"] == _preferred_role(str(row["stage"]))
            and row["spatial_mode"] == "local"
            and row["modality"] in {"hbo", "hbr"}
        ]
        source_path = run_dir / f"figure_sources/timecourse_{family_name}.csv"
        _write_csv(source_path, [dict(row) for row in subset])
        lookup: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in subset:
            lookup[(str(row["task_id"]), str(row["modality"]), str(row["model"]))].append(row)
        with matplotlib.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
            fig, axes = plt.subplots(
                len(tasks),
                2,
                figsize=(14, max(4.5, 3.2 * len(tasks))),
                layout="constrained",
                squeeze=False,
                sharex="row",
            )
            for row_index, task_id in enumerate(tasks):
                for column, modality in enumerate(("hbo", "hbr")):
                    axis = axes[row_index, column]
                    observed_rows = sorted(
                        lookup[(task_id, modality, "adaptive_joint")],
                        key=lambda value: float(value["time_s"]),
                    )
                    if not observed_rows:
                        axis.text(
                            0.5,
                            0.5,
                            "No completed cell",
                            ha="center",
                            va="center",
                            transform=axis.transAxes,
                            color="#6F6F6F",
                        )
                        axis.set_title(f"{_task_label(task_id)} — {modality.upper()}")
                        continue
                    time = np.asarray([float(value["time_s"]) for value in observed_rows])
                    observed = np.asarray([float(value["observed_mean"]) for value in observed_rows])
                    observed_low = np.asarray([float(value["observed_ci_low"]) for value in observed_rows])
                    observed_high = np.asarray([float(value["observed_ci_high"]) for value in observed_rows])
                    axis.plot(time, observed, color="#222222", linewidth=1.5, label="Observed")
                    axis.fill_between(time, observed_low, observed_high, color="#777777", alpha=0.14)
                    for model in (
                        "adaptive_joint",
                        "adaptive_eeg_only",
                        "adaptive_fnirs_only",
                    ):
                        values = sorted(
                            lookup[(task_id, modality, model)],
                            key=lambda value: float(value["time_s"]),
                        )
                        reconstruction = np.asarray([float(value["reconstructed_mean"]) for value in values])
                        low = np.asarray([float(value["reconstructed_ci_low"]) for value in values])
                        high = np.asarray([float(value["reconstructed_ci_high"]) for value in values])
                        linestyle = {
                            "adaptive_joint": "-",
                            "adaptive_eeg_only": "--",
                            "adaptive_fnirs_only": "-.",
                        }[model]
                        axis.plot(time, reconstruction, color=colors[model], linestyle=linestyle, label=labels[model])
                        axis.fill_between(time, low, high, color=colors[model], alpha=0.10)
                    if float(time[0]) < 0.0:
                        axis.axvline(0.0, color="#666666", linestyle=":", linewidth=0.8)
                    axis.axhline(0.0, color="#AAAAAA", linewidth=0.6)
                    axis.set_title(f"{_task_label(task_id)} — {modality.upper()}")
                    axis.set_xlabel("Task-specific relative time (s)")
                    axis.set_ylabel("Canonical robust-SD coordinate")
                    axis.grid(alpha=0.18)
            axes[0, 0].legend(fontsize=8, ncol=3)
            fig.suptitle(
                f"Observed and reconstructed group trajectories: {family_name.replace('_', ' ')}",
                fontsize=14,
            )
        artifacts.extend(
            _save_figure(
                fig,
                run_dir / f"figures/timecourse_{family_name}",
                alt_text=(
                    f"Task-specific HbO and HbR group time courses for {family_name.replace('_', ' ')}. "
                    "Black lines are equal-subject observed means, orange solid lines are joint reconstructions, "
                    "blue dashed lines are EEG-only reconstructions, and green dash-dot lines are fNIRS-only reconstructions; shaded regions are 95% subject-bootstrap "
                    "intervals. Each row is a separate task estimand."
                ),
                source_path=source_path,
                source_role="time-resolved equal-subject means and subject-bootstrap intervals",
            )
        )
    return artifacts


def build_jobs(config: Mapping[str, Any], *, smoke: bool) -> list[Job]:
    data = config["data"]
    jobs: list[Job] = []
    for task_value in data["tasks"]:
        task = dict(task_value)
        if task["stage"] == "core":
            subjects = _core_subjects(task, data)
            if smoke:
                subjects = [next(value for value in subjects if value[1] == "development_validation")]
        elif task["dataset_id"] == "visual_cognitive_motivation":
            subjects = [(f"S{index:02d}", "descriptive") for index in range(1, 17)]
            if smoke:
                subjects = subjects[:1]
        elif task["dataset_id"] == "refed":
            subjects = [(str(index), "descriptive") for index in range(1, 33)]
            if smoke:
                subjects = subjects[:1]
        elif task["task_id"] == "simultaneous_dsr":
            excluded = set(str(value) for value in data["dsr_excluded_subjects"])
            subjects = [
                (f"VP{index:03d}", "descriptive")
                for index in range(1, 24)
                if f"VP{index:03d}" not in excluded
            ]
            if smoke:
                subjects = subjects[:1]
        else:
            raise ValueError(f"cannot construct jobs for {task['task_id']}")
        for subject, role in subjects:
            jobs.append(
                Job(
                    task=task,
                    subject=subject,
                    role=role,
                    config=dict(config),
                    smoke_max_units=6 if smoke else None,
                )
            )
    protected = {
        str(value)
        for values in data["protected_or_unused"].values()
        for value in values
    }
    overlap = sorted({job.subject for job in jobs}.intersection(protected))
    if overlap:
        raise PermissionError(f"job matrix opens protected/unused subjects: {overlap}")
    return jobs


def _summary_markdown(
    summary_rows: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
    task_order: Sequence[str],
    *,
    failure_count: int,
) -> str:
    selected = _selected_summary_rows(summary_rows, task_order)
    lookup = {(str(row["task_id"]), str(row["model"])): row for row in selected}
    lines = [
        "# SSM reconstruction reliability: full development audit",
        "",
        "This is an exploratory reliability profile, not an independent confirmation of a shared physiological source. "
        "Core rows use development-validation subjects 19–23; Visual, REFED, and DSR rows are task-specific descriptive annexes. "
        "No protected Single-Trial subject (24–29) or unused Simultaneous subject (VP024–VP026) was opened.",
        "",
        "The EEG observation is the train-fold log-power PCA proxy rather than the raw high-rate EEG waveform. "
        "Joint smoothing conditions on the held-out EEG, HbO, and HbR observations, so its coverage is a posterior fit diagnostic. "
        "EEG-only to fNIRS and fNIRS-only to the EEG proxy are the two directional out-of-modality checks; "
        "fNIRS-only HbO/HbR reconstruction itself remains a within-modality self-observation diagnostic.",
        "",
        "## Task-level reliability profile",
        "",
        "Values are equal-subject means with 95% percentile subject-bootstrap intervals in `task_summary.csv`. "
        "Task rows are kept separate; there is no cross-task scalar.",
        "",
        "| Task | Model | Subjects | HbO NRMSE | HbO temporal SD ratio | HbO std. residual RMS | HbO 95% coverage | HbR NRMSE | EEG-proxy NRMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task_id in task_order:
        for model in (
            "adaptive_joint",
            "adaptive_eeg_only",
            "adaptive_fnirs_only",
        ):
            row = lookup.get((task_id, model))
            if row is None:
                lines.append(f"| {_task_label(task_id)} | {model} | — | failed/missing | — | — | — | — | — |")
                continue
            lines.append(
                f"| {_task_label(task_id)} | {model.replace('adaptive_', '')} | {int(row['subjects'])} | "
                f"{float(row['hbo_trajectory_deviation_nrmse']):.3f} "
                f"[{float(row['hbo_trajectory_deviation_nrmse_ci_low']):.3f}, {float(row['hbo_trajectory_deviation_nrmse_ci_high']):.3f}] | "
                f"{float(row['hbo_temporal_sd_ratio']):.3f} | "
                f"{float(row['hbo_standardized_residual_rms']):.3f} | "
                f"{float(row['hbo_predictive_95_coverage']):.3f} | "
                f"{float(row['hbr_trajectory_deviation_nrmse']):.3f} | "
                f"{float(row['eeg_trajectory_deviation_nrmse']):.3f} |"
            )
    eeg_only = [row for row in selected if row["model"] == "adaptive_eeg_only"]
    ranked = sorted(eeg_only, key=lambda row: float(row["hbo_trajectory_deviation_nrmse"]))
    lines.extend(
        [
            "",
            "## Reading the diagnostics",
            "",
            "- Trajectory NRMSE quantifies observed–reconstructed deviation after normalizing by the observed within-window temporal SD; lower is closer.",
            "- Temporal SD ratio compares reconstructed and observed within-window variation; one indicates matched spread, not matched timing.",
            "- Posterior predictive SD and 95% coverage quantify model uncertainty separately from reconstruction deviation. Standardized residual RMS near one is the corresponding scale check.",
            "- Subject dots in the overview figure are the replication units. Windows are first averaged within trial/probe-pair/video/block, then within subject.",
        ]
    )
    if ranked:
        lines.extend(
            [
                "",
                "## Task heterogeneity",
                "",
                f"Within the EEG-only HbO task profile, the smallest mean NRMSE occurred for {_task_label(str(ranked[0]['task_id']))} "
                f"({float(ranked[0]['hbo_trajectory_deviation_nrmse']):.3f}) and the largest for {_task_label(str(ranked[-1]['task_id']))} "
                f"({float(ranked[-1]['hbo_trajectory_deviation_nrmse']):.3f}). This is a descriptive ranking across non-identical estimands, not a pooled test.",
            ]
        )
    status_counts = Counter(str(row.get("status", "completed")) for row in inventory_rows)
    lines.extend(
        [
            "",
            "## Completeness and evidence",
            "",
            f"- Subject/task cells: {len(inventory_rows)} total; {status_counts.get('completed', 0)} completed; {failure_count} explicit failures.",
            "- `window_metrics.csv` is the window-level evidence; `dependency_metrics.csv`, `subject_metrics.csv`, and `task_summary.csv` expose each aggregation transition.",
            "- `timecourse_summary.csv` and the five task-family figures retain task-specific time origins and between-subject bootstrap intervals.",
            "- `bootstrap_draws.npz` retains the 10,000 subject-bootstrap draws for the twelve primary deviation/spread/calibration metrics.",
            "- `trajectories.csv.gz` retains observed/reconstructed trajectories, posterior predictive SD, latent states, and posterior state SD.",
            "- Figure source tables, SVG/PNG exports, alt text, and per-figure provenance manifests are under `figure_sources/` and `figures/`.",
            "",
            "## Claim boundary",
            "",
            "The audit measures reconstruction behavior of this fitted SSM family. Good joint fit can arise because fNIRS enters the smoother; poor EEG-only fit weakens an EEG-to-hemodynamics claim. "
            "Neither good fit nor calibrated intervals identify a unique neural state, demonstrate shared/private latent structure, or authorize a VQ bottleneck.",
            "",
        ]
    )
    return "\n".join(lines)


def _input_sources(config_path: Path, cache_root: Path) -> list[Path]:
    candidates = [
        config_path,
        Path(__file__),
        REPO_ROOT / "docs/analysis/SSM_RECONSTRUCTION_RELIABILITY_PLAN.md",
        REPO_ROOT / "experiments/evaluate_adaptive_shared_neural_ssm.py",
        REPO_ROOT / "experiments/evaluate_shared_neural_driver_unified.py",
        REPO_ROOT / "src/inference/adaptive_neurovascular_ssm.py",
        REPO_ROOT / "src/metrics/trajectory_reliability.py",
        REPO_ROOT / "src/data/unified_physiology.py",
        cache_root / "cache_manifest.json",
        cache_root / "event_index/event_manifest.json",
        cache_root / "channel_geometry/geometry_manifest.json",
    ]
    missing = [str(path) for path in candidates if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing provenance source(s): {missing}")
    return candidates


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.smoke:
        config["analysis"]["folds"] = 2
        config["analysis"]["workers"] = min(4, int(config["analysis"]["workers"]))
        config["analysis"]["bootstrap_iterations"] = 100
        config["analysis"]["ssm"]["max_iterations"] = 6
        config["analysis"]["ssm"]["q_scale_candidates"] = [1.0]
        config["analysis"]["ssm"]["fnirs_noise_scale_candidates"] = [1.0]
    run_name = (
        "20260819_ssm_reconstruction_reliability_smoke_v1"
        if args.smoke
        else "20260819_ssm_reconstruction_reliability_full_v1"
    )
    run_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["root"]) / run_name
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.staging-", dir=run_dir.parent))
    try:
        (staging / "figures").mkdir()
        (staging / "figure_sources").mkdir()
        (staging / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        jobs = build_jobs(config, smoke=bool(args.smoke))
        window_rows: list[dict[str, Any]] = []
        fit_rows: list[dict[str, Any]] = []
        inventory_rows: list[dict[str, Any]] = []
        subject_curve_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        trajectory_row_count = 0
        trajectory_path = staging / "trajectories.csv.gz"
        with gzip.open(trajectory_path, "wt", newline="", encoding="utf-8", compresslevel=6) as handle:
            trajectory_writer = csv.DictWriter(handle, fieldnames=list(TRAJECTORY_FIELDS))
            trajectory_writer.writeheader()
            workers = int(config["analysis"]["workers"])
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    results = executor.map(_execute_job_safe, jobs, chunksize=1)
                    for job_index, (job, result) in enumerate(zip(jobs, results, strict=True), start=1):
                        if result.failure is not None:
                            failures.append({**result.inventory, "traceback": result.failure})
                            inventory_rows.append(result.inventory)
                        else:
                            result.inventory["status"] = "completed"
                            inventory_rows.append(result.inventory)
                            fit_rows.extend(result.fit_rows)
                            subject_curve_rows.extend(result.subject_curve_rows)
                            window_rows.extend(prediction_metrics(value) for value in result.predictions)
                            trajectory_row_count += _write_predictions(trajectory_writer, result.predictions)
                        print(
                            f"[{job_index}/{len(jobs)}] {job.task['task_id']} {job.subject}: "
                            f"{'failed' if result.failure else 'completed'}",
                            flush=True,
                        )
            else:
                for job_index, job in enumerate(jobs, start=1):
                    result = _execute_job_safe(job)
                    if result.failure is not None:
                        failures.append({**result.inventory, "traceback": result.failure})
                        inventory_rows.append(result.inventory)
                    else:
                        result.inventory["status"] = "completed"
                        inventory_rows.append(result.inventory)
                        fit_rows.extend(result.fit_rows)
                        subject_curve_rows.extend(result.subject_curve_rows)
                        window_rows.extend(prediction_metrics(value) for value in result.predictions)
                        trajectory_row_count += _write_predictions(trajectory_writer, result.predictions)
                    print(
                        f"[{job_index}/{len(jobs)}] {job.task['task_id']} {job.subject}: "
                        f"{'failed' if result.failure else 'completed'}",
                        flush=True,
                    )
        if not window_rows:
            raise RuntimeError("all subject/task cells failed; no reliability metrics were produced")
        dependency_rows, subject_rows, summary_rows, bootstrap_draws, bootstrap_keys = aggregate_metrics(
            window_rows,
            bootstrap_iterations=int(config["analysis"]["bootstrap_iterations"]),
            seed=int(config["analysis"]["seed"]),
        )
        timecourse_rows = aggregate_timecourses(
            subject_curve_rows,
            bootstrap_iterations=int(config["analysis"]["bootstrap_iterations"]),
            seed=int(config["analysis"]["seed"]),
        )
        _write_csv(staging / "window_metrics.csv", window_rows)
        _write_csv(staging / "dependency_metrics.csv", dependency_rows)
        _write_csv(staging / "subject_metrics.csv", subject_rows)
        _write_csv(staging / "task_summary.csv", summary_rows)
        _write_csv(staging / "subject_timecourses.csv", subject_curve_rows)
        _write_csv(staging / "timecourse_summary.csv", timecourse_rows)
        _write_csv(staging / "fit_parameters.csv", fit_rows)
        _write_csv(staging / "split_inventory.csv", inventory_rows)
        _write_json(staging / "failure_records.json", {"failures": failures})
        np.savez_compressed(
            staging / "bootstrap_draws.npz",
            schema=np.asarray("ssm_reliability_subject_bootstrap_v1"),
            draws=bootstrap_draws,
            keys=np.asarray(bootstrap_keys, dtype=str),
            iterations=np.asarray(int(config["analysis"]["bootstrap_iterations"])),
            seed=np.asarray(int(config["analysis"]["seed"])),
        )
        task_order = [str(value["task_id"]) for value in config["data"]["tasks"]]
        figure_artifacts = []
        figure_artifacts.extend(plot_reliability_overview(subject_rows, summary_rows, task_order, staging))
        figure_artifacts.extend(plot_spread_calibration(subject_rows, summary_rows, task_order, staging))
        figure_artifacts.extend(plot_timecourse_families(timecourse_rows, task_order, staging))
        (staging / "summary.md").write_text(
            _summary_markdown(
                summary_rows,
                inventory_rows,
                task_order,
                failure_count=len(failures),
            ),
            encoding="utf-8",
        )
        cache_root = Path(config["data"]["cache_root"])
        if not cache_root.is_absolute():
            cache_root = REPO_ROOT / cache_root
        sources = _input_sources(config_path, cache_root)
        artifact_paths = sorted(
            path for path in staging.rglob("*") if path.is_file() and path.name != "manifest.json"
        )
        manifest = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "smoke" if args.smoke else "full_development_exploratory",
            "protected_open": False,
            "protected_or_unused_subjects": config["data"]["protected_or_unused"],
            "git": _git_payload(),
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "input_hashes": [
                {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)} for path in sources
            ],
            "cell_count": len(jobs),
            "completed_cell_count": len(jobs) - len(failures),
            "failed_cell_count": len(failures),
            "window_metric_row_count": len(window_rows),
            "trajectory_row_count": trajectory_row_count,
            "reliability_contract": {
                "primary_deviation_metric": "trajectory_deviation_nrmse",
                "aggregation": "window_to_dependency_group_to_subject_equal_to_task_cell",
                "bootstrap_unit": "subject",
                "bootstrap_iterations": int(config["analysis"]["bootstrap_iterations"]),
                "predictive_interval": "normal_95_percent_z_1.959963984540054",
                "combined_cross_task_scalar": False,
                "teacher_input_masks": "model_specific_observed_modalities",
                "scoring_masks": "target_modality_specific",
                "partial_missingness": "passed_as_NaN_to_missing_aware_RTS_smoother",
                "fit_window_support": "fully_observed_training_windows_only",
            },
            "task_time_origins": {
                str(task["task_id"]): str(task["time_origin"]) for task in config["data"]["tasks"]
            },
            "observation_modes": _jsonable(MODEL_OBSERVATION_CONTRACT),
            "fit_provenance": {
                "source": "within_subject_crossfit_training_dependency_groups",
                "parameter_hash_field": "fit_parameter_hash",
                "parameter_table": "fit_parameters.csv",
            },
            "figure_artifacts": figure_artifacts,
            "artifacts": [
                {
                    "path": str(path.relative_to(staging)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in artifact_paths
            ],
            "claim_boundary": [
                "joint smoothing is a posterior fit diagnostic because held-out fNIRS enters smoothing",
                "fNIRS-only smoothing is a within-modality self-observation diagnostic, not cross-modal evidence",
                "EEG-only smoothing is the out-of-modality fNIRS reconstruction check",
                "EEG observation is a train-fold log-power PCA proxy, not the raw waveform",
                "task-specific descriptive annexes are not pooled with the core matrix",
                "reconstruction reliability does not identify a unique shared neural state",
            ],
        }
        _write_json(staging / "manifest.json", manifest)
        os.rename(staging, run_dir)
    except Exception:
        if staging.exists() and staging.parent == run_dir.parent and staging.name.startswith(f".{run_dir.name}.staging-"):
            shutil.rmtree(staging)
        raise
    print(run_dir, flush=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/physiology_semantic_tokenizer/ssm_reconstruction_reliability.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
