#!/usr/bin/env python3
"""Fit-selection-only architecture optimization for the LC-SPVQ M1 model.

This runner is intentionally separate from :mod:`run_lag_conditioned_spvq`.
The reviewed runner remains fail-closed for full mode; this executable imports
its reviewed preparation, model, objective, and evaluation helpers while owning
the architecture-search contract and the post-selection development boundary.

The executable is exploratory, single-seed, and provisional.  Development data
is not prepared, loaded, or evaluated until one global candidate has been
selected from fit-selection metrics for both registered tasks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from copy import deepcopy
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cmp_to_key
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep the reviewed runner's full-mode guards untouched.  All data/model/loss/
# evaluation primitives used below are imported from that reviewed module.
from experiments import run_lag_conditioned_spvq as reviewed


REPO_ROOT = reviewed.REPO_ROOT
REGISTERED_CONFIG_PATH = (
    REPO_ROOT
    / "experiments/configs/physiology_semantic_tokenizer/"
    / "lag_conditioned_spvq_architecture_optimization.yaml"
).resolve()
REGISTERED_BASE_CONFIG_PATH = (
    REPO_ROOT
    / "experiments/configs/physiology_semantic_tokenizer/lag_conditioned_spvq.yaml"
).resolve()
REGISTERED_PROTOCOL_PATH = (
    REPO_ROOT / "docs/analysis/LC_SPVQ_ARCHITECTURE_OPTIMIZATION_PROTOCOL.md"
).resolve()
REGISTERED_PROTOCOL_SHA256 = (
    "8f1dff3510936ff136ac5df2dc1e0ebd625554b4b065df980a5844fbdf6fc7a8"
)
REGISTERED_OPTIMIZATION_SHA256 = (
    "d6d65fb9e6ad35065582e50c0e30668e765a9dc64e4d72ccca0def388822fab9"
)
REGISTERED_BASE_SHA256 = (
    "a5f5953aa3102407cef28291ebf8e5509190fd9f88765f672db423f0bf7363f5"
)

# The SHA above is the immutable anchor. Loading the exact registered file once
# avoids maintaining a second, drift-prone copy of its 181-line contract in
# executable source while still rejecting every in-memory or on-disk change.
_REGISTERED_CONTRACT: Mapping[str, Any] = yaml.safe_load(
    REGISTERED_CONFIG_PATH.read_text(encoding="utf-8")
)
TASKS = tuple(map(str, _REGISTERED_CONTRACT["execution"]["tasks"]))
VARIANT = str(_REGISTERED_CONTRACT["execution"]["variant"])
SAMPLES_PER_SUBJECT_CLASS = int(
    _REGISTERED_CONTRACT["sample_budget"]["samples_per_subject_class"]
)
CANDIDATE_CONSTANTS: Mapping[str, Any] = deepcopy(
    _REGISTERED_CONTRACT["candidate_constants"]
)
CANDIDATES: tuple[Mapping[str, Any], ...] = tuple(
    deepcopy(_REGISTERED_CONTRACT["candidates"])
)
CANDIDATE_IDS = tuple(str(row["candidate_id"]) for row in CANDIDATES)
REFERENCE_CANDIDATE_ID = str(
    _REGISTERED_CONTRACT["checkpoint_selection"]["reference_candidate_id"]
)
LONG_CONTROL_CANDIDATE_ID = str(
    _REGISTERED_CONTRACT["checkpoint_selection"]["long_control_candidate_id"]
)
REGISTERED_SCHEDULE: Mapping[str, Any] = deepcopy(_REGISTERED_CONTRACT["schedule"])
REGISTERED_SAMPLE_BUDGET: Mapping[str, Any] = deepcopy(
    _REGISTERED_CONTRACT["sample_budget"]
)
REGISTERED_CHECKPOINT_SELECTION: Mapping[str, Any] = deepcopy(
    _REGISTERED_CONTRACT["checkpoint_selection"]
)


_DEVELOPMENT_PERMIT_ISSUER = object()


@dataclass(init=False)
class _DevelopmentSelectionPermit:
    global_selection_digest: str
    decision_path: Path
    consumed_tasks: set[str]
    applied_models: set[tuple[str, str]]

    def __init__(
        self,
        global_selection_digest: str,
        decision_path: Path,
        *,
        issuer: object | None = None,
    ) -> None:
        if issuer is not _DEVELOPMENT_PERMIT_ISSUER:
            raise PermissionError("development permit can only be issued by orchestration")
        self.global_selection_digest = str(global_selection_digest)
        self.decision_path = Path(decision_path).resolve()
        self.consumed_tasks = set()
        self.applied_models = set()
        self.verify_decision()

    def verify_decision(self) -> Mapping[str, Any]:
        if self.decision_path.name != "global_candidate_selection.json":
            raise PermissionError("development permit is not bound to the decision artifact")
        if not self.decision_path.is_file():
            raise PermissionError("global selection decision artifact is missing")
        payload = json.loads(self.decision_path.read_text(encoding="utf-8"))
        observed = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if observed != self.global_selection_digest:
            raise PermissionError("global selection decision artifact changed")
        if (
            payload.get("schema") != "lc_spvq_global_candidate_selection_v1"
            or payload.get("selected_candidate_id") not in CANDIDATE_IDS
            or payload.get("reference_candidate_id") != REFERENCE_CANDIDATE_ID
            or payload.get("development_values_used") is not False
        ):
            raise PermissionError("global selection decision artifact is malformed")
        return payload

    def consume(self, task_id: str, decision_digest: str) -> None:
        self.verify_decision()
        if decision_digest != self.global_selection_digest:
            raise PermissionError("development decision digest does not match permit")
        if task_id not in TASKS:
            raise PermissionError("development permit received an unregistered task")
        if task_id in self.consumed_tasks:
            raise PermissionError(
                f"development partition for {task_id} was already materialized"
            )
        self.consumed_tasks.add(task_id)

    def consume_application(
        self,
        task_id: str,
        candidate_id: str,
        decision_digest: str,
    ) -> None:
        payload = self.verify_decision()
        if decision_digest != self.global_selection_digest:
            raise PermissionError("development application decision digest drifted")
        allowed = {
            str(payload["reference_candidate_id"]),
            str(payload["selected_candidate_id"]),
        }
        key = (str(task_id), str(candidate_id))
        if task_id not in self.consumed_tasks or candidate_id not in allowed:
            raise PermissionError("development application is not decision-authorized")
        if key in self.applied_models:
            raise PermissionError(
                f"development model {task_id}/{candidate_id} was already applied"
            )
        self.applied_models.add(key)


# ---------------------------------------------------------------------------
# Small serialization, hashing, and atomic publication helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    return reviewed._jsonable(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    fields: list[str] = []
    for row in values:
        for key in row:
            key = str(key)
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {key: _jsonable(row.get(key, "")) for key in fields}
                for row in values
            )
        else:
            handle.write("\n")
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    if temporary.exists():
        temporary.unlink()
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _issue_development_selection_permit(
    selection: Mapping[str, Any],
    decision_path: Path,
) -> _DevelopmentSelectionPermit:
    path = Path(decision_path).resolve()
    if not path.is_file():
        raise PermissionError("cannot issue permit before global decision publication")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if _canonical_json(loaded) != _canonical_json(selection):
        raise PermissionError("published global decision differs from selector output")
    digest = hashlib.sha256(_canonical_json(selection).encode("utf-8")).hexdigest()
    return _DevelopmentSelectionPermit(
        digest,
        path,
        issuer=_DEVELOPMENT_PERMIT_ISSUER,
    )


def _git_payload() -> dict[str, str]:
    def call(*args: str) -> str:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "status_short": call("git", "status", "--short"),
        "branch": call("git", "branch", "--show-current"),
    }


def _resolve(path: str | Path) -> Path:
    return reviewed._resolve(path)


def _path_is_repository_local(path: Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise PermissionError(f"{label} must be repository-local") from exc
    return resolved


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path == root / "manifest.json":
            continue
        output.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return output


# ---------------------------------------------------------------------------
# Exact optimization/base configuration binding and validation
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"{label} drifted from the registered optimization contract")


def _require_integral(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integral budget")
    result = int(value)
    if result < int(minimum):
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _validate_budgets(config: Mapping[str, Any]) -> None:
    budget = _require_mapping(config.get("sample_budget"), "sample_budget")
    for key in (
        "fit_parameter_subjects_per_dataset",
        "fit_selection_subjects_per_dataset",
        "development_apply_subjects_per_dataset",
        "samples_per_subject_class",
    ):
        _require_integral(budget.get(key), f"sample_budget.{key}")
    if budget.get("development_materialization") != (
        "after_global_candidate_selection_only"
    ):
        raise PermissionError("development materialization must be post-selection only")
    _require_exact(budget, REGISTERED_SAMPLE_BUDGET, "sample_budget")


def validate_optimization_config(
    config: Mapping[str, Any],
    *,
    config_path: Path | None = None,
    base_config_path: Path | None = None,
    base_config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate the immutable optimization YAML and its SHA-bound base YAML.

    The structural checks deliberately duplicate the small optimization contract
    instead of accepting arbitrary values from the YAML.  This catches semantic
    drift even when a caller passes an in-memory object rather than loading the
    registered file.
    """

    root = _require_mapping(config, "optimization config")
    required_top = {
        "experiment",
        "base_config",
        "execution",
        "sample_budget",
        "schedule",
        "candidate_constants",
        "candidates",
        "checkpoint_selection",
        "frozen_apply",
        "output",
    }
    if set(root) != required_top:
        raise ValueError("optimization config top-level keys drifted")

    experiment = _require_mapping(root["experiment"], "experiment")
    _require_exact(experiment, _REGISTERED_CONTRACT["experiment"], "experiment")
    if experiment.get("protected_open") is not False:
        raise PermissionError("optimization requires protected_open=false")

    bound = _require_mapping(root["base_config"], "base_config")
    _require_exact(
        bound,
        {
            "path": "experiments/configs/physiology_semantic_tokenizer/lag_conditioned_spvq.yaml",
            "sha256": REGISTERED_BASE_SHA256,
        },
        "base_config binding",
    )

    execution = _require_mapping(root["execution"], "execution")
    _require_exact(execution, _REGISTERED_CONTRACT["execution"], "execution")
    _require_integral(execution.get("seed"), "execution.seed")
    if execution.get("multi_seed_repetition") is not False:
        raise ValueError("multi-seed repetition is prohibited")
    if tuple(execution.get("tasks", ())) != TASKS or execution.get("variant") != VARIANT:
        raise ValueError("task/variant contract drifted")

    _validate_budgets(root)
    schedule = _require_mapping(root["schedule"], "schedule")
    integral_schedule_keys = {
        "batch_size",
        "continuous_pretrain_optimizer_steps",
        "vq_optimizer_steps",
        "task_head_optimizer_steps",
        "selection_evaluation_interval_steps",
        "early_stopping_patience_evaluations",
        "pretrain_minimum_steps",
        "vq_minimum_steps",
        "head_minimum_steps",
        "warmup_optimizer_steps",
        "long_control_step_multiplier",
        "vq_anneal_optimizer_steps",
    }
    numeric_schedule_keys = {
        "learning_rate",
        "head_learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "posterior_temperature_start",
        "posterior_temperature_end",
        "quantization_strength_start",
        "quantization_strength_end",
    }
    for key in REGISTERED_SCHEDULE:
        if key not in schedule:
            raise ValueError(f"schedule.{key} is missing")
        if key in integral_schedule_keys:
            _require_integral(
                schedule[key],
                f"schedule.{key}",
                minimum=0 if key == "warmup_optimizer_steps" else 1,
            )
        elif key in numeric_schedule_keys:
            if isinstance(schedule[key], (bool, np.bool_)) or not isinstance(
                schedule[key], (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"schedule.{key} must be numeric")
    _require_exact(schedule, REGISTERED_SCHEDULE, "schedule")
    if int(schedule["pretrain_minimum_steps"]) > int(
        schedule["continuous_pretrain_optimizer_steps"]
    ):
        raise ValueError("pretrain minimum exceeds pretrain budget")
    if int(schedule["vq_minimum_steps"]) > int(schedule["vq_optimizer_steps"]):
        raise ValueError("VQ minimum exceeds VQ budget")
    if int(schedule["head_minimum_steps"]) > int(schedule["task_head_optimizer_steps"]):
        raise ValueError("head minimum exceeds head budget")

    constants = _require_mapping(root["candidate_constants"], "candidate_constants")
    _require_exact(constants, CANDIDATE_CONSTANTS, "candidate_constants")

    candidates = root["candidates"]
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be an ordered list")
    if len(candidates) != len(CANDIDATES):
        raise ValueError("candidate count drifted")
    for index, (actual, expected) in enumerate(zip(candidates, CANDIDATES, strict=True)):
        _require_exact(actual, expected, f"candidate[{index}]")
        if set(_require_mapping(actual, "candidate")) != {
            "candidate_id",
            "role",
            "comparison_class",
            "eeg_shared_history_tokens",
            "fnirs_shared_history_tokens",
            "lag_loss_weight",
            "step_multiplier",
        }:
            raise ValueError("candidate config may override only registered history/objective fields")
        _require_integral(
            actual.get("eeg_shared_history_tokens"),
            "candidate.eeg_shared_history_tokens",
            minimum=0,
        )
        _require_integral(
            actual.get("fnirs_shared_history_tokens"),
            "candidate.fnirs_shared_history_tokens",
            minimum=0,
        )
        _require_integral(actual.get("step_multiplier"), "candidate.step_multiplier")
        if isinstance(actual.get("lag_loss_weight"), (bool, np.bool_)) or not isinstance(
            actual.get("lag_loss_weight"), (int, float, np.integer, np.floating)
        ):
            raise ValueError("candidate.lag_loss_weight must be numeric")
        if float(actual["lag_loss_weight"]) < 0.0:
            raise ValueError("candidate.lag_loss_weight must be non-negative")
    if tuple(str(row["candidate_id"]) for row in candidates) != CANDIDATE_IDS:
        raise ValueError("candidate order drifted")

    _require_exact(
        root["checkpoint_selection"], REGISTERED_CHECKPOINT_SELECTION, "checkpoint_selection"
    )
    selection = _require_mapping(root["checkpoint_selection"], "checkpoint_selection")
    if selection.get("development_values_available_to_selector") is not False:
        raise PermissionError("development values are closed to candidate selection")
    _require_exact(
        root["frozen_apply"],
        _REGISTERED_CONTRACT["frozen_apply"],
        "frozen_apply",
    )
    _require_exact(root["output"], _REGISTERED_CONTRACT["output"], "output")

    if not REGISTERED_PROTOCOL_PATH.is_file() or _sha256(
        REGISTERED_PROTOCOL_PATH
    ) != REGISTERED_PROTOCOL_SHA256:
        raise ValueError("frozen optimization protocol SHA256 drifted")

    if config_path is not None:
        opt_path = _path_is_repository_local(config_path, label="optimization config")
        if not opt_path.is_file():
            raise FileNotFoundError(opt_path)
        if _sha256(opt_path) != REGISTERED_OPTIMIZATION_SHA256:
            raise ValueError("optimization YAML SHA256 drifted from the registered file")
        loaded = yaml.safe_load(opt_path.read_text(encoding="utf-8"))
        if _canonical_json(loaded) != _canonical_json(root):
            raise ValueError("in-memory optimization config differs from bound YAML")

    selected_base_path = (
        _path_is_repository_local(base_config_path, label="base config")
        if base_config_path is not None
        else _path_is_repository_local(_resolve(str(bound["path"])), label="base config")
    )
    if selected_base_path != REGISTERED_BASE_CONFIG_PATH:
        raise PermissionError("base config path is not the registered repository-local YAML")
    if not selected_base_path.is_file():
        raise FileNotFoundError(selected_base_path)
    if _sha256(selected_base_path) != REGISTERED_BASE_SHA256:
        raise ValueError("SHA-bound base YAML drifted")
    loaded_base = yaml.safe_load(selected_base_path.read_text(encoding="utf-8"))
    if base_config is not None and _canonical_json(base_config) != _canonical_json(loaded_base):
        raise ValueError("in-memory base config differs from bound base YAML")
    # This is the reviewed validator that owns the canonical cache, branch,
    # split, objective, model, quantizer, and protected boundary checks.
    reviewed.validate_config(loaded_base)
    return loaded_base


def _validate_bound_config(
    config: Mapping[str, Any], config_path: Path
) -> tuple[Path, Path, Mapping[str, Any]]:
    """Return the exact optimization and SHA-bound base YAML pair."""

    opt_path = _path_is_repository_local(config_path, label="optimization config")
    base_path = _path_is_repository_local(
        _resolve(str(config["base_config"]["path"])), label="base config"
    )
    base = validate_optimization_config(
        config,
        config_path=opt_path,
        base_config_path=base_path,
    )
    return opt_path, base_path, base


# ---------------------------------------------------------------------------
# Fit-only preparation.  Development has no field on this object by design.
# ---------------------------------------------------------------------------


@dataclass
class OptimizationFitTask:
    task_id: str
    dataset_id: str
    parameter: reviewed.PreparedPartition
    selection: reviewed.PreparedPartition
    eeg_standardizer: reviewed.ChannelStandardizer
    fnirs_standardizer: reviewed.ChannelStandardizer
    eeg_native_standardizer: Any
    fnirs_native_standardizer: Any
    protected_metadata_indexed: bool
    measured_access_count: int
    protected_measured_access_count: int
    _preparation_capability: object = field(repr=False, compare=False)
    governance_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "governance_digest", _fit_governance_digest(self))

    def validate_governance(self, base_config: Mapping[str, Any]) -> None:
        _validate_fit_governance(self, base_config)


@dataclass(frozen=True)
class OptimizationDevelopmentTask:
    task_id: str
    dataset_id: str
    partition: reviewed.PreparedPartition
    measured_access_count: int
    protected_measured_access_count: int
    protected_metadata_indexed: bool
    global_selection_digest: str
    _selection_capability: object = field(repr=False, compare=False)

    @property
    def parameter(self) -> reviewed.PreparedPartition:
        # Reviewed model constructors consume ``prepared.parameter``.  During
        # post-selection application the one development partition is the
        # dimension/signature source and is never passed to candidate training.
        return self.partition


def _partition_governance_payload(partition: reviewed.PreparedPartition) -> dict[str, Any]:
    return {
        "role": partition.role,
        "sample_id": np.asarray(partition.sample_id).astype(str).tolist(),
        "subject": np.asarray(partition.subject).astype(str).tolist(),
        "condition": np.asarray(partition.condition).astype(str).tolist(),
        "target": np.asarray(partition.target, dtype=int).tolist(),
        "record_id": np.asarray(partition.record_id).astype(str).tolist(),
        "eeg_event_time_ms": np.asarray(partition.eeg_event_time_ms, dtype=float).tolist(),
        "fnirs_event_time_ms": np.asarray(partition.fnirs_event_time_ms, dtype=float).tolist(),
        "donor_index": np.asarray(partition.donor_index, dtype=int).tolist(),
    }


def _fit_governance_digest(task: OptimizationFitTask) -> str:
    payload = {
        "schema": "lc_spvq_fit_selection_governance_v1",
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "protected_metadata_indexed": bool(task.protected_metadata_indexed),
        "measured_access_count": int(task.measured_access_count),
        "protected_measured_access_count": int(task.protected_measured_access_count),
        "partitions": [
            _partition_governance_payload(task.parameter),
            _partition_governance_payload(task.selection),
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _subject_class_counts(partition: reviewed.PreparedPartition) -> Counter[tuple[str, str]]:
    return Counter(
        (str(subject), str(condition))
        for subject, condition in zip(
            partition.subject, partition.condition, strict=True
        )
    )


def _assert_exact_fit_counts(
    partition: reviewed.PreparedPartition,
    expected_subjects: Sequence[str],
    *,
    samples_per_subject_class: int = SAMPLES_PER_SUBJECT_CLASS,
) -> None:
    subjects = {str(value) for value in partition.subject}
    if subjects != {str(value) for value in expected_subjects}:
        raise PermissionError("prepared subjects differ from the configured nonprotected split")
    class_names = tuple(reviewed.TASK_SPECS[partition.task_id].class_names) if hasattr(partition, "task_id") else None
    # PreparedPartition intentionally does not retain task_id.  Its target class
    # support is therefore checked from the task-specific caller below.
    counts = _subject_class_counts(partition)
    if any(int(value) != int(samples_per_subject_class) for value in counts.values()):
        raise ValueError("every configured subject/class must contribute exactly 8 samples")
    if not counts:
        raise RuntimeError("prepared partition is empty")
    del class_names


def _assert_exact_task_counts(
    partition: reviewed.PreparedPartition,
    task_id: str,
    expected_subjects: Sequence[str],
    *,
    samples_per_subject_class: int,
) -> None:
    subjects = {str(value) for value in partition.subject}
    if subjects != {str(value) for value in expected_subjects}:
        raise PermissionError("prepared subjects differ from configured nonprotected split")
    class_names = set(reviewed.TASK_SPECS[task_id].class_names)
    counts = _subject_class_counts(partition)
    expected = {
        (str(subject), str(condition)): int(samples_per_subject_class)
        for subject in expected_subjects
        for condition in class_names
    }
    if counts != expected:
        raise ValueError(
            "every configured subject/class must contribute exactly 8 samples; "
            f"observed={dict(counts)}"
        )


def _assert_target_mapping(
    partition: reviewed.PreparedPartition,
    task_id: str,
) -> None:
    class_to_index = {
        str(name): index
        for index, name in enumerate(reviewed.TASK_SPECS[task_id].class_names)
    }
    expected = np.asarray(
        [class_to_index[str(condition)] for condition in partition.condition],
        dtype=np.int64,
    )
    observed_raw = np.asarray(partition.target)
    if observed_raw.dtype.kind not in {"i", "u"}:
        raise PermissionError("partition integer targets must have an integer dtype")
    observed = observed_raw.astype(np.int64, copy=False)
    if observed.shape != expected.shape or not np.array_equal(observed, expected):
        raise PermissionError("partition integer targets drifted from canonical conditions")


def _validate_fit_governance(
    task: OptimizationFitTask, base_config: Mapping[str, Any]
) -> None:
    if task._preparation_capability is not reviewed._PREPARATION_CAPABILITY:
        raise PermissionError("fit-only task lacks the opaque reviewed preparation capability")
    if task.governance_digest != _fit_governance_digest(task):
        raise PermissionError("fit-only task governance metadata changed after preparation")
    if not task.protected_metadata_indexed:
        raise RuntimeError("protected metadata boundary was not indexed")
    if int(task.protected_measured_access_count) != 0:
        raise PermissionError("protected measured access occurred")
    if task.task_id not in TASKS or task.dataset_id != reviewed.TASK_SPECS[task.task_id].dataset_id:
        raise ValueError("fit-only task identity drifted")
    split = base_config["data_split"]
    for partition, role in (
        (task.parameter, "fit_parameter_subjects"),
        (task.selection, "fit_selection_subjects"),
    ):
        expected_subjects = split[role][task.dataset_id]
        expected_role = "fit_parameter" if role == "fit_parameter_subjects" else "fit_selection"
        if partition.role != expected_role:
            raise ValueError("fit-only partition role drifted")
        _assert_exact_task_counts(
            partition,
            task.task_id,
            expected_subjects,
            samples_per_subject_class=SAMPLES_PER_SUBJECT_CLASS,
        )
        _assert_target_mapping(partition, task.task_id)
        for sample_id, subject, record_id in zip(
            np.asarray(partition.sample_id).astype(str),
            np.asarray(partition.subject).astype(str),
            np.asarray(partition.record_id).astype(str),
            strict=True,
        ):
            prefix = f"{task.dataset_id}|{subject}|{record_id}|"
            if not sample_id.startswith(prefix):
                raise ValueError("prepared canonical sample identity drifted")
            if record_id not in reviewed.TASK_SPECS[task.task_id].record_ids:
                raise ValueError("prepared record identity leaves the task contract")
    total = len(task.parameter.sample_id) + len(task.selection.sample_id)
    if int(task.measured_access_count) != total:
        raise RuntimeError("fit-only measured-access counter differs from sample support")


def _build_base_dataset(base_config: Mapping[str, Any], task_id: str) -> Any:
    spec = reviewed.TASK_SPECS[task_id]
    source = base_config["source"]
    return reviewed.UnifiedPhysiologyWindowDataset(
        cache_root=_resolve(str(source["cache_root"])),
        dataset_ids=(spec.dataset_id,),
        window_duration_s=float(source["window_duration_s"]),
        window_offset_s=float(source["window_offset_s"]),
        eeg_signal_branch=str(source["eeg_signal_branch"]),
        require_eeg_artifact_cache=spec.dataset_id == "eeg_fnirs_single_trial",
    )


def _role_dataset(
    base: Any,
    base_config: Mapping[str, Any],
    task_id: str,
    role: str,
) -> Any:
    spec = reviewed.TASK_SPECS[task_id]
    return reviewed.LagConditionedTaskDataset(
        task_id=task_id,
        admitted_subjects=base_config["data_split"][role][spec.dataset_id],
        forbidden_subjects=base_config["data_split"]["protected_or_unused"][spec.dataset_id],
        cache_root=_resolve(str(base_config["source"]["cache_root"])),
        eeg_signal_branch=str(base_config["source"]["eeg_signal_branch"]),
        base_dataset=base,
    )


def _spread_positions(length: int, count: int) -> tuple[int, ...]:
    if count < 1 or length < count:
        raise ValueError("event-time spread requires length >= count >= 1")
    if count == 1:
        return (length // 2,)
    raw = np.linspace(0, length - 1, num=count)
    positions = tuple(int(round(value)) for value in raw)
    if len(set(positions)) != count:
        # This should not occur when count <= length, but fail closed rather
        # than silently reducing the registered group support.
        raise RuntimeError("event-time spread produced duplicate positions")
    return positions


def _stratified_dataset_view(
    dataset: Any,
    *,
    samples_per_subject_class: int,
    sample_registry_seed: int,
) -> Any:
    """Return a shallow measured-data view with a frozen diverse row registry.

    Rows are chosen metadata-only. Multiple records are allocated round-robin
    (normally 3/3/2 for the three motor-imagery records at cap 8), and picks are
    spread over event time within each record. The seed rotates which record
    receives the extra slot and resolves otherwise arbitrary ordering ties.
    """

    cap = _require_integral(
        samples_per_subject_class, "samples_per_subject_class", minimum=2
    )
    grouped: dict[tuple[str, str], list[Any]] = {}
    for row in dataset.rows:
        grouped.setdefault((str(row.subject), str(row.condition)), []).append(row)
    if not grouped:
        raise RuntimeError("sample registry received an empty admitted dataset")

    selected: list[Any] = []
    registry_rows: list[dict[str, Any]] = []
    for (subject, condition), rows in sorted(grouped.items()):
        by_record: dict[str, list[Any]] = {}
        for row in rows:
            by_record.setdefault(str(row.record_id), []).append(row)
        record_ids = sorted(by_record)
        digest = hashlib.sha256(
            f"{int(sample_registry_seed)}|{subject}|{condition}".encode("utf-8")
        ).digest()
        rotation = int.from_bytes(digest[:4], "big") % len(record_ids)
        record_ids = record_ids[rotation:] + record_ids[:rotation]
        allocations = {record_id: 0 for record_id in record_ids}
        for slot in range(cap):
            allocations[record_ids[slot % len(record_ids)]] += 1
        if any(len(by_record[key]) < allocations[key] for key in record_ids):
            raise ValueError(
                f"insufficient exact support for {subject}/{condition}: "
                f"allocations={allocations}"
            )
        group_selected: list[Any] = []
        for record_id in record_ids:
            ordered = sorted(
                by_record[record_id],
                key=lambda row: (
                    float(row.event_time_ms),
                    float(row.fnirs_event_time_ms),
                    str(row.sample_id),
                ),
            )
            for position in _spread_positions(len(ordered), allocations[record_id]):
                group_selected.append(ordered[position])
        if len(group_selected) != cap or len({row.sample_id for row in group_selected}) != cap:
            raise RuntimeError("sample registry failed exact unique group support")
        group_selected.sort(
            key=lambda row: (str(row.record_id), float(row.event_time_ms), str(row.sample_id))
        )
        selected.extend(group_selected)
        for row in group_selected:
            registry_rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "record_id": str(row.record_id),
                    "event_time_ms": float(row.event_time_ms),
                    "fnirs_event_time_ms": float(row.fnirs_event_time_ms),
                    "sample_id": str(row.sample_id),
                }
            )

    view = copy.copy(dataset)
    view.rows = tuple(selected)
    view.derangement = None
    view.measured_access_count = 0
    view.protected_measured_access_count = 0
    view._optimization_sample_registry = tuple(registry_rows)
    return view


def _apply_fit_standardizers(
    partition: reviewed.PreparedPartition,
    *,
    eeg_standardizer: reviewed.ChannelStandardizer,
    fnirs_standardizer: reviewed.ChannelStandardizer,
    eeg_native_standardizer: Any,
    fnirs_native_standardizer: Any,
) -> None:
    partition.eeg = reviewed.apply_channel_standardizer(
        partition.eeg,
        partition.eeg_point_mask,
        partition.eeg_channel_mask,
        eeg_standardizer,
    )
    partition.fnirs = reviewed.apply_channel_standardizer(
        partition.fnirs,
        partition.fnirs_point_mask,
        partition.fnirs_channel_mask,
        fnirs_standardizer,
    )
    partition.eeg_native = reviewed.apply_masked_standardizer(
        partition.eeg_native, eeg_native_standardizer
    )
    partition.fnirs_native = reviewed.apply_masked_standardizer(
        partition.fnirs_native, fnirs_native_standardizer
    )


def _protected_metadata_indexed(base: Any, dataset_id: str, base_config: Mapping[str, Any]) -> bool:
    forbidden = set(base_config["data_split"]["protected_or_unused"][dataset_id])
    observed = {
        str(ref.record.canonical_subject_id)
        for ref in base.windows
    }
    return observed.intersection(forbidden) == forbidden


def _base_from_optimization_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    path = _path_is_repository_local(
        _resolve(str(config["base_config"]["path"])), label="base config"
    )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    reviewed.validate_config(loaded)
    return loaded


def prepare_fit_selection_task(
    config: Mapping[str, Any],
    task_id: str,
    *,
    base_config: Mapping[str, Any] | None = None,
    derangement_seed: int | None = None,
) -> OptimizationFitTask:
    """Prepare only fit_parameter and fit_selection measured partitions.

    There is deliberately no development argument or development dataset in
    this function.  Calling it cannot materialize a development partition.
    """

    validate_optimization_config(config, base_config=base_config)
    if task_id not in TASKS:
        raise ValueError(f"optimization task must be one of {TASKS}")
    base = _base_from_optimization_config(config) if base_config is None else base_config
    reviewed.validate_config(base)
    seed = int(
        config["execution"]["derangement_seed"]
        if derangement_seed is None
        else derangement_seed
    )
    if seed != int(config["execution"]["derangement_seed"]):
        raise ValueError("derangement seed override drifted from the frozen registry")
    sample_registry_seed = int(config["execution"]["sample_registry_seed"])
    spec = reviewed.TASK_SPECS[task_id]
    source = base["source"]
    unified = _build_base_dataset(base, task_id)
    roles = ("fit_parameter_subjects", "fit_selection_subjects")
    role_datasets = {
        role: _stratified_dataset_view(
            _role_dataset(unified, base, task_id, role),
            samples_per_subject_class=SAMPLES_PER_SUBJECT_CLASS,
            sample_registry_seed=sample_registry_seed,
        )
        for role in roles
    }
    partitions = {
        role: reviewed.prepare_partition(
            role_datasets[role],
            role="fit_parameter" if role == "fit_parameter_subjects" else "fit_selection",
            max_per_subject_class=SAMPLES_PER_SUBJECT_CLASS,
            derangement_seed=seed,
        )
        for role in roles
    }
    _assert_exact_task_counts(
        partitions["fit_parameter_subjects"],
        task_id,
        base["data_split"]["fit_parameter_subjects"][spec.dataset_id],
        samples_per_subject_class=SAMPLES_PER_SUBJECT_CLASS,
    )
    _assert_exact_task_counts(
        partitions["fit_selection_subjects"],
        task_id,
        base["data_split"]["fit_selection_subjects"][spec.dataset_id],
        samples_per_subject_class=SAMPLES_PER_SUBJECT_CLASS,
    )
    signatures = {
        (
            part.eeg_channel_names,
            part.fnirs_channel_names,
            part.fnirs_component_roles,
        )
        for part in partitions.values()
    }
    if len(signatures) != 1:
        raise RuntimeError("channel/component signature drifted across fit partitions")

    # Fit every input/native standardizer from fit_parameter only, then apply
    # those frozen statistics to fit_selection.  No development object exists.
    parameter = partitions["fit_parameter_subjects"]
    selection = partitions["fit_selection_subjects"]
    eeg_stats = reviewed.fit_channel_standardizer(
        parameter.eeg, parameter.eeg_point_mask, parameter.eeg_channel_mask
    )
    fnirs_stats = reviewed.fit_channel_standardizer(
        parameter.fnirs, parameter.fnirs_point_mask, parameter.fnirs_channel_mask
    )
    eeg_native_stats = reviewed.fit_masked_standardizer(parameter.eeg_native)
    fnirs_native_stats = reviewed.fit_masked_standardizer(parameter.fnirs_native)
    for partition in (parameter, selection):
        _apply_fit_standardizers(
            partition,
            eeg_standardizer=eeg_stats,
            fnirs_standardizer=fnirs_stats,
            eeg_native_standardizer=eeg_native_stats,
            fnirs_native_standardizer=fnirs_native_stats,
        )

    measured_access_count = sum(
        int(dataset.measured_access_count) for dataset in role_datasets.values()
    )
    protected_access_count = sum(
        int(dataset.protected_measured_access_count)
        for dataset in role_datasets.values()
    )
    if protected_access_count != 0:
        raise PermissionError("protected measured access occurred during fit preparation")
    if not _protected_metadata_indexed(unified, spec.dataset_id, base):
        raise RuntimeError("unified loader did not index the complete protected boundary")
    result = OptimizationFitTask(
        task_id=task_id,
        dataset_id=spec.dataset_id,
        parameter=parameter,
        selection=selection,
        eeg_standardizer=eeg_stats,
        fnirs_standardizer=fnirs_stats,
        eeg_native_standardizer=eeg_native_stats,
        fnirs_native_standardizer=fnirs_native_stats,
        protected_metadata_indexed=True,
        measured_access_count=measured_access_count,
        protected_measured_access_count=protected_access_count,
        _preparation_capability=reviewed._PREPARATION_CAPABILITY,
    )
    result.validate_governance(base)
    return result


def prepare_development_task(
    config: Mapping[str, Any],
    fit_task: OptimizationFitTask,
    *,
    base_config: Mapping[str, Any] | None = None,
    derangement_seed: int | None = None,
    selection_capability: object | None = None,
    global_selection_digest: str | None = None,
) -> OptimizationDevelopmentTask:
    """Materialize development exactly once, after global selection."""

    if not isinstance(selection_capability, _DevelopmentSelectionPermit):
        raise PermissionError(
            "development preparation requires the post-selection capability"
        )
    if not isinstance(global_selection_digest, str) or len(global_selection_digest) != 64:
        raise PermissionError("development preparation requires an immutable decision digest")
    if not isinstance(fit_task, OptimizationFitTask):
        raise TypeError("development preparation requires the fit-only task object")
    base = _base_from_optimization_config(config) if base_config is None else base_config
    validate_optimization_config(config, base_config=base)
    fit_task.validate_governance(base)
    seed = int(
        config["execution"]["derangement_seed"]
        if derangement_seed is None
        else derangement_seed
    )
    if seed != int(config["execution"]["derangement_seed"]):
        raise ValueError("derangement seed override drifted from the frozen registry")
    sample_registry_seed = int(config["execution"]["sample_registry_seed"])
    task_id = fit_task.task_id
    selection_capability.consume(task_id, str(global_selection_digest))
    spec = reviewed.TASK_SPECS[task_id]
    unified = _build_base_dataset(base, task_id)
    dataset = _stratified_dataset_view(
        _role_dataset(unified, base, task_id, "development_apply_subjects"),
        samples_per_subject_class=SAMPLES_PER_SUBJECT_CLASS,
        sample_registry_seed=sample_registry_seed,
    )
    partition = reviewed.prepare_partition(
        dataset,
        role="development_apply",
        max_per_subject_class=SAMPLES_PER_SUBJECT_CLASS,
        derangement_seed=seed,
    )
    _assert_exact_task_counts(
        partition,
        task_id,
        base["data_split"]["development_apply_subjects"][spec.dataset_id],
        samples_per_subject_class=SAMPLES_PER_SUBJECT_CLASS,
    )
    _assert_target_mapping(partition, task_id)
    signature = (
        partition.eeg_channel_names,
        partition.fnirs_channel_names,
        partition.fnirs_component_roles,
    )
    fit_signature = (
        fit_task.parameter.eeg_channel_names,
        fit_task.parameter.fnirs_channel_names,
        fit_task.parameter.fnirs_component_roles,
    )
    if signature != fit_signature:
        raise RuntimeError("development channel/component signature drifted")
    _apply_fit_standardizers(
        partition,
        eeg_standardizer=fit_task.eeg_standardizer,
        fnirs_standardizer=fit_task.fnirs_standardizer,
        eeg_native_standardizer=fit_task.eeg_native_standardizer,
        fnirs_native_standardizer=fit_task.fnirs_native_standardizer,
    )
    protected_access_count = int(dataset.protected_measured_access_count)
    if protected_access_count != 0:
        raise PermissionError("protected measured access occurred during development preparation")
    if not _protected_metadata_indexed(unified, spec.dataset_id, base):
        raise RuntimeError("unified loader did not index the complete protected boundary")
    return OptimizationDevelopmentTask(
        task_id=task_id,
        dataset_id=spec.dataset_id,
        partition=partition,
        measured_access_count=int(dataset.measured_access_count),
        protected_measured_access_count=protected_access_count,
        protected_metadata_indexed=True,
        global_selection_digest=global_selection_digest,
        _selection_capability=selection_capability,
    )


# ---------------------------------------------------------------------------
# Candidate runtime configuration and fit-selection checkpoint selection
# ---------------------------------------------------------------------------


def _candidate_by_id(candidate_id: str) -> Mapping[str, Any]:
    for candidate in CANDIDATES:
        if str(candidate["candidate_id"]) == str(candidate_id):
            return candidate
    raise KeyError(f"unknown registered candidate {candidate_id!r}")


def _candidate_override_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "eeg_shared_history_tokens",
            "fnirs_shared_history_tokens",
            "lag_loss_weight",
            "step_multiplier",
        )
    }


def candidate_runtime_config(
    base_config: Mapping[str, Any],
    optimization_config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int,
) -> Mapping[str, Any]:
    """Merge the base model with only registered history/objective overrides."""

    _require_exact(candidate, _candidate_by_id(str(candidate["candidate_id"])), "candidate")
    runtime = deepcopy(dict(base_config))
    model = runtime["model"]
    head = runtime["head"]
    quantizer = runtime["quantizer"]
    objective = runtime["objective"]
    constants = optimization_config["candidate_constants"]
    for key in (
        "encoder_depth",
        "encoder_num_heads",
        "encoder_feedforward_dim",
        "shared_dim",
        "eeg_private_dim",
        "fnirs_private_dim",
        "dropout",
        "projection_dim",
    ):
        model[key] = constants[key]
    head["coupling_rank"] = int(constants["coupling_rank"])
    quantizer["eeg_codebook_size"] = int(constants["codebook_size"])
    quantizer["fnirs_codebook_size"] = int(constants["codebook_size"])
    quantizer["embedding_dim"] = int(constants["shared_dim"])
    model["eeg_shared_history_tokens"] = int(candidate["eeg_shared_history_tokens"])
    model["fnirs_shared_history_tokens"] = int(candidate["fnirs_shared_history_tokens"])
    # The reviewed helper consumes the first entry. A singleton list makes the
    # candidate-specific training lambda explicit rather than silently falling
    # back to the base list order.
    objective["lag_loss_weight_candidates"] = [float(candidate["lag_loss_weight"])]

    schedule = optimization_config["schedule"]
    training = runtime["training"]
    training.update(
        {
            "seeds": [int(seed)],
            "batch_size": int(schedule["batch_size"]),
            "learning_rate": float(schedule["learning_rate"]),
            "head_learning_rate": float(schedule["head_learning_rate"]),
            "weight_decay": float(schedule["weight_decay"]),
            "betas": list(map(float, schedule["betas"])),
            "grad_clip_norm": float(schedule["grad_clip_norm"]),
            "deterministic_algorithms": True,
            "amp": False,
            "num_workers": 0,
        }
    )
    return runtime


def _set_seed(seed: int) -> None:
    reviewed._set_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _evaluation_steps(total_steps: int, interval: int) -> tuple[int, ...]:
    total = _require_integral(total_steps, "total_steps")
    every = _require_integral(interval, "evaluation_interval")
    return tuple(sorted(set(range(every, total + 1, every)) | {total}))


def _lag_weights_text(lag_details: Mapping[str, Any] | None, lag_module: Any) -> str:
    if lag_details is not None and "lag_weights" in lag_details:
        value = lag_details["lag_weights"]
    else:
        value = getattr(lag_module, "lag_mixture_weights", torch.empty(0))
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    else:
        values = list(value)
    return ";".join(f"{float(item):.8f}" for item in values)


def _float_tensor(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _selection_pretraining_loss(
    model: Any,
    lag_module: Any,
    partition: reviewed.PreparedPartition,
    runtime_config: Mapping[str, Any],
    *,
    device: torch.device,
    variant: str = VARIANT,
    include_commitment: bool,
    seed: int,
) -> float:
    loader = reviewed.make_prepared_loader(
        partition,
        batch_size=int(runtime_config["training"]["batch_size"]),
        shuffle=False,
        seed=int(seed),
        num_workers=0,
    )
    total = 0.0
    count = 0
    model.eval()
    lag_module.eval()
    previous_strength = float(model.get_quantization_strength())
    previous_temperature = float(model.get_posterior_temperature())
    if include_commitment:
        selection = REGISTERED_CHECKPOINT_SELECTION
        model.set_quantization_strength(
            float(selection["vq_evaluation_quantization_strength"])
        )
        model.set_posterior_temperature(
            float(selection["vq_evaluation_posterior_temperature"])
        )
    try:
        with torch.no_grad():
            for raw_batch in loader:
                batch = reviewed._batch_to_device(raw_batch, device)
                output = reviewed._forward_lc_spvq(model, batch, variant=variant)
                _, components, _ = reviewed._lc_spvq_pretraining_losses(
                    model,
                    lag_module,
                    output,
                    batch,
                    runtime_config,
                    variant=variant,
                    include_commitment=include_commitment,
                )
                # Candidate-independent no-task selection criterion. Raw/private
                # reconstruction and VQ commitment are deliberately excluded.
                fixed = (
                    0.5 * components["eeg_native"]
                    + 0.5 * components["fnirs_native"]
                    + float(
                        REGISTERED_CHECKPOINT_SELECTION[
                            "representation_fixed_lag_weight"
                        ]
                    )
                    * components["lag"]
                )
                batch_count = int(batch["target"].shape[0])
                total += _float_tensor(fixed) * batch_count
                count += batch_count
    finally:
        model.set_quantization_strength(previous_strength)
        model.set_posterior_temperature(previous_temperature)
    if count <= 0:
        raise RuntimeError("fit-selection evaluation admitted no samples")
    return float(total / count)


def _selection_head_metrics(
    model: Any,
    partition: reviewed.PreparedPartition,
    runtime_config: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int,
) -> tuple[Mapping[str, Any], float, float, Mapping[str, np.ndarray]]:
    metrics, arrays, _ = reviewed._evaluate_lc_spvq(
        model,
        partition,
        config=runtime_config,
        device=device,
        seed=int(seed),
    )
    combined = metrics["coupling_plus_private"]
    coupling_only = metrics["coupling_only"]
    logits = torch.from_numpy(np.asarray(arrays["combined_logits"], dtype=np.float32))
    targets = torch.from_numpy(np.asarray(arrays["target"], dtype=np.int64))
    cross_entropy = float(torch.nn.functional.cross_entropy(logits, targets).item())
    primary = float(combined["subject_equal_macro_f1"])
    coupling_primary = float(coupling_only["subject_equal_macro_f1"])
    return metrics, primary, coupling_primary, {
        "combined_cross_entropy": np.asarray(cross_entropy),
        **arrays,
    }


def _clone_state(module: Any) -> dict[str, Any]:
    return {
        str(key): value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
        for key, value in module.state_dict().items()
    }


def _restore_state(module: Any, state: Mapping[str, Any], device: torch.device) -> None:
    module.load_state_dict(state, strict=True)
    module.to(device)


def _frozen_state_digest(model: Any, lag_module: Any) -> str:
    digest = hashlib.sha256()
    for prefix, module in (("model", model), ("lag", lag_module)):
        for key, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{prefix}:{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    digest.update(f"strength={model.get_quantization_strength():.17g}".encode("utf-8"))
    digest.update(f"temperature={model.get_posterior_temperature():.17g}".encode("utf-8"))
    return digest.hexdigest()


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _save_checkpoint_once(
    path: Path,
    *,
    task_id: str,
    candidate: Mapping[str, Any],
    seed: int,
    stage: str,
    score: float,
    model_state: Mapping[str, Any],
    lag_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None = None,
    quantization_strength: float | None = None,
    posterior_temperature: float | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    reviewed._atomic_torch_save(
        {
            "schema": "lc_spvq_architecture_optimization_checkpoint_v1",
            "task_id": str(task_id),
            "variant": VARIANT,
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_config_overrides": _candidate_override_payload(candidate),
            "seed": int(seed),
            "stage": str(stage),
            "fit_selection_score": float(score),
            "model_state": dict(model_state),
            "lag_objective_state": dict(lag_state),
            "optimizer_state": {} if optimizer_state is None else dict(optimizer_state),
            "quantization_strength": quantization_strength,
            "posterior_temperature": posterior_temperature,
            "rng_state": dict(_capture_rng_state() if rng_state is None else rng_state),
            "protected_open": False,
            "development_used": False,
        },
        path,
    )


def _make_optimizer(parameters: Iterable[torch.nn.Parameter], runtime: Mapping[str, Any], *, head: bool) -> torch.optim.Optimizer:
    values = [parameter for parameter in parameters if parameter.requires_grad]
    if not values:
        raise RuntimeError("optimizer received no trainable parameters")
    return torch.optim.AdamW(
        values,
        lr=float(runtime["training"]["head_learning_rate"] if head else runtime["training"]["learning_rate"]),
        betas=tuple(map(float, runtime["training"]["betas"])),
        weight_decay=float(runtime["training"]["weight_decay"]),
    )


def _append_train_row(
    rows: list[dict[str, Any]],
    *,
    candidate: Mapping[str, Any],
    task_id: str,
    seed: int,
    stage: str,
    step: int,
    total_loss: float,
    gradient_norm: float,
    components: Mapping[str, Any] | None = None,
    lag_weights: str = "",
    quantization_strength: float | None = None,
    posterior_temperature: float | None = None,
    **extra: Any,
) -> None:
    row: dict[str, Any] = {
        "task_id": str(task_id),
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_role": str(candidate["role"]),
        "variant": VARIANT,
        "seed": int(seed),
        "stage": str(stage),
        "step": int(step),
        "total_loss": float(total_loss),
        "gradient_norm": float(gradient_norm),
        "lag_weights": lag_weights,
        "quantization_strength": "" if quantization_strength is None else float(quantization_strength),
        "posterior_temperature": "" if posterior_temperature is None else float(posterior_temperature),
        "selection_fixed_native_plus_lag_loss": "",
        "selection_coupling_plus_private_subject_equal_macro_f1": "",
        "selection_coupling_only_subject_equal_macro_f1": "",
        "selection_combined_cross_entropy": "",
        "checkpoint_marker": "",
    }
    if components:
        row.update({f"{name}_loss": _float_tensor(value) for name, value in components.items()})
    row.update(extra)
    rows.append(row)


def _append_eval_row(
    rows: list[dict[str, Any]],
    *,
    candidate: Mapping[str, Any],
    task_id: str,
    seed: int,
    stage: str,
    step: int,
    representation_loss: float,
    head_primary: float | None = None,
    coupling_primary: float | None = None,
    cross_entropy: float | None = None,
    checkpoint_marker: str = "",
) -> None:
    row: dict[str, Any] = {
        "task_id": str(task_id),
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_role": str(candidate["role"]),
        "variant": VARIANT,
        "seed": int(seed),
        "stage": str(stage),
        "step": int(step),
        "total_loss": "",
        "gradient_norm": "",
        "lag_weights": "",
        "quantization_strength": "",
        "posterior_temperature": "",
        "selection_fixed_native_plus_lag_loss": float(representation_loss),
        "selection_coupling_plus_private_subject_equal_macro_f1": "" if head_primary is None else float(head_primary),
        "selection_coupling_only_subject_equal_macro_f1": "" if coupling_primary is None else float(coupling_primary),
        "selection_combined_cross_entropy": "" if cross_entropy is None else float(cross_entropy),
        "checkpoint_marker": str(checkpoint_marker),
    }
    rows.append(row)


def _head_is_better(
    candidate_primary: float,
    candidate_cross_entropy: float,
    incumbent_primary: float | None,
    incumbent_cross_entropy: float | None,
    *,
    min_delta: float = 1.0e-6,
    tie_tolerance: float = 1.0e-8,
) -> bool:
    if incumbent_primary is None or incumbent_cross_entropy is None:
        return True
    if float(candidate_primary) > float(incumbent_primary) + float(min_delta):
        return True
    if math.isclose(
        float(candidate_primary),
        float(incumbent_primary),
        rel_tol=0.0,
        abs_tol=float(tie_tolerance),
    ):
        return float(candidate_cross_entropy) < float(incumbent_cross_entropy) - float(min_delta)
    return False


def train_candidate(
    prepared: OptimizationFitTask,
    base_config: Mapping[str, Any],
    optimization_config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one M1 candidate using fit_parameter and fit_selection only."""

    prepared.validate_governance(base_config)
    candidate = _candidate_by_id(str(candidate["candidate_id"]))
    runtime = candidate_runtime_config(
        base_config, optimization_config, candidate, seed=int(seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(int(seed))
    model = reviewed._lc_spvq_model(prepared, runtime).to(device)
    model._lc_spvq_task_id = prepared.task_id
    lag_module = reviewed._lag_matching_module(runtime).to(device)
    total_parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
        + sum(parameter.numel() for parameter in lag_module.parameters())
    )
    head_modules = (
        model.coupling_head,
        model.shared_marginal_classifier,
        model.private_classifier,
    )
    for module in head_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model.set_quantization_strength(float(runtime["quantizer"]["quantization_strength_start"]))
    train_loader = reviewed.make_prepared_loader(
        prepared.parameter,
        batch_size=int(runtime["training"]["batch_size"]),
        shuffle=True,
        seed=int(seed),
        num_workers=0,
    )
    history: list[dict[str, Any]] = []
    schedule = optimization_config["schedule"]
    interval = int(schedule["selection_evaluation_interval_steps"])
    patience = int(schedule["early_stopping_patience_evaluations"])
    multiplier = int(candidate["step_multiplier"])
    pretrain_total = int(schedule["continuous_pretrain_optimizer_steps"]) * multiplier
    pretrain_minimum = int(schedule["pretrain_minimum_steps"]) * multiplier
    vq_total = int(schedule["vq_optimizer_steps"]) * multiplier
    vq_minimum = int(schedule["vq_minimum_steps"]) * multiplier
    head_total = int(schedule["task_head_optimizer_steps"]) * multiplier
    head_minimum = int(schedule["head_minimum_steps"]) * multiplier
    if multiplier > 1:
        # The duration control is an exact 2x-budget control, not merely a
        # larger maximum that could early-stop at the common-budget boundary.
        pretrain_minimum = pretrain_total
        vq_minimum = vq_total
        head_minimum = head_total
    representation_min_delta = float(schedule["representation_early_stopping_min_delta"])
    head_min_delta = float(schedule["head_early_stopping_min_delta"])
    grad_clip = float(runtime["training"]["grad_clip_norm"])

    pretrain_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ] + list(lag_module.parameters())
    optimizer = _make_optimizer(pretrain_parameters, runtime, head=False)
    loader_iterator = reviewed._iterate_loader(train_loader)
    model.train()
    model.eeg_quantizer.eval()
    model.fnirs_quantizer.eval()
    lag_module.train()
    best_pretrain_score: float | None = None
    best_pretrain_state: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_pretrain_optimizer_state: dict[str, Any] | None = None
    best_pretrain_rng_state: dict[str, Any] | None = None
    best_pretrain_step: int | None = None
    stale = 0
    for step in range(1, pretrain_total + 1):
        batch = reviewed._batch_to_device(next(loader_iterator), device)
        optimizer.zero_grad(set_to_none=True)
        output = reviewed._forward_lc_spvq(model, batch, variant=VARIANT)
        loss, components, lag_details = reviewed._lc_spvq_pretraining_losses(
            model,
            lag_module,
            output,
            batch,
            runtime,
            variant=VARIANT,
            include_commitment=False,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            pretrain_parameters, grad_clip
        )
        optimizer.step()
        _append_train_row(
            history,
            candidate=candidate,
            task_id=prepared.task_id,
            seed=seed,
            stage="continuous_pretrain",
            step=step,
            total_loss=_float_tensor(loss),
            gradient_norm=_float_tensor(gradient_norm),
            components=components,
            lag_weights=_lag_weights_text(lag_details, lag_module),
        )
        if step in _evaluation_steps(pretrain_total, interval):
            selection_loss = _selection_pretraining_loss(
                model,
                lag_module,
                prepared.selection,
                runtime,
                device=device,
                include_commitment=False,
                seed=seed,
            )
            improved = (
                best_pretrain_score is None
                or selection_loss < best_pretrain_score - representation_min_delta
            )
            marker = ""
            if improved:
                best_pretrain_score = float(selection_loss)
                best_pretrain_state = (_clone_state(model), _clone_state(lag_module))
                best_pretrain_optimizer_state = deepcopy(optimizer.state_dict())
                best_pretrain_rng_state = _capture_rng_state()
                best_pretrain_step = int(step)
                marker = "best_pretrain"
                stale = 0
            else:
                stale += 1
            _append_eval_row(
                history,
                candidate=candidate,
                task_id=prepared.task_id,
                seed=seed,
                stage="continuous_pretrain_eval",
                step=step,
                representation_loss=selection_loss,
                checkpoint_marker=marker,
            )
            if step >= pretrain_minimum and stale >= patience:
                break
            model.train()
            model.eeg_quantizer.eval()
            model.fnirs_quantizer.eval()
            lag_module.train()
    pretrain_actual_steps = int(step)
    pretrain_stop_reason = (
        "early_stopping_patience"
        if pretrain_actual_steps < pretrain_total
        else "optimizer_budget"
    )
    if (
        best_pretrain_state is None
        or best_pretrain_optimizer_state is None
        or best_pretrain_rng_state is None
        or best_pretrain_score is None
        or best_pretrain_step is None
    ):
        raise RuntimeError("pretraining produced no fit-selection checkpoint")
    _restore_state(model, best_pretrain_state[0], device)
    _restore_state(lag_module, best_pretrain_state[1], device)
    _save_checkpoint_once(
        checkpoint_dir / "pretrain_best.pt",
        task_id=prepared.task_id,
        candidate=candidate,
        seed=seed,
        stage="continuous_pretrain",
        score=best_pretrain_score,
        model_state=best_pretrain_state[0],
        lag_state=best_pretrain_state[1],
        optimizer_state=best_pretrain_optimizer_state,
        quantization_strength=float(model.get_quantization_strength()),
        posterior_temperature=float(model.get_posterior_temperature()),
        rng_state=best_pretrain_rng_state,
    )

    kmeans = reviewed._initialize_codebooks_from_fit_parameter(
        model,
        prepared.parameter,
        config=runtime,
        device=device,
        seed=seed,
    )
    model.train()
    model.eeg_quantizer.train()
    model.fnirs_quantizer.train()
    lag_module.train()
    loader_iterator = reviewed._iterate_loader(train_loader)
    best_vq_score: float | None = None
    best_vq_state: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_vq_optimizer_state: dict[str, Any] | None = None
    best_vq_rng_state: dict[str, Any] | None = None
    best_vq_step: int | None = None
    stale = 0
    anneal_steps = int(schedule["vq_anneal_optimizer_steps"])
    for step in range(1, vq_total + 1):
        denominator = max(anneal_steps - 1, 1)
        fraction = min((step - 1) / denominator, 1.0)
        strength = float(schedule["quantization_strength_start"]) + fraction * (
            float(schedule["quantization_strength_end"])
            - float(schedule["quantization_strength_start"])
        )
        temperature = float(schedule["posterior_temperature_start"]) + fraction * (
            float(schedule["posterior_temperature_end"])
            - float(schedule["posterior_temperature_start"])
        )
        model.set_quantization_strength(strength)
        model.set_posterior_temperature(temperature)
        batch = reviewed._batch_to_device(next(loader_iterator), device)
        optimizer.zero_grad(set_to_none=True)
        output = reviewed._forward_lc_spvq(model, batch, variant=VARIANT)
        loss, components, lag_details = reviewed._lc_spvq_pretraining_losses(
            model,
            lag_module,
            output,
            batch,
            runtime,
            variant=VARIANT,
            include_commitment=True,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            pretrain_parameters, grad_clip
        )
        optimizer.step()
        _append_train_row(
            history,
            candidate=candidate,
            task_id=prepared.task_id,
            seed=seed,
            stage="vq_anneal",
            step=step,
            total_loss=_float_tensor(loss),
            gradient_norm=_float_tensor(gradient_norm),
            components=components,
            lag_weights=_lag_weights_text(lag_details, lag_module),
            quantization_strength=strength,
            posterior_temperature=temperature,
        )
        if step in _evaluation_steps(vq_total, interval):
            selection_loss = _selection_pretraining_loss(
                model,
                lag_module,
                prepared.selection,
                runtime,
                device=device,
                include_commitment=True,
                seed=seed,
            )
            improved = (
                best_vq_score is None
                or selection_loss < best_vq_score - representation_min_delta
            )
            marker = ""
            if improved:
                best_vq_score = float(selection_loss)
                best_vq_state = (_clone_state(model), _clone_state(lag_module))
                best_vq_optimizer_state = deepcopy(optimizer.state_dict())
                best_vq_rng_state = _capture_rng_state()
                best_vq_step = int(step)
                marker = "best_vq"
                stale = 0
            else:
                stale += 1
            _append_eval_row(
                history,
                candidate=candidate,
                task_id=prepared.task_id,
                seed=seed,
                stage="vq_anneal_eval",
                step=step,
                representation_loss=selection_loss,
                checkpoint_marker=marker,
            )
            if step >= vq_minimum and stale >= patience:
                break
            model.train()
            model.eeg_quantizer.train()
            model.fnirs_quantizer.train()
            lag_module.train()
    vq_actual_steps = int(step)
    vq_stop_reason = (
        "early_stopping_patience"
        if vq_actual_steps < vq_total
        else "optimizer_budget"
    )
    if (
        best_vq_state is None
        or best_vq_optimizer_state is None
        or best_vq_rng_state is None
        or best_vq_score is None
        or best_vq_step is None
    ):
        raise RuntimeError("VQ annealing produced no fit-selection checkpoint")
    _restore_state(model, best_vq_state[0], device)
    _restore_state(lag_module, best_vq_state[1], device)
    model.set_quantization_strength(
        float(REGISTERED_CHECKPOINT_SELECTION["vq_evaluation_quantization_strength"])
    )
    model.set_posterior_temperature(
        float(REGISTERED_CHECKPOINT_SELECTION["vq_evaluation_posterior_temperature"])
    )
    # Refresh the state after restoring Python-only posterior temperature.
    best_vq_state = (_clone_state(model), _clone_state(lag_module))
    _save_checkpoint_once(
        checkpoint_dir / "vq_best.pt",
        task_id=prepared.task_id,
        candidate=candidate,
        seed=seed,
        stage="vq_anneal",
        score=best_vq_score,
        model_state=best_vq_state[0],
        lag_state=best_vq_state[1],
        optimizer_state=best_vq_optimizer_state,
        quantization_strength=float(model.get_quantization_strength()),
        posterior_temperature=float(model.get_posterior_temperature()),
        rng_state=best_vq_rng_state,
    )

    # Representation and codebooks are frozen; only the registered task heads
    # are optimized in this stage.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head_parameters: list[torch.nn.Parameter] = []
    for module in head_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            head_parameters.append(parameter)
    for parameter in lag_module.parameters():
        parameter.requires_grad_(False)
    head_optimizer = _make_optimizer(head_parameters, runtime, head=True)
    auxiliary = float(runtime["head"]["ablation_auxiliary_cross_entropy_weight"])
    loader_iterator = reviewed._iterate_loader(train_loader)
    model.eval()
    for module in head_modules:
        module.train()
    best_head_primary: float | None = None
    best_head_cross_entropy: float | None = None
    best_head_metrics: Mapping[str, Any] | None = None
    best_head_state: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_head_optimizer_state: dict[str, Any] | None = None
    best_head_rng_state: dict[str, Any] | None = None
    best_head_step: int | None = None
    stale = 0
    for step in range(1, head_total + 1):
        batch = reviewed._batch_to_device(next(loader_iterator), device)
        head_optimizer.zero_grad(set_to_none=True)
        output = reviewed._forward_lc_spvq(model, batch, variant=VARIANT)
        head_loss = torch.nn.functional.cross_entropy(
            output["combined_logits"], batch["target"]
        ) + auxiliary * (
            torch.nn.functional.cross_entropy(
                output["coupling_only_logits"], batch["target"]
            )
            + torch.nn.functional.cross_entropy(
                output["shared_marginal_only_logits"], batch["target"]
            )
            + torch.nn.functional.cross_entropy(
                output["private_only_logits"], batch["target"]
            )
        )
        head_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(head_parameters, grad_clip)
        head_optimizer.step()
        _append_train_row(
            history,
            candidate=candidate,
            task_id=prepared.task_id,
            seed=seed,
            stage="task_head",
            step=step,
            total_loss=_float_tensor(head_loss),
            gradient_norm=_float_tensor(gradient_norm),
        )
        if step in _evaluation_steps(head_total, interval):
            model.eval()
            metrics, primary, coupling_primary, arrays = _selection_head_metrics(
                model,
                prepared.selection,
                runtime,
                device=device,
                seed=seed,
            )
            cross_entropy = float(np.asarray(arrays["combined_cross_entropy"]).item())
            improved = _head_is_better(
                primary,
                cross_entropy,
                best_head_primary,
                best_head_cross_entropy,
                min_delta=head_min_delta,
                tie_tolerance=float(
                    optimization_config["checkpoint_selection"]["numeric_tie_tolerance"]
                ),
            )
            marker = ""
            if improved:
                best_head_primary = float(primary)
                best_head_cross_entropy = float(cross_entropy)
                best_head_metrics = metrics
                best_head_state = (_clone_state(model), _clone_state(lag_module))
                best_head_optimizer_state = deepcopy(head_optimizer.state_dict())
                best_head_rng_state = _capture_rng_state()
                best_head_step = int(step)
                marker = "best_head"
                stale = 0
            else:
                stale += 1
            representation_loss = _selection_pretraining_loss(
                model,
                lag_module,
                prepared.selection,
                runtime,
                device=device,
                include_commitment=True,
                seed=seed,
            )
            _append_eval_row(
                history,
                candidate=candidate,
                task_id=prepared.task_id,
                seed=seed,
                stage="task_head_eval",
                step=step,
                representation_loss=representation_loss,
                head_primary=primary,
                coupling_primary=coupling_primary,
                cross_entropy=cross_entropy,
                checkpoint_marker=marker,
            )
            if step >= head_minimum and stale >= patience:
                break
            for module in head_modules:
                module.train()
    head_actual_steps = int(step)
    head_stop_reason = (
        "early_stopping_patience"
        if head_actual_steps < head_total
        else "optimizer_budget"
    )
    if (
        best_head_state is None
        or best_head_optimizer_state is None
        or best_head_rng_state is None
        or best_head_primary is None
        or best_head_cross_entropy is None
        or best_head_metrics is None
        or best_head_step is None
    ):
        raise RuntimeError("task-head training produced no fit-selection checkpoint")
    _restore_state(model, best_head_state[0], device)
    _restore_state(lag_module, best_head_state[1], device)
    _save_checkpoint_once(
        checkpoint_dir / "head_best.pt",
        task_id=prepared.task_id,
        candidate=candidate,
        seed=seed,
        stage="task_head",
        score=best_head_primary,
        model_state=best_head_state[0],
        lag_state=best_head_state[1],
        optimizer_state=best_head_optimizer_state,
        quantization_strength=float(model.get_quantization_strength()),
        posterior_temperature=float(model.get_posterior_temperature()),
        rng_state=best_head_rng_state,
    )
    _, _, fit_codebook_health = reviewed._evaluate_lc_spvq(
        model,
        prepared.parameter,
        config=runtime,
        device=device,
        seed=int(seed),
    )
    expected_codes = int(optimization_config["candidate_constants"]["codebook_size"])
    all_fit_parameter_codes_active = all(
        int(fit_codebook_health[modality]["active_codes"]) == expected_codes
        for modality in ("eeg", "fnirs")
    )

    _write_csv_atomic(output_dir / "step_curves.csv", history)
    result = {
        "schema": "lc_spvq_architecture_candidate_result_v1",
        "task_id": prepared.task_id,
        "variant": VARIANT,
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_role": str(candidate["role"]),
        "candidate_comparison_class": str(candidate["comparison_class"]),
        "candidate_config_overrides": {
            key: candidate[key]
            for key in (
                "eeg_shared_history_tokens",
                "fnirs_shared_history_tokens",
                "lag_loss_weight",
                "step_multiplier",
            )
        },
        "seed": int(seed),
        "status": "completed",
        "step_multiplier": int(candidate["step_multiplier"]),
        "trainable_parameter_count": total_parameter_count,
        "total_parameter_count": total_parameter_count,
        "development_used": False,
        "development_values_seen": False,
        "pretrain_steps": pretrain_actual_steps,
        "vq_steps": vq_actual_steps,
        "head_steps": head_actual_steps,
        "pretrain_stop_reason": pretrain_stop_reason,
        "vq_stop_reason": vq_stop_reason,
        "head_stop_reason": head_stop_reason,
        "best_pretrain_step": int(best_pretrain_step),
        "best_vq_step": int(best_vq_step),
        "best_head_step": int(best_head_step),
        "selection_fixed_native_plus_lag_loss": float(best_vq_score),
        # Compatibility alias retained for curve helpers; its value is the
        # fixed candidate-independent criterion, not candidate training loss.
        "selection_total_registered_pretraining_loss": float(best_vq_score),
        "selection_primary_metric": float(best_head_primary),
        "selection_coupling_plus_private_subject_equal_macro_f1": float(best_head_primary),
        "selection_coupling_only_subject_equal_macro_f1": float(
            best_head_metrics["coupling_only"]["subject_equal_macro_f1"]
        ),
        "selection_combined_cross_entropy": float(best_head_cross_entropy),
        "fit_selection_metrics": best_head_metrics,
        "fit_parameter_codebook_health": fit_codebook_health,
        "all_fit_parameter_codes_active": all_fit_parameter_codes_active,
        "derangement_nonoverlap_verified": True,
        "complete_registered_task_support": True,
        "kmeans_initialization": kmeans,
        "quantization_strength_final": float(model.get_quantization_strength()),
        "posterior_temperature_final": float(model.get_posterior_temperature()),
        "lag_weights": [
            float(value)
            for value in lag_module.lag_mixture_weights.detach().cpu().tolist()
        ],
        "checkpoints": {
            "pretrain_best": "checkpoints/pretrain_best.pt",
            "vq_best": "checkpoints/vq_best.pt",
            "head_best": "checkpoints/head_best.pt",
        },
        "protected_open": False,
        "protected_measured_access_count": int(prepared.protected_measured_access_count),
        "step_curves": history,
    }
    _write_json_atomic(output_dir / "result.json", result)
    return result


# ---------------------------------------------------------------------------
# Exact checkpoint and global architecture selection
# ---------------------------------------------------------------------------


def _result_metric(result: Mapping[str, Any], name: str) -> float:
    direct = result.get(name)
    if direct is not None:
        return float(direct)
    if name == "selection_fixed_native_plus_lag_loss":
        alias = result.get("selection_total_registered_pretraining_loss")
        if alias is not None:
            return float(alias)
    if name == "selection_coupling_plus_private_subject_equal_macro_f1":
        alias = result.get("selection_primary_metric")
        if alias is not None:
            return float(alias)
    metrics = result.get("fit_selection_metrics", result.get("selection_metrics", {}))
    if name in {
        "selection_primary_metric",
        "selection_coupling_plus_private_subject_equal_macro_f1",
    }:
        return float(metrics["coupling_plus_private"]["subject_equal_macro_f1"])
    if name == "selection_coupling_only_subject_equal_macro_f1":
        return float(metrics["coupling_only"]["subject_equal_macro_f1"])
    if name == "selection_combined_cross_entropy":
        if "combined_cross_entropy" in metrics:
            return float(metrics["combined_cross_entropy"])
    raise KeyError(f"candidate result lacks {name}")


def _result_selection_record(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        "primary": _result_metric(
            result, "selection_coupling_plus_private_subject_equal_macro_f1"
        ),
        "coupling_only": _result_metric(
            result, "selection_coupling_only_subject_equal_macro_f1"
        ),
        "combined_cross_entropy": _result_metric(
            result, "selection_combined_cross_entropy"
        ),
        "representation_loss": _result_metric(
            result, "selection_fixed_native_plus_lag_loss"
        ),
        "actual_optimizer_steps": float(
            int(result["pretrain_steps"])
            + int(result["vq_steps"])
            + int(result["head_steps"])
        ),
        "parameter_count": float(result["trainable_parameter_count"]),
    }


def _flatten_candidate_results(
    candidate_results: Mapping[Any, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    flattened: dict[str, dict[str, Mapping[str, Any]]] = {
        candidate_id: {} for candidate_id in CANDIDATE_IDS
    }

    def admit(task_id: Any, candidate_id: Any, result: Any) -> None:
        task = str(task_id)
        candidate = str(candidate_id)
        if task not in TASKS:
            raise ValueError(f"unregistered task result {task}")
        if candidate not in CANDIDATE_IDS:
            raise ValueError(f"unregistered candidate result {candidate}")
        if not isinstance(result, Mapping):
            raise ValueError("candidate result must be a mapping")
        if task in flattened[candidate]:
            raise ValueError(f"duplicate candidate result {task}/{candidate}")
        flattened[candidate][task] = result

    if isinstance(candidate_results, Mapping):
        for key, value in candidate_results.items():
            if isinstance(key, tuple) and len(key) == 2:
                admit(key[0], key[1], value)
            elif str(key) in TASKS and isinstance(value, Mapping):
                for candidate_id, result in value.items():
                    admit(key, candidate_id, result)
            elif str(key) in CANDIDATE_IDS and isinstance(value, Mapping):
                for task_id, result in value.items():
                    admit(task_id, key, result)
            else:
                raise ValueError(
                    "candidate results must be keyed by (task,candidate), task, or candidate"
                )
    else:
        for result in candidate_results:
            if not isinstance(result, Mapping):
                raise ValueError("candidate result sequence entries must be mappings")
            admit(result.get("task_id"), result.get("candidate_id"), result)
    for candidate_id in CANDIDATE_IDS:
        if set(flattened[candidate_id]) != set(TASKS):
            raise ValueError(f"candidate {candidate_id} lacks both task results")
    return flattened


def _global_candidate_key(
    candidate_id: str,
    flattened: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    tie_tolerance: float | None = None,
) -> tuple[float, float, float, float, float, float, int]:
    records = [_result_selection_record(flattened[candidate_id][task]) for task in TASKS]
    values = [
        float(np.mean([row["primary"] for row in records])),
        float(min(row["primary"] for row in records)),
        float(np.mean([row["coupling_only"] for row in records])),
        -float(np.mean([row["combined_cross_entropy"] for row in records])),
    ]
    tolerance = float(
        REGISTERED_CHECKPOINT_SELECTION["numeric_tie_tolerance"]
        if tie_tolerance is None
        else tie_tolerance
    )
    if tolerance <= 0.0:
        raise ValueError("numeric tie tolerance must be positive")
    actual_steps = float(np.mean([row["actual_optimizer_steps"] for row in records]))
    parameter_count = float(np.mean([row["parameter_count"] for row in records]))
    order = CANDIDATE_IDS.index(candidate_id)
    # This is the raw auditable tuple. Ranking applies ``tolerance`` pairwise
    # to the first four terms before consulting steps, parameters, and order.
    return (*values, -actual_steps, -parameter_count, -order)


def _compare_global_candidates(
    left_id: str,
    right_id: str,
    flattened: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    tie_tolerance: float,
) -> int:
    left = [_result_selection_record(flattened[left_id][task]) for task in TASKS]
    right = [_result_selection_record(flattened[right_id][task]) for task in TASKS]
    left_values = (
        float(np.mean([row["primary"] for row in left])),
        float(min(row["primary"] for row in left)),
        float(np.mean([row["coupling_only"] for row in left])),
        -float(np.mean([row["combined_cross_entropy"] for row in left])),
    )
    right_values = (
        float(np.mean([row["primary"] for row in right])),
        float(min(row["primary"] for row in right)),
        float(np.mean([row["coupling_only"] for row in right])),
        -float(np.mean([row["combined_cross_entropy"] for row in right])),
    )
    for left_value, right_value in zip(left_values, right_values, strict=True):
        if abs(left_value - right_value) > float(tie_tolerance):
            return 1 if left_value > right_value else -1
    left_steps = float(np.mean([row["actual_optimizer_steps"] for row in left]))
    right_steps = float(np.mean([row["actual_optimizer_steps"] for row in right]))
    if left_steps != right_steps:
        return 1 if left_steps < right_steps else -1
    left_parameters = float(np.mean([row["parameter_count"] for row in left]))
    right_parameters = float(np.mean([row["parameter_count"] for row in right]))
    if left_parameters != right_parameters:
        return 1 if left_parameters < right_parameters else -1
    left_order = CANDIDATE_IDS.index(left_id)
    right_order = CANDIDATE_IDS.index(right_id)
    if left_order == right_order:
        return 0
    return 1 if left_order < right_order else -1


def select_global_candidate(
    candidate_results: Mapping[Any, Any] | Sequence[Mapping[str, Any]],
    optimization_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one candidate across both tasks without seeing development values."""

    flattened = _flatten_candidate_results(candidate_results)
    expected_seed = int(
        (optimization_config or _REGISTERED_CONTRACT)["execution"]["seed"]
    )
    for candidate_id in CANDIDATE_IDS:
        for task_id in TASKS:
            result = flattened[candidate_id][task_id]
            expected_identity = {
                "task_id": task_id,
                "candidate_id": candidate_id,
                "seed": expected_seed,
                "variant": VARIANT,
                "status": "completed",
                "protected_open": False,
                "protected_measured_access_count": 0,
                "development_used": False,
                "development_values_seen": False,
            }
            for key, expected in expected_identity.items():
                if _canonical_json(result.get(key)) != _canonical_json(expected):
                    raise ValueError(
                        f"candidate result {task_id}/{candidate_id} {key} provenance drifted"
                    )
            for key in (
                "all_fit_parameter_codes_active",
                "complete_registered_task_support",
                "derangement_nonoverlap_verified",
            ):
                if not isinstance(result.get(key), (bool, np.bool_)):
                    raise ValueError(
                        f"candidate result {task_id}/{candidate_id} lacks boolean {key}"
                    )
            for key in ("pretrain_steps", "vq_steps", "head_steps"):
                _require_integral(
                    result.get(key),
                    f"candidate result {task_id}/{candidate_id}.{key}",
                )
            _require_integral(
                result.get("trainable_parameter_count"),
                f"candidate result {task_id}/{candidate_id}.trainable_parameter_count",
            )
    selection_contract = (optimization_config or {}).get(
        "checkpoint_selection", REGISTERED_CHECKPOINT_SELECTION
    )
    validity: dict[str, dict[str, Any]] = {}
    valid_candidate_ids: list[str] = []
    for candidate_id in CANDIDATE_IDS:
        reasons: list[str] = []
        for task_id in TASKS:
            result = flattened[candidate_id][task_id]
            record = _result_selection_record(result)
            if not all(math.isfinite(float(value)) for value in record.values()):
                reasons.append(f"{task_id}:non_finite_metric")
            if result.get("all_fit_parameter_codes_active") is False:
                reasons.append(f"{task_id}:inactive_fit_parameter_code")
            if result.get("derangement_nonoverlap_verified") is False:
                reasons.append(f"{task_id}:derangement_overlap")
            if result.get("complete_registered_task_support") is False:
                reasons.append(f"{task_id}:incomplete_support")
        validity[candidate_id] = {
            "rankable": not reasons,
            "reasons": reasons,
        }
        if not reasons:
            valid_candidate_ids.append(candidate_id)
    if not valid_candidate_ids:
        raise RuntimeError("no valid architecture candidate remains rankable")
    for required_control in (REFERENCE_CANDIDATE_ID, LONG_CONTROL_CANDIDATE_ID):
        if required_control not in valid_candidate_ids:
            raise RuntimeError(
                f"required comparison control {required_control} is not rankable"
            )
    tie_tolerance = float(selection_contract["numeric_tie_tolerance"])
    ranking = sorted(
        valid_candidate_ids,
        key=cmp_to_key(
            lambda left, right: _compare_global_candidates(
                left,
                right,
                flattened,
                tie_tolerance=tie_tolerance,
            )
        ),
        reverse=True,
    )
    proposed = ranking[0]
    reference_records = [
        _result_selection_record(flattened[REFERENCE_CANDIDATE_ID][task]) for task in TASKS
    ]
    proposed_records = [
        _result_selection_record(flattened[proposed][task]) for task in TASKS
    ]
    reference_mean = float(np.mean([row["primary"] for row in reference_records]))
    proposed_mean = float(np.mean([row["primary"] for row in proposed_records]))
    minimum_improvement = float(
        (optimization_config or {}).get("checkpoint_selection", REGISTERED_CHECKPOINT_SELECTION).get(
            "minimum_descriptive_improvement", 0.01
        )
    )
    used_reference = (
        proposed != REFERENCE_CANDIDATE_ID
        and proposed_mean - reference_mean < minimum_improvement
    )
    selected = proposed
    recommended = REFERENCE_CANDIDATE_ID if used_reference else proposed
    long_records = [
        _result_selection_record(flattened[LONG_CONTROL_CANDIDATE_ID][task])
        for task in TASKS
    ]
    long_mean = float(np.mean([row["primary"] for row in long_records]))
    proposed_class = str(_candidate_by_id(proposed)["comparison_class"])
    history_ids = tuple(
        candidate_id
        for candidate_id in ranking
        if str(_candidate_by_id(candidate_id)["comparison_class"]) == "architecture"
    )
    best_history_id = history_ids[0] if history_ids else None
    if best_history_id is None:
        history_mean: float | None = None
        architecture_improvement = False
    else:
        history_records = [
            _result_selection_record(flattened[best_history_id][task])
            for task in TASKS
        ]
        history_mean = float(np.mean([row["primary"] for row in history_records]))
        architecture_improvement = bool(
            history_mean >= reference_mean + minimum_improvement
            and history_mean >= long_mean + minimum_improvement
        )
    if proposed == LONG_CONTROL_CANDIDATE_ID:
        improvement_interpretation = "optimization_duration_improvement_not_architecture"
    elif architecture_improvement and proposed_class == "architecture":
        improvement_interpretation = "exploratory_history_architecture_improvement"
    elif architecture_improvement:
        improvement_interpretation = (
            "history_architecture_improvement_observed_but_not_global_numeric_best"
        )
    elif proposed_class == "objective" and not used_reference:
        improvement_interpretation = "objective_weight_improvement_not_history_architecture"
    elif used_reference:
        improvement_interpretation = "minimum_improvement_not_met_retain_reference"
    else:
        improvement_interpretation = "no_history_architecture_improvement"
    summary = []
    for candidate_id in CANDIDATE_IDS:
        records = [_result_selection_record(flattened[candidate_id][task]) for task in TASKS]
        candidate_validity = validity[candidate_id]

        def safe(value: float) -> float | None:
            return float(value) if math.isfinite(float(value)) else None

        summary.append(
            {
                "candidate_id": candidate_id,
                "candidate_role": _candidate_by_id(candidate_id)["role"],
                "comparison_class": _candidate_by_id(candidate_id)["comparison_class"],
                "rankable": bool(candidate_validity["rankable"]),
                "invalid_reasons": list(candidate_validity["reasons"]),
                "task_mean_coupling_plus_private_subject_equal_macro_f1": safe(
                    float(np.mean([row["primary"] for row in records]))
                ),
                "task_minimum_coupling_plus_private_subject_equal_macro_f1": safe(
                    float(min(row["primary"] for row in records))
                ),
                "task_mean_coupling_only_subject_equal_macro_f1": safe(
                    float(np.mean([row["coupling_only"] for row in records]))
                ),
                "task_mean_selection_combined_cross_entropy": safe(
                    float(np.mean([row["combined_cross_entropy"] for row in records]))
                ),
                "global_tie_break_tuple_raw": (
                    list(
                        _global_candidate_key(
                            candidate_id, flattened, tie_tolerance=tie_tolerance
                        )
                    )
                    if candidate_validity["rankable"]
                    else None
                ),
                "selected_for_descriptive_transfer": candidate_id == selected,
                "recommended": candidate_id == recommended,
            }
        )
    return {
        "schema": "lc_spvq_global_candidate_selection_v1",
        "selected_candidate_id": selected,
        "recommended_candidate_id": recommended,
        "proposed_candidate_id": proposed,
        "reference_candidate_id": REFERENCE_CANDIDATE_ID,
        "long_control_candidate_id": LONG_CONTROL_CANDIDATE_ID,
        "minimum_descriptive_improvement": minimum_improvement,
        "numeric_tie_tolerance": tie_tolerance,
        "proposed_minus_reference_primary": proposed_mean - reference_mean,
        "proposed_minus_long_control_primary": proposed_mean - long_mean,
        "best_history_candidate_id": best_history_id,
        "best_history_task_mean_primary": history_mean,
        "best_history_minus_reference_primary": (
            None if history_mean is None else history_mean - reference_mean
        ),
        "best_history_minus_long_control_primary": (
            None if history_mean is None else history_mean - long_mean
        ),
        "used_reference_due_to_minimum_improvement": bool(used_reference),
        "architecture_improvement": architecture_improvement,
        "improvement_interpretation": improvement_interpretation,
        "task_head_performance_qc_only": True,
        "coupling_endpoint_claim": False,
        "development_values_used": False,
        "ranking": ranking,
        "validity": validity,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Post-selection development evaluation and figures
# ---------------------------------------------------------------------------


def _load_torch_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch versions before the weights_only keyword
        return torch.load(path, map_location=device)


def _validated_development_checkpoint(
    candidate_dir: Path,
    checkpoint_relative: Any,
    *,
    task_id: str,
    candidate: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> Mapping[str, Any]:
    checkpoint_rel = Path(str(checkpoint_relative))
    if checkpoint_rel.is_absolute() or ".." in checkpoint_rel.parts:
        raise PermissionError("development checkpoint path must stay candidate-local")
    if checkpoint_rel.as_posix() != "checkpoints/head_best.pt":
        raise ValueError("development requires the registered head_best checkpoint")
    checkpoint_path = (candidate_dir / checkpoint_rel).resolve()
    try:
        checkpoint_path.relative_to(candidate_dir.resolve())
    except ValueError as exc:
        raise PermissionError("development checkpoint escaped candidate directory") from exc
    checkpoint = _load_torch_checkpoint(checkpoint_path, device)
    expected_checkpoint_fields = {
        "schema": "lc_spvq_architecture_optimization_checkpoint_v1",
        "task_id": str(task_id),
        "variant": VARIANT,
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_config_overrides": _candidate_override_payload(candidate),
        "seed": int(seed),
        "stage": "task_head",
        "protected_open": False,
        "development_used": False,
    }
    for key, expected in expected_checkpoint_fields.items():
        if _canonical_json(checkpoint.get(key)) != _canonical_json(expected):
            raise PermissionError(f"development checkpoint {key} provenance drifted")
    if not math.isfinite(float(checkpoint.get("fit_selection_score", float("nan")))):
        raise ValueError("development checkpoint selection score is not finite")
    for required_state in (
        "model_state",
        "lag_objective_state",
        "optimizer_state",
        "rng_state",
    ):
        if not isinstance(checkpoint.get(required_state), Mapping):
            raise ValueError(f"development checkpoint lacks {required_state}")
    if checkpoint.get("quantization_strength") is None or checkpoint.get(
        "posterior_temperature"
    ) is None:
        raise ValueError("development checkpoint lacks the serialized VQ surface")
    return checkpoint


def evaluate_development_candidate(
    development: OptimizationDevelopmentTask,
    base_config: Mapping[str, Any],
    optimization_config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    candidate_dir: Path,
    seed: int,
    device: torch.device,
    result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    """Evaluate a restored frozen head on development after selection only."""

    if not isinstance(
        development._selection_capability, _DevelopmentSelectionPermit
    ):
        raise PermissionError("development object lacks post-selection capability")
    if (
        development._selection_capability.global_selection_digest
        != development.global_selection_digest
    ):
        raise PermissionError("development object decision digest drifted")
    if len(development.global_selection_digest) != 64:
        raise PermissionError("development object lacks decision provenance")
    development._selection_capability.consume_application(
        development.task_id,
        str(candidate["candidate_id"]),
        development.global_selection_digest,
    )
    runtime = candidate_runtime_config(
        base_config, optimization_config, candidate, seed=int(seed)
    )
    checkpoint = _validated_development_checkpoint(
        candidate_dir,
        result["checkpoints"]["head_best"],
        task_id=development.task_id,
        candidate=candidate,
        seed=seed,
        device=device,
    )
    model = reviewed._lc_spvq_model(development, runtime).to(device)
    model._lc_spvq_task_id = development.task_id
    lag_module = reviewed._lag_matching_module(runtime).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    lag_module.load_state_dict(checkpoint["lag_objective_state"], strict=True)
    model.set_quantization_strength(float(checkpoint["quantization_strength"]))
    model.set_posterior_temperature(float(checkpoint["posterior_temperature"]))
    model.eval()
    lag_module.eval()
    for parameter in tuple(model.parameters()) + tuple(lag_module.parameters()):
        parameter.requires_grad_(False)
    before_digest = _frozen_state_digest(model, lag_module)
    metrics, arrays, _ = reviewed._evaluate_lc_spvq(
        model,
        development.partition,
        config=runtime,
        device=device,
        seed=int(seed),
    )
    logits = torch.from_numpy(np.asarray(arrays["combined_logits"], dtype=np.float32))
    targets = torch.from_numpy(np.asarray(arrays["target"], dtype=np.int64))
    cross_entropy = float(torch.nn.functional.cross_entropy(logits, targets).item())
    after_digest = _frozen_state_digest(model, lag_module)
    if before_digest != after_digest:
        raise RuntimeError("frozen development apply changed model/VQ state")
    enriched = dict(metrics)
    enriched["development_combined_cross_entropy"] = cross_entropy
    enriched["development_apply_frozen"] = True
    enriched["state_digest_before"] = before_digest
    enriched["state_digest_after"] = after_digest
    enriched["quantization_strength"] = float(model.get_quantization_strength())
    enriched["posterior_temperature"] = float(model.get_posterior_temperature())
    return enriched, arrays


def _development_primary(metrics: Mapping[str, Any]) -> float:
    if "coupling_plus_private" in metrics:
        return float(metrics["coupling_plus_private"]["subject_equal_macro_f1"])
    if "development_primary_metric" in metrics:
        return float(metrics["development_primary_metric"])
    raise KeyError("development metrics lack coupling-plus-private primary metric")


def _write_curve_figure(
    output_dir: Path,
    curve_rows: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, str]:
    """Write a truthful provisional three-panel selection/development figure."""

    source_rows: list[dict[str, Any]] = []
    source_rows.extend(dict(row) for row in curve_rows)
    source_rows.extend(dict(row) for row in development_rows)
    _write_csv_atomic(output_dir / "curve_figure_source_data.csv", source_rows)
    alt_text = (
        "Provisional three-panel LC-SPVQ architecture-optimization figure, single seed "
        f"{seed}. Panel A plots the fixed native-plus-lag fit-selection loss over cumulative representation optimizer steps, with continuous-pretraining and VQ stages drawn as disconnected segments. "
        "Panel B plots fit-selection coupling-plus-private subject-equal macro-F1 at "
        "head evaluation steps. Panel C shows the same selection primary metric for "
        "each task and post-selection development values for only the reference and "
        "globally selected candidate. Candidate identity is repeated by color, line style, "
        "and marker; task identity is repeated by line width and opacity. No confidence "
        "intervals are shown; this is not "
        "a journal or accessibility certification."
    )
    if (output_dir / "curve_figure_alt_text.txt").exists():
        raise FileExistsError("refusing overwrite: curve_figure_alt_text.txt")
    (output_dir / "curve_figure_alt_text.txt").write_text(alt_text + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), layout="constrained")
    colors = {
        candidate_id: color
        for candidate_id, color in zip(
            CANDIDATE_IDS,
            ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"),
            strict=True,
        )
    }
    markers = dict(zip(CANDIDATE_IDS, ("o", "s", "^", "D", "P"), strict=True))
    linestyles = dict(
        zip(CANDIDATE_IDS, ("-", "--", "-.", ":", (0, (5, 1))), strict=True)
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in curve_rows:
        grouped.setdefault((str(row.get("task_id", "")), str(row.get("candidate_id", ""))), []).append(row)
    for (task_id, candidate_id), rows in sorted(grouped.items()):
        task_linewidth = 1.4 if task_id == TASKS[0] else 2.4
        task_alpha = 0.72 if task_id == TASKS[0] else 1.0
        representation = [row for row in rows if str(row.get("curve_kind", "")) == "representation"]
        head = [row for row in rows if str(row.get("curve_kind", "")) == "head"]
        if representation:
            for stage_index, stage_name in enumerate(
                ("continuous_pretrain_eval", "vq_anneal_eval")
            ):
                stage_rows = [
                    row for row in representation if str(row.get("stage")) == stage_name
                ]
                if not stage_rows:
                    continue
                axes[0].plot(
                    [int(row["step"]) for row in stage_rows],
                    [
                        float(row["selection_representation_loss"])
                        for row in stage_rows
                    ],
                    color=colors.get(candidate_id, "#333333"),
                    marker=markers.get(candidate_id, "o"),
                    linestyle=linestyles.get(candidate_id, "-"),
                    linewidth=task_linewidth,
                    alpha=task_alpha,
                    label=(
                        f"{task_id}/{candidate_id}"
                        if stage_index == 0
                        else "_nolegend_"
                    ),
                )
        if head:
            axes[1].plot(
                [int(row["step"]) for row in head],
                [float(row["selection_head_macro_f1"]) for row in head],
                color=colors.get(candidate_id, "#333333"),
                marker=markers.get(candidate_id, "o"),
                linestyle=linestyles.get(candidate_id, "-"),
                linewidth=task_linewidth,
                alpha=task_alpha,
                label=f"{task_id}/{candidate_id}",
            )

    task_positions = {task_id: index for index, task_id in enumerate(TASKS)}
    for candidate_id in CANDIDATE_IDS:
        rows = [row for row in curve_rows if str(row.get("candidate_id")) == candidate_id]
        for row in rows:
            if str(row.get("curve_kind")) != "architecture_selection":
                continue
            x = task_positions[str(row["task_id"])]
            axes[2].scatter(
                x,
                float(row["selection_primary_metric"]),
                color=colors[candidate_id],
                marker=markers[candidate_id],
                label=f"selection {candidate_id}",
            )
    for row in development_rows:
        x = task_positions[str(row["task_id"])]
        candidate_id = str(row["candidate_id"])
        axes[2].scatter(
            x,
            float(row["development_primary_metric"]),
            facecolors="none",
            edgecolors=colors.get(candidate_id, "#333333"),
            marker=markers.get(candidate_id, "o"),
            linewidths=1.5,
            label=f"development {candidate_id}",
        )

    axes[0].set(title="Fit-selection representation", xlabel="Cumulative optimizer step", ylabel="Fixed native + lag loss")
    axes[1].set(title="Fit-selection head", xlabel="Evaluation step", ylabel="Subject-equal macro-F1")
    axes[2].set(title="Selection / development", xlabel="Task", ylabel="Primary metric", xticks=list(task_positions.values()), xticklabels=list(task_positions))
    for axis in axes:
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(fontsize=7, loc="best")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(fontsize=7, loc="best")
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        # De-duplicate labels while retaining redundant marker/color encodings.
        seen: set[str] = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels, strict=True):
            if label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
        axes[2].legend(unique_handles, unique_labels, fontsize=6, loc="best")
    fig.suptitle(f"LC-SPVQ architecture optimization — single seed {seed}")
    paths: dict[str, str] = {}
    for extension in ("png", "pdf"):
        path = output_dir / f"selection_development_curves.{extension}"
        if path.exists():
            plt.close(fig)
            raise FileExistsError(f"refusing overwrite: {path}")
        temporary = output_dir / f".selection_development_curves.{os.getpid()}.tmp.{extension}"
        fig.savefig(temporary, format=extension, dpi=180)
        os.replace(temporary, path)
        paths[extension] = path.name
    plt.close(fig)
    manifest = {
        "schema": "lc_spvq_architecture_optimization_figure_manifest_v1",
        "figure_status": "provisional_general_qc_not_journal_certified",
        "single_seed": True,
        "seed": int(seed),
        "uncertainty_displayed": False,
        "confidence_intervals_displayed": False,
        "journal_certified": False,
        "accessibility_certified": False,
        "source_data": "curve_figure_source_data.csv",
        "alt_text": "curve_figure_alt_text.txt",
        "panels": [
            "fit-selection representation loss",
            "fit-selection head subject-equal macro-F1",
            "selection and post-selection development primary metric",
        ],
        "encodings": (
            "candidate identity uses redundant color, line style, and marker shape; "
            "task identity also uses line width and opacity"
        ),
        "files": paths,
    }
    _write_json_atomic(output_dir / "curve_figure_manifest.json", manifest)
    return paths


# ---------------------------------------------------------------------------
# Orchestration, audit artifacts, and atomic top-level publication
# ---------------------------------------------------------------------------


def _partition_registry_rows(
    task_id: str,
    partition: reviewed.PreparedPartition,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    donor = np.asarray(partition.donor_index, dtype=np.int64)
    for index, sample_id in enumerate(partition.sample_id):
        donor_index = int(donor[index])
        rows.append(
            {
                "task_id": str(task_id),
                "partition": str(partition.role),
                "row_index": int(index),
                "sample_id": str(sample_id),
                "subject": str(partition.subject[index]),
                "condition": str(partition.condition[index]),
                "record_id": str(partition.record_id[index]),
                "eeg_event_time_ms": float(partition.eeg_event_time_ms[index]),
                "fnirs_event_time_ms": float(partition.fnirs_event_time_ms[index]),
                "donor_index": donor_index,
                "donor_sample_id": str(partition.sample_id[donor_index]),
                "sample_registry_seed": int(
                    _REGISTERED_CONTRACT["execution"]["sample_registry_seed"]
                ),
                "derangement_seed": int(
                    _REGISTERED_CONTRACT["execution"]["derangement_seed"]
                ),
            }
        )
    return rows


def _registry_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(list(rows)).encode("utf-8")).hexdigest()


def _sample_identity_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            key: row[key]
            for key in (
                "task_id",
                "partition",
                "row_index",
                "sample_id",
                "subject",
                "condition",
                "record_id",
                "eeg_event_time_ms",
                "fnirs_event_time_ms",
            )
        }
        for row in rows
    ]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _donor_registry_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "task_id": row["task_id"],
            "partition": row["partition"],
            "row_index": row["row_index"],
            "sample_id": row["sample_id"],
            "donor_index": row["donor_index"],
            "donor_sample_id": row["donor_sample_id"],
        }
        for row in rows
    ]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _fit_audit_rows(task: Any, base_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset_id = str(task.dataset_id)
    rows = []
    for role, partition, split_role in (
        ("fit_parameter", task.parameter, "fit_parameter_subjects"),
        ("fit_selection", task.selection, "fit_selection_subjects"),
    ):
        rows.append(
            {
                "task_id": str(task.task_id),
                "partition": role,
                "configured_subject_count": len(base_config["data_split"][split_role][dataset_id]),
                "prepared_subject_count": len(set(map(str, partition.subject))),
                "samples_per_subject_class": SAMPLES_PER_SUBJECT_CLASS,
                "sample_count": len(partition.sample_id),
                "measured_sample_access_count": int(len(partition.sample_id)),
                "protected_measured_access_count": int(task.protected_measured_access_count),
                "materialized_before_global_selection": True,
                "development_materialized": False,
            }
        )
    return rows


def _development_audit_row(
    task: OptimizationDevelopmentTask,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_id = str(task.dataset_id)
    return {
        "task_id": str(task.task_id),
        "partition": "development_apply",
        "configured_subject_count": len(base_config["data_split"]["development_apply_subjects"][dataset_id]),
        "prepared_subject_count": len(set(map(str, task.partition.subject))),
        "samples_per_subject_class": SAMPLES_PER_SUBJECT_CLASS,
        "sample_count": len(task.partition.sample_id),
        "measured_sample_access_count": int(task.measured_access_count),
        "protected_measured_access_count": int(task.protected_measured_access_count),
        "materialized_before_global_selection": False,
        "development_materialized": True,
        "global_selection_digest": str(task.global_selection_digest),
        "post_selection_capability_verified": isinstance(
            task._selection_capability, _DevelopmentSelectionPermit
        ),
    }


def _result_curve_rows(
    result: Mapping[str, Any],
    candidate: Mapping[str, Any],
    task_id: str,
) -> list[dict[str, Any]]:
    rows = []
    for row in result.get("step_curves", ()):
        stage = str(row.get("stage", ""))
        if stage.endswith("_eval"):
            if stage in {"continuous_pretrain_eval", "vq_anneal_eval"}:
                rows.append(
                    {
                        "task_id": str(task_id),
                        "candidate_id": str(candidate["candidate_id"]),
                        "candidate_role": str(candidate["role"]),
                        "seed": int(result.get("seed", 0)),
                        "curve_kind": "representation",
                        "stage": stage,
                        "stage_step": int(row["step"]),
                        "step": (
                            int(row["step"])
                            if stage == "continuous_pretrain_eval"
                            else int(result.get("pretrain_steps", 0)) + int(row["step"])
                        ),
                        "selection_representation_loss": float(row["selection_fixed_native_plus_lag_loss"]),
                        "selection_head_macro_f1": "",
                        "selection_primary_metric": "",
                        "development_primary_metric": "",
                    }
                )
            elif stage == "task_head_eval":
                rows.append(
                    {
                        "task_id": str(task_id),
                        "candidate_id": str(candidate["candidate_id"]),
                        "candidate_role": str(candidate["role"]),
                        "seed": int(result.get("seed", 0)),
                        "curve_kind": "head",
                        "stage": stage,
                        "stage_step": int(row["step"]),
                        "step": int(row["step"]),
                        "selection_representation_loss": float(row["selection_fixed_native_plus_lag_loss"]),
                        "selection_head_macro_f1": float(row["selection_coupling_plus_private_subject_equal_macro_f1"]),
                        "selection_primary_metric": "",
                        "development_primary_metric": "",
                    }
                )
    rows.append(
        {
            "task_id": str(task_id),
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_role": str(candidate["role"]),
            "seed": int(result.get("seed", 0)),
            "curve_kind": "architecture_selection",
            "stage": "global_candidate_selection_input",
            "stage_step": "",
            "step": "",
            "selection_representation_loss": float(
                result.get(
                    "selection_fixed_native_plus_lag_loss",
                    result.get("selection_total_registered_pretraining_loss", np.nan),
                )
            ),
            "selection_head_macro_f1": float(result.get("selection_primary_metric", np.nan)),
            "selection_primary_metric": float(_result_metric(result, "selection_primary_metric")),
            "development_primary_metric": "",
        }
    )
    return rows


def _candidate_summary_rows(
    candidate_results: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for task_id in TASKS:
        for candidate in CANDIDATES:
            result = candidate_results[(task_id, str(candidate["candidate_id"]))]
            record = _result_selection_record(result)
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_id": str(candidate["candidate_id"]),
                    "candidate_role": str(candidate["role"]),
                    "comparison_class": str(candidate["comparison_class"]),
                    "eeg_shared_history_tokens": int(
                        candidate["eeg_shared_history_tokens"]
                    ),
                    "fnirs_shared_history_tokens": int(
                        candidate["fnirs_shared_history_tokens"]
                    ),
                    "lag_loss_weight": float(candidate["lag_loss_weight"]),
                    "step_multiplier": int(candidate["step_multiplier"]),
                    "seed": int(result.get("seed", 0)),
                    "selection_fixed_native_plus_lag_loss": (
                        record["representation_loss"]
                        if math.isfinite(record["representation_loss"])
                        else None
                    ),
                    "selection_coupling_plus_private_subject_equal_macro_f1": (
                        record["primary"] if math.isfinite(record["primary"]) else None
                    ),
                    "selection_coupling_only_subject_equal_macro_f1": (
                        record["coupling_only"]
                        if math.isfinite(record["coupling_only"])
                        else None
                    ),
                    "selection_combined_cross_entropy": (
                        record["combined_cross_entropy"]
                        if math.isfinite(record["combined_cross_entropy"])
                        else None
                    ),
                    "actual_optimizer_steps": record["actual_optimizer_steps"],
                    "total_parameter_count": record["parameter_count"],
                    "development_used_during_candidate_training": False,
                }
            )
    return rows


def orchestrate_optimization(
    optimization_config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    output_dir: Path,
    *,
    device: torch.device,
    fit_tasks: Mapping[str, Any] | None = None,
    candidate_runner: Callable[..., Mapping[str, Any]] | None = None,
    development_preparer: Callable[..., Any] | None = None,
    development_evaluator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run fit-only candidate cells, select globally, then apply development.

    ``fit_tasks`` and callback arguments are explicit seams for focused tests;
    the production defaults perform measured preparation/training/evaluation.
    """

    validate_optimization_config(optimization_config, base_config=base_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(optimization_config["execution"]["seed"])
    runner = train_candidate if candidate_runner is None else candidate_runner
    prepare_dev = prepare_development_task if development_preparer is None else development_preparer
    evaluate_dev = evaluate_development_candidate if development_evaluator is None else development_evaluator

    fit_by_task: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    fit_registry_rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        fit_task = (
            fit_tasks[task_id]
            if fit_tasks is not None
            else prepare_fit_selection_task(
                optimization_config,
                task_id,
                base_config=base_config,
                derangement_seed=int(optimization_config["execution"]["derangement_seed"]),
            )
        )
        if isinstance(fit_task, OptimizationFitTask):
            fit_task.validate_governance(base_config)
            audit_rows.extend(_fit_audit_rows(fit_task, base_config))
            fit_registry_rows.extend(
                _partition_registry_rows(task_id, fit_task.parameter)
            )
            fit_registry_rows.extend(
                _partition_registry_rows(task_id, fit_task.selection)
            )
        fit_by_task[task_id] = fit_task

    _write_csv_atomic(output_dir / "fit_selection_sample_registry.csv", fit_registry_rows)
    _write_json_atomic(
        output_dir / "fit_selection_sample_registry.json",
        {
            "schema": "lc_spvq_optimization_sample_registry_v1",
            "phase": "fit_parameter_and_fit_selection",
            "rows": fit_registry_rows,
            "registry_sha256": _registry_digest(fit_registry_rows),
            "sample_identity_sha256": _sample_identity_digest(fit_registry_rows),
            "donor_registry_sha256": _donor_registry_digest(fit_registry_rows),
            "development_included": False,
        },
    )

    candidate_results: dict[tuple[str, str], Mapping[str, Any]] = {}
    curve_rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        for candidate in CANDIDATES:
            candidate_id = str(candidate["candidate_id"])
            candidate_dir = output_dir / "tasks" / task_id / "candidates" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            result = dict(
                runner(
                    fit_by_task[task_id],
                    base_config,
                    optimization_config,
                    candidate,
                    seed=seed,
                    device=device,
                    output_dir=candidate_dir,
                )
            )
            expected_result_identity = {
                "task_id": task_id,
                "candidate_id": candidate_id,
                "candidate_role": candidate["role"],
                "seed": seed,
                "variant": VARIANT,
                "status": "completed",
                "protected_open": False,
                "protected_measured_access_count": 0,
                "development_used": False,
                "development_values_seen": False,
            }
            for key, expected in expected_result_identity.items():
                if _canonical_json(result.get(key)) != _canonical_json(expected):
                    raise ValueError(
                        f"candidate result {task_id}/{candidate_id} {key} drifted"
                    )
            candidate_results[(task_id, candidate_id)] = result
            result_path = candidate_dir / "result.json"
            if not result_path.exists():
                _write_json_atomic(result_path, result)
            curve_rows.extend(_result_curve_rows(result, candidate, task_id))

    selection = select_global_candidate(
        {
            (task_id, candidate_id): result
            for (task_id, candidate_id), result in candidate_results.items()
        },
        optimization_config,
    )
    selected_id = str(selection["selected_candidate_id"])
    reference_id = REFERENCE_CANDIDATE_ID
    _write_json_atomic(output_dir / "global_candidate_selection.json", selection)
    global_selection_digest = hashlib.sha256(
        _canonical_json(selection).encode("utf-8")
    ).hexdigest()
    development_permit = _issue_development_selection_permit(
        selection,
        output_dir / "global_candidate_selection.json",
    )
    summary_rows = _candidate_summary_rows(candidate_results)
    _write_csv_atomic(output_dir / "candidate_summary.csv", summary_rows)
    _write_json_atomic(
        output_dir / "candidate_summary.json",
        {"schema": "lc_spvq_candidate_summary_v1", "rows": summary_rows, "selection": selection},
    )
    _write_csv_atomic(output_dir / "step_curves.csv", curve_rows)

    # This is the only point at which development preparation is reachable.
    development_rows: list[dict[str, Any]] = []
    development_metrics: dict[str, Any] = {}
    development_prepared_tasks: dict[str, Any] = {}
    development_registry_rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        development = prepare_dev(
            optimization_config,
            fit_by_task[task_id],
            base_config=base_config,
            derangement_seed=int(optimization_config["execution"]["derangement_seed"]),
            selection_capability=development_permit,
            global_selection_digest=global_selection_digest,
        )
        development_prepared_tasks[task_id] = development
        if isinstance(development, OptimizationDevelopmentTask):
            audit_rows.append(_development_audit_row(development, base_config))
            development_registry_rows.extend(
                _partition_registry_rows(task_id, development.partition)
            )
        evaluate_ids = (reference_id,) if reference_id == selected_id else (reference_id, selected_id)
        for candidate_id in evaluate_ids:
            candidate = _candidate_by_id(candidate_id)
            candidate_dir = output_dir / "tasks" / task_id / "candidates" / candidate_id
            result = candidate_results[(task_id, candidate_id)]
            evaluated = evaluate_dev(
                development,
                base_config,
                optimization_config,
                candidate,
                candidate_dir=candidate_dir,
                seed=seed,
                device=device,
                result=result,
            )
            if isinstance(evaluated, tuple) and len(evaluated) == 2:
                metrics, arrays = evaluated
            else:
                metrics, arrays = evaluated, {}
            primary = _development_primary(metrics)
            row = {
                "task_id": task_id,
                "candidate_id": candidate_id,
                "candidate_role": candidate["role"],
                "reference_candidate": candidate_id == reference_id,
                "globally_selected_candidate": candidate_id == selected_id,
                "evaluated_once_when_reference_equals_selected": reference_id == selected_id,
                "selection_primary_metric": _result_metric(result, "selection_primary_metric"),
                "development_primary_metric": primary,
                "development_combined_cross_entropy": float(
                    metrics.get("development_combined_cross_entropy", np.nan)
                ),
                "seed": seed,
            }
            development_rows.append(row)
            development_metrics.setdefault(task_id, {})[candidate_id] = {
                "metrics": metrics,
                "primary_metric": primary,
                "evaluated_once_when_reference_equals_selected": reference_id == selected_id,
            }
            prediction_dir = output_dir / "development" / task_id / candidate_id
            prediction_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(prediction_dir / "metrics.json", metrics)
            if arrays:
                _write_npz_atomic(
                    prediction_dir / "outputs.npz",
                    schema=np.asarray("lc_spvq_architecture_optimization_development_outputs_v1"),
                    task_id=np.asarray(task_id),
                    candidate_id=np.asarray(candidate_id),
                    seed=np.asarray(seed, dtype=np.int64),
                    **{str(key): value for key, value in arrays.items()},
                )

    production_development = all(
        isinstance(task, OptimizationDevelopmentTask)
        for task in development_prepared_tasks.values()
    )
    if production_development:
        expected_applications = {
            (task_id, candidate_id)
            for task_id in TASKS
            for candidate_id in (
                (reference_id,)
                if reference_id == selected_id
                else (reference_id, selected_id)
            )
        }
        if development_permit.applied_models != expected_applications:
            raise RuntimeError("development frozen-model application registry is incomplete")

    _write_csv_atomic(
        output_dir / "development_sample_registry.csv", development_registry_rows
    )
    _write_json_atomic(
        output_dir / "development_sample_registry.json",
        {
            "schema": "lc_spvq_optimization_sample_registry_v1",
            "phase": "post_selection_development_apply",
            "rows": development_registry_rows,
            "registry_sha256": _registry_digest(development_registry_rows),
            "sample_identity_sha256": _sample_identity_digest(
                development_registry_rows
            ),
            "donor_registry_sha256": _donor_registry_digest(
                development_registry_rows
            ),
            "materialized_after_global_decision": True,
        },
    )
    _write_csv_atomic(output_dir / "development_comparison.csv", development_rows)
    _write_json_atomic(
        output_dir / "development_comparison.json",
        {
            "schema": "lc_spvq_development_comparison_v1",
            "selection": selection,
            "global_selection_digest": global_selection_digest,
            "development_values_used_for_selection": False,
            "application_count_definition": optimization_config["frozen_apply"][
                "application_count_definition"
            ],
            "distinct_models_per_task": 1 if reference_id == selected_id else 2,
            "total_frozen_model_applications": len(development_rows),
            "permit_registered_applications": (
                len(development_permit.applied_models)
                if production_development
                else None
            ),
            "unselected_candidate_applications": 0,
            "rows": development_rows,
            "metrics": development_metrics,
        },
    )
    _write_csv_atomic(output_dir / "sample_access_audit.csv", audit_rows)
    _write_json_atomic(
        output_dir / "sample_access_audit.json",
        {
            "schema": "lc_spvq_architecture_optimization_sample_access_audit_v1",
            "rows": audit_rows,
            "fit_only_preparation_completed_before_global_selection": True,
            "development_preparation_completed_after_global_selection": True,
            "protected_measured_access_count": int(
                sum(int(row.get("protected_measured_access_count", 0)) for row in audit_rows)
            ),
            "protected_open": False,
        },
    )

    _write_curve_figure(
        output_dir,
        curve_rows,
        development_rows,
        seed=seed,
    )
    return {
        "schema": "lc_spvq_architecture_optimization_orchestration_v1",
        "selection": selection,
        "candidate_results": candidate_results,
        "development_rows": development_rows,
        "development_metrics": development_metrics,
        "sample_access_audit": audit_rows,
        "curve_rows": curve_rows,
        "development_prepared_tasks": development_prepared_tasks,
    }


def _input_inventory(opt_path: Path, base_path: Path) -> list[dict[str, Any]]:
    paths: list[tuple[Path, str]] = [
        (opt_path, "optimization_configuration"),
        (base_path, "base_configuration"),
        (REGISTERED_PROTOCOL_PATH, "frozen_optimization_protocol"),
        (Path(__file__).resolve(), "runner"),
    ]
    module_paths = (
        reviewed,
        reviewed.UnifiedPhysiologyWindowDataset,
        reviewed.LagConditionedTaskDataset,
        reviewed.LCSPVQModel,
        reviewed.LagAwareContinuousMatchingLoss,
        reviewed.native_feature_prediction_loss,
        reviewed.evaluate_logit_ablations,
    )
    for module_or_object in module_paths:
        module_file = getattr(module_or_object, "__file__", None)
        if module_file:
            paths.append((Path(module_file).resolve(), "reviewed_runtime_module"))
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cache_root = _resolve(str(base["source"]["cache_root"]))
    paths.extend(
        [
            (cache_root / "cache_manifest.json", "cache_manifest"),
            (
                cache_root / "eeg_artifact_clean_v4" / "cache_manifest.json",
                "cache_manifest",
            ),
            (
                cache_root / "simultaneous_eeg_eog_clean_v1" / "cache_manifest.json",
                "cache_manifest",
            ),
        ]
    )
    output: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path, kind in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(f"provenance input is missing: {resolved}")
        try:
            display = str(resolved.relative_to(REPO_ROOT))
        except ValueError:
            display = str(resolved)
        output.append({"path": display, "sha256": _sha256(resolved), "source_kind": kind})
    return output


def run_optimization(
    config: Mapping[str, Any],
    config_path: Path,
    target: Path,
    *,
    requested_device: str | None = None,
) -> Path:
    """Validate, atomically stage, and publish one optimization run."""

    opt_path, base_path, base_config = _validate_bound_config(config, config_path)
    git_before = _git_payload()
    if git_before["commit"] == "unknown" or git_before["status_short"]:
        raise RuntimeError(
            "measured optimization requires a committed clean worktree before access"
        )
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    device = torch.device(requested_device or str(config["execution"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _set_seed(int(config["execution"]["seed"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        shutil.copy2(opt_path, staging / "optimization_config.yaml")
        shutil.copy2(base_path, staging / "base_config.yaml")
        orchestration = orchestrate_optimization(
            config,
            base_config,
            staging,
            device=device,
        )
        inputs = _input_inventory(opt_path, base_path)
        manifest = {
            "schema": "lc_spvq_architecture_optimization_manifest_v1",
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": config["experiment"],
            "tasks": list(TASKS),
            "variant": VARIANT,
            "seed": int(config["execution"]["seed"]),
            "seeds": [int(config["execution"]["seed"])],
            "single_seed": True,
            "candidate_ids": list(CANDIDATE_IDS),
            "selected_candidate_id": orchestration["selection"]["selected_candidate_id"],
            "recommended_candidate_id": orchestration["selection"][
                "recommended_candidate_id"
            ],
            "improvement_interpretation": orchestration["selection"][
                "improvement_interpretation"
            ],
            "architecture_improvement": orchestration["selection"][
                "architecture_improvement"
            ],
            "development_values_used_for_selection": False,
            "global_selection_digest": hashlib.sha256(
                _canonical_json(orchestration["selection"]).encode("utf-8")
            ).hexdigest(),
            "post_selection_development": True,
            "frozen_development_application_count": len(
                orchestration["development_rows"]
            ),
            "protected_open": False,
            "protected_closed": True,
            "protected_measured_access_count": int(
                sum(int(row.get("protected_measured_access_count", 0)) for row in orchestration["sample_access_audit"])
            ),
            "old_2_of_16_immutable": True,
            "old_continuous_2_of_16_verdict_mutable": False,
            "claims": {
                "single_seed": True,
                "post_selection_development": True,
                "protected_closed": True,
                "old_2_of_16_immutable": True,
                "development_not_seen_by_candidate_training": True,
                "task_head_performance_qc_only": True,
                "coupling_endpoint_claim": False,
                "m1_vs_n1_claim": False,
                "m1_vs_b0_claim": False,
                "multi_seed_validation": False,
            },
            "git": git_before,
            "inputs": inputs,
            "artifacts": _artifact_inventory(staging),
            "figure_status": config["output"]["figure_status"],
            "journal_certified": False,
            "accessibility_certified": False,
        }
        _write_json_atomic(staging / "manifest.json", manifest)
        os.replace(staging, target)
        return target
    except Exception:
        # Retaining a failed staging directory preserves forensic evidence and
        # does not publish a partial target.
        raise


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_optimization_config(config, config_path=config_path)
    target = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else _resolve(str(config["output"]["root"]))
    )
    return run_optimization(
        config,
        config_path,
        target,
        requested_device=args.device,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REGISTERED_CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory; defaults deterministically to output.root",
    )
    parser.add_argument("--device", help="optional torch device override")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(run(parse_args()))
