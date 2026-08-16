#!/usr/bin/env python3
"""Public-only BrainFusion protocol, window, and component diagnostics.

This module is deliberately separate from the protected campaign runner.  It
only reads the immutable public tensor caches and public split manifests, and
it refuses paths containing ``protected``.  The diagnostics are intended to
answer three first-round questions:

* F1: how much of the paper/project gap is attributable to the evaluation
  protocol (source-like paper claim versus the public strict split)?
* F2: how much usable NVC signal remains as the *observed* window is shortened
  to 2/4/6/8 seconds, and is an extended post-stimulus window available?
* F3: what is the public strict-split contribution of EEG, HbO, HbR, NVC, and
  the fold-local stack?

No value from this script is table-admissible.  In particular, the script does
not open the protected test and does not select a method/window using protected
outcomes.  Prefix-window results are diagnostics of the existing 8-second
cache; they are not an experiment with post-window fNIRS context.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "comparative_methods/runs/performance_analysis/20260816_p0/brainfusion_bridge"
)
CONFIG_PATH = METHOD_ROOT / "configs/public_development_v2.yaml"
ALIGNMENT_PATH = METHOD_ROOT / "configs/alignment_v2.yaml"
PUBLIC_RESULT_PATH = REPO_ROOT / "docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md"
PUBLIC_SUMMARY_PATH = METHOD_ROOT / "evidence/public_development_v2/matrix_completion_summary.json"
TENSOR_ROOT = METHOD_ROOT / "runs/tensor_cache_v2"

for import_path in (REPO_ROOT, METHOD_ROOT, METHOD_ROOT / "adapters"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import load_config as load_alignment_config
from alignment_data import load_public_inventory
from brainfusion_gpu.nvc import NVCConfig, brainfusion_nvc_contribution_timeseries
import run_public_development_v2 as public_runner


SCHEMA = "brainfusion_performance_analysis_bridge_v1"
SUPPORTED_TASKS = ("motor_imagery", "mental_arithmetic", "wg", "nback", "visual")
WINDOWS_S = (2.0, 4.0, 6.0, 8.0)
STRICT_VALUES = {
    "motor_imagery": 0.5493,
    "mental_arithmetic": 0.5502,
    "wg": 0.5428,
    "nback": 0.3337,
    "visual": 0.2222,
}
TASK_DISPLAY = {
    "motor_imagery": "Motor imagery",
    "mental_arithmetic": "Mental arithmetic",
    "wg": "Word generation",
    "nback": "N-back",
    "visual": "Visual",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(path: Path, *, allow_repo_root: bool = True) -> Path:
    """Resolve a path and fail closed on protected artifacts."""

    resolved = path.expanduser().resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"protected artifact is not allowed: {resolved}")
    if allow_repo_root and REPO_ROOT not in resolved.parents and resolved != REPO_ROOT:
        raise PermissionError(f"diagnostic path is outside the repository: {resolved}")
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class CacheRecord:
    task: str
    tensor_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    payload: Mapping[str, torch.Tensor]
    sample_ids: tuple[str, ...]
    groups: tuple[str, ...]
    dataset_indices: tuple[int, ...]
    inventory: Any


def _single_manifest(task: str) -> tuple[Path, Path]:
    root = _safe_path(TENSOR_ROOT / task)
    manifests = sorted(root.glob("*.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected exactly one public tensor manifest for {task}, found {manifests}")
    manifest_path = _safe_path(manifests[0])
    tensor_path = _safe_path(manifest_path.with_suffix(".pt"))
    if not tensor_path.is_file():
        raise RuntimeError(f"tensor cache is missing for {task}: {tensor_path}")
    return tensor_path, manifest_path


def _load_cache(task: str, alignment: Mapping[str, Any]) -> CacheRecord:
    tensor_path, manifest_path = _single_manifest(task)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "brainfusion_full_public_tensor_cache_v2":
        raise RuntimeError(f"unexpected tensor cache schema for {task}")
    if manifest.get("protected_test_opened") is not False:
        raise PermissionError(f"tensor cache reports protected access for {task}")
    identity = manifest.get("identity", {})
    if identity.get("fitted_or_supervised_state_included") is not False:
        raise RuntimeError(f"tensor cache includes fitted/supervised state for {task}")
    if identity.get("protected_test_opened") is not False:
        raise PermissionError(f"tensor cache identity reports protected access for {task}")
    file_sha = _sha256(tensor_path)
    if file_sha != manifest.get("file_sha256"):
        raise RuntimeError(f"tensor cache hash drifted for {task}")

    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "eeg", "hbo", "hbr", "targets", "dataset_indices"
    }:
        raise RuntimeError(f"unexpected tensor cache fields for {task}")
    if any(not isinstance(payload[name], torch.Tensor) for name in payload):
        raise RuntimeError(f"tensor cache fields are not tensors for {task}")
    if any(not bool(torch.isfinite(payload[name]).all()) for name in ("eeg", "hbo", "hbr")):
        raise RuntimeError(f"non-finite modality values in {task} cache")

    # Loading the inventory reuses the already frozen public split registry and
    # supplies sample IDs/groups without opening any protected fold artifact.
    inventory = load_public_inventory(alignment, task=task)
    expected_indices = tuple(int(value) for value in inventory.indices)
    actual_indices = tuple(int(value) for value in payload["dataset_indices"].tolist())
    if actual_indices != expected_indices:
        raise RuntimeError(f"tensor cache dataset index order drifted for {task}")
    # ``lightweight_metadata.class_index`` is the source dataset's declared
    # class number.  The public adapter intentionally reindexes labels through
    # its task-local ``class_to_index`` mapping (notably for the three-class
    # n-back task), so use the same mapping as the registered runner.
    expected_targets = [
        int(
            inventory.dataset.class_to_index[
                str(inventory.dataset.lightweight_metadata(index)["condition"])
            ]
        )
        for index in expected_indices
    ]
    if payload["targets"].tolist() != expected_targets:
        raise RuntimeError(f"tensor cache targets drifted for {task}")
    sample_ids = tuple(
        str(inventory.sample_ids[position]) for position in range(len(expected_indices))
    )
    groups = tuple(
        str(inventory.dataset.lightweight_metadata(index)["subject"])
        for index in expected_indices
    )
    contract = identity.get("tensor_contract", {})
    duration = float(contract.get("duration_s", -1))
    if duration != 8.0 or float(contract.get("eeg_sample_rate_hz", -1)) != 200.0:
        raise RuntimeError(f"unexpected EEG window contract for {task}: {contract}")
    if float(contract.get("fnirs_sample_rate_hz", -1)) != 10.0:
        raise RuntimeError(f"unexpected fNIRS rate contract for {task}: {contract}")
    return CacheRecord(
        task=task,
        tensor_path=tensor_path,
        manifest_path=manifest_path,
        manifest=manifest,
        payload=payload,
        sample_ids=sample_ids,
        groups=groups,
        dataset_indices=expected_indices,
        inventory=inventory,
    )


def _cache_audit(record: CacheRecord) -> dict[str, Any]:
    payload = record.payload
    contract = record.manifest["identity"]["tensor_contract"]
    return {
        "task": record.task,
        "tensor_path": str(record.tensor_path.relative_to(REPO_ROOT)),
        "manifest_path": str(record.manifest_path.relative_to(REPO_ROOT)),
        "tensor_sha256": _sha256(record.tensor_path),
        "manifest_sha256": _sha256(record.manifest_path),
        "sample_count": int(payload["targets"].numel()),
        "subject_count": len(set(record.groups)),
        "class_counts": {
            str(key): int(value)
            for key, value in Counter(int(value) for value in payload["targets"].tolist()).items()
        },
        "eeg_shape": list(payload["eeg"].shape),
        "hbo_shape": list(payload["hbo"].shape),
        "hbr_shape": list(payload["hbr"].shape),
        "duration_s": float(contract["duration_s"]),
        "eeg_sample_rate_hz": float(contract["eeg_sample_rate_hz"]),
        "fnirs_sample_rate_hz": float(contract["fnirs_sample_rate_hz"]),
        "observed_interval_only": True,
        "extra_pre_anchor_context_s": 0.0,
        "extra_post_interval_context_s": 0.0,
        "fitted_or_supervised_state_included": bool(
            record.manifest["identity"]["fitted_or_supervised_state_included"]
        ),
        "protected_test_opened": bool(record.manifest["protected_test_opened"]),
    }


def _existing_result_summary(task: str) -> dict[str, Any]:
    public_mean = None
    public_sd = None
    if PUBLIC_SUMMARY_PATH.is_file():
        summary = json.loads(PUBLIC_SUMMARY_PATH.read_text(encoding="utf-8"))
        for item in summary.get("tasks", []):
            if item.get("task") == task:
                public_mean = float(item["public_validation_macro_f1_cell_mean"])
                public_sd = float(item["public_validation_macro_f1_cell_sd"])
                break
    return {
        "paper_case_reported_value": 0.955 if task == "motor_imagery" else None,
        "paper_case_metric": "within-subject participant-level accuracy" if task == "motor_imagery" else None,
        "public_development_mean_macro_f1": public_mean,
        "public_development_cell_sd": public_sd,
        "strict_cross_subject_macro_f1": STRICT_VALUES.get(task),
        "strict_value_source": str(PUBLIC_RESULT_PATH.relative_to(REPO_ROOT)),
        "table_admissible": False,
        "comparability_warning": (
            "Paper MI value is within-subject participant-level accuracy; project values are "
            "macro-F1 under public/strict cross-subject splits and therefore are not a numeric "
            "reproduction claim."
            if task == "motor_imagery"
            else "No source-case numerical value applies to this cross-task adaptation."
        ),
    }


def _batch_ranges(count: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, count, batch_size):
        yield start, min(start + batch_size, count)


def _nvc_batch_stats(
    eeg: torch.Tensor,
    hbo: torch.Tensor,
    hbr: torch.Tensor,
    labels: torch.Tensor,
    *,
    window_s: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    eeg_samples = int(round(window_s * 200.0))
    fnirs_samples = int(round(window_s * 10.0))
    if eeg_samples % 20 != 0 or eeg_samples <= 0:
        raise ValueError(f"window is not compatible with 20-sample EEG averaging: {window_s}")
    if eeg_samples > eeg.shape[-1] or fnirs_samples > hbo.shape[-1]:
        raise ValueError(f"window exceeds cached 8-second support: {window_s}")

    labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    means: list[np.ndarray] = []
    abs_means: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    correlations: list[np.ndarray] = []
    for start, stop in _batch_ranges(int(eeg.shape[0]), batch_size):
        batch_eeg = eeg[start:stop, :, :eeg_samples].to(device)
        batch_hbo = hbo[start:stop, :, :fnirs_samples].to(device)
        batch_hbr = hbr[start:stop, :, :fnirs_samples].to(device)
        _, _, corr = brainfusion_nvc_contribution_timeseries(
            batch_eeg, batch_hbo, batch_hbr,
            NVCConfig(eeg_window_samples=20),
        )
        corr_cpu = corr.detach().cpu()
        correlations.append(corr_cpu.numpy())
        means.append(corr_cpu.mean(dim=1).numpy())
        abs_means.append(corr_cpu.abs().mean(dim=1).numpy())
        maxima.append(corr_cpu.abs().amax(dim=1).numpy())
        del batch_eeg, batch_hbo, batch_hbr, corr
        if device.type == "cuda":
            torch.cuda.empty_cache()

    corr_all = np.concatenate(correlations, axis=0)
    mean_signed = np.concatenate(means)
    mean_abs = np.concatenate(abs_means)
    max_abs = np.concatenate(maxima)
    class_values: dict[str, Any] = {}
    class_ids = sorted(set(labels_np.tolist()))
    if len(class_ids) == 2:
        a, b = class_ids
        delta = mean_abs[labels_np == b].mean() - mean_abs[labels_np == a].mean()
        pooled = np.sqrt(
            0.5 * (
                mean_abs[labels_np == a].var(ddof=1)
                + mean_abs[labels_np == b].var(ddof=1)
            )
        )
        class_values = {
            "class_ids": [a, b],
            "mean_abs_by_class": {
                str(a): float(mean_abs[labels_np == a].mean()),
                str(b): float(mean_abs[labels_np == b].mean()),
            },
            "mean_abs_class_delta_b_minus_a": float(delta),
            "mean_abs_cohens_d_b_minus_a": float(delta / pooled) if pooled > 0 else None,
            "mean_abs_auc_class_b": float(
                roc_auc_score(labels_np, mean_abs)
            ),
        }
    pair_signed = corr_all.mean(axis=0)
    pair_abs = np.abs(corr_all).mean(axis=0)
    return {
        "window_s": float(window_s),
        "eeg_samples": eeg_samples,
        "fnirs_samples": fnirs_samples,
        "pair_count": int(corr_all.shape[1]),
        "pair_signed_mean": float(corr_all.mean()),
        "pair_signed_sd": float(corr_all.std(ddof=1)),
        "pair_abs_mean": float(corr_all.__abs__().mean()),
        "pair_abs_sd": float(corr_all.__abs__().std(ddof=1)),
        "sample_mean_signed_mean": float(mean_signed.mean()),
        "sample_mean_abs_mean": float(mean_abs.mean()),
        "sample_max_abs_mean": float(max_abs.mean()),
        "pair_signed_extrema": [float(pair_signed.min()), float(pair_signed.max())],
        "pair_abs_extrema": [float(pair_abs.min()), float(pair_abs.max())],
        "class_separation": class_values,
        "interpretation": (
            "Prefix-only NVC diagnostic on the cached canonical interval; no post-stimulus "
            "fNIRS samples were used."
        ),
    }


def _component_scores_for_fold(
    record: CacheRecord,
    alignment: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    # This uses exactly the public development fold membership.  It calls the
    # registered fold-local feature/stacking implementation and then exposes
    # validation scores for each selected base view; no validation labels enter
    # selection.
    inventory, train_indices, validation_indices, manifest_path, manifest_sha = public_runner._fold_membership(
        alignment, task=record.task, outer_fold=outer_fold
    )
    if tuple(inventory.indices) != record.dataset_indices:
        raise RuntimeError(f"public fold inventory/cache mismatch for {record.task}")
    payload = record.payload
    train = public_runner.fold_data_from_cache(payload, inventory, train_indices)
    validation = public_runner.fold_data_from_cache(payload, inventory, validation_indices)
    train_tensors = tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in train[:4])
    validation_tensors = tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in validation[:4])
    pipe = public_runner._pipeline(config, seed=seed, smoke=False)
    pipe.fit(*train_tensors, groups=train[5], sample_ids=train[4])
    values = pipe.features.transform(*validation_tensors[:3])
    scores: dict[str, Any] = {}
    y_val = validation[3].detach().cpu().numpy()
    for view, tensor in values.items():
        x_val = tensor.detach().cpu().numpy()
        scaler = pipe.stacking.view_scalers_[view]
        estimator = pipe.stacking.base_estimators_[view]
        prediction = estimator.predict(scaler.transform(x_val))
        scores[view] = {
            "macro_f1": float(f1_score(y_val, prediction, average="macro")),
            "selected_candidate": str(pipe.stacking.selected_candidates_[view]),
            "feature_dimension": int(x_val.shape[1]),
        }
    stack_prediction = pipe.predict(*validation_tensors[:3])
    scores["stack"] = {
        "macro_f1": float(f1_score(y_val, stack_prediction, average="macro")),
        "selected_candidate": "grouped_oof_linear_svm_meta",
        "feature_dimension": 2 * len(pipe.features.csps["eeg"].filters_),
    }
    audit = pipe.audit_state()
    if not audit["all_fitted_state_outer_training_only"]:
        raise RuntimeError("BrainFusion diagnostic pipeline reports non-train-local state")
    return {
        "task": record.task,
        "outer_fold": int(outer_fold),
        "seed": int(seed),
        "public_manifest_path": str(Path(manifest_path).resolve().relative_to(REPO_ROOT)),
        "public_manifest_sha256": str(manifest_sha),
        "train_sample_count": len(train_indices),
        "validation_sample_count": len(validation_indices),
        "train_subject_count": len(set(train[5])),
        "validation_subject_count": len(set(validation[5])),
        "component_scores": scores,
        "protected_test_opened": False,
        "table_admissible": False,
    }


def _protocol_capability() -> dict[str, Any]:
    return {
        "source_like_within_subject_10fold": {
            "status": "not_run",
            "reason": (
                "The paper-case per-participant CSP/AutoML/stacking implementation is not in "
                "the pinned public checkout; a source-like score would be an independent variant, "
                "not a numerical source reproduction."
            ),
            "paper_metric": "participant-level accuracy",
            "paper_mi_value": 0.955,
        },
        "group_safe_within_subject": {
            "status": "available_as_future_public_experiment",
            "reason": "Public tensor caches contain subject/session/event IDs, but no source-case exact pipeline.",
        },
        "strict_cross_subject_public_development": {
            "status": "available",
            "reason": "Five public folds and three fixed seeds are complete; results are not table-admissible.",
        },
        "strict_cross_subject_protected": {
            "status": "locked",
            "protected_test_opened": False,
        },
        "long_post_stimulus_fnirs": {
            "status": "unavailable",
            "available_max_s": 8.0,
            "extra_post_interval_context_s": 0.0,
            "reason": "The cache contract is canonical time-zero 8 s support; HRF convolution is local and cropped.",
        },
    }


def run(
    *,
    tasks: Sequence[str],
    output_root: Path,
    device: str,
    run_components: bool,
    component_tasks: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    output_root = _safe_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    alignment, checked_alignment_path = load_alignment_config(ALIGNMENT_PATH)
    if checked_alignment_path != ALIGNMENT_PATH.resolve():
        raise RuntimeError("alignment config path drifted")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")

    records: dict[str, CacheRecord] = {}
    audits: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    protocol = _protocol_capability()
    for task in tasks:
        record = _load_cache(task, alignment)
        records[task] = record
        audits.append(_cache_audit(record))
        for window_s in WINDOWS_S:
            stats = _nvc_batch_stats(
                record.payload["eeg"], record.payload["hbo"], record.payload["hbr"],
                record.payload["targets"], window_s=window_s, device=selected_device,
                batch_size=batch_size,
            )
            stats.update({"task": task, "display_name": TASK_DISPLAY[task]})
            window_rows.append(stats)

    if run_components:
        for task in component_tasks:
            if task not in records:
                record = _load_cache(task, alignment)
                records[task] = record
                audits.append(_cache_audit(record))
            record = records[task]
            for outer_fold in range(5):
                # One fixed public-development seed is enough for this first
                # mechanism pass; the frozen campaign itself has three seeds.
                result = _component_scores_for_fold(
                    record, alignment, config, outer_fold=outer_fold, seed=17,
                    device=selected_device,
                )
                for view, values in result["component_scores"].items():
                    component_rows.append(
                        {
                            "task": task,
                            "display_name": TASK_DISPLAY[task],
                            "outer_fold": result["outer_fold"],
                            "seed": result["seed"],
                            "view": view,
                            **values,
                            "train_subject_count": result["train_subject_count"],
                            "validation_subject_count": result["validation_subject_count"],
                            "protected_test_opened": False,
                            "table_admissible": False,
                        }
                    )
                if selected_device.type == "cuda":
                    torch.cuda.empty_cache()

    component_summary: list[dict[str, Any]] = []
    if component_rows:
        for task in sorted({str(row["task"]) for row in component_rows}):
            for view in ("eeg", "hbo", "hbr", "nvc", "stack"):
                values = [
                    float(row["macro_f1"])
                    for row in component_rows
                    if row["task"] == task and row["view"] == view
                ]
                if values:
                    component_summary.append(
                        {
                            "task": task,
                            "display_name": TASK_DISPLAY[task],
                            "view": view,
                            "outer_fold_count": len(values),
                            "macro_f1_mean": float(np.mean(values)),
                            "macro_f1_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                            "macro_f1_min": float(np.min(values)),
                            "macro_f1_max": float(np.max(values)),
                            "strict_cross_subject_existing_value": STRICT_VALUES.get(task),
                            "diagnostic_only": True,
                        }
                    )

    report = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "status": "pass",
        "method_id": "brainfusion_nvc_csp_stacking_reimplementation",
        "protected_test_opened": False,
        "table_admissible": False,
        "analysis_boundary": {
            "public_tensor_cache_only": True,
            "public_split_manifests_only": True,
            "no_protected_selection": True,
            "window_diagnostics_are_prefix_only": True,
        },
        "protocol_capability": protocol,
        "source_fidelity": {
            "upstream_revision": "1d9dcf4026f237efed7f0dd44ba44ef0bf87915b",
            "public_nvc_component": True,
            "paper_case_csp_and_stacking_public": False,
            "reported_method_name": "brainfusion_nvc_csp_stacking_reimplementation",
            "original_numeric_reproduction_claim_allowed": False,
        },
        "existing_results": {
            task: _existing_result_summary(task) for task in tasks
        },
        "cache_audit": audits,
        "window_diagnostics": window_rows,
        "component_diagnostics": component_rows,
        "component_summary": component_summary,
        "next_experiments": {
            "F1": "Run an explicitly exploratory within-subject/group-safe bridge on public data; do not label it source reproduction.",
            "F2": "Acquire or authorize longer post-stimulus fNIRS records before attempting long-window/HRF dose experiments.",
            "F3": "Repeat component rows with all fixed seeds and add modality-shuffle and HRF-null negative controls.",
        },
    }
    _write_json(output_root / "brainfusion_protocol_bridge.json", _json_safe(report))
    _write_csv(output_root / "nvc_window_diagnostics.csv", [_json_safe(row) for row in window_rows])
    if component_rows:
        _write_csv(output_root / "component_diagnostics.csv", [_json_safe(row) for row in component_rows])
        _write_csv(output_root / "component_summary.csv", [_json_safe(row) for row in component_summary])
    _write_json(output_root / "cache_audit.json", {"schema": SCHEMA, "tasks": audits, "protected_test_opened": False})
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS, default=list(SUPPORTED_TASKS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--run-components", action="store_true")
    parser.add_argument("--component-tasks", nargs="+", choices=SUPPORTED_TASKS, default=["motor_imagery", "mental_arithmetic"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(
        tasks=args.tasks,
        output_root=args.output_root,
        device=args.device,
        run_components=args.run_components,
        component_tasks=args.component_tasks,
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "status": result["status"],
        "output_root": str(_safe_path(args.output_root)),
        "tasks": list(args.tasks),
        "component_rows": len(result["component_diagnostics"]),
        "window_rows": len(result["window_diagnostics"]),
        "protected_test_opened": result["protected_test_opened"],
    }, indent=2, sort_keys=True))
