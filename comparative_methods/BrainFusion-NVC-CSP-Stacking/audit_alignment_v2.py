#!/usr/bin/env python3
"""Run BrainFusion's full-public A0-A7 adapter audit without protected reads."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
ADAPTER_ROOT = METHOD_ROOT / "adapters"
for import_path in (REPO_ROOT, METHOD_ROOT, ADAPTER_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import (
    METHOD_ID,
    SUPPORTED_TASKS,
    BrainFusionPublicView,
    PublicInventory,
    data_branch_fingerprints,
    load_config,
    load_public_inventory,
    stable_hash,
)
from brainfusion_gpu.nvc import NVCConfig, brainfusion_nvc_contribution_timeseries
from comparative_methods.audit_adapter_alignment import audit as audit_evidence
from comparative_methods.audit_public_preflight import sha256_file
from efrm_pytorch.tasks import TASK_SPECS


DEFAULT_CONFIG = METHOD_ROOT / "configs/alignment_v2.yaml"
ALIGNMENT_CONTRACT = REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"
DEFAULT_OUTPUT_ROOT = METHOD_ROOT / "evidence/alignment_v2"
EVIDENCE_SCHEMA = "adapter_alignment_cell_evidence_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected BrainFusion evidence path: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_alignment_contract() -> Mapping[str, Any]:
    value = yaml.safe_load(ALIGNMENT_CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "adapter_alignment_gate_contract_v2":
        raise ValueError("adapter-alignment v2 contract is unavailable")
    active = value.get("execution_policy", {}).get("active_delivery_method")
    if active != METHOD_ID:
        raise PermissionError(f"BrainFusion is not the active delivery method: {active!r}")
    return value


def comparison_fields(
    *,
    task: str,
    inventory: PublicInventory,
    alignment_contract: Mapping[str, Any],
    branch_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    task_contract = alignment_contract["task_contracts"][task]
    observation = task_contract["primary_observation"]
    target_schema = str(task_contract["target_schema"])
    measured = {
        "eeg": sorted(inventory.eeg_channels),
        "fnirs_hbo": sorted(inventory.fnirs_locations),
        "fnirs_hbr": sorted(inventory.fnirs_locations),
    }
    samples = {
        "eeg": int(round(inventory.duration_s * 200.0)),
        "fnirs_hbo": int(round(inventory.duration_s * 10.0)),
        "fnirs_hbr": int(round(inventory.duration_s * 10.0)),
    }
    support = {
        "sample_inventory_sha256": inventory.sample_inventory_sha256,
        "measured_channel_identity_set": measured,
        "sample_count": len(inventory.indices),
        "samples_per_item": samples,
        "mask_semantics": "all_true_recorded_and_analysis_valid_support_no_padding",
    }
    dataset_id = str(task_contract["dataset_id"])
    return {
        "dataset_id": dataset_id,
        "task_id": task,
        "sample_inventory_sha256": inventory.sample_inventory_sha256,
        "split_fingerprint": inventory.split_fingerprint,
        "target_schema": target_schema,
        "target_valid_mask": {
            "semantics": "classification_scalar_all_observed",
            "sha256": stable_hash(
                {
                    "sample_inventory_sha256": inventory.sample_inventory_sha256,
                    "target_schema": target_schema,
                    "mask": "all_true_scalar",
                }
            ),
        },
        "primary_endpoint": str(task_contract["primary_endpoint"]),
        "observation_anchor": str(observation["anchor"]),
        "modality_intervals_s": {
            "eeg": list(observation["eeg_interval_s"]),
            "fnirs_hbo": list(observation["fnirs_interval_s"]),
            "fnirs_hbr": list(observation["fnirs_interval_s"]),
        },
        "modality_identity": ["eeg", "fnirs_hbo", "fnirs_hbr"],
        "measured_channel_identity_set": measured,
        "recorded_support_mask": {**support, "sha256": stable_hash(support)},
        "canonical_signal_branch": {
            "schema": "canonical_multimodal_signal_branch_identity_v2",
            "eeg": {
                "sample_rate_hz": 200.0,
                "filter_band_hz": [1.0, 45.0],
                "unit": "robust_standard_deviation",
                "dataset_branch": {
                    "eeg_fnirs_single_trial": "single_trial_eeg_artifact_clean_v4",
                    "simultaneous_eeg_nirs": "simultaneous_eeg_eog_clean_v1",
                    "visual_cognitive_motivation": "raw_with_ocular_artifact",
                }[dataset_id],
            },
            "fnirs": {
                "sample_rate_hz": 10.0,
                "component_roles": ["HbO", "HbR"],
                "unit": "robust_standard_deviation",
                "source_coordinate": alignment_contract["dataset_contracts"][dataset_id].get(
                    "fnirs_source_coordinate", "released_hbo_hbr_concentration"
                ),
            },
            "fingerprints": dict(sorted(branch_fingerprints.items())),
        },
    }


def adapter_identity(
    *, task: str, inventory: PublicInventory, config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    source_paths = {
        "alignment_data": METHOD_ROOT / "alignment_data.py",
        "alignment_audit": Path(__file__),
        "nvc": ADAPTER_ROOT / "brainfusion_gpu/nvc.py",
        "features": ADAPTER_ROOT / "brainfusion_gpu/features.py",
        "stacking": ADAPTER_ROOT / "brainfusion_gpu/stacking.py",
        "pipeline": ADAPTER_ROOT / "brainfusion_gpu/pipeline.py",
        "source_nvc": METHOD_ROOT / "upstream/src/BrainFusion/pipeLine/coupling_analysis.py",
        "source_ml_ui": METHOD_ROOT
        / "upstream/src/BrainFusion/pipeLine/machine_learning_dialog.py",
        "source_fidelity": METHOD_ROOT / "sources/SOURCE_FIDELITY.md",
        "method_manifest": METHOD_ROOT / "sources/method_manifest.yaml",
        "observation_budget_audit": METHOD_ROOT / "OBSERVATION_BUDGET_AUDIT.md",
        "config": config_path,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"BrainFusion identity source is missing: {missing}")
    classes = len(TASK_SPECS[task].class_names)
    csp_dimension = 4 if classes == 2 else 2 * classes
    eeg_samples = int(round(inventory.duration_s * 200.0))
    fnirs_samples = int(round(inventory.duration_s * 10.0))
    return {
        "method_id": METHOD_ID,
        "reporting_name": str(config["source"]["reporting_name"]),
        "upstream_revision": str(config["source"]["upstream_revision"]),
        "source_file_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
        "delivered_channel_order": {
            "eeg": list(inventory.eeg_channels),
            "fnirs_hbo": list(inventory.fnirs_locations),
            "fnirs_hbr": list(inventory.fnirs_locations),
        },
        "delivered_channel_order_sha256": stable_hash(
            {
                "eeg": list(inventory.eeg_channels),
                "fnirs_hbo": list(inventory.fnirs_locations),
                "fnirs_hbr": list(inventory.fnirs_locations),
            }
        ),
        "input_shape": {
            "eeg": ["batch", len(inventory.eeg_channels), eeg_samples],
            "fnirs_hbo": ["batch", len(inventory.fnirs_locations), fnirs_samples],
            "fnirs_hbr": ["batch", len(inventory.fnirs_locations), fnirs_samples],
        },
        "output_shape": {
            "eeg_csp": ["batch", csp_dimension],
            "hbo_csp": ["batch", csp_dimension],
            "hbr_csp": ["batch", csp_dimension],
            "selected_nvc_csp": ["batch", csp_dimension],
            "stacking_decision": ["batch", classes],
        },
        "output_layer": "four_fold_local_csp_views_then_oof_linear_svm_stack",
        "patch_and_token_grid": {
            "eeg_nonoverlapping_average_samples": 20,
            "eeg_summary_rate_hz": 10.0,
            "fnirs_rate_hz": 10.0,
            "nvc_unselected_pair_count": (
                len(inventory.eeg_channels) * len(inventory.fnirs_locations) * 2
            ),
            "nvc_selected_pair_count": min(
                32, len(inventory.eeg_channels) * len(inventory.fnirs_locations) * 2
            ),
        },
        "geometry_encoding": "none_channel_names_only_no_fabricated_coregistration",
        "pooling": "csp_log_normalized_variance",
        "architecture": (
            "EEG_HbO_HbR_dynamic_NVC_CSP_views__selected_SVM_or_RF_bases__"
            "linear_SVM_OOF_stack"
        ),
        "deterministic_source_declared_sample_transform": {
            "nvc": "per_sample_channel_minmax_then_causal_SPM_HRF",
            "hrf": dict(config["adapter"]["hrf"]),
        },
        "train_partition_fitted_transform": list(config["adapter"]["learned_transforms"]),
        "trainable_parameter_boundary": "all_fitted_state_outer_training_only",
        "source_deviation": [
            "CSP_and_stacking_case_execution_not_public_independent_reimplementation",
            "canonical_shared_measurement_branch_replaces_paper_case_preprocessing",
            "dynamic_NVC_defined_as_timewise_Pearson_contributions",
            "cross_task_cells_are_adaptations_not_original_numeric_reproductions",
        ],
        "target_corpus_exposure": {
            "pretraining": "not_applicable_no_pretraining",
            "motor_imagery": "source_case_corpus_supervised_fit_within_outer_train_only",
            "other_tasks": "cross_task_supervised_fit_within_outer_train_only",
        },
        "original_numeric_reproduction_claim_allowed": False,
    }


def _update_digest(digest: Any, sample_id: str, values: Sequence[np.ndarray]) -> None:
    digest.update(sample_id.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        digest.update(np.ascontiguousarray(value, dtype=np.float32).tobytes())


def run_task(
    *,
    task: str,
    config: Mapping[str, Any],
    config_path: Path,
    alignment_contract: Mapping[str, Any],
    branch_fingerprints: Mapping[str, str],
    device: torch.device,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    started = time.perf_counter()
    inventory = load_public_inventory(config, task=task)
    view = BrainFusionPublicView(inventory)
    digest = hashlib.sha256()
    observed_ids: list[str] = []
    observed_indices: list[int] = []
    minimum_std = {"eeg": float("inf"), "hbo": float("inf"), "hbr": float("inf")}
    replay_exact = False
    replay_nonconstant_pairs = 0
    nvc_config = NVCConfig(
        eeg_sampling_rate_hz=int(config["data"]["eeg_sample_rate_hz"]),
        fnirs_sampling_rate_hz=int(config["data"]["fnirs_sample_rate_hz"]),
        eeg_window_samples=20,
        hrf_tr=float(config["adapter"]["hrf"]["tr"]),
        hrf_oversampling=int(config["adapter"]["hrf"]["oversampling"]),
        hrf_time_length=float(config["adapter"]["hrf"]["time_length_s"]),
    )
    print(f"[{task}] auditing {len(inventory.indices)} public inputs", flush=True)
    with torch.inference_mode():
        for number, index in enumerate(inventory.indices, start=1):
            item = view[index]
            arrays = {
                "eeg": item["eeg"].numpy(),
                "hbo": item["hbo"].numpy(),
                "hbr": item["hbr"].numpy(),
            }
            for name, value in arrays.items():
                standard_deviation = float(np.std(value, dtype=np.float64))
                if not np.isfinite(value).all() or standard_deviation <= 1e-8:
                    raise RuntimeError(f"{task}/{item['sample_id']} has invalid {name} input")
                minimum_std[name] = min(minimum_std[name], standard_deviation)
            _update_digest(digest, str(item["sample_id"]), list(arrays.values()))
            observed_ids.append(str(item["sample_id"]))
            observed_indices.append(int(item["dataset_index"]))
            if number == 1:
                inputs = [item[name].unsqueeze(0).to(device) for name in ("eeg", "hbo", "hbr")]
                first = brainfusion_nvc_contribution_timeseries(*inputs, nvc_config)[1]
                second = brainfusion_nvc_contribution_timeseries(*inputs, nvc_config)[1]
                replay_exact = bool(torch.equal(first, second))
                replay_nonconstant_pairs = int((first.std(dim=-1) > 1e-8).sum().item())
                if not replay_exact or replay_nonconstant_pairs == 0:
                    raise RuntimeError(f"BrainFusion public NVC replay failed for {task}")
            if number % 500 == 0 or number == len(inventory.indices):
                print(f"[{task}] {number}/{len(inventory.indices)} samples", flush=True)

    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(
        inventory.sample_ids
    ):
        raise RuntimeError(f"BrainFusion public identity coverage drifted for {task}")
    if set(observed_indices) != set(inventory.indices):
        raise RuntimeError(f"BrainFusion dataset-index coverage drifted for {task}")
    fields = comparison_fields(
        task=task,
        inventory=inventory,
        alignment_contract=alignment_contract,
        branch_fingerprints=branch_fingerprints,
    )
    identity = adapter_identity(
        task=task, inventory=inventory, config=config, config_path=config_path
    )
    cache_identity = {
        "comparison_fields": fields,
        "adapter_identity": identity,
        "audit_code_sha256": sha256_file(Path(__file__)),
        "registry_sha256": str(config["registry"]["registry_sha256"]),
    }
    class_counts = Counter(
        str(inventory.dataset.lightweight_metadata(index)["condition"])
        for index in inventory.indices
    )
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "cell_id": f"brainfusion__{task}__support_matched_direct__v2",
        "comparison_group_id": f"multimodal_synchronous__{task}__support_matched_direct__v2",
        "method_id": METHOD_ID,
        "task_id": task,
        "track": str(config["tasks"][task]["track"]),
        "alignment_profile": "support_matched_direct",
        "evidence_scope": "public_complete",
        "cell_status": "pending",
        "comparison_fields": fields,
        "adapter_identity": identity,
        "gate_status": {
            "A0": "pass",
            "A1": "pass",
            "A2": "pass",
            "A3": "pass",
            "A4": "pass",
            "A5": "pass",
            "A6": "pass",
            "A7": "pass",
            "A8": "pending",
        },
        "public_audit": {
            "unique_sample_count": len(inventory.indices),
            "all_unique_public_samples_audited": True,
            "outer_fold_count": len(inventory.split_rows),
            "class_counts": dict(sorted(class_counts.items())),
            "input_shape": identity["input_shape"],
            "input_sha256": digest.hexdigest(),
            "minimum_per_sample_modality_std": minimum_std,
            "all_model_inputs_finite_and_nonconstant": True,
            "deterministic_nvc_replay_exact": replay_exact,
            "replay_nonconstant_nvc_pair_count": replay_nonconstant_pairs,
            "cache_identity_sha256": stable_hash(cache_identity),
            "elapsed_seconds": time.perf_counter() - started,
            "protected_test_opened": False,
        },
    }
    output_path = output_root / f"{task}.json"
    write_json(output_path, evidence)
    del inventory, view
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return evidence, output_path


def unsupported_cell(
    *, task: str, config: Mapping[str, Any], alignment_contract: Mapping[str, Any]
) -> dict[str, Any]:
    task_contract = alignment_contract["task_contracts"][task]
    observation = task_contract["primary_observation"]
    reason_code = str(config["tasks"][task]["unsupported_reason_code"])
    reason = {
        "dsr": (
            "The canonical two-second fNIRS interval is block context, is shorter than the "
            "frozen NVC support minimum, and cannot be extended in a direct profile."
        ),
        "refed_regression": (
            "The source case is a classification stack and has no masked continuous-regression "
            "or partial-terminal-support contract."
        ),
    }[task]
    return {
        "schema": EVIDENCE_SCHEMA,
        "cell_id": f"brainfusion__{task}__support_matched_direct__v2",
        "comparison_group_id": f"multimodal_synchronous__{task}__support_matched_direct__v2",
        "method_id": METHOD_ID,
        "task_id": task,
        "track": "preregistered_unsupported",
        "alignment_profile": "support_matched_direct",
        "evidence_scope": "static",
        "cell_status": "unsupported",
        "unsupported_reason_code": reason_code,
        "unsupported_reason": reason,
        "comparison_fields": {
            "dataset_id": str(task_contract["dataset_id"]),
            "task_id": task,
            "sample_inventory_sha256": "not_dereferenced_preregistered_unsupported",
            "split_fingerprint": "method_neutral_registry_identity_only",
            "target_schema": str(task_contract["target_schema"]),
            "target_valid_mask": "not_dereferenced_preregistered_unsupported",
            "primary_endpoint": str(task_contract["primary_endpoint"]),
            "observation_anchor": str(observation["anchor"]),
            "modality_intervals_s": {
                "eeg": list(observation["eeg_interval_s"]),
                "fnirs_hbo": list(observation["fnirs_interval_s"]),
                "fnirs_hbr": list(observation["fnirs_interval_s"]),
            },
            "modality_identity": ["eeg", "fnirs_hbo", "fnirs_hbr"],
            "measured_channel_identity_set": "not_dereferenced_preregistered_unsupported",
            "recorded_support_mask": "not_dereferenced_preregistered_unsupported",
            "canonical_signal_branch": "not_dereferenced_preregistered_unsupported",
        },
        "adapter_identity": {
            "method_id": METHOD_ID,
            "disposition": "unsupported_before_target_performance",
            "source_revision": str(config["source"]["upstream_revision"]),
        },
        "gate_status": {
            "A0": "pass",
            "A1": "not_applicable",
            "A2": "unsupported" if task == "dsr" else "pass",
            "A3": "not_applicable",
            "A4": "unsupported",
            "A5": "unsupported",
            "A6": "pass",
            "A7": "unsupported",
            "A8": "pending",
        },
        "protected_test_opened": False,
    }


def parse_tasks(values: Sequence[str]) -> tuple[str, ...]:
    tasks = tuple(values) if values else SUPPORTED_TASKS
    if len(tasks) != len(set(tasks)):
        raise ValueError("BrainFusion audit tasks must be unique")
    unknown = sorted(set(tasks) - set(SUPPORTED_TASKS))
    if unknown:
        raise ValueError(f"unknown or unsupported BrainFusion audit tasks: {unknown}")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    contract = load_alignment_contract()
    tasks = parse_tasks(args.task)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for BrainFusion public replay but is unavailable")
    output_root = args.output_root.resolve()
    if "protected" in {part.lower() for part in output_root.parts}:
        raise PermissionError(f"refusing protected BrainFusion output: {output_root}")
    branch_fingerprints = data_branch_fingerprints(config)
    cells: list[dict[str, Any]] = []
    paths: list[Path] = []
    for task in tasks:
        cell, path = run_task(
            task=task,
            config=config,
            config_path=config_path,
            alignment_contract=contract,
            branch_fingerprints=branch_fingerprints,
            device=device,
            output_root=output_root,
        )
        cells.append(cell)
        paths.append(path)
    for task in ("dsr", "refed_regression"):
        cell = unsupported_cell(task=task, config=config, alignment_contract=contract)
        path = output_root / f"{task}.json"
        write_json(path, cell)
        cells.append(cell)
        paths.append(path)
    schema_report = audit_evidence(ALIGNMENT_CONTRACT, paths)
    complete = set(tasks) == set(SUPPORTED_TASKS)
    summary = {
        "schema": "brainfusion_adapter_alignment_summary_v2",
        "status": (
            "implementation_review_complete_A0_A7_pass_A8_pending"
            if complete
            else "partial_public_audit"
        ),
        "method_id": METHOD_ID,
        "created_at": utc_now(),
        "config_path": portable_path(config_path),
        "config_sha256": sha256_file(config_path),
        "audited_supported_tasks": list(tasks),
        "supported_unique_public_sample_count": sum(
            int(cell.get("public_audit", {}).get("unique_sample_count", 0)) for cell in cells
        ),
        "tasks": [
            {
                "task": cell["task_id"],
                "status": (
                    "A0-A7_pass_A8_pending"
                    if cell["cell_status"] == "pending"
                    else "unsupported"
                ),
                "path": portable_path(path),
                "cell_status": cell["cell_status"],
            }
            for cell, path in zip(cells, paths, strict=True)
        ],
        "schema_audit": schema_report,
        "protected_test_opened": False,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
