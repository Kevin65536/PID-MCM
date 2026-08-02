#!/usr/bin/env python3
"""Audit NormWear's A0-A4 public data boundary without executing the model."""

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
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import (
    METHOD_ID,
    SUPPORTED_TASKS,
    NormWearPublicView,
    PublicInventory,
    data_branch_fingerprints,
    load_config,
    load_public_inventory,
    stable_hash,
)
from comparative_methods.audit_adapter_alignment import audit as audit_evidence
from comparative_methods.audit_public_preflight import sha256_file


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
        raise PermissionError(f"refusing protected NormWear evidence path: {resolved}")
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
        raise PermissionError(f"NormWear is not the active delivery method: {active!r}")
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
        "data_boundary_audit": Path(__file__),
        "identity_audit": METHOD_ROOT / "audit_identity_v2.py",
        "upstream_model": METHOD_ROOT / "upstream/modules/normwear.py",
        "upstream_entrypoint": METHOD_ROOT / "upstream/main_model.py",
        "source_fidelity": METHOD_ROOT / "sources/SOURCE_FIDELITY.md",
        "method_manifest": METHOD_ROOT / "sources/method_manifest.yaml",
        "config": config_path,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"NormWear identity source is missing: {missing}")
    eeg_samples = int(round(inventory.duration_s * 200.0))
    fnirs_samples = int(round(inventory.duration_s * 10.0))
    delivered = list(inventory.delivered_channel_names)
    return {
        "method_id": METHOD_ID,
        "reporting_name": str(config["source"]["reporting_name"]),
        "upstream_revision": str(config["source"]["upstream_revision"]),
        "checkpoint_sha256": (
            "36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561"
        ),
        "source_file_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
        "delivered_channel_order": delivered,
        "delivered_channel_order_sha256": stable_hash(delivered),
        "canonical_input_shape": {
            "eeg": ["batch", len(inventory.eeg_channels), eeg_samples],
            "fnirs_hbo": ["batch", len(inventory.fnirs_locations), fnirs_samples],
            "fnirs_hbr": ["batch", len(inventory.fnirs_locations), fnirs_samples],
        },
        "planned_model_input_shape": [
            "batch",
            len(delivered),
            int(round(inventory.duration_s * 65.0)),
        ],
        "planned_output_shape": ["batch", len(delivered) * 768],
        "measurement_to_adapter_transform": {
            "eeg": "scipy.signal.resample_poly_up13_down40",
            "fnirs_hbo": "scipy.signal.resample_poly_up13_down2",
            "fnirs_hbr": "scipy.signal.resample_poly_up13_down2",
            "fit_scope": "deterministic_stateless_no_target_information",
        },
        "output_layer": "final_encoder_layer_norm_tokens",
        "pooling": "mean_all_tokens_per_channel_then_concatenate_real_channels",
        "trainable_parameter_boundary": "frozen_encoder_outer_training_linear_probe_only",
        "source_deviation": [
            "fNIRS_is_an_explicit_cross_modality_adaptation",
            "canonical_shared_measurement_coordinate_replaces_source_dataset_preprocessing",
            "official_helper_rate_condition_replaced_by_explicit_200_and_10_to_65_resampling",
        ],
        "target_corpus_exposure": "none_by_declared_dataset_identity",
        "production_adapter_executed": False,
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
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    started = time.perf_counter()
    inventory = load_public_inventory(config, task=task)
    view = NormWearPublicView(inventory)
    digest = hashlib.sha256()
    observed_ids: list[str] = []
    observed_indices: list[int] = []
    minimum_std = {"eeg": float("inf"), "hbo": float("inf"), "hbr": float("inf")}
    print(f"[{task}] auditing {len(inventory.indices)} public inputs", flush=True)
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
        if number % 500 == 0 or number == len(inventory.indices):
            print(f"[{task}] {number}/{len(inventory.indices)} samples", flush=True)

    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(
        inventory.sample_ids
    ):
        raise RuntimeError(f"NormWear public identity coverage drifted for {task}")
    if set(observed_indices) != set(inventory.indices):
        raise RuntimeError(f"NormWear dataset-index coverage drifted for {task}")
    fields = comparison_fields(
        task=task,
        inventory=inventory,
        alignment_contract=alignment_contract,
        branch_fingerprints=branch_fingerprints,
    )
    identity = adapter_identity(
        task=task, inventory=inventory, config=config, config_path=config_path
    )
    class_counts = Counter(
        str(inventory.dataset.lightweight_metadata(index)["condition"])
        for index in inventory.indices
    )
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "cell_id": f"normwear__{task}__support_matched_direct__v2",
        "comparison_group_id": f"multimodal_synchronous__{task}__support_matched_direct__v2",
        "method_id": METHOD_ID,
        "task_id": task,
        "track": str(config["track"]),
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
            "A5": "pending",
            "A6": "pass",
            "A7": "pending",
            "A8": "pending",
        },
        "public_data_audit": {
            "unique_sample_count": len(inventory.indices),
            "all_unique_public_samples_audited": True,
            "outer_fold_count": len(inventory.split_rows),
            "class_counts": dict(sorted(class_counts.items())),
            "canonical_input_shape": identity["canonical_input_shape"],
            "canonical_input_sha256": digest.hexdigest(),
            "minimum_per_sample_modality_std": minimum_std,
            "all_canonical_inputs_finite_and_nonconstant": True,
            "all_recorded_and_analysis_support_complete": True,
            "all_channels_real_measured_and_unpadded": True,
            "elapsed_seconds": time.perf_counter() - started,
            "protected_test_opened": False,
        },
    }
    output_path = output_root / f"{task}.json"
    write_json(output_path, evidence)
    del inventory, view
    gc.collect()
    return evidence, output_path


def unsupported_refed_cell(
    *, config: Mapping[str, Any], alignment_contract: Mapping[str, Any]
) -> dict[str, Any]:
    task = "refed_regression"
    task_contract = alignment_contract["task_contracts"][task]
    observation = task_contract["primary_observation"]
    task_config = config["tasks"][task]
    return {
        "schema": EVIDENCE_SCHEMA,
        "cell_id": "normwear__refed_regression__support_matched_direct__v2",
        "comparison_group_id": (
            "multimodal_synchronous__refed_regression__support_matched_direct__v2"
        ),
        "method_id": METHOD_ID,
        "task_id": task,
        "track": "preregistered_unsupported",
        "alignment_profile": "support_matched_direct",
        "evidence_scope": "static",
        "cell_status": "unsupported",
        "unsupported_reason_code": str(task_config["unsupported_reason_code"]),
        "unsupported_reason": str(task_config["unsupported_reason"]),
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
            "A2": "pass",
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
    if not values:
        return SUPPORTED_TASKS
    tasks = tuple(str(value) for value in values)
    if len(tasks) != len(set(tasks)):
        raise ValueError("NormWear task list must be unique")
    unknown = sorted(set(tasks) - set(SUPPORTED_TASKS))
    if unknown:
        raise ValueError(f"unknown or unsupported NormWear tasks: {unknown}")
    return tasks


def run(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    tasks: Sequence[str] = (),
) -> dict[str, Any]:
    selected = parse_tasks(tasks)
    config, resolved_config = load_config(config_path)
    alignment_contract = load_alignment_contract()
    branch_fingerprints = data_branch_fingerprints(config)
    output_root = output_root.resolve()
    if "protected" in {part.lower() for part in output_root.parts}:
        raise PermissionError("refusing protected NormWear audit output")
    started_at = utc_now()
    evidence_paths: list[Path] = []
    reports: list[dict[str, Any]] = []
    for task in selected:
        report, path = run_task(
            task=task,
            config=config,
            config_path=resolved_config,
            alignment_contract=alignment_contract,
            branch_fingerprints=branch_fingerprints,
            output_root=output_root,
        )
        reports.append(report)
        evidence_paths.append(path)
    if set(selected) == set(SUPPORTED_TASKS):
        unsupported = unsupported_refed_cell(
            config=config, alignment_contract=alignment_contract
        )
        unsupported_path = output_root / "refed_regression.json"
        write_json(unsupported_path, unsupported)
        reports.append(unsupported)
        evidence_paths.append(unsupported_path)
    schema_audit = audit_evidence(ALIGNMENT_CONTRACT, evidence_paths)
    summary = {
        "schema": "normwear_data_boundary_audit_summary_v2",
        "status": "A0_A4_pass_A5_A8_pending_protected_locked",
        "method_id": METHOD_ID,
        "started_at": started_at,
        "completed_at": utc_now(),
        "tasks": list(selected),
        "supported_cell_count": len(selected),
        "unsupported_cell_count": int(set(selected) == set(SUPPORTED_TASKS)),
        "audited_unique_public_sample_count": sum(
            int(report.get("public_data_audit", {}).get("unique_sample_count", 0))
            for report in reports
        ),
        "evidence_paths": [portable_path(path) for path in evidence_paths],
        "schema_audit": schema_audit,
        "protected_test_opened": False,
    }
    write_json(output_root / "data_boundary_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task", action="append", default=[])
    args = parser.parse_args(argv)
    summary = run(config_path=args.config, output_root=args.output_root, tasks=args.task)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
