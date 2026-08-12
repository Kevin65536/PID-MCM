#!/usr/bin/env python3
"""Build final EFRM A0-A8 cells from the completed audited public matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from comparative_methods.audit_adapter_alignment import audit as audit_alignment  # noqa: E402
from run_downstream_public_v2 import (  # noqa: E402
    DEFAULT_CONFIG,
    METHOD_ID,
    PROTOCOL_ID,
    PublicSurface,
    load_config,
    load_public_surface,
    portable_path,
    resolve_repo_path,
    sha256_file,
    stable_hash,
    validate_feature_arrays,
    write_json,
)


TASKS = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
)
ALIGNMENT_CONTRACT = REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"
COMPLETION_PATH = METHOD_ROOT / "evidence/public_development_v2/matrix_completion_summary.json"
OUTPUT_ROOT = METHOD_ROOT / "evidence/alignment_v2"
REFERENCE_ROOT = REPO_ROOT / "comparative_methods/NormWear/evidence/alignment_v2"


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected EFRM alignment input: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"EFRM alignment input must be an object: {resolved}")
    if value.get("protected_test_opened", False):
        raise PermissionError(f"EFRM alignment input reports protected access: {resolved}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def common_sample_ids(surface: PublicSurface) -> list[str]:
    values: list[str] = []
    for index in surface.full_public_indices:
        row = surface.dataset.lightweight_metadata(index)
        values.append(
            f"{row['join_key']}|event={int(row['event_index'])}"
            f"|offset_ms={int(round(float(row['window_offset_s']) * 1000.0))}"
        )
    require(len(values) == len(set(values)), f"duplicate common sample IDs: {surface.task}")
    return values


def common_split_rows(surface: PublicSurface) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        fold = surface.folds[outer_fold]
        manifest = load_json(fold.public_manifest_path)
        rows.append(
            {
                "outer_fold": outer_fold,
                "public_manifest_sha256": fold.public_manifest_sha256,
                "split_sha256": fold.public_split_sha256,
                "metadata_sha256": str(manifest["metadata_sha256"]),
                "train_sample_count": len(fold.train_indices),
                "validation_sample_count": len(fold.validation_indices),
            }
        )
    return rows


def ordered_array_sha256(sample_ids: Sequence[str], values: np.ndarray) -> str:
    array = np.asarray(values)
    require(len(sample_ids) == len(array), "ordered array/sample identity length mismatch")
    digest = hashlib.sha256()
    for sample_id, row in zip(sample_ids, array, strict=True):
        value = np.ascontiguousarray(row)
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def feature_artifacts(
    config: Mapping[str, Any], task: str
) -> tuple[PublicSurface, dict[str, Any], dict[str, np.ndarray], Path, dict[str, Any]]:
    run_root = resolve_repo_path(config["resources"]["run_root"])
    run_dir = run_root / task / "outer0" / "seed17"
    manifest = load_json(run_dir / "manifest.json")
    require(manifest.get("mode") == "public_selection_and_refit", "alignment source is smoke")
    require(manifest.get("table_admissible") is False, "public source claims table admission")
    require(manifest.get("target_dataset_exposure") is False, "source reports target exposure")
    feature = dict(manifest["feature_cache"])
    cache_path = resolve_repo_path(feature.pop("path"))
    retained_file_sha256 = str(feature.pop("file_sha256"))
    feature.pop("cache_hit", None)
    cache_manifest = load_json(cache_path.with_suffix(".json"))
    require(feature == cache_manifest, "run/cache feature identities differ")
    require(retained_file_sha256 == sha256_file(cache_path), "feature file hash drifted")
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    surface = load_public_surface(config, task=task)
    validate_feature_arrays(arrays, surface, embedding_dim=768)
    return surface, cache_manifest, arrays, cache_path, manifest


def inventory_values(arrays: Mapping[str, np.ndarray], name: str) -> list[list[str]]:
    output: list[list[str]] = []
    for value in arrays[name].astype(str).tolist():
        decoded = json.loads(value)
        require(isinstance(decoded, list) and decoded, f"invalid cached inventory: {name}")
        output.append([str(item) for item in decoded])
    return output


def assert_classification_reference(
    *,
    fields: Mapping[str, Any],
    surface: PublicSurface,
    arrays: Mapping[str, np.ndarray],
    sample_ids: Sequence[str],
) -> None:
    require(
        fields["sample_inventory_sha256"] == stable_hash(sorted(sample_ids)),
        f"common sample inventory differs: {surface.task}",
    )
    require(
        fields["split_fingerprint"] == stable_hash(common_split_rows(surface)),
        f"common split fingerprint differs: {surface.task}",
    )
    expected = fields["measured_channel_identity_set"]
    eeg_rows = inventory_values(arrays, "eeg_channel_inventory_json")
    fnirs_rows = inventory_values(arrays, "fnirs_location_inventory_json")
    require(
        all(sorted(row) == list(expected["eeg"]) for row in eeg_rows),
        f"EFRM EEG inventory differs from direct reference: {surface.task}",
    )
    require(
        all(sorted(row) == list(expected["fnirs_hbo"]) for row in fnirs_rows),
        f"EFRM fNIRS inventory differs from direct reference: {surface.task}",
    )
    full_eeg = int(round(surface.dataset.spec.input_duration_s * 200.0))
    full_fnirs = int(round(surface.dataset.spec.input_duration_s * 10.0))
    require(
        bool((arrays["eeg_time_valid_count"] == full_eeg).all())
        and bool((arrays["fnirs_time_valid_count"] == full_fnirs).all()),
        f"classification input has partial recorded support: {surface.task}",
    )
    require(bool(arrays["target_valid_mask"].all()), "classification target mask is partial")


def refed_comparison_fields(
    *,
    surface: PublicSurface,
    arrays: Mapping[str, np.ndarray],
    sample_ids: Sequence[str],
    cache_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    eeg_rows = inventory_values(arrays, "eeg_channel_inventory_json")
    fnirs_rows = inventory_values(arrays, "fnirs_location_inventory_json")
    unique_eeg = sorted({tuple(sorted(row)) for row in eeg_rows})
    unique_fnirs = sorted({tuple(sorted(row)) for row in fnirs_rows})
    inventory_audit = {
        "eeg_unique_inventories": [list(row) for row in unique_eeg],
        "fnirs_hbo_unique_inventories": [list(row) for row in unique_fnirs],
        "fnirs_hbr_unique_inventories": [list(row) for row in unique_fnirs],
        "per_sample_inventory_sha256": stable_hash(
            [
                {"sample_id": sample_id, "eeg": eeg, "fnirs": fnirs}
                for sample_id, eeg, fnirs in zip(sample_ids, eeg_rows, fnirs_rows, strict=True)
            ]
        ),
    }
    support_rows = [
        {
            "sample_id": sample_id,
            "eeg_time_valid_count": int(arrays["eeg_time_valid_count"][position]),
            "fnirs_time_valid_count": int(arrays["fnirs_time_valid_count"][position]),
            "eeg_valid_channel_count": int(arrays["eeg_valid_channel_count"][position]),
            "fnirs_valid_location_count": int(arrays["fnirs_valid_location_count"][position]),
            "eeg_time_valid_sha256": str(arrays["eeg_time_valid_sha256"][position]),
            "fnirs_time_valid_sha256": str(arrays["fnirs_time_valid_sha256"][position]),
            "eeg_patch_valid_sha256": str(arrays["eeg_patch_valid_sha256"][position]),
            "fnirs_patch_valid_sha256": str(arrays["fnirs_patch_valid_sha256"][position]),
        }
        for position, sample_id in enumerate(sample_ids)
    ]
    full_eeg = int(round(surface.dataset.spec.input_duration_s * 200.0))
    full_fnirs = int(round(surface.dataset.spec.input_duration_s * 10.0))
    recorded_support = {
        "semantics": "recorded_time_channel_and_patch_masks_retained_without_padding_as_data",
        "sample_count": len(sample_ids),
        "partial_input_sample_count": int(
            np.count_nonzero(
                (arrays["eeg_time_valid_count"] < full_eeg)
                | (arrays["fnirs_time_valid_count"] < full_fnirs)
            )
        ),
        "per_sample_support_sha256": stable_hash(support_rows),
    }
    recorded_support["sha256"] = stable_hash(recorded_support)
    target_mask = np.asarray(arrays["target_valid_mask"], dtype=bool)
    target_valid = {
        "semantics": "refed_native_coordinate_partial_target_mask_retained",
        "shape": list(target_mask.shape),
        "valid_value_count": int(target_mask.sum()),
        "invalid_value_count": int(target_mask.size - target_mask.sum()),
        "sha256": ordered_array_sha256(sample_ids, target_mask),
    }
    data_hashes = dict(cache_manifest["data_branch_sha256"])
    source_hashes = dict(cache_manifest["source_sha256"])
    canonical_signal = {
        "schema": "canonical_multimodal_signal_branch_identity_v2",
        "eeg": {
            "dataset_branch": "raw_with_ocular_artifact",
            "filter_band_hz": [1, 45],
            "sample_rate_hz": 200,
            "unit": "robust_standard_deviation",
        },
        "fnirs": {
            "source_coordinate": "released_hbo_hbr_hbt_export_with_native_unit_provenance",
            "component_roles": ["HbO", "HbR"],
            "sample_rate_hz": 10,
            "unit": "robust_standard_deviation",
        },
        "fingerprints": {
            **data_hashes,
            "task_adapter": source_hashes["task_dataset"],
            "unified_loader": source_hashes["unified_loader"],
        },
    }
    return {
        "dataset_id": "refed",
        "task_id": "refed_regression",
        "sample_inventory_sha256": stable_hash(sorted(sample_ids)),
        "split_fingerprint": stable_hash(common_split_rows(surface)),
        "target_schema": "valence_arousal_2_by_20_with_coordinate_mask",
        "target_valid_mask": target_valid,
        "primary_endpoint": "masked_native_coordinate_ccc",
        "observation_anchor": "canonical_registry_window_start",
        "modality_intervals_s": {
            "eeg": [0.0, 20.0],
            "fnirs_hbo": [0.0, 20.0],
            "fnirs_hbr": [0.0, 20.0],
        },
        "modality_identity": ["eeg", "fnirs_hbo", "fnirs_hbr"],
        "measured_channel_identity_set": inventory_audit,
        "recorded_support_mask": recorded_support,
        "canonical_signal_branch": canonical_signal,
    }


def build_cell(
    *,
    config: Mapping[str, Any],
    task: str,
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    surface, cache_manifest, arrays, cache_path, run_manifest = feature_artifacts(config, task)
    sample_ids = common_sample_ids(surface)
    if task == "refed_regression":
        fields = refed_comparison_fields(
            surface=surface,
            arrays=arrays,
            sample_ids=sample_ids,
            cache_manifest=cache_manifest,
        )
        comparison_group_id = "multimodal_synchronous__refed_regression__support_matched_direct__v2"
    else:
        reference_path = REFERENCE_ROOT / f"{task}.json"
        reference = load_json(reference_path)
        require(reference.get("cell_status") == "pass", f"reference cell is not pass: {task}")
        fields = dict(reference["comparison_fields"])
        assert_classification_reference(
            fields=fields,
            surface=surface,
            arrays=arrays,
            sample_ids=sample_ids,
        )
        comparison_group_id = str(reference["comparison_group_id"])

    support_names = (
        "eeg_channel_inventory_json",
        "fnirs_location_inventory_json",
        "eeg_time_valid_count",
        "fnirs_time_valid_count",
        "eeg_patch_valid_count",
        "fnirs_patch_valid_count",
        "eeg_valid_channel_count",
        "fnirs_valid_location_count",
        "eeg_time_valid_sha256",
        "fnirs_time_valid_sha256",
        "eeg_patch_valid_sha256",
        "fnirs_patch_valid_sha256",
    )
    support_digest = stable_hash(
        {
            name: arrays[name].astype(str).tolist()
            for name in support_names
        }
    )
    adapter_identity = {
        "method_id": METHOD_ID,
        "protocol_id": PROTOCOL_ID,
        "checkpoint": cache_manifest["checkpoint"],
        "adapter_manifest": cache_manifest["adapter_manifest"],
        "feature_semantics": cache_manifest["feature_semantics"],
        "source_sha256": cache_manifest["source_sha256"],
        "data_branch_sha256": cache_manifest["data_branch_sha256"],
        "feature_cache_key": cache_manifest["feature_cache_key"],
        "feature_cache_sha256": sha256_file(cache_path),
        "output_layer": "eeg_and_fnirs_valid_patch_masked_encoder_means",
        "pooling": "elementwise_sum_before_trainable_layer_norm",
        "downstream_head": "trainable_layernorm_dropout_linear_probe_only",
        "trainable_parameter_boundary": "target_excluded_backbone_frozen_probe_only",
        "target_corpus_exposure": "none_target_dataset_excluded",
    }
    return {
        "schema": "adapter_alignment_cell_evidence_v2",
        "cell_id": f"efrm__{task}__support_matched_direct__v2",
        "comparison_group_id": comparison_group_id,
        "method_id": METHOD_ID,
        "task_id": task,
        "track": "multimodal_target_dataset_excluded_frozen_linear_probe",
        "alignment_profile": "support_matched_direct",
        "evidence_scope": "public_complete",
        "cell_status": "pass",
        "comparison_fields": fields,
        "adapter_identity": adapter_identity,
        "gate_status": {f"A{index}": "pass" for index in range(9)},
        "public_adapter_audit": {
            "unique_sample_count": len(sample_ids),
            "dataset_indices_sha256": stable_hash(
                arrays["dataset_indices"].astype(int).tolist()
            ),
            "common_sample_inventory_sha256": stable_hash(sorted(sample_ids)),
            "input_support_audit_sha256": support_digest,
            "target_valid_mask_sha256": ordered_array_sha256(
                sample_ids, arrays["target_valid_mask"].astype(bool)
            ),
            "feature_sha256": ordered_array_sha256(sample_ids, arrays["features"]),
            "feature_nonconstant": bool(
                (arrays["features"].std(axis=0, dtype=np.float64) > 1e-8).any()
            ),
            "feature_cache_path": portable_path(cache_path),
            "feature_cache_manifest_sha256": sha256_file(cache_path.with_suffix(".json")),
            "source_run_manifest_path": portable_path(
                resolve_repo_path(config["resources"]["run_root"])
                / task
                / "outer0"
                / "seed17"
                / "manifest.json"
            ),
            "source_run_manifest_sha256": sha256_file(
                resolve_repo_path(config["resources"]["run_root"])
                / task
                / "outer0"
                / "seed17"
                / "manifest.json"
            ),
        },
        "protocol_freeze": {
            "public_matrix_completion_path": portable_path(COMPLETION_PATH),
            "public_matrix_completion_sha256": sha256_file(COMPLETION_PATH),
            "matrix_identity_sha256": completion["matrix_identity_sha256"],
            "protected_evaluation_authorized": False,
            "protected_test_opened": False,
        },
        "target_dataset_exposure": False,
        "protected_test_opened": False,
        "table_admissible": False,
    }


def build(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, _resolved_config = load_config(config_path)
    completion = load_json(COMPLETION_PATH)
    require(completion.get("status") == "pass", "EFRM public matrix is not finalized")
    require(int(completion.get("completed_job_count", -1)) == 105, "EFRM matrix is incomplete")
    require(completion.get("protected_test_opened") is False, "completion opened protected data")
    cell_paths: list[Path] = []
    summary_tasks: list[dict[str, Any]] = []
    for task in TASKS:
        cell = build_cell(config=config, task=task, completion=completion)
        path = OUTPUT_ROOT / f"{task}.json"
        write_json(path, cell)
        cell_paths.append(path)
        summary_tasks.append(
            {
                "task": task,
                "path": portable_path(path),
                "cell_status": "pass",
                "sample_count": cell["public_adapter_audit"]["unique_sample_count"],
                "status": "A0-A8_pass_public_matrix_complete_protected_locked",
            }
        )
    audit = audit_alignment(ALIGNMENT_CONTRACT, cell_paths)
    for report in audit["cell_reports"]:
        report["source"] = portable_path(Path(str(report["source"])))
    summary = {
        "schema": "efrm_adapter_alignment_summary_v2",
        "status": "pass",
        "method_id": METHOD_ID,
        "evidence_scope": "public_complete",
        "tasks": summary_tasks,
        "alignment_audit": audit,
        "completed_gates": [f"A{index}" for index in range(9)],
        "protected_evaluation_authorized": False,
        "target_dataset_exposure": False,
        "protected_test_opened": False,
        "table_admissible": False,
    }
    write_json(OUTPUT_ROOT / "summary_final.json", summary)
    return summary


def main() -> int:
    summary = build()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
