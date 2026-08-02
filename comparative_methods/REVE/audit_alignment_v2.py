#!/usr/bin/env python3
"""Run REVE's full-public adapter-alignment v2 audit without protected reads."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from comparative_methods.BIOT.audit_alignment_v2 import (
    comparison_fields as _method_neutral_comparison_fields,
)
from comparative_methods.CBraMod.audit_alignment_v2 import (
    exclusive_gpu_lock,
    jsonable,
    portable_path,
    utc_now,
    write_json,
)
from comparative_methods.REVE.adapters.reve import (
    REVEFrozenEncoder,
    load_verified_reve_base,
)
from comparative_methods.REVE.alignment_data import (
    SUPPORTED_TASKS,
    REVEPublicView,
    data_branch_fingerprints,
    load_config,
    load_public_inventory,
    make_loader,
    resolve_repo_path,
    stable_hash,
)
from comparative_methods.audit_adapter_alignment import audit as audit_evidence
from comparative_methods.audit_public_preflight import sha256_file


DEFAULT_CONFIG = METHOD_ROOT / "configs/alignment_v2.yaml"
ALIGNMENT_CONTRACT = REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"
DEFAULT_OUTPUT_ROOT = METHOD_ROOT / "evidence/alignment_v2"
PEER_ROOTS = (
    REPO_ROOT / "comparative_methods/BIOT/evidence/alignment_v2",
    REPO_ROOT / "comparative_methods/CBraMod/evidence/alignment_v2",
)
EVIDENCE_SCHEMA = "adapter_alignment_cell_evidence_v2"


def load_alignment_contract() -> Mapping[str, Any]:
    value = yaml.safe_load(ALIGNMENT_CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "adapter_alignment_gate_contract_v2":
        raise ValueError("adapter-alignment v2 contract is unavailable")
    active = value.get("execution_policy", {}).get("active_delivery_method")
    if active != "reve":
        raise PermissionError(f"REVE is not the active serial delivery method: {active!r}")
    return value


def comparison_fields(
    *,
    task: str,
    inventory: Any,
    alignment_contract: Mapping[str, Any],
    branch_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    """Use the already-frozen method-neutral implementation byte-for-byte."""

    return _method_neutral_comparison_fields(
        task=task,
        inventory=inventory,
        alignment_contract=alignment_contract,
        branch_fingerprints=branch_fingerprints,
    )


def adapter_identity(
    *,
    metadata: Any,
    config: Mapping[str, Any],
    config_path: Path,
    panel: Sequence[str],
    samples: int,
) -> dict[str, Any]:
    paths = {
        "adapter": METHOD_ROOT / "adapters/reve.py",
        "alignment_data": METHOD_ROOT / "alignment_data.py",
        "alignment_audit": METHOD_ROOT / "audit_alignment_v2.py",
        "identity_and_representation_audit": METHOD_ROOT
        / "IDENTITY_AND_REPRESENTATION_AUDIT.md",
        "checkpoint_model_code": METHOD_ROOT / "checkpoints/reve-base/modeling_reve.py",
        "checkpoint_position_code": METHOD_ROOT
        / "checkpoints/reve-positions/position_bank.py",
        "upstream_classifier": METHOD_ROOT / "upstream/src/models/classifier.py",
        "upstream_freeze_utility": METHOD_ROOT
        / "upstream/src/downstream_tasks/utils.py",
        "config": config_path,
    }
    identity = jsonable(asdict(metadata))
    patch_samples = int(metadata.patch_samples)
    patch_step = patch_samples - int(metadata.patch_overlap)
    patch_count = 1 + (int(samples) - patch_samples) // patch_step
    covered_samples = patch_samples + (patch_count - 1) * patch_step
    identity.update(
        {
            "path": portable_path(Path(metadata.path)),
            "position_path": portable_path(Path(metadata.position_path)),
            "method_id": "reve",
            "source_file_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
            "delivered_channel_order": list(panel),
            "delivered_channel_order_sha256": stable_hash(list(panel)),
            "input_shape": ["batch", len(panel), int(samples)],
            "output_shape": ["batch", int(metadata.embedding_dim)],
            "output_layer": str(config["adapter"]["output_layer"]),
            "patch_and_token_grid": {
                "patch_samples": patch_samples,
                "patch_overlap": int(metadata.patch_overlap),
                "patch_step": patch_step,
                "patches_per_channel": patch_count,
                "latent_tokens": len(panel) * patch_count,
                "latent_width": int(metadata.embedding_dim),
                "covered_input_samples": covered_samples,
                "discarded_real_tail_samples": int(samples) - covered_samples,
            },
            "geometry_encoding": (
                "registry_provenance_checked_exact_name_lookup_in_verified_"
                "official_543_electrode_position_bank"
            ),
            "pooling": str(config["adapter"]["pooling"]),
            "deterministic_source_declared_sample_transform": str(
                config["adapter"]["deterministic_source_declared_sample_transform"]
            ),
            "train_partition_fitted_transform": "none_in_alignment_audit",
            "trainable_parameter_boundary": "no_trainable_parameters",
            "source_deviation": str(config["adapter"]["source_deviation"]),
            "target_corpus_exposure": {
                "eeg_fnirs_single_trial": "confirmed_overlap_via_Shin2017A",
                "simultaneous_eeg_nirs": "none_in_published_exhaustive_list",
                "visual_cognitive_motivation": "none_in_published_exhaustive_list",
                "refed": "none_in_published_exhaustive_list",
            },
        }
    )
    return identity


def feature_digest(sample_ids: Sequence[str], features: np.ndarray) -> str:
    if features.shape[0] != len(sample_ids):
        raise ValueError("feature rows do not match sample identities")
    digest = hashlib.sha256()
    for identifier, row in sorted(zip(sample_ids, features, strict=True)):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(row, dtype=np.float32).tobytes())
    return digest.hexdigest()


def run_task(
    *,
    task: str,
    config: Mapping[str, Any],
    config_path: Path,
    alignment_contract: Mapping[str, Any],
    branch_fingerprints: Mapping[str, str],
    model: REVEFrozenEncoder,
    checkpoint_metadata: Any,
    device: torch.device,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    started = time.perf_counter()
    inventory = load_public_inventory(config, task=task)
    view = REVEPublicView(
        inventory, sample_rate_hz=float(config["data"]["eeg_sample_rate_hz"])
    )
    loader = make_loader(
        view,
        batch_size=int(config["resources"]["feature_batch_size"]),
        workers=int(config["resources"]["data_loader_workers"]),
    )
    print(
        f"[{task}] auditing {len(inventory.indices)} unique public samples "
        f"in {len(loader)} record-grouped batches",
        flush=True,
    )
    all_features: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    all_support_counts: list[np.ndarray] = []
    deterministic_replay = False
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            eeg = batch["eeg"].to(device, non_blocking=True)
            batch_size, channels, samples = eeg.shape
            kwargs = {
                "sampling_rate_hz": float(config["data"]["eeg_sample_rate_hz"]),
                "channel_names": inventory.panel,
                "channel_valid": torch.ones(
                    (batch_size, channels), dtype=torch.bool, device=device
                ),
                "sample_valid": torch.ones(
                    (batch_size, samples), dtype=torch.bool, device=device
                ),
            }
            embedding = model(eeg, **kwargs)
            if batch_number == 1:
                replay = model(eeg, **kwargs)
                deterministic_replay = bool(torch.equal(embedding, replay))
                if not deterministic_replay:
                    raise RuntimeError("REVE deterministic replay differed for identical input")
            if embedding.shape != (batch_size, 512) or not bool(
                torch.isfinite(embedding).all()
            ):
                raise RuntimeError(f"REVE produced invalid public features: {tuple(embedding.shape)}")
            all_features.append(embedding.float().cpu().numpy())
            all_indices.append(batch["dataset_index"].numpy())
            all_sample_ids.extend(str(value) for value in batch["sample_id"])
            all_support_counts.append(batch["recorded_support_count"].numpy())
            if batch_number % 100 == 0 or batch_number == len(loader):
                print(f"[{task}] {batch_number}/{len(loader)} batches", flush=True)

    features = np.concatenate(all_features).astype(np.float32, copy=False)
    dataset_indices = np.concatenate(all_indices).astype(np.int64, copy=False)
    support_counts = np.concatenate(all_support_counts).astype(np.int64, copy=False)
    expected_ids = set(inventory.sample_ids)
    observed_ids = set(all_sample_ids)
    if len(all_sample_ids) != len(observed_ids) or observed_ids != expected_ids:
        raise RuntimeError(f"REVE public coverage is incomplete or duplicated for {task}")
    if set(dataset_indices.tolist()) != set(inventory.indices):
        raise RuntimeError(f"REVE public dataset-index coverage drifted for {task}")
    required_samples = int(round(inventory.duration_s * 200.0))
    if not bool((support_counts == required_samples).all()):
        raise RuntimeError(f"REVE recorded-support counts drifted for {task}")
    coordinate_std = features.std(axis=0, dtype=np.float64)
    if not bool(np.isfinite(features).all()) or not bool((coordinate_std > 1e-8).any()):
        raise RuntimeError(f"REVE public features are non-finite or globally constant for {task}")

    fields = comparison_fields(
        task=task,
        inventory=inventory,
        alignment_contract=alignment_contract,
        branch_fingerprints=branch_fingerprints,
    )
    identity = adapter_identity(
        metadata=checkpoint_metadata,
        config=config,
        config_path=config_path,
        panel=inventory.panel,
        samples=required_samples,
    )
    cache_identity = {
        "adapter_identity": identity,
        "comparison_fields": fields,
        "audit_code_sha256": sha256_file(Path(__file__)),
    }
    class_counts = Counter(
        str(inventory.dataset.lightweight_metadata(index)["condition"])
        for index in inventory.indices
    )
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "cell_id": f"reve__{task}__support_matched_direct__v2",
        "comparison_group_id": f"single_modal_eeg__{task}__support_matched_direct__v2",
        "method_id": "reve",
        "task_id": task,
        "track": str(config["tasks"][task]["track"]),
        "alignment_profile": str(config["alignment_profile"]),
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
            "feature_shape": list(features.shape),
            "feature_sha256": feature_digest(all_sample_ids, features),
            "nonconstant_coordinate_count": int((coordinate_std > 1e-8).sum()),
            "minimum_coordinate_std": float(coordinate_std.min()),
            "maximum_coordinate_std": float(coordinate_std.max()),
            "deterministic_replay_exact": deterministic_replay,
            "cache_identity_sha256": stable_hash(cache_identity),
            "elapsed_seconds": time.perf_counter() - started,
            "protected_test_opened": False,
        },
    }
    output_path = output_root / f"{task}.json"
    write_json(output_path, evidence)
    del inventory, view, loader, features, all_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return evidence, output_path


def unsupported_refed_cell(
    *, config: Mapping[str, Any], alignment_contract: Mapping[str, Any]
) -> dict[str, Any]:
    task = "refed_regression"
    task_contract = alignment_contract["task_contracts"][task]
    task_config = config["tasks"][task]
    return {
        "schema": EVIDENCE_SCHEMA,
        "cell_id": "reve__refed_regression__support_matched_direct__v2",
        "comparison_group_id": "single_modal_eeg__refed_regression__support_matched_direct__v2",
        "method_id": "reve",
        "task_id": task,
        "track": str(task_config["track"]),
        "alignment_profile": str(config["alignment_profile"]),
        "evidence_scope": "static",
        "cell_status": "unsupported",
        "unsupported_reason_code": str(task_config["unsupported_reason_code"]),
        "unsupported_reason": str(task_config["unsupported_reason"]),
        "comparison_fields": {
            "dataset_id": str(task_contract["dataset_id"]),
            "task_id": task,
            "sample_inventory_sha256": "not_dereferenced_unsupported_before_public_signal_audit",
            "split_fingerprint": "method_neutral_registry_identity_only",
            "target_schema": str(task_contract["target_schema"]),
            "target_valid_mask": "partial_coordinate_mask_required_but_not_supported",
            "primary_endpoint": str(task_contract["primary_endpoint"]),
            "observation_anchor": str(task_contract["primary_observation"]["anchor"]),
            "modality_intervals_s": {
                "eeg": list(task_contract["primary_observation"]["eeg_interval_s"])
            },
            "modality_identity": ["eeg"],
            "measured_channel_identity_set": "not_applicable_unsupported",
            "recorded_support_mask": "partial_terminal_support_not_admitted",
            "canonical_signal_branch": "not_dereferenced_unsupported",
        },
        "adapter_identity": {
            "method_id": "reve",
            "artifact_id": str(config["adapter"]["artifact_id"]),
            "position_artifact_id": str(config["adapter"]["position_artifact_id"]),
            "time_mask_contract": "full_support_only",
            "source_deviation": str(config["adapter"]["source_deviation"]),
        },
        "gate_status": {
            "A0": "pass",
            "A1": "pass",
            "A2": "pass",
            "A3": "not_applicable",
            "A4": "unsupported",
            "A5": "not_applicable",
            "A6": "pass",
            "A7": "unsupported",
            "A8": "pending",
        },
        "protected_test_opened": False,
    }


def parse_tasks(values: Sequence[str]) -> tuple[str, ...]:
    tasks = tuple(values) if values else SUPPORTED_TASKS
    unknown = sorted(set(tasks) - set(SUPPORTED_TASKS))
    if unknown:
        raise ValueError(f"unknown or unsupported REVE audit tasks: {unknown}")
    if len(set(tasks)) != len(tasks):
        raise ValueError("REVE audit tasks must be unique")
    return tasks


def peer_paths(tasks: Sequence[str], *, include_refed: bool) -> list[Path]:
    names = list(tasks)
    if include_refed:
        names.append("refed_regression")
    paths = [root / f"{task}.json" for root in PEER_ROOTS for task in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen alignment peers are missing: {missing}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", action="append", default=[])
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    tasks = parse_tasks(args.task)
    output_root = resolve_repo_path(args.output_root)
    device = torch.device(args.device or config["resources"]["default_device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full REVE alignment audit")
    alignment_contract = load_alignment_contract()
    branch_fingerprints = data_branch_fingerprints(config)
    lock_path = Path(str(config["resources"]["gpu_lock_path"]))
    cell_paths: list[Path] = []
    task_reports: list[dict[str, Any]] = []
    with exclusive_gpu_lock(lock_path):
        encoder, position_bank, metadata = load_verified_reve_base(device=device)
        model = REVEFrozenEncoder(
            encoder,
            position_bank,
            sampling_rate_hz=float(config["data"]["eeg_sample_rate_hz"]),
            pooling=str(config["adapter"]["pooling"]),
        ).to(device).eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("REVE alignment audit found trainable encoder parameters")
        for task in tasks:
            evidence, path = run_task(
                task=task,
                config=config,
                config_path=config_path,
                alignment_contract=alignment_contract,
                branch_fingerprints=branch_fingerprints,
                model=model,
                checkpoint_metadata=metadata,
                device=device,
                output_root=output_root,
            )
            cell_paths.append(path)
            task_reports.append(
                {
                    "task": task,
                    "path": portable_path(path),
                    "sample_count": evidence["public_audit"]["unique_sample_count"],
                    "feature_sha256": evidence["public_audit"]["feature_sha256"],
                    "track": evidence["track"],
                    "status": "A0-A7_pass_A8_pending",
                }
            )

    full_run = set(tasks) == set(SUPPORTED_TASKS)
    if full_run:
        refed = unsupported_refed_cell(config=config, alignment_contract=alignment_contract)
        refed_path = output_root / "refed_regression.json"
        write_json(refed_path, refed)
        cell_paths.append(refed_path)
        task_reports.append(
            {
                "task": "refed_regression",
                "path": portable_path(refed_path),
                "status": "unsupported",
                "reason_code": refed["unsupported_reason_code"],
            }
        )

    peers = peer_paths(tasks, include_refed=full_run)
    schema_report = audit_evidence(ALIGNMENT_CONTRACT, [*peers, *cell_paths])
    for report in schema_report["cell_reports"]:
        report["source"] = portable_path(Path(str(report["source"])))
    summary = {
        "schema": "reve_adapter_alignment_audit_report_v2",
        "status": "implementation_review_complete_A0_A7_pass_A8_pending",
        "created_at": utc_now(),
        "method_id": "reve",
        "config_path": portable_path(config_path),
        "config_sha256": sha256_file(config_path),
        "alignment_contract_path": portable_path(ALIGNMENT_CONTRACT),
        "alignment_contract_sha256": sha256_file(ALIGNMENT_CONTRACT),
        "direct_alignment_peers": ["biot", "cbramod"],
        "peer_cell_paths": [portable_path(path) for path in peers],
        "tasks": task_reports,
        "schema_audit": schema_report,
        "protected_test_opened": False,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
