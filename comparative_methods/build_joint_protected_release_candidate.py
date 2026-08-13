#!/usr/bin/env python3
"""Build the immutable-input manifest for the 540-job protected campaign.

The builder consumes only public evidence plus the method-neutral registry's
already-recorded protected hashes.  It never opens a protected fold manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.protected_campaign_common import (  # noqa: E402
    CAMPAIGN_SCHEMA,
    CampaignError,
    environment_fingerprint,
    portable_path,
    read_json,
    repo_path,
    sha256_file,
    stable_hash,
    write_json_atomic,
)


CAMPAIGN_ID = "joint-comparison-protected-20260813-v2"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "comparative_methods/evidence/protected_campaign/joint_release_candidate_v1.json"
)
UNLOCK_CANDIDATE = (
    REPO_ROOT / "comparative_methods/evidence/joint_protected_unlock_candidate_v2.json"
)
REGISTRY = (
    REPO_ROOT
    / "comparative_methods/EFRM-PyTorch/runs/formal/"
    "efrm_lodo_full_target_fivefold_v2/protocol/"
    "shared_full_target_fold_registry/registry_manifest.json"
)
METRIC_TARGETS = REPO_ROOT / "comparative_methods/comparison_metric_targets_v1.yaml"
LANE_MANIFEST = (
    REPO_ROOT / "comparative_methods/evidence/protected_campaign/lane_manifest_v1.json"
)

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (17, 42, 73)


METHODS: dict[str, dict[str, Any]] = {
    "biot": {
        "method_id": "biot",
        "audit": "comparative_methods/BIOT/runs/public_development_v2/matrix_v2/completed_public_audit.json",
        "tasks": ("motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"),
        "kind": "biot_live_eeg",
        "runner": "comparative_methods/BIOT/run_public_development_v2.py",
        "alignment": "comparative_methods/BIOT/configs/alignment_v2.yaml",
    },
    "cbramod": {
        "method_id": "cbramod",
        "audit": "comparative_methods/CBraMod/runs/public_development_v2/matrix_v2/completed_public_audit.json",
        "tasks": ("motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"),
        "kind": "cbramod_live_eeg",
        "runner": "comparative_methods/CBraMod/run_public_development_v2.py",
        "alignment": "comparative_methods/CBraMod/configs/alignment_v2.yaml",
    },
    "reve": {
        "method_id": "reve",
        "audit": "comparative_methods/REVE/runs/public_development_v2/matrix_v2/completed_public_audit.json",
        "tasks": ("motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"),
        "kind": "reve_live_eeg",
        "runner": "comparative_methods/REVE/run_public_development_v2.py",
        "alignment": "comparative_methods/REVE/configs/alignment_v2.yaml",
    },
    "efrm": {
        "method_id": "efrm_sync_200_10_variable_channel_v1",
        "audit": (
            "comparative_methods/EFRM-PyTorch/runs/formal/"
            "efrm_lodo_full_target_fivefold_v2/downstream_public_v2/"
            "a7_complete_matrix/completed_public_audit.json"
        ),
        "tasks": (
            "motor_imagery",
            "mental_arithmetic",
            "wg",
            "nback",
            "dsr",
            "visual",
            "refed_regression",
        ),
        "kind": "efrm_npz",
        "runner": "comparative_methods/EFRM-PyTorch/run_downstream_public_v2.py",
        "alignment": "comparative_methods/EFRM-PyTorch/configs/downstream_public_v2.yaml",
    },
    "normwear": {
        "method_id": "normwear_eeg_fnirs_adapted",
        "audit": "comparative_methods/NormWear/runs/public_development_v2/matrix_v2/completed_public_audit.json",
        "tasks": ("motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"),
        "kind": "normwear_memmap",
        "runner": "comparative_methods/NormWear/run_public_development_v2.py",
        "alignment": "comparative_methods/NormWear/configs/alignment_v2.yaml",
    },
    "brainfusion": {
        "method_id": "brainfusion_nvc_csp_stacking_reimplementation",
        "audit": (
            "comparative_methods/BrainFusion-NVC-CSP-Stacking/"
            "runs/public_development_v2/matrix_v2/completed_public_audit.json"
        ),
        "tasks": ("motor_imagery", "mental_arithmetic", "wg", "nback", "visual"),
        "kind": "brainfusion_pipeline",
        "runner": (
            "comparative_methods/BrainFusion-NVC-CSP-Stacking/"
            "run_public_development_v2.py"
        ),
        "alignment": (
            "comparative_methods/BrainFusion-NVC-CSP-Stacking/"
            "configs/alignment_v2.yaml"
        ),
    },
}


def _serialized_json_sha256(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _file_descriptor(path: Path, role: str, cache: dict[Path, str]) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise CampaignError(f"missing frozen artifact: {resolved}")
    digest = cache.get(resolved)
    if digest is None:
        digest = sha256_file(resolved)
        cache[resolved] = digest
    return {
        "role": role,
        "path": portable_path(resolved),
        "sha256": digest,
        "size_bytes": resolved.stat().st_size,
    }


def _tensor_digest(values: Mapping[str, Any]) -> str:
    payload: list[dict[str, Any]] = []
    for name, value in sorted(values.items()):
        if value is None:
            payload.append({"name": name, "value": None})
        elif isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            payload.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "bytes_sha256": __import__("hashlib").sha256(array.tobytes()).hexdigest(),
                }
            )
        else:
            payload.append({"name": name, "value": value})
    return stable_hash(payload)


LIVE_EEG_KINDS = {"biot_live_eeg", "cbramod_live_eeg", "reve_live_eeg"}


def _checkpoint_contract(path: Path, kind: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if kind in {"linear_npz", "normwear_memmap", *LIVE_EEG_KINDS}:
        return {
            "schema": checkpoint.get("schema"),
            "method_id": checkpoint.get("method_id"),
            "task": checkpoint.get("task"),
            "outer_fold": int(checkpoint.get("outer_fold")),
            "seed": int(checkpoint.get("seed")),
            "standardizer_sha256": _tensor_digest(
                {
                    "feature_mean": checkpoint.get("feature_mean"),
                    "feature_scale": checkpoint.get("feature_scale"),
                }
            ),
            "head_sha256": _tensor_digest(checkpoint.get("head_state", {})),
        }
    if kind == "efrm_npz":
        return {
            "schema": checkpoint.get("schema"),
            "method_id": checkpoint.get("method_id"),
            "task": checkpoint.get("task"),
            "outer_fold": int(checkpoint.get("outer_fold")),
            "seed": int(checkpoint.get("seed")),
            "task_type": checkpoint.get("task_type"),
            "probe_sha256": _tensor_digest(checkpoint.get("probe_state", {})),
            "target_scaler_sha256": _tensor_digest(
                {
                    "target_center": checkpoint.get("target_center"),
                    "target_scale": checkpoint.get("target_scale"),
                }
            ),
        }
    raise CampaignError(f"unsupported checkpoint kind: {kind}")


def _run_report_path(row: Mapping[str, Any], kind: str) -> Path:
    if kind == "brainfusion_pipeline":
        return repo_path(str(row["run_report_path"]))
    return repo_path(str(row["run_dir"])) / "manifest.json"


def _artifacts_for_job(
    row: Mapping[str, Any],
    kind: str,
    digest_cache: dict[Path, str],
    *,
    expected_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path = _run_report_path(row, kind)
    report = read_json(report_path)
    observed_identity = {
        "method_id": report.get("method_id"),
        "task": report.get("task"),
        "outer_fold": int(report.get("outer_fold", -1)),
        "seed": int(report.get("seed", -1)),
    }
    if observed_identity != dict(expected_identity):
        raise CampaignError(f"public run identity differs: {report_path}")
    if (
        report.get("status") not in {"pass", "completed"}
        or report.get("protected_test_opened") is not False
    ):
        raise CampaignError(f"public run is not a protected-closed terminal: {report_path}")
    artifacts = [_file_descriptor(report_path, "public_run_manifest", digest_cache)]
    public_split_path = repo_path(str(report["public_manifest_path"]))
    public_split = _file_descriptor(
        public_split_path, "public_split_manifest", digest_cache
    )
    if public_split["sha256"] != report.get("public_manifest_sha256"):
        raise CampaignError(f"public split hash differs from its run report: {report_path}")
    artifacts.append(public_split)
    if kind == "brainfusion_pipeline":
        run_dir = report_path.parent
        checkpoint_dir = run_dir / "checkpoint"
        for filename, role in (
            ("manifest.json", "pipeline_manifest"),
            ("feature_state.pt", "pipeline_feature_state"),
            ("stacking.joblib", "pipeline_stacking"),
        ):
            descriptor = _file_descriptor(checkpoint_dir / filename, role, digest_cache)
            artifacts.append(descriptor)
            if role == "pipeline_manifest" and descriptor["sha256"] != report.get(
                "checkpoint_manifest_sha256"
            ):
                raise CampaignError("BrainFusion pipeline manifest hash drifted")
        cache_path = repo_path(str(report["tensor_cache"]["path"]))
        artifacts.append(_file_descriptor(cache_path, "feature_cache", digest_cache))
        cache_manifest = repo_path(str(report["tensor_cache"]["manifest_path"]))
        artifacts.append(
            _file_descriptor(cache_manifest, "feature_cache_manifest", digest_cache)
        )
        pipeline_hash = stable_hash(
            [{"role": item["role"], "sha256": item["sha256"]} for item in artifacts]
        )
        pipeline_manifest = read_json(checkpoint_dir / "manifest.json")
        audit = pipeline_manifest.get("audit", {})
        if (
            report.get("checkpoint_reload_exact") is not True
            or report.get("train_validation_overlap") is not False
            or pipeline_manifest.get("protected_test_opened") is not False
            or audit.get("all_fitted_state_outer_training_only") is not True
        ):
            raise CampaignError("BrainFusion frozen pipeline audit is not admissible")
        contract = {
            **observed_identity,
            "pipeline_sha256": pipeline_hash,
            "standardizer_sha256": "embedded_in_frozen_pipeline",
            "cache_contract_sha256": str(report["tensor_cache"]["identity_sha256"]),
        }
        return artifacts, contract

    checkpoint_path = report_path.parent / "checkpoint_public_refit.pt"
    artifacts.append(_file_descriptor(checkpoint_path, "downstream_checkpoint", digest_cache))
    if kind == "normwear_memmap":
        directory = repo_path(str(report["feature_cache"]["directory"]))
        cache_path = directory / "features.npy"
        metadata_path = directory / "metadata.npz"
        identity_path = directory / "identity.json"
        artifacts.extend(
            [
                _file_descriptor(cache_path, "feature_cache", digest_cache),
                _file_descriptor(metadata_path, "feature_metadata", digest_cache),
                _file_descriptor(identity_path, "feature_cache_manifest", digest_cache),
            ]
        )
        cache_identity = read_json(identity_path)
    else:
        feature = report["feature_cache"]
        cache_path = repo_path(str(feature["path"]))
        artifacts.append(_file_descriptor(cache_path, "feature_cache", digest_cache))
        manifest_path = cache_path.with_suffix(".json")
        artifacts.append(
            _file_descriptor(manifest_path, "feature_cache_manifest", digest_cache)
        )
        cache_identity = read_json(manifest_path)
    if kind in LIVE_EEG_KINDS:
        method_identity = report.get("method_identity", {})
        encoder = _file_descriptor(
            repo_path(str(method_identity.get("path", ""))),
            "encoder_checkpoint",
            digest_cache,
        )
        if (
            encoder["sha256"] != method_identity.get("sha256")
            or encoder["size_bytes"] != int(method_identity.get("size_bytes", -1))
        ):
            raise CampaignError("live EEG encoder identity differs from its public run")
        artifacts.append(encoder)
        if kind == "reve_live_eeg":
            position = _file_descriptor(
                repo_path(str(method_identity.get("position_path", ""))),
                "position_bank",
                digest_cache,
            )
            if (
                position["sha256"] != method_identity.get("position_sha256")
                or position["size_bytes"]
                != int(method_identity.get("position_size_bytes", -1))
            ):
                raise CampaignError("REVE position identity differs from its public run")
            artifacts.append(position)
    contract = _checkpoint_contract(checkpoint_path, kind)
    for field, expected in expected_identity.items():
        if contract.get(field) != expected:
            raise CampaignError(
                f"checkpoint {field} differs from its public run: {checkpoint_path}"
            )
    contract["cache_contract_sha256"] = stable_hash(cache_identity)
    if kind in LIVE_EEG_KINDS:
        contract["protected_feature_source"] = (
            "hash_pinned_frozen_encoder_over_exact_authorized_indices"
        )
        contract["encoder_checkpoint_sha256"] = encoder["sha256"]
        if kind == "reve_live_eeg":
            contract["position_bank_sha256"] = position["sha256"]
    return artifacts, contract


def _registry_entries() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    registry = read_json(REGISTRY)
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for row in registry["folds"]:
        if row.get("protocol") != "strict_cross_subject":
            continue
        key = (str(row["task"]), int(row["outer_fold"]))
        if key in entries:
            raise CampaignError(f"duplicate strict registry entry: {key}")
        entries[key] = dict(row)
    expected = {(task, fold) for task in (
        "motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual", "refed_regression"
    ) for fold in FOLDS}
    if set(entries) != expected:
        raise CampaignError("strict protected registry is incomplete")
    return entries, registry


def _cells() -> list[dict[str, Any]]:
    source = read_json(UNLOCK_CANDIDATE)
    output: list[dict[str, Any]] = []
    for row in source["cells"]:
        supported = row["cell_status"] == "pass"
        overlap = (
            supported
            and row["method_id"] == "reve"
            and row["task_id"] in {"motor_imagery", "mental_arithmetic"}
        )
        disposition = "overlap" if overlap else "direct" if supported else "unsupported"
        output.append(
            {
                **row,
                "campaign_disposition": disposition,
                "job_count": 15 if supported else 0,
                "metric_target": (
                    "native_coordinate_masked_ccc"
                    if row["task_id"] == "refed_regression"
                    else "macro_f1"
                ),
                "claim_boundary": (
                    "overlap_track_only"
                    if overlap
                    else "unsupported_no_job"
                    if not supported
                    else "support_matched_direct"
                ),
            }
        )
    return output


def _source_snapshot() -> list[dict[str, str]]:
    paths = {
        Path(__file__).resolve(),
        REPO_ROOT / "comparative_methods/protected_campaign_common.py",
        REPO_ROOT / "comparative_methods/protected_campaign_worker.py",
        REPO_ROOT / "comparative_methods/protected_campaign_controller.py",
        REPO_ROOT / "comparative_methods/aggregate_protected_campaign.py",
        REPO_ROOT / "comparative_methods/prepare_protected_authorization.py",
        REPO_ROOT / "comparative_methods/benchmark_protected_campaign_shadow.py",
        UNLOCK_CANDIDATE,
        METRIC_TARGETS,
    }
    for spec in METHODS.values():
        paths.add(repo_path(spec["runner"]))
        paths.add(repo_path(spec["alignment"]))
    for method_root, extra in (
        (
            REPO_ROOT / "comparative_methods/BIOT",
            ("alignment_data.py", "adapters/biot.py", "upstream/model/biot.py", "sources/method_manifest.yaml"),
        ),
        (
            REPO_ROOT / "comparative_methods/CBraMod",
            (
                "alignment_data.py",
                "adapters/cbramod.py",
                "upstream/models/cbramod.py",
                "upstream/models/criss_cross_transformer.py",
                "sources/method_manifest.yaml",
            ),
        ),
        (
            REPO_ROOT / "comparative_methods/REVE",
            (
                "alignment_data.py",
                "adapters/reve.py",
                "checkpoints/reve-base/configuration_reve.py",
                "checkpoints/reve-base/modeling_reve.py",
                "checkpoints/reve-positions/configuration_bank.py",
                "checkpoints/reve-positions/position_bank.py",
                "sources/method_manifest.yaml",
            ),
        ),
    ):
        paths.update(method_root / value for value in extra)
    brainfusion_adapters = (
        REPO_ROOT / "comparative_methods/BrainFusion-NVC-CSP-Stacking/adapters"
    )
    paths.update(brainfusion_adapters.rglob("*.py"))
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        raise CampaignError(f"controlled source is missing: {missing[0]}")
    return [
        {"path": portable_path(path), "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]


def build_candidate() -> dict[str, Any]:
    entries, registry = _registry_entries()
    cells = _cells()
    cell_lookup = {
        (str(row["method_id"]), str(row["task_id"])): row for row in cells
    }
    digest_cache: dict[Path, str] = {}
    split_entries = [
        {
            "task": task,
            "outer_fold": fold,
            "dataset_id": entries[(task, fold)]["dataset_id"],
            "protocol": entries[(task, fold)]["protocol"],
            "public_manifest_sha256": entries[(task, fold)]["public_sha256"],
            "protected_manifest_sha256": entries[(task, fold)]["protected_sha256"],
            "protected_indices_sha256": entries[(task, fold)]["protected_indices_sha256"],
            "train_sample_count": entries[(task, fold)]["train_sample_count"],
            "validation_sample_count": entries[(task, fold)]["validation_sample_count"],
            "protected_sample_count": entries[(task, fold)]["protected_sample_count"],
        }
        for task, fold in sorted(entries)
    ]
    split_fingerprint_sha256 = stable_hash(split_entries)
    registry_file_sha256 = sha256_file(REGISTRY)
    jobs: list[dict[str, Any]] = []
    for method_slug, spec in METHODS.items():
        audit_path = repo_path(spec["audit"])
        audit = read_json(audit_path)
        if audit.get("status") != "pass" or audit.get("protected_test_opened") is not False:
            raise CampaignError(f"public completion is not a protected-closed pass: {audit_path}")
        reports = {
            (str(row["task"]), int(row["outer_fold"]), int(row["seed"])): row
            for row in audit["run_reports"]
        }
        expected = {(task, fold, seed) for task in spec["tasks"] for fold in FOLDS for seed in SEEDS}
        if set(reports) != expected:
            raise CampaignError(f"public artifact matrix drifted for {method_slug}")
        for task in spec["tasks"]:
            cell = cell_lookup[(str(spec["method_id"]), task)]
            if cell["campaign_disposition"] == "unsupported":
                raise CampaignError(f"supported job routed to unsupported cell: {method_slug}/{task}")
            for fold in FOLDS:
                protected = entries[(task, fold)]
                for seed in SEEDS:
                    public_row = reports[(task, fold, seed)]
                    expected_identity = {
                        "method_id": spec["method_id"],
                        "task": task,
                        "outer_fold": fold,
                        "seed": seed,
                    }
                    artifacts, frozen_contract = _artifacts_for_job(
                        public_row,
                        str(spec["kind"]),
                        digest_cache,
                        expected_identity=expected_identity,
                    )
                    public_artifact = next(
                        artifact
                        for artifact in artifacts
                        if artifact["role"] == "public_split_manifest"
                    )
                    if public_artifact["sha256"] != protected["public_sha256"]:
                        raise CampaignError(
                            f"public split differs from registry: {method_slug}/{task}/outer{fold}"
                        )
                    job_id = f"{method_slug}__{task}__outer{fold}__seed{seed}"
                    input_contract = {
                        "registry_sha256": registry_file_sha256,
                        "split_fingerprint_sha256": split_fingerprint_sha256,
                        "protected_manifest_path": portable_path(
                            Path(str(protected["protected_path"]))
                        ),
                        "protected_manifest_sha256": str(protected["protected_sha256"]),
                        "protected_indices_sha256": str(protected["protected_indices_sha256"]),
                        "protected_sample_count": int(protected["protected_sample_count"]),
                        "dataset_id": str(protected["dataset_id"]),
                        "protocol": "strict_cross_subject",
                    }
                    jobs.append(
                        {
                            "job_id": job_id,
                            "method_slug": method_slug,
                            "method_id": spec["method_id"],
                            "task": task,
                            "track": cell["track"],
                            "campaign_disposition": cell["campaign_disposition"],
                            "outer_fold": fold,
                            "seed": seed,
                            "worker_kind": spec["kind"],
                            "metric_target": cell["metric_target"],
                            "artifacts": artifacts,
                            "frozen_inference_contract": frozen_contract,
                            "input_contract": {
                                **input_contract,
                                "sha256": stable_hash(input_contract),
                            },
                            "expected_outputs": [
                                "job_manifest.json",
                                "status.json",
                                "protected_predictions.npz",
                                "artifact_checksums.json",
                                "audit_report.json",
                            ],
                        }
                    )
    jobs.sort(key=lambda row: str(row["job_id"]))
    if len(jobs) != 540 or len({row["job_id"] for row in jobs}) != 540:
        raise CampaignError("formal job matrix is not exactly 540 unique jobs")

    direct = sum(row["campaign_disposition"] == "direct" for row in cells)
    overlap = sum(row["campaign_disposition"] == "overlap" for row in cells)
    unsupported = sum(row["campaign_disposition"] == "unsupported" for row in cells)
    if (direct, overlap, unsupported) != (34, 2, 6):
        raise CampaignError("cell disposition counts drifted")
    source_snapshot = _source_snapshot()
    environment = environment_fingerprint()
    lane_manifest: dict[str, Any] | None = None
    if LANE_MANIFEST.is_file():
        lane_manifest = {
            "path": portable_path(LANE_MANIFEST),
            "sha256": sha256_file(LANE_MANIFEST),
            "value": read_json(LANE_MANIFEST),
        }
    lane_ready = bool(lane_manifest and lane_manifest["value"].get("status") == "pass")
    candidate = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "candidate_version": 1,
        "created_on": "2026-08-12",
        "state": "REVIEWED" if lane_ready else "DRAFT",
        "orr_decision": "PENDING_DUAL_GO" if lane_ready else "NO_GO_PENDING_SHADOW_LANE",
        "git_commit": _git_commit(),
        "code_snapshot": {
            "files": source_snapshot,
            "sha256": stable_hash(source_snapshot),
        },
        "environment": environment,
        "data_registry": {
            "path": portable_path(REGISTRY),
            "sha256": registry_file_sha256,
            "registry_sha256": registry.get("registry_sha256"),
            "protected_test_default": registry.get("protected_test_default"),
        },
        "split_fingerprint": {
            "schema": "joint_protected_campaign_split_fingerprint_v1",
            "entries": split_entries,
            "sha256": split_fingerprint_sha256,
        },
        "metric_targets": {
            "path": portable_path(METRIC_TARGETS),
            "sha256": sha256_file(METRIC_TARGETS),
            "classification_primary": "macro_f1",
            "refed_primary": "native_coordinate_masked_ccc",
            "classification_companions": [
                "accuracy",
                "balanced_accuracy",
                "cohen_kappa",
            ],
            "refed_companions": ["pearson", "spearman", "r2", "mae", "rmse"],
            "oof_companion": "primary_metric_recomputed_on_all_five_fold_payloads_per_seed",
            "aggregation": "seed_metrics_mean_within_fold_then_five_fold_mean_and_sample_sd",
        },
        "disposition_counts": {
            "direct": direct,
            "overlap": overlap,
            "supported": direct + overlap,
            "unsupported": unsupported,
            "jobs": len(jobs),
        },
        "cells": cells,
        "jobs": jobs,
        "job_matrix_sha256": stable_hash(jobs),
        "lane_manifest": lane_manifest,
        "pre_lane_candidate_sha256": None,
        "failure_policy": {
            "maximum_attempts_per_job": 2,
            "retry_only_for": [
                "gpu_or_process_failure",
                "temporary_io_failure",
                "non_human_worker_interruption",
            ],
            "invalid_output_is_not_retryable": True,
            "performance_based_retry_forbidden": True,
            "attempt_2_device_policy": "same_frozen_gpu_uuid_only",
            "unavailable_assigned_gpu_terminal": (
                "INCOMPLETE_TECHNICAL_requires_new_candidate_and_dual_authorization"
            ),
            "second_technical_failure_terminal": "INCOMPLETE_TECHNICAL",
        },
        "state_machine": {
            "states": [
                "DRAFT",
                "REVIEWED",
                "AUTHORIZED",
                "RUNNING",
                "INCOMPLETE_TECHNICAL",
                "SEALED_COMPLETE",
                "UNBLINDED",
                "AGGREGATED",
                "RELEASED",
            ],
            "fail_closed": True,
            "authorization_and_unblind_are_separate_dual_signature_transitions": True,
        },
        "blinding_policy": {
            "worker_computes_metrics": False,
            "controller_displays_metrics": False,
            "operator_prediction_access_forbidden_before_unblind": True,
            "aggregator_requires_sealed_complete_and_dual_unblind": True,
        },
        "storage_policy": {
            "temporary_directory_then_atomic_commit": True,
            "incomplete_attempt_destination": "quarantine",
            "completed_output_overwrite": "forbidden",
            "minimum_free_space_multiplier": 2.0,
        },
        "sta_net": {
            "new_job_count": 0,
            "disposition": "method_native_context_reference",
        },
        "protected_evaluation_authorized": False,
        "protected_test_opened": False,
    }
    if lane_manifest is not None:
        pre_lane_candidate = {
            **candidate,
            "state": "DRAFT",
            "orr_decision": "NO_GO_PENDING_SHADOW_LANE",
            "lane_manifest": None,
            "pre_lane_candidate_sha256": None,
        }
        pre_lane_sha256 = _serialized_json_sha256(pre_lane_candidate)
        if (
            lane_manifest["value"].get("candidate_sha256_before_lane_freeze")
            != pre_lane_sha256
        ):
            raise CampaignError(
                "lane manifest was not benchmarked against the current pre-lane candidate"
            )
        candidate["pre_lane_candidate_sha256"] = pre_lane_sha256
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    candidate = build_candidate()
    output = args.output.resolve()
    if args.check:
        retained = read_json(output)
        if retained != candidate:
            raise CampaignError(f"stale release candidate: {output}")
    else:
        write_json_atomic(output, candidate)
    print(
        json.dumps(
            {
                "status": "pass",
                "candidate": portable_path(output),
                "campaign_id": candidate["campaign_id"],
                "orr_decision": candidate["orr_decision"],
                "cells": len(candidate["cells"]),
                "jobs": len(candidate["jobs"]),
                "protected_test_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
