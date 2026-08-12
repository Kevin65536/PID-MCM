#!/usr/bin/env python3
"""Aggregate a sealed campaign only after a separately dual-signed unblind."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.protected_campaign_common import (  # noqa: E402
    CampaignError,
    artifact_map,
    index_jobs,
    portable_path,
    read_json,
    repo_path,
    sha256_file,
    stable_hash,
    utc_now,
    verify_authorization,
    verify_candidate_file,
    verify_file,
    verify_runtime_environment,
    verify_unblind,
    write_json_atomic,
)
from comparative_methods.protected_campaign_controller import campaign_status  # noqa: E402
from comparative_methods.protected_campaign_worker import (  # noqa: E402
    _validate_predictions,
)


def _ccc(target: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> float:
    values = []
    for channel in range(target.shape[1]):
        mask = valid[:, channel].reshape(-1)
        truth = target[:, channel].reshape(-1)[mask]
        estimate = prediction[:, channel].reshape(-1)[mask]
        if len(truth) < 2:
            continue
        truth_mean = float(truth.mean())
        estimate_mean = float(estimate.mean())
        truth_var = float(np.mean(np.square(truth - truth_mean)))
        estimate_var = float(np.mean(np.square(estimate - estimate_mean)))
        covariance = float(np.mean((truth - truth_mean) * (estimate - estimate_mean)))
        denominator = truth_var + estimate_var + (truth_mean - estimate_mean) ** 2
        values.append(0.0 if denominator <= 0.0 else 2.0 * covariance / denominator)
    if not values:
        raise CampaignError("masked CCC has no valid channel")
    return float(np.mean(values))


def _companions(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    metric: str,
    valid: np.ndarray | None = None,
) -> dict[str, float]:
    if metric == "macro_f1":
        return {
            "accuracy": float(accuracy_score(target, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
            "cohen_kappa": float(cohen_kappa_score(target, prediction)),
        }
    if valid is None:
        raise CampaignError("REFED companion metrics require a validity mask")
    values: dict[str, list[float]] = {
        "pearson": [],
        "spearman": [],
        "r2": [],
        "mae": [],
        "rmse": [],
    }
    for channel in range(target.shape[1]):
        mask = valid[:, channel].reshape(-1)
        truth = target[:, channel].reshape(-1)[mask]
        estimate = prediction[:, channel].reshape(-1)[mask]
        if len(truth) < 2:
            continue
        values["pearson"].append(
            0.0 if np.std(truth) == 0 or np.std(estimate) == 0 else float(pearsonr(truth, estimate).statistic)
        )
        values["spearman"].append(
            0.0 if np.std(truth) == 0 or np.std(estimate) == 0 else float(spearmanr(truth, estimate).statistic)
        )
        values["r2"].append(float(r2_score(truth, estimate)))
        values["mae"].append(float(mean_absolute_error(truth, estimate)))
        values["rmse"].append(float(np.sqrt(mean_squared_error(truth, estimate))))
    if not values["mae"]:
        raise CampaignError("REFED companion metrics have no valid channel")
    output = {name: float(np.mean(rows)) for name, rows in values.items()}
    if any(not math.isfinite(value) for value in output.values()):
        raise CampaignError("non-finite companion metric")
    return output


def _primary(path: Path, metric: str) -> tuple[float, np.ndarray, dict[str, float]]:
    with np.load(path, allow_pickle=False) as payload:
        target = payload["target"]
        prediction = payload["prediction"]
        if metric == "macro_f1":
            value = float(f1_score(target, prediction, average="macro"))
            companions = _companions(target, prediction, metric=metric)
        else:
            valid = payload["target_valid_mask"].astype(bool)
            value = _ccc(target, prediction, valid)
            companions = _companions(
                target, prediction, metric=metric, valid=valid
            )
    if not math.isfinite(value):
        raise CampaignError(f"non-finite primary metric: {path}")
    return value, target, companions


def _training_targets(job: Mapping[str, Any]) -> np.ndarray:
    artifacts = artifact_map(job)
    kind = str(job["worker_kind"])
    if kind == "brainfusion_pipeline":
        report = read_json(repo_path(str(artifacts["public_run_manifest"]["path"])))
        public = read_json(repo_path(str(report["public_manifest_path"])))
        indices = [int(value) for value in public["train_indices"]]
        payload = torch.load(
            repo_path(str(artifacts["feature_cache"]["path"])),
            map_location="cpu",
            weights_only=True,
        )
        dataset_indices = payload["dataset_indices"].numpy().astype(np.int64)
        targets = payload["targets"].numpy().astype(np.int64)
    else:
        checkpoint = torch.load(
            repo_path(str(artifacts["downstream_checkpoint"]["path"])),
            map_location="cpu",
            weights_only=True,
        )
        indices = [int(value) for value in checkpoint["refit_dataset_indices"].tolist()]
        if kind == "normwear_memmap":
            metadata_path = repo_path(str(artifacts["feature_metadata"]["path"]))
            with np.load(metadata_path, allow_pickle=False) as payload:
                dataset_indices = payload["dataset_indices"].astype(np.int64)
                targets = payload["targets"].astype(np.int64)
        else:
            with np.load(
                repo_path(str(artifacts["feature_cache"]["path"])), allow_pickle=False
            ) as payload:
                dataset_indices = payload["dataset_indices"].astype(np.int64)
                targets = payload["targets"].astype(np.int64)
    lookup = {int(value): row for row, value in enumerate(dataset_indices.tolist())}
    return np.asarray([targets[lookup[index]] for index in indices], dtype=np.int64)


def _macro_f1_baseline(training: np.ndarray, evaluation: np.ndarray) -> float:
    labels = np.asarray(sorted(set(training.tolist()) | set(evaluation.tolist())), dtype=np.int64)
    counts = np.asarray([(training == label).sum() for label in labels], dtype=np.float64)
    prior = counts / counts.sum()
    evaluation_prior = np.asarray(
        [(evaluation == label).sum() for label in labels], dtype=np.float64
    )
    evaluation_prior /= evaluation_prior.sum()
    majority_label = labels[int(np.argmax(counts))]
    majority = float(
        f1_score(
            evaluation,
            np.full(len(evaluation), majority_label, dtype=np.int64),
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    denominator = prior + evaluation_prior
    expected_per_class = np.divide(
        2.0 * prior * evaluation_prior,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return max(majority, float(expected_per_class.mean()))


def _verify_job(
    directory: Path,
    *,
    job: Mapping[str, Any],
    candidate_sha256: str,
    authorization_sha256: str,
    environment_sha256: str,
) -> tuple[Path, dict[str, str]]:
    status = read_json(directory / "status.json")
    if (
        status.get("schema") != "joint_protected_campaign_job_v1"
        or status.get("job_id") != job["job_id"]
        or status.get("status") != "COMPLETED"
        or status.get("surface") != "protected"
        or status.get("protected_test_opened") is not True
        or status.get("performance_computed") is not False
        or status.get("failure_code") is not None
        or int(status.get("attempt", 0)) not in {1, 2}
        or status.get("candidate_sha256") != candidate_sha256
        or status.get("authorization_sha256") != authorization_sha256
    ):
        raise CampaignError(f"job is not a completed protected terminal: {directory.name}")
    manifest = read_json(directory / "job_manifest.json")
    expected_identity = {
        "job_id": job["job_id"],
        "method_id": job["method_id"],
        "task": job["task"],
        "outer_fold": job["outer_fold"],
        "seed": job["seed"],
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()) or (
        manifest.get("schema") != "protected_job_manifest_v1"
        or manifest.get("candidate_sha256") != candidate_sha256
        or manifest.get("authorization_sha256") != authorization_sha256
        or manifest.get("attempt") != status.get("attempt")
        or manifest.get("surface") != "protected"
        or manifest.get("protected_test_opened") is not True
        or manifest.get("performance_computed") is not False
        or manifest.get("input_manifest_sha256")
        != job["input_contract"]["protected_manifest_sha256"]
        or manifest.get("input_contract_sha256") != job["input_contract"]["sha256"]
        or manifest.get("frozen_inference_contract_sha256")
        != stable_hash(job["frozen_inference_contract"])
        or manifest.get("environment_sha256") != environment_sha256
        or manifest.get("artifact_sha256")
        != {
            role: value["sha256"]
            for role, value in sorted(artifact_map(job).items())
        }
    ):
        raise CampaignError(f"job manifest differs from the frozen job: {directory.name}")
    lane = next(
        row
        for row in job.get("_lane_assignments", [])
        if row.get("method_slug") == job["method_slug"]
    ) if job.get("_lane_assignments") else None
    if lane is not None and manifest.get("device_uuid") != lane.get("gpu_uuid"):
        raise CampaignError(f"job device differs from frozen lane: {directory.name}")
    determinism = {
        "seed": int(job["seed"]),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }
    if (
        manifest.get("determinism_sha256") != stable_hash(determinism)
        or status.get("device_uuid") != manifest.get("device_uuid")
    ):
        raise CampaignError(f"job determinism/device binding differs: {directory.name}")
    if int(status["attempt"]) == 2:
        retained_attempts = [
            read_json(path / "status.json")
            for path in sorted(
                (directory.parent / "quarantine").glob(
                    f"{job['job_id']}.attempt1*"
                )
            )
            if (path / "status.json").is_file()
        ]
        if not any(
            row.get("attempt") == 1
            and row.get("failure_code") == "FAILED_TECHNICAL"
            and row.get("performance_computed") is False
            and row.get("candidate_sha256") == candidate_sha256
            and row.get("authorization_sha256") == authorization_sha256
            for row in retained_attempts
        ):
            raise CampaignError(f"attempt-2 recovery lacks an eligible attempt-1: {directory.name}")
    checksums = read_json(directory / "artifact_checksums.json")
    expected_checksum_files = {
        "job_manifest.json",
        "protected_predictions.npz",
        "audit_report.json",
        "status.json",
    }
    if (
        checksums.get("schema") != "protected_job_artifact_checksums_v1"
        or checksums.get("job_id") != job["job_id"]
        or set(checksums.get("files", {})) != expected_checksum_files
    ):
        raise CampaignError(f"job checksum contract differs: {directory.name}")
    for name, expected in checksums["files"].items():
        if sha256_file(directory / name) != expected:
            raise CampaignError(f"job artifact checksum drifted: {directory / name}")
    prediction = directory / "protected_predictions.npz"
    if not prediction.is_file():
        raise CampaignError(f"missing protected predictions: {directory.name}")
    protected_manifest_path = verify_file(
        str(job["input_contract"]["protected_manifest_path"]),
        str(job["input_contract"]["protected_manifest_sha256"]),
        label=f"unblinded protected manifest for {job['job_id']}",
    )
    protected_manifest = read_json(protected_manifest_path)
    indices = [int(value) for value in protected_manifest.get("test_indices", [])]
    if (
        len(indices) != int(job["input_contract"]["protected_sample_count"])
        or stable_hash(sorted(indices))
        != job["input_contract"]["protected_indices_sha256"]
    ):
        raise CampaignError(f"protected coverage differs for {directory.name}")
    with np.load(prediction, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    observed_audit = _validate_predictions(
        arrays,
        expected_count=len(indices),
        expected_indices=indices,
        job=job,
    )
    audit = read_json(directory / "audit_report.json")
    expected_audit = {
        **observed_audit,
        "job_id": job["job_id"],
        "surface": "protected",
        "protected_test_opened": True,
    }
    if audit != expected_audit:
        raise CampaignError(f"job audit cannot be reproduced: {directory.name}")
    for runtime_name in (
        "job_manifest.json",
        "status.json",
        "audit_report.json",
        "artifact_checksums.json",
    ):
        serialized = (directory / runtime_name).read_text(encoding="utf-8").lower()
        if any(
            token in serialized
            for token in ("target", "logits", "metric", "confusion", "sample_id")
        ):
            raise CampaignError(f"runtime log redaction failed: {directory.name}")
    return prediction, {
        "job_id": str(job["job_id"]),
        "prediction_sha256": sha256_file(prediction),
        "artifact_checksums_sha256": sha256_file(
            directory / "artifact_checksums.json"
        ),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    candidate, candidate_sha256 = verify_candidate_file(
        args.candidate.resolve(), verify_artifacts=True
    )
    environment_sha256 = verify_runtime_environment(candidate)
    _authorization, authorization_sha256 = verify_authorization(
        args.authorization.resolve(),
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        enforce_window=False,
    )
    verify_unblind(
        args.unblind.resolve(),
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        authorization_sha256=authorization_sha256,
        authorization=_authorization,
    )
    campaign_root = args.campaign_root.resolve()
    status = campaign_status(
        candidate, campaign_root, candidate_sha256=candidate_sha256
    )
    if status["state"] != "SEALED_COMPLETE":
        raise CampaignError(f"aggregator requires SEALED_COMPLETE, observed {status['state']}")
    audit_log = campaign_root / "audit/events.jsonl"
    if not audit_log.is_file():
        raise CampaignError("campaign append-only audit log is absent")
    audit_text = audit_log.read_text(encoding="utf-8")
    if any(
        token in audit_text.lower()
        for token in ("target", "logits", "metric", "confusion", "sample_id")
    ):
        raise CampaignError("campaign append-only audit log violates redaction")
    audit_events = [json.loads(line) for line in audit_text.splitlines() if line.strip()]
    if not audit_events or any(not isinstance(row, dict) for row in audit_events):
        raise CampaignError("campaign append-only audit log is malformed")

    jobs = index_jobs(candidate)
    lane_assignments = candidate["lane_manifest"]["value"]["assignments"]
    for job in jobs.values():
        job["_lane_assignments"] = lane_assignments
    seed_rows: list[dict[str, Any]] = []
    prediction_paths: dict[str, Path] = {}
    job_traceability: list[dict[str, str]] = []
    for job_id, job in sorted(jobs.items()):
        prediction_path, trace = _verify_job(
            campaign_root / job_id,
            job=job,
            candidate_sha256=candidate_sha256,
            authorization_sha256=authorization_sha256,
            environment_sha256=environment_sha256,
        )
        prediction_paths[job_id] = prediction_path
        job_traceability.append(trace)
        value, target, companions = _primary(
            prediction_path, str(job["metric_target"])
        )
        baseline = 0.0
        if job["metric_target"] == "macro_f1":
            baseline = _macro_f1_baseline(_training_targets(job), target)
        seed_rows.append(
            {
                "job_id": job_id,
                "method_slug": job["method_slug"],
                "method_id": job["method_id"],
                "task": job["task"],
                "track": job["track"],
                "campaign_disposition": job["campaign_disposition"],
                "outer_fold": job["outer_fold"],
                "seed": job["seed"],
                "metric": job["metric_target"],
                "value": value,
                "companion_metrics": companions,
                "B0": baseline,
            }
        )

    oof_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in seed_rows:
        key = (str(row["method_slug"]), str(row["task"]), int(row["seed"]))
        oof_groups.setdefault(key, []).append(row)
    oof_rows: list[dict[str, Any]] = []
    for (method, task, seed), rows in sorted(oof_groups.items()):
        if sorted(int(row["outer_fold"]) for row in rows) != [0, 1, 2, 3, 4]:
            raise CampaignError(f"OOF fold support incomplete: {method}/{task}/seed{seed}")
        targets = []
        predictions = []
        masks = []
        for row in sorted(rows, key=lambda value: int(value["outer_fold"])):
            with np.load(prediction_paths[str(row["job_id"])], allow_pickle=False) as payload:
                targets.append(payload["target"])
                predictions.append(payload["prediction"])
                if "target_valid_mask" in payload.files:
                    masks.append(payload["target_valid_mask"].astype(bool))
        target = np.concatenate(targets, axis=0)
        prediction = np.concatenate(predictions, axis=0)
        metric = str(rows[0]["metric"])
        if metric == "macro_f1":
            value = float(f1_score(target, prediction, average="macro"))
            companions = _companions(target, prediction, metric=metric)
        else:
            valid = np.concatenate(masks, axis=0)
            value = _ccc(target, prediction, valid)
            companions = _companions(
                target, prediction, metric=metric, valid=valid
            )
        oof_rows.append(
            {
                "method_slug": method,
                "method_id": rows[0]["method_id"],
                "task": task,
                "track": rows[0]["track"],
                "seed": seed,
                "metric": metric,
                "value": value,
                "companion_metrics": companions,
                "companion_only": True,
                "job_ids": [
                    row["job_id"]
                    for row in sorted(rows, key=lambda value: int(value["outer_fold"]))
                ],
            }
        )

    fold_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in seed_rows:
        key = (str(row["method_slug"]), str(row["task"]), int(row["outer_fold"]))
        fold_groups.setdefault(key, []).append(row)
    fold_rows = []
    for (method, task, fold), rows in sorted(fold_groups.items()):
        if sorted(int(row["seed"]) for row in rows) != [17, 42, 73]:
            raise CampaignError(f"seed support incomplete: {method}/{task}/outer{fold}")
        fold_rows.append(
            {
                "method_slug": method,
                "method_id": rows[0]["method_id"],
                "task": task,
                "track": rows[0]["track"],
                "campaign_disposition": rows[0]["campaign_disposition"],
                "outer_fold": fold,
                "metric": rows[0]["metric"],
                "seed_mean": float(np.mean([row["value"] for row in rows])),
                "seed_sample_sd": float(statistics.stdev(row["value"] for row in rows)),
                "B0_seed_mean": float(np.mean([row["B0"] for row in rows])),
                "companion_metrics_seed_mean": {
                    name: float(
                        np.mean([row["companion_metrics"][name] for row in rows])
                    )
                    for name in rows[0]["companion_metrics"]
                },
                "job_ids": [row["job_id"] for row in sorted(rows, key=lambda x: x["seed"])],
            }
        )

    cell_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in fold_rows:
        cell_groups.setdefault((str(row["method_slug"]), str(row["task"])), []).append(row)
    cell_rows = []
    for (method, task), rows in sorted(cell_groups.items()):
        if sorted(int(row["outer_fold"]) for row in rows) != [0, 1, 2, 3, 4]:
            raise CampaignError(f"fold support incomplete: {method}/{task}")
        value = float(np.mean([row["seed_mean"] for row in rows]))
        fold_sd = float(statistics.stdev(row["seed_mean"] for row in rows))
        baseline = float(np.mean([row["B0_seed_mean"] for row in rows]))
        minimum = min(1.0, baseline + 0.02)
        decision = "TABLE_READY_WITH_NOTE" if value > baseline and value >= minimum else "REJECTED_VALUE"
        if rows[0]["campaign_disposition"] == "overlap":
            terminal = "OVERLAP_TRACK_ONLY"
        else:
            terminal = decision
        cell_rows.append(
            {
                "method_slug": method,
                "method_id": rows[0]["method_id"],
                "task": task,
                "track": rows[0]["track"],
                "metric": rows[0]["metric"],
                "value": value,
                "fold_sample_sd": fold_sd,
                "B0": baseline,
                "minimum_admissible": minimum,
                "preferred_target": minimum,
                "numeric_acceptance": decision,
                "terminal": terminal,
                "fold_values": [row["seed_mean"] for row in sorted(rows, key=lambda x: x["outer_fold"])],
            }
        )
    for cell in candidate["cells"]:
        if cell["campaign_disposition"] == "unsupported":
            cell_rows.append(
                {
                    "method_id": cell["method_id"],
                    "task": cell["task_id"],
                    "track": cell["track"],
                    "metric": cell["metric_target"],
                    "value": None,
                    "fold_sample_sd": None,
                    "B0": None,
                    "minimum_admissible": None,
                    "preferred_target": None,
                    "numeric_acceptance": "UNSUPPORTED",
                    "terminal": "UNSUPPORTED",
                    "fold_values": [],
                }
            )
    cell_rows.sort(key=lambda row: (str(row["method_id"]), str(row["task"])))
    if len(cell_rows) != 42:
        raise CampaignError(f"aggregate cell coverage drifted: {len(cell_rows)}")

    output = {
        "schema": "joint_protected_campaign_aggregate_v1",
        "campaign_id": candidate["campaign_id"],
        "candidate_sha256": candidate_sha256,
        "authorization_sha256": authorization_sha256,
        "unblind_sha256": sha256_file(args.unblind.resolve()),
        "audit_log_sha256": sha256_file(audit_log),
        "aggregation": candidate["metric_targets"]["aggregation"],
        "seed_rows": seed_rows,
        "fold_rows": fold_rows,
        "oof_companion_rows": oof_rows,
        "job_traceability": job_traceability,
        "cells": cell_rows,
        "sta_net": candidate["sta_net"],
        "created_at": utc_now(),
    }
    output["traceability_sha256"] = stable_hash(
        {
            "candidate_sha256": candidate_sha256,
            "jobs": job_traceability,
            "cells": cell_rows,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output_dir / "aggregate.json", output)
    for filename, schema, rows in (
        ("seed_rows.json", "joint_protected_seed_rows_v1", seed_rows),
        ("fold_rows.json", "joint_protected_fold_rows_v1", fold_rows),
        ("oof_companion_rows.json", "joint_protected_oof_companion_rows_v1", oof_rows),
        ("job_traceability.json", "joint_protected_job_traceability_v1", job_traceability),
    ):
        write_json_atomic(
            args.output_dir / filename,
            {"schema": schema, "campaign_id": candidate["campaign_id"], "rows": rows},
        )
    write_json_atomic(
        args.output_dir / "sta_net_context_reference.json",
        {
            "schema": "joint_protected_sta_net_context_reference_v1",
            "campaign_id": candidate["campaign_id"],
            "terminal": "CONTEXT_REFERENCE",
            **candidate["sta_net"],
        },
    )
    with (args.output_dir / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "method_id", "task", "track", "metric", "value", "fold_sample_sd",
            "B0", "minimum_admissible", "preferred_target", "numeric_acceptance", "terminal",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cell_rows)
    for filename, disposition in (
        ("direct_cells.csv", "direct"),
        ("overlap_cells.csv", "overlap"),
        ("unsupported_cells.csv", "unsupported"),
    ):
        selected = [
            row
            for row in cell_rows
            if (
                (disposition == "unsupported" and row["terminal"] == "UNSUPPORTED")
                or (
                    disposition == "overlap"
                    and row["terminal"] == "OVERLAP_TRACK_ONLY"
                )
                or (
                    disposition == "direct"
                    and row["terminal"]
                    not in {"UNSUPPORTED", "OVERLAP_TRACK_ONLY"}
                )
            )
        ]
        with (args.output_dir / filename).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
    lines = [
        "# Joint protected campaign results",
        "",
        "| Method | Task | Track | Metric | Mean | Fold SD | Terminal |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in cell_rows:
        value = "—" if row["value"] is None else f"{row['value']:.4f}"
        sd = "—" if row["fold_sample_sd"] is None else f"{row['fold_sample_sd']:.4f}"
        lines.append(
            f"| {row['method_id']} | {row['task']} | {row['track']} | {row['metric']} | {value} | {sd} | {row['terminal']} |"
        )
    (args.output_dir / "cells.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    latex = ["\\begin{tabular}{llllrrl}", "Method & Task & Track & Metric & Mean & SD & Status \\\\", "\\hline"]
    for row in cell_rows:
        value = "--" if row["value"] is None else f"{row['value']:.4f}"
        sd = "--" if row["fold_sample_sd"] is None else f"{row['fold_sample_sd']:.4f}"
        latex.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["method_id"], row["task"], row["track"], row["metric"], value, sd, row["terminal"]
            )
        )
    latex.append("\\end{tabular}")
    (args.output_dir / "cells.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--unblind", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    output = aggregate(parse_args())
    print(
        json.dumps(
            {
                "status": "AGGREGATED",
                "campaign_id": output["campaign_id"],
                "cell_count": len(output["cells"]),
                "traceability_sha256": output["traceability_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
