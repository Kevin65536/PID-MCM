#!/usr/bin/env python3
"""Benchmark six metric-blind public shadows and freeze deterministic GPU lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
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
    sha256_file,
    stable_hash,
    utc_now,
    verify_candidate_file,
    write_json_atomic,
)


WORKER = REPO_ROOT / "comparative_methods/protected_campaign_worker.py"
METHOD_ORDER = ("biot", "cbramod", "reve", "efrm", "normwear", "brainfusion")
FORBIDDEN_RUNTIME_TOKENS = ("target", "logits", "metric", "confusion", "sample_id")


def _gpus() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version,"
            "temperature.gpu,utilization.gpu,ecc.mode.current",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        values = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(values[0]),
                "uuid": values[1],
                "name": values[2],
                "memory_total_mib": int(values[3]),
                "memory_free_mib": int(values[4]),
                "driver_version": values[5],
                "temperature_c": int(values[6]),
                "utilization_percent": int(values[7]),
                "ecc_mode": values[8],
            }
        )
    return rows


def _equivalent(left: Path, right: Path) -> tuple[bool, float]:
    maximum = 0.0
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if a.files != b.files:
            return False, float("inf")
        for name in a.files:
            if a[name].dtype != b[name].dtype or a[name].shape != b[name].shape:
                return False, float("inf")
            if a[name].dtype.kind in "f":
                difference = float(np.max(np.abs(a[name].astype(np.float64) - b[name].astype(np.float64))))
                maximum = max(maximum, difference)
                if not np.allclose(a[name], b[name], rtol=1e-5, atol=1e-6):
                    return False, maximum
            elif not np.array_equal(a[name], b[name]):
                return False, float("inf")
    return True, maximum


def _shadow_job(candidate: dict[str, Any], method: str) -> dict[str, Any]:
    matches = [
        row
        for row in candidate["jobs"]
        if row["method_slug"] == method
        and row["task"] == "motor_imagery"
        and row["outer_fold"] == 0
        and row["seed"] == 17
    ]
    if len(matches) != 1:
        raise CampaignError(f"could not resolve one shadow job for {method}")
    return matches[0]


def _validate_shadow_directory(
    directory: Path,
    *,
    candidate: dict[str, Any],
    candidate_sha256: str,
    job: dict[str, Any],
    expected_device: str,
    expected_device_uuid: str,
) -> list[dict[str, str]]:
    runtime_files: list[dict[str, str]] = []
    for json_path in sorted(directory.glob("*.json")):
        serialized = json_path.read_text(encoding="utf-8").lower()
        if any(token in serialized for token in FORBIDDEN_RUNTIME_TOKENS):
            raise CampaignError(f"runtime JSON redaction failed for {job['method_slug']}")
        runtime_files.append(
            {"path": portable_path(json_path), "sha256": sha256_file(json_path)}
        )
    if not runtime_files:
        raise CampaignError("shadow runtime JSON evidence is absent")
    manifest = read_json(directory / "job_manifest.json")
    status = read_json(directory / "status.json")
    audit = read_json(directory / "audit_report.json")
    artifacts = artifact_map(job)
    public_split = read_json(REPO_ROOT / str(artifacts["public_split_manifest"]["path"]))
    expected_count = len(public_split.get("validation_indices", []))
    determinism = {
        "seed": int(job["seed"]),
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "float32_matmul_precision": "high",
    }
    identity = {
        "job_id": job["job_id"],
        "method_id": job["method_id"],
        "task": job["task"],
        "outer_fold": job["outer_fold"],
        "seed": job["seed"],
    }
    if any(manifest.get(key) != value for key, value in identity.items()) or (
        manifest.get("candidate_sha256") != candidate_sha256
        or manifest.get("authorization_sha256") != "not_applicable_public_shadow"
        or manifest.get("surface") != "shadow"
        or manifest.get("device") != expected_device
        or manifest.get("device_uuid") != expected_device_uuid
        or manifest.get("input_contract_sha256") != job["input_contract"]["sha256"]
        or manifest.get("frozen_inference_contract_sha256")
        != stable_hash(job["frozen_inference_contract"])
        or manifest.get("artifact_sha256")
        != {
            role: value["sha256"]
            for role, value in sorted(artifacts.items())
        }
        or manifest.get("environment_sha256") != candidate["environment"]["sha256"]
        or manifest.get("determinism_sha256") != stable_hash(determinism)
        or manifest.get("protected_test_opened") is not False
        or manifest.get("performance_computed") is not False
        or status.get("job_id") != job["job_id"]
        or status.get("candidate_sha256") != candidate_sha256
        or status.get("status") != "COMPLETED"
        or status.get("surface") != "shadow"
        or status.get("device_uuid") != expected_device_uuid
        or status.get("protected_test_opened") is not False
        or status.get("performance_computed") is not False
        or audit.get("job_id") != job["job_id"]
        or audit.get("status") != "pass"
        or int(audit.get("sample_count", -1)) != expected_count
        or int(audit.get("unique_identity_count", -1)) != expected_count
        or audit.get("protected_test_opened") is not False
        or audit.get("performance_computed") is not False
    ):
        raise CampaignError(f"shadow evidence identity differs for {job['method_slug']}")
    return runtime_files


def _cpu_repeatability(
    *,
    candidate: dict[str, Any],
    candidate_sha256: str,
    first_root: Path,
    second_root: Path,
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for method in METHOD_ORDER:
        job = _shadow_job(candidate, method)
        directories = (first_root / method, second_root / method)
        predictions = tuple(path / "shadow_predictions.npz" for path in directories)
        runtime_json_files: list[dict[str, str]] = []
        if any(not path.is_file() for path in predictions):
            raise CampaignError(f"missing two-pass CPU shadow prediction for {method}")
        for directory in directories:
            runtime_json_files.extend(
                _validate_shadow_directory(
                    directory,
                    candidate=candidate,
                    candidate_sha256=candidate_sha256,
                    job=job,
                    expected_device="cpu",
                    expected_device_uuid="CPU",
                )
            )
        with np.load(predictions[0], allow_pickle=False) as left, np.load(
            predictions[1], allow_pickle=False
        ) as right:
            if left.files != right.files:
                raise CampaignError(f"CPU shadow fields differ for {method}")
            array_contract = []
            for name in left.files:
                if (
                    left[name].dtype != right[name].dtype
                    or left[name].shape != right[name].shape
                    or not np.array_equal(left[name], right[name])
                ):
                    raise CampaignError(f"CPU shadow is not bitwise repeatable for {method}")
                array_contract.append(
                    {
                        "name": name,
                        "dtype": str(left[name].dtype),
                        "shape": list(left[name].shape),
                    }
                )
        evidence[method] = {
            "bitwise_equal": True,
            "runtime_json_redaction_pass": True,
            "candidate_sha256_before_lane_freeze": candidate_sha256,
            "first_prediction_path": portable_path(predictions[0]),
            "first_prediction_sha256": sha256_file(predictions[0]),
            "second_prediction_path": portable_path(predictions[1]),
            "second_prediction_sha256": sha256_file(predictions[1]),
            "array_contract_sha256": stable_hash(array_contract),
            "runtime_json_files": runtime_json_files,
        }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "comparative_methods/evidence/protected_campaign/lane_manifest_v1.json",
    )
    parser.add_argument(
        "--cpu-shadow-first",
        type=Path,
        default=REPO_ROOT
        / "comparative_methods/evidence/protected_campaign/shadow_cpu_pass_v1",
    )
    parser.add_argument(
        "--cpu-shadow-second",
        type=Path,
        default=REPO_ROOT
        / "comparative_methods/evidence/protected_campaign/shadow_cpu_pass_v1_repeat",
    )
    parser.add_argument(
        "--shadow-root",
        type=Path,
        default=REPO_ROOT / "comparative_methods/evidence/protected_campaign/shadow_gpu_v1",
    )
    parser.add_argument(
        "--single-gpu-uuid",
        help="freeze all methods to one exact healthy GPU UUID",
    )
    parser.add_argument("--single-gpu-protocol-owner")
    parser.add_argument("--single-gpu-run-owner")
    args = parser.parse_args()
    candidate, candidate_sha256 = verify_candidate_file(
        args.candidate.resolve(), verify_artifacts=False
    )
    cpu_repeatability = _cpu_repeatability(
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        first_root=args.cpu_shadow_first.resolve(),
        second_root=args.cpu_shadow_second.resolve(),
    )
    gpus = _gpus()
    healthy_idle = [
        row
        for row in gpus
        if row["temperature_c"] < 85
        and row["utilization_percent"] < 20
        and row["memory_free_mib"] >= 6 * 1024
    ]
    if args.single_gpu_uuid:
        owners = {
            str(args.single_gpu_protocol_owner or "").strip(),
            str(args.single_gpu_run_owner or "").strip(),
        }
        matches = [row for row in healthy_idle if row["uuid"] == args.single_gpu_uuid]
        if len(owners) != 2 or "" in owners:
            raise CampaignError("single-GPU override requires two distinct named owners")
        if len(matches) != 1:
            raise CampaignError("requested single frozen GPU is not healthy and idle")
        selected_gpus = matches
        execution_policy = "single_frozen_gpu_user_override"
    else:
        if len(healthy_idle) < 2:
            raise CampaignError(
                f"shadow benchmark requires two healthy idle GPUs, observed {len(healthy_idle)}"
            )
        selected_gpus = sorted(healthy_idle, key=lambda row: row["uuid"])[:2]
        execution_policy = "dual_gpu_equivalence"
    jobs = index_jobs(candidate)
    benchmark_jobs = {
        method: _shadow_job(candidate, method) for method in METHOD_ORDER
    }

    timings: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    for gpu in selected_gpus:
        for method in METHOD_ORDER:
            job = benchmark_jobs[method]
            output_dir = args.shadow_root.resolve() / gpu["uuid"] / method
            if output_dir.exists():
                raise FileExistsError(f"shadow benchmark output exists: {output_dir}")
            command = [
                str(REPO_ROOT / ".venv/bin/python"),
                str(WORKER),
                "--candidate",
                str(args.candidate.resolve()),
                "--job-id",
                str(job["job_id"]),
                "--surface",
                "shadow",
                "--device",
                f"cuda:{gpu['index']}",
                "--output-dir",
                str(output_dir),
            ]
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                raise CampaignError(f"shadow benchmark failed for {method} on {gpu['uuid']}")
            if any(
                token in completed.stdout.lower()
                for token in FORBIDDEN_RUNTIME_TOKENS
            ):
                raise CampaignError(f"worker stdout redaction failed for {method}")
            prediction = output_dir / "shadow_predictions.npz"
            status = read_json(output_dir / "status.json")
            runtime_json_files = _validate_shadow_directory(
                output_dir,
                candidate=candidate,
                candidate_sha256=candidate_sha256,
                job=job,
                expected_device=f"cuda:{gpu['index']}",
                expected_device_uuid=str(gpu["uuid"]),
            )
            timings[method].append(
                {
                    "candidate_sha256_before_lane_freeze": candidate_sha256,
                    "job_id": job["job_id"],
                    "gpu_uuid": gpu["uuid"],
                    "gpu_index": gpu["index"],
                    "wall_seconds": float(status["wall_seconds"]),
                    "controller_wall_seconds": elapsed,
                    "cuda_peak_allocated_bytes": int(
                        status["cuda_peak_allocated_bytes"]
                    ),
                    "cuda_peak_reserved_bytes": int(
                        status["cuda_peak_reserved_bytes"]
                    ),
                    "prediction_path": portable_path(prediction),
                    "prediction_sha256": sha256_file(prediction),
                    "runtime_json_redaction_pass": True,
                    "runtime_json_files": runtime_json_files,
                }
            )

    equivalence = {}
    for method, rows in timings.items():
        if execution_policy == "single_frozen_gpu_user_override":
            equivalent, maximum = True, 0.0
            comparison_mode = "single_frozen_gpu_self_consistency"
        else:
            equivalent, maximum = _equivalent(
                REPO_ROOT / rows[0]["prediction_path"],
                REPO_ROOT / rows[1]["prediction_path"],
            )
            comparison_mode = "cross_gpu_numeric_equivalence"
        equivalence[method] = {
            "equivalent": equivalent,
            "maximum_absolute_difference": maximum,
            "comparison_mode": comparison_mode,
        }
        if not equivalent:
            raise CampaignError(f"GPU shadow equivalence failed for {method}")

    job_counts = {method: 0 for method in METHOD_ORDER}
    for row in candidate["jobs"]:
        job_counts[str(row["method_slug"])] += 1
    method_benchmarks = {
        method: {
            "median_wall_seconds": float(
                np.median([row["wall_seconds"] for row in timings[method]])
            ),
            "peak_cuda_allocated_bytes": max(
                row["cuda_peak_allocated_bytes"] for row in timings[method]
            ),
            "peak_cuda_reserved_bytes": max(
                row["cuda_peak_reserved_bytes"] for row in timings[method]
            ),
        }
        for method in METHOD_ORDER
    }
    estimates = {
        method: method_benchmarks[method]["median_wall_seconds"] * job_counts[method]
        for method in METHOD_ORDER
    }
    loads = {gpu["uuid"]: 0.0 for gpu in selected_gpus}
    assignments = []
    gpu_by_uuid = {gpu["uuid"]: gpu for gpu in selected_gpus}
    for method in sorted(METHOD_ORDER, key=lambda name: (-estimates[name], name)):
        uuid = min(loads, key=lambda value: (loads[value], value))
        gpu = gpu_by_uuid[uuid]
        assignments.append(
            {
                "method_slug": method,
                "gpu_index": gpu["index"],
                "gpu_uuid": uuid,
                "estimated_total_seconds": estimates[method],
                "job_count": job_counts[method],
            }
        )
        loads[uuid] += estimates[method]
    assignments.sort(key=lambda row: row["method_slug"])
    manifest = {
        "schema": "joint_protected_campaign_lane_manifest_v1",
        "campaign_id": candidate["campaign_id"],
        "candidate_sha256_before_lane_freeze": candidate_sha256,
        "status": "pass",
        "execution_policy": execution_policy,
        "minimum_healthy_idle_gpus": len(selected_gpus),
        "gpu_snapshot": selected_gpus,
        "torch_cuda": torch.version.cuda,
        "benchmarks": timings,
        "method_benchmarks": method_benchmarks,
        "gpu_equivalence": equivalence,
        "cpu_repeatability": cpu_repeatability,
        "assignment_algorithm": (
            "all_methods_to_exact_user_authorized_single_gpu"
            if execution_policy == "single_frozen_gpu_user_override"
            else "descending_estimated_total_to_lowest_cumulative_lane_then_method_slug_gpu_uuid"
        ),
        "assignments": assignments,
        "estimated_lane_seconds": loads,
        "backup_gpu_uuids": (
            []
            if execution_policy == "single_frozen_gpu_user_override"
            else [gpu["uuid"] for gpu in selected_gpus]
        ),
        "protected_test_opened": False,
        "created_at": utc_now(),
    }
    if execution_policy == "single_frozen_gpu_user_override":
        manifest["single_gpu_policy_authorization"] = {
            "protocol_owner": str(args.single_gpu_protocol_owner).strip(),
            "run_owner": str(args.single_gpu_run_owner).strip(),
            "all_540_jobs_fixed_to_one_gpu": True,
            "automatic_gpu_migration_forbidden": True,
        }
    manifest["assignment_sha256"] = stable_hash(assignments)
    write_json_atomic(args.output.resolve(), manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": portable_path(args.output.resolve()),
                "assignments": assignments,
                "protected_test_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
