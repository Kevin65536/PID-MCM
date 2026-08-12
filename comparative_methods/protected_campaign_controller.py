#!/usr/bin/env python3
"""Fail-closed controller for preflight, execution, and status inspection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.protected_campaign_common import (  # noqa: E402
    CampaignError,
    append_jsonl,
    artifact_map,
    index_jobs,
    portable_path,
    read_json,
    sha256_file,
    stable_hash,
    utc_now,
    verify_authorization,
    verify_candidate_file,
    verify_runtime_environment,
    write_json_atomic,
)


WORKER = REPO_ROOT / "comparative_methods/protected_campaign_worker.py"
DEFAULT_ROOT = REPO_ROOT / "comparative_methods/runs/protected_campaign"


def _gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version,"
        "temperature.gpu,utilization.gpu,ecc.mode.current",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 9:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "driver_version": fields[5],
                "temperature_c": int(fields[6]),
                "utilization_percent": int(fields[7]),
                "ecc_mode": fields[8],
            }
        )
    return rows


def _required_output_bytes(candidate: Mapping[str, Any]) -> int:
    # Conservative prediction payload estimate: identities/targets/logits plus
    # JSON audit overhead.  It is intentionally independent of performance.
    total = 0
    for job in candidate["jobs"]:
        count = int(job["input_contract"]["protected_sample_count"])
        if job["task"] == "refed_regression":
            total += count * 2 * 150 * 4 * 3
        else:
            total += count * 256
        total += 64 * 1024
    return total


def campaign_status(
    candidate: Mapping[str, Any],
    root: Path,
    *,
    candidate_sha256: str | None = None,
) -> dict[str, Any]:
    completed = 0
    technical_failures = 0
    invalid_failures = 0
    missing = 0
    protected_opened = False
    attempts: dict[str, int] = {}
    for row in candidate["jobs"]:
        job_id = str(row["job_id"])
        directory = root / job_id
        status_path = directory / "status.json"
        if not status_path.is_file():
            quarantined = sorted((root / "quarantine").glob(f"{job_id}.attempt*"))
            if quarantined:
                retained = [
                    read_json(path / "status.json")
                    for path in quarantined
                    if (path / "status.json").is_file()
                ]
                latest = max(retained, key=lambda row: int(row.get("attempt", 0)), default={})
                attempts[job_id] = int(latest.get("attempt", 0))
                protected_opened = protected_opened or bool(
                    latest.get("protected_test_opened", False)
                )
                if latest.get("failure_code") == "FAILED_INVALID_OUTPUT":
                    invalid_failures += 1
                else:
                    technical_failures += 1
                continue
            orphaned = sorted(root.glob(f".{job_id}.attempt*.tmp"))
            if orphaned:
                latest_path = orphaned[-1] / "status.json"
                latest = read_json(latest_path) if latest_path.is_file() else {}
                attempts[job_id] = int(latest.get("attempt", 1))
                protected_opened = protected_opened or bool(
                    latest.get("protected_test_opened", False)
                )
                technical_failures += 1
                continue
            missing += 1
            continue
        status = read_json(status_path)
        attempts[job_id] = int(status.get("attempt", 0))
        manifest_path = directory / "job_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        binding_ok = (
            candidate_sha256 is None
            or (
                status.get("candidate_sha256") == candidate_sha256
                and manifest.get("candidate_sha256") == candidate_sha256
                and manifest.get("job_id") == job_id
            )
        )
        if status.get("status") == "COMPLETED" and binding_ok:
            completed += 1
        else:
            if status.get("failure_code") == "FAILED_INVALID_OUTPUT":
                invalid_failures += 1
            else:
                technical_failures += 1
        protected_opened = protected_opened or bool(
            status.get("protected_test_opened", False)
        )
    failures = technical_failures + invalid_failures
    state = (
        "SEALED_COMPLETE"
        if completed == len(candidate["jobs"]) and failures == 0 and missing == 0
        else "INCOMPLETE_TECHNICAL"
        if failures
        else "RUNNING"
        if completed
        else "AUTHORIZED"
    )
    return {
        "schema": "joint_protected_campaign_status_v1",
        "campaign_id": candidate["campaign_id"],
        "state": state,
        "expected_job_count": len(candidate["jobs"]),
        "completed_job_count": completed,
        "failed_job_count": failures,
        "technical_failure_job_count": technical_failures,
        "invalid_output_job_count": invalid_failures,
        "missing_job_count": missing,
        "protected_test_opened": protected_opened,
        "updated_at": utc_now(),
    }


def preflight(args: argparse.Namespace, *, require_authorization: bool) -> dict[str, Any]:
    reasons: list[str] = []
    candidate, candidate_sha256 = verify_candidate_file(
        args.candidate.resolve(), verify_artifacts=True
    )
    verify_runtime_environment(candidate)
    if candidate.get("state") != "REVIEWED":
        reasons.append(f"candidate_state={candidate.get('state')}")
    if candidate.get("orr_decision") != "PENDING_DUAL_GO":
        reasons.append(f"candidate_orr_decision={candidate.get('orr_decision')}")
    lane_wrapper = candidate.get("lane_manifest")
    if not lane_wrapper or lane_wrapper.get("value", {}).get("status") != "pass":
        reasons.append("frozen_lane_manifest_missing_or_not_pass")
    gpus = _gpu_snapshot()
    if len(gpus) < 2:
        reasons.append("fewer_than_two_visible_gpus")
    healthy_idle_gpus = [
        row
        for row in gpus
        if row["temperature_c"] < 85
        and row["utilization_percent"] < 20
        and row["memory_free_mib"] >= 6 * 1024
    ]
    if len(healthy_idle_gpus) < 2:
        reasons.append("fewer_than_two_healthy_idle_gpus")
    lane_assignments = (lane_wrapper or {}).get("value", {}).get("assignments", [])
    available = {row["uuid"]: row for row in gpus}
    frozen_gpus = {
        str(row["uuid"]): row
        for row in (lane_wrapper or {}).get("value", {}).get("gpu_snapshot", [])
    }
    for assignment in lane_assignments:
        gpu = available.get(assignment.get("gpu_uuid"))
        if gpu is None:
            reasons.append(f"assigned_gpu_missing:{assignment.get('gpu_uuid')}")
            continue
        frozen = frozen_gpus.get(str(gpu["uuid"]), {})
        if int(assignment.get("gpu_index", -1)) != int(gpu["index"]):
            reasons.append(f"assigned_gpu_index_drift:{gpu['uuid']}")
        for field in ("name", "memory_total_mib", "driver_version", "ecc_mode"):
            if frozen.get(field) != gpu.get(field):
                reasons.append(f"assigned_gpu_{field}_drift:{gpu['uuid']}")
        if (
            gpu["temperature_c"] >= 85
            or gpu["utilization_percent"] >= 20
            or gpu["memory_free_mib"] < 6 * 1024
        ):
            reasons.append(f"assigned_gpu_not_healthy_idle:{gpu['uuid']}")

    output_root = args.output_root.resolve()
    usage = shutil.disk_usage(output_root.parent if output_root.parent.exists() else REPO_ROOT)
    estimated = _required_output_bytes(candidate)
    if usage.free < 2 * estimated:
        reasons.append("free_storage_below_twice_estimated_output")

    authorization_sha256 = None
    if require_authorization or args.authorization is not None:
        if args.authorization is None or not args.authorization.is_file():
            reasons.append("authorization_missing")
        else:
            try:
                _authorization, authorization_sha256 = verify_authorization(
                    args.authorization.resolve(),
                    candidate=candidate,
                    candidate_sha256=candidate_sha256,
                )
            except CampaignError:
                reasons.append("authorization_invalid")
    report = {
        "schema": "joint_protected_campaign_preflight_v1",
        "campaign_id": candidate["campaign_id"],
        "status": "GO" if not reasons else "NO_GO",
        "candidate_sha256": candidate_sha256,
        "authorization_sha256": authorization_sha256,
        "cell_count": len(candidate["cells"]),
        "job_count": len(candidate["jobs"]),
        "gpu_snapshot": gpus,
        "healthy_idle_gpu_count": len(healthy_idle_gpus),
        "estimated_output_bytes": estimated,
        "free_storage_bytes": usage.free,
        "protected_test_opened": False,
        "reasons": reasons,
        "checked_at": utc_now(),
    }
    if args.report is not None:
        write_json_atomic(args.report.resolve(), report)
    return report


def _completed_output_matches(
    directory: Path,
    *,
    job: Mapping[str, Any],
    candidate_sha256: str,
    authorization_sha256: str,
) -> bool:
    required = {
        "status.json",
        "job_manifest.json",
        "protected_predictions.npz",
        "audit_report.json",
        "artifact_checksums.json",
    }
    if not all((directory / name).is_file() for name in required):
        return False
    status = read_json(directory / "status.json")
    manifest = read_json(directory / "job_manifest.json")
    checksums = read_json(directory / "artifact_checksums.json")
    checksum_files = {
        "status.json",
        "job_manifest.json",
        "protected_predictions.npz",
        "audit_report.json",
    }
    if set(checksums.get("files", {})) != checksum_files or any(
        sha256_file(directory / name) != expected
        for name, expected in checksums.get("files", {}).items()
    ):
        return False
    identity = {
        "job_id": job["job_id"],
        "method_id": job["method_id"],
        "task": job["task"],
        "outer_fold": job["outer_fold"],
        "seed": job["seed"],
    }
    return (
        status.get("status") == "COMPLETED"
        and status.get("surface") == "protected"
        and status.get("protected_test_opened") is True
        and status.get("candidate_sha256") == candidate_sha256
        and status.get("authorization_sha256") == authorization_sha256
        and all(manifest.get(key) == value for key, value in identity.items())
        and manifest.get("surface") == "protected"
        and manifest.get("protected_test_opened") is True
        and manifest.get("candidate_sha256") == candidate_sha256
        and manifest.get("authorization_sha256") == authorization_sha256
        and manifest.get("input_contract_sha256") == job["input_contract"]["sha256"]
        and manifest.get("frozen_inference_contract_sha256")
        == stable_hash(job["frozen_inference_contract"])
        and manifest.get("artifact_sha256")
        == {
            role: value["sha256"]
            for role, value in sorted(artifact_map(job).items())
        }
    )


def _recover_or_quarantine_orphans(output_root: Path, job_id: str) -> None:
    for temporary in sorted(output_root.glob(f".{job_id}.attempt*.tmp")):
        if not temporary.is_dir() or temporary.parent != output_root:
            continue
        status_path = temporary / "status.json"
        status = read_json(status_path) if status_path.is_file() else {}
        output_dir = output_root / job_id
        checksums_path = temporary / "artifact_checksums.json"
        complete = status.get("status") == "COMPLETED" and checksums_path.is_file()
        if complete:
            checksums = read_json(checksums_path)
            complete = all(
                (temporary / name).is_file()
                and sha256_file(temporary / name) == expected
                for name, expected in checksums.get("files", {}).items()
            )
        if complete and not output_dir.exists():
            os.replace(temporary, output_dir)
            continue
        attempt = int(status.get("attempt", 1))
        write_json_atomic(
            status_path,
            {
                "schema": "joint_protected_campaign_job_v1",
                "job_id": job_id,
                "attempt": attempt,
                "status": "FAILED_TECHNICAL",
                "surface": "protected",
                "protected_test_opened": bool(
                    status.get("protected_test_opened", False)
                ),
                "candidate_sha256": status.get("candidate_sha256"),
                "authorization_sha256": status.get("authorization_sha256"),
                "failure_code": "FAILED_TECHNICAL",
                "performance_computed": False,
                "completed_at": utc_now(),
            },
        )
        quarantine_root = output_root / "quarantine"
        quarantine_root.mkdir(parents=True, exist_ok=True)
        destination = quarantine_root / (
            f"{job_id}.attempt{attempt}.orphan.{temporary.name.rsplit('.', 1)[0]}"
        )
        os.replace(temporary, destination)


def _run_lane(
    *,
    jobs: list[Mapping[str, Any]],
    gpu_index: int,
    candidate_path: Path,
    authorization_path: Path,
    candidate_sha256: str,
    authorization_sha256: str,
    output_root: Path,
    audit_log: Path,
    stop: threading.Event,
    recover_technical: bool,
) -> list[dict[str, Any]]:
    results = []
    for job in jobs:
        if stop.is_set():
            break
        job_id = str(job["job_id"])
        output_dir = output_root / job_id
        attempt = 1
        _recover_or_quarantine_orphans(output_root, job_id)
        if (output_dir / "status.json").is_file():
            retained = read_json(output_dir / "status.json")
            if retained.get("status") == "COMPLETED":
                if _completed_output_matches(
                    output_dir,
                    job=job,
                    candidate_sha256=candidate_sha256,
                    authorization_sha256=authorization_sha256,
                ):
                    continue
                append_jsonl(
                    audit_log,
                    {
                        "event": "STALE_COMPLETED_OUTPUT_REFUSED",
                        "job_id": job_id,
                        "at": utc_now(),
                    },
                )
                results.append({"job_id": job_id, "returncode": 1})
                stop.set()
                break
            stop.set()
            break
        quarantined = sorted((output_root / "quarantine").glob(f"{job_id}.attempt*"))
        if quarantined:
            retained_failures = [
                read_json(path / "status.json")
                for path in quarantined
                if (path / "status.json").is_file()
            ]
            latest = max(
                retained_failures,
                key=lambda row: int(row.get("attempt", 0)),
                default={},
            )
            previous_attempt = int(latest.get("attempt", 0))
            if (
                recover_technical
                and previous_attempt == 1
                and latest.get("failure_code") == "FAILED_TECHNICAL"
            ):
                attempt = 2
            else:
                stop.set()
                break
        command = [
            str(REPO_ROOT / ".venv/bin/python"),
            str(WORKER),
            "--candidate",
            str(candidate_path),
            "--authorization",
            str(authorization_path),
            "--job-id",
            job_id,
            "--surface",
            "protected",
            "--device",
            f"cuda:{gpu_index}",
            "--attempt",
            str(attempt),
            "--output-dir",
            str(output_dir),
            "--expected-candidate-sha256",
            candidate_sha256,
            "--expected-authorization-sha256",
            authorization_sha256,
            "--audit-log",
            str(audit_log),
        ]
        append_jsonl(
            audit_log,
            {
                "event": "JOB_START",
                "job_id": job_id,
                "attempt": attempt,
                "gpu_index": gpu_index,
                "at": utc_now(),
                "parameters": {
                    "surface": "protected",
                    "attempt": attempt,
                    "gpu_index": gpu_index,
                    "output_dir": portable_path(output_dir),
                    "candidate_sha256": candidate_sha256,
                    "authorization_sha256": authorization_sha256,
                },
                "argv_sha256": __import__("hashlib").sha256(
                    json.dumps(command, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
        )
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        result = {"job_id": job_id, "returncode": completed.returncode}
        results.append(result)
        append_jsonl(
            audit_log,
            {
                "event": "JOB_END",
                "job_id": job_id,
                "attempt": attempt,
                "gpu_index": gpu_index,
                "returncode": completed.returncode,
                "at": utc_now(),
            },
        )
        if completed.returncode != 0:
            retained_failure = any(
                (path / "status.json").is_file()
                for path in (output_root / "quarantine").glob(
                    f"{job_id}.attempt{attempt}*"
                )
            )
            orphaned = any(
                path.is_dir()
                for path in output_root.glob(f".{job_id}.attempt{attempt}*.tmp")
            )
            if not retained_failure and not orphaned:
                failure_code = "FAILED_TECHNICAL"
                try:
                    worker_report = json.loads(completed.stdout.strip().splitlines()[-1])
                    if worker_report.get("status") in {
                        "FAILED_TECHNICAL",
                        "FAILED_INVALID_OUTPUT",
                    }:
                        failure_code = str(worker_report["status"])
                except (IndexError, json.JSONDecodeError):
                    pass
                quarantine = output_root / "quarantine" / (
                    f"{job_id}.attempt{attempt}.controller_retained"
                )
                quarantine.mkdir(parents=True, exist_ok=False)
                write_json_atomic(
                    quarantine / "status.json",
                    {
                        "schema": "joint_protected_campaign_job_v1",
                        "job_id": job_id,
                        "attempt": attempt,
                        "status": failure_code,
                        "surface": "protected",
                        "protected_test_opened": False,
                        "candidate_sha256": candidate_sha256,
                        "authorization_sha256": authorization_sha256,
                        "failure_code": failure_code,
                        "performance_computed": False,
                        "completed_at": utc_now(),
                    },
                )
            stop.set()
            break
    return results


def execute(args: argparse.Namespace) -> dict[str, Any]:
    report = preflight(args, require_authorization=True)
    if report["status"] != "GO":
        raise CampaignError(f"formal execution refused by preflight: {report['reasons']}")
    candidate, candidate_sha256 = verify_candidate_file(
        args.candidate.resolve(), verify_artifacts=True
    )
    _authorization, authorization_sha256 = verify_authorization(
        args.authorization.resolve(),
        candidate=candidate,
        candidate_sha256=candidate_sha256,
    )
    if (
        candidate_sha256 != report["candidate_sha256"]
        or authorization_sha256 != report["authorization_sha256"]
    ):
        raise CampaignError("candidate or authorization changed after preflight")
    output_root = args.output_root.resolve() / str(candidate["campaign_id"])
    output_root.mkdir(parents=True, exist_ok=True)
    audit_log = output_root / "audit/events.jsonl"
    append_jsonl(
        audit_log,
        {
            "event": "CAMPAIGN_EXECUTE",
            "campaign_id": candidate["campaign_id"],
            "candidate_sha256": candidate_sha256,
            "authorization_sha256": authorization_sha256,
            "at": utc_now(),
        },
    )
    assignments = {
        str(row["method_slug"]): row
        for row in candidate["lane_manifest"]["value"]["assignments"]
    }
    lanes: dict[int, list[Mapping[str, Any]]] = {}
    for job in candidate["jobs"]:
        assignment = assignments[str(job["method_slug"])]
        lanes.setdefault(int(assignment["gpu_index"]), []).append(job)
    for rows in lanes.values():
        rows.sort(key=lambda row: str(row["job_id"]))
    stop = threading.Event()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = [
            executor.submit(
                _run_lane,
                jobs=rows,
                gpu_index=index,
                candidate_path=args.candidate.resolve(),
                authorization_path=args.authorization.resolve(),
                candidate_sha256=candidate_sha256,
                authorization_sha256=authorization_sha256,
                output_root=output_root,
                audit_log=audit_log,
                stop=stop,
                recover_technical=bool(args.recover_technical),
            )
            for index, rows in sorted(lanes.items())
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    status = campaign_status(
        candidate, output_root, candidate_sha256=candidate_sha256
    )
    if any(row["returncode"] != 0 for row in results):
        status["state"] = "INCOMPLETE_TECHNICAL"
    write_json_atomic(output_root / "campaign_status.json", status)
    append_jsonl(
        audit_log,
        {
            "event": "CAMPAIGN_STATE",
            "campaign_id": candidate["campaign_id"],
            "state": status["state"],
            "completed_job_count": status["completed_job_count"],
            "failed_job_count": status["failed_job_count"],
            "at": utc_now(),
        },
    )
    return {**status, "worker_invocations": len(results)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "execute", "status"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--recover-technical",
        action="store_true",
        help="permit attempt 2 only for a retained attempt-1 technical failure",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        report = preflight(args, require_authorization=True)
    elif args.command == "execute":
        report = execute(args)
    else:
        candidate, digest = verify_candidate_file(
            args.candidate.resolve(), verify_artifacts=False
        )
        report = campaign_status(
            candidate,
            args.output_root.resolve() / str(candidate["campaign_id"]),
            candidate_sha256=digest,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") in {"GO", None} else 2


if __name__ == "__main__":
    raise SystemExit(main())
