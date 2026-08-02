#!/usr/bin/env python3
"""Build the reviewed, BrainFusion-only public-development v2 job matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import METHOD_ID
from run_public_development_v2 import (
    load_runner_config,
    portable_path,
    resolve_repo_path,
    sha256_file,
    stable_hash,
    write_json,
)


MATRIX_SCHEMA = "brainfusion_public_job_matrix_candidate_v2"
DEFAULT_PROTOCOL = METHOD_ROOT / "configs/public_job_matrix_candidate_v2.yaml"


def load_protocol(path: str | Path) -> tuple[dict[str, Any], Path]:
    protocol_path = resolve_repo_path(path)
    value = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != MATRIX_SCHEMA:
        raise ValueError(f"expected {MATRIX_SCHEMA}: {protocol_path}")
    if value.get("method_id") != METHOD_ID or value.get("mode") != "public_development_only":
        raise PermissionError("job matrix must remain BrainFusion public development only")
    if value.get("protected_test_default") != "locked":
        raise PermissionError("protected test must remain locked")
    authorization = value.get("authorization", {})
    if authorization.get("protected_evaluation_authorized") is not False:
        raise PermissionError("candidate matrix may not authorize protected evaluation")
    if authorization.get("public_matrix_launch_authorized") is not False:
        raise PermissionError("candidate matrix may not launch itself")
    if int(value["matrix"]["max_concurrent_jobs"]) != 1:
        raise ValueError("BrainFusion public jobs must remain serial")
    if int(value["failure_policy"]["automatic_retry_count"]) != 0:
        raise ValueError("BrainFusion candidate matrix may not retry automatically")

    pilot_path = resolve_repo_path(value["pilot_evidence"]["path"])
    if sha256_file(pilot_path) != str(value["pilot_evidence"]["sha256"]):
        raise RuntimeError("BrainFusion full-fold pilot evidence fingerprint drifted")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != value["pilot_evidence"]["required_status"]:
        raise RuntimeError("BrainFusion full-fold pilot did not pass")
    if pilot.get("mode") != value["pilot_evidence"]["required_mode"]:
        raise RuntimeError("BrainFusion candidate is not bound to a full-fold pilot")
    if pilot.get("cached_validation_matches_raw_adapter") is not True:
        raise RuntimeError("BrainFusion full-fold pilot did not verify its tensor cache")
    if pilot.get("protected_test_opened") is not False:
        raise PermissionError("BrainFusion pilot reports protected access")

    runner_path = resolve_repo_path(value["runner"]["path"])
    if not runner_path.is_file():
        raise FileNotFoundError(f"missing BrainFusion runner: {runner_path}")
    config_path = resolve_repo_path(value["runner"]["config"])
    if sha256_file(config_path) != str(value["runner"]["config_sha256"]):
        raise RuntimeError("BrainFusion runner config fingerprint drifted")
    config, _resolved, _alignment, _alignment_path = load_runner_config(config_path)
    matrix = value["matrix"]
    if list(matrix["task_order"]) != list(config["job_matrix"]["tasks"]):
        raise ValueError("matrix task order differs from the reviewed runner config")
    if list(matrix["outer_fold_order"]) != list(config["job_matrix"]["outer_folds"]):
        raise ValueError("matrix fold order differs from the reviewed runner config")
    if list(matrix["seed_order"]) != list(config["job_matrix"]["seeds"]):
        raise ValueError("matrix seed order differs from the reviewed runner config")
    expected = (
        len(matrix["task_order"])
        * len(matrix["outer_fold_order"])
        * len(matrix["seed_order"])
    )
    if expected != 75 or expected != int(matrix["expected_job_count"]):
        raise ValueError("BrainFusion candidate matrix must contain exactly 75 jobs")
    return value, protocol_path


def build_matrix(protocol_path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol, resolved_protocol = load_protocol(protocol_path)
    matrix = protocol["matrix"]
    runner_path = resolve_repo_path(protocol["runner"]["path"])
    config_path = resolve_repo_path(protocol["runner"]["config"])
    output_root = resolve_repo_path(matrix["output_root"])
    jobs: list[dict[str, Any]] = []
    order = 0
    for task in matrix["task_order"]:
        for outer_fold in matrix["outer_fold_order"]:
            for seed in matrix["seed_order"]:
                output_dir = output_root / str(task) / f"outer{outer_fold}" / f"seed{seed}"
                command = [
                    ".venv/bin/python",
                    portable_path(runner_path),
                    "--config",
                    portable_path(config_path),
                    "--task",
                    str(task),
                    "--outer-fold",
                    str(outer_fold),
                    "--seed",
                    str(seed),
                    "--device",
                    "cuda:1",
                    "--output-dir",
                    portable_path(output_dir),
                ]
                if "protected" in " ".join(command).lower():
                    raise PermissionError("BrainFusion public job crossed protected boundary")
                jobs.append(
                    {
                        "order": order,
                        "job_id": f"brainfusion__{task}__outer{outer_fold}__seed{seed}",
                        "task": str(task),
                        "outer_fold": int(outer_fold),
                        "seed": int(seed),
                        "output_dir": portable_path(output_dir),
                        "command": command,
                        "initial_status": "queued_not_authorized",
                    }
                )
                order += 1
    if len(jobs) != 75:
        raise RuntimeError("generated BrainFusion public matrix has the wrong job count")
    identity: Mapping[str, Any] = {
        "protocol_sha256": sha256_file(resolved_protocol),
        "pilot_evidence_sha256": protocol["pilot_evidence"]["sha256"],
        "runner_sha256": sha256_file(runner_path),
        "runner_config_sha256": sha256_file(config_path),
        "jobs": jobs,
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
    }
    return {
        "schema": "brainfusion_public_job_matrix_v2",
        "status": "candidate_not_launch_authorized",
        "method_id": METHOD_ID,
        "protocol_path": portable_path(resolved_protocol),
        "protocol_sha256": sha256_file(resolved_protocol),
        "pilot_evidence_sha256": protocol["pilot_evidence"]["sha256"],
        "runner_path": portable_path(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "runner_config_path": portable_path(config_path),
        "runner_config_sha256": sha256_file(config_path),
        "job_count": len(jobs),
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
        "jobs": jobs,
        "matrix_identity_sha256": stable_hash(identity),
        "public_matrix_launch_authorized": False,
        "protected_evaluation_authorized": False,
        "protected_test_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    matrix = build_matrix(args.protocol)
    if args.output is not None:
        write_json(resolve_repo_path(args.output), matrix)
    print(json.dumps(matrix, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
