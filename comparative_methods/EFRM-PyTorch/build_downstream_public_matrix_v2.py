#!/usr/bin/env python3
"""Build the non-self-authorizing EFRM LODO-v2 public downstream matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from run_downstream_public_v2 import (  # noqa: E402
    DEFAULT_CONFIG,
    METHOD_ID,
    load_config,
    portable_path,
    resolve_repo_path,
    sha256_file,
    stable_hash,
    write_json,
)


MATRIX_SCHEMA = "efrm_lodo_downstream_public_job_matrix_v2"


def build_matrix(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, resolved_config = load_config(config_path)
    runner = METHOD_ROOT / "run_downstream_public_v2.py"
    output_root = resolve_repo_path(config["resources"]["run_root"])
    jobs: list[dict[str, Any]] = []
    order = 0
    for task in config["job_matrix"]["tasks"]:
        for outer_fold in config["job_matrix"]["outer_folds"]:
            for seed in config["job_matrix"]["seeds"]:
                output_dir = output_root / task / f"outer{outer_fold}" / f"seed{seed}"
                command = [
                    ".venv/bin/python",
                    portable_path(runner),
                    "--config",
                    portable_path(resolved_config),
                    "--task",
                    str(task),
                    "--outer-fold",
                    str(outer_fold),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    portable_path(output_dir),
                ]
                if "protected" in " ".join(command).lower():
                    raise PermissionError("EFRM public matrix crossed the protected boundary")
                jobs.append(
                    {
                        "order": order,
                        "job_id": f"efrm__{task}__outer{outer_fold}__seed{seed}",
                        "task": task,
                        "outer_fold": int(outer_fold),
                        "seed": int(seed),
                        "output_dir": portable_path(output_dir),
                        "command": command,
                        "initial_status": "queued_not_authorized",
                    }
                )
                order += 1
    if len(jobs) != int(config["job_matrix"]["expected_public_jobs"]):
        raise RuntimeError("generated EFRM v2 public matrix has the wrong job count")
    identity = {
        "config_sha256": sha256_file(resolved_config),
        "runner_sha256": sha256_file(runner),
        "jobs": jobs,
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
    }
    return {
        "schema": MATRIX_SCHEMA,
        "status": "candidate_not_launch_authorized",
        "protocol_id": config["protocol_id"],
        "method_id": METHOD_ID,
        "runner_path": portable_path(runner),
        "runner_sha256": sha256_file(runner),
        "config_path": portable_path(resolved_config),
        "config_sha256": sha256_file(resolved_config),
        "job_count": len(jobs),
        "max_concurrent_jobs": 1,
        "automatic_retry_count": 0,
        "jobs": jobs,
        "matrix_identity_sha256": stable_hash(identity),
        "public_matrix_launch_authorized": False,
        "protected_evaluation_authorized": False,
        "target_dataset_exposure": False,
        "protected_test_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    matrix = build_matrix(args.config)
    if args.output is not None:
        write_json(resolve_repo_path(args.output), matrix)
    print(json.dumps(matrix, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
