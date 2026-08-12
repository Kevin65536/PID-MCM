#!/usr/bin/env python3
"""Build the serial GPU1 job matrix for public EEG performance development."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from comparative_methods.audit_public_preflight import (
    public_json,
    registry_manifest,
    strict_public_entry,
)
from comparative_methods.single_modal_eeg.contract import (
    SUPPORTED_METHODS,
    load_config,
    resolve_repo_path,
    stable_hash,
)
from comparative_methods.single_modal_eeg.run_public_performance import write_json


SCHEMA = "single_modal_eeg_public_job_matrix_v1"


def build_matrix(
    *,
    config_path: str | Path,
    output_root: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    config, resolved_config = load_config(config_path)
    root = resolve_repo_path(output_root)
    if "protected" in {part.lower() for part in root.parts}:
        raise PermissionError(f"refusing protected run root: {root}")
    registry_path = resolve_repo_path(config["registry"]["manifest"])
    registry = registry_manifest(registry_path)
    configured_device = str(config["resources"]["default_device"])
    selected_device = str(device or configured_device)
    if selected_device != configured_device:
        raise ValueError(
            f"v1 resource contract freezes device={configured_device}, got {selected_device}"
        )
    if not selected_device.startswith("cuda:"):
        raise ValueError("public performance matrix requires a CUDA device")
    seeds = [int(value) for value in config["probe"]["seed_set"]]
    supported_tasks = [
        task for task, task_config in config["tasks"].items() if task_config["supported"]
    ]
    unsupported = {
        task: str(task_config.get("reason", "unspecified"))
        for task, task_config in config["tasks"].items()
        if not task_config["supported"]
    }
    jobs: list[dict[str, Any]] = []
    for method in SUPPORTED_METHODS:
        for task in supported_tasks:
            for outer_fold in range(5):
                entry = strict_public_entry(
                    registry, task=task, outer_fold=outer_fold
                )
                public_path = Path(str(entry["public_path"])).resolve()
                public_manifest = public_json(public_path)
                for seed in seeds:
                    job_id = f"{method}__{task}__outer{outer_fold}__seed{seed}"
                    output_dir = root / method / task / f"outer{outer_fold}" / f"seed{seed}"
                    command = [
                        ".venv/bin/python",
                        "-m",
                        "comparative_methods.single_modal_eeg.run_public_performance",
                        "--config",
                        str(resolved_config),
                        "--method",
                        method,
                        "--task",
                        task,
                        "--outer-fold",
                        str(outer_fold),
                        "--seed",
                        str(seed),
                        "--device",
                        selected_device,
                        "--output-dir",
                        str(output_dir),
                    ]
                    jobs.append(
                        {
                            "job_id": job_id,
                            "method": method,
                            "task": task,
                            "outer_fold": outer_fold,
                            "seed": seed,
                            "device": selected_device,
                            "public_manifest_path": str(public_path),
                            "public_manifest_sha256": str(entry["public_sha256"]),
                            "public_split_sha256": str(public_manifest["split_sha256"]),
                            "train_sample_count": int(entry["train_sample_count"]),
                            "validation_sample_count": int(
                                entry["validation_sample_count"]
                            ),
                            "output_dir": str(output_dir),
                            "command": command,
                            "status": "pending",
                            "protected_test_opened": False,
                        }
                    )
    expected = len(SUPPORTED_METHODS) * len(supported_tasks) * 5 * len(seeds)
    if len(jobs) != expected or len({job["job_id"] for job in jobs}) != expected:
        raise RuntimeError("public performance job matrix is incomplete or duplicated")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": config["protocol_id"],
        "mode": "public_development_only",
        "config_path": str(resolved_config),
        "registry_sha256": registry["registry_sha256"],
        "methods": list(SUPPORTED_METHODS),
        "supported_tasks": supported_tasks,
        "unsupported_tasks": unsupported,
        "outer_folds": 5,
        "seeds": seeds,
        "job_count": len(jobs),
        "scheduling": "single_serial_comparison_job_on_gpu1",
        "jobs": jobs,
        "protected_test_opened": False,
    }
    payload["matrix_sha256"] = stable_hash(payload)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_matrix(
        config_path=args.config,
        output_root=args.output_root,
        device=args.device,
    )
    destination = resolve_repo_path(args.output)
    write_json(destination, payload)
    print(
        json.dumps(
            {
                "status": "ready",
                "job_count": payload["job_count"],
                "matrix_sha256": payload["matrix_sha256"],
                "protected_test_opened": False,
                "output": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
