#!/usr/bin/env python3
"""Freeze and launch STA-Net strict-vs-sample-random five-fold evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch import STANetUnifiedTaskDataset, get_sta_net_task_spec
from sta_net_pytorch.splits import build_sample_random_registry, write_registry

TASKS = (
    "motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual",
    "refed_regression",
)
OLD_SELECTION = (
    METHOD_ROOT / "runs" / "tuning" / "20260722_sta_net_hpo_v2_checkpoint_objective_100ep"
    / "final_validation_selection_v2" / "selection_manifest.json"
)
FINAL_SELECTION = (
    METHOD_ROOT / "runs" / "tuning" / "20260724_sta_net_mi_wg_final_targeted_v1_100ep"
    / "final_validation_selection_v2" / "selection_manifest.json"
)
SECONDS_PER_TRAIN_SAMPLE = {
    "motor_imagery": 0.0163,
    "mental_arithmetic": 0.0154,
    "wg": 0.0081,
    "nback": 0.0152,
    "dsr": 0.0034,
    "visual": 0.0030,
    "refed_regression": 0.0984,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_configs() -> dict[str, dict[str, Any]]:
    old = json.loads(OLD_SELECTION.read_text(encoding="utf-8"))["tasks"]
    final = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))["tasks"]
    merged = {**old, **final}
    result = {}
    for task in TASKS:
        selected = merged[task]["selected"]
        config = Path(selected["config"]).resolve()
        if sha256(config) != selected["config_sha256"]:
            raise RuntimeError(f"selected config hash drift for {task}")
        result[task] = {
            "config": str(config),
            "config_sha256": selected["config_sha256"],
            "source_trial": int(selected["trial_number"]),
            "selection_manifest": str(FINAL_SELECTION if task in final else OLD_SELECTION),
        }
    return result


def cached_execution_configs(
    configs: Mapping[str, Mapping[str, Any]],
    tasks: Sequence[str],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Copy selected configs with semantics-preserving input-pipeline overrides."""

    result: dict[str, dict[str, Any]] = {}
    for task in tasks:
        source = Path(str(configs[task]["config"])).resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload.setdefault("training", {})["num_workers"] = 1
        payload["training"]["adapted_sample_cache_size"] = 10_000
        task_override = payload.setdefault("task_overrides", {}).setdefault(task, {})
        task_override["num_workers"] = 1
        task_override["adapted_sample_cache_size"] = 10_000
        destination = output_dir / f"{task}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        result[task] = {
            **dict(configs[task]),
            "source_selected_config": str(source),
            "source_selected_config_sha256": configs[task]["config_sha256"],
            "config": str(destination.resolve()),
            "config_sha256": sha256(destination),
            "runtime_only_overrides": {
                "num_workers": 1,
                "adapted_sample_cache_size": 10_000,
                "numerical_semantics": (
                    "unchanged deterministic tensors and batch ordering; adapted samples are "
                    "memoized in each persistent loader worker"
                ),
            },
        }
    return result


def cache_root(config_path: str) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw = Path(str(config["data"]["cache_root"]))
    return raw if raw.is_absolute() else REPO_ROOT / raw


def verify_partitions(
    public: Mapping[str, Any],
    protected: Mapping[str, Any],
    *,
    dataset_count: int,
) -> None:
    train = {int(value) for value in public["train_indices"]}
    validation = {int(value) for value in public["validation_indices"]}
    test = {int(value) for value in protected["test_indices"]}
    if train & validation or train & test or validation & test:
        raise RuntimeError("train, validation, and protected test partitions overlap")
    if train | validation | test != set(range(dataset_count)):
        raise RuntimeError("fold partitions do not cover the complete task dataset")
    if public["protected_test"]["indices_sha256"] != protected["indices_sha256"]:
        raise RuntimeError("public descriptor and protected indices have different hashes")


def make_job(
    *,
    task: str,
    protocol_key: str,
    fold_id: str,
    outer_fold: int,
    public_path: Path,
    protected_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    public = json.loads(public_path.read_text(encoding="utf-8"))
    protected = json.loads(protected_path.read_text(encoding="utf-8"))
    dataset_count = (
        int(public["train_sample_count"])
        + int(public["validation_sample_count"])
        + int(public["protected_test"]["sample_count"])
    )
    verify_partitions(public, protected, dataset_count=dataset_count)
    if public["task"] != task or protected["task"] != task:
        raise RuntimeError(f"task identity mismatch for {public_path}")
    if int(public["protected_test"]["outer_fold"]) != outer_fold:
        raise RuntimeError(f"public outer-fold mismatch for {public_path}")
    if int(protected["outer_fold"]) != outer_fold:
        raise RuntimeError(f"protected outer-fold mismatch for {protected_path}")
    if public["protocol"] != protected["protocol"]:
        raise RuntimeError(f"protocol mismatch for {public_path}")
    return {
        "protocol_key": protocol_key,
        "protocol": public["protocol"],
        "task": task,
        "fold_id": fold_id,
        "outer_fold": outer_fold,
        "config": config["config"],
        "config_sha256": config["config_sha256"],
        "source_trial": config["source_trial"],
        "public_manifest": str(public_path.resolve()),
        "public_manifest_sha256": sha256(public_path),
        "protected_manifest": str(protected_path.resolve()),
        "protected_manifest_sha256": sha256(protected_path),
        "train_sample_count": int(public["train_sample_count"]),
        "validation_sample_count": int(public["validation_sample_count"]),
        "test_sample_count": int(public["protected_test"]["sample_count"]),
        "dataset_sample_count": dataset_count,
    }


def strict_jobs(tasks: Sequence[str], configs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for task in tasks:
        directory = METHOD_ROOT / "split_registry" / task / "cross_subject"
        for outer_fold in range(5):
            jobs.append(make_job(
                task=task,
                protocol_key="strict_cross_subject",
                fold_id=f"outer{outer_fold}",
                outer_fold=outer_fold,
                public_path=directory / "public" / f"outer{outer_fold}_inner0.json",
                protected_path=directory / "protected" / f"outer{outer_fold}.json",
                config=configs[task],
            ))
    return jobs


def sample_random_jobs(
    tasks: Sequence[str],
    configs: Mapping[str, Mapping[str, Any]],
    registry_root: Path,
) -> list[dict[str, Any]]:
    jobs = []
    for task in tasks:
        dataset = STANetUnifiedTaskDataset(
            get_sta_net_task_spec(task),
            cache_root=str(cache_root(str(configs[task]["config"]))),
        )
        public, protected = build_sample_random_registry(
            dataset, seed=42, outer_folds=5, inner_folds=3
        )
        directory = registry_root / task / "sample_random"
        write_registry(public, protected, directory)
        for outer_fold in range(5):
            jobs.append(make_job(
                task=task,
                protocol_key="sample_random",
                fold_id=f"outer{outer_fold}",
                outer_fold=outer_fold,
                public_path=directory / "public" / f"outer{outer_fold}.json",
                protected_path=directory / "protected" / f"outer{outer_fold}.json",
                config=configs[task],
            ))
    return jobs


def distribute(
    jobs: Sequence[Mapping[str, Any]], lane_count: int, epochs: int
) -> tuple[list[list[dict[str, Any]]], list[float]]:
    lanes: list[list[dict[str, Any]]] = [[] for _ in range(lane_count)]
    loads = [0.0] * lane_count
    weighted = sorted(
        jobs,
        key=lambda row: (
            float(row["train_sample_count"])
            * SECONDS_PER_TRAIN_SAMPLE[str(row["task"])]
            * epochs
        ),
        reverse=True,
    )
    for source in weighted:
        job = dict(source)
        weight = (
            float(job["train_sample_count"])
            * SECONDS_PER_TRAIN_SAMPLE[str(job["task"])]
            * epochs
        )
        lane = min(range(lane_count), key=loads.__getitem__)
        job["estimated_training_seconds"] = weight
        lanes[lane].append(job)
        loads[lane] += weight
    return lanes, loads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d_sta_net_strict_vs_sample_random_5fold_v1_100ep"),
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--lanes-per-gpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unlock-protected-test", action="store_true")
    args = parser.parse_args()
    if args.epochs != 100:
        raise ValueError("formal five-fold protocol is frozen at 100 epochs")
    if not args.dry_run and not args.unlock_protected_test:
        raise RuntimeError("launch requires explicit --unlock-protected-test")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("tasks must be unique")
    configs = selected_configs()

    if args.dry_run:
        strict = strict_jobs(args.tasks, configs)
        print(json.dumps({
            "status": "dry_run",
            "tasks": args.tasks,
            "strict_job_count": len(strict),
            "planned_sample_random_job_count": 5 * len(args.tasks),
            "total_job_count": 10 * len(args.tasks),
            "protected_test_opened": False,
        }, indent=2))
        return

    root = METHOD_ROOT / "runs" / "fivefold" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    configs = cached_execution_configs(configs, args.tasks, root / "runtime_configs")
    jobs = strict_jobs(args.tasks, configs)
    jobs.extend(sample_random_jobs(args.tasks, configs, root / "split_registry"))
    lane_count = len(args.gpus) * args.lanes_per_gpu
    lanes, estimated_loads = distribute(jobs, lane_count, args.epochs)
    jobs = [job for lane in lanes for job in lane]
    protocol = {
        "schema": "sta_net_strict_vs_sample_random_5fold_protocol_freeze_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "protocols": {
            "strict_cross_subject": {
                "protocol": "cross_subject_nested_cv",
                "outer_split_axis": "canonical subject",
                "outer_folds": 5,
                "selected_inner_fold": 0,
                "dependency_group_isolation": True,
            },
            "sample_random": {
                "protocol": "sample_random_nested_cv",
                "outer_split_axis": "dataset sample index",
                "outer_folds": 5,
                "selected_inner_fold": 0,
                "dependency_group_isolation": False,
            },
        },
        "tasks": list(args.tasks),
        "epochs": args.epochs,
        "seed": 42,
        "selection_rule": (
            "fixed final cross-subject-selected hyperparameters; select the best checkpoint "
            "within 100 epochs using only the public inner0 validation partition"
        ),
        "runtime_input_pipeline": (
            "one persistent DataLoader worker per loader with deterministic adapted-sample "
            "memoization; this changes neither tensor values nor batch order"
        ),
        "classification_primary_endpoint": "outer-fold macro-F1 mean and sample SD",
        "regression_primary_endpoint": "outer-fold concordance-correlation mean and sample SD",
        "source_aligned_endpoint": "outer-fold Accuracy mean and sample SD for MI/MA/WG",
        "configs": {task: configs[task] for task in args.tasks},
        "jobs": jobs,
        "jobs_sha256": sha256_json(jobs),
        "job_count": len(jobs),
        "lane_count": lane_count,
        "estimated_lane_training_seconds": estimated_loads,
        "trainer_sha256": sha256(METHOD_ROOT / "train.py"),
        "evaluator_sha256": sha256(METHOD_ROOT / "evaluate_protocol.py"),
        "worker_sha256": sha256(METHOD_ROOT / "fivefold_worker.py"),
        "aggregator_sha256": sha256(METHOD_ROOT / "aggregate_fivefold.py"),
        "source_paper": {
            "citation": (
                "Liu et al. (2025), STA-Net: Spatial-temporal alignment network for "
                "hybrid EEG-fNIRS decoding, Information Fusion 119, 103023"
            ),
            "doi": "10.1016/j.inffus.2025.103023",
            "url": "https://doi.org/10.1016/j.inffus.2025.103023",
            "reported_accuracy_mean_sd": {
                "motor_imagery": [0.6965, 0.0952],
                "mental_arithmetic": [0.8514, 0.0717],
                "wg": [0.7903, 0.0841],
            },
            "comparison_scope": "contextual; original paper used subject-specific evaluation",
        },
        "protected_test_opened": True,
        "protected_open_authorization": (
            "explicit user request to open protected subjects and use all dataset content "
            "for strict and direct sample-level five-fold evaluation"
        ),
    }
    write_json(root / "protocol_freeze_manifest.json", protocol)

    launches = []
    for lane_index, jobs_for_lane in enumerate(lanes):
        lane_id = f"lane_{lane_index:02d}"
        jobs_path = root / "jobs" / f"{lane_id}.json"
        write_json(jobs_path, jobs_for_lane)
        gpu = args.gpus[lane_index % len(args.gpus)]
        log_path = root / "logs" / f"{lane_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            str(METHOD_ROOT / "fivefold_worker.py"),
            "--run-root",
            str(root),
            "--jobs",
            str(jobs_path),
            "--lane-id",
            lane_id,
            "--epochs",
            str(args.epochs),
            "--workers",
            str(args.workers),
            "--unlock-protected-test",
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launches.append({
            "lane_id": lane_id,
            "physical_gpu": gpu,
            "pid": process.pid,
            "job_count": len(jobs_for_lane),
            "estimated_training_seconds": estimated_loads[lane_index],
            "jobs": str(jobs_path),
            "log": str(log_path),
        })

    supervisor_log = root / "logs" / "supervisor.log"
    supervisor_command = [
        sys.executable,
        "-u",
        str(METHOD_ROOT / "fivefold_supervisor.py"),
        "--run-root",
        str(root),
        "--lane-count",
        str(lane_count),
    ]
    with supervisor_log.open("w", encoding="utf-8") as log:
        supervisor = subprocess.Popen(
            supervisor_command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch = {
        "schema": "sta_net_strict_vs_sample_random_5fold_launch_v1",
        "status": "launched",
        "created_at": utc_now(),
        "run_root": str(root),
        "jobs": len(jobs),
        "lanes": launches,
        "supervisor_pid": supervisor.pid,
        "supervisor_log": str(supervisor_log),
    }
    write_json(root / "launch_manifest.json", launch)
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
