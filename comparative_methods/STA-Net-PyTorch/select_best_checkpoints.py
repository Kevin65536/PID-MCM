#!/usr/bin/env python3
"""Select the best predictive checkpoint across completed 100-epoch trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import optuna
import torch

TASKS = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
)
SCHEMA = "sta_net_predictive_checkpoint_selection_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric_contract(task: str) -> tuple[str, str]:
    return (
        ("masked_rmse_scaled", "min")
        if task == "refed_regression"
        else ("macro_f1", "max")
    )


def select_candidate(candidates: Sequence[Mapping[str, Any]], mode: str) -> Mapping[str, Any]:
    if not candidates:
        raise RuntimeError("no completed 100-epoch candidates were found")
    if mode not in {"min", "max"}:
        raise ValueError(f"unsupported selection mode: {mode}")
    if mode == "min":
        return min(candidates, key=lambda row: (float(row["checkpoint_metric"]), int(row["trial_number"])))
    return max(candidates, key=lambda row: (float(row["checkpoint_metric"]), -int(row["trial_number"])))


def validation_row(run_dir: Path, epoch: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics" / "validation_epochs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if int(row["epoch"]) == epoch]
    if len(matches) != 1:
        raise RuntimeError(f"expected one validation row for epoch={epoch} under {run_dir}")
    return matches[0]


def collect_candidates(
    root: Path,
    study_id: str,
    task: str,
    storage: str,
) -> tuple[list[dict[str, Any]], int]:
    study = optuna.load_study(
        study_name=f"{study_id}__{task}__development_cross_subject",
        storage=storage,
    )
    metric_name, mode = metric_contract(task)
    candidates: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        trial_dir = root / "trials" / task / f"trial_{trial.number:05d}"
        trial_manifest = json.loads((trial_dir / "trial_manifest.json").read_text(encoding="utf-8"))
        if trial_manifest.get("status") != "completed_100_epochs":
            raise RuntimeError(f"Optuna COMPLETE trial lacks a 100-epoch manifest: {trial_dir}")
        run_dir = trial_dir / "run"
        checkpoint_path = run_dir / "checkpoint_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection_metric") != metric_name:
            raise RuntimeError(f"checkpoint selection metric drift for {task} trial {trial.number}")
        epoch = int(checkpoint["epoch"])
        checkpoint_metric = float(checkpoint["best_validation_metric"])
        row = validation_row(run_dir, epoch)
        observed_metric = float(row[metric_name])
        if not math.isclose(checkpoint_metric, observed_metric, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"checkpoint/metric row mismatch for {task} trial {trial.number}")
        config_path = trial_dir / "config.yaml"
        candidates.append({
            "trial_number": int(trial.number),
            "trial_endpoint_objective": float(trial.value),
            "trial_endpoint_metric": float(-trial.value if mode == "min" else trial.value),
            "checkpoint_epoch": epoch,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "config": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "run_dir": str(run_dir.resolve()),
            "validation_metrics": {
                key: row.get(key)
                for key in (
                    "accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa",
                    "masked_mae_scaled", "masked_rmse_scaled", "sample_count",
                )
                if key in row
            },
        })
        del checkpoint
    return candidates, int(study.best_trial.number)


def run(args: argparse.Namespace) -> Path:
    root = Path(args.run_root).resolve()
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "final_validation_selection_v1"
    )
    output.mkdir(parents=True, exist_ok=False)
    selected_runs = output / "selected_runs"
    selected_runs.mkdir()
    storage = f"sqlite:///{root / 'optuna.sqlite3'}"
    task_payload: dict[str, Any] = {}
    for task in args.tasks:
        metric_name, mode = metric_contract(task)
        candidates, endpoint_winner = collect_candidates(root, args.study_id, task, storage)
        selected = dict(select_candidate(candidates, mode))
        source_run = Path(selected["run_dir"])
        link = selected_runs / task
        link.symlink_to(os.path.relpath(source_run, start=selected_runs), target_is_directory=True)
        task_payload[task] = {
            "selection_metric": metric_name,
            "selection_mode": mode,
            "completed_100_epoch_candidate_count": len(candidates),
            "study_endpoint_winner_trial": endpoint_winner,
            "selected_trial": selected["trial_number"],
            "selection_changed_trial": int(selected["trial_number"]) != endpoint_winner,
            "selected": selected,
            "candidates": sorted(candidates, key=lambda row: int(row["trial_number"])),
        }
    manifest = {
        "schema": SCHEMA,
        "study_id": args.study_id,
        "source_run_root": str(root),
        "selected_runs": str(selected_runs),
        "selection_partition": "development_cross_subject_validation",
        "selection_rule": (
            "Across Optuna COMPLETE trials that reached 100 epochs, select the historical "
            "checkpoint_best with maximum validation macro-F1 for classification or minimum "
            "validation masked scaled RMSE for regression."
        ),
        "protected_test_opened": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": task_payload,
    }
    write_json(output / "selection_manifest.json", manifest)
    print(json.dumps({
        "status": "completed",
        "output_dir": str(output),
        "selected_trials": {task: row["selected_trial"] for task, row in task_payload.items()},
    }, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
