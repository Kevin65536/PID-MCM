#!/usr/bin/env python3
"""Freeze a completed 100-epoch tuning winner for confirmatory evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import optuna

METHOD_ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    storage = f"sqlite:///{root / 'optuna.sqlite3'}"
    name = f"{args.study_id}__{args.task}__development_cross_subject"
    study = optuna.load_study(study_name=name, storage=storage)
    trial = study.best_trial
    trial_dir = root / "trials" / args.task / f"trial_{trial.number:05d}"
    trial_manifest = json.loads((trial_dir / "trial_manifest.json").read_text(encoding="utf-8"))
    if trial_manifest.get("status") != "completed_100_epochs":
        raise RuntimeError("only a completed 100-epoch trial can be frozen")
    config = trial_dir / "config.yaml"
    checkpoint = trial_dir / "run" / "checkpoint_best.pt"
    split = root / "splits" / args.task / "development_cross_subject.json"
    payload = {
        "schema": "sta_net_frozen_tuning_winner_v1",
        "study_id": args.study_id,
        "study_name": name,
        "task": args.task,
        "trial_number": trial.number,
        "objective_value": trial.value,
        "parameters": trial.params,
        "config": str(config),
        "config_sha256": sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "split_manifest": str(split),
        "split_manifest_sha256": sha256(split),
        "trainer_sha256": sha256(METHOD_ROOT / "train.py"),
        "model_sha256": sha256(METHOD_ROOT / "sta_net_pytorch" / "model.py"),
        "adapter_sha256": sha256(METHOD_ROOT / "sta_net_pytorch" / "data.py"),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protected_test_opened": False,
    }
    output = Path(args.output).resolve() if args.output else root / "frozen" / f"{args.task}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "manifest": str(output), "trial": trial.number}, indent=2))


if __name__ == "__main__":
    main()
