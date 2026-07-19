#!/usr/bin/env python3
"""Launch two detached STA-Net tuning workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna

METHOD_ROOT = Path(__file__).resolve().parent
# Multiple lightweight STA-Net trials share each 24-GB GPU. Historical runs
# used only about 1 GB for the source tasks and left the SM mostly idle.
LANES = (
    (0, 0, ("motor_imagery", "dsr")),
    (0, 1, ("wg",)),
    (0, 2, ("refed_regression",)),
    (1, 0, ("mental_arithmetic", "nback")),
    (1, 1, ("visual",)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default=datetime.now().strftime("%Y%m%d_%H%M%S_sta_net_hpo"))
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--base-config", default=str(METHOD_ROOT / "configs" / "tuning_base.yaml"))
    args = parser.parse_args()
    run_root = METHOD_ROOT / "runs" / "tuning" / args.study_id
    run_root.mkdir(parents=True, exist_ok=False)
    storage = f"sqlite:///{(run_root / 'optuna.sqlite3').resolve()}"
    for task in sorted({task for _, _, tasks in LANES for task in tasks}):
        optuna.create_study(
            study_name=f"{args.study_id}__{task}__development_cross_subject",
            storage=storage, direction="maximize", load_if_exists=True,
        )
    launches = []
    for gpu, lane, tasks in LANES:
        command = [
            sys.executable, "-u", str(METHOD_ROOT / "tune.py"),
            "--study-id", args.study_id, "--run-root", str(run_root),
            "--base-config", str(Path(args.base_config).resolve()),
            "--physical-gpu", str(gpu), "--n-trials", str(args.n_trials),
            "--tasks", *tasks,
        ]
        log_path = run_root / f"gpu{gpu}_lane{lane}_tuning.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=METHOD_ROOT.parents[1], stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        launches.append({
            "physical_gpu": gpu, "lane": lane, "pid": process.pid,
            "tasks": list(tasks), "log": str(log_path),
        })
    manifest = {
        "schema": "sta_net_optuna_launch_v1", "status": "workers_launched",
        "study_id": args.study_id, "run_root": str(run_root),
        "rung_epochs": [2, 8, 20, 40, 100], "n_trials_per_task": args.n_trials,
        "gpu_concurrency": {"0": 3, "1": 2},
        "created_at": datetime.now(timezone.utc).isoformat(), "workers": launches,
    }
    (run_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
