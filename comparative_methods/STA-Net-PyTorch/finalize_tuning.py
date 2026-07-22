#!/usr/bin/env python3
"""Wait for detached tuning workers, then select checkpoints and report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

METHOD_ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_workers(root: Path, manifest: Mapping[str, Any], poll_seconds: float) -> None:
    workers = list(manifest["workers"])
    while True:
        states: dict[str, str] = {}
        failures: list[str] = []
        for worker in workers:
            worker_id = str(worker["worker_id"])
            path = root / "workers" / f"{worker_id}.json"
            if path.exists():
                status = read_json(path)
                state = str(status.get("status", "unknown"))
                states[worker_id] = state
                if state == "failed":
                    failures.append(f"{worker_id}: {status.get('error_type')}: {status.get('error')}")
            elif not pid_is_alive(int(worker["pid"])):
                states[worker_id] = "dead_without_status"
                failures.append(f"{worker_id}: process exited before writing status")
            else:
                states[worker_id] = "starting"
        write_json(root / "supervisor_status.json", {
            "schema": "sta_net_tuning_supervisor_v1",
            "status": "failed" if failures else "waiting",
            "workers": states,
            "updated_at": utc_now(),
        })
        if failures:
            raise RuntimeError("; ".join(failures))
        if states and all(state == "completed" for state in states.values()):
            return
        time.sleep(poll_seconds)


def run_command(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=METHOD_ROOT.parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def finalize(root: Path, poll_seconds: float) -> None:
    manifest_path = root / "launch_manifest.json"
    manifest = read_json(manifest_path)
    tasks = list(manifest["trial_allocation"])
    wait_for_workers(root, manifest, poll_seconds)

    finalization_log = root / "finalization.log"
    selection_dir = root / "final_validation_selection_v2"
    analysis_dir = root / "analysis"
    run_command([
        sys.executable,
        str(METHOD_ROOT / "select_best_checkpoints.py"),
        "--run-root", str(root),
        "--study-id", str(manifest["study_id"]),
        "--output-dir", str(selection_dir),
        "--tasks", *tasks,
    ], finalization_log)
    run_command([
        sys.executable,
        str(METHOD_ROOT / "analyze_tuning.py"),
        "--run-root", str(root),
        "--output-dir", str(analysis_dir),
    ], finalization_log)

    selection = read_json(selection_dir / "selection_manifest.json")
    summary = read_json(analysis_dir / "summary.json")
    selected_rows = [
        (
            task,
            int(row["selected_trial"]),
            int(row["selected"]["checkpoint_epoch"]),
            float(row["selected"]["checkpoint_metric"]),
        )
        for task, row in selection["tasks"].items()
    ]
    report_lines = [
        f"# STA-Net tuning rerun completion: `{manifest['study_id']}`",
        "",
        f"Completed: {utc_now()}",
        "",
        "> Development-split tuning only. Protected test data were not opened.",
        "",
        f"Objective policy: `{manifest['objective_policy']}`.",
        "",
        (
            f"The rerun recorded {summary['trial_count']} trials; "
            + ", ".join(
                f"{state}={count}" for state, count in sorted(summary["overall_states"].items())
            )
            + "."
        ),
        "",
        "| Task | Selected trial | Checkpoint epoch | Validation metric |",
        "| --- | ---: | ---: | ---: |",
    ]
    report_lines.extend(
        f"| {task} | T{trial} | {epoch} | {metric:.6f} |"
        for task, trial, epoch, metric in selected_rows
    )
    report_lines.extend([
        "",
        "Detailed trajectories, failure analysis, metrics, and figures are in `analysis/tuning_report.md`.",
        "The hash-pinned checkpoint decision is in `final_validation_selection_v2/selection_manifest.json`.",
        "",
    ])
    (root / "completion_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest.update({
        "status": "completed",
        "completed_at": utc_now(),
        "analysis": str(analysis_dir),
        "selection": str(selection_dir),
        "completion_report": str(root / "completion_report.md"),
    })
    write_json(manifest_path, manifest)
    write_json(root / "supervisor_status.json", {
        "schema": "sta_net_tuning_supervisor_v1",
        "status": "completed",
        "analysis": str(analysis_dir),
        "selection": str(selection_dir),
        "updated_at": utc_now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    try:
        finalize(root, max(1.0, args.poll_seconds))
    except Exception as error:
        manifest_path = root / "launch_manifest.json"
        manifest = read_json(manifest_path)
        manifest.update({
            "status": "failed",
            "failed_at": utc_now(),
            "failure_type": type(error).__name__,
            "failure": str(error),
        })
        write_json(manifest_path, manifest)
        write_json(root / "supervisor_status.json", {
            "schema": "sta_net_tuning_supervisor_v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_at": utc_now(),
        })
        raise


if __name__ == "__main__":
    main()
