#!/usr/bin/env python3
"""Monitor STA-Net within-subject lanes and aggregate a complete run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

METHOD_ROOT = Path(__file__).resolve().parent


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--lane-count", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    status_path = root / "supervisor_status.json"
    while True:
        statuses = []
        for lane in range(args.lane_count):
            path = root / "status" / f"lane_{lane:02d}.json"
            statuses.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
        failed = [row for row in statuses if row is not None and row.get("status") == "failed"]
        completed = sum(row is not None and row.get("status") == "completed" for row in statuses)
        write_json(status_path, {
            "schema": "sta_net_within_subject_supervisor_v1",
            "status": "failed" if failed else ("aggregating" if completed == args.lane_count else "running"),
            "completed_lanes": completed,
            "lane_count": args.lane_count,
            "failed_lanes": [row["lane_id"] for row in failed],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        if failed:
            raise RuntimeError(f"within-subject lanes failed: {[row['lane_id'] for row in failed]}")
        if completed == args.lane_count:
            break
        time.sleep(args.poll_seconds)
    command = [
        sys.executable,
        str(METHOD_ROOT / "aggregate_within_subject.py"),
        "--run-root",
        str(root),
    ]
    with (root / "aggregate.log").open("w", encoding="utf-8") as log:
        completed_process = subprocess.run(
            command, cwd=METHOD_ROOT.parents[1], stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed_process.returncode != 0:
        write_json(status_path, {
            "schema": "sta_net_within_subject_supervisor_v1",
            "status": "aggregation_failed",
            "return_code": completed_process.returncode,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        raise SystemExit(completed_process.returncode)
    write_json(status_path, {
        "schema": "sta_net_within_subject_supervisor_v1",
        "status": "completed",
        "completed_lanes": args.lane_count,
        "lane_count": args.lane_count,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
