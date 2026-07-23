#!/usr/bin/env python3
"""Generate a read-only scientific audit of an EFRM pretraining run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efrm_pytorch.pretraining_analysis import analyze_pretraining_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--stale-after-hours",
        type=float,
        default=1.0,
        help="classify a non-terminal run as stale after this many hours without metrics",
    )
    args = parser.parse_args()
    result = analyze_pretraining_run(
        args.run_dir,
        args.output_dir,
        stale_after_hours=args.stale_after_hours,
    )
    print(json.dumps({
        "run_id": result["run_id"],
        "run_state": result["audit"]["run_state"],
        "completed_epochs": result["audit"]["completed_epoch_count"],
        "best_epoch_1_based": result["audit"]["best_epoch"] + 1,
        "saved_batch_alignment_above_chance": result["interpretation"][
            "saved_batch_alignment_above_chance"
        ],
        "alignment_warning_level": result["interpretation"]["alignment_warning_level"],
    }, indent=2))


if __name__ == "__main__":
    main()
