#!/usr/bin/env python3
"""Write a post-hoc protocol boundary and expectation audit for a completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METHOD_ROOT = Path(__file__).resolve().parent
PAPER_ACCURACY = {
    "motor_imagery": 0.6965,
    "mental_arithmetic": 0.8514,
    "wg": 0.7903,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    aggregate_path = root / "aggregate" / "summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    source_runner = METHOD_ROOT.parent / "STA-Net" / "run_sta_net.py"
    comparisons: dict[str, Any] = {}
    for task, paper_value in PAPER_ACCURACY.items():
        metric = aggregate["task_summaries"][task]["subject_level"]["accuracy"]
        comparisons[task] = {
            "shared_protocol_accuracy": metric["mean"],
            "shared_protocol_subject_bootstrap_95_ci": metric["subject_bootstrap_95_ci"],
            "paper_reported_accuracy_reference": paper_value,
            "delta_percentage_points": 100.0 * (metric["mean"] - paper_value),
        }
    payload = {
        "schema": "sta_net_within_subject_protocol_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outcome_reviewed": True,
        "role": "post-hoc interpretation; does not modify frozen metrics or model selection",
        "formal_aggregate": str(aggregate_path),
        "formal_aggregate_sha256": sha256(aggregate_path),
        "released_runner": str(source_runner),
        "released_runner_sha256": sha256(source_runner),
        "shared_within_subject_run_complete": aggregate["fold_count"] == 763,
        "shared_within_subject_result_valid": True,
        "source_paper_protocol_matched": False,
        "source_paper_reproduction_supported": False,
        "protocol_difference": {
            "released_mi_ma_runner": (
                "For each held-out 200-sample session, selects 80 validation samples from "
                "the other 400, trains on 320, then refits on all 400 development samples."
            ),
            "shared_registry_mi_ma": (
                "For each held-out 20-sample session, trains on one 20-sample session and "
                "selects on another 20-sample session; no train+validation refit is performed."
            ),
            "implication": (
                "The shared protocol is a much lower-data dependency-group generalization test. "
                "Its scores must not be labeled a matched reproduction of paper accuracies."
            ),
        },
        "source_task_accuracy_comparison": comparisons,
        "expectation_assessment": {
            "meets_paper_accuracy_reference": False,
            "shows_clear_non_cross_subject_advantage": False,
            "evidence": (
                "MI accuracy is near binary chance and its Kappa interval includes zero; "
                "Nback, DSR, and Visual Kappa are also near zero; REFED mean subject CCC is negative. "
                "MA and WG are above chance but remain far below paper accuracy references."
            ),
        },
    }
    output_json = root / "aggregate" / "protocol_audit.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STA-Net within-subject protocol audit",
        "",
        "The 763-fold shared within-subject run is complete and internally valid, but it is",
        "**not** a matched reproduction of the released STA-Net subject-specific protocol.",
        "",
        "| Task | Shared Accuracy | 95% subject CI | Paper reference | Delta (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, row in comparisons.items():
        ci = row["shared_protocol_subject_bootstrap_95_ci"]
        lines.append(
            f"| {task} | {row['shared_protocol_accuracy']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{row['paper_reported_accuracy_reference']:.4f} | "
            f"{row['delta_percentage_points']:.2f} |"
        )
    lines.extend([
        "",
        "The released MI/MA runner uses roughly 320 train, 80 validation, and 200 test",
        "samples per fold, followed by a refit on all 400 development samples. The shared",
        "registry uses 20 train, 20 validation, and 20 test samples without refitting.",
        "",
        "Conclusion: retain these values as the shared dependency-group within-subject",
        "benchmark. Do not claim paper-level source-protocol reproduction or a clear",
        "within-subject advantage from this run.",
    ])
    (root / "aggregate" / "protocol_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
