#!/usr/bin/env python
"""Finalize tokenizer coupling capacity suite summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def final_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json").get("final_metrics", {})
    gates = read_json(run_dir / "analysis/gate_summary.json")
    gate_verdicts = gates.get("final_summary", {}).get("gate_verdicts", {})
    codebooks = gates.get("gates", {}).get("gate1", {}).get("metrics", {}).get("codebooks", {})
    gate3 = gates.get("gates", {}).get("gate3", {}).get("metrics", {})
    row = {
        "run": run_dir.name,
        "best_epoch": metrics.get("best_epoch"),
        "best_val_loss": metrics.get("best_monitor"),
        "val_loss": metrics.get("val_loss"),
        "val_eeg_rec_loss": metrics.get("val_eeg_rec_loss"),
        "val_fnirs_rec_loss": metrics.get("val_fnirs_rec_loss"),
        "val_source_target_loss": metrics.get("val_source_target_loss"),
        "val_source_target_corr_loss": metrics.get("val_source_target_corr_loss"),
        "val_observation_loss": metrics.get("val_observation_loss"),
        "gate0": gate_verdicts.get("gate0"),
        "gate1": gate_verdicts.get("gate1"),
        "gate2": gate_verdicts.get("gate2"),
        "gate3": gate_verdicts.get("gate3"),
        "gate4": gate_verdicts.get("gate4"),
        "promotion_verdict": gates.get("promotion_verdict"),
        "gate3_cross_modal_accuracy": gate3.get("cross_modal_token_predictability", {}).get("accuracy"),
        "gate3_cross_modal_chance": gate3.get("cross_modal_token_predictability", {}).get("chance_accuracy"),
    }
    for branch, payload in codebooks.items():
        row[f"{branch}_k"] = payload.get("codebook_size")
        row[f"{branch}_active"] = payload.get("active_codes")
        row[f"{branch}_dead"] = payload.get("dead_code_count")
        row[f"{branch}_ppl"] = payload.get("perplexity")
        row[f"{branch}_gini"] = payload.get("gini")
    return row


def audit_status(suite_dir: Path, run_name: str) -> dict[str, bool]:
    return {
        "token_export": (suite_dir / "token_exports" / run_name / "manifest.json").exists(),
        "position_audit": (suite_dir / "position_audit" / run_name / "position_event_lag_audit.json").exists(),
        "information_ladder": (suite_dir / "position_audit" / run_name / "physiological_information_ladder.json").exists(),
        "frozen_pairing": (suite_dir / "frozen_pairing_audit" / run_name / "frozen_model_results.json").exists(),
        "factorized_frozen_pairing": any((suite_dir / "frozen_pairing_audit" / run_name).glob("*LR*_probabilities.npy"))
        if (suite_dir / "frozen_pairing_audit" / run_name).exists() else False,
    }


def summarize_frozen(suite_dir: Path, run_name: str) -> dict[str, Any]:
    matrix = suite_dir / "frozen_pairing_audit" / run_name / "model_matrix.csv"
    if not matrix.exists():
        return {"available": False}
    rows = list(csv.DictReader(matrix.open(encoding="utf-8")))
    global_rows = [
        row for row in rows
        if row["train_domain"] == "global" and row["test_domain"] == "global"
    ]
    best = None
    if global_rows:
        best = max(global_rows, key=lambda row: float(row["nuisance_adjusted_gain"]))
    task_positive = [
        row for row in rows
        if row["train_domain"] == row["test_domain"]
        and row["test_domain"] != "global"
        and float(row["nuisance_bootstrap_ci_low"]) > 0
    ]
    return {
        "available": True,
        "best_global_model": None if best is None else best["model"],
        "best_global_nuisance_adjusted_gain": None if best is None else float(best["nuisance_adjusted_gain"]),
        "task_positive_count": len(task_positive),
        "task_positive_domains": sorted({row["test_domain"] for row in task_positive}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    run_dirs = sorted((suite_dir / "capacity_sweep").glob("s2_croce_local_highwl_v2_capacity_*"))
    rows = []
    summary = {
        "schema_version": "tokenizer_coupling_capacity_summary_v1",
        "suite_dir": str(suite_dir),
        "runs": {},
        "decision": {},
    }
    for run_dir in run_dirs:
        if not (run_dir / "metrics.json").exists() or not (run_dir / "analysis/gate_summary.json").exists():
            continue
        row = final_metrics(run_dir)
        row.update({f"audit_{key}": value for key, value in audit_status(suite_dir, run_dir.name).items()})
        frozen = summarize_frozen(suite_dir, run_dir.name)
        row["frozen_available"] = frozen["available"]
        row["frozen_best_global_nuisance_adjusted_gain"] = frozen.get("best_global_nuisance_adjusted_gain")
        row["frozen_task_positive_count"] = frozen.get("task_positive_count")
        rows.append(row)
        summary["runs"][run_dir.name] = {"metrics": row, "frozen": frozen}

    k256_rows = [row for row in rows if "k256" in row["run"]]
    summary["decision"] = {
        "status": "capacity_phase_complete" if rows else "no_completed_runs",
        "k256_dense_training": "do_not_promote",
        "k256_reason": (
            "K256 improves reconstruction/capacity but Gate3/Gate4 fail; use as capacity upper-bound probe."
            if k256_rows else "K256 run not found"
        ),
        "default_training_k": "K64",
        "conditional_training_k": "K128",
        "requires_discovery_audits": True,
    }
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (suite_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    (suite_dir / "decision.json").write_text(json.dumps(summary["decision"], indent=2, sort_keys=True) + "\n")
    report = [
        "# Tokenizer Coupling Capacity Suite",
        "",
        f"Status: {summary['decision']['status']}",
        "",
        "K256 conclusion: do not promote to dense coupling training; keep as capacity upper-bound probe.",
        "",
        "Completed runs:",
    ]
    for row in rows:
        report.append(
            f"- `{row['run']}`: best_val={row.get('best_val_loss')}, "
            f"Gate0/1/2/3/4={row.get('gate0')}/{row.get('gate1')}/{row.get('gate2')}/{row.get('gate3')}/{row.get('gate4')}, "
            f"frozen={row.get('frozen_available')}"
        )
    (suite_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
