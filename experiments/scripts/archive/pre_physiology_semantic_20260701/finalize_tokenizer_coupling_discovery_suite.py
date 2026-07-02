#!/usr/bin/env python
"""Finalize local codebook/interaction discovery suite summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def best_validation_epoch(metrics: dict[str, Any]) -> dict[str, Any]:
    val_epochs = [
        epoch
        for epoch in metrics.get("epochs", [])
        if epoch.get("val_loss") is not None
    ]
    if not val_epochs:
        return {}
    return min(val_epochs, key=lambda epoch: float(epoch["val_loss"]))


def final_validation_epoch(metrics: dict[str, Any]) -> dict[str, Any]:
    val_epochs = [
        epoch
        for epoch in metrics.get("epochs", [])
        if epoch.get("val_loss") is not None
    ]
    return val_epochs[-1] if val_epochs else {}


def gate_verdicts(gate_payload: dict[str, Any]) -> dict[str, str]:
    return gate_payload.get("final_summary", {}).get("gate_verdicts", {})


def gate3_metrics(gate_payload: dict[str, Any]) -> dict[str, Any]:
    gates = gate_payload.get("splits", {}).get("test", {}).get("gates")
    if not gates:
        gates = gate_payload.get("gates", {})
    return gates.get("gate3", {}).get("metrics", {})


def row_for_run(run_dir: Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    gates = read_json(run_dir / "analysis/gate_summary.json")
    best = best_validation_epoch(metrics)
    final = final_validation_epoch(metrics)
    best_metrics = best.get("metrics", {})
    final_metrics = final.get("metrics", {})
    g3 = gate3_metrics(gates)
    predictability = g3.get("cross_modal_token_predictability", {})
    empirical = g3.get("lag_balanced_empirical_audit", {})
    verdicts = gate_verdicts(gates)
    return {
        "run": run_dir.name,
        "best_epoch": best.get("epoch"),
        "best_val_loss": best.get("val_loss"),
        "best_val_primary_loss": best_metrics.get("val_primary_loss"),
        "final_epoch": final.get("epoch"),
        "final_val_loss": final.get("val_loss"),
        "final_val_primary_loss": final_metrics.get("val_primary_loss"),
        "val_source_target_loss": best_metrics.get("val_source_target_loss"),
        "val_observation_loss": best_metrics.get("val_observation_loss"),
        "val_eeg_source_perplexity": best_metrics.get("val_eeg_source_perplexity"),
        "val_fnirs_source_perplexity": best_metrics.get("val_fnirs_source_perplexity"),
        "val_source_code_overlap": best_metrics.get("val_source_code_overlap"),
        "val_local_residual_coupling_loss": best_metrics.get("val_source_coupling_local_residual_loss"),
        "val_interaction_aux_loss": best_metrics.get("val_interaction_aux_loss"),
        "val_shared_state_bottleneck_loss": best_metrics.get("val_shared_state_bottleneck_loss"),
        "gate0": verdicts.get("gate0"),
        "gate1": verdicts.get("gate1"),
        "gate2": verdicts.get("gate2"),
        "gate3": verdicts.get("gate3"),
        "gate4": verdicts.get("gate4"),
        "promotion_verdict": gates.get("final_summary", {}).get("promotion_verdict") or gates.get("promotion_verdict"),
        "gate3_accuracy": predictability.get("accuracy"),
        "gate3_uniform_chance": predictability.get("uniform_chance_accuracy"),
        "gate3_weighted_true_probability": predictability.get("model_weighted_true_token_probability"),
        "gate3_best_loso_gain": empirical.get("best_lag_loso_gain"),
        "gate3_best_mi_above_shuffle": empirical.get("best_mi_above_shuffle"),
    }


def audit_completion(suite_dir: Path) -> dict[str, Any]:
    return {
        "codebook_geometry_runs": len(list((suite_dir / "codebook_geometry").glob("*/summary.json"))),
        "information_drop_runs": len(list((suite_dir / "information_drop_audit").glob("*/summary.json"))),
        "local_coupling_runs": len(list((suite_dir / "local_coupling_audit").glob("*/summary.json"))),
        "audit_done": (suite_dir / "queue_logs/audit.done").exists(),
        "formal_gpu0_done": (suite_dir / "queue_logs/formal_gpu0.done").exists(),
        "formal_gpu1_done": (suite_dir / "queue_logs/formal_gpu1.done").exists(),
        "smoke_gpu0_done": (suite_dir / "queue_logs/smoke_gpu0.done").exists(),
        "smoke_gpu1_done": (suite_dir / "queue_logs/smoke_gpu1.done").exists(),
    }


def condition_key(run_name: str) -> str:
    parts = run_name.split("_")
    for part in parts:
        if part.startswith("t") and part[1:].isdigit():
            return part.upper()
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    run_dirs = sorted((suite_dir / "tokenizer_interventions").glob("k64_t*_seed*"))
    rows = [
        row_for_run(run_dir)
        for run_dir in run_dirs
        if (run_dir / "metrics.json").exists() and (run_dir / "analysis/gate_summary.json").exists()
    ]
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({condition_key(row["run"]) for row in rows}):
        subset = [row for row in rows if condition_key(row["run"]) == condition]
        numeric_keys = [
            "best_val_loss",
            "best_val_primary_loss",
            "val_source_target_loss",
            "val_observation_loss",
            "val_eeg_source_perplexity",
            "val_fnirs_source_perplexity",
            "val_source_code_overlap",
            "gate3_weighted_true_probability",
            "gate3_best_loso_gain",
            "gate3_best_mi_above_shuffle",
        ]
        aggregate: dict[str, Any] = {"n": len(subset)}
        for key in numeric_keys:
            values = [row.get(key) for row in subset if isinstance(row.get(key), (int, float))]
            aggregate[f"{key}_mean"] = sum(values) / len(values) if values else None
        aggregate["gate4_pass_count"] = sum(1 for row in subset if row.get("gate4") == "pass")
        by_condition[condition] = aggregate

    summary = {
        "schema_version": "tokenizer_coupling_discovery_summary_v1",
        "suite_dir": str(suite_dir),
        "audit_completion": audit_completion(suite_dir),
        "conditions": by_condition,
        "runs": rows,
    }
    decision = {
        "status": "complete" if summary["audit_completion"]["audit_done"] and rows else "partial",
        "training_matrix_complete": bool(rows) and summary["audit_completion"]["formal_gpu0_done"] and summary["audit_completion"]["formal_gpu1_done"],
        "audit_complete": summary["audit_completion"]["audit_done"],
        "local_residual_coupling": "do_not_promote",
        "shared_state_bottleneck": "needs_post_training_token_audit",
        "reason": (
            "T1-T4 did not improve Gate3; T6 has a small validation/Gate4 signal but still fails Gate3."
        ),
    }

    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (suite_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_lines = [
        "# Tokenizer Coupling Discovery Suite",
        "",
        f"Status: {decision['status']}",
        f"Training matrix complete: {decision['training_matrix_complete']}",
        f"Audit complete: {decision['audit_complete']}",
        "",
        "Condition means:",
    ]
    for condition, payload in by_condition.items():
        report_lines.append(
            f"- {condition}: best_val={payload.get('best_val_loss_mean')}, "
            f"Gate3 weighted true p={payload.get('gate3_weighted_true_probability_mean')}, "
            f"Gate4 passes={payload.get('gate4_pass_count')}/{payload.get('n')}"
        )
    (suite_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
