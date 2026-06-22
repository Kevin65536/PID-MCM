#!/usr/bin/env python
"""Finalize context-conditioned coupling suite summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def best_validation_epoch(metrics: dict[str, Any]) -> dict[str, Any]:
    val_epochs = [epoch for epoch in metrics.get("epochs", []) if epoch.get("val_loss") is not None]
    if not val_epochs:
        return {}
    return min(val_epochs, key=lambda epoch: float(epoch["val_loss"]))


def final_validation_epoch(metrics: dict[str, Any]) -> dict[str, Any]:
    val_epochs = [epoch for epoch in metrics.get("epochs", []) if epoch.get("val_loss") is not None]
    return val_epochs[-1] if val_epochs else {}


def gate_payload(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis/gate_summary.json"
    return read_json(path) if path.exists() else {}


def gate_verdicts(payload: dict[str, Any]) -> dict[str, str]:
    return payload.get("final_summary", {}).get("gate_verdicts", {})


def gate3_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    gates = payload.get("splits", {}).get("test", {}).get("gates")
    if not gates:
        gates = payload.get("gates", {})
    return gates.get("gate3", {}).get("metrics", {})


def condition_key(run_name: str) -> str:
    for part in run_name.split("_"):
        if part.startswith("c") and part[1:].isdigit():
            return part.upper()
    return "unknown"


def row_for_run(run_dir: Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    gates = gate_payload(run_dir)
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
        "condition": condition_key(run_dir.name),
        "best_epoch": best.get("epoch"),
        "best_val_loss": best.get("val_loss"),
        "best_val_primary_loss": best_metrics.get("val_primary_loss"),
        "final_epoch": final.get("epoch"),
        "final_val_loss": final.get("val_loss"),
        "final_val_primary_loss": final_metrics.get("val_primary_loss"),
        "val_eeg_source_perplexity": best_metrics.get("val_eeg_source_perplexity"),
        "val_fnirs_source_perplexity": best_metrics.get("val_fnirs_source_perplexity"),
        "val_source_code_overlap": best_metrics.get("val_source_code_overlap"),
        "val_context_residual_loss": best_metrics.get("val_source_coupling_context_residual_loss"),
        "val_context_pair_likelihood_loss": best_metrics.get("val_source_coupling_context_pair_likelihood_loss"),
        "val_context_entropy_loss": best_metrics.get("val_source_coupling_context_entropy_loss"),
        "val_context_balance_loss": best_metrics.get("val_source_coupling_context_balance_loss"),
        "val_context_residual_l1_loss": best_metrics.get("val_source_coupling_context_residual_l1_loss"),
        "val_context_entropy": best_metrics.get("val_source_coupling_context_entropy"),
        "val_context_max_prob": best_metrics.get("val_source_coupling_context_max_prob"),
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


def mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def aggregate_conditions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    numeric_keys = [
        "best_val_loss",
        "best_val_primary_loss",
        "val_eeg_source_perplexity",
        "val_fnirs_source_perplexity",
        "val_context_residual_loss",
        "val_context_pair_likelihood_loss",
        "val_context_entropy",
        "val_context_max_prob",
        "gate3_best_loso_gain",
        "gate3_best_mi_above_shuffle",
    ]
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        aggregate: dict[str, Any] = {"n": len(subset)}
        for key in numeric_keys:
            aggregate[f"{key}_mean"] = mean([row.get(key) for row in subset])
        aggregate["gate4_pass_count"] = sum(1 for row in subset if row.get("gate4") == "pass")
        by_condition[condition] = aggregate
    return by_condition


def decision_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    conditions = summary.get("conditions", {})
    c0 = conditions.get("C0", {})
    decisions: dict[str, Any] = {
        "status": "complete" if summary.get("runs") else "partial",
        "training_matrix_complete": bool(summary.get("runs")),
        "gate3a_global_shared_coupling": "use_existing_gate3_for_global_baseline",
        "gate3b_context_local_coupling": "needs_context_audit_outputs",
        "gate3c_leakage_guard": "not_promotable_until_leakage_probe_is_available",
    }
    baseline_mi = c0.get("gate3_best_mi_above_shuffle_mean")
    promoted = []
    for condition, payload in conditions.items():
        if condition == "C0":
            continue
        mi = payload.get("gate3_best_mi_above_shuffle_mean")
        loso = payload.get("gate3_best_loso_gain_mean")
        visual_pass = (
            isinstance(mi, (int, float))
            and isinstance(baseline_mi, (int, float))
            and float(mi) >= 1.25 * float(baseline_mi)
        )
        utility_pass = isinstance(loso, (int, float)) and float(loso) > 0.0
        if visual_pass and utility_pass:
            promoted.append(condition)
    decisions["context_coupling_candidates"] = promoted
    if promoted:
        decisions["next_step"] = "run K128 confirmatory plus leakage/context audits"
    else:
        decisions["next_step"] = "do not promote coupling-only mechanism without stronger context audit evidence"
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    run_dirs = sorted((suite_dir / "tokenizer_interventions").glob("k64_c*_seed*"))
    rows = [
        row_for_run(run_dir)
        for run_dir in run_dirs
        if (run_dir / "metrics.json").exists()
    ]
    summary = {
        "schema_version": "tokenizer_context_coupling_summary_v1",
        "suite_dir": str(suite_dir),
        "conditions": aggregate_conditions(rows),
        "runs": rows,
    }
    decision = decision_from_summary(summary)

    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (suite_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Tokenizer Context Coupling Suite",
        "",
        f"Status: {decision['status']}",
        f"Next step: {decision['next_step']}",
        "",
        "Condition means:",
    ]
    for condition, payload in summary["conditions"].items():
        lines.append(
            f"- {condition}: best_val={payload.get('best_val_loss_mean')}, "
            f"context_pair={payload.get('val_context_pair_likelihood_loss_mean')}, "
            f"best_LOSO={payload.get('gate3_best_loso_gain_mean')}, "
            f"best_MI_above_shuffle={payload.get('gate3_best_mi_above_shuffle_mean')}, "
            f"Gate4 passes={payload.get('gate4_pass_count')}/{payload.get('n')}"
        )
    (suite_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
