#!/usr/bin/env python
"""Aggregate causal shared-private tokenizer experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def condition(name: str) -> str:
    return next((part.upper() for part in name.split("_") if part.startswith("m") and part[1:].isdigit()), "unknown")


def values(epoch: dict[str, Any]) -> dict[str, Any]:
    return {**epoch, **epoch.get("loss_breakdown", {}), **epoch.get("metrics", {}), **epoch.get("scalars", {})}


def select_epochs(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    epochs = [values(item) for item in metrics.get("epochs", []) if item.get("val_loss") is not None]
    if not epochs:
        return {}, {}
    hard = min(epochs, key=lambda item: float(item.get("val_primary_loss", item["val_loss"])))
    eligible = [item for item in epochs if item.get("val_cross_modal_alignment_unscaled_loss") is not None]
    alignment = min(eligible, key=lambda item: float(item["val_cross_modal_alignment_unscaled_loss"])) if eligible else hard
    return hard, alignment


def gate_values(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis/gate_summary.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    gates = payload.get("splits", {}).get("test", {}).get("gates", {})
    gate3 = gates.get("gate3", {}).get("metrics", {})
    audit = gate3.get("lag_balanced_empirical_audit", {})
    return {
        "gate1": gates.get("gate1", {}).get("status"),
        "gate3": gates.get("gate3", {}).get("status"),
        "gate4": gates.get("gate4", {}).get("status"),
        "gate3_best_mi_above_shuffle": number(audit.get("best_mi_above_shuffle")),
        "gate3_best_mi_lag": audit.get("best_lag_by_mi_above_shuffle"),
        "gate3_best_loso_gain": number(audit.get("best_lag_loso_gain")),
        "gate3_best_loso_lag": audit.get("best_lag_by_loso_gain"),
    }


def row(run_dir: Path, stage: str) -> dict[str, Any]:
    hard, alignment = select_epochs(read_json(run_dir / "metrics.json"))
    selected = alignment if condition(run_dir.name) == "M0" else hard
    result = {
        "stage": stage,
        "run": run_dir.name,
        "condition": condition(run_dir.name),
        "seed": run_dir.name.rsplit("seed", 1)[-1],
        "hard_epoch": hard.get("epoch"),
        "alignment_epoch": alignment.get("epoch"),
        "val_primary_loss": number(hard.get("val_primary_loss")),
        "val_eeg_source_perplexity": number(hard.get("val_eeg_source_perplexity")),
        "val_fnirs_source_perplexity": number(hard.get("val_fnirs_source_perplexity")),
        "continuous_masked_loss": number(selected.get("val_cross_modal_masked_latent_loss")),
        "continuous_pairing_gain": number(selected.get("val_cross_modal_masked_pairing_gain")),
        "continuous_margin_loss": number(selected.get("val_cross_modal_paired_margin_loss")),
        "temporal_nce": number(selected.get("val_cross_modal_temporal_nce_loss")),
        "cross_token_objective": number(selected.get("val_cross_token_objective")),
        "cross_token_pairing_gain": number(selected.get("val_cross_token_pairing_gain")),
        "cross_token_hard_pairing_gain": number(selected.get("val_cross_token_hard_pairing_gain")),
        "cross_token_pair_match_rate": number(selected.get("val_cross_token_pair_match_rate")),
        "cross_token_shuffled_match_rate": number(selected.get("val_cross_token_shuffled_match_rate")),
        "cross_token_predicted_perplexity": number(selected.get("val_cross_token_predicted_perplexity")),
        "cross_token_teacher_perplexity": number(selected.get("val_cross_token_teacher_perplexity")),
        "adaptive_lag_entropy": number(selected.get("val_cross_modal_adaptive_lag_entropy")),
        "adaptive_lag_mass_2_3": number(selected.get("val_cross_modal_adaptive_lag_mass")),
        "gradient_ratio": number(selected.get("alignment_gradient_ratio")),
    }
    result.update(gate_values(run_dir))
    return result


def collect(suite: Path) -> list[dict[str, Any]]:
    rows = []
    for stage, leaf in (("smoke", "smoke"), ("screening", "screening"), ("confirmatory", "tokenizer_interventions")):
        for run_dir in sorted((suite / leaf).glob("*m*_causal_shared_private_seed*")):
            if (run_dir / "metrics.json").exists():
                rows.append(row(run_dir, stage))
    return rows


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage = "confirmatory" if any(row["stage"] == "confirmatory" for row in rows) else (
        "screening" if any(row["stage"] == "screening" for row in rows) else "smoke"
    )
    selected = [row for row in rows if row["stage"] == stage]
    by_condition = {key: [row for row in selected if row["condition"] == key] for key in {row["condition"] for row in selected}}
    m0_gain = max((row.get("continuous_pairing_gain") or 0.0 for row in by_condition.get("M0", [])), default=0.0)
    cross_candidates = []
    for key in ("M3", "M4", "M5"):
        subset = by_condition.get(key, [])
        if subset and all((row.get("cross_token_pairing_gain") or 0.0) > 0 for row in subset):
            cross_candidates.append(key)
    return {
        "status": "complete" if selected else "pending",
        "evaluated_stage": stage,
        "continuous_ceiling_positive": m0_gain > 0,
        "cross_token_positive_conditions": cross_candidates,
        "next_step": (
            "inspect task-local and hard-token audits before confirmatory launch"
            if stage == "screening" else
            "launch screening only after smoke finite/gradient checks pass"
            if stage == "smoke" else
            "compare two-seed held-out and downstream evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    suite = Path(args.suite_dir).resolve()
    rows = collect(suite)
    decision = decide(rows)
    fields = sorted({key for item in rows for key in item})
    if fields:
        with (suite / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {"schema_version": "tokenizer_causal_shared_private_summary_v1", "rows": rows, "decision": decision}
    (suite / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (suite / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    lines = ["# Causal Shared-Private Tokenizer Suite", "", f"Stage: {decision['evaluated_stage']}", ""]
    for item in rows:
        if item["stage"] != decision["evaluated_stage"]:
            continue
        lines.append(
            f"- {item['condition']}: continuous_gain={item.get('continuous_pairing_gain')}, "
            f"cross_gain={item.get('cross_token_pairing_gain')}, hard_cross_gain={item.get('cross_token_hard_pairing_gain')}, "
            f"MI={item.get('gate3_best_mi_above_shuffle')}, LOSO={item.get('gate3_best_loso_gain')}"
        )
    (suite / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
