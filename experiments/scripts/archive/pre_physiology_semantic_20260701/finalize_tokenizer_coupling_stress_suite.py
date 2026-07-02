#!/usr/bin/env python
"""Finalize extreme global-coupling stress test summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_key(run_name: str) -> str:
    for part in run_name.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            return part.upper()
    return "unknown"


def seed_key(run_name: str) -> str:
    if "seed" not in run_name:
        return "unknown"
    return run_name.rsplit("seed", 1)[-1]


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


def best_empirical_value(empirical: dict[str, Any], key: str) -> float | None:
    values: list[float] = []
    for item in empirical.get("per_lag", []):
        if key == "loso_gain":
            value = item.get("leave_one_subject_out", {}).get("accuracy_gain")
        else:
            value = item.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return max(values) if values else None


def gradient_conflict_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    rec_prefixes = ("grad_cosine_eeg_rec_loss__vs__source_coupling", "grad_cosine_fnirs_rec_loss__vs__source_coupling")
    coupling_names = (
        "source_coupling_lag_focus_loss",
        "source_coupling_joint_entropy_loss",
        "source_coupling_pair_likelihood_loss",
        "source_coupling_lag_evidence_loss",
    )
    cosines: list[float] = []
    named: dict[str, list[float]] = {name: [] for name in coupling_names}
    shares: list[float] = []
    for epoch in metrics.get("epochs", []):
        epoch_metrics = epoch.get("metrics", {})
        for key, value in epoch_metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith(rec_prefixes):
                cosines.append(float(value))
                for coupling_name in coupling_names:
                    if coupling_name in key:
                        named[coupling_name].append(float(value))
            if key.startswith("grad_share_source_coupling") or key.startswith("grad_group_share_source_coupling"):
                shares.append(float(value))
    return {
        "rec_coupling_cosine_min": min(cosines) if cosines else None,
        "rec_coupling_cosine_mean": sum(cosines) / len(cosines) if cosines else None,
        "rec_coupling_negative_fraction": (
            sum(1 for value in cosines if value < 0.0) / len(cosines)
            if cosines else None
        ),
        "joint_entropy_rec_cosine_mean": (
            sum(named["source_coupling_joint_entropy_loss"]) / len(named["source_coupling_joint_entropy_loss"])
            if named["source_coupling_joint_entropy_loss"] else None
        ),
        "pair_likelihood_rec_cosine_mean": (
            sum(named["source_coupling_pair_likelihood_loss"]) / len(named["source_coupling_pair_likelihood_loss"])
            if named["source_coupling_pair_likelihood_loss"] else None
        ),
        "coupling_grad_share_mean": sum(shares) / len(shares) if shares else None,
    }


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
    grad = gradient_conflict_summary(metrics)
    return {
        "run": run_dir.name,
        "condition": condition_key(run_dir.name),
        "seed": seed_key(run_dir.name),
        "best_epoch": best.get("epoch"),
        "best_val_loss": best.get("val_loss"),
        "best_val_primary_loss": best_metrics.get("val_primary_loss"),
        "final_epoch": final.get("epoch"),
        "final_val_loss": final.get("val_loss"),
        "final_val_primary_loss": final_metrics.get("val_primary_loss"),
        "best_val_eeg_rec_loss": best_metrics.get("val_eeg_rec_loss"),
        "best_val_fnirs_rec_loss": best_metrics.get("val_fnirs_rec_loss"),
        "final_val_eeg_rec_loss": final_metrics.get("val_eeg_rec_loss"),
        "final_val_fnirs_rec_loss": final_metrics.get("val_fnirs_rec_loss"),
        "best_val_coupling_weighted": best_metrics.get("val_source_coupling_weighted_loss"),
        "final_val_coupling_weighted": final_metrics.get("val_source_coupling_weighted_loss"),
        "best_val_lag_focus": best_metrics.get("val_source_coupling_lag_focus_loss"),
        "best_val_joint_entropy": best_metrics.get("val_source_coupling_joint_entropy_loss"),
        "best_val_pair_likelihood": best_metrics.get("val_source_coupling_pair_likelihood_loss"),
        "best_val_lag_evidence": best_metrics.get("val_source_coupling_lag_evidence_loss"),
        "gate0": verdicts.get("gate0"),
        "gate1": verdicts.get("gate1"),
        "gate2": verdicts.get("gate2"),
        "gate3": verdicts.get("gate3"),
        "gate4": verdicts.get("gate4"),
        "promotion_verdict": gates.get("final_summary", {}).get("promotion_verdict") or gates.get("promotion_verdict"),
        "gate3_row_entropy_ratio": g3.get("row_entropy_ratio_to_logk"),
        "gate3_concentration_ratio": g3.get("concentration_ratio"),
        "gate3_joint_entropy_ratio": g3.get("joint_entropy_ratio"),
        "gate3_lag_entropy_ratio": g3.get("lag_entropy_ratio_to_logl"),
        "gate3_weighted_true_probability": predictability.get("model_weighted_true_token_probability"),
        "gate3_best_loso_gain": best_empirical_value(empirical, "loso_gain"),
        "gate3_best_mi_above_shuffle": best_empirical_value(empirical, "mi_above_shuffle"),
        **grad,
    }


def mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def relative_change(value: Any, baseline: Any) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    denominator = max(abs(float(baseline)), 1e-12)
    return (float(value) - float(baseline)) / denominator


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baselines = {
        row["seed"]: row
        for row in rows
        if row.get("condition") == "S0"
    }
    delta_keys = [
        "best_val_loss",
        "best_val_eeg_rec_loss",
        "best_val_fnirs_rec_loss",
        "gate3_best_mi_above_shuffle",
        "gate3_best_loso_gain",
    ]
    for row in rows:
        baseline = baselines.get(row.get("seed"))
        for key in delta_keys:
            row[f"{key}_delta_vs_s0"] = None if baseline is None else relative_change(row.get(key), baseline.get(key))


def aggregate_conditions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    numeric_keys = [
        "best_val_loss",
        "best_val_eeg_rec_loss",
        "best_val_fnirs_rec_loss",
        "best_val_coupling_weighted",
        "best_val_lag_focus",
        "best_val_joint_entropy",
        "best_val_pair_likelihood",
        "best_val_lag_evidence",
        "gate3_row_entropy_ratio",
        "gate3_concentration_ratio",
        "gate3_joint_entropy_ratio",
        "gate3_lag_entropy_ratio",
        "gate3_weighted_true_probability",
        "gate3_best_loso_gain",
        "gate3_best_mi_above_shuffle",
        "rec_coupling_cosine_min",
        "rec_coupling_cosine_mean",
        "rec_coupling_negative_fraction",
        "joint_entropy_rec_cosine_mean",
        "pair_likelihood_rec_cosine_mean",
        "coupling_grad_share_mean",
        "best_val_loss_delta_vs_s0",
        "best_val_eeg_rec_loss_delta_vs_s0",
        "best_val_fnirs_rec_loss_delta_vs_s0",
        "gate3_best_mi_above_shuffle_delta_vs_s0",
        "gate3_best_loso_gain_delta_vs_s0",
    ]
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        aggregate: dict[str, Any] = {"n": len(subset)}
        for key in numeric_keys:
            aggregate[f"{key}_mean"] = mean([row.get(key) for row in subset])
        aggregate["gate3_pass_count"] = sum(1 for row in subset if row.get("gate3") == "pass")
        aggregate["gate4_pass_count"] = sum(1 for row in subset if row.get("gate4") == "pass")
        by_condition[condition] = aggregate
    return by_condition


def decision_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    conditions = summary.get("conditions", {})
    promoted = []
    for condition, payload in conditions.items():
        if condition == "S0":
            continue
        mi_delta = payload.get("gate3_best_mi_above_shuffle_delta_vs_s0_mean")
        loso_delta = payload.get("gate3_best_loso_gain_delta_vs_s0_mean")
        rec_delta = payload.get("best_val_loss_delta_vs_s0_mean")
        visual_pass = isinstance(mi_delta, (int, float)) and float(mi_delta) >= 0.5
        utility_pass = isinstance(loso_delta, (int, float)) and float(loso_delta) > 0.0
        degradation_large = isinstance(rec_delta, (int, float)) and float(rec_delta) > 0.1
        if visual_pass and utility_pass and not degradation_large:
            promoted.append(condition)
    next_step = (
        "forced coupling produced a candidate pattern; inspect reconstruction and gradient conflict before any use"
        if promoted else
        "use this suite as a final coupling-only stress boundary; do not promote unless manual heatmaps overturn summary"
    )
    return {
        "status": "complete" if summary.get("runs") else "partial",
        "stress_candidates": promoted,
        "next_step": next_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    run_dirs = sorted((suite_dir / "tokenizer_interventions").glob("k64_s*_global_coupling_stress_seed*"))
    rows = [
        row_for_run(run_dir)
        for run_dir in run_dirs
        if (run_dir / "metrics.json").exists()
    ]
    add_baseline_deltas(rows)
    summary = {
        "schema_version": "tokenizer_coupling_stress_summary_v1",
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
        "# Tokenizer Coupling Stress Suite",
        "",
        f"Status: {decision['status']}",
        f"Next step: {decision['next_step']}",
        "",
        "Condition means:",
    ]
    for condition, payload in summary["conditions"].items():
        lines.append(
            f"- {condition}: best_val={payload.get('best_val_loss_mean')}, "
            f"val_delta_vs_S0={payload.get('best_val_loss_delta_vs_s0_mean')}, "
            f"MI={payload.get('gate3_best_mi_above_shuffle_mean')}, "
            f"MI_delta_vs_S0={payload.get('gate3_best_mi_above_shuffle_delta_vs_s0_mean')}, "
            f"LOSO={payload.get('gate3_best_loso_gain_mean')}, "
            f"row_entropy={payload.get('gate3_row_entropy_ratio_mean')}, "
            f"rec_coupling_cos={payload.get('rec_coupling_cosine_mean_mean')}, "
            f"conflict_frac={payload.get('rec_coupling_negative_fraction_mean')}"
        )
    (suite_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
