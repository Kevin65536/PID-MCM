#!/usr/bin/env python
"""Finalize causal cross-modal exchange tokenizer suite summaries."""

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
        if part.startswith("x") and part[1:].isdigit():
            return part.upper()
    return "unknown"


def seed_key(run_name: str) -> str:
    return run_name.rsplit("seed", 1)[-1] if "seed" in run_name else "unknown"


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def mean(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(nums) / len(nums) if nums else None


def relative_change(value: Any, baseline: Any) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return (float(value) - float(baseline)) / max(abs(float(baseline)), 1e-12)


def best_validation_epoch(metrics: dict[str, Any]) -> dict[str, Any]:
    val_epochs = [epoch for epoch in metrics.get("epochs", []) if epoch.get("val_loss") is not None]
    if not val_epochs:
        return {}
    return min(val_epochs, key=lambda epoch: float(epoch["val_loss"]))


def final_validation_epoch(metrics: dict[str, Any]) -> dict[str, Any]:
    val_epochs = [epoch for epoch in metrics.get("epochs", []) if epoch.get("val_loss") is not None]
    return val_epochs[-1] if val_epochs else {}


def epoch_metrics(epoch: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for bucket in ("metrics", "scalars", "loss_breakdown"):
        payload = epoch.get(bucket, {})
        if isinstance(payload, dict):
            result.update(payload)
    return result


def split_gate(payload: dict[str, Any], split: str, gate: str) -> dict[str, Any]:
    split_payload = payload.get("splits", {}).get(split, {})
    if isinstance(split_payload.get("gates"), dict):
        return split_payload["gates"].get(gate, {})
    return payload.get("gates", {}).get(gate, {})


def gate_payload(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis/gate_summary.json"
    return read_json(path) if path.exists() else {}


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


def info_summary(suite_dir: Path, run_name: str) -> dict[str, Any]:
    path = suite_dir / "information_drop_audit" / run_name / "summary.json"
    if not path.exists():
        return {}
    test = read_json(path).get("splits", {}).get("test", {})
    summary = test.get("summary", {})
    levels = test.get("levels", {})
    continuous = levels.get("continuous_source_latent", {})
    hard = levels.get("hard_token_onehot", {})
    return {
        "info_hard_cmi": numeric(summary.get("hard_token_nuisance_adjusted_gain")),
        "info_soft_to_hard_drop": numeric(summary.get("soft_to_hard_drop")),
        "info_continuous_to_soft_retention": numeric(summary.get("continuous_to_soft_retention")),
        "info_continuous_cmi": numeric(continuous.get("conditional_mi_gaussian")),
        "info_hard_loso_ridge": numeric(hard.get("loso_ridge", {}).get("mean")),
        "info_hard_loso_cca": numeric(hard.get("loso_cca_correlation")),
    }


def downstream_summary(suite_dir: Path, run_name: str) -> dict[str, Any]:
    path = suite_dir / "downstream_probe" / run_name / "summary.json"
    if not path.exists():
        return {}
    result: dict[str, Any] = {}
    for row in read_json(path).get("rows", []):
        if row.get("split") != "test" or row.get("feature") not in {"bot", "bot_bigram"}:
            continue
        task = row.get("task")
        feature = row.get("feature")
        if task in {"nback_load_0_vs_2_vs_3", "motor_lmi_vs_rmi", "mental_arithmetic_bl_vs_ma", "nback_vs_wg"}:
            result[f"downstream_{task}_{feature}_balanced_accuracy"] = numeric(row.get("balanced_accuracy"))
            result[f"downstream_{task}_{feature}_macro_f1"] = numeric(row.get("macro_f1"))
    fine_values = [
        value for key, value in result.items()
        if key.endswith("_balanced_accuracy")
        and ("nback_load_0_vs_2_vs_3" in key or "motor_lmi_vs_rmi" in key or "mental_arithmetic_bl_vs_ma" in key)
    ]
    result["downstream_best_fine_balanced_accuracy"] = max(fine_values) if fine_values else None
    return result


def local_coupling_summary(suite_dir: Path, run_name: str) -> dict[str, Any]:
    path = suite_dir / "local_coupling_audit" / run_name / "summary.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    scopes = payload.get("scopes", {})
    best_scope = payload.get("decision_features", {}).get("best_scope")
    best_payload = scopes.get(best_scope, {}) if best_scope else {}
    return {
        "local_best_scope": best_scope,
        "local_best_ci_low": numeric(best_payload.get("best_ci_low")),
        "local_mean_gain": numeric(best_payload.get("mean_gain")),
        "local_any_group_ci_low_positive": payload.get("decision_features", {}).get("any_group_ci_low_positive"),
    }


def gradient_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    align_names = (
        "source_coupling_pair_likelihood_loss",
        "source_coupling_context_residual_loss",
    )
    rec_names = ("eeg_rec_loss", "fnirs_rec_loss")
    cosines: list[float] = []
    shares: list[float] = []
    exchange_shares: list[float] = []
    for epoch in metrics.get("epochs", []):
        for key, value in epoch_metrics(epoch).items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith("grad_share_") and any(name in key for name in align_names):
                shares.append(float(value))
            if key.startswith("grad_group_share_") and "__cross_modal_exchange" in key:
                exchange_shares.append(float(value))
            if not key.startswith("grad_cosine_"):
                continue
            if any(rec in key for rec in rec_names) and any(name in key for name in align_names):
                cosines.append(float(value))
    return {
        "grad_alignment_rec_cosine_min": min(cosines) if cosines else None,
        "grad_alignment_rec_cosine_mean": mean(cosines),
        "grad_alignment_rec_negative_fraction": (
            sum(1 for value in cosines if value < 0.0) / len(cosines)
            if cosines else None
        ),
        "grad_alignment_share_mean": mean(shares),
        "grad_cross_modal_exchange_group_share_mean": mean(exchange_shares),
    }


def row_for_run(suite_dir: Path, run_dir: Path, stage: str) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    gates = gate_payload(run_dir)
    g3 = split_gate(gates, "test", "gate3").get("metrics", {})
    empirical = g3.get("lag_balanced_empirical_audit", {})
    predictability = g3.get("cross_modal_token_predictability", {})
    leakage = split_gate(gates, "test", "gate4").get("metrics", {})
    source_leak = (leakage.get("source_subject_leakage_probe") or {}).get("normalized_lift")
    task_lift = (leakage.get("source_task_signal_probe") or {}).get("normalized_lift")
    best = best_validation_epoch(metrics)
    final = final_validation_epoch(metrics)
    best_metrics = epoch_metrics(best)
    final_metrics = epoch_metrics(final)
    row = {
        "stage": stage,
        "run": run_dir.name,
        "condition": condition_key(run_dir.name),
        "seed": seed_key(run_dir.name),
        "best_epoch": best.get("epoch"),
        "best_val_loss": best.get("val_loss"),
        "best_val_primary_loss": best_metrics.get("val_primary_loss"),
        "best_val_eeg_rec_loss": best_metrics.get("val_eeg_rec_loss"),
        "best_val_fnirs_rec_loss": best_metrics.get("val_fnirs_rec_loss"),
        "best_val_eeg_source_perplexity": best_metrics.get("val_eeg_source_perplexity"),
        "best_val_fnirs_source_perplexity": best_metrics.get("val_fnirs_source_perplexity"),
        "best_val_cross_modal_exchange_update_norm": best_metrics.get("val_cross_modal_exchange_update_norm"),
        "best_val_cross_modal_exchange_lag_entropy": best_metrics.get("val_cross_modal_exchange_lag_entropy"),
        "best_val_context_residual_loss": best_metrics.get("val_source_coupling_context_residual_loss"),
        "best_val_coupling_weighted": best_metrics.get("val_source_coupling_weighted_loss"),
        "final_epoch": final.get("epoch"),
        "final_val_loss": final.get("val_loss"),
        "final_val_primary_loss": final_metrics.get("val_primary_loss"),
        "gate3_row_entropy_ratio": g3.get("row_entropy_ratio_to_logk"),
        "gate3_weighted_true_probability": predictability.get("model_weighted_true_token_probability"),
        "gate3_best_mi_above_shuffle": best_empirical_value(empirical, "mi_above_shuffle"),
        "gate3_best_loso_gain": best_empirical_value(empirical, "loso_gain"),
        "source_subject_leakage_normalized_lift": numeric(source_leak),
        "source_task_signal_normalized_lift": numeric(task_lift),
    }
    row.update(gradient_summary(metrics))
    row.update(info_summary(suite_dir, run_dir.name))
    row.update(downstream_summary(suite_dir, run_dir.name))
    row.update(local_coupling_summary(suite_dir, run_dir.name))
    return row


def collect_rows(suite_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for stage, leaf in (("formal", "tokenizer_interventions"), ("smoke", "smoke")):
        for run_dir in sorted((suite_dir / leaf).glob("*causal_exchange_seed*")):
            if (run_dir / "metrics.json").exists():
                rows.append(row_for_run(suite_dir, run_dir, stage))
    return rows


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = {(row["stage"], row["seed"]): row for row in rows if row.get("condition") == "X0"}
    keys = [
        "best_val_primary_loss",
        "best_val_eeg_rec_loss",
        "best_val_fnirs_rec_loss",
        "best_val_eeg_source_perplexity",
        "best_val_fnirs_source_perplexity",
        "gate3_best_mi_above_shuffle",
        "gate3_best_loso_gain",
        "info_hard_cmi",
        "info_hard_loso_ridge",
        "downstream_best_fine_balanced_accuracy",
        "source_subject_leakage_normalized_lift",
        "source_task_signal_normalized_lift",
    ]
    for row in rows:
        base = baseline.get((row.get("stage"), row.get("seed")))
        for key in keys:
            row[f"{key}_delta_vs_x0"] = None if base is None else relative_change(row.get(key), base.get(key))
        if base is None:
            row["downstream_best_fine_balanced_accuracy_abs_delta_vs_x0"] = None
            row["source_subject_leakage_abs_delta_vs_x0"] = None
            continue
        value = row.get("downstream_best_fine_balanced_accuracy")
        base_value = base.get("downstream_best_fine_balanced_accuracy")
        row["downstream_best_fine_balanced_accuracy_abs_delta_vs_x0"] = (
            None if not isinstance(value, (int, float)) or not isinstance(base_value, (int, float))
            else float(value) - float(base_value)
        )
        leak = row.get("source_subject_leakage_normalized_lift")
        base_leak = base.get("source_subject_leakage_normalized_lift")
        row["source_subject_leakage_abs_delta_vs_x0"] = (
            None if not isinstance(leak, (int, float)) or not isinstance(base_leak, (int, float))
            else float(leak) - float(base_leak)
        )


def aggregate_conditions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    formal_rows = [row for row in rows if row.get("stage") == "formal"]
    if not formal_rows:
        formal_rows = rows
    keys = sorted({key for row in formal_rows for key, value in row.items() if isinstance(value, (int, float))})
    conditions: dict[str, dict[str, Any]] = {}
    for condition in sorted({row["condition"] for row in formal_rows}):
        subset = [row for row in formal_rows if row["condition"] == condition]
        payload: dict[str, Any] = {"n": len(subset)}
        for key in keys:
            payload[f"{key}_mean"] = mean([row.get(key) for row in subset])
        conditions[condition] = payload
    return conditions


def decision_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    conditions = summary.get("conditions", {})
    if not conditions:
        return {"status": "pending", "promoted_conditions": [], "next_step": "no completed runs found"}
    promoted = []
    diagnostics = {}
    for condition, payload in conditions.items():
        if condition == "X0":
            continue
        info_pass = (
            (payload.get("info_hard_cmi_delta_vs_x0_mean") or 0.0) >= 0.25
            or (payload.get("info_hard_loso_ridge_delta_vs_x0_mean") or 0.0) >= 0.25
        )
        fine_pass = (payload.get("downstream_best_fine_balanced_accuracy_abs_delta_vs_x0_mean") or 0.0) >= 0.03
        coupling_pass = (payload.get("gate3_best_mi_above_shuffle_delta_vs_x0_mean") or 0.0) >= 0.20
        heldout_pass = (payload.get("gate3_best_loso_gain_mean") or 0.0) > 0.0
        gradient_pass = (
            payload.get("grad_alignment_rec_cosine_mean_mean") is None
            or float(payload.get("grad_alignment_rec_cosine_mean_mean")) >= -0.2
        )
        token_health_pass = (
            (payload.get("best_val_primary_loss_delta_vs_x0_mean") is None or payload.get("best_val_primary_loss_delta_vs_x0_mean") <= 0.05)
            and (payload.get("best_val_eeg_source_perplexity_delta_vs_x0_mean") is None or payload.get("best_val_eeg_source_perplexity_delta_vs_x0_mean") >= -0.20)
            and (payload.get("best_val_fnirs_source_perplexity_delta_vs_x0_mean") is None or payload.get("best_val_fnirs_source_perplexity_delta_vs_x0_mean") >= -0.20)
        )
        leakage_pass = (payload.get("source_subject_leakage_abs_delta_vs_x0_mean") or 0.0) <= 0.05
        diagnostics[condition] = {
            "information_retention": info_pass,
            "fine_task_retention": fine_pass,
            "coupling_visualization": coupling_pass,
            "heldout_utility": heldout_pass,
            "gradient_compatibility": gradient_pass,
            "token_health": token_health_pass,
            "leakage_guard": leakage_pass,
        }
        if info_pass and fine_pass and gradient_pass and token_health_pass and leakage_pass:
            promoted.append(condition)
    if any(condition in promoted for condition in ("X3", "X4")):
        next_step = "promote causal exchange with coupling prior; extend candidate to 80 epochs"
    elif "X2" in promoted:
        next_step = "promote causal exchange; keep coupling as audit until it passes"
    elif promoted:
        next_step = "inspect promoted diagnostic condition manually before stronger architecture"
    else:
        next_step = "do not promote causal low-rank exchange yet; inspect formal failures before full cross-attention"
    return {
        "status": "complete",
        "promoted_conditions": promoted,
        "gate_diagnostics": diagnostics,
        "next_step": next_step,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    rows = collect_rows(suite_dir)
    add_baseline_deltas(rows)
    summary = {
        "schema_version": "tokenizer_cross_modal_exchange_summary_v1",
        "suite_dir": str(suite_dir),
        "categories": {
            "information_retention": ["info_hard_cmi", "info_hard_loso_ridge", "info_soft_to_hard_drop"],
            "fine_task_probe": ["downstream_best_fine_balanced_accuracy", "downstream_motor_lmi_vs_rmi_bot_balanced_accuracy"],
            "gate3_coupling": ["gate3_best_mi_above_shuffle", "gate3_best_loso_gain", "gate3_row_entropy_ratio"],
            "gradient_conflict": ["grad_alignment_rec_cosine_mean", "grad_alignment_share_mean", "grad_cross_modal_exchange_group_share_mean"],
            "token_health": ["best_val_primary_loss", "best_val_eeg_source_perplexity", "best_val_fnirs_source_perplexity"],
            "decision": ["promoted_conditions", "next_step"],
        },
        "conditions": aggregate_conditions(rows),
        "runs": rows,
    }
    decision = decision_from_summary(summary)
    write_csv(suite_dir / "summary.csv", rows)
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Tokenizer Cross-Modal Exchange Suite",
        "",
        f"Status: {decision['status']}",
        f"Next step: {decision['next_step']}",
        "",
        "Condition means:",
    ]
    for condition, payload in summary["conditions"].items():
        lines.append(
            f"- {condition}: primary={payload.get('best_val_primary_loss_mean')}, "
            f"primary_delta={payload.get('best_val_primary_loss_delta_vs_x0_mean')}, "
            f"exchange_norm={payload.get('best_val_cross_modal_exchange_update_norm_mean')}, "
            f"info_cmi={payload.get('info_hard_cmi_mean')}, "
            f"fine_acc={payload.get('downstream_best_fine_balanced_accuracy_mean')}, "
            f"MI={payload.get('gate3_best_mi_above_shuffle_mean')}, "
            f"MI_delta={payload.get('gate3_best_mi_above_shuffle_delta_vs_x0_mean')}, "
            f"grad_cos={payload.get('grad_alignment_rec_cosine_mean_mean')}"
        )
    (suite_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
