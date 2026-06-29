#!/usr/bin/env python
"""Aggregate lag-aware full cross-modal tokenizer suite results."""

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


def mean(values: list[Any]) -> float | None:
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def relative(value: Any, baseline: Any) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return (float(value) - float(baseline)) / max(abs(float(baseline)), 1e-12)


def condition(run_name: str) -> str:
    for part in run_name.split("_"):
        if part.startswith("z") and part[1:].isdigit():
            return part.upper()
    return "unknown"


def seed(run_name: str) -> str:
    return run_name.rsplit("seed", 1)[-1] if "seed" in run_name else "unknown"


def epoch_metrics(epoch: dict[str, Any]) -> dict[str, Any]:
    return {**epoch.get("loss_breakdown", {}), **epoch.get("metrics", {}), **epoch.get("scalars", {})}


def best_epoch(metrics: dict[str, Any]) -> dict[str, Any]:
    epochs = [item for item in metrics.get("epochs", []) if item.get("val_loss") is not None]
    return min(epochs, key=lambda item: float(epoch_metrics(item).get("val_primary_loss", item["val_loss"]))) if epochs else {}


def gate_metrics(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_dir / "analysis/gate_summary.json"
    if not path.exists():
        return {}, {}
    payload = read_json(path)
    gates = payload.get("splits", {}).get("test", {}).get("gates") or payload.get("gates", {})
    return gates.get("gate3", {}).get("metrics", {}), gates.get("gate4", {}).get("metrics", {})


def best_empirical(audit: dict[str, Any], key: str) -> tuple[float | None, int | None]:
    values = []
    for item in audit.get("per_lag", []):
        value = (
            item.get("leave_one_subject_out", {}).get("accuracy_gain")
            if key == "loso" else item.get("mi_above_shuffle")
        )
        if isinstance(value, (int, float)):
            values.append((float(value), int(item.get("lag", 0))))
    return max(values) if values else (None, None)


def external_audits(suite: Path, name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    info_path = suite / "information_drop_audit" / name / "summary.json"
    if info_path.exists():
        test = read_json(info_path).get("splits", {}).get("test", {})
        summary = test.get("summary", {})
        levels = test.get("levels", {})
        result.update({
            "info_hard_cmi": number(summary.get("hard_token_nuisance_adjusted_gain")),
            "info_soft_to_hard_drop": number(summary.get("soft_to_hard_drop")),
            "info_hard_loso_ridge": number(levels.get("hard_token_onehot", {}).get("loso_ridge", {}).get("mean")),
        })
    downstream_path = suite / "downstream_probe" / name / "summary.json"
    if downstream_path.exists():
        values = []
        for item in read_json(downstream_path).get("rows", []):
            if item.get("split") != "test" or item.get("feature") not in {"bot", "bot_bigram"}:
                continue
            if item.get("task") in {"nback_load_0_vs_2_vs_3", "motor_lmi_vs_rmi"}:
                value = number(item.get("balanced_accuracy"))
                if value is not None:
                    values.append(value)
        result["downstream_best_fine_balanced_accuracy"] = max(values) if values else None
    return result


def row_for_run(suite: Path, run_dir: Path, stage: str) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    best = best_epoch(metrics)
    best_values = epoch_metrics(best)
    final = metrics.get("epochs", [])[-1] if metrics.get("epochs") else {}
    final_values = epoch_metrics(final)
    g3, g4 = gate_metrics(run_dir)
    mi, mi_lag = best_empirical(g3.get("lag_balanced_empirical_audit", {}), "mi")
    loso, loso_lag = best_empirical(g3.get("lag_balanced_empirical_audit", {}), "loso")
    source_leak = g4.get("source_subject_leakage_probe", {})
    source_task = g4.get("source_task_signal_probe", {})
    row = {
        "stage": stage,
        "run": run_dir.name,
        "condition": condition(run_dir.name),
        "seed": seed(run_dir.name),
        "best_epoch": best.get("epoch"),
        "best_val_primary_loss": number(best_values.get("val_primary_loss")),
        "best_val_loss": number(best.get("val_loss")),
        "best_val_eeg_rec_loss": number(best_values.get("val_eeg_rec_loss")),
        "best_val_fnirs_rec_loss": number(best_values.get("val_fnirs_rec_loss")),
        "best_val_eeg_source_perplexity": number(best_values.get("val_eeg_source_perplexity")),
        "best_val_fnirs_source_perplexity": number(best_values.get("val_fnirs_source_perplexity")),
        "best_val_temporal_nce": number(best_values.get("val_cross_modal_temporal_nce_loss")),
        "best_val_masked_latent": number(best_values.get("val_cross_modal_masked_latent_loss")),
        "best_val_soft_code": number(best_values.get("val_cross_modal_soft_code_distillation_loss")),
        "best_val_physiologic_lag_mass": number(best_values.get("val_cross_modal_fusion_physiologic_lag_mass")),
        "best_val_fusion_lag_entropy": number(best_values.get("val_cross_modal_fusion_lag_entropy")),
        "best_val_masked_pairing_gain": number(best_values.get("val_cross_modal_masked_pairing_gain")),
        "best_alignment_gradient_ratio": number(best_values.get("alignment_gradient_ratio")),
        "best_alignment_gradient_scale": number(best_values.get("alignment_gradient_scale")),
        "forced_hard": number(best_values.get("val_forced_hard_quantization")),
        "final_epoch": final.get("epoch"),
        "final_val_primary_loss": number(final_values.get("val_primary_loss")),
        "gate3_best_mi_above_shuffle": mi,
        "gate3_best_mi_lag": mi_lag,
        "gate3_best_loso_gain": loso,
        "gate3_best_loso_lag": loso_lag,
        "gate3_row_entropy_ratio": number(g3.get("row_entropy_ratio_to_logk")),
        "source_subject_leakage": number(source_leak.get("normalized_lift")),
        "source_task_signal": number(source_task.get("normalized_lift")),
    }
    row.update(external_audits(suite, run_dir.name))
    return row


def collect_rows(suite: Path) -> list[dict[str, Any]]:
    rows = []
    for stage, leaf in (("confirmatory", "tokenizer_interventions"), ("screening", "screening"), ("smoke", "smoke")):
        for run_dir in sorted((suite / leaf).glob("*z*_full_alignment_seed*")):
            if (run_dir / "metrics.json").exists():
                rows.append(row_for_run(suite, run_dir, stage))
    return rows


def add_deltas(rows: list[dict[str, Any]]) -> None:
    baselines = {(row["stage"], row["seed"]): row for row in rows if row["condition"] == "Z0"}
    keys = (
        "best_val_primary_loss", "best_val_eeg_rec_loss", "best_val_fnirs_rec_loss",
        "best_val_eeg_source_perplexity", "best_val_fnirs_source_perplexity",
        "gate3_best_mi_above_shuffle", "gate3_best_loso_gain", "info_hard_cmi", "info_hard_loso_ridge",
    )
    for row in rows:
        baseline = baselines.get((row["stage"], row["seed"]))
        for key in keys:
            row[f"{key}_delta_vs_z0"] = relative(row.get(key), baseline.get(key)) if baseline else None
        value = row.get("downstream_best_fine_balanced_accuracy")
        base_value = baseline.get("downstream_best_fine_balanced_accuracy") if baseline else None
        row["downstream_fine_abs_delta_vs_z0"] = (
            float(value) - float(base_value)
            if isinstance(value, (int, float)) and isinstance(base_value, (int, float)) else None
        )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected = [row for row in rows if row["stage"] == "confirmatory"]
    if not selected:
        selected = [row for row in rows if row["stage"] == "screening"] or rows
    keys = {key for row in selected for key, value in row.items() if isinstance(value, (int, float))}
    result = {}
    for key in sorted({row["condition"] for row in selected}):
        subset = [row for row in selected if row["condition"] == key]
        result[key] = {"n": len(subset), **{f"{name}_mean": mean([row.get(name) for row in subset]) for name in keys}}
    return result


def decide(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    diagnostics = {}
    promoted = []
    for key, payload in conditions.items():
        if key == "Z0":
            continue
        info = max(payload.get("info_hard_cmi_delta_vs_z0_mean") or 0, payload.get("info_hard_loso_ridge_delta_vs_z0_mean") or 0) >= 0.25
        fine = (payload.get("downstream_fine_abs_delta_vs_z0_mean") or 0) >= 0.03
        coupling = (payload.get("gate3_best_mi_above_shuffle_delta_vs_z0_mean") or 0) >= 0.20
        heldout = (payload.get("gate3_best_loso_gain_mean") or 0) > 0
        health = (
            (payload.get("best_val_primary_loss_delta_vs_z0_mean") or 0) <= 0.05 and
            (payload.get("best_val_eeg_source_perplexity_delta_vs_z0_mean") or 0) >= -0.20 and
            (payload.get("best_val_fnirs_source_perplexity_delta_vs_z0_mean") or 0) >= -0.20
        )
        gradient = payload.get("best_alignment_gradient_ratio_mean") is None or 0.15 <= payload["best_alignment_gradient_ratio_mean"] <= 0.40
        diagnostics[key] = {
            "information_retention": info, "fine_task": fine, "coupling": coupling,
            "heldout_utility": heldout, "token_health": health, "gradient_strength": gradient,
        }
        if info and fine and coupling and heldout and health and gradient:
            promoted.append(key)
    return {
        "status": "complete" if conditions else "pending",
        "promoted_conditions": promoted,
        "gate_diagnostics": diagnostics,
        "next_step": (
            "run Z7 coupling confirmation and 80-epoch extension" if promoted
            else "do not promote until confirmatory information/downstream audits pass"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    suite = Path(args.suite_dir).resolve()
    rows = collect_rows(suite)
    add_deltas(rows)
    conditions = aggregate(rows)
    decision = decide(conditions)
    summary = {
        "schema_version": "tokenizer_full_cross_modal_alignment_summary_v1",
        "suite_dir": str(suite),
        "categories": {
            "architecture": [
                "best_val_physiologic_lag_mass", "best_val_fusion_lag_entropy", "best_val_masked_pairing_gain"
            ],
            "information_retention": ["info_hard_cmi", "info_hard_loso_ridge", "info_soft_to_hard_drop"],
            "fine_task": ["downstream_best_fine_balanced_accuracy"],
            "coupling": ["gate3_best_mi_above_shuffle", "gate3_best_loso_gain"],
            "gradient": ["best_alignment_gradient_ratio", "best_alignment_gradient_scale"],
            "token_health": ["best_val_primary_loss", "best_val_eeg_source_perplexity", "best_val_fnirs_source_perplexity"],
            "leakage": ["source_subject_leakage", "source_task_signal"],
        },
        "conditions": conditions,
        "runs": rows,
    }
    fields = sorted({key for row in rows for key in row})
    if fields:
        with (suite / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (suite / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (suite / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Full Cross-Modal Alignment Suite", "", f"Status: {decision['status']}", f"Next: {decision['next_step']}", ""]
    for key, payload in conditions.items():
        lines.append(
            f"- {key}: primary={payload.get('best_val_primary_loss_mean')}, "
            f"MI={payload.get('gate3_best_mi_above_shuffle_mean')}, "
            f"LOSO={payload.get('gate3_best_loso_gain_mean')}, "
            f"fine={payload.get('downstream_best_fine_balanced_accuracy_mean')}, "
            f"grad_ratio={payload.get('best_alignment_gradient_ratio_mean')}"
        )
    (suite / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
