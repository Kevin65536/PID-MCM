#!/usr/bin/env python3
"""Aggregate tokenizer exploration suites into a cross-run meta-analysis.

The analysis intentionally treats this as exploratory evidence. The included
suites differ in objectives, epoch budgets, checkpoint policies, and data
subsets, so the output emphasizes effect sizes, rankings, and failure modes
instead of formal hypothesis tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOTS = (
    "coupling_design_audit",
    "tokenizer_context_coupling",
    "tokenizer_coupling_capacity",
    "tokenizer_coupling_discovery",
    "tokenizer_coupling_stress",
    "tokenizer_next_stage",
)

NUMERIC_FIELDS = (
    "best_val_loss",
    "best_val_primary_loss",
    "final_val_loss",
    "final_val_primary_loss",
    "best_val_eeg_rec_loss",
    "best_val_fnirs_rec_loss",
    "final_val_eeg_rec_loss",
    "final_val_fnirs_rec_loss",
    "best_val_observation_loss",
    "final_val_observation_loss",
    "best_val_source_target_loss",
    "final_val_source_target_loss",
    "best_val_source_target_corr_loss",
    "final_val_source_target_corr_loss",
    "best_val_coupling_weighted",
    "final_val_coupling_weighted",
    "best_val_source_coupling_loss",
    "final_val_source_coupling_loss",
    "best_val_eeg_source_perplexity",
    "best_val_fnirs_source_perplexity",
    "final_val_eeg_source_perplexity",
    "final_val_fnirs_source_perplexity",
    "best_val_source_code_overlap",
    "final_val_source_code_overlap",
    "gate3_best_mi_above_shuffle",
    "gate3_mean_mi_above_shuffle",
    "gate3_best_loso_gain",
    "gate3_mean_loso_gain",
    "gate3_row_entropy_ratio",
    "gate3_joint_entropy_ratio",
    "gate3_weighted_true_probability",
    "gate3_cross_modal_accuracy",
    "gate3_cross_modal_chance",
)

LOWER_IS_BETTER = {
    "best_val_loss",
    "best_val_primary_loss",
    "final_val_loss",
    "final_val_primary_loss",
    "best_val_eeg_rec_loss",
    "best_val_fnirs_rec_loss",
    "final_val_eeg_rec_loss",
    "final_val_fnirs_rec_loss",
    "best_val_observation_loss",
    "final_val_observation_loss",
    "best_val_source_target_loss",
    "final_val_source_target_loss",
    "best_val_source_target_corr_loss",
    "final_val_source_target_corr_loss",
    "best_val_coupling_weighted",
    "final_val_coupling_weighted",
    "best_val_source_coupling_loss",
    "final_val_source_coupling_loss",
    "gate3_row_entropy_ratio",
    "gate3_joint_entropy_ratio",
}

HIGHER_IS_BETTER = {
    "best_val_eeg_source_perplexity",
    "best_val_fnirs_source_perplexity",
    "final_val_eeg_source_perplexity",
    "final_val_fnirs_source_perplexity",
    "gate3_best_mi_above_shuffle",
    "gate3_mean_mi_above_shuffle",
    "gate3_best_loso_gain",
    "gate3_mean_loso_gain",
    "gate3_weighted_true_probability",
    "gate3_cross_modal_accuracy",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
    return None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(xs) if xs else None


def stdev(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.stdev(xs) if len(xs) > 1 else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 10:
            return f"{value:.3f}"
        return f"{value:.{digits}f}"
    return str(value)


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for path in (current, *current.parents):
        if (path / "experiments" / "runs").exists():
            return path
    raise RuntimeError(f"Could not locate repository root from {start}")


def suite_family(path: Path) -> str:
    parts = path.parts
    try:
        idx = parts.index("runs")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def suite_id(path: Path) -> str:
    parts = path.parts
    try:
        idx = parts.index("runs")
        return parts[idx + 2]
    except (ValueError, IndexError):
        return "unknown"


def run_group(path: Path) -> str:
    parts = path.parts
    try:
        idx = parts.index(suite_id(path))
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ""


def parse_run_metadata(run_name: str, family: str, sid: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    seed = re.search(r"seed(\d+)", run_name)
    if seed:
        meta["seed"] = int(seed.group(1))
    k = re.search(r"\bk(\d+)|_k(\d+)", run_name)
    if k:
        meta["k"] = int(next(g for g in k.groups() if g))
    dim = re.search(r"dim(\d+)|baseline_dim(\d+)", run_name)
    if dim:
        meta["dim"] = int(next(g for g in dim.groups() if g))

    condition = None
    if "capacity" in family:
        condition = f"K{meta.get('k', 'unknown')}"
    elif "next_stage" in family:
        if "cognitive" in run_name:
            condition = "cognitive_nback_wg"
        elif "dim" in run_name:
            condition = f"dim{meta.get('dim', 'unknown')}"
    else:
        for pattern, prefix in (
            (r"(?:^|_)t(\d+)(?:_|$)", "T"),
            (r"(?:^|_)c(\d+)(?:_|$)", "C"),
            (r"(?:^|_)s(\d+)(?:_|$)", "S"),
        ):
            match = re.search(pattern, run_name)
            if match:
                condition = f"{prefix}{match.group(1)}"
                break
    meta["condition"] = condition or ""

    meta["cognitive_only"] = "cognitive" in run_name or "nback_wg" in run_name
    meta["superseded"] = (
        family == "tokenizer_context_coupling"
        and sid.startswith("20260622_")
    )
    return meta


def metric_from_epoch(epoch: Dict[str, Any], key: str) -> Optional[float]:
    if key in epoch:
        return scalar(epoch.get(key))
    metrics = epoch.get("metrics", {})
    if isinstance(metrics, dict):
        return scalar(metrics.get(key))
    return None


def best_epoch(epochs: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    best: Optional[Dict[str, Any]] = None
    best_value: Optional[float] = None
    for epoch in epochs:
        value = metric_from_epoch(epoch, "val_loss")
        if value is None:
            continue
        if best_value is None or value < best_value:
            best = epoch
            best_value = value
    return best, best_value


def first_epoch_reaching_fraction(
    epochs: Sequence[Dict[str, Any]],
    metric: str,
    fraction: float,
) -> Optional[int]:
    values = [(int(e.get("epoch", idx + 1)), metric_from_epoch(e, metric)) for idx, e in enumerate(epochs)]
    values = [(e, v) for e, v in values if v is not None]
    if len(values) < 2:
        return None
    start = values[0][1]
    best = min(v for _, v in values)
    improvement = start - best
    if improvement <= 1e-12:
        return None
    threshold = start - fraction * improvement
    for epoch, value in values:
        if value <= threshold:
            return epoch
    return None


def slope_between(
    epochs: Sequence[Dict[str, Any]],
    metric: str,
    first_idx: int,
    last_idx: int,
) -> Optional[float]:
    if not epochs:
        return None
    n = len(epochs)
    i = max(0, min(first_idx, n - 1))
    j = max(0, min(last_idx, n - 1))
    if i == j:
        return None
    vi = metric_from_epoch(epochs[i], metric)
    vj = metric_from_epoch(epochs[j], metric)
    if vi is None or vj is None:
        return None
    return (vj - vi) / (j - i)


def metric_with_fallback(epoch: Dict[str, Any], primary: str, fallback: str) -> Optional[float]:
    value = metric_from_epoch(epoch, primary)
    return value if value is not None else metric_from_epoch(epoch, fallback)


def first_epoch_reaching_fraction_with_fallback(
    epochs: Sequence[Dict[str, Any]],
    primary: str,
    fallback: str,
    fraction: float,
) -> Optional[int]:
    values = [
        (int(e.get("epoch", idx + 1)), metric_with_fallback(e, primary, fallback))
        for idx, e in enumerate(epochs)
    ]
    values = [(e, v) for e, v in values if v is not None]
    if len(values) < 2:
        return None
    start = values[0][1]
    best = min(v for _, v in values)
    improvement = start - best
    if improvement <= 1e-12:
        return None
    threshold = start - fraction * improvement
    for epoch, value in values:
        if value <= threshold:
            return epoch
    return None


def slope_between_with_fallback(
    epochs: Sequence[Dict[str, Any]],
    primary: str,
    fallback: str,
    first_idx: int,
    last_idx: int,
) -> Optional[float]:
    if not epochs:
        return None
    n = len(epochs)
    i = max(0, min(first_idx, n - 1))
    j = max(0, min(last_idx, n - 1))
    if i == j:
        return None
    vi = metric_with_fallback(epochs[i], primary, fallback)
    vj = metric_with_fallback(epochs[j], primary, fallback)
    if vi is None or vj is None:
        return None
    return (vj - vi) / (j - i)


def extract_gate_metrics(run_dir: Path) -> Dict[str, Any]:
    gate_path = run_dir / "analysis" / "gate_summary.json"
    if not gate_path.exists():
        return {}
    payload = read_json(gate_path)
    out: Dict[str, Any] = {
        "promotion_verdict": payload.get("promotion_verdict", ""),
    }
    gates = payload.get("gates", {})
    if isinstance(gates, dict):
        for name in ("gate0", "gate1", "gate2", "gate3", "gate4"):
            gate = gates.get(name, {})
            if isinstance(gate, dict):
                out[name] = gate.get("status", "")
    gate3 = gates.get("gate3", {}) if isinstance(gates, dict) else {}
    metrics = gate3.get("metrics", {}) if isinstance(gate3, dict) else {}
    if not isinstance(metrics, dict):
        return out
    out["gate3_row_entropy_ratio"] = scalar(metrics.get("row_entropy_ratio_to_logk"))
    out["gate3_joint_entropy_ratio"] = scalar(metrics.get("joint_entropy_ratio"))
    pred = metrics.get("cross_modal_token_predictability", {})
    if isinstance(pred, dict):
        out["gate3_cross_modal_accuracy"] = scalar(pred.get("accuracy"))
        out["gate3_cross_modal_chance"] = scalar(pred.get("chance_accuracy"))
        out["gate3_weighted_true_probability"] = scalar(
            pred.get("model_weighted_true_token_probability")
        )
    audit = metrics.get("lag_balanced_empirical_audit", {})
    per_lag = audit.get("per_lag", []) if isinstance(audit, dict) else []
    mi_values: List[float] = []
    loso_values: List[float] = []
    best_lag: Optional[int] = None
    best_mi: Optional[float] = None
    if isinstance(per_lag, list):
        for entry in per_lag:
            if not isinstance(entry, dict):
                continue
            mi = scalar(entry.get("mi_above_shuffle"))
            if mi is not None:
                mi_values.append(mi)
                if best_mi is None or mi > best_mi:
                    best_mi = mi
                    best_lag = int(entry.get("lag", -1))
            loso = entry.get("leave_one_subject_out", {})
            if isinstance(loso, dict):
                gain = scalar(loso.get("accuracy_gain"))
                if gain is not None:
                    loso_values.append(gain)
    out["gate3_best_mi_above_shuffle"] = best_mi
    out["gate3_best_mi_lag"] = best_lag
    out["gate3_mean_mi_above_shuffle"] = mean(mi_values)
    out["gate3_best_loso_gain"] = max(loso_values) if loso_values else None
    out["gate3_mean_loso_gain"] = mean(loso_values)
    return out


def collect_runs(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    runs_root = repo / "experiments" / "runs"
    metric_paths: List[Path] = []
    for root in ROOTS:
        base = runs_root / root
        if base.exists():
            metric_paths.extend(base.glob("**/metrics.json"))
    formal_paths = sorted(p for p in metric_paths if "smoke" not in p.parts)

    rows: List[Dict[str, Any]] = []
    dynamics: List[Dict[str, Any]] = []
    for metrics_path in formal_paths:
        run_dir = metrics_path.parent
        run_name = run_dir.name
        family = suite_family(metrics_path)
        sid = suite_id(metrics_path)
        group = run_group(metrics_path)
        try:
            data = read_json(metrics_path)
        except Exception as exc:
            rows.append({
                "suite_family": family,
                "suite_id": sid,
                "run_group": group,
                "run": run_name,
                "metrics_path": str(metrics_path),
                "read_error": str(exc),
            })
            continue

        epochs = data.get("epochs", [])
        if not isinstance(epochs, list):
            epochs = []
        best_ep, best_loss = best_epoch(epochs)
        final_ep = epochs[-1] if epochs else {}
        meta = parse_run_metadata(run_name, family, sid)
        row: Dict[str, Any] = {
            "suite_family": family,
            "suite_id": sid,
            "run_group": group,
            "run": run_name,
            "metrics_path": str(metrics_path),
            "n_epochs": len(epochs),
            "started_at": data.get("started_at", ""),
            "completed_at": data.get("completed_at", ""),
            **meta,
        }
        if best_ep:
            row["best_epoch"] = int(best_ep.get("epoch", 0))
            for key in (
                "val_loss",
                "val_primary_loss",
                "val_eeg_rec_loss",
                "val_fnirs_rec_loss",
                "val_observation_loss",
                "val_source_target_loss",
                "val_source_target_corr_loss",
                "val_source_coupling_loss",
                "val_source_coupling_weighted_loss",
                "val_weighted_coupling_objective_loss",
                "val_eeg_source_perplexity",
                "val_fnirs_source_perplexity",
                "val_eeg_observation_perplexity",
                "val_fnirs_observation_perplexity",
                "val_source_code_overlap",
            ):
                compact = key.replace("val_", "best_val_")
                if compact == "best_val_source_coupling_weighted_loss":
                    compact = "best_val_coupling_weighted"
                if compact == "best_val_weighted_coupling_objective_loss":
                    compact = "best_val_coupling_weighted"
                row[compact] = metric_from_epoch(best_ep, key)
        if final_ep:
            row["final_epoch"] = int(final_ep.get("epoch", len(epochs)))
            for key in (
                "val_loss",
                "val_primary_loss",
                "val_eeg_rec_loss",
                "val_fnirs_rec_loss",
                "val_observation_loss",
                "val_source_target_loss",
                "val_source_target_corr_loss",
                "val_source_coupling_loss",
                "val_source_coupling_weighted_loss",
                "val_weighted_coupling_objective_loss",
                "val_eeg_source_perplexity",
                "val_fnirs_source_perplexity",
                "val_eeg_observation_perplexity",
                "val_fnirs_observation_perplexity",
                "val_source_code_overlap",
            ):
                compact = key.replace("val_", "final_val_")
                if compact == "final_val_source_coupling_weighted_loss":
                    compact = "final_val_coupling_weighted"
                if compact == "final_val_weighted_coupling_objective_loss":
                    compact = "final_val_coupling_weighted"
                row[compact] = metric_from_epoch(final_ep, key)

        row.update(extract_gate_metrics(run_dir))
        rows.append(row)

        if epochs:
            first_primary = metric_with_fallback(epochs[0], "val_primary_loss", "val_loss")
            final_primary = metric_with_fallback(final_ep, "val_primary_loss", "val_loss")
            best_primary = row.get("best_val_primary_loss")
            if best_primary is None:
                best_primary = row.get("best_val_loss")
            dyn = {
                "suite_family": family,
                "suite_id": sid,
                "run_group": group,
                "run": run_name,
                "condition": row.get("condition", ""),
                "seed": row.get("seed", ""),
                "n_epochs": len(epochs),
                "best_epoch": row.get("best_epoch"),
                "best_epoch_fraction": (
                    float(row["best_epoch"]) / len(epochs)
                    if row.get("best_epoch") and epochs
                    else None
                ),
                "first_val_primary_loss": first_primary,
                "best_val_primary_loss": best_primary,
                "final_val_primary_loss": final_primary,
                "primary_total_improvement": (
                    first_primary - best_primary
                    if first_primary is not None and best_primary is not None
                    else None
                ),
                "primary_final_vs_best_gap": (
                    final_primary - best_primary
                    if final_primary is not None and best_primary is not None
                    else None
                ),
                "epoch_to_50pct_primary_gain": first_epoch_reaching_fraction_with_fallback(
                    epochs, "val_primary_loss", "val_loss", 0.50
                ),
                "epoch_to_90pct_primary_gain": first_epoch_reaching_fraction_with_fallback(
                    epochs, "val_primary_loss", "val_loss", 0.90
                ),
                "epoch_to_95pct_primary_gain": first_epoch_reaching_fraction_with_fallback(
                    epochs, "val_primary_loss", "val_loss", 0.95
                ),
                "early_primary_slope_epoch1_5": slope_between_with_fallback(
                    epochs, "val_primary_loss", "val_loss", 0, 4
                ),
                "late_primary_slope_last5": slope_between_with_fallback(
                    epochs, "val_primary_loss", "val_loss", max(0, len(epochs) - 5), len(epochs) - 1
                ),
                "final_val_coupling_weighted": row.get("final_val_coupling_weighted"),
            }
            dynamics.append(dyn)
    return rows, dynamics


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred: Sequence[str] = ()) -> None:
    fieldnames: List[str] = []
    for key in preferred:
        if key not in fieldnames:
            fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_conditions(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("suite_family", "")),
            str(row.get("suite_id", "")),
            str(row.get("run_group", "")),
            str(row.get("condition", "")),
        )
        groups[key].append(row)
    out: List[Dict[str, Any]] = []
    for (family, sid, group, condition), group_rows in sorted(groups.items()):
        summary: Dict[str, Any] = {
            "suite_family": family,
            "suite_id": sid,
            "run_group": group,
            "condition": condition,
            "n_runs": len(group_rows),
            "n_seeds": len({r.get("seed") for r in group_rows if r.get("seed") not in ("", None)}),
            "superseded": any(bool(r.get("superseded")) for r in group_rows),
        }
        for field in NUMERIC_FIELDS:
            values = [scalar(r.get(field)) for r in group_rows]
            summary[f"{field}_mean"] = mean(values)
            summary[f"{field}_sd"] = stdev(values)
            present = [v for v in values if v is not None]
            summary[f"{field}_n"] = len(present)
        out.append(summary)
    return out


def build_leaderboard(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    leaders: List[Dict[str, Any]] = []
    active_rows = [r for r in rows if not r.get("superseded")]
    for field in NUMERIC_FIELDS:
        candidates = [r for r in active_rows if scalar(r.get(field)) is not None]
        if not candidates:
            continue
        reverse = field in HIGHER_IS_BETTER
        if field in LOWER_IS_BETTER:
            reverse = False
        candidates = sorted(candidates, key=lambda r: scalar(r.get(field)), reverse=reverse)
        for rank, row in enumerate(candidates[:5], start=1):
            leaders.append({
                "metric": field,
                "rank": rank,
                "direction": "higher" if reverse else "lower",
                "value": scalar(row.get(field)),
                "suite_family": row.get("suite_family"),
                "suite_id": row.get("suite_id"),
                "run_group": row.get("run_group"),
                "condition": row.get("condition"),
                "run": row.get("run"),
                "seed": row.get("seed"),
                "superseded": row.get("superseded"),
            })
    return leaders


def condition_lookup(conditions: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    return {
        (c["suite_family"], c["suite_id"], c["run_group"], c["condition"]): c
        for c in conditions
    }


def write_report(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    conditions: Sequence[Dict[str, Any]],
    leaders: Sequence[Dict[str, Any]],
    dynamics: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> None:
    active_rows = [r for r in rows if not r.get("superseded")]
    lines: List[str] = []
    lines.append("# Tokenizer Exploration Meta-Analysis")
    lines.append("")
    lines.append("This report aggregates formal non-smoke runs from `coupling_design_audit`, `tokenizer_coupling_capacity`, `tokenizer_coupling_discovery`, `tokenizer_next_stage`, `tokenizer_context_coupling`, and `tokenizer_coupling_stress`.")
    lines.append("")
    lines.append("Interpretation rule: this is exploratory evidence. The suites were not a balanced factorial design, so rankings and effect sizes are more reliable than p-values.")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Formal metrics files: {len(rows)}")
    lines.append(f"- Active formal runs excluding superseded context suite: {len(active_rows)}")
    lines.append(f"- Runs with Gate3 summaries: {sum(1 for r in rows if r.get('gate3'))}")
    lines.append(f"- Output directory: `{out_dir}`")
    lines.append("")

    lines.append("### Runs by suite")
    lines.append("")
    lines.append("| suite | runs | active runs | conditions |")
    lines.append("|---|---:|---:|---|")
    by_suite: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_suite[row["suite_family"]].append(row)
    for family, suite_rows in sorted(by_suite.items()):
        active = [r for r in suite_rows if not r.get("superseded")]
        conds = sorted({str(r.get("condition", "")) for r in suite_rows if r.get("condition")})
        lines.append(f"| {family} | {len(suite_rows)} | {len(active)} | {', '.join(conds[:12])} |")
    lines.append("")

    def leader(metric: str, include_superseded: bool = False) -> Optional[Dict[str, Any]]:
        matches = [
            l for l in leaders
            if l["metric"] == metric and (include_superseded or not l.get("superseded"))
        ]
        return matches[0] if matches else None

    lines.append("## Metric Leaders")
    lines.append("")
    selected_metrics = [
        "best_val_primary_loss",
        "final_val_primary_loss",
        "best_val_eeg_rec_loss",
        "best_val_fnirs_rec_loss",
        "best_val_observation_loss",
        "best_val_source_target_loss",
        "best_val_eeg_source_perplexity",
        "best_val_fnirs_source_perplexity",
        "gate3_best_mi_above_shuffle",
        "gate3_mean_mi_above_shuffle",
        "gate3_best_loso_gain",
        "gate3_weighted_true_probability",
    ]
    lines.append("| metric | direction | best value | suite | condition | run |")
    lines.append("|---|---|---:|---|---|---|")
    for metric in selected_metrics:
        top = leader(metric)
        if not top:
            continue
        lines.append(
            f"| {metric} | {top['direction']} | {fmt(top['value'])} | "
            f"{top['suite_family']} | {top.get('condition','')} | {top['run']} |"
        )
    lines.append("")

    lines.append("## Capacity And Vector Dimension")
    lines.append("")
    cap_rows = [
        c for c in conditions
        if c["suite_family"] == "tokenizer_coupling_capacity"
        and c["run_group"] == "capacity_sweep"
    ]
    if cap_rows:
        lines.append("| condition | n | best val loss | EEG source ppl | fNIRS source ppl | Gate3 MI above shuffle |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for c in sorted(cap_rows, key=lambda x: str(x["condition"])):
            lines.append(
                f"| {c['condition']} | {c['n_runs']} | "
                f"{fmt(c.get('best_val_loss_mean'))} | "
                f"{fmt(c.get('best_val_eeg_source_perplexity_mean'))} | "
                f"{fmt(c.get('best_val_fnirs_source_perplexity_mean'))} | "
                f"{fmt(c.get('gate3_best_mi_above_shuffle_mean'))} |"
            )
        lines.append("")
        lines.append("Capacity trend: larger K improves reconstruction/primary loss and codebook capacity, but Gate3 remains failed. K256 gives the best reconstruction-like scores in the capacity sweep, while K128 is the pragmatic training setting because K256 was treated as a probe rather than the main dense-training target.")
        lines.append("")

    next_rows = [
        c for c in conditions
        if c["suite_family"] == "tokenizer_next_stage"
        and c["run_group"] == "vector_dim_sweep"
    ]
    if next_rows:
        lines.append("Vector dimension sweep:")
        lines.append("")
        lines.append("| condition | n | best val primary | EEG source ppl | fNIRS source ppl | source overlap |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for c in sorted(next_rows, key=lambda x: str(x["condition"])):
            lines.append(
                f"| {c['condition']} | {c['n_runs']} | "
                f"{fmt(c.get('best_val_primary_loss_mean'))} | "
                f"{fmt(c.get('best_val_eeg_source_perplexity_mean'))} | "
                f"{fmt(c.get('best_val_fnirs_source_perplexity_mean'))} | "
                f"{fmt(c.get('best_val_source_code_overlap_mean'))} |"
            )
        lines.append("")
        lines.append("The dim128 runs dominate dim48/dim96 on primary loss in this batch, which supports the decision to move the codebook vector dimension upward. The gain is reconstruction/representation capacity, not interpretable cross-modal token pairing.")
        lines.append("")

    lines.append("## Coupling Mechanism Results")
    lines.append("")
    for family in ("tokenizer_coupling_discovery", "tokenizer_context_coupling", "tokenizer_coupling_stress"):
        family_rows = [
            c for c in conditions
            if c["suite_family"] == family and not c.get("superseded")
        ]
        if not family_rows:
            continue
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| condition | n | best primary | final primary | Gate3 MI | Gate3 LOSO | row entropy | weighted true p |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for c in sorted(family_rows, key=lambda x: (x["suite_id"], x["run_group"], str(x["condition"]))):
            lines.append(
                f"| {c['condition']} | {c['n_runs']} | "
                f"{fmt(c.get('best_val_primary_loss_mean'))} | "
                f"{fmt(c.get('final_val_primary_loss_mean'))} | "
                f"{fmt(c.get('gate3_best_mi_above_shuffle_mean'))} | "
                f"{fmt(c.get('gate3_best_loso_gain_mean'))} | "
                f"{fmt(c.get('gate3_row_entropy_ratio_mean'))} | "
                f"{fmt(c.get('gate3_weighted_true_probability_mean'))} |"
            )
        lines.append("")
    lines.append("Coupling-only conclusion: global/local/context/stress coupling losses can change auxiliary tensors and sometimes produce modest empirical MI fluctuations, but they do not create a sparse, seed-stable, held-out-subject-valid token pairing. The stress suite is the strongest negative control: weighted coupling loss dominates training, yet tensor entropy stays near uniform and LOSO utility declines.")
    lines.append("")

    lines.append("## Training Dynamics")
    lines.append("")
    dyn_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for dyn in dynamics:
        if any(r["run"] == dyn["run"] and r.get("superseded") for r in rows):
            continue
        dyn_groups[(dyn["suite_family"], str(dyn.get("condition", "")))].append(dyn)
    dyn_rows: List[Dict[str, Any]] = []
    for (family, condition), group in sorted(dyn_groups.items()):
        dyn_rows.append({
            "suite_family": family,
            "condition": condition,
            "n": len(group),
            "best_epoch_fraction_mean": mean([scalar(g.get("best_epoch_fraction")) for g in group]),
            "epoch_to_90pct_primary_gain_mean": mean([scalar(g.get("epoch_to_90pct_primary_gain")) for g in group]),
            "early_primary_slope_mean": mean([scalar(g.get("early_primary_slope_epoch1_5")) for g in group]),
            "late_primary_slope_mean": mean([scalar(g.get("late_primary_slope_last5")) for g in group]),
            "primary_final_vs_best_gap_mean": mean([scalar(g.get("primary_final_vs_best_gap")) for g in group]),
            "final_val_coupling_weighted_mean": mean([scalar(g.get("final_val_coupling_weighted")) for g in group]),
        })
    lines.append("| suite | condition | n | best epoch frac | epoch to 90% gain | early slope | late slope | final-best gap | final coupling weight |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in dyn_rows:
        lines.append(
            f"| {d['suite_family']} | {d['condition']} | {d['n']} | "
            f"{fmt(d['best_epoch_fraction_mean'])} | {fmt(d['epoch_to_90pct_primary_gain_mean'])} | "
            f"{fmt(d['early_primary_slope_mean'])} | {fmt(d['late_primary_slope_mean'])} | "
            f"{fmt(d['primary_final_vs_best_gap_mean'])} | {fmt(d['final_val_coupling_weighted_mean'])} |"
        )
    lines.append("")
    lines.append("Training bottleneck pattern: capacity and dim sweeps keep improving over many epochs, suggesting ordinary representation capacity and optimization still matter. Coupling stress/configured coupling runs often select early best epochs because the auxiliary coupling term grows faster than the actual discrete co-occurrence structure improves. This is an objective mismatch, not just under-training.")
    lines.append("")

    lines.append("## Interpretation For Next Design")
    lines.append("")
    lines.append("1. Codebook capacity should remain large enough: K64 is useful for screening, but K128/dim128 is the better default for serious tokenizer work; K256 is informative as a probe but costly.")
    lines.append("2. Pure coupling losses are not a reliable mechanism for producing interpretable EEG -> lagged fNIRS token maps. They mainly bias or regularize existing discrete assignments.")
    lines.append("3. The missing ingredient is encoder-level information exchange or a downstream sequence model that explicitly learns cross-modal temporal relations after tokenization.")
    lines.append("4. Future experiments should report both best-checkpoint and final-checkpoint audits; coupling objectives can make best checkpoint selection hide the intended late-stage pressure.")
    lines.append("")

    lines.append("## Generated Tables")
    lines.append("")
    lines.append("- `formal_runs.csv`: one row per formal training run")
    lines.append("- `condition_summary.csv`: grouped means/SDs by suite/group/condition")
    lines.append("- `metric_leaderboard.csv`: top five runs per metric")
    lines.append("- `training_dynamics.csv`: convergence-speed and plateau diagnostics")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to current working directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated tables and report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = find_repo_root(args.repo_root)
    out_dir = args.output_dir or (
        repo / "experiments" / "runs" / "tokenizer_stage_meta_analysis" / "20260623_coupling_bottleneck_meta_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, dynamics = collect_runs(repo)
    conditions = summarize_conditions(rows)
    leaders = build_leaderboard(rows)

    write_csv(
        out_dir / "formal_runs.csv",
        rows,
        preferred=(
            "suite_family",
            "suite_id",
            "run_group",
            "condition",
            "run",
            "seed",
            "k",
            "dim",
            "superseded",
            "n_epochs",
            "best_epoch",
            "final_epoch",
        ),
    )
    write_csv(
        out_dir / "condition_summary.csv",
        conditions,
        preferred=("suite_family", "suite_id", "run_group", "condition", "n_runs", "n_seeds", "superseded"),
    )
    write_csv(
        out_dir / "metric_leaderboard.csv",
        leaders,
        preferred=("metric", "rank", "direction", "value", "suite_family", "condition", "run", "seed"),
    )
    write_csv(
        out_dir / "training_dynamics.csv",
        dynamics,
        preferred=(
            "suite_family",
            "suite_id",
            "run_group",
            "condition",
            "run",
            "seed",
            "n_epochs",
            "best_epoch",
            "best_epoch_fraction",
        ),
    )
    write_report(out_dir / "meta_analysis_report.md", rows, conditions, leaders, dynamics, out_dir)

    print(json.dumps({
        "output_dir": str(out_dir),
        "formal_runs": len(rows),
        "condition_groups": len(conditions),
        "leaderboard_rows": len(leaders),
        "training_dynamics_rows": len(dynamics),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
