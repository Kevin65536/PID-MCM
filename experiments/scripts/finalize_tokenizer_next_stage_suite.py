#!/usr/bin/env python
"""Finalize and configure the K128 dim/cognitive-transfer next-stage suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / (
    "experiments/configs/source_observation/croce_local/"
    "highwl_v2_branch_norm_coupling0_lr2e4_compile.yaml"
)
PREVIOUS_CAPACITY = PROJECT_ROOT / "experiments/runs/tokenizer_coupling_capacity/20260617_185619_codebook_mission_bias_v1"
COGNITIVE_SEEDS = (20260625, 20260626)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def best_epoch(metrics: Dict[str, Any]) -> Dict[str, Any]:
    epochs = [epoch for epoch in metrics.get("epochs", []) if epoch.get("val_loss") is not None]
    if not epochs:
        return {}

    def score(epoch: Dict[str, Any]) -> float:
        metric = epoch.get("metrics", {}).get("val_primary_loss", epoch.get("val_loss"))
        return float(metric)

    return min(epochs, key=score)


def dim_from_name(name: str) -> int:
    for part in name.split("_"):
        if part.startswith("dim") and part[3:].isdigit():
            return int(part[3:])
    if "k128" in name:
        return 48
    return -1


def summarize_run(run_dir: Path, source: str) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    metrics = read_json(metrics_path)
    best = best_epoch(metrics)
    best_metrics = best.get("metrics", {})
    row = {
        "run": run_dir.name,
        "source": source,
        "dim": dim_from_name(run_dir.name),
        "best_epoch": best.get("epoch"),
        "best_val_loss": best.get("val_loss"),
        "best_val_primary_loss": best_metrics.get("val_primary_loss", best.get("val_loss")),
        "val_source_target_loss": best_metrics.get("val_source_target_loss"),
        "val_observation_loss": best_metrics.get("val_observation_loss"),
        "val_eeg_source_perplexity": best_metrics.get("val_eeg_source_perplexity"),
        "val_fnirs_source_perplexity": best_metrics.get("val_fnirs_source_perplexity"),
        "val_source_code_overlap": best_metrics.get("val_source_code_overlap"),
    }
    gate_path = run_dir / "analysis/gate_summary.json"
    if gate_path.exists():
        gates = read_json(gate_path).get("final_summary", {}).get("gate_verdicts", {})
        for gate in ("gate0", "gate1", "gate2", "gate3", "gate4"):
            row[gate] = gates.get(gate)
    return row


def collect_rows(suite_dir: Path) -> list[Dict[str, Any]]:
    rows = []
    previous_root = PREVIOUS_CAPACITY / "capacity_sweep"
    for run_dir in sorted(previous_root.glob("s2_croce_local_highwl_v2_capacity_k128_coupling0_seed*")):
        row = summarize_run(run_dir, "baseline_capacity_dim48")
        if row:
            rows.append(row)
    for run_dir in sorted((suite_dir / "vector_dim_sweep").glob("k128_dim*_seed*")):
        row = summarize_run(run_dir, "next_stage_vector_dim")
        if row:
            rows.append(row)
    for run_dir in sorted((suite_dir / "cognitive_tokenizer").glob("k128_dim*_cognitive_nback_wg_seed*")):
        row = summarize_run(run_dir, "cognitive_tokenizer")
        if row:
            rows.append(row)
    return rows


def mean_by_dim(rows: list[Dict[str, Any]]) -> Dict[int, float]:
    grouped: Dict[int, list[float]] = {}
    for row in rows:
        if row.get("source") != "next_stage_vector_dim":
            continue
        value = row.get("best_val_primary_loss")
        if isinstance(value, (int, float)):
            grouped.setdefault(int(row["dim"]), []).append(float(value))
    return {dim: sum(values) / len(values) for dim, values in grouped.items() if values}


def select_dim(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    means = mean_by_dim(rows)
    selected = 96
    reason = "default_dim96_until_vector_dim_sweep_has_clear_evidence"
    if 96 in means and 128 in means:
        improvement = (means[96] - means[128]) / max(abs(means[96]), 1e-8)
        if improvement >= 0.02:
            selected = 128
            reason = "dim128_improved_mean_val_primary_loss_by_at_least_2_percent_over_dim96"
        else:
            reason = "dim128_did_not_clear_2_percent_margin_over_dim96"
    elif 128 in means and 96 not in means:
        selected = 128
        reason = "only_dim128_completed"
    return {
        "selected_dim": selected,
        "reason": reason,
        "mean_val_primary_loss_by_dim": {str(key): value for key, value in sorted(means.items())},
    }


def cognitive_run_name(dim: int, seed: int) -> str:
    return f"k128_dim{dim}_cognitive_nback_wg_seed{seed}"


def cognitive_config(suite_name: str, dim: int, seed: int, device: str, *, smoke: bool) -> Dict[str, Any]:
    run_group_leaf = "smoke" if smoke else "cognitive_tokenizer"
    return {
        "_base_": str(BASE_CONFIG),
        "experiment": {
            "name": cognitive_run_name(dim, seed),
            "run_group": f"tokenizer_next_stage/{suite_name}/{run_group_leaf}",
            "seed": int(seed),
            "device": device,
            "description": "K128 source tokenizer trained only on simultaneous cognitive nback/wg windows.",
        },
        "data": {
            "seed": int(seed),
            "entry_filters": {
                "include_source_names": ["simultaneous_cognitive"],
                "include_label_names": ["nback", "wg"],
            },
        },
        "model": {
            "source": {
                "codebook_size": 128,
                "codebook_dim": int(dim),
            },
            "eeg_observation": {"codebook_size": 256},
            "fnirs_observation": {"codebook_size": 128},
        },
        "loss": {
            "coupling": {
                "weight": 0.0,
                "max_lag_tokens": 5,
                "lag_focus_weight": 0.0,
                "smoothness_weight": 0.0,
                "pair_likelihood_weight": 0.0,
                "lag_evidence_weight": 0.0,
                "effective_smoothness_weight": 0.0,
                "interaction_lag_sparsity_weight": 0.0,
            }
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 2e-4,
            "min_lr": 5e-5,
            "epochs": 2 if smoke else 160,
            "early_stopping": {
                "enabled": not smoke,
                "start_epoch": 41,
                "patience": 60,
                "metric": "val_primary_loss",
                "mode": "min",
            },
            "validation": {
                "interval_epochs": 1 if smoke else 2,
                "start_epoch": 1,
                "max_batches": 1 if smoke else None,
            },
            "checkpoint": {"save_every": 1 if smoke else 20},
        },
    }


def write_cognitive_configs(suite_dir: Path, selection: Dict[str, Any]) -> list[str]:
    dim = int(selection["selected_dim"])
    suite_name = suite_dir.name
    paths = []
    for index, seed in enumerate(COGNITIVE_SEEDS):
        device = f"cuda:{index % 2}"
        config_path = suite_dir / "cognitive_tokenizer" / "configs" / f"{cognitive_run_name(dim, seed)}.yaml"
        write_yaml(config_path, cognitive_config(suite_name, dim, seed, device, smoke=False))
        paths.append(str(config_path))
    return paths


def write_summary(suite_dir: Path, rows: list[Dict[str, Any]], selection: Dict[str, Any]) -> None:
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (suite_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "schema_version": "tokenizer_next_stage_k128_dim_cognitive_transfer_v1",
        "suite_dir": str(suite_dir),
        "selection": selection,
        "runs": rows,
    }
    write_json(suite_dir / "summary.json", summary)
    write_json(suite_dir / "selected_dim.json", selection)
    decision = {
        "status": "complete" if any(row.get("source") == "cognitive_tokenizer" for row in rows) else "pending_cognitive",
        "selected_dim": selection["selected_dim"],
        "reason": selection["reason"],
        "task_aware_coupling": "diagnostic_only_not_promoted",
    }
    write_json(suite_dir / "decision.json", decision)
    report_lines = [
        "# Tokenizer Next Stage Suite",
        "",
        f"Selected source codebook dim: {selection['selected_dim']}",
        f"Reason: {selection['reason']}",
        "",
        "Rows:",
    ]
    for row in rows:
        report_lines.append(
            f"- {row.get('run')}: source={row.get('source')} dim={row.get('dim')} "
            f"best_val_primary={row.get('best_val_primary_loss')}"
        )
    (suite_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--write-cognitive-configs", action="store_true")
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    rows = collect_rows(suite_dir)
    selection = select_dim(rows)
    if args.write_cognitive_configs:
        selection["cognitive_config_paths"] = write_cognitive_configs(suite_dir, selection)
    write_summary(suite_dir, rows, selection)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
