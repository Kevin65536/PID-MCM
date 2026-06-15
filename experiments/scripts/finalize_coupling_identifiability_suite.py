#!/usr/bin/env python
"""Aggregate Phase C, enforce promotion gates, and optionally launch Phase D."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from experiments.scripts.launch_coupling_identifiability_suite import NEUTRAL_CHECKPOINT, q, training_command
from src.analysis import load_export_split


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aligned_token_metrics(reference: np.ndarray, candidate: np.ndarray, k: int = 32) -> Dict[str, float]:
    left = reference.reshape(-1).astype(np.int64)
    right = candidate.reshape(-1).astype(np.int64)
    confusion = np.bincount(left * k + right, minlength=k * k).reshape(k, k)
    row, col = linear_sum_assignment(-confusion)
    mapping = np.arange(k)
    mapping[col] = row
    aligned = mapping[right]
    return {
        "nmi": float(normalized_mutual_info_score(left, right)),
        "flip_rate": float(np.mean(left != aligned)),
    }


def export_metrics(suite_dir: Path, condition: str, seed: int) -> Dict[str, Any]:
    return load_export_split(suite_dir / f"token_exports/{condition}_seed{seed}", "test")


def run_metrics(suite_dir: Path, condition: str, seed: int) -> Dict[str, float]:
    path = suite_dir / f"tokenizer_interventions/{condition}_seed{seed}/metrics.json"
    return read_json(path).get("final_metrics", {})


def frozen_metrics(suite_dir: Path, condition: str, seed: int) -> Dict[str, Any]:
    payload = read_json(
        suite_dir / f"frozen_calibration/{condition}_seed{seed}/frozen_model_results.json"
    )
    return payload["global"]["F4"]["test_matrix"]


def audit_metrics(suite_dir: Path, condition: str, seed: int) -> Dict[str, Any]:
    return read_json(suite_dir / f"data_position_audit/{condition}_seed{seed}/summary.json")


def relative_difference(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), 1e-12)


def candidate_summary(suite_dir: Path, condition: str, seed: int, baseline: Dict[str, Any]) -> Dict[str, Any]:
    metrics = run_metrics(suite_dir, condition, seed)
    frozen = frozen_metrics(suite_dir, condition, seed)
    audit = audit_metrics(suite_dir, condition, seed)
    global_frozen = frozen.get("global", {})
    task_ci = [
        value.get("nuisance_subject_bootstrap_gain", {}).get("ci_low", -float("inf"))
        for domain, value in frozen.items() if domain != "global"
    ]
    source_reconstruction = float(metrics.get("val_source_target_loss", float("inf")))
    observation_reconstruction = float(metrics.get("val_observation_loss", float("inf")))
    result = {
        "condition": condition,
        "seed": seed,
        "f4_global_nll_gain": float(global_frozen.get("nuisance_adjusted_nll_gain", float("nan"))),
        "f4_global_ci_low": float(global_frozen.get("nuisance_subject_bootstrap_gain", {}).get("ci_low", -float("inf"))),
        "best_task_ci_low": float(max(task_ci)) if task_ci else -float("inf"),
        "position_accuracy": float(audit["position_event_lag"]["position_loso_accuracy"]),
        "event_phase_accuracy": float(audit["position_event_lag"]["event_phase_loso_accuracy"]),
        "subject_leakage": float(audit["leakage"]["subject_classification_accuracy"]),
        "task_leakage": float(audit["leakage"]["task_classification_accuracy"]),
        "source_reconstruction": source_reconstruction,
        "observation_reconstruction": observation_reconstruction,
        "eeg_source_perplexity": float(metrics.get("val_eeg_source_perplexity", 0.0)),
        "fnirs_source_perplexity": float(metrics.get("val_fnirs_source_perplexity", 0.0)),
    }
    result["position_increase"] = result["position_accuracy"] - baseline["position_accuracy"]
    result["event_phase_increase"] = result["event_phase_accuracy"] - baseline["event_phase_accuracy"]
    result["subject_leakage_increase"] = result["subject_leakage"] - baseline["subject_leakage"]
    result["source_reconstruction_change"] = relative_difference(
        baseline["source_reconstruction"], source_reconstruction,
    )
    result["observation_reconstruction_change"] = relative_difference(
        baseline["observation_reconstruction"], observation_reconstruction,
    )
    result["passes"] = bool(
        result["f4_global_nll_gain"] > baseline["f4_global_nll_gain"]
        and (result["f4_global_ci_low"] > 0 or result["best_task_ci_low"] > 0)
        and result["position_increase"] <= 0.02
        and result["event_phase_increase"] <= 0.02
        and result["source_reconstruction_change"] <= 0.02
        and result["observation_reconstruction_change"] <= 0.02
        and min(result["eeg_source_perplexity"], result["fnirs_source_perplexity"]) >= 24.0
        and result["subject_leakage_increase"] <= 0.03
    )
    return result


def make_baseline(suite_dir: Path, seed: int) -> Dict[str, Any]:
    metrics = run_metrics(suite_dir, "t0", seed)
    frozen = frozen_metrics(suite_dir, "t0", seed)["global"]
    audit = audit_metrics(suite_dir, "t0", seed)
    return {
        "f4_global_nll_gain": float(frozen.get("nuisance_adjusted_nll_gain", float("nan"))),
        "position_accuracy": float(audit["position_event_lag"]["position_loso_accuracy"]),
        "event_phase_accuracy": float(audit["position_event_lag"]["event_phase_loso_accuracy"]),
        "subject_leakage": float(audit["leakage"]["subject_classification_accuracy"]),
        "source_reconstruction": float(metrics.get("val_source_target_loss", float("inf"))),
        "observation_reconstruction": float(metrics.get("val_observation_loss", float("inf"))),
    }


def negative_control(suite_dir: Path, seed: int) -> Dict[str, Any]:
    t0 = export_metrics(suite_dir, "t0", seed)
    t1 = export_metrics(suite_dir, "t1", seed)
    eeg = aligned_token_metrics(t0["eeg_source_tokens"], t1["eeg_source_tokens"])
    fnirs = aligned_token_metrics(t0["fnirs_source_tokens"], t1["fnirs_source_tokens"])
    t0_metrics = run_metrics(suite_dir, "t0", seed)
    t1_metrics = run_metrics(suite_dir, "t1", seed)
    source_diff = relative_difference(
        float(t0_metrics["val_source_target_loss"]), float(t1_metrics["val_source_target_loss"]),
    )
    observation_diff = relative_difference(
        float(t0_metrics["val_observation_loss"]), float(t1_metrics["val_observation_loss"]),
    )
    return {
        "seed": seed,
        "eeg": eeg,
        "fnirs": fnirs,
        "source_reconstruction_difference": source_diff,
        "observation_reconstruction_difference": observation_diff,
        "passes": bool(
            min(eeg["nmi"], fnirs["nmi"]) > 0.995
            and max(source_diff, observation_diff) < 0.01
        ),
    }


def write_long_config(
    suite_dir: Path,
    condition: str,
    seed: int,
    device: str,
    *,
    resume: bool,
) -> tuple[Path, Path | None, str]:
    short_config = suite_dir / f"tokenizer_interventions/configs/{condition}_seed{seed}.yaml"
    if condition == "t5":
        short_config = suite_dir / f"tokenizer_interventions/configs/t5_seed{seed}.yaml"
    if not short_config.exists():
        short_config = suite_dir / f"tokenizer_interventions/configs/{condition}_seed20260615.yaml"
        if condition == "t5":
            short_config = suite_dir / "tokenizer_interventions/configs/t5_seed20260615.yaml"
    payload = yaml.safe_load(short_config.read_text(encoding="utf-8"))
    payload["experiment"]["run_group"] = f"coupling_design_audit/{suite_dir.name}/long_validation"
    payload["experiment"]["name"] = f"long_{condition}_seed{seed}"
    payload["experiment"]["device"] = device
    payload["experiment"]["seed"] = seed
    payload["training"]["epochs"] = 280
    payload["training"]["early_stopping"] = {"enabled": False}
    output = suite_dir / f"long_validation/configs/{condition}_seed{seed}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    resume_path = None
    if resume:
        resume_path = suite_dir / f"tokenizer_interventions/{condition}_seed{seed}/checkpoints/checkpoint_epoch_120.pt"
    return output, resume_path, f"long_{condition}_seed{seed}"


def launch_long_validation(suite_dir: Path, candidate: str) -> Dict[str, int]:
    queue_specs = {0: [], 1: []}
    assignments = [
        (0, "t0", 20260615, True),
        (1, "t0", 20260616, True),
        (0, "t0", 20260617, False),
        (0, candidate, 20260615, True),
        (1, candidate, 20260616, True),
        (1, candidate, 20260617, False),
    ]
    for gpu, condition, seed, resume in assignments:
        config, resume_path, run_name = write_long_config(
            suite_dir, condition, seed, f"cuda:{gpu}", resume=resume,
        )
        command = training_command(config, run_name, smoke=False)
        if resume_path is not None:
            command += f" --resume {q(resume_path)}"
        queue_specs[gpu].append(command)
    pids = {}
    for gpu, commands in queue_specs.items():
        queue = suite_dir / f"queue_logs/long_gpu{gpu}.sh"
        queue.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"cd {q(project_root)}\n" + "\n".join(commands) + "\n",
            encoding="utf-8",
        )
        queue.chmod(0o755)
        log = (suite_dir / f"queue_logs/long_gpu{gpu}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", str(queue)], cwd=project_root, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()
        pids[f"gpu{gpu}"] = process.pid
    return pids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--launch-long", action="store_true")
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    if args.wait:
        while not all((suite_dir / f"queue_logs/gpu{gpu}.done").exists() for gpu in (0, 1)):
            time.sleep(300)

    negative = [negative_control(suite_dir, seed) for seed in (20260615, 20260616)]
    negative_passes = all(item["passes"] for item in negative)
    available = ["t2", "t3", "t4"]
    if all((suite_dir / f"tokenizer_interventions/t5_seed{seed}").exists() for seed in (20260615, 20260616)):
        available.append("t5")
    candidates = []
    for condition in available:
        seed_results = []
        for seed in (20260615, 20260616):
            baseline = make_baseline(suite_dir, seed)
            t1_gain = float(frozen_metrics(suite_dir, "t1", seed)["global"]["nuisance_adjusted_nll_gain"])
            result = candidate_summary(suite_dir, condition, seed, baseline)
            result["beats_t1"] = result["f4_global_nll_gain"] > t1_gain
            result["passes"] = bool(result["passes"] and result["beats_t1"])
            seed_results.append(result)
        candidates.append({
            "condition": condition,
            "seeds": seed_results,
            "passes": all(item["passes"] for item in seed_results),
            "mean_gain": float(np.mean([item["f4_global_nll_gain"] for item in seed_results])),
        })
    passing = [item for item in candidates if item["passes"]]
    best = max(passing, key=lambda item: item["mean_gain"], default=None)
    status = "promoted" if negative_passes and best is not None else "redesign_required"
    decision = {
        "status": status,
        "negative_control": negative,
        "candidates": candidates,
        "selected_candidate": None if best is None else best["condition"],
    }
    long_pids = None
    if status == "promoted" and args.launch_long:
        long_pids = launch_long_validation(suite_dir, best["condition"])
        decision["status"] = "long_validation_running"
        decision["long_validation_pids"] = long_pids
    (suite_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    summary = {"negative_control_passes": negative_passes, "candidate_count": len(candidates), **decision}
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (suite_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "passes", "mean_gain"])
        writer.writeheader()
        writer.writerows({key: item[key] for key in writer.fieldnames} for item in candidates)
    report_lines = [
        "# Coupling Identifiability Audit",
        "",
        f"Decision: `{decision['status']}`",
        f"Tensor-only negative control passed: `{negative_passes}`",
        f"Selected candidate: `{decision['selected_candidate']}`",
        "",
        "The machine-readable evidence is stored in `summary.json` and `decision.json`.",
    ]
    (suite_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
