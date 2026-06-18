#!/usr/bin/env python
"""Frozen local residual coupling audit for exported source tokens."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.analysis import conditional_probabilities_from_counts, load_export_split, subject_block_bootstrap_gain


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def task_domain(task: str, label: str) -> str:
    text = f"{task} {label}".lower()
    if "nback" in text or "n-back" in text:
        return "nback"
    if label.lower() == "wg" or "word generation" in text:
        return "wg"
    if "mental" in text or "arithmetic" in text or label.lower() == "math":
        return "mental_arithmetic"
    if "motor" in text:
        return "motor"
    return "unknown"


def domains(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([task_domain(task, label) for task, label in zip(data["source_task"], data["label_name"])])


def event_phase(values: np.ndarray) -> np.ndarray:
    return np.where(values < 0, "pre", np.where(values < 10, "early", "late"))


def group_labels(data: dict[str, np.ndarray], scope: str) -> np.ndarray:
    dataset = np.asarray([str(value) for value in data["source_name"]])
    task = domains(data)
    phase = event_phase(data["event_relative_start_s"].astype(float))
    if scope == "global":
        return np.asarray(["global"] * len(task))
    if scope == "dataset":
        return dataset
    if scope == "task":
        return task
    if scope == "dataset_task":
        return np.asarray([f"{a}|{b}" for a, b in zip(dataset, task)])
    if scope == "dataset_task_phase":
        return np.asarray([f"{a}|{b}|{c}" for a, b, c in zip(dataset, task, phase)])
    raise ValueError(f"unknown scope {scope!r}")


def counts_by_lag(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    *,
    k_eeg: int,
    k_fnirs: int,
    max_lag_tokens: int,
) -> np.ndarray:
    n_lags = min(max_lag_tokens + 1, eeg.shape[1])
    counts = np.zeros((n_lags, k_eeg, k_fnirs), dtype=np.float64)
    for lag in range(n_lags):
        valid = eeg.shape[1] - lag
        flat = eeg[:, :valid].reshape(-1) * k_fnirs + fnirs[:, lag:].reshape(-1)
        counts[lag] = np.bincount(flat, minlength=k_eeg * k_fnirs).reshape(k_eeg, k_fnirs)
    return counts


def marginal_probabilities(counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    prior = counts.sum(axis=1) + alpha
    return prior / np.maximum(prior.sum(axis=-1, keepdims=True), 1e-12)


def per_subject_nll(
    probabilities: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    marginal: bool,
    max_lag_tokens: int,
) -> dict[int, float]:
    result = {}
    eeg = data["eeg_source_tokens"].astype(np.int64)
    fnirs = data["fnirs_source_tokens"].astype(np.int64)
    for subject in np.unique(data["subject_id"]):
        mask = data["subject_id"] == subject
        losses = []
        for lag in range(min(max_lag_tokens + 1, eeg.shape[1])):
            valid = eeg.shape[1] - lag
            target = fnirs[mask, lag:lag + valid]
            if marginal:
                prob = probabilities[lag][target]
            else:
                source = eeg[mask, :valid]
                prob = probabilities[lag][source, target]
            losses.append(float(-np.log(np.maximum(prob, 1e-12)).mean()))
        result[int(subject)] = float(np.mean(losses)) if losses else float("nan")
    return result


def subset(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in data.items()}


def shuffle_eeg_within_position(data: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = {key: value.copy() for key, value in data.items()}
    eeg = shuffled["eeg_source_tokens"].copy()
    for position in range(eeg.shape[1]):
        eeg[:, position] = eeg[rng.permutation(eeg.shape[0]), position]
    shuffled["eeg_source_tokens"] = eeg
    return shuffled


def evaluate_group(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    k_eeg: int,
    k_fnirs: int,
    max_lag_tokens: int,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    counts = counts_by_lag(
        train["eeg_source_tokens"].astype(np.int64),
        train["fnirs_source_tokens"].astype(np.int64),
        k_eeg=k_eeg,
        k_fnirs=k_fnirs,
        max_lag_tokens=max_lag_tokens,
    )
    conditional = conditional_probabilities_from_counts(counts, alpha=0.5)
    marginal = marginal_probabilities(counts)
    model_nll = per_subject_nll(conditional, test, marginal=False, max_lag_tokens=max_lag_tokens)
    marginal_nll = per_subject_nll(marginal, test, marginal=True, max_lag_tokens=max_lag_tokens)
    bootstrap = subject_block_bootstrap_gain(model_nll, marginal_nll, n_bootstrap=bootstraps, seed=seed)

    shuffled = shuffle_eeg_within_position(train, seed)
    shuffled_counts = counts_by_lag(
        shuffled["eeg_source_tokens"].astype(np.int64),
        shuffled["fnirs_source_tokens"].astype(np.int64),
        k_eeg=k_eeg,
        k_fnirs=k_fnirs,
        max_lag_tokens=max_lag_tokens,
    )
    shuffled_conditional = conditional_probabilities_from_counts(shuffled_counts, alpha=0.5)
    shuffled_model_nll = per_subject_nll(
        shuffled_conditional,
        test,
        marginal=False,
        max_lag_tokens=max_lag_tokens,
    )
    shuffled_bootstrap = subject_block_bootstrap_gain(
        shuffled_model_nll,
        marginal_nll,
        n_bootstrap=bootstraps,
        seed=seed,
    )
    return {
        "train_samples": int(len(train["subject_id"])),
        "test_samples": int(len(test["subject_id"])),
        "nll_gain": bootstrap["mean"],
        "ci_low": bootstrap["ci_low"],
        "ci_high": bootstrap["ci_high"],
        "position_shuffle_gain": shuffled_bootstrap["mean"],
        "position_shuffle_ci_low": shuffled_bootstrap["ci_low"],
        "beats_position_shuffle": bool(bootstrap["ci_low"] > shuffled_bootstrap["ci_high"]),
    }


def analyze_scope(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    scope: str,
    k_eeg: int,
    k_fnirs: int,
    max_lag_tokens: int,
    min_samples: int,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    train_labels = group_labels(train, scope)
    test_labels = group_labels(test, scope)
    groups = sorted(set(train_labels) & set(test_labels))
    results = {}
    for index, group in enumerate(groups):
        train_mask = train_labels == group
        test_mask = test_labels == group
        if train_mask.sum() < min_samples or test_mask.sum() < max(20, min_samples // 4):
            continue
        results[group] = evaluate_group(
            subset(train, train_mask),
            subset(test, test_mask),
            k_eeg=k_eeg,
            k_fnirs=k_fnirs,
            max_lag_tokens=max_lag_tokens,
            bootstraps=bootstraps,
            seed=seed + index,
        )
    gains = [value["nll_gain"] for value in results.values()]
    lows = [value["ci_low"] for value in results.values()]
    return {
        "scope": scope,
        "group_count": len(results),
        "mean_gain": float(np.mean(gains)) if gains else float("nan"),
        "best_ci_low": float(np.max(lows)) if lows else float("nan"),
        "groups": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-lag-tokens", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_export_split(args.export_dir, "train")
    test = load_export_split(args.export_dir, "test")
    k_eeg = int(max(train["eeg_source_tokens"].max(), test["eeg_source_tokens"].max())) + 1
    k_fnirs = int(max(train["fnirs_source_tokens"].max(), test["fnirs_source_tokens"].max())) + 1
    scopes = ("global", "dataset", "task", "dataset_task", "dataset_task_phase")
    payload = {
        "schema_version": "local_residual_coupling_audit_v1",
        "run_name": args.run_name,
        "export_dir": str(Path(args.export_dir).resolve()),
        "k_eeg": k_eeg,
        "k_fnirs": k_fnirs,
        "max_lag_tokens": args.max_lag_tokens,
        "scopes": {},
    }
    rows = []
    for scope in scopes:
        scope_result = analyze_scope(
            train,
            test,
            scope=scope,
            k_eeg=k_eeg,
            k_fnirs=k_fnirs,
            max_lag_tokens=args.max_lag_tokens,
            min_samples=args.min_samples,
            bootstraps=args.bootstraps,
            seed=args.seed,
        )
        payload["scopes"][scope] = scope_result
        for group, metrics in scope_result["groups"].items():
            rows.append({"scope": scope, "group": group, **metrics})
    payload["decision_features"] = {
        "best_scope": max(
            payload["scopes"],
            key=lambda name: (
                payload["scopes"][name]["best_ci_low"]
                if not np.isnan(payload["scopes"][name]["best_ci_low"]) else -1e9
            ),
        ),
        "any_group_ci_low_positive": any(row["ci_low"] > 0 for row in rows),
        "any_group_beats_position_shuffle": any(row["beats_position_shuffle"] for row in rows),
    }
    write_json(output_dir / "summary.json", payload)
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload["decision_features"], indent=2))


if __name__ == "__main__":
    main()
