#!/usr/bin/env python
"""Analyze deterministic coupling-audit exports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.analysis import (
    build_lag_pair_table,
    conditional_probabilities_from_counts,
    gaussian_conditional_mutual_information,
    lag_mutual_information,
    load_export_split,
    loso_ridge_scores,
    subject_block_bootstrap_gain,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def load_exports(export_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        split: load_export_split(export_dir, split)
        for split in ("train", "val", "test")
    }


def domain_name(task: str, label: str) -> str:
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


def add_domains(data: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([
        domain_name(task, label)
        for task, label in zip(data["source_task"], data["label_name"])
    ], dtype=str)


def event_phase(event_time: np.ndarray) -> np.ndarray:
    values = np.asarray(event_time)
    return np.where(values < 0, "pre", np.where(values < 10, "early", "late"))


def distributions_by(values: np.ndarray, groups: np.ndarray, k: int) -> tuple[list[str], np.ndarray]:
    labels = sorted(str(item) for item in np.unique(groups))
    rows = []
    for label in labels:
        counts = np.bincount(values[groups.astype(str) == label], minlength=k).astype(np.float64) + 0.5
        rows.append(counts / counts.sum())
    return labels, np.stack(rows)


def js_matrix(distributions: np.ndarray) -> np.ndarray:
    result = np.zeros((len(distributions), len(distributions)), dtype=np.float64)
    for i in range(len(distributions)):
        for j in range(len(distributions)):
            result[i, j] = float(jensenshannon(distributions[i], distributions[j]) ** 2)
    return result


def loso_position_accuracy(features: np.ndarray, targets: np.ndarray, subjects: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if len(features) > 250_000:
        selected = rng.choice(len(features), 250_000, replace=False)
        features, targets, subjects = features[selected], targets[selected], subjects[selected]
    scores = []
    for subject in np.unique(subjects):
        train = subjects != subject
        test = ~train
        if np.unique(targets[train]).size < 2:
            continue
        scaler = StandardScaler().fit(features[train])
        model = RidgeClassifier(alpha=10.0).fit(scaler.transform(features[train]), targets[train])
        scores.append(accuracy_score(targets[test], model.predict(scaler.transform(features[test]))))
    return float(np.mean(scores)) if scores else float("nan")


def fixed_position_sequences(data: Dict[str, np.ndarray], position: int) -> tuple[np.ndarray, np.ndarray]:
    eeg_rows, fnirs_rows = [], []
    ids = data["cache_entry_id"].astype(str)
    for entry_id in np.unique(ids):
        indices = np.flatnonzero(ids == entry_id)
        order = indices[np.argsort(data["crop_start_s"][indices])]
        if len(order) < 2:
            continue
        eeg_rows.append(data["eeg_source_tokens"][order, position])
        fnirs_rows.append(data["fnirs_source_tokens"][order, position])
    if not eeg_rows:
        return np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.int64)
    width = min(len(row) for row in eeg_rows)
    return np.stack([row[:width] for row in eeg_rows]), np.stack([row[:width] for row in fnirs_rows])


def counts_by_lag(eeg: np.ndarray, fnirs: np.ndarray, k_eeg: int, k_fnirs: int) -> np.ndarray:
    table = build_lag_pair_table(eeg, fnirs)
    counts = np.zeros((eeg.shape[1], k_eeg, k_fnirs), dtype=np.float64)
    for lag in range(eeg.shape[1]):
        mask = table.lag == lag
        flat = table.eeg[mask] * k_fnirs + table.fnirs[mask]
        counts[lag] = np.bincount(flat, minlength=k_eeg * k_fnirs).reshape(k_eeg, k_fnirs)
    return counts


def evaluate_nll_by_subject(
    probabilities: np.ndarray,
    data: Dict[str, np.ndarray],
    *,
    marginal: bool,
) -> Dict[int, float]:
    eeg = data["eeg_source_tokens"].astype(np.int64)
    fnirs = data["fnirs_source_tokens"].astype(np.int64)
    subjects = data["subject_id"].astype(np.int64)
    result = {}
    for subject in np.unique(subjects):
        indices = subjects == subject
        losses = []
        for lag in range(eeg.shape[1]):
            valid = eeg.shape[1] - lag
            if marginal:
                logp = np.log(np.maximum(probabilities[lag, fnirs[indices, lag:]], 1e-12))
            else:
                logp = np.log(np.maximum(
                    probabilities[lag, eeg[indices, :valid], fnirs[indices, lag:]], 1e-12,
                ))
            losses.append(-float(logp.mean()))
        result[int(subject)] = float(np.mean(losses))
    return result


def permutation_max_mi(
    data: Dict[str, np.ndarray],
    observed: np.ndarray,
    *,
    k_eeg: int,
    k_fnirs: int,
    n_permutations: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    eeg = data["eeg_source_tokens"].astype(np.int64)
    fnirs = data["fnirs_source_tokens"].astype(np.int64)
    domains = add_domains(data)
    phase = event_phase(data["token_event_time_s"][:, 0])
    strata = np.asarray([
        f"{subject}|{task}|{anchor}|{event_phase_value}"
        for subject, task, anchor, event_phase_value in zip(
            data["subject_id"], domains, data["anchor"], phase,
        )
    ], dtype=str)
    group_indices = [np.flatnonzero(strata == group) for group in np.unique(strata)]
    null_max = np.empty(n_permutations, dtype=np.float64)
    for permutation in range(n_permutations):
        permuted_eeg = eeg.copy()
        for indices in group_indices:
            if len(indices) > 1:
                permuted_eeg[indices] = eeg[rng.permutation(indices)]
        null_max[permutation] = float(lag_mutual_information(
            permuted_eeg, fnirs, k_eeg=k_eeg, k_fnirs=k_fnirs,
        ).max())
    corrected_p = [float((1 + np.sum(null_max >= value)) / (len(null_max) + 1)) for value in observed]
    return {
        "corrected_p": corrected_p,
        "null_max_mean": float(null_max.mean()),
        "null_max_95": float(np.quantile(null_max, 0.95)),
    }


def cca_loso(x: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> float:
    scores = []
    for subject in np.unique(subjects):
        train = subjects != subject
        test = ~train
        if train.sum() < 10 or test.sum() < 2:
            continue
        xs = StandardScaler().fit(x[train])
        ys = StandardScaler().fit(y[train])
        components = min(3, x.shape[1], y.shape[1])
        model = CCA(n_components=components, max_iter=500).fit(xs.transform(x[train]), ys.transform(y[train]))
        x_proj, y_proj = model.transform(xs.transform(x[test]), ys.transform(y[test]))
        correlations = [np.corrcoef(x_proj[:, i], y_proj[:, i])[0, 1] for i in range(components)]
        scores.append(float(np.nanmean(correlations)))
    return float(np.nanmean(scores)) if scores else float("nan")


def information_ladder(all_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    domains = add_domains(all_data)
    subjects = np.repeat(all_data["subject_id"], 10)
    nuisance = np.stack([
        np.repeat(domains, 10),
        event_phase(all_data["token_event_time_s"].reshape(-1)),
        all_data["token_relative_position"].reshape(-1).astype(str),
    ], axis=1)
    representations = {
        "raw": (all_data["eeg_raw_features"], all_data["fnirs_raw_features"]),
        "source": (all_data["eeg_source_features"], all_data["fnirs_source_features"]),
        "observation": (all_data["eeg_observation_features"], all_data["fnirs_observation_features"]),
        "continuous": (all_data["eeg_source_latent"].astype(np.float32), all_data["fnirs_source_latent"].astype(np.float32)),
        "discrete": (
            np.eye(int(all_data["eeg_source_tokens"].max()) + 1, dtype=np.float32)[all_data["eeg_source_tokens"]],
            np.eye(int(all_data["fnirs_source_tokens"].max()) + 1, dtype=np.float32)[all_data["fnirs_source_tokens"]],
        ),
    }
    result = {}
    for domain in ("global", "nback", "wg", "motor", "mental_arithmetic"):
        sample_mask = np.ones(len(domains), dtype=bool) if domain == "global" else domains == domain
        token_mask = np.repeat(sample_mask, 10)
        if token_mask.sum() < 100:
            continue
        domain_result = {}
        for level, (x, y) in representations.items():
            x_flat = x.reshape(-1, x.shape[-1])[token_mask]
            y_flat = y.reshape(-1, y.shape[-1])[token_mask]
            subject_flat = subjects[token_mask]
            nuisance_flat = nuisance[token_mask]
            if len(x_flat) > 200_000:
                rng = np.random.default_rng(20260615)
                selected = rng.choice(len(x_flat), 200_000, replace=False)
                x_flat, y_flat = x_flat[selected], y_flat[selected]
                subject_flat, nuisance_flat = subject_flat[selected], nuisance_flat[selected]
            domain_result[level] = {
                "loso_ridge": loso_ridge_scores(x_flat, y_flat, subject_flat),
                "loso_cca_correlation": cca_loso(x_flat, y_flat, subject_flat),
                "conditional_mi_gaussian": gaussian_conditional_mutual_information(
                    x_flat, y_flat, nuisance_flat,
                ),
            }
        result[domain] = domain_result
    return result


def leakage_scores(all_data: Dict[str, np.ndarray], seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    eeg_k = int(all_data["eeg_source_tokens"].max()) + 1
    fnirs_k = int(all_data["fnirs_source_tokens"].max()) + 1
    features = np.concatenate([
        np.eye(eeg_k, dtype=np.float32)[all_data["eeg_source_tokens"]].mean(axis=1),
        np.eye(fnirs_k, dtype=np.float32)[all_data["fnirs_source_tokens"]].mean(axis=1),
    ], axis=1)
    order = rng.permutation(len(features))
    cut = int(0.8 * len(order))
    train, test = order[:cut], order[cut:]
    scaler = StandardScaler().fit(features[train])
    subject_model = RidgeClassifier(alpha=10.0).fit(
        scaler.transform(features[train]), all_data["subject_id"][train],
    )
    domains = add_domains(all_data)
    task_model = RidgeClassifier(alpha=10.0).fit(
        scaler.transform(features[train]), domains[train],
    )
    return {
        "subject_classification_accuracy": float(accuracy_score(
            all_data["subject_id"][test], subject_model.predict(scaler.transform(features[test])),
        )),
        "task_classification_accuracy": float(accuracy_score(
            domains[test], task_model.predict(scaler.transform(features[test])),
        )),
    }


def concatenate_splits(splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = set.intersection(*(set(data) for data in splits.values()))
    return {key: np.concatenate([splits[split][key] for split in ("train", "val", "test")]) for key in keys}


def plot_position_audit(output_dir: Path, labels: list[str], distributions: np.ndarray, js: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(distributions, aspect="auto", cmap="viridis")
    axes[0].set_title("P(fNIRS token | relative position)")
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_xlabel("fNIRS token")
    image = axes[1].imshow(js, cmap="magma")
    axes[1].set_title("Position marginal JS divergence")
    axes[1].set_xticks(range(len(labels)), labels, rotation=45)
    axes[1].set_yticks(range(len(labels)), labels)
    figure.colorbar(image, ax=axes[1])
    figure.tight_layout()
    figure.savefig(output_dir / "position_token_marginals.png", dpi=180)
    plt.close(figure)


def plot_task_phase_audit(
    output_dir: Path,
    task_labels: list[str],
    task_distributions: np.ndarray,
    phase_labels: list[str],
    phase_distributions: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(task_distributions, aspect="auto", cmap="viridis")
    axes[0].set_title("P(fNIRS token | task)")
    axes[0].set_yticks(range(len(task_labels)), task_labels)
    axes[0].set_xlabel("fNIRS token")
    axes[1].imshow(phase_distributions, aspect="auto", cmap="viridis")
    axes[1].set_title("P(fNIRS token | event phase)")
    axes[1].set_yticks(range(len(phase_labels)), phase_labels)
    axes[1].set_xlabel("fNIRS token")
    figure.tight_layout()
    figure.savefig(output_dir / "task_phase_token_marginals.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()
    export_dir = Path(args.export_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_exports(export_dir)
    all_data = concatenate_splits(splits)
    k_eeg = int(all_data["eeg_source_tokens"].max()) + 1
    k_fnirs = int(all_data["fnirs_source_tokens"].max()) + 1

    token_values = all_data["fnirs_source_tokens"].reshape(-1)
    position_values = all_data["token_relative_position"].reshape(-1).astype(str)
    labels, position_distributions = distributions_by(token_values, position_values, k_fnirs)
    position_js = js_matrix(position_distributions)
    plot_position_audit(output_dir, labels, position_distributions, position_js)

    token_features = np.eye(k_fnirs, dtype=np.float32)[token_values]
    repeated_subjects = np.repeat(all_data["subject_id"], 10)
    position_accuracy = loso_position_accuracy(
        token_features, position_values, repeated_subjects, args.seed,
    )
    phase_values = event_phase(all_data["token_event_time_s"].reshape(-1))
    phase_accuracy = loso_position_accuracy(token_features, phase_values, repeated_subjects, args.seed)
    phase_labels, phase_distributions = distributions_by(token_values, phase_values, k_fnirs)
    repeated_domains = np.repeat(add_domains(all_data), 10)
    task_labels, task_distributions = distributions_by(token_values, repeated_domains, k_fnirs)
    plot_task_phase_audit(
        output_dir, task_labels, task_distributions, phase_labels, phase_distributions,
    )
    sample_histograms = np.eye(k_fnirs, dtype=np.float32)[all_data["fnirs_source_tokens"]].mean(axis=1)
    crop_start_bins = np.rint(all_data["event_relative_start_s"] / 2.0).astype(np.int64)
    crop_start_accuracy = loso_position_accuracy(
        sample_histograms, crop_start_bins, all_data["subject_id"], args.seed,
    )

    test = splits["test"]
    observed_mi = lag_mutual_information(
        test["eeg_source_tokens"], test["fnirs_source_tokens"], k_eeg=k_eeg, k_fnirs=k_fnirs,
    )
    fixed_mi = {}
    for position in (0, 5, 9):
        eeg_fixed, fnirs_fixed = fixed_position_sequences(test, position)
        fixed_mi[str(position)] = (
            lag_mutual_information(eeg_fixed, fnirs_fixed, k_eeg=k_eeg, k_fnirs=k_fnirs).tolist()
            if eeg_fixed.size else []
        )
    permutation = permutation_max_mi(
        test, observed_mi, k_eeg=k_eeg, k_fnirs=k_fnirs,
        n_permutations=args.permutations, seed=args.seed,
    )

    train_counts = counts_by_lag(
        splits["train"]["eeg_source_tokens"], splits["train"]["fnirs_source_tokens"], k_eeg, k_fnirs,
    )
    conditional = conditional_probabilities_from_counts(train_counts, alpha=0.5)
    marginal = (train_counts.sum(axis=1) + 0.5)
    marginal /= marginal.sum(axis=-1, keepdims=True)
    model_nll = evaluate_nll_by_subject(conditional, test, marginal=False)
    marginal_nll = evaluate_nll_by_subject(marginal, test, marginal=True)
    bootstrap = subject_block_bootstrap_gain(
        model_nll, marginal_nll, n_bootstrap=args.bootstraps, seed=args.seed,
    )
    domain_lag_audit = {}
    train_domains = add_domains(splits["train"])
    test_domains = add_domains(test)
    for domain in ("global", "nback", "wg", "motor", "mental_arithmetic"):
        train_mask = np.ones(len(train_domains), dtype=bool) if domain == "global" else train_domains == domain
        test_mask = np.ones(len(test_domains), dtype=bool) if domain == "global" else test_domains == domain
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        train_subset = {key: value[train_mask] for key, value in splits["train"].items()}
        test_subset = {key: value[test_mask] for key, value in test.items()}
        domain_mi = lag_mutual_information(
            test_subset["eeg_source_tokens"], test_subset["fnirs_source_tokens"],
            k_eeg=k_eeg, k_fnirs=k_fnirs,
        )
        domain_permutation = permutation_max_mi(
            test_subset, domain_mi, k_eeg=k_eeg, k_fnirs=k_fnirs,
            n_permutations=args.permutations, seed=args.seed + len(domain_lag_audit),
        )
        domain_counts = counts_by_lag(
            train_subset["eeg_source_tokens"], train_subset["fnirs_source_tokens"], k_eeg, k_fnirs,
        )
        domain_conditional = conditional_probabilities_from_counts(domain_counts, alpha=0.5)
        domain_marginal = domain_counts.sum(axis=1) + 0.5
        domain_marginal /= domain_marginal.sum(axis=-1, keepdims=True)
        domain_model_nll = evaluate_nll_by_subject(domain_conditional, test_subset, marginal=False)
        domain_marginal_nll = evaluate_nll_by_subject(domain_marginal, test_subset, marginal=True)
        domain_lag_audit[domain] = {
            "mi": domain_mi.tolist(),
            "max_over_lag_permutation": domain_permutation,
            "subject_bootstrap_nll_gain": subject_block_bootstrap_gain(
                domain_model_nll, domain_marginal_nll,
                n_bootstrap=args.bootstraps, seed=args.seed,
            ),
        }

    position_report = {
        "k_eeg": k_eeg,
        "k_fnirs": k_fnirs,
        "position_labels": labels,
        "position_token_distributions": position_distributions.tolist(),
        "position_js_divergence": position_js.tolist(),
        "position_loso_accuracy": position_accuracy,
        "crop_start_loso_accuracy": crop_start_accuracy,
        "event_phase_loso_accuracy": phase_accuracy,
        "event_phase_labels": phase_labels,
        "event_phase_token_distributions": phase_distributions.tolist(),
        "task_labels": task_labels,
        "task_token_distributions": task_distributions.tolist(),
        "ordinary_window_mi": observed_mi.tolist(),
        "fixed_position_cross_window_mi": fixed_mi,
        "max_over_lag_permutation": permutation,
        "test_conditional_nll_by_subject": model_nll,
        "test_marginal_nll_by_subject": marginal_nll,
        "subject_bootstrap_nll_gain": bootstrap,
        "domain_lag_audit": domain_lag_audit,
    }
    write_json(output_dir / "position_event_lag_audit.json", position_report)

    ladder = information_ladder(all_data)
    leakage = leakage_scores(all_data, args.seed)
    write_json(output_dir / "physiological_information_ladder.json", ladder)
    rows = []
    for domain, levels in ladder.items():
        for level, metrics in levels.items():
            rows.append({
                "domain": domain,
                "level": level,
                "loso_ridge_r2": metrics["loso_ridge"]["mean"],
                "loso_cca_correlation": metrics["loso_cca_correlation"],
                "conditional_mi_gaussian": metrics["conditional_mi_gaussian"],
            })
    with (output_dir / "physiological_information_ladder.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["domain"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "summary.json", {
        "position_event_lag": position_report,
        "information_ladder": ladder,
        "leakage": leakage,
    })


if __name__ == "__main__":
    main()
