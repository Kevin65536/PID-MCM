#!/usr/bin/env python
"""Compare frozen-token coupling models F0-F6 on identical exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.analysis import (
    conditional_probabilities_from_counts,
    effective_conditional_probabilities,
    load_export_split,
    subject_block_bootstrap_gain,
)


DOMAINS = ("global", "nback", "wg", "motor", "mental_arithmetic")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def domain_names(data: Dict[str, np.ndarray]) -> np.ndarray:
    result = []
    for task, label in zip(data["source_task"], data["label_name"]):
        text = f"{task} {label}".lower()
        if "nback" in text or "n-back" in text:
            result.append("nback")
        elif label.lower() == "wg" or "word generation" in text:
            result.append("wg")
        elif "mental" in text or "arithmetic" in text or label.lower() == "math":
            result.append("mental_arithmetic")
        elif "motor" in text:
            result.append("motor")
        else:
            result.append("unknown")
    return np.asarray(result, dtype=str)


def filter_domain(data: Dict[str, np.ndarray], domain: str) -> Dict[str, np.ndarray]:
    mask = np.ones(len(data["subject_id"]), dtype=bool) if domain == "global" else domain_names(data) == domain
    return {key: value[mask] for key, value in data.items()}


def lag_count_from_max(tokens_per_window: int, max_lag_tokens: int | None) -> int:
    if max_lag_tokens is None:
        return int(tokens_per_window)
    return min(max(int(max_lag_tokens), 0) + 1, int(tokens_per_window))


def lag_counts(
    data: Dict[str, np.ndarray],
    k_eeg: int,
    k_fnirs: int,
    *,
    max_lag_tokens: int | None = None,
) -> np.ndarray:
    eeg = data["eeg_source_tokens"].astype(np.int64)
    fnirs = data["fnirs_source_tokens"].astype(np.int64)
    n_lags = lag_count_from_max(eeg.shape[1], max_lag_tokens)
    counts = np.zeros((n_lags, k_eeg, k_fnirs), dtype=np.float64)
    for lag in range(n_lags):
        valid = eeg.shape[1] - lag
        flat = eeg[:, :valid].reshape(-1) * k_fnirs + fnirs[:, lag:].reshape(-1)
        counts[lag] = np.bincount(flat, minlength=k_eeg * k_fnirs).reshape(k_eeg, k_fnirs)
    return counts


def lag_prior(counts: np.ndarray) -> np.ndarray:
    prior = counts.sum(axis=1) + 0.5
    return prior / prior.sum(axis=-1, keepdims=True)


def occupancy(counts: np.ndarray) -> np.ndarray:
    value = counts.sum(axis=-1) + 0.5
    return value / value.sum(axis=-1, keepdims=True)


def conditional_nll(probabilities: np.ndarray, counts: np.ndarray) -> float:
    per_lag = []
    for lag in range(counts.shape[0]):
        total = counts[lag].sum()
        if total > 0:
            per_lag.append(-float(np.sum(counts[lag] * np.log(np.maximum(probabilities[lag], 1e-12))) / total))
    return float(np.mean(per_lag)) if per_lag else float("nan")


def marginal_nll(prior: np.ndarray, counts: np.ndarray) -> float:
    probabilities = np.broadcast_to(prior[:, None, :], counts.shape)
    return conditional_nll(probabilities, counts)


def per_subject_nll(probabilities: np.ndarray, data: Dict[str, np.ndarray]) -> Dict[int, float]:
    result = {}
    for subject in np.unique(data["subject_id"]):
        subset = {key: value[data["subject_id"] == subject] for key, value in data.items()}
        counts = lag_counts(
            subset,
            probabilities.shape[1],
            probabilities.shape[2],
            max_lag_tokens=probabilities.shape[0] - 1,
        )
        result[int(subject)] = conditional_nll(probabilities, counts)
    return result


def subject_macro_nll(probabilities: np.ndarray, data: Dict[str, np.ndarray]) -> float:
    values = list(per_subject_nll(probabilities, data).values())
    return float(np.mean(values)) if values else float("nan")


def event_phase(value: float) -> str:
    return "pre" if value < 0 else ("early" if value < 10 else "late")


def fit_nuisance_marginal(data: Dict[str, np.ndarray], k_fnirs: int) -> Dict[str, np.ndarray]:
    counts: Dict[str, np.ndarray] = {}
    domains = domain_names(data)
    tokens = data["fnirs_source_tokens"].astype(np.int64)
    times = data["token_event_time_s"]
    for sample in range(len(tokens)):
        for position in range(tokens.shape[1]):
            keys = (
                f"{domains[sample]}|{position}|{event_phase(float(times[sample, position]))}",
                f"*|{position}|{event_phase(float(times[sample, position]))}",
                "*|*|*",
            )
            for key in keys:
                counts.setdefault(key, np.full(k_fnirs, 0.5, dtype=np.float64))[tokens[sample, position]] += 1
    return {key: value / value.sum() for key, value in counts.items()}


def nuisance_and_model_nll_by_subject(
    probabilities: np.ndarray,
    data: Dict[str, np.ndarray],
    nuisance: Dict[str, np.ndarray],
) -> tuple[Dict[int, float], Dict[int, float]]:
    domains = domain_names(data)
    eeg = data["eeg_source_tokens"].astype(np.int64)
    fnirs = data["fnirs_source_tokens"].astype(np.int64)
    times = data["token_event_time_s"]
    nuisance_loss = np.empty_like(fnirs, dtype=np.float64)
    for sample in range(len(fnirs)):
        for target_position in range(fnirs.shape[1]):
            phase = event_phase(float(times[sample, target_position]))
            key = f"{domains[sample]}|{target_position}|{phase}"
            fallback = f"*|{target_position}|{phase}"
            nuisance_probability = nuisance.get(key, nuisance.get(fallback, nuisance["*|*|*"]))
            nuisance_loss[sample, target_position] = -np.log(max(
                nuisance_probability[fnirs[sample, target_position]], 1e-12,
            ))
    model_result: Dict[int, float] = {}
    nuisance_result: Dict[int, float] = {}
    for subject in np.unique(data["subject_id"]):
        indices = np.flatnonzero(data["subject_id"] == subject)
        model_lags, nuisance_lags = [], []
        for lag in range(probabilities.shape[0]):
            valid = eeg.shape[1] - lag
            eeg_values = eeg[indices, :valid]
            fnirs_values = fnirs[indices, lag:]
            model_probability = probabilities[lag][eeg_values, fnirs_values]
            model_lags.append(float(-np.log(np.maximum(model_probability, 1e-12)).mean()))
            nuisance_lags.append(float(nuisance_loss[indices, lag:].mean()))
        model_result[int(subject)] = float(np.mean(model_lags))
        nuisance_result[int(subject)] = float(np.mean(nuisance_lags))
    return model_result, nuisance_result


def effective_q(
    logits: torch.Tensor,
    prior: torch.Tensor,
    eeg_occupancy: torch.Tensor,
    residual: bool,
) -> torch.Tensor:
    if not residual:
        return F.softmax(logits, dim=-1)
    return effective_conditional_probabilities(logits, prior, eeg_occupancy)


def empirical_gain_target(counts: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    empirical = (counts + 0.5) / (counts.sum(dim=-1, keepdim=True) + 0.5 * counts.shape[-1])
    gain = (empirical * (empirical.clamp_min(1e-8).log() - prior[:, None, :].clamp_min(1e-8).log())).sum(-1)
    return F.softmax(gain.transpose(0, 1) / 0.25, dim=-1)


def regularizer(
    model_id: str,
    logits: torch.Tensor,
    q: torch.Tensor,
    counts: torch.Tensor,
    prior: torch.Tensor,
) -> torch.Tensor:
    if model_id not in {"F5", "F6"}:
        return logits.new_zeros(())
    if model_id == "F5":
        joint = F.softmax(logits.flatten(), dim=0).reshape_as(logits)
        lag_mass = joint.sum(dim=-1).transpose(0, 1)
        lag_mass = lag_mass / lag_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        focus = -(lag_mass * lag_mass.clamp_min(1e-8).log()).sum(dim=-1).mean()
        smooth = (q[:, 1:] - q[:, :-1]).square().mean()
        target = empirical_gain_target(counts, prior)
        evidence = -(target * lag_mass.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return focus + smooth + evidence
    effective_smooth = (q[:, 1:] - q[:, :-1]).square().mean()
    interaction_energy = torch.sqrt(logits.square().sum(dim=(1, 2)) + 1e-8).sum()
    return effective_smooth + interaction_energy / math.sqrt(logits.shape[1] * logits.shape[2])


def fit_logits(
    model_id: str,
    train_counts: np.ndarray,
    val_counts: np.ndarray,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    *,
    regularization_weight: float,
    seed: int,
    steps: int,
    max_lag_tokens: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    counts = torch.tensor(train_counts, dtype=torch.float32, device=device)
    validation = torch.tensor(val_counts, dtype=torch.float32, device=device)
    validation_subject_counts = torch.stack([
        torch.tensor(
            lag_counts(
                {key: value[val_data["subject_id"] == subject] for key, value in val_data.items()},
                val_counts.shape[1],
                val_counts.shape[2],
                max_lag_tokens=max_lag_tokens,
            ),
            dtype=torch.float32,
            device=device,
        )
        for subject in np.unique(val_data["subject_id"])
    ])
    prior = torch.tensor(lag_prior(train_counts), dtype=torch.float32, device=device)
    eeg_occupancy = torch.tensor(occupancy(train_counts), dtype=torch.float32, device=device)
    validation_prior = torch.tensor(lag_prior(val_counts), dtype=torch.float32, device=device)
    validation_occupancy = torch.tensor(occupancy(val_counts), dtype=torch.float32, device=device)
    logits = torch.nn.Parameter(0.02 * torch.randn_like(counts))
    optimizer = torch.optim.Adam([logits], lr=0.05)
    best_nll = float("inf")
    best_q = None
    best_logits = None
    residual = model_id in {"F3", "F4", "F5", "F6"}
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        active_counts = counts
        active_occupancy = eeg_occupancy
        active_prior = prior
        if model_id == "F3":
            sample_count = len(train_data["subject_id"])
            selected = rng.choice(sample_count, min(256, sample_count), replace=False)
            batch_data = {key: value[selected] for key, value in train_data.items()}
            batch_counts_np = lag_counts(
                batch_data,
                train_counts.shape[1],
                train_counts.shape[2],
                max_lag_tokens=max_lag_tokens,
            )
            active_counts = torch.tensor(batch_counts_np, dtype=torch.float32, device=device)
            active_prior = torch.tensor(lag_prior(batch_counts_np), dtype=torch.float32, device=device)
            active_occupancy = torch.tensor(occupancy(batch_counts_np), dtype=torch.float32, device=device)
        q = effective_q(logits, active_prior, active_occupancy, residual=residual)
        totals = active_counts.sum(dim=(1, 2)).clamp_min(1.0)
        loss = -((active_counts * q.clamp_min(1e-8).log()).sum(dim=(1, 2)) / totals).mean()
        loss = loss + float(regularization_weight) * regularizer(model_id, logits, q, active_counts, active_prior)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            q_validation = effective_q(
                logits,
                validation_prior if model_id == "F3" else prior,
                validation_occupancy if model_id == "F3" else eeg_occupancy,
                residual=residual,
            )
            val_totals = validation_subject_counts.sum(dim=(2, 3)).clamp_min(1.0)
            val_by_subject_lag = -(
                validation_subject_counts * q_validation.clamp_min(1e-8).log().unsqueeze(0)
            ).sum(dim=(2, 3)) / val_totals
            val_nll = float(val_by_subject_lag.mean())
            if val_nll < best_nll:
                best_nll = val_nll
                best_q = q_validation.detach().cpu().numpy().copy()
                best_logits = logits.detach().cpu().numpy().copy()
    if best_q is None:
        raise RuntimeError("Frozen coupling optimization failed")
    return best_q, best_logits, best_nll


def factorized_residual_logits(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ler,lfr->lef", left, right) / math.sqrt(max(int(left.shape[-1]), 1))


def fit_factorized_logits(
    train_counts: np.ndarray,
    val_counts: np.ndarray,
    val_data: Dict[str, np.ndarray],
    *,
    rank: int,
    seed: int,
    steps: int,
    max_lag_tokens: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    counts = torch.tensor(train_counts, dtype=torch.float32, device=device)
    validation_subject_counts = torch.stack([
        torch.tensor(
            lag_counts(
                {key: value[val_data["subject_id"] == subject] for key, value in val_data.items()},
                val_counts.shape[1],
                val_counts.shape[2],
                max_lag_tokens=max_lag_tokens,
            ),
            dtype=torch.float32,
            device=device,
        )
        for subject in np.unique(val_data["subject_id"])
    ])
    prior = torch.tensor(lag_prior(train_counts), dtype=torch.float32, device=device)
    eeg_occupancy = torch.tensor(occupancy(train_counts), dtype=torch.float32, device=device)
    left = torch.nn.Parameter(0.02 * torch.randn(
        train_counts.shape[0], train_counts.shape[1], int(rank), device=device,
    ))
    right = torch.nn.Parameter(0.02 * torch.randn(
        train_counts.shape[0], train_counts.shape[2], int(rank), device=device,
    ))
    optimizer = torch.optim.Adam([left, right], lr=0.05)
    best_nll = float("inf")
    best_q = None
    best_logits = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = factorized_residual_logits(left, right)
        q = effective_q(logits, prior, eeg_occupancy, residual=True)
        totals = counts.sum(dim=(1, 2)).clamp_min(1.0)
        loss = -((counts * q.clamp_min(1e-8).log()).sum(dim=(1, 2)) / totals).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits_validation = factorized_residual_logits(left, right)
            q_validation = effective_q(logits_validation, prior, eeg_occupancy, residual=True)
            val_totals = validation_subject_counts.sum(dim=(2, 3)).clamp_min(1.0)
            val_by_subject_lag = -(
                validation_subject_counts * q_validation.clamp_min(1e-8).log().unsqueeze(0)
            ).sum(dim=(2, 3)) / val_totals
            val_nll = float(val_by_subject_lag.mean())
            if val_nll < best_nll:
                best_nll = val_nll
                best_q = q_validation.detach().cpu().numpy().copy()
                best_logits = logits_validation.detach().cpu().numpy().copy()
    if best_q is None:
        raise RuntimeError("Frozen factorized coupling optimization failed")
    return best_q, best_logits, best_nll


def mean_pairwise_js(probabilities: Iterable[np.ndarray]) -> float:
    values = list(probabilities)
    scores = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            left = np.maximum(values[i], 1e-12)
            right = np.maximum(values[j], 1e-12)
            middle = 0.5 * (left + right)
            js = 0.5 * np.sum(left * np.log(left / middle), axis=-1) + 0.5 * np.sum(right * np.log(right / middle), axis=-1)
            scores.append(float(js.mean()))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_candidates(
    candidates: Dict[str, list[np.ndarray]],
    hyperparameters: Dict[str, Any],
    *,
    train_domain: str,
    train_counts: np.ndarray,
    prior: np.ndarray,
    splits: Dict[str, Dict[str, np.ndarray]],
    k_eeg: int,
    k_fnirs: int,
    nuisance_marginal: Dict[str, np.ndarray],
    bootstraps: int,
    max_lag_tokens: int | None,
    output_dir: Path,
    rows: list[Dict[str, Any]],
) -> Dict[str, Any]:
    domain_results: Dict[str, Any] = {}
    for model_id, distributions in candidates.items():
        probability = np.mean(distributions, axis=0)
        test_matrix = {}
        for test_domain in DOMAINS:
            test = filter_domain(splits["test"], test_domain)
            if len(test["subject_id"]) == 0:
                continue
            test_counts = lag_counts(test, k_eeg, k_fnirs, max_lag_tokens=max_lag_tokens)
            if model_id == "F3":
                mean_logits = np.mean(hyperparameters[model_id]["fitted_logits"], axis=0)
                probability = effective_q(
                    torch.tensor(mean_logits, dtype=torch.float32),
                    torch.tensor(lag_prior(test_counts), dtype=torch.float32),
                    torch.tensor(occupancy(test_counts), dtype=torch.float32),
                    residual=True,
                ).detach().cpu().numpy()
            model_subject = per_subject_nll(probability, test)
            baseline_probability = np.broadcast_to(prior[:, None, :], probability.shape)
            baseline_subject = per_subject_nll(baseline_probability, test)
            nll = float(np.mean(list(model_subject.values())))
            baseline = float(np.mean(list(baseline_subject.values())))
            bootstrap = subject_block_bootstrap_gain(
                model_subject, baseline_subject,
                n_bootstrap=bootstraps, seed=20260615,
            )
            nuisance_model_subject, nuisance_subject = nuisance_and_model_nll_by_subject(
                probability, test, nuisance_marginal,
            )
            nuisance_bootstrap = subject_block_bootstrap_gain(
                nuisance_model_subject, nuisance_subject,
                n_bootstrap=bootstraps, seed=20260615,
            )
            nuisance_nll = float(np.mean(list(nuisance_subject.values())))
            nuisance_model_nll = float(np.mean(list(nuisance_model_subject.values())))
            test_matrix[test_domain] = {
                "nll": nll,
                "marginal_nll": baseline,
                "nll_gain": baseline - nll,
                "subject_bootstrap_gain": bootstrap,
                "nuisance_marginal_nll": nuisance_nll,
                "nuisance_adjusted_model_nll": nuisance_model_nll,
                "nuisance_adjusted_nll_gain": nuisance_nll - nuisance_model_nll,
                "nuisance_subject_bootstrap_gain": nuisance_bootstrap,
            }
            rows.append({
                "train_domain": train_domain,
                "test_domain": test_domain,
                "model": model_id,
                "nll": nll,
                "marginal_nll": baseline,
                "nll_gain": baseline - nll,
                "bootstrap_ci_low": bootstrap["ci_low"],
                "nuisance_adjusted_gain": nuisance_nll - nuisance_model_nll,
                "nuisance_bootstrap_ci_low": nuisance_bootstrap["ci_low"],
                "initialization_js": hyperparameters[model_id].get("initialization_js", 0.0),
            })
        domain_results[model_id] = {
            "hyperparameters": {
                key: value for key, value in hyperparameters[model_id].items()
                if key != "fitted_logits"
            },
            "test_matrix": test_matrix,
        }
        np.save(output_dir / f"{train_domain}_{model_id}_probabilities.npy", probability)
        if model_id == "F6":
            residual = np.log(np.maximum(probability, 1e-12)) - np.log(np.maximum(prior[:, None, :], 1e-12))
            eeg_occupancy = occupancy(train_counts)
            residual -= np.einsum("le,lef->lf", eeg_occupancy, residual)[:, None, :]
            np.savez(
                output_dir / f"{train_domain}_F6_parameters.npz",
                probabilities=probability,
                residual_logits=residual,
                fnirs_prior=prior,
                eeg_occupancy=eeg_occupancy,
            )
    return domain_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--position-audit")
    parser.add_argument("--max-lag-tokens", type=int, default=None)
    parser.add_argument("--factorized-only", action="store_true")
    parser.add_argument("--factorized-ranks", nargs="+", type=int, default=[16, 32])
    args = parser.parse_args()
    export_dir = Path(args.export_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {name: load_export_split(export_dir, name) for name in ("train", "val", "test")}
    k_eeg = max(int(data["eeg_source_tokens"].max()) for data in splits.values()) + 1
    k_fnirs = max(int(data["fnirs_source_tokens"].max()) for data in splits.values()) + 1
    seeds = (20260615, 20260616, 20260617)
    weights = (0.0, 0.01, 0.03, 0.1)
    alpha_grid = (0.1, 0.5, 1.0, 2.0, 5.0)
    results: Dict[str, Any] = {}
    rows = []

    for train_domain in DOMAINS:
        train = filter_domain(splits["train"], train_domain)
        val = filter_domain(splits["val"], train_domain)
        if len(train["subject_id"]) == 0 or len(val["subject_id"]) == 0:
            continue
        train_counts = lag_counts(train, k_eeg, k_fnirs, max_lag_tokens=args.max_lag_tokens)
        val_counts = lag_counts(val, k_eeg, k_fnirs, max_lag_tokens=args.max_lag_tokens)
        prior = lag_prior(train_counts)
        nuisance_marginal = fit_nuisance_marginal(train, k_fnirs)
        domain_results: Dict[str, Any] = {}

        f0 = np.broadcast_to(prior[:, None, :], train_counts.shape).copy()
        if args.factorized_only:
            candidates = {"F0": [f0]}
            hyperparameters: Dict[str, Any] = {"F0": {}}
            for rank in args.factorized_ranks:
                distributions, logits_by_seed, scores = [], [], []
                for seed in seeds:
                    distribution, fitted_logits, score = fit_factorized_logits(
                        train_counts,
                        val_counts,
                        val,
                        rank=rank,
                        seed=seed,
                        steps=args.steps,
                        max_lag_tokens=args.max_lag_tokens,
                    )
                    distributions.append(distribution)
                    logits_by_seed.append(fitted_logits)
                    scores.append(score)
                model_id = f"LR{int(rank)}"
                candidates[model_id] = distributions
                hyperparameters[model_id] = {
                    "rank": int(rank),
                    "validation_nll": float(np.mean(scores)),
                    "initialization_js": mean_pairwise_js(distributions),
                    "fitted_logits": logits_by_seed,
                }
            results[train_domain] = evaluate_candidates(
                candidates,
                hyperparameters,
                train_domain=train_domain,
                train_counts=train_counts,
                prior=prior,
                splits=splits,
                k_eeg=k_eeg,
                k_fnirs=k_fnirs,
                nuisance_marginal=nuisance_marginal,
                bootstraps=args.bootstraps,
                max_lag_tokens=args.max_lag_tokens,
                output_dir=output_dir,
                rows=rows,
            )
            continue

        alpha_scores = {}
        for alpha in alpha_grid:
            candidate = conditional_probabilities_from_counts(train_counts, alpha=alpha, prior=prior)
            alpha_scores[str(alpha)] = subject_macro_nll(candidate, val)
        best_alpha = min(alpha_scores, key=alpha_scores.get)
        f1 = conditional_probabilities_from_counts(train_counts, alpha=float(best_alpha), prior=prior)
        candidates = {"F0": [f0], "F1": [f1]}
        hyperparameters: Dict[str, Any] = {
            "F0": {}, "F1": {"alpha": float(best_alpha), "validation_nll": alpha_scores[best_alpha]},
        }

        for model_id in ("F2", "F3", "F4", "F5", "F6"):
            best_weight = 0.0
            best_score = float("inf")
            best_distributions = []
            model_weights = weights if model_id in {"F5", "F6"} else (0.0,)
            best_logits = []
            for weight in model_weights:
                distributions, logits_by_seed, scores = [], [], []
                for seed in seeds:
                    distribution, fitted_logits, score = fit_logits(
                        model_id, train_counts, val_counts,
                        train, val,
                        regularization_weight=weight, seed=seed, steps=args.steps,
                        max_lag_tokens=args.max_lag_tokens,
                    )
                    distributions.append(distribution)
                    logits_by_seed.append(fitted_logits)
                    scores.append(score)
                score = float(np.mean(scores))
                if score < best_score:
                    best_score = score
                    best_weight = weight
                    best_distributions = distributions
                    best_logits = logits_by_seed
            candidates[model_id] = best_distributions
            hyperparameters[model_id] = {
                "regularization_weight": best_weight,
                "validation_nll": best_score,
                "initialization_js": mean_pairwise_js(best_distributions),
            }
            hyperparameters[model_id]["fitted_logits"] = best_logits

        for model_id, distributions in candidates.items():
            probability = np.mean(distributions, axis=0)
            test_matrix = {}
            for test_domain in DOMAINS:
                test = filter_domain(splits["test"], test_domain)
                if len(test["subject_id"]) == 0:
                    continue
                test_counts = lag_counts(test, k_eeg, k_fnirs, max_lag_tokens=args.max_lag_tokens)
                if model_id == "F3":
                    mean_logits = np.mean(hyperparameters[model_id]["fitted_logits"], axis=0)
                    probability = effective_q(
                        torch.tensor(mean_logits, dtype=torch.float32),
                        torch.tensor(lag_prior(test_counts), dtype=torch.float32),
                        torch.tensor(occupancy(test_counts), dtype=torch.float32),
                        residual=True,
                    ).detach().cpu().numpy()
                model_subject = per_subject_nll(probability, test)
                baseline_probability = np.broadcast_to(prior[:, None, :], probability.shape)
                baseline_subject = per_subject_nll(baseline_probability, test)
                nll = float(np.mean(list(model_subject.values())))
                baseline = float(np.mean(list(baseline_subject.values())))
                bootstrap = subject_block_bootstrap_gain(
                    model_subject, baseline_subject,
                    n_bootstrap=args.bootstraps, seed=20260615,
                )
                nuisance_model_subject, nuisance_subject = nuisance_and_model_nll_by_subject(
                    probability, test, nuisance_marginal,
                )
                nuisance_bootstrap = subject_block_bootstrap_gain(
                    nuisance_model_subject, nuisance_subject,
                    n_bootstrap=args.bootstraps, seed=20260615,
                )
                nuisance_nll = float(np.mean(list(nuisance_subject.values())))
                nuisance_model_nll = float(np.mean(list(nuisance_model_subject.values())))
                test_matrix[test_domain] = {
                    "nll": nll,
                    "marginal_nll": baseline,
                    "nll_gain": baseline - nll,
                    "subject_bootstrap_gain": bootstrap,
                    "nuisance_marginal_nll": nuisance_nll,
                    "nuisance_adjusted_model_nll": nuisance_model_nll,
                    "nuisance_adjusted_nll_gain": nuisance_nll - nuisance_model_nll,
                    "nuisance_subject_bootstrap_gain": nuisance_bootstrap,
                }
                rows.append({
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    "model": model_id,
                    "nll": nll,
                    "marginal_nll": baseline,
                    "nll_gain": baseline - nll,
                    "bootstrap_ci_low": bootstrap["ci_low"],
                    "nuisance_adjusted_gain": nuisance_nll - nuisance_model_nll,
                    "nuisance_bootstrap_ci_low": nuisance_bootstrap["ci_low"],
                    "initialization_js": hyperparameters[model_id].get("initialization_js", 0.0),
                })
            domain_results[model_id] = {
                "hyperparameters": {
                    key: value for key, value in hyperparameters[model_id].items()
                    if key != "fitted_logits"
                },
                "test_matrix": test_matrix,
            }
            np.save(output_dir / f"{train_domain}_{model_id}_probabilities.npy", probability)
            if model_id == "F6":
                residual = np.log(np.maximum(probability, 1e-12)) - np.log(np.maximum(prior[:, None, :], 1e-12))
                eeg_occupancy = occupancy(train_counts)
                residual -= np.einsum("le,lef->lf", eeg_occupancy, residual)[:, None, :]
                np.savez(
                    output_dir / f"{train_domain}_F6_parameters.npz",
                    probabilities=probability,
                    residual_logits=residual,
                    fnirs_prior=prior,
                    eeg_occupancy=eeg_occupancy,
                )
        results[train_domain] = domain_results

    f6_global = results.get("global", {}).get("F6", {})
    f6_tests = f6_global.get("test_matrix", {})
    stable = f6_global.get("hyperparameters", {}).get("initialization_js", float("inf")) < 0.005
    positive_global = f6_tests.get("global", {}).get("nuisance_subject_bootstrap_gain", {}).get("ci_low", -float("inf")) > 0
    positive_task = any(
        metrics.get("nuisance_subject_bootstrap_gain", {}).get("ci_low", -float("inf")) > 0
        for domain, metrics in f6_tests.items() if domain != "global"
    )
    lag_corrected = False
    if args.position_audit:
        position_payload = json.loads(Path(args.position_audit).read_text(encoding="utf-8"))
        for domain, audit in position_payload.get("domain_lag_audit", {}).items():
            if domain == "global":
                continue
            corrected = audit.get("max_over_lag_permutation", {}).get("corrected_p", [])
            gain_low = audit.get("subject_bootstrap_nll_gain", {}).get("ci_low", -float("inf"))
            if corrected and min(corrected) < 0.05 and gain_low > 0:
                lag_corrected = True
                break
    decision = {
        "f6_promoted": bool(stable and positive_global and positive_task and lag_corrected),
        "criteria": {
            "initialization_js_below_0_005": stable,
            "global_bootstrap_nll_gain_ci_low_above_zero": positive_global,
            "at_least_one_task_bootstrap_gain_ci_low_above_zero": positive_task,
            "lag_corrected_loso_gain": lag_corrected,
            "raw_logit_concentration_not_required": True,
        },
    }
    write_json(output_dir / "frozen_model_results.json", results)
    write_json(output_dir / "decision.json", decision)
    with (output_dir / "model_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
