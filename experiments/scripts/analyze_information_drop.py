#!/usr/bin/env python
"""Audit cross-modal information drop from continuous latents to hard tokens."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.analysis import (
    gaussian_conditional_mutual_information,
    load_export_split,
    loso_ridge_scores,
)
from src.tokenizers import create_tokenizer
from src.utils import load_checkpoint_file


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def domain_names(data: dict[str, np.ndarray]) -> np.ndarray:
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


def event_phase(times: np.ndarray) -> np.ndarray:
    return np.where(times < 0, "pre", np.where(times < 10, "early", "late"))


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
        model = CCA(n_components=components, max_iter=500)
        model.fit(xs.transform(x[train]), ys.transform(y[train]))
        x_proj, y_proj = model.transform(xs.transform(x[test]), ys.transform(y[test]))
        correlations = [np.corrcoef(x_proj[:, i], y_proj[:, i])[0, 1] for i in range(components)]
        scores.append(float(np.nanmean(correlations)))
    return float(np.nanmean(scores)) if scores else float("nan")


def flatten_level(data: dict[str, np.ndarray], left_key: str, right_key: str, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = data[left_key].reshape(-1, data[left_key].shape[-1])[indices].astype(np.float32)
    right = data[right_key].reshape(-1, data[right_key].shape[-1])[indices].astype(np.float32)
    return left, right


def one_hot(tokens: np.ndarray, k: int, indices: np.ndarray) -> np.ndarray:
    flat = tokens.reshape(-1)[indices].astype(np.int64)
    return np.eye(k, dtype=np.float32)[flat]


def codebook_weight(model: torch.nn.Module, branch: str) -> np.ndarray:
    return getattr(model, f"{branch}_quantizer").get_codebook_weight().detach().cpu().float().numpy()


def soft_assignment(latent: np.ndarray, weight: np.ndarray, indices: np.ndarray, temperature: float) -> np.ndarray:
    values = latent.reshape(-1, latent.shape[-1])[indices].astype(np.float32)
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    weights = weight / np.maximum(np.linalg.norm(weight, axis=1, keepdims=True), 1e-12)
    logits = values @ weights.T / max(float(temperature), 1e-3)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs.astype(np.float32)


def quantized_embedding(tokens: np.ndarray, weight: np.ndarray, indices: np.ndarray) -> np.ndarray:
    flat = tokens.reshape(-1)[indices].astype(np.int64)
    return weight[flat].astype(np.float32)


def score_level(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    nuisance: np.ndarray,
) -> dict[str, Any]:
    return {
        "loso_ridge": loso_ridge_scores(x, y, subjects),
        "loso_cca_correlation": cca_loso(x, y, subjects),
        "conditional_mi_gaussian": gaussian_conditional_mutual_information(x, y, nuisance),
    }


def analyze_split(
    data: dict[str, np.ndarray],
    model: torch.nn.Module,
    *,
    split: str,
    max_tokens: int,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    total_tokens = int(data["eeg_source_tokens"].size)
    selected = np.arange(total_tokens)
    if total_tokens > max_tokens:
        selected = np.sort(rng.choice(total_tokens, size=max_tokens, replace=False))
    token_per_sample = data["eeg_source_tokens"].shape[1]
    sample_indices = selected // token_per_sample
    subjects = np.repeat(data["subject_id"], token_per_sample)[selected]
    domains = np.repeat(domain_names(data), token_per_sample)[selected]
    nuisance = np.stack([
        domains,
        event_phase(data["token_event_time_s"].reshape(-1)[selected]),
        data["token_relative_position"].reshape(-1)[selected].astype(str),
    ], axis=1)

    eeg_source_weight = codebook_weight(model, "eeg_source")
    fnirs_source_weight = codebook_weight(model, "fnirs_source")
    eeg_k = eeg_source_weight.shape[0]
    fnirs_k = fnirs_source_weight.shape[0]
    levels: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "raw_patch_features": flatten_level(data, "eeg_raw_features", "fnirs_raw_features", selected),
        "source_target_features": flatten_level(data, "eeg_source_features", "fnirs_source_features", selected),
        "observation_target_features": flatten_level(data, "eeg_observation_features", "fnirs_observation_features", selected),
        "continuous_source_latent": flatten_level(data, "eeg_source_latent", "fnirs_source_latent", selected),
        "quantizer_input": flatten_level(data, "eeg_source_latent", "fnirs_source_latent", selected),
        "soft_assignment": (
            soft_assignment(data["eeg_source_latent"], eeg_source_weight, selected, temperature),
            soft_assignment(data["fnirs_source_latent"], fnirs_source_weight, selected, temperature),
        ),
        "hard_token_onehot": (
            one_hot(data["eeg_source_tokens"], eeg_k, selected),
            one_hot(data["fnirs_source_tokens"], fnirs_k, selected),
        ),
        "quantized_embedding": (
            quantized_embedding(data["eeg_source_tokens"], eeg_source_weight, selected),
            quantized_embedding(data["fnirs_source_tokens"], fnirs_source_weight, selected),
        ),
    }
    result: dict[str, Any] = {
        "split": split,
        "sample_count": int(len(np.unique(sample_indices))),
        "token_count": int(len(selected)),
        "levels": {},
        "domains": {},
    }
    for name, (x, y) in levels.items():
        result["levels"][name] = score_level(x, y, subjects, nuisance)
    for domain in ("global", "nback", "wg", "motor", "mental_arithmetic"):
        mask = np.ones(len(selected), dtype=bool) if domain == "global" else domains == domain
        if mask.sum() < 500:
            continue
        result["domains"][domain] = {
            name: score_level(x[mask], y[mask], subjects[mask], nuisance[mask])
            for name, (x, y) in levels.items()
        }

    continuous = result["levels"]["continuous_source_latent"]["conditional_mi_gaussian"]
    soft = result["levels"]["soft_assignment"]["conditional_mi_gaussian"]
    hard = result["levels"]["hard_token_onehot"]["conditional_mi_gaussian"]
    result["summary"] = {
        "continuous_to_soft_retention": float(soft / continuous) if continuous > 0 else float("nan"),
        "soft_to_hard_drop": float(soft - hard),
        "hard_token_nuisance_adjusted_gain": float(hard),
        "task_stratified_retention": {
            domain: (
                float(values["hard_token_onehot"]["conditional_mi_gaussian"] /
                      max(values["continuous_source_latent"]["conditional_mi_gaussian"], 1e-12))
                if "hard_token_onehot" in values else float("nan")
            )
            for domain, values in result["domains"].items()
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = load_checkpoint_file(Path(args.checkpoint).resolve(), device="cpu")
    model = create_tokenizer(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "continuous_to_discrete_information_drop_v1",
        "run_name": args.run_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "export_dir": str(Path(args.export_dir).resolve()),
        "splits": {},
    }
    rows = []
    for split in args.splits:
        data = load_export_split(args.export_dir, split)
        split_result = analyze_split(
            data,
            model,
            split=split,
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
        )
        payload["splits"][split] = split_result
        for level, metrics in split_result["levels"].items():
            rows.append({
                "split": split,
                "domain": "global",
                "level": level,
                "loso_ridge_mean": metrics["loso_ridge"]["mean"],
                "loso_cca_correlation": metrics["loso_cca_correlation"],
                "conditional_mi_gaussian": metrics["conditional_mi_gaussian"],
            })
        for domain, domain_values in split_result["domains"].items():
            if domain == "global":
                continue
            for level, metrics in domain_values.items():
                rows.append({
                    "split": split,
                    "domain": domain,
                    "level": level,
                    "loso_ridge_mean": metrics["loso_ridge"]["mean"],
                    "loso_cca_correlation": metrics["loso_cca_correlation"],
                    "conditional_mi_gaussian": metrics["conditional_mi_gaussian"],
                })
    write_json(output_dir / "summary.json", payload)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["splits"], indent=2, default=float))


if __name__ == "__main__":
    main()
