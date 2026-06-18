#!/usr/bin/env python
"""Analyze source/observation tokenizer codebook geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.tokenizers import create_tokenizer
from src.utils import load_checkpoint_file


BRANCHES = ("eeg_source", "fnirs_source", "eeg_observation", "fnirs_observation")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def codebook_weight(model: torch.nn.Module, branch: str) -> np.ndarray:
    quantizer = getattr(model, f"{branch}_quantizer")
    return quantizer.get_codebook_weight().detach().cpu().float().numpy()


def effective_rank(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    if float(energy.sum()) <= 0:
        return 0.0
    probs = energy / energy.sum()
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return float(np.exp(entropy))


def pca2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T if vt.shape[0] >= 2 else np.pad(centered @ vt[:1].T, ((0, 0), (0, 1)))
    variance = np.square(singular)
    ratio = variance / max(float(variance.sum()), 1e-12)
    return coords, ratio[: min(8, len(ratio))]


def plot_branch(output_dir: Path, run_name: str, branch: str, weight: np.ndarray) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = weight / np.maximum(np.linalg.norm(weight, axis=1, keepdims=True), 1e-12)
    cosine = normalized @ normalized.T
    coords, ratio = pca2(weight)
    norms = np.linalg.norm(weight, axis=1)

    cosine_path = output_dir / f"{run_name}_{branch}_codebook_cosine_matrix.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cosine, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(f"{branch} codebook cosine")
    ax.set_xlabel("token")
    ax.set_ylabel("token")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(cosine_path, dpi=160)
    plt.close(fig)

    pca_path = output_dir / f"{run_name}_{branch}_prototype_pca.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=14)
    ax.set_title(f"{branch} prototype PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.text(0.02, 0.98, f"var={ratio[:2].sum():.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(pca_path, dpi=160)
    plt.close(fig)

    norm_path = output_dir / f"{run_name}_{branch}_prototype_norm_usage.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(norms)), norms, marker=".", linewidth=1)
    ax.set_title(f"{branch} prototype norms")
    ax.set_xlabel("token")
    ax.set_ylabel("L2 norm")
    fig.tight_layout()
    fig.savefig(norm_path, dpi=160)
    plt.close(fig)

    return {
        "cosine_matrix": str(cosine_path),
        "prototype_pca": str(pca_path),
        "prototype_norm_usage": str(norm_path),
    }


def branch_metrics(weight: np.ndarray) -> dict[str, Any]:
    normalized = weight / np.maximum(np.linalg.norm(weight, axis=1, keepdims=True), 1e-12)
    cosine = normalized @ normalized.T
    off_diag = cosine[~np.eye(cosine.shape[0], dtype=bool)]
    coords, ratio = pca2(weight)
    norms = np.linalg.norm(weight, axis=1)
    nearest = np.partition(cosine + np.eye(cosine.shape[0]) * -2.0, -1, axis=1)[:, -1]
    return {
        "codebook_size": int(weight.shape[0]),
        "dim": int(weight.shape[1]),
        "effective_rank": effective_rank(weight),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "cosine_offdiag_mean": float(off_diag.mean()) if off_diag.size else 0.0,
        "cosine_offdiag_std": float(off_diag.std()) if off_diag.size else 0.0,
        "nearest_cosine_mean": float(nearest.mean()) if nearest.size else 0.0,
        "nearest_cosine_p95": float(np.quantile(nearest, 0.95)) if nearest.size else 0.0,
        "pca_variance_ratio": ratio.tolist(),
        "pca_first2_variance": float(ratio[:2].sum()) if ratio.size else 0.0,
    }


def alignment_metrics(eeg_weight: np.ndarray, fnirs_weight: np.ndarray) -> dict[str, Any]:
    common = min(eeg_weight.shape[1], fnirs_weight.shape[1])
    left = eeg_weight[:, :common]
    right = fnirs_weight[:, :common]
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(left.T @ right, full_matrices=False)
    rotated = left @ (u @ vt)
    left_norm = rotated / np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    similarity = left_norm @ right_norm.T
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-similarity)
    except Exception:
        rows = np.arange(similarity.shape[0])
        cols = np.argmax(similarity, axis=1)
    matched = similarity[rows, cols]
    return {
        "shared_dim": int(common),
        "mean_best_cosine": float(similarity.max(axis=1).mean()),
        "hungarian_mean_cosine": float(matched.mean()) if matched.size else 0.0,
        "hungarian_median_cosine": float(np.median(matched)) if matched.size else 0.0,
        "matched_pairs": int(len(matched)),
    }


def plot_alignment(output_dir: Path, run_name: str, eeg_weight: np.ndarray, fnirs_weight: np.ndarray) -> str:
    common = min(eeg_weight.shape[1], fnirs_weight.shape[1])
    left = eeg_weight[:, :common]
    right = fnirs_weight[:, :common]
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    similarity = left @ right.T
    path = output_dir / f"{run_name}_eeg_fnirs_geometry_alignment.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(similarity, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title("EEG source x fNIRS source prototype cosine")
    ax.set_xlabel("fNIRS source token")
    ax.set_ylabel("EEG source token")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint = load_checkpoint_file(checkpoint_path, device="cpu")
    model = create_tokenizer(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    weights = {branch: codebook_weight(model, branch) for branch in BRANCHES}
    artifacts = {}
    metrics = {}
    rows = []
    for branch, weight in weights.items():
        artifacts[branch] = plot_branch(output_dir, args.run_name, branch, weight)
        metrics[branch] = branch_metrics(weight)
        rows.append({"branch": branch, **metrics[branch]})
    metrics["eeg_fnirs_source_alignment"] = alignment_metrics(weights["eeg_source"], weights["fnirs_source"])
    artifacts["eeg_fnirs_source_alignment"] = plot_alignment(
        output_dir,
        args.run_name,
        weights["eeg_source"],
        weights["fnirs_source"],
    )
    payload = {
        "schema_version": "codebook_geometry_v1",
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "artifacts": artifacts,
    }
    write_json(output_dir / "summary.json", payload)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
