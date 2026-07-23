#!/usr/bin/env python3
"""Build a visual, validation-only E1 tokenizer health and coupling audit.

The script deliberately keeps the protected test split closed.  It combines the
registered E1 occupancy summary with checkpoint-level codebook geometry and, for
an explicitly selected lineage of runs, replays the validation loader to measure
paired EEG/fNIRS token association against a within-subject-and-label permutation
null.  The association outputs are diagnostic traces, not an independent
neurovascular-coupling certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.factory import create_configured_multimodal_dataloaders
from src.tokenizers.registry import create_tokenizer
import src.tokenizers  # noqa: F401  (register tokenizer implementations)


SCHEMA = "e1_tokenizer_health_and_coupling_audit_v1"
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#6B7280",
}

DEFAULT_COUPLING_RUNS = (
    "20260719_e1_t0_semantic_only_short_formal_v2",
    "20260720_e1_t0_gradient_balance_normalized_annealed_hard_short_formal_v12",
    "20260720_e1_t0_archived_revival_bundle_short_formal_v14",
    "20260720_e1_t0_balance_temperature2_short_formal_v17",
    "20260720_e1_t0_post_revival_retention_seed20260721_v21",
    "20260720_e1_t0_post_revival_diverse_farthest_seed20260721_v22",
    "20260720_e1_t0_post_revival_diverse_farthest_seed20260719_v23",
    "20260720_e1_t0_post_revival_diverse_farthest_seed20260720_v23",
)

FINAL_GATE_RUNS = (
    "20260720_e1_t0_post_revival_diverse_farthest_seed20260721_v22",
    "20260720_e1_t0_post_revival_diverse_farthest_seed20260719_v23",
    "20260720_e1_t0_post_revival_diverse_farthest_seed20260720_v23",
)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for suffix in ("svg", "pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths[suffix] = str(path.resolve())
    plt.close(fig)
    return paths


def short_label(run_name: str) -> str:
    match = re.search(r"_v(\d+)$", run_name)
    if match:
        label = f"v{match.group(1)}"
    else:
        label = run_name[-12:]
    seed = re.search(r"seed(\d+)", run_name)
    if seed:
        label += f"/s{seed.group(1)[-2:]}"
    return label


def _entropy(probability: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    positive = probability[probability > 0]
    return float(-(positive * np.log(positive)).sum()) if positive.size else 0.0


def _gini(probability: np.ndarray) -> float:
    values = np.sort(np.asarray(probability, dtype=np.float64))
    total = float(values.sum())
    if values.size == 0 or total <= 0:
        return 0.0
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    return float((2.0 * np.dot(ranks, values) / total - (values.size + 1)) / values.size)


def distribution_metrics(counts: np.ndarray) -> dict[str, float | int]:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    probability = counts / total if total > 0 else np.zeros_like(counts)
    entropy = _entropy(probability)
    sorted_probability = np.sort(probability)[::-1]
    size = int(counts.size)
    return {
        "total": total,
        "active_codes": int(np.count_nonzero(counts > 0)),
        "active_fraction": float(np.mean(counts > 0)) if size else 0.0,
        "effective_codes": float(np.exp(entropy)),
        "entropy_nats": entropy,
        "entropy_ratio": float(entropy / math.log(size)) if size > 1 else 0.0,
        "gini": _gini(probability),
        "top1_mass": float(sorted_probability[:1].sum()),
        "top5_mass": float(sorted_probability[:5].sum()),
        "top10_mass": float(sorted_probability[:10].sum()),
    }


def checkpoint_geometry(checkpoint_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for modality in ("eeg", "fnirs"):
        prefix = f"{modality}_branch.quantizer."
        codebook = state[prefix + "codebook"].detach().float().cpu().numpy().astype(np.float64)
        counts = state[prefix + "ema_count"].detach().float().cpu().numpy().astype(np.float64)
        norms = np.linalg.norm(codebook, axis=1, keepdims=True)
        normalized = codebook / np.clip(norms, 1e-12, None)
        cosine = normalized @ normalized.T
        np.fill_diagonal(cosine, -np.inf)
        nearest = cosine.max(axis=1)
        finite_pairs = cosine[np.isfinite(cosine)]
        singular = np.linalg.svd(codebook - codebook.mean(axis=0, keepdims=True), compute_uv=False)
        squared = singular**2
        spectral_probability = squared / np.clip(squared.sum(), 1e-12, None)
        participation_rank = float(squared.sum() ** 2 / np.clip((squared**2).sum(), 1e-12, None))
        spectral_entropy_rank = float(np.exp(_entropy(spectral_probability)))
        occupancy = distribution_metrics(counts)
        rows.append(
            {
                "modality": modality,
                "checkpoint": str(checkpoint_path.resolve()),
                "codebook_size": int(codebook.shape[0]),
                "embedding_dim": int(codebook.shape[1]),
                "ema_effective_codes": occupancy["effective_codes"],
                "ema_entropy_ratio": occupancy["entropy_ratio"],
                "ema_gini": occupancy["gini"],
                "ema_top10_mass": occupancy["top10_mass"],
                "nearest_cosine_mean": float(nearest.mean()),
                "nearest_cosine_q95": float(np.quantile(nearest, 0.95)),
                "nearest_cosine_max": float(nearest.max()),
                "pairwise_cosine_mean": float(finite_pairs.mean()),
                "duplicate_pairs_cosine_ge_0p99": int(np.triu(cosine >= 0.99, k=1).sum()),
                "participation_rank": participation_rank,
                "spectral_entropy_rank": spectral_entropy_rank,
                "pc1_variance_ratio": float(spectral_probability[0]),
                "pc1_pc2_variance_ratio": float(spectral_probability[:2].sum()),
                "matrix_rank": int(np.linalg.matrix_rank(codebook)),
            }
        )
        arrays[modality] = {
            "codebook": codebook,
            "ema_count": counts,
            "nearest_cosine": nearest,
        }
    return rows, arrays


def _pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ right[:2].T
    variance = singular**2
    ratio = variance[:2] / np.clip(variance.sum(), 1e-12, None)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
        ratio = np.pad(ratio, (0, 2 - ratio.shape[0]))
    return coords[:, :2], ratio[:2]


def _batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _batch_to_device(item, device) for key, item in value.items()}
    return value


def _load_checkpoint_model_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load E1 checkpoints while allowing only known post-v2 scalar buffers."""
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "eeg_branch.quantizer.initialized",
        "eeg_branch.quantizer.quantization_strength",
        "fnirs_branch.quantizer.initialized",
        "fnirs_branch.quantizer.quantization_strength",
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint incompatibility outside the registered E1 buffer migration: "
            f"missing={sorted(unexpected_missing)}, unexpected={sorted(incompatible.unexpected_keys)}"
        )


def collect_validation_tokens(
    run_dir: Path,
    *,
    device: torch.device,
    cache_path: Path,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    if cache_path.exists() and manifest_path.exists():
        manifest = _read_json(manifest_path)
        if (
            manifest.get("checkpoint_sha256") == checkpoint_hash
            and manifest.get("split") == "val"
            and manifest.get("max_batches") == max_batches
        ):
            with np.load(cache_path, allow_pickle=False) as payload:
                return {key: payload[key] for key in payload.files}

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = create_tokenizer(config).to(device)
    _load_checkpoint_model_state(model, checkpoint["model_state"])
    model.eval()
    loader = create_configured_multimodal_dataloaders(config)["val"]

    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            eeg = batch["eeg"].to(device, non_blocking=True)
            fnirs = batch["fnirs"].to(device, non_blocking=True)
            masks = _batch_to_device(batch.get("token_valid_mask", {}), device)
            outputs = model(eeg, fnirs, token_valid_masks=masks)
            for modality in ("eeg", "fnirs"):
                chunks[f"{modality}_tokens"].append(
                    outputs[modality].quantizer.hard_ids.detach().cpu().numpy().astype(np.int16)
                )
                mask = batch.get("token_valid_mask", {}).get(modality)
                if mask is None:
                    mask = torch.ones_like(outputs[modality].quantizer.hard_ids, dtype=torch.bool)
                chunks[f"{modality}_mask"].append(mask.detach().cpu().numpy().astype(bool))
            count = int(eeg.shape[0])
            subject = batch.get("subject_key", batch.get("subject", ["unknown"] * count))
            task = batch.get("task_namespace", ["unknown"] * count)
            label = batch.get("label", torch.full((count,), -1))
            chunks["subject"].append(np.asarray(subject, dtype=np.str_))
            chunks["task"].append(np.asarray(task, dtype=np.str_))
            chunks["label"].append(np.asarray(label, dtype=np.int64))

    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    _write_json(
        manifest_path,
        {
            "schema": "e1_validation_hard_token_cache_v1",
            "split": "val",
            "protected_test_opened": False,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "max_batches": max_batches,
            "run_name": run_dir.name,
            "samples": int(arrays["eeg_tokens"].shape[0]),
            "arrays": {key: list(value.shape) for key, value in arrays.items()},
        },
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays


def _lagged_pairs(arrays: Mapping[str, np.ndarray], lag: int) -> tuple[np.ndarray, ...]:
    eeg = np.asarray(arrays["eeg_tokens"], dtype=np.int64)
    fnirs = np.asarray(arrays["fnirs_tokens"], dtype=np.int64)
    eeg_mask = np.asarray(arrays["eeg_mask"], dtype=bool)
    fnirs_mask = np.asarray(arrays["fnirs_mask"], dtype=bool)
    token_count = min(eeg.shape[1], fnirs.shape[1])
    eeg, fnirs = eeg[:, :token_count], fnirs[:, :token_count]
    eeg_mask, fnirs_mask = eeg_mask[:, :token_count], fnirs_mask[:, :token_count]
    if lag >= 0:
        usable = token_count - lag
        left, right = eeg[:, :usable], fnirs[:, lag : lag + usable]
        mask = eeg_mask[:, :usable] & fnirs_mask[:, lag : lag + usable]
    else:
        offset = -lag
        usable = token_count - offset
        left, right = eeg[:, offset : offset + usable], fnirs[:, :usable]
        mask = eeg_mask[:, offset : offset + usable] & fnirs_mask[:, :usable]
    subject = np.repeat(np.asarray(arrays["subject"]).reshape(-1, 1), usable, axis=1)
    label = np.repeat(np.asarray(arrays["label"]).reshape(-1, 1), usable, axis=1)
    return left[mask], right[mask], subject[mask], label[mask]


def pair_statistics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_vocab: int = 128,
    right_vocab: int = 128,
) -> tuple[dict[str, float | int], np.ndarray]:
    left = np.asarray(left, dtype=np.int64).reshape(-1)
    right = np.asarray(right, dtype=np.int64).reshape(-1)
    counts = np.zeros((left_vocab, right_vocab), dtype=np.int64)
    if left.size:
        np.add.at(counts, (left, right), 1)
    total = int(counts.sum())
    if total == 0:
        return {
            "pairs": 0,
            "mi_nats": 0.0,
            "normalized_mi": 0.0,
            "conditional_top1_accuracy": 0.0,
            "marginal_top1_accuracy": 0.0,
            "conditional_accuracy_delta": 0.0,
        }, counts
    joint = counts.astype(np.float64) / total
    p_left = joint.sum(axis=1)
    p_right = joint.sum(axis=0)
    expected = p_left[:, None] * p_right[None, :]
    valid = joint > 0
    mi = float((joint[valid] * np.log(joint[valid] / np.clip(expected[valid], 1e-12, None))).sum())
    h_left, h_right = _entropy(p_left), _entropy(p_right)
    normalized = mi / math.sqrt(max(h_left * h_right, 1e-12))
    conditional_prediction = counts.argmax(axis=1)
    marginal_prediction = int(counts.sum(axis=0).argmax())
    conditional_accuracy = float(np.mean(conditional_prediction[left] == right))
    marginal_accuracy = float(np.mean(right == marginal_prediction))
    return {
        "pairs": total,
        "mi_nats": mi,
        "normalized_mi": float(normalized),
        "conditional_top1_accuracy": conditional_accuracy,
        "marginal_top1_accuracy": marginal_accuracy,
        "conditional_accuracy_delta": conditional_accuracy - marginal_accuracy,
    }, counts


def leave_one_subject_out_accuracy(
    left: np.ndarray,
    right: np.ndarray,
    subjects: np.ndarray,
    *,
    left_vocab: int = 128,
    right_vocab: int = 128,
) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    subjects = np.asarray(subjects)
    correct_conditional = 0
    correct_marginal = 0
    total = 0
    evaluated_subjects = 0
    for subject in np.unique(subjects):
        test = subjects == subject
        train = ~test
        if not test.any() or not train.any():
            continue
        counts = np.full((left_vocab, right_vocab), 0.5, dtype=np.float64)
        np.add.at(counts, (left[train], right[train]), 1.0)
        conditional_prediction = counts.argmax(axis=1)
        marginal_prediction = int(counts.sum(axis=0).argmax())
        correct_conditional += int(np.sum(conditional_prediction[left[test]] == right[test]))
        correct_marginal += int(np.sum(right[test] == marginal_prediction))
        total += int(test.sum())
        evaluated_subjects += 1
    conditional = correct_conditional / max(total, 1)
    marginal = correct_marginal / max(total, 1)
    return {
        "loso_subjects": evaluated_subjects,
        "loso_pairs": total,
        "loso_conditional_accuracy": conditional,
        "loso_marginal_accuracy": marginal,
        "loso_accuracy_delta": conditional - marginal,
    }


def within_group_permutation_null(
    left: np.ndarray,
    right: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    seed: int,
    left_vocab: int = 128,
    right_vocab: int = 128,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    groups = np.asarray([f"{subject}|{label}" for subject, label in zip(subjects, labels)])
    group_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    result = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = left.copy()
        for group_index in group_indices:
            shuffled[group_index] = rng.permutation(shuffled[group_index])
        result[index] = float(
            pair_statistics(shuffled, right, left_vocab=left_vocab, right_vocab=right_vocab)[0][
                "normalized_mi"
            ]
        )
    return result


def coupling_audit(
    arrays: Mapping[str, np.ndarray],
    *,
    lags: Iterable[int],
    permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    matrices: dict[int, np.ndarray] = {}
    for lag in lags:
        left, right, subjects, labels = _lagged_pairs(arrays, lag)
        statistics, matrix = pair_statistics(left, right)
        loso = leave_one_subject_out_accuracy(left, right, subjects)
        null = within_group_permutation_null(
            left,
            right,
            subjects,
            labels,
            permutations=permutations,
            seed=seed + 1009 * (lag + 20),
        )
        observed = float(statistics["normalized_mi"])
        null_mean = float(null.mean())
        null_std = float(null.std(ddof=1)) if null.size > 1 else 0.0
        rows.append(
            {
                "lag_tokens": int(lag),
                "lag_seconds": int(lag) * 2,
                **statistics,
                **loso,
                "null_permutations": int(permutations),
                "null_nmi_mean": null_mean,
                "null_nmi_sd": null_std,
                "nmi_above_null": observed - null_mean,
                "nmi_null_z": (observed - null_mean) / max(null_std, 1e-12),
                "nmi_empirical_p": float((1 + np.sum(null >= observed)) / (null.size + 1)),
                "null_policy": "within_subject_and_label_eeg_token_permutation",
            }
        )
        matrices[int(lag)] = matrix
    return rows, matrices


def _flatten_run_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, run in enumerate(runs):
        factors = run["factors"]
        validation = run["final_validation"]
        row: dict[str, Any] = {
            "order": order,
            "run_name": run["run_name"],
            "label": short_label(run["run_name"]),
            "seed": run["seed"],
            "status": run["status"],
            "global_step": run["global_step"],
            "best_validation": run["best_validation"],
            "assignment": factors["assignment"],
            "normalize_latents": factors["normalize_latents"],
            "kmeans_init": factors["kmeans_init"],
            "ema_decay": factors["ema_decay"],
            "balance_weight": factors["balance_weight"],
            "eeg_balance_temperature": factors["eeg_balance_temperature"],
            "fnirs_balance_temperature": factors["fnirs_balance_temperature"],
            "revive_dead_codes": factors["revive_dead_codes"],
            "revival_strategy": factors["revival_strategy"],
            "revival_stop_after_steps": factors["revival_stop_after_steps"],
        }
        for modality in ("eeg", "fnirs"):
            health = validation[modality]
            for metric in (
                "epoch_active_codes",
                "epoch_active_fraction",
                "effective_codes",
                "assignment_entropy",
                "ema_active_fraction",
                "nearest_neighbor_cosine",
                "effective_rank",
                "total_revivals",
                "valid_tokens",
            ):
                row[f"{modality}_{metric}"] = health.get(metric)
        rows.append(row)
    return rows


def _trajectory_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for entry in run.get("trajectory", []):
            for modality in ("eeg", "fnirs"):
                rows.append(
                    {
                        "run_name": run["run_name"],
                        "label": short_label(run["run_name"]),
                        "epoch": entry.get("epoch"),
                        "global_step": entry.get("global_step"),
                        "modality": modality,
                        **entry.get(modality, {}),
                    }
                )
    return rows


def plot_graphical_abstract(figures_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    stages = [
        (0.25, "24 E1 runs", "K=128 fixed\nreconstruction ablations", OKABE_ITO["gray"]),
        (3.25, "Anti-collapse repair", "cosine + K-means + L2\nbalance + warmup + revival", OKABE_ITO["blue"]),
        (6.25, "Health evidence", "active/effective codes\ngeometry + retention", OKABE_ITO["green"]),
        (9.25, "Coupling trace", "paired validation tokens\nlag + LOSO + permutation null", OKABE_ITO["orange"]),
    ]
    for x, title, body, color in stages:
        box = FancyBboxPatch(
            (x, 1.05),
            2.4,
            2.05,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.add_patch(box)
        text_color = "white" if color != OKABE_ITO["orange"] else "black"
        ax.text(x + 1.2, 2.55, title, ha="center", va="center", weight="bold", fontsize=11, color=text_color)
        ax.text(x + 1.2, 1.7, body, ha="center", va="center", fontsize=9, color=text_color, linespacing=1.35)
    for x in (2.7, 5.7, 8.7):
        ax.add_patch(FancyArrowPatch((x, 2.08), (x + 0.5, 2.08), arrowstyle="-|>", mutation_scale=18, linewidth=1.5))
    ax.text(
        6,
        0.45,
        "Result: G1 health passes after revival stops; EEG–fNIRS association remains diagnostic and is not an E9 coupling certificate",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    return _save_figure(fig, figures_dir, "00_graphical_abstract")


def plot_methods_flow(figures_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    boxes = [
        (0.3, 3.2, 2.2, 1.2, "Registered artifacts", "config, health JSONL,\nvalidation metrics"),
        (3.0, 3.2, 2.2, 1.2, "Checkpoint audit", "EMA occupancy, codebook\nPCA and cosine geometry"),
        (5.7, 3.2, 2.2, 1.2, "Validation replay", "hard IDs only;\nprotected test closed"),
        (8.4, 3.2, 2.2, 1.2, "Paired association", "lag −8…+8 s, LOSO,\nwithin-group null"),
        (2.0, 0.75, 3.0, 1.2, "Health conclusion", "coverage, effective usage,\nrevival retention, seed stability"),
        (6.0, 0.75, 3.0, 1.2, "Claim calibration", "trace ≠ causal physiology;\nE7–E9 remain required"),
    ]
    colors = [OKABE_ITO["sky"], OKABE_ITO["purple"], OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
    for (x, y, w, h, title, body), color in zip(boxes, colors):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=color, edgecolor="black", alpha=0.85)
        ax.add_patch(patch)
        foreground = "white" if color in {OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["purple"]} else "black"
        ax.text(x + w / 2, y + 0.82 * h, title, ha="center", va="center", weight="bold", fontsize=10, color=foreground)
        ax.text(x + w / 2, y + 0.37 * h, body, ha="center", va="center", fontsize=8.5, color=foreground)
    for x in (2.5, 5.2, 7.9):
        ax.add_patch(FancyArrowPatch((x, 3.8), (x + 0.45, 3.8), arrowstyle="-|>", mutation_scale=16))
    ax.add_patch(FancyArrowPatch((7.3, 3.15), (7.3, 2.05), arrowstyle="-|>", mutation_scale=16))
    ax.add_patch(FancyArrowPatch((4.1, 3.15), (3.7, 2.05), arrowstyle="-|>", mutation_scale=16))
    ax.set_title("Reproducible E1 audit workflow", fontsize=13, weight="bold", pad=4)
    return _save_figure(fig, figures_dir, "01_methods_flow")


def plot_health_landscape(run_rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> dict[str, str]:
    x = np.arange(len(run_rows))
    labels = [str(row["label"]) for row in run_rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    for modality, color, marker in (("eeg", OKABE_ITO["blue"], "o"), ("fnirs", OKABE_ITO["orange"], "s")):
        active = np.asarray([row[f"{modality}_epoch_active_codes"] for row in run_rows], dtype=float)
        effective = np.asarray([row[f"{modality}_effective_codes"] for row in run_rows], dtype=float)
        revival = np.asarray([row[f"{modality}_total_revivals"] for row in run_rows], dtype=float)
        axes[0, 0].plot(x, active, marker=marker, color=color, label=modality.upper(), linewidth=1.5, markersize=4)
        axes[0, 1].plot(x, effective, marker=marker, color=color, label=modality.upper(), linewidth=1.5, markersize=4)
        axes[1, 0].plot(x, np.divide(effective, np.clip(active, 1, None)), marker=marker, color=color, label=modality.upper(), linewidth=1.5, markersize=4)
        axes[1, 1].plot(x, revival, marker=marker, color=color, label=modality.upper(), linewidth=1.5, markersize=4)
    axes[0, 0].axhline(64, color="0.75", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Codes active in validation epoch")
    axes[0, 0].set_title("A  Coverage")
    axes[0, 1].set_ylabel("exp(assignment entropy)")
    axes[0, 1].set_title("B  Effective usage")
    axes[1, 0].set_ylabel("Effective / active codes")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("C  Occupancy evenness")
    axes[1, 1].set_ylabel("Cumulative revived codes")
    axes[1, 1].set_title("D  Revival dependence")
    for ax in axes.reshape(-1):
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, ncol=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right")
    fig.suptitle("E1 ablation landscape (final validation epoch; K=128)", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(fig, figures_dir, "02_e1_health_landscape")


def plot_performance_tradeoff(run_rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, modality, color in zip(axes, ("eeg", "fnirs"), (OKABE_ITO["blue"], OKABE_ITO["orange"])):
        x = np.asarray([row[f"{modality}_effective_codes"] for row in run_rows], dtype=float)
        y = np.asarray([row["best_validation"] for row in run_rows], dtype=float)
        revival = np.asarray([row[f"{modality}_total_revivals"] for row in run_rows], dtype=float)
        sizes = 24 + 1.4 * np.sqrt(np.clip(revival, 0, None))
        ax.scatter(x, y, s=sizes, color=color, alpha=0.75, edgecolor="black", linewidth=0.4)
        for row, xx, yy in zip(run_rows, x, y):
            if row["label"] in {"v2", "v12", "v14", "v17", "v20", "v21/s21", "v22/s21", "v23/s19", "v23/s20"}:
                ax.annotate(str(row["label"]), (xx, yy), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel(f"{modality.upper()} effective codes")
        ax.set_ylabel("Best validation loss")
        ax.set_title(f"{chr(65 + len(ax.get_figure().axes) - 2)}  {modality.upper()} health–loss trade-off")
        ax.grid(alpha=0.2)
    axes[0].set_title("A  EEG health–loss trade-off")
    axes[1].set_title("B  fNIRS health–loss trade-off")
    fig.suptitle("Health improved without a monotonic reconstruction-loss benefit\n(marker area increases with cumulative revival)", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save_figure(fig, figures_dir, "03_health_performance_tradeoff")


def plot_retention(
    runs_by_name: Mapping[str, Mapping[str, Any]],
    figures_dir: Path,
) -> dict[str, str]:
    selected = [name for name in runs_by_name if "post_revival" in name and runs_by_name[name]["factors"].get("revival_stop_after_steps") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    palette = plt.get_cmap("tab10")
    for index, name in enumerate(selected):
        run = runs_by_name[name]
        style = "-" if "diverse_farthest" in name else "--"
        color = palette(index % 10)
        for ax, modality in zip(axes, ("eeg", "fnirs")):
            steps = [entry["global_step"] for entry in run["trajectory"]]
            values = [entry[modality]["effective_codes"] for entry in run["trajectory"]]
            ax.plot(steps, values, linestyle=style, color=color, marker="o", markersize=3, linewidth=1.3, label=short_label(name))
    for ax, modality, threshold in zip(axes, ("EEG", "fNIRS"), (32, 24)):
        ax.axvline(200, color="black", linestyle=":", linewidth=1.3, label="revival stop (step 200)")
        ax.axhline(threshold, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.1, label=f"gate floor ({threshold})")
        ax.set_xlabel("Global optimizer step")
        ax.set_ylabel("Effective codes")
        ax.set_title(f"{modality} post-revival retention")
        ax.grid(alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Top-error failure and diverse-farthest repair across registered seeds", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    return _save_figure(fig, figures_dir, "04_post_revival_retention")


def plot_codebook_geometry(
    geometry_arrays: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    figures_dir: Path,
) -> dict[str, str]:
    names = [name for name in FINAL_GATE_RUNS if name in geometry_arrays]
    fig, axes = plt.subplots(2, len(names), figsize=(4.0 * len(names), 7.2), squeeze=False)
    for column, name in enumerate(names):
        for row, modality in enumerate(("eeg", "fnirs")):
            arrays = geometry_arrays[name][modality]
            coords, ratio = _pca_2d(arrays["codebook"])
            counts = arrays["ema_count"]
            probability = counts / np.clip(counts.sum(), 1e-12, None)
            sizes = 8 + 180 * np.sqrt(probability)
            scatter = axes[row, column].scatter(
                coords[:, 0],
                coords[:, 1],
                c=probability,
                s=sizes,
                cmap="cividis",
                edgecolor="black",
                linewidth=0.2,
                alpha=0.85,
            )
            axes[row, column].set_xlabel(f"PC1 ({100 * ratio[0]:.1f}%)")
            axes[row, column].set_ylabel(f"PC2 ({100 * ratio[1]:.1f}%)")
            axes[row, column].set_title(f"{short_label(name)} · {modality.upper()}")
            axes[row, column].grid(alpha=0.15)
            fig.colorbar(scatter, ax=axes[row, column], fraction=0.046, pad=0.04, label="EMA occupancy")
    fig.suptitle("Final-gate codebook geometry; point size/color encode EMA occupancy", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, figures_dir, "05_final_codebook_pca")


def plot_geometry_diagnostics(
    geometry_rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> dict[str, str]:
    rows = [row for row in geometry_rows if row["run_name"] in FINAL_GATE_RUNS]
    labels = [f"{row['label']}\n{str(row['modality']).upper()}" for row in rows]
    x = np.arange(len(rows))
    colors = [OKABE_ITO["blue"] if row["modality"] == "eeg" else OKABE_ITO["orange"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
    axes[0, 0].bar(x, [row["nearest_cosine_mean"] for row in rows], color=colors, alpha=0.85, label="Mean")
    axes[0, 0].scatter(x, [row["nearest_cosine_q95"] for row in rows], marker="D", color="black", s=20, label="95th percentile")
    axes[0, 0].scatter(x, [row["nearest_cosine_max"] for row in rows], marker="x", color=OKABE_ITO["vermillion"], s=30, label="Maximum")
    axes[0, 0].axhline(0.99, color="0.45", linestyle="--", linewidth=1, label="0.99 reference")
    axes[0, 0].set_ylabel("Cosine similarity")
    axes[0, 0].set_ylim(0.90, 1.002)
    axes[0, 0].set_title("A  Nearest-neighbor similarity")
    axes[0, 0].legend(frameon=False, fontsize=6, ncol=2)

    axes[0, 1].bar(x, [row["duplicate_pairs_cosine_ge_0p99"] for row in rows], color=colors)
    axes[0, 1].set_ylabel("Code pairs")
    axes[0, 1].set_title("B  Near-duplicate pairs (cosine ≥ 0.99)")

    width = 0.36
    axes[1, 0].bar(x - width / 2, [row["matrix_rank"] for row in rows], width, color="0.65", label="Algebraic matrix rank")
    axes[1, 0].bar(x + width / 2, [row["participation_rank"] for row in rows], width, color=colors, label="Spectral participation rank")
    axes[1, 0].set_ylabel("Rank")
    axes[1, 0].set_title("C  Full algebraic rank masks spectral concentration")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(x - width / 2, [100 * row["pc1_variance_ratio"] for row in rows], width, color=colors, label="PC1")
    axes[1, 1].bar(x + width / 2, [100 * row["pc1_pc2_variance_ratio"] for row in rows], width, color="0.55", label="PC1 + PC2")
    axes[1, 1].set_ylabel("Centered codebook variance (%)")
    axes[1, 1].set_title("D  Variance captured by leading axes")
    axes[1, 1].legend(frameon=False)

    for ax in axes.reshape(-1):
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(x, labels, rotation=55, ha="right")
    fig.suptitle("Final-gate occupancy pass does not imply isotropic prototype geometry", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, figures_dir, "06_codebook_geometry_diagnostics")


def plot_normalization(input_audit: Mapping[str, Any], figures_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    modalities = ("eeg", "fnirs")
    colors = (OKABE_ITO["blue"], OKABE_ITO["orange"])
    quantiles = ("q05", "q50", "q95")
    x = np.arange(3)
    width = 0.34
    for index, (modality, color) in enumerate(zip(modalities, colors)):
        payload = input_audit["modalities"][modality]
        axes[0].bar(x + (index - 0.5) * width, [payload["channel_std"][q] for q in quantiles], width, label=modality.upper(), color=color)
        axes[1].bar(x + (index - 0.5) * width, [payload["within_window_std_ratio"][q] for q in quantiles], width, label=modality.upper(), color=color)
        axes[2].bar(index, 100 * payload["channel_std_outside_0p5_2_fraction"], color=color, width=0.55, label=modality.upper())
    axes[0].set_title("A  Channel-scale distribution")
    axes[0].set_ylabel("Channel SD")
    axes[1].set_title("B  Within-window scale ratio")
    axes[1].set_ylabel("max SD / min SD")
    axes[2].set_title("C  Scale outside [0.5, 2]")
    axes[2].set_ylabel("Channel-windows (%)")
    for ax in axes[:2]:
        ax.set_xticks(x, ["q05", "q50", "q95"])
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
    axes[2].set_xticks([0, 1], ["EEG", "fNIRS"])
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("Training-input normalization retains stronger fNIRS window-level scale heterogeneity", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save_figure(fig, figures_dir, "07_input_normalization_audit")


def plot_coupling_summary(coupling_rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> dict[str, str]:
    lag_zero = [row for row in coupling_rows if int(row["lag_tokens"]) == 0]
    x = np.arange(len(lag_zero))
    labels = [str(row["label"]) for row in lag_zero]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    observed = np.asarray([row["normalized_mi"] for row in lag_zero], dtype=float)
    null = np.asarray([row["null_nmi_mean"] for row in lag_zero], dtype=float)
    null_sd = np.asarray([row["null_nmi_sd"] for row in lag_zero], dtype=float)
    axes[0].bar(x - 0.18, observed, 0.36, label="Observed", color=OKABE_ITO["blue"])
    axes[0].bar(x + 0.18, null, 0.36, yerr=null_sd, label="Permutation null", color="0.65", capsize=2)
    axes[0].set_ylabel("Normalized mutual information")
    axes[0].set_title("A  Contemporaneous association")
    axes[0].legend(frameon=False)
    axes[1].bar(x, [row["loso_accuracy_delta"] for row in lag_zero], color=OKABE_ITO["green"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("LOSO conditional − marginal accuracy")
    axes[1].set_title("B  Subject-held-out predictability")
    axes[2].bar(x, [row["nmi_above_null"] for row in lag_zero], color=OKABE_ITO["orange"])
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Observed NMI − null mean")
    axes[2].set_title("C  Excess over within-group null")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=65, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("EEG–fNIRS token association is a validation-only diagnostic trace", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save_figure(fig, figures_dir, "08_cross_modal_coupling_summary")


def plot_lag_profiles(coupling_rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> dict[str, str]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in coupling_rows:
        grouped[str(row["run_name"])].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    palette = plt.get_cmap("tab10")
    for index, (name, rows) in enumerate(grouped.items()):
        rows = sorted(rows, key=lambda item: int(item["lag_tokens"]))
        seconds = [int(row["lag_seconds"]) for row in rows]
        color = palette(index % 10)
        axes[0].plot(seconds, [row["nmi_above_null"] for row in rows], marker="o", markersize=3, linewidth=1.2, color=color, label=short_label(name))
        axes[1].plot(seconds, [row["loso_accuracy_delta"] for row in rows], marker="o", markersize=3, linewidth=1.2, color=color, label=short_label(name))
    axes[0].set_ylabel("NMI above null")
    axes[0].set_title("A  Lagged association beyond permutation")
    axes[1].set_ylabel("LOSO conditional − marginal accuracy")
    axes[1].set_title("B  Lagged held-out predictability")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="0.5", linestyle=":", linewidth=0.9)
        ax.set_xlabel("Lag (s; positive = EEG leads fNIRS)")
        ax.grid(alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Lag profile does not by itself identify neurovascular causality", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0.15, 1, 0.93))
    return _save_figure(fig, figures_dir, "09_cross_modal_lag_profiles")


def plot_joint_enrichment(
    matrices: Mapping[str, np.ndarray],
    figures_dir: Path,
) -> dict[str, str]:
    names = [name for name in FINAL_GATE_RUNS if name in matrices]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.9), squeeze=False, layout="constrained")
    images = []
    enrichments = []
    for name in names:
        counts = matrices[name].astype(np.float64)
        total = counts.sum()
        joint = counts / max(total, 1.0)
        expected = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
        enrichment = np.log2((joint + 1.0 / max(total, 1.0)) / (expected + 1.0 / max(total, 1.0)))
        eeg_order = np.argsort(joint.sum(axis=1))[::-1][:64]
        fnirs_order = np.argsort(joint.sum(axis=0))[::-1][:64]
        enrichments.append(enrichment[np.ix_(eeg_order, fnirs_order)])
    bound = max(float(np.quantile(np.abs(value), 0.98)) for value in enrichments) if enrichments else 1.0
    for ax, name, enrichment in zip(axes[0], names, enrichments):
        image = ax.imshow(enrichment, cmap="PuOr", vmin=-bound, vmax=bound, aspect="auto", interpolation="nearest")
        images.append(image)
        ax.set_title(short_label(name))
        ax.set_xlabel("fNIRS code (top 64 by use)")
        ax.set_ylabel("EEG code (top 64 by use)")
    if images:
        fig.colorbar(images[-1], ax=list(axes[0]), fraction=0.02, pad=0.02, label="log2 observed / independent")
    fig.suptitle("Final-gate lag-0 token-pair enrichment (validation only)", fontsize=11, weight="bold")
    return _save_figure(fig, figures_dir, "10_final_gate_joint_enrichment")


def build_report_summary(
    run_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    coupling_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final = [row for row in run_rows if row["run_name"] in FINAL_GATE_RUNS]
    lag_zero = [row for row in coupling_rows if row["run_name"] in FINAL_GATE_RUNS and int(row["lag_tokens"]) == 0]
    geometry_final = [row for row in geometry_rows if row["run_name"] in FINAL_GATE_RUNS]

    def describe(values: Sequence[float]) -> dict[str, float | int | None]:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return {"count": 0, "mean": None, "sample_sd": None, "min": None, "max": None}
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return {
        "schema": SCHEMA,
        "run_count": len(run_rows),
        "coupling_run_count": len({row["run_name"] for row in coupling_rows}),
        "protected_test_opened": False,
        "final_gate": {
            "runs": [row["run_name"] for row in final],
            "eeg_active_codes": describe([float(row["eeg_epoch_active_codes"]) for row in final]),
            "eeg_effective_codes": describe([float(row["eeg_effective_codes"]) for row in final]),
            "fnirs_active_codes": describe([float(row["fnirs_epoch_active_codes"]) for row in final]),
            "fnirs_effective_codes": describe([float(row["fnirs_effective_codes"]) for row in final]),
            "best_validation": describe([float(row["best_validation"]) for row in final]),
            "geometry": {
                modality: {
                    "nearest_cosine_max": describe([float(row["nearest_cosine_max"]) for row in geometry_final if row["modality"] == modality]),
                    "participation_rank": describe([float(row["participation_rank"]) for row in geometry_final if row["modality"] == modality]),
                }
                for modality in ("eeg", "fnirs")
            },
            "lag0_coupling_trace": {
                "normalized_mi": describe([float(row["normalized_mi"]) for row in lag_zero]),
                "nmi_above_null": describe([float(row["nmi_above_null"]) for row in lag_zero]),
                "loso_accuracy_delta": describe([float(row["loso_accuracy_delta"]) for row in lag_zero]),
                "empirical_p": [float(row["nmi_empirical_p"]) for row in lag_zero],
                "null_policy": "within_subject_and_label_eeg_token_permutation",
            },
        },
        "claim_boundary": (
            "E1 establishes validation-only codebook health under the registered fixed-K=128 contract. "
            "The cross-modal analysis is a retrospective association trace and does not establish physiological, "
            "causal, or independently certified EEG-fNIRS coupling."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Definitive E1 occupancy summary.json")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-normalization-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coupling-run", action="append", default=[])
    parser.add_argument("--lags", default="-4,-3,-2,-1,0,1,2,3,4")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-coupling", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    _configure_style()
    payload = _read_json(args.summary)
    runs = payload["runs"]
    if any(bool(run.get("protected_test_opened")) for run in runs):
        raise RuntimeError("Refusing to analyze E1 summary containing opened protected-test evidence")
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    token_dir = output_dir / "validation_tokens"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = _flatten_run_rows(runs)
    trajectory_rows = _trajectory_rows(runs)
    runs_by_name = {run["run_name"]: run for run in runs}

    geometry_rows: list[dict[str, Any]] = []
    geometry_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for run in runs:
        run_dir = args.run_root / run["run_name"]
        checkpoint_path = run_dir / "checkpoints" / "last.pt"
        if not checkpoint_path.exists():
            continue
        rows, arrays = checkpoint_geometry(checkpoint_path)
        for row in rows:
            row["run_name"] = run["run_name"]
            row["label"] = short_label(run["run_name"])
        geometry_rows.extend(rows)
        geometry_arrays[run["run_name"]] = arrays

    coupling_rows: list[dict[str, Any]] = []
    lag_zero_matrices: dict[str, np.ndarray] = {}
    requested_coupling_runs = tuple(args.coupling_run) if args.coupling_run else DEFAULT_COUPLING_RUNS
    if not args.skip_coupling:
        device = torch.device(args.device)
        lags = tuple(int(value) for value in args.lags.split(",") if value.strip())
        for run_index, run_name in enumerate(requested_coupling_runs):
            if run_name not in runs_by_name:
                raise KeyError(f"Coupling run absent from E1 summary: {run_name}")
            run_dir = args.run_root / run_name
            arrays = collect_validation_tokens(
                run_dir,
                device=device,
                cache_path=token_dir / f"{run_name}.npz",
                max_batches=args.max_batches,
            )
            rows, matrices = coupling_audit(
                arrays,
                lags=lags,
                permutations=args.permutations,
                seed=args.seed + 100_003 * run_index,
            )
            for row in rows:
                row["run_name"] = run_name
                row["label"] = short_label(run_name)
                row["seed"] = runs_by_name[run_name]["seed"]
            coupling_rows.extend(rows)
            if 0 in matrices:
                lag_zero_matrices[run_name] = matrices[0]
                matrix_rows = []
                for eeg_code, fnirs_code in zip(*np.nonzero(matrices[0])):
                    matrix_rows.append(
                        {
                            "eeg_code": int(eeg_code),
                            "fnirs_code": int(fnirs_code),
                            "count": int(matrices[0][eeg_code, fnirs_code]),
                        }
                    )
                _write_csv(
                    tables_dir / "joint_counts" / f"{run_name}_lag0.csv",
                    matrix_rows,
                    ("eeg_code", "fnirs_code", "count"),
                )

    run_fields = list(run_rows[0]) if run_rows else []
    trajectory_fields = list(trajectory_rows[0]) if trajectory_rows else []
    geometry_fields = list(geometry_rows[0]) if geometry_rows else []
    coupling_fields = list(coupling_rows[0]) if coupling_rows else []
    _write_csv(tables_dir / "e1_run_metrics.csv", run_rows, run_fields)
    _write_csv(tables_dir / "e1_health_trajectories.csv", trajectory_rows, trajectory_fields)
    _write_csv(tables_dir / "e1_checkpoint_geometry.csv", geometry_rows, geometry_fields)
    if coupling_rows:
        _write_csv(tables_dir / "e1_coupling_lag_metrics.csv", coupling_rows, coupling_fields)

    figures: dict[str, dict[str, str]] = {}
    figures["graphical_abstract"] = plot_graphical_abstract(figures_dir)
    figures["methods_flow"] = plot_methods_flow(figures_dir)
    figures["health_landscape"] = plot_health_landscape(run_rows, figures_dir)
    figures["performance_tradeoff"] = plot_performance_tradeoff(run_rows, figures_dir)
    figures["retention"] = plot_retention(runs_by_name, figures_dir)
    figures["codebook_geometry"] = plot_codebook_geometry(geometry_arrays, figures_dir)
    figures["geometry_diagnostics"] = plot_geometry_diagnostics(geometry_rows, figures_dir)
    input_audit = _read_json(args.input_normalization_audit)
    if input_audit.get("protected_test_opened"):
        raise RuntimeError("Refusing to include input-normalization audit that opened protected test data")
    figures["input_normalization"] = plot_normalization(input_audit, figures_dir)
    if coupling_rows:
        figures["coupling_summary"] = plot_coupling_summary(coupling_rows, figures_dir)
        figures["lag_profiles"] = plot_lag_profiles(coupling_rows, figures_dir)
        figures["joint_enrichment"] = plot_joint_enrichment(lag_zero_matrices, figures_dir)

    summary = build_report_summary(run_rows, geometry_rows, coupling_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "manifest.json",
        {
            "schema": SCHEMA,
            "protected_test_opened": False,
            "source_summary": str(args.summary.resolve()),
            "source_summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
            "run_root": str(args.run_root.resolve()),
            "run_count": len(run_rows),
            "geometry_run_count": len({row["run_name"] for row in geometry_rows}),
            "coupling_runs": list(requested_coupling_runs) if not args.skip_coupling else [],
            "coupling_split": "val" if not args.skip_coupling else None,
            "coupling_permutations": args.permutations if not args.skip_coupling else 0,
            "figures": figures,
            "tables": {
                "run_metrics": str((tables_dir / "e1_run_metrics.csv").resolve()),
                "health_trajectories": str((tables_dir / "e1_health_trajectories.csv").resolve()),
                "checkpoint_geometry": str((tables_dir / "e1_checkpoint_geometry.csv").resolve()),
                "coupling_lag_metrics": str((tables_dir / "e1_coupling_lag_metrics.csv").resolve()) if coupling_rows else None,
            },
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "runs": len(run_rows), "coupling_runs": len(set(row["run_name"] for row in coupling_rows))}, sort_keys=True))
    return output_dir


if __name__ == "__main__":
    run(parse_args())
