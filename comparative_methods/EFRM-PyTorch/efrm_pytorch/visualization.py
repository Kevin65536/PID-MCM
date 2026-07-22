"""Auditable EFRM positive-pair visualizations and retrieval diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


EVIDENCE_SCHEMA = "efrm_clip_alignment_evidence_v1"
PHYSIOLOGY_EVIDENCE_SCHEMA = "directional_lag_coupling_evidence_v1"


def _as_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def retrieval_metrics(cosine_similarity: np.ndarray) -> dict[str, Any]:
    cosine = np.asarray(cosine_similarity, dtype=np.float64)
    if cosine.ndim != 2 or cosine.shape[0] != cosine.shape[1]:
        raise ValueError("EFRM retrieval evidence requires a square paired similarity matrix")

    def ranks(matrix: np.ndarray) -> np.ndarray:
        order = np.argsort(-matrix, axis=1)
        target = np.arange(matrix.shape[0])[:, None]
        return np.argmax(order == target, axis=1) + 1

    eeg_ranks = ranks(cosine)
    fnirs_ranks = ranks(cosine.T)
    result: dict[str, Any] = {
        "pair_count": int(cosine.shape[0]),
        "positive_cosine_mean": float(np.diag(cosine).mean()),
        "negative_cosine_mean": float(cosine[~np.eye(len(cosine), dtype=bool)].mean()),
    }
    for name, values in (("eeg_to_fnirs", eeg_ranks), ("fnirs_to_eeg", fnirs_ranks)):
        result[name] = {
            "mean_rank": float(values.mean()),
            "median_rank": float(np.median(values)),
            "mrr": float(np.mean(1.0 / values)),
            "top1": float(np.mean(values <= 1)),
            "top5": float(np.mean(values <= min(5, len(values)))),
            "ranks": values.astype(int).tolist(),
        }
    return result


def export_alignment_evidence(
    output_dir: str | Path,
    *,
    eeg_embeddings: Any,
    fnirs_embeddings: Any,
    metadata: Sequence[Mapping[str, Any]],
    logit_multiplier: float = 0.1,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    eeg = _as_numpy(eeg_embeddings).astype(np.float32)
    fnirs = _as_numpy(fnirs_embeddings).astype(np.float32)
    if eeg.shape != fnirs.shape or eeg.ndim != 2 or eeg.shape[0] != len(metadata):
        raise ValueError("embedding arrays and pair metadata must have matching [pair,dimension] shapes")
    eeg_normalized = eeg / np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
    fnirs_normalized = fnirs / np.maximum(np.linalg.norm(fnirs, axis=1, keepdims=True), 1e-12)
    cosine = eeg_normalized @ fnirs_normalized.T
    path = directory / "clip_alignment_evidence.npz"
    np.savez_compressed(
        path,
        schema=np.asarray(EVIDENCE_SCHEMA),
        eeg_embeddings=eeg,
        fnirs_embeddings=fnirs,
        cosine_similarity=cosine.astype(np.float32),
        scaled_logits=(float(logit_multiplier) * cosine).astype(np.float32),
        positive_pair_mask=np.eye(len(metadata), dtype=bool),
        logit_multiplier=np.asarray(float(logit_multiplier)),
        metadata_json=np.asarray([json.dumps(dict(row), sort_keys=True) for row in metadata]),
    )
    return path


def _load_evidence(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files if key != "metadata_json"}
        metadata = [json.loads(str(value)) for value in payload["metadata_json"].tolist()]
    if str(arrays["schema"].item()) != EVIDENCE_SCHEMA:
        raise ValueError(f"unsupported evidence schema: {arrays['schema'].item()}")
    return arrays, metadata


def _save_figure(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _similarity_figure(cosine: np.ndarray, metadata: Sequence[Mapping[str, Any]], base: Path) -> None:
    size = len(cosine)
    figure, axis = plt.subplots(figsize=(max(6.0, size * 0.24), max(5.2, size * 0.22)))
    image = axis.imshow(cosine, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    for index in range(size):
        axis.add_patch(
            Rectangle(
                (index - 0.46, index - 0.46), 0.92, 0.92, fill=False,
                edgecolor="#F0E442", lw=2.2,
            )
        )
        if size <= 16:
            axis.text(index, index, "P+", ha="center", va="center", fontsize=7, fontweight="bold")
    labels = [
        f"{row.get('subject','?')}:{row.get('condition','')}@{float(row.get('crop_start_s', 0.0)):.1f}s"
        for row in metadata
    ]
    if size <= 32:
        axis.set_xticks(np.arange(size), labels=labels, rotation=90, fontsize=7)
        axis.set_yticks(np.arange(size), labels=labels, fontsize=7)
    axis.set_xlabel("fNIRS synchronized windows")
    axis.set_ylabel("EEG synchronized windows")
    axis.set_title("EFRM symmetric retrieval: yellow P+ diagonal cells are declared positives")
    figure.colorbar(image, ax=axis, label="raw cosine similarity")
    _save_figure(figure, base)


def _distribution_figure(cosine: np.ndarray, metadata: Sequence[Mapping[str, Any]], base: Path) -> None:
    diagonal = np.diag(cosine)
    off_diagonal = ~np.eye(len(cosine), dtype=bool)
    all_negative = cosine[off_diagonal]
    hard_negative = np.where(off_diagonal, cosine, -np.inf).max(axis=1)
    within_record = []
    for row, left in enumerate(metadata):
        for column, right in enumerate(metadata):
            if row != column and left.get("join_key") == right.get("join_key"):
                within_record.append(cosine[row, column])
    series = [diagonal, all_negative, hard_negative]
    labels = ["positive diagonal", "all off-diagonal", "hardest per EEG"]
    if within_record:
        series.append(np.asarray(within_record))
        labels.append("within-record negatives")
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.boxplot(series, tick_labels=labels, showmeans=True)
    axis.axhline(0.0, color="#888888", lw=0.8)
    axis.set_ylabel("raw cosine similarity")
    axis.set_title("EFRM positive and negative pair evidence")
    axis.tick_params(axis="x", rotation=18)
    _save_figure(figure, base)


def _rank_figure(metrics: Mapping[str, Any], base: Path) -> None:
    eeg = np.asarray(metrics["eeg_to_fnirs"]["ranks"], dtype=int)
    fnirs = np.asarray(metrics["fnirs_to_eeg"]["ranks"], dtype=int)
    maximum = max(eeg.max(), fnirs.max())
    k = np.arange(1, maximum + 1)
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.step(k, [(eeg <= value).mean() for value in k], where="post", label="EEG→fNIRS")
    axis.step(k, [(fnirs <= value).mean() for value in k], where="post", label="fNIRS→EEG")
    axis.set_xlabel("retrieval rank k")
    axis.set_ylabel("fraction with positive rank ≤ k")
    axis.set_ylim(0, 1.02)
    axis.set_title("True synchronized-pair retrieval")
    axis.legend()
    _save_figure(figure, base)


def _projection_figure(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    base: Path,
) -> None:
    combined = np.concatenate((eeg, fnirs), axis=0).astype(np.float64)
    combined -= combined.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(combined, full_matrices=False)
    projected = combined @ right[:2].T
    eeg_xy, fnirs_xy = projected[: len(eeg)], projected[len(eeg) :]
    datasets = sorted({str(row.get("dataset_id", "unknown")) for row in metadata})
    colors = {name: plt.cm.tab10(index % 10) for index, name in enumerate(datasets)}
    figure, axis = plt.subplots(figsize=(7.0, 6.2))
    for index, row in enumerate(metadata):
        color = colors[str(row.get("dataset_id", "unknown"))]
        axis.plot(
            [eeg_xy[index, 0], fnirs_xy[index, 0]],
            [eeg_xy[index, 1], fnirs_xy[index, 1]],
            color=color,
            alpha=0.35,
            lw=0.8,
        )
        axis.scatter(*eeg_xy[index], marker="o", color=color, s=25)
        axis.scatter(*fnirs_xy[index], marker="^", color=color, s=28)
    for dataset, color in colors.items():
        axis.scatter([], [], marker="o", color=color, label=dataset)
    axis.scatter([], [], marker="o", color="#333333", label="EEG")
    axis.scatter([], [], marker="^", color="#333333", label="fNIRS")
    axis.set_xlabel("joint PCA axis 1")
    axis.set_ylabel("joint PCA axis 2")
    axis.set_title("EFRM pooled embeddings; each line is one acquisition pair")
    axis.legend(fontsize=7, loc="best")
    _save_figure(figure, base)


def _comparison_figure(cosine: np.ndarray, physiology_path: str | Path, base: Path) -> None:
    with np.load(physiology_path, allow_pickle=False) as payload:
        schema = str(payload["schema"].item())
        if schema != PHYSIOLOGY_EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported physiological coupling schema: {schema}")
        lags = np.asarray(payload["lag_seconds"], dtype=np.float64)
        scores = np.asarray(payload["coupling_scores"], dtype=np.float64)
    profile = scores.reshape(scores.shape[0], -1).mean(axis=1)
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    axes[0].imshow(cosine, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    for index in range(len(cosine)):
        axes[0].add_patch(Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False, edgecolor="black", lw=0.9))
    axes[0].set_title("EFRM: symmetric co-window retrieval")
    axes[0].set_xlabel("fNIRS window index")
    axes[0].set_ylabel("EEG window index")
    axes[1].plot(lags, profile, marker="o", color="#7A1FA2")
    axes[1].axvline(0.0, color="#888888", ls="--", lw=0.8)
    axes[1].set_title("Project model: directional lag-conditioned coupling")
    axes[1].set_xlabel("EEG→fNIRS lag (s)")
    axes[1].set_ylabel("mean coupling evidence")
    figure.suptitle(
        "Different claims: acquisition-pair similarity is not a directional physiological mechanism",
        fontsize=11,
    )
    _save_figure(figure, base)


def render_alignment_report(
    evidence_path: str | Path,
    output_dir: str | Path,
    *,
    physiology_coupling_evidence: str | Path | None = None,
) -> dict[str, Any]:
    arrays, metadata = _load_evidence(evidence_path)
    directory = Path(output_dir)
    figures = directory / "figures"
    figure_data = directory / "figure_data"
    figures.mkdir(parents=True, exist_ok=True)
    figure_data.mkdir(parents=True, exist_ok=True)
    cosine = arrays["cosine_similarity"]
    metrics = retrieval_metrics(cosine)

    _similarity_figure(cosine, metadata, figures / "clip_similarity_positive_pairs")
    _distribution_figure(cosine, metadata, figures / "clip_pair_distributions")
    _rank_figure(metrics, figures / "clip_retrieval_ranks")
    _projection_figure(
        arrays["eeg_embeddings"], arrays["fnirs_embeddings"], metadata,
        figures / "clip_paired_embedding_projection",
    )
    if physiology_coupling_evidence is not None:
        _comparison_figure(
            cosine,
            physiology_coupling_evidence,
            figures / "efrm_vs_directional_physiological_coupling",
        )

    np.savez_compressed(
        figure_data / "clip_similarity_and_positive_mask.npz",
        cosine_similarity=cosine,
        positive_pair_mask=arrays["positive_pair_mask"],
    )
    with (figure_data / "pair_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted(set().union(*(row.keys() for row in metadata)))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    (directory / "alignment_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = [
        "# EFRM CLIP alignment report",
        "",
        f"- Pairs: {metrics['pair_count']}",
        f"- Positive cosine mean: {metrics['positive_cosine_mean']:.6f}",
        f"- Negative cosine mean: {metrics['negative_cosine_mean']:.6f}",
        f"- EEG→fNIRS top-1 / MRR: {metrics['eeg_to_fnirs']['top1']:.4f} / {metrics['eeg_to_fnirs']['mrr']:.4f}",
        f"- fNIRS→EEG top-1 / MRR: {metrics['fnirs_to_eeg']['top1']:.4f} / {metrics['fnirs_to_eeg']['mrr']:.4f}",
        "",
        "The boxed diagonal is defined by synchronized acquisition indices. It is symmetric and does not establish EEG→fNIRS direction, a hemodynamic lag, or physiological causality.",
    ]
    (directory / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics
