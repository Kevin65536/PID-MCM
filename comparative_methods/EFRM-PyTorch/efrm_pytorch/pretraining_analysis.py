"""Post-hoc audit and publication-ready diagnostics for EFRM pretraining runs.

The analyzer is deliberately read-only with respect to the training run.  It
uses only public train/validation logs and the explicitly exported alignment
evidence; protected test data are never opened.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .visualization import EVIDENCE_SCHEMA, retrieval_metrics


ANALYSIS_SCHEMA = "efrm_pretraining_analysis_v1"
LOSS_KEYS = (
    "loss",
    "eeg_reconstruction_loss",
    "fnirs_reconstruction_loss",
    "clip_alignment_loss",
)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#666666",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _configure_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13, 1.06, label, transform=axis.transAxes, fontsize=11,
        fontweight="bold", va="top",
    )


def _relative_change(first: float, last: float) -> float:
    return float((last - first) / first) if first != 0 else math.nan


def _harmonic(number: int) -> float:
    return float(sum(1.0 / value for value in range(1, number + 1)))


def _effective_rank(embeddings: np.ndarray) -> dict[str, float]:
    centered = np.asarray(embeddings, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    if energy.sum() == 0:
        return {"effective_rank": 0.0, "first_axis_energy_fraction": 1.0}
    probability = energy / energy.sum()
    nonzero = probability[probability > 0]
    return {
        "effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
        "first_axis_energy_fraction": float(probability[0]),
    }


def _within_modality_metrics(embeddings: np.ndarray) -> dict[str, float]:
    values = np.asarray(embeddings, dtype=np.float64)
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~np.eye(len(values), dtype=bool)]
    result = {
        "off_diagonal_cosine_mean": float(off_diagonal.mean()),
        "off_diagonal_cosine_std": float(off_diagonal.std()),
    }
    result.update(_effective_rank(values))
    return result


def _permutation_p_value(
    cosine: np.ndarray, *, permutations: int = 10_000, seed: int = 20260723
) -> float:
    matrix = np.asarray(cosine, dtype=np.float64)
    observed = float(np.diag(matrix).mean())
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    rows = np.arange(len(matrix))
    for index in range(permutations):
        null[index] = matrix[rows, rng.permutation(len(matrix))].mean()
    return float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))


def analyze_alignment_evidence(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["schema"].item()) != EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported alignment evidence schema in {path}")
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = [json.loads(str(value)) for value in payload["metadata_json"].tolist()]

    cosine = np.asarray(arrays["cosine_similarity"], dtype=np.float64)
    base = retrieval_metrics(cosine)
    size = len(cosine)
    positive = np.diag(cosine)
    negative_mask = ~np.eye(size, dtype=bool)
    negative = cosine[negative_mask]
    hardest = np.where(negative_mask, cosine, -np.inf).max(axis=1)
    positive_greater = (
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )
    datasets = sorted({str(row.get("dataset_id", "unknown")) for row in metadata})
    subjects = sorted({
        f"{row.get('dataset_id', 'unknown')}:{row.get('subject', 'unknown')}"
        for row in metadata
    })
    base.update({
        "evidence_scope": {
            "kind": "single_exported_validation_batch",
            "dataset_ids": datasets,
            "dataset_count": len(datasets),
            "subject_ids": subjects,
            "subject_count": len(subjects),
            "representative_of_full_validation": False,
        },
        "chance": {
            "top1": 1.0 / size,
            "top5": min(5, size) / size,
            "mean_rank": (size + 1.0) / 2.0,
            "mrr": _harmonic(size) / size,
        },
        "positive_minus_negative_cosine": float(positive.mean() - negative.mean()),
        "positive_vs_all_negative_auc": float(positive_greater),
        "positive_minus_hardest_negative_mean": float(np.mean(positive - hardest)),
        "identity_pair_permutation_p_one_sided": _permutation_p_value(cosine),
        "logit_multiplier": float(arrays["logit_multiplier"].item()),
        "eeg_embedding_geometry": _within_modality_metrics(arrays["eeg_embeddings"]),
        "fnirs_embedding_geometry": _within_modality_metrics(arrays["fnirs_embeddings"]),
    })
    return base, arrays, metadata


def _training_audit(
    run_dir: Path,
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    epochs: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    *,
    stale_after_hours: float,
) -> dict[str, Any]:
    epoch_ids = [int(row["epoch"]) for row in epochs]
    if epoch_ids != list(range(len(epoch_ids))):
        raise ValueError(f"completed epoch log is not contiguous: {epoch_ids}")
    grouped_steps: dict[int, int] = {}
    for row in steps:
        epoch = int(row["epoch"])
        grouped_steps[epoch] = grouped_steps.get(epoch, 0) + 1
    expected_batches = int(round(np.median([
        grouped_steps[epoch] for epoch in epoch_ids
    ]))) if epoch_ids else 0
    partial_epochs = {
        str(epoch): count for epoch, count in grouped_steps.items() if epoch not in epoch_ids
    }
    last_event_path = max(
        (run_dir / "metrics/epochs.jsonl", run_dir / "metrics/train_steps.jsonl"),
        key=lambda item: item.stat().st_mtime,
    )
    age_hours = max(0.0, (datetime.now().timestamp() - last_event_path.stat().st_mtime) / 3600.0)
    manifest_status = str(manifest.get("status", "unknown"))
    status_status = str(status.get("status", "unknown"))
    stale_running = (
        (manifest_status == "running" or status_status == "running")
        and age_hours >= stale_after_hours
    )
    planned_epochs = int(_load_planned_epochs(run_dir))
    if manifest_status == "completed" and len(epochs) >= 1:
        run_state = "completed"
    elif stale_running:
        run_state = "interrupted_or_stale"
    elif manifest_status == "running":
        run_state = "running"
    else:
        run_state = manifest_status

    best_epoch = min(epoch_ids, key=lambda index: float(epochs[index]["validation"]["loss"]))
    latest_checkpoint = run_dir / "checkpoints/latest.pt"
    best_checkpoint = run_dir / "checkpoints/best.pt"
    checkpoint_same = (
        latest_checkpoint.is_file()
        and best_checkpoint.is_file()
        and latest_checkpoint.stat().st_size == best_checkpoint.stat().st_size
        and _sha256(latest_checkpoint) == _sha256(best_checkpoint)
    )
    return {
        "run_state": run_state,
        "manifest_status": manifest_status,
        "status_file_status": status_status,
        "completed_epoch_count": len(epochs),
        "completed_epoch_ids": epoch_ids,
        "last_completed_epoch": epoch_ids[-1] if epoch_ids else None,
        "planned_epoch_count": planned_epochs,
        "minimum_epoch_count": int(_load_min_epochs(run_dir)),
        "best_epoch": best_epoch,
        "best_validation_loss": float(epochs[best_epoch]["validation"]["loss"]),
        "logged_step_count": len(steps),
        "expected_batches_per_epoch": expected_batches,
        "partial_epoch_step_counts": partial_epochs,
        "partial_epoch_fraction": {
            epoch: count / expected_batches if expected_batches else math.nan
            for epoch, count in partial_epochs.items()
        },
        "last_metric_age_hours": age_hours,
        "stale_after_hours": stale_after_hours,
        "termination_reason": (
            "not_recorded; no completed/failed terminal marker"
            if run_state == "interrupted_or_stale" else None
        ),
        "checkpoint": {
            "latest_exists": latest_checkpoint.is_file(),
            "best_exists": best_checkpoint.is_file(),
            "latest_size_bytes": latest_checkpoint.stat().st_size if latest_checkpoint.is_file() else None,
            "best_size_bytes": best_checkpoint.stat().st_size if best_checkpoint.is_file() else None,
            "latest_and_best_identical_sha256": checkpoint_same,
            "checkpoint_epoch_inferred_from_log": epoch_ids[-1] if checkpoint_same and epoch_ids else None,
        },
        "protected_test_opened": bool(
            manifest.get("protected_test_opened", False)
            or status.get("protected_test_opened", False)
        ),
    }


def _simple_yaml_scalar(path: Path, section: str, key: str, default: int) -> int:
    active = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw and not raw.startswith(" "):
            active = raw.rstrip() == f"{section}:"
            continue
        if active and raw.strip().startswith(f"{key}:"):
            return int(raw.split(":", 1)[1].strip())
    return default


def _load_planned_epochs(run_dir: Path) -> int:
    return _simple_yaml_scalar(run_dir / "resolved_config.yaml", "training", "epochs", 0)


def _load_min_epochs(run_dir: Path) -> int:
    return _simple_yaml_scalar(run_dir / "resolved_config.yaml", "training", "min_epochs", 0)


def _training_metrics(
    epochs: list[dict[str, Any]], steps: list[dict[str, Any]]
) -> dict[str, Any]:
    first, last = epochs[0], epochs[-1]
    component_changes: dict[str, dict[str, float]] = {}
    for split in ("train", "validation"):
        component_changes[split] = {}
        for key in LOSS_KEYS:
            component_changes[split][key] = _relative_change(
                float(first[split][key]), float(last[split][key])
            )

    step_gradient = np.asarray([float(row["gradient_norm"]) for row in steps])
    maximum_gradient_row = steps[int(np.argmax(step_gradient))]
    clip_threshold = 5.0
    train_clip_chance_residual: dict[str, float] = {}
    for epoch in sorted({int(row["epoch"]) for row in steps}):
        rows = [row for row in steps if int(row["epoch"]) == epoch]
        weights = np.asarray([float(row["pair_count"]) for row in rows])
        residual = np.asarray([
            float(row["clip_alignment_loss"]) - math.log(float(row["pair_count"]))
            for row in rows
        ])
        train_clip_chance_residual[str(epoch)] = float(np.average(residual, weights=weights))

    return {
        "first_to_last_relative_change": component_changes,
        "validation_generalization_gap_last": float(
            last["validation"]["loss"] - last["train"]["loss"]
        ),
        "validation_loss_monotonic_nonincreasing": bool(all(
            float(right["validation"]["loss"]) <= float(left["validation"]["loss"])
            for left, right in zip(epochs, epochs[1:])
        )),
        "gradient_norm": {
            "median": float(np.median(step_gradient)),
            "p95": float(np.quantile(step_gradient, 0.95)),
            "maximum": float(step_gradient.max()),
            "maximum_location": {
                "epoch": int(maximum_gradient_row["epoch"]),
                "batch": int(maximum_gradient_row["batch"]),
                "pair_count": int(maximum_gradient_row["pair_count"]),
            },
            "fraction_above_clip_threshold": float(np.mean(step_gradient > clip_threshold)),
            "clip_threshold": clip_threshold,
        },
        "train_clip_loss_minus_exact_log_batch_chance_by_epoch": train_clip_chance_residual,
        "total_completed_epoch_seconds": float(sum(float(row["seconds"]) for row in epochs)),
        "mean_completed_epoch_seconds": float(np.mean([float(row["seconds"]) for row in epochs])),
        "cuda_peak_allocated_gib": float(max(row["cuda_peak_allocated_gib"] for row in epochs)),
        "cuda_peak_reserved_gib": float(max(row["cuda_peak_reserved_gib"] for row in epochs)),
    }


def _plot_training_overview(epochs: list[dict[str, Any]], output: Path) -> None:
    x = np.asarray([int(row["epoch"]) + 1 for row in epochs])
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(x, [row["train"]["loss"] for row in epochs], "o-", color=COLORS["blue"], label="train")
    axis.plot(x, [row["validation"]["loss"] for row in epochs], "s-", color=COLORS["orange"], label="validation")
    axis.set(xlabel="completed epoch", ylabel="pair-weighted loss", title="Total objective")
    axis.set_xticks(x)
    axis.grid(alpha=0.2)
    axis.legend()
    _panel_label(axis, "A")

    axis = axes[0, 1]
    for key, label, color, marker in (
        ("eeg_reconstruction_loss", "EEG reconstruction", COLORS["blue"], "o"),
        ("fnirs_reconstruction_loss", "fNIRS reconstruction", COLORS["green"], "^"),
        ("clip_alignment_loss", "CLIP alignment", COLORS["red"], "s"),
    ):
        axis.plot(
            x, [row["validation"][key] for row in epochs],
            marker=marker, color=color, label=label,
        )
    axis.axhline(math.log(32), color=COLORS["grey"], ls="--", lw=1, label="log(32) nominal chance")
    axis.set(xlabel="completed epoch", ylabel="validation loss", title="Validation objective decomposition")
    axis.set_xticks(x)
    axis.grid(alpha=0.2)
    axis.legend()
    _panel_label(axis, "B")

    axis = axes[1, 0]
    for key, label, color in (
        ("eeg_reconstruction_loss", "EEG reconstruction", COLORS["blue"]),
        ("fnirs_reconstruction_loss", "fNIRS reconstruction", COLORS["green"]),
        ("clip_alignment_loss", "CLIP alignment", COLORS["red"]),
    ):
        values = np.asarray([row["validation"][key] for row in epochs], dtype=float)
        axis.plot(x, values / values[0], "o-", color=color, label=label)
    axis.axhline(1.0, color=COLORS["grey"], lw=0.8)
    axis.set(
        xlabel="completed epoch", ylabel="fraction of epoch-1 value",
        title="Component-specific learning progress",
    )
    axis.set_xticks(x)
    axis.grid(alpha=0.2)
    axis.legend()
    _panel_label(axis, "C")

    axis = axes[1, 1]
    allocated = [row["cuda_peak_allocated_gib"] for row in epochs]
    reserved = [row["cuda_peak_reserved_gib"] for row in epochs]
    axis.plot(x, allocated, "o-", color=COLORS["purple"], label="allocated")
    axis.plot(x, reserved, "s--", color=COLORS["sky"], label="reserved")
    axis.set(xlabel="completed epoch", ylabel="peak GPU memory (GiB)", title="Resource stability")
    axis.set_xticks(x)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    duration = axis.twinx()
    duration.plot(
        x, [row["seconds"] / 60 for row in epochs],
        "d:", color=COLORS["grey"], alpha=0.8, label="epoch duration",
    )
    duration.set_ylabel("epoch duration (min)")
    duration.spines["right"].set_visible(True)
    _panel_label(axis, "D")
    figure.suptitle("EFRM synchronized pretraining: completed-epoch diagnostics", fontsize=12)
    _save_figure(figure, output)


def _epoch_step_summary(
    steps: list[dict[str, Any]], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epoch_ids = np.asarray(sorted({int(row["epoch"]) for row in steps}))
    median, low, high = [], [], []
    for epoch in epoch_ids:
        values = np.asarray([float(row[key]) for row in steps if int(row["epoch"]) == epoch])
        median.append(np.median(values))
        low.append(np.quantile(values, 0.1))
        high.append(np.quantile(values, 0.9))
    return epoch_ids + 1, np.asarray(median), np.asarray(low), np.asarray(high)


def _plot_optimization_diagnostics(
    epochs: list[dict[str, Any]], steps: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    for axis, key, title, color in (
        (axes[0, 0], "loss", "Per-step total loss", COLORS["blue"]),
        (axes[0, 1], "gradient_norm", "Pre-clipping gradient norm", COLORS["purple"]),
    ):
        x, median, low, high = _epoch_step_summary(steps, key)
        axis.fill_between(x, low, high, color=color, alpha=0.18, label="10th–90th percentile")
        axis.plot(x, median, "o-", color=color, label="median")
        if key == "gradient_norm":
            axis.axhline(5.0, color=COLORS["red"], ls="--", lw=1, label="clip threshold")
            axis.set_yscale("log")
        axis.set(xlabel="epoch represented in step log", ylabel=key.replace("_", " "), title=title)
        axis.grid(alpha=0.2)
        axis.legend()
    _panel_label(axes[0, 0], "A")
    _panel_label(axes[0, 1], "B")

    axis = axes[1, 0]
    x = np.asarray([int(row["epoch"]) + 1 for row in epochs])
    bottom = np.zeros(len(x))
    for key, label, color in (
        ("eeg_reconstruction_loss", "EEG reconstruction", COLORS["blue"]),
        ("fnirs_reconstruction_loss", "fNIRS reconstruction", COLORS["green"]),
        ("clip_alignment_loss", "CLIP alignment", COLORS["red"]),
    ):
        values = np.asarray([row["validation"][key] for row in epochs])
        axis.bar(x, values, bottom=bottom, color=color, label=label, width=0.72)
        bottom += values
    axis.set(xlabel="completed epoch", ylabel="validation loss", title="Loss composition")
    axis.set_xticks(x)
    axis.legend()
    _panel_label(axis, "C")

    axis = axes[1, 1]
    residual_epochs, residual_values = [], []
    for epoch in sorted({int(row["epoch"]) for row in steps}):
        rows = [row for row in steps if int(row["epoch"]) == epoch]
        weights = np.asarray([float(row["pair_count"]) for row in rows])
        values = np.asarray([
            float(row["clip_alignment_loss"]) - math.log(float(row["pair_count"]))
            for row in rows
        ])
        residual_epochs.append(epoch + 1)
        residual_values.append(np.average(values, weights=weights))
    axis.plot(residual_epochs, residual_values, "o-", color=COLORS["red"])
    axis.axhline(0.0, color=COLORS["grey"], ls="--", lw=1)
    axis.set(
        xlabel="epoch represented in step log",
        ylabel="CLIP CE − log(actual batch size)",
        title="Alignment gain over exact random-logit baseline",
    )
    axis.grid(alpha=0.2)
    _panel_label(axis, "D")
    figure.suptitle(
        "EFRM optimizer diagnostics (epoch 9 is partial when present)", fontsize=12
    )
    _save_figure(figure, output)


def _plot_alignment_diagnostics(
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    metadata: list[dict[str, Any]],
    output: Path,
) -> None:
    cosine = np.asarray(arrays["cosine_similarity"], dtype=np.float64)
    size = len(cosine)
    diagonal = np.diag(cosine)
    off_mask = ~np.eye(size, dtype=bool)
    negative = cosine[off_mask]
    hard = np.where(off_mask, cosine, -np.inf).max(axis=1)
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), constrained_layout=True)

    axis = axes[0, 0]
    center = float(negative.mean())
    spread = max(float(np.quantile(np.abs(cosine - center), 0.98)), 1e-6)
    image = axis.imshow(
        cosine, cmap="RdBu_r", vmin=center - spread, vmax=center + spread,
        interpolation="nearest", aspect="auto",
    )
    for index in range(size):
        axis.add_patch(Rectangle(
            (index - 0.48, index - 0.48), 0.96, 0.96,
            fill=False, edgecolor=COLORS["yellow"], lw=1.2,
        ))
    axis.set(
        xlabel="fNIRS candidate index", ylabel="EEG query index",
        title="Cosine matrix; yellow diagonal = declared P+",
    )
    figure.colorbar(image, ax=axis, label="cosine similarity (adaptive scale)")
    _panel_label(axis, "A")

    axis = axes[0, 1]
    violin = axis.violinplot(
        [diagonal, negative, hard], positions=[1, 2, 3],
        showmeans=True, showextrema=True,
    )
    for body, color in zip(
        violin["bodies"], (COLORS["orange"], COLORS["sky"], COLORS["red"])
    ):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    axis.set_xticks([1, 2, 3], ["positive", "all negative", "hardest negative"])
    axis.set(ylabel="cosine similarity", title="Pair-separation evidence")
    axis.grid(axis="y", alpha=0.2)
    _panel_label(axis, "B")

    axis = axes[1, 0]
    k = np.arange(1, size + 1)
    for direction, label, color in (
        ("eeg_to_fnirs", "EEG→fNIRS", COLORS["blue"]),
        ("fnirs_to_eeg", "fNIRS→EEG", COLORS["orange"]),
    ):
        ranks = np.asarray(metrics[direction]["ranks"])
        axis.step(k, [(ranks <= value).mean() for value in k], where="post", color=color, label=label)
    axis.plot(k, k / size, "--", color=COLORS["grey"], label="random-ranking expectation")
    axis.set(
        xlabel="retrieval rank k", ylabel="fraction with P+ rank ≤ k",
        title="Bidirectional synchronized-pair retrieval", xlim=(1, size), ylim=(0, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend()
    _panel_label(axis, "C")

    axis = axes[1, 1]
    geometry = [
        metrics["eeg_embedding_geometry"],
        metrics["fnirs_embedding_geometry"],
    ]
    values = [item["off_diagonal_cosine_mean"] for item in geometry]
    bars = axis.bar(
        ["EEG", "fNIRS"], values, color=[COLORS["blue"], COLORS["green"]], width=0.58
    )
    axis.axhline(1.0, color=COLORS["grey"], ls="--", lw=0.8)
    axis.set(
        ylabel="mean within-modality off-diagonal cosine",
        title="Angular concentration of validation embeddings", ylim=(0, 1.04),
    )
    for bar, item in zip(bars, geometry):
        axis.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.06,
            f"effective rank\n{item['effective_rank']:.2f}",
            ha="center", va="top", fontsize=8,
        )
    _panel_label(axis, "D")
    dataset_text = ", ".join(sorted({str(row.get("dataset_id")) for row in metadata}))
    subject_text = ", ".join(sorted({str(row.get("subject")) for row in metadata}))
    figure.suptitle(
        f"EFRM CLIP evidence: one exported batch only (n={size}; {dataset_text}; {subject_text})",
        fontsize=12,
    )
    _save_figure(figure, output)


def _write_epoch_csv(path: Path, epochs: Iterable[Mapping[str, Any]]) -> None:
    fieldnames = [
        "epoch", "seconds", "learning_rate", "train_loss", "validation_loss",
        "train_eeg_reconstruction_loss", "validation_eeg_reconstruction_loss",
        "train_fnirs_reconstruction_loss", "validation_fnirs_reconstruction_loss",
        "train_clip_alignment_loss", "validation_clip_alignment_loss",
        "cuda_peak_allocated_gib", "cuda_peak_reserved_gib",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in epochs:
            writer.writerow({
                "epoch": int(row["epoch"]),
                "seconds": row["seconds"],
                "learning_rate": row["learning_rate"],
                "train_loss": row["train"]["loss"],
                "validation_loss": row["validation"]["loss"],
                "train_eeg_reconstruction_loss": row["train"]["eeg_reconstruction_loss"],
                "validation_eeg_reconstruction_loss": row["validation"]["eeg_reconstruction_loss"],
                "train_fnirs_reconstruction_loss": row["train"]["fnirs_reconstruction_loss"],
                "validation_fnirs_reconstruction_loss": row["validation"]["fnirs_reconstruction_loss"],
                "train_clip_alignment_loss": row["train"]["clip_alignment_loss"],
                "validation_clip_alignment_loss": row["validation"]["clip_alignment_loss"],
                "cuda_peak_allocated_gib": row["cuda_peak_allocated_gib"],
                "cuda_peak_reserved_gib": row["cuda_peak_reserved_gib"],
            })


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _alignment_interpretation(
    training: Mapping[str, Any],
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Grade an alignment warning without overgeneralizing one saved batch."""

    clip_change = float(
        training["first_to_last_relative_change"]["validation"]["clip_alignment_loss"]
    )
    chance = alignment["chance"]
    retrieval_at_chance = bool(
        alignment["eeg_to_fnirs"]["mrr"] <= 1.1 * chance["mrr"]
        and alignment["fnirs_to_eeg"]["mrr"] <= 1.1 * chance["mrr"]
    )
    no_pair_separation = bool(
        alignment["positive_vs_all_negative_auc"] <= 0.52
        and alignment["positive_minus_negative_cosine"] <= 0.0
        and alignment["identity_pair_permutation_p_one_sided"] >= 0.05
    )
    clip_plateau = bool(abs(clip_change) < 0.01)
    warning = bool(clip_plateau and retrieval_at_chance and no_pair_separation)

    pair_count = int(alignment["pair_count"])
    multiplier = float(alignment["logit_multiplier"])
    random_ce = math.log(pair_count)
    ideal_bounded_ce = math.log(
        1.0 + (pair_count - 1.0) * math.exp(-2.0 * multiplier)
    )
    return {
        "alignment_failure_warning": warning,
        "alignment_warning_level": (
            "serious_scope_limited_warning" if warning else "not_triggered"
        ),
        "warning_components": {
            "validation_clip_plateau_under_1_percent": clip_plateau,
            "saved_batch_bidirectional_retrieval_at_chance": retrieval_at_chance,
            "saved_batch_no_positive_pair_separation": no_pair_separation,
        },
        "dataset_level_alignment_impossibility_claim_supported": False,
        "source_scale_ce_geometry": {
            "pair_count": pair_count,
            "fixed_logit_multiplier": multiplier,
            "random_ce": random_ce,
            "ideal_cosine_bound_ce": ideal_bounded_ce,
            "maximum_possible_ce_reduction": random_ce - ideal_bounded_ce,
        },
    }


def _markdown_report(
    run_dir: Path,
    audit: Mapping[str, Any],
    training: Mapping[str, Any],
    alignment: Mapping[str, Any],
    epochs: list[dict[str, Any]],
) -> str:
    changes = training["first_to_last_relative_change"]["validation"]
    chance = alignment["chance"]
    evidence_scope = alignment["evidence_scope"]
    interpretation = _alignment_interpretation(training, alignment)
    scale_geometry = interpretation["source_scale_ce_geometry"]
    partial = audit["partial_epoch_step_counts"]
    partial_text = (
        ", ".join(
            f"epoch {int(epoch) + 1}: {count}/{audit['expected_batches_per_epoch']} "
            f"({_percent(audit['partial_epoch_fraction'][epoch])})"
            for epoch, count in partial.items()
        )
        if partial else "none"
    )
    lines = [
        "# EFRM synchronized pretraining analysis",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Analysis schema: `{ANALYSIS_SCHEMA}`",
        "- Scope: public train/validation artifacts only; protected test remained locked.",
        "",
        "## Executive conclusion",
        "",
        f"1. **Run integrity:** `{audit['run_state']}`. There are "
        f"{audit['completed_epoch_count']} complete epochs; partial work: {partial_text}. "
        "The process left no terminal success/failure marker, so the termination cause cannot "
        "be recovered from the run artifacts.",
        f"2. **Reconstruction learned:** validation total loss changed by "
        f"{_percent(changes['loss'])}; EEG reconstruction by "
        f"{_percent(changes['eeg_reconstruction_loss'])}; fNIRS reconstruction by "
        f"{_percent(changes['fnirs_reconstruction_loss'])}.",
        f"3. **Cross-modal alignment did not emerge in the saved evidence:** validation CLIP "
        f"loss changed by {_percent(changes['clip_alignment_loss'])}; EEG→fNIRS and "
        f"fNIRS→EEG Top-1 are {alignment['eeg_to_fnirs']['top1']:.4f} and "
        f"{alignment['fnirs_to_eeg']['top1']:.4f}, versus chance {chance['top1']:.4f}. "
        f"Positive-minus-negative cosine is {alignment['positive_minus_negative_cosine']:.6g}.",
        "4. **Evidence limitation:** the trainer exported only the final validation batch, not "
        "the full validation set. Therefore retrieval and embedding-collapse findings are "
        "strong diagnostics for that batch, but not dataset-wide performance estimates.",
        f"5. **Failure-warning grade:** `{interpretation['alignment_warning_level']}`. This is "
        "a warning that the source-faithful alignment branch has not activated, not evidence "
        "that synchronized EEG-fNIRS contains no alignable physiological relationship.",
        "",
        "## Run integrity",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Manifest/status | `{audit['manifest_status']}` / `{audit['status_file_status']}` |",
        f"| Complete/planned epochs | {audit['completed_epoch_count']} / {audit['planned_epoch_count']} |",
        f"| Minimum epochs before early stop | {audit['minimum_epoch_count']} |",
        f"| Last complete / best epoch (1-based) | {audit['last_completed_epoch'] + 1} / {audit['best_epoch'] + 1} |",
        f"| Best validation loss | {audit['best_validation_loss']:.6f} |",
        f"| Logged optimizer steps | {audit['logged_step_count']} |",
        f"| Partial epoch | {partial_text} |",
        f"| Best/latest checkpoints byte-identical | {audit['checkpoint']['latest_and_best_identical_sha256']} |",
        f"| Protected test opened | {audit['protected_test_opened']} |",
        "",
        "A stale `running` marker is not a completed run. For this audited run, the latest "
        "durable checkpoint represents the last complete epoch; optimizer updates from the "
        "partial epoch were not checkpointed.",
        "",
        "## Optimization and reconstruction",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Validation total change | {_percent(changes['loss'])} |",
        f"| Validation EEG reconstruction change | {_percent(changes['eeg_reconstruction_loss'])} |",
        f"| Validation fNIRS reconstruction change | {_percent(changes['fnirs_reconstruction_loss'])} |",
        f"| Validation CLIP change | {_percent(changes['clip_alignment_loss'])} |",
        f"| Final validation−train total-loss gap | {training['validation_generalization_gap_last']:.6f} |",
        f"| Median / p95 pre-clip gradient norm | {training['gradient_norm']['median']:.3f} / {training['gradient_norm']['p95']:.3f} |",
        f"| Maximum pre-clip gradient norm | {training['gradient_norm']['maximum']:.3f} "
        f"(epoch {training['gradient_norm']['maximum_location']['epoch'] + 1}, "
        f"batch {training['gradient_norm']['maximum_location']['batch'] + 1}, "
        f"n={training['gradient_norm']['maximum_location']['pair_count']}) |",
        f"| Steps exceeding clip threshold 5 | {_percent(training['gradient_norm']['fraction_above_clip_threshold'])} |",
        f"| Mean complete-epoch duration | {training['mean_completed_epoch_seconds'] / 60:.2f} min |",
        f"| Peak allocated / reserved GPU memory | {training['cuda_peak_allocated_gib']:.2f} / {training['cuda_peak_reserved_gib']:.2f} GiB |",
        "",
        "The total objective is improving because both masked-autoencoder reconstruction "
        "objectives improve. The CLIP term remains near the random-logit cross-entropy "
        "baseline. The source-faithful fixed multiplier 0.1 also compresses all logits into "
        "a narrow range, so retrieval ranks and cosine separation—not CLIP loss alone—are "
        "the decisive diagnostics.",
        "",
        "With batch size "
        f"{scale_geometry['pair_count']} and fixed multiplier "
        f"{scale_geometry['fixed_logit_multiplier']:.3g}, cosine logits are bounded to "
        f"±{scale_geometry['fixed_logit_multiplier']:.3g}. Even the unattainably idealized "
        "matrix with every positive cosine at +1 and every negative at −1 can reduce CE only "
        f"from {scale_geometry['random_ce']:.6f} to "
        f"{scale_geometry['ideal_cosine_bound_ce']:.6f}. Therefore a nearly flat scalar CLIP "
        "loss is not decisive by itself.",
        "",
        "## Saved CLIP alignment evidence",
        "",
        "| Metric | Observed | Random expectation |",
        "|---|---:|---:|",
        f"| Pair count | {alignment['pair_count']} | — |",
        f"| EEG→fNIRS Top-1 | {alignment['eeg_to_fnirs']['top1']:.4f} | {chance['top1']:.4f} |",
        f"| fNIRS→EEG Top-1 | {alignment['fnirs_to_eeg']['top1']:.4f} | {chance['top1']:.4f} |",
        f"| EEG→fNIRS MRR | {alignment['eeg_to_fnirs']['mrr']:.4f} | {chance['mrr']:.4f} |",
        f"| fNIRS→EEG MRR | {alignment['fnirs_to_eeg']['mrr']:.4f} | {chance['mrr']:.4f} |",
        f"| EEG→fNIRS mean rank | {alignment['eeg_to_fnirs']['mean_rank']:.3f} | {chance['mean_rank']:.3f} |",
        f"| fNIRS→EEG mean rank | {alignment['fnirs_to_eeg']['mean_rank']:.3f} | {chance['mean_rank']:.3f} |",
        f"| Positive cosine mean | {alignment['positive_cosine_mean']:.6f} | — |",
        f"| Negative cosine mean | {alignment['negative_cosine_mean']:.6f} | — |",
        f"| Positive−negative cosine | {alignment['positive_minus_negative_cosine']:.6f} | > 0 desired |",
        f"| Positive-vs-negative AUC | {alignment['positive_vs_all_negative_auc']:.4f} | 0.5000 |",
        f"| Mean positive−hardest-negative margin | {alignment['positive_minus_hardest_negative_mean']:.6f} | > 0 desired |",
        f"| Identity-pair permutation p (one-sided) | {alignment['identity_pair_permutation_p_one_sided']:.4f} | — |",
        "",
        f"Evidence scope: {alignment['pair_count']} pairs from "
        f"{', '.join(evidence_scope['dataset_ids'])}; subject keys "
        f"{', '.join(evidence_scope['subject_ids'])}. This is one exported batch and cannot "
        "support cross-dataset or subject-level generalization claims.",
        "",
        "### Embedding geometry",
        "",
        "| Modality | Mean within-modality cosine | Centered effective rank | First-axis energy |",
        "|---|---:|---:|---:|",
        f"| EEG | {alignment['eeg_embedding_geometry']['off_diagonal_cosine_mean']:.6f} | "
        f"{alignment['eeg_embedding_geometry']['effective_rank']:.3f} | "
        f"{_percent(alignment['eeg_embedding_geometry']['first_axis_energy_fraction'])} |",
        f"| fNIRS | {alignment['fnirs_embedding_geometry']['off_diagonal_cosine_mean']:.6f} | "
        f"{alignment['fnirs_embedding_geometry']['effective_rank']:.3f} | "
        f"{_percent(alignment['fnirs_embedding_geometry']['first_axis_energy_fraction'])} |",
        "",
        "Within-modality cosines near 1 indicate strong angular concentration. In combination "
        "with chance retrieval, this is consistent with a collapsed or weakly discriminative "
        "contrastive representation on the saved batch; it does not imply that the "
        "reconstruction branches failed.",
        "",
        "## Recommended decision",
        "",
        "Do not label this run as completed and do not begin protected-test evaluation. Preserve "
        "the epoch-8 checkpoint as the faithful baseline. Before the epoch-20 decision gate, "
        "extend validation evidence capture to a deterministic, dataset/subject-stratified "
        "sample and archive it per epoch. Resume the faithful 0.1-multiplier run to its minimum "
        "20 epochs. If full public-validation retrieval remains at chance, positive-vs-negative "
        "AUC remains near 0.5, and positive margins remain non-positive at epoch 20, record "
        "`source_faithful_alignment_failed_on_sync_track`; this is a method/data-regime result, "
        "not a claim that physiological coupling is absent. A learned or conventional divisive "
        "temperature must be trained from scratch as a separately named diagnostic ablation. "
        "The faithful and ablation results must not be conflated.",
        "",
        "## Figures and data",
        "",
        "- `figures/training_overview.{svg,png}`",
        "- `figures/optimization_diagnostics.{svg,png}`",
        "- `figures/alignment_diagnostics.{svg,png}`",
        "- `tables/epoch_metrics.csv`",
        "- `analysis_metrics.json`",
        "",
        "All figures use public validation evidence only. SVG is the editable vector source; "
        "PNG is rendered at 300 dpi.",
        "",
    ]
    return "\n".join(lines)


def analyze_pretraining_run(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    stale_after_hours: float = 1.0,
) -> dict[str, Any]:
    """Audit one EFRM run and generate figures, tables, JSON, and Markdown."""

    _configure_style()
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else run / "analysis"
    required = (
        run / "manifest.json",
        run / "status.json",
        run / "resolved_config.yaml",
        run / "metrics/epochs.jsonl",
        run / "metrics/train_steps.jsonl",
        run / "figure_data/clip_alignment_evidence.npz",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing EFRM run artifacts: {missing}")

    manifest = _read_json(run / "manifest.json")
    status = _read_json(run / "status.json")
    if manifest.get("protected_test_opened") or status.get("protected_test_opened"):
        raise PermissionError("refusing to analyze a run that reports opened protected test data")
    epochs = _read_jsonl(run / "metrics/epochs.jsonl")
    steps = _read_jsonl(run / "metrics/train_steps.jsonl")
    if not epochs or not steps:
        raise ValueError("analysis requires at least one completed epoch and optimizer step")

    audit = _training_audit(
        run, manifest, status, epochs, steps, stale_after_hours=stale_after_hours
    )
    training = _training_metrics(epochs, steps)
    alignment, arrays, metadata = analyze_alignment_evidence(
        run / "figure_data/clip_alignment_evidence.npz"
    )
    interpretation = _alignment_interpretation(training, alignment)
    interpretation.update({
        "reconstruction_learning_observed": bool(
            training["first_to_last_relative_change"]["validation"][
                "eeg_reconstruction_loss"
            ] < 0
            and training["first_to_last_relative_change"]["validation"][
                "fnirs_reconstruction_loss"
            ] < 0
        ),
        "saved_batch_alignment_above_chance": bool(
            alignment["eeg_to_fnirs"]["top1"] > alignment["chance"]["top1"]
            or alignment["fnirs_to_eeg"]["top1"] > alignment["chance"]["top1"]
        ),
        "full_validation_alignment_claim_supported": False,
    })
    result = {
        "schema": ANALYSIS_SCHEMA,
        "run_id": manifest.get("run_id", run.name),
        "generated_at": datetime.now().isoformat(),
        "source_run": str(run),
        "audit": audit,
        "training": training,
        "alignment": alignment,
        "interpretation": interpretation,
    }

    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    _plot_training_overview(epochs, output / "figures/training_overview")
    _plot_optimization_diagnostics(epochs, steps, output / "figures/optimization_diagnostics")
    _plot_alignment_diagnostics(
        arrays, alignment, metadata, output / "figures/alignment_diagnostics"
    )
    _write_epoch_csv(output / "tables/epoch_metrics.csv", epochs)
    (output / "analysis_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        _markdown_report(run, audit, training, alignment, epochs), encoding="utf-8"
    )
    return result
