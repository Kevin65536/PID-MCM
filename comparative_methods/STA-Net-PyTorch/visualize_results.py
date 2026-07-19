#!/usr/bin/env python3
"""Re-evaluate completed STA-Net runs and build a reproduction report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch import STANet, STANetConfig, STANetUnifiedTaskDataset, get_sta_net_task_spec
from train import make_loader, move_batch

REPORT_SCHEMA = "sta_net_reproduction_report_v1"
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in (("svg", {}), ("png", {"dpi": 300})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def classification_metrics(target: np.ndarray, probability: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    predicted = probability.argmax(axis=1)
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        target, predicted, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(target, predicted, labels=labels)
    row_total = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_total, out=np.zeros_like(matrix, dtype=float), where=row_total > 0)
    confidence = probability.max(axis=1)
    correctness = (predicted == target).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    bin_index = np.minimum(np.digitize(confidence, bins[1:-1]), 9)
    calibration = []
    ece = 0.0
    for index in range(10):
        keep = bin_index == index
        count = int(keep.sum())
        mean_confidence = float(confidence[keep].mean()) if count else None
        mean_accuracy = float(correctness[keep].mean()) if count else None
        if count:
            ece += count / len(target) * abs(mean_accuracy - mean_confidence)
        calibration.append({
            "lower": float(bins[index]), "upper": float(bins[index + 1]), "count": count,
            "mean_confidence": mean_confidence, "accuracy": mean_accuracy,
        })
    one_hot = np.eye(len(class_names), dtype=float)[target]
    metrics: dict[str, Any] = {
        "sample_count": int(len(target)),
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_f1": float(f1_score(target, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(target, predicted, labels=labels, average="weighted", zero_division=0)),
        "multiclass_brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "expected_calibration_error_10bin": float(ece),
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "calibration_bins": calibration,
        "per_class": {
            name: {
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]), "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
    }
    try:
        metrics["macro_roc_auc_ovr"] = float(
            roc_auc_score(one_hot, probability, average="macro", multi_class="ovr")
        )
        metrics["macro_average_precision"] = float(
            average_precision_score(one_hot, probability, average="macro")
        )
    except ValueError:
        metrics["macro_roc_auc_ovr"] = None
        metrics["macro_average_precision"] = None
    return metrics


def concordance_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    covariance = float(np.mean((target - target.mean()) * (prediction - prediction.mean())))
    denominator = float(target.var() + prediction.var() + (target.mean() - prediction.mean()) ** 2)
    return 2.0 * covariance / denominator if denominator > 0 else float("nan")


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    target_names: Sequence[str],
) -> dict[str, Any]:
    per_target: dict[str, Any] = {}
    all_target: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    for coordinate, name in enumerate(target_names):
        keep = np.asarray(valid_mask[:, coordinate], dtype=bool)
        truth = np.asarray(target[:, coordinate][keep], dtype=np.float64)
        estimate = np.asarray(prediction[:, coordinate][keep], dtype=np.float64)
        error = estimate - truth
        pearson = float(np.corrcoef(truth, estimate)[0, 1]) if truth.size > 1 else float("nan")
        per_target[name] = {
            "valid_count": int(truth.size),
            "mae_native": float(np.mean(np.abs(error))),
            "rmse_native": float(np.sqrt(np.mean(error ** 2))),
            "r2_native": float(r2_score(truth, estimate)) if truth.size > 1 else None,
            "pearson_r": pearson,
            "concordance_correlation": concordance_correlation(truth, estimate),
            "bias_native": float(np.mean(error)),
        }
        all_target.append(truth)
        all_prediction.append(estimate)
    truth = np.concatenate(all_target)
    estimate = np.concatenate(all_prediction)
    error = estimate - truth
    return {
        "valid_coordinate_count": int(truth.size),
        "mae_native": float(np.mean(np.abs(error))),
        "rmse_native": float(np.sqrt(np.mean(error ** 2))),
        "per_target": per_target,
    }


def plot_training_curves(task_key: str, task_dir: Path, output_dir: Path) -> list[str]:
    train = read_jsonl(task_dir / "metrics" / "train_epochs.jsonl")
    validation = read_jsonl(task_dir / "metrics" / "validation_epochs.jsonl")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    if train:
        epochs = [row["epoch"] for row in train]
        axes[0, 0].plot(epochs, [row["mean_train_loss"] for row in train], marker="o", ms=2, label="Train")
        axes[1, 0].plot(epochs, [row.get("samples_per_second", np.nan) for row in train], marker="o", ms=2)
    if validation:
        val_epochs = [row["epoch"] for row in validation]
        axes[0, 0].plot(val_epochs, [row["loss"] for row in validation], marker="s", ms=2, label="Validation")
        primary_key = "accuracy" if "accuracy" in validation[0] else "masked_mae_scaled"
        axes[0, 1].plot(val_epochs, [row.get(primary_key, np.nan) for row in validation], marker="o", ms=2)
        axes[0, 1].set_ylabel(primary_key.replace("_", " ").title())
        axes[1, 1].plot(val_epochs, [row.get("elapsed_seconds", np.nan) for row in validation], marker="o", ms=2)
    axes[0, 0].set(title="Optimization loss", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title="Validation primary diagnostic", xlabel="Epoch")
    axes[1, 0].set(title="Training throughput", xlabel="Epoch", ylabel="Samples / second")
    axes[1, 1].set(title="Validation wall time", xlabel="Epoch", ylabel="Seconds")
    for label, ax in zip("ABCD", axes.flat, strict=True):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", va="top")
        ax.grid(alpha=0.2)
    fig.suptitle(f"STA-Net training diagnostics — {task_key}", fontweight="bold")
    return save_figure(fig, output_dir / "training_curves")


def plot_confusion(metrics: Mapping[str, Any], class_names: Sequence[str], output_dir: Path) -> list[str]:
    matrices = (
        (np.asarray(metrics["confusion_matrix"]), "Validation confusion (count)", "Count"),
        (np.asarray(metrics["normalized_confusion_matrix"]), "Validation confusion (row normalized)", "Fraction"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    for ax, (matrix, title, color_label) in zip(axes, matrices, strict=True):
        image = ax.imshow(matrix, cmap="cividis", aspect="equal", vmin=0)
        threshold = float(matrix.max()) / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                text = f"{value:.2f}" if matrix.dtype.kind == "f" else str(int(value))
                ax.text(column, row, text, ha="center", va="center", color="white" if value > threshold else "black")
        ax.set(title=title, xlabel="Predicted class", ylabel="True class")
        ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        fig.colorbar(image, ax=ax, label=color_label, fraction=0.046)
    return save_figure(fig, output_dir / "confusion_matrices")


def plot_classification_diagnostics(
    target: np.ndarray,
    probability: np.ndarray,
    metrics: Mapping[str, Any],
    class_names: Sequence[str],
    output_dir: Path,
) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), constrained_layout=True)
    x = np.arange(len(class_names))
    width = 0.24
    for offset, key, color in zip((-width, 0.0, width), ("precision", "recall", "f1"), OKABE_ITO[:3], strict=True):
        axes[0, 0].bar(x + offset, [metrics["per_class"][name][key] for name in class_names], width, label=key.title(), color=color)
    axes[0, 0].set(xticks=x, xticklabels=class_names, ylim=(0, 1), ylabel="Score", title="Per-class performance")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].legend(frameon=False)
    labels = np.arange(len(class_names))
    one_hot = np.eye(len(class_names))[target]
    for index, name in enumerate(class_names):
        if len(np.unique(one_hot[:, index])) < 2:
            continue
        false_positive, true_positive, _ = roc_curve(one_hot[:, index], probability[:, index])
        axes[0, 1].plot(false_positive, true_positive, label=name)
        precision, recall, _ = precision_recall_curve(one_hot[:, index], probability[:, index])
        axes[1, 0].plot(recall, precision, label=name)
    axes[0, 1].plot((0, 1), (0, 1), "--", color="#777777", lw=1)
    axes[0, 1].set(title="One-vs-rest ROC", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[1, 0].set(title="One-vs-rest precision–recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axes[0, 1].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    calibration = [row for row in metrics["calibration_bins"] if row["count"]]
    axes[1, 1].plot((0, 1), (0, 1), "--", color="#777777", lw=1, label="Perfect")
    axes[1, 1].plot([row["mean_confidence"] for row in calibration], [row["accuracy"] for row in calibration], marker="o", label="Observed")
    axes[1, 1].set(title=f"Reliability (ECE={metrics['expected_calibration_error_10bin']:.3f})", xlabel="Mean confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].legend(frameon=False)
    for label, ax in zip("ABCD", axes.flat, strict=True):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", va="top")
        ax.grid(alpha=0.2)
    return save_figure(fig, output_dir / "classification_diagnostics")


def plot_regression_diagnostics(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    target_names: Sequence[str],
    output_dir: Path,
    max_examples: int,
) -> list[str]:
    fig, axes = plt.subplots(2, len(target_names), figsize=(7.4, 5.6), constrained_layout=True, squeeze=False)
    for coordinate, name in enumerate(target_names):
        keep = valid_mask[:, coordinate].astype(bool)
        truth = target[:, coordinate][keep]
        estimate = prediction[:, coordinate][keep]
        axes[0, coordinate].scatter(truth, estimate, s=5, alpha=0.25, rasterized=True)
        lower = float(min(truth.min(), estimate.min()))
        upper = float(max(truth.max(), estimate.max()))
        axes[0, coordinate].plot((lower, upper), (lower, upper), "--", color="#777777")
        axes[0, coordinate].set(title=f"{name}: prediction agreement", xlabel="Target (native)", ylabel="Prediction (native)")
        residual = estimate - truth
        residual_span = float(np.ptp(residual))
        residual_scale = max(1.0, float(np.max(np.abs(residual))))
        if residual_span <= np.finfo(np.float64).eps * residual_scale * 8.0:
            center = float(np.mean(residual))
            half_width = max(1e-6, abs(center) * 1e-6)
            histogram_bins: int | np.ndarray = np.linspace(center - half_width, center + half_width, 4)
        else:
            histogram_bins = min(40, max(5, int(np.sqrt(residual.size))))
        axes[1, coordinate].hist(residual, bins=histogram_bins, color=OKABE_ITO[1], alpha=0.8)
        axes[1, coordinate].axvline(0, color="#000000", lw=1)
        axes[1, coordinate].set(title=f"{name}: residual distribution", xlabel="Prediction − target", ylabel="Count")
    for label, ax in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", axes.flat, strict=False):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", va="top")
        ax.grid(alpha=0.2)
    paths = save_figure(fig, output_dir / "regression_diagnostics")
    example_count = min(max_examples, target.shape[0])
    fig, axes = plt.subplots(example_count, len(target_names), figsize=(7.4, max(2.2, 1.8 * example_count)), constrained_layout=True, squeeze=False)
    for sample in range(example_count):
        for coordinate, name in enumerate(target_names):
            time_index = np.arange(target.shape[-1])
            keep = valid_mask[sample, coordinate].astype(bool)
            axes[sample, coordinate].plot(time_index[keep], target[sample, coordinate, keep], marker="o", ms=2, label="Target")
            axes[sample, coordinate].plot(time_index[keep], prediction[sample, coordinate, keep], marker="s", ms=2, label="Prediction")
            axes[sample, coordinate].set(title=f"Example {sample + 1}: {name}", xlabel="Target time index", ylabel="Native value")
            axes[sample, coordinate].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    paths.extend(save_figure(fig, output_dir / "regression_sequence_examples"))
    return paths


def plot_alignment_diagnostics(lag_attention: np.ndarray, fusion_weights: np.ndarray, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    lag_mean = lag_attention.mean(axis=0)
    lag_std = lag_attention.std(axis=0)
    x = np.arange(len(lag_mean))
    axes[0].plot(x, lag_mean, marker="o", ms=3)
    axes[0].fill_between(x, lag_mean - lag_std, lag_mean + lag_std, alpha=0.2)
    axes[0].set(title="EGTA lag attention", xlabel="fNIRS lag index", ylabel="Attention weight")
    axes[1].hist(fusion_weights[:, 0], bins=25, alpha=0.75, label="Fusion branch")
    axes[1].hist(fusion_weights[:, 1], bins=25, alpha=0.60, label="fNIRS branch")
    axes[1].set(title="Decision-gate distribution", xlabel="Gate weight", ylabel="Count", xlim=(0, 1))
    axes[1].legend(frameon=False)
    for label, ax in zip("AB", axes, strict=True):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", va="top")
        ax.grid(alpha=0.2)
    return save_figure(fig, output_dir / "alignment_and_fusion_diagnostics")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for result evaluation but unavailable")
    return torch.device(value)


def evaluate_task(
    task_dir: Path,
    output_dir: Path,
    *,
    config_path: Path,
    device: torch.device,
    workers: int,
    allow_incomplete: bool,
    max_sequence_examples: int,
) -> dict[str, Any]:
    manifest = read_json(task_dir / "manifest.json")
    task_key = str(manifest["task"]["key"])
    status = str(manifest.get("status", "unknown"))
    if status != "completed" and not allow_incomplete:
        raise RuntimeError(f"Task {task_key} has status={status!r}; pass --allow-incomplete for a preview")
    task_output = output_dir / task_key
    figure_dir = task_output / "figures"
    artifacts = plot_training_curves(task_key, task_dir, figure_dir)
    checkpoint_path = task_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = task_dir / "checkpoint_latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No best/latest checkpoint found for {task_key}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    spec = get_sta_net_task_spec(task_key)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = STANetUnifiedTaskDataset(spec, cache_root=str(config["data"]["cache_root"]))
    scaler = checkpoint.get("target_scaler")
    if scaler is not None:
        dataset.adapter.set_target_scaler(scaler["center"], scaler["scale"])
    split = read_json(task_dir / "split_manifest.json")
    validation_subjects = set(split["validation_subjects"])
    validation_indices = [
        index for index in range(len(dataset))
        if dataset.lightweight_metadata(index)["subject"] in validation_subjects
    ]
    if len(validation_indices) != int(split["validation_sample_count"]):
        raise RuntimeError(f"Validation split drift for {task_key}: {len(validation_indices)} != {split['validation_sample_count']}")
    batch_size = int(manifest.get("batch_size", 16))
    loader, _ = make_loader(
        dataset, validation_indices, batch_size=batch_size, workers=workers, shuffle=False,
        seed=int(split["seed"]),
    )
    model = STANet(STANetConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    amp_enabled = bool(manifest.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if manifest.get("amp_dtype") == "bfloat16" else torch.float16
    predictions, targets, masks, lags, gates = [], [], [], [], []
    sample_ids: list[str] = []
    subjects: list[str] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
            with context:
                outputs = model(batch["eeg"], batch["fnirs"])
            predictions.append(outputs["prediction"].detach().float().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
            masks.append(batch["target_valid_mask"].detach().cpu().numpy())
            lags.append(outputs["lag_attention"].detach().float().cpu().numpy())
            gates.append(outputs["fusion_weights"].detach().float().cpu().numpy())
            sample_ids.extend(batch["sample_id"])
            subjects.extend(batch["subject"])
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    valid_mask = np.concatenate(masks)
    lag_attention = np.concatenate(lags)
    fusion_weights = np.concatenate(gates)
    prediction_path = task_output / "validation_predictions.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path, sample_id=np.asarray(sample_ids, dtype=str), subject=np.asarray(subjects, dtype=str),
        prediction=prediction, target=target, target_valid_mask=valid_mask,
        lag_attention=lag_attention, fusion_weights=fusion_weights,
    )
    if spec.task_type == "classification":
        metrics = classification_metrics(target, prediction, spec.class_names)
        artifacts.extend(plot_confusion(metrics, spec.class_names, figure_dir))
        artifacts.extend(plot_classification_diagnostics(target, prediction, metrics, spec.class_names, figure_dir))
    else:
        if scaler is None:
            raise RuntimeError("Regression visualization requires the train-only target scaler in the checkpoint")
        center = np.asarray(scaler["center"], dtype=np.float32)[None, :, None]
        scale = np.asarray(scaler["scale"], dtype=np.float32)[None, :, None]
        target_native = target * scale + center
        prediction_native = prediction * scale + center
        metrics = regression_metrics(target_native, prediction_native, valid_mask, spec.target_names)
        artifacts.extend(plot_regression_diagnostics(
            target_native, prediction_native, valid_mask, spec.target_names, figure_dir, max_sequence_examples
        ))
    artifacts.extend(plot_alignment_diagnostics(lag_attention, fusion_weights, figure_dir))
    task_summary = {
        "schema": REPORT_SCHEMA,
        "task": task_key,
        "task_type": spec.task_type,
        "source_run_status": status,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "evaluation_partition": "validation",
        "validation_subjects": sorted(validation_subjects),
        "protected_test_opened": False,
        "metrics": metrics,
        "prediction_artifact": str(prediction_path),
        "figures": artifacts,
        "claim_boundary": "validation reproduction diagnostics; not protected-test or comparative superiority evidence",
    }
    write_json(task_output / "summary.json", task_summary)
    return task_summary


def discover_task_dirs(run_root: Path, selected: Sequence[str] | None) -> list[Path]:
    if (run_root / "manifest.json").exists() and "task" in read_json(run_root / "manifest.json"):
        candidates = [run_root]
    else:
        candidates = [path for path in sorted(run_root.iterdir()) if (path / "manifest.json").exists()]
    if selected:
        selected_set = set(selected)
        candidates = [path for path in candidates if read_json(path / "manifest.json").get("task", {}).get("key") in selected_set]
    return candidates


def plot_suite_overview(summaries: Sequence[Mapping[str, Any]], output_dir: Path) -> list[str]:
    classification = [row for row in summaries if row["task_type"] == "classification"]
    regression = [row for row in summaries if row["task_type"] == "regression"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)
    if classification:
        tasks = [row["task"] for row in classification]
        x = np.arange(len(tasks))
        width = 0.26
        for offset, key, color in zip((-width, 0.0, width), ("accuracy", "balanced_accuracy", "macro_f1"), OKABE_ITO[:3], strict=True):
            axes[0].bar(x + offset, [row["metrics"][key] for row in classification], width, label=key.replace("_", " ").title(), color=color)
        axes[0].set(xticks=x, xticklabels=tasks, ylim=(0, 1), ylabel="Validation score", title="Classification reproduction")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].legend(frameon=False)
    else:
        axes[0].text(0.5, 0.5, "No classification tasks", ha="center", va="center")
    if regression:
        tasks = [row["task"] for row in regression]
        x = np.arange(len(tasks))
        axes[1].bar(x - 0.18, [row["metrics"]["mae_native"] for row in regression], 0.36, label="MAE", color=OKABE_ITO[1])
        axes[1].bar(x + 0.18, [row["metrics"]["rmse_native"] for row in regression], 0.36, label="RMSE", color=OKABE_ITO[0])
        axes[1].set(xticks=x, xticklabels=tasks, ylabel="Native-coordinate error", title="Regression reproduction")
        axes[1].legend(frameon=False)
    else:
        axes[1].text(0.5, 0.5, "No regression tasks", ha="center", va="center")
    for label, ax in zip("AB", axes, strict=True):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold", va="top")
        ax.grid(axis="y", alpha=0.2)
    return save_figure(fig, output_dir / "suite_overview")


def write_metrics_csv(summaries: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = []
    for summary in summaries:
        metrics = summary["metrics"]
        row = {"task": summary["task"], "task_type": summary["task_type"]}
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_roc_auc_ovr", "macro_average_precision", "mae_native", "rmse_native"):
            row[key] = metrics.get(key)
        rows.append(row)
    fieldnames = ["task", "task_type", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_roc_auc_ovr", "macro_average_precision", "mae_native", "rmse_native"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(summaries: Sequence[Mapping[str, Any]], output_dir: Path, suite_figures: Sequence[str]) -> None:
    lines = [
        "# STA-Net reproduction evaluation", "",
        "> Validation-only diagnostics. Protected-test data were not opened, and this report does not establish comparative superiority.", "",
        "## Suite overview", "",
        f"![Suite overview]({Path(suite_figures[0]).relative_to(output_dir)})" if suite_figures else "No suite figure.", "",
        "| Task | Type | Accuracy | Balanced accuracy | Macro-F1 | MAE | RMSE |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        metrics = summary["metrics"]
        format_value = lambda key: "—" if metrics.get(key) is None else f"{metrics[key]:.4f}"
        lines.append(
            f"| {summary['task']} | {summary['task_type']} | {format_value('accuracy')} | "
            f"{format_value('balanced_accuracy')} | {format_value('macro_f1')} | "
            f"{format_value('mae_native')} | {format_value('rmse_native')} |"
        )
    for summary in summaries:
        task = summary["task"]
        lines.extend(["", f"## {task}", "", f"- [Task metrics]({task}/summary.json)", f"- [Validation predictions]({task}/validation_predictions.npz)", ""])
        figure_dir = output_dir / task / "figures"
        for path in sorted(figure_dir.glob("*.svg")):
            lines.extend([f"### {path.stem.replace('_', ' ').title()}", "", f"![{path.stem}]({path.relative_to(output_dir)})", ""])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    configure_style()
    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_root / "reproduction_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    task_dirs = discover_task_dirs(run_root, args.tasks)
    if not task_dirs:
        raise RuntimeError("No STA-Net task manifests found under the requested run root")
    device = resolve_device(args.device)
    summaries = [
        evaluate_task(
            task_dir, output_dir, config_path=config_path, device=device, workers=args.workers,
            allow_incomplete=args.allow_incomplete, max_sequence_examples=args.max_sequence_examples,
        )
        for task_dir in task_dirs
    ]
    suite_figures = plot_suite_overview(summaries, output_dir)
    write_metrics_csv(summaries, output_dir / "metrics.csv")
    write_report(summaries, output_dir, suite_figures)
    summary = {
        "schema": REPORT_SCHEMA,
        "source_run_root": str(run_root),
        "task_count": len(summaries),
        "tasks": summaries,
        "suite_figures": suite_figures,
        "protected_test_opened": False,
        "claim_boundary": "validation-only reproduction evaluation",
    }
    write_json(output_dir / "summary.json", summary)
    artifact_manifest = {
        str(path.relative_to(output_dir)): sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    write_json(output_dir / "artifact_manifest.json", {"schema": REPORT_SCHEMA, "artifacts": artifact_manifest})
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "tasks": [row["task"] for row in summaries]}, indent=2))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=str(METHOD_ROOT / "configs" / "train_all_tasks.yaml"))
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-sequence-examples", type=int, default=4)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
