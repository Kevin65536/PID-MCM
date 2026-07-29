#!/usr/bin/env python3
"""Run the exploratory R2-D C-J continuous observability experiment.

This entrypoint is deliberately narrow: it trains two modality-only continuous
students with one shared trajectory decoder. It contains no VQ, raw
reconstruction, cross-modal objective, coupling objective, or protected-test
loader. R1-D uses development-crossfit targets and is never promotion eligible.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.shared_driver_dataset import SharedDriverWindowDataset
from src.tokenizers.shared_driver_semantic_vq import SharedDriverContinuousModel


SCHEMA = "r2d_continuous_observability_run_v1"
TRAIN_SUBJECTS = tuple(
    f"eeg_fnirs_single_trial|subject_{index:02d}" for index in range(1, 19)
)
VALIDATION_SUBJECTS = tuple(
    f"eeg_fnirs_single_trial|subject_{index:02d}" for index in range(19, 24)
)
PROTECTED_SUBJECTS = tuple(
    f"eeg_fnirs_single_trial|subject_{index:02d}" for index in range(24, 30)
)
MODALITIES = ("eeg", "fnirs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/configs/physiology_semantic_tokenizer/"
        "r2d_continuous_observability.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trajectory-sidecar-root", type=Path)
    parser.add_argument("--raw-view-registry-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build real train/validation data, fit train-only statistics, and "
        "run one forward batch per split without constructing an optimizer.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the configured small number of optimizer steps.",
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = run("status", "--porcelain=v1")
    diff = run("diff", "--binary", "HEAD")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty_worktree": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def validate_config_contract(config: Mapping[str, Any]) -> None:
    """Fail closed on architecture, objective, split, and protected contracts."""

    required_values = {
        "architecture_generation": "shared_driver_semantic_vq_v1",
        "teacher_schema": "shared_driver_trajectory_sidecar_v1",
        "teacher_family": "adaptive_joint_full_trajectory",
        "teacher_parameter_scope": "development_crossfit",
        "tokenizer_inputs": "measured_modality_and_boundary_finite_mask_only",
        "token_temporal_scope": "bidirectional_full_window",
        "evaluator_temporal_mode": "semantic_only",
        "artifact_mask_policy": "annotation_only",
        "protected_open": False,
        "promotion_eligible": False,
    }
    for key, expected in required_values.items():
        if config.get(key) != expected:
            raise ValueError(
                f"R2-D config requires {key}={expected!r}, got {config.get(key)!r}"
            )

    objective = config.get("objective", {})
    objective_required = {
        "row": "C-J",
        "target": "target_shared_driver",
        "loss": "masked_point_mse",
        "vector_quantization": False,
        "raw_reconstruction": False,
        "cross_modal_loss": False,
        "coupling_loss": False,
    }
    for key, expected in objective_required.items():
        if objective.get(key) != expected:
            raise ValueError(
                f"R2-D objective requires {key}={expected!r}, "
                f"got {objective.get(key)!r}"
            )

    split = config.get("data", {}).get("split", {})
    train = tuple(str(value) for value in split.get("train_subject_keys", ()))
    validation = tuple(
        str(value) for value in split.get("validation_subject_keys", ())
    )
    protected = tuple(
        str(value) for value in split.get("protected_subject_keys", ())
    )
    if train != TRAIN_SUBJECTS:
        raise ValueError("R2-D train split must be exactly subjects 01–18")
    if validation != VALIDATION_SUBJECTS:
        raise ValueError("R2-D validation split must be exactly subjects 19–23")
    if protected != PROTECTED_SUBJECTS:
        raise ValueError("R2-D protected registry must be exactly subjects 24–29")
    if set(train + validation).intersection(protected):
        raise PermissionError("Protected subjects entered a development split")
    forbidden_test_keys = {"test_subject_keys", "test_subjects", "test_loader"}
    if forbidden_test_keys.intersection(split):
        raise PermissionError("R2-D config must not define or construct a test loader")

    model = config.get("model", {})
    expected_model = {
        "type": "shared_driver_continuous",
        "eeg_channels": 6,
        "fnirs_channels": 2,
        "eeg_patch_samples": 400,
        "fnirs_patch_samples": 20,
        "num_tokens": 10,
        "latent_dim": 64,
        "target_points": 20,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(
                f"R2-D model requires {key}={expected!r}, got {model.get(key)!r}"
            )


def masked_point_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    point_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over the exact registered pointwise intersection."""

    if prediction.shape != target.shape or point_mask.shape != target.shape:
        raise ValueError("prediction, target, and point_mask must have equal shape")
    mask = point_mask.to(device=prediction.device, dtype=torch.bool)
    count = mask.sum()
    if int(count.detach().cpu()) == 0:
        raise ValueError("masked_point_mse received no supported points")
    residual = prediction.float() - target.to(prediction.device, dtype=torch.float32)
    return residual.square().masked_select(mask).mean()


def make_point_masks(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    teacher = batch["teacher"]
    teacher_mask = teacher["teacher_mask"].bool()
    target_points = teacher["target_point_valid_mask"].bool()
    if target_points.shape[:2] != teacher_mask.shape:
        raise ValueError("Teacher patch and point masks are misaligned")
    output = {}
    for modality in MODALITIES:
        measurement = batch["token_valid_mask"][modality].bool()
        if measurement.shape != teacher_mask.shape:
            raise ValueError(f"{modality} measurement mask is misaligned")
        output[modality] = (
            measurement.unsqueeze(-1)
            & teacher_mask.unsqueeze(-1)
            & target_points
        )
    return output


def fit_phase_baseline(
    conditions: Sequence[str],
    targets: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Fit modality-matched condition × relative-time means on training rows."""

    target = np.asarray(targets, dtype=np.float64)
    if target.ndim != 3:
        raise ValueError("targets must have shape [sample, token, point]")
    if len(conditions) != target.shape[0]:
        raise ValueError("conditions and targets have different sample counts")
    result: dict[str, dict[str, np.ndarray]] = {modality: {} for modality in MODALITIES}
    for modality in MODALITIES:
        mask = np.asarray(masks[modality], dtype=bool)
        if mask.shape != target.shape:
            raise ValueError(f"{modality} mask shape differs from target")
        for condition in sorted(set(map(str, conditions))):
            rows = np.asarray([str(value) == condition for value in conditions])
            selected_mask = mask[rows]
            counts = selected_mask.sum(axis=0)
            if np.any(counts == 0):
                raise ValueError(
                    f"Phase baseline lacks {modality} train support for {condition!r}"
                )
            sums = np.where(selected_mask, target[rows], 0.0).sum(axis=0)
            result[modality][condition] = (sums / counts).astype(np.float32)
    return result


def phase_predictions(
    baseline: Mapping[str, Mapping[str, np.ndarray]],
    modality: str,
    conditions: Sequence[str],
) -> np.ndarray:
    values = []
    for condition in conditions:
        key = str(condition)
        if key not in baseline[modality]:
            raise KeyError(f"Validation condition {key!r} is absent from train baseline")
        values.append(np.asarray(baseline[modality][key], dtype=np.float32))
    return np.stack(values, axis=0)


def compute_subject_delta_r2(
    subjects: Sequence[str],
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    phase: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Aggregate SSE within subject before computing modality-specific ΔR²."""

    target = np.asarray(target, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for subject in sorted(set(map(str, subjects))):
        selected = np.asarray([str(value) == subject for value in subjects])
        modality_values: dict[str, float] = {}
        for modality in MODALITIES:
            mask = np.asarray(masks[modality], dtype=bool)[selected]
            observed = target[selected]
            model = np.asarray(predictions[modality], dtype=np.float64)[selected]
            null = np.asarray(phase[modality], dtype=np.float64)[selected]
            model_sse = float(np.square(observed - model)[mask].sum())
            phase_sse = float(np.square(observed - null)[mask].sum())
            if phase_sse <= 0.0:
                raise ValueError(f"Non-positive phase SSE for {subject}, {modality}")
            delta = 1.0 - model_sse / phase_sse
            modality_values[modality] = delta
            rows.append(
                {
                    "subject": subject,
                    "modality": modality,
                    "supported_points": int(mask.sum()),
                    "model_sse": model_sse,
                    "phase_sse": phase_sse,
                    "delta_r2": delta,
                    "positive": bool(delta > 0.0),
                }
            )
        equal = 0.5 * (modality_values["eeg"] + modality_values["fnirs"])
        rows.append(
            {
                "subject": subject,
                "modality": "equal_modalities",
                "supported_points": "",
                "model_sse": "",
                "phase_sse": "",
                "delta_r2": equal,
                "positive": bool(equal > 0.0),
            }
        )
    return rows


def subject_cluster_bootstrap(
    subject_rows: Sequence[Mapping[str, Any]],
    *,
    modality: str,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(
        [
            float(row["delta_r2"])
            for row in subject_rows
            if row["modality"] == modality
        ],
        dtype=np.float64,
    )
    if values.size == 0 or iterations <= 0:
        raise ValueError("Subject bootstrap requires subjects and positive iterations")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0,1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(iterations), values.size))
    draws = values[indices].mean(axis=1)
    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "subject_count": int(values.size),
        "subject_equal_mean_delta_r2": float(values.mean()),
        "confidence_level": float(confidence_level),
        "cluster_bootstrap_iterations": int(iterations),
        "cluster_bootstrap_ci": [float(lower), float(upper)],
        "positive_subject_count": int((values > 0.0).sum()),
    }


def _dataset_subjects(dataset: SharedDriverWindowDataset) -> set[str]:
    return {str(entry.subject_key) for entry in dataset.entries}


def _build_datasets(config: Mapping[str, Any]) -> dict[str, SharedDriverWindowDataset]:
    data = config["data"]
    split = data["split"]
    common = {
        "cache_root": str(_resolve_path(data["cache_root"])),
        "raw_view_registry_root": str(
            _resolve_path(data["raw_view_registry_root"])
        ),
        "trajectory_sidecar_root": str(
            _resolve_path(data["trajectory_sidecar_root"])
        ),
        "expected_teacher_scope": str(config["teacher_parameter_scope"]),
        "expected_target_family": str(config["teacher_family"]),
        "require_trajectory_target": True,
        "restrict_to_registered_views": True,
        "dataset_ids": tuple(data["dataset_ids"]),
        "task_namespaces": tuple(data["task_namespaces"]),
        "window_duration_s": float(data["window"]["duration_s"]),
        "window_offset_s": float(data["window"]["offset_s"]),
        "reject_unknown_labels": bool(data["reject_unknown_labels"]),
        "eeg_signal_branch": str(data["eeg_signal_branch"]),
    }
    # There is intentionally no test/protected branch in this function.
    datasets = {
        "train": SharedDriverWindowDataset(
            subject_keys=tuple(split["train_subject_keys"]), **common
        ),
        "validation": SharedDriverWindowDataset(
            subject_keys=tuple(split["validation_subject_keys"]), **common
        ),
    }
    expected = {
        "train": (set(TRAIN_SUBJECTS), int(split["expected_train_samples"])),
        "validation": (
            set(VALIDATION_SUBJECTS),
            int(split["expected_validation_samples"]),
        ),
    }
    for name, dataset in datasets.items():
        subjects, count = expected[name]
        if _dataset_subjects(dataset) != subjects:
            raise ValueError(f"{name} dataset subjects differ from the frozen split")
        if len(dataset) != count:
            raise ValueError(
                f"{name} target coverage is not exact: observed={len(dataset)}, "
                f"expected={count}"
            )
    return datasets


def _loader(
    dataset: SharedDriverWindowDataset,
    config: Mapping[str, Any],
    *,
    shuffle: bool,
    workers: int | None = None,
) -> DataLoader:
    data = config["data"]
    training = config["training"]
    count = int(data["num_workers"] if workers is None else workers)
    options: dict[str, Any] = {
        "batch_size": int(training["batch_size"]),
        "shuffle": bool(shuffle),
        "num_workers": count,
        "drop_last": False,
        "pin_memory": bool(data["dataloader"]["pin_memory"]),
    }
    if count:
        options.update(
            {
                "persistent_workers": bool(
                    data["dataloader"]["persistent_workers"]
                ),
                "prefetch_factor": int(data["dataloader"]["prefetch_factor"]),
            }
        )
    return DataLoader(dataset, **options)


def _collect_train_statistics(
    loader: DataLoader,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    channel_sum = {
        "eeg": np.zeros(6, dtype=np.float64),
        "fnirs": np.zeros(2, dtype=np.float64),
    }
    channel_square = {key: np.zeros_like(value) for key, value in channel_sum.items()}
    channel_count = {key: np.zeros_like(value) for key, value in channel_sum.items()}
    targets: list[np.ndarray] = []
    conditions: list[str] = []
    masks = {modality: [] for modality in MODALITIES}
    sample_ids: list[str] = []
    subjects: list[str] = []

    for batch in loader:
        point_masks = make_point_masks(batch)
        target = batch["teacher"]["target_shared_driver"].numpy()
        targets.append(target)
        conditions.extend(map(str, batch["condition"]))
        subjects.extend(map(str, batch["subject_key"]))
        sample_ids.extend(map(str, batch["sample_id"]))
        for modality, patch_samples in (("eeg", 400), ("fnirs", 20)):
            values = batch[modality].double().numpy()
            valid = (
                batch["token_valid_mask"][modality]
                .bool()
                .repeat_interleave(patch_samples, dim=1)
                .numpy()
            )
            expanded = valid[:, None, :]
            channel_sum[modality] += np.where(expanded, values, 0.0).sum(
                axis=(0, 2)
            )
            channel_square[modality] += np.where(
                expanded, np.square(values), 0.0
            ).sum(axis=(0, 2))
            channel_count[modality] += expanded.sum(axis=(0, 2))
            masks[modality].append(point_masks[modality].numpy())

    target_array = np.concatenate(targets, axis=0)
    mask_arrays = {
        modality: np.concatenate(parts, axis=0)
        for modality, parts in masks.items()
    }
    normalization: dict[str, dict[str, np.ndarray]] = {}
    for modality in MODALITIES:
        if np.any(channel_count[modality] == 0):
            raise ValueError(f"No train normalization support for {modality}")
        mean = channel_sum[modality] / channel_count[modality]
        variance = (
            channel_square[modality] / channel_count[modality] - np.square(mean)
        )
        scale = np.sqrt(np.maximum(variance, 1e-12))
        if np.any(~np.isfinite(scale)) or np.any(scale < 1e-6):
            raise ValueError(f"Degenerate train-only input scale for {modality}")
        normalization[modality] = {
            "mean": mean.astype(np.float32),
            "scale": scale.astype(np.float32),
            "count": channel_count[modality].astype(np.int64),
        }

    phase = fit_phase_baseline(conditions, target_array, mask_arrays)
    stats = {
        "normalization": normalization,
        "phase_baseline": phase,
        "fit_subjects": sorted(set(subjects)),
        "fit_sample_count": len(sample_ids),
        "fit_sample_order_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest(),
        "mask_intersection_sha256": _mask_hash(sample_ids, mask_arrays),
    }
    arrays = {
        "target": target_array,
        "conditions": np.asarray(conditions, dtype=np.str_),
        "subjects": np.asarray(subjects, dtype=np.str_),
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
        **{f"{modality}_mask": value for modality, value in mask_arrays.items()},
    }
    return stats, arrays


def _mask_hash(sample_ids: Sequence[str], masks: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, sample_ids)).encode("utf-8"))
    for modality in MODALITIES:
        digest.update(np.ascontiguousarray(masks[modality], dtype=np.uint8))
    return digest.hexdigest()


def _stats_json(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "r2d_train_only_statistics_v1",
        "normalization_policy": "train_only_channel_zscore",
        "phase_baseline_policy": "train_only_condition_by_relative_time_mean",
        "fit_subjects": list(stats["fit_subjects"]),
        "fit_sample_count": int(stats["fit_sample_count"]),
        "fit_sample_order_sha256": stats["fit_sample_order_sha256"],
        "mask_intersection_sha256": stats["mask_intersection_sha256"],
        "normalization": {
            modality: {
                key: np.asarray(value).tolist()
                for key, value in stats["normalization"][modality].items()
            }
            for modality in MODALITIES
        },
        "phase_baseline": {
            modality: {
                condition: np.asarray(values).tolist()
                for condition, values in stats["phase_baseline"][modality].items()
            }
            for modality in MODALITIES
        },
    }


def _normalize(
    signal: torch.Tensor,
    valid_mask: torch.Tensor,
    stats: Mapping[str, np.ndarray],
    patch_samples: int,
) -> torch.Tensor:
    mean = torch.as_tensor(stats["mean"], device=signal.device).view(1, -1, 1)
    scale = torch.as_tensor(stats["scale"], device=signal.device).view(1, -1, 1)
    point_mask = valid_mask.bool().repeat_interleave(patch_samples, dim=1)
    return ((signal - mean) / scale).masked_fill(~point_mask.unsqueeze(1), 0.0)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    output = dict(batch)
    for key in ("eeg", "fnirs"):
        output[key] = batch[key].to(device, non_blocking=True)
    output["token_valid_mask"] = {
        key: value.to(device, non_blocking=True)
        for key, value in batch["token_valid_mask"].items()
    }
    output["teacher"] = {
        key: value.to(device, non_blocking=True)
        for key, value in batch["teacher"].items()
    }
    return output


def _forward_loss(
    model: SharedDriverContinuousModel,
    batch: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    masks = make_point_masks(batch)
    eeg = _normalize(
        batch["eeg"],
        batch["token_valid_mask"]["eeg"],
        stats["normalization"]["eeg"],
        400,
    )
    fnirs = _normalize(
        batch["fnirs"],
        batch["token_valid_mask"]["fnirs"],
        stats["normalization"]["fnirs"],
        20,
    )
    output = model(
        eeg,
        fnirs,
        eeg_token_valid_mask=batch["token_valid_mask"]["eeg"],
        fnirs_token_valid_mask=batch["token_valid_mask"]["fnirs"],
    )
    target = batch["teacher"]["target_shared_driver"]
    losses = {
        modality: masked_point_mse(
            output[f"{modality}_decoded"], target, masks[modality]
        )
        for modality in MODALITIES
    }
    losses["equal_modalities"] = 0.5 * (losses["eeg"] + losses["fnirs"])
    return output, masks, losses


def _model_from_config(config: Mapping[str, Any]) -> SharedDriverContinuousModel:
    model = config["model"]
    return SharedDriverContinuousModel(
        eeg_channels=int(model["eeg_channels"]),
        fnirs_channels=int(model["fnirs_channels"]),
        eeg_patch_samples=int(model["eeg_patch_samples"]),
        fnirs_patch_samples=int(model["fnirs_patch_samples"]),
        num_tokens=int(model["num_tokens"]),
        latent_dim=int(model["latent_dim"]),
        target_points=int(model["target_points"]),
        encoder_depth=int(model["encoder_depth"]),
        encoder_num_heads=int(model["encoder_num_heads"]),
        encoder_feedforward_dim=int(model["encoder_feedforward_dim"]),
        decoder_hidden_dim=int(model["decoder_hidden_dim"]),
        dropout=float(model["dropout"]),
    )


def _resolve_device(value: str) -> torch.device:
    device = torch.device(str(value))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist")
    return device


def _loss_pass(
    model: SharedDriverContinuousModel,
    loader: DataLoader,
    stats: Mapping[str, Any],
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    sse = {modality: 0.0 for modality in MODALITIES}
    count = {modality: 0 for modality in MODALITIES}
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            with torch.amp.autocast(
                device_type=device.type, enabled=bool(amp and device.type == "cuda")
            ):
                output, masks, _ = _forward_loss(model, batch, stats)
            target = batch["teacher"]["target_shared_driver"]
            for modality in MODALITIES:
                residual = output[f"{modality}_decoded"].float() - target.float()
                sse[modality] += float(
                    residual.square().masked_select(masks[modality]).sum().cpu()
                )
                count[modality] += int(masks[modality].sum().cpu())
    means = {modality: sse[modality] / count[modality] for modality in MODALITIES}
    means["equal_modalities"] = 0.5 * (means["eeg"] + means["fnirs"])
    return means


def _collect_predictions(
    model: SharedDriverContinuousModel,
    loader: DataLoader,
    stats: Mapping[str, Any],
    device: torch.device,
    amp: bool,
    *,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "target": [],
        "eeg_prediction": [],
        "fnirs_prediction": [],
        "eeg_mask": [],
        "fnirs_mask": [],
        "eeg_phase": [],
        "fnirs_phase": [],
        "subject": [],
        "condition": [],
        "sample_id": [],
    }
    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            conditions = list(map(str, cpu_batch["condition"]))
            batch = _move_batch(cpu_batch, device)
            with torch.amp.autocast(
                device_type=device.type, enabled=bool(amp and device.type == "cuda")
            ):
                output, masks, _ = _forward_loss(model, batch, stats)
            collected["target"].append(
                batch["teacher"]["target_shared_driver"].float().cpu().numpy()
            )
            for modality in MODALITIES:
                collected[f"{modality}_prediction"].append(
                    output[f"{modality}_decoded"].float().cpu().numpy()
                )
                collected[f"{modality}_mask"].append(
                    masks[modality].cpu().numpy()
                )
                collected[f"{modality}_phase"].append(
                    phase_predictions(
                        stats["phase_baseline"], modality, conditions
                    )
                )
            collected["subject"].append(np.asarray(cpu_batch["subject_key"], dtype=np.str_))
            collected["condition"].append(np.asarray(conditions, dtype=np.str_))
            collected["sample_id"].append(np.asarray(cpu_batch["sample_id"], dtype=np.str_))
    return {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}


def _write_subject_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "subject",
        "modality",
        "supported_points",
        "model_sse",
        "phase_sse",
        "delta_r2",
        "positive",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_run_dir(config: Mapping[str, Any], requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group = str(config["experiment"]["run_group"])
        name = str(config["experiment"]["name"])
        path = REPO_ROOT / "experiments/runs" / group / f"{stamp}_{name}"
    else:
        path = requested if requested.is_absolute() else REPO_ROOT / requested
    active_root = (
        REPO_ROOT / "experiments/runs/physiology_semantic_tokenizer"
    ).resolve()
    resolved_parent = path.resolve().parent
    if active_root != resolved_parent and active_root not in resolved_parent.parents:
        raise ValueError("R2-D outputs must remain in the active physiology namespace")
    if path.exists():
        raise FileExistsError(f"Output directory exists; refusing overwrite: {path}")
    for child in ("checkpoints", "metrics", "predictions", "statistics"):
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def _input_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for label, key in (
        ("trajectory_sidecar", "trajectory_sidecar_root"),
        ("raw_view_registry", "raw_view_registry_root"),
    ):
        root = _resolve_path(config["data"][key])
        manifest_path = root / "manifest.json"
        arrays_path = root / "arrays.npz"
        if not manifest_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(f"Incomplete {label} artifact at {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protected_test_included") or manifest.get("protected_open"):
            raise PermissionError(f"{label} contains protected-test data")
        if manifest.get("promotion_eligible"):
            raise ValueError("R1-D inputs must remain promotion_eligible=false")
        result[label] = {
            "root": str(root),
            "manifest_sha256": _sha256(manifest_path),
            "arrays_sha256": _sha256(arrays_path),
            "schema": manifest.get("schema"),
            "sample_count": manifest.get("sample_count"),
        }
    return result


def _source_hashes(config_path: Path) -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "config": config_path.resolve(),
        "model": REPO_ROOT / "src/tokenizers/shared_driver_semantic_vq.py",
        "dataset": REPO_ROOT / "src/data/shared_driver_dataset.py",
        "target_reader": REPO_ROOT / "src/data/shared_driver_targets.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _serialize_checkpoint(
    path: Path,
    *,
    model: SharedDriverContinuousModel,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    optimizer_steps: int,
    validation_loss: float | None,
    stats_json: Mapping[str, Any],
    config_sha256: str,
    dry_run: bool,
) -> None:
    torch.save(
        {
            "schema": "r2d_continuous_checkpoint_v1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                None if optimizer is None else optimizer.state_dict()
            ),
            "epoch": int(epoch),
            "optimizer_steps": int(optimizer_steps),
            "validation_equal_modality_mse": validation_loss,
            "train_statistics": stats_json,
            "resolved_config_sha256": config_sha256,
            "dry_run_untrained": bool(dry_run),
            "promotion_eligible": False,
            "protected_open": False,
        },
        path,
    )


def _manifest_base(
    config: Mapping[str, Any],
    config_path: Path,
    resolved_config_path: Path,
    mode: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "architecture_generation": config["architecture_generation"],
        "semantic_objective": "C-J_masked_point_mse",
        "model_components": {
            "independent_modality_encoders": True,
            "shared_driver_decoder": True,
            "vector_quantization": False,
            "raw_reconstruction": False,
            "cross_modal_loss": False,
            "coupling_loss": False,
        },
        "teacher_schema": config["teacher_schema"],
        "teacher_family": config["teacher_family"],
        "teacher_parameter_scope": config["teacher_parameter_scope"],
        "validity_policy": config["validity_policy"],
        "artifact_mask_policy": config["artifact_mask_policy"],
        "token_temporal_scope": config["token_temporal_scope"],
        "evaluator_temporal_mode": config["evaluator_temporal_mode"],
        "promotion_eligible": False,
        "protected_open": False,
        "protected_loader_constructed": False,
        "fit_subject_keys": list(TRAIN_SUBJECTS),
        "validation_subject_keys": list(VALIDATION_SUBJECTS),
        "closed_protected_subject_keys": list(PROTECTED_SUBJECTS),
        "config_source": str(config_path.resolve()),
        "resolved_config_sha256": _sha256(resolved_config_path),
        "git": _git_state(),
        "source_files_sha256": _source_hashes(config_path),
        "input_artifacts": _input_provenance(config),
    }


def _amp_enabled(config: Mapping[str, Any], device: torch.device) -> bool:
    return bool(config["training"]["amp"] and device.type == "cuda")


def _run_dry(
    *,
    model: SharedDriverContinuousModel,
    datasets: Mapping[str, SharedDriverWindowDataset],
    config: Mapping[str, Any],
    stats: Mapping[str, Any],
    stats_json: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    config_sha256: str,
) -> dict[str, Any]:
    amp = _amp_enabled(config, device)
    contract: dict[str, Any] = {}
    for split in ("train", "validation"):
        loader = _loader(datasets[split], config, shuffle=False, workers=0)
        cpu_batch = next(iter(loader))
        batch = _move_batch(cpu_batch, device)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, enabled=amp
        ):
            output, masks, losses = _forward_loss(model, batch, stats)
        contract[split] = {
            "batch_size": int(batch["eeg"].shape[0]),
            "eeg_shape": list(batch["eeg"].shape),
            "fnirs_shape": list(batch["fnirs"].shape),
            "target_shape": list(
                batch["teacher"]["target_shared_driver"].shape
            ),
            "eeg_prediction_shape": list(output["eeg_decoded"].shape),
            "fnirs_prediction_shape": list(output["fnirs_decoded"].shape),
            "eeg_supported_points": int(masks["eeg"].sum().cpu()),
            "fnirs_supported_points": int(masks["fnirs"].sum().cpu()),
            "eeg_masked_point_mse": float(losses["eeg"].cpu()),
            "fnirs_masked_point_mse": float(losses["fnirs"].cpu()),
            "equal_modality_masked_point_mse": float(
                losses["equal_modalities"].cpu()
            ),
        }
    predictions = _collect_predictions(
        model,
        _loader(datasets["validation"], config, shuffle=False, workers=0),
        stats,
        device,
        amp,
        max_batches=1,
    )
    np.savez_compressed(
        run_dir / "predictions/dry_run_contract_batch.npz", **predictions
    )
    _serialize_checkpoint(
        run_dir / "checkpoints/contract_forward_untrained.pt",
        model=model,
        optimizer=None,
        epoch=-1,
        optimizer_steps=0,
        validation_loss=None,
        stats_json=stats_json,
        config_sha256=config_sha256,
        dry_run=True,
    )
    summary = {
        "schema": "r2d_continuous_observability_summary_v1",
        "mode": "dry_run",
        "contract_forward_passed": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "datasets": {
            "train_samples": len(datasets["train"]),
            "validation_samples": len(datasets["validation"]),
            "protected_loader_constructed": False,
        },
        "batch_contract": contract,
        "promotion_eligible": False,
        "interpretation": (
            "Real-data contract/shape validation only; no optimization and no "
            "scientific observability inference."
        ),
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _train(
    *,
    model: SharedDriverContinuousModel,
    datasets: Mapping[str, SharedDriverWindowDataset],
    config: Mapping[str, Any],
    stats: Mapping[str, Any],
    stats_json: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    config_sha256: str,
    smoke: bool,
) -> tuple[dict[str, Any], int]:
    training = config["training"]
    amp = _amp_enabled(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(map(float, training["betas"])),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_loader = _loader(datasets["train"], config, shuffle=True)
    validation_loader = _loader(datasets["validation"], config, shuffle=False)
    history: list[dict[str, Any]] = []
    best = float("inf")
    best_epoch = -1
    stale = 0
    steps = 0
    step_limit = int(training["smoke_optimizer_steps"]) if smoke else None

    for epoch in range(int(training["epochs"])):
        model.train()
        train_accumulator = {modality: 0.0 for modality in MODALITIES}
        batches = 0
        for cpu_batch in train_loader:
            batch = _move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                _, _, losses = _forward_loss(model, batch, stats)
                total = losses["equal_modalities"]
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["grad_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            for modality in MODALITIES:
                train_accumulator[modality] += float(losses[modality].detach().cpu())
            batches += 1
            steps += 1
            if step_limit is not None and steps >= step_limit:
                break

        validation = _loss_pass(
            model, validation_loader, stats, device, amp
        )
        train_means = {
            modality: train_accumulator[modality] / max(batches, 1)
            for modality in MODALITIES
        }
        row = {
            "epoch": epoch,
            "optimizer_steps": steps,
            "train_eeg_mse": train_means["eeg"],
            "train_fnirs_mse": train_means["fnirs"],
            "train_equal_modalities_mse": 0.5
            * (train_means["eeg"] + train_means["fnirs"]),
            "validation_eeg_mse": validation["eeg"],
            "validation_fnirs_mse": validation["fnirs"],
            "validation_equal_modalities_mse": validation["equal_modalities"],
        }
        history.append(row)
        value = validation["equal_modalities"]
        if value < best - float(training["early_stopping_min_delta"]):
            best = value
            best_epoch = epoch
            stale = 0
            _serialize_checkpoint(
                run_dir / "checkpoints/best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                optimizer_steps=steps,
                validation_loss=value,
                stats_json=stats_json,
                config_sha256=config_sha256,
                dry_run=False,
            )
        else:
            stale += 1
        if step_limit is not None and steps >= step_limit:
            break
        if stale >= int(training["early_stopping_patience"]):
            break

    with (run_dir / "metrics/loss_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(
        run_dir / "checkpoints/best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = _collect_predictions(
        model, validation_loader, stats, device, amp
    )
    np.savez_compressed(
        run_dir / "predictions/validation_predictions.npz", **predictions
    )
    masks = {
        modality: predictions[f"{modality}_mask"] for modality in MODALITIES
    }
    predicted = {
        modality: predictions[f"{modality}_prediction"]
        for modality in MODALITIES
    }
    phase = {
        modality: predictions[f"{modality}_phase"] for modality in MODALITIES
    }
    subject_rows = compute_subject_delta_r2(
        predictions["subject"].tolist(),
        predictions["target"],
        predicted,
        phase,
        masks,
    )
    _write_subject_metrics(
        run_dir / "metrics/per_subject_metrics.csv", subject_rows
    )
    statistics = config["statistics"]
    bootstrap = {
        modality: subject_cluster_bootstrap(
            subject_rows,
            modality=modality,
            iterations=(
                min(1000, int(statistics["bootstrap_iterations"]))
                if smoke
                else int(statistics["bootstrap_iterations"])
            ),
            confidence_level=float(statistics["confidence_level"]),
            seed=int(statistics["bootstrap_seed"]) + index,
        )
        for index, modality in enumerate((*MODALITIES, "equal_modalities"))
    }
    summary = {
        "schema": "r2d_continuous_observability_summary_v1",
        "mode": "smoke" if smoke else "formal_one_seed",
        "seed": int(training["seed"]),
        "best_epoch": int(best_epoch),
        "optimizer_steps": int(steps),
        "best_validation_equal_modality_mse": float(best),
        "subject_level_delta_r2": bootstrap,
        "exploratory_r2d_endpoints_evaluated": not smoke,
        "registered_r2p_gate_evaluated": False,
        "registered_gate_evaluated": False,
        "registered_gate_passed": False,
        "promotion_eligible": False,
        "teacher_scope": "development_crossfit",
        "interpretation": (
            "Exploratory R2-D development-crossfit feasibility only. "
            "R2-P requires a population-frozen teacher and three seeds."
        ),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "continuous_feasibility.json", summary)
    return summary, steps


def main() -> None:
    args = _parse_args()
    if args.dry_run and args.smoke:
        raise ValueError("--dry-run and --smoke are mutually exclusive")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.trajectory_sidecar_root is not None:
        config["data"]["trajectory_sidecar_root"] = str(
            args.trajectory_sidecar_root
        )
    if args.raw_view_registry_root is not None:
        config["data"]["raw_view_registry_root"] = str(
            args.raw_view_registry_root
        )
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.seed is not None:
        config["training"]["seed"] = int(args.seed)
    config["runtime"] = {
        "mode": "dry_run" if args.dry_run else ("smoke" if args.smoke else "formal"),
        "trajectory_sidecar_root_overridden": args.trajectory_sidecar_root is not None,
        "raw_view_registry_root_overridden": args.raw_view_registry_root is not None,
    }
    validate_config_contract(config)

    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _resolve_device(str(config["training"]["device"]))

    run_dir = _prepare_run_dir(config, args.output_dir)
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    manifest_path = run_dir / "manifest.json"
    manifest = _manifest_base(
        config,
        config_path,
        resolved_path,
        "dry_run" if args.dry_run else ("smoke" if args.smoke else "formal_one_seed"),
    )
    _write_json(manifest_path, manifest)

    try:
        datasets = _build_datasets(config)
        statistics_loader = _loader(
            datasets["train"], config, shuffle=False, workers=0
        )
        stats, _ = _collect_train_statistics(statistics_loader)
        if stats["fit_subjects"] != list(TRAIN_SUBJECTS):
            raise PermissionError("Train-only statistics used a non-training subject")
        stats_json = _stats_json(stats)
        _write_json(run_dir / "statistics/train_only_statistics.json", stats_json)
        model = _model_from_config(config).to(device)
        config_sha256 = _sha256(resolved_path)

        if args.dry_run:
            summary = _run_dry(
                model=model,
                datasets=datasets,
                config=config,
                stats=stats,
                stats_json=stats_json,
                device=device,
                run_dir=run_dir,
                config_sha256=config_sha256,
            )
            optimizer_steps = 0
        else:
            summary, optimizer_steps = _train(
                model=model,
                datasets=datasets,
                config=config,
                stats=stats,
                stats_json=stats_json,
                device=device,
                run_dir=run_dir,
                config_sha256=config_sha256,
                smoke=args.smoke,
            )
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "amp_enabled": _amp_enabled(config, device),
                "seed": seed,
                "dataset_counts": {
                    "train": len(datasets["train"]),
                    "validation": len(datasets["validation"]),
                    "protected": "not_loaded",
                },
                "train_statistics_sha256": _sha256(
                    run_dir / "statistics/train_only_statistics.json"
                ),
                "mask_intersection_hash": stats["mask_intersection_sha256"],
                "optimizer_steps": optimizer_steps,
                "summary_sha256": _sha256(run_dir / "summary.json"),
                "registered_gate_passed": bool(
                    summary.get("registered_gate_passed", False)
                ),
            }
        )
        _write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _write_json(manifest_path, manifest)
        raise

    print(run_dir)


if __name__ == "__main__":
    main()
