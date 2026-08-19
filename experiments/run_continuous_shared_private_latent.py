#!/usr/bin/env python3
"""Train the registered no-VQ continuous shared/private validation suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_ssm_reconstruction_reliability import (
    Job,
    _sample_to_trial,
    _sha256,
    _unit_is_full,
)
from src.data.unified_physiology import UnifiedPhysiologyWindowDataset
from src.tokenizers.continuous_shared_private import ContinuousSharedPrivateModel


SCHEMA = "continuous_shared_private_suite_v1"
CELL_SCHEMA = "continuous_shared_private_cell_v1"
ENDPOINTS = (
    "eeg_target_delta_r2",
    "fnirs_target_delta_r2",
    "fnirs_to_eeg_swap_delta_r2",
    "eeg_to_fnirs_swap_delta_r2",
)
TASK_ORDER = (
    "mental_arithmetic",
    "motor_imagery",
    "word_generation",
    "n_back",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{key: _jsonable(row.get(key, "")) for key in fields} for row in values]
        )


def _git_payload() -> dict[str, Any]:
    def call(*args: str) -> str:
        result = subprocess.run(
            args, cwd=REPO_ROOT, check=False, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "status_short": call("git", "status", "--short"),
    }


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def validate_config(config: Mapping[str, Any]) -> None:
    if config["source"].get("protected_open") is not False:
        raise PermissionError("continuous suite requires protected_open=false")
    if config["objective"].get("vector_quantization") is not False:
        raise ValueError("vector quantization must remain disabled")
    if config["objective"].get("raw_shared_gradient") != "stopped":
        raise ValueError("raw reconstruction must stop the shared gradient")
    if tuple(task["task_id"] for task in config["tasks"]) != TASK_ORDER:
        raise ValueError("task matrix differs from the registered four-task order")
    if tuple(config["statistics"]["primary_endpoints"]) != ENDPOINTS:
        raise ValueError("primary endpoint family differs from the registered 16 cells")
    model = config["model"]
    required = {
        "type": "continuous_shared_private",
        "eeg_channels": 6,
        "fnirs_channels": 2,
        "eeg_patch_samples": 400,
        "fnirs_patch_samples": 20,
        "num_tokens": 10,
        "shared_dim": 64,
        "eeg_private_dim": 64,
        "fnirs_private_dim": 32,
        "target_points": 20,
    }
    for key, expected in required.items():
        if model.get(key) != expected:
            raise ValueError(f"model contract requires {key}={expected!r}")
    forbidden = set()
    for task in config["tasks"]:
        forbidden.update(task["source_task_ids"])
    if forbidden != {
        "single_ma",
        "single_lmi",
        "single_rmi",
        "simultaneous_wg",
        "simultaneous_0back",
        "simultaneous_2back",
        "simultaneous_3back",
    }:
        raise ValueError("source task cells differ from the registered core matrix")


def _verify_source_run(run_dir: Path, expected_schema: str) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != expected_schema:
        raise ValueError("SSM source schema mismatch")
    if manifest.get("protected_open") is not False:
        raise PermissionError("SSM source opened a protected cohort")
    if manifest.get("completed_cell_count") != manifest.get("cell_count"):
        raise RuntimeError("SSM source run is incomplete")
    if manifest.get("failed_cell_count") != 0:
        raise RuntimeError("SSM source run contains failures")
    required = {"config.yaml", "window_metrics.csv", "trajectories.csv.gz"}
    artifacts = {entry["path"]: entry for entry in manifest["artifacts"]}
    if not required.issubset(artifacts):
        raise FileNotFoundError("SSM source run lacks required artifacts")
    for name in sorted(required):
        path = run_dir / name
        if _sha256(path) != artifacts[name]["sha256"]:
            raise RuntimeError(f"SSM source artifact hash mismatch: {name}")
    return manifest


@dataclass
class SourceBundle:
    eeg: np.ndarray
    fnirs: np.ndarray
    eeg_mask: np.ndarray
    fnirs_mask: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    sample_id: np.ndarray
    subject: np.ndarray
    role: np.ndarray
    condition: np.ndarray
    task_id: np.ndarray
    dataset_id: np.ndarray
    dependency_group: np.ndarray
    fold_index: np.ndarray
    eeg_channels: np.ndarray
    fnirs_channels: np.ndarray
    raw_sha256: np.ndarray

    def subset(self, indices: np.ndarray) -> "SourceBundle":
        return SourceBundle(
            **{name: np.asarray(getattr(self, name))[indices] for name in self.__dataclass_fields__}
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, schema=np.asarray("continuous_shared_private_source_v1"),
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        )


def _load_source_tables(
    run_dir: Path, source_task_ids: set[str], model: str, spatial_mode: str
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]]:
    meta_columns = [
        "task_id", "subject", "unit_id", "fold_index", "model", "spatial_mode",
        "selected_eeg_channels", "selected_fnirs_channels",
    ]
    metadata = pd.read_csv(run_dir / "window_metrics.csv", usecols=meta_columns)
    metadata = metadata[
        metadata.task_id.isin(source_task_ids)
        & (metadata.model == model)
        & (metadata.spatial_mode == spatial_mode)
    ]
    meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in metadata.itertuples(index=False):
        key = (str(row.task_id), str(row.subject), str(row.unit_id))
        if key in meta:
            raise RuntimeError(f"duplicate SSM window identity: {key}")
        meta[key] = {
            "fold_index": int(row.fold_index),
            "eeg_channels": tuple(str(row.selected_eeg_channels).split("|")),
            "fnirs_channels": tuple(str(row.selected_fnirs_channels).split("|")),
        }

    trajectory_columns = [
        "task_id", "subject", "unit_id", "model", "spatial_mode", "time_s",
        "shared_driver", "eeg_valid", "fnirs_valid",
    ]
    relevant: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        run_dir / "trajectories.csv.gz",
        usecols=trajectory_columns,
        chunksize=250_000,
        dtype={
            "task_id": str,
            "subject": str,
            "unit_id": str,
            "model": str,
            "spatial_mode": str,
            "time_s": float,
            "shared_driver": float,
            "eeg_valid": str,
            "fnirs_valid": str,
        },
    ):
        selected = chunk[
            chunk.task_id.isin(source_task_ids)
            & (chunk.model == model)
            & (chunk.spatial_mode == spatial_mode)
        ]
        if len(selected):
            relevant.append(selected)
    frame = pd.concat(relevant, ignore_index=True)
    targets: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, rows in frame.groupby(["task_id", "subject", "unit_id"], sort=False):
        ordered = rows.sort_values("time_s")
        if len(ordered) != 200:
            raise RuntimeError(f"SSM target does not contain 200 points: {key}")
        values = ordered.shared_driver.to_numpy(dtype=np.float32).reshape(10, 20)
        mask = (
            ordered.eeg_valid.str.lower().eq("true").to_numpy()
            & ordered.fnirs_valid.str.lower().eq("true").to_numpy()
            & np.isfinite(ordered.shared_driver.to_numpy(dtype=float))
        ).reshape(10, 20)
        targets[tuple(map(str, key))] = (values, mask)
    if set(meta) != set(targets):
        missing_meta = sorted(set(targets) - set(meta))[:5]
        missing_target = sorted(set(meta) - set(targets))[:5]
        raise RuntimeError(
            f"SSM metadata/target join mismatch: metadata={missing_meta}, target={missing_target}"
        )
    return meta, targets


def _raw_sha256(eeg: np.ndarray, fnirs: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(eeg).view(np.uint8))
    digest.update(np.ascontiguousarray(fnirs).view(np.uint8))
    return digest.hexdigest()


def _token_mask(point_mask: np.ndarray, patch_samples: int) -> np.ndarray:
    mask = np.asarray(point_mask, dtype=bool)
    if len(mask) % patch_samples:
        raise ValueError("point mask cannot be partitioned into registered patches")
    return mask.reshape(-1, patch_samples).all(axis=1)


def _subject_sets(source_config: Mapping[str, Any], dataset_id: str, smoke: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
    key = "single_trial" if dataset_id == "eeg_fnirs_single_trial" else "simultaneous"
    fit = list(map(str, source_config["data"]["core_fit_subjects"][key]))
    development = list(map(str, source_config["data"]["core_development_subjects"][key]))
    if smoke is not None:
        fit = fit[: int(smoke["fit_subjects_per_dataset"])]
        development = development[: int(smoke["validation_subjects_per_dataset"])]
    protected = set(map(str, source_config["data"]["protected_or_unused"][key]))
    if set(fit + development).intersection(protected):
        raise PermissionError("protected subject entered the continuous source build")
    return fit, development


def build_source_bundle(
    config: Mapping[str, Any], source_config: Mapping[str, Any], run_dir: Path, *, smoke: bool
) -> tuple[SourceBundle, list[dict[str, Any]]]:
    source_ids = {value for task in config["tasks"] for value in task["source_task_ids"]}
    meta, targets = _load_source_tables(
        run_dir,
        source_ids,
        str(config["source"]["trajectory_model"]),
        str(config["source"]["spatial_mode"]),
    )
    task_lookup = {str(task["task_id"]): task for task in source_config["data"]["tasks"]}
    group_lookup = {
        str(source_id): str(task["task_id"])
        for task in config["tasks"]
        for source_id in task["source_task_ids"]
    }
    smoke_cfg = config["smoke"] if smoke else None
    dataset_cache: dict[tuple[str, float, float], UnifiedPhysiologyWindowDataset] = {}
    values: dict[str, list[Any]] = defaultdict(list)
    index_rows: list[dict[str, Any]] = []

    for source_id in sorted(source_ids):
        task = task_lookup[source_id]
        dataset_id = str(task["dataset_id"])
        fit, development = _subject_sets(source_config, dataset_id, smoke_cfg)
        role = {subject: "fit" for subject in fit} | {
            subject: "development_validation" for subject in development
        }
        cache_key = (
            dataset_id,
            float(task["window_duration_s"]),
            float(task["window_offset_s"]),
        )
        if cache_key not in dataset_cache:
            dataset_cache[cache_key] = UnifiedPhysiologyWindowDataset(
                cache_root=source_config["data"]["cache_root"],
                dataset_ids=(dataset_id,),
                window_duration_s=float(task["window_duration_s"]),
                window_offset_s=float(task["window_offset_s"]),
                eeg_signal_branch=str(source_config["data"]["eeg_signal_branch"]),
                require_eeg_artifact_cache=dataset_id == "eeg_fnirs_single_trial",
            )
        dataset = dataset_cache[cache_key]
        admitted_by_subject: dict[str, int] = defaultdict(int)
        for dataset_index, ref in enumerate(dataset.windows):
            subject = str(ref.record.canonical_subject_id)
            if subject not in role:
                continue
            if ref.record.base_record_id != task["record_id"]:
                continue
            if str(ref.event.get("label")) != str(task["label"]):
                continue
            limit = None if smoke_cfg is None else int(smoke_cfg["samples_per_subject_condition"])
            if limit is not None and admitted_by_subject[subject] >= limit:
                continue
            sample = dataset[dataset_index]
            trial = _sample_to_trial(sample, task, unit_index=dataset_index)
            unit_id = f"{subject}|{trial.record_id}|event={trial.event_index}"
            identity = (source_id, subject, unit_id)
            if identity not in meta or identity not in targets:
                raise KeyError(f"raw/SSM identity join failed: {identity}")
            if not _unit_is_full(
                type("UnitLike", (), {"trial": trial, "unit_id": unit_id})()
            ):
                raise RuntimeError(f"registered core unit is incomplete: {identity}")
            eeg_names = meta[identity]["eeg_channels"]
            fnirs_names = meta[identity]["fnirs_channels"]
            if len(eeg_names) != 6 or len(fnirs_names) != 2:
                raise ValueError(f"registered raw-view shape mismatch: {identity}")
            eeg_indices = [trial.eeg_channel_names.index(name) for name in eeg_names]
            fnirs_indices = [trial.fnirs_channel_names.index(name) for name in fnirs_names]
            eeg = np.asarray(trial.eeg[:, eeg_indices].T, dtype=np.float32)
            fnirs = np.asarray(trial.fnirs[:, fnirs_indices].T, dtype=np.float32)
            eeg_mask = _token_mask(np.asarray(trial.eeg_valid_mask, dtype=bool), 400)
            fnirs_mask = _token_mask(np.asarray(trial.fnirs_valid_mask, dtype=bool), 20)
            target, target_mask = targets[identity]
            sample_id = f"{source_id}|{unit_id}|fold={meta[identity]['fold_index']}"
            raw_hash = _raw_sha256(eeg, fnirs)
            for name, value in (
                ("eeg", eeg), ("fnirs", fnirs), ("eeg_mask", eeg_mask),
                ("fnirs_mask", fnirs_mask), ("target", target),
                ("target_mask", target_mask), ("sample_id", sample_id),
                ("subject", subject), ("role", role[subject]),
                ("condition", source_id), ("task_id", group_lookup[source_id]),
                ("dataset_id", dataset_id), ("dependency_group", unit_id),
                ("fold_index", meta[identity]["fold_index"]),
                ("eeg_channels", "|".join(eeg_names)),
                ("fnirs_channels", "|".join(fnirs_names)),
                ("raw_sha256", raw_hash),
            ):
                values[name].append(value)
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "task_id": group_lookup[source_id],
                    "source_task_id": source_id,
                    "dataset_id": dataset_id,
                    "subject": subject,
                    "role": role[subject],
                    "dependency_group": unit_id,
                    "fold_index": meta[identity]["fold_index"],
                    "selected_eeg_channels": "|".join(eeg_names),
                    "selected_fnirs_channels": "|".join(fnirs_names),
                    "raw_sha256": raw_hash,
                    "target_valid_points": int(target_mask.sum()),
                }
            )
            admitted_by_subject[subject] += 1

    order = np.argsort(np.asarray(values["sample_id"], dtype=str))
    bundle = SourceBundle(
        eeg=np.stack(values["eeg"])[order],
        fnirs=np.stack(values["fnirs"])[order],
        eeg_mask=np.stack(values["eeg_mask"])[order],
        fnirs_mask=np.stack(values["fnirs_mask"])[order],
        target=np.stack(values["target"])[order],
        target_mask=np.stack(values["target_mask"])[order],
        sample_id=np.asarray(values["sample_id"], dtype=str)[order],
        subject=np.asarray(values["subject"], dtype=str)[order],
        role=np.asarray(values["role"], dtype=str)[order],
        condition=np.asarray(values["condition"], dtype=str)[order],
        task_id=np.asarray(values["task_id"], dtype=str)[order],
        dataset_id=np.asarray(values["dataset_id"], dtype=str)[order],
        dependency_group=np.asarray(values["dependency_group"], dtype=str)[order],
        fold_index=np.asarray(values["fold_index"], dtype=np.int16)[order],
        eeg_channels=np.asarray(values["eeg_channels"], dtype=str)[order],
        fnirs_channels=np.asarray(values["fnirs_channels"], dtype=str)[order],
        raw_sha256=np.asarray(values["raw_sha256"], dtype=str)[order],
    )
    if len(set(bundle.sample_id.tolist())) != len(bundle.sample_id):
        raise RuntimeError("continuous source contains duplicate canonical identities")
    expected = 56 if smoke else len(meta)
    if len(bundle.sample_id) != expected:
        raise RuntimeError(f"continuous source count mismatch: {len(bundle.sample_id)} != {expected}")
    return bundle, sorted(index_rows, key=lambda row: row["sample_id"])


def _masked_channel_stats(signal: np.ndarray, token_mask: np.ndarray, patch: int) -> tuple[np.ndarray, np.ndarray]:
    point = np.repeat(token_mask, patch, axis=1)[:, None, :]
    count = point.sum(axis=(0, 2)).astype(np.float64)
    total = np.where(point, signal, 0.0).sum(axis=(0, 2), dtype=np.float64)
    square = np.where(point, np.square(signal), 0.0).sum(axis=(0, 2), dtype=np.float64)
    mean = total / count
    scale = np.sqrt(np.maximum(square / count - np.square(mean), 1e-12))
    if np.any(scale < 1e-6) or np.any(~np.isfinite(scale)):
        raise ValueError("degenerate train-only input normalization")
    return mean.astype(np.float32), scale.astype(np.float32)


def _masked_mean(values: np.ndarray, mask: np.ndarray, axis: int = 0) -> np.ndarray:
    count = mask.sum(axis=axis)
    if np.any(count == 0):
        raise ValueError("train-only phase baseline lacks support")
    return (np.where(mask, values, 0.0).sum(axis=axis) / count).astype(np.float32)


def fit_task_statistics(bundle: SourceBundle) -> dict[str, Any]:
    fit = bundle.role == "fit"
    if not fit.any():
        raise ValueError("task has no fit subjects")
    eeg_mean, eeg_scale = _masked_channel_stats(bundle.eeg[fit], bundle.eeg_mask[fit], 400)
    fnirs_mean, fnirs_scale = _masked_channel_stats(bundle.fnirs[fit], bundle.fnirs_mask[fit], 20)
    target_values = bundle.target[fit][bundle.target_mask[fit]]
    target_mean = float(np.mean(target_values))
    target_scale = float(np.std(target_values))
    if target_scale < 1e-6 or not np.isfinite(target_scale):
        raise ValueError("degenerate train-only SSM target scale")
    phase: dict[str, dict[str, np.ndarray]] = {"target": {}, "eeg": {}, "fnirs": {}}
    for condition in sorted(set(bundle.condition[fit].tolist())):
        rows = fit & (bundle.condition == condition)
        phase["target"][condition] = _masked_mean(
            bundle.target[rows], bundle.target_mask[rows]
        )
        for modality, patch in (("eeg", 400), ("fnirs", 20)):
            signal = getattr(bundle, modality)[rows]
            token = getattr(bundle, f"{modality}_mask")[rows]
            point = np.repeat(token, patch, axis=1)[:, None, :]
            phase[modality][condition] = _masked_mean(
                signal, np.broadcast_to(point, signal.shape)
            )
    return {
        "normalization": {
            "eeg": {"mean": eeg_mean, "scale": eeg_scale},
            "fnirs": {"mean": fnirs_mean, "scale": fnirs_scale},
            "target": {"mean": target_mean, "scale": target_scale},
        },
        "phase": phase,
        "fit_subjects": sorted(set(bundle.subject[fit].tolist())),
        "fit_sample_count": int(fit.sum()),
    }


def normalize_task(bundle: SourceBundle, stats: Mapping[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for modality in ("eeg", "fnirs"):
        signal = getattr(bundle, modality)
        mean = np.asarray(stats["normalization"][modality]["mean"])[None, :, None]
        scale = np.asarray(stats["normalization"][modality]["scale"])[None, :, None]
        patch = 400 if modality == "eeg" else 20
        point = np.repeat(getattr(bundle, f"{modality}_mask"), patch, axis=1)[:, None, :]
        output[modality] = np.where(point, (signal - mean) / scale, 0.0).astype(np.float32)
    target_stats = stats["normalization"]["target"]
    output["target"] = np.where(
        bundle.target_mask,
        (bundle.target - float(target_stats["mean"])) / float(target_stats["scale"]),
        0.0,
    ).astype(np.float32)
    output["eeg_mask"] = bundle.eeg_mask
    output["fnirs_mask"] = bundle.fnirs_mask
    output["target_mask"] = bundle.target_mask
    return output


class _ArrayDataset(Dataset):
    def __init__(self, arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        return {
            "index": torch.tensor(row, dtype=torch.long),
            "eeg": torch.from_numpy(self.arrays["eeg"][row]),
            "fnirs": torch.from_numpy(self.arrays["fnirs"][row]),
            "eeg_mask": torch.from_numpy(self.arrays["eeg_mask"][row]),
            "fnirs_mask": torch.from_numpy(self.arrays["fnirs_mask"][row]),
            "target": torch.from_numpy(self.arrays["target"][row]),
            "target_mask": torch.from_numpy(self.arrays["target_mask"][row]),
        }


def _model(config: Mapping[str, Any]) -> ContinuousSharedPrivateModel:
    kwargs = dict(config["model"])
    kwargs.pop("type")
    return ContinuousSharedPrivateModel(**kwargs)


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    admitted = mask.to(device=prediction.device, dtype=torch.bool)
    if prediction.shape != target.shape or prediction.shape != admitted.shape:
        raise ValueError("masked MSE tensors differ in shape")
    if int(admitted.sum().detach().cpu()) == 0:
        raise ValueError("masked MSE has no supported points")
    return (prediction.float() - target.float()).square().masked_select(admitted).mean()


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _losses(model: ContinuousSharedPrivateModel, batch: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    output = model(
        batch["eeg"], batch["fnirs"], batch["eeg_mask"], batch["fnirs_mask"]
    )
    target_mask = batch["target_mask"].bool()
    losses = {
        "shared_eeg": _masked_mse(output["eeg_driver"], batch["target"], target_mask & batch["eeg_mask"].unsqueeze(-1)),
        "shared_fnirs": _masked_mse(output["fnirs_driver"], batch["target"], target_mask & batch["fnirs_mask"].unsqueeze(-1)),
    }
    for modality, patch in (("eeg", 400), ("fnirs", 20)):
        point = batch[f"{modality}_mask"].repeat_interleave(patch, dim=1).unsqueeze(1)
        point = point.expand_as(batch[modality])
        losses[f"raw_{modality}"] = _masked_mse(output[f"{modality}_raw"], batch[modality], point)
    losses["shared_equal"] = 0.5 * (losses["shared_eeg"] + losses["shared_fnirs"])
    losses["raw_equal"] = 0.5 * (losses["raw_eeg"] + losses["raw_fnirs"])
    return output, losses


def _validation_loss(model: ContinuousSharedPrivateModel, loader: DataLoader, device: torch.device, amp: bool) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for cpu in loader:
            batch = _move(cpu, device)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                _, losses = _losses(model, batch)
            total += float(losses["shared_equal"].cpu())
            batches += 1
    return total / batches


def _collect(model: ContinuousSharedPrivateModel, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for cpu in loader:
            batch = _move(cpu, device)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                output, _ = _losses(model, batch)
            collected["index"].append(batch["index"].cpu().numpy())
            for key in (
                "eeg_shared", "fnirs_shared", "eeg_private", "fnirs_private",
                "eeg_driver", "fnirs_driver", "eeg_raw", "fnirs_raw",
            ):
                collected[key].append(output[key].float().cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in collected.items()}


def make_derangement(
    subjects: Sequence[str], conditions: Sequence[str], sample_ids: Sequence[str], seed: int
) -> np.ndarray:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (subject, condition) in enumerate(zip(subjects, conditions, strict=True)):
        groups[(str(subject), str(condition))].append(index)
    donor = np.full(len(sample_ids), -1, dtype=int)
    for key, indices in groups.items():
        if len(indices) < 2:
            raise ValueError(f"derangement group has fewer than two samples: {key}")
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}|{sample_ids[index]}".encode("utf-8")
            ).hexdigest(),
        )
        for position, target in enumerate(ranked):
            donor[target] = ranked[(position + 1) % len(ranked)]
    if np.any(donor < 0) or np.any(donor == np.arange(len(donor))):
        raise RuntimeError("non-identity derangement construction failed")
    return donor


def _decode_modes(
    model: ContinuousSharedPrivateModel,
    collected: Mapping[str, np.ndarray],
    donor: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    count = len(donor)
    model.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            target_indices = np.arange(start, stop)
            donor_indices = donor[target_indices]
            for modality, source in (("eeg", "fnirs"), ("fnirs", "eeg")):
                private = torch.from_numpy(collected[f"{modality}_private"][target_indices]).to(device)
                own_shared = torch.from_numpy(collected[f"{modality}_shared"][target_indices]).to(device)
                matched = torch.from_numpy(collected[f"{source}_shared"][target_indices]).to(device)
                deranged = torch.from_numpy(collected[f"{source}_shared"][donor_indices]).to(device)
                zeros_shared = torch.zeros_like(matched)
                zeros_private = torch.zeros_like(private)
                modes = {
                    "self": model.decode_raw(modality, own_shared, private),
                    "matched": model.decode_raw(modality, matched, private),
                    "deranged": model.decode_raw(modality, deranged, private),
                    "shared_only": model.decode_raw(modality, matched, zeros_private),
                    "private_only": model.decode_raw(modality, zeros_shared, private),
                }
                for mode, tensor in modes.items():
                    output[f"{modality}_{mode}"].append(tensor.float().cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in output.items()}


def _phase_arrays(bundle: SourceBundle, stats: Mapping[str, Any], normalized: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    target_mean = float(stats["normalization"]["target"]["mean"])
    target_scale = float(stats["normalization"]["target"]["scale"])
    for condition in bundle.condition:
        key = str(condition)
        result["target"].append(
            (np.asarray(stats["phase"]["target"][key]) - target_mean) / target_scale
        )
        for modality in ("eeg", "fnirs"):
            mean = np.asarray(stats["normalization"][modality]["mean"])[:, None]
            scale = np.asarray(stats["normalization"][modality]["scale"])[:, None]
            result[modality].append(
                (np.asarray(stats["phase"][modality][key]) - mean) / scale
            )
    return {key: np.stack(values).astype(np.float32) for key, values in result.items()}


def _broadcast_raw_mask(token_mask: np.ndarray, channels: int, patch: int) -> np.ndarray:
    points = np.repeat(token_mask, patch, axis=1)[:, None, :]
    return np.broadcast_to(points, (len(token_mask), channels, points.shape[-1]))


def _sse(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    return float(np.square(observed.astype(np.float64) - predicted.astype(np.float64))[mask].sum())


def compute_subject_metrics(
    task_id: str,
    seed: int,
    bundle: SourceBundle,
    normalized: Mapping[str, np.ndarray],
    collected: Mapping[str, np.ndarray],
    modes: Mapping[str, np.ndarray],
    phase: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for subject in sorted(set(bundle.subject.tolist())):
        selected = bundle.subject == subject
        for modality in ("eeg", "fnirs"):
            target_mask = bundle.target_mask[selected] & getattr(bundle, f"{modality}_mask")[selected, :, None]
            model_sse = _sse(
                normalized["target"][selected],
                collected[f"{modality}_driver"][selected],
                target_mask,
            )
            phase_sse = _sse(
                normalized["target"][selected], phase["target"][selected], target_mask
            )
            endpoint_rows.append(
                {
                    "task_id": task_id,
                    "dataset_id": str(bundle.dataset_id[0]),
                    "seed": seed,
                    "subject": subject,
                    "endpoint": f"{modality}_target_delta_r2",
                    "value": 1.0 - model_sse / phase_sse,
                    "numerator_sse": model_sse,
                    "denominator_sse": phase_sse,
                    "supported_points": int(target_mask.sum()),
                }
            )
            raw_mask = _broadcast_raw_mask(
                getattr(bundle, f"{modality}_mask")[selected],
                6 if modality == "eeg" else 2,
                400 if modality == "eeg" else 20,
            )
            observed = normalized[modality][selected]
            matched_sse = _sse(observed, modes[f"{modality}_matched"][selected], raw_mask)
            deranged_sse = _sse(observed, modes[f"{modality}_deranged"][selected], raw_mask)
            source = "fnirs" if modality == "eeg" else "eeg"
            endpoint_rows.append(
                {
                    "task_id": task_id,
                    "dataset_id": str(bundle.dataset_id[0]),
                    "seed": seed,
                    "subject": subject,
                    "endpoint": f"{source}_to_{modality}_swap_delta_r2",
                    "value": 1.0 - matched_sse / deranged_sse,
                    "numerator_sse": matched_sse,
                    "denominator_sse": deranged_sse,
                    "supported_points": int(raw_mask.sum()),
                }
            )
            baseline_sse = _sse(observed, phase[modality][selected], raw_mask)
            for mode in ("self", "matched", "deranged", "shared_only", "private_only"):
                value_sse = _sse(observed, modes[f"{modality}_{mode}"][selected], raw_mask)
                raw_rows.append(
                    {
                        "task_id": task_id,
                        "dataset_id": str(bundle.dataset_id[0]),
                        "seed": seed,
                        "subject": subject,
                        "modality": modality,
                        "mode": mode,
                        "r2_vs_train_phase": 1.0 - value_sse / baseline_sse,
                        "mse": value_sse / int(raw_mask.sum()),
                        "supported_points": int(raw_mask.sum()),
                    }
                )
    return endpoint_rows, raw_rows


def _ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, components: int, alpha: float) -> np.ndarray:
    mean_x = train_x.mean(axis=0, keepdims=True)
    centered = train_x - mean_x
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(int(components), vt.shape[0], vt.shape[1])
    basis = vt[:count].T
    x = centered @ basis
    test = (test_x - mean_x) @ basis
    mean_y = train_y.mean(axis=0, keepdims=True)
    y = train_y - mean_y
    weights = np.linalg.solve(x.T @ x + float(alpha) * np.eye(count), x.T @ y)
    return (test @ weights + mean_y).astype(np.float32)


def ridge_probe_metrics(
    task_id: str,
    seed: int,
    train_bundle: SourceBundle,
    validation_bundle: SourceBundle,
    train_collected: Mapping[str, np.ndarray],
    validation_collected: Mapping[str, np.ndarray],
    train_normalized: Mapping[str, np.ndarray],
    validation_normalized: Mapping[str, np.ndarray],
    validation_phase: np.ndarray,
    *,
    components: int,
    alpha: float,
) -> list[dict[str, Any]]:
    train_target = train_normalized["target"].reshape(-1, 20)
    train_mask = train_bundle.target_mask.reshape(-1, 20).all(axis=1)
    validation_target = validation_normalized["target"].reshape(-1, 20)
    validation_mask = validation_bundle.target_mask.reshape(-1, 20).all(axis=1)
    rows: list[dict[str, Any]] = []
    for modality in ("eeg", "fnirs"):
        for latent_class in ("shared", "private"):
            key = f"{modality}_{latent_class}"
            train_x = train_collected[key].reshape(-1, train_collected[key].shape[-1])
            validation_x = validation_collected[key].reshape(-1, validation_collected[key].shape[-1])
            prediction = _ridge_fit_predict(
                train_x[train_mask], train_target[train_mask], validation_x, components, alpha
            ).reshape(validation_normalized["target"].shape)
            for subject in sorted(set(validation_bundle.subject.tolist())):
                selected = validation_bundle.subject == subject
                mask = validation_bundle.target_mask[selected]
                model_sse = _sse(validation_normalized["target"][selected], prediction[selected], mask)
                phase_sse = _sse(validation_normalized["target"][selected], validation_phase[selected], mask)
                rows.append(
                    {
                        "task_id": task_id,
                        "dataset_id": str(validation_bundle.dataset_id[0]),
                        "seed": seed,
                        "subject": subject,
                        "modality": modality,
                        "latent_class": latent_class,
                        "components": int(components),
                        "alpha": float(alpha),
                        "target_delta_r2": 1.0 - model_sse / phase_sse,
                    }
                )
    return rows


def _effective_rank(values: np.ndarray) -> tuple[float, float]:
    matrix = values.reshape(-1, values.shape[-1]).astype(np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    eigen = np.linalg.svd(matrix, compute_uv=False, full_matrices=False) ** 2
    total = eigen.sum()
    if total <= 0:
        return 0.0, 0.0
    probability = eigen / total
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum())), float(np.var(matrix, axis=0).mean())


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1, x.shape[-1]).astype(np.float64)
    y = y.reshape(-1, y.shape[-1]).astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    numerator = np.square(x.T @ y).sum()
    denominator = np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum())
    return float(numerator / denominator) if denominator > 0 else float("nan")


def latent_diagnostics(
    task_id: str,
    seed: int,
    bundle: SourceBundle,
    collected: Mapping[str, np.ndarray],
    donor: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("eeg_shared", "fnirs_shared", "eeg_private", "fnirs_private"):
        rank, variance = _effective_rank(collected[key])
        rows.append(
            {
                "task_id": task_id,
                "dataset_id": str(bundle.dataset_id[0]),
                "seed": seed,
                "diagnostic": "representation",
                "representation": key,
                "effective_rank": rank,
                "mean_dimension_variance": variance,
                "linear_cka": "",
            }
        )
    rows.extend(
        [
            {
                "task_id": task_id,
                "dataset_id": str(bundle.dataset_id[0]),
                "seed": seed,
                "diagnostic": "shared_cross_modal",
                "representation": "matched",
                "effective_rank": "",
                "mean_dimension_variance": "",
                "linear_cka": _linear_cka(collected["eeg_shared"], collected["fnirs_shared"]),
            },
            {
                "task_id": task_id,
                "dataset_id": str(bundle.dataset_id[0]),
                "seed": seed,
                "diagnostic": "shared_cross_modal",
                "representation": "deranged",
                "effective_rank": "",
                "mean_dimension_variance": "",
                "linear_cka": _linear_cka(collected["eeg_shared"], collected["fnirs_shared"][donor]),
            },
        ]
    )
    return rows


def _serialize_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    return _jsonable(stats)


def train_cell(
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    seed: int,
    bundle: SourceBundle,
    cell_dir: Path,
    device: torch.device,
    *,
    smoke: bool,
) -> dict[str, list[dict[str, Any]]]:
    _set_seed(seed)
    task_indices = np.flatnonzero(bundle.task_id == task["task_id"])
    task_bundle = bundle.subset(task_indices)
    stats = fit_task_statistics(task_bundle)
    normalized = normalize_task(task_bundle, stats)
    train_indices = np.flatnonzero(task_bundle.role == "fit")
    validation_indices = np.flatnonzero(task_bundle.role == "development_validation")
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError("task cell lacks a fit or development split")
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _ArrayDataset(normalized, train_indices),
        batch_size=int(training["batch_size"]), shuffle=True,
        num_workers=int(training["num_workers"]), generator=generator,
    )
    train_eval_loader = DataLoader(
        _ArrayDataset(normalized, train_indices), batch_size=int(training["batch_size"]), shuffle=False
    )
    validation_loader = DataLoader(
        _ArrayDataset(normalized, validation_indices), batch_size=int(training["batch_size"]), shuffle=False
    )
    model = _model(config).to(device)
    amp = bool(training["amp"] and device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]), betas=tuple(map(float, training["betas"])),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = float("inf")
    best_epoch = -1
    stale = 0
    steps = 0
    history: list[dict[str, Any]] = []
    step_limit = int(config["smoke"]["optimizer_steps"]) if smoke else None
    for epoch in range(int(training["epochs"])):
        model.train()
        accumulator = defaultdict(float)
        batches = 0
        for cpu in train_loader:
            batch = _move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                _, losses = _losses(model, batch)
                total = losses["shared_equal"] + float(config["objective"]["raw_loss_weight"]) * losses["raw_equal"]
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            for key, value in losses.items():
                accumulator[key] += float(value.detach().cpu())
            batches += 1
            steps += 1
            if step_limit is not None and steps >= step_limit:
                break
        validation_loss = _validation_loss(model, validation_loader, device, amp)
        history.append(
            {
                "task_id": task["task_id"], "seed": seed, "epoch": epoch,
                "optimizer_steps": steps,
                **{f"train_{key}": value / batches for key, value in accumulator.items()},
                "validation_shared_equal": validation_loss,
            }
        )
        if validation_loss < best - float(training["early_stopping_min_delta"]):
            best = validation_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "schema": "continuous_shared_private_checkpoint_v1",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "optimizer_steps": steps,
                    "validation_shared_equal": validation_loss,
                    "train_statistics": _serialize_stats(stats),
                    "seed": seed,
                    "task_id": task["task_id"],
                    "vector_quantization": False,
                    "protected_open": False,
                },
                cell_dir / "best.pt",
            )
        else:
            stale += 1
        if (step_limit is not None and steps >= step_limit) or stale >= int(training["early_stopping_patience"]):
            break
    checkpoint = torch.load(cell_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _write_csv(cell_dir / "loss_curves.csv", history)
    _write_json(cell_dir / "train_statistics.json", _serialize_stats(stats))

    train_collected_full = _collect(model, train_eval_loader, device, amp)
    validation_collected_full = _collect(model, validation_loader, device, amp)
    train_order = np.argsort(train_collected_full["index"])
    validation_order = np.argsort(validation_collected_full["index"])
    train_collected = {key: value[train_order] for key, value in train_collected_full.items() if key != "index"}
    validation_collected = {key: value[validation_order] for key, value in validation_collected_full.items() if key != "index"}
    train_bundle = task_bundle.subset(train_indices)
    validation_bundle = task_bundle.subset(validation_indices)
    train_normalized = {key: value[train_indices] for key, value in normalized.items()}
    validation_normalized = {key: value[validation_indices] for key, value in normalized.items()}
    donor = make_derangement(
        validation_bundle.subject,
        validation_bundle.condition,
        validation_bundle.sample_id,
        seed,
    )
    modes = _decode_modes(
        model, validation_collected, donor, device, int(training["batch_size"])
    )
    phase = _phase_arrays(validation_bundle, stats, validation_normalized)
    endpoint_rows, raw_rows = compute_subject_metrics(
        str(task["task_id"]), seed, validation_bundle, validation_normalized,
        validation_collected, modes, phase,
    )
    probe_rows = ridge_probe_metrics(
        str(task["task_id"]), seed, train_bundle, validation_bundle,
        train_collected, validation_collected, train_normalized,
        validation_normalized, phase["target"],
        components=int(config["statistics"]["ridge_probe_components"]),
        alpha=float(config["statistics"]["ridge_probe_alpha"]),
    )
    diagnostic_rows = latent_diagnostics(
        str(task["task_id"]), seed, validation_bundle, validation_collected, donor
    )
    derangement_rows = [
        {
            "task_id": task["task_id"], "seed": seed,
            "subject": validation_bundle.subject[index],
            "condition": validation_bundle.condition[index],
            "target_sample_id": validation_bundle.sample_id[index],
            "donor_sample_id": validation_bundle.sample_id[int(donor[index])],
            "identity_preserved": False,
            "same_subject": bool(validation_bundle.subject[index] == validation_bundle.subject[donor[index]]),
            "same_condition": bool(validation_bundle.condition[index] == validation_bundle.condition[donor[index]]),
            "same_token_time": True,
        }
        for index in range(len(donor))
    ]
    _write_csv(cell_dir / "subject_endpoints.csv", endpoint_rows)
    _write_csv(cell_dir / "raw_ablation_metrics.csv", raw_rows)
    _write_csv(cell_dir / "ridge_probe_metrics.csv", probe_rows)
    _write_csv(cell_dir / "latent_diagnostics.csv", diagnostic_rows)
    _write_csv(cell_dir / "derangement_registry.csv", derangement_rows)
    np.savez_compressed(
        cell_dir / "validation_predictions.npz",
        schema=np.asarray("continuous_shared_private_predictions_v1"),
        sample_id=validation_bundle.sample_id,
        subject=validation_bundle.subject,
        condition=validation_bundle.condition,
        donor_index=donor,
        target=validation_normalized["target"],
        target_mask=validation_bundle.target_mask,
        eeg_observed=validation_normalized["eeg"],
        fnirs_observed=validation_normalized["fnirs"],
        eeg_mask=validation_bundle.eeg_mask,
        fnirs_mask=validation_bundle.fnirs_mask,
        **validation_collected,
        **modes,
    )
    cell_manifest = {
        "schema": CELL_SCHEMA,
        "status": "completed",
        "task_id": task["task_id"],
        "dataset_id": task["dataset_id"],
        "source_task_ids": list(task["source_task_ids"]),
        "seed": seed,
        "fit_sample_count": len(train_bundle.sample_id),
        "validation_sample_count": len(validation_bundle.sample_id),
        "fit_subject_count": len(set(train_bundle.subject.tolist())),
        "validation_subject_count": len(set(validation_bundle.subject.tolist())),
        "best_epoch": best_epoch,
        "optimizer_steps": steps,
        "best_validation_shared_equal": best,
        "vector_quantization": False,
        "protected_open": False,
        "artifacts": [],
    }
    for path in sorted(cell_dir.iterdir()):
        if path.name == "manifest.json":
            continue
        cell_manifest["artifacts"].append(
            {"path": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    _write_json(cell_dir / "manifest.json", cell_manifest)
    return {
        "endpoints": endpoint_rows,
        "raw": raw_rows,
        "probes": probe_rows,
        "diagnostics": diagnostic_rows,
        "derangements": derangement_rows,
        "history": history,
    }


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "manifest.json" and path.parent == root:
            continue
        output.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return output


def run(args: argparse.Namespace) -> Path:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    source_run = _resolve(config["source"]["ssm_run"])
    source_manifest = _verify_source_run(source_run, str(config["source"]["ssm_schema"]))
    source_config = yaml.safe_load((source_run / "config.yaml").read_text(encoding="utf-8"))
    target = (
        _resolve(args.output_dir)
        if args.output_dir is not None
        else _resolve(config["output"]["root"]) / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config["experiment"]["name"]
        )
    )
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        shutil.copy2(config_path, staging / "config.yaml")
        bundle, source_rows = build_source_bundle(
            config, source_config, source_run, smoke=bool(args.smoke)
        )
        bundle.save(staging / "source_samples.npz")
        _write_csv(staging / "source_index.csv", source_rows)
        seeds = list(map(int, config["training"]["seeds"]))
        if args.smoke:
            seeds = seeds[: int(config["smoke"]["seeds"])]
        device = torch.device(args.device or config["training"]["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        combined: dict[str, list[dict[str, Any]]] = defaultdict(list)
        cells = 0
        for task in config["tasks"]:
            for seed in seeds:
                cell_dir = staging / "cells" / str(task["task_id"]) / f"seed_{seed}"
                cell_dir.mkdir(parents=True)
                result = train_cell(
                    config, task, seed, bundle, cell_dir, device, smoke=bool(args.smoke)
                )
                for key, rows in result.items():
                    combined[key].extend(rows)
                cells += 1
                print(
                    f"completed {task['task_id']} seed={seed} "
                    f"({cells}/{len(config['tasks']) * len(seeds)})",
                    flush=True,
                )
        for key, filename in (
            ("endpoints", "subject_endpoints.csv"),
            ("raw", "raw_ablation_metrics.csv"),
            ("probes", "ridge_probe_metrics.csv"),
            ("diagnostics", "latent_diagnostics.csv"),
            ("derangements", "derangement_registry.csv"),
            ("history", "loss_curves.csv"),
        ):
            _write_csv(staging / filename, combined[key])
        manifest = {
            "schema": SCHEMA,
            "status": "completed",
            "mode": "smoke" if args.smoke else "full",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": config["experiment"],
            "source_run": str(source_run.relative_to(REPO_ROOT)),
            "source_manifest_sha256": _sha256(source_run / "manifest.json"),
            "source_completed_cells": source_manifest["completed_cell_count"],
            "source_sample_count": len(bundle.sample_id),
            "task_count": len(config["tasks"]),
            "seed_count": len(seeds),
            "cell_count": cells,
            "completed_cell_count": cells,
            "failed_cell_count": 0,
            "fit_subjects_by_dataset": {
                dataset: sorted(set(bundle.subject[(bundle.dataset_id == dataset) & (bundle.role == "fit")].tolist()))
                for dataset in sorted(set(bundle.dataset_id.tolist()))
            },
            "validation_subjects_by_dataset": {
                dataset: sorted(set(bundle.subject[(bundle.dataset_id == dataset) & (bundle.role == "development_validation")].tolist()))
                for dataset in sorted(set(bundle.dataset_id.tolist()))
            },
            "vector_quantization": False,
            "protected_open": False,
            "protected_loader_constructed": False,
            "primary_endpoint_family_size": len(config["tasks"]) * len(ENDPOINTS),
            "git": _git_payload(),
            "inputs": [
                {"path": str(config_path.relative_to(REPO_ROOT)), "sha256": _sha256(config_path)},
                {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "sha256": _sha256(Path(__file__).resolve())},
                {"path": "src/tokenizers/continuous_shared_private.py", "sha256": _sha256(REPO_ROOT / "src/tokenizers/continuous_shared_private.py")},
                {"path": str((source_run / "manifest.json").relative_to(REPO_ROOT)), "sha256": _sha256(source_run / "manifest.json")},
            ],
            "artifacts": _artifact_inventory(staging),
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        return target
    except Exception:
        print(f"failed staging retained at {staging}", file=sys.stderr)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/continuous_shared_private_latent.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(output)
