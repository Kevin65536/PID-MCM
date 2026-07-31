#!/usr/bin/env python3
"""Run one BIOT public selection/full-public-refit job under protocol v2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
EFRM_ROOT = REPO_ROOT / "comparative_methods/EFRM-PyTorch"
for import_path in (REPO_ROOT, METHOD_ROOT, EFRM_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from comparative_methods.BIOT.adapters.biot import (
    BIOTFrozenEncoder,
    load_verified_biot_encoder,
)
from comparative_methods.BIOT.alignment_data import (
    SUPPORTED_TASKS,
    BIOTPublicView,
    PublicInventory,
    RecordGroupedBatchSampler,
    data_branch_fingerprints,
    load_config as load_alignment_config,
    load_public_inventory,
    resolve_repo_path,
    stable_hash,
)
from comparative_methods.audit_public_preflight import (
    public_json,
    registry_manifest,
    sha256_file,
    strict_public_entry,
)
from efrm_pytorch.metrics import classification_metrics


CONFIG_SCHEMA = "biot_public_development_v2"
RUN_SCHEMA = "biot_public_development_run_v2"
FEATURE_SCHEMA = "biot_full_public_feature_cache_v2"
CHECKPOINT_SCHEMA = "biot_public_refit_checkpoint_v2"
DEFAULT_CONFIG = METHOD_ROOT / "configs/public_development_v2.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected output path: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


@contextmanager
def exclusive_gpu_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"BIOT public GPU lane is already locked: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_runner_config(path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected {CONFIG_SCHEMA} config: {config_path}")
    if config.get("method_id") != "biot" or config.get("mode") != "public_development_only":
        raise PermissionError("runner config must remain BIOT public development only")
    if config.get("protected_test_default") != "locked":
        raise PermissionError("protected test must default to locked")
    alignment_path = resolve_repo_path(config["alignment"]["config"])
    if sha256_file(alignment_path) != str(config["alignment"]["config_sha256"]):
        raise RuntimeError("BIOT alignment config fingerprint drifted")
    alignment, resolved_alignment_path = load_alignment_config(alignment_path)
    evidence_path = resolve_repo_path(config["alignment"]["evidence_summary"])
    if sha256_file(evidence_path) != str(config["alignment"]["evidence_summary_sha256"]):
        raise RuntimeError("BIOT alignment evidence fingerprint drifted")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != config["alignment"]["required_status"]:
        raise RuntimeError("BIOT alignment evidence does not have the required reviewed status")
    if evidence.get("protected_test_opened", False):
        raise PermissionError("BIOT alignment evidence reports protected access")
    matrix = config["job_matrix"]
    tasks = tuple(str(value) for value in matrix["tasks"])
    folds = tuple(int(value) for value in matrix["outer_folds"])
    seeds = tuple(int(value) for value in matrix["seeds"])
    if tasks != SUPPORTED_TASKS or folds != tuple(range(5)) or len(set(seeds)) != len(seeds):
        raise ValueError("BIOT public job matrix differs from the reviewed task/fold contract")
    if int(matrix["expected_public_jobs"]) != len(tasks) * len(folds) * len(seeds):
        raise ValueError("BIOT expected public job count is inconsistent")
    if int(config["failure_policy"]["automatic_retry_count"]) != 0:
        raise ValueError("automatic retries are not admitted before A8 freeze")
    return config, config_path, alignment, resolved_alignment_path


@dataclass(frozen=True)
class PublicFold:
    inventory: PublicInventory
    outer_fold: int
    public_manifest_path: Path
    public_manifest_sha256: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.inventory.dataset.spec.class_names)

    @property
    def public_indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.train_indices).union(self.validation_indices)))

    @property
    def public_sample_ids(self) -> tuple[str, ...]:
        lookup = dict(zip(self.inventory.indices, self.inventory.sample_ids, strict=True))
        return tuple(lookup[index] for index in self.public_indices)

    @property
    def public_sample_inventory_sha256(self) -> str:
        return stable_hash(sorted(self.public_sample_ids))


def load_public_fold(
    alignment: Mapping[str, Any], *, task: str, outer_fold: int
) -> PublicFold:
    inventory = load_public_inventory(alignment, task=task)
    registry = registry_manifest(resolve_repo_path(alignment["registry"]["manifest"]))
    entry = strict_public_entry(registry, task=task, outer_fold=int(outer_fold))
    public_path = Path(str(entry["public_path"])).resolve()
    public_manifest = public_json(public_path)
    digest = sha256_file(public_path)
    if digest != str(entry["public_sha256"]):
        raise RuntimeError("selected public split fingerprint drifted")
    train, validation = inventory.dataset.validate_shared_public_split(public_path)
    if set(train).intersection(validation):
        raise RuntimeError("selected public train/validation partitions overlap")
    if len(train) != int(entry["train_sample_count"]):
        raise RuntimeError("selected public train sample count drifted")
    if len(validation) != int(entry["validation_sample_count"]):
        raise RuntimeError("selected public validation sample count drifted")
    if str(public_manifest["metadata_sha256"]) != str(inventory.split_rows[0]["metadata_sha256"]):
        raise RuntimeError("selected public metadata identity differs from the full inventory")
    return PublicFold(
        inventory=inventory,
        outer_fold=int(outer_fold),
        public_manifest_path=public_path,
        public_manifest_sha256=digest,
        train_indices=tuple(int(index) for index in train),
        validation_indices=tuple(int(index) for index in validation),
    )


def balanced_subset(
    fold: PublicFold, indices: Sequence[int], *, samples_per_class: int
) -> tuple[int, ...]:
    grouped: dict[str, list[int]] = {name: [] for name in fold.class_names}
    for index in indices:
        condition = str(fold.inventory.dataset.lightweight_metadata(int(index))["condition"])
        if condition not in grouped:
            raise RuntimeError(f"unexpected public class in {fold.inventory.task}: {condition}")
        grouped[condition].append(int(index))
    selected: list[int] = []
    for name in fold.class_names:
        if len(grouped[name]) < samples_per_class:
            raise RuntimeError(f"public class {name} is too small for the smoke subset")
        selected.extend(grouped[name][:samples_per_class])
    return tuple(selected)


def adapter_identity(metadata: Any, alignment_path: Path) -> dict[str, Any]:
    identity = jsonable(asdict(metadata))
    identity["path"] = portable_path(Path(metadata.path))
    identity["source_file_sha256"] = {
        "adapter": sha256_file(METHOD_ROOT / "adapters/biot.py"),
        "alignment_data": sha256_file(METHOD_ROOT / "alignment_data.py"),
        "upstream_model": sha256_file(METHOD_ROOT / "upstream/model/biot.py"),
        "method_manifest": sha256_file(METHOD_ROOT / "sources/method_manifest.yaml"),
        "alignment_config": sha256_file(alignment_path),
    }
    identity.update(
        {
            "output_layer": "upstream_biot_encoder_mean_embedding",
            "embedding_dim": 256,
            "trainable_parameter_boundary": "frozen_encoder_linear_probe_only",
        }
    )
    return identity


def feature_cache_identity(
    *,
    fold: PublicFold,
    method_identity: Mapping[str, Any],
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema": FEATURE_SCHEMA,
        "method_identity": method_identity,
        "task": fold.inventory.task,
        "outer_fold": fold.outer_fold,
        "public_manifest_sha256": fold.public_manifest_sha256,
        "sample_inventory_sha256": fold.public_sample_inventory_sha256,
        "panel": list(fold.inventory.panel),
        "duration_s": fold.inventory.duration_s,
        "data_branch_sha256": data_branch_fingerprints(alignment),
        "feature_extraction": {
            "sampling_rate_hz": float(alignment["data"]["eeg_sample_rate_hz"]),
            "channel_policy": str(alignment["data"]["channel_policy"]),
            "time_support_policy": str(alignment["data"]["time_support_policy"]),
            "embedding_dim": 256,
        },
        "protected_test_opened": False,
    }
    value["feature_cache_key"] = stable_hash(value)
    return value


def feature_sha256(sample_ids: Sequence[str], features: np.ndarray) -> str:
    digest = hashlib.sha256()
    for identifier, row in zip(sample_ids, features, strict=True):
        digest.update(str(identifier).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(row, dtype=np.float32).tobytes())
    return digest.hexdigest()


def extract_or_load_features(
    *,
    fold: PublicFold,
    model: BIOTFrozenEncoder,
    method_identity: Mapping[str, Any],
    alignment: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool, Path]:
    identity = feature_cache_identity(
        fold=fold, method_identity=method_identity, alignment=alignment
    )
    cache_root = resolve_repo_path(config["resources"]["feature_cache_root"])
    cache_path = cache_root / fold.inventory.task / f"{identity['feature_cache_key']}.npz"
    manifest_path = cache_path.with_suffix(".json")
    if "protected" in {part.lower() for part in cache_path.resolve().parts}:
        raise PermissionError("feature cache path crosses the protected boundary")
    if cache_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != identity:
            raise RuntimeError("BIOT feature cache manifest differs from its identity")
        with np.load(cache_path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        validate_feature_arrays(arrays, fold)
        return arrays, identity, True, cache_path

    view = BIOTPublicView(
        fold.inventory, sample_rate_hz=float(alignment["data"]["eeg_sample_rate_hz"])
    )
    workers = int(config["resources"]["data_loader_workers"])
    sampler = RecordGroupedBatchSampler(
        view.dataset,
        fold.public_indices,
        batch_size=int(config["resources"]["feature_batch_size"]),
        seed=42,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(view, **loader_kwargs)
    feature_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            eeg = batch["eeg"].to(device, non_blocking=True)
            batch_size, channels, samples = eeg.shape
            embedding = model(
                eeg,
                sampling_rate_hz=float(alignment["data"]["eeg_sample_rate_hz"]),
                channel_names=fold.inventory.panel,
                channel_valid=torch.ones(
                    (batch_size, channels), dtype=torch.bool, device=device
                ),
                sample_valid=torch.ones(
                    (batch_size, samples), dtype=torch.bool, device=device
                ),
            )
            if embedding.shape != (batch_size, 256) or not bool(torch.isfinite(embedding).all()):
                raise RuntimeError("BIOT feature extraction returned invalid output")
            feature_parts.append(embedding.float().cpu().numpy())
            index_parts.append(batch["dataset_index"].numpy())
            sample_ids.extend(str(value) for value in batch["sample_id"])
            if batch_number % 100 == 0 or batch_number == len(loader):
                print(
                    f"[{fold.inventory.task}] feature cache {batch_number}/{len(loader)} batches",
                    flush=True,
                )
    features = np.concatenate(feature_parts).astype(np.float32, copy=False)
    dataset_indices = np.concatenate(index_parts).astype(np.int64, copy=False)
    order = np.argsort(dataset_indices)
    features = features[order]
    dataset_indices = dataset_indices[order]
    sorted_ids = np.asarray(sample_ids, dtype=str)[order]
    rows = [
        fold.inventory.dataset.lightweight_metadata(int(index)) for index in dataset_indices
    ]
    arrays = {
        "features": features,
        "targets": np.asarray(
            [fold.inventory.dataset.class_to_index[str(row["condition"])] for row in rows],
            dtype=np.int64,
        ),
        "dataset_indices": dataset_indices,
        "subjects": np.asarray([str(row["subject"]) for row in rows], dtype=str),
        "sample_ids": sorted_ids,
    }
    validate_feature_arrays(arrays, fold)
    save_npz(cache_path, **arrays)
    write_json(manifest_path, identity)
    return arrays, identity, False, cache_path


def validate_feature_arrays(arrays: Mapping[str, np.ndarray], fold: PublicFold) -> None:
    required = {"features", "targets", "dataset_indices", "subjects", "sample_ids"}
    if set(arrays) != required:
        raise RuntimeError(f"BIOT feature cache arrays differ: {sorted(arrays)}")
    count = len(fold.public_indices)
    if arrays["features"].shape != (count, 256):
        raise RuntimeError("BIOT feature cache has an invalid feature shape")
    if any(len(arrays[name]) != count for name in required - {"features"}):
        raise RuntimeError("BIOT feature cache arrays have inconsistent row counts")
    if tuple(arrays["dataset_indices"].astype(int).tolist()) != fold.public_indices:
        raise RuntimeError("BIOT feature cache does not cover the exact public inventory")
    if tuple(arrays["sample_ids"].astype(str).tolist()) != fold.public_sample_ids:
        raise RuntimeError("BIOT feature cache sample identity order drifted")
    if not bool(np.isfinite(arrays["features"]).all()):
        raise RuntimeError("BIOT feature cache contains non-finite values")
    if not bool((arrays["features"].std(axis=0, dtype=np.float64) > 1e-8).any()):
        raise RuntimeError("BIOT feature cache is globally constant")


def rows_for_indices(arrays: Mapping[str, np.ndarray], indices: Sequence[int]) -> np.ndarray:
    lookup = {
        int(index): position
        for position, index in enumerate(arrays["dataset_indices"].astype(int).tolist())
    }
    missing = [int(index) for index in indices if int(index) not in lookup]
    if missing:
        raise RuntimeError(f"public fold indices are absent from feature cache: {missing[:5]}")
    return np.asarray([lookup[int(index)] for index in indices], dtype=np.int64)


def standardizer(features: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    raw_scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(raw_scale > float(epsilon), raw_scale, 1.0).astype(np.float32)
    if not bool(np.isfinite(mean).all()) or not bool(np.isfinite(scale).all()):
        raise FloatingPointError("feature standardizer is non-finite")
    return mean, scale


def class_weights(target: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(target.astype(np.int64), minlength=class_count).astype(np.float64)
    if bool((counts <= 0).any()):
        raise RuntimeError(f"training membership has an empty class: {counts.tolist()}")
    return (counts.sum() / (class_count * counts)).astype(np.float32)


def train_epochs(
    *,
    features: torch.Tensor,
    target: torch.Tensor,
    class_count: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    weights: torch.Tensor,
    seed: int,
    device: torch.device,
    validation_features: torch.Tensor | None = None,
    validation_target: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
) -> tuple[nn.Linear, list[dict[str, Any]], dict[str, Any] | None]:
    set_seed(seed)
    head = nn.Linear(features.shape[1], int(class_count)).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for epoch in range(1, int(epochs) + 1):
        head.train()
        permutation = torch.randperm(len(target), generator=generator)
        losses: list[float] = []
        for start in range(0, len(permutation), int(batch_size)):
            selected = permutation[start : start + int(batch_size)].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features[selected])
            loss = torch.nn.functional.cross_entropy(logits, target[selected], weight=weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("BIOT linear probe produced non-finite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        row: dict[str, Any] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if validation_features is not None:
            if validation_target is None or class_names is None:
                raise ValueError("validation features require targets and class names")
            head.eval()
            with torch.inference_mode():
                validation_logits = head(validation_features).float().cpu().numpy()
            metrics = classification_metrics(
                validation_target, validation_logits, class_names
            )
            metric = float(metrics["macro_f1"])
            row["validation_macro_f1"] = metric
            if best is None or metric > float(best["metric"]):
                best = {
                    "metric": metric,
                    "epoch": epoch,
                    "state": {
                        name: value.detach().cpu().clone()
                        for name, value in head.state_dict().items()
                    },
                    "logits": validation_logits,
                }
        history.append(row)
    return head, history, best


def baseline_metrics(
    train_target: np.ndarray, validation_target: np.ndarray, class_names: Sequence[str]
) -> dict[str, Any]:
    counts = np.bincount(train_target, minlength=len(class_names)).astype(np.float64)
    majority = int(counts.argmax())
    majority_logits = np.zeros((len(validation_target), len(class_names)), dtype=np.float64)
    majority_logits[:, majority] = 1.0
    prior = counts / counts.sum()
    prior_logits = np.broadcast_to(np.log(prior.clip(1e-12)), majority_logits.shape).copy()
    return {
        "majority_class": str(class_names[majority]),
        "train_class_counts": counts.astype(int).tolist(),
        "majority": classification_metrics(
            validation_target, majority_logits, class_names
        ),
        "train_prior": classification_metrics(
            validation_target, prior_logits, class_names
        ),
    }


def select_and_refit(
    *,
    arrays: Mapping[str, np.ndarray],
    fold: PublicFold,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    config: Mapping[str, Any],
    config_path: Path,
    alignment_path: Path,
    method_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    seed: int,
    device: torch.device,
    smoke: bool,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    train_rows = rows_for_indices(arrays, train_indices)
    validation_rows = rows_for_indices(arrays, validation_indices)
    train_x = arrays["features"][train_rows].astype(np.float32)
    validation_x = arrays["features"][validation_rows].astype(np.float32)
    train_y = arrays["targets"][train_rows].astype(np.int64)
    validation_y = arrays["targets"][validation_rows].astype(np.int64)
    epsilon = float(config["selection"]["feature_standardization_epsilon"])
    selection_mean, selection_scale = standardizer(train_x, epsilon)
    train_standard = (train_x - selection_mean) / selection_scale
    validation_standard = (validation_x - selection_mean) / selection_scale
    if not bool(np.isfinite(train_standard).all()) or not bool(
        np.isfinite(validation_standard).all()
    ):
        raise FloatingPointError("outer-train feature standardization is non-finite")

    section = config["smoke"] if smoke else config["selection"]
    epochs = int(section["epoch_cap"])
    learning_rates = [float(value) for value in section["learning_rates"]]
    weight_decays = [float(value) for value in section["weight_decays"]]
    batch_size = int(config["selection"]["batch_size"])
    class_count = len(fold.class_names)
    weights_np = class_weights(train_y, class_count)
    train_features = torch.from_numpy(train_standard).to(device)
    validation_features = torch.from_numpy(validation_standard).to(device)
    train_target = torch.from_numpy(train_y).to(device)
    weights = torch.from_numpy(weights_np).to(device)
    candidate_reports: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate_index, (learning_rate, weight_decay) in enumerate(
        itertools.product(learning_rates, weight_decays)
    ):
        _head, history, best = train_epochs(
            features=train_features,
            target=train_target,
            class_count=class_count,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            weights=weights,
            seed=seed,
            device=device,
            validation_features=validation_features,
            validation_target=validation_y,
            class_names=fold.class_names,
        )
        if best is None:
            raise RuntimeError("BIOT public selection produced no candidate checkpoint")
        candidate = {
            "candidate_index": candidate_index,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "best_epoch": int(best["epoch"]),
            "best_macro_f1": float(best["metric"]),
            "history": history,
            "state": best["state"],
            "logits": best["logits"],
        }
        candidate_reports.append(
            {key: value for key, value in candidate.items() if key not in {"state", "logits"}}
        )
        if selected is None or float(candidate["best_macro_f1"]) > float(
            selected["best_macro_f1"]
        ):
            selected = candidate
    if selected is None:
        raise RuntimeError("BIOT public selection grid is empty")
    validation_logits = np.asarray(selected["logits"], dtype=np.float32)
    validation_metrics = classification_metrics(
        validation_y, validation_logits, fold.class_names
    )

    refit_indices = tuple(dict.fromkeys([*train_indices, *validation_indices]))
    if set(refit_indices) != set(train_indices).union(validation_indices):
        raise RuntimeError("BIOT refit membership differs from public train+validation")
    refit_rows = rows_for_indices(arrays, refit_indices)
    refit_x = arrays["features"][refit_rows].astype(np.float32)
    refit_y = arrays["targets"][refit_rows].astype(np.int64)
    refit_mean, refit_scale = standardizer(refit_x, epsilon)
    refit_standard = (refit_x - refit_mean) / refit_scale
    refit_features = torch.from_numpy(refit_standard).to(device)
    refit_target = torch.from_numpy(refit_y).to(device)
    refit_weights = torch.from_numpy(class_weights(refit_y, class_count)).to(device)
    refit_head, refit_history, _unused = train_epochs(
        features=refit_features,
        target=refit_target,
        class_count=class_count,
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        epochs=int(selected["best_epoch"]),
        batch_size=batch_size,
        weights=refit_weights,
        seed=seed,
        device=device,
    )
    refit_head.eval()
    with torch.inference_mode():
        expected_refit_logits = refit_head(refit_features).float().cpu().numpy()
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "method_id": "biot",
        "task": fold.inventory.task,
        "outer_fold": fold.outer_fold,
        "seed": int(seed),
        "head_state": {
            name: value.detach().cpu().clone()
            for name, value in refit_head.state_dict().items()
        },
        "feature_mean": torch.from_numpy(refit_mean),
        "feature_scale": torch.from_numpy(refit_scale),
        "class_names": list(fold.class_names),
        "selected_learning_rate": float(selected["learning_rate"]),
        "selected_weight_decay": float(selected["weight_decay"]),
        "selected_epoch": int(selected["best_epoch"]),
        "refit_dataset_indices": torch.tensor(refit_indices, dtype=torch.long),
        "runner_config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "alignment_config_sha256": sha256_file(alignment_path),
        "method_identity_sha256": stable_hash(method_identity),
        "feature_cache_identity_sha256": stable_hash(cache_identity),
        "protected_test_opened": False,
    }
    checkpoint_path = output_dir / "checkpoint_public_refit.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reload_head = nn.Linear(256, class_count).to(device)
    reload_head.load_state_dict(reloaded["head_state"], strict=True)
    reload_head.eval()
    with torch.inference_mode():
        reloaded_logits = reload_head(refit_features).float().cpu().numpy()
    if not np.array_equal(refit_mean, reloaded["feature_mean"].numpy()):
        raise RuntimeError("weights-only reload changed the BIOT refit feature mean")
    if not np.array_equal(refit_scale, reloaded["feature_scale"].numpy()):
        raise RuntimeError("weights-only reload changed the BIOT refit feature scale")
    if not np.allclose(reloaded_logits, expected_refit_logits, rtol=1e-5, atol=1e-6):
        raise RuntimeError("weights-only reload changed BIOT refit logits")

    report = {
        "selection_metric": "macro_f1",
        "selection_mode": "max",
        "tie_break": "lower_candidate_index",
        "selected_candidate": {
            "candidate_index": int(selected["candidate_index"]),
            "learning_rate": float(selected["learning_rate"]),
            "weight_decay": float(selected["weight_decay"]),
            "best_epoch": int(selected["best_epoch"]),
        },
        "candidates": candidate_reports,
        "validation_metrics": validation_metrics,
        "baselines": baseline_metrics(train_y, validation_y, fold.class_names),
        "selection_standardizer": {
            "fit_membership": "outer_train_only",
            "epsilon": epsilon,
            "state_sha256": stable_hash(
                {"mean": selection_mean.tolist(), "scale": selection_scale.tolist()}
            ),
        },
        "public_refit": {
            "membership": (
                "smoke_train_plus_validation_subset"
                if smoke
                else "outer_train_plus_public_validation"
            ),
            "sample_count": len(refit_indices),
            "epochs": int(selected["best_epoch"]),
            "history": refit_history,
            "standardizer_state_sha256": stable_hash(
                {"mean": refit_mean.tolist(), "scale": refit_scale.tolist()}
            ),
            "checkpoint_path": portable_path(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "weights_only_reload_match": True,
        },
    }
    return report, validation_logits


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_path, alignment, alignment_path = load_runner_config(args.config)
    if args.task not in config["job_matrix"]["tasks"]:
        raise ValueError(f"task is outside the frozen BIOT matrix: {args.task}")
    if int(args.outer_fold) not in config["job_matrix"]["outer_folds"]:
        raise ValueError("outer fold is outside the frozen BIOT matrix")
    if int(args.seed) not in config["job_matrix"]["seeds"]:
        raise ValueError("seed is outside the frozen BIOT matrix")
    output_dir = resolve_repo_path(args.output_dir)
    run_root = resolve_repo_path(config["resources"]["run_root"])
    try:
        output_dir.relative_to(run_root)
    except ValueError as exc:
        raise PermissionError(f"BIOT public output must remain under {run_root}") from exc
    if output_dir == run_root:
        raise ValueError("BIOT output directory must identify one job below the run root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"BIOT public run output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "status": "initializing",
            "pid": os.getpid(),
            "started_at": utc_now(),
            "protected_test_opened": False,
        },
    )
    started = time.perf_counter()
    fold = load_public_fold(alignment, task=args.task, outer_fold=int(args.outer_fold))
    train_indices: Sequence[int] = fold.train_indices
    validation_indices: Sequence[int] = fold.validation_indices
    if args.smoke:
        train_indices = balanced_subset(
            fold,
            train_indices,
            samples_per_class=int(config["smoke"]["train_samples_per_class"]),
        )
        validation_indices = balanced_subset(
            fold,
            validation_indices,
            samples_per_class=int(config["smoke"]["validation_samples_per_class"]),
        )

    device_name = str(config["resources"]["device"])
    if not torch.cuda.is_available() or not device_name.startswith("cuda:"):
        raise RuntimeError("BIOT public development requires the frozen CUDA lane")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    set_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")
    lock_path = Path(str(config["resources"]["gpu_lock_path"]))
    with exclusive_gpu_lock(lock_path):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / 2**30
        required_gib = float(config["resources"]["minimum_free_gpu_gib"])
        if free_gib < required_gib:
            raise RuntimeError(
                f"GPU has {free_gib:.2f} GiB free, below required {required_gib:.2f} GiB"
            )
        torch.cuda.reset_peak_memory_stats(device)
        encoder, metadata = load_verified_biot_encoder(
            str(alignment["adapter"]["artifact_id"]), device=device
        )
        model = BIOTFrozenEncoder(encoder).to(device).eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("BIOT public runner found trainable encoder parameters")
        method_identity = adapter_identity(metadata, alignment_path)
        arrays, cache_identity, cache_hit, cache_path = extract_or_load_features(
            fold=fold,
            model=model,
            method_identity=method_identity,
            alignment=alignment,
            config=config,
            device=device,
        )
        del model, encoder
        torch.cuda.empty_cache()
        probe_report, validation_logits = select_and_refit(
            arrays=arrays,
            fold=fold,
            train_indices=train_indices,
            validation_indices=validation_indices,
            config=config,
            config_path=config_path,
            alignment_path=alignment_path,
            method_identity=method_identity,
            cache_identity=cache_identity,
            seed=int(args.seed),
            device=device,
            smoke=bool(args.smoke),
            output_dir=output_dir,
        )
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30

    validation_rows = rows_for_indices(arrays, validation_indices)
    save_npz(
        output_dir / "public_validation_predictions.npz",
        logits=validation_logits.astype(np.float32),
        target=arrays["targets"][validation_rows].astype(np.int64),
        dataset_index=arrays["dataset_indices"][validation_rows].astype(np.int64),
        subject=arrays["subjects"][validation_rows].astype(str),
        sample_id=arrays["sample_ids"][validation_rows].astype(str),
    )
    write_json(output_dir / "public_selection_report.json", probe_report)
    manifest = {
        "schema": RUN_SCHEMA,
        "status": "completed",
        "mode": "smoke_only" if args.smoke else "public_selection_and_refit",
        "method_id": "biot",
        "task": args.task,
        "outer_fold": int(args.outer_fold),
        "seed": int(args.seed),
        "track": str(config["track"]),
        "device": device_name,
        "runner_config_path": portable_path(config_path),
        "runner_config_sha256": sha256_file(config_path),
        "runner_path": portable_path(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "alignment_config_path": portable_path(alignment_path),
        "alignment_config_sha256": sha256_file(alignment_path),
        "public_manifest_path": portable_path(fold.public_manifest_path),
        "public_manifest_sha256": fold.public_manifest_sha256,
        "method_identity": method_identity,
        "feature_cache": {
            **cache_identity,
            "path": portable_path(cache_path),
            "file_sha256": sha256_file(cache_path),
            "feature_sha256": feature_sha256(
                arrays["sample_ids"].astype(str).tolist(), arrays["features"]
            ),
            "cache_hit": cache_hit,
        },
        "selection_train_sample_count": len(train_indices),
        "selection_validation_sample_count": len(validation_indices),
        "probe": probe_report,
        "wall_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_gib": peak_allocated,
        "cuda_peak_reserved_gib": peak_reserved,
        "table_admissible": False,
        "claim_boundary": (
            "connectivity_smoke_not_performance_evidence"
            if args.smoke
            else "public_validation_development_only_not_protected_or_final_table_evidence"
        ),
        "protected_test_opened": False,
        "completed_at": utc_now(),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "status.json", manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "mode": manifest["mode"],
                "task": args.task,
                "outer_fold": int(args.outer_fold),
                "seed": int(args.seed),
                "validation_macro_f1": probe_report["validation_metrics"]["macro_f1"],
                "feature_cache_hit": cache_hit,
                "protected_test_opened": False,
                "output": portable_path(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", required=True)
    parser.add_argument("--outer-fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        output = resolve_repo_path(args.output_dir)
        protected_path = "protected" in {part.lower() for part in output.parts}
        existing_manifest = (output / "manifest.json").exists()
        existing_status: dict[str, Any] = {}
        if (output / "status.json").is_file():
            try:
                existing_status = json.loads(
                    (output / "status.json").read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                existing_status = {"status": "unknown_existing"}
        may_record_failure = (
            not protected_path
            and not existing_manifest
            and existing_status.get("status") == "initializing"
        )
        if may_record_failure:
            write_json(
                output / "status.json",
                {
                    "schema": RUN_SCHEMA,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "protected_test_opened": False,
                    "failed_at": utc_now(),
                },
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
