#!/usr/bin/env python3
"""Run one NormWear public selection/full-public-refit job under protocol v2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
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
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
EFRM_ROOT = REPO_ROOT / "comparative_methods/EFRM-PyTorch"
for import_path in (REPO_ROOT, METHOD_ROOT, EFRM_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import (  # noqa: E402
    METHOD_ID,
    SUPPORTED_TASKS,
    PublicInventory,
    load_config as load_alignment_config,
    load_public_inventory,
    resolve_repo_path,
    stable_hash,
)
from comparative_methods.audit_public_preflight import (  # noqa: E402
    public_json,
    registry_manifest,
    sha256_file,
    strict_public_entry,
)
from efrm_pytorch.metrics import classification_metrics  # noqa: E402


CONFIG_SCHEMA = "normwear_public_development_v2"
RUN_SCHEMA = "normwear_public_development_run_v2"
CHECKPOINT_SCHEMA = "normwear_public_refit_checkpoint_v2"
FEATURE_SCHEMA = "normwear_full_public_feature_cache_v2"
DEFAULT_CONFIG = METHOD_ROOT / "configs/public_development_v2.yaml"
ALIGNMENT_CONTRACT = REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"


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
        raise PermissionError(f"refusing protected NormWear output path: {resolved}")
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
            raise RuntimeError(f"NormWear public GPU lane is already locked: {path}") from exc
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


def load_runner_config(
    path: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected {CONFIG_SCHEMA} config: {config_path}")
    if config.get("method_id") != METHOD_ID or config.get("mode") != "public_development_only":
        raise PermissionError("runner config must remain NormWear public development only")
    if config.get("protected_test_default") != "locked":
        raise PermissionError("protected test must default to locked")
    contract = yaml.safe_load(ALIGNMENT_CONTRACT.read_text(encoding="utf-8"))
    active = contract.get("execution_policy", {}).get("active_delivery_method")
    if active != METHOD_ID:
        raise PermissionError(f"NormWear is not the active serial delivery method: {active!r}")

    alignment_path = resolve_repo_path(config["alignment"]["config"])
    if sha256_file(alignment_path) != str(config["alignment"]["config_sha256"]):
        raise RuntimeError("NormWear alignment config fingerprint drifted")
    alignment, resolved_alignment_path = load_alignment_config(alignment_path)
    evidence_path = resolve_repo_path(config["alignment"]["evidence_summary"])
    if sha256_file(evidence_path) != str(config["alignment"]["evidence_summary_sha256"]):
        raise RuntimeError("NormWear alignment evidence fingerprint drifted")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != config["alignment"]["required_status"]:
        raise RuntimeError("NormWear alignment evidence lacks the required reviewed status")
    if evidence.get("protected_test_opened", False):
        raise PermissionError("NormWear alignment evidence reports protected access")

    matrix = config["job_matrix"]
    tasks = tuple(str(value) for value in matrix["tasks"])
    folds = tuple(int(value) for value in matrix["outer_folds"])
    seeds = tuple(int(value) for value in matrix["seeds"])
    if tasks != SUPPORTED_TASKS or folds != tuple(range(5)) or len(set(seeds)) != len(seeds):
        raise ValueError("NormWear public matrix differs from the reviewed task/fold contract")
    if int(matrix["expected_public_jobs"]) != len(tasks) * len(folds) * len(seeds):
        raise ValueError("NormWear expected public job count is inconsistent")
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
        raise RuntimeError("selected public metadata identity differs from full inventory")
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


def selected_feature_digest(
    *,
    features: np.ndarray,
    rows: np.ndarray,
    sample_ids: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(rows), 64):
        block_rows = rows[start : start + 64]
        block = np.asarray(features[block_rows], dtype=np.float32)
        for identifier, feature in zip(
            sample_ids[start : start + 64], block, strict=True
        ):
            digest.update(str(identifier).encode("utf-8"))
            digest.update(b"\0")
            digest.update(np.ascontiguousarray(feature).tobytes())
    return digest.hexdigest()


def load_verified_feature_cache(
    *, config: Mapping[str, Any], fold: PublicFold
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path, dict[str, Any]]:
    evidence_root = resolve_repo_path(config["alignment"]["evidence_root"])
    cell_path = evidence_root / f"{fold.inventory.task}.json"
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    if cell.get("method_id") != METHOD_ID or cell.get("task_id") != fold.inventory.task:
        raise RuntimeError("NormWear A7 cell identity drifted")
    gates = cell.get("gate_status", {})
    if any(gates.get(f"A{number}") != "pass" for number in range(8)):
        raise RuntimeError("NormWear feature cache lacks a complete A0-A7 gate chain")
    if cell.get("protected_test_opened", False):
        raise PermissionError("NormWear A7 cell reports protected access")
    report = cell.get("public_adapter_audit", {})
    if not report.get("all_unique_public_samples_executed", False):
        raise RuntimeError("NormWear A7 cell lacks full-public adapter execution")
    cache_dir = resolve_repo_path(report["cache_directory"])
    if "protected" in {part.lower() for part in cache_dir.parts}:
        raise PermissionError("NormWear feature cache crosses the protected boundary")
    identity_path = cache_dir / "identity.json"
    status_path = cache_dir / "status.json"
    features_path = cache_dir / "features.npy"
    metadata_path = cache_dir / "metadata.npz"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if identity.get("schema") != FEATURE_SCHEMA:
        raise RuntimeError("NormWear feature cache schema drifted")
    if identity.get("task") != fold.inventory.task:
        raise RuntimeError("NormWear feature cache task drifted")
    if stable_hash(identity) != str(report["feature_cache_identity_sha256"]):
        raise RuntimeError("NormWear feature cache identity digest drifted")
    if identity.get("feature_cache_key") != report.get("feature_cache_key"):
        raise RuntimeError("NormWear feature cache key drifted")
    if status.get("state") != "complete" or status.get("protected_test_opened", False):
        raise RuntimeError("NormWear feature cache is incomplete or crossed protected data")
    if status.get("feature_cache_key") != identity.get("feature_cache_key"):
        raise RuntimeError("NormWear feature status identity drifted")
    if not features_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("NormWear A7 feature cache files are missing")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    count = len(fold.inventory.indices)
    dimension = len(fold.inventory.delivered_channel_names) * 768
    if features.shape != (count, dimension) or features.dtype != np.float32:
        raise RuntimeError("NormWear feature shape or dtype drifted")
    if list(features.shape) != list(report["feature_shape"]):
        raise RuntimeError("NormWear feature shape differs from A7 evidence")

    # The A7 cache covers the union of public inventories across all folds.  Do
    # not load its global target/subject arrays here: A8 materializes labels only
    # for the selected fold's public train/validation membership.
    dataset_indices = np.asarray(fold.inventory.indices, dtype=np.int64)
    sample_ids = np.asarray(fold.inventory.sample_ids, dtype=str)
    targets = np.full(count, -1, dtype=np.int64)
    subjects = np.full(count, "", dtype="U128")
    row_lookup = {int(index): row for row, index in enumerate(dataset_indices.tolist())}
    for index in fold.public_indices:
        row = fold.inventory.dataset.lightweight_metadata(int(index))
        position = row_lookup[int(index)]
        targets[position] = fold.inventory.dataset.class_to_index[str(row["condition"])]
        subjects[position] = str(row["subject"])
    public_rows = np.asarray(
        [row_lookup[index] for index in fold.public_indices], dtype=np.int64
    )
    if bool((targets[public_rows] < 0).any()) or bool((subjects[public_rows] == "").any()):
        raise RuntimeError("NormWear selected public membership lacks labels or subjects")
    if not bool(np.isfinite(np.asarray(features[public_rows[0]])).all()):
        raise RuntimeError("NormWear first selected public feature row is non-finite")
    arrays: dict[str, np.ndarray] = {
        "features": features,
        "targets": targets,
        "dataset_indices": dataset_indices,
        "subjects": subjects,
        "sample_ids": sample_ids,
    }
    verification = {
        "a7_cell_path": portable_path(cell_path),
        "a7_cell_sha256": sha256_file(cell_path),
        "feature_file_sha256_from_a7": str(report["feature_file_sha256"]),
        "metadata_file_sha256_from_a7": str(report["metadata_file_sha256"]),
        "feature_sha256_from_a7": str(report["feature_sha256"]),
        "global_target_metadata_loaded": False,
        "selected_public_rows_materialized": len(public_rows),
        "selected_public_indices_sha256": stable_hash(list(fold.public_indices)),
        "selected_public_feature_sha256": selected_feature_digest(
            features=features,
            rows=public_rows,
            sample_ids=sample_ids[public_rows].astype(str).tolist(),
        ),
        "protected_test_opened": False,
    }
    return arrays, identity, cache_dir, verification


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


def standardize_copy(
    features: np.ndarray, rows: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    result = np.asarray(features[rows], dtype=np.float32)
    np.subtract(result, mean, out=result)
    np.divide(result, scale, out=result)
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("standardized NormWear features are non-finite")
    return result


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
                raise FloatingPointError("NormWear linear probe produced non-finite loss")
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
            metrics = classification_metrics(validation_target, validation_logits, class_names)
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
        "majority": classification_metrics(validation_target, majority_logits, class_names),
        "train_prior": classification_metrics(validation_target, prior_logits, class_names),
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
    cache_identity: Mapping[str, Any],
    seed: int,
    device: torch.device,
    smoke: bool,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    train_rows = rows_for_indices(arrays, train_indices)
    validation_rows = rows_for_indices(arrays, validation_indices)
    train_y = arrays["targets"][train_rows].astype(np.int64)
    validation_y = arrays["targets"][validation_rows].astype(np.int64)
    epsilon = float(config["selection"]["feature_standardization_epsilon"])
    raw_train = np.asarray(arrays["features"][train_rows], dtype=np.float32)
    selection_mean, selection_scale = standardizer(raw_train, epsilon)
    np.subtract(raw_train, selection_mean, out=raw_train)
    np.divide(raw_train, selection_scale, out=raw_train)
    train_standard = raw_train
    validation_standard = standardize_copy(
        arrays["features"], validation_rows, selection_mean, selection_scale
    )

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
            raise RuntimeError("NormWear public selection produced no candidate checkpoint")
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
        raise RuntimeError("NormWear public selection grid is empty")
    validation_logits = np.asarray(selected["logits"], dtype=np.float32)
    validation_metrics = classification_metrics(
        validation_y, validation_logits, fold.class_names
    )

    del train_features, validation_features, train_target, weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    refit_indices = tuple(dict.fromkeys([*train_indices, *validation_indices]))
    if set(refit_indices) != set(train_indices).union(validation_indices):
        raise RuntimeError("NormWear refit membership differs from public train+validation")
    refit_rows = rows_for_indices(arrays, refit_indices)
    raw_refit = np.asarray(arrays["features"][refit_rows], dtype=np.float32)
    refit_y = arrays["targets"][refit_rows].astype(np.int64)
    refit_mean, refit_scale = standardizer(raw_refit, epsilon)
    np.subtract(raw_refit, refit_mean, out=raw_refit)
    np.divide(raw_refit, refit_scale, out=raw_refit)
    refit_features = torch.from_numpy(raw_refit).to(device)
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
    feature_dimension = int(arrays["features"].shape[1])
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "method_id": METHOD_ID,
        "task": fold.inventory.task,
        "outer_fold": fold.outer_fold,
        "seed": int(seed),
        "head_state": {
            name: value.detach().cpu().clone() for name, value in refit_head.state_dict().items()
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
        "feature_cache_identity_sha256": stable_hash(cache_identity),
        "protected_test_opened": False,
    }
    checkpoint_path = output_dir / "checkpoint_public_refit.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reload_head = nn.Linear(feature_dimension, class_count).to(device)
    reload_head.load_state_dict(reloaded["head_state"], strict=True)
    reload_head.eval()
    with torch.inference_mode():
        reloaded_logits = reload_head(refit_features).float().cpu().numpy()
    if not np.array_equal(refit_mean, reloaded["feature_mean"].numpy()):
        raise RuntimeError("weights-only reload changed NormWear refit feature mean")
    if not np.array_equal(refit_scale, reloaded["feature_scale"].numpy()):
        raise RuntimeError("weights-only reload changed NormWear refit feature scale")
    if not np.allclose(reloaded_logits, expected_refit_logits, rtol=1e-5, atol=1e-6):
        raise RuntimeError("weights-only reload changed NormWear refit logits")

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
        raise ValueError(f"task is outside the frozen NormWear matrix: {args.task}")
    if int(args.outer_fold) not in config["job_matrix"]["outer_folds"]:
        raise ValueError("outer fold is outside the frozen NormWear matrix")
    if int(args.seed) not in config["job_matrix"]["seeds"]:
        raise ValueError("seed is outside the frozen NormWear matrix")
    output_dir = resolve_repo_path(args.output_dir)
    run_root = resolve_repo_path(config["resources"]["run_root"])
    try:
        output_dir.relative_to(run_root)
    except ValueError as exc:
        raise PermissionError(f"NormWear public output must remain under {run_root}") from exc
    if output_dir == run_root:
        raise ValueError("NormWear output directory must identify one job below run root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"NormWear public run output already exists: {output_dir}")
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
        raise RuntimeError("NormWear public development requires the frozen CUDA lane")
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
        arrays, cache_identity, cache_dir, cache_verification = load_verified_feature_cache(
            config=config, fold=fold
        )
        probe_report, validation_logits = select_and_refit(
            arrays=arrays,
            fold=fold,
            train_indices=train_indices,
            validation_indices=validation_indices,
            config=config,
            config_path=config_path,
            alignment_path=alignment_path,
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
        "method_id": METHOD_ID,
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
        "feature_cache": {
            "identity": cache_identity,
            "directory": portable_path(cache_dir),
            **cache_verification,
            "a7_verified": True,
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
