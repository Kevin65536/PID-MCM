#!/usr/bin/env python3
"""Extract frozen EEG features and fit one public-validation linear probe."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
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


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
EFRM_ROOT = REPO_ROOT / "comparative_methods/EFRM-PyTorch"
for import_path in (REPO_ROOT, EFRM_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from comparative_methods.BIOT.adapters import BIOTFrozenEncoder, load_verified_biot_encoder
from comparative_methods.CBraMod.adapters import (
    CBraModFrozenEncoder,
    load_verified_cbramod_encoder,
)
from comparative_methods.REVE.adapters import REVEFrozenEncoder, load_verified_reve_base
from comparative_methods.audit_public_preflight import sha256_file
from comparative_methods.single_modal_eeg.contract import (
    SCHEMA,
    SUPPORTED_METHODS,
    EEGTaskView,
    PublicTaskContract,
    balanced_subset,
    data_branch_fingerprints,
    load_public_contract,
    make_feature_loader,
    resolve_repo_path,
    stable_hash,
)
from efrm_pytorch.metrics import classification_metrics


RUN_SCHEMA = "single_modal_eeg_public_run_v1"
FEATURE_SCHEMA = "single_modal_eeg_public_feature_cache_v1"
CHECKPOINT_SCHEMA = "single_modal_eeg_linear_probe_checkpoint_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
            raise RuntimeError(f"comparison GPU lane is already locked: {path}") from exc
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


def adapter_source_path(method: str) -> Path:
    names = {"biot": "BIOT", "cbramod": "CBraMod", "reve": "REVE"}
    return REPO_ROOT / f"comparative_methods/{names[method]}/adapters/{method}.py"


def load_frozen_model(
    method: str, *, device: torch.device, config: Mapping[str, Any]
) -> tuple[nn.Module, dict[str, Any]]:
    method_config = config["methods"][method]
    if method == "biot":
        encoder, metadata = load_verified_biot_encoder(
            str(method_config["artifact_id"]), device=device
        )
        model: nn.Module = BIOTFrozenEncoder(encoder)
    elif method == "cbramod":
        encoder, metadata = load_verified_cbramod_encoder(device=device)
        model = CBraModFrozenEncoder(
            encoder, token_pooling=str(method_config["token_pooling"])
        )
    elif method == "reve":
        encoder, position_bank, metadata = load_verified_reve_base(device=device)
        model = REVEFrozenEncoder(encoder, position_bank)
    else:
        raise KeyError(f"unsupported method: {method}")
    model.requires_grad_(False)
    model.eval()
    identity = jsonable(asdict(metadata))
    identity.update(
        {
            "method": method,
            "adapter_path": str(adapter_source_path(method).resolve()),
            "adapter_sha256": sha256_file(adapter_source_path(method)),
            "source_input_note": str(method_config["source_input_note"]),
        }
    )
    return model, identity


def feature_cache_identity(
    *,
    method_identity: Mapping[str, Any],
    contract: PublicTaskContract,
    data_fingerprints: Mapping[str, str],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    smoke: bool,
) -> dict[str, Any]:
    payload = {
        "schema": FEATURE_SCHEMA,
        "method_identity": method_identity,
        "contract": contract.manifest(),
        "data_branch_sha256": dict(data_fingerprints),
        "train_indices_sha256": stable_hash([int(value) for value in train_indices]),
        "validation_indices_sha256": stable_hash(
            [int(value) for value in validation_indices]
        ),
        "smoke": bool(smoke),
        "protected_test_opened": False,
    }
    payload["feature_cache_key"] = stable_hash(payload)
    return payload


def extract_partition(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    contract: PublicTaskContract,
    device: torch.device,
) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dataset_indices: list[np.ndarray] = []
    subjects: list[str] = []
    sample_ids: list[str] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            batch_size, channels, samples = eeg.shape
            channel_valid = torch.ones(
                (batch_size, channels), dtype=torch.bool, device=device
            )
            sample_valid = torch.ones(
                (batch_size, samples), dtype=torch.bool, device=device
            )
            embedding = model(
                eeg,
                sampling_rate_hz=float(contract.config["data"]["eeg_sample_rate_hz"]),
                channel_names=contract.panel,
                channel_valid=channel_valid,
                sample_valid=sample_valid,
            )
            if embedding.ndim != 2 or embedding.shape[0] != batch_size:
                raise RuntimeError(f"invalid frozen embedding shape: {tuple(embedding.shape)}")
            if not bool(torch.isfinite(embedding).all()):
                raise FloatingPointError("frozen encoder produced non-finite features")
            features.append(embedding.detach().float().cpu().numpy())
            targets.append(batch["target"].numpy())
            dataset_indices.append(batch["dataset_index"].numpy())
            subjects.extend(str(value) for value in batch["subject"])
            sample_ids.extend(str(value) for value in batch["sample_id"])
    if not features:
        raise RuntimeError("feature extraction produced no public samples")
    output = {
        "features": np.concatenate(features).astype(np.float32, copy=False),
        "targets": np.concatenate(targets).astype(np.int64, copy=False),
        "dataset_indices": np.concatenate(dataset_indices).astype(np.int64, copy=False),
        "subjects": np.asarray(subjects, dtype=str),
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "elapsed_seconds": np.asarray([time.perf_counter() - started], dtype=np.float64),
    }
    if len(set(output["sample_ids"].tolist())) != len(output["sample_ids"]):
        raise RuntimeError("feature partition contains duplicate sample identities")
    return output


def build_or_load_features(
    *,
    model: nn.Module,
    contract: PublicTaskContract,
    method: str,
    method_identity: Mapping[str, Any],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    device: torch.device,
    feature_cache_root: Path,
    smoke: bool,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    data_fingerprints = data_branch_fingerprints(contract)
    identity = feature_cache_identity(
        method_identity=method_identity,
        contract=contract,
        data_fingerprints=data_fingerprints,
        train_indices=train_indices,
        validation_indices=validation_indices,
        smoke=smoke,
    )
    key = str(identity["feature_cache_key"])
    cache_path = feature_cache_root / method / contract.task / f"{key}.npz"
    manifest_path = cache_path.with_suffix(".json")
    if "protected" in {part.lower() for part in cache_path.resolve().parts}:
        raise PermissionError(f"refusing protected feature-cache path: {cache_path}")
    if cache_path.is_file() and manifest_path.is_file():
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached_manifest != identity:
            raise RuntimeError(f"feature-cache manifest drifted: {manifest_path}")
        with np.load(cache_path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        expected = len(train_indices) + len(validation_indices)
        if arrays["features"].shape[0] != expected:
            raise RuntimeError("feature-cache sample count differs from public contract")
        return arrays, identity, True

    view = EEGTaskView(contract)
    resources = contract.config["resources"]
    batch_size = int(resources["feature_batch_size"])
    workers = int(resources["data_loader_workers"])
    train_loader = make_feature_loader(
        view, train_indices, batch_size=batch_size, workers=workers, seed=seed
    )
    validation_loader = make_feature_loader(
        view, validation_indices, batch_size=batch_size, workers=workers, seed=seed
    )
    train = extract_partition(
        model=model, loader=train_loader, contract=contract, device=device
    )
    validation = extract_partition(
        model=model, loader=validation_loader, contract=contract, device=device
    )
    arrays = {
        "features": np.concatenate((train["features"], validation["features"])),
        "targets": np.concatenate((train["targets"], validation["targets"])),
        "dataset_indices": np.concatenate(
            (train["dataset_indices"], validation["dataset_indices"])
        ),
        "subjects": np.concatenate((train["subjects"], validation["subjects"])),
        "sample_ids": np.concatenate((train["sample_ids"], validation["sample_ids"])),
        "partitions": np.asarray(
            ["train"] * len(train_indices) + ["validation"] * len(validation_indices),
            dtype=str,
        ),
        "partition_elapsed_seconds": np.concatenate(
            (train["elapsed_seconds"], validation["elapsed_seconds"])
        ),
    }
    if set(arrays["dataset_indices"][: len(train_indices)].tolist()) != set(train_indices):
        raise RuntimeError("extracted train membership differs from public split")
    if set(arrays["dataset_indices"][len(train_indices) :].tolist()) != set(
        validation_indices
    ):
        raise RuntimeError("extracted validation membership differs from public split")
    save_npz(cache_path, **arrays)
    write_json(manifest_path, identity)
    return arrays, identity, False


def class_weights(target: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(target.astype(np.int64), minlength=class_count).astype(np.float64)
    if bool((counts <= 0).any()):
        raise RuntimeError(f"training split has an empty class: {counts.tolist()}")
    weights = counts.sum() / (class_count * counts)
    return weights.astype(np.float32)


def train_candidate(
    *,
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    validation_features: torch.Tensor,
    validation_target: np.ndarray,
    class_names: Sequence[str],
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    weights: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    head = nn.Linear(train_features.shape[1], len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        head.train()
        permutation = torch.randperm(len(train_target), generator=generator)
        losses = []
        for start in range(0, len(permutation), int(batch_size)):
            selected = permutation[start : start + int(batch_size)].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_features[selected])
            loss = torch.nn.functional.cross_entropy(
                logits, train_target[selected], weight=weights
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("linear probe produced non-finite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():
            validation_logits = head(validation_features).float().cpu().numpy()
        metrics = classification_metrics(validation_target, validation_logits, class_names)
        metric = float(metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": metric,
            }
        )
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in head.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("linear probe did not produce a best checkpoint")
    head.load_state_dict(best_state, strict=True)
    head.eval()
    with torch.inference_mode():
        logits = head(validation_features).float().cpu().numpy()
    return {
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "best_epoch": best_epoch,
        "best_macro_f1": best_metric,
        "head_state": best_state,
        "validation_logits": logits,
        "history": history,
    }


def baseline_metrics(
    train_target: np.ndarray,
    validation_target: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    counts = np.bincount(train_target, minlength=len(class_names)).astype(np.float64)
    majority = int(counts.argmax())
    majority_logits = np.zeros((len(validation_target), len(class_names)), dtype=np.float64)
    majority_logits[:, majority] = 1.0
    prior = counts / counts.sum()
    prior_logits = np.broadcast_to(np.log(prior.clip(1e-12)), majority_logits.shape).copy()
    return {
        "majority_class": class_names[majority],
        "train_class_counts": counts.astype(int).tolist(),
        "majority": classification_metrics(
            validation_target, majority_logits, class_names
        ),
        "train_prior": classification_metrics(validation_target, prior_logits, class_names),
    }


def fit_probe(
    *,
    arrays: Mapping[str, np.ndarray],
    contract: PublicTaskContract,
    method: str,
    seed: int,
    device: torch.device,
    smoke: bool,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    partition = arrays["partitions"].astype(str)
    train_mask = partition == "train"
    validation_mask = partition == "validation"
    train_x = arrays["features"][train_mask].astype(np.float32)
    validation_x = arrays["features"][validation_mask].astype(np.float32)
    train_y = arrays["targets"][train_mask].astype(np.int64)
    validation_y = arrays["targets"][validation_mask].astype(np.int64)
    epsilon = float(contract.config["probe"]["feature_standardization_epsilon"])
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > epsilon, scale, 1.0).astype(np.float32)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all():
        raise FloatingPointError("train-only feature standardization produced non-finite values")

    section = contract.config["smoke"] if smoke else contract.config["probe"]
    epochs = int(section["epochs"])
    learning_rates = [float(value) for value in section["learning_rates"]]
    weight_decays = [float(value) for value in section["weight_decays"]]
    batch_size = int(contract.config["probe"]["batch_size"])
    weights_np = class_weights(train_y, len(contract.class_names))
    train_features = torch.from_numpy(train_x).to(device)
    validation_features = torch.from_numpy(validation_x).to(device)
    train_target = torch.from_numpy(train_y).to(device)
    weights = torch.from_numpy(weights_np).to(device)
    candidates = []
    best: dict[str, Any] | None = None
    for candidate_index, (learning_rate, weight_decay) in enumerate(
        itertools.product(learning_rates, weight_decays)
    ):
        candidate = train_candidate(
            train_features=train_features,
            train_target=train_target,
            validation_features=validation_features,
            validation_target=validation_y,
            class_names=contract.class_names,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            weights=weights,
            seed=seed,
            device=device,
        )
        candidate["candidate_index"] = candidate_index
        candidates.append(
            {key: value for key, value in candidate.items() if key not in {"head_state", "validation_logits"}}
        )
        if best is None or float(candidate["best_macro_f1"]) > float(best["best_macro_f1"]):
            best = candidate
    if best is None:
        raise RuntimeError("linear probe grid is empty")
    final_metrics = classification_metrics(
        validation_y, best["validation_logits"], contract.class_names
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "method": method,
        "task": contract.task,
        "outer_fold": contract.outer_fold,
        "seed": int(seed),
        "head_state": best["head_state"],
        "feature_mean": torch.from_numpy(mean),
        "feature_scale": torch.from_numpy(scale),
        "class_names": list(contract.class_names),
        "learning_rate": float(best["learning_rate"]),
        "weight_decay": float(best["weight_decay"]),
        "best_epoch": int(best["best_epoch"]),
        "protected_test_opened": False,
    }
    checkpoint_path = output_dir / "checkpoint_best.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reload_head = nn.Linear(mean.size, len(contract.class_names)).to(device)
    reload_head.load_state_dict(reloaded["head_state"], strict=True)
    reload_head.eval()
    with torch.inference_mode():
        reloaded_logits = reload_head(validation_features).float().cpu().numpy()
    if not np.allclose(reloaded_logits, best["validation_logits"], rtol=1e-5, atol=1e-6):
        raise RuntimeError("weights-only linear-probe checkpoint reload changed predictions")
    report = {
        "selection_metric": "macro_f1",
        "selection_mode": "max",
        "selected_candidate": {
            "learning_rate": best["learning_rate"],
            "weight_decay": best["weight_decay"],
            "best_epoch": best["best_epoch"],
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "validation_metrics": final_metrics,
        "baselines": baseline_metrics(train_y, validation_y, contract.class_names),
        "train_only_standardization": {
            "epsilon": epsilon,
            "flat_feature_count": int((scale == 1.0).sum()),
        },
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "weights_only_reload_match": True,
    }
    return report, reloaded_logits


def run(args: argparse.Namespace) -> None:
    output_dir = resolve_repo_path(args.output_dir)
    if "protected" in {part.lower() for part in output_dir.parts}:
        raise PermissionError(f"refusing protected output path: {output_dir}")
    if (output_dir / "status.json").exists() or (output_dir / "manifest.json").exists():
        raise FileExistsError(f"run output already exists: {output_dir}")
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
    contract = load_public_contract(
        args.config, task=args.task, outer_fold=int(args.outer_fold)
    )
    if args.method not in SUPPORTED_METHODS:
        raise KeyError(f"unsupported method: {args.method}")
    allowed_seeds = [int(value) for value in contract.config["probe"]["seed_set"]]
    if int(args.seed) not in allowed_seeds:
        raise ValueError(f"seed {args.seed} is outside frozen seed set {allowed_seeds}")
    configured_device = str(contract.config["resources"]["default_device"])
    device_name = str(args.device or configured_device)
    if device_name != configured_device:
        raise ValueError(
            f"v1 resource contract freezes device={configured_device}, got {device_name}"
        )
    if not torch.cuda.is_available() or not device_name.startswith("cuda:"):
        raise RuntimeError("single-modal public performance extraction requires CUDA")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    set_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")
    train_indices: Sequence[int] = contract.train_indices
    validation_indices: Sequence[int] = contract.validation_indices
    if args.smoke:
        train_indices = balanced_subset(
            contract,
            train_indices,
            samples_per_class=int(contract.config["smoke"]["train_samples_per_class"]),
        )
        validation_indices = balanced_subset(
            contract,
            validation_indices,
            samples_per_class=int(
                contract.config["smoke"]["validation_samples_per_class"]
            ),
        )
    feature_cache_root = resolve_repo_path(
        args.feature_cache_root
        or "comparative_methods/single_modal_eeg/runs/feature_cache"
    )
    lock_path = Path(str(contract.config["resources"]["gpu_lock_path"]))
    started = time.perf_counter()
    with exclusive_gpu_lock(lock_path):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / 2**30
        required_gib = float(contract.config["resources"]["minimum_free_gpu_gib"])
        if free_gib < required_gib:
            raise RuntimeError(
                f"GPU has {free_gib:.2f} GiB free, below required {required_gib:.2f} GiB"
            )
        torch.cuda.reset_peak_memory_stats(device)
        model, method_identity = load_frozen_model(
            args.method, device=device, config=contract.config
        )
        arrays, cache_identity, cache_hit = build_or_load_features(
            model=model,
            contract=contract,
            method=args.method,
            method_identity=method_identity,
            train_indices=train_indices,
            validation_indices=validation_indices,
            device=device,
            feature_cache_root=feature_cache_root,
            smoke=bool(args.smoke),
            seed=int(args.seed),
        )
        del model
        torch.cuda.empty_cache()
        probe_report, validation_logits = fit_probe(
            arrays=arrays,
            contract=contract,
            method=args.method,
            seed=int(args.seed),
            device=device,
            smoke=bool(args.smoke),
            output_dir=output_dir,
        )
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30

    partitions = arrays["partitions"].astype(str)
    validation_mask = partitions == "validation"
    save_npz(
        output_dir / "validation_predictions.npz",
        logits=validation_logits.astype(np.float32),
        target=arrays["targets"][validation_mask].astype(np.int64),
        dataset_index=arrays["dataset_indices"][validation_mask].astype(np.int64),
        subject=arrays["subjects"][validation_mask].astype(str),
        sample_id=arrays["sample_ids"][validation_mask].astype(str),
    )
    write_json(output_dir / "validation_metrics.json", probe_report)
    manifest = {
        "schema": RUN_SCHEMA,
        "status": "completed",
        "mode": "smoke_only" if args.smoke else "public_development_performance",
        "method": args.method,
        "task": args.task,
        "outer_fold": int(args.outer_fold),
        "seed": int(args.seed),
        "device": device_name,
        "track": contract.config["track"],
        "contract": contract.manifest(),
        "method_identity": method_identity,
        "feature_cache": {
            **cache_identity,
            "cache_hit": cache_hit,
            "feature_dimension": int(arrays["features"].shape[1]),
        },
        "train_sample_count": int((partitions == "train").sum()),
        "validation_sample_count": int(validation_mask.sum()),
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
                "status": manifest["status"],
                "mode": manifest["mode"],
                "method": args.method,
                "task": args.task,
                "validation_macro_f1": probe_report["validation_metrics"]["macro_f1"],
                "feature_cache_hit": cache_hit,
                "protected_test_opened": False,
                "output": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True, choices=SUPPORTED_METHODS)
    parser.add_argument("--task", required=True)
    parser.add_argument("--outer-fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--feature-cache-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
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
            and existing_status.get("status") in {None, "initializing"}
        )
        if may_record_failure:
            output.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
