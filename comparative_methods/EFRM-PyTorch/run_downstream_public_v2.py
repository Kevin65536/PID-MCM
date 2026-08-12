#!/usr/bin/env python3
"""Run one EFRM LODO-v2 public selection and full-outer-refit job.

This entry point is intentionally public-only.  It builds a task-wide cache
from the union of the five public manifests, never accepts a protected
manifest, selects only the training epoch on one outer-training/public-
validation split, and refits the probe on that fold's complete public support.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from comparative_methods.audit_public_preflight import (  # noqa: E402
    EXPECTED_REGISTRY_SHA256,
    public_json,
    registry_manifest,
    sha256_file,
    strict_public_entry,
)
from efrm_pytorch.metrics import classification_metrics, regression_metrics  # noqa: E402
from efrm_pytorch.model import EFRMSyncModel  # noqa: E402
from efrm_pytorch.tasks import (  # noqa: E402
    EFRMUnifiedTaskDataset,
    TASK_SPECS,
    collate_efrm_task,
)
from train_downstream import RecordGroupedBatchSampler, move_batch  # noqa: E402


CONFIG_SCHEMA = "efrm_lodo_downstream_public_v2"
RUN_SCHEMA = "efrm_lodo_downstream_public_run_v2"
FEATURE_SCHEMA = "efrm_lodo_full_public_feature_cache_v2"
CHECKPOINT_SCHEMA = "efrm_lodo_public_refit_checkpoint_v2"
PROTOCOL_ID = "efrm_lodo_full_target_fivefold_v2"
METHOD_ID = "efrm_sync_200_10_variable_channel_v1"
DEFAULT_CONFIG = METHOD_ROOT / "configs/downstream_public_v2.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
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
        raise PermissionError(f"refusing protected EFRM output path: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected EFRM cache path: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected EFRM checkpoint path: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


@contextmanager
def exclusive_gpu_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"EFRM v2 GPU lane is already locked: {path}") from exc
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected {CONFIG_SCHEMA}: {config_path}")
    if config.get("protocol_id") != PROTOCOL_ID or config.get("method_id") != METHOD_ID:
        raise ValueError("EFRM downstream config method/protocol identity drifted")
    if config.get("mode") != "public_development_only":
        raise PermissionError("EFRM v2 downstream must remain public development only")
    if config.get("protected_test_default") != "locked":
        raise PermissionError("protected test must remain locked")
    matrix = config["job_matrix"]
    if tuple(matrix["tasks"]) != tuple(TASK_SPECS):
        raise ValueError("EFRM v2 task order differs from the seven-task contract")
    if tuple(int(value) for value in matrix["outer_folds"]) != tuple(range(5)):
        raise ValueError("EFRM v2 outer folds must be exactly 0-4")
    if tuple(int(value) for value in matrix["seeds"]) != (17, 42, 73):
        raise ValueError("EFRM v2 downstream seeds drifted")
    expected_jobs = len(matrix["tasks"]) * 5 * 3
    if int(matrix["expected_public_jobs"]) != expected_jobs:
        raise ValueError("EFRM v2 expected public job count is inconsistent")
    if int(config["failure_policy"]["automatic_retry_count"]) != 0:
        raise ValueError("automatic public retries are not frozen")

    protocol_root = resolve_repo_path(config["protocol"]["root"])
    status = json.loads((protocol_root / "status.json").read_text(encoding="utf-8"))
    if (
        status.get("schema") != "efrm_lodo_protocol_status_v2"
        or status.get("protocol_id") != PROTOCOL_ID
        or status.get("status") != "lodo_pretraining_completed"
        or int(status.get("selection_completed", -1)) != 4
        or int(status.get("final_refit_completed", -1)) != 4
    ):
        raise RuntimeError("EFRM v2 LODO pretraining is not in its completed frozen state")
    if status.get("protected_test_opened") or status.get("target_dataset_exposure"):
        raise PermissionError("EFRM v2 pretraining status crossed its target/protected boundary")
    registry_path = resolve_repo_path(config["protocol"]["registry"])
    registry = registry_manifest(registry_path)
    if registry.get("registry_sha256") != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("method-neutral fold registry identity drifted")
    if config["protocol"]["registry_sha256"] != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("configured fold registry fingerprint drifted")
    return config, config_path


@dataclass(frozen=True)
class PublicFold:
    task: str
    outer_fold: int
    public_manifest_path: Path
    public_manifest_sha256: str
    public_split_sha256: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass
class PublicSurface:
    task: str
    dataset: EFRMUnifiedTaskDataset
    folds: dict[int, PublicFold]
    full_public_indices: tuple[int, ...]
    public_inventory_sha256: str
    split_registry_sha256: str


def load_public_surface(config: Mapping[str, Any], *, task: str) -> PublicSurface:
    if task not in TASK_SPECS:
        raise KeyError(f"unknown EFRM v2 task: {task}")
    dataset = EFRMUnifiedTaskDataset(
        TASK_SPECS[task], cache_root=str(resolve_repo_path(config["data"]["cache_root"]))
    )
    registry_path = resolve_repo_path(config["protocol"]["registry"])
    registry = registry_manifest(registry_path)
    folds: dict[int, PublicFold] = {}
    union: set[int] = set()
    split_rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        entry = strict_public_entry(registry, task=task, outer_fold=outer_fold)
        public_path = Path(str(entry["public_path"])).resolve()
        if "protected" in {part.lower() for part in public_path.parts}:
            raise PermissionError("public registry entry resolves inside a protected path")
        digest = sha256_file(public_path)
        if digest != str(entry["public_sha256"]):
            raise RuntimeError(f"public split hash drifted: {public_path}")
        manifest = public_json(public_path)
        train, validation = dataset.validate_shared_public_split(public_path)
        if len(train) != int(entry["train_sample_count"]):
            raise RuntimeError("public train sample count drifted")
        if len(validation) != int(entry["validation_sample_count"]):
            raise RuntimeError("public validation sample count drifted")
        if set(train).intersection(validation):
            raise RuntimeError("public train and validation indices overlap")
        fold = PublicFold(
            task=task,
            outer_fold=outer_fold,
            public_manifest_path=public_path,
            public_manifest_sha256=digest,
            public_split_sha256=str(manifest["split_sha256"]),
            train_indices=tuple(int(value) for value in train),
            validation_indices=tuple(int(value) for value in validation),
        )
        folds[outer_fold] = fold
        union.update(train)
        union.update(validation)
        split_rows.append(
            {
                "outer_fold": outer_fold,
                "public_sha256": digest,
                "split_sha256": fold.public_split_sha256,
                "train_indices_sha256": stable_hash(fold.train_indices),
                "validation_indices_sha256": stable_hash(fold.validation_indices),
            }
        )
    full_indices = tuple(sorted(union))
    if full_indices != tuple(range(len(dataset))):
        missing = sorted(set(range(len(dataset))) - union)
        raise RuntimeError(
            f"five public folds do not cover the complete target inventory: {missing[:5]}"
        )
    inventory_payload = [
        dataset.lightweight_metadata(index) for index in full_indices
    ]
    return PublicSurface(
        task=task,
        dataset=dataset,
        folds=folds,
        full_public_indices=full_indices,
        public_inventory_sha256=stable_hash(inventory_payload),
        split_registry_sha256=stable_hash(split_rows),
    )


def frozen_checkpoint_identity(
    config: Mapping[str, Any], *, dataset_id: str
) -> dict[str, Any]:
    protocol_root = resolve_repo_path(config["protocol"]["root"])
    freeze_path = protocol_root / f"protocol/final_refits/exclude_{dataset_id}.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema") != "efrm_lodo_stage_b_refit_freeze_v2"
        or freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("excluded_target_dataset") != dataset_id
        or freeze.get("target_dataset_exposure") is not False
    ):
        raise RuntimeError(f"invalid EFRM v2 final-refit freeze: {freeze_path}")
    checkpoint_path = Path(str(freeze["terminal_checkpoint"])).resolve()
    run_root = checkpoint_path.parent.parent
    run_manifest_path = run_root / "manifest.json"
    boundary_path = run_root / "boundary_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    if sha256_file(run_manifest_path) != str(freeze["run_manifest_sha256"]):
        raise RuntimeError("EFRM v2 final-refit run manifest hash drifted")
    if (
        run_manifest.get("status") != "completed"
        or run_manifest.get("protected_test_opened") is not False
        or run_manifest.get("target_dataset_exposure") is not False
    ):
        raise PermissionError("EFRM v2 final-refit run crossed its frozen boundary")
    if (
        boundary.get("schema") != "efrm_pretraining_boundary_v1"
        or boundary.get("mode") != "lodo_final_refit_v2"
        or boundary.get("excluded_target_dataset") != dataset_id
        or boundary.get("target_dataset_exposure") is not False
        or boundary.get("protected_test_opened") is not False
    ):
        raise PermissionError("EFRM v2 checkpoint boundary is not target-excluded")
    if run_manifest.get("boundary_sha256") != boundary.get("boundary_sha256"):
        raise RuntimeError("EFRM v2 run/boundary identity drifted")
    if str(run_manifest.get("config_sha256")) != str(freeze["final_refit_config_sha256"]):
        raise RuntimeError("EFRM v2 final-refit config identity drifted")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing EFRM v2 terminal checkpoint: {checkpoint_path}")
    return {
        "freeze_path": portable_path(freeze_path),
        "freeze_sha256": sha256_file(freeze_path),
        "checkpoint_path": portable_path(checkpoint_path),
        "checkpoint_sha256": str(freeze["terminal_checkpoint_sha256"]),
        "run_manifest_path": portable_path(run_manifest_path),
        "run_manifest_sha256": str(freeze["run_manifest_sha256"]),
        "boundary_path": portable_path(boundary_path),
        "boundary_sha256": str(boundary["boundary_sha256"]),
        "excluded_target_dataset": dataset_id,
        "selected_epoch_count": int(freeze["selected_epoch_count"]),
        "target_dataset_exposure": False,
        "protected_test_opened": False,
    }


def build_backbone(config: Mapping[str, Any]) -> EFRMSyncModel:
    model = config["model"]
    return EFRMSyncModel(
        eeg_patch_samples=int(model["eeg_patch_samples"]),
        fnirs_patch_samples=int(model["fnirs_patch_samples"]),
        mask_ratio=float(model["mask_ratio"]),
        embed_dim=int(model["embedding_dim"]),
        depth=int(model["encoder_depth"]),
        num_heads=int(model["encoder_heads"]),
        decoder_embed_dim=int(model["decoder_embedding_dim"]),
        decoder_depth=int(model["decoder_depth"]),
        decoder_num_heads=int(model["decoder_heads"]),
        mlp_ratio=float(model["mlp_ratio"]),
        clip_logit_multiplier=float(model["clip_logit_multiplier"]),
        activation_checkpointing=False,
    )


def load_frozen_backbone(
    config: Mapping[str, Any], identity: Mapping[str, Any], *, device: torch.device
) -> EFRMSyncModel:
    checkpoint_path = resolve_repo_path(identity["checkpoint_path"])
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != identity["checkpoint_sha256"]:
        raise RuntimeError("EFRM v2 terminal checkpoint hash drifted")
    # The checkpoint is locally generated by this repository and is admitted
    # only after the frozen SHA-256/boundary chain above is verified.  Its
    # historical RNG tuple prevents PyTorch's restricted weights-only loader;
    # no external or unverified pickle is accepted here.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != "efrm_sync_pretrain_checkpoint_v1"
        or checkpoint.get("boundary_sha256") != identity["boundary_sha256"]
    ):
        raise RuntimeError("EFRM v2 checkpoint payload identity drifted")
    backbone = build_backbone(config)
    backbone.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    backbone.requires_grad_(False)
    backbone.eval().to(device)
    return backbone


class IndexedTaskView(Dataset[dict[str, Any]]):
    def __init__(self, dataset: EFRMUnifiedTaskDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def lightweight_metadata(self, index: int) -> dict[str, Any]:
        return self.dataset.lightweight_metadata(index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[int(index)]
        item["dataset_index"] = torch.tensor(int(index), dtype=torch.long)
        return item


def mask_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().numpy().astype(bool, copy=False)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(np.packbits(array.reshape(-1)).tobytes())
    return digest.hexdigest()


def collate_indexed(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = collate_efrm_task(samples)
    batch["dataset_index"] = torch.stack(
        [sample["dataset_index"] for sample in samples]
    )
    batch["eeg_channel_inventory_json"] = [
        json.dumps(list(sample["eeg_channel_names"]), separators=(",", ":"))
        for sample in samples
    ]
    batch["fnirs_location_inventory_json"] = [
        json.dumps(list(sample["fnirs_location_names"]), separators=(",", ":"))
        for sample in samples
    ]
    batch["eeg_time_valid_count"] = torch.tensor(
        [int(sample["eeg_time_valid"].sum()) for sample in samples], dtype=torch.int32
    )
    batch["fnirs_time_valid_count"] = torch.tensor(
        [int(sample["fnirs_time_valid"].sum()) for sample in samples], dtype=torch.int32
    )
    batch["eeg_patch_valid_count"] = torch.tensor(
        [int(sample["eeg_patch_valid"].sum()) for sample in samples], dtype=torch.int32
    )
    batch["fnirs_patch_valid_count"] = torch.tensor(
        [int(sample["fnirs_patch_valid"].sum()) for sample in samples], dtype=torch.int32
    )
    batch["eeg_valid_channel_count"] = torch.tensor(
        [int(sample["eeg_patch_valid"].any(dim=1).sum()) for sample in samples],
        dtype=torch.int32,
    )
    batch["fnirs_valid_location_count"] = torch.tensor(
        [int(sample["fnirs_patch_valid"].any(dim=1).sum()) for sample in samples],
        dtype=torch.int32,
    )
    for name in (
        "eeg_time_valid",
        "fnirs_time_valid",
        "eeg_patch_valid",
        "fnirs_patch_valid",
    ):
        batch[f"{name}_sha256"] = [mask_sha256(sample[name]) for sample in samples]
    return batch


def feature_cache_identity(
    config: Mapping[str, Any], surface: PublicSurface, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    task_batch_sizes = config["feature_extraction"].get("task_batch_sizes", {})
    batch_size = int(
        task_batch_sizes.get(
            surface.task, config["feature_extraction"]["batch_size"]
        )
    )
    model_sources = {
        "feature_materializer": sha256_file(Path(__file__)),
        "model": sha256_file(METHOD_ROOT / "efrm_pytorch/model.py"),
        "paired_adapter": sha256_file(METHOD_ROOT / "efrm_pytorch/data.py"),
        "task_dataset": sha256_file(METHOD_ROOT / "efrm_pytorch/tasks.py"),
        "unified_loader": sha256_file(REPO_ROOT / "src/data/unified_physiology.py"),
    }
    cache_root = resolve_repo_path(config["data"]["cache_root"])
    data_branch_paths = {
        "measurement_adapter": REPO_ROOT / "src/data/physiology_measurement_adapter.py",
        "cache_manifest": cache_root / "cache_manifest.json",
        "event_manifest": cache_root / "event_index/event_manifest.json",
        # This manifest is the authority for measured channel names and
        # geometry.  Including it prevents a stale feature cache from
        # surviving a channel-order/support/coordinate correction.
        "geometry_manifest": cache_root / "channel_geometry/geometry_manifest.json",
        "single_trial_eeg_branch": cache_root / "eeg_artifact_clean_v4/cache_manifest.json",
        "simultaneous_eeg_branch": (
            cache_root / "simultaneous_eeg_eog_clean_v1/cache_manifest.json"
        ),
    }
    missing = [str(path) for path in data_branch_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"EFRM feature-cache data identity is incomplete: {missing}")
    identity = {
        "schema": FEATURE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "task": surface.task,
        "dataset_id": surface.dataset.spec.dataset_id,
        "checkpoint": dict(checkpoint),
        "public_inventory_sha256": surface.public_inventory_sha256,
        "split_registry_sha256": surface.split_registry_sha256,
        "metadata_sha256": surface.dataset.metadata_fingerprint(),
        "full_public_indices_sha256": stable_hash(surface.full_public_indices),
        "source_sha256": model_sources,
        "data_branch_sha256": {
            name: sha256_file(path) for name, path in data_branch_paths.items()
        },
        "adapter_manifest": surface.dataset.adapter.manifest(),
        "feature_semantics": {
            "eeg_encoder_output": "valid_patch_masked_mean",
            "fnirs_encoder_output": "valid_patch_masked_mean",
            "paired_fusion": "elementwise_sum_before_trainable_layer_norm",
            "embedding_dim": int(config["model"]["embedding_dim"]),
            "autocast_dtype": str(config["feature_extraction"]["amp_dtype"]),
            "batch_size": batch_size,
        },
        "target_dataset_exposure": False,
        "protected_test_opened": False,
    }
    identity["feature_cache_key"] = stable_hash(identity)
    return identity


def validate_feature_arrays(
    arrays: Mapping[str, np.ndarray], surface: PublicSurface, *, embedding_dim: int
) -> None:
    required = {
        "features",
        "targets",
        "target_valid_mask",
        "dataset_indices",
        "subjects",
        "sample_ids",
        "eeg_channel_inventory_json",
        "fnirs_location_inventory_json",
        "eeg_time_valid_count",
        "fnirs_time_valid_count",
        "eeg_patch_valid_count",
        "fnirs_patch_valid_count",
        "eeg_valid_channel_count",
        "fnirs_valid_location_count",
        "eeg_time_valid_sha256",
        "fnirs_time_valid_sha256",
        "eeg_patch_valid_sha256",
        "fnirs_patch_valid_sha256",
    }
    if set(arrays) != required:
        raise RuntimeError(f"EFRM feature-cache arrays differ: {sorted(arrays)}")
    count = len(surface.full_public_indices)
    if arrays["features"].shape != (count, embedding_dim):
        raise RuntimeError("EFRM feature cache has an invalid feature shape")
    if tuple(arrays["dataset_indices"].astype(int).tolist()) != surface.full_public_indices:
        raise RuntimeError("EFRM feature cache does not cover the exact public inventory")
    if any(len(arrays[name]) != count for name in required - {"features"}):
        raise RuntimeError("EFRM feature-cache row counts differ")
    if len(set(arrays["sample_ids"].astype(str).tolist())) != count:
        raise RuntimeError("EFRM feature cache contains duplicate sample identities")
    for name in (
        "eeg_time_valid_sha256",
        "fnirs_time_valid_sha256",
        "eeg_patch_valid_sha256",
        "fnirs_patch_valid_sha256",
    ):
        if any(len(str(value)) != 64 for value in arrays[name]):
            raise RuntimeError(f"EFRM feature cache has an invalid support digest: {name}")
    eeg_duration_samples = int(round(surface.dataset.spec.input_duration_s * 200.0))
    fnirs_duration_samples = int(round(surface.dataset.spec.input_duration_s * 10.0))
    for position in range(count):
        eeg_names = json.loads(str(arrays["eeg_channel_inventory_json"][position]))
        fnirs_names = json.loads(str(arrays["fnirs_location_inventory_json"][position]))
        if (
            not eeg_names
            or not fnirs_names
            or len(eeg_names) != len(set(eeg_names))
            or len(fnirs_names) != len(set(fnirs_names))
        ):
            raise RuntimeError("EFRM feature cache has an invalid measured-channel inventory")
        eeg_time = int(arrays["eeg_time_valid_count"][position])
        fnirs_time = int(arrays["fnirs_time_valid_count"][position])
        eeg_channels = int(arrays["eeg_valid_channel_count"][position])
        fnirs_locations = int(arrays["fnirs_valid_location_count"][position])
        eeg_patches = int(arrays["eeg_patch_valid_count"][position])
        fnirs_patches = int(arrays["fnirs_patch_valid_count"][position])
        # REFED terminal windows may truthfully contain zero recorded support
        # for one modality.  The EFRM encoder consumes the retained all-false
        # mask and contributes no valid patches for that modality; deleting
        # the sample or inventing padding-as-data would change the estimand.
        if not (0 <= eeg_time <= eeg_duration_samples and 0 <= fnirs_time <= fnirs_duration_samples):
            raise RuntimeError("EFRM feature cache has invalid recorded-time support")
        if not (0 <= eeg_channels <= len(eeg_names) and 0 <= fnirs_locations <= len(fnirs_names)):
            raise RuntimeError("EFRM feature cache has invalid measured-channel support")
        if not (
            0 <= eeg_patches <= len(eeg_names) * (eeg_duration_samples // 50)
            and 0 <= fnirs_patches <= len(fnirs_names) * (fnirs_duration_samples // 20)
        ):
            raise RuntimeError("EFRM feature cache has invalid patch-mask support")
    if not bool(np.isfinite(arrays["features"]).all()):
        raise FloatingPointError("EFRM feature cache contains non-finite embeddings")
    if not bool((arrays["features"].std(axis=0, dtype=np.float64) > 1e-8).any()):
        raise RuntimeError("EFRM feature cache is globally constant")
    if surface.dataset.spec.task_type == "classification":
        if arrays["targets"].shape != (count,):
            raise RuntimeError("EFRM classification target cache has an invalid shape")
    else:
        expected = (
            count,
            surface.dataset.spec.output_dim,
            surface.dataset.spec.target_length,
        )
        if arrays["targets"].shape != expected or arrays["target_valid_mask"].shape != expected:
            raise RuntimeError("EFRM regression target/mask cache has an invalid shape")
        valid = arrays["target_valid_mask"].astype(bool)
        if not bool(valid.any()) or not bool(np.isfinite(arrays["targets"][valid]).all()):
            raise RuntimeError("EFRM regression cache has no finite valid target support")


def extract_or_load_features(
    *,
    config: Mapping[str, Any],
    surface: PublicSurface,
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool, Path]:
    identity = feature_cache_identity(config, surface, checkpoint)
    root = resolve_repo_path(config["resources"]["feature_cache_root"])
    cache_path = root / surface.task / f"{identity['feature_cache_key']}.npz"
    manifest_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and manifest_path.is_file():
        retained = json.loads(manifest_path.read_text(encoding="utf-8"))
        if retained != identity:
            raise RuntimeError("EFRM retained feature-cache identity drifted")
        with np.load(cache_path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        validate_feature_arrays(
            arrays, surface, embedding_dim=int(config["model"]["embedding_dim"])
        )
        return arrays, identity, True, cache_path

    backbone = load_frozen_backbone(config, checkpoint, device=device)
    view = IndexedTaskView(surface.dataset)
    task_batch_sizes = config["feature_extraction"].get("task_batch_sizes", {})
    batch_size = int(
        task_batch_sizes.get(
            surface.task, config["feature_extraction"]["batch_size"]
        )
    )
    sampler = RecordGroupedBatchSampler(
        view,
        surface.full_public_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
    )
    loader = DataLoader(
        view,
        batch_sampler=sampler,
        num_workers=int(config["feature_extraction"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_indexed,
    )
    amp_enabled = bool(config["feature_extraction"]["amp"])
    amp_dtype = (
        torch.bfloat16
        if config["feature_extraction"]["amp_dtype"] == "bfloat16"
        else torch.float16
    )
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    subjects: list[str] = []
    sample_ids: list[str] = []
    eeg_channel_inventories: list[str] = []
    fnirs_location_inventories: list[str] = []
    support_digests: dict[str, list[str]] = {
        name: []
        for name in (
            "eeg_time_valid_sha256",
            "fnirs_time_valid_sha256",
            "eeg_patch_valid_sha256",
            "fnirs_patch_valid_sha256",
        )
    }
    support_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "eeg_time_valid_count",
            "fnirs_time_valid_count",
            "eeg_patch_valid_count",
            "fnirs_patch_valid_count",
            "eeg_valid_channel_count",
            "fnirs_valid_location_count",
        )
    }
    with torch.inference_mode():
        for batch_number, raw in enumerate(loader, start=1):
            batch = move_batch(raw, device)
            context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with context:
                eeg_embedding, fnirs_embedding = backbone.encode(
                    batch["eeg"],
                    batch["fnirs"],
                    batch["eeg_patch_valid"],
                    batch["fnirs_patch_valid"],
                )
                features = eeg_embedding + fnirs_embedding
            if features.ndim != 2 or not bool(torch.isfinite(features).all()):
                raise RuntimeError("EFRM frozen encoder returned invalid paired features")
            feature_parts.append(features.float().cpu().numpy())
            target_parts.append(batch["target"].detach().cpu().numpy())
            mask_parts.append(batch["target_valid_mask"].detach().cpu().numpy())
            index_parts.append(batch["dataset_index"].detach().cpu().numpy())
            subjects.extend(str(value) for value in raw["subject"])
            sample_ids.extend(str(value) for value in raw["sample_id"])
            eeg_channel_inventories.extend(
                str(value) for value in raw["eeg_channel_inventory_json"]
            )
            fnirs_location_inventories.extend(
                str(value) for value in raw["fnirs_location_inventory_json"]
            )
            for name in support_parts:
                support_parts[name].append(
                    raw[name].detach().cpu().numpy().astype(np.int32, copy=False)
                )
            for name in support_digests:
                support_digests[name].extend(str(value) for value in raw[name])
            if batch_number % 100 == 0 or batch_number == len(loader):
                print(
                    f"[{surface.task}] EFRM feature cache {batch_number}/{len(loader)} batches",
                    flush=True,
                )
    dataset_indices = np.concatenate(index_parts).astype(np.int64, copy=False)
    order = np.argsort(dataset_indices)
    arrays = {
        "features": np.concatenate(feature_parts).astype(np.float32, copy=False)[order],
        "targets": np.concatenate(target_parts)[order],
        "target_valid_mask": np.concatenate(mask_parts).astype(bool, copy=False)[order],
        "dataset_indices": dataset_indices[order],
        "subjects": np.asarray(subjects, dtype=str)[order],
        "sample_ids": np.asarray(sample_ids, dtype=str)[order],
        "eeg_channel_inventory_json": np.asarray(
            eeg_channel_inventories, dtype=str
        )[order],
        "fnirs_location_inventory_json": np.asarray(
            fnirs_location_inventories, dtype=str
        )[order],
    }
    arrays.update(
        {
            name: np.concatenate(parts).astype(np.int32, copy=False)[order]
            for name, parts in support_parts.items()
        }
    )
    arrays.update(
        {
            name: np.asarray(values, dtype=str)[order]
            for name, values in support_digests.items()
        }
    )
    if surface.dataset.spec.task_type == "classification":
        arrays["targets"] = arrays["targets"].astype(np.int64, copy=False)
    else:
        arrays["targets"] = arrays["targets"].astype(np.float32, copy=False)
    validate_feature_arrays(
        arrays, surface, embedding_dim=int(config["model"]["embedding_dim"])
    )
    save_npz(cache_path, **arrays)
    write_json(manifest_path, identity)
    del backbone
    torch.cuda.empty_cache()
    return arrays, identity, False, cache_path


def rows_for_indices(arrays: Mapping[str, np.ndarray], indices: Sequence[int]) -> np.ndarray:
    lookup = {
        int(index): position
        for position, index in enumerate(arrays["dataset_indices"].astype(int).tolist())
    }
    missing = [int(index) for index in indices if int(index) not in lookup]
    if missing:
        raise RuntimeError(f"EFRM fold indices are absent from the feature cache: {missing[:5]}")
    return np.asarray([lookup[int(index)] for index in indices], dtype=np.int64)


def fit_target_scaler(
    target: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    values = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.ndim != 3 or values.shape != valid.shape:
        raise ValueError("EFRM target scaler received incompatible target/mask arrays")
    centers: list[float] = []
    scales: list[float] = []
    counts: list[int] = []
    for coordinate in range(values.shape[1]):
        selected = values[:, coordinate][valid[:, coordinate]]
        if not selected.size:
            raise RuntimeError("EFRM target-scaler coordinate has no valid training support")
        centers.append(float(selected.mean()))
        standard_deviation = float(selected.std())
        scales.append(standard_deviation if standard_deviation > 1e-6 else 1.0)
        counts.append(int(selected.size))
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
        counts,
    )


def scale_targets(
    target: np.ndarray,
    valid_mask: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    scaled = (target - center[None, :, None]) / scale[None, :, None]
    return np.where(valid_mask, scaled, 0.0).astype(np.float32)


class EFRMProbe(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        target_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(embedding_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(embedding_dim), int(output_dim) * int(target_length))
        self.output_dim = int(output_dim)
        self.target_length = int(target_length)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.head(self.dropout(self.norm(features)))
        if self.target_length > 1:
            output = output.reshape(-1, self.output_dim, self.target_length)
        return output


def evaluate_probe(
    probe: EFRMProbe,
    features: torch.Tensor,
    target_native: np.ndarray,
    target_valid: np.ndarray,
    *,
    task_type: str,
    names: Sequence[str],
    target_center: np.ndarray | None,
    target_scale: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    probe.eval()
    with torch.inference_mode():
        prediction = probe(features).float().cpu().numpy()
    if task_type == "classification":
        return classification_metrics(target_native, prediction, names), prediction
    if target_center is None or target_scale is None:
        raise RuntimeError("EFRM regression evaluation requires a train-only target scaler")
    prediction_native = (
        prediction * target_scale[None, :, None] + target_center[None, :, None]
    )
    native = regression_metrics(target_native, prediction_native, target_valid, names)
    scaled_target = scale_targets(target_native, target_valid, target_center, target_scale)
    valid_float = target_valid.astype(np.float64)
    scaled_rmse = float(
        np.sqrt(
            (np.square(prediction - scaled_target) * valid_float).sum()
            / max(1.0, valid_float.sum())
        )
    )
    return {"masked_rmse_scaled": scaled_rmse, **native}, prediction_native


def train_probe(
    *,
    train_features: np.ndarray,
    train_target_native: np.ndarray,
    train_target_valid: np.ndarray,
    validation_features: np.ndarray | None,
    validation_target_native: np.ndarray | None,
    validation_target_valid: np.ndarray | None,
    task_type: str,
    names: Sequence[str],
    output_dim: int,
    target_length: int,
    target_center: np.ndarray | None,
    target_scale: np.ndarray | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    minimum_learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device: torch.device,
) -> tuple[EFRMProbe, list[dict[str, Any]], dict[str, Any] | None]:
    set_seed(seed)
    probe = EFRMProbe(
        train_features.shape[1], output_dim, target_length, dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.95),
    )
    steps_per_epoch = math.ceil(len(train_features) / int(batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs) * steps_per_epoch),
        eta_min=float(minimum_learning_rate),
    )
    features = torch.from_numpy(np.asarray(train_features, dtype=np.float32)).to(device)
    valid = torch.from_numpy(np.asarray(train_target_valid, dtype=bool)).to(device)
    if task_type == "classification":
        target = torch.from_numpy(
            np.asarray(train_target_native, dtype=np.int64)
        ).to(device)
    else:
        if target_center is None or target_scale is None:
            raise RuntimeError("EFRM regression training requires a target scaler")
        target = torch.from_numpy(
            scale_targets(
                train_target_native, train_target_valid, target_center, target_scale
            )
        ).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for epoch in range(1, int(epochs) + 1):
        probe.train()
        permutation = torch.randperm(len(features), generator=generator)
        losses: list[float] = []
        for start in range(0, len(permutation), int(batch_size)):
            selected = permutation[start : start + int(batch_size)].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(features[selected])
            if task_type == "classification":
                loss = F.cross_entropy(prediction, target[selected])
            else:
                selected_valid = valid[selected].to(dtype=prediction.dtype)
                elementwise = F.smooth_l1_loss(
                    prediction, target[selected], reduction="none"
                )
                loss = (elementwise * selected_valid).sum() / selected_valid.sum().clamp_min(1)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("EFRM public probe produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if validation_features is not None:
            assert validation_target_native is not None
            assert validation_target_valid is not None
            metrics, prediction = evaluate_probe(
                probe,
                torch.from_numpy(
                    np.asarray(validation_features, dtype=np.float32)
                ).to(device),
                validation_target_native,
                validation_target_valid,
                task_type=task_type,
                names=names,
                target_center=target_center,
                target_scale=target_scale,
            )
            metric_name = "macro_f1" if task_type == "classification" else "masked_rmse_scaled"
            mode = "max" if task_type == "classification" else "min"
            value = float(metrics[metric_name])
            row[f"validation_{metric_name}"] = value
            improved = (
                best is None
                or (mode == "max" and value > float(best["metric"]))
                or (mode == "min" and value < float(best["metric"]))
            )
            if improved:
                best = {
                    "epoch": epoch,
                    "metric": value,
                    "metric_name": metric_name,
                    "mode": mode,
                    "state": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in probe.state_dict().items()
                    },
                    "metrics": metrics,
                    "prediction": prediction,
                }
        history.append(row)
    return probe, history, best


def baseline_metrics(
    *,
    task_type: str,
    train_target: np.ndarray,
    train_valid: np.ndarray,
    validation_target: np.ndarray,
    validation_valid: np.ndarray,
    names: Sequence[str],
) -> dict[str, Any]:
    if task_type == "classification":
        counts = np.bincount(train_target.astype(np.int64), minlength=len(names))
        majority = int(counts.argmax())
        logits = np.full((len(validation_target), len(names)), -1.0, dtype=np.float32)
        logits[:, majority] = 1.0
        return {
            "name": "outer_train_majority",
            "train_class_counts": counts.tolist(),
            "metrics": classification_metrics(validation_target, logits, names),
        }
    center, _scale, counts = fit_target_scaler(train_target, train_valid)
    prediction = np.broadcast_to(
        center[None, :, None], validation_target.shape
    ).copy()
    return {
        "name": "outer_train_coordinate_mean",
        "train_valid_value_count": counts,
        "metrics": regression_metrics(
            validation_target, prediction, validation_valid, names
        ),
    }


def smoke_subset(
    surface: PublicSurface,
    indices: Sequence[int],
    *,
    samples_per_class: int,
    regression_samples: int,
) -> tuple[int, ...]:
    if surface.dataset.spec.task_type == "regression":
        if len(indices) < regression_samples:
            raise RuntimeError("EFRM regression smoke partition is too small")
        return tuple(int(value) for value in indices[:regression_samples])
    grouped: dict[str, list[int]] = {
        name: [] for name in surface.dataset.spec.class_names
    }
    for index in indices:
        condition = str(surface.dataset.lightweight_metadata(int(index))["condition"])
        grouped[condition].append(int(index))
    selected: list[int] = []
    for name in surface.dataset.spec.class_names:
        if len(grouped[name]) < samples_per_class:
            raise RuntimeError(f"EFRM smoke class {name} is too small")
        selected.extend(grouped[name][:samples_per_class])
    return tuple(selected)


def select_and_refit(
    *,
    arrays: Mapping[str, np.ndarray],
    surface: PublicSurface,
    fold: PublicFold,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    config: Mapping[str, Any],
    config_path: Path,
    cache_identity: Mapping[str, Any],
    seed: int,
    device: torch.device,
    smoke: bool,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    train_rows = rows_for_indices(arrays, train_indices)
    validation_rows = rows_for_indices(arrays, validation_indices)
    train_features = arrays["features"][train_rows].astype(np.float32)
    validation_features = arrays["features"][validation_rows].astype(np.float32)
    train_target = arrays["targets"][train_rows]
    validation_target = arrays["targets"][validation_rows]
    train_valid = arrays["target_valid_mask"][train_rows].astype(bool)
    validation_valid = arrays["target_valid_mask"][validation_rows].astype(bool)
    spec = surface.dataset.spec
    names = spec.class_names if spec.task_type == "classification" else spec.target_names
    section = config["smoke"] if smoke else config["selection"]
    epochs = int(section["epoch_cap"])
    target_center: np.ndarray | None = None
    target_scale: np.ndarray | None = None
    target_counts: list[int] | None = None
    if spec.task_type == "regression":
        target_center, target_scale, target_counts = fit_target_scaler(
            train_target, train_valid
        )
    _selection_probe, selection_history, best = train_probe(
        train_features=train_features,
        train_target_native=train_target,
        train_target_valid=train_valid,
        validation_features=validation_features,
        validation_target_native=validation_target,
        validation_target_valid=validation_valid,
        task_type=spec.task_type,
        names=names,
        output_dim=spec.output_dim,
        target_length=spec.target_length,
        target_center=target_center,
        target_scale=target_scale,
        epochs=epochs,
        batch_size=int(config["selection"]["batch_size"]),
        learning_rate=float(config["selection"]["learning_rate"]),
        minimum_learning_rate=float(config["selection"]["minimum_learning_rate"]),
        weight_decay=float(config["selection"]["weight_decay"]),
        dropout=float(config["selection"]["dropout"]),
        seed=seed,
        device=device,
    )
    if best is None:
        raise RuntimeError("EFRM public epoch selection produced no checkpoint")

    refit_indices = tuple(dict.fromkeys([*train_indices, *validation_indices]))
    if set(refit_indices) != set(train_indices).union(validation_indices):
        raise RuntimeError("EFRM refit membership differs from train+validation")
    refit_rows = rows_for_indices(arrays, refit_indices)
    refit_target = arrays["targets"][refit_rows]
    refit_valid = arrays["target_valid_mask"][refit_rows].astype(bool)
    refit_center: np.ndarray | None = None
    refit_scale: np.ndarray | None = None
    refit_counts: list[int] | None = None
    if spec.task_type == "regression":
        refit_center, refit_scale, refit_counts = fit_target_scaler(
            refit_target, refit_valid
        )
    refit_probe, refit_history, _unused = train_probe(
        train_features=arrays["features"][refit_rows].astype(np.float32),
        train_target_native=refit_target,
        train_target_valid=refit_valid,
        validation_features=None,
        validation_target_native=None,
        validation_target_valid=None,
        task_type=spec.task_type,
        names=names,
        output_dim=spec.output_dim,
        target_length=spec.target_length,
        target_center=refit_center,
        target_scale=refit_scale,
        epochs=int(best["epoch"]),
        batch_size=int(config["selection"]["batch_size"]),
        learning_rate=float(config["selection"]["learning_rate"]),
        minimum_learning_rate=float(config["selection"]["minimum_learning_rate"]),
        weight_decay=float(config["selection"]["weight_decay"]),
        dropout=float(config["selection"]["dropout"]),
        seed=seed,
        device=device,
    )
    refit_probe.eval()
    refit_features = torch.from_numpy(
        arrays["features"][refit_rows].astype(np.float32)
    ).to(device)
    with torch.inference_mode():
        expected_refit_prediction = refit_probe(refit_features).float().cpu().numpy()
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "task": surface.task,
        "outer_fold": fold.outer_fold,
        "seed": int(seed),
        "task_type": spec.task_type,
        "names": list(names),
        "output_dim": spec.output_dim,
        "target_length": spec.target_length,
        "embedding_dim": int(arrays["features"].shape[1]),
        "dropout": float(config["selection"]["dropout"]),
        "probe_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in refit_probe.state_dict().items()
        },
        "target_center": (
            torch.from_numpy(refit_center) if refit_center is not None else None
        ),
        "target_scale": (
            torch.from_numpy(refit_scale) if refit_scale is not None else None
        ),
        "selected_epoch": int(best["epoch"]),
        "learning_rate": float(config["selection"]["learning_rate"]),
        "minimum_learning_rate": float(config["selection"]["minimum_learning_rate"]),
        "weight_decay": float(config["selection"]["weight_decay"]),
        "refit_dataset_indices": torch.tensor(refit_indices, dtype=torch.long),
        "runner_config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "feature_cache_identity_sha256": stable_hash(cache_identity),
        "target_dataset_exposure": False,
        "protected_test_opened": False,
    }
    checkpoint_path = output_dir / "checkpoint_public_refit.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reload_probe = EFRMProbe(
        int(reloaded["embedding_dim"]),
        int(reloaded["output_dim"]),
        int(reloaded["target_length"]),
        float(reloaded["dropout"]),
    ).to(device)
    reload_probe.load_state_dict(reloaded["probe_state"], strict=True)
    reload_probe.eval()
    with torch.inference_mode():
        reloaded_prediction = reload_probe(refit_features).float().cpu().numpy()
    if not np.allclose(
        reloaded_prediction, expected_refit_prediction, rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("weights-only reload changed EFRM public-refit predictions")

    report = {
        "selection_metric": str(best["metric_name"]),
        "selection_mode": str(best["mode"]),
        "selected_epoch": int(best["epoch"]),
        "validation_metrics": best["metrics"],
        "selection_history": selection_history,
        "selection_target_scaler": (
            {
                "fit_membership": "outer_train_only",
                "center": target_center.tolist(),
                "scale": target_scale.tolist(),
                "valid_value_count": target_counts,
            }
            if target_center is not None and target_scale is not None
            else None
        ),
        "baselines": baseline_metrics(
            task_type=spec.task_type,
            train_target=train_target,
            train_valid=train_valid,
            validation_target=validation_target,
            validation_valid=validation_valid,
            names=names,
        ),
        "public_refit": {
            "membership": (
                "smoke_train_plus_validation_subset"
                if smoke
                else "outer_train_plus_public_validation"
            ),
            "sample_count": len(refit_indices),
            "epochs": int(best["epoch"]),
            "history": refit_history,
            "target_scaler": (
                {
                    "fit_membership": "outer_train_plus_public_validation",
                    "center": refit_center.tolist(),
                    "scale": refit_scale.tolist(),
                    "valid_value_count": refit_counts,
                }
                if refit_center is not None and refit_scale is not None
                else None
            ),
            "checkpoint_path": portable_path(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "weights_only_reload_match": True,
        },
    }
    return report, np.asarray(best["prediction"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_path = load_config(args.config)
    matrix = config["job_matrix"]
    if args.task not in matrix["tasks"]:
        raise ValueError(f"task is outside the frozen EFRM v2 matrix: {args.task}")
    if int(args.outer_fold) not in matrix["outer_folds"]:
        raise ValueError("outer fold is outside the frozen EFRM v2 matrix")
    if int(args.seed) not in matrix["seeds"]:
        raise ValueError("seed is outside the frozen EFRM v2 matrix")
    output_dir = resolve_repo_path(args.output_dir)
    run_root = resolve_repo_path(config["resources"]["run_root"])
    try:
        output_dir.relative_to(run_root)
    except ValueError as exc:
        raise PermissionError(f"EFRM public output must remain under {run_root}") from exc
    if output_dir == run_root:
        raise ValueError("EFRM output directory must identify one job below the run root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"EFRM public run output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "status.json",
        {
            "schema": RUN_SCHEMA,
            "status": "initializing",
            "pid": os.getpid(),
            "started_at": utc_now(),
            "target_dataset_exposure": False,
            "protected_test_opened": False,
        },
    )
    started = time.perf_counter()
    surface = load_public_surface(config, task=args.task)
    fold = surface.folds[int(args.outer_fold)]
    checkpoint = frozen_checkpoint_identity(
        config, dataset_id=surface.dataset.spec.dataset_id
    )
    train_indices: Sequence[int] = fold.train_indices
    validation_indices: Sequence[int] = fold.validation_indices
    if args.smoke:
        train_indices = smoke_subset(
            surface,
            train_indices,
            samples_per_class=int(config["smoke"]["samples_per_class"]),
            regression_samples=int(config["smoke"]["regression_samples"]),
        )
        validation_indices = smoke_subset(
            surface,
            validation_indices,
            samples_per_class=int(config["smoke"]["samples_per_class"]),
            regression_samples=int(config["smoke"]["regression_samples"]),
        )
    device_name = str(config["resources"]["device"])
    if not torch.cuda.is_available() or not device_name.startswith("cuda:"):
        raise RuntimeError("EFRM v2 public development requires its frozen CUDA lane")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    set_seed(int(args.seed))
    with exclusive_gpu_lock(resolve_repo_path(config["resources"]["gpu_lock_path"])):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / 2**30
        if free_gib < float(config["resources"]["minimum_free_gpu_gib"]):
            raise RuntimeError(
                f"GPU has {free_gib:.2f} GiB free, below the EFRM v2 minimum"
            )
        torch.cuda.reset_peak_memory_stats(device)
        arrays, cache_identity, cache_hit, cache_path = extract_or_load_features(
            config=config,
            surface=surface,
            checkpoint=checkpoint,
            device=device,
        )
        probe_report, validation_prediction = select_and_refit(
            arrays=arrays,
            surface=surface,
            fold=fold,
            train_indices=train_indices,
            validation_indices=validation_indices,
            config=config,
            config_path=config_path,
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
        prediction=validation_prediction,
        target=arrays["targets"][validation_rows],
        target_valid_mask=arrays["target_valid_mask"][validation_rows],
        dataset_index=arrays["dataset_indices"][validation_rows],
        subject=arrays["subjects"][validation_rows],
        sample_id=arrays["sample_ids"][validation_rows],
    )
    write_json(output_dir / "public_selection_report.json", probe_report)
    manifest = {
        "schema": RUN_SCHEMA,
        "status": "completed",
        "mode": "smoke_only" if args.smoke else "public_selection_and_refit",
        "protocol_id": PROTOCOL_ID,
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
        "public_manifest_path": portable_path(fold.public_manifest_path),
        "public_manifest_sha256": fold.public_manifest_sha256,
        "checkpoint_identity": checkpoint,
        "feature_cache": {
            **cache_identity,
            "path": portable_path(cache_path),
            "file_sha256": sha256_file(cache_path),
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
        "target_dataset_exposure": False,
        "protected_test_opened": False,
        "completed_at": utc_now(),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "status.json", manifest)
    primary = (
        probe_report["validation_metrics"]["macro_f1"]
        if surface.dataset.spec.task_type == "classification"
        else probe_report["validation_metrics"]["ccc"]
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "mode": manifest["mode"],
                "task": args.task,
                "outer_fold": int(args.outer_fold),
                "seed": int(args.seed),
                "public_validation_primary": primary,
                "feature_cache_hit": cache_hit,
                "target_dataset_exposure": False,
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
    parser.add_argument("--task", required=True, choices=tuple(TASK_SPECS))
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
        status_path = output / "status.json"
        existing_status: dict[str, Any] = {}
        if status_path.is_file():
            try:
                existing_status = json.loads(status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_status = {"status": "unknown_existing"}
        if (
            not protected_path
            and not existing_manifest
            and existing_status.get("status") == "initializing"
        ):
            write_json(
                status_path,
                {
                    "schema": RUN_SCHEMA,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "target_dataset_exposure": False,
                    "protected_test_opened": False,
                    "failed_at": utc_now(),
                },
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
