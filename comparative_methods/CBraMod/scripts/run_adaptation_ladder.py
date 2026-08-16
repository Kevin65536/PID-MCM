#!/usr/bin/env python3
"""Run the pre-declared public-only CBraMod adaptation-capacity pilot.

The pilot is deliberately separate from the protected comparison runner.  It
uses one fixed public task/fold, canonical balanced subsets, a fixed epoch and
batch budget, and reports validation metrics without selecting a model from
them.  The supported capacities are:

* ``frozen_linear``: official pretrained latent cache + linear head;
* ``frozen_mlp``: official pretrained latent cache + one hidden layer;
* ``last_block_linear``: pretrained backbone with only the last criss-cross
  transformer block trainable;
* ``full_finetune_linear``: all pretrained backbone parameters trainable;
* ``random_linear``/``random_mlp``: the same architecture with random encoder
  initialization, frozen during head training.

This script is an analysis pilot, not table evidence.  Protected inputs and
outputs are refused by construction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
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


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.CBraMod.adapters.cbramod import (
    CBraModFrozenEncoder,
    _load_upstream_class,
    load_verified_cbramod_encoder,
)
from comparative_methods.CBraMod.alignment_data import (
    CBraModPublicView,
    RecordGroupedBatchSampler,
    SUPPORTED_TASKS,
    load_config as load_alignment_config,
    resolve_repo_path,
    stable_hash,
)
from comparative_methods.CBraMod.run_public_development_v2 import (
    PublicFold,
    balanced_subset,
    load_public_fold,
    rows_for_indices,
    standardizer,
    validate_feature_arrays,
)
from comparative_methods.audit_public_preflight import sha256_file
from efrm_pytorch.metrics import classification_metrics


CONFIG_SCHEMA = "cbramod_adaptation_ladder_pilot_v1"
RUN_SCHEMA = "cbramod_adaptation_ladder_run_v1"
FEATURE_SCHEMA = "cbramod_full_public_feature_cache_v2"
DEFAULT_CONFIG = METHOD_ROOT / "configs/adaptation_ladder_pilot.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "comparative_methods/runs/performance_analysis/20260816_p0/cbramod_ladder"
CAPACITIES = (
    "frozen_linear",
    "frozen_mlp",
    "last_block_linear",
    "full_finetune_linear",
    "random_linear",
    "random_mlp",
)


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
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


@contextmanager
def exclusive_gpu_lock(path: Path) -> Iterator[None]:
    """Serialize the analysis lane without changing any project lock state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"CBraMod adaptation GPU lane is locked: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_pilot_config(path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected {CONFIG_SCHEMA} config: {config_path}")
    if value.get("method_id") != "cbramod" or value.get("mode") != "public_development_only":
        raise PermissionError("adaptation ladder must remain public-only CBraMod work")
    if value.get("protected_test_default") != "locked":
        raise PermissionError("protected test must default to locked")
    task = str(value.get("task"))
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"adaptation ladder task is unsupported: {task}")
    capacities = tuple(str(item) for item in value["ladder"]["capacities"])
    if capacities != CAPACITIES:
        raise ValueError(f"capacity order must be the pre-declared order: {CAPACITIES}")
    alignment_path = resolve_repo_path(value["data"]["alignment_config"])
    alignment, resolved_alignment = load_alignment_config(alignment_path)
    if alignment.get("protected_test_default") != "locked":
        raise PermissionError("alignment config does not lock protected test")
    if alignment.get("mode") != "public_audit_only":
        raise PermissionError("adaptation ladder requires the public audit alignment profile")
    if int(value["data"]["patch_samples"]) != int(alignment["adapter"]["patch_samples"]):
        raise ValueError("pilot patch size differs from the audited adapter")
    if float(value["data"]["sampling_rate_hz"]) != float(alignment["data"]["eeg_sample_rate_hz"]):
        raise ValueError("pilot sampling rate differs from the audited adapter")
    if int(value["ladder"]["epochs"]) < 1 or int(value["ladder"]["batch_size"]) < 1:
        raise ValueError("pilot budget must contain positive epochs and batch size")
    return value, config_path, alignment, resolved_alignment


def output_guard(path: Path) -> Path:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"analysis output crosses protected boundary: {resolved}")
    return resolved


def cache_for_fold(
    *, fold: PublicFold, alignment: Mapping[str, Any], cache_root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path, Path]:
    """Resolve exactly one reviewed cache for the selected public fold."""

    candidates: list[tuple[Path, Path, dict[str, Any]]] = []
    task_root = cache_root / fold.inventory.task
    for manifest_path in sorted(task_root.glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        npz_path = manifest_path.with_suffix(".npz")
        if (
            manifest.get("schema") == FEATURE_SCHEMA
            and manifest.get("task") == fold.inventory.task
            and int(manifest.get("outer_fold", -1)) == fold.outer_fold
            and manifest.get("public_manifest_sha256") == fold.public_manifest_sha256
            and manifest.get("protected_test_opened") is False
            and npz_path.is_file()
        ):
            candidates.append((manifest_path, npz_path, manifest))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one public cache for {fold.inventory.task}/outer{fold.outer_fold}, "
            f"found {len(candidates)}"
        )
    manifest_path, cache_path, manifest = candidates[0]
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    validate_feature_arrays(arrays, fold)
    if str(manifest.get("feature_cache_key")) != cache_path.stem:
        raise RuntimeError("feature cache manifest key does not match its file name")
    return arrays, manifest, cache_path, manifest_path


def metadata_rows(fold: PublicFold, indices: Sequence[int]) -> list[dict[str, Any]]:
    return [fold.inventory.dataset.lightweight_metadata(int(index)) for index in indices]


def target_for_indices(fold: PublicFold, indices: Sequence[int]) -> np.ndarray:
    class_to_index = fold.inventory.dataset.class_to_index
    return np.asarray(
        [class_to_index[str(row["condition"])] for row in metadata_rows(fold, indices)],
        dtype=np.int64,
    )


def sample_membership(
    fold: PublicFold, train_per_class: int, validation_per_class: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    train = balanced_subset(fold, fold.train_indices, samples_per_class=int(train_per_class))
    validation = balanced_subset(
        fold, fold.validation_indices, samples_per_class=int(validation_per_class)
    )
    if set(train).intersection(validation):
        raise RuntimeError("pilot train and validation memberships overlap")
    if set(train) - set(fold.train_indices) or set(validation) - set(fold.validation_indices):
        raise RuntimeError("pilot membership escaped the reviewed public split")
    return tuple(train), tuple(validation)


class CBraModLatentClassifier(nn.Module):
    """Official latent pooling followed by a declared linear/MLP head."""

    def __init__(self, backbone: nn.Module, output_dim: int, head_kind: str, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        if head_kind == "linear":
            self.head = nn.Linear(200, int(output_dim))
        elif head_kind == "mlp":
            self.head = nn.Sequential(
                nn.Linear(200, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), int(output_dim))
            )
        else:
            raise ValueError(f"unknown head kind: {head_kind}")

    def latent(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[-1] % 200:
            raise ValueError(f"expected [B,C,T] with T divisible by 200, got {tuple(eeg.shape)}")
        patches = eeg.reshape(eeg.shape[0], eeg.shape[1], eeg.shape[-1] // 200, 200)
        tokens = self.backbone(patches)
        if tokens.ndim != 4 or tokens.shape[-1] != 200:
            raise RuntimeError(f"unexpected CBraMod latent shape: {tuple(tokens.shape)}")
        return tokens.mean(dim=(1, 2))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.head(self.latent(eeg))


def build_backbone(*, pretrained: bool, device: torch.device) -> nn.Module:
    if pretrained:
        backbone, _metadata = load_verified_cbramod_encoder(device=device)
        return backbone
    encoder_class = _load_upstream_class(METHOD_ROOT)
    backbone = encoder_class(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
    )
    backbone.proj_out = nn.Identity()
    return backbone.to(device)


def configure_capacity(
    model: CBraModLatentClassifier, capacity: str
) -> tuple[list[nn.Parameter], dict[str, Any]]:
    """Set trainability and deterministic eval/train boundaries for one capacity."""

    model.backbone.requires_grad_(False)
    model.head.requires_grad_(True)
    model.backbone.eval()
    head_kind = "mlp" if capacity.endswith("_mlp") else "linear"
    if capacity.startswith("last_block"):
        layers = getattr(getattr(model.backbone, "encoder", None), "layers", None)
        if layers is None or len(layers) < 1:
            raise RuntimeError("CBraMod backbone does not expose encoder.layers for last-block pilot")
        layers[-1].requires_grad_(True)
        layers[-1].train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        model.head.train()
        return trainable, {
            "backbone_initialization": "official_pretrained",
            "trainable_boundary": "encoder.layers[-1]_plus_head",
            "head": head_kind,
        }
    if capacity.startswith("full_finetune"):
        model.backbone.requires_grad_(True)
        model.backbone.train()
        model.head.train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return trainable, {
            "backbone_initialization": "official_pretrained",
            "trainable_boundary": "all_backbone_plus_head",
            "head": head_kind,
        }
    model.backbone.eval()
    model.head.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return trainable, {
        "backbone_initialization": (
            "official_pretrained" if not capacity.startswith("random") else "random_initialization"
        ),
        "trainable_boundary": "head_only",
        "head": head_kind,
    }


def make_loader(
    view: CBraModPublicView,
    indices: Sequence[int],
    *,
    batch_size: int,
    workers: int,
    seed: int,
    shuffle_groups: bool = True,
) -> DataLoader:
    sampler = RecordGroupedBatchSampler(
        view.dataset, indices, batch_size=int(batch_size), seed=int(seed) if shuffle_groups else 0
    )
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": int(workers),
        "pin_memory": True,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(view, **kwargs)


def encode_indices(
    *,
    backbone: nn.Module,
    view: CBraModPublicView,
    indices: Sequence[int],
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract random-init embeddings using only the selected public samples."""

    loader = make_loader(view, indices, batch_size=batch_size, workers=workers, seed=seed)
    embeddings: list[np.ndarray] = []
    dataset_indices: list[np.ndarray] = []
    sample_ids: list[str] = []
    subjects: list[str] = []
    backbone.eval()
    with torch.inference_mode():
        for batch in loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            patches = eeg.reshape(eeg.shape[0], eeg.shape[1], eeg.shape[-1] // 200, 200)
            tokens = backbone(patches)
            embedding = tokens.mean(dim=(1, 2))
            if embedding.shape != (len(eeg), 200) or not bool(torch.isfinite(embedding).all()):
                raise RuntimeError("random feature extraction returned invalid embeddings")
            embeddings.append(embedding.float().cpu().numpy())
            dataset_indices.append(batch["dataset_index"].cpu().numpy())
            sample_ids.extend(str(item) for item in batch["sample_id"])
            subjects.extend(str(item).split("|subject_", 1)[-1].split("|", 1)[0] for item in batch["sample_id"])
    features = np.concatenate(embeddings).astype(np.float32, copy=False)
    positions = np.concatenate(dataset_indices).astype(np.int64, copy=False)
    order = np.argsort(positions)
    return features[order], positions[order], np.asarray(sample_ids, dtype=str)[order], np.asarray(subjects, dtype=str)[order]


def prepare_cached_features(
    arrays: Mapping[str, np.ndarray],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_rows = rows_for_indices(arrays, train_indices)
    val_rows = rows_for_indices(arrays, validation_indices)
    train_x = arrays["features"][train_rows].astype(np.float32)
    val_x = arrays["features"][val_rows].astype(np.float32)
    mean, scale = standardizer(train_x, 1e-6)
    train_x = (train_x - mean) / scale
    val_x = (val_x - mean) / scale
    return train_x, val_x, arrays["targets"][train_rows].astype(np.int64), arrays["targets"][val_rows].astype(np.int64)


def class_counts(target: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(target.astype(np.int64), minlength=int(class_count))
    if bool((counts <= 0).any()):
        raise RuntimeError(f"pilot membership has an empty class: {counts.tolist()}")
    return counts


def train_feature_head(
    *,
    capacity: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    class_names: Sequence[str],
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    set_seed(seed)
    head_kind = "mlp" if capacity.endswith("_mlp") else "linear"
    model = CBraModLatentClassifier(nn.Identity(), len(class_names), head_kind, hidden_dim).to(device)
    # Identity backbone makes the cached feature route explicit.
    model.latent = lambda _eeg: _eeg  # type: ignore[method-assign]
    model.head.train()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    train_features = torch.from_numpy(train_x).to(device)
    val_features = torch.from_numpy(val_x).to(device)
    target = torch.from_numpy(train_y).to(device)
    weights = torch.from_numpy((len(train_y) / (len(class_names) * class_counts(train_y, len(class_names)))).astype(np.float32)).to(device)
    history: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(len(target), generator=generator)
        losses: list[float] = []
        for start in range(0, len(permutation), int(batch_size)):
            selected = permutation[start : start + int(batch_size)].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.head(train_features[selected])
            loss = torch.nn.functional.cross_entropy(logits, target[selected], weight=weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{capacity} produced a non-finite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        with torch.inference_mode():
            val_logits = model.head(val_features).float().cpu().numpy()
        metrics = classification_metrics(val_y, val_logits, class_names)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics})
    with torch.inference_mode():
        final_logits = model.head(val_features).float().cpu().numpy()
    return {"history": history, "final_validation_metrics": classification_metrics(val_y, final_logits, class_names)}, final_logits


def train_raw_capacity(
    *,
    capacity: str,
    fold: PublicFold,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    alignment: Mapping[str, Any],
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    workers: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    set_seed(seed)
    pretrained = not capacity.startswith("random")
    head_kind = "mlp" if capacity.endswith("_mlp") else "linear"
    backbone = build_backbone(pretrained=pretrained, device=device)
    model = CBraModLatentClassifier(backbone, len(fold.class_names), head_kind, hidden_dim).to(device)
    trainable, boundary = configure_capacity(model, capacity)
    if capacity.startswith("random") and not capacity.endswith("_linear") and not capacity.endswith("_mlp"):
        raise ValueError(f"unsupported random capacity: {capacity}")
    view = CBraModPublicView(
        fold.inventory, sample_rate_hz=float(alignment["data"]["eeg_sample_rate_hz"])
    )
    train_loader = make_loader(view, train_indices, batch_size=batch_size, workers=workers, seed=seed)
    val_loader = make_loader(view, validation_indices, batch_size=batch_size, workers=workers, seed=0, shuffle_groups=False)
    train_y = target_for_indices(fold, train_indices)
    val_y = target_for_indices(fold, validation_indices)
    weights = torch.from_numpy((len(train_y) / (len(fold.class_names) * class_counts(train_y, len(fold.class_names)))).astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=float(weight_decay))
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        # A random-init control is a frozen random representation plus a
        # trainable head.  Re-entering ``train`` above would otherwise turn on
        # dropout in the frozen backbone and make the representation change
        # between optimizer steps, which is not the intended control.
        if capacity.startswith("random_"):
            model.backbone.eval()
            model.head.train()
        if capacity.startswith("last_block"):
            model.backbone.patch_embedding.eval()
            for layer in model.backbone.encoder.layers[:-1]:
                layer.eval()
            model.backbone.encoder.layers[-1].train()
        losses: list[float] = []
        for batch in train_loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            y = torch.from_numpy(target_for_indices(fold, batch["dataset_index"].cpu().numpy())).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(eeg)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{capacity} produced a non-finite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_logits_parts: list[np.ndarray] = []
        val_target_parts: list[np.ndarray] = []
        val_index_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in val_loader:
                eeg = batch["eeg"].to(device, non_blocking=True)
                logits = model(eeg).float().cpu().numpy()
                val_logits_parts.append(logits)
                batch_indices = batch["dataset_index"].cpu().numpy().astype(np.int64)
                val_target_parts.append(target_for_indices(fold, batch_indices))
                val_index_parts.append(batch_indices)
        val_logits = np.concatenate(val_logits_parts)
        val_targets = np.concatenate(val_target_parts)
        val_indices = np.concatenate(val_index_parts)
        order = np.argsort(val_indices)
        val_logits = val_logits[order]
        val_targets = val_targets[order]
        metrics = classification_metrics(val_targets, val_logits, fold.class_names)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics})
        print(f"[{capacity}] epoch {epoch}/{epochs} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
    final_metrics = classification_metrics(val_targets, val_logits, fold.class_names)
    return {
        "history": history,
        "final_validation_metrics": final_metrics,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
        "trainable_tensor_count": int(len(trainable)),
        "boundary": boundary,
        "wall_seconds": time.perf_counter() - start_time,
    }, val_logits, val_targets


def dry_run_report(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    alignment_path: Path,
    fold: PublicFold,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    cache_manifest: Mapping[str, Any],
    cache_path: Path,
    cache_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_files = {
        "upstream_model": METHOD_ROOT / "upstream/models/cbramod.py",
        "upstream_transformer": METHOD_ROOT / "upstream/models/criss_cross_transformer.py",
        "official_finetune_main": METHOD_ROOT / "upstream/finetune_main.py",
        "official_finetune_trainer": METHOD_ROOT / "upstream/finetune_trainer.py",
        "official_downstream_wrapper": METHOD_ROOT / "upstream/models/model_for_physio.py",
        "adapter": METHOD_ROOT / "adapters/cbramod.py",
    }
    missing_source = [portable_path(path) for path in source_files.values() if not path.is_file()]
    if missing_source:
        raise FileNotFoundError(f"CBraMod source audit files are missing: {missing_source}")
    return {
        "schema": "cbramod_adaptation_ladder_dry_run_v1",
        "status": "pass",
        "mode": "dry_run",
        "config_path": portable_path(config_path),
        "config_sha256": sha256_file(config_path),
        "alignment_config_path": portable_path(alignment_path),
        "alignment_config_sha256": sha256_file(alignment_path),
        "task": fold.inventory.task,
        "outer_fold": fold.outer_fold,
        "public_manifest_sha256": fold.public_manifest_sha256,
        "train_sample_count": len(train_indices),
        "validation_sample_count": len(validation_indices),
        "train_indices_sha256": stable_hash(list(train_indices)),
        "validation_indices_sha256": stable_hash(list(validation_indices)),
        "feature_cache": {
            "manifest_path": portable_path(cache_manifest_path),
            "manifest_sha256": sha256_file(cache_manifest_path),
            "cache_path": portable_path(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "feature_cache_key": cache_manifest["feature_cache_key"],
            "schema": cache_manifest["schema"],
            "protected_test_opened": cache_manifest["protected_test_opened"],
        },
        "source_audit": {
            "files": {
                name: {
                    "path": portable_path(path),
                    "sha256": sha256_file(path),
                }
                for name, path in source_files.items()
            },
            "official_finetune_defaults_audited": {
                "optimizer": "AdamW",
                "learning_rate": 1.0e-4,
                "weight_decay": 5.0e-2,
                "clip_value": 1.0,
                "multi_lr": True,
                "frozen_default": False,
                "note": "values are the pinned upstream finetune_main.py defaults; pilot budget is separately pre-declared",
            },
            "representation_boundary": "backbone.proj_out=Identity_then_mean(channel,patch)_latent_tokens",
        },
        "capacities": list(config["ladder"]["capacities"]),
        "fixed_budget": {
            "epochs": int(config["ladder"]["epochs"]),
            "batch_size": int(config["ladder"]["batch_size"]),
            "learning_rate": float(config["ladder"]["learning_rate"]),
            "weight_decay": float(config["ladder"]["weight_decay"]),
            "validation_used_for_selection": False,
        },
        "output_dir": portable_path(output_dir),
        "protected_test_opened": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_path, alignment, alignment_path = load_pilot_config(args.config)
    output_dir = output_guard(Path(args.output_dir) if args.output_dir else Path(config["output"]["default_root"]))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"adaptation ladder output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fold = load_public_fold(alignment, task=str(config["task"]), outer_fold=int(config["outer_fold"]))
    train_indices, validation_indices = sample_membership(
        fold,
        int(config["data"]["train_samples_per_class"]),
        int(config["data"]["validation_samples_per_class"]),
    )
    cache_root = resolve_repo_path(config["data"]["feature_cache_root"])
    arrays, cache_manifest, cache_path, cache_manifest_path = cache_for_fold(
        fold=fold, alignment=alignment, cache_root=cache_root
    )
    report = dry_run_report(
        config=config,
        config_path=config_path,
        alignment_path=alignment_path,
        fold=fold,
        train_indices=train_indices,
        validation_indices=validation_indices,
        cache_manifest=cache_manifest,
        cache_path=cache_path,
        cache_manifest_path=cache_manifest_path,
        output_dir=output_dir,
    )
    write_json(output_dir / "dry_run_manifest.json", report)
    if args.dry_run:
        return report
    device_name = str(args.device or config["resources"]["device"])
    if not torch.cuda.is_available() or not device_name.startswith("cuda:"):
        capability = {
            **report,
            "status": "blocked_capability",
            "blocker": "CUDA device is required for raw last-block/full-finetune capacities",
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "requested_device": device_name,
            "protected_test_opened": False,
        }
        write_json(output_dir / "capability_report.json", capability)
        return capability
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    set_seed(int(config["seed"]))
    lock_path = Path(str(config["resources"]["gpu_lock_path"]))
    started = time.perf_counter()
    ladder_rows: list[dict[str, Any]] = []
    try:
        with exclusive_gpu_lock(lock_path):
            for capacity_index, capacity in enumerate(config["ladder"]["capacities"]):
                capacity = str(capacity)
                print(f"[pilot] start {capacity}", flush=True)
                capacity_seed = int(config["seed"]) + capacity_index
                # Keep the random-init representation identical for the
                # linear-vs-MLP comparison; only the head capacity changes.
                if capacity.startswith("random_"):
                    capacity_seed = int(config["seed"]) + 100
                if capacity in {"frozen_linear", "frozen_mlp"}:
                    train_x, val_x, train_y, val_y = prepare_cached_features(
                        arrays, train_indices, validation_indices
                    )
                    capacity_report, logits = train_feature_head(
                        capacity=capacity,
                        train_x=train_x,
                        val_x=val_x,
                        train_y=train_y,
                        val_y=val_y,
                        class_names=fold.class_names,
                        hidden_dim=int(config["ladder"]["mlp_hidden_dim"]),
                        epochs=int(config["ladder"]["epochs"]),
                        batch_size=int(config["ladder"]["batch_size"]),
                        learning_rate=float(config["ladder"]["learning_rate"]),
                        weight_decay=float(config["ladder"]["weight_decay"]),
                        seed=capacity_seed,
                        device=device,
                    )
                    targets = val_y
                else:
                    capacity_report, logits, targets = train_raw_capacity(
                        capacity=capacity,
                        fold=fold,
                        train_indices=train_indices,
                        validation_indices=validation_indices,
                        alignment=alignment,
                        hidden_dim=int(config["ladder"]["mlp_hidden_dim"]),
                        epochs=int(config["ladder"]["epochs"]),
                        batch_size=int(config["ladder"]["batch_size"]),
                        learning_rate=float(config["ladder"]["learning_rate"]),
                        weight_decay=float(config["ladder"]["weight_decay"]),
                        workers=int(config["resources"]["data_loader_workers"]),
                        seed=capacity_seed,
                        device=device,
                    )
                prediction_path = output_dir / f"{capacity}_validation_predictions.npz"
                save_npz(
                    prediction_path,
                    logits=np.asarray(logits, dtype=np.float32),
                    target=np.asarray(targets, dtype=np.int64),
                )
                ladder_rows.append(
                    {
                        "capacity": capacity,
                        "capacity_index": capacity_index,
                        "seed": capacity_seed,
                        "validation_metrics": capacity_report["final_validation_metrics"],
                        "report": capacity_report,
                        "prediction_path": portable_path(prediction_path),
                        "prediction_sha256": sha256_file(prediction_path),
                    }
                )
                write_json(output_dir / "partial_report.json", {**report, "results": ladder_rows})
    except Exception as error:
        capability = {
            **report,
            "status": "blocked_or_failed",
            "blocker": type(error).__name__,
            "error": str(error),
            "completed_capacities": [row["capacity"] for row in ladder_rows],
            "results": ladder_rows,
            "wall_seconds": time.perf_counter() - started,
            "protected_test_opened": False,
        }
        write_json(output_dir / "capability_report.json", capability)
        return capability
    final = {
        **report,
        "schema": RUN_SCHEMA,
        "status": "completed",
        "mode": "public_pilot",
        "results": ladder_rows,
        "wall_seconds": time.perf_counter() - started,
        "device": device_name,
        "protected_test_opened": False,
        "claim_boundary": "public_development_pilot_not_protected_or_final_table_evidence",
        "completed_at": utc_now(),
    }
    write_json(output_dir / "report.json", final)
    write_json(output_dir / "status.json", final)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as error:
        output = output_guard(Path(args.output_dir))
        report = {
            "schema": "cbramod_adaptation_ladder_capability_report_v1",
            "status": "blocked_or_failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "protected_test_opened": False,
            "created_at": utc_now(),
        }
        write_json(output / "capability_report.json", report)
        print(json.dumps(jsonable(report), indent=2, sort_keys=True), flush=True)
        return 2
    print(json.dumps(jsonable(result), indent=2, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"pass", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
