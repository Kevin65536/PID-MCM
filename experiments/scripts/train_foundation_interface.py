#!/usr/bin/env python
"""Train downstream foundation-style models on exported source/observation tokens."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import require_standard_training_launcher


BRANCHES = (
    "eeg_source_tokens",
    "fnirs_source_tokens",
    "eeg_observation_tokens",
    "fnirs_observation_tokens",
)

BRANCH_MODALITY = torch.tensor([0, 1, 0, 1], dtype=torch.long)

TASKS: dict[str, dict[str, Any]] = {
    "croce_label_6way": {
        "source_task": None,
        "label_field": "label_name",
        "label_names": ("BL", "MA", "LMI", "RMI", "nback", "wg"),
    },
    "cognitive_task_type_5way": {
        "source_task": "cognitive",
        "label_field": "task_type_label_name",
        "label_names": ("0-back", "2-back", "3-back", "BL", "WG"),
    },
    "nback_load_0_vs_2_vs_3": {
        "source_task": "cognitive",
        "label_field": "task_type_label_name",
        "label_names": ("0-back", "2-back", "3-back"),
    },
    "nback_load_0_vs_2_3": {
        "source_task": "cognitive",
        "label_field": "task_type_label_name",
        "label_names": ("0-back", "2/3-back"),
        "label_aliases": {"2-back": "2/3-back", "3-back": "2/3-back"},
    },
    "wg_bl_vs_wg": {
        "source_task": "cognitive",
        "label_field": "task_type_label_name",
        "label_names": ("BL", "WG"),
    },
    "nback_vs_wg": {
        "source_task": None,
        "label_field": "label_name",
        "label_names": ("nback", "wg"),
    },
    "mental_arithmetic_bl_vs_ma": {
        "source_task": "mental_arithmetic",
        "label_field": "label_name",
        "label_names": ("BL", "MA"),
    },
    "motor_lmi_vs_rmi": {
        "source_task": "motor_imagery",
        "label_field": "label_name",
        "label_names": ("LMI", "RMI"),
    },
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("experiment", {}).get("device", config.get("device", "auto")))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{now_stamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_split(token_run_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = token_run_dir / "tokens" / f"{split}_tokens.npz"
    if not path.exists():
        raise FileNotFoundError(f"Token split not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def available_splits(token_run_dir: Path) -> list[str]:
    token_dir = token_run_dir / "tokens"
    preferred = {"train": 0, "val": 1, "test": 2}
    splits = [path.name[: -len("_tokens.npz")] for path in sorted(token_dir.glob("*_tokens.npz"))]
    return sorted(splits, key=lambda item: (preferred.get(item, 99), item))


def vocab_sizes(token_run_dir: Path, splits: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, int]:
    manifest_path = token_run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    semantics = manifest.get("token_semantics", {}) if isinstance(manifest.get("token_semantics"), dict) else {}
    sizes: dict[str, int] = {}
    for branch in BRANCHES:
        key = branch.replace("_tokens", "_vocab_size")
        if key in semantics:
            sizes[branch] = int(semantics[key])
        else:
            sizes[branch] = max(int(np.max(split[branch])) + 1 for split in splits.values())
    return sizes


def task_mask(data: Mapping[str, np.ndarray], task_name: str | None) -> np.ndarray:
    n_samples = int(np.asarray(data[BRANCHES[0]]).shape[0])
    if task_name is None:
        return np.ones(n_samples, dtype=bool)
    spec = TASKS[task_name]
    label_field = str(spec.get("label_field", "label_name"))
    if label_field not in data:
        return np.zeros(n_samples, dtype=bool)
    labels = np.asarray(data[label_field]).astype(str)
    label_aliases = {str(key): str(value) for key, value in spec.get("label_aliases", {}).items()}
    accepted_labels = set(str(label) for label in spec["label_names"]) | set(label_aliases)
    mask = np.isin(labels, np.asarray(sorted(accepted_labels), dtype=str))
    source_task = spec.get("source_task")
    if source_task is not None:
        mask &= np.asarray(data["source_task"]).astype(str) == str(source_task)
    return mask


def encode_labels(data: Mapping[str, np.ndarray], task_name: str, mask: np.ndarray) -> np.ndarray:
    spec = TASKS[task_name]
    labels = np.asarray(data[str(spec.get("label_field", "label_name"))]).astype(str)[mask]
    label_aliases = {str(key): str(value) for key, value in spec.get("label_aliases", {}).items()}
    labels = np.asarray([label_aliases.get(str(label), str(label)) for label in labels], dtype=str)
    class_to_id = {label: index for index, label in enumerate(spec["label_names"])}
    return np.asarray([class_to_id[label] for label in labels], dtype=np.int64)


class TokenSplitDataset(Dataset):
    def __init__(self, data: Mapping[str, np.ndarray], *, task_name: str | None = None):
        self.task_name = task_name
        mask = task_mask(data, task_name)
        self.indices = np.flatnonzero(mask).astype(np.int64)
        if self.indices.size == 0:
            raise ValueError(f"No samples available for task={task_name!r}")
        branch_arrays = [np.asarray(data[branch], dtype=np.int64)[self.indices] for branch in BRANCHES]
        self.tokens = np.stack(branch_arrays, axis=1)
        self.subject_ids = np.asarray(data.get("subject_id", np.full(len(mask), -1)), dtype=np.int64)[self.indices]
        if task_name is None:
            self.labels = np.full(self.indices.shape[0], -1, dtype=np.int64)
            self.classes: list[str] = []
        else:
            self.labels = encode_labels(data, task_name, mask)
            self.classes = list(TASKS[task_name]["label_names"])

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.as_tensor(self.tokens[index], dtype=torch.long),
            "label": torch.as_tensor(self.labels[index], dtype=torch.long),
            "subject_id": torch.as_tensor(self.subject_ids[index], dtype=torch.long),
        }


@dataclass(frozen=True)
class ModelSpec:
    vocab_sizes: dict[str, int]
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    max_branch_len: int


class SourceObservationTokenTransformer(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.branches = BRANCHES
        self.vocab_sizes = dict(spec.vocab_sizes)
        self.max_branch_len = int(spec.max_branch_len)
        hidden_dim = int(spec.hidden_dim)
        self.token_embeddings = nn.ModuleDict(
            {branch: nn.Embedding(int(self.vocab_sizes[branch]) + 2, hidden_dim, padding_idx=0) for branch in BRANCHES}
        )
        self.branch_embedding = nn.Embedding(len(BRANCHES), hidden_dim)
        self.modality_embedding = nn.Embedding(2, hidden_dim)
        self.position_embedding = nn.Embedding(self.max_branch_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(spec.num_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=float(spec.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(spec.num_layers))
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlm_heads = nn.ModuleDict(
            {branch: nn.Linear(hidden_dim, int(self.vocab_sizes[branch])) for branch in BRANCHES}
        )
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, n_branches, seq_len = tokens.shape
        if n_branches != len(BRANCHES):
            raise ValueError(f"Expected {len(BRANCHES)} branches, got {n_branches}")
        if seq_len > self.max_branch_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_branch_len={self.max_branch_len}")
        position_ids = torch.arange(seq_len, device=tokens.device)
        branch_ids = torch.arange(n_branches, device=tokens.device)
        modality_ids = BRANCH_MODALITY.to(tokens.device)
        branch_states = []
        for index, branch in enumerate(BRANCHES):
            states = self.token_embeddings[branch](tokens[:, index, :])
            states = states + self.position_embedding(position_ids)[None, :, :]
            states = states + self.branch_embedding(branch_ids[index]).view(1, 1, -1)
            states = states + self.modality_embedding(modality_ids[index]).view(1, 1, -1)
            branch_states.append(states)
        sequence = torch.cat(branch_states, dim=1)
        sequence = self.encoder(sequence)
        sequence = self.norm(sequence)
        return sequence.view(batch_size, n_branches, seq_len, -1)

    def pooled(self, tokens: torch.Tensor) -> torch.Tensor:
        states = self.encode(tokens)
        return states.mean(dim=(1, 2))

    def mlm_logits(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        states = self.encode(tokens)
        return [self.mlm_heads[branch](states[:, index, :, :]) for index, branch in enumerate(BRANCHES)]

    def contrastive_loss(self, tokens: torch.Tensor, temperature: float) -> torch.Tensor:
        states = self.encode(tokens)
        eeg_state = states[:, [0, 2], :, :].mean(dim=(1, 2))
        fnirs_state = states[:, [1, 3], :, :].mean(dim=(1, 2))
        eeg_state = nn.functional.normalize(self.contrastive_projection(eeg_state), dim=-1)
        fnirs_state = nn.functional.normalize(self.contrastive_projection(fnirs_state), dim=-1)
        logits = eeg_state @ fnirs_state.T / max(float(temperature), 1e-6)
        labels = torch.arange(tokens.shape[0], device=tokens.device)
        return 0.5 * (
            nn.functional.cross_entropy(logits, labels)
            + nn.functional.cross_entropy(logits.T, labels)
        )


class ClassificationModel(nn.Module):
    def __init__(self, backbone: SourceObservationTokenTransformer, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        hidden_dim = next(backbone.parameters()).shape[-1]
        self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(hidden_dim, int(num_classes)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.pooled(tokens))


def offset_tokens(raw_tokens: torch.Tensor) -> torch.Tensor:
    return raw_tokens + 2


def make_span_mask(
    tokens: torch.Tensor,
    *,
    mask_ratio: float,
    min_span: int,
    max_span: int,
) -> torch.Tensor:
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    _, n_branches, seq_len = tokens.shape
    target = max(1, int(round(seq_len * float(mask_ratio))))
    for sample in range(tokens.shape[0]):
        for branch in range(n_branches):
            covered = 0
            guard = 0
            while covered < target and guard < seq_len * 4:
                span = random.randint(int(min_span), int(max_span))
                start = random.randint(0, max(seq_len - 1, 0))
                end = min(seq_len, start + span)
                before = int(mask[sample, branch].sum().item())
                mask[sample, branch, start:end] = True
                after = int(mask[sample, branch].sum().item())
                covered += after - before
                guard += 1
    return mask


def data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(num_workers) > 0,
    )


def save_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def batch_limit(config: Mapping[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value in (None, "null", "None"):
        return None
    return int(value)


def run_mlm_epoch(
    model: SourceObservationTokenTransformer,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: Mapping[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_mlm = 0.0
    total_contrastive = 0.0
    total_batches = 0
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            raw_tokens = batch["tokens"].to(device, non_blocking=True)
            inputs = offset_tokens(raw_tokens)
            targets = raw_tokens.clone()
            mask = make_span_mask(
                raw_tokens,
                mask_ratio=float(config.get("span_mask_ratio", 0.25)),
                min_span=int(config.get("min_span", 2)),
                max_span=int(config.get("max_span", 4)),
            ).to(device)
            inputs = inputs.masked_fill(mask, 1)
            targets = targets.masked_fill(~mask, -100)
            logits = model.mlm_logits(inputs)
            mlm_loss = sum(
                ce(branch_logits.reshape(-1, branch_logits.shape[-1]), targets[:, index, :].reshape(-1))
                for index, branch_logits in enumerate(logits)
            ) / len(logits)
            contrastive_weight = float(config.get("contrastive_weight", 0.0))
            if contrastive_weight > 0:
                contrastive = model.contrastive_loss(offset_tokens(raw_tokens), float(config.get("temperature", 0.07)))
            else:
                contrastive = torch.zeros((), device=device)
            loss = mlm_loss + contrastive_weight * contrastive
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = float(config.get("grad_clip_norm", 1.0))
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += float(loss.detach().item())
            total_mlm += float(mlm_loss.detach().item())
            total_contrastive += float(contrastive.detach().item())
            total_batches += 1
    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "mlm_loss": total_mlm / denom,
        "contrastive_loss": total_contrastive / denom,
        "batches": float(total_batches),
    }


def evaluate_classifier(
    model: ClassificationModel,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            raw_tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(offset_tokens(raw_tokens))
            loss = ce(logits, labels)
            pred = logits.argmax(dim=-1)
            losses.append(float(loss.detach().item()))
            targets.append(labels.detach().cpu().numpy())
            predictions.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(targets) if targets else np.asarray([], dtype=np.int64)
    y_pred = np.concatenate(predictions) if predictions else np.asarray([], dtype=np.int64)
    if y_true.size == 0:
        return {"loss": math.nan, "balanced_accuracy": math.nan, "macro_f1": math.nan}, y_true, y_pred
    metrics = {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics, y_true, y_pred


def run_classifier_epoch(
    model: ClassificationModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None,
    grad_clip_norm: float,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss(weight=class_weights)
    losses = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        raw_tokens = batch["tokens"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(offset_tokens(raw_tokens))
        loss = ce(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return {"loss": float(np.mean(losses)) if losses else math.nan, "batches": float(len(losses))}


def build_backbone(config: Mapping[str, Any], sizes: Mapping[str, int], max_branch_len: int) -> SourceObservationTokenTransformer:
    model_cfg = config.get("model", {})
    spec = ModelSpec(
        vocab_sizes={key: int(value) for key, value in sizes.items()},
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_branch_len=int(model_cfg.get("max_branch_len", max_branch_len)),
    )
    return SourceObservationTokenTransformer(spec)


def optimizer_for(parameters: Any, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = config.get("optimization", {})
    return torch.optim.AdamW(
        parameters,
        lr=float(opt_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
    )


def run_pretraining(
    config: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, np.ndarray]],
    sizes: Mapping[str, int],
    max_branch_len: int,
    run_dir: Path,
    device: torch.device,
    log: RunLogger,
) -> Path | None:
    pre_cfg = config.get("pretraining", {})
    if not bool(pre_cfg.get("enabled", True)):
        return None
    output_dir = run_dir / "mlm_pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = TokenSplitDataset(splits["train"], task_name=None)
    val_set = TokenSplitDataset(splits["val"], task_name=None)
    opt_cfg = config.get("optimization", {})
    train_loader = data_loader(
        train_set,
        batch_size=int(opt_cfg.get("batch_size", 512)),
        shuffle=True,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        device=device,
    )
    val_loader = data_loader(
        val_set,
        batch_size=int(opt_cfg.get("eval_batch_size", opt_cfg.get("batch_size", 512))),
        shuffle=False,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        device=device,
    )
    model = build_backbone(config, sizes, max_branch_len).to(device)
    optimizer = optimizer_for(model.parameters(), config)
    rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = output_dir / "checkpoints" / "best_model.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    patience = int(pre_cfg.get("patience", 5))
    stale_epochs = 0
    epochs = int(pre_cfg.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        train_metrics = run_mlm_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            config={**pre_cfg, **config.get("optimization", {})},
            max_batches=batch_limit(pre_cfg, "max_train_batches"),
        )
        val_metrics = run_mlm_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            config={**pre_cfg, **config.get("optimization", {})},
            max_batches=batch_limit(pre_cfg, "max_eval_batches"),
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)
        save_rows_csv(output_dir / "metrics.csv", rows)
        write_json(output_dir / "metrics.json", {"rows": rows, "best_val_loss": best_val})
        log(
            "pretrain epoch "
            f"{epoch}/{epochs} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_sizes": dict(sizes),
                    "max_branch_len": int(max_branch_len),
                    "config": dict(config),
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                log(f"pretrain early stopping after epoch {epoch}")
                break
    write_json(output_dir / "summary.json", {"best_checkpoint": str(best_path), "best_val_loss": best_val, "rows": rows})
    return best_path


def class_weights_for(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_one_task(
    *,
    task_name: str,
    experiment_name: str,
    pretrained_path: Path | None,
    config: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, np.ndarray]],
    sizes: Mapping[str, int],
    max_branch_len: int,
    run_dir: Path,
    device: torch.device,
    log: RunLogger,
) -> dict[str, Any]:
    fine_cfg = config.get("finetuning", {})
    output_dir = run_dir / experiment_name / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = TokenSplitDataset(splits["train"], task_name=task_name)
    val_set = TokenSplitDataset(splits["val"], task_name=task_name)
    test_set = TokenSplitDataset(splits["test"], task_name=task_name)
    opt_cfg = config.get("optimization", {})
    train_loader = data_loader(
        train_set,
        batch_size=int(opt_cfg.get("batch_size", 512)),
        shuffle=True,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        device=device,
    )
    val_loader = data_loader(
        val_set,
        batch_size=int(opt_cfg.get("eval_batch_size", opt_cfg.get("batch_size", 512))),
        shuffle=False,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        device=device,
    )
    test_loader = data_loader(
        test_set,
        batch_size=int(opt_cfg.get("eval_batch_size", opt_cfg.get("batch_size", 512))),
        shuffle=False,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        device=device,
    )
    backbone = build_backbone(config, sizes, max_branch_len)
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        backbone.load_state_dict(checkpoint["model_state_dict"])
    model = ClassificationModel(
        backbone,
        num_classes=len(train_set.classes),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(device)
    optimizer = optimizer_for(model.parameters(), config)
    class_weights = None
    if bool(fine_cfg.get("class_weighted_loss", True)):
        class_weights = class_weights_for(train_set.labels, len(train_set.classes), device)
    rows: list[dict[str, Any]] = []
    best_val = -float("inf")
    best_epoch = 0
    best_path = output_dir / "checkpoints" / "best_model.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    stale_epochs = 0
    patience = int(fine_cfg.get("patience", 8))
    epochs = int(fine_cfg.get("epochs", 30))
    for epoch in range(1, epochs + 1):
        train_metrics = run_classifier_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            class_weights=class_weights,
            grad_clip_norm=float(config.get("optimization", {}).get("grad_clip_norm", 1.0)),
            max_batches=batch_limit(fine_cfg, "max_train_batches"),
        )
        val_metrics, _, _ = evaluate_classifier(
            model,
            val_loader,
            device=device,
            max_batches=batch_limit(fine_cfg, "max_eval_batches"),
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)
        save_rows_csv(output_dir / "metrics.csv", rows)
        log(
            f"{experiment_name}/{task_name} epoch {epoch}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["balanced_accuracy"] > best_val:
            best_val = float(val_metrics["balanced_accuracy"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "task_name": task_name,
                    "classes": train_set.classes,
                    "epoch": epoch,
                    "val_balanced_accuracy": best_val,
                    "config": dict(config),
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                log(f"{experiment_name}/{task_name} early stopping after epoch {epoch}")
                break
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_true, val_pred = evaluate_classifier(model, val_loader, device=device, max_batches=None)
    test_metrics, test_true, test_pred = evaluate_classifier(model, test_loader, device=device, max_batches=None)
    cm_payload = {
        "classes": train_set.classes,
        "val": confusion_matrix(val_true, val_pred, labels=list(range(len(train_set.classes)))).tolist(),
        "test": confusion_matrix(test_true, test_pred, labels=list(range(len(train_set.classes)))).tolist(),
    }
    write_json(output_dir / "confusion_matrix.json", cm_payload)
    summary = {
        "experiment": experiment_name,
        "task": task_name,
        "classes": train_set.classes,
        "train_samples": int(len(train_set)),
        "val_samples": int(len(val_set)),
        "test_samples": int(len(test_set)),
        "train_subjects": int(np.unique(train_set.subject_ids).size),
        "val_subjects": int(np.unique(val_set.subject_ids).size),
        "test_subjects": int(np.unique(test_set.subject_ids).size),
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "pretrained_checkpoint": str(pretrained_path) if pretrained_path is not None else None,
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def load_linear_probe_baseline(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any] | None:
    baseline_cfg = config.get("linear_probe_baseline", {})
    summary_path = baseline_cfg.get("summary_path")
    if not summary_path:
        return None
    path = Path(str(summary_path))
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    output_path = run_dir / "linear_probe_baseline" / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output_path)
    return {"source_summary_path": str(path), "copied_summary_path": str(output_path), "rows": payload.get("rows", [])}


def dry_run(
    config: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, np.ndarray]],
    sizes: Mapping[str, int],
    max_branch_len: int,
    run_dir: Path,
    device: torch.device,
    log: RunLogger,
) -> None:
    task_names = list(config.get("finetuning", {}).get("tasks", []))
    shape_rows = []
    for split in ("train", "val", "test"):
        dataset = TokenSplitDataset(splits[split], task_name=None)
        batch = next(iter(data_loader(dataset, batch_size=4, shuffle=False, num_workers=0, device=device)))
        shape_rows.append(
            {
                "split": split,
                "samples": int(len(dataset)),
                "tokens_shape": list(batch["tokens"].shape),
                "subject_ids": [int(value) for value in batch["subject_id"].tolist()],
            }
        )
    task_rows = []
    for task_name in task_names:
        dataset = TokenSplitDataset(splits["train"], task_name=task_name)
        task_rows.append(
            {
                "task": task_name,
                "classes": dataset.classes,
                "train_samples": int(len(dataset)),
                "class_counts": {
                    dataset.classes[index]: int(count)
                    for index, count in enumerate(np.bincount(dataset.labels, minlength=len(dataset.classes)))
                },
            }
        )
    payload = {
        "vocab_sizes": dict(sizes),
        "max_branch_len": int(max_branch_len),
        "branch_order": list(BRANCHES),
        "split_batches": shape_rows,
        "tasks": task_rows,
    }
    write_json(run_dir / "dry_run_summary.json", payload)
    log(f"dry-run complete: {run_dir / 'dry_run_summary.json'}")


def run_experiment(config_path: Path, *, dry_run_only: bool) -> None:
    config = load_config(config_path)
    seed = int(config.get("experiment", {}).get("seed", 20260623))
    set_seed(seed)
    run_dir = Path(str(config.get("experiment", {}).get("output_dir")))
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir / "training.log")
    shutil.copy2(config_path, run_dir / "config.yaml")
    device = resolve_device(config)
    log(f"loaded config={config_path} device={device} seed={seed}")
    token_run_dir = Path(str(config.get("data", {}).get("token_run_dir")))
    splits = {split: load_split(token_run_dir, split) for split in available_splits(token_run_dir)}
    missing = {"train", "val", "test"} - set(splits)
    if missing:
        raise KeyError(f"Token run is missing required splits: {sorted(missing)}")
    sizes = vocab_sizes(token_run_dir, splits)
    max_branch_len = max(int(splits["train"][branch].shape[1]) for branch in BRANCHES)
    manifest = {
        "schema_version": "source_observation_foundation_task_type_v1",
        "config_path": str(config_path),
        "token_run_dir": str(token_run_dir),
        "run_dir": str(run_dir),
        "device": str(device),
        "seed": seed,
        "branch_order": list(BRANCHES),
        "vocab_sizes": dict(sizes),
        "max_branch_len": int(max_branch_len),
        "task_specs": {name: TASKS[name] for name in config.get("finetuning", {}).get("tasks", [])},
        "linear_probe_baseline": load_linear_probe_baseline(config, run_dir),
    }
    write_json(run_dir / "manifest.json", manifest)
    dry_run(config, splits, sizes, max_branch_len, run_dir, device, log)
    if dry_run_only:
        return
    pretrained_path = run_pretraining(config, splits, sizes, max_branch_len, run_dir, device, log)
    summaries: list[dict[str, Any]] = []
    fine_cfg = config.get("finetuning", {})
    tasks = list(fine_cfg.get("tasks", []))
    if bool(fine_cfg.get("run_scratch", True)):
        for task_name in tasks:
            summaries.append(
                train_one_task(
                    task_name=task_name,
                    experiment_name="supervised_transformer_scratch",
                    pretrained_path=None,
                    config=config,
                    splits=splits,
                    sizes=sizes,
                    max_branch_len=max_branch_len,
                    run_dir=run_dir,
                    device=device,
                    log=log,
                )
            )
    if bool(fine_cfg.get("run_pretrained", True)):
        if pretrained_path is None:
            raise RuntimeError("run_pretrained=true requires pretraining.enabled=true")
        for task_name in tasks:
            summaries.append(
                train_one_task(
                    task_name=task_name,
                    experiment_name="mlm_pretrain_then_finetune",
                    pretrained_path=pretrained_path,
                    config=config,
                    splits=splits,
                    sizes=sizes,
                    max_branch_len=max_branch_len,
                    run_dir=run_dir,
                    device=device,
                    log=log,
                )
            )
    save_rows_csv(
        run_dir / "summary.csv",
        [
            {
                "experiment": item["experiment"],
                "task": item["task"],
                "best_epoch": item["best_epoch"],
                "train_samples": item["train_samples"],
                "val_samples": item["val_samples"],
                "test_samples": item["test_samples"],
                "val_balanced_accuracy": item["val"]["balanced_accuracy"],
                "val_macro_f1": item["val"]["macro_f1"],
                "test_balanced_accuracy": item["test"]["balanced_accuracy"],
                "test_macro_f1": item["test"]["macro_f1"],
            }
            for item in summaries
        ],
    )
    write_json(run_dir / "summary.json", {"rows": summaries})
    log(f"experiment complete: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train source/observation token downstream foundation models")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/downstream/source_observation_foundation_task_type_smoke.yaml",
        help="Path to the foundation interface yaml config",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load one batch per split and write shape/task summary")
    args = parser.parse_args()

    require_standard_training_launcher("foundation-interface")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    run_experiment(config_path, dry_run_only=bool(args.dry_run))


if __name__ == "__main__":
    main()
