#!/usr/bin/env python
"""Pretrain and fine-tune whole-brain source/observation token models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
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


TASKS: dict[str, dict[str, Any]] = {
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
    "cognitive_task_type_5way": {
        "source_task": "cognitive",
        "label_field": "task_type_label_name",
        "label_names": ("0-back", "2-back", "3-back", "BL", "WG"),
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
    "croce_label_6way": {
        "source_task": None,
        "label_field": "label_name",
        "label_names": ("BL", "MA", "LMI", "RMI", "nback", "wg"),
    },
}

BRANCHES = ("eeg_source", "fnirs_source", "eeg_observation", "fnirs_observation")
BRANCH_MODALITY = torch.tensor([0, 1, 0, 1], dtype=torch.long)
BRANCH_ROLE = torch.tensor([0, 0, 1, 1], dtype=torch.long)
CONTRASTIVE_PAIRS = ((0, 1), (2, 3), (0, 2), (1, 3))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, text: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_config(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("experiment", {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_split(dataset_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = dataset_dir / "tokens" / f"{split}_wholebrain_tokens.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def infer_vocab_sizes(splits: Mapping[str, Mapping[str, np.ndarray]]) -> list[int]:
    sizes = []
    for branch_index in range(4):
        max_value = 0
        for data in splits.values():
            tokens = data["wholebrain_tokens"][:, :, branch_index, :]
            valid = tokens[tokens >= 0]
            if valid.size:
                max_value = max(max_value, int(valid.max()))
        sizes.append(max_value + 1)
    return sizes


def task_mask(data: Mapping[str, np.ndarray], task_name: str) -> np.ndarray:
    spec = TASKS[task_name]
    labels = np.asarray(data[str(spec["label_field"])]).astype(str)
    aliases = {str(key): str(value) for key, value in spec.get("label_aliases", {}).items()}
    accepted = set(str(label) for label in spec["label_names"]) | set(aliases)
    mask = np.isin(labels, np.asarray(sorted(accepted), dtype=str))
    source_task = spec.get("source_task")
    if source_task is not None:
        mask &= np.asarray(data["source_task"]).astype(str) == str(source_task)
    return mask


def encode_labels(data: Mapping[str, np.ndarray], task_name: str, mask: np.ndarray) -> np.ndarray:
    spec = TASKS[task_name]
    labels = np.asarray(data[str(spec["label_field"])]).astype(str)[mask]
    aliases = {str(key): str(value) for key, value in spec.get("label_aliases", {}).items()}
    labels = np.asarray([aliases.get(str(label), str(label)) for label in labels], dtype=str)
    class_to_id = {label: index for index, label in enumerate(spec["label_names"])}
    return np.asarray([class_to_id[label] for label in labels], dtype=np.int64)


class WholeBrainPretrainDataset(Dataset):
    def __init__(self, data: Mapping[str, np.ndarray]):
        self.tokens = np.asarray(data["wholebrain_tokens"], dtype=np.int64)
        self.anchor_mask = np.asarray(data["anchor_mask"], dtype=np.int64)
        self.subject_id = np.asarray(data["subject_id"], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.as_tensor(self.tokens[index], dtype=torch.long),
            "anchor_mask": torch.as_tensor(self.anchor_mask[index], dtype=torch.bool),
            "subject_id": torch.as_tensor(self.subject_id[index], dtype=torch.long),
        }


class WholeBrainTaskDataset(Dataset):
    def __init__(self, data: Mapping[str, np.ndarray], task_name: str):
        mask = task_mask(data, task_name)
        self.indices = np.flatnonzero(mask).astype(np.int64)
        if self.indices.size == 0:
            raise ValueError(f"No samples for task={task_name}")
        self.tokens = np.asarray(data["wholebrain_tokens"], dtype=np.int64)[self.indices]
        self.anchor_mask = np.asarray(data["anchor_mask"], dtype=np.int64)[self.indices]
        self.labels = encode_labels(data, task_name, mask)
        self.subject_id = np.asarray(data["subject_id"], dtype=np.int64)[self.indices]
        self.classes = list(TASKS[task_name]["label_names"])

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.as_tensor(self.tokens[index], dtype=torch.long),
            "anchor_mask": torch.as_tensor(self.anchor_mask[index], dtype=torch.bool),
            "label": torch.as_tensor(self.labels[index], dtype=torch.long),
            "subject_id": torch.as_tensor(self.subject_id[index], dtype=torch.long),
        }


class WholeBrainBackbone(nn.Module):
    def __init__(
        self,
        *,
        vocab_sizes: list[int],
        anchor_count: int,
        hidden_dim: int,
        num_heads: int,
        temporal_layers: int,
        spatial_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.vocab_sizes = list(vocab_sizes)
        self.anchor_count = int(anchor_count)
        self.hidden_dim = int(hidden_dim)
        self.token_embeddings = nn.ModuleList(
            [nn.Embedding(int(vocab) + 2, hidden_dim, padding_idx=0) for vocab in self.vocab_sizes]
        )
        self.anchor_embedding = nn.Embedding(anchor_count, hidden_dim)
        self.branch_embedding = nn.Embedding(4, hidden_dim)
        self.modality_embedding = nn.Embedding(2, hidden_dim)
        self.role_embedding = nn.Embedding(2, hidden_dim)
        self.time_embedding = nn.Embedding(32, hidden_dim)
        temporal_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, temporal_layers)
        branch_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.branch_encoder = nn.TransformerEncoder(branch_layer, 1)
        spatial_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, spatial_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def raw_to_input_ids(self, raw_tokens: torch.Tensor, mask_positions: torch.Tensor | None = None) -> torch.Tensor:
        token_valid = raw_tokens >= 0
        input_ids = (raw_tokens + 2).masked_fill(~token_valid, 0)
        if mask_positions is not None:
            input_ids = input_ids.masked_fill(mask_positions & token_valid, 1)
        return input_ids

    def forward(
        self,
        raw_tokens: torch.Tensor,
        anchor_mask: torch.Tensor,
        *,
        mask_positions: torch.Tensor | None = None,
        return_token_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        batch_size, anchor_count, branch_count, token_count = raw_tokens.shape
        branch_ids = torch.arange(branch_count, device=raw_tokens.device)
        anchor_ids = torch.arange(anchor_count, device=raw_tokens.device)
        time_ids = torch.arange(token_count, device=raw_tokens.device)
        modality_ids = BRANCH_MODALITY.to(raw_tokens.device)
        role_ids = BRANCH_ROLE.to(raw_tokens.device)
        token_valid = raw_tokens >= 0
        input_tokens = self.raw_to_input_ids(raw_tokens, mask_positions=mask_positions)
        branch_states = []
        branch_token_states = []
        for branch_index in range(branch_count):
            states = self.token_embeddings[branch_index](input_tokens[:, :, branch_index, :])
            states = states + self.anchor_embedding(anchor_ids).view(1, anchor_count, 1, -1)
            states = states + self.time_embedding(time_ids).view(1, 1, token_count, -1)
            states = states + self.branch_embedding(branch_ids[branch_index]).view(1, 1, 1, -1)
            states = states + self.modality_embedding(modality_ids[branch_index]).view(1, 1, 1, -1)
            states = states + self.role_embedding(role_ids[branch_index]).view(1, 1, 1, -1)
            states = states.reshape(batch_size * anchor_count, token_count, -1)
            valid = token_valid[:, :, branch_index, :].reshape(batch_size * anchor_count, token_count)
            encoded = self.temporal_encoder(states)
            valid_float = valid.float().unsqueeze(-1)
            pooled = (encoded * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
            branch_states.append(pooled.reshape(batch_size, anchor_count, -1))
            branch_token_states.append(encoded.reshape(batch_size, anchor_count, token_count, -1))

        stacked = torch.stack(branch_states, dim=2)
        branch_context = self.branch_encoder(stacked.reshape(batch_size * anchor_count, branch_count, -1))
        branch_context = branch_context.reshape(batch_size, anchor_count, branch_count, -1)
        fused = branch_context.mean(dim=2)
        spatial = self.spatial_encoder(fused, src_key_padding_mask=~anchor_mask)
        spatial = self.norm(spatial)
        weights = anchor_mask.float().unsqueeze(-1)
        event_embedding = (spatial * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        branch_event_embeddings = (
            branch_context * anchor_mask[:, :, None, None].float()
        ).sum(dim=1) / anchor_mask.float().sum(dim=1).clamp_min(1.0).view(batch_size, 1, 1)
        output: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "event_embedding": event_embedding,
            "branch_event_embeddings": branch_event_embeddings,
            "spatial_states": spatial,
            "branch_context": branch_context,
        }
        if return_token_states:
            output["token_states"] = branch_token_states
        return output


class WholeBrainPretrainModel(nn.Module):
    def __init__(self, backbone: WholeBrainBackbone, dropout: float):
        super().__init__()
        self.backbone = backbone
        hidden_dim = backbone.hidden_dim
        self.mlm_heads = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, int(vocab)))
                for vocab in backbone.vocab_sizes
            ]
        )
        self.contrast_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, raw_tokens: torch.Tensor, anchor_mask: torch.Tensor, mask_positions: torch.Tensor) -> dict[str, Any]:
        encoded = self.backbone(
            raw_tokens,
            anchor_mask,
            mask_positions=mask_positions,
            return_token_states=True,
        )
        token_states = encoded["token_states"]
        spatial_states = encoded["spatial_states"]
        branch_context = encoded["branch_context"]
        logits = []
        for branch_index, states in enumerate(token_states):
            contextual_states = (
                states
                + branch_context[:, :, branch_index, :].unsqueeze(2)
                + spatial_states.unsqueeze(2)
            )
            logits.append(self.mlm_heads[branch_index](contextual_states))
        branch_embeddings = self.contrast_projection(encoded["branch_event_embeddings"])
        return {"mlm_logits": logits, "branch_embeddings": branch_embeddings}


class WholeBrainClassifier(nn.Module):
    def __init__(self, backbone: WholeBrainBackbone, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.hidden_dim, num_classes))

    def forward(self, raw_tokens: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.backbone(raw_tokens, anchor_mask)
        return self.classifier(encoded["event_embedding"])


def make_loader(
    dataset: Dataset,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    shuffle: bool,
    stage: str,
) -> DataLoader:
    stage_cfg = config.get(stage, {})
    batch_size = int(stage_cfg.get("batch_size", config.get("optimization", {}).get("batch_size", 128)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )


def build_backbone(
    *,
    config: Mapping[str, Any],
    vocab_sizes: list[int],
    anchor_count: int,
) -> WholeBrainBackbone:
    model_cfg = config.get("model", {})
    return WholeBrainBackbone(
        vocab_sizes=vocab_sizes,
        anchor_count=anchor_count,
        hidden_dim=int(model_cfg.get("hidden_dim", 192)),
        num_heads=int(model_cfg.get("num_heads", 6)),
        temporal_layers=int(model_cfg.get("temporal_layers", 1)),
        spatial_layers=int(model_cfg.get("spatial_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )


def make_span_mask(
    raw_tokens: torch.Tensor,
    *,
    mask_ratio: float,
    min_span: int,
    max_span: int,
) -> torch.Tensor:
    valid = raw_tokens >= 0
    span = random.randint(min_span, max_span)
    start_probability = min(1.0, max(mask_ratio / float(max(span, 1)), 1e-6))
    starts = (torch.rand(raw_tokens.shape, device=raw_tokens.device) < start_probability) & valid
    mask = torch.zeros_like(starts)
    for offset in range(span):
        if offset == 0:
            mask |= starts
        else:
            mask[..., offset:] |= starts[..., :-offset]
    mask &= valid
    has_valid = valid.any(dim=-1)
    has_mask = mask.any(dim=-1)
    needs_one = has_valid & ~has_mask
    if bool(needs_one.any()):
        scores = torch.rand(raw_tokens.shape, device=raw_tokens.device).masked_fill(~valid, -1.0)
        positions = scores.argmax(dim=-1, keepdim=True)
        mask.scatter_(-1, positions, needs_one.unsqueeze(-1))
    return mask


def mlm_loss_and_accuracy(
    logits_by_branch: list[torch.Tensor],
    raw_tokens: torch.Tensor,
    mask_positions: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    losses = []
    correct = 0
    total = 0
    for branch_index, logits in enumerate(logits_by_branch):
        branch_mask = mask_positions[:, :, branch_index, :]
        if not bool(branch_mask.any()):
            continue
        targets = raw_tokens[:, :, branch_index, :][branch_mask]
        predictions = logits[branch_mask]
        losses.append(nn.functional.cross_entropy(predictions, targets))
        predicted_ids = predictions.argmax(dim=-1)
        correct += int((predicted_ids == targets).sum().item())
        total += int(targets.numel())
    if not losses:
        return raw_tokens.new_tensor(0.0, dtype=torch.float32), 0.0
    return torch.stack(losses).mean(), float(correct / max(total, 1))


def contrastive_loss(branch_embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    embeddings = nn.functional.normalize(branch_embeddings, dim=-1)
    labels = torch.arange(embeddings.shape[0], device=embeddings.device)
    losses = []
    for left, right in CONTRASTIVE_PAIRS:
        logits = embeddings[:, left, :] @ embeddings[:, right, :].T / temperature
        losses.append(nn.functional.cross_entropy(logits, labels))
        losses.append(nn.functional.cross_entropy(logits.T, labels))
    return torch.stack(losses).mean()


def run_pretrain_epoch(
    model: WholeBrainPretrainModel,
    loader: DataLoader,
    device: torch.device,
    config: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    pre_cfg = config.get("pretraining", {})
    opt_cfg = config.get("optimization", {})
    total_losses = []
    mlm_losses = []
    contrastive_losses = []
    accuracies = []
    max_batches = int(pre_cfg.get("max_train_batches", 0)) if training else int(pre_cfg.get("max_eval_batches", 0))
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches > 0 and batch_index > max_batches:
                break
            tokens = batch["tokens"].to(device, non_blocking=True)
            anchor_mask = batch["anchor_mask"].to(device, non_blocking=True)
            mask_positions = make_span_mask(
                tokens,
                mask_ratio=float(pre_cfg.get("mask_ratio", 0.25)),
                min_span=int(pre_cfg.get("span_min", 2)),
                max_span=int(pre_cfg.get("span_max", 4)),
            )
            outputs = model(tokens, anchor_mask, mask_positions)
            mlm_loss, accuracy = mlm_loss_and_accuracy(outputs["mlm_logits"], tokens, mask_positions)
            nce_loss = contrastive_loss(
                outputs["branch_embeddings"],
                temperature=float(pre_cfg.get("contrastive_temperature", 0.1)),
            )
            total_loss = mlm_loss + float(pre_cfg.get("contrastive_weight", 0.1)) * nce_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_clip = float(opt_cfg.get("grad_clip_norm", 1.0))
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_losses.append(float(total_loss.item()))
            mlm_losses.append(float(mlm_loss.item()))
            contrastive_losses.append(float(nce_loss.item()))
            accuracies.append(float(accuracy))
    return {
        "loss": float(np.mean(total_losses)),
        "mlm_loss": float(np.mean(mlm_losses)),
        "contrastive_loss": float(np.mean(contrastive_losses)),
        "mlm_accuracy": float(np.mean(accuracies)),
    }


def train_pretraining(
    *,
    config: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, np.ndarray]],
    vocab_sizes: list[int],
    anchor_count: int,
    run_dir: Path,
    device: torch.device,
    log: Logger,
) -> Path:
    out_dir = run_dir / "pretraining"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(WholeBrainPretrainDataset(splits["train"]), config, device, shuffle=True, stage="pretraining")
    val_loader = make_loader(WholeBrainPretrainDataset(splits["val"]), config, device, shuffle=False, stage="pretraining")
    backbone = build_backbone(config=config, vocab_sizes=vocab_sizes, anchor_count=anchor_count)
    model = WholeBrainPretrainModel(backbone, dropout=float(config.get("model", {}).get("dropout", 0.2))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("pretraining", {}).get("learning_rate", config.get("optimization", {}).get("learning_rate", 2e-4))),
        weight_decay=float(config.get("optimization", {}).get("weight_decay", 0.05)),
    )
    best_path = out_dir / "checkpoints" / "best_model.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    epochs = int(config.get("pretraining", {}).get("epochs", 20))
    patience = int(config.get("pretraining", {}).get("patience", 5))
    for epoch in range(1, epochs + 1):
        train_metrics = run_pretrain_epoch(model, train_loader, device, config, optimizer=optimizer)
        val_metrics = run_pretrain_epoch(model, val_loader, device, config, optimizer=None)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)
        save_rows_csv(out_dir / "metrics.csv", rows)
        log(
            "pretrain epoch "
            f"{epoch}/{epochs} train_mlm={train_metrics['mlm_loss']:.4f} "
            f"val_mlm={val_metrics['mlm_loss']:.4f} val_acc={val_metrics['mlm_accuracy']:.4f}"
        )
        if val_metrics["mlm_loss"] < best_val:
            best_val = val_metrics["mlm_loss"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "backbone_state_dict": model.backbone.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "vocab_sizes": vocab_sizes,
                    "anchor_count": anchor_count,
                    "epoch": epoch,
                    "val": val_metrics,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    write_json(
        out_dir / "summary.json",
        {
            "best_epoch": int(best_epoch),
            "best_checkpoint": str(best_path),
            "best_val_mlm_loss": float(best_val),
            "rows": rows,
        },
    )
    return best_path


def class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return torch.as_tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["anchor_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(tokens, mask)
            losses.append(float(ce(logits, labels).item()))
            y_true.append(labels.cpu().numpy())
            y_pred.append(logits.argmax(dim=-1).cpu().numpy())
    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    return {
        "loss": float(np.mean(losses)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
    }, true, pred


def train_task(
    *,
    task_name: str,
    config: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, np.ndarray]],
    vocab_sizes: list[int],
    anchor_count: int,
    pretrain_checkpoint: Path,
    run_dir: Path,
    device: torch.device,
    log: Logger,
) -> dict[str, Any]:
    train_set = WholeBrainTaskDataset(splits["train"], task_name)
    val_set = WholeBrainTaskDataset(splits["val"], task_name)
    test_set = WholeBrainTaskDataset(splits["test"], task_name)
    backbone = build_backbone(config=config, vocab_sizes=vocab_sizes, anchor_count=anchor_count)
    checkpoint = torch.load(pretrain_checkpoint, map_location="cpu")
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    model = WholeBrainClassifier(
        backbone,
        num_classes=len(train_set.classes),
        dropout=float(config.get("model", {}).get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("finetuning", {}).get("learning_rate", config.get("optimization", {}).get("learning_rate", 2e-4))),
        weight_decay=float(config.get("optimization", {}).get("weight_decay", 0.05)),
    )
    train_loader = make_loader(train_set, config, device, shuffle=True, stage="finetuning")
    val_loader = make_loader(val_set, config, device, shuffle=False, stage="finetuning")
    test_loader = make_loader(test_set, config, device, shuffle=False, stage="finetuning")
    weights = class_weights(train_set.labels, len(train_set.classes), device)
    ce = nn.CrossEntropyLoss(weight=weights)
    out_dir = run_dir / "mlm_pretrain_then_finetune" / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_epoch = 0
    best_path = out_dir / "checkpoints" / "best_model.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    stale = 0
    epochs = int(config.get("finetuning", {}).get("epochs", 30))
    patience = int(config.get("finetuning", {}).get("patience", 8))
    grad_clip = float(config.get("optimization", {}).get("grad_clip_norm", 1.0))
    max_batches = int(config.get("finetuning", {}).get("max_train_batches", 0))
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if max_batches > 0 and batch_index > max_batches:
                break
            tokens = batch["tokens"].to(device, non_blocking=True)
            mask = batch["anchor_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(tokens, mask)
            loss = ce(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_metrics, _, _ = evaluate_classifier(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)
        save_rows_csv(out_dir / "metrics.csv", rows)
        log(f"{task_name} epoch {epoch}/{epochs} train_loss={row['train_loss']:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f}")
        if val_metrics["balanced_accuracy"] > best_val:
            best_val = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone_state_dict": model.backbone.state_dict(),
                    "classes": train_set.classes,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_true, val_pred = evaluate_classifier(model, val_loader, device)
    test_metrics, test_true, test_pred = evaluate_classifier(model, test_loader, device)
    write_json(
        out_dir / "confusion_matrix.json",
        {
            "classes": train_set.classes,
            "val": confusion_matrix(val_true, val_pred, labels=list(range(len(train_set.classes)))).tolist(),
            "test": confusion_matrix(test_true, test_pred, labels=list(range(len(train_set.classes)))).tolist(),
        },
    )
    summary = {
        "experiment": "mlm_pretrain_then_finetune",
        "task": task_name,
        "classes": train_set.classes,
        "train_samples": int(len(train_set)),
        "val_samples": int(len(val_set)),
        "test_samples": int(len(test_set)),
        "train_subjects": int(np.unique(train_set.subject_id).size),
        "val_subjects": int(np.unique(val_set.subject_id).size),
        "test_subjects": int(np.unique(test_set.subject_id).size),
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "pretrain_checkpoint": str(pretrain_checkpoint),
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def ensure_architecture_figure(run_dir: Path, log: Logger) -> str:
    output = run_dir / "figures" / "model_architecture.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    script = project_root / "experiments" / "scripts" / "create_wholebrain_architecture_figure.py"
    subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        check=True,
        cwd=str(project_root),
    )
    log(f"saved architecture figure={output}")
    return str(output)


def split_summary(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    anchor_mask = np.asarray(data["anchor_mask"], dtype=np.int64)
    tokens = np.asarray(data["wholebrain_tokens"], dtype=np.int64)
    return {
        "samples": int(tokens.shape[0]),
        "shape": list(tokens.shape),
        "effective_anchor_min": int(anchor_mask.sum(axis=1).min()),
        "effective_anchor_max": int(anchor_mask.sum(axis=1).max()),
        "effective_anchor_mean": float(anchor_mask.sum(axis=1).mean()),
        "subjects": int(np.unique(np.asarray(data["subject_id"])).size),
    }


def build_dry_run_summary(
    *,
    dataset_dir: Path,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    vocab_sizes: list[int],
    anchor_count: int,
    tasks: list[str],
) -> dict[str, Any]:
    dry: dict[str, Any] = {
        "schema_version": "wholebrain_pretrain_dry_run_v1",
        "dataset_dir": str(dataset_dir),
        "anchor_count": anchor_count,
        "branch_order": list(BRANCHES),
        "vocab_sizes": vocab_sizes,
        "splits": {split: split_summary(data) for split, data in splits.items()},
        "tasks": [],
    }
    for task in tasks:
        entry: dict[str, Any] = {"task": task, "classes": list(TASKS[task]["label_names"])}
        for split, data in splits.items():
            dataset = WholeBrainTaskDataset(data, task)
            entry[split] = {
                "samples": int(len(dataset)),
                "subjects": int(np.unique(dataset.subject_id).size),
                "class_counts": {
                    dataset.classes[index]: int(count)
                    for index, count in enumerate(np.bincount(dataset.labels, minlength=len(dataset.classes)))
                },
            }
        dry["tasks"].append(entry)
    return dry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    require_standard_training_launcher("wholebrain-pretrain")
    config_path = Path(args.config)
    config = read_yaml(config_path)
    seed = int(config.get("experiment", {}).get("seed", 20260624))
    set_seed(seed)
    run_dir = Path(str(config["experiment"]["output_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    log = Logger(run_dir / "training.log")
    architecture_figure = ensure_architecture_figure(run_dir, log)
    device = device_from_config(config)
    dataset_dir = Path(str(config["data"]["wholebrain_token_dir"]))
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = {split: load_split(dataset_dir, split) for split in ("train", "val", "test")}
    vocab_sizes = infer_vocab_sizes(splits)
    anchor_count = int(manifest["anchor_count"])
    tasks = [str(task) for task in config.get("finetuning", {}).get("tasks", [])]
    dry = build_dry_run_summary(
        dataset_dir=dataset_dir,
        splits=splits,
        vocab_sizes=vocab_sizes,
        anchor_count=anchor_count,
        tasks=tasks,
    )
    write_json(run_dir / "dry_run_summary.json", dry)
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "source_observation_wholebrain_pretrain_v1",
            "config_path": str(config_path),
            "dataset_dir": str(dataset_dir),
            "run_dir": str(run_dir),
            "device": str(device),
            "seed": seed,
            "dataset_manifest": str(dataset_dir / "manifest.json"),
            "architecture_figure": architecture_figure,
            "branch_order": list(BRANCHES),
            "vocab_sizes": vocab_sizes,
            "anchor_count": anchor_count,
        },
    )
    log(f"loaded config={config_path} device={device} seed={seed}")
    if args.dry_run:
        log("dry-run complete")
        return 0
    pretrain_checkpoint = train_pretraining(
        config=config,
        splits=splits,
        vocab_sizes=vocab_sizes,
        anchor_count=anchor_count,
        run_dir=run_dir,
        device=device,
        log=log,
    )
    summaries = []
    for task in tasks:
        summaries.append(
            train_task(
                task_name=task,
                config=config,
                splits=splits,
                vocab_sizes=vocab_sizes,
                anchor_count=anchor_count,
                pretrain_checkpoint=pretrain_checkpoint,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
