#!/usr/bin/env python
"""Train whole-brain downstream models on grouped source/observation token samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
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


class WholeBrainTokenModel(nn.Module):
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
        num_classes: int,
    ):
        super().__init__()
        self.vocab_sizes = list(vocab_sizes)
        self.anchor_count = int(anchor_count)
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
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, raw_tokens: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
        batch_size, anchor_count, branch_count, token_count = raw_tokens.shape
        branch_ids = torch.arange(branch_count, device=raw_tokens.device)
        anchor_ids = torch.arange(anchor_count, device=raw_tokens.device)
        time_ids = torch.arange(token_count, device=raw_tokens.device)
        modality_ids = BRANCH_MODALITY.to(raw_tokens.device)
        role_ids = BRANCH_ROLE.to(raw_tokens.device)
        branch_states = []
        token_valid = raw_tokens >= 0
        input_tokens = (raw_tokens + 2).masked_fill(~token_valid, 0)
        for branch_index in range(branch_count):
            states = self.token_embeddings[branch_index](input_tokens[:, :, branch_index, :])
            states = states + self.anchor_embedding(anchor_ids).view(1, anchor_count, 1, -1)
            states = states + self.time_embedding(time_ids).view(1, 1, token_count, -1)
            states = states + self.branch_embedding(branch_ids[branch_index]).view(1, 1, 1, -1)
            states = states + self.modality_embedding(modality_ids[branch_index]).view(1, 1, 1, -1)
            states = states + self.role_embedding(role_ids[branch_index]).view(1, 1, 1, -1)
            states = states.reshape(batch_size * anchor_count, token_count, -1)
            key_padding = ~token_valid[:, :, branch_index, :].reshape(batch_size * anchor_count, token_count)
            # Missing anchors are removed by the later spatial mask. Avoid passing
            # all-masked temporal rows to attention, which can produce NaNs.
            encoded = self.temporal_encoder(states)
            valid = (~key_padding).float().unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            branch_states.append(pooled.reshape(batch_size, anchor_count, -1))
        stacked = torch.stack(branch_states, dim=2).reshape(batch_size * anchor_count, branch_count, -1)
        fused = self.branch_encoder(stacked).mean(dim=1).reshape(batch_size, anchor_count, -1)
        spatial = self.spatial_encoder(fused, src_key_padding_mask=~anchor_mask)
        spatial = self.norm(spatial)
        weights = anchor_mask.float().unsqueeze(-1)
        pooled = (spatial * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


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


def make_loader(dataset: Dataset, config: Mapping[str, Any], device: torch.device, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config.get("optimization", {}).get("batch_size", 128)),
        shuffle=shuffle,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
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


def class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return torch.as_tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)


def train_task(
    *,
    task_name: str,
    config: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, np.ndarray]],
    vocab_sizes: list[int],
    anchor_count: int,
    run_dir: Path,
    device: torch.device,
    log: Logger,
) -> dict[str, Any]:
    train_set = WholeBrainTaskDataset(splits["train"], task_name)
    val_set = WholeBrainTaskDataset(splits["val"], task_name)
    test_set = WholeBrainTaskDataset(splits["test"], task_name)
    model_cfg = config.get("model", {})
    model = WholeBrainTokenModel(
        vocab_sizes=vocab_sizes,
        anchor_count=anchor_count,
        hidden_dim=int(model_cfg.get("hidden_dim", 192)),
        num_heads=int(model_cfg.get("num_heads", 6)),
        temporal_layers=int(model_cfg.get("temporal_layers", 1)),
        spatial_layers=int(model_cfg.get("spatial_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        num_classes=len(train_set.classes),
    ).to(device)
    opt_cfg = config.get("optimization", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
    )
    train_loader = make_loader(train_set, config, device, shuffle=True)
    val_loader = make_loader(val_set, config, device, shuffle=False)
    test_loader = make_loader(test_set, config, device, shuffle=False)
    weights = class_weights(train_set.labels, len(train_set.classes), device)
    ce = nn.CrossEntropyLoss(weight=weights)
    out_dir = run_dir / "supervised_wholebrain_scratch" / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_epoch = 0
    best_path = out_dir / "checkpoints" / "best_model.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    stale = 0
    epochs = int(config.get("finetuning", {}).get("epochs", 30))
    patience = int(config.get("finetuning", {}).get("patience", 8))
    grad_clip = float(opt_cfg.get("grad_clip_norm", 1.0))
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
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
        val_metrics, _, _ = evaluate(model, val_loader, device)
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
            torch.save({"model_state_dict": model.state_dict(), "classes": train_set.classes, "epoch": epoch}, best_path)
        else:
            stale += 1
            if stale >= patience:
                break
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_true, val_pred = evaluate(model, val_loader, device)
    test_metrics, test_true, test_pred = evaluate(model, test_loader, device)
    write_json(
        out_dir / "confusion_matrix.json",
        {
            "classes": train_set.classes,
            "val": confusion_matrix(val_true, val_pred, labels=list(range(len(train_set.classes)))).tolist(),
            "test": confusion_matrix(test_true, test_pred, labels=list(range(len(train_set.classes)))).tolist(),
        },
    )
    summary = {
        "experiment": "supervised_wholebrain_scratch",
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
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    require_standard_training_launcher("wholebrain-foundation")
    config_path = Path(args.config)
    config = read_yaml(config_path)
    seed = int(config.get("experiment", {}).get("seed", 20260624))
    set_seed(seed)
    run_dir = Path(str(config["experiment"]["output_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    log = Logger(run_dir / "training.log")
    device = device_from_config(config)
    dataset_dir = Path(str(config["data"]["wholebrain_token_dir"]))
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = {split: load_split(dataset_dir, split) for split in ("train", "val", "test")}
    vocab_sizes = infer_vocab_sizes(splits)
    anchor_count = int(manifest["anchor_count"])
    dry = {
        "schema_version": "wholebrain_foundation_dry_run_v1",
        "dataset_dir": str(dataset_dir),
        "anchor_count": anchor_count,
        "branch_order": list(BRANCHES),
        "vocab_sizes": vocab_sizes,
        "tasks": [],
    }
    for task in config.get("finetuning", {}).get("tasks", []):
        dataset = WholeBrainTaskDataset(splits["train"], task)
        dry["tasks"].append(
            {
                "task": task,
                "classes": dataset.classes,
                "train_samples": int(len(dataset)),
                "class_counts": {
                    dataset.classes[index]: int(count)
                    for index, count in enumerate(np.bincount(dataset.labels, minlength=len(dataset.classes)))
                },
            }
        )
    write_json(run_dir / "dry_run_summary.json", dry)
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "source_observation_wholebrain_foundation_v1",
            "config_path": str(config_path),
            "dataset_dir": str(dataset_dir),
            "run_dir": str(run_dir),
            "device": str(device),
            "seed": seed,
            "dataset_manifest": str(dataset_dir / "manifest.json"),
            "architecture_figure": str(run_dir / "figures" / "model_architecture.png"),
        },
    )
    log(f"loaded config={config_path} device={device} seed={seed}")
    if args.dry_run:
        log("dry-run complete")
        return 0
    summaries = []
    for task in config.get("finetuning", {}).get("tasks", []):
        summaries.append(
            train_task(
                task_name=task,
                config=config,
                splits=splits,
                vocab_sizes=vocab_sizes,
                anchor_count=anchor_count,
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
