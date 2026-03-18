#!/usr/bin/env python
"""
Train Shared-Codebook EEG-fNIRS Tokenizer with Contrastive Alignment.

Instead of training EEG and fNIRS tokenizers separately, this script trains
both modalities jointly with a SINGLE shared codebook and an InfoNCE contrastive
loss that explicitly aligns EEG and fNIRS token representations from the same
cognitive trial.

Design rationale
----------------
- Single codebook → EEG and fNIRS tokens live in the *same* discrete vocabulary.
- Contrastive loss (InfoNCE) → pulls same-trial (EEG, fNIRS) embeddings together
  and pushes apart cross-trial pairs within each batch.
- Index-matching loss → softly encourages same codebook entry to be selected by
  both modalities for the same trial.

Usage
-----
    python train_shared_codebook.py \\
        --data-root data/EEG+NIRS\\ Single-Trial \\
        --codebook-size 1024 \\
        --embedding-dim 64 \\
        --epochs 100 \\
        --device cuda

Background run (nohup):
    nohup python train_shared_codebook.py --data-root ... > training.log 2>&1 &
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── project root ───────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.tokenizers.shared_codebook import SharedCodebookTokenizer
from src.losses.contrastive_losses import SharedCodebookLoss
from src.metrics.codebook_health import compute_codebook_health
from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset


# ============================================================================
# TeeLogger – simultaneous stdout + file logging
# ============================================================================

class TeeLogger:
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "a", buffering=1)

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ============================================================================
# Data helpers
# ============================================================================

def create_paired_dataloaders(
    data_root: str,
    window_duration_s: float = 4.0,
    batch_size: int = 32,
    num_workers: int = 0,
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    test_subjects: Optional[List[int]] = None,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders that return *paired* EEG+fNIRS batches."""
    train_subjects = train_subjects or list(range(1, 21))
    val_subjects   = val_subjects   or list(range(21, 26))
    test_subjects  = test_subjects  or list(range(26, 30))

    loaders = {}
    for split, subjects in [
        ("train", train_subjects),
        ("val",   val_subjects),
        ("test",  test_subjects),
    ]:
        dataset = MultiModalEEGfNIRSDataset(
            data_root=data_root,
            subject_ids=subjects,
            task="motor_imagery",
            window_duration_s=window_duration_s,
            exclude_eog=True,
            hbo_only=True,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
    return loaders


# ============================================================================
# Train / evaluate one epoch
# ============================================================================

def train_epoch(
    model: SharedCodebookTokenizer,
    criterion: SharedCodebookLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()

    totals: Dict[str, float] = {}
    n_batches = 0
    all_indices_eeg: List[torch.Tensor] = []
    all_indices_fnirs: List[torch.Tensor] = []

    for batch in loader:
        x_eeg   = batch["eeg"].to(device)    # [B, C_eeg,   T_eeg]
        x_fnirs = batch["fnirs"].to(device)   # [B, C_fnirs, T_fnirs]

        optimizer.zero_grad()

        outputs = model(x_eeg, x_fnirs)
        total_loss, losses = criterion(outputs, x_eeg, x_fnirs)

        total_loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            totals[k] = totals.get(k, 0.0) + val

        all_indices_eeg.append(outputs["indices_eeg"].detach().cpu())
        all_indices_fnirs.append(outputs["indices_fnirs"].detach().cpu())
        n_batches += 1

    metrics = {k: v / n_batches for k, v in totals.items()}

    # Codebook health from EEG indices
    indices_eeg = torch.cat(all_indices_eeg, dim=0)
    cb_health = compute_codebook_health(indices_eeg, model.get_codebook_size())
    metrics.update({f"eeg_{k}": v for k, v in cb_health.items()})

    # From fNIRS indices
    indices_fnirs = torch.cat(all_indices_fnirs, dim=0)
    cb_health_f = compute_codebook_health(indices_fnirs, model.get_codebook_size())
    metrics.update({f"fnirs_{k}": v for k, v in cb_health_f.items()})

    return metrics


@torch.no_grad()
def evaluate(
    model: SharedCodebookTokenizer,
    criterion: SharedCodebookLoss,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    totals: Dict[str, float] = {}
    n_batches = 0
    all_indices_eeg: List[torch.Tensor] = []
    all_indices_fnirs: List[torch.Tensor] = []

    for batch in loader:
        x_eeg   = batch["eeg"].to(device)
        x_fnirs = batch["fnirs"].to(device)

        outputs = model(x_eeg, x_fnirs)
        _, losses = criterion(outputs, x_eeg, x_fnirs)

        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            totals[k] = totals.get(k, 0.0) + val

        all_indices_eeg.append(outputs["indices_eeg"].cpu())
        all_indices_fnirs.append(outputs["indices_fnirs"].cpu())
        n_batches += 1

    metrics = {k: v / n_batches for k, v in totals.items()}

    indices_eeg = torch.cat(all_indices_eeg, dim=0)
    cb_health = compute_codebook_health(indices_eeg, model.get_codebook_size())
    metrics.update({f"eeg_{k}": v for k, v in cb_health.items()})

    indices_fnirs = torch.cat(all_indices_fnirs, dim=0)
    cb_health_f = compute_codebook_health(indices_fnirs, model.get_codebook_size())
    metrics.update({f"fnirs_{k}": v for k, v in cb_health_f.items()})

    return metrics


# ============================================================================
# Main training function
# ============================================================================

def train(args: argparse.Namespace) -> Path:
    # ── output directory ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        project_root
        / "experiments"
        / "runs"
        / f"shared_codebook_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()

    # Redirect stdout/stderr to log file
    tee = TeeLogger(run_dir / "training.log")
    sys.stdout = tee
    sys.stderr = tee

    print(f"\n{'='*70}")
    print("Shared-Codebook EEG-fNIRS Contrastive Tokenizer Training")
    print(f"{'='*70}")
    print(f"Run directory : {run_dir}")
    print(f"Device        : {args.device}")
    print(f"Codebook size : {args.codebook_size}")
    print(f"Embedding dim : {args.embedding_dim}")
    print(f"Contrastive λ : {args.contrastive_weight}")
    print(f"Index-match λ : {args.index_match_weight}")
    print(f"Temperature   : {args.temperature}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")

    device = torch.device(args.device)

    # ── data ──────────────────────────────────────────────────────────────
    print(f"\nLoading data from: {args.data_root}")
    loaders = create_paired_dataloaders(
        data_root=args.data_root,
        window_duration_s=4.0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Infer channel and sample counts from first batch
    sample_batch = next(iter(loaders["train"]))
    _, C_eeg,   T_eeg   = sample_batch["eeg"].shape
    _, C_fnirs, T_fnirs = sample_batch["fnirs"].shape
    print(f"\nData shapes:")
    print(f"  EEG   : [B, {C_eeg},   {T_eeg}]  (channels × samples)")
    print(f"  fNIRS : [B, {C_fnirs}, {T_fnirs}]  (channels × samples)")
    print(f"  Train batches : {len(loaders['train'])}")
    print(f"  Val   batches : {len(loaders['val'])}")
    print(f"  Test  batches : {len(loaders['test'])}")

    # ── model ─────────────────────────────────────────────────────────────
    print("\nBuilding SharedCodebookTokenizer …")
    model = SharedCodebookTokenizer(
        eeg_seq_length=T_eeg,
        eeg_patch_size=args.eeg_patch_size,
        eeg_channels=C_eeg,
        eeg_hidden_dim=args.eeg_hidden_dim,
        fnirs_seq_length=T_fnirs,
        fnirs_patch_size=args.fnirs_patch_size,
        fnirs_channels=C_fnirs,
        fnirs_hidden_dim=args.fnirs_hidden_dim,
        codebook_size=args.codebook_size,
        embedding_dim=args.embedding_dim,
        num_encoder_layers=args.num_encoder_layers,
        vq_beta=args.vq_beta,
        vq_decay=args.vq_decay,
        kmeans_init=True,
    ).to(device)

    # ── loss ──────────────────────────────────────────────────────────────
    criterion = SharedCodebookLoss(
        contrastive_weight=args.contrastive_weight,
        index_match_weight=args.index_match_weight,
        temperature=args.temperature,
        index_match_temperature=args.index_match_temperature,
    )

    # ── optimizer / scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history: Dict[str, List] = {"train": [], "val": []}

    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}")
    print(
        f"{'Epoch':>6} | {'total':>8} | {'rec_eeg':>8} | {'rec_fnirs':>9} | "
        f"{'contrast':>8} | {'idx_match':>9} | {'eeg_util%':>10} | "
        f"{'fnirs_util%':>11} | {'val_total':>9} | {'val_align%':>10}"
    )
    print("-" * 110)

    start_time = time.time()

    for epoch in range(args.epochs):
        t_metrics = train_epoch(model, criterion, loaders["train"],
                                optimizer, device, args.grad_clip)
        v_metrics = evaluate(model, criterion, loaders["val"], device)

        scheduler.step()

        history["train"].append({**t_metrics, "lr": optimizer.param_groups[0]["lr"]})
        history["val"].append(v_metrics)

        # ── checkpoint ────────────────────────────────────────────────────
        val_loss = v_metrics.get("total", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                ckpt_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        # ── logging ───────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            eeg_util   = t_metrics.get("eeg_code_utilization", 0.0) * 100
            fnirs_util = t_metrics.get("fnirs_code_utilization", 0.0) * 100
            val_align  = v_metrics.get("index_match_rate", 0.0) * 100
            print(
                f"{epoch:>6} | "
                f"{t_metrics.get('total', 0):>8.4f} | "
                f"{t_metrics.get('rec_eeg', 0):>8.4f} | "
                f"{t_metrics.get('rec_fnirs', 0):>9.4f} | "
                f"{t_metrics.get('contrastive', 0):>8.4f} | "
                f"{t_metrics.get('index_match', 0):>9.4f} | "
                f"{eeg_util:>9.1f}% | "
                f"{fnirs_util:>10.1f}% | "
                f"{val_loss:>9.4f} | "
                f"{val_align:>9.1f}%"
            )

        # ── early stopping ────────────────────────────────────────────────
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    training_time = time.time() - start_time

    # ── final evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Final Test Evaluation")
    print(f"{'='*70}")

    checkpoint = torch.load(ckpt_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, criterion, loaders["test"], device)

    print(f"Test total loss      : {test_metrics.get('total', 0):.4f}")
    print(f"Test rec_eeg MSE     : {test_metrics.get('rec_eeg', 0):.6f}")
    print(f"Test rec_fnirs MSE   : {test_metrics.get('rec_fnirs', 0):.6f}")
    print(f"Test contrastive     : {test_metrics.get('contrastive', 0):.4f}")
    print(f"Test retrieval acc   : {test_metrics.get('retrieval_acc', 0)*100:.1f}%")
    print(f"Test index match     : {test_metrics.get('index_match_rate', 0)*100:.1f}%")
    print(f"EEG  codebook util   : {test_metrics.get('eeg_code_utilization', 0)*100:.1f}%")
    print(f"EEG  perplexity      : {test_metrics.get('eeg_perplexity', 0):.1f}")
    print(f"fNIRS codebook util  : {test_metrics.get('fnirs_code_utilization', 0)*100:.1f}%")
    print(f"fNIRS perplexity     : {test_metrics.get('fnirs_perplexity', 0):.1f}")
    print(f"Training time        : {training_time / 60:.1f} min")

    # ── save results ──────────────────────────────────────────────────────
    results = {
        "args": vars(args),
        "eeg_shape": [C_eeg, T_eeg],
        "fnirs_shape": [C_fnirs, T_fnirs],
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "training_time_s": training_time,
        "best_val_loss": best_val_loss,
        "test_metrics": {
            k: (v.item() if isinstance(v, torch.Tensor) else float(v))
            for k, v in test_metrics.items()
        },
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save training history
    history_serialisable = {}
    for split, records in history.items():
        history_serialisable[split] = [
            {
                k: (v.item() if isinstance(v, torch.Tensor) else float(v))
                for k, v in record.items()
            }
            for record in records
        ]
    with open(run_dir / "history.json", "w") as f:
        json.dump(history_serialisable, f, indent=2)

    print(f"\nResults saved to: {run_dir}")
    tee.close()
    return run_dir


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Shared-Codebook EEG-fNIRS Contrastive Tokenizer"
    )

    # Data
    p.add_argument(
        "--data-root",
        type=str,
        default="data/EEG+NIRS Single-Trial",
        help="Path to the EEG+NIRS dataset root directory",
    )
    p.add_argument("--num-workers", type=int, default=0)

    # Architecture
    p.add_argument("--eeg-patch-size",       type=int,   default=200,
                   help="EEG samples per patch (1 s @ 200 Hz)")
    p.add_argument("--fnirs-patch-size",     type=int,   default=10,
                   help="fNIRS samples per patch (1 s @ 10 Hz)")
    p.add_argument("--eeg-hidden-dim",       type=int,   default=256)
    p.add_argument("--fnirs-hidden-dim",     type=int,   default=128)
    p.add_argument("--num-encoder-layers",   type=int,   default=2)
    p.add_argument("--codebook-size",        type=int,   default=1024,
                   help="Shared codebook size K")
    p.add_argument("--embedding-dim",        type=int,   default=64,
                   help="Token embedding dimension D")

    # VQ hyper-params
    p.add_argument("--vq-beta",  type=float, default=1.0,  help="Commitment loss β")
    p.add_argument("--vq-decay", type=float, default=0.99, help="EMA decay for codebook")

    # Loss weights
    p.add_argument("--contrastive-weight",       type=float, default=1.0,
                   help="InfoNCE contrastive loss weight λ_c")
    p.add_argument("--index-match-weight",       type=float, default=0.5,
                   help="Index-matching loss weight λ_idx")
    p.add_argument("--temperature",              type=float, default=0.1,
                   help="InfoNCE temperature")
    p.add_argument("--index-match-temperature",  type=float, default=0.05,
                   help="Index-matching temperature")

    # Training
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--patience",   type=int,   default=20,
                   help="Early stopping patience (epochs)")
    p.add_argument("--device",     type=str,   default="cuda",
                   choices=["cuda", "cpu"])

    args = p.parse_args()

    # Fallback to CPU if CUDA is not available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    return args


if __name__ == "__main__":
    args = parse_args()
    run_dir = train(args)
    print(f"\nDone. Results in: {run_dir}")
