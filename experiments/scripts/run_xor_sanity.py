#!/usr/bin/env python
"""Run XOR synergy sanity check.

This script trains a minimal ELP-style model on the explicit XOR target:
  - z_u1 depends ONLY on X1
  - z_u2 depends ONLY on X2
  - z_r comes from both but is aligned across modalities
  - z_s is computed from joint information
  - Y is decoded ONLY from z_s

Then it reports:
  - acc_joint(z_s→Y)
  - acc_x1_only(X1→Y)
  - acc_x2_only(X2→Y)
  - acc_token_ablation probes for each token

Usage:
  python experiments/scripts/run_xor_sanity.py --seq-len 64 --epochs 15
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Add project root to path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.synthetic_xor_timeseries import XORTimeseriesConfig, XORTimeseriesDataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class XOR_ELP(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int = 128, token_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len

        # Per-time-step encoders (avoid bottleneck when target is per-step XOR)
        self.enc1_step = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.enc2_step = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.enc_joint_step = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.head_u1 = nn.Linear(hidden_dim, token_dim)
        self.head_u2 = nn.Linear(hidden_dim, token_dim)
        self.head_r1 = nn.Linear(hidden_dim, token_dim)
        self.head_r2 = nn.Linear(hidden_dim, token_dim)
        self.head_s = nn.Linear(hidden_dim, token_dim)

        # Decode per-step logits ONLY from z_s (token_dim -> 1)
        self.dec_y_step = nn.Linear(token_dim, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x1,x2: [B,T] -> [B,T,1]
        h1 = self.enc1_step(x1.unsqueeze(-1))
        h2 = self.enc2_step(x2.unsqueeze(-1))

        z_u1 = self.head_u1(h1)  # [B,T,D]
        z_u2 = self.head_u2(h2)  # [B,T,D]

        z_r1 = self.head_r1(h1)  # [B,T,D]
        z_r2 = self.head_r2(h2)  # [B,T,D]
        z_r = 0.5 * (z_r1 + z_r2)

        h_joint = self.enc_joint_step(torch.cat([h1, h2], dim=-1))
        z_s = self.head_s(h_joint)  # [B,T,D]

        y_logits = self.dec_y_step(z_s).squeeze(-1)  # [B,T]

        return {
            "z_r": z_r,
            "z_r1": z_r1,
            "z_r2": z_r2,
            "z_u1": z_u1,
            "z_u2": z_u2,
            "z_s": z_s,
            "y_logits": y_logits,
        }


def orthogonality_loss(tokens: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    # Average absolute cosine similarity over all pairs (pool over time first)
    loss = 0.0
    count = 0
    pooled = [t.mean(dim=1) if t.dim() == 3 else t for t in tokens]
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            cos = F.cosine_similarity(pooled[i], pooled[j], dim=-1).abs().mean()
            loss = loss + cos
            count += 1
    return loss / max(count, 1)


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    # logits,y: [B, T]
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds.eq(y).float().mean()).item()


class LinearSeqProbe(nn.Module):
    def __init__(self, in_dim: int, seq_len: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,T,D] -> logits: [B,T]
        b, t, d = z.shape
        logits = self.fc(z.reshape(b * t, d)).reshape(b, t)
        return logits


def train_linear_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-2,
) -> float:
    device = z_train.device
    probe = LinearSeqProbe(z_train.shape[-1], y_train.shape[-1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(z_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        acc = accuracy_from_logits(probe(z_test), y_test)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--lambda-orth", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    cfg = XORTimeseriesConfig(n_samples=args.n_samples, seq_length=args.seq_len, seed=args.seed)
    dataset = XORTimeseriesDataset(cfg)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = XOR_ELP(seq_len=args.seq_len, hidden_dim=args.hidden_dim, token_dim=args.token_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print("[Config]", {**asdict(cfg), **{k: getattr(args, k) for k in [
        "epochs", "batch_size", "hidden_dim", "token_dim", "lr", "lambda_align", "lambda_orth"
    ]}})

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        accs = []
        for batch in train_loader:
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            y = batch["y"].to(device)

            out = model(x1, x2)
            l_y = loss_fn(out["y_logits"], y)
            l_align = F.mse_loss(out["z_r1"], out["z_r2"])
            l_orth = orthogonality_loss((out["z_r"], out["z_u1"], out["z_u2"], out["z_s"]))
            loss = l_y + args.lambda_align * l_align + args.lambda_orth * l_orth

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())
            accs.append(accuracy_from_logits(out["y_logits"].detach(), y))

        # Quick eval each epoch
        model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            y = batch["y"].to(device)
            out = model(x1, x2)
            acc_test = accuracy_from_logits(out["y_logits"], y)

        print(f"[Epoch {epoch:02d}] loss={np.mean(losses):.4f} train_acc={np.mean(accs):.4f} test_acc={acc_test:.4f}")

    # Final evaluation on full test set
    model.eval()
    all_acc = []
    with torch.no_grad():
        for batch in test_loader:
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            y = batch["y"].to(device)
            out = model(x1, x2)
            all_acc.append(accuracy_from_logits(out["y_logits"], y))
    acc_joint = float(np.mean(all_acc))

    # Single-modality baselines: train a linear per-sequence classifier from X1->Y and X2->Y
    # (should be ~0.5 for XOR)
    def train_x_only_probe(x_key: str) -> float:
        # collect a single big batch for simplicity (fits in memory for default sizes)
        train_loader2 = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
        test_loader2 = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        btr = next(iter(train_loader2))
        bte = next(iter(test_loader2))
        x_tr = btr[x_key].to(device)
        y_tr = btr["y"].to(device)
        x_te = bte[x_key].to(device)
        y_te = bte["y"].to(device)

        # Per-step logistic: 1 -> 1, applied independently per time step
        probe = nn.Linear(1, 1).to(device)
        optp = torch.optim.AdamW(probe.parameters(), lr=1e-2)
        for _ in range(50):
            optp.zero_grad()
            logits = probe(x_tr.unsqueeze(-1)).squeeze(-1)
            loss = loss_fn(logits, y_tr)
            loss.backward()
            optp.step()
        with torch.no_grad():
            return accuracy_from_logits(probe(x_te.unsqueeze(-1)).squeeze(-1), y_te)

    acc_x1_only = train_x_only_probe("x1")
    acc_x2_only = train_x_only_probe("x2")

    # Token ablation probes (freeze encoder)
    with torch.no_grad():
        train_loader2 = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
        test_loader2 = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        btr = next(iter(train_loader2))
        bte = next(iter(test_loader2))
        x1_tr = btr["x1"].to(device)
        x2_tr = btr["x2"].to(device)
        y_tr = btr["y"].to(device)
        x1_te = bte["x1"].to(device)
        x2_te = bte["x2"].to(device)
        y_te = bte["y"].to(device)

        out_tr = model(x1_tr, x2_tr)
        out_te = model(x1_te, x2_te)

    acc_probe = {
        "z_r": train_linear_probe(out_tr["z_r"], y_tr, out_te["z_r"], y_te),
        "z_u1": train_linear_probe(out_tr["z_u1"], y_tr, out_te["z_u1"], y_te),
        "z_u2": train_linear_probe(out_tr["z_u2"], y_tr, out_te["z_u2"], y_te),
        "z_s": train_linear_probe(out_tr["z_s"], y_tr, out_te["z_s"], y_te),
    }

    print("\n[Results]")
    print(f"acc_joint(z_s→Y)   : {acc_joint:.4f}")
    print(f"acc_x1_only(X1→Y)  : {acc_x1_only:.4f}")
    print(f"acc_x2_only(X2→Y)  : {acc_x2_only:.4f}")
    print("acc_token_ablation :")
    for k, v in acc_probe.items():
        print(f"  {k:4s}: {v:.4f}")


if __name__ == "__main__":
    main()
