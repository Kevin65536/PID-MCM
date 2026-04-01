#!/usr/bin/env python
"""Minimal training entry for the unified multimodal token foundation interface."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.foundation import (
    FoundationModelConfig,
    TokenBatchAdapter,
    UnifiedMultimodalFoundationModel,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: dict) -> UnifiedMultimodalFoundationModel:
    model_cfg = config["model"]
    foundation_cfg = FoundationModelConfig(
        eeg_vocab_size=int(model_cfg["eeg_vocab_size"]),
        fnirs_vocab_size=int(model_cfg["fnirs_vocab_size"]),
        max_seq_len=int(model_cfg.get("max_seq_len", 2048)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        pad_token_id=int(model_cfg.get("pad_token_id", 0)),
        contrastive_temperature=float(model_cfg.get("contrastive_temperature", 0.07)),
    )
    return UnifiedMultimodalFoundationModel(foundation_cfg)


def run_dummy_step(config: dict):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    model.train()

    adapter = TokenBatchAdapter(
        pad_token_id=int(config["model"].get("pad_token_id", 0)),
        mask_token_id=int(config["model"].get("mask_token_id", 1)),
    )

    batch_size = int(config.get("debug", {}).get("batch_size", 4))
    eeg_seq_len = int(config.get("debug", {}).get("eeg_seq_len", 128))
    fnirs_seq_len = int(config.get("debug", {}).get("fnirs_seq_len", 128))

    eeg_tokens = torch.randint(2, int(config["model"]["eeg_vocab_size"]), (batch_size, eeg_seq_len), device=device)
    fnirs_tokens = torch.randint(2, int(config["model"]["fnirs_vocab_size"]), (batch_size, fnirs_seq_len), device=device)
    labels = torch.randint(0, int(config["model"].get("num_classes", 2)), (batch_size,), device=device)

    batch = adapter.from_token_ids(eeg_tokens, fnirs_tokens, labels=labels).to(device)

    pretrain_cfg = config.get("pretraining", {})
    masked_batch, targets = adapter.create_mlm_inputs(
        batch,
        mask_ratio=float(pretrain_cfg.get("mask_ratio", 0.15)),
    )
    targets = targets.to(device)

    pretrain_out = model.forward_pretrain(
        batch=masked_batch,
        targets=targets,
        lambda_mlm=float(pretrain_cfg.get("lambda_mlm", 1.0)),
        lambda_contrastive=float(pretrain_cfg.get("lambda_contrastive", 1.0)),
    )

    downstream_out = model.forward_downstream(batch=batch, labels=labels)

    print("[Debug] pretrain loss:", float(pretrain_out.loss.item()) if pretrain_out.loss is not None else None)
    print("[Debug] downstream loss:", float(downstream_out.loss.item()) if downstream_out.loss is not None else None)
    if "accuracy" in downstream_out.metrics:
        print("[Debug] downstream acc:", float(downstream_out.metrics["accuracy"].item()))


def main():
    parser = argparse.ArgumentParser(description="Run unified multimodal foundation interface debug step")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/phase1a/foundation_multimodal_interface.yaml",
        help="Path to the foundation interface yaml config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))
    run_dummy_step(config)


if __name__ == "__main__":
    main()
