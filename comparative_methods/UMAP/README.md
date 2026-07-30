# UMAP Comparative Method — EEG + fNIRS Multimodal Fusion

> **Historical-result boundary.** The 2026-03 UMAP best-test scores followed
> repeated test inspection, so they are diagnostic history rather than valid
> final comparison numbers. A new table value must satisfy the lightweight
> [final-number acceptance rules](../../docs/physiology_semantic_tokenizer/13_COMPARATIVE_METHOD_FINAL_METRIC_ACCEPTANCE.md).

## Overview

**UMAP** (Unified Multi-task Pre-training) adapts BLIP-2's Q-Former architecture
for multimodal physiological signal fusion. It uses three pretraining objectives
to align and fuse two modalities, then supports finetuning with missing modalities.

**Paper**: *Multimodal Emotion Recognition with Missing Modality via A Unified
Multi-task Pre-training Framework*

The paper evaluates EEG + eye-tracking emotion classification on
SEED-IV/V/VII with Accuracy and weighted F1. This project replaces eye
tracking with fNIRS and changes the downstream task family, so the paper's
absolute values are `not_applicable` to project MI/WG and other tasks. The
paper-domain relation `paired > EEG-only > eye-only` is a descriptive
original-domain prior only; it is not a constraint on project task ordering.

## Directory Structure

```
UMAP/
├── README.md                 # This file
├── train_umap.py             # Training entrypoint (pretrain + finetune)
├── umap_dataset.py           # Dataset adapter (project data → UMAP format)
├── configs/
│   ├── pretrain.yaml         # Pretraining configuration
│   └── finetune.yaml         # Finetuning configuration
├── model/                    # Original UMAP model code (upstream)
│   ├── __init__.py
│   ├── umap_qformer.py       # Core Q-Former Transformer
│   ├── umap_pretrain.py      # Pretraining wrapper (CON+MAT+GEN)
│   ├── umap_finetune.py      # Finetuning wrapper (classification)
│   ├── umap_utils.py         # Utilities (LR schedule, DDP, metrics)
│   ├── config.py             # Original YAML config loader
│   ├── demo.py               # Original demo script
│   ├── config_pretrain.yaml  # Original pretrain config (EEG+Eye)
│   └── config_finetune.yaml  # Original finetune config (EEG+Eye)
├── runs/                     # Experiment outputs (auto-generated)
│   └── <run_name>/
│       ├── config.json
│       ├── training.log
│       ├── results.json
│       ├── history.json
│       ├── checkpoints/
│       └── plots/
└── paper.pdf                 # Reference paper
```

### Separation Principle

| Layer | Files | Purpose |
|-------|-------|---------|
| **Model** (upstream) | `model/` | Original UMAP code, minimal modifications (only import fixes) |
| **Adapter** (ours) | `umap_dataset.py` | Converts project's `MultiModalEEGfNIRSDataset` → UMAP format |
| **Training** (ours) | `train_umap.py` | Config-driven training with logging, plots, checkpoints |
| **Config** (ours) | `configs/` | YAML configs for our EEG+fNIRS experiments |

## Architecture (from paper)

```
                    ┌───────────────────────────────┐
                    │        Q-Former Encoder        │
                    │                                │
EEG ──► Linear ──► │  Modality-specific FFN (EEG)  │
   + pos_emb       │  Modality-specific FFN (fNIRS) │ ──► Task heads
   + type_emb      │  Shared Self-Attention          │
fNIRS ──► Linear ──►  Fusion FFN (SeqFusion)        │
   + pos_emb       │                                │
   + type_emb      └───────────────────────────────┘
```

**Three pretraining tasks:**

1. **Contrastive (CON)**: Align CLS tokens of both modalities. Uses block-diagonal
   attention mask so modalities don't see each other during encoding.
2. **Matching (MAT)**: Binary classification — are the two modality inputs from the
   same trial? Hard negatives selected by contrastive similarity.
3. **Generation (GEN)**: Causal cross-modal reconstruction — reconstruct modality B
   from modality A using causal attention mask.

**Missing-modality finetuning:**
- Only present modality branch receives input
- Fusion FFN still processes available information
- Classification uses CLS token from available modality

## Modality Mapping

| UMAP Original | Our Adaptation |
|---------------|----------------|
| EEG (310-dim, 5 timepoints) | EEG (30 channels, 5 segments) |
| Eye tracking (33-dim, 5 timepoints) | fNIRS HbO (36 channels, 5 segments) |

## Quick Start

```bash
cd comparative_methods/UMAP

# Pretrain
python train_umap.py pretrain --config configs/pretrain.yaml

# Finetune (multimodal)
python train_umap.py finetune --config configs/finetune.yaml

# Finetune (EEG only — missing fNIRS)
python train_umap.py finetune --config configs/finetune.yaml --modality eeg

# Finetune (fNIRS only — missing EEG)
python train_umap.py finetune --config configs/finetune.yaml --modality eye

# Smoke test
python train_umap.py pretrain --config configs/pretrain.yaml --epochs 3 --run_name smoke
```

## Experiment Plan

See `EXPERIMENT_DESIGN.md` for the full comparative evaluation plan.
