# Neuro-Tokenization

Tokenizing EEG and fNIRS neural signals into discrete codebook representations.

## Overview

This project aims to learn **discrete token representations** (codebooks) for EEG and fNIRS signals. The core idea is to map continuous neural signals to a finite set of reusable "neural tokens", enabling:

- **Robust representations**: Discrete tokens are more stable than continuous embeddings
- **Cross-modal analysis**: Compare EEG and fNIRS in a shared token space
- **Downstream tasks**: Use tokens for classification, retrieval, and interpretation

## Directory Structure

```
src/
  tokenizers/         # FSQ and VQ-VAE tokenizer implementations
  metrics/            # Codebook health and reconstruction metrics
  data/               # Dataset loading (placeholder + real data)
  models/             # (archived) ELP encoder
  losses/             # Training losses
  utils/              # Logging and utilities
  visualization/      # Standardized training/post-training visualization

experiments/
  configs/
    base.yaml         # Base configuration
    phase0/           # Phase 0 experiments (real data tokenization)
    archive/          # Old PID experiment configs
  scripts/
    launch_training_nohup.sh  # Canonical training launcher
    train_*.py             # Task-specific training entrypoints
    analyze_alignment.py   # Standardized manual analysis rerun
    probe/                 # Exploratory, non-standardized probe experiments only
    archive/            # Old experiment scripts
  runs/               # Experiment outputs
  results/            # Comparison results

docs/
  THEORY.md           # Theoretical background
  THEORY_v1_ELP.md    # Archived ELP/PID theory
```

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib, PyYAML

### Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch numpy scipy matplotlib pyyaml
```

### Launch Training

```bash
# Source/observation multimodal tokenizer
bash experiments/scripts/launch_training_nohup.sh \
  --task source-observation-tokenizer \
  --config debug/simultaneous_nback_short_train.yaml

# Single-modality tokenizer
bash experiments/scripts/launch_training_nohup.sh \
  --task tokenizer \
  --config phase0plus/eeg_labram_vqnsp.yaml

# Downstream classifier
bash experiments/scripts/launch_training_nohup.sh \
  --task downstream \
  --config phase1a/P1A_eeg_classification.yaml
```

The launcher prints the background PID and command. Training logs remain available inside each run directory, for example `experiments/runs/<run_name>/training.log`.

For the full task list and task-specific parameters, see [docs/TRAIN_LAUNCH_STANDARD.md](docs/TRAIN_LAUNCH_STANDARD.md).

Direct execution of `experiments/scripts/train_*.py` is intentionally disabled to avoid multiple competing launch paths.

## Current Status

**Phase 0: Real Data Tokenization** (Active)
- [ ] Data loading for real EEG/fNIRS
- [ ] FSQ tokenizer validation
- [ ] VQ-VAE comparison
- [ ] Cross-subject generalization

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed roadmap.

## Key Concepts

### Tokenization

```
Input x [B, T]
    → Encoder
    → Continuous Latent z [B, T', D]
    → Quantizer (FSQ or VQ)
    → Token Indices q [B, T'] ∈ {1, ..., K}
    → Decoder
    → Reconstruction x̂ [B, T]
```

### Codebook Health

A healthy codebook should:
- **High perplexity**: Codes are used uniformly (not collapsed)
- **Good utilization**: Most codes are active
- **Few dead codes**: No unused codes

### Quantization Methods

| Method | Pros | Cons |
|--------|------|------|
| **FSQ** | No collapse, simple gradients | Limited expressiveness |
| **VQ-VAE** | Flexible, learnable codebook | Needs EMA/reset for stability |

## References

- van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE), 2017
- Mentzer et al., "Finite Scalar Quantization" (FSQ), 2023
- Défossez et al., "High Fidelity Neural Audio Compression" (EnCodec), 2022
