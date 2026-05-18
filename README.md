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
  tokenizers/         # VQ-VAE, FSQ, NeuroRVQ, LaBraM, Source/Observation tokenizers
  metrics/            # Codebook health and reconstruction metrics
  data/               # Dataset loading, channel adjacency, factory
  models/             # (archived) ELP encoder
  losses/             # Training losses (reconstruction, coupling priors)
  utils/              # Logging, checkpointing, launcher utilities
  visualization/      # Standardized training/post-training visualization
  inference/          # Neurovascular SMC filter (Croce 2017)
  classifiers/        # Downstream classifiers
  foundation/         # Foundation model interface

experiments/
  configs/
    base.yaml         # Base configuration
    source_observation/  # Source/Observation configs (phase1, phase2, phase2a)
    phase0plus/       # Advanced single-modality tokenizer configs
    archive/          # Old experiment configs
  scripts/
    launch_training_nohup.sh  # Canonical training launcher
    train_*.py             # Task-specific training entrypoints
    compare_run_metrics.py # Batch run comparison tool
    probe/                 # Exploratory, non-standardized probe experiments only
    archive/            # Old experiment scripts
  runs/               # Experiment outputs
  results/            # Comparison results

docs/
  ARCHITECTURE.md     # Current architecture specification
  PHYSIOLOGICAL_COUPLING_PLAN.md  # Physiological coupling constraints design
  SEMANTIC_TOKEN_SCORECARD.md     # 4-Gate evaluation framework
  EXPERIMENT_LOG.md   # Formal experiment conclusions
  THEORY.md           # Theoretical background

croce_validation/
  README.md           # Independent workspace for Croce real-data validation on Single-Trial motor imagery
  scripts/            # Standardized validation runners and utilities
  results/            # Validation outputs, manifests, and figures
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

### Compare Completed Runs

Use the batch comparison tool to aggregate frozen configs, best-checkpoint metrics, and Gate summaries across multiple runs. Each invocation now creates a uniquely named report directory under `experiments/comparison_reports/`, containing CSV, JSON, Markdown, trajectory analysis, and PNG visualizations:

```bash
python experiments/scripts/compare_run_metrics.py \
  --glob 'gate1_health_*' \
  --baseline gate1_health_uniform32 \
  --report-name gate1_health_iteration_05
```

The tool reads `metrics.json`, `final_summary.json`, and `analysis/split_<split>.json`, then prints a sortable table with run-level outcomes, key config knobs, Gate1 bottlenecks, and writes a report package like:

```text
experiments/comparison_reports/20260510_221530_gate1_health_iteration_05/
  analysis.json
  report.md
  metadata.json
  summary.csv
  summary.json
  figures/
    best_val_loss_ranking.png
    gate1_health_overview.png
    trajectory_patterns.png
    branch_perplexity_trajectories.png
    stability_overview.png
```

The generated report now includes three layers of evidence in one place: final summary tables, TensorBoard-style multi-metric trajectory plots, and an automatically written pattern-analysis section that groups runs, explains likely causes of the observed differences, and judges whether the modification appears effective relative to the chosen baseline.

You can still pass `--output-csv`, `--output-json`, or `--output-md` if you want extra copies in custom locations, but the report directory is now created automatically on every run.

## Current Status

**Phase 2B: Croce 2017 Physical Model + Coupling Priors** (Current)
- [x] Source/Observation dual-decoder tokenizer
- [x] HRF convolution source target
- [x] Lag focus + joint smoothness coupling priors
- [ ] Gate 3 (Structure) validation
- [ ] Downstream utility evaluation

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
