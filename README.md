# PID-MCM Implementation

This directory contains the implementation of the **Explicit Latent Partitioning (ELP)** framework for PID-guided multimodal pretraining of EEG and fNIRS data.

## Directory Structure

- `data/`: Dataset storage (synthetic and raw).
- `src/`: Source code.
  - `models/`: PyTorch models (ELPEncoder, Transformer).
  - `losses/`: Custom loss functions (Alignment, Orthogonality, Synergy).
  - `data/`: Dataset classes and masking logic.
  - `utils/`: Metrics and helper functions.
- `experiments/`: Configuration files and training scripts.
- `notebooks/`: Jupyter notebooks for analysis and visualization.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib

### Phase 1: Synthetic Verification
Run the synthetic data generation and basic model training to verify the ELP logic.

```bash
# Example usage (coming soon)
python -m src.data.synthetic
```
