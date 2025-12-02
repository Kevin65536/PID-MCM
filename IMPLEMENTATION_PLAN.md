# PID-MCM Implementation Plan

> **Last Updated**: 2025-12-02  
> **Status**: Phase 1 - Foundation  
> **Reference Documents**: See `docs/` for theoretical background

---

## Overview

This implementation realizes the **Explicit Latent Partitioning (ELP)** framework for PID-guided multimodal pretraining. The project is divided into 4 phases, each with clear deliverables and success criteria.

### Quick Reference
- **Theory**: `docs/pid_mcm_proposal.md`
- **Data Guidelines**: `docs/pid_explicit_decomposition_analysis.md`
- **Future Work**: `docs/research_directions.md`

---

## Phase 1: Synthetic Verification (Week 1-2)

### Goal
Validate that the ELP architecture can recover known latent structures on controlled synthetic data.

### Deliverables

#### 1.1 Loss Functions Module (`src/losses/pid_losses.py`)
**Tasks**:
- [ ] Implement `AlignmentLoss`: MSE between $z_r$ from different views
  ```python
  L_align = ||z_r^A - z_r^B||^2
  ```
- [ ] Implement `OrthogonalityLoss`: Cosine similarity penalty
  ```python
  L_orth = |cos(z_r, z_u)| + |cos(z_r, z_s)| + |cos(z_u, z_s)|
  ```
- [ ] Implement `SynergyLoss`: Penalty for lack of change under masking
  ```python
  L_syn = -||z_s^{joint} - z_s^{single}||^2
  ```
- [ ] Implement `ReconstructionLoss`: Standard MSE for signal reconstruction

**Success Criteria**:
- All losses are differentiable
- Gradient flow verified with toy examples

#### 1.2 Masking Strategy (`src/data/masking.py`)
**Tasks**:
- [ ] Implement `CrossModalMask`: Mask 80% of X1, keep X2 full
- [ ] Implement `UniModalMask`: Mask 50% of X1, drop X2 entirely
- [ ] Implement `JointMask`: Random 50% masking on both modalities
- [ ] Create `MixedBatchSampler`: 25% cross, 25% uni, 50% joint

**Success Criteria**:
- Batch composition matches intended proportions
- Masking is random and reproducible with seed

#### 1.3 Training Loop (`experiments/train_synthetic.py`)
**Tasks**:
- [ ] Load `PIDSyntheticDataset`
- [ ] Initialize `ELPEncoder`
- [ ] Combine losses with weights: $\lambda_1=0.5, \lambda_2=0.3, \lambda_3=0.2$
- [ ] Train for 100 epochs with Adam optimizer
- [ ] Log losses to TensorBoard/WandB

**Success Criteria**:
- Training converges (loss decreases)
- No NaN or gradient explosion

#### 1.4 Evaluation (`notebooks/phase1_analysis.ipynb`)
**Tasks**:
- [ ] Extract learned $z_r, z_u, z_s$ from trained model
- [ ] Compute correlation with ground truth latents $w_r, w_{u1}, w_{u2}, w_s$
- [ ] Visualize with t-SNE/PCA
- [ ] Measure HSIC (Hilbert-Schmidt Independence Criterion) between tokens

**Success Criteria**:
- $\text{corr}(z_r, w_r) > 0.7$
- $\text{HSIC}(z_r, z_u) < 0.1$ (low dependence)
- Redundancy token aligns across modalities

---

## Phase 2: Real-World Data Pipeline (Week 3-4)

### Goal
Prepare simultaneous EEG-fNIRS datasets for training.

### Deliverables

#### 2.1 Dataset Selection
**Candidates**:
1. **OpenBMI** - Motor imagery tasks
2. **Shin et al. (2018)** - Mental arithmetic (N-back)
3. **BIP Datasets** - Standard BCI benchmarks

**Tasks**:
- [ ] Download datasets and verify integrity
- [ ] Inspect data formats (`.mat`, `.fif`, `.npy`)
- [ ] Check alignment between EEG and fNIRS timestamps

#### 2.2 Preprocessing Pipeline (`src/data/real_data.py`)
**EEG Preprocessing**:
- [ ] Bandpass filter: 1-50 Hz
- [ ] Downsample to 200 Hz
- [ ] Remove artifacts (ICA or simple thresholding)
- [ ] Epoch into 2-second windows

**fNIRS Preprocessing**:
- [ ] Convert raw intensity to HbO/HbR
- [ ] Bandpass filter: 0.01-0.2 Hz
- [ ] Resample to align with EEG epochs
- [ ] Handle motion artifacts

**Tasks**:
- [ ] Implement `EEGPreprocessor` class
- [ ] Implement `fNIRSPreprocessor` class
- [ ] Create `MultimodalDataset` that loads aligned pairs
- [ ] Split into train/val/test (70/15/15)

**Success Criteria**:
- No data leakage across splits
- Aligned time windows (visual inspection)
- Reasonable SNR (>5 dB for EEG)

---

## Phase 3: Model Training (Week 5-6)

### Goal
Train ELP on real data and tune hyperparameters.

### Deliverables

#### 3.1 Model Enhancements
**Tasks**:
- [ ] Add positional encoding to handle temporal sequences
- [ ] Implement MAE-style masking (learnable `[MASK]` token)
- [ ] Add reconstruction decoder (separate heads for EEG/fNIRS)
- [ ] Implement stop-gradient for $z_r$ during unique learning

#### 3.2 Hyperparameter Tuning
**Grid Search**:
- Hidden dim: `[128, 256, 512]`
- Num layers: `[2, 4, 6]`
- Loss weights: $\lambda_1 \in [0.3, 0.5, 0.7]$

**Tasks**:
- [ ] Create config files (YAML) for each setting
- [ ] Run grid search with early stopping
- [ ] Select best model based on validation reconstruction loss

#### 3.3 Training Script (`experiments/train_real.py`)
**Tasks**:
- [ ] Load `MultimodalDataset`
- [ ] Train for 200 epochs with cosine annealing LR schedule
- [ ] Save checkpoints every 10 epochs
- [ ] Log metrics: loss breakdown, HSIC between tokens

**Success Criteria**:
- Reconstruction MSE < 0.5 (normalized data)
- $\text{HSIC}(z_r, z_u) < 0.15$
- Model generalizes to validation set

---

## Phase 4: Evaluation & Baselines (Week 7-8)

### Goal
Demonstrate ELP superiority over baselines.

### Deliverables

#### 4.1 Baselines
**Implement**:
1. **Vanilla MAE**: Single `[CLS]` token, no latent partitioning
2. **Contrastive**: CLIP-style alignment (captures only $R$)
3. **CCA**: Linear baseline for comparison

**Tasks**:
- [ ] Train each baseline with same data/budget
- [ ] Extract representations
- [ ] Freeze and use for downstream tasks

#### 4.2 Downstream Tasks
**Tasks**:
1. **Mental Arithmetic Classification**: Predict N-back level (0/2/3)
2. **Motor Imagery**: Predict left/right hand movement

**Evaluation**:
- [ ] Freeze ELP encoder and train linear classifier on $[z_r, z_u, z_s]$
- [ ] Compare with baselines on accuracy/F1

#### 4.3 PID Analysis
**Tasks**:
- [ ] Estimate PID using PIDF on extracted representations
- [ ] Compare with PID on raw signals
- [ ] Verify $z_r$ captures high redundancy, $z_s$ captures synergy

**Metrics**:
- Redundancy: $I(z_r^{EEG}; z_r^{fNIRS})$
- Unique: $I(z_{u\_eeg}; X_{eeg} | z_r) - I(z_{u\_eeg}; X_{fnirs} | z_r)$
- Synergy: Improvement in task performance when adding $z_s$

#### 4.4 Ablation Studies
**Tasks**:
- [ ] Train without $\mathcal{L}_{orth}$: Does it hurt disjointness?
- [ ] Train without $\mathcal{L}_{syn}$: Does $z_s$ lose meaning?
- [ ] Use only Cross-Modal masking: Can it learn all components?

---

## Implementation Checklist

### Week 1
- [/] Set up directory structure
- [/] Implement `PIDSyntheticDataset`
- [/] Implement `ELPEncoder` skeleton
- [ ] Implement all loss functions

### Week 2
- [ ] Implement masking strategies
- [ ] Complete training loop for synthetic data
- [ ] Run Phase 1 evaluation

### Week 3-4
- [ ] Download and preprocess real datasets
- [ ] Create `MultimodalDataset`
- [ ] Validate data quality

### Week 5-6
- [ ] Enhance model architecture
- [ ] Run hyperparameter search
- [ ] Train final model on real data

### Week 7-8
- [ ] Implement and train baselines
- [ ] Run downstream task evaluation
- [ ] Complete PID analysis and ablations

---

## Debugging & Validation Strategy

### Common Issues
1. **Gradient Vanishing**: Check loss scales, use gradient clipping
2. **Token Collapse**: All tokens become identical
   - **Fix**: Increase $\lambda_{orth}$, use stop-gradient
3. **Synergy Token Ignored**: Model doesn't use $z_s$
   - **Fix**: Increase $\lambda_{syn}$, add synergy-specific tasks

### Validation Checkpoints
- After every 10 epochs: Visualize token distances (should increase)
- After training: Check if $z_r$ is same for different views of same sample
- Before real data: Ensure synthetic experiment succeeds

---

## Success Metrics Summary

| Phase | Metric | Target |
|:------|:-------|:-------|
| Phase 1 | $\text{corr}(z_r, w_r)$ | > 0.7 |
| Phase 1 | $\text{HSIC}(z_r, z_u)$ | < 0.1 |
| Phase 3 | Reconstruction MSE | < 0.5 |
| Phase 4 | Classification Accuracy | > Baselines |
| Phase 4 | PID Synergy (θ) | Detectable in $z_s$ |

---

## Next Immediate Steps

1. Implement `src/losses/pid_losses.py`
2. Complete `src/data/masking.py`
3. Run `experiments/train_synthetic.py`
4. Validate with `notebooks/phase1_analysis.ipynb`
