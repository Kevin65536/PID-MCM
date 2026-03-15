# Theory V2: Interpretable EEG-fNIRS Foundation Model

> Version: 1.0
> Date: 2026-03-15
> Purpose: Define the next-stage theoretical framework after archived tokenizer and baseline experiments.

---

## 1. Problem Restatement

From archived experiments, the key challenge is not a single model component but a coupled failure pattern:

1. Cross-subject generalization collapse:
   - Many settings remain near chance-level on held-out subjects.
2. Subject-specific representation dominance:
   - Models can fit subject identity cues but fail to learn task-invariant features.
3. Temporal-resolution mismatch across modalities:
   - EEG (high-rate) and fNIRS (low-rate, delayed hemodynamics) are hard to align with naive point-wise synchronization.
4. Objective mismatch:
   - Reconstruction and generic alignment losses do not reliably improve downstream MI/emotion discrimination.

The next-stage model must explicitly separate:
- Subject-related factors
- Task-related factors
- Shared cross-modal factors

---

## 2. Core Hypothesis

A robust multimodal foundation model for EEG and fNIRS should be built around structured factorization, not only shared embedding.

Let each sample produce latent variables:
- Zs: subject-specific factors
- Zt: task-relevant factors
- Zm: modality-shared physiological factors

Target properties:
1. Zt should be predictive for downstream tasks and weakly predictive for subject identity.
2. Zs should absorb subject/domain bias and be discouraged from leaking into task heads.
3. Zm should capture cross-modal consistency under temporal-scale mismatch.

---

## 3. Design Principles

### 3.1 Do not flatten too early
Each modality first keeps rich multi-scale token sequences, then applies learned compression (query-based resampling) instead of global pooling at input stage.

### 3.2 Align at multiple scales
Perform alignment at short, medium, and long temporal scales (event-level, segment-level, trial-level), not only at one global embedding.

### 3.3 Separate invariance and predictiveness
Use explicit constraints to push subject invariance in task pathway while preserving modality/subject factors in dedicated branches.

### 3.4 Interpretability is first-class
Every prediction should provide:
- key temporal tokens
- cross-modal attention contribution
- factor-level attribution (subject/task/shared)

---

## 4. Proposed Architecture (Foundation V2)

### 4.1 Modality-specific encoders
- EEG encoder:
  - multi-scale temporal backbone (Transformer/Mamba style)
  - optional electrode graph branch for spatial topology
- fNIRS encoder:
  - low-frequency trend + event-change branch
  - explicit delay-aware temporal encoding

### 4.2 Physio-resampler (learned temporal compression)
For each modality, use a small set of learned temporal queries to extract event-centric summary tokens from long sequences.

### 4.3 Factorized latent heads
From fused representation, branch into:
- Subject head: predicts subject ID from Zs
- Task head: predicts MI/emotion/etc from Zt
- Shared head: cross-modal consistency learning on Zm

### 4.4 Invariance module
Apply adversarial subject confusion on task pathway:
- Task encoder maximizes downstream performance
- Subject discriminator tries to recover subject ID
- Gradient reversal or adversarial loss reduces subject leakage in Zt

### 4.5 Cross-modal multi-scale fusion
Cross-attention blocks at multiple temporal scales:
- short scale: event-level coupling
- medium scale: segment-level interaction
- long scale: trial/session context

---

## 5. Learning Objectives

Overall objective:
L = Lt + lambda_cm * Lcm + lambda_inv * Linv + lambda_rec * Lrec + lambda_fac * Lfac

Where:
- Lt: downstream supervised loss (classification/regression)
- Lcm: cross-modal consistency loss (contrastive or matching)
- Linv: subject-invariance loss (adversarial confusion / mutual-info minimization)
- Lrec: lightweight reconstruction regularizer (not dominant objective)
- Lfac: factor disentanglement regularizer (orthogonality / decorrelation)

Practical recommendation:
- Keep Lrec as regularization term only.
- Prioritize Lt + Lcm + Linv in model selection.

---

## 6. How to identify subject features vs task features

### 6.1 Direct probing
Train linear probes on frozen latent spaces:
- Probe(Zt -> subject): should be low
- Probe(Zt -> task): should be high
- Probe(Zs -> subject): should be high

### 6.2 Mutual-information style diagnostics
Estimate relative dependence:
- I(Zt; Subject) should decrease during training
- I(Zt; Task) should stay high or increase

### 6.3 Representation geometry checks
- Subject clustering score on Zt should drop.
- Task clustering score on Zt should rise.
- For Zs, opposite trend is acceptable.

### 6.4 Counterfactual consistency
Within same task label, replace subject style statistics (normalization/statistical perturbation):
- Prediction should remain stable if model truly relies on task factors.

---

## 7. Temporal and spatial heterogeneity handling

### 7.1 Temporal mismatch strategy
Do not force same-rate alignment.
Use event-aware and multi-scale alignment:
- local event alignment window
- medium segment alignment
- long context alignment

### 7.2 Spatial mismatch strategy
Treat sensors as structured nodes, not simple channels:
- EEG electrode graph / region pooling
- fNIRS channel neighborhood graph
- optional anatomical region-level shared indexing

### 7.3 Text/label semantic anchoring (optional extension)
Use language-conditioned heads for multi-task transfer, with text as semantic anchor for heterogeneous tasks.

---

## 8. Foundation readiness criteria

A candidate model is considered foundation-ready only if all conditions hold:

1. Cross-subject robustness:
   - clear improvement over chance across multiple seeds/folds.
2. Multi-task transfer:
   - pretrain on task A, efficient adaptation to task B with limited labels.
3. Factor interpretability:
   - measurable separation between subject and task factors.
4. Cross-dataset stability:
   - controlled drop when switching datasets/protocols.

---

## 9. Expected failure modes and safeguards

1. Over-invariance collapse:
   - If invariance too strong, task signal is removed.
   - Safeguard: monitor Probe(Zt -> task) and early-stop on joint criterion.
2. Shortcut fusion:
   - Model uses one modality shortcut only.
   - Safeguard: modality-dropout and missing-modality stress test.
3. Alignment without discrimination:
   - High cross-modal similarity but weak class separability.
   - Safeguard: class-conditional alignment terms and balanced metrics.

---

## 10. Immediate implementation guidance

1. Keep DE features as current stable starting point for UMAP-like baselines.
2. Move from reconstruction-first to discriminative+invariance-first objectives.
3. Build factor probes and diagnostics before large architecture expansion.
4. Use full-fold and multi-seed protocol as default for claims.
