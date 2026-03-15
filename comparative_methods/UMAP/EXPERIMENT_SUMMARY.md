# UMAP Experiment Summary

> UMAP comparative track summary for EEG+fNIRS motor imagery experiments.

## Scope
This document summarizes the UMAP experiment process, current performance bottlenecks, likely root causes, and next-step solutions.

## Experiment Timeline

### UMAP-01 (2026-03-13): Baseline Reproduction
- Goal: establish UMAP baseline under cross-subject split.
- Setup: train subjects 1-20, val 21-25, test 26-29.
- Findings:
  - Most runs stayed near chance-level accuracy (~50%).
  - Pretraining did not improve downstream test metrics.
  - A key checkpoint-loading prefix issue was fixed (UMAP.* -> Qformer.*), but final performance remained weak.

### UMAP-02 (2026-03-14): Comprehensive Visualization and Re-evaluation
- Goal: diagnose failure modes using richer visual and structural analysis.
- Added visualization suite in umap_plots.py, including:
  - attention heatmaps and summary
  - per-head cross-modal attention
  - fusion gate weight distribution
  - embedding t-SNE and modality alignment
  - confusion matrix and classification dashboard
  - gradient-flow and training-dynamics diagnostics
- Findings:
  - Cross-modal attention existed but was weakly informative.
  - Embeddings were dominated by subject identity rather than class.
  - Classification outputs showed strong class bias in several runs.
  - Pretraining still did not yield robust gain.

### UMAP-03 (2026-03-15): Subject-Dependent and Feature-Mode Expansion
- Goal: test whether protocol and feature changes unlock performance.
- Experiments:
  - subject-dependent finetune (channel_avg)
  - cross-subject finetune (band_power)
  - cross-subject finetune (DE)
- Findings:
  - DE (cross-subject, no pretrain) produced the best observed UMAP result:
    - test accuracy: 55.83%
    - macro-F1: 0.558
  - band_power degraded performance.
  - single-fold subject-dependent setup remained around chance.

### UMAP-04 (2026-03-15): DE Base Pretrain + Subject-Dependent Finetune
- Goal: test an upper-bound style pipeline (DE pretrain base model + subject-dependent finetune).
- Pipeline:
  - pretrain with DE features (best checkpoint retained)
  - subject-dependent finetune initialized from the DE pretrain checkpoint
- Findings:
  - test accuracy: 49.66%
  - macro-F1: 0.479
  - validation peaked early then degraded, indicating overfitting.

## Current Best UMAP Result
- Run: U5-FT-de-multi-np
- Setting: cross-subject, multimodal, DE features, no pretrain
- Performance:
  - test accuracy: 55.83%
  - macro-F1: 0.558

## Current Performance Bottlenecks

### 1. Cross-Subject Generalization Gap
- The dominant limitation is still subject shift.
- Even when validation improves, test performance often stays near chance for many configurations.

### 2. Weakly Informative Cross-Modal Fusion
- Cross-modal attention maps are present, but quantitative summaries are often close to uniform mixing.
- This suggests the model attends across modalities without learning reliably discriminative fusion patterns.

### 3. Representation Misalignment with Task Objective
- Embeddings frequently cluster by subject identity, not by motor imagery class.
- Paired modality alignment in weaker runs does not translate into class-separable representations.

### 4. Classification Instability and Bias
- Confusion-matrix patterns in several runs show one-class preference.
- Performance variation across runs/protocols indicates optimization is not robust.

### 5. Pretrain-Objective Mismatch
- Pretraining optimizes alignment/matching/generation objectives.
- These objectives do not consistently improve downstream cross-subject MI classification.

## Likely Root Causes
- Objective mismatch between pretraining tasks and final cross-subject discriminative target.
- Insufficient subject-invariant constraints during representation learning.
- Model can fit training data but fails to learn transferable decision boundaries.
- Single-fold subject-dependent evaluation is too unstable to act as a reliable upper bound.

## Future Solutions

### A. Evaluation Protocol Upgrades
- Run full subject-dependent multi-fold evaluation (not single-fold).
- Report mean/std across folds and seeds for stable comparison.

### B. Representation-Level Improvements
- Keep DE as default feature mode for UMAP comparisons.
- Add explicit subject-invariance regularization or domain-alignment loss.
- Add class-discriminative constraints during pretraining or warm-start finetuning.

### C. Fusion Mechanism Stabilization
- Introduce constraints that discourage near-uniform cross-modal attention.
- Regularize SeqFusion gate behavior to avoid unstable layer-wise collapse.

### D. Training Robustness
- Multi-seed sweeps for each key setting.
- Stronger anti-overfitting schedule for subject-dependent runs:
  - lower LR
  - stronger regularization
  - tighter early stopping and model selection criteria

## Recommended Next Experiments
1. DE + cross-subject multi-seed rerun (baseline stabilization).
2. DE + subject-dependent full-fold evaluation (formal upper-bound estimate).
3. DE + subject-invariance objective ablation (verify bottleneck hypothesis).

## Key Artifact Paths
- Runs: comparative_methods/UMAP/runs/
- Plot code: comparative_methods/UMAP/umap_plots.py
- Training entry: comparative_methods/UMAP/train_umap.py
- Configs: comparative_methods/UMAP/configs/
