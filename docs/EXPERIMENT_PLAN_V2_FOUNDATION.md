# Experiment Plan V2: EEG-fNIRS Foundation Model

> Version: 1.0
> Date: 2026-03-15
> Theory: docs/THEORY_V2_FOUNDATION_MODEL.md

---

## 1. Goal

Build an interpretable multimodal foundation model for EEG and fNIRS that:
1. generalizes across subjects and datasets,
2. supports multiple downstream tasks (MI, emotion classification, etc.),
3. explicitly separates subject factors from task factors.

---

## 2. Phase Structure

## Phase A: Diagnostic Infrastructure (must-have)

### A1. Factor probes
Implement probe pipeline for Zs, Zt, Zm:
- Probe(Zt -> task)
- Probe(Zt -> subject)
- Probe(Zs -> subject)

Deliverables:
- scripts under experiments/scripts/
- report table per run

Success criteria:
- Probe(Zt -> subject) decreases versus baseline
- Probe(Zt -> task) does not collapse

### A2. Metrics expansion
Add mandatory metrics in all downstream runs:
- balanced accuracy
- macro-F1
- per-subject mean/std
- class recall balance
- calibration (ECE optional first pass)

Deliverables:
- unified metrics json schema
- aggregated summary script

---

## Phase B: Foundation V0 (minimal architecture risk)

### B1. Baseline backbone with DE features
- Start with current robust setting: DE feature pipeline.
- Keep architecture close to existing stack; add only factorized heads + invariance loss.

### B2. Subject-invariance branch
- Add subject discriminator with gradient reversal.
- Apply only to task pathway (Zt).

### B3. Multi-objective training
Train with:
- Lt (task)
- Lcm (cross-modal consistency)
- Linv (subject invariance)
- lightweight Lrec

Ablation matrix:
1. Lt only
2. Lt + Lcm
3. Lt + Linv
4. Lt + Lcm + Linv
5. full objective

Success criteria:
- cross-subject test accuracy > current robust baseline
- macro-F1 and class recall balance improve simultaneously
- subject leakage in Zt decreases

---

## Phase C: Foundation V1 (multi-scale temporal fusion)

### C1. Physio-resampler
- Add learned temporal query compression for each modality.
- Keep fixed query count per scale initially.

### C2. Multi-scale cross-attention
- short / medium / long temporal fusion blocks.
- compare against single-scale fusion.

Ablations:
1. single-scale fusion
2. multi-scale fusion without invariance
3. multi-scale fusion with invariance

Success criteria:
- improved robustness under missing modality
- improved transfer to a second downstream task

---

## Phase D: Cross-task and cross-dataset transfer

### D1. Task transfer
- pretrain/foundation on MI + auxiliary objectives
- finetune on emotion task (or inverse direction)

### D2. Cross-dataset transfer
- train on one dataset split/protocol
- evaluate adaptation cost on another dataset/protocol

Success criteria:
- fewer-shot finetuning reaches strong baseline quickly
- lower transfer performance drop than non-foundation baselines

---

## 3. Experimental Protocol Standards

1. Multi-seed default:
- at least 3 seeds for all key claims

2. Subject-dependent claims:
- full-fold protocol required (single fold is exploratory only)

3. Model selection:
- avoid selecting only by val accuracy
- use joint criterion:
  - val task metric
  - subject leakage indicator
  - class balance indicator

4. Reporting format:
- mean ± std across seeds/folds
- include confusion matrix and per-subject statistics

---

## 4. Immediate Task List (next 2 weeks)

## Week 1
1. Implement factor probes and unified metrics.
2. Add subject discriminator + gradient reversal in existing pipeline.
3. Run Phase B ablations on DE features (cross-subject).

## Week 2
1. Add Physio-resampler (single-scale first).
2. Add multi-scale fusion variant.
3. Run missing-modality stress test and compare stability.

---

## 5. Decision Gates

Gate G1 (after Phase B):
- If no robust improvement over DE baseline, prioritize diagnostics and objective redesign before architecture scaling.

Gate G2 (after Phase C):
- If multi-scale improves only val but not test stability, increase regularization and simplify fusion depth.

Gate G3 (after Phase D):
- Promote to "foundation candidate" only if transfer + interpretability criteria both pass.

---

## 6. Risk Register

1. Risk: over-invariance hurts task learning
- Mitigation: gradual lambda_inv schedule + probe monitoring

2. Risk: fusion overfits one modality
- Mitigation: modality dropout and per-modality performance tracking

3. Risk: reproducibility issues across seeds
- Mitigation: fixed reporting template and strict run registry

---

## 7. Deliverable Checklist

1. Theory document update
- docs/THEORY_V2_FOUNDATION_MODEL.md

2. Plan document update
- docs/EXPERIMENT_PLAN_V2_FOUNDATION.md

3. Code milestones
- factor probe script
- invariance branch module
- multi-scale fusion module

4. Reporting milestones
- standardized result table
- failure-case appendix with confusion and per-subject stats
