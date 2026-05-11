# Experiment Log

> 实验记录文档，按时间倒序记录每次实验的配置、结果和结论。

## ⚠️ Lessons Learned from Pre-Experiments

**Archived Results:** Previous experiment runs and logs have been archived to `docs/archive/logs/ARCHIVED_PRE_EXPERIMENTS.md` and `experiments/runs/archive/pre_experiments`.

**Key Bottleneck:**
The downstream Motor Imagery (MI) classification task suffered from a severe lack of cross-subject generalization (hovering around ~50% accuracy, essentially chance level for binary classification), despite performing reasonably well on within-subject tests. 

**Root Causes & Observations:**
1. Tokenizers (e.g., VQ-VAE, FSQ, LaBraM VQNSP) tend to encode subject-specific identity features rather than generalized semantic MI features.
2. The extreme inter-subject variability in EEG/fNIRS signals makes standard training overfit to the training subjects.

**Strategies for Future First Stage Experiments:**
- Explore advanced domain adaptation or alignment techniques to remove subject-specific features.
- Consider utilizing larger, more diverse datasets.
- Implement stronger data augmentation strategies specifically aimed at cross-subject invariance.
- Re-evaluate the tokenizer training objective to encourage learning generalized representations instead of perfect reconstruction, which may be forcing the model to remember subject identity.

---

## Experiment Index

| Date | ID | Phase | Description | Status |
|------|----|-------|-------------|--------|
| 2026-05-11 | EXP-P1-GATE1-LOCK | Phase 1 | Source/observation Gate1 stabilization, best-baseline lock, and archive handoff | Completed |

---

UMAP-related experiment records have been moved to:
- comparative_methods/UMAP/EXPERIMENT_SUMMARY.md

This project-level log now tracks only project-wide milestones.

---

## EXP-P1-GATE1-LOCK: Phase 1 Gate1 Stabilization and Archive Handoff (2026-05-11)

### Objective

Close the Phase 1 Gate1 tuning loop, mark the best no-phase baseline, and archive the full search surface in a way that preserves run provenance.

### Configuration

- Locked baseline: [experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml](../experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml)
- Best config alias: [experiments/configs/source_observation/phase1/gate1_best_current.yaml](../experiments/configs/source_observation/phase1/gate1_best_current.yaml)
- Best long-warmup run: [experiments/runs/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718](../experiments/runs/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718)
- Reference long run: [experiments/runs/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_long_bs128_20260511_174538](../experiments/runs/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_long_bs128_20260511_174538)
- Archive log: [docs/archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md](archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md)
- Result index: [experiments/results/source_observation_index.json](../experiments/results/source_observation_index.json)

### Results

| Artifact | Gate 1 | Gate 2 | Gate 3 | Gate 4 | Best epoch | Best val_loss | Verdict |
|----------|--------|--------|--------|--------|------------|---------------|---------|
| longwarmup 320e | pass | fail | fail | fail | 278 | 1.6395270029703777 | hold_repair |
| long 240e | pass | fail | fail | fail | 171 | 1.6470870176951091 | hold_repair |

### Analysis

- Explicit phase supervision is no longer part of the source/observation mainline. The stable Gate1 baseline now relies on time-domain reconstruction plus FFT amplitude supervision.
- The 320-epoch slow-warmup run is the current best Gate1 handoff because it keeps Gate1 passing while achieving the lowest validated val_loss among the no-phase runs.
- The phase is closed as “archive in place” rather than by moving run directories, because existing analysis artifacts and comparison reports already encode their original paths.
- The project is not promotion-ready: Gate2-Gate4 still fail, so the correct next step is Phase 2 HRF source-target work, not more Phase 1 loss churn.

### Conclusion

Phase 1 now has a formal best baseline and a formal archive bundle. Future source/observation work should start from [experiments/configs/source_observation/phase1/gate1_best_current.yaml](../experiments/configs/source_observation/phase1/gate1_best_current.yaml), and any new mainline claim must be compared against this locked handoff artifact.

---

## [Template] EXP-XXX: [Title] (YYYY-MM-DD)

### Objective
[What is the goal of this experiment?]

### Configuration
[Key differences from baseline, file paths to config, model parameters, etc.]

### Results
[Tables, metrics, confusion matrices, etc.]

### Analysis
[Why did we get these results? Deep dive into the data.]

### Conclusion
[Final takeaway and next steps.]
