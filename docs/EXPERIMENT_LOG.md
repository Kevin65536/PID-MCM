# Experiment Log

> 实验记录文档，按时间倒序记录每次实验的配置、结果和结论。

## ⚠️ Lessons Learned from Pre-Experiments

**Archived Results:** Previous experiment runs and logs have been archived under `experiments/runs/archive/`, `experiments/archive/`, `croce_validation/archive/`, and `docs/archive/logs/`. Historical shared/private stage experiments are available via git history.

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
| 2026-05-14 | EXP-P2B-ARCH-STABLE | Phase 2B | Architecture stabilized: Croce 2017 physical model + coupling structure priors implemented and merged | Completed |
| 2026-06-04 | EXP-CROCE-HIGHWL-LOCAL | Croce local tokenizer | HighWL-only local cache adapter, explicit source/observation targets, Gate0 contract, launch configs, and first training launch | Active |

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
- Best long-warmup run: [experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718](../experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718)
- Reference long run: [experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_long_bs128_20260511_174538](../experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_long_bs128_20260511_174538)
- Archive log: [docs/archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md](archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md)
- Result index: [experiments/archive/results/pre_croce_local_highwl_20260604/source_observation_index.json](../experiments/archive/results/pre_croce_local_highwl_20260604/source_observation_index.json)

### Results

| Artifact | Gate 1 | Gate 2 | Gate 3 | Gate 4 | Best epoch | Best val_loss | Verdict |
|----------|--------|--------|--------|--------|------------|---------------|---------|
| longwarmup 320e | pass | fail | fail | fail | 278 | 1.6395270029703777 | hold_repair |
| long 240e | pass | fail | fail | fail | 171 | 1.6470870176951091 | hold_repair |

### Analysis

- Explicit phase supervision is no longer part of the source/observation mainline. The stable Gate1 baseline now relies on time-domain reconstruction plus FFT amplitude supervision.
- The 320-epoch slow-warmup run is the current best Gate1 handoff because it keeps Gate1 passing while achieving the lowest validated val_loss among the no-phase runs.
- The phase is closed under the formal Phase 1 archive bundle; the reference long run was moved into that bundle during the 2026-06-04 storage cleanup.
- The project is not promotion-ready: Gate2-Gate4 still fail, so the correct next step is Phase 2 HRF source-target work, not more Phase 1 loss churn.

### Conclusion

Phase 1 now has a formal best baseline and a formal archive bundle. Future source/observation work should start from [experiments/configs/source_observation/phase1/gate1_best_current.yaml](../experiments/configs/source_observation/phase1/gate1_best_current.yaml), and any new mainline claim must be compared against this locked handoff artifact.

---

## EXP-P2B-ARCH-STABLE: Phase 2B Architecture Stabilization (2026-05-14)

### Objective

Complete the Croce 2017 physical model source target implementation, integrate coupling structure priors (lag focus + joint smoothness), and declare architecture stable. Mechanism C abandoned; Phase 2C mechanisms deferred to long-term.

### Configuration

- Mainline: `SourceObservationLaBraMVQNSP` in [src/tokenizers/factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py)
- Losses: lag_focus_loss + joint_smoothness_loss in [src/losses/multimodal_tokenizer.py](../src/losses/multimodal_tokenizer.py)
- SMC validation: [src/inference/neurovascular_smc.py](../src/inference/neurovascular_smc.py)
- Spatial priors: [src/data/channel_adjacency.py](../src/data/channel_adjacency.py)
- Architecture changelog: [docs/architecture_changelog/2026-05-13_phase2b_croce2017_physical_model_targets.md](architecture_changelog/2026-05-13_phase2b_croce2017_physical_model_targets.md)

### Key Architecture Decisions

1. Shared neural state s(t) via AR(1) smoothing (α=0.90) drives both modalities
2. EEG source target: signed RMS carrier (μV) — same physical units as raw EEG
3. fNIRS source target: HRF(s(t)) — neurovascular delay absorbed by convolution
4. Coupling priors directly constrain matrix shape without KL data matching
5. 4 independent decoders (source + observation per modality), additive reconstruction
6. Mechanism C (causal asymmetry) abandoned — Croce model provides sufficient directional structure

### Conclusion

Architecture is now stabilized. No further major architectural exploration planned. Current focus: Gate 3 (Structure) validation, diagnostic refinement, and downstream evaluation.

---

## EXP-CROCE-HIGHWL-LOCAL: Croce Local HighWL-Only Tokenizer Setup (2026-06-04)

### Objective

Start the next tokenizer training phase from generated Croce source/observation caches, using local spatial anchors and highWL-only fNIRS input while preserving explicit source/observation targets.

### Configuration

- Dataset adapter: [src/data/croce_local_cache_dataset.py](../src/data/croce_local_cache_dataset.py)
- Training base config: `experiments/configs/source_observation/croce_local/highwl_base.yaml`
- Sweep configs: `highwl_lr2e4.yaml`, `highwl_fnirsobs64.yaml`
- Canonical cache roots:
  - `croce_validation/cache/croce_local/highwl_v1/single_trial_motor_imagery`
  - `croce_validation/cache/croce_local/highwl_v1/single_trial_mental_arithmetic`
  - `croce_validation/cache/croce_local/highwl_v1/simultaneous_cognitive`
- Future run namespace: `experiments/runs/source_observation/croce_local/highwl_v1/<run_name>/`
- Current live run, launched before namespace normalization: `experiments/runs/s2_croce_local_highwl_base_20260604_153549/`
- Storage layout: [docs/STORAGE_LAYOUT.md](STORAGE_LAYOUT.md)

### Key Decisions

1. Tokenizer input is local, not whole-brain: EEG `[B,6,4000]`, fNIRS `[B,1,200]`.
2. fNIRS uses `highWL` only: `source_fnirs_optical_channel_0` and `obs_fnirs_optical_channel_0`.
3. `lowWL` remains in cache metadata and must be reported by Gate0 as ignored.
4. The highWL signal remains optical measurement-space, not HbO concentration.
5. Default codebooks: source K=32, EEG observation K=64, fNIRS observation K=32; sweep fNIRS observation K=64.
6. Default fNIRS lowpass for raw dataset preprocessing is 0.2 Hz; downstream task config absence is ignored in registry tests for this checkout.

### Validation

Smoke dataloader creation on the real cache produced:

| Split | Samples | EEG shape | fNIRS shape | Gate0 pair |
|-------|---------|-----------|-------------|------------|
| train | 90000 | `[B,6,4000]` | `[B,1,200]` | `wavelength: highWL/lowWL` |
| val | 41472 | `[B,6,4000]` | `[B,1,200]` | `wavelength: highWL/lowWL` |
| test | 33048 | `[B,6,4000]` | `[B,1,200]` | `wavelength: highWL/lowWL` |

### Conclusion

The cache/input contract is ready and the first highWL-only local tokenizer run has been launched. Gate0 is now required before interpreting semantic gates.

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
