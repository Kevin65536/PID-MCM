# LC-SPVQ implementation and smoke-validation record

**Date:** 2026-08-20
**Status:** implementation and measured-smoke gates passed; scientific full training not started
**Analysis class:** exploratory development
**Protected cohorts:** closed

## Scope and claim boundary

This record covers the first LC-SPVQ development generation: an existing-export lag probe, the continuous B0 baseline, matched-pair M1, the deranged-pair N1 training null, coupling/proper-score analysis, tests, and reproducible smoke artifacts. It does **not** report a completed scientific model comparison.

The completed legacy continuous shared/private result remains **2/16 success criteria** and is unchanged. Subjects 19–23 (and VP019–VP023) have already influenced checkpoint selection and method development; all new results on them are post-selection development. The protected Single-Trial subjects 24–29 and Simultaneous subjects VP024–VP026 were not measured or opened.

## Implemented estimand and architecture

- EEG and fNIRS have independent K16, D64 EMA codebooks; code IDs are modality-specific and are never swapped, aligned by identity, or interpreted as the same state.
- Positive lag means `fNIRS target time − EEG source time`; the registered bank is 0, 2, 4, 6, 8, and 10 s.
- M1/N1 shared encoders are local/causal: EEG sees the current plus one preceding patch; fNIRS sees the current plus two preceding patches. Private encoders remain full-window bidirectional.
- Shared continuous pre-VQ tokens predict modality-native targets. Raw reconstruction uses `[stop_gradient(shared), private]`, preventing the raw objective from turning the shared branch into a waveform autoencoder.
- Stage A learns continuous native/raw/lag objectives. Stage B initializes both codebooks from all admitted fit-parameter continuous latents, then anneals posterior temperature and quantization strength. Stage C freezes representations/codebooks and fits coupling, shared-marginal, and private task-head contributions.
- The rank-8 coupling head consumes K16 posteriors directly and uses lag-specific low-rank factors only at registered non-negative lags. The registered primary combined head is coupling plus private; shared marginals are a diagnostic ablation.
- N1 replaces the positive EEG–fNIRS training pair with a deterministic same-subject, same-condition, nonidentity donor while preserving the same architecture and optimization stages. New-model donor construction rejects 20 s window overlap on either the EEG or fNIRS event clock within a record and fails closed if a complete group permutation is impossible.
- The lag loss admits in-batch negatives only when subject and condition match, physical trials differ, and relative token times match; separately appended donor banks use only the registered row-aligned donor at the same token time in both directions.

## Software and synthetic gates

The complete targeted suite passed: **92 tests**. Covered contracts include:

1. 20 s windows, ten 2 s patches, immutable metadata-first protected filtering, complete same-group derangements, and independently verifiable EEG/fNIRS nonoverlap;
2. train-only channel/native standardization, native-feature coordinates, and per-sample channel-valid aggregation;
3. future perturbations cannot affect earlier local/causal shared tokens;
4. raw gradients do not enter shared encoders, while native and lag losses do;
5. invalid tokens/channels/coordinates are removed and admitted non-finite predictions fail closed;
6. independent K16 codebooks and fit-parameter-only K-means initialization;
7. a synthetic two-patch delay is recovered by the learnable lag mixture with greater than 0.95 weight on the true lag;
8. soft co-occurrence, conditional log-lift, deranged residuals, train-only ordering, q0/q1 log-loss/Brier gains, and subject-block bootstrap interfaces;
9. repository-bound public APIs, opaque prepared-data governance capabilities, paired-export identity checks, transitive runtime/cache provenance, deterministic manifests, measured protected-access counters, and atomic publication.

The full repository regression produced **540 passed, 10 failed, 11 deselected**. All 10 failures are the existing fail-closed R1P prevalidation-seal checks and report the same already-changed sealed source, `experiments/evaluate_adaptive_shared_neural_ssm.py`. No LC-SPVQ test failed. The historical R1P seal was not weakened or rewritten to make these checks pass.

## Existing-export lag probe

The frozen legacy continuous checkpoint was evaluated without encoder retraining. Because the original run did not retain fit latents, fit representations were re-derived by eval-only forward passes from the manifest-bound source arrays and frozen checkpoints. The probe never used `target`, `eeg_driver`, or `fnirs_driver`; `target_mask` was validity-only.

The full probe evaluated 12 cells (four tasks × three seeds), two representation pairs, eight fixed lags from −4 to +10 s, and matched/deranged/circular-shift conditions. The development subjects were the same subjects used for legacy checkpoint selection, and the legacy shared tokens are bidirectional full-window tokens. The legacy exports omit event timestamps, so their deranged null verifies nonidentity but cannot establish that two 20 s source windows do not overlap; no legacy-null nonoverlap claim is made. Therefore these results are offline delayed associations, not causal or future physiology prediction.

Best matched subject-equal mean delta-R² after averaging the three seed rows at each fixed lag:

| Task | Representation | Best fixed lag | Mean delta-R² |
|---|---|---:|---:|
| Mental arithmetic | EEG shared → fNIRS shared | 0 s | −0.0122 |
| Motor imagery | EEG shared → fNIRS shared | −2 s | −0.0075 |
| N-back | EEG shared → fNIRS shared | +4 s | −0.0010 |
| Word generation | EEG shared → fNIRS shared | +6 s | −0.0064 |
| Mental arithmetic | EEG native → fNIRS native | −2 s | −0.0068 |
| Motor imagery | EEG native → fNIRS native | −4 s | −0.0113 |
| N-back | EEG native → fNIRS native | +8 s | +0.0265 |
| Word generation | EEG native → fNIRS native | −2 s | −0.0239 |

The legacy **shared-latent** probe yielded no positive task-level best mean delta-R². The positive N-back native-feature value was small, post-selection, selected from multiple task/lag views, positive in three of five subjects per seed, and only +0.0083 above the deranged mean at +8 s. It is an exploratory implementation signal, not confirmatory evidence. It supports retaining native targets and fixed lag-bank evaluation, but does not rehabilitate the legacy shared-token claim.

Primary artifact: `experiments/runs/physiology_semantic_tokenizer/existing_lagged_predictability/20260820_full_v3`. The provenance-complete no-checkpoint synthetic governance smoke is `experiments/runs/physiology_semantic_tokenizer/existing_lagged_predictability/20260820_synthetic_smoke_v4`.

## Measured smoke runs

Preparation was rerun with truthful governance fields: protected metadata can be indexed by the unified loader, but measured protected access and protected sample `__getitem__` calls are both zero.

B0, M1, and N1 completed for both first-round tasks with one seed and the registered smoke caps. These runs exercise optimization, K-means, VQ annealing, checkpoint/export serialization, classifier ablations, and measured masks; they are not endpoint estimates. Each development smoke contains one subject and four samples.

Fit-parameter codebook health after the two-step VQ smoke:

| Task/variant | EEG active / 16 | EEG perplexity | fNIRS active / 16 | fNIRS perplexity | Positive pairing |
|---|---:|---:|---:|---:|---|
| Motor imagery M1 | 16 | 13.99 | 16 | 14.95 | matched |
| Motor imagery N1 | 16 | 13.04 | 16 | 14.86 | deranged |
| Word generation M1 | 16 | 14.16 | 16 | 14.10 | matched |
| Word generation N1 | 16 | 13.86 | 16 | 14.50 | deranged |

Every M1/N1 smoke initialized each codebook from all 80 admitted fit-parameter token latents, reached quantization strength 1.0 and posterior temperature 0.1, retained pre-VQ/posterior/hard-ID/expected-embedding exports, and completed frozen-head fitting. No codebook collapse occurred on the fit smoke. This is a connectivity/health gate only.

The coupling analyses produced matched and registered-deranged soft co-occurrence tensors, residual conditional log-lift maps, fit-only display orders, q0/q1 subject rows, paired subject-level M1-minus-N1 proper-score rows at every fixed lag, and PNG/PDF QC figures with one shared symmetric scale across M1/N1 within each task. All 48 categorical q0/q1 fits converged; the observed iteration maxima were 3884 for q0 and 4140 for q1 under the registered 5000-iteration ceiling. A fixed 5% train-only label-smoothing mixture prevents absent smoke-training classes from producing an infinite categorical intercept; held-out scoring uses the original unsmoothed soft posterior targets. The one-subject bootstrap intervals are intentionally degenerate and have no inferential meaning. The 1728×1248 PNGs and paired PDFs are provisional engineering QC displays with machine-readable NPZ/CSV sources and local descriptions, not journal-ready or accessibility-certified figures.

Primary smoke artifacts:

- preparation: `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq/20260820_lc_spvq_preparation_smoke_v9`
- motor-imagery B0/M1/N1 suite: `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq/20260820_b0_m1_n1_motor_imagery_smoke_v8`
- word-generation B0/M1/N1 suite: `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq/20260820_b0_m1_n1_word_generation_smoke_v8`
- motor-imagery coupling QC: `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq_analysis/20260820_motor_imagery_smoke_v12`
- word-generation coupling QC: `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq_analysis/20260820_word_generation_smoke_v11`

Artifact hashes recorded in each manifest were independently recomputed and matched.

## Gates still closed

The following are deliberately not claimed or launched:

- no multi-seed/full-epoch B0/M1/N1 scientific comparison;
- no fit-selection choice between lag-loss weights 0.1 and 0.5;
- no K32 capacity ablation;
- no primary-vs-N1 paired subject bootstrap with meaningful subject count;
- no freeze-and-apply result for mental arithmetic or n-back under the new model;
- no protected-cohort evaluation;
- no causal/future physiology claim;
- no reinterpretation of the legacy 2/16 result.

Before a scientific full run, the full-mode selection loop must choose the lag-loss weight and checkpoints using fit-selection subjects only, then freeze all choices before development application. Protected evaluation requires a later explicit human authorization and an independently frozen protocol.

## Evidence provenance and accountability

The machine-readable local evidence ledger is `docs/analysis/20260820_LC_SPVQ_VALIDATION_EVIDENCE.json`. It binds the test/compile command outcomes and the seven final run-manifest hashes used by this record. Artifact contents were checked locally against their manifest inventories. Human scientific verification is still marked pending; this engineering record is not submission-ready, and accountable human investigators retain all interpretation and release decisions.

## Reproducible commands

```bash
# Contract/synthetic tests
.venv/bin/python -m pytest -q \
  tests/test_lag_conditioned_dataset.py \
  tests/test_lag_conditioned_native_features.py \
  tests/test_lag_conditioned_baseline.py \
  tests/test_lag_conditioned_shared_private_vq.py \
  tests/test_lag_conditioned_losses.py \
  tests/test_lag_conditioned_downstream_metrics.py \
  tests/test_lag_conditioned_spvq_runner.py \
  tests/test_lagged_token_coupling.py \
  tests/test_probe_existing_lagged_predictability.py \
  tests/test_analyze_lag_conditioned_spvq.py

# Legacy existing-export probe (post-selection development only)
.venv/bin/python experiments/probe_existing_lagged_predictability.py \
  --output-dir experiments/runs/physiology_semantic_tokenizer/existing_lagged_predictability/20260820_full_v3

# One-seed measured smoke for both first-round tasks and all variants
.venv/bin/python experiments/run_lag_conditioned_spvq.py \
  --smoke --tasks motor_imagery word_generation --variants B0 M1 N1 \
  --output-dir <new-output-directory>

# Coupling/proper-score analysis for one task
.venv/bin/python experiments/analyze_lag_conditioned_spvq.py \
  --m1-export <M1-token_exports.npz> \
  --n1-export <N1-token_exports.npz> \
  --output-dir <new-analysis-directory> \
  --bootstrap-iterations 100
```
