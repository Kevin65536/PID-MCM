# LC-SPVQ single-seed architecture-optimization protocol

**Date:** 2026-08-20
**Status:** frozen before measured optimization execution
**Analysis class:** exploratory architecture optimization and implementation QC
**Scientific endpoint:** no

## 1. Scope and immutable context

This protocol tests whether a small, controlled set of LC-SPVQ architecture and objective changes improves fit-selection performance relative to the committed first-round reference architecture. It is not a full, multi-seed scientific comparison. The starting architecture implementation is commit `7099eb918f59f7c44ae92ea4fe733dcb31f60e74`.

The earlier continuous shared/private result remains an immutable negative result (`2/16`). This optimization must not reinterpret it. Protected cohorts remain closed. Subjects 19--23 / VP019--VP023 are reused post-selection development data, not a fresh held-out cohort.

The executable contract is `experiments/configs/physiology_semantic_tokenizer/lag_conditioned_spvq_architecture_optimization.yaml`, SHA-256 `d6d65fb9e6ad35065582e50c0e30668e765a9dc64e4d72ccca0def388822fab9`. It binds the committed base LC-SPVQ YAML by SHA-256.

## 2. Data use and deferred development access

The two first-round tasks are optimized jointly:

- motor imagery, single-trial dataset;
- word generation, simultaneous dataset.

Each task uses the canonical nonprotected subject roles and eight deterministic samples per subject and class. Within every subject-condition group, selection cycles across available record IDs and spreads picks across event time within each record rather than taking the first eight rows:

| Role | Subjects per dataset | Classes | Samples per subject/class | Samples per task |
|---|---:|---:|---:|---:|
| fit parameter | 15 | 2 | 8 | 240 |
| fit selection | 3 | 2 | 8 | 48 |
| development apply | 5 | 2 | 8 | 80 |

For comparison, the prior measured smoke used 8/4/4 samples per task for fit/selection/development. The optimization therefore increases fit support by 30 times, selection support by 12 times, and development support by 20 times.

Input and native-target standardizers are fitted only on the 240 fit-parameter samples. The sample registry is generated once with seed `20260824`; donor permutations are generated once with seed `20260825`. Their identities and hashes must be identical across candidates. Candidate training and checkpoint selection may access only fit-parameter and fit-selection partitions. Development arrays must not be materialized until one immutable global selection decision exists. Phase-specific measured/protected access counters must prove this boundary. The selected candidate and the preregistered reference are then frozen and each distinct model is applied once; if the selected candidate is the reference, the application is deduplicated. Unselected candidates are never evaluated on development.

## 3. Fixed seed and candidate chain

Only seed `20260820` is used. There is no multi-seed repetition.

The candidate chain changes one registered factor at each step:

| Candidate | Role | EEG history (current + past) | fNIRS history (current + past) | Lag-loss weight | Step multiplier |
|---|---|---:|---:|---:|---:|
| `reference_h23_lag01` | common-budget reference | 1 + 1 | 1 + 2 | 0.1 | 1 |
| `lag05_h23` | objective-weight test | 1 + 1 | 1 + 2 | 0.5 | 1 |
| `h13_lag01` | EEG current-only history test | 1 + 0 | 1 + 2 | 0.1 | 1 |
| `h12_lag01` | shorter history in both modalities | 1 + 0 | 1 + 1 | 0.1 | 1 |
| `reference_h23_lag01_long` | exact longer-training control | 1 + 1 | 1 + 2 | 0.1 | 2 |

All other contracts remain fixed across candidates: matched M1 training, encoder depth 2, four heads, feed-forward width 256, D64 shared state, 64/32 private dimensions, dropout 0.1, full-window private encoders, independent EEG/fNIRS K16 EMA codebooks, fit-only K-means initialization, rank-8 coupling, the `0/2/4/6/8/10 s` lag bank, and registered hard-negative masks. Every standard candidate has identical parameter count and optimizer budget.

The ordinary contrasts are exact and path-specific: lag weight `0.1→0.5` at the reference histories, EEG history `(current+1)→current-only` with fNIRS fixed, and then fNIRS history `(current+2)→(current+1)` after the EEG shortening. The long control changes only optimizer budget. A long-control win is an optimization-duration result, not an architecture result. A history candidate can support an exploratory architecture-improvement label only if it exceeds both common-budget and long references under the frozen selection rule. No numerical comparison to the earlier short smoke is used as optimization evidence.

## 4. Optimization schedule and checkpoint selection

Each standard task/candidate run has at most 64 continuous-pretraining, 32 VQ-annealing, and 64 frozen-representation task-head optimizer steps. The exact long reference doubles these maxima to 128/64/128. Batch size is 16. Fit-selection evaluation occurs every eight optimizer steps. Early stopping requires at least 32/16/32 steps for the three stages and four consecutive non-improving selection evaluations; minimum improvement is `1e-4` for representation loss and `1e-6` for head selection.

AdamW uses the base `3e-4`/head `1e-3` learning rates, betas `(0.9, 0.98)`, weight decay `0.01`, no warmup, no scheduler, deterministic FP32, and no AMP. Quantization strength changes linearly from 0 to 1 and posterior temperature from 1.0 to 0.1 over the first 32 VQ steps for every candidate. The long control holds that final VQ surface during its additional 32 steps rather than annealing more slowly. Shared/private representations and both codebooks are fully frozen during head training.

Continuous and VQ checkpoints minimize the same candidate-independent fit-selection criterion without using task labels:

`L_rep = 0.5 L_EEG-native + 0.5 L_fNIRS-native + 0.5 L_lag`.

All terms are masked means in the train-standardized coordinates already defined by the base protocol. Raw/private reconstruction and commitment losses are excluded from checkpoint selection. VQ selection evaluations use quantization strength 1.0 and posterior temperature 0.1 regardless of the current training anneal state. Candidate-specific weighted training losses remain in the curve record but are not compared as the checkpoint metric.

The task-head checkpoint and global architecture comparison deliberately use fit-selection task labels: the head maximizes subject-equal macro-F1 for the coupling-plus-private output, with lower fit-selection combined cross-entropy as a tie-breaker. Every selected checkpoint is restored before the next stage or final export.

## 5. Global architecture selection

One architecture is selected globally across both tasks. The primary score is the unweighted task mean of fit-selection coupling-plus-private subject-equal macro-F1. Tie-breakers, in order, are:

1. the minimum task-specific value of the same metric;
2. task-mean fit-selection coupling-only subject-equal macro-F1;
3. lower task-mean fit-selection combined cross-entropy;
4. fewer actual optimizer steps;
5. lower trainable parameter count;
6. the registered candidate order.

Comparisons use a numeric tie tolerance of `1e-8`. A candidate is rankable only if all required metrics are finite and both tasks have complete registered support; the run fails closed if none is valid. A descriptive optimization gain is called present only when the numerical winner exceeds `reference_h23_lag01` by at least `0.01`. If it does not, the common-budget reference remains the recommendation, while the numerical best may still receive the preregistered descriptive frozen-development transfer check. A long-control win is labeled duration/optimization improvement. A history architecture can be labeled improved only if it beats both reference arms. Development deltas are not a second selection criterion.

This selector is explicitly a **task-head performance QC** endpoint. Because the combined head includes the private branch and rank-dependent coupling head, a macro-F1 gain is not evidence for the q0/q1 coupling endpoint or for stronger cross-modal sharedness. This run makes no B0, N1, or q0/q1 proper-score comparison.

## 6. Required records and displays

The run must retain:

- per-step train losses and gradient norms;
- every fit-selection loss and macro-F1 evaluation;
- selected-checkpoint markers and stopping reasons;
- candidate checkpoints and fit-selection metrics;
- sample counts, identities, per-record/event-time coverage, sample/donor registry hashes, measured-access counters, and protected-access counters;
- global candidate ranking and the exact deterministic tie-break tuple;
- frozen reference/selected development metrics, arrays, application counts, and pre/post model-state digests;
- SHA-256 input and artifact inventories.

The primary figure contains selection representation-loss curves, head selection macro-F1 curves, and reference/selected selection-development comparisons. It has no confidence intervals because there is one seed. Line style and markers duplicate color encoding. PNG/PDF figures are provisional general QC outputs; no journal or accessibility certification is claimed. The source-data CSV and an explicit text description are authoritative companions.

## 7. Interpretation limits

This medium-budget run can answer whether the registered candidate chain improves task-head QC under this larger single-seed development workflow. It cannot establish population-level reliability, seed stability, coupling/sharedness improvement, causal prediction, protected-cohort generalization, or a scientific endpoint. Reusing the same three fit-selection subjects for representation checkpointing, head checkpointing, and architecture ranking makes the reported selection score optimistic. All curves/checkpoints are retained so this reuse remains visible. Positive development transfer remains post-selection and descriptive. Negative or inconsistent results must be retained.

The standard LC-SPVQ full-training entrypoint remains fail-closed. Opening protected evaluation requires a separate frozen protocol and explicit human authorization.
