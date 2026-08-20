# LC-SPVQ architecture optimization results

**Date:** 2026-08-20

**Status:** completed single-seed exploratory optimization; not a comprehensive scientific validation

**Analysis boundary:** fit-selection-only candidate choice followed by frozen, descriptive development application

**Protected cohorts:** closed; measured protected access count = 0

**Human scientific verification:** pending

## Executive result

The registered selector chose `lag05_h23`, which changes the lag-loss training weight from 0.1 to 0.5 while retaining the reference history. Its task-mean fit-selection coupling-plus-private subject-equal macro-F1 was 0.674903, versus 0.622284 for `reference_h23_lag01` (difference +0.052618). This exceeded the prospectively fixed 0.01 threshold. The gain was task-asymmetric: +0.105237 for motor imagery and exactly 0.000000 at the reported precision for word generation. [E2, E3]

This was **not an architecture-history improvement**. The best history candidate, `h13_lag01`, improved the task mean by only +0.000572 over the common reference and was 0.008519 below the long-duration control. The exact-reference long-duration control improved the task mean by +0.009091, below the registered 0.01 threshold and with opposite task-level directions. [E2, E3]

The post-selection development application did **not** corroborate transferable improvement. `lag05_h23` was lower than the reference on both tasks: −0.023609 for motor imagery and −0.015324 for word generation; the two-task mean difference was −0.019467. Its development cross-entropy was also higher on both tasks. Development subjects 19–23 were already reused for method development, so these values are descriptive and cannot revise the frozen selection or provide independent confirmation. [E4]

Accordingly, the narrow answer is:

- **Fit-selection objective optimization improved the registered task-head QC score.**
- **No tested history architecture met the improvement rule.**
- **The apparent selection gain did not transfer to reused development subjects.**
- **There is no basis here for a robust, causal, sharedness, or protected-cohort performance claim.**

## Prospective boundary and implementation record

The prior LC-SPVQ architecture implementation was committed as `7099eb918f59f7c44ae92ea4fe733dcb31f60e74`. The optimization protocol, configuration, runner, and focused tests were committed prospectively as `39b75b5` before measured execution. [E7]

The first measured attempt stopped during the first candidate's task-head evaluation because the optimization runner requested a non-existent `combined_logits` export key instead of the reviewed runtime's `coupling_plus_private_logits` key. No global decision or development registry was created in that failed staging directory. The correction and regression test were committed as `ce2b7c7394c4030bafd421a5ce72f589f2fa0282`; the successful run began from that clean commit. The failed staging sample registry is identical to the successful deterministic registry by file SHA-256. [E8]

The successful run used:

- one training seed, `20260820`;
- deterministic sample-registry seed `20260824` and donor seed `20260825`;
- `cuda:1`, FP32, strict deterministic algorithms, and no AMP;
- five candidates over motor imagery and word generation;
- standard schedules of 64 pretrain, 32 VQ, and 64 frozen-head optimizer steps;
- an exact-reference long-duration control of 128, 64, and 128 steps;
- batch size 16 and evaluation every eight optimizer steps;
- candidate-independent representation selection loss: `0.5*EEG native MSE + 0.5*fNIRS native MSE + 0.5*lag matching loss`;
- global primary metric: task-mean coupling-plus-private subject-equal macro-F1. [E1, E7]

## Samples and access governance

The successful run materialized 736 unique nonprotected task-partition samples, compared with 32 in the preceding two-task smoke record:

| Partition | Motor imagery | Word generation | Total |
|---|---:|---:|---:|
| Fit parameter, subjects 01–15 | 240 | 240 | 480 |
| Fit selection, subjects 16–18 | 48 | 48 | 96 |
| Frozen development apply, subjects 19–23 | 80 | 80 | 160 |
| **Total** | **368** | **368** | **736** |

Every subject-condition cell contributed exactly eight samples. Motor-imagery sampling spread those eight samples across three records as 3/3/2; word generation used eight event-time-spread samples from its single record. Development was materialized only after the persisted global decision and its digest existed. The protected Single-Trial subjects 24–29 and simultaneous VP024–VP026 remained closed; the run records zero protected measured accesses. [E5, E6]

## Candidate comparison on fit selection

Higher primary and coupling-only macro-F1 are better; lower combined cross-entropy is better. The primary metric is task-head QC only and is not evidence that the coupling/sharedness construct is valid. [E2]

| Rank | Candidate | Class | Task-mean primary | Task-min primary | Mean coupling-only F1 | Mean CE | Δ primary vs reference |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `lag05_h23` | objective weight | 0.674903 | 0.622533 | 0.333333 | 0.662929 | +0.052618 |
| 2 | `reference_h23_lag01_long` | duration control | 0.631375 | 0.598387 | 0.394343 | 0.675551 | +0.009091 |
| 3 | `h13_lag01` | history architecture | 0.622856 | 0.622035 | 0.354267 | 0.675532 | +0.000572 |
| 4 | `reference_h23_lag01` | common reference | 0.622284 | 0.622035 | 0.425827 | 0.665596 | 0.000000 |
| 5 | `h12_lag01` | history architecture | 0.612072 | 0.602109 | 0.333333 | 0.675600 | −0.010212 |

All candidates were rankable. Within each task they had equal trainable parameter counts: 5,580,199 for motor imagery and 5,272,199 for word generation. All ten candidate-task cells retained all 16 EEG and all 16 fNIRS fit-parameter codes. [E2, E3]

`lag05_h23` improved the combined primary score but reduced task-mean coupling-only macro-F1 by 0.092494 relative to the reference (0.333333 versus 0.425827). Therefore the registered result is an **objective-weight/task-head QC improvement**, not stronger evidence for the lagged coupling endpoint. [E2]

## Optimization curves

The run preserved 254 figure-source rows: 144 representation evaluations, 96 frozen-head evaluations, 10 task-level selection points, and four frozen development applications. Standard candidates contribute eight pretrain, four VQ, and eight head evaluations per task; the long control contributes 16, eight, and 16. Pretrain and VQ representation stages are deliberately drawn as disconnected segments on a cumulative optimizer-step axis. No smoothing, uncertainty interval, or multi-seed aggregation was applied. [E5]

The representation curves generally decreased within the standard budgets. For example, the motor-imagery reference moved from 2.840056 to 2.439976 during pretraining and reached a best VQ selection loss of 2.352779; the word-generation reference moved from 2.221572 to 1.932095 and reached a best VQ value of 1.874586. The higher lag-loss-weight candidate had higher final fixed representation loss on both tasks despite its stronger combined head score. [E3, E5]

The head curves were not monotone and expose selection instability rather than hiding it. The motor-imagery `lag05_h23` curve reached 0.727272 before ending at 0.704365. The word-generation long control reached its best value, 0.664364, at its first head evaluation and ended at 0.538873. All stages nevertheless ran to the prospectively fixed optimizer budgets; best checkpoints, rather than final steps, were used. [E3, E5]

Curve deliverables:

- `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq_optimization/selection_development_curves.png`
- `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq_optimization/selection_development_curves.pdf`
- `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq_optimization/curve_figure_source_data.csv`
- `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq_optimization/curve_figure_alt_text.txt`

The figure uses redundant candidate color/line/marker encoding and task width/opacity encoding. It is explicitly provisional general QC, with no confidence intervals, journal certification, or accessibility certification. Manual inspection found no clipping or false connection across the stage gaps. However, the panels do not label those stages directly, the legends obscure substantial portions of Panels A and B, and equal-valued points overlap in Panel C. Panel C shows selected-checkpoint values rather than necessarily final head endpoints, while its generic axis compares fit-selection and descriptive development values; exact interpretation therefore requires the source table and this report. The four development source rows also leave curve-kind, stage, and step blank. The PNG is 2520×827 opaque RGBA at approximately 180 DPI. The PDF is an untagged, one-page vector export with an embedded subsetted Type-3 DejaVu Sans font. These files are retained only as internal provisional QC. [E5]

## Frozen development application

Only the common reference and globally selected candidate were applied, once per task, after selection. Four permit-registered applications were recorded, with no unselected candidate export. Every application had identical model-state digests before and after evaluation. [E4, E6]

| Task | Candidate | Selection primary | Development primary | Development CE | Selected − reference on development |
|---|---|---:|---:|---:|---:|
| Motor imagery | `reference_h23_lag01` | 0.622035 | 0.520938 | 0.694462 | — |
| Motor imagery | `lag05_h23` | 0.727272 | 0.497330 | 0.700171 | −0.023609 |
| Word generation | `reference_h23_lag01` | 0.622533 | 0.558969 | 0.679296 | — |
| Word generation | `lag05_h23` | 0.622533 | 0.543645 | 0.688764 | −0.015324 |

The reference's two-task development mean was 0.539954; the selected candidate's was 0.520487. This descriptive reversal is consistent with candidate-selection optimism, task heterogeneity, or sampling/training variability, but a single seed and reused development subjects cannot distinguish those explanations. [E4]

## Artifact and software checks

- The successful manifest inventories 77 artifacts and eight input sources; all recorded file sizes and SHA-256 values were recomputed successfully. [E1]
- Ten candidate-task cells produced exactly 30 stage checkpoints. Every checkpoint matched its task, candidate, seed, variant, stage, VQ surface, nonprotected status, and no-development status, and retained optimizer and RNG state. [E1, E3]
- Four frozen development state-digest pairs were unchanged. [E4]
- All 254 present numeric figure-source values were finite. [E5]
- Focused optimization tests passed: 18 tests after the runtime-key regression was added. [E8]
- The final repository suite excluding two known historical R1P prevalidation-seal modules passed 535 tests with 11 deselected. The full suite retained the same 10 unrelated, fail-closed stale-seal failures; historical seals were not rewritten. [E8]
- Independent governance audits found no remaining execution blocker before the successful run. A separate post-run artifact audit is recorded in the evidence ledger. [E8]

## Interpretation and decision

Under the frozen selector, `lag05_h23` remains the **registered candidate for a future, separately frozen nonprotected evaluation**, because development was not allowed to alter selection. It should not replace the reference as a validated model on the basis of this run: its gain was concentrated in motor-imagery fit selection, its coupling-only QC was lower, and both descriptive development tasks favored the reference.

The architecture-history optimization question is answered negatively for this candidate set: neither shorter-history candidate met the registered improvement rule, and the best history candidate did not beat both reference controls. Longer optimization was task-dependent and did not reach the 0.01 global threshold.

Any future claim of transferable performance would require a newly frozen design and genuinely independent evaluation data; if protected cohorts were involved, explicit authorization would be an additional requirement, not a substitute for independence. This run does not authorize protected access, multi-seed inference, causal interpretation of positive lags, or reinterpretation of the immutable legacy continuous result (only 2/16 simultaneous lower-bound criteria exceeded zero).

## Limitations

1. The declared run scope used one seed; there are no seed-variance estimates or confidence intervals.
2. Candidate selection repeatedly used subjects 16–18, making selection scores optimistic for the chosen candidate.
3. Subjects 19–23 were reused post-selection development subjects, not a new independent holdout.
4. Task-head macro-F1 and cross-entropy are QC endpoints, not q0/q1 coupling/sharedness evidence.
5. The tested set changes only history length, lag-loss weight, or training duration; it does not exhaust architecture space.
6. Positive offline lags describe associative predictability, not causality or proof of future physiology.
7. Independent EEG and fNIRS code IDs remain modality-specific and must not be aligned cellwise.
8. The provisional figure has a dense legend and is not certified for a journal or for accessibility.
9. Accountable human investigators have not yet verified or approved external scientific use.

## Reproduction

```bash
# Successful run (requires a clean worktree and an absent registered output target)
.venv/bin/python experiments/optimize_lag_conditioned_spvq_architecture.py --device cuda:1

# Focused optimization tests
.venv/bin/python -m pytest -q \
  tests/test_optimize_lag_conditioned_spvq_architecture.py

# Repository regression excluding the two known stale historical R1P seal modules
.venv/bin/python -m pytest -q \
  --ignore=tests/test_build_r1p_train_only_perturbation_bundles.py \
  --ignore=tests/test_qualify_r1p_population_frozen_teacher.py
```

## Evidence map

| ID | Source | SHA-256 / locator |
|---|---|---|
| E1 | Successful run manifest | `experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq_optimization/manifest.json`; `c4abb9d81dd559103b0e6bee40ca7138d86db46326b08c2fa1184586b6694ade` |
| E2 | Frozen global selection | `global_candidate_selection.json`; `dd7f6950532dff420803b7cf04427f9bcfca45c2806d17aa88ff1fb5607e5593` |
| E3 | Candidate results/summary | `candidate_summary.json`; `f776fd6edaff7bf778ba7277e472da13f1297a81be08180e19ee3a5434d388ee` |
| E4 | Frozen development comparison | `development_comparison.json`; `7a9919e9916b3388631bb33594f82c711d93861832986fbfb93d7e36cfc96b43` |
| E5 | Curve source and figure manifest | `curve_figure_source_data.csv`; `f0fe02216dd874f296594231588cfca06d0aff685ee11561f1d7ef99da8ed015`; `curve_figure_manifest.json`; `56179dd00d6d84a4bc10b1a7a6244bf234a40938549bcd8632556cb3291fcfa6` |
| E6 | Access audit and registries | `sample_access_audit.json`; `aa40bb639e83287f9db0fae09fcb3c0d2f41418311e925ded2297ea303d0bca6`; registry hashes are bound in E1 |
| E7 | Frozen protocol/configuration | protocol `8f1dff3510936ff136ac5df2dc1e0ebd625554b4b065df980a5844fbdf6fc7a8`; configuration `d6d65fb9e6ad35065582e50c0e30668e765a9dc64e4d72ccca0def388822fab9` |
| E8 | Git/test/audit record | commits `7099eb9`, `39b75b5`, `ce2b7c7`; detailed checks are recorded in `20260820_LC_SPVQ_ARCHITECTURE_OPTIMIZATION_EVIDENCE.json` |

Accountable human investigators must open the bound sources, verify every scientific proposition, and approve any external use. This local report is not submission-ready.
