# EFRM resource-bounded dual-protocol final results

**Protocol:** `efrm_resource_bounded_dual_protocol_v1`

**Method:** `efrm_sync_200_10_variable_channel_v1`

**Completed:** 2026-07-27

**Status:** complete; 70/70 public jobs and 70/70 protected evaluations

## Result authority

The authoritative generated summary is
`runs/formal/efrm_resource_bounded_dual_protocol_v1/aggregate/summary.json`.
The generated artifacts remain below the ignored `runs/` tree; this document
is the version-controlled result record.

- Cohort manifest SHA-256:
  `44a45d2e68b458a4999c21ab66662f9060c9bd1cd3256bb80f954d8b32b72f7e`
- Source checkpoint SHA-256:
  `70367851bebed1a28a8406e3d4e1d87e603486f93ce460b212f98964e05fe8fa`
- Job matrix SHA-256:
  `bcc58418a58e0b92b94d2bc311b2b5b09889d4a6554d42371db335267e576854`
- Metric registry SHA-256:
  `5b467ed0f26fa6c0bdc2d6fb72945673a05dc59e88ecc39829e21230af88c9f1`
- Aggregate summary SHA-256:
  `da6daba692e502883481f273ae42c7af75460f8669b4a0b1bb212dfe8a8f27a9`
- Frozen implementation audit: every implementation hash in the job matrix
  still matches the source file used to generate the protected results.

The source/target cohort had zero subject overlap. The single source-only
checkpoint was selected without target access. Target folds were materialized
only after that checkpoint was frozen. All five protected test folds partition
each eligible target task exactly once.

## Source-only pretraining

The formal run used seed 42, exact two-pass gradient caching, chunk size 8,
and the frozen synchronized-pair architecture and objectives.

- Source train: 6,189 windows.
- Source validation: 1,929 windows.
- Training completed all 100 epochs.
- Best total source-validation loss: `3.943938` at epoch 85.
- Best source-validation CLIP loss: `3.393163` at epoch 11.
- Peak reserved CUDA memory: approximately 15.88 GiB.
- Final one-batch alignment evidence reported EEG→fNIRS top-1 `0.0625` and
  fNIRS→EEG top-1 `0.09375` for 32 pairs. This is descriptive evidence from
  one validation batch, not a physiological-coupling or causal result.

## Primary five-fold results

Values are fold mean ± sample SD across five target outer folds (`ddof=1`).
Intervals are the frozen two-sided fold-level 95% t intervals with four
degrees of freedom. Classification uses macro-F1; REFED uses native-coordinate
CCC.

| Task | Strict cross-subject | Sample-random | Random − strict |
| --- | ---: | ---: | ---: |
| Motor imagery | 0.4707 ± 0.0229 [0.4422, 0.4992] | 0.4756 ± 0.0142 [0.4580, 0.4932] | +0.0049 |
| Mental arithmetic | 0.5521 ± 0.0438 [0.4976, 0.6065] | 0.5987 ± 0.0498 [0.5369, 0.6606] | +0.0467 |
| Word generation | 0.5338 ± 0.0469 [0.4756, 0.5920] | 0.5601 ± 0.0259 [0.5279, 0.5923] | +0.0263 |
| N-back | 0.3289 ± 0.0503 [0.2664, 0.3914] | 0.3511 ± 0.0289 [0.3153, 0.3870] | +0.0222 |
| DSR | 0.4977 ± 0.0277 [0.4633, 0.5322] | 0.4627 ± 0.0428 [0.4095, 0.5158] | −0.0351 |
| Visual motivation | 0.2098 ± 0.0257 [0.1779, 0.2416] | 0.2783 ± 0.0040 [0.2733, 0.2832] | +0.0685 |
| REFED regression | 0.04785 ± 0.00636 [0.03995, 0.05574] | 0.06183 ± 0.00952 [0.05001, 0.07366] | +0.01399 |

Source-aligned companion endpoints:

| Task | Strict cross-subject | Sample-random |
| --- | ---: | ---: |
| Motor imagery Accuracy | 0.4867 ± 0.0271 | 0.4772 ± 0.0156 |
| Mental arithmetic Accuracy | 0.5608 ± 0.0423 | 0.6009 ± 0.0509 |
| Word generation Accuracy | 0.5422 ± 0.0469 | 0.5608 ± 0.0261 |
| N-back Accuracy | 0.3463 ± 0.0380 | 0.3594 ± 0.0392 |
| DSR Accuracy | 0.6962 ± 0.0131 | 0.7017 ± 0.0047 |
| Visual motivation Accuracy | 0.3406 ± 0.0249 | 0.3890 ± 0.0100 |
| REFED native RMSE | 57.4292 ± 4.3020 | 57.1289 ± 1.5891 |

## REFED coordinate-level descriptive supplement

The frozen fold metrics contain R², Pearson, and Spearman values under
`native_coordinates` for valence and arousal. The v1 aggregator did not lift
these nested values into the top-level summary. The following means and sample
SDs were computed directly from those five immutable fold metrics after the
primary aggregation. They are descriptive supplements and do not alter the
frozen v1 aggregate.

| Protocol | Coordinate | CCC | R² | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| Strict | Valence | 0.0724 ± 0.0148 | 0.0002 ± 0.0450 | 0.1617 ± 0.0300 | 0.1607 ± 0.0372 |
| Strict | Arousal | 0.0233 ± 0.0090 | −0.0353 ± 0.0255 | 0.0666 ± 0.0376 | 0.0537 ± 0.0350 |
| Sample-random | Valence | 0.0812 ± 0.0129 | 0.0132 ± 0.0213 | 0.1646 ± 0.0286 | 0.1527 ± 0.0345 |
| Sample-random | Arousal | 0.0425 ± 0.0065 | 0.0027 ± 0.0113 | 0.1164 ± 0.0199 | 0.1060 ± 0.0182 |

## Interpretation against expectations

The result is **partially consistent** with the frozen protocol's expectation
that direct sample-random transfer is optimistic:

1. Sample-random is higher on the primary endpoint for six of seven tasks.
   The largest absolute classification sensitivity is visual motivation
   (`+0.0685` macro-F1, about `+32.7%` relative to strict). Its fold-level
   intervals do not overlap. This is the clearest evidence that participant or
   acquisition context crossing the split improves the diagnostic estimate.
2. Mental arithmetic, word generation, and N-back move in the expected
   direction, but their fold intervals overlap substantially. These are modest
   protocol-sensitivity signals, not evidence of a statistically established
   transfer gap.
3. REFED CCC rises by `0.01399` (`+29.2%` relative), but this large relative
   percentage is caused by a very small strict denominator. Native RMSE changes
   by only `−0.30`, and coordinate-level R² remains approximately zero. The
   model therefore captures at most weak affective variation.
4. DSR is the exception: sample-random Accuracy is slightly higher, while
   macro-F1, balanced Accuracy, and Kappa are lower. Strict versus random
   balanced Accuracy is `0.5307` versus `0.5192`, and Kappa is `0.0777` versus
   `0.0498`. With the strong Go/No-go imbalance, Accuracy is dominated by the
   majority class; macro-F1 correctly exposes unstable minority-class
   performance. DSR does not support a blanket “non-cross is always better”
   claim.

Absolute performance is weaker than a strong-transfer expectation:

- Motor imagery is at or below balanced binary chance, with negative mean
  Kappa in both protocols.
- N-back is close to three-class chance and has near-zero Kappa.
- Strict visual balanced Accuracy is `0.2525`, essentially four-class chance;
  sample-random improves it to only `0.2916`.
- Mental arithmetic and word generation retain modest above-chance signal.
- REFED CCC is positive but small, with approximately zero R².

The correct scientific conclusion is therefore not that EFRM provides a
strong cross-subject representation. The experiment shows a weak,
task-dependent frozen-linear-probe representation and a measurable split
sensitivity concentrated in visual motivation. It also confirms why the
sample-random result must not be described as new-subject generalization.

## Claim boundary

These numbers estimate transfer from one fixed source-only EFRM checkpoint to
a disjoint target cohort. They do not estimate pretraining-seed uncertainty,
do not reproduce the paper's 1,247.5-hour pretraining result, and do not
validate physiological EEG-fNIRS alignment. They must not be directly ranked
against the existing full-dataset STA-Net five-fold aggregate. A direct method
comparison still requires STA-Net on the exact target cohort and fold
manifests.
