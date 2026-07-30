# Comparison experiment status

_Snapshot: 2026-07-30; live EFRM state must be read from its status file_

## Current decision surface

| Method / track | State | Evidence and next action |
| --- | --- | --- |
| STA-Net strict five-fold | **Complete** | 70/70 fold trainings completed under the frozen convergence rule. Retain the aggregate and current formal checkpoints. |
| EFRM LODO v2 Stage A | **Running** | 3/4 target-excluded selection jobs completed at status update `2026-07-30T00:33:59Z`; Visual selection is running. |
| EFRM LODO v2 Stage B | **Next** | 0/4 full non-target refits completed. Starts only after all Stage-A selections freeze their selected epochs. |
| EFRM v2 downstream matrix | **Queued** | After Stage B: feature cache, strict `7 tasks × 5 folds × 3 seeds` linear probes, gates, one-time protected evaluation, aggregate, then separately labeled sample-random results. |
| EFRM v2 full fine-tuning | **Conditional** | Resource-contingent secondary track; cannot replace the frozen linear-probe matrix. |
| UMAP protocol / formal rerun | **Protocol next; run conditional** | Freeze a new cross-subject multi-seed protocol before authorizing data access. Historical best test accuracy `55.83%`, macro-F1 `0.558` is diagnostic after repeated test viewing. |
| Cross-method final table | **Blocked on comparable formal values** | Run the metric-acceptance audit only after EFRM v2 and any admitted UMAP rerun complete. |

EFRM v2 protected data remain closed:
`protected_test_opened=false`, `target_dataset_exposure=false`. Its live source
of truth is
[`status.json`](../../comparative_methods/EFRM-PyTorch/runs/formal/efrm_lodo_full_target_fivefold_v2/status.json).

## Completed STA-Net reference

The formal aggregate is
[`paper_table.md`](../../comparative_methods/STA-Net-PyTorch/runs/fivefold/20260727_sta_net_no_artifact_mask_converged_5fold_v1/aggregate/paper_table.md).
Strict cross-subject primary endpoints are:

| Task | Endpoint | Mean ± fold SD |
| --- | --- | ---: |
| MI | macro-F1 | 56.40 ± 1.58% |
| MA | macro-F1 | 62.84 ± 4.25% |
| WG | macro-F1 | 62.11 ± 3.13% |
| n-back | macro-F1 | 37.52 ± 2.32% |
| DSR | macro-F1 | 60.69 ± 2.38% |
| Visual | macro-F1 | 25.01 ± 0.77% |
| REFED | CCC | 0.081 ± 0.048 |

These values are a completed benchmark artifact, not evidence that the method
matches its original subject-specific paper protocol.

## Historical context

EFRM resource-bounded dual-protocol v1 completed its public and protected
matrices, but used a different cohort/fold estimand. It is retained as
historical context rather than ranked directly against the v2 target. Its
strict primary endpoints were MI `0.4707`, MA `0.5521`, WG `0.5338`, n-back
`0.3289`, DSR `0.4977`, Visual `0.2098`, and REFED CCC `0.0479`.

The status labels here describe execution and admissibility. Scientific
interpretation and final-number acceptance remain separate gates.
