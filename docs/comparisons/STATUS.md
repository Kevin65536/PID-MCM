# Comparison experiment status

_Snapshot: 2026-08-14; the frozen candidate, sealed campaign status, dual-signature
unblind record, aggregate, and cell-level acceptance output are the source of truth_

## Current decision surface

| Method / track | State | Evidence and next action |
| --- | --- | --- |
| STA-Net strict five-fold | **Complete; context reference** | 70/70 fold trainings completed. The retained scores use STA-Net's method-native observation budget and remain separate from the support-matched direct surface. |
| EFRM LODO v2 Stage A | **Complete** | 4/4 target-dataset-excluded selection jobs completed and selected epochs were frozen. |
| EFRM LODO v2 Stage B | **Complete** | 4/4 full non-target refits completed on 2026-08-03; checkpoint, boundary, config, and run-manifest identities are frozen. |
| EFRM v2 downstream matrix | **Protected complete; aggregated** | 105/105 protected jobs completed. Four cells are `TABLE_READY_WITH_NOTE`; three are `REJECTED_VALUE`. |
| EFRM v2 full fine-tuning | **Conditional** | Resource-contingent secondary track; cannot replace the frozen linear-probe matrix. |
| BIOT | **Protected complete; aggregated** | 90/90 protected jobs completed. Three cells are `TABLE_READY_WITH_NOTE`, three are `REJECTED_VALUE`, and REFED is preregistered unsupported. |
| CBraMod | **Protected complete; aggregated** | 90/90 protected jobs completed. Four cells are `TABLE_READY_WITH_NOTE`, two are `REJECTED_VALUE`, and REFED is preregistered unsupported. |
| REVE | **Protected complete; aggregated** | 90/90 protected jobs completed. Three cells are `TABLE_READY_WITH_NOTE`, one is `REJECTED_VALUE`, MI/MA remain `OVERLAP_TRACK_ONLY`, and REFED is unsupported. |
| BrainFusion NVC-CSP Stacking | **Protected complete; aggregated** | 75/75 protected jobs completed. Three cells are `TABLE_READY_WITH_NOTE`, two are `REJECTED_VALUE`, and DSR/REFED are preregistered unsupported. |
| NormWear EEG-fNIRS adapted | **Protected complete; aggregated** | 90/90 protected jobs completed. Five cells are `TABLE_READY_WITH_NOTE`, one is `REJECTED_VALUE`, and REFED is preregistered unsupported. |
| UMAP | **Retired from active queue** | No formal rerun is planned; prior repeatedly viewed test results remain historical Git context only. |
| Cross-method final table | **Complete; acceptance assigned** | The single-GPU campaign completed 540/540 jobs with zero failures, was dual-signature unblinded, and produced all 42 cell terminals: 22 `TABLE_READY_WITH_NOTE`, 12 `REJECTED_VALUE`, 2 `OVERLAP_TRACK_ONLY`, and 6 `UNSUPPORTED`. |

The completed campaign opened the authorized protected evaluation exactly for
the frozen 540-job scope: its retained status reports
`protected_test_opened=true`, `completed_job_count=540`, and
`state=SEALED_COMPLETE`, with zero failed, invalid-output, missing, or technical
failure jobs. This comparison-campaign access does not alter the separate main-method
decision that protected subjects 24–29 remain closed.

The canonical adapter evidence is method-specific and is jointly checked by
[`audit_adapter_alignment.py`](../../comparative_methods/audit_adapter_alignment.py).
Across BIOT, CBraMod, REVE, BrainFusion, NormWear, and EFRM it registers 42
method-task cells: 36 supported and 6 preregistered unsupported cells. The
alignment audit itself does not dereference protected identity or arrays;
protected execution and aggregation occurred later under the frozen campaign.
Execution order and unlock prerequisites are defined in
[`EXPERIMENT_PLAN.md`](../../comparative_methods/EXPERIMENT_PLAN.md).
The release candidate intentionally remains a non-authorizing immutable contract
with `protected_evaluation_authorized=false`; authorization, execution, and
unblinding are separate records.

The frozen pre-execution identity is the locally retained, Git-ignored
`comparative_methods/evidence/protected_campaign/joint_release_candidate_v1.json`
plus its authorization and
`comparative_methods/evidence/protected_campaign/orr_preflight_v1.json`.
The candidate contains 540 unique jobs over 36 supported cells; six unsupported
cells have zero jobs and STA-Net has zero new jobs. The retained ORR is the
successful pre-run `GO` snapshot, not a live completion record. Post-run status,
unblinding, aggregate, and 540-row traceability remain in the ignored local run
tree. Their hashes, terminal counts, and the complete primary-result table are
recorded in
[`PROTECTED_CAMPAIGN_RESULTS_20260814.md`](PROTECTED_CAMPAIGN_RESULTS_20260814.md).

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
