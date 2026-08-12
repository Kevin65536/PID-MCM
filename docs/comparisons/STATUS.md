# Comparison experiment status

_Snapshot: 2026-08-12; retained run manifests and final alignment summaries are
the source of truth_

## Current decision surface

| Method / track | State | Evidence and next action |
| --- | --- | --- |
| STA-Net strict five-fold | **Complete; context reference** | 70/70 fold trainings completed. The retained scores use STA-Net's method-native observation budget and are not yet support-matched direct-table evidence. |
| EFRM LODO v2 Stage A | **Complete** | 4/4 target-dataset-excluded selection jobs completed and selected epochs were frozen. |
| EFRM LODO v2 Stage B | **Complete** | 4/4 full non-target refits completed on 2026-08-03; checkpoint, boundary, config, and run-manifest identities are frozen. |
| EFRM v2 downstream matrix | **Public A0-A8 complete; protected locked** | All 105/105 public frozen-feature probe jobs completed serially with 0 failures and 0 retries. The seven retained cells passed independent full-public replay and alignment audits; all public summaries remain `table_admissible=false`. |
| EFRM v2 full fine-tuning | **Conditional** | Resource-contingent secondary track; cannot replace the frozen linear-probe matrix. |
| BIOT / CBraMod | **Public A0-A8 complete; protected locked** | Each method completed 90/90 public jobs over six classification tasks. REFED v1 is preregistered unsupported. The retained public-validation summaries are explicitly not table-admissible. |
| REVE | **Public A0-A8 complete; protected locked** | 90/90 public jobs completed. Single-Trial MI/MA remain in the declared target-corpus-overlap track; REFED v1 is unsupported. The retained public-validation summaries are not table-admissible. |
| BrainFusion NVC-CSP Stacking | **Public A0-A8 complete; protected locked** | 75/75 public jobs completed over MI, MA, WG, n-back, and Visual. DSR and REFED are preregistered unsupported. |
| NormWear EEG-fNIRS adapted | **Public A0-A8 complete; protected locked** | 90/90 public jobs completed over the six classification tasks. REFED is preregistered unsupported because a truthful partial-time-mask regression contract is absent. |
| UMAP | **Retired from active queue** | No formal rerun is planned; prior repeatedly viewed test results remain historical Git context only. |
| Cross-method final table | **ORR NO-GO; protected closed** | The 42-cell eligibility review and 540-job release candidate are frozen. Two complete CPU shadow passes succeeded, but only one GPU was healthy and idle, so no lane manifest exists; the authorization template is false/pending and unsigned. |

All new-method and EFRM v2 protected data remain closed:
`protected_test_opened=false`, `target_dataset_exposure=false`. Its live source
of truth is
[`status.json`](../../comparative_methods/EFRM-PyTorch/runs/formal/efrm_lodo_full_target_fivefold_v2/status.json).

The canonical adapter evidence is method-specific and is jointly checked by
[`audit_adapter_alignment.py`](../../comparative_methods/audit_adapter_alignment.py).
Across BIOT, CBraMod, REVE, BrainFusion, NormWear, and EFRM it currently
registers 42 method-task cells: 36 direct-profile passes and 6 preregistered
unsupported cells. No protected identity or array is dereferenced by this
audit. Execution order and unlock prerequisites are defined in
[`EXPERIMENT_PLAN.md`](../../comparative_methods/EXPERIMENT_PLAN.md).
The candidate is adapter eligibility evidence only and retains
`protected_evaluation_authorized=false`.

The current protected-campaign source of truth is the
[`joint_release_candidate_v1.json`](../../comparative_methods/evidence/protected_campaign/joint_release_candidate_v1.json)
plus the latest
[`orr_preflight_v1.json`](../../comparative_methods/evidence/protected_campaign/orr_preflight_v1.json).
The candidate contains 540 unique jobs over 36 supported cells; six unsupported
cells have zero jobs and STA-Net has zero new jobs. The ORR is deliberately
`NO_GO`: candidate state `DRAFT`, lane missing, fewer than two healthy idle
GPUs, and no valid dual-signature authorization. No formal protected campaign
directory exists and `protected_test_opened=false`.

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
