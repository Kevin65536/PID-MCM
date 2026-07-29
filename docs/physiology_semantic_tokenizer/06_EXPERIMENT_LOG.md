# Physiology-semantic tokenizer experiment log

_Current decision registry; detailed historical reports remain under `analysis/`_

## Current state

E0–E2 is a closed historical generation. Its final development decision was
`no_semantic_row_admitted_retain_T0`, followed by an architecture return to a
minimal Shared-Driver Semantic VQ design. The R-series then tested whether a
continuous shared-driver target was strong enough to justify quantization.

That prerequisite was not met. R0-P did not reproduce the preregistered raw
alpha–HbO lag association. R2-D failed the bilateral full-trajectory
observability criterion. R1-P produced a non-degenerate and bilaterally
decodable population-frozen coordinate, but failed the preregistered physical
reconstruction consistency gate. D1B validation terminated before endpoint
evaluation because of a serializer defect and is scientifically undetermined.

The active decision is:

```text
promotion_eligible = false
next_action = do_not_enter_r2_p
protected_subjects_24_29 = closed
```

The authoritative R-series methods, numerical results, diagnostic
interpretation and lessons are consolidated in
[`analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md`](analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md).

## R-series registry

| Date | ID | Scope | Outcome |
| --- | --- | --- | --- |
| 2026-07-28 | `R0-P` | Raw EEG bandpower–fNIRS lag baseline, subjects 01–23 | Registered alpha–HbO endpoint negative; no 30-family FWER discovery |
| 2026-07-28 | `R1-D` | Subject-fitted teacher geometry | Exploratory correction geometry only |
| 2026-07-28 | `R2-D` | One-seed continuous raw-only trajectory prediction | Bilateral endpoint failed; VQ not authorized |
| 2026-07-28 | `R1-P-STRUCTURE` | Population-frozen bundle structure and leakage | Passed; 1,080 fit and 300 pure-apply windows |
| 2026-07-28 | `R1-P-FORMAL-V3` | Frozen six-gate teacher qualification | G2 failed; G3–G6 passed; G1 invalid due dtype contract |
| 2026-07-28 | `R1-P-POSTFORMAL` | Failure localization and adaptation diagnostics | Subject/chromophore heterogeneity; no stable common correction phase |
| 2026-07-28 | `D1B-TRAIN` | Nested train-only adaptation search | Shrinkage 0.1 selected; asymmetric compensation |
| 2026-07-28 | `D1B-VALIDATION-V2` | Single measured-access validation attempt | Serializer stopped before endpoints; scientific result undetermined |

## Historical generation

The earlier generation remains reproducible through its tracked configs and
reports, but none of its intermediate statuses supersedes the R-series stop
decision.

| Period | Scope | Final interpretation |
| --- | --- | --- |
| 2026-07-01 to 2026-07-18 | E0 physical-teacher development | Sign-calibrated adaptive teacher accepted for development supervision, without an identifiability claim |
| 2026-07-19 to 2026-07-22 | E1 quantizer health and retention | K=128 occupancy/retention gate passed for the selected development candidate |
| 2026-07-22 to 2026-07-25 | E2 semantic objective suite | 9/9 completed; no semantic row admitted, retain T0 |
| 2026-07-25 | Architecture return | Replace the multi-entry route with continuous shared-driver qualification before any new VQ |

The detailed historical chronology is preserved by the dated E0/E1/E2
analysis files and local run manifests. This log intentionally records only
generation-level decisions so it cannot drift into a second, competing
technical report.

_Last updated: 2026-07-29_
