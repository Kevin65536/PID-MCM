# E0-v3 adaptive teacher admission decision

_Design-decision record · 2026-07-16 · protected subjects 24–29 remain closed_

## Decision

The adaptive local fixed-interval SSM **passes the E0 optional target-family
development gate** as a `physiology_shaped_multimodal_consensus_proxy`.

This is a scoped admission for offline privileged supervision. It is not a
claim that the SSM recovered the true neural source, uniquely identified its
physiological parameters, or independently predicted the complete fNIRS
trajectory from EEG. The protected confirmatory split has not been opened.

## Why the E0 estimand changed

Earlier E0 diagnostics used EEG-only recovery to test a stronger question:
whether a shared state was independently identifiable from either modality and
could itself constitute cross-modal coupling evidence. That is not the target
of the adaptive joint teacher. The intended teacher consumes both EEG and
fNIRS to form a compromise posterior with a neurovascular-dynamics prior, then
provides detached targets to independent modality students.

The active theoretical contract already permits this privileged-information
path. Independent inference means that the trained EEG tokenizer reads only EEG
and the trained fNIRS tokenizer reads only fNIRS. It does not mean that the
offline joint teacher must be reconstructible in full from EEG alone. Frozen
independent student tokens, rather than the fused teacher posterior, remain the
required inputs to the later coupling evaluation.

## Accepted teacher identity

The SSM is treated as a physiology-shaped regularizer and proxy-state
generator. Its bounded Croce/Balloon dynamics restrict the solution to a
neurovascularly plausible temporal family while joint smoothing negotiates the
fast EEG evidence and delayed hemodynamic evidence. The fitted coefficients are
not interpreted as recovered subject physiology.

The admitted paper-level description is:

> We construct a physiology-constrained multimodal consensus proxy from paired
> EEG and fNIRS and use it as privileged supervision for independent modality
> tokenizers.

The following stronger descriptions are not admitted by E0:

- recovered true neural driver;
- uniquely identified subject-specific hemodynamic constants;
- causal neurovascular coupling;
- EEG-only recovery of the complete fNIRS trajectory;
- coupling discovered directly from the fused teacher posterior.

## Evidence supporting admission

| Evidence layer | Result | Admission role |
| --- | --- | --- |
| Measurement/provenance contract | pass | blocking |
| Train-fold chromophore gauge | 230/230 finite and non-singular; reconstruction delta `1.776e-15` | blocking |
| Required EEG local targets | `r_mean` R² `0.451`; `r_slope` R² `0.866` | blocking |
| Required fNIRS local targets | HbO/HbR mean R² `0.725/0.734`; slope R² `0.356/0.472` | blocking |
| K=128 fNIRS target geometry | R² `0.881`, random q95 `0.853`, 92 active codes | blocking |
| Joint compromise behavior | joint/EEG-only driver PCC about `0.927`; fNIRS-induced shift about `0.357` EEG-only SD | supporting |
| Temporal behavior | non-monotonic driver and restored HbO/HbR variation | supporting |
| Continuous coupling upper bound | positive beyond shuffled EEG | bridge diagnostic only |

These results establish a stable and learnable joint proxy. They do not by
themselves establish downstream utility; teacher-free and matched-control
training comparisons remain later-stage experiments.

## Reclassified diagnostics

| Diagnostic | Observed result | Revised role |
| --- | --- | --- |
| fNIRS clean-versus-history physical gain | `-0.08446` | Non-blocking because the proxy is not defined as a raw-signal MSE predictor |
| EEG-only HbO reconstruction | poor | Tests translation/independent recovery, not joint-teacher validity |
| SSM parameter boundary solutions | frequent | Prohibits parameter-recovery claims; compatible with a shaped regularizer |
| Synthetic HbR posterior coverage | outside the old frozen band | Uncertainty weighting not admitted until separately calibrated |
| Raw flow patch-local observability | negative | Flow remains context-only and is excluded from local supervision |

No failed metric is deleted or relabeled as numerically passing. The design
decision changes its role because the target-family estimand is narrower than
the historical physical-source contract.

## Admitted target and loss boundary

- EEG local targets: `r_mean`, `r_slope`.
- EEG `s_mean`, `s_slope`: optional local/prototype development coordinates,
  not blocking coordinates. Their validation R² values are `0.106/0.365`, and
  both exceed their coordinate-wise permutation q95.
- fNIRS local targets: observation-aligned HbO/HbR mean and slope.
- Flow mean/slope: context/coupling-only development coordinates; not
  patch-local or prototype supervision.
- Fitted SSM parameters: provenance/diagnostics only.
- Posterior covariance: stored if available, but not used for inverse-variance
  weighting until a later calibration gate passes.
- Joint teacher: training-only, detached, and absent from tokenizer inference.
- Coupling evaluation: uses frozen independently generated EEG and fNIRS
  student tokens with fNIRS-history, marginal, time-shift, shuffle, subject and
  source controls.

## Governance and next-stage status

The original E0-v3 and gauge-correction run artifacts remain immutable. Their
machine conjunction failed under the earlier rule that treated physical gain
and posterior calibration as blocking layers. This decision is a documented
post-validation estimand revision, not a retroactive claim that the old
preregistered conjunction passed.

The practical status is:

```text
E0 optional target-family development gate: PASS
teacher identity: physiology-shaped multimodal consensus proxy
physical-source identification: NOT CLAIMED
protected confirmatory evidence: NOT OPENED
adaptive runtime/cache integration: NEXT IMPLEMENTATION STEP
```

The next experiments may implement and train this optional teacher on the
existing train/validation development split. Before any confirmatory or paper
claim, the revised contract must be frozen and evaluated under its declared
protected-test policy. Later coupling claims remain governed by the E7–E9
preserve–discover–certify sequence and cannot
be inferred from E0 admission alone.

The subsequent entry-specific routing decision is authoritative for how these
coordinates enter tokenizer and foundation development:
[`20260719_PHYSICAL_TEACHER_GRADIENT_ENTRY_DECISION.md`](20260719_PHYSICAL_TEACHER_GRADIENT_ENTRY_DECISION.md).

## Evidence sources

- [`E0-v3 gauge correction and gate gain`](E0_V3_GAUGE_CORRECTION_GATE_GAIN.md)
- [`E0-D7 adaptive shared SSM`](E0_D7_ADAPTIVE_SHARED_NEURAL_SSM.md)
- [`E0-D8 task-parameter audit`](E0_D8_ADAPTIVE_SSM_TASK_PARAMETER_AUDIT.md)
- [`Active experiment log`](../06_EXPERIMENT_LOG.md)
- `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_teacher_e0_v3_gauge_corrected_validation_v1/`
