# E0-v3 adaptive teacher admission decision

_Design-decision record · 2026-07-16 · protected subjects 24–29 remain closed_

## Decision

The adaptive local fixed-interval SSM, after observation-aligned sign
calibration, **passes complete E0**. Its physiological outputs, including the
fNIRS state/observation content, are fully accepted for offline physical-teacher
supervision.

The earlier scoped `physiology_shaped_multimodal_consensus_proxy` wording is
superseded by the final 2026-07-24 E0 correction. E0 acceptance does not by
itself claim unique parameter identification, causal direction, or downstream
token/coupling performance.

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

The SSM is accepted as the physical teacher. Its bounded Croce/Balloon dynamics
restrict the solution to a neurovascularly plausible temporal family while
joint smoothing negotiates the fast EEG evidence and delayed hemodynamic
evidence. The fitted coefficients are
not interpreted as recovered subject physiology.

The admitted paper-level description is:

> We construct a sign-calibrated, physiology-constrained physical teacher from
> paired EEG and fNIRS and use its accepted SSM physiological information as
> privileged supervision for independent modality tokenizers.

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

These results establish acceptable physiological supervision from the
sign-calibrated adaptive SSM physical teacher. They do not by themselves
establish downstream utility; teacher-free and matched-control training
comparisons remain later-stage experiments.

## Reclassified diagnostics

| Diagnostic | Observed result | Revised role |
| --- | --- | --- |
| fNIRS clean-versus-history physical gain | `-0.08446` | Retired pre-sign-calibration diagnostic; carries no current E0 gate status |
| EEG-only HbO reconstruction | poor | Tests translation/independent recovery, not joint-teacher validity |
| SSM parameter boundary solutions | frequent | Prohibits parameter-recovery claims; compatible with a shaped regularizer |
| Synthetic HbR posterior coverage | outside the old frozen band | Uncertainty weighting not admitted until separately calibrated |
| Raw flow patch-local observability | negative | Flow remains context-only and is excluded from local supervision |

No historical number is deleted. The sign-calibrated decision corrects its
interpretation: a metric computed in the sign-ambiguous coordinate system is
not a valid failure indicator for the calibrated physical teacher.

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
old machine label is superseded for current governance because it evaluated the
pre-calibration sign-ambiguous representation. The authoritative current result
is complete E0 `PASS`.

The practical status is:

```text
complete E0: PASS
teacher identity: sign-calibrated adaptive SSM physical teacher
SSM-derived physiological information: FULLY ACCEPTABLE
physical-teacher supervision: AUTHORIZED
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
