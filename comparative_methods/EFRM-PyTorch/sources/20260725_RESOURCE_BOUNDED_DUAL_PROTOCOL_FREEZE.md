# EFRM resource-bounded dual-protocol evaluation freeze

**Protocol ID:** `efrm_resource_bounded_dual_protocol_v1`
**Status:** protocol frozen; cohort and fold manifests not yet materialized
**Frozen on:** 2026-07-25
**Machine-readable contract:** [`resource_bounded_dual_protocol_v1.yaml`](resource_bounded_dual_protocol_v1.yaml)

## Authority and purpose

This document is the normative protocol for collecting downstream performance
from `efrm_sync_200_10_variable_channel_v1` under the current compute budget.
It authorizes one source-only EFRM pretraining run followed by two target-only
five-fold downstream evaluations: strict cross-subject transfer and direct
sample-level random transfer. It does not authorize five fold-specific EFRM
pretraining runs, because that computation is unavailable, and it does not
represent one global in-domain checkpoint as an inductive five-fold model.

The protocol changes the estimand relative to the current full-dataset STA-Net
five-fold benchmark. EFRM estimates transfer from a fixed source cohort to a
disjoint target cohort. Direct numerical ranking against the existing
full-dataset STA-Net aggregate is therefore prohibited. A direct method
comparison requires STA-Net to be evaluated on the exact EFRM target cohort
and fold manifests; otherwise the EFRM results occupy a separate
resource-bounded transfer table.

This freeze supersedes the earlier default of retraining EFRM inside every
outer fold only for the active resource-bounded track. Fold-specific
pretraining remains the requirement for a future exact full-dataset,
fold-matched EFRM benchmark.

## Frozen cohort boundary

Source and target membership is assigned once at the dataset level, before any
target outcome is inspected. Canonical subject identifiers are sorted and
passed to a shuffled three-fold `KFold` with seed 42. The first test partition
is the source cohort and its complement is the target cohort. Tasks from the
same dataset inherit the same dataset-level subject assignment; task-specific
eligibility is applied only after this assignment. The build must fail if any
task retains fewer than five eligible target subjects.

The source cohort is the complete allowable input to EFRM pretraining. Within
each dataset, its canonical source subjects are subdivided with shuffled
`KFold`, seed 43, and `n_splits=min(5, number_of_source_subjects)`. The first
test partition is source validation and its complement is source training.
The build fails when a dataset cannot supply at least two source subjects.
Only this deterministic source-validation partition may be used for
self-supervised stopping and checkpoint selection. No target subject, target
sample, target-fitted statistic, or target-derived training decision may
enter pretraining. The target cohort is the complete allowable input to both
downstream five-fold protocols and may not be used for pretraining monitoring,
early stopping, checkpoint selection, normalization, or diagnostic-driven
model revision.

Every cohort manifest must record the event-index and metadata fingerprints,
the exact canonical subject lists by dataset, task-eligible subject counts,
sample-index hashes, construction seed, builder implementation hash, and
`target_opened_during_pretraining: false`. Changing a cohort member, admission
rule, cache fingerprint, or subject canonicalization creates a new protocol
version.

## Single source-only pretraining run

EFRM is initialized once from random weights and trained on source-cohort
samples using the architecture, objectives, sampling rates, patch durations,
mask policy, optimizer, and synchronized-pair data regime in
`configs/pretrain_sync.yaml`. Pretraining seed 42 is fixed. The best checkpoint
minimizes total validation loss on the recorded source-validation partition;
the configured minimum-epoch and patience rules remain unchanged. The
selected checkpoint, resolved configuration, source boundary, implementation
hashes, and checkpoint SHA-256 must be frozen before target fold training
begins.

There is no claim that one pretraining seed estimates pretraining
initialization uncertainty. The five-fold standard deviation defined below
measures target-fold variation conditional on this one frozen checkpoint.
Snapshot checkpoints, different pretraining epochs, or checkpoints selected
after target outcomes are observed are not independent replicates and cannot
be substituted for fold variation.

The existing `20260722_efrm_sync_dev_v5` checkpoint is development evidence
only. It was trained and selected under a different public boundary and cannot
populate either primary result family in this protocol.

## Target-only strict cross-subject five-fold protocol

For each task, eligible target subjects are split with shuffled five-fold
`KFold` using seed 42. Each outer test fold contains subjects absent from the
other four folds. The remaining target subjects form the outer development
partition and are subdivided into three subject-level inner folds using the
frozen seed `43 + outer_index`; inner fold zero is the validation partition
and the other two inner folds are the labeled training partition.

The five outer test folds must partition all eligible target subjects and all
their task samples exactly once. Test subjects are absent from EFRM
pretraining, downstream training, downstream validation, fitted
normalization, regression scaling, class weighting, checkpoint selection, and
hyperparameter choice. This result family is reported as
`efrm_source_target_strict_cross_subject_5fold_v1`.

## Target-only sample-random five-fold protocol

For classification, all eligible target sample indices are split with
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. REFED uses
shuffled five-fold `KFold` with the same seed. The remaining 80% outer
development samples are divided into three inner folds, with classification
again stratified by class, using seed `43 + outer_index`, and inner fold zero
fixed as validation. The other two inner folds form labeled training.

This protocol deliberately does not isolate subject, record, session, trial,
or video dependency groups. The same participant and acquisition context may
therefore occur in train, validation, and test, matching the intended direct
sample-level diagnostic. Exact sample indices remain mutually exclusive, and
the five outer test folds must partition all eligible target samples exactly
once. Target regression scalers and every other learned preprocessing
quantity are fitted on the current outer fold's labeled training indices
only. This result family is reported as
`efrm_source_target_sample_random_5fold_v1` and must never be described as
new-subject generalization.

The sample-random result is an intentionally optimistic estimate of
performance when participant and acquisition dependencies are allowed to
cross the split. Its difference from the strict result is a protocol
sensitivity measure, not evidence that subject identity or repeated-context
information is physiologically meaningful.

## Downstream training and selection

Frozen-backbone paired-modality linear probing with pretrained initialization
is the required primary EFRM transfer mode for all seven tasks. Each outer
fold starts from the identical frozen source-only checkpoint and an
independently initialized downstream head with seed 42. The hyperparameters
and task overrides originate from `configs/downstream_public_pilot.yaml` and
must be copied into a hash-pinned protocol-specific runtime configuration
before protected target evaluation; target outcomes cannot tune them.
Classification checkpoints maximize validation macro-F1. REFED checkpoints
minimize masked scaled RMSE, with scalers fitted only on the current training
partition.

Full fine-tuning is a secondary resource-contingent experiment. It may begin
only after the complete two-protocol linear-probe matrix is immutable and
must use the same folds, checkpoint, label access, selection rules, and metric
definitions. A partial full-fine-tuning matrix cannot replace missing
linear-probe cells or become the headline EFRM result.

The required primary grid contains 70 completed linear-probe jobs: seven
tasks, two protocols, and five outer folds at one fixed downstream seed.
Failures remain visible with a pre-outcome reason. Results are not aggregated
or selectively published from an incomplete grid unless the affected task
was declared ineligible before any target outcome was inspected.

## Metrics and uncertainty

Classification uses outer-fold macro-F1 as its primary endpoint and
outer-fold Accuracy as its source-aligned companion endpoint. Balanced
Accuracy, Cohen's Kappa, per-class precision, recall and F1, confusion
matrices, calibration, loss, sample counts, and subject counts are required
supporting outputs. REFED uses native-coordinate concordance correlation
coefficient as its primary endpoint and requires native MAE, RMSE, R-squared,
Pearson and Spearman correlation, valid-target coverage, and valid counts.

For a metric \(m\), the primary mean and sample standard deviation are
computed over the five outer-fold values:

\[
\bar m = \frac{1}{5}\sum_{j=1}^{5}m_j,\qquad
s = \sqrt{\frac{1}{4}\sum_{j=1}^{5}(m_j-\bar m)^2}.
\]

The report must label this quantity as `sample SD across five target outer
folds (ddof=1)`. A two-sided fold-level 95% t interval with four degrees of
freedom is required as an uncertainty companion. Concatenated out-of-fold
predictions and pooled metrics are retained as diagnostics but cannot replace
the fold mean or fold SD. If extra downstream seeds are later authorized,
their metrics are averaged within each outer fold before the five fold means
and fold SD are computed; seed SD is reported separately and never relabeled
as fold SD.

## Protected evaluation and artifact requirements

Public manifests expose only the current fold's training and validation
indices. Protected manifests expose only the corresponding test indices and
remain locked until the source-only checkpoint, downstream configuration,
selection rules, metric registry, job matrix, and implementation hashes are
frozen. Each protected fold is evaluated once. Any revision after protected
outcomes are opened requires a new protocol ID and new protected evidence.

Every result cell must resolve to the cohort manifest, public and protected
fold manifests, source-only checkpoint hash, resolved downstream
configuration, implementation hashes, best-validation checkpoint,
predictions, fold metrics, runtime and memory measurements, completion
status, and `protected_test_opened` provenance. The aggregator must verify
five-fold completeness and exact target-test partition coverage before
writing a summary.

## Claim and comparison boundaries

The two admitted result families are source-to-target transfer estimates
conditioned on one source-only EFRM checkpoint. They are not numerical
reproductions of the paper's 1,247.5-hour pretraining result, do not estimate
pretraining-seed uncertainty, and do not validate EEG-fNIRS CLIP alignment.
The regression task remains the explicitly named
`efrm_sync_regression_adapter`.

A checkpoint pretrained on any target sample may be evaluated only under the
separate name `efrm_transductive_sample_random_diagnostic`. Such a diagnostic
is excluded from the primary strict and sample-random tables, even when labels
were hidden during pretraining. Bootstrap intervals, repeated downstream
seeds, snapshot ensembles, warm-started fold adaptation, and five downstream
folds cannot repair target exposure during pretraining.

Direct EFRM-versus-STA-Net claims require matched target cohort membership,
fold manifests, labels, modality, endpoint, and result family. Until that
matched STA-Net rerun exists, the current full-dataset STA-Net five-fold
aggregate and this EFRM resource-bounded aggregate must appear in separate
tables with no shared rank or unqualified performance delta.

## Implementation gates

Implementation may proceed only in the following order:

1. Materialize and audit immutable dataset-level source/target cohort
   manifests without opening target outcomes.
2. Build source-only pretraining train/validation manifests and prove zero
   target membership or fitted state.
3. Complete and freeze the single source-only EFRM checkpoint.
4. Materialize both target-only five-fold registries and verify exact
   partition coverage, class support, and hashes.
5. Freeze the linear-probe job matrix and metric registry.
6. Run all public training/validation jobs.
7. Open protected target folds once and aggregate only after complete success.
8. Run a matched STA-Net target-cohort benchmark before making direct
   cross-method claims.

No later implementation document, launcher default, or experiment note may
silently weaken these gates. A deliberate change must identify the affected
clause, increment the protocol version, and preserve this document as the
historical frozen record.
