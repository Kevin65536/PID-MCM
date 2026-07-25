# STA-Net final MI/WG targeted development round

Date frozen: 2026-07-24, before launching the study.

## Scope and claim boundary

This is the last development-split hyperparameter exploration for the STA-Net
cross-subject MI and WG adapters. It uses the existing immutable
`development_cross_subject` subject split and must not open protected test
manifests. The round selects final candidate configurations; its validation
scores are not themselves protected cross-subject performance estimates.

No additional search round may be started after inspecting these results.
Implementation failures may be repaired only if the affected trials are
retained in the audit and the same repair is applied without reference to task
outcomes.

## Motivation

The preceding v2 study selected MI at epoch 4 and WG at epoch 6. Early-rung
rank correlations were not uniformly reliable across the seven-task suite, so
the final targeted round separates two questions:

1. whether low-learning-rate configurations were previously removed before
   they could converge; and
2. whether stronger regularization reduces the early-peak/late-degradation
   pattern in MI and WG.

## Frozen design

- Tasks: `motor_imagery`, `wg`.
- Trials: 16 per task.
- Seed and split: unchanged (`42` and the existing development cross-subject
  split contract).
- Rungs: 2, 8, 20, 40, and 100 epochs.
- Pruning: disabled before epoch 20; Hyperband may prune only at epoch 20 or
  40.
- Objective: best validation macro-F1 observed through each rung.
- TPE startup setting: 8 trials per task.
- Four enqueued anchors per task:
  - `control_like`: the preceding winner's local neighborhood;
  - `slow`: lower learning rate and longer warmup;
  - `regularized`: stronger dropout, weight decay, and label smoothing;
  - `slow_regularized`: both interventions.
- Remaining trials use the frozen conditional ranges in
  `configs/final_mi_wg_targeted.yaml`.

## Outcome reporting

The audit must report, for every intervention family:

- trial state and budget reached;
- best validation macro-F1 and checkpoint epoch;
- epoch-20 and epoch-100 performance where available;
- best-to-epoch-100 degradation;
- configuration and checkpoint hashes.

The final development candidate is the completed 100-epoch trial with the
highest historical validation macro-F1 under the frozen objective. Stability
and endpoint degradation are mandatory secondary diagnostics, but are not used
post hoc to replace the primary selection rule.

The previous v2 winner remains an external development reference, not an extra
candidate silently merged into the new Optuna study. Its control-like anchor is
rerun inside this study under the same 100-epoch completion requirement.

## Formal-result boundary

Adopting a final cross-subject performance number requires freezing the
selected configuration and then evaluating the protected outer-subject
protocol once. The development score selected here must remain labeled as a
development/tuning estimate.
