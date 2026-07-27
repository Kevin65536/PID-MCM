# STA-Net no-artifact-mask convergence rerun

Date: 2026-07-27

## Data-validity decision

STA-Net reads `valid_mask` directly. This mask represents only real record
support and padding boundaries. It does not read `analysis_valid_mask` or
`artifact_mask`, so historical dataset-specific artifact detections cannot
zero samples or invalidate time points. The frozen policy identifier is
`disabled_all_false_no_invalid_authority_v1`.

Independently defined bad-channel handling remains part of spatial projection;
it is not a sample-level artifact interval label.

## Convergence decision

The task configurations selected on development data remain fixed. The rerun
does not select new hyperparameters from previously opened outer-test scores.
It replaces the fixed 100-epoch cutoff with one common convergence controller:

- maximum safety budget: 300 epochs;
- checkpoint monitor: validation Macro-F1 for classification and validation
  scaled RMSE for REFED regression;
- reduce learning rate by 0.5 after 8 validation epochs without an absolute
  improvement of at least `1e-4`;
- minimum learning rate: `1e-6`;
- train for at least 40 epochs;
- stop after 30 validation epochs without checkpoint-metric improvement.

Every formal fold must terminate through the convergence rule. Reaching the
maximum budget is a failed protocol, and protected evaluation/aggregation is
not permitted for that fold.

## Reporting

The formal aggregate reports arithmetic mean and sample standard deviation
(`ddof=1`) across five outer folds. Classification primary results use
Macro-F1; REFED regression uses CCC. Accuracy and the remaining classification
and regression metrics are retained for paper tables and diagnostics.
