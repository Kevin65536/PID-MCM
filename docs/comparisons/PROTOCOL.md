# Shared comparison protocol

_Active benchmark contract, consolidated 2026-07-30_

This document defines the common comparison surface. Method-specific frozen
protocols may be stricter, but may not weaken the data, split, metric, or
protected-evaluation boundaries below. Current execution state is reported in
[`STATUS.md`](STATUS.md); admission of a number to a paper table is governed by
[`METRIC_ACCEPTANCE.md`](METRIC_ACCEPTANCE.md) and the
[machine-readable targets](../../comparative_methods/comparison_metric_targets_v1.yaml).

## Benchmark matrix

The benchmark contains four measured datasets and seven dataset-native tasks.
`croce_local_cache` is derived supervision and is never a fifth dataset.

| Dataset | Task | Target | Primary endpoint |
| --- | --- | --- | --- |
| Single-Trial | motor imagery | LMI/RMI | macro-F1 |
| Single-Trial | mental arithmetic | MA/baseline | macro-F1 |
| Simultaneous | word generation | WG/baseline | macro-F1 |
| Simultaneous | n-back | 0/2/3-back | macro-F1 |
| Simultaneous | DSR | EEG-native Go/No-go | macro-F1 |
| Visual | cognitive motivation | RR/RF/FF/FR | macro-F1 |
| REFED | valence/arousal sequence | masked continuous `[2,T]` | CCC |

Accuracy, class-wise metrics, MAE/RMSE, calibration, runtime, and resource use
are companion outcomes. Classification and regression are not pooled into a
synthetic ranking.

## Shared invariants

- Consume the registry and loader contract in
  [`../DATA_CONTRACT.md`](../DATA_CONTRACT.md); preserve exact task semantics,
  masks, record dependencies, channel identities, and branch hashes.
- Use identical eligible subjects, sample inventories, outer folds, targets,
  and endpoints for methods presented as direct comparisons.
- Label strict cross-subject, within-subject, and sample-random protocols
  separately. A sample-random result is an information-visible diagnostic and
  cannot support new-subject generalization.
- Fit normalization, adapters, target scaling, hyperparameters, checkpoint
  selection, and thresholds only on the training/development partitions
  allowed by the owning frozen protocol.
- Keep protected indices non-dereferenceable until that protocol's explicit
  unlock gate. A completed earlier protocol does not authorize a later one.
- Aggregate outer folds before comparing methods. Classification uses
  macro-F1 as the primary endpoint; REFED uses masked sequence CCC with
  coverage and error companions.
- Preserve seeds and seed dispersion. A best seed, best fold, or repeatedly
  viewed test result is not a formal estimate.
- Every reported value must trace to a frozen config, split registry, manifest,
  code revision, completion status, prediction/metric artifact, and aggregate.

## Execution ladder

| Stage | Required evidence | What it establishes |
| --- | --- | --- |
| C0 — contract | adapter, target/mask assertions, split fingerprint | comparable inputs |
| C1 — software smoke | finite forward/backward, optimizer step, artifact write | connectivity only |
| C2 — source fidelity | pinned upstream revision, deviations, source-task check | named-method boundary |
| C3 — public development | train/validation convergence and diagnostics | tuning evidence |
| C4 — protocol freeze | immutable folds, seeds, endpoints, stopping and unlock rule | estimand fixed |
| C5 — formal execution | every planned fold/seed completes under the freeze | candidate aggregate |
| C6 — final-number audit | machine target and metric-acceptance checks pass | table-admissible value |

Failures, negative results, aborted jobs, and excluded folds remain visible.
Software fidelity, loader correctness, or convergence does not establish
field-wide SOTA or physiological coupling.

## Method boundaries

### STA-Net

The isolated PyTorch implementation lives in
[`comparative_methods/STA-Net-PyTorch`](../../comparative_methods/STA-Net-PyTorch/README.md).
MI/MA/WG are source-task adaptations; n-back, DSR, Visual, and REFED require
explicit multiclass, context, or regression adapters. Original-paper
subject-specific values are contextual and are never treated as same-protocol
comparisons.

### EFRM

The synchronized-data adaptation lives in
[`comparative_methods/EFRM-PyTorch`](../../comparative_methods/EFRM-PyTorch/README.md).
Its active v2 estimand is leave-one-dataset-out pretraining followed by strict
five-fold target-dataset evaluation. Frozen-backbone linear probing is the
primary representation track; full fine-tuning is a separately named,
resource-contingent track. The source-code contrastive multiplier and all
deviations from the upstream data regime remain explicit.

### UMAP

UMAP remains a diagnostic comparison surface. Its repeatedly viewed historical
test results are not admitted as formal benchmark values. A formal rerun must
start from a new frozen cross-subject split/seed contract and may not select
settings using the historical test.

## Minimum retained artifact

For each formal method keep:

```text
frozen protocol + resolved configs + split fingerprints
completion/status manifest + implementation identity
fold/seed metrics + aggregate table + necessary predictions
figures/source data used for interpretation
checkpoint only when replay or continuing work still requires it
```

Large rebuildable caches, superseded tuning checkpoints, and routine smoke
weights may be removed after this evidence package is recorded in
[`../../experiments/RESULTS_INDEX.md`](../../experiments/RESULTS_INDEX.md).
