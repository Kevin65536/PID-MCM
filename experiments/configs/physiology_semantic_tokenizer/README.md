# Physiology-semantic tokenizer configurations

This directory contains executable experiment contracts. Generated variants,
local tuning files, run outputs and abandoned preregistries are intentionally
excluded from Git.

## Current boundary

The E0–E2 generation is historical. Its tracked YAML files remain available
for exact replay, but new work must not infer an R-series contract from them.
The current R-series conclusion is `do_not_enter_r2_p`: no VQ or token
co-occurrence experiment is authorized.

The versioned R-series surface is deliberately limited to:

| File | Purpose | Status |
| --- | --- | --- |
| `r0p_raw_lag_baseline.yaml` | Preregistered raw EEG–fNIRS lag benchmark | Completed; primary result negative |
| `r1p_population_frozen_teacher.yaml` | Fit on subjects 01–18 and pure-apply on 19–23 | Completed; structural audit passed |
| `r1p_teacher_qualification_registry.json` | Frozen G1–G6 gate definitions | Formal-v3 did not qualify |
| `r1p_teacher_perturbation_registry.json` | Three finite train-only G4 stress bundles | Completed |
| `r2d_continuous_observability.yaml` | One-seed development continuous observability | Completed; bilateral endpoint failed |
| `token_physiology_atlas.yaml` | Versioned descriptive analysis contract for an already trained tokenizer | Development-only; does not authorize a new VQ or coupling experiment |

Two matching evidence contracts live under
`docs/physiology_semantic_tokenizer/architecture/`:
`r0p_raw_lag_baseline_preregistry.json` and
`r1p_prevalidation_seal.json`. They are required to replay the corresponding
formal scripts and must not be edited retrospectively.

## Execution contract

Every new versioned config must have a parser test, shape and split assertions,
a dry-run or synthetic execution path, and an output namespace below:

```text
experiments/runs/physiology_semantic_tokenizer/<suite>/<run>/
```

Subjects 01–18 are the development-fit cohort, subjects 19–23 are
development pure-apply, and subjects 24–29 remain protected. A config flag
cannot relax this boundary. Teacher-supervised runs also require exact
registry and seal identities; teacher-free runs must set all teacher-derived
loss weights to zero.

The consolidated methods, results, interpretation and stop decision are in
[`20260728_R_SERIES_EXPERIMENT_REPORT.md`](../../../docs/physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md).

_Last updated: 2026-07-29_
