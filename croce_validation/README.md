# Croce Real-Data Validation Workspace

This directory is intentionally independent from `experiments/`.

It is reserved for standardized validation of the Croce 2017 physical-model approximations on the EEG+NIRS Single-Trial motor-imagery task.

## Scope

This workspace is only for:

1. forward-approximation comparisons,
2. inverse-stability and identifiability checks,
3. event-locked physiological plausibility analysis,
4. denoising-utility comparisons on real data.

This workspace is not for:

1. tokenizer training,
2. cross-task extensions,
3. general experiment runs unrelated to Croce real-data validation.

## Layout

```text
croce_validation/
  README.md
  scripts/
  results/
```

## Conventions

1. New standardized Croce validation scripts should be created under `croce_validation/scripts/`.
2. New outputs should be written under `croce_validation/results/<run_name>/`.
3. Existing exploratory assets under `experiments/` are reference material only and should not receive new standardized outputs.