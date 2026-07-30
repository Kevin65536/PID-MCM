# Experiment workspace

_Configs, executable workflows, active evidence, and explicit archives_

## Directory roles

| Path | Role |
| --- | --- |
| [`configs/physiology_semantic_tokenizer/`](configs/physiology_semantic_tokenizer/README.md) | versioned E0–E2, R-series, and analysis contracts |
| `scripts/` | training, qualification, evaluation, analysis, and figure entrypoints |
| [`runs/`](runs/README.md) | active-design generated evidence |
| [`archive/`](archive/) | historical generated evidence; never default-discovered |
| `configs/archive/`, `scripts/archive/` | explicit compatibility surfaces |
| [`RESULTS_INDEX.md`](RESULTS_INDEX.md) | retained-result map and pruning record |

Comparison methods own their code, configs, runs, and caches below
`comparative_methods/<method>/`. Croce validation owns
`croce_validation/`. Do not create a second generic results root.

## Current authorization

The E2-compatible launcher remains runnable, but the main method is stopped at
`do_not_enter_r2_p`. The presence of R-series scripts/configs is evidence of
completed diagnostics, not authorization for R2-P or a new VQ run.

At the 2026-07-30 cleanup snapshot, EFRM LODO v2 is live. Its entire
`comparative_methods/EFRM-PyTorch/runs/` tree, method caches, and the clean
physiology cache it consumes are excluded from cleanup.

## Run contract

New physiology-semantic output uses:

```text
experiments/runs/physiology_semantic_tokenizer/<suite>/<immutable-run>/
```

A result is auditable only when its resolved config, manifest, split/cache/code
identity, completion status, primary endpoint, summary/table, and necessary
prediction or figure source data are present. Suite summaries do not override
run records.

Generated payloads remain ignored by Git. Lightweight manifests, decision
summaries, and evidence indexes may be force-tracked intentionally. Historical
analysis always names an exact archive path; active tools never recurse through
archives.

Launch and evidence conventions are in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
The full schedule is [`../docs/EXPERIMENT_PLAN.md`](../docs/EXPERIMENT_PLAN.md).
