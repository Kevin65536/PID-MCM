# Experiment workspace

_Retained contracts, replay workflows, and evidence; no active experiment flow_

## Directory roles

| Path | Role |
| --- | --- |
| [`configs/physiology_semantic_tokenizer/`](configs/physiology_semantic_tokenizer/README.md) | retained stopped R-series and analysis contracts |
| `scripts/` | replay/evidence, state, and figure tools |
| [`runs/`](runs/README.md) | retained generated evidence; future run root only after registration |
| `archive/` | local superseded generations; Git-ignored and never default-discovered |
| [`RESULTS_INDEX.md`](RESULTS_INDEX.md) | retained-result map and pruning record |

Comparison methods own their code, configs, runs, and caches below
`comparative_methods/<method>/`. Croce validation owns
`croce_validation/`. Do not create a second generic results root.

## Recorded state

This workspace does not maintain a second status summary. Query the generated
[`PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md) or run:

```bash
.venv/bin/python experiments/scripts/project_state.py show --format agent
```

Live-process checks must be performed immediately before touching generated run
or cache directories.

## Frozen method boundary and implementation candidates

The theory/architecture principles are retained in
[`METHOD_RATIONALE.md`](../docs/METHOD_RATIONALE.md). The v2 exploration is an
abandoned, not-yet-implemented pre-freeze candidate map. It is
recorded in the [design note](../docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json)
and its [framework diagram](../docs/physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg).
No YAML or measured-data run is authorized by those artifacts.

The existing physiology-semantic YAML/runtime surface is stopped historical and replay-only;
do not clone or reinterpret it as a new contract. An implementation inside the
frozen boundary must first
pass synthetic software, target/teacher, tensor-shape, split, and null checks.
No development comparison is currently proposed; any future flow would first record
those checks.
Protected data requires the owning protocol and a separate, explicit user
authorization for that measured action.

## Run contract

Any future registered physiology-semantic output uses:

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
The clean-slate entry (with no schedule) is [`../docs/EXPERIMENT_PLAN.md`](../docs/EXPERIMENT_PLAN.md).
