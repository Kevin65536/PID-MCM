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

## Current state

This workspace does not maintain a second status summary. Query the generated
[`PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md) or run:

```bash
.venv/bin/python experiments/scripts/project_state.py show --format agent
```

Launcher availability is an implementation fact, not evidence that a scientific
prerequisite has passed. Live-process checks must still be performed immediately
before touching generated run or cache directories.

## Observation–source exploration boundary

The v2 exploration is a replaceable, not-yet-implemented candidate map. It is
recorded in the [design note](../docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json)
and its [framework diagram](../docs/physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg).
No YAML or measured-data run is authorized by those artifacts.

The existing physiology-semantic YAML/runtime surface is historical and replay-only;
do not clone or reinterpret it as a new contract. A selected candidate must first
pass synthetic software, target/teacher, tensor-shape, split, and null checks.
Only after those checks are recorded may a development comparison be proposed.
Protected data requires the owning protocol and a separate, explicit user
authorization for that measured action.

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
