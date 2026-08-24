# Development and experiment guide

_Active repository conventions, consolidated 2026-07-30_

This is the operational guide for code, tests, experiment launches, results,
and documentation. Current execution and scientific verdicts are generated in
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md); stable experiment ordering lives
in [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md).

## Repository boundaries

```text
src/                         reusable library code
tests/                       unit, integration, contract, and evidence tests
experiments/configs/         reviewed executable experiment contracts
experiments/scripts/         launch, evaluation, and analysis entrypoints
experiments/runs/            active generated results (ignored by Git)
experiments/archive/         historical generated evidence (ignored by Git)
comparative_methods/         isolated comparison-method implementations
croce_validation/            physical-model validation and derived caches
docs/                        active contracts, status, evidence, and history
data/                        original datasets and derived caches (ignored)
```

Reusable classes and functions belong in `src/`. Executable workflows belong
under `experiments/` or the isolated comparison-method package that owns them.
New physiology-semantic runs write below
`experiments/runs/physiology_semantic_tokenizer/<suite>/<run>/`; comparison
methods write only below their owning package. Archive discovery is always
explicit: active tools do not recursively search `experiments/archive/`, and
compatibility code/configuration remains in its dated namespace.

The R1-P qualification surface is a dated evidence package. Keep its model,
scripts, registries, configuration, and tests together when revisiting that
historical result; ordinary cleanup should not silently mix it with a new
experiment generation.

## Environment

Use the repository environment:

```bash
source .venv/bin/activate
python -m pytest --collect-only -q
```

`requirements.txt` is a CUDA-capable environment snapshot, not a minimal
library specification. Do not assume system Python has the required packages.

## Data contract

Before using a dataset:

1. read [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md);
2. read the original dataset documentation named there;
3. use the central registry and unified loader rather than a new ad hoc parser;
4. preserve dataset-native units, task semantics, masks, channel identities,
   timing anchors, and source provenance;
5. fit normalization or target transforms on the permitted training partition
   only;
6. keep the train/validation/test split and any protected-sample boundary
   declared by the owning protocol.

Raw datasets are reference inputs. Derived caches should carry a schema/version,
source identity, transformation record, and a small manifest. A cached artifact
mask is audit metadata, not an automatic signal-validity mask.

## Configuration and launch

Reviewed YAML/JSON contracts live below `experiments/configs/`. A new
experiment needs:

- an unambiguous experiment ID and output namespace;
- resolved tensor, split, target, mask, and seed assertions;
- primary/secondary/diagnostic endpoint labels;
- explicit null, baseline, stopping rule, and protected-data boundary;
- parser and shape tests;
- dry-run or synthetic execution before measured data access.

The observation–source v2 material is an exploratory design note, not an
executable configuration or fixed architecture. Do not repurpose an E0–E2 or
R-series YAML for it. Any candidate taken forward needs its own versioned
software, tensor-shape, split, and null checks on synthetic data first. A
protected measured run additionally requires its owning protocol and separate
explicit user authorization.

Use the owning launcher. For the physiology-semantic namespace:

```bash
bash experiments/scripts/launch_training_nohup.sh \
  --task physiology-semantic-tokenizer \
  --config experiments/configs/physiology_semantic_tokenizer/p2_p5_software_smoke.yaml \
  --dry-run
```

This launcher can replay the E0–E2-compatible runtime. Launcher availability does
not change the execution state or scientific verdict in the unified registry;
inspect the generated status before defining a new VQ generation. It does not
implement the observation–source exploration map.

## Evidence ladder

Verify work in this order:

1. unit tests;
2. integration and contract tests;
3. dry run;
4. smoke run;
5. short formal run, if the protocol defines one;
6. full public/development run;
7. one-time protected evaluation, only when the owning protocol gate and the
   explicitly user-approved scope are both satisfied.

A run is evidence only when its manifest, resolved configuration, completion
status, summary, and declared endpoint are present. A suite summary cannot
override an individual run record. Failed, negative, aborted, and
scientifically undetermined outcomes remain part of the record.

## Result retention

Keep the smallest package that preserves the scientific conclusion and normal
comparison use:

- resolved configuration, split/registry identities, manifests, summaries,
  tables, figures, alt text, and decision records;
- a checkpoint only when it is still needed for an active run, a recurring
  analysis, a consumer interface, or an irreplaceable reference;
- raw predictions or arrays only when the reported result cannot be audited or
  regenerated from retained material at acceptable cost.

Routine smoke checkpoints, older tuning checkpoints, duplicated token
exports, and rebuildable caches do not belong in the long-term result surface.
The retained evidence map is
[`experiments/RESULTS_INDEX.md`](experiments/RESULTS_INDEX.md).

Never clean a directory used by a live process. Query the unified project state and
inspect the owning process/controller immediately before cleanup; dated prose is not
a live-process detector.

## Tests

The active test taxonomy is documented in [`tests/README.md`](tests/README.md).
Run targeted tests after a local change, then:

```bash
python -m pytest --collect-only -q
git diff --check
```

Some formal-artifact tests intentionally depend on local data or sealed
evidence. Do not weaken or mark a sealed test merely to make a clean checkout
green; introduce a separate small fixture or revise the seal explicitly.

## Documentation

[`docs/README.md`](docs/README.md) is the only documentation authority map.
Current execution and scientific verdicts live in
[`research_state/registry.json`](research_state/registry.json), which has exactly
those two status axes. Do not copy current state, job counts, or next actions into
hand-written README files.

When an experiment changes state, edit the corresponding current item in
`research_state/registry.json`, then run:

```bash
.venv/bin/python experiments/scripts/project_state.py validate
.venv/bin/python experiments/scripts/project_state.py render
```

Dated reports preserve what was known at the time. Extra evidence and audit checks
are optional when preparing a paper-ready frozen result; they are not required for
ordinary progress updates.

Generated reports, manuscript exports, and literature PDFs are communication
or reference assets, not active implementation authority.
