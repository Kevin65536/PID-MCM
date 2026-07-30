# Development and experiment guide

_Active repository conventions, consolidated 2026-07-30_

This is the operational guide for code, tests, experiment launches, results,
and documentation. Scientific status and experiment ordering live in
[`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md); this file does not
authorize a blocked experiment.

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

The R1-P qualification seal fixes exact paths and hashes for its model,
scripts, registries, configuration, and tests. Those files must not be moved or
edited as ordinary cleanup. A scientific seal revision is required first.

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
   timing anchors, and source hashes;
5. fit normalization or target transforms on the permitted training partition
   only;
6. keep protected samples closed until the owning frozen protocol explicitly
   opens them.

Raw datasets are immutable. Derived caches must carry a schema/version,
source identity, transformation record, and manifest. A cached artifact mask
is audit metadata, not an automatic signal-validity mask.

## Configuration and launch

Reviewed YAML/JSON contracts live below `experiments/configs/`. A new
experiment needs:

- an unambiguous experiment ID and output namespace;
- resolved tensor, split, target, mask, and seed assertions;
- primary/secondary/diagnostic endpoint labels;
- explicit null, baseline, stopping rule, and protected-data boundary;
- parser and shape tests;
- dry-run or synthetic execution before measured data access.

Use the owning launcher. For the physiology-semantic namespace:

```bash
bash experiments/scripts/launch_training_nohup.sh \
  --task physiology-semantic-tokenizer \
  --config experiments/configs/physiology_semantic_tokenizer/p2_p5_software_smoke.yaml \
  --dry-run
```

This launcher can replay the E0–E2-compatible runtime. It does not override the
current `do_not_enter_r2_p` decision and is not permission to start a new VQ
generation.

## Evidence ladder

Verify work in this order:

1. unit tests;
2. integration and contract tests;
3. dry run;
4. smoke run;
5. short formal run, if the protocol defines one;
6. full public/development run;
7. one-time protected evaluation, only when its gate is satisfied.

A run is evidence only when its manifest, resolved configuration, completion
status, summary, and declared endpoint are present. A suite summary cannot
override an individual run record. Failed, negative, aborted, and
scientifically undetermined outcomes remain part of the record.

## Result retention

Keep the smallest package that preserves the scientific conclusion and normal
comparison use:

- immutable configuration, split/registry identities, manifests, summaries,
  tables, figures, alt text, and decision records;
- a checkpoint only when it is still needed for an active run, a recurring
  analysis, a consumer interface, or an irreplaceable reference;
- raw predictions or arrays only when the reported result cannot be audited or
  regenerated from retained material at acceptable cost.

Routine smoke checkpoints, superseded tuning checkpoints, duplicated token
exports, and rebuildable caches do not belong in the long-term result surface.
The retained evidence map is
[`experiments/RESULTS_INDEX.md`](experiments/RESULTS_INDEX.md).

Never clean a directory used by a live process. At the 2026-07-30
consolidation snapshot, EFRM LODO v2 was running and its entire `runs/` and
cache surface was excluded from cleanup.

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
Update [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) when an experiment
changes state. Dated reports preserve what was known at the time; later
corrections must be new records, not silent edits to an immutable run or
preregistration.

Generated reports, manuscript exports, and literature PDFs are communication
or reference assets, not active implementation authority.
