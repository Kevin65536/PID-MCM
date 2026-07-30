# Documentation map

_Single authority index; updated 2026-07-30_

## Active documents

| Question | Authority |
| --- | --- |
| What is the complete experiment schedule and current state? | [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) |
| What scientific conclusion is currently authorized? | [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md) |
| What data, masks, joins, geometry, and splits are valid? | [`DATA_CONTRACT.md`](DATA_CONTRACT.md) |
| What are the dataset-native facts and original sources? | [`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md) |
| What code/runtime exists today? | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| What happened in E0–E2 and R0–R2? | [`physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) |
| What is the full R-series evidence? | [`20260728_R_SERIES_EXPERIMENT_REPORT.md`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md) |
| What is the comparison contract? | [`comparisons/PROTOCOL.md`](comparisons/PROTOCOL.md) |
| Which comparisons are running or complete? | [`comparisons/STATUS.md`](comparisons/STATUS.md) |
| Which values can enter a final table? | [`comparisons/METRIC_ACCEPTANCE.md`](comparisons/METRIC_ACCEPTANCE.md) |
| What did the current Token Atlas show? | [`analysis/TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md) |
| Where are retained result artifacts? | [`../experiments/RESULTS_INDEX.md`](../experiments/RESULTS_INDEX.md) |

## Frozen and historical evidence

[`physiology_semantic_tokenizer/README.md`](physiology_semantic_tokenizer/README.md)
indexes the 2026-07-25 SD-SVQ proposal, its preregistration, machine seals, and
dated E0/E1/E2/R reports. Those documents preserve the generation that was
actually tested; they are not active instructions to proceed past a failed
gate.

Other evidence layers:

- [`architecture_changelog/`](architecture_changelog/INDEX.md): architecture
  and data-contract decisions;
- [`project_changelog/`](project_changelog/INDEX.md): repository and
  operational changes;
- `physiology_semantic_tokenizer/analysis/`: dated result reports;
- `reliable_survey/`, `references/`, and `notes/`: literature/background, not
  implementation authority;
- ignored manuscript/report/PDF trees: communication or reference assets.

## Lifecycle rules

- Current contracts and current status live only in the active documents
  above.
- Preregistrations, seals, manifests, and dated result reports remain
  immutable; issue a new correction record instead of silently rewriting
  them.
- A successful software test does not override a failed scientific gate.
- A completed earlier protected evaluation does not unlock a new protocol.
- Update `EXPERIMENT_PLAN.md` and its source-data snapshot whenever an
  experiment changes state.
- Archived code, configs, tests, and results require an explicit path and do
  not participate in default discovery.
