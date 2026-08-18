# Documentation map

_Authority index. Current execution and scientific verdicts are generated from the
machine-readable research-state registry. Paper-facing evidence routes are listed
in [`PAPER_EVIDENCE_INDEX.md`](PAPER_EVIDENCE_INDEX.md)._

## Active documents

| Question | Authority |
| --- | --- |
| What is currently complete, running, blocked, or scientifically resolved? | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) (generated) |
| What is the stable dependency order? | [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) |
| What rationale and evidence support the scientific conclusion? | [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md) |
| Where should paper Methods, Results, Discussion, and Figures take material from? | [`PAPER_EVIDENCE_INDEX.md`](PAPER_EVIDENCE_INDEX.md) |
| What data, masks, joins, geometry, and splits are valid? | [`DATA_CONTRACT.md`](DATA_CONTRACT.md) |
| What are the dataset-native facts and original sources? | [`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md) |
| What code/runtime exists today? | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| What happened in E0–E2 and R0–R2? | [`physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) |
| What is the full R-series evidence? | [`20260728_R_SERIES_EXPERIMENT_REPORT.md`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md) |
| What is the comparison contract? | [`comparisons/PROTOCOL.md`](comparisons/PROTOCOL.md) |
| Which comparisons are running or complete? | [`PROJECT_STATUS.md#对比实验`](PROJECT_STATUS.md#对比实验) (generated) |
| What is the short comparison-method summary? | [`comparisons/STATUS.md`](comparisons/STATUS.md) |
| What are the completed protected-comparison results? | [`comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md) |
| Which method sources and weights are prepared? | [`../comparative_methods/ASSET_STATUS.md`](../comparative_methods/ASSET_STATUS.md) |
| Which values can enter a final table? | [`comparisons/METRIC_ACCEPTANCE.md`](comparisons/METRIC_ACCEPTANCE.md) |
| What did the retained Token Atlas result show? | [`analysis/TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md) |
| Where are retained result artifacts? | [`../experiments/RESULTS_INDEX.md`](../experiments/RESULTS_INDEX.md) |

## Frozen and historical evidence

[`physiology_semantic_tokenizer/README.md`](physiology_semantic_tokenizer/README.md)
indexes the 2026-07-25 SD-SVQ proposal, its preregistration, and dated E0/E1/E2/R
reports. Those documents preserve the generation that was actually tested; they
are not active instructions to proceed past a failed gate.

Other evidence layers:

- [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md) and
  [`06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) are
  dated decision snapshots. Any “current” wording inside them is local to that
  snapshot; it does not replace the generated project status;
- [`architecture_changelog/`](architecture_changelog/INDEX.md): architecture
  and data-contract decisions;
- [`project_changelog/`](project_changelog/INDEX.md): repository and
  operational changes;
- `physiology_semantic_tokenizer/analysis/`: dated result reports;
- `reliable_survey/`, `references/`, and `notes/`: literature/background, not
  implementation authority;
- ignored manuscript/report/PDF trees: communication or reference assets.

## Lightweight lifecycle

- Current execution and scientific verdicts are read from
  [`../research_state/registry.json`](../research_state/registry.json) and its
  generated [`PROJECT_STATUS.md`](PROJECT_STATUS.md). Hand-written documents
  explain methods, evidence, and interpretation.
- Keep one concise current entry per track and avoid copying job counts or next
  actions into several README/STATUS pages.
- For a routine update, edit the current registry item and run `validate`, then
  `render`. Evidence and audit checks are optional extras when preparing a
  paper-ready frozen result.
- Protocol and data-access constraints stay in their owning contracts. A passing
  software test does not override a failed scientific gate.
- Preregistrations, manifests, and result reports are dated snapshots; if a
  narrative needs correction, point to a newer dated note rather than making the
  history difficult to follow.
- Completion in one lane does not change another lane's dependency order.
- Archived code, configs, tests, and results require an explicit path and do not
  participate in default discovery.
