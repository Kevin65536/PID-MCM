# Documentation map

_Authority index. Current execution and scientific verdicts are generated from the
machine-readable research-state registry. The current theory/architecture freeze
is owned by [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md), projected into
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md);
the 2026-08-22 [tracked design note](physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json) is a pre-freeze candidate snapshot;
paper-facing evidence routes are listed in
[`PAPER_EVIDENCE_INDEX.md`](PAPER_EVIDENCE_INDEX.md)._

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
| What theory and architecture principles are frozen? | [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md#frozen-theory-and-architecture-contract-unimplemented) |
| What pre-freeze implementation candidates remain available? | [tracked design note](physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json) · [candidate snapshot figure](physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg) |
| What did the 2026-08-21 v1 QC actually measure? | [`analysis/SSM_OBSERVATION_AND_COUPLING_QC_RESULTS_20260821.md`](analysis/SSM_OBSERVATION_AND_COUPLING_QC_RESULTS_20260821.md) |
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

The 2026-08-22 theory note is now superseded as a method-boundary source. Its
candidate branches remain optional implementation ideas, not runtime or
scientific results; its optional-grammar wording does not override the current
frozen endpoint/proper-score/null evidence kernel. The dependency order in
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) deliberately keeps the v1 2026-08-21
QC, the current E2-compatible runtime, and future candidate selection separate.

Within the 2026-08-21 QC report, “v2” and “v3” label historical LC-SPVQ
mask-contract generations only; they do not name a current method generation.

Other evidence layers:

- [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md) is the active 2026-08-22 v1/v2
  claim-boundary and architecture-rationale owner; it explains what evidence can
  support a claim but does not replace the registry's current status;
- [`06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) is a
  dated E0–E2/R0–R2 decision snapshot. Any “current” wording inside it is local to
  that snapshot and does not replace the generated project status;
- [`architecture_changelog/`](architecture_changelog/INDEX.md): architecture
  and data-contract decisions;
- [`project_changelog/`](project_changelog/INDEX.md): repository and
  operational changes;
- `physiology_semantic_tokenizer/analysis/`: dated result reports;
- `reliable_survey/`, `references/`, and `notes/`: literature/background, not
  implementation authority;
- ignored manuscript/report/PDF trees: communication or reference assets. In
  particular, `docs/paper/科学通报/` is a local communication asset; its figures and
  manuscript drafts do not replace the tracked runtime, exploration note, or dated
  evidence reports.

Accordingly, PID or partial-information-decomposition language in notes,
references, archives, or local manuscript drafts is background or historical
material. The active claim boundary treats PID only as a replaceable later
pretraining exploration—not as core innovation, method identity, or a frozen
component.

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
- Completion in one lane does not change another lane's dependency order. A passing
  software check, a v1 QC result, or an exploration figure does not promote the
  corresponding method stage.
- Archived code, configs, tests, and results require an explicit path and do not
  participate in default discovery.
