# Paper evidence index

This is the reader-facing route from the project tree to manuscript material. The
registry and generated status pages are useful navigation for progress, but they are
not substitutes for the dated reports and contracts listed below.

## Methods

Use these documents to describe what was designed, measured, and scientifically
specified:

- [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md): research question, claim levels,
  method rationale, and scientific boundaries.
- [`DATA_CONTRACT.md`](DATA_CONTRACT.md) and
  [`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md): units, modality semantics,
  masks, joins, normalization, and split rules.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) and the canonical runtime description in
  [`physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json`](physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json):
  implementation and data-flow structure.
- [`physiology_semantic_tokenizer/05_EXPERIMENT_DESIGN.md`](physiology_semantic_tokenizer/05_EXPERIMENT_DESIGN.md):
  frozen main-method design and gate definitions.
- [`comparisons/PROTOCOL.md`](comparisons/PROTOCOL.md),
  [`comparative_methods/README.md`](../comparative_methods/README.md), and
  [`comparative_methods/ASSET_STATUS.md`](../comparative_methods/ASSET_STATUS.md):
  comparison protocol, method identity, and source/weight preparation.
- [`../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md`](../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md):
  Croce Synthetic Phase 1 and Real Phase 2 design.

## Results

Use the dated report or retained table that directly contains each result. Use
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) only to locate the current lane and next
step.

- [`physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md):
  compact E0–E2/R0–R2 decision snapshot.
- [`physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md):
  full R-series methods, numerical results, and gate interpretation.
- [`analysis/TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md):
  retained development-only T0 Atlas result and its support limits.
- [`comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md):
  tracked 42-cell comparison results and cell-level acceptance states.
- [`comparisons/METRIC_ACCEPTANCE.md`](comparisons/METRIC_ACCEPTANCE.md):
  rules for deciding which comparison cells can enter tables.
- [`comparisons/PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md`](comparisons/PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md):
  exploratory degradation analysis; it does not provide a single causal result.
- [`../experiments/RESULTS_INDEX.md`](../experiments/RESULTS_INDEX.md):
  retained artifact locations and links to reports. Some payload paths there are
  local Git-ignored artifacts; the linked tracked reports remain the paper-facing
  record.

## Discussion

Build interpretation and limitations from:

- the evidence-supported and unsupported claim tables in
  [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md);
- the R-series interpretation and failure boundary in the
  [`R-series report`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md);
- the support/stability and non-coupling boundaries in the
  [`Token Atlas report`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md);
- the track labels, overlap conditions, rejected values, and unsupported cells in
  the [`comparison result report`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md)
  and [`metric acceptance rules`](comparisons/METRIC_ACCEPTANCE.md).

Keep engineering completion, descriptive findings, and failed or undetermined
scientific gates separate in the manuscript.

## Figures

Use the canonical architecture figure for the Methods overview:

- [`physiology_semantic_architecture.svg`](physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg)
  with its editable/runtime source in the
  [`architecture` directory](physiology_semantic_tokenizer/architecture/).
- [`shared_driver_semantic_return_plan.svg`](physiology_semantic_tokenizer/figures/plans/shared_driver_semantic_return_plan.svg)
  is a historical pre-gate return-plan figure, not a report of a completed run.
- [`experiment_plan.svg`](figures/experiment_plan.svg),
  [`experiment_plan.png`](figures/experiment_plan.png),
  [`alt text`](figures/experiment_plan.alt.txt), and its
  [`manifest`](figures/experiment_plan.manifest.json) are the 2026-08-14 plan
  snapshot.
- [`project_workflow_progress.svg`](project_workflow_progress.svg) is the optional
  four-lane snapshot dated 2026-08-16; it is for navigation and does not add
  scientific evidence.

## Not current paper evidence

- The local Git-ignored `docs/paper/aaai27_pid_mcm/` tree is an older manuscript
  snapshot containing old live-run/epoch-97 material. It is not the current Methods
  or Results source.
- Local Git-ignored `docs/paper_pdf/` and `docs/report/` trees are
  communication/reference assets; use the tracked reports above for current claims.
- [`../data/DATASETS_DESCRIPTION.md`](../data/DATASETS_DESCRIPTION.md) is only a
  compatibility pointer to the authoritative dataset description.
