# Experiment sequencing

This page is the clean-slate entry for experiment planning. No experiment
sequence is active, and this document does not authorize a run, open a protected
cohort, or define a new experiment ID. Historical plans and reports retain their
original wording for reproducibility; words such as “active”, “current”, or
“next” inside those dated snapshots describe the state at that time, not a
current instruction.

## Lifecycle boundary

The following single table is the lifecycle overlay for the superseded flow. It
does not rewrite hashed or dated evidence; the linked reports remain the evidence
owners. `stopped` means the work reached a terminal historical outcome or was
closed after its recorded result. `abandoned` means an uncompleted candidate or
follow-on is retained only as a non-runnable snapshot.

| Historical item | Lifecycle | Evidence or retained plan | Retained use |
| --- | --- | --- | --- |
| E0–E2 and R0–R2 generations | **stopped** | [`06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) and [`20260728_R_SERIES_EXPERIMENT_REPORT.md`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md) | Historical results and failure boundaries only |
| SSM reliability screen | **stopped** | [`20260819 SSM reconstruction reliability results`](analysis/20260819_SSM_RECONSTRUCTION_RELIABILITY_RESULTS.md) | Exploratory reliability evidence only |
| Continuous-latent screen | **stopped** | [`20260819 continuous shared/private latent results`](analysis/20260819_CONTINUOUS_SHARED_PRIVATE_LATENT_RESULTS.md) | Exploratory latent evidence only |
| LC-SPVQ optimization and QC | **stopped** | Dated LC-SPVQ reports under [`analysis/`](analysis/) | Negative/undetermined evidence only |
| Token Atlas Core (T0) | **stopped** | [`TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md) | Development-only retained result |
| Protected comparison campaign and P0 degradation | **stopped** | [`PROTECTED_CAMPAIGN_RESULTS_20260814.md`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md) and [`PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md`](comparisons/PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md) | Retained comparison evidence only |
| Croce legacy solver and audits | **stopped** | [`CROCE2017_REAL_DATA_VALIDATION_PLAN.md`](../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md) | Historical qualification/audit evidence only |
| Comparison P1/P2 and unexecuted follow-up | **abandoned** | [`PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md`](comparisons/PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md) | Unstarted comparison candidates only |
| D1B follow-on | **abandoned** | Dated D1B entries in [`06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) | Historical non-result; not a restart point |
| Future R/VQ branches | **abandoned** | R/VQ candidate notes and retained configuration references | Non-runnable historical candidates |
| LC-SPVQ full development | **abandoned** | LC-SPVQ development plans and smoke records under [`analysis/`](analysis/) | Non-runnable historical candidate |
| Observation/source candidate map | **abandoned** | [`observation_source_exploration_v2.json`](physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json) | Replaceable design snapshot only |
| Token Atlas Statistical and Full tiers | **abandoned** | Atlas plan and retained T0 report | No unstarted tier is scheduled |
| Croce Synthetic Phase 1 and Real Phase 2 | **abandoned** | Croce validation plan | Unstarted follow-on; no run is scheduled |

The table intentionally separates stopped evidence from abandoned candidates.
Neither state is an active sequence, a promotion gate, or permission to access
measured/protected data. Other protocol-owned evidence remains at its recorded
lifecycle in its owning document; it is not silently re-sequenced here.

## Clean-slate entry

There is currently no new flow to list: no stages, dependency order, selected
implementation, estimator, split, threshold, or holdout has been registered.
When a future flow is approved, its owner must first add one versioned contract
and registry record; this page can then link to that record. Until that happens,
the blank entry below is deliberate.

| Entry | State | Authority |
| --- | --- | --- |
| New experiment flow | **not registered** | No contract or owner yet |
| Data, mask, split, and protected boundary | **retained contract** | [`DATA_CONTRACT.md`](DATA_CONTRACT.md) |
| Historical results | **stopped evidence** | Dated reports and [`PAPER_EVIDENCE_INDEX.md`](PAPER_EVIDENCE_INDEX.md) |
| Unstarted candidate plans | **abandoned snapshots** | Explicit paths listed in the lifecycle table above |

## Updating lifecycle state

The machine-readable registry is the sole owner of any future registered
execution state. Do not copy a new sequence or next action into this page before
that registry record and its versioned contract exist. Existing reports, source
hashes, dates, and evidence artifacts are immutable historical records.

The historical visualization below is retained for navigation only; it is not a
current plan or execution instruction.

## Historical visualization (2026-08-14)

The `experiment_plan*` files below are a **Historical snapshot (2026-08-14)**,
retained for navigation and historical reproducibility only. They are not current project
state and are no longer an input to the unified registry. Historical
campaign-specific access fields in that frozen snapshot are not read or projected
by the current registry.

- [Historical SVG](figures/experiment_plan.svg) · [PNG export](figures/experiment_plan.png)
- [Alt text](figures/experiment_plan.alt.txt) · [source JSON](figures/experiment_plan_status.json)
- [render manifest](figures/experiment_plan.manifest.json)
