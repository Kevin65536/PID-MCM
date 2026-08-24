# Experiment sequencing

This document defines the active dependency order and a compact reader navigation.
The forward theory and architecture principles are frozen in
[`METHOD_RATIONALE.md`](METHOD_RATIONALE.md); the 2026-08-22 observation–source
note remains a pre-freeze implementation-candidate snapshot. Detailed current
execution, scientific verdicts, job counts, and registry-owned next actions are generated from
[`research_state/registry.json`](../research_state/registry.json) in
[`PROJECT_STATUS.md`](PROJECT_STATUS.md). A candidate or gate is not a runtime
result and does not open a protected cohort.

## Dependency lanes

| Lane | Stable dependency order | Owning contract |
| --- | --- | --- |
| Main method | v1 2026-08-21 QC → frozen theory/architecture boundary → versioned implementation and synthetic checks → optional public development → separately preregister any independent holdout | [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md), [`DATA_CONTRACT.md`](DATA_CONTRACT.md), [`20260821 v1 QC`](analysis/SSM_OBSERVATION_AND_COUPLING_QC_RESULTS_20260821.md) |
| Token Atlas | retained representation → Core → Statistical → a separately frozen confirmatory question | [`analysis/TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md) |
| Comparisons | source fidelity → adapter/input alignment → software/public evidence → frozen formal execution → cell-level interpretation | [`comparisons/PROTOCOL.md`](comparisons/PROTOCOL.md) |
| Croce validation | reference implementation → Synthetic Phase 1 → Real Phase 2 | [`../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md`](../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md) |

Completion of one lane does not imply a scientific conclusion in another lane.
In particular, engineering completion, descriptive analysis, and a failed scientific
gate remain distinct in the unified two-axis state model.

## Main-method implementation sequence

The theory/architecture freeze fixes only the nine boundaries in
[`METHOD_RATIONALE.md`](METHOD_RATIONALE.md). It does not preselect coordinates,
teacher family, network, VQ, capacities, hierarchy, or grammar network. The
shortest valid progression is:

Partial information decomposition (PID), if tested at all, belongs only to
optional pretraining development. It does not define the main method or its
novelty and is not frozen at candidate-definition time.

| Stage | Required work and boundary | Status / promotion rule |
| --- | --- | --- |
| **v1 — 2026-08-21 QC** | Preserve the historical LC-SPVQ checkpoint controls and the three-seed continuous SSM screen as fit-selection evidence. Record the endpoint-aligned mask correction, leakage controls, and fail-closed K16/q0-q1 decision. | **Complete as exploratory QC.** It is not an independent confirmation, used no VQ, and does not support SSM superiority or a coupling claim. |
| **Theory/architecture freeze** | Freeze data boundaries, modality input ownership, continuous-before-token order, teacher epistemic boundary, conditional VQ semantics, observation/source functions, and the endpoint-aligned evidence kernel; keep fine-to-coarse open and cross masking undefined. | **Complete as documentation contract.** It is not implementation or data authorization. |
| **Versioned implementation and synthetic checks** | Select coordinates, teacher family, network, optional VQ/hierarchy, and grammar network within the frozen boundary. Declare observation/source falsification endpoints. Cross masking is excluded unless separately defined as an information intervention. | **Planned.** Software, tensor, mask, provenance, baseline, proper-score, and null contracts must pass on synthetic data. No protected data is read. |
| **Optional public development** | If synthetic evidence justifies it, run a versioned fit/selection comparison without changing the frozen method kernel. | **Not yet specified.** Fit/development data only; implementation screening is not a final performance estimate. |
| **Independent holdout** | Only after selection, version the chosen implementation and preregister the estimand, split, estimator, nulls, thresholds, and stopping rule for one independent evaluation. | **Closed and unauthorized.** No development result is relabelled as holdout evidence. |

Dynamic teachers are candidate implementation families, not a claim that the existing
runtime already performs low-rank continuous-time state estimation. The Croce/Balloon
model is used first as a synthetic/solver-recovery qualification target; its parameters
and synthetic results must not be presented as measured EEG–fNIRS evidence. The earlier
E0 sign-calibrated adaptive local fixed-interval admission record
([`E0_V3_ADAPTIVE_TEACHER_ADMISSION_DECISION.md`](physiology_semantic_tokenizer/analysis/E0_V3_ADAPTIVE_TEACHER_ADMISSION_DECISION.md))
and the legacy Croce PF track
([`CROCE2017_REAL_DATA_VALIDATION_PLAN.md`](../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md))
are qualification references, not evidence that a future teacher is already R1-P
qualified.

## Current runtime versus exploratory options

The current production surface and the v2 exploration are intentionally separated.
The first column below describes what exists or was screened in the 2026-08-21 report;
the second lists options that may be implemented and compared.

| Object | Current runtime / v1 screen | Exploratory options |
| --- | --- | --- |
| Time axis and fNIRS target | E2-compatible 20 s windows are cut into ten 2 s patches; the v1 observation SSM sees patch positions with patch-internal samples flattened as features. | Fit a modality-specific teacher on the continuous 10 Hz trajectory (for example `[B,200,2]` for fNIRS), then smooth and only afterward derive token targets. |
| EEG target and stem | v1 uses per-patch absolute log band-power features with the generic patch stem; the motor-imagery EEG endpoint failed. | Use baseline-relative band-envelope/ERD trajectories and an amplitude-preserving, frequency-aware stem; the exact filterbank, normalization, and context remain open gates. |
| Dynamic teacher | `modality_observation_ssm.py` is a full observation-space ridge AR(1) smoother with identity observation map and diagonal Q/R; NATIVE, SSM-SELF, SSM-JOINT, and low-weight XPRED were screened, without a VQ. | Compare NATIVE/direct targets with selected low-rank or Croce-inspired teachers; joint teachers remain privileged ablations. |
| Observation branch | The v1 residual is O − O~ in observation-feature space; under NATIVE it is zero, so it is not a fair information-preservation baseline. | Compare raw/masked or pretrained-feature reconstruction targets; teacher innovation is an optional residual target, not the branch name. |
| Gate baseline and coupling | The v1 ΔR² screen used a condition×time mean as a strong secondary residual oracle; historical LC-SPVQ controls are v2 same-token-time diagnostics, not v3 evidence. | Preserve the frozen endpoint-aligned increment/baseline/proper-score/null kernel; exact task-specific choices are preregistered before held-out access. |
| Discretization and hierarchy | The E2-compatible runtime has independent source VQ K=128 and continuous residual branches; the 2026-08-21 v1 screen stopped before K16/q0-q1. | If selected, search fine (K_f,D) and modality-specific coarse (K_c) source tokens with an explicit fine→coarse mapping and development-only map-quality objective. |

This table is a navigation aid, not a runtime specification. The tracked
design note and figure preserve pre-freeze implementation ideas; the local Science
Bulletin drawing is only a historical communication projection (see
[`PAPER_EVIDENCE_INDEX.md`](PAPER_EVIDENCE_INDEX.md)).

## Four-lane reader navigation

| Lane | Design | Current result | Next | Evidence to read |
| --- | --- | --- | --- | --- |
| Main method | [`v1 QC`](analysis/SSM_OBSERVATION_AND_COUPLING_QC_RESULTS_20260821.md) → frozen theory/architecture boundary → versioned implementation → synthetic checks → optional public development → separately preregister any holdout | v1 remains negative/inconclusive evidence; the nine theory/architecture boundaries are now frozen, while implementation and measured evaluation have not started | Define the smallest implementation inside the frozen boundary; keep development and protected data boundaries separate | [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md), [`DATA_CONTRACT.md`](DATA_CONTRACT.md), [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Token Atlas | [`TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md) | T0 Core complete; Statistical tier not started | Run the Statistical tier on frozen T0; freeze a separate coupling-null question before any Full tier | [`TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md), [`PROJECT_STATUS.md#token-atlas`](PROJECT_STATUS.md#token-atlas) |
| Comparisons | [`comparisons/PROTOCOL.md`](comparisons/PROTOCOL.md) | 540/540 jobs; 42 cells: 22 ready-with-note, 12 rejected, 2 overlap-only, 6 unsupported | Use admitted cells with their track notes for paper tables; this row does not imply a new campaign | [`PROTECTED_CAMPAIGN_RESULTS_20260814.md`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md), [`METRIC_ACCEPTANCE.md`](comparisons/METRIC_ACCEPTANCE.md) |
| Croce validation | [`../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md`](../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md) | Legacy diagnostics complete; Synthetic Phase 1 not started | Run Synthetic Phase 1, then Real Phase 2 only if its decision rule is met | [`CROCE2017_REAL_DATA_VALIDATION_PLAN.md`](../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md), [`PROJECT_STATUS.md#croce-验证`](PROJECT_STATUS.md#croce-验证) |

## Updating current state

Edit the current item for the affected track in
[`research_state/registry.json`](../research_state/registry.json), then run:

```bash
.venv/bin/python experiments/scripts/project_state.py validate
.venv/bin/python experiments/scripts/project_state.py render
```

The renderer refreshes the root README status block and `docs/PROJECT_STATUS.md`.
An optional audit/freeze check is useful for paper-ready results but is not required
for an ordinary progress update. Protocol and data-access rules stay in their owning
contracts; they are not a third project-status axis.

## Quick workflow overview (registry snapshot)

The human-readable four-track overview is a presentation snapshot rendered from
the `2026-08-16T20:54:18+08:00` registry state:
[`project_workflow_progress.svg`](project_workflow_progress.svg) and its editable
[`Draw.io source`](project_workflow_progress.drawio). It is useful for orientation
and paper discussion, but it is not a second status authority; read
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the current text projection and
[`research_state/registry.json`](../research_state/registry.json) for the source
records. Animated dependency paths in the Draw.io source indicate workflow
direction only, never scientific support.

![Quick workflow overview (registry snapshot 2026-08-16)](project_workflow_progress.svg)

## Historical visualization (2026-08-14)

The `experiment_plan*` files below are a **Historical snapshot (2026-08-14)**,
retained for navigation and historical reproducibility only. They are not current project
state and are no longer an input to the unified registry. Historical
campaign-specific access fields in that frozen snapshot are not read or projected
by the current registry.

- [Historical SVG](figures/experiment_plan.svg) · [PNG export](figures/experiment_plan.png)
- [Alt text](figures/experiment_plan.alt.txt) · [source JSON](figures/experiment_plan_status.json)
- [render manifest](figures/experiment_plan.manifest.json)
