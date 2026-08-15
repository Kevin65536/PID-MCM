# Retained result index

_Evidence surface updated 2026-08-14; payload-pruning record retained from
2026-07-30_

This index identifies the experiment material kept for routine reading,
comparison, and scientific audit. A retained result is defined by its
conclusion and provenance package, not by keeping every intermediate tensor or
checkpoint.

## Main-method evidence

| Stage | Retained authority / artifact | Why it remains |
| --- | --- | --- |
| E0 final teacher revalidation | [`20260723_adaptive_teacher_e0_v3_line_clean_v4_revalidation_v1`](runs/physiology_semantic_tokenizer/e0_teacher_validity/20260723_adaptive_teacher_e0_v3_line_clean_v4_revalidation_v1/summary.md) | admitted development teacher surface and claim boundary |
| E1 K128 health | [`20260722_e1_health_coupling_visual_report_v1`](runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/summary.json) and retained multi-seed summaries | software/occupancy reference |
| E2 weight calibration | [`20260723_e2_v4_training_gradient_weight_calibration_v1`](runs/physiology_semantic_tokenizer/e2_weight_calibration/20260723_e2_v4_training_gradient_weight_calibration_v1/analysis/summary.md) | explains frozen semantic-objective scale |
| E2 final decision | [`20260723_e2_v4_semantic_objective_suite_v1/decision`](runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/decision/summary.md) | no semantic row admitted; retain T0 |
| R0-P | [`20260728_r1p_raw_lag_formal_v1`](runs/physiology_semantic_tokenizer/r0p_raw_lag_baseline/20260728_r1p_raw_lag_formal_v1/summary.json) | registered raw-lag negative result |
| R1-D | [`20260728_e0_v3_reanalysis_v1`](runs/physiology_semantic_tokenizer/r1d_teacher_geometry/20260728_e0_v3_reanalysis_v1/summary.json) | exploratory correction geometry |
| R1-P structure | [`20260728_r1p_bundle_qualification_structure_v2`](runs/physiology_semantic_tokenizer/r1p_structural_audit/20260728_r1p_bundle_qualification_structure_v2/AUDIT.md) | confirms formal bundle integrity |
| R1-P formal panel | [`20260728_r1p_population_frozen_formal_v3`](runs/physiology_semantic_tokenizer/r1p_teacher_qualification/20260728_r1p_population_frozen_formal_v3/panel_summary.json) | G2 failure and final qualification decision |
| R1-P post-formal | [`20260728_r1p_formal_v3_postformal_v1`](runs/physiology_semantic_tokenizer/r1p_cross_session_hemodynamic_adaptation/20260728_r1p_formal_v3_postformal_v1/summary.json) | failure interpretation without gate revision |
| D1B | [`20260728_d1b_train_only_grid_v1`](runs/physiology_semantic_tokenizer/r1p_d1b_train_only_hyperparameter_seal/20260728_d1b_train_only_grid_v1/summary.json) | train-only evidence; validation remained undetermined |
| R2-D formal | [`20260728_r2d_cj_seed20260728_formal_v1`](runs/physiology_semantic_tokenizer/r2_continuous_observability/20260728_r2d_cj_seed20260728_formal_v1/summary.json) | bilateral observability failure |
| R2-D statistical audit | [`20260728_r1d_cj_seed20260728_v2_stat_audit`](runs/physiology_semantic_tokenizer/r2_continuous_observability_analysis/20260728_r1d_cj_seed20260728_v2_stat_audit/diagnostic_summary.json) | uncertainty and diagnostic reference |

The compact decision registry is
[`06_EXPERIMENT_LOG.md`](../docs/physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md);
the integrated report is
[`20260728_R_SERIES_EXPERIMENT_REPORT.md`](../docs/physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md).
All R1-P sealed source/config/test paths and hashes remain untouched.

## Token Atlas

Keep the full current E2 T0 Core artifact:

[`token_physiology_atlas_standard_loader_core_20260730`](runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/runs/t0_seed20260719/analysis/token_physiology_atlas_standard_loader_core_20260730/)

It contains the manifest, summaries, 12 figures with sidecars/alt text, source
tables, compact assignments, measurement cache, and sequence counts. It is
development-only and did not open protected data. Future Statistical-tier work
should reuse this frozen checkpoint/cache identity.

## Comparison methods

| Method | Retained surface | Policy |
| --- | --- | --- |
| STA-Net | complete [`20260727` formal run](../comparative_methods/STA-Net-PyTorch/runs/fivefold/20260727_sta_net_no_artifact_mask_converged_5fold_v1/) including 140 current formal checkpoints and [`aggregate`](../comparative_methods/STA-Net-PyTorch/runs/fivefold/20260727_sta_net_no_artifact_mask_converged_5fold_v1/aggregate/paper_table.md) | recurring seven-task reference |
| Joint protected campaign | tracked [`42-cell result report`](../docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md), sealed local 540-job status, dual-unblind record, aggregate, and traceability manifest | complete; 22 ready-with-note, 12 rejected, 2 overlap-only, 6 unsupported; run payload remains ignored |
| EFRM v2 | entire [`efrm_lodo_full_target_fivefold_v2`](../comparative_methods/EFRM-PyTorch/runs/formal/efrm_lodo_full_target_fivefold_v2/) run/protocol/status plus method runs and caches | complete; retained as frozen campaign evidence |
| BIOT / CBraMod / REVE | source-fidelity and alignment evidence plus frozen public/protected campaign identities | complete; protected payload remains ignored |
| BrainFusion / NormWear | reimplementation/adaptation evidence plus frozen public/protected campaign identities | complete; labels and track boundaries remain mandatory |
| EFRM v1 | aggregate/status and lightweight tables/figures | historical different-estimand context |
| UMAP | code, configs, README, design, and [`EXPERIMENT_SUMMARY.md`](../comparative_methods/UMAP/EXPERIMENT_SUMMARY.md) | diagnostic only; no old run directory retained |

The STA-Net checkpoints outside the latest formal run were superseded tuning,
smoke, personalized, or earlier formal payloads. Their configs, manifests,
metrics, aggregates, logs, and figures remain.

## Croce validation

Keep `croce_validation` design documents, scripts, reports, manifests,
figures, and the expensive current
`cache/croce_local/highwl_v2/` surface. Historical archive NPZ payloads are
rebuildable from the retained manifests/configuration and measured data. The
redesigned Synthetic Phase 1 and Real Phase 2 have not started.

## Historical source/observation generation

The pre-2026-07 physiology-semantic archive retains its README/inventory,
resolved configs, manifests, metric logs, summaries, CSV/JSON tables, figures,
and reports. Two X3 causal-exchange checkpoints are retained because this is
the most frequently referenced audited control:

- [`seed20260651/best_model.pt`](archive/pre_physiology_semantic_20260701/runs/tokenizer_cross_modal_exchange/20260626_173718_causal_cross_adapter_v1/tokenizer_interventions/k128_dim128_x3_causal_exchange_seed20260651/checkpoints/best_model.pt)
- [`seed20260652/best_model.pt`](archive/pre_physiology_semantic_20260701/runs/tokenizer_cross_modal_exchange/20260626_173718_causal_cross_adapter_v1/tokenizer_interventions/k128_dim128_x3_causal_exchange_seed20260652/checkpoints/best_model.pt)

All other archived `.pt` and `.npz` payloads were removed. Their conclusions
remain readable and comparable, but exact checkpoint/array replay is no longer
locally available without rebuilding from retained code/configuration and raw
data.

## 2026-07-30 pruning record

The cleanup was frozen after checking all local processes. The only active
project compute was EFRM LODO v2; none of the targets below intersected an
open file. The Atlas and EFRM paths were outside every deletion scope.

| Removed payload | Files | Bytes |
| --- | ---: | ---: |
| `experiments/archive/**/*.npz` | 7,264 | 510,838,674,895 |
| `experiments/archive/**/*.pt`, except the two X3 controls | 423 | 75,775,750,670 |
| `experiments/runs/.../software_smoke/**/*.pt` | 46 | 874,468,394 |
| STA-Net `.pt` outside the retained 20260727 formal run | 2,369 | 94,131,041,391 |
| `croce_validation/archive/**/*.npz` | 2,343 | 8,702,201,778 |
| **Total binary payload** | **12,445** | **690,322,137,128** |

That is approximately `690.3 GB` decimal (`642.9 GiB`). Generated
`__pycache__` and `.pytest_cache` directories were also removed separately and
are excluded from this byte total.

This deletion is not recoverable from the local working tree. Git-tracked
source/docs were not removed by the payload cleanup; ignored binary payloads
would require a backup or experiment rebuild.
