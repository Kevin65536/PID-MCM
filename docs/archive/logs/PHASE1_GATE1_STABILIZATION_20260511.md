# Phase 1 Gate1 Stabilization Archive

> Date: 2026-05-11
> Scope: source/observation Phase 1 Gate1 tuning closure and Phase 2 handoff

## Summary

Phase 1 Gate1 work is now formally closed. The tokenizer mainline has removed explicit phase supervision and phase-only reconstruction visualizations, aligned frequency visualization with the actual amplitude loss, and established a stable Gate1-passing no-phase baseline.

This archive was first recorded while preserving original run paths. During the 2026-06-04 storage cleanup, the remaining top-level Phase 1 reference run was moved into the same formal archive bundle to keep `experiments/runs/` reserved for active runs.

## Best Handoff Artifact

| Artifact | Path | Notes |
|----------|------|-------|
| Best config alias | [experiments/configs/source_observation/phase1/gate1_best_current.yaml](../../../experiments/configs/source_observation/phase1/gate1_best_current.yaml) | Alias to the 320-epoch slow-warmup no-phase baseline |
| Locked baseline | [experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml](../../../experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml) | Clean reusable Phase 1 Gate1 baseline |
| Best run | [Archived best run](../../../experiments/archive/source_observation_phase1_gate1_stabilization_20260511/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718) | Gate1 pass, best epoch 278, best val_loss 1.6395270029703777 |
| Reference run | [Archived reference run](../../../experiments/archive/source_observation_phase1_gate1_stabilization_20260511/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_long_bs128_20260511_174538) | First long no-phase run that restored Gate1 pass |
| Run archive manifest | [Phase 1 archive manifest](../../../experiments/archive/source_observation_phase1_gate1_stabilization_20260511/manifest.json) | Formal bundle of all Phase 1 Gate1 attempts |
| Result index | [experiments/archive/results/pre_croce_local_highwl_20260604/source_observation_index.json](../../../experiments/archive/results/pre_croce_local_highwl_20260604/source_observation_index.json) | Queryable project-level summary of historical source/observation results |

## Gate Outcome

- Gate1: stable pass on the no-phase baseline.
- Gate2: fail. Cross-modal token predictability remains below chance and HRF source target metrics are still unavailable.
- Gate3: fail. Coupling rows remain fully diffuse.
- Gate4: fail. Source subject leakage remains high and semantic selectivity is weak.
- Promotion verdict: hold_repair.

## Handoff Rule

Phase 2 work should start from [experiments/configs/source_observation/phase1/gate1_best_current.yaml](../../../experiments/configs/source_observation/phase1/gate1_best_current.yaml), keeping Gate1 stable while introducing source semantics through the HRF target path.
