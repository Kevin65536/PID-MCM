# Storage Layout

> Status: Active as of 2026-06-04.

This document records the canonical storage layout for generated Croce cache
artifacts and experiment outputs. Generated data and run outputs remain ignored
by git; tracked configs and docs point to the stable entry paths below.

## Active Croce Cache Namespace

Current tokenizer training uses the highWL-only Croce local cache contract:

```text
croce_validation/cache/croce_local/highwl_v1/
  single_trial_motor_imagery -> ../../EEG_fNIRS_single_trail_pf_full
  single_trial_mental_arithmetic -> ../../EEG_fNIRS_single_trail_pf_full_mental_arithmetic_regen_20260603_182938
  simultaneous_cognitive -> ../../simultaneous_optical_pf_cache
```

The symlinks are the canonical config-facing paths. The underlying cache roots
stay in place while the current highWL training job is running.

## Experiment Run Namespace

Future source/observation Croce local runs should be written under:

```text
experiments/runs/source_observation/croce_local/highwl_v1/<run_name>/
```

Configs opt into this with:

```yaml
experiment:
  run_group: source_observation/croce_local/highwl_v1
```

The live run launched before this namespace normalization remains at:

```text
experiments/runs/s2_croce_local_highwl_base_20260604_153549/
```

Do not move that directory while the training process is active.

## Archive Layout

Historical artifacts that are no longer compatible with the current highWL
local tokenizer paradigm were archived on 2026-06-04 instead of deleted.

```text
croce_validation/archive/cache/legacy_pre_highwl_v1_20260604/
croce_validation/archive/results/pre_croce_local_highwl_20260604/
croce_validation/archive/analysis/pre_croce_local_highwl_20260604/

experiments/runs/archive/pre_croce_local_highwl_20260604/
experiments/archive/results/pre_croce_local_highwl_20260604/
experiments/archive/comparison_reports/pre_croce_local_highwl_20260604/
```

The Phase 1 Gate1 reference long run was moved into its existing formal archive
bundle:

```text
experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/
```

## Discovery Rules

- Active training configs should use canonical cache symlink roots, not dated
  physical cache roots.
- New highWL local tokenizer results should use `experiment.run_group`.
- Comparison tooling discovers nested run directories by searching for
  `metrics.json` and skips `experiments/runs/archive/` by default.
- Archived artifacts can still be inspected directly by absolute path, but they
  should not be used as current-paradigm evidence.
