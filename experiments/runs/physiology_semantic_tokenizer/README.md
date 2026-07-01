# Physiology-semantic tokenizer result namespace

_Canonical output schema for E0–E9 experiments; no formal run exists yet_

---

## 🗂️ Suite structure

```text
physiology_semantic_tokenizer/
├── e0_teacher_validity/
├── e1_quantizer_correctness/
├── e2_semantic_supervision/
├── e3_masked_state/
├── e4_residual_strategy/
├── e5_optical_representation/
├── e6_information_ladder/
├── e7_frozen_coupling/
├── e8_wholebrain_downstream/
└── e9_visualization_stability/
```

Directories are created only when a dry run writes a valid manifest.

## 📦 Run structure

Each run uses `<timestamp>_<descriptive_name>/` and follows the artifact contract in [`04_IMPLEMENTATION_VALIDATION_PLAN.md`](../../../docs/physiology_semantic_tokenizer/04_IMPLEMENTATION_VALIDATION_PLAN.md):

```text
<suite>/<timestamp>_<name>/
├── config.yaml
├── resolved_config.yaml
├── manifest.json
├── environment.json
├── checkpoints/
├── metrics/
├── diagnostics/
├── predictions/
├── figures/
├── figure_data/
└── summary.md
```

## 🚦 Status values

Suite and run manifests use only:

- `planned`
- `dry_run_passed`
- `smoke_passed`
- `formal_running`
- `formal_complete`
- `gate_passed`
- `gate_failed`

The absence of a completion marker is never interpreted as success.

_Last updated: 2026-07-01_
