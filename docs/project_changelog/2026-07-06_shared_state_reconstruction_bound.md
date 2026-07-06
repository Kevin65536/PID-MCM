# Shared-state reconstruction-bound diagnostic

- **Date:** 2026-07-06
- **Scope:** non-gate experiment tooling, result provenance, and architecture-plan evidence
- **Scientific gate impact:** none; E0 remains blocked and protected subjects remain closed

## Change

Added a Croce-independent diagnostic that estimates capacity-conditional reconstruction error directly from paired EEG and fNIRS observations. The workflow compares validation-oracle and train-fitted joint PCA, CCA-constrained shared states, single-modality CCA inference, and separate modality PCA over latent dimensions 1–64.

The formal diagnostic uses subjects 1–18 for fitting and 19–23 for validation. It hashes only those cache inputs and does not read protected subjects 24–29. The run emits the standard config, protocol, metric registry, calibration, manifest, environment, summary, CSV, SVG, and PNG artifacts under the active E0 namespace, while declaring every result diagnostic and non-gating.

## Result boundary

The rank-limited validation-oracle PCA is documented as a lower bound only within its linear model class. Held-out PCA/CCA errors are achievable generalization estimates, not universal biological noise floors. The result supports a future shared-semantic plus modality-private observation plan, recorded in a plan-specific SVG overlay, but does not modify the canonical architecture or admit physical-teacher training.

## Artifacts

- `experiments/evaluate_shared_state_reconstruction_bound.py`
- `experiments/configs/physiology_semantic_tokenizer/shared_state_reconstruction_bound.yaml`
- `docs/physiology_semantic_tokenizer/09_SHARED_STATE_RECONSTRUCTION_BOUND.md`
- `docs/physiology_semantic_tokenizer/architecture/shared_state_reconstruction_bound_plan.json`
- `docs/physiology_semantic_tokenizer/figures/plans/shared_state_reconstruction_bound_plan.svg`
- `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_105937_shared_state_reconstruction_bound_v1/`

## Validation

- capacity-oracle monotonicity and feature-shape tests;
- current architecture SVG drift check;
- overlay reference and XML tests;
- targeted E0-v2, measurement-adapter, and new diagnostic tests.

