# Physiology-semantic tokenizer configurations

_Active target namespace; full trainer available, physical-state supervision blocked by E0_

---

## 📋 Contract

New configuration files are added here only when their parser, shape assertions, dry-run behavior, and output namespace have tests. Every configuration must resolve its output below:

```text
experiments/runs/physiology_semantic_tokenizer/<suite>/<run>/
```

Earlier `source_observation`, `downstream`, `phase0`, and `phase0plus` families are isolated under [`../archive/pre_physiology_semantic_20260701/`](../archive/pre_physiology_semantic_20260701/README.md) and are not config fallbacks or templates for the redesign.

## Available configuration

| Config | Scope | Gate status |
| --- | --- | --- |
| [`p1_e0_contract_smoke.yaml`](p1_e0_contract_smoke.yaml) | One real anchor/event for mutually exclusive train/validation/test subjects | P1 smoke passed; G0 not evaluated |
| [`p2_p5_software_smoke.yaml`](p2_p5_software_smoke.yaml) | Loader-to-export correctness with fixed target dimensions | Passed; optimizer blocked until E0 |
| [`e0_teacher_validity_pilot.yaml`](e0_teacher_validity_pilot.yaml) | Subject-held-out posterior-predictive teacher validation | Validation blocked; protected test unopened |
| [`e0_teacher_validity_v2.yaml`](e0_teacher_validity_v2.yaml) | Four-dataset measurement audit, layered target/coupling validation, and replayable visual review | Validation blocked on physical observation and posterior calibration; protected test unopened |
| [`shared_state_reconstruction_bound.yaml`](shared_state_reconstruction_bound.yaml) | Croce-independent capacity curves for shared-only and modality-private reconstruction | Diagnostic only; protected test unopened |
| [`tokenizer_training_pilot.yaml`](tokenizer_training_pilot.yaml) | Full physical-state-supervised training protocol | Blocked by E0 decision artifact |
| [`tokenizer_optimizer_smoke.yaml`](tokenizer_optimizer_smoke.yaml) | Minimal teacher-supervised optimizer guard check | Correctly rejects blocked E0 |
| [`tokenizer_reconstruction_baseline_pilot.yaml`](tokenizer_reconstruction_baseline_pilot.yaml) | Teacher-free reconstruction-plus-VQ baseline | CUDA smoke and resume passed |
| [`e2_semantic_objective_suite.yaml`](e2_semantic_objective_suite.yaml) | Matched T0/T1/T2 development suite using the E1-selected quantizer | Software path passed; formal E2 blocked by channel-aware E0 rebuild |

Run the two mandatory early stages with:

```bash
.venv/bin/python experiments/scripts/validate_physiology_semantic_contract.py --mode dry-run
.venv/bin/python experiments/scripts/validate_physiology_semantic_contract.py --mode smoke
```

The smoke config validates the real solver-cache-loader chain, tensor shapes, posterior variance, causal masks, split isolation, and additive raw-space normalization. It does not measure the E0 posterior-predictive endpoint and cannot promote G0.

Run the migration software stages with:

```bash
.venv/bin/python experiments/train_physiology_semantic_tokenizer.py --config experiments/configs/physiology_semantic_tokenizer/p2_p5_software_smoke.yaml --dry-run
.venv/bin/python experiments/train_physiology_semantic_tokenizer.py --config experiments/configs/physiology_semantic_tokenizer/p2_p5_software_smoke.yaml --smoke
```

Teacher-supervised runs require a concrete passed E0 decision artifact whose split, contract, cache roots, protocol, registry, and calibration hashes match the run. A boolean flag cannot bypass this check. Teacher-free runs must set every teacher-derived loss weight to zero; they may optimize for quantizer and reconstruction characterization without claiming E0 success.

The E2 implementation and the currently blocking channel audit are documented
in [`20260722_E2_IMPLEMENTATION_AND_EXPERIMENT_PLAN.md`](../../../docs/physiology_semantic_tokenizer/analysis/20260722_E2_IMPLEMENTATION_AND_EXPERIMENT_PLAN.md).

Run the E0-v2 validation and regenerate its visual package with:

```bash
.venv/bin/python experiments/evaluate_physical_teacher_e0_v2.py \
  --config experiments/configs/physiology_semantic_tokenizer/e0_teacher_validity_v2.yaml
.venv/bin/python experiments/scripts/visualize_e0_v2_audit.py --run-dir <E0-v2-run-dir>
```

Visual review is registered only through `finalize_e0_v2_visual_review.py`, which verifies figure hashes and cannot open the protected test by itself.

Run the non-gate shared-state bound diagnostic with:

```bash
.venv/bin/python experiments/evaluate_shared_state_reconstruction_bound.py \
  --config experiments/configs/physiology_semantic_tokenizer/shared_state_reconstruction_bound.yaml
```

Its validation-oracle PCA is a lower bound only inside the declared rank-limited linear model class. Subject-held-out PCA/CCA results are achievable errors, not universal biological noise floors.

_Last updated: 2026-07-06_
