# Physiology-semantic tokenizer configurations

_Active target namespace; P1-P5 software validation complete, scientific gates pending_

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

With `validation.e0_passed: false`, the entrypoint always writes `optimizer_steps: 0`; changing the requested step count cannot bypass the gate.

_Last updated: 2026-07-02_
