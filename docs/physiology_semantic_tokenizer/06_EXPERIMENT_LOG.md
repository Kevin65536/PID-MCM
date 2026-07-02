# Physiology-semantic tokenizer experiment log

_Active run registry for experiments executed under the 2026-07-01 target contract_

---

## 📋 Current status

No tokenizer training run has yet been promoted under the new architecture. P1-P5 software interfaces now exist, and the real-cache pipeline passed a dry-run, an E0-gated smoke, checkpoint serialization, and a P5 validation export. The smoke performed zero optimizer steps because `e0_passed` is false. G0 remains unevaluated because posterior-predictive validity, observability, calibration, and preregistered coverage have not yet been tested.

| Date | ID | Suite | Status | Result root |
| --- | --- | --- | --- | --- |
| 2026-07-01 | `PST-DESIGN-FREEZE` | Documentation | Complete | Not applicable |
| 2026-07-02 | `PST-P1-DRYRUN` | E0 contract dry-run | Passed; G0 not evaluated | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260702_191234_p1_contract_dry_run/` |
| 2026-07-02 | `PST-P1-SMOKE` | E0 contract smoke | Passed; G0 not evaluated | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260702_191234_p1_contract_smoke/` |
| 2026-07-02 | `PST-P2-P5-DRYRUN` | Migration integration | Passed; correctness only | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260702_235450_p2_p5_software_smoke/` |
| 2026-07-02 | `PST-P2-P5-SMOKE` | Migration software smoke and P5 export | Passed; optimizer blocked by E0; no gate evaluated | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260702_235459_p2_p5_software_smoke/` |
| Pending | `PST-E0` | Teacher validity | Planned | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/` |
| Pending | `PST-E1` | Quantizer correctness | Planned | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/` |

## 🚦 Scientific-result admission rule

A correctness-only dry-run or smoke may be logged with an explicit non-gate status. A scientific result or gate decision is promoted only when it has:

1. a run or suite manifest under the active result root;
2. an immutable resolved configuration and split hash;
3. a declared primary endpoint from [`05_EXPERIMENT_DESIGN.md`](05_EXPERIMENT_DESIGN.md);
4. a versioned `decision_protocol.yaml`, `metric_registry.json`, and `evidence_calibration.json`;
5. a completion status that distinguishes smoke, short-formal, and full-formal evidence;
6. a link to the run-level summary rather than only a pooled suite report.

## 🗂️ Historical results

All source/observation, coupling-strengthening, exchange, alignment, and old downstream results were moved to:

```text
experiments/archive/pre_physiology_semantic_20260701/runs/
```

Their narrative log is preserved at [`source_observation/EXPERIMENT_LOG.md`](../archive/pre_physiology_semantic_20260701/source_observation/EXPERIMENT_LOG.md). Historical results are baseline evidence and never appear in this table.

## 🔗 Related documents

- [Experiment design](05_EXPERIMENT_DESIGN.md)
- [Implementation and validation plan](04_IMPLEMENTATION_VALIDATION_PLAN.md)
- [Code migration plan](07_CODE_MIGRATION_PLAN.md)
- [Storage layout](../STORAGE_LAYOUT.md)
- [Archived-run inventory](../../experiments/archive/pre_physiology_semantic_20260701/README.md)

_Last updated: 2026-07-02_
