# Physiology-semantic tokenizer experiment log

_Active run registry for experiments executed under the 2026-07-01 target contract_

---

## 📋 Current status

No tokenizer training run has yet been promoted under the new architecture. The design and archive freeze is complete; implementation begins at P1/G0.

| Date | ID | Suite | Status | Result root |
| --- | --- | --- | --- | --- |
| 2026-07-01 | `PST-DESIGN-FREEZE` | Documentation | Complete | Not applicable |
| Pending | `PST-E0` | Teacher validity | Planned | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/` |
| Pending | `PST-E1` | Quantizer correctness | Planned | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/` |

## 🚦 Admission rule

A result is added here only when it has:

1. a run or suite manifest under the active result root;
2. an immutable resolved configuration and split hash;
3. a declared primary endpoint from [`05_EXPERIMENT_DESIGN.md`](05_EXPERIMENT_DESIGN.md);
4. a completion status that distinguishes smoke, short-formal, and full-formal evidence;
5. a link to the run-level summary rather than only a pooled suite report.

## 🗂️ Historical results

All source/observation, coupling-strengthening, exchange, alignment, and old downstream results were moved to:

```text
experiments/archive/pre_physiology_semantic_20260701/runs/
```

Their narrative log is preserved at [`source_observation/EXPERIMENT_LOG.md`](../archive/pre_physiology_semantic_20260701/source_observation/EXPERIMENT_LOG.md). Historical results are baseline evidence and never appear in this table.

## 🔗 Related documents

- [Experiment design](05_EXPERIMENT_DESIGN.md)
- [Implementation and validation plan](04_IMPLEMENTATION_VALIDATION_PLAN.md)
- [Storage layout](../STORAGE_LAYOUT.md)
- [Archived-run inventory](../../experiments/archive/pre_physiology_semantic_20260701/README.md)

_Last updated: 2026-07-01_
