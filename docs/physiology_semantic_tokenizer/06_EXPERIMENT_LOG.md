# Physiology-semantic tokenizer experiment log

_Active run registry for experiments executed under the 2026-07-01 target contract_

---

## 📋 Current status

The complete tokenizer training loop is runnable, but E0-v2 remains blocked at validation. The cross-dataset measurement adapter, local target observability, K=128 target transmissibility, and continuous coupling upper bound passed their declared validation checks. The fNIRS physical-observation mean and posterior uncertainty calibration failed both numerical and visual review. The protected test was not opened, and physical-state-supervised training remains blocked. A teacher-free reconstruction-plus-VQ smoke completed two CUDA optimizer steps and resumed to four; it is eligible for an E1 short-formal software/quantizer pilot but does not validate semantic supervision.

| Date | ID | Suite | Status | Result root |
| --- | --- | --- | --- | --- |
| 2026-07-01 | `PST-DESIGN-FREEZE` | Documentation | Complete | Not applicable |
| 2026-07-02 | `PST-P1-DRYRUN` | E0 contract dry-run | Passed; G0 not evaluated | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260702_191234_p1_contract_dry_run/` |
| 2026-07-02 | `PST-P1-SMOKE` | E0 contract smoke | Passed; G0 not evaluated | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260702_191234_p1_contract_smoke/` |
| 2026-07-02 | `PST-P2-P5-DRYRUN` | Migration integration | Passed; correctness only | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260702_235450_p2_p5_software_smoke/` |
| 2026-07-02 | `PST-P2-P5-SMOKE` | Migration software smoke and P5 export | Passed; optimizer blocked by E0; no gate evaluated | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260702_235459_p2_p5_software_smoke/` |
| 2026-07-03 | `PST-E0-PILOT-V1` | Teacher validity | Blocked on validation; protected test unopened | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260703_165153_e0_teacher_validity_pilot_v1/` |
| 2026-07-03 | `PST-E0-V2-VALIDATION` | Layered teacher information contract and visual audit | Blocked on physical observation and uncertainty calibration; protected test unopened | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260703_232754_e0_teacher_validity_v2/` |
| 2026-07-03 | `PST-TRAIN-DRYRUN-V1` | Full trainer dry-run | Passed; no optimizer step | `experiments/runs/physiology_semantic_tokenizer/tokenizer_training/20260703_164728_physiology_semantic_tokenizer_pilot_v1/` |
| 2026-07-03 | `PST-E1-TF-SMOKE-V1` | Teacher-free reconstruction/VQ | Passed; CUDA, 2 optimizer steps | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260703_165220_tokenizer_reconstruction_baseline_pilot_v1/` |
| 2026-07-03 | `PST-E1-TF-RESUME-V1` | Teacher-free checkpoint resume | Passed; resumed to 4 optimizer steps | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260703_165236_tokenizer_reconstruction_baseline_pilot_v1/` |

### E0 validation decision

On the five validation subjects, EEG normalized predictive gain was `+0.756030` with a subject-bootstrap 95% interval of `[0.691380, 0.805454]`. fNIRS normalized predictive gain was `-1.503607` with interval `[-2.383524, -0.662615]`. Although all nine fNIRS state-summary coordinates exceeded their feature-permutation observability null, the teacher's posterior-predictive clean observation failed the declared history-baseline endpoint. This distinction prevents coordinate observability from being mistaken for physical-teacher validity.

### E0-v2 layered validation and visual review

E0-v2 evaluated 17,280 train and 4,800 validation local patches plus 8,640 train and 2,400 validation fixed-history rows. Four raw fNIRS datasets were audited without reading their protected partitions. Train-only full-record baseline/robust-scale adapters achieved exact crop-position invariance and a validation pooled-pair scale ratio of `1.463`; original voltage, concentration, and exported relative-signal semantics remain explicit.

Patch-local observability admitted EEG `r_mean`, `r_slope`, and `s_slope`, while excluding `s_mean`. All six fNIRS level/slope coordinates exceeded their label-permutation references, with validation R² from `0.046` to `0.258`. The admitted target geometry remained transmissible through 128 prototypes: global standardized reconstruction R² was `0.918` for EEG and `0.949` for fNIRS, with perplexities `96.8` and `107.6`. The continuous upper bound was positive beyond shuffled EEG history for both fNIRS levels (`0.177` nats) and innovations (`0.172` nats); gains concentrated in `delta_f` and `delta_hbo`, while `delta_hb` added approximately zero.

Two independent layers failed. Corrected wavelength-space fNIRS clean prediction had MSE `2.193` versus the selected history baseline `0.834`, a subject-mean gain of `-1.359` with `0/5` positive subjects. Synthetic-truth variance scaling improved posterior coverage but `delta_f`, `delta_hbo`, and `delta_hb` remained outside the sample-size-derived 95% band; real-data student errors were also much larger than teacher posterior SD. The visual review confirmed the fNIRS baseline/amplitude mismatch and uncertainty under-coverage, so its overall status is `fail`. Protected-test eligibility is `false`.

Every numerical layer has a corresponding replayable visual artifact with source JSON, SVG, 300 dpi PNG, hashes, and review notes in the run root. The definitive decision files are `summary.json`, `visual_audit_manifest.json`, `visual_review.json`, and `target_contract.json`.

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

_Last updated: 2026-07-03_
