# E0 gate and physiology-semantic training runtime

- **Date:** 2026-07-03
- **Phase:** Phase 3 E0/P4
- **Status:** Historical pre-sign-calibration record; superseded by complete E0 pass on 2026-07-24
- **Commits:** `43bdef1`, `2b4f3b3`, `1dc69ce`, `4c66905`

---

## 📋 Decision

> **Current-status correction (2026-07-24):** This record documents the
> original pilot behavior only. Observation-aligned sign calibration supersedes
> its gate label: the adaptive SSM physical teacher passes complete E0, its
> physiological information including fNIRS is fully acceptable, and
> physical-teacher supervision is authorized.

The physiology-semantic mainline has a complete training runtime, and optimizer
authorization is objective-specific. Physical-state-supervised loss requires a
concrete passed E0 decision bound to the split, data contract, cache roots,
decision protocol, metric registry, and evidence calibration. The current
sign-calibrated decision satisfies that requirement. A teacher-free
reconstruction-plus-VQ objective remains independently runnable.

The first E0 pilot emitted a negative fNIRS posterior-predictive label in the
pre-sign-calibration coordinate system. That historical label does not block
the calibrated physical teacher and is not the current E0 result.

```mermaid
flowchart TD
    accTitle: E0-gated tokenizer training paths
    accDescr: Historical runtime logic evaluated teacher validity on training and validation subjects; the later sign-calibrated adaptive SSM decision passes complete E0.

    cache["Versioned paired-optical cache"] --> e0["E0 train and validation evaluation"]
    e0 --> decision{"Posterior-predictive validation passed?"}
    decision -->|Yes| mask["Admitted teacher coordinates"]
    mask --> supervised["Physical-state-supervised trainer"]
    decision -->|No| blocked["Block teacher-derived losses"]
    blocked --> baseline["Teacher-free reconstruction + VQ"]
    supervised --> runtime["Validation, best/last checkpoints, resume"]
    baseline --> runtime
    decision -. "Historical negative label" .-> protected["Protected test remains closed"]

    classDef data fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decisionNode fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef blockedNode fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef runtimeNode fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class cache,e0,mask data
    class decision decisionNode
    class blocked,protected blockedNode
    class supervised,baseline,runtime runtimeNode
```

## 🧱 Runtime changes

- The cache batch generator accepts deterministic anchor lists, sample caps, and partition-specific manifests.
- E0 records its protocol and calibration hashes, cache provenance, split hash, duplicate checks, observability diagnostics, posterior-predictive comparisons, nulls, confidence intervals, and source data for SVG/PNG figures.
- The trainer implements dry-run, smoke, and formal modes; epoch training and validation; AMP; gradient clipping; AdamW; warm-up/cosine scheduling; early stopping; best/last checkpoints; and exact-state resume.
- Loss routing applies E0 coordinate masks and excludes unadmitted or invalid teacher targets.
- The runtime still requires a concrete E0 decision artifact; the current sign-calibrated decision authorizes teacher-supervised optimization.

## 🔬 Pilot evidence

The subject split contains 18 training, 5 validation, and 6 protected-test
subjects. On validation, EEG normalized predictive gain was `+0.756030` with a
subject-bootstrap 95% interval of `[0.691380, 0.805454]`. The historical
pre-sign-calibration fNIRS gain was `-1.503607` with interval
`[-2.383524, -0.662615]`. This number is retained for provenance and is not the
current E0 status.

The teacher-free CUDA smoke completed two optimizer steps with best validation loss `2.067416`. Reloading `last.pt` resumed to four steps and improved the best validation loss to `2.063996`. These runs establish optimizer, validation, checkpoint, and resume correctness only.

## 🚧 Evidence boundary

- Current E0 status is `pass`.
- The sign-calibrated adaptive SSM physical teacher is accepted.
- Physical-state-supervised training is authorized.
- Teacher-free smoke does not establish semantic state validity, information retention, or downstream utility.
- The original decision artifact remains immutable historical evidence; the
  2026-07-24 acceptance decision is authoritative for current training.

## 🔗 Key artifacts

- E0 evaluator: `experiments/evaluate_physical_teacher_e0.py`
- Full trainer: `experiments/train_physiology_semantic_tokenizer.py`
- E0 pilot result: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260703_165153_e0_teacher_validity_pilot_v1/`
- Teacher-free smoke: `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260703_165220_tokenizer_reconstruction_baseline_pilot_v1/`
- Resume verification: `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260703_165236_tokenizer_reconstruction_baseline_pilot_v1/`
