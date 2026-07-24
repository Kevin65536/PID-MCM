# P2-P5 physiology-semantic code migration

**Date:** 2026-07-02

**Phase:** Phase 3 P2-P5 software implementation

**Status:** Merged; scientific gates pending

---

## 📋 Decision

Implement the approved physiology-semantic runtime behind the new `physiology_semantic` registry name without modifying or automatically registering the archived source/observation model.

The implementation includes:

- a count-and-sum EMA semantic quantizer isolated from the legacy quantizer;
- a stop-gradient physical-state patch teacher;
- independent patch-local EEG and fNIRS semantic/residual branches;
- a post-quantization five-token causal-history module;
- mask-aware state, prototype, context, reconstruction, and VQ losses;
- a gated training entry, versioned export, and four whole-brain consumer modes.

## 🏗️ Before and after

```mermaid
flowchart LR
    accTitle: Physiology Semantic Runtime Migration
    accDescr: The active target changes from a data-only contract and blocked launcher to independent local tokenizers, corrected quantization, teacher supervision, gated training, and versioned exports.

    before["P1 data contract only"] --> blocked["Blocked target launcher"]
    data["Paired EEG and optical input"] --> teacher["Physical teacher adapter"]
    data --> local["Independent patch-local encoders"]
    local --> vq["Correct count-and-sum EMA VQ"]
    vq --> context["Fixed causal history"]
    teacher --> losses["Masked semantic losses"]
    context --> losses
    losses --> gated["E0-gated training"]
    gated --> export["Versioned P5 export"]

    classDef retired fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef active fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef guarded fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef output fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class before,blocked retired
    class data,teacher,local,vq,context,losses active
    class gated guarded
    class export output
```

## 🔒 Compatibility and evidence boundary

- `NormEMAVectorQuantizer` is unchanged for historical checkpoint interpretation.
- Active inference accepts only one modality per `encode_eeg` or `encode_fnirs` call.
- Teacher and decomposition tensors are training/audit inputs, never inference inputs.
- `e0_passed: false` forces zero optimizer steps even in software-smoke mode.
- The successful dry-run, smoke, checkpoint, and export establish integration correctness only. E0, G0, G2, G3, and all coupling claims remain unevaluated.

## ✅ Verification

- Full active suite: 65 tests passed, including distributed quantizer statistics.
- Real-cache dry-run: `20260702_235450_p2_p5_software_smoke`.
- Real-cache smoke under the historical pre-calibration E0 state: `20260702_235459_p2_p5_software_smoke`.
- P5 validation export: one validation sample with top-8 posterior and manifest.

**Implementation commit:** `f13363e`
