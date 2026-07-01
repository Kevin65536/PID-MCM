# Phase 1 Gate1 baseline lock and archive

_Date: 2026-05-11 · Phase: Phase 1 · Status: Merged_

_Links: [archive handoff log](../archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md) · [current runtime architecture](../ARCHITECTURE.md)_

---

## 🎯 Motivation

Phase 1 produced a no-phase source/observation baseline that passed the historical Gate1 health criteria. The search was closed and preserved before source-target redesign began so later experiments could not silently redefine the baseline.

## 🔀 Architecture delta

### Before

```mermaid
flowchart LR
    accTitle: Open Gate1 tuning loop
    accDescr: Multiple source observation tuning runs competed without a locked reusable configuration or a single archived handoff artifact.

    tuning_runs["🧪 Gate1 tuning runs"] --> changing_baseline["⚠️ Changing baseline"]
    changing_baseline --> phase2["🔧 Phase 2 experiments"]

    classDef warning fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    class tuning_runs,changing_baseline,phase2 warning
```

### After

```mermaid
flowchart LR
    accTitle: Locked Gate1 handoff
    accDescr: One configuration and one best run were frozen as the Phase 1 reference, with the complete tuning surface retained in a formal archive.

    tuning_runs["🧪 Gate1 tuning runs"] --> archive["🗂️ Formal run archive"]
    tuning_runs --> locked_config["🔒 Locked baseline config"]
    locked_config --> phase2["🔧 Phase 2 handoff"]
    archive -.-> phase2

    classDef archive_style fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef locked fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class tuning_runs,archive archive_style
    class locked_config,phase2 locked
```

## 🧱 Component changes

| Component | Change | Description |
| --- | --- | --- |
| `gate1_baseline_locked_bs128.yaml` | Added | Reusable no-phase baseline |
| `gate1_best_current.yaml` | Updated | Alias for the best historical Gate1 run |
| Phase 1 run archive | Added | Preserved all tuning attempts and manifests |
| Archive handoff log | Added | Recorded Gate1–Gate4 outcome and promotion boundary |

## 🚦 Gate outcome

| Gate | Historical result | Meaning |
| --- | --- | --- |
| Gate1 | Pass | Codebook health baseline locked |
| Gate2 | Fail | Cross-modal predictability unresolved |
| Gate3 | Fail | Coupling structure diffuse |
| Gate4 | Fail | Utility and leakage unresolved |

The baseline was approved as a structural handoff, not as a physiological-coupling solution.

## 🔗 Archived artifacts

- [Locked baseline config](../../experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml)
- [Best baseline alias](../../experiments/configs/source_observation/phase1/gate1_best_current.yaml)
- [Best archived run](../../experiments/archive/source_observation_phase1_gate1_stabilization_20260511/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718)
- [Archive manifest](../../experiments/archive/source_observation_phase1_gate1_stabilization_20260511/manifest.json)

_Reconstructed from the preserved handoff log on 2026-07-01._
