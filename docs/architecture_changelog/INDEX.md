# Architecture Changelog Index

> The authoritative chronological record of every architectural change to the neuro-tokenization mainline.
> Current architecture state: [ARCHITECTURE.md](../ARCHITECTURE.md)
> Implementation plan: [IMPLEMENTATION_PLAN.md](../../IMPLEMENTATION_PLAN.md)

---

## Timeline

```mermaid
gantt
    title Architecture Evolution Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %m/%d
    section Phase 1: Structural
        Shared/Private baseline (archived) :done, 2026-04-01, 2026-05-05
        Source/Observation Migration      :done, 2026-05-06, 2026-05-08
    section Phase 2: Source Target
        HRF Convolution Target            :active, 2026-05-08, 2026-05-20
    section Phase 2A: Q-Coupling
        Coupling-Aware Quantization       :      2026-05-20, 2026-05-30
    section Phase 3: Concentration
        Concentration Prior               :      2026-05-30, 2026-06-10
    section Mechanism A
        Coupling Smoothness               :      2026-06-10, 2026-06-20
    section Mechanism C
        Causal Asymmetry                  :      2026-06-20, 2026-06-30
```

## Change Records

| # | Date | Phase | Title | Key Files | Status |
|---|------|-------|-------|-----------|--------|
| 1 | 2026-05-06 | Phase 1 | [Source/Observation Architecture Migration](2026-05-06_source_observation_migration.md) | `factorized_labram_vqnsp.py`, `registry.py`, `multimodal_tokenizer.py`, `__init__.py` | Merged |

## How to Add a New Entry

1. Copy [`template.md`](template.md) to `YYYY-MM-DD_short_title.md`
2. Fill in all sections — especially the **Before/After Mermaid diagrams**
3. Add a row to the Change Records table above
4. Update the Timeline gantt chart if needed
5. Update [ARCHITECTURE.md](../ARCHITECTURE.md) to reflect the new current state
6. If the change completes a phase, update IMPLEMENTATION_PLAN.md §10 (Implementation Order)

## Conventions

- **File naming**: `YYYY-MM-DD_short_snake_case_title.md`
- **Diagram format**: [Mermaid](https://mermaid.js.org/) — renders natively on GitHub
- **Status values**: `Planned` → `In Progress` → `Merged`
- **Git references**: Use short hashes (`abc1234..def5678`) or tags
- **Link hygiene**: Use relative links to files within the repo; all file paths from repo root
