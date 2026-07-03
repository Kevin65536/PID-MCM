# SVG architecture visualization system

**Date:** 2026-07-03

**Scope:** Documentation and architecture-maintenance infrastructure

---

## 📋 Change

Added a deterministic SVG architecture system for the physiology-semantic tokenizer:

- a JSON source specification for current nodes, tensor boundaries, statuses, and edges;
- a generated canonical SVG with accessible labels and scientific-gate status;
- plan overlays for `add`, `modify`, and `remove` annotations;
- a renderer drift check and XML/coverage tests;
- a rule requiring annotated SVGs for future model modification plans.

This entry belongs to the project changelog because it changes documentation maintenance and planning policy, not model computation.

## 🔗 Artifacts

- [`Current visualization and maintenance contract`](../physiology_semantic_tokenizer/08_ARCHITECTURE_VISUALIZATION.md)
- [`Canonical SVG`](../physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg)
- `experiments/scripts/render_physiology_semantic_architecture.py`
