# Architecture visualization v2

**Date:** 2026-07-16

**Scope:** Documentation and architecture-maintenance infrastructure

---

> Historical record, not current style guidance. The role/status palette described
> below was retired on 2026-08-25; the current style owner is
> [`physiology_semantic_architecture.drawio`](../physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.drawio).

## Change

Improved the drawing and maintenance system for the existing physiology-semantic tokenizer architecture without changing its design or experiment plans:

- preserved the canonical v1 architecture JSON and expanded its geometry only at render time for more legible text and spacing;
- normalized functional role, runtime scope, implementation state, and scientific evidence into independent visual channels without modifying source statuses;
- added stable edge IDs, orthogonal routed paths, matching arrowheads, label chips, and accessible edge descriptions;
- upgraded plan overlays from outline-only annotations to non-mutating proposed after-state composition with node replacement, edge replacement, and dynamic numbered callouts;
- retained backward compatibility with the two existing plan JSON files and regenerated only their SVG presentation;
- expanded deterministic tests to cover canonical/plan drift, content preservation, non-mutation, stable/accessibly named edges, dynamic footer coverage, v1 compatibility, and validation failures.

This entry is strictly a visualization/tooling change. It does not add, remove, approve, or revise any model component, loss, target, gate, experiment, or scientific claim.

## Artifacts

- [`Visualization standard`](../physiology_semantic_tokenizer/08_ARCHITECTURE_VISUALIZATION.md)
- [`Canonical runtime SVG`](../physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg)
- `experiments/scripts/render_physiology_semantic_architecture.py`
- `tests/test_physiology_semantic_architecture_svg.py`
