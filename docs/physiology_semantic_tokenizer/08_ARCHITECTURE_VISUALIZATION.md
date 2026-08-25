# Architecture visualization standard

_Ownership and maintenance contract for retained runtime and candidate views_

## Maintained views

| Owning source | Derived view | Role |
| --- | --- | --- |
| [`architecture/physiology_semantic_architecture.json`](architecture/physiology_semantic_architecture.json) | renderer output in memory | machine-readable E2 runtime semantics |
| [`architecture/physiology_semantic_runtime_overview.drawio`](architecture/physiology_semantic_runtime_overview.drawio) | [`runtime overview SVG`](figures/physiology_semantic_runtime_overview.svg) | current-runtime presentation view |
| [`architecture/physiology_semantic_architecture.drawio`](architecture/physiology_semantic_architecture.drawio) | [`detailed candidate SVG`](figures/physiology_semantic_architecture.svg) | exploratory detailed view, not runtime |
| [`architecture/observation_source_exploration_v2.drawio`](architecture/observation_source_exploration_v2.drawio) | [`exploration SVG`](figures/plans/observation_source_exploration_v2.svg) | pre-freeze replaceable candidate map |

The registry alone owns current scientific state. Runtime JSON owns exact
implementation semantics; each Draw.io file owns its matching layout and wording;
SVG is an exported review artifact. No visualization changes a scientific gate,
method freeze, or data authorization.

The v2 exploration is not a target architecture. Its semantic
[`JSON note`](architecture/observation_source_exploration_v2.json), editable Draw.io
source, exported SVG, and
[`alt text`](figures/plans/observation_source_exploration_v2.alt.txt) preserve
replaceable candidates from before the current freeze. Current principles are owned
by [`../METHOD_RATIONALE.md`](../METHOD_RATIONALE.md).

## Visual and semantic contract

- white canvas, Helvetica/CJK sans-serif fallbacks, flat pastel cards, no shadows;
- color indicates functional role, while text and line style redundantly show
  scope, implementation, and evidence state;
- `implemented`, `planned`, and `removed` must remain visually distinct;
- `admitted`, `guarded`, `blocked`, and `n_a` are evidence labels, not inferred
  from color;
- stable node/edge IDs, accessible title/description, and orthogonal edge routes;
- Draw.io-owned SVGs embed the editable diagram and must not be hand-edited.

The Python renderer still accepts an in-memory v2 overlay for validation and future
versioned designs. No tracked historical overlay is part of the active surface.

## Regeneration and checks

```bash
drawio --export --format svg --embed-diagram \
  --output docs/physiology_semantic_tokenizer/figures/physiology_semantic_runtime_overview.svg \
  docs/physiology_semantic_tokenizer/architecture/physiology_semantic_runtime_overview.drawio

drawio --export --format svg --embed-diagram \
  --output docs/physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg \
  docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.drawio

drawio --export --format svg --embed-diagram \
  --output docs/physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg \
  docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.drawio

.venv/bin/python -m pytest -q tests/test_physiology_semantic_architecture_svg.py
```

The test compares embedded Draw.io cells/styles with their sources, validates
accessible metadata and IDs, and checks the runtime renderer without recreating any
archived plan.

_Last updated: 2026-08-25_
