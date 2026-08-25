# Architecture visualization standard

_Ownership and maintenance contract for current and historical architecture views_

## Maintained views

| Owning source | Derived view | Role |
| --- | --- | --- |
| [`architecture/physiology_semantic_architecture.json`](architecture/physiology_semantic_architecture.json) | renderer output in memory | stopped E2 runtime semantics |
| [`architecture/physiology_semantic_runtime_overview.drawio`](architecture/physiology_semantic_runtime_overview.drawio) | [`runtime overview SVG`](figures/physiology_semantic_runtime_overview.svg) | stopped/replay-only presentation view |
| [`architecture/physiology_semantic_architecture.drawio`](architecture/physiology_semantic_architecture.drawio) | [`detailed candidate SVG`](figures/physiology_semantic_architecture.svg) | exploratory detailed view, not runtime |
| [`architecture/observation_source_exploration_v2.drawio`](architecture/observation_source_exploration_v2.drawio) | [`exploration SVG`](figures/plans/observation_source_exploration_v2.svg) | abandoned pre-freeze candidate map |

The registry alone owns recorded scientific state. Runtime JSON owns exact
implementation semantics; each Draw.io file owns its matching layout and wording;
SVG is an exported review artifact. No visualization changes a scientific gate,
method freeze, or data authorization.

The v2 exploration is an abandoned candidate, not a target architecture. Its semantic
[`JSON note`](architecture/observation_source_exploration_v2.json), editable Draw.io
source, exported SVG, and
[`alt text`](figures/plans/observation_source_exploration_v2.alt.txt) preserve
replaceable candidates from before the retained boundary. Retained principles are
owned by [`../METHOD_RATIONALE.md`](../METHOD_RATIONALE.md).

## Visual and semantic contract

- [`architecture/physiology_semantic_architecture.drawio`](architecture/physiology_semantic_architecture.drawio)
  and its exact exported SVG are the sole current visual-style reference;
- white canvas, Helvetica/CJK sans-serif fallbacks, flat pastel cards, no shadows,
  solid rounded macro-panels, centered card text, and left-aligned panel headings;
- color identifies modality or pathway content: EEG/source blue, fNIRS coral,
  observation teal, teacher purple, coupling orange, retained output green, and
  neutral notes gray; color does not encode implementation or evidence status;
- scientific state is written in the title/banner and accessible descriptions;
  dashed outlines are reserved for an explicit guarded boundary or diagnostic,
  not repeated as per-node status pills or a functional-role legend;
- stable node/edge IDs, accessible title/description, and orthogonal edge routes;
- Draw.io-owned SVGs embed the editable diagram and must not be hand-edited.

The stopped runtime and abandoned exploration figures preserve their historical
appearance and are not style references. They are not rewritten in place. The
Python renderer mirrors the current reference for code-native plans and still
accepts in-memory v2 overlays without changing scientific state.

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

.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py \
  --spec docs/physiology_semantic_tokenizer/architecture/pst_discovery_v1_experiment_plan.json \
  --output docs/physiology_semantic_tokenizer/figures/pst_discovery_v1_experiment_plan.svg

.venv/bin/python -m pytest -q tests/test_physiology_semantic_architecture_svg.py
```

The test compares embedded Draw.io cells/styles with their sources, validates
accessible metadata and IDs, and checks the runtime renderer without recreating any
archived plan.

_Last updated: 2026-08-25_
