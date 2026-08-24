# Architecture visualization standard

_Drawing and maintenance contract for runtime, candidate, and historical views_

---

## 🗺️ Maintained views

Current runtime facts and candidate drawings are deliberately separate. A polished
diagram must never turn a proposal into an implemented component or a selected
method.

![Detailed observation–source candidate architecture](figures/physiology_semantic_architecture.svg)

| Owning source | Derived view | Role |
| --- | --- | --- |
| [`architecture/physiology_semantic_architecture.json`](architecture/physiology_semantic_architecture.json) | none | Machine-readable current-runtime semantics; not a visual-style owner |
| [`architecture/physiology_semantic_runtime_overview.drawio`](architecture/physiology_semantic_runtime_overview.drawio) | [`runtime overview SVG`](figures/physiology_semantic_runtime_overview.svg) | Current-runtime presentation view |
| [`architecture/physiology_semantic_architecture.drawio`](architecture/physiology_semantic_architecture.drawio) | [`detailed candidate SVG`](figures/physiology_semantic_architecture.svg) | Editable detailed candidate view; exploratory, not runtime |
| [`architecture/observation_source_exploration_v2.drawio`](architecture/observation_source_exploration_v2.drawio) | [`exploration SVG`](figures/plans/observation_source_exploration_v2.svg) | Compact observation–source candidate map |
| Overlay JSON files | Historical plan SVGs | Text-diffable historical content rendered by `experiments/scripts/render_physiology_semantic_architecture.py` |

The registry remains the only owner of current scientific state. The runtime JSON
owns machine-readable implementation semantics. A Draw.io file owns the layout,
wording, and visual style of its matching hand-authored SVG; the SVG is only an
export. Historical overlay JSON owns its overlay content, and the renderer owns
only those generated overlay SVGs. This separation prevents either renderer from
overwriting a requested visual design.

The plan overlays below are kept only where they help explain why the current
runtime has the boundaries it does. Their labels are deliberately explicit so a
reader does not mistake a historical proposal for an active component:

| Overlay | Reading status | What it is useful for |
| --- | --- | --- |
| [`shared_driver_semantic_return_plan`](figures/plans/shared_driver_semantic_return_plan.svg) | **Historical pre-gate plan** (2026-07-25), not runtime | Records the shared-driver/SD-SVQ proposal before R1-P and R2-D evidence; it is not an active after-state. |
| [`measurement_first_input_contract_plan`](figures/plans/measurement_first_input_contract_plan.svg) | **Merged historical overlay**, not a separate runtime | Shows how the measured-input contract subsumed the earlier entrance proposal. |
| [`physical_teacher_gradient_entry_plan`](figures/plans/physical_teacher_gradient_entry_plan.svg) | **Superseded historical plan**, not runtime | Preserves the E2-era preserve–discover–certify proposal for traceability. |
| [`shared_state_reconstruction_bound_plan`](figures/plans/shared_state_reconstruction_bound_plan.svg) | **Diagnostic-only historical overlay**, not runtime | Captures a bounded shared/private-state diagnostic that was never a promotion gate. |
| [`observation_source_exploration_v2`](figures/plans/observation_source_exploration_v2.svg) | **Pre-freeze candidate snapshot**, not runtime or current architecture contract | Preserves earlier implementation candidates. Its optional-grammar wording is superseded by the frozen endpoint/proper-score/null kernel in `METHOD_RATIONALE.md`; exact modules remain replaceable. |
| [`physiology_semantic_runtime_overview`](figures/physiology_semantic_runtime_overview.svg) | **Quick overview / paper candidate** | Human-readable current-runtime orientation; use the runtime JSON for exact implementation detail. |
| [`physiology_semantic_architecture`](figures/physiology_semantic_architecture.svg) | **Detailed exploratory candidate view** | Shows replaceable source/observation candidates; it is not a target or current-runtime contract. |

The shared-driver source is
[`architecture/shared_driver_semantic_return_plan.json`](architecture/shared_driver_semantic_return_plan.json).
It records raw-only modality streams, pre-VQ full-window context, independent
K128 codebooks, a full joint-driver-proxy primary target, removal of mandatory
residual paths, and separate R6A offline/R6B strict-cutoff raw-fNIRS evaluations.
Those are historical plan contents; the source banner and dashed implementation
styling are evidence-boundary cues, not a claim that the R-series plan ran.

The v2 exploration is a pre-freeze design snapshot, not a planned after-state,
current runtime, or current method-boundary owner. Its semantic note
[`architecture/observation_source_exploration_v2.json`](architecture/observation_source_exploration_v2.json),
editable visual source
[`architecture/observation_source_exploration_v2.drawio`](architecture/observation_source_exploration_v2.drawio),
exported figure, and concise
[`alt text`](figures/plans/observation_source_exploration_v2.alt.txt) jointly
describe the earlier candidate continuous teachers, modality-independent paths,
token hierarchies, grammar, optional conditional analyses, and evaluation
boundary. The symbolic inputs contain no measured values; current frozen
principles are owned by [`../METHOD_RATIONALE.md`](../METHOD_RATIONALE.md).

The JSON is a text-diffable semantic note, not a frozen method contract or visual
source. Generated SVG files are committed for direct review and documentation
rendering. Change a Draw.io-authored figure in its matching `.drawio` source;
manual SVG edits remain prohibited.

## 🔒 Content-preservation boundary

The drawing system does not redesign the runtime. Runtime semantics remain in the
runtime JSON; historical overlay semantics remain in their overlay JSON; candidate
wording and layout remain in the matching Draw.io source. None of these visual
sources can advance the registry's scientific state.

For the legacy v1 specification, the renderer uniformly expands the canvas and geometry at render time so titles and detail lines remain legible. This transformation is deliberately absent from the source JSON diff: it changes presentation, not architecture. New routing or visual metadata must not be interpreted as a model edge, loss, gate, or experiment update.

## 🧭 Four independent state axes

The legacy `status` field mixes functional role, runtime availability, and scientific validity. The renderer normalizes it into four visual axes without changing the canonical v1 source. The optional v2 overlay schema can declare the axes explicitly:

| Axis | Values | Visual channel |
| --- | --- | --- |
| `role` | `data`, `teacher`, `encoder`, `latent`, `quantizer`, `objective`, `lifecycle`, `interface`, `evaluator` | Color-blind-safe card fill and accent |
| `scope` | `inference`, `training_only`, `export`, `evaluation`, `governance` | Text label and section placement |
| `implementation` | `implemented`, `planned`, `removed` | Solid border, dashed border, or crossed/faded card |
| `evidence` | `admitted`, `guarded`, `blocked`, `n_a` | Redundant text pill; never color alone |

This prevents a passing software smoke test from being read as a scientific admission while preserving the original architecture status text.

## 🎨 Shared visual language

All actively maintained architecture figures use the same quiet, paper-ready
visual language:

- white canvas; Helvetica first, followed by CJK-capable sans-serif fallbacks;
- 28 px dark title, 14 px slate subtitle, 15 px card title, and at least 12 px detail text;
- 16 px rounded cards, 2 px flat borders, pastel fills, no gradients or shadows;
- pale dashed zones for runtime/training scope and a neutral gray status banner;
- blue for measured data, amber for observation/transform paths, purple for teacher/source candidates, green for vocabularies/exports, rose for evaluation;
- solid blue measured flow, dashed purple training-only flow, and gray or rose guarded/evaluation flow; status must also be written in text and never encoded by color alone;
- `source` and `observation` are the branch names. `innovation` may appear only as a mathematical residual field, never as a branch, method, or project identity.

Dense detailed views may left-align the title; compact overview and exploration
views center it. Node height must fit every detail line. A crowded plan must group
or split content instead of shrinking text.

## 🔗 Edge and routing contract

Every canonical edge has a stable `id`, declared `style`, and resolvable endpoints. Edges are rendered as orthogonal paths with matching-color arrowheads.

| Edge style | Meaning |
| --- | --- |
| solid gray-blue | measured/inference data flow |
| purple dashed | training-only supervision |
| amber dotted | permitted gradient path |
| gray dash-dot | lifecycle/control transition |
| rose dashed | preregistered evaluation path |
| amber/gray long-dash | guarded or blocked transition |

Use `route.from_side`, `route.to_side`, optional endpoint fractions, and `route.via` waypoints to keep paths out of unrelated cards. Edge labels are rendered on opaque rounded chips and placed on an explicit or longest safe segment. Flow semantics and plan-delta semantics remain separate: adding a training edge does not turn its arrow green.

## 🧩 Plan and historical overlays

An overlay composes a separate drawing while preserving the canonical JSON in memory and on disk. Overlay content remains a plan or historical artifact; upgrading its rendering does not approve or revise the plan.

```json
{
  "schema": "physiology_semantic_architecture_changes_v2",
  "plan_id": "example_plan",
  "title": "Template Overlay · Example Plan",
  "subtitle": "Plan-specific evidence boundary",
  "banner": "TEMPLATE ONLY, not runtime · replace every placeholder before use.",
  "changes": [
    {
      "node_id": "eeg_quantizer",
      "kind": "modify",
      "note": "State the architectural effect",
      "replace": {
        "label": "Proposed EEG vocabulary",
        "details": ["after-state content"],
        "implementation": "planned"
      }
    }
  ]
}
```

Overlay fields:

- `replace` may change `label`, `details`, `role`, `scope`, `implementation`, and `evidence`, but never the stable node ID;
- `layout` may adjust `x`, `y`, `width`, and `height` for the plan view only;
- `add_nodes` and `add_edges` declare new plan objects;
- `edge_changes` can replace a baseline edge's label, style, or route, or mark it removed;
- every node/edge delta requires a concise `note` and receives a visible `A<n>`, `M<n>`, or `R<n>` reference tied to the callout panel.

The renderer retains canonical implementation/evidence values in `data-canonical-status` and `data-canonical-evidence` on modified nodes. A v1 overlay remains readable for historical compatibility. New plans may use v2 after-state replacements, but a rendered overlay is still only a plan/documentation view until the experiment registry records implementation and evidence.

## 🔄 Regeneration and drift checks

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
  --changes docs/physiology_semantic_tokenizer/architecture/measurement_first_input_contract_plan.json \
  --output docs/physiology_semantic_tokenizer/figures/plans/measurement_first_input_contract_plan.svg

.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py \
  --changes docs/physiology_semantic_tokenizer/architecture/shared_driver_semantic_return_plan.json \
  --output docs/physiology_semantic_tokenizer/figures/plans/shared_driver_semantic_return_plan.svg

.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py \
  --changes docs/physiology_semantic_tokenizer/architecture/shared_driver_semantic_return_plan.json \
  --output docs/physiology_semantic_tokenizer/figures/plans/shared_driver_semantic_return_plan.svg \
  --check

.venv/bin/python -m pytest -q tests/test_physiology_semantic_architecture_svg.py
```

Draw.io exports must retain `role="img"`, a concise title, and a description.
The test checks the embedded Draw.io cell values/styles against the matching source.
The historical-overlay renderer rejects either Draw.io-owned SVG path as output.

## ✅ Acceptance contract

- Draw.io-authored SVGs embed cell values and styles matching their editable source.
- Registered historical-overlay SVGs are byte-identical to fresh renders.
- Every root SVG has an accessible title and description; generated overlay nodes, edges, and callouts retain stable accessible IDs.
- Node, edge, change, and endpoint IDs are unique and resolvable.
- Architecture edges contain orthogonal line segments, not unconstrained curves through unrelated cards.
- Plan title, description, and callouts are plan-specific when an overlay declares them.
- Added/modified/removed objects are distinguishable by text/shape as well as color.
- Dynamic callouts remain inside the expanded viewBox.
- Diagram language preserves the statuses already declared by the architecture source; drawing changes cannot advance a scientific or experiment gate.

When model behavior changes, update the architecture changelog. When only renderer or maintenance policy changes, use the project changelog.

_Last updated: 2026-08-24_
