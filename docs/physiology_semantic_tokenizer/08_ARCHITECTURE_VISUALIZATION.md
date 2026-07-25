# SVG architecture visualization standard

_Drawing and maintenance contract for canonical runtime diagrams and plan overlays_

---

## 🗺️ Maintained views

The canonical figure describes the checkout's current runtime and evidence boundary. It must never display a proposed component as implemented merely because the proposal is scientifically plausible.

![Current physiology-semantic tokenizer runtime](figures/physiology_semantic_architecture.svg)

| Artifact | Role |
| --- | --- |
| [`architecture/physiology_semantic_architecture.json`](architecture/physiology_semantic_architecture.json) | Canonical sections, node semantics, geometry, evidence axes, and routed edges |
| [`figures/physiology_semantic_architecture.svg`](figures/physiology_semantic_architecture.svg) | Generated current-runtime figure |
| [`architecture/change_overlay.example.json`](architecture/change_overlay.example.json) | Overlay template |
| `experiments/scripts/render_physiology_semantic_architecture.py` | Deterministic renderer, validator, and drift checker |

The active proposed after-state is rendered separately from the runtime view:

![Proposed Shared-Driver Semantic VQ](figures/plans/shared_driver_semantic_return_plan.svg)

Its source is
[`architecture/shared_driver_semantic_return_plan.json`](architecture/shared_driver_semantic_return_plan.json).
It records raw-only modality streams, pre-VQ full-window context, independent
K128 codebooks, a full joint-driver-proxy primary target, removal of mandatory
residual paths, and separate R6A offline/R6B strict-cutoff raw-fNIRS evaluations. Its plan banner and
dashed implementation styling are part of the evidence boundary: no R-series
gate has run.

The older
[`physical_teacher_gradient_entry_plan`](figures/plans/physical_teacher_gradient_entry_plan.svg)
is retained as a superseded historical plan. It documents the E2-era
local/prototype/context routing and preserve–discover–certify proposal; it is no
longer the approved after-state.

The JSON is the text-diffable source of truth. Generated SVG files are committed for direct review and documentation rendering. Manual SVG edits are prohibited.

## 🔒 Content-preservation boundary

The drawing system does not redesign the model. Node labels, tensor contracts, statuses, sections, and graph relationships remain authoritative in the existing architecture and overlay JSON files.

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

## 📐 Geometry and typography

- Canvas and section positions are declared in the canonical JSON; legacy geometry is scaled uniformly in memory and the source file remains unchanged.
- Node height must fit every declared detail line. The renderer rejects text-overflow geometry before producing an SVG.
- Node titles use at least 15 px and details at least 12.5 px in the native viewBox.
- Cards use a redundant left accent, implementation border, scope label, and evidence pill.
- The font stack includes common CJK fonts before system fallbacks so English and Chinese annotations remain legible.
- Footer legends and plan callouts are outside the architecture content region. Callout height expands the SVG viewBox dynamically.
- A plan with too many relationships must add a grouped objective/bus or be split into linked figures; shrinking text is not an accepted remedy.

## 🔗 Edge and routing contract

Every canonical edge has a stable `id`, declared `style`, and resolvable endpoints. Edges are rendered as orthogonal paths with matching-color arrowheads.

| Edge style | Meaning |
| --- | --- |
| solid gray-blue | measured/inference data flow |
| purple dashed | training-only supervision |
| amber dotted | permitted gradient path |
| gray dash-dot | lifecycle/control transition |
| rose dashed | frozen evaluation path |
| amber/gray long-dash | guarded or blocked transition |

Use `route.from_side`, `route.to_side`, optional endpoint fractions, and `route.via` waypoints to keep paths out of unrelated cards. Edge labels are rendered on opaque rounded chips and placed on an explicit or longest safe segment. Flow semantics and plan-delta semantics remain separate: adding a training edge does not turn its arrow green.

## 🧩 Proposed after-state overlays

An overlay composes a separate drawing while preserving the canonical JSON in memory and on disk. Overlay content remains a plan artifact; upgrading its rendering does not approve or revise the plan.

```json
{
  "schema": "physiology_semantic_architecture_changes_v2",
  "plan_id": "example_plan",
  "title": "Proposed After-State · Example",
  "subtitle": "Plan-specific evidence boundary",
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

The renderer retains canonical implementation/evidence values in `data-canonical-status` and `data-canonical-evidence` on modified nodes. A v1 overlay remains readable for historical compatibility, but new plans must use v2 after-state replacements.

## 🔄 Regeneration and drift checks

```bash
.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py

.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py \
  --changes docs/physiology_semantic_tokenizer/architecture/measurement_first_input_contract_plan.json \
  --output docs/physiology_semantic_tokenizer/figures/plans/measurement_first_input_contract_plan.svg

.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py \
  --changes docs/physiology_semantic_tokenizer/architecture/shared_driver_semantic_return_plan.json \
  --output docs/physiology_semantic_tokenizer/figures/plans/shared_driver_semantic_return_plan.svg

.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py --check

.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py \
  --changes docs/physiology_semantic_tokenizer/architecture/shared_driver_semantic_return_plan.json \
  --output docs/physiology_semantic_tokenizer/figures/plans/shared_driver_semantic_return_plan.svg \
  --check

.venv/bin/python -m pytest -q tests/test_physiology_semantic_architecture_svg.py
```

A plan overlay is rejected if its output path would overwrite the canonical SVG.

## ✅ Acceptance contract

- Canonical and registered plan SVGs are byte-identical to fresh renders.
- Root SVG, every node, every edge, and every callout has an accessible name and stable XML ID.
- Node, edge, change, and endpoint IDs are unique and resolvable.
- Architecture edges contain orthogonal line segments, not unconstrained curves through unrelated cards.
- Plan title, description, and callouts are plan-specific when an overlay declares them.
- Added/modified/removed objects are distinguishable by text/shape as well as color.
- Dynamic callouts remain inside the expanded viewBox.
- Diagram language preserves the statuses already declared by the architecture source; drawing changes cannot advance a scientific or experiment gate.

When model behavior changes, update the architecture changelog. When only renderer or maintenance policy changes, use the project changelog.

_Last updated: 2026-07-25_
