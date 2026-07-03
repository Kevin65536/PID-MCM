# SVG architecture visualization system

_Maintenance contract for the current model diagram and plan-specific change overlays_

---

## 🖼️ Current architecture

The generated SVG below is the maintained visual representation of the currently implemented physiology-semantic runtime. It distinguishes software implementation from training-only supervision, gated execution, and blocked future work.

![Current physiology-semantic tokenizer architecture](figures/physiology_semantic_architecture.svg)

The JSON specification is the text-diffable source of truth. The SVG is a generated, reviewable artifact; both are committed.

| Artifact | Role |
| --- | --- |
| [`architecture/physiology_semantic_architecture.json`](architecture/physiology_semantic_architecture.json) | Nodes, tensor labels, status, position, and edges |
| [`figures/physiology_semantic_architecture.svg`](figures/physiology_semantic_architecture.svg) | Canonical current-implementation diagram |
| [`architecture/change_overlay.example.json`](architecture/change_overlay.example.json) | Template for plan annotations |
| `experiments/scripts/render_physiology_semantic_architecture.py` | Deterministic renderer and drift checker |

## 🎨 Visual semantics

| Style | Meaning |
| --- | --- |
| Blue | Implemented inference or model component |
| Green | Runtime interface or exported representation |
| Purple | Privileged training-only teacher or objective |
| Amber | Implemented but blocked by an explicit gate |
| Gray | Not implemented or scientifically blocked |
| Red dashed outline | Existing component modified by a proposed plan |
| Green dashed outline | Component added by a proposed plan |
| Dark red cross | Component removed by a proposed plan |

The baseline SVG contains no plan overlay. Its subtitle must state the current scientific-gate boundary so a software-complete component cannot be mistaken for a validated scientific claim.

## 🔄 Updating the current diagram

After a merged model, data-flow, loss, training-lifecycle, export, or consumer change:

1. inspect the merged implementation and update the JSON specification;
2. update affected node details, statuses, and edges;
3. regenerate the canonical SVG;
4. run the drift and XML tests;
5. update the architecture changelog when model behavior changed, or the project changelog when only visualization infrastructure changed.

```bash
.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py
.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py --check
.venv/bin/python -m pytest -q tests/test_physiology_semantic_architecture_svg.py
```

Manual edits to the generated SVG are prohibited because regeneration would discard them.

## 📌 Annotating modification plans

Every future model modification plan is incomplete until it includes a plan-specific annotated SVG. The plan author must:

1. copy `change_overlay.example.json` to a plan-specific JSON file;
2. reference stable baseline node IDs in `changes`;
3. declare new components in `add_nodes` and their connections in `add_edges`;
4. generate a plan-specific SVG without overwriting the canonical current diagram;
5. embed or link that annotated SVG from the plan document.

```bash
.venv/bin/python experiments/scripts/render_physiology_semantic_architecture.py \
  --changes docs/physiology_semantic_tokenizer/architecture/<plan_id>.json \
  --output docs/physiology_semantic_tokenizer/figures/plans/<plan_id>.svg
```

Allowed change kinds are `add`, `modify`, and `remove`. Each change requires a concise `note` that states the intended architectural effect rather than only a filename. After implementation, the canonical specification is updated to the new current state; the plan-specific SVG remains historical design evidence.

## ✅ Acceptance contract

- The canonical SVG must be byte-identical to a fresh render from the JSON specification.
- Every SVG must contain an accessible title and description.
- Every node has a stable XML ID and machine-readable implementation status.
- Every edge endpoint and overlay reference must resolve to a declared node.
- Plan overlays cannot overwrite the canonical current SVG.
- Diagram status must preserve the distinction between software correctness and scientific gate completion.

_Last updated: 2026-07-03_
