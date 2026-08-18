# Source-code authority map

_Reusable and compatibility boundaries, updated 2026-07-30_

New code may depend on reusable packages but must not import the dated
compatibility namespace. Historical tools opt into compatibility explicitly.

| Package | Active responsibility |
| --- | --- |
| `data/` | registry, unified loaders, preprocessing, alignment, masks, geometry, caches, E0–E2 and R-series joins |
| `inference/` | neurovascular SSM inference and the sealed R1-P teacher |
| `tokenizers/` | tokenizer interfaces, corrected EMA VQ, E2 runtime, R2 diagnostic model |
| `analysis/` | E2 evaluation and Token Physiology Atlas |
| `losses/` | physiology-semantic reconstruction/routed objectives plus historical reusable losses |
| `metrics/` | reconstruction, codebook-health, and compatibility metrics |
| `foundation/` | downstream token consumer interface; not required by the stopped mainline |
| `visualization/` | Atlas figures and older reusable dashboards |
| `utils/` | I/O, run comparison, logging/launch compatibility utilities |
| `compatibility/pre_physiology_semantic_20260701/` | dated source/observation checkpoint and replay surface |

Current primary files:

- `data/registry.py`, `factory.py`, `unified_physiology.py`;
- `data/physiology_semantic_local.py`,
  `physiology_semantic_targets.py`;
- `data/shared_driver_targets.py`, `shared_driver_dataset.py`;
- `tokenizers/ema_vector_quantizer.py`,
  `physiology_semantic_tokenizer.py`,
  `shared_driver_semantic_vq.py`;
- `inference/adaptive_neurovascular_ssm.py`;
- `analysis/physiological_patch_features.py`,
  `token_physiology.py`, `token_information_ledger.py`,
  `token_sequence.py`, `token_physiology_atlas.py`.

The target/loader and raw-view/teacher modules remain separate to enforce
identity and leakage boundaries; they are not duplication to merge away.
Several general tokenizer, metric, loss, and visualization modules exist only
for historical configs/checkpoints. They remain in place until checkpoint
serialization compatibility is audited.

The R1-P prevalidation seal fixes exact source, script, config, registry, and
test paths/hashes. Ordinary cleanup must not move or edit them. Likewise,
experiment source-snapshot tests rely on several current paths. See
[`../tests/README.md`](../tests/README.md) before reorganizing packages.

Executable workflows belong under `experiments/` or the comparison package
that owns them, not under `src/`. Current execution and scientific verdicts are
generated in [`../docs/PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md); dependency
order is in [`../docs/EXPERIMENT_PLAN.md`](../docs/EXPERIMENT_PLAN.md). Implemented
code does not imply scientific support.
