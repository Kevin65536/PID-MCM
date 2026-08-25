# Source-code authority map

_Active and frozen runtime boundaries, updated 2026-08-25_

| Package | Current responsibility |
| --- | --- |
| `data/` | registry, unified loaders, preprocessing, alignment, masks, geometry, and retained target contracts |
| `inference/` | retained neurovascular inference used by frozen evidence |
| `tokenizers/` | E2 tokenizer, corrected EMA VQ, and retained R2 diagnostic component |
| `analysis/` | Token Physiology Atlas |
| `losses/` | losses required by the retained E2 runtime |
| `metrics/` | trajectory reliability required by the sealed evaluator |
| `foundation/` | frozen-token consumer interface |
| `visualization/` | Token Physiology Atlas figures |
| `utils/` | shared I/O and experiment utilities |

The current public tokenizer registry has one owner:
`PhysiologySemanticTokenizer`. Generic tokenizers and the failed continuous,
lag-conditioned, and observation-SSM generations were moved to the local ignored
archive; active packages do not import that archive.

The target/loader and raw-view/teacher modules remain separate to enforce identity
and leakage boundaries. The observation–source exploration note does not change
this code map: its adapters, teachers, paths, hierarchies, and grammar remain
replaceable candidates. Add a selected implementation beside the nearest existing
owner rather than modifying frozen evidence code.

The R1-P prevalidation seal fixes exact source, script, config, registry, and test
paths/hashes. The retained T0 exporter/consumer path is also left in place. See
[`../tests/README.md`](../tests/README.md) before reorganizing either surface.

Executable workflows belong under `experiments/` or the comparison package that
owns them. Current state is generated in
[`../docs/PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md); implemented code does not
imply scientific support or data authorization.
