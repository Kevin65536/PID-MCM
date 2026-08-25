# Source-code authority map

_Retained code ownership; no active forward runtime, updated 2026-08-25_

| Package | Retained responsibility |
| --- | --- |
| `data/` | registry, unified loaders, preprocessing, alignment, masks, geometry, and retained target contracts |
| `inference/` | retained neurovascular inference used by frozen evidence |
| `tokenizers/` | stopped E2 tokenizer, corrected EMA VQ, and retained R2 diagnostic component (replay-only) |
| `analysis/` | stopped Token Physiology Atlas |
| `losses/` | losses required by the stopped E2 runtime |
| `metrics/` | trajectory reliability required by the sealed evaluator |
| `foundation/` | frozen-token consumer interface |
| `visualization/` | Token Physiology Atlas figures |
| `utils/` | shared I/O and experiment utilities |

The retained public tokenizer surface has one owner:
`PhysiologySemanticTokenizer`. Generic tokenizers and the failed continuous,
lag-conditioned, and observation-SSM generations were moved to the local ignored
archive; packages do not import that archive. The E2 surface is stopped and
replay-only.

The target/loader and raw-view/teacher modules remain separate to enforce identity
and leakage boundaries. The observation–source exploration note is an abandoned
candidate snapshot and does not change this code map. Its adapters, teachers,
paths, hierarchies, and grammar are not selected; a future clean-slate flow must
version a new owner rather than modifying stopped evidence code.

The R1-P prevalidation seal fixes exact source, script, config, registry, and test
paths/hashes. The retained T0 exporter/consumer path is also left in place. See
[`../tests/README.md`](../tests/README.md) before reorganizing either surface.

Executable workflows belong under `experiments/` or the comparison package that
owns them. Registered state is generated in
[`../docs/PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md); implemented code does not
imply scientific support or data authorization.
