# Pre-redesign experiment archive

_Read-only generated artifacts moved out of the active result namespace on 2026-07-01_

---

## 📋 Scope

This archive contains every experiment family that existed under `experiments/runs/` at the physiology-semantic design freeze. The move was a same-filesystem rename: no run payload was deleted or rewritten.

The archived run families contain approximately 715 GiB across 17,614 files. The two comparison-report packages add 20 lightweight files.

## 🗂️ Layout

| Path | Contents |
| --- | --- |
| `runs/` | Source/observation, coupling, exchange, alignment, meta-analysis, and downstream run families |
| `comparison_reports/` | HighWL comparison reports formerly under `experiments/comparison_reports/` |
| [`archive_manifest.tsv`](archive_manifest.tsv) | Original path, archive path, byte count, and file count |

## 📊 Inventory summary

| Run family | Size | Files | Role |
| --- | ---: | ---: | --- |
| `coupling_design_audit` | 210 GiB | 4,260 | Identifiability audit |
| `tokenizer_next_stage` | 139 GiB | 2,677 | Capacity/transfer suites |
| `tokenizer_coupling_capacity` | 129 GiB | 2,563 | Coupling-capacity experiments |
| `tokenizer_lightweight_alignment` | 63 GiB | 1,459 | Pre-VQ alignment |
| `tokenizer_cross_modal_exchange` | 42 GiB | 1,066 | Causal exchange, including X3 |
| Remaining run families | 135 GiB | 5,589 | Source/observation, coupling, downstream, and meta-analysis |

## 🛡️ Access rule

Archive analysis requires an explicit archive path. Active discovery and comparison tools must never include this directory by default. Absolute paths embedded inside old manifests remain provenance records and may refer to the original pre-move location.

## 🔗 Active replacement

- [Active result root](../../runs/README.md)
- [New suite schema](../../runs/physiology_semantic_tokenizer/README.md)
- [Storage policy](../../../docs/STORAGE_LAYOUT.md)
- [Legacy design postmortem](../../../docs/physiology_semantic_tokenizer/01_LEGACY_DESIGN_POSTMORTEM.md)

_Archived: 2026-07-01_
