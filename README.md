# EEG–fNIRS physiology-semantic tokenization

<!-- project-state:begin -->
## Current research status

_Generated from `research_state/registry.json`; do not edit this block._

- **主方法**（LC-SPVQ 有时延概率通道代际）— 已完成 / 证据不足：新代际已完成单种子、fit-selection-only 架构优化：lag-loss 权重 0.5 在 selection 上相对 reference 提高 task-mean macro-F1 0.0526，但历史架构未改善，且冻结 development 两任务均反向；因此仅保留 objective-weight QC 候选，不构成可迁移性能或 sharedness 科学结论。
- **Token Atlas**（Atlas Statistical tier）— 未开始 / 尚未判定：Core tier 已完成；subject bootstrap、information ledger 与 train-validation uncertainty 尚未运行。
- **对比实验**（六方法联合正式 campaign）— 已完成 / 混合结论：540/540 jobs 完成且无技术失败；42 个 cell 中 22 个可带注释报告、12 个数值被拒、2 个仅 overlap track、6 个不适用。
- **Croce 验证**（新版 Synthetic Phase 1）— 未开始 / 尚未判定：新版合成可识别性与 solver recovery 验证尚未开始。

### Next steps
- **Token Atlas** — 在冻结的 T0 上运行，不生成新 VQ。
- **Croce 验证** — 按预定 decision rule 运行 Synthetic Phase 1。
- **主方法** — 保持 protected cohort 关闭；若继续验证 lag05_h23，须先冻结新的独立评估设计，并把 selection 增益与 development 反向同时作为前置证据。

See the [generated project status](docs/PROJECT_STATUS.md) for evidence links, dependencies, and next steps.
<!-- project-state:end -->

Start with the [generated project status](docs/PROJECT_STATUS.md),
[experiment sequencing](docs/EXPERIMENT_PLAN.md), and
[documentation map](docs/README.md). For manuscript work, use the
[paper evidence index](docs/PAPER_EVIDENCE_INDEX.md).

## Main entrypoints

| Need | Document |
| --- | --- |
| What is complete, running, next, or scientifically resolved? | [Generated project status](docs/PROJECT_STATUS.md) |
| What depends on what? | [Experiment sequencing](docs/EXPERIMENT_PLAN.md) |
| What does the evidence permit us to claim? | [Method rationale](docs/METHOD_RATIONALE.md) |
| What data/mask/split contract is active? | [Data contract](docs/DATA_CONTRACT.md) |
| What code is currently runnable? | [Architecture](docs/ARCHITECTURE.md) |
| Where are experiment commands and outputs organized? | [Experiment workspace](experiments/README.md) |
| What results were retained? | [Results index](experiments/RESULTS_INDEX.md) |
| Which sources should feed the manuscript? | [Paper evidence index](docs/PAPER_EVIDENCE_INDEX.md) |
| How are comparisons run and admitted? | [Comparison protocol](docs/comparisons/PROTOCOL.md) |
| Which comparison sources and weights are prepared? | [Comparison asset status](comparative_methods/ASSET_STATUS.md) |
| How should a frozen tokenizer be analyzed? | [Token Physiology Atlas](docs/analysis/TOKEN_PHYSIOLOGY_ATLAS.md) |
| How should code and experiments be changed? | [Contributor guide](CONTRIBUTING.md) |

## Repository layout

```text
src/                    active reusable library code
tests/                  default software and shared-contract tests
experiments/            main-method configs/workflows; local runs and explicit archives
comparative_methods/    method owners plus frozen comparison history; see its README
croce_validation/       isolated physical-model validation and derived caches
docs/                   authority map, concise contracts, and dated history
data/                   immutable measured inputs and owner-local derived caches
```

Generated payloads are ignored by Git. Active tools do not recursively search
archives, upstream mirrors, caches, checkpoints, or run trees, and comparison
packages write only to their own run roots. Frozen paths stay in place when reports
or hashes depend on them; directory cleanup is versioned rather than hidden behind
compatibility layers.

## Environment and checks

Use the repository environment; system Python is not assumed to be complete.

```bash
source .venv/bin/activate
python -m pytest --collect-only -q
```

The existing physiology-semantic launcher can replay the E0–E2-compatible
runtime:

```bash
bash experiments/scripts/launch_training_nohup.sh \
  --task physiology-semantic-tokenizer \
  --config experiments/configs/physiology_semantic_tokenizer/p2_p5_software_smoke.yaml \
  --dry-run
```

Launcher availability does not change the scientific prerequisites recorded in the
unified state. A replay should preserve the original experiment configuration; a
new method generation should define its own split and analysis plan.

Historical negative and failed results are evidence, not clutter. Large
rebuildable arrays/checkpoints may be pruned only according to
[`experiments/RESULTS_INDEX.md`](experiments/RESULTS_INDEX.md), and no directory
used by a live process may be cleaned.
