# EEG–fNIRS physiology-semantic tokenization

_Repository status and contributor entrypoint, 2026-07-01_

---

## 📋 Current status

The repository is transitioning from the frozen source/observation and tokenizer-coupling lineage to an approved physiology-semantic architecture. The target code has not yet been implemented or experimentally validated.

The new design separates:

1. uncertainty-aware physical-state supervision;
2. independently inferred EEG and fNIRS semantic tokens;
3. private/residual representations for information preservation;
4. frozen-token EEG-sequence-to-fNIRS-distribution analysis.

Start with the [documentation authority map](docs/README.md). Do not use archived source/observation plans as implementation instructions.

## 🧭 Authority map

| Need | Document |
| --- | --- |
| Design entrypoint | [Physiology-semantic archive](docs/physiology_semantic_tokenizer/README.md) |
| Target architecture and tensors | [Target architecture](docs/physiology_semantic_tokenizer/02_TARGET_ARCHITECTURE.md) |
| Theory and claim limits | [Theoretical foundations](docs/physiology_semantic_tokenizer/03_THEORETICAL_FOUNDATIONS.md) |
| Implementation and correctness plan | [Implementation and validation](docs/physiology_semantic_tokenizer/04_IMPLEMENTATION_VALIDATION_PLAN.md) |
| Experiment suites | [Experiment design](docs/physiology_semantic_tokenizer/05_EXPERIMENT_DESIGN.md) |
| New-design results | [Active experiment log](docs/physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) |
| Output paths | [Storage layout](docs/STORAGE_LAYOUT.md) |
| Runnable frozen implementation | [Current code architecture](docs/ARCHITECTURE.md) |

## 🏗️ Repository structure

```text
src/                                  # Reusable model, data, loss, and analysis code
experiments/
├── configs/
│   ├── physiology_semantic_tokenizer/ # Active target configs; initially empty
│   └── ...                            # Frozen compatibility configs
├── scripts/                           # Executable entrypoints
├── runs/
│   └── physiology_semantic_tokenizer/ # Only active generated-result namespace
└── archive/
    └── pre_physiology_semantic_20260701/ # All runs present before design freeze
docs/
├── README.md                          # Documentation authority map
├── physiology_semantic_tokenizer/     # Active design, plan, experiments, log
├── ARCHITECTURE.md                    # Runnable frozen implementation truth
├── STORAGE_LAYOUT.md                  # Active/archive storage contract
└── archive/pre_physiology_semantic_20260701/ # Superseded plans and analyses
croce_validation/                      # Physical-model validation and caches
tests/                                 # Unit and integration tests
```

## 🚀 Environment

Use the repository virtual environment:

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

The supported launcher remains:

```bash
bash experiments/scripts/launch_training_nohup.sh --task TASK [task arguments]
```

No target-architecture training task is registered yet. Existing `source-observation-tokenizer`, token-export, whole-brain, and coupling-suite commands reproduce the archived lineage only.

## 📦 Result policy

New outputs must use:

```text
experiments/runs/physiology_semantic_tokenizer/<suite>/<timestamp>_<name>/
```

The directory `experiments/runs/` contains no pre-redesign results. Historical runs are indexed in [the dated experiment archive](experiments/archive/pre_physiology_semantic_20260701/README.md). Analysis tools must receive an explicit archive path when reproducing historical evidence.

## 🧪 Implementation order

The required order is:

1. validate data, cache, and teacher contracts;
2. correct and instrument the quantizer;
3. train independent semantic and continuous-residual branches;
4. export IDs, posteriors, codebook embeddings, residuals, and masks;
5. evaluate frozen sequence coupling and whole-brain utility;
6. generate signature-ordered, marginal-controlled figures.

Each module must pass code-correctness and scientific-validity gates before the next expensive experiment begins.

## 🗂️ Historical evidence

Historical material is retained, not deleted:

- [Superseded theory and plans](docs/archive/pre_physiology_semantic_20260701/README.md)
- [Pre-redesign runs and comparison reports](experiments/archive/pre_physiology_semantic_20260701/README.md)
- [Architecture changelog](docs/architecture_changelog/INDEX.md)

_Last updated: 2026-07-01_
