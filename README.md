# EEG–fNIRS physiology-semantic tokenization

_Repository status and contributor entrypoint, 2026-07-25_

---

## 📋 Current status

The E2-compatible physiology-semantic runtime is implemented and its fixed
`K=128` quantizers pass the registered software-health gate. E2 found no
semantic-endpoint gain from the weak multi-entry physical-teacher objectives.
The next, still planned, architecture is Shared-Driver Semantic VQ: raw-only
modality-specific full-window encoders, independent `K=128,D=64` codebooks, and
full joint-driver-proxy trajectory reconstruction as the primary semantic
objective. Frozen bidirectional tokens first support offline delayed-association
tests; future raw-fNIRS prediction requires a separate strict-cutoff evaluation.

Start with the [documentation authority map](docs/README.md). Do not use archived source/observation plans as implementation instructions.

## 🧭 Authority map

| Need | Document |
| --- | --- |
| Design entrypoint | [Physiology-semantic archive](docs/physiology_semantic_tokenizer/README.md) |
| Target architecture and tensors | [Target architecture](docs/physiology_semantic_tokenizer/02_TARGET_ARCHITECTURE.md) |
| Theory and claim limits | [Theoretical foundations](docs/physiology_semantic_tokenizer/03_THEORETICAL_FOUNDATIONS.md) |
| Architecture-return synthesis | [Method lessons](docs/physiology_semantic_tokenizer/12_ARCHITECTURE_RETURN_AND_METHOD_LESSONS.md) |
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
│   ├── physiology_semantic_tokenizer/ # Active E0–E2 configs; R-series namespace planned
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

The active launcher supports the implemented physiology-semantic runtime:

```bash
bash experiments/scripts/launch_training_nohup.sh \
  --task physiology-semantic-tokenizer
```

The proposed Shared-Driver VQ generation must use new `r0_...`–`r7_...`
configs and may not silently reuse E2 semantics. Source/observation entrypoints
remain isolated under the
[dated script archive](experiments/scripts/archive/pre_physiology_semantic_20260701/README.md).

## 📦 Result policy

New outputs must use:

```text
experiments/runs/physiology_semantic_tokenizer/<suite>/<timestamp>_<name>/
```

The directory `experiments/runs/` contains no pre-redesign results. Historical runs are indexed in [the dated experiment archive](experiments/archive/pre_physiology_semantic_20260701/README.md). Analysis tools must receive an explicit archive path when reproducing historical evidence.

## 🧪 Implementation order

The required order is:

1. freeze E2 evidence and the new sample/mask contract;
2. build full-trajectory R1-D and population-frozen R1-P teacher sidecars, then
   revalidate the R1-P teacher panel;
3. test continuous EEG-only and fNIRS-only observability;
4. only then train independent fixed-K128 semantic quantizers;
5. export IDs, posteriors, embeddings, continuous latents, and driver signatures;
6. after R5, choose R6A offline association and/or the independent R6B
   completed-window cutoff test; reserve independent confirmation for R7.

Each module must pass code-correctness and scientific-validity gates before the next expensive experiment begins.

## 🗂️ Historical evidence

Historical material is retained, not deleted:

- [Superseded theory and plans](docs/archive/pre_physiology_semantic_20260701/README.md)
- [Pre-redesign runs and comparison reports](experiments/archive/pre_physiology_semantic_20260701/README.md)
- [Architecture changelog](docs/architecture_changelog/INDEX.md)
- [Project operations changelog](docs/project_changelog/INDEX.md)

_Last updated: 2026-07-25_
