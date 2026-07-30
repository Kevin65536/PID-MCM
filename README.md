# EEG–fNIRS physiology-semantic tokenization

_Project entrypoint; status snapshot 2026-07-30_

The main tokenizer line is **stopped at its scientific gate**. E0–E2 are
complete; the R-series prerequisite tests did not qualify the shared-driver
target. `promotion_eligible=false`, `next_action=do_not_enter_r2_p`, and
subjects 24–29 remain closed. R2-P, R3–R7, and a new SD-SVQ/VQ generation are
not authorized.

Comparison work continues independently. STA-Net's formal five-fold benchmark
is complete. EFRM LODO v2 is running its fourth Stage-A target-excluded
selection job; its protected evaluation remains closed. UMAP still needs a new
formal rerun.

Start with the [full experiment plan](docs/EXPERIMENT_PLAN.md) and
[documentation map](docs/README.md).

## Main entrypoints

| Need | Document |
| --- | --- |
| What is complete, running, next, or blocked? | [Experiment plan](docs/EXPERIMENT_PLAN.md) |
| What does the evidence permit us to claim? | [Method rationale](docs/METHOD_RATIONALE.md) |
| What data/mask/split contract is active? | [Data contract](docs/DATA_CONTRACT.md) |
| What code is currently runnable? | [Architecture](docs/ARCHITECTURE.md) |
| What results were retained? | [Results index](experiments/RESULTS_INDEX.md) |
| How are comparisons run and admitted? | [Comparison protocol](docs/comparisons/PROTOCOL.md) |
| How should a frozen tokenizer be analyzed? | [Token Physiology Atlas](docs/analysis/TOKEN_PHYSIOLOGY_ATLAS.md) |
| How should code and experiments be changed? | [Contributor guide](CONTRIBUTING.md) |

## Repository layout

```text
src/                    reusable data, inference, tokenizer, and analysis code
tests/                  active contract, scientific-gate, and regression tests
experiments/            configs, executable workflows, active runs, archives
comparative_methods/    isolated STA-Net, EFRM, and UMAP implementations
croce_validation/       physical-model validation and expensive legacy cache
docs/                   active contracts, status, evidence, history, literature
data/                   immutable measured data and derived caches
```

Generated payloads are ignored by Git. Active tools do not recursively search
archives, and comparison packages write only to their own run roots.

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

That command is not permission to start R2-P or a new VQ generation. Follow the
frozen experiment gate and do not touch paths or hashes protected by the R1-P
prevalidation seal.

Historical negative and failed results are evidence, not clutter. Large
rebuildable arrays/checkpoints may be pruned only according to
[`experiments/RESULTS_INDEX.md`](experiments/RESULTS_INDEX.md), and no directory
used by a live process may be cleaned.
