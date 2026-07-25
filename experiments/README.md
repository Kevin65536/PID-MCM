# Experiment workspace

_Operational entrypoint for active configs, scripts, runs, and historical artifacts_

---

## 📋 Active namespace

The only active result namespace for new tokenizer work is:

```text
experiments/runs/physiology_semantic_tokenizer/
```

The E2-compatible physiology-semantic runtime is implemented. The proposed
Shared-Driver Semantic VQ generation is not yet implemented and must use a new
R-series config namespace; it may not silently change the semantics of E2
configs. Source/observation scripts remain compatibility baselines and must
write reruns to an explicitly named archive or compatibility root.

## 🗂️ Directory roles

| Path | Role |
| --- | --- |
| [`configs/physiology_semantic_tokenizer/`](configs/physiology_semantic_tokenizer/README.md) | Active target-architecture configurations |
| `scripts/` | Executable training and analysis entrypoints |
| [`runs/`](runs/README.md) | Active generated outputs only |
| [`archive/pre_physiology_semantic_20260701/`](archive/pre_physiology_semantic_20260701/README.md) | Pre-redesign runs and reports |
| `configs/archive/` | Retired configuration snapshots |
| `scripts/archive/` | Retired executable workflows |

## 🛡️ Rules

- Never write new outputs directly below `experiments/runs/`.
- Every target run uses a suite directory and an immutable run directory.
- Historical reruns must not reuse the active suite names.
- Analysis defaults must not recurse through `experiments/archive/`.
- Run outputs remain ignored by Git; only storage policy, manifests, and selected lightweight summaries are tracked intentionally.

_Last updated: 2026-07-25_
