# Pre-physiology-semantic experiment scripts

_Frozen training, suite, audit, export, and downstream entrypoints, 2026-07-02_

---

## 🗂️ Status

These scripts reproduce the source/observation and tokenizer-coupling lineages that preceded the active redesign. They are excluded from the active script root so that searches for `train_*`, `launch_*`, or coupling analyses do not suggest them as current implementation templates.

The scripts use the dated compatibility package and dated config archive. They may reference archived checkpoints and result schemas. Outputs from any deliberate rerun are historical evidence and must remain outside `experiments/runs/physiology_semantic_tokenizer/`.

## ⚠️ Execution

The compatibility launcher is [`launch_legacy_training_nohup.sh`](launch_legacy_training_nohup.sh). Use it only for an explicitly scoped historical reproduction. Suite launchers are frozen research records; inspect their checkpoint and output paths before execution.

The active launcher at `experiments/scripts/launch_training_nohup.sh` never dispatches these tasks.

_Last updated: 2026-07-02_
