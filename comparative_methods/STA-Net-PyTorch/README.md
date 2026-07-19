# STA-Net PyTorch comparison project

This directory is the isolated home of the project's independent PyTorch
STA-Net reimplementation. It contains the FGSA/EGTA model, unified-loader
adapter, classification and regression task variants, launch configuration,
tests, provenance notes, and every generated STA-Net artifact. It does not use
TensorFlow and must not write into the project's own `experiments/runs` tree.

The untouched upstream checkout remains in `../STA-Net/`. The reimplementation
is a project-adapted comparative method, not a claim of source-level numerical
equivalence.

## Layout

- `sta_net_pytorch/`: model and unified-loader adapter
- `smoke.py`: one-step correctness runner
- `train.py`: record-grouped, subject-split training/validation runner
- `queue_worker.py`: sequential task queue for one physical GPU
- `launch_all.py`: two detached GPU queues covering all seven tasks
- `visualize_results.py`: validation-only reproduction report and figure generator
- `configs/`: smoke and training protocols
- `tests/`: method-local unit tests
- `sources/`: paper/code feasibility and provenance notes
- `runs/`: smoke and formal-development training artifacts

Launch all tasks from the repository root:

```bash
.venv/bin/python comparative_methods/STA-Net-PyTorch/launch_all.py
```

The current training protocol creates deterministic subject-grouped
train/validation/reserved-test partitions. Reserved-test signals are not loaded.
REFED target standardization is fit on training subjects only. These runs are
development training runs until the shared cross-method protocol is frozen.
The optimized protocol keeps one active task per GPU, uses record-grouped
batches to preserve the unified loader's record cache, and emits validation
metrics plus latest/best checkpoints after every epoch. Each task manifest pins
the trainer, model, adapter, and resolved configuration hashes.

## Reproduction report

After a task or an all-task suite has completed, generate a self-contained
validation report from its best checkpoint:

```bash
.venv/bin/python comparative_methods/STA-Net-PyTorch/visualize_results.py \
  --run-root comparative_methods/STA-Net-PyTorch/runs/training/20260719_sta_net_all_tasks_v4_optimized_frozen \
  --device cuda:0
```

The command reconstructs exactly the validation indices recorded in each
`split_manifest.json`; it does not load the reserved test indices. Incomplete
runs are rejected by default. `--allow-incomplete` is available only for
previewing the report layout and must not be used as performance evidence.

Each task report contains the raw validation predictions, a machine-readable
metric summary, a flat metric table, a Markdown report, training curves, and
STA-Net lag-attention/fusion diagnostics. Classification reports additionally
contain count and row-normalized confusion matrices, per-class precision,
recall and F1, ROC/PR curves, and a reliability diagram. REFED regression
reports contain native-coordinate MAE/RMSE/R2/Pearson/CCC/bias, prediction and
residual plots, and masked target-versus-prediction sequences. Figures are
written as editable SVG and 300-DPI PNG; a suite overview compares all completed
tasks without pooling incompatible endpoints.
