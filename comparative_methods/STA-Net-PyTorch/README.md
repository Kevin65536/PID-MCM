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

## Automated tuning and dual evaluation protocols

The tuning path uses task-metric checkpoint selection and five multi-fidelity
rungs: 2, 8, 20, 40, and 100 epochs. Optuna TPE proposes configurations and a
Hyperband pruner prevents weak trials from consuming the 100-epoch budget. At
every rung, the value reported to Optuna is the best validation checkpoint seen
through that epoch—not merely the last epoch—so pruning, TPE, trainer checkpoint
selection, and final model selection share one objective. The launcher runs
three trial lanes per GPU and shards the long Visual and REFED studies across
two workers while preserving the requested total trial count per task:

```bash
.venv/bin/python comparative_methods/STA-Net-PyTorch/launch_tuning.py \
  --study-id 20260719_sta_net_hpo_v1 --n-trials 12
```

Audit an existing tuning run without opening protected data:

```bash
.venv/bin/python comparative_methods/STA-Net-PyTorch/analyze_tuning.py \
  --run-root comparative_methods/STA-Net-PyTorch/runs/tuning/20260719_sta_net_hpo_v1_100ep
```

The analyzer reconstructs Optuna states, rung and per-epoch validation
trajectories, failure causes, budget use, and the distinction between the
Optuna objective winner, the epoch-100 endpoint, and the independently
reconstructed best historical validation checkpoint. It
writes machine-readable CSV/JSON, a Markdown audit, and SVG/300-DPI PNG figures
under the run's `analysis/` directory.

The final MI/WG development round uses the frozen
`configs/final_mi_wg_targeted.yaml` profile. It enqueues control-like, slow,
regularized, and slow-plus-regularized anchors, gives every trial at least 20
epochs before pruning, and assigns MI/WG to separate GPUs:

```bash
.venv/bin/python comparative_methods/STA-Net-PyTorch/launch_tuning.py \
  --study-id 20260724_sta_net_mi_wg_final_targeted_v1_100ep \
  --n-trials 16 --startup-trials 8 \
  --base-config comparative_methods/STA-Net-PyTorch/configs/final_mi_wg_targeted.yaml \
  --tasks motor_imagery wg
```

This profile is a final development selection procedure, not permission to
open protected manifests. The selected configuration must still be frozen
before the one-time protected outer-subject evaluation.

Training uses only the public development split. A winner may be frozen only
after it completes the 100-epoch rung:

```bash
.venv/bin/python comparative_methods/STA-Net-PyTorch/freeze_study.py \
  --run-root comparative_methods/STA-Net-PyTorch/runs/tuning/20260719_sta_net_hpo_v1 \
  --study-id 20260719_sta_net_hpo_v1 --task motor_imagery
```

`build_split_registry.py` creates separate public and protected manifests for
single-subject nested folds and 5x3 cross-subject nested folds. `train.py`
accepts only public manifests and rejects exposed protected indices.
`evaluate_protocol.py` is the only protected-fold consumer and requires both a
hash-pinned freeze manifest and the explicit `--unlock-protected-test` flag.

Run the separately labeled non-cross-subject protocol after its endpoint and
task configurations are frozen:

```bash
.venv/bin/python comparative_methods/STA-Net-PyTorch/launch_within_subject.py \
  --run-id 20260724_sta_net_within_subject_all_tasks_v1_100ep \
  --unlock-protected-test
```

This protocol permits training and test samples from the same subject while
keeping session, record, video, or semantic-trial dependency groups disjoint.
It trains all registered subject/fold pairs, evaluates each protected group
once, and writes fold-level, subject-level, source-aligned MI/MA/WG, and shared
within-subject summaries under `runs/within_subject/<run-id>/aggregate/`.
Pooled-window metrics remain diagnostics and do not replace the subject-level
primary endpoints.

Classification checkpoints maximize validation macro-F1; REFED checkpoints
minimize masked scaled RMSE. Accuracy, balanced accuracy, macro-F1 and Kappa
remain available for reporting. The composite STA-Net loss is an optimization
diagnostic, not a checkpoint-selection endpoint.

The 2026-07-27 formal rerun follows
`sources/20260727_no_artifact_mask_convergence_protocol.md`: the adapter reads
only record-support `valid_mask`, never artifact-gated
`analysis_valid_mask`, and formal training uses validation-driven learning-rate
reduction plus audited early convergence instead of a fixed 100-epoch cutoff.
