# Single-modal EEG public performance runner

This package is the shared executable path for BIOT, CBraMod, and REVE-base.
It consumes only the train and validation identities exposed by the
method-neutral strict-cross-subject registry.  It cannot accept a protected
manifest path and never discovers test indices from the registry.

The first performance contract is
[`configs/public_performance_v1.yaml`](configs/public_performance_v1.yaml).
All three methods receive the same task-specific panel of 16 measured EEG
channels, in the same frozen order.  Channels are selected and reordered by
identity; no channel is copied or padded.  BIOT therefore uses the official
PREST-16 checkpoint as a positional transfer to native electrodes.  This is
an explicit domain-adaptation deviation from the upstream PREST bipolar
montage and is not reported as an original-task reproduction.

REFED is disabled in v1 because the shared inventory intentionally retains
partial terminal windows while the current frozen encoders do not implement a
truthful time-padding mask.  The six classification tasks remain available.

Run a non-admissible connectivity test on GPU1 with:

```bash
.venv/bin/python -m comparative_methods.single_modal_eeg.run_public_performance \
  --config comparative_methods/single_modal_eeg/configs/public_performance_v1.yaml \
  --method biot --task motor_imagery --outer-fold 0 --seed 17 \
  --device cuda:1 --smoke \
  --output-dir comparative_methods/single_modal_eeg/runs/biot_mi_outer0_seed17_smoke
```

Omit `--smoke` only for a public-development performance run.  Such a run
extracts every public train/validation feature, fits normalization and the
linear head on training features only, selects the configured head on public
validation macro-F1, and retains metrics, predictions, resource use, and a
weights-only reload check.  No command in this package performs protected
evaluation.

Generate the complete public queue with:

```bash
.venv/bin/python -m comparative_methods.single_modal_eeg.build_public_job_matrix \
  --config comparative_methods/single_modal_eeg/configs/public_performance_v1.yaml \
  --device cuda:1 \
  --output-root comparative_methods/single_modal_eeg/runs/public_performance_v1 \
  --output comparative_methods/single_modal_eeg/runs/public_performance_v1/job_matrix.json
```

The matrix contains 270 serial jobs. To start the first real public-performance
cell directly, use the smoke command above without `--smoke` and choose a new
output directory under `runs/public_performance_v1/`. Feature caches are keyed
by method checkpoint, adapter, task panel, data-branch hashes, and public split;
they are reused across the three frozen seeds without fitting any target-wide
transform.
