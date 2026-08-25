# Single-modal EEG public performance runner

> **STATUS: ABANDONED — COMPATIBILITY-ONLY (frozen 2026 workflow).** The early
> shared public runner and its queue are retained for replay only. Do not use it for
> new BIOT, CBraMod, or REVE development; commands below are not a current entrypoint
> and do not authorize a protected or formal run.

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

Historical non-admissible connectivity replay (do not launch a new run):

```bash
.venv/bin/python -m comparative_methods.single_modal_eeg.run_public_performance \
  --config comparative_methods/single_modal_eeg/configs/public_performance_v1.yaml \
  --method biot --task motor_imagery --outer-fold 0 --seed 17 \
  --device cuda:1 --smoke \
  --output-dir comparative_methods/single_modal_eeg/runs/biot_mi_outer0_seed17_smoke
```

The former non-smoke public-development route is abandoned. No command in this
package performs protected evaluation.

Historical queue definition (do not launch):

```bash
.venv/bin/python -m comparative_methods.single_modal_eeg.build_public_job_matrix \
  --config comparative_methods/single_modal_eeg/configs/public_performance_v1.yaml \
  --device cuda:1 \
  --output-root comparative_methods/single_modal_eeg/runs/public_performance_v1 \
  --output comparative_methods/single_modal_eeg/runs/public_performance_v1/job_matrix.json
```

The former matrix contained 270 serial jobs; its first real public-performance
cell is not to be started. Feature-cache and split details remain only for
historical replay and audit.
