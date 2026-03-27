# Dataset Registry And Validation Plan

This document defines the shared dataset registration interface that now lives in src/data/registry.py and the pre-adaptation validation plan that lives in src/data/validation.py.

The immediate goal is to stop hard-coding EEG+NIRS Single-Trial assumptions into every training entry point. The registry now provides a single place to resolve:

- canonical dataset id
- default data root
- supported modalities
- nominal EEG and fNIRS sampling rates
- synchronization strategy
- loader implementation status
- original documentation references

## Registered datasets

The current canonical ids are:

- eeg_fnirs_single_trial
- refed
- visual_cognitive_motivation
- simultaneous_eeg_nirs

Loader-ready datasets now include:

- eeg_fnirs_single_trial
- simultaneous_eeg_nirs

The remaining two datasets are still registered ahead of implementation so future adapters can plug into a stable interface instead of inventing new config keys.

## Shared config interface

All experiment configs are now normalized through one shared path:

- src/data/registry.py: load_experiment_config
- src/data/registry.py: normalize_experiment_config
- src/data/registry.py: normalize_data_config

Normalized data config guarantees:

- data.dataset is always a canonical dataset id
- data.data_root is always present
- data.root is preserved as a compatibility alias
- data.split is always materialized
- data.dataset_params always exists
- data.dataset_registry contains resolved metadata and documentation references

This shared interface is now used by:

- src/utils/logger.py
- experiments/scripts/train_downstream.py

The actual dataset object creation path is now also unified in:

- src/data/factory.py
- experiments/scripts/train_tokenizer.py
- experiments/scripts/train_shared_tokenizer.py
- experiments/scripts/train_downstream.py

Because train_tokenizer.py and train_shared_tokenizer.py already load configs through ExperimentLogger, they also inherit the shared registry interface automatically.

## Original documentation reviewed

The validation plan below was derived from the original dataset documents under data, not only from the project summary.

### EEG+NIRS Single-Trial

- Original source states EEG and NIRS triggers were sent simultaneously via parallel port.
- EEG uses BBCI cnt and mrk cell arrays with six sessions.
- NIRS also uses cnt and mrk cell arrays, but marker numbers differ from EEG and require modality-specific mapping.

### REFED

- Original README confirms EEG and fNIRS are stored per video.
- Labels are dynamic valence/arousal traces stored separately under annotations.
- Synchronization is annotation-aligned time series, not shared discrete trial markers.

### Visual Cognitive Motivation

- Original readme confirms EEG trigger channel DC9/DC09 and fNIRS Mark encode stimulus onset, offset, and response.
- EEG has both raw EDF and preprocessed epoched MAT data.
- fNIRS remains in raw per-part/per-probe CSV format.
- Correct multimodal synchronization therefore requires event reconstruction, not direct file-level pairing.

### Simultaneous EEG&NIRS

- Original MATLAB and BrainVision/NIRx descriptions confirm synchronized trigger delivery via parallel port.
- EEG and fNIRS task files are stored separately for n-back, DSR, and WG, with three sessions concatenated per task.
- EEG markers and fNIRS markers use different numeric codes and need dataset-specific mapping.
- Current adapter progress: src/data/simultaneous_eeg_nirs_dataset.py now exposes training-ready single-modality window datasets and task-dependent multimodal datasets through the shared factory in src/data/factory.py.
- Updated alignment conclusion for n-back and DSR: the correct first step is session-level alignment, not trial-level alignment. For subject VP001, n-back session labels match exactly across EEG and fNIRS with three stable offset blocks of 9 sessions each. DSR also aligns at session level after skipping one extra EEG session marker, producing three stable offset blocks of 6, 5, and 6 sessions. This supports a blockwise session-alignment strategy derived from the original session-folder organization in the dataset documentation.
- Updated usage conclusion: DSR is now explicitly deprecated in this repository and is excluded from training-ready loaders and future adaptation checks until a stable scientific use case is defined.
- Updated segmentation conclusion:
	- n-back uses trial-level segmentation for EEG-only loading, session-level segmentation for fNIRS-only loading, and session-level segmentation for multimodal loading.
	- WG uses trial-level segmentation for EEG, fNIRS, and multimodal loading.

## Validation strategy

Validation is split into four stages so the same framework can be reused before and after each loader adapter is implemented.

### 1. Pre-loader checks

- Verify configured data_root exists.
- Verify original documentation files referenced by the registry exist.
- Verify subject or record discovery works for the dataset-specific folder layout.

### 2. Loader smoke checks

- Open one representative EEG record and verify channel labels, sampling rate, shape, and finite values.
- Open one representative fNIRS record and verify channel labels, signal types, sampling rate, shape, and finite values.
- Confirm modality metadata matches the loaded arrays.

### 3. Post-loader correctness checks

- Build minimal train and eval splits.
- Sample several windows and verify deterministic tensor shape, non-empty labels, and no NaN or Inf after preprocessing.
- Verify channel filtering or signal-type selection does not silently change label count or window count.

### 4. Synchronization checks

The synchronization stage depends on the dataset’s declared sync_strategy.

#### shared_parallel_port_markers

Applies to:

- eeg_fnirs_single_trial
- simultaneous_eeg_nirs

Checks:

- per-session or per-task EEG/fNIRS marker counts match after mapping
- class labels match after applying modality-specific marker code mapping
- onset offsets are stable across the record
- inter-event interval drift stays within tolerance

#### continuous_annotation_alignment

Applies to:

- refed

Checks:

- EEG, fNIRS, and annotation durations agree per video
- video ids align across all three sources
- annotation resampling preserves coverage for every model window

#### cross_device_event_reconstruction

Applies to:

- visual_cognitive_motivation

Checks:

- reconstruct stimulus onset, stimulus offset, and response triplets from EEG and fNIRS independently
- compare trial order and timing across modalities
- join reconstructed trials with behavioral type labels and verify one-to-one mapping

## Recommended implementation order

1. Implement simultaneous_eeg_nirs first because it is structurally closest to the existing Single-Trial loader.
2. Implement refed next because it forces the registry to support continuous labels and per-record windows.
3. Implement visual_cognitive_motivation last because it requires cross-format event reconstruction.

## Test entry points

The code-level validation plans are exposed through:

- src/data/validation.py: build_dataset_validation_plan
- src/data/validation.py: build_all_validation_plans
- src/data/validation.py: render_validation_plan_markdown

These are intended to be consumed before each new dataset adapter is written, and then reused as the acceptance checklist for the implementation.

## Visual inspection tool

In addition to unit tests, the repository now includes a human-readable continuous-record inspection script:

- [experiments/scripts/signal_visualization/visualize_continuous_alignment.py](experiments/scripts/signal_visualization/visualize_continuous_alignment.py)

Purpose:

- print raw continuous EEG and fNIRS recordings on a shared aligned time axis
- print a local zoom view around one selected event
- show EEG and fNIRS event onsets together
- shade task windows after each onset
- export a JSON summary with onset-count comparison and residual timing statistics after alignment

Current scope:

- implemented for eeg_fnirs_single_trial
- implemented for simultaneous_eeg_nirs
- for simultaneous_eeg_nirs, the script now visualizes task segmentation directly:
	- n-back: session-level segmentation regions
	- WG: trial-level segmentation regions
	- DSR: intentionally unsupported because the task is deprecated

Example:

```bash
python experiments/scripts/signal_visualization/visualize_continuous_alignment.py \
	--config phase0plus/shared_labram_lag_warmstart_eeg_fnirs_30s_2s_cb512.yaml \
	--subject-id 1 \
	--session-idx 0 \
	--focus-trial-idx 0
```

Expected output:

- one PNG figure under logs/continuous_alignment/... showing event track plus stacked raw EEG and fNIRS traces
- one local PNG figure under logs/continuous_alignment/... showing a zoomed view around the selected event
- one summary.json file containing selected channels, sample rates, initial offset, residual event drift, and label agreement

Factory smoke-test status:

- unified unimodal Simultaneous WG dataset: subject VP001 smoke-tested successfully
- unified multimodal Simultaneous WG dataset: subject VP001 smoke-tested successfully with 60 aligned segments
- continuous alignment visualization for Single-Trial still passes

For future dataset adapters, the visual inspection stage is not optional: each newly adapted dataset should pass both the global continuous plot and the local event zoom inspection before it is considered loader-ready.
