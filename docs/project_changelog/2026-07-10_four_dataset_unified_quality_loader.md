# Four-dataset unified quality loader

Date: 2026-07-10

## Scope correction

The dataset-quality tooling now audits exactly four original datasets:

1. EEG+NIRS Single-Trial;
2. REFED;
3. Visual Cognitive Motivation;
4. Simultaneous EEG&NIRS.

`croce_local_cache` is retained in the legacy/training registry because existing
source/observation workflows consume it, but it is no longer counted or visualized
as another dataset. It is explicitly reported as Croce-2017-derived
source/observation supervision provenance.

## Implementation

- Added `src/data/unified_physiology.py`.
- Joined clean-cache fNIRS records to original MATLAB/EDF EEG records.
- Standardized EEG to 200 Hz and fNIRS to 10 Hz.
- Standardized numerical units to dimensionless full-record robust standard
  deviations while retaining native-unit metadata.
- Standardized fNIRS components to interleaved HbO/HbR names and roles.
- Added `canonical_task_label_v1` and `canonical_channel_geometry_v1` output.
- Used EEG and fNIRS event clocks separately when slicing aligned windows.
- Added an alignment-admission filter: continuous-drift/unstable records remain
  auditable in the event sidecar but do not enter unified training windows.
- Rebuilt Visual events from paired EDF DC9 stimulus onsets and fNIRS Mark=1;
  RR/RF/FF/FR is now the condition label, and Part2 starts at epoch 126.
- Replaced the untracked dataset-quality report implementation so all four
  datasets produce real waveforms, post-unification amplitude distributions,
  spectra, and geometry plots.

## Verification

The final report is stored under:

`experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_four_dataset_check_20260710/`

All four datasets passed the deterministic format/unit/component/sample-rate/
label/timing/geometry-schema checks. This loader result did not itself decide
E0; the later sign-calibrated adaptive SSM decision passes complete E0. The
loader audit still does not establish cross-dataset scientific equivalence.
