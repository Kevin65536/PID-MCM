# Single-Trial EEG artifact-cleaning candidate

Date: 2026-07-14

## Scope

Implemented an auditable Single-Trial EEG cleaning candidate without changing
the registered training default.  The controlled `cnt_artifact` recordings are
calibration controls, not a fifth dataset and not task samples.

## Implementation

- Added record-level `single_trial_eeg_artifact_clean_v2` preprocessing.
- Retained VEOG/HEOG for robust lagged regression before event windowing.
- Added adaptive consensus bad-channel detection, geometry-aware interpolation,
  high-frequency masks, and complete cleaning provenance.
- Added raw/clean branches and artifact, bad-channel, and analysis-valid masks to
  `UnifiedPhysiologyWindowDataset`.
- Added registry status fields and kept `raw_with_ocular_artifact` as default.
- Extended the four-dataset quality report with explicit EEG branch selection and
  artifact-mask summaries.
- Added a streaming all-subject audit with JSONL/CSV, controlled-artifact
  calibration, raw/clean PSD, preservation metrics, manifest, and admission YAML.

## Verification and decision

The final candidate audit covered 29 subjects and 174 task records.  It found no
sample/channel loss, reduced median EOG correlation from 0.531 to 0.028, and
preserved non-frontal alpha topology with median correlation 0.967.  A real
unified-loader raw/clean comparison preserved window count, event, label,
alignment, fNIRS values, geometry, and boundary masks.

The candidate remains **not admitted** because high-frequency muscle activity is
currently mask-only and has not passed a controlled sham/null correction test.
No formal cleaned training cache was materialized and the registry default was
not changed.
