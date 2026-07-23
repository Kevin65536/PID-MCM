# Single-Trial EEG v4: 50 Hz removal without dataset-specific bad-channel masks

## Decision

The unified four-dataset pipeline no longer exposes a dataset-specific
bad-channel interpretation for the EEG+NIRS Single-Trial dataset.  The v4
branch improves the signal itself where a common correction is supported, then
returns an all-false EEG `bad_channel_mask` like the other datasets.

The historical v2/v3 branches remain named and selectable.  The default branch
is now `single_trial_eeg_artifact_clean_v4`.

## Processing change

- Apply a zero-phase 50 Hz IIR notch (`Q=30`) before the existing 1–45 Hz
  zero-phase Butterworth passband.
- Retain the admitted EOG regression and mask-gated 30–45 Hz burst attenuation.
- Disable bad-channel rejection and whole-channel spatial interpolation.
- Retain channel-quality scores as audit provenance only; they do not produce a
  mask or alter a complete channel.
- Record per-channel 48–52 Hz / 1–80 Hz ratios before and after line-noise
  removal.

## Versioned artifacts

- Signal branch: `single_trial_eeg_artifact_clean_v4`
- Cache schema: `single_trial_eeg_artifact_cache_v4`
- Cache root:
  `data/cache/physiology_semantic_clean_v1/eeg_artifact_clean_v4/`
- Audit:
  `experiments/runs/physiology_semantic_tokenizer/data_quality_audit/single_trial_eeg_artifact_v4/full_29_subject_line_clean_no_bad_mask_20260723/`

## Full-data validation

The cache was rebuilt from all 29 subjects and all six task sessions:

- 174 records and 20,917,720 canonical EEG samples;
- 30 EEG channels retained in every record;
- zero non-finite cached values;
- zero true values across all EEG bad-channel masks;
- zero whole-channel interpolation operations;
- 48–52 Hz ratio reduced in all 174 records;
- record-level median ratio: 1.902% before, 0.010% after;
- maximum record-level median after cleaning: 0.161%;
- all admission gates passed.

`subject_05/session_00`, whose 30 channels were jointly dominated by 50 Hz and
therefore escaped the former within-record outlier score, fell from a median
99.995% line-noise ratio to 0.077%.

## Downstream consequence

Derived teacher sidecars, checkpoints, or reports admitted under the v3
bad-channel contract remain historical artifacts.  They must be rebuilt before
being used as evidence under the v4 measured-data contract.
