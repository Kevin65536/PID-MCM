# Clean EEG-fNIRS Cache Structure

_Created: 2026-07-08_

This cache is a generated local artifact and is not tracked by git.  The
canonical full-cache target is:

```text
data/cache/physiology_semantic_clean_v1/
  cache_manifest.json
  event_index/
    event_manifest.json
    events.jsonl
    alignment_reports.jsonl
  eeg_fnirs_single_trial/
    subject_01/
      session_00.npz
      session_00.manifest.json
  simultaneous_eeg_nirs/
    VP001-NIRS/
      cnt_dsr.npz
      cnt_dsr.manifest.json
  refed/
    1/
      video_1_hbo_hbr.npz
      video_1_hbo_hbr.manifest.json
      video_1_absorbance_780_805_830.npz
      video_1_absorbance_780_805_830.manifest.json
  visual_cognitive_motivation/
    S01/
      S01_Part1_Probe1.npz
      S01_Part1_Probe1.manifest.json
```

## Top-Level Manifest

`cache_manifest.json` is the index and reproducibility boundary.  It contains:

- `schema`: `clean_eeg_fnirs_cache_v1`
- `homer2_alignment_schema`: the HOMER2-alignment contract version
- `parameters`: dataset list, subject/record limits, sample cap, REFED absorbance flag
- `homer2_compatibility`: per-dataset available/missing HOMER2 inputs
- `records`: embedded record manifests for fast inspection
- `record_count`

## Record NPZ

Each `.npz` contains:

- `native_input_fnirs`: the raw exported fNIRS values in their native dataset coordinate
- `raw_native_fnirs`: full-record native measurement standardization output
- `homer2_aligned_fnirs`: best-effort HOMER2-aligned branch output
- `time_s`: sample timestamps
- `native_channel_names`
- `homer2_channel_names`

The channel semantics differ by branch.  For Single-Trial, native channels are
low/high wavelength channels, while HOMER2-aligned channels are HbO/HbR pairs
after OD and MBLL.  For post-conversion datasets, native and HOMER2 channel
names are intentionally the same because OD/MBLL cannot be replayed.

## Record Manifest

Each `*.manifest.json` contains:

- source file relative paths and sha256 hashes
- dataset id, subject, record id, sample rate
- native measurement contract and dataset metadata
- native and HOMER2 channel names
- `raw_native_contract`: array key, summary, standardization state, quality
- `homer2_aligned_contract`: array key, summary, alignment state, quality

The `homer2_aligned_contract.alignment_state` is the main audit field.  It
records `applied_steps`, `skipped_steps`, and `missing_inputs`, so downstream
analysis must not assume all datasets are equally HOMER2-complete.

## Event Index

The event index is a lightweight sidecar generated from original marker and
label files:

```bash
.venv/bin/python experiments/build_clean_event_index.py \
  --subjects-per-dataset 1000 \
  --records-per-subject 1000 \
  --output-dir data/cache/physiology_semantic_clean_v1/event_index \
  --overwrite
```

`events.jsonl` stores one canonical event per line with:

- `dataset_id`, `subject`, `record_id`, `event_index`
- `event_type`: `trial`, `session_block`, `video_segment_with_continuous_labels`, or `fnirs_csv_mark`
- `label` and optional `label_index`
- modality-specific timestamps: `eeg_time_ms`, `fnirs_time_ms`, `onset_ms`, `duration_ms`
- `alignment_role`
- dataset-specific metadata, including source files

`alignment_reports.jsonl` stores one timing report per record.  For datasets
with paired EEG/fNIRS marker streams, it records:

- number of EEG, fNIRS, and aligned events
- `alignment_case`: fixed offset, piecewise offset, skipped-marker piecewise offset, continuous drift, etc.
- `label_sequence_match`
- offset mean/std and linear drift slope
- offset blocks and skipped marker indices

Dataset-specific conventions:

- Single-Trial uses paired EEG/NIRS trial markers.  EEG and fNIRS have different recording starts, so event rows keep both timestamps and the offset.
- Simultaneous EEG&NIRS uses task-aware alignment: `wg` aligns trial markers, while `nback` and `dsr` align session-level markers because fNIRS MATLAB markers are session/block-level there.  DSR may skip one extra EEG session marker before blockwise alignment.
- REFED stores one video-segment event per subject/video and embeds the continuous valence/arousal label stream in event metadata.  Alignment is by shared segment index rather than marker time.
- Visual Cognitive Motivation stores fNIRS CSV Mark events (`stimulus_onset`, `stimulus_offset`, `participant_response`) and maps stimulus epochs to `RR/RF/FF/FR` labels from the subject xlsx type table when present.  The current local data do not expose a paired EEG marker stream in the same format, so EEG alignment remains unresolved in the report.

## Full Build Command

Use the full command below after the smoke path passes:

```bash
.venv/bin/python experiments/build_clean_eeg_fnirs_cache.py \
  --subjects-per-dataset 1000 \
  --records-per-subject 1000 \
  --include-refed-absorbance \
  --output-dir data/cache/physiology_semantic_clean_v1 \
  --overwrite
```

`data/` is ignored by git in this repository, so the generated full cache stays
local.  Commit only code, docs, and lightweight manifests that are intentionally
outside `data/`.
