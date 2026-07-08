# Clean EEG-fNIRS Cache Structure

_Created: 2026-07-08_

This cache is a generated local artifact and is not tracked by git.  The
canonical full-cache target is:

```text
data/cache/physiology_semantic_clean_v1/
  cache_manifest.json
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
