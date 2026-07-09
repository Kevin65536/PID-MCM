# Unified Dataset Loading and Training Readiness Audit

_Created: 2026-07-09_

## Verdict

The current `data/cache/physiology_semantic_clean_v1` cache is a solid
fNIRS-side signal/provenance cache and a useful event-label sidecar, but it is
not yet a complete unified EEG-fNIRS training dataset.

It is ready for:

- fNIRS-only loader development;
- raw-native vs HOMER2-aligned fNIRS comparisons;
- event/label audit across the four datasets;
- teacher-free fNIRS reconstruction/VQ smoke tests.

It is not yet ready for:

- a single unified multimodal training loader;
- cross-modal EEG-fNIRS window sampling without dataset-specific join logic;
- physical-teacher-supervised training;
- protected split/formal training runs.

## Requirements for Unified Loading

A training-ready unified dataset layer must satisfy all of the following.

| Requirement | Why it is required |
| --- | --- |
| Stable record identity | Every signal record, event row, split row, and future export must join by the same canonical key. |
| Modality availability declaration | Loader must know whether EEG, fNIRS, or both are available for each record and branch. |
| Signal arrays with shape and channel contracts | Training code must know time axis, channel axis, sample rate, channel names, units, and branch semantics. |
| Event and label contract | Task labels, event granularity, continuous labels, and marker provenance must be explicit. |
| Cross-modal timing alignment | EEG/fNIRS offsets, drift, skipped markers, and unresolved alignments must be machine-readable. |
| Window sampling contract | Event-relative windows, continuous windows, context history, and invalid boundary masks must be deterministic. |
| Split/protected-test boundary | Subject-level train/validation/protected-test membership and hashes must be fixed before formal runs. |
| Artifact and quality masks | EEG artifacts, fNIRS bad channels, motion/rejection marks, and missing files must be propagated. |
| Teacher target availability | Physical teacher targets/masks must be present only when scientifically admitted. |
| Loader API tests | A common loader must construct batches with identical keys, dtypes, masks, and provenance across datasets. |

## Current Cache Status

### Satisfied

- `cache_manifest.json` exists and indexes `1267` fNIRS records.
- Every record has `.npz` plus `*.manifest.json`.
- Required fNIRS arrays exist for every record:
  - `native_input_fnirs`
  - `raw_native_fnirs`
  - `homer2_aligned_fnirs`
  - `time_s`
  - `native_channel_names`
  - `homer2_channel_names`
- All audited fNIRS arrays are finite; minimum finite fraction by dataset is `1.0`.
- Channel-name counts match array channel dimensions.
- Single-Trial `homer2_channel_names` have been corrected to `<spatial_pair>_HbO/<spatial_pair>_HbR`; low/high wavelength labels are inputs, not output suffixes.
- HOMER2-aligned branch provenance is explicit:
  - Single-Trial: OD + motion suppression + bandpass + MBLL.
  - REFED, Visual, Simultaneous: post-conversion motion suppression + bandpass only.
- `event_index/` exists with:
  - `30270` event rows;
  - `787` alignment reports;
  - zero label-sequence mismatches for paired marker streams.

### Partially Satisfied

| Area | Current state | Remaining issue |
| --- | --- | --- |
| Record identity | Signal records have `dataset_id`, `subject`, `record_id`; events have the same fields. | Keys are not yet canonical across all datasets. REFED events use `video_1`, while signal records use `video_1_hbo_hbr` / `video_1_absorbance_780_805_830`. Simultaneous events use `VP001`, while fNIRS records use `VP001-NIRS`. |
| Event granularity | Event index captures trials, session blocks, video segments, and fNIRS CSV marks. | A unified window sampler must respect event type; one `trial` abstraction is insufficient. |
| Alignment reports | Single-Trial fixed offsets and Simultaneous piecewise/drift cases are recorded. | Alignment is not yet converted into reusable sample transforms for EEG/fNIRS window extraction. |
| Labels | Labels are normalized enough to avoid false mismatches. | There is not yet a unified label namespace such as `task_family`, `task_name`, `label_name`, `label_role`. |
| Artifact metadata | Visual Mark rows include movement/removal flags; REFED records include reservation availability in HOMER2 compatibility. | Bad-channel/rejection masks are not yet applied or exposed as training masks. EEG artifact handling is not represented in the clean cache. |

### Not Satisfied

- EEG arrays are not present in `physiology_semantic_clean_v1`; this is currently fNIRS-only signal cache plus event sidecar.
- There is no unified multimodal loader API over this cache.
- There is no split manifest for train/validation/protected-test subjects.
- There are no split hashes or protected-test access guards.
- There is no materialized window index.
- There are no causal context masks for fixed-history tokenization.
- There are no physical teacher targets, uncertainty fields, or teacher-valid masks in this cache.
- REFED and Visual cannot be treated as complete HOMER2-clean multimodal records because raw optical inputs or paired EEG marker streams are missing from the local cache.

## Dataset-Specific Loading Implications

### EEG+NIRS Single-Trial

Status: closest to unified event-window training.

- Signal cache has 174 fNIRS session records.
- Event index has 3480 trial events and 174 alignment reports.
- EEG/fNIRS trial labels are semantically mapped.
- Alignment case is `stable_fixed_offset` for all sessions.

Required next step:

- Add EEG signal cache or a loader bridge to original EEG `.mat` files.
- Build event-relative paired windows using each modality's own timestamp.

### Simultaneous EEG&NIRS

Status: usable only with task-aware alignment.

- `wg` supports trial-level alignment.
- `nback` and `dsr` require session-block alignment, not trial-level pairing.
- Reports show piecewise offsets for most records and one continuous-drift case.
- DSR has one skipped-marker piecewise alignment case.

Required next step:

- Make the loader consume `alignment_reports.jsonl` and branch by event type:
  - `trial` for `wg`;
  - `session_block` for `nback`/`dsr`.
- Encode offset blocks/drift as alignment transforms rather than assuming one global offset.

### REFED

Status: segment-level multimodal candidate, not trial-marker aligned.

- Signal cache has 960 fNIRS branch records.
- Event index has 480 video-segment rows, each with continuous valence/arousal labels.
- Events align by shared video index, not by marker timestamps.

Required next step:

- Add canonical base record IDs so one event can link to both `hbo_hbr` and `absorbance` branches.
- Add EEG video signal cache or loader bridge.
- Treat valence/arousal as continuous label streams, not class labels.

### Visual Cognitive Motivation

Status: fNIRS event stream is indexed; paired EEG alignment remains unresolved.

- Signal cache has 55 paired Oxy/Deoxy fNIRS records.
- Event index has 23581 fNIRS Mark rows.
- Stimulus epochs map mostly to `RR/RF/FF/FR`; 28 stimulus-onset rows have invalid raw type values `00/01` and are normalized to `unknown`.

Required next step:

- Determine whether local EEG preprocessed files expose compatible epoch IDs or marker timestamps.
- Until then, treat this as fNIRS-only or label-only for unified loader smoke tests.

## Immediate Engineering Tasks

1. Add canonical join fields:
   - `canonical_subject_id`;
   - `base_record_id`;
   - `signal_branch`;
   - `cache_record_id`.

2. Regenerate signal and event manifests with those fields.

3. Add `CleanPhysiologyCacheIndex`:
   - reads `cache_manifest.json`, `event_manifest.json`, `events.jsonl`, and `alignment_reports.jsonl`;
   - validates join coverage;
   - exposes dataset-specific missingness explicitly.

4. Add `CleanPhysiologyWindowDataset`:
   - supports fNIRS-only first;
   - then EEG+fNIRS where EEG loader bridges exist;
   - uses event-relative windows and alignment reports;
   - returns masks for unavailable modalities, invalid windows, and unresolved alignments.

5. Add split manifest:
   - subject-level deterministic split;
   - protected-test lock;
   - split hash recorded in run manifests.

6. Add quality masks:
   - fNIRS bad/reserved channels;
   - Visual removal/body-movement marks;
   - EEG artifact status where available.

## Training Readiness Judgment

Current cache status: **partial P1 data-contract evidence, not unified training ready**.

Supported:

- fNIRS records are finite, versioned, and provenance-rich.
- Event labels and timing diagnostics are now centralized.
- The cache can support fNIRS-only and alignment-audit experiments.

Not supported yet:

- A single train loop consuming all four datasets as equivalent multimodal samples.
- Physical teacher supervision.
- Protected formal runs.
- Claims that Visual and REFED have fully resolved EEG/fNIRS event-time alignment.
