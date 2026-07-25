# Disable EEG artifact-mask authority

Date: 2026-07-25

## Decision

The unified four-dataset measurement contract no longer exposes detected EEG
artifact intervals as labels and no longer grants them invalidity authority.
This avoids treating unavailable artifact evidence in REFED and Visual as
equivalent to evidence-driven EOG or burst detections in Single-Trial and
Simultaneous.

The policy identifier is
`disabled_all_false_no_invalid_authority_v1`.

## Runtime contract

- EEG signal corrections remain unchanged, including Single-Trial line-noise
  removal, EOG regression and mask-gated high-frequency attenuation, plus the
  Simultaneous EOG regression.
- `artifact_mask.eeg` and `artifact_mask.fnirs` are compatibility fields and
  are always all false.
- `analysis_valid_mask` is identical to `valid_mask`; it represents only
  record-boundary/data-presence validity.
- The local-view adapter reads `valid_mask` directly, so a historical
  `analysis_valid_mask` cannot silently zero samples or invalidate tokenizer
  patches.
- Detected masks already stored in cleaning caches are retained only as
  historical audit provenance. The unified loader records their aggregate
  fraction in preprocessing state but never exposes the sample-level mask.

## Compatibility

No raw or cleaned signal cache needs to be rebuilt because signal values are
unchanged. Existing experiment outputs generated under artifact-gated validity
remain historical and must not be compared as if they used this policy.
Detached target sidecars may be reused when their value construction did not
consume artifact validity, but the downstream run must record the new policy.
