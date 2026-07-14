# Single-Trial EEG artifact-cleaning v3 admission

Date: 2026-07-14

## Scope

Finalized the Single-Trial EEG correction after interpreting `cnt_artifact` as
five controlled recordings (EOG, EMG, eye blinking, teeth clenching, and mouth
opening), not as task data or a clean target.

## Implementation

- Added `single_trial_eeg_artifact_clean_v3`, retaining raw and historical v2.
- Attenuated only the 30–45 Hz component inside adaptively detected burst masks,
  with a tapered boundary and persistent artifact masks.
- Validated EMG, teeth-clenching, and mouth-opening corrections against equal-mask
  circular-shift sham corrections at subject level.
- Materialized a provenance-checked 174-record cache with join-key and source-file
  signature validation.
- Switched the registry, unified loader, and quality report defaults to v3.

## Verification and decision

The audit covered 29 subjects and 174 task records without sample or channel
loss. Median EEG/EOG correlation fell from 0.531 to 0.029; median non-frontal
alpha-topology correlation was 0.967 with no negative records. All admission
gates passed, including controlled-artifact target-versus-sham checks.

The v3 branch is admitted as the default conservative preprocessing branch. This
does not claim complete artifact removal and does not grant scientific admission
to a physical teacher or downstream model.
