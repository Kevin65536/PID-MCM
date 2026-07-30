# EEG–fNIRS data contract

_Active measured-data, alignment, mask, and cache rules; consolidated 2026-07-30_

This document is the active data entrypoint. Dataset-native formats, tasks,
units, and original-document locations remain in the reference catalog
[`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md); the full dated
implementation audit remains in
[`physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md`](physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md).

## Registered measured datasets

| Registry ID | Main task surface | EEG | fNIRS | Important boundary |
| --- | --- | --- | --- | --- |
| `eeg_fnirs_single_trial` | motor imagery; mental arithmetic | 200 Hz analysis view | released dual-wavelength intensity, canonical 10 Hz view | 29 subjects; use native trial/session identity |
| `refed` | continuous valence/arousal video response | released 64-channel EEG | released HbO/HbR | targets are time-aligned continuous sequences, not video-class labels |
| `visual_cognitive_motivation` | RR/RF/FF/FR | released EEG | released HbO/HbR | reject `unknown`; S06 Part1 remains excluded under current evidence |
| `simultaneous_eeg_nirs` | n-back; word generation; DSR | 28 scalp channels after the admitted EOG-auxiliary branch | released HbO/HbR | DSR labels are EEG-native Go/No-go; fNIRS is synchronized context |

`croce_local_cache` is derived teacher evidence, not a fifth measured dataset.
Every analysis must name the measured dataset, task, record/session, subject,
condition/event, branch, and window/patch identity it consumes.

Read the original dataset documentation listed in
[`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md) before changing a loader
or interpreting units, task labels, timing, or licensing.

## Canonical sample identity

The join key is composed from stable dataset-native identity, not array order:

```text
dataset_id
subject_id
record_id / session_id
task_id
condition or event_id
window_start / window_end
modality branch
```

Teacher targets, raw views, trajectories, labels, masks, geometry, split
registries, and exported tokens must round-trip this identity exactly. A join
that succeeds only because two arrays currently share an ordering is invalid.

## Signal branches

The loader distinguishes measurement provenance from model coordinates:

- `raw_*`: dataset-native/released measured arrays and their source metadata;
- `homer2_aligned_fnirs`: a consistent HbO/HbR modeling branch with explicit
  component and transform provenance;
- `simultaneous_eeg_eog_clean_v1`: 28 scalp EEG channels with HEOG/VEOG used
  as auxiliary detection inputs, never as model channels;
- `single_trial_line_clean_v4`: the admitted Single-Trial line-clean branch;
- teacher/trajectory sidecars: privileged or derived targets joined to a
  measured raw view, never substituted for that view.

The Single-Trial v2/v3 artifact-removal branches are historical. The v4
decision removed cached artifact detections as a validity authority. New runs
use real recorded support from `valid_mask`; artifact annotations may be
reported as diagnostics but cannot silently zero samples.

## fNIRS measurement coordinate

Native sources are not falsely described as one physical unit:

- Single-Trial enters from released optical intensities and requires the
  recorded optical-density/MBLL transformation to form HbO/HbR.
- REFED, Visual, and Simultaneous enter from released chromophore exports.
- Subsequent robust centering/scaling creates a dimensionless model
  coordinate and must retain the native source/transform record.

HbO and HbR roles remain explicit. A linear model-space normalization does not
make the upstream physical measurements identical. High-wavelength-only
Croce caches are historical derived supervision and are not the default
measured fNIRS input.

## Time and event alignment

- Window timestamps are expressed in a common record-relative coordinate with
  the dataset-native anchor preserved.
- EEG and fNIRS support must overlap the requested window; missing support is
  not filled and called observed data.
- DSR formal labels come from released EEG codes 16/32. The current event
  registry retains 8,980 Go/No-go windows from 25 admitted subjects and
  excludes VP005 for continuous clock drift.
- Visual timing follows the documented appearance-to-disappearance semantics;
  every-third-row heuristics are forbidden.
- REFED continuous targets use a `[2,T]` valence/arousal sequence and a
  coordinate-wise `target_valid_mask`. A padded value is not a measured
  target.

Offline full-window analysis and strict-cutoff future prediction are different
estimands. A full-window encoder may not be described as causal or prospective.

## Masks

Keep these concepts separate:

| Mask | Meaning |
| --- | --- |
| `valid_mask` | actual recorded signal support |
| channel/component mask | measured channel or HbO/HbR pair availability |
| `target_valid_mask` | observed target support |
| padding mask | batching or fixed-length padding |
| artifact/QC annotation | diagnostic metadata unless a frozen protocol explicitly promotes it |

Zero, missing, censored, excluded, and padded values are not interchangeable.
Every loss and metric must consume the mask that belongs to its tensor.

## Geometry

Channel labels and geometry sidecars are versioned inputs. Template/projection
coordinates can support adjacency and qualitative spatial structure, but they
cannot support exact distance or co-registration claims. Geometry missingness
must remain visible; copying or mirroring a channel to fill a model input is
not allowed in the primary shared benchmark.

## Splits and protected data

- Fit normalizers, adapters, target scalers, hyperparameters, and model
  selection only on the partition authorized by the owning protocol.
- Group by subject and by any record/trial/video dependency that could cross a
  split.
- Keep sample-random, within-subject, and strict cross-subject protocols
  separately labeled.
- R-series subjects 24–29 remain closed.
- Each comparison protocol controls its own protected boundary. Completed
  STA-Net and historical EFRM v1 evaluations do not authorize EFRM LODO v2;
  the v2 protected folds remain closed until its explicit unlock path passes.

The presence of an index file is not permission to dereference its protected
arrays.

## Cache contract

Every derived cache records:

- schema and branch version;
- source paths/identifiers and hashes;
- transformation and software identity;
- sample/join-key inventory;
- shapes, sampling rates, units/coordinates, channel roles, and masks;
- success, exclusion, and failure counts;
- creation time and atomic completion marker.

Raw data is immutable. Rebuildable caches may be removed after their manifest,
summary, and retained-result status are recorded. Never clean
`data/cache/physiology_semantic_clean_v1/` while the active EFRM protocol is
using it.

## Current audit state

The post-DSR unified audit traversed all 22,952 then-admitted windows and
confirmed finite loading and stable Simultaneous channel signatures. It also
preserved explicit warnings and blockers rather than turning a successful
loader pass into scientific validation. The exact counts remain a dated cache
snapshot; formal protocols must record fresh inventory and hashes.

The active data layer is ready for the implemented STA-Net and EFRM adapters.
That readiness establishes a software/data contract only. It does not qualify
a physical teacher, authorize SD-SVQ/VQ experiments, or validate physiological
coupling.

## Validation entrypoints

Representative checks:

```bash
.venv/bin/python -m pytest -q \
  tests/test_dataset_registry.py \
  tests/test_unified_physiology.py \
  tests/test_event_alignment.py \
  tests/test_fnirs_standardization.py \
  tests/test_channel_geometry.py \
  tests/test_channel_adjacency.py
```

The full real-data audit and visualization commands remain documented in the
dated audit. Their generated reports are evidence artifacts, not an additional
source of data-contract authority.
