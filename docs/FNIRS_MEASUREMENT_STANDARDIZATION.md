# fNIRS Measurement Standardization Contract

## Scope

This contract standardizes the numerical input coordinate used before Croce
2017 testing. It does not rename or reinterpret any Croce state variable and it
does not claim that voltage, absorbance, and chromophore concentration share a
physical unit.

The implementation is `src/data/fnirs_standardization.py`; the cross-dataset
audit entry point is `experiments/audit_fnirs_measurement_standardization.py`.

## Native dataset contracts

| Dataset | Default signal | Measurement family | Native unit status |
|---|---|---|---|
| EEG+NIRS Single-Trial | lowWL 760 nm / highWL 850 nm | optical intensity | `V`, recorded in MAT metadata |
| REFED | HbO/HbR; Abs780/805/830 audited separately | chromophore export and absorbance | unit is not declared in the distributed README/MAT files |
| Visual Cognitive Motivation | ETG-7100 Oxy/Deoxy export | chromophore export | unit is not declared in the CSV export |
| Simultaneous EEG&NIRS | Oxy/Deoxy | chromophore concentration | `mmol/L`, recorded in MAT metadata |

Unknown units remain `unreported_*`. They must not be guessed from magnitude.
REFED signal types must be selected explicitly; six signal types must never be
flattened and treated as if they shared one unit.

## Canonical numerical coordinate

For each full continuous record and channel:

1. retain the native measurement family and unit in the contract;
2. interpolate isolated non-finite samples and record their counts;
3. estimate a robust linear drift from block medians over the full record;
4. absorb the remaining channel median into that reversible baseline;
5. divide the residual by its channel MAD-to-standard-deviation scale, with IQR
   and ordinary standard deviation only as fallbacks;
6. crop only after this transformation;
7. retain intercept, slope, scale, sample rate, repair counts, and contract in a
   versioned standardization state.

The output is dimensionless and preserves native signal semantics. It solves
numerical unit, offset, linear drift, and channel-scale differences without
claiming physical equivalence between measurement families.

The transform is reversible for a full record or crop when the stored state and
crop start sample are supplied. The active loaders skip their old session/window
z-score when this record-level contract is enabled, preventing a second,
crop-dependent coordinate change.

## Drift boundary

Only offset and robust linear drift are removed automatically. Nonlinear drift,
motion artifacts, superficial physiology, and task-locked hemodynamics overlap
in frequency and cannot be separated safely by a generic high-pass filter.
The audit therefore reports block-median range and residual drift metrics. Large
values are a data-quality gate, not permission to erase additional low-frequency
content automatically.

## 2026-07-06 all-subject audit

The audit inspected one full record for every discoverable subject: 29
Single-Trial records, 32 REFED subjects in both HbO/HbR and Abs spaces (64
records), 15 Visual subjects with available fNIRS files, and 26 Simultaneous
records. No protected physiology-semantic test data were opened.

| Dataset | Median native channel SD | Median canonical channel SD | Median residual linear drift (SD/min) |
|---|---:|---:|---:|
| Single-Trial | 0.00884087 V | 1.01219 | 0.000563 |
| REFED | 0.0335593 mixed native scale | 1.00452 | 0.000675 |
| Visual | 0.249943 unreported export scale | 1.04127 | 0.000007 |
| Simultaneous | 0.00181334 mmol/L | 1.02044 | 0.000038 |

The detailed generated artifacts are under
`experiments/runs/fnirs_measurement_audit/20260706_all_subjects_v2/`.

## Croce testing rule

Croce model parameters and state meanings remain unchanged. The solver receives
the full-record standardized fNIRS coordinate and must not fit another fNIRS
mean/standard deviation on a selected event or crop. The run manifest stores the
standardization state and marks the fNIRS arrays as prestandardized. EEG retains
its existing solver-side normalization.

