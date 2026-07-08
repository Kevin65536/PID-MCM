# Lin 2024 Simultaneous EEG&NIRS raw TRTD diagnostic

## Purpose

This diagnostic repeats the raw-record Lin-style TRTD/NVC test on one additional dataset: Simultaneous EEG&NIRS subject `VP001`, task `wg`, target class `WG`.

The goal is not to open a new E0 gate. It checks whether the weak fNIRS prediction observed in the Single-Trial raw session also appears when fNIRS is already provided as oxy/deoxy concentration rather than optical wavelength intensity.

## Data

- Dataset: `data/Simultaneous EEG&NIRS`
- Subject/task: `VP001`, `wg`
- Raw files:
  - `data/Simultaneous EEG&NIRS/VP001-EEG/cnt_wg.mat`
  - `data/Simultaneous EEG&NIRS/VP001-EEG/mrk_wg.mat`
  - `data/Simultaneous EEG&NIRS/VP001-NIRS/cnt_wg.mat`
  - `data/Simultaneous EEG&NIRS/VP001-NIRS/mrk_wg.mat`
- EEG: 30 channels, 200 Hz, `uV`
- fNIRS: 36 oxy + 36 deoxy channels, 10 Hz, `mmol/L`
- Events: 60 aligned class labels, 30 `WG` and 30 `BL`

The EEG and fNIRS marker label sequences match exactly. Marker timestamps are offset in three stable 20-event blocks:

| Block | Events | Mean fNIRS-EEG offset |
| --- | ---: | ---: |
| 1 | 20 | 17.226 s |
| 2 | 20 | 69.675 s |
| 3 | 20 | 120.608 s |

Epoch extraction therefore used each modality's own marker time for the same event index instead of forcing a single global timestamp offset.

## Method

The script follows the same diagnostic pattern as the preceding Single-Trial raw-session run:

1. Read continuous original `cnt/mrk` MATLAB files.
2. Extract 20 s epochs from `-5` to `15` s around `WG` onset.
3. Filter EEG to `1-40 Hz`.
4. Build log-power EEG time-frequency tensors with approximately `0.5 Hz` frequency resolution.
5. Fit shared spatial/frequency CP factors with trial-specific temporal factors.
6. Apply TRCA to obtain one task-related EEG temporal component per trial.
7. Filter oxy/deoxy fNIRS to `0.01-0.2 Hz`.
8. Select active oxy channels by canonical stimulus-HRF GLM.
9. Fit canonical and optimized double-gamma HRF mappings from EEG component to mean selected-channel oxy.
10. Evaluate leave-one-trial and in-sample upper-bound fits.

Dataset limitations remain important:

- The task is word generation, not Lin 2024 finger tapping.
- fNIRS is already provided as oxy/deoxy; no HOMER reprocessing or short-distance regression is available here.
- No BSS-CCA muscle-artifact removal was applied.
- The result is a subject/task diagnostic, not a population estimate.

## Results

Selected active oxy channels were `C5h`, `CCP3`, and `C4h`.

| Model | Validation | HRF | R2 | PCC | MSE | Amplitude ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| TRTD | in-sample upper bound | optimized | 0.003606 | 0.060050 | 0.996394 | 0.060050 |
| TRTD | leave-one-trial | optimized | -0.615709 | 0.150430 | 1.054345 | 0.063269 |
| trial mean | leave-one-trial | none | -0.554704 | 0.311964 | 1.012840 | 0.412408 |
| self-persistence | leave-one-trial | none | 0.991695 | 0.995936 | 0.004322 | 0.997021 |

The optimized HRF stayed at the canonical parameter set for this run. This should not be read as successful physiological identification; the fitted EEG-driven output still had only about `6%` of the observed fNIRS amplitude and did not recover trial-specific oxy trajectories.

## Visual Evidence

The run includes:

- `figures/simultaneous_waveform_overlay.svg`: EEG traces, TRTD component, HRF driver, observed oxy/deoxy, recovered oxy, and residual for one held-out trial.
- `figures/loo_trial_heatmap.svg`: all leave-one-trial observed oxy, predicted oxy, and residual heatmaps.
- `figures/hrf_parameter_folds.svg`: foldwise HRF time-to-peak, time-to-undershoot, undershoot ratio, and amplitude ratio.
- `figures/active_channel_scores.svg`: GLM active-channel scores.
- `figures/trtd_shared_factors.svg`: shared EEG spatial/frequency tensor factors.

The heatmap reproduces the same failure mode as the Single-Trial raw run: predicted fNIRS is compressed near a weak smooth trajectory while the observed fNIRS trial structure remains in the residual.

## Interpretation

This second raw-record check strengthens the current E0-D4 conclusion. The weak result is not explained only by Single-Trial optical-to-HbO conversion. Even when using Simultaneous EEG&NIRS data with provided oxy/deoxy concentration in `mmol/L`, a one-dimensional Lin-style EEG TRTD component plus subject/task-specific HRF does not recover a useful shared fNIRS trajectory.

The result supports the following design stance:

- keep subject/task-specific hemodynamic adapters as a necessary modeling layer;
- do not promote a Lin-style EEG-derived shared teacher as the fNIRS semantic state;
- preserve fNIRS private slow trajectory, baseline, and vascular-history branches;
- treat EEG-to-fNIRS NVC as a narrow auxiliary diagnostic unless a later model demonstrates non-circular improvement over self-history baselines.

## Artifacts

- Script: `experiments/evaluate_lin2024_simultaneous_raw_trtd.py`
- Run: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_170809_lin2024_simultaneous_raw_trtd_vp001_wg/`
- Summary: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_170809_lin2024_simultaneous_raw_trtd_vp001_wg/summary.md`
- Metrics: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_170809_lin2024_simultaneous_raw_trtd_vp001_wg/metrics.csv`
- Fold metrics: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_170809_lin2024_simultaneous_raw_trtd_vp001_wg/fold_metrics.csv`
