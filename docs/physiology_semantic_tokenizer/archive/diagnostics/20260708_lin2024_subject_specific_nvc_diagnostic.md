# Lin 2024 subject-specific NVC diagnostic

_Croce-local diagnostic inspired by Lin et al. 2024; protected test closed_

---

## Decision

The current Croce-local E0 pilot cache does **not** support the stronger claim that a Lin-style subject-specific EEG-HRF model can extract a stable shared neural state from the paired EEG and fNIRS observations.

Subject-specific modeling remains a plausible direction, but this diagnostic does not rescue E0. The supported conclusion is narrower: the next teacher should include explicit subject/dataset measurement adapters and modality-private hemodynamic state, but shared-token supervision should not be promoted from the current EEG-derived HRF fit.

## What Lin 2024 contributes

Lin et al. propose a subject-specific EEG-fNIRS NVC pipeline:

1. Convert EEG trials into high-order time-frequency-channel tensors.
2. Use task-related tensor decomposition to extract reproducible task-related EEG temporal components.
3. Select active fNIRS channels with a GLM.
4. Fit a subject-specific double-gamma HRF so the EEG temporal component predicts HbO.
5. Compare predicted and true hemodynamic traces by leave-one-trial validation.

The relevant lesson for the current E0 stall is that fixed Croce-style dynamics may be too rigid if the main variance is subject-specific HRF shape or measurement mapping. Therefore this diagnostic tests whether subject-specific HRF fitting and a task-related EEG component materially improve the EEG-to-fNIRS shared-state path.

## Implementation boundary

This is a Lin-inspired diagnostic, not a direct reproduction.

- The current cache provides paired optical fNIRS channels, not Lin's HbO concentration traces.
- The Python implementation does not use Tensorlab CP decomposition. It approximates the task-related EEG component with 1D EEG components from band-average features, task-supervised PLS, and fNIRS-supervised PLS.
- The HRF family follows Lin's double-gamma parameterization with canonical and optimized modes.
- The protected subjects 24-29 remained unopened.

## Protocol

Data source:

`croce_validation/cache/croce_local/physiology_semantic_v2_e0_pilot/single_trial_mental_arithmetic`

Split:

- Train subjects: 1-18
- Validation subjects: 19-23
- Protected test subjects: 24-29, unopened

Windows:

- Baseline: event-relative -5 to 0 s
- Analysis: event-relative 0 to 20 s
- Task regressor: first 10 s of the analysis window

Three evaluation regimes were run:

1. `subject_specific_leave_one_event`: the Lin-aligned validation. For validation subjects 19-23, fit within the same subject and anchor on three events, then test the held-out event.
2. `subject_specific_fit_all`: a fit-quality diagnostic. For each validation subject and anchor, fit and evaluate on all four events. This is not generalization evidence; it shows whether the model family can recover a plausible subject/task trajectory at all.
3. `subject_held_out_group`: a stress-control only. Fit EEG component and HRF on subjects 1-18 and evaluate on subjects 19-23. This is expected to degrade when HRFs and measurement maps are subject-specific, so it is not the primary test of Lin's claim.

The main shared-state candidates were:

- `stimulus`: task regressor convolved with HRF.
- `band_average`: averaged EEG band-power envelope.
- `task_pls_eeg`: task-supervised EEG component, the closest lightweight approximation to Lin's task-related EEG component.
- `fnirs_pls_eeg`: modified EEG component supervised by the training fNIRS optical target.

Two non-shared fNIRS baselines were included:

- `fnirs_task_mean`: mean training fNIRS trajectory for the anchor.
- `fnirs_self_persistence`: one-step fNIRS persistence, a private/history reference rather than shared-state evidence.

## Results

| Split | Component | HRF | MSE | R2 | PCC | amplitude ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| subject-held-out | band_average | canonical | 0.713 | 0.012 | 0.104 | 0.102 |
| subject-held-out | stimulus | canonical | 0.714 | 0.010 | 0.119 | 0.064 |
| subject-held-out | fnirs_pls_eeg | canonical | 0.716 | 0.007 | 0.057 | 0.063 |
| subject-held-out | task_pls_eeg | canonical | 0.724 | -0.005 | -0.010 | 0.062 |
| subject-held-out | fNIRS self-persistence | none | 0.001 | 0.998 | 0.999 | 0.997 |
| subject-specific fit-all | band_average | optimized | 0.924 | 0.076 | 0.243 | 0.261 |
| subject-specific fit-all | stimulus | optimized | 0.950 | 0.050 | 0.190 | 0.212 |
| subject-specific fit-all | fnirs_pls_eeg | optimized | 0.949 | 0.051 | 0.188 | 0.225 |
| subject-specific leave-one-event | stimulus | optimized | 2.113 | -0.024 | 0.133 | 0.359 |
| subject-specific leave-one-event | task_pls_eeg | optimized | 2.189 | -0.053 | -0.012 | 0.282 |
| subject-specific leave-one-event | fnirs_pls_eeg | optimized | 2.217 | -0.152 | -0.005 | 0.393 |
| subject-specific leave-one-event | fNIRS self-persistence | none | 0.003 | 0.997 | 0.997 | 0.999 |

The optimized HRF did not materially improve either the group stress-control or the subject-specific leave-one-event result. Most optimized folds stayed near the canonical HRF shape, indicating that the current signal does not provide a stable subject-specific HRF estimate under this model class.

Some individual validation folds reached positive R2, but the effect was not stable. In subject-specific leave-one-event, the best stimulus-HRF folds reached about 0.20-0.36 R2, while the median fold was approximately zero and several folds were strongly negative. Even the in-sample subject-specific fit-all diagnostic remained weak for EEG-derived HRF paths, with R2 around `0.04-0.08`.

## Visual diagnostics

The revised run adds overlay figures for raw EEG, EEG-derived driver, true fNIRS, predicted fNIRS, and residuals. The example overlays show that the fitted model mostly recovers a low-amplitude smooth HRF template, while the observed fNIRS has larger late-window peaks and channel-specific drift. This directly matches the low amplitude ratios in the table.

The HRF parameter figures show that many optimized folds remain near the canonical `TTP=6 s`, `TTU=16 s`, and `c=6` values, with only scattered deviations. Therefore the negative result is not only a cross-subject issue; the subject-specific optimizer is often not finding a stable alternative HRF family from the current data.

Visual artifacts:

- `figures/trajectory_subject19_AF3AFz_event_000_fnirs_pls_eeg_optimized.svg`
- `figures/trajectory_subject19_AF3AFz_event_000_stimulus_optimized.svg`
- `figures/lin2024_hrf_parameter_distribution.svg`
- `figures/lin2024_hrf_curves.svg`
- `figure_data/trajectory_examples.csv`

## Interpretation

The diagnostic does not show that no neurovascular coupling exists. It shows that, in the current cached representation and split, the Lin-style route does not recover a subject-calibrated shared state strong enough to serve as E0 teacher evidence. The cross-subject stress-control is reported separately and should not be used as the primary reason to reject a subject-specific method.

The decisive comparison is not against zero alone. It is against the private/history fNIRS reference: fNIRS self-persistence remains near 0.997-0.998 R2, while EEG-derived HRF predictions stay near zero or negative R2. This means most predictable fNIRS structure in these windows is carried by fNIRS-private temporal continuity, not by the tested EEG-derived shared driver.

The modified fNIRS-supervised EEG component also did not help. Its subject-held-out R2 was about `0.007`, subject-specific fit-all R2 was about `0.051`, and subject-specific leave-one-event R2 was about `-0.152`. This argues against the simple fix "learn a better one-dimensional EEG component, then fit subject-specific HRF" for the current E0 blocker.

## Computing a Croce-like shared trajectory

Croce 2017 estimates a latent neural activity trajectory `r(t)` jointly from EEG and fNIRS through a state-space model. The Lin-style diagnostic does not estimate the same physical state. It yields a one-dimensional EEG-derived hemodynamic driver:

\[
u_{s,a,e}(t)=w^\top \phi_{EEG,s,a,e}(t),
\]

where `phi_EEG` is the EEG band-power feature trajectory and `w` is the fitted task/fNIRS component projection for a subject `s`, anchor `a`, and event `e`. The fitted HRF then gives:

\[
\hat r_{s,a,e}(t)=zscore((h_{\theta_{s,a}} * u_{s,a,e})(t)).
\]

The predicted fNIRS channels are an affine observation of this trajectory:

\[
\hat F_{s,a,e,c}(t)=\alpha_{s,a,c}\hat r_{s,a,e}(t)+\beta_{s,a,c}.
\]

In the current run, `figure_data/trajectory_examples.csv` stores this trajectory as `shared_driver_z`. This is the closest output to a Croce-like `r(t)` in the tested Lin-style model. It should be interpreted as an EEG-derived candidate driver, not as an admitted physical teacher state, because it does not beat the fNIRS-private/history controls.

## Consequence for E0

Status: **E0 remains blocked.**

This run supports revising the teacher family, not promoting physical-state-supervised tokenizer training. The next candidate should separate:

- a narrow admitted shared driver;
- a subject/dataset measurement adapter;
- a delayed hemodynamic transition state;
- modality-private fNIRS observation and persistence state.

Shared token identity should be supervised only by components that beat time-shift, pairing-permutation, and private-history controls under the declared validation regime. Subject-specific HRF fitting alone is insufficient evidence.

## Artifacts

- Revised run with visualizations: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_113000_lin2024_subject_specific_nvc_v1/`
- Original non-visual run: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_110952_lin2024_subject_specific_nvc_v1/`
- Script: `experiments/evaluate_lin2024_subject_specific_nvc.py`
- Local config copy: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_113000_lin2024_subject_specific_nvc_v1/config.yaml`
- Summary: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_113000_lin2024_subject_specific_nvc_v1/summary.md`
- Metrics: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_113000_lin2024_subject_specific_nvc_v1/metrics.csv`
- Per-anchor/fold metrics: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_113000_lin2024_subject_specific_nvc_v1/anchor_metrics.csv`

**Status:** diagnostic complete; protected test closed; does not support E0 pass or physical-state-supervised tokenizer training.
