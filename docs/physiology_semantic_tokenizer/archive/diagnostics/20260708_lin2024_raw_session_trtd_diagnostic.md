# Lin 2024 raw-session TRTD diagnostic

_Single-subject continuous-record upper-bound probe; protected cache not used_

---

## Decision

A closer Lin-style implementation on one continuous raw session still does **not** support using the tested EEG-derived HRF trajectory as a reliable shared neural state for the current data.

This run is more favorable than the previous cache-window diagnostic because it reads the original continuous MATLAB `cnt/mrk` records, fits within one subject and one session, and includes an in-sample upper-bound diagnostic. Even under this favorable setting, the TRTD+subject-specific HRF path explains almost none of the active-channel fNIRS/HbO trajectory.

## Paper Procedure Checked

The Lin 2024 workflow was reread and translated into the closest local equivalent:

1. EEG continuous record filtered at 1-40 Hz.
2. Event epochs extracted as 20 s windows from -5 to 15 s around task onset.
3. EEG time-frequency tensors built at 0.5 Hz spectral resolution.
4. Task-related tensor decomposition fitted with shared nonnegative spatial/frequency factors and trial-specific temporal factors.
5. TRCA filter fitted over the temporal components.
6. fNIRS optical intensity converted to approximate HbO from paired 760/850 nm channels by MBLL.
7. fNIRS filtered at 0.01-0.2 Hz.
8. Active fNIRS channels selected by GLM with canonical HRF task regressors.
9. Top three active channels averaged.
10. Canonical and optimized double-gamma HRF models evaluated by leave-one-trial validation.

Important dataset limits:

- The local dataset is EEG+NIRS BCI mental arithmetic, not Lin's finger-tapping HbO dataset.
- The files expose raw optical intensity, so HbO conversion is approximate and coefficient-scale dependent.
- No short-distance fNIRS channels are available for physiological noise regression.
- No BSS-CCA muscle-artifact removal was applied.

## Data

Subject/session:

- Subject: `19`
- Session: `2`, zero-based index `1`
- Task: mental arithmetic, NIRS marker `1`
- Trials used: 10 MA trials from the same continuous session
- EEG source: `data/EEG+NIRS Single-Trial/EEG_01-29/subject 19/with occular artifact/cnt.mat`
- fNIRS source: `data/EEG+NIRS Single-Trial/NIRS_01-29/subject 19/cnt.mat`

Selected active approximate-HbO channels:

- `C1FC1`
- `C1C3`
- `C2C4`

## Results

| Model | Validation | HRF | R2 | PCC | MSE | amplitude ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| TRTD | in-sample upper bound | canonical | 0.004 | 0.064 | 0.996 | 0.064 |
| TRTD | in-sample upper bound | optimized | 0.022 | 0.148 | 0.978 | 0.148 |
| TRTD | leave-one-trial | canonical | -1.870 | -0.254 | 1.450 | 0.121 |
| TRTD | leave-one-trial | optimized | -1.889 | -0.269 | 1.457 | 0.150 |
| trial mean | leave-one-trial | none | -1.631 | 0.234 | 1.456 | 0.379 |
| fNIRS self-persistence | leave-one-trial | none | 0.997 | 0.999 | 0.001 | 1.000 |

The in-sample optimized result is the local model-family upper-bound check. Its R2 of only `0.022` means the failure is not mainly caused by leave-one-trial generalization. The fitted EEG-derived driver does not span the dominant fNIRS trajectory even when the same trials are used for fitting and evaluation.

The optimized leave-one-trial HRF parameters stayed close to canonical values on average: `TTP=6.04 s`, `TTU=16.01 s`, `c=5.99`. This suggests the optimizer does not find a subject-specific HRF deformation that aligns EEG-derived TRTD trajectories with the fNIRS state.

## Visual Interpretation

The trajectory overlay shows that the prediction is a low-amplitude smooth HRF template, while the true active-channel HbO contains a much larger late-window change.

The trial heatmap is more decisive: true HbO contains strong positive and negative trial-specific states, but predicted HbO is compressed near zero and leaves the true pattern in the residual. This is an amplitude, direction, and state-structure mismatch, not just a small timing lag.

Artifacts:

- `figures/loo_trajectory_example.svg`
- `figures/loo_trial_heatmap.svg`
- `figures/performance_summary.svg`
- `figures/trtd_shared_factors.svg`
- `figure_data/loo_trial_trajectories.csv`

## What This Says About fNIRS Prediction

The essential problem is not that fNIRS is unrecoverable. The self-persistence baseline reaches `R2=0.997`, showing that the selected fNIRS trajectory is highly predictable from its own immediate history.

The problem is that the tested EEG-derived state is not the state that controls the observed fNIRS variation in this session. The fNIRS signal is dominated by slow within-channel trajectory, trial-specific baseline/drift, and likely measurement/vascular components that are not captured by the one-dimensional TRTD temporal component convolved with a double-gamma HRF.

Therefore, a Croce-like shared trajectory cannot be safely obtained by:

\[
\hat r(t)=h_{\theta} * T_{EEG}(t)
\]

under the current model family. That expression is available as an EEG-derived candidate driver, but it is not validated as a shared neural state.

## Implication For Next Model

If the goal is a family of trajectories analogous to Croce 2017's jointly inferred `r(t)`, the next test should estimate `r(t)` as a latent state from both modalities jointly, not as a one-way EEG component:

\[
E_t = A_E r_t + \epsilon^E_t,\qquad
F_t = A_F (h_\theta * r)_t + b_{s,c}(t) + \epsilon^F_t.
\]

The model needs explicit subject/session/channel baseline or drift terms `b_{s,c}(t)` and a private fNIRS state. A Kalman/EM or variational smoother over the continuous session is a better next teacher-family control than further tuning one-dimensional EEG-HRF fitting.

## Artifacts

- Run: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_160736_lin2024_raw_session_trtd_s19_sess2/`
- Script: `experiments/evaluate_lin2024_raw_session_trtd.py`
- Summary: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_160736_lin2024_raw_session_trtd_s19_sess2/summary.md`
- Metrics: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_160736_lin2024_raw_session_trtd_s19_sess2/metrics.csv`
- Fold metrics: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_160736_lin2024_raw_session_trtd_s19_sess2/fold_metrics.csv`

**Status:** diagnostic complete; does not change E0; supports moving to a joint latent-state teacher-family control with fNIRS-private state.
