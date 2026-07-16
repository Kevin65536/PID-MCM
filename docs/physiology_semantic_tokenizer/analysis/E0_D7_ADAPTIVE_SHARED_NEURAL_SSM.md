# E0-D7 adaptive physiology-constrained shared neural state

_Exploratory model-revision diagnostic, 2026-07-16_

## Outcome

The model revision resolves the specific formal-v3 failure in which reconstructed
HbO retained only a very slow, nearly monotonic component.  It also produces the
requested intermediate state: fNIRS changes the joint driver by about one third
of an EEG-only driver standard deviation, while the joint trajectory remains
strongly correlated with and reconstructs the EEG observation.

This supports continuing with a physiology-constrained shared-state teacher as
a **soft multimodal candidate**. At the time of this diagnostic it did not by
itself admit a new target into E0. The subsequent gauge-corrected validation and
estimand review admit it for development as a physiology-shaped multimodal
consensus proxy. The strict EEG-only path, parameter non-identifiability and
offline smoothing remain claim boundaries rather than joint-teacher vetoes.

Final run:
[`20260716_adaptive_shared_neural_ssm_formal_v2`](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_shared_neural_ssm_formal_v2/)

## What was wrong in formal-v3

The implementation audit changed the interpretation of the 2026-07-15 result.

1. The v3 experiment did not retain the local Croce protocol.  It selected the
   three highest-scoring HbO channels anywhere on the head and averaged them,
   while the Croce path reduced all EEG channels to a whole-head PC1.  This was
   not an average of six local EEG channels; it was a stronger whole-head linear
   mixture.  EEG names and coordinates were not retained in `Trial`, so the
   local protocol could not be executed by that script.
2. The tested `croce_joint` was not the Croce five-state posterior.  It was a
   scalar AR(1) driver followed by a fixed 24-second double-gamma HRF.  The
   plotted vasodilation and flow trajectories were post-hoc companion states,
   not fitted posterior states.
3. The driver was low-pass filtered at 0.2 Hz before another strong HRF
   low-pass.  Its AR coefficient was clipped to `[0.90, 0.999]`; the two visualized
   folds reached approximately `0.995` and `0.999`.
4. Each 20-second event was initialized with zero state and zero HRF history.
   A causal filter cannot use delayed fNIRS evidence to revise a driver value
   already emitted several seconds earlier.  The boundary condition and fixed
   HRF therefore create the observed within-window climb.
5. The bootstrap particle filter reset importance weights whenever resampling
   did not occur, rather than carrying the preceding posterior.  Its stated
   stationary initial covariance was `Q`, not the discrete Lyapunov solution.
6. The Lin CP implementation used Khatri-Rao products in an order inconsistent
   with NumPy C-order tensor flattening.  The dimensions matched, so this error
   did not raise an exception.

The particle-weight, stationary-covariance, RNG, saved-driver, and CP-axis bugs
were fixed and regression-tested before rerunning the old model families.

## Corrected legacy reference

Correcting the algorithms did not solve the fixed-HRF amplitude failure.

| Dataset | Model | HbO R2 | HbO PCC | Variance ratio |
| --- | --- | ---: | ---: | ---: |
| Single-Trial clean v3 | corrected Croce joint | -0.0770 | 0.2012 | 0.0137 |
| Single-Trial clean v3 | corrected Croce EEG-only | -0.1884 | -0.1843 | 0.0182 |
| Single-Trial clean v3 | corrected Lin optimized | -0.1861 | -0.1189 | 0.1013 |
| Simultaneous | corrected Croce joint | -0.0933 | 0.0961 | 0.0162 |
| Simultaneous | corrected Croce EEG-only | -0.1978 | -0.2476 | 0.0171 |
| Simultaneous | corrected Lin optimized | -0.3288 | -0.2859 | 0.0783 |

The corrected reference is in
[`20260716_shared_neural_driver_unified_corrected_v4`](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_shared_neural_driver_unified_corrected_v4/).

## Revised state-space model

### Spatial protocol

The local path now uses one fNIRS spatial anchor selected inside each training
fold.  Its paired HbO/HbR values are the two fNIRS observations.  The six scalp
EEG channels closest to that anchor are selected from the unified geometry.
The all-scalp-EEG path is retained as a same-fold ablation; EOG/ECG/EMG names
are explicitly excluded.

EEG is transformed to 10 Hz log-power and a training-fitted PC1.  It is not
pre-low-passed to 0.2 Hz.  The PC loading and all normalization quantities are
fit on the nine training trials and applied unchanged to the held-out trial.

### Dynamics

The revised model linearizes the Croce/Balloon states around rest:

\[
x(t) = [s(t),\,\Delta f(t),\,\Delta HbO(t),\,\Delta HbR(t),\,r(t)]^\top .
\]

The continuous hemodynamic part is

\[
\dot s = \epsilon r-k_{as}s-k_{af}\Delta f,\qquad
\dot{\Delta f}=s,
\]

\[
\dot{\Delta HbO}
=\frac{\Delta f-\Delta HbO/\alpha}{\tau_0},
\]

\[
\dot{\Delta HbR}
=\frac{c_E\Delta f-(1/\alpha-1)\Delta HbO-\Delta HbR}{\tau_0},
\]

where `c_E` is the resting-flow derivative of the oxygen-extraction term.  The
neural driver follows a training-estimated AR(1) transition whose coefficient
is allowed down to `0.45`, rather than being forced above `0.90`.

The system is exactly discretized at 10 Hz.  A Kalman filter followed by a
Rauch-Tung-Striebel fixed-interval backward pass estimates the complete window.
Unlike a causal filter, delayed HbO/HbR evidence can revise earlier `r(t)`.
The initial covariance is the stationary discrete covariance, so the window is
not forced to begin with zero hemodynamic history.

### Fitted quantities and constraints

Inside every training fold, bounded optimization fits `kas`, `kaf`, `tau0`,
`alpha`, and `E0`; HbO/HbR measurement gains, driver persistence/process noise,
and an explicit EEG/fNIRS observation-noise balance are also estimated.  The
balance objective uses all three standardized observations symmetrically and
penalizes unequal modality reconstruction errors.

The event inputs are already baseline corrected.  Zero remains the rest
coordinate, and predicted chromophore trajectories are re-anchored over the
same five-second baseline interval.  Because the EEG driver has arbitrary
units, `epsilon` and the chromophore gains contain a scale gauge.  That gauge is
fixed from training trials so the 99.5th percentile forward `delta_f` stays
within 25% of resting flow.  This does not create observation fit; it prevents
arbitrary internal units from being misplotted as negative physical flow.

## Formal leave-one-trial results

### Local joint compromise

| Dataset | Subjects | HbO R2 [95% bootstrap] | HbO PCC | SD ratio | Variance ratio | EEG R2 | EEG PCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-Trial clean v3 | 5 | 0.1867 [-0.1081, 0.5117] | 0.8042 | 0.8234 | 0.7439 | 0.7538 | 0.9224 |
| Simultaneous | 3 | 0.1058 [-0.1194, 0.3302] | 0.7669 | 0.6887 | 0.6145 | 0.7259 | 0.9188 |

Relative to corrected Croce joint, the Single-Trial variance ratio increased
from `0.0137` to `0.7439` and the Simultaneous ratio from `0.0162` to `0.6145`.
The change is not merely an affine scale repair: held-out waveform correlations
rose to `0.804/0.767`, and mean R2 became positive in both datasets.  The small
subject counts leave both R2 bootstrap intervals crossing zero, so this remains
exploratory rather than an admission endpoint.

### Is the output an actual compromise?

The joint and EEG-only drivers were compared in the same held-out folds.

| Dataset | Spatial path | Joint-versus-EEG-only driver PCC | fNIRS-induced driver shift / EEG-only SD |
| --- | --- | ---: | ---: |
| Single-Trial clean v3 | local | 0.9268 | 0.3570 |
| Simultaneous | local | 0.9364 | 0.3285 |

The joint driver therefore preserves the dominant EEG trajectory but is not an
EEG copy.  Its fNIRS-induced change is about one third of the EEG-only driver's
standard deviation.  Together with EEG R2 near `0.73-0.75` and HbO PCC near
`0.77-0.80`, this is the intended intermediate state.

The strict EEG-only HbO result remains poor: R2 is `-4.206` in Single-Trial and
`-2.363` in Simultaneous, with PCC only `0.111/0.035`.  The revised state is thus
useful as privileged multimodal supervision, not as evidence that EEG alone
predicts the full held-out fNIRS waveform.

### Periodicity and spectrum

The joint driver monotonic-direction fraction is `0.520/0.518`, close to the
non-monotonic reference of 0.5 rather than the formal-v3 upward trend.  Observed
HbO has `2.40/2.53` prominent turning points per 20-second window, while local
joint reconstruction has `4.36/4.00`.  The fraction of admitted 0.01-0.20 Hz
power below 0.075 Hz is `0.694/0.658` in the observations and `0.658/0.692` in
the reconstructions.  The extremely slow-only failure is resolved, although
the excess turning points warn that the current state is somewhat
over-oscillatory.

### Local versus whole-head EEG

Local joint inference improves HbO R2 over the all-scalp path by `+0.241` in
Single-Trial and `+0.216` in Simultaneous.  PCC improves by `+0.048/+0.077`,
with essentially no EEG-R2 cost (`+0.005/+0.005`).  Variance recovery is almost
unchanged in Single-Trial and increases by `+0.048` in Simultaneous.

This supports restoring spatial locality, but it also answers the original
averaging concern narrowly: whole-head mixing contributed to shape error, yet
it was not the main cause of the monotonic waveform.  Fixed-HRF double
smoothing, causal inference, and window initialization were the larger causes.

### State validity and identifiability

All 320 formal predictions retain positive relative flow; the lowest point over
all folds is `0.627`, and the non-positive-flow fraction is zero.

The parameter audit is less favorable.  Although all 160 optimizations
converged numerically, boundary solutions occurred in `49/160` fits for `kas`,
`88/160` for `kaf`, `90/160` for `tau0`, and `49/160` for `alpha`.  The smallest
fNIRS-noise multiplier was selected in `131/160` fits.  Therefore these fitted
numbers must not be interpreted as recovered subject physiology.  The bounded
dynamics currently act as a useful regularizer and delay model, not a uniquely
identified mechanistic explanation.

## Decision and next use

The decision at the time of this diagnostic was:

> Keep the physiology-constrained path.  Use the local fixed-interval joint
> driver as a candidate privileged/soft tokenizer guide, with explicit
> uncertainty and modality-private residual paths.  Do not call it a recovered
> physical neural source, and do not promote it through E0 from this post-hoc
> exploratory run alone.

This historical caution remains valid for physical-source and causal claims.
It is superseded only for the narrower optional target-family development gate
by
[`E0_V3_ADAPTIVE_TEACHER_ADMISSION_DECISION.md`](E0_V3_ADAPTIVE_TEACHER_ADMISSION_DECISION.md),
which explicitly treats the joint state as a privileged consensus proxy.

Before formal admission, the next version should preregister the modality-noise
range, reduce parameter freedom or use hierarchical shrinkage, add an fNIRS-
private baseline/AR state that is excluded from shared tokens, and evaluate
time-shift/null sensitivity.  A causal or fixed-lag student may later imitate
the offline smoother, but causal performance must be assessed separately.

## Artifacts

- Run summary: [`summary.md`](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_shared_neural_ssm_formal_v2/summary.md)
- Full trajectories: [`trajectories.csv`](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_shared_neural_ssm_formal_v2/trajectories.csv)
- Representative full states: [`adaptive_representative_full_trajectories.svg`](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_shared_neural_ssm_formal_v2/figures/adaptive_representative_full_trajectories.svg)
- Compromise and local/global audit: [`adaptive_posthoc_summary.md`](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_shared_neural_ssm_formal_v2/adaptive_posthoc_summary.md)
- Parameter audit: [`parameter_identifiability_audit.csv`](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_shared_neural_ssm_formal_v2/parameter_identifiability_audit.csv)
- Replay: [`evaluate_adaptive_shared_neural_ssm.py`](../../../experiments/evaluate_adaptive_shared_neural_ssm.py)
- Model: [`adaptive_neurovascular_ssm.py`](../../../src/inference/adaptive_neurovascular_ssm.py)

**Status at the time of E0-D7:** model-revision diagnostic complete;
slow-only/monotonic reconstruction failure resolved for joint inference;
candidate soft teacher retained, formal E0 admission pending. **Current scoped
status:** the later gauge-corrected target passed the optional target-family
development gate as a multimodal consensus proxy; physical-source and causal
claims remain unadmitted.
