# SSM reconstruction reliability experiment plan

_Protocol status: active exploratory plan; current execution status is owned by
[`research_state/registry.json`](../../research_state/registry.json); 2026-08-19_

## Purpose and claim boundary

This experiment audits how far the current adaptive EEG–fNIRS state-space
model reconstructs its admitted observation trajectories and how much posterior
uncertainty accompanies those reconstructions. It does not requalify the failed
R1-P teacher, establish a unique physiological source, or authorize VQ.

The model input called EEG below is the train-fold log-power PCA proxy used by
the adaptive SSM, not the original high-rate EEG waveform. fNIRS is reported as
separate HbO and HbR trajectories in the measurement coordinate emitted by the
unified loader and the fold-fitted gauge. Dataset-native physical units must not
be pooled when the upstream coordinates differ.

This work is exploratory. Single-Trial subjects 24–29 remain closed. No result
from subjects 01–23 can be described as a new independent confirmation.

## Experiment sequence

The implementation and measured-data work proceed in four bounded stages:

1. **S0 — software and synthetic contract:** expose observation-level
   posterior predictive standard deviation, implement the metric contract, and
   verify it on deterministic synthetic trajectories.
2. **S1 — current-condition audit:** rerun the existing Single-Trial mental
   arithmetic and Simultaneous word-generation surfaces with the new metrics.
3. **S2 — comparable-task extension:** add Single-Trial motor imagery and
   Simultaneous n-back under the same 20 s window and cross-fit contract.
4. **S3 — non-comparable descriptive annex:** analyze Visual, REFED, and DSR
   only under task-specific estimands. These results are not pooled with S1/S2.

S0 and targeted tests are ordinary implementation work. S1–S3 are measured,
potentially costly runs and are launched separately after their software smoke
tests pass.

## Data matrix and splits

### Core 20 s event-window matrix

| Task cell | Admitted conditions | Fit subjects | Development validation | Unused/protected |
| --- | --- | --- | --- | --- |
| Single-Trial mental arithmetic | MA from `session_01` | subjects 01–18 | subjects 19–23 | subjects 24–29 closed |
| Single-Trial motor imagery | LMI/RMI from `session_00` | subjects 01–18 | subjects 19–23 | subjects 24–29 closed |
| Simultaneous word generation | WG from `cnt_wg` | VP001–VP018 | VP019–VP023 | VP024–VP026 unused |
| Simultaneous n-back | 0/2/3-back from `cnt_nback` | VP001–VP018 | VP019–VP023 | VP024–VP026 unused |

Each core window spans event-relative `[-5, 15] s`. The first five seconds are
the declared event baseline. Cross-fitting is within subject, condition
stratified, and grouped by the native trial/dependency identity. Channel
selection, EEG PCA, SSM parameters, noise balance, and all scales are fitted
only on the training folds. Missing subjects, records, conditions, or support
cause the cell to fail closed; they are not replaced silently.

### Descriptive annex

- Visual uses admitted RR/RF/FR/FF 12 s trials and rejects `unknown`; probe-pair
  dependency groups remain intact.
- REFED uses non-overlapping 20 s video segments grouped by video. Continuous
  valence/arousal labels are not converted into event classes for this audit.
- DSR uses non-overlapping 20 s block segments. Go/No-go is EEG-native while
  fNIRS has a block anchor, so no event-level fNIRS reconstruction claim is
  made.

These annexes have separate window, baseline, and interpretation fields in the
manifest. A combined cross-task scalar is forbidden.

## Reliability metrics

All metrics consume the exact valid mask belonging to the observed tensor.
Missing, unsupported, padded, and zero values remain distinct.

For valid observed values \(y\) and reconstructions \(\hat y\), the primary
trajectory-deviation index is

\[
U_{\mathrm{dev}}
=\frac{\sqrt{\operatorname{mean}(y-\hat y)^2}}
        {\operatorname{SD}(y)}.
\]

The denominator is computed from the same valid points with population SD
(`ddof=0`). If the observed SD is non-finite or at most `1e-8`,
`U_dev` is undefined, the row records `low_observed_variance=true`, and the
value is not replaced by an epsilon-based finite score.

Deviation and uncertainty are deliberately separate. The SSM additionally
reports observation-level posterior predictive uncertainty

\[
\sigma_{y,t}
=\sqrt{\operatorname{diag}(H P^{\mathrm{smooth}}_t H^\top+R)},
\]

mapped through the same EEG/HbO/HbR measurement gauge as the reconstruction.
The following quantities are retained independently:

- observed and reconstructed temporal SD and their ratio;
- mean/median posterior state SD and observation predictive SD;
- standardized residual RMS,
  `sqrt(mean(((y - y_hat) / predictive_sd) ** 2))`;
- empirical 95% predictive coverage using `1.96 * predictive_sd`;
- MSE, bias, PCC, R², and valid-point count.

For joint smoothing, standardized residuals are posterior fit diagnostics
because the held-out observation participates in smoothing. For the EEG-only
path, HbO/HbR diagnostics are out-of-modality reconstructions. The two roles
must be labeled separately in tables and prose.

## Aggregation and uncertainty

The aggregation path is exactly window/fold → subject → task cell. A subject's
windows are combined before any group estimate, and subjects receive equal
weight. Group intervals use 10,000 deterministic subject-bootstrap draws.

Trajectory spread has three separately named meanings:

1. temporal SD within one observed or reconstructed trajectory;
2. posterior state/observation SD emitted by the SSM;
3. between-subject spread of a time-resolved group curve.

The report must not call one of these simply “standard deviation” without the
qualifier. Subject points accompany group intervals when the figure remains
legible. Missing time points are gaps, not interpolated lines.

## Implementation and artifact contract

The SSM result interface gains measurement-coordinate predictive SD without
removing the existing `state_std`. A shared NumPy metric helper owns the mask,
low-variance, NRMSE, standardized-residual, and coverage definitions. The
experiment entrypoint writes only below the existing physiology-semantic run
root and refuses overwrite.

Every completed cell produces:

- fold/window, subject, task-summary, and time-course CSV tables;
- trajectories with observed, reconstructed, state SD, and predictive SD;
- bootstrap draws in NPZ form;
- resolved config, split inventory, source/data hashes, software identity, and
  `protected_open=false` in the manifest;
- SVG and PNG figures, source tables, a figure provenance record, and alt text;
- a summary that binds every numeric statement to the generated artifacts.

Default figures use aligned panels rather than dual axes, show the estimator
and named uncertainty, use color plus marker/line style, and preserve missing
values. No significance symbol is emitted unless it corresponds to a recorded
analysis.

## Tests and completion criteria

S0 is complete only when tests cover:

- smoothed covariance propagation through `H` and `R`, including the HbO/HbR
  gauge;
- output shapes, finite non-negative predictive SD, and unchanged existing
  reconstruction/state outputs;
- exact-mask NRMSE, scale invariance, low-variance missingness, standardized
  residual RMS, and coverage;
- subject-equal aggregation and deterministic bootstrap behavior;
- manifest serialization, atomic publication, and overwrite refusal;
- a synthetic calibrated case and a deliberately miscalibrated negative case.

S1/S2 completion means every registered cell either publishes a complete
artifact set or an explicit failure record. The experiment has no pass/fail
threshold for physiological validity: its output is a reliability profile that
can support, weaken, or leave unresolved the suitability of the SSM target.
