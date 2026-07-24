# E0-D6 unified-loader Croce/Lin shared-driver retest

_Exploratory validation diagnostic, 2026-07-15_

## Decision

> **Status correction (2026-07-24):** This is a pre-sign-calibration
> diagnostic. Its raw-coordinate fNIRS label is superseded by the calibrated
> adaptive SSM physical-teacher acceptance and is not a current E0 result.

The admitted Single-Trial EEG artifact correction changed model behavior but
did not, by itself, resolve the historical raw-coordinate fNIRS mismatch.

- **Supported:** the 2026-07-14 unified measured-data entrance and admitted
  Single-Trial v3 branch are usable for a paired raw-versus-clean model audit.
- **Supported:** EEG artifact correction improves the Croce joint filter's
  paired leave-one-trial score on these five validation subjects.
- **Not supported:** the improvement restores fNIRS amplitude, variance,
  baseline, or trajectory direction sufficiently to admit a shared neural
  driver.
- **Not supported:** either the Croce-2017-inspired SMC driver or the
  Lin-2024-inspired TRTD/HRF driver is ready to supervise shared token identity.

This diagnostic does not reopen the historical protected E0-v2 protocol. It is
an exploratory model-family comparison under the 2026-07-14 measurement-first
contract. Its result supports keeping the Croce/Lin target families blocked.

## Why this rerun was needed

The earlier Croce and Lin diagnostics preceded the final data-quality work.
The relevant implementation lineage is:

- `3308257`: four-dataset loading and quality auditing were unified;
- `220c04d`: the architecture entrance became measurement-first;
- `67fb36c`: an auditable Single-Trial EEG cleaning candidate was added;
- `97179ed`: the controlled-artifact v3 branch passed its sham/null admission
  and became the Single-Trial loader default.

The paired rerun is stronger than comparing the old and new reports. The raw
and v3 conditions use the same subjects, event indices, event windows, clean
fNIRS cache, active-channel selection rule, folds, and random seeds. Only the
Single-Trial EEG branch changes.

The boundary remains important: Single-Trial v3 has a versioned artifact-clean
branch, while the other datasets retain their dataset-specific artifact and
provenance limitations. Unified format, timing, numerical coordinates, labels,
and geometry do not imply that all four datasets are physically or
artifact-wise identical.

## Protocol

The run reads observations only through `UnifiedPhysiologyWindowDataset`; it
does not use the derived Croce source/observation cache.

| Condition | Subjects | Record/task | EEG branch | Trials per subject |
| --- | --- | --- | --- | ---: |
| Single-Trial clean | 19-23 | `session_01`, `MA` | `single_trial_eeg_artifact_clean_v3` | 10 |
| Single-Trial raw | same 19-23 | same events | `raw_with_ocular_artifact` | 10 |
| Simultaneous replication | VP019-VP021 | `cnt_wg`, `WG` | dataset-native unified path | 10 |

Each event window is `-5` to `+15` seconds. The first five seconds define the
within-event fNIRS baseline. Three HbO channels are selected inside each
training fold with a canonical-HRF GLM score; the held-out trial never selects
its own target channels. The main evaluation is subject-specific
leave-one-trial. An in-sample fit-quality upper bound is reported separately.

The compared paths are:

1. `croce_joint`: scalar Croce-inspired SMC filtering with both held-out EEG and
   held-out fNIRS observations;
2. `croce_eeg_only`: the same fitted observation model driven by the EEG-only
   posterior, which is the stricter cross-modal inference control;
3. `lin_trtd`: shared EEG time-frequency CP factors, TRCA temporal extraction,
   and a subject-specific optimized double-gamma HRF;
4. trial-mean and one-step fNIRS self-persistence references.

These are repository implementations of the testable model ideas, not exact
reproductions of either publication. Subject bootstrap intervals are
exploratory because only five paired Single-Trial and three Simultaneous
subjects are used.

## Main leave-one-trial result

### Single-Trial v3-clean EEG

| Path | R2 [95% subject bootstrap] | PCC | SD ratio | Variance ratio | Baseline bias | Direction agreement | Affine-oracle R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Croce joint | -0.0727 [-0.2004, 0.0662] | 0.2228 | 0.2609 | 0.0694 | 0.3767 | 0.5100 | 0.0880 |
| Croce EEG-only | -0.1838 [-0.2847, -0.0954] | -0.1886 | 0.1493 | 0.0224 | 0.3761 | 0.5197 | 0.0848 |
| Lin TRTD, optimized HRF | -0.2946 [-0.3813, -0.2227] | -0.1964 | 0.2668 | 0.0804 | 0.3174 | 0.4777 | 0.0744 |
| fNIRS self-persistence | 0.9967 [0.9932, 0.9989] | 0.9984 | 0.9967 | 0.9935 | -0.0025 | 0.9771 | 0.9968 |

The amplitude problem therefore persists in the admitted clean branch. The
recovered standard deviation is only `14.9%` to `26.7%` of the observed value,
and the recovered variance is only `2.2%` to `8.0%`. Croce joint receives the
held-out fNIRS samples during filtering, yet still recovers only `6.9%` of their
variance scale. The EEG-only path is more decisive for shared inference and is
negative in both R2 and correlation.

The mismatch is not only an amplitude scale error:

- baseline bias remains `0.32-0.38` canonical robust-SD units;
- pointwise trend-direction agreement remains near chance (`0.48-0.52`);
- post-stimulus slope signs agree for only `3/5`, `1/5`, and `2/5` subjects for
  Croce joint, Croce EEG-only, and Lin, respectively;
- even a non-deployable held-out affine rescaling explains less than `9%` of
  the waveform variance.

### Paired effect of artifact correction

| Path | Mean clean-minus-raw delta R2 | Mean delta PCC | Mean delta SD ratio | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Croce joint | +0.1039 [0.0393, 0.1919] | +0.2600 [0.0214, 0.5069] | +0.0462 | R2 improves in 5/5 subjects, but scale change is inconsistent |
| Croce EEG-only | +0.0556 [0.0096, 0.1018] | +0.0557 [-0.0794, 0.2209] | -0.0309 | small score change; clean result remains negative |
| Lin TRTD | -0.1003 [-0.2065, 0.0265] | -0.1469 [-0.2994, 0.1078] | -0.0088 | no artifact-cleaning rescue |

This paired result rules out two overly broad interpretations. The old Croce
score was not completely insensitive to EEG artifacts, because v3 produces a
reproducible relative improvement. But the teacher failure was also not caused
only by those artifacts: after correction, Croce joint R2 remains negative,
the stricter EEG-only path remains negative, and amplitude/variance ratios stay
far below one.

## In-sample model-family upper bound

| Condition | Path | In-sample R2 | SD ratio | Variance ratio |
| --- | --- | ---: | ---: | ---: |
| Single-Trial clean | Croce joint | 0.1023 | 0.1771 | 0.0451 |
| Single-Trial clean | Lin optimized | 0.1053 | 0.2890 | 0.1053 |
| Simultaneous | Croce joint | 0.0245 | 0.0987 | 0.0161 |
| Simultaneous | Lin optimized | 0.0571 | 0.2113 | 0.0571 |

Even same-trial fitting cannot recover most of the fNIRS trajectory. This makes
ordinary held-out generalization error an insufficient explanation. The
tested one-dimensional shared-driver plus fixed/optimized HRF model classes do
not span the dominant fNIRS observation variance.

## Simultaneous EEG&NIRS replication

The separate concentration-data replication shows the same pattern:

| Path | Leave-one-trial R2 | PCC | SD ratio | Variance ratio |
| --- | ---: | ---: | ---: | ---: |
| Croce joint | -0.1375 | 0.0990 | 0.2105 | 0.0470 |
| Croce EEG-only | -0.2385 | -0.2065 | 0.1808 | 0.0346 |
| Lin optimized | -0.2642 | -0.1727 | 0.2980 | 0.1009 |
| fNIRS self-persistence | 0.9968 | 0.9984 | 0.9952 | 0.9905 |

This replication does not prove that artifacts are absent in Simultaneous EEG.
It shows that the compression and trajectory failure are not specific to the
Single-Trial raw optical representation or its admitted EEG correction.

## Scientific interpretation

The self-persistence result is not evidence for a shared neural state because
it consumes the target modality's own previous sample. It is nevertheless an
important falsifier of the statement that the selected fNIRS waveform is
generally unpredictable. The waveform is highly predictable from its own
history; what fails is prediction from the tested shared-driver families.

The supported model-level conclusion is therefore:

> Current fNIRS variance is dominated by modality-private slow trajectory,
> baseline, vascular, and history-specific structure that is not represented by
> either tested one-dimensional shared neural driver.

The data-quality work narrows the diagnosis from “data or teacher” toward
“teacher/model family,” but it does not prove that every remaining
dataset-specific nuisance has been removed. A later teacher candidate should
be judged against the same raw/clean paired control and must preserve explicit
fNIRS-private state. A joint filter's access to held-out fNIRS cannot substitute
for evidence that the driver is independently inferable from EEG.

## Artifacts and replay

- Final run: `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260715_shared_neural_driver_unified_formal_v3/`
- Run summary: `summary.md`
- Subject-bootstrap metrics: `summary_metrics.csv`
- Per-subject metrics: `subject_metrics.csv`
- Per-fold metrics: `fold_metrics.csv`
- Paired branch contrasts: `artifact_branch_contrast.csv` and
  `artifact_branch_contrast_summary.csv`
- Complete trajectories: `trajectories.csv`
- Figures: `figures/model_metric_summary.svg` and
  `figures/representative_raw_vs_clean_overlay.svg`
- Replay script: `experiments/evaluate_shared_neural_driver_unified.py`
- Tests: `tests/test_shared_neural_driver_unified.py`

The manifest binds the resolved configuration, code, unified cache manifest,
event manifest, geometry manifest, and v3 artifact-cache manifest by SHA-256.

**Status: diagnostic complete; the amplitude/variance, baseline, and direction
mismatch persists; Croce/Lin shared-driver supervision remains unsupported.**
